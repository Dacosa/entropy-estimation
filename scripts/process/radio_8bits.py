import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import random
import time

SEED_HEX = 'ad5423'
SEED_INT = int(SEED_HEX, 16)

SAMPLING_BLOCK_SIZE = 99840  # Size of the block within each minute from which to sample (0 to SAMPLING_BLOCK_SIZE-1)
BYTES_PER_MINUTE = 960_000   # Based on 128kbps audio

SELECTED_POSITIONS = None  # Will be initialized in main

# --- MP3 Frame Parsing Constants and Helpers ---
# MPEG versions (internal representation)
MPEG_VERSION_1 = 0
MPEG_VERSION_2 = 1
MPEG_VERSION_2_5 = 2 # Often grouped with MPEG_VERSION_2 for tables

# Layers (internal representation)
LAYER_1 = 0
LAYER_2 = 1
LAYER_3 = 2

# Bitrate lookup table (kbps). Structure: BITRATES[mpeg_version][layer_index][bitrate_code]
# Using simplified structure for common MP3s (V1L3, V2L3)
# Index: 4 bits from header. Values in kbps.
BITRATES_KBPS = {
    MPEG_VERSION_1: {
        LAYER_3: [0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, -1]
    },
    MPEG_VERSION_2: { # Covers MPEG 2 and 2.5
        LAYER_3: [0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160, -1]
    }
}
# MPEG 2.5 uses MPEG_VERSION_2 table, bitrates are the same for Layer 3.

# Sampling rate lookup table (Hz). Structure: SAMPLING_RATES[mpeg_version][samplerate_code]
SAMPLING_RATES_HZ = {
    MPEG_VERSION_1: [44100, 48000, 32000, 0],    # Index 0, 1, 2
    MPEG_VERSION_2: [22050, 24000, 16000, 0],    # Index 0, 1, 2 (for MPEG 2)
    MPEG_VERSION_2_5: [11025, 12000, 8000, 0]   # Index 0, 1, 2 (for MPEG 2.5)
}

# Samples per frame (constant for a given version and layer)
SAMPLES_PER_FRAME = {
    MPEG_VERSION_1: { LAYER_3: 1152 },
    MPEG_VERSION_2: { LAYER_3: 576 },  # MPEG 2 Layer III
    MPEG_VERSION_2_5: { LAYER_3: 576 } # MPEG 2.5 Layer III
}

def find_next_mp3_frame_offset(data, start_offset):
    """Finds the offset of the next MP3 frame sync word."""
    for i in range(start_offset, len(data) - 1):
        if data[i] == 0xFF and (data[i+1] & 0xE0) == 0xE0:
            # Basic sync word found. More validation could be done in parse_mp3_frame_header.
            return i
    return -1

def parse_mp3_frame_header(header_bytes):
    """Parses 4-byte MP3 frame header. Returns dict with info or None if invalid."""
    if len(header_bytes) < 4:
        return None

    # Sync word check (already partially done by find_next_mp3_frame_offset)
    if not (header_bytes[0] == 0xFF and (header_bytes[1] & 0xE0) == 0xE0):
        return None

    # MPEG Version ID
    version_code = (header_bytes[1] >> 3) & 0x03
    if version_code == 0b11: mpeg_version = MPEG_VERSION_1
    elif version_code == 0b10: mpeg_version = MPEG_VERSION_2
    elif version_code == 0b00: mpeg_version = MPEG_VERSION_2_5 # MPEG 2.5 often 00
    # elif version_code == 0b01: return None # Reserved
    else: return None # Reserved or invalid

    # Layer description
    layer_code = (header_bytes[1] >> 1) & 0x03
    if layer_code == 0b01: layer = LAYER_3       # Layer III
    # elif layer_code == 0b10: layer = LAYER_2    # Layer II
    # elif layer_code == 0b11: layer = LAYER_1    # Layer I
    else: return None # We only support Layer III for simplicity, or reserved layer
    
    if mpeg_version not in BITRATES_KBPS or layer not in BITRATES_KBPS[mpeg_version]:
        return None # Unsupported MPEG version/layer combination for bitrate lookup

    # Bitrate
    bitrate_index = (header_bytes[2] >> 4) & 0x0F
    bitrate_kbps = BITRATES_KBPS[mpeg_version][layer].get(bitrate_index) if isinstance(BITRATES_KBPS[mpeg_version][layer], dict) else BITRATES_KBPS[mpeg_version][layer][bitrate_index]
    if bitrate_kbps is None or bitrate_kbps == 0 or bitrate_kbps == -1: # 0=free, -1=bad
        return None 

    # Sampling Rate
    samplerate_index = (header_bytes[2] >> 2) & 0x03
    if mpeg_version not in SAMPLING_RATES_HZ or samplerate_index >= len(SAMPLING_RATES_HZ[mpeg_version]):
        return None
    sampling_rate_hz = SAMPLING_RATES_HZ[mpeg_version][samplerate_index]
    if sampling_rate_hz == 0:
        return None

    # Padding bit
    padding_bit = (header_bytes[2] >> 1) & 0x01

    # Calculate Frame Size
    # FrameSize = floor(SamplesPerFrame / 8 * BitRate / SampleRate) + Padding
    # BitRate is in bps, so kbps * 1000
    if mpeg_version not in SAMPLES_PER_FRAME or layer not in SAMPLES_PER_FRAME[mpeg_version]:
        return None # Unsupported for samples per frame lookup
        
    samples = SAMPLES_PER_FRAME[mpeg_version][layer]
    frame_size = int( (samples / 8 * bitrate_kbps * 1000) / sampling_rate_hz ) + padding_bit

    if frame_size <= 4: # Frame size must be greater than header size
        return None

    return {
        'mpeg_version': mpeg_version,
        'layer': layer,
        'bitrate_kbps': bitrate_kbps,
        'sampling_rate_hz': sampling_rate_hz,
        'padding': padding_bit,
        'frame_size': frame_size,
        'header_size': 4 # Standard MP3 frame header
    }

# --- End MP3 Frame Parsing Helpers ---

def get_id3v2_tag_size(file_data):
    """Detects and returns the size of an ID3v2 tag, including its 10-byte header."""
    if len(file_data) < 10:
        return 0  # Not enough data for an ID3v2 header
    header_prefix = file_data[:10]
    if header_prefix[0:3] == b'ID3':
        # Basic check for ID3 prefix. More robust parsing would check version/flags.
        # ID3v2 size is a 4-byte synchsafe integer (MSB of each byte is 0).
        size_bytes = header_prefix[6:10]
        size = 0
        try:
            size = (size_bytes[0] & 0x7F) * (1 << 21) + \
                   (size_bytes[1] & 0x7F) * (1 << 14) + \
                   (size_bytes[2] & 0x7F) * (1 << 7) + \
                   (size_bytes[3] & 0x7F)
            return size + 10  # Total size to skip = tag content size + 10 byte header itself
        except IndexError:
             return 0 # Should not happen if len(file_data) >= 10 and size_bytes is header_prefix[6:10]
    return 0  # No ID3v2 tag identified

def get_random_positions(seed, num_positions=8, min_pos=0, max_pos=SAMPLING_BLOCK_SIZE):
    """Generates sorted random 0-indexed positions within [min_pos, max_pos-1]."""
    random.seed(seed)
    # random.sample samples from population range(min_pos, max_pos), which is min_pos to max_pos-1
    return sorted(random.sample(range(min_pos, max_pos), num_positions))

def extract_byte_from_positions(data_segment, selected_positions):
    """Extracts a byte from LSBs of data_segment at selected_positions."""
    bits = []
    for pos in selected_positions:  # selected_positions are 0-indexed
        if pos < len(data_segment):
            bits.append((data_segment[pos] & 1))  # Take the Least Significant Bit
        else:
            bits.append(0)  # Pad with 0 if position is out of bounds for the current segment
    byte = 0
    for i, bit in enumerate(bits):
        byte |= (bit << (7 - i))  # Assemble bits into a byte (MSB first)
    return bytes([byte])

def process_radio_file(file_path, selected_positions):
    """Processes a single radio file: skips ID3v2, parses MP3 frames, extracts bytes per minute from payload."""
    try:
        with open(file_path, 'rb') as f:
            file_content = f.read()
    except FileNotFoundError:
        return None, f"Error processing {file_path}: File not found."
    except Exception as e:
        return None, f"Error reading {file_path}: {e} ({type(e).__name__})"

    try:
        id3_header_size = get_id3v2_tag_size(file_content)
        data_for_sampling = file_content[id3_header_size:]

        byte_list = []
        current_data_offset = 0  # Offset in data_for_sampling

        # Loop to extract one byte per conceptual 'minute' of the original file content
        while current_data_offset < len(data_for_sampling):
            minute_audio_payload = bytearray()
            payload_collected_for_sample = 0
            raw_bytes_scanned_for_this_minute = 0
            start_scan_offset_for_minute = current_data_offset

            # Scan frames to collect payload for one sample, or until BYTES_PER_MINUTE of raw data is scanned
            while current_data_offset < len(data_for_sampling) and \
                  raw_bytes_scanned_for_this_minute < BYTES_PER_MINUTE:
                
                frame_start_offset = find_next_mp3_frame_offset(data_for_sampling, current_data_offset)

                if frame_start_offset == -1: # No more frames found
                    current_data_offset = len(data_for_sampling) # Mark as EOF
                    break
                
                # Account for bytes skipped until frame header
                skipped_bytes = frame_start_offset - current_data_offset
                raw_bytes_scanned_for_this_minute += skipped_bytes
                current_data_offset = frame_start_offset

                if current_data_offset + 4 > len(data_for_sampling): # Not enough data for a full header
                    current_data_offset = len(data_for_sampling)
                    break

                header_bytes = data_for_sampling[current_data_offset : current_data_offset + 4]
                frame_info = parse_mp3_frame_header(header_bytes)

                if not frame_info or frame_info['frame_size'] <= frame_info['header_size']:
                    # Invalid frame or bad frame size, skip this potential header and search again
                    current_data_offset += 1
                    raw_bytes_scanned_for_this_minute += 1
                    continue

                frame_size = frame_info['frame_size']
                header_len = frame_info['header_size']

                if current_data_offset + frame_size > len(data_for_sampling): # Frame data truncated at EOF
                    current_data_offset = len(data_for_sampling)
                    break
                
                # Extract payload if we still need it for the current sample byte
                if payload_collected_for_sample < SAMPLING_BLOCK_SIZE:
                    payload = data_for_sampling[current_data_offset + header_len : current_data_offset + frame_size]
                    needed_payload = SAMPLING_BLOCK_SIZE - payload_collected_for_sample
                    
                    minute_audio_payload.extend(payload[:needed_payload])
                    payload_collected_for_sample += len(payload[:needed_payload])
                
                current_data_offset += frame_size
                raw_bytes_scanned_for_this_minute += frame_size

                # If we've collected enough payload for this sample, break from inner frame-scan loop
                if payload_collected_for_sample >= SAMPLING_BLOCK_SIZE:
                    break
            
            # After scanning for a minute or collecting enough payload for a sample:
            if minute_audio_payload:
                # The `extract_byte_from_positions` handles if len(minute_audio_payload) < SAMPLING_BLOCK_SIZE
                # (by padding with 0-bits for out-of-bounds positions)
                extracted_byte = extract_byte_from_positions(bytes(minute_audio_payload), selected_positions)
                byte_list.append(extracted_byte)
            
            # If no progress was made in scanning raw bytes (e.g., stuck at EOF or bad data patch),
            # ensure we break out of the outer while loop to prevent infinite loops.
            if current_data_offset <= start_scan_offset_for_minute and current_data_offset < len(data_for_sampling):
                 current_data_offset +=1 # Force advance if stuck
            
            if current_data_offset >= len(data_for_sampling): # Reached effective EOF
                break
        
        return b''.join(byte_list), len(byte_list)

    except Exception as e:
        # Catch-all for unexpected errors during parsing or processing
        return None, f"Error processing {file_path} (in main try block): {e} ({type(e).__name__})"

def process_radio_files(input_dir, selected_positions, output_path):
    input_dir = Path(input_dir)
    radio_files = sorted([p for p in input_dir.rglob('*') if p.is_file()])
    print(f"Found {len(radio_files)} files in {input_dir}")
    total_files = len(radio_files)
    all_bytes = []
    start_time = time.time()
    last_printed_percent = -1

    with ThreadPoolExecutor() as executor:
        future_to_file = {
            executor.submit(process_radio_file, radio_file_path, selected_positions): radio_file_path
            for radio_file_path in radio_files
        }
        completed_count = 0
        for future in as_completed(future_to_file):
            # file_path_processed = future_to_file[future]
            result_bytes, result_info = future.result()
            completed_count += 1
            percent = int((completed_count / total_files) * 100) if total_files > 0 else 0

            if percent > last_printed_percent or completed_count == total_files:
                elapsed = time.time() - start_time
                if completed_count > 0:
                    avg_time_per_file = elapsed / completed_count
                    est_total_time = avg_time_per_file * total_files
                    est_remaining_time = est_total_time - elapsed
                    print(f"Progress: {percent}% ({completed_count}/{total_files}) | Elapsed: {elapsed:.1f}s | Est. remaining: {est_remaining_time:.1f}s")
                else:
                    print(f"Progress: {percent}% ({completed_count}/{total_files}) | Elapsed: {elapsed:.1f}s")
                last_printed_percent = percent
            
            if result_bytes is not None:
                all_bytes.append(result_bytes)
            else: # Error occurred
                print(result_info) # Print the error message

    print("Done processing files.")
    output_file_path = Path(output_path)
    try:
        with open(output_file_path, 'wb') as f:
            for b_sequence in all_bytes:
                f.write(b_sequence)
        print(f"Output written to {output_file_path}")
    except IOError as e:
        print(f"Error writing output file {output_file_path}: {e}")

if __name__ == "__main__":
    input_dir_path = 'raw_data/radio'
    output_dir_path = 'processed_data/radio'

    # Initialize SELECTED_POSITIONS using the updated get_random_positions
    # It now defaults to 0-indexed positions within [0, SAMPLING_BLOCK_SIZE-1]
    SELECTED_POSITIONS = get_random_positions(SEED_INT)
    print(f"Random 0-indexed positions used (within first {SAMPLING_BLOCK_SIZE} bytes of each minute's content): {SELECTED_POSITIONS}")

    output_file_name = '8bits.bin'
    final_output_path = os.path.join(output_dir_path, output_file_name)
    
    Path(output_dir_path).mkdir(parents=True, exist_ok=True)
    
    process_radio_files(input_dir_path, SELECTED_POSITIONS, final_output_path)