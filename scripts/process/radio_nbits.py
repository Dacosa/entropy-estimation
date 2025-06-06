# Parameterized version of radio_8bits.py supporting n bits per output byte.
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import random
import time
import argparse

# MP3 frame parsing helpers (copied from radio_8bits.py)
MPEG_VERSION_1 = 0
MPEG_VERSION_2 = 1
MPEG_VERSION_2_5 = 2
LAYER_1 = 0
LAYER_2 = 1
LAYER_3 = 2
BITRATES_KBPS = {
    MPEG_VERSION_1: {LAYER_3: [0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, -1]},
    MPEG_VERSION_2: {LAYER_3: [0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160, -1]}
}
SAMPLING_RATES_HZ = {
    MPEG_VERSION_1: [44100, 48000, 32000, 0],
    MPEG_VERSION_2: [22050, 24000, 16000, 0],
    MPEG_VERSION_2_5: [11025, 12000, 8000, 0]
}
SAMPLES_PER_FRAME = {
    MPEG_VERSION_1: {LAYER_3: 1152},
    MPEG_VERSION_2: {LAYER_3: 576},
    MPEG_VERSION_2_5: {LAYER_3: 576}
}
SAMPLING_BLOCK_SIZE = 99840
BYTES_PER_MINUTE = 960_000

def get_id3v2_tag_size(file_data):
    if len(file_data) < 10:
        return 0
    header_prefix = file_data[:10]
    if header_prefix[0:3] == b'ID3':
        size_bytes = header_prefix[6:10]
        size = (size_bytes[0] & 0x7F) * (1 << 21) + \
               (size_bytes[1] & 0x7F) * (1 << 14) + \
               (size_bytes[2] & 0x7F) * (1 << 7) + \
               (size_bytes[3] & 0x7F)
        return size + 10
    return 0

def find_next_mp3_frame_offset(data, start_offset):
    for i in range(start_offset, len(data) - 1):
        if data[i] == 0xFF and (data[i+1] & 0xE0) == 0xE0:
            return i
    return -1

def parse_mp3_frame_header(header_bytes):
    if len(header_bytes) < 4:
        return None
    if not (header_bytes[0] == 0xFF and (header_bytes[1] & 0xE0) == 0xE0):
        return None
    version_code = (header_bytes[1] >> 3) & 0x03
    if version_code == 0b11: mpeg_version = MPEG_VERSION_1
    elif version_code == 0b10: mpeg_version = MPEG_VERSION_2
    elif version_code == 0b00: mpeg_version = MPEG_VERSION_2_5
    else: return None
    layer_code = (header_bytes[1] >> 1) & 0x03
    if layer_code == 0b01: layer = LAYER_3
    else: return None
    if mpeg_version not in BITRATES_KBPS or layer not in BITRATES_KBPS[mpeg_version]:
        return None
    bitrate_index = (header_bytes[2] >> 4) & 0x0F
    bitrate_kbps = BITRATES_KBPS[mpeg_version][layer][bitrate_index]
    if bitrate_kbps == 0 or bitrate_kbps == -1:
        return None
    samplerate_index = (header_bytes[2] >> 2) & 0x03
    if mpeg_version not in SAMPLING_RATES_HZ or samplerate_index >= len(SAMPLING_RATES_HZ[mpeg_version]):
        return None
    sampling_rate_hz = SAMPLING_RATES_HZ[mpeg_version][samplerate_index]
    if sampling_rate_hz == 0:
        return None
    padding_bit = (header_bytes[2] >> 1) & 0x01
    if mpeg_version not in SAMPLES_PER_FRAME or layer not in SAMPLES_PER_FRAME[mpeg_version]:
        return None
    samples = SAMPLES_PER_FRAME[mpeg_version][layer]
    frame_size = int((samples / 8 * bitrate_kbps * 1000) / sampling_rate_hz) + padding_bit
    if frame_size <= 4:
        return None
    return {'mpeg_version': mpeg_version, 'layer': layer, 'bitrate_kbps': bitrate_kbps, 'sampling_rate_hz': sampling_rate_hz, 'padding': padding_bit, 'frame_size': frame_size, 'header_size': 4}

def get_random_positions(seed, num_positions=8, min_pos=0, max_pos=SAMPLING_BLOCK_SIZE):
    random.seed(seed)
    return sorted(random.sample(range(min_pos, max_pos), num_positions))

def extract_bits_from_positions(data_segment, selected_positions, n_bits):
    bits = []
    for pos in selected_positions:
        if pos < len(data_segment):
            bits.append((data_segment[pos] & 1))
        else:
            bits.append(0)
    bits = bits[:n_bits] + [0] * max(0, n_bits - len(bits))
    return bits

def process_radio_file(file_path, selected_positions, n_bits):
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
        all_bits = []
        current_data_offset = 0
        while current_data_offset < len(data_for_sampling):
            minute_audio_payload = bytearray()
            payload_collected_for_sample = 0
            raw_bytes_scanned_for_this_minute = 0
            start_scan_offset_for_minute = current_data_offset
            while current_data_offset < len(data_for_sampling) and \
                  raw_bytes_scanned_for_this_minute < BYTES_PER_MINUTE:
                frame_start_offset = find_next_mp3_frame_offset(data_for_sampling, current_data_offset)
                if frame_start_offset == -1:
                    current_data_offset = len(data_for_sampling)
                    break
                skipped_bytes = frame_start_offset - current_data_offset
                raw_bytes_scanned_for_this_minute += skipped_bytes
                current_data_offset = frame_start_offset
                if current_data_offset + 4 > len(data_for_sampling):
                    current_data_offset = len(data_for_sampling)
                    break
                header_bytes = data_for_sampling[current_data_offset : current_data_offset + 4]
                frame_info = parse_mp3_frame_header(header_bytes)
                if not frame_info or frame_info['frame_size'] <= frame_info['header_size']:
                    current_data_offset += 1
                    raw_bytes_scanned_for_this_minute += 1
                    continue
                frame_size = frame_info['frame_size']
                header_len = frame_info['header_size']
                if current_data_offset + frame_size > len(data_for_sampling):
                    current_data_offset = len(data_for_sampling)
                    break
                if payload_collected_for_sample < SAMPLING_BLOCK_SIZE:
                    payload = data_for_sampling[current_data_offset + header_len : current_data_offset + frame_size]
                    needed_payload = SAMPLING_BLOCK_SIZE - payload_collected_for_sample
                    minute_audio_payload.extend(payload[:needed_payload])
                    payload_collected_for_sample += len(payload[:needed_payload])
                current_data_offset += frame_size
                raw_bytes_scanned_for_this_minute += frame_size
                if payload_collected_for_sample >= SAMPLING_BLOCK_SIZE:
                    break
            if minute_audio_payload:
                extracted_bits = extract_bits_from_positions(bytes(minute_audio_payload), selected_positions, n_bits)
                all_bits.extend(extracted_bits)
            if current_data_offset <= start_scan_offset_for_minute and current_data_offset < len(data_for_sampling):
                 current_data_offset +=1
            if current_data_offset >= len(data_for_sampling):
                break
        return all_bits, len(all_bits)
    except Exception as e:
        return None, f"Error processing {file_path} (in main try block): {e} ({type(e).__name__})"

def process_radio_files(input_dir, selected_positions, output_path, n_bits):
    input_dir = Path(input_dir)
    radio_files = sorted([p for p in input_dir.rglob('*') if p.is_file()])
    print(f"Found {len(radio_files)} files in {input_dir}")
    total_files = len(radio_files)
    all_bits = []
    start_time = time.time()
    last_printed_percent = -1
    with ThreadPoolExecutor() as executor:
        future_to_file = {
            executor.submit(process_radio_file, path, selected_positions, n_bits): path
            for path in radio_files
        }
        completed = 0
        for future in as_completed(future_to_file):
            result, _ = future.result()
            completed += 1
            percent = int((completed / total_files) * 100) if total_files > 0 else 0
            if percent > last_printed_percent or completed == total_files:
                elapsed = time.time() - start_time
                if completed > 0:
                    avg_time_per_file = elapsed / completed
                    est_total_time = avg_time_per_file * total_files
                    est_remaining_time = est_total_time - elapsed
                    print(f"Progress: {percent}% ({completed}/{total_files}) | Elapsed: {elapsed:.1f}s | Est. remaining: {est_remaining_time:.1f}s")
                else:
                    print(f"Progress: {percent}% ({completed}/{total_files}) | Elapsed: {elapsed:.1f}s")
                last_printed_percent = percent
            if result is not None:
                all_bits.extend(result)
    print("Done processing files.")
    output_file_path = Path(output_path)
    try:
        out_bytes = bytearray()
        for i in range(0, len(all_bits), 8):
            chunk = all_bits[i:i+8]
            byte = 0
            for j, bit in enumerate(chunk):
                byte |= (bit << j)
            out_bytes.append(byte)
        with open(output_file_path, 'wb') as f:
            f.write(out_bytes)
        print(f"Output written to {output_file_path}. Total bits: {len(all_bits)}, total bytes: {len(out_bytes)}")
    except IOError as e:
        print(f"Error writing output file {output_file_path}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract n bits from all radio MP3 files.")
    parser.add_argument("--n-bits", "-n", type=int, default=8, help="Number of bits to extract per sample (default 8).")
    parser.add_argument("--seed", type=int, default=0xAD5423, help="Random seed for position selection.")
    args = parser.parse_args()
    n_bits = args.n_bits
    input_dir = os.path.join("raw_data", "radio")
    output_dir = os.path.join("processed_data", "radio")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"nbits_{n_bits}.bin")
    print(f"Input directory: {input_dir}")
    print(f"Output file: {output_file}")
    selected_positions = get_random_positions(args.seed, num_positions=n_bits, max_pos=SAMPLING_BLOCK_SIZE)
    print(f"Random positions used: {selected_positions}")

    # Gather all files recursively as in radio_8bits.py
    radio_files = sorted([p for p in Path(input_dir).rglob('*') if p.is_file()])
    print(f"Found {len(radio_files)} files in {input_dir}")

    all_bits = []
    start_time = time.time()
    last_printed_percent = -1

    with ThreadPoolExecutor() as executor:
        future_to_file = {
            executor.submit(process_radio_file, str(radio_file_path), selected_positions, n_bits): radio_file_path
            for radio_file_path in radio_files
        }
        completed_count = 0
        for future in as_completed(future_to_file):
            result_bytes, result_info = future.result()
            completed_count += 1
            percent = int((completed_count / len(radio_files)) * 100) if len(radio_files) > 0 else 0
            if percent > last_printed_percent or completed_count == len(radio_files):
                elapsed = time.time() - start_time
                if completed_count > 0:
                    avg_time_per_file = elapsed / completed_count
                    est_total_time = avg_time_per_file * len(radio_files)
                    est_remaining_time = est_total_time - elapsed
                    print(f"Progress: {percent}% ({completed_count}/{len(radio_files)}) | Elapsed: {elapsed:.1f}s | Est. remaining: {est_remaining_time:.1f}s")
                else:
                    print(f"Progress: {percent}% ({completed_count}/{len(radio_files)}) | Elapsed: {elapsed:.1f}s")
                last_printed_percent = percent
            if result_bytes is not None:
                # Unpack bytes to bits
                for b in result_bytes:
                    for i in reversed(range(8)):
                        all_bits.append((b >> i) & 1)
            else:
                print(result_info)

    # Write bits as a true bitstream: pack bits into bytes, in extraction order
    out_bytes = bytearray()
    for i in range(0, len(all_bits), 8):
        chunk = all_bits[i:i+8]
        byte = 0
        for j, bit in enumerate(chunk):
            byte |= (bit << j)
        out_bytes.append(byte)
    with open(output_file, 'wb') as f:
        f.write(out_bytes)
    print(f"Output written to {output_file}. Total bits: {len(all_bits)}, total bytes: {len(out_bytes)}")
