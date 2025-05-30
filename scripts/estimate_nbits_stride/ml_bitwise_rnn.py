import os
import numpy as np
import argparse
from tensorflow.keras.utils import Sequence
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, GRU, Dense

sources = {
    'bus': 'processed_data/bus/8bits.bin',
    'radio': 'processed_data/radio/8bits.bin',
    'sismology': 'processed_data/sismology/8bits.bin',
    'ethereum': 'processed_data/ethereum/8bits.bin',
    'generated': 'processed_data/generated/8bits.bin',
}

parser = argparse.ArgumentParser()
parser.add_argument('--stride', type=int, default=2, help='Stride for sampling (default: 2)')
parser.add_argument('--bits', type=int, default=32, help='Bits per symbol (default: 32)')
parser.add_argument('--history', type=int, default=32, help='History length (default: 32)')
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs (default: 10)')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size (default: 128)')
parser.add_argument('--source', type=str, default='all', choices=list(sources.keys()) + ['all'], help='Data source to use (default: all)')
args = parser.parse_args()
stride = args.stride
bits = args.bits
history_length = args.history
epochs = args.epochs
batch_size = args.batch_size
source = args.source

# Helper: load file as sequence of symbols, then expand to bits

def load_bin_file_as_bit_sequence(filepath, stride=1, bits=32):
    with open(filepath, 'rb') as f:
        byte_arr = np.frombuffer(f.read(), dtype=np.uint8)
    total_bits = len(byte_arr) * 8
    n_symbols = total_bits // bits
    print(f"[LOG] File: {filepath}, bytes: {len(byte_arr)}, total_bits: {total_bits}, n_symbols: {n_symbols}")
    if n_symbols == 0:
        return np.array([], dtype=np.uint8)
    all_bits = np.unpackbits(byte_arr)
    all_bits = all_bits[:n_symbols * bits]
    symbols = all_bits.reshape(-1, bits)
    # Apply stride at symbol level
    symbols = symbols[::stride]
    # Flatten to bit sequence
    bit_seq = symbols.reshape(-1)
    print(f"[LOG] Bit sequence length after stride and flatten: {len(bit_seq)}")
    return bit_seq

class BitwiseSequenceBatchGenerator(Sequence):
    def __init__(self, bit_seq, history_length, batch_size, start=0, end=None, **kwargs):
        super().__init__(**kwargs)
        self.bit_seq = bit_seq
        self.history_length = history_length
        self.batch_size = batch_size
        self.start = start
        self.end = end if end is not None else len(bit_seq)
        self.N = self.end - self.start - history_length

    def __len__(self):
        return int(np.ceil(self.N / self.batch_size))

    def __getitem__(self, idx):
        start_idx = self.start + idx * self.batch_size
        end_idx = min(self.start + self.N, start_idx + self.batch_size)
        if idx == 0 or idx % 100 == 0:
            print(f"[LOG] Generating batch {idx}: {start_idx} to {end_idx}")
        X = np.zeros((end_idx - start_idx, self.history_length), dtype=np.uint8)
        y = np.zeros((end_idx - start_idx,), dtype=np.uint8)
        for i, j in enumerate(range(start_idx, end_idx)):
            X[i] = self.bit_seq[j:j+self.history_length]
            y[i] = self.bit_seq[j+self.history_length]
        return X, y

if source == 'all':
    selected_sources = sources.items()
else:
    selected_sources = [(source, sources[source])]

for name, input_file in selected_sources:
    print(f"\nProcessing source: {name}")
    try:
        bit_seq = load_bin_file_as_bit_sequence(input_file, stride=stride, bits=bits)
        N = len(bit_seq)
        print(f"[LOG] Final bit sequence length: {N}")
        if N <= history_length:
            print("Bit sequence too short for history length.")
            continue
        train_gen = BitwiseSequenceBatchGenerator(bit_seq, history_length, batch_size, 0, int(0.7*N))
        val_gen = BitwiseSequenceBatchGenerator(bit_seq, history_length, batch_size, int(0.7*N), int(0.85*N))
        test_gen = BitwiseSequenceBatchGenerator(bit_seq, history_length, batch_size, int(0.85*N), N)
        print(f"[LOG] Train batches: {len(train_gen)}, Val batches: {len(val_gen)}, Test batches: {len(test_gen)}")

        inp = Input(shape=(history_length,))
        x = Embedding(2, 16)(inp)  # Only 2 classes: 0 or 1
        x = GRU(32)(x)
        out = Dense(1, activation='sigmoid')(x)
        model = Model(inputs=inp, outputs=out)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        print("[LOG] Starting model training...")
        model.fit(train_gen, validation_data=val_gen, epochs=epochs, verbose=2)
        print("[LOG] Training complete. Evaluating on test set...")
        loss, accuracy = model.evaluate(test_gen, verbose=0)

        # Predict probabilities for test set
        probs = model.predict(test_gen, verbose=0).flatten()
        # Clamp to avoid log(0)
        probs = np.clip(probs, 1e-8, 1-1e-8)
        # True bits
        y_true = np.concatenate([test_gen[i][1] for i in range(len(test_gen))])
        # Compute per-bit cross-entropy
        bitwise_entropy = -np.mean(y_true * np.log2(probs) + (1-y_true)*np.log2(1-probs))

        result = f"Length (bits): {N}\nModel: Bitwise-RNN\nBitwise Cross-Entropy (bits/bit): {bitwise_entropy:.6f}\nStride: {stride}\nBits per symbol: {bits}\n"
        print(result)
        out_name = f"{name}_stride{stride}_bits{bits}_bitwise.txt"
        out_dir = "results/nbits_stride/ml_bitwise_rnn"
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, out_name), 'w') as f:
            f.write(result)
    except Exception as e:
        print(f"Failed to process {name}: {e}")
