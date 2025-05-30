import os
import numpy as np
import argparse
from tensorflow.keras.utils import Sequence
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense

sources = {
    'bus': 'processed_data/bus/8bits.bin',
    'radio': 'processed_data/radio/8bits.bin',
    'sismology': 'processed_data/sismology/8bits.bin',
    'ethereum': 'processed_data/ethereum/8bits.bin',
    'generated': 'processed_data/generated/8bits.bin',
}

parser = argparse.ArgumentParser()
parser.add_argument('--stride', type=int, default=2, help='Stride for sampling (default: 2)')
parser.add_argument('--bits', type=int, default=8, help='Bits per symbol (default: 8)')
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
alphabet_size = 2 ** bits
source = args.source

def load_bin_file_as_int_sequence(filepath, stride=1, bits=8):
    with open(filepath, 'rb') as f:
        byte_arr = np.frombuffer(f.read(), dtype=np.uint8)
    total_bits = len(byte_arr) * 8
    n_symbols = total_bits // bits
    if n_symbols == 0:
        return np.array([], dtype=np.uint32)
    all_bits = np.unpackbits(byte_arr)
    all_bits = all_bits[:n_symbols * bits]
    symbols = all_bits.reshape(-1, bits).dot(1 << np.arange(bits)[::-1])
    symbols = symbols[::stride]
    return symbols.astype(np.uint32)

class SequenceBatchGenerator(Sequence):
    def __init__(self, seq, history_length, batch_size, num_classes, start=0, end=None, **kwargs):
        super().__init__(**kwargs)
        self.seq = seq
        self.history_length = history_length
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.start = start
        self.end = end if end is not None else len(seq)
        self.N = self.end - self.start - history_length

    def __len__(self):
        return int(np.ceil(self.N / self.batch_size))

    def __getitem__(self, idx):
        start_idx = self.start + idx * self.batch_size
        end_idx = min(self.start + self.N, start_idx + self.batch_size)
        X = np.zeros((end_idx - start_idx, self.history_length), dtype=np.uint32)
        y = np.zeros((end_idx - start_idx,), dtype=np.uint32)
        for i, j in enumerate(range(start_idx, end_idx)):
            X[i] = self.seq[j:j+self.history_length]
            y[i] = self.seq[j+self.history_length]
        return X, y

if source == 'all':
    selected_sources = sources.items()
else:
    selected_sources = [(source, sources[source])]

for name, input_file in selected_sources:
    print(f"\nProcessing source: {name}")
    try:
        seq = load_bin_file_as_int_sequence(input_file, stride=stride, bits=bits)
        N = len(seq)
        if N <= history_length:
            print("Sequence too short for history length.")
            continue
        train_gen = SequenceBatchGenerator(seq, history_length, batch_size, alphabet_size, 0, int(0.7*N))
        val_gen = SequenceBatchGenerator(seq, history_length, batch_size, alphabet_size, int(0.7*N), int(0.85*N))
        test_gen = SequenceBatchGenerator(seq, history_length, batch_size, alphabet_size, int(0.85*N), N)

        inp = Input(shape=(history_length,))
        x = Embedding(alphabet_size, 64)(inp)
        x = LSTM(64)(x)
        out = Dense(alphabet_size, activation='softmax')(x)
        model = Model(inputs=inp, outputs=out)
        # Use sparse categorical crossentropy for memory efficiency
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        # For even larger output spaces, consider using tf.nn.sampled_softmax_loss or adaptive softmax.
        model.fit(train_gen, validation_data=val_gen, epochs=epochs, verbose=2)
        loss, accuracy = model.evaluate(test_gen, verbose=0)

        from scipy.stats import entropy
        probs = model.predict(test_gen, verbose=0)
        max_probs = np.max(probs, axis=1)
        min_entropy = -np.log2(np.mean(max_probs))

        result = f"Length: {N}\nAlphabet size: {alphabet_size}\nModel: LSTM\nMin-Entropy (bits/symbol): {min_entropy:.6f}\nStride: {stride}\nBits per symbol: {bits}\n"
        print(result)
        out_name = f"{name}_stride{stride}_bits{bits}.txt"
        out_dir = "results/nbits_stride/ml_lstm"
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, out_name), 'w') as f:
            f.write(result)
    except Exception as e:
        print(f"Failed to process {name}: {e}")
