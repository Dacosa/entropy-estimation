import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.utils import to_categorical, Sequence
from tensorflow.keras import mixed_precision

# Enable mixed precision for faster training on modern GPUs
mixed_precision.set_global_policy('mixed_float16')

# --------- CONFIG ---------
sources = {
    'bus': 'processed_data/bus/8bits.bin',
    'radio': 'processed_data/radio/8bits.bin',
    'sismology': 'processed_data/sismology/8bits.bin',
    'ethereum': 'processed_data/ethereum/8bits.bin',
    'generated': 'processed_data/generated/8bits.bin',
}

RESULTS_DIR = os.path.join('results', '8bits_stride', 'ml_lstm')
os.makedirs(RESULTS_DIR, exist_ok=True)

# Parameters
history_length = 10  # Number of previous symbols for prediction
alphabet_size = 256  # 8 bits per symbol (byte)
epochs = 10
batch_size = 128  # Increased batch size for faster training
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--stride', type=int, default=int(os.environ.get('ESTIMATOR_STRIDE', 2)), help='Stride for sampling (default: 2)')
args = parser.parse_args()
stride = args.stride

def load_bin_file_as_int_sequence(filepath, stride=1):
    with open(filepath, 'rb') as f:
        byte_arr = np.frombuffer(f.read(), dtype=np.uint8)
    return byte_arr[::stride]

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
        X = np.zeros((end_idx - start_idx, self.history_length), dtype=np.uint8)
        y = np.zeros((end_idx - start_idx,), dtype=np.uint8)
        for i, j in enumerate(range(start_idx, end_idx)):
            X[i] = self.seq[j:j+self.history_length]
            y[i] = self.seq[j+self.history_length]
        y_cat = to_categorical(y, num_classes=self.num_classes)
        return X, y_cat

for name, input_file in sources.items():
    print(f"\nProcessing source: {name}")
    try:
        seq = load_bin_file_as_int_sequence(input_file, stride=stride)
        N = len(seq)
        print(f"Loaded sequence of length {N} from {input_file} (stride={stride})")
        if N <= history_length:
            print(f"Sequence too short for history length {history_length}.")
            continue
        # Split indices for train/val/test (60/20/20)
        n_train = int(0.6 * (N - history_length))
        n_val = int(0.2 * (N - history_length))
        n_test = (N - history_length) - n_train - n_val
        print(f"Train: {n_train}, Val: {n_val}, Test: {n_test}")
        # Generators
        train_gen = SequenceBatchGenerator(seq, history_length, batch_size=batch_size, num_classes=alphabet_size, start=0, end=n_train+history_length)
        val_gen = SequenceBatchGenerator(seq, history_length, batch_size=batch_size, num_classes=alphabet_size, start=n_train, end=n_train+n_val+history_length)
        test_gen = SequenceBatchGenerator(seq, history_length, batch_size=batch_size, num_classes=alphabet_size, start=n_train+n_val, end=N)
        # --------- MODEL ---------
        from tensorflow.keras.callbacks import EarlyStopping
        print("Building model (LSTM) ...")
        model = Sequential([
            Embedding(input_dim=alphabet_size, output_dim=8),
            LSTM(16),
            Dense(alphabet_size, activation='softmax', dtype='float32')
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        # --------- TRAIN ---------
        early_stop = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
        print(f"Training model for {epochs} epochs ...")
        model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=epochs,
            verbose=2,
            callbacks=[early_stop]
        )
        # --------- EVALUATE ---------
        print("Evaluating model on test set ...")
        loss, accuracy = model.evaluate(test_gen, verbose=0)
        print(f"Test set accuracy: {accuracy:.6f}")

        # --------- MIN-ENTROPY ESTIMATION (max predicted probability) ---------
        print("Computing min-entropy from predicted probabilities ...")
        y_probs = model.predict(test_gen, verbose=1)
        max_probs = np.max(y_probs, axis=1)
        avg_max_prob = np.mean(max_probs)
        h_ml = -np.log2(avg_max_prob) if avg_max_prob > 0 else float('nan')
        h_ml = min(h_ml, 8.0)  # Clamp to max 8 bits for 8-bit symbols
        result = (
            f"Source: {name}\n"
            f"Sequence length: {N}\n"
            f"History length: {history_length}\n"
            f"Alphabet size: {alphabet_size}\n"
            f"Test accuracy (P_ML): {accuracy:.6f}\n"
            f"Avg max predicted prob: {avg_max_prob:.6f}\n"
            f"Estimated min-entropy (h_ML): {h_ml:.6f} bits/symbol\n"
        )
        print(result)
        result = (
            f"Length: {len(seq)}\n"
            f"Alphabet size: {alphabet_size}\n"
            f"Model: LSTM\n"
            f"Min-Entropy (bits/symbol): {h_ml:.6f}\n"
            f"Stride: {stride}\n"
        )
        out_name = f"{name}_stride{stride}.txt"
        with open(os.path.join(RESULTS_DIR, out_name), 'w', encoding='utf-8') as f:
            f.write(result)
    except Exception as e:
        print(f"Failed to process {name}: {e}")
