import os
import numpy as np
import argparse
import tensorflow as tf
from tensorflow.keras.utils import Sequence, to_categorical
from tensorflow.keras import mixed_precision
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Dense, LayerNormalization, Dropout
from tensorflow.keras.layers import MultiHeadAttention, Add

# Enable mixed precision for faster training on modern GPUs
mixed_precision.set_global_policy('mixed_float16')

sources = {
    'bus': 'processed_data/bus/8bits.bin',
    'radio': 'processed_data/radio/8bits.bin',
    'sismology': 'processed_data/sismology/8bits.bin',
    'ethereum': 'processed_data/ethereum/8bits.bin',
    'generated': 'processed_data/generated/8bits.bin',
}

RESULTS_DIR = os.path.join('results', '8bits', 'ml_transformer')
os.makedirs(RESULTS_DIR, exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument('--history', type=int, default=32, help='History length (context length, default: 32)')
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs (default: 10)')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size (default: 128)')
args = parser.parse_args()
history_length = args.history
epochs = args.epochs
batch_size = args.batch_size
alphabet_size = 256

def load_bin_file_as_int_sequence(filepath):
    with open(filepath, 'rb') as f:
        byte_arr = np.frombuffer(f.read(), dtype=np.uint8)
    return byte_arr

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

def transformer_encoder_block(embed_dim, num_heads, ff_dim, dropout_rate=0.1):
    inputs = Input(shape=(None, embed_dim))
    attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)(inputs, inputs)
    attn_output = Dropout(dropout_rate)(attn_output)
    out1 = Add()([inputs, attn_output])
    out1 = LayerNormalization(epsilon=1e-6)(out1)
    ffn = Dense(ff_dim, activation='relu')(out1)
    ffn = Dense(embed_dim)(ffn)
    ffn = Dropout(dropout_rate)(ffn)
    out2 = Add()([out1, ffn])
    out2 = LayerNormalization(epsilon=1e-6)(out2)
    return Model(inputs=inputs, outputs=out2, name='transformer_encoder')

for name, input_file in sources.items():
    print(f"\nProcessing source: {name}")
    try:
        seq = load_bin_file_as_int_sequence(input_file)
        N = len(seq)
        print(f"Loaded sequence of length {N} from {input_file}")
        if N <= history_length:
            print(f"Sequence too short for history length {history_length}.")
            continue
        n_train = int(0.6 * (N - history_length))
        n_val = int(0.2 * (N - history_length))
        n_test = (N - history_length) - n_train - n_val
        print(f"Train: {n_train}, Val: {n_val}, Test: {n_test}")
        train_gen = SequenceBatchGenerator(seq, history_length, batch_size=batch_size, num_classes=alphabet_size, start=0, end=n_train+history_length)
        val_gen = SequenceBatchGenerator(seq, history_length, batch_size=batch_size, num_classes=alphabet_size, start=n_train, end=n_train+n_val+history_length)
        test_gen = SequenceBatchGenerator(seq, history_length, batch_size=batch_size, num_classes=alphabet_size, start=n_train+n_val, end=N)
        # --- MODEL ---
        embed_dim = 32
        num_heads = 4
        ff_dim = embed_dim * 2
        dropout_rate = 0.1
        print(f"Building Transformer model: embed_dim={embed_dim}, heads={num_heads}, ff_dim={ff_dim}, context={history_length}")
        inp = Input(shape=(history_length,), dtype='int32')
        x = Embedding(input_dim=alphabet_size, output_dim=embed_dim)(inp)
        x = transformer_encoder_block(embed_dim, num_heads, ff_dim, dropout_rate)(x)
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        out = Dense(alphabet_size, activation='softmax', dtype='float32')(x)
        model = Model(inputs=inp, outputs=out)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        # --- TRAIN ---
        from tensorflow.keras.callbacks import EarlyStopping
        early_stop = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
        print(f"Training model for {epochs} epochs ...")
        model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=epochs,
            verbose=2,
            callbacks=[early_stop]
        )
        # --- EVALUATE ---
        print("Evaluating model on test set ...")
        loss, accuracy = model.evaluate(test_gen, verbose=0)
        print(f"Test set accuracy: {accuracy:.6f}")
        # --- MIN-ENTROPY ESTIMATION ---
        print("Computing min-entropy from predicted probabilities ...")
        y_probs = model.predict(test_gen, verbose=1)
        max_probs = np.max(y_probs, axis=1)
        avg_max_prob = np.mean(max_probs)
        h_ml = -np.log2(avg_max_prob) if avg_max_prob > 0 else float('nan')
        h_ml = min(h_ml, 8.0)  # Clamp to max 8 bits for 8-bit symbols
        result = (
            f"Length: {len(seq)}\n"
            f"Alphabet size: {alphabet_size}\n"
            f"Model: Transformer\n"
            f"Min-Entropy (bits/symbol): {h_ml:.6f}\n"
            f"History length: {history_length}\n"
            f"Test accuracy (P_ML): {accuracy:.6f}\n"
            f"Avg max predicted prob: {avg_max_prob:.6f}\n"
        )
        print(result)
        out_name = f"{name}.txt"
        with open(os.path.join(RESULTS_DIR, out_name), 'w', encoding='utf-8') as f:
            f.write(result)
    except Exception as e:
        print(f"Failed to process {name}: {e}")
