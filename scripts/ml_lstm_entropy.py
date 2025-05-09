import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.utils import to_categorical

# --------- CONFIG ---------
sources = {
    'bus': 'processed_data/bus/8bits.bin',
    'radio': 'processed_data/radio/8bits.bin',
    'sismology': 'processed_data/sismology/8bits.bin',
    'ethereum': 'processed_data/ethereum/8bits.bin',
}

RESULTS_DIR = os.path.join('results', 'ml')
os.makedirs(RESULTS_DIR, exist_ok=True)

# Parameters
history_length = 10  # Number of previous symbols for prediction
alphabet_size = 256  # 8 bits per symbol (byte)
epochs = 10
batch_size = 64

# --------- DATA LOADER ---------
def load_bin_file_as_int_sequence(filepath):
    with open(filepath, 'rb') as f:
        byte_arr = np.frombuffer(f.read(), dtype=np.uint8)
    return byte_arr

# --------- MAIN LOOP ---------
for name, input_file in sources.items():
    print(f"\nProcessing source: {name}")
    try:
        seq = load_bin_file_as_int_sequence(input_file)
        N = len(seq)
        print(f"Loaded sequence of length {N} from {input_file}")
        if N <= history_length:
            print(f"Sequence too short for history length {history_length}.")
            continue
        # Create supervised dataset (X: histories, y: next symbol)
        print("Creating supervised dataset (X, y) ...")
        X, y = [], []
        for i in range(N - history_length):
            X.append(seq[i:i+history_length])
            y.append(seq[i+history_length])
        X = np.array(X)
        y = np.array(y)
        print(f"Supervised dataset: X shape {X.shape}, y shape {y.shape}")
        y_cat = to_categorical(y, num_classes=alphabet_size)
        # Split into train/val/test (60/20/20)
        print("Splitting dataset into train/val/test ...")
        X_train, X_temp, y_train, y_temp = train_test_split(X, y_cat, test_size=0.4, shuffle=False)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, shuffle=False)
        print(f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
        # --------- MODEL ---------
        print("Building model ...")
        model = Sequential([
            Embedding(input_dim=alphabet_size, output_dim=32, input_length=history_length),
            LSTM(64),
            Dense(alphabet_size, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        # --------- TRAIN ---------
        print(f"Training model for {epochs} epochs ...")
        model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size, verbose=2)
        # --------- EVALUATE ---------
        print("Evaluating model on test set ...")
        loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
        print(f"Test set accuracy: {accuracy:.6f}")

        # --------- MIN-ENTROPY ESTIMATION (max predicted probability) ---------
        print("Computing min-entropy from predicted probabilities ...")
        y_probs = model.predict(X_test, batch_size=batch_size, verbose=1)
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
        with open(os.path.join(RESULTS_DIR, f"{name}.txt"), 'w', encoding='utf-8') as f:
            f.write(result)
    except Exception as e:
        print(f"Failed to process {name}: {e}")
