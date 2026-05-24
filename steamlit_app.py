# ============================================================
# INSTALL REQUIRED LIBRARIES
# ============================================================

!pip install -q numpy pandas matplotlib scipy scikit-learn

# ============================================================
# IMPORTS
# ============================================================

import os
import csv
import time
import random
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import scipy.signal as ssig
from scipy.fftpack import dct

from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# ============================================================
# RANDOM SEED
# ============================================================

random.seed(42)
np.random.seed(42)

# ============================================================
# GLOBAL SETTINGS
# ============================================================

SR = 22050
DURATION = 3
N_SAMPLES = SR * DURATION

N_MFCC = 40
N_FFT = 2048
HOP_LEN = 512
N_MELS = 64

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATASET_CSV = "dataset.csv"
RESULTS_CSV = "results.csv"

# ============================================================
# SYNTHETIC DATA GENERATOR
# ============================================================

class SyntheticDataGenerator:

    TAMPERING_TYPES = [
        "splice",
        "cut",
        "noise_burst",
        "compression"
    ]

    def __init__(self, n_total=200):

        self.n_total = n_total
        self.n_clean = n_total // 2
        self.n_tampered = n_total - self.n_clean

    def _base_signal(self):

        t = np.linspace(0, DURATION, N_SAMPLES, endpoint=False)

        f0 = np.random.uniform(80, 300)

        sig = np.zeros(N_SAMPLES)

        for k in range(1, 7):

            amp = 1.0 / k

            sig += amp * np.sin(
                2 * np.pi * f0 * k * t +
                np.random.uniform(0, np.pi)
            )

        sig += np.random.randn(N_SAMPLES) * 0.02

        sig = sig / (np.max(np.abs(sig)) + 1e-9)

        return sig.astype(np.float32)

    def _apply_splice(self, sig):

        out = sig.copy()

        start = np.random.randint(SR // 2, N_SAMPLES - SR)
        length = np.random.randint(SR // 4, SR)

        replacement = self._base_signal()[start:start + length]

        out[start:start + length] = replacement

        return out

    def _apply_cut(self, sig):

        start = np.random.randint(SR // 4, N_SAMPLES // 2)
        length = np.random.randint(SR // 4, SR)

        out = np.concatenate([
            sig[:start],
            sig[start + length:]
        ])

        out = np.pad(out, (0, N_SAMPLES - len(out)))

        return out.astype(np.float32)

    def _apply_noise_burst(self, sig):

        out = sig.copy()

        start = np.random.randint(0, N_SAMPLES - SR // 2)
        length = np.random.randint(SR // 8, SR // 2)

        out[start:start + length] += np.random.randn(length) * 0.5

        out = out / (np.max(np.abs(out)) + 1e-9)

        return out.astype(np.float32)

    def _apply_compression(self, sig):

        bits = np.random.choice([4, 6, 8])

        steps = 2 ** bits

        out = np.round(sig * steps) / steps

        return out.astype(np.float32)

    def generate(self):

        signals = {}
        rows = []

        # CLEAN

        for i in range(self.n_clean):

            sid = f"clean_{i:04d}"

            signals[sid] = self._base_signal()

            rows.append({
                "id": sid,
                "label": "clean",
                "tampering_type": "none"
            })

        # TAMPERED

        for i in range(self.n_tampered):

            sid = f"tampered_{i:04d}"

            ttype = self.TAMPERING_TYPES[i % 4]

            base = self._base_signal()

            if ttype == "splice":
                tampered = self._apply_splice(base)

            elif ttype == "cut":
                tampered = self._apply_cut(base)

            elif ttype == "noise_burst":
                tampered = self._apply_noise_burst(base)

            else:
                tampered = self._apply_compression(base)

            signals[sid] = tampered

            rows.append({
                "id": sid,
                "label": "tampered",
                "tampering_type": ttype
            })

        random.shuffle(rows)

        split_idx = int(0.8 * len(rows))

        for i, row in enumerate(rows):

            row["split"] = "train" if i < split_idx else "test"

        df = pd.DataFrame(rows)

        df.to_csv(DATASET_CSV, index=False)

        print("Dataset generated successfully")

        return signals, df

# ============================================================
# PREPROCESSING
# ============================================================

class PreprocessingModule:

    def normalize(self, sig):

        return sig / (np.max(np.abs(sig)) + 1e-9)

    def process(self, sig):

        sig = self.normalize(sig)

        if len(sig) < N_SAMPLES:

            sig = np.pad(sig, (0, N_SAMPLES - len(sig)))

        else:

            sig = sig[:N_SAMPLES]

        return sig.astype(np.float32)

# ============================================================
# FEATURE EXTRACTION
# ============================================================

class FeatureExtractionModule:

    def __init__(self):

        self._build_filterbank()

    def _build_filterbank(self):

        def hz2mel(h):
            return 2595 * np.log10(1 + h / 700)

        def mel2hz(m):
            return 700 * (10 ** (m / 2595) - 1)

        low = hz2mel(80)
        high = hz2mel(SR / 2)

        mel_points = np.linspace(low, high, N_MELS + 2)

        hz_points = mel2hz(mel_points)

        bins = np.floor((N_FFT + 1) * hz_points / SR).astype(int)

        self.fb = np.zeros((N_MELS, N_FFT // 2 + 1))

        for m in range(1, N_MELS + 1):

            f1 = bins[m - 1]
            f2 = bins[m]
            f3 = bins[m + 1]

            for k in range(f1, f2):
                self.fb[m - 1, k] = (k - f1) / (f2 - f1 + 1e-9)

            for k in range(f2, f3):
                self.fb[m - 1, k] = (f3 - k) / (f3 - f2 + 1e-9)

    def _stft(self, sig):

        frames = []

        for i in range(0, len(sig) - N_FFT, HOP_LEN):

            frame = sig[i:i + N_FFT]

            frame = frame * np.hanning(N_FFT)

            spec = np.abs(np.fft.rfft(frame))

            frames.append(spec)

        return np.array(frames).T

    def extract(self, sig):

        mag = self._stft(sig)

        mel = self.fb @ mag

        log_mel = np.log(mel + 1e-9)

        mfcc = dct(log_mel, axis=0, norm="ortho")[:N_MFCC]

        feat = np.concatenate([
            mfcc.mean(axis=1),
            mfcc.std(axis=1)
        ])

        return feat.astype(np.float32)

    def extract_batch(self, signals, df):

        X = []
        y = []
        ids = []

        for _, row in df.iterrows():

            vec = self.extract(signals[row["id"]])

            X.append(vec)

            y.append(1 if row["label"] == "tampered" else 0)

            ids.append(row["id"])

        return np.array(X), np.array(y), ids

# ============================================================
# TRAINING
# ============================================================

class TrainingModule:

    MODELS = {

        "ANN": MLPClassifier(
            hidden_layer_sizes=(256, 128),
            max_iter=200,
            early_stopping=True,
            random_state=42
        ),

        "SVM": SVC(
            probability=True,
            random_state=42
        ),

        "RF": RandomForestClassifier(
            n_estimators=100,
            random_state=42
        ),

        "GBM": GradientBoostingClassifier(
            random_state=42
        )
    }

    def __init__(self):

        self.pipelines = {}

    def build(self, n_features):

        pca_components = min(40, n_features)

        for name, clf in self.MODELS.items():

            self.pipelines[name] = Pipeline([

                ("scaler", StandardScaler()),

                ("pca", PCA(
                    n_components=pca_components,
                    random_state=42
                )),

                ("clf", clf)
            ])

    def train(self, X_train, y_train):

        best_acc = 0
        best_model = None
        best_name = None

        for name, pipe in self.pipelines.items():

            print(f"\nTraining {name}")

            scores = cross_val_score(
                pipe,
                X_train,
                y_train,
                cv=5,
                scoring="accuracy"
            )

            print("Accuracy:", scores.mean())

            pipe.fit(X_train, y_train)

            if scores.mean() > best_acc:

                best_acc = scores.mean()
                best_model = pipe
                best_name = name

        print("\nBest Model:", best_name)

        return best_model, best_name

# ============================================================
# EVALUATION
# ============================================================

class EvaluationModule:

    def evaluate(self, model, X_test, y_test):

        pred = model.predict(X_test)

        probs = model.predict_proba(X_test)

        acc = accuracy_score(y_test, pred)

        print("\nAccuracy:", acc)

        print("\nClassification Report:\n")

        print(classification_report(
            y_test,
            pred,
            target_names=["clean", "tampered"]
        ))

        cm = confusion_matrix(y_test, pred)

        plt.figure(figsize=(5, 4))

        plt.imshow(cm, cmap="Blues")

        plt.title("Confusion Matrix")

        plt.colorbar()

        plt.xticks([0,1], ["clean", "tampered"])
        plt.yticks([0,1], ["clean", "tampered"])

        for i in range(2):
            for j in range(2):
                plt.text(j, i, cm[i,j],
                         ha="center",
                         va="center",
                         color="black")

        plt.savefig(f"{OUTPUT_DIR}/confusion_matrix.png")

        plt.close()

        print("Confusion matrix saved")

# ============================================================
# MAIN
# ============================================================

def main():

    print("\nAUDIO TAMPERING DETECTION SYSTEM\n")

    # DATASET

    generator = SyntheticDataGenerator()

    signals, df = generator.generate()

    # PREPROCESSING

    pre = PreprocessingModule()

    processed = {}

    for sid, sig in signals.items():

        processed[sid] = pre.process(sig)

    # FEATURES

    fe = FeatureExtractionModule()

    X, y, ids = fe.extract_batch(processed, df)

    print("Feature Shape:", X.shape)

    # SPLIT

    train_ids = set(df[df["split"] == "train"]["id"])

    train_mask = np.array([i in train_ids for i in ids])

    test_mask = ~train_mask

    X_train = X[train_mask]
    y_train = y[train_mask]

    X_test = X[test_mask]
    y_test = y[test_mask]

    # TRAINING

    trainer = TrainingModule()

    trainer.build(X.shape[1])

    best_model, best_name = trainer.train(
        X_train,
        y_train
    )

    # EVALUATION

    evaluator = EvaluationModule()

    evaluator.evaluate(
        best_model,
        X_test,
        y_test
    )

    print("\nDONE SUCCESSFULLY")

# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()

