from sklearn.utils import shuffle

import os
import glob
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import ShortTermFeatures

# --- 1. Feature Extraction Function ---
# def extract_features(file_path, keep_deltas=True):
#     [Fs, x] = audioBasicIO.read_audio_file(file_path)
#     F, f_names = ShortTermFeatures.feature_extraction(x, Fs, 0.050*Fs, 0.025*Fs)
#     features = F.mean(axis=1)
#     if not keep_deltas:
#         features = features[:34]
#     return features

indices_to_drop = [
    # ZCR
    #0,                    # zcr, delta zcr

    # Energy
    #1, 2,              # energy, energy_entropy, delta energy, delta energy_entropy

    # Spectral
    #3, 4, 5, 6, 7,            # spectral features
    #*range(8,20),

    # Chroma
    #*range(21, 34),           # chroma features
]


def extract_features(file_path, keep_deltas=False, drop_indices=indices_to_drop):
    [Fs, x] = audioBasicIO.read_audio_file(file_path)
    F, f_names = ShortTermFeatures.feature_extraction(x, Fs, 0.050*Fs, 0.025*Fs)
    features = F.mean(axis=1)
    if not keep_deltas:
        features = features[:34]
    features = np.delete(features, drop_indices)
    return features

# --- 2. Collect Features and Labels ---
X = []
y = []

# --- 2.1 Read OBD files ---
obd_folder = "data/obd"
for file in glob.glob(os.path.join(obd_folder, "*.wav")):
    features = extract_features(file)
    X.append(features)
    y.append(0)  # Label: OBD

# --- 2.2 Read Noise files ---
noise_folder = "data/noise"
for file in glob.glob(os.path.join(noise_folder, "*.wav")):
    features = extract_features(file)
    X.append(features)
    y.append(1)  # Label: Noise

X = np.array(X)
y = np.array(y)
X, y = shuffle(X, y, random_state=0)

print(f"Total samples: {len(y)}")

# --- 3. Split Data ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y)

print(f"Train samples: {len(y_train)}, Test samples: {len(y_test)}")

# --- 4. Train KNN Classifier ---
knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train, y_train)

# --- 5. Evaluate KNN Model ---
y_pred_knn = knn.predict(X_test)
print("\n--- KNN Results ---")
print("Accuracy:", accuracy_score(y_test, y_pred_knn))
print(classification_report(y_test, y_pred_knn, target_names=["OBD", "Noise"]))

# --- 6. Train Random Forest Classifier ---
rf = RandomForestClassifier(n_estimators=50, max_depth=2, min_samples_leaf=3, random_state=42)
rf.fit(X_train, y_train)

# --- 7. Evaluate Random Forest Model ---
y_pred_rf = rf.predict(X_test)
print("\n--- Random Forest Results ---")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf, target_names=["OBD", "Noise"]))

# --- 8. Train SVM Classifier ---
svm = SVC(kernel='linear', probability=True, random_state=42)
svm.fit(X_train, y_train)

# --- 9. Evaluate SVM Model ---
y_pred_svm = svm.predict(X_test)
print("\n--- SVM Results ---")
print("Accuracy:", accuracy_score(y_test, y_pred_svm))
print(classification_report(y_test, y_pred_svm, target_names=["OBD", "Noise"]))