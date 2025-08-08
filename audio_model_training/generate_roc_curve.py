#############################################################################
# note - use to create ROC curve of model without reduce mean layer only    #
#############################################################################

import tensorflow as tf
import numpy as np
import os
from scipy.signal import resample
import matplotlib.pyplot as plt

# === Helper: load WAV and resample to 16k mono ===
def load_wav_16k_mono(filename):
    file_contents = tf.io.read_file(filename)
    wav, sample_rate = tf.audio.decode_wav(file_contents, desired_channels=1)
    wav = tf.squeeze(wav, axis=-1)
    wav_np = wav.numpy()
    sample_rate = int(sample_rate.numpy())

    if sample_rate != 16000:
        duration = len(wav_np) / sample_rate
        target_len = int(round(duration * 16000))
        wav_np = resample(wav_np, target_len)

    return tf.convert_to_tensor(wav_np, dtype=tf.float32)

# === Paths ===
model_path = r'path/to/autoguard_model_without_reduce_mean'
dir_path = r'your/dataset/path'

# === Load model ===
my_model = tf.saved_model.load(model_path)
my_classes = ['obd', 'noise']
class_counts = {'obd': 0, 'noise': 0}

# === Count examples per class ===
for filename in os.listdir(dir_path):
    if filename.endswith('.wav'):
        try:
            label_index = int(filename.split('-')[-1].split('.')[0])
            label = my_classes[label_index]
            class_counts[label] += 1
        except (IndexError, ValueError):
            print(f"Skipping malformed filename: {filename}")

print(f"Total files:")
print(f"  obd:   {class_counts['obd']}")
print(f"  noise: {class_counts['noise']}")

# === ROC calculation ===
thresholds = np.arange(0.0, 1.01, 0.05)
tpr_list = []
fpr_list = []

for thr_value in thresholds:
    tp = fp = tn = fn = 0

    for filename in os.listdir(dir_path):
        if filename.endswith('.wav'):
            file_path = os.path.join(dir_path, filename)
            wav_data = load_wav_16k_mono(file_path)
            results = my_model(wav_data)
            probs = tf.nn.softmax(results, axis=1).numpy()  # (frames, 2)

            try:
                true_label_index = int(filename.split('-')[-1].split('.')[0])
                true_label = my_classes[true_label_index]
            except:
                continue

            for frame_prob in probs:
                prob_obd = frame_prob[0]
                predicted_label = 'obd' if prob_obd > thr_value else 'noise'

                if true_label == 'obd':
                    if predicted_label == 'obd':
                        tp += 1
                    else:
                        fn += 1
                elif true_label == 'noise':
                    if predicted_label == 'obd':
                        fp += 1
                    else:
                        tn += 1

    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    tpr_list.append(tpr)
    fpr_list.append(fpr)

# === Sort and compute AUC ===
sorted_indices = np.argsort(fpr_list)
sorted_fpr = np.array(fpr_list)[sorted_indices]
sorted_tpr = np.array(tpr_list)[sorted_indices]
sorted_thresholds = np.array(thresholds)[sorted_indices]

print("\nROC Data:")
for thr, tpr, fpr in zip(sorted_thresholds, sorted_tpr, sorted_fpr):
    print(f"Threshold: {thr:.2f} | TPR: {tpr:.3f} | FPR: {fpr:.3f}")

auc_value = np.trapz(sorted_tpr, sorted_fpr)

# === Plot ROC ===
plt.figure(figsize=(8, 6))
plt.plot(fpr_list, tpr_list, marker='o', label='ROC curve')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'Frame-Level ROC Curve (AUC = {auc_value:.3f})')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("frame_level_roc_curve.svg", format='svg')
plt.show()
