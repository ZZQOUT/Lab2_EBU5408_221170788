import os
import glob

import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import soundfile as sf
from sklearn.decomposition import PCA
from scipy.signal import medfilt, butter, filtfilt
# Import necessary for 3D plotting
from mpl_toolkits.mplot3d import Axes3D
import warnings




# -----------------------------
# Parameters
# -----------------------------
frame_size = 1024
hop_size = frame_size // 2           # 50% overlap
window = np.hanning(frame_size)      # Hann window

# PCA threshold
variance_threshold = 0.95

# Post-filter parameters
median_kernel = 5
butter_order = 4
lowpass_cutoff = 0.35  # as fraction of Nyquist
highpass_cutoff = 0.01       # as fraction of Nyquist (≈ 1% of Nyquist, e.g. ~240Hz @ 48kHz)

# Directories
input_folder = 'MysteryAudioLab2'
image_output = os.path.join('image_output', 'PartA')
audio_output = 'Audio_output'
os.makedirs(image_output, exist_ok=True)
os.makedirs(audio_output, exist_ok=True)

# -----------------------------
# Load and Preprocess ALL Files
# -----------------------------
file_paths = sorted(glob.glob(os.path.join(input_folder, '*.wav'))) # Sort for consistent order
all_y_processed = []
all_sr = []
all_bases = []
min_len = float('inf')
sr = None # Assume all files have the same sample rate initially
warnings.filterwarnings("ignore", category=RuntimeWarning)
print("Loading and preprocessing all audio files...")
for file_path in file_paths:
    base = os.path.splitext(os.path.basename(file_path))[0]
    print(f'  Loading {base}...')

    # 1. Load (keep multi-channel, then avg to mono)
    y_multi, current_sr = librosa.load(file_path, sr=None, mono=False)

    # Check sample rate consistency
    if sr is None:
        sr = current_sr
    elif sr != current_sr:
        raise ValueError(f"Sample rate mismatch: Expected {sr} Hz, found {current_sr} Hz in {base}. Please ensure all files have the same sample rate.")

    if y_multi.ndim > 1:
        y = np.mean(y_multi, axis=0)
    else:
        y = y_multi

    # --- Low‑pass filter ---
    nyq = sr / 2                      # Nyquist frequency
    b_lp, a_lp = butter(butter_order, lowpass_cutoff, btype='low')
    y = filtfilt(b_lp, a_lp, y)
    # -------------------------------------------------------

    # --- High‑pass filter ---
    b_hp, a_hp = butter(butter_order, highpass_cutoff, btype='high')
    y = filtfilt(b_hp, a_hp, y)
    # ------------------------------------------------------------

    all_y_processed.append(y)
    all_sr.append(sr) # Store sr for each file (though assumed same)
    all_bases.append(base)
    min_len = min(min_len, len(y))

print(f"Found {len(file_paths)} files. Truncating all to minimum length: {min_len} samples.")
# Truncate all signals to the minimum length
all_y_truncated = [y[:min_len] for y in all_y_processed]

# -----------------------------
# Inter-File PCA Analysis (Treating files as dimensions)
# -----------------------------
num_files = len(all_y_truncated)
if num_files == 3:
    print("\nPerforming Inter-File PCA (treating files as dimensions)...")
    # Stack the signals: shape becomes (min_len, 3)
    Y_combined = np.stack(all_y_truncated, axis=1)

    # Center the data (subtract mean of each signal/dimension)
    Y_mean = np.mean(Y_combined, axis=0)
    Y_centered = Y_combined - Y_mean

    # Perform PCA
    pca_interfile = PCA(n_components=3, svd_solver='full')
    pca_interfile.fit(Y_centered)
    Y_pca = pca_interfile.transform(Y_centered) # Project data onto principal components

    # Explained variance plot
    explained_variance_ratio = pca_interfile.explained_variance_ratio_
    plt.figure()
    plt.bar(range(1, 4), explained_variance_ratio, alpha=0.7, align='center',
            label='Individual explained variance')
    plt.step(range(1, 4), np.cumsum(explained_variance_ratio), where='mid',
             label='Cumulative explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal component index')
    plt.xticks(range(1, 4))
    plt.title('Inter-File PCA Explained Variance')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(image_output, 'pca_interfile_variance.png'))
    plt.close()
    print(f"  Explained Variance Ratios: {explained_variance_ratio}")

    # 3D Scatter plot of PCA projections
    # Plot only a subset of points for clarity if needed
    plot_subset_ratio = 0.1 # Plot 10% of the points
    subset_indices = np.random.choice(Y_pca.shape[0], size=int(Y_pca.shape[0] * plot_subset_ratio), replace=False)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    # Scatter plot using the subset
    ax.scatter(Y_pca[subset_indices, 0], Y_pca[subset_indices, 1], Y_pca[subset_indices, 2], alpha=0.3, s=5) # s is marker size
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_zlabel('Principal Component 3')
    ax.set_title(f'Inter-File PCA Projection (Subset {int(plot_subset_ratio*100)}%)')
    plt.tight_layout()
    plt.savefig(os.path.join(image_output, 'pca_interfile_3d_scatter.png'))
    plt.close()
    print("  Saved inter-file PCA variance and 3D scatter plots.")

else:
    print(f"\nSkipping Inter-File PCA: Expected 3 audio files, but found {num_files}.")


# -----------------------------
# Per-File Processing Loop (using preprocessed data)
# -----------------------------
# Now loop through the pre-loaded, filtered, and truncated data
for i, y in enumerate(all_y_truncated):
    base = all_bases[i]
    # sr is already defined and checked for consistency
    print(f'\nProcessing individual analysis for {base}...')

    # 1. Waveform plot (uses preprocessed y)
    plt.figure()
    times = np.linspace(0, len(y)/sr, len(y))
    plt.plot(times, y)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title(f'Waveform: {base}')
    plt.tight_layout()
    plt.savefig(os.path.join(image_output, f'waveform_{base}.png'))
    plt.close()

    # 2. Spectrogram plot (uses preprocessed y)
    D = np.abs(librosa.stft(y, n_fft=frame_size, hop_length=hop_size))
    DB = librosa.amplitude_to_db(D, ref=np.max)
    plt.figure()
    librosa.display.specshow(DB, sr=sr, hop_length=hop_size, x_axis='time', y_axis='hz')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'Spectrogram: {base}')
    plt.tight_layout()
    plt.savefig(os.path.join(image_output, f'spectrogram_{base}.png'))
    plt.close()

    # 3. Frame signal: 50% overlap + Hann window (uses preprocessed y)
    # No need to align length here, as it's already done
    num_frames = 1 + (len(y) - frame_size) // hop_size
    frames = np.stack([
        y[i*hop_size : i*hop_size + frame_size]
        for i in range(num_frames)
    ])
    frames_windowed = frames * window

    # 4. Center & robust Z-score normalization
    frames_centered = frames_windowed - np.mean(frames_windowed, axis=1, keepdims=True)
    frame_stds = np.std(frames_centered, axis=1, keepdims=True)
    frame_stds[frame_stds < 1e-8] = 1e-8
    frames_norm = frames_centered / frame_stds
    # clean up extremes
    frames_norm = np.nan_to_num(frames_norm, nan=0.0, posinf=0.0, neginf=0.0)
    frames_norm = np.clip(frames_norm, -3.0, 3.0)

    # 5. PCA exploration on valid frames (Per-File Frame PCA)
    var_norm = np.var(frames_norm, axis=1)
    valid_idx = var_norm > 1e-6
    frames_norm_valid = frames_norm[valid_idx]

    pca = PCA(svd_solver='full')
    pca.fit(frames_norm_valid)
    cum_var = np.cumsum(pca.explained_variance_ratio_)

    # find elbow
    k = np.argmax(cum_var >= variance_threshold) + 1

    # 6. Plot cumulative explained variance (Per-File Frame PCA)
    plt.figure()
    plt.plot(np.arange(1, len(cum_var)+1), cum_var, label='Cumulative Variance')
    plt.axvline(k, linestyle='--', linewidth=1)
    plt.axhline(variance_threshold, linestyle='--', linewidth=1)
    ax = plt.gca()
    ax.text(0.02, variance_threshold, f'{variance_threshold:.2f}',
            va='bottom', ha='left', transform=ax.get_yaxis_transform())
    plt.scatter([k], [cum_var[k-1]], marker='o', label=f'{k} pcs')
    plt.xlabel('Number of PCA Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title(f'Per-Frame PCA Explained Variance: {base}')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(image_output, f'pca_cumvar_{base}.png'))
    plt.close()

    print(f'  → kept {k} components for per-frame analysis, saved audio & plots.')


# Combined waveform peaks plot for all files (using preprocessed data)
plt.figure()
times = np.linspace(0, min_len / sr, num=min_len) # Use min_len and shared sr
for i, y in enumerate(all_y_truncated):
    base = all_bases[i]
    # Plot full waveform (positive and negative peaks)
    plt.plot(times, y, label=base)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude Envelope')
plt.title('Combined Waveform Peaks Comparison')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(image_output, 'wave_peaks_comparison.png'))
plt.close()

print('\nAll done.')

