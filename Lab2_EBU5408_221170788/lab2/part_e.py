import os
import glob
import numpy as np
import librosa.display
import matplotlib.pyplot as plt
import soundfile as sf
from sklearn.decomposition import FastICA
from scipy.signal import butter, filtfilt
from scipy.signal import iirnotch, wiener


# -----------------------------
# Parameters
# -----------------------------
frame_size = 1024 # Kept for potential use in spectrograms, but not for ICA preprocessing
hop_size = frame_size // 2  # Kept for potential use in spectrograms

# Low‑pass filter parameters
butter_order = 4            # 4th‑order Butterworth
lowpass_cutoff = 0.35        # 20 % of Nyquist
highpass_cutoff = 0.01      # 1 % of Nyquist (≈ 0.5–240 Hz depending on sr)

# Post‑processing filter parameters
speech_lp_cutoff = 0.35   # 0.35 of Nyquist  (≈ 8 kHz @ 48 kHz)
hum_freq = 50.0           # Hz (mains hum)
hum_q = 30                # Quality factor for notch

# Directories
input_folder = 'MysteryAudioLab2_part_e'
output_folder = 'Audio_output/PartE'
image_output = os.path.join('image_output', 'PartE')

# Make sure directories exist
os.makedirs(output_folder, exist_ok=True)
os.makedirs(image_output, exist_ok=True)

# Print the absolute path to help debug
abs_input_path = os.path.abspath(input_folder)
print(f"Looking for audio files in: {abs_input_path}")


# -----------------------------
# Load and preprocess all microphone recordings
# -----------------------------
file_paths = sorted(glob.glob(os.path.join(input_folder, '*.wav')))

# Check if files were found, if not try other extensions
if not file_paths:
    print("No .wav files found. Trying other audio formats...")
    for ext in ['.mp3', '.flac', '.ogg', '.aac']:
        file_paths = sorted(glob.glob(os.path.join(input_folder, f'*{ext}')))
        if file_paths:
            print(f"Found {len(file_paths)} files with extension {ext}")
            break

# If still no files, exit gracefully
if not file_paths:
    print("Error: No audio files found in the input directory.")
    print("Please check that the files exist and the path is correct.")
    print(f"Current directory: {os.getcwd()}")
    print(f"Files in input directory: {os.listdir(input_folder) if os.path.exists(input_folder) else 'Directory not found'}")
    exit(1)

audio_data = []
sr = None  # Will store sampling rate

print(f"Found {len(file_paths)} microphone recordings.")

for file_path in file_paths:
    base = os.path.splitext(os.path.basename(file_path))[0]
    print(f'Loading and preprocessing {base}...')

    # Load audio, handle multi-channel by averaging to mono
    y_multi, file_sr = librosa.load(file_path, sr=None, mono=False)
    if y_multi.ndim > 1:
        y = np.mean(y_multi, axis=0)
        print(f'  Converted multi-channel to mono for {base}')
    else:
        y = y_multi

    # Set or verify sampling rate
    if sr is None:
        sr = file_sr
    elif sr != file_sr:
        print(f"  Warning: Sample rate mismatch! {file_sr} vs {sr}. Resampling...")
        # Resample if needed
        y = librosa.resample(y, orig_sr=file_sr, target_sr=sr)

    # --- Low-pass filter (applied before normalization) ---
    nyq = sr / 2
    b_lp, a_lp = butter(butter_order, lowpass_cutoff, btype='low')
    y = filtfilt(b_lp, a_lp, y)
    # -------------------------------------------------------

    # --- High-pass filter to remove DC / ultra‑low frequencies ---
    b_hp, a_hp = butter(butter_order, highpass_cutoff, btype='high')
    y = filtfilt(b_hp, a_hp, y)
    # ------------------------------------------------------------

    # Normalize the signal (Z-score normalization)
    y_mean = np.mean(y)
    y_std = np.std(y)
    if y_std > 1e-8: # Avoid division by zero for silent signals
        y_norm = (y - y_mean) / y_std
    else:
        y_norm = y # Keep silent signal as is
    print(f'  Applied Z-score normalization to {base}')
    audio_data.append(y_norm)

# Make sure all recordings have the same length
min_length = min(len(y) for y in audio_data)
audio_data = [y[:min_length] for y in audio_data]

# Create the observed mixture signals matrix (channels x samples)
X = np.vstack(audio_data)  # This is our observed mixture matrix
print(f"Created mixture matrix X with shape: {X.shape}")

# Center the data (subtract the mean from each microphone signal)
X_mean = np.mean(X, axis=1, keepdims=True)
X_centered = X - X_mean
print("Centered the mixture data matrix X.")

# -----------------------------
# Visualization of the observed mixtures
# -----------------------------
plt.figure(figsize=(12, 8))
times_plot = np.arange(X_centered.shape[1]) / sr
for i, signal_centered in enumerate(X_centered):
    plt.subplot(len(audio_data), 1, i + 1)
    plt.plot(times_plot, signal_centered)
    plt.title(f'Observed Centered Mixture Signal from Mic {i + 1}')
    plt.ylabel('Amplitude')
plt.xlabel('Time (s)')
plt.tight_layout()
plt.savefig(os.path.join(image_output, 'observed_centered_mixtures.png'))
plt.close()

# -----------------------------
# Apply ICA
# -----------------------------
# Assume number of sources equals number of microphones
n_components = len(audio_data)

print(f"Applying ICA to separate {n_components} sources...")

# Initialize FastICA - whiten='unit-variance' handles scaling/variance normalization
# Setting whiten='unit-variance' makes explicit centering somewhat redundant, but doesn't hurt.
# If whiten=False was used, centering X_centered would be crucial.
ica = FastICA(n_components=n_components, random_state=42, max_iter=1000, tol=0.001, whiten='unit-variance')

# Apply ICA to the centered data to get the estimated source signals
# S = W * X_centered
S = ica.fit_transform(X_centered.T).T  # Estimated source signals (n_components x samples)
W = ica.components_  # Demixing matrix (W) (n_components x n_features/microphones)

# The mixing matrix A (which satisfies X_centered ≈ A * S) is the pseudo-inverse of W
# Or, it can be obtained from ica.mixing_ if whiten=True
A = ica.mixing_ # Mixing matrix (A) (n_features/microphones x n_components)

print("ICA completed successfully.")
print("Demixing matrix (W) shape:", W.shape)
print("Mixing matrix (A) shape:", A.shape)
print("Estimated sources (S) shape:", S.shape)

# Print the matrices
print("\nDemixing Matrix (W):")
print(W)
print("\nMixing Matrix (A):")
print(A)

# -----------------------------
# Visualize the ICA results
# -----------------------------
plt.figure(figsize=(12, 8))
for i, source in enumerate(S):
    plt.subplot(n_components, 1, i + 1)
    plt.plot(np.arange(len(source)) / sr, source)
    plt.title(f'Estimated Source Signal {i + 1}')
    plt.ylabel('Amplitude')
plt.xlabel('Time (s)')
plt.tight_layout()
plt.savefig(os.path.join(image_output, 'estimated_sources.png'))
plt.close()

# Visualize the mixing matrix
plt.figure(figsize=(8, 6))
plt.imshow(A, cmap='viridis')
plt.colorbar(label='Coefficient Value')
plt.title('Mixing Matrix (A)')
plt.xlabel('Source')
plt.ylabel('Microphone')
plt.tight_layout()
plt.savefig(os.path.join(image_output, 'mixing_matrix.png'))
plt.close()

# -----------------------------
# Save separated audio files
# -----------------------------
for i, source in enumerate(S):
    # --- Post‑processing chain --------------------------------------
    # 1) Low‑pass to remove high‑freq hiss
    b_sp, a_sp = butter(butter_order, speech_lp_cutoff, btype='low')
    proc = filtfilt(b_sp, a_sp, source)

    # 2) Notch filter at mains hum frequency
    b_notch, a_notch = iirnotch(w0=hum_freq/(sr/2), Q=hum_q)
    proc = filtfilt(b_notch, a_notch, proc)

    # 3) Wiener smoothing (adaptive spectral subtraction style)
    proc = wiener(proc)

    # ---------------------------------------------------------------

    # Robust normalisation to avoid clipping
    max_val = np.percentile(np.abs(proc), 99.9)
    if max_val < 1e-6:
        proc_norm = proc
    else:
        proc_norm = proc / max_val * 0.9

    # Save filtered source
    output_path = os.path.join(output_folder, f'separated_source_{i + 1}.wav')
    sf.write(output_path, proc_norm, sr)
    print(f"Saved filtered & separated source {i + 1} to {output_path}")

# -----------------------------
# Optional: Spectrograms of the separated sources
# -----------------------------
plt.figure(figsize=(15, 10))
for i, source in enumerate(S):
    plt.subplot(n_components, 1, i + 1)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(source, n_fft=frame_size, hop_length=hop_size)), ref=np.max)
    librosa.display.specshow(D, sr=sr, hop_length=hop_size, x_axis='time', y_axis='hz')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'Spectrogram: Separated Source {i + 1}')
plt.tight_layout()
plt.savefig(os.path.join(image_output, 'separated_spectrograms.png'))
plt.close()

# -----------------------------
# Cocktail Party Effect Visualization
# -----------------------------
# Create a mixing/demixing visualization to demonstrate the cocktail party problem
plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
plt.imshow(A, cmap='RdBu', vmin=-np.max(np.abs(A)), vmax=np.max(np.abs(A)))
plt.colorbar()
plt.title('Mixing Matrix (A)\nX = A · S')
plt.xlabel('Source Signals')
plt.ylabel('Microphone Recordings')

plt.subplot(1, 2, 2)
plt.imshow(W, cmap='RdBu', vmin=-np.max(np.abs(W)), vmax=np.max(np.abs(W)))
plt.colorbar()
plt.title('Demixing Matrix (W)\nS = W · X')
plt.xlabel('Microphone Recordings')
plt.ylabel('Estimated Source Signals')
plt.tight_layout()
plt.savefig(os.path.join(image_output, 'cocktail_party_matrices.png'))
plt.close()

print('ICA source separation completed successfully.')

