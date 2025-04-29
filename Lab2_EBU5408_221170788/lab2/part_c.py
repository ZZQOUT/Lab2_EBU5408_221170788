import os
import glob
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import soundfile as sf
from sklearn.decomposition import FastICA
from scipy.signal import butter, filtfilt
from scipy.signal import iirnotch, wiener


# -----------------------------------------------------------
# Helper: load reference sources from PartB outputs
def load_partb_references(sr, n_components):

    ref_dir = os.path.join('Audio_output', 'PartB')
    refs = []
    for i in range(1, n_components + 1):
        fp = os.path.join(ref_dir, f'separated_source_{i}.wav')
        if os.path.exists(fp):
            sig, sr_file = sf.read(fp)
            if sig.ndim > 1:
                sig = np.mean(sig, axis=1)
            if sr_file != sr:
                sig = librosa.resample(sig, orig_sr=sr_file, target_sr=sr)
            refs.append(sig)
        else:
            print(f"Warning: reference file {fp} not found.")
            refs.append(None)
    return refs
# -----------------------------------------------------------
# Quick sanity-check: make sure PartB references are not identical to mic tracks
def validate_references(refs, mic_signals):

    for i, ref in enumerate(refs):
        if ref is None:
            continue
        mic = mic_signals[i]
        n = min(len(ref), len(mic))
        if n < 256:
            continue
        corr = np.corrcoef(ref[:n], mic[:n])[0, 1]
        print(f"Validation: Corr(Ref{i+1}, Mic{i+1}) = {corr:.3f}")
        if corr > 0.95:
            print(" High correlation suggests the reference may still be the raw microphone.")


# -----------------------------------------------------------
# Ensure current ICA input (mic‑derived) is not identical to PartB references
def validate_input_vs_ref(X_mics, refs, threshold=0.95):

    for ch in range(len(refs)):
        ref = refs[ch]
        if ref is None:
            continue
        mic = X_mics[ch]
        n = min(len(ref), len(mic))
        if n < 256:
            continue
        corr = np.corrcoef(ref[:n], mic[:n])[0, 1]
        if corr > threshold:
            print(f"Corr(Mic{ch+1}, Ref{ch+1}) = {corr:.3f}  > {threshold}. "
                  f"Check that ICA input is not accidentally using PartB output.")
        else:
            print(f"Input‑Ref correlation (Mic{ch+1}) = {corr:.3f}")



# -----------------------------
# Objective‑metric helper
# -----------------------------
def compute_snr(reference, estimate, eps=1e-12):
    """Return segment‑level SNR in dB."""
    noise = reference - estimate
    return 10 * np.log10(np.sum(reference ** 2) / (np.sum(noise ** 2) + eps))

# -----------------------------------------------------------
# Intermodulation Distortion (IMD) helper
def compute_imd(reference, estimate, eps=1e-12):
    """
    Intermodulation Distortion (simplified): ratio of intermodulation artefacts
    to desired signal, expressed in dB.  Here we treat everything that is not
    linearly mapped (ref→est) as intermodulation error.
    """
    error = reference - estimate
    imd = 20 * np.log10(np.sqrt(np.sum(error ** 2)) / (np.sqrt(np.sum(reference ** 2)) + eps))
    return imd
# -----------------------------------------------------------
# Mean‑Squared Error helper
def compute_mse(reference, estimate):
    """Return mean‑squared error between reference and estimate."""
    return np.mean((reference - estimate) ** 2)

# -----------------------------------------------------------
# Total Harmonic Distortion helper
def compute_thd(signal_in, sr, eps=1e-12):
    """
    Compute THD (dB) of a mono signal.
    1) Remove DC
    2) FFT -> find fundamental (largest bin excluding DC)
    3) THD = 20*log10(rms(harmonics)/rms(fundamental))
    Simplified single‑frame estimate.
    """
    x = signal_in - np.mean(signal_in)
    N = len(x)
    # Hann window to reduce spectral leakage
    window = np.hanning(N)
    X = np.abs(np.fft.rfft(x * window))
    # Ignore DC bin
    X[0] = 0.0
    # Fundamental = max magnitude bin
    fund_idx = np.argmax(X)
    fund_mag = X[fund_idx]
    # RMS of fundamental (single bin)
    rms_fund = fund_mag / np.sqrt(2)
    # THD power = sum of squares of all other bins
    harm_power = np.sum(X**2) - fund_mag**2
    rms_harm = np.sqrt(harm_power) / np.sqrt(2)
    thd = 20 * np.log10((rms_harm + eps) / (rms_fund + eps))
    return thd


# -----------------------------
# Parameters
# -----------------------------
frame_size = 1024 # Kept for potential use in spectrograms, but not for ICA preprocessing
hop_size = frame_size // 2  # Kept for potential use in spectrograms
# Low‑pass filter parameters
butter_order = 4            # 4th‑order Butterworth
lowpass_cutoff = 0.35        # 20% of Nyquist
highpass_cutoff = 0.01      # 1% of Nyquist (≈0.5–240Hz depending on sr)
# Post‑processing filter parameters
speech_lp_cutoff = 0.35   # 0.35 of Nyquist  (≈ 8kHz @ 48kHz)
hum_freq = 50.0           # Hz (mains hum)
hum_q = 30                # Quality factor for notch
# Directories
input_folder = 'MysteryAudioLab2'
output_folder = 'Audio_output/PartC'
image_output = os.path.join('image_output', 'PartC')
os.makedirs(output_folder, exist_ok=True)
os.makedirs(image_output, exist_ok=True)

# -----------------------------
# Load and preprocess all microphone recordings
# -----------------------------
file_paths = sorted(glob.glob(os.path.join(input_folder, '*.wav')))
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

ref_sources_baseline = load_partb_references(sr, len(audio_data))
validate_input_vs_ref(X, ref_sources_baseline)

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

def run_ica_and_save(
    X_centered, audio_data, sr, param_tag,
    n_components, tol, whiten, max_iter,
    butter_order, lowpass_cutoff, highpass_cutoff,
    speech_lp_cutoff, hum_freq, hum_q,
    output_folder, image_output
):
    # --- Create subfolders for this parameter set ---
    audio_subfolder = os.path.join(output_folder, param_tag)
    image_subfolder = os.path.join(image_output, param_tag)
    os.makedirs(audio_subfolder, exist_ok=True)
    os.makedirs(image_subfolder, exist_ok=True)

    # --- Fix for FastICA whiten parameter ---
    if whiten is True:
        whiten_param = 'arbitrary-variance'
    elif whiten is False:
        whiten_param = False
        print("Note: When whiten=False, n_components is ignored and set to n_features.")
        # Store the original n_components for plotting purposes
        orig_n_components = n_components
        # When whiten=False, n_components must equal n_features to avoid warning
        n_components = X_centered.shape[0]
    else:
        whiten_param = whiten

    # --- Fix n_components to avoid warning ---
    # n_components cannot exceed min(n_samples, n_features)
    max_possible_components = min(X_centered.shape[0], X_centered.shape[1])
    if n_components > max_possible_components:
        print(f"Warning: n_components={n_components} is too large for data shape {X_centered.shape}. "
              f"Setting to maximum possible value: {max_possible_components}")
        n_components = max_possible_components

    # Validate current parameter set's input vs references
    current_refs = load_partb_references(sr, X_centered.shape[0])
    validate_input_vs_ref(X_centered, current_refs)

    # ICA with convergence handling
    try:
        # Suppress specific warnings during ICA
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.filterwarnings("ignore", message="Ignoring n_components with whiten=False.")

            # Check for problematic values in the data
            if np.any(~np.isfinite(X_centered)):
                raise ValueError("Input data contains NaN or Inf values")

            ica = FastICA(
                n_components=n_components,
                random_state=42,
                max_iter=max_iter,
                tol=tol,
                whiten=whiten_param
            )
            S = ica.fit_transform(X_centered.T).T

            # Check for convergence warning
            convergence_warning = any("did not converge" in str(warning.message) for warning in w)
            if convergence_warning:
                print(f"Warning: FastICA did not converge with the given parameters. Results may be suboptimal.")
                # Add note to metrics title
                convergence_note = " (Did not converge)"
            else:
                convergence_note = ""

            W = ica.components_
            A = ica.mixing_

    except Exception as e:
        print(f"Error during ICA: {str(e)}")
        print("Using fallback PCA decomposition...")
        # Fallback to PCA if ICA completely fails
        from sklearn.decomposition import PCA

        try:
            # Use with_mean=False if there are numerical issues
            pca = PCA(n_components=n_components)
            S = pca.fit_transform(X_centered.T).T

            # Create equivalent matrices for PCA
            W = pca.components_  # PCA components are similar to ICA's demixing matrix

            # Calculate pseudo mixing matrix (similar to ICA's mixing_)
            # For PCA: X ≈ X_mean + S @ W
            # We want A such that X ≈ A @ S
            # A is approximately the pseudoinverse of W
            try:
                A = np.linalg.pinv(W)
            except np.linalg.LinAlgError:
                print("Warning: Could not compute pseudo-inverse. Using transpose instead.")
                A = W.T  # Approximation when W is orthogonal

            convergence_note = " (Used PCA fallback)"

        except Exception as pca_error:
            print(f"PCA fallback also failed: {str(pca_error)}")
            print("Using simple SVD decomposition as last resort...")

            # Last resort: simple SVD
            U, s, Vh = np.linalg.svd(X_centered, full_matrices=False)
            n = min(n_components, len(s))
            S = Vh[:n]  # Take only n_components rows
            W = Vh[:n]  # Equivalent to demixing matrix
            A = U[:, :n] * s[:n]  # Equivalent to mixing matrix
            convergence_note = " (Used SVD fallback)"

    # Plot estimated sources
    plt.figure(figsize=(12, 8))
    for i, source in enumerate(S):
        plt.subplot(len(S), 1, i + 1)
        plt.plot(np.arange(len(source)) / sr, source)
        plt.title(f'Estimated Source Signal {i + 1}{convergence_note}')
        plt.ylabel('Amplitude')
    plt.xlabel('Time (s)')
    plt.tight_layout()
    plt.savefig(os.path.join(image_subfolder, f'estimated_sources.png'))
    plt.close()

    # Plot mixing matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(A, cmap='viridis')
    plt.colorbar(label='Coefficient Value')
    plt.title(f'Mixing Matrix (A){convergence_note}')
    plt.xlabel('Source')
    plt.ylabel('Microphone')
    plt.tight_layout()
    plt.savefig(os.path.join(image_subfolder, f'mixing_matrix.png'))
    plt.close()

    # Save separated audio
    for i, source in enumerate(S):
        # Post-processing
        b_sp, a_sp = butter(butter_order, speech_lp_cutoff, btype='low')
        proc = filtfilt(b_sp, a_sp, source)
        b_notch, a_notch = iirnotch(w0=hum_freq/(sr/2), Q=hum_q)
        proc = filtfilt(b_notch, a_notch, proc)
        proc = wiener(proc)
        max_val = np.percentile(np.abs(proc), 99.9)
        proc_norm = proc if max_val < 1e-6 else proc / max_val * 0.9
        output_path = os.path.join(audio_subfolder, f'separated_source_{i + 1}.wav')
        sf.write(output_path, proc_norm, sr)

    # Metrics
    snr_vals, imd_vals, mse_vals, thd_vals = [], [], [], []
    ref_sources = load_partb_references(sr, len(S))
    validate_references(ref_sources, audio_data)  # global audio_data available
    for idx in range(len(S)):

        ref = ref_sources[idx] if ref_sources[idx] is not None else audio_data[idx]
        deg = S[idx]
        ref_len = min(len(ref), len(deg))  # Use shorter length

        snr_val = compute_snr(ref[:ref_len], deg[:ref_len])
        imd_val = compute_imd(ref[:ref_len], deg[:ref_len])
        mse_val = compute_mse(ref[:ref_len], deg[:ref_len])
        thd_val = compute_thd(deg[:ref_len], sr)
        snr_vals.append(snr_val)
        imd_vals.append(imd_val)
        mse_vals.append(mse_val)
        thd_vals.append(thd_val)

    avg_metrics = {
        'snr': np.mean(snr_vals),
        'imd': np.mean(imd_vals),
        'mse': np.mean(mse_vals),
        'thd': np.mean(thd_vals),
        'converged': not bool(convergence_note)  # Track convergence status
    }

    # Spectrograms
    plt.figure(figsize=(15, 10))
    for i, source in enumerate(S):
        plt.subplot(len(S), 1, i + 1)
        D = librosa.amplitude_to_db(np.abs(librosa.stft(source, n_fft=frame_size, hop_length=hop_size)), ref=np.max)
        librosa.display.specshow(D, sr=sr, hop_length=hop_size, x_axis='time', y_axis='hz')
        plt.colorbar(format='%+2.0f dB')
        plt.title(f'Spectrogram: Separated Source {i + 1}{convergence_note}')
    plt.tight_layout()
    plt.savefig(os.path.join(image_subfolder, f'separated_spectrograms.png'))
    plt.close()

    return avg_metrics

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
baseline_post_processed = []   # 保存后处理后的信号供指标计算
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
    baseline_post_processed.append(proc_norm)

# -----------------------------
# Objective evaluation: SNR, IMD, MSE, THD
# -----------------------------
print("\nObjective metrics between PartB results and its matched separated source:")
snr_vals, imd_vals, mse_vals, thd_vals = [], [], [], []
ref_sources = load_partb_references(sr, n_components)
validate_references(ref_sources, audio_data)

for idx in range(len(baseline_post_processed)):
    ref = ref_sources[idx]  # Always use Part B reference
    if ref is None:
        print(f"Skipping MSE calculation for source {idx + 1}: No reference available.")
        continue  # Skip if reference is missing

    deg = baseline_post_processed[idx]
    ref_len = min(len(ref), len(deg))
    ref = ref[:ref_len]
    deg = deg[:ref_len]

    # --- SNR ---
    snr_val = compute_snr(ref, deg)
    snr_vals.append(snr_val)

    # --- IMD (Intermodulation Distortion) ---
    imd_val = compute_imd(ref, deg)
    imd_vals.append(imd_val)

    # --- MSE (Mean‑Squared Error) ---
    mse_val = compute_mse(ref, deg)  # Use only Part B reference
    mse_vals.append(mse_val)

    # --- THD (Total Harmonic Distortion) ---
    thd_val = compute_thd(deg, sr)
    thd_vals.append(thd_val)

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

# -----------------------------
# Parameter sweep for analysis
# -----------------------------
# IMPORTANT: Create a subfolder for parameter sweep results to avoid overwriting baseline results
param_sweep_folder = os.path.join(output_folder, 'param_sweep')
os.makedirs(param_sweep_folder, exist_ok=True)

param_grid = [
    # (n_components, tol, whiten, max_iter, butter_order, lowpass_cutoff, highpass_cutoff, speech_lp_cutoff, hum_freq, hum_q)
    # Baseline (reasonable parameters)
    (len(audio_data), 0.001, 'unit-variance', 1000, 4, 0.35, 0.01, 0.35, 50.0, 30),

    # Extreme 1: Minimal components (1), very loose tolerance, minimal filtering
    (1, 0.5, 'unit-variance', 50, 1, 0.95, 0.3, 0.95, 150.0, 1),

    # Extreme 2: Maximum components, extremely strict tolerance, heavy filtering
    (min(len(audio_data)+2, 10), 1e-8, 'arbitrary-variance', 5000, 12, 0.02, 0.001, 0.02, 10.0, 200),

    # Extreme 3: No whitening, very few iterations
    (len(audio_data), 0.05, False, 20, 3, 0.4, 0.1, 0.4, 60.0, 10),

    # Extreme 4: Minimal iterations, max tolerance (barely converges)
    (len(audio_data), 0.9, 'unit-variance', 5, 1, 0.01, 0.8, 0.01, 300.0, 0.5),

    # Extreme 5: Focus on minimal low-pass filtering (keeps high frequencies)
    (2, 0.001, 'arbitrary-variance', 1000, 6, 0.99, 0.01, 0.99, 50.0, 30),

    # Extreme 7: Very high filter order
    (len(audio_data), 0.001, 'unit-variance', 1000, 20, 0.35, 0.01, 0.35, 50.0, 30),

    # Extreme 8: No filtering (pass everything)
    (len(audio_data), 0.001, 'unit-variance', 1000, 1, 0.9999, 0.0001, 0.9999, 50.0, 1),

    # Extreme 9: Severe band-limited filtering (narrow passband)
    (len(audio_data), 0.001, 'unit-variance', 1000, 8, 0.1, 0.09, 0.1, 50.0, 100),
]

metrics_results = []
for idx, (n_components, tol, whiten, max_iter, butter_order, lowpass_cutoff, highpass_cutoff, speech_lp_cutoff, hum_freq, hum_q) in enumerate(param_grid):
    param_tag = f"nc{n_components}_tol{tol}_w{whiten}_mi{max_iter}_bo{butter_order}_lpc{lowpass_cutoff}_hpc{highpass_cutoff}_slpc{speech_lp_cutoff}_hf{hum_freq}_hq{hum_q}"
    print(f"\n=== Running parameter set {idx+1}/{len(param_grid)}: {param_tag} ===")

    # Update output folder to use param_sweep_folder instead of output_folder
    # This ensures parameter sweep results don't overwrite baseline results
    curr_output_folder = os.path.join(param_sweep_folder, param_tag)
    os.makedirs(curr_output_folder, exist_ok=True)
    
    try:
        # Re-filter and normalize audio_data for each filter parameter set
        filtered_audio_data = []
        for y in audio_data:
            nyq = sr / 2

            # Safety limits for filter parameters to prevent numerical issues
            safe_butter_order = min(butter_order, 8)  # Limit order to avoid instability
            safe_lowpass_cutoff = max(0.05, min(lowpass_cutoff, 0.95))  # Keep in stable range
            safe_highpass_cutoff = max(0.001, min(highpass_cutoff, 0.9))  # Keep in stable range

            # Apply lowpass filter
            b_lp, a_lp = butter(safe_butter_order, safe_lowpass_cutoff, btype='low')
            y_filt = filtfilt(b_lp, a_lp, y)

            # Check for NaNs after lowpass
            if np.any(np.isnan(y_filt)):
                print(f"Warning: NaN detected after lowpass filter. Using safer parameters.")
                b_lp, a_lp = butter(2, 0.5, btype='low')
                y_filt = filtfilt(b_lp, a_lp, y)

            # Apply highpass filter
            b_hp, a_hp = butter(safe_butter_order, safe_highpass_cutoff, btype='high')
            y_filt = filtfilt(b_hp, a_hp, y_filt)

            # Check for NaNs after highpass
            if np.any(np.isnan(y_filt)):
                print(f"Warning: NaN detected after highpass filter. Using safer parameters.")
                b_hp, a_hp = butter(2, 0.01, btype='high')
                y_filt = filtfilt(b_hp, a_hp, y)

            # Normalize
            y_mean = np.mean(y_filt)
            y_std = np.std(y_filt)
            if y_std > 1e-6:
                y_norm = (y_filt - y_mean) / y_std
            else:
                y_norm = y_filt

            # Final NaN check and replacement
            if np.any(np.isnan(y_norm)):
                print(f"Warning: NaN values in normalized signal. Replacing with zeros.")
                y_norm = np.nan_to_num(y_norm)

            filtered_audio_data.append(y_norm)

        min_length = min(len(y) for y in filtered_audio_data)
        filtered_audio_data = [y[:min_length] for y in filtered_audio_data]
        X_param = np.vstack(filtered_audio_data)
        X_mean = np.mean(X_param, axis=1, keepdims=True)
        X_centered_param = X_param - X_mean

        # Validate current parameter set's input vs references
        current_refs = load_partb_references(sr, X_param.shape[0])
        validate_input_vs_ref(X_param, current_refs)

        # Final NaN check for entire matrix
        if np.any(np.isnan(X_centered_param)):
            print(f"Warning: NaN values found in data matrix. Replacing with zeros.")
            X_centered_param = np.nan_to_num(X_centered_param)

        # Use safe parameter values for processing
        metrics = run_ica_and_save(
            X_centered_param, filtered_audio_data, sr, param_tag,
            n_components, tol, whiten, max_iter,
            safe_butter_order, safe_lowpass_cutoff, safe_highpass_cutoff,
            max(0.05, min(speech_lp_cutoff, 0.95)), hum_freq, min(100, max(1, hum_q)),
            param_sweep_folder, image_output  # Use param_sweep_folder here
        )
        metrics['param_tag'] = param_tag
        metrics_results.append(metrics)
    except Exception as e:
        print(f"Error processing parameter set {idx+1}: {str(e)}")
        # Add dummy metrics to maintain the correct length of results
        metrics_results.append({
            'snr': float('nan'),
            'imd': float('nan'),
            'mse': float('nan'),
            'thd': float('nan'),
            'param_tag': param_tag
        })

# Save metrics summary plot
plt.figure(figsize=(16, 7))
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
markers = ['o', 's', '^', 'D']
metrics_to_plot = ['snr', 'imd', 'mse', 'thd']
x = np.arange(len(metrics_results))

for idx, metric in enumerate(metrics_to_plot):
    vals = [m[metric] for m in metrics_results]
    plt.plot(x, vals, label=metric.upper(), color=colors[idx], marker=markers[idx], linewidth=2, markersize=7)

# Mark non-converged runs
for i, result in enumerate(metrics_results):
    if result.get('converged') is False:
        plt.axvspan(i-0.3, i+0.3, alpha=0.2, color='gray')

plt.xticks(x, [m['param_tag'] for m in metrics_results], rotation=45, ha='right', fontsize=9)
plt.xlabel('Parameter Set', fontsize=13)
plt.ylabel('Metric Value', fontsize=13)
plt.title('Separation Quality Metrics vs Parameter Sets', fontsize=15, fontweight='bold')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout(rect=[0, 0, 0.8, 1])  # Leave space for legend

# Place legend outside the plot area on the right
plt.legend(title='Metrics', bbox_to_anchor=(1.02, 0.5), loc='center left', borderaxespad=0, fontsize=12, title_fontsize=13, frameon=True)

plt.savefig(os.path.join(image_output, 'metrics_vs_parameters.png'), bbox_inches='tight', dpi=150)
plt.close()



