import os
import numpy as np
import librosa
import librosa.display
import tensorflow as tf
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from settings.settings import PATHS, NAMES

# --- Settings ---
sr = 16000
DURATION = 5  # seconds
N_FFT = 1024
HOP_LENGTH = 256
N_MELS = 128

MODEL_CHECKPOINT_PATH = PATHS['model_file']
AUDIO_INPUT_FOLDER = PATHS['input_directory']
AUDIO_OUTPUT_FOLDER = PATHS['output_directory']

input_filename = NAMES['input_file']
output_filename = NAMES['output_file']

input_file_path = os.path.join(AUDIO_INPUT_FOLDER, input_filename)
output_file_path = os.path.join(AUDIO_OUTPUT_FOLDER, output_filename)

# --- Helper functions ---

def generate_spectrogram_from_array(audio_array, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH):
    """Generate a spectrogram directly from an audio array."""
    try:
        target_length = int(sr * DURATION)
        if len(audio_array) < target_length:
            audio_array = np.pad(audio_array, (0, target_length - len(audio_array)), mode='constant')
        else:
            audio_array = audio_array[:target_length]

        stft_result = librosa.stft(audio_array, n_fft=n_fft, hop_length=hop_length)
        magnitude_spectrogram = np.abs(stft_result)
        log_magnitude_spectrogram = librosa.amplitude_to_db(magnitude_spectrogram, ref=np.max)
        return log_magnitude_spectrogram
    except Exception as e:
        print(f"Error processing audio array: {e}")
        return None

def normalize_spectrogram(spec):
    min_val = np.min(spec)
    max_val = np.max(spec)
    if max_val == min_val:
        return np.zeros_like(spec)
    return (spec - min_val) / (max_val - min_val)

def denormalize_spectrogram(norm_spec, original_min, original_max):
    if original_max == original_min:
        return np.full_like(norm_spec, original_min)
    return norm_spec * (original_max - original_min) + original_min

def spectrogram_to_audio(spec_norm, n_fft=N_FFT, hop_length=HOP_LENGTH,  original_min=-80.0, original_max=0.0):
    denorm_spec_db = denormalize_spectrogram(spec_norm, original_min, original_max)
    amplitude_spec = librosa.db_to_amplitude(denorm_spec_db)
    amplitude_spec = np.squeeze(amplitude_spec)

    expected_bins = 1 + n_fft // 2
    current_bins = amplitude_spec.shape[0]

    if current_bins > expected_bins:
        amplitude_spec = amplitude_spec[:expected_bins, :]
    elif current_bins < expected_bins:
        pad_width = expected_bins - current_bins
        amplitude_spec = np.pad(amplitude_spec, ((0, pad_width), (0, 0)), mode='constant')

    audio = librosa.griffinlim(amplitude_spec, n_iter=64, hop_length=hop_length, n_fft=n_fft)
    return audio


def pad_to_multiple_of_32(spec):
    height, width = spec.shape[:2]
    pad_h = (32 - height % 32) % 32
    pad_w = (32 - width % 32) % 32
    return np.pad(spec, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant')


def denoise_audio():
    # --- Load model ---
    print(f"Loading model from: {MODEL_CHECKPOINT_PATH}")
    model = tf.keras.models.load_model(MODEL_CHECKPOINT_PATH)

    # --- Load full input audio ---
    print(f"Loading input audio: {input_file_path}")
    y_full, _ = librosa.load(input_file_path, sr=sr, mono=True)

    # --- Prepare chunks with 50% overlap ---
    chunk_length_samples = sr * DURATION  # Samples in one chunk (5 sec)
    hop_samples = chunk_length_samples // 2  # 50% overlap (2.5 sec shift)

    chunks = []
    start = 0
    while start < len(y_full):
        end = start + chunk_length_samples
        chunk = y_full[start:end]
        if len(chunk) < chunk_length_samples:
            chunk = np.pad(chunk, (0, chunk_length_samples - len(chunk)), mode='constant')
        chunks.append(chunk)
        start += hop_samples  # move by 50%

    print(f"Total chunks: {len(chunks)}")

    # --- Denoise each chunk and reconstruct ---
    final_audio = np.zeros(len(y_full) + chunk_length_samples)  # slightly bigger
    overlap_counter = np.zeros(len(y_full) + chunk_length_samples)

    current_pos = 0

    for idx, chunk in enumerate(chunks):
        print(f"Processing chunk {idx+1}/{len(chunks)}")

        # Generate spectrogram
        noisy_spec = generate_spectrogram_from_array(chunk)
        if noisy_spec is None:
            continue

        norm_noisy_spec = normalize_spectrogram(noisy_spec)
        input_spec = pad_to_multiple_of_32(norm_noisy_spec[..., np.newaxis])
        input_spec = np.expand_dims(input_spec, axis=0)

        # Predict denoised spectrogram
        predicted_spec_norm = model.predict(input_spec)[0]

        expected_freq_bins = 1 + N_FFT // 2
        if predicted_spec_norm.shape[0] > expected_freq_bins:
            predicted_spec_norm = predicted_spec_norm[:expected_freq_bins, :, :]

        # Convert spectrogram to audio
        predicted_audio = spectrogram_to_audio(predicted_spec_norm)

        # Apply Hann window to smooth edges
        hann_window = np.hanning(len(predicted_audio))
        predicted_audio *= hann_window

        # Blend into final audio
        end_pos = current_pos + len(predicted_audio)
        final_audio[current_pos:end_pos] += predicted_audio
        overlap_counter[current_pos:end_pos] += 1

        current_pos += hop_samples  # move by 2.5 sec

    # --- Final normalization of overlaps ---
    overlap_counter[overlap_counter == 0] = 1  # prevent division by zero
    final_audio = final_audio / overlap_counter
    final_audio = final_audio[:len(y_full)]  # trim to original length

    # --- Match RMS energy to input audio ---
    def rms(x):
        return np.sqrt(np.mean(np.square(x)))

    input_rms = rms(y_full)
    output_rms = rms(final_audio)

    if output_rms > 0:
        final_audio *= input_rms / output_rms

    # --- Save final output ---
    sf.write(output_file_path, final_audio, sr)
    print(f"Denoised audio saved to: {output_file_path}")




