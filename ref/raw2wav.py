import os
import numpy as np
from scipy.io import wavfile


def raw_to_wav(raw_file_path, wav_file_path, sample_rate=44100, num_channels=1, bit_depth=16):
    # Read raw file data
    with open(raw_file_path, 'rb') as raw_file:
        raw_data = raw_file.read()

    # Convert raw data to numpy array
    if bit_depth == 16:
        dtype = np.int16
    elif bit_depth == 8:
        dtype = np.int8
    else:
        raise ValueError("Unsupported bit depth: {}".format(bit_depth))

    audio_data = np.frombuffer(raw_data, dtype=dtype)

    # Reshape the audio data according to the number of channels
    if num_channels > 1:
        audio_data = audio_data.reshape(-1, num_channels)

    # Write data to WAV file
    wavfile.write(wav_file_path, sample_rate, audio_data)


def convert_all_raw_to_wav(input_folder, output_folder, sample_rate=44100, num_channels=1, bit_depth=16):
    # Walk through the directory tree
    for root, _, files in os.walk(input_folder):
        for filename in files:
            if filename.endswith(".raw"):
                raw_file_path = os.path.join(root, filename)
                relative_path = os.path.relpath(root, input_folder)
                wav_output_dir = os.path.join(output_folder, relative_path)

                # Ensure the output directory exists
                os.makedirs(wav_output_dir, exist_ok=True)

                wav_file_name = os.path.splitext(filename)[0] + '.wav'
                wav_file_path = os.path.join(wav_output_dir, wav_file_name)

                raw_to_wav(raw_file_path, wav_file_path, sample_rate, num_channels, bit_depth)
                print(f"Converted {raw_file_path} to {wav_file_path}")


# Usage example
input_folder = '/Users/taewonkim/Develop/Python/LectureMachineLearning/kim/202401ml_fmcc/raw16k'
output_folder = '/Users/taewonkim/Develop/Python/LectureMachineLearning/kim/202401ml_fmcc/wav16k'
sample_rate = 16000  # Set the sample rate of your raw audio
num_channels = 1  # Set the number of channels
bit_depth = 16  # Set the bit depth of your raw audio

convert_all_raw_to_wav(input_folder, output_folder, sample_rate, num_channels, bit_depth)