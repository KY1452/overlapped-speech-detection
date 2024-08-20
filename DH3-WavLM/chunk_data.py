import os
import pickle
import librosa

# Path to the directory containing the .wav files
WAV_DIR = "/home/users/ntu/scsekyad/scratch/raw_data/DH3-WavLM/eval_wav"
# Path to the directory containing the .rttm files
RTTM_DIR = "/home/users/ntu/scsekyad/scratch/raw_data/third_dihard_challenge_eval/data/rttm"
# Output Pickle file for inspection
OUTPUT_PICKLE = "/home/users/ntu/scsekyad/scratch/OSD/final deployment package/klass-osd-kedro-pipeline/klass-osd/data/05_model_input/DH3_eval_dataset.pkl"

def parse_rttm(file_path):
    segments = {}
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            file_id = parts[1]
            start_time = float(parts[3])
            duration = float(parts[4])
            end_time = start_time + duration
            if file_id not in segments:
                segments[file_id] = []
            segments[file_id].append((start_time, end_time))
    return segments

def chunk_audio(y, chunk_size):
    return [y[i:i + chunk_size] for i in range(0, len(y), chunk_size)]

def get_label_for_chunk(start, end, segments):
    speaker_count = 0
    for segment in segments:
        if start < segment[1] and end > segment[0]:  # Overlap condition
            speaker_count += 1
    if speaker_count == 0:
        return 0  # non-speech
    elif speaker_count == 1:
        return 1  # one-speaker-speech
    else:
        return 2  # overlapped-speech

def process_audio_files(wav_directory, segments, chunk_length=0.6, sr=16000):
    data = []
    for wav_file in os.listdir(wav_directory):
        if wav_file.endswith('.wav'):
            file_id = os.path.splitext(wav_file)[0]
            y, _ = librosa.load(os.path.join(wav_directory, wav_file), sr=sr)
            chunked_audio = chunk_audio(y, int(chunk_length * sr))
            for i, chunk in enumerate(chunked_audio):
                start_time = i * chunk_length
                end_time = start_time + chunk_length
                label = get_label_for_chunk(start_time, end_time, segments.get(file_id, []))
                data.append({
                    'audio_id': file_id,
                    'input_values': chunk.tolist(),
                    'chunk_filenames': f"{file_id}_{start_time}_{end_time}",
                    'label': label
                })
    return data

# Parse RTTM files
all_segments = {}
for rttm_file in os.listdir(RTTM_DIR):
    if rttm_file.endswith('.rttm'):
        segments = parse_rttm(os.path.join(RTTM_DIR, rttm_file))
        all_segments.update(segments)

# Process audio files and generate the dataset
dataset = process_audio_files(WAV_DIR, all_segments)

# Save the dataset to a Pickle file for inspection
with open(OUTPUT_PICKLE, 'wb') as f:
    pickle.dump(dataset, f)

print(f"Dataset saved to {OUTPUT_PICKLE}")


# import os
# import json
# import librosa

# # Path to the directory containing the .wav files
# WAV_DIR = "/home/users/ntu/scsekyad/scratch/raw_data/DH3-WavLM/eval_wav"
# # Path to the specific .rttm file
# RTTM_FILE = "/home/users/ntu/scsekyad/scratch/raw_data/third_dihard_challenge_eval/data/rttm/DH_EVAL_0009.rttm"
# # Output JSON file for inspection
# OUTPUT_JSON = "/home/users/ntu/scsekyad/scratch/raw_data/DH3-WavLM/DH_EVAL_0009.json"

# # Time range to extract in seconds
# START_TIME = 0.0
# END_TIME = 6.0
# CHUNK_LENGTH = 0.6
# SR = 16000

# def parse_rttm(file_path):
#     segments = []
#     with open(file_path, 'r') as file:
#         for line in file:
#             parts = line.strip().split()
#             start_time = float(parts[3])
#             duration = float(parts[4])
#             end_time = start_time + duration
#             segments.append((start_time, end_time))
#     return segments

# def chunk_audio(y, start_time, end_time, chunk_length, sr):
#     start_sample = int(start_time * sr)
#     end_sample = int(end_time * sr)
#     y = y[start_sample:end_sample]
#     return [y[i:i + int(chunk_length * sr)] for i in range(0, len(y), int(chunk_length * sr))]

# def get_label_for_chunk(start, end, segments):
#     speaker_count = 0
#     for segment in segments:
#         if start < segment[1] and end > segment[0]:  # Overlap condition
#             speaker_count += 1
#     if speaker_count == 0:
#         return 0  # non-speech
#     elif speaker_count == 1:
#         return 1  # one-speaker-speech
#     else:
#         return 2  # overlapped-speech

# def process_audio_file(wav_directory, rttm_file, start_time, end_time, chunk_length, sr):
#     data = []
#     file_id = os.path.splitext(os.path.basename(rttm_file))[0]
#     wav_file = f"{file_id}.wav"
#     if os.path.exists(os.path.join(wav_directory, wav_file)):
#         y, _ = librosa.load(os.path.join(wav_directory, wav_file), sr=sr)
#         chunked_audio = chunk_audio(y, start_time, end_time, chunk_length, sr)
#         segments = parse_rttm(rttm_file)
#         for i, chunk in enumerate(chunked_audio):
#             chunk_start_time = start_time + i * chunk_length
#             chunk_end_time = chunk_start_time + chunk_length
#             if chunk_end_time > end_time:
#                 break
#             label = get_label_for_chunk(chunk_start_time, chunk_end_time, segments)
#             data.append({
#                 'audio_id': file_id,
#                 'input_values': chunk.tolist(),
#                 'chunk_filenames': f"{file_id}_{chunk_start_time}_{chunk_end_time}",
#                 'label': label
#             })
#     return data

# # Process the specific audio file and generate the dataset
# dataset = process_audio_file(WAV_DIR, RTTM_FILE, START_TIME, END_TIME, CHUNK_LENGTH, SR)

# # Save the dataset to a JSON file for inspection
# with open(OUTPUT_JSON, 'w') as f:
#     json.dump(dataset, f, indent=4)

