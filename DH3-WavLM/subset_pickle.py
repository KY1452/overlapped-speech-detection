import os
import json
import librosa
import pickle

# Path to the directory containing the .wav files
WAV_DIR = "/home/users/ntu/scsekyad/scratch/raw_data/DH3-WavLM/eval_wav"
# Path to the directory containing the .rttm files
RTTM_DIR = "/home/users/ntu/scsekyad/scratch/raw_data/third_dihard_challenge_eval/data/rttm"
# Output pickle file
OUTPUT_PICKLE = "/home/users/ntu/scsekyad/scratch/OSD/final deployment package/klass-osd-kedro-pipeline/klass-osd/data/05_model_input/subset_DH3_pickle.pkl"

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

def process_audio_files(wav_directory, segments, chunk_length=0.6, sr=16000, max_files=5):
    data = []
    file_count = 0
    for wav_file in os.listdir(wav_directory):
        if wav_file.endswith('.wav'):
            file_count += 1
            if file_count > max_files:
                break
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
small_dataset = process_audio_files(WAV_DIR, all_segments)

# Save the dataset to a pickle file
with open(OUTPUT_PICKLE, 'wb') as f:
    pickle.dump(small_dataset, f)