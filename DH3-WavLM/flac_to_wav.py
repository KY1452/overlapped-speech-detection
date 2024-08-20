from pydub import AudioSegment
import os

def convert_flac_to_wav(flac_dir, wav_dir):
    if not os.path.exists(wav_dir):
        os.makedirs(wav_dir)
    for flac_file in os.listdir(flac_dir):
        if flac_file.endswith('.flac'):
            wav_file = os.path.splitext(flac_file)[0] + '.wav'
            audio = AudioSegment.from_file(os.path.join(flac_dir, flac_file))
            audio.export(os.path.join(wav_dir, wav_file), format='wav')

flac_directory = '/home/users/ntu/scsekyad/scratch/raw_data/third_dihard_challenge_dev/data/flac'
wav_directory = '/home/users/ntu/scsekyad/scratch/raw_data/DH3-WavLM/dev_wav'
convert_flac_to_wav(flac_directory, wav_directory)
