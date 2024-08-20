import os

def parse_rttm(rttm_file):
    segments = []
    with open(rttm_file, 'r') as file:
        lines = file.readlines()
        for line in lines:
            parts = line.strip().split()
            start_time = float(parts[3])
            duration = float(parts[4])
            speaker = parts[7]
            end_time = start_time + duration
            segments.append((start_time, end_time, speaker))
    return segments

if __name__ == "__main__":
    rttm_directory = '/home/users/ntu/scsekyad/scratch/raw_data/third_dihard_challenge_dev/data/rttm'
    rttm_files = [os.path.join(rttm_directory, file) for file in os.listdir(rttm_directory) if file.endswith('.rttm')]

    all_segments = {}
    for rttm_file in rttm_files:
        file_id = os.path.splitext(os.path.basename(rttm_file))[0]
        all_segments[file_id] = parse_rttm(rttm_file)
    
    # Save the parsed segments to a file for later use
    import pickle
    with open('parsed_segments_dev.pkl', 'wb') as f:
        pickle.dump(all_segments, f)
