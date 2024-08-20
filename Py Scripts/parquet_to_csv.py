import pandas as pd

# Load the Parquet file
parquet_file_path = '/home/users/ntu/scsekyad/scratch/OSD/final deployment package/klass-osd-kedro-pipeline/klass-osd/data/08_reporting/models_wavlm/Libri2Mix/evaluation/DH3_dev_predicted_labels_0.6.parquet'
data = pd.read_parquet(parquet_file_path)

# Convert the DataFrame to CSV
csv_file_path = '/home/users/ntu/scsekyad/scratch/OSD/final deployment package/klass-osd-kedro-pipeline/klass-osd/data/08_reporting/models_wavlm/Libri2Mix/evaluation/DH3_dev_predicted_labels_0.6.csv'
data.to_csv(csv_file_path, index=False)

print(f"CSV file saved to {csv_file_path}")
