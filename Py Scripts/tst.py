import pandas as pd

file_path = "/home/users/ntu/scsekyad/scratch/OSD/final deployment package/klass-osd-kedro-pipeline/klass-osd/data/02_intermediate/Libri2Mix/wav16k/max/metadata/train_360_clean_part21_2mix_osd_labels.parquet.gzip"
data = pd.read_parquet(file_path)
print(data.head())
print(data.info())
