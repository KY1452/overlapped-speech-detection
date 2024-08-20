import os
import pandas as pd
from typing import List

def combine_part_files(part_files: List[str], output_file: str) -> None:
    combined_df = pd.DataFrame()
    for file in part_files:
        print(f"Reading {file}")
        part_df = pd.read_csv(file)
        combined_df = pd.concat([combined_df, part_df], ignore_index=True)
    
    combined_df.to_csv(output_file, index=False)
    print(f"Combined dataset saved to {output_file}")

# List all part files
part_files = [
    "/home/users/ntu/scsekyad/scratch/OSD/final deployment package/klass-osd-kedro-pipeline/klass-osd/data/02_intermediate/libri2mix_clean_overall_detailed_train_360_part1.csv",
    "/home/users/ntu/scsekyad/scratch/OSD/final deployment package/klass-osd-kedro-pipeline/klass-osd/data/02_intermediate/libri2mix_clean_overall_detailed_train_360_part2.csv",
    "/home/users/ntu/scsekyad/scratch/OSD/final deployment package/klass-osd-kedro-pipeline/klass-osd/data/02_intermediate/libri2mix_clean_overall_detailed_train_360_part3.csv",
    "/home/users/ntu/scsekyad/scratch/OSD/final deployment package/klass-osd-kedro-pipeline/klass-osd/data/02_intermediate/libri2mix_clean_overall_detailed_train_360_part4.csv",
    "/home/users/ntu/scsekyad/scratch/OSD/final deployment package/klass-osd-kedro-pipeline/klass-osd/data/02_intermediate/libri2mix_clean_overall_detailed_train_360_part5.csv",
    "/home/users/ntu/scsekyad/scratch/OSD/final deployment package/klass-osd-kedro-pipeline/klass-osd/data/02_intermediate/libri2mix_clean_overall_detailed_train_360_part6.csv",
    "/home/users/ntu/scsekyad/scratch/OSD/final deployment package/klass-osd-kedro-pipeline/klass-osd/data/02_intermediate/libri2mix_clean_overall_detailed_train_360_part7.csv",
    "/home/users/ntu/scsekyad/scratch/OSD/final deployment package/klass-osd-kedro-pipeline/klass-osd/data/02_intermediate/libri2mix_clean_overall_detailed_train_360_part8.csv",
    "/home/users/ntu/scsekyad/scratch/OSD/final deployment package/klass-osd-kedro-pipeline/klass-osd/data/02_intermediate/libri2mix_clean_overall_detailed_train_360_part9.csv",
    "/home/users/ntu/scsekyad/scratch/OSD/final deployment package/klass-osd-kedro-pipeline/klass-osd/data/02_intermediate/libri2mix_clean_overall_detailed_train_360_part10.csv",
    "/home/users/ntu/scsekyad/scratch/OSD/final deployment package/klass-osd-kedro-pipeline/klass-osd/data/02_intermediate/libri2mix_clean_overall_detailed_train_360_part11.csv",
    "/home/users/ntu/scsekyad/scratch/OSD/final deployment package/klass-osd-kedro-pipeline/klass-osd/data/02_intermediate/libri2mix_clean_overall_detailed_train_360_part12.csv",
    "/home/users/ntu/scsekyad/scratch/OSD/final deployment package/klass-osd-kedro-pipeline/klass-osd/data/02_intermediate/libri2mix_clean_overall_detailed_train_360_part13.csv",
    "/home/users/ntu/scsekyad/scratch/OSD/final deployment package/klass-osd-kedro-pipeline/klass-osd/data/02_intermediate/libri2mix_clean_overall_detailed_train_360_part14.csv",
    "/home/users/ntu/scsekyad/scratch/OSD/final deployment package/klass-osd-kedro-pipeline/klass-osd/data/02_intermediate/libri2mix_clean_overall_detailed_train_360_part15.csv",
    "/home/users/ntu/scsekyad/scratch/OSD/final deployment package/klass-osd-kedro-pipeline/klass-osd/data/02_intermediate/libri2mix_clean_overall_detailed_train_360_part16.csv",
    "/home/users/ntu/scsekyad/scratch/OSD/final deployment package/klass-osd-kedro-pipeline/klass-osd/data/02_intermediate/libri2mix_clean_overall_detailed_train_360_part17.csv",
    "/home/users/ntu/scsekyad/scratch/OSD/final deployment package/klass-osd-kedro-pipeline/klass-osd/data/02_intermediate/libri2mix_clean_overall_detailed_train_360_part18.csv",
    "/home/users/ntu/scsekyad/scratch/OSD/final deployment package/klass-osd-kedro-pipeline/klass-osd/data/02_intermediate/libri2mix_clean_overall_detailed_train_360_part19.csv",
    "/home/users/ntu/scsekyad/scratch/OSD/final deployment package/klass-osd-kedro-pipeline/klass-osd/data/02_intermediate/libri2mix_clean_overall_detailed_train_360_part20.csv",
    "/home/users/ntu/scsekyad/scratch/OSD/final deployment package/klass-osd-kedro-pipeline/klass-osd/data/02_intermediate/libri2mix_clean_overall_detailed_train_360_part21.csv",

    # Add all other part files here
]

# Combine all part files into one final dataset
combine_part_files(part_files, "/home/users/ntu/scsekyad/scratch/OSD/final deployment package/klass-osd-kedro-pipeline/klass-osd/data/02_intermediate/libri2mix_clean_overall_detailed_train_360.csv")
