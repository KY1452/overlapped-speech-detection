# OverlappedSpeechDetection


---

# Speaker Counting using WavLM

This repository contains the code and documentation for a final year thesis project (by Kanishk Yadav) that focuses on **speaker counting** using deep learning techniques. The project implements various models and experiments with different datasets to improve the accuracy of detecting the number of speakers (0, 1, >=2) in overlapping speech segments.

## Table of Contents

- [Introduction](#introduction)
- [Datasets](#datasets)
- [Models](#models)
- [Pipeline](#pipeline)
- [Results](#results)

## Introduction

Speaker counting is an essential task in speaker diarization and other audio processing applications. This project implements a speaker counting pipeline using the WavLM model, a transformer-based architecture that captures sequential information effectively. The model has been trained and fine-tuned on datasets like LibriSpeech and DIHARD III to improve its performance in detecting overlapping speech and counting the number of speakers.

## Datasets

The following datasets were used in this project:

- **Libri2Mix**: Mixtures generated from two random clean audio files from the LibriSpeech Dataset. These are mono-channel audio files sampled at 16 kHz.
- **SparseLibriMix**: A variation of Libri2Mix with different levels of overlap and optional noise addition.
- **DIHARD III**: A challenging dataset modified into a Hugging Face object for use with the WavLM model.

## Models

### WavLM Model

- **Architecture**: Transformer-based with a gated relative position bias for better sequential information capture.
- **Training**: The model was trained on a large-scale dataset of 94k hours, including data from Libri-Light, GigaSpeech, and VoxPopuli.
- **Fine-tuning**: The model was fine-tuned on LibriSpeech and DIHARD III datasets, with various experiments conducted to optimize performance.

<img width="1207" alt="WavLM Architecture" src="https://github.com/user-attachments/assets/a594bd80-e061-40a5-9a9c-cec13b665c8a">


## Pipeline

The overall pipeline involves:

1. **Pre-processing**: Preparing the audio data, segmenting it into chunks, and converting it into a format suitable for the model.
2. **Model Training**: Training the WavLM model on the training datasets.
3. **Inference**: Running the model on test data to predict the number of speakers and evaluate the performance.

<img width="871" alt="OSD(AISG) Pipeline" src="https://github.com/user-attachments/assets/873d40f1-3f30-46fb-a4c6-0cb435b6e748">


## Results

The results are summarized as follows:

- The WavLM model trained on the LibriSpeech dataset achieved an accuracy of 91% on the Libri2Mix test set.
- Fine-tuning on DIHARD III improved the model's performance on challenging data, achieving an accuracy of 87%.
- Sequential fine-tuning on combined datasets revealed variations in performance, with ongoing efforts to optimize this process.

<img width="955" alt="OSD_Results_KanishkYadav" src="https://github.com/user-attachments/assets/8b3c0586-c894-4e0a-ad09-d1696b2d07d1">

For detailed performance metrics, including precision, recall, and F1-score, please refer to the 'Results Reports' and 'Speaker Counting.pdf' in the repository.





