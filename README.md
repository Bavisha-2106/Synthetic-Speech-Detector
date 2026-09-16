# Synthetic Speech Detector

Deep Learning-Based Synthetic Speech Detection Using Mel-Spectrograms.

## Description

A CNN-based system that classifies speech audio as BONAFIDE
(human speech) or SPOOF (synthetic/manipulated speech).

## Technologies

- Python
- PyTorch
- Librosa
- FastAPI
- HTML/CSS/JavaScript

## Model

Input: Mel-Spectrogram
Model: Convolutional Neural Network
Classes: BONAFIDE / SPOOF

## Results

Validation Accuracy: 81.45%

## How to Run

Install dependencies:

```pip install -r requirements.txt```

Run backend:

```uvicorn backend:app --reload```

Open the web interface and upload or record an audio sample.
