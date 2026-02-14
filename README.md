# Tourism Wellness Package - Purchase Prediction (MLOps Project)

An end-to-end MLOps pipeline for predicting whether a customer will purchase a Wellness Tourism Package, using Hugging Face Hub for data/model registry, GitHub Actions for CI/CD, MLflow for experiment tracking, and Streamlit for deployment.

## Project Structure

```
tourism-mlops-project/
├── .github/
│   └── workflows/
│       └── pipeline.yml          # GitHub Actions CI/CD pipeline
├── tourism_project/
│   ├── data/
│   │   └── tourism.csv           # Raw dataset
│   ├── model_building/
│   │   ├── register_data.py      # Upload dataset to HF Hub
│   │   ├── data_preparation.py   # Clean, encode, split data
│   │   ├── model_training.py     # Train, tune, evaluate, register model
│   │   └── deploy_to_hf.py       # Deploy app to HF Space
│   ├── deployment/
│   │   ├── Dockerfile            # Docker config for HF Space
│   │   ├── app.py                # Streamlit web application
│   │   └── requirements.txt      # App dependencies
│   └── requirements_pipeline.txt # Pipeline dependencies
├── .gitignore
├── LICENSE
└── README.md
```

## Pipeline Overview

The GitHub Actions pipeline has 4 sequential jobs:

1. **register-dataset** → Uploads `tourism.csv` to Hugging Face Dataset Hub
2. **data-prep** → Cleans data, encodes features, splits into train/test, uploads to HF
3. **model-training** → Trains Gradient Boosting model with GridSearchCV, logs to MLflow, registers model on HF
4. **deploy-hosting** → Pushes Dockerfile + app.py + requirements.txt to HF Space

## Tech Stack

- **ML Model**: Gradient Boosting Classifier (scikit-learn)
- **Hyperparameter Tuning**: GridSearchCV
- **Experiment Tracking**: MLflow
- **Data & Model Registry**: Hugging Face Hub
- **Deployment**: Streamlit on Hugging Face Spaces (Docker)
- **CI/CD**: GitHub Actions

## Setup

1. Add `HF_TOKEN` (Hugging Face Write token) as a GitHub Secret
2. Push code to `main` branch
3. GitHub Actions pipeline triggers automatically

## Live App

🚀 [Tourism Package Predictor](https://huggingface.co/spaces/Matheshrangasamy/tourism-app)
