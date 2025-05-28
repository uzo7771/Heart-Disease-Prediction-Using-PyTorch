# 🫀 Heart Disease Prediction Using PyTorch

This project implements a neural network in PyTorch to predict the presence of heart disease using the Cleveland dataset. It includes data loading, preprocessing, training, and evaluation.

## 📂 Dataset

The dataset used is the **Cleveland Heart Disease dataset**, available [here](https://www.kaggle.com/datasets/ritwikb3/heart-disease-cleveland).

- File: `heart_cleveland_upload.csv`
- Contains 13 features and 1 binary target column (`0` = no heart disease, `1` = presence of heart disease)

## 🧠 Model Architecture

A fully connected feedforward neural network built with PyTorch:
- `Linear(13 → 64)` → `Tanh` → `Dropout`
- `Linear(64 → 32)` → `Tanh` → `Dropout`
- `Linear(32 → 16)` → `Tanh` → `Dropout`
- `Linear(16 → 1)` → `Sigmoid`

The model outputs probabilities interpreted as binary class predictions using a threshold of 0.5.

## ⚙️ Training Configuration

- **Loss Function**: Binary Cross Entropy (`BCELoss`)
- **Optimizer**: Adam
- **Learning Rate**: 0.001
- **Dropout**: 0.5
- **Batch Size**: 25
- **Epochs**: 500
- **Train/Test Split**: 80% / 20%
- **Accuracy Metric**: Proportion of correct predictions (using threshold 0.5)

## 📈 Example Output

Training and testing accuracy is printed every 50 epochs. Example:

```
Epoch: [500/500], Loss: 0.2614, Training Accuracy: 0.8987, Testing Accuracy: 0.8833
```
