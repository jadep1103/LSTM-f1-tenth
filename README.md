# F1-Tenth Biosignal Control System 🏎️🤖

A biomedical engineering project for controlling a 1/10 scale F1 car using multi-modal biosignals captured through the Sifilab Biopoint wearable sensor. This system leverages EMG, IMU, and other physiological signals, I focused on implementing an LSTM neural network to classify biosignal patterns into precise vehicle control commands.

## 🚧 Project Status

**Currently in standby - Development paused but not finished**

This project represents my contribution to the Université Laval AI Club's winter 2025 initiative. While the core LSTM model and multi-modal biosignal processing pipeline are functional, the project is currently on hold and requires further development for full deployment and integration within the F1-tenth.

## 🎯 Project Overview

**University**: Université Laval AI Club (Winter 2025)  
**Objective**: Control a 1/10 scale F1 car replica using physiological signals and hand movements  
**Technology**: Sifilab Biopoint multi-sensor wearable + LibEMG library  

### 🔬 Biopoint Sensor Overview
The Sifilab Biopoint is a comprehensive wearable biosensor featuring EMG, 6-axis IMU, ECG, PPG, EDA/GSR, and skin temperature sensors in a smartwatch form factor with 4GB storage and full-day battery life.

## My Contribution: LSTM-Based Pattern Recognition

This repository contains my specific work on the neural network component of the project:

### ✅ What I Implemented
- **Custom LSTM Architecture** - 2-layer LSTM with 128 hidden units for sequence classification
- **Time-Series Data Processing** - Sliding window approach (50 timesteps) for temporal pattern capture
- **Custom PyTorch Dataset** - IMUDataset class for efficient data loading and preprocessing
- **Training Infrastructure** - Complete training loop with weighted loss and class balancing
- **Model Evaluation System** - Comprehensive metrics, confusion matrix, and performance analysis
- **Multi-Class Classification** - 5-class gesture classification system
- **Model Persistence** - Automatic model saving with timestamps for version control

## 📊 Technical Specifications

### LSTM Model Architecture
- **Input Layer**: 7-dimensional sensor data from csv file (ax, ay, az, qx, qy, qz, qw from IMU)
- **Sequence Length**: 50 timesteps sliding window for temporal context
- **LSTM Layers**: 2-layer LSTM with 128 hidden units each
- **Output Layer**: Fully connected layer for 5-class classification
- **Activation**: Final softmax for probability distribution over gesture classes
- **Framework**: PyTorch with custom forward pass implementation

### Dataset Processing
- **125,000+ data points** from sensor recordings
- **7-dimensional features**: IMU sensor data (accelerometer + quaternion)
- **Sliding window approach**: 50-sample sequences for temporal learning
- **Data normalization**: Z-score standardization for stable training
- **Class balancing**: Weighted CrossEntropyLoss to handle class imbalance
- **Train/test split**: 70/30 split with random sampling for robust evaluation

## 🚀 Quick Start

### Prerequisites
```bash
pip install torch pandas matplotlib scikit-learn
# LibEMG for comprehensive biosignal processing
git clone https://github.com/LibEMG/libemg.git
# Biopoint SDK and drivers 
```

### Training the Model
```bash
python train.py
```

### Testing and Evaluation
```bash
python testexamples.py
```

## 📁 Project Structure

```
lstm/
├── models/
│   └── model.py              # LSTM model architecture
├── dataset/
│   └── imudataset.py         # Custom PyTorch dataset for IMU data
├── rawdata/
│   └── imu_data.csv          # Training data (125k+ samples)
├── train.py                  # Model training script
├── testexamples.py           # Model evaluation and metrics
├── checkdata.py              # Data analysis and preprocessing
└── *.pth                     # Trained model checkpoints
```

## Classification System

**5-class gesture classification** for vehicle control commands using IMU sensor patterns from hand movements.

## 📈 Performance Metrics

- **Training Loss**: Optimized with weighted CrossEntropyLoss
- **Test Accuracy**: Evaluated on held-out dataset
- **F1-Score**: Weighted average for multi-class performance
- **Confusion Matrix**: Detailed classification analysis

##  Remaining Work & Future Enhancements

### Next Steps 
- [ ] Real-time multi-modal biosignal processing pipeline
- [ ] Full Biopoint device integration with F1-tenth car
- [ ] Live physiological-pattern-to-control command mapping
- [ ] Multi-sensor fusion algorithm implementation (ECG, PPG, EDA integration)
- [ ] System integration testing and validation
- [ ] Performance optimization for real-time multi-modal signal processing

## 📝 Note

This repository contains only the IMU processing and  classification component of the Université Laval AI Club's F1-tenth project. The complete autonomous racing system includes additional components that remain private as they involve collaborative university work.

## 🔗 References

- **Sifilab Biopoint**: https://sifilabs.com/biopoint/
- **LibEMG Library**: https://github.com/LibEMG/libemg.git
- **Université Laval AI Club**: Winter 2025 Project

## 🛠️ Technologies Used

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red?style=flat-square&logo=pytorch)
![Pandas](https://img.shields.io/badge/Pandas-1.3+-green?style=flat-square&logo=pandas)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange?style=flat-square&logo=scikit-learn)

---
