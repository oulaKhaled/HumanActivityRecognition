# Human Activity Recognition using LSTM

This project focuses on **Human Activity Recognition (HAR)** using data collected from smartphone sensors (accelerometer and gyroscope).  
The model predicts **6 different human activities** (such as walking, sitting, and standing) based on **time-series motion data**.

---
Human Activity Recognition (HAR) aims to automatically identify physical activities from wearable sensors.  
In this project, we use the [UCI HAR Dataset](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones), which provides triaxial acceleration and angular velocity data recorded from smartphones.

Each sample consists of:
- **128 time steps**
- **9 features** (3 acceleration axes, 3 gyroscope axes, 3 total acceleration axes)

The goal is to classify each sequence into one of six activities.

---
Project Structure
```
src/  
project/
│
├── data/                 # Contains training and testing datasets
├── src/
│   ├── preprocess.py     # Data preprocessing functions
│   ├── model.py          # Model creation function
│   ├── train.py          # Script to train and evaluate the model
│   ├── utils.py
│   ├── visualize.py       # Visualize Data and model results
│
├── requirements.txt
├── .gitignore
└── README.md
```

### Model Architecture

The model is built using **Long Short-Term Memory (LSTM)** layers to capture _temporal dependencies_ in sensor data.  
All model hyperparameters, including the number of units, dropout rate, and learning rate, were **tuned using KerasTuner** to achieve optimal performance.  

**Model Highlights:**  
- Two stacked LSTM layers  
- Optional Dropout and Batch Normalization for regularization  
- Adam optimizer with tuned learning rate  
- Softmax output layer for multiclass classification  
