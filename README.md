# Real-Time Fraud Detection using Hybrid CNN-GRU

An end-to-end deep learning solution for credit card fraud detection utilizing a hybrid **CNN-GRU** architecture trained on the Kaggle IEEE-CIS Fraud Detection dataset. Includes a real-time interactive Streamlit web app for transaction simulation.

**Live Application:** [Streamlit Real-Time Demo](https://cnn-gru-real-time-simulation.streamlit.app/)

**Data Demo:** [Google Drive Storage](https://drive.google.com/drive/folders/1KQl52ONwLQ6ujSevkayZ3_zg1VtjVvGw?usp=sharing)

---

## Project Overview

Detecting financial fraud in high-volume transaction stream data presents severe challenges due to extreme class imbalance and complex non-linear feature correlations. This project implements a hybrid **CNN + GRU** pipeline
---

## 📊 Dataset

* **Source:** [Kaggle IEEE-CIS Fraud Detection Dataset](https://www.kaggle.com/c/ieee-fraud-detection)
* **Components:**
* `train_transaction.csv` & `train_identity.csv`
* `test_transaction.csv` & `test_identity.csv`


* **Target:** `isFraud` (Binary classification: `0` for Legitimate, `1` for Fraudulent).
