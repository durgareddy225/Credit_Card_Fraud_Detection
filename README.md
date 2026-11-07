# Big Data-Driven Credit Card Fraud Detection Using PySpark MLlib

## 📌 Project Overview

This project focuses on detecting fraudulent credit card transactions using **Apache PySpark and Machine Learning**. It uses distributed data processing techniques to analyze transaction records, identify suspicious patterns, and classify transactions as legitimate or fraudulent.

Three machine learning algorithms Logistic Regression, Decision Tree, and Random Forest are implemented using Spark MLlib. Their performance is evaluated and compared to identify the most suitable classification model.

## 🎯 Objectives

- Analyze credit card transaction data and identify fraudulent transaction patterns.
- Clean and preprocess the dataset by handling missing values and duplicate records.
- Implement Logistic Regression, Decision Tree, and Random Forest classification algorithms.
- Evaluate model performance and visualize results to identify the best-performing model.

## 🛠️ Technologies Used

- Python
- Apache Spark
- PySpark
- Spark MLlib
- Pandas
- Matplotlib
- Seaborn
- Java JDK

## 📂 Dataset Description

The project uses a credit card transaction dataset obtained from Kaggle.

- **Dataset:** Credit Card Fraud Transaction Data
- **Records:** Approximately 100,000
- **Attributes:** 16
- **Legitimate Transactions:** 92,805
- **Fraudulent Transactions:** 7,195
- **Target Column:** Fraud

**Dataset Source:** [Kaggle – Credit Card Fraud Transaction Data](https://www.kaggle.com/datasets/anurag629/credit-card-fraud-transaction-data)

## ⚙️ Project Workflow

1. **Dataset Loading:** Load transaction data into a PySpark DataFrame.
2. **Data Exploration:** Examine dataset structure, attributes, missing values, and fraud distribution.
3. **Data Preprocessing:** Handle missing values, clean transaction amounts, extract date-related features, and remove duplicate records.
4. **Feature Engineering:** Apply StringIndexer, OneHotEncoder, and VectorAssembler.
5. **Model Training:** Train three machine learning classification models using Spark MLlib.
6. **Model Evaluation:** Compare model performance using evaluation metrics and confusion matrices.
7. **Visualization:** Generate charts to understand transaction patterns and model performance.
8. **Model Selection:** Select the best-performing model based on weighted F1 Score.

## 🤖 Machine Learning Algorithms

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier

## 📊 Evaluation Metrics

The models are evaluated using:

- Accuracy
- Weighted Precision
- Weighted Recall
- Weighted F1 Score
- ROC-AUC
- Confusion Matrix

## 📈 Results

| Algorithm | Accuracy | Weighted F1 Score | ROC-AUC |
|---|---:|---:|---:|
| Logistic Regression | 97.19% | 97.11% | 97.77% |
| Decision Tree | 97.59% | 97.40% | 94.59% |
| Random Forest | 96.07% | 95.38% | 98.66% |

### 🏆 Best Performing Model: Decision Tree

The Decision Tree classifier achieved the highest weighted F1 Score of **97.40%** and an accuracy of **97.59%**.

## 📉 Visualizations

The project generates the following visualizations:

- Credit Card Fraud Distribution
- Transaction Amount by Fraud Status
- Fraud Distribution by Entry Mode
- Logistic Regression Confusion Matrix
- Decision Tree Confusion Matrix
- Random Forest Confusion Matrix
- Machine Learning Model Performance Comparison

## 📁 Project Structure

```text
Credit-Card-Fraud-Detection/
│
├── Dataset/
│   └── CreditCardData.csv
│
├── Visualizations/
│   ├── 01_fraud_distribution.png
│   ├── 02_transaction_amount.png
│   ├── 03_fraud_by_entry_mode.png
│   ├── logistic_regression_confusion_matrix.png
│   ├── decision_tree_confusion_matrix.png
│   ├── random_forest_confusion_matrix.png
│   └── 04_model_performance.png
│
├── Credit_Card_Fraud_Detection.py
└── README.md

```

## 📝 Conclusion

This project demonstrates the application of Apache PySpark and machine learning techniques for credit card fraud detection. By comparing Logistic Regression, Decision Tree, and Random Forest, the project identifies Decision Tree as the best-performing model based on weighted F1 Score. The visualizations and evaluation results provide insights into transaction patterns and classification performance.
