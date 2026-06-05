# ❤️ Heart Disease Prediction using Machine Learning and Ensemble Techniques

## 📌 Overview

Heart Disease Prediction using Machine Learning and Ensemble Techniques is an end-to-end healthcare analytics project developed using the Cleveland Heart Disease Dataset. The project aims to predict the presence of heart disease based on patient clinical attributes using multiple machine learning algorithms, ensemble learning, and hybrid ensemble techniques.

The system compares the performance of various machine learning models and identifies the most reliable approach for heart disease prediction. A web-deployable solution was also developed using R Plumber API.

---

## 🎯 Objectives

* Predict the presence of heart disease using patient health records.
* Perform Exploratory Data Analysis (EDA) to identify patterns and trends.
* Compare multiple machine learning algorithms.
* Improve prediction accuracy using Ensemble and Hybrid Ensemble techniques.
* Deploy the final model using R Plumber API for real-world usability.

---

## 📊 Dataset

**Dataset Used:** Cleveland Heart Disease Dataset

### Dataset Information

* Approximately 303 patient records
* 14 attributes
* Binary Classification Problem

### Target Variable

* 0 → No Heart Disease
* 1 → Heart Disease Present

### Features

* Age
* Sex
* Chest Pain Type
* Resting Blood Pressure
* Cholesterol
* Fasting Blood Sugar
* Resting ECG
* Maximum Heart Rate
* Exercise-Induced Angina
* ST Depression (Oldpeak)
* Slope
* Number of Major Vessels (CA)
* Thalassemia

---

## 🔄 Project Workflow

1. Data Collection
2. Data Preprocessing
3. Missing Value Analysis
4. Feature Engineering
5. Exploratory Data Analysis (EDA)
6. Train-Test Split (70:30)
7. Model Training
8. Model Evaluation
9. Ensemble Learning
10. Hybrid Ensemble Development
11. API Deployment using R Plumber

---

## 🧹 Data Preprocessing

The following preprocessing steps were performed:

* Missing value checking
* Data cleaning
* Conversion of categorical variables into factors
* Feature scaling where required
* Train-Test Split (70% Training, 30% Testing)

---

## 📈 Exploratory Data Analysis

EDA was conducted to understand:

* Target class distribution
* Age distribution
* Cholesterol trends
* Feature relationships
* Correlation between variables

Visualizations include:

* Histograms
* Bar Charts
* Box Plots
* Correlation Analysis
* ROC Curves

---

## 🤖 Machine Learning Models Implemented

### Baseline Models

* Logistic Regression
* K-Nearest Neighbors (KNN)
* Decision Tree
* Random Forest
* Support Vector Machine (SVM)

### Advanced Models

* Naive Bayes
* Neural Network
* XGBoost
* Gradient Boosting Machine (GBM)
* AdaBoost

### Ensemble Models

* Voting Ensemble
* Hybrid Ensemble Model

---

## 📊 Model Evaluation Metrics

The models were evaluated using:

* Accuracy
* Confusion Matrix
* Sensitivity
* Specificity
* ROC Curve Analysis

---

## 🏆 Results

| Model               | Accuracy |
| ------------------- | -------- |
| Logistic Regression | ~82%     |
| Random Forest       | ~82%     |
| Naive Bayes         | ~83.7%   |
| Neural Network      | ~79%     |
| XGBoost             | ~74%     |
| Gradient Boosting   | ~78-82%  |
| AdaBoost            | ~78-81%  |
| Ensemble Model      | ~82-84%  |
| Hybrid Ensemble     | ~83-85%  |

### Best Performing Model

**Hybrid Ensemble Model**

The Hybrid Ensemble model achieved the highest overall performance by combining the strengths of multiple machine learning algorithms.

---

## 🛠️ Technologies Used

### Programming Language

* R

### Development Environment

* RStudio

### Libraries

* caret
* e1071
* randomForest
* rpart
* nnet
* xgboost
* gbm
* adabag
* pROC
* ggplot2

### Deployment

* R Plumber API

### Frontend

* HTML
* CSS
* Bootstrap

---

## 📂 Repository Structure

```text
├── Dataset
├── Source Code
├── Reports
├── Presentation
├── Images
├── API Deployment Files
└── README.md
```

---

## 🚀 Future Enhancements

* Hyperparameter Optimization
* Deep Learning Approaches
* Larger Healthcare Datasets
* Real-Time Prediction System
* Cloud Deployment
* Hospital Integration

---

## 👨‍💻 Authors

### Evan Joe Nelson G B

M.Tech Integrated Computer Science with Business Analytics
Vellore Institute of Technology (VIT), Chennai

### Shyam Kumar M

Vellore Institute of Technology (VIT), Chennai

---

## 📜 License

This project is developed for academic and educational purposes.

---

## ⭐ Acknowledgements

* Cleveland Heart Disease Dataset
* Vellore Institute of Technology (VIT), Chennai
* Programming for Data Science Course
* Open-source R Community
