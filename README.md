# 🧠 Customer Churn Prediction using Machine Learning

This project applies **Machine Learning models** to predict whether customers will **exit a service** based on given features. The dataset used is **P1_Churn_Modelling.csv**.

## 📌 Features & Workflow
The pipeline follows these key steps:
1. **Data Preprocessing**:
   - Remove unnecessary columns (`RowNumber`, `CustomerId`, `Surname`).
   - Encode categorical variables (`Gender`, `Geography`).
   - Handle class imbalance using **SMOTE**.

2. **Exploratory Data Analysis (EDA)**:
   - Visualize distributions (age, credit score, balance).
   - Analyze **correlation heatmaps** & class distributions.

3. **Model Training & Evaluation**:
   - Train different **ML models**.
   - Compare results using **classification reports, accuracy, and confusion matrices**.

4. **Model Comparison**:
   - Plot accuracy scores of all models to identify the best performer.

---

## 🚀 Models Used
The following models were tested:
- **K-Nearest Neighbors (KNN)**
- **Naive Bayes (GaussianNB)**
- **Support Vector Machine (SVM)**
- **Decision Tree Classifier**

| Model              | Accuracy Score |
|--------------------|---------------|
| KNN Model         | **87% (Train), 83% (Test)** |
| Naive Bayes       | **Moderate Performance (Stable but Lower Accuracy)** |
| SVM Model         | **Best Accuracy (Highest Performance)** |
| Decision Tree     | **Overfitting (100% Train, 80% Test)** |

📊 **SVM achieved the best accuracy among all models.**

---

## ⚙️ How to Run the Project
### **🔹 Prerequisites**
Ensure you have Python installed and the required libraries:
```bash
pip install numpy pandas matplotlib seaborn plotly scikit-learn imbalanced-learn
