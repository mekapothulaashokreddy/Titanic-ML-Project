# Titanic Survival Prediction 🚢

## 📌 Problem
Predict whether a passenger survived the Titanic disaster using machine learning.

## 📊 Features Used
- Passenger Class (pclass)
- Gender (sex)
- Age
- Fare
- Family Size
- Is Alone

## ⚙️ Model Used
- XGBoost Classifier
- Cross-validation for evaluation

## 📈 Performance
- Accuracy: ~82%
- Cross-validation Accuracy: ~84%

## 🚀 Key Improvements
- Feature engineering (family_size, is_alone)
- Model tuning (XGBoost)
- Handling missing values

## ▶️ How to Run
```bash
python train.py
python predict.py