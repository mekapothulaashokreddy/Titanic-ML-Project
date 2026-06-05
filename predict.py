import pandas as pd
import joblib

model = joblib.load("model/titanic_model.pkl")

def predict_survival(pclass, sex, age, fare):
    data = pd.DataFrame([{
        "pclass": pclass,
        "sex": sex,
        "age": age,
        "fare": fare
    }])
    
    result = model.predict(data)
    
    return "Survived" if result[0] == 1 else "Did not survive"

print(predict_survival(1, 1, 25, 100))