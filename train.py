# import pandas as pd
# import seaborn as sns
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# import joblib

# # Load dataset
# df = sns.load_dataset("titanic")

# # Select features
# df = df[["survived", "pclass", "sex", "age", "fare"]]

# # Handle missing values
# df["age"] = df["age"].fillna(df["age"].mean())

# # Encode categorical
# df["sex"] = df["sex"].map({"male": 0, "female": 1})

# # Split
# X = df.drop("survived", axis=1)
# y = df["survived"]

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# # Train model
# model = LogisticRegression()
# model.fit(X_train, y_train)

# # Save model
# joblib.dump(model, "model/titanic_model.pkl")

# print("Model trained and saved!")
# pip install xgboost
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import joblib

# Load dataset
df = sns.load_dataset("titanic")

# Select more features
df = df[["survived", "pclass", "sex", "age", "fare", "sibsp", "parch"]]

# Handle missing values
df["age"] = df["age"].fillna(df["age"].median())

# Convert categorical
df["sex"] = df["sex"].map({"male": 0, "female": 1})


# Feature engineering (IMPORTANT)
df["family_size"] = df["sibsp"] + df["parch"] 

# Drop old columns
df = df.drop(["sibsp", "parch"], axis=1)

# Split features and target
X = df.drop("survived", axis=1)
y = df["survived"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = XGBClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=4,
    random_state=42
)
df = df.dropna()
# Train
model.fit(X_train, y_train)

# Accuracy check
from sklearn.metrics import accuracy_score
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Save model
joblib.dump(model, "model/titanic_model.pkl")
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X, y, cv=5)
print("Cross Validation Accuracy:", scores.mean())

print("Model trained and saved!")