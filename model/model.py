import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import pickle
import os



data_path = "../dataset/cancer patient data sets.csv"

df = pd.read_csv(data_path)

print("Dataset Loaded Successfully")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nDataset Information:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())


df = df.drop(columns=["index", "Patient Id"])



label_encoder = LabelEncoder()

df["Level"] = label_encoder.fit_transform(df["Level"])

print("\nEncoded Target Values:")
print(df["Level"].value_counts())



X = df.drop("Level", axis=1)

y = df["Level"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)



model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)



model.fit(X_train, y_train)

y_pred = model.predict(X_test)



accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:")
print(accuracy)

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=label_encoder.classes_
))


cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=label_encoder.classes_,
    yticklabels=label_encoder.classes_
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.show()




model_path = "cancer_model.pkl"

with open(model_path, "wb") as file:
    pickle.dump(model, file)



encoder_path = "label_encoder.pkl"

with open(encoder_path, "wb") as file:
    pickle.dump(label_encoder, file)


print("\nModel Saved Successfully!")
print("cancer_model.pkl created")
print("label_encoder.pkl created")