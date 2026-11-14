#importing Libraries

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

# Loading the dataset

df = pd.read_csv("Data/features_3_sec.csv")

X = df.drop(columns=['filename', 'label', 'length'])
y = df['label']

# Label Encoding

le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Training Validating and Testing(70%, 20%, 10%)

# Training and temping(validating and testing)

X_train , X_temp , y_train , y_temp = train_test_split(
    X, y_encoded, test_size = 0.30, random_state = 42, stratify = y_encoded)

# Temping(Validating and Testing)

X_val , X_test , y_val , y_test = train_test_split(
    X_temp , y_temp, test_size = (1/3), random_state = 42, stratify = y_temp)

# Using the randomForestClassifier

rf = RandomForestClassifier(
    n_estimators = 200,
    max_depth = 14,
    random_state = 42,
    n_jobs = -1
    )

rf.fit(X_train, y_train)

# Model Evaluation

Val_pred = rf.predict(X_val)
val_accuracy = accuracy_score(y_val, Val_pred)

print(f"Validation Accuracy : {val_accuracy * 100:.2f} %")

Test_pred = rf.predict(X_test)
test_accuracy = accuracy_score(y_test, Test_pred)

print(f"Testing Accuracy : {test_accuracy * 100:.2f}")

# Classfication Report

print(classification_report(y_test, Test_pred, target_names = le.classes_))

# Storing Trained Model and LabelEncoder

joblib.dump(rf, 'rf_genre_model.pkl')
joblib.dump(le, 'label_encoder.pkl')