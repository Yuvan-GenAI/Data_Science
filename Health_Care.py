import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


data = pd.read_csv("heart_attack_prediction_dataset.csv.zip_dataset.csv")
X = data.drop("revisit", axis=1)
y = data["revisit"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
model = RandomForestClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
new_patient = [[55, 120, 80, 200, 1]]
new_patient = scaler.transform(new_patient)
prediction = model.predict(new_patient)

if prediction[0] == 1:
    print("Visit Needed")
else:
    print("No Visit Needed")
