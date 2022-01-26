import pickle
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

with open("data/data_bow.pkl", "rb") as f:
    x_train = pickle.load(f)

Y = x_train[:, 0]
X = x_train[:, 1:]

model = XGBClassifier(objective="multi:softmax",
                      learning_rate=0.3)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=2022)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(accuracy_score(y_pred, y_test))
