import pandas as pd
from preprocessing import preprocessing
from modeling import learning_model
from catboost import CatBoostClassifier
from metrics import show_metrics

df = pd.read_csv('cs-training.csv')
df = preprocessing(df)
X_test, y_test = learning_model(df)
model = CatBoostClassifier()
model.load_model('catboost_model.cbm')
print("\n9. Evaluating Model")
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]
show_metrics(y_test, y_pred, y_pred_proba)


