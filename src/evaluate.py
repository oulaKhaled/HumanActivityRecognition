from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
)
import pandas as pd
import numpy as np
from train import model, x_test, y_test

## you should call model from saved_model folder


def get_prediction(model, x_test):
    predict = model.predict(x_test)
    predict = np.argmax(predict, axis=1)
    predict = pd.get_dummies(predict).to_numpy()
    return predict


prediction = get_prediction(model, x_test)
score = accuracy_score(prediction, y_test)
matrix = confusion_matrix(prediction, y_test)
report = classification_report(prediction, y_test)
_f1_score = f1_score(prediction, y_test)

print(f"Accuracy Score : {score} \n")
print(f"Confusion Matrix : {matrix} \n")
print(f"Classification report : {report} \n")
print(f"F1 Score : {_f1_score} \n")
