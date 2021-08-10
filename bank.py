import csv 
import pandas as pd
import numpy as np
import plotly_express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import random
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler 
from sklearn.metrics import confusion_matrix
import seaborn as sns

df = pd.read_csv("bank.csv")
varianceList = df["variance"]
resultList = df["class"]

variance_train, variance_test, result_train, result_test = train_test_split(varianceList, resultList, test_size = 0.25, random_state = 0)

x = np.reshape(variance_train.ravel(), (len(variance_train), 1))
y = np.reshape(result_train.ravel(), (len(result_train), 1))

classifier = LogisticRegression(random_state = 0)
classifier.fit(x, y.ravel())

x_test = np.reshape(variance_test.ravel(), (len(variance_test), 1))
y_test = np.reshape(result_test.ravel(), (len(result_test), 1))

xPred = classifier.predict(x_test)

predValues = []

for i in xPred:
    if i == 0:
        predValues.append("Authorized")
    else:
        predValues.append("Forged")

actualValues = []

for i in y_test.ravel():
    if i == 0:
        actualValues.append("Authorized")
    else:
        actualValues.append("Forged")

labels = ["Forged", "Authorized"]

cm = confusion_matrix(actualValues, predValues, labels)
ax = plt.subplot()
sns.heatmap(cm, annot = True, ax = ax)

ax.set_xlabel('Predicted')
ax.set_ylabel('Actual') 
ax.set_title('Confusion Matrix')
ax.xaxis.set_ticklabels(labels); ax.yaxis.set_ticklabels(labels)
plt.show()

accuracy = (36 + 16) / (36 + 16 + 17 + 7)

print("Accuracy: "+str(accuracy * 100)+"%")
