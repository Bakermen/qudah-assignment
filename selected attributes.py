import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.tree import DecisionTreeClassifier

dataframe = pd.read_csv("diabetes_dataset.csv")

attributes = dataframe[['Glucose', 'Insulin', 'DiabetesPedigreeFunction']]
outcome = dataframe.loc[:, 'Outcome']

train_att = attributes[:700]
train_outcome = outcome[:700]
test_att = attributes[700:]
test_outcome = outcome[700:]

model = DecisionTreeClassifier()
model = model.fit(train_att, train_outcome)
 
prediction = model.predict(test_att)

con_matrix = confusion_matrix(prediction, test_outcome)
print(accuracy_score(test_outcome, prediction) , con_matrix)
