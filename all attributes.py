import pandas as pd
from scipy.sparse.construct import rand
from sklearn import tree
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

dataframe = pd.read_csv("diabetes_dataset.csv")

attributes = dataframe.iloc[:, :-1].values
outcome = dataframe.iloc[:, -1].values

train_att, test_att, train_outcome, test_outcome = train_test_split(attributes, outcome, test_size= 0.2, random_state= 0)

scalar = StandardScaler()
train_att = scalar.fit_transform(train_att)
test_att = scalar.transform(test_att)

model = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
model = model.fit(train_att, train_outcome)
 
prediction = model.predict( test_att)

tree.plot_tree(model)

con_matrix = confusion_matrix(prediction, test_outcome)
print(accuracy_score(test_outcome, prediction), con_matrix)
