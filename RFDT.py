import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

#Loading loan_data.csv file into a dataframe
loans = pd.read_csv('loan_data.csv')

print(loans.info(), '\n', loans.head(), '\n', loans.describe())


#More than one histogram on top of each other

plt.figure(figsize = (10,6))
plt.rcParams["patch.force_edgecolor"] = True #Without this, no edge on the bars
loans[loans['credit.policy']==1]['fico'].hist(bins =35,color = 'blue', label = 'Credit Policy =1', alpha = 0.6)
loans[loans['credit.policy']==0]['fico'].hist(bins = 35, color = 'red', label = 'Credit Policy =0',alpha = 0.6)
plt.legend()

plt.figure(figsize = (10,6))
loans[loans['not.fully.paid']==1]['fico'].hist(bins =35,color = 'blue', label = 'Credit Policy =1', alpha = 0.6)
loans[loans['not.fully.paid']==0]['fico'].hist(bins = 35, color = 'red', label = 'Credit Policy =0',alpha = 0.6)
plt.legend()

#Count plot with hue as not.fully.paid
plt.figure(figsize = (10,6))
sns.countplot(x='purpose', data = loans, hue = 'not.fully.paid', palette='Set1')

sns.jointplot(x = 'fico', y = 'int.rate', data = loans, color = 'purple')

plt.figure(figsize=(11,7))
sns.lmplot(y='int.rate',x='fico',data=loans,hue='credit.policy',
           col='not.fully.paid',palette='Set1')

plt.show()


#   ----Random Forest part ----

cat_feats = ['purpose']
final_data = pd.get_dummies(loans, columns = cat_feats, drop_first = True)
print(final_data.head())

X = final_data.drop('not.fully.paid',axis=1)
y = final_data['not.fully.paid']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)

#Decision Tree
dtree = DecisionTreeClassifier()
dtree.fit(X_train,y_train)

predictions = dtree.predict(X_test)
print(classification_report(y_test,predictions), '\n', confusion_matrix(y_test,predictions))

#Random Forest
rfc = RandomForestClassifier(n_estimators=600)
rfc.fit(X_train,y_train)
rpredictions = rfc.predict(X_test)
print(classification_report(y_test,rpredictions), '\n', confusion_matrix(y_test,rpredictions))