import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.ensemble import ExtraTreesRegressor,RandomForestRegressor,RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import sklearn.metrics
import csv
from sklearn import linear_model
import warnings

# warnings.filterwarnings('ignore')

# load data
df = pd.read_csv('C:\\Users\\lpc\\Desktop\\public_dataset\\train.csv')
df = df.dropna()
print(df.shape)
# general information
# print(df.describe())

# print(df.corr())

df['bmi_int'] = df['bmi'].apply(lambda x: int(x))
variables = ['sex', 'smoker', 'region', 'age', 'bmi_int', 'children']

# data distribution analysys
'''print('Data distribution analysys')
for v in variables:
    df = df.sort_values(by=[v])
    df[v].value_counts().plot(kind = 'bar')
    plt.title(v)
    plt.show()'''

'''print('Mean cost analysys:')
for v in variables:
    group_df = df.groupby(pd.Grouper(key=v)).mean()
    group_df = group_df.sort_index()
    group_df.plot(y = ['charges'],kind = 'bar')
    plt.show()'''

'''print('Variables pairplot:')
variables = ['sex','smoker','region','age','bmi_int','children','charges']
sns_plot = sns.pairplot(df[variables])
plt.show()'''

print('Model training and evaluating\n\n')
# transform categorical data
le_sex = LabelEncoder()
le_smoker = LabelEncoder()
le_region = LabelEncoder()

df['sex'] = le_sex.fit_transform(df['sex'])
df['smoker'] = le_smoker.fit_transform(df['smoker'])
df['region'] = le_region.fit_transform(df['region'])

variables = ['sex', 'smoker', 'region', 'age', 'bmi', 'children']

X = df[variables]
sc = StandardScaler()
X = sc.fit_transform(X)
Y = df['charges']
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1)

# train model

#regressor = ExtraTreesRegressor(n_estimators=100)
regressor = RandomForestRegressor(n_estimators=100)

regressor.fit(X_train, y_train)

# prediction and evaluation
y_train_pred = regressor.predict(X_train)
y_test_pred = regressor.predict(X_test)


print('ExtraTreesRegressor evaluating result:')
print("Train MAE: ", sklearn.metrics.mean_absolute_error(y_train, y_train_pred))
print("Train RMSE: ", np.sqrt(sklearn.metrics.mean_squared_error(y_train, y_train_pred)))
print("Test MAE: ", sklearn.metrics.mean_absolute_error(y_test, y_test_pred))
print("Test RMSE: ", np.sqrt(sklearn.metrics.mean_squared_error(y_test, y_test_pred)))
print("aaaa",sklearn.metrics.explained_variance_score(y_test, y_test_pred))
print('Feature importance ranking\n\n')
importances = regressor.feature_importances_
std = np.std([tree.feature_importances_ for tree in regressor.estimators_], axis=0)
indices = np.argsort(importances)[::-1]

importance_list = []
for f in range(X.shape[1]):
    variable = variables[indices[f]]
    importance_list.append(variable)
    print("%d.%s(%f)" % (f + 1, variable, importances[indices[f]]))

# Plot the feature importances of the forest
'''plt.figure()
plt.title("Feature importances")
plt.bar(importance_list, importances[indices],
       color="r", yerr=std[indices], align="center")
plt.show()'''
outs = []
print('Predicting on new data\n\n')
i=0
with open('C:\\Users\\lpc\\Desktop\\public_dataset\\test_sample.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        if i != 0:
            age = row[0]
            sex = row[1]
            bmi = row[2]
            children = row[3]
            smoker = row[4]
            region = row[5]
            one = [sex, smoker, region, age, bmi, children]
            out = row
            #print(one)
            one[0] = le_sex.transform([one[0]])[0]
            one[1] = le_smoker.transform([one[1]])[0]
            one[2] = le_region.transform([one[2]])[0]
            X = sc.transform([one])
            cost = regressor.predict(X)[0]
            #print(one, cost)
            out[6] = cost
            outs.append(out)
        else:
            i = 1
            outs.append(row)

with open('C:\\Users\\lpc\\Desktop\\public_dataset\\submission.csv','w',newline='') as file:
    writer = csv.writer(file)
    writer.writerows(outs)
#print(outs)
'''billy = ['male','yes','southeast',25,30.5,2]
print('Billy - ',str(billy))

billy[0] = le_sex.transform([billy[0]])[0]
billy[1] = le_smoker.transform([billy[1]])[0]
billy[2] = le_region.transform([billy[2]])[0]

X = sc.transform([billy])

cost_for_billy = regressor.predict(X)[0]
print('Cost for Billy = ',cost_for_billy,'\n\n')'''
