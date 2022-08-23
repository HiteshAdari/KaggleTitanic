import sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler

pd.options.display.max_columns = 99
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

print(train_data.info())


def cleantens(train_data):
    X = train_data
    X.pop('Ticket')
    X.pop('Cabin')
    X.pop('PassengerId')
    X.pop('Name')
    X = X.dropna()
    y = X.pop('Survived')
    non_num_cat = ['Sex', 'Embarked']

    for col in non_num_cat:
        X[col] = X[col].astype('category').cat.codes
    return X, y


X, y = cleantens(train_data)
print(X.info())
print(y)

model = RandomForestClassifier(n_jobs=-1, random_state=29)
