# import pandas as pd
# from  sklearn import preprocessing
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.preprocessing import LabelEncoder
# from sklearn.metrics import accuracy_score

# train_data = pd.read_csv('train.csv')
# test_data = pd.read_csv('train.csv')

# dropColumns_train = train_data['Name','Ticket','Cabin']
# train_data['Age'].fillna(train_data['Age'].median(),implace=True)

# test_data['Age'].fillna(test_data['Age'].median(),implace=True)

# train_data.drop(columns=['Cabin'],inplace=True)
# test_data.drop(columns=['Cabin'],inplace=True)

# train_data.drop(columns=['PassengerId',"Name","Ticket"],inplace=True)
# test_data.drop(columns=['PassengerId',"Name","Ticket"],inplace=True)

# print(train_data.head())

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

drop_columns = ['Name', 'Ticket', 'Cabin']
train_data.drop(columns=drop_columns, inplace=True)
test_data.drop(columns=drop_columns, inplace=True)

train_data['Age'].fillna(train_data['Age'].median(), inplace=True)
test_data['Age'].fillna(test_data['Age'].median(), inplace=True)


label_encoder = LabelEncoder()
for col in ['Sex','Embarked']:
    train_data[col] = label_encoder.fit_transform(train_data[col].astype(str))
    test_data[col] = label_encoder.transform(test_data[col])

train_data.drop(columns=['PassengerId'],inplace=True)

print(train_data.head())

model = RandomForestClassifier(n_estimators=120, random_state=45)

X = train_data.drop(columns=['Survived'])
Y = train_data['Survived']

Xtrain,Xtest,Ytrain,Ytest = train_test_split(X,Y,test_size=0.25,random_state=45)
model.fit(Xtrain,Ytrain)
ypred = model.predict(Xtest)
print(accuracy_score(Ytest,ypred))

test_features = test_data.drop(columns=['PassengerId']).copy()
testPredictions = model.predict(test_features)


submission_Data = pd.DataFrame({'PassengerId': test_data['PassengerId'], 'Survived': testPredictions})
submission_Data.to_csv('submission.csv', index=False)
