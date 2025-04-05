import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

trainData = pd.read_csv(r'C:\Users\Krishna\Desktop\Machine learning\Kaggle competions\rainprediction\train.csv')
testData = pd.read_csv(r'C:\Users\Krishna\Desktop\Machine learning\Kaggle competions\rainprediction\test.csv')
# import matplotlib.pylab as plt

# print(trainData.isnull())
# value = trainData['rainfall'].value_counts()
# x = value.get(1,0)
# y = value.get(0,0)
# plt.bar(x,y, color=['red','green'])
# plt.show()

x = trainData.drop(columns=['rainfall','id'],axis=1)
y = trainData['rainfall']

model = RandomForestClassifier(n_estimators=200,random_state=45)

xtrain,xtest,ytrain,ytest = train_test_split(x,y,random_state=42)

model.fit(xtrain,ytrain)
from sklearn.metrics import accuracy_score, classification_report

testData_x = testData.drop(columns=['id'])
ypred = model.predict(xtest)
accuracy = accuracy_score(ypred,ytest)
print(accuracy)



predictions = model.predict(testData_x)
submission = pd.DataFrame({
    'id':testData['id'],
    'Rainfall':predictions
})

submission.to_csv('submissionusinghyperparameter.csv',index=False)





