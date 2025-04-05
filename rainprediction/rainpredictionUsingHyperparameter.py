import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

trainData = pd.read_csv(r'C:\Users\Krishna\Desktop\Machine learning\Kaggle competions\rainprediction\train.csv')
testData = pd.read_csv(r'C:\Users\Krishna\Desktop\Machine learning\Kaggle competions\rainprediction\test.csv')


HyperParameter_Dict = {
    'n_estimators':[50,100,125,150],
    'max_depth':[7,10,12,None],
    'min_samples_split': [5, 10, 15],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}
rf = RandomForestClassifier()
random_search = RandomizedSearchCV(estimator=rf, param_distributions=HyperParameter_Dict, 
                                   n_iter=10, cv=5, n_jobs=-1, verbose=2, random_state=42)

#loading the dataset

X = trainData.drop(columns=['id','rainfall']) # removing the target variable and Id
Y = trainData['rainfall'] # spliiting the target variable
xtrain,xtest,ytrain,ytest = train_test_split(X,Y,random_state=45,test_size=0.2)

random_search.fit(xtrain,ytrain)
random_search.predict(xtest)

print("Best Parameters:", random_search.best_params_)
print("Best Score:", random_search.best_score_)


result = testData.drop(columns=['id'])
prediction = random_search.best_estimator_.predict(result)

submission = pd.DataFrame({
    'id':testData['id'],
    'Rainfall':prediction
})


submission.to_csv("submissionUsingHP1.csv",index=False)

from sklearn.metrics import accuracy_score

best_rf = random_search.best_estimator_

train_preds = best_rf.predict(xtrain)
test_preds = best_rf.predict(xtest)

train_acc = accuracy_score(ytrain, train_preds)
test_acc = accuracy_score(ytest, test_preds)

print(f"Train Accuracy: {train_acc}")
print(f"Test Accuracy: {test_acc}")

if train_acc > test_acc + 0.10:
   print(" Possible Overfitting: Training accuracy is much higher than test accuracy.")
elif train_acc < 0.80 and test_acc < 0.80:
    print(" Possible Underfitting: Both training and test accuracy are low.")
else:
    print(" Model is well-fitted!")
