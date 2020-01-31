import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.ensemble import RandomForestClassifier
import os
for dirname, _, filenames in os.walk('./input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
        
# Load Data
train_data = pd.read_csv("./input/train.csv")
train_data.head()
test_data = pd.read_csv("./input/test.csv")
test_data.head()

#Age Median
medianTrain =  train_data["Age"].median()
medianTest =  test_data["Age"].median()
train_data["Age"] = train_data["Age"].fillna(medianTrain)
test_data["Age"] = test_data["Age"].fillna(medianTest)

# Replace Embarked with interpretable int numbers.
train_data["Embarked"] = train_data["Embarked"].apply(lambda embarked: 0.0 if embarked == 'C' else 1.0 if embarked == 'S' else 2.0 if embarked == 'Q' else np.NaN)
test_data["Embarked"] = test_data["Embarked"].apply(lambda embarked: 0.0 if embarked == 'C' else 1.0 if embarked == 'S' else 2.0 if embarked == 'Q' else np.NaN)

medianTrain = train_data["Embarked"].median()
medianTest =  test_data["Embarked"].median()

train_data["Embarked"] = train_data["Embarked"].apply(lambda embarked: embarked if not np.isnan(embarked) else medianTrain)
test_data["Embarked"] = test_data["Embarked"].apply(lambda embarked: embarked if not np.isnan(embarked) else medianTest)

#Fare Median, depending on class
meanTrainClass1 = train_data["Fare"][train_data["Pclass"] == 1].mean()
meanTrainClass2 = train_data["Fare"][train_data["Pclass"] == 2].mean()
meanTrainClass3 = train_data["Fare"][train_data["Pclass"] == 3].mean()

meanTestClass1 =  test_data["Fare"][test_data["Pclass"] == 1].mean()
meanTestClass2 =  test_data["Fare"][test_data["Pclass"] == 2].mean()
meanTestClass3 =  test_data["Fare"][test_data["Pclass"] == 3].mean()

train_data["Fare"][train_data["Pclass"] == 1] = train_data["Fare"][train_data["Pclass"] == 1].fillna(meanTrainClass1)
train_data["Fare"][train_data["Pclass"] == 2] = train_data["Fare"][train_data["Pclass"] == 2].fillna(meanTrainClass2)
train_data["Fare"][train_data["Pclass"] == 3] = train_data["Fare"][train_data["Pclass"] == 3].fillna(meanTrainClass3)

test_data["Fare"][test_data["Pclass"] == 1] = test_data["Fare"][test_data["Pclass"] == 1].fillna(meanTestClass1)
test_data["Fare"][test_data["Pclass"] == 2] = test_data["Fare"][test_data["Pclass"] == 2].fillna(meanTestClass2)
test_data["Fare"][test_data["Pclass"] == 3] = test_data["Fare"][test_data["Pclass"] == 3].fillna(meanTestClass3)


# Write our current table to compare..
print("WRITING TO OUTPUT")
train_data.to_csv('./output/train_afterModif.csv', index=False)
test_data.to_csv('./output/test_afterModif.csv', index=False)

y = train_data["Survived"]

features = ["Pclass", "Sex", "SibSp", "Parch", "Age", "Fare", "Embarked"]
X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])

model = RandomForestClassifier(n_estimators=1000, max_depth=None, random_state=1, max_features=None, min_samples_split=2)
model.fit(X, y)
predictions = model.predict(X_test)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('./output/my_submission.csv', index=False)
print("Your submission was successfully saved!")