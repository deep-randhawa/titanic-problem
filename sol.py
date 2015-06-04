# coding: utf-8

## -- MODULES -- ##
import numpy as np
import pandas as pd
import sklearn.ensemble as ske
from patsy import dmatrices

## -- HELPER FUNCTIONS -- ##
def name_extract(word):
    return word.split(',')[1].split('.')[0].strip()

def group_salutation(old_salutation):
    if old_salutation == 'Mr' or old_salutation == 'Mrs' or old_salutation == 'Master' or old_salutation == 'Miss':
        return (old_salutation)
    else:
        return ('Others')

## -- PREPROSESSING ANALYSIS -- ##
# Removing rows and columns that are None
df = pd.read_csv("train.csv")
df = df.drop(['Ticket', 'Cabin'], axis=1)
# Remove NaN values
df = df.dropna()

### --- Groups passengers by their salutation (Mr., Mrs., Miss, Master) --- ###
df2 = pd.DataFrame({'Salutation': df['Name'].apply(name_extract)})
df = pd.merge(df, df2, left_index=True, right_index=True)  # merges on index
temp1 = df.groupby('Salutation').PassengerId.count()

df3 = pd.DataFrame({'New_Salutation': df['Salutation'].apply(group_salutation)})
df = pd.merge(df, df3, left_index=True, right_index=True)
temp1 = df3.groupby('New_Salutation').count()


## -- MODEL -- ##
# model formula
# here the ~ sign is an = sign, and the features of our dataset
# are written as a formula to predict survived. The C() lets our
# regression know that those variables are categorical.
formula = 'Survived ~ C(Pclass) + C(Sex) + Age + SibSp  + C(Embarked) + C(New_Salutation)'
# create a results dictionary to hold our regression results for easy analysis later
results = {}

# Create the random forest model and fit the model to our training data
y, x = dmatrices(formula, data=df, return_type='dataframe')
# RandomForestClassifier expects a 1 demensional NumPy array, so we convert
y = np.asarray(y).ravel()
# instantiate and fit our model
results_rf = ske.RandomForestClassifier(n_estimators=100).fit(x, y)

# Score the results
score = results_rf.score(x, y)
print "Mean accuracy of Random Forest Predictions on the data was: {0}".format(score)
