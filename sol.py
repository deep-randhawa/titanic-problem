# coding: utf-8

## -- MODULES -- ##
import numpy as np
import pandas as pd
import sklearn.ensemble as ske
from patsy import dmatrices
from sklearn.cross_validation import train_test_split

## -- HELPER FUNCTIONS -- ##
def name_extract(word):
    return word.split(',')[1].split('.')[0].strip()


def group_salutation(old_salutation):
    if old_salutation == 'Mr' or old_salutation == 'Master':
        return (old_salutation)
    elif old_salutation == 'Mrs' or old_salutation == 'Mme':
        return (old_salutation)
    elif old_salutation == 'Miss' or old_salutation == 'Mlle':
        return (old_salutation)
    else:
        return ('Others')


def group_age(age):
    if (age < 12):
        return ('Children')
    elif (age < 50):
        return ('Adults')
    else:
        return ('Elderly')


### --- Groups passengers by their salutation (Mr., Mrs., Miss, Master) --- ###
def map_salutation(df):
    df2 = pd.DataFrame({'Salutation': df['Name'].apply(name_extract)})
    df = pd.merge(df, df2, left_index=True, right_index=True)  # merges on index

    df3 = pd.DataFrame({'New_Salutation': df['Salutation'].apply(group_salutation)})
    df = pd.merge(df, df3, left_index=True, right_index=True)
    return df


### --- Groups passengers by their Age (Children, Adults, Elderly) --- ###
def map_ages(df):
    df4 = pd.DataFrame({'Age_Class': df['Age'].apply(group_age)})
    df = pd.merge(df, df4, left_index=True, right_index=True)
    return df

## -- PREPROSESSING ANALYSIS -- ##
# Removing rows and columns that are None
df = pd.read_csv("train.csv")
df = df.drop(['Ticket', 'Cabin'], axis=1)
# Remove NaN values
df = df.dropna()

df = map_salutation(df)
df = map_ages(df)



## -- MODEL -- ##
# model formula
# here the ~ sign is an = sign, and the features of our dataset
# are written as a formula to predict survived. The C() lets our
# regression know that those variables are categorical.
formula = 'Survived ~ C(Pclass) + C(Sex) + Fare + SibSp  + C(Embarked) + C(New_Salutation) + C(Age_Class)'
# create a results dictionary to hold our regression results for easy analysis later
results = {}

train_data = df[0:500]
test_data = df[501:]

# Create the random forest model and fit the model to our training data
y, x = dmatrices(formula, data=train_data, return_type='dataframe')
# RandomForestClassifier expects a 1 demensional NumPy array, so we convert
y = np.asarray(y).ravel()
# instantiate and fit our model
results_rf = ske.RandomForestClassifier(n_estimators=100).fit(x, y)

# Score the results
y, x = dmatrices(formula, data=test_data, return_type='dataframe')
score = results_rf.score(x, y)
print "Mean accuracy of Random Forest Predictions on the data was: {0}".format(score)


## -- PROBLEMS -- ##
# A single parent will go with his/her children, and this can be determined using family names and ages
