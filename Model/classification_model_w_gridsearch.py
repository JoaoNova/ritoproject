from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

# To remove the scientific notation from numpy arrays
np.set_printoptions(suppress=True)

TitanicSurvivalDataNumeric = pd.read_pickle('../Data/TitanicSurvivalDataNumeric.pkl')
TitanicSurvivalDataNumeric.head()

TargetVariable=['Survived']
Predictors=['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare',
            'Embarked_C', 'Embarked_Q', 'Embarked_S']

X=TitanicSurvivalDataNumeric[Predictors].values
y=TitanicSurvivalDataNumeric[TargetVariable].values


### Sandardization of data ###
### We does not standardize the Target variable for classification
from sklearn.preprocessing import StandardScaler
PredictorScaler=StandardScaler()

# Storing the fit object for later reference
PredictorScalerFit=PredictorScaler.fit(X)

# Generating the standardized values of X and y
X=PredictorScalerFit.transform(X)

# Split the data into training and testing set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Quick sanity check with the shapes of Training and Testing datasets
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# Function to generate Deep ANN model
def make_classification_ann(Optimizer_Trial, Neurons_Trial):
    # Creating the classifier ANN model
    classifier = Sequential()
    classifier.add(Dense(units=Neurons_Trial, input_dim=9, kernel_initializer='uniform', activation='relu'))
    classifier.add(Dense(units=Neurons_Trial, kernel_initializer='uniform', activation='relu'))
    classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
    classifier.compile(optimizer=Optimizer_Trial, loss='binary_crossentropy', metrics=['accuracy'])

    return classifier


########################################

from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier

Parameter_Trials = {'batch_size': [10, 20, 30],
                    'epochs': [10, 20],
                    'Optimizer_Trial': ['adam', 'rmsprop'],
                    'Neurons_Trial': [5, 10]
                    }

# Creating the classifier ANN
classifierModel = KerasClassifier(make_classification_ann, verbose=0)

########################################

# Creating the Grid search space
# See different scoring methods by using sklearn.metrics.SCORERS.keys()
grid_search = GridSearchCV(estimator=classifierModel, param_grid=Parameter_Trials, scoring='f1', cv=5, n_jobs=3)

########################################

# Measuring how much time it took to find the best params
import time

StartTime = time.time()

# Running Grid Search for different paramenters
grid_search.fit(X_train, y_train, verbose=1)

EndTime = time.time()
print("############### Total Time Taken: ", round((EndTime - StartTime) / 60), 'Minutes #############')

########################################

# printing the best parameters
print('\n#### Best hyperparamters ####')
print(grid_search.best_params_)
