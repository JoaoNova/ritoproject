from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

from Model.callbacks import EarlyStoppingAtMinLoss

BRChallengerMatches = pd.read_csv("../Data/dados_historico.csv")
BRChallengerMatches.head()

# Win = 0 (Blue), 1 (Red)
TargetVariable = ['Win']
Predictors = ['b1', 'b2', 'b3', 'b4', 'b5',
              'r1', 'r2', 'r3', 'r4', 'r5']

X = BRChallengerMatches[Predictors].values
y = BRChallengerMatches[TargetVariable].values

### Sandardization of data ###
### We does not standardize the Target variable for classification
PredictorScaler = StandardScaler()

# Storing the fit object for later reference
PredictorScalerFit = PredictorScaler.fit(X)

# Generating the standardized values of X and y
X = PredictorScalerFit.transform(X)

print(X)

# Split the data into training and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Quick sanity check with the shapes of Training and Testing datasets
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

classifier = Sequential()

# Defining the Input layer and FIRST hidden layer,both are same!
# relu means Rectifier linear unit function
classifier.add(Dense(units=10, input_dim=10, kernel_initializer='uniform', activation='relu'))

#Defining the SECOND hidden layer, here we have not defined input because it is
# second layer and it will get input as the output of first hidden layer
classifier.add(Dense(units=64, kernel_initializer='uniform', activation='relu'))
classifier.add(Dropout(0.3))
classifier.add(Dense(units=64, kernel_initializer='uniform', activation='relu'))
classifier.add(Dropout(0.3))
classifier.add(Dense(units=64, kernel_initializer='uniform', activation='relu'))
classifier.add(Dropout(0.3))
classifier.add(Dense(units=10, kernel_initializer='uniform', activation='relu'))
classifier.add(Dropout(0.3))

# Defining the Output layer
# sigmoid means sigmoid activation function
# for Multiclass classification the activation ='softmax'
# And output_dim will be equal to the number of factor levels
classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))

# Optimizer== the algorithm of SGG to keep updating weights
# loss== the loss function to measure the accuracy
# metrics== the way we will compare the accuracy after each step of SGD
classifier.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

# fitting the Neural Network on the training data
challengerMatches_ANN_Model = classifier.fit(X_train,y_train, batch_size=64, epochs=10000,
                                             callbacks=EarlyStoppingAtMinLoss(patience=250),
                                             verbose=1)

# Test the already trained model vs the test data
loss, accuracy = classifier.evaluate(x=X_test, y=y_test)
print("Loss: %.3f, Accuracy: %.3f" % (loss, accuracy))