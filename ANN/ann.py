# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('finalcopy-1.csv')
X = dataset.iloc[:, 9:-1].values
y = dataset.iloc[:, -1].values


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Part 2 - Making the ANN

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
import theano.ifelse

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer with dropout
classifier.add(Dense(units = 14, kernel_initializer = 'uniform', activation = 'relu', input_dim = 27))
classifier.add(Dropout(rate=0.4))

# Adding the second hidden layer
classifier.add(Dense(units = 14, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dropout(rate=0.4))

classifier.add(Dense(units = 14, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dropout(rate=0.4))

classifier.add(Dense(units = 14, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dropout(rate=0.4))


# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 25, epochs = 500)

# Part 3 - Making predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

dataset_con_flo=pd.read_csv('con_st.csv')
con_flo_x=dataset_con_flo.iloc[:31, 1:].values
con_flo_x=sc.transform(con_flo_x)
y_pred_con_flo=classifier.predict(con_flo_x)
y_pred_con_flo = (y_pred_con_flo > 0.5)
wpercent=0
total=0
for i in y_pred_con_flo:
    total=total+1
    if i == True:
        wpercent=wpercent+1
print(wpercent/total)
print(wpercent)
print(total)

np.any(np.isnan(dataset_con_flo))
np.all(np.isfinite(dataset_con_flo))

print(dataset_con_flo)

#Evaluating the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(units = 14, kernel_initializer = 'uniform', activation = 'relu', input_dim = 27))
    classifier.add(Dropout(rate=0.4))
    classifier.add(Dense(units = 14, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dropout(rate=0.4))
    classifier.add(Dense(units = 14, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dropout(rate=0.4))
    classifier.add(Dense(units = 14, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dropout(rate=0.4))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, epochs = 100)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, n_jobs = 1)
mean = accuracies.mean()
variance = accuracies.std()

#Improving ANN
def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(units = 14, kernel_initializer = 'uniform', activation = 'relu', input_dim = 27))
    classifier.add(Dropout(rate=0.4))
    classifier.add(Dense(units = 14, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dropout(rate=0.4))
    classifier.add(Dense(units = 14, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dropout(rate=0.4))
    classifier.add(Dense(units = 14, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dropout(rate=0.4))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, epochs = 100)

parameters = {'batch_size':[25,32],
              'nb_epoch':[100,500],
              'optimizer':['adam','rmsprop']}

grid_search = GridSearchCV(estimator=classifier,
                           param_grid=parameters,
                           scoring = 'accuracy',
                           cv=10)
grid_search=grid_search.fit(X_train,y_train)

best_paramters= grid_search.best_params_
best_accuracy= grid_search.best_score_















