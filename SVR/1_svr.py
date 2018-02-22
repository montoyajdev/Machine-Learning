# Regression Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values
 
# Feature Scaling
#Need to apply feature scaling since
#SVR does not naturally apply feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = np.ravel(sc_y.fit_transform(y.reshape(-1, 1)))

# Fitting the SVR to the dataset
from sklearn.svm import SVR
#rbf means it is non-linear, specifically Gaussian
regressor = SVR(kernel='rbf')
#fit matrix of features and dependent variable
regressor.fit(X,y)

# Predicting a new result
#Need to transform it into array
#.fit is not needed since it has already been fitted via sc_X.fit_transform(X)
#double brackets is needed for array. Single brackets means vector
#sc_y inverse in needed to revert the feature scaling so we can 
#use a valid dependent variable such as $125,000 instead of a scaled variable such as -1.034 
y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform(np.array([[6.5]]))))

# Visualising the SVR results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()