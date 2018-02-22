# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, -1].values

#Fitting Linear Regression to the data set
#For polynomial regression, apply linear regression first then
#apply polynomial regression becuase polynomial regression
#will add its variables to nth exponent like a polynomial regression formula
#once linear regression has been applied
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
#fit to X and Y since the data set is too small to split it to train and test sets
lin_reg.fit(X,y)

#Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=4)
#Fit and transform poly_reg(X) into X_poly with 4 degrees
#X_poly[2:5] are [1] but to the nth power and [0] are all ones because that is needed
#to represent the constant Bzero
X_poly = poly_reg.fit_transform(X)

#Create new linear Regression object and fit to X_poly and y
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly, y)

#Visualize linear regression results
#.scatter is the real results
plt.scatter(X,y, color='red')
#.plot is the predicted results
plt.plot(X, lin_reg.predict(X), color='blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position Level')
plt.show()

#Visualize polynomial regression results
X_grid = np.arange(min(X), max(X),0.1)
#len(#of elements in x grid, 1 column)
X_grid= X_grid.reshape((len(X_grid),1))
#.scatter is the real results
plt.scatter(X,y, color='red')
#.plot is the predicted results
#lin_reg2 becuase that has polynomial features
#poly_reg.fit_transform(X) becuase X_poly may change but poly_reg.fit_transform(X) will not
plt.plot(X_grid, lin_reg2.predict(poly_reg.fit_transform(X_grid)), color='blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position Level')
plt.show()

import statsmodels.formula.api as sm
def backwardElimination(x, SL):
    numVars = len(x[0])
    temp = np.zeros((50,6)).astype(int)
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        adjR_before = regressor_OLS.rsquared_adj.astype(float)
        if maxVar > SL:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    temp[:,j] = x[:, j]
                    x = np.delete(x, j, 1)
                    tmp_regressor = sm.OLS(y, x).fit()
                    adjR_after = tmp_regressor.rsquared_adj.astype(float)
                    if (adjR_before >= adjR_after):
                        x_rollback = np.hstack((x, temp[:,[0,j]]))
                        x_rollback = np.delete(x_rollback, j, 1)
                        print (regressor_OLS.summary())
                        return x_rollback
                    else:
                        continue
    regressor_OLS.summary()
    return x
 
SL = 0.05
X_opt = X_poly[:, [0,1,2,3,4]]
X_Modeled = backwardElimination(X_opt, SL)

#Predicting a new result with Linear Regression
#Predicting what someone with a job level of 6.5 will receive in salary
lin_reg.predict(6.5)
#predicted salary is $330,378 based off linear regression

#predicting a new result with Polynomial Regression
#Predicting what someone with a job level of 6.5 will receive in salary
lin_reg2.predict(poly_reg.fit_transform(6.5))
#predicted salary is $158,862


