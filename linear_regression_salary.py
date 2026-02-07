import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


#Creating a dataset
# 1. DATA: GPA vs Starting Salary (in $1000s)
X =np.array([2.5, 3.0, 3.2, 3.5, 3.8, 4.0]).reshape(-1, 1) # GPA
y =np.array([45, 55, 62, 70, 85, 95]) #Salary

#Training and Initializing the model
# This is where the AI 'learns' the best-fit line
model =LinearRegression()
model.fit(X,y)

#Going through the Prediction
new_gpa =np.array([[3.6]]) # New GPA to predict salary for  
predicted_salary= model.predict(new_gpa)


print(f"------Linear Regression Prediction------")
print(f"predicted Salary for 3.6 gpa is: ${predicted_salary[0]:.2f}K")
#Equation: y=mX+b
print(f"Equation of the line: Salary = {model.coef_[0]:.2f} * GPA + {model.intercept_: .2f}")


#Visualizing(proof)
plt.scatter(X,y,color='green', label='Actual Data')
plt.plot(X,model.predict(X), color='yellow', label='Regression or Best-fit Line')
plt.xlabel('GPA')
plt.ylabel('Starting Salary ($k)')
plt.title('Hercodes-ux : GPA vs Starting Salary Linear Regression example')
plt.legend()
plt.show()
                            