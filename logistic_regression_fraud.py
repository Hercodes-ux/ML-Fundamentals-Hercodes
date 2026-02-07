import numpy as numpy
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report

# 1. DATA: Transaction Amount ($) vs Is_Fraud (0=No, 1=Yes)
# Generally, higher amounts have a higher chance of being fraud in this sample
X = numpy.array([10, 50, 100, 500, 1000, 5000, 10000]).reshape(-1, 1)
y =numpy.array([0, 0, 0, 0, 1, 1, 1])

#2.Training and Initializing the model  
# 'Sigmoid' function happens inside here to classify fraud vs non-fraud 
clf =LogisticRegression()
clf.fit(X,y)

#3.Predicting the Probability
# What is the chance that a $1200 transaction is fraud?
test_amount =numpy.array([[1200]])
prob = clf.predict_proba(test_amount)[0][1]  # Probability of class '1' (Fraud)
decision =clf.predict(test_amount)[0]  # Final decision: 0 or 1

print(f"------Logistic Regression Fraud Detection------")
print(f"Transaction amount:${test_amount[0]}")
print(f"Probability of fraud: {prob:.2f}")
print(f"AI Decision {'Fraud-suspicious' if decision==1 else 'Safe'}")


#Evaluvation(confusion matrix)
y_pred = clf.predict(X)
cm = confusion_matrix(y, y_pred)
print(f"\nConfusion Matrix:\n{cm}")
print("\n--- Detailed Business Metrics ---")
print(classification_report(y, y_pred))