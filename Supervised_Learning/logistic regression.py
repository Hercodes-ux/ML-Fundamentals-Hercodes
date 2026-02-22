# --- STEP 1: IMPORTING THE LIBRARIES ---
# Pandas is for data tables, Scikit-Learn is for the AI, Matplotlib is for charts
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler

# --- STEP 2: CREATING THE DATASET ---
# Imagine we have 10 students with their study hours and Result (0=Fail, 1=Pass)
data = {
    'Hours_Studied': [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 4.5, 5.0, 6.0],
    'Result': [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
    'Attendance_Percentage': [20, 30, 40, 50, 60, 70, 80, 90, 95, 100]
}
df = pd.DataFrame(data)

# --- STEP 3: SPLITTING DATA INTO X (Input) and Y (Output) ---
X = df[['Hours_Studied', 'Attendance_Percentage']] # use both features for multiple logistic regression
y = df['Result']          # Output is the label we want to predict

# Split into Training (80%) and Testing (20%)
# This ensures the AI is tested on data it has NEVER seen before
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features (recommended for multivariate models)
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- STEP 4: INITIALIZING & TRAINING THE MODEL ---
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# --- STEP 5: MAKING PREDICTIONS ---
# Let's predict the results for our test set (use scaled test data)
y_pred = model.predict(X_test_scaled)

# --- STEP 6: EVALUATING THE AI ---
# Accuracy tells us how many it got right out of 100
score = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print(f"Model Accuracy: {score * 100}%")
print("Confusion Matrix:")
print(cm)

# Print learned coefficients and odds ratios
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
odds_ratios = np.exp(model.coef_)
print("Odds Ratios:", odds_ratios)

# --- STEP 7: 2D PROBABILITY CONTOUR (Hours_Studied x Attendance_Percentage) ---
plt.figure(figsize=(8,6))

# build grid over the feature ranges
hours_min, hours_max = df.Hours_Studied.min()-1, df.Hours_Studied.max()+1
att_min, att_max = df.Attendance_Percentage.min()-10, df.Attendance_Percentage.max()+10
hours_range = np.linspace(hours_min, hours_max, 200)
att_range = np.linspace(att_min, att_max, 200)
H, A = np.meshgrid(hours_range, att_range)
grid = np.c_[H.ravel(), A.ravel()]

# scale grid the same way as training data
grid_scaled = scaler.transform(grid)
probs = model.predict_proba(grid_scaled)[:, 1]
Z = probs.reshape(H.shape)

# contour plot of predicted probability
cf = plt.contourf(H, A, Z, levels=20, cmap='RdYlBu', alpha=0.8)
plt.colorbar(cf, label='Probability of Passing')

# plot actual data points
plt.scatter(df.Hours_Studied, df.Attendance_Percentage, c=df.Result, cmap='bwr', edgecolor='k')
plt.xlabel('Hours Studied')
plt.ylabel('Attendance Percentage')
plt.title('Predicted Probability of Passing (contour)')
plt.show()