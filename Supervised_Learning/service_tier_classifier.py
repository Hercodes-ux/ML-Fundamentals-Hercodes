import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

# 1. THE DATA: Telecom Customer Behavior
# Features: Monthly_Data_GB, Support_Calls
# Target: Service_Tier (1=Premium Priority, 0=Standard)
data = {
    'Data_Usage_GB': [10, 85, 20, 95, 5, 70, 15, 60, 40, 30, 80, 2],
    'Support_Calls': [1, 5, 2, 4, 0, 4, 1, 3, 5, 2, 4, 1],
    'Priority_Tier': [0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0]
}
df = pd.DataFrame(data)

X = df[['Data_Usage_GB', 'Support_Calls']]
y = df['Priority_Tier']

# 2.Feature Scaling
# Scaling ensures the 'Support_Calls' isn't ignored by the math
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 3. BUILD THE MODEL (K=3)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# 4. PREDICT FOR A NEW BUSINESS CASE
# A user with 50GB usage and 2 support calls
new_customer = scaler.transform([[50, 2]])
prediction = knn.predict(new_customer)
probability = knn.predict_proba(new_customer)[0][1]

print(f"--- 📡 Telecom Customer Intelligence ---")
print(f"Customer Priority Probability: {probability:.2%}")
print(f"Classification: {'PRIORITY TIER' if prediction[0] == 1 else 'STANDARD TIER'}")

# 5. VISUALIZATION: The Decision Map
plt.figure(figsize=(8,6))
sns.scatterplot(x=X_scaled[:,0], y=X_scaled[:,1], hue=y, s=200, palette='coolwarm')
plt.title("KNN: Priority Tier Classification (Standardized Space)")
plt.xlabel("Data Usage (Scaled)")
plt.ylabel("Support Calls (Scaled)")
plt.savefig("knn_priority_map.png")
plt.show()