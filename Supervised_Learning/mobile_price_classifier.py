import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# 1. DATA (Same as before)
data = {
    'RAM': [512, 1024, 2048, 4096, 8192, 1024, 2048, 4096, 6144, 512, 8192, 4096, 12000, 512, 3000, 6000],
    'Battery': [2000, 2500, 3000, 4000, 5000, 2200, 3100, 4200, 4500, 1800, 5000, 3800, 6000, 1900, 3500, 4800],
    'Memory': [16, 32, 64, 128, 256, 32, 64, 128, 256, 16, 512, 128, 512, 16, 64, 256],
    'Price_Range': [0, 0, 1, 1, 2, 0, 1, 1, 2, 0, 2, 1, 2, 0, 1, 2] 
}
df = pd.DataFrame(data)
X = df[['RAM', 'Battery', 'Memory']]
y = df['Price_Range']

# FIXED: Added stratify=y
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# 2. TRAINING
dt_clf = DecisionTreeClassifier(random_state=42)
dt_clf.fit(X_train, y_train)
dt_preds = dt_clf.predict(X_test)

rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train, y_train)
rf_preds = rf_clf.predict(X_test)

# 3. EVALUATION
print("\n📊 HERSHINE ANALYTICS: MODEL BENCHMARKING")
target_names = ['Budget', 'Mid-Range', 'Flagship']

# FIXED: Added labels and zero_division
print("\n[SCALING INSIGHT] Classification Report (Random Forest):")
print(classification_report(y_test, rf_preds, target_names=target_names, labels=[0, 1, 2], zero_division=0))

# 4. VISUALS (Same as before)
fig, ax = plt.subplots(1, 2, figsize=(16, 6))
cm = confusion_matrix(y_test, rf_preds, labels=[0, 1, 2])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax[0], xticklabels=target_names, yticklabels=target_names)
ax[0].set_title('Hercodes-ux: Confusion Matrix')

importances = rf_clf.feature_importances_
sns.barplot(x=importances, y=X.columns, hue=X.columns, palette='viridis', ax=ax[1], legend=False)
ax[1].set_title('Strategic Analysis: Feature Importance')
plt.tight_layout()
plt.savefig('mobile_model_results.png', dpi=300) 
plt.show()