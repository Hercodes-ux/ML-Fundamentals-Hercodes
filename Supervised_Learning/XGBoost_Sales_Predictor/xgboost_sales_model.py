import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# 1. ENHANCED DATA (More rows to help the AI learn)
data = {
    'Cart_Items': [1, 5, 0, 10, 2, 8, 1, 12, 3, 7, 0, 15, 2, 6, 1, 11, 4, 9, 2, 13, 0, 14, 5, 8],
    'Minutes_Spent': [2, 15, 1, 30, 5, 20, 2, 45, 10, 25, 1, 60, 4, 18, 3, 35, 12, 28, 6, 50, 1, 55, 14, 22],
    'Promo_Used': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1],
    'Made_Purchase': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1]
}
df = pd.DataFrame(data)

X = df[['Cart_Items', 'Minutes_Spent', 'Promo_Used']]
y = df['Made_Purchase']

# 2. STRATIFIED SPLIT (Ensures test set isn't 'empty' of one category)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 3. XGBOOST TUNING
# For small data, we use fewer estimators so it doesn't overthink
model = XGBClassifier(
    n_estimators=10, 
    learning_rate=0.05, 
    max_depth=3, 
    eval_metric='logloss'
)
model.fit(X_train, y_train)

# 4. PREDICT & ANALYZE
preds = model.predict(X_test)

print(f"--- 🚀 Hercodes-ux: XGBoost Sales Engine (Optimized) ---")
print(f"Accuracy Score: {accuracy_score(y_test, preds):.2%}")
print("\n[Industrial Metric] Classification Report:")
print(classification_report(y_test, preds))

#chart of the Feature Importance
features = ['Cart_Items', 'Minutes_Spent', 'Promo_Used']
plt.figure(figsize=(10, 6))
sns.barplot(x=model.feature_importances_, y=features, hue=features, palette='magma', legend=False)
plt.title('Hercodes-ux: XGBoost Feature Impact Analysis')
plt.xlabel('Importance Score (0 to 1)')
plt.ylabel('User Behavior')

# Now Feature Importance will show numbers!
importances = model.feature_importance() if hasattr(model, 'feature_importance') else model.feature_importances_
print(f"Feature Importance (Cart/Time/Promo): {importances}")

# SAVE THE IMAGE for GitHub
plt.savefig("xgboost_feature_importance.png")
plt.show()