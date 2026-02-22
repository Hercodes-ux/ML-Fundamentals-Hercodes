import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. SETUP: Visualization Style
sns.set_theme(style="darkgrid")
plt.rcParams['figure.figsize'] = [10, 6]

# 2. DATA: Simulated High-Net-Worth Customer Portfolio
data = {
    'Client_ID': range(5001, 5021),
    'Age': [25, 34, 45, 23, 67, 33, 41, 52, 21, 30, 38, 48, 29, 55, 40, 31, 22, 60, 35, 44],
    'Annual_Income_USD': [50000, 80000, 120000, 40000, 150000, 75000, 90000, 110000, 35000, 70000, 85000, 130000, 65000, 140000, 95000, 78000, 42000, 160000, 82000, 115000],
    'Asset_Score': [30, 50, 85, 20, 95, 45, 60, 80, 15, 40, 55, 90, 50, 88, 65, 52, 25, 98, 48, 75], 
    'Wealth_Tier': ['Standard', 'Silver', 'Gold', 'Standard', 'Elite', 'Silver', 'Silver', 'Gold', 'Standard', 'Silver', 'Silver', 'Elite', 'Silver', 'Elite', 'Gold', 'Silver', 'Standard', 'Elite', 'Silver', 'Gold']
}
df = pd.DataFrame(data)

# --- EDA PHASE ---

# A. Univariate: Distribution of Annual Income
print("Visualizing Income Distributions...")
sns.histplot(df['Annual_Income_USD'], kde=True, color='teal')
plt.title("Client Base: Annual Income Distribution (Retail Banking Sector)")
plt.xlabel("Annual Income (USD)")
plt.savefig("client_income_distribution.png")
plt.show()

# B. Bivariate: Income vs. Asset Score (Segmentation View)
print("Generating Segmentation Scatter Plot...")
sns.scatterplot(x='Annual_Income_USD', y='Asset_Score', hue='Wealth_Tier', s=150, palette='viridis', data=df)
plt.title("Strategic Segmentation: Income vs. Asset Score by Wealth Tier")
plt.xlabel("Annual Income (USD)")
plt.ylabel("Asset Utilization Score")
plt.savefig("segmentation_matrix.png")
plt.show()

# C. Summary for Decision Makers
print("\n--- SUMMARY BY TIER ---")
summary = df.groupby('Wealth_Tier')[['Annual_Income_USD', 'Asset_Score']].mean().round(2)
print(summary)

print("\nAnalysis Complete: Financial Insights exported.")