import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 1. THE DATA: Netflix User Behavior (Unlabeled)
# Features: Monthly Hours Watched, Number of Genres Explored
data = {
    'Hours_Watched': [10, 12, 60, 65, 15, 70, 5, 80, 20, 75, 100, 110],
    'Genres_Explored': [1, 2, 10, 12, 2, 11, 1, 15, 2, 13, 20, 18]
}
df = pd.DataFrame(data)

# 2. CRITICAL: Scaling (K-Means is distance-based, so this is MUST!)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# 3. THE ELBOW METHOD (Finding the Perfect K)
wcss = []
for i in range(1, 7):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# Plotting the Elbow
plt.figure(figsize=(6,4))
plt.plot(range(1, 7), wcss, marker='o', color='red')
plt.title('The Elbow Method (Finding K)')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS (Error)')
plt.show()

# 4. FINAL CLUSTERING (We see the 'elbow' at K=2 or 3)
kmeans = KMeans(n_clusters=2, random_state=42)
clusters = kmeans.fit_predict(X_scaled)
df['Segment'] = clusters

# 5. VISUALIZATION: The Persona Map
plt.figure(figsize=(8,6))
sns.scatterplot(x=df['Hours_Watched'], y=df['Genres_Explored'], hue=df['Segment'], s=200, palette='viridis')
plt.title("Hercodes-ux: Netflix Customer Personas")
plt.savefig("customer_segments.png")
plt.show()

print("Clustering Complete. User Personas identified.")