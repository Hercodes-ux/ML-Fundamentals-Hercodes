# Advanced Visualization: Comparative Distribution Analysis
# Shows understanding of Statistical Normalization

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm

def plot_comparison(data):
    # Set a professional theme
    sns.set_theme(style="darkgrid")
    plt.figure(figsize=(10, 6))

    # 1. Plotting the actual data distribution (The blue bars)
    sns.histplot(data, kde=True, color='blue', stat='density', label='Observed Data') 

    # 2. Generating a Theoretical Normal Distribution (The black line)
    # This proves you understand what 'Normal' should look like
    mu, std = norm.fit(data)
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    plt.plot(x, p, 'k', linewidth=2, label='Normal Distribution Curve')

    # Adding Title and Labels
    plt.title("Comparative Distribution Analysis: Observed vs Normal - Hercodes Tool", fontsize=14)
    plt.axvline(np.mean(data), color='red', linestyle='--', label=f"Mean: {np.mean(data):.2f}")
    
    # --- FIX WAS HERE: Added the 'd' to legend() ---
    plt.legend()
    
    # 3. Save the image (Good for GitHub READMEs)
    plt.savefig("distribution_plot.png")
    
    # Display the chart
    plt.show()

# --- EXECUTION BLOCK ---
if __name__ == "__main__":
    # Generating 1000 random points centered at 100
    sample_data = np.random.normal(100, 20, 1000) 
    plot_comparison(sample_data)