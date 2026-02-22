import plotly.express as px
import pandas as pd


def plot_interactive_trend(df, x_col, y_col, color_col, title):
    """
    Creates a professional interactive scatter plot using Plotly.
    Allows stakeholders to hover and explore data points dynamically.
    
   
    """
    #Creating the figure
    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        color=color_col,
        size='Revenue',
        title=title,
        template="plotly_dark",
        hover_name=color_col,
        log_x=True,
        size_max=60,
        labels={x_col: "Marketing Spend ($)", y_col: "Sales Growth (%)"}



    )
    print(f"--- Generating Interactive Visualization: {title} ---")
    fig.show()


# --- EXECUTION BLOCK ---
if __name__ == "__main__":
    # 1. Creating the dataset -
    data = {
        'Company': ['Samsung', 'Apple', 'Google', 'Microsoft', 'Nvidia', 'Meta', 'Amazon', 'Intel'],
        'Marketing_Spend': [5000, 4500, 6000, 5500, 2000, 4800, 7000, 3000],
        'Growth_Percentage': [12, 10, 15, 14, 45, 18, 22, 5], # Capital P
        'Revenue': [100, 120, 150, 140, 300, 110, 200, 80],
        'Region': ['Asia', 'Americas', 'Americas', 'Americas', 'Americas', 'Americas', 'Americas', 'Americas']
    }
    df = pd.DataFrame(data)

    # 2. Running the Plotly engine -
    plot_interactive_trend(
        df, 
        x_col='Marketing_Spend', 
        y_col='Growth_Percentage', # This must match the dictionary exactly!
        color_col='Company', 
        title="2026 Tech Market Analysis: ROI vs. Ad Spend"
    )