import pandas as pd
import numpy as np

class DataWrangler:
    """Hercodes-ux: Professional Data Reshaping Utility.
    Demonstrates advanced Pandas techniques: Pivot Tables, GroupBy, and Lambda transformations.
    
    """
    def __init__(self, df):
        self.df = df


    
    def get_business_summary(self,index_col, values_col):
         # Using GroupBy to find counts and averages simultaneously
        summary= self.df.groupby(index_col)[values_col].agg(['count','mean','sum']).reset_index()
        return summary
    
    def create_pivot_report(self, rows, columns, values):
         # Professional Pivot Table for multi-dimensional analysis
        pivot = pd.pivot_table(self.df,values=values, index=rows, columns=columns, aggfunc='sum', fill_value=0)
        return pivot
    
    def apply_custom_logic(self, col_name):
        # Demonstrating Lambda functions for feature transformation
        median_value = self.df[col_name].median()
        self.df['Performance_Tier'] = self.df[col_name].apply(lambda x: 'High' if x>= median_value else 'Low')
        return self.df




#Execution Block
if __name__ == "__main__" :
    #Creating a messay dataset
    data ={
        'Store' : ['North', 'North', 'South', 'South', 'East', 'East', 'West', 'West'],
        'Product' : ['Tablet', 'Laptop', 'Tablet', 'Laptop', 'Tablet', 'Laptop', 'Tablet', 'Laptop'],
        'Sales' : [1200,450,110,300,1500,600,900,200],
        'Units_Sold' : [10,15,8,12,20,25,5,10]
    }
    df = pd.DataFrame(data)


    #Initialize the DataWrangler
    wrangler = DataWrangler(df)
    print("--- 📊 DATA WRANGLING REPORT: HERSHINE ANALYTICS ---")

    #Test: Business Summary
    print("\n[Step 1] Business Summary by Store:")
    summary = wrangler.get_business_summary('Store', 'Sales')
    print(summary)

    #Test: Pivot Report
    print("\n[Step 2] Cross-Tab Pivot (Store vs Product):")
    pivot =wrangler.create_pivot_report('Store', 'Product', 'Sales')
    print(pivot)

    #Test: Lambda Transformation
    print("\n[Step 3] Categorized Performance Tiers:")
    final_df = wrangler.apply_custom_logic('Sales')
    print(final_df[['Store', 'Product', 'Sales', 'Performance_Tier']])
