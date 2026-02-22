import pandas as pd
import numpy as np


class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()

    def handle_missing_values(self):
        # Fill missing numerical with median, categorical with mode (if available)
        for col in self.df.columns:
            if np.issubdtype(self.df[col].dtype, np.number):
                median_value = self.df[col].median()
                self.df[col] = self.df[col].fillna(median_value)
            else:
                mode_series = self.df[col].mode()
                if not mode_series.empty:
                    fill_value = mode_series[0]
                else:
                    fill_value = ""
                self.df[col] = self.df[col].fillna(fill_value)
        return self.df





if __name__ == "__main__":
    #creating a messy dataset
    data={
        'Sales':[100, 200, np.nan, 500, 459, np.nan],
        'Products':['Amazon', 'Google', 'Samsung', np.nan, 'Open-AI', 'Microsoft']

    }
    messydata_df=pd.DataFrame(data)

    print("----Messy dataset before cleaning----")
    print(messydata_df)

    cleaner = DataCleaner(messydata_df)
    clean_df =cleaner.handle_missing_values()
    print("----Cleaned dataset after handling missing values----")
    print(clean_df)
    