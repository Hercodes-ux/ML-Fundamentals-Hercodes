
import pandas as pd
from data_wrangler import DataWrangler

def test_summary_logic():
    # Create a tiny test dataset
    df = pd.DataFrame({'Store': ['A', 'A'], 'Sales': [10, 20]})
    wrangler = DataWrangler(df)
    summary = wrangler.get_business_summary('Store', 'Sales')
    assert summary.iloc[0]['sum'] == 30 # Check if the math is correct
