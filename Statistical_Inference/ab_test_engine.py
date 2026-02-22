import numpy as np
from statsmodels.stats.proportion import proportions_ztest # FIXED THIS
import pandas as pd

def run_ab_test(control_clicks, control_total, variant_clicks, variant_total):
    """
    Hercodes-ux: Statistical Significance Engine.
    Used to determine if a UI change truly impacts user behavior.
    """
    # 1. Calculate Conversion Rates
    rate_a = control_clicks / control_total
    rate_b = variant_clicks / variant_total
    
    # 2. Perform Z-Test
    # We pass the number of successes and the total trials
    z_score, p_value = proportions_ztest([control_clicks, variant_clicks], 
                                         [control_total, variant_total])
    
    print(f"--- 📈 Experiment Results ---")
    print(f"Control Conversion: {rate_a:.2%}")
    print(f"Variant Conversion: {rate_b:.2%}")
    print(f"P-Value: {p_value:.4f}")
    
    if p_value < 0.05:
        print("✅ Result: Statistically Significant. Deploy the change!")
    else:
        print("❌ Result: Not Significant. The difference is likely due to noise.")

# --- SIMULATION ---
if __name__ == "__main__":
    # Control: 1000 users, 100 clicks (10%)
    # Variant: 1000 users, 135 clicks (13.5%)
    run_ab_test(100, 1000, 135, 1000)