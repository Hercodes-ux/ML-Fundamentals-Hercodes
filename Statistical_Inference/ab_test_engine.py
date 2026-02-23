import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from statsmodels.stats.proportion import proportions_ztest
from statsmodels.stats.power import TTestIndPower

class NetflixExperimentSuite:
    def __init__(self, alpha=0.05, power=0.8):
        self.alpha = alpha
        self.power = power

    def calculate_sample_needed(self, effect_size=0.1):
        """[PHASE 1: DESIGN] Pre-experiment sizing."""
        analysis = TTestIndPower()
        sample_size = analysis.solve_power(effect_size=effect_size, power=self.power, alpha=self.alpha, ratio=1.0)
        return int(np.ceil(sample_size))

    def run_frequentist_test(self, count_a, total_a, count_b, total_b):
        """[PHASE 2: EXECUTION] Standard Z-test."""
        z_stat, p_val = proportions_ztest([count_a, count_b], [total_a, total_b])
        return p_val

    def run_bayesian_test(self, count_a, total_a, count_b, total_b):
        """[PHASE 3: INTERPRETATION] Bayesian Probability of Success."""
        samples_a = np.random.beta(count_a + 1, (total_a - count_a) + 1, 10000)
        samples_b = np.random.beta(count_b + 1, (total_b - count_b) + 1, 10000)
        return (samples_b > samples_a).mean()

    def plot_results(self, count_a, total_a, count_b, total_b):
        """[PHASE 4: VISUALIZATION] Bayesian Density Mapping."""
        x = np.linspace(0.05, 0.25, 1000)
        y_a = stats.beta.pdf(x, count_a + 1, (total_a - count_a) + 1)
        y_b = stats.beta.pdf(x, count_b + 1, (total_b - count_b) + 1)
        
        plt.figure(figsize=(12, 6))
        plt.plot(x, y_a, label='Control (Current UI)', color='blue', lw=3)
        plt.plot(x, y_b, label='Variant (New Feature)', color='red', lw=3)
        plt.fill_between(x, 0, y_a, color='blue', alpha=0.1)
        plt.fill_between(x, 0, y_b, color='red', alpha=0.1)
        plt.title('Hercodes-ux: Bayesian Probability Density of Conversion Rates', fontsize=15)
        plt.xlabel('Conversion Rate (%)')
        plt.ylabel('Confidence Density')
        plt.legend()
        plt.savefig("experiment_results_distribution.png")
        plt.show()

if __name__ == "__main__":
    suite = NetflixExperimentSuite()
    # Data: 1000 users each, 120 clicks vs 155 clicks
    ca, ta, cb, tb = 120, 1000, 155, 1000
    
    print(f"🎯 P-Value: {suite.run_frequentist_test(ca, ta, cb, tb):.4f}")
    print(f"📊 Bayesian Confidence: {suite.run_bayesian_test(ca, ta, cb, tb):.2%}")
    suite.plot_results(ca, ta, cb, tb)