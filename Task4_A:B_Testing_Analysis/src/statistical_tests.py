"""
Statistical testing module for A/B Testing Analysis
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import ttest_ind, chi2_contingency, mannwhitneyu, beta, norm
import statsmodels.stats.power as smp
from statsmodels.stats.proportion import proportions_ztest
from statsmodels.stats.multitest import multipletests
import warnings
warnings.filterwarnings('ignore')

class ABTestStatistics:
    """Class for performing statistical tests in A/B testing"""

    def __init__(self, alpha=0.05):
        self.alpha = alpha
        self.results = {}

    def cohens_d(self, group1, group2):
        """Calculate Cohen's d effect size"""
        n1, n2 = len(group1), len(group2)
        s1, s2 = np.std(group1, ddof=1), np.std(group2, ddof=1)

        # Pooled standard deviation
        pooled_std = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))

        # Cohen's d
        d = (np.mean(group1) - np.mean(group2)) / pooled_std
        return d

    def interpret_effect_size(self, d):
        """Interpret Cohen's d effect size"""
        abs_d = abs(d)
        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"

    def t_test(self, control_group, treatment_group, test_name="T-Test"):
        """Perform independent t-test"""
        # Remove any NaN values
        control_clean = control_group.dropna()
        treatment_clean = treatment_group.dropna()

        # Perform t-test
        t_stat, p_value = ttest_ind(control_clean, treatment_clean)

        # Calculate effect size
        effect_size = self.cohens_d(control_clean, treatment_clean)

        # Calculate confidence interval for difference in means
        n1, n2 = len(control_clean), len(treatment_clean)
        mean_diff = np.mean(treatment_clean) - np.mean(control_clean)

        # Pooled standard error
        s1, s2 = np.std(control_clean, ddof=1), np.std(treatment_clean, ddof=1)
        pooled_se = np.sqrt(s1**2/n1 + s2**2/n2)

        # Degrees of freedom
        df = n1 + n2 - 2
        t_critical = stats.t.ppf(1 - self.alpha/2, df)

        ci_lower = mean_diff - t_critical * pooled_se
        ci_upper = mean_diff + t_critical * pooled_se

        result = {
            'test_name': test_name,
            'test_type': 'Independent T-Test',
            't_statistic': t_stat,
            'p_value': p_value,
            'degrees_of_freedom': df,
            'effect_size_cohens_d': effect_size,
            'effect_size_interpretation': self.interpret_effect_size(effect_size),
            'mean_control': np.mean(control_clean),
            'mean_treatment': np.mean(treatment_clean),
            'mean_difference': mean_diff,
            'confidence_interval_95': (ci_lower, ci_upper),
            'significant': p_value < self.alpha,
            'sample_size_control': n1,
            'sample_size_treatment': n2
        }

        self.results[test_name] = result
        return result

    def mann_whitney_u_test(self, control_group, treatment_group, test_name="Mann-Whitney U Test"):
        """Perform Mann-Whitney U test (non-parametric alternative to t-test)"""
        control_clean = control_group.dropna()
        treatment_clean = treatment_group.dropna()

        u_stat, p_value = mannwhitneyu(control_clean, treatment_clean, alternative='two-sided')

        # Calculate effect size (r = Z / sqrt(N))
        n1, n2 = len(control_clean), len(treatment_clean)
        n_total = n1 + n2
        z_score = stats.norm.ppf(1 - p_value/2)  # Approximate z-score
        effect_size_r = z_score / np.sqrt(n_total)

        result = {
            'test_name': test_name,
            'test_type': 'Mann-Whitney U Test',
            'u_statistic': u_stat,
            'p_value': p_value,
            'effect_size_r': effect_size_r,
            'median_control': np.median(control_clean),
            'median_treatment': np.median(treatment_clean),
            'significant': p_value < self.alpha,
            'sample_size_control': n1,
            'sample_size_treatment': n2
        }

        self.results[test_name] = result
        return result

    def chi_square_test(self, contingency_table, test_name="Chi-Square Test"):
        """Perform Chi-square test for categorical data"""
        chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)

        # Calculate Cramér's V (effect size for chi-square)
        n = contingency_table.sum().sum()
        cramers_v = np.sqrt(chi2_stat / (n * (min(contingency_table.shape) - 1)))

        result = {
            'test_name': test_name,
            'test_type': 'Chi-Square Test',
            'chi2_statistic': chi2_stat,
            'p_value': p_value,
            'degrees_of_freedom': dof,
            'effect_size_cramers_v': cramers_v,
            'expected_frequencies': expected,
            'significant': p_value < self.alpha
        }

        self.results[test_name] = result
        return result

    def proportion_test(self, successes_control, n_control, successes_treatment, n_treatment,
                       test_name="Proportion Test"):
        """Perform two-proportion z-test"""
        counts = np.array([successes_control, successes_treatment])
        nobs = np.array([n_control, n_treatment])

        z_stat, p_value = proportions_ztest(counts, nobs)

        # Calculate proportions
        prop_control = successes_control / n_control
        prop_treatment = successes_treatment / n_treatment
        prop_diff = prop_treatment - prop_control

        # Calculate effect size (Cohen's h)
        h = 2 * (np.arcsin(np.sqrt(prop_treatment)) - np.arcsin(np.sqrt(prop_control)))

        result = {
            'test_name': test_name,
            'test_type': 'Two-Proportion Z-Test',
            'z_statistic': z_stat,
            'p_value': p_value,
            'proportion_control': prop_control,
            'proportion_treatment': prop_treatment,
            'proportion_difference': prop_diff,
            'effect_size_cohens_h': h,
            'significant': p_value < self.alpha,
            'sample_size_control': n_control,
            'sample_size_treatment': n_treatment
        }

        self.results[test_name] = result
        return result

    def power_analysis(self, effect_size, sample_size=None, power=None, alpha=None):
        """Perform power analysis for t-test"""
        if alpha is None:
            alpha = self.alpha

        if sample_size is None and power is not None:
            # Calculate required sample size
            try:
                # Handle edge cases for effect size
                if abs(effect_size) < 0.01:
                    return {'required_sample_size_per_group': 'Effect size too small (< 0.01)'}

                sample_size_calc = smp.ttest_power(abs(effect_size), power, alpha, alternative='two-sided')
                if np.isnan(sample_size_calc) or np.isinf(sample_size_calc) or sample_size_calc <= 0:
                    return {'required_sample_size_per_group': 'Unable to calculate (effect size too small)'}
                if sample_size_calc > 100000:
                    return {'required_sample_size_per_group': 'Very large sample required (>100,000)'}
                return {'required_sample_size_per_group': int(np.ceil(sample_size_calc))}
            except Exception as e:
                return {'required_sample_size_per_group': f'Error: {str(e)}'}

        elif power is None and sample_size is not None:
            # Calculate achieved power
            try:
                # Handle edge cases for effect size
                if abs(effect_size) < 0.01:
                    return {'achieved_power': 'Effect size too small (< 0.01)'}

                power_calc = smp.ttest_power(abs(effect_size), sample_size, alpha, alternative='two-sided')
                if np.isnan(power_calc) or np.isinf(power_calc):
                    return {'achieved_power': 'Unable to calculate'}
                return {'achieved_power': float(power_calc)}
            except Exception as e:
                return {'achieved_power': f'Error: {str(e)}'}

        else:
            return {'error': 'Specify either sample_size or power, not both'}

    def bayesian_ab_test(self, control_successes, control_total, treatment_successes, treatment_total,
                        prior_alpha=1, prior_beta=1, test_name="Bayesian A/B Test"):
        """Perform Bayesian A/B test for conversion rates"""
        # Posterior distributions
        control_posterior_alpha = prior_alpha + control_successes
        control_posterior_beta = prior_beta + control_total - control_successes

        treatment_posterior_alpha = prior_alpha + treatment_successes
        treatment_posterior_beta = prior_beta + treatment_total - treatment_successes

        # Sample from posterior distributions
        n_samples = 100000
        control_samples = beta.rvs(control_posterior_alpha, control_posterior_beta, size=n_samples)
        treatment_samples = beta.rvs(treatment_posterior_alpha, treatment_posterior_beta, size=n_samples)

        # Calculate probability that treatment > control
        prob_treatment_better = np.mean(treatment_samples > control_samples)

        # Calculate credible intervals
        control_ci = np.percentile(control_samples, [2.5, 97.5])
        treatment_ci = np.percentile(treatment_samples, [2.5, 97.5])

        # Calculate expected lift
        expected_lift = np.mean((treatment_samples - control_samples) / control_samples)
        lift_ci = np.percentile((treatment_samples - control_samples) / control_samples, [2.5, 97.5])

        result = {
            'test_name': test_name,
            'test_type': 'Bayesian A/B Test',
            'control_rate': control_successes / control_total,
            'treatment_rate': treatment_successes / treatment_total,
            'prob_treatment_better': prob_treatment_better,
            'control_credible_interval': control_ci,
            'treatment_credible_interval': treatment_ci,
            'expected_lift': expected_lift,
            'lift_credible_interval': lift_ci,
            'control_posterior': (control_posterior_alpha, control_posterior_beta),
            'treatment_posterior': (treatment_posterior_alpha, treatment_posterior_beta)
        }

        self.results[test_name] = result
        return result

    def bootstrap_test(self, control_group, treatment_group, n_bootstrap=10000, test_name="Bootstrap Test"):
        """Perform bootstrap hypothesis test"""
        control_clean = control_group.dropna()
        treatment_clean = treatment_group.dropna()

        # Observed difference
        observed_diff = np.mean(treatment_clean) - np.mean(control_clean)

        # Bootstrap under null hypothesis (no difference)
        combined = np.concatenate([control_clean, treatment_clean])
        n_control = len(control_clean)
        n_treatment = len(treatment_clean)

        bootstrap_diffs = []
        for _ in range(n_bootstrap):
            # Resample under null
            resampled = np.random.choice(combined, size=len(combined), replace=True)
            boot_control = resampled[:n_control]
            boot_treatment = resampled[n_control:]
            bootstrap_diffs.append(np.mean(boot_treatment) - np.mean(boot_control))

        bootstrap_diffs = np.array(bootstrap_diffs)

        # Calculate p-value (two-tailed)
        p_value = 2 * min(np.mean(bootstrap_diffs >= observed_diff),
                         np.mean(bootstrap_diffs <= observed_diff))

        # Bootstrap confidence interval for the difference
        ci_lower, ci_upper = np.percentile(bootstrap_diffs, [2.5, 97.5])

        result = {
            'test_name': test_name,
            'test_type': 'Bootstrap Test',
            'observed_difference': observed_diff,
            'p_value': p_value,
            'bootstrap_ci_95': (ci_lower, ci_upper),
            'significant': p_value < self.alpha,
            'n_bootstrap': n_bootstrap,
            'sample_size_control': n_control,
            'sample_size_treatment': n_treatment
        }

        self.results[test_name] = result
        return result

    def multiple_testing_correction(self, method='bonferroni'):
        """Apply multiple testing correction to all p-values"""
        if not self.results:
            return "No test results available for correction"

        # Extract p-values
        p_values = []
        test_names = []
        for test_name, result in self.results.items():
            if 'p_value' in result and isinstance(result['p_value'], (int, float)):
                p_values.append(result['p_value'])
                test_names.append(test_name)

        if not p_values:
            return "No valid p-values found for correction"

        # Apply correction
        rejected, p_corrected, alpha_sidak, alpha_bonf = multipletests(
            p_values, alpha=self.alpha, method=method
        )

        # Update results
        correction_results = {}
        for i, test_name in enumerate(test_names):
            correction_results[test_name] = {
                'original_p_value': p_values[i],
                'corrected_p_value': p_corrected[i],
                'significant_after_correction': rejected[i],
                'correction_method': method
            }

            # Update original result
            if test_name in self.results:
                self.results[test_name]['corrected_p_value'] = p_corrected[i]
                self.results[test_name]['significant_after_correction'] = rejected[i]

        return {
            'correction_method': method,
            'alpha_bonferroni': alpha_bonf,
            'alpha_sidak': alpha_sidak,
            'results': correction_results
        }

    def sequential_testing(self, control_group, treatment_group, alpha_spending_func='obrien_fleming',
                          max_looks=5, test_name="Sequential Test"):
        """Perform sequential A/B testing with alpha spending"""
        control_clean = control_group.dropna()
        treatment_clean = treatment_group.dropna()

        n_control = len(control_clean)
        n_treatment = len(treatment_clean)
        min_n = min(n_control, n_treatment)

        # Define look points
        look_points = np.linspace(int(min_n * 0.2), min_n, max_looks).astype(int)

        results_by_look = []
        cumulative_alpha = 0

        for i, n in enumerate(look_points):
            # Current data
            current_control = control_clean[:n]
            current_treatment = treatment_clean[:n]

            # Perform t-test
            t_stat, p_value = ttest_ind(current_control, current_treatment)

            # Alpha spending (simplified O'Brien-Fleming)
            if alpha_spending_func == 'obrien_fleming':
                alpha_spent = self.alpha * (2 * (1 - norm.cdf(2.963 / np.sqrt((i + 1) / max_looks))))
            else:  # Pocock
                alpha_spent = self.alpha * np.log(1 + (i + 1) * (np.exp(1) - 1) / max_looks)

            cumulative_alpha += alpha_spent

            look_result = {
                'look_number': i + 1,
                'sample_size': n,
                'p_value': p_value,
                'alpha_spent': alpha_spent,
                'cumulative_alpha': cumulative_alpha,
                'significant': p_value < alpha_spent,
                'mean_control': np.mean(current_control),
                'mean_treatment': np.mean(current_treatment)
            }

            results_by_look.append(look_result)

            # Early stopping if significant
            if p_value < alpha_spent:
                break

        result = {
            'test_name': test_name,
            'test_type': 'Sequential Test',
            'alpha_spending_function': alpha_spending_func,
            'max_looks': max_looks,
            'looks_performed': len(results_by_look),
            'final_significant': results_by_look[-1]['significant'],
            'total_alpha_spent': cumulative_alpha,
            'results_by_look': results_by_look
        }

        self.results[test_name] = result
        return result

    def summary_report(self):
        """Generate a summary report of all tests performed"""
        if not self.results:
            return "No tests have been performed yet."

        report = "=== A/B TESTING STATISTICAL ANALYSIS SUMMARY ===\n\n"

        for test_name, result in self.results.items():
            report += f"Test: {test_name}\n"
            report += f"Type: {result.get('test_type', 'Unknown')}\n"

            p_value = result.get('p_value', 'N/A')
            if isinstance(p_value, (int, float)):
                report += f"P-value: {p_value:.6f}\n"
            else:
                report += f"P-value: {p_value}\n"

            report += f"Significant: {'Yes' if result.get('significant', False) else 'No'} (α = {self.alpha})\n"

            if 'effect_size_cohens_d' in result:
                report += f"Effect Size (Cohen's d): {result['effect_size_cohens_d']:.4f} ({result['effect_size_interpretation']})\n"
            elif 'effect_size_cramers_v' in result:
                report += f"Effect Size (Cramér's V): {result['effect_size_cramers_v']:.4f}\n"
            elif 'effect_size_cohens_h' in result:
                report += f"Effect Size (Cohen's h): {result['effect_size_cohens_h']:.4f}\n"

            report += "-" * 50 + "\n\n"

        return report

    def get_results_dataframe(self):
        """Convert results to a pandas DataFrame"""
        if not self.results:
            return pd.DataFrame()

        summary_data = []
        for test_name, result in self.results.items():
            summary_row = {
                'Test_Name': test_name,
                'Test_Type': result.get('test_type', 'Unknown'),
                'P_Value': result.get('p_value', 'N/A'),
                'Significant': result.get('significant', False)
            }

            # Add effect size information
            if 'effect_size_cohens_d' in result:
                summary_row['Effect_Size'] = result['effect_size_cohens_d']
                summary_row['Effect_Size_Type'] = "Cohen's d"
                summary_row['Effect_Size_Interpretation'] = result['effect_size_interpretation']
            elif 'effect_size_cramers_v' in result:
                summary_row['Effect_Size'] = result['effect_size_cramers_v']
                summary_row['Effect_Size_Type'] = "Cramér's V"
            elif 'effect_size_cohens_h' in result:
                summary_row['Effect_Size'] = result['effect_size_cohens_h']
                summary_row['Effect_Size_Type'] = "Cohen's h"

            summary_data.append(summary_row)

        return pd.DataFrame(summary_data)
