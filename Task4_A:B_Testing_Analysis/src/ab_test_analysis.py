"""
Main A/B Testing Analysis Module for Wine Dataset
"""

import pandas as pd
import numpy as np
from scipy import stats
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import *
from src.data_preprocessing import WineDataPreprocessor
from src.statistical_tests import ABTestStatistics
from src.visualization import ABTestVisualizer

class WineABTestAnalyzer:
    """Main class for conducting A/B testing analysis on wine dataset"""

    def __init__(self, alpha=ALPHA):
        self.preprocessor = WineDataPreprocessor()
        self.stats_analyzer = ABTestStatistics(alpha=alpha)
        self.visualizer = None  # Will be initialized when needed
        self.data = None
        self.results = {}

    def load_and_prepare_data(self):
        """Load and prepare data for analysis"""
        print("Loading and preparing wine dataset...")

        # Load data
        self.data = self.preprocessor.load_data()
        if self.data is None:
            raise ValueError("Failed to load data")

        # Explore data
        exploration_results = self.preprocessor.explore_data()

        # Clean data
        self.data = self.preprocessor.clean_data()

        # Save exploration results
        self.results['data_exploration'] = exploration_results

        return self.data

    def create_ab_test_scenario(self, feature, threshold_percentile=50, scenario_name=None):
        """Create A/B test scenario based on a feature"""
        if scenario_name is None:
            scenario_name = f"{feature}_ab_test"

        print(f"\nCreating A/B test scenario: {scenario_name}")
        print(f"Feature: {feature}, Threshold percentile: {threshold_percentile}")

        # Create A/B groups
        self.data = self.preprocessor.create_ab_test_groups(
            feature=feature,
            threshold_percentile=threshold_percentile
        )

        # Store scenario information
        self.results[scenario_name] = {
            'feature': feature,
            'threshold_percentile': threshold_percentile,
            'threshold_value': np.percentile(self.data[feature], threshold_percentile)
        }

        return self.data

    def analyze_continuous_metric(self, metric, scenario_name, test_type='t_test'):
        """Analyze continuous metric between A/B groups"""
        if 'ab_group' not in self.data.columns:
            raise ValueError("A/B groups not created. Run create_ab_test_scenario first.")

        print(f"\nAnalyzing continuous metric: {metric}")

        # Split data by groups
        control_data = self.data[self.data['ab_group'] == 'Control'][metric]
        treatment_data = self.data[self.data['ab_group'] == 'Treatment'][metric]

        # Perform statistical test
        if test_type == 't_test':
            test_result = self.stats_analyzer.t_test(
                control_data, treatment_data,
                test_name=f"{scenario_name}_{metric}_ttest"
            )
        elif test_type == 'mann_whitney':
            test_result = self.stats_analyzer.mann_whitney_u_test(
                control_data, treatment_data,
                test_name=f"{scenario_name}_{metric}_mannwhitney"
            )
        else:
            raise ValueError("test_type must be 't_test' or 'mann_whitney'")

        # Store results
        if scenario_name not in self.results:
            self.results[scenario_name] = {}
        self.results[scenario_name][f'{metric}_analysis'] = test_result

        return test_result

    def analyze_categorical_metric(self, metric, scenario_name):
        """Analyze categorical metric between A/B groups"""
        if 'ab_group' not in self.data.columns:
            raise ValueError("A/B groups not created. Run create_ab_test_scenario first.")

        print(f"\nAnalyzing categorical metric: {metric}")

        # Create contingency table
        contingency_table = pd.crosstab(self.data['ab_group'], self.data[metric])

        # Perform chi-square test
        test_result = self.stats_analyzer.chi_square_test(
            contingency_table.values,
            test_name=f"{scenario_name}_{metric}_chisquare"
        )

        # Store results
        if scenario_name not in self.results:
            self.results[scenario_name] = {}
        self.results[scenario_name][f'{metric}_analysis'] = test_result
        self.results[scenario_name][f'{metric}_contingency_table'] = contingency_table

        return test_result

    def run_comprehensive_analysis(self, primary_feature='alcohol', metrics_to_analyze=None):
        """Run comprehensive A/B testing analysis"""
        if metrics_to_analyze is None:
            # Analyze key wine characteristics
            metrics_to_analyze = [
                'total_phenols', 'flavanoids', 'color_intensity',
                'hue', 'proline', 'malic_acid'
            ]

        scenario_name = f"{primary_feature}_comprehensive"

        print("=== COMPREHENSIVE A/B TESTING ANALYSIS ===")

        # Create A/B test scenario
        self.create_ab_test_scenario(primary_feature, scenario_name=scenario_name)

        # Analyze each metric
        analysis_results = {}
        for metric in metrics_to_analyze:
            if metric in self.data.columns:
                try:
                    # Check if metric is continuous or categorical
                    if self.data[metric].dtype in ['int64', 'float64'] and self.data[metric].nunique() > 10:
                        # Continuous metric - use t-test
                        result = self.analyze_continuous_metric(metric, scenario_name, 't_test')
                        analysis_results[metric] = result
                    else:
                        # Categorical metric - use chi-square test
                        result = self.analyze_categorical_metric(metric, scenario_name)
                        analysis_results[metric] = result

                except Exception as e:
                    print(f"Error analyzing {metric}: {e}")
                    continue

        # Store comprehensive results
        self.results[scenario_name]['comprehensive_analysis'] = analysis_results

        return analysis_results

    def perform_power_analysis(self, effect_sizes=[0.2, 0.5, 0.8], sample_sizes=[50, 100, 200, 500]):
        """Perform power analysis for different effect sizes and sample sizes"""
        print("\n=== POWER ANALYSIS ===")

        power_results = {}

        for effect_size in effect_sizes:
            power_results[f'effect_size_{effect_size}'] = {}

            for sample_size in sample_sizes:
                power_result = self.stats_analyzer.power_analysis(
                    effect_size=effect_size,
                    sample_size=sample_size
                )
                power_results[f'effect_size_{effect_size}'][f'n_{sample_size}'] = power_result

        # Also calculate required sample sizes for 80% power
        required_samples = {}
        for effect_size in effect_sizes:
            try:
                sample_result = self.stats_analyzer.power_analysis(
                    effect_size=effect_size,
                    power=0.8
                )
                required_samples[f'effect_size_{effect_size}'] = sample_result
            except Exception as e:
                print(f"Warning: Could not calculate sample size for effect size {effect_size}: {e}")
                required_samples[f'effect_size_{effect_size}'] = {
                    'required_sample_size_per_group': f'Error: {str(e)}'
                }

        power_results['required_sample_sizes_80_power'] = required_samples
        self.results['power_analysis'] = power_results

        return power_results

    def run_bayesian_analysis(self, metric, scenario_name, prior_alpha=1, prior_beta=1):
        """Run Bayesian A/B test analysis"""
        if 'ab_group' not in self.data.columns:
            raise ValueError("A/B groups not created. Run create_ab_test_scenario first.")

        print(f"\nRunning Bayesian analysis for: {metric}")

        # Convert continuous metric to binary (above/below median)
        median_value = self.data[metric].median()
        control_data = self.data[self.data['ab_group'] == 'Control']
        treatment_data = self.data[self.data['ab_group'] == 'Treatment']

        control_successes = sum(control_data[metric] > median_value)
        control_total = len(control_data)
        treatment_successes = sum(treatment_data[metric] > median_value)
        treatment_total = len(treatment_data)

        # Perform Bayesian test
        bayesian_result = self.stats_analyzer.bayesian_ab_test(
            control_successes, control_total, treatment_successes, treatment_total,
            prior_alpha, prior_beta, f"{scenario_name}_{metric}_bayesian"
        )

        # Store results
        if scenario_name not in self.results:
            self.results[scenario_name] = {}
        self.results[scenario_name][f'{metric}_bayesian'] = bayesian_result

        return bayesian_result

    def run_bootstrap_analysis(self, metric, scenario_name, n_bootstrap=10000):
        """Run bootstrap A/B test analysis"""
        if 'ab_group' not in self.data.columns:
            raise ValueError("A/B groups not created. Run create_ab_test_scenario first.")

        print(f"\nRunning bootstrap analysis for: {metric}")

        control_data = self.data[self.data['ab_group'] == 'Control'][metric]
        treatment_data = self.data[self.data['ab_group'] == 'Treatment'][metric]

        # Perform bootstrap test
        bootstrap_result = self.stats_analyzer.bootstrap_test(
            control_data, treatment_data, n_bootstrap, f"{scenario_name}_{metric}_bootstrap"
        )

        # Store results
        if scenario_name not in self.results:
            self.results[scenario_name] = {}
        self.results[scenario_name][f'{metric}_bootstrap'] = bootstrap_result

        return bootstrap_result

    def run_sequential_analysis(self, metric, scenario_name, max_looks=5):
        """Run sequential A/B test analysis"""
        if 'ab_group' not in self.data.columns:
            raise ValueError("A/B groups not created. Run create_ab_test_scenario first.")

        print(f"\nRunning sequential analysis for: {metric}")

        control_data = self.data[self.data['ab_group'] == 'Control'][metric]
        treatment_data = self.data[self.data['ab_group'] == 'Treatment'][metric]

        # Perform sequential test
        sequential_result = self.stats_analyzer.sequential_testing(
            control_data, treatment_data, max_looks=max_looks,
            test_name=f"{scenario_name}_{metric}_sequential"
        )

        # Store results
        if scenario_name not in self.results:
            self.results[scenario_name] = {}
        self.results[scenario_name][f'{metric}_sequential'] = sequential_result

        return sequential_result

    def run_advanced_comprehensive_analysis(self, primary_feature='alcohol', metrics_to_analyze=None):
        """Run comprehensive analysis with all advanced methods"""
        if metrics_to_analyze is None:
            metrics_to_analyze = ['total_phenols', 'flavanoids', 'color_intensity']

        scenario_name = f"{primary_feature}_advanced"

        print("=== ADVANCED COMPREHENSIVE A/B TESTING ANALYSIS ===")

        # Create A/B test scenario
        self.create_ab_test_scenario(primary_feature, scenario_name=scenario_name)

        # Run all types of analysis for each metric
        advanced_results = {}
        for metric in metrics_to_analyze:
            if metric in self.data.columns:
                print(f"\n--- Analyzing {metric} ---")

                try:
                    # Traditional t-test
                    ttest_result = self.analyze_continuous_metric(metric, scenario_name, 't_test')

                    # Bayesian analysis
                    bayesian_result = self.run_bayesian_analysis(metric, scenario_name)

                    # Bootstrap analysis
                    bootstrap_result = self.run_bootstrap_analysis(metric, scenario_name)

                    # Sequential analysis
                    sequential_result = self.run_sequential_analysis(metric, scenario_name)

                    advanced_results[metric] = {
                        'ttest': ttest_result,
                        'bayesian': bayesian_result,
                        'bootstrap': bootstrap_result,
                        'sequential': sequential_result
                    }

                except Exception as e:
                    print(f"Error analyzing {metric}: {e}")
                    continue

        # Apply multiple testing correction
        print("\n--- Applying Multiple Testing Correction ---")
        correction_result = self.stats_analyzer.multiple_testing_correction(method='bonferroni')

        # Store comprehensive results
        self.results[scenario_name]['advanced_analysis'] = advanced_results
        self.results[scenario_name]['multiple_testing_correction'] = correction_result

        return advanced_results

    def calculate_minimum_detectable_effect(self, baseline_mean, baseline_std, sample_size_per_group,
                                          alpha=0.05, power=0.8):
        """Calculate minimum detectable effect size"""
        from scipy.stats import norm

        # Critical values
        z_alpha = norm.ppf(1 - alpha/2)
        z_beta = norm.ppf(power)

        # Pooled standard error
        pooled_se = baseline_std * np.sqrt(2 / sample_size_per_group)

        # Minimum detectable effect
        mde = (z_alpha + z_beta) * pooled_se

        # Relative MDE
        relative_mde = mde / baseline_mean

        return {
            'absolute_mde': mde,
            'relative_mde': relative_mde,
            'baseline_mean': baseline_mean,
            'baseline_std': baseline_std,
            'sample_size_per_group': sample_size_per_group,
            'alpha': alpha,
            'power': power
        }

    def simulate_ab_test(self, effect_size, sample_size_per_group, n_simulations=1000):
        """Simulate A/B test to estimate power and Type I error"""
        print(f"\nSimulating A/B test: effect_size={effect_size}, n_per_group={sample_size_per_group}")

        significant_results = 0
        p_values = []

        for _ in range(n_simulations):
            # Generate data
            control = np.random.normal(0, 1, sample_size_per_group)
            treatment = np.random.normal(effect_size, 1, sample_size_per_group)

            # Perform t-test
            _, p_value = stats.ttest_ind(control, treatment)
            p_values.append(p_value)

            if p_value < 0.05:
                significant_results += 1

        empirical_power = significant_results / n_simulations

        return {
            'effect_size': effect_size,
            'sample_size_per_group': sample_size_per_group,
            'n_simulations': n_simulations,
            'empirical_power': empirical_power,
            'mean_p_value': np.mean(p_values),
            'p_values': p_values
        }

    def generate_insights_and_recommendations(self):
        """Generate actionable insights and recommendations"""
        print("\n=== GENERATING INSIGHTS AND RECOMMENDATIONS ===")

        insights = {
            'summary': {},
            'significant_findings': [],
            'recommendations': [],
            'limitations': []
        }

        # Analyze all statistical test results
        all_tests = self.stats_analyzer.results
        significant_tests = [name for name, result in all_tests.items() if result.get('significant', False)]

        insights['summary'] = {
            'total_tests_performed': len(all_tests),
            'significant_tests': len(significant_tests),
            'significance_rate': len(significant_tests) / len(all_tests) if all_tests else 0
        }

        # Extract significant findings
        for test_name in significant_tests:
            result = all_tests[test_name]
            finding = {
                'test': test_name,
                'p_value': result['p_value'],
                'effect_size': result.get('effect_size_cohens_d', result.get('effect_size_cramers_v', 'N/A')),
                'interpretation': result.get('effect_size_interpretation', 'N/A')
            }
            insights['significant_findings'].append(finding)

        # Generate recommendations
        if significant_tests:
            insights['recommendations'].extend([
                "Focus on features that showed significant differences between groups",
                "Consider the practical significance of effect sizes, not just statistical significance",
                "Validate findings with additional data collection or replication studies",
                "Implement changes gradually and monitor key metrics"
            ])
        else:
            insights['recommendations'].extend([
                "No significant differences found - consider larger sample sizes",
                "Explore different grouping criteria or features",
                "Review data quality and measurement methods",
                "Consider non-parametric tests if data distributions are skewed"
            ])

        # Add limitations
        insights['limitations'] = [
            "Analysis based on observational data, not randomized controlled trial",
            "Multiple testing may increase Type I error rate",
            "Effect sizes should be interpreted in business context",
            "Correlation does not imply causation"
        ]

        self.results['insights_and_recommendations'] = insights
        return insights

    def save_results(self, filename='ab_test_results.csv'):
        """Save analysis results to file"""
        output_path = os.path.join(PROCESSED_DATA_DIR, filename)

        # Create summary DataFrame
        results_df = self.stats_analyzer.get_results_dataframe()

        if not results_df.empty:
            results_df.to_csv(output_path, index=False)
            print(f"Results saved to: {output_path}")
        else:
            print("No results to save.")

        return results_df

    def print_summary_report(self):
        """Print comprehensive summary report"""
        print("\n" + "="*60)
        print("WINE A/B TESTING ANALYSIS - SUMMARY REPORT")
        print("="*60)

        # Print statistical summary
        print(self.stats_analyzer.summary_report())

        # Print insights
        if 'insights_and_recommendations' in self.results:
            insights = self.results['insights_and_recommendations']

            print("INSIGHTS AND RECOMMENDATIONS:")
            print("-" * 30)
            print(f"Total tests performed: {insights['summary']['total_tests_performed']}")
            print(f"Significant findings: {insights['summary']['significant_tests']}")
            print(f"Significance rate: {insights['summary']['significance_rate']:.2%}")

            if insights['significant_findings']:
                print("\nSignificant Findings:")
                for finding in insights['significant_findings']:
                    print(f"- {finding['test']}: p={finding['p_value']:.4f}, effect_size={finding['effect_size']}")

            print("\nRecommendations:")
            for rec in insights['recommendations']:
                print(f"- {rec}")

            print("\nLimitations:")
            for lim in insights['limitations']:
                print(f"- {lim}")

if __name__ == "__main__":
    # Example usage
    analyzer = WineABTestAnalyzer()

    # Load and prepare data
    data = analyzer.load_and_prepare_data()

    # Run comprehensive analysis
    results = analyzer.run_comprehensive_analysis(primary_feature='alcohol')

    # Perform power analysis
    power_results = analyzer.perform_power_analysis()

    # Generate insights
    insights = analyzer.generate_insights_and_recommendations()

    # Save results
    analyzer.save_results()

    # Print summary report
    analyzer.print_summary_report()
