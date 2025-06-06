"""
Visualization module for A/B Testing Analysis
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from scipy import stats
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *

class ABTestVisualizer:
    """Class for creating visualizations for A/B testing analysis"""

    def __init__(self, style=STYLE, figsize=FIGURE_SIZE, dpi=DPI):
        plt.style.use('default')  # Use default style as seaborn-v0_8 might not be available
        self.figsize = figsize
        self.dpi = dpi
        sns.set_palette("husl")

    def plot_group_comparison(self, data, metric, group_col='ab_group', title=None):
        """Create comparison plots between A/B groups"""
        if title is None:
            title = f'Distribution Comparison: {metric}'

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(title, fontsize=16, fontweight='bold')

        # Box plot
        sns.boxplot(data=data, x=group_col, y=metric, ax=axes[0,0])
        axes[0,0].set_title('Box Plot Comparison')
        axes[0,0].grid(True, alpha=0.3)

        # Violin plot
        sns.violinplot(data=data, x=group_col, y=metric, ax=axes[0,1])
        axes[0,1].set_title('Violin Plot Comparison')
        axes[0,1].grid(True, alpha=0.3)

        # Histogram
        for group in data[group_col].unique():
            group_data = data[data[group_col] == group][metric]
            axes[1,0].hist(group_data, alpha=0.7, label=group, bins=20)
        axes[1,0].set_xlabel(metric)
        axes[1,0].set_ylabel('Frequency')
        axes[1,0].set_title('Histogram Comparison')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)

        # Q-Q plot
        from scipy import stats
        control_data = data[data[group_col] == 'Control'][metric].dropna()
        treatment_data = data[data[group_col] == 'Treatment'][metric].dropna()

        # Standardize data for Q-Q plot
        control_std = (control_data - control_data.mean()) / control_data.std()
        treatment_std = (treatment_data - treatment_data.mean()) / treatment_data.std()

        stats.probplot(control_std, dist="norm", plot=axes[1,1])
        axes[1,1].get_lines()[0].set_markerfacecolor('blue')
        axes[1,1].get_lines()[0].set_label('Control')

        stats.probplot(treatment_std, dist="norm", plot=axes[1,1])
        axes[1,1].get_lines()[2].set_markerfacecolor('orange')
        axes[1,1].get_lines()[2].set_label('Treatment')

        axes[1,1].set_title('Q-Q Plot (Normality Check)')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_effect_size_comparison(self, results_dict, title="Effect Size Comparison"):
        """Plot effect sizes from multiple tests"""
        effect_sizes = []
        test_names = []
        p_values = []

        for test_name, result in results_dict.items():
            if 'effect_size_cohens_d' in result:
                effect_sizes.append(abs(result['effect_size_cohens_d']))
                test_names.append(test_name)
                p_values.append(result['p_value'])

        if not effect_sizes:
            print("No effect sizes found to plot")
            return None

        # Create DataFrame for plotting
        plot_data = pd.DataFrame({
            'Test': test_names,
            'Effect_Size': effect_sizes,
            'P_Value': p_values,
            'Significant': ['Yes' if p < 0.05 else 'No' for p in p_values]
        })

        fig, ax = plt.subplots(figsize=self.figsize)

        # Create bar plot with color coding for significance
        colors = ['red' if sig == 'Yes' else 'gray' for sig in plot_data['Significant']]
        bars = ax.bar(plot_data['Test'], plot_data['Effect_Size'], color=colors, alpha=0.7)

        # Add effect size interpretation lines
        ax.axhline(y=0.2, color='green', linestyle='--', alpha=0.5, label='Small Effect (0.2)')
        ax.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='Medium Effect (0.5)')
        ax.axhline(y=0.8, color='red', linestyle='--', alpha=0.5, label='Large Effect (0.8)')

        ax.set_xlabel('Tests')
        ax.set_ylabel('Effect Size (|Cohen\'s d|)')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        return fig

    def plot_power_analysis(self, power_results, title="Power Analysis"):
        """Plot power analysis results"""
        if 'required_sample_sizes_80_power' not in power_results:
            print("No power analysis results found")
            return None

        # Extract required sample sizes with error handling
        effect_sizes = []
        sample_sizes = []
        valid_data = []

        for key, value in power_results['required_sample_sizes_80_power'].items():
            if 'effect_size_' in key:
                try:
                    effect_size = float(key.split('_')[-1])
                    sample_size_raw = value.get('required_sample_size_per_group', 0)

                    # Handle different types of sample size values
                    if isinstance(sample_size_raw, (int, float)) and not np.isnan(sample_size_raw):
                        sample_size = int(sample_size_raw)
                        effect_sizes.append(effect_size)
                        sample_sizes.append(sample_size)
                        valid_data.append((effect_size, sample_size, 'Valid'))
                    elif isinstance(sample_size_raw, str):
                        # Handle string responses like "Unable to calculate"
                        if 'unable' in sample_size_raw.lower() or 'error' in sample_size_raw.lower():
                            effect_sizes.append(effect_size)
                            sample_sizes.append(10000)  # Use a high value for plotting
                            valid_data.append((effect_size, 10000, sample_size_raw))
                        else:
                            try:
                                sample_size = int(sample_size_raw)
                                effect_sizes.append(effect_size)
                                sample_sizes.append(sample_size)
                                valid_data.append((effect_size, sample_size, 'Valid'))
                            except ValueError:
                                # Skip this data point
                                continue
                    else:
                        # Skip invalid data points
                        continue

                except (ValueError, TypeError) as e:
                    print(f"Warning: Could not process effect size {key}: {e}")
                    continue

        if not effect_sizes:
            print("No valid sample size data found for plotting")
            return None

        fig, ax = plt.subplots(figsize=self.figsize)

        # Separate valid and invalid data points
        valid_es = [data[0] for data in valid_data if data[2] == 'Valid']
        valid_ss = [data[1] for data in valid_data if data[2] == 'Valid']
        invalid_es = [data[0] for data in valid_data if data[2] != 'Valid']
        invalid_ss = [data[1] for data in valid_data if data[2] != 'Valid']

        # Plot valid data points
        if valid_es:
            ax.plot(valid_es, valid_ss, 'bo-', linewidth=2, markersize=8, label='Calculable')

            # Add annotations for valid points
            for es, ss in zip(valid_es, valid_ss):
                if ss < 10000:  # Only annotate reasonable sample sizes
                    ax.annotate(f'n={ss}', (es, ss), textcoords="offset points",
                               xytext=(0,10), ha='center')

        # Plot invalid data points differently
        if invalid_es:
            ax.scatter(invalid_es, invalid_ss, color='red', s=100, marker='x',
                      linewidth=3, label='Unable to calculate', zorder=5)

            # Add annotations for invalid points
            for es, ss in zip(invalid_es, invalid_ss):
                ax.annotate('N/A', (es, ss), textcoords="offset points",
                           xytext=(0,10), ha='center', color='red')

        ax.set_xlabel('Effect Size (Cohen\'s d)')
        ax.set_ylabel('Required Sample Size per Group')
        ax.set_title(f'{title}\n(80% Power, α = 0.05)')
        ax.grid(True, alpha=0.3)

        # Set y-axis limit to reasonable range
        if valid_ss:
            max_valid_ss = max([ss for ss in valid_ss if ss < 10000])
            ax.set_ylim(0, min(max_valid_ss * 1.2, 5000))

        # Add legend if we have both types of data
        if valid_es and invalid_es:
            ax.legend()

        # Add explanatory text
        if invalid_es:
            ax.text(0.02, 0.98, 'Note: Some effect sizes too small to calculate reliable sample sizes',
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        plt.tight_layout()
        return fig

    def plot_correlation_matrix(self, data, title="Feature Correlation Matrix"):
        """Plot correlation matrix of features"""
        # Select only numeric columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        correlation_matrix = data[numeric_cols].corr()

        fig, ax = plt.subplots(figsize=(12, 10))

        # Create heatmap
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, ax=ax, fmt='.2f')

        ax.set_title(title)
        plt.tight_layout()
        return fig

    def create_interactive_dashboard(self, data, results_dict):
        """Create interactive dashboard using Plotly"""
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Group Size Comparison', 'Effect Sizes',
                          'P-Value Distribution', 'Sample Statistics'),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "histogram"}, {"type": "table"}]]
        )

        # Group size comparison
        if 'ab_group' in data.columns:
            group_counts = data['ab_group'].value_counts()
            fig.add_trace(
                go.Bar(x=group_counts.index, y=group_counts.values,
                      name="Group Sizes", showlegend=False),
                row=1, col=1
            )

        # Effect sizes
        effect_sizes = []
        test_names = []
        for test_name, result in results_dict.items():
            if 'effect_size_cohens_d' in result:
                effect_sizes.append(abs(result['effect_size_cohens_d']))
                test_names.append(test_name.replace('_', ' ').title())

        if effect_sizes:
            fig.add_trace(
                go.Bar(x=test_names, y=effect_sizes,
                      name="Effect Sizes", showlegend=False),
                row=1, col=2
            )

        # P-value distribution
        p_values = [result['p_value'] for result in results_dict.values()
                   if 'p_value' in result]

        if p_values:
            fig.add_trace(
                go.Histogram(x=p_values, nbinsx=10,
                           name="P-Values", showlegend=False),
                row=2, col=1
            )

        # Sample statistics table
        if 'ab_group' in data.columns:
            numeric_cols = data.select_dtypes(include=[np.number]).columns[:5]  # First 5 numeric columns
            stats_data = []

            for col in numeric_cols:
                control_mean = data[data['ab_group'] == 'Control'][col].mean()
                treatment_mean = data[data['ab_group'] == 'Treatment'][col].mean()
                stats_data.append([col, f"{control_mean:.2f}", f"{treatment_mean:.2f}"])

            fig.add_trace(
                go.Table(
                    header=dict(values=['Feature', 'Control Mean', 'Treatment Mean']),
                    cells=dict(values=list(zip(*stats_data)))
                ),
                row=2, col=2
            )

        fig.update_layout(
            title_text="A/B Testing Analysis Dashboard",
            showlegend=False,
            height=800
        )

        return fig

    def plot_bayesian_results(self, bayesian_result, title="Bayesian A/B Test Results"):
        """Plot Bayesian A/B test results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(title, fontsize=16, fontweight='bold')

        # Extract posterior parameters
        control_alpha, control_beta = bayesian_result['control_posterior']
        treatment_alpha, treatment_beta = bayesian_result['treatment_posterior']

        # Generate samples for plotting
        x = np.linspace(0, 1, 1000)
        control_pdf = stats.beta.pdf(x, control_alpha, control_beta)
        treatment_pdf = stats.beta.pdf(x, treatment_alpha, treatment_beta)

        # Plot posterior distributions
        axes[0,0].plot(x, control_pdf, label='Control', color='blue', linewidth=2)
        axes[0,0].plot(x, treatment_pdf, label='Treatment', color='red', linewidth=2)
        axes[0,0].fill_between(x, control_pdf, alpha=0.3, color='blue')
        axes[0,0].fill_between(x, treatment_pdf, alpha=0.3, color='red')
        axes[0,0].set_xlabel('Conversion Rate')
        axes[0,0].set_ylabel('Density')
        axes[0,0].set_title('Posterior Distributions')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)

        # Plot credible intervals
        control_ci = bayesian_result['control_credible_interval']
        treatment_ci = bayesian_result['treatment_credible_interval']

        groups = ['Control', 'Treatment']
        means = [bayesian_result['control_rate'], bayesian_result['treatment_rate']]
        ci_lower = [control_ci[0], treatment_ci[0]]
        ci_upper = [control_ci[1], treatment_ci[1]]

        axes[0,1].bar(groups, means, yerr=[np.array(means) - np.array(ci_lower),
                                          np.array(ci_upper) - np.array(means)],
                     capsize=5, color=['blue', 'red'], alpha=0.7)
        axes[0,1].set_ylabel('Conversion Rate')
        axes[0,1].set_title('Conversion Rates with 95% Credible Intervals')
        axes[0,1].grid(True, alpha=0.3)

        # Plot probability that treatment is better
        prob_better = bayesian_result['prob_treatment_better']
        axes[1,0].bar(['Treatment Better', 'Control Better'],
                     [prob_better, 1 - prob_better],
                     color=['green' if prob_better > 0.5 else 'red',
                           'red' if prob_better > 0.5 else 'green'], alpha=0.7)
        axes[1,0].set_ylabel('Probability')
        axes[1,0].set_title(f'P(Treatment > Control) = {prob_better:.3f}')
        axes[1,0].grid(True, alpha=0.3)

        # Plot expected lift
        expected_lift = bayesian_result['expected_lift']
        lift_ci = bayesian_result['lift_credible_interval']

        axes[1,1].bar(['Expected Lift'], [expected_lift],
                     yerr=[[expected_lift - lift_ci[0]], [lift_ci[1] - expected_lift]],
                     capsize=5, color='purple', alpha=0.7)
        axes[1,1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[1,1].set_ylabel('Lift (%)')
        axes[1,1].set_title(f'Expected Lift: {expected_lift:.1%}')
        axes[1,1].grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_sequential_testing(self, sequential_result, title="Sequential Testing Results"):
        """Plot sequential testing results"""
        looks_data = sequential_result['results_by_look']

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(title, fontsize=16, fontweight='bold')

        # Extract data
        look_numbers = [look['look_number'] for look in looks_data]
        p_values = [look['p_value'] for look in looks_data]
        alpha_spent = [look['alpha_spent'] for look in looks_data]
        cumulative_alpha = [look['cumulative_alpha'] for look in looks_data]
        sample_sizes = [look['sample_size'] for look in looks_data]

        # Plot p-values over time
        axes[0,0].plot(look_numbers, p_values, 'bo-', linewidth=2, markersize=8)
        axes[0,0].plot(look_numbers, alpha_spent, 'r--', linewidth=2, label='Alpha Spending')
        axes[0,0].set_xlabel('Look Number')
        axes[0,0].set_ylabel('P-value')
        axes[0,0].set_title('P-values vs Alpha Spending')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        axes[0,0].set_yscale('log')

        # Plot cumulative alpha spending
        axes[0,1].plot(look_numbers, cumulative_alpha, 'go-', linewidth=2, markersize=8)
        axes[0,1].axhline(y=sequential_result.get('alpha', 0.05), color='red',
                         linestyle='--', label='Total Alpha')
        axes[0,1].set_xlabel('Look Number')
        axes[0,1].set_ylabel('Cumulative Alpha')
        axes[0,1].set_title('Alpha Spending Over Time')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)

        # Plot sample sizes
        axes[1,0].bar(look_numbers, sample_sizes, alpha=0.7, color='purple')
        axes[1,0].set_xlabel('Look Number')
        axes[1,0].set_ylabel('Sample Size')
        axes[1,0].set_title('Sample Size at Each Look')
        axes[1,0].grid(True, alpha=0.3)

        # Plot means over time
        control_means = [look['mean_control'] for look in looks_data]
        treatment_means = [look['mean_treatment'] for look in looks_data]

        axes[1,1].plot(look_numbers, control_means, 'bo-', label='Control', linewidth=2)
        axes[1,1].plot(look_numbers, treatment_means, 'ro-', label='Treatment', linewidth=2)
        axes[1,1].set_xlabel('Look Number')
        axes[1,1].set_ylabel('Mean Value')
        axes[1,1].set_title('Group Means Over Time')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_bootstrap_distribution(self, bootstrap_result, title="Bootstrap Test Results"):
        """Plot bootstrap test results"""
        # This is a simplified version - in practice you'd store the bootstrap samples
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle(title, fontsize=16, fontweight='bold')

        # Simulate bootstrap distribution for visualization
        observed_diff = bootstrap_result['observed_difference']
        ci_lower, ci_upper = bootstrap_result['bootstrap_ci_95']

        # Create simulated bootstrap distribution
        bootstrap_diffs = np.random.normal(0, (ci_upper - ci_lower) / 4, 10000)

        # Plot bootstrap distribution
        axes[0].hist(bootstrap_diffs, bins=50, alpha=0.7, color='skyblue', density=True)
        axes[0].axvline(observed_diff, color='red', linestyle='--', linewidth=2,
                       label=f'Observed Difference: {observed_diff:.3f}')
        axes[0].axvline(ci_lower, color='green', linestyle='--', alpha=0.7,
                       label=f'95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]')
        axes[0].axvline(ci_upper, color='green', linestyle='--', alpha=0.7)
        axes[0].set_xlabel('Difference in Means')
        axes[0].set_ylabel('Density')
        axes[0].set_title('Bootstrap Distribution')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Plot p-value visualization
        p_value = bootstrap_result['p_value']
        x = np.linspace(-4, 4, 1000)
        y = stats.norm.pdf(x, 0, 1)

        axes[1].plot(x, y, 'b-', linewidth=2, label='Null Distribution')

        # Shade p-value regions
        critical_value = abs(observed_diff) / (ci_upper - ci_lower) * 4  # Approximate
        x_left = x[x <= -critical_value]
        x_right = x[x >= critical_value]

        if len(x_left) > 0:
            axes[1].fill_between(x_left, 0, stats.norm.pdf(x_left, 0, 1),
                                alpha=0.3, color='red', label=f'P-value = {p_value:.4f}')
        if len(x_right) > 0:
            axes[1].fill_between(x_right, 0, stats.norm.pdf(x_right, 0, 1),
                                alpha=0.3, color='red')

        axes[1].set_xlabel('Standard Deviations')
        axes[1].set_ylabel('Density')
        axes[1].set_title('P-value Visualization')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_multiple_testing_correction(self, correction_results, title="Multiple Testing Correction"):
        """Plot multiple testing correction results"""
        if 'results' not in correction_results:
            return None

        results = correction_results['results']
        test_names = list(results.keys())
        original_p = [results[name]['original_p_value'] for name in test_names]
        corrected_p = [results[name]['corrected_p_value'] for name in test_names]
        significant_before = [p < 0.05 for p in original_p]
        significant_after = [results[name]['significant_after_correction'] for name in test_names]

        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle(title, fontsize=16, fontweight='bold')

        # Before and after comparison
        x = np.arange(len(test_names))
        width = 0.35

        axes[0].bar(x - width/2, original_p, width, label='Original P-values', alpha=0.7)
        axes[0].bar(x + width/2, corrected_p, width, label='Corrected P-values', alpha=0.7)
        axes[0].axhline(y=0.05, color='red', linestyle='--', alpha=0.7, label='α = 0.05')
        axes[0].set_xlabel('Tests')
        axes[0].set_ylabel('P-value')
        axes[0].set_title('P-values Before and After Correction')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels([name.replace('_', '\n') for name in test_names], rotation=45)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].set_yscale('log')

        # Significance changes
        changes = []
        colors = []
        for i in range(len(test_names)):
            if significant_before[i] and significant_after[i]:
                changes.append('Still Significant')
                colors.append('green')
            elif significant_before[i] and not significant_after[i]:
                changes.append('Lost Significance')
                colors.append('red')
            elif not significant_before[i] and significant_after[i]:
                changes.append('Gained Significance')
                colors.append('blue')
            else:
                changes.append('Still Not Significant')
                colors.append('gray')

        change_counts = pd.Series(changes).value_counts()
        axes[1].pie(change_counts.values, labels=change_counts.index, autopct='%1.1f%%',
                   colors=['green', 'red', 'blue', 'gray'][:len(change_counts)])
        axes[1].set_title('Significance Changes After Correction')

        plt.tight_layout()
        return fig

    def save_plots(self, figures, output_dir="plots"):
        """Save all plots to files"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        saved_files = []
        for i, fig in enumerate(figures):
            if fig is not None:
                filename = os.path.join(output_dir, f"plot_{i+1}.png")
                fig.savefig(filename, dpi=self.dpi, bbox_inches='tight')
                saved_files.append(filename)
                plt.close(fig)  # Close figure to free memory

        print(f"Saved {len(saved_files)} plots to {output_dir}/")
        return saved_files

if __name__ == "__main__":
    # Example usage
    print("Visualization module loaded successfully!")
    print("Use ABTestVisualizer class to create plots for your A/B testing analysis.")
