"""
Main execution script for Wine A/B Testing Analysis
Run this script to perform complete A/B testing analysis
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from config import *
from src.ab_test_analysis import WineABTestAnalyzer
from src.visualization import ABTestVisualizer
from src.reporting import ABTestReporter

def main():
    """Main function to run complete A/B testing analysis"""

    print("="*60)
    print("WINE DATASET A/B TESTING ANALYSIS")
    print("="*60)
    print()

    try:
        # Initialize analyzer
        print("🔧 Initializing A/B Test Analyzer...")
        analyzer = WineABTestAnalyzer(alpha=ALPHA)

        # Load and prepare data
        print("📊 Loading and preparing wine dataset...")
        data = analyzer.load_and_prepare_data()

        if data is None:
            print("❌ Failed to load data. Please check the data file path.")
            return

        print(f"✅ Data loaded successfully: {data.shape[0]} samples, {data.shape[1]} features")

        # Create A/B test scenarios
        print("\n🧪 Creating A/B test scenarios...")

        # Scenario 1: Alcohol content impact
        print("   📈 Scenario 1: Alcohol content impact on wine characteristics")
        analyzer.create_ab_test_scenario(
            feature='alcohol',
            threshold_percentile=50,
            scenario_name='alcohol_impact'
        )

        # Run comprehensive analysis
        print("\n📊 Running comprehensive statistical analysis...")
        metrics_to_analyze = [
            'total_phenols', 'flavanoids', 'color_intensity',
            'hue', 'proline', 'malic_acid'
        ]

        analysis_results = analyzer.run_comprehensive_analysis(
            primary_feature='alcohol',
            metrics_to_analyze=metrics_to_analyze
        )

        print(f"✅ Analysis completed for {len(analysis_results)} metrics")

        # Run advanced analysis
        print("\n🚀 Running advanced A/B testing methods...")
        advanced_metrics = ['total_phenols', 'flavanoids', 'color_intensity']

        try:
            advanced_results = analyzer.run_advanced_comprehensive_analysis(
                primary_feature='alcohol',
                metrics_to_analyze=advanced_metrics
            )
            print(f"✅ Advanced analysis completed for {len(advanced_results)} metrics")
        except Exception as e:
            print(f"⚠️ Advanced analysis encountered issues: {e}")
            advanced_results = {}

        # Perform power analysis
        print("\n⚡ Performing power analysis...")
        power_results = analyzer.perform_power_analysis(
            effect_sizes=[0.2, 0.5, 0.8],
            sample_sizes=[50, 100, 200, 500]
        )

        # Generate insights and recommendations
        print("\n💡 Generating insights and recommendations...")
        insights = analyzer.generate_insights_and_recommendations()

        # Create visualizations
        print("\n📈 Creating visualizations...")
        visualizer = ABTestVisualizer()

        # Create plots directory
        plots_dir = "plots"
        if not os.path.exists(plots_dir):
            os.makedirs(plots_dir)

        figures = []

        # 1. Correlation matrix
        try:
            fig_corr = visualizer.plot_correlation_matrix(
                data, "Wine Features Correlation Matrix"
            )
            figures.append(fig_corr)
            print("   ✅ Correlation matrix created")
        except Exception as e:
            print(f"   ⚠️ Could not create correlation matrix: {e}")

        # 2. Group comparisons for key metrics
        key_metrics = ['total_phenols', 'flavanoids', 'color_intensity']
        for metric in key_metrics:
            if metric in data.columns:
                try:
                    fig_comp = visualizer.plot_group_comparison(
                        data, metric,
                        title=f'A/B Group Comparison: {metric.replace("_", " ").title()}'
                    )
                    figures.append(fig_comp)
                    print(f"   ✅ Group comparison for {metric} created")
                except Exception as e:
                    print(f"   ⚠️ Could not create comparison for {metric}: {e}")

        # 3. Effect sizes plot
        try:
            fig_effects = visualizer.plot_effect_size_comparison(
                analyzer.stats_analyzer.results,
                "Effect Sizes Across All Tests"
            )
            figures.append(fig_effects)
            print("   ✅ Effect sizes plot created")
        except Exception as e:
            print(f"   ⚠️ Could not create effect sizes plot: {e}")

        # 4. Power analysis plot
        try:
            fig_power = visualizer.plot_power_analysis(power_results)
            figures.append(fig_power)
            print("   ✅ Power analysis plot created")
        except Exception as e:
            print(f"   ⚠️ Could not create power analysis plot: {e}")

        # 5. Advanced visualizations
        if advanced_results:
            for metric, results in advanced_results.items():
                # Bayesian plots
                if 'bayesian' in results:
                    try:
                        fig_bayesian = visualizer.plot_bayesian_results(
                            results['bayesian'], f'Bayesian Analysis: {metric.title()}'
                        )
                        figures.append(fig_bayesian)
                        print(f"   ✅ Bayesian plot for {metric} created")
                    except Exception as e:
                        print(f"   ⚠️ Could not create Bayesian plot for {metric}: {e}")

                # Bootstrap plots
                if 'bootstrap' in results:
                    try:
                        fig_bootstrap = visualizer.plot_bootstrap_distribution(
                            results['bootstrap'], f'Bootstrap Analysis: {metric.title()}'
                        )
                        figures.append(fig_bootstrap)
                        print(f"   ✅ Bootstrap plot for {metric} created")
                    except Exception as e:
                        print(f"   ⚠️ Could not create Bootstrap plot for {metric}: {e}")

                # Sequential plots
                if 'sequential' in results:
                    try:
                        fig_sequential = visualizer.plot_sequential_testing(
                            results['sequential'], f'Sequential Analysis: {metric.title()}'
                        )
                        figures.append(fig_sequential)
                        print(f"   ✅ Sequential plot for {metric} created")
                    except Exception as e:
                        print(f"   ⚠️ Could not create Sequential plot for {metric}: {e}")

        # 6. Multiple testing correction plot
        if 'alcohol_advanced' in analyzer.results and 'multiple_testing_correction' in analyzer.results['alcohol_advanced']:
            try:
                correction_results = analyzer.results['alcohol_advanced']['multiple_testing_correction']
                fig_correction = visualizer.plot_multiple_testing_correction(
                    correction_results, "Multiple Testing Correction Results"
                )
                figures.append(fig_correction)
                print("   ✅ Multiple testing correction plot created")
            except Exception as e:
                print(f"   ⚠️ Could not create multiple testing correction plot: {e}")

        # Save plots
        if figures:
            saved_files = visualizer.save_plots(figures, plots_dir)
            print(f"   💾 Saved {len(saved_files)} plots to {plots_dir}/")

        # Save results
        print("\n💾 Saving analysis results...")
        results_df = analyzer.save_results('wine_ab_test_results.csv')

        # Generate comprehensive reports
        print("\n📋 Generating comprehensive reports...")
        reporter = ABTestReporter(analyzer)

        try:
            generated_reports = reporter.generate_all_reports()
            print(f"✅ Generated {len(generated_reports)} report files:")
            for report in generated_reports:
                print(f"   📄 {os.path.basename(report)}")
        except Exception as e:
            print(f"⚠️ Error generating reports: {e}")

        # Generate executive summary
        try:
            exec_summary = reporter.generate_executive_summary()
            print("\n📊 EXECUTIVE SUMMARY")
            print("="*40)
            print(f"Test Date: {exec_summary['test_date']}")
            print(f"Total Tests: {exec_summary['total_tests_performed']}")
            print(f"Significant Results: {exec_summary['significant_results']}")
            print(f"Success Rate: {exec_summary['significance_rate']}")
            print(f"Priority Level: {exec_summary['recommendation_priority']}")

            if exec_summary['top_findings']:
                print("\nTop Findings (by effect size):")
                for i, (test, effect) in enumerate(exec_summary['top_findings'], 1):
                    print(f"  {i}. {test}: Effect size = {effect:.3f}")
        except Exception as e:
            print(f"⚠️ Error generating executive summary: {e}")

        # Print summary report
        print("\n📋 FINAL SUMMARY REPORT")
        print("="*40)
        analyzer.print_summary_report()

        # Additional summary statistics
        print("\n📊 ADDITIONAL SUMMARY STATISTICS")
        print("="*40)

        if not results_df.empty:
            significant_tests = results_df[results_df['Significant'] == True]
            print(f"Total tests performed: {len(results_df)}")
            print(f"Significant results: {len(significant_tests)}")
            print(f"Significance rate: {len(significant_tests)/len(results_df):.2%}")

            if len(significant_tests) > 0:
                print(f"\nSignificant tests:")
                for _, row in significant_tests.iterrows():
                    print(f"  - {row['Test_Name']}: p={row['P_Value']:.4f}")

        # Run simulations for power validation
        print("\n🎲 Running A/B test simulations...")
        try:
            # Simulate different effect sizes
            simulation_results = []
            for effect_size in [0.2, 0.5, 0.8]:
                sim_result = analyzer.simulate_ab_test(
                    effect_size=effect_size,
                    sample_size_per_group=100,
                    n_simulations=1000
                )
                simulation_results.append(sim_result)
                print(f"   Effect size {effect_size}: Empirical power = {sim_result['empirical_power']:.3f}")
        except Exception as e:
            print(f"⚠️ Simulation error: {e}")

        # Calculate minimum detectable effects
        print("\n📏 Calculating minimum detectable effects...")
        try:
            if 'alcohol' in data.columns:
                baseline_mean = data['alcohol'].mean()
                baseline_std = data['alcohol'].std()

                for sample_size in [50, 100, 200]:
                    mde_result = analyzer.calculate_minimum_detectable_effect(
                        baseline_mean=baseline_mean,
                        baseline_std=baseline_std,
                        sample_size_per_group=sample_size
                    )
                    print(f"   Sample size {sample_size}: MDE = {mde_result['relative_mde']:.1%}")
        except Exception as e:
            print(f"⚠️ MDE calculation error: {e}")

        # Recommendations for next steps
        print("\n🎯 NEXT STEPS RECOMMENDATIONS")
        print("="*40)
        print("1. Review significant findings for business relevance")
        print("2. Consider replication with larger sample sizes")
        print("3. Implement randomized controlled trials for causal inference")
        print("4. Monitor key metrics in production environment")
        print("5. Consider multiple testing corrections for family-wise error control")
        print("6. Use Bayesian methods for continuous monitoring")
        print("7. Implement sequential testing for early stopping")
        print("8. Consider stratified randomization for better balance")

        print(f"\n✅ Analysis completed successfully!")
        print(f"📁 Results saved in: {PROCESSED_DATA_DIR}")
        print(f"📊 Plots saved in: {plots_dir}/")
        print(f"📓 For interactive analysis, run: jupyter notebook notebooks/ab_testing_analysis.ipynb")

    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        print("Please check your data files and dependencies.")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
