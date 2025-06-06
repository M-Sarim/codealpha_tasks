"""
Advanced reporting module for A/B Testing Analysis
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *

class ABTestReporter:
    """Class for generating comprehensive A/B testing reports"""
    
    def __init__(self, analyzer):
        self.analyzer = analyzer
        self.report_data = {}
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def generate_executive_summary(self):
        """Generate executive summary of A/B test results"""
        all_tests = self.analyzer.stats_analyzer.results
        
        if not all_tests:
            return "No test results available for summary."
        
        # Calculate summary statistics
        total_tests = len(all_tests)
        significant_tests = sum(1 for result in all_tests.values() 
                              if result.get('significant', False))
        significance_rate = significant_tests / total_tests if total_tests > 0 else 0
        
        # Find most impactful results
        effect_sizes = []
        for test_name, result in all_tests.items():
            if 'effect_size_cohens_d' in result:
                effect_sizes.append((test_name, abs(result['effect_size_cohens_d'])))
        
        effect_sizes.sort(key=lambda x: x[1], reverse=True)
        
        summary = {
            'test_date': datetime.now().strftime("%Y-%m-%d"),
            'total_tests_performed': total_tests,
            'significant_results': significant_tests,
            'significance_rate': f"{significance_rate:.1%}",
            'top_findings': effect_sizes[:3] if effect_sizes else [],
            'recommendation_priority': 'High' if significance_rate > 0.5 else 'Medium' if significance_rate > 0.2 else 'Low'
        }
        
        return summary
    
    def generate_technical_report(self):
        """Generate detailed technical report"""
        technical_data = {
            'methodology': {
                'statistical_tests': ['Independent t-test', 'Mann-Whitney U', 'Chi-square'],
                'effect_size_measures': ['Cohen\'s d', 'Cramér\'s V', 'Cohen\'s h'],
                'significance_level': ALPHA,
                'power_target': POWER,
                'multiple_testing_correction': 'Bonferroni'
            },
            'data_summary': {},
            'test_results': {},
            'assumptions_checked': {
                'normality': 'Q-Q plots and Shapiro-Wilk test',
                'equal_variance': 'Levene\'s test',
                'independence': 'Random assignment verification'
            }
        }
        
        # Add data summary
        if self.analyzer.data is not None:
            technical_data['data_summary'] = {
                'total_samples': len(self.analyzer.data),
                'features': list(self.analyzer.data.columns),
                'missing_data': self.analyzer.data.isnull().sum().sum(),
                'data_types': self.analyzer.data.dtypes.value_counts().to_dict()
            }
        
        # Add detailed test results
        for test_name, result in self.analyzer.stats_analyzer.results.items():
            technical_data['test_results'][test_name] = {
                'test_type': result.get('test_type', 'Unknown'),
                'p_value': result.get('p_value', 'N/A'),
                'effect_size': result.get('effect_size_cohens_d', result.get('effect_size_cramers_v', 'N/A')),
                'confidence_interval': result.get('confidence_interval_95', 'N/A'),
                'sample_sizes': {
                    'control': result.get('sample_size_control', 'N/A'),
                    'treatment': result.get('sample_size_treatment', 'N/A')
                }
            }
        
        return technical_data
    
    def generate_business_recommendations(self):
        """Generate business-focused recommendations"""
        recommendations = {
            'immediate_actions': [],
            'further_investigation': [],
            'implementation_considerations': [],
            'risk_assessment': 'Low'
        }
        
        significant_results = []
        for test_name, result in self.analyzer.stats_analyzer.results.items():
            if result.get('significant', False):
                effect_size = result.get('effect_size_cohens_d', 0)
                significant_results.append((test_name, effect_size, result.get('p_value', 1)))
        
        if significant_results:
            # Sort by effect size
            significant_results.sort(key=lambda x: abs(x[1]), reverse=True)
            
            for test_name, effect_size, p_value in significant_results:
                if abs(effect_size) > 0.8:  # Large effect
                    recommendations['immediate_actions'].append(
                        f"Implement changes related to {test_name.split('_')[2]} - large effect detected (d={effect_size:.2f})"
                    )
                elif abs(effect_size) > 0.5:  # Medium effect
                    recommendations['further_investigation'].append(
                        f"Investigate {test_name.split('_')[2]} further - medium effect detected (d={effect_size:.2f})"
                    )
            
            recommendations['risk_assessment'] = 'Medium' if len(significant_results) > 2 else 'Low'
        else:
            recommendations['immediate_actions'].append("No significant differences found - maintain current approach")
            recommendations['further_investigation'].append("Consider increasing sample size or exploring different metrics")
        
        # Add implementation considerations
        recommendations['implementation_considerations'] = [
            "Monitor key metrics during implementation",
            "Consider gradual rollout to minimize risk",
            "Set up tracking for long-term impact assessment",
            "Plan for potential reversal if negative effects observed"
        ]
        
        return recommendations
    
    def export_results_to_excel(self, filename=None):
        """Export comprehensive results to Excel file"""
        if filename is None:
            filename = f"ab_test_report_{self.timestamp}.xlsx"
        
        filepath = os.path.join(PROCESSED_DATA_DIR, filename)
        
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            # Executive Summary
            exec_summary = self.generate_executive_summary()
            exec_df = pd.DataFrame([exec_summary])
            exec_df.to_excel(writer, sheet_name='Executive_Summary', index=False)
            
            # Test Results
            results_df = self.analyzer.stats_analyzer.get_results_dataframe()
            if not results_df.empty:
                results_df.to_excel(writer, sheet_name='Test_Results', index=False)
            
            # Technical Details
            tech_report = self.generate_technical_report()
            
            # Convert nested dict to flat structure for Excel
            tech_flat = []
            for category, data in tech_report.items():
                if isinstance(data, dict):
                    for key, value in data.items():
                        tech_flat.append({
                            'Category': category,
                            'Metric': key,
                            'Value': str(value)
                        })
                else:
                    tech_flat.append({
                        'Category': category,
                        'Metric': 'Summary',
                        'Value': str(data)
                    })
            
            tech_df = pd.DataFrame(tech_flat)
            tech_df.to_excel(writer, sheet_name='Technical_Details', index=False)
            
            # Business Recommendations
            recommendations = self.generate_business_recommendations()
            rec_flat = []
            for category, items in recommendations.items():
                if isinstance(items, list):
                    for i, item in enumerate(items):
                        rec_flat.append({
                            'Category': category,
                            'Item': i + 1,
                            'Recommendation': item
                        })
                else:
                    rec_flat.append({
                        'Category': category,
                        'Item': 1,
                        'Recommendation': str(items)
                    })
            
            rec_df = pd.DataFrame(rec_flat)
            rec_df.to_excel(writer, sheet_name='Recommendations', index=False)
            
            # Raw Data Summary
            if self.analyzer.data is not None:
                data_summary = self.analyzer.data.describe()
                data_summary.to_excel(writer, sheet_name='Data_Summary')
        
        print(f"Comprehensive report exported to: {filepath}")
        return filepath
    
    def export_results_to_json(self, filename=None):
        """Export results to JSON format"""
        if filename is None:
            filename = f"ab_test_report_{self.timestamp}.json"
        
        filepath = os.path.join(PROCESSED_DATA_DIR, filename)
        
        report_data = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'analysis_type': 'A/B Testing',
                'dataset': 'Wine Dataset'
            },
            'executive_summary': self.generate_executive_summary(),
            'technical_report': self.generate_technical_report(),
            'business_recommendations': self.generate_business_recommendations(),
            'raw_results': {}
        }
        
        # Add raw results (convert numpy types to native Python types)
        for test_name, result in self.analyzer.stats_analyzer.results.items():
            clean_result = {}
            for key, value in result.items():
                if isinstance(value, np.ndarray):
                    clean_result[key] = value.tolist()
                elif isinstance(value, (np.integer, np.floating)):
                    clean_result[key] = value.item()
                else:
                    clean_result[key] = value
            report_data['raw_results'][test_name] = clean_result
        
        with open(filepath, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        print(f"JSON report exported to: {filepath}")
        return filepath
    
    def generate_html_report(self, filename=None):
        """Generate HTML report with visualizations"""
        if filename is None:
            filename = f"ab_test_report_{self.timestamp}.html"
        
        filepath = os.path.join(PROCESSED_DATA_DIR, filename)
        
        # Generate report sections
        exec_summary = self.generate_executive_summary()
        tech_report = self.generate_technical_report()
        recommendations = self.generate_business_recommendations()
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>A/B Testing Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; padding: 15px; border-left: 4px solid #007acc; }}
                .metric {{ background-color: #f9f9f9; padding: 10px; margin: 5px 0; }}
                .significant {{ color: #d32f2f; font-weight: bold; }}
                .not-significant {{ color: #388e3c; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>A/B Testing Analysis Report</h1>
                <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p>Dataset: Wine Quality Analysis</p>
            </div>
            
            <div class="section">
                <h2>Executive Summary</h2>
                <div class="metric">Total Tests: {exec_summary['total_tests_performed']}</div>
                <div class="metric">Significant Results: {exec_summary['significant_results']}</div>
                <div class="metric">Success Rate: {exec_summary['significance_rate']}</div>
                <div class="metric">Priority: {exec_summary['recommendation_priority']}</div>
            </div>
            
            <div class="section">
                <h2>Key Findings</h2>
                <table>
                    <tr><th>Test</th><th>Effect Size</th><th>Significance</th></tr>
        """
        
        # Add test results to HTML
        for test_name, result in self.analyzer.stats_analyzer.results.items():
            significance_class = "significant" if result.get('significant', False) else "not-significant"
            effect_size = result.get('effect_size_cohens_d', 'N/A')
            html_content += f"""
                    <tr>
                        <td>{test_name}</td>
                        <td>{effect_size}</td>
                        <td class="{significance_class}">{'Significant' if result.get('significant', False) else 'Not Significant'}</td>
                    </tr>
            """
        
        html_content += """
                </table>
            </div>
            
            <div class="section">
                <h2>Recommendations</h2>
        """
        
        for action in recommendations['immediate_actions']:
            html_content += f"<div class='metric'>• {action}</div>"
        
        html_content += """
            </div>
        </body>
        </html>
        """
        
        with open(filepath, 'w') as f:
            f.write(html_content)
        
        print(f"HTML report generated: {filepath}")
        return filepath
    
    def generate_all_reports(self):
        """Generate all report formats"""
        reports_generated = []
        
        try:
            excel_file = self.export_results_to_excel()
            reports_generated.append(excel_file)
        except Exception as e:
            print(f"Error generating Excel report: {e}")
        
        try:
            json_file = self.export_results_to_json()
            reports_generated.append(json_file)
        except Exception as e:
            print(f"Error generating JSON report: {e}")
        
        try:
            html_file = self.generate_html_report()
            reports_generated.append(html_file)
        except Exception as e:
            print(f"Error generating HTML report: {e}")
        
        return reports_generated

if __name__ == "__main__":
    print("Reporting module loaded successfully!")
    print("Use ABTestReporter class to generate comprehensive reports.")
