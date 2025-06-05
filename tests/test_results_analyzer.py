"""
Test Results Analysis and Visualization
=====================================

This script analyzes the results from systematic engine evaluation
and generates comprehensive reports and visualizations.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
from datetime import datetime

class TestResultsAnalyzer:
    """Analyzes and visualizes test results from systematic evaluation."""
    
    def __init__(self, results_file: str):
        """Initialize analyzer with results file."""
        self.results_file = Path(results_file)
        self.output_dir = Path("test_logs/analysis")
        self.output_dir.mkdir(exist_ok=True)
        
        # Load results
        with open(self.results_file, 'r') as f:
            self.raw_results = json.load(f)
        
        # Convert to DataFrame for analysis
        self.df = self._create_dataframe()
        print(f"Loaded {len(self.df)} test configurations")
    
    def _create_dataframe(self) -> pd.DataFrame:
        """Convert raw results to pandas DataFrame."""
        
        rows = []
        for result in self.raw_results:
            config = result['config']
            
            row = {
                # Configuration parameters
                'engine_type': config['engine_type'],
                'embedding_model': config['embedding_model'],
                'use_splade': config['use_splade'],
                'splade_model': config['splade_model'],
                'sparse_weight': config['sparse_weight'],
                'expansion_k': config['expansion_k'],
                'max_sparse_length': config['max_sparse_length'],
                'similarity_threshold': config['similarity_threshold'],
                'top_k': config['top_k'],
                
                # Performance metrics
                'avg_retrieval_time': result['avg_retrieval_time'],
                'avg_answer_time': result['avg_answer_time'],
                'avg_total_time': result['avg_total_time'],
                'avg_similarity': result['avg_similarity'],
                
                # Quality metrics
                'context_hit_rate': result['context_hit_rate'],
                'avg_correctness': result['avg_correctness'],
                'avg_completeness': result['avg_completeness'],
                'avg_specificity': result['avg_specificity'],
                
                # Feedback metrics
                'positive_feedback_rate': result['positive_feedback_rate'],
                'negative_feedback_rate': result['negative_feedback_rate'],
                
                # Test metrics
                'total_queries': len(result['query_results']),
                'errors_encountered': result['errors_encountered'],
                'total_test_time': result['total_test_time']
            }
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def generate_summary_report(self) -> str:
        """Generate a comprehensive summary report."""
        
        report = []
        report.append("# Systematic Engine Evaluation Results")
        report.append(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Total Configurations Tested: {len(self.df)}")
        report.append("")
        
        # Overall statistics
        report.append("## Overall Performance Summary")
        report.append("")
        
        metrics = ['context_hit_rate', 'avg_correctness', 'avg_completeness', 'avg_specificity', 'avg_retrieval_time']
        for metric in metrics:
            mean_val = self.df[metric].mean()
            std_val = self.df[metric].std()
            min_val = self.df[metric].min()
            max_val = self.df[metric].max()
            
            report.append(f"**{metric.replace('_', ' ').title()}:**")
            report.append(f"- Mean: {mean_val:.4f} ± {std_val:.4f}")
            report.append(f"- Range: {min_val:.4f} - {max_val:.4f}")
            report.append("")
        
        # Best configurations by different metrics
        report.append("## Top Performing Configurations")
        report.append("")
        
        top_metrics = {
            'context_hit_rate': 'Context Hit Rate',
            'avg_correctness': 'Answer Correctness', 
            'positive_feedback_rate': 'Positive Feedback Rate',
            'avg_retrieval_time': 'Retrieval Speed (lower is better)'
        }
        
        for metric, title in top_metrics.items():
            report.append(f"### Best {title}")
            
            if metric == 'avg_retrieval_time':
                # For time, lower is better
                top_configs = self.df.nsmallest(3, metric)
            else:
                # For other metrics, higher is better
                top_configs = self.df.nlargest(3, metric)
            
            for i, (_, config) in enumerate(top_configs.iterrows()):
                report.append(f"{i+1}. **{config['engine_type']}** + {config['embedding_model']}")
                report.append(f"   - Value: {config[metric]:.4f}")
                report.append(f"   - SPLADE: {config['use_splade']}")
                if config['use_splade']:
                    report.append(f"   - SPLADE Model: {config['splade_model']}")
                    report.append(f"   - Sparse Weight: {config['sparse_weight']}")
                report.append("")
        
        # Engine type comparison
        report.append("## Engine Type Performance Comparison")
        report.append("")
        
        engine_comparison = self.df.groupby('engine_type').agg({
            'context_hit_rate': ['mean', 'std'],
            'avg_correctness': ['mean', 'std'],
            'avg_retrieval_time': ['mean', 'std'],
            'positive_feedback_rate': ['mean', 'std']
        }).round(4)
        
        for engine in engine_comparison.index:
            report.append(f"### {engine.title()} Engine")
            hit_rate = engine_comparison.loc[engine, ('context_hit_rate', 'mean')]
            hit_std = engine_comparison.loc[engine, ('context_hit_rate', 'std')]
            correctness = engine_comparison.loc[engine, ('avg_correctness', 'mean')]
            correctness_std = engine_comparison.loc[engine, ('avg_correctness', 'std')]
            speed = engine_comparison.loc[engine, ('avg_retrieval_time', 'mean')]
            speed_std = engine_comparison.loc[engine, ('avg_retrieval_time', 'std')]
            
            report.append(f"- Context Hit Rate: {hit_rate:.3f} ± {hit_std:.3f}")
            report.append(f"- Answer Correctness: {correctness:.3f} ± {correctness_std:.3f}")
            report.append(f"- Avg Retrieval Time: {speed:.3f}s ± {speed_std:.3f}s")
            report.append("")
        
        # Embedding model comparison
        report.append("## Embedding Model Performance Comparison")
        report.append("")
        
        embedding_comparison = self.df.groupby('embedding_model').agg({
            'context_hit_rate': ['mean', 'std', 'count'],
            'avg_correctness': ['mean', 'std'],
            'avg_retrieval_time': ['mean', 'std']
        }).round(4)
        
        # Sort by context hit rate
        embedding_comparison = embedding_comparison.sort_values(('context_hit_rate', 'mean'), ascending=False)
        
        for i, embedding in enumerate(embedding_comparison.index):
            hit_rate = embedding_comparison.loc[embedding, ('context_hit_rate', 'mean')]
            hit_std = embedding_comparison.loc[embedding, ('context_hit_rate', 'std')]
            correctness = embedding_comparison.loc[embedding, ('avg_correctness', 'mean')]
            count = embedding_comparison.loc[embedding, ('context_hit_rate', 'count')]
            
            report.append(f"{i+1}. **{embedding}**")
            report.append(f"   - Context Hit Rate: {hit_rate:.3f} ± {hit_std:.3f} ({count} tests)")
            report.append(f"   - Answer Correctness: {correctness:.3f}")
            report.append("")
        
        # SPLADE analysis (if any SPLADE tests were run)
        splade_tests = self.df[self.df['use_splade'] == True]
        if len(splade_tests) > 0:
            report.append("## SPLADE Configuration Analysis")
            report.append("")
            
            # Sparse weight analysis
            weight_analysis = splade_tests.groupby('sparse_weight')['context_hit_rate'].agg(['mean', 'std', 'count']).round(4)
            report.append("### Sparse Weight Impact")
            for weight in sorted(weight_analysis.index):
                hit_rate = weight_analysis.loc[weight, 'mean']
                std = weight_analysis.loc[weight, 'std']
                count = weight_analysis.loc[weight, 'count']
                report.append(f"- Weight {weight}: {hit_rate:.3f} ± {std:.3f} ({count} tests)")
            report.append("")
            
            # Expansion K analysis
            expansion_analysis = splade_tests.groupby('expansion_k')['context_hit_rate'].agg(['mean', 'std', 'count']).round(4)
            report.append("### Expansion K Impact")
            for k in sorted(expansion_analysis.index):
                hit_rate = expansion_analysis.loc[k, 'mean']
                std = expansion_analysis.loc[k, 'std']
                count = expansion_analysis.loc[k, 'count']
                report.append(f"- K={k}: {hit_rate:.3f} ± {std:.3f} ({count} tests)")
            report.append("")
        
        # Recommendations
        report.append("## Recommendations")
        report.append("")
        
        best_overall = self.df.loc[self.df['context_hit_rate'].idxmax()]
        fastest = self.df.loc[self.df['avg_retrieval_time'].idxmin()]
        
        report.append("### Best Overall Performance")
        report.append(f"- **Engine**: {best_overall['engine_type']}")
        report.append(f"- **Embedding Model**: {best_overall['embedding_model']}")
        report.append(f"- **Use SPLADE**: {best_overall['use_splade']}")
        if best_overall['use_splade']:
            report.append(f"- **SPLADE Model**: {best_overall['splade_model']}")
            report.append(f"- **Sparse Weight**: {best_overall['sparse_weight']}")
            report.append(f"- **Expansion K**: {best_overall['expansion_k']}")
        report.append(f"- **Context Hit Rate**: {best_overall['context_hit_rate']:.3f}")
        report.append(f"- **Answer Correctness**: {best_overall['avg_correctness']:.3f}")
        report.append("")
        
        report.append("### Fastest Configuration")
        report.append(f"- **Engine**: {fastest['engine_type']}")
        report.append(f"- **Embedding Model**: {fastest['embedding_model']}")
        report.append(f"- **Retrieval Time**: {fastest['avg_retrieval_time']:.3f}s")
        report.append(f"- **Context Hit Rate**: {fastest['context_hit_rate']:.3f}")
        report.append("")
        
        return "\n".join(report)
    
    def create_visualizations(self):
        """Create comprehensive visualizations."""
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Engine Performance Comparison
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Engine Performance Comparison', fontsize=16)
        
        # Context Hit Rate by Engine
        sns.boxplot(data=self.df, x='engine_type', y='context_hit_rate', ax=axes[0,0])
        axes[0,0].set_title('Context Hit Rate by Engine Type')
        axes[0,0].set_ylabel('Context Hit Rate')
        
        # Answer Correctness by Engine
        sns.boxplot(data=self.df, x='engine_type', y='avg_correctness', ax=axes[0,1])
        axes[0,1].set_title('Answer Correctness by Engine Type')
        axes[0,1].set_ylabel('Average Correctness')
        
        # Retrieval Time by Engine
        sns.boxplot(data=self.df, x='engine_type', y='avg_retrieval_time', ax=axes[1,0])
        axes[1,0].set_title('Retrieval Time by Engine Type')
        axes[1,0].set_ylabel('Average Retrieval Time (s)')
        
        # Feedback Rate by Engine
        sns.boxplot(data=self.df, x='engine_type', y='positive_feedback_rate', ax=axes[1,1])
        axes[1,1].set_title('Positive Feedback Rate by Engine Type')
        axes[1,1].set_ylabel('Positive Feedback Rate')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'engine_performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Embedding Model Performance
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Embedding Model Performance Comparison', fontsize=16)
        
        # Context Hit Rate by Embedding Model
        embedding_means = self.df.groupby('embedding_model')['context_hit_rate'].mean().sort_values(ascending=False)
        sns.barplot(x=embedding_means.values, y=embedding_means.index, ax=axes[0,0])
        axes[0,0].set_title('Context Hit Rate by Embedding Model')
        axes[0,0].set_xlabel('Context Hit Rate')
        
        # Answer Correctness by Embedding Model
        correctness_means = self.df.groupby('embedding_model')['avg_correctness'].mean().sort_values(ascending=False)
        sns.barplot(x=correctness_means.values, y=correctness_means.index, ax=axes[0,1])
        axes[0,1].set_title('Answer Correctness by Embedding Model')
        axes[0,1].set_xlabel('Average Correctness')
        
        # Retrieval Speed by Embedding Model
        speed_means = self.df.groupby('embedding_model')['avg_retrieval_time'].mean().sort_values(ascending=True)
        sns.barplot(x=speed_means.values, y=speed_means.index, ax=axes[1,0])
        axes[1,0].set_title('Retrieval Speed by Embedding Model (Lower is Better)')
        axes[1,0].set_xlabel('Average Retrieval Time (s)')
        
        # Performance vs Speed Trade-off
        model_stats = self.df.groupby('embedding_model').agg({
            'context_hit_rate': 'mean',
            'avg_retrieval_time': 'mean'
        })
        
        scatter = axes[1,1].scatter(model_stats['avg_retrieval_time'], model_stats['context_hit_rate'], 
                                   s=100, alpha=0.7)
        axes[1,1].set_xlabel('Average Retrieval Time (s)')
        axes[1,1].set_ylabel('Context Hit Rate')
        axes[1,1].set_title('Performance vs Speed Trade-off')
        
        # Add labels to points
        for model, row in model_stats.iterrows():
            axes[1,1].annotate(model.split('/')[-1], (row['avg_retrieval_time'], row['context_hit_rate']), 
                              xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'embedding_model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. SPLADE Analysis (if applicable)
        splade_data = self.df[self.df['use_splade'] == True]
        if len(splade_data) > 0:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('SPLADE Configuration Analysis', fontsize=16)
            
            # Sparse Weight Impact
            weight_stats = splade_data.groupby('sparse_weight')['context_hit_rate'].agg(['mean', 'std'])
            axes[0,0].errorbar(weight_stats.index, weight_stats['mean'], yerr=weight_stats['std'], 
                              marker='o', capsize=5)
            axes[0,0].set_title('Context Hit Rate vs Sparse Weight')
            axes[0,0].set_xlabel('Sparse Weight')
            axes[0,0].set_ylabel('Context Hit Rate')
            axes[0,0].grid(True, alpha=0.3)
            
            # Expansion K Impact
            k_stats = splade_data.groupby('expansion_k')['context_hit_rate'].agg(['mean', 'std'])
            axes[0,1].errorbar(k_stats.index, k_stats['mean'], yerr=k_stats['std'], 
                              marker='o', capsize=5)
            axes[0,1].set_title('Context Hit Rate vs Expansion K')
            axes[0,1].set_xlabel('Expansion K')
            axes[0,1].set_ylabel('Context Hit Rate')
            axes[0,1].grid(True, alpha=0.3)
            
            # SPLADE Model Comparison
            if len(splade_data['splade_model'].unique()) > 1:
                sns.boxplot(data=splade_data, x='splade_model', y='context_hit_rate', ax=axes[1,0])
                axes[1,0].set_title('Context Hit Rate by SPLADE Model')
                axes[1,0].set_ylabel('Context Hit Rate')
                axes[1,0].tick_params(axis='x', rotation=45)
            
            # Performance Correlation Matrix
            splade_corr = splade_data[['sparse_weight', 'expansion_k', 'max_sparse_length', 
                                     'context_hit_rate', 'avg_correctness', 'avg_retrieval_time']].corr()
            sns.heatmap(splade_corr, annot=True, cmap='coolwarm', center=0, ax=axes[1,1])
            axes[1,1].set_title('SPLADE Parameter Correlation Matrix')
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'splade_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 4. Overall Performance Heatmap
        if len(self.df) > 10:  # Only create if we have enough data
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            
            # Create pivot table for heatmap
            pivot_data = self.df.pivot_table(
                values='context_hit_rate', 
                index='engine_type', 
                columns='embedding_model',
                aggfunc='mean'
            )
            
            sns.heatmap(pivot_data, annot=True, cmap='YlOrRd', ax=ax, fmt='.3f')
            ax.set_title('Context Hit Rate: Engine Type vs Embedding Model')
            plt.xticks(rotation=45, ha='right')
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'performance_heatmap.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"Visualizations saved to {self.output_dir}")
    
    def save_analysis(self):
        """Save complete analysis to files."""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save summary report
        report = self.generate_summary_report()
        report_file = self.output_dir / f"analysis_report_{timestamp}.md"
        with open(report_file, 'w') as f:
            f.write(report)
        
        # Save processed DataFrame
        csv_file = self.output_dir / f"processed_results_{timestamp}.csv"
        self.df.to_csv(csv_file, index=False)
        
        # Create visualizations
        self.create_visualizations()
        
        print(f"Analysis complete!")
        print(f"Report saved: {report_file}")
        print(f"Data saved: {csv_file}")
        print(f"Visualizations saved to: {self.output_dir}")
        
        return report_file, csv_file

def analyze_results(results_file: str):
    """Main function to analyze test results."""
    
    if not Path(results_file).exists():
        print(f"Error: Results file {results_file} not found")
        return
    
    print(f"Analyzing results from {results_file}")
    
    analyzer = TestResultsAnalyzer(results_file)
    analyzer.save_analysis()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze systematic engine evaluation results")
    parser.add_argument("results_file", help="Path to JSON results file")
    args = parser.parse_args()
    
    analyze_results(args.results_file)
