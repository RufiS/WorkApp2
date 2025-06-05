# Comprehensive SPLADE Testing Procedure & Analysis Guide

## Overview
This document outlines the procedure for running and analyzing the comprehensive SPLADE evaluation framework. The system is designed for **complete terminal independence** during 6-12 hour test runs.

## 🚀 **Running the Tests**

### Quick Start
```bash
# Run focused evaluation (3,996 configurations, 6-12 hours)
python tests/test_comprehensive_systematic_evaluation.py --mode focused

# Run full evaluation (50,000+ configurations, 24-48 hours)  
python tests/test_comprehensive_systematic_evaluation.py --mode full
```

### Terminal Independence
- **You can completely ignore terminal output** during the run
- All meaningful data is logged to files in `test_logs/`
- Progress, errors, and results are captured in structured files
- The terminal can be closed/disconnected without affecting the test

## 📁 **Generated Files Structure**

### During Execution (Every 10 Configurations)
```
test_logs/
├── comprehensive_systematic_evaluation.log           # Main execution log
├── comprehensive_checkpoint_10_20250604_032857.json  # Progress checkpoint
├── comprehensive_checkpoint_20_20250604_032858.json  # Progress checkpoint
├── ...                                               # Continued checkpoints
├── progress_summary_20250604_032857.md              # Human-readable progress
├── model_switch_log_20250604_032857.json            # Model transition tracking
└── error_analysis_20250604_032857.json              # Error pattern analysis
```

### Final Results
```
test_logs/
├── focused_comprehensive_detailed_20250604_040315.json    # Complete raw results
├── focused_comprehensive_summary_20250604_040315.json     # Aggregated metrics  
├── focused_comprehensive_summary_20250604_040315.csv      # Excel-friendly data
├── analysis_report_20250604_040315.md                     # Auto-generated analysis
└── top_configurations_20250604_040315.json               # Best performers
```

## 📊 **File Contents & Analysis**

### 1. Main Execution Log (`comprehensive_systematic_evaluation.log`)
**Purpose**: Complete trace of execution  
**Contents**:
- Configuration switches
- Model loading status
- Index rebuild confirmations
- Query processing progress
- Error details with stack traces

**Analysis**: Use for debugging failures and understanding execution flow

### 2. Progress Checkpoints (`comprehensive_checkpoint_*.json`)
**Purpose**: Recovery and progress tracking  
**Contents**:
```json
{
  "checkpoint_info": {
    "configs_completed": 120,
    "total_configs": 3996,
    "timestamp": "20250604_032857",
    "elapsed_hours": 2.1
  },
  "results": [...]  // All results so far
}
```

**Analysis**: 
- Monitor progress: `configs_completed / total_configs`
- Estimate completion: `elapsed_hours / configs_completed * total_configs`
- Resume from failure: Load latest checkpoint

### 3. Summary Data (`*_summary_*.json` & `.csv`)
**Purpose**: High-level analysis and comparison  
**Key Metrics**:
```json
{
  "config_id": "FOCUSED_001234",
  "embedding_model": "intfloat/e5-large-v2",
  "pipeline_type": "pure_splade",
  "avg_retrieval_time": 0.125,
  "context_hit_rate": 0.847,
  "avg_correctness": 0.792,
  "success_rate": 0.989
}
```

**Analysis Priority** (QUALITY FIRST):
1. **ANSWER QUALITY**: `context_hit_rate`, `avg_correctness`, `avg_completeness`, `avg_specificity`
2. **RELIABILITY**: `success_rate`, `errors_encountered`
3. **Performance**: `avg_retrieval_time` (NOTE: Test timing ≠ Production timing)

### 4. Auto-Generated Analysis (`analysis_report_*.md`)
**Purpose**: Ready-to-read insights  
**Contents**:
- Top 10 configurations by quality metrics
- Performance comparisons by embedding model
- SPLADE parameter effectiveness analysis
- Error pattern summaries
- Recommendations for production

## 🔍 **Analysis Workflow**

### **NO REAL-TIME MONITORING REQUIRED**
- **Start the test and walk away** - no monitoring needed during 6-12 hour run
- **All progress automatically saved** to checkpoint files
- **Complete hands-off operation** - check results only when finished

### Post-Run Analysis (ONLY After Test Completion)

#### 1. Quick Overview
```bash
# Open final analysis report
cat test_logs/analysis_report_*.md

# Check top performers
head -20 test_logs/focused_comprehensive_summary_*.csv
```

#### 2. Detailed Analysis
```python
import pandas as pd
import json

# Load summary data
df = pd.read_csv('test_logs/focused_comprehensive_summary_*.csv')

# Top configurations by context hit rate
top_configs = df.nlargest(10, 'context_hit_rate')

# Performance by embedding model
perf_by_model = df.groupby('embedding_model')['avg_retrieval_time'].mean()

# SPLADE parameter effectiveness
splade_configs = df[df['pipeline_type'] == 'pure_splade']
splade_analysis = splade_configs.groupby('sparse_weight')['avg_correctness'].mean()
```

#### 3. Configuration Comparison
```python
# Compare pipeline types
pipeline_comparison = df.groupby('pipeline_type').agg({
    'context_hit_rate': 'mean',
    'avg_correctness': 'mean', 
    'avg_retrieval_time': 'mean',
    'success_rate': 'mean'
})

# Best SPLADE parameters
best_splade = df[df['pipeline_type'] == 'pure_splade'].nlargest(5, 'context_hit_rate')
```

## 🏆 **Key Analysis Questions**

### Performance Analysis
1. **Which embedding model is fastest?** → Sort by `avg_retrieval_time`
2. **Quality vs Speed tradeoff?** → Plot `context_hit_rate` vs `avg_retrieval_time`
3. **Most reliable configuration?** → Sort by `success_rate`

### SPLADE Optimization
1. **Optimal sparse weight?** → Group by `sparse_weight`, analyze `avg_correctness`
2. **Best expansion K?** → Group by `expansion_k`, analyze `context_hit_rate`
3. **Memory vs quality?** → Plot `max_sparse_length` vs quality metrics

### Production Recommendations
1. **Best overall configuration?** → Weighted score of quality + performance + reliability
2. **Resource-constrained choice?** → Filter by `avg_retrieval_time < threshold`
3. **High-accuracy requirement?** → Filter by `context_hit_rate > 0.9`

## 🚨 **Error Analysis**

### Common Error Patterns
- **Model loading failures**: Check embedding model compatibility
- **Index corruption**: Verify cleanup between model switches
- **Memory errors**: Monitor GPU/RAM usage during long runs
- **API timeouts**: Check LLM service reliability

### Recovery Procedures
```bash
# Resume from latest checkpoint
python tests/test_comprehensive_systematic_evaluation.py --mode focused --resume

# Skip failed configurations
python tests/test_comprehensive_systematic_evaluation.py --mode focused --skip-errors
```

## 🎯 **Success Metrics**

### Test Run Success
- ✅ **Completion Rate**: >95% configurations completed
- ✅ **Data Quality**: <5% query failures per configuration  
- ✅ **Model Isolation**: No index leakage between embedding models
- ✅ **Performance**: <2s average per query across all configurations

### Analysis Success
- ✅ **Clear Winner**: >10% improvement in top configuration
- ✅ **Actionable Insights**: Specific parameter recommendations
- ✅ **Production Ready**: Validated configuration for deployment
- ✅ **Resource Planning**: Accurate performance/quality tradeoffs

## 📋 **Troubleshooting**

### Test Won't Start
- Check `KTI Dispatch Guide.txt` exists in root directory
- Verify all dependencies installed: `pip install -r requirements.txt`
- Ensure sufficient disk space: 10GB+ for focused mode

### Test Stops Unexpectedly  
- Check latest checkpoint: `ls test_logs/comprehensive_checkpoint_*.json | tail -1`
- Review error log: `tail -50 test_logs/comprehensive_systematic_evaluation.log`
- Resume from checkpoint (if supported)

### Poor Results Quality
- Verify source document loaded correctly
- Check embedding model compatibility
- Review query failure rates in summary data

This procedure ensures **complete terminal independence** and **comprehensive analysis capability** for the 6-12 hour test runs.
