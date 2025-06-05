# SPLADE-Focused Brute Force Testing Strategy
## Comprehensive SPLADE parameter optimization with robust progress tracking

## 🎯 **Revised Problem & Solution**
- **Focus on SPLADE pipeline variants** - Vector/reranking produce sub-par results
- **Test ALL pipeline combinations** - Question-to-answer end-to-end evaluation
- **~40,000+ total configurations** - manageable with robust checkpointing
- **Parameter interactions unknown** - brute force captures all combinations
- **CRITICAL DISCOVERY:** Missing pipeline variants need implementation

## 🧠 **SPLADE-Focused Brute Force Approach**

### **Strategy: Complete SPLADE Parameter Space Coverage**

**SPLADE Parameter Space (Complete Coverage):**
```python
# Embedding models - test all viable options
embedding_models = [
    "intfloat/e5-base-v2",              # Current baseline
    "intfloat/e5-large-v2",             # Quality upgrade
    "intfloat/e5-small-v2",             # Speed option
    "BAAI/bge-base-en-v1.5",           # Strong alternative
    "BAAI/bge-large-en-v1.5",          # Large BGE
    "sentence-transformers/all-MiniLM-L6-v2",  # Fast baseline
    "microsoft/mpnet-base"              # General purpose
]

# Reranker models - test key options
reranker_models = [
    "cross-encoder/ms-marco-MiniLM-L-6-v2",   # Current default
    "cross-encoder/ms-marco-MiniLM-L-12-v2",  # Larger version
    "cross-encoder/ms-marco-distilbert-base", # Faster
    "cross-encoder/stsb-roberta-large",       # Different domain
    "cross-encoder/nli-deberta-v3-base"       # Advanced model
]

# SPLADE parameters - full range testing
sparse_weights = [0.1, 0.3, 0.5, 0.7, 0.9]
expansion_k_values = [50, 100, 150, 200, 300]
max_sparse_lengths = [128, 256, 512, 1024]

# Retrieval parameters - comprehensive coverage
similarity_thresholds = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
top_k_values = [5, 10, 15, 20, 25, 30]
```

**Total Configuration Space:**
```
7 embeddings × 5 rerankers × 5 sparse_weights × 5 expansion_k × 4 max_lengths × 7 thresholds × 6 top_k
= 36,750 SPLADE configurations
```

**Why Brute Force is Better:**
1. **Parameter interactions unknown** - SPLADE parameters may interact with embeddings in unexpected ways
2. **You found optimal settings once but lost them** - suggests optimal region might be non-obvious
3. **Accuracy > Speed** - finding the absolute best configuration is priority
4. **SPLADE-focused scope** - already eliminated 75% of search space

## 🎯 **Implementation: Robust SPLADE Brute Force**

### **Phase 1: Complete Configuration Generation**

```python
def generate_all_splade_configurations():
    """Generate ALL SPLADE parameter combinations."""
    
    embedding_models = [
        "intfloat/e5-base-v2", "intfloat/e5-large-v2", "intfloat/e5-small-v2",
        "BAAI/bge-base-en-v1.5", "BAAI/bge-large-en-v1.5", 
        "sentence-transformers/all-MiniLM-L6-v2", "microsoft/mpnet-base"
    ]
    
    reranker_models = [
        "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "cross-encoder/ms-marco-MiniLM-L-12-v2", 
        "cross-encoder/ms-marco-distilbert-base",
        "cross-encoder/stsb-roberta-large",
        "cross-encoder/nli-deberta-v3-base"
    ]
    
    sparse_weights = [0.1, 0.3, 0.5, 0.7, 0.9]
    expansion_k_values = [50, 100, 150, 200, 300]
    max_sparse_lengths = [128, 256, 512, 1024]
    similarity_thresholds = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
    top_k_values = [5, 10, 15, 20, 25, 30]
    
    configs = []
    config_id = 0
    
    for embedding in embedding_models:
        for reranker in reranker_models:
            for sparse_weight in sparse_weights:
                for expansion_k in expansion_k_values:
                    for max_length in max_sparse_lengths:
                        for threshold in similarity_thresholds:
                            for top_k in top_k_values:
                                config_id += 1
                                configs.append(SpladeConfiguration(
                                    config_id=f"SPLADE_{config_id:06d}",
                                    embedding_model=embedding,
                                    reranker_model=reranker,
                                    sparse_weight=sparse_weight,
                                    expansion_k=expansion_k,
                                    max_sparse_length=max_length,
                                    similarity_threshold=threshold,
                                    top_k=top_k
                                ))
    
    return configs  # ~36,750 configurations
```

### **Phase 2: Robust Progress Tracking & Checkpointing**

```python
def save_checkpoint_after_each_config(config, results):
    """Save progress after every single configuration."""
    
    checkpoint_data = {
        "config_id": config.config_id,
        "timestamp": datetime.now().isoformat(),
        "config": asdict(config),
        "results": [asdict(r) for r in results],
        "completed_configs": get_completed_count(),
        "total_configs": get_total_count(),
        "estimated_remaining_time": calculate_eta()
    }
    
    # Save individual config result
    config_file = f"test_logs/splade_config_{config.config_id}.json"
    with open(config_file, 'w') as f:
        json.dump(checkpoint_data, f, indent=2)
    
    # Update master progress file
    update_master_progress(config.config_id, checkpoint_data)

def resume_from_checkpoint():
    """Resume testing from last completed configuration."""
    
    completed_configs = load_completed_config_ids()
    all_configs = generate_all_splade_configurations()
    
    remaining_configs = [c for c in all_configs if c.config_id not in completed_configs]
    
    logger.info(f"Resuming: {len(completed_configs)} completed, {len(remaining_configs)} remaining")
    return remaining_configs
```

### **Phase 3: API Key Monitoring & Graceful Pausing**

```python
def monitor_api_usage():
    """Track API usage and pause before limits."""
    
    current_usage = get_openai_usage()
    if current_usage > API_LIMIT_THRESHOLD:
        logger.warning("Approaching API limit - pausing tests")
        save_pause_state()
        return False
    return True

def graceful_shutdown_handler():
    """Handle interruptions gracefully."""
    
    logger.info("Shutdown requested - saving current progress...")
    save_emergency_checkpoint()
    cleanup_temp_files()
    logger.info("Progress saved. Safe to restart.")
```

## 📊 **Expected Results**

| Aspect | SPLADE Brute Force | Benefits |
|--------|-------------------|----------|
| **Configurations** | ~36,750 | **Complete parameter coverage** |
| **Total Evaluations** | ~2.6 million | **No missed interactions** |
| **Runtime** | 6-12 hours | **Acceptable with checkpointing** |
| **Quality** | Maximum | **Find absolute best settings** |
| **Robustness** | High | **Resume from any interruption** |

## 🚀 **Implementation Priority**

1. **Immediate:** Implement robust checkpointing system
2. **Critical:** Add API usage monitoring and pause capability
3. **Essential:** Progress tracking with ETA calculations
4. **Important:** Graceful shutdown and resume functionality

## 💡 **Key Benefits**

- **Complete coverage** - Test all parameter interactions
- **Bulletproof progress tracking** - Never lose work to interruptions
- **API-safe operation** - Automatic pausing before limits
- **Reproducible results** - Can restart from any point
- **Maximum confidence** - Know you found the absolute best configuration

This brute force approach with robust infrastructure ensures we find the optimal SPLADE configuration while maintaining operational reliability during long test runs.
