# System Architecture

Complete overview of the F1 Tyre Strategy system architecture, design decisions, and implementation patterns.

## Table of Contents

1. [System Overview](#system-overview)
2. [Architecture Principles](#architecture-principles)
3. [Component Design](#component-design)
4. [Data Flow](#data-flow)
5. [Module Organization](#module-organization)
6. [Design Patterns](#design-patterns)
7. [Technology Stack](#technology-stack)
8. [Design Decisions](#design-decisions)
9. [Scalability Considerations](#scalability-considerations)
10. [Future Architecture](#future-architecture)

---

## System Overview

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER INTERFACES                          │
├─────────────────────────────────────────────────────────────────┤
│  Jupyter Notebooks  │    CLI Interface    │  Streamlit App      │
│   (Interactive)     │   (Automation)      │  (Visualization)    │
└──────────┬──────────┴──────────┬──────────┴──────────┬──────────┘
           │                     │                     │
           └──────────────┬──────┴──────┬──────────────┘
                          │             │
                    ┌─────▼─────────────▼─────┐
                    │   CORE PYTHON MODULES   │
                    │      (src/f1ts/)        │
                    └─────┬──────────────┬────┘
                          │              │
              ┌───────────┴───┐    ┌────▼────────┐
              │  DATA LAYER   │    │  ML MODELS  │
              │  (Parquet/CSV)│    │  (LightGBM) │
              └───────────────┘    └─────────────┘
                     │
              ┌──────▼──────┐
              │  FastF1 API │
              │ (External)  │
              └─────────────┘
```

### System Layers

1. **Data Layer**: Flat files (Parquet, CSV)
2. **Core Layer**: Python modules (data processing, ML)
3. **Interface Layer**: Notebooks, CLI, Web app
4. **External Layer**: FastF1 API (data source)

---

## Architecture Principles

### 1. Flat File Architecture

**Principle**: No databases, all data in files

**Benefits**:
- ✅ Easy inspection (view files directly)
- ✅ Version control friendly (git LFS)
- ✅ Simple deployment (no DB setup)
- ✅ Portable (copy directory = copy everything)

**Trade-offs**:
- ❌ No ACID transactions
- ❌ No concurrent writes
- ❌ Limited query optimization

**Rationale**: For ML research and analysis, simplicity > database features.

### 2. Reproducibility First

**Principle**: All results must be reproducible

**Implementation**:
- Fixed random seeds (`RANDOM_SEED = 42`)
- Environment variable (`PYTHONHASHSEED=0`)
- Pinned dependencies (`requirements.txt`)
- Deterministic algorithms (LightGBM)

**Validation**: Run pipeline twice → same outputs (bit-for-bit)

### 3. Validation-Driven Development

**Principle**: Validate early and often

**Pattern**: Every notebook includes validation cells
```python
# After each transformation
validation.validate_schema(df, required_cols)
validation.validate_unique_key(df, key_cols)
```

**Benefits**:
- Catch data quality issues immediately
- Clear error messages
- Self-documenting expectations

### 4. Separation of Concerns

**Principle**: Each module has single responsibility

**Example**:
- `ingest.py`: Only fetches data
- `clean.py`: Only cleans data
- `features.py`: Only engineers features
- No mixing responsibilities

### 5. Configuration as Code

**Principle**: All configuration in `config.py`

**No hardcoded values** in notebooks or modules:
```python
# ✅ Good
threshold = config.DEG_MAE_THRESHOLD

# ❌ Bad
threshold = 0.08  # Hardcoded
```

---

## Component Design

### Data Layer

**Structure**:
```
data/
├── raw/          # Immutable source data
├── interim/      # Cleaned but not joined
├── processed/    # Joined and enriched
├── features/     # Model-ready features
└── lookups/      # Reference data
```

**Design pattern**: Sequential refinement
- Each stage adds value
- Previous stages never modified
- Clear lineage: raw → interim → processed → features

**File format choices**:
- **Parquet**: Large tables (laps, stints, features)
  - Columnar storage = fast filtering
  - Built-in compression
  - Type-safe
- **CSV**: Small tables (sessions, lookups)
  - Human-readable
  - Git-friendly
  - Easy to edit

### Core Modules

**Layered architecture**:
```
┌─────────────────────────────────────────┐
│           Application Layer             │
│  (cli.py, notebook cells)               │
├─────────────────────────────────────────┤
│          Business Logic Layer           │
│  (optimizer.py, models_*.py)            │
├─────────────────────────────────────────┤
│           Data Processing Layer         │
│  (clean.py, features.py, foundation.py) │
├─────────────────────────────────────────┤
│             Utility Layer               │
│  (io_flat.py, validation.py, utils.py) │
├─────────────────────────────────────────┤
│          Configuration Layer            │
│              (config.py)                │
└─────────────────────────────────────────┘
```

**Dependencies flow downward only** (no circular dependencies).

### ML Pipeline

**Training pipeline**:
```
Data → Features → Train → Validate → Save
  ↓                          ↓
Validation              Quality Gate
```

**Inference pipeline**:
```
New Data → Features → Load Model → Predict → Use
```

---

## Data Flow

### End-to-End Pipeline

```
[FastF1 API]
     ↓
[01_ingest] → data/raw/
     ↓
[02_clean] → data/interim/
     ↓
[03_foundation] → data/processed/
     ↓
[04_features] → data/features/
     ↓
[05-07_models] → models/, metrics/
     ↓
[08_optimizer] → data/features/strategy_decisions.parquet
     ↓
[09_backtest] → metrics/backtest_summary.json
     ↓
[10_export] → app-ready files
     ↓
[Streamlit App] → Visualizations
```

### Data Transformations

**Key transformations**:

1. **Ingestion** (`ingest.py`):
   - FastF1 objects → Parquet files
   - API calls → Cached data

2. **Cleaning** (`clean.py`):
   - Raw compounds → Standard compounds (SOFT/MEDIUM/HARD)
   - Lap sequences → Stints
   - Outlier detection and removal

3. **Foundation** (`foundation.py`):
   - Laps + Weather → Laps with weather
   - Laps → Events (SC, VSC)
   - Laps → Stint aggregations

4. **Features** (`features.py`):
   - Laps → Rolling features
   - Laps → Degradation estimates
   - Laps + Lookups → Complete features

5. **Modeling** (`models_*.py`):
   - Features → Trained models
   - Models → Predictions

6. **Optimization** (`optimizer.py`):
   - Strategies × Models → Simulated times
   - Simulations → Ranked strategies

---

## Module Organization

### Module Dependency Graph

```
config.py (no dependencies)
    ↓
utils.py (depends on: config)
    ↓
io_flat.py (depends on: config, utils)
    ↓
validation.py (depends on: config)
    ↓
ingest.py (depends on: config, io_flat, utils)
    ↓
clean.py (depends on: config, validation)
    ↓
foundation.py (depends on: config, validation)
    ↓
features.py (depends on: config)
    ↓
models_*.py (depends on: config, validation, io_flat)
    ↓
optimizer.py (depends on: config, models_*)
    ↓
cli.py (depends on: all modules)
```

### Module Interfaces

**Consistent patterns**:

**Data processing modules**:
```python
def process(input_df: pd.DataFrame) -> pd.DataFrame:
    """
    Single input, single output.
    Pure function (no side effects).
    """
    result = transform(input_df)
    return result
```

**I/O modules**:
```python
def read_data(path: Path) -> pd.DataFrame:
    """Load data with logging."""
    
def write_data(df: pd.DataFrame, path: Path) -> None:
    """Save data with logging."""
```

**Model modules**:
```python
def train(X, y, **params) -> model:
    """Train and return model."""
    
def predict(model, X) -> predictions:
    """Generate predictions."""
    
def evaluate(model, X, y) -> metrics:
    """Compute performance metrics."""
```

---

## Design Patterns

### 1. Pipeline Pattern

**Usage**: Sequential data transformations

**Example** (`clean.py`):
```python
def clean_pipeline(laps_raw):
    laps = standardize_compounds(laps_raw)
    laps = derive_stints(laps)
    laps = attach_tyre_age(laps)
    laps = fix_dtypes(laps)
    laps = remove_outliers(laps)
    return laps
```

**Benefits**:
- Clear flow
- Easy to test each step
- Can add/remove steps

### 2. Factory Pattern

**Usage**: Model creation

**Example** (`models_degradation.py`):
```python
def create_model(params):
    return lgb.LGBMRegressor(**params)
```

### 3. Strategy Pattern

**Usage**: Different training strategies

**Example**:
```python
# Strategy 1: Simple train/test
model = train(X_train, y_train)

# Strategy 2: Cross-validation
model, metrics = train_with_cv(X, y, groups)
```

### 4. Template Method Pattern

**Usage**: Validation framework

**Example** (`validation.py`):
```python
def validate_table(df, schema):
    validate_columns(df, schema.required_cols)
    validate_dtypes(df, schema.dtype_map)
    validate_unique_key(df, schema.key_cols)
    # Template: steps always run in order
```

### 5. Facade Pattern

**Usage**: CLI simplifies complex operations

**Example** (`cli.py`):
```python
def cmd_pipeline(args):
    """Simple interface to complex multi-step process."""
    cmd_ingest(args)
    cmd_clean(args)
    cmd_foundation(args)
    # ... etc
```

---

## Technology Stack

### Core Technologies

| Layer | Technology | Version | Purpose |
|-------|------------|---------|---------|
| Language | Python | 3.11+ | Core language |
| Data | pandas | 2.1.4 | DataFrames |
| Data | NumPy | 1.26+ | Numerical computing |
| ML | LightGBM | 4.1.0 | Gradient boosting |
| ML | scikit-learn | 1.3+ | ML utilities |
| Data Source | FastF1 | 3.2.0 | F1 data API |
| Web | Streamlit | 1.28+ | Web app framework |
| Notebooks | JupyterLab | 4.0+ | Interactive development |

### File Formats

| Format | Use Case | Library |
|--------|----------|---------|
| Parquet | Large tables | pandas, pyarrow |
| CSV | Small tables, lookups | pandas |
| JSON | Metrics, config | json |
| Pickle | Trained models | pickle |

### Development Tools

| Tool | Purpose |
|------|---------|
| ruff | Linting |
| mypy | Type checking |
| pytest | Unit testing |
| black | Code formatting (optional) |
| git | Version control |

---

## Design Decisions

### Why Flat Files?

**Decision**: Use Parquet/CSV instead of database

**Rationale**:
- **Simplicity**: No DB setup, maintenance, or admin
- **Portability**: Copy directory = copy everything
- **Version control**: Git LFS for data versioning
- **Inspection**: View files directly with pandas
- **Scale**: Sufficient for ML research (< 1GB data)

**Trade-off**: No SQL queries, but pandas provides similar functionality.

### Why LightGBM?

**Decision**: Use LightGBM for degradation model

**Rationale**:
- **Categorical support**: Handles compound, circuit natively
- **Performance**: Fast training and inference
- **Quality**: SOTA gradient boosting
- **Memory efficient**: Histogram-based learning
- **Feature importance**: Built-in SHAP values

**Alternatives considered**:
- XGBoost: Similar performance, slightly slower
- RandomForest: Less accurate for tabular data
- Neural networks: Overkill for small dataset

### Why Notebooks + Modules?

**Decision**: Hybrid approach (notebooks for exploration, modules for reuse)

**Rationale**:
- **Notebooks**: Great for exploration, visualization, documentation
- **Modules**: Great for reusability, testing, production
- **Best of both**: Use both where appropriate

**Pattern**:
```python
# In notebook
from f1ts import features
df = features.add_rolling_pace(df)  # Calls module function
```

### Why No Database?

**Decision**: Flat files instead of PostgreSQL/MongoDB

**Rationale**:
- **Small data**: < 1GB total, fits in memory
- **Read-heavy**: Mostly sequential reads (pipelines)
- **Single user**: No concurrent access needed
- **Research focus**: Iterating on features, not serving requests

**When to reconsider**: If scaling to 100+ races or 100+ concurrent users.

### Why Streamlit?

**Decision**: Use Streamlit for web app

**Rationale**:
- **Fast prototyping**: Build app in hours, not days
- **Python-native**: No HTML/CSS/JS required
- **Interactive**: Widgets and state management built-in
- **Visualization**: Integrates with matplotlib, plotly

**Alternatives considered**:
- Flask/FastAPI: More flexible but requires more code
- Dash: Similar to Streamlit, slightly heavier
- Jupyter widgets: Limited interactivity

---

## Scalability Considerations

### Current Limits

**Data scale**:
- ✅ 10-20 races: Excellent
- ✅ 50-100 races: Good
- ⚠️ 200+ races: May need optimization
- ❌ 1000+ races: Requires database

**Concurrent users**:
- ✅ 1-5 users: Excellent
- ⚠️ 10-20 users: May slow down
- ❌ 50+ users: Requires deployment architecture

### Optimization Strategies

**If data grows**:
1. Use Dask for larger-than-memory processing
2. Implement lazy loading (only load needed data)
3. Add database (PostgreSQL with parquet extension)

**If users grow**:
1. Deploy Streamlit with load balancing
2. Add Redis for caching
3. Separate compute from serving (microservices)

**If compute grows**:
1. Use joblib for parallel processing
2. GPU-accelerate LightGBM (if available)
3. Distribute training (Ray, Dask)

---

## Future Architecture

### Planned Enhancements

**Real-time data**:
```
[Live Race Feed] → [Streaming Pipeline] → [Real-time Dashboard]
         ↓
  [Kafka/Redis]
```

**API Layer**:
```
[REST API]
    ↓
[FastAPI Backend]
    ↓
[Model Serving] (MLflow, BentoML)
```

**Multi-user deployment**:
```
[Load Balancer]
    ↓
[Streamlit Instances] × N
    ↓
[Shared Data Store] (S3, PostgreSQL)
    ↓
[Model Registry] (MLflow)
```

### Long-term Vision

**Phase 1** (Current): Single-user research system
**Phase 2**: Multi-user web application
**Phase 3**: Real-time race strategy API
**Phase 4**: Mobile app and integrations

---

## Summary

This architecture document covers:
- **System overview** and component design
- **Architecture principles** and patterns
- **Technology choices** and rationale
- **Design decisions** and trade-offs
- **Scalability** considerations
- **Future architecture** plans

For implementation details, see [MODULE_DOCUMENTATION.md](MODULE_DOCUMENTATION.md).

For data design, see [DATA_SCHEMAS.md](DATA_SCHEMAS.md).

For model architecture, see [MODEL_GUIDE.md](MODEL_GUIDE.md).
