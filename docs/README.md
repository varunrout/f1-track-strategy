# Documentation Index

Complete documentation for the F1 Tyre Strategy system.

## Overview

This directory contains comprehensive documentation covering all aspects of the system:
- **Processes**: Data pipeline, model training, optimization
- **Analysis**: Feature engineering, model evaluation, strategy simulation
- **Models**: Architecture, training, evaluation
- **Files**: Schemas, formats, locations
- **Results**: Metrics, quality gates, outputs
- **Impacts**: Design decisions, trade-offs
- **Tests**: Unit tests, validation, quality assurance

## Quick Links

### 🚀 Getting Started
- **New users**: Start with [../README.md](../README.md)
- **Quick setup**: See [../QUICKSTART.md](../QUICKSTART.md)
- **Implementation details**: See [../IMPLEMENTATION.md](../IMPLEMENTATION.md)

### 📚 Documentation Files

| Document | Description | When to Read |
|----------|-------------|--------------|
| [MODULE_DOCUMENTATION.md](MODULE_DOCUMENTATION.md) | Complete reference for all Python modules | When using Python modules or API |
| [NOTEBOOK_GUIDE.md](NOTEBOOK_GUIDE.md) | Comprehensive notebook walkthrough (00-10) | When running notebooks |
| [DATA_SCHEMAS.md](DATA_SCHEMAS.md) | Detailed data contracts and schemas | When working with data files |
| [MODEL_GUIDE.md](MODEL_GUIDE.md) | In-depth model architecture and training | When training or tuning models |
| [TESTING_GUIDE.md](TESTING_GUIDE.md) | Test documentation and quality gates | When writing tests or debugging |
| [APP_USER_GUIDE.md](APP_USER_GUIDE.md) | Streamlit application user manual | When using the web app |
| [ARCHITECTURE.md](ARCHITECTURE.md) | System architecture and design decisions | When understanding system design |
| [TROUBLESHOOTING.md](TROUBLESHOOTING.md) | Common issues and solutions | When encountering errors |

---

## Documentation by Topic

### Data Pipeline

**Understanding the pipeline**:
1. [NOTEBOOK_GUIDE.md](NOTEBOOK_GUIDE.md) - Complete walkthrough
2. [DATA_SCHEMAS.md](DATA_SCHEMAS.md) - Data formats at each stage
3. [MODULE_DOCUMENTATION.md](MODULE_DOCUMENTATION.md) - Module APIs

**Running the pipeline**:
1. [../README.md](../README.md#quickstart) - Quick start commands
2. [NOTEBOOK_GUIDE.md#running-the-complete-pipeline](NOTEBOOK_GUIDE.md#running-the-complete-pipeline)
3. [TROUBLESHOOTING.md#data-pipeline-issues](TROUBLESHOOTING.md#data-pipeline-issues)

### Machine Learning Models

**Model architecture**:
1. [MODEL_GUIDE.md](MODEL_GUIDE.md) - Complete model guide
2. [ARCHITECTURE.md#ml-pipeline](ARCHITECTURE.md#ml-pipeline)
3. [MODULE_DOCUMENTATION.md#models_degradationpy](MODULE_DOCUMENTATION.md#models_degradationpy)

**Training models**:
1. [MODEL_GUIDE.md#model-training-pipeline](MODEL_GUIDE.md#model-training-pipeline)
2. [NOTEBOOK_GUIDE.md#05-model-degradationipynb](NOTEBOOK_GUIDE.md#05-model-degradationipynb)
3. [TROUBLESHOOTING.md#model-training-issues](TROUBLESHOOTING.md#model-training-issues)

**Evaluating models**:
1. [MODEL_GUIDE.md#evaluation-and-metrics](MODEL_GUIDE.md#evaluation-and-metrics)
2. [TESTING_GUIDE.md#quality-gates](TESTING_GUIDE.md#quality-gates)

### Data Schemas

**Understanding data**:
1. [DATA_SCHEMAS.md](DATA_SCHEMAS.md) - Complete schema reference
2. [NOTEBOOK_GUIDE.md](NOTEBOOK_GUIDE.md) - Data transformations
3. [MODULE_DOCUMENTATION.md#io_flatpy](MODULE_DOCUMENTATION.md#io_flatpy)

**Working with data**:
1. [DATA_SCHEMAS.md#example-queries](DATA_SCHEMAS.md#example-queries)
2. [MODULE_DOCUMENTATION.md](MODULE_DOCUMENTATION.md) - Module functions
3. [TROUBLESHOOTING.md#data-quality-issues](TROUBLESHOOTING.md#data-quality-issues)

### Streamlit Application

**Using the app**:
1. [APP_USER_GUIDE.md](APP_USER_GUIDE.md) - Complete user guide
2. [../README.md#streamlit-app-pages](../README.md#streamlit-app-pages)
3. [TROUBLESHOOTING.md#streamlit-app-issues](TROUBLESHOOTING.md#streamlit-app-issues)

**App features**:
1. [APP_USER_GUIDE.md#home-page](APP_USER_GUIDE.md#home-page)
2. [APP_USER_GUIDE.md#race-explorer](APP_USER_GUIDE.md#race-explorer)
3. [APP_USER_GUIDE.md#strategy-sandbox](APP_USER_GUIDE.md#strategy-sandbox)

### Testing and Validation

**Writing tests**:
1. [TESTING_GUIDE.md](TESTING_GUIDE.md) - Complete testing guide
2. [MODULE_DOCUMENTATION.md#validationpy](MODULE_DOCUMENTATION.md#validationpy)
3. [TESTING_GUIDE.md#adding-new-tests](TESTING_GUIDE.md#adding-new-tests)

**Quality assurance**:
1. [TESTING_GUIDE.md#quality-gates](TESTING_GUIDE.md#quality-gates)
2. [MODEL_GUIDE.md#performance-metrics](MODEL_GUIDE.md#performance-metrics)
3. [TESTING_GUIDE.md#validation-framework](TESTING_GUIDE.md#validation-framework)

### System Architecture

**Understanding the system**:
1. [ARCHITECTURE.md](ARCHITECTURE.md) - Complete architecture guide
2. [MODULE_DOCUMENTATION.md](MODULE_DOCUMENTATION.md) - Module organization
3. [ARCHITECTURE.md#design-decisions](ARCHITECTURE.md#design-decisions)

**Design decisions**:
1. [ARCHITECTURE.md#design-decisions](ARCHITECTURE.md#design-decisions)
2. [ARCHITECTURE.md#architecture-principles](ARCHITECTURE.md#architecture-principles)
3. [ARCHITECTURE.md#technology-stack](ARCHITECTURE.md#technology-stack)

### Troubleshooting

**Common issues**:
1. [TROUBLESHOOTING.md](TROUBLESHOOTING.md) - Complete troubleshooting guide
2. [TROUBLESHOOTING.md#faq](TROUBLESHOOTING.md#faq)
3. [TROUBLESHOOTING.md#error-messages](TROUBLESHOOTING.md#error-messages)

**Debugging**:
1. [TESTING_GUIDE.md#troubleshooting](TESTING_GUIDE.md#troubleshooting)
2. [NOTEBOOK_GUIDE.md#troubleshooting](NOTEBOOK_GUIDE.md#troubleshooting)
3. [APP_USER_GUIDE.md#troubleshooting](APP_USER_GUIDE.md#troubleshooting)

---

## Documentation by Persona

### 👨‍💻 Developer

**Start here**:
1. [MODULE_DOCUMENTATION.md](MODULE_DOCUMENTATION.md) - API reference
2. [ARCHITECTURE.md](ARCHITECTURE.md) - System design
3. [TESTING_GUIDE.md](TESTING_GUIDE.md) - Writing tests

**Dive deeper**:
- [DATA_SCHEMAS.md](DATA_SCHEMAS.md) - Data contracts
- [MODEL_GUIDE.md](MODEL_GUIDE.md) - ML implementation
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md) - Debugging

### 📊 Data Scientist

**Start here**:
1. [NOTEBOOK_GUIDE.md](NOTEBOOK_GUIDE.md) - Notebook workflows
2. [MODEL_GUIDE.md](MODEL_GUIDE.md) - Model training
3. [DATA_SCHEMAS.md](DATA_SCHEMAS.md) - Data formats

**Dive deeper**:
- [MODULE_DOCUMENTATION.md#featurespy](MODULE_DOCUMENTATION.md#featurespy) - Feature engineering
- [TESTING_GUIDE.md#quality-gates](TESTING_GUIDE.md#quality-gates) - Model validation
- [MODEL_GUIDE.md#hyperparameter-tuning](MODEL_GUIDE.md#hyperparameter-tuning)

### 🧑‍🔬 Analyst/User

**Start here**:
1. [APP_USER_GUIDE.md](APP_USER_GUIDE.md) - Using the web app
2. [NOTEBOOK_GUIDE.md](NOTEBOOK_GUIDE.md) - Running analysis
3. [TROUBLESHOOTING.md](TROUBLESHOOTING.md) - Common issues

**Dive deeper**:
- [DATA_SCHEMAS.md](DATA_SCHEMAS.md) - Understanding data
- [MODEL_GUIDE.md#overview](MODEL_GUIDE.md#overview) - Model basics

### 🏗️ DevOps/SRE

**Start here**:
1. [ARCHITECTURE.md](ARCHITECTURE.md) - System architecture
2. [TROUBLESHOOTING.md](TROUBLESHOOTING.md) - Operations issues
3. [TESTING_GUIDE.md#cicd-pipeline](TESTING_GUIDE.md#cicd-pipeline)

**Dive deeper**:
- [ARCHITECTURE.md#scalability-considerations](ARCHITECTURE.md#scalability-considerations)
- [ARCHITECTURE.md#future-architecture](ARCHITECTURE.md#future-architecture)

---

## Documentation Statistics

| Document | Lines | Topics Covered |
|----------|-------|----------------|
| MODULE_DOCUMENTATION.md | ~850 | 13 modules, APIs, examples |
| NOTEBOOK_GUIDE.md | ~850 | 11 notebooks, processes, troubleshooting |
| DATA_SCHEMAS.md | ~550 | 15+ schemas, relationships, validation |
| MODEL_GUIDE.md | ~650 | 3 models, training, evaluation |
| TESTING_GUIDE.md | ~550 | Tests, validation, quality gates |
| APP_USER_GUIDE.md | ~400 | 5 pages, features, troubleshooting |
| ARCHITECTURE.md | ~500 | Design, patterns, decisions |
| TROUBLESHOOTING.md | ~500 | Issues, solutions, FAQ |
| **Total** | **~4,850 lines** | **Comprehensive coverage** |

---

## How to Use This Documentation

### First Time Users

1. **Setup**: Read [../README.md](../README.md) for installation
2. **Run**: Follow [NOTEBOOK_GUIDE.md](NOTEBOOK_GUIDE.md) to execute pipeline
3. **Explore**: Use [APP_USER_GUIDE.md](APP_USER_GUIDE.md) for web app

### Developers

1. **Architecture**: Understand [ARCHITECTURE.md](ARCHITECTURE.md)
2. **Modules**: Reference [MODULE_DOCUMENTATION.md](MODULE_DOCUMENTATION.md)
3. **Testing**: Follow [TESTING_GUIDE.md](TESTING_GUIDE.md)

### Data Scientists

1. **Notebooks**: Work through [NOTEBOOK_GUIDE.md](NOTEBOOK_GUIDE.md)
2. **Models**: Study [MODEL_GUIDE.md](MODEL_GUIDE.md)
3. **Data**: Reference [DATA_SCHEMAS.md](DATA_SCHEMAS.md)

### When Things Break

1. **Check**: [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
2. **Search**: Look for error message in documentation
3. **Debug**: Use testing/validation frameworks

---

## Contributing to Documentation

### Documentation Standards

**Good documentation**:
- Clear examples
- Step-by-step instructions
- Error messages and solutions
- Cross-references to related docs

**Adding new documentation**:
1. Create markdown file in `docs/`
2. Follow existing structure and style
3. Add to this index
4. Update main README.md links

### Documentation Templates

**Module documentation**:
```markdown
## module_name.py

**Purpose**: Brief description

### Functions

#### function_name()
Description and usage example
```

**Troubleshooting entry**:
```markdown
### Error: "Error message"

**Symptoms**: What user sees

**Cause**: Why it happens

**Solution**: How to fix
```

---

## Additional Resources

### External Documentation

- [FastF1 Documentation](https://docs.fastf1.dev/) - F1 data API
- [LightGBM Documentation](https://lightgbm.readthedocs.io/) - ML library
- [Streamlit Documentation](https://docs.streamlit.io/) - Web app framework
- [pandas Documentation](https://pandas.pydata.org/docs/) - Data manipulation

### Related Files

- [../README.md](../README.md) - Project overview and quickstart
- [../QUICKSTART.md](../QUICKSTART.md) - Installation and setup
- [../IMPLEMENTATION.md](../IMPLEMENTATION.md) - Implementation summary
- [../LICENSE](../LICENSE) - MIT License
- [../requirements.txt](../requirements.txt) - Dependencies

---

## Documentation Changelog

**Version 1.0** (Current):
- Complete documentation suite
- 8 comprehensive guides
- ~4,850 lines of documentation
- Coverage: modules, notebooks, data, models, testing, app, architecture, troubleshooting

**Future additions**:
- Video tutorials
- API reference (OpenAPI spec)
- Performance tuning guide
- Deployment guide
- Migration guides

---

## Getting Help

**Documentation not clear?**
1. Open GitHub issue with "docs" label
2. Suggest improvements
3. Submit pull request with updates

**Found an error?**
1. Check if it's a typo or outdated info
2. Open issue or submit PR to fix

**Need more examples?**
1. Check existing code in `src/` and `notebooks/`
2. Request specific examples in GitHub issues

---

## Summary

This documentation index provides:
- **Quick links** to all documentation
- **Topic-based navigation**
- **Persona-based guides**
- **Usage instructions**
- **Contributing guidelines**

**Total documentation**: 8 comprehensive guides covering all aspects of the F1 Tyre Strategy system.

Start with the document most relevant to your needs, and follow cross-references for deeper understanding.
