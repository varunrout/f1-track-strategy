#!/usr/bin/env python3
"""
F1 Tyre Strategy - Project Validation Script
Validates the complete project structure and requirements.
"""

import sys
from pathlib import Path

def validate_structure():
    """Validate directory structure."""
    print("="*70)
    print("F1 TYRE STRATEGY - PROJECT VALIDATION")
    print("="*70)
    print()
    
    required_dirs = [
        'app', 'app/pages', 'data/raw', 'data/interim', 'data/processed',
        'data/features', 'data/lookups', 'notebooks', 'src/f1ts', 
        'models', 'metrics'
    ]
    
    print("📁 Checking Directory Structure...")
    all_dirs_ok = True
    for d in required_dirs:
        exists = Path(d).is_dir()
        status = "✓" if exists else "✗"
        print(f"  {status} {d}")
        if not exists:
            all_dirs_ok = False
    
    return all_dirs_ok


def validate_core_files():
    """Validate core project files."""
    print("\n📄 Checking Core Files...")
    
    core_files = [
        'README.md', 'QUICKSTART.md', 'LICENSE', 'Makefile',
        'requirements.txt', '.gitignore'
    ]
    
    all_ok = True
    for f in core_files:
        exists = Path(f).is_file()
        status = "✓" if exists else "✗"
        print(f"  {status} {f}")
        if not exists:
            all_ok = False
    
    return all_ok


def validate_modules():
    """Validate Python modules."""
    print("\n🐍 Checking Python Modules...")
    
    modules = [
        'src/f1ts/__init__.py',
        'src/f1ts/config.py',
        'src/f1ts/io_flat.py',
        'src/f1ts/validation.py',
        'src/f1ts/utils.py',
        'src/f1ts/ingest.py',
        'src/f1ts/clean.py',
        'src/f1ts/foundation.py',
        'src/f1ts/features.py',
        'src/f1ts/models_degradation.py',
        'src/f1ts/models_pitloss.py',
        'src/f1ts/models_hazards.py',
        'src/f1ts/optimizer.py',
    ]
    
    all_ok = True
    for m in modules:
        exists = Path(m).is_file()
        status = "✓" if exists else "✗"
        print(f"  {status} {m}")
        if not exists:
            all_ok = False
    
    return all_ok


def validate_notebooks():
    """Validate notebooks."""
    print("\n📓 Checking Notebooks...")
    
    notebooks = [
        f'notebooks/{i:02d}_{name}.ipynb' 
        for i, name in [
            (0, 'setup_env'),
            (1, 'ingest_fastf1'),
            (2, 'clean_normalize'),
            (3, 'build_foundation_sets'),
            (4, 'features_stint_lap'),
            (5, 'model_degradation'),
            (6, 'model_pitloss'),
            (7, 'model_hazards'),
            (8, 'strategy_optimizer'),
            (9, 'backtest_replay'),
            (10, 'export_for_app'),
        ]
    ]
    
    all_ok = True
    for nb in notebooks:
        exists = Path(nb).is_file()
        status = "✓" if exists else "✗"
        print(f"  {status} {nb}")
        if not exists:
            all_ok = False
    
    return all_ok


def validate_app():
    """Validate Streamlit app."""
    print("\n🖥️  Checking Streamlit App...")
    
    app_files = [
        'app/Home.py',
        'app/pages/1_Race_Explorer.py',
        'app/pages/2_Strategy_Sandbox.py',
        'app/pages/3_Model_QC.py',
        'app/pages/4_Data_Health.py',
    ]
    
    all_ok = True
    for f in app_files:
        exists = Path(f).is_file()
        status = "✓" if exists else "✗"
        print(f"  {status} {f}")
        if not exists:
            all_ok = False
    
    return all_ok


def validate_lookups():
    """Validate lookup files."""
    print("\n📊 Checking Lookup Files...")
    
    lookups = [
        'data/lookups/pitloss_by_circuit.csv',
        'data/lookups/hazard_priors.csv',
    ]
    
    all_ok = True
    for f in lookups:
        exists = Path(f).is_file()
        status = "✓" if exists else "✗"
        
        if exists:
            # Check if file has content
            size = Path(f).stat().st_size
            if size > 0:
                print(f"  {status} {f} ({size} bytes)")
            else:
                print(f"  ✗ {f} (empty file)")
                all_ok = False
        else:
            print(f"  {status} {f}")
            all_ok = False
    
    return all_ok


def count_total_files():
    """Count total files in project."""
    print("\n📈 Project Statistics...")
    
    py_files = list(Path('src').rglob('*.py'))
    notebooks = list(Path('notebooks').rglob('*.ipynb'))
    app_files = list(Path('app').rglob('*.py'))
    
    print(f"  Python modules: {len(py_files)}")
    print(f"  Notebooks: {len(notebooks)}")
    print(f"  App files: {len(app_files)}")
    print(f"  Total Python code: {len(py_files) + len(app_files)} files")


def main():
    """Run all validations."""
    results = []
    
    results.append(("Directory Structure", validate_structure()))
    results.append(("Core Files", validate_core_files()))
    results.append(("Python Modules", validate_modules()))
    results.append(("Notebooks", validate_notebooks()))
    results.append(("Streamlit App", validate_app()))
    results.append(("Lookup Files", validate_lookups()))
    
    count_total_files()
    
    # Summary
    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)
    
    all_passed = True
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {status:12s} {name}")
        if not passed:
            all_passed = False
    
    print("="*70)
    
    if all_passed:
        print("✓ ALL VALIDATIONS PASSED - Project is ready!")
        print("\nNext steps:")
        print("  1. Install dependencies: pip install -r requirements.txt")
        print("  2. Run notebooks 00-10 in sequence")
        print("  3. Launch Streamlit app: streamlit run app/Home.py")
        return 0
    else:
        print("✗ VALIDATION FAILED - Some components are missing")
        return 1


if __name__ == '__main__':
    sys.exit(main())
