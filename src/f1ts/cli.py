"""
Command-line interface for F1 Tyre Strategy pipeline.
Provides commands for each stage of the data pipeline.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import pandas as pd

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from f1ts import config, ingest, clean, foundation, features
from f1ts import models_degradation, models_pitloss, models_hazards
from f1ts import optimizer, io_flat


def parse_rounds(rounds_str: str) -> List[int]:
    """
    Parse rounds string into list of integers.
    Supports: "1", "1,2,3", "1-5", "1-5,8,10-12"
    
    Args:
        rounds_str: String specification of rounds
    
    Returns:
        List of round numbers
    """
    rounds = []
    for part in rounds_str.split(','):
        part = part.strip()
        if '-' in part:
            start, end = part.split('-')
            rounds.extend(range(int(start), int(end) + 1))
        else:
            rounds.append(int(part))
    return sorted(set(rounds))


def parse_seasons(seasons_str: str) -> List[int]:
    """
    Parse seasons string into list of year integers.
    Supports: "2023", "2022,2023", "2018-2024"
    
    Args:
        seasons_str: String specification of seasons
    
    Returns:
        List of season years
    """
    seasons = []
    for part in seasons_str.split(','):
        part = part.strip()
        if '-' in part:
            start, end = part.split('-')
            seasons.extend(range(int(start), int(end) + 1))
        else:
            seasons.append(int(part))
    return sorted(set(seasons))


def cmd_ingest(args):
    """Ingest race data from FastF1."""
    # Parse seasons and rounds
    seasons = parse_seasons(args.seasons) if hasattr(args, 'seasons') else [args.season]
    rounds = parse_rounds(args.rounds)
    
    print(f"Ingesting races for seasons {seasons}, rounds {rounds}")
    
    # Build race list
    races = [(season, round_num) for season in seasons for round_num in rounds]
    
    print(f"Total races to fetch: {len(races)}")
    
    # Fetch and save
    ingest.fetch_and_save_races(races, session_code=args.session_code)
    
    # Log data manifest
    import json
    metrics_dir = config.paths()['metrics']
    metrics_dir.mkdir(parents=True, exist_ok=True)
    
    manifest = {
        'seasons': seasons,
        'rounds': rounds,
        'total_races': len(races),
        'session_code': args.session_code,
    }
    
    with open(metrics_dir / 'data_manifest.json', 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"\n✓ Ingestion complete: {len(races)} races")
    print(f"✓ Data manifest saved to metrics/data_manifest.json")


def cmd_clean(args):
    """Clean and normalize data."""
    print("Cleaning and normalizing data...")
    
    # Load raw laps
    raw_dir = config.paths()['data_raw']
    laps_files = list(raw_dir.glob('*_laps.parquet'))
    
    if not laps_files:
        print("✗ No lap files found. Run 'ingest' first.")
        return
    
    all_laps = []
    for laps_file in laps_files:
        laps = io_flat.read_parquet(laps_file, verbose=False)
        all_laps.append(laps)
    
    laps_raw = pd.concat(all_laps, ignore_index=True)
    print(f"Loaded {len(laps_raw):,} laps from {len(laps_files)} files")
    
    # Clean
    laps_clean, stints = clean.clean_pipeline(laps_raw)
    
    # Save
    interim_dir = config.paths()['data_interim']
    io_flat.write_parquet(laps_clean, interim_dir / 'laps_interim.parquet')
    io_flat.write_parquet(stints, interim_dir / 'stints_interim.parquet')
    
    print(f"✓ Cleaning complete: {len(laps_clean):,} laps, {len(stints):,} stints")


def cmd_foundation(args):
    """Build foundation tables."""
    print("Building foundation tables...")
    
    import pandas as pd
    
    # Load interim data
    interim_dir = config.paths()['data_interim']
    raw_dir = config.paths()['data_raw']
    
    laps_interim = io_flat.read_parquet(interim_dir / 'laps_interim.parquet')
    sessions = io_flat.read_csv(raw_dir / 'sessions.csv')
    
    # Load weather
    weather_files = list(raw_dir.glob('*_weather.csv'))
    weather_data = []
    for wf in weather_files:
        weather_data.append(pd.read_csv(wf))
    weather_raw = pd.concat(weather_data, ignore_index=True) if weather_data else pd.DataFrame()
    
    # Build foundation
    laps_processed, stints, events = foundation.foundation_pipeline(
        laps_interim, weather_raw, sessions
    )
    
    # Save
    processed_dir = config.paths()['data_processed']
    io_flat.write_parquet(laps_processed, processed_dir / 'laps_processed.parquet')
    io_flat.write_parquet(stints, processed_dir / 'stints.parquet')
    io_flat.write_parquet(events, processed_dir / 'events.parquet')
    
    print(f"✓ Foundation complete: {len(laps_processed):,} laps, {len(events):,} events")


def cmd_features(args):
    """Engineer features."""
    print("Engineering features...")
    
    # Load processed data
    processed_dir = config.paths()['data_processed']
    raw_dir = config.paths()['data_raw']
    
    laps_processed = io_flat.read_parquet(processed_dir / 'laps_processed.parquet')
    sessions = io_flat.read_csv(raw_dir / 'sessions.csv')
    
    # Build features
    pitloss_csv = str(config.paths()['data_lookups'] / 'pitloss_by_circuit.csv')
    hazard_csv = str(config.paths()['data_lookups'] / 'hazard_priors.csv')
    
    stint_features = features.assemble_feature_table(
        laps_processed, sessions, pitloss_csv, hazard_csv
    )
    
    # Save
    features_dir = config.paths()['data_features']
    io_flat.write_parquet(stint_features, features_dir / 'stint_features.parquet')
    
    # Save degradation training subset
    deg_train = stint_features[stint_features['target_deg_ms'].notna()].copy()
    io_flat.write_parquet(deg_train, features_dir / 'degradation_train.parquet')
    
    print(f"✓ Features complete: {len(stint_features):,} rows, {len(stint_features.columns)} columns")


def cmd_model_deg(args):
    """Train degradation model."""
    print("Training degradation model...")
    
    # Load features
    features_dir = config.paths()['data_features']
    deg_train = io_flat.read_parquet(features_dir / 'degradation_train.parquet')
    
    # Train
    model, metrics = models_degradation.train_and_evaluate(deg_train)
    
    # Save
    models_dir = config.paths()['models']
    metrics_dir = config.paths()['metrics']
    
    io_flat.save_model(model, models_dir / 'degradation_v0.pkl')
    io_flat.save_json(metrics, metrics_dir / 'degradation_metrics.json')
    
    print(f"✓ Degradation model trained: MAE = {metrics['mae_s']:.3f}s")


def cmd_pitloss(args):
    """Train pit loss model."""
    print("Training pit loss model...")
    
    import pandas as pd
    
    # Load data
    raw_dir = config.paths()['data_raw']
    
    pitstops_files = list(raw_dir.glob('*_pitstops.csv'))
    all_pitstops = []
    for pf in pitstops_files:
        if pf.stat().st_size > 0:
            all_pitstops.append(pd.read_csv(pf))
    
    if all_pitstops:
        pitstops = pd.concat(all_pitstops, ignore_index=True)
        sessions = io_flat.read_csv(raw_dir / 'sessions.csv')
        
        circuit_avg = models_pitloss.compute_circuit_averages(pitstops, sessions)
        print(f"✓ Pit loss computed for {len(circuit_avg)} circuits")
        
        # Save to lookups
        lookups_dir = config.paths()['data_lookups']
        circuit_avg.to_csv(lookups_dir / 'pitloss_computed.csv', index=False)
    else:
        print("⚠ No pit stop data available")


def cmd_hazards(args):
    """Train hazards model."""
    print("Training hazards model...")
    
    # Load data
    processed_dir = config.paths()['data_processed']
    raw_dir = config.paths()['data_raw']
    
    events = io_flat.read_parquet(processed_dir / 'events.parquet')
    laps = io_flat.read_parquet(processed_dir / 'laps_processed.parquet')
    sessions = io_flat.read_csv(raw_dir / 'sessions.csv')
    
    # Compute hazard rates
    hazard_rates = models_hazards.compute_circuit_hazard_rates(events, laps, sessions)
    
    print(f"✓ Hazard rates computed for {len(hazard_rates)} circuits")
    
    # Save to lookups
    lookups_dir = config.paths()['data_lookups']
    hazard_rates.to_csv(lookups_dir / 'hazard_computed.csv', index=False)


def cmd_optimize(args):
    """Run strategy optimizer."""
    print("Running strategy optimizer...")
    
    # Example optimization
    current_state = {
        'current_lap': 20,
        'total_laps': 57,
        'compounds_available': ['SOFT', 'MEDIUM', 'HARD'],
        'base_lap_time_s': 90.0,
        'deg_rate_ms_per_lap': 50,
        'pit_loss_s': 24.0,
    }
    
    strategies = optimizer.optimize_strategy(current_state, max_stops=2)
    
    # Save
    features_dir = config.paths()['data_features']
    io_flat.write_parquet(strategies, features_dir / 'strategy_decisions.parquet')
    
    print(f"✓ Optimizer complete: {len(strategies)} strategies evaluated")


def cmd_backtest(args):
    """Run backtest."""
    print("Running backtest...")
    
    # Load strategies
    features_dir = config.paths()['data_features']
    strategies = io_flat.read_parquet(features_dir / 'strategy_decisions.parquet')
    
    # Compute regret
    if len(strategies) > 0:
        best_time = strategies['exp_finish_time_s'].min()
        strategies['regret_s'] = strategies['exp_finish_time_s'] - best_time
        
        backtest_summary = {
            'n_strategies_evaluated': len(strategies),
            'best_finish_time_s': float(best_time),
            'mean_regret_s': float(strategies['regret_s'].mean()),
        }
        
        metrics_dir = config.paths()['metrics']
        io_flat.save_json(backtest_summary, metrics_dir / 'backtest_summary.json')
        
        print(f"✓ Backtest complete: Mean regret = {backtest_summary['mean_regret_s']:.2f}s")
    else:
        print("⚠ No strategies to backtest")


def cmd_export(args):
    """Export data for app."""
    print("Exporting data for app...")
    
    import pandas as pd
    
    # Load data
    raw_dir = config.paths()['data_raw']
    processed_dir = config.paths()['data_processed']
    features_dir = config.paths()['data_features']
    
    sessions = io_flat.read_csv(raw_dir / 'sessions.csv')
    laps = io_flat.read_parquet(processed_dir / 'laps_processed.parquet')
    
    # Create app export directory
    app_export_dir = features_dir / 'app_export'
    app_export_dir.mkdir(exist_ok=True)
    
    # Race index
    race_index = sessions[['session_key', 'circuit_name', 'date']].copy()
    io_flat.write_csv(race_index, app_export_dir / 'race_index.csv', verbose=False)
    
    # Export per-race files
    for session_key in sessions['session_key'].unique():
        session_laps = laps[laps['session_key'] == session_key]
        
        if len(session_laps) > 0:
            lap_cols = ['session_key', 'driver', 'lap', 'lap_time_ms', 'compound', 'stint_id', 'tyre_age_laps']
            available_cols = [c for c in lap_cols if c in session_laps.columns]
            io_flat.write_parquet(
                session_laps[available_cols],
                app_export_dir / f'{session_key}_laps.parquet',
                verbose=False
            )
    
    print(f"✓ Export complete for {len(sessions)} races")


def cmd_pipeline(args):
    """Run complete pipeline."""
    print("Running complete pipeline...")
    
    # Run all steps in sequence
    cmd_ingest(args)
    cmd_clean(args)
    cmd_foundation(args)
    cmd_features(args)
    cmd_model_deg(args)
    cmd_pitloss(args)
    cmd_hazards(args)
    cmd_optimize(args)
    cmd_backtest(args)
    cmd_export(args)
    
    print("\n✓ Pipeline complete!")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='F1 Tyre Strategy Pipeline CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Pipeline commands')
    
    # Ingest command
    ingest_parser = subparsers.add_parser('ingest', help='Ingest race data')
    ingest_parser.add_argument('--season', type=int, default=2023, help='Season year (deprecated, use --seasons)')
    ingest_parser.add_argument('--seasons', type=str, default=None, help='Seasons (e.g., 2018-2024 or 2022,2023)')
    ingest_parser.add_argument('--rounds', type=str, default='1-3', help='Rounds (e.g., 1-10 or 1,2,3)')
    ingest_parser.add_argument('--session-code', type=str, default='R', help='Session code (R, Q, etc.)')
    ingest_parser.add_argument('--era-aware', action='store_true', help='Enable era-aware modeling')
    ingest_parser.set_defaults(func=cmd_ingest)
    
    # Clean command
    clean_parser = subparsers.add_parser('clean', help='Clean and normalize data')
    clean_parser.set_defaults(func=cmd_clean)
    
    # Foundation command
    foundation_parser = subparsers.add_parser('foundation', help='Build foundation tables')
    foundation_parser.set_defaults(func=cmd_foundation)
    
    # Features command
    features_parser = subparsers.add_parser('features', help='Engineer features')
    features_parser.set_defaults(func=cmd_features)
    
    # Model degradation command
    model_deg_parser = subparsers.add_parser('model-deg', help='Train degradation model')
    model_deg_parser.set_defaults(func=cmd_model_deg)
    
    # Pit loss command
    pitloss_parser = subparsers.add_parser('pitloss', help='Train pit loss model')
    pitloss_parser.set_defaults(func=cmd_pitloss)
    
    # Hazards command
    hazards_parser = subparsers.add_parser('hazards', help='Train hazards model')
    hazards_parser.set_defaults(func=cmd_hazards)
    
    # Optimize command
    optimize_parser = subparsers.add_parser('optimize', help='Run strategy optimizer')
    optimize_parser.set_defaults(func=cmd_optimize)
    
    # Backtest command
    backtest_parser = subparsers.add_parser('backtest', help='Run backtest')
    backtest_parser.set_defaults(func=cmd_backtest)
    
    # Export command
    export_parser = subparsers.add_parser('export', help='Export data for app')
    export_parser.set_defaults(func=cmd_export)
    
    # Pipeline command
    pipeline_parser = subparsers.add_parser('pipeline', help='Run complete pipeline')
    pipeline_parser.add_argument('--season', type=int, default=2023, help='Season year (deprecated, use --seasons)')
    pipeline_parser.add_argument('--seasons', type=str, default=None, help='Seasons (e.g., 2018-2024 or 2022,2023)')
    pipeline_parser.add_argument('--rounds', type=str, default='1-10', help='Rounds (e.g., 1-10 or 1,2,3)')
    pipeline_parser.add_argument('--session-code', type=str, default='R', help='Session code')
    pipeline_parser.add_argument('--monte-carlo', type=int, default=None, help='Monte Carlo samples for optimizer')
    pipeline_parser.set_defaults(func=cmd_pipeline)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Ensure directories exist
    config.ensure_dirs()
    
    # Run command
    args.func(args)


if __name__ == '__main__':
    main()
