#!/usr/bin/env python3
"""
Main Pipeline Orchestration Script
Runs all phases of the Heat Pump Retrofit Analysis.

Usage:
    python run_pipeline.py              # Run all phases
    python run_pipeline.py --phase 4    # Run only phase 4
    python run_pipeline.py --from 3     # Run from phase 3 onwards
"""

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from config import PROJECT_ROOT, OUTPUT_DIR, FIGURES_DIR, RANDOM_SEED

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(OUTPUT_DIR / 'pipeline.log')
    ]
)
logger = logging.getLogger(__name__)


def run_phase1():
    """Phase 1: Literature Review and Problem Definition."""
    from phase1_literature import main as phase1_main
    return phase1_main()


def run_phase2():
    """Phase 2: Data Collection and Validation."""
    from phase2_data import main as phase2_main
    return phase2_main()


def run_phase3():
    """Phase 3: Feature Engineering."""
    from phase3_features import main as phase3_main
    return phase3_main()


def run_phase4():
    """Phase 4: Model Training and Validation."""
    from phase4_modeling import main as phase4_main
    return phase4_main()


def run_phase5():
    """Phase 5: SHAP, Sobol, and Monte Carlo Analysis."""
    from phase5_analysis import main as phase5_main
    return phase5_main()


def run_phase6():
    """Phase 6: Scenario Enumeration and Economic Evaluation."""
    from phase6_scenarios import main as phase6_main
    return phase6_main()


def run_phase7():
    """Phase 7: Visualization."""
    from phase7_visualization import main as phase7_main
    return phase7_main()


PHASES = {
    1: ('Literature Review', run_phase1),
    2: ('Data Collection', run_phase2),
    3: ('Feature Engineering', run_phase3),
    4: ('Model Training', run_phase4),
    5: ('Sensitivity Analysis', run_phase5),
    6: ('Scenario Evaluation', run_phase6),
    7: ('Visualization', run_phase7)
}


def run_pipeline(start_phase=1, end_phase=7, single_phase=None):
    """Run the complete pipeline or specified phases."""
    
    logger.info("=" * 70)
    logger.info("HEAT PUMP RETROFIT ANALYSIS PIPELINE")
    logger.info(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Random Seed: {RANDOM_SEED}")
    logger.info("=" * 70)
    
    if single_phase:
        phases_to_run = [single_phase]
    else:
        phases_to_run = range(start_phase, end_phase + 1)
    
    results = {}
    total_start = time.time()
    
    for phase_num in phases_to_run:
        if phase_num not in PHASES:
            logger.warning(f"Phase {phase_num} not found, skipping")
            continue
        
        phase_name, phase_func = PHASES[phase_num]
        
        logger.info("\n" + "=" * 70)
        logger.info(f"PHASE {phase_num}: {phase_name.upper()}")
        logger.info("=" * 70)
        
        phase_start = time.time()
        
        try:
            result = phase_func()
            results[phase_num] = {'status': 'SUCCESS', 'result': result}
            phase_time = time.time() - phase_start
            logger.info(f"\n✅ Phase {phase_num} completed in {phase_time:.1f}s")
            
        except Exception as e:
            results[phase_num] = {'status': 'FAILED', 'error': str(e)}
            logger.error(f"\n❌ Phase {phase_num} failed: {e}")
            import traceback
            traceback.print_exc()
            
            # Ask whether to continue
            if phase_num < max(phases_to_run):
                logger.warning("Continuing to next phase...")
    
    total_time = time.time() - total_start
    
    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("PIPELINE SUMMARY")
    logger.info("=" * 70)
    
    for phase_num in phases_to_run:
        if phase_num in results:
            status = results[phase_num]['status']
            symbol = "✅" if status == 'SUCCESS' else "❌"
            phase_name = PHASES[phase_num][0]
            logger.info(f"  {symbol} Phase {phase_num}: {phase_name} - {status}")
    
    logger.info(f"\nTotal time: {total_time:.1f}s ({total_time/60:.1f} min)")
    logger.info(f"Output directory: {OUTPUT_DIR}")
    logger.info(f"Figures directory: {FIGURES_DIR}")
    
    # Count outputs
    output_files = list(OUTPUT_DIR.glob('*.csv')) + list(OUTPUT_DIR.glob('*.txt'))
    figure_files = list(FIGURES_DIR.glob('*.png'))
    
    logger.info(f"\nOutputs generated:")
    logger.info(f"  - Data files: {len(output_files)}")
    logger.info(f"  - Figures: {len(figure_files)}")
    
    logger.info("\n" + "=" * 70)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 70)
    
    return results


def main():
    """Main entry point with argument parsing."""
    
    parser = argparse.ArgumentParser(
        description='Run Heat Pump Retrofit Analysis Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_pipeline.py              # Run all phases (1-7)
    python run_pipeline.py --phase 4    # Run only phase 4
    python run_pipeline.py --from 3     # Run phases 3-7
    python run_pipeline.py --to 5       # Run phases 1-5
    python run_pipeline.py --from 2 --to 5  # Run phases 2-5
        """
    )
    
    parser.add_argument('--phase', type=int, help='Run only this phase')
    parser.add_argument('--from', dest='start', type=int, default=1, 
                       help='Start from this phase (default: 1)')
    parser.add_argument('--to', dest='end', type=int, default=7,
                       help='End at this phase (default: 7)')
    
    args = parser.parse_args()
    
    if args.phase:
        run_pipeline(single_phase=args.phase)
    else:
        run_pipeline(start_phase=args.start, end_phase=args.end)


if __name__ == "__main__":
    main()
