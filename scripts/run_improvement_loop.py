"""CLI script to execute the automated retrieval self-improvement loop."""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.improvement.runner import ImprovementLoopRunner, is_protected_artifact

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run_improvement_loop")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run automated retrieval self-improvement loop for JioStar Contract Assistant (Safe Replay Mode)."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/improvement_search.yaml",
        help="Path to improvement search configuration file.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=5,
        help="Number of improvement iterations to execute beyond baseline (minimum 5).",
    )
    parser.add_argument(
        "--log-path",
        type=str,
        default="experiments/replay_log.jsonl",
        help="Path for append-only JSONL replay log (default: experiments/replay_log.jsonl). Frozen submission logs are protected from overwrite.",
    )
    parser.add_argument(
        "--summary-path",
        type=str,
        default="experiments/replay_summary.json",
        help="Path for final summary JSON report (default: experiments/replay_summary.json). Frozen submission summary is protected from overwrite.",
    )
    parser.add_argument(
        "--final-config-path",
        type=str,
        default="configs/replay_final.yaml",
        help="Path to output final selected YAML configuration (default: configs/replay_final.yaml). Frozen configs/final.yaml is protected from overwrite.",
    )

    args = parser.parse_args()

    # Guard against accidental overwrite of frozen submission artifacts
    for label, target_path in [
        ("Log path", args.log_path),
        ("Summary path", args.summary_path),
        ("Final config path", args.final_config_path),
    ]:
        if is_protected_artifact(target_path):
            logger.error(
                f"[SECURITY ERROR] {label} '{target_path}' targets an immutable frozen submission artifact! "
                f"Replaying the self-improvement loop must not overwrite original evaluation evidence. "
                f"Please specify a non-destructive destination (e.g., experiments/replay_log.jsonl, "
                f"experiments/replay_summary.json, configs/replay_final.yaml)."
            )
            return 1

    logger.info("Initializing Retrieval Self-Improvement Loop...")
    try:
        runner = ImprovementLoopRunner(
            config_path=args.config,
            iterations=args.iterations,
            log_path=args.log_path,
            summary_path=args.summary_path,
            final_config_path=args.final_config_path,
        )
    except PermissionError as pe:
        logger.error(f"[SECURITY ERROR] {pe}")
        return 1

    try:
        summary = runner.run()
        logger.info("\n" + "=" * 70)
        logger.info("SELF-IMPROVEMENT LOOP COMPLETED SUCCESSFULLY")
        logger.info(f"Total Iterations: {summary['total_iterations_executed']}")
        logger.info(f"Accepted: {summary['iterations_accepted']}")
        logger.info(f"Rejected: {summary['iterations_rejected']}")
        logger.info(f"Final Configuration written to: {args.final_config_path}")
        logger.info(f"Full Experiment Summary saved to: {args.summary_path}")
        logger.info("=" * 70)
        return 0
    except PermissionError as pe:
        logger.error(f"[SECURITY ERROR] {pe}")
        return 1
    except Exception as e:
        logger.error(f"Critical failure during self-improvement loop execution: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
