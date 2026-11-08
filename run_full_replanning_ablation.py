"""Entry point for the full-replanning ablation."""
from run_gap_experiments import run_study
from ablation_common import STUDY_FULL_REPLANNING
import argparse
if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('--smoke', action='store_true'); a=p.parse_args(); run_study(STUDY_FULL_REPLANNING, a.smoke)
