"""Entry point for itinerary-density sensitivity."""
from run_gap_experiments import run_study
from ablation_common import STUDY_DENSITY
import argparse
if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('--smoke', action='store_true'); a=p.parse_args(); run_study(STUDY_DENSITY, a.smoke)
