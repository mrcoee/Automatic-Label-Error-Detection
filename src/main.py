import argparse
from config import cfg
from prepare_data import load_data
from analyze_metrics import evaluate

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--load", help="Load Data", action="store_true")
    parser.add_argument("--metrics", help="Calculate metrics from loaded data", action="store_true")
    parser.add_argument("--evaluate", help="Evaluate metrics and produce label error proposals", action="store_true")
    
    args = parser.parse_args()
    if args.load:
        load_data()
    if args.evaluate:
        evaluate()

if __name__ == '__main__':
    main()