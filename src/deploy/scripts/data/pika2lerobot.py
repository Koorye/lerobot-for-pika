import sys
sys.path.append('.')

import argparse
import importlib

from src.data.pika_data_processor import PikaDataProcessor


def main(args):
    config = importlib.import_module('scripts.data.configs.' + args.config).DataConfig()
    processor = PikaDataProcessor(config)
    processor.process_data()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process Pika2LeRobot data.")
    parser.add_argument(
        "--config",
        type=str,
        help="Path to the configuration module.",
    )
    args = parser.parse_args()
    main(args)
