import sys
sys.path.append('.')

import argparse

from src.data.configuration_data_processor import RGBMultiArmDeltaGripperDataProcessorConfig
from src.data.pika_data_processor import PikaDataProcessor


def main(args):
    config = RGBMultiArmDeltaGripperDataProcessorConfig(
        source_data_roots=args.config.source_data_roots,
    )
    processor = PikaDataProcessor(config)
    processor.process_data()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process Pika2LeRobot data.")
    parser.add_argument(
        '--source_data_roots',
        type=str,
        nargs='+',
        required=True,
        help='List of source data directories to process.'
    )
    args = parser.parse_args()
    main(args)
