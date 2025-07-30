import sys
sys.path.append('.')

import argparse
import importlib
import time

from src.deploy.client import Client


def main(args):
    config = importlib.import_module('scripts.deploy.configs.' + args.config).DeployConfig()
    client = Client(
        camera=config.camera,
        robot=config.robot,
        host=config.host,
        port=config.port,
        frequency=config.frequency,
        prompt=config.prompt,
        visualize_every=config.visualize_every,
    )

    time.sleep(3)
    input('Press Enter to start the client...')
    client.run()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Deploy the OpenPI client.')
    parser.add_argument(
        '--config', 
        type=str, 
        help='Path to the configuration file.'
    )
    args = parser.parse_args()
    main(args)
    