import argparse
from pathlib import Path

import yaml


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--first_step_only",
        help="only calculate the cooccurrence matrix",
        action="store_true"
    )
    parser.add_argument(
        "--second_step_only",
        help="train the word vectors given the cooccurrence matrix",
        action="store_true"
    )
    return parser.parse_args()


def load_config():
    config_filepath = Path(__file__).absolute().parents[1] / "config.yaml"
    with config_filepath.open() as f:
        config_dict = yaml.load(f, Loader=yaml.FullLoader)
    config = argparse.Namespace()
    for key, value in config_dict.items():
        setattr(config, key, value)
    return config


def main():
    args = parse_args()
    config = load_config()
    if not args.second_step_only:



if __name__ == "__main__":
    main()
