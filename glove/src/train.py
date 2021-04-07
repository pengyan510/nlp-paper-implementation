import argparse
import pickle
import os
from pathlib import Path

import yaml
import matplotlib.pyplot as plt
import torch
import torch.optim
from tqdm import tqdm

from vectorizer import Vectorizer
from cooccurrence_entries import CooccurrenceEntries
from glove import GloVe
from hdf5_dataloader import HDF5DataLoader


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--first-step-only",
        help="only calculate the cooccurrence matrix",
        action="store_true"
    )
    parser.add_argument(
        "--second-step-only",
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


def calculate_cooccurrence(config):
    with open(config.input_filepath, "rb") as f:
        corpus = pickle.load(f)
    vectorizer = Vectorizer.from_corpus(
        corpus=corpus,
        vocab_size=config.vocab_size
    )
    cooccurrence = CooccurrenceEntries.setup(
        corpus=corpus,
        vectorizer=vectorizer
    )
    cooccurrence.build(
        window_size=config.window_size,
        num_partitions=config.num_partitions,
        chunk_size=config.chunk_size,
        output_directory=config.cooccurrence_dir
    ) 


def train_glove(config):
    dataloader = HDF5DataLoader(
        filepath=os.path.join(config.cooccurrence_dir, "cooccurrence.hdf5"),
        dataset_name="cooccurrence",
        batch_size=config.batch_size,
        device=config.device
    )
    model = GloVe(
        vocab_size=config.vocab_size,
        embedding_size=config.embedding_size,
        x_max=config.x_max,
        alpha=config.alpha
    )
    model.to(config.device)
    optimizer = torch.optim.Adagrad(
        model.parameters(),
        lr=config.learning_rate
    )
    with dataloader.open():
        model.train()
        losses = []
        for epoch in tqdm(range(config.num_epochs)):
            epoch_loss = 0
            for batch in tqdm(dataloader.iter_batches()):
                loss = model(
                    batch[0][:, 0],
                    batch[0][:, 1],
                    batch[1]
                )
                epoch_loss += loss.detach().item()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            losses.append(epoch_loss)
            print(f"Epoch {epoch}: loss = {epoch_loss}")
            torch.save(model.state_dict(), config.output_filepath)
    
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()


def main():
    args = parse_args()
    config = load_config()
    if not args.second_step_only:
        calculate_cooccurrence(config)
    if not args.first_step_only:
        train_glove(config)


if __name__ == "__main__":
    main()
