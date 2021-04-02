from pathlib import Path
import os
import argparse
import pickle

import torch
import yaml
from gensim.models.keyedvectors import KeyedVectors
from glove import GloVe
import h5py


def load_config():
    config_filepath = Path(__file__).absolute().parents[1] / "config.yaml"
    with config_filepath.open() as f:
        config_dict = yaml.load(f, Loader=yaml.FullLoader)
    config = argparse.Namespace()
    for key, value in config_dict.items():
        setattr(config, key, value)
    return config


def main():
    config = load_config()
    with open(os.path.join(config.cooccurrence_dir, "vocab.pkl"), "rb") as f:
        vocab = pickle.load(f)

    model = GloVe(
        vocab_size=config.vocab_size,
        embedding_size=config.embedding_size,
        x_max=config.x_max,
        alpha=config.alpha
    )
    model.load_state_dict(torch.load(config.output_filepath))
    
    keyed_vectors = KeyedVectors(vector_size=config.embedding_size)
    keyed_vectors.add_vectors(
        keys=[vocab.get_token(index) for index in range(config.vocab_size)],
        weights=(model.weight.weight.detach()
            + model.weight_tilde.weight.detach()).numpy()
    )
    
    print("How similar is man and woman:")
    print(keyed_vectors.similarity("woman", "man"))
    print("How similar is man and apple:")
    print(keyed_vectors.similarity("apple", "man"))
    print("How similar is woman and apple:")
    print(keyed_vectors.similarity("apple", "woman"))
    for word in ["computer", "united", "early"]:
        print(f"Most similar words of {word}:")
        most_similar_words = [word for word, _ in keyed_vectors.similar_by_word(word)]
        print(most_similar_words)


if __name__ == "__main__":
    main()
