"""Creating the demo data

For simplicity, the input data to the glove training procedure is assumed to
a pickle file storing a python list, which contains all the tokens.
For demo purpose, the token list is created from the Text8 dataset.
"""
import pickle
import itertools
from pathlib import Path

import gensim.downloader as api


dataset = api.load("text8")
corpus = list(itertools.chain.from_iterable(dataset))

demo_data = Path(__file__).absolute().parents[1] / "data" / "demo_data.pkl" 
with demo_data.open("wb+") as f:
    pickle.dump(corpus, f)

