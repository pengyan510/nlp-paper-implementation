# GloVe
To train the model, first fill out the `config.yaml` file. The meaning of each parameter is well documented there. Then run
```
python train.py
```
and it will do the full training and save the word vectors to disk.


If you only want to do the first step, i.e. count the cooccurring pairs, you can run
```
python train.py --first-step-only 
```
Later you can do the second step by calling
```
python train.py --second-step-only
```
