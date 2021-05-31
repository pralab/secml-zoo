# Drebin

Drebin is a dataset of Android malware applications.
See https://www.sec.cs.tu-bs.de/~danarp/drebin/ to obtain the full version.

We provide a script `create_ds.py` which creates a `CDataset` with the proper 
format once the files are obtained from the authors above.

## Models

The models are trained on 60000 samples randomly picked from the full dataset.

- `drebin-svm`:
  Support Vector Machine.
  Hyperparameters: 'C': 1.0