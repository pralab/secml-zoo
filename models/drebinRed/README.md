# DrebinRed

DrebinRed is a toy dataset of Android malware applications, available in this
repository at: `datasets/DrebinRed`.

It is a subsample of the *Drebin* dataset.
See https://www.sec.cs.tu-bs.de/~danarp/drebin/ to obtain the full version.

## Models

The models are trained on 6000 samples randomly picked from the full dataset.

- `drebin-reduced-svm`:
  Support Vector Machine.
  Hyperparameters: 'C': 0.1