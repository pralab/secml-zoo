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
- `drebin-reduced-sec-svm`:
  Secure Support Vector Machine from Demontis et al. "Yes, machine learning 
  can be more secure! a case study on android malware detection."
  IEEE TDSC 2017. https://arxiv.org/abs/1704.08996.  
  Upper and lower bounds ('ub' and 'lb') are chosen accordingly to encourage 
  feature addition and penalize feature removal.  
  Hyperparameters: 'C': 0.1, 'ub': 0.5, 'lb': -0.1