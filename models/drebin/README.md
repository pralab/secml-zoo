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
- `drebin-sec-svm`:
  Secure Support Vector Machine from Demontis et al. "Yes, machine learning 
  can be more secure! a case study on android malware detection."
  IEEE TDSC 2017. https://arxiv.org/abs/1704.08996.  
  Upper and lower bounds ('ub' and 'lb') are chosen accordingly to encourage 
  feature addition and penalize feature removal.  
  Hyperparameters: 'C': 1.0, 'ub': 0.5, 'lb': -0.1