## DrebinRed: toy dataset of Android malware applications

The dataset contains 12,000 benign and 550 malicious applications,
with 1,227,080 boolean features each. Data is stored in `CSR` sparse format.

It is a subsample of the *Drebin* dataset. See https://www.sec.cs.tu-bs.de/~danarp/drebin/ 
to obtain the full version. 

### How to use
The `drebin-reduced.tar.gz` file is an export in pickle format of the `secml.data.CDataset` object,
containing the samples in sparse format and the labels.

Should be loaded as follows:
  ```python
  ds = CDataset.load('drebin-reduced.tar.gz')
  ```

The dataset header contains a single attribute `feat_desc`. This is a tuple with a string
description of each feature.  
Example: `suspicious_calls::com/dnaml/DNLEpunReader/DnlReaderApp;->getBookLanguage`

### WARNING
`DrebinRed` ia a toy dataset created for demonstration purposes. It should not be used
 as a benchmark for any research or professional work.
