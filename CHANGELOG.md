## 24/07/2020 (v0.13)
- This changelog includes all changes until SecML v0.13 release.

### Added (1 change)
- `mnist-cnn`: `CClassifierPyTorch` CNN trained on half MNIST dataset used in *Sotgiu et al. "Deep neural rejection against adversarial examples", EURASIP J. on Info. Security (2020).*

### Changed (1 change)
- Now using simple `CClassifierSVM` for mnist-svm model as it natively supports multiclass.
- Update models `mnist-svm`, `mnist59-svm` and `mnist59-svm-rbf` after changes to `CClassifierSVM` in v0.13.

### Fixed (1 change)
- #2 `CKernelRBF` is now imported from new package `ml.kernels` in `svm_rbf`. Updated `mnist59-svm-rbf` model.


## 02/12/2019 (v0.12)
- First release of the SecML models and databases zoo.

### Added (4 changes)
- `mnist-svm`: multiclass `CClassifierSVM` trained on MNIST.
- `mnist59-svm`: multiclass `CClassifierSVM` with RBF Kernel trained on MNIST59.
- `mnist59-svm-rbf`: multiclass `CClassifierSVM` with RBF Kernel trained on MNIST59.
- `mnist159-cnn`: `CClassifierPyTorch` CNN trained on MNIST159.
