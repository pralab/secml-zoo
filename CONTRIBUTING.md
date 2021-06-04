# Contributing to SecML-Zoo

**Your contribution to the development of SecML is foundamental!**

If you wish to contribute with new models and datasets for the SecML zoo please
follow this contribution guide.

We also appreciate the contributions that extend the SecML library.
Please see https://secml.gitlab.io/developers for more information.


## Proposing new models/datasets

Issues should be the starting point for every new proposal. Please describe in detail:
- what kind of models or datasets are you proposing
- for models, specify if a code implementation is already available. See [Adding new models/dataset](#adding-new-modelsdatasets) for the requirements
- for datasets, specify the applicable LICENSE. The content of the SecML zoo
should be covered by Apache License Version 2.0 or equivalent license.

Issue can be opened here: https://gitlab.com/secml/secml-zoo/-/issues


## Adding new models/datasets

Additions to the SecML-Zoo should be sent via a [Merge Request](https://gitlab.com/secml/secml-zoo/-/merge_requests).

Please follow the structure below and be sure that proper LICENSE covers the contribution.
The content of the SecML zoo should be covered by Apache License Version 2.0 or equivalent license.

### Coding convention

All code in SecML-Zoo should follow the same standards and conventions used in 
the SecML library.

The developer's guide covering these aspects is available at: 
https://secml.gitlab.io/developers/contributing.code.html#coding-guidelines

### Structure of SecML-Zoo

The zoo has two main folders:
- `models`, with pre-trained models
- `datasets`, with dataset files

#### Datasets

A new subfolder of `datasets` should be added containing the dataset files and at least 
a `README.md` file explaining the content/purpose.

Datasets can be packed in `CDataset` format directly, by using the `CDataset.save()` method which
exports a `.tar.gz` file, or in any other format. In the latter case please submit the necessary
`CDataLoader` class as a [merge request for SecML](https://gitlab.com/secml/secml/-/merge_requests).

#### Models

Models are defined in the SecML-Zoo via the [models_dict.json](models_dict.json) file.
An entry of the models dictionary file is the following:
```json
"mnist-svm": {
    "model": "svm",
    "state": "mnist/mnist-svm",
    "model_md5": "938ca44db79ee1c2f66dc456ef4d221c",
    "state_md5": "a444f0b8acec44fcdef4444b56df5171"
  }
```

Each item is defined via a model key (`mnist-svm` above), a model path (`svm`), a model 
state path (`mnist/mnist-svm`) and the md5 sum of both the model and the model state 
(`model_md5` and `state_md5`, respectively).

The **model** itself should be a Python file containing a single function which returns 
the `CClassifier` instance which will be pre-trained as required. The file should named
identical to the function name. An example of such model file is [svm.py](models/svm.py):
```python
from secml.ml.classifiers import CClassifierSVM


def svm():
    """Linear Support Vector Machine."""
    return CClassifierSVM()
```

If the model file is specific to a certain dataset, should be put in a separate 
subfolder of the `models` directory. Otherwise can be added directly to the 
`models` folder.

The **model state** should be an pickle export of the pre-trained `CClassifier`.
To obtain it, one can use the `CClassifier.save()` method, which returns 
a `.tar.gz` file. State files should be put in a separate subfolder of the 
`models` directory relative to the specific dataset/configuration.

Finally, for each model, an **exporter** and proper **unittests** should be defined.

### Exporters and Tests

An **exporter** file trains the model on a dataset using a specific configuration 
(if needed) and stores the resulting `CClassifier` instance (model state) as 
explained before. An example is available [here](
https://gitlab.com/secml/secml-zoo/-/blob/master/models/mnist/_exporters/mnist-svm.py).

We also suggest adding a routine to print the md5 hash of the exported `.tar.gz` 
model state file as follows:
```python
from hashlib import md5
md5_hash = md5()
a_file = open(state_path, "rb")  # Path to stored model state file
content = a_file.read()
md5_hash.update(content)

print('md5: ' + md5_hash.hexdigest())
```
The model state hash is required in the `models_dict.json` entry as explained before.

To assess the model performance, especially in case of updates to SecML,
proper unittests should be defined. Test scripts should be put in a `tests` 
subpackage and will be executed automatically by our CI/CD routines.
An example can be found [here](
https://gitlab.com/secml/secml-zoo/-/blob/master/models/mnist/tests/test_models_mnist.py).  

Please avoid loading large datasets as part of the unittests of a model.
Instead, use a small subset of the original dataset which can be put in the 
same folder of the unittests scripts.