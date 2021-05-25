# iCubWorld

iCubWorld datasets are collections of images recording the visual experience 
of iCub while observing objects in its typical environment, a laboratory 
or an office.

More information: https://robotology.github.io/iCubWorld/#datasets

## Models

We provide pre-trained models on the iCubWorld datasets, which can be loaded
using `CDataLoaderICubWorld` subclasses from `secml`.

The classification system is based on the AlexNet network from `torchvision`,
from which we extract the output of the `classifier:4` layer.
A One-vs-All multiclass classifier is then trained on top of this output.
The pretrained models provided by the zoo only include the multiclass 
classifier. The AlexNet network should be attached to it as a preprocessor 
before use. We provide the utility function `attach_alexnet` function 
from `models.iCubWorld.utils` to facilitate this operation.

Finally, be sure to convert data from numpy's format to pytorch's format 
using `ds_numpy_to_pytorch` function from `models.iCubWorld.utils`.

### iCubWorld28
The following parameters should be passed to the 
`CDataLoaderICubWorld28.load()` method:
- `resize_shape=(256, 256)`
- `crop_shape=(224, 224)`
- `normalize=True`

Available models:
- `icubworld28-day4-svm`: 
  Support Vector Machine trained on 'day4' data.
  Hyperparameters: 'C': 1e-2, 'class_weight': 'balanced'
- `icubworld28-day4-svm-rbf`: 
  Support Vector Machine with RBF kernel trained on 'day4' data.
  Hyperparameters: 'C': 1e3, 'kernel.gamma': 2e-4, 'class_weight': 'balanced'

### iCubWorld7
A reduced version of the iCubWorld28 dataset can be exported by passing
`icub7=True` to the `CDataLoaderICubWorld28.load()` method. This includes
the following objects: `cup3`, `dishwashing-detergent3`, `laundry-detergent3`,
`plate3`, `soap3`, `sponge3`, `sprayer3`.

Available models:
- `icubworld7-day4-svm`: 
  Support Vector Machine trained on 'day4' data.
  Hyperparameters: 'C': 1e-1, 'class_weight': 'balanced'
- `icubworld7-day4-svm-rbf`: 
  Support Vector Machine with RBF kernel trained on 'day4' data.
  Hyperparameters: 'C': 1e2, 'kernel.gamma': 2e-4, 'class_weight': 'balanced'
  