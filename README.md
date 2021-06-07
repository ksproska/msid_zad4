# INTRODUCTION

***Aim:*** By using available libraries (in my example mainly [scikit-learn](https://pypi.org/project/scikit-learn/)) create a model for [Fashion Mnist](https://github.com/zalandoresearch/fashion-mnist) data (matrixes containging matrices of flattened 28x28 pictures of clothing).

Fashion mnist contains 4 matrices, downloaded from

https://github.com/zalandoresearch/fashion-mnist/tree/master/data/fashion to [original](original):

     └─── original
          ├─── t10k-images-idx3-ubyte.gz
          ├─── t10k-labels-idx1-ubyte.gz
          ├─── train-images-idx3-ubyte.gz
          └─── train-labels-idx1-ubyte.gz

Because of github upload limitations in my repository only 3/4 files are added.

Using method [load_nist](feature_extraction.py/#L8) we extract [4 matrices](feature_extraction.py#L118):
| name | size |
| --- | --- |
| X_train | 60000x784 |
| y_train | 60000x1 |
| X_test | 10000x784 |
| y_test | 10000x1 |

- **X_... matrices** contain an image for each row as a flattened numpay array (from 2D 28x28 pixels image [->](feature_extraction.py#L52) 1x784; pixel values (0-255))
- **y_... matrices** contain numbers of labels from [table](using_models.py#L23) for each picture in **X_... matrices**:

```python
['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
```
# METHODS

## Feature extraction from picture.
### [No preprocessing](feature_extraction.py/#L39) (for kNN is unnecessary)
```python
def load_original(path='original'):
    X_train, y_train = load_mnist(path, kind='train')
    X_test, y_test = load_mnist(path, kind='t10k')
    return X_train, y_train, X_test, y_test
```
![image](https://user-images.githubusercontent.com/61067969/120994600-ed8d4100-c784-11eb-9b92-e77162947ef7.png)

### [Preprocessing](feature_extraction.py#L66)
[enchancing contrast](feature_extraction.py/#L85) (using cv2 library); for different variables: 9, 12
```python
def contrast(mx, contr):
    return cv2.filter2D(mx, -1, np.array([[-1, -1, -1], [-1, contr, -1], [-1, -1, -1]]))
```
![image](https://user-images.githubusercontent.com/61067969/120994299-a737e200-c784-11eb-961c-3aa0c0ef9767.png)

[HOG](feature_extraction.py/#L95) = Histogram of oriented gradients (from skimage.feature import hog); for 9 orientations, 2x2, 4x4, 6x6 pixel rates
```python
def hog_features(mx, orientations, pixel_cell, cell_block):
    fd, hog_image = hog(mx, orientations=orientations, pixels_per_cell=(pixel_cell, pixel_cell),
                        cells_per_block=(cell_block, cell_block), visualize=True, multichannel=True)
    return hog_image
```
![image](https://user-images.githubusercontent.com/61067969/120994756-157ca480-c785-11eb-9e47-00afa70a8208.png)

## Model selection and implementation.
- kNN = [k-Nearest Neighbor](KNeighbors.py/#L6)

## Learning algorithm selection and implementation.
- [neighbours](KNeighbors.py/#L16) [3, 4, 5]

- feature extraction (3 methods mentioned above)

## Prediction for new image using created model.
The best model is chosen according to it's accuracy (knn_model.prediction(...))

# RESULTS

Results of models with different parameters are avaliable in [knn_results.txt](knn_results.txt).
Best ones for each approach:
| accuracy | neighbors | preprocessing |
| --- | --- | --- |
| 0.8675 | 4 | hog, pixels rate = 4x4 |
| 0.8613 | 4 | contrast = 12 |
| 0.8597 | 4 | no preprocessing |

Comparing to resuts form: http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/#
![image](https://user-images.githubusercontent.com/61067969/120996131-57f2b100-c786-11eb-90c5-e92a9c33a53b.png)

HOG only slightly increases accuracy.


# USAGE
## Models creation was seperated into three steps:

### 1. Accessing and preprocessing data - [feature_extraction.py](feature_extraction.py)
  
  Since I decided to use HOG for feature extraction, time for feature extraction increased dramaticly, and I decided to seperate this proces from the others. In this file the matrices containing images are preprocessed and saved in .pkl file in preprocessing directory.
  
### 2. Generating 3 models with best accuracy - [KNeighbors.py](KNeighbors.py)

  By accessing original matrices from original directory and preprocessed matrices from preprocessing directory, I generate models with previously mentioned parameters. The 3 models with best accuracy are then saved in models directory as .sav files.

### 3. Using a single model to predict label of a single peacture at the time - [using_models.py](using_models.py)

  This file is created in order to demonstrate the usage of generated model - how to predict the label of a single picture.
     

## Used libraries:

     ├─> skilearn
     ├─> skimage
     ├─> numpy
     ├─> cv2
     └─> pickle

## Files placement:

     ├─── feature_extraction.py
     ├─── KNeighbors.py
     ├─── using_models.py
     │
     ├─── original
     │    ├─── t10k-images-idx3-ubyte.gz
     │    ├─── t10k-labels-idx1-ubyte.gz
     │    ├─── train-images-idx3-ubyte.gz
     │    └─── train-labels-idx1-ubyte.gz
     ├─── preprocessing
     │    ├─── contrast 12.pkl
     │    ├─── hog 9 2 2.pkl
     │    ├─── ...
     │    └─── ...
     └─── models
          ├─── knn_hog 9 2 2_distance_4.sav
          ├─── knn_hog 9 2 2_distance_5.sav
          ├─── knn_hog 9 2 2_uniform_4.sav
          └─── knn_hog 9 2 2_uniform_5.sav
