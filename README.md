# INTRODUCTION
*Overview of all exercises, reasoning for selected approaches.*

***Aim of the project:*** The exercise is separated into two subexercises:
1. using methods derived for previous exercises for MSiD ([Naive Bayes](zad3.py#L238), K-NN, Logistic Regression)
   
2. using available libraries (in my example mainly [scikit-learn](https://pypi.org/project/scikit-learn/)) 
   
create a model for [Fashion Mnist](https://github.com/zalandoresearch/fashion-mnist) data (matrixes containging matrices of flattened 28x28 pictures of clothing).
Results compare to ones aveliable on [Fashion MNIST Benchmark](http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/#) for each selected model.

My selected approaches:
1. For first exercise I've chosen [Naive Bayes](zad3.py#L238).
2. For second [K-NN](KNeighbors.py/#L6) from library [scikit-learn](https://pypi.org/project/scikit-learn/).

For both excercises I tested three preprocessing approaches:
1. [No preprocessing](feature_extraction.py/#L39) - since for Naive Bayes and kNN it is unnecessary.
2. [enchancing contrast](feature_extraction.py/#L85) - the reasoning was that shapes of the clothing might be more 
   indicating of the label than the shades itself, hence increasing contrast could help with identification.
3. [HOG](feature_extraction.py/#L95) (Histogram of oriented gradients) - similar reason to previous one, only more 
   reliable; however at the same time more time-consuming.

### Accessing data
*Explanation of where and how data is allocated, how it is accessed for following exercises.*

Fashion mnist contains 4 matrices, downloaded from https://github.com/zalandoresearch/fashion-mnist/tree/master/data/fashion:

    ├─── t10k-images-idx3-ubyte.gz
    ├─── t10k-labels-idx1-ubyte.gz
    ├─── train-images-idx3-ubyte.gz
    └─── train-labels-idx1-ubyte.gz

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
For example picture and label for same instance:
```python
index = 7
picture = X_train[index]
label = Y_train[index]
```
In following exercises data is accessed from those 4 matrices in this manor.
# METHODS
*In details how selected methods were implemented.*
## Feature extraction from picture
*How data is accessed in original form, and overview of methods for data preprocessing.*
### [No preprocessing](feature_extraction.py/#L39)
Original matrixes are accessed using [load_nist](feature_extraction.py/#L8) method (details in introduction).\
In result of using the following method we access matrices in their unchanged (original) form.
```python
def load_original(path='original'):
    X_train, y_train = load_mnist(path, kind='train')
    X_test, y_test = load_mnist(path, kind='t10k')
    return X_train, y_train, X_test, y_test
```
Visual representation of single picture in it's original form:

![image](https://user-images.githubusercontent.com/61067969/120994600-ed8d4100-c784-11eb-9b92-e77162947ef7.png)

### [Preprocessing](feature_extraction.py#L66)
*Overview of two chosen preprocessing approaches and their implementations.*
1. [enchancing contrast](feature_extraction.py/#L85) - by using this method we focus more on the overall shape 
   of the clothing.\
   It is less effective than HOG; however less time-consuming.
   
Method implementation using cv2 library.\
Method intakes picture matrix (in 28x28 format) and contrast rate (I tested values from range 9-12
since they were ones with more visible results).
```python
def contrast(mx, contr):
    return cv2.filter2D(mx, -1, np.array([[-1, -1, -1], [-1, contr, -1], [-1, -1, -1]]))
```
Method returns processed picture matrix (in 28x28 format). Visual example below (for the same example as above):

![image](https://user-images.githubusercontent.com/61067969/120994299-a737e200-c784-11eb-961c-3aa0c0ef9767.png)

2. [HOG](feature_extraction.py/#L95) (Histogram of oriented gradients) - method focuses 
   more on contours than on areas in the same color (since it measures gradient orientations of chosen blocks of pixels), hence the method is great for shape recognition.\
   the method is effective, nevertheless quite time-consuming.
   
Method implementation imported from skimage.feature.\
Method intakes picture matrix (in 28x28 format), number of orientations (gradient orientations), pixel and block rates (sizes of analysed blocks for gradients).\
For parameters, I chose 9 orientatons (standard value), and 3 pixel rates (2x2, 4x4, 6x6) - 
since pictures are 28x28 pixels other rates vere nonsensical.
```python
def hog_features(mx, orientations, pixel_cell, cell_block):
    fd, hog_image = hog(mx, orientations=orientations, pixels_per_cell=(pixel_cell, pixel_cell),
                        cells_per_block=(cell_block, cell_block), visualize=True, multichannel=True)
    return hog_image
```
Method returns processed picture matrix (in 28x28 format). Visual example below (for the same example as above):

![image](https://user-images.githubusercontent.com/61067969/120994756-157ca480-c785-11eb-9e47-00afa70a8208.png)

## Model selection and implementation
*For both exercises (using methods derived for previous exercises for MSiD [[Naive Bayes](zad3.py#L238)], 
using available libraries [[K-NN](KNeighbors.py/#L6) from library [scikit-learn](https://pypi.org/project/scikit-learn/)]) 
description of how both models were implemented and derived.*

1. using methods derived for previous exercises for MSiD - [Naive Bayes](zad3.py#L238)
   
The previously derived methods are placed in [zad3.py](zad3.py).\
Model selection method: [model_selection_n](zad3.py#L238). It intakes train matrices along with lists of parameters for a and b.
#### Parameters:
![image](https://user-images.githubusercontent.com/61067969/121890887-209e7a00-cd1b-11eb-846c-bc8904fa6002.png)
- a
- b

Method returns the best error along with the best parameters a and b.

2. using available libraries - [k-Nearest Neighbors](KNeighbors.py/#L6) from library [scikit-learn](https://pypi.org/project/scikit-learn/)

#### Parametrs:
- [neighbours](KNeighbors.py/#L16) - number of neares neighbours (for time-saving purposes 
  I've chosen only values around 4, since in most cases that value is most reliable).
```python
neighbours = [3, 4, 5]
```
- [weights](KNeighbors.py#L17) - it has to types: uniform treats each neighbour equally, 
  second is taking into account distance of each neighbour in probability calculations.
```python
weights = ['uniform', 'distance']
```
Method for model training - it intakes previously mentioned parameters and train matrices to derive a model.\
The test matrices are also given since the method also calculates model score (percentage of correctly chosen labels for all training data).
```python
def train_model(neighs, X_train, y_train, X_test, y_test, weights='uniform'):
    knn_model = KNeighborsClassifier(n_neighbors=neighs, weights=weights)
    knn_model.fit(X_train, y_train)
    score = knn_model.score(X_test, y_test)
    # f1 = f1_score()
    return knn_model, score
```
The methods returns derived model along with it's score (as a float between 0 and 1).

#### Models for other two preprocessing methods
The methodology presented above in this point is implemented also for 
[2 preprocessing methods](KNeighbors.py#L18) mentioned [above](README.md#preprocessing).\
For preprocessing methods *contrast* and *hog_features* we use *[preprocessing](feature_extraction.py#L66)*.\
It intakes picture matrices and preprocessing method and returns matrices of preprocessed pictures.\
For example:
```python
process_fun = lambda mx, o=9, p=2, c=2: hog_features(mx, o, p, c)
X_train_preprocessed, X_test_preprocessed = preprocessing(X_train, X_test, process_fun)
```

## Prediction for new image using created model
The best model is chosen according to it's [accuracy](KNeighbors.py#L50). To get a single picture [prediction](using_models.py#L15):
```python
def get_predict_int(mx, model, preprocess_fun):
    tested = np.reshape(mx, (28, 28, 1))
    tested = preprocess_fun(tested)
    tested = np.reshape(tested, (1, 28 * 28))
    return model.predict(tested)
```

# RESULTS
*Comparison of results of derived methods.*

Results of models with different parameters are avaliable in [knn_results.txt](knn_results.txt).
Best ones for each approach:

| accuracy | neighbors | weight | preprocessing |
| --- | --- | --- | --- |
| [0.8675](knn_results.txt#L2) | 4 | uniform | hog, pixels rate = 4x4 |
| [0.8613](knn_results.txt#L13) | 4 | distance | contrast = 12 |
| [0.8597](knn_results.txt#L16) | 4 | distance | no preprocessing |

Comparing to resuts form: [Fashion MNIST Benchmark](http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/#)

![image](https://user-images.githubusercontent.com/61067969/120996131-57f2b100-c786-11eb-90c5-e92a9c33a53b.png)

HOG only slightly increases accuracy.


# USAGE
*Instruction on program download and setup.*
## Models creation was seperated into three steps:

### 1. Accessing and preprocessing data - [feature_extraction.py](feature_extraction.py)
  
  Since I decided to use HOG for feature extraction, time for feature extraction increased dramaticly, and I decided to seperate this proces from the others. In this file the matrices containing images are preprocessed and saved in .pkl file in preprocessing directory.
  
### 2. Generating 3 models with best accuracy - [KNeighbors.py](KNeighbors.py)

  By accessing original matrices from original directory and preprocessed matrices from preprocessing directory, I generate models with previously mentioned parameters. The 3 models with best accuracy are then saved in models directory as .sav files.

### 3. Using a single model to predict label of a single peacture at the time - [using_models.py](using_models.py)

  This file is created in order to demonstrate the usage of generated model - how to predict the label of a single picture.
     

## Used libraries:
- skilearn
- skimage
- numpy
- cv2
- pickle

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
