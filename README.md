INTRODUCTION

Aim: By using available libraries (in my example mainly sklearn) create a model for Fashion Mnist (matrixes containging matrices of flattened 28x28 pictures of clothing).

Fashion mnist contains 4 matrices, downloaded from

https://github.com/zalandoresearch/fashion-mnist/tree/master/data/fashion to:

     └─── original
          ├─── t10k-images-idx3-ubyte.gz
          ├─── t10k-labels-idx1-ubyte.gz
          ├─── train-images-idx3-ubyte.gz
          └─── train-labels-idx1-ubyte.gz

Because of github upload limitations in my repository only 3/4 files are added.

Using method load_nist from feature_extraction.py we extract 4 matrices:

     name      size
     X_train   60000x784
     y_train   60000x1
     X_test    10000x784
     y_test    10000x1

X_... matrices contain an image for each row.

     Image is saved in a row as a flattened numpay array (from 2D 28x28 pixels image, pixel values (0-255))

y_... matrices contain number of label from:

     ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']



METHODS

Steps for creating model:

• Feature extraction from picture.

     - no preprocessing (for kNN is unnecessary)
![image](https://user-images.githubusercontent.com/61067969/120994600-ed8d4100-c784-11eb-9b92-e77162947ef7.png)

     - enchancing contrast (using cv2 library); for different variables: 9, 12
![image](https://user-images.githubusercontent.com/61067969/120994299-a737e200-c784-11eb-961c-3aa0c0ef9767.png)

     - HOG = Histogram of oriented gradients for contours analysis (from skimage.feature import hog); 
       for 9 orientations, 2x2, 4x4, 6x6 pixel rates
![image](https://user-images.githubusercontent.com/61067969/120994756-157ca480-c785-11eb-9e47-00afa70a8208.png)

• Model selection and implementation.

     - kNN = k-Nearest Neighbor (KNeighborsClassifier(n_neighbors=neighs, weights=weights))

• Learning algorithm selection and implementation.

     - neighbours [3, 4, 5]
     - feature extraction (3 methods mentioned above)

• Prediction for new image using created model.

     The best model is chosen according to it's accuracy (knn_model.prediction(...))

RESULTS

Results of models with different parameters are avaliable in knn_results.txt.
Best ones for each approach:

     - 0.8675 accuracy, neighbours = 4, hog, pixels rate = 4
     - 0.8613 accuracy, neighbours = 4, contrast = 12
     - 0.8597 accuracy, neighbours = 4, no preprocessing

USAGE
Models creation was seperated into three steps:

     - Accessing and preprocessing data (feature_extraction.py)
          Since I decided to use HOG for feature extraction, time for feature extraction increased dramaticly, and I decided to seperate this proces from the others.
          In this file the matrices containing images are preprocessed and saved in .pkl file in preprocessing directory.
     - Generating 3 models with best accuracy (KNeighbors.py)
          By accessing original matrices from original directory and preprocessed matrices from preprocessing directory, I generate models with previously mentioned parameters.
          The 3 models with best accuracy are then saved in models directory as .sav files.
     - Using a single model to predict label of a single peacture at the time (using_models.py)
          This file is created in order to demonstrate the usage of generated model - how to predict the label of a single picture.
     

Used libraries:

     ├─> skilearn
     ├─> skimage
     ├─> numpy
     ├─> cv2
     └─> pickle

Files placement:

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
