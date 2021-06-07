INTRODUCTION
Aim: By using available libraries (in my example sklearn) create a model for Fashion Mnist (matrixes containging matrices of flattened 28x28 pictures of clothing).

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
• X_train   60000x784
• y_train   60000x1
• X_test    10000x784
• y_test    10000x1

X_... matrices contain an image for each row.
Image is saved in a row as a flattened numpay array (from 2D 28x28 pixels image, pixel values (0-255))

y_... matrices contain number of label from:
['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


METHODS
Steps for creating model:
• Feature extraction from picture.
├─> no preprocessing (for kNN is unnecessary)
└─> HOG = Histogram of oriented gradients for contours analysis (from skimage.feature import hog)

• Model selection and implementation.
└─> kNN = k-Nearest Neighbor (KNeighborsClassifier(n_neighbors=neighs, weights=weights))

• Learning algorithm selection and implementation.
└─> for kNN for [3, 4, 5] neighbours (we choose one with higher score)

• Prediction for new image using created model.
└─> for kNN = knn_model.prediction(...)

Tutaj jest należy opisać metody jakie zostały zastosowane celem osiągnięcia
zamierzonych efektów. Warto zamieścić tutaj odnośniki do metod z których czerpali
Państwo inspiracje, oraz ilustracje pozwalające na zrozumienie Państwa podejścia.


RESULTS
Results of models with different parameters are avaliable in knn_results.txt.

W tym przypadku należy opisać wyniki, które otrzymali Państwo dla opisanych
wyżej metod. Wyniki warto przedstawić w tabelce, zestawiając z wynikami otrzymanymi
dla rozwiązań referencyjnych (tymi opisanymi w sekcji Benchmark).


USAGE
Models creation was seperated into three steps:
- Accessing and preprocessing data (feature_extraction.py)
- Generating 3 models with best accuracy (KNeighbors.py)
- Using a single model to predict label of a single peacture at the time (using_models.py)

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


W tej sekcji należy opisać w jaki sposób uruchomić przygotowany przez Pań-
stwa projekt. W opisie należy uwzględnić w jaki sposób należy umieścić dane uczące i
testowe (a może pobieranie danych odbywa się automatycznie?), jakie biblioteki należy
zainstalować, w jaki sposób uruchomić program, aby otrzymać wyniki z sekcji Results,
czy można gdzieś pobrać wyuczone modele i w jaki sposób ich użyć ?
