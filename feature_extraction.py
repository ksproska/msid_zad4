import pickle
import cv2
import numpy as np
from skimage.feature import hog


# from github
def load_mnist(path, kind='train'):
    import os
    import gzip
    import numpy as np

    """Load MNIST data from `path`"""
    labels_path = os.path.join(path, '%s-labels-idx1-ubyte.gz' % kind)
    images_path = os.path.join(path, '%s-images-idx3-ubyte.gz' % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(len(labels), 784)

    return images, labels


# methods for matrix pickle
def save_matrixes(filename, mx1, mx2):
    with open(filename, 'wb') as f:
        pickle.dump((mx1, mx2), f)


def load_matrixes(filename):
    with open(filename, 'rb') as f:
        mxs = pickle.load(f)
        return mxs


# loading matrices
def load_original(path='original'):
    X_train, y_train = load_mnist(path, kind='train')
    X_test, y_test = load_mnist(path, kind='t10k')
    return X_train, y_train, X_test, y_test


def load_from_file(filename, path='original'):
    _, y_train, _, y_test = load_original(path)
    X_train, X_test = load_matrixes(filename)
    return X_train, y_train, X_test, y_test


# reshaping
def shape_squares(Xtrain, Xtest):
    size_xy = 28
    im_shape = (size_xy, size_xy, 1)
    X_train_inner = Xtrain.reshape(Xtrain.shape[0], *im_shape)
    X_test_inner = Xtest.reshape(Xtest.shape[0], *im_shape)
    return X_train_inner, X_test_inner


def flatten_squares(Xtrain, Xtest):
    X_train_inner = Xtrain.reshape(Xtrain.shape[0], Xtrain.shape[1] * Xtrain.shape[2])
    X_test_inner = Xtest.reshape(Xtest.shape[0], Xtest.shape[1] * Xtest.shape[2])
    return X_train_inner, X_test_inner


def preprocessing(X_train, X_test, process_fun):
    X_train_sqr, X_test_sqr = shape_squares(X_train, X_test)
    X_train_temp = []
    X_test_temp = []

    for r in X_train_sqr:
        X_train_temp.append(process_fun(r))

    for r in X_test_sqr:
        X_test_temp.append(process_fun(r))

    X_train_sqr = np.array(X_train_temp)
    X_test_sqr = np.array(X_test_temp)

    X_train_inner, X_test_inner = flatten_squares(X_train_sqr, X_test_sqr)
    return X_train_inner, X_test_inner


# processing methods
def contrast(mx, contr):
    return cv2.filter2D(mx, -1, np.array([[-1, -1, -1], [-1, contr, -1], [-1, -1, -1]]))


def contour(mx):
    contours, hierarchy = cv2.findContours(mx, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    img_contours = np.zeros(mx.shape)
    return cv2.drawContours(img_contours, contours, -1, 255)


def hog_features(mx, orientations, pixel_cell, cell_block):
    fd, hog_image = hog(mx, orientations=orientations, pixels_per_cell=(pixel_cell, pixel_cell),
                        cells_per_block=(cell_block, cell_block), visualize=True, multichannel=True)
    return hog_image


def blur(mx):
    return cv2.blur(mx, (1, 1))


def gabor_filter(mx, ksize=2, sigma=5, theta=180*(np.pi/180), lamda=1, gamma=1, phi=0):
    kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, phi, ktype=cv2.CV_32F)
    return cv2.filter2D(mx, cv2.CV_8UC3, kernel)


# display
def display(mx, label='', size_x=450, size_y=450):
    cv2.imshow(label, cv2.resize(mx, (size_x, size_y)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    X_train_orig, y_train_orig, X_test_orig, y_test_orig = load_original()
    files = [
        ('contrast 12', (lambda mx, y=12: contrast(mx, y))),
        ('hog 9 2 2', (lambda mx, o=9, p=2, c=2: hog_features(mx, o, p, c))),
        ('hog 9 3 2', lambda mx, o=9, p=3, c=2: hog_features(mx, o, p, c)),
        ('hog 9 4 2', lambda mx, o=9, p=4, c=2: hog_features(mx, o, p, c)),
    ]

    for filename, process_fun in files:
        print(f'Preprocessing {filename}...')
        X_train, X_test = preprocessing(X_train_orig, X_test_orig, process_fun)
        print(f'Preprocessed {filename}')
        save_matrixes('preprocessing\\' + filename + '.pkl', X_train, X_test)
        print(f'{filename} saved.')
