import os

from feature_extraction import *


def load_model(filename):
    return pickle.load(open(filename, 'rb'))


def display_from_tab(tested, label):
    tested = np.reshape(tested, (28, 28, 1))
    display(tested, label)


def get_predict_int(mx, model, preprocess_fun):
    tested = np.reshape(mx, (28, 28, 1))
    tested = preprocess_fun(tested)
    tested = np.reshape(tested, (1, 28 * 28))
    return model.predict(tested)


if __name__ == '__main__':
    labels_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress',
                    'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    files = [
        ('contrast 12', (lambda mx, y=12: contrast(mx, y))),
        ('hog 9 2 2', (lambda mx, o=9, p=2, c=2: hog_features(mx, o, p, c))),
        ('hog 9 3 2', lambda mx, o=9, p=3, c=2: hog_features(mx, o, p, c)),
        ('hog 9 4 2', lambda mx, o=9, p=4, c=2: hog_features(mx, o, p, c)),
    ]
    filename, func = files[1]
    filename = 'models\\knn_' + filename + '_uniform_4.sav'
    model = load_model(filename)
    X_train, y_train, X_test, y_test = load_original()

    for i in range(25):
        mx = X_test[i]

        real_int = int(y_test[i])
        pred_int = int(get_predict_int(mx, model, func))

        real_label = labels_names[real_int]
        pred_label = labels_names[pred_int]

        is_correct = pred_label == real_label
        display_from_tab(mx, f'({int(is_correct)}) PREDICTION: {pred_label}; REAL: {real_label}')
