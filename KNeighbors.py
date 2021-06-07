from sklearn.neighbors import KNeighborsClassifier
from feature_extraction import *
from sklearn.metrics import f1_score


def train_model(neighs, X_train, y_train, X_test, y_test, weights='uniform'):
    knn_model = KNeighborsClassifier(n_neighbors=neighs, weights=weights)
    knn_model.fit(X_train, y_train)
    score = knn_model.score(X_test, y_test)
    # f1 = f1_score()
    return knn_model, score


if __name__ == '__main__':
    # parameters ______________________________________________________________________________________________________
    neighbours = [3, 4, 5]
    weights = ['uniform', 'distance']
    files = [
        ('contrast 12', (lambda mx, y=12: contrast(mx, y))),
        ('hog 9 2 2', (lambda mx, o=9, p=2, c=2: hog_features(mx, o, p, c))),
        ('hog 9 3 2', lambda mx, o=9, p=3, c=2: hog_features(mx, o, p, c)),
        ('hog 9 4 2', lambda mx, o=9, p=4, c=2: hog_features(mx, o, p, c)),
    ]
    counter = 0
    all_sum = len(neighbours) * (len(files) + 1) * len(weights)
    models = []

    # train models ____________________________________________________________________________________________________
    for weight in weights:
        filename = 'no_preprocessing_' + weight
        X_train_orig, y_train_orig, X_test_orig, y_test_orig = load_original()
        for n in neighbours:
            print(f'{n} neighbours: {filename}...')
            model, accuracy_scores = train_model(n, X_train_orig, y_train_orig, X_test_orig, y_test_orig, weight)
            counter += 1
            print(f'-> accuracy: {accuracy_scores}\t({counter}/{all_sum})')
            models.append((f'{filename}_{n}', model, accuracy_scores))

        for filename, _ in files:
            X_train, y_train, X_test, y_test = load_from_file('preprocessing\\' + filename + '.pkl')
            filename += '_' + weight
            for n in neighbours:
                print(f'{n} neighbours: {filename}...')
                model, accuracy_scores = train_model(n, X_train, y_train, X_test, y_test, weight)
                counter += 1
                print(f'-> accuracy: {accuracy_scores}\t({counter}/{all_sum})')
                models.append((f'{filename}_{n}', model, accuracy_scores))

    # print accuracies ________________________________________________________________________________________________
    models = sorted(models, key=lambda x: x[2], reverse=True)
    print('\naccuracy\ttype')
    for m in models:
        filename, model, accuracy = m
        print(f'{accuracy}\t{filename}')

    # save models _____________________________________________________________________________________________________
    for i in range(4):
        filename, model, _ = models[i]
        pickle.dump(model, open(f'models\\knn_{filename}.sav', 'wb'))
