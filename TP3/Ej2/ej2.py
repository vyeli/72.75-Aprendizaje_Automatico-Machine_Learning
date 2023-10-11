import numpy as np
import cv2
import csv
import time
import threading

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

def svm_train_test(set, rest, test_size, _SVC):
    y_set = np.ones(set.shape[0])
    y_rest = (-1) * np.ones(rest.shape[0])

    X = np.concatenate((set, rest))
    y = np.concatenate((y_set, y_rest))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    match kernel:
        case 'linear':
            model = make_pipeline(StandardScaler(), _SVC)
        case 'rbf':
            model = make_pipeline(StandardScaler(), _SVC)
        case 'poly':
            model = make_pipeline(StandardScaler(), _SVC)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return confusion_matrix(y_test, y_pred).flatten()

def get_SVM(kernel, C=1000, degree=10, gamma='auto'):
    match kernel:
        case 'rbf':
            return SVC(C=C, kernel=kernel, gamma=gamma)
        case 'poly':
            return SVC(C=C, kernel=kernel, degree=degree, gamma=gamma)
        case _:     # Default to linear
            return SVC(C=C, kernel=kernel)

def run_SVM(svm, writer, sets, set_names, test_size):
    start = time.time()
    tp, fp, fn, tn = svm_train_test(sets[0], sets[1], test_size, svm)
    end = time.time()
    writer.writerow([set_names[0], set_names[1], end - start, tp, fp, fn, tn])
    print(f'Finish {set_names[0]} vs {set_names[1]} {i}: {end - start}s')

import os

# Create output folder
if not os.path.exists('output'):
    os.makedirs('output')

csv_output = 'output/mixed_kernel_.csv'
images_folder = "imagenes/"

# imread returns RGB colors
cow_image = cv2.imread(images_folder + "vaca.jpg").reshape(-1, 3)
sky_image = cv2.imread(images_folder + "cielo.jpg").reshape(-1, 3)
grass_image = cv2.imread(images_folder + "pasto.jpg").reshape(-1, 3)

with open(csv_output, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

    writer.writerow(['set_1', 'set_2', 'time', 'tp', 'fp', 'fn', 'tn'])

    # Choose parameters
    test_size = 0.2
    kernel = 'rbf'
    C_linear = 5
    C_radial = 20
    runs = 10

    # Sweet spots
    # Linear: C=5
    # RBF (cow vs grass): C=20
    for i in range(runs):
        _svm = get_SVM('linear', C_linear)
        run_SVM(_svm, writer, [sky_image, cow_image], ['sky', 'cow'], test_size)

        _svm = get_SVM('linear', C_linear)
        run_SVM(_svm, writer, [sky_image, grass_image], ['sky', 'grass'], test_size)

        _svm = get_SVM('rbf', C_radial)
        run_SVM(_svm, writer, [grass_image, cow_image], ['grass', 'cow'], test_size)