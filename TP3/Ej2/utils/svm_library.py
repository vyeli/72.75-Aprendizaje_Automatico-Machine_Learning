import numpy as np
import time

from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

def svm_train_test(set, rest, test_size, _SVC):
    y_set = np.ones(set.shape[0])
    y_rest = (-1) * np.ones(rest.shape[0])

    X = np.concatenate((set, rest))
    y = np.concatenate((y_set, y_rest))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

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

def run_SVM(svm, writer, sets, set_names, test_size, C):
    run_SVM_with_params(svm, writer, sets, set_names, test_size, C, '', '')
    
def run_SVM_gamma(svm, writer, sets, set_names, test_size, C, gamma):
    run_SVM_with_params(svm, writer, sets, set_names, test_size, C, gamma, '')

def run_SVM_degree(svm, writer, sets, set_names, test_size, C, degree):
    run_SVM_with_params(svm, writer, sets, set_names, test_size, C, '', degree)

def run_SVM_with_params(svm, writer, sets, set_names, test_size, C, gamma, degree):
    start = time.time()
    tp, fp, fn, tn = svm_train_test(sets[0], sets[1], test_size, svm)
    end = time.time()
    writer.writerow([set_names[0], set_names[1], end - start, tp, fp, fn, tn, C, gamma, degree])
    print(f'Finish {set_names[0]} vs {set_names[1]}: {end - start}s')

def SVM_train(svm, sets):
    y_set = np.ones(sets[0].shape[0])
    y_rest = (-1) * np.ones(sets[1].shape[0])

    X = np.concatenate((sets[0], sets[1]))
    y = np.concatenate((y_set, y_rest))

    model = make_pipeline(StandardScaler(), svm)
    model.fit(X, y)
    print('SVM trained')
    return model

def SVM_predict(model, test_set):
    return model.predict(test_set)