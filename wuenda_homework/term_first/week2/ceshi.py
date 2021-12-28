'''
调用sklearn库里的逻辑回归api来实现识别猫图片
'''
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import h5py

def load_dataset():
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])  # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])  # your train set labels

    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])  # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # your test set labels

    classes = np.array(test_dataset["list_classes"][:])  # the list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

if __name__ == '__main__':
    train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

    train_set_y=np.squeeze(train_set_y)
    test_set_y = np.squeeze(test_set_y)

    train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1)
    test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1)

    train_set_x = train_set_x_flatten / 255
    test_set_x = test_set_x_flatten / 255

    LR = LogisticRegression(C=1000.0, random_state=0,max_iter=100000)
    LR.fit(train_set_x, train_set_y)

    acc = LR.score(test_set_x, test_set_y)
    print(acc)
    print(test_set_y)
    prepro = LR.predict(test_set_x)
    print(prepro)


