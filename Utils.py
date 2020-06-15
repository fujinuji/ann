import glob
import matplotlib.pyplot as plt

import cv2
import numpy as np


def load_data():
    inputs = []
    outputs = []

    width = 50
    height = 50

    #load sad
    for filename in glob.glob('images\sad\*.png'):
        image = cv2.imread(filename)
        image = cv2.resize(image, (width, height)).flatten()
        image = image.astype("float") / 255.0

        inputs.append(image)
        outputs.append(0)

    #load happy
    for filename in glob.glob('images\happy\*.png'):
        image = cv2.imread(filename)
        image = cv2.resize(image, (width, height)).flatten()
        image = image.astype("float") / 255.0

        inputs.append(image)
        outputs.append(1)

    return inputs, outputs, ["sad", "happy"]

def flatten(mat):
    x = []
    for line in mat:
        for el in line:
            x.append(el)
    return x

def evalMultiClass(realLabels, computedLabels, labelNames):
    from sklearn.metrics import confusion_matrix

    confMatrix = confusion_matrix(realLabels, computedLabels)
    acc = sum([confMatrix[i][i] for i in range(len(labelNames))]) / len(realLabels)
    precision = {}
    recall = {}
    for i in range(len(labelNames)):
        precision[labelNames[i]] = confMatrix[i][i] / sum([confMatrix[j][i] for j in range(len(labelNames))])
        recall[labelNames[i]] = confMatrix[i][i] / sum([confMatrix[i][j] for j in range(len(labelNames))])
    return acc, precision, recall, confMatrix

def plotConfusionMatrix(cm, classNames, title):
    from sklearn.metrics import confusion_matrix
    import itertools

    classes = classNames
    plt.figure()
    plt.imshow(cm, interpolation = 'nearest', cmap = 'Blues')
    plt.title('Confusion Matrix ' + title)
    plt.colorbar()
    tick_marks = np.arange(len(classNames))
    plt.xticks(tick_marks, classNames, rotation=45)
    plt.yticks(tick_marks, classNames)

    text_format = 'd'
    thresh = cm.max() / 2.
    for row, column in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(column, row, format(cm[row, column], text_format),
                horizontalalignment = 'center',
                color = 'white' if cm[row, column] > thresh else 'black')

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

    plt.show()

def load_data_sepia():
    inputs = []
    outputs = []

    width = 50
    height = 50

    #load sad
    for filename in glob.glob('images\sepia\*.jpg'):
        image = cv2.imread(filename)
        image = cv2.resize(image, (width, height)).flatten()
        image = image.astype("float") / 255.0

        inputs.append(image)
        outputs.append(0)

    #load happy
    for filename in glob.glob('images\inormal\*.jpg'):
        image = cv2.imread(filename)
        image = cv2.resize(image, (width, height)).flatten()
        image = image.astype("float") / 255.0

        inputs.append(image)
        outputs.append(1)

    return inputs, outputs, ["pic_sepia", "pic_normal"]
