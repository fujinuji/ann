import numpy as np
from sklearn import neural_network
from sklearn.preprocessing import OneHotEncoder

from MyANN import MyANN
from Utils import evalMultiClass, plotConfusionMatrix, load_data_sepia

np.random.seed(5)
def pre_process_data(train_x, train_y, test_x, test_y):
    # Normalize
    #rain_x = train_x / 255.
    #test_x = test_x / 255.

    enc = OneHotEncoder(sparse=False, categories='auto')
    train_y = enc.fit_transform(train_y.reshape(len(train_y), -1))
    test_y = enc.transform(test_y.reshape(len(test_y), -1))

    return train_x, train_y, test_x, test_y

def flatten(mat):
    x = []
    for line in mat:
        for el in line:
            x.append(el)
    return x


def loadDigitData():
    from sklearn.datasets import load_digits

    data = load_digits()
    inputs = data.images
    outputs = data['target']
    outputNames = data['target_names']

    # shuffle the original data
    noData = len(inputs)
    permutation = np.random.permutation(noData)
    inputs = inputs[permutation]
    outputs = outputs[permutation]

    return inputs, outputs, outputNames


def splitData(inputs, outputs):
    np.random.seed(5)
    indexes = [i for i in range(len(inputs))]
    trainSample = np.random.choice(indexes, int(0.8 * len(inputs)), replace=False)
    testSample = [i for i in indexes if not i in trainSample]

    trainInputs = [inputs[i] for i in trainSample]
    trainOutputs = [outputs[i] for i in trainSample]
    testInputs = [inputs[i] for i in testSample]
    testOutputs = [outputs[i] for i in testSample]

    return trainInputs, trainOutputs, testInputs, testOutputs

inputs, outputs, outputNames = loadDigitData()
trainInputs, train_y, testInputs, test_y = splitData(inputs, outputs)

train_x = [flatten(el) for el in trainInputs]
test_x = [flatten(el) for el in testInputs]

train_x = np.array(train_x)
train_y = np.array(train_y)
test_x = np.array(test_x)
test_y = np.array(test_y)

train_x, train_y, test_x, test_y = pre_process_data(train_x, train_y, test_x, test_y)


myann = MyANN(len(train_x[0]), [10], 10)
myann.train(train_x, train_y, 200, 0.2)

for inp in testInputs:
    print(myann.forward_propagate(inp))

predicted = [myann.choose(myann.forward_propagate(inp)) for inp in testInputs]
print(predicted)
print(testOutputs)
