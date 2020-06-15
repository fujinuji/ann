import numpy as np
from sklearn import neural_network

from Utils import  evalMultiClass, plotConfusionMatrix, load_data_sepia

np.random.seed(5)
inputs, outputs, labels = load_data_sepia()

indexes = [i for i in range(len(inputs))]
trainSampleIndexes = np.random.choice(indexes, int(0.8 * len(inputs)), replace=False)
testSampleIndexes = [i for i in indexes if i not in trainSampleIndexes]

trainInputs = [(inputs[i]) for i in trainSampleIndexes]
trainOutputs = [outputs[i] for i in trainSampleIndexes]

testInputs = [inputs[i] for i in testSampleIndexes]
testOutputs = [outputs[i] for i in testSampleIndexes]


classifier = neural_network.MLPClassifier(hidden_layer_sizes=(10, ), activation='relu', max_iter=100, solver='sgd', verbose=10, random_state=1, learning_rate_init=0.005)
classifier.fit(trainInputs, trainOutputs)

predictedLabels = classifier.predict(testInputs)
acc, prec, recall, cm = evalMultiClass(np.array(testOutputs), predictedLabels, labels)

print(np.random.rand(2,3))
print("Accyracy: ", acc)
print("Precision: ", prec)
plotConfusionMatrix(cm, labels, "Seia")
