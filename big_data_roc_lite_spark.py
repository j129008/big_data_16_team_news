from os import listdir
from os.path import isfile, join
import pickle
from pprint import pprint

# My modules
# from sklearn.externals.six.moves import zip
from pyspark.mllib.tree import RandomForest, RandomForestModel
from pyspark.mllib.util import MLUtils
from pyspark.mllib.regression import LabeledPoint
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn import cross_validation
# from sklearn.metrics import roc_curve, auc
from pyspark.mllib.evaluation import BinaryClassificationMetrics

import matplotlib
import numpy
matplotlib.use("Agg")
import matplotlib.pyplot as plt
# ----------------------- prepare data -------------- #
# # FIXME change your data path/folder here

train_file = './trainLite' # folder
test_file = './testLite' # folder

pred_fname = './submission_SGDClassifier.csv' # predicitons

# ##########################################

tr_vec = []
te_vec = []
te_ids = []

print('load training data ...')

index = dict()
counter = dict()
for i in range(2, 24):
    index[i] = dict()
    counter[i] = dict()

# # counter
# for line in open(train_file,'r'):
#     fields = line.split(',')
#     fields[-1] = int(fields[-1].strip())
#     tr_ans.append(fields[1])
#     fields[2] = int(fields[2][6:8])
#     for featureID in range(2,len(fields)):
#         featureType = fields[featureID]
#         try:
#             counter[featureID][featureType] += 1
#         except:
#             counter[featureID][featureType] = 1

# train
for line in open(train_file,'r'):
    fields = line.split(',')
    fields[-1] = int(fields[-1].strip())
    fields[2] = int(fields[2][6:8])
    countAll = 0
    data = []
    for featureID in range(2,len(fields)):
        try:
            data.append(index[featureID][fields[featureID]])
        except:
            index[featureID][fields[featureID]] = len(index[featureID])
            data.append(index[featureID][fields[featureID]])
        if featureID == 2 or featureID == 11 or featureID ==12:
            countAll += counter[featureID][fields[featureID]]
    tr_vec.append((data + [countAll],fields[1]))

print('load testing data ...')

# test

for line in open(test_file):
    fields = line.split(',')
    fields[-1] = int(fields[-1].strip())
    te_ids.append(fields[0])
    fields[1] = int(fields[1][6:8])
    countAll = 0
    data = []
    for featureID in range(1,len(fields)):
        try:
            data.append(index[featureID+1][fields[featureID]])
        except:
            index[featureID+1][fields[featureID]] = len(index[featureID+1])
            data.append(index[featureID+1][fields[featureID]])
        if featureID == 1 or featureID == 10 or featureID ==11:
            try:
                countAll += counter[featureID+1][fields[featureID]]
            except:
                pass
    te_vec.append(data+[countAll])



print('split data for cross validation...')
(trainingData, testData) = tr_vec.randomSplit([0.7, 0.3])

print('build classifier ...')
clf = RandomForest.trainClassifier(trainingData, numClasses=2, categoricalFeaturesInfo={},
                                     numTrees=3, featureSubsetStrategy="auto",
                                     impurity='gini', maxDepth=4, maxBins=32)

# Evaluate model on test instances and compute test error
predictionAndLabels = testData.map(lambda lp: (float(clf.predict(lp.features)), lp.label))
# Instantiate metrics object
metrics = BinaryClassificationMetrics(predictionAndLabels)

# Area under precision-recall curve
print("Area under PR = %s" % metrics.areaUnderPR)
roc_auc = metrics.areaUnderROC
# Area under ROC curve
print("Area under ROC = %s" % roc_auc)

result = clf.predict(te_vec)

