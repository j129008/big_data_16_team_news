from pyspark import SparkContext
from pyspark.mllib.tree import RandomForest, RandomForestModel
from pyspark.mllib.util import MLUtils
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.evaluation import BinaryClassificationMetrics

sc = SparkContext("local[4]", "big data")

print('load training data ...')
data = MLUtils.loadLibSVMFile( sc, '/home/bigdata/hadoop-2.7.2/data/trainOut' )
(trainingData, testData) = data.randomSplit([0.7, 0.3])

print('build classifier ...')
clf = RandomForest.trainClassifier(trainingData, numClasses=2, categoricalFeaturesInfo={},
        numTrees=3, featureSubsetStrategy="auto",
        impurity='gini', maxDepth=4, maxBins=32)

# Evaluate model on test instances and compute test error
predictionAndLabels = testData.map(lambda lp: (float(clf.predict(lp.features)), lp.label))
"""
# Instantiate metrics object
metrics = BinaryClassificationMetrics(predictionAndLabels)

# Area under precision-recall curve
print("Area under PR = %s" % metrics.areaUnderPR)
roc_auc = metrics.areaUnderROC
# Area under ROC curve
print("Area under ROC = %s" % roc_auc)
"""
