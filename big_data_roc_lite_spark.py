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
model = RandomForest.trainClassifier(trainingData, numClasses=2, categoricalFeaturesInfo={},
        numTrees=3, featureSubsetStrategy="auto",
        impurity='gini', maxDepth=4, maxBins=32)

print('start predict ...')
predictions = model.predict(testData.map(lambda x: x.features))
labelsAndPredictions = testData.map(lambda lp: lp.label).zip(predictions)
testErr = labelsAndPredictions.filter(lambda (v, p): v != p).count() / float(testData.count())
print('Test Error = ' + str(testErr))
print('Learned classification forest model:')
# print(model.toDebugString())
print(labelsAndPredictions)
# Instantiate metrics object
metrics = BinaryClassificationMetrics(labelsAndPredictions)

# Area under precision-recall curve
print("Area under PR = %s" % metrics.areaUnderPR)

# Area under ROC curve
print("Area under ROC = %s" % metrics.areaUnderROC)
