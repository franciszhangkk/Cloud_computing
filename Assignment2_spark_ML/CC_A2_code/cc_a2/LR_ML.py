from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.ml.feature import PCA
from pyspark.ml.clustering import KMeans
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import monotonically_increasing_id
from pyspark.sql import Row
from pyspark.ml.classification import LogisticRegression, OneVsRest
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import MultilayerPerceptronClassifier
import numpy as np
import csv

# Create the spark object, so that the code can run on the spark.
spark = SparkSession \
    .builder \
    .appName("Python Spark KNN l3n200") \
    .getOrCreate()

# define the path of data
# test_datafile = "hdfs://soit-hdp-pro-1.ucc.usyd.edu.au/share/MNIST/Test-label-28x28.csv"
# train_datafile= "hdfs://soit-hdp-pro-1.ucc.usyd.edu.au/share/MNIST/Train-label-28x28.csv"

# local path
test_datafile = "hdfs://soit-hdp-pro-1.ucc.usyd.edu.au/user/zzha7690/test_set.csv"
train_datafile = "hdfs://soit-hdp-pro-1.ucc.usyd.edu.au/user/zzha7690/train_set.csv"

# read the test_data_set and train_data_set
test_df = spark.read.csv(test_datafile, header=False, inferSchema="true")
train_df = spark.read.csv(train_datafile, header=False, inferSchema="true")

# transfer the test_df in to the dataframe with 2 column label and features so that we can do the further process
assembler_test = VectorAssembler(inputCols=test_df.columns[:1024], outputCol="fectures")
test_vectors_withlabel = assembler_test.transform(test_df).selectExpr("_c1024 as label_test", "fectures")

# transfer the train_df in to the dataframe with 2 column label and features so that we can do the further process
assembler_train = VectorAssembler(inputCols=train_df.columns[:1024], outputCol="fectures")
train_vectors_withlabel = assembler_train.transform(train_df).selectExpr("_c1024 as label_train", "fectures")

# fit the pca of train_vector first
# we set the k=50, so that we can keep 90% data of MINST
# After fit process, we can get the model of pca_200.
# Therefore, we can use the model to transform the test and train data.
pca = PCA(k=200, inputCol="fectures", outputCol="pca_vector")
model_200 = pca.fit(train_vectors_withlabel)
pca_train_result = model_200.transform(train_vectors_withlabel).selectExpr('label_train as label',
                                                                           'pca_vector as features')
pca_test_result = model_200.transform(test_vectors_withlabel).selectExpr('label_test as label', 'pca_vector as features')

lr = LogisticRegression(maxIter=200, tol=1E-6, fitIntercept=True)
ovr2 = OneVsRest(classifier=lr)
model2 = ovr2.fit(pca_train_result)
result = model2.transform(pca_test_result)
result_lp = result.selectExpr("label", "cast (prediction as int) prediction")
final_result = result_lp.rdd

# calculate the accuracy

neutral_zero_value = 0


def seqOp(a, b):
    if b[0] == b[1]:
        return a
    else:
        return a + 1


combOp = (lambda a, b: a + b)

diff = final_result.aggregate(neutral_zero_value, seqOp, combOp)
diff = float(diff)

accuracy = (26032 - diff) / 26032
print("The accuracy is", accuracy)

# Confusion matrix
confusion_matrix = np.zeros(shape=(10, 10))
confusion_matrix = confusion_matrix.astype(int)


def matrix_addvalue(matrix, value):
    matrix[int(value[1])][int(value[0])] = matrix[int(value[1])][int(value[0])] + 1
    return matrix


combine_matrix = (lambda m1, m2: m1 + m2)

result_confusion = final_result.aggregate(confusion_matrix, matrix_addvalue, combine_matrix)
with open('/home/zzha7690/ML_A2/Confusion_matrix_lr.csv', 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',', quotechar='\n', quoting=csv.QUOTE_MINIMAL)
    spamwriter.writerows(result_confusion)


# Statistic table

def statistic(confusion_test):
    re_list = []
    label = np.arange(0, 10)
    for i in label:
        TP = confusion_test[i, i]
        FN = np.sum(confusion_test[i]) - TP
        FP = np.sum(confusion_test[:, i]) - TP
        TN = np.sum(confusion_test) - TP - FN - FP
        precision = (TP / (TP + FP))
        recall = TP / (TP + FN)
        F_measure = TP / (2 * TP + FP + FN)
        Support = (TP + FN)
        row = [int(label[i]), round(float(precision), 3), round(float(recall), 3), round(float(F_measure), 3),
               round(float(Support), 0)]
        re_list.append(row)
    return re_list


statistic_list = statistic(result_confusion)
statistic_rdd = spark.sparkContext.parallelize(statistic_list)
schema = StructType([
    StructField("Label", IntegerType(), True),
    StructField("Precision", FloatType(), True),
    StructField("recall", FloatType(), True),
    StructField("F1-score", FloatType(), True),
    StructField("Support", FloatType(), True)
])
statistic_df = spark.createDataFrame(statistic_rdd, schema)
statistic_df.show()
