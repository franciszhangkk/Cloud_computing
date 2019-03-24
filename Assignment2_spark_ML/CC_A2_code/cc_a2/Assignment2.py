# Import all necessary libraries and setup ≥˘the environment for matplotlib
# %matplotlib inline

from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.ml.feature import PCA
from pyspark.ml.clustering import KMeans
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import monotonically_increasing_id
from pyspark.sql import Row
import numpy as np
import csv
import argparse

# Create the spark object, so that the code can run on the spark.
spark = SparkSession \
    .builder \
    .appName("Python Spark KNN") \
    .getOrCreate()

# define the path of data

test_datafile = "hdfs://soit-hdp-pro-1.ucc.usyd.edu.au/share/MNIST/Test-label-28x28.csv"
train_datafile= "hdfs://soit-hdp-pro-1.ucc.usyd.edu.au/share/MNIST/Train-label-28x28.csv"
# read the test_data_set and train_data_set
test_df = spark.read.csv(test_datafile,header=False,inferSchema="true")
train_df = spark.read.csv(train_datafile,header=False,inferSchema="true")

# transfer the test_df in to the dataframe with 2 column label and features so that we can do the further process
assembler_test = VectorAssembler(inputCols = test_df.columns[1:],outputCol="features")
test_vectors_withlabel = assembler_test.transform(test_df).selectExpr("_c0 as label","features")

# transfer the train_df in to the dataframe with 2 column label and features so that we can do the further process
assembler_train = VectorAssembler(inputCols = train_df.columns[1:],outputCol="features")
train_vectors_withlabel = assembler_train.transform(train_df).selectExpr("_c0 as label","features")

# fit the pca of train_vector first
# we set the k=200, so that we can keep 90% data of MINST
# After fit process, we can get the model of pca_200. 
# Therefore, we can use the model to transform the test and train data.
pca = PCA(k=10, inputCol="features", outputCol="pca200")
model_200 = pca.fit(train_vectors_withlabel)
pca_train_result = model_200.transform(train_vectors_withlabel).select('label','pca200')
pca_test_result = model_200.transform(test_vectors_withlabel).select('label','pca200')


# transfer the dataframe into rdd
test_rdd = pca_test_result.rdd
train_rdd = pca_train_result.rdd


# create the broadcast, so that every singe cluster can use it
trainbc = spark.sparkContext.broadcast(train_rdd.collect())
# give the k to KNN and set the broadcast of k
k=5
kbc = spark.sparkContext.broadcast(k)


# write the algorithm of KNN
# a=rdd 的某一行
def knn (a):
    # set an empty list to store the result
    result_lists =[]
    # extract the test vector and correct label
    vector_test = a[1]
    result_label = a[0]
    # get the list of training data from broadcast
    train_list = trainbc.value
    # get k
    k = kbc.value
    # compute the distance between test vector and train vector
    for train_data in train_list:
        vector_train = train_data[1]
        label_train = train_data[0]
        dis = vector_test.squared_distance(vector_train)
        # put the label and distance into the result list
        result_list = [label_train,dis]
        result_lists.append(result_list)
    # sort the list according to the distance
    result_lists.sort(reverse = False, key = lambda d:d[1])
    # only use first k data
    k_list = result_lists[:k]
    # create a dictionary to count
    dic_count = {}
    for singel_list in k_list:
        num = dic_count.setdefault(singel_list[0],0)
        dic_count[singel_list[0]] = num + 1
    #sort the dictionary and get the first one
    sort_dic_list = sorted(dic_count.items(), key=lambda d:d[1], reverse = True)
    pre_label = sort_dic_list[0][0]
    row = [result_label, pre_label]
    # return a row object, so that it is easy to transfer the rdd to dataframe
    return row


final_result = test_rdd.map(knn)

print("final result count", final_result.count())
final_result.cache()

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

accuracy = (10000 - diff) / 10000
print("The accuracy is", accuracy)

# Confusion matrix
confusion_matrix = np.zeros(shape=(10, 10))
confusion_matrix = confusion_matrix.astype(int)


def matrix_addvalue(matrix, value):
    matrix[int(value[1])][int(value[0])] = matrix[int(value[1])][int(value[0])] + 1
    return matrix


combine_matrix = (lambda m1, m2: m1 + m2)

result_confusion = final_result.aggregate(confusion_matrix, matrix_addvalue, combine_matrix)
with open('/home/zzha7690/Cloud_computing/Confusion_matrix.csv', 'w', newline='') as csvfile:
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





