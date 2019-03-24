# Import all necessary libraries and setup the environment for matplotlib

from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.ml.feature import PCA
from pyspark.ml.clustering import KMeans
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import monotonically_increasing_id
from pyspark.sql import Row
import numpy as np

# Create the spark object, so that the code can run on the spark.
spark = SparkSession \
    .builder \
    .appName("Python Spark KNN") \
    .getOrCreate()

# define the path of data
test_datafile = "hdfs://soit-hdp-pro-1.ucc.usyd.edu.au/share/MNIST/Test-label-28x28.csv"
train_datafile = "hdfs://soit-hdp-pro-1.ucc.usyd.edu.au/share/MNIST/Train-label-28x28.csv"

# read the test_data_set and train_data_set
test_df = spark.read.csv(test_datafile, header=False, inferSchema="true")
train_df = spark.read.csv(train_datafile, header=False, inferSchema="true")

# transfer the test_df in to the dataframe with 2 column label and features so that we can do the further process
assembler_test = VectorAssembler(inputCols=test_df.columns[1:], outputCol="fectures")
test_vectors_withlabel = assembler_test.transform(test_df).selectExpr("_c0 as label_test", "fectures")

# transfer the train_df in to the dataframe with 2 column label and features so that we can do the further process
assembler_train = VectorAssembler(inputCols=train_df.columns[1:], outputCol="fectures")
train_vectors_withlabel = assembler_train.transform(train_df).selectExpr("_c0 as label_train", "fectures")

# fit the pca of train_vector first
# we set the k=50, so that we can keep 90% data of MINST
# After fit process, we can get the model of pca_200.
# Therefore, we can use the model to transform the test and train data.
pca = PCA(k=50, inputCol="fectures", outputCol="pca_vector")
model_200 = pca.fit(train_vectors_withlabel)
pca_train_result = model_200.transform(train_vectors_withlabel).selectExpr('label_train', 'pca_vector as train_vector')
pca_test_result = model_200.transform(test_vectors_withlabel).selectExpr('label_test', 'pca_vector as test_vector')
pca_test_result = pca_test_result.withColumn("id", monotonically_increasing_id())
# pca_test_result.show(2)

test_rdd = pca_test_result.rdd
train_rdd = pca_train_result.rdd
rdd_combine = test_rdd.cartesian(train_rdd)

# print for test
# print(rdd_combine.take(1))
# print("num of rdd format", rdd_combine.count())

# give the k to KNN and set the broadcast of k
k = 5
kbc = spark.sparkContext.broadcast(k)


# compute the distance and return the tuple
def com_distance(a):
    test_id = a[0][2]
    test_label = a[0][0]
    vector1 = a[0][1]
    train_label = a[1][0]
    vector2 = a[1][1]
    distance = vector2.squared_distance(vector1)
    distance = float(distance)
    row = [distance, test_label, train_label]
    return (test_id, row)


dis_rdd = rdd_combine.map(com_distance)
dis_rdd.cache()
# print("dis_rdd.take(5)")
# print(dis_rdd.take(5))
# print("count dis rdd", dis_rdd.count())


# define the aggregate by ke methods
zeroValue = []


def mergeVal(a, b):
    list1 = list(a)
    list1 = list1 + [b]
    if len(list1) < 5:
        return list1
    else:
        list1.sort(reverse=False, key=lambda d: d[0])
        list2 = list1[:5]
        return list2


def mergeComb(a, b):
    list1 = list(a)
    list2 = list(b)
    list3 = list1 + list2
    if len(list3) < 5:
        return list3
    else:
        list3.sort(reverse=False, key=lambda d: d[0])
        list4 = list3[:5]
        return list4


aggresult = dis_rdd.aggregateByKey(zeroValue, mergeVal, mergeComb)


# print("aggresult.take(4)")
# print(aggresult.take(4))
# print("aggresult.count()", aggresult.count())


def predict(a):
    grouped_list = list(a[1])
    test_result_label = grouped_list[1][1]
    grouped_list.sort(reverse=False, key=lambda d: d[0])
    k_list = grouped_list[:5]
    dic_count = {}
    for singel_list in k_list:
        num = dic_count.setdefault(singel_list[2], 0)
        dic_count[singel_list[2]] = num + 1
    sort_dic_list = sorted(dic_count.items(), key=lambda d: d[1], reverse=True)
    pre_label = sort_dic_list[0][0]
    row = [test_result_label, pre_label]
    return row


final_result = aggresult.map(predict)
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


def matrix_addvalue(matrix, value):
    matrix[int(value[1])][int(value[0])] = matrix[int(value[1])][int(value[0])] + 1
    return matrix


combine_matrix = (lambda m1, m2: m1 + m2)

result_confusion = final_result.aggregate(confusion_matrix, matrix_addvalue, combine_matrix)
print(result_confusion)


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