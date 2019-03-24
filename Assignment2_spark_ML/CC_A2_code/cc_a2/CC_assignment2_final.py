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
import csv
import argparse


if __name__ == "__main__":
    spark = SparkSession \
        .builder \
        .appName("Python Spark KNN 10-5k") \
        .getOrCreate()
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", help="The dimension you want to retain after PCA.", default=10)
    parser.add_argument("--k", help="The k for KNN.", default=5)
    args = parser.parse_args()
    dimk = int(args.dim)
    knnk = int(args.k)

    # define the path of data
    test_datafile = "hdfs://soit-hdp-pro-1.ucc.usyd.edu.au/share/MNIST/Test-label-28x28.csv"
    train_datafile = "hdfs://soit-hdp-pro-1.ucc.usyd.edu.au/share/MNIST/Train-label-28x28.csv"

    # read the test_data_set and train_data_set
    test_df = spark.read.csv(test_datafile, header=False, inferSchema="true")
    train_df = spark.read.csv(train_datafile, header=False, inferSchema="true")
    # train_df.cache()
    # test_df.cache()

    # transfer the test_df in to the dataframe with 2 column label and features so that we can do the further process
    assembler_test = VectorAssembler(inputCols=test_df.columns[1:], outputCol="fectures")
    test_vectors_withlabel = assembler_test.transform(test_df).selectExpr("_c0 as label_test", "fectures")

    # transfer the train_df in to the dataframe with 2 column label and features so that we can do the further process
    assembler_train = VectorAssembler(inputCols=train_df.columns[1:], outputCol="fectures")
    train_vectors_withlabel = assembler_train.transform(train_df).selectExpr("_c0 as label_train", "fectures")

    pca = PCA(k=dimk, inputCol="fectures", outputCol="pca_vector")
    model_200 = pca.fit(train_vectors_withlabel)

    pca_test_result = model_200.transform(test_vectors_withlabel).selectExpr('label_test', 'pca_vector as test_vector')
    pca_test_result = pca_test_result.withColumn("id", monotonically_increasing_id())

    pca_train_result = model_200.transform(train_vectors_withlabel).selectExpr('label_train',
                                                                               'pca_vector as train_vector')

    test_rdd = pca_test_result.rdd
    train_rdd = pca_train_result.rdd
    rdd_combine = test_rdd.cartesian(train_rdd)

    kbc = spark.sparkContext.broadcast(knnk)


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

    zeroValue = []


    def mergeVal(a, b):
        k1 = kbc.value
        list1 = list(a)
        list1 = list1 + [b]
        if len(list1) < k1:
            return list1
        else:
            list1.sort(reverse=False, key=lambda d: d[0])
            list2 = list1[:k1]
            return list2


    def mergeComb(a, b):
        k1 = kbc.value
        list1 = list(a)
        list2 = list(b)
        list3 = list1 + list2
        if len(list3) < k1:
            return list3
        else:
            list3.sort(reverse=False, key=lambda d: d[0])
            list4 = list3[:k1]
            return list4


    aggresult = dis_rdd.aggregateByKey(zeroValue, mergeVal, mergeComb)


    # print("aggresult.take(4)")
    # print(aggresult.take(4))
    # print("aggresult.count()", aggresult.count())

    def predict(a):
        k1 = kbc.value
        _id = a[0]
        grouped_list = list(a[1])
        test_result_label = grouped_list[1][1]
        grouped_list.sort(reverse=False, key=lambda d: d[0])
        k_list = grouped_list[:k1]
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

    schema2 = StructType([
    StructField("Correct label", IntegerType(), True),
    StructField("Predicted label", IntegerType(), True),
    ])

    result_df = spark.createDataFrame(final_result, schema2)

    result_df.toPandas().to_csv('/home/zzha7690/CC_A2/result.csv')

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
    with open('/home/zzha7690/CC_A2/Confusion_matrix.csv', 'w', newline='') as csvfile:
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



