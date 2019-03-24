import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.ml.feature import PCA
from pyspark.ml.clustering import KMeans
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import MultilayerPerceptronClassifier
import numpy as np
import csv

# Create the spark object, so that the code can run on the spark.
spark = SparkSession \
    .builder \
    .appName("Python Spark KNN") \
    .getOrCreate()

train_pan_df = pd.read_csv("/home/zzha7690/ML_A2/train_set.csv",header = None)
confusion_matrix_kk = np.zeros(shape=(10, 10))
confusion_matrix_kk = confusion_matrix_kk.astype(int)
cumulate_acc = 0.0

for i in range(0, 10):
    if i == 0:
        train_set = train_pan_df[7300:]
        validation_set = train_pan_df[0:7300]
    elif i == 9:
        train_set = train_pan_df[0:(73257 - 7300)]
        validation_set = train_pan_df[(73257 - 7300):]
    else:
        validation_set = train_pan_df[i * 7300:(i + 1) * 7300]
        train_set1 = train_pan_df[0:i * 7300]
        train_set2 = train_pan_df[(i + 1) * 7300:]
        train_set = train_set1.append(train_set2)
    print(i, "split array finish")

    # transfer to spark.df
    train_list = train_set.values.tolist()
    test_list = validation_set.values.tolist()
    test_df = spark.sparkContext.parallelize(test_list).toDF()
    train_df = spark.sparkContext.parallelize(train_list).toDF()

    assembler_test = VectorAssembler(inputCols=test_df.columns[:1024], outputCol="fectures")
    test_vectors_withlabel = assembler_test.transform(test_df).selectExpr("_c1024 as label_test", "fectures")

    assembler_train = VectorAssembler(inputCols=train_df.columns[:1024], outputCol="fectures")
    train_vectors_withlabel = assembler_train.transform(train_df).selectExpr("_c1024 as label_train", "fectures")

    pca = PCA(k=200, inputCol="fectures", outputCol="pca_vector")
    model_200 = pca.fit(train_vectors_withlabel)
    pca_train_result = model_200.transform(train_vectors_withlabel).selectExpr('label_train as label',
                                                                               'pca_vector as feature')
    pca_test_result = model_200.transform(test_vectors_withlabel).selectExpr('label_test as label',
                                                                             'pca_vector as feature')

    # define parameters
    input_layer = 200  # number of features
    output_layer = 10  # output 0~9
    hidden_1 = 150
    hidden_2 = 150
    layers = [input_layer, hidden_1, hidden_2, output_layer]

    MPC = MultilayerPerceptronClassifier(featuresCol='feature', labelCol='label', predictionCol='prediction',
                                         maxIter=400, layers=layers, blockSize=128, seed=123)

    model = MPC.fit(pca_train_result)

    result = model.transform(pca_test_result).select("label", "prediction")
    result_lp = result.selectExpr("label", "cast (prediction as int) prediction")
    final_result = result_lp.rdd
    count = final_result.count()

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

    accuracy = (count - diff) / count
    cumulate_acc = cumulate_acc + accuracy
    print("The accuracy of NO.", i, "test is", accuracy)

    # Confusion matrix
    confusion_matrix = np.zeros(shape=(10, 10))
    confusion_matrix = confusion_matrix.astype(int)


    def matrix_addvalue(matrix, value):
        matrix[int(value[1])][int(value[0])] = matrix[int(value[1])][int(value[0])] + 1
        return matrix


    combine_matrix = (lambda m1, m2: m1 + m2)

    result_confusion = final_result.aggregate(confusion_matrix, matrix_addvalue, combine_matrix)

    confusion_matrix_kk = confusion_matrix_kk + result_confusion





total_acc = cumulate_acc / 10
print("Ten fold test finish!!!! Average Training Accuracy:", str(total_acc))

with open('/home/zzha7690/ML_A2/Confusion_matrix_10fold_MLP.csv', 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',', quotechar='\n', quoting=csv.QUOTE_MINIMAL)
    spamwriter.writerows(confusion_matrix_kk)


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


statistic_list = statistic(confusion_matrix_kk)
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