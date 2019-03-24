#!/bin/bash

spark-submit --master yarn --deploy-mode client --executor-memory 2G --conf spark.default.parallelism=50  --num-executors 10 --executor-cores 4 /home/zzha7690/Cloud_computing/assignment2_aggknn.py


spark-submit --master yarn --deploy-mode client --executor-memory 2G  --num-executors 10 --executor-cores 4 /home/zzha7690/ML_A2/MultiLayer_ML.py

spark-submit --master yarn --deploy-mode client --executor-memory 4G --num-executors 5 --executor-cores 4 /home/zzha7690/ML_A2/Assignment2-KNN.py 300 5
