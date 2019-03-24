import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import csv
import math


# def show_confusion_matrix(confusion, xlabels, ylabels):
#     plt.figure(figsize=(14, 11))
#     df_cm = pd.DataFrame(confusion, range(10), range(10))
#     df_cm.astype(int)
#
#     sn.heatmap(df_cm, annot=True, xticklabels=xlabels, yticklabels=ylabels, vmin=0, vmax=8000, cmap=sn.cm.rocket_r,
#                fmt='.5g')
#     plt.show()
#     print('Fig. 3 Confusion Matrix of the test-data(X_axis is the predicted labels & Y_axis is the actual labels)')
#     return

matrix = []
with open('/Users/zekunzhang/Desktop/ML_A2/matrix/Confusion_MLP.csv') as training_label0:
    spamreader_label = csv.reader(training_label0, quotechar=',')
    for row in spamreader_label:
        arr=[]
        for k in row:
            arr.append(int(math.floor(float(k))))
        matrix.append(arr)



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


statistic_list = statistic(matrix)

