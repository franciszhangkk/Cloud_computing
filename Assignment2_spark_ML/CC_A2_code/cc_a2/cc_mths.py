# compute the distance and return the tuple
def com_distance(a):
    test_id = a[2]
    test_label = a[0]
    vector1 = a[1]
    train_label = a[3]
    vector2 = a[4]
    distance = vector2.squared_distance(vector1)
    distance = float(distance)
    row = [distance, test_label, train_label]
    return (test_id, row)

def mergeVal (a,b):
    list1 = list(a)
    list1 = list1 + [b]
    if len(list1)<5:
        return list1
    else:
        list1.sort(reverse=False, key=lambda d: d[0])
        list2 = list1[:5]
        return list2

def predict(a):
    grouped_list = list(a[1])
    test_result_label = grouped_list[1][1]
    grouped_list.sort(reverse = False, key = lambda d:d[0])
    k_list = grouped_list[:5]
    dic_count = {}
    for singel_list in k_list:
        num = dic_count.setdefault(singel_list[2],0)
        dic_count[singel_list[2]] = num + 1
    sort_dic_list = sorted(dic_count.items(), key=lambda d:d[1], reverse = True)
    pre_label = sort_dic_list[0][0]
    row = [test_result_label, pre_label]
    return (a[0],row)