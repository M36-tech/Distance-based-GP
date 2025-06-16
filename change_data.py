# import numpy as np
# def test1():
#     x = 1
#     y = 2
#     return x,y,
# l = test1()
# print(l)
# x = [1,2,3,4,5,6,7,6,5]
# print(len(x),len(set(x)))
# print(x)
# method = ['dist','auc']
# def dist():
#     return sum(x)
# def auc():
#     return x
# for i in method:
#     if i == 'dist':
#         print('yes')
#     else:
#         print('no')
# a = [55,67,78]
# v = [1,2,3,1,2,3,1,2,3,1,2,3,1,2,3]
# c = [0.99,0.87,0.65]
# bv = np.std(c,ddof = 1 )
# # bv = np.std(a)
# print(bv)
# a = [1,2,3,1]
# x =5
# iw=0
# for i in a:
#     iw = iw +(i-x)**2
# print(iw)
# import random
# bw = []
# random.seed(0)
# for i in range(30):
#     a = random.randint(0, 32)
#     bw.append(a)
from read_data import read_arff
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import train_test_split
import os
import numpy as np
from collections import Counter
dir_name = r'D:\file\two_criterion\datasets'
def getfiles():
    filenames=os.listdir(r'D:\file\two_criterion\datasets')
    return filenames

def flatclass():
    # datalist = getfiles()
    datalist = ['lapointe-2004-v2',"Lymphoma",'su-2001','tomlins-2006','lung']
    #datalist = ['lapointe-2004-v2','armstrong-2002-v1','golub-1999-v1','colon','leukemia','shipp-2002-v1','laiho-2007','gordon-2002','yeoh-2002-v1','su-2001','tomlins-2006','lung','dlbcl']
    for i in datalist:
        # i = i[0:-5]
        all_datas, flat_class= read_arff(dir_name, i)
        all_datas =np.array(all_datas)
        print(i)
#     if flat_class>2:

#         for index,i in enumerate(all_datas[:,-1]):
#             if i == 0.0:
#                 pass
#             else:
#                 all_datas[index,-1] = 1.0
    # num_datas = all_datas[:, :-1]
    # num_label = all_datas[:, -1]
    # # print(all_datas[0])


    
    


#保存改变的数据
def main():

    filename = 'lung'
    all_datas, flat_class = read_arff(dir_name, filename)
    all_datas =np.array(all_datas)
    for index,i in enumerate(all_datas[:,-1]):
        if i == 3:
            all_datas[index,-1] = 1.0
        else:
            all_datas[index,-1] = 0.0
    path = 'D:/file/two_criterion/changingdatasets/'+ filename +'.csv'
    np.savetxt(path, all_datas, delimiter="," )
    data = np.loadtxt(path,dtype=np.float64,delimiter=',',unpack=False)
    num_datas = data[:, :-1]
    num_label = data[:, -1]
    print(Counter(num_label))
    # print(all_datas[0])

    # 划分数据集
    train_set,test_set,train_label,test_label = train_test_split(num_datas, num_label,
                                                                train_size=0.7, test_size=0.3,
                                                                stratify=num_label)
    majdatas = []
    mindatas = []
    for i in train_label:
        if i == 0:
            majdatas.append(i)
        else:
            mindatas.append(i)
    print(test_label)
    print(train_label)

    print(len(majdatas),len(mindatas))
    print(len(majdatas)/len(mindatas))
if __name__ == "__main__":
    # flatclass()
    main()
    pass














