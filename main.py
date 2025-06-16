from read_data import read_arff
# from gp_nsga2 import gp_nsga2_classifier, count_selected_feat
import random
from functools import partial
from itertools import chain
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score,recall_score,f1_score,confusion_matrix
from numpy import mean
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import numpy as np
from baselinemethod import single_objective
import time
from scipy.stats import kstest
from scipy import stats
from eval_func import aucc,tpr_tnr
from func_tools import *
def g_mean(y_true,y_pred):
    # tn,fp,fn,tp=confusion_matrix(y_true,y_pred).ravel
    c=confusion_matrix(y_true,y_pred).flatten()
    tn,fp,fn,tp = c[0],c[1],c[2],c[3]
    sensitivity = tp/(tp+fn)
    specificity = tn/(tn+fp)
    gmean = math.sqrt(sensitivity*specificity)
    return gmean
resultspath = './result'

dir_name = './datasets'

methods = ['fsl_dist']

allpopauc = []
allpopf1 = []
allpopgmean = []
trainning_time_list = []
all_size = []

file_container = ['lapointe-2004-v2','colon','golub-1999-v1','leukemia','ionosphere','armstrong-2002-v1','shipp-2002-v1','dlbcl','gordon-2002','yeoh-2002-v1',"Lymphoma",'su-2001','tomlins-2006','lung','Dry_Bean_Dataset','yeast','letter-recognition']



with open(resultspath + '.txt', 'a') as f:

    f.write('method\t\t\t\tauc\t\t\t\tf1\t\t\t\tgmean\t\t\t\tsize\t\t\t\tfeatnum\t\t\t\tdataset\t\t\t\t\ttraingtime\n')
    for file_name in file_container:
        path = './changingdatasets/'+ file_name +'.csv'
        for method in methods:
                               
            for circulation in range(30):
                print(method)
                if file_name == 'Dry_Bean_Dataset' or file_name=='yeast'or file_name=='letter-recognition':
                    all_datas = np.loadtxt(path,dtype=np.float64,delimiter = ',',unpack=False)
                    feat_num= len(all_datas[0]) -1
                else:
                    all_datas, flat_class = read_arff(dir_name, file_name)
                    feat_num = len(all_datas[0]) - 1
                    
                    if flat_class == 2:
                        all_datas = np.array(all_datas)


                    else:
                        all_datas = np.loadtxt(path,dtype=np.float64,delimiter=',',unpack=False)
                num_datas = all_datas[:, :-1]
                num_label = all_datas[:, -1]

                print(len(num_datas))
                print(len(num_label))

                train_set,test_set,train_label,test_label = train_test_split(num_datas, num_label,
                                                                            train_size=0.7, test_size=0.3,
                                                                            random_state=circulation, stratify=num_label)

                all_datas1 = []
                for index, b in enumerate(train_label):
                    i = np.append(train_set[index], b)
                    i = i.tolist()
                    all_datas1.append(i)

                all_datas2 = []
                for index, b in enumerate(test_label):
                    i = np.append(test_set[index], b)
                    i = i.tolist()
                    all_datas2.append(i)

                majdatas1 = []
                mindatas1 = []
                for datas in all_datas1:
                    if datas[-1] == 1:
                        mindatas1.append(datas)
                    elif datas[-1] == 0:
                        majdatas1.append(datas)
                if len(majdatas1) <= len(mindatas1):
                    majdatas1, mindatas1 = mindatas1, majdatas1

                majdatas2 = []
                mindatas2 = []
                for datas in all_datas2:
                    if datas[-1] == 1:
                        mindatas2.append(datas)
                    elif datas[-1] == 0:
                        majdatas2.append(datas)
                print(len(majdatas2), len(mindatas2))
                if len(majdatas2) <= len(mindatas2):
                    majdatas2, mindatas2 = mindatas2, majdatas2
                minnum = len(mindatas1)
                majnum = len(majdatas1)
                ir = majnum/minnum
                print("训练集少数类，多数类", minnum, majnum)
                test_num = (len(mindatas2), len(majdatas2))
                print("测试集少数类，多数类：", test_num)
                data_testing = mindatas2 + majdatas2
                N = (len(mindatas1),len(majdatas1))
                start = time.time()
                new_mindatas1,new_majdatas1 = divide_dataset(mindatas1,majdatas1)
                pop,hof,log, toolbox,tools,pset = single_objective(method,all_datas1,majdatas1,mindatas1,new_mindatas1,new_majdatas1, feat_num,minnum,majnum,all_datas2)
                end = time.time()
                trainning_time_list.append(end-start) 
                all_size.append(len(hof[0]))
                func = toolbox.compile(expr=hof[0])
                change_data_testing = np.array(data_testing)
                y = change_data_testing[:, -1]
                output = list(map(lambda a: func(a[:-1]), data_testing))
                auc = roc_auc_score(y,output)   
                '''
                Note: The `roc_auc_score` function assumes that positive samples receive higher scores than negative ones.
                However, the GPD method does not explicitly enforce this ordering, which may result in negative samples
                having higher scores than positive ones. In such cases, the computed AUC might be less than 0.5.
                To correct this, simply use: auc = 1 - auc
                '''

                if auc < 0.5:
                    auc = 1-auc         
                pre_label_list = test_fsl(func,new_mindatas1,new_majdatas1,all_datas2)

                f1score = f1_score(test_label,pre_label_list)
                f1 = f1_score(test_label,pre_label_list)
                # auc = aucc(hof[0], toolbox, mindatas2, majdatas2, N)[0]
                gmean = g_mean(test_label,pre_label_list)
                print(auc,f1,gmean)
                # re_score_list.append(re_socre)
                allpopauc.append(auc)
                allpopf1.append(f1)
                allpopgmean.append(gmean)


                print("第", circulation, "次循环")
            meansize = mean(all_size)
            aveauc = mean(allpopauc)
            avef1 = mean(allpopf1)
            avegmean = mean(allpopgmean)
            bestauc = max(allpopauc)
            stdauc = np.std(allpopauc,ddof = 1)
            mean_training_time = mean(trainning_time_list)
            print("30次平均auc是", aveauc)
            print("30次best_auc",bestauc)
            print('30次标准差为',stdauc)

            mean_training_time = mean(trainning_time_list)
            f.write( str(method) +'\t\t\t\t'+ str(round( aveauc,4))
                + '\t\t\t\t'+str(round(avef1,4))+'\t\t\t\t'+str(round(avegmean,4))+'\t\t\t\t'+str(round(meansize,4))
                 +'\t\t\t\t'+str(round(feat_num,4))+'\t\t\t\t'+file_name+'\t\t\t\t'+str(mean_training_time)+'\n'+str(allpopauc)+'\n'+str(allpopf1)+'\n'+str(allpopgmean)+'\n')
            f.flush()
            del allpopauc[:]
            del allpopf1[:]
            del allpopgmean[:]
            del all_size[:]
            del trainning_time_list[:]
            print('列表已清空：',allpopauc)
            # del ach_list[:]
            # del best_pop[:]
            # del best_pop2[:]
            # del best_pop3[:]
            # del first_best_ensembles_auc[:]
            # del second_best_ensembles_auc[:]
            # del third_best_ensembles_auc[:]
    f.close()

