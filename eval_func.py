import numpy as np
import math
from sklearn.metrics import roc_auc_score,average_precision_score
from scipy.stats import rankdata
from deap import gp
import re
from func_tools import get_mahalanobis,divide_two_subtrees,get_diff_class_mahalanobis,test_fsl
def count_terms(ind):
    num = 0
    for info in ind:
        if isinstance(info, gp.Terminal):
            num += 1
    return num


def count_prims(ind):
    num = 0
    for info in ind:
        if isinstance(info, gp.Primitive):
            num += 1
    return num


def count_fs(ind):
    """
    统计特征（不重复）
    :param ind: 
    :return: 
    """
    terms = []
    rands = []
    for info in ind:
        # print(info)
        if isinstance(info, gp.Terminal):
            terms.append(info)
        if isinstance(info, gp.rand):
            rands.append(info)
    c = len(set(rands))
    result = len(set(terms))-c
    return result
def count_fs1(ind):
    """
    统计特征（不重复）
    :param ind: 
    :return: 
    """
    # terms = []
    # # rands = []
    # for info in ind:
    #     # print(info)
    #     if isinstance(info, gp.Terminal):
    #         terms.append(str(info))
    # new_list = list(dict.fromkeys(terms))
    result_list=re.findall(r'\d+(?:\.\d+)?', str(ind))
    result = len(set(result_list))
    # print(result)
    return result
def acc_comp(ind, toolbox, Cmin, Cmaj, min_num, maj_num):
    func = toolbox.compile(expr=ind)
    try:
        func([])
        return 0, 1000
    except:
        Pc_min = list(map(lambda a: func(a[:-1]), Cmin))  # 少数类的输出（正类）
        tp = operate_count(Pc_min, 0, ">=")
        Pc_maj = list(map(lambda a: func(a[:-1]), Cmaj))  # 多数类的输出（负类）
        tn = operate_count(Pc_maj, 0, "<")
        acc = (tp+tn)/(min_num+maj_num)
        comp = len(ind)
        return acc, comp


def gpmo2(ind, a, feat_num, toolbox, Cmin, Cmaj, min_num, maj_num):
    func = toolbox.compile(expr=ind)
    try:
        func([])
        return 1,
    except:
        Pc_min = list(map(lambda a: func(a[:-1]), Cmin))  # 少数类的输出（正类）
        fn = operate_count(Pc_min, 0, ">=")
        Pc_maj = list(map(lambda a: func(a[:-1]), Cmaj))  # 多数类的输出（负类）
        fp = operate_count(Pc_maj, 0, "<")
        error_rate = (fp + fn) / (min_num + maj_num)
        gpmo = a*error_rate + (1-a)*(count_fs1(ind)/feat_num)
        return gpmo,


def bojar(ind, toolbox, k, Cmin, Cmaj, min_num, maj_num):
    func = toolbox.compile(expr=ind)
    try:
        func([])
        return 0,
    except:
        Pc_min = list(map(lambda a: func(a[:-1]), Cmin))  # 少数类的输出（正类）
        tp = operate_count(Pc_min, 0, ">=")
        Pc_maj = list(map(lambda a: func(a[:-1]), Cmaj))  # 多数类的输出（负类）
        tn = operate_count(Pc_maj, 0, "<")
        se = tp/min_num
        sp = tn/maj_num
        maxnodes = 2**k-1
        sy = (maxnodes-0.5*(len(ind))-0.5)/(maxnodes-1)
        return se*sp*sy,


def muni(ind, toolbox, a, Cmin, Cmaj, n):
    func = toolbox.compile(expr=ind)
    try:
        func([])
        return 0,
    except:
        Pc_min = list(map(lambda a: func(a[:-1]), Cmin))  # 少数类的输出（正类）
        tp = operate_count(Pc_min, 0, ">=")
        Pc_maj = list(map(lambda a: func(a[:-1]), Cmaj))  # 多数类的输出（负类）
        tn = operate_count(Pc_maj, 0, "<")
        f = tp + tn
        r = count_fs1(ind)
        fs = f * (1 + a * np.exp(-r/n))
        return fs,


def muni2(ind, toolbox, a, Cmin, Cmaj, n):
    func = toolbox.compile(expr=ind)
    try:
        func([])
        return 0,
    except:
        Pc_min = list(map(lambda a: func(a[:-1]), Cmin))  # 少数类的输出（正类）
        tp = operate_count(Pc_min, 0, ">=")
        Pc_maj = list(map(lambda a: func(a[:-1]), Cmaj))  # 多数类的输出（负类）
        tn = operate_count(Pc_maj, 0, "<")
        f = tp + tn
        r = count_fs(ind)
        fs = f * (1 + a * np.exp(-r/n))
        return fs,


def nag(ind, toolbox, Cmin, Cmaj, min_num, maj_num):
    func = toolbox.compile(expr=ind)
    try:
        func([])
        return 1, 0, 1000
    except:
        Pc_min = list(map(lambda a: func(a[:-1]), Cmin))  # 少数类的输出（正类）
        tp = operate_count(Pc_min, 0, ">=")
        Pc_maj = list(map(lambda a: func(a[:-1]), Cmaj))  # 多数类的输出（负类）
        fp = operate_count(Pc_maj, 0, ">=")
        tpr = tp / min_num
        fpr = fp / maj_num
        terms = count_terms(ind)
        return fpr, tpr, terms


def wang(ind, toolbox, Cmin, Cmaj, min_num, maj_num):
    func = toolbox.compile(expr=ind)
    try:
        func([])
        return 1, 0
    except:
        Pc_min = list(map(lambda a: func(a[:-1]), Cmin))  # 少数类的输出（正类）
        tp = operate_count(Pc_min, 0, ">=")
        Pc_maj = list(map(lambda a: func(a[:-1]), Cmaj))  # 多数类的输出（负类）
        fp = operate_count(Pc_maj, 0, ">=")
        tpr = tp/min_num
        fpr = fp/maj_num
        return fpr, tpr
def Izt(r, k, c):
    """   
    :return: r,0
    """
    if k >= 0 > c:
        return r
    else:
        return 0
def dist(ind, toolbox, Cmin, Cmaj):
    """
    :return: 正负类平均数之间的距离，越大越好
    """
    # print(ind)
    func = toolbox.compile(expr=ind)
    try:
        func([])
        return 0,
    except:
        Pc_min = list(map(lambda a: func(a[:-1]), Cmin))  # 少数类的输出（正类）
        # print(len(Pc_min), type(Pc_min), Pc_min)
        Pc_maj = list(map(lambda a: func(a[:-1]), Cmaj))  # 多数类的输出（负类）
        # print(len(Pc_maj), type(Pc_maj), Pc_maj)
        umin = np.mean(Pc_min)
        umaj = np.mean(Pc_maj)
        omin = np.std(Pc_min)
        omaj = np.std(Pc_maj)
        # print("umin:", umin)
        # print("umaj:", umaj)
        # print("omin:", omin)
        # print("omaj:", omaj)
        if omin+omaj == 0:
            return 0,
        else:
            result = (abs(umin-umaj)/(omin+omaj))*Izt(2, umin, umaj)
        # print("适应度函数值：", result)
        return result,
def operate_count(a, number, operator):
    num = 0
    if operator == '>=':
        for i in a:
            if i >= number:
                num += 1
    elif operator == '<':
        for i in a:
            if i < number:
                num += 1
    else:
        print('something wrong!')
    return num
def ave(ind, w, toolbox, Cmin, Cmaj, min_num, maj_num):
    func = toolbox.compile(expr=ind)
    Pc_min = list(map(lambda a: func(a[:-1]), Cmin))  # 少数类的输出（正类）
    tpr = operate_count(Pc_min, 0, ">=")/min_num
    Pc_maj = list(map(lambda a: func(a[:-1]), Cmaj))  # 多数类的输出（负类）
    tnr = operate_count(Pc_maj, 0, "<")/maj_num
    ave = w*tpr + (1-w)*tnr
    return ave,
def sig(x):
    # 对sigmoid函数的优化，避免了出现极大的数据溢出
    if x >= 0:
        return 2.0/(1+np.exp(-x))-1
    else:
        return (2*np.exp(x))/(1+np.exp(x))-1
def amse(ind, toolbox, Cmin, Cmaj):
    """  
    :param ind: 
    :param toolbox: 
    :param k: 
    :param Cmin: 
    :param Cmaj: 
    :return: 
    """
    func = toolbox.compile(expr=ind)
    try:
        func([])
        return 0,
    except:
        Nmin = len(Cmin)
        Nmaj = len(Cmaj)
        k = [(0.5, Nmin, Cmin), (-0.5, Nmaj, Cmaj)]
        result = []
        for c in k:
            b = list(map(lambda a: pow(sig(func(a[:-1]))-c[0], 2)/(c[1]*2), c[2]))
            result.append(1-sum(b))
        result = sum(result)/2
        # print("适应度值为：", result)
        return result,
def tpr_fpr(threshold, list, Lmin, min_num, maj_num):
    tp = 0
    fp = 0
    for item in list:
        # print(item[0], threshold, item[1], Lmin)
        if item[0] >= threshold:
            if item[1] == Lmin:
                tp += 1
            else:
                fp += 1
    # print("-"*50, tp, fp)
    tpr, fpr = tp / min_num, fp / maj_num
    # print("-"*50, tpr, fpr)
    return tpr, fpr

def tpr_tnr(threshold, list, Lmin, min_num, maj_num):
    tp = 0
    fp = 0
    for item in list:
        # print(item[0], threshold, item[1], Lmin)
        if item[0] >= threshold:
            if item[1] == Lmin:
                tp += 1
            else:
                fp += 1
    # print("-"*50, tp, fp)
    tpr, tnr = tp / min_num, 1 - (fp / maj_num)
    # print("-"*50, tpr, fpr)
    return tpr, tnr

def my_auc(ind, toolbox, Cmin, Cmaj, test_num):
    datatesting = Cmin + Cmaj
    func = toolbox.compile(ind)
    outp = list(map(lambda a: func(a[:-1]), datatesting))
    label = [data[-1] for data in datatesting]
    outp_label = list(zip(outp, label))
    #print(outp_label)
    b = sorted(outp_label, key=(lambda x: x[0]))
    #print(b)
    tprs = []
    fprs = []
    c = list(set(outp))
    # print("-"*50, len(c))
    c.sort(reverse=True)
    # print("-" * 50, len(c), c)
    for threshold in c:
        tpr, fpr = tpr_fpr(threshold, b, 1, test_num[0], test_num[1])
        tprs.append(tpr)
        fprs.append(fpr)
    # print(tprs)
    # print(fprs)
    # graphl(fprs, tprs)
    result = []
    for i in range(len(c) - 1):
        s = (tprs[i + 1] + tprs[i]) * (fprs[i + 1] - fprs[i]) / 2
        # print("梯形的面积为：", s)
        result.append(s)
        # print("结果为：", result)
        myauc = sum(result)
    # return sum(result)
    return myauc,
#双标准函数
def two_criterion(individual,toolbox,majdatas,mindatas):
    func = toolbox.compile(expr=individual)
    majnum = len(majdatas)
    minnum = len(mindatas)
    mindata = []
    for i in mindatas:
        if i not in mindata:
            mindata.append(i)
    minnum1 = len(mindata)
    #c1
    iw = 0
    Pc_min = list(map(lambda a: func(a[:-1]), mindatas))
    Pc_maj = list(map(lambda a: func(a[:-1]), majdatas))
    t = max(Pc_maj)
    for i in Pc_min:
        if i>t and i>=0:
            iw += 1
        else:
            pass
    c1 = iw/minnum1
    #c2
    umin = np.mean(Pc_min)
    umaj = np.mean(Pc_maj)
    u = (minnum*umin+majnum*umaj)/(minnum+majnum)
    pumin = list(map(lambda a:(a-u)**2),Pc_min)
    pumaj = list(map(lambda a:(a-u)**2),Pc_maj)
    pu = sum(pumin)+sum(pumaj)

    if pu == 0:
        c2 =0
    else:
        c2 = math.sqrt((minnum*(umin-u)**2+majnum*(umaj-u)**2)/pu)
    a_s = (c1,c2)
    return a_s
def Izrcorr(r,umin,umaj):
    if umin>=0 and umaj <0:
        return r
    else:
        return 0


def corr(individual,toolbox,majdatas,mindatas):

    func = toolbox.compile(expr=individual)
    majnum = len(majdatas)
    minnum = len(mindatas)

    Pc_min = list(map(lambda a: func(a[:-1]), mindatas))
    Pc_maj = list(map(lambda a: func(a[:-1]), majdatas))
    umin = np.mean(Pc_min)
    umaj = np.mean(Pc_maj)
    u = (minnum*umin+majnum*umaj)/(minnum+majnum)
    pumin = list(map(lambda a:(a-u)**2,Pc_min))
    pumaj = list(map(lambda a:(a-u)**2,Pc_maj))
    pu = sum(pumin)+sum(pumaj)
    if pu == 0:
        r = 0
    else:
        r = math.sqrt((minnum*(umin-u)**2+majnum*(umaj-u)**2)/pu)
    corr = (r+Izrcorr(1, umin, umaj))/2
    return corr,
    

    pass
def G_mean(ind, toolbox, Cmin, Cmaj, min_num, maj_num):
    func = toolbox.compile(expr=ind)
    Pc_min = list(map(lambda a: func(a[:-1]), Cmin))  # 少数类的输出（正类）
    tpr = operate_count(Pc_min, 0, ">=")/min_num
    Pc_maj = list(map(lambda a: func(a[:-1]), Cmaj))  # 多数类的输出（负类）
    tnr = operate_count(Pc_maj, 0, "<")/maj_num
    g_mean= math.sqrt(tpr + tnr)
    return g_mean,


            


    
def aucw(ind,toolbox,majdatas,mindatas):
    func = toolbox.compile(expr=ind)
    majdata = []
    mindata = []
    for i in majdatas:
        if i not in majdata:
            majdata.append(i)
    for i in mindatas:
        if i not in mindata:
            mindata.append(i)
        
    majnum = len(majdata)
    minnum = len(mindata)
    Pc_min = list(map(lambda a: func(a[:-1]), mindatas))
    Pc_maj = list(map(lambda a: func(a[:-1]), majdatas))
    iw = 0

    multidata = float(majnum*minnum)
    for i in Pc_min:
        for j in Pc_maj:
            if i>j and i>=0:
                iw+=1
            else:
                pass
    iw = float(iw)
    aucww = iw/multidata
    return aucww,
def no_zero_aucw(ind,toolbox,majdatas,mindatas):
    func = toolbox.compile(expr=ind)
    majdata = []
    mindata = []
    for i in majdatas:
        if i not in majdata:
            majdata.append(i)
    for i in mindatas:
        if i not in mindata:
            mindata.append(i)
        
    majnum = len(majdata)
    minnum = len(mindata)
    Pc_min = list(map(lambda a: func(a[:-1]), mindatas))
    Pc_maj = list(map(lambda a: func(a[:-1]), majdatas))
    iw = 0

    multidata = float(majnum*minnum)
    for i in Pc_min:
        for j in Pc_maj:
            if i>j :
                iw+=1
            else:
                pass
    iw = float(iw)
    aucww = iw/multidata
    return aucww,
def realauc(ind, toolbox, data_training):
    func = toolbox.compile(expr=ind)
    change_data_training = np.array(data_training)
    y = change_data_training[:, -1]
    myre = list(map(lambda a: func(a[:-1]), data_training))
    myre = np.array(myre)
    realauc = roc_auc_score(y,myre)
    return realauc,
def auc_dist(ind,toolbox,majdatas,mindatas):
    aucww = aucw(ind,toolbox,majdatas,mindatas)
    dist1 = dist(ind, toolbox,mindatas, majdatas)
    if aucww[0] +dist1[0] == 0:
        auc_dist = 0
    else:
        auc_dist = (aucww[0]*dist1[0])/(0.95*aucww[0]+0.05*dist1[0])
    return auc_dist,
def prauc(ind, toolbox, data_training):
    func = toolbox.compile(expr=ind)
    change_data_training = np.array(data_training)
    y = change_data_training[:, -1]
    myre = list(map(lambda a: func(a[:-1]), data_training))
    myre = np.array(myre)
    prauc = average_precision_score(y,myre)
    return prauc,
def aucc(ind, toolbox, Cmin, Cmaj, N):
    func = toolbox.compile(expr=ind)
    Pc_min = list(map(lambda a: func(a[:-1]), Cmin))  # 少数类的输出（正类）
    Pc_maj = list(map(lambda a: func(a[:-1]), Cmaj))  # 多数类的输出（负类）
    o = np.array(Pc_min + Pc_maj)
    r = rankdata(o)
    sum_r_min = sum(r[:N[0]])
    auc = (sum_r_min - N[0]*(N[0]+1)/2)/(N[0]*N[1])
    return auc,

def m_dist(ind, toolbox, Cmin, Cmaj):
    sub_tuple = divide_two_subtrees(ind)
    if len(sub_tuple) == 2:
        sub1,sub2 = sub_tuple[0],sub_tuple[1]
        func1 = toolbox.compile(expr=sub1)
        func2 = toolbox.compile(expr=sub2)
        Pc_min1 = list(map(lambda a: func1(a[:-1]), Cmin))  # 少数类的输出（正类）
        Pc_maj1 = list(map(lambda a: func1(a[:-1]), Cmaj))  # 多数类的输出（负类）
        Pc_min2 = list(map(lambda a: func2(a[:-1]), Cmin))  # 少数类的输出（正类）
        Pc_maj2 = list(map(lambda a: func2(a[:-1]), Cmaj))  # 多数类的输出（负类）
        Pc_min11 = np.array(Pc_min1).reshape(-1, 1)
        Pc_min22 = np.array(Pc_min2).reshape(-1, 1)
        Pc_maj11 = np.array(Pc_maj1).reshape(-1, 1)
        Pc_maj22 = np.array(Pc_maj2).reshape(-1, 1)
        Pc_min = np.hstack((Pc_min11,Pc_min22))
        Pc_maj = np.hstack((Pc_maj11,Pc_maj22))
        same_class_dist = []
        diff_class_dist = []
        for index,row in enumerate(Pc_min):
            same_class_dist.append(get_mahalanobis(Pc_min, index, -1))


            diff_class_dist.append(get_diff_class_mahalanobis(Pc_maj, Pc_min[index], -1))
        for index,row in enumerate(Pc_maj):
            same_class_dist.append(get_mahalanobis(Pc_maj, index, -1))
            diff_class_dist.append(get_diff_class_mahalanobis(Pc_min, Pc_maj[index], -1))




        # same_class_dist = list(map(lambda a: get_mahalanobis(Pc_min, a, -1), Pc_min)) + list(map(lambda a: get_mahalanobis(Pc_maj, a, -1), Pc_maj))
        # diff_class_dist = list(map(lambda a: get_diff_class_mahalanobis(Pc_min, a, Pc_maj), Pc_min)) +list(map(lambda a: get_diff_class_mahalanobis(Pc_maj, a, Pc_min), Pc_maj))
        same_class_std = np.std(same_class_dist)
        diff_class_std = np.std(diff_class_dist)
        # div_list = [b/a for a,b in zip(same_class_dist,diff_class_dist)]
        if np.mean(same_class_dist)*(same_class_std+diff_class_std) == 0:
            return 0,
        else:
            mdist = (np.mean(diff_class_dist)/np.mean(same_class_dist))/(same_class_std+diff_class_std)
            return mdist,
    else:
        return 0,
# def fsl_dist(ind, toolbox, Cmin, Cmaj):
#     min_correct = 0
#     maj_correct = 0
#     func = toolbox.compile(expr=ind)
#     for data in Cmin:

#         # print(data['min'][0][:-1])
#         minput = func(data['min'][:-1])
#         majput = func(data['maj'][:-1])
#         preput = func(data['pre'][:-1])
    
#         if abs(minput - preput) >= abs(majput - preput):
#             pass
#         else:
#             min_correct += 1
#     for data in Cmaj:
#         minput = func(data['min'][:-1])
#         majput = func(data['maj'][:-1])
#         preput = func(data['pre'][:-1])
    
#         if abs(minput - preput) >= abs(majput - preput):
#             maj_correct +=1
            
#         else:
#             pass
#     tpr,tnr = min_correct/len(Cmin),maj_correct/len(Cmaj)
#     g_mean = math.sqrt(tpr*tnr)
#     return g_mean,

def fsl_dist(ind, toolbox, Cmin, Cmaj,new_datamin,new_datamaj):
    min_correct = 0
    maj_correct = 0
    func = toolbox.compile(expr=ind)
    for data in new_datamin:

        # print(data['min'][0][:-1])
        minput = func(data['min'][:-1])
        majput = func(data['maj'][:-1])
        preput = func(data['pre'][:-1])
    
        if abs(minput - preput) >= abs(majput - preput):
            pass
        else:
            min_correct += 1
    for data in new_datamaj:
        minput = func(data['min'][:-1])
        majput = func(data['maj'][:-1])
        preput = func(data['pre'][:-1])
    
        if abs(minput - preput) >= abs(majput - preput):
            maj_correct +=1
            
        else:
            pass
    Pc_min = list(map(lambda a: func(a[:-1]), Cmin))  # 少数类的输出（正类）
    # print(len(Pc_min), type(Pc_min), Pc_min)
    Pc_maj = list(map(lambda a: func(a[:-1]), Cmaj))  # 多数类的输出（负类）
    # print(len(Pc_maj), type(Pc_maj), Pc_maj)
    umin = np.mean(Pc_min)
    umaj = np.mean(Pc_maj)
    omin = np.std(Pc_min)
    omaj = np.std(Pc_maj)
    # print("umin:", umin)
    # print("umaj:", umaj)
    # print("omin:", omin)
    # print("omaj:", omaj)
    if omin+omaj == 0:
        return 0,
    else:
        result = (abs(umin-umaj)/(omin+omaj))*((min_correct+maj_correct)/len(new_datamin))

    return result,

def pfc(population,toolbox,new_mindatas1,new_majdatas1,all_datas2):
    func_list = []
    PFC_list = []
    pre_label_list = []
    all_datas2 = np.array(all_datas2)
    for ind in population:
        func_list.append(toolbox.compile(expr = ind))
    for func in func_list:
        pre_label_list.append(np.array(test_fsl(func,new_mindatas1,new_majdatas1,all_datas2)))
    for index,pre_label in enumerate(pre_label_list):
        pfc = 0
        y = pre_label_list[:]
        del y[index]
        for pre_label1 in y:          
            i = sum(pre_label!=pre_label1) #个数
            err1 = sum(pre_label != all_datas2[:,-1])
            err2 = sum(pre_label1 != all_datas2[:,-1])
            if (err1+ err2)  == 0:
                pfc = 0
            else:
                pfc += i/(err1+err2)
        PFC_list.append(pfc/499)
    return PFC_list


        

        






