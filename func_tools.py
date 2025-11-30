import matplotlib.pyplot as plt
from collections import defaultdict, namedtuple
import math
import numpy as np
from numpy import dot
import heapq
import random
from functools import reduce
import pandas as pd
# ✅ 从列表中删除 N 个随机项目
def pfc():
    pass

def remove_n_random_items(lst, n):
    to_delete = set(random.sample(range(len(lst)), n))

    return [
        item for index, item in enumerate(lst)
        if not index in to_delete
    ]
def select_n_random_items(lst, n):
    to_select = set(random.sample(range(len(lst)), n))

    return [
        item for index, item in enumerate(lst)
        if index   in to_select
    ]

def DictinList_duplicate(data_list):
    """
    列表套字典去重
    :return:
    """
    seen = set()
    new_l = []
    for d in data_list:
        t = tuple(d.items())
        if t not in seen:
            seen.add(t)
            new_l.append(d)
    return new_l

# 👇️ ['.', 'com', 'a', 'c']
def distinct(data_list):
    new_data_list = []
    for data in data_list:
        if data not in new_data_list:
            new_data_list.append(data)
    return new_data_list

# def divide_dataset(data_training_min,data_traing_maj):
#     new_maj_data_list = []
#     new_min_data_list = []
#     maj_lenth = len(data_traing_maj)
#     min_lenth = len(data_training_min)
#     int_ir = int(maj_lenth//min_lenth)
#     for index,data in enumerate(data_traing_maj):
#         y = data_traing_maj[:]
#         del y[index]
#         new_maj_data = {'min': np.random.choice(data_training_min),'maj': np.random.choice(y),'pre':data}
#         new_maj_data_list.append(new_maj_data)
#     for _ in range(int_ir):
#         for index,data in enumerate(data_training_min):
#             y = data_training_min[:]
#             del y[index]
#             new_min_data = {'min': np.random.choice(y),'maj': np.random.choice(data_traing_maj),'pre':data}
#             new_min_data_list.append(new_min_data)
#     devi = maj_lenth - min_lenth* int_ir
#     for index,data in enumerate(data_training_min[:devi]):

#         y = data_training_min[:]
#         del y[index]
#         new_min_data = {'min': np.random.choice(y),'maj': np.random.choice(data_traing_maj),'pre':data}
#         new_min_data_list.append(new_min_data)
#         devi -= 1
        
def divide_dataset(data_training_min,data_traing_maj):
    new_maj_data_list = []
    new_min_data_list = []
    maj_lenth = len(data_traing_maj)
    for _ in range(1):
        for index,data in enumerate(data_traing_maj):
            y = data_traing_maj[:]
            del y[index]
            new_maj_data = {'min': (select_n_random_items(data_training_min, 1))[0],'maj': select_n_random_items(y, 1)[0],'pre':data}
            new_maj_data_list.append(new_maj_data)
    new_maj_data_list = distinct(new_maj_data_list)
    if len(new_maj_data_list) > maj_lenth:
        n = len(new_maj_data_list) - maj_lenth
        new_maj_data_list = remove_n_random_items(new_maj_data_list, n)
    # for _ in range(int_ir*2):
    #     for index,data in enumerate(data_training_min):
    #         y = data_training_min[:]
    #         del y[index]
    #         new_min_data = {'min': np.random.choice(y),'maj': np.random.choice(data_traing_maj),'pre':data}
    #         new_min_data_list.append(new_min_data)
    while len(new_min_data_list) <len(new_maj_data_list):
        for index,data in enumerate(data_training_min):
            y = data_training_min[:]
            del y[index]
            new_min_data = {'min': (select_n_random_items(y, 1))[0],'maj': select_n_random_items(data_traing_maj, 1)[0],'pre':data}
            new_min_data_list.append(new_min_data)
        new_min_data_list = distinct(new_min_data_list)
    new_min_data_list = distinct(new_min_data_list)
    if len(new_min_data_list) > maj_lenth:
        n = len(new_min_data_list) - maj_lenth
        new_min_data_list = remove_n_random_items(new_min_data_list, n)
    return new_min_data_list,new_maj_data_list


def test_gpd(func, Cmin, Cmaj,testdatalist):
    prelabel_list = []
    test_list = []
    # for index,data in enumerate(Cmin):

    #     # print(data['min'][0][:-1])
    #     minput = func(data['min'][:-1])
    #     majput = func(data['maj'][:-1])
    #     preput = func(data['pre'][:-1])
    
    #     if abs(minput - preput) < abs(majput - preput):
    #         min_index_list.append(index)
    #     else:
    #         pass
    for data in Cmin:
        minput = func(data['min'][:-1])
        majput = func(data['maj'][:-1])
        preput = func(data['pre'][:-1])

    
        if abs(minput - preput) < abs(majput - preput):
            test_list.append(data)
            
        else:
            pass
    
    for data in Cmaj:
        minput = func(data['min'][:-1])
        majput = func(data['maj'][:-1])
        preput = func(data['pre'][:-1])

    
        if abs(minput - preput) >= abs(majput - preput):
            test_list.append(data)
            
        else:
            pass
    for data in testdatalist:
        min_num = 0
        maj_num = 0
        for i in test_list:
            minput = func(i['min'][:-1])
            majput = func(i['maj'][:-1])
            preput = func(data[:-1])
            if abs(minput - preput) >= abs(majput - preput):
                
                maj_num +=1


            
            else:
                min_num +=1
        if maj_num >= min_num:
            prelabel_list.append(0)
        else:
            prelabel_list.append(1)
            
            
    return prelabel_list       

  
            
            


    
def max_n_list(n,m):
    '''求list前n个最大值'''
    copy_list = m[:]
    max_number = heapq.nlargest(n,copy_list) 
    max_index = []
    for t in max_number:
        index = copy_list.index(t)
        max_index.append(index)
        copy_list[index] = 0
    return max_index

    

    
def slop(x_list,y_list):
    z1 = np.polyfit(x_list, y_list, 2) #用3次多项式拟合，输出系数从高到0
    p1 = np.poly1d(z1)
    slop_list = []
    for index,x in enumerate(x_list):

        if index != 0:
            if (x_list[index]-x_list[index-1]) == 0:
                slop_list.append(1)
            else:

                slop_list.append((p1(x_list[index])-p1(x_list[index-1]))/(x_list[index]-x_list[index-1]))
    return slop_list
def threshold_class(func,threshold,data):
    if func(data[:-1]) >= threshold:
        return 1.0
    else:
        return 0.0

    
def acc_gpknn(ind,toolbox,data_trainning_min,data_traing_maj,data_testing):
    min_list = [1]*len(data_trainning_min)
    maj_list = [0]*len(data_traing_maj)
    ind1,ind2 =divide_two_subtrees(ind)
    func1 = toolbox.compile(expr=ind1)
    func2 = toolbox.compile(expr=ind2)
    Pc_min1 = list(map(lambda a: func1(a[:-1]), data_trainning_min))  # 少数类的输出（正类）
    Pc_maj1 = list(map(lambda a: func1(a[:-1]), data_traing_maj))  # 多数类的输出（负类）
    Pc_min2 = list(map(lambda a: func2(a[:-1]), data_trainning_min))  # 少数类的输出（正类）
    Pc_maj2 = list(map(lambda a: func2(a[:-1]), data_traing_maj))  # 多数类的输出（负类）
    Pc_min11 = np.array(Pc_min1).reshape(-1, 1)
    Pc_min22 = np.array(Pc_min2).reshape(-1, 1)
    Pc_maj11 = np.array(Pc_maj1).reshape(-1, 1)
    Pc_maj22 = np.array(Pc_maj2).reshape(-1, 1)
    Pc_min = np.hstack((Pc_min11,Pc_min22))
    Pc_maj = np.hstack((Pc_maj11,Pc_maj22))
    Pc_min_withlabel = np.hstack((Pc_min,np.array(min_list).reshape(-1, 1)))
    Pc_maj_withlabel = np.hstack((Pc_maj,np.array(maj_list).reshape(-1, 1)))
    Pc_withlabel = np.vstack((Pc_min_withlabel,Pc_maj_withlabel))
    test_op1 = np.array(list(map(lambda a: func1(a[:-1]),data_testing))).reshape(-1, 1)
    test_op2 = np.array(list(map(lambda a: func2(a[:-1]),data_testing))).reshape(-1, 1)
    test_op = np.hstack((test_op1,test_op2))
    dist_list = []
    predict_label = []
    for row in test_op:
        distlist = []
        for index,row1 in enumerate(Pc_withlabel):


            distlist.append(get_diff_class_mahalanobis(Pc_withlabel[:,:2],row, index))
        dist_list.append(distlist)
    for dist in dist_list:

        min_number = heapq.nlargest(5, dist)
        min_index = []
        for t in min_number:
            index = dist.index(t)
            min_index.append(index)
            a = float('inf')
            dist[index] = a
        label_list = Pc_withlabel[min_index,[2]].tolist()
        if label_list.count(0)/5 >= 3/5:
            predict_label.append(int(0))
        else:
            predict_label.append(int(1))
    true_false_list = np.array(predict_label) == np.array(data_testing)[:,2]

    min_acc = np.sum(true_false_list == 1)/np.sum(Pc_withlabel[:,2] == 1)
    maj_acc = np.sum(true_false_list == 0)/np.sum(Pc_withlabel[:,2] == 0)
    return min_acc,maj_acc





    
    
    











def get_diff_class_mahalanobis(x, y, j):
    '''
    不同类
    :param x: 两点矩阵
    :param y: 点
    :param j: 到第j个点  （j为-1时代表均值）  
    :return: distance 
    '''
    xy = np.vstack((x,y))
    xT = xy.T  # 求转置
    D = np.cov(xT)  # 求协方差矩阵
    invD = np.linalg.pinv(D)  # 协方差逆矩阵
    # assert 0 <= i < x.shape[0], "点 1 索引超出样本范围。"
    assert -1 <= j < xy.shape[0], "点 2 索引超出样本范围。"
    x_A = y
    x_B = x.mean(axis=0) if j == -1 else xy[j]
    tp = x_A - x_B
    return np.sqrt(dot(dot(tp, invD), tp.T))
def get_mahalanobis(x, i, j):
    """
    马氏距离
    :param x: 两点矩阵
    :param i: 求第i个点到均值之间的马氏距离（j为-1时代表均值）
    :param j: 到第j个点  （j为-1时代表均值）  
    :return: distance 
    """
    xT = x.T  # 求转置
    D = np.cov(xT)  # 求协方差矩阵
    invD = np.linalg.pinv(D)  # 协方差逆矩阵
    assert 0 <= i < x.shape[0], "点 1 索引超出样本范围。"
    assert -1 <= j < x.shape[0], "点 2 索引超出样本范围。"
    x_A = x[i]
    x_B = x.mean(axis=0) if j == -1 else x[j]
    tp = x_A - x_B
    return np.sqrt(dot(dot(tp, invD), tp.T))

def divide_two_subtrees(ind):
    try:
        ind = str(ind)
        sub_ind = ind[4:-1]
        symbol_list1 = []
        symbol_list2 = []
        sub_ind1 = ''
        sub_ind2 = ''
        # print(ind)
        if sub_ind[0] == 'f':
            sub_ind1 = sub_ind[:sub_ind.find(",")]
            sub_ind2 = sub_ind[2+len(sub_ind1):]
        else:


            for index,i in enumerate(sub_ind):
                if i == '(':
                    symbol_list1.append(i)
                elif i == ')':
                    symbol_list2.append(i)
                else:
                    pass
                if len(symbol_list1) == len(symbol_list2) and len(symbol_list1) != 0 :
                    sub_ind1 = sub_ind[:index+1]
                    sub_ind2 = sub_ind[index +3:]
                    break

        return (sub_ind1,sub_ind2)
    except:
        print(ind)
        return ind,
# 统计叶子结点个数
def count_leaf_nodes(ind):
    ind_str = str(ind)
    leaf_nodes = ind_str.count("f")
    return leaf_nodes


# 统计选择出来的特征
def count_selected_feat(ind, pset):
    list = []
    ind_str = str(ind)
    for f in reversed(pset.arguments):
        if ind_str.find(f) != -1:
            list.append(f)
            ind_str = ind_str.replace(f, '')
    return list


# 生成二维空列表
def init_two_dimensional_list(rows):
    list = []
    for row in range(rows):
        list.append([])
    return list


def graph(list, file_name):
    list1 = []
    list2 = []
    for comp ,acc in list:
        list1.append(comp)
        list2.append(acc)

    #print(list2)


    #plt.annotate('局部最大', xy=(2, 1), xytext=(3, 1.5), arrowprops=dict(facecolor='black', shrink=0.05))
    # plt.grid(axis = 'y')
    for a ,b in list:
         plt.text(a,b,'%.4f' % b)
    plt.plot(list1, list2)
    # for comp, acc in list:
    #     plt.plot(acc, comp, "r--", marker="*", lw=2)
    #matplotlib将使用rcParams字典中的配置进行绘图
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 步骤一（替换sans-serif字体）
    plt.rcParams['axes.unicode_minus'] = False  # 步骤二（解决坐标轴负数的负号显示问题）
    plt.title(file_name + '-pareto前沿')
    plt.xlabel('特征数')
    # plt.ylabel('复杂度')
    plt.ylabel('-准确率')
    plt.axis([0,1,0,1])
    #plt.tight_layout()
    plt.grid()
    plt.savefig(f"C:\\Users\89301\\Desktop\\data image3\\{file_name}")
    plt.show()


def graph_inviduals(inviduals, file_name, label):
    for i, ind in enumerate(inviduals):
        # print("种群中最优个体%d及其适应度值为：" % (i + 1), ind, ind.fitness.values)
        plt.plot(ind.fitness.values[0], ind.fitness.values[1], "r--", marker="o", lw=2)
        # plt.scatter(ind.fitness.values[0], ind.fitness.values[1], marker="o", s=10)
    # for ind in pareto_first_front:
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 步骤一（替换sans-serif字体）
    plt.rcParams['axes.unicode_minus'] = False  # 步骤二（解决坐标轴负数的负号显示问题）
    plt.title(file_name + '-pareto前沿' + "(" + label + ")")
    plt.xlabel('-准确率')
    # plt.ylabel('复杂度')
    plt.ylabel('特征数')
    plt.tight_layout()
    plt.grid()
    plt.show()


# 去重
def distinct(pareto_first_front):
    pareto_first_front_str = map(str, pareto_first_front)
    pareto_first_front_str = set(pareto_first_front_str)
    # for ind in pareto_first_front_str:
    #     print(ind)
    pareto_no_repeat = []
    for ind in pareto_first_front:
        if str(ind) in pareto_first_front_str:
            pareto_no_repeat.append(ind)
            pareto_first_front_str.remove(str(ind))
    return pareto_no_repeat

#支配关系确认：特征数和正确率组成
def my_dominates (onepoint, otherpoint):

    not_equal = False
    for one_point, other_point in zip(onepoint, otherpoint):
        if one_point < other_point:
            not_equal = True
        elif one_point > other_point:
            return False
    return not_equal

#对于特征数和正确率非支配排序
def my_nondominedsort(individuals, k,first_front_only):
    fits = []
    for ind in individuals:
        fits.append(ind)
    #print("q"*5,fits)

    current_front = []
    next_front = []
    dominating_fits = defaultdict(int)
    dominated_fits = defaultdict(list)
    # the first sorted
    for i in range(0,len(fits)-1):
        fit_i = fits[i]
        for fit_j in fits[i + 1:]:
            if my_dominates(fit_i,fit_j):
                dominating_fits[fit_j] += 1
                dominated_fits[fit_i].append(fit_j)
            elif my_dominates(fit_j,fit_i):
                dominating_fits[fit_i] += 1
                dominated_fits[fit_j].append(fit_i)
        if dominating_fits[fit_i] == 0:
            current_front.append(fit_i)

    #print("z"*5,current_front)

    fronts = [[]]

    # sava the first pareto
    for fit in current_front:
        fronts[-1].append(fit)
    pareto_sorted = len(fronts[-1])

    # continue the remain sorted
    if not first_front_only:
        N = min(len(individuals), k)
        while pareto_sorted < N:
            fronts.append([])
            for fit_p in current_front:
                for fit_d in dominated_fits[fit_p]:
                    dominating_fits[fit_d] -= 1
                    if dominating_fits[fit_d] == 0:
                        next_front.append(fit_d)
                        pareto_sorted += len(fit_d)
                        fronts[-1].append(fit_d)
            current_front = next_front[:]
            next_front = []

    return fronts

def find_best_individual(pareto_first_front,pset):

    distances = []
    for ind in pareto_first_front:
        value = ind.fitness.values
        dis = math.sqrt(math.pow(value[0],2)+math.pow((value[1]-1),2))
        distances.append(dis)

    c = min(distances)
    min_list = []
    #看最优目标是否是唯一个体
    if distances.count(c) > 1:
        first_pos = 0
        for i in range(distances.count(c)):
            new_list = distances[first_pos:]
            select_index = new_list.index(c) + first_pos
            first_pos = select_index + 1
            k = len(count_selected_feat(pareto_first_front[select_index],pset))
            leaf_num = count_leaf_nodes(pareto_first_front[select_index])
            min_list.append((select_index,k,leaf_num))
        min_list.sort(key=lambda a:(a[1],a[2]))
        ev_min_dis = min_list[0]
        index = ev_min_dis[0]
        return pareto_first_front[index]
    else:
        index = distances.index(c)
        return pareto_first_front[index]

def point_line_distance(k, b, point):
    c = k * point[0] - point[1] + b
    a = k * k + 1
    dis = abs(c) / math.sqrt(a)
    return dis


def find_min_distance(k, b, pareto_first_front, pset):
    dis_list = []

    for ind in pareto_first_front:
        i = point_line_distance(k, b, ind.fitness.values)
        dis_list.append(i)

    min_distance = min(dis_list)
    if dis_list.count(min_distance) > 1:
        min_list = []
        first_pos = 0
        for i in range(dis_list.count(min_distance)):
            new_list = dis_list[first_pos:]
            select_index = new_list.index(min_distance) + first_pos
            first_pos = select_index + 1
            feat_num = len(count_selected_feat(pareto_first_front[select_index], pset))
            leaf_num = count_leaf_nodes(pareto_first_front[select_index])
            min_list.append((select_index, feat_num,leaf_num))

        min_list.sort(key=lambda a: (a[1],a[2]))
        ev_min_dis = min_list[0]
        index = ev_min_dis[0]
        return pareto_first_front[index]
    else:
        index = dis_list.index(min_distance)
        return pareto_first_front[index]
#计算两点之间的斜率
def two_point_slope(first_point, second_point):
    two_fpr = first_point[0]-second_point[0]
    two_tpr = first_point[1]-second_point[1]
    if two_fpr == 0:
        return -10000
    else:
        k = two_tpr/two_fpr
        return k

#找到斜率之间的集成分类器
def find_best_ensembles (pareto_first_front,min_slope,max_slope):
    selected_best_ensembles = []
    perfect_point = (0, 1)

    for ind in pareto_first_front:
        current_point = ind.fitness.values
        k = two_point_slope(perfect_point,current_point)
        if(k > max_slope):
            break
        else:
            if(k >= min_slope):
                selected_best_ensembles.append(ind)

    return selected_best_ensembles

def simple_average_ensemble(ensembles,all_datas,toolbox):
    complie_list = []
    final_output = []
    for ind in ensembles:
        ind = toolbox.compile(ind)
        complie_list.append(ind)
    for data in all_datas:
        temp = []
        for individual in complie_list:
            k = individual(data[:-1])
            temp.append(k)
        z = sum(temp)/len(temp)
        final_output.append(z)
    return final_output
def my_class(individual,data):
    if individual(data[:-1]) >= 0:
        return 1.0
    else:
        return 0.0

def vote_ensemble(ensembles,all_datas,toolbox):
    final_list = []
    complie_list = []
    for ind in ensembles:
        ind = toolbox.compile(ind)
        complie_list.append(ind)
    for data in all_datas:
        temp = []
        for individual in complie_list:
            k = my_class(individual,data)
            temp.append(k)
        labels = set(temp)
        # print(labels)
        z = []
        for label in labels:
            a = temp.count(label)
            print(a)
            z.append(a)
        z.sort(reverse=True)
        final_list.append(z[0])
    print(final_list)
    print(len(final_list))
    return final_list

def select_mininze_feature(pop,pset):
    feature_list = []
    for ind in pop:
        k = len(count_selected_feat(ind,pset))
        leaf_num = count_leaf_nodes(ind)
        feature_list.append((k,leaf_num))
    min1 = min(feature_list,key=lambda v: (v[0], v[1]))
    index = feature_list.index(min1)
    return pop[index]

def find_unrepeat_ensembles(pareto_first_front,pset):
    final_list = []
    fitness_list = []
    for ind in pareto_first_front:
        fitness_list.append(ind.fitness.values)
    pre_point = 0
    temp = []
    for index, ind in enumerate(fitness_list):
        # print(ind)
        if index != 0:
            if temp[-1] == ind:
                temp.append(ind)
            else:
                new_list = pareto_first_front[pre_point:index]
                k = select_mininze_feature(new_list,pset)
                pre_point = index
                final_list.append(k)
                temp.append(ind)
        else:
            temp.append(ind)
    #处理最后一段
    new_list = pareto_first_front[pre_point:]
    k = select_mininze_feature(new_list, pset)
    final_list.append(k)

    return final_list


if __name__ == '__main__':
    a = (10, 5)
    k = 1
    b = 1
    z = point_line_distance(k,b,a)

    print(z)




