import random
import operator

import itertools

import numpy as np

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp



import matplotlib.pyplot as plt
import time

from func_tools import *
from eval_func import *





def single_objective(method,all_datas1,majdatas1,mindatas1,new_mindatas1,new_majdatas1, feat_num,minnum,majnum,all_datas2):
    N = [minnum,majnum]

    # for i in range(len(data)):
    #     print("%s" % i, str(data[i]).rjust(20550, "-"))

    # defined a new primitive set for strongly typed GP
    #  创建一个迭代器，它返回指定次数的对象。如果未指定，则无限返回对象。
    pset = gp.PrimitiveSetTyped("MAIN", itertools.repeat(float, feat_num), float, "f")
    def analytical_quotient(x, y):
        return x / math.sqrt(y*y+1)

    def Div(left, right):
        try:
            return left / right
        except ZeroDivisionError:
            return 1

    pset.addPrimitive(operator.add, [float, float], float)
    pset.addPrimitive(operator.sub, [float, float], float)
    pset.addPrimitive(operator.mul, [float, float], float)
    pset.addPrimitive(Div, [float, float], float)


    creator.create("Fitnessmax", base.Fitness, weights=(1.0, )) 
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.Fitnessmax)

    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=2, max_=6)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)
    if method == 'dist':
        toolbox.register("evaluate", dist, toolbox = toolbox,Cmin=mindatas1, Cmaj=majdatas1)
    elif method == 'aucw':
        toolbox.register("evaluate", aucw, toolbox = toolbox,majdatas = majdatas1,mindatas= mindatas1)
    elif method == 'amse':
        toolbox.register("evaluate", amse, toolbox = toolbox,Cmin=mindatas1, Cmaj=majdatas1)
    elif method == 'corr':
        toolbox.register("evaluate", corr, toolbox = toolbox,majdatas = majdatas1,mindatas = mindatas1)
    elif method == 'ave':
        toolbox.register("evaluate", ave,w =0.5, toolbox = toolbox,Cmin=mindatas1, Cmaj=majdatas1,min_num= minnum, maj_num= majnum)
    elif method == 'G_mean':
        toolbox.register("evaluate", G_mean, toolbox = toolbox,Cmin=mindatas1, Cmaj=majdatas1,min_num= minnum, maj_num= majnum)
    elif method == 'realauc':
        toolbox.register("evaluate", aucc, toolbox = toolbox,Cmin = mindatas1, Cmaj=majdatas1, N = N)
    elif method == 'auc_dist':
        toolbox.register("evaluate", auc_dist, toolbox = toolbox,majdatas = majdatas1,mindatas= mindatas1)
    elif method == 'prauc':
        toolbox.register("evaluate", prauc, toolbox = toolbox,data_training = all_datas1)
    elif method == 'gpmo2':
        toolbox.register("evaluate", gpmo2, a = 0.95, feat_num = feat_num, toolbox = toolbox, Cmin = mindatas1, Cmaj = majdatas1, min_num = minnum, maj_num = majnum)
    elif method == 'muni':
        toolbox.register("evaluate", muni, toolbox = toolbox, a = 0.1, Cmin = mindatas1, Cmaj = majdatas1, n = feat_num)
    elif method == 'bojar':
        toolbox.register("evaluate", bojar, toolbox = toolbox, k = 23, Cmin = mindatas1, Cmaj = majdatas1, min_num = minnum, maj_num = majnum)
    elif method == 'm_dist':
        toolbox.register("evaluate", m_dist, toolbox = toolbox,Cmaj = majdatas1,Cmin= mindatas1)
    elif method == 'no_zero_aucw':
        toolbox.register("evaluate", no_zero_aucw, toolbox = toolbox,majdatas = majdatas1,mindatas= mindatas1)
    elif method == 'gpd':
        toolbox.register("evaluate", d_a, toolbox = toolbox,Cmin = mindatas1,Cmaj = majdatas1,new_datamin = new_mindatas1,new_datamaj = new_majdatas1)


    
    


    else:
        print('wrong method')





    toolbox.register('select', tools.selTournament, tournsize=6)
    # toolbox.register("select", tools.selNSGA2)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genFull, min_=0, max_=3)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

    # 装饰器  使用指定的装饰器装饰别名，别名必须是当前工具箱中的已注册函数。
    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

    # random.seed(10)
    N_POP = 500
    N_GEN = 50
    CXPB = 0.8  # 交叉概率，参数过小，族群不能有效更新
    MUTPB = 0.2  # 突变概率，参数过小，容易陷入局部最优

    pop = toolbox.population(n=N_POP)
    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)


    mstats.register("avg", np.average)
    # mstats.register("std", np.std)
    mstats.register("max", np.max)
    # mstats.register("median", np.median)
    mstats.register("min", np.min)
    hof = tools.HallOfFame(1)  # 名人堂



    pop,log= algorithms.eaSimple(pop, toolbox, CXPB, MUTPB, N_GEN, stats=mstats, halloffame=hof, verbose=True)

    return pop,hof,log, toolbox,tools,pset



