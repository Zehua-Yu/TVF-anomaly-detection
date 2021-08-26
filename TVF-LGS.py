# Coding: UTF-8
# Author: Zehua Yu
# Date: 2021/8/24 19:28
# IDE: PyCharm

import pywt
from operator import itemgetter
from statsmodels.tsa.arima_model import ARMA
import pandas as pd
import csv
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
import numpy as np
import SGWT
import time as ti



def DWT(data, time_len, vertices_num, bd):
    plot_series = []
    max_value_p_all = []
    # DWT for each time-series, then, collect the max dwt coefficient by vertex number and time slot.
    for i in range(vertices_num):
        t_s = []
        for j in range(time_len):
            t_s.append(df1.iloc[j, i])
        y = np.abs(t_s)
        x = range(len(y))
        ca, cd = pywt.dwt(y, 'db4')
        ya = pywt.idwt(ca, None, 'db4')  # approximated component
        yd = pywt.idwt(None, cd, 'db4')  # detailed component
        max_value1 = max(yd)
        p_max_value1 = np.where(yd == max_value1)
        if len(p_max_value1[0]) > 1 and max_value1 >= bd:
            for k in range(len(p_max_value1[0])):
                max_value_p_single = (i + 1, int(p_max_value1[0][k]), max_value1)
                max_value_p_all.append(max_value_p_single)
            print(i + 1, 'is multi time.')
        elif max_value1 < bd:
            print('pass')
        else:
            max_value_p_single = (i + 1, int(p_max_value1[0]), max_value1)
            max_value_p_all.append(max_value_p_single)
            print(i + 1, 'is single time.')
    max_value_p_all_sorted = sorted(max_value_p_all, key=itemgetter(1))
    r = []
    for i in range(np.shape(max_value_p_all_sorted)[0]):
        r.append(max_value_p_all_sorted[i][1])
    return r

def splitting(data, sp_list, name):
    df_in = data
    a = 0
    for i in range(len(sp_list)):
        for j in range(sp_list[i]):
            if i == 0:
                with open(f'{name}_{i + 1}.csv', "a", newline='') as csv_file:
                    writer = csv.writer(csv_file)
                    writer.writerow(df_in.iloc[j, :])
                    csv_file.close()
            else:
                a = sp_list[i - 1]
                if j >= a:
                    with open(f'{name}_{i + 1}.csv', "a", newline='') as csv_file:
                        writer = csv.writer(csv_file)
                        writer.writerow(df_in.iloc[j, :])
                        csv_file.close()
                else:
                    print('pass.')
        print(i, 'is done.')
    for k in range(len(df_in.iloc[:, 0])):
        a = sp_list[len(sp_list)-1]
        if k >= a:
            with open(f'{name}_{len(sp_list)+1}.csv', "a", newline='') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(df_in.iloc[k, :])
                csv_file.close()

    print('Splitting is done!')

def Generation_adj(name, index, p, q):

    dist_name = f'{name}armadist_{index}.csv'
    arima_vec = f'{name}arma_{index}.csv'
    od_name = f'{name}_{index}.csv'
    df1 = pd.read_csv(od_name, header=None)
    nodes = len(df1.iloc[1, :])
    B = []
    for i in range(len(df1.iloc[0, :])):
        timeseries = []
        for j in range(len(df1.iloc[:, 0])):
            timeseries.append(df1.iloc[j, i])
        print(timeseries)
        if np.mean(timeseries) == 0:
            B.append(0)
        else:
            model = ARMA(timeseries, order=(p, q)).fit()
            data_h = model.summary()
            print(data_h)
            B = []
            for k in range(p + q):
                B.append(data_h[k])
        with open(arima_vec, "a", newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(B)
            csv_file.close()
            B = []
        print(i + 1)

    df2 = pd.read_csv(arima_vec, header=None)
    vec = []
    A = []
    for a in range(nodes):
        A.append(complex(df2.iloc[a, 0]))
        vec.append(1)
        vec[a] = A
        A = []
    distA = pdist(vec, metric='euclidean')
    distB = squareform(distA)
    for c in range(nodes):
        A = []
        for d in range(nodes):
            A.append(distB[c, d])
        with open(dist_name, "a", newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(A)
            csv_file.close()

if __name__ == '__main__':

    df1 = pd.read_csv('tf_20200106_71.csv', header=None)
    starttime = ti.time()
    data_name = 'tf71'
    time_len = len(df1.iloc[:, 0])
    vertices_num = len(df1.iloc[0, :])
    bd = 210
    s_p = DWT(df1, time_len, vertices_num, bd)
    print(s_p)
    splitting(df1, s_p, data_name)
    for i in range(len(s_p)+1):
        Generation_adj(data_name, i+1, 5, 0)
        print(i+1, 'of ADJ is done.')
    for j in range(len(s_p)+1):
        A = pd.read_csv(f'{data_name}armadist_{j+1}.csv', header=None)
        N = len(A.iloc[0, :])
        jcenter = 71
        d = SGWT.delta(N, jcenter)
        L = SGWT.lil_matrix(A)
        lmax = SGWT.rough_l_max(L)
        lmax = 4.039353399475705
        Nscales = 3
        (g, gp, t) = SGWT.filter_design(lmax, Nscales)
        m = 50
        arange = (0, lmax)
        c = [SGWT.cheby_coeff(g[i], m, m + 1, arange) for i in range(len(g))]
        wpall = SGWT.cheby_op(d, L, c, arange)
        print(np.size(wpall), wpall)
        endtime = ti.time()
        print(endtime - starttime, 's')
