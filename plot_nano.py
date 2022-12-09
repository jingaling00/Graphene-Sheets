# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 16:24:59 2022

@author: jingy
"""

import numpy as np
import matplotlib.pyplot as plt

def find_connectivity(points, low_bound, up_bound):
    connect_vals = []
    for i in range(points.shape[0]):
        for j in range(points.shape[0]):
            value = ((points[i][0]-points[j][0])**2) + ((points[i][1]-points[j][1])**2)
            if value >= low_bound and value <= up_bound:
                connect_vals.append(value)
            else:
                connect_vals.append(0)
                
    connect_vals = np.array(connect_vals)
    connect_vals = np.reshape(connect_vals,(points.shape[0],points.shape[0]))
    return connect_vals

def get_connectivity_ind(points, low_bound, up_bound):
    conn_values = find_connectivity(points, low_bound, up_bound)
    non_zero_values = np.where(conn_values != 0)
    x = [p for p in non_zero_values[0]]
    y = [p for p in non_zero_values[1]]
    return x,y

def plot_carbon_carbon(p1, p2):
    x = [p1[0], p2[0]]
    y = [p1[1], p2[1]]
    plt.plot(x, y,'b')
    plt.plot(x, y, marker='o',markerfacecolor='r',markersize=19,alpha=0.3)
    

def plot_mesh(xx, yy, points):
    for i in range(len(xx)):
        plot_carbon_carbon(points[xx[i]],points[yy[i]])
    plt.show()

if __name__ == '__main__':
    a = np.array([[0. , 0. ],
                  [2.598, 1.5 ],
                    [1.732, 1. ],
                    [0.866, 1.5 ],
                    [0. , 1. ],
                    [5.196, 3. ],
                    [4.33 , 2.5 ],
                    [3.464, 3. ],
                    [2.598, 2.5 ]])
    low_th = 0.95
    up_th = 1.05
    xx,yy = get_connectivity_ind(a,low_th,up_th)
    plot_mesh(xx,yy,a)