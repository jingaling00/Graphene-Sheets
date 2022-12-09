# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 16:24:59 2022

@author: jingy
"""

import numpy as np
import matplotlib.pyplot as plt

def find_connectivity(points, low_bound, up_bound):
    """

    Parameters
    ----------
    points : Numpy array of points on Cartesian plane.
    low_bound : Lower bound for distance between carbons.
    up_bound : Upper bound for distance between carbons.
    
    Returns
    -------
    connect_vals : Square array of distance values between points 
    that satisfy bounnds conditions. 
    0 in the ith, jth position indicates points i and j are not connected.

    """
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
    """
    Parameters
    ----------
    points, low_bound, up_bound to call find_connectivity function. 

    Returns
    -------
    Two one-dimensional arrays of indices corresponding to 
    points that are connected in the graphene sheet.

    """
    conn_values = np.triu(find_connectivity(points, low_bound, up_bound))
    non_zero_values = np.where(conn_values != 0)
    x = [p for p in non_zero_values[0]]
    y = [p for p in non_zero_values[1]]
    return [x,y]

def plot_carbon_carbon(p1, p2):
    """
    Parameters
    ----------
    p1 : a point in the Cartesian plane.
    p2 : a point in the Cartesian plane.
    
    Returns
    -------
    Plots connection between p1 and p2.

    """
    x = [p1[0], p2[0]]
    y = [p1[1], p2[1]]
    plt.plot(x, y,'b')
    plt.plot(x, y, marker='o',markerfacecolor='r',markersize=19,alpha=0.3)
    
def plot_mesh(xx, yy, points):
    """
    Parameters
    ----------
    xx and yy are one dimensional arrays of indices whose ith elements
    correspond to connected points in the graphene sheets.
    points : Numpy array of all points in the sheet.
    
    Returns
    -------
    Plots the entire graphene sheet.

    """
    for i in range(len(xx)):
        plot_carbon_carbon(points[xx[i]],points[yy[i]])
    plt.show()   
