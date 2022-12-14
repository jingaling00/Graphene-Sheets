# -*- coding: utf-8 -*-
"""
Created on Sat Nov 26 08:43:25 2022

"""

import math
from plot_nano import get_connectivity_ind, plot_mesh
from misc import include_atoms_gr,  grid_pq
import numpy as np

class CarbonFiller:
    """
    This is a class to construct carbon nanofillers, e.g. Graphene sheets and/
    or carbon nanotubes.
    """
    
    def __init__(self, n, m, l):
        """
        Initializes attributes of carbon filler.

        Parameters
        ----------
        n : number of jumps along the a1 vector (sqrt 3, 0).
        m : number of jumps along the a2 vector ((sqrt 3)/2, -1.5)
        l : length of normal vector

        Returns
        -------
        None.

        """
        self.n = n
        self.m = m
        self.l = l
        self.name = 'Carbon Filler'
        
    def vector(self):
        """
        Returns
        -------
        Calculates horizontal vector of graphene sheet based on attributes.

        """
        a1 = np.array([math.sqrt(3),0])
        a2 = np.array([((math.sqrt(3))/2),-1.5])
        a1 *= self.n
        a2 *= self.m
        Ch_vector = a1+a2
        return np.around(Ch_vector, decimals=3)
    
    def TVector(self, Ch_vector):      
        """
        Parameters
        ----------
        Ch_vector : Must input horizontal vector of graphene sheet.
        
        Returns
        -------
        The normal vector to Ch_vector; the vertical vector of graphene sheet.
        """
        T_vec = CarbonFiller.normTvector(Ch_vector)
        normT_vec = CarbonFiller.normVector(T_vec)
        return self.l * normT_vec[0]
    
    @staticmethod
    def normVector(vector):
        """
        Parameters
        ----------
        vector : Arbitrary vector.

        Returns
        -------
        norm_vec : Input vector normalized.
        norm : Magnitude of input vector.

        """
        norm = np.around(np.linalg.norm(vector), decimals=3)
        norm_vec = np.around(vector / norm, decimals=3)
        return norm_vec, norm
    
    @staticmethod
    def normTvector(c_hat):
        """
        Parameters
        ----------
        c_hat : Arbitrary vector.

        Returns
        -------
        t_hat : Normal vector to c_hat.

        """
        t_hat = np.array([-c_hat[1], c_hat[0]])
        return t_hat
        
    @staticmethod
    def pq(Ch, T):
        """
        Parameters
        ----------
        Ch : Horizontal vector of graphene sheet.
        T : Vertical vector of graphene sheet.

        Returns
        -------
        Two sets of bounds for locus of points included in the graphene sheet.

        """
        p_min = 0
        q_min = math.floor(Ch[1]*(2/3))
        p_max = math.ceil(((Ch[0] + T[0])*(2/math.sqrt(3))))
        q_max = math.ceil(T[1]*(2/3))
        return [np.array([p_min,p_max]),np.array([q_min,q_max])]
    
    @staticmethod
    def coordinates(pg, qg):
        """

        Parameters
        ----------
        pg : Horizontal bounds for carbon vertices.
        qg : Vertical bounds for carbon vertices.

        Returns
        -------
        pg : Transforms horizontal bounds to points on the Cartesian plane.
        qg : Transforms vertical bounds to points on the Cartesian plane.

        """
        pg = np.array([np.array(p) for p in pg],dtype=float)
        qg = np.array([np.array(q) for q in qg],dtype=float)
        for row in pg:
            row *= np.around(math.sqrt(3)/2,decimals=3)
        for row in qg:
            for i in range(len(row)):
                if row[i] % 2 == 0 and i % 2 == 0:
                    row[i] *= 1.5
                elif row[i] % 2 == 0 and i % 2 == 1:
                    row[i] = row[i] * 1.5 - 0.5
                elif row[i] % 2 == 1 and i % 2 == 0:
                    row[i] = row[i] * 1.5 - 0.5
                elif row[i] % 2 == 1 and i % 2 == 1:
                    row[i] *= 1.5
        return pg, qg
    
    @staticmethod
    def distance(x, y, c_hat):
        """
        Parameters
        ----------
        x : All x-coordinates of points in graphene sheet.
        y : All y-coordinates of points in graphene sheet.
        c_hat : Horizontal vector of graphene sheet. 

        Returns
        -------
        s : Array of dot product values of every point and c_hat.
        t : Array of dot product values of every point and t_hat (normal to c_hat).

        """
        s = np.array([])
        for i in range(x.shape[0]):
            vals = []
            for j in range(len(x[i])):
                value = round((x[i][j] * c_hat[0]) + (y[i][j] * c_hat[1]),3)
                vals.append(value)
            s = np.concatenate((s, vals), axis = 0)
        s = np.reshape(s,(x.shape[0],x.shape[1]))
        
        t = np.array([])
        for i in range(x.shape[0]):
            vals = []
            for j in range(len(x[i])):
                value = round((x[i][j] * -c_hat[1]) + (y[i][j] * c_hat[0]),3)
                vals.append(value)
            t = np.concatenate((t, vals), axis = 0)
        t = np.reshape(t,(x.shape[0],x.shape[1]))
        
        return s, t
    
    def include_atoms_gr(x, y, s, t, arclen, l):
        """
        A function used to find which atoms to include in the graphene sheet.
        Input:
            x: coordinates of atoms in the x-axis
            y: corrdinates of atoms in the y-axis
            s: distance along the C-direction.
            t: distance along the T-direction.
            arclen: Length of the Ch vector.
            l: length of the perpendicular edge T.
        Output:
            pos: A numpy array containing the coordinates of the atoms
                   included in the graphene sheet, (pos_nt.shape = (N, 2)).
            
        """
        if isinstance(x, list):
            x = np.array(x).T
        if isinstance(y, list):
            y = np.array(y).T
        if isinstance(s, list):
            s = np.array(s).T
        if isinstance(t, list):
            t = np.array(t).T
        
        tol=0.1;
        cond1 = s+tol>0
        cond2 = s-tol<arclen
        cond3 = t+tol>0
        cond4 = t-tol<l
    
        include = np.ones(shape=(s.shape[0], s.shape[1]), dtype=bool)
        for i in range(s.shape[0]):
            for j in range(s.shape[1]):
                if (cond1[i, j] == False) or (cond2[i, j] == False) or (cond3[i, j] == False) or (cond4[i, j] == False):
                    include[i, j] = False
    
        return np.hstack([x[include].reshape(-1, 1), y[include].reshape(-1, 1)])
    
    def include_atoms_nt(pos, c_hat, arclen, tubrad):
        """
        A function used to find which atoms to include in the nanotube
        Input:
            pos: A numpy array containing the coordinates of the atoms, calculated
                 with the Graphene(n, m, l) function (pos.shape = (N, 2)).
            c_hat: Numpy array  holding the coordinates of normalized Ch vector.
            arclen: Norm of vector Ch (type: float).
            tubrad: Radius of the nanotube.
        Output:
              pos_nt: A numpy array containing the coordinates of the atoms
              included in the nanotube, (pos_nt.shape = (N, 3)).
        """
        
        tol=0.1;
        s = c_hat[0]*pos[:,0] + c_hat[1]*pos[:,1]
        t = -c_hat[1]*pos[:,0] + c_hat[0]*pos[:,1]
        
        tol=0.1;
        cond1 = s+tol>0
        cond2 = s+tol<arclen
        
        include = np.full((s.shape[0]), True, dtype=bool)
        
        for i in range(s.shape[0]):
            if (cond1[i] == False) or (cond2[i] == False):
                include[i] = False
        
        pos_ = [tubrad*np.cos(s[include]/tubrad), tubrad*np.sin(s[include]/tubrad),
                t[include]]
        pos_nt = np.vstack((pos_[0],pos_[1],pos_[2])).T
        
        return pos_nt
    
def Graphene(n, m ,l):
    """
    Parameters
    ----------
    Uses parameters n, m, l to create object of CarbonFiller class.

    Returns
    -------
    pos_gr : All points of graphene sheet.

    """
    Cf = CarbonFiller(n, m, l)
    Ch = Cf.vector()
    T = Cf.TVector(Ch)
    p, q = Cf.pq(Ch, T)
    Pgrid, Qgrid = grid_pq(p, q)
    x, y = Cf.coordinates(Pgrid, Qgrid)
    c_hat, arclen = Cf.normVector(Ch)
    s, t = Cf.distance(x, y, c_hat)
    pos_gr = include_atoms_gr(x, y, s, t, arclen, l)
    return pos_gr
