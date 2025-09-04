"""
Oblig 4 - Quaternions
BMA1020 VÅR 2025

The program includes functions that can be used to add, multiply of a quaternion.
In addition, it includes finding the inverse, rotation and absolute value of 
a quaternion. 

@file: quaternions.py
@author: Ekansh Misra
"""

import numpy as np

class Quaternion:
    def __init__(self, scalar, vec):
        if (len(vec) != 3) or not(isinstance(1.0*scalar, float)):
            raise ValueError('Cannot initialize quaternion with fewer or more than four parameters')
        self.scalar = scalar
        self.vec = np.array(vec)
    def __repr__(self):
        return f'quaternion {self.scalar} + i*{self.vec[0]} + j*{self.vec[1]} + k*{self.vec[2]}'
    def __add__(self, other):
        # adding two Quaternion instances "self" and "other"
        # ***********************************************
        return Quaternion(self.scalar + other.scalar, self.vec + other.vec) # add/adapt code
        # ***********************************************
    def __mul__(self, other):
        # multiplication of two quaterions, note that we do not implement
        # the product scalar * Quaternion (which could be done as well)
        # ***********************************************
       return Quaternion(self.scalar* other.scalar - np.dot(self.vec, other.vec), 
                         self.scalar * other.vec + other.scalar * self.vec + 
                         np.cross(self.vec, other.vec))
        # ***********************************************        
    def __abs__(self):
        # return the length of a quaternion
        # ***********************************************
        return (np.sqrt(self.scalar**2 + np.dot(self.vec, self.vec)))
        # ***********************************************        
    def inverse(self):
        # return the inverse quaternion; there is no need to exclude (0,[0,0,0])
        # for this task but in a bigger project, that would be good practice
        # ***********************************************
        scalarA = self.scalar/(self.__abs__() * self.__abs__())
        vectorA = -(self.vec)/(self.__abs__() * self.__abs__())
        return Quaternion(scalarA, vectorA)
        # ***********************************************        
    @classmethod
    def from_rotation(cls, vector, angle):
        # This is an alternative instantiator for Quaternions: From a given 
        # vector and angle the quaternion is created that rotates around the
        # vector by that angle
        return cls(np.cos(angle/2), np.sin(angle/2)*np.array(vector))
    def rotate_vector(self, vector):
        # ***********************************************
        vector = vector.T
        B = np.ndarray((len(vector[0]),3))      
        for i in range(len(vector[0])):  #goes through the array and rotates it
            B[i] = (self * Quaternion(0, vector[:,i]) * self.inverse()).vec
        return B