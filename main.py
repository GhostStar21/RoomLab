"""
Assignment 1
BMA1020 VÅR 2025

This program involves a room made through triangles (elements involving 3 nodes 
each). The goal of the program is to move and rotate the room in the x, y and z 
direction.  

@file: Assignment1.py
@author: Ekansh Misra
"""

import pyglet
import numpy as np
from quaternions import Quaternion as q

def project(x1, y1, z1, x2, y2, z2, x3, y3, z3, 
            f, color, batch, width, height) -> pyglet.shapes.Triangle:
    """
    Takes in the x, y and z co-ordinates of each element and projects it to 
    the window. 
    """
    fxy = [0] * 6
                
    # project 3D coordinates to 2D screen coordinates using focal length formula 
    fxy[0] = ((f * x1) / z1) + width / 2  # fx1
    fxy[1] = ((f * y1) / z1) + height / 2  # fy1
                
    fxy[2] = ((f * x2) / z2) + width / 2  # fx2
    fxy[3] = ((f * y2) / z2) + height / 2  # fy2
    
    fxy[4] = ((f * x3) / z3) + width / 2  # fx3
    fxy[5] = ((f * y3) / z3) + height / 2  # fy3

    # create a triangle at the projected coordinates with appropriate color
    return pyglet.shapes.Triangle(
        fxy[0], fxy[1], fxy[2], fxy[3], fxy[4], fxy[5],
        color=color,
        batch=batch
    )

class Model_3d:
    def __init__(self, nodes, elements, colors):
        self.nodes = nodes
        self.elements = elements
        self.colors = colors
        self.batch = pyglet.graphics.Batch()
        self.triangles = {}
    # ***************************************************
    def add_nodes(self, nodes): 
        self.nodes.append(nodes)
    def delete_node(self, j): 
        self.nodes.pop(j)
    def add_element(self, element, color, back=True): 
        self.elements.append(element)
        self.colors.append(color)
    def delete_element(self, i): 
        self.elements.pop(i)
        self.colors.pop(i)        
    def rotate(self, quaternion): 
        self.nodes = quaternion.rotate_vector(self.nodes)
    def translate(self, vector):
        for i in range(len(self.nodes)):    
            #adds the vector to the nodes to move it
            self.nodes[i] = self.nodes[i] + vector  
    
    def draw_element(self, element_i, focal_length, width, height):
        # get indices of vertices for an element
        a, b, c = self.elements[element_i]
        
        # get coordinates of vertices
        x1, y1, z1 = self.nodes[a]
        x2, y2, z2 = self.nodes[b]
        x3, y3, z3 = self.nodes[c]
        
        # check if any vertex is behind the camera
        if np.any(np.array([z1, z2, z3]) <= 0):
            divided_elements = self.divide_element(element_i)
            # if there are 2 points behind the camera
            if len(divided_elements) == 3:
                p1, p2, p3 = divided_elements   
                #projects the divided elements
                self.triangles[f"divided_{element_i}_1"] = project(     
                    p1[0], p1[1], p1[2], 
                    p2[0], p2[1], p2[2], 
                    p3[0], p3[1], p3[2], 
                    focal_length, 
                    self.colors[element_i], 
                    self.batch, 
                    width, height
                    )
            # if there is 1 point behind the camera
            elif len(divided_elements) == 4:
                p1, p2, p3, p4 = divided_elements
                #projects the first and second triangle
                self.triangles[f"divided_{element_i}_1"] = project(
                    p1[0], p1[1], p1[2], 
                    p2[0], p2[1], p2[2], 
                    p3[0], p3[1], p3[2], 
                    focal_length, 
                    self.colors[element_i], 
                    self.batch, 
                    width, height
                )
                self.triangles[f"divided_{element_i}_2"] = project(
                    p3[0], p3[1], p3[2], 
                    p4[0], p4[1], p4[2], 
                    p1[0], p1[1], p1[2], 
                    focal_length, 
                    self.colors[element_i], 
                    self.batch, 
                    width, height
                )
        else:
            # if all the z-coordinates are in front of the camera, draw normally
            self.triangles[element_i] = project(
                x1, y1, z1, 
                x2, y2, z2, 
                x3, y3, z3, 
                focal_length, 
                self.colors[element_i], 
                self.batch, 
                width, height
            )
    def divide_element(self, element_i):
        # get vertex indices for this element
        a, b, c = self.elements[element_i]
        
        # get the coordinates for these vertices
        p1 = np.array(self.nodes[a])
        p2 = np.array(self.nodes[b])
        p3 = np.array(self.nodes[c])

        # check which vertices are behind the camera
        points = np.array([p1, p2, p3])
        z_values = points[:, 2]
        
        # get indices of points behind/in front of camera
        behind_indices = np.where(z_values <= 0.0)[0]
        front_indices = np.where(z_values > 0.0)[0]
        
        # if all vertices are behind the camera, don't draw anything
        if len(front_indices) == 0:
            return []
        
        # if all vertices are in front of the camera, return the original points
        if len(behind_indices) == 0:
            return [element_i]
        
        # if 1 vertex is behind the camera and 2 are in front of it
        if len(behind_indices) == 1:
            # Get the indices of one vertex behind and the two in front
            #could have done it in one line but this is easier to understand
            behind_index = behind_indices[0]
            front_index1 = front_indices[0]
            front_index2 = front_indices[1]
            
            # Get the actual points
            p_behind = points[behind_index]
            p_front1 = points[front_index1]
            p_front2 = points[front_index2]

        #got assistance from TA and ChatGPT for the linear interpolation formula
        # formula: p_intersect = p_front + t * (p_behind - p_front) 
            h1 = (p_front1[2] - p_behind[2])
            if h1 == 0.0:
                h1 = 0.1
            h2 = (p_front2[2] - p_behind[2])
            if h2 == 0.0:
                h2 = 0.1

            t1 = p_front1[2] / h1
            t2 = p_front2[2] / h2
            
            # calculate the new points 
            i1 = p_front1 + (t1 * (p_behind - p_front1))
            i2 = p_front2 + (t2 * (p_behind - p_front2))
            
            # Set z-coordinate to a small positive value to avoid division by zero
            i1[2] = 0.1
            i2[2] = 0.1
            
            #returns 4 points, since there are four edges due to one node being behind the camera
            return [i1, p_front1, p_front2, i2]
            
        # if 2 vertices are behind the camera and 1 is in front of it
        if len(behind_indices) == 2:
            # get the index of the one vertex in front and the two behind
            front_index = front_indices[0]
            behind_index1 = behind_indices[0] 
            behind_index2 = behind_indices[1]
            
            # Get the actual points
            p_front = points[front_index]
            p_behind1 = points[behind_index1]
            p_behind2 = points[behind_index2]
            
            # calculate intersection points with z=0 plane
            h1 = (p_front[2] - p_behind1[2])
            if h1 == 0:
                h1 = 0.1
            h2 = (p_front[2] - p_behind2[2])
            if h2 == 0:
                h2 = 0.1
            t1 = p_front[2] / h1
            t2 = p_front[2] / h2
            
            # calculate the new points
            i1 = p_front + (t1 * (p_behind1 - p_front))
            i2 = p_front + (t2 * (p_behind2 - p_front))
            
            # set z-coordinate to a small positive value to avoid division by zero
            i1[2] = 0.1
            i2[2] = 0.1
            
            return [p_front, i1, i2]


    def draw(self, focal_length, window_width, window_height):
        # remove the previous traingles
        self.clean_triangles()
        # goes through for loop to draw each element
        for i in range(len(self.elements)):
            self.draw_element(i, focal_length, window_width, window_height)
    def clean_triangles(self):
        for triangle in self.triangles.values():
            triangle.delete()
    # clear the dictionary
        self.triangles.clear()

#Nodes, restructured with help from TA (Khai)
Nodes = 100*np.array([
    [-1., -1.,  -1.],    # 0
    [ 1., -1.,  -1.],    # 1
    [ 1.,  1.,  -1.],    # 2
    [ -1., 1.,  -1.],    # 3
    [ 1., -1.,  1.],     # 4
    [ -1., -1., 1.],     # 5
    [-1.,  1.,  1.],     # 6
    [ 1.,  1.,  1.]      # 7
])

#Elements, restructured with help from TA (Khai)
Elements = np.array([
                    [0,1,2],        #front      #0
                    [2,3,0],        #front      #1
                    [4,5,6],        #rear       #2
                    [6,7,4],        #rear       #3
                    [5,0,3],        #left rear  #4
                    [3,6,5],        #left rear  #5
                    [1,4,7],        #right      #6
                    [7,2,1],        #right      #7  
                    [3,2,7],        #top        #8
                    [7,6,3],        #top        #9  
                    [0,1,4],        #bottom     #10 
                    [4,5,0]         #bottom     #11
])

R = [255,0,0]
G = [0,255,0]
B = [0,0,255]
M = [255,0,255]
Y = [255,255,0]
C = [0,255,255]

Colors = np.array([R,R,G,G,B,B,M,M,Y,Y,C,C],dtype=np.uint32)

# Build the model
model2 = Model_3d(Nodes,Elements,Colors)

# Rotations about different axes
# You can choose different rotation increments if you want
angle = np.radians(15) 
qx = q.from_rotation([1, 0, 0], angle)
qy = q.from_rotation([0, 1, 0], angle)
qz = q.from_rotation([0, 0, 1], angle)

# Insert focal length here
f = 100
width = 1000
height = 1000
# Draw to the screen
model2.draw(f, width, height)

window = pyglet.window.Window(width= width,height=height,resizable=True)

@window.event
def on_draw():
    window.clear()
    model2.draw(f, width, height) 
    model2.batch.draw()
    
@window.event 
def on_key_press(symbol, modifiers): # 10 points
    if symbol == pyglet.window.key.D:   #go right
        model2.translate([50, 0, 0])
    if symbol == pyglet.window.key.A:   #go left
        model2.translate([-50, 0, 0])
    if symbol == pyglet.window.key.W:   #go up
        model2.translate([0, 50, 0])
    if symbol == pyglet.window.key.S:   #go down
        model2.translate([0, -50, 0])
    if symbol == pyglet.window.key.E:   #go in
        model2.translate([0, 0, 10])
    if symbol == pyglet.window.key.Q:   #go out
        model2.translate([0, 0, -10])
    if symbol == pyglet.window.key.R:   #rotate in the clockwise x direction
        model2.rotate(qx)
    if symbol == pyglet.window.key.T:   #rotate in the anticlockwise x direction
        model2.rotate(qx.inverse())    
    if symbol == pyglet.window.key.F:   #rotate in the clockwise y direction
        model2.rotate(qy)
    if symbol == pyglet.window.key.G:   #rotate in the anticlockwise y direction
        model2.rotate(qy.inverse())
    if symbol == pyglet.window.key.C:   #rotate in the clockwise z direction
        model2.rotate(qz)
    if symbol == pyglet.window.key.V:   #rotate in the anticlockwise z direction
        model2.rotate(qz.inverse())

pyglet.app.run()