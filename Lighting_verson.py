"""
Assignment 2
BMA1020 VÅR 2025

This program involves a room made through triangles (elements involving 3 nodes 
each) where light is reflected.

@file: Assignment2.py
@author: Ekansh Misra
"""

import pyglet
import numpy as np
from quaternions import Quaternion as q



A = np.loadtxt('LMS.csv', delimiter=';')
# First, interpolate to get values of L,M,S(lambda)
# Use Hermite interpolation
# hermite(s,t,x,v) interpolates data x(t), v(t) at point s

def hermite2(t, x):

    v = np.zeros_like(x)
    for i in range(len(t)-1):
        v[i] = (x[i+1] - x[i]) / (t[i+1] - t[i])
    # The final derivative is set to zero
    v[-1] = 0
    
    # Return a function that implements Hermite interpolation
    def interpolate(s):
        # Find the appropriate interval for interpolation
        if s <= t[0]:
            return x[0]
        if s >= t[-1]:
            return x[-1]
            
        # Find the interval [t[i], t[i+1]] containing s
        i = 0
        while i < len(t)-1 and s > t[i+1]:
            i += 1
        
        # Create a simple cubic Hermite interpolation manually
        # since the imported hermite function is causing issues
        h = t[i+1] - t[i]
        s_norm = (s - t[i]) / h
        
        h00 = 2*s_norm**3 - 3*s_norm**2 + 1
        h10 = s_norm**3 - 2*s_norm**2 + s_norm
        h01 = -2*s_norm**3 + 3*s_norm**2
        h11 = s_norm**3 - s_norm**2
        
        return h00*x[i] + h10*h*v[i] + h01*x[i+1] + h11*h*v[i+1]
    
    return interpolate

# the functions L, M, S
# normalized to max value 1
gl = hermite2(A[:,0], A[:,1])
gm = hermite2(A[:,0], A[:,2])
gs = hermite2(A[:,0], A[:,3])

### b)
def LMS(x):
    return np.array([gl(x), gm(x), gs(x)])

# Integrate so that E(lambda) gives an (L,M,S) value 
x = np.linspace(390, 830, 500)

# Takes as argument a function E(x)
# returns the LMS values by integrating E*LMS with the trapezoid rule
# across the interval of interpolation with 500 points
# output: an array with 3 elements.
def Spectrum_to_LMS(E):
    # Define the integration range
    lambda_min = 390
    lambda_max = 830
    n_points = 500
    
    # Setup for trapezoid rule integration
    x = np.linspace(lambda_min, lambda_max, n_points)
    
    # Compute the values at each point
    L_values = np.array([gl(lambda_val) * E(lambda_val) for lambda_val in x])
    M_values = np.array([gm(lambda_val) * E(lambda_val) for lambda_val in x])
    S_values = np.array([gs(lambda_val) * E(lambda_val) for lambda_val in x])
    
    # Trapezoid rule integration
    L = np.trapz(L_values, x)
    M = np.trapz(M_values, x)
    S = np.trapz(S_values, x)
    
    return np.array([L, M, S])

### c) 
def R(x):
    return np.exp(-(x - 620)**2/1000)/np.sqrt(2000)

def G(x):
    return np.exp(-(x - 510)**2/1000)/np.sqrt(2000)

def B(x):
    return np.exp(-(x - 460)**2/1000)/np.sqrt(2000)
    
# A program to convert (R,G,B) to (L,M,S)
R_LMS = Spectrum_to_LMS(R)  # integrate R against (gl, gm, gs)
G_LMS = Spectrum_to_LMS(G)  # integrate G against (gl, gm, gs)
B_LMS = Spectrum_to_LMS(B)  # integrate B against (gl, gm, gs)

# These can be assembled to give a matrix that converts from (R,G,B) to (L,M,S).
# Each column of B represents what happens when we have a unit of R, G, or B light
Banana = np.column_stack((R_LMS, G_LMS, B_LMS))


def project(x1, y1, z1, x2, y2, z2, x3, y3, z3, 
            f, color, batch, width, height) -> pyglet.shapes.Triangle:
    """
    Takes in the x, y and z co-ordinates of each element and projects it to 
    the window. 
    """
    fxy = [0] * 6
                
    # project 3D coordinates to 2D screen coordinates using focal length formula 
    fxy[0] = ((f * x1) / z1) + width / 2.0  # fx1
    fxy[1] = ((f * y1) / z1) + height / 2.0  # fy1
                
    fxy[2] = ((f * x2) / z2) + width / 2.0  # fx2
    fxy[3] = ((f * y2) / z2) + height / 2.0  # fy2
    
    fxy[4] = ((f * x3) / z3) + width / 2.0  # fx3
    fxy[5] = ((f * y3) / z3) + height / 2.0  # fy3

    # create a triangle at the projected coordinates with appropriate color
    return pyglet.shapes.Triangle(
        fxy[0], fxy[1], fxy[2], fxy[3], fxy[4], fxy[5],
        color = color.astype(np.int64),
        batch=batch
    )
   
#light model
class Model_3d:
    def __init__(self, nodes, elements, emit, reflect):
        self.nodes = nodes
        self.elements = elements
        self.batch = pyglet.graphics.Batch()
        self.triangles = {}
        self.emit = emit
        self.reflect = reflect

        self.normals, self.areas = self.update_geometry()

        self.colors = self.find_colors(3)

    # ***************************************************
    def add_nodes(self, nodes): 
        self.nodes.append(nodes)
    def delete_node(self, j): 
        self.nodes.pop(j)
    def add_element(self, element, color, emit, reflect, back=True): 
        self.elements.append(element)
        self.emit.append(emit)
        self.reflect.append(reflect)
        # self.colors.append(color)
    def delete_element(self, i): 
        self.elements.pop(i)
        self.emit.pop(i)
        self.reflect.pop(i)
        # self.colors.pop(i)        
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
    #oopgave 2
    @classmethod
    def model_rect(cls, x0, x1, m, y0, y1, n,
                   emit=np.array([0, 0, 0]), reflect=np.array([0, 0, 0.5]),
                   vec=[0, 0, 0], axis=[0, 0, 1], angle=0):
        x_coords = np.linspace(x0, x1, m)
        y_coords = np.linspace(y0, y1, n)
        XX, YY = np.meshgrid(x_coords, y_coords)
        Z = np.transpose(np.vstack((XX.reshape(n*m),YY.reshape(n*m))))
        d = np.zeros((m*n,1)) 
        nodes = np.hstack((Z, d))
        elements = []
        for j in range(n-1):        #hjelp fra Khai
            for i in range(m-1):
                A = j*n + i 
                B = A + 1
                C = B + m
                D = C - 1                
                elements.append([A, B, C])
                elements.append([C, D, A])
        #sets the values 
        elements = np.array(elements)   
        num_elems = len(elements)
        #replicates "emit" and "reflects" #hjelp fra Zsombor
        emits = np.tile(np.array(emit), (num_elems, 1))     
        reflects = np.tile(np.array(reflect), (num_elems, 1))
        
        #create quaternion that can be used for rotation
        quat = q.from_rotation(axis, angle)
        nodes = quat.rotate_vector(nodes)
        nodes += vec
        #sends the elements to be drawn
        return cls(nodes, elements, emits, reflects) 
    def add_model(self, other):
        offset = len(self.nodes)        #Help from khai
        self.nodes = np.vstack((self.nodes, other.nodes))
        self.elements = np.vstack((self.elements, other.elements + offset))
        self.emit = np.vstack((self.emit, other.emit))
        self.reflect = np.vstack((self.reflect, other.reflect))
        #gets the values of normals and areas through update geometry
        self.normals, self.areas =  self.update_geometry()
        self.colors = self.find_colors(10)  #get the color values
    def update_geometry(self):
        normals = []
        areas = []

        for i, el in enumerate(self.elements):
           v0 = self.nodes[el[0]]      #A
           v1 = self.nodes[el[1]]      #B
           v2 = self.nodes[el[2]]      #C
           AB = v1 - v0                #B - A
           AC = v2 - v0                #C - A
            
           cross = np.cross(AB, AC)    #AB x AC vector
           normals.append(cross / np.linalg.norm(cross) )  #v / |v|
           areas  .append(0.5 * np.linalg.norm(cross))        #Area of triangle = 0.5 * absolute value of the cross product of AB and AC.

        return normals, areas
    def update_form_factors(self, i, j):
        # return form_factor_matrix(self.nodes, self.elements, self.normals, self.areas)     
        pi = np.mean(self.nodes[self.elements[i]], axis=0)  # midpoint of i
        pj = np.mean(self.nodes[self.elements[j]], axis=0)  # midpoint of j
        
        ni = self.normals[i]                           #normal vector for i
        nj = self.normals[j]                           #normal vector for j

        r_vec = pi-pj                           
        r_magnitude = np.linalg.norm(r_vec)       # Distance between midpoints
        
        if r_magnitude < 1e-10:                   # If the points are the same then ignore
            return 0.0                            
        
        #calculates the angle between normal at i and vector of j
        cos_theta_i = np.dot(r_vec, ni)    
        cos_theta_j = np.dot(-r_vec, nj)   
        
        if cos_theta_i <= 0 or cos_theta_j <= 0:  
            return 0.0                            
        
        #form factor
        f_ij = self.areas[j] * (cos_theta_i * cos_theta_j) / (np.pi * r_magnitude**2)
                                                
        return f_ij
    

    def form_factor_matrix(self):
        n = len(self.elements)  
        F = np.zeros((n, n))  
        #goes through all the elements
        for i in range(n):
            for j in range(n):
                if i != j:  #as long as it doesnt find form factor of itself 
                    F[i, j] = self.update_form_factors(i, j)
        return F

    def find_luminances(self, n):
        emit = np.copy(self.emit)       
        form_factor = self.form_factor_matrix()
        #goes through n to get the emit and reflect values at i and j to find luminance
        for _ in range(n):
            #saves the old luminance before the new one is created
            old_luminance = np.copy(emit)   
            for i in range(len(self.elements)):
                for j in range(len(self.elements)):
                    if i != j:
                        emit[i] = self.emit[i] + self.reflect[i] * form_factor[i][j] * old_luminance[j]
        return emit

   
    
    def find_colors(self, n):
        bro = self.find_luminances(n)
        return LMS_to_RGB(bro)  


    def draw(self, focal_length, window_width, window_height):
        # remove the previous traingles
        self.clean_triangles()
        # goes through for loop to draw each element

        for i in range(len(self.elements)):
            self.draw_element(i, focal_length, window_width, window_height)
        self.batch.draw()
    def clean_triangles(self):
        for triangle in self.triangles.values():
            triangle.delete()
    # clear the dictionary
        self.triangles.clear()
    
    
def LMS_to_RGB(luminances):
    #takes the value calculated from the Spectrum_to_LMS function to find RGB
    RGBLin = np.dot(luminances, np.linalg.inv(Banana))   #fikk hjelp av Khai
    

    return np.clip(RGBLin, 0, 255)
        



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


#Create the walls, ceiling and light source 
# Create floor (using model_rect)
floor = Model_3d.model_rect(
    -100, 100, 5,  # x-range and divisions
    -100, 100, 5,  # y-range and divisions
    [0, 0, 0],  # emit 
    [0.2,1,1.5],  # reflect
    [0, -100, 0],  # position at bottom
    [1, 0, 0],  # rotation axis
    np.pi/2 # rotation angle
)

# Create ceiling
ceiling = Model_3d.model_rect(
    -100, 100, 5,
    -100, 100, 5,
    [0, 0, 0],  # emit 
    [3,0.5,7],  # reflect 
    [0, 60, 0],  # position at top
    [1, 0, 0],
    np.pi/2  
)

# Create walls (each facing inward)
# Front wall
front_wall = Model_3d.model_rect(
    -100, 100, 5,
    -100, 100, 5,
    [0, 0, 0],
    [0.3,5,7],
    [0, 0 , 100],
    [0, 0, 1],
    np.pi/2
)

# Back wall
back_wall = Model_3d.model_rect(
    -100, 100, 5,
    -100, 100, 5,
    [0, 0, 0],
    [2, 0.7, 0.1],
    [0, 0, -100],
    [0, 0, 1],
    np.pi/2
)

# Left wall 
left_wall = Model_3d.model_rect(
    -100, 100, 5,
    -100, 100, 5,
    [0, 0, 0],
    [0.5, 0.7, 2],
    [-100, 0, 0],
    [0, 1, 0],
    -np.pi/2
)

# Right wall 
right_wall = Model_3d.model_rect(
    -100, 100, 5,
    -100, 100, 5,
    [0, 0, 0],
    [0.01, 1.0, 0.5],
    [100, 0, 0],
    [0, 1, 0],
    np.pi/2
)



ceiling_light = Model_3d.model_rect(
    -20, 20, 5,
    -20, 20, 5,
    np.array([8, 3, 2])*4,  
    [0,0,0],    # No reflection
    [0, 90, 100],  
    [1, 0, 0],
    -np.pi/2  
)


window = pyglet.window.Window(width= width,height=height,resizable=True)

    
    
@window.event 
def on_key_press(symbol, modifiers): 
    if symbol == pyglet.window.key.D:   #go right
        floor.translate([50, 0, 0])
    if symbol == pyglet.window.key.L:
        ceiling_light.translate([50,0,0])
    if symbol == pyglet.window.key.A:   #go left
        floor.translate([-50, 0, 0])
    if symbol == pyglet.window.key.W:   #go up
        floor.translate([0, 50, 0])
    if symbol == pyglet.window.key.S:   #go down
        floor.translate([0, -50, 0])
    if symbol == pyglet.window.key.E:   #go in
        floor.translate([0, 0, 10])
    if symbol == pyglet.window.key.Q:   #go out
        floor.translate([0, 0, -10])
    if symbol == pyglet.window.key.R:   #rotate in the clockwise x direction
        floor.rotate(qx)
    if symbol == pyglet.window.key.T:   #rotate in the anticlockwise x direction
        floor.rotate(qx.inverse())    
    if symbol == pyglet.window.key.F:   #rotate in the clockwise y direction
        floor.rotate(qy)
    if symbol == pyglet.window.key.G:   #rotate in the anticlockwise y direction
        floor.rotate(qy.inverse())
    if symbol == pyglet.window.key.C:   #rotate in the clockwise z direction
        floor.rotate(qz)
    if symbol == pyglet.window.key.V:   #rotate in the anticlockwise z direction
        floor.rotate(qz.inverse())


# Add all components
floor.add_model(ceiling) 
floor.add_model(front_wall) 
floor.add_model(back_wall) 
floor.add_model(left_wall) 
floor.add_model(right_wall) 
floor.add_model(ceiling_light)

#update the on_draw event to use our room model
@window.event
def on_draw():
    window.clear()
    floor.draw(f, width, height)

pyglet.app.run()
