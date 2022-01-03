import itertools
import random

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import time
import csv
import pandas as pd

from osgeo import ogr

#G = nx.read_graphml("graph.graphml")
#G.nodes()
#G.edges()
#G.nodes()
#nx.draw_networkx(G, node_size=.25)
#plt.show()

'''
allowed = []
board = np.array(
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0])
for i in range(len(board)):
    if i >= len(board) - 7:
        if board[i] == 0:
            allowed.append(i)
    else:
        if board[i] == 0 and board[i + 7] != 0:
            allowed.append(i)

print(allowed)
'''


'''
Q = 0
U = 0

maxQU = -9999

if Q + U > maxQU:
    print("jaasti")
'''


#dataframe = geopandas.read_file("grid_inside_gpd.csv")
#print(dataframe)

'''
points = []

df = pd.read_csv('grid_inside_gpd.csv')
points = df['geometry']

home_nodes = []

for i in range(1, 60):
    home_nodes.append(points[20*i])

grid_points = []
for point in points:
    multipoint = ogr.CreateGeometryFromWkt(str(point))
    #grid_point = multipoint.GetPoint()
    grid_points.append(multipoint.GetPoint())

def calc_distance(node1, node2):
    return ((node2[0] - node1[0]) ** 2 + (node2[1] - node1[1]) ** 2) ** 0.5

print(calc_distance(grid_points[0], grid_points[1]))
print(calc_distance(grid_points[0], grid_points[2]))
print(calc_distance(grid_points[0], grid_points[3]))
print(calc_distance(grid_points[0], grid_points[54]))
print(calc_distance(grid_points[0], grid_points[55]))


print(abs(-0.0008))

'''
'''
points = []

df = pd.read_csv('grid_inside_gpd.csv')
points = df['geometry']


grid_points = []
for point in points:
    multipoint = ogr.CreateGeometryFromWkt(str(point))
    grid_point = multipoint.GetPoint()
    if grid_point[0] > 8.908829000791403 and grid_point[0] <= 8.909149000791405 and grid_point[1] <= 48.77679495302782:
        grid_points.append((grid_point[0], grid_point[1]))

home_nodes = []
home_nodes.append(grid_points[0])



grid_points.plot(markersize=0.1,color='k').axis('off');
'''
#edges = [0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1]
#res = [idx for idx, val in enumerate(edges) if val != 0]
#random.shuffle(res)
#print(np.nonzero(edges))

list1 = [1, 2, 3]
list2 = [1, 2]
if all(item in list1 for item in list2):
    print(1)
if all(item in list2 for item in list1):
    print(2)