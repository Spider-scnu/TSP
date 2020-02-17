import time
import argparse
import pprint as pp
import os

import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
import sys
sys.path.append(r'/workData/kb/pyconcorde')
import pyximport
pyximport.install()
#from concorde.tsp import TSPSolver

def convert_tsplib(tspinstance, num_node = 20, num_instance = 0, scale = 10000):
    derivate = tspinstance.split(' ')
    opt_sol = np.array(derivate[2*num_node+1:-1], dtype = np.int32) - 1 
    buff = np.zeros(shape=(num_node, 2), dtype = np.float64)
    buff[:, 0] = np.array(derivate[0:2*num_node:2], dtype = np.float64)
    buff[:, 1] = np.array(derivate[1:2*num_node:2], dtype = np.float64)
    # raw distance matrix
    distRawA = pdist(buff, metric='euclidean')
    distRawB = squareform(distRawA)
    # scale-variable distance matrix
    buff = np.array(buff * scale, dtype = np.int64)
    distA = pdist(buff, metric='euclidean')
    distB = squareform(distA)
    opt_value = np.int64(np.sum(distB[opt_sol[:-1], opt_sol[1:]]))
    #opt_value = np.sum(distB[opt_sol[:-1], opt_sol[1:]])
    with open('./LKH-3.0.6/data/tsp{}/uniform{}.tsp'.format(num_node, num_instance), 'w') as f1:
        f1.write('NAME : uniform{}\n'.format(num_node))
        f1.write('COMMENT : {}-city problem\n'.format(num_node))
        f1.write('TYPE : TSP\n')
        f1.write('DIMENSION : {}\n'.format(num_node))
        f1.write('EDGE_WEIGHT_TYPE : EUC_2D\n')
        f1.write('NODE_COORD_SECTION\n')
        for i, (x, y) in enumerate(buff):
            f1.write('{} {} {}\n'.format(i+1, x, y))
        f1.write('EOF\n')
    with open('./LKH-3.0.6/data/tsp{}/uniform{}.par'.format(num_node, num_instance), 'w') as f2:
        f2.write('PROBLEM_FILE = ./LKH-3.0.6/data/tsp{}/uniform{}.tsp\n'.format(num_node, num_instance))
        #f2.write('OPTIMUM = {}\n'.format(opt_value))
        f2.write('MOVE_TYPE = 5\n')
        f2.write('PATCHING_C = 3\n')
        f2.write('PATCHING_A = 2\n')
        f2.write('OUTPUT_TOUR_FILE = ./LKH-3.0.6/results/tsp{}/uniform{}.opt\n'.format(num_node, num_instance))
        output_filepath = './LKH-3.0.6/results/tsp{}/uniform{}.opt'.format(num_node, num_instance)
    return distRawB, opt_sol, output_filepath

def tsp_instance_reader(tspinstance, buff, num_node = 20):
    derivate = tspinstance.split(' ')
    buff[:, 0] = np.array(derivate[0:2*num_node:2], dtype = np.float64)
    buff[:, 1] = np.array(derivate[1:2*num_node:2], dtype = np.float64)
    opt_sol = np.array(derivate[2*num_node+1:-1], dtype = np.int32) - 1 
    return buff, opt_sol
    

def tsp_instances_generator(num_samples = 10000, num_nodes = 20, node_dim = 2, filename = None,
                           solved = True):
    if filename is None:
        filename = f"tsp{num_nodes}_concorde.txt"
    
    # Pretty print the run args
    #pp.pprint(vars(opts))
    
    set_nodes_coord = np.random.random([num_samples, num_nodes, node_dim])
    if solved:
        with open(filename, "w") as f:
            start_time = time.time()
            for nodes_coord in set_nodes_coord:
                solver = TSPSolver.from_data(nodes_coord[:,0], nodes_coord[:,1], norm="GEO")  
                solution = solver.solve()
                f.write( " ".join( str(x)+str(" ")+str(y) for x,y in nodes_coord) )
                f.write( str(" ") + str('output') + str(" ") )
                f.write( str(" ").join( str(node_idx+1) for node_idx in solution.tour) )
                f.write( str(" ") + str(solution.tour[0]+1) + str(" ") )
                f.write( "\n" )
            end_time = time.time() - start_time
    
        print(f"Completed generation of {num_samples} samples of TSP{num_nodes}.")
        print(f"Total time: {end_time}s")
        print(f"Average time: {end_time/num_samples}s")
        #print(f"Total time: {end_time/3600:.1f}h")
        #print(f"Average time: {(end_time/3600)/num_samples:.1f}h")
    return set_nodes_coord