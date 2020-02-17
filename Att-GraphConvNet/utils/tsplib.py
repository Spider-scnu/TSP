import numpy as np

def read_tsplib_coor(tsplib_file, dims = 2):
    '''
    args:
        tsplib_file : the path of tsplib files, eg. './data/experiment-data/tsp225.tsp'
    
    return:
        coor : 
    '''
    with open(tsplib_file, 'r') as f:
        lines = f.readlines()
        city_num = int(lines[0])
        coor = np.zeros(shape = (1, city_num, dims), dtype = np.float64)
        for i in lines[4:4+city_num]:
            
            coor_xy = i[:-1].split(' ')
            while  '' in coor_xy:
                coor_xy.remove('')
            coor[0, int(coor_xy[0])-1, 0] = np.float64(coor_xy[1])
            coor[0, int(coor_xy[0])-1, 1] = np.float64(coor_xy[2])
    return coor
def read_tsplib_opt(tsplib_opt_file):
    '''
    args: 
        tsplib_opt_file : the path of tsplib opt files
    
    return:
        opt : 
    '''
    with open(tsplib_opt_file, 'r') as f:
        lines = f.readlines()
        city_num = int(lines[0])
        opt = np.zeros(shape = (city_num, ), dtype = np.int32)
        for i, j in enumerate(lines[4:4+city_num]):
            opt[i] = int(j) - 1
        opt = np.append(opt, 0)
    return opt

def write_tsplib_prob(tsp_instance_name, edge_prob, num_node, mean, fnn = 0, greater_zero = '---'):
    '''
    args:
        tsp_instance_name : the name of tsplib instances
        edge_prob : a symmetric matrix, which inludes edges' probability belonged to the optimization solution
        num_node : the number of cities in tsplib instances
        fnn : the number of false negative edges
   
    return:
        NONE
    '''
    with open('{}'.format(tsp_instance_name), 'w') as f:
        f.write('the mean of prob-rank is {}\n'.format(mean))
        f.write('the number of false negetive edges is {}\n'.format(fnn))
        f.write('the number of probability greater than 0 is {}/edge\n'.format(greater_zero))
        f.write('TYPE: Probability\n')
        f.write('DIMENSIOn: {}\n'.format(num_node))
        for i in range(num_node):
            for j in range(num_node):
                #f.write(' {}'.format(edge_prob[i, j]))
                f.write(' {}'.format('%5f'%edge_prob[i, j]))
            f.write('\n')