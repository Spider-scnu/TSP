#ifndef Kernel_H
#define Kernel_H

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <string.h>
#include <time.h>
#include <ctime>
#include <vector>
#include <string.h>
#include <math.h>

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <curand_kernel.h>
#include <helper_functions.h>
#include <device_functions.h>
#include <device_launch_parameters.h>

using namespace std;

#define thread_per_block 64//64
#define block_num 2
#define Max_Candidate_Num  10
#define Inf_Cost 1000000000
#define Beta 10
#define Null -1 
#define Magnify_Rate       1000000
#define Param_H 10
#define Max_Depth  		   10
#define Alpha 1

struct Struct_Node
{
	int Pre_City;
	int Next_City;
	int Salesman;
};
extern "C" void Init_GPU(int *Current_Instance_Best_Distance_G, double *Weight_G, int *Chosen_Times_G, int *Total_Simulation_Times_G, int Virtual_City_Num, int Total_thread_num);

extern "C" void Excute_GPU(int randseed, int *Distance_G, Struct_Node *Best_All_Node_G, Struct_Node *All_Node_G, double* Weight_G, int* Chosen_Times_G, int* Total_Simulation_Times_G, int Virtual_City_Num, int Total_thread_num, int *Candidate_G, int *Candidate_Num_G, int *Current_Instance_Best_Distance_G, int *Temp_City_Sequence_G, int *Temp_Pair_Num_G, int *Probabilistic_G, int *Promising_City_G, double *Avg_Weight_G, int *Promising_City_Num_G, int *Pair_City_Num_G, int *Real_Gain_G, int *Gain_G, int *Solution_G, int *City_Sequence_G, double *Current_Solution_Double_Distance_G, double *Coordinate_X_G, double *Coordinate_Y_G);
//extern "C" void excute_test(int* d_test);

#endif // !Kernel_H