#include "kernel.h"

__global__ void MCTS_Init_GPU(double* Edge_Heatmap_G, int *Current_Instance_Best_Distance_G, double* Weight_G, int* Chosen_Times_G, int* Total_Simulation_Times_G, int Virtual_City_Num, int Total_thread_num)
{
	int idx = blockDim.x*blockIdx.x + threadIdx.x;
	for (int i = 0; i < Virtual_City_Num; i++)
	{
		for (int j = 0; j < Virtual_City_Num; j++)
		{
			Weight_G[idx * Total_thread_num + i * Virtual_City_Num + j] = Edge_Heatmap_G[i * Virtual_City_Num + j] * 100;//1;
			Chosen_Times_G[idx * Total_thread_num + i * Virtual_City_Num + j] = 0;
			//printf("%d %f %d\n", idx * Total_thread_num + i * Virtual_City_Num + j, Weight_G[idx * Total_thread_num + i * Virtual_City_Num + j], Chosen_Times_G[idx * Total_thread_num + i * Virtual_City_Num + j]);
		}
	}
	Current_Instance_Best_Distance_G[idx] = Inf_Cost;
	Total_Simulation_Times_G[idx] = 0;
	//printf("MCTS_Init_GPU \n");
}

__global__ void Get_Init_Solution_GPU(Struct_Node *All_Node_G, int Virtual_City_Num)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int Cur_City = 0, Next_City;
	printf("%d ", Cur_City);
	while (true)
	{
		Next_City = All_Node_G[idx * Virtual_City_Num + Cur_City].Next_City;
		if (Next_City == 0)
			break;
		printf("%d ", Next_City);
		Cur_City = Next_City;
	}
	printf("\n");
}

__device__ void Candidate_GPU(int threadid, int Virtual_City_Num, int Total_thread_num, int *Candidate_G)
{
	for (int i = 0; i < Virtual_City_Num; i++)
	{
		for (int j = 0; j < Max_Candidate_Num; j++)
		{
			printf("%d ", Candidate_G[threadid*Total_thread_num + i*Max_Candidate_Num + j]);
		}
		printf("\n");
	}
	printf("\n-----------------------------\n");
}

__global__ void Get_Candidate_GPU(int Virtual_City_Num, int Total_thread_num, int *Candidate_G)
{
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	Candidate_GPU(index, Virtual_City_Num, Total_thread_num, Candidate_G);
}

__device__ bool Check_If_Two_City_Same_Or_Adjacent_GPU(int index, int Virtual_City_Num, int First_City, int Second_City, Struct_Node *All_Node_G)
{
	if (First_City == Second_City || All_Node_G[index * Virtual_City_Num + First_City].Next_City == Second_City || All_Node_G[index * Virtual_City_Num + Second_City].Next_City == First_City)
		return true;
	else
		return false;
}

__device__ int Get_Distance_GPU(int First_City, int Second_City, int *Distance_G, int Virtual_City_Num)
{
	return Distance_G[First_City*Virtual_City_Num + Second_City];
}

__device__ int Get_2Opt_Delta_GPU(int *Distance_G, int *Total_Simulation_Times_G, int Total_thread_num, int index, int Virtual_City_Num, int First_City, int Second_City, Struct_Node *All_Node_G, int *Chosen_Times_G)
{
	if (Check_If_Two_City_Same_Or_Adjacent_GPU(index, Virtual_City_Num, First_City, Second_City, All_Node_G) == true)
		return -Inf_Cost;

	int First_Next_City = All_Node_G[index * Virtual_City_Num + First_City].Next_City;
	int Second_Next_City = All_Node_G[index * Virtual_City_Num + Second_City].Next_City;

	int Delta = Get_Distance_GPU(First_City, First_Next_City, Distance_G, Virtual_City_Num) + Get_Distance_GPU(Second_City, Second_Next_City, Distance_G, Virtual_City_Num)
		- Get_Distance_GPU(First_City, Second_City, Distance_G, Virtual_City_Num) - Get_Distance_GPU(First_Next_City, Second_Next_City, Distance_G, Virtual_City_Num);

	Chosen_Times_G[index * Total_thread_num + First_City * Virtual_City_Num + Second_City] ++;
	Chosen_Times_G[index * Total_thread_num + Second_City * Virtual_City_Num + First_City] ++;
	Chosen_Times_G[index * Total_thread_num + First_Next_City * Virtual_City_Num + Second_Next_City] ++;
	Chosen_Times_G[index * Total_thread_num + Second_Next_City * Virtual_City_Num + First_Next_City] ++;

	Total_Simulation_Times_G[index]++;

	return Delta;
}

__device__ int Get_Solution_Total_Distance_GPU(int *Distance_G, int threadid, int Virtual_City_Num, Struct_Node *All_Node_G)
{
	int Solution_Total_Distance = 0;
	for (int i = 0; i<Virtual_City_Num; i++)
	{
		int Temp_Next_City = All_Node_G[threadid * Virtual_City_Num + i].Next_City;
		if (Temp_Next_City != Null)
			Solution_Total_Distance += Get_Distance_GPU(i, Temp_Next_City, Distance_G, Virtual_City_Num);
		else
		{
			printf("\nGet_Solution_Total_Distance() fail!\n");
			return Inf_Cost;
		}
	}

	return Solution_Total_Distance;
}

__device__ void Reverse_Sub_Path_GPU(int First_City, int Second_City, Struct_Node *All_Node_G, int threadid, int Virtual_City_Num)
{
	int Cur_City = First_City;
	int Temp_Next_City = All_Node_G[threadid * Virtual_City_Num + Cur_City].Next_City;
	int Temp_City = All_Node_G[threadid * Virtual_City_Num + Cur_City].Pre_City;
	All_Node_G[threadid * Virtual_City_Num + Cur_City].Pre_City = All_Node_G[threadid * Virtual_City_Num + Cur_City].Next_City;
	All_Node_G[threadid * Virtual_City_Num + Cur_City].Next_City = Temp_City;
	if(All_Node_G[threadid * Virtual_City_Num + Cur_City].Pre_City==-1){
		printf("All_Node_G[threadid * Virtual_City_Num + Cur_City].Pre_City = %d. \n", All_Node_G[threadid * Virtual_City_Num + Cur_City].Pre_City);
	}
	while (Cur_City != Second_City)
	{
		

		/*if (Cur_City == Second_City)
			break;*/

		Cur_City = Temp_Next_City;
		Temp_Next_City = All_Node_G[threadid * Virtual_City_Num + Cur_City].Next_City;
		
		Temp_City = All_Node_G[threadid * Virtual_City_Num + Cur_City].Pre_City;
		All_Node_G[threadid * Virtual_City_Num + Cur_City].Pre_City = All_Node_G[threadid * Virtual_City_Num + Cur_City].Next_City;
		All_Node_G[threadid * Virtual_City_Num + Cur_City].Next_City = Temp_City;
	}
}

__device__ void Apply_2Opt_Move_GPU(double *Weight_G, int *Total_Simulation_Times_G, int Total_thread_num, int *Distance_G,
 int threadid, int Virtual_City_Num, Struct_Node *All_Node_G, int First_City, int Second_City, int *Chosen_Times_G)
{
	int Before_Distance = Get_Solution_Total_Distance_GPU(Distance_G, threadid, Virtual_City_Num, All_Node_G);
	int Delta = Get_2Opt_Delta_GPU(Distance_G, Total_Simulation_Times_G, Total_thread_num, threadid, Virtual_City_Num, First_City, Second_City, All_Node_G, Chosen_Times_G);

	int First_Next_City = All_Node_G[threadid * Virtual_City_Num + First_City].Next_City;
	int Second_Next_City = All_Node_G[threadid * Virtual_City_Num + Second_City].Next_City;

	Reverse_Sub_Path_GPU(First_Next_City, Second_City, All_Node_G, threadid, Virtual_City_Num);
	All_Node_G[threadid * Virtual_City_Num + First_City].Next_City = Second_City;
	All_Node_G[threadid * Virtual_City_Num + Second_City].Pre_City = First_City;
	All_Node_G[threadid * Virtual_City_Num + First_Next_City].Next_City = Second_Next_City;
	All_Node_G[threadid * Virtual_City_Num + Second_Next_City].Pre_City = First_Next_City;

	double Increase_Rate = Beta*(pow(2.718, (double)(Delta) / (double)(Before_Distance)) - 1);

	Weight_G[threadid * Total_thread_num + First_City * Virtual_City_Num + Second_City] += Increase_Rate;
	Weight_G[threadid * Total_thread_num + Second_City * Virtual_City_Num + First_City] += Increase_Rate;
	Weight_G[threadid * Total_thread_num + First_Next_City * Virtual_City_Num + Second_Next_City] += Increase_Rate;
	Weight_G[threadid * Total_thread_num + Second_Next_City * Virtual_City_Num + First_Next_City] += Increase_Rate;
}

__device__ bool Improve_By_2Opt_Move_GPU(int threadid, double *Weight_G, int *Distance_G, int *Total_Simulation_Times_G, int Virtual_City_Num, int Cur_thread_id, int *Candidate_Num_G, int *Candidate_G, int Total_thread_num, Struct_Node *All_Node_G, int *Chosen_Times_G)
{
	bool If_Improved = false;
	//printf("Starting to local search. Virtual_City_Num = %d\n", Virtual_City_Num);
	for (int i = 0; i<Virtual_City_Num; i++)
	{
		//printf("Starting to local search Step-0, Candidate_Num = %d.\n", Candidate_Num_G[threadid * Virtual_City_Num + i]);
		//Candidate_Num_G[threadid * Virtual_City_Num + i] = 0;
		//printf("Starting to local search Step-0, Candidate_Num = %d.\n", Candidate_Num_G[threadid * Virtual_City_Num + i]);
		for (int j = 0; j<Candidate_Num_G[threadid * Virtual_City_Num + i]; j++)
		{
			//printf("Starting to local search Step-1.\n");
			int Candidate_City = Candidate_G[Cur_thread_id*Total_thread_num + i*Max_Candidate_Num + j];
			//printf("Starting to local search Step-2.\n");
			// Step-3
			
			//printf("Output = %d\n", Get_2Opt_Delta_GPU(Distance_G, Total_Simulation_Times_G, Total_thread_num, Cur_thread_id, Virtual_City_Num, i, Candidate_City, All_Node_G, Chosen_Times_G));
			if (Get_2Opt_Delta_GPU(Distance_G, Total_Simulation_Times_G, Total_thread_num, Cur_thread_id, Virtual_City_Num, i, Candidate_City, All_Node_G, Chosen_Times_G)>0)
			{
				//printf("Starting to local search Step-3.\n");
				Apply_2Opt_Move_GPU(Weight_G, Total_Simulation_Times_G, Total_thread_num, Distance_G, Cur_thread_id, Virtual_City_Num, All_Node_G, i, Candidate_City, Chosen_Times_G);
				If_Improved = true;
				break;
			}
			//printf("Finishing. \n");
		}
	}
	return If_Improved;
}

__device__ void Store_Best_Solution_GPU(int idx, int Virtual_City_Num, Struct_Node *Best_All_Node_G, Struct_Node *All_Node_G)
{
	for (int i = 0; i<Virtual_City_Num; i++)
	{
		Best_All_Node_G[idx * Virtual_City_Num + i].Salesman = All_Node_G[idx * Virtual_City_Num + i].Salesman;
		Best_All_Node_G[idx * Virtual_City_Num + i].Next_City = All_Node_G[idx * Virtual_City_Num + i].Next_City;
		Best_All_Node_G[idx * Virtual_City_Num + i].Pre_City = All_Node_G[idx * Virtual_City_Num + i].Pre_City;
	}
}

__global__ void Local_Search_by_2Opt_Move_GPU(Struct_Node *Best_All_Node_G, double *Weight_G, int *Distance_G,
	int *Total_Simulation_Times_G, int Virtual_City_Num, int *Candidate_Num_G, int *Candidate_G, int Total_thread_num,
	Struct_Node *All_Node_G, int *Chosen_Times_G, int *Current_Instance_Best_Distance_G)
{
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	//printf("ThreadID-1 = %d\n", index);
	
	while (Improve_By_2Opt_Move_GPU(index, Weight_G, Distance_G, Total_Simulation_Times_G, Virtual_City_Num, index, Candidate_Num_G, Candidate_G, Total_thread_num, All_Node_G, Chosen_Times_G) == true)
		;
		
	//printf("ThreadID-2 = %d\n", index);
	//printf("%d\n", Current_Instance_Best_Distance_G[index]);
	int Cur_Solution_Total_Distance = Get_Solution_Total_Distance_GPU(Distance_G, index, Virtual_City_Num, All_Node_G);
	if (Cur_Solution_Total_Distance < Current_Instance_Best_Distance_G[index])
	{
		Current_Instance_Best_Distance_G[index] = Cur_Solution_Total_Distance;
		Store_Best_Solution_GPU(index, Virtual_City_Num, Best_All_Node_G, All_Node_G);
	}
	//printf("ThreadID-3 = %d, Current_Instance_Best_Distance_G = %d. \n", index, Current_Instance_Best_Distance_G[index]);
	//printf("%f\n", (double)Current_Instance_Best_Distance_G[index] / Magnify_Rate);
}

__device__ int Get_Random_Int_GPU(int randseed, int threadid, int Virtual_City_Num)
{
	curandState state;
	long seed = (long)(randseed);
	curand_init(seed, threadid, 0, &state);
	int randomCity = round(abs(curand_uniform_double(&state)) * (Virtual_City_Num - 1));
	//printf("%d\n", randomCity);
	return randomCity;
}

__device__ bool Convert_All_Node_To_Solution_GPU(int Virtual_City_Num, int *Solution_G, int threadid, Struct_Node *All_Node_G)
{
	for (int i = 0; i<Virtual_City_Num; i++)
		Solution_G[threadid * Virtual_City_Num + i] = Null;

	int Cur_Index = 0;
	Solution_G[threadid * Virtual_City_Num + Cur_Index] = 0;

	int Cur_City = 0;
	do
	{
		Cur_Index++;

		Cur_City = All_Node_G[threadid * Virtual_City_Num + Cur_City].Next_City;
		if (Cur_City == Null || Cur_Index >= Virtual_City_Num)
			return false;

		Solution_G[threadid * Virtual_City_Num + Cur_Index] = Cur_City;
	} while (All_Node_G[threadid * Virtual_City_Num + Cur_City].Next_City != 0);

	return true;
}

__device__ void Convert_Solution_To_All_Node_GPU(int Virtual_City_Num, int *Solution_G, int threadid, Struct_Node *All_Node_G)
{
	int Temp_Cur_City;
	int Temp_Pre_City;
	int Temp_Next_City;
	int Cur_Salesman = 0;

	for (int i = 0; i<Virtual_City_Num; i++)
	{
		Temp_Cur_City = Solution_G[threadid * Virtual_City_Num + i];
		Temp_Pre_City = Solution_G[threadid * Virtual_City_Num + (i - 1 + Virtual_City_Num) % Virtual_City_Num];
		Temp_Next_City = Solution_G[threadid * Virtual_City_Num + (i + 1 + Virtual_City_Num) % Virtual_City_Num];

		All_Node_G[threadid * Virtual_City_Num + Temp_Cur_City].Pre_City = Temp_Pre_City;
		All_Node_G[threadid * Virtual_City_Num + Temp_Cur_City].Next_City = Temp_Next_City;
		All_Node_G[threadid * Virtual_City_Num + Temp_Cur_City].Salesman = Cur_Salesman;
	}
}

__device__ double Get_Avg_Weight_GPU(int Cur_City, int Virtual_City_Num, double *Weight_G, int threadid, int Total_thread_num)
{
	double Total_Weight = 0;
	for (int i = 0; i<Virtual_City_Num; i++)
	{
		if (i == Cur_City)
			continue;

		Total_Weight += Weight_G[threadid * Total_thread_num + Cur_City * Virtual_City_Num + i];
	}

	return Total_Weight / (Virtual_City_Num - 1);
}

__device__ double Get_Potential_GPU(int First_City, int Second_City, int Total_thread_num, int Virtual_City_Num, double *Weight_G, int threadid, int *Total_Simulation_Times_G, double *Avg_Weight_G, int *Chosen_Times_G)
{
	double Potential = Weight_G[threadid * Total_thread_num + First_City * Virtual_City_Num + Second_City] / Avg_Weight_G[threadid] + Alpha*sqrt((Total_Simulation_Times_G[threadid] + 1) / (0.434*(Chosen_Times_G[threadid * Total_thread_num + First_City * Virtual_City_Num + Second_City] + 1)));
	//	double Potential = Weight_G[threadid * Total_thread_num + First_City * Virtual_City_Num + Second_City] / Avg_Weight_G[threadid] + Alpha*sqrt(log(Total_Simulation_Times_G[threadid] + 1) / (0.434*(Chosen_Times_G[threadid * Total_thread_num + First_City * Virtual_City_Num + Second_City] + 1)));
	return Potential;
}

__device__ void Identify_Promising_City_GPU(int Cur_City, int Begin_City, int *Promising_City_G, double *Weight_G, int *Total_Simulation_Times_G, double *Avg_Weight_G, int *Chosen_Times_G, int Total_thread_num, int Virtual_City_Num, Struct_Node *All_Node_G, int *Candidate_G, int *Candidate_Num_G, int *Promising_City_Num_G, int threadid)
{
	Promising_City_Num_G[threadid] = 0;
	for (int i = 0; i<Candidate_Num_G[threadid * Virtual_City_Num + Cur_City]; i++)
	{
		int Temp_City = Candidate_G[threadid*Total_thread_num + Cur_City*Max_Candidate_Num + i];
		if (Temp_City == Begin_City)
			continue;
		if (Temp_City == All_Node_G[threadid * Virtual_City_Num + Cur_City].Next_City)
			continue;
		if (Get_Potential_GPU(Cur_City, Temp_City, Total_thread_num, Virtual_City_Num, Weight_G, threadid, Total_Simulation_Times_G, Avg_Weight_G, Chosen_Times_G) < 1)
			continue;

		Promising_City_G[threadid * Virtual_City_Num + Promising_City_Num_G[threadid]++] = Temp_City;
	}
}

__device__ bool Get_Probabilistic_GPU(int Cur_City, int *Probabilistic_G, int *Promising_City_Num_G, int threadid, int *Promising_City_G, int Virtual_City_Num, int Total_thread_num, double *Weight_G, int *Total_Simulation_Times_G, double *Avg_Weight_G, int *Chosen_Times_G)
{
	if (Promising_City_Num_G[threadid] == 0)
		return false;

	double Total_Potential = 0;
	for (int i = 0; i<Promising_City_Num_G[threadid]; i++)
		Total_Potential += Get_Potential_GPU(Cur_City, Promising_City_G[threadid * Virtual_City_Num + i], Total_thread_num, Virtual_City_Num, Weight_G, threadid, Total_Simulation_Times_G, Avg_Weight_G, Chosen_Times_G);

	Probabilistic_G[threadid * Virtual_City_Num + 0] = (int)(1000 * Get_Potential_GPU(Cur_City, Promising_City_G[threadid * Virtual_City_Num + 0], Total_thread_num, Virtual_City_Num, Weight_G, threadid, Total_Simulation_Times_G, Avg_Weight_G, Chosen_Times_G) / Total_Potential);
	for (int i = 1; i<Promising_City_Num_G[threadid] - 1; i++)
		Probabilistic_G[threadid * Virtual_City_Num + i] = Probabilistic_G[threadid * Virtual_City_Num + i - 1] + (int)(1000 * Get_Potential_GPU(Cur_City, Promising_City_G[threadid * Virtual_City_Num + i], Total_thread_num, Virtual_City_Num, Weight_G, threadid, Total_Simulation_Times_G, Avg_Weight_G, Chosen_Times_G) / Total_Potential);
	Probabilistic_G[threadid * Virtual_City_Num + Promising_City_Num_G[threadid] - 1] = 1000;

	return true;
}

__device__ int Probabilistic_Get_City_To_Connect_GPU(int randseed, int threadid, int Virtual_City_Num, int *Promising_City_Num_G, int *Probabilistic_G, int *Promising_City_G)
{
	int Random_Num = Get_Random_Int_GPU(randseed, threadid, 1000);
	for (int i = 0; i<Promising_City_Num_G[threadid]; i++)
		if (Random_Num < Probabilistic_G[threadid * Virtual_City_Num + i])
			return Promising_City_G[threadid * Virtual_City_Num + i];

	return Null;
}

__device__ int Choose_City_To_Connect_GPU(int Cur_City, int Begin_City, int randseed, int *Probabilistic_G, int *Promising_City_G, int *Total_Simulation_Times_G, int *Chosen_Times_G, double *Avg_Weight_G, Struct_Node *All_Node_G, int *Candidate_G, int *Candidate_Num_G, int *Promising_City_Num_G, int threadid, int Virtual_City_Num, double *Weight_G, int Total_thread_num)
{
	Avg_Weight_G[threadid] = Get_Avg_Weight_GPU(Cur_City, Virtual_City_Num, Weight_G, threadid, Total_thread_num);
	Identify_Promising_City_GPU(Cur_City, Begin_City, Promising_City_G, Weight_G, Total_Simulation_Times_G, Avg_Weight_G, Chosen_Times_G, Total_thread_num, Virtual_City_Num, All_Node_G, Candidate_G, Candidate_Num_G, Promising_City_Num_G, threadid);
	Get_Probabilistic_GPU(Cur_City, Probabilistic_G, Promising_City_Num_G, threadid, Promising_City_G, Virtual_City_Num, Total_thread_num, Weight_G, Total_Simulation_Times_G, Avg_Weight_G, Chosen_Times_G);

	return Probabilistic_Get_City_To_Connect_GPU(randseed, threadid, Virtual_City_Num, Promising_City_Num_G, Probabilistic_G, Promising_City_G);
}

__device__ int Get_Simulated_Action_Delta_GPU(int Begin_City, int randseed, int *Probabilistic_G, int *Promising_City_G, int *Total_Simulation_Times_G,
 double *Avg_Weight_G, int *Candidate_G, int *Candidate_Num_G, int *Promising_City_Num_G, double *Weight_G, int Total_thread_num, int *Chosen_Times_G, int *Pair_City_Num_G,
 int *Real_Gain_G, int *Gain_G, int *Distance_G, int Virtual_City_Num, int *Solution_G, int threadid, Struct_Node *All_Node_G, int *City_Sequence_G)
{
	// Store the current solution to Solution[]
	if (Convert_All_Node_To_Solution_GPU(Virtual_City_Num, Solution_G, threadid, All_Node_G) == false)
		return -Inf_Cost;

	int Next_City = All_Node_G[threadid * Virtual_City_Num + Begin_City].Next_City;   // a_1=Begin city, b_1=Next_City

	All_Node_G[threadid * Virtual_City_Num + Begin_City].Next_City = Null;
	All_Node_G[threadid * Virtual_City_Num + Next_City].Pre_City = Null;

	City_Sequence_G[threadid * Virtual_City_Num + 0] = Begin_City;
	City_Sequence_G[threadid * Virtual_City_Num + 1] = Next_City;

	Gain_G[threadid * 2 * Virtual_City_Num + 0] = Get_Distance_GPU(Begin_City, Next_City, Distance_G, Virtual_City_Num);
	Real_Gain_G[threadid * 2 * Virtual_City_Num + 0] = Gain_G[threadid * 2 * Virtual_City_Num + 0] - Get_Distance_GPU(Next_City, Begin_City, Distance_G, Virtual_City_Num);
	Pair_City_Num_G[threadid] = 1;

	bool If_Changed = false;
	int Cur_City = Next_City;
	//printf("Step 1 of Get_Simulated_Action_Delta_GPU. \n");
	while (Pair_City_Num_G[threadid] < Max_Depth)
	{
		//printf("Step 2 of Get_Simulated_Action_Delta_GPU, %d. \n", Pair_City_Num_G[threadid]);
		int Next_City_To_Connect = Choose_City_To_Connect_GPU(Cur_City, Begin_City, randseed, Probabilistic_G, Promising_City_G, Total_Simulation_Times_G, Chosen_Times_G, Avg_Weight_G, All_Node_G, Candidate_G, Candidate_Num_G, Promising_City_Num_G, threadid, Virtual_City_Num, Weight_G, Total_thread_num);
		//printf("%d, %d. \n", Next_City_To_Connect, Cur_City);
		if (Next_City_To_Connect == Null )
			break;
		if (Next_City_To_Connect!=Cur_City){
			//Update the chosen times, used in MCTS	
			Chosen_Times_G[threadid * Total_thread_num + Cur_City * Virtual_City_Num + Next_City_To_Connect] ++;
			Chosen_Times_G[threadid * Total_thread_num + Next_City_To_Connect * Virtual_City_Num + Cur_City] ++;
			//printf("Step 3 of Get_Simulated_Action_Delta_GPU. \n");
			
			int Next_City_To_Disconnect = All_Node_G[threadid * Virtual_City_Num + Next_City_To_Connect].Pre_City;
			//printf("Step 4.1 of Get_Simulated_Action_Delta_GPU. \n");
			
			City_Sequence_G[threadid * Virtual_City_Num + 2 * Pair_City_Num_G[threadid]] = Next_City_To_Connect;
			City_Sequence_G[threadid * Virtual_City_Num + 2 * Pair_City_Num_G[threadid] + 1] = Next_City_To_Disconnect;
			//printf("Step 4.2 of Get_Simulated_Action_Delta_GPU. \n");
			
			Gain_G[threadid * 2 * Virtual_City_Num + Pair_City_Num_G[threadid]] = Gain_G[threadid * 2 * Virtual_City_Num + Pair_City_Num_G[threadid] - 1] - Get_Distance_GPU(Cur_City, Next_City_To_Connect, Distance_G, Virtual_City_Num) + Get_Distance_GPU(Next_City_To_Connect, Next_City_To_Disconnect, Distance_G, Virtual_City_Num);
			//printf("Step 4.3.1 of Get_Simulated_Action_Delta_GPU. \n");
			
			//printf("Distance_G = %d, %d, %d, %d. \n", Next_City_To_Disconnect, Begin_City, Next_City_To_Connect, Cur_City);
			Real_Gain_G[threadid * 2 * Virtual_City_Num + Pair_City_Num_G[threadid]] = Gain_G[threadid * 2 * Virtual_City_Num + Pair_City_Num_G[threadid]] - Get_Distance_GPU(Next_City_To_Disconnect, Begin_City, Distance_G, Virtual_City_Num);
			//printf("Step 4.3.2 of Get_Simulated_Action_Delta_GPU. \n");
			
			
			//printf("Step 4.3.3 of Get_Simulated_Action_Delta_GPU. \n");
			
			// Reverse the cities between b_i and b_{i+1}
			Reverse_Sub_Path_GPU(Cur_City, Next_City_To_Disconnect, All_Node_G, threadid, Virtual_City_Num);
			All_Node_G[threadid * Virtual_City_Num + Cur_City].Next_City = Next_City_To_Connect;
			All_Node_G[threadid * Virtual_City_Num + Next_City_To_Connect].Pre_City = Cur_City;
			All_Node_G[threadid * Virtual_City_Num + Next_City_To_Disconnect].Pre_City = Null;
			If_Changed = true;
			//printf("Step 5 of Get_Simulated_Action_Delta_GPU. \n");
			
			// Turns to the next iteration
			Cur_City = Next_City_To_Disconnect;
		

			// Close the loop is meeting an improving action, or the depth reaches its upper bound	
			/*if (Real_Gain_G[threadid * 2 * Virtual_City_Num + Pair_City_Num_G[threadid] - 1] > 0 || Pair_City_Num_G[threadid] > Max_Depth)
				break;*/
			if (Real_Gain_G[threadid * 2 * Virtual_City_Num + Pair_City_Num_G[threadid] - 1] > 0)
				break;
			//printf("Step 6 of Get_Simulated_Action_Delta_GPU. \n");
			
			Pair_City_Num_G[threadid]++;
			
		}
		randseed = randseed * 2 + 1;
	}
	//printf("Step 7 of Get_Simulated_Action_Delta_GPU. \n");
	// Restore the solution before simulation	
	if (If_Changed)
		Convert_Solution_To_All_Node_GPU(Virtual_City_Num, Solution_G, threadid, All_Node_G);
	else
	{
		All_Node_G[threadid * Virtual_City_Num + Begin_City].Next_City = Next_City;
		All_Node_G[threadid * Virtual_City_Num + Next_City].Pre_City = Begin_City;
	}

	int Max_Real_Gain = -Inf_Cost;
	int Best_Index = 1;
	for (int i = 1; i<Pair_City_Num_G[threadid]; i++)
		if (Real_Gain_G[threadid * 2 * Virtual_City_Num + i] > Max_Real_Gain)
		{
			Max_Real_Gain = Real_Gain_G[threadid * 2 * Virtual_City_Num + i];
			Best_Index = i;
		}

	Pair_City_Num_G[threadid] = Best_Index + 1;

	return Max_Real_Gain;
}

__device__ int Simulation_GPU(int Max_Simulation_Times, int *Temp_City_Sequence_G, int *Temp_Pair_Num_G, int threadid, int randseed,
 int Virtual_City_Num, int *Total_Simulation_Times_G, int *Probabilistic_G, int *Promising_City_G, double *Avg_Weight_G, int *Candidate_G,
 int *Candidate_Num_G, int *Promising_City_Num_G, double *Weight_G, int Total_thread_num, int *Chosen_Times_G, int *Pair_City_Num_G,
 int *Real_Gain_G, int *Gain_G, int *Distance_G, int *Solution_G, Struct_Node *All_Node_G, int *City_Sequence_G)
{
	int Best_Action_Delta = -Inf_Cost;
	for (int i = 0; i < Max_Simulation_Times; i++)
	{
		//printf("Step 1 of Simulation_GPU. \n");
		int Begin_City = Get_Random_Int_GPU(randseed, threadid, Virtual_City_Num);
		//printf("Step 2 of Simulation_GPU. \n");
		int Action_Delta = Get_Simulated_Action_Delta_GPU(Begin_City, randseed, Probabilistic_G, Promising_City_G, Total_Simulation_Times_G, Avg_Weight_G, Candidate_G, Candidate_Num_G, Promising_City_Num_G, Weight_G, Total_thread_num, Chosen_Times_G, Pair_City_Num_G, Real_Gain_G, Gain_G, Distance_G, Virtual_City_Num, Solution_G, threadid, All_Node_G, City_Sequence_G);
		
		Total_Simulation_Times_G[threadid]++;
		//printf("Step 3 of Simulation_GPU, Action_Delta=%d. \n", Action_Delta);
		if (Action_Delta > Best_Action_Delta)
		{
			Best_Action_Delta = Action_Delta;

			Temp_Pair_Num_G[threadid] = Pair_City_Num_G[threadid];
			for (int j = 0; j < 2 * Pair_City_Num_G[threadid]; j++)
				Temp_City_Sequence_G[threadid * Virtual_City_Num + j] = City_Sequence_G[threadid * Virtual_City_Num + j];
		}

		if (Best_Action_Delta > 0)
			break;
	}

	// Restore the action with the best delta
	Pair_City_Num_G[threadid] = Temp_Pair_Num_G[threadid];
	for (int i = 0; i<2 * Pair_City_Num_G[threadid]; i++)
		City_Sequence_G[threadid * Virtual_City_Num + i] = Temp_City_Sequence_G[threadid * Virtual_City_Num + i];

	return Best_Action_Delta;
}

__device__ void Back_Propagation_GPU(int Before_Simulation_Distance, int Action_Delta, int Total_thread_num, int threadid, int *Pair_City_Num_G, int *City_Sequence_G, int Virtual_City_Num, double *Weight_G)
{
	for (int i = 0; i<Pair_City_Num_G[threadid]; i++)
	{
		int First_City = City_Sequence_G[threadid * Virtual_City_Num + 2 * i];
		int Second_City = City_Sequence_G[threadid * Virtual_City_Num + 2 * i + 1];
		int Third_City;
		if (i<Pair_City_Num_G[threadid] - 1)
			Third_City = City_Sequence_G[threadid * Virtual_City_Num + 2 * i + 2];
		else
			Third_City = City_Sequence_G[threadid * Virtual_City_Num + 0];

		if (Action_Delta >0)
		{
			double Increase_Rate = Beta*(pow(2.718, (double)(Action_Delta) / (double)(Before_Simulation_Distance)) - 1);
			Weight_G[threadid * Total_thread_num + Second_City * Virtual_City_Num + Third_City] += Increase_Rate;
			Weight_G[threadid * Total_thread_num + Third_City * Virtual_City_Num + Second_City] += Increase_Rate;
		}
	}
}

__device__ void Execute_Best_Action_GPU(int threadid, int Virtual_City_Num, int *City_Sequence_G, Struct_Node *All_Node_G, int *Pair_City_Num_G)
{
	int Begin_City = City_Sequence_G[threadid * Virtual_City_Num + 0];
	int Cur_City = City_Sequence_G[threadid * Virtual_City_Num + 1];
	All_Node_G[threadid * Virtual_City_Num + Begin_City].Next_City = Null;
	All_Node_G[threadid * Virtual_City_Num + Cur_City].Pre_City = Null;
	for (int i = 1; i<Pair_City_Num_G[threadid]; i++)
	{
		int Next_City_To_Connect = City_Sequence_G[threadid * Virtual_City_Num + 2 * i];
		int Next_City_To_Disconnect = City_Sequence_G[threadid * Virtual_City_Num + 2 * i + 1];

		Reverse_Sub_Path_GPU(Cur_City, Next_City_To_Disconnect, All_Node_G, threadid, Virtual_City_Num);

		All_Node_G[threadid * Virtual_City_Num + Cur_City].Next_City = Next_City_To_Connect;
		All_Node_G[threadid * Virtual_City_Num + Next_City_To_Connect].Pre_City = Cur_City;
		All_Node_G[threadid * Virtual_City_Num + Next_City_To_Disconnect].Pre_City = Null;

		Cur_City = Next_City_To_Disconnect;
	}
	//printf("Execute_Best_Action_GPU. \n");
	All_Node_G[threadid * Virtual_City_Num + Begin_City].Next_City = Cur_City;
	All_Node_G[threadid * Virtual_City_Num + Cur_City].Pre_City = Begin_City;
}

__global__ void MCTS_GPU(int Threshold, int Virtual_City_Num, Struct_Node *Best_All_Node_G, Struct_Node *All_Node_G, int *Distance_G,
	int *Current_Instance_Best_Distance_G, int *Temp_City_Sequence_G, int *Temp_Pair_Num_G, int randseed, int *Total_Simulation_Times_G,
	int *Probabilistic_G, int *Promising_City_G, double *Avg_Weight_G, int *Candidate_G, int *Candidate_Num_G, int *Promising_City_Num_G,
	double *Weight_G, int Total_thread_num, int *Chosen_Times_G, int *Pair_City_Num_G, int *Real_Gain_G, int *Gain_G, int *Solution_G,
	int *City_Sequence_G)
{
	int threadid = blockDim.x*blockIdx.x + threadIdx.x;
	//printf("Start to Run the MCTS_GPU. \n");
	//int Threshold = 100;
	int runNum = 0;
	while (Threshold > runNum)
	{
		//printf("Step 1 of MCTS_GPU. \n");
		int Before_Simulation_Distance = Get_Solution_Total_Distance_GPU(Distance_G, threadid, Virtual_City_Num, All_Node_G);
		//printf("Step 2 of MCTS_GPU. \n");
		int Best_Delta = Simulation_GPU(Param_H*Virtual_City_Num, Temp_City_Sequence_G, Temp_Pair_Num_G, threadid, randseed,
			Virtual_City_Num, Total_Simulation_Times_G, Probabilistic_G, Promising_City_G, Avg_Weight_G, Candidate_G, Candidate_Num_G,
			Promising_City_Num_G, Weight_G, Total_thread_num, Chosen_Times_G, Pair_City_Num_G, Real_Gain_G, Gain_G, Distance_G,
			Solution_G, All_Node_G, City_Sequence_G);
		//printf("Step 3 of MCTS_GPU. \n");
		Back_Propagation_GPU(Before_Simulation_Distance, Best_Delta, Total_thread_num, threadid, Pair_City_Num_G, City_Sequence_G,
			Virtual_City_Num, Weight_G);
		//printf("Step 4 of MCTS_GPU, Best_Delta=%d. \n", Best_Delta);
		if (Best_Delta > 0)
		{
			//printf("Step 4 of MCTS_GPU, Best_Delta=%d. \n", Best_Delta);
			//printf("Step 5 of MCTS_GPU. \n");
			Execute_Best_Action_GPU(threadid, Virtual_City_Num, City_Sequence_G, All_Node_G, Pair_City_Num_G);
			//printf("Step 6 of MCTS_GPU. \n");
			int Cur_Solution_Total_Distance = Get_Solution_Total_Distance_GPU(Distance_G, threadid, Virtual_City_Num, All_Node_G);
			//printf("Step 7 of MCTS_GPU. \n");
			if (Cur_Solution_Total_Distance < Current_Instance_Best_Distance_G[threadid])
			{
				//printf("Step 8 of MCTS_GPU, L0=%d, L1=%d, L2=%d. \n", Before_Simulation_Distance, Cur_Solution_Total_Distance, Current_Instance_Best_Distance_G[threadid]);
				Current_Instance_Best_Distance_G[threadid] = Cur_Solution_Total_Distance;
				Store_Best_Solution_GPU(threadid, Virtual_City_Num, Best_All_Node_G, All_Node_G);
				
				//printf("Update %d .\n", Current_Instance_Best_Distance_G[threadid]);
			}
		}
		else
			break;
		runNum += 1;
	}
	//printf("MCTS_GPU. \n");
	if ((double)Current_Instance_Best_Distance_G[threadid] / Magnify_Rate < 0.5)
		printf("%f\n", (double)Current_Instance_Best_Distance_G[threadid] / Magnify_Rate);
}

__device__ void Restore_Best_Solution_GPU(int threadid, int Virtual_City_Num, Struct_Node *Best_All_Node_G, Struct_Node *All_Node_G)
{
	for (int i = 0; i<Virtual_City_Num; i++)
	{
		All_Node_G[threadid * Virtual_City_Num + i].Salesman = Best_All_Node_G[threadid * Virtual_City_Num + i].Salesman;
		All_Node_G[threadid * Virtual_City_Num + i].Next_City = Best_All_Node_G[threadid * Virtual_City_Num + i].Next_City;
		All_Node_G[threadid * Virtual_City_Num + i].Pre_City = Best_All_Node_G[threadid * Virtual_City_Num + i].Pre_City;
	}
}

__device__ double Calculate_Double_Distance_GPU(int First_City, int Second_City, double *Coordinate_X_G, double *Coordinate_Y_G)
{
	return sqrt((Coordinate_X_G[First_City] - Coordinate_X_G[Second_City])*(Coordinate_X_G[First_City] - Coordinate_X_G[Second_City]) +
		(Coordinate_Y_G[First_City] - Coordinate_Y_G[Second_City])*(Coordinate_Y_G[First_City] - Coordinate_Y_G[Second_City]));
}

__device__ double Get_Current_Solution_Double_Distance_GPU(int threadid, int Virtual_City_Num, Struct_Node *All_Node_G,
 double *Coordinate_X_G, double *Coordinate_Y_G)
{
	double Current_Solution_Double_Distance = 0;
	for (int i = 0; i<Virtual_City_Num; i++)
	{
		int Temp_Next_City = All_Node_G[threadid * Virtual_City_Num + i].Next_City;
		if (Temp_Next_City != Null)
			Current_Solution_Double_Distance += Calculate_Double_Distance_GPU(i, Temp_Next_City, Coordinate_X_G, Coordinate_Y_G);
		else
		{
			printf("\nGet_Current_Solution_Double_Distance() fail!\n");
			return Inf_Cost;
		}
	}

	return Current_Solution_Double_Distance;
}

__global__ void Restore_Best_Solution_And_Calculate_Result_GPU(double *Current_Solution_Double_Distance_G, int Virtual_City_Num, 
	Struct_Node *Best_All_Node_G, Struct_Node *All_Node_G, double *Coordinate_X_G, double *Coordinate_Y_G)
{
	int threadid = blockDim.x * blockIdx.x + threadIdx.x;
	Restore_Best_Solution_GPU(threadid, Virtual_City_Num, Best_All_Node_G, All_Node_G);
	Current_Solution_Double_Distance_G[threadid] = Get_Current_Solution_Double_Distance_GPU(threadid, Virtual_City_Num,
													All_Node_G, Coordinate_X_G, Coordinate_Y_G);
	//printf("%d = Best-Result = %f \n", threadid, Current_Solution_Double_Distance_G[threadid]);
}



__global__ void test_device(int randseed, int Virtual_City_Num)
{
	int threadid = blockDim.x * blockIdx.x + threadIdx.x;
	Get_Random_Int_GPU(randseed, threadid, Virtual_City_Num);
}

extern "C" void Init_GPU(double* Edge_Heatmap_G, int *Current_Instance_Best_Distance_G, double *Weight_G, int *Chosen_Times_G,
	int *Total_Simulation_Times_G, int Virtual_City_Num, int Total_thread_num)
{
	
	MCTS_Init_GPU << <block_num, thread_per_block >> > (Edge_Heatmap_G, Current_Instance_Best_Distance_G, Weight_G, Chosen_Times_G, Total_Simulation_Times_G, Virtual_City_Num, Total_thread_num);
	
}

extern "C" void Excute_GPU(int randseed, int Threshold, int *Distance_G, Struct_Node *Best_All_Node_G, Struct_Node *All_Node_G, double* Weight_G,
	int* Chosen_Times_G, int* Total_Simulation_Times_G, int Virtual_City_Num, int Total_thread_num, int *Candidate_G, int *Candidate_Num_G,
	int *Current_Instance_Best_Distance_G, int *Temp_City_Sequence_G, int *Temp_Pair_Num_G, int *Probabilistic_G, int *Promising_City_G,
	double *Avg_Weight_G, int *Promising_City_Num_G, int *Pair_City_Num_G, int *Real_Gain_G, int *Gain_G, int *Solution_G,
	int *City_Sequence_G, double *Current_Solution_Double_Distance_G, double *Coordinate_X_G, double *Coordinate_Y_G)
{
	Local_Search_by_2Opt_Move_GPU << <block_num, thread_per_block >> > (Best_All_Node_G, Weight_G, Distance_G, Total_Simulation_Times_G,
		Virtual_City_Num, Candidate_Num_G, Candidate_G, Total_thread_num, All_Node_G, Chosen_Times_G, Current_Instance_Best_Distance_G);
	//printf("Local_Search_by_2Opt_Move_GPU. \n");
	
	MCTS_GPU << <block_num, thread_per_block >> > (Threshold, Virtual_City_Num, Best_All_Node_G, All_Node_G, Distance_G, Current_Instance_Best_Distance_G,
		Temp_City_Sequence_G, Temp_Pair_Num_G, randseed, Total_Simulation_Times_G, Probabilistic_G, Promising_City_G, Avg_Weight_G, Candidate_G,
		Candidate_Num_G, Promising_City_Num_G, Weight_G, Total_thread_num, Chosen_Times_G, Pair_City_Num_G, Real_Gain_G, Gain_G, Solution_G,
		City_Sequence_G);
	//printf("MCTS_GPU. \n");
	
	Restore_Best_Solution_And_Calculate_Result_GPU << <block_num, thread_per_block >> > (Current_Solution_Double_Distance_G, Virtual_City_Num,
		Best_All_Node_G, All_Node_G, Coordinate_X_G, Coordinate_Y_G);
	//printf("Restore_Best_Solution_And_Calculate_Result_GPU. \n");
}