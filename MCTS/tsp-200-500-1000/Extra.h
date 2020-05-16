#ifndef Extra_H
#define Extra_H

#include "kernel.h"
#include "include/TSP_IO.h"
#include "include/TSP_Init.h"

Struct_Node *All_Node_H;
int Total_thread_num;

double *Weight_G, *Weight_H;
int *Chosen_Times_G, *Chosen_Times_H;
int *Total_Simulation_Times_G, *Total_Simulation_Times_H;
Struct_Node *All_Node_G;
int *Current_Instance_Best_Distance_H;
int *Current_Instance_Best_Distance_G;
int *Candidate_Num_H;
int *Candidate_Num_G;
int *Candidate_H;
int *Candidate_G;
int *Distance_H;
int *Distance_G;
Struct_Node *Best_All_Node_H;
Struct_Node *Best_All_Node_G;
int *Solution_H;
int *Solution_G;
int *City_Sequence_H;
int *City_Sequence_G;
int *Real_Gain_H, *Gain_H;
int *Real_Gain_G, *Gain_G;
int *Pair_City_Num_H;
int *Pair_City_Num_G;
double *Avg_Weight_H;
double *Avg_Weight_G;
int *Promising_City_Num_H;
int *Promising_City_Num_G;
int *Promising_City_H;
int *Promising_City_G;
int *Probabilistic_H;
int *Probabilistic_G;
int *Temp_Pair_Num_H;
int *Temp_Pair_Num_G;
int *Temp_City_Sequence_H;
int *Temp_City_Sequence_G;
double *Current_Solution_Double_Distance_H;
double *Current_Solution_Double_Distance_G;
double *Coordinate_X_H;
double *Coordinate_X_G;
double *Coordinate_Y_H;
double *Coordinate_Y_G;

void Copy_AllNode_to_All_Node_H(int threadid)
{
	for (int i = 0; i < Virtual_City_Num; i++)
	{
		All_Node_H[threadid*Virtual_City_Num + i].Salesman = All_Node[i].Salesman;
		All_Node_H[threadid*Virtual_City_Num + i].Next_City = All_Node[i].Next_City;
		All_Node_H[threadid*Virtual_City_Num + i].Pre_City = All_Node[i].Pre_City;
	}
}

void Copy_Candidate_to_Candidate_H(int threadid)
{
	for (int i = 0; i < Virtual_City_Num; i++)
	{
		for (int j = 0; j < Max_Candidate_Num; j++)
		{
			Candidate_H[threadid*Total_thread_num + i*Max_Candidate_Num + j] = Candidate[i][j];
		}
	}
}

void Copy_Distance_to_Distance_H()
{
	for (int i = 0; i < Virtual_City_Num; i++)
	{
		for (int j = 0; j < Virtual_City_Num; j++)
		{
			Distance_H[i*Virtual_City_Num + j] = Distance[i][j];
		}
	}
}

void Copy_Coordinate_to_Coordinate_H()
{
	for (int i = 0; i < Virtual_City_Num; i++)
	{
		Coordinate_X_H[i] = Coordinate_X[i];
		Coordinate_Y_H[i] = Coordinate_Y[i];
	}
}

void Copy_Data_to_GPU()
{
	for (int i = 0; i < Total_thread_num; i++)
	{
		Copy_Candidate_to_Candidate_H(i);
	}
	Copy_Distance_to_Distance_H();
	Copy_Coordinate_to_Coordinate_H();
}

void Generate_Multi_Solution()
{
	for (int i = 0; i < Total_thread_num; i++)
	{
		Generate_Initial_Solution();
		Copy_AllNode_to_All_Node_H(i);
	}
}

void Print_Solution(int thread_index)
{
	/*int Cur_City = 0, Next_City;
	cout << Cur_City << " ";
	while (true)
	{
	Next_City = All_Node_H[thread_index * Virtual_City_Num + Cur_City].Next_City;
	if (Next_City == 0)
	break;
	cout << Next_City << " ";
	Cur_City = Next_City;
	}
	cout << endl;*/

	for (int i = 0; i < Virtual_City_Num; i++)
	{
		for (int j = 0; j < Max_Candidate_Num; j++)
		{
			cout << Candidate_H[thread_index*Total_thread_num + i*Max_Candidate_Num + j] << " ";
		}
		cout << endl;
	}
	cout << endl;
	cout << "-----------------------------\n";
}

void Print_Multi_Solution()
{
	for (int i = 0; i < Total_thread_num; i++)
	{
		Print_Solution(i);
	}
}

void cuda_mem_malloc()
{
	Total_thread_num = thread_per_block*block_num;

	All_Node_H = new Struct_Node[Total_thread_num * Virtual_City_Num];
	Weight_H = new double[Total_thread_num * Virtual_City_Num * Virtual_City_Num];
	Chosen_Times_H = new int[Total_thread_num * Virtual_City_Num * Virtual_City_Num];
	Total_Simulation_Times_H = new int[Total_thread_num];
	Current_Instance_Best_Distance_H = new int[Total_thread_num];
	Candidate_Num_H = new int[Total_thread_num * Virtual_City_Num];
	Candidate_H = new int[Total_thread_num * Virtual_City_Num * Max_Candidate_Num];
	Distance_H = new int[Virtual_City_Num * Virtual_City_Num];
	Best_All_Node_H = new Struct_Node[Total_thread_num * Virtual_City_Num];
	Solution_H = new int[Total_thread_num * Virtual_City_Num];
	City_Sequence_H = new int[Total_thread_num * Virtual_City_Num];
	Real_Gain_H = new int[Total_thread_num * 2 * Virtual_City_Num];
	Gain_H = new int[Total_thread_num * 2 * Virtual_City_Num];
	Pair_City_Num_H = new int[Total_thread_num];
	Avg_Weight_H = new double[Total_thread_num];
	Promising_City_Num_H = new int[Total_thread_num];
	Promising_City_H = new int[Total_thread_num * Virtual_City_Num];
	Probabilistic_H = new int[Total_thread_num * Virtual_City_Num];
	Temp_Pair_Num_H = new int[Total_thread_num];
	Temp_City_Sequence_H = new int[Total_thread_num * Virtual_City_Num];
	Current_Solution_Double_Distance_H = new double[Total_thread_num];
	Coordinate_X_H = new double[Virtual_City_Num];
	Coordinate_Y_H = new double[Virtual_City_Num];

	cudaMalloc((void**)&Weight_G, Total_thread_num * Virtual_City_Num * Virtual_City_Num * sizeof(double));
	cudaMalloc((void**)&Chosen_Times_G, Total_thread_num * Virtual_City_Num * Virtual_City_Num * sizeof(int));
	cudaMalloc((void**)&Total_Simulation_Times_G, Total_thread_num * sizeof(int));
	cudaMalloc((void**)&All_Node_G, Total_thread_num * Virtual_City_Num * sizeof(Struct_Node));
	cudaMalloc((void**)&Current_Instance_Best_Distance_G, Total_thread_num * sizeof(int));
	cudaMalloc((void**)&Candidate_Num_G, Total_thread_num * Virtual_City_Num * sizeof(int));
	cudaMalloc((void**)&Candidate_G, Total_thread_num * Virtual_City_Num * Max_Candidate_Num * sizeof(int));
	cudaMalloc((void**)&Distance_G, Virtual_City_Num * Virtual_City_Num * sizeof(int));
	cudaMalloc((void**)&Best_All_Node_G, Total_thread_num * Virtual_City_Num * sizeof(Struct_Node));
	cudaMalloc((void**)&Solution_G, Total_thread_num * Virtual_City_Num * sizeof(int));
	cudaMalloc((void**)&City_Sequence_G, Total_thread_num * Virtual_City_Num * sizeof(int));
	cudaMalloc((void**)&Real_Gain_G, Total_thread_num * 2 * Virtual_City_Num * sizeof(int));
	cudaMalloc((void**)&Gain_G, Total_thread_num * 2 * Virtual_City_Num * sizeof(int));
	cudaMalloc((void**)&Pair_City_Num_G, Total_thread_num * sizeof(int));
	cudaMalloc((void**)&Avg_Weight_G, Total_thread_num * sizeof(double));
	cudaMalloc((void**)&Promising_City_Num_G, Total_thread_num * sizeof(int));
	cudaMalloc((void**)&Promising_City_G, Total_thread_num * Virtual_City_Num * sizeof(int));
	cudaMalloc((void**)&Probabilistic_G, Total_thread_num * Virtual_City_Num * sizeof(int));
	cudaMalloc((void**)&Temp_Pair_Num_G, Total_thread_num * sizeof(int));
	cudaMalloc((void**)&Temp_City_Sequence_G, Total_thread_num * Virtual_City_Num * sizeof(int));
	cudaMalloc((void**)&Current_Solution_Double_Distance_G, Total_thread_num * sizeof(double));
	cudaMalloc((void**)&Coordinate_X_G, Virtual_City_Num * sizeof(double));
	cudaMalloc((void**)&Coordinate_Y_G, Virtual_City_Num * sizeof(double));

}

void Transfer_All_Node_to_GPU()
{
	cudaMemcpy((void*)All_Node_G, (void *)All_Node_H, Total_thread_num * Virtual_City_Num * sizeof(Struct_Node), cudaMemcpyHostToDevice);
}

void Transfer_CPU_to_GPU()
{
	cudaMemcpy((void*)Weight_G, (void *)Weight_H, Total_thread_num * Virtual_City_Num * Virtual_City_Num * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy((void*)Chosen_Times_G, (void *)Chosen_Times_H, Total_thread_num * Virtual_City_Num * Virtual_City_Num * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy((void*)Total_Simulation_Times_G, (void *)Total_Simulation_Times_H, Total_thread_num * sizeof(int), cudaMemcpyHostToDevice);
	//cudaMemcpy((void*)All_Node_G, (void *)All_Node_H, Total_thread_num * Virtual_City_Num * sizeof(Struct_Node), cudaMemcpyHostToDevice);
	cudaMemcpy((void*)Current_Instance_Best_Distance_G, (void *)Current_Instance_Best_Distance_H, Total_thread_num * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy((void*)Candidate_Num_G, (void *)Candidate_Num_H, Total_thread_num * Virtual_City_Num * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy((void*)Candidate_G, (void *)Candidate_H, Total_thread_num * Virtual_City_Num * Max_Candidate_Num * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy((void*)Distance_G, (void *)Distance_H, Virtual_City_Num * Virtual_City_Num * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy((void*)Best_All_Node_G, (void *)Best_All_Node_H, Total_thread_num * Virtual_City_Num * sizeof(Struct_Node), cudaMemcpyHostToDevice);
	cudaMemcpy((void*)Solution_G, (void *)Solution_H, Total_thread_num * Virtual_City_Num * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy((void*)City_Sequence_G, (void *)City_Sequence_H, Total_thread_num * Virtual_City_Num * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy((void*)Real_Gain_G, (void *)Real_Gain_H, Total_thread_num * 2 * Virtual_City_Num * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy((void*)Gain_G, (void *)Gain_H, Total_thread_num * 2 * Virtual_City_Num * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy((void*)Pair_City_Num_G, (void *)Pair_City_Num_H, Total_thread_num * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy((void*)Avg_Weight_G, (void *)Avg_Weight_H, Total_thread_num * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy((void*)Promising_City_Num_G, (void *)Promising_City_Num_H, Total_thread_num * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy((void*)Promising_City_G, (void *)Promising_City_H, Total_thread_num * Virtual_City_Num * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy((void*)Probabilistic_G, (void *)Probabilistic_H, Total_thread_num * Virtual_City_Num * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy((void*)Temp_Pair_Num_G, (void *)Temp_Pair_Num_H, Total_thread_num * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy((void*)Temp_City_Sequence_G, (void *)Temp_City_Sequence_H, Total_thread_num * Virtual_City_Num * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy((void*)Current_Solution_Double_Distance_G, (void *)Current_Solution_Double_Distance_H, Total_thread_num * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy((void*)Coordinate_X_G, (void *)Coordinate_X_H, Virtual_City_Num * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy((void*)Coordinate_Y_G, (void *)Coordinate_Y_H, Virtual_City_Num * sizeof(double), cudaMemcpyHostToDevice);

}

void deal_with_cuda_result()
{
	cudaMemcpy((void*)Weight_H, (void *)Weight_G, Total_thread_num * Virtual_City_Num * Virtual_City_Num * sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy((void*)Chosen_Times_H, (void *)Chosen_Times_G, Total_thread_num * Virtual_City_Num * Virtual_City_Num * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy((void*)Total_Simulation_Times_H, (void *)Total_Simulation_Times_G, Total_thread_num * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy((void*)All_Node_H, (void *)All_Node_G, Total_thread_num * Virtual_City_Num * sizeof(Struct_Node), cudaMemcpyDeviceToHost);
	cudaMemcpy((void*)Current_Instance_Best_Distance_H, (void *)Current_Instance_Best_Distance_G, Total_thread_num * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy((void*)Candidate_Num_H, (void *)Candidate_Num_G, Total_thread_num * Virtual_City_Num * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy((void*)Candidate_H, (void *)Candidate_G, Total_thread_num * Virtual_City_Num * Max_Candidate_Num * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy((void*)Distance_H, (void *)Distance_G, Total_thread_num * Virtual_City_Num * Max_Candidate_Num * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy((void*)Best_All_Node_H, (void *)Best_All_Node_G, Total_thread_num * Virtual_City_Num * sizeof(Struct_Node), cudaMemcpyDeviceToHost);
	cudaMemcpy((void*)Solution_H, (void *)Solution_G, Total_thread_num * Virtual_City_Num * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy((void*)City_Sequence_H, (void *)City_Sequence_G, Total_thread_num * Virtual_City_Num * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy((void*)Real_Gain_H, (void *)Real_Gain_G, Total_thread_num * 2 * Virtual_City_Num * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy((void*)Gain_H, (void *)Gain_G, Total_thread_num * 2 * Virtual_City_Num * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy((void*)Pair_City_Num_H, (void *)Pair_City_Num_G, Total_thread_num * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy((void*)Avg_Weight_H, (void *)Avg_Weight_G, Total_thread_num * sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy((void*)Promising_City_Num_H, (void *)Promising_City_Num_G, Total_thread_num * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy((void*)Promising_City_H, (void *)Promising_City_G, Total_thread_num * Virtual_City_Num * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy((void*)Probabilistic_H, (void *)Probabilistic_G, Total_thread_num * Virtual_City_Num * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy((void*)Temp_Pair_Num_H, (void *)Temp_Pair_Num_G, Total_thread_num * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy((void*)Temp_City_Sequence_H, (void *)Temp_City_Sequence_G, Total_thread_num * Virtual_City_Num * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy((void*)Current_Solution_Double_Distance_H, (void *)Current_Solution_Double_Distance_G, Total_thread_num * sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy((void*)Coordinate_X_H, (void *)Coordinate_X_G, Virtual_City_Num * sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy((void*)Coordinate_Y_H, (void *)Coordinate_Y_G, Virtual_City_Num * sizeof(double), cudaMemcpyDeviceToHost);

	cudaFree(Weight_G);
	cudaFree(Chosen_Times_G);
	cudaFree(Total_Simulation_Times_G);
	cudaFree(All_Node_G);
	cudaFree(Current_Instance_Best_Distance_G);
	cudaFree(Candidate_Num_G);
	cudaFree(Candidate_G);
	cudaFree(Distance_G);
	cudaFree(Best_All_Node_G);
	cudaFree(Solution_G);
	cudaFree(City_Sequence_G);
	cudaFree(Real_Gain_G);
	cudaFree(Gain_G);
	cudaFree(Pair_City_Num_G);
	cudaFree(Avg_Weight_G);
	cudaFree(Promising_City_Num_G);
	cudaFree(Promising_City_G);
	cudaFree(Probabilistic_G);
	cudaFree(Temp_Pair_Num_G);
	cudaFree(Temp_City_Sequence_G);
	cudaFree(Current_Solution_Double_Distance_G);
	cudaFree(Coordinate_X_G);
	cudaFree(Coordinate_Y_G);

	//delete[] Total_Simulation_Times_H;
}

int* h_test, *d_test;
void Test_Mem_malloc()
{
	h_test = new int[4];
	cudaMalloc((void**)&d_test, 4 * sizeof(int));
	cudaMemcpy((void*)d_test, (void *)h_test, 4 * sizeof(int), cudaMemcpyHostToDevice);
}

void deal_with_result()
{
	cudaMemcpy((void*)h_test, (void *)d_test, 4 * sizeof(int), cudaMemcpyDeviceToHost);
	cudaFree(d_test);
}

double Get_Current_Best_Double_Distance()
{
	double Current_Best_Double_Distance = Inf_Cost;
	for (int i = 0; i < Total_thread_num; i++)
	{
		if (Current_Best_Double_Distance > Current_Solution_Double_Distance_H[i])
			Current_Best_Double_Distance = Current_Solution_Double_Distance_H[i];
	}
	return Current_Best_Double_Distance/* / 2.0f*/;
}

void Free_H_Mem()
{
	delete[] Weight_H;
	delete[] Chosen_Times_H;
	delete[] Total_Simulation_Times_H;
	delete[] All_Node_H;
	delete[] Current_Instance_Best_Distance_H;
	delete[] Candidate_Num_H;
	delete[] Candidate_H;
	delete[] Distance_H;
	delete[] Best_All_Node_H;
	delete[] Solution_H;
	delete[] City_Sequence_H;
	delete[] Real_Gain_H;
	delete[] Gain_H;
	delete[] Pair_City_Num_H;
	delete[] Avg_Weight_H;
	delete[] Promising_City_Num_H;
	delete[] Promising_City_H;
	delete[] Probabilistic_H;
	delete[] Temp_Pair_Num_H;
	delete[] Temp_City_Sequence_H;
	delete[] Current_Solution_Double_Distance_H;
	delete[] Coordinate_X_H;
	delete[] Coordinate_Y_H;
}

void Markov_Decision_Process_GPU()
{
	cuda_mem_malloc();//GPU
	Copy_Data_to_GPU();
	Transfer_CPU_to_GPU();
	Init_GPU(Current_Instance_Best_Distance_G, Weight_G, Chosen_Times_G, Total_Simulation_Times_G, Virtual_City_Num, Total_thread_num);
	cudaDeviceSynchronize();
	/*while (((double)clock() - Current_Instance_Begin_Time) / CLOCKS_PER_SEC<Param_T*Virtual_City_Num)
	{*/
	Generate_Multi_Solution();
	Transfer_All_Node_to_GPU();

	int randseed = rand() % 10000;
	Excute_GPU(randseed, Distance_G, Best_All_Node_G, All_Node_G, Weight_G, Chosen_Times_G, Total_Simulation_Times_G, Virtual_City_Num, Total_thread_num, Candidate_G, Candidate_Num_G, Current_Instance_Best_Distance_G, Temp_City_Sequence_G, Temp_Pair_Num_G, Probabilistic_G, Promising_City_G, Avg_Weight_G, Promising_City_Num_G, Pair_City_Num_G, Real_Gain_G, Gain_G, Solution_G, City_Sequence_G, Current_Solution_Double_Distance_G, Coordinate_X_G, Coordinate_Y_G);
	//}
	//Markov_Decision_Process();
	deal_with_cuda_result();
}

#endif // !Extra_H
