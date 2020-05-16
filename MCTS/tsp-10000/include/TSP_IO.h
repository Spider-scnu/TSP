#ifndef TSP_IO_H
#define TSP_IO_H


#include "kernel.h"
//using namespace std;

#define Max_Inst_Num       16
#define Max_City_Num       10000 

//Hyper parameters 
double Param_T = 0.006;  	 //used to control the termination condition 0.006


							 //#define Default_Random_Seed  (unsigned)time(NULL);
#define Default_Random_Seed  489663920;    
unsigned Random_Seed = Default_Random_Seed;

typedef int    Distance_Type;

//Information of the 30,000 TSP20-50-100 instances
#define Total_Instance_Num	   16
char *Input_File_Name;
int Temp_City_Num;

double Stored_Coordinates_X[Max_Inst_Num][Max_City_Num];
double Stored_Coordinates_Y[Max_Inst_Num][Max_City_Num];
int    Stored_Opt_Solution[Max_Inst_Num][Max_City_Num];
int Inst_Num_Per_Batch = 1 + Total_Instance_Num / 2;
int Index_In_Batch;


//Used to store the statistic results
char *Statistics_File_Name;
int    Test_Inst_Num = 0;
int    Beat_Best_Known_Times = 0;
int    Match_Best_Known_Times = 0;
int    Miss_Best_Known_Times = 0;
double Sum_Opt_Distance = 0;
double Sum_My_Distance = 0;
double Sum_Gap = 0;

//Used to store the input information of a given instance
int City_Num;
int Start_City;
int Salesman_Num;            //This program was proposed for the multiple TSP. If Salesman_Num=1, it reduces to the TSP
int Virtual_City_Num;
double *Coordinate_X;
double *Coordinate_Y;
Distance_Type **Distance;


double Current_Instance_Begin_Time;
Distance_Type Current_Instance_Best_Distance;

Struct_Node *All_Node;        //Store the incumbent tour
Struct_Node *Best_All_Node;   //Store the best found tour

							  //Used to store a solution in an array 
int *Solution;

//Used to store a set of candidate neighbors of each city
int *Candidate_Num;
int **Candidate;
bool *If_City_Selected;

//Used to store the information of an action
int Pair_City_Num;
int Temp_Pair_Num;
int *City_Sequence;
int *Temp_City_Sequence;
Distance_Type *Gain;
Distance_Type *Real_Gain;

//Used in MCTS
double **Weight;
double Avg_Weight;
int    **Chosen_Times;
int    *Promising_City;
int    *Probabilistic;
int    Promising_City_Num;
int    Total_Simulation_Times;

Distance_Type Get_Solution_Total_Distance();
void 		  Convert_Solution_To_All_Node();

void Allocate_Memory(int City_Num)
{
	Coordinate_X = new double[City_Num];
	Coordinate_Y = new double[City_Num];

	Distance = new Distance_Type *[City_Num];
	for (int i = 0; i<City_Num; i++)
		Distance[i] = new Distance_Type[City_Num];

	All_Node = new Struct_Node[City_Num];
	Best_All_Node = new Struct_Node[City_Num];
	Solution = new int[City_Num];

	Candidate_Num = new int[City_Num];
	Candidate = new int *[City_Num];
	for (int i = 0; i<City_Num; i++)
		Candidate[i] = new int[Max_Candidate_Num];
	If_City_Selected = new bool[City_Num];

	City_Sequence = new int[City_Num];
	Temp_City_Sequence = new int[City_Num];
	Gain = new Distance_Type[2 * City_Num];
	Real_Gain = new Distance_Type[2 * City_Num];

	Weight = new double *[City_Num];
	for (int i = 0; i<City_Num; i++)
		Weight[i] = new double[City_Num];

	Chosen_Times = new int *[City_Num];
	for (int i = 0; i<City_Num; i++)
		Chosen_Times[i] = new int[City_Num];

	Promising_City = new int[City_Num];
	Probabilistic = new int[City_Num];
}

void Release_Memory(int City_Num)
{
	delete[]Coordinate_X;
	delete[]Coordinate_Y;

	for (int i = 0; i<City_Num; i++)
		delete[]Distance[i];
	delete[]Distance;

	delete[]All_Node;
	delete[]Best_All_Node;
	delete[]Solution;

	delete[]Candidate_Num;
	for (int i = 0; i<City_Num; i++)
		delete[]Candidate[i];
	delete[]Candidate;
	delete[]If_City_Selected;

	delete[]City_Sequence;
	delete[]Temp_City_Sequence;
	delete[]Gain;
	delete[]Real_Gain;

	for (int i = 0; i<City_Num; i++)
		delete[]Weight[i];
	delete[]Weight;

	for (int i = 0; i<City_Num; i++)
		delete[]Chosen_Times[i];
	delete[]Chosen_Times;

	delete[]Promising_City;
	delete[]Probabilistic;
}

//For TSP20-50-100 instances 
bool Fetch_Stored_Instance_Info(int Inst_Index)
{
	City_Num = Temp_City_Num;
	Start_City = 0;
	Salesman_Num = 1;
	Virtual_City_Num = City_Num + Salesman_Num - 1;

	Allocate_Memory(Virtual_City_Num);

	for (int i = 0; i<City_Num; i++)
	{
		Coordinate_X[i] = Stored_Coordinates_X[Inst_Index][i] * Magnify_Rate;
		Coordinate_Y[i] = Stored_Coordinates_Y[Inst_Index][i] * Magnify_Rate;
	}

	return true;
}

bool Save_Solution(char *Output_File_Name)
{
	FILE *fp;
	fp = fopen(Output_File_Name, "w+");

	fprintf(fp, "%d\n", City_Num);
	fprintf(fp, "%d\n", Start_City + 1);
	fprintf(fp, "%d\n", Salesman_Num);

	int Cur_City = Start_City;
	do
	{
		fprintf(fp, "%d\n ", Cur_City + 1);
		Cur_City = All_Node[Cur_City].Next_City;
	} while (Cur_City != Null && Cur_City != Start_City);

	fprintf(fp, "%d\n ", Get_Solution_Total_Distance());

	fclose(fp);
	return true;
}

bool Read_Solution(char *Input_File_Name)
{
	ifstream FIC;
	FIC.open(Input_File_Name);

	if (FIC.fail())
	{
		cout << "Read_Solution() fail! Fail to open the input file " << Input_File_Name << endl;
		return false;
	}

	FIC >> City_Num;
	FIC >> Start_City;
	Start_City--;
	FIC >> Salesman_Num;

	Virtual_City_Num = City_Num + Salesman_Num - 1;

	int Cur_City;
	for (int i = 0; i<Virtual_City_Num; i++)
	{
		FIC >> Cur_City;
		Solution[i] = Cur_City - 1;
	}
	FIC.close();

	Convert_Solution_To_All_Node();
	//cout<<"Read_Solution() finished. Begin to search. "<<endl<<endl; 
	return true;
}

//Print the cities of a solution one by one 
void Print_TSP_Tour(int Begin_City)
{
	cout << "\nThe current tour is:" << endl;
	int Cur_City = Begin_City;
	do
	{
		printf("%d ", Cur_City + 1);
		Cur_City = All_Node[Cur_City].Next_City;
	} while (Cur_City != Null && Cur_City != Begin_City);
}

#endif // !TSP_IO_H
