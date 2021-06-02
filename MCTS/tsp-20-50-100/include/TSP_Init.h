#ifndef TSP_Init_H
#define TSP_Init_H

#include "TSP_Basic_Functions.h"

// The following funcitons are used to randomly generate an initial solution 
// Starting from an arbitrarily chosen city, it iteratively selects a random city until forming a TSP tour
int Select_Random_City(int Cur_City)
{
	int Random_Index = Get_Random_Int(Virtual_City_Num);
	for (int i = 0; i<Virtual_City_Num; i++)
	{
		int Candidate_City = (Random_Index + i) % Virtual_City_Num;
		if (!If_City_Selected[Candidate_City])
			return Candidate_City;
	}

	return Null;
}

bool Generate_Initial_Solution()
{
	for (int i = 0; i<Virtual_City_Num; i++)
	{
		Solution[i] = Null;
		If_City_Selected[i] = false;
	}

	int Selected_City_Num = 0;
	int Cur_City = Start_City;
	int Next_City;

	Solution[Selected_City_Num++] = Cur_City;
	If_City_Selected[Cur_City] = true;
	//cout<<"Route : "<<Cur_City;
	do
	{
		Next_City = Select_Random_City(Cur_City);
		if (Next_City != Null)
		{
			Solution[Selected_City_Num++] = Next_City;
			If_City_Selected[Next_City] = true;
			Cur_City = Next_City;
			//cout<<" - "<<Cur_City;
		}
	} while (Next_City != Null);
	//cout<<endl;
	Convert_Solution_To_All_Node();
	return Check_Solution_Feasible();
}


#endif // !TSP_Init_H
