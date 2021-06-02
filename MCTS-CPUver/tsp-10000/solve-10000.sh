#!/bin/bash
# author: 
rm -r ./code/TSP.o
rm -r ./test
make

Total_Instance_Num=(16)
threads=$1
Temp_City_Num=(10000)
Inst_Num_Per_Batch=$(($Total_Instance_Num/$threads))

tsp=("./tsp10000_test_concorde.txt")
j=0
sources10000=("./results/10000/result_1.txt")
for ((i=0;i<$threads;i++));do
{
	./test $i ${sources10000[i]} ${tsp[j]} ${Temp_City_Num[j]} ${Inst_Num_Per_Batch}
}&
done
wait
echo "Done."