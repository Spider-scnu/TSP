#!/bin/bash
# author: 
rm -r ./mcts-code/TSP.o
rm -r ./test
make

Total_Instance_Num=(128)
Temp_City_Num=(500)
threads=$1
Inst_Num_Per_Batch=$(($Total_Instance_Num/$threads))

tsp=("./tsp500_test_concorde.txt")
j=0
sources500=("./results/500/result_1.txt" "./results/500/result_2.txt" "./results/500/result_3.txt" "./results/500/result_4.txt" "./results/500/result_5.txt" "./results/500/result_6.txt" "./results/500/result_7.txt" "./results/500/result_8.txt")
threads=$1
for ((i=0;i<$threads;i++));do
{
	./test $i ${sources500[i]} ${tsp[j]} ${Temp_City_Num[j]} ${Inst_Num_Per_Batch}
}&
done
wait

echo "Done."