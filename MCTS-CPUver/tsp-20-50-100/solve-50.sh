#!/bin/bash
# author: 
rm -r ./code/TSP.o
rm -r ./test
make

Total_Instance_Num=(10000)
Temp_City_Num=(50)
threads=$1
Inst_Num_Per_Batch=$((1+$Total_Instance_Num/$threads))

tsp=("./tsp50_test_concorde.txt")
j=0
sources50=("./results/50/result_1.txt" "./results/50/result_2.txt" "./results/50/result_3.txt" "./results/50/result_4.txt" "./results/50/result_5.txt" "./results/50/result_6.txt" "./results/50/result_7.txt" "./results/50/result_8.txt" "./results/50/result_9.txt" "./results/50/result_10.txt" "./results/50/result_11.txt" "./results/50/result_12.txt" "./results/50/result_13.txt" "./results/50/result_14.txt" "./results/50/result_15.txt" "./results/50/result_16.txt" "./results/50/result_17.txt" "./results/50/result_18.txt" "./results/50/result_19.txt" "./results/50/result_20.txt" "./results/50/result_21.txt" "./results/50/result_22.txt" "./results/50/result_23.txt" "./results/50/result_24.txt" "./results/50/result_25.txt" "./results/50/result_26.txt" "./results/50/result_27.txt" "./results/50/result_28.txt" "./results/50/result_29.txt" "./results/50/result_30.txt" "./results/50/result_31.txt" "./results/50/result_32.txt")
for ((i=0;i<$threads;i++));do
{
	./test $i ${sources50[i]} ${tsp[j]} ${Temp_City_Num[j]} ${Inst_Num_Per_Batch}
}&
done
wait

echo "Done."