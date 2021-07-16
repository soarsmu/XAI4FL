#!/bin/bash


my_function () {
    temp_string="${project_name[$dir1]}\t${x}\t${formulas[$formula]}\t"
    temp_string_2=$(tr -d '\n' < ${project_name[$dir1]}/${x}/sbfl/formula-${formulas[$formula]}/totaldefn-tests/scoring-first/rank.txt)
	temp_string="${temp_string}\t${temp_string_2}"
    temp_string_2=$(tr -d '\n' < ${project_name[$dir1]}/${x}/sbfl/formula-${formulas[$formula]}/totaldefn-tests/scoring-median/rank.txt)
	temp_string="${temp_string}\t${temp_string_2}"
    temp_string_2=$(tr -d '\n' < ${project_name[$dir1]}/${x}/sbfl/formula-${formulas[$formula]}/totaldefn-tests/scoring-last/rank.txt)
	temp_string="${temp_string}\t${temp_string_2}"
	echo -e "${temp_string}" >> combine_sbfl.txt	
	
}

project_name=("Chart" "Closure" "Lang" "Math" "Mockito" "Time")
project_number=(26 133 65 106 38 27)
formulas=("barinel" "dstar2" "ochiai" "opt2" "tarantula")
x=1
pwd=$(pwd)
echo -e 'project\tbug\tformula\trank_best_case\trank_average_case\trank_worst_case' > combine_sbfl.txt
for dir1 in "${!project_name[@]}"; do
  echo "$dir1"
  echo "${project_name[$dir1]}"
  echo "${project_number[$dir1]}"
  x=1
  while [ $x -le ${project_number[$dir1]} ]
  do
     for formula in "${!formulas[@]}"; do
		my_function
	 done
     x=$(( $x + 1 ))
  done
done

