#!/bin/bash


my_function () {
    temp_string="${project_name[$dir1]}\t${x}\t${formulas[$formula]}\t"
    temp_string_2=$(tr -d '\n' < ${project_name[$dir1]}/${x}/sbfl/formula-${formulas[$formula]}/totaldefn-tests/scoring-first/score.txt)
	temp_string="${temp_string}\t${temp_string_2}"
    temp_string_2=$(tr -d '\n' < ${project_name[$dir1]}/${x}/sbfl/formula-${formulas[$formula]}/totaldefn-tests/scoring-median/score.txt)
	temp_string="${temp_string}\t${temp_string_2}"
    temp_string_2=$(tr -d '\n' < ${project_name[$dir1]}/${x}/sbfl/formula-${formulas[$formula]}/totaldefn-tests/scoring-last/score.txt)
	temp_string="${temp_string}\t${temp_string_2}"
	echo -e "${temp_string}" >> combine_exam_sbfl.txt	
	
}

project_name=("Chart" "Closure" "Lang" "Math" "Mockito" "Time")
project_number=(26 133 65 106 38 27)
formulas=("barinel" "dstar2" "ochiai" "opt2" "tarantula")
x=1
pwd=$(pwd)
echo -e 'project\tbug\tformula\texam_best_case_b\texam_best_case_f\texam_average_case_b\texam_average_case_f\texam_worst_case_b\texam_worst_case_f' > combine_exam_sbfl.txt
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

