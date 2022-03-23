#!/bin/bash


my_function () {
	
	mkdir -p "${project_name[$dir1]}/${project_name[$dir1]}-${x}"
	
}

second_function () {
    python XAI4FL.py "${pwd}/${x}/spectra" "${pwd}/${x}/matrix" "${pwd}/${project_name[$dir1]}/${project_name[$dir1]}-${x}/${project_name[$dir1]}-${x}"
	pwd

}
project_name=("Math")
project_number=(103)
x=80
pwd=$(pwd)
for dir1 in "${!project_name[@]}"; do
  mkdir -p "\${project_name[$dir1]}"
  echo "$dir1"
  echo "${project_name[$dir1]}"
  echo "${project_number[$dir1]}"
  while [ $x -le ${project_number[$dir1]} ]
  do
     my_function
     second_function
     x=$(( $x + 1 ))
  done
done

