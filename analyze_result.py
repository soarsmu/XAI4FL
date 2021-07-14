import os                                                                                                             
import csv

import pandas as pd
import sys

if len(sys.argv) > 1:
	df = pd.read_csv(sys.argv[1],sep='\t') 
else:
	df = pd.read_csv('output.txt',sep='\t') 
print(df)
colnames = {'project_id': 0, 'bug_id':1, 'temp_len':2, 'sloc':3, 'sloc_total':4, 'buggy_line':5, 'is_omission':6, 'is_not_found':7, 'shap_max':8, 'shap_min':9, 'shap_mean':10, 'buggy_line_max':11, 'buggy_line_min':12, 'buggy_line_mean':13}

data = df.to_numpy()
print(data)
all_result = [['project_id', 'bug_id', 'rank_mean_best', 'rank_min_best', 'rank_max_best', 'rank_mean_average', 'rank_min_average', 'rank_max_average', 'rank_mean_worst', 'rank_min_worst', 'rank_max_worst', 'exam_mean_best', 'exam_min_best', 'exam_max_best', 'exam_mean_average', 'exam_min_average', 'exam_max_average', 'exam_mean_worst', 'exam_min_worst', 'exam_max_worst' ]]
i = 0
top_200_mean, top_200_min, top_200_max = 0, 0, 0
top_10_mean, top_10_min, top_10_max = 0, 0, 0
top_5_mean, top_5_min, top_5_max = 0, 0, 0
while i < len(data):
	#print(data[i])
	project = data[i][0]
	bug = data[i][1]
	sloc = data[i][colnames['sloc']]
	temp_len = data[i][colnames['temp_len']]

	temp_mean_best = []
	temp_min_best = []
	temp_max_best = []
	# print(data[i])
	while i < len(data) and data[i][0] == project and data[i][1] == bug:
		if data[i][colnames['is_not_found']] == '1':
			not_found = ((sloc - temp_len) / 2) + temp_len
			temp_mean_best.append(not_found)
			temp_min_best.append(not_found)
			temp_max_best.append(not_found)
		else:
			temp_mean_best.append(float(data[i][colnames['shap_mean']]))
			temp_min_best.append(float(data[i][colnames['shap_min']]))
			temp_max_best.append(float(data[i][colnames['shap_max']]))
		i += 1
	#cek ulang bagian sortingnya
	temp_mean_best.sort()
	temp_min_best.sort()
	temp_max_best.sort()


	#print(temp_tarantula_best)
	size = len(temp_mean_best)
	if size == 1:
		half_size = 1
	else:
		half_size = int(size / 2)

	if temp_mean_best[0] <= 5:
		top_5_mean += 1
		top_10_mean += 1
		top_200_mean += 1
	elif temp_mean_best[0] <= 10:
		top_10_mean += 1
		top_200_mean += 1
	elif temp_mean_best[0] <= 200:
		top_200_mean += 1


	if temp_min_best[0] <= 5:
		top_5_min += 1
		top_10_min += 1
		top_200_min += 1
	elif temp_min_best[0] <= 10:
		top_10_min += 1
		top_200_min += 1
	elif temp_min_best[0] <= 200:
		top_200_min += 1


	if temp_max_best[0] <= 5:
		top_5_max += 1
		top_10_max += 1
		top_200_max += 1
	elif temp_max_best[0] <= 10:
		top_10_max += 1
		top_200_max += 1
	elif temp_max_best[0] <= 200:
		top_200_max += 1
	temp_array = [project, bug] 
	temp_array.extend([temp_mean_best[0], temp_min_best[0],temp_max_best[0]])
	temp_array.extend([temp_mean_best[int(len(temp_mean_best)/2)], temp_min_best[int(len(temp_min_best)/2)],temp_max_best[int(len(temp_max_best)/2)]])
	temp_array.extend([temp_mean_best[-1], temp_min_best[-1],temp_max_best[-1]])
	temp_array.extend([temp_mean_best[0]/sloc, temp_min_best[0]/sloc, temp_max_best[0]/sloc])
	temp_array.extend([float(sum(temp_mean_best[:half_size]) / len(temp_mean_best[:half_size])) / sloc, float(sum(temp_min_best[:half_size]) / len(temp_min_best[:half_size])) / sloc,float(sum(temp_max_best[:half_size]) / len(temp_max_best[:half_size])) / sloc])
	temp_array.extend([float(sum(temp_mean_best) / len(temp_mean_best)) / sloc, float(sum(temp_min_best) / len(temp_min_best)) / sloc, float(sum(temp_max_best) / len(temp_max_best)) / sloc])

	all_result.append(temp_array)

size_all = len(all_result) - 1
print(size_all)

all_name = ["MEAN", "MIN", "MAX"]
top_name = ["TOP 5: ", "TOP 10: ", "TOP 200: "]
top_list = [[top_5_mean, top_10_mean, top_200_mean], [top_5_min, top_10_min, top_200_min], [top_5_max, top_10_max, top_200_max]]

for i, n in enumerate(all_name):
	print(n)
	for j, top in enumerate(top_list[i]):
		temp_string = top_name[j]
		print(temp_string+ str(top) + "(" + str((100 * float(top)/float(size_all))) + "%)")



output_name = "combine.txt"
if len(sys.argv) > 2:
	output_name = sys.argv[2]
with open(output_name,'w', newline='', encoding='utf-8') as output:
	writer = csv.writer(output, delimiter='\t')
	for i in all_result:
		writer.writerow(i)
