import os                                                                                                             
import csv

import pandas as pd
import sys

def top_print(df, rank):
	print("BEST CASE - NUMBER")
	print("TOP 1")
	print(len(df[df[rank]<=1]))

	print("TOP 5")
	print(len(df[df[rank]<=5]))

	print("TOP 10")
	print(len(df[df[rank]<=10]))

	print("TOP 200")
	print(len(df[df[rank]<=200]))


	print("BEST CASE - PERCENT")
	print("TOP 1")
	print(len(df[df[rank]<=1])/len(df[rank]))

	print("TOP 5")
	print(len(df[df[rank]<=5])/len(df[rank]))

	print("TOP 10")
	print(len(df[df[rank]<=10])/len(df[rank]))

	print("TOP 200")
	print(len(df[df[rank]<=200])/len(df[rank]))

def improvement(a,b,name_a,name_b):
	top_1 = 0
	top_5 = 0
	top_10 = 0
	top_200 = 0
	print("BEST-CASES IMPROVEMENT "+name_b+" Improve by "+name_a)
	temp_value = [0,0,0,0,0,0]
	values_store = {0:[], 1:[], 2:[], 3:[], 4:[], 5:[]}
	for proj in a:
		for bug in a[proj]:
			if proj in b and bug in b[proj]:
				val = b[proj][bug]
				val_2 = a[proj][bug]
				if val <= 1:
					top_1 += 1					
				if val <= 5:
					top_5 += 1
				if val <= 10:
					top_10 += 1
				if val <= 200:
					top_200 += 1	
				if val > 200:
					if val_2 < 200 and val_2 > 10:
						temp_value[0] += 1
						values_store[0].append(proj+" "+str(bug))
					elif val_2 <= 10 and val_2 > 5:
						temp_value[1] += 1
						values_store[1].append(proj+" "+str(bug))
					elif val_2 <= 5:
						temp_value[2] += 1
						values_store[2].append(proj+" "+str(bug))
				elif val <= 200 and val > 10:
					if val_2 <= 10 and val_2 > 5:
						temp_value[3] += 1
						values_store[3].append(proj+" "+str(bug))
					elif val_2 <= 5:
						temp_value[4] += 1
						values_store[4].append(proj+" "+str(bug))
				elif val <= 10 and val > 5:
					if val_2 <= 5:
						temp_value[5] += 1
						values_store[5].append(proj+" "+str(bug))
	print(temp_value)
	print(values_store)
	print(name_b)
	print(top_1)
	print(top_5)
	print(top_10)
	print(top_200)


if len(sys.argv) > 2:
	df = pd.read_csv(sys.argv[1],sep='\t') 
	df_sbfl = pd.read_csv(sys.argv[2],sep='\t') 
else:
	df = pd.read_csv('combine.txt',sep='\t') 
	df_sbfl = pd.read_csv('SBFL_Results/combine_sbfl.txt',sep='\t',index_col=False)
#print(df_sbfl)
temp_df = df_sbfl[df_sbfl['formula']=='dstar2']
for a in ["Chart", "Closure", "Lang", "Math", "Mockito", "Time"]:
	print("DSTAR "+a)
	top_print(temp_df[temp_df['project']==a], 'rank_best_case')
	print("RF+ SHAP (MEAN CASE) "+a)
	top_print(df[df['project_id']==a], 'rank_mean_best')

	temp_result = temp_df[temp_df['project']==a].values


	dstar_process = {}
	for i, val in enumerate(temp_result):
		if val[0] in dstar_process:
			dstar_process[val[0]][val[1]] = val[3]
		else:	
			dstar_process[val[0]] = {val[1] : val[3]}

	temp_result = df[df['project_id']==a].values
	rf_process = {}
	for i, val in enumerate(temp_result):
		if val[0] in rf_process:
			rf_process[val[0]][val[1]] = val[2]
		else:	
			rf_process[val[0]] = {val[1] : val[2]}

	improvement(dstar_process,rf_process,'DSTAR', 'RFSHAPMEAN')
	improvement(rf_process,dstar_process,'RFSHAPMEAN','DSTAR')

print("DSTAR ")
top_print(temp_df, 'rank_best_case')
print("RF+ SHAP (MEAN CASE) ")
top_print(df, 'rank_mean_best')

temp_result = temp_df.values


dstar_process = {}
for i, val in enumerate(temp_result):
	if val[0] in dstar_process:
		dstar_process[val[0]][val[1]] = val[3]
	else:	
		dstar_process[val[0]] = {val[1] : val[3]}

temp_result = df.values
rf_process = {}
for i, val in enumerate(temp_result):
	if val[0] in rf_process:
		rf_process[val[0]][val[1]] = val[2]
	else:	
		rf_process[val[0]] = {val[1] : val[2]}

improvement(dstar_process,rf_process,'DSTAR', 'RFSHAPMEAN')
improvement(rf_process,dstar_process,'RFSHAPMEAN','DSTAR')
