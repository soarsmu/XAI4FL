import os
import pandas as pd 
import sys
from scipy.stats import rankdata
cwd_1 = ""
if len(sys.argv) > 1:
	cwd_1 = sys.argv[1]
cwd = os.getcwd()

def get_results():
	file_names = {'_shap_max.txt':'max', '_shap_min.txt':'min', '_shap_mean.txt':'mean'}
	colnames = ['statements', 'effect_size']
	all_rank = {}
	if cwd_1 == "":
		directories = [x[0] for x in os.walk(cwd)]
	else:
		directories = [x[0] for x in os.walk(cwd_1)]
	for x in directories:
		temp_name = x.split('\\')
		if len(temp_name) < 2:
			temp_name = x.split('/')
		file_name = temp_name[-1].replace(' ','-')
		check_file = file_name
		check_file += '_shap_max.txt'
		if os.path.isfile(os.path.join(x,check_file)):
			print(x)
			project = file_name.split('-')
			for i in file_names:
				#temp = pd.read_csv(os.path.join(x,file_name+i), sep=';', names=colnames, header=None)
				temp_1 = []
				temp_2 = []
				file = open(os.path.join(x,file_name+i), 'r')
				lines = file.readlines()
				for line in lines:
					temp_read = line.rstrip().split(';')
					temp_1.append(temp_read[0])
					temp_2.append(temp_read[1])
					# temp[temp_read[0]] = temp_read[1]
				file.close()
				if project[0] not in all_rank :
					all_rank[project[0]] = {project[1]: {'max':{}, 'min':{}, 'mean':{}}}
				elif project[1] not in all_rank[project[0]]:
					all_rank[project[0]][project[1]] = {'max':{}, 'min':{}, 'mean':{}}
				all_rank[project[0]][project[1]][file_names[i]] = [temp_1,temp_2]
	return all_rank

def process_line(line):
	#print(line)
	split_line = line.split('#')
	line_number = split_line[1]
	split_line_1 = split_line[0].split('.')
	file_name_wo_line = split_line_1[0].replace('/','.')
	return file_name_wo_line +'#'+ line_number

def back_line(line):
	split_line = line.split('#')
	line_number = split_line[1]
	file_name_wo_line = split_line[0].replace('.','/')
	return file_name_wo_line +'.java#'+ line_number


def get_buggy_line(all_rank):
	buggy_line = {}
	path = os.path.join(cwd,'Buggy_Line')
	for project in all_rank:
		for bug_number in all_rank[project]:
			file_name = project+'-'+bug_number+'.buggy.lines'
			if os.path.isfile(os.path.join(path, file_name)):
				if project not in buggy_line:
					buggy_line[project] = {bug_number:{}}
				elif bug_number not in buggy_line[project]:
					buggy_line[project][bug_number] = {}
				temp_read = {}
				file = open(os.path.join(path, file_name), 'r')
				lines = file.readlines()
				for line in lines:
					temp_read[process_line(line.rstrip())] = []
				file.close()
				file_name_candidates = project+'-'+bug_number+'.candidates'
				if os.path.isfile(os.path.join(path, file_name_candidates)):
					file = open(os.path.join(path, file_name_candidates), 'r')
					lines = file.readlines()
					for line in lines:
						temp_split = line.rstrip().split(',')
						first = process_line(temp_split[0])
						second = process_line(temp_split[1])
						if first in temp_read:
							temp_read[first].append(second)
					file.close()
				buggy_line[project][bug_number] = temp_read
	return buggy_line
	# path = os.path.join(cwd,'Buggy_Line')
	# onlyfiles = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
	# print(onlyfiles)

def get_sloc():
	return pd.read_csv(os.path.join(os.path.join(cwd,'SLOC'),'sloc.csv'), sep=',')

def new_rank(data):
	return rankdata([float(-1) * float(data[x]) for x in range(len(data))], method='average')

def get_line(proj, bug, buggy_l):
	# print(proj)
	# print(bug)
	# print(buggy_l)
	buggy_line = back_line(buggy_l)
	# print(buggy_line)
	file1 = open("source-code-lines/"+proj+"-"+bug+"b.source-code.lines","r")
	count = 0
	get_from = ""
	temp_1 = []
	temp_2 = []
	while True:
	 
	    # Get next line from file
		line = file1.readline()
	 
	    # if line is empty
	    # end of file is reached
		if not line:
			break
		temp_arr = line.strip().split(":")
		temp_1.append(temp_arr[0])
		temp_2.append(temp_arr[1])
		if temp_arr[1] == buggy_line:
			get_from = temp_arr[0]
		count += 1
	file1.close()

	while get_from in temp_2:
		temp_check = temp_2.index(get_from)
		get_from = temp_1[temp_check]
	if get_from == "":
		return buggy_l
	temp_line = process_line(get_from)
	return temp_line


def get_rank(all_rank,buggy_line,sloc_data):
	ranked_final = buggy_line.copy()
	temp_all_rank = all_rank.copy()
	for project in all_rank:
		for bug_number in all_rank[project]:
			sorted_max = new_rank(all_rank[project][bug_number]['max'][1])
			sorted_min = new_rank(all_rank[project][bug_number]['min'][1])
			sorted_mean = new_rank(all_rank[project][bug_number]['mean'][1])
			ranked_final[project][bug_number]['len_statements'] = len(sorted_max)
			temp_sloc = sloc_data.loc[(sloc_data['project_id'] == project) & (sloc_data['bug_id'] == int(bug_number))]
			# print(temp_sloc)
			ranked_final[project][bug_number]['sloc'] = temp_sloc['sloc'].values[0]
			ranked_final[project][bug_number]['sloc_total'] = temp_sloc['sloc_total'].values[0]
			# print(ranked_final[project][bug_number])
			for stmts in buggy_line[project][bug_number]:
				#print(stmts)
				#print(buggy_line[project][bug_number][stmts])
				if stmts != 'len_statements' and stmts != 'sloc' and stmts != 'sloc_total':
					if len(buggy_line[project][bug_number][stmts]) != 0:
						minimal_rank_max = len(sorted_max) + 1
						minimal_rank_min = len(sorted_max) + 1
						minimal_rank_mean = len(sorted_max) + 1
						stmts_max, stmts_min, stmts_mean = '', '', ''
						if stmts in all_rank[project][bug_number]['max'][0]:
							minimal_rank_max = sorted_max[all_rank[project][bug_number]['max'][0].index(stmts)]
							minimal_rank_min = sorted_max[all_rank[project][bug_number]['min'][0].index(stmts)]
							minimal_rank_mean = sorted_max[all_rank[project][bug_number]['mean'][0].index(stmts)]
							stmts_max = stmts
							stmts_min = stmts
							stmts_mean = stmts
						else:
							temp_stmts = stmts.split('#')
							#print(temp_stmts[0])
							#rint(temp_stmts[1])
							for loop_in in temp_all_rank[project][bug_number]['max'][0]:
								# print(loop_in)
								if temp_stmts[0] in loop_in and temp_stmts[1] in loop_in:
									minimal_rank_max = sorted_max[all_rank[project][bug_number]['max'][0].index(loop_in)]
									minimal_rank_min = sorted_max[all_rank[project][bug_number]['min'][0].index(loop_in)]
									minimal_rank_mean = sorted_max[all_rank[project][bug_number]['mean'][0].index(loop_in)]
									stmts_max = loop_in
									stmts_min = loop_in
									stmts_mean = loop_in
						if stmts_max == '':
							new_stmts = get_line(project, bug_number, stmts)
							if new_stmts in all_rank[project][bug_number]['max'][0]:
								minimal_rank_max = sorted_max[all_rank[project][bug_number]['max'][0].index(new_stmts)]
								minimal_rank_min = sorted_max[all_rank[project][bug_number]['min'][0].index(new_stmts)]
								minimal_rank_mean = sorted_max[all_rank[project][bug_number]['mean'][0].index(new_stmts)]
								stmts_max = new_stmts
								stmts_min = new_stmts
								stmts_mean = new_stmts
							else:
								temp_stmts = new_stmts.split('#')
								#print(temp_stmts[0])
								#print(temp_stmts[1])
								for loop_in in temp_all_rank[project][bug_number]['max'][0]:
									# print(loop_in)
									if temp_stmts[0] in loop_in and temp_stmts[1] in loop_in:
										minimal_rank_max = sorted_max[all_rank[project][bug_number]['max'][0].index(loop_in)]
										minimal_rank_min = sorted_max[all_rank[project][bug_number]['min'][0].index(loop_in)]
										minimal_rank_mean = sorted_max[all_rank[project][bug_number]['mean'][0].index(loop_in)]
										stmts_max = loop_in
										stmts_min = loop_in
										stmts_mean = loop_in

						for candidate in buggy_line[project][bug_number][stmts]:
							if candidate in all_rank[project][bug_number]['max'][0]:
								temp_max = sorted_max[all_rank[project][bug_number]['max'][0].index(candidate)]
								temp_min = sorted_max[all_rank[project][bug_number]['min'][0].index(candidate)]
								temp_mean = sorted_max[all_rank[project][bug_number]['mean'][0].index(candidate)]
								if temp_max < minimal_rank_max:
									minimal_rank_max = temp_max
									stmts_max = candidate
								if temp_min < minimal_rank_min:
									minimal_rank_min = temp_min
									stmts_min = candidate
								if temp_mean < minimal_rank_mean:
									minimal_rank_mean = temp_mean
									stmts_mean = candidate
							else:
								temp_stmts = candidate.split('#')
								not_found_1 = True
								#print(temp_stmts[0])
								#print(temp_stmts[1])
								for loop_in in temp_all_rank[project][bug_number]['max'][0]:
									# print(loop_in)
									if temp_stmts[0] in loop_in and temp_stmts[1] in loop_in:
										not_found_1 = False
										#print("FOUNDDDDD")
										temp_max = sorted_max[all_rank[project][bug_number]['max'][0].index(loop_in)]
										temp_min = sorted_max[all_rank[project][bug_number]['min'][0].index(loop_in)]
										temp_mean = sorted_max[all_rank[project][bug_number]['mean'][0].index(loop_in)]
										if temp_max < minimal_rank_max:
											minimal_rank_max = temp_max
											stmts_max = loop_in
										if temp_min < minimal_rank_min:
											minimal_rank_min = temp_min
											stmts_min = loop_in
										if temp_mean < minimal_rank_mean:
											minimal_rank_mean = temp_mean
											stmts_mean = loop_in
										break
								if not not_found_1:
									new_candidate = get_line(project, bug_number, candidate)
									if new_candidate in all_rank[project][bug_number]['max'][0]:
										temp_max = sorted_max[all_rank[project][bug_number]['max'][0].index(new_candidate)]
										temp_min = sorted_max[all_rank[project][bug_number]['min'][0].index(new_candidate)]
										temp_mean = sorted_max[all_rank[project][bug_number]['mean'][0].index(new_candidate)]
										if temp_max < minimal_rank_max:
											minimal_rank_max = temp_max
											stmts_max = new_candidate
										if temp_min < minimal_rank_min:
											minimal_rank_min = temp_min
											stmts_min = new_candidate
										if temp_mean < minimal_rank_mean:
											minimal_rank_mean = temp_mean
											stmts_mean = new_candidate
									else:
										temp_stmts = new_candidate.split('#')
										#print(temp_stmts[0])
										#print(temp_stmts[1])
										for loop_in in temp_all_rank[project][bug_number]['max'][0]:
											# print(loop_in)
											if temp_stmts[0] in loop_in and temp_stmts[1] in loop_in:
												#print("FOUNDDDDD")
												temp_max = sorted_max[all_rank[project][bug_number]['max'][0].index(loop_in)]
												temp_min = sorted_max[all_rank[project][bug_number]['min'][0].index(loop_in)]
												temp_mean = sorted_max[all_rank[project][bug_number]['mean'][0].index(loop_in)]
												if temp_max < minimal_rank_max:
													minimal_rank_max = temp_max
													stmts_max = loop_in
												if temp_min < minimal_rank_min:
													minimal_rank_min = temp_min
													stmts_min = loop_in
												if temp_mean < minimal_rank_mean:
													minimal_rank_mean = temp_mean
													stmts_mean = loop_in
												break
						ranked_final[project][bug_number][stmts] = {}
						ranked_final[project][bug_number][stmts]['max'] = minimal_rank_max
						ranked_final[project][bug_number][stmts]['min'] = minimal_rank_min
						ranked_final[project][bug_number][stmts]['mean'] = minimal_rank_mean
						ranked_final[project][bug_number][stmts]['max_stmts'] = stmts_max
						ranked_final[project][bug_number][stmts]['min_stmts'] = stmts_min
						ranked_final[project][bug_number][stmts]['mean_stmts'] = stmts_mean
						if minimal_rank_min == len(sorted_max) + 1 or minimal_rank_max == len(sorted_max) + 1 or minimal_rank_mean == len(sorted_max) + 1:
							ranked_final[project][bug_number][stmts]['not_found'] = '1'
						else:
							ranked_final[project][bug_number][stmts]['not_found'] = '0'
						ranked_final[project][bug_number][stmts]['omission'] = '1'
					else:
						ranked_final[project][bug_number][stmts] = {}
						# print(project+bug_number)
						if stmts in all_rank[project][bug_number]['max'][0]:
							ranked_final[project][bug_number][stmts]['max'] = sorted_max[all_rank[project][bug_number]['max'][0].index(stmts)]
							ranked_final[project][bug_number][stmts]['min'] = sorted_max[all_rank[project][bug_number]['min'][0].index(stmts)]
							ranked_final[project][bug_number][stmts]['mean'] = sorted_max[all_rank[project][bug_number]['mean'][0].index(stmts)]
							ranked_final[project][bug_number][stmts]['not_found'] = '0'
						else:
							temp_stmts = stmts.split('#')
							#print(temp_stmts[0])
							#print(temp_stmts[1])
							#print(temp_all_rank)
							not_found_temp = 1
							for loop_in in temp_all_rank[project][bug_number]['max'][0]:
								# print(loop_in)
								if temp_stmts[0] in loop_in and temp_stmts[1] in loop_in:
									#print("FOUNDDDDD")
									ranked_final[project][bug_number][stmts]['max'] = sorted_max[all_rank[project][bug_number]['max'][0].index(loop_in)]
									ranked_final[project][bug_number][stmts]['min'] = sorted_max[all_rank[project][bug_number]['min'][0].index(loop_in)]
									ranked_final[project][bug_number][stmts]['mean'] = sorted_max[all_rank[project][bug_number]['mean'][0].index(loop_in)]
									ranked_final[project][bug_number][stmts]['max_stmts'] = loop_in
									ranked_final[project][bug_number][stmts]['min_stmts'] = loop_in
									ranked_final[project][bug_number][stmts]['mean_stmts'] = loop_in
									ranked_final[project][bug_number][stmts]['not_found'] = '0'
									not_found_temp = 0
									break
							if not_found_temp == 1:
								new_stmts = get_line(project, bug_number, stmts)
								if new_stmts in all_rank[project][bug_number]['max'][0]:
									ranked_final[project][bug_number][stmts]['max'] = sorted_max[all_rank[project][bug_number]['max'][0].index(new_stmts)]
									ranked_final[project][bug_number][stmts]['min'] = sorted_max[all_rank[project][bug_number]['min'][0].index(new_stmts)]
									ranked_final[project][bug_number][stmts]['mean'] = sorted_max[all_rank[project][bug_number]['mean'][0].index(new_stmts)]
									ranked_final[project][bug_number][stmts]['not_found'] = '0'
									ranked_final[project][bug_number][stmts]['max_stmts'] = new_stmts
									ranked_final[project][bug_number][stmts]['min_stmts'] = new_stmts
									ranked_final[project][bug_number][stmts]['mean_stmts'] = new_stmts
								else:
									temp_stmts = new_stmts.split('#')
									#print(temp_stmts[0])
									#print(temp_stmts[1])
									#print(temp_all_rank)
									not_found_temp = 1
									for loop_in in temp_all_rank[project][bug_number]['max'][0]:
										# print(loop_in)
										if temp_stmts[0] in loop_in and temp_stmts[1] in loop_in:
											#print("FOUNDDDDD")
											ranked_final[project][bug_number][stmts]['max'] = sorted_max[all_rank[project][bug_number]['max'][0].index(loop_in)]
											ranked_final[project][bug_number][stmts]['min'] = sorted_max[all_rank[project][bug_number]['min'][0].index(loop_in)]
											ranked_final[project][bug_number][stmts]['mean'] = sorted_max[all_rank[project][bug_number]['mean'][0].index(loop_in)]
											ranked_final[project][bug_number][stmts]['max_stmts'] = loop_in
											ranked_final[project][bug_number][stmts]['min_stmts'] = loop_in
											ranked_final[project][bug_number][stmts]['mean_stmts'] = loop_in
											ranked_final[project][bug_number][stmts]['not_found'] = '0'
											not_found_temp = 0
											break
									if not_found_temp == 1:
										ranked_final[project][bug_number][stmts]['max'] = len(sorted_max) + 1
										ranked_final[project][bug_number][stmts]['min'] = len(sorted_max) + 1
										ranked_final[project][bug_number][stmts]['mean'] = len(sorted_max) + 1
										ranked_final[project][bug_number][stmts]['not_found'] = '1'
						ranked_final[project][bug_number][stmts]['omission'] = '0'
	return ranked_final
	# print(ranked_final)

def print_result(ranked_final):
	file = open("output.txt","w")
	file.write('project_id\tbug_id\ttemp_len\tsloc\tsloc_total\tbuggy_line\tis_omission\tis_not_found\tshap_max\tshap_min\tshap_mean\tbuggy_line_max\tbuggy_line_min\tbuggy_line_mean\n')
	for project in ranked_final:
		for bug_number in ranked_final[project]:
			temp_len = ranked_final[project][bug_number]['len_statements']
			temp_sloc = ranked_final[project][bug_number]['sloc']
			temp_sloc_total = ranked_final[project][bug_number]['sloc_total']
			for stmts in ranked_final[project][bug_number]:
				if stmts != 'len_statements' and stmts != 'sloc' and stmts != 'sloc_total':
					file.write(project+'\t'+bug_number+'\t'+str(temp_len)+'\t'+str(temp_sloc)+'\t'+str(temp_sloc_total)+'\t')
					file.write(stmts+'\t')
					file.write(ranked_final[project][bug_number][stmts]['omission']+'\t'+ranked_final[project][bug_number][stmts]['not_found']+'\t')
					file.write(str(ranked_final[project][bug_number][stmts]['max'])+'\t'+str(ranked_final[project][bug_number][stmts]['min'])+'\t'+str(ranked_final[project][bug_number][stmts]['mean']))
					if 'max_stmts' in ranked_final[project][bug_number][stmts]:
						file.write('\t'+str(ranked_final[project][bug_number][stmts]['max_stmts'])+'\t'+str(ranked_final[project][bug_number][stmts]['min_stmts'])+'\t')
						file.write(str(ranked_final[project][bug_number][stmts]['mean_stmts']))

					file.write('\n')
	file.close()
rank_data = get_results()
#print(rank_data)
buggy_line_data = get_buggy_line(rank_data)
sloc_data = get_sloc()
#print(sloc_data)
final_rank = get_rank(rank_data,buggy_line_data,sloc_data)
print_result(final_rank)





