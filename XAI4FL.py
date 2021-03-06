# -*- coding: utf-8 -*-


# Please specify the file name of spectra, matrix and output file when running the code
# python XAI4FL.py <spectra_file_location> <matrix_file_location> <output_file_name>
# e.g., python XAI4FL.py F:\Defects4J\Closure\1\spectra F:\Defects4J\Closure\1\matrix Closure-1 

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import json
import re
import ast
import sys
import shap


def make_line_in_txt_shap(shap_value_exp_obj_instance, instance_name):
    feature_indexes = []
    new_list = []
    for i in range(len(df.iloc[instance_name])):
        if df.iloc[instance_name][i] == 1:
            feature_indexes.append(i)
    for i in feature_indexes:
        new_list.append(";".join([list(x.columns)[i], str(shap_value_exp_obj_instance[i].values)]))
    return new_list

def write_output_file(filename, obj, sp_list, is_reverse=False):
    with open(filename, 'w') as file:
        sorted_dict = sorted(obj.items(), key=lambda x: x[1], reverse=is_reverse)
        added_aft = sp_list.copy()
        for i in sorted_dict:
            file.write(";".join([i[0], str(i[1])]) + "\n")
            print(i[0])
            added_aft.remove(i[0])
        for i in added_aft:
            file.write(";".join([i, str("-1.0")]) + "\n")
            

f = open(sys.argv[1], "r")
spectra_list = [line.rstrip('\n') for line in f]
spectra_list.append("Pass/Fail")
f.close()

#case if there are duplicate in list
seen = {}
for i, x in enumerate(spectra_list):
    if x not in seen:
        seen[x] = 1
    else:
        seen[x] += 1
        num = seen[x]
        temp = x.split('#')
        temp_name = temp[0] + str(num)+ '#' + temp[1]
        spectra_list[i] = temp_name 


df = pd.read_csv(sys.argv[2], sep=' ', names=spectra_list, header=None)
df = df.replace(["-","+"], ["FAIL","PASS"])
x = df[spectra_list[:len(spectra_list)-1]]
y = df["Pass/Fail"]
print(spectra_list)
print(df)

fails = []
passes = []
i = 0
for result in y:
    if result == "FAIL":
        fails.append(i)
    else:
        passes.append(i)
    i += 1

if len(fails) == 1:
    df = df.append(df.iloc[fails[0]])
    x = df[spectra_list[:len(spectra_list)-1]]
    y = df["Pass/Fail"]

if (df.shape[0] < 6):
    df = df.append(df)
    x = df[spectra_list[:len(spectra_list)-1]]
    y = df["Pass/Fail"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=1)

model = RandomForestClassifier(n_estimators=100, bootstrap=True, max_features='sqrt', random_state=1)

model.fit(x_train, y_train)

shap_to_txt = {}

shap_values = shap.TreeExplainer(model).shap_values(x)

for i in fails:
    shap_value_exp_obj = shap.Explanation(shap_values[0][i], feature_names=list(x.columns))
    shap_to_txt[x.iloc[i].name] = []
    shap_to_txt[x.iloc[i].name] += make_line_in_txt_shap(list(shap_value_exp_obj), i)

shap_lines_output_val = {}

for i in shap_to_txt:
    for j in shap_to_txt[i]:
        feature_and_value = j.split(";")
        feature = feature_and_value[0]
        value =  float(feature_and_value[1])
        if feature not in shap_lines_output_val:
            shap_lines_output_val[feature] = []
        shap_lines_output_val[feature].append(value)

shap_means = {}
shap_max = {}
shap_min = {}

for i in shap_lines_output_val:
    if i not in shap_means:
        shap_means[i] = []
    if i not in shap_min:
        shap_min[i] = shap_lines_output_val[i][0]
    if i not in shap_max:
        shap_max[i] = shap_lines_output_val[i][0]
    for j in shap_lines_output_val[i]:
        shap_means[i].append(j)
        if abs(shap_max[i]) < abs(j):
            shap_max[i] = j
        if abs(shap_min[i]) > abs(j):
            shap_min[i] = j

for i in shap_means:
    shap_means[i] = sum(shap_means[i]) / len(shap_means[i])

with open(sys.argv[3]+"_shap_results.txt", "w") as file:
    for i in shap_to_txt:
        file.write(str(i) + "\n")
        for j in range(len(shap_to_txt[i])):
            file.write(shap_to_txt[i][j] + "\n")

spectra_list_new = spectra_list[:len(spectra_list)-1]

write_output_file(sys.argv[3]+'_shap_mean.txt', shap_means, spectra_list_new, is_reverse=True)
write_output_file(sys.argv[3]+'_shap_min.txt', shap_min, spectra_list_new, is_reverse=True)
write_output_file(sys.argv[3]+'_shap_max.txt', shap_max, spectra_list_new, is_reverse=True)
