# -*- coding: utf-8 -*-
"""agnositc-model-testing.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1vdBs9o9kyVs4JsnsKCxvknJJmxl-oDzW
"""

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from google.colab import files
import pandas as pd
import json
import re
import ast

def process_spectra(file_obj, file_name):
    file_string = str(file_obj[file_name])

    return file_string.split("\\n")

def process_matrix(file_obj, file_name):
    file_string = str(file_obj[file_name])

    result = file_string.split("\\n")
    result[0] = result[0].replace("b'", "")

    for i in range(len(result)):
        result[i] = result[i].split(" ")
        result[i] = [int(x) if x.isdigit() else x for x in result[i]]
        lastel = result[i][-1]
        if lastel == "+":
            result[i][-1] = "PASS"
        elif lastel == "-":
            result[i][-1] = "FAIL"
    
    return result

def make_line_in_txt_shap(shap_value_exp_obj_instance, instance_name):
    feature_indexes = []
    new_list = []
    for i in range(len(df.iloc[instance_name])):
        if df.iloc[instance_name][i] == 1:
            feature_indexes.append(i)
    for i in feature_indexes:
        new_list.append(";".join([list(x.columns)[i], str(shap_value_exp_obj_instance[i].values)]))
    return new_list

spectra_file = files.upload()

spectra_list = process_spectra(spectra_file, list(spectra_file.keys())[0])
spectra_list[-1] = "Pass/Fail"
spectra_list[0] = spectra_list[0].replace("b'", "")

matrix_list = process_matrix(matrix_file, list(matrix_file.keys())[0])

matrix_list.pop()

df = pd.DataFrame(data=matrix_list, columns=spectra_list)

x = df[spectra_list[:len(spectra_list)-1]]
y = df["Pass/Fail"]

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

with open("Lang-37_shap_results.txt", "w") as file:
    for i in shap_to_txt:
        file.write(str(i) + "\n")
        for j in range(len(shap_to_txt[i])):
            file.write(shap_to_txt[i][j] + "\n")

write_output_file('Lang-37_shap_mean.txt', shap_means, is_reverse=True)
write_output_file('Lang-37_shap_min.txt', shap_min, is_reverse=True)
write_output_file('Lang-37_shap_max.txt', shap_max, is_reverse=True)