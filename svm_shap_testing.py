# -*- coding: utf-8 -*-
"""SVM-SHAP-Testing.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1dWSZCcbglM4aWF78dtQGfIhGUqq-k06S
"""

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from google.colab import files
import pandas as pd
import json
import re
import ast
import numpy as np
from sklearn.metrics import precision_score, recall_score, accuracy_score
import numpy as np
import shap
from sklearn import metrics

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
            result[i][-1] = 1
        elif lastel == "-":
            result[i][-1] = 0
    
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

def write_output_file(filename, obj, is_reverse=False):
    with open(filename, 'w') as file:
        sorted_dict = sorted(obj.items(), key=lambda x: x[1], reverse=is_reverse)
        for i in sorted_dict:
            file.write(";".join([i[0], str(i[1])]) + "\n")

spectra_file = files.upload()
matrix_file = files.upload()

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
    if result == 0:
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

model = SVC(kernel='linear')

model.fit(x_train, y_train)

y_pred=model.predict(x_test)

print(metrics.classification_report(y_test, y_pred))

shap_to_txt = {}

explainer = shap.LinearExplainer(model, x_train)
shap_values = explainer.shap_values(x)


# for i in shap_values:
#     print(len(i))
for i in fails:
    temp_list = []
    for index, j in enumerate(shap_values[i]):
        temp_list.append(spectra_list[index] + ";" + str(j))
    shap_to_txt[x.iloc[i].name] = temp_list

shap_lines_output_val = {}

for i in shap_to_txt:
    for index, value in enumerate(shap_to_txt[i]):
        feature_shapval = value.split(";")
        feature = feature_shapval[0]
        shapval = feature_shapval[1]
        if feature not in shap_lines_output_val:
            shap_lines_output_val[feature] = []
        shap_lines_output_val[feature].append(float(shapval))

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


with open("Math-8_svm-shap_results.txt", "w") as file:
    for i in shap_to_txt:
        file.write(str(i) + "\n")
        for j in range(len(shap_to_txt[i])):
            file.write(shap_to_txt[i][j] + "\n")

write_output_file('Math-8_svm-shap_mean.txt', shap_means, is_reverse=True)
write_output_file('Math-8_svm-shap_min.txt', shap_min, is_reverse=True)
write_output_file('Math-8_svm-shap_max.txt', shap_max, is_reverse=True)

