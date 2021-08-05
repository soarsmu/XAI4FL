# -*- coding: utf-8 -*-
"""KNN-SHAP-Testing.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/13NwwRb5LH14HL2S5vAHXPcljZfXaBcHZ
"""

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import json
import re
import ast
import numpy as np
from sklearn.metrics import precision_score, recall_score, accuracy_score
import numpy as np
import shap
import sys
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

def write_output_file(filename, obj, is_reverse=False):
    with open(filename, 'w') as file:
        sorted_dict = sorted(obj.items(), key=lambda x: x[1], reverse=is_reverse)
        for i in sorted_dict:
            file.write(";".join([i[0], str(i[1])]) + "\n")

#Input and process files into dataframe
#Input: File paths, both spectra and matrix. System finds the files
#Output: Dataframe (Spectra List + PASS/FAIL labels as column and Test instances as rows with its execution matrix)

f = open(sys.argv[1], "r")
spectra_list = [line.rstrip('\n') for line in f]
spectra_list.append("Pass/Fail")
f.close()

#Case if there are duplicate in list
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

#Split dataframe into variables and classes as well as handle any insufficient data 
#(Handles insufficient number labelled instance by appending a copy of that instance; Handles insufficient total number of instance by reappending the dataframe)
#Input: Dataframe created from spectra and matrix files
#Output: Dataset for model training and model testing (e.g., x_train -> instance feature values for training, y_test -> instance label for training) 

x = df[spectra_list[:len(spectra_list)-1]]
y = df["Pass/Fail"]

fails = []
passes = []
i = 0
for result in y:
    if result == 0 or result == "FAIL":
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

#Declare and train the model (Current model is K-Nearest-Neighbor)
#Input: Training dataset (x_train, y_train)
#Output: Trained model; prints classification report of model

model = KNeighborsClassifier()

model.fit(x_train, y_train)

y_pred=model.predict(x_test)

print(metrics.classification_report(y_test, y_pred))

#Declares SHAP explainer object and explains variable's contribution using SHAP Kernel explainer
#Input: Trained model and array of "FAIL"-labelled instances
#Output: Dictionary of each line of code (feature)'s SHAP values 

shap_to_txt = {}

explainer = shap.KernelExplainer(model.predict_proba, x_train)
shap_values = explainer.shap_values(x)


for i in fails:
    temp_list = []
    for index, j in enumerate(shap_values[0][i]):
        temp_list.append(spectra_list[index] + ";" + str(j))
    shap_to_txt[x.iloc[i].name] = temp_list

#Process individual SHAP values as well as mean, max, and min SHAP values for each test case
#Input: Dictionary of each line of code (feature)'s SHAP values
#Output: Dictionaries of each line of code's mean, max, and min SHAP values

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

#Write output file
#Input: Dictionaries of each line of code's mean, max, and min SHAP values
#Output: .txt Files containing each line of code's SHAP values, mean, max, and min SHAP values

with open(sys.argv[3]+"_knn-shap_results.txt", "w") as file:
    for i in shap_to_txt:
        file.write(str(i) + "\n")
        for j in range(len(shap_to_txt[i])):
            file.write(shap_to_txt[i][j] + "\n")

write_output_file(sys.argv[3]+'_knn-shap_mean.txt', shap_means, is_reverse=True)
write_output_file(sys.argv[3]+'_knn-shap_min.txt', shap_min, is_reverse=True)
write_output_file(sys.argv[3]+'_knn-shap_max.txt', shap_max, is_reverse=True)
