# -*- coding: utf-8 -*-
"""MAPLE-Testing.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1X59bzLNU0gLwVydTQhOApzYsAFKRTdvV
"""

# Remember to git clone https://github.com/GDPlumb/MAPLE.git into same directory as this script

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from google.colab import files
import pandas as pd
import json
import re
import ast
from MAPLE.Code import MAPLE
import numpy as np

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

def write_output_file(filename, obj, is_reverse=False):
    with open(filename, 'w') as file:
        sorted_dict = sorted(obj.items(), key=lambda x: x[1], reverse=is_reverse)
        for i in sorted_dict:
            file.write(";".join([i[0], str(i[1])]) + "\n")

#Input and process files into dataframe

spectra_file = files.upload()
matrix_file = files.upload()

spectra_list = process_spectra(spectra_file, list(spectra_file.keys())[0])
spectra_list[-1] = "Pass/Fail"
spectra_list[0] = spectra_list[0].replace("b'", "")

matrix_list = process_matrix(matrix_file, list(matrix_file.keys())[0])

matrix_list.pop()

df = pd.DataFrame(data=matrix_list, columns=spectra_list)

#Split dataframe into variables and classes as well as handle any insufficient data

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

#Declare and train model

model = RandomForestClassifier(n_estimators=100, bootstrap=True, max_features='sqrt', random_state=1)

model.fit(x_train, y_train)

#Declare MAPLE explainer object

exp = MAPLE.MAPLE(np.array(x_train), np.array(model.predict(x_train)), np.array(x_test), np.array(model.predict(x_test)))

#Explain each feature's contribution based on its coefficient

feature_coefficients_based_on_error_instance = {}

for i in fails:
    cur_coefs = list(exp.explain(np.array(x.iloc[i]))['coefs'])
    feature_coefficients_based_on_error_instance[i] = []
    for index, val in enumerate(cur_coefs[:-1]):
        test_string = spectra_list[index] + ";" + str(val)
        feature_coefficients_based_on_error_instance[i].append(test_string)

#Process individual coefficient along with mean, min, and max values

coef_values_on_predict = {}

for i in feature_coefficients_based_on_error_instance:
    for j in feature_coefficients_based_on_error_instance[i]:
        feature_and_value = j.split(";")
        value = float(feature_and_value[1])
        feature = feature_and_value[0]
        if feature not in coef_values_on_predict:
            coef_values_on_predict[feature] = []
        coef_values_on_predict[feature].append(value)

feature_coef_means = {}
feature_coef_max = {}
feature_coef_min = {}

for i in coef_values_on_predict:
    feautre_coef_means[i] = sum(coef_values_on_predict[i]) / len(coef_values_on_predict[i])
    feature_coef_max[i] = max(coef_values_on_predict[i])
    feature_coef_min[i] = min(coef_values_on_predict[i])

#Write output file

with open("Math-8_maple_results.txt", "w") as file:
    for i in feature_coefficients_based_on_error_instance:
        file.write(str(i) + "\n")
        for j in range(len(feature_coefficients_based_on_error_instance[i])):
            file.write(feature_coefficients_based_on_error_instance[i][j] + "\n")

write_output_file('Math-8_maple_mean.txt', feautre_coef_means)
write_output_file('Math-8_maple_max.txt', feature_coef_max)
write_output_file('Math-8_maple_min.txt', feature_coef_min)

