# -*- coding: utf-8 -*-
"""Skope-testing.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1R0bHK6EF6z05j6gscP7fAXqHMapSssbP
"""

"""
Remember to pip install skope-rules
"""


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import json
import re
import ast
import sys
from skrules import SkopeRules
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

#Input and process files into dataframe

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

#Create Skope Rules explanation model

skope_rules_clf = SkopeRules(feature_names=spectra_list[:-1], random_state=42, n_estimators=30,
                               recall_min=0.05, precision_min=0.9,
                               max_samples=0.7,
                               max_depth_duplication= 4, max_depth = 5)

skope_rules_clf.fit(x_train, y_train)

#Print rules generated from explanation model

for i_rule, rule in enumerate(skope_rules_clf.rules_):
    print(rule)

#Write output file

with open(sys.argv[3] + "-Skope-rules.txt", "w") as file:
    for i in skope_rules_clf.rules_:
        file.write(str(i) + "\n")
