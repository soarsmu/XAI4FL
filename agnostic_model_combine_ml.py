

# Please specify the file name of spectra, matrix and output file when running the code
# python agnositc_model_combine_ml.py <spectra_file_location> <matrix_file_location> <output_file_name>
# e.g., python agnositc_model_combine_ml.py F:\Defects4J\Closure\1\spectra F:\Defects4J\Closure\1\matrix Closure-1 

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn import metrics
import pandas as pd
import json
import re
import ast
import sys
import shap
import numpy as np
import xgboost as xgb

def choose_model(model_name, x_train, x_test, y_train, y_test):
    if model_name == "RF":
        model = RandomForestClassifier(n_estimators=100, bootstrap=True, max_features='sqrt', random_state=1)
        model.fit(x_train, y_train)
    elif model_name == "XGB":
        data_train = xgb.DMatrix(x_train, label=y_train)
        data_test = xgb.DMatrix(x_test, label=y_test)
        param = {
            'eta': 0.3, 
            'max_depth': 6,  
            'objective': 'multi:softprob',  
            'num_class': 2}
        #Declare and train model
        model = xgb.train(param, data_train)
        pred = model.predict(data_test)
        best_preds = np.asarray([np.argmax(line) for line in pred])
        print("Precision = {}".format(precision_score(y_test, best_preds, average='macro')))
        print("Recall = {}".format(recall_score(y_test, best_preds, average='macro')))
        print("Accuracy = {}".format(accuracy_score(y_test, best_preds)))
    elif model_name == "KNN":
        model = KNeighborsClassifier()
        model.fit(x_train, y_train)
        y_pred=model.predict(x_test)
        print(metrics.classification_report(y_test, y_pred))
    elif model_name == "log-reg":
        model = LogisticRegression()
        model.fit(x_train, y_train)
        y_pred=model.predict(x_test)
        print(metrics.classification_report(y_test, y_pred))
    elif model_name == "SVM":
        model = SVC(kernel='linear')
        model.fit(x_train, y_train)
        y_pred=model.predict(x_test)
        print(metrics.classification_report(y_test, y_pred))

    return model


#Helper function that takes in shap.Explanation object and instance name or index to generate list of SHAP values based on given instance
def make_line_in_txt_shap(shap_value_exp_obj_instance, instance_name):
    feature_indexes = []
    new_list = []
    # Iterate through dataframe row with index = instance_name and store lines of code executed inside feature_indexes array.
    for i in range(len(df.iloc[instance_name])):
        if df.iloc[instance_name][i] == 1:
            feature_indexes.append(i)
    # Iterate through feature_indexes array to find lines of code executed and pair those lines of code with their respective SHAP values intro string
    # Pair string: <line_of_code>;<hsap_value>
    # Pair strings are stored in new_list array.
    for i in feature_indexes:
        new_list.append(";".join([list(x.columns)[i], str(shap_value_exp_obj_instance[i].values)]))
    return new_list
    # Output: new_list -> ['org.apache.commons.math3.random.Well19937c#95;0.008888888888888889', 'org.apache.commons.math3.random.Well19937c#96;0.017103535353535353']

def write_output_file(filename, obj, is_reverse=False):
    with open(filename, 'w') as file:
        sorted_dict = sorted(obj.items(), key=lambda x: x[1], reverse=is_reverse)
        for i in sorted_dict:
            file.write(";".join([i[0], str(i[1])]) + "\n")

if len(sys.argv) < 5:
    print("Please give input of <model_name> <spectra_file_location> <matrix_file_location> <output_file_name>")
    #print("e.g., python agnositc_model_combine_ml.py RF 1\\spectra 1\\matrix Closure-1 ")
    exit()
if sys.argv[1] not in ["RF", "KNN", "log-reg", "SVM", "XGB"]:
    print("Please choose from this model names: RF, KNN, log-reg, SVM, XGB")
    exit()
    
model_name = sys.argv[1]
input_file_spectra = sys.argv[2]
input_file_matrix = sys.argv[3]
output_file_name = sys.argv[4]
f = open(input_file_spectra, "r")
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


df = pd.read_csv(input_file_matrix, sep=' ', names=spectra_list, header=None)
if model_name == "log-reg" or model_name == "SVM" or model_name == "XGB":
    df = df.replace(["-","+"], [0,1])
else:
    df = df.replace(["-","+"], ["FAIL","PASS"])

x = df[spectra_list[:len(spectra_list)-1]]
y = df["Pass/Fail"]
print(spectra_list)
print(df)

fails = []
passes = []
i = 0
if model_name == "log-reg" or model_name == "SVM":
    for result in y:
        if result == 0:
            fails.append(i)
        else:
            passes.append(i)
        i += 1
else:
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

model = choose_model(model_name, x_train, x_test, y_train, y_test)

shap_to_txt = {}

if model_name == "KNN" or model_name == "log-reg" or model_name == "SVM":

    # Creates SHAP Explainer Object based on machine learning model
    # KernelExplainer -> KNN ; LinearExplainer -> LogReg, SVM
    # Input: Machine learning model object, training dataset (x_train, x_test, y_train, y_test)
    # Output: SHAP values dictionary per failed test case instance (shap_to_txt)

    if model_name == "KNN":
        explainer = shap.KernelExplainer(model.predict_proba, x_train)
        shap_values = explainer.shap_values(x)

        # shap_values output (3D array with length = 2); shap_values[0] -> 3D array with positive SHAP values (SHAP values * 1) (the one we're using) with length = number of spectra; 
        # shap_values[1] -> 3D array with negative SHAP values (SHAP values * -1) with length = number of spectra
        # shap_values output (3D array with length = 2):
        # [array([[0., 0., 0., ..., 0., 0., 0.], [0., 0., 0., ..., 0., 0., 0.], [0., 0., 0., ..., 0., 0., 0.], ..., [0., 0., 0., ..., 0., 0., 0.]]),
        #  array([[0., 0., 0., ..., 0., 0., 0.], [0., 0., 0., ..., 0., 0., 0.], [0., 0., 0., ..., 0., 0., 0.], ..., [0., 0., 0., ..., 0., 0., 0.]])]
        # shap_values[0] output (2D array with length = number of instances):
        # [[0. 0. 0. ... 0. 0. 0.], [0. 0. 0. ... 0. 0. 0.], ... , [0. 0. 0. ... 0. 0. 0.], [0. 0. 0. ... 0. 0. 0.]]
        # shap_values[0][18] output (1D array with length = number of spectra): 
        # array([ 0. , 0. , 0. , 0. , 0. , ... , 0.00537892  0.01126959  0.0039343   0.0089556, -0.01115984 ... , 0.00311491, 0. , 0. , ... , 0. ])
    
    elif model_name == "log-reg" or model_name == "SVM":
        explainer = shap.LinearExplainer(model, x_train)
        shap_values = explainer.shap_values(x)

        # shap_values output (2D array with length = number of instance):
        # array([[0., 0., 0., ..., 0., 0., 0.], [0., 0., 0., ..., 0., 0., 0.], ..., [0., 0., 0., ..., 0., 0., 0.], [0., 0., 0., ..., 0., 0., 0.]])
        # shap_values[18] output (1D array with length = number of spectra): 
        # array([ 0. , 0. , 0. , 0. , 0. , ... , , -0.00232071, -0.00232071, -0.00232071, -0.00232071, -0.00232071, ... , -0.01180167, 0. , 0. , ... , 0.  ])

    if model_name == "log-reg" or model_name == "SVM":
        for i in fails:
            temp_list = []
            for index, j in enumerate(shap_values[i]):
                # We can directly iterate the SHAP values of each feature given an instance as the array is only 2-Dimensional 
                temp_list.append(spectra_list[index] + ";" + str(j))
            shap_to_txt[x.iloc[i].name] = temp_list
    elif model_name == "KNN":
        for i in fails:
            temp_list = []
            for index, j in enumerate(shap_values[0][i]):
                # We need to determine which SHAP Values (in our case it's positive SHAP values, i.e., 0th index) before iterating the SHAP values 
                temp_list.append(spectra_list[index] + ";" + str(j))
            shap_to_txt[x.iloc[i].name] = temp_list

    # Output: shap_to_txt dictionary (key -> index of failed test case instance; value -> array of <line_of_code>;<shap_value>):
    # {18: ['org.apache.commons.math3.util.MathArrays$1#674;0.0',
    #   'org.apache.commons.math3.util.MathArrays$1#678;0.0',
    #   'org.apache.commons.math3.util.MathArrays$1#680;0.0', 
    #   ..., 'org.apache.commons.math3.random.RandomDataImpl#581;0.0']}
    
    # Process individual SHAP values as well as mean, max, and min SHAP values for each test case
    # Input: SHAP values dictionary per failed test case instance (shap_to_txt)
    # Output: SHAP values dictionary per line of code (shap_lines_output_val)

    shap_lines_output_val = {}

    for i in shap_to_txt:
        for index, value in enumerate(shap_to_txt[i]):
            feature_shapval = value.split(";")
            feature = feature_shapval[0]
            shapval = feature_shapval[1]
            if feature not in shap_lines_output_val:
                shap_lines_output_val[feature] = []
            shap_lines_output_val[feature].append(float(shapval))
    
    # Output: shap_lines_output_val (key -> line of code; value -> array of SHAP values obtained from each failed instance 
    # with array length = number of failed test case instances) :
    # {'org.apache.commons.math3.distribution.AbstractRealDistribution#112': [0.0],
    # 'org.apache.commons.math3.distribution.AbstractRealDistribution#113': [0.0], ... , 
    # 'org.apache.commons.math3.distribution.AbstractRealDistribution#246': [-0.0029355801005312245],
    # 'org.apache.commons.math3.distribution.AbstractRealDistribution#247': [-0.0029355801005312245], ...
    # 'org.apache.commons.math3.util.Pair#94': [0.0]}

else:
    # Creates SHAP Tree Explainer Object for XGBoost and Random Forest
    # Input: Machine learning tree-based model object, training dataset (x_train, x_test, y_train, y_test) 
    # Output: SHAP values dictionary per failed test case instance (shap_to_txt)

    shap_values = shap.TreeExplainer(model).shap_values(x)
    # BUGGED make_line_in_txt_shap function causes lines of code to be lost during compile. FIXED by using normal iteration    
    for i in fails:
        temp_list = []
        for index, j in enumerate(shap_values[0][i]):
            temp_list.append(spectra_list[index] + ";" + str(j))
        shap_to_txt[x.iloc[i].name] = temp_list

    # Output: shap_to_txt dictionary (key -> index of failed test case instance; value -> array of <line_of_code>;<shap_value>):
    # {18: ['org.apache.commons.math3.util.MathArrays$1#674;0.0',
    # 'org.apache.commons.math3.util.MathArrays$1#678;0.0', ... , 
    # 'org.apache.commons.math3.distribution.DiscreteRealDistribution#66;0.0033760683760683764',
    # 'org.apache.commons.math3.distribution.DiscreteRealDistribution#67;0.006817901234567901',
    # 'org.apache.commons.math3.distribution.DiscreteRealDistribution#87;0.005427328843995511', ... , 
    # 'org.apache.commons.math3.random.RandomDataImpl#565;0.0',
    # 'org.apache.commons.math3.random.RandomDataImpl#581;0.0']}

    # Process individual SHAP values as well as mean, max, and min SHAP values for each test case
    # Input: SHAP values dictionary per failed test case instance (shap_to_txt)
    # Output: SHAP values dictionary per line of code (shap_lines_output_val)

    shap_lines_output_val = {}

    for i in shap_to_txt:
        for index, value in enumerate(shap_to_txt[i]):
            feature_shapval = value.split(";")
            feature = feature_shapval[0]
            shapval = feature_shapval[1]
            if feature not in shap_lines_output_val:
                shap_lines_output_val[feature] = []
            shap_lines_output_val[feature].append(float(shapval))
    
    # Output: shap_lines_output_val (key -> line of code; value -> array of SHAP values obtained from each failed instance 
    # with array length = number of failed test case instances) :
    # {'org.apache.commons.math3.distribution.AbstractRealDistribution#112': [0.0],
    # 'org.apache.commons.math3.distribution.AbstractRealDistribution#113': [0.0], ... , 
    # 'org.apache.commons.math3.distribution.DiscreteDistribution#148': [0.006186868686868687],
    # 'org.apache.commons.math3.distribution.DiscreteDistribution#157': [0.015071895424836602],
    # 'org.apache.commons.math3.distribution.DiscreteDistribution#158': [0.025824074074074076], ... , 
    # 'org.apache.commons.math3.util.Pair#84': [0.0],
    # 'org.apache.commons.math3.util.Pair#94': [0.0]}

#Process individual SHAP values as well as mean, max, and min SHAP values for each test case
#Input: SHAP values dictionary per line of code (shap_lines_output_val)
#Output: Dictionaries of each line of code's mean, max, and min SHAP values (shap_means, shap_max, shap_min)

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

# Output: shap_means, shap_max, shap_min; Each dictionary follows the format: key -> line of code, value -> min/max/mean SHAP value
# Example for shap_means:
# {'org.apache.commons.math3.distribution.AbstractRealDistribution#112': 0.0,
# 'org.apache.commons.math3.distribution.AbstractRealDistribution#113': 0.0, ... , 
# 'org.apache.commons.math3.distribution.DiscreteRealDistribution#159': 0.004962962962962964,
# 'org.apache.commons.math3.distribution.DiscreteRealDistribution#161': 0.001574074074074074, ... ,
# 'org.apache.commons.math3.util.Pair#94': 0.0} 

#Write output file
#Input: Dictionaries of each line of code's mean, max, and min SHAP values
#Output: .txt Files containing each line of code's SHAP values, mean, max, and min SHAP values

with open(output_file_name+"_"+model_name+"_shap_results.txt", "w") as file:
    for i in shap_to_txt:
        file.write(str(i) + "\n")
        for j in range(len(shap_to_txt[i])):
            file.write(shap_to_txt[i][j] + "\n")

write_output_file(output_file_name+"_"+model_name+'_shap_mean.txt', shap_means, is_reverse=True)
write_output_file(output_file_name+"_"+model_name+'_shap_min.txt', shap_min, is_reverse=True)
write_output_file(output_file_name+"_"+model_name+'_shap_max.txt', shap_max, is_reverse=True)
