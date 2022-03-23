# XAI4FL: Enhancing Spectrum-Based Fault Localization with Explainable Artificial Intelligence
Manually finding the program unit responsible for a fault is tedious and time-consuming. To mitigate this problem, many fault localization techniques have been proposed. In this study, we propose a novel idea that first models the SBFL task as a classification problem of predicting whether a test case will fail or pass based on spectra information on program units. We subsequently apply eXplainable Artificial Intelligence (XAI) techniques to infer the local importance of each program unit to the prediction of each executed test case. Such a design can automatically learn the unique contributions of failed test cases to the suspiciousness of a program unit by learning the different and collaborative contributions of program units to each test case's executed result. As far as we know, this is the first XAI-supported SBFL approach. We evaluate the new approach on the Defects4J benchmark dataset.

## Requirements
- Python 3
- shap (pip install shap)
- sklearn (pip install scikit-learn)
- pandas (pip install pandas)
- scipy (pip install scipy)

## Examples
To get the suspiciousness list using XAI4FL approach, please specify the file name of spectra, matrix and output file when running the code
```
python XAI4FL.py <spectra_file_location> <matrix_file_location> <output_file_name>
e.g., python XAI4FL.py F:\Defects4J\Closure\1\spectra F:\Defects4J\Closure\1\matrix Closure-1 
```
We can get the spectra and matrix by using [GZoltar](https://gzoltar.com/). The example of the Gzoltar output is on the "GZoltar_Output_Example/". The output of "XAI4FL.py" is the list of program units from most to least suspicious.


## Repo Structure
```
- XAI4FL.py                      # Our approach of fault localization
- get_rank.py                    # Evaluation: get the rank of buggy line from defects4j suspiciousness list
- analyze_results.py             # Combine the results from get_rank
- analyze_sbfl_vs_our.py         # Compare the results between sbfl vs xai4fl for defects4j
- combine_fin.txt                # Combined results of running xai4fl in defects4j 
- Buggy_Line/                    # Buggy lines data (Defects4J)
- GZoltar_Output_Example/        # Examples of program spectra from Defects4J
- Results/                       # XAI4FL Results for Defects4J
- SBFL_Results/                  # SBFL results for Defects4J
- SLOC/                          # Lines of codes from Defects4J
- source-code-lines/             # Source code lines from Defects4J projects
```