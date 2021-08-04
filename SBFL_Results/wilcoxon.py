
import os                                                                                                             
import csv

import pandas as pd
from scipy import stats

from statistics import mean, stdev
from math import sqrt
import sys



def cliffsDelta(lst1, lst2, **dull):

    """Returns delta and true if there are more than 'dull' differences"""
    if not dull:
        dull = {'small': 0.147, 'medium': 0.33, 'large': 0.474} # effect sizes from (Hess and Kromrey, 2004)
    m, n = len(lst1), len(lst2)
    lst2 = sorted(lst2)
    j = more = less = 0
    for repeats, x in runs(sorted(lst1)):
        while j <= (n - 1) and lst2[j] < x:
            j += 1
        more += j*repeats
        while j <= (n - 1) and lst2[j] == x:
            j += 1
        less += (n - j)*repeats
    d = (more - less) / (m*n)
    size = lookup_size(d, dull)
    return d, size


def lookup_size(delta: float, dull: dict) -> str:
    """
    :type delta: float
    :type dull: dict, a dictionary of small, medium, large thresholds.
    """
    delta = abs(delta)
    if delta < dull['small']:
        return 'negligible'
    if dull['small'] <= delta < dull['medium']:
        return 'small'
    if dull['medium'] <= delta < dull['large']:
        return 'medium'
    if delta >= dull['large']:
        return 'large'


def runs(lst):
    """Iterator, chunks repeated values"""
    for j, two in enumerate(lst):
        if j == 0:
            one, i = two, 0
        if one != two:
            yield j - i, one
            i = j
        one = two
    yield j - i + 1, two


def getting_all(dstar, op2, barinel, ochiai, tarantula, case):
	print(stats.normaltest(dstar[case]))
	print(stats.normaltest(op2[case]))
	print(stats.normaltest(barinel[case]))
	print(stats.normaltest(ochiai[case]))
	print(stats.normaltest(tarantula[case]))
	

	print("OCHIAI > TARANTULA")
	print(stats.ttest_rel(ochiai[case],tarantula[case]))
	print((mean(ochiai[case]) - mean(tarantula[case])) / (sqrt((stdev(ochiai[case]) ** 2 + stdev(tarantula[case]) ** 2) / 2)))
	print(stats.ranksums(ochiai[case],tarantula[case]))
	print(cliffsDelta(ochiai[case],tarantula[case]))
	print("BARINEL > OCHIAI")
	print(stats.ttest_ind(barinel[case],ochiai[case]))
	print((mean(barinel[case]) - mean(ochiai[case])) / (sqrt((stdev(barinel[case]) ** 2 + stdev(ochiai[case]) ** 2) / 2)))
	print(stats.ranksums(barinel[case],ochiai[case]))
	print(cliffsDelta(barinel[case],ochiai[case]))
	print("BARINEL > TARANTULA")
	print(stats.ttest_ind(barinel[case],tarantula[case]))
	print((mean(barinel[case]) - mean(tarantula[case])) / (sqrt((stdev(barinel[case]) ** 2 + stdev(tarantula[case]) ** 2) / 2)))
	print(stats.ranksums(barinel[case],tarantula[case]))
	print(cliffsDelta(barinel[case],tarantula[case]))
	print("OP2 > OCHIAI")
	print(stats.ttest_rel(op2[case],ochiai[case]))
	print((mean(op2[case]) - mean(ochiai[case])) / (sqrt((stdev(op2[case]) ** 2 + stdev(ochiai[case]) ** 2) / 2)))
	print(stats.ranksums(op2[case],ochiai[case]))
	print(cliffsDelta(op2[case],ochiai[case]))
	print("OP2 > TARANTULA")
	print(stats.ttest_ind(op2[case],tarantula[case]))
	print((mean(op2[case]) - mean(tarantula[case])) / (sqrt((stdev(op2[case]) ** 2 + stdev(tarantula[case]) ** 2) / 2)))
	print(stats.ranksums(op2[case],ochiai[case]))
	print(cliffsDelta(op2[case],tarantula[case]))
	print("DSTAR > OCHIAI")
	print(stats.ttest_ind(dstar[case],ochiai[case]))
	print((mean(dstar[case]) - mean(ochiai[case])) / (sqrt((stdev(dstar[case]) ** 2 + stdev(ochiai[case]) ** 2) / 2)))
	print(stats.ranksums(op2[case],ochiai[case]))
	print(cliffsDelta(dstar[case],ochiai[case]))
	print("DSTAR > TARANTULA")
	print(stats.ttest_ind(dstar[case],tarantula[case]))
	print((mean(dstar[case]) - mean(tarantula[case])) / (sqrt((stdev(dstar[case]) ** 2 + stdev(tarantula[case]) ** 2) / 2)))
	print(stats.ranksums(op2[case],ochiai[case]))
	print(cliffsDelta(dstar[case],tarantula[case]))


df_sbfl = pd.read_csv('combine_exam_sbfl.txt',sep='\t',index_col=False)
print(df_sbfl)
dstar = df_sbfl[df_sbfl['formula']=='dstar2']
op2 = df_sbfl[df_sbfl['formula']=='opt2']
barinel = df_sbfl[df_sbfl['formula']=='barinel']
ochiai = df_sbfl[df_sbfl['formula']=='ochiai']
tarantula = df_sbfl[df_sbfl['formula']=='tarantula']
for a in ["exam_best_case_b", "exam_best_case_f", "exam_average_case_b", "exam_average_case_f", "exam_worst_case_b", "exam_worst_case_f"]:
	print("CASE: "+a)
	getting_all(dstar, op2, barinel, ochiai, tarantula, a)
