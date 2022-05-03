import pandas as pd
from scipy.stats import iqr
from numpy import median, round
import os
for d in ["results_test","baseline_svm_balance","baseline_rf_balance","baseline_lr_balance"]:
    print("current directory is: "+d)
    lst = os.listdir(d)
    lst.sort()
    for f in lst:
        if f.endswith(".tsv"):
            print("current file is: "+f)
            df=pd.read_csv(d+"/"+f, sep='\t',header=None)
            array = df.values
            #print(array)
            print("           Accuracy   Precision   Recall   F1-score")
            print("Median: ", str(round(median(array, axis=0),2)))
            print("IQR:    ", str(round(iqr(array, axis=0),2)))
    print("="*100+"\n")
