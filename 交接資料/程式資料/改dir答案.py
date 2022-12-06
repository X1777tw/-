import csv
import pandas as pd
import copy
import numpy as np
import math

data = pd.read_csv('4wayDataV17.csv')
redata = data.drop(['Group','Dir'], axis = 1)
npdatacsv = np.array(redata)
print(len(npdatacsv))

a = npdatacsv.tolist()
for i in range(72000):
    a[i].insert(0,i)
    if(i < 32000):
        a[i].append(0)
    else:
        a[i].append(1)    
with open('4wayDataBinary.csv', 'w', newline='') as student_file:
    writer = csv.writer(student_file)
    writer.writerows(a)