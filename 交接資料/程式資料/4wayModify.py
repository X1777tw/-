import csv
import pandas as pd
import copy
import numpy as np
import math


data = pd.read_csv('1way.csv')
redata = data.drop(['Group','Dir'], axis = 1)
npdatacsv = np.array(redata)
print(len(npdatacsv))
reshapeData = []
for i in range(0, len(npdatacsv)):
    reshapeData.append(np.reshape(npdatacsv[i], (4, 4)))


data90 = []
data180 = []
data270 = []
print(len(reshapeData))
for i in range(0, len(reshapeData)):
    data90.append(np.rot90(reshapeData[i], 1)) #i跑每2000是一個手勢 總共16000
    data180.append(np.rot90(reshapeData[i], 2))
    data270.append(np.rot90(reshapeData[i], 3))

a = []
for i in range(32): #每種4 x 8種
    for j in range(2000):
        if(i % 4 == 0): #0 4 8 12 16 20 24 28 32...
            a.append(list(reshapeData[j].flat))
        if(i % 4 == 1): #1 5 9 13...
            a.append(list(data90[j].flat))
        if(i % 4 == 2):
            a.append(list(data180[j].flat))
        if(i % 4 == 3):
            a.append(list(data270[j].flat))
    if(i % 4 == 0):
        del reshapeData[0:2000]
    if(i % 4 == 1):
        del data90[0:2000]
    if(i % 4 == 2):
        del data180[0:2000]
    if(i % 4 == 3):
        del data270[0:2000]
   
print(len(a))
#len(a) = 20 x 100 x 4 x 8 = 64000
for i in range(64000):
    index = math.ceil((i + 1) /20)
    a[i].insert(0,index)
    ans = math.ceil(index /400)
    a[i].append(ans - 1)
with open('4way.csv', 'w', newline='') as student_file:
    writer = csv.writer(student_file)
    writer.writerows(a)