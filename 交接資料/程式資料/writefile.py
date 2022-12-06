from asyncore import write
import serial  # 引用pySerial模組本身led
from tensorflow import keras
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import copy
# from keras.models import load_model
import csv

COM_PORT = 'COM3'    # 指定通訊埠名稱
BAUD_RATES = 9600    # 設定傳輸速率
flag = 1
# ser = serial.Serial(COM_PORT, BAUD_RATES)   # 初始化序列通訊埠
ser = serial.Serial('/dev/ttyACM0', 9600)
demo1=b"0"
demo2=b"1"
demo3=b"2"
demo4=b"3"
demo5=b"4"
all_datas = []
try:
    while True:
        while ser.in_waiting:          # 若收到序列資料…
            data_raw = ser.readline()  # 讀取一行
            #pred_dir = LSTM_Model.predict(data_raw)

            temp = data_raw.decode()  # 用預設的UTF-8解碼
            
            print(temp)###加的
            
            if "請揮" in temp:
                continue

            data = temp.split(',')
            data = data[:-1]

            if len(data) < 1:
                data = temp
            else:
                data = [int(digi) for digi in data]
              
                all_datas.append(data)
            
            if "finished" not in temp:
                continue
            # print(all_datas)

            #-----------寫檔--------------
            a = copy.deepcopy(all_datas)
            # print(a)
            for i in range(len(a)):
                a[i].insert(0,flag)
                a[i].append(0)
            print(flag)
            flag += 1
            
            with open('1wayTaoDegree0.csv', 'a', newline='') as student_file:
                writer = csv.writer(student_file)
                writer.writerows(a)
            
            student_file.close()
            #------------------------------------
            all_datas = []
            ser.write(demo1)
except KeyboardInterrupt:
    ser.close()    # 清除序列通訊物件
    print('再見！')



















