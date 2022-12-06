from asyncore import write
import serial  # 引用pySerial模組本身led
from tensorflow import keras
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import copy
from tensorflow.keras.models import load_model
import csv
import time

COM_PORT = 'COM3'    # 指定通訊埠名稱
BAUD_RATES = 9600    # 設定傳輸速率
flag = 91
# ser = serial.Serial(COM_PORT, BAUD_RATES)   # 初始化序列通訊埠
ser = serial.Serial('/dev/ttyACM0', 9600)
demo1=b"0"
demo2=b"1"
demo3=b"2"
demo4=b"3"
demo5=b"4"
# print(tf.config.list_physical_devices())
# LSTM_Model = tf.keras.models.load_model('my_model.h5')
# LSTM_Model = load_model('my_model.h5', compile = False)
LSTM_Model = tf.keras.models.load_model('1wayEnv.h5')
# 總共會有20列筆資料
all_datas = []
count = 0
ct = 0
env = [370,487,399,400,468,457,473,418,296,306,526,536,358,525,486,437]
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
            for i in range(20):
                for j in range(16):
                    all_datas[i][j] -= env[j]
            start = time.time()
            mean =[]
            max = []
            min = []
            std = []
            for t in range(20):
                tmp = copy.deepcopy(all_datas[t])
                tmp.sort()
                mean.append(np.mean(tmp))
                max.append(np.max(tmp))
                min.append(np.min(tmp)) 
                std.append(np.std(tmp))
            mean = np.array(mean).astype(float)
            max  = np.array(max).astype(float)
            min = np.array(min).astype(float)
            std = np.array(std).astype(float)

            all_datas = np.array(all_datas).astype(float)
            nums = all_datas
            all_datas = []
            
            for j in range(20):
                nums[j] = [(element - mean[idx]) / (max[idx] - min[idx]) for idx, element in enumerate(nums[j])] #原本

            # # (1, 20, 16)
            npdata = np.asarray(nums)[np.newaxis, :]
            print(LSTM_Model.predict(npdata))
            pred_dir = np.argmax(LSTM_Model.predict(npdata), axis=-1)
            #pred_dir = LSTM_Model.predict_classes(npdata)

            # print('預測:',pred_dir)
            pred_dir = int(pred_dir[0])
            end = time.time()
            # ct += 1
            if pred_dir == 0:
                print('預測:', '數字1')
                
            elif pred_dir == 1:
                print('預測:', '數字2')
                
            elif pred_dir == 2:
                print('預測:', '數字3') 
                
            elif pred_dir == 3:
                print('預測:', '左到右')
                
            elif pred_dir == 4:
                print('預測:', '右到左')
                
            elif pred_dir == 5:
                print('預測:', '上到下')
                
            elif pred_dir == 6:
                print('預測:', '翻轉')
                
            elif pred_dir == 7:
                print('預測:', '拍手')
                # count += 1
                # print('count',count)
            # print('次數', ct, '機率', count/ct)
            print("運算時間：%f 秒" % (end - start))
            ser.write(demo1)
            # if pred_dir == 0:
            #     ser.write(demo1)

            # elif pred_dir == 1:
            #      ser.write(demo2)

            # elif pred_dir == 2:
            #      ser.write(demo3)

            # elif pred_dir == 3:
            #      ser.write(demo4)

            # elif pred_dir == 4:
            #      ser.write(demo5)

            # #print('接收到的原始資料：', data_raw)
            # print('接收到的資料：', data2)
            # print('手勢指令:', data3)



except KeyboardInterrupt:
    ser.close()    # 清除序列通訊物件
    print('再見！')
