from asyncore import write
import serial  # 引用pySerial模組本身led
# from tensorflow import keras
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import copy
from scipy import interpolate, ndimage
# from tensorflow.keras.models import load_model
# from tensorflow import keras
import keras
import csv
import pandas as pd




COM_PORT = 'COM3'    # 指定通訊埠名稱
BAUD_RATES = 9600    # 設定傳輸速率
flag = 91
ser = serial.Serial(COM_PORT, BAUD_RATES)   # 初始化序列通訊埠
demo1=b"0"
demo2=b"1"
demo3=b"2"
demo4=b"3"
demo5=b"4"
# print(tf.config.list_physical_devices())
# LSTM_Model = tf.keras.models.load_model('my_model.h5')
# LSTM_Model = load_model('my_model.h5', compile = False)
LSTM_Model = keras.models.load_model('my_model.h5')
LSTM_ModelStatic = keras.models.load_model('my_modelStatic.h5')
# 總共會有20列筆資料
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

            #-------------------------------------------------------
            mediData = np.array(all_datas).astype(float)
            min = []
            for j in range(16):
                min.append(np.min(mediData[:,[j]]))  
            max = []
            for j in range(16):
                max.append(np.max(mediData[:,[j]]))    
            dif = []
            for i in range(len(max)):
                dif.append(max[i]-min[i])

            medi = np.median(dif[:])
            print(medi)
            #-------------------------------------------------------
            if(medi < 50):
                #------------------------------------------------------------------
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
                    nums[j] = [(element - mean[idx]) / (max[idx] - min[idx]) for idx, element in enumerate(nums[j])] 
                nums = nums.reshape(nums.shape[0],nums.shape[1],1).astype('float32')
                
                npdata = np.asarray(nums)[np.newaxis, :]
                print(LSTM_ModelStatic.predict(npdata))
                pred_dir = np.argmax(LSTM_ModelStatic.predict(npdata), axis=-1)
               
                print('靜態預測:',pred_dir)
                pred_dir = int(pred_dir[0])
                #---------------------------------------------------------------------------------------------------
                #---------------------插植 
                # reshapeData = np.reshape(all_datas[9], (4, 4))
                # reshapeData = np.array(reshapeData)
                # # print(ndimage.zoom(reshapeData[0],2,order = 3)) #cubic
                
                # interpolateData = ndimage.zoom(reshapeData,2,order = 3)
                
                # mean =[]
                # max = []
                # min = []
                # std = []
                
                # tmp = list(interpolateData.flat)
                # mean.append(np.mean(tmp))
                # max.append(np.max(tmp))
                # min.append(np.min(tmp)) 
                # std.append(np.std(tmp))
                # mean = np.array(mean).astype(float)
                # max  = np.array(max).astype(float)
                # min = np.array(min).astype(float)
                # std = np.array(std).astype(float)
            
                # interpolateData = np.array(interpolateData).astype(float)
                # nums = []
                # nums.append(interpolateData)
                
                # # print(nums)
                # nums = [(element - mean[idx]) / (max[idx] - min[idx]) for idx, element in enumerate(nums)] 
                # # print(nums)
                # nums = np.reshape(nums, (8, 8))

                # npdata = np.asarray(nums)[np.newaxis, :]
                # print(LSTM_ModelStatic.predict(npdata))
                # pred_dir = np.argmax(LSTM_ModelStatic.predict(npdata), axis=-1)
               
                # print('靜態預測:',pred_dir)
                # pred_dir = int(pred_dir[0])
                #--------------------------------------------------------------------------
            
            elif(medi >= 50):
                # a = copy.deepcopy(all_datas)
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
                    # nums[j] = [(element - mean[idx]) / (std[idx]) for idx, element in enumerate(nums[j])]
                    # nums[j] = [(element - min[idx]) / (max[idx] - min[idx]) for idx, element in enumerate(nums[j])]
                    nums[j] = [(element - mean[idx]) / (max[idx] - min[idx]) for idx, element in enumerate(nums[j])] #原本
               
                # # (1, 20, 16)
                npdata = np.asarray(nums)[np.newaxis, :]
                print(LSTM_Model.predict(npdata))
                pred_dir = np.argmax(LSTM_Model.predict(npdata), axis=-1)
                #pred_dir = LSTM_Model.predict_classes(npdata)

                print('動態預測:','[',int(pred_dir[0])+4,']')
                pred_dir = int(pred_dir[0])

            

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
