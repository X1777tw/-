import serial  # 引用pySerial模組本身led
from tensorflow import keras
import numpy as np

COM_PORT = 'COM3'    # 指定通訊埠名稱
BAUD_RATES = 9600    # 設定傳輸速率
ser = serial.Serial(COM_PORT, BAUD_RATES)   # 初始化序列通訊埠
demo1=b"0"
demo2=b"1"
demo3=b"2"
demo4=b"3"
demo5=b"4"
LSTM_Model = keras.models.load_model('my_model.h5')

# 總共會有20列筆資料
all_datas = []

try:
    while True:
        while ser.in_waiting:          # 若收到序列資料…
            data_raw = ser.readline()  # 讀取一行
            #pred_dir = LSTM_Model.predict(data_raw)

            temp = data_raw.decode()  # 用預設的UTF-8解碼
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

            all_datas = np.array(all_datas).astype(float)

            mean = np.load('mean.npy')
            max = np.load('max.npy')
            min = np.load('min.npy')

            nums = all_datas
            all_datas = []

            for j in range(20):
                nums[j] = [(i - mean[idx]) / (max[idx] - min[idx]) for idx, i in enumerate(nums[j])]


            # # (1, 20, 16)
            npdata = np.asarray(nums)[np.newaxis, :]
            pred_dir = LSTM_Model.predict_classes(npdata)

            print('預測:',pred_dir)
            pred_dir = int(pred_dir[0])
            if pred_dir == 0:
                ser.write(demo1)

            elif pred_dir == 1:
                 ser.write(demo2)

            elif pred_dir == 2:
                 ser.write(demo3)

            elif pred_dir == 3:
                 ser.write(demo4)

            elif pred_dir == 4:
                 ser.write(demo5)

            # #print('接收到的原始資料：', data_raw)
            # print('接收到的資料：', data2)
            # print('手勢指令:', data3)



except KeyboardInterrupt:
    ser.close()    # 清除序列通訊物件
    print('再見！')
