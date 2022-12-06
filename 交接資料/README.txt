1way0 是裝置0度的單一方向資料
1way0+90 是裝置0度跟90度的兩個方向的資料
1way0+90+180 是裝置0度跟90度跟180度三個方向的資料
4way是4個方向(0、90、180、270)的資料
// 各自對應到各自的.h5



*******   arduinoread.py是用來執行裝置的 *****
*******   LSTM.py是用來訓練資料的 ****
*******   demo.h5是最終版本的模型 ****
******    ard.ino是arduino的code  ****
******    pi.txt是樹莓派的版本*****
*******   piEnvironment.yaml是anaconda的版本 ****

其他可能會用到的:

// 改dir答案.py是用來改csv的dir的column用的
// writefile.py是用來把揮的手勢資料寫進csv用的
// LSTMstatic.py是用來測試單獨訓練靜態資料的
// envread.py是用來蒐集減去初始環境光度的資料
// CNN.py是用CNN的訓練方式訓練靜態資料
// 4wayModify.py是用軟體的方式做資料轉向
// read.py是各自用靜態跟動態的model去各自做預測(用中位數來判斷是靜態還是動態)

 