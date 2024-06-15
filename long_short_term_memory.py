import yfinance as yf
import pandas as pd
import numpy as np
import datetime
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, LSTM

# ตั้งวันที่เริ่มต้นและวันที่สิ้นสุด
start = datetime.datetime(2010, 1, 1)
end = datetime.datetime(2024, 6, 15)

# ดึงข้อมูลหุ้นไทยจาก Yahoo Finance
df = yf.download('BTS.BK', start=start, end=end)

# สเกลข้อมูล
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(df['Close'].values.reshape(-1,1))

# แบ่งข้อมูลเป็น training set และ test set
training_data_len = int(np.ceil(len(scaled_data) * 0.8))

train_data = scaled_data[0:int(training_data_len), :]
test_data = scaled_data[int(training_data_len):, :]

# สร้างชุดข้อมูลที่ใช้สำหรับการฝึก
def create_dataset(data, time_step=1):
    X, Y = [], []
    for i in range(len(data)-time_step-1):
        a = data[i:(i+time_step), 0]
        X.append(a)
        Y.append(data[i + time_step, 0])
    return np.array(X), np.array(Y)

time_step = 60
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

# ปรับรูปร่างข้อมูลสำหรับ LSTM
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# สร้างโมเดล LSTM
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dense(units=25))
model.add(Dense(units=1))

# คอมไพล์โมเดล
model.compile(optimizer='adam', loss='mean_squared_error')

# ฝึกโมเดล
model.fit(X_train, y_train, batch_size=1, epochs=1)

# ทำนายราคาหุ้นในอดีต
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)

# ประเมินโมเดล
rmse = np.sqrt(np.mean(((predictions - scaler.inverse_transform(y_test.reshape(-1,1)))**2)))
print(f'RMSE: {rmse}')

# ทำนายราคาหุ้นในอนาคต 30 วัน
last_60_days = scaled_data[-60:]
predicted_stock_price = []
for _ in range(30):
    X_pred = last_60_days[-60:].reshape(1, 60, 1)
    pred_price = model.predict(X_pred)
    predicted_stock_price.append(pred_price[0,0])
    last_60_days = np.append(last_60_days, pred_price)
    last_60_days = last_60_days[1:]

predicted_stock_price = scaler.inverse_transform(np.array(predicted_stock_price).reshape(-1,1))

# แสดงกราฟ
train = df[:training_data_len]
valid = df[training_data_len:]
valid = valid.iloc[:len(predictions)]  # Adjust the length of the validation set to match predictions
valid['Predictions'] = predictions

plt.figure(figsize=(16,8))
plt.title('LSTM Model')
plt.xlabel('Date')
plt.ylabel('Close Price USD')
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.plot(pd.date_range(start=end, periods=30, freq='B'), predicted_stock_price, color='red', linestyle='dashed', label='Future Predictions')
# plt.legend(['Train', 'Val', 'Predictions', 'Future Predictions'], loc='lower right')
plt.show()
