# Import các thư viện cần thiết
from datetime import timedelta
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import os

# Tải dữ liệu
file_path = r'C:\Users\USER\OneDrive - uel.edu.vn\Documents\UiPath\Project_1\Data Bitcoin\Data.xlsx'
df = pd.read_excel(file_path)

# Nhóm dữ liệu theo ngày và tính giá đóng cửa trung bình mỗi ngày
df_close = df.groupby('Open Time')['Close'].mean().reset_index()
df_close['Open Time'] = pd.to_datetime(df_close['Open Time'])
df_close.set_index('Open Time', inplace=True)

# Chuẩn hóa dữ liệu
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(np.array(df_close['Close']).reshape(-1, 1))

# Chia dữ liệu thành tập huấn luyện và kiểm tra
train_size = int(len(scaled_data) * 0.75)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size - 60:]

# Chuẩn bị dữ liệu huấn luyện cho LSTM
def create_dataset(data, n_steps):
    x, y = [], []
    for i in range(n_steps, len(data)):
        x.append(data[i-n_steps:i])
        y.append(data[i])
    return np.array(x), np.array(y)

n_steps = 60
x_train, y_train = create_dataset(train_data, n_steps)

x_train = torch.tensor(x_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)

# Định nghĩa mô hình LSTM
class LSTMModel(nn.Module):
    def __init__(self):
        super(LSTMModel, self).__init__()
        self.lstm1 = nn.LSTM(input_size=1, hidden_size=100, num_layers=2, batch_first=True, dropout=0.2)
        self.fc1 = nn.Linear(100, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        out, _ = self.lstm1(x)
        out = out[:, -1, :]  # Chỉ lấy đầu ra cuối cùng của chuỗi
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out

# Tạo và huấn luyện mô hình
model_LSTM = LSTMModel()
criterion = nn.MSELoss()
optimizer = optim.Adam(model_LSTM.parameters(), lr=0.001)

# Huấn luyện mô hình và lưu lịch sử lỗi
epochs = 200  # Tăng số lượng epochs
history = {'loss': [], 'mean_absolute_error': []}

for epoch in range(epochs):
    model_LSTM.train()
    optimizer.zero_grad()
    outputs = model_LSTM(x_train.view(-1, n_steps, 1))
    loss = criterion(outputs, y_train.view(-1, 1))
    loss.backward()
    optimizer.step()

    # Lưu lỗi qua các epochs
    history['loss'].append(loss.item())
    mae = mean_absolute_error(y_train.detach().numpy(), outputs.detach().numpy())
    history['mean_absolute_error'].append(mae)

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}, MAE: {mae:.4f}')

# Dự đoán trên tập huấn luyện
model_LSTM.eval()
y_pred_train = model_LSTM(x_train.view(-1, n_steps, 1)).detach().numpy()

# Tính toán chỉ số lỗi
mae = mean_absolute_error(y_train.numpy(), y_pred_train)
mse = mean_squared_error(y_train.numpy(), y_pred_train)
r2 = r2_score(y_train.numpy(), y_pred_train)

# Lưu kết quả dự đoán và lỗi vào file Excel
results = pd.DataFrame({
    "True Price": y_train.numpy().flatten(),
    "Predicted Price": y_pred_train.flatten()
})

# Đường dẫn tương đối tới tệp cần lưu
file_path1 = "C:/Users/USER/OneDrive - uel.edu.vn/Documents/UiPath/Project_1/Report/Predicted_Bitcoin_Prices.xlsx"
os.makedirs(os.path.dirname(file_path1), exist_ok=True)

# Lưu tệp vào đường dẫn đã chỉ định
results.to_excel(file_path1, index=False)

# Lưu các chỉ số lỗi vào file khác
metrics = pd.DataFrame({
    "Metric": ["Mean Absolute Error", "Mean Squared Error", "R^2 Score"],
    "Value": [mae, mse, r2]
})
file_path1 = "C:/Users/USER/OneDrive - uel.edu.vn/Documents/UiPath/Project_1/Report/Prediction_Metrics.xlsx"
metrics.to_excel(file_path1, index=False)

# Vẽ biểu đồ so sánh True vs Predicted Prices
plt.figure(figsize=(10, 5))
plt.plot(y_train.numpy(), label='True Data')
plt.plot(y_pred_train, label='Predicted Data')
plt.legend()
plt.title("True vs Predicted Prices")
plt.xlabel("Time Steps")
plt.ylabel("Price")
image_path = "C:/Users/USER/OneDrive - uel.edu.vn/Documents/UiPath/Project_1/Report/Training_Plot.png"
plt.savefig(image_path)
plt.close()

# Dự đoán giá trên tập kiểm tra
x_test, y_test = create_dataset(test_data, n_steps)
x_test = torch.tensor(x_test, dtype=torch.float32)

# Dự đoán giá trị sử dụng mô hình
model_LSTM.eval()
predictions = model_LSTM(x_test.view(-1, n_steps, 1)).detach().numpy()

# Đảo ngược tỉ lệ hóa của các dự đoán để thu được các giá trị thực tế
predictions = scaler.inverse_transform(predictions)

# Đảo ngược việc tỉ lệ hóa trên dữ liệu đầu ra của tập kiểm tra để thu được các giá trị thực tế
y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

# Vẽ biểu đồ dự đoán
plt.figure(figsize=(16, 6))
plt.title('Dự đoán giá đóng cửa Bitcoin', fontsize=18)
plt.xlabel('Ngày', fontsize=18)
plt.ylabel('Giá đóng cửa', fontsize=18)
plt.plot(df_close.index[train_size:], y_test, label='Giá thực tế', linewidth=2)
plt.plot(df_close.index[train_size:], predictions, label='Giá dự đoán', linewidth=2)
plt.legend()
plt.grid()
image_path = "C:/Users/USER/OneDrive - uel.edu.vn/Documents/UiPath/Project_1/Report/Bitcoin_Price_Predictions.png"
plt.savefig(image_path)
plt.close()

# Dự đoán tương lai
n_future = 30
future_forecast = []

input_data = scaled_data[-n_steps:]

for _ in range(n_future):
    input_tensor = torch.FloatTensor(input_data).view(1, n_steps, 1)
    with torch.no_grad():
        pred_price = model_LSTM(input_tensor)
        future_forecast.append(pred_price.item())
        input_data = np.append(input_data[1:], pred_price.numpy())

# Chuyển đổi kết quả dự đoán về dạng numpy
future_forecast = np.array(future_forecast).reshape(-1, 1)

# Đảo ngược chuẩn hóa để thu được giá thực
forecast_prices = scaler.inverse_transform(future_forecast)

# Tạo các ngày để gán cho dự đoán
last_date = df_close.index[-1]
future_dates = [last_date + timedelta(days=i) for i in range(1, n_future + 1)]

# Tạo DataFrame cho dự đoán tương lai
future_df = pd.DataFrame(forecast_prices, columns=['Predicted Price'], index=future_dates)

file_path1 = "C:/Users/USER/OneDrive - uel.edu.vn/Documents/UiPath/Project_1/Report/Future_Bitcoin_Prices.xlsx"
# Lưu kết quả vào file Excel
future_df.to_excel(file_path1)

# Vẽ biểu đồ dự đoán giá Bitcoin trong 30 ngày tới
plt.figure(figsize=(10, 5))
plt.plot(df_close.index, df_close['Close'], label='Giá thực tế', color='blue', linewidth=2)
plt.plot(future_df.index, future_df['Predicted Price'], label='Dự đoán trong 30 ngày tới', color='orange', linestyle='--', linewidth=2)
plt.title('Dự đoán giá Bitcoin trong 30 ngày tới', fontsize=18)
plt.xlabel('Ngày', fontsize=14)
plt.ylabel('Giá Bitcoin', fontsize=14)
plt.legend()
plt.xticks(rotation=45)
plt.grid()

image_path = "C:/Users/USER/OneDrive - uel.edu.vn/Documents/UiPath/Project_1/Report/Future_Bitcoin_Price_Predictions.png"
# Lưu biểu đồ vào file
plt.savefig(image_path)