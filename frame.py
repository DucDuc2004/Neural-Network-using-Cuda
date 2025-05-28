import matplotlib.pyplot as plt
import pandas as pd

# Đọc dữ liệu từ file CSV
df = pd.read_csv('/home/datto/code/Neural-Network-using-Cuda/training_results.csv')

# Khởi tạo biểu đồ
fig, ax1 = plt.subplots(figsize=(12, 6))

# Trục Y bên trái cho Loss và Accuracy
ax1.set_title("Training Summary: Loss, Accuracy and Time per Epoch")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss / Accuracy")
l1 = ax1.plot(df['Epoch'], df['Loss'], 'ro-', label='Loss')
l2 = ax1.plot(df['Epoch'], df['Accuracy'], 'b^-', label='Accuracy (%)')

# Trục Y bên phải cho Time
ax2 = ax1.twinx()
ax2.set_ylabel("Time (seconds)", color='green')
l3 = ax2.plot(df['Epoch'], df['Time (seconds)'], 'g-', label='Time (s)')

# Gộp legend từ cả hai trục
lines = l1 + l2 + l3
labels = [line.get_label() for line in lines]
ax1.legend(lines, labels, loc='upper center')

plt.tight_layout()
plt.show()
