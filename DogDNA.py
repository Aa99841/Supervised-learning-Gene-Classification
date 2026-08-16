import numpy as np  # 數值運算
import pandas as pd  # 數據處理
import matplotlib.pyplot as plt  # 繪圖工具
import os

# 載入 DNA 資料
dog_dna  = pd.read_table('data/dog.txt')

# 顯示前幾列確認資料格式
print(dog_dna .head())

# 繪製類別分佈圖
dog_dna ['class'].value_counts().sort_index().plot.bar()

# 加標題
plt.title("Class distribution of Human DNA")
plt.xlabel("Class")
plt.ylabel("Count")

# 顯示圖形
plt.show()
