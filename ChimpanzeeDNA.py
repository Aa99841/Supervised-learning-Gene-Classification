import numpy as np  # 數值運算
import pandas as pd  # 數據處理
import matplotlib.pyplot as plt  # 繪圖工具
import os

# 載入DNA 資料
chimp_dna = pd.read_table('data/chimpanzee.txt')

# 顯示前幾列確認資料格式
print(chimp_dna.head())

# 繪製類別分佈圖
chimp_dna['class'].value_counts().sort_index().plot.bar()

# 加標題
plt.title("Class distribution of Human DNA")
plt.xlabel("Class")
plt.ylabel("Count")

# 顯示圖形
plt.show()
