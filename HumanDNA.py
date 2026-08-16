import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import os

# %matplotlib inline  # <- 這行在 Jupyter 中才需要

# 載入人類 DNA 資料
human_dna = pd.read_table('data/human.txt') 

# 顯示前幾列確認資料格式
print(human_dna.head())

# 繪製類別分佈圖
human_dna['class'].value_counts().sort_index().plot.bar()

# 加標題
plt.title("Class distribution of Human DNA")
plt.xlabel("Class")
plt.ylabel("Count")

# 顯示圖形
plt.show()
