import numpy as np
import re
from sklearn.preprocessing import LabelEncoder

# Step 1: 清理 DNA 序列並轉換成字元陣列
def string_to_array(seq_string):
    """
    將 DNA 序列轉為小寫，替換非 A/C/G/T 為 'n'，並轉成 NumPy 陣列
    """
    seq_string = seq_string.lower() # 將序列轉為小寫
    seq_string = re.sub('[^acgt]', 'n', seq_string)    # 將非 A/C/G/T 的字元替換為 'n'
    return np.array(list(seq_string)) # 將字串轉為 NumPy 陣列

# Step 2: 建立標籤編碼器
label_encoder = LabelEncoder()
label_encoder.fit(np.array(['a', 'c', 'g', 't', 'z'])) # 使用 'z' 代替 'n' 作為未知字元的編碼佔位符

# Step 3: 將 DNA 序列轉換成序數特徵向量
def ordinal_encoder(dna_array):
    """
    將 DNA 陣列使用序數編碼轉換成 float 向量
    A=0.25, C=0.50, G=0.75, T=1.00, 其他（N）=0.00
    """
    dna_array = np.where(dna_array == 'n', 'z', dna_array)# 將 'n' 轉為 'z' 處理
    # np.where(condition, x, y) 是 NumPy 的條件選擇函數，當 condition 為 True 時，選擇 x，否則選擇 y
     
    integer_encoded = label_encoder.transform(dna_array) # 轉換為整數編碼
    
    # 映射為浮點數
    float_encoded = integer_encoded.astype(float) #  
    float_encoded[float_encoded == 0] = 0.25  # A
    float_encoded[float_encoded == 1] = 0.50  # C
    float_encoded[float_encoded == 2] = 0.75  # G
    float_encoded[float_encoded == 3] = 1.00  # T
    float_encoded[float_encoded == 4] = 0.00  # 其他（N）
    
    return float_encoded

# Step 4: 測試範例
if __name__ == "__main__":
    test_sequence = "TTCAGCCAGTG"
    cleaned_array = string_to_array(test_sequence)
    encoded_vector = ordinal_encoder(cleaned_array)
    
    print("原始序列：\n", test_sequence)
    print("字元陣列：\n", cleaned_array)
    print("編碼結果：\n", encoded_vector)
    
