import numpy as np
import re
from sklearn.preprocessing import LabelEncoder

# Step 1: 清理 DNA 序列並轉換成字元陣列
def string_to_array(seq_string):
    seq_string = seq_string.lower() 
    seq_string = re.sub('[^acgt]', 'n', seq_string)
    return np.array(list(seq_string))

# Step 2: 建立標籤編碼器
label_encoder = LabelEncoder()
label_encoder.fit(np.array(['a', 'c', 'g', 't', 'z']))

# Step 3: 將 DNA 序列轉換成序數特徵向量
def ordinal_encoder(dna_array):
    dna_array = np.where(dna_array == 'n', 'z', dna_array)
    integer_encoded = label_encoder.transform(dna_array)
    float_encoded = integer_encoded.astype(float)
    
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
    
