import numpy as np
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

# Step 1: 清理 DNA 序列並轉換成字元陣列
def string_to_array(seq_string):
    """
    將 DNA 序列轉為小寫，替換非 A/C/G/T 為 'n'，並轉成 NumPy 陣列
    """
    seq_string = seq_string.lower()
    seq_string = re.sub('[^acgt]', 'n', seq_string)
    return np.array(list(seq_string))

# Step 2: 建立標籤編碼器
label_encoder = LabelEncoder()
label_encoder.fit(np.array(['a', 'c', 'g', 't', 'z']))

def one_hot_encoder(seq_string):
    seq_string = np.where(seq_string == 'n', 'z', seq_string)
    int_encoded = label_encoder.transform(seq_string)
    onehot_encoder = OneHotEncoder(sparse_output=False, dtype=int) # sparse_output=False 會返回一個密集的 NumPy 陣列
    int_encoded = int_encoded.reshape(len(int_encoded), 1) # 將一維陣列轉為二維陣列
    onehot_encoded = onehot_encoder.fit_transform(int_encoded)
    onehot_encoded = np.delete(onehot_encoded, -1, 1) # 刪除最後一列（'n' 的編碼）
    print(onehot_encoded)
    return onehot_encoded


#So let’s try it out with a simple short sequence:
seq_test = 'GAATTCTCGAA'
print(one_hot_encoder(string_to_array(seq_test)))