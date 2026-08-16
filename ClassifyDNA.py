import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import os

# 載入人類 DNA 資料
human_dna = pd.read_table('data/human.txt')
# 載入黑猩猩 DNA 資料
chimp_dna = pd.read_table('data/chimpanzee.txt') 
# 載入狗 DNA 資料
dog_dna  = pd.read_table('data/dog.txt')

# Step 1: 將 DNA 序列轉為 k-mer 詞語
def Kmers_funct(seq, size=6):
    return [seq[x:x+size].lower() for x in range(len(seq) - size + 1)]

# Step 2: 套用 k-mer
# 將每一筆資料中的 sequence 欄位轉為 k-mer 詞語清單（list），並存到新欄位 words 中
human_dna['words'] = human_dna.apply(lambda x: Kmers_funct(x['sequence']), axis=1)
chimp_dna['words'] = chimp_dna.apply(lambda x: Kmers_funct(x['sequence']), axis=1)
dog_dna['words'] = dog_dna.apply(lambda x: Kmers_funct(x['sequence']), axis=1)

print("Step 2 輸出：\n Human:", human_dna[['class', 'words']].head())
print("\n Chimp: ", chimp_dna[['class', 'words']].head())
print("\n Dog: ", dog_dna[['class', 'words']].head())

# Step 3: 將詞語清單轉為句子字串
# humana_dna 是 DataForm不是 String ，所以需要將每個 k-mer 列表轉換為字符串
human_texts = [' '.join(kmers) for kmers in human_dna['words']]
chimp_texts = [' '.join(kmers) for kmers in chimp_dna['words']]
dog_texts = [' '.join(kmers) for kmers in dog_dna['words']]

print("Step 3 輸出：\n Human: ", human_texts[0][:100], "...")
print("\n Chimp: ", chimp_texts[0][:100], "...")
print("\n Dog: ", dog_texts[0][:100], "...")

# Step 4: 建立類別標籤 y
y_human = human_dna['class'].values
y_chim = chimp_dna['class'].values
y_dog = dog_dna['class'].values

print("Step 4 輸出：\n Human:", y_human[:10])
print("\n Chimp: ", y_chim[:10])
print("\n Dog: ", y_dog[:10])

# Step 5: 使用 CountVectorizer 建立特徵矩陣（Bag of Words）
cv = CountVectorizer(ngram_range=(4,4))
X = cv.fit_transform(human_texts)
X_chimp = cv.transform(chimp_texts)
X_dog = cv.transform(dog_texts)

print("Step 5 輸出：")
print("Human shape:", X.shape)
print("Chimp shape:", X_chimp.shape)
print("Dog shape:", X_dog.shape)


# Step 6: 拆分訓練與測試資料
X_train, X_test, y_train, y_test = train_test_split(X, y_human, test_size=0.20, random_state=42)

print("Step 6 輸出：")
print("X_train.shape:", X_train.shape)
print("X_test.shape:", X_test.shape)

# Step 7: 訓練 Naive Bayes 分類器並擬合
classifier = MultinomialNB(alpha=0.1)
classifier.fit(X_train, y_train)

print("Step 7 輸出：", classifier)
