# Step 1: k-mer - 將序列依size切割成 k-mer
def Kmers_funct(seq, size):
    return [seq[x:x+size].lower() for x in range(len(seq) - size + 1)]

# Step 2: DNA 序列
mySeq0 = 'GTGCCCAGGTTCAGTGGGCAGGTGCCCAGGTTC'
mySeq1 = 'GTGCCCAGGTTCAGTGAGTGACACAGGCAG'
mySeq2 = 'TCTCACACATGTGCCAATCACTGTCACCC'

# Step 3: 將序列轉為 k-mer「句子」，k=6
k = 6
sentence0 = ' '.join(Kmers_funct(mySeq0, size=k))
sentence1 = ' '.join(Kmers_funct(mySeq1, size=k))
sentence2 = ' '.join(Kmers_funct(mySeq2, size=k))

print("Sentence :\n", sentence0)
# print("Sentence 1:\n", sentence1)
# print("Sentence 2:\n", sentence2)

# Step 4: Bag of Words 模型
# 將文本轉換成詞頻矩陣，再用fit_transform計算出現地頻率
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform([sentence0, sentence1,sentence2]).toarray()
cv.get_feature_names_out()

# Step 5: 印出結果
print("\nFeature Names (詞彙表):")
print(cv.get_feature_names_out())
print("\nBag of Words Vectors:")
print(X)

