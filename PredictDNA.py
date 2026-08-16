import pandas as pd
import numpy as np
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# 載入DNA 資料
human_dna = pd.read_table('data/human.txt')
chimp_dna = pd.read_table('data/chimpanzee.txt') 
dog_dna  = pd.read_table('data/dog.txt')

# Step 1: 將 DNA 序列轉為 k-mer 詞語
def Kmers_funct(seq, size=6):
    return [seq[x:x+size].lower() for x in range(len(seq) - size + 1)]

# Step 2: 套用 k-mer
human_dna['words'] = human_dna.apply(lambda x: Kmers_funct(x['sequence']), axis=1)
chimp_dna['words'] = chimp_dna.apply(lambda x: Kmers_funct(x['sequence']), axis=1)
dog_dna['words'] = dog_dna.apply(lambda x: Kmers_funct(x['sequence']), axis=1)

# Step 3: 將詞語清單轉為句子字串
human_texts = [' '.join(kmers) for kmers in human_dna['words']]
chimp_texts = [' '.join(kmers) for kmers in chimp_dna['words']]
dog_texts = [' '.join(kmers) for kmers in dog_dna['words']]

# Step 4: 建立類別標籤 y
y_human = human_dna['class'].values
y_chim = chimp_dna['class'].values
y_dog = dog_dna['class'].values


# Step 5: 使用 CountVectorizer 建立特徵矩陣（Bag of Words）
cv = CountVectorizer(ngram_range=(4,4))
X = cv.fit_transform(human_texts)
X_chimp = cv.transform(chimp_texts)
X_dog = cv.transform(dog_texts)


# Step 6: 拆分訓練與測試資料
X_train, X_test, y_train, y_test = train_test_split(X, y_human, test_size=0.20, random_state=42)


# Step 7: 訓練 Naive Bayes 分類器並擬合
classifier = MultinomialNB(alpha=0.1)
classifier.fit(X_train, y_train)


def get_metrics(y_test, y_predicted):
    accuracy = accuracy_score(y_test, y_predicted)
    precision = precision_score(y_test, y_predicted, average='weighted')
    recall = recall_score(y_test, y_predicted, average='weighted')
    f1 = f1_score(y_test, y_predicted, average='weighted')
    return accuracy, precision, recall, f1


# Step 8: 預測人類、黑猩猩和狗的 DNA 序列
y_pred = classifier.predict(X_test)

print("Confusion matrix for predictions on human test DNA sequence\n")
print(pd.crosstab(pd.Series(y_test, name='Actual'), pd.Series(y_pred, name='Predicted')))
accuracy, precision, recall, f1 = get_metrics(y_test, y_pred)
print("accuracy = %.3f \nprecision = %.3f \nrecall = %.3f \nf1 = %.3f\n" % (accuracy, precision, recall, f1))

y_pred_chimp = classifier.predict(X_chimp)

print("Confusion matrix for predictions on Chimpanzee test DNA sequence\n")
print(pd.crosstab(pd.Series(y_chim, name='Actual'), pd.Series(y_pred_chimp, name='Predicted')))
accuracy, precision, recall, f1 = get_metrics(y_chim, y_pred_chimp)
print("accuracy = %.3f \nprecision = %.3f \nrecall = %.3f \nf1 = %.3f\n" % (accuracy, precision, recall, f1))

y_pred_dog = classifier.predict(X_dog)

print("Confusion matrix for predictions on Dog test DNA sequence\n")
print(pd.crosstab(pd.Series(y_dog, name='Actual'), pd.Series(y_pred_dog, name='Predicted')))
accuracy, precision, recall, f1 = get_metrics(y_dog, y_pred_dog)
print("accuracy = %.3f \nprecision = %.3f \nrecall = %.3f \nf1 = %.3f\n" % (accuracy, precision, recall, f1))
