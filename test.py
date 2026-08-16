import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import seaborn as sns

# 載入DNA 資料
human_dna = pd.read_table('data/human.txt')
chimp_dna = pd.read_table('data/chimpanzee.txt')
dog_dna = pd.read_table('data/dog.txt')

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
X_human = cv.fit_transform(human_texts)
X_chimp = cv.transform(chimp_texts)
X_dog = cv.transform(dog_texts)

# Step 6: 拆分訓練與測試資料 (for Human DNA only, as per original setup)
X_train, X_test, y_train, y_test = train_test_split(X_human, y_human, test_size=0.20, random_state=42)

# Step 7: 訓練 Naive Bayes 分類器並擬合
classifier = MultinomialNB(alpha=0.1)
classifier.fit(X_train, y_train) # Classifier trained on human DNA

def get_metrics(y_true, y_predicted):
    accuracy = accuracy_score(y_true, y_predicted)
    precision = precision_score(y_true, y_predicted, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_predicted, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_predicted, average='weighted', zero_division=0)
    return accuracy, precision, recall, f1

# Step 8: 預測人類、黑猩猩和狗的 DNA 序列
y_pred_human = classifier.predict(X_test)
y_pred_chimp = classifier.predict(X_chimp)
y_pred_dog = classifier.predict(X_dog)


print("--- Metrics for Human DNA Predictions ---")
accuracy, precision, recall, f1 = get_metrics(y_test, y_pred_human)
print("accuracy = %.3f \nprecision = %.3f \nrecall = %.3f \nf1 = %.3f\n" % (accuracy, precision, recall, f1))

cm_human = confusion_matrix(y_test, y_pred_human)
class_labels_human = sorted(list(np.unique(y_human)))

cm_df_human = pd.DataFrame(cm_human, index=class_labels_human, columns=class_labels_human)

plt.figure(figsize=(8, 6))
sns.heatmap(cm_df_human, annot=True, fmt='d', cmap='Blues', cbar=True, linewidths=.5)
plt.title('Confusion Matrix for Human DNA Predictions')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()


print("\n--- Metrics for Chimpanzee DNA Predictions ---")
accuracy, precision, recall, f1 = get_metrics(y_chim, y_pred_chimp) # Use y_chim for true labels
print("accuracy = %.3f \nprecision = %.3f \nrecall = %.3f \nf1 = %.3f\n" % (accuracy, precision, recall, f1))

cm_chimp = confusion_matrix(y_chim, y_pred_chimp)
class_labels_chimp = sorted(list(np.unique(y_chim)))

cm_df_chimp = pd.DataFrame(cm_chimp, index=class_labels_chimp, columns=class_labels_chimp)

plt.figure(figsize=(8, 6))
sns.heatmap(cm_df_chimp, annot=True, fmt='d', cmap='YlGnBu', cbar=True, linewidths=.5)
plt.title('Confusion Matrix for Chimpanzee DNA Predictions')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()


print("\n--- Metrics for Dog DNA Predictions ---")
accuracy, precision, recall, f1 = get_metrics(y_dog, y_pred_dog) 
print("accuracy = %.3f \nprecision = %.3f \nrecall = %.3f \nf1 = %.3f\n" % (accuracy, precision, recall, f1))

cm_dog = confusion_matrix(y_dog, y_pred_dog)
class_labels_dog = sorted(list(np.unique(y_dog)))

cm_df_dog = pd.DataFrame(cm_dog, index=class_labels_dog, columns=class_labels_dog)

plt.figure(figsize=(8, 6))
sns.heatmap(cm_df_dog, annot=True, fmt='d', cmap='Reds', cbar=True, linewidths=.5)
plt.title('Confusion Matrix for Dog DNA Predictions')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()