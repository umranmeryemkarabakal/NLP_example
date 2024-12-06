# sentimentit polarity : duygu kutuplaşması
# bir metindeki duygunun pozitif, negatif veya nötr olduğunu belirleme işlemidir.

import numpy as np
import pandas as pd

comments = pd.read_csv('Restaurant_Reviews.csv')

import re
import nltk

from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

nltk.download('stopwords')
from nltk.corpus import stopwords
# corpus : metin külliyat

#Preprocessing (Önişleme)
corpus = [] # yorumların tutulduğu liste
for i in range(1000):
    # regular expression : düzenli ifade
    # metinlerde belirli desenleri tanımlamak ve bu desenlere göre metin arama, değiştirme veya ayrıştırma işlemleri yapmaktır
    # imla için 
    # alpha numeric karakterleri
    comment = re.sub('[^a-zA-Z]',' ',comments['Review'][i])
    # sub fonksiyonuyla harfler dışındaki karakterleri boşlukla değiştir, tüm harfleri küçük harfe çevir ve sadece alfabetik karakterleri bırak
    comment = comment.lower() # tüm metindeki harfleri küçültür
    comment = comment.split() # tüm kelimeleri listeye çevirir
    # stop words : durdurma sözcükleri 
    # anlamsız kelimeler
    # porter stemmer 
    # kelime gövdeleri köke indirgenir (stem)
    comment = [ps.stem(kelime) for kelime in comment if not kelime in set(stopwords.words('english'))]
    comment = ' '.join(comment) # tekrar cümle string formatına çevirir
    corpus.append(comment)


# Feautre Extraction ( Öznitelik Çıkarımı)
# Bag of Words (BOW)
# yorumlardaki kelimelerin sayısal temsilini oluşturur
from sklearn.feature_extraction.text import CountVectorizer
# count vectorizer 
# kelime frekanası hesaplama , öznitelik çıkarımı, boşluklu matris, max features
# kelime vektörü sayaç 
cv = CountVectorizer(max_features = 2000)
# sparce matrix : boşluklu matris
# matrixin çok büyük kısmı boşluktan oluşması
# kelime vektörü oluşturulur tekrarlar yakalanır
X = cv.fit_transform(corpus).toarray() # bağımsız değişken
y = comments.iloc[:,1].values # bağımlı değişken


# machine learning
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train,y_train)

y_pred = gnb.predict(X_test)

# evaluations değerlendirmeler
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test,y_pred)
print(cm)
print("accuracy score: ",accuracy_score(y_test,y_pred))
