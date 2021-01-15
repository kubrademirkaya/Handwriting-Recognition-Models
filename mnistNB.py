# kütüphaneler
print("Kütüphaneler yükleniyor")
from sklearn.metrics import classification_report
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.metrics import confusion_matrix
from tensorflow.keras.datasets import mnist

# mnist veriseti karışık olarak yükleniyor
print("veri seti yükleniyor")
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# veri önişleme
print("veri ön işleme yapılıyor")
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)

# görselleştirme
# plt.imshow(x_train[0].reshape(28, 28))

# normalizasyon işlemi
print("normalizasyon yapılıyor")
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# model oluşturuluyor
print("model oluşturuluyor")
modelNB = MultinomialNB()

# model eğitiliyor
print("model eğitiliyor")
modelNB.fit(x_train, y_train)

# test işlemi yapılıyor
print("test işlemi yapılıp, tahminler kaydediliyor")
y_pred = modelNB.predict(x_test)

# confusion matrix ve classification raporu ekrana yazdırılıyor
print("confusion matrix ve classification report oluşturuluyor")
cm = confusion_matrix(y_test, y_pred)
print("\n")
print(cm)

print("\n CLASSIFICATION REPORT \n")
print(classification_report(y_test, y_pred))

#confusion matrix grafiği yazdırılıyor
plt.subplot(1, 2, 2) 
plt.title('Naive Bayes \nAlgoritması', fontsize=16, color='purple')
plt.imshow(cm, interpolation='nearest',cmap=plt.cm.RdPu);
plt.tight_layout();
plt.colorbar();
plt.ylabel('True', color='purple');
plt.xlabel('Predicted', color='purple');
plt.xticks(np.arange(10));
plt.yticks(np.arange(10));
print()




















