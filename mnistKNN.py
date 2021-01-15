import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.datasets import mnist
from sklearn.neighbors import KNeighborsClassifier

print("veriler yükleniyor")
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# veri önişleme
print("veri ön işleme yapılıyor")
x_train = x_train.reshape(60000,784)
x_test = x_test.reshape(10000,784)

# normalizasyon
print("normalizasyon işlemi yapılıyor")
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

print("model oluşturuluyor")
knn = KNeighborsClassifier(n_neighbors=1, metric='minkowski')

print("model eğitiliyor")
knn.fit(x_train,y_train)

print("tahmin işlemi yapılıp kaydediliyor")
y_pred = knn.predict(x_test)

#print("tahmin sonuçlarına son işleme uygulanıyor")
#y_pred=np.argmax(y_pred, axis=1)

# confusion matrix ve classification raporu ekrana yazdırılıyor
print("confusion matrix ve classification report oluşturuluyor")
print("\n CONFUSION MATRIX \n")
cm = confusion_matrix(y_test, y_pred)
print("\n")
print(cm)

print("\n CLASSIFICATION REPORT \n")
print(classification_report(y_test, y_pred))

#confusion matrix grafiği yazdırılıyor
plt.subplot(1, 2, 2) 
plt.title('K-Nearest Neighbors \nAlgoritması', fontsize=16, color='purple')
plt.imshow(cm, interpolation='nearest',cmap=plt.cm.RdPu);
plt.tight_layout();
plt.colorbar();
plt.ylabel('True', color='purple');
plt.xlabel('Predicted', color='purple');
plt.xticks(np.arange(10));
plt.yticks(np.arange(10));
print()

