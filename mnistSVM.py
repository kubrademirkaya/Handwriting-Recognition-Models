print("kütüphaneler yükleniyor")
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from keras.datasets import mnist
import numpy as np

print("veri seti yükleniyor")
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print("veri ön işleme yapılıyor")
x_train = x_train.reshape(x_train.shape[0], 784)
x_test = x_test.reshape(x_test.shape[0], 784)

print("normalizasyon yapılıyor")
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

print("model oluşturuluyor")
modelSVM = SVC(kernel='poly')

print("model eğitiliyor")
modelSVM.fit(x_train, y_train)

print("test işlemi yapılıp, tahminler kaydediliyor")
y_pred = modelSVM.predict(x_test)

print("confusion matrix olusturuluyor")
cm = confusion_matrix(y_test, y_pred);

print("\n CONFUSION MATRIX \n")
print( cm, "\n")

print("\n CLASSIFICATION REPORT \n")
print(classification_report(y_test, y_pred))


plt.subplot(1, 2, 1)    
plt.title('Destek Vektör \n Makinesi', fontsize=16, color='purple')
plt.imshow(cm, interpolation='nearest',cmap=plt.cm.RdPu);
plt.tight_layout();
plt.colorbar();
plt.ylabel('True', color='purple')
plt.xlabel('Predicted', color='purple');
plt.xticks(np.arange(10));
plt.yticks(np.arange(10));

