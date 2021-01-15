#kütüphaneler
print("kütüphaneler yükleniyor")
import numpy as np
from keras.datasets import mnist
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

print("veri seti yükleniyor")
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#veri ön işleme
print("veri ön işleme yapılıyor")
num_pixels = x_train.shape[1] * x_train.shape[2]
x_train = x_train.reshape(x_train.shape[0], num_pixels).astype('float32')
x_test = x_test.reshape(x_test.shape[0], num_pixels).astype('float32')

#model oluşturuluyor
print("model oluşturuluyor")
modelDT = DecisionTreeClassifier(criterion='gini', #default
                                 min_impurity_decrease=0.0, #default
                                 min_samples_leaf=1, #default
                                 min_samples_split=2, #default
                                 min_weight_fraction_leaf=0.0, #default
                                 splitter='random');
                            
print("model eğitiliyor")
modelDT.fit(x_train,y_train)

print("model test edilip tahminler kaydediliyor")
y_pred = modelDT.predict(x_test)

print("confusion matrix ve classification report oluşturuluyor")
cm = confusion_matrix(y_test, y_pred);

print("\n CONFUSION MATRIX \n")
print( cm, "\n")

print("\n CLASSIFICATION REPORT \n")
print(classification_report(y_test, y_pred))

#confusion matrix grafiği yazdırılıyor
plt.subplot(1, 2, 1)    
plt.title('Decision Tree \nAlgoritması', fontsize=16, color='purple')
plt.imshow(cm, interpolation='nearest',cmap=plt.cm.RdPu);
plt.tight_layout();
plt.colorbar();
plt.ylabel('True', color='purple')
plt.xlabel('Predicted', color='purple');
plt.xticks(np.arange(10));
plt.yticks(np.arange(10));
