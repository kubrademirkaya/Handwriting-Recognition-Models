print("kütüphaneler yükleniyor")
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
import numpy as np

num_classes = 10
batch_size = 128
epochs = 30

#Veri setinin yüklenmesi
print("veri seti yükleniyor")
(x_train, y_train),(x_test, y_test) = mnist.load_data()

#Sinir ağları için veri ön işleme
print("veri önişleme yapılıyor")
img_rows = 28
img_cols = 28

x_train = x_train.reshape( x_train.shape[0],img_rows,img_cols,1)
x_test = x_test.reshape( x_test.shape[0],img_rows,img_cols,1)
input_shape = (img_rows,img_cols,1)

#normalizasyon
x_train = x_train.astype("float32")
x_test = x_test.astype("float32")

x_train /= 255 
x_test /= 255

#One Hot Encoding işlemi
print("\nverilere OHE uygulanıyor")
y_train = keras.utils.to_categorical(y_train,num_classes)
y_test = keras.utils.to_categorical(y_test,num_classes)

#CNN yapısı
print("CNN yapısı oluşturuluyor")
model = Sequential()

model.add(Conv2D(16, kernel_size = (3, 3),
                     activation = "relu",
                     input_shape = input_shape))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Conv2D(32, kernel_size = (3, 3),
                 activation = "relu"))
model.add(Conv2D(64, kernel_size = (3, 3),
                 activation = "relu"))
model.add(Conv2D(128, kernel_size = (3, 3),
                 activation = "relu"))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.25))    
model.add(Flatten())
model.add(Dense(256, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(num_classes,
                activation = "softmax"))

print("\n Oluşturulan CNN modeli ayrıntıları \n")
model.summary()
    
model.compile(loss="categorical_crossentropy", 
              optimizer="adadelta",
              metrics=["accuracy"])
    
model.fit(x_train, y_train, 
          batch_size = batch_size, 
          epochs = epochs, 
          verbose = 1,
          validation_data = (x_test,y_test))


y_pred=model.predict(x_test)
y_pred=np.argmax(y_pred, axis=1)
y_pred = keras.utils.to_categorical(y_pred, num_classes)

# classification report hazırlanıp ekrana yazdırılıyor
print("classification report ve diğer performans değerleri hazırlanıyor \n\n\n")
from sklearn.metrics import classification_report
print("\n CLASSIFICATION REPORT \n")
print(classification_report(y_test, y_pred))
print("\n")
score = model.evaluate(x_test, y_test, verbose=0)
print("\n" "'ADADELTA' FONKSİYONU KULLANILAN CNN MODELİ")
print("Test loss:", score[0])
print("Test accuracy:", score[1])

