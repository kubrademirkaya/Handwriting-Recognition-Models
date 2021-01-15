#from __future__ import print_function
# kütüphaneler
print("kütüphaneler yükleniyor")
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
import numpy as np
from sklearn.metrics import classification_report

# her bir döngüde "128" resim alınıyor
batch_size = 128
# ayırt etmek istediğimiz "10" rakam (0-9)
num_classes = 10 
# eğitim 12 döngü kadar sürüyor
epochs = 30

# veri seti yükleniyor
print("veri seti yükleniyor")
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# veri önişleme
print("veri önişleme yapılıyor")
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# veriler hakkında bilgilendirme 
print('\neğitim verileri giriş şekli:', x_train.shape)
print(x_train.shape[0], 'eğitim verileri')
print(x_test.shape[0], 'test verileri')

# sınıf vektörleri binary formununa dönüştürülüyor
# "to_catogorical" fonksiyonu ile one-hot-encoding yapılıyor
print("\nverilere OHE uygulanıyor")
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

##
# Ağımızı kuralım
# Burada basit bir 3 katmanlı tam bağlantılı ağ yapacağız.(fully connected network ("Dense"))
# Ayrıca eğitim sırasında aşırı öğrenme (overfitting) olmaması için bırakma/atılma ("Dropout") uygulayacağız.
# Dropout tekniği 2014 yılında bir makale de önerilmiştir ve o zamandan beri benimsenerek kullanılmıştır. (http://jmlr.org/papers/v15/srivastava14a.html)
# Pratikte %20 ile %50 arasında dropout uygulandığı görülüyor.

print("YSA-MLP yapısı oluşturuluyor")
model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(784,)))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax')) 

#oluşturulan sinir ağının özet tablosu
print("\n\nOluşturulan YSA-MLP modeli ayrıntıları")
model.summary()

# Ağ derlenirken kayıp fonksiyonunuzu ve optimizatörünüzü belirtmemiz isteniyor.
# Optimizasyon tiplerinden "RMSprop" ve yitim (loss) fonksiyonu olarak "categorical_crossentropy" kullanıyoruz.
# Son parametre olarak da hali hazırda tanımlanmış tek metrik fonksiyonu olan "accuracy" yi kullanıyoruz. Accuracy bize 0-1 arasında bir doğruluk değeri verecektir.
print("oluşturulan YSA-MLP modeli derleniyor")
model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

# Burası en eğlenceli kısım, ağımızı eğitme kısmı :)
# Daha önce yüklenmiş eğitim verileri ile sınıflandırmayı öğrenmeye çalışır.
# Fit fonksiyonu eğitim sırasındaki yitim(loss)/doğrulama başarımı (accuracy) değerleri ile bir çok ayrıntı döndürür.
# eğitim işlemi
print("model eğitiliyor")
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))


# tahminleri y_pred değişkenine kaydededip one-hot-encoding yapılıyor
print("test işlemi yapılıp tahminler kaydediliyor")
y_pred=model.predict(x_test)
y_pred=np.argmax(y_pred, axis=1)
y_pred = keras.utils.to_categorical(y_pred, num_classes)

# classification report hazırlanıp ekrana yazdırılıyor
print("classification report ve diğer performans değerleri hazırlanıyor \n\n\n")
print("\n CLASSIFICATION REPORT \n")
print(classification_report(y_test, y_pred))
print("\n")
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

