from Augmentation_ import Augment_img
from preprocess import *

import glob
from train_test_split import train_test_split
from unet import unet
from tensorflow.keras.optimizers import Adam 
from tensorflow.keras.callbacks import EarlyStopping ,ReduceLROnPlateau ,ModelCheckpoint ,TensorBoard

data_path = r'E:\Aleena\segmentation'
train_data_path , test_data_path = train_test_split(data_path)
train_out_dir = Augment_img(data_path)


X_train, y_train = PreprocessData(train_out_dir)

X_test, y_test = PreprocessData_test(test_data_path)

model = unet(input_size=(256,256,3))
model.summary()
optimizer_adam = Adam(learning_rate=0.001,beta_1=0.9,beta_2=0.99)
model.compile(optimizer=optimizer_adam,loss='binary_crossentropy',metrics=['accuracy'])
history=model.fit(X_train, y_train, validation_data=(X_test,y_test),epochs=10,batch_size=16, verbose=1)

model.save('UNET.h5')

