from tensorflow.keras.models import load_model
from preprocess import PreprocessData_test
import matplotlib.pyplot as plt
from evaluation_metrics import evaluation_results, jaccard_coef
import numpy as np
import tensorflow as tf



model = load_model('UNET.h5') #same file path
X_test, y_test = PreprocessData_test(r'E:\Aleena\segmentation\Test_data')
model.evaluate(X_test, y_test)

# VisualizeResults(index):
index = 15
img = X_test[index,:,:]
img = img.reshape(1, 256,256,3)
pred_y = model.predict(img)
pred_y = pred_y.reshape(256,256)
#pred_y = pred_y.reshape(256,256)
#Visualize
plt.imshow(X_test[index])
plt.title('Processed Image')
plt.show()
plt.imshow(y_test[index])
plt.title('Actual Masked Image ')
plt.show()
pred_y = (pred_y > 0.1).astype(np.uint8)
plt.imshow(pred_y)
plt.title('Predicted Masked Image ')
plt.show()

#jaccard coefficient
jaccard_coef(y_test[index], pred_y)
#Dice score
evaluation_results(X_test, y_test, model)
