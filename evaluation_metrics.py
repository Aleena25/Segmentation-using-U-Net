import tensorflow as tf
import numpy as np
import tensorflow.keras.backend as K
def evaluation_results(X_test, y_test, model):
    print(X_test.shape, y_test.shape)
    y_pred = model.predict(X_test)
    print(y_pred.shape)
    y_pred = y_pred.reshape(y_pred.shape[0],256,256)
    y_test = (y_test > 0.3).astype(np.uint8)
    y_pred=(y_pred > 0.3).astype(np.uint8)
    y_true_f = tf.reshape(tf.dtypes.cast(y_test, tf.float32), [-1])
    y_pred_f = tf.reshape(tf.dtypes.cast(y_pred, tf.float32), [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    dice_score = (2. * intersection + 1.) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + 1.)
    print ("Dice Similarity: {}".format(dice_score))
    
def jaccard_coef(y_true, y_pred):
    intersection = K.sum(y_true * y_pred)
    union = K.sum(y_true + y_pred)
    jac = (intersection + 1.) / (union - intersection + 1.)
    print('Jaccard coefficient: {}'.format(jac))
    return K.mean(jac)    
    
