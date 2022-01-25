import numpy as np
import glob
import cv2
import os
def PreprocessData(data_path):
    imgs = glob.glob(os.path.join(data_path, 'Train','*.jpg'))
    masks = glob.glob(os.path.join(data_path ,'Labels','*.png'))
    
    X = []
    y = []
    for i in range(len(imgs)):
        img = cv2.imread(imgs[i])
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img = cv2.resize(img,(256,256))
        img = img/255
        X.append(img)
        
        mask = cv2.imread(masks[i])
        mask = cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
        mask = cv2.resize(mask,(256,256))
        mask = mask/255
        y.append(list(mask))
    X = np.array(X)
    y = np.array(y)         
    return X, y   

def PreprocessData_test(data_path):
    imgs = glob.glob(os.path.join(data_path, 'Test','*.jpg'))
    print('number of images:{}'.format(len(imgs)))
    
    masks = glob.glob(os.path.join(data_path ,'Labels','*.png'))
    print('number of labels:{}'.format(len(masks)))
    
    X = []
    y = []
    for i in range(len(imgs)):
        img = cv2.imread(imgs[i])
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img = cv2.resize(img,(256,256))
        img = img/255
        X.append(img)
        
        mask = cv2.imread(masks[i])
        mask = cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
        mask = cv2.resize(mask,(256,256))
        mask = mask/255
        y.append(list(mask))
    X = np.array(X)
    y = np.array(y)         
    return X, y   


    
