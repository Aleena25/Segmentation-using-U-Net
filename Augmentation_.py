import glob
import albumentations as A
import cv2
import os

def Augment_img(train_data_path):
    transform = A.Compose([
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
    ])
    
    try:
        train_out_dir = os.path.join(train_data_path ,'Augmented')
        os.mkdir(train_out_dir)
        os.mkdir(os.path.join(train_out_dir, 'Train'))
        os.mkdir(os.path.join(train_out_dir, 'Labels'))


    except Exception as e:
        print(e)
    train_data = glob.glob(os.path.join(train_data_path, 'Train_data', 'Train', '*.jpg'))
    train_labels = glob.glob(os.path.join(train_data_path,'Train_data', 'Labels', '*.png'))
       
    k = 0
    for i, m in zip(train_data, train_labels):
            image = cv2.imread(str(i))
            mask = cv2.imread(str(m))
            for j in range(10):
                transformed = transform(image=image, mask=mask)
                transformed_image = transformed['image']
                transformed_image = transformed['image']
                transformed_mask = transformed['mask']
                transformed_mask = transformed_mask
                tr_filename = train_out_dir+'\Train' + '\img_'+str(k)+'_'+str(j)+'.jpg'
                cv2.imwrite(tr_filename, transformed_image)
                lbl_filename = train_out_dir + '\Labels'+ '\img_'+str(k)+'_'+str(j)+'.png'
                cv2.imwrite(lbl_filename, transformed_mask)
            k += 1
    
    return train_out_dir
