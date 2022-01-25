import os
import glob
import shutil



def train_test_split(main_dir):
    img_files = sorted(glob.glob(os.path.join(main_dir,'boxes_','*.jpg')))
    print('Number of images: {}'.format(len(img_files)))
    label_files = sorted(glob.glob(os.path.join(main_dir,'boxes_mask','*.png')))
    print('Number of masks: {}'.format(len(label_files)))
    num = len(img_files)
    num_train = int((num*90)/100)
    num_test = num-num_train
    train_data_path = os.path.join(main_dir , 'Train_data')
    test_data_path = os.path.join(main_dir, 'Test_data')
    try:
        
        os.mkdir(train_data_path)
        os.mkdir(test_data_path)
        
      
        os.mkdir(os.path.join(train_data_path , 'Train' ))
        os.mkdir(os.path.join(test_data_path , 'Test' ))
        os.mkdir(os.path.join(train_data_path , 'Labels' ))
        os.mkdir(os.path.join(test_data_path ,'Labels' ))
    except Exception as e:
        print(e)
    
    train_files = img_files[0:num_train]
    train_labels = label_files[0:num_train]
    test_files = img_files[num_train:]
    test_labels = label_files[num_train:]
    
    for i in range(num_train):
        train_file = train_files[i]
        label_file = train_labels[i]
        img_copy_path = os.path.join(train_data_path , 'Train', train_file.split('\\')[4])
        shutil.copyfile(train_file, img_copy_path)
        label_copy_path = os.path.join(train_data_path ,'Labels',label_file.split('\\')[4])
        shutil.copyfile(label_file, label_copy_path)
    for i in range(num_test):
        test_file = test_files[i]
        label_file = test_labels[i]
        img_copy_path = os.path.join(test_data_path ,'Test', test_file.split('\\')[4])
        shutil.copyfile(test_file, img_copy_path)
        label_copy_path = os.path.join(test_data_path ,'Labels',label_file.split('\\')[4])
        shutil.copyfile(label_file, label_copy_path)
    
    return train_data_path, test_data_path   
 
