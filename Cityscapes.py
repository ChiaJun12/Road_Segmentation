import sys
import os 
os.chdir(os.path.join('D:\Study Material\ML & DL\FYP (Road Segmentation)\Coding')) 

import matplotlib.pyplot as plt
import numpy as np
from glob import glob 
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from PIL import Image
from patchify import patchify
import cv2 as cv
import enum
from tensorflow.keras.utils import to_categorical 
   
from keras import backend as K
from keras.models import load_model
from keras_tuner import RandomSearch

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from tensorflow.keras.optimizers import Adamax, Adam

from Models import unet 

EPOCHS = 100
patch_size = 128
 
# define target classes
class MaskColorMap(enum.Enum):
    unlabeled =             (0, 0, 0),
    dynamic =               (111, 74, 0),
    ground =                (81, 0, 81),
    road =                  (128, 64, 128),
    sidewalk =              (244, 35, 232),
    parking =               (250, 170, 160),
    # rail_track =            (230, 150, 140),
    building =              (70, 70, 70),
    wall =                  (102, 102, 156),
    fence =                 (190, 153, 153),
    # guard_rail =            (180, 165, 180),
    # bridge =                (150, 100, 100),
    # tunnel =                (150, 120, 90),
    pole =                  (153, 153, 153), 
    traffic_light =         (250, 170, 30),
    traffic_sign =          (220, 220, 0),
    vegetation =            (107, 142, 35),
    terrain =               (152, 251, 152),
    sky =                   (70, 130, 180),
    person =                (220, 20, 60),
    rider =                 (255, 0, 0),
    car =                   (0, 0, 142),
    truck =                 (0, 0, 70),
    bus =                   (0, 60, 100),
    caravan =               (0, 0, 90),
    # trailer =               (0, 0,110),
    # train =                 (0, 80, 100),
    motorcycle =            (0, 0, 230),
    bicycle =               (119, 11, 32),
    # license_plate =         (0, 0, 142),

 
# main model 
class Road_Segmentation: 
    image_dataset = []
    mask_dataset = []
    
    # get image dataset and mask dataset
    def __init__(self, image_dataset, mask_dataset):
        self.image_dataset = np.array(image_dataset)
        self.mask_dataset = np.array(mask_dataset)
     
    
    # classify different objects 
    def one_hot_encode_masks(self, masks, num_classes):
        integer_encoded_labels = []
        for mask in tqdm(masks):
            _height, _width, _channels = mask.shape
            encoded_image = np.zeros((_height, _width, 1)).astype(int)
            
            for i, cls in enumerate(MaskColorMap):
                encoded_image[np.all(mask == cls.value, axis=-1)] = i
                
            integer_encoded_labels.append(encoded_image)
            
        return to_categorical(y = integer_encoded_labels, num_classes = num_classes)

  
    # convert image path to image data
    def transform_data(self, image_set):
        dataset = []

        for img_path in image_set:
            img = cv.imread(img_path)
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            dataset.append(cv.resize(img, None, fx=0.25, fy=0.25))
            
        return dataset


    # read image
    def patch_image(self, img, patch_size = patch_size):  
        instances = []
        
        for i, image in tqdm(enumerate(img)):  
            image = Image.fromarray(image)
             
            patch_img = patchify(np.array(image), (patch_size, patch_size, 3), step=patch_size)  
            
            for i in range(patch_img.shape[0]):
                for j in range(patch_img.shape[1]):
                    single_patch_img = patch_img[i, j]
                    instances.append(np.squeeze(single_patch_img))
                
        return np.array(instances)
     
      

    # generate x_train, x_test, y_train, y_test
    def split_data(self):
        # number of classes in segmentation dataset
        n_classes = 23
        
        # create (X, Y) training data
        x, y = self.image_dataset, self.mask_dataset
        x = self.transform_data(x)
        y = self.transform_data(y)
        X = self.patch_image(x) 
        Y = self.patch_image(y) 
        
        
        # extract X_train shape parameters
        m, img_height, img_width, img_channels = X.shape 

        # convert RGB values to integer encoded labels for categorial_crossentropy
        Y = self.one_hot_encode_masks(Y, num_classes=n_classes) 
        
        # split dataset into training and test groups
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=1)
        
        return X_train, X_test, Y_train, Y_test



    # jaccard similarity: the size of the intersection divided by the size of the union of two sets
    def jaccard_index(self, y_true, y_pred):
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        return (intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + 1.0)

      
    # convert image from normalized format to rgb format
    def rgb_encode_mask(self, mask):
        # initialize rgb image with equal spatial resolution
        rgb_encode_image = np.zeros((mask.shape[0], mask.shape[1], 3))

        # iterate over MaskColorMap
        for j, color in enumerate(MaskColorMap):
            # convert single integer channel to RGB channels
            rgb_encode_image[(mask == j)] = np.array(color.value) / 255.
        return rgb_encode_image

      
    # optional: display one patch image
    def display_images(self, instances, rows=2, titles=None):
        """
        :param instances:  list of images
        :param rows: number of rows in subplot
        :param titles: subplot titles
        :return:
        """
        n = len(instances)
        cols = n // rows if (n / rows) % rows == 0 else (n // rows) + 1

        # iterate through images and display subplots
        for j, image in enumerate(instances):
            plt.subplot(rows, cols, j + 1)
            plt.title('') if titles is None else plt.title(titles[j])
            plt.axis("off")
            image = cv.resize(image, None, fx=1.5, fy=1.5)
            plt.imshow(image)

        # show the figure
        plt.show()


    def image_concatenation(self, ori_image, predicted_img):
        concatenated_image = []
        
        shape = np.array(ori_image).shape
        row = shape[1] // patch_size
        col = shape[2] // patch_size
        
        for r in range(row):
            temp = []
            temp = np.concatenate((predicted_img[r * col: col + (r * col)]), axis=1)
            if r == 0:
                concatenated_image = np.concatenate((np.expand_dims(temp, axis=0)), axis=0)
            else:
                concatenated_image = np.concatenate((concatenated_image, temp), axis=0)
    
        return concatenated_image
        

    # concatenate predict images and display as full image
    def get_full_image_prediction(self, model):
        # initialise lists
        rand = np.random.randint(0, len(self.image_dataset))
        
        rand_image = self.image_dataset[rand]
        mask_image = self.mask_dataset[rand] 
    
        ori_image = self.transform_data([rand_image])
        image = self.patch_image(ori_image)
     
        predicted_img = []
        for img in image:
            predicted_img.append(model.predict(np.expand_dims(img, axis = 0)))
    
        for i, img in enumerate(predicted_img):
            predicted_img[i] = np.squeeze(img)
            predicted_img[i] = np.argmax(predicted_img[i], axis=-1)
            predicted_img[i] = self.rgb_encode_mask(predicted_img[i])
         
        mask_image = self.transform_data([mask_image])        
         
        concatenated_image = self.image_concatenation(ori_image, predicted_img)
                  
        return np.squeeze(ori_image), np.squeeze(mask_image), concatenated_image
         
    
    # train cnn model
    def train_model(self):  
        model = unet()
        
        # add callbacks, compile model and fit training data
        # save best model with maximum validation accuracy
        checkpoint = ModelCheckpoint('best_model_unet.h5', monitor="val_accuracy", verbose=1, save_best_only=True, mode="max")
        
        # create log 
        csv_logger = CSVLogger('training.log', separator=",", append=False)
        
        # create list of callbacks
        callbacks_list = [checkpoint, csv_logger] 

        # get data
        X_train, X_test, Y_train, Y_test = self.split_data()
 
        # compile model
        model.compile(optimizer=Adamax(learning_rate=0.00225), loss="categorical_crossentropy", metrics=["accuracy", self.jaccard_index])
     
        model.fit(X_train, Y_train, epochs=EPOCHS, batch_size=8, validation_data=(X_test, Y_test), callbacks=callbacks_list, verbose=1)
      
        return model 
    
    
    # get video frame
    def predict_video(self, video_name):
        model = load_model('best_model_unet.h5', custom_objects={'jaccard_index': self.jaccard_index})

        # load video data
        source = cv.VideoCapture(video_name)
        while True:
            ret, frame = source.read()

            frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            frame = cv.resize(frame, None, fx=0.25, fy=0.25)
            
            image = self.patch_image([frame])
            
            predicted_img = []
            for img in image:
                predicted_img.append(model.predict(np.expand_dims(img, axis = 0)))
        
            for i, img in enumerate(predicted_img):
                predicted_img[i] = np.squeeze(img)
                predicted_img[i] = np.argmax(predicted_img[i], axis=-1)
                predicted_img[i] = self.rgb_encode_mask(predicted_img[i]) 
                 
            concatenate_image = self.image_concatenation(np.expand_dims(frame, axis=0), predicted_img)

            concatenate_image = concatenate_image * 255
            concatenate_image = concatenate_image.astype(np.uint8)

            final = cv.addWeighted(frame, 1, concatenate_image, 0.5, 0)

            final = cv.resize(final, None, fx=3, fy=3)
            
            cv.imshow('prediction', final)

            # press 'Q' to stop
            if cv.waitKey(1) == ord('q'):
               break
            
        source.release()


    # further training
    def further_training(self): 
        # load model
        model = load_model('best_model_unet.h5', custom_objects = {'jaccard_index': self.jaccard_index})
         
        X_train, X_test, Y_train, Y_test = self.split_data()

        # save the best val_accuracy model
        checkpoint = ModelCheckpoint('best_model_unet.h5', monitor="val_accuracy", verbose=1, save_best_only=True, mode="max")
        
        # stop model if no improvement after 10 epochs
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode="min")
        
        # create log 
        csv_logger = CSVLogger('training.log', separator=",", append=False)
        
        # create list of callbacks
        callbacks_list = [checkpoint, csv_logger, early_stopping] 

        model.compile(optimizer=Adamax(learning_rate=0.002), loss="categorical_crossentropy", metrics=["accuracy", self.jaccard_index])
     
        model.fit(X_train, Y_train, epochs=EPOCHS, batch_size=8, validation_data=(X_test, Y_test), callbacks=callbacks_list, verbose=1)
  
        for i in range(5):
            image, mask_image, pred_image = self.get_full_image_prediction(model)
                
            self.display_images(
                [image, mask_image, pred_image],
                rows=1, titles=['original image', 'mask image', 'predicted image']
            )

         
# load data from local system
def get_training_data(dataset):
    # initialise lists
    image_dataset = []
    mask_dataset = []
    
    image = 'leftImg8bit'
    mask = 'gtFine'

    input_image_subdir = dataset  

    for dirs in [image, mask]: 
        for subdirs in os.listdir(os.path.join(os.getcwd(), dirs, 'train')):
            if subdirs in input_image_subdir: 
                if dirs == image:
                    image_dataset.extend(glob(os.path.join(os.getcwd(), dirs, 'train', subdirs, '*.png')))
                else:
                    mask_dataset.extend(glob(os.path.join(os.getcwd(), dirs, 'train', subdirs, '*color.png')))
    
    return image_dataset, mask_dataset

    

# main function
if __name__ == '__main__':
    # for first training: 'jena', 'erfurt', 'krefeld' 
 
    dataset = sys.argv[1:-1]
    training_option = sys.argv[-1]
  
    image_dataset, mask_dataset = get_training_data(dataset)
    semantic_model = Road_Segmentation(image_dataset, mask_dataset)

    if training_option == 'new':
        model = semantic_model.train_model()
        
        for i in range(5):
            image, mask_image, pred_image = semantic_model.get_full_image_prediction(model)
            
            semantic_model.display_images(
                [image, mask_image, pred_image],
                rows=1, titles=['original image', 'mask image', 'predicted image']
            )  
    elif training_option == 'continue':
        semantic_model.further_training()
    
    elif training_option == 'video':
        video_name = input('Enter the video file name: ')
        semantic_model.predict_video(video_name)  
    
