import os 
os.chdir(os.path.join(r'C:\Users\Leong Teng Man\Desktop\Road_Segmentation'))
  
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
import multiprocessing

from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D, Dropout, Conv2DTranspose, MaxPooling2D
from tensorflow.keras.layers import Rescaling, concatenate
from keras import backend as K

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import GridSearchCV
from scikeras.wrappers import KerasClassifier
import tensorflow as tf
 
EPOCHS = 100
patch_size = 128


# define target classes
class MaskColorMap(enum.Enum):
    # these correspond to RGB values in train and target picture
    unlabeled =             (0, 0, 0),
    ego_vehicle =           (0, 0, 0),
    rectification_border =  (0, 0, 0),
    out_of_roi =            (0, 0, 0),
    static =                (0, 0, 0),
    dynamic =               (111, 74, 0),
    ground =                (81, 0, 81),
    road =                  (128, 64, 128),
    sidewalk =              (244, 35, 232),
    parking =               (250, 170, 160),
    rail_track =            (230, 150, 140),
    building =              (70, 70, 70),
    wall =                  (102, 102, 156),
    fence =                 (190, 153, 153),
    guard_rail =            (180, 165, 180),
    bridge =                (150, 100, 100),
    tunnel =                (150, 120, 90),
    pole =                  (153, 153, 153),
    polegroup =             (153, 153, 153),
    traffic_light =         (250, 170, 30),
    traffic_sign =          (220, 220, 0),
    vegetation =            (107, 142, 35),
    terrain =               (152, 251, 152),
    sky =                   (70, 130, 180),
    person =                (220, 20, 60),
    rider =                 (255, 0, 0),
    car =                   (0, 0,142),
    truck =                 (0, 0, 70),
    bus =                   (0, 60,100),
    caravan =               (0, 0, 90),
    trailer =               (0, 0,110),
    train =                 (0, 80, 100),
    motorcycle =            (0, 0, 230),
    bicycle =               (119, 11, 32),
    license_plate =         (0, 0, 142),



class Road_Segmentation: 
    def one_hot_encode_masks(self, masks, num_classes):
        integer_encoded_labels = []
        for mask in tqdm(masks):
            _height, _width, _channels = mask.shape
            encoded_image = np.zeros((_height, _width, 1)).astype(int)
            
            for i, cls in enumerate(MaskColorMap):
                encoded_image[np.all(mask == cls.value, axis=-1)] = i
                
            integer_encoded_labels.append(encoded_image)
            
        return to_categorical(y = integer_encoded_labels, num_classes = num_classes)


    def load_images(self, dir, pattern, size_ratio):
        '''
        :param dir: current directory
        :param pattern: glob pattern
        :return: return the files are in dir/pattern
        '''
        dataset = []
        dirs = list(glob(os.path.join(dir, pattern)))
        dirs = dirs[:int(len(dirs)*size_ratio)]
        for path in tqdm(dirs):
            img = cv.imread(path)
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            dataset.append(cv.resize(img, None, fx=0.75, fy=0.75))
        return dataset

    def get_training_data(self):
        size_ratio = 0.15
        # initialise lists
        image_dataset = []
        mask_dataset = []

        image = 'leftImg8bit'   # original image directory
        mask = 'gtFine'         # labelled image directory

        input_image_subdir = ['aachen']
        ignore_fn = lambda dir: dir[0] != '.' and os.path.isdir(dir)
        for dirs in filter(ignore_fn, os.listdir(os.getcwd())):
            for subdirs in filter(lambda dir: dir in input_image_subdir,
                                  os.listdir(os.path.join(os.getcwd(), dirs, 'train'))):
                cur_dir = os.path.join(os.getcwd(), dirs, 'train', subdirs)
                if dirs == image:
                    image_dataset.extend(self.load_images(cur_dir, '*.png',size_ratio))
                if dirs == mask:
                    mask_dataset.extend(self.load_images(cur_dir, '*color.png', size_ratio))
        return np.array(image_dataset), np.array(mask_dataset)
        return np.array(image_dataset)[:int(len(image_dataset) * 0.15)], np.array(mask_dataset)[:int(len(image_dataset) * 0.15)]



    # read image
    def parse_image(self, img, patch_size = patch_size):  
        instances = []
        
        for i, image in tqdm(enumerate(img)):  
            image = Image.fromarray(image)
             
            patch_img = patchify(np.array(image), (patch_size, patch_size, 3), step=patch_size)  
            
            for i in range(patch_img.shape[0]):
                for j in range(patch_img.shape[1]):
                    single_patch_img = patch_img[i, j]
                    instances.append(np.squeeze(single_patch_img))
                
        return np.array(instances)
     
      

    def split_data(self):
        # number of classes in segmentation dataset
        n_classes = 35
        
        # create (X, Y) training data
        x, y = self.get_training_data()
        X = self.parse_image(x) 
        Y = self.parse_image(y) 
        
        # extract X_train shape parameters
        m, img_height, img_width, img_channels = X.shape
        print('number of patched image training data:', m)
          
        # convert RGB values to integer encoded labels for categorial_crossentropy
        Y = self.one_hot_encode_masks(Y, num_classes=n_classes) 
        
        # split dataset into training and test groups
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=42)
        
        return X_train, X_test, Y_train, Y_test


    # unet is a CNN type
    def build_unet(self):
        # input layer shape is equal to patch image size
        inputs = Input(shape=(patch_size, patch_size, 3))

        # rescale images from (0, 255) to (0, 1)
        rescale = Rescaling(scale=1. / 255, input_shape=(patch_size, patch_size, 3))(inputs)
        previous_block_activation = rescale  # Set aside residual

        contraction = {}
        no_filter = [64, 128]
        # # Contraction path: Blocks 1 through 5 are identical apart from the feature depth
        for f in no_filter:
            x = Conv2D(f, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(previous_block_activation)
            x = Dropout(0.2)(x)
            x = Conv2D(f, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(x)
            contraction[f'conv{f}'] = x
            x = MaxPooling2D((2, 2))(x)
            previous_block_activation = x
            
        c5 = Conv2D(patch_size, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(previous_block_activation)          
        c5 = Dropout(0.1)(c5)
        c5 = Conv2D(patch_size, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)
        previous_block_activation = c5 
     
        # Expansive path: Second half of the network: upsampling inputs
        for f in reversed(no_filter):
            x = Conv2DTranspose(f, (2, 2), strides=(2, 2), padding='same')(previous_block_activation)
            x = concatenate([x, contraction[f'conv{f}']])
            x = Conv2D(f, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(x)
            x = Dropout(0.2)(x)
            x = Conv2D(f, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(x)
            previous_block_activation = x

        outputs = Conv2D(filters=35, kernel_size=(1, 1), activation="softmax")(previous_block_activation)

        return Model(inputs=inputs, outputs=outputs)
     
     
     
    # jaccard similarity: the size of the intersection divided by the size of the union of two sets
    def jaccard_index(self, y_true, y_pred):
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        return (intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + 1.0)

      

    def rgb_encode_mask(self, mask):
        # initialize rgb image with equal spatial resolution
        rgb_encode_image = np.zeros((mask.shape[0], mask.shape[1], 3))

        # iterate over MaskColorMap
        for j, color in enumerate(MaskColorMap):
            # convert single integer channel to RGB channels
            rgb_encode_image[(mask == j)] = np.array(color.value) / 255.
        return rgb_encode_image

      

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
            plt.imshow(image)

        # show the figure
        plt.show()

         

    def train_model(self):  
        model = self.build_unet()
        
        # add callbacks, compile model and fit training data
        # save best model with maximum validation accuracy
        checkpoint = ModelCheckpoint('best_model_unet.h5', monitor="val_accuracy", verbose=1, save_best_only=True, mode="max")
        
        # stop model training early if validation loss doesn't continue to decrease over 2 iterations
        early_stopping = EarlyStopping(monitor='val_loss', patience=2, verbose=1, mode="min")
        
        # log training console output to csv
        csv_logger = CSVLogger('training.log', separator=",", append=False)
        
        # create list of callbacks
        callbacks_list = [checkpoint, csv_logger] 

        # get data
        X_train, X_test, Y_train, Y_test = self.split_data() 
  
        # compile model
        model.compile(optimizer=Adam(learning_rate=0.0005), loss="categorical_crossentropy", metrics=["accuracy", self.jaccard_index])
  
        model.fit(X_train, Y_train, epochs=EPOCHS, batch_size=16, validation_data=(X_test, Y_test), callbacks=callbacks_list, verbose=1)
     
        # tf.saved_model.save(model, 'D:\\Study Material\\ML & DL\\FYP (Road Segmentation)\\Coding\\latest_model\\aug-28(1)')
        
        for _ in range(20):
            # choose random number from 0 to test set size
            test_img_number = np.random.randint(0, len(X_test))
        
            # extract test input image
            test_img = X_test[test_img_number]
        
            # ground truth test label converted from one-hot to integer encoding
            ground_truth = np.argmax(Y_test[test_img_number], axis=-1)
        
            # expand first dimension as U-Net requires (m, h, w, nc) input shape
            test_img_input = np.expand_dims(test_img, 0)
        
            # make prediction with model and remove extra dimension
            prediction = np.squeeze(model.predict(test_img_input))
        
            # convert softmax probabilities to integer values
            predicted_img = np.argmax(prediction, axis=-1)
         
            # convert integer encoding to rgb values
            rgb_image = self.rgb_encode_mask(predicted_img)
            rgb_ground_truth = self.rgb_encode_mask(ground_truth)
        
            # visualize model predictions
            self.display_images(
                [test_img, rgb_ground_truth, rgb_image],
                rows=1, titles=['Aerial', 'Ground Truth', 'Prediction']
            )
        
 
if __name__ == '__main__':
    model = Road_Segmentation()
    model.train_model()
