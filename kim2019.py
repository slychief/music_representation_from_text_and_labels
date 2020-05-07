
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.constraints import MinMaxNorm, unit_norm
from tensorflow.keras.initializers import he_normal, he_uniform, glorot_uniform
from tensorflow.keras.layers import (ELU, AlphaDropout, AveragePooling2D,
                                     BatchNormalization, Concatenate, Conv2D,
                                     Dense, Dropout, Flatten, GaussianNoise,
                                     GlobalAveragePooling2D, Input, Lambda,
                                     LeakyReLU, MaxPooling2D, ReLU, Layer,
                                     SpatialDropout2D)
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import SGD, Adagrad, Adam, Nadam, RMSprop
from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras.utils import normalize
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops, math_ops

import numpy as np


#import keras


from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.framework import tensor_shape
import tensorflow as tf

class RandomCropping2D(Layer):

    def __init__(self, cropping=((0, 0), (0, 0)), data_format=None, **kwargs):
        super(RandomCropping2D, self).__init__(**kwargs)
        self.data_format = conv_utils.normalize_data_format(data_format)
        if isinstance(cropping, int):
          self.cropping_orig = ((cropping, cropping), (cropping, cropping))
        elif hasattr(cropping, '__len__'):

          if len(cropping) != 2:
            raise ValueError('`cropping` should have two elements. '
                             'Found: ' + str(cropping))
          height_cropping = conv_utils.normalize_tuple(cropping[0], 2,
                                                       '1st entry of cropping')
          width_cropping = conv_utils.normalize_tuple(cropping[1], 2,
                                                      '2nd entry of cropping')

          self.cropping_orig = (height_cropping, width_cropping)
        else:
          raise ValueError('`cropping` should be either an int, '
                           'a tuple of 2 ints '
                           '(symmetric_height_crop, symmetric_width_crop), '
                           'or a tuple of 2 tuples of 2 ints '
                           '((top_crop, bottom_crop), (left_crop, right_crop)). '
                           'Found: ' + str(cropping))
        self.input_spec = InputSpec(ndim=4)

    def compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape).as_list()
        # pylint: disable=invalid-unary-operand-type
        if self.data_format == 'channels_first':
          return tensor_shape.TensorShape([
              input_shape[0], input_shape[1],
              input_shape[2] - self.cropping[0][0] - self.cropping[0][1]
              if input_shape[2] else None,
              input_shape[3] - self.cropping[1][0] - self.cropping[1][1]
              if input_shape[3] else None
          ])
        else:
          return tensor_shape.TensorShape([
              input_shape[0],
              self.cropping[0][1] - self.cropping[0][0]
              if input_shape[1] else None,
              self.cropping[1][1] - self.cropping[1][0]
              if input_shape[2] else None, input_shape[3]
          ])
        # pylint: enable=invalid-unary-operand-type

    def _get_training_value(self, training=None):

        if training is None:
            training = K.learning_phase()

        if isinstance(training, int):
            training = bool(training)

        return training

    @tf.function
    def call(self, inputs, training = None, **kwargs):

        #training       = self._get_training_value(training)
        #training_value = tf.cast(training, tf.bool)

        self.cropping = self.cropping_orig
        
        if training:

            if self.cropping_orig[0][0] > 0:
                start         = np.random.randint(0, inputs.shape[1] - self.cropping_orig[0][0] + 1)
                #start         = tf.cast(start, tf.int32)
                t             = (start, start+self.cropping_orig[0][0])
                self.cropping = (t, self.cropping[1])
            else:
                self.cropping = ((0,inputs.shape[1]), self.cropping[1])

            if self.cropping_orig[1][0] > 0:
                start         = np.random.randint(0, inputs.shape[2] - self.cropping_orig[1][0] + 1)
                #start         = tf.cast(start, tf.int32)
                t             = (start, start+self.cropping_orig[1][0])
                self.cropping = (self.cropping[0], t)
            else:
                self.cropping = (self.cropping[0], (0,inputs.shape[2]))
                
        else:
            
            if self.cropping_orig[0][0] > 0:
                start         = 0#tf.cast(0, tf.int32)
                t             = (start, start+self.cropping_orig[0][0])
                self.cropping = (t, self.cropping[1])
            else:
                self.cropping = ((0,inputs.shape[1]), self.cropping[1])

            if self.cropping_orig[1][0] > 0:
                start         = 0#tf.cast(0, tf.int32)
                t             = (start, start+self.cropping_orig[1][0])
                self.cropping = (self.cropping[0], t)
            else:
                self.cropping = (self.cropping[0], (0,inputs.shape[2]))
        
        # pylint: disable=invalid-unary-operand-type
        if self.data_format == 'channels_first':
          if (self.cropping[0][1] == 0) and (self.cropping[1][1] == 0):
            return inputs[:, :, self.cropping[0][0]:, self.cropping[1][0]:]
          elif self.cropping[0][1] == 0:
            return inputs[:, :, self.cropping[0][0]:, self.cropping[1][0]:
                          -self.cropping[1][1]]
          elif self.cropping[1][1] == 0:
            return inputs[:, :, self.cropping[0][0]:-self.cropping[0][1],
                          self.cropping[1][0]:]
          return inputs[:, :, self.cropping[0][0]:-self.cropping[0][1],
                        self.cropping[1][0]:-self.cropping[1][1]]
        else:
          if (self.cropping[0][1] == 0) and (self.cropping[1][1] == 0):
            return inputs[:, self.cropping[0][0]:, self.cropping[1][0]:, :]
          else :
            return inputs[:, self.cropping[0][0]:self.cropping[0][1], self.cropping[1][0]:self.cropping[1][1], :]
          
    def get_config(self):
        config = {'cropping': self.cropping, 'data_format': self.data_format}
        base_config = super(Cropping2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))




# ## Model Definition

def get_model(embeddings_dimensions):

    # --- input layers
    kim2019_inp = Input((128,216,2), name="input_ref") # reference track
       
    cnn  = Conv2D(16, (5,5), padding="same", kernel_initializer=glorot_uniform(1), activation="relu", strides=(1,2), name="conv1")(kim2019_inp)
    cnn = MaxPooling2D((2,2), name="max-pool1")(cnn)
    
    cnn  = Conv2D(32, (3,3), padding="same", kernel_initializer=glorot_uniform(2), activation="relu", name="conv2")(cnn)
    cnn = MaxPooling2D((2,2), name="max-pool2")(cnn)
    
    cnn  = Conv2D(64, (3,3), padding="same", kernel_initializer=glorot_uniform(3), activation="relu", name="conv3")(cnn)
    cnn = MaxPooling2D((2,2), name="max-pool3")(cnn)
    
    cnn  = Conv2D(64, (3,3), padding="same", kernel_initializer=glorot_uniform(4), activation="relu", name="conv4")(cnn)
    cnn = MaxPooling2D((2,2), name="max-pool4")(cnn)
    
    cnn  = Conv2D(128, (3,3), padding="same", kernel_initializer=glorot_uniform(5), activation="relu", name="conv5")(cnn)
    cnn = MaxPooling2D((2,2), name="max-pool5")(cnn)
    
    cnn  = Conv2D(128, (3,3), padding="same", kernel_initializer=glorot_uniform(6), activation="relu", name="conv61")(cnn)
    cnn  = Conv2D(256, (1,1), padding="same", kernel_initializer=glorot_uniform(7), activation="relu", name="conv62")(cnn)
    
    gap = GlobalAveragePooling2D(name="gap")(cnn)
    
    out = Dense(256, activation="relu", kernel_initializer=glorot_uniform(7), name="fc-feature")(gap)
    
    # --- build model
    model_kim2019   = Model(inputs=[kim2019_inp], outputs=out)
    
    
    input_eval_model = Input((128,880,2))
    
    cropped_input    = RandomCropping2D(cropping=(0,216))(input_eval_model)
    
    mk2019_out       = model_kim2019(cropped_input)
    
    mk2019_out       = Dropout(0.5)(mk2019_out)
    
    fc_output        = Dense(embeddings_dimensions, activation="elu", kernel_initializer=glorot_uniform(8), name="fc-output")(mk2019_out)
    
    model_eval      = Model(inputs=[input_eval_model], outputs=fc_output)
    
    return model_eval
    

if __name__ == "__main__":

    model = get_model(256)

    model.summary()