from base.base_model import BaseModel
from keras.models import Sequential, Model
from keras.layers import Input, Dense, TimeDistributed, merge, Lambda
from keras.applications.vgg16 import VGG16

# from keras.layers.core import *
# from keras.layers.convolutional import *
vgg16 = VGG16(include_top=False, weights='imagenet', input_tensor=input_tensor)

class ED_TCN(BaseModel):
    def __init__(self, config):
        super(ED_TCN, self).__init__(config)
        self.build_model()
        
        
    def build_model(self):
        self.model = Sequential()
        self.model.add(Flatten(imput_shape=vgg16.output_shape[1:]))
        self.model.add()
        
        
    
        self.model.compile(loss=loss, optimizer=oprimizer, sample_weight_model="temporal", metrics=['accuracy'])
        
        
