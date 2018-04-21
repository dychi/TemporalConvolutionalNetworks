from keras.models import Sequential, Model
from keras.layers import Input, Dense, TimeDistributed, merge, Lambda
from keras.layers.core import *
from keras.layers.convolutional import *
from keras.layers.recurrent import *

import tensorflow as tf
from keras import backend as K

from keras.activations import relu
from functools import partial
clipped_relu = partial(relu, max_value=5)


def max_filter(x):
    # Max over the best filter score (like ICRA paper)
    max_values = K.max(x, 2, keepdims=True)
    max_flag = tf.greater_equal(x, max_values)
    out = x * tf.cast(max_flag, tf.float32)
    return out

def channel_normalization(x):
    # Normalize by the highest activation
    max_values = K.max(K.abs(x), 2, keepdims=True)+1e-5
    out = x / max_values
    return out

def WaveNet_activation(x):
    tanh_out = Activation('tanh')(x)
    sigm_out = Activation('sigmoid')(x)  
    return Merge(mode='mul')([tanh_out, sigm_out])




def ED_TCN(n_nodes, conv_len, n_classes, n_feat, max_len, 
            loss='categorical_crossentropy', causal=False, 
            optimizer="rmsprop", activation='norm_relu',
            return_param_str=False):
    n_layers = len(n_nodes)

    inputs = Input(shape=(max_len,n_feat)) # 期待される入力ベクトルが(max_len, n_feat)のバッチ。バッチサイズは含まれない(552,128)
    model = inputs

    # ---- Encoder ----
    for i in range(n_layers):
        # Pad beginning of sequence to prevent usage of future data
        if causal: model = ZeroPadding1D(padding=(conv_len//2, 0))(model) # 始めと終わりにそれぞれいくつのゼロを加えるか。(left_pad, right_pad) 
        model = Convolution1D(n_nodes[i], kernel_size=conv_len, border_mode='same')(model) # 出力(max_len,64), kernel_size=conv, 
        if causal: model = Cropping1D(padding=(0, conv_len//2))(model) # (10/2, 10//2)=(5.0, 5)

        model = SpatialDropout1D(0.3)(model)
        
        if activation=='norm_relu': 
            model = Activation('relu')(model)            
            model = Lambda(channel_normalization, name="encoder_norm_{}".format(i))(model)
        elif activation=='wavenet': 
            model = WaveNet_activation(model) 
        else:
            model = Activation(activation)(model)            
        
        model = MaxPooling1D(pool_size=2)(model)

    # ---- Decoder ----
    for i in range(n_layers):
        model = UpSampling1D(2)(model) # 時間軸方向にそれぞれの時間ステップをsize=2回繰り返す
        if causal: model = ZeroPadding1D((conv_len//2,0))(model)
        model = Convolution1D(n_nodes[-i-1], conv_len, border_mode='same')(model)
        if causal: model = Cropping1D((0,conv_len//2))(model)

        model = SpatialDropout1D(0.3)(model)

        if activation=='norm_relu': 
            model = Activation('relu')(model)
            model = Lambda(channel_normalization, name="decoder_norm_{}".format(i))(model)
        elif activation=='wavenet': 
            model = WaveNet_activation(model) 
        else:
            model = Activation(activation)(model)

    # Output FC layer
    model = TimeDistributed(Dense(n_classes, activation="softmax"))(model) # units=出力次元数、 タイムスタンプのレイヤーそれぞれにDenseを適用するため
    
    model = Model(input=inputs, output=model)
    model.compile(loss=loss, optimizer=optimizer, sample_weight_mode="temporal", metrics=['accuracy']) # 訓練過程の設定, "temporal"=時間ごとのサンプルの重み付け（2次元の重み）を行う

    if return_param_str:
        param_str = "ED-TCN_C{}_L{}".format(conv_len, n_layers)
        if causal:
            param_str += "_causal"
    
        return model, param_str
    else:
        return model