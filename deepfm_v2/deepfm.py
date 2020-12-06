import numpy as np
import pandas as pd
import json
import functools
import operator
import re
from pathlib import Path

from tensorflow.keras.layers import (Dense, Embedding,
                                     Permute, Lambda,
                                     RepeatVector, TimeDistributed,
                                     Concatenate, Dot, 
                                     Reshape,Add, Dropout,
                                     Flatten, Layer,
                                     Activation, Input)
from tensorflow.keras.models import Model, load_model
import tensorflow.keras.backend as K
from tensorflow.keras.utils import plot_model
from tensorflow.keras.metrics import AUC
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras.preprocessing.sequence import pad_sequences

from utils.prep import encode_categorical_cols, encode_multilevel_categorical_cols, discretize_continous_cols


class EmbedContinousCol(Layer):
    """Embed Continous (2-D) Tensor into 3-D
    
    Args:
        @output_dim: Int
    Return:
        3-D Tensor (None, Lnum, output_dim)
        
    Example:
    
    Lnum = 5
    output_dim = 8
    input_num = Input((Lnum, ))
    EmbedContinousCol(output_dim)(input_num)
    """
    
    def __init__(self, output_dim, *args, **kwargs):
        super(EmbedContinousCol, self).__init__(*args, **kwargs)
        self.output_dim = output_dim

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[1], self.output_dim),
            initializer="random_normal",
            trainable=True,
            name = 'NumEmbedWeight'
        )

    def call(self, inputs):
        out = K.expand_dims(inputs, 2)
        out = out * self.w
        return out

    def get_config(self):

        config = super(EmbedContinousCol, self).get_config()
        config.update({'output_dim': self.output_dim})
        return config


def avg_pooling(x):
    return K.mean(x, axis = 1, keepdims=True)
    
def fm1(x):
    return K.sum(x, axis = 1)

def fm2(x):
    sq_sum = K.square(K.sum(x, axis = 1))
    sum_sq = K.sum(K.square(x), axis = 1)
    out = K.sum((sq_sum - sum_sq), axis = 1, keepdims=True)
    return 0.5 * out

   
class DeepFM():

    def __init__(self, fnames_mhe = [], fnames_cat = [], fnames_num = [], 
                 encoder_mhe = None, encoder_cat = None, bins_num = None, 
                 num_bins = 64, data_prep_method = ''):

        self.config = {
            'fnames_mhe': fnames_mhe,
            'fnames_cat': fnames_cat,
            'fnames_num': fnames_num,
            'encoder_mhe': encoder_mhe,
            'encoder_cat': encoder_cat,
            'bins_num': bins_num,
            'num_bins': num_bins,
            'data_prep_method': data_prep_method
        }

    def prepare_data(self,
                     data,
                     label=None,
                     in_train=None):

        fnames_mhe = self.config['fnames_mhe']
        fnames_cat = self.config['fnames_cat']
        fnames_num = self.config['fnames_num']
        encoder_mhe = self.config['encoder_mhe']
        encoder_cat = self.config['encoder_cat']
        bins_num = self.config['bins_num']
        data_prep_method = self.config['data_prep_method']

        data_encoded = data.copy()

        if data_prep_method == 'num2cat':
            data_encoded, bins_num = discretize_continous_cols(
                data_encoded, fnames_num, bins_num)
            fnames_cat = fnames_cat + fnames_num
            fnames_num = []

        data_encoded, encoder_cat = encode_categorical_cols(
            data_encoded, fnames_cat, encoder_cat, combined=True)
        data_encoded, encoder_mhe = encode_multilevel_categorical_cols(
            data_encoded, fnames_mhe, encoder_mhe)

        # update config
        if self.config['encoder_mhe'] is None:
            self.config['encoder_mhe'] = encoder_mhe
        if self.config['encoder_cat'] is None:
            self.config['encoder_cat'] = encoder_cat
        if self.config['bins_num'] is None:
            self.config['bins_num'] = bins_num

        # update fnames
        self.updated_fnames = {
            'fnames_mhe': fnames_mhe,
            'fnames_cat': fnames_cat,
            'fnames_num': fnames_num
        }
        
        x = []

        if len(fnames_mhe) > 0:
            for fname in fnames_mhe:
                x += [pad_sequences(data_encoded[fname])]
        if len(fnames_cat) > 0:
            x += [data_encoded[fnames_cat].values]
        if len(fnames_num) > 0:
            x += [data_encoded[fnames_num].values]

        if label is None:
            # test set for inference mode
            return x

        else:
            # training mode
            if in_train is None:
                train = (x, label)
                return train

            else:
                in_train = np.array(in_train)
                x_train = [i[in_train, :] for i in x]
                x_valid = [i[~in_train, :] for i in x]
                y_train = label[in_train]
                y_valid = label[~in_train]
                train = (x_train, y_train)
                valid = (x_valid, y_valid)

        return train, valid

    def _config2modelcols(self):

        config = self.config
        fnames = self.updated_fnames
        encoder_cat = config['encoder_cat']
        ModelCols = {
            'MHECols': {},
            'CatCols': {},
            'NumCols': {}
        }

        if len(fnames['fnames_mhe']) > 0:
            ModelCols['MHECols'] = {
                k: {
                    'vocab_size': (max(v.values()) + 1)
                }
                for k, v in config['encoder_mhe'].items()
            }

        if len(fnames['fnames_cat']) > 0:
            ModelCols['CatCols'] = {
                'cat': {
                    'maxlen': len(fnames['fnames_cat']),
                    'vocab_size': max(encoder_cat.values()) + 1
                }
            }

        if len(fnames['fnames_num']) > 0:
            ModelCols['NumCols'] = {
                'num': {
                    'maxlen': len(fnames['fnames_num'])
                }
            }

        return ModelCols



    def _define_fm_inputs(self, ModelCols):

        MHECols = ModelCols['MHECols']
        CatCols = ModelCols['CatCols']
        NumCols = ModelCols['NumCols']

        inputs = []

        if len(MHECols) > 0:
            for k, v in MHECols.items():
                input_mhe_i = Input((None, ), name='input_' + k)
                inputs += [input_mhe_i]

        if len(CatCols) > 0:
            maxlen = CatCols['cat']['maxlen']
            input_cat = Input((maxlen, ), name='input_cat')
            inputs += [input_cat]

        if len(NumCols) > 0:
            maxlen = NumCols['num']['maxlen']
            input_num = Input((maxlen, ), name='input_num')
            inputs += [input_num]

        return inputs
            
    def _embed_fm_cols(self, inputs, ModelCols, 
        embed_dim, embed_reg = 0.0):
        
        MHECols = ModelCols['MHECols']
        CatCols = ModelCols['CatCols']
        NumCols = ModelCols['NumCols']
        
        embeds = []
        name = 'fm1' if embed_dim == 1 else 'fm2'
        embed_reg = l2(embed_reg) if embed_reg > 0 else None

        if len(MHECols) > 0:
                    
            for i, (k,v) in enumerate(MHECols.items()):
                vocab_size = v['vocab_size']
                input_mhe_i = inputs[i]
                embed_mhe_i = Embedding(vocab_size, embed_dim, 
                                        mask_zero = True,
                                        embeddings_regularizer = embed_reg, 
                                        name = f'{name}_embed_{k}')(input_mhe_i)
                embed_mhe_i = Lambda(avg_pooling, name = f'{name}_avgpool_{k}')(embed_mhe_i)
                embeds += [embed_mhe_i]

        if len(CatCols) > 0:
            
            i = len(MHECols)
            vocab_size = CatCols['cat']['vocab_size']
            input_cat = inputs[i]
            embed_cat = Embedding(vocab_size, embed_dim, name = f'{name}_embed_cat')(input_cat)
            embeds += [embed_cat]

        if len(NumCols) > 0:
            
            i = len(MHECols) + len(CatCols)
            input_num = inputs[i]
            embed_num = EmbedContinousCol(embed_dim, name = f'{name}_embed_num')(input_num)
            embeds += [embed_num]
        
        if len(embeds) > 1:
            embeds = Concatenate(axis = 1, name = f'{name}_concat')(embeds)
        else:
            embeds = embeds[0]
        
        return embeds

    def build_model(self, embed_dim = 32, 

        deep_hidden_dims = [32], fm_reg = 0.0, 
        deep_dr = 0.0, is_deep = True, is_fm = True):

        ModelCols = self._config2modelcols()

        # define inputs
        inputs = self._define_fm_inputs(ModelCols)
        
        # first order fm 
        embeds_1d = self._embed_fm_cols(inputs, ModelCols, 1)
        y_fm1 = Lambda(fm1, name = 'fm1')(embeds_1d)
        
        # second order fm
        embeds_2d = self._embed_fm_cols(inputs, ModelCols, embed_dim, embed_reg=fm_reg)
        y_fm2 = Lambda(fm2, name = 'fm2')(embeds_2d)
        
        # deep
        y_deep = Flatten(name = 'embed_2d_flatten')(embeds_2d)
        for i, h in enumerate(deep_hidden_dims):
            y_deep = Dense(h, 'relu', name = f'deep_fc{i}')(y_deep)
            if deep_dr > 0:
                y_deep = Dropout(deep_dr, name = f'deep_dropout{i}')(y_deep)
                
        # selective deep/fm part
        y = []
        if is_fm:
            y += [y_fm1, y_fm2]
        if is_deep:
            y += [y_deep]
        if len(y) > 1:
            y = Concatenate(axis = 1, name = 'final_concat')(y)
        else:
            y = y[0]
            
        # final sigmoid output    
        y = Dense(1, 'sigmoid', name = 'final_output')(y)

        self.model = Model(inputs, y)
        self.model.compile('adam', 'binary_crossentropy', [AUC(name = 'auc')])

        return None

    def save_model(self, saved_model_dir, model_id):

        saved_model_dir = Path(saved_model_dir)
        config_dir = saved_model_dir.joinpath(f'config_{model_id}.json')
        model_dir = saved_model_dir.joinpath(f'model_{model_id}.h5')

        with open(config_dir, 'w') as f:
            json.dump(self.config, f)

        print(f'[config] is saved at [{config_dir}]')

        self.model.save(model_dir)
        print(f'[model] is saved at [{model_dir}]')

    def load_model(self, saved_model_dir, model_id):

        saved_model_dir = Path(saved_model_dir)
        config_dir = saved_model_dir.joinpath(f'config_{model_id}.json')
        model_dir = saved_model_dir.joinpath(f'model_{model_id}.h5')

        with open(config_dir, 'r') as f:
            self.config = json.load(f)
        print(f'[config] is loaded from [{config_dir}]')

        self.model = load_model(model_dir, 
            custom_objects = {'EmbedContinousCol': EmbedContinousCol})
        print(f'[model] is loaded from [{model_dir}]')