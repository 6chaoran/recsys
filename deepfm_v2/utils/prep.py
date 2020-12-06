import numpy as np
import pandas as pd
import json
import functools
import operator


def encode_multilevel_categorical_cols(data, fnames_mhe, encoder_mhe=None):
    """encode multilevel categorical column
    Args
        @data: DataFrame
        @fnames_mhe: str column name
        @encoder_mhe: Dict(fname: idx)
    Return:
        tuple(data_encoded, encoder, MHECols)
    """

    data_encoded = data.copy()

    # build encoders
    if encoder_mhe is None:

        encoder_mhe = {}

        for fname in fnames_mhe:

            levels_i = set(functools.reduce(operator.iconcat, data[fname], []))
            levels_i = [f'{fname}:{i}' for i in list(levels_i)]
            encoder_i = dict(zip(levels_i, list(range(1, len(levels_i) + 1))))
            encoder_mhe[fname] = encoder_i

    # encode each fname
    for fname in fnames_mhe:

        encoder_i = encoder_mhe[fname]
        series_i = data[fname].apply(
            (lambda x: [encoder_i.get(f'{fname}:{i}', 0) for i in x]))
        data_encoded[fname] = series_i

        maxlen = data[fname].apply(len).max()
        vocab_size = max(encoder_i.values()) + 1

    return data_encoded, encoder_mhe


def encode_categorical_cols(data,
                            fnames_cat,
                            encoder_cat=None,
                            combined=False):
    """encode categorical columns
    Args:
        @data (DataFrame): input dataframe
        @fnames_cat (List[str]): categorical variable names
        @encoder_cat (Dict): categorical level -> index
        @combined (bool): whether combine all levels of categorical variables in single encoder
    Return:
        tuple of (data_encoded, encoder)
    """

    # build the encoder/encoders
    if encoder_cat is None:

        if combined:
            levels = []
            for fname in fnames_cat:
                series_i = data[fname].fillna('nan').astype(str)
                levels_i = series_i.map(
                    lambda x: f'{fname}:{x}').unique().tolist()
                levels += levels_i
            encoder_cat = dict(zip(levels, list(range(1, len(levels) + 1))))
        else:
            encoder_cat = {}
            for fname in fnames_cat:
                series_i = data[fname].fillna('nan').astype(str)
                levels_i = series_i.map(
                    lambda x: f'{fname}:{x}').unique().tolist()
                encoder_i = dict(
                    zip(levels_i, list(range(1,
                                             len(levels_i) + 1))))
                encoder_cat[fname] = encoder_i

    # encode categorical variables
    data_encoded = data.copy()
    for fname in fnames_cat:
        series_i = data[fname].fillna('nan').astype(str)
        if combined:
            data_encoded[fname] = series_i.map(
                lambda x: encoder_cat.get(f'{fname}:{x}', 0))
        else:
            encoder_i = encoder_cat[fname]
            data_encoded[fname] = series_i.map(
                lambda x: encoder_i.get(f'{fname}:{x}', 0))

    return data_encoded, encoder_cat


def discretize_continous_cols(data,
                              fnames_num,
                              bins=None,
                              num_bins=64,
                              max_bins=128):
    """discretize continous columns
    Args:
        @data (DataFrame): input dataframe
        @fnames_num (List[str]): numeric variable names
        @num_bins (Int): number of bins
        @bins: Dict(fname : bins cutoff)
        @max_bins (Int): max number of bins allowed
    Return:
        tuple of (data_binned, bins)
    """

    data_binned = data.copy()

    if bins is None:
        bins = {}
        for fname in fnames_num:
            num_bins_i = min(num_bins - 1, data[fname].nunique(),
                             max_bins - 1)  # leave a bin for NAs
            labels_i = [f'bin{i:03d}' for i in range(num_bins_i)]
            series_i, bins_i = pd.cut(data[fname],
                                      bins=num_bins_i,
                                      retbins=True,
                                      labels=labels_i)
            series_i = series_i.cat.add_categories('nan')
            series_i = series_i.fillna('nan').astype(str)
            bins[fname] = bins_i.round(4).tolist()
            data_binned[fname] = series_i
    else:
        for fname in fnames_num:
            num_bins_i = len(bins[fname]) - 1
            labels_i = [f'bin{i:03d}' for i in range(num_bins_i)]
            series_i = pd.cut(data[fname], bins=bins[fname], labels=labels_i)
            series_i = series_i.cat.add_categories('nan')
            series_i = series_i.fillna('nan').astype(str)
            data_binned[fname] = series_i

    return data_binned, bins