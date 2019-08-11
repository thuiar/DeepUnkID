from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import *
from keras.layers.core import Lambda
from keras.constraints import unit_norm
from keras.optimizers import Adam
from keras.utils import plot_model

def large_margin_cosine_loss(y_true, y_pred, scale=30, margin=0.35):
    y_pred = y_true * (y_pred - margin) + (1 - y_true) * y_pred
    y_pred *= scale
    return K.categorical_crossentropy(y_true, y_pred, from_logits=True)


def BiLSTM_LMCL(max_seq_len, max_features, embedding_dim, output_dim, model_img_path=None, embedding_matrix=None):
    model = Sequential()
    if embedding_matrix is None:
        model.add(Embedding(max_features, embedding_dim, input_length=max_seq_len, mask_zero=True))
    else:
        model.add(Embedding(max_features, embedding_dim, input_length=max_seq_len, mask_zero=True,
                            weights=[embedding_matrix], trainable=True))

    model.add(Bidirectional(LSTM(128, dropout=0.5)))
    model.add(Dropout(0.5))
    model.add(Lambda(lambda x: K.l2_normalize(x, axis=1)))
    adam = Adam(lr=0.003, clipnorm=5.)
    
    model.add(Dense(output_dim, use_bias=False, kernel_constraint=unit_norm()))
    model.add(Activation('softmax'))
    model.compile(loss=large_margin_cosine_loss, optimizer=adam, metrics=['accuracy'])

    if model_img_path:
        plot_model(model, to_file=model_img_path, show_shapes=True, show_layer_names=False)
    
    return model
