import pandas as pd
from data_visualization import DataPlots
from keras import models
from keras import callbacks
import utils

action = 2
# 0 --> simulate CNN neural network
# 1 --> sweep lineal models
# 2 --> load CNN neural network, simulate optimal lineal models and ensembled them
# 3 --> analysis for logreg coeffs
epochs = 40
learning_rate = 0.0001
dropout = 0.6
l2_reg = 0.1
batch_size = 32
vocabulary_words = 10000
max_words_review = 300
model_to_load = 'CNN network dropout=0.5 l2 reg=0.1 - Trained model.h5'

visualization = DataPlots()
source_df = pd.read_csv('tripadvisor_hotel_reviews.csv')
utils.data_analytics(source_df, visualization)
x_train, x_test, x_val, y_train, y_test, x_train_keras, x_val_keras, y_train2, y_val, x_test_keras, features =\
    utils.dataframe_modification_and_split(source_df, visualization, max_words_review, vocabulary_words)

if action == 0:
    tag = 'CNN network dropout=' + str(dropout) + ' l2 reg=' + str(l2_reg)
    model = utils.create_1D_CNN(dropout, l2_reg, learning_rate, vocabulary_words, max_words_review)
    callbacks_list = callbacks.ModelCheckpoint(tag + ' - Trained model.h5', monitor='val_acc', save_best_only=True,
                                               verbose=1, mode='max')
    history = model.fit(x_train_keras, y_train2, batch_size=batch_size, callbacks=callbacks_list, epochs=epochs,
                        validation_data=(x_val_keras, y_val))
    model = models.load_model(tag + ' - Trained model.h5')
    results = model.evaluate(x_test_keras, y_test, batch_size=batch_size)
    visualization.plot_results(tag, results, history.history['acc'], history.history['val_acc'],
                               history.history['loss'], history.history['val_loss'])

elif action == 1:
    utils.sweep_linear_models(x_train, y_train)

elif action == 2:
    utils.test_set_prediction(x_train, y_train, x_test, x_test_keras, y_test, x_val, x_val_keras, y_val,
                              model_to_load, batch_size)
elif action == 3:
    utils.linearmodels_coeffs_analysis(visualization, x_train, y_train, features)
