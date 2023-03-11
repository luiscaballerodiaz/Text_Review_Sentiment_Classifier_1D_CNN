import pandas as pd
import utils
from data_visualization import DataPlots
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from keras import layers
from keras import optimizers
from keras import models
from keras import regularizers
from keras import callbacks


action = 2
# 0 --> simulate CNN neural network
# 1 --> sweep lineal models
# 2 --> load CNN neural network, simulate optimal lineal models and ensembled them
epochs = 40
learning_rate = 0.0001
dropout = 0.5
l2_reg = 0.1
batch_size = 32
vocabulary_words = 10000
max_words_review = 300
model_to_load = 'CNN network dropout=0.5 l2 reg=0.15 - Trained model.h5'

visualization = DataPlots()
source_df = pd.read_csv('tripadvisor_hotel_reviews.csv')
# utils.data_analytics(source_df, visualization)
x_train, x_test, y_train, y_test, x_train_keras, x_val_keras, y_train2, y_val, x_test_keras =\
    utils.dataframe_modification_and_split(source_df, visualization, max_words_review, vocabulary_words)

if action == 0:
    tag = 'CNN network dropout=' + str(dropout) + ' l2 reg=' + str(l2_reg)
    callbacks_list = callbacks.ModelCheckpoint(tag + ' - Trained model.h5', monitor='val_acc', save_best_only=True,
                                               verbose=1, mode='max')
    model = models.Sequential()
    model.add(layers.Embedding(vocabulary_words, 128, input_length=max_words_review))
    model.add(layers.Dropout(dropout))
    model.add(layers.Conv1D(32, 7, activation='relu', kernel_regularizer=regularizers.l2(l2_reg)))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(dropout))
    model.add(layers.MaxPooling1D(3))
    model.add(layers.Conv1D(32, 7, activation='relu', kernel_regularizer=regularizers.l2(l2_reg)))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(dropout))
    model.add(layers.GlobalMaxPooling1D())
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer=optimizers.RMSprop(learning_rate=learning_rate), loss='binary_crossentropy',
                  metrics=['acc'])
    model.summary()
    history = model.fit(x_train_keras, y_train2, batch_size=batch_size, callbacks=callbacks_list, epochs=epochs,
                        validation_data=(x_val_keras, y_val))
    model = models.load_model(tag + ' - Trained model.h5')
    results = model.evaluate(x_test_keras, y_test, batch_size=batch_size)
    print('test {}'.format(results[1]))
    print('val {}'.format(model.evaluate(x_val_keras, y_val, batch_size=batch_size)[1]))
    print('train {}'.format(model.evaluate(x_train_keras, y_train2, batch_size=batch_size)[1]))
    visualization.plot_results(tag, results, history.history['acc'], history.history['val_acc'],
                               history.history['loss'], history.history['val_loss'])

elif action == 1:
    model = LinearSVC(random_state=0, dual=False)
    param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}
    grid_search = GridSearchCV(model, param_grid, cv=5)
    grid_search.fit(x_train, y_train)
    print('LINEAR SVC')
    print("Best parameters: {}".format(grid_search.best_params_))
    print("Best cross-validation score: {:.4f}".format(grid_search.best_score_))
    print(pd.DataFrame(grid_search.cv_results_)[['mean_test_score', 'params']])

    model = LogisticRegression(random_state=0)
    param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}
    grid_search = GridSearchCV(model, param_grid, cv=5)
    grid_search.fit(x_train, y_train)
    print('LOGISTIC REGRESSION')
    print("Best parameters: {}".format(grid_search.best_params_))
    print("Best cross-validation score: {:.4f}".format(grid_search.best_score_))
    print(pd.DataFrame(grid_search.cv_results_)[['mean_test_score', 'params']])

elif action == 2:
    model = LinearSVC(C=0.01, random_state=0, dual=False)
    model = CalibratedClassifierCV(model)
    model.fit(x_train, y_train)
    print('LINEARSVC MODEL TEST SCORE: {:.4f}'.format(model.score(x_test, y_test)))
    linearsvc_preds = model.predict_proba(x_test)[:, 1]

    model = LogisticRegression(C=0.1, random_state=0)
    model.fit(x_train, y_train)
    print('LOGISTIC REGRESSION MODEL TEST SCORE: {:.4f}'.format(model.score(x_test, y_test)))
    logreg_preds = model.predict_proba(x_test)[:, 1]

    model = models.load_model(model_to_load)
    print('CNN MODEL TEST SCORE: {:.4f}'.format(model.evaluate(x_test_keras, y_test, batch_size=batch_size)[1]))
    cnn_preds = model.predict(x_test_keras, batch_size=1)

    ok = 0
    y_model = []
    for i in range(len(y_test)):
        y_pred = (1/3) * cnn_preds[i] + (1/3) * linearsvc_preds[i] + (1/3) * logreg_preds[i]
        if y_pred >= 0.5:
            y_model.append(1)
        else:
            y_model.append(0)
        if y_model[i] == y_test[i]:
            ok += 1
    print('ENSEMBLED MODEL TEST SCORE: {:.4f}'.format(ok/len(y_test)))
