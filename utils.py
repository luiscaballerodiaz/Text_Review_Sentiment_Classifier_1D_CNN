from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from keras.utils import pad_sequences
from keras import layers
from keras import models
from keras import optimizers
from keras import regularizers
from scipy.optimize import minimize
import pandas as pd
import numpy as np


def minimize_acc(weights, y_true, y_preds):
    """ Calculate the score of a weighted model predictions"""
    return -accuracy_score(weights, y_true, y_preds)


def accuracy_score(weights, y_true, y_preds):
    ok = 0
    for i in range(len(y_true)):
        y_pred = np.round(np.dot(y_preds[i], weights))
        if y_pred == y_true[i]:
            ok += 1
    return ok / len(y_true)


def calculate_optimal_weights(y_true, y_preds):
    acc_opt = 100
    acc_weights_opt = 0
    for i in range(100):
        weights_ini = np.random.rand(y_preds.shape[1])
        weights_ini /= np.sum(weights_ini)
        acc_minimizer = minimize(fun=minimize_acc,
                                 x0=weights_ini,
                                 method='SLSQP',
                                 args=(y_true, y_preds),
                                 bounds=[(0, 1)] * y_preds.shape[1],
                                 options={'disp': True, 'maxiter': 10000, 'eps': 1e-10, 'ftol': 1e-8},
                                 constraints={'type': 'eq', 'fun': lambda w: w.sum() - 1})
        if acc_minimizer.fun < acc_opt:
            acc_opt = acc_minimizer.fun
            acc_weights_opt = acc_minimizer.x
    return acc_weights_opt


def linearmodels_coeffs_analysis(visualization, x_train, y_train, features, words=25):
    for sim in range(2):
        if sim == 0:
            model = LogisticRegression(C=0.1, random_state=0)
            tag = 'Logistic regression'
        elif sim == 1:
            model = LinearSVC(C=0.01, random_state=0, dual=False)
            tag = 'Linear SVC'
        model.fit(x_train, y_train)
        coeffs = model.coef_
        max_ind = np.argsort(-coeffs)[:words]
        min_ind = np.argsort(coeffs)[:words]
        max_coeffs = []
        min_coeffs = []
        max_feats = []
        min_feats = []
        for i in range(words):
            max_coeffs.append(coeffs[0][max_ind[0][i]])
            min_coeffs.append(coeffs[0][min_ind[0][i]])
            max_feats.append([key for key, value in features.items() if value == max_ind[0][i]])
            min_feats.append([key for key, value in features.items() if value == min_ind[0][i]])
        visualization.linearmodels_coeffs_plot(tag, max_feats, max_coeffs, min_feats, min_coeffs)


def test_set_prediction(x_train, y_train, x_test, x_test_keras, y_test, x_val, x_val_keras, y_val, load, batch_size):
    model = LinearSVC(C=0.01, random_state=0, dual=False)
    model = CalibratedClassifierCV(model)
    model.fit(x_train, y_train)
    print('LINEARSVC MODEL TEST SCORE: {:.4f}'.format(model.score(x_test, y_test)))
    linearsvc_val_preds = model.predict_proba(x_val)[:, 1].reshape(-1, 1)
    linearsvc_test_preds = model.predict_proba(x_test)[:, 1].reshape(-1, 1)

    model = LogisticRegression(C=0.1, random_state=0)
    model.fit(x_train, y_train)
    print('LOGISTIC REGRESSION MODEL TEST SCORE: {:.4f}'.format(model.score(x_test, y_test)))
    logreg_val_preds = model.predict_proba(x_val)[:, 1].reshape(-1, 1)
    logreg_test_preds = model.predict_proba(x_test)[:, 1].reshape(-1, 1)

    model = models.load_model(load)
    print('CNN MODEL TEST SCORE: {:.4f}'.format(model.evaluate(x_test_keras, y_test, batch_size=batch_size)[1]))
    cnn_val_preds = model.predict(x_val_keras, batch_size=1)
    cnn_test_preds = model.predict(x_test_keras, batch_size=1)

    y_test = np.array(y_test)
    y_val = np.array(y_val)

    y_val_preds = np.c_[linearsvc_val_preds, logreg_val_preds, cnn_val_preds]
    y_test_preds = np.c_[linearsvc_test_preds, logreg_test_preds, cnn_test_preds]
    opt_ensemble_weights = calculate_optimal_weights(y_val, y_val_preds)

    opt_ensemble_weights = [0, 2/3, 1/3]
    acc_ensemble_pred = accuracy_score(opt_ensemble_weights, y_test, y_test_preds)
    print('\nENSEMBLE 1D CNN + LOGISTIC REGRESSION TEST SCORE: {:.4f}'.format(acc_ensemble_pred))
    print('\nOPTIMAL WEIGHTS: {}'.format(opt_ensemble_weights))

    opt_ensemble_weights = [1/2, 1/2, 0]
    acc_ensemble_pred = accuracy_score(opt_ensemble_weights, y_test, y_test_preds)
    print('\nENSEMBLE LINEAR SVC + LOGISTIC REGRESSION TEST SCORE: {:.4f}'.format(acc_ensemble_pred))
    print('\nOPTIMAL WEIGHTS: {}'.format(opt_ensemble_weights))


def sweep_linear_models(x_train, y_train):
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


def create_1D_CNN(dropout, l2_reg, learning_rate, vocabulary_words, max_words_review):
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
    return model


def dataframe_modification_and_split(df, visualization, max_words_review, vocabulary_words):
    df = df[df['Rating'] != 3].reset_index(drop=True)
    df['Sentiment'] = np.where(df['Rating'] > 3, 1, 0)
    x_train, x_test, y_train, y_test = train_test_split(df['Review'], df['Sentiment'], test_size=0.2,
                                                        shuffle=True, stratify=df['Sentiment'], random_state=0)
    x_train2, x_val, y_train2, y_val = train_test_split(x_train, y_train, test_size=0.2,
                                                        shuffle=True, stratify=y_train, random_state=0)
    grouped = df.groupby(['Sentiment']).size()
    weights = grouped.values
    visualization.pie_plot(['negative', 'positive'], weights, 'Sentiment', 'full set')
    visualization.pie_plot(['positive', 'negative'],
                           [sum(y_train), len(y_train)-sum(y_train)], 'Sentiment', 'train set')
    visualization.pie_plot(['positive', 'negative'],
                           [sum(y_train2), len(y_train2) - sum(y_train2)], 'Sentiment', 'train neural network set')
    visualization.pie_plot(['positive', 'negative'], [sum(y_test), len(y_test)-sum(y_test)], 'Sentiment', 'test set')
    visualization.pie_plot(['positive', 'negative'], [sum(y_val), len(y_val) - sum(y_val)], 'Sentiment', 'val set')

    hashing = layers.Hashing(num_bins=vocabulary_words, output_mode='int')
    x_train_enc = [hashing(review.split()) for review in x_train2]
    x_train_keras = pad_sequences(x_train_enc, maxlen=max_words_review, padding='post')
    x_val_enc = [hashing(review.split()) for review in x_val]
    x_val_keras = pad_sequences(x_val_enc, maxlen=max_words_review, padding='post')
    x_test_enc = [hashing(review.split()) for review in x_test]
    x_test_keras = pad_sequences(x_test_enc, maxlen=max_words_review, padding='post')

    vect = CountVectorizer(min_df=10, max_df=0.75, stop_words='english')
    vect.fit(x_train)
    x_train = vect.transform(x_train).toarray()
    x_test = vect.transform(x_test).toarray()
    x_val = vect.transform(x_val).toarray()
    features = vect.vocabulary_
    y_test = y_test.reset_index(drop=True)
    y_val = y_val.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_train2 = y_train2.reset_index(drop=True)

    return x_train, x_test, x_val, y_train, y_test, x_train_keras, x_val_keras, y_train2, y_val, x_test_keras, features


def data_analytics(df, visualization):
    grouped = df.groupby(['Rating']).size()
    labels = grouped.index.values
    weights = grouped.values
    visualization.pie_plot(labels, weights, 'Rating', 'full set')
    vect = CountVectorizer(min_df=10, max_df=0.75, stop_words='english')
    ratings = [1, 2, 3, 4, 5]
    features = []
    repeat = []
    repeat_ind = []
    lengths = []
    list_length = []
    for rating in ratings:
        data = df[df['Rating'] == rating]['Review']
        list_length.append([len(data.tolist()[i].split()) for i in range(len(data))])
        lengths.append(np.array(list_length[rating-1]).mean())
        vect.fit(data)
        bag_of_words = vect.transform(data).toarray()
        print(f'Number of words with rating {rating}: {bag_of_words.shape[1]}')
        repeat.append(np.sum(bag_of_words, axis=0))
        repeat_ind.append(np.argsort(-repeat[rating-1]))
        features.append(vect.get_feature_names_out())

    visualization.plot_most_common_words(ratings, repeat, repeat_ind, features, 25, 'Most common')
    visualization.plot_length_histogram(ratings, list_length)
    visualization.plot_lengths(ratings, lengths)
    matrix = np.zeros([len(ratings), len(ratings)])
    exc_feats = []
    for i in range(len(ratings)):
        setA = set(features[i])
        setC = setA.copy()
        for j in range(len(ratings)):
            setB = set(features[j])
            if i != j:
                setC -= setB
            matrix[i, j] = len(setA & setB)
        exc_feats.append(np.array(list(setC)))
    visualization.plot_shared_words(ratings, matrix)

    exc_repeat = []
    exc_repeat_ind = []
    for i in range(len(features)):
        exc_rep = []
        for word in exc_feats[i]:
            exc_rep.append(repeat[i][features[i].tolist().index(word)])
        exc_repeat.append(np.array(exc_rep))
        exc_repeat_ind.append(np.argsort(-exc_repeat[i]))
    visualization.plot_most_common_words(ratings, exc_repeat, exc_repeat_ind, exc_feats, 25, 'Exclusive')

    set1 = set(features[0])
    set2 = set(features[1])
    set4 = set(features[3])
    set5 = set(features[4])

    sent_repeat = []
    sent_repeat_ind = []
    feat_sentiment = [np.array(list((set1 & set2) - set5 - set4)), np.array(list((set5 & set4) - set1 - set2))]
    sent = ['negative', 'positive']
    for i in range(len(sent)):
        sent_rep = []
        for word in feat_sentiment[i]:
            sent_rep.append(repeat[i*3][features[i*3].tolist().index(word)] + repeat[i*3+1][features[i*3+1].tolist().index(word)])
        sent_repeat.append(np.array(sent_rep))
        sent_repeat_ind.append(np.argsort(-sent_repeat[i]))
    visualization.plot_most_common_words(sent, sent_repeat, sent_repeat_ind, feat_sentiment, 25, 'Sentiment')
