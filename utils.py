from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from keras.utils import pad_sequences
from keras import layers
import numpy as np


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

    one_hot = layers.Hashing(num_bins=vocabulary_words, output_mode='int')
    x_train_enc = [one_hot(review.split()) for review in x_train2]
    x_train_keras = pad_sequences(x_train_enc, maxlen=max_words_review, padding='post')
    x_val_enc = [one_hot(review.split()) for review in x_val]
    x_val_keras = pad_sequences(x_val_enc, maxlen=max_words_review, padding='post')
    x_test_enc = [one_hot(review.split()) for review in x_test]
    x_test_keras = pad_sequences(x_test_enc, maxlen=max_words_review, padding='post')

    vect = CountVectorizer(min_df=10, max_df=0.75, stop_words='english')
    vect.fit(x_train)
    x_train = vect.transform(x_train).toarray()
    x_test = vect.transform(x_test).toarray()
    y_test = y_test.reset_index(drop=True)

    return x_train, x_test, y_train, y_test, x_train_keras, x_val_keras, y_train2, y_val, x_test_keras


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
    set3 = set(features[2])
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
