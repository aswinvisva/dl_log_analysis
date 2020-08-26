import numpy as np
import torch

from models.lstm_next_key import DeepLog
from models.seq2seq import EncoderRNN, DecoderRNN, trainIters, MAX_LENGTH, tensorsFromPair
from models.sdae import SDAE
from preprocessing.preprocessing import *
from preprocessing import dataloader
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.cluster import KMeans
from sklearn.metrics import *
from torch.utils.data import DataLoader, Dataset
from sklearn.decomposition import LatentDirichletAllocation

device = torch.device("cuda")


def run_sdae(n_topics=32):
    (x_train, y_train), (x_test, y_test) = dataloader.load_HDFS('data/HDFS_100k.log_structured.csv',
                                                                label_file='data/anomaly_label.csv',
                                                                train_ratio=0.6)
    x_train = x_train[y_train != 1]

    f = FeatureExtractor()
    x_train = f.fit_transform(x_train)
    x_test = f.transform(x_test)

    lda = LatentDirichletAllocation(n_components=n_topics)
    x_train = lda.fit_transform(x_train)
    x_test = lda.transform(x_test)

    x_train = torch.tensor(x_train, dtype=torch.float, device=device)
    x_test = torch.tensor(x_test, dtype=torch.float, device=device)

    train_loader = LDADataset(x_train, x_train)
    test_loader = LDADataset(x_test, x_test)

    model = SDAE(n_topics=n_topics)
    model.fit(train_loader, 3)
    model.evaluate(test_loader, y_test)


def run_deeplog(window_size=4):
    (x_train, window_y_train, y_train), (x_test, window_y_test, y_test) = dataloader.load_HDFS(
        'data/HDFS_100k.log_structured.csv', label_file='data/anomaly_label.csv', window='session',
        window_size=window_size, train_ratio=0.2, split_type='uniform')

    feature_extractor = Vectorizer()
    train_dataset = feature_extractor.fit_transform(x_train, window_y_train, y_train)
    test_dataset = feature_extractor.transform(x_test, window_y_test, y_test)

    train_loader = Iterator(train_dataset, batch_size=32, shuffle=True, num_workers=2).iter
    test_loader = Iterator(test_dataset, batch_size=32, shuffle=False, num_workers=2).iter

    model = DeepLog(num_labels=feature_extractor.num_labels, hidden_size=32, num_directions=2,
                    topk=5, device=device)
    model.fit(train_loader, 50)

    print('Train validation:')
    metrics = model.evaluate(train_loader)

    print('Test validation:')
    metrics = model.evaluate(test_loader)


def run_seq2seq():
    (x_train, y_train), (x_test, y_test) = dataloader.load_HDFS('data/HDFS_100k.log_structured.csv',
                                                                label_file='data/anomaly_label.csv',
                                                                train_ratio=0.8)

    tknzr = Tokenizer(lower=True, split=" ")
    tknzr.fit_on_texts(x_train)

    # making sequences:
    X_train = tknzr.texts_to_sequences(x_train)
    X_test = tknzr.texts_to_sequences(x_test)

    X_train = [(x, x) for x in X_train]
    X_test = [(x, x) for x in X_test]

    print(tknzr.word_index.keys())
    print(X_test)

    hidden_size = 4
    encoder1 = EncoderRNN(len(tknzr.word_index.keys()) + 3, hidden_size).to(device)
    decoder1 = DecoderRNN(hidden_size, len(tknzr.word_index.keys()) + 3).to(device)
    encoder_hidden = encoder1.initHidden()

    trainIters(encoder1, decoder1, X_train, 50000, print_every=100)

    testing_pairs = [tensorsFromPair(i) for i in X_test]

    y_pred_outputs = []

    for iter in range(1, len(testing_pairs) + 1):
        testing_pair = testing_pairs[iter - 1]
        input_tensor = testing_pair[0]
        target_tensor = testing_pair[1]

        input_length = input_tensor.size(0)

        encoder_outputs = torch.zeros(MAX_LENGTH, encoder1.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder1(
                input_tensor[ei], encoder_hidden)
            encoder_outputs[ei] = encoder_output[0, 0]

        y_pred_outputs.append(encoder_outputs.cpu().data.numpy().flatten())

        # x_test_output = encoder1(x_test)
    kmeans = KMeans(n_clusters=2)
    y_pred = kmeans.fit_predict(y_pred_outputs)
    print(len(y_pred))
    print(len(y_test))

    print(y_pred)
    print(y_test)

    print(Counter(y_pred))
    print(Counter(y_test))

    print("Homogeneity Score: %s" % str(homogeneity_score(y_test, y_pred)))
    print("completeness_score: %s" % str(completeness_score(y_test, y_pred)))
    print("v_measure_score: %s" % str(v_measure_score(y_test, y_pred)))
    print("F1 score %s" % str(f1_score(y_test, y_pred)))


if __name__ == '__main__':
    run_sdae()
