import numpy as np
import torch

from models.lstm_next_key import DeepLog
from models.seq2seq import EncoderRNN, DecoderRNN, trainIters, MAX_LENGTH, tensorsFromPair
from models.sdae import SDAE
from preprocessing.preprocessing import *
from preprocessing import dataloader
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import *
from torch.utils.data import DataLoader, Dataset
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

device = torch.device("cuda")


def run_sdae(n_topics=8):
    (x_train, y_train), (x_test, y_test) = dataloader.load_HDFS('data/HDFS.npz',
                                                                train_ratio=0.5)

    f = FeatureExtractor()
    x_train = f.fit_transform(x_train)
    x_test = f.transform(x_test)

    lda = LatentDirichletAllocation(n_components=n_topics, n_jobs=8)
    x_train = lda.fit_transform(x_train)
    x_test = lda.transform(x_test)

    x_train = x_train[y_train != 1]

    x_train = torch.tensor(x_train, dtype=torch.float, device=device)
    x_test = torch.tensor(x_test, dtype=torch.float, device=device)

    train_loader = LDADataset(x_train, x_train)
    test_loader = LDADataset(x_test, x_test)
    train_loader = DataLoader(dataset=train_loader, batch_size=2048, shuffle=False)
    test_loader = DataLoader(dataset=test_loader, batch_size=2048, shuffle=False)

    model = SDAE(n_topics=n_topics)
    model.fit(train_loader, 10)
    metrics, predictions, anomalies = model.evaluate(test_loader, y_test)

    X_embedded = TSNE(n_components=2, learning_rate='auto', init='random', n_jobs=8).fit_transform(predictions)

    plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=anomalies)
    plt.savefig("tsne.png")
    plt.clf()

def run_deeplog(window_size=10):
    (x_train, window_y_train, y_train), (x_test, window_y_test, y_test) = dataloader.load_HDFS(
        'data/HDFS.npz', window='session', window_size=window_size, train_ratio=0.025, split_type='uniform')

    feature_extractor = Vectorizer()
    train_dataset = feature_extractor.fit_transform(x_train, window_y_train, y_train)
    test_dataset = feature_extractor.transform(x_test, window_y_test, y_test)

    train_loader = Iterator(train_dataset, batch_size=5192, shuffle=True, num_workers=4).iter
    test_loader = Iterator(test_dataset, batch_size=5192, shuffle=False, num_workers=4).iter

    model = DeepLog(num_labels=feature_extractor.num_labels, hidden_size=16, num_directions=2,
                    topk=8, device=device)
    model.fit(train_loader, 10)

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

    hidden_size = 32
    encoder1 = EncoderRNN(len(tknzr.word_index.keys()) + 3, hidden_size).to(device)
    decoder1 = DecoderRNN(hidden_size, len(tknzr.word_index.keys()) + 3).to(device)
    encoder_hidden = encoder1.initHidden()

    trainIters(encoder1, decoder1, X_train, 5000, print_every=100)

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

        y_pred_outputs.append(encoder_output.cpu().data.numpy().flatten())

    dbscan = DBSCAN(eps=0.075, min_samples=100,  metric="cosine")
    y_pred = dbscan.fit_predict(y_pred_outputs).tolist()

    y_pred = np.array([1 if i == -1 else 0 for i in y_pred])

    print("Homogeneity Score: %s" % str(homogeneity_score(y_test, y_pred)))
    print("completeness_score: %s" % str(completeness_score(y_test, y_pred)))
    print("v_measure_score: %s" % str(v_measure_score(y_test, y_pred)))
    print("F1 score %s" % str(f1_score(y_test, y_pred)))
    print("Precision score %s" % str(precision_score(y_test, y_pred)))
    print("Recall score %s" % str(recall_score(y_test, y_pred)))


if __name__ == '__main__':
    run_sdae()
