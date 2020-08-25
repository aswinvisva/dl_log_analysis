import numpy as np
import torch

from models.seq2seq import EncoderRNN, DecoderRNN, trainIters, tensorFromSentence, MAX_LENGTH, tensorsFromPair
from preprocessing.preprocessing import *
from preprocessing import dataloader
from tensorflow.keras.preprocessing.text import Tokenizer, text_to_word_sequence
from sklearn.cluster import KMeans
from sklearn.metrics import *

device = torch.device("cuda")


def main():
    (x_train, y_train), (x_test, y_test) = dataloader.load_HDFS('data/HDFS_100k.log_structured.csv',
                                                                label_file='data/anomaly_label.csv')

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
    main()
