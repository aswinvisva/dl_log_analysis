import random

import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.autograd import Variable
from torch import optim


class SDAE(nn.Module):
    def __init__(self, n_topics=60):
        super(SDAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(n_topics, 100),
            nn.Sigmoid(),
            nn.Linear(100, 150),
            nn.Sigmoid()
        )

        self.decoder = nn.Sequential(
            nn.Linear(150, 100),
            nn.Sigmoid(),
            nn.Linear(100, n_topics),
            nn.Sigmoid()
        )

        self.criterion = nn.MSELoss()
        self.rl_threshold = 0.001
        self.device = torch.device("cuda")

    def forward(self, data_dict):
        x = data_dict[0]
        y = data_dict[1]

        encoder_output = self.encoder(x)
        y_pred = self.decoder(encoder_output)

        loss = self.criterion(y_pred, y)
        return_dict = {'loss': loss, 'y_pred': y_pred}

        return return_dict

    def fit(self, train_loader, epochs=100):
        self.to(self.device)
        model = self.train()
        optimizer = optim.Adam(model.parameters())

        for epoch in range(epochs):
            batch_cnt = 0
            epoch_loss = 0
            for batch_input in train_loader:
                loss = model.forward(batch_input)["loss"]
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                epoch_loss += loss.item()
                batch_cnt += 1
            epoch_loss = epoch_loss / batch_cnt
            print("Epoch {}/{}, training loss: {:.5f}".format(epoch + 1, epochs, epoch_loss))

    def evaluate(self, test_loader, anomaly_true):
        self.eval()  # set to evaluation mode

        anomaly_pred = []
        with torch.no_grad():
            for batch_input in test_loader:
                return_dict = self.forward(batch_input)
                y_pred = return_dict["y_pred"]
                y_true = batch_input[1]

                reconstruction_loss = torch.mean(torch.square(y_pred - y_true)).data.cpu().numpy()

                anomaly_pred.append(reconstruction_loss > self.rl_threshold)

        metrics = {"acc": accuracy_score(anomaly_true, anomaly_pred),
                   "f1": f1_score(anomaly_true, anomaly_pred),
                   "recall": recall_score(anomaly_true, anomaly_pred),
                   "precision": precision_score(anomaly_true, anomaly_pred)}

        print([(k, round(v, 5)) for k, v in metrics.items()])
        return metrics
