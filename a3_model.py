import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch import optim
import random
from sklearn.metrics import f1_score, recall_score, accuracy_score, precision_score
# Whatever other imports you need

# You can implement classes and helper functions here too.

class AuthorPredict(nn.Module):
    def __init__(self, input_size, output_size=1):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        m = self.linear(x)
        n = self.sigmoid(m)
        return n

class AuthorFFNN:
    def __init__(self, lr =0.01):
        self.lr = lr
    def train(self, inputs, samplesize): 
        self.model = AuthorPredict((inputs.shape[1]-2)*2)
        criterion = nn.BCELoss()
        optimizer = optim.SGD(self.model.parameters(), lr=self.lr)

        for i in range(samplesize):
            documents = inputs.drop(inputs.columns[[0,1]], axis=1)
            labels = inputs.iloc[:, 1:2]
            in1 = random.randint(0, len(documents))
            auth = labels.iloc[in1, 0]
            in_same = labels[labels.iloc[:, 0] == auth].index
            in_diff = labels[labels.iloc[:, 0] != auth].index
            coin = random.randint(0, 1)
            if coin == 0:
                in2 = random.choice(in_diff)
            if coin == 1:
                in2 = random.choice(in_same)
            docs = documents.to_numpy()
            doc1 = torch.Tensor(docs[in1])
            doc2 = torch.Tensor(docs[in2])
            instance = torch.cat((doc1, doc2), 0)
            output = self.model(instance)
            label = torch.Tensor([coin])
            loss = criterion(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def test(self, inputs, samplesize):
        pred_list = []
        label_list = []
        for i in range(samplesize):
            documents = inputs.drop(inputs.columns[[0,1]], axis=1)
            labels = inputs.iloc[:, 1:2]
            in1 = random.randint(0, len(documents))
            auth = labels.iloc[in1, 0]
            in_same = labels[labels.iloc[:, 0] == auth].index
            in_diff = labels[labels.iloc[:, 0] != auth].index
            coin = random.randint(0, 1)
            if coin == 0:
                in2 = random.choice(in_diff)
            if coin == 1:
                in2 = random.choice(in_same)
            label_list.append(coin)
            docs = documents.to_numpy()
            doc1 = torch.Tensor(docs[in1])
            doc2 = torch.Tensor(docs[in2])
            instance = torch.cat((doc1, doc2), 0)

            output = self.model(instance)
            if output > 0.5:
                predict = 1
            else:
                predict = 0
            pred_list.append(predict)
        print(label_list)

        accuracy = accuracy_score(label_list, pred_list)
        f = f1_score(label_list, pred_list, average='weighted')
        recall = recall_score(label_list, pred_list, average='weighted')
        precision = precision_score(label_list, pred_list, average='weighted')
        print('accuracy:', accuracy, 'precision:', precision, 'recall:', recall, 'f1_score:', f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and test a model on features.")
    parser.add_argument("featurefile", type=str, help="The file containing the table of instances and features.")
    parser.add_argument("samplesize", type = int, help="The number of text pairs used to train the FFNN.")
    # Add options here for part 3 -- hidden layer and nonlinearity,
    # and any other options you may think you want/need.  Document
    # everything.
    
    args = parser.parse_args()

    print("Reading {}...".format(args.featurefile))
    data = pd.read_csv(args.featurefile, header = None)
    train = data[data[0] == "train"]
    train.reset_index(inplace=True, drop=True)
    test = data[data[0] == "test"]
    test.reset_index(inplace=True, drop=True)

    ffnn = AuthorFFNN()
    ffnn.train(train, args.samplesize)
    ffnn.test(test, args.samplesize)

    # implement everything you need here
    

