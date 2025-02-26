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
import matplotlib.pyplot as plt

# Whatever other imports you need

# You can implement classes and helper functions here too.

class AuthorPredict(nn.Module):
    def __init__(self, input_size, hiddensize, nonlin, output_size=1):
        super().__init__()
        self.hiddensize = hiddensize
        nonlinearities = {"ReLU": nn.ReLU(inplace=False), "Tanh": torch.nn.Tanh()}
        self.nonlin = nonlin
        if hiddensize > 0:
            self.linear1 = nn.Linear(input_size, hiddensize)
            self.linear2 = nn.Linear(hiddensize, output_size)
        else:
            self.linear = nn.Linear(input_size, output_size)
        if nonlin != " ":
            self.nonlinearity = nonlinearities[nonlin]
        else:
            self.nonlinearity = 0
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        if self.hiddensize > 0:
            if self.nonlinearity:            
                m = self.linear1(x)
                n = self.nonlinearity(m)
                o = self.linear2(n)
            else:
                m = self.linear1(x)
                o = self.linear2(m)
        else:
            o = self.linear(x)
        z = self.sigmoid(o)
        return z

class AuthorFFNN:
    def __init__(self, lr =0.01):
        self.lr = lr
    def train(self, inputs, hiddensize, nonlin, samplesize, epochs=3): 
        self.model = AuthorPredict((inputs.shape[1]-2)*2, hiddensize, nonlin)
        criterion = nn.BCELoss()
        optimizer = optim.SGD(self.model.parameters(), lr=self.lr)
        documents = inputs.drop(inputs.columns[[0,1]], axis=1)
        labels = inputs.iloc[:, 1:2]


        for epoch in range(epochs):
            for i in range(samplesize):
        #        documents = inputs.drop(inputs.columns[[0,1]], axis=1)
         #       labels = inputs.iloc[:, 1:2]
                in1 = random.randint(0, len(documents)-1)
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
        documents = inputs.drop(inputs.columns[[0,1]], axis=1)
        labels = inputs.iloc[:, 1:2]
        for i in range(samplesize):
            in1 = random.randint(0, len(documents)-1)
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
        return(recall, precision)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and test a model on features.")
    parser.add_argument("featurefile", type=str, help="The file containing the table of instances and features.")
    parser.add_argument("--size", dest="samplesize", default = 150, type = int, help="The number of text pairs used to train the FFNN.")
    parser.add_argument("--testsize", dest="test_samplesize", default = 30, type = int, help="number of test samples")
    parser.add_argument("--hidden", dest="hiddensize", default = 0, type = int, help = "Size of hidden layer")
    parser.add_argument("--nonlin", dest ="nonlinearity", default = " ", type = str, help = "Set one of two nonlinearities ReLU or Tanh")
    parser.add_argument("outputfile", type = str, help = "path of the output file (graph)")
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

    hiddensize = [0, 10, 50, 70, 100]
    precision = []
    recall = []
    for size in hiddensize:
        ffnn = AuthorFFNN()
        ffnn.train(train, size, args.nonlinearity, args.samplesize)
        results = ffnn.test(test, args.test_samplesize)
        recall.append(results[0])
        precision.append(results[1])
    prec_rec = {}
    prec_rec["precision"] = precision
    prec_rec["recall"] = recall
    df = pd.DataFrame.from_dict(prec_rec)
    df.sort_values("recall")
    plot1 = df.plot(x="recall", y="precision", kind = "line")    
    plot1.get_figure().savefig(args.outputfile)
