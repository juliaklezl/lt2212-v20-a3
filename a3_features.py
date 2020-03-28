import os
import sys
import argparse
import numpy as np
import pandas as pd
# Whatever other imports you need
from glob import glob
from  sklearn.decomposition import PCA
from random import choices
import csv

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert directories into table.")
    parser.add_argument("inputdir", type=str, help="The root of the author directories.")
    parser.add_argument("outputfile", type=str, help="The name of the output file containing the table of instances.")
    parser.add_argument("dims", type=int, help="The output feature dimensions.")
    parser.add_argument("--test", "-T", dest="testsize", type=int, default="20", help="The percentage (integer) of instances to label as test.")

    args = parser.parse_args()

    print("Reading {}...".format(args.inputdir))
    # Do what you need to read the documents here.
    directories = glob("{}/*".format(args.inputdir))
    rows = [] #initiate list of dictionaries (one per news post)
    index = 0 # initiate index (of words)
    index_dict = {} # initiate dict to keep track of words and indexes
    author_list = [] # initiate list for author labels
    print("Constructing table with {} feature dimensions and {}% test instances...".format(args.dims, args.testsize))
    # Build the table here.
    for directory in directories:   
        author = directory.split("/")[-1]
        filenames = glob("{}/*".format(directory))
        for file in filenames:
            row_counts = {} # initiate dict to keep track of words and indexes
            author_list.append(author)
            with open(file, "r") as the_file:
                wholetext = the_file.readlines()
                for  line in wholetext:
                    for word in line.split():
                        if word.isalpha():
                            word = word.lower() # lowercase
                            if word in index_dict:
                                i = index_dict[word] # get index, if already assigned,
                            else:
                                index_dict[word] = index # otherwise assign index
                                i = index
                                index += 1 # and raise index counter by one
                            if i in row_counts: # get word counts per text
                                row_counts[i] += 1
                            else:
                                row_counts[i] = 1
            rows.append(row_counts)
    features = np.zeros((len(rows), len(index_dict))) # instantiate empty ndarray
    for n, dict in enumerate(rows):
        for ind, val in dict.items():
            features[n, ind] = val # replace zeros by counts
    pca = PCA(n_components = args.dims)
    red_features = pca.fit_transform(features)
    author_array = np.array([author_list])
    red_features = np.concatenate((author_array.T, red_features), axis=1)
    test = args.testsize/100
    train = 1-test
    train_test_list = choices(["train", "test"], [train, test], k = len(author_list))
    tt_array = np.array([train_test_list])
    finished_features = np.concatenate((tt_array.T, red_features), axis = 1)
    # write to csv
    print("Writing to {}...".format(args.outputfile))
    with open(args.outputfile, 'w+') as output:
        csvWriter = csv.writer(output, delimiter=',')
        csvWriter.writerows(finished_features)
#    np.savetxt(output, finished_features, delimiter=',')
    # Write the table out here.
    print(finished_features)
    print("Done!")
    
