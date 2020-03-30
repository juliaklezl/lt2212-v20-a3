# LT2212 V20 Assignment 3
 
## Part 1
I loop through all the subdirectories, keeping track of the author names (directory names), create a word-based feature vector (tokenized by spaces again, removed punctuation and numbers, and lowercased everything) for every file and collect all vectors in a numpy array. I reused parts of my code for assignment 2 for this step. Then I reduce the dimensionality of the whole dataset, and add a column for the author names. I use a random 80/20 split for creating a test and training set and add another column with these labels to the array. Finally, I use the csv module to save the array in a csv file. 
I did not add additional arguments, so the file is run as follows: python3 a3_features inputdirectory outputfile_path number_of_dimensions (-- test size_of_testset). An example is: python3 a3_features.py enron_sample/ out.csv 500

## Part 2
In this part, I added two additional command line arguments, size, and testsize. With these, the number of samples for raining and testing the model can be adjusted. The default settings are rather small with 150 for training and 50 for testing, mainly in order to keep the processing time manageable while I was working and debugging. 
I read the training and testing sets from the csv file into two dataframes, which are used for the sampling process. For sampling, I separate the actual vectors from the author labels again. I choose a random index for the first of the two texts, then separate the remaining dataset into those with the same author and those with another author. In the last step of the sampling process, I randomly choose one text from either of those groups, with a probability of 50% each in order to keep the data balanced. 
The number of epochs is set to 3, since we used that number in the example in class, and again to limit the training time. 
In the test function, I use the predictions, actual labels, and sklearns metrics module to get f1-score, precision, recall, and accuracy. 

## Part 3
I added optional arguments for the hidden layer size and the nonlinearity. The default settings are 0 and " ". For the nonlinearity, I chose Tanh and ReLU, since these are mentioned in the course literature. So the program is now run in this format: python3 a3_model.py featurefile_path (--size number_of_training_samples) (--testsize number_of_testsamples) (--hidden size_of_hidden_layer) (--nonlin choice_of_nonlinearity) An example call is python3 a3_model.py out.csv --hidden 20 --nonlin ReLU
Results with different hidden layer sizes and nonlinearities:

|  Hidden layer | Non-lin       | Accuracy | Precision | Recall | F1-score|
| ------------- |:-------------:| --------:| ---------:| ------:| -------:|
| 300           | ReLU          | 0.54     | 0.29      | 0.54   | 0.38    |
| 300           | Tanh          | 0.42     | 0.4       | 0.42   | 0.35    |
| 300           |     -         | 0.48     | 0.51      | 0.48   | 0.48    |
| 50            | ReLU          | 0.46     | 0.48      | 0.46   | 0.35    |
| 50            | Tanh          | 0.54     | 0.58      | 0.54   | 0.50    |
| 50            |     -         | 0.56     | 0.54      | 0.56   | 0.53    |
| 10            | ReLU          | 0.6      | 0.62      | 0.60   | 0.53    |
| 10            | Tanh          | 0.56     | 0.54      | 0.56   | 0.53    |
| 10            |     -         | 0.56     | 0.56      | 0.56   | 0.52    |

I tried 3 sizes for the hidden layer each with ReLU, Tanh, and without nonlinearity. The results all are relatively similar, but overall performance seems to rise with the smaller sizes of the hidden layer. There is not clear pattern regarding the use of nonlinearities.

## Part Bonus
In a3_model_bonus I added the command line argument outputfile, which takes the path of the png-file with the graph. I tried to plot a precision-recall curve, but while my code seems to make sense to me, the resulting graph looks strange. I upload the one from the call python3 a3_model_bonus.py out.csv --hidden 10 precision-recall-curve.png. 
