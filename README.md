# Deep-Learning-Project-Cracking-the-genetic-code-

The big picture of this DL project is described in the submitted paper entitled 'Cracking the genetic code by neural networks'.

To have a direct visual grasp of what this Deep Learning project does, please play the .gif videos in the FIGURES library. You will see how the genetic code is learned by the different neural network architectures and how well and how fast the genetic code is cracked and which accuracies are achieved depending on the settings.

DATA:

In the data library there are 3 files:
-  two datasets: 'humanORFs.txt' and 'humanProteins.txt';
-  a pdf screenshot of the two files open in a notepad.

The size of the ORFs dataset is 97 MB. It is a tab delimited formatted .txt file with two columns. 
Column 1 is a unique cdsID, Column 2 is the mRNA open reading frame with nucleotides coded as A, U, C, G.

The size of proteins dataset is 33 MB. It is a tab delimited formatted .txt file with two columns.
Column 1 is a unique cdsID, Column 2 is the protein sequence with amino acid coded with the standard one letter code.

The number of rows in both files are exactly the same. The two files can be mapped row by row with their common key i.e., the unique cdsID.
There are 69,768 transcripts and their corresponding protein sequences.

PYTHON SCRIPTS AND PROGRAMS:

The python script 'tokenizerOHE.py' entails functions to load file data and to convert the characters of an alphabet to tensors. These will be used as input and target for the neural network architecture. This script also inludes a function designed to build heatmaps of size 64 x 21 (64 codons by 21 amino acid + stop) of the current dictionnary matrix mapping the codons to the target amino acids. The columns in this matrix contains the vector of 21 softmax probabilities (likelihood that a given codon maps a given amino acid or stop mark).

The python script 'mlpCrackCodon.py' implements the dataloader to build a training dataset and a test dataset with batches of size 64 of codons tokenized as tensor of size 1 x 64. It also implements a MLP, Multilayer Perceptron (feed forward fully connected neural network) that will be trained and tested through 40 iterations training and test loops on the 2 datasets.

The python script 'mlpCrackCodon12bits.py' deals with batches of size 64 codons that will be tokenized as tensors of size 3 x 4 (one hot encoding) = 12 bits.

The python script 'mlpCodonEmbeddings64to10.py' entails a codon embeddings layer of dimensionality set to 10 (or 2) and entails fully connected layers with two hyperbolic tangent activation functions. The script also includes the code to plot the embeddings weight features in 2D.

The python script 'MyRNNCrackCodon.py' entails the RNN or GRU or LSTM architectures. The settings are the ones for the LSTM but commenting and uncommenting lines in the RNN class allows to change the LSTM to a RNN or a GRU. All these architecures are built with 2 hidden layers.

The python script 'DLplot.py' can be used to plot the scoring metrics (accuracy and loss function both for training and testing) of different previously saved scenario in the very same graph.
"# deep-learning-project-cracking-the-genetic-code" 
"# deep-learning-project-cracking-the-genetic-code" 
