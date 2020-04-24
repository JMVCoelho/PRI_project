to run:

python exercise-4.py

Dataset_preprocessed is the DUC-2001 dataset, preprocessed.
Goldenset_test are the keyphrases for the test set.
Goldenset_train are the keyphrases for the train set.
Test_set_preprocessed is a folder with all the documents in the test set, preprocessed.
Train_set_preprocessed is a folder with all the documents in the train set, preprocessed.


Running will create some files/folders:

- Keyphrases_experimental: Folder with the extracted keyphrases, one for each document.

- Metric: Folder with the individual metric results for each document 


Running will print in stdout the average metric results for all files.

Note: this was tested and coded on a Windows machine. Unix may have errors with Paths. If that happens, please change "/" in the global variables to "\".