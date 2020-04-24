to run:

python exercise-3.py

Dataset_orinal is the DUC-2001 dataset, in its original form (XML).
test.reader.json is the golden set of keyphrases, in its original form (JSON)



Running will create some files/folders:

- Dataset_as_txt: folder the dataset files in txt instead of xml. (line 390 will be commented as this will be provided)

- Dataset_preprocessed: preprocessing the files in the folder above; (line 391 will be commented as this will be provided)

- Keyphrases_golden_set: parsing the json file to multiple files, one for each document. (line 392 will be commented as this will be provided)

- Keyphrases_experimental: Folder with the extracted keyphrases, one for each document.

- Metric: Folder with the individual metric results for each document 


Running will print in stdout the average metric results for all files.

Note: this was tested and coded on a Windows machine. Unix may have errors with Paths. If that happens, please change "/" in the global variables to "\".