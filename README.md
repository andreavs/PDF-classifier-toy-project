# PDF classifier toy project

In this repo you will find some test code for a document classifier task. This code was produced in a few hours and should not in any circumstance be used in any live product 

## structure of this repo 
To tour this repo, it makes the most sense to do it in the following order: 
1. `eda.py` - here you will find some exploritory data analysis of the pdfs. The code will produce a few figures when run 
2. `model_training.py` - here we use a simple decision tree model to train a classifier on the data. When run, the code will store the classifier and the label encoder 
3. `classifier.py` - Here the classifier packaged and used to classify new documents 
4. `just_ask_chatgpt.py` Just for fun, we here explore the approach of just sending the first page of the pdf to chatgpt. 

## Running the code 
the requirements needed to run the code are stored in the `requirements.txt` file 

