# DASS-21-diagnostic-sytem
This project is the code which was produced as part of the ECM3401 individual literature review and project module. 
# Introduction 

The purpose of this project was to create a depression and anxiety diagnostic system for authors of text using the ERT dataset and the DASS-21 scale.
This purpose was achieved by converting an individuals text into emotions, vectorising the emotion to match emotions of the ERT dataset, 
then passing the vector through two multilayer perceptron regression models for depression and anxiety respectively. 

### Prerequisites
To be able to run this program effectively the following prerequisites are needed:
- An installation of python 3.9
- The training_data.txt file 
- The frequencies.json file 
- The testdoc.txt file or any other txt file describing how an individual feels 

### Installation
This module relies on the following module dependencies:
- matplotlib
- numpy
- pandas
- seaborn 
- spacy 
- nltk (natural language toolkit)
- sklearn (scikit-learn) 


Here is how you install each of the dependencies above:
```sh
$ pip3 install matplotlib 
```
```sh
$ pip3 install numpy 
```
```sh
$ pip3 install pandas
```
```sh
$ pip3 install seaborn
```

```sh
$ pip3 install nltk
```

```sh
$ pip3 install scikit-learn
```
### Getting started 
To get started with this module you firstly need to run the main.py program with all the prerequisites above located in the same file as main.py . 
These prequisites are essential to the running of the program as it speeds up the diagnostic process by using training data which has already been
converted into vectors, target output DASS-21 scores and corpus frequencies needed to train the regression models, and therefore not having to run the 
computaionally expensive process of converting all documents in the corpus each time the program is run. Once the user has run the main.py program, 
the user will be presented with a basic graphical user interface consisting of three options each of which is run by entering the corresponding number into
the prompt.The first option "validate" tells the user what percentage of the documents labelled as having an author with depression and having an author 
without depression had DASS-21 scores for depression and anxiety equivalent or above the severitiy of modarate. This option then uses the number given by the
user (in a prompt before the threshold percentages are displayed) to train the depression and anxiety regression models over the course of one or multiple runs
(given by the number entered) to produce a line graph showing how the R squared value changes in both the depression and anxiety regression model over multiple 
runs.The second option "diagnose" asks the user (through a prompt) to enter the path (or name if the text file is in the same folder as main.py) of a text file to be diagnosed
the system then procceds to report the predicted depression and anxiety diagnosis of the individual who wrote the text file given to the program. 
The final option "exit" exits the program in a clean way to make sure there is no data loss. This is useful for individuals who accidently ran the program.

### Details 
Author: Pierre Siddall. 
Written in: Python 3.9. 
Released on: 03/05/2023. 
licence: CC BY-NC. 
