# Homework 4 (Bounus 10\%)

## Submission instructions

* Due date and time: April 23 (Friday), 11:59 pm ET

* Carmen submission: Submit a .zip file named `name.number.zip` (e.g., `tabassum.13.zip`), which contains your full code and a instruction to run it. When running the code it should output the accuracy on the test data.
 
* Collaboration: You may discuss the homework with your classmates. However, you need to write your own solutions and submit them separately. In your submission, you need to list with whom you have discussed the homework. 


* Please write your own solutions from scratch. 

* Caution! Please do not import packages (like scikit learn or nltk).


# Dataset

* You will see a [`Data`](`HW4/Data/`) directory, which contains train data (`train/Positive.txt`, `train/Neutral.txt`, `train/Negative.txt`) and the test data (`test/Positive.txt`, `test/Neutral.txt`, `test/Negative.txt`). Note that this is the same data that you have used in HW2.



The directory structure of the [Data](./Data) folder is given below:

```
./Data/
├── test
│   ├── Negative.txt
│   ├── Neutral.txt
│   └── Positive.txt
└── train
    ├── Negative.txt
    ├── Neutral.txt
    └── Positive.txt

```


* The [train](./Data/train/) sub-folder contains the data for training your Naive Bayes model. 
	* There are 3098 total sentences in the train data. 
	* [Negative.txt](./Data/train/Negative.txt) file cotnains 893 tweets with Negative Sentiment
	* [Neutral.txt](./Data/train/Neutral.txt) file cotnains 1256 tweets with Neutral Sentiment
	* [Positive.txt](./Data/train/Positive.txt) file cotnains 949 tweets with Positive Sentiment


* The [test](./Data/test/) sub-folder contains the data for that we will used to test the performance of your Naive Bayes model. 
	* There are 775 total sentences in the test data. 
	* [Negative.txt](./Data/test/Negative.txt) file cotnains 224 tweets with Negative Sentiment
	* [Neutral.txt](./Data/test/Neutral.txt) file cotnains 314 tweets with Neutral Sentiment
	* [Positive.txt](./Data/test/Positive.txt) file cotnains 237 tweets with Positive Sentiment




# Logistic Regression Classilier (100 pts)

In this homework, you are to implement logisitc regression algorithm for tweet classifications. You will play with the Twitter Senitment Analysis data, where each tweet is tagged as with either Positive, Negative, or Neutral sentiment.





# What to submit:

* Your completed Logistic Regreesion Classifier code.
* A readme file with instruction to run your code.
* A plot which shows the decrease of error rate with each iteration (over the training data).


