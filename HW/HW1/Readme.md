# CSE 5521 Homework 1: Decision Trees

In this assignment you will implement the ID3 decision tree learning algorithm and apply it to a dataset of poisonous and edible mushrooms. The dataset can be found in the [Data](./Data) folder, where the data is stored in a CSV format files:

      Data
      ├── test.csv
      └── train.csv

You will use the `train.csv` file for training your model and `test.csv` file for testing your model. 


Below is a snapshot of the data

      e,x,s,y,t,l,f,c,b,g,e,c,s,s,w,w,p,w,o,p,k,n,g
      e,f,s,n,f,n,a,c,b,o,e,?,s,s,o,o,p,n,o,p,b,v,l
      p,k,s,e,f,f,f,c,n,b,t,?,k,k,p,p,p,w,o,e,w,v,d
      e,f,f,g,f,n,f,w,b,k,t,e,s,f,w,w,p,w,o,e,k,s,g
      e,x,f,n,t,n,f,c,b,w,t,b,s,s,p,w,p,w,o,p,n,v,d
      e,f,y,n,t,l,f,c,b,w,e,r,s,y,w,w,p,w,o,p,k,s,p
      p,x,y,g,f,f,f,c,b,h,e,b,k,k,p,n,p,w,o,l,h,v,g
      p,f,s,w,t,n,f,c,b,w,e,b,s,s,w,w,p,w,t,p,r,v,m
      e,x,f,g,t,n,f,c,b,w,t,b,s,s,w,w,p,w,o,p,n,y,d
      ...

Here each row corresponds to a mushroom. The first column is the label indicating whether the mushroom is edible (e) or poisonous (p). (This is the output that we wish to predict from the other columns). 

Information about the meaning of the other columns is listed in the [Readme](./Data/Readme.md) file under the [Data](./Data) folder


## What to Turn In
Please turn in the following to Carmen by **02/17/2021 11:59 PM**:

- Your code with and clear instruction to run it
- A brief writeup (text file format) that includes the accuracy on test set as well as the calculation of the information gain for the **root** node of your final decision tree.


