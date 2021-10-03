import pandas as pd
import numpy as np
from TreeNode import TreeNode
from TreeID3 import TreeID3
from statistics import mode

df = pd.read_csv('mushrooms.csv')

train=df.sample(frac=0.7,random_state=42)
df = pd.read_csv('mushrooms.csv')

test=df.drop(train.index) #remove the training data, leaving the testing data

targets = test["class"]
test = test.drop(columns="veil-type") #attribute is entirely homogenous
test = test.drop(columns="class")

t = TreeID3()

t.fit(train, "class")

print("Accuracy: ", t.make_decision(test, targets))
