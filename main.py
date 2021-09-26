import pandas as pd
import numpy as np
from TreeNode import TreeNode
from TreeID3 import TreeID3
from statistics import mode

df = pd.read_csv('shuttle-landing-control.csv')

#print(df.keys())

t = TreeID3()
#THIS IS YOUR TREE
t.fit(df, "G")
for value, child in t.root.children.items():
  print(t.root.split_attribute)
  print(child.split_attribute)
#print(t.root.children.items())
t.root.pretty_print()

# def predict(unknown_data, tree):
#   for attr in unknown_data.keys():
#     if attr in tree.keys():
#       try:
        


# def predict(data,tree,default = 1):
#   for key in list(data.keys()):
#     if key in list(tree.keys()):
#         #2.
#         try:
#             result = tree[key][query[key]] 
#         except:
#             return default

#         #3.
#         result = tree[key][data[key]]
#         #4.
#         if isinstance(result,dict):
#             return predict(data,result)

#         else:
#             return result

# print(df["variety"].value_counts(normalize=True))
