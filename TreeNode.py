import pandas as pd
import numpy as np
from statistics import mode

class TreeNode:
  def __init__(self, training_data, target):
    self.decision = None
    self.training_data = training_data
    self.target = target
    self.split_attribute = None
    self.children = None

  #function to create the tree
  def make(self):
    training_data = self.training_data
    target = self.target
    #if there are no more attributes, set the decision to the most 
    #common target value
    if len(training_data) == 0:
      self.decision = "pass"
      return
    #if there is only one unique target value left, 
    #then all rows' target value = that target value
    elif len(training_data[target].unique()) == 1:
      self.decision = training_data[target].unique()[0]

      return
    #if data is still unclassified begin 
    #parent node -> children node traversal
    else:
      ig_max = 0
      #loop through the attribute labels excluding the target label 
      #(because it cannot be used for prediction, because its the target)
      for attr in training_data.keys():
        if attr == target:
          continue
        #compute information gain for the given attribute
        attr_ig = compute_information_gain(training_data, attr, target)
        #if the given attribute information gain is greater than 
        #the maximum information gain
        if attr_ig > ig_max:
          #set max to given attribute's information gain
          ig_max = attr_ig
          #make given attribute a node
          self.split_attribute = attr
      #for each node, print attribute and associated information gain
      print(f"Split by {self.split_attribute}, IG: {ig_max:.2f}")
      #set children to empty
      self.children = {}
      print(self.split_attribute)
      #loop through unique values for the node's attribute 
      for value in training_data[self.split_attribute].unique():
        #cool array comparison to check whether the given value is equal 
        #to each value for the node's attribute
        index = training_data[self.split_attribute] == value
        #perform tree creation algorithm for children (wherein the child 
        #of this parent node becomes the new parent, etc)
        self.children[value] = TreeNode(training_data[index], target)
        self.children[value].make()

#function to handle entropy calculation
def compute_entropy(sub):
  #if there is only one value, then the entropy is 0
  if len(sub) < 2:
    return 0
  #transform series into a usable array of the frequencies of each value
  freq = np.array(sub.value_counts(normalize=True))
  #perform entropy calculation and return result (addition of 1e-6 for rounding)
  return -(freq * np.log2(freq + 1e-6)).sum()

#function to handle information gain calculation
def compute_information_gain(training_data, attr, target):
  #calculation of the percentage occurance (frequency) of each value
  #for a given attribute
  values = training_data[attr].value_counts(normalize=True)
  split_ent = 0
  #loop through values and their corresponding frequencies
  for value, freq in values.iteritems():
    sub_ent = compute_entropy(
      #find target value for the rows where the values within the given 
      # attribute are equal to the given value from the loop
      #then compute entropy on that data
      training_data[training_data[attr]==value][target]
    )
    #increase overall 'entropy' by entropy for that value scaled 
    #by its frequency
    split_ent += freq * sub_ent
  #calculate entropy of the target value without any splits
  ent = compute_entropy(training_data[target])
  #compare entropy after division, with the entropy of target
  return ent - split_ent
