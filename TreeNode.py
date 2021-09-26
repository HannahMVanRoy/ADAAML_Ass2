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
    #if there is no data, set the decision to the most common target value
    if len(training_data) == 0:
      self.decision = mode(training_data[target])
      return
    #if there is only one unique target value left, then all rows' target value = that target value (arrived at a leaf)
    elif len(training_data[target].unique()) == 1:
      self.decision = training_data[target].unique()[0]
      return
    #if data is still unclassified begin parent node -> children node traversal
    else:
      ig_max = 0
      #loop through the atrribute labels excluding the target label (because it cannot be used for prediction, because its the target)
      for attr in training_data.keys():
        if attr == target:
          continue
        #compute information gain for the given attribute
        attr_ig = compute_info_gain(training_data, attr, target)
        #if the given attribute information gain is greater than the maximum information gain
        if attr_ig > ig_max:
          #set max to given attribute's information gain
          ig_max = attr_ig
          #make given attribute a node
          self.split_attribute = attr
      #for each node, print attribute and associated information gain
      print(f"Split by {self.split_attribute}, IG: {ig_max:.2f}")
      #set children to empty
      self.children = {}
      #loop through unigue values for the node's attribute 
      for value in training_data[self.split_attribute].unique():
        #cool array comparison to check whether the given value is equal to each value for the node's attribute
        index = training_data[self.split_attribute] == value
        #perform tree creation algorithm for children (wherein the child of this parent node becomes the new parent, etc)
        self.children[value] = TreeNode(training_data[index], target)
        self.children[value].make()

  #function to handle tree printing
  def pretty_print(self, prefix=''):
      if self.split_attribute is not None:
          #loop through dictionary of children from make(), each child consists of a TreeNode object and an attribute value
          for value, child in self.children.items():
              #print functionality which treats the prior statement as a prefix for the next one, resulting in an ordered tree
              child.pretty_print(f"{prefix}:When {self.split_attribute} is {value}")
      else:
          print(f"{prefix}:{self.decision}")

#function to handle entropy calculation
def compute_entropy(y):
  if len(y) < 2:
    return 0
  #transform frequencies into usable array with attached values
  freq = np.array(y.value_counts(normalize=True))
  #perform entropy calculation and return result
  return -(freq * np.log2(freq + 1e-6)).sum()

#function to handle information gain calculation
def compute_info_gain(training_data, attr, target):
  #calculation of the percentage occurance (frequency) of each value for a given attribute
  values = training_data[attr].value_counts(normalize=True)
  split_ent = 0
  #loop through values and their corresponding frequencies
  for value, freq in values.iteritems():
    sub_ent = compute_entropy(
      #find target value for the rows where the values within the given attribute are equal to the given value from the loop
      #then compute entropy on that data
      training_data[training_data[attr]==value][target]
    )
    #increase overall 'entropy' by entropy for that value scaled by its frequency (to establish significance)
    split_ent += freq * sub_ent
  #calculate entropy when observing the target attribute (to get the target attribute, therefore should be perfect)
  ent = compute_entropy(training_data[target])
  #compare entropy calculated from observing each different attribute to determine target, with the entropy calculated from obersving target
  return ent - split_ent
