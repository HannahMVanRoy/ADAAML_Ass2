import pandas as pd
import numpy as np
from TreeNode import TreeNode

class TreeID3:
  def __init__(self):
    self.root = None

  def fit(self, training_data, target):
    self.root = TreeNode(training_data, target)
    self.root.make()

  def make_decision(self, testing, targets):
    accurate = 0 
    for index in testing.index:
      current_node = self.root
      while current_node.decision is None:
        
          attribute_to_test = current_node.split_attribute
          attribute_value = testing[attribute_to_test][index]

          print("Testing ", attribute_to_test, "->", attribute_value)
          current_node = current_node.children[attribute_value]

      print("Decision:", current_node.decision)
      #if found value is the same as the actual value
      if current_node.decision == targets.get(key=index):
        accurate+=1
    return accurate/len(targets)

  #def make_decision(self, sample):
   # current_node = self.root
   # while current_node.decision is None:
   #     attribute_to_test = current_node.split_attribute
   #     attribute_value = sample[attribute_to_test]
   #     print("Testing ", attribute_to_test, "->", attribute_value)
     #   values = current_node.children.items()
    #    current_node = values[attribute_value]
   # print("Decision:", current_node.decision)
   # return current_node.decision