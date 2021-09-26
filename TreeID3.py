import pandas as pd
import numpy as np
from TreeNode import TreeNode

class TreeID3:
  def __init__(self):
    self.root = None

  def fit(self, training_data, target):
    self.root = TreeNode(training_data, target)
    self.root.make()