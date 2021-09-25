import pandas as pd
import numpy as np
from TreeNode import TreeNode

class TreeID3:
  def __init__(self):
    self.root = None

  def fit(self, samples, target):
    self.root = TreeNode(samples, target)
    self.root.make()