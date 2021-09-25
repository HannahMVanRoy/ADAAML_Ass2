import pandas as pd
import numpy as np

class TreeNode:
  def __init__(self, samples, target):
    self.decision = None
    self.samples = samples
    self.target = target
    self.split_attribute = None
    self.children = None

  def make(self):
    samples = self.samples
    target = self.target
    if len(samples) == 0:
      self.decision = "Yes"
      return
    elif len(samples[target].unique()) == 1:
      self.decision = samples[target].unique()[0]
      return
    else:
      ig_max = 0
      for a in samples.keys():
        if a == target:
          continue
        aig = compute_info_gain(samples, a, target)
        if aig > ig_max:
          ig_max = aig
          self.split_attribute = a
      print(f"Split by {self.split_attribute}, IG: {ig_max:.2f}")
      self.children = {}
      for v in samples[self.split_attribute].unique():
        #print("v: ", v)
        index = samples[self.split_attribute] == v
        #print("samples:", samples[self.split_attribute])
        #print("samplesIndex:", samples[self.split_attribute] == v)
        #print("index: ", index)
        self.children[v] = TreeNode(samples[index], target)
        self.children[v].make()

  def pretty_print(self, prefix=''):
      if self.split_attribute is not None:
          for k, v in self.children.items():
              v.pretty_print(f"{prefix}:When {self.split_attribute} is {k}")
              #v.pretty_print(f"{prefix}:{k}:")
      else:
          print(f"{prefix}:{self.decision}")

def compute_entropy(y):
  if len(y) < 2:
    return 0
  freq = np.array(y.value_counts(normalize=True))
  return -(freq * np.log2(freq + 1e-6)).sum()

def compute_info_gain(samples, attr, target):
  values = samples[attr].value_counts(normalize=True)
  split_ent = 0
  for v, fr in values.iteritems():
    sub_ent = compute_entropy(
      samples[samples[attr]==v][target]
    )
    split_ent += fr * sub_ent
  ent = compute_entropy(samples[target])
  return ent - split_ent
