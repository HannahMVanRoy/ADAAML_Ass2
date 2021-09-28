import pandas as pd
import numpy as np
from TreeNode import TreeNode
from TreeID3 import TreeID3
from statistics import mode

df = pd.read_csv('shuttle-landing-control.csv')

t = TreeID3()

t.fit(df, "G")

#t.root.pretty_print()

targets = []
for tar_val in df["G"]:
  targets.append(tar_val)

print("Accuracy: ", t.make_decision(df, targets))