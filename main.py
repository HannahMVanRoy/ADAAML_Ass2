import pandas as pd
import numpy as np
from TreeNode import TreeNode
from TreeID3 import TreeID3

df = pd.read_csv('shuttle-landing-control.csv')

t = TreeID3()
t.fit(df, "G")
t.root.pretty_print()



# print(df["variety"].value_counts(normalize=True))
