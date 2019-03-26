# -----------------------------------------------------
# Authors: KrishnaSandeep Gaddam, Chandana Dasari
# Description: script to extract a random sample of 1000 rows from gold-standard dataset
# -----------------------------------------------------

import pandas as pd

pd.read_csv('../data/gold.csv').sample(1000).to_csv('../data/gold_sample.csv', index=False)