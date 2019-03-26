# -----------------------------------------------------
# Authors: KrishnaSandeep Gaddam, Chandana Dasari
# Description: script to extract samples from mturk dataset with the same movie review id present in the gold sample
# -----------------------------------------------------

import pandas as pd
# loading the data , and extracting movie review id
gold_sample = pd.read_csv('../data/gold_sample.csv')['id']
mturk = pd.read_csv('../data/mturk.csv')
# extracting the samples which has same movie reviews present in the gold_sample.csv
mturk_sample = mturk[mturk['id'].isin(gold_sample)]
# Write the extracted samples to mturk_sample.csv
mturk_sample.to_csv('../data/mturk_sample.csv', index=False)