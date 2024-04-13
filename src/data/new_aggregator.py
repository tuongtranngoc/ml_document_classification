from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import pandas as pd
import os

from src.config.load_config import configuration

config = configuration('new_aggregator')

# 1. Read dataset
data = pd.read_csv(os.path.join(config['Dataset']['data_path'], 'newsCorpora.csv'), sep='\t', 
                   names=['id', 'title', 'url', 'publisher', 'category', 'story', 'hostname', 'timestamp'])

# 2. Select publishers: "Reuters", "Huffington Post", "Businessweek", "Contactmusic.com", "Daily Mail"
publisher = ['Reuters', 'Huffington Post', 'Bussinessweek', 'Contactmusic.com', 'Daily Mail']
data = data[data.publisher.isin(publisher)]

# 3. Shuffle
data = data.sample(frac=1)


# 4. Train, valid, test splitting
train = data.sample(frac=0.8, random_state=200)
valid_test = data.drop(train.index)

valid = valid_test.sample(frac=0.5, random_state=200)
test = valid_test.drop(valid.index)

# Save to txt file
cols = ['title','category']
train[cols].reset_index(drop=True).to_csv(os.path.join(config['Dataset']['data_path'], 'train.txt'), sep='\t')
valid[cols].reset_index(drop=True).to_csv(os.path.join(config['Dataset']['data_path'], 'valid.txt'), sep='\t')
test[cols].reset_index(drop=True).to_csv(os.path.join(config['Dataset']['data_path'], 'test.txt'), sep='\t')