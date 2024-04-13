from __future__ import division
from __future__ import print_function
from __future__ import absolute_import


import os 
import glob
import matplotlib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from src.config.load_config import configuration
config = configuration('new_aggregator')

sns.set_style("whitegrid")
matplotlib.rcParams['figure.figsize'] = 15, 8


word_eda = []
category_eda = []

for type_data in ['train', 'valid', 'test']:
    # Count sample
    data_path = os.path.join(config['Dataset']['data_path'], type_data + '.txt')
    data = pd.read_csv(data_path, sep='\t')
    eda_df = data.category.value_counts().reset_index(name=type_data)
    category_eda.append(eda_df)

    # Number of average word, maximum word, minimum word
    data['num_words'] = data.title.map(lambda x: len(x.split()))
    word_eda.append(
        pd.DataFrame({'operator': ['mean', 'max', 'min'], type_data: [data.num_words.mean(), data.num_words.max(), data.num_words.min()]})
    )

category_eda = pd.concat([df.set_index('category') for df in category_eda], axis=1).reset_index()
word_eda = pd.concat([df.set_index('operator') for df in word_eda], axis=1).reset_index()

plt.xlabel(xlabel='category',rotation=0)
os.makedirs(config['Dataset']['eda'], exist_ok=True)
category_eda.plot(x='category', y=['train', 'valid', 'test'], kind='bar')
plt.savefig(os.path.join(config['Dataset']['eda'], 'category_eda.png'))

word_eda.plot(x='operator', y=['train', 'valid', 'test'], kind='bar')
plt.savefig(os.path.join(config['Dataset']['eda'], 'word_eda.png'))
