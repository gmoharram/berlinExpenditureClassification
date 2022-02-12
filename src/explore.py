# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 23:33:34 2022

@author: gmoha
"""

import pandas as pd
from collections import Counter

def class_frequency(data, column = 'Politikbereich'):
    """Plots the occurence frequency of each value in a dataframe column.
    
    Parameters
    ----------
    data: DataFrame,
          DataFrame with column where unique values are counted.
    column: str (default: 'Politikbereich'),
            column name where unique values are counted.
    
    Returns
    -------
    frequency_plot: AxesSubplot,
                    Plot of unique value frequencies.
    
    """
    
    cat_dist = data.groupby(column).count().sort_values(by = ['Zweck'], axis = 0, ascending  = False)
    return cat_dist.plot.bar(figsize = (12, 4), fontsize = 14)


def word_frequency(tokenized_data, category):
    """Plots the occurence frequency of tokenized words in 'Zweck' for a given 'Politikbereich' value.
    
    Parameters
    ----------
    tokenized_data: DataFrame,
                    DataFrame where 'Zweck' column has been tokenized.
    category: str,
              'Politikbereich' value for which word frequency is counted.
              
    
    Returns
    -------
    frequency_plot: AxesSubplot,
                    Plot of unique word frequencies.
    
    """    
   
    cat_words = tokenized_data[tokenized_data['Politikbereich'] == category]['Zweck'].sum()
    cat_word_dist = Counter(cat_words)
    cat_word_dist = pd.DataFrame(sorted(cat_word_dist.items(), key=lambda x: x[1], reverse=True), columns = ['Wort','Anzahl'])
    return cat_word_dist[:40].plot.bar(x = 'Wort', y = 'Anzahl', title = 'Wortfrequenz im Politikbereich "{}"'.format(category),figsize = (12,4), fontsize = 14)