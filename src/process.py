import pandas as pd
import numpy as np
import nltk
    

def merge_import_from_xlsx(start_yr: int, end_yr: int, path_blueprint: str):
    """ Merges and imports yearly data as a pandas dataframe.

    Parameters
    ----------
    start_yr: int,
              First year of available data.
    end_yr: int, 
            Last year of available data.
    path_blueprint: str,
                    Path of excel sheets where year is replaced with '{}'.

    Returns
    -------
    raw_data: DataFrame.
        Merged and imported data.
    
    """

    for yr in range(start_yr, end_yr + 1):
        if yr == start_yr:
            raw_data = pd.read_excel(path_blueprint.format(yr))
        else:
            raw_data = raw_data.append(pd.read_excel(path_blueprint.format(yr)))
    
    raw_data = raw_data.reindex()
    return raw_data

def select_clean_data(raw_data, min_instances):
    """ Cleans and selects relevant data for classification task
    
    Parameters
    ----------
    raw_data: DataFrame,
              Raw data as imported from excel sheets.
    min_instances: int,
                   Number of minimum instances required to consider class.
    
    Returns
    -------
    data: DataFrame.
                   Cleaned data used for classification task.
                  
    """
    data = raw_data[['Politikbereich', 'Zweck']].dropna(axis = 0)
    data = data.drop_duplicates(subset = 'Zweck')
    
    cat_dist = data.groupby('Politikbereich').count().sort_values(by = ['Zweck'], axis = 0, ascending  = False)
    cat_considered = cat_dist.index[cat_dist['Zweck'] > min_instances]
    data = data.loc[data['Politikbereich'].isin(cat_considered)]
    
    return data


def stratified_train_test_split(data, test_ratio = .3, random_state=0):
    """ Splits dataset into train and test via stratified sampling
    
    Parameters
    ----------
    data: DataFrame,
          All available datapoints.
    test_ratio: float (default: .2),
                Ratio of dataset to use for testing.       
    random_state: int (default: 0),
                  Seed for random number generator.
    
    Returns
    -------
    train_data, test_data: Tuple of DataFrames,
                           Datapoints for training, Datapoints for testing.
    
    """
    
    categories = data['Politikbereich'].unique()
    
    train_data = data.iloc[:0,:].copy()
    test_data = data.iloc[:0,:].copy()

    for category in categories:    
        shuffled_data = data[data['Politikbereich'] == category].sample(frac = 1, random_state = random_state)
        cutoff_index = int(shuffled_data.shape[0]*test_ratio)
        test_data = test_data.append(shuffled_data.iloc[:cutoff_index, :])
        train_data = train_data.append(shuffled_data.iloc[cutoff_index:, :])

    train_data = train_data.sample(frac = 1, random_state = random_state)
    test_data = test_data.sample(frac = 1, random_state = random_state)
    
    return train_data, test_data

def tokenize(data):
    """Tokenizes 'Zweck' sentence into list of words using the nltk library
    
    Parameters
    ----------
    data: DataFrame,
          Cleaned classification DataFrame with original 'Zweck' column.
    
    Returns
    -------
    tokenized_data: DataFrame,
                    Cleaned classification DataFrame with tokenized 'Zweck' column.
                    
    """
    
    #leave original dataframe untouched
    tokenized_data = data.copy()
    
    tokenized_data['Zweck'] = tokenized_data['Zweck'].str.lower()
    #remove special characters
    tokenized_data['Zweck'] = tokenized_data['Zweck'].str.replace('[^a-zA-ZÀ-ʸ ]', ' ', regex = True)
    #tokenize
    tokenized_data['Zweck'] = tokenized_data['Zweck'].apply(lambda row: nltk.word_tokenize(row, language = 'german'))
    #remove stopwords and one character words
    stopwords = nltk.corpus.stopwords.words('german') + nltk.corpus.stopwords.words('english')
    tokenized_data['Zweck'] = tokenized_data['Zweck'].apply(lambda row : [x for x in row if x not in stopwords and len(x)>1])
   
                                                  
    return tokenized_data


def create_sentence_vector(data, lookup_table, tokenized = False):
    """Turn raw data into sentence vector needed as prediction model input based on vocabulary in lookup_table.
    
    Parameters
    ----------
    data: DataFrame,
          Data with 'Zweck' column.
    lookup_table: DataFrame,
                  Table to look up the word2vector mapping of the corpus vocabulary.
    tokenized: bool (default: False),
               If True, 'Zweck' column is assumed to have already been tokenized.
    
    
    Returns
    -------
    transformed_data: DataFrame,
                Transformed data ready to be used as model input.
    
    """
    
    #leave original dataframe untouched
    transformed_data = data.copy()
    
    if not tokenized:
        transformed_data = tokenize(transformed_data)
    
    vocab2index = dict(zip(lookup_table['word'], lookup_table.index))
    
    transformed_data['vectorized_sentence'] = transformed_data['Zweck'].apply(lambda row: [vocab2index[x] if x in vocab2index.keys() else 0 for x in row])
    sentence_length = transformed_data['vectorized_sentence'].apply(lambda sentence: len(sentence))
    max_length = sentence_length.max()
    transformed_data['padded_sentence'] = transformed_data['vectorized_sentence'].apply(lambda sentence: sentence + (max_length -len(sentence))*[0])
    
    return transformed_data

def create_mean_model_data(data, lookup_table, tokenized = False, transformed = False):
    """Transform sentence to mean of word embeddings based on lookup_table.
    
    Parameters
    ----------
    data: DataFrame,
          Data with 'Zweck' column.
    lookup_table: DataFrame,
                  Table to look up the word2vector mapping of the corpus vocabulary.
    tokenized: bool (default: False),
               If True, 'Zweck' column is assumed to have already been tokenized.
    transformed: bool (default: False),
                 If True, 'Zweck' column is assumed to have already beeen vectorized.
    
    Returns
    -------
    mean_model_data: DataFrame,
                     Data with mean word vector column.
    
    """
    
    mean_model_data = data.copy()
    
    if not transformed:
        mean_model_data = create_sentence_vector(mean_model_data, lookup_table, tokenized = tokenized)
    
    
    mean_model_data['mean_word_vector'] = mean_model_data['vectorized_sentence'].apply(lambda row: np.array(lookup_table.iloc[row]['vector'].tolist()).mean(axis = 0))
    mean_model_data = mean_model_data.dropna(axis = 0) #if none of the word embeddings were in the sentence
    
    return mean_model_data

def create_lstm_model_data(data, lookup_table, preembed = False,  tokenized = False, transformed = False):
    """Transform sentence to mean of word embeddings based on lookup_table.
    
    Parameters
    ----------
    data: DataFrame,
          Data with 'Zweck' column.
    lookup_table: DataFrame,
                  Table to look up the word2vector mapping of the corpus vocabulary.
    preembed: bool (default: False),
              If True, padded word index vectors (padded_sentence) will be transformed to list of word embeddings.
    tokenized: bool (default: False),
               If True, 'Zweck' column is assumed to have already been tokenized.
    transformed: bool (default: False),
                 If True, 'Zweck' column is assumed to have already beeen vectorized.
    
    Returns
    -------
    mean_model_data: DataFrame,
                     Data with mean word vector column.
    
    """
    
    lstm_model_data = data.copy()
    
    if not transformed:
        lstm_model_data = create_sentence_vector(lstm_model_data, lookup_table, tokenized = tokenized)
    
    if preembed:
        lstm_model_data['word_vectors'] = lstm_model_data['padded_sentence'].apply(lambda row: lookup_table.iloc[row]['vector'].tolist())
    
    return lstm_model_data
