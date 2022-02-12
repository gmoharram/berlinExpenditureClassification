# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 23:54:03 2022

@author: gmoha
"""

import tensorflow as tf
import numpy as np
import pandas as pd
from collections import Counter

import sklearn.tree as tree
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import f1_score, multilabel_confusion_matrix, recall_score

def top_words(processed_data, k, category = None, return_overlap_ratio = False):
    """Takes top k words for every class to include in bag-of-words underlying set
    
    Parameters
    ----------
    processed_data: DataFrame,
                    Processed training data with processed 'Zweck' column as list of words.
    k: int,
       Number of top words to include for each class.
    category: str (default: None),
              Category to investigate, defaults to combining the top words of each category.
    return_overlap_ratio: bool (default: False),
                          Returns ratio of words in top k of multiple classes if True.
                          
    
    Returns
    -------
    bag_set: set,
             Underlying set to use for bag-of-words sentence embedding.
    overlap_ratio: float (optional),
                   Ratio of words in top k of multiple classes.
                
    """
    if category == None:
        categories = processed_data['Politikbereich'].unique()
    else:
        categories = [category]
     
    top_words = []
    for cat in categories :
        cat_words = processed_data[processed_data['Politikbereich'] == cat]['Zweck'].sum()
        cat_words_dist = Counter(cat_words)
        cat_words_dist = pd.DataFrame(sorted(cat_words_dist.items(), key=lambda x: x[1], reverse=True), columns = ['Wort','Anzahl'])
        top_words.extend(cat_words_dist['Wort'][:k])
    top_words = set(top_words)
    
    if return_overlap_ratio:
        overlap_ratio = 1 - len(top_words)/(len(categories)*k)
        
        return top_words, overlap_ratio
    
    else:
        
        return top_words
    

def create_w2v_lookup(tokenized_data, window_size = 2, negative_samples = 5, embedding_dim = 128):
    """Take a corpus of documents and create a word2vector lookup table.
    
    Parameters
    ----------
    tokenized_data: DataFrame,
                    Processed training data with processed 'Zweck' column.
    window_size: int (default: 2),
                 Number of words to consider similar to the left and right of the target word.
    negative_samples: int (default: 5),
                 Number of negative context words to sample.
    
    embedding_dim: int (default: 128),
                   Dimension of the real space the words will be embedded in.
    
    Returns
    -------
    lookup_table: DataFrame,
                  Table to look up the word2vector mapping of the corpus vocabulary.
                  
    """
    
    #don't modify input data
    data = tokenized_data.copy()
    
    #Create index to word mapping
    vocab = top_words(data, None)
    vocab = list(vocab)
    vocab.insert(0, '<pad>')
    indices = list(range(0, len(vocab)))
    vocab2index = dict(zip(vocab, indices))
    
    vocab_size = len(vocab)
    
    
    #Create sentence vector of word indices and add padding for uniform sentence lengths
    data['vectorized_sentence'] = data['Zweck'].apply(lambda row: [vocab2index[x] for x in row])
    max_length = data['vectorized_sentence'].apply(lambda sentence: len(sentence)).max()
    data['padded_sentence'] = data['vectorized_sentence'].apply(lambda sentence: sentence + (max_length -len(sentence))*[0])
    
    
    #function to generate negative sampling skip gram training data
    def generate_targets_contexts_labels(skip_gram_pairs, window_size, num_ns, vocab_size, random_state = 0):

        targets, contexts, labels = [], [], []
        for target_word, context_word in skip_gram_pairs:
            context_class = tf.expand_dims( tf.constant([context_word], dtype="int64"), 1)
            negative_sampling_candidates = tf.random.log_uniform_candidate_sampler(true_classes=context_class,
                                                                                 num_true=1,
                                                                                 num_sampled=num_ns,
                                                                                 unique=True,
                                                                                 range_max=vocab_size,
                                                                                 seed=random_state,
                                                                                 name="negative_sampling")[0]

            # Build context and label vectors (for one target word)
            negative_sampling_candidates = tf.expand_dims(negative_sampling_candidates, 1)
            context = tf.concat([context_class, negative_sampling_candidates], 0)
            label = tf.constant([1] + [0]*num_ns, dtype="int64")
    
            # Append each element from the training example to global lists.
            targets.append(target_word)
            contexts.append(context)
            labels.append(label)

        return targets, contexts, labels

    #Create positive skip gram training data pairs
    
    # Build the sampling table for vocab with `vocab_size` words
    sampling_table = tf.keras.preprocessing.sequence.make_sampling_table(vocab_size)
    skip_gram_pairs = data['padded_sentence'].apply(lambda sentence: tf.keras.preprocessing.sequence.skipgrams(sentence,vocabulary_size=vocab_size, sampling_table=sampling_table,
 window_size=window_size, negative_samples=0)[0])
    skip_gram_pairs = [pair for sublist in skip_gram_pairs.tolist() for pair in sublist]
    
    #Create training data with negative samples
    targets, contexts, labels = generate_targets_contexts_labels(skip_gram_pairs, window_size = window_size, num_ns = negative_samples, vocab_size = vocab_size, random_state = 0)
    
    targets = np.array(targets)
    contexts = np.array(contexts)[:,:,0]
    labels = np.array(labels)
    
    BATCH_SIZE = 1024
    BUFFER_SIZE = targets.size
    dataset = tf.data.Dataset.from_tensor_slices(((targets, contexts), labels))
    dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
    
    class Word2Vec(tf.keras.Model):
        def __init__(self, vocab_size, num_ns, embedding_dim):
            super(Word2Vec, self).__init__()
            self.target_embedding = tf.keras.layers.Embedding(vocab_size,
                                  embedding_dim,
                                  input_length=1,
                                  name="w2v_embedding")
            self.context_embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim,input_length=num_ns+1)

        def call(self, pair):
            target, context = pair
            
            if len(target.shape) == 2:
                target = tf.squeeze(target, axis=1)
            word_emb = self.target_embedding(target)
            context_emb = self.context_embedding(context)
            dots = tf.einsum('be,bce->bc', word_emb, context_emb)
            return dots
    
    word2vec = Word2Vec(vocab_size, negative_samples,  embedding_dim)
    word2vec.compile(optimizer='adam',
             loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
             metrics=['accuracy'])
    
    
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="logs")
    word2vec.fit(dataset, epochs=20, callbacks=[tensorboard_callback])
    
    weights = word2vec.get_layer('w2v_embedding').get_weights()[0]
    lookup_table = pd.DataFrame(vocab, columns = ['word'])
    lookup_table['vector'] = weights.tolist()
    
    return lookup_table

def mean_word_model(mean_model_data, test = True, dim_reduction = False):
    """Classify sentence based on mean word embedding with possible dimensionality reduction.
    
    Parameters
    ----------
    mean_model_data: DataFrame,
                     Data with mean word vector of 'Zweck' column.
    test: bool (default: True),
               if True, 20% of the data will be used for testing.
    dim_reduction: bool (default: False),
                   if True, dimensionality reduction on word embeddings is performed before classification.
    
    
    Returns
    -------
    clf: sklearn.tree,
         trained decision tree classification model.
    performance: DataFrame (optional: if "test"),
                 dataframe with classifier performance metrics.

    
    """
    
    X = np.array(mean_model_data['mean_word_vector'].tolist())
    y = mean_model_data['Politikbereich']
    
    if test:
        #train, validation split is performed
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
        for train_index, test_index in sss.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            
        
        #dimensionality reduction while keeping around 95% of the "energy"
        if dim_reduction:
            d = X.shape[1] #original dimension
            energy = 100
            while energy > 95:
                d -= 1
                S = np.cov(X_train.T)
                L,U = np.linalg.eig(S)
                ind = np.argpartition(L, -d)[-d:]
                energy = sum(L[ind])/sum(L)
            X_train = np.matmul(X_train, U[:, ind])
            X_test = np.matmul(X_test, U[:, ind])
    
        #Train and test classifier
        clf = tree.DecisionTreeClassifier(random_state=0, class_weight = 'balanced')
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        
        performance = pd.DataFrame(f1_score(y_test, y_pred, average = None, labels = y_test.unique()), index = y_test.unique(), columns =  ['f1_score'])
        performance['confusion_matrix_TN_FP_FN_TP'] = multilabel_confusion_matrix(y_test, y_pred, labels = y_test.unique()).tolist()
        performance['recall_score'] = recall_score(y_test, y_pred, average = None, labels = y_test.unique())
        
        return clf, performance
    
    else:
        
        #dimensionality reduction while keeping around 95% of the "energy"
        if dim_reduction:
            d = X.shape[1] #original dimension
            energy = 100
            while energy > 95:
                d -= 1
                S = np.cov(X.T)
                L,U = np.linalg.eig(S)
                ind = np.argpartition(L, -d)[-d:]
                energy = sum(L[ind])/sum(L)
            X = np.matmul(X, U[:, ind])
        
        clf = tree.DecisionTreeClassifier(random_state=0, class_weight = 'balanced')
        clf.fit(X, y)
        
        return clf
        
    
    