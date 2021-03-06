# SHORT STORY ENDING (Group-6)
## Author & Contributor List

```
Bandaru Aditya Siva Sasi Prasanth (14EE10008)

Chinmoy Samant (14EE10011)

Jithin Sukumar (14EE10022)

Mekala Rajashekhar Reddy (14EE10027)
```


## Files List:
```
sentiment.py: Discriminative model based on Sentiment Analysis

seq2seq_learning.py: Generative Model using Encoder-Decoder Network (consists of all variants viz. Teacher Forcing, Bidirectional, Reverse Input)

train_word_embed.py: Used to generate vector representation of words

word2vec.py: Discriminative Model which uses cosine similarity between word embedding od story and endings

Attention_model.py: Attention model on top of Generative Model.

```

# How to run files:
## sentiment.py
1. To run sentiment.py, you need to Stanford CoreNLP using this link: https://stanfordnlp.github.io/CoreNLP/download.html#steps-to-setup-from-the-official-release
2. Next install pycorenlp using 'pip install pycorenlp'
3. Now run the code using 'python sentiment.py'

## seq2seq_learning.py
1. This code requires PyTorch to be installed. PyTorch can be installed from here http://pytorch.org/
2. Also NLTK needs to be installed.
3. To run this file use 'python seq2seq_learning.py'
4. There are many parameters that can be set from the code.
    1. Line-13: NUM_EPOCHS: Number of epochs for training.
    2. Line-15: USE_TEACHER_FORCING: If set True, uses teacher forcing or else does the training without teacher forcing.
    3. Line-16: BIDIECTIONAL: Uses Bidirectional GRU cells if set to True or else uses Unidirectional GRU
    4. Line-17: REVERSE: Reverses the input string.

## train_word_embed.py
1. This code also needs PyTorch and NLTK, which can be installed as mentioned above
2. It requires scikit-learn as well, which can be installed using 'pip install -U numpy scipy scikit-learn'
3. To run this code, use 'train_word_embed.py'
4. This generates the vector representation for words in the training corpus and stores in the file 'word2vec'

## word2vec.py
1. This code also requires PyTorch, NLTK and scikit-learn
2. This script uses 'word2vec', generated by train_word_embed.py
3. Using word2vec, it generates the average vector representation of the 4 sentences of story and that of endings as well
4. Then it computes cosine similarity to find the best match
5. This can be run using 'python word2vec.py'

## Attention_model.py
1. This code requires PyTorch,NLTK installation.
2. To run this file use 'python Attention_model.py'
3. There are some parameters that can be changed as per requiement.
    1. Line-13: NUM_EPOCHS: Number of epochs for training.
    2. Line-16: BIDIECTIONAL: To be set True or else uses Unidirectional GRU for Encoder.
4.This file runs Attention model on top of Encoder-Decoder used in seq2seq_learning.py

