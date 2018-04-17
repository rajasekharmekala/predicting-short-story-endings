import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import csv
import numpy as np
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
ps = PorterStemmer()
stop_words = set( stopwords.words('english') )
torch.manual_seed(1)
torch.cuda.manual_seed(1)

CONTEXT_SIZE = 2
EMBEDDING_DIM = 128

vocab = list()
class NGramLanguageModeler(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_size):
        super(NGramLanguageModeler, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim, sparse=True)
        self.linear1 = nn.Linear(context_size*embedding_dim, 128)
        self.linear2 = nn.Linear(128, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((1, -1))
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs

ngrams = list()
with open('ROCStories__spring2016 - ROCStories_spring2016.csv', 'rb') as f:
    f_csv = csv.reader(f, delimiter=',')
    for line in f_csv:
        story = ' '.join( [ l for l in line[1:] ] )
        tokens = word_tokenize( story )
        filtered_tokens = [ w for w in tokens if w not in stop_words ]
        ngrams.extend([ ([filtered_tokens[i-2], filtered_tokens[i-1]], filtered_tokens[i]) for i in range( len( filtered_tokens ) - 2 )])
        vocab.extend( set(filtered_tokens) )
    vocab = set(vocab)
    word_to_idx = {word:i for i, word in enumerate(vocab)}
print 'Total No. of n-grams', len(ngrams)
print 'Vocab Size', len( vocab )
losses = []
loss_function = nn.NLLLoss()
model = NGramLanguageModeler(len(vocab), EMBEDDING_DIM, CONTEXT_SIZE)
optimizer = optim.SGD(model.parameters(), lr=0.001)
use_cuda = True
if use_cuda:
    model.cuda()
for epoch in range(3):
    total_loss = torch.cuda.FloatTensor([0.0])
    cnt = 0
    for context, target in ngrams:
        context_to_idxs = [word_to_idx[w] for w in context]
        context_var = autograd.Variable(torch.LongTensor(context_to_idxs).cuda())
        model.zero_grad()
        log_probs = model(context_var)
        loss = loss_function( log_probs, autograd.Variable(torch.LongTensor([word_to_idx[target]]).cuda() ) )
        loss.backward()
        optimizer.step()
        print cnt, '/', len(ngrams)
        cnt += 1
        total_loss += loss.data
    print total_loss
    print losses.append( total_loss )

print losses
torch.save( model.state_dict(), 'word2vec' )
