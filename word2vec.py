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
from sklearn.metrics.pairwise import cosine_similarity


reload(sys)
sys.setdefaultencoding('utf-8')

stop_words = set(stopwords.words('english'))
flag = False
EMBEDDING_DIM = 128
CONTEXT_SIZE = 2

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
    def get_embedding(self, inputs):
	embeds = self.embeddings(inputs).view((1, -1))
	return embeds

ngrams = list()
vocab = list()
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

model = NGramLanguageModeler(len(vocab), EMBEDDING_DIM, CONTEXT_SIZE)
model.load_state_dict( torch.load('word2vec') )
model.cuda()

def compute_cosine_sim( story_vec, endings_vec ):
	print story_vec.shape, endings_vec.shape
	story_vec = torch.t(story_vec)
	endings_vec = torch.t(endings_vec)
	return cosine_similarity( story_vec, endings_vec )

def compute_doc2vec( document ):
	vec_tensor = torch.FloatTensor().cuda()
	doc_tokens = [w for w in word_tokenize(document) if w not in stop_words]
	cnt = 0 
	for word in doc_tokens:
		if word in word_to_idx:
			cnt += 1
			word_list = [ word_to_idx[word] ]
			word_tensor = autograd.Variable( torch.LongTensor(word_list).cuda() )
			embed = model.get_embedding( word_tensor )
			vec_tensor = torch.cat( ( vec_tensor, torch.cuda.FloatTensor( embed.data ) ) )
	#print vec_tensor.shape
	doc2vec = torch.mean( vec_tensor, dim=0 )
	return doc2vec

cnt = 0.0
corr_pred = 0.0
with open('cloze_test_test__spring2016 - cloze_test_ALL_test.csv', 'rb') as f:
	f_csv = csv.reader(f, delimiter=',')
	for line in f_csv:
		print cnt
		cnt += 1.0
		if flag:
			story = line[4] #' '.join( line[2:5] )
			ending1 = line[5]
			ending2 = line[6]
			correct_label = int( line[7] )
			story_vec = compute_doc2vec( story )
			end1_vec = compute_doc2vec( ending1 )
			end2_vec = compute_doc2vec( ending2 )
			print story_vec.shape, end1_vec.shape, end2_vec.shape
			end_vec = torch.cat( ( end1_vec, end2_vec ) )
			end_vec = end_vec.view( (EMBEDDING_DIM, 2) )
			cos_sim = compute_cosine_sim( story_vec.view((128, 1)), end_vec )[0]
			if cos_sim[0] > cos_sim[1]:
				pred = 1
			else:
				pred = 2
			if pred == correct_label:
				corr_pred += 1.0
		else:
			flag = True
print 'Accuracy', corr_pred/cnt*100.0, '%'
