import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import csv
import re
import argparse
from nltk.translate.bleu_score import sentence_bleu

SOS_token = 0
EOS_token = 1
NUM_EPOCHS = 20
GRAD_CLIP = 5.0
USE_TEACHER_FORCING = False
BIDIRECTIONAL = False
REVERSE = True
ENCODER_MODEL = 'encoder_rev_256_2_0_04.mdl'
DECODER_MODEL = 'decoder_rev_256_2_0_04.mdl'

cuda_available = torch.cuda.is_available()

if cuda_available:
	print 'Using CUDA'
else:
	print 'CUDA Device not found'

class Language():
	def __init__(self):
		self.word_count = dict()
		self.word2idx = {'SOS': SOS_token, 'EOS': EOS_token}
		self.idx2word = {SOS_token: 'SOS', EOS_token: 'EOS'}
		self.num_words = 2
		self.trim_threshold_freq = 0
	def add_sentence(self, sentence):
		for word in sentence.split():
			self.add_word( word )
	def add_word(self, word):
		if word not in self.word_count:
			self.word_count[word] = 0
		self.word_count[word] += 1
	def build_trimmed_vocab(self):
		for word in self.word_count:
			if self.word_count[word] > self.trim_threshold_freq:
				if word not in self.word2idx:
					self.word2idx[word] = self.num_words
					self.idx2word[self.num_words] = word
					self.num_words += 1

class RNNEncoder(nn.Module):
	def __init__(self, input_size, hidden_size, num_layers):
		super(RNNEncoder, self).__init__()
		self.hidden_size = hidden_size
		self.num_layers = num_layers

		self.embedding = nn.Embedding(input_size, hidden_size)
		self.gru = nn.GRU(hidden_size, hidden_size, num_layers, bidirectional=BIDIRECTIONAL)

	def forward(self, words, hidden):
		seq_len = len(words)
		embeds = self.embedding(words).view(seq_len, 1, -1)
		output = embeds
		output, hidden = self.gru(output, hidden)
		return output, hidden

	def init_hidden(self):
		if BIDIRECTIONAL:
			hidden = Variable(torch.zeros(2*self.num_layers, 1, self.hidden_size))
		else:
			hidden = Variable(torch.zeros(self.num_layers, 1, self.hidden_size))
		if cuda_available:
			hidden = hidden.cuda()
		return hidden

class RNNDecoder(nn.Module):
	def __init__(self, output_size, hidden_size, num_layers):
		super(RNNDecoder, self).__init__()
		self.hidden_size = hidden_size
		self.num_layers = num_layers

		self.embedding = nn.Embedding(output_size, hidden_size)
		self.lstm = nn.GRU(hidden_size, hidden_size, num_layers, bidirectional=BIDIRECTIONAL)
		if BIDIRECTIONAL:
			self.out = nn.Linear(2*hidden_size, output_size)
		else:
			self.out = nn.Linear(hidden_size, output_size)
	def forward(self, input, hidden):
		seq_len = len(input)
		output = self.embedding(input).view(seq_len, 1, -1)
		output = F.relu(output)
		output, hidden = self.lstm(output, hidden)
		output = self.out(output)
		output  = F.log_softmax(output[0], dim=1)
		return output, hidden

	def init_hidden(self):
		hidden = Variable(torch.zeros(self.num_layers, 1, self.hidden_size))
		if cuda_available:
			hidden = hidden.cuda()
		return hidden

def sentence2idx( sentence, language, reverse=False ):
	sentence_idx = [ language.word2idx[word] for word in sentence.split() if word in language.word2idx ]
	if reverse:
		sentence_idx.reverse( )
	sentence_idx.append( EOS_token )
	return sentence_idx

story_lang = Language()

cnt = 0

flag = False
train_set = list()
with open('ROCStories__spring2016 - ROCStories_spring2016.csv', 'rb') as f:
	f_csv = csv.reader(f, delimiter=',')
	for line in f_csv:
		if not flag:
			flag = True
			continue
		if cnt == 20000:
			break
		cnt += 1
		story = ' '.join( [ l for l in line[2:7] ] ).lower()
		story = re.sub( '[^a-zA-Z ]', ' ', story )
		story_lang.add_sentence( story )
		short_story = ' '.join( [ l for l in line[2:6]] )
		ending = line[6].lower()
		short_story = re.sub( '[^a-zA-Z ]', ' ', short_story )
		ending = re.sub( '[^a-zA-Z ]', ' ', ending )
		train_set.append( ( short_story, ending ) )

story_lang.build_trimmed_vocab()

assert story_lang.num_words == len( story_lang.word2idx )
assert len( story_lang.word2idx ) == len( story_lang.idx2word )
print 'Built Language with', story_lang.num_words, 'words!!!'
print 'Training Set loaded'
#print story_lang.word2idx.keys()
flag = False
test_set = list()
with open('cloze_test_test__spring2016 - cloze_test_ALL_test.csv', 'rb') as f:
        f_csv = csv.reader(f, delimiter=',')
        for line in f_csv:
		if not flag:
			flag = True
			continue
		story = ' '.join( line[1:5] ).lower()
		target = line[7]
		if target == '1':
			ending = line[5]
		else:
			ending = line[6]
		ending = ending.lower()
		story = re.sub( '[^a-zA-Z ]', ' ', story )
		ending = re.sub( '[^a-zA-Z ]', ' ', ending )
		test_set.append( ( story, ending ) )
print 'Test Set loaded'

encoder = RNNEncoder(story_lang.num_words, 256, 2)
decoder = RNNDecoder(story_lang.num_words, 256, 2)

try:
	encoder.load_state_dict( torch.load('models/' + ENCODER_MODEL) )
	decoder.load_state_dict( torch.load('models/' + DECODER_MODEL) )
except:
	pass

if cuda_available:
	encoder.cuda()
	decoder.cuda()

encoder_optimizer = optim.SGD(encoder.parameters(), lr=0.04)
decoder_optimizer = optim.SGD(decoder.parameters(), lr=0.04)
loss_function = nn.NLLLoss()
for epoch in range(NUM_EPOCHS):
	total_loss = torch.FloatTensor([0.0]) #.cuda()
	if cuda_available:
		total_loss = total_loss.cuda()

	for iter, (input_var, output_var) in enumerate( train_set ):
		#if iter == 20001:
		#	break
		loss = 0
		encoder_optimizer.zero_grad()
		decoder_optimizer.zero_grad()
		encoder_hidden = encoder.init_hidden()
		input_var = sentence2idx( input_var, story_lang, REVERSE )
		output_var = sentence2idx( output_var, story_lang )
		input_var = Variable( torch.LongTensor( input_var ) )
		output_var = Variable( torch.LongTensor( output_var ) )

		if cuda_available:
			input_var = input_var.cuda()
			output_var = output_var.cuda()

		input_length = input_var.size()[0]
		output_length = output_var.size()[0]
		for ei in range( input_length ):
			encoder_output, encoder_hidden = encoder(input_var[ei], encoder_hidden)

		decoder_input = Variable(torch.LongTensor([[SOS_token]]))
		if cuda_available:
			decoder_input = decoder_input.cuda()

		decoder_hidden = encoder_hidden

		for di in range(output_length):
			decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
			top_var, top_idx = decoder_output.data.topk(1)
			ni = top_idx[0][0]

			loss += loss_function( decoder_output, output_var[di] )
			if ni == EOS_token:
				break
			if USE_TEACHER_FORCING:
				decoder_input = output_var[di]
			else:
				decoder_input = Variable(torch.LongTensor([[ni]]))
				if cuda_available:
					decoder_input = decoder_input.cuda()
			total_loss += loss.data
		loss.backward()
		torch.nn.utils.clip_grad_norm( encoder.parameters(), GRAD_CLIP )
		torch.nn.utils.clip_grad_norm( decoder.parameters(), GRAD_CLIP )
		encoder_optimizer.step()
		decoder_optimizer.step()
		
		#if iter % 500 == 0:
		print 'Epoch: [', epoch + 1, '/', NUM_EPOCHS ,'] Iter: [', iter + 1, '/', len(train_set), ']'
	print 'Total Loss', total_loss[0] / len(train_set)
	torch.save(encoder.state_dict(), 'models/' + ENCODER_MODEL)
	torch.save(decoder.state_dict(), 'models/' + DECODER_MODEL)

f = open('ending.txt', 'w')
bleu_list = list()
for iter, ( input, output_var ) in enumerate( test_set ):
	input_var = sentence2idx( input, story_lang, REVERSE )
	input_var = Variable( torch.LongTensor( input_var ) )

	if cuda_available:
		input_var = input_var.cuda()

	input_length = input_var.size()[0]
	encoder_hidden = encoder.init_hidden()
	for ei in range(input_length):
		encoder_output, encoder_hidden = encoder(input_var[ei], encoder_hidden)
	decoder_input = Variable( torch.LongTensor([[SOS_token]]) )
	if cuda_available:
		decoder_input = decoder_input.cuda()
	decoder_hidden = encoder_hidden
	predicted = list()
	for di in range( 10 ):
		decoder_output, decoder_hidden = decoder( decoder_input, decoder_hidden )
		top_var, top_idx = decoder_output.data.topk(1)
		ni = top_idx[0][0]
		decoder_input = Variable( torch.LongTensor([[ni]]) )
		if cuda_available:
			decoder_input = decoder_input.cuda()
		predicted.append( story_lang.idx2word[ ni ] )
		if ni == EOS_token:
			break
	reference = [ output_var.split() ]
	bleu_list.append( sentence_bleu( reference, predicted ) )
	f.write( 'Story:' + input + '\n')
	f.write( 'Correct Ending:' + output_var + '\n' )
	f.write( 'Predicted Ending:' + ' '.join( predicted ) + '\n' )
	f.write( '--------------------------------------------\n' )

print 'Average BLEU Score', float( sum( bleu_list ) ) / len( bleu_list )
