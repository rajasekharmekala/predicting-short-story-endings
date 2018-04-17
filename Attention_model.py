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
NUM_EPOCHS = 8
GRAD_CLIP = 5.0
USE_TEACHER_FORCING = False
BIDIRECTIONAL = False
REVERSE = False
ENCODER_MODEL = 'encoder_0_06_TF0_03.mdl'
DECODER_MODEL = 'decoder_0_06_TF0_03.mdl'
MAX_LENGTH = 100
cuda_available = torch.cuda.is_available()

if cuda_available:
	print('Using CUDA')
else:
	print('CUDA Device not found')

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

def sentence2idx( sentence, language ):
	sentence_idx = [ language.word2idx[word] for word in sentence.split() if word in language.word2idx ]
	sentence_idx.append( EOS_token )
	return sentence_idx




class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result



story_lang = Language()

cnt = 0

flag = False
train_set = list()
with open('ROCStories__spring2016 - ROCStories_spring2016.csv', 'r') as f:
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
print('Built Language with', story_lang.num_words, 'words!!!')
print('Training Set loaded')
#print (story_lang.word2idx.keys())
flag = False
test_set = list()
with open('cloze_test_test__spring2016 - cloze_test_ALL_test.csv', 'r') as f:
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
print('Test Set loaded')


encoder1 = RNNEncoder(story_lang.num_words, 512, 4)
attn_decoder1 = AttnDecoderRNN(512,story_lang.num_words, dropout_p=0.1)

if cuda_available:
    encoder1 = encoder1.cuda()
    attn_decoder1 = attn_decoder1.cuda()

encoder1_optimizer = optim.SGD(encoder1.parameters(), lr=0.03)
attn_decoder1_optimizer = optim.SGD(attn_decoder1.parameters(), lr=0.03)
loss_function = nn.NLLLoss()
for epoch in range(NUM_EPOCHS):
	total_loss = torch.FloatTensor([0.0]) #.cuda()
	if cuda_available:
		total_loss = total_loss.cuda()

	for iter, (input_var, output_var) in enumerate( train_set ):
		loss = 0
		encoder1_optimizer.zero_grad()
		attn_decoder1_optimizer.zero_grad()
		encoder_hidden = encoder1.init_hidden()
		input_var = sentence2idx( input_var, story_lang )
		output_var = sentence2idx( output_var, story_lang )
		input_var = Variable( torch.LongTensor( input_var ) )
		output_var = Variable( torch.LongTensor( output_var ) )

		if cuda_available:
			input_var = input_var.cuda()
			output_var = output_var.cuda()

		input_length = input_var.size()[0]
		output_length = output_var.size()[0]
		encoder1_outputs = Variable(torch.zeros(MAX_LENGTH, encoder1.hidden_size))
		encoder1_outputs = encoder1_outputs.cuda() if cuda_available else encoder1_outputs

		for ei in range( input_length ):
			encoder1_output, encoder1_hidden = encoder1(input_var[ei], encoder_hidden)
			encoder1_outputs[ei] = encoder1_output[0][0]

		attn_decoder1_input = Variable(torch.LongTensor([[SOS_token]]))
		if cuda_available:
			attn_decoder1_input = attn_decoder1_input.cuda()
			encoder1_hidden = encoder1_hidden.cuda()
		attn_decoder1_hidden = encoder_hidden[-1].view(-1,encoder_hidden[-1].size()[0],encoder_hidden[-1].size()[1])

		for di in range(output_length):
			# print(attn_decoder1_input.size(), attn_decoder1_hidden.size(), encoder1_outputs.size())
			attn_decoder1_output, attn_decoder1_hidden, decoder1_attention = attn_decoder1(attn_decoder1_input, attn_decoder1_hidden, encoder1_outputs)
			top_var, top_idx = attn_decoder1_output.data.topk(1)
			ni = top_idx[0][0]

			loss += loss_function( attn_decoder1_output, output_var[di] )
			if ni == EOS_token:
				break
			if USE_TEACHER_FORCING:
				attn_decoder1_input = output_var[di]
			else:
				attn_decoder1_input = Variable(torch.LongTensor([[ni]]))
				if cuda_available:
					attn_decoder1_input = attn_decoder1_input.cuda()
			total_loss += loss.data
		loss.backward()
		torch.nn.utils.clip_grad_norm( encoder1.parameters(), GRAD_CLIP )
		torch.nn.utils.clip_grad_norm( attn_decoder1.parameters(), GRAD_CLIP )
		encoder1_optimizer.step()
		attn_decoder1_optimizer.step()	
		#if iter % 500 == 0:
		print('Epoch: [', epoch + 1, '/', NUM_EPOCHS ,'] Iter: [', iter + 1, '/', len(train_set), ']')
	print('Total Loss', total_loss[0] / len(train_set))
	torch.save(encoder1.state_dict(), 'models/' + ENCODER_MODEL)
	torch.save(attn_decoder1.state_dict(), 'models/' + DECODER_MODEL)

f1 = open('attention_ending.txt', 'w')
a_bleu_list = list()
for iter, ( input, output_var ) in enumerate( test_set ):
	input_var = sentence2idx( input, story_lang )
	input_var = Variable( torch.LongTensor( input_var ) )

	if cuda_available:
		input_var = input_var.cuda()

	input_length = input_var.size()[0]

	encoder1_hidden = encoder1.init_hidden()

	encoder1_outputs = Variable(torch.zeros(MAX_LENGTH, encoder1.hidden_size))
	encoder1_outputs = encoder1_outputs.cuda() if cuda_available else encoder1_outputs
	for ei in range(input_length):
		encoder1_output, encoder1_hidden = encoder1(input_var[ei], encoder_hidden)
		encoder1_outputs[ei] = encoder1_output[0][0]

	attn_decoder1_input = Variable( torch.LongTensor([[SOS_token]]) )
	if cuda_available:
		attn_decoder1_input = attn_decoder1_input.cuda()
		encoder1_hidden = encoder1_hidden.cuda()
	attn_decoder1_hidden = encoder_hidden[-1].view(-1,encoder_hidden[-1].size()[0],encoder_hidden[-1].size()[1])

	decoder1_attentions = torch.zeros(MAX_LENGTH, MAX_LENGTH)

	a_predicted = list()
	for di in range(MAX_LENGTH):
		attn_decoder1_output, attn_decoder1_hidden, decoder1_attention = attn_decoder1(attn_decoder1_input, attn_decoder1_hidden, encoder1_outputs)
		decoder1_attentions[di] = decoder1_attention.data
		top_var, top_idx = attn_decoder1_output.data.topk(1)
		ni = top_idx[0][0]
		attn_decoder1_input = Variable( torch.LongTensor([[ni]]) )
		if cuda_available:
			attn_decoder1_input = attn_decoder1_input.cuda()
		a_predicted.append( story_lang.idx2word[ ni ] )
		if ni == EOS_token:
			break
	reference = [ output_var.split() ]
	a_bleu_list.append( sentence_bleu( reference, a_predicted ) )
	f1.write( 'Story:' + input + '\n')
	f1.write( 'Correct Ending:' + output_var + '\n' )
	f1.write( 'Predicted Ending:' + ' '.join( a_predicted ) + '\n' )
	f1.write( '--------------------------------------------\n' )

print('Attention_model Average BLEU Score', float( sum( a_bleu_list ) ) / len( a_bleu_list ))


