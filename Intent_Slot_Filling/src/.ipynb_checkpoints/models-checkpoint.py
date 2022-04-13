import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn
import torch
#import torchnlp
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch import nn

from crf import CRF
from utils import clip_pad

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True


class IntentSlotsClassifier(nn.Module):
    def __init__(
        self,
        vocab_size,
        output_size,
        num_slots,
        embedding_dim,
        embedding_matrix,
        hidden_size,
        num_layers,
        drop_prob=0.5,
    ):
        super(IntentSlotsClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weights = nn.Parameter(
            torch.from_numpy(embedding_matrix))
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            embedding_dim, hidden_size, num_layers, batch_first=True, bidirectional=True
        )
        self.num_slots = num_slots
        self.fc = nn.Linear(hidden_size * 2, output_size)  # 2 for bidirection
        self.fc_slot = nn.Linear(hidden_size * 2, num_slots)
        self.dropout = nn.Dropout(drop_prob)

    #### A very simple network composed of an embedding layer, a stack of (num_layers) LSTMs with dropout and finally a fully connected layer ####

    # Last hidden state goes into a Dense+softmax to produce intent, other hidden states go into another dense+softmax to predict slots

    def forward(self, x):
        decode = []
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(
            device
        )  # 2 for bidirection
        c0 = torch.zeros(self.num_layers * 2, x.size(0),
                         self.hidden_size).to(device)

        # Forward propagate LSTM
        length = x.size(1)
        out = self.embedding(x)
        out, _ = self.lstm(
            out, (h0, c0)
        )  # out: tensor of shape (batch_size, seq_length, hidden_size*2)
        out = self.dropout(out)

        for i in range(length):
            score = self.fc_slot(out[:, i, :])
            decode.append(score)

        # Decode the hidden state of the last time step
        intent = self.fc(out[:, -1, :])

        decode = torch.cat(decode, 1)

        return intent, decode.view(x.size(0), self.num_slots, length)


class IntentClassifier(nn.Module):
    def __init__(
        self,
        vocab_size,
        output_size,
        embedding_dim,
        embedding_matrix,
        hidden_size,
        num_layers,
        drop_prob=0.5,
    ):
        super(IntentClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weights = nn.Parameter(
            torch.from_numpy(embedding_matrix))
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            embedding_dim, hidden_size, num_layers, batch_first=True, bidirectional=True
        )
        self.fc = nn.Linear(hidden_size * 2, output_size)  # 2 for bidirection
        self.dropout = nn.Dropout(drop_prob)

    #### A very simple network composed of an embedding layer, a stack of (num_layers) LSTMs with dropout and finally a fully connected layer ####
    # Last hidden state goes into dense+softmax to predict intent
    def forward(self, x):
        decode = []
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(
            device
        )  # 2 for bidirection
        c0 = torch.zeros(self.num_layers * 2, x.size(0),
                         self.hidden_size).to(device)

        # Forward propagate LSTM
        length = x.size(1)
        out = self.embedding(x)
        out, _ = self.lstm(
            out, (h0, c0)
        )  # out: tensor of shape (batch_size, seq_length, hidden_size*2)
        out = self.dropout(out)

        # Decode the hidden state of the last time step
        intent = self.fc(out[:, -1, :])

        return intent


class SlotsClassifier(nn.Module):
    def __init__(
        self,
        vocab_size,
        num_slots,
        embedding_dim,
        embedding_matrix,
        hidden_size,
        num_layers,
        drop_prob=0.5,
    ):
        super(SlotsClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weights = nn.Parameter(
            torch.from_numpy(embedding_matrix))
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            embedding_dim, hidden_size, num_layers, batch_first=True, bidirectional=True
        )
        self.num_slots = num_slots
        self.fc_slot = nn.Linear(hidden_size * 2, num_slots)
        self.dropout = nn.Dropout(drop_prob)

    #### A very simple network composed of an embedding layer, a stack of (num_layers) LSTMs with dropout and finally a fully connected layer ####
    # Hidden state of all step go into Dense+softmax to predict slots
    def forward(self, x):
        decode = []
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(
            device
        )  # 2 for bidirection
        c0 = torch.zeros(self.num_layers * 2, x.size(0),
                         self.hidden_size).to(device)

        # Forward propagate LSTM
        length = x.size(1)
        out = self.embedding(x)
        out, _ = self.lstm(
            out, (h0, c0)
        )  # out: tensor of shape (batch_size, seq_length, hidden_size*2)
        out = self.dropout(out)

        for i in range(length):
            score = self.fc_slot(out[:, i, :])
            decode.append(score)

        decode = torch.cat(decode, 1)

        return decode.view(x.size(0), self.num_slots, length)

########LALALALA#######


class VanillaEncoderDecoder(nn.Module):

    def __init__(self, vocab_size, embed_size, embedding_matrix, hidden_size, num_intent, num_slots, n_layers=1, dropout_rate=0.2):

        super(VanillaEncoderDecoder, self).__init__()
        self.model_embeddings = nn.Embedding(vocab_size, embed_size)
        self.model_embeddings.weights = nn.Parameter(
            torch.from_numpy(embedding_matrix))
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate

        self.encoder = None
        self.decoder = None
        self.fc_slots = None
        self.fc_intent = None
        self.h_projection = None
        self.c_projection = None
        self.num_slots = num_slots
        self.num_intent = num_intent

        # self.att_projection = None
        # self.combined_output_projection = None
        self.n_layers = n_layers
        self.encoder = nn.LSTM(embed_size, hidden_size,
                               num_layers=self.n_layers, bias=True, bidirectional=True, batch_first=True)
        self.decoder = nn.LSTM(
            embed_size, hidden_size * 2, bias=True, batch_first=True)
        self.fc = nn.Linear(hidden_size * 2, num_intent)  # 2 for bidirection
        self.fc_slot = nn.Linear(hidden_size*2, num_slots)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, input):

        enc_hiddens, (hidden, cell) = self.encode(input)

        start_decode = Variable(torch.LongTensor(
            [[1]*input.size(0)])).cuda().transpose(1, 0)

        decode_slots = []
        length = input.size(1)
        hidden = hidden.unsqueeze(0)
        cell = cell.unsqueeze(0)
        for i in range(length):
            outputs, hidden, cell = self.decode(start_decode, (hidden, cell))

            score = self.fc_slot(hidden.squeeze(0))
            decode_slots.append(score)

        decode_slots = torch.cat(decode_slots, 1)

        # Decode the hidden state of the last time step
        intent = self.fc(hidden.squeeze(0))

        return intent, decode_slots.view(input.size(0), self.num_slots, length)

    def encode(self, input):
        h0 = torch.zeros(self.n_layers * 2, input.size(0), self.hidden_size).to(
            device
        )  # 2 for bidirection
        c0 = torch.zeros(self.n_layers * 2, input.size(0),
                         self.hidden_size).to(device)
        input = self.model_embeddings(input)
        enc_hidden, (last_hidden, last_cell) = self.encoder(input, (h0, c0))
        init_decoder_hidden = torch.cat((last_hidden[0], last_hidden[1]), 1)
        init_decoder_cell = torch.cat((last_cell[0], last_cell[1]), 1)

        return enc_hidden, (init_decoder_hidden, init_decoder_cell)

    def decode(self, input, last):
        hidden, cell = last

        embedded = self.dropout(self.model_embeddings(input))
        output, (hidden, cell) = self.decoder(embedded, (hidden, cell))

        return output, hidden, cell


class CRFEncoderDecoder(nn.Module):

    def __init__(self, vocab_size, embed_size, embedding_matrix, hidden_size, num_intent, num_slots, n_layers=1, dropout_rate=0.2):

        super(CRFEncoderDecoder, self).__init__()
        self.model_embeddings = nn.Embedding(vocab_size, embed_size)
        self.model_embeddings.weights = nn.Parameter(
            torch.from_numpy(embedding_matrix))
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate

        self.encoder = None
        self.decoder = None
        self.fc_slots = None
        self.fc_intent = None
        self.h_projection = None
        self.c_projection = None
        self.num_slots = num_slots
        self.num_intent = num_intent
        self.crf = CRF(hidden_size*2, self.num_slots)

        # self.att_projection = None
        # self.combined_output_projection = None
        self.n_layers = n_layers
        self.encoder = nn.LSTM(embed_size, hidden_size,
                               num_layers=self.n_layers, bias=True, bidirectional=True, batch_first=True)
        self.decoder = nn.LSTM(
            embed_size, hidden_size * 2, bias=True, batch_first=True)
        self.fc = nn.Linear(hidden_size * 2, num_intent)  # 2 for bidirection
        self.fc_slot = nn.Linear(hidden_size*2, num_slots)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, input):

        enc_hiddens, (hidden, cell) = self.encode(input)

        start_decode = Variable(torch.LongTensor(
            [[1]*input.size(0)])).cuda().transpose(1, 0)

        decode_slots = []
        length = input.size(1)
        hidden = hidden.unsqueeze(0)
        cell = cell.unsqueeze(0)
        all_outputs = []
        for i in range(length):
            outputs, hidden, cell = self.decode(start_decode, (hidden, cell))
            all_outputs.append(outputs)

            score = self.fc_slot(hidden.squeeze(0))
            decode_slots.append(score)

        print('\n', 'len decode slots', len(decode_slots))

        print([len(elm) for elm in decode_slots])

        all_outputs = torch.cat(all_outputs, 1)
        masks = input.gt(0)
        scores, tag_seq = self.crf(all_outputs, masks)

        print(len(tag_seq))
        # print(len(decode_slots))  # L x B x ?
        decode_slots = torch.cat(decode_slots, 1)
        
        tag_seq = [torch.Tensor(clip_pad(elm, input.size(1), [129])) for elm in tag_seq]  # B x L x ?
        print('scores_shape', scores)
        print([len(elm) for elm in tag_seq])
        print('\ntaaaaag seeeq', tag_seq[0].size())
        tag_seq = torch.stack(tag_seq, dim=0)
        print(tag_seq.size())
        # Decode the hidden state of the last time step
        intent = self.fc(hidden.squeeze(0))

        # return intent, decode_slots.view(input.size(0), self.num_slots, length)
        return intent, tag_seq

    def encode(self, input):
        h0 = torch.zeros(self.n_layers * 2, input.size(0), self.hidden_size).to(
            device
        )  # 2 for bidirection
        c0 = torch.zeros(self.n_layers * 2, input.size(0),
                         self.hidden_size).to(device)
        input = self.model_embeddings(input)
        enc_hidden, (last_hidden, last_cell) = self.encoder(input, (h0, c0))
        init_decoder_hidden = torch.cat((last_hidden[0], last_hidden[1]), 1)
        init_decoder_cell = torch.cat((last_cell[0], last_cell[1]), 1)

        return enc_hidden, (init_decoder_hidden, init_decoder_cell)

    def decode(self, input, last):
        hidden, cell = last

        embedded = self.dropout(self.model_embeddings(input))
        output, (hidden, cell) = self.decoder(embedded, (hidden, cell))

        return output, hidden, cell

#############################################################################################################
##Everything below this line is directly taken from : https://github.com/pengshuang/Joint-Slot-Filling.git###
#############################################################################################################


USE_CUDA = torch.cuda.is_available()


class Encoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, batch_size=16, n_layers=1):
        super(Encoder, self).__init__()

        self.input_size = input_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.batch_size = batch_size

        self.embedding = nn.Embedding(input_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size,
                            n_layers, batch_first=True, bidirectional=True)

    def init_weights(self):
        self.embedding.weight.data.uniform_(-0.1, 0.1)

    def init_hidden(self, input):
        if USE_CUDA:
            hidden = torch.zeros(
                self.n_layers*2, input.size(0), self.hidden_size).cuda()
        else:
            hidden = Variable(torch.zeros(
                self.n_layers*2, input.size(0), self.hidden_size))

        if USE_CUDA:
            context = Variable(torch.zeros(
                self.n_layers*2, input.size(0), self.hidden_size)).cuda()
        else:
            context = Variable(torch.zeros(
                self.n_layers*2, input.size(0), self.hidden_size))

        return hidden, context

    def forward(self, input, input_masking):
        """
        input : B,T
        input_masking : B,T
        output : B,T,D  B,1,D
        """
        hidden = self.init_hidden(input)
        # B,T,D
        embedded = self.embedding(input)
        # B,T,D
        output, hidden = self.lstm(embedded, hidden)

        real_context = []

        for i, o in enumerate(output):
            # B,T,D
            real_length = input_masking[i].data.tolist().count(0)
            real_context.append(o[real_length-1])

        return output, torch.cat(real_context).view(input.size(0), -1).unsqueeze(1)


class VanillaEncoderDecoder(nn.Module):

    def __init__(self, vocab_size, embed_size, embedding_matrix, hidden_size, num_intent, num_slots, n_layers=1, dropout_rate=0.2):

        super(VanillaEncoderDecoder, self).__init__()
        self.model_embeddings = nn.Embedding(vocab_size, embed_size)
        self.model_embeddings.weights = nn.Parameter(
            torch.from_numpy(embedding_matrix))
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate

        self.encoder = None
        self.decoder = None
        self.fc_slots = None
        self.fc_intent = None
        self.h_projection = None
        self.c_projection = None
        self.num_slots = num_slots
        self.num_intent = num_intent

        # self.att_projection = None
        # self.combined_output_projection = None
        self.n_layers = n_layers
        self.encoder = nn.LSTM(embed_size, hidden_size,
                               num_layers=self.n_layers, bias=True, bidirectional=True, batch_first=True)
        self.decoder = nn.LSTM(
            embed_size, hidden_size * 2, bias=True, batch_first=True)
        self.fc = nn.Linear(hidden_size * 2, num_intent)  # 2 for bidirection
        self.fc_slot = nn.Linear(hidden_size*2, num_slots)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, input):

        enc_hiddens, (hidden, cell) = self.encode(input)

        start_decode = Variable(torch.LongTensor(
            [[1]*input.size(0)])).cuda().transpose(1, 0)

        decode_slots = []
        length = input.size(1)
        hidden = hidden.unsqueeze(0)
        cell = cell.unsqueeze(0)
        for i in range(length):
            outputs, hidden, cell = self.decode(start_decode, (hidden, cell))

            score = self.fc_slot(hidden.squeeze(0))
            decode_slots.append(score)

        decode_slots = torch.cat(decode_slots, 1)

        # Decode the hidden state of the last time step
        intent = self.fc(hidden.squeeze(0))

        return intent, decode_slots.view(input.size(0), self.num_slots, length)

    def encode(self, input):
        h0 = torch.zeros(self.n_layers * 2, input.size(0), self.hidden_size).to(
            device
        )  # 2 for bidirection
        c0 = torch.zeros(self.n_layers * 2, input.size(0),
                         self.hidden_size).to(device)
        input = self.model_embeddings(input)
        enc_hidden, (last_hidden, last_cell) = self.encoder(input, (h0, c0))
        init_decoder_hidden = torch.cat((last_hidden[0], last_hidden[1]), 1)
        init_decoder_cell = torch.cat((last_cell[0], last_cell[1]), 1)

        return enc_hidden, (init_decoder_hidden, init_decoder_cell)

    def decode(self, input, last):
        hidden, cell = last

        embedded = self.dropout(self.model_embeddings(input))
        output, (hidden, cell) = self.decoder(embedded, (hidden, cell))

        return output, hidden, cell


class Decoder(nn.Module):

    def __init__(self, slot_size, intent_size, embedding_size, hidden_size, batch_size=16, n_layers=1, dropout_p=0.1):
        super(Decoder, self).__init__()

        self.hidden_size = hidden_size
        self.slot_size = slot_size
        self.intent_size = intent_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.embedding_size = embedding_size
        self.batch_size = batch_size

        # Define the layers
        self.embedding = nn.Embedding(self.slot_size, self.embedding_size)

        self.lstm = nn.LSTM(self.embedding_size+self.hidden_size*2,
                            self.hidden_size, self.n_layers, batch_first=True)
        self.attn = nn.Linear(self.hidden_size, self.hidden_size)
        self.slot_out = nn.Linear(self.hidden_size*2, self.slot_size)
        self.intent_out = nn.Linear(self.hidden_size*2, self.intent_size)

    def init_weights(self):
        self.embedding.weight.data.uniform_(-0.1, 0.1)

    def Attention(self, hidden, encoder_outputs, encoder_maskings):
        """
        hidden : 1,B,D
        encoder_outputs : B,T,D
        encoder_maskings : B,T
        output : B,1,D
        """
        # (1,B,D) -> (B,D,1)
        hidden = hidden.squeeze(0).unsqueeze(2)
        # B
        batch_size = encoder_outputs.size(0)
        # T
        max_len = encoder_outputs.size(1)
        # (B*T,D) -> (B*T,D)
        energies = self.attn(
            encoder_outputs.contiguous().view(batch_size * max_len, -1))
        # (B*T,D) -> B,T,D
        energies = energies.view(batch_size, max_len, -1)
        # (B,T,D) * (B,D,1) -> (B,1,T)
        attn_energies = energies.bmm(hidden).transpose(1, 2)
        # PAD masking
        attn_energies = attn_energies.squeeze(
            1).masked_fill(encoder_maskings, -1e12)

        # B,T
        alpha = F.softmax(attn_energies)
        # B,1,T
        alpha = alpha.unsqueeze(1)
        # B,1,T * B,T,D => B,1,D
        context = alpha.bmm(encoder_outputs)

        return context

    def init_hidden(self, input):

        if USE_CUDA:
            hidden = torch.zeros(self.n_layers, input.size(
                0), self.hidden_size).cuda()
        else:
            hidden = Variable(torch.zeros(
                self.n_layers, input.size(0), self.hidden_size))

        if USE_CUDA:
            context = Variable(torch.zeros(
                self.n_layers, input.size(0), self.hidden_size)).cuda()
        else:
            context = Variable(torch.zeros(
                self.n_layers, input.size(0), self.hidden_size))

        return hidden, context

    def forward(self, input, context, encoder_outputs, encoder_maskings, training=True):
        """
        input : B,1
        context : B,1,D
        encoder_outputs : B,T,D
        output: B*T,slot_size  B,D
        """
        # B,1 -> B,1,D
        embedded = self.embedding(input)
        hidden = self.init_hidden(input)
        decode = []
        # B,T,D -> T,B,D
        aligns = encoder_outputs.transpose(0, 1)
        # T
        length = encoder_outputs.size(1)
        for i in range(length):
            # B,D -> B,1,D
            aligned = aligns[i].unsqueeze(1)
            _, hidden = self.lstm(
                torch.cat((embedded, context, aligned), 2), hidden)

            # for Intent Detection
            if i == 0:
                # 1,B,D
                intent_hidden = hidden[0].clone()
                # B,1,D
                intent_context = self.Attention(
                    intent_hidden, encoder_outputs, encoder_maskings)
                # 1,B,D
                concated = torch.cat(
                    (intent_hidden, intent_context.transpose(0, 1)), 2)
                # B,D
                intent_score = self.intent_out(concated.squeeze(0))

            # 1,B,D -> 1,B,2*D
            concated = torch.cat((hidden[0], context.transpose(0, 1)), 2)
            # B,slot_size
            score = self.slot_out(concated.squeeze(0))
            softmaxed = F.log_softmax(score)
            decode.append(softmaxed)
            # B
            _, input = torch.max(softmaxed, 1)
            # B,1 -> B,1,D
            embedded = self.embedding(input.unsqueeze(1))
            # B,1,D
            context = self.Attention(
                hidden[0], encoder_outputs, encoder_maskings)

        # B,slot_size*T
        slot_scores = torch.cat(decode, 1)

        # B*T,slot_size  B,D
        return slot_scores.view(input.size(0)*length, -1), intent_score
