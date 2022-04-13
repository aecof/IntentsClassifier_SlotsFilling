import os
import random
import pickle
from torch.autograd import Variable
import shutil
import torch
import pandas as pd
import numpy as np
import unidecode


#### from https://github.com/mohammedterry/slots_intents/blob/master/intents_slots/utils.py ####
def tokenise(sentence):
    """Takes a sentence as input, does lowercase, remove some punctuation, handles hours written as hh:mmAM (originally made for ATIS dataset)

    Args:
        sentence (str): input sentence

    Returns:
        list: tokenises sentence
    """

    tokens = ''.join([unidecode.unidecode(char) if ord('a') <= ord(unidecode.unidecode(char).lower()) <= ord('z') or char.isdigit(
    ) or char == "'" else ' ' for char in f'{sentence} '.replace(':', '').replace("`", "'").replace('pm ', ' pm ')])
    ts = []
    for token in tokens.split():
        # avoid splitting words like ham, spam, sam, etc
        if "am " in f'{token} ' and len(token) > 2 and token[-3].isdigit():
            ts.extend([token[:-2], "am"])
        else:
            ts.append(token)
    return ts

#### https://github.com/mohammedterry/slots_intents/blob/master/intents_slots/utils.py ####


def normalise(sentence):
    """Takes an input sentence, tokenise it and use WordNet Lemmatizer to produce lemmas for all tokens

    Args:
        sentence (str): input sentence

    Returns:
        list: preprocessed sentence with lemmas and good tokens
    """
    return ["*" * len(token) if token.isdigit() else token for token in tokenise(sentence)]


# Source : https://towardsdatascience.com/how-to-save-and-load-a-model-in-pytorch-with-a-complete-example-c2920e617dee
def save_ckp(state, is_best, checkpoint_path, best_model_path):
    """
    state: checkpoint we want to save
    is_best: is this the best checkpoint; min validation loss
    checkpoint_path: path to save checkpoint
    best_model_path: path to save best model
    """
    f_path = checkpoint_path
    # save checkpoint data to the path given, checkpoint_path
    torch.save(state, f_path)
    # if it is a best model, min validation loss
    if is_best:
        best_fpath = best_model_path
        # copy that checkpoint file to best path given, best_model_path
        shutil.copyfile(f_path, best_fpath)

# Source : https://towardsdatascience.com/how-to-save-and-load-a-model-in-pytorch-with-a-complete-example-c2920e617dee


def load_ckp(checkpoint_fpath, model, optimizer, test=False):
    """
    checkpoint_path: path to save checkpoint
    model: model that we want to load checkpoint parameters into
    optimizer: optimizer we defined in previous training
    """
    # load check point
    checkpoint = torch.load(checkpoint_fpath)
    # initialize state_dict from checkpoint to model

    model.load_state_dict(checkpoint["state_dict"])
    # initialize optimizer from checkpoint to optimizer
    if not test:
        optimizer.load_state_dict(checkpoint["optimizer"])
    # initialize valid_loss_min from checkpoint to valid_loss_min
        valid_loss_min = checkpoint["valid_loss_min"]
    # return model, optimizer, epoch value, min validation loss
        return model, optimizer, checkpoint["epoch"], valid_loss_min.item()
    else:
        return model, None, None, None


def one_hot_encoding(x, dimensions):
    """ returns one hot encoding of a label given the number of labels"""
    res = np.zeros(dimensions, dtype=int)
    res[x] = 1
    return res


def clip_pad(x, length, pad=[0]):
    """ pad/clip a input sequence given the maximum 'length' allowed and the padding token"""
    x = x[:length]  # clips it if too long
    x = x + pad * (length - len(x))  # pads it if too short
    return x


def get_save_dir(base_dir, name, training, id_max=100):
    """Get a unique save directory by appending the smallest positive integer
    `id < id_max` that is not already taken (i.e., no dir exists with that id).
    Args:
        base_dir (str): Base directory in which to make save directories.
        name (str): Name to identify this training run. Need not be unique.
        training (bool): Save dir. is for training (determines subdirectory).
        id_max (int): Maximum ID number before raising an exception.
    Returns:
        save_dir (str): Path to a new directory with a unique name.
    """
    for uid in range(1, id_max):
        subdir = 'train' if training else 'test'
        save_dir = os.path.join(base_dir, subdir, f'{name}-{uid:02d}')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            return save_dir

    raise RuntimeError('Too many save directories created with the same name. \
                       Delete old save directories or use another name.')


def clip_pad_by_group(x, pad=[0]):
    """ pad/clip a input sequence given the maximum 'length' allowed and the padding token"""

    x, group = x[0], x[1]

    if group == 'short':
        length = 8

    elif group == 'medium':
        length = 12

    else:
        length = 30

    x = x[:length]  # clips it if too long
    x = x + pad * (length - len(x))  # pads it if too short

    return x


def get_embedding_matrix(vocab, stoi, glove_vectors_path, embedding_dim=50):
    """Returns the embedding matrix of our models given by the vocab set of all tokens, the word2idx (stoi) matrix, and glove vectors computed by glove

    Args:
        vocab (set): set of all tokens found in the training dataset,
        stoi (dict): string to int dictionnary (same as word2idx)
        glove_vectors_path (path): path to the Glove words vectors

    Returns:
        np.array: embedding matrix 
    """
    embeddings_index = {}
    f = open(glove_vectors_path)
    vocab_len = len(vocab)
    for line in f:
        values = line.split(' ')
        word = values[0]  # The first entry is the word
        # These are the vecotrs representing the embedding for the word
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    print('GloVe data loaded')
    embedding_matrix = np.zeros((vocab_len+1, embedding_dim))
    for word in vocab:
        embedding_matrix[stoi[word]] = embeddings_index[word]

    # <pad> is not in vocab or in glove word vectors, so I initialize it randomly.
    embedding_matrix[0] = .4*np.random.randn(embedding_dim)
    # TODO : should find a better way to initialize pad vector
    # It is okay though because embedding matrix will still be trainable, and pad does not provide any info...
    return embedding_matrix


def group_sentence_by_length(x):
    if x < 8:
        return 'short'
    elif x < 12:
        return 'medium'
    else:
        return 'long'


def init_train_df(path_intent, path_slots):
    """Takes train.tsv and train_slots.tsv paths and returns some merely preprocessed useful objects

    Args:
        path_intent (str): path to train.tsv 
        path_slots (str): path to train_slots.tsv

    Returns:
        dataframe: df_train, with normalised input sentences (using normalise(x) above)
        set : vocab, containing all tokens in train after normalisation
        dict : stoi, string2idx dictionnary
    """

    df_train = pd.read_csv(path_intent, sep='\t')
    df_train['sentence'] = df_train['sentence'].apply(lambda x: normalise(x))
    df_train['sentence_group'] = df_train['sentence'].apply(
        lambda x: group_sentence_by_length(len(x)))

    vocab = set()
    for list_str in df_train['sentence']:
        vocab.update(list_str)
    vocab.update(set(['<unk>']))

    stoi = {elm: i+1 for i, elm in enumerate(sorted(list(vocab)))}

    df_train['slots'] = pd.read_csv(path_slots, sep='\t', header=None)[0]

    return df_train, vocab, stoi


def init_test_df(path_intent, path_slots):
    """Same as init_train_df but does not return vocab and stoi

    Args:
        path_intent (str): path to test.tsv
        path_slots (str): path to test_slots.tsv

    Returns:
        dataframe: test dataframe
    """

    df = pd.read_csv(path_intent, sep='\t')
    df['sentence'] = df['sentence'].apply(lambda x: normalise(x))
    df['slots'] = pd.read_csv(path_slots, sep='\t', header=None)[0]

    return df


def word2idx(x, stoi):
    """ given a sentence and stoi dict, returns word2idx sentence"""
    res = []
    for word in x:
        if stoi.get(word):
            idx = stoi.get(word)
        else:
            idx = stoi.get('<unk>')
        res.append(idx)
    return res


def preprocess_sentences(df, stoi, max_len):
    """ takes a sentence in, apply word2idx and clip/pad"""
    tmp = df['sentence'].apply(lambda x: word2idx(x, stoi))
    tmp = tmp.apply(lambda x: clip_pad(x, max_len))
    return tmp


def preprocess_one_sentence(sent, stoi, max_len = 30):
    """ takes a sentence in, apply word2idx"""
    sent = normalise(sent)
    return clip_pad(word2idx(sent, stoi), max_len)


def preprocess_sentences_by_group(df, stoi):
    """ takes a sentence in, apply word2idx and clip/pad"""

    tmp = df.copy()
    tmp['sentence_to_idx'] = tmp['sentence'].apply(lambda x: word2idx(x, stoi))

    tmp['padded_w2idx'] = tmp[['sentence_to_idx', 'sentence_group']].apply(
        lambda x: clip_pad_by_group(x), axis=1)
    return tmp['padded_w2idx']


def preprocess_slots_by_group(df, pad):
    """ takes a slot sequence, apply clip/pad """

    tmp = df.copy()
    tmp['split_slots'] = tmp['slots'].apply(lambda x: x.split(' '))
    tmp['split_splots'] = tmp['split_slots'].apply(
        lambda x: [int(xs) for xs in x])
    tmp['to_return'] = tmp[['split_slots', 'sentence_group']].apply(
        lambda x: clip_pad_by_group(x, pad), axis=1)
    return tmp['to_return']


def preprocess_slots(df, max_len, pad):
    """ takes a slot sequence, apply clip/pad """

    tmp = df['slots'].apply(lambda x: x.split(' '))
    tmp = tmp.apply(lambda x: [int(xs) for xs in x])
    tmp = tmp.apply(lambda x: clip_pad(x, max_len, pad))
    return tmp


def init_train_df_joint_slot(path_intent, path_slots):
    """ Same as init_train_df above, except applies some specific preprocessing needed for attention based model"""
    df_train = pd.read_csv(path_intent, sep='\t')

    df_train['sentence'] = df_train['sentence'].apply(lambda x: normalise(x))
    df_train['sentence'] = df_train['sentence'].apply(
        lambda x: ['<SOS>'] + x + ['<EOS>'])
    vocab = set()
    for list_str in df_train['sentence']:
        vocab.update(list_str)
    vocab.update(set(['<unk>']))
    stoi = {elm: i+1 for i, elm in enumerate(sorted(list(vocab)))}
    df_train['slots'] = pd.read_csv(path_slots, sep='\t', header=None)[0]
    return df_train, vocab, stoi


def init_test_df_joint_slot(path_intent, path_slots):
    """Same as init_train_df_joint_slot but does not return vocab and stoi

    Args:
        path_intent (str): path to test.tsv
        path_slots (str): path to test_slots.tsv

    Returns:
        dataframe: test dataframe
    """

    df = pd.read_csv(path_intent, sep='\t')
    df['sentence'] = df['sentence'].apply(lambda x: normalise(x))
    df['sentence'] = df['sentence'].apply(
        lambda x: ['<SOS>'] + x + ['<EOS>'])
    df['slots'] = pd.read_csv(path_slots, sep='\t', header=None)[0]

    return df


def preprocess_data_from_nemo(df_train):
    """ Takes a dataframe and output train data in the format required by the attention based model """

    train_data = []
    for tr in df_train.iterrows():

        temp = Variable(torch.LongTensor([tr[1]['preprocessed_sentences']]))

        if USE_CUDA:
            temp = temp.cuda()

        temp1 = Variable(torch.LongTensor([tr[1]['preprocessed_slots']]))

        if USE_CUDA:
            temp1 = temp1.cuda()

        temp2 = Variable(torch.LongTensor([tr[1]['label']]))

        if USE_CUDA:
            temp2 = temp2.cuda()

        train_data.append((temp, temp1, temp2))

    print("Preprocessing complete!")

    return train_data

#############################################################################################################
##Everything below this line is directly taken from : https://github.com/pengshuang/Joint-Slot-Filling.git###
#############################################################################################################


USE_CUDA = torch.cuda.is_available()
def flatten(l): return [item for sublist in l for item in sublist]


def prepare_sequence(seq, to_ix):
    idxs = list(
        map(lambda w: to_ix[w] if w in to_ix.keys() else to_ix["<UNK>"], seq))
    tensor = Variable(torch.LongTensor(idxs)).cuda(
    ) if USE_CUDA else Variable(torch.LongTensor(idxs))
    return tensor


def preprocessing(file_path, length):
    """
    atis-2.train.w-intent.iob
    """
    processed_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "data/")
    print("processed_data_path : %s" % processed_path)

    if os.path.exists(os.path.join(processed_path, "processed_train_data.pkl")):
        train_data, word2index, tag2index, intent2index = pickle.load(open(os.path.join(processed_path,
                                                                                        "processed_train_data.pkl"),
                                                                           "rb"))
        return train_data, word2index, tag2index, intent2index

    if not os.path.exists(processed_path):
        os.makedirs(processed_path)

    try:
        train = open(file_path, "r").readlines()
        print("Successfully load data. # of set : %d " % len(train))
    except:
        print("No such file!")
        return None, None, None, None

    try:
        train = [t[:-1] for t in train]
        train = [[t.split("\t")[0].split(" "), t.split("\t")[1].split(
            " ")[:-1], t.split("\t")[1].split(" ")[-1]] for t in train]
        train = [[t[0][1:-1], t[1][1:], t[2]] for t in train]

        seq_in, seq_out, intent = list(zip(*train))
        vocab = set(flatten(seq_in))
        slot_tag = set(flatten(seq_out))
        intent_tag = set(intent)
        print("# of vocab : {vocab}, # of slot_tag : {slot_tag}, # of intent_tag : {intent_tag}"
              .format(vocab=len(vocab), slot_tag=len(slot_tag), intent_tag=len(intent_tag)))
    except:
        return None, None, None, None

    sin = []
    sout = []

    for i in range(len(seq_in)):
        temp = seq_in[i]
        if len(temp) < length:
            temp.append('<EOS>')
            while len(temp) < length:
                temp.append('<PAD>')
        else:
            temp = temp[:length]
            temp[-1] = '<EOS>'
        sin.append(temp)

        temp = seq_out[i]
        if len(temp) < length:
            while len(temp) < length:
                temp.append('<PAD>')
        else:
            temp = temp[:length]
            temp[-1] = '<EOS>'
        sout.append(temp)

    word2index = {'<PAD>': 0, '<UNK>': 1, '<SOS>': 2, '<EOS>': 3}
    for token in vocab:
        if token not in word2index.keys():
            word2index[token] = len(word2index)

    tag2index = {'<PAD>': 0}
    for tag in slot_tag:
        if tag not in tag2index.keys():
            tag2index[tag] = len(tag2index)

    intent2index = {}
    for ii in intent_tag:
        if ii not in intent2index.keys():
            intent2index[ii] = len(intent2index)

    train = list(zip(sin, sout, intent))

    train_data = []

    for tr in train:

        temp = prepare_sequence(tr[0], word2index)
        temp = temp.view(1, -1)

        temp2 = prepare_sequence(tr[1], tag2index)
        temp2 = temp2.view(1, -1)

        temp3 = Variable(torch.LongTensor([intent2index[tr[2]]])).cuda(
        ) if USE_CUDA else Variable(torch.LongTensor([intent2index[tr[2]]]))

        train_data.append((temp, temp2, temp3))

    pickle.dump((train_data, word2index, tag2index, intent2index), open(
        os.path.join(processed_path, "processed_train_data.pkl"), "wb"))
    pickle
    print("Preprocessing complete!")

    return train_data, word2index, tag2index, intent2index


def getBatch(batch_size, train_data):
    random.shuffle(train_data)
    sindex = 0
    eindex = batch_size
    while eindex < len(train_data):
        batch = train_data[sindex:eindex]
        temp = eindex
        eindex = eindex+batch_size
        sindex = temp

        yield batch


def load_dictionary():

    processed_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "data/")

    if os.path.exists(os.path.join(processed_path, "processed_train_data.pkl")):
        _, word2index, tag2index, intent2index \
            = pickle.load(open(os.path.join(processed_path, "processed_train_data.pkl"), "rb"))
        return word2index, tag2index, intent2index
    else:
        print("Please, preprocess data first")
        return None, None, None
