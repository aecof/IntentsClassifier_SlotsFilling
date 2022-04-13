import torch
from torch.autograd import Variable
import pickle
import random
import os
import pickle
import nltk
nltk.download('wordnet')
from nltk.stem.wordnet import WordNetLemmatizer 
wn_lemmatiser = WordNetLemmatizer()


USE_CUDA = torch.cuda.is_available()
flatten = lambda l: [item for sublist in l for item in sublist]
import pandas as pd

def init_train_df(path_intent, path_slots):
    df_train = pd.read_csv(path_intent, sep = '\t')

    df_train['sentence'] = df_train['sentence'].apply(lambda x : normalise(x))
    df_train['sentence'] = df_train['sentence'].apply(lambda x : ['<SOS>'] + x + ['<EOS>'])
    vocab = set()
    for list_str in df_train['sentence']:
        vocab.update(list_str)
    vocab.update(set(['<unk>']))
    stoi = {elm : i+1 for i,elm in enumerate(list(vocab))}
    df_train['slots'] = pd.read_csv(path_slots,sep='\t', header = None)[0]
    return df_train, vocab, stoi


def preprocess_data_from_nemo(df_train):
    
    train_data = []
    for tr in df_train.iterrows() :
        
        temp = Variable(torch.LongTensor([tr[1]['preprocessed_sentences']]))
        
        if USE_CUDA :
            temp = temp.cuda()
        
        temp1 = Variable(torch.LongTensor([tr[1]['preprocessed_slots']]))
        
        if USE_CUDA :
            temp1 = temp1.cuda()
            
        temp2 = Variable(torch.LongTensor([tr[1]['label']]))
        
        if USE_CUDA :
            temp2 = temp2.cuda()
            
        train_data.append((temp, temp1, temp2))
            

    print("Preprocessing complete!")
              
    return train_data

def preprocess_sentences(df, stoi, max_len):
    tmp = df['sentence'].apply(lambda x : word2idx(x, stoi))
    tmp = tmp.apply(lambda x : clip_pad(x, max_len))
    #print(tmp)
    return tmp

def preprocess_slots(df, max_len, pad):
    tmp = df['slots'].apply(lambda x : x.split(' '))
    tmp = tmp.apply(lambda x : [int(xs) for xs in x])
    tmp = tmp.apply(lambda x : clip_pad(x, max_len, pad))
    return tmp
    
    
def tokenise(sentence):
  tokens = ''.join([char if ord('a') <= ord(char.lower()) <= ord('z') or char.isdigit() else ' ' for char in f'{sentence} '.replace(':','').replace("`","'").replace('pm ',' pm ')])
  ts = []
  for token in tokens.split():
    if "am " in f'{token} ' and len(token) > 2 and token[-3].isdigit(): #avoid splitting words like ham, spam, sam, etc
      ts.extend([token[:-2],"am"])
    else:
      ts.append(token)
  return ts

def normalise(sentence): 
  return ["*" * len(token) if token.isdigit() else wn_lemmatiser.lemmatize(token.lower(),'v') for token in tokenise(sentence)] 



def one_hot_encoding(x, dimensions):
    res = np.zeros(dimensions, dtype = int)
    res[x] = 1
    return res

def clip_pad(x,length,pad=[0]):
    x = x[:length]   #clips it if too long
    x = x + pad * (length - len(x)) #pads it if too short
    return x



def word2idx(x, stoi):
    res = []
    for word in x :
        if stoi.get(word):
            idx = stoi.get(word)
        else :
            idx = stoi.get('<unk>')
        res.append(idx)
    return res