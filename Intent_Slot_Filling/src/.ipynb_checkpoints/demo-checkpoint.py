from model import Encoder, Decoder
import json
import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt

import torch

from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.utils.data._utils.collate import default_convert
from utils import *
from models import IntentClassifier, SlotsClassifier, IntentSlotsClassifier
from dataset import SentenceDataset
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
import argparse
import sys
from model import Encoder, Decoder

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True


def main(args):
    slots_dict = pd.read_csv(
        '../data_dir/dict.slots.csv', header=None)[0].to_dict()
    intents_dict = pd.read_csv(
        '../data_dir/dict.intents.csv', header=None)[0].to_dict()
    num_slots = len(slots_dict)

    with open('./stoi.json', 'r') as f:
        stoi = json.load(f)
    
    

    sentence = ' '.join(args.sentence)
    normalise_sentence = normalise(sentence)
    df = pd.DataFrame()
    df['sentence'] = [normalise_sentence]

    preprocessed_sentence = preprocess_sentences(df, stoi, max_len=12)

    preprocessed_sentence = torch.IntTensor(
        preprocessed_sentence.iloc[0]).view(1, -1).to(device)

    embedding_matrix = np.load('embedding_matrix.npy')

    if args.model_type == 'joint':
        ckpt_path = args.ckpt_path[0]
        model, _, _, _ = load_ckp(ckpt_path, IntentSlotsClassifier(embedding_matrix.shape[0], len(intents_dict), len(
            slots_dict), embedding_matrix.shape[1], embedding_matrix, 128, 3, drop_prob=0.0), None, test=True)
        model.to(device)
        model.eval()

        intent, slots = model(preprocessed_sentence)
        

    elif args.model_type == 'separate':
        ckpt_intent, ckpt_slots = args.ckpt_path
        model_intent, _, _, _ = load_ckp(ckpt_intent, IntentClassifier(
            embedding_matrix.shape[0], len(intents_dict), embedding_matrix.shape[1], embedding_matrix, 128, 3, drop_prob=0.0), None, test=True)

        model_intent.to(device)
        model_intent.eval()

        model_slots, _, _, _ = load_ckp(ckpt_slots, SlotsClassifier(
            embedding_matrix.shape[0], len(slots_dict), embedding_matrix.shape[1], embedding_matrix, 128, 3, drop_prob=0.0), None, test=True)
        model_slots.to(device)
        model_slots.eval()

        intent = model_intent(preprocessed_sentence)
        slots = model_slots(preprocessed_sentence)
        
    
    else : 
        ckpt_encoder, ckpt_decoder = args.ckpt_path
        
        encoder = Encoder(embedding_matrix.shape[0]+2, 50, 128)
        decoder = Decoder(len(slots_dict), len(intents_dict), len(slots_dict)//3, 128*2)
        encoder.load_state_dict(torch.load(ckpt_encoder))
        decoder.load_state_dict(torch.load(ckpt_decoder))
        encoder.eval()
        
        decoder.eval()
        encoder.to(device)
        decoder.to(device)
        with open('./stoi_joint_slot.json','r') as f:
            stoi_joint_slot = json.load(f)
        x = ['<SOS>'] + normalise_sentence + ['<EOS>']
        normalise_sentence = x
        df_joint_slot = pd.DataFrame()
        df_joint_slot['sentence'] = [x]
        x = preprocess_sentences(df_joint_slot, stoi_joint_slot, max_len=60)
        x = torch.IntTensor(
        x.iloc[0]).view(1, -1).to(device)
        x_mask = torch.cat([Variable(torch.ByteTensor(tuple(map(lambda s: s == 0, t.data)))).cuda() if USE_CUDA
                                    else Variable(torch.ByteTensor(tuple(map(lambda s: s == 0, t.data)))) for t in x])\
                    .view(1, -1)
        x_mask = x_mask > 0
        
        output, hidden_c = encoder(x,x_mask)
        if USE_CUDA:
            start_decode = Variable(torch.LongTensor([[1]*1])).cuda().transpose(1, 0)
        else:
            start_decode = Variable(torch.LongTensor([[1]*1])).transpose(1, 0) 
            
        slots, intent = decoder(start_decode, hidden_c, output, x_mask)
        print(slots, intent)
        
        
        
    _, predictions_intent = torch.max(intent.data, 1)
    _, predictions_slots = torch.max(slots.data, 1)   
        
        
        
    print(predictions_slots)

    print(sentence)
    print(normalise_sentence)
    print('Intent :', intents_dict[int(predictions_intent)])
    if args.model_type == 'attention':
        for i, elm in enumerate(predictions_slots):
            if i < len(normalise_sentence):
                print('\t', normalise_sentence[i], ':', slots_dict[int(elm)])
            else:
                print('\t', '<pad> :', slots_dict[int(elm)])
    else: 
        for i, elm in enumerate(predictions_slots[0]):
            if i < len(normalise_sentence):
                print('\t', normalise_sentence[i], ':', slots_dict[int(elm)])
            else:
                print('\t', '<pad> :', slots_dict[int(elm)])


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sentence', nargs='+')
    parser.add_argument(
        '--model_type', choices=['joint', 'separate', 'attention'], required=True)
    parser.add_argument('--ckpt_path', nargs='+')
    return parser.parse_args()


if __name__ == '__main__':

    args = argparser()
    main(args)
