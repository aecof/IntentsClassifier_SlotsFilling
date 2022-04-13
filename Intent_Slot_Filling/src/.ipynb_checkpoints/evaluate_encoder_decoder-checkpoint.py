import argparse

import json
import logging
import os
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np

import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)
from torchsummary import summary
from models import Decoder, Encoder
from utils import *

logging.basicConfig(
    format="%(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
warnings.filterwarnings("ignore")

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dict_slots_path',
                        default='../data_dir/dict.slots.csv')
    parser.add_argument('--dict_intent_path',
                        default='../data_dir/dict.intents.csv')
    parser.add_argument('--stoi_path', default='./stoi_joint_slot.json')
    parser.add_argument('--test_path', default='../data_dir/test.tsv')
    parser.add_argument('--test_slots_path',
                        default='../data_dir/test_slots.tsv')
    parser.add_argument('--encoder_ckpt_path', required=True)
    parser.add_argument('--decoder_ckpt_path', required=True)
    parser.add_argument('--save_dir', default='plots')

    return parser.parse_args()


def evaluate_attention(args):

    #### Loading slots and intents dictionnaries ####
    slots_dict = pd.read_csv(args.dict_slots_path, header=None)
    intents_dict = pd.read_csv(args.dict_intent_path, header=None)

    num_slots = len(slots_dict)

    #### Opening stoi matrix ####
    with open(args.stoi_path, 'r') as f:
        stoi = json.load(f)

    #### Initializing test df ####
    df_test = init_test_df_joint_slot(path_intent=args.test_path,
                                      path_slots=args.test_slots_path)

    #### Preprocessing sentences and slots ####
    df_test['preprocessed_sentences'] = preprocess_sentences(
        df_test, stoi, max_len=60)  # preprocessing sentences
    df_test['preprocessed_slots'] = preprocess_slots(df_test, max_len=60, pad=[list(
        slots_dict[0]).index('O')])  # Preprocessing slots

    test_data = preprocess_data_from_nemo(df_test)

    ### Initializing a few variables ###
    vocab_length = len(stoi) + 1
    num_slots = len(slots_dict)
    num_intents = len(intents_dict)

    # Loading models

    encoder = Encoder(vocab_length, 50, 128)
    decoder = Decoder(num_slots, num_intents, num_slots//3, 128*2)
    logging.info('Encoder Summary')
    summary(encoder)
    logging.info('Decoder Summary')
    summary(decoder)

    if USE_CUDA:
        encoder = encoder.cuda()
        decoder = decoder.cuda()

    encoder.load_state_dict(torch.load(args.encoder_ckpt_path))
    decoder.load_state_dict(torch.load(args.decoder_ckpt_path))
    encoder.eval()
    decoder.eval()

    logging.info('Start evaluation...')
    with torch.no_grad():
        # init lists which will store predictions and ground truth
        test_intent_predictions = []
        test_slots_predictions = []
        intent_gt = []
        slots_gt = []
        times = []
        for i, batch in enumerate(getBatch(1, test_data)):
            x, y_1, y_2 = zip(*batch)
            x = torch.cat(x)
            gt_slots = torch.cat(y_1)
            gt_intent = torch.cat(y_2)

            # Init mask for encoder
            x_mask = torch.cat([Variable(torch.ByteTensor(tuple(map(lambda s: s == 0, t.data)))).cuda() if USE_CUDA
                                else Variable(torch.ByteTensor(tuple(map(lambda s: s == 0, t.data)))) for t in x]).view(1, -1)
            x_mask = x_mask > 0

            t0 = time.time()
            output, hidden_c = encoder(x, x_mask)

            if USE_CUDA:
                start_decode = Variable(torch.LongTensor(
                    [[1]*1])).cuda().transpose(1, 0)
            else:
                start_decode = Variable(torch.LongTensor(
                    [[1]*1])).transpose(1, 0)

            slots, intent = decoder(start_decode, hidden_c, output, x_mask)
            times.append(time.time()-t0)

            # From predictions score get predictions
            _, predictions_intent = torch.max(intent.data, 1)
            _, predictions_slots_val = torch.max(slots.data, 1)

            # Simply renaming because code below is copied/pasted from other evaluate file
            targets_slots_val = gt_slots
            targets_intents_val = gt_intent.data

            # Storing all predictions and ground truth into lists

            test_intent_predictions += list(
                predictions_intent.detach().cpu().numpy())
            test_slots_predictions += list(
                predictions_slots_val.detach().cpu().numpy().flatten())
            intent_gt += list(targets_intents_val.detach().cpu().numpy())
            slots_gt += list(targets_slots_val.detach().cpu().numpy().flatten())

    # Sometimes slots are not represented at all so we need to know which are present to put in abscissa
    # which means we can't simply have x = range(num_slots)
    all_slots = set(slots_gt)
    all_slots.update(test_slots_predictions)
    ##########

    # Compute scores
    f1_intent = f1_score(intent_gt, test_intent_predictions, average=None)

    f1_slots = f1_score(slots_gt, test_slots_predictions, average=None)

    accu_intent = accuracy_score(intent_gt, test_intent_predictions)

    accu_slots = accuracy_score(slots_gt, test_slots_predictions)

    prec_intent = precision_score(
        intent_gt, test_intent_predictions, average=None)

    prec_slots = precision_score(
        slots_gt, test_slots_predictions, average=None)

    recall_intent = recall_score(
        intent_gt, test_intent_predictions, average=None)

    recall_slots = recall_score(slots_gt, test_slots_predictions, average=None)

    time_str = f'\nAverage inference time: {np.mean(times)}s \n'
    intent_str = f'Intent : \n\t f1 : {np.mean(f1_intent)} \n\t accuracy : {accu_intent} \n\t precision : {np.mean(prec_intent)} \n\t recall : {np.mean(recall_intent)} \n'
    slots_str = f'Slots : \n\t f1 : {np.mean(f1_slots)} \n\t accuracy : {accu_slots} \n\t precision : {np.mean(prec_slots)} \n\t recall : {np.mean(recall_slots)} \n'

    logging.info(time_str)
    logging.info(intent_str)
    logging.info(slots_str)

    metric_str_path = os.path.join(
        'plots', 'attention_test_metrics.txt')

    with open(metric_str_path, 'w') as f:
        f.write(time_str)
        f.write(intent_str)
        f.write(slots_str)

    # Putting scores into a dataframe to get an easy plot with df.plot..
    df_result_slots = pd.DataFrame()
    df_result_slots['index'] = list(all_slots)
    df_result_slots['slots'] = df_result_slots['index'].apply(
        lambda x: slots_dict.to_dict()[0][x])
    df_result_slots['f1'] = f1_slots
    df_result_slots['prec'] = prec_slots
    df_result_slots['recall'] = recall_slots
    df_result_slots.plot.bar(x='slots', y=['f1', 'prec', 'recall'])
    plt.savefig(os.path.join('plots', 'attention_test_slots_f1precrecall.png'))

    # once again
    df_result_intent = pd.DataFrame()
    df_result_intent['index'] = list(set(intent_gt))

    df_result_intent['slots'] = df_result_intent['index'].apply(
        lambda x: intents_dict.to_dict()[0][x])

    df_result_intent['f1'] = f1_intent
    df_result_intent['prec'] = prec_intent
    df_result_intent['recall'] = recall_intent
    df_result_intent.plot.bar(x='slots', y=['f1', 'prec', 'recall'])
    plt.savefig(os.path.join(
        'plots', 'attention_test_intent_f1precrecall.png'))


if __name__ == '__main__':
    args = argparser()
    evaluate_attention(args)
