import json
import torch
import pandas as pd
import re
import numpy as np
import os
import argparse
import time
#import category_encoders as ce
import pandas as pd
import seaborn as sns
import torch
import matplotlib.pyplot as plt
import torch
from torchsummary import summary
from torch import nn
from torch.nn.utils.rnn import pad_sequence, pad_packed_sequence, pack_padded_sequence
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from prettytable import PrettyTable
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)
from tqdm import tqdm
import shutil
import json
import itertools
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.utils.data._utils.collate import default_convert
from utils import *
from models import IntentClassifier, SlotsClassifier
from dataset import SentenceDataset

import logging
import warnings
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
    parser.add_argument('--stoi_path', default='./stoi.json')
    parser.add_argument('--test_path', default='../data_dir/test.tsv')
    parser.add_argument('--test_slots_path',
                        default='../data_dir/test_slots.tsv')
    parser.add_argument('--intent_ckpt_path', required=True)
    parser.add_argument('--slots_ckpt_path', required=True)
    parser.add_argument('--sequence_length', default=30,
                        help='max sequence length, HAS TO BE THE SAME THAN IN TRAINING')
    parser.add_argument('--save_dir', default='plots')
    parser.add_argument('--exp_name', required=True,
                        help='Prefix to make the difference between experiments')

    return parser.parse_args()


def evaluate_separate(args):

    #### Loading slots and intents dictionnaries ####
    slots_dict = pd.read_csv(args.dict_slots_path, header=None)
    intents_dict = pd.read_csv(args.dict_intent_path, header=None)

    num_slots = len(slots_dict)

    #### Opening stoi matrix ####
    with open(args.stoi_path, 'r') as f:
        stoi = json.load(f)

    #### Initializing test df ####
    df_test = init_test_df(path_intent=args.test_path,
                           path_slots=args.test_slots_path)

    df_test['preprocessed_sentences'] = preprocess_sentences(
        df_test, stoi, max_len=args.sequence_length)  # preprocessing sentences
    df_test['preprocessed_slots'] = preprocess_slots(df_test, max_len=args.sequence_length, pad=[list(
        slots_dict[0]).index('O')])  # Preprocessing slots

    # Loading glove embedding matrix computed in train
    embedding_matrix = np.load('embedding_matrix.npy')

    # Initialize Dataset and DataLoader
    test_dataset = SentenceDataset(
        df_test, lambda x: x, lambda x: one_hot_encoding(x, len(intents_dict)), lambda x: np.array([
            one_hot_encoding(int(xs), num_slots) for xs in x])
    )

    # Batch size = 1, we will also evaluate speed
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    model_intent, _, _, _ = load_ckp(args.intent_ckpt_path, IntentClassifier(
        embedding_matrix.shape[0], len(intents_dict), embedding_matrix.shape[1], embedding_matrix, 128, 3, drop_prob=0.0), None, test=True)

    summary(model_intent, input_size=(
        args.sequence_length,), dtypes=torch.IntTensor)

    model_intent.to(device)
    model_intent.eval()

    model_slots, _, _, _ = load_ckp(args.slots_ckpt_path, SlotsClassifier(
        embedding_matrix.shape[0], len(slots_dict), embedding_matrix.shape[1], embedding_matrix, 128, 3, drop_prob=0.0), None, test=True)
    summary(model_slots)

    model_slots.to(device)
    model_slots.eval()

    with torch.no_grad():

        test_intent_predictions = []
        test_slots_predictions = []
        intent_gt = []
        slots_gt = []
        times = []
        logging.info('Begin evaluation...')

        for seq, gt_intent, gt_slots in test_loader:
            seq = seq.to(device)
            gt_intent = gt_intent.to(device)
            gt_slots = gt_slots.to(device)

            t0 = time.time()
            intent = model_intent(seq)
            slots = model_slots(seq)
            times.append(time.time()-t0)

            _, predictions_intent = torch.max(intent.data, 1)
            _, predictions_slots_val = torch.max(slots.data, 1)
            _, targets_slots_val = torch.max(gt_slots.data, 1)
            _, targets_intents_val = torch.max(gt_intent.data, 1)

            test_intent_predictions += list(
                predictions_intent.detach().cpu().numpy())
            test_slots_predictions += list(
                predictions_slots_val.detach().cpu().numpy().flatten())
            intent_gt += list(targets_intents_val.detach().cpu().numpy())
            slots_gt += list(targets_slots_val.detach().cpu().numpy().flatten())

    all_slots = set(slots_gt)
    all_slots.update(test_slots_predictions)
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
        args.save_dir, args.exp_name, 'test_metrics.txt')

    with open(metric_str_path, 'w') as f:
        f.write(time_str)
        f.write(intent_str)
        f.write(slots_str)

    df_result_slots = pd.DataFrame()
    df_result_slots['index'] = list(all_slots)
    # print(slots_dict.to_dict())
    df_result_slots['slots'] = df_result_slots['index'].apply(
        lambda x: slots_dict.to_dict()[0][x])
    df_result_slots['f1'] = f1_slots
    df_result_slots['prec'] = prec_slots
    df_result_slots['recall'] = recall_slots
    df_result_slots.plot.bar(x='slots', y=['f1', 'prec', 'recall'])
    plt.savefig(os.path.join(args.save_dir,
                args.exp_name, 'test_slots_f1.png'))

    df_result_intent = pd.DataFrame()
    df_result_intent['index'] = list(set(intent_gt))
    # print(slots_dict.to_dict())
    df_result_intent['slots'] = df_result_intent['index'].apply(
        lambda x: intents_dict.to_dict()[0][x])
    df_result_intent['f1'] = f1_intent
    df_result_intent['prec'] = prec_intent
    df_result_intent['recall'] = recall_intent
    df_result_intent.plot.bar(x='slots', y=['f1', 'prec', 'recall'])
    plt.savefig(os.path.join(args.save_dir,
                args.exp_name, 'test_intent_f1.png'))


if __name__ == '__main__':
    args = argparser()
    save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    evaluate_separate(args)
