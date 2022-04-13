import argparse
import json
import logging
import os
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import path
import seaborn as sns
import torch
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchsummary import summary

from src.dataset import SentenceDataset
from src.models import IntentSlotsClassifier, IntentClassifier, SlotsClassifier, VanillaEncoderDecoder
from src.utils import *
from src.args import get_eval_args

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True
logging.basicConfig(
    format="%(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
warnings.filterwarnings("ignore")


def main(args):

    separate = (args.model_type.lower() == 'separate')
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

    # Loading model using args.checkpoint
    if not separate:
        if args.model_type == 'joint':
            model, _, _, _ = load_ckp(args.ckpt_path, IntentSlotsClassifier(embedding_matrix.shape[0], len(intents_dict), len(
                slots_dict), embedding_matrix.shape[1], embedding_matrix, 128, 3, drop_prob=0.0), None, test=True)

            summary(model, input_size=(args.sequence_length,),
                    dtypes=torch.IntTensor)
        elif args.model_type == 'encdec':
            model, _, _, _ = load_ckp(args.ckpt_path, VanillaEncoderDecoder(embedding_matrix.shape[0], embedding_matrix.shape[1], embedding_matrix, 128, len(intents_dict),
                                                                            len(slots_dict), n_layers=1, dropout_rate=0.0), None, test=True)

            model.to(device)
            model.eval()

    else:
        model, _, _, _ = load_ckp(args.ckpt_path+'.intent', IntentClassifier(embedding_matrix.shape[0], len(intents_dict), len(
            slots_dict), embedding_matrix.shape[1], embedding_matrix, 128, 3, drop_prob=0.0), None, test=True)

        summary(model, input_size=(args.sequence_length,),
                dtypes=torch.IntTensor)
        model_slots, _, _, _ = load_ckp(args.ckpt_path+'.slots', SlotsClassifier(embedding_matrix.shape[0], len(intents_dict), len(
            slots_dict), embedding_matrix.shape[1], embedding_matrix, 128, 3, drop_prob=0.0), None, test=True)

        summary(model_slots, input_size=(
            args.sequence_length,), dtypes=torch.IntTensor)
        model_slots.to(device)
        model_slots.eval()

    model.to(device)
    model.eval()
    with torch.no_grad():
        if separate:
            final_evaluations = evaluate(model, test_loader, device,
                                         None, None, model2=model_slots, test=True)
        else:
            final_evaluations = evaluate(
                model, test_loader, device, None, None, test=True)
        # Init lists to store predictions
        # Sometimes slots are not represented at all so we need to know which are present to put in abscissa
    # which means we can't simply have x = range(num_slots)
    all_slots = set(final_evaluations['slots_gt'])
    all_slots.update(final_evaluations['slots_pred'])
    ##########

    # Compute scores

    f1_intent = final_evaluations['f1_intent']

    f1_slots = final_evaluations['f1_slots']

    times = final_evaluations['times']

    accu_intent = accuracy_score(
        final_evaluations['intent_gt'], final_evaluations['intent_pred'])

    accu_slots = accuracy_score(
        final_evaluations['slots_gt'], final_evaluations['slots_pred'])

    prec_intent = precision_score(
        final_evaluations['intent_gt'], final_evaluations['intent_pred'], average=None)

    prec_slots = precision_score(
        final_evaluations['slots_gt'], final_evaluations['slots_pred'], average=None)

    recall_intent = recall_score(
        final_evaluations['intent_gt'], final_evaluations['intent_pred'], average=None)

    recall_slots = recall_score(
        final_evaluations['slots_gt'], final_evaluations['slots_pred'], average=None)

    time_str = f'\nAverage inference time: {np.mean(times)}s \n'
    intent_str = f'Intent (Test set): \n\t f1 : {np.mean(f1_intent)} \n\t accuracy : {accu_intent} \n\t precision : {np.mean(prec_intent)} \n\t recall : {np.mean(recall_intent)} \n'
    slots_str = f'Slots (Test set): \n\t f1 : {np.mean(f1_slots)} \n\t accuracy : {accu_slots} \n\t precision : {np.mean(prec_slots)} \n\t recall : {np.mean(recall_slots)} \n'

    logging.info(time_str)
    logging.info(intent_str)
    logging.info(slots_str)

    with open(os.path.join(
            args.plot_dir, 'train', args.name + '-' + args.exp_number, 'test_metrics.txt'), 'w') as f:
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
    plt.savefig(os.path.join(args.plot_dir, 'train', args.name +
                '-' + args.exp_number, 'test_slots.png'))

    df_result_slots.plot.bar(x='slots', y='f1')
    plt.savefig(os.path.join(args.plot_dir, 'train', args.name +
                '-' + args.exp_number, 'test_slots_f1.png'))

    # Once again
    df_result_intent = pd.DataFrame()
    df_result_intent['index'] = list(set(final_evaluations['intent_gt']))
    df_result_intent['slots'] = df_result_intent['index'].apply(
        lambda x: intents_dict.to_dict()[0][x])
    df_result_intent['f1'] = f1_intent
    df_result_intent['prec'] = prec_intent
    df_result_intent['recall'] = recall_intent
    df_result_intent.plot.bar(x='slots', y=['f1', 'prec', 'recall'])
    plt.savefig(os.path.join(args.plot_dir, 'train',
                args.name + '-' + args.exp_number, 'test_intent.png'))

    # Using dataframe to get easy plotting with df.plot
    slots_plot_path = os.path.join(
        args.plot_dir, 'train', args.name + '-' + args.exp_number, 'test_slots_f1.png')
    plot_f1_prec_recall(list(all_slots), f1_slots, prec_slots,
                        recall_slots, slots_dict, slots_plot_path, True)

    intents_plot_path = os.path.join(
        args.plot_dir, 'train', args.name + '-' + args.exp_number, 'test_intent_f1.png')

    plot_f1_prec_recall(list(set(final_evaluations['intent_gt'])), f1_intent,
                        prec_intent, recall_intent, intents_dict, intents_plot_path, False)


def evaluate(model, data_loader, device, criterion, criterion2, model2=None, test=False):

    # Init variables and list to count / store predictions
    model.to(device)
    model.eval()
    correct_val_intent = 0
    correct_val_slots = 0
    total_val = 0
    total_val_slots = 0
    val_loss_inside_epoch = []
    val_intent_loss_inside_epoch = []
    val_slots_loss_inside_epoch = []
    val_intent_predictions = []
    val_slots_predictions = []
    intent_gt = []
    slots_gt = []
    times = []

    evaluation = {}

    for seq, gt_intent, gt_slots in data_loader:

        assert seq.shape[-1] == gt_slots.shape[-1], f'{seq.shape[-1]}, {gt_slots.shape[-1]}'

        seq = seq.to(device)
        gt_intent = gt_intent.to(device)
        gt_slots = gt_slots.to(device)

        _, val_intent = torch.max(gt_intent.data, 1)
        _, targets_slots_val = torch.max(gt_slots.data, 1)
        _, targets_intents_val = torch.max(gt_intent.data, 1)

        if test:
            t0 = time.time()
        # Forward pass
        if not model2:

            intent, slots = model(seq)
        else:
            intent = model(seq)
            slots = model2(seq)

        if test:
            times.append(time.time()-t0)

        # From predictions score to predictions
        _, predictions_intent = torch.max(intent.data, 1)
        _, predictions_slots_val = torch.max(slots.data, 1)

        # Storing predictions and ground truth
        val_intent_predictions += list(
            predictions_intent.detach().cpu().numpy())
        val_slots_predictions += list(
            predictions_slots_val.detach().cpu().numpy().flatten())
        intent_gt += list(targets_intents_val.detach().cpu().numpy())
        slots_gt += list(targets_slots_val.detach().cpu().numpy().flatten())

        # Computing loss
        if test:
            val_loss1 = 0
            val_loss2 = 0
            val_loss = 0
            val_loss_inside_epoch.append(0)
            if model2:
                val_intent_loss_inside_epoch.append(0)
                val_slots_loss_inside_epoch.append(0)
        else:
            val_loss1 = criterion(intent, targets_intents_val)

            val_loss2 = criterion2(slots, targets_slots_val)
            val_loss = val_loss1 + val_loss2

        # Storing loss
            val_loss_inside_epoch.append(val_loss.item())

            if model2:
                val_intent_loss_inside_epoch.append(val_loss1.item())
                val_slots_loss_inside_epoch.append(val_loss2.item())

        # Need to know how many predictions were made to divide for accuracy
        total_val += intent.size(0)
        total_val_slots += gt_slots.size(0) * gt_slots.size(2)

        correct_val_intent += (predictions_intent ==
                               targets_intents_val).sum().item()
        correct_val_slots += (predictions_slots_val ==
                              targets_slots_val).sum().item()
    valid_loss = np.mean(val_loss_inside_epoch)

    if model2:
        valid_intent_loss = np.mean(val_intent_loss_inside_epoch)
        valid_slots_loss = np.mean(val_slots_loss_inside_epoch)
        evaluation['valid_intent_loss'] = valid_intent_loss
        evaluation['valid_slots_loss'] = valid_slots_loss

    evaluation['intent_gt'] = intent_gt
    evaluation['slots_gt'] = slots_gt
    evaluation['intent_pred'] = val_intent_predictions
    evaluation['slots_pred'] = val_slots_predictions
    evaluation['num_correct_intent'] = correct_val_intent
    evaluation['num_correct_slots'] = correct_val_slots
    evaluation['valid_loss'] = valid_loss
    evaluation['total_val'] = total_val
    evaluation['total_val_slots'] = total_val_slots
    evaluation['f1_intent'] = f1_score(
        intent_gt, val_intent_predictions, average=None)
    evaluation['f1_slots'] = f1_score(
        slots_gt, val_slots_predictions, average=None)
    if test:
        evaluation['times'] = times
    model.train()
    if model2:
        model2.train()
    return evaluation


def evaluate_crf(model, data_loader, device, criterion, criterion2, test=False):

    # Init variables and list to count / store predictions
    model.to(device)
    model.eval()
    correct_val_intent = 0
    correct_val_slots = 0
    total_val = 0
    total_val_slots = 0
    val_loss_inside_epoch = []
    val_intent_loss_inside_epoch = []
    val_slots_loss_inside_epoch = []
    val_intent_predictions = []
    val_slots_predictions = []
    intent_gt = []
    slots_gt = []
    times = []

    evaluation = {}

    for seq, gt_intent, gt_slots in data_loader:

        assert seq.shape[-1] == gt_slots.shape[-1], f'{seq.shape[-1]}, {gt_slots.shape[-1]}'

        seq = seq.to(device)
        gt_intent = gt_intent.to(device)
        gt_slots = gt_slots.to(device)

        _, val_intent = torch.max(gt_intent.data, 1)
        _, targets_slots_val = torch.max(gt_slots.data, 1)
        _, targets_intents_val = torch.max(gt_intent.data, 1)

        if test:
            t0 = time.time()
        # Forward pass

        intent, predictions_slots_val, _, masks, features = model(seq)
        predictions_slots_val = [
            elm + [129] * (30 - len(elm)) for elm in predictions_slots_val]

        predictions_slots_val = torch.Tensor(
            predictions_slots_val).to(device)

        if test:
            times.append(time.time()-t0)

        # From predictions score to predictions
        _, predictions_intent = torch.max(intent.data, 1)

        # Storing predictions and ground truth
        val_intent_predictions += list(
            predictions_intent.detach().cpu().numpy())
        val_slots_predictions += list(
            predictions_slots_val.detach().cpu().numpy().flatten())
        intent_gt += list(targets_intents_val.detach().cpu().numpy())
        slots_gt += list(targets_slots_val.detach().cpu().numpy().flatten())

        # Computing loss
        if test:
            val_loss1 = 0
            val_loss2 = 0
            val_loss = 0
            val_loss_inside_epoch.append(0)
            if model2:
                val_intent_loss_inside_epoch.append(0)
                val_slots_loss_inside_epoch.append(0)
        else:
            val_loss1 = criterion(intent, targets_intents_val)

            val_loss2 = model.loss(features, masks, targets_slots_val)
            val_loss = val_loss1 + val_loss2/10

        # Storing loss
            val_loss_inside_epoch.append(val_loss.item())

        # Need to know how many predictions were made to divide for accuracy
        total_val += intent.size(0)
        total_val_slots += gt_slots.size(0) * gt_slots.size(2)

        correct_val_intent += (predictions_intent ==
                               targets_intents_val).sum().item()
        correct_val_slots += (predictions_slots_val ==
                              targets_slots_val).sum().item()
    valid_loss = np.mean(val_loss_inside_epoch)

    evaluation['intent_gt'] = intent_gt
    evaluation['slots_gt'] = slots_gt
    evaluation['intent_pred'] = val_intent_predictions
    evaluation['slots_pred'] = val_slots_predictions
    evaluation['num_correct_intent'] = correct_val_intent
    evaluation['num_correct_slots'] = correct_val_slots
    evaluation['valid_loss'] = valid_loss
    evaluation['total_val'] = total_val
    evaluation['total_val_slots'] = total_val_slots
    evaluation['f1_intent'] = f1_score(
        intent_gt, val_intent_predictions, average=None)
    evaluation['f1_slots'] = f1_score(
        slots_gt, val_slots_predictions, average=None)
    if test:
        evaluation['times'] = times
    model.train()

    return evaluation


def plot_f1_prec_recall(index, f1, prec, recall, dictionnary, path, slots=True):

    plot_type = "slots" if slots else "intent"
    df = pd.DataFrame()
    df['index'] = index
    df[plot_type] = df['index'].apply(lambda x: dictionnary.to_dict()[0][x])
    df['f1'] = f1
    df['prec'] = prec
    df['recall'] = recall
    df.plot.bar(x=plot_type, y=['f1', 'prec', 'recall'])
    plt.savefig(path)


if __name__ == '__main__':
    args = get_eval_args()
    save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    main(args)
