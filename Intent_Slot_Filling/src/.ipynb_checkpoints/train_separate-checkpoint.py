import json
import torch
import pandas as pd
import re
import numpy as np
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import itertools
from torchsummary import summary
import warnings

from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.utils.data._utils.collate import default_convert
import torch.nn.functional as F
import torch.optim as optim
from prettytable import PrettyTable
from tqdm import tqdm
import shutil
import argparse
from utils import *
from losses import FocalLoss, dice_loss
from models import IntentClassifier, SlotsClassifier
from dataset import SentenceDataset
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
import logging


logging.basicConfig(
    format="%(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
warnings.filterwarnings("ignore")


def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--sequence_length", default=30, type=int)
    parser.add_argument("--epochs", default=20, type=int)
    parser.add_argument("--focal_loss", action='store_true')
    parser.add_argument("--dice_loss", action='store_true')
    parser.add_argument("--save_dir", default="plots")
    parser.add_argument("--exp_name", required=True)
    return parser.parse_args()


def train(batch_size, sequence_length, num_epochs, args, focal_loss=False):

    logging.info('logging data...')
    slots_dict = pd.read_csv('../data_dir/dict.slots.csv', header=None)
    intents_dict = pd.read_csv('../data_dir/dict.intents.csv', header=None)
    df_train, vocab, stoi = init_train_df(
        path_intent='../data_dir/train.tsv', path_slots='../data_dir/train_slots.tsv')

    with open('stoi.json', 'w') as f:
        json.dump(stoi, f)

    logging.info('preprocessing data...')

    df_train['preprocessed_sentences'] = preprocess_sentences(
        df_train, stoi, max_len=sequence_length)
    df_train['preprocessed_slots'] = preprocess_slots(df_train, max_len=sequence_length, pad=[list(
        slots_dict[0]).index('O')])  # We used the NeMo data-format, which uses 'O' as no slots label

    df_val = df_train.sample(frac=0.2)
    df_train = df_train.drop(df_val.index)

    embedding_matrix = get_embedding_matrix(
        vocab, stoi, glove_vectors_path='../vectors.txt')
    np.save('embedding_matrix.npy', embedding_matrix,
            fix_imports=True, allow_pickle=False)

    train_dataset = SentenceDataset(
        df_train, lambda x: x, lambda x: one_hot_encoding(x, len(
            intents_dict)), lambda x: np.array([one_hot_encoding(int(xs), num_slots) for xs in x])
    )
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = SentenceDataset(
        df_val, lambda x: x, lambda x: one_hot_encoding(x, len(intents_dict)), lambda x: np.array([
            one_hot_encoding(int(xs), num_slots) for xs in x])
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.backends.cudnn.benchmark = True

    # Defining HyperParameters
    vocab_size = len(vocab)+1  # Number of AA
    output_size = len(intents_dict)  # Number of targets
    num_slots = len(slots_dict)

    # HyperParameters we could optimize over if we had more time
    embedding_dim = 50
    hidden_dim = 128
    n_layers = 3

    # Instanciate the model
    logging.info('Loading model...')
    model = IntentClassifier(
        vocab_size, output_size, embedding_dim, embedding_matrix, hidden_dim, n_layers, drop_prob=0.2
    )
    model_slots = SlotsClassifier(
        vocab_size, num_slots, embedding_dim, embedding_matrix, hidden_dim, n_layers, drop_prob=0.2
    )
    logging.info('intent model summary :')
    summary(model, input_size=(sequence_length,), dtypes=torch.IntTensor)
    logging.info('slots model summary :')
    summary(model_slots, input_size=(sequence_length,), dtypes=torch.IntTensor)
    model.to(device)
    model_slots.to(device)

    # Define learning rate, could
    learning_rate = 0.001
    criterion = nn.CrossEntropyLoss()
    if focal_loss:
        criterion2 = FocalLoss()
        logging.info('Using FocalLoss instead of CrossEntropy')
    elif args.dice_loss:
        criterion2 = dice_loss
        logging.info('Using DiceLoss instead of CrossEntropy')
    else:
        criterion2 = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    optimizer_slots = torch.optim.Adam(
        model_slots.parameters(), lr=learning_rate)

    logging.info('Begin training..')
    # Train the model
    total_step = len(train_loader)
    valid_intent_loss_min = (
        np.Inf
    )  # Initialize a minimum loss such that, if it is lower, it is the best checkpoint and we save the model
    valid_slots_loss_min = (np.Inf)

    train_intent_loss_epoch = []
    val_intent_loss_epoch = []
    train_slots_loss_epoch = []
    val_slots_loss_epoch = []
    train_accuracy_epoch = []
    train_accuracy_slots_epoch = []
    val_accuracy_epoch = []
    val_accuracy_slots_epoch = []

    if not os.path.exists("checkpoints"):
        os.mkdir("checkpoints")
    if not os.path.exists(os.path.join("checkpoints", args.exp_name)):
        os.mkdir(os.path.join("checkpoints", args.exp_name))

    if focal_loss:
        checkpoint_slots_path = os.path.join(
            "checkpoints", args.exp_name, "last_slots_focal_loss.pt")
        best_slots_model_path = os.path.join(
            "checkpoints", args.exp_name, "best_slots_focal_loss.pt")
    else:
        checkpoint_slots_path = os.path.join(
            "checkpoints", args.exp_name, "last_slots.pt")
        best_slots_model_path = os.path.join(
            "checkpoints", args.exp_name, "best_slots.pt")

    checkpoint_intent_path = os.path.join(
        "checkpoints", args.exp_name, "last_intent.pt")
    best_intent_model_path = os.path.join(
        "checkpoints", args.exp_name, "best_intent.pt")

    for epoch in range(num_epochs):
        with tqdm(train_loader, unit="batch") as tepoch:
            model.train()
            model_slots.train()
            for sequence, gt_intents, gt_slots in tepoch:
                tepoch.set_description(f"Epoch {epoch}")

                seq = sequence.to(device)
                gt_intents = gt_intents.to(device)
                gt_slots = gt_slots.to(device)

                # print(images)
                # Forward pass
                intent = model(seq)  # slots B x L x S
                slots = model_slots(seq)

                _, predictions_intent = torch.max(intent.data, 1)
                _, predictions_slots = torch.max(slots.data, 1)

                _, target_intent = torch.max(gt_intents.data, 1)
                _, targets_slots = torch.max(gt_slots.data, 1)

                #print(slots.size(), targets_slots.size())
                loss1 = criterion(intent, target_intent)
                loss2 = criterion2(slots, targets_slots)

                correct_intent = (predictions_intent ==
                                  target_intent).sum().item()
                accuracy_intent = 1.0 * correct_intent / gt_intents.size(0)
                correct_slots = (predictions_slots ==
                                 targets_slots).sum().item()
                accuracy_slots = correct_slots / \
                    (gt_slots.size(0) * gt_slots.size(2))

                optimizer.zero_grad()
                loss1.backward()
                optimizer.step()

                optimizer_slots.zero_grad()
                loss2.backward()
                optimizer_slots.step()

                tepoch.set_postfix(loss_intent=loss1.item(), loss_slots=loss2.item(
                ), accuracy_intent=100.0 * accuracy_intent, accuracy_slots=100 * accuracy_slots)

            # At the end of each epoch, evaluate and store train_loss, val_loss, train_accuracy, val_accuracy

            train_accuracy_epoch.append(100.0 * accuracy_intent)
            train_accuracy_slots_epoch.append(100.0 * accuracy_slots)
            train_intent_loss_epoch.append(loss1.item())
            train_slots_loss_epoch.append(loss2.item())

            with torch.no_grad():

                model.eval()
                model_slots.eval()
                correct_val_intent = 0
                correct_val_slots = 0
                total_val = 0
                total_val_slots = 0
                val_intent_loss_inside_epoch = []
                val_slots_loss_inside_epoch = []
                val_intent_predictions = []
                val_slots_predictions = []
                intent_gt = []
                slots_gt = []

                for seq, gt_intent, gt_slots in val_loader:
                    seq = seq.to(device)
                    gt_intent = gt_intent.to(device)
                    gt_slots = gt_slots.to(device)

                    _, targets_slots_val = torch.max(gt_slots.data, 1)
                    _, targets_intents_val = torch.max(gt_intent.data, 1)

                    intent = model(seq)
                    slots = model_slots(seq)

                    _, predictions_intent = torch.max(intent.data, 1)
                    _, predictions_slots_val = torch.max(slots.data, 1)

                    val_intent_predictions += list(
                        predictions_intent.detach().cpu().numpy())
                    val_slots_predictions += list(
                        predictions_slots_val.detach().cpu().numpy().flatten())
                    intent_gt += list(targets_intents_val.detach().cpu().numpy())
                    slots_gt += list(targets_slots_val.detach().cpu().numpy().flatten())

                    _, predictions_slots_val = torch.max(slots.data, 1)
                    _, targets_slots_val = torch.max(gt_slots.data, 1)

                    val_loss1 = criterion(intent, targets_intents_val)
                    val_loss2 = criterion2(slots, targets_slots_val)

                    val_intent_loss_inside_epoch.append(val_loss1.item())
                    val_slots_loss_inside_epoch.append(val_loss2.item())

                    total_val += intent.size(0)
                    total_val_slots += gt_slots.size(0) * gt_slots.size(2)

                    correct_val_intent += (predictions_intent ==
                                           targets_intents_val).sum().item()
                    correct_val_slots += (predictions_slots_val ==
                                          targets_slots_val).sum().item()

                f1_intent = f1_score(
                    intent_gt, val_intent_predictions, average=None)
                f1_slots = f1_score(
                    slots_gt, val_slots_predictions, average=None)
                valid_intent_loss = np.mean(val_intent_loss_inside_epoch)
                valid_slots_loss = np.mean(val_slots_loss_inside_epoch)

                # Storing val loss and val accuracy
                val_intent_loss_epoch.append(valid_intent_loss)
                val_slots_loss_epoch.append(valid_slots_loss)

                val_accuracy_epoch.append(100 * correct_val_intent / total_val)
                val_accuracy_slots_epoch.append(
                    100 * correct_val_slots / total_val_slots)

                print(
                    "Epoch {}/{}, train_intent_loss = {:4f}, train_slots_loss = {:4f}, train_accuracy = {:4f}, val_intent_loss = {:4f}, val_slots_loss = {:4f}, val_f1_intent = {:4f}, val_f1_slots = {:4f}".format(
                        epoch,
                        num_epochs,
                        train_intent_loss_epoch[-1],
                        train_slots_loss_epoch[-1],
                        train_accuracy_epoch[-1],
                        val_intent_loss_epoch[-1],
                        val_slots_loss_epoch[-1],
                        f1_intent.mean(),
                        f1_slots.mean()
                    )
                )

            checkpoint_intent = {
                "epoch": epoch,
                "valid_loss_min": valid_intent_loss,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }

            checkpoint_slots = {
                "epoch": epoch,
                "valid_loss_min": valid_slots_loss,
                "state_dict": model_slots.state_dict(),
                "optimizer": optimizer_slots.state_dict(),
            }

            save_ckp(checkpoint_intent, False,
                     checkpoint_intent_path, best_intent_model_path)
            save_ckp(checkpoint_slots, False,
                     checkpoint_slots_path, best_slots_model_path)

            if valid_intent_loss <= valid_intent_loss_min:
                print(
                    "Validation intent loss decreased ({:.6f} --> {:.6f}).  Saving model ...".format(
                        valid_intent_loss_min, valid_intent_loss
                    )
                )
                # save checkpoint as best model
                save_ckp(checkpoint_intent, True,
                         checkpoint_intent_path, best_intent_model_path)
                valid_intent_loss_min = valid_intent_loss

            if valid_slots_loss <= valid_slots_loss_min:
                print(
                    "Validation slots loss decreased ({:.6f} --> {:.6f}).  Saving model ...".format(
                        valid_slots_loss_min, valid_slots_loss
                    )
                )
                # save checkpoint as best model
                save_ckp(checkpoint_slots, True,
                         checkpoint_slots_path, best_slots_model_path)
                valid_slots_loss_min = valid_slots_loss

    # One last evaluation with the best checkpoint
    logging.info(
        'training finished, evaluate on val set and save metrics plots')
    eval_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)
    model_intent, _, _, _ = load_ckp(best_intent_model_path, IntentClassifier(
        vocab_size, output_size, embedding_dim, embedding_matrix, hidden_dim, n_layers, drop_prob=0.0), optimizer)

    model_intent.to(device)
    model_intent.eval()

    model_slots, _, _, _ = load_ckp(best_slots_model_path, SlotsClassifier(
        vocab_size, num_slots, embedding_dim, embedding_matrix, hidden_dim, n_layers, drop_prob=0.0), optimizer_slots)
    model_slots.to(device)
    model_slots.eval()

    with torch.no_grad():

        val_intent_predictions = []
        val_slots_predictions = []
        intent_gt = []
        slots_gt = []

        for seq, gt_intent, gt_slots in eval_loader:
            seq = seq.to(device)
            gt_intent = gt_intent.to(device)
            gt_slots = gt_slots.to(device)

            intent = model_intent(seq)
            slots = model_slots(seq)

            _, predictions_intent = torch.max(intent.data, 1)
            _, predictions_slots_val = torch.max(slots.data, 1)
            _, targets_slots_val = torch.max(gt_slots.data, 1)
            _, targets_intents_val = torch.max(gt_intent.data, 1)

            val_intent_predictions += list(
                predictions_intent.detach().cpu().numpy())
            val_slots_predictions += list(
                predictions_slots_val.detach().cpu().numpy().flatten())
            intent_gt += list(targets_intents_val.detach().cpu().numpy())
            slots_gt += list(targets_slots_val.detach().cpu().numpy().flatten())

    all_slots = set(slots_gt)
    all_slots.update(val_slots_predictions)
    f1_intent = f1_score(intent_gt, val_intent_predictions, average=None)
    f1_slots = f1_score(slots_gt, val_slots_predictions, average=None)
    accu_intent = accuracy_score(intent_gt, val_intent_predictions)
    accu_slots = accuracy_score(slots_gt, val_slots_predictions)
    prec_intent = precision_score(
        intent_gt, val_intent_predictions, average=None)
    prec_slots = precision_score(slots_gt, val_slots_predictions, average=None)
    recall_intent = recall_score(
        intent_gt, val_intent_predictions, average=None)
    recall_slots = recall_score(slots_gt, val_slots_predictions, average=None)

    intent_str = f'Intent (Val set): \n\t f1 : {np.mean(f1_intent)} \n\t accuracy : {accu_intent} \n\t precision : {np.mean(prec_intent)} \n\t recall : {np.mean(recall_intent)} \n'
    slots_str = f'Slots (Val set): \n\t f1 : {np.mean(f1_slots)} \n\t accuracy : {accu_slots} \n\t precision : {np.mean(prec_slots)} \n\t recall : {np.mean(recall_slots)} \n'

    logging.info(intent_str)
    logging.info(slots_str)

    path_prefix = ''
    if focal_loss:
        path_prefix = 'focal_'

    metric_str_path = os.path.join(
        args.save_dir, args.exp_name, 'val_metrics.txt')

    with open(metric_str_path, 'w') as f:
        f.write(intent_str)
        f.write(slots_str)

    plt.plot(range(num_epochs), train_intent_loss_epoch,
             label='train intent loss')
    plt.plot(range(num_epochs), val_intent_loss_epoch,
             label='validation intent loss')
    plt.plot(range(num_epochs), train_slots_loss_epoch,
             label='train slots loss')
    plt.plot(range(num_epochs), val_slots_loss_epoch,
             label='validation slots loss')
    plt.xlabel('epochs')
    plt.legend()

    plt.savefig(os.path.join(args.save_dir, args.exp_name, path_prefix+'loss_epochs' +
                f'_{epochs}epochs_{sequence_length}_seqlength.png'))

    df_result_slots = pd.DataFrame()
    df_result_slots['index'] = list(all_slots)
    # print(slots_dict.to_dict())
    df_result_slots['slots'] = df_result_slots['index'].apply(
        lambda x: slots_dict.to_dict()[0][x])
    df_result_slots['f1'] = f1_slots
    df_result_slots['prec'] = prec_slots
    df_result_slots['recall'] = recall_slots
    df_result_slots.plot.bar(x='slots', y=['f1', 'prec', 'recall'])
    plt.savefig(os.path.join(args.save_dir, args.exp_name, path_prefix+'validation_slots_f1' +
                f'_{epochs}epochs_{sequence_length}_seqlength.png'))

    df_result_intent = pd.DataFrame()
    df_result_intent['index'] = list(set(intent_gt))
    # print(slots_dict.to_dict())
    df_result_intent['slots'] = df_result_intent['index'].apply(
        lambda x: intents_dict.to_dict()[0][x])
    df_result_intent['f1'] = f1_intent
    df_result_intent['prec'] = prec_intent
    df_result_intent['recall'] = recall_intent
    df_result_intent.plot.bar(x='slots', y=['f1', 'prec', 'recall'])
    plt.savefig(os.path.join(args.save_dir, args.exp_name, path_prefix+'validation_intent_f1' +
                f'_{epochs}epochs_{sequence_length}_seqlength.png'))


if __name__ == '__main__':
    args = parse_arg()
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    if os.path.exists(os.path.join(args.save_dir, args.exp_name)):
        logging.warning(
            'Provided existing experience name, files WILL BE OVERWRITTEN')
    else:
        os.mkdir(os.path.join(args.save_dir, args.exp_name))
    batch_size = args.batch_size
    epochs = args.epochs
    sequence_length = args.sequence_length
    focal_loss = args.focal_loss
    train(batch_size, sequence_length, epochs, args, focal_loss)
