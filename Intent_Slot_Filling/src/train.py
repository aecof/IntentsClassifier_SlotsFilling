import argparse
import json
import logging
import warnings
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.utils.data._utils.collate import default_convert
from torchsummary import summary
from tqdm import tqdm

from src.dataset import SentenceDataset
from src.losses import FocalLoss, dice_loss
from src.models import IntentSlotsClassifier, IntentClassifier, SlotsClassifier, VanillaEncoderDecoder, CRFEncoderDecoder
from src.utils import *
from src.args import get_train_args
from src.evaluate import evaluate, plot_f1_prec_recall, evaluate_crf

logging.basicConfig(
    format="%(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
warnings.filterwarnings("ignore")


def main(batch_size, sequence_length, num_epochs, args, focal_loss=False):

    separate = (args.model_type.lower() == 'separate')
    logging.info('loading data...')
    # Loading slots and intent dicts

    embedding_dim = 300

    if args.dataset == 'atis':
        slots_dict = pd.read_csv(
            '../data_dir/atis.slots.dict.txt', header=None)
        intents_dict = pd.read_csv(
            '../data_dir/atis.intent.dict.txt', header=None)

        # Init df_train, vocab, string2idx dict
        df_train, vocab, stoi = init_train_df(
            path_intent='../data_dir/train_atis.tsv', path_slots='../data_dir/train_slots_atis.tsv')

        # Saving stoi because we need it for eval and demo
        with open('stoi.json', 'w') as f:
            json.dump(stoi, f)

        logging.info('preprocessing data...')
        # preprocessing sentences and slots
        # df_val, _, _ = init_train_df(
        #    path_intent='../data_dir/dev_atis.tsv', path_slots='../data_dir/dev_slots_atis.tsv')

        df_train['preprocessed_sentences'] = preprocess_sentences(
            df_train, stoi, max_len=sequence_length)
        df_train['preprocessed_slots'] = preprocess_slots(df_train, max_len=sequence_length, pad=[list(
            slots_dict[0]).index('O')])  # We used the NeMo data-format, which uses 'O' as no slots label

        # df_val['preprocessed_sentences'] = preprocess_sentences(
        #    df_val, stoi, max_len=sequence_length)
        # df_val['preprocessed_slots'] = preprocess_slots(df_val, max_len=sequence_length, pad=[list(
        #    slots_dict[0]).index('O')])  # We used the NeMo data-format, which uses 'O' as no slots label

        df_val = df_train.sample(frac=0.2, random_state=42)
        df_train = df_train.drop(df_val.index)
        # Loading embedding matrix from GloVe Vectors
        embedding_matrix = get_embedding_matrix(
            vocab, stoi, glove_vectors_path='../data_dir/vectors_atis.txt', embedding_dim=embedding_dim)
        # Saving it in case we need it
        np.save('embedding_matrix.npy', embedding_matrix,
                fix_imports=True, allow_pickle=False)

    # Init Datasets and DataLoaders
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

    # init cuda related vars
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.backends.cudnn.benchmark = True

    # Defining HyperParameters
    vocab_size = len(vocab)+1  # Number of tokens
    output_size = len(intents_dict)  # Number of targets
    num_slots = len(slots_dict)

    # HyperParameters we could optimize over if we had more time
    embedding_dim = 300
    hidden_dim = 128
    n_layers = 3

    if not separate:

        if args.model_type == 'joint':
            # Instanciate the model
            logging.info('Loading model...')
            model = IntentSlotsClassifier(
                vocab_size, output_size, num_slots, embedding_dim, embedding_matrix, hidden_dim, n_layers, drop_prob=0.2
            )
            logging.info('model summary')
            summary(model, input_size=(sequence_length,),
                    dtypes=torch.IntTensor)
            model.to(device)
            model.train()
        if args.model_type == 'encdec':
            model = VanillaEncoderDecoder(
                vocab_size, embedding_dim, embedding_matrix, hidden_dim, output_size, num_slots, n_layers=1)
            model.to(device)
            model.train()

        if args.model_type == 'crfencdec':
            model = CRFEncoderDecoder(
                vocab_size, embedding_dim, embedding_matrix, hidden_dim, output_size, num_slots, n_layers=1)
            model.to(device)
            model.crf.to(device)
            model.train()

    else:
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
        summary(model_slots, input_size=(
            sequence_length,), dtypes=torch.IntTensor)
        model.to(device)
        model_slots.to(device)
        model.train()
        model_slots.train()

    # Define learning rate
    learning_rate = 0.001

    if args.model_type == 'crfencdec':
        learning_rate = 0.01

    # Init losses
    criterion = nn.CrossEntropyLoss()
    if focal_loss:
        criterion2 = FocalLoss()
        logging.info('Using FocalLoss instead of CrossEntropy')
    elif args.dice_loss:
        criterion2 = dice_loss
        logging.info('Using DiceLoss instead of CrossEntropy')
    else:
        criterion2 = nn.CrossEntropyLoss()

    # Init optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    if separate:
        optimizer_slots = torch.optim.Adam(
            model_slots.parameters(), lr=learning_rate)
    # Train the model
    total_step = len(train_loader)
    valid_loss_min = (
        np.Inf
    )  # Initialize a minimum loss such that, if it is lower, it is the best checkpoint and we save the model

    # Init lists to store loss and accuracy for plots later
    train_loss_epoch = []
    train_slots_loss_epoch = []
    train_intent_loss_epoch = []
    val_loss_epoch = []
    if separate:
        val_slots_loss_epoch = []
        val_intent_loss_epoch = []
        valid_loss_min_slots = np.Inf
    train_accuracy_epoch = []
    train_accuracy_slots_epoch = []
    val_accuracy_epoch = []
    val_accuracy_slots_epoch = []

    checkpoint_path = os.path.join(args.ckpt_dir, "last.pt")
    best_model_path = os.path.join(args.ckpt_dir, "best.pt")

    logging.info('Begin training..')
    for epoch in range(num_epochs):
        with tqdm(train_loader, unit="batch") as tepoch:
            for sequence, gt_intents, gt_slots in tepoch:
                tepoch.set_description(f"Epoch {epoch}")

                seq = sequence.to(device)
                gt_intents = gt_intents.to(device)
                gt_slots = gt_slots.to(device)

                # Forward pass
                if not separate:

                    if args.model_type == 'crfencdec':
                        intent, predictions_slots, _, masks, features = model(
                            seq)
                        predictions_slots = [
                            elm + [129] * (30 - len(elm)) for elm in predictions_slots]

                        predictions_slots = torch.Tensor(
                            predictions_slots).to(device)
                    else:
                        intent, slots = model(seq)  # slots B x L x S
                else:
                    intent = model(seq)
                    slots = model_slots(seq)

                # From predictions scores to predictions
                _, predictions_intent = torch.max(intent.data, 1)

                if args.model_type != 'crfencdec':
                    _, predictions_slots = torch.max(slots.data, 1)

                else:
                    pass

                _, target_intent = torch.max(gt_intents.data, 1)
                _, targets_slots = torch.max(gt_slots.data, 1)

                # Compute losses
                loss1 = criterion(intent, target_intent)
                if args.model_type != 'crfencdec':
                    loss2 = criterion2(slots, targets_slots)
                else:
                    loss2 = model.loss(features.to(device), masks,
                                       targets_slots.to(device),)
                loss = loss1 + loss2

                # Backward pass
                if not separate:
                    model.train()
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                else:
                    model.train()
                    model_slots.train()
                    optimizer.zero_grad()
                    loss1.backward()
                    optimizer.step()

                    optimizer_slots.zero_grad()
                    loss2.backward()
                    optimizer_slots.step()

                # Store correct predictions
                correct_intent = (predictions_intent ==
                                  target_intent).sum().item()
                accuracy_intent = 1.0 * correct_intent / gt_intents.size(0)
                correct_slots = (predictions_slots ==
                                 targets_slots).sum().item()
                accuracy_slots = correct_slots / \
                    (gt_slots.size(0) * gt_slots.size(2))

                tepoch.set_postfix(loss=loss.item(
                ), accuracy_intent=100.0 * accuracy_intent, accuracy_slots=100 * accuracy_slots)

            # At the end of each epoch, evaluate and store train_loss, val_loss, train_accuracy, val_accuracy

            train_accuracy_epoch.append(100.0 * accuracy_intent)
            train_accuracy_slots_epoch.append(100.0 * accuracy_slots)
            train_loss_epoch.append(loss.item())
            train_intent_loss_epoch.append(loss1.item())
            train_slots_loss_epoch.append(loss2.item())

            # Evaluate on validation set
            with torch.no_grad():

                if not separate:
                    if args.model_type == 'crfencdec':
                        evaluations = evaluate_crf(
                            model, val_loader, device, criterion, criterion2)
                    else:
                        evaluations = evaluate(
                            model, val_loader, device, criterion, criterion2)

                else:
                    evaluations = evaluate(
                        model, val_loader, device, criterion, criterion2, model_slots)

                valid_loss = evaluations['valid_loss']
                val_loss_epoch.append(valid_loss)
                # Computing metrics

                f1_intent = evaluations['f1_intent']
                f1_slots = evaluations['f1_slots']

                # Storing val loss and val accuracy

                if separate:
                    val_intent_loss_epoch.append(
                        evaluations['valid_intent_loss'])
                    val_slots_loss_epoch.append(
                        evaluations['valid_slots_loss'])
                    valid_intent_loss = evaluations['valid_intent_loss']
                    valid_slots_loss = evaluations['valid_slots_loss']
                val_accuracy_epoch.append(
                    100 * evaluations['num_correct_intent'] / evaluations['total_val'])
                val_accuracy_slots_epoch.append(
                    100 * evaluations['num_correct_slots'] / evaluations['total_val_slots'])

                if not separate:
                    print(
                        "Epoch {}/{}, train_loss = {:4f}, train_accuracy = {:4f}, val_loss = {:4f}, val_f1_intent = {:4f}, val_f1_slots = {:4f}".format(
                            epoch,
                            num_epochs,
                            train_loss_epoch[-1],
                            train_accuracy_epoch[-1],
                            val_loss_epoch[-1],
                            f1_intent.mean(),
                            f1_slots.mean()
                        )
                    )
                else:
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

            if not separate:
                checkpoint = {
                    "epoch": epoch,
                    "valid_loss_min": valid_loss,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }

                valid_loss_min = saving_checkpoints(
                    checkpoint, checkpoint_path, best_model_path, valid_loss, valid_loss_min)

            else:
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
                valid_loss_min = saving_checkpoints(
                    checkpoint_intent, checkpoint_path+'.intent', best_model_path+'.intent', valid_intent_loss, valid_loss_min, sep='intent')
                valid_loss_min_slots = saving_checkpoints(
                    checkpoint_slots, checkpoint_path+'.slots', best_model_path+'.slots', valid_slots_loss, valid_loss_min_slots, sep='slots')

    # One last evaluation with the best checkpoint
    logging.info(
        'training finished, evaluate on val set and save metrics plots')
    # Init dataloader and load model
    eval_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)
    if not separate:
        if args.model_type == 'joint':
            model, _, _, _ = load_ckp(best_model_path, IntentSlotsClassifier(vocab_size, output_size,
                                                                             num_slots, embedding_dim, embedding_matrix, hidden_dim, n_layers, drop_prob=0.0), optimizer)

            model.to(device)
            model.eval()
            with torch.no_grad():
                final_evaluations = evaluate(
                    model, eval_loader, device, criterion, criterion2)
        elif args.model_type == 'encdec':
            model, _, _, _ = load_ckp(best_model_path, VanillaEncoderDecoder(vocab_size, embedding_dim, embedding_matrix, hidden_dim, output_size,
                                                                             num_slots, n_layers=1, dropout_rate=0.0), optimizer)

            model.to(device)
            model.eval()
            with torch.no_grad():
                final_evaluations = evaluate(
                    model, eval_loader, device, criterion, criterion2)

        elif args.model_type == 'crfencdec':
            model, _, _, _ = load_ckp(best_model_path, CRFEncoderDecoder(vocab_size, embedding_dim, embedding_matrix, hidden_dim, output_size,
                                                                         num_slots, n_layers=1, dropout_rate=0.0), optimizer)

            model.to(device)
            model.eval()
            with torch.no_grad():
                final_evaluations = evaluate_crf(
                    model, eval_loader, device, criterion, criterion2)

    else:
        model_intent, _, _, _ = load_ckp(best_model_path+'.intent', IntentClassifier(
            vocab_size, output_size, embedding_dim, embedding_matrix, hidden_dim, n_layers, drop_prob=0.0), optimizer)

        model_intent.to(device)
        model_intent.eval()

        model_slots, _, _, _ = load_ckp(best_model_path+'.slots', SlotsClassifier(
            vocab_size, num_slots, embedding_dim, embedding_matrix, hidden_dim, n_layers, drop_prob=0.0), optimizer_slots)
        model_slots.to(device)
        model_slots.eval()

        with torch.no_grad():
            final_evaluations = evaluate(
                model, eval_loader, device, criterion, criterion2, model2=model_slots)

    # Sometimes slots are not represented at all so we need to know which are present to put in abscissa
    # which means we can't simply have x = range(num_slots)
    all_slots = set(final_evaluations['slots_gt'])
    all_slots.update(final_evaluations['slots_pred'])
    ##########

    # Compute scores

    f1_intent = final_evaluations['f1_intent']

    f1_slots = final_evaluations['f1_slots']

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

    intent_str = f'Intent (Val set): \n\t f1 : {np.mean(f1_intent)} \n\t accuracy : {accu_intent} \n\t precision : {np.mean(prec_intent)} \n\t recall : {np.mean(recall_intent)} \n'
    slots_str = f'Slots (Val set): \n\t f1 : {np.mean(f1_slots)} \n\t accuracy : {accu_slots} \n\t precision : {np.mean(prec_slots)} \n\t recall : {np.mean(recall_slots)} \n'

    logging.info(intent_str)
    logging.info(slots_str)

    with open(os.path.join(
            args.plot_dir, 'val_metrics.txt'), 'w') as f:
        f.write(intent_str)
        f.write(slots_str)

    # Loss plot
    plt.plot(range(num_epochs), train_loss_epoch, label='train loss')
    plt.plot(range(num_epochs), val_loss_epoch, label='validation loss')
    plt.xlabel('epochs')
    plt.legend()

    plt.savefig(os.path.join(args.plot_dir, 'loss_epochs' +
                f'_{epochs}epochs_{sequence_length}_seqlength.png'))

    # Using dataframe to get easy plotting with df.plot
    slots_plot_path = os.path.join(args.plot_dir, 'validation_slots_f1' +
                                   f'_{epochs}epochs_{sequence_length}_seqlength.png')
    plot_f1_prec_recall(list(all_slots), f1_slots, prec_slots,
                        recall_slots, slots_dict, slots_plot_path, True)

    intents_plot_path = os.path.join(args.plot_dir, 'validation_intent_f1' +
                                     f'_{epochs}epochs_{sequence_length}_seqlength.png')

    plot_f1_prec_recall(list(set(final_evaluations['intent_gt'])), f1_intent,
                        prec_intent, recall_intent, intents_dict, intents_plot_path, False)


def saving_checkpoints(checkpoint, checkpoint_path, best_model_path, loss, min_loss, sep=None):

    save_ckp(checkpoint, False, checkpoint_path, best_model_path)
    # Saving ckpt if best
    if loss <= min_loss:
        if sep == 'slots':
            print(
                "Slots Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...".format(
                    min_loss, loss
                )
            )
        elif sep == 'intent':
            print(
                "Intent Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...".format(
                    min_loss, loss
                )
            )
        else:
            print(
                "Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...".format(
                    min_loss, loss
                )
            )
        # save checkpoint as best model
        save_ckp(checkpoint, True,
                 checkpoint_path, best_model_path)
        min_loss = loss
    return min_loss


if __name__ == '__main__':

    args = get_train_args()
    args.plot_dir = get_save_dir(args.plot_dir, args.name, training=True)
    args.ckpt_dir = get_save_dir(args.ckpt_dir, args.name, training=True)

    batch_size = args.batch_size
    epochs = args.epochs
    sequence_length = args.sequence_length
    focal_loss = args.focal_loss
    main(batch_size, sequence_length, epochs, args, focal_loss)
