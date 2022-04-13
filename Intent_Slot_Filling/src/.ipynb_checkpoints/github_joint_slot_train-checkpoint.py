import torch.nn as nn
import torch.optim as optim
from torchsummary import summary
import argparse
import numpy as np
from utils import *
from models import Encoder, Decoder
import json
import os
import pandas as pd
import matplotlib.pyplot as plt


import logging
logging.basicConfig(filename="github_joint_slot_supervised_training.log",
    filemode="a",
    format="%(name)s - %(levelname)s - %(message)s",
    level = logging.DEBUG
)
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score

USE_CUDA = torch.cuda.is_available()

# train_data, word2index, tag2index, intent2index = preprocessing('./data/atis-2.train.w-intent.iob', 60)

def train(config):
    df_train, vocab, stoi = init_train_df_joint_slot(path_intent = '../data_dir/train.tsv', path_slots = '../data_dir/train_slots.tsv')
    
    with open('stoi_joint_slot.json','w') as f :
        json.dump(stoi, f)
    slots_dict = pd.read_csv('../data_dir/dict.slots.csv',header=None)
    intents_dict = pd.read_csv('../data_dir/dict.intents.csv',header=None)
    df_train['preprocessed_sentences'] = preprocess_sentences(df_train, stoi, max_len = config.max_length)
    df_train['preprocessed_slots'] = preprocess_slots(df_train, max_len = config.max_length, pad = [list(slots_dict[0]).index('O')]) # We used the NeMo data-format, which uses 'O' as no slots label
    
    df_val = df_train.sample(frac = 0.2, random_state = 42 )
    df_train = df_train.drop(df_val.index)
    
    train_data = preprocess_data_from_nemo(df_train)
    val_data = preprocess_data_from_nemo(df_val)
    
    vocab_length = len(vocab) + 1 #+1 for padding
    num_slots = len(slots_dict)
    num_intents = len(intents_dict)
    
    
    if not train_data:
        print("Please check your data or its path")
        return
    
    encoder = Encoder(vocab_length, config.embedding_size, config.hidden_size)
    decoder = Decoder(num_slots, num_intents, num_slots//3, config.hidden_size*2)
    logging.info('Encoder Summary')
    summary(encoder)
    logging.info('Decoder Summary')
    summary(decoder)
    if USE_CUDA:
        encoder = encoder.cuda()
        decoder = decoder.cuda()

    encoder.init_weights()
    decoder.init_weights()

    loss_function_1 = nn.CrossEntropyLoss(ignore_index=0)
    loss_function_2 = nn.CrossEntropyLoss()
    enc_optim = optim.Adam(encoder.parameters(), lr=config.learning_rate)
    dec_optim = optim.Adam(decoder.parameters(), lr=config.learning_rate)
    
    for epoch in range(config.num_epochs):
        losses = []
        encoder.train()
        decoder.train()
        for i, batch in enumerate(getBatch(config.batch_size, train_data)):
            x, y_1, y_2 = zip(*batch)
            x = torch.cat(x)
            tag_target = torch.cat(y_1)

            intent_target = torch.cat(y_2)
            x_mask = torch.cat([Variable(torch.ByteTensor(tuple(map(lambda s: s == 0, t.data)))).cuda() if USE_CUDA
                                else Variable(torch.ByteTensor(tuple(map(lambda s: s == 0, t.data)))) for t in x])\
                .view(config.batch_size, -1)
            x_mask = x_mask > 0
            
            encoder.zero_grad()
            decoder.zero_grad()

            output, hidden_c = encoder(x, x_mask)

            if USE_CUDA:
                start_decode = Variable(torch.LongTensor([[1]*config.batch_size])).cuda().transpose(1, 0)
            else:
                start_decode = Variable(torch.LongTensor([[1]*config.batch_size])).transpose(1, 0)
            
            tag_score, intent_score = decoder(start_decode, hidden_c, output, x_mask)

            loss_1 = loss_function_1(tag_score, tag_target.view(-1))
            loss_2 = loss_function_2(intent_score, intent_target)

            loss = loss_1+loss_2
            losses.append(loss.item())
            loss.backward()

            torch.nn.utils.clip_grad_norm(encoder.parameters(), 5.0)
            torch.nn.utils.clip_grad_norm(decoder.parameters(), 5.0)

            enc_optim.step()
            dec_optim.step()

            if i % 100 == 0:
                logging.info(f'Epoch : {epoch},  step : {i}, loss : {np.mean(losses)}')
                losses = []
                
        ## Evaluating on val dataset
        with torch.no_grad():

            correct_val_intent = 0
            correct_val_slots = 0
            total_val = 0
            total_val_slots = 0
            val_loss_inside_epoch = []
            val_intent_predictions = []
            val_slots_predictions = []
            intent_gt = []
            slots_gt = []
            
            encoder.eval()
            decoder.eval()
            for i, batch in enumerate(getBatch(config.batch_size, val_data)):
                x, y_1, y_2 = zip(*batch)
                x = torch.cat(x)
                tag_target = torch.cat(y_1)

                intent_target = torch.cat(y_2)
                x_mask = torch.cat([Variable(torch.ByteTensor(tuple(map(lambda s: s == 0, t.data)))).cuda() if USE_CUDA
                                    else Variable(torch.ByteTensor(tuple(map(lambda s: s == 0, t.data)))) for t in x])\
                    .view(config.batch_size, -1)


                x_mask = x_mask > 0
                encoder.zero_grad()
                decoder.zero_grad()

                output, hidden_c = encoder(x, x_mask)

                if USE_CUDA:
                    start_decode = Variable(torch.LongTensor([[1]*config.batch_size])).cuda().transpose(1, 0)
                else:
                    start_decode = Variable(torch.LongTensor([[1]*config.batch_size])).transpose(1, 0)

                slots, intent = decoder(start_decode, hidden_c, output, x_mask)
                
                
                targets_intents_val = intent_target
                _, predictions_intent = torch.max(intent.data, 1)
                
                targets_slots_val = tag_target

                
                
                _, predictions_slots_val = torch.max(slots.data,1)





                val_intent_predictions += list(predictions_intent.detach().cpu().numpy())
                val_slots_predictions += list(predictions_slots_val.detach().cpu().numpy().flatten())
                intent_gt+=list(targets_intents_val.detach().cpu().numpy())
                slots_gt+=list(targets_slots_val.detach().cpu().numpy().flatten())


                _, predictions_slots_val = torch.max(slots.data,1)
                



                val_loss1 = loss_function_1(intent, targets_intents_val.view(-1))
                val_loss2 = loss_function_2(slots, targets_slots_val.view(-1))
                val_loss = val_loss1 + val_loss2
                val_loss_inside_epoch.append(val_loss.item())




                total_val += intent.size(0)
                total_val_slots += targets_slots_val.size(0) * num_slots

                correct_val_intent += (predictions_intent == targets_intents_val).sum().item()
                correct_val_slots += (predictions_slots_val == targets_slots_val.view(-1)).sum().item()

            f1_intent = f1_score(intent_gt, val_intent_predictions, average = None)
            f1_slots = f1_score(slots_gt, val_slots_predictions, average = None)
            valid_loss = np.mean(val_loss_inside_epoch)
            if epoch == (config.num_epochs -1):
                all_slots = set(slots_gt)
                all_slots.update(val_slots_predictions)
                accu_intent = accuracy_score(intent_gt, val_intent_predictions)
                accu_slots = accuracy_score(slots_gt, val_slots_predictions)
                prec_intent = precision_score(
                    intent_gt, val_intent_predictions, average=None)
                prec_slots = precision_score(slots_gt, val_slots_predictions, average=None)
                recall_intent = recall_score(
                    intent_gt, val_intent_predictions, average=None)
                recall_slots = recall_score(slots_gt, val_slots_predictions, average=None)
                        # Storing val loss and val accuracy
                    
                intent_str = f'Intent (Val set): \n\t f1 : {np.mean(f1_intent)} \n\t accuracy : {accu_intent} \n\t precision : {np.mean(prec_intent)} \n\t recall : {np.mean(recall_intent)} \n'
                slots_str = f'Slots (Val set): \n\t f1 : {np.mean(f1_slots)} \n\t accuracy : {accu_slots} \n\t precision : {np.mean(prec_slots)} \n\t recall : {np.mean(recall_slots)} \n'
                
                logging.info(intent_str)
                logging.info(slots_str)
                
                df_result_slots = pd.DataFrame()
                df_result_slots['index'] = list(all_slots)
                # print(slots_dict.to_dict())
                df_result_slots['slots'] = df_result_slots['index'].apply(
                    lambda x: slots_dict.to_dict()[0][x])
                df_result_slots['f1'] = f1_slots
                df_result_slots['prec'] = prec_slots
                df_result_slots['recall'] = recall_slots
                df_result_slots.plot.bar(x='slots', y=['f1', 'prec', 'recall'])
                plt.savefig(os.path.join('plots','attention_slots_f1precrecall.png'))

                df_result_intent = pd.DataFrame()
                df_result_intent['index'] = list(set(intent_gt))
                # print(slots_dict.to_dict())
                df_result_intent['slots'] = df_result_intent['index'].apply(
                    lambda x: intents_dict.to_dict()[0][x])
                df_result_intent['f1'] = f1_intent
                df_result_intent['prec'] = prec_intent
                df_result_intent['recall'] = recall_intent
                df_result_intent.plot.bar(x='slots', y=['f1', 'prec', 'recall'])
                plt.savefig(os.path.join('plots','attention_intent_f1precrecall.png'))
            
            

            logging.info(
                "Epoch {}/{}, val_loss = {:4f}, val_f1_intent = {:4f}, val_f1_slots = {:4f}".format(
                    epoch,
                    config.num_epochs,
                    valid_loss,
                    f1_intent.mean(),
                    f1_slots.mean()
                )
            )
                

            
    
    if not os.path.exists(config.model_dir):
        os.makedirs(config.model_dir)
    
    torch.save(decoder.state_dict(), os.path.join(config.model_dir, 'jointnlu-decoder.pkl'))
    torch.save(encoder.state_dict(), os.path.join(config.model_dir, 'jointnlu-encoder.pkl'))
    logging.info("Train Complete!")
    
                
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', type=str, default='./data/atis-2.train.w-intent.iob', help='path of train data')
    parser.add_argument('--model_dir', type=str, default='./models/', help='path for saving trained models')

    # Model parameters
    parser.add_argument('--max_length', type=int, default=60, help='max sequence length')
    parser.add_argument('--embedding_size', type=int, default=50, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int, default=128, help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int, default=1, help='number of layers in lstm')
    
    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    config = parser.parse_args()
    train(config)
