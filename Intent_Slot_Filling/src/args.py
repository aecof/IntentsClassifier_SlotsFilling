import argparse


def get_train_args():
    """ get arguments needed in train.py"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--sequence_length", default=30, type=int)
    parser.add_argument("--epochs", default=20, type=int)
    parser.add_argument("--focal_loss", action='store_true')
    parser.add_argument("--dice_loss", action='store_true')
    parser.add_argument("--plot_dir", default="plots")
    parser.add_argument("--ckpt_dir", default="save")
    parser.add_argument("--name", "-n", required=True)
    parser.add_argument("--model_type", required=True)
    parser.add_argument("--dataset", default='atis')
    return parser.parse_args()


def get_eval_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dict_slots_path',
                        default='../data_dir/atis.slots.dict.txt')
    parser.add_argument('--dict_intent_path',
                        default='../data_dir/atis.intent.dict.txt')
    parser.add_argument('--stoi_path', default='./stoi.json')
    parser.add_argument('--test_path', default='../data_dir/test_atis.tsv')
    parser.add_argument('--test_slots_path',
                        default='../data_dir/test_slots_atis.tsv')
    parser.add_argument('--ckpt_path', required=True)
    parser.add_argument("--plot_dir", default="plots")
    parser.add_argument('--sequence_length', default=30,
                        help='max sequence length, HAS TO BE THE SAME THAN IN TRAINING')
    parser.add_argument('--save_dir', default='plots')
    parser.add_argument('--name', '-n', required=True,
                        help='Prefix to make the difference between experiments')
    parser.add_argument('--exp_number', required=True,
                        help='experiment number to use')
    parser.add_argument("--model_type", required=True)
    return parser.parse_args()
