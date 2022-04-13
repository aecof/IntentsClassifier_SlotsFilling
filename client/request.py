
import requests
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--text', nargs='+', default="show me all flights from denver to san francisco next wednesday which leave after noon")
    parser.add_argument('--model_type', choices=['joint', 'encdec'])
    return parser.parse_args()


def main(args):
    input_text = ' '.join(args.text)

    if args.model_type == 'joint':
        result = requests.get(
            "http://my_server:8000/nlu-joint/", data=input_text).text
    elif args.model_type == 'encdec':
        result = requests.get(
            "http://my_server:8000/nlu/", data=input_text).text

    intent, slots = result.split('///')
    split_text = input_text.split()
    print(len(split_text))
    slots = slots[1:-1].split(',')

    print('Intent :', intent)
    print('Slot Filling : ')

    for i, slot in enumerate(slots[:len(split_text)]):
        if i < len(split_text):
            print('\t', '{:20}'.format(split_text[i]), ':', slot)
        else:
            print('\t', '<pad> :', slot)


if __name__ == '__main__':
    args = parse_args()
    main(args)
