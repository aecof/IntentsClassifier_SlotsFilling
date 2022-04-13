import json


def main():
    """Load json datasets and write four files : train.tsv, train_slots.tsv, test.tsv, test_slots.tsv 

        -dict.intents.tsv and dict.slots.tsv were written manually given README.md
        -train.tsv format is : {sentence str} \t {intent index in dict.intents.tsv}
        - train_slots.tsv format is : {{word slot} for word in sentence}

     (This is Nvidia NeMo formatting)
    """

    #### Load json data ######
    dev_json = json.load(open(
        'data/dataset_dev.json', 'r', encoding='utf-8'))['intents']
    test_json = json.load(open(
        'data/dataset_test.json', 'r', encoding='utf-8'))['intents']
    ##########################

    #### get intents list ####
    with open('data_dir/dict.intents.csv', 'r') as f:
        intents_list = f.readlines()
        intents_list = [elm.split('\n')[0] for elm in intents_list]

    ##### get slots list #####
    with open('data_dir/dict.slots.csv', 'r') as f:
        slots_list = f.readlines()
        slots_list = [elm.split('\n')[0] for elm in slots_list]

    #### Initialise train.tsv and test.tsv with their header ####
    with open('train.tsv', 'w') as f:
        f.write('sentence\tlabel\n')
    with open('test.tsv', 'w') as f:
        f.write('sentence\tlabel\n')

    for intent, utterance in dev_json.items():

        intent_idx = intents_list.index(intent)

        for sentence in utterance['utterances']:  # iterate over every sentences

            slots_str = []
            sentence_str = []

            for words in sentence['data']:

                text_snippet = words['text'].replace(
                    '\n', '').lower()  # get text snippet, lowercase
                print(text_snippet)
                slot = words.get('entity')  # get corresponding slot
                
                if slot:
                    slot = intent+'.'+slot  # produce intent / slot pair
                else:
                    slot = 'O'  # affect slot 'O' if no slots
                try:
                    # in case intent/slot pair is not known
                    
                    slot_idx = slots_list.index(slot)
                    found = True
                except:
                    slot_idx = slots_list.index('O')
                    found = False
                print(intent,slot, slot_idx, found)
                
                sentence_str.append(text_snippet)
                slots_str += [str(slot_idx)
                              for i in range(len(text_snippet.strip().split(' ')))]  # for snippet that have multiple words, each word has the same slot index
            
            

            sentence_str = ''.join(sentence_str)
            print(sentence_str)
            slots_str = ' '.join(slots_str)

            # Writing on files
            with open('train.tsv', 'a', encoding='utf-8') as f:
                f.write(sentence_str + '\t' + str(intent_idx)+'\n')
            with open('train_slots.tsv', 'a', encoding='utf-8') as f:
                f.write(slots_str+'\n')

    ##### The same thing all over again with the test set #####
    for intent, utterance in test_json.items():
        intent_idx = intents_list.index(intent)
        for sentence in utterance['utterances']:

            slots_str = []
            sentence_str = []
            for words in sentence['data']:

                text_snippet = words['text'].replace('\n', '').lower()
                slot = words.get('entity')
                if slot:
                    slot = intent+'.'+slot
                else:
                    slot = 'O'
                try:
                    # Some intents/slots pairs are not expected
                    slot_idx = slots_list.index(slot)
                except:
                    slot_idx = slots_list.index('O')
                sentence_str.append(text_snippet)
                slots_str += [str(slot_idx)
                              for i in range(len(text_snippet.split(' ')))]

            sentence_str = ''.join(sentence_str)
            slots_str = ' '.join(slots_str)

            with open('test.tsv', 'a', encoding='utf-8') as f:
                f.write(sentence_str + '\t' + str(intent_idx)+'\n')
            with open('test_slots.tsv', 'a', encoding='utf-8') as f:
                f.write(slots_str+'\n')


if __name__ == '__main__':
    main()
