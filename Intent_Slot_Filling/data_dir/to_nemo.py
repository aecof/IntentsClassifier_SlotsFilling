
with open('data/atis.intent.dict.txt', 'r') as f:
    intents_list = f.readlines()
    intents_list = [elm.split('\n')[0] for elm in intents_list]

##### get slots list #####
with open('data/atis.slots.dict.txt', 'r') as f:
    slots_list = f.readlines()
    slots_list = [elm.split('\n')[0] for elm in slots_list]

print(slots_list)
print(intents_list)
with open('train.tsv', 'a') as g:
    g.write(f'sentence\tintent\n')

with open('dev.tsv', 'a') as g:
    g.write(f'sentence\tintent\n')
with open('test.tsv', 'a') as g:
    g.write(f'sentence\tintent\n')

with open('data/atis-2.train.w-intent.iob', 'r') as f:

    lines = f.readlines()
    lines = [line[:-1].strip() for line in lines]
    print(lines[0])

    for line in lines:
        line = line.split('\t')
        sentence, labels = line[0], line[1]
        labels = labels.strip()
        labels = labels.split(' ')
        print(labels)
        intent = labels[-1]
        slots = labels[:-1]
        intent = intents_list.index(intent)
        slots = [str(slots_list.index(slot)) for slot in slots]
        slots += slots[0]

        with open('train.tsv', 'a') as g:
            g.write(f'{sentence}\t{intent}\n')

        with open('train_slots.tsv', 'a') as h:
            h.write(' '.join(slots) + '\n')

with open('data/atis.train.w-intent.iob', 'r') as f:

    lines = f.readlines()
    lines = [line[:-1].strip() for line in lines]
    print(lines[0])

    for line in lines:
        line = line.split('\t')
        sentence, labels = line[0], line[1]
        labels = labels.strip()
        labels = labels.split(' ')
        print(labels)
        intent = labels[-1]
        slots = labels[:-1]
        intent = intents_list.index(intent)
        slots = [str(slots_list.index(slot)) for slot in slots]
        slots += slots[0]

        with open('train_atis.tsv', 'a') as g:
            g.write(f'{sentence}\t{intent}\n')

        with open('train_slots_atis.tsv', 'a') as h:
            h.write(' '.join(slots) + '\n')

with open('data/atis-2.dev.w-intent.iob', 'r') as f:

    lines = f.readlines()
    lines = [line[:-1].strip() for line in lines]
    print(lines[0])

    for line in lines:
        line = line.split('\t')
        sentence, labels = line[0], line[1]
        labels = labels.strip()
        labels = labels.split(' ')
        print(labels)
        intent = labels[-1]
        slots = labels[:-1]
        intent = intents_list.index(intent)
        slots = [str(slots_list.index(slot)) for slot in slots]
        slots += slots[0]

        with open('dev_atis.tsv', 'a') as g:
            g.write(f'{sentence}\t{intent}\n')

        with open('dev_slots_atis.tsv', 'a') as h:
            h.write(' '.join(slots) + '\n')


with open('data/atis.test.w-intent.iob', 'r') as f:

    lines = f.readlines()
    lines = [line[:-1].strip() for line in lines]
    print(lines[0])

    for line in lines:
        line = line.split('\t')
        sentence, labels = line[0], line[1]
        labels = labels.strip()
        labels = labels.split(' ')
        print(labels)
        intent = labels[-1]
        slots = labels[:-1]
        intent = intents_list.index(intent)
        slots = [str(slots_list.index(slot)) for slot in slots]
        slots += slots[0]

        with open('test_atis.tsv', 'a') as g:
            g.write(f'{sentence}\t{intent}\n')

        with open('test_slots_atis.tsv', 'a') as h:
            h.write(' '.join(slots) + '\n')
