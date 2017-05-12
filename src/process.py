from itertools import combinations
import numpy as np


def char_rep(word, count, maxlen):
    rep = np.full(maxlen, -1, dtype=np.int)
    for i, char in enumerate(word):
        rep[i] = ord(char) - ord('a')
        # if rep[i] > 25 or rep[i] < 0:
        #     print(word)
    return np.concatenate(([count], [len(word)], rep))


def generate_adj_matrix(filepath):
    with open(filepath) as f:
        lines = [line.rstrip('\n').split() for line in f.readlines()]

        dictionary = {}
        count = 0
        maxlen = 0
        for line in lines:
            # if count > 15:
            #     break
            for word in line:
                if word not in dictionary:
                    maxlen = max(len(word), maxlen)
                    dictionary[word] = count
                    count += 1

        M = np.eye(len(dictionary))
        for line in lines:
            for c1, c2 in combinations(line, 2):
                try:
                    w1 = dictionary[c1]
                    w2 = dictionary[c2]
                    if w1 != w2:
                        M[w1][w2] += 1
                        M[w2][w1] += 1
                except:
                    pass

    return M, dictionary


def generate_data(filepath, char_level=False):
    with open(filepath) as f:
        lines = [line.rstrip('\n').split() for line in f.readlines()]

        dictionary = {}
        count = 0
        maxlen = 0
        for line in lines:
            for word in line:
                if word not in dictionary:
                    maxlen = max(len(word), maxlen)
                    dictionary[word] = count
                    count += 1

        if char_level:
            char_dictionary = {word: char_rep(word, count, maxlen)
                               for word, count in dictionary.items()}

        # lines = [[dictionary[word] for word in line] for line in lines]

        data = []
        for line in lines:
            for comb in combinations(line, 2):
                if char_level:
                    comb_rep = (char_dictionary[comb[0]], dictionary[comb[1]])
                    comb_rep_rev = (char_dictionary[comb[1]], dictionary[comb[0]])
                else:
                    comb_rep = (dictionary[comb[0]], dictionary[comb[1]])
                    comb_rep_rev = (dictionary[comb[1]], dictionary[comb[0]])

                data.append(comb_rep)
                data.append(comb_rep_rev)
                # data.append(comb)
                # data.append(comb[::-1])
        for word in dictionary.keys():
            if char_level:
                data.append((char_dictionary[word], dictionary[word]))
            else:
                data.append((dictionary[word], dictionary[word]))

    if char_level:
        return char_dictionary, maxlen, dictionary, data
    else:
        return dictionary, data


def generate_batch(data, batch_size):
    batch = np.random.choice(len(data), batch_size)

    inputs = np.array([data[b][0] for b in batch])
    labels = np.array([data[b][1] for b in batch])

    return inputs, labels[:, None]


def batch_generator(data, batch_size):
    data = np.random.permutation(data)

    idx = 0
    while True:
        batch = data.take(np.arange(idx, idx + batch_size),
                          axis=0, mode='wrap')
        inputs = np.array([b[0] for b in batch])
        labels = np.array([b[1] for b in batch])
        yield inputs, labels[:, None]

        idx += batch_size
        if idx > len(data):
            idx %= len(data)
            data = np.random.permutation(data)


def write_vocab_file(dictionary, filepath):
    vocab = sorted(dictionary, key=dictionary.get)

    with open(filepath, 'w') as f:
        f.writelines('\n'.join(vocab))
