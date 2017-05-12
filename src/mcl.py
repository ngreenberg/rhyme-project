import numpy as np

import process


def normalize(matrix):
    col_sums = matrix.sum(axis=0)
    return matrix / col_sums


M, dictionary = process.generate_adj_matrix('../data/rhymes.txt')
reverse_dictionary = {index: word for word, index in dictionary.items()}

e = 2  # expansion parameter
r = 2  # inflation parameter

M = normalize(M)
i = 0
while True:
    # print(i)
    i += 1

    M_old = M
    M = np.linalg.matrix_power(M, e)  # expansion
    M = np.power(M, r)  # inflation
    M = normalize(M)
    if np.array_equal(M, M_old):
        break

used = set()
clusters = []
for i, row in enumerate(M):
    cluster = set(np.nonzero(row)[0])
    if len(cluster) > 0 and i not in used:
        clusters.append(cluster)
        used |= cluster

for cluster in clusters:
    print([reverse_dictionary[word] for word in cluster])

print(e)
print(r)
print(len(clusters))
