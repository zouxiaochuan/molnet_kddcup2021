import sys
import os
import pickle

sys.path.append('./pcqm4m_v3')

use_onecore = False
if len(sys.argv) > 1:
    use_onecore = sys.argv[1]=='onecore'
    pass


import dataio

if use_onecore:
    import multiprocessing.dummy as mp
else:
    import multiprocessing as mp
    pass

from tqdm import tqdm

def process_smiles_func(param):
    i, s, y, folder = param
    graph, xyz_success, mmf_result = dataio.smiles2graph(s)

    return graph, xyz_success, mmf_result


if __name__ == '__main__':

    smiles = []
    with open('./test_smiles.txt') as fin:
        for line in fin:
            smiles.append(line.strip())
            pass
        pass

    # smiles = smiles[:100]
    print(len(smiles))
    folder = ''
    params = [(i, s, 0, folder) for i, s in enumerate(smiles)]
    
    pool = mp.Pool()
    results = pool.imap(
        process_smiles_func,
        params)
    pool.close()

    failCnt = 0
    failMMFCnt = 0

    graphs = []
    for g, failEmbed, mmf_result in tqdm(results, total=len(smiles)):
        if failEmbed:
            failCnt += 1
            pass

        if mmf_result == -1:
            failMMFCnt += 1
            pass

        graphs.append(g)
        pass

    print(failCnt)
    print(failMMFCnt)

    with open('./test_graph.pk', 'wb') as fout:
        pickle.dump(graphs, fout, protocol=pickle.HIGHEST_PROTOCOL)
        pass
    
    pass
