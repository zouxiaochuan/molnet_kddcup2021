
import ogb.lsc
from tqdm import tqdm
from rdkit import Chem
import dataio2 as dataio
from rdkit.Chem import AllChem
import numpy as np
import multiprocessing as mp


root = '../../datasets'
dataset = dataio.SimplePCQM4MDataset(
    root, transforms=None)


def find_proc(param):
    i = param
    graph, y = dataset[i]
    if graph['bond_feat_cate'].shape[0] > 0:
        bond_feat = np.max(graph['bond_feat_cate'], axis=0)
        pass
    else:
        bond_feat = None
    
    atom_feat = np.max(graph['atom_feat_cate'], axis=0)

    graph_feat = graph['graph_feat_cate']

    spl = np.max(graph['shortest_path_length'])

    src = np.max(graph['atom_same_ring_count'])

    return atom_feat, bond_feat, graph_feat, spl, src


if __name__ == '__main__':

    maxvalues_atom = None
    minvalues_atom = None
    maxvalues_bond = None
    maxvalues_graph = None
    maxvalues_path = None
    maxvalues_same_ring = None

    pool = mp.Pool()
    values = pool.imap(find_proc, tqdm(list(range(len(dataset)))))
    pool.close()

    values = list(values)

    for atom_feat, bond_feat, graph_feat, spl, src in tqdm(values):
        if maxvalues_bond is not None and bond_feat is not None:
            maxvalues_bond = np.maximum(
                maxvalues_bond, bond_feat)
        else:
            maxvalues_bond = bond_feat
            pass
        pass

        if maxvalues_atom is not None:
            maxvalues_atom = np.maximum(
                maxvalues_atom, atom_feat)
            minvalues_atom = np.minimum(
                minvalues_atom, atom_feat)
            pass
        else:
            maxvalues_atom = atom_feat
            minvalues_atom = atom_feat
            pass

        if maxvalues_graph is not None:
            maxvalues_graph = np.maximum(
                maxvalues_graph, graph_feat)
        else:
            maxvalues_graph = graph_feat
            pass

        if maxvalues_path is not None:
            maxvalues_path = max(
                maxvalues_path, spl)
        else:
            maxvalues_path = spl
            pass

        if maxvalues_same_ring is not None:
            maxvalues_same_ring = max(
                maxvalues_same_ring, src)
        else:
            maxvalues_same_ring = src
            pass
        pass

    print(maxvalues_atom)
    print(minvalues_atom)
    print(maxvalues_bond)
    print(maxvalues_graph)
    print(maxvalues_path)
    print(maxvalues_same_ring)
    pass
