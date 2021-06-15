import torch
import os
import pandas as pd
import pickle5 as pickle
import numpy as np
from tqdm import tqdm
import shutil
import ogb

import multiprocessing as mp
from scipy.spatial.transform import Rotation as R
from scipy.stats import special_ortho_group
import scipy.sparse
import features
import pandas as pd
import networkx as nx
import sklearn.manifold

MAX_ATOM_NUM_WITHIN_MOL = 51
MAX_ATOM_NUM = 52


def cartesian_product(x, y):
    return np.array([np.tile(x, len(y)), np.repeat(y, len(x))])


def make_fully_index(num):
    arange = np.arange(num, dtype='int64')
    
    return cartesian_product(arange, arange)


EDGE_FULLY_CONNECT = [make_fully_index(i) for i in
                      range(MAX_ATOM_NUM_WITHIN_MOL)]

EDGE_INDEX_DICT = [
    {tuple(eii): i for i, eii in enumerate(np.transpose(ei))}
    for ei in EDGE_FULLY_CONNECT]


def extract_mol_feature(mol):
    fea = np.zeros(4, dtype='float32')
    fea[0] = mol.GetNumAtoms()
    fea[1] = mol.GetNumBonds()
    fea[2] = mol.GetNumHeavyAtoms()
    ringinfo = mol.GetRingInfo()
    fea[3] = ringinfo.NumRings()
    return fea


def shortest_path_length(edge_index, num_nodes):
    if edge_index.shape[1] > 0:
        adj = scipy.sparse.coo_matrix(
            (np.ones(edge_index.shape[1]), (edge_index[0], edge_index[1])),
            shape=(num_nodes, num_nodes)).tocsr()

        try:
            dist = scipy.sparse.csgraph.shortest_path(adj, directed=False)
        except ValueError as e:
            print(edge_index)
            print(type(adj))

            raise e
            pass
        
        dist[np.isinf(dist)] = -1
    else:
        dist = -np.ones((num_nodes, num_nodes), dtype='int16')
        pass
    
    return dist.astype('int16')
    pass


def path_node2edge(nodes, edge_dict):

    edge_path = []
    for i in range(1, len(nodes)):
        eid = edge_dict.get((nodes[i-1], nodes[i]))
        if eid is None:
            eid = edge_dict[(nodes[i], nodes[i-1])]
            pass
        edge_path.append(eid)

        pass

    return edge_path
    pass


def shortest_path_length2(edge_index, num_nodes):
    dist = -np.ones((num_nodes, num_nodes), dtype='int16')
    paths = [[set() for _ in range(num_nodes)] for _ in range(num_nodes)]
    pathsAtom = [[set() for _ in range(num_nodes)] for _ in range(num_nodes)]

    if edge_index.shape[1] > 0:
        g = nx.Graph()
        g.add_edges_from(
            edge_index.T
        )
        
        edge_dict = {
            (edge_index[0, i], edge_index[1, i]): i for i in range(
                edge_index.shape[1])
        }
        
        res = nx.all_pairs_shortest_path(g)

        # for i in range(num_nodes):
        #     for j in range(num_nodes):
        #         if i == j:
        #             dist[i, j] = 0
        #         else:
        #             spaths = list(nx.all_shortest_paths(g, i, j))
        # 
        #             if len(spaths) > 0:
        #                 dist[i, j] = len(spaths[0])
        # 
        #                 for p in spaths:
        #                     edge_path = path_node2edge(p, edge_dict)
        #                     paths[i][j].update(edge_path)
        #                     pass
        #                 pass
        #             pass
        #         pass
        #     pass
        for sid, path_dict in res:
            for did, path in path_dict.items():
                if len(path) == 1:
                    dist[sid, did] = 0
                else:
                    dist[sid, did] = len(path) - 1
                    edge_path = path_node2edge(path, edge_dict)
                    paths[sid][did].update(edge_path)
                    pathsAtom[sid][did].update(path[1:-1])
                pass
            pass
        pass
    
    else:
        pass
    
    return dist.astype('int16'), paths, pathsAtom


def atom_same_ring_count(mol):
    ringInfo = mol.GetRingInfo()

    atom_rings = ringInfo.AtomRings()

    num_atoms = mol.GetNumAtoms()
    num_rings = len(atom_rings)

    if num_rings == 0:
        return np.zeros((num_atoms, num_atoms), dtype='float32'), \
            np.zeros((num_atoms, num_atoms), dtype='float32')

    atom_groups = np.zeros((num_atoms, num_rings), dtype='float32')
    for i, ring_atoms in enumerate(atom_rings):
        atom_groups[ring_atoms, i] = 1
        pass

    ring_sizes = np.array(
        [len(r) for r in atom_rings], dtype='float32').reshape(1, 1, -1)

    is_same_ring = atom_groups[:, None, :] * atom_groups[None, :, :]
    
    return np.sum(is_same_ring, axis=-1), np.min(
        is_same_ring * ring_sizes, axis=-1)


def spectral_embedding(bond_index, num_nodes):
    ones = np.ones(bond_index.shape[1])
    adj = scipy.sparse.coo_matrix(
        (ones, (bond_index[0], bond_index[1])), shape=(num_nodes, num_nodes)
    ).todense()
    adj += adj.T

    emb = np.zeros((num_nodes, 4), dtype='float32')
    semb = sklearn.manifold.spectral_embedding(np.asarray(adj), n_components=4)
    emb[:, :semb.shape[1]] = semb

    return emb
    

def smiles2graph(s):
    import rdkit

    mol = rdkit.Chem.MolFromSmiles(s)

    # calculate xyz may change the order of atoms

    xyz_res = features.extract_mol_multi_xyz(mol, 1)

    if xyz_res is None:
        xyz = np.zeros((0, mol.GetNumAtoms(), 3), dtype='float32')
        xyz_success = False
        mmf_result = -1
        pass
    else:
        xyz, mol, mmf_result = xyz_res
        xyz_success = True
        pass

    atom_float_feat = features.extract_atom_float_feat(mol)
    atom_cate_feat = features.extract_atom_cate_feat(mol)

    bond_index, bond_cate_feat, bond_float_feat = \
        features.extract_bond_feat(mol)

    spectral_emb = spectral_embedding(
        bond_index, atom_cate_feat.shape[0]).astype('float32')
    
    graph = dict()
    atom_ring_count, atom_ring_min = atom_same_ring_count(mol)
    graph['atom_same_ring_count'] = atom_ring_count + 1
    graph['atom_same_ring_min'] = atom_ring_min + 1

    atom_cate_feat = np.hstack(
        (atom_cate_feat, graph['atom_same_ring_count'].diagonal().reshape(
            -1, 1).astype('int32') - 1))
    graph['bond_index'] = bond_index
    graph['bond_feat_cate'] = bond_cate_feat
    graph['bond_feat_float'] = bond_float_feat
    graph['atom_feat_cate'] = atom_cate_feat
    graph['atom_feat_float'] = np.hstack((atom_float_feat, spectral_emb))
    graph['xyz'] = xyz
    graph['num_atoms'] = atom_cate_feat.shape[0]
    graph['num_bonds'] = bond_index.shape[1]
    dist, paths, pathsAtom = shortest_path_length2(
        bond_index, atom_cate_feat.shape[0])

    graph['shortest_path_length'] = dist + 1
    graph['shortest_path'] = paths
    graph['shortest_path_atom'] = pathsAtom
    
    graph['graph_feat_cate'] = features.extract_mol_feat_cate(mol)
    graph['graph_fp'] = features.extrac_mol_fingerprint(mol)
    graph['graph_fp_size'] = 2048
    
    return graph, xyz_success, mmf_result


def process_smiles_func(param):
    i, s, y, folder = param
    graph, xyz_success, mmf_result = smiles2graph(s)

    path = os.path.join('data', str(i // 1000))
    
    filename = os.path.join(path, str(i) + '.pk')

    with open(os.path.join(folder, filename), 'wb') as fout:
        pickle.dump((graph, y), fout, protocol=pickle.HIGHEST_PROTOCOL)
        pass
    
    return filename, xyz_success, mmf_result


def bond_index2bond_edge(bond_index):
    num_bonds = bond_index.shape[1]
    range_bonds = np.arange(num_bonds)
    df = pd.DataFrame(
        {
            'a': np.hstack((bond_index[0], bond_index[1])),
            'b': np.hstack((range_bonds, range_bonds))
        })

    df = pd.merge(df, df, on=['a'], suffixes=['_l', '_r'])
    bond_edge = df[['b_l', 'b_r']].values

    not_self = bond_edge[:, 0] != bond_edge[:, 1]
    bond_edge = bond_edge[not_self]
    bond_atom = df['a'].values.flatten()[not_self]

    return bond_edge.T, bond_atom
    pass


def collate_graph(datas):
    max_atom_num = np.max([data['num_atoms'] for data, _ in datas])
    max_bond_num = np.max([data['num_bonds'] for data, _ in datas])
    # max_bond_edge = np.max(
    #     [data['num_bond_edge'] for data, _ in datas])

    dim_atom_cate_feat = datas[0][0]['atom_feat_cate'].shape[1]
    dim_atom_float_feat = datas[0][0]['atom_feat_float'].shape[1]

    dim_bond_cate_feat = datas[0][0]['bond_feat_cate'].shape[1]
    dim_bond_float_feat = datas[0][0]['bond_feat_float'].shape[1]

    dim_graph_cate_feat = datas[0][0]['graph_feat_cate'].shape[0]

    atom_feats_cate = torch.zeros(
        (len(datas), max_atom_num, dim_atom_cate_feat), dtype=torch.int64)
    atom_feats_float = torch.zeros(
        (len(datas), max_atom_num, dim_atom_float_feat))
    bond_feats_cate = torch.zeros(
        (len(datas), max_bond_num, dim_bond_cate_feat), dtype=torch.int64)
    bond_feats_float = torch.zeros(
        (len(datas), max_bond_num, dim_bond_float_feat))

    atom_mask = torch.zeros((len(datas), max_atom_num))
    bond_mask = torch.zeros((len(datas), max_bond_num))

    max_edge_num = max_bond_num * 2
    edge_index_ab = torch.zeros(
        (len(datas), 2, max_edge_num), dtype=torch.int64)
    
    edge_mask_ab = torch.zeros(
        (len(datas), max_edge_num))

    edge_index_aa = torch.zeros(
        (len(datas), 2, max_edge_num), dtype=torch.int64)
    edge_mask_aa = torch.zeros(
        (len(datas), max_edge_num))
    edge_aa_bond = torch.zeros(
        (len(datas), max_edge_num), dtype=torch.int64)

    shortest_path_length = torch.zeros(
        (len(datas), max_atom_num, max_atom_num), dtype=torch.int64)
    same_ring_count = torch.zeros(
        (len(datas), max_atom_num, max_atom_num), dtype=torch.int64)
    same_ring_size_min = torch.zeros(
        (len(datas), max_atom_num, max_atom_num), dtype=torch.int64)

    paths2 = -torch.ones(
        (len(datas), max_atom_num, max_atom_num, 2), dtype=torch.int64)
    paths2_atom = -torch.ones(
        (len(datas), max_atom_num, max_atom_num), dtype=torch.int64)
    paths3_atom = -torch.ones(
        (len(datas), max_atom_num, max_atom_num, 2), dtype=torch.int64)
    # edge_index_bb = torch.zeros(
    #     (len(datas), 2, max_bond_edge), dtype=torch.int64)
    # edge_mask_bb = torch.zeros(
    #     (len(datas), max_bond_edge))
    # edge_bb_atom = torch.zeros(
    #     (len(datas), max_bond_edge), dtype=torch.int64)
    
    graph_feats_cate = torch.zeros(
        (len(datas), dim_graph_cate_feat), dtype=torch.int64)
    graph_fps = torch.zeros((len(datas), 2048))

    ys = list()
    
    for i, (data, y) in enumerate(datas):
        bond_index = data['bond_index']
        num_atoms = data['num_atoms']
        atom_feats_cate[i, :num_atoms, :] = data['atom_feat_cate']
        atom_feats_float[i, :num_atoms, :] = data['atom_feat_float']
        atom_mask[i, :num_atoms] = 1

        num_bonds = data['num_bonds']
        bond_feats_cate[i, :num_bonds, :] = data['bond_feat_cate']
        bond_feats_float[i, :num_bonds, :] = data['bond_feat_float']
        bond_mask[i, :num_bonds] = 1

        edge_index_ab[i, 0, :num_bonds] = bond_index[0]
        edge_index_ab[i, 1, :num_bonds] = torch.arange(num_bonds)
        edge_index_ab[i, 0, num_bonds:(2 * num_bonds)] = bond_index[1]
        edge_index_ab[i, 1, num_bonds:(2 * num_bonds)] = torch.arange(
            num_bonds)
        edge_mask_ab[i, :(2 * num_bonds)] = 1

        edge_index_aa[i, 0, :num_bonds] = bond_index[0]
        edge_index_aa[i, 1, :num_bonds] = bond_index[1]
        edge_index_aa[i, 0, num_bonds:(2*num_bonds)] = bond_index[1]
        edge_index_aa[i, 1, num_bonds:(2*num_bonds)] = bond_index[0]

        edge_aa_bond[i, :num_bonds] = torch.arange(num_bonds)
        edge_aa_bond[i, num_bonds:(2*num_bonds)] = torch.arange(num_bonds)

        edge_mask_aa[i, :(2 * num_bonds)] = 1

        shortest_path_length[i, :num_atoms, :num_atoms] = \
            data['shortest_path_length']

        same_ring_count[i, :num_atoms, :num_atoms] = \
            data['atom_same_ring_count']

        same_ring_size_min[i, :num_atoms, :num_atoms] = \
            data['atom_same_ring_min']
        
        ipath = data['shortest_path']
        ipath_atom = data['shortest_path_atom']
        row, col = torch.where(data['shortest_path_length'] == 3)
        for ip in range(len(row)):
            paths2[i, row[ip], col[ip], :] = torch.LongTensor(
                list(ipath[row[ip]][col[ip]]))
            paths2_atom[i, row[ip], col[ip]] = list(
                ipath_atom[row[ip]][col[ip]])[0]
            pass

        row, col = torch.where(data['shortest_path_length'] == 4)
        for ip in range(len(row)):
            paths3_atom[i, row[ip], col[ip], :] = torch.LongTensor(list(
                ipath_atom[row[ip]][col[ip]]))
            pass
        
        # num_edge_bond = data['num_bond_edge']
        # bond_edge = data['bond_edge_index']
        # edge_index_bb[i, 0, :num_edge_bond] = bond_edge[0]
        # edge_index_bb[i, 1, :num_edge_bond] = bond_edge[1]
        # edge_mask_bb[i, :num_edge_bond] = 1
        # 
        # edge_bb_atom[i, :num_edge_bond] = data['bond_edge_atom']
        
        graph_feats_cate[i] = data['graph_feat_cate']
        graph_fps[i, data['graph_fp'].long()] = 1

        ys.append(y)
        pass
    
    return {
        'atom_feat_cate': atom_feats_cate,
        'atom_feat_float': atom_feats_float,
        'atom_mask': atom_mask,
        'bond_feat_cate': bond_feats_cate,
        'bond_feat_float': bond_feats_float,
        'bond_mask': bond_mask,
        'edge_index_ab': edge_index_ab,
        'edge_mask_ab': edge_mask_ab,
        'edge_index_aa': edge_index_aa,
        'edge_mask_aa': edge_mask_aa,
        'edge_aa_bond': edge_aa_bond,
        'shortest_path_length': shortest_path_length,
        'same_ring_count': same_ring_count,
        'same_ring_size_min': same_ring_size_min,
        'paths2': paths2,
        'paths2_atom': paths2_atom,
        'paths3_atom': paths3_atom,
        # 'edge_index_bb': edge_index_bb,
        # 'edge_mask_bb': edge_mask_bb,
        # 'edge_bb_atom': edge_bb_atom,
        'graph_feat_cate': graph_feats_cate,
        'graph_fingerprint': graph_fps,
        'labels': torch.FloatTensor(ys),
    }


def transform_graph_fully_connect(data):
    graph, y = data

    ograph = dict()
    nnodes = graph['num_nodes']

    dim_edge = graph['edge_feat'].shape[1]
    num_edge = graph['edge_feat'].shape[0]
    edge_index_fc = EDGE_FULLY_CONNECT[nnodes]
    edge_feat = np.zeros((edge_index_fc.shape[1], dim_edge), dtype='int64')

    edge_index_dict = EDGE_INDEX_DICT[nnodes]
    data_edge_index = graph['edge_index']
    data_edge_feat = graph['edge_feat']
    for j in range(num_edge):
        edge_feat[
            edge_index_dict[tuple(data_edge_index[:, j].flatten())]] \
            = data_edge_feat[j] + 1
        pass

    ograph['edge_index'] = torch.from_numpy(edge_index_fc)
    ograph['edge_feat'] = torch.from_numpy(edge_feat)
    ograph['node_feat_cate'] = torch.from_numpy(graph['node_feat_cate'])
    ograph['node_feat_float'] = torch.from_numpy(graph['node_feat_float'])
    ograph['num_nodes'] = graph['num_nodes']

    return ograph, y


def transform_graph_preprocess(data):
    graph, y = data

    extra_bond_feat = features.calc_bond_spatial_feat(
        graph['bond_index'], graph['atom_feat_float'][:, :3])
    graph['bond_feat_float'] = torch.cat(
        (graph['bond_feat_float'], extra_bond_feat), dim=-1)
    return graph, y


def transform_graph_add_virtual(data):
    graph, y = data

    ograph = dict()
    ograph['graph_feat'] = graph['graph_feat']
    num_nodes = graph['num_nodes']
    ograph['num_nodes'] = num_nodes + 1

    nrange = np.arange(start=1, stop=(num_nodes+1), dtype='int64')
    nzeros = np.zeros(num_nodes, dtype='int64')
    
    added_edge1 = np.stack(
        (nzeros, nrange))
    added_edge2 = np.stack(
        (nrange, nzeros))

    ograph['edge_index'] = np.hstack(
        (added_edge1, added_edge2, graph['edge_index']+1))

    added_edge_feat = -1 * np.ones(
        (2*num_nodes, graph['edge_feat'].shape[1]), dtype='int64')
    
    ograph['edge_feat'] = np.vstack((added_edge_feat, graph['edge_feat']))

    added_node_feat_cate = -1 * np.ones(
        (1, graph['node_feat_cate'].shape[1]), dtype='int64')
    ograph['node_feat_cate'] = np.vstack(
        (added_node_feat_cate, graph['node_feat_cate']))

    added_node_feat_float = np.zeros(
        (1, graph['node_feat_float'].shape[1]), dtype='float32')
    ograph['node_feat_float'] = np.vstack(
        (added_node_feat_float, graph['node_feat_float']))

    return ograph, y


def transform_graph_to_tensor(data):
    graph, y = data

    ograph = dict()

    for key, value in graph.items():
        if isinstance(value, np.ndarray):
            value = torch.from_numpy(value)
            if key == 'bond_index':
                value = value.long()
            pass
        else:
            pass

        ograph[key] = value
        pass
    
    return ograph, y


def transform_graph_random_rotate(data):
    mat3d = torch.from_numpy(R.random().as_matrix().astype('float32'))
    mat2d = torch.FloatTensor([[0, -1], [1, 0]])

    data[0]['atom_feat_float'][:, :3] = data[0]['atom_feat_float'][:, :3]\
        .matmul(mat3d)

    for i in range(np.random.randint(4)):
        data[0]['atom_feat_float'][:, 3:5] = data[0]['atom_feat_float'][:, 3:5]\
            .matmul(mat2d)

    return data
    pass


def transform_graph_add_bond_edge(data):
    graph, y = data
    bond_edge, bond_edge_atom = bond_index2bond_edge(graph['bond_index'])
    graph['bond_edge_index'] = bond_edge
    graph['bond_edge_atom'] = bond_edge_atom
    graph['num_bond_edge'] = bond_edge.shape[1]

    return graph, y


def transform_graph_select_xyz(data):
    graph, y = data
    graph = graph.copy()
    
    xyz = graph['xyz']

    if xyz.shape[0] == 0:
        xyz = np.zeros((xyz.shape[1], 3), dtype='float32')
        pass
    else:
        rint = np.random.randint(xyz.shape[0])
        xyz = xyz[rint]
        pass

    graph['atom_feat_float'] = np.hstack((xyz, graph['atom_feat_float']))

    graph.pop('xyz')
    
    return graph, y


class WrapperDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, indices, transforms=None):
        super().__init__()
        self.dataset = dataset
        self.indices = indices
        self.transforms = transforms
        pass

    def __getitem__(self, idx):
        '''Get datapoint with index'''

        if isinstance(idx, (int, np.integer)):
            if self.transforms is not None:
                data = self.dataset[self.indices[idx]]
                for trans in self.transforms:
                    data = trans(data)
                    pass
                return data
            
            else:
                return self.dataset[self.indices[idx]]

        raise IndexError(
            'Only integer is valid index (got {}).'.format(type(idx).__name__))

    def __len__(self):
        return len(self.indices)
    

class SimplePCQM4MDataset(torch.utils.data.Dataset):
    def __init__(self, root: str, only_smiles=False,
                 transforms = [transform_graph_to_tensor]):
        self.original_root = root
        self.only_smiles = only_smiles
        self.folder = os.path.join(root, 'pcqm4m_kddcup2021_simple')
            
        self.transforms = transforms

        super().__init__()

        if self.only_smiles:
            self.prepare_smiles()
        else:
            self.prepare_graph()
            pass
            
    def prepare_smiles(self):
        data_df = pd.read_csv(os.path.join(self.folder, 'data.csv.gz'))
        smiles_list = data_df['smiles'].values
        homolumogap_list = data_df['homolumogap'].values
        self.graphs = list(smiles_list)
        self.labels = homolumogap_list

    def prepare_graph(self):
        self.filenames = []
        with open(os.path.join(self.folder, 'filenames.txt')) as fin:
            for line in fin:
                self.filenames.append(line.strip())
                pass
            pass

    def get_idx_split(self):
        with open(os.path.join(self.folder, 'split_dict.pt'), 'rb') as fin:
            split_dict = pickle.load(fin)
            pass
        return split_dict
    
    def __getitem__(self, idx):
        '''Get datapoint with index'''

        if isinstance(idx, (int, np.integer)):
            filename = os.path.join(self.folder, self.filenames[idx])

            with open(filename, 'rb') as fin:
                g, y = pickle.load(fin)
                pass
            
            if self.transforms is None:
                return g, y
            
            for trans in self.transforms:
                g, y = trans((g, y))
                pass
            return g, y

        raise IndexError(
            'Only integer is valid index (got {}).'.format(type(idx).__name__))

    def __len__(self):
        '''Length of the dataset
        Returns
        -------
        int
            Length of Dataset
        '''
        return len(self.filenames)

    def __repr__(self):  # pragma: no cover
        return '{}({})'.format(self.__class__.__name__, len(self))

    pass


def convert_from_smiles(csvfile, splits, root):
    import rdkit
    import rdkit.Chem
    import rdkit.Chem.AllChem

    df = pd.read_csv(csvfile)

    folder = os.path.join(root, 'pcqm4m_kddcup2021_simple')
    filenames_file = os.path.join(folder, 'filenames.txt')

    labels = df['homolumogap'].values
    smiles = df['smiles'].values

    os.system(f'mkdir -p {folder}/data')

    for i in range((len(smiles) // 1000) + 1):
        path = os.path.join(folder, 'data', str(i))
        if not os.path.exists(path):
            os.mkdir(path)
            pass
        pass
    
    process_params = [(i, s, y, folder) for i, (s, y) in enumerate(
        zip(smiles, labels))]
    pool = mp.Pool()
    results = pool.imap(process_smiles_func, tqdm(process_params))
    pool.close()

    filenames = []
    failCnt = 0
    failMMFCnt = 0
    for fn, failEmbed, mmf_result in results:
        filenames.append(fn)

        if failEmbed:
            failCnt += 1
            pass

        if mmf_result == -1:
            failMMFCnt += 1
            pass
        pass

    print(failCnt)
    print(failMMFCnt)
    
    with open(filenames_file, 'w') as fout:
        for fn in filenames:
            fout.write(fn)
            fout.write('\n')
            pass
        pass

    with open(
            os.path.join(folder, 'split_dict.pt'), 'wb') as fout:
        pickle.dump(
            splits, fout, protocol=pickle.HIGHEST_PROTOCOL)
        pass
    pass
