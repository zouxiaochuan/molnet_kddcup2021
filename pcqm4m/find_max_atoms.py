
import antio
import dataio

if __name__ == '__main__':
    root = '../../datasets'
    antio.download_pcqm4m_simple(root)
    dataset = dataio.SimplePCQM4MDataset(root)

    max_atoms = 0
    for data, _ in dataset:
        max_atoms = max(max_atoms, data['node_feat'].shape[0])
        pass

    print(max_atoms)
