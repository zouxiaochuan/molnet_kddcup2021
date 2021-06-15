import sys
import ogb.lsc
import dataio2 as dataio
import os

if __name__ == '__main__':
    root = sys.argv[1]
    dataset_origin = ogb.lsc.PCQM4MDataset(root, only_smiles=True)

    dataio.convert_from_smiles(
        os.path.join(dataset_origin.folder, 'raw', 'data.csv.gz'),
        dataset_origin.get_idx_split(),
        root)
    pass
    
