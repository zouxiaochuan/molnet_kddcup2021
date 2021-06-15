import antio
import pytorch_lightning as pyl
import model_gnn_fc as model_gnn
import ogb.lsc
import torch_geometric as pyg
import torch_geometric.nn as pyg_nn
import json
import torch
import torch.nn as nn
import dataio2 as dataio
import time
from run_molnet import GNNLightning
import sys
import numpy as np


if __name__ == '__main__':
    root = '../../datasets'

    model_path = sys.argv[1]
    split = sys.argv[2]
    
    with open('./molnet.json') as fin:
        config = json.load(fin)
        pass

    print('begin load dataset')
    start = time.time()
    dataset = dataio.SimplePCQM4MDataset(
        root, transforms=[dataio.transform_graph_select_xyz,
                          dataio.transform_graph_to_tensor,
                          dataio.transform_graph_preprocess])
    
    splits = dataset.get_idx_split()
    dataset_test = dataio.WrapperDataset(dataset, splits[split])
    print(len(splits[split]))
    
    print(f'load dataset time: {time.time()-start}')
    split_idx = dataset.get_idx_split()
    
    test_loader = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=config['batch_size'], shuffle=False,
        num_workers=config['num_data_workers'],
        collate_fn=dataio.collate_graph)

    model = GNNLightning(config)

    model.load_state_dict(
        torch.load(
            model_path,
            # map_location=torch.device('cpu')
        )['state_dict']
    )

    trainer = pyl.Trainer(
        gpus=[0]
        )

    # torch.autograd.set_detect_anomaly(True)

    model.eval()
    with torch.no_grad():
        pred = trainer.predict(
            model, test_loader)
        pass

    pred = torch.cat(pred, dim=0).cpu().detach().numpy().flatten()

    pred[pred>50] = 50
    pred[pred<0] = 0

    print(pred.shape)

    if split == 'test':
        input_dict = {'y_pred': pred}
        evaluator = ogb.lsc.PCQM4MEvaluator()
        evaluator.save_test_submission(
            input_dict = input_dict, dir_path = './')
    else:
        np.save('predict_' + split + '.npy', pred)
        pass
    
