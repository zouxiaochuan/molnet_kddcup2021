
import pytorch_lightning as pyl
import ogb.lsc
import torch_geometric as pyg
import torch_geometric.nn as pyg_nn
import json
import torch
import torch.nn as nn
import dataio as dataio
import time
from run_molnet import GNNLightning
import sys
import numpy as np
import pickle
import time


if __name__ == '__main__':

    start = time.time()
    datafile = sys.argv[1]
    modelfile = sys.argv[2]

    with open(datafile, 'rb') as fin:
        graphs = pickle.load(fin)
        pass
    
    with open('./molnet.json') as fin:
        config = json.load(fin)
        pass

    graphs = [(g, 0) for g in graphs]
    print('begin load dataset')
    start = time.time()
    dataset_test = dataio.WrapperDataset(
        graphs, list(range(len(graphs))),
        transforms=[dataio.transform_graph_select_xyz,
                    dataio.transform_graph_to_tensor,
                    dataio.transform_graph_preprocess])
    
    print(f'load dataset time: {time.time()-start}')
    
    test_loader = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=config['batch_size'], shuffle=False,
        num_workers=1,
        collate_fn=dataio.collate_graph)

    model = GNNLightning(config)

    model.load_state_dict(
        torch.load(
            modelfile,
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

    input_dict = {'y_pred': pred}
    evaluator = ogb.lsc.PCQM4MEvaluator()
    evaluator.save_test_submission(
        input_dict = input_dict, dir_path = './')

    print('time cost: {0}'.format(time.time() - start))
    pass
    
