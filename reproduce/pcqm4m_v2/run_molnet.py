
import pytorch_lightning as pyl
import model_molnet
import ogb.lsc
import torch_geometric as pyg
import torch_geometric.nn as pyg_nn
import json
import torch
import torch.nn as nn
import dataio as dataio
import time
import numpy as np
import sys


class GNNLightning(pyl.LightningModule):

    def __init__(self, config):
        super().__init__()

        self.gnn = model_molnet.MoleculeNet(config)
        self.regressor = nn.Sequential(
            nn.Linear(config['hidden_size'],
                      config['hidden_size']),
            nn.Tanh(),
            nn.Linear(config['hidden_size'], 1))
        self.config = config
        pass

    def forward(self, data):
        atom_feat_cate = data['atom_feat_cate']
        atom_feat_float = data['atom_feat_float']
        atom_mask = data['atom_mask']
        bond_feat_cate = data['bond_feat_cate']
        bond_feat_float = data['bond_feat_float']
        bond_mask = data['bond_mask']
        edge_index_ab = data['edge_index_ab']
        edge_mask_ab = data['edge_mask_ab']
        edge_index_aa = data['edge_index_aa']
        edge_mask_aa = data['edge_mask_aa']
        edge_aa_bond = data['edge_aa_bond']
        shortest_path_length = data['shortest_path_length']
        same_ring_count = data['same_ring_count']
        paths2 = data['paths2']
        # edge_index_bb = data['edge_index_bb']
        # edge_mask_bb = data['edge_mask_bb']
        # edge_bb_atom = data['edge_bb_atom']

        graph_feat_cate = data['graph_feat_cate']
        graph_fingerprint = data['graph_fingerprint']

        emb = self.gnn(
            atom_feat_cate, atom_feat_float, atom_mask,
            bond_feat_cate, bond_feat_float, bond_mask,
            edge_index_ab, edge_mask_ab, edge_index_aa, edge_mask_aa,
            edge_aa_bond,
            shortest_path_length, same_ring_count, paths2,
            # edge_index_bb, edge_mask_bb, edge_bb_atom,
            graph_feat_cate, graph_fingerprint)

        emb = emb[:, 0, :]
        
        # emb = pyg_nn.global_mean_pool(emb, batch_idx)

        return self.regressor(emb)

    def training_step(self, batch, batch_idx):
        pred = self(batch)
        y = batch['labels']
        loss = nn.functional.l1_loss(pred.view(-1), y)
        return loss

    def validation_step(self, batch, batch_idx):
        y = batch['labels']
        pred = torch.clamp(self(batch).view(-1), 0, 50)
        loss = nn.functional.mse_loss(pred, y, reduction='sum')
        absdiff = nn.functional.l1_loss(pred, y, reduction='sum')

        return loss, absdiff, y.shape[0]

    def validation_epoch_end(self, results):
        total_num = sum([num for _, _, num in results])
        valid_mse = torch.sum(torch.stack(
            [loss for loss, _, _ in results]))/total_num
        valid_mae = torch.sum(torch.stack(
            [absdiff for _, absdiff, _ in results]))/total_num
        self.log('valid loss', valid_mse, sync_dist=True)
        self.log('valid mae', valid_mae, sync_dist=True)
        self.log('learning rate', self.optimizers(0).param_groups[0]['lr'])
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay'])
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=10, gamma=0.8)
        
        return [optimizer], [scheduler]


if __name__ == '__main__':
    root = '../../datasets'
    
    with open('./molnet.json') as fin:
        config = json.load(fin)
        pass

    print('begin load dataset')
    start = time.time()
    dataset = dataio.SimplePCQM4MDataset(
        root, transforms=[
            dataio.transform_graph_select_xyz,
            dataio.transform_graph_to_tensor])
    
    splits = dataset.get_idx_split()
    dataset_train = dataio.WrapperDataset(
        dataset, np.hstack((splits['train'], splits['valid'])),
        transforms=[
            dataio.transform_graph_random_rotate,
            dataio.transform_graph_preprocess
        ])
    
    dataset_valid = dataio.WrapperDataset(
        dataset, splits['valid'], [
            dataio.transform_graph_preprocess
        ])
    
    print(f'load dataset time: {time.time()-start}')
    split_idx = dataset.get_idx_split()
    
    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=config['batch_size'], shuffle=True,
        num_workers=config['num_data_workers'],
        collate_fn=dataio.collate_graph)
    valid_loader = torch.utils.data.DataLoader(
        dataset_valid,
        batch_size=config['batch_size'],
        num_workers=config['num_data_workers'],
        collate_fn=dataio.collate_graph)

    model = GNNLightning(config)

    if len(sys.argv) > 1:
        model.load_state_dict(torch.load(sys.argv[1])['state_dict'])
        pass

    trainer = pyl.Trainer(
        logger=pyl.loggers.CSVLogger('./lightning_logs/logs.csv', 'big_trainvalid_mid'),
        gpus=8,
        max_epochs=100,
        accelerator='ddp')

    # torch.autograd.set_detect_anomaly(True)
    trainer.fit(
        model, train_dataloader=train_loader,
        val_dataloaders=valid_loader)
    
