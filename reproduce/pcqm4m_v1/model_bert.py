import torch
import torch.nn as nn

import transformers
from ogb.lsc import PCQM4MDataset


class PCQM4MCharactorDataset(torch.utils.data.Dataset):
    def __init__(self, root: str, split: str = None):
        '''
        param split: value of train, valid, test
        '''
        self._raw_dataset = PCQM4MDataset(root=root, only_smiles=True)
        self._max_length = 180
        all_chs = set()
        for smile, _ in self._raw_dataset:
            all_chs.update(list(smile))
            pass

        self._ch_dict = {ch: i+3 for i, ch in enumerate(sorted(list(all_chs)))}
        if split is not None:
            self._indices = self._raw_dataset.get_idx_split()[split]
        else:
            self._indices = range(len(self._raw_dataset))
            pass
        
        # if split == 'valid':
        #     self._indices = self._indices[: 10000]
        pass

    def __len__(self):
        return len(self._indices)

    def __getitem__(self, idx):
        smile, y = self._raw_dataset[self._indices[idx]]

        # print(len(smile))
        x = [1] + [self._ch_dict[ch] for ch in list(smile)] + [2]
        xx = torch.zeros(self._max_length, dtype=torch.long)
        xx[:len(x)] = torch.LongTensor(x)
        y = torch.FloatTensor([y])

        mask = torch.zeros(self._max_length)
        mask[:len(x)] = 1

        return {'input_ids': xx, 'attention_mask': mask, 'labels': y}
    pass


class PCQM4MCharactorModel(nn.Module):
    def __init__(self, encoder, hidden_size):
        super().__init__()

        self._encoder = encoder
        self._regressor = nn.Linear(hidden_size, 1)
        pass

    def forward(self, **data):
        input_ids = data['input_ids']
        mask = data['attention_mask']
        
        emb = self._encoder(input_ids, mask).pooler_output

        return (self._regressor(emb),)
        pass
    pass


class PCQM4MTrainer(transformers.Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        pred = outputs[0]
        loss_fct = torch.nn.MSELoss()
        loss = loss_fct(pred.view(-1), labels.float().view(-1))

        # print(loss.detach().item())
        return (loss, (loss, outputs)) if return_outputs else loss
    pass
