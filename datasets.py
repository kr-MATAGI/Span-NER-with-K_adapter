import numpy as np
import torch
import re

from torch.utils.data import Dataset

#===============================================================
class SpanNERDataset(Dataset):
#===============================================================
    def __init__(
            self,
            data: np.ndarray, label_ids: np.ndarray,
            all_span_idx:np.ndarray, all_span_len: np.ndarray,
            real_span_mask: np.ndarray, span_only_label: np.ndarray,
            pos_ids: np.ndarray
    ):
        self.input_ids = torch.LongTensor(data[:][:, :, 0])
        self.attn_mask = torch.LongTensor(data[:][:, :, 1])
        self.token_type_ids = torch.LongTensor(data[:][:, :, 2])
        self.label_ids = torch.LongTensor(label_ids)

        self.all_span_idx = torch.LongTensor(all_span_idx)
        self.all_span_len = torch.LongTensor(all_span_len)
        self.real_span_mask = torch.LongTensor(real_span_mask)
        self.span_only_label = torch.LongTensor(span_only_label)

        self.pos_ids = torch.LongTensor(pos_ids)

    def __len__(self):
        return self.input_ids.size()[0]

    def __getitem__(self, idx):
        items = {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attn_mask[idx],
            "token_type_ids": self.token_type_ids[idx],
            "label_ids": self.label_ids[idx],

            "all_span_idx": self.all_span_idx[idx],
            "all_span_len": self.all_span_len[idx],
            "real_span_mask": self.real_span_mask[idx],
            "span_only_label": self.span_only_label[idx],

            "pos_ids": self.pos_ids[idx],
        }

        return items

#===============================================================
class RcAdapterDatasets(Dataset):
#===============================================================
    def __init__(
            self,
            input_ids: np.ndarray, label_ids: np.ndarray,
            attn_mask: np.ndarray, tok_type_ids: np.ndarray,
            subj_start_id: np.ndarray, obj_start_id: np.ndarray
    ):
        self.input_ids = torch.LongTensor(input_ids)
        self.attn_mask = torch.LongTensor(attn_mask)
        self.tok_type_ids = torch.LongTensor(tok_type_ids)
        self.label_ids = torch.LongTensor(label_ids)
        self.subj_start_id = torch.FloatTensor(subj_start_id)
        self.obj_start_id = torch.FloatTensor(obj_start_id)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        items = {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attn_mask[idx],
            "token_type_ids": self.tok_type_ids[idx],
            "label_ids": self.label_ids[idx],
            "subj_start_id": self.subj_start_id[idx],
            "obj_start_id": self.obj_start_id[idx]
        }

        return items

#===============================================================
class DpAdapterDatasets(Dataset):
#===============================================================
    def __init__(self,
                 input_ids: np.ndarray, attn_mask: np.ndarray,
                 bpe_head_mask: np.ndarray, bpe_tail_mask: np.ndarray,
                 dep_ids: np.ndarray, head_ids: np.ndarray, pos_ids: np.ndarray):

        self.input_ids = torch.LongTensor(input_ids)
        self.attn_mask = torch.LongTensor(attn_mask)
        self.bpe_head_mask = torch.LongTensor(bpe_head_mask)
        self.bpe_tail_mask = torch.LongTensor(bpe_tail_mask)
        self.dep_ids = torch.LongTensor(dep_ids)
        self.head_ids = torch.LongTensor(head_ids)
        self.pos_ids = torch.LongTensor(pos_ids)

        self.max_word_length = max(torch.sum(self.bpe_head_mask, dim=1)).item()

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        item = (
            self.input_ids[idx],
            self.attn_mask[idx],
            self.bpe_head_mask[idx],
            self.bpe_tail_mask[idx],
            self.dep_ids[idx],
            self.head_ids[idx],
            self.pos_ids[idx]
        )

        return item