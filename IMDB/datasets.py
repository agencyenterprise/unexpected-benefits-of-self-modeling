import torch
import torchtext
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader
from typing import Optional

import random
import numpy as np
import lightning as L
from models import configure_random_seed

tokenizer = get_tokenizer("basic_english")


def seed_worker(worker_id):
    random_seed = torch.initial_seed() % 2 ** 32
    configure_random_seed(random_seed)


def yield_tokens(data_iter):
    for label, text in data_iter:
        yield tokenizer(text)


class CustomSampler(torch.utils.data.Sampler):
    def __init__(self, data_source, seeds):
        self.data_source = data_source
        self.seeds = seeds

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seeds)
        indices = torch.randperm(len(self.data_source), generator=g).tolist()
        return iter(indices)

    def __len__(self):
        return len(self.data_source)


class TextDataModule(L.LightningDataModule):
    def __init__(
            self,
            data_dir: str = "./data",
            dataset: str = "AG_NEWS",
            batch_size: int = 128,
            random_seed: Optional[int] = None,
    ):
        super().__init__()
        self.dataset_name = dataset
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.random_seed = random_seed
        configure_random_seed(random_seed)

        dataset_class = getattr(torchtext.datasets, dataset)
        self.train_split = list(dataset_class(root=self.data_dir, split="train"))
        self.val_split = list(dataset_class(root=self.data_dir, split="test"))

        self.vocab = build_vocab_from_iterator(
            yield_tokens(iter(self.train_split)), specials=["<unk>"]
        )
        self.vocab.set_default_index(self.vocab["<unk>"])
        self.with_offsets = True

    def collate_batch(self, batch):
        label_list, text_list, offsets = [], [], [0]
        for label, text in batch:
            label_list.append(int(label) - 1)
            processed_text = torch.tensor(
                self.vocab(tokenizer(text)), dtype=torch.int64
            )
            text_list.append(processed_text)
            offsets.append(processed_text.size(0))
        label_list = torch.tensor(label_list, dtype=torch.int64)
        offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
        text_list = torch.cat(text_list)
        if self.with_offsets:
            return label_list, text_list, offsets
        else:
            num_data = torch.tensor([len(label_list)])
            input = torch.cat([offsets, text_list, num_data], dim=0)
            return input, label_list

    def train_dataloader(self):
        return DataLoader(
            self.train_split,
            batch_size=self.batch_size,
            shuffle=(self.random_seed is None),
            sampler=(
                CustomSampler(self.train_split, self.random_seed)
                if self.random_seed is not None
                else None
            ),
            collate_fn=self.collate_batch,
            num_workers=2,
            persistent_workers=True,
            worker_init_fn=seed_worker,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_split,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_batch,
            num_workers=2,
            persistent_workers=True,
            worker_init_fn=seed_worker,
        )

    def test_dataloader(self):
        return DataLoader(
            self.val_split,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_batch,
            num_workers=2,
            persistent_workers=True,
            worker_init_fn=seed_worker,
        )

    def get_num_classes(self):
        classes = set()
        if self.with_offsets:
            for labels, _, _ in self.train_dataloader():
                classes.update(labels.tolist())
        else:
            for labels, _ in self.train_dataloader():
                classes.update(labels.tolist())
        return len(classes)


if __name__ == "__main__":
    data_module = TextDataModule(dataset="IMDB")
    data_module.prepare_data()
    print(data_module.get_num_classes())
    print(len(data_module.vocab))