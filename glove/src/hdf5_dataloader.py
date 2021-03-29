from dataclasses import dataclass
import random
import contextlib

import torch


@dataclass
class CooccurrenceDataset(Dataset):
    token_ids: torch.Tensor
    cooccurr_counts: torch.Tensor

    def __getitem__(self, index):
        return [self.token_ids[index], self.cooccurr_counts[index]]
    
    def __len__(self):
        return self.token_ids.size()[0]


@dataclass
class HDF5DataLoader:
    filepath: str
    dataset_name: str
    batch_size: int

    def iter_batches(self, dataset):
        chunks = list(dataset.iter_chunks())
        random.shuffle(chunks)
        for chunk in chunks:
            chunked_dataset = dataset[chunk]
            dataloader = DataLoader(
                dataset=CooccurrenceDataset(
                    token_ids=torch.from_numpy(chunked_dataset[:,:2]).long(),
                    cooccurr_counts=torch.from_numpy(chunked_dataset[:,
                        2]).double()
                ),
                batch_size=self.batch_size,
                shuffle=True,
                pin_memory=True
            )
            for batch in dataloader:
                yield batch

    @contextlib.contextmanager
    def generator(self):
        file = h5py.File(self.filepath, "r")
        dataset = file[self.dataset_name]
        yield self.iter_batches(dataset)
        file.close()
