"""Single-process shuffled batches with a serializable cursor (no worker prefetch)."""

import torch
from torch.utils.data import default_collate


class StatefulBatchStream:
    """Infinite epochs, retaining the short final batch; dataset access must be deterministic."""

    def __init__(
        self, dataset, batch_size: int, *, data_id: str, seed: int = 0, shuffle: bool = True
    ):
        if len(dataset) == 0 or batch_size <= 0 or not data_id:
            raise ValueError("Nonempty dataset, positive batch_size and data_id required")
        self.dataset = dataset
        self.batch_size = batch_size
        self.data_id = data_id
        self.shuffle = shuffle
        self.generator = torch.Generator().manual_seed(seed)
        self.order = (
            torch.randperm(len(dataset), generator=self.generator)
            if shuffle
            else torch.arange(len(dataset))
        )
        self.cursor = 0
        self.epoch = 0
        self.last_indices = []

    def __iter__(self):
        return self

    def __next__(self):
        if self.cursor == len(self.dataset):
            self.order = (
                torch.randperm(len(self.dataset), generator=self.generator)
                if self.shuffle
                else torch.arange(len(self.dataset))
            )
            self.cursor = 0
            self.epoch += 1
        end = min(self.cursor + self.batch_size, len(self.dataset))
        indices = self.order[self.cursor : end].tolist()
        batch = default_collate([self.dataset[i] for i in indices])
        self.cursor = end
        self.last_indices = indices
        return batch

    def state_dict(self):
        return {
            "version": 1,
            "data_id": self.data_id,
            "shuffle": self.shuffle,
            "size": len(self.dataset),
            "batch_size": self.batch_size,
            "epoch": self.epoch,
            "cursor": self.cursor,
            "order": self.order.clone(),
            "generator": self.generator.get_state().clone(),
        }

    def load_state_dict(self, state):
        expected = (1, self.data_id, len(self.dataset), self.batch_size, self.shuffle)
        actual = (
            state["version"],
            state["data_id"],
            state["size"],
            state["batch_size"],
            state["shuffle"],
        )
        if actual != expected:
            raise ValueError("Data identity, size or batch configuration changed")
        order = state["order"].cpu()
        if order.dtype != torch.int64 or not torch.equal(
            order.sort().values, torch.arange(len(self.dataset))
        ):
            raise ValueError("Invalid saved data permutation")
        if not 0 <= state["cursor"] <= len(self.dataset) or state["epoch"] < 0:
            raise ValueError("Invalid epoch/cursor")
        self.order = order.clone()
        self.cursor = state["cursor"]
        self.epoch = state["epoch"]
        self.generator.set_state(state["generator"].cpu())
        self.last_indices = []
