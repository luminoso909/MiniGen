"""单进程确定数据访问的排列与游标恢复，无 worker 预取。"""

from typing import Any, Self

import torch
from torch.utils.data import Dataset, default_collate

from minigen._typing import Batch, Checkpoint


class StatefulBatchStream:
    """可序列化排列与游标的单进程无限 batch 流。

    Args:
        dataset: 访问确定的数据集，无 worker 预取。
        batch_size: 正 batch 大小；保留每轮最后不足一批的样本。
        data_id: 用于恢复兼容性检查的非空数据身份。
        seed: 独立排序发生器种子。
        shuffle: 是否每轮重新生成随机排列。

    Raises:
        ValueError: 空数据、非正 batch_size 或空 data_id。

    Results:
        StatefulBatchStream: 自身可迭代的 batch 流，state_dict 可保存精确数据位置。

    副作用:
        创建实例内独立排列 RNG、游标和轮次，不改变全局 Torch RNG。
    """

    def __init__(
        self,
        dataset: Dataset[Any],
        batch_size: int,
        *,
        data_id: str,
        seed: int = 0,
        shuffle: bool = True,
    ) -> None:
        """初始化独立随机排列、轮次与读取游标。

        Args:
            dataset: 支持长度与按索引读取的 Torch 数据集；样本结构由 default_collate 保留。
            batch_size: 每批最多读取的样本数。
            data_id: 重建时必须一致的数据身份。
            seed: 排序发生器种子，不替代其他训练 RNG。
            shuffle: False 时按顺序读取，适合固定监督检查。

        Raises:
            ValueError: 数据为空、batch_size 非正或 data_id 为空。

        副作用:
            创建实例内独立排列 RNG、游标和轮次，不改变全局 Torch RNG。
        """
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

    def __iter__(
        self,
    ) -> Self:
        """返回自身以支持 iter 和训练循环。

        Results:
            Self: 当前 StatefulBatchStream 实例；不会重置游标。
        """
        return self

    def __next__(
        self,
    ) -> Batch:
        """读取下一批并在轮次结束时建立新排列。

        Raises:
            IndexError/TypeError/RuntimeError: 数据读取或样本无法 collate 时由下层透传。

        Results:
            Batch: default_collate 得到的 batch: 本项目使用张量或张量列表/元组；最后一批可以较短。

        副作用:
            推进 cursor 和 last_indices；换轮时更新 epoch、排列及实例 RNG。
        """
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

    def state_dict(
        self,
    ) -> Checkpoint:
        """导出数据身份、排列、游标和独立排序 RNG。

        Results:
            Checkpoint: 可恢复的数据流状态。
                data_id/size/batch_size/shuffle: 数据和分批兼容性条件。
                epoch/cursor: 轮次与已消费样本位置。
                order: [n_s] 整数排列副本。
                generator: 实例排序 RNG 的字节状态。
        """
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

    def load_state_dict(
        self,
        state: Checkpoint,
    ) -> None:
        """验证身份与排列，再恢复精确的单进程数据位置。

        Args:
            state: 由 state_dict 生成的状态，需匹配版本、数据规模、batch_size 和 shuffle。

        Raises:
            ValueError: 身份/配置不符、排列非法、epoch 或 cursor 越界。
            KeyError: 必要状态键缺失。

        副作用:
            覆盖当前流的排列、游标、轮次、排序 RNG 和 last_indices。
        """
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
