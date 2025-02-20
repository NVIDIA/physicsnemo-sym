# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Union, List

import torch
import logging
from torch.utils.data import DataLoader, BatchSampler, SequentialSampler, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from typing import Union, List

from physicsnemo.sym.node import Node
from physicsnemo.sym.constants import tf_dt
from physicsnemo.sym.distributed.manager import DistributedManager
from physicsnemo.sym.dataset import Dataset, IterableDataset
from physicsnemo.sym.loss import Loss
from physicsnemo.sym.graph import Graph
from physicsnemo.sym.key import Key

logger = logging.getLogger(__name__)
Tensor = torch.Tensor


class Constraint:
    """Base class for constraints"""

    def __init__(
        self,
        nodes: List[Node],
        dataset: Union[Dataset, IterableDataset],
        loss: Loss,
        batch_size: int,
        shuffle: bool,
        drop_last: bool,
        num_workers: int,
    ):
        # Get DDP manager
        self.manager = DistributedManager()
        self.device = self.manager.device
        if not drop_last and self.manager.cuda_graphs:
            logger.info("drop_last must be true when using cuda graphs")
            drop_last = True

        # get dataset and dataloader
        self.dataset = dataset
        self.dataloader = iter(
            Constraint.get_dataloader(
                dataset=self.dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                drop_last=drop_last,
                num_workers=num_workers,
            )
        )

        # construct model from nodes
        self.model = Graph(
            nodes,
            Key.convert_list(self.dataset.invar_keys),
            Key.convert_list(self.dataset.outvar_keys),
        )
        self.model.to(self.device)
        if self.manager.distributed:
            # https://pytorch.org/docs/master/notes/cuda.html#id5
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                self.model = DistributedDataParallel(
                    self.model,
                    device_ids=[self.manager.local_rank],
                    output_device=self.device,
                    broadcast_buffers=self.manager.broadcast_buffers,
                    find_unused_parameters=self.manager.find_unused_parameters,
                    process_group=self.manager.group(
                        "data_parallel"
                    ),  # None by default
                )
            torch.cuda.current_stream().wait_stream(s)

        self._input_names = Key.convert_list(dataset.invar_keys)
        self._output_names = Key.convert_list(dataset.outvar_keys)

        self._input_vars = None
        self._target_vars = None
        self._lambda_weighting = None

        # put loss on device
        self._loss = loss.to(self.device)

    @property
    def input_names(self) -> List[Key]:
        return self._input_names

    @property
    def output_names(self) -> List[Key]:
        return self._output_names

    def load_data(self):
        raise NotImplementedError("Subclass of Constraint needs to implement this")

    def load_data_static(self):
        raise NotImplementedError("Subclass of Constraint needs to implement this")

    def loss(self, step: int):
        raise NotImplementedError("Subclass of Constraint needs to implement this")

    def save_batch(self, filename: str):
        raise NotImplementedError("Subclass of Constraint needs to implement this")

    @staticmethod
    def _set_device(tensor_dict, device=None, requires_grad=False):

        # convert np to torch if needed
        tensor_dict = {
            key: torch.as_tensor(value, dtype=tf_dt, device=device)
            for key, value in tensor_dict.items()
        }

        # set requires_grad if needed
        if requires_grad:
            tensor_dict = {
                key: value.requires_grad_(requires_grad)
                for key, value in tensor_dict.items()
            }

        return tensor_dict

    @staticmethod
    def get_dataloader(
        dataset: Union[Dataset, IterableDataset],
        batch_size: int,
        shuffle: bool,
        drop_last: bool,
        num_workers: int,
        distributed: bool = None,
        infinite: bool = True,
    ):
        "Return an appropriate dataloader given a dataset"

        assert isinstance(dataset, Dataset) or isinstance(
            dataset, IterableDataset
        ), "error, dataset must be a subclass of Dataset or IterableDataset"

        manager = DistributedManager()

        # use persistent workers
        # this is important for small datasets - torch would otherwise spend a lot of CPU overhead spawning workers each epoch
        persistent_workers = True if num_workers > 0 else False

        # map-style
        if isinstance(dataset, Dataset):

            assert batch_size is not None, "error, batch_size must be specified"
            assert shuffle is not None, "error, shuffle must be specified"
            assert drop_last is not None, "error, drop_last must be specified"

            # if distributed, use distributed sampler
            if distributed is not False and manager.distributed:
                sampler = DistributedSampler(
                    dataset,
                    num_replicas=manager.group_size("data_parallel"),
                    rank=manager.group_rank("data_parallel"),
                    shuffle=shuffle,
                    drop_last=drop_last,
                )

            # otherwise use standard sampler
            else:
                if shuffle:
                    sampler = RandomSampler(dataset)
                else:
                    sampler = SequentialSampler(dataset)

            # get batch sampler
            batch_sampler = BatchSampler(sampler, batch_size, drop_last)

            # if the dataset does auto collation, turn off automatic batching in dataloader
            # this passes batched indices directly to dataset
            # i.e. the dataloader yields default_convert(dataset[idx])
            # see https://github.com/pytorch/pytorch/blob/master/torch/utils/data/_utils/fetch.py
            # note: may need to use torch.set_num_threads if array indexing tensors in dataset to avoid excessive threading
            if dataset.auto_collation:
                dataloader = DataLoader(
                    dataset,
                    batch_size=None,
                    sampler=batch_sampler,
                    pin_memory=True,
                    num_workers=num_workers,
                    worker_init_fn=dataset.worker_init_fn,
                    persistent_workers=persistent_workers,
                )

            # otherwise turn on automatic batching in dataloader
            # this passes single indices to the dataset
            # i.e. the dataloader yields default_collate([dataset[i] for i in idx])
            else:
                dataloader = DataLoader(
                    dataset,
                    batch_sampler=batch_sampler,
                    pin_memory=True,
                    num_workers=num_workers,
                    worker_init_fn=dataset.worker_init_fn,
                    persistent_workers=persistent_workers,
                )

        # iterable-style
        elif isinstance(dataset, IterableDataset):

            # for iterable datasets, must do batching/sampling within dataset
            dataloader = DataLoader(
                dataset,
                batch_size=None,
                pin_memory=True,
                num_workers=num_workers,
                worker_init_fn=dataset.worker_init_fn,
                persistent_workers=persistent_workers,
            )

        # make dataloader infinite
        if infinite:
            dataloader = InfiniteDataLoader(dataloader)

        # initialise dataset if on main thread
        if num_workers == 0:
            dataset.worker_init_fn(0)

        return dataloader


class InfiniteDataLoader:
    "An infinite dataloader, for use with map-style datasets to avoid StopIteration after each epoch"

    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.epoch = 0

    def __iter__(self):
        while True:
            dataloader = iter(self.dataloader)
            for batch in dataloader:
                yield batch
            self.epoch += 1
