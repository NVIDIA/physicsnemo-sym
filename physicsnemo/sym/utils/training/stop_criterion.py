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

import numpy as np
from typing import Any, Dict


class StopCriterion:
    """
    Stop criterion for training

    Parameters
    ----------
    metric : str
        Metric to be monitored during the training
    min_delta : float
        minimum required change in the metric to qualify as a training improvement
    patience : float
        Number of training steps to wait for a training improvement to happen
    mode: str
        Choose 'min' if the metric is to be minimized, or 'max' if the metric is to be maximized
    freq: int
        Frequency of evaluating the stop criterion
    strict: bool
        If True, raises an error in case the metric is not valid.
    monitor_freq: Any
        Frequency of evaluating the monitor domain
    validation_freq: Any
        Frequency of evaluating the validation domain
    """

    def __init__(
        self,
        metric: str,
        min_delta: float = 0.0,
        patience: float = 0,
        mode: str = "min",
        freq: int = 1000,
        strict: bool = True,
        monitor_freq: Any = None,
        validation_freq: Any = None,
    ):
        self.metric = metric
        self.min_delta = min_delta
        self.patience = patience
        self.mode = mode
        self.freq = freq
        self.strict = strict
        self.monitor_freq = monitor_freq
        self.validation_freq = validation_freq

        self.best_score = None
        self.counter = 0
        self.check_freqs = True

        if self.freq > self.patience:
            raise RuntimeError(
                "Stop criterion patience should be greater than or equal to the freq for stopping criterion"
            )
        self.mode_dict = {"min": np.less, "max": np.greater}
        if self.mode not in self.mode_dict.keys():
            raise RuntimeError("Stop criterion mode can be either min or max")
        self.mode_op = self.mode_dict[self.mode]
        self.min_delta *= 1 if self.mode == "max" else -1

    def evaluate(self, metric_dict: Dict[str, float]) -> bool:
        """
        Evaluate the stop criterion

        Parameters
        ----------
        metric_dict : str
            Dictionary of available metrics to compute
        """

        if self.check_freqs:
            self._check_frequencies(metric_dict)
        score = self._get_score(metric_dict, self.target_key)
        if self.best_score is None:
            self.best_score = score
        elif self.mode_op(self.best_score + self.min_delta, score):
            if self.mode_op(score, self.best_score):
                self.best_score = score
            self.counter += self.freq
            if self.counter >= self.patience:
                return True
        else:
            self.best_score = score
            self.counter = 0
            return False

    def _check_frequencies(self, metric_dict):
        found_metric = False
        for key in metric_dict.keys():
            for k in metric_dict[key].keys():
                if self.metric == k:
                    self.target_key = key
                    found_metric = True
                    break
        if not found_metric and self.strict:
            raise RuntimeError(
                "[physicsnemo.sym.stop_criterion] the specified metric for stopping criterion is not valid"
            )
        if self.target_key == "monitor" and (
            self.freq % self.monitor_freq != 0 or self.freq == 0
        ):
            raise RuntimeError(
                "Stop criterion frequency should be a multiple of the monitor frequency"
            )
        elif self.target_key == "validation" and (
            self.freq % self.validation_freq != 0 or self.freq == 0
        ):
            raise RuntimeError(
                "Stop criterion frequency should be a multiple of the validation frequency"
            )
        self.check_freqs = False

    def _get_score(self, metric_dict, target_key):
        return metric_dict[target_key][self.metric]
