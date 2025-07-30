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

import glob
import os
import numpy as np
from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt


def read_tensorboard_eventfiles(path):
    assert len(glob.glob(path + "*events*")) != 0, "No event files found"

    if len(glob.glob(path + "*events*")) > 1:
        # print("Found more than one event files")
        events_data = []
        run_time = []
        for i in glob.glob(path + "*events*"):
            ea = event_accumulator.EventAccumulator(
                i, size_guidance={event_accumulator.TENSORS: 0}
            )
            ea.Reload()
            tensors_list = ea.Tags()["tensors"]
            tensors_list.remove("config/text_summary")

            event_data = {}
            for t in tensors_list:
                event_data[t] = tf_log_to_np(ea.Tensors(t))

            if (
                len(event_data["Train/learning_rate"]["w_time"]) != 1
            ):  # skip the eventfiles that have only one entry in the runtime computation
                run_time.append(
                    event_data["Train/learning_rate"]["w_time"][-1]
                    - event_data["Train/learning_rate"]["w_time"][0]
                )
            events_data.append(event_data)

        run_time = sum(run_time)
        data = {}
        for t in events_data[0].keys():
            combined_data = {}
            for k in events_data[0][t].keys():
                combined_data[k] = np.concatenate(
                    [events_data[i][t][k] for i in range(len(events_data))]
                )
            data[t] = combined_data

        # sort the data
        idx = np.argsort(data["Train/learning_rate"]["step"])
        for t in data.keys():
            for k in data[t].keys():
                data[t][k] = data[t][k][idx]
    else:
        data = {}
        # print("Found only one event file")
        ea = event_accumulator.EventAccumulator(
            glob.glob(path + "*events*")[0],
            size_guidance={event_accumulator.TENSORS: 0},
        )
        ea.Reload()
        tensors_list = ea.Tags()["tensors"]
        tensors_list.remove("config/text_summary")

        for t in tensors_list:
            data[t] = tf_log_to_np(ea.Tensors(t))

        if len(data["Train/learning_rate"]["w_time"]) != 1:
            run_time = (
                data["Train/learning_rate"]["w_time"][-1]
                - data["Train/learning_rate"]["w_time"][0]
            )

    return data, run_time


def tf_log_to_np(tensor_list):
    w_time, step, val = zip(*tensor_list)
    val_floats = []
    for i in range(len(val)):
        val_floats.append(val[i].float_val)

    np_data = {
        "w_time": np.array(w_time),
        "step": np.array(step),
        "value": np.array(val_floats).flatten(),
    }

    return np_data


def print_final_results(np_log_data, run_time):
    for key in np_log_data.keys():
        print(
            str(key)
            + " at step "
            + str(np_log_data[key]["step"][-1])
            + " is "
            + str(np_log_data[key]["value"][-1])
        )
    print("Total runtime minutes: " + str(run_time / 60))


def save_final_results(np_log_data, path):
    for key in np_log_data.keys():
        np.savetxt(
            path + key[key.rfind("/") + 1 :] + ".csv",
            np.concatenate(
                (
                    np.reshape(np_log_data[key]["w_time"], (1, -1)),
                    np.reshape(np_log_data[key]["step"], (1, -1)),
                    np.reshape(np_log_data[key]["value"], (1, -1)),
                ),
                axis=0,
            ).T,
            delimiter=",",
            header="w_time, step, value",
            comments="",
        )


def plot_results(np_log_data, save_path):
    keys_list = list(np_log_data.keys())
    train_keys = [key for key in keys_list if "Train" in key]
    validator_keys = [key for key in keys_list if "Validators" in key]
    train_keys.insert(0, train_keys.pop(train_keys.index("Train/learning_rate")))
    train_keys.insert(1, train_keys.pop(train_keys.index("Train/loss_aggregated")))
    ordered_keys_list = train_keys + validator_keys

    fig, axs = plt.subplots(
        len(ordered_keys_list), figsize=(4, 3 * len(ordered_keys_list))
    )

    for i in range(len(ordered_keys_list)):
        axs[i].plot(
            np_log_data[ordered_keys_list[i]]["step"],
            np_log_data[ordered_keys_list[i]]["value"],
        )
        axs[i].set_yscale("log")
        axs[i].set_title(ordered_keys_list[i])

    plt.tight_layout()
    plt.savefig(save_path + "/train_plots")


def check_validation_error(path, threshold, save_path):
    os.makedirs(save_path, exist_ok=True)
    np_log_data, run_time = read_tensorboard_eventfiles(path)
    for key in np_log_data.keys():
        if "Validators" in key:
            assert np_log_data[key]["value"][-1] < threshold, (
                "Validation error for {} is not below the specified threshold of {}".format(
                    key, threshold
                )
            )
    plot_results(np_log_data, save_path)
    print_final_results(np_log_data, run_time)
    save_final_results(np_log_data, save_path)
