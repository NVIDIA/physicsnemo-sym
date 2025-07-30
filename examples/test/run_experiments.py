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

import argparse
import GPUtil
import json
from pathlib import Path
from que import Que


def get_experiments(
    tag: str, run_opt: str = "full", json_file: Path = Path("./experiments.json")
):
    """Gets dictionary of experiments to run from JSON file

    Parameters
    ----------
    tag : str
        Tag of experiments to get
    run_opt : str, optional
        Run option, by default "full"
    json_file : Path, optional
        Filename/path to JSON file, by default Path("./experiments.json")
    """
    assert json_file.is_file(), f"Invalid experiment JSON path {json_file}"
    with open("experiments.json") as json_file:
        data = json.load(json_file)
    # Run option must be present in run options field
    assert run_opt in data["run_opts"], f"Invalid experiment run option {run_opt}"

    experiments = {}
    for key, value in data["experiments"].items():
        if tag in value["tags"]:
            experiments[f"{key}"] = {
                "path": value["path"],
                "run_cmd": value["run_cmd"]
                + "".join([f" {cmd}" for cmd in data["run_opts"][run_opt]]),
            }

    return experiments


if __name__ == "__main__":
    # get inputs
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", default="single_step", choices=["full", "single_step", "unit_tests"]
    )
    parser.add_argument("--gpus", default=None)
    args = parser.parse_args()

    # get gpus
    if args.gpus is None:
        available_gpus = GPUtil.getAvailable(limit=8)
    else:
        available_gpus = [int(x) for x in args.gpus.split(",")]
    if not available_gpus:
        raise ValueError("At least 1 GPU is required to run this script")

    # set experiments
    if args.mode == "full":
        tags = ["first", "second"]
        run_opt = "full"
    elif args.mode == "single_step":
        tags = ["first", "second"]
        run_opt = "single"
    elif args.mode == "unit_tests":
        tags = ["unit"]
        run_opt = "full"

    for tag in tags:
        print(f"Collecting experiments with tag: {tag}")
        experiments = get_experiments(tag, run_opt)

        q = Que(available_gpus=available_gpus, print_type="que", exit_on_fail=False)
        for key, value in experiments.items():
            q.enque_cmds(name=key, cmds=value["run_cmd"], cwds=value["path"])
        # Run experiment queue
        q.start_que_runner()
