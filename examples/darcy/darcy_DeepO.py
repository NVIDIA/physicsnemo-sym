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

import physicsnemo.sym
from physicsnemo.sym.hydra import instantiate_arch, PhysicsNeMoConfig
from physicsnemo.sym.key import Key

from physicsnemo.sym.solver import Solver
from physicsnemo.sym.domain import Domain
from physicsnemo.sym.domain.constraint.continuous import DeepONetConstraint
from physicsnemo.sym.domain.validator import PointwiseValidator


from utilities import download_FNO_dataset, load_deeponet_dataset


@physicsnemo.sym.main(config_path="conf", config_name="config_DeepO")
def run(cfg: PhysicsNeMoConfig) -> None:
    # [datasets]
    # load training/ test data
    branch_input_keys = [Key("coeff")]
    [Key("x"), Key("y")]
    output_keys = [Key("sol")]

    download_FNO_dataset("Darcy_241", outdir="datasets/")
    invar_train, outvar_train = load_deeponet_dataset(
        "datasets/Darcy_241/piececonst_r241_N1024_smooth1.hdf5",
        [k.name for k in branch_input_keys],
        [k.name for k in output_keys],
        n_examples=1000,
    )
    invar_test, outvar_test = load_deeponet_dataset(
        "datasets/Darcy_241/piececonst_r241_N1024_smooth2.hdf5",
        [k.name for k in branch_input_keys],
        [k.name for k in output_keys],
        n_examples=10,
    )
    # [datasets]

    # [init-model]
    # make list of nodes to unroll graph on
    branch_net = instantiate_arch(
        cfg=cfg.arch.branch,
    )
    trunk_net = instantiate_arch(
        cfg=cfg.arch.trunk,
    )
    deeponet = instantiate_arch(
        cfg=cfg.arch.deeponet,
        branch_net=branch_net,
        trunk_net=trunk_net,
    )
    nodes = [deeponet.make_node(name="deepo")]
    # [init-model]

    # [constraint]
    # make domain
    domain = Domain()

    # add constraint to domain
    data = DeepONetConstraint.from_numpy(
        nodes=nodes,
        invar=invar_train,
        outvar=outvar_train,
        batch_size=cfg.batch_size.train,
    )
    domain.add_constraint(data, "data")
    # [constraint]

    # [validator]
    # add validators
    val = PointwiseValidator(
        nodes=nodes,
        invar=invar_test,
        true_outvar=outvar_test,
        plotter=None,
    )
    domain.add_validator(val, "val")
    # [validator]

    # make solver
    slv = Solver(cfg, domain)

    # start solver
    slv.solve()


if __name__ == "__main__":
    run()
