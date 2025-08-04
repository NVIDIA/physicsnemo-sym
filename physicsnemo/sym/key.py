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

"""Key"""

from typing import Union, List
from functools import reduce
from .constants import diff_str, NO_OP_SCALE


class Key(object):
    """
    Class describing keys used for graph unroll.
    The most basic key is just a simple string
    however you can also add dimension information
    and even information on how to scale inputs
    to networks.

    Parameters
    ----------
    name : str
      String used to refer to the variable (e.g. 'x', 'y'...).
    size : int=1
      Dimension of variable.
    derivatives : List=[]
      This signifies that this key holds a derivative with
      respect to that key.
    scale: (float, float)
      Characteristic location and scale of quantity: used for normalisation.
    """

    def __init__(self, name, size=1, derivatives=[], base_unit=None, scale=NO_OP_SCALE):
        super(Key, self).__init__()
        self.name = name
        self.size = size
        self.derivatives = derivatives
        self.base_unit = base_unit
        self.scale = scale

    @classmethod
    def from_str(cls, name):
        split_name = name.split(diff_str)
        var_name = split_name[0]
        diff_names = Key.convert_list(split_name[1:])
        return cls(var_name, size=1, derivatives=diff_names)

    @classmethod
    def from_tuple(cls, name_size):
        split_name = name_size[0].split(diff_str)
        var_name = split_name[0]
        diff_names = Key.convert_list(split_name[1:])
        return cls(var_name, size=name_size[1], derivatives=diff_names)

    @classmethod
    def convert(cls, name_or_tuple):
        if isinstance(name_or_tuple, str):
            key = Key.from_str(name_or_tuple)
        elif isinstance(name_or_tuple, tuple):
            key = cls.from_tuple(name_or_tuple)
        elif isinstance(name_or_tuple, cls):
            key = name_or_tuple
        else:
            raise ValueError("can only convert string or tuple to key")
        return key

    @staticmethod
    def convert_list(ls):
        keys = []
        for name_or_tuple in ls:
            keys.append(Key.convert(name_or_tuple))
        return keys

    @staticmethod
    def convert_config(key_cfg: Union[List, str]):
        """Converts a config input/output key string/list into a key
        This provides a quick alternative method for defining keys in models

        Parameters
        ----------
        key_cfg : Union[List, str]
            Config list or string

        Returns
        -------
        List[Key]
            List of keys generated

        Example
        -------
        The following are some config examples for constructing keys in the YAML file.

        Defining input/output keys with size of 1

        >>> arch:
        >>>    full_connected:
        >>>        input_keys: input
        >>>        output_keys: output

        Defining input/output keys with different sizes

        >>> arch:
        >>>    full_connected:
        >>>        input_keys: [input, 2] # Key('input',size=2)
        >>>        output_keys: [output, 3] # Key('output',size=3)

        Multiple input/output keys with size of 1
        >>> arch:
        >>>    full_connected:
        >>>        input_keys: [a, b, c]
        >>>        output_keys: [u, w, v]

        Multiple input/output keys with different sizes
        >>> arch:
        >>>    full_connected:
        >>>        input_keys: [[a,2], [b,3]] # Key('a',size=2), Key('b',size=3)
        >>>        output_keys: [[u,3],w] # Key('u',size=3), Key('w',size=1)

        """
        # Just single key name
        if isinstance(key_cfg, str):
            keys = [Key.convert(key_cfg.lstrip())]
        # Multiple keys
        elif isinstance(key_cfg, list):
            keys = []
            for cfg_obj in key_cfg:
                if isinstance(cfg_obj, str):
                    key = Key.convert(cfg_obj)
                    keys.append(key)
                elif isinstance(cfg_obj, int) and len(keys) > 0:
                    keys[-1].size = cfg_obj
                elif isinstance(cfg_obj, list):
                    key_name = cfg_obj[0]
                    key = Key.convert(key_name)
                    try:
                        key_size = int(cfg_obj[1])
                        key.size = key_size
                    except:
                        key.size = 1
                    keys.append(key)
                # Manually provided
                elif isinstance(cfg_obj, Key):
                    keys.append(cfg_obj)
                else:
                    raise ValueError(f"Invalid key parameter set in config {key_cfg}")
        else:
            raise ValueError(f"Invalid key parameter set in config {key_cfg}")
        return keys

    @property
    def unit(self):
        return self.base_unit / reduce(
            lambda x, y: x.base_unit * y.base_unit, self.derivatives
        )

    def __str__(self):
        diff_str = "".join(["__" + x.name for x in self.derivatives])
        return self.name + diff_str

    def __repr__(self):
        return str(self)

    def __eq__(self, obj):
        return isinstance(obj, Key) and str(self) == str(obj)

    def __lt__(self, obj):
        assert isinstance(obj, Key)
        return str(self) < str(obj)

    def __gt__(self, obj):
        assert isinstance(obj, Key)
        return str(self) > str(obj)

    def __hash__(self):
        return hash(str(self))


def _length_key_list(list_keys):
    length = 0
    for key in list_keys:
        length += key.size
    return length
