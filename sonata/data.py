# Copyright (c) Meta Platforms, Inc. and affiliates.
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


import os
import numpy as np
from huggingface_hub import hf_hub_download


DATAS = ["sample1"]


def load(
    name: str = "sonata",
    download_root: str = None,
):
    if name in DATAS:
        print(f"Loading data from HuggingFace: {name} ...")
        data_path = hf_hub_download(
            repo_id="pointcept/demo",
            filename=f"{name}.npz",
            repo_type="dataset",
            revision="main",
            local_dir=download_root or os.path.expanduser("~/.cache/sonata/data"),
        )
    elif os.path.isfile(name):
        print(f"Loading data in local path: {name} ...")
        data_path = name
    else:
        raise RuntimeError(f"Data {name} not found; available models = {DATAS}")
    return dict(np.load(data_path))
