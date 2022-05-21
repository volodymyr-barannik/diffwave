# Copyright 2020 LMNT, Inc. All Rights Reserved.
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
# ==============================================================================

from argparse import ArgumentParser, Namespace

import matplotlib
import torch
from matplotlib import pyplot as plt
from torch.cuda import device_count
from torch.multiprocessing import spawn

from diffwave.learner import train, train_distributed
from diffwave.model import DiffusionEmbedding
from diffwave.params import params, runconfig, runconfig_collab


def _get_free_port():
    import socketserver
    with socketserver.TCPServer(('localhost', 0), None) as s:
        return s.server_address[1]


def main(args):
    replica_count = device_count()
    if replica_count > 1:
        if params.batch_size % replica_count != 0:
            raise ValueError(f'Batch size {params.batch_size} is not evenly divisble by # GPUs {replica_count}.')
        params.batch_size = params.batch_size // replica_count
        port = _get_free_port()
        spawn(train_distributed, args=(replica_count, port, args, params), nprocs=replica_count, join=True)
    else:
        train(args, params)


def run_from_colab():
    parameters = Namespace(model_dir=runconfig_collab.model_dir, data_dirs=[runconfig_collab.data_dir], max_steps=runconfig_collab.max_steps, fp16=False)
    main(parameters)

def run():

    parser = ArgumentParser(description='train (or resume training) a DiffWave model')
    parser.add_argument('model_dir', nargs='?', default=runconfig.model_dir,
                        help='directory in which to store model checkpoints and training logs')
    parser.add_argument('data_dirs', nargs='?', default=[runconfig.data_dir],
                        help='space separated list of directories from which to read .wav files for training')
    parser.add_argument('max_steps', nargs='?', default=runconfig.max_steps, type=int,
                        help='maximum number of training steps')
    parser.add_argument('fp16', action='store_true', default=runconfig.fp16,
                        help='use 16-bit floating point operations for training')
    main(parser.parse_args())


if __name__ == '__main__':

    run_from_colab()

    if False:
        # mplstyle.use('fast')
        # matplotlib.use('QtAgg')

        print(params)
        emb = DiffusionEmbedding(len(params.noise_schedule))
        print("emb =", emb)
        print("emb.embedding =", emb.embedding, "shape =", emb.embedding.shape)
        sx0 = torch.tensor(0)  # this is int64
        sy0 = emb.forward(sx0)
        print("sx0 =", sx0, " | sy0 =", sy0)

        plt.show()
