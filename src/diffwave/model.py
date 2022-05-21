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
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from math import sqrt

from matplotlib import pyplot as plt

from diffwave import Plot
from diffwave.params import AttrDict
from diffwave.utility import *

Linear = nn.Linear
ConvTranspose2d = nn.ConvTranspose2d


def Conv1d(*args, **kwargs):
    layer = nn.Conv1d(*args, **kwargs)
    nn.init.kaiming_normal_(layer.weight)
    return layer


# Just an activation function. It's like relu but continuous
@torch.jit.script
def silu(x):
    return x * torch.sigmoid(x)


# TODO: what's that? what does it mean
class DiffusionEmbedding(nn.Module):
    def __init__(self, max_steps):
        super().__init__()

        self.DEBUGPLOT = False

        # (max_steps)x(128) table
        self.register_buffer('embedding', self._build_embedding(max_steps), persistent=False)
        print("embedding.shape=", self.get_buffer('embedding').shape)

        # just a random 512x128 matrix (exactly that order!).
        # gets Transposed when applying
        self.projection1 = Linear(in_features=128, out_features=512)

        # just a random 512x512 matrix
        self.projection2 = Linear(in_features=512, out_features=512)

    def forward(self, diffusion_step):

        debugcache_xs = []
        debugcache_xs_descriptions = []

        def debugcachex(description):
            if not DEBUGPLOT:
                return
            nonlocal debugcache_xs
            nonlocal debugcache_xs_descriptions
            nonlocal x
            debugcache_xs.append(x.clone())
            debugcache_xs_descriptions.append(description)

        def debugcachex_plot():
            if not DEBUGPLOT:
                return
            nonlocal debugcache_xs
            nonlocal debugcache_xs_descriptions

            fig4 = Plot.newfig(4, 8, 32)
            fig4.tight_layout()

            debug_limk = 1.2
            debug_ylim = [debug_limk * min(flatten(debugcache_xs)).item(), debug_limk * max(flatten(debugcache_xs)).item()]

            for i, x in enumerate(debugcache_xs):
                axes4i = fig4.add_subplot(len(debugcache_xs), 1, i+1)
                print("i=",i,"x=",x)
                Plot.plotCont1d(axes4i, x.detach().numpy(),
                                title=debugcache_xs_descriptions[i], ylim=debug_ylim)

        if diffusion_step.dtype in [torch.int32, torch.int64]:
            x = self.embedding[diffusion_step]
        else:
            x = self._lerp_embedding(diffusion_step)

        debugcachex("x = embedding")

        x = self.projection1(x)
        debugcachex("x = self.projection1(x)")

        x = silu(x)
        debugcachex("x = silu(x)")

        x = self.projection2(x)
        debugcachex("x = self.projection2(x)")

        x = silu(x)
        debugcachex("x = silu(x)")

        debugcachex_plot()

        return x

    def _lerp_embedding(self, t):
        low_idx = torch.floor(t).long()
        high_idx = torch.ceil(t).long()
        low = self.embedding[low_idx]
        high = self.embedding[high_idx]
        return low + (high - low) * (t - low_idx)

    ''' returns (max_steps)x(64) table. 
        First, each i'th row is exp. interpolation between i and i*1000.
        Then, we concat sin of that table and cos of that table's values.
        That is our result. '''
    def _build_embedding(self, max_steps):
        steps = torch.arange(max_steps).unsqueeze(1)  # [T,1]
        dims = torch.arange(64).unsqueeze(0)  # [1,64]
        modified_dims = (dims * 4.0 / 63.0)
        table = steps * 10.0 ** modified_dims  # [T,64]

        #fig1 = Plot.newfig(1)
        #axes1 = fig1.add_subplot(1, 1, 1)
        #fig1.tight_layout()
        #Plot.plotHeatmap2d(axes1, table, title="_build_embedding table (before sin-cos):")

        #fig3 = Plot.newfig(3, 6, 8)
        #axes3 = fig3.add_subplot(1, 1, 1)
        #fig3.tight_layout()
        #Plot.plotCont1d(axes3, table[5], title="1st row of pre sin cos")

        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)

        #fig2 = Plot.newfig(2)
        #axes2 = fig2.add_subplot(1, 1, 1)
        #fig2.tight_layout()
        #Plot.plotHeatmap2d(axes2, table, title="_build_embedding table (after sin-cos):")

        return table


class SpectrogramUpsampler(nn.Module):
    def __init__(self, n_mels):
        super().__init__()
        self.conv1 = ConvTranspose2d(1, 1, [3, 32], stride=[1, 16], padding=[1, 8])
        self.conv2 = ConvTranspose2d(1, 1, [3, 32], stride=[1, 16], padding=[1, 8])

    def forward(self, x):
        x = torch.unsqueeze(x, 1)
        x = self.conv1(x)
        x = F.leaky_relu(x, 0.4)
        x = self.conv2(x)
        x = F.leaky_relu(x, 0.4)
        x = torch.squeeze(x, 1)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, n_mels, residual_channels, dilation, uncond=False):
        '''
    :param n_mels: inplanes of conv1x1 for spectrogram conditional
    :param residual_channels: audio conv
    :param dilation: audio conv dilation
    :param uncond: disable spectrogram conditional
    '''
        super().__init__()
        self.dilated_conv = Conv1d(residual_channels, 2 * residual_channels, 3, padding=dilation, dilation=dilation)
        self.diffusion_projection = Linear(512, residual_channels)
        if not uncond:  # conditional model
            self.conditioner_projection = Conv1d(n_mels, 2 * residual_channels, 1)
        else:  # unconditional model
            self.conditioner_projection = None

        self.output_projection = Conv1d(residual_channels, 2 * residual_channels, 1)

    figcounter = 0

    def forward(self, x, diffusion_step, conditioner=None):
        assert (conditioner is None and self.conditioner_projection is None) or \
               (conditioner is not None and self.conditioner_projection is not None)

        #print("%s: i am %s: diffusion_step on screen" % (str(datetime.now()), str(self)))

        #fig = Plot.newfig(ResidualBlock.figcounter, 10, 12)
        #ax1 = fig.add_subplot(4, 1, 1)
        #ax2 = fig.add_subplot(4, 1, 2)
        #ax3 = fig.add_subplot(4, 1, 3)
        #ax4 = fig.add_subplot(4, 1, 4)
        #fig.tight_layout()

        #Plot.plotHeatmap2d(ax1, data=x.cpu().detach().numpy(), title='x', filetype='jpeg')
        #Plot.plotHeatmap2d(ax2, data=diffusion_step.cpu().detach().numpy(), title='diffusion_step before diffusion_projection=%d' % self.figcounter, filetype='jpeg')

        diffusion_step = self.diffusion_projection(diffusion_step).unsqueeze(-1)

        #Plot.plotHeatmap2d(ax3, data=diffusion_step.cpu().detach().numpy(), title='diffusion_step after diffusion_projection', filetype='jpeg')
        #ResidualBlock.figcounter += 1
        #plt.show()

        y = x + diffusion_step
        #Plot.plotHeatmap2d(ax4, data=diffusion_step.cpu().detach().numpy(), title='y = x + diffusion_step', filetype='jpeg')

        if self.conditioner_projection is None:  # using a unconditional model
            y = self.dilated_conv(y)
        else:
            conditioner = self.conditioner_projection(conditioner)
            y = self.dilated_conv(y) + conditioner

        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)

        y = self.output_projection(y)
        residual, skip = torch.chunk(y, 2, dim=1)
        return (x + residual) / sqrt(2.0), skip


class DiffWave(nn.Module):
    def __init__(self, params: AttrDict):
        super().__init__()
        self.params = params
        self.input_projection = Conv1d(1, params.residual_channels, 1)
        self.diffusion_embedding = DiffusionEmbedding(len(params.noise_schedule))
        if self.params.unconditional:  # use unconditional model
            self.spectrogram_upsampler = None
        else:
            self.spectrogram_upsampler = SpectrogramUpsampler(params.n_mels)

        self.residual_layers = nn.ModuleList([
            ResidualBlock(params.n_mels, params.residual_channels, 2 ** (i % params.dilation_cycle_length),
                          uncond=params.unconditional)
            for i in range(params.residual_layers)
        ])
        self.skip_projection = Conv1d(params.residual_channels, params.residual_channels, 1)
        self.output_projection = Conv1d(params.residual_channels, 1, 1)
        nn.init.zeros_(self.output_projection.weight)

    def forward(self, audio, diffusion_step, spectrogram=None):
        assert (spectrogram is None and self.spectrogram_upsampler is None) or \
               (spectrogram is not None and self.spectrogram_upsampler is not None)
        x = audio.unsqueeze(1)
        x = self.input_projection(x)
        x = F.relu(x)

        diffusion_step = self.diffusion_embedding(diffusion_step)
        if self.spectrogram_upsampler:  # use conditional model
            spectrogram = self.spectrogram_upsampler(spectrogram)

        skip = None
        for layer in self.residual_layers:
            x, skip_connection = layer(x, diffusion_step, spectrogram)
            skip = skip_connection if skip is None else skip_connection + skip

        x = skip / sqrt(len(self.residual_layers))
        x = self.skip_projection(x)
        x = F.relu(x)
        x = self.output_projection(x)
        return x
