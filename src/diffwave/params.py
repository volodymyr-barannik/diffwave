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

import numpy as np


class AttrDict(dict):
  def __init__(self, *args, **kwargs):
      super(AttrDict, self).__init__(*args, **kwargs)
      self.__dict__ = self

  def override(self, attrs):
    if isinstance(attrs, dict):
      self.__dict__.update(**attrs)
    elif isinstance(attrs, (list, tuple, set)):
      for attr in attrs:
        self.override(attr)
    elif attrs is not None:
      raise NotImplementedError
    return self


runconfig = AttrDict(
    data_dir='../../samples/LJSpeech/wavs',
    model_dir='../../models',
    max_steps=None,
    fp16=False,
    inference_model_path='../../models/weights-47415.pt',
    default_inference_spectrogram='../../samples/LJ-Speech/reference_0.wav.spec.npy',
)

gdrive_data_dir = '/content/diffwave/samples/LJSpeech/wavs'
gdrive_model_dir = '/content/drive/MyDrive/diffwave/models'
gdrive_output_dir = '/content/drive/MyDrive/diffwave/output'
runconfig_colab = AttrDict(
    data_dir=gdrive_data_dir,
    model_dir=gdrive_model_dir,
    max_steps=None,
    fp16=False,
    inference_model_path=gdrive_model_dir + '/weights-47415.pt',
    default_inference_spectrogram=gdrive_data_dir + '/LJ001-0001.wav.spec.npy',
    inference_output_path=gdrive_output_dir + '/output.wav',
)

params = AttrDict(
    # Training params
    batch_size=24,
    learning_rate=2e-4,
    max_grad_norm=None,

    # Data params
    sample_rate=22050,
    n_mels=80,
    n_fft=1024,
    hop_samples=256,
    crop_mel_frames=62,  # Probably an error in paper.

    # Model params
    residual_layers=30,
    residual_channels=64,
    dilation_cycle_length=10,
    unconditional = False,
    noise_schedule=np.linspace(1e-4, 0.05, 50).tolist(),
    inference_noise_schedule=[0.0001, 0.001, 0.01, 0.05, 0.2, 0.5],

    # unconditional sample len
    audio_len = 22050*5, # unconditional_synthesis_samples
)
