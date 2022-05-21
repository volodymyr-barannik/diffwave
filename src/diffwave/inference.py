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
import os
import torch
import torchaudio

from argparse import ArgumentParser, Namespace

from diffwave.params import AttrDict, params as base_params, runconfig, runconfig_colab
from diffwave.model import DiffWave

models = {}


def predict(spectrogram=None, model_dir=None, params=None, device=torch.device('cuda'), fast_sampling=False):
    # Lazy load model.
    if not model_dir in models:
        if os.path.exists(f'{model_dir}/weights.pt'):
            checkpoint = torch.load(f'{model_dir}/weights.pt')
        else:
            checkpoint = torch.load(model_dir)
        model = DiffWave(AttrDict(base_params)).to(device)
        model.load_state_dict(checkpoint['model'])
        model.eval()
        models[model_dir] = model

    model = models[model_dir]
    model.params.override(params)
    with torch.no_grad():
        # Change in notation from the DiffWave paper for fast sampling.
        # DiffWave paper -> Implementation below
        # --------------------------------------
        # alpha -> talpha
        # beta -> training_noise_schedule
        # gamma -> alpha
        # eta -> beta
        training_noise_schedule = np.array(model.params.noise_schedule)
        inference_noise_schedule = np.array(model.params.inference_noise_schedule) if fast_sampling else training_noise_schedule

        print("training_noise_schedule =", training_noise_schedule)
        print("inference_noise_schedule =", inference_noise_schedule)

        talpha = 1 - training_noise_schedule
        talpha_cum = np.cumprod(talpha)  # [ talpha[0],  talpha[0] * talpha[1],  talpha[0] * talpha[1] * talpha[2], ... ]

        print("talpha =", talpha)
        print("talpha_cum =", talpha_cum)

        beta: np.array = inference_noise_schedule
        alpha: np.array = 1 - beta
        alpha_cum = np.cumprod(alpha)  # [ alpha[0],  alpha[0] * alpha[1],  alpha[0] * alpha[1] * alpha[2], ... ]

        print("beta =", beta)
        print("alpha =", alpha)
        print("alpha_cum =", alpha_cum)

        T = []
        for i_infer in range(len(inference_noise_schedule)):

            for i_train in range(len(training_noise_schedule) - 1):

                # if next_train_alpha <= curr_infer_alpha <= curr_train_alpha:
                if talpha_cum[i_train + 1] <= alpha_cum[i_infer] <= talpha_cum[i_train]:

                    twiddle = (np.sqrt(talpha_cum[i_train]) - np.sqrt(alpha_cum[i_infer])) / (
                                np.sqrt(talpha_cum[i_train]) - np.sqrt(talpha_cum[i_train + 1]))
                    T.append(i_train + twiddle)
                    print("talpha =", talpha)
                    print("twiddle =", twiddle)
                    break

        T = np.array(T, dtype=np.float32)
        print("T =", T)

        from diffwave import Plot
        fig5 = Plot.newfig(5, 10, 20)
        ax5 = fig5.add_subplot(1, 1, 1)

        if model.params.unconditional:
            audio = torch.randn(1, params.audio_len, device=device)
        else:
            # TODO: wtf {
            print(spectrogram)

            print("spectrogram =", spectrogram)

            #Plot.plotHeatmap2d(ax5, spectrogram_flat, "conditioner spectrogram")
            if len(spectrogram.shape) == 2:  # Expand rank 2 tensors by adding a batch dimension.
                spectrogram = spectrogram.unsqueeze(0)
            spectrogram = spectrogram.to(device)
            audio = torch.randn(spectrogram.shape[0], model.params.hop_samples * spectrogram.shape[-1], device=device)
            # }
        #print("audio =", audio)
        #Plot.plotCont1d(ax5, torch.flatten(audio.cpu()), "conditioner spectrogram")
        torchaudio.save("audio_variable.wav", audio.cpu(), sample_rate=model.params.sample_rate)

        noise_scale = torch.from_numpy(alpha_cum ** 0.5).float().unsqueeze(1).to(device)

        for n in range(len(alpha) - 1, -1, -1):
            c1 = 1 / np.sqrt(alpha[n])
            c2 = beta[n] / np.sqrt(1 - alpha_cum[n])
            audio = c1 * (audio - c2 * model(audio, torch.tensor([T[n]], device=audio.device), spectrogram).squeeze(1))
            if n > 0:
                noise = torch.randn_like(audio)
                sigma = ((1.0 - alpha_cum[n - 1]) / np.sqrt((1.0 - alpha_cum[n]) * beta[n]))
                audio += sigma * noise
            audio = torch.clamp(audio, -1.0, 1.0)
            torchaudio.save("audio_v"+str(n)+".wav", audio.cpu(), sample_rate=model.params.sample_rate)
    return audio, model.params.sample_rate


def main(args):
    if args.spectrogram_path:
        spectrogram = torch.from_numpy(np.load(args.spectrogram_path))
    else:
        spectrogram = None
    audio, sr = predict(spectrogram, model_dir=args.model_dir, fast_sampling=args.fast, params=base_params)
    torchaudio.save(args.output, audio.cpu(), sample_rate=sr)

def run():
    parser = ArgumentParser(description='runs inference on a spectrogram file generated by diffwave.preprocess')
    parser.add_argument('model_dir', nargs='?', default=runconfig.inference_model_path,
                        help='directory containing a trained model (or full path to weights.pt file)')
    parser.add_argument('--spectrogram_path', '-s', default=runconfig.default_inference_spectrogram,
                        help='path to a spectrogram file generated by diffwave.preprocess')
    parser.add_argument('--output', '-o', default='output.wav',
                        help='output file name')
    parser.add_argument('--fast', '-f', default='false', action='store_true',
                        help='fast sampling procedure')
    main(parser.parse_args())

def run_from_collab():
    parameters = Namespace(fast='false', model_dir=runconfig_colab.model_dir, output='output.wav', spectrogram_path=runconfig_colab.default_inference_spectrogram)
    main(parameters)

if __name__ == '__main__':
    run_from_collab()
