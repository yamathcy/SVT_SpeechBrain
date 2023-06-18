"""This lobe enables the integration of huggingface pretrained wav2vec2/hubert/data2vec models.

Reference: https://arxiv.org/abs/2006.11477
Reference: https://arxiv.org/abs/1904.05862
Reference: https://arxiv.org/abs/2106.07447
Reference: https://arxiv.org/abs/2202.03555
Transformer from HuggingFace needs to be installed:
https://huggingface.co/transformers/installation.html

Authors
 * Xiangming Gu 2022
"""

import os
import torch
import logging
import pathlib
import numpy as np
import torch.nn.functional as F
from torch import nn
from huggingface_hub import model_info
from speechbrain.pretrained.fetching import fetch
import transformers
from transformers import Wav2Vec2Model, HubertModel, Data2VecAudioModel, WavLMConfig, AutoModel
from transformers import Wav2Vec2Config, HubertConfig, Data2VecAudioConfig, WavLMModel, AutoConfig
from transformers import Wav2Vec2FeatureExtractor, AutoFeatureExtractor
# from transformers import Wav2Vec2ForPreTraining, AutoForPreTraining
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    _compute_mask_indices,
)

# # We check if transformers is installed.
# try:
#     import transformers
#     from transformers import Wav2Vec2Model, HubertModel, Data2VecAudioModel, WavLMConfig, AutoModel
#     from transformers import Wav2Vec2Config, HubertConfig, Data2VecAudioConfig, WavLMModel, AutoConfig
#     from transformers import Wav2Vec2FeatureExtractor, AutoFeatureExtractor
#     from transformers import Wav2Vec2ForPreTraining, AutoForPreTraining
#     from transformers.models.wav2vec2.modeling_wav2vec2 import (
#         _compute_mask_indices,
#     )

# except ImportError:
#     MSG = "Please install transformers from HuggingFace to use wav2vec2 / Hubert / Data2vec\n"
#     MSG += "E.G. run: pip install transformers"
#     raise ImportError(MSG)

logger = logging.getLogger(__name__)

HF_models = {"wav2vec2": Wav2Vec2Model, "hubert": HubertModel, "data2vec": Data2VecAudioModel, "wavlm": WavLMModel}

HF_config = {"wav2vec2": Wav2Vec2Config, "hubert": HubertConfig, "data2vec": Data2VecAudioConfig, "wavlm": WavLMConfig}


class HuggingFaceWav2Vec2(nn.Module):
    """This lobe enables the integration of HuggingFace and SpeechBrain
    pretrained wav2vec2.0/Hubert/data2vec models.

    Source paper wav2vec2.0: https://arxiv.org/abs/2006.11477
    Source paper Hubert: https://arxiv.org/abs/2106.07447
    Source paper data2vec: https://arxiv.org/abs/2202.03555
    Source paper wavlm: https://arxiv.org/abs/2110.13900
    Transformer from HuggingFace needs to be installed:
    https://huggingface.co/transformers/installation.html

    The model can be used as a fixed feature extractor or can be finetuned. It
    will download automatically the model from HuggingFace or use a local path.

    Arguments
    ---------
    source : str
        HuggingFace hub name: e.g "facebook/wav2vec2-large-lv60"
    save_path : str
        Path (dir) of the downloaded model.
    output_norm : bool (default: True)
        If True, a layer_norm (affine) will be applied to the output obtained
        from the wav2vec model.
    freeze : bool (default: True)
        If True, the model is frozen. If False, the model will be trained
        alongside with the rest of the pipeline.
    freeze_feature_extractor :  bool (default: False)
        When freeze = False and freeze_feature_extractor True, the featue_extractor module of the model is Frozen. If False
        all the wav2vec model will be trained including featue_extractor module.
    apply_spec_augment : bool (default: False)
        If True, the model will apply spec augment on the output of feature extractor
        (inside huggingface Wav2VecModel() class).
        If False, the model will not apply spec augment. We set this to false to prevent from doing it twice.
    Example
    -------
    >>> inputs = torch.rand([10, 600])
    >>> model_hub = "facebook/wav2vec2-base-960h"
    >>> save_path = "savedir"
    >>> model = HuggingFaceWav2Vec2(model_hub, save_path)
    >>> outputs = model(inputs)
    """

    def __init__(
        self,
        source,
        save_path,
        pretrain=True,
        output_norm=True,
        freeze=True,
        freeze_feature_extractor=False,
        apply_spec_augment=False,
        feat_dim=768,
        output_neurons=20,
    ):
        super().__init__()

        # Download the extractor from HuggingFace.
        # The extractor is only used to retrieve the normalisation information
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            source, cache_dir=save_path

        )

        # Select specific self-supervised loader (eg. Wav2Vec2, Hubert)
        if "hubert" in source:
            config = HF_config.get("hubert")
            model = HF_models.get("hubert")
        elif "data2vec" in source:
            config = HF_config.get("data2vec")
            model = HF_models.get("data2vec")
        elif "wavlm" in source:
            config = HF_config.get("wavlm")
            model = HF_models.get("wavlm")
        elif "wav2vec2" in source:
            config = HF_config.get("wav2vec2")
            model = HF_models.get("wav2vec2")

        # # Download and load the model
        self._from_pretrained(
            source, config=AutoConfig, model=AutoModel, save_path=save_path
        )

        self.probe = Probe(output_neurons, input_size=feat_dim)

        # set apply_spec_augment
        self.model.config.apply_spec_augment = apply_spec_augment

        # We check if inputs need to be normalized w.r.t pretrained wav2vec2
        self.normalize_wav = self.feature_extractor.do_normalize

        self.freeze = freeze
        self.freeze_feature_extractor = freeze_feature_extractor
        self.output_norm = output_norm
        if self.freeze:
            logger.warning(
                "speechbrain.lobes.models.huggingface_wav2vec - wav2vec 2.0 is frozen."
            )
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False
        else:
            self.model.train()
            if self.freeze_feature_extractor:
                self.model.feature_extractor._freeze_parameters()
        
        # Randomly initialized layers if pretrain is False
        if not (pretrain):
            print("Reset layers")
            self.reset_layer(self.model)
        
        

    def reset_layer(self, model):
        """Reinitializes the parameters of the network"""
        if hasattr(model, "reset_parameters"):
            print("reset parameters!!!")
            model.reset_parameters()
        for child_layer in model.children():
            if model != child_layer:
                self.reset_layer(child_layer)

    def _from_pretrained(self, source, config, model, save_path):
        """This function manages the source checking and loading of the params.
        # 1. Is the model from HF or a local path
        # 2. Is the model pretrained with HF or SpeechBrain
        # 3. Download (if appropriate) and load with respect to 1. and 2.
        """
        is_sb, ckpt_file = self._check_model_source(source)
        if is_sb:
            config = config.from_pretrained(source, cache_dir=save_path, trust_remote_code=True)
            self.model = model(config)
            self.model.gradient_checkpointing_disable()  # Required by DDP
            # fetch the checkpoint file
            ckpt_full_path = fetch(
                filename=ckpt_file, source=source, savedir=save_path
            )
            # We transfer the parameters from the checkpoint.
            self._load_sb_pretrained_w2v2_parameters(ckpt_full_path)
        else:
            self.model = model.from_pretrained(source, cache_dir=save_path, trust_remote_code=True)

    def _load_sb_pretrained_w2v2_parameters(self, path):
        """Loads the parameter of a w2v2 model pretrained with SpeechBrain and the
        HuggingFaceWav2Vec2Pretrain Object. It is necessary to perform a custom
        loading because HuggingFace adds a level to the checkpoint when storing
        the model breaking the compatibility between HuggingFaceWav2Vec2Pretrain
        and HuggingFaceWav2Vec2.

        In practice a typical HuggingFaceWav2Vec2 checkpoint for a given parameter
        would be: model.conv.weight.data while for HuggingFaceWav2Vec2Pretrain it
        is: model.wav2vec2.weight.data (wav2vec2 must be removed before loading).
        """

        modified_state_dict = {}
        orig_state_dict = torch.load(path, map_location="cpu")

        # We remove the .wav2vec2 in the state dict.
        for key, params in orig_state_dict.items():
            if "wav2vec2." in key:
                save_key = key.replace("model.wav2vec2.", "")
                modified_state_dict[save_key] = params

        incompatible_keys = self.model.load_state_dict(
            modified_state_dict, strict=False
        )
        for missing_key in incompatible_keys.missing_keys:
            logger.warning(
                f"During parameter transfer to {self.model} loading from "
                + f"{path}, the transferred parameters did not have "
                + f"parameters for the key: {missing_key}"
            )
        for unexpected_key in incompatible_keys.unexpected_keys:
            logger.warning(
                f"The param with the key: {unexpected_key} is discarded as it "
                + "is useless for wav2vec 2.0 finetuning."
            )

    def _check_model_source(self, path):
        """Checks if the pretrained model has been trained with SpeechBrain and
        is hosted locally or on a HuggingFace hub.
        """
        checkpoint_filename = ""
        source = pathlib.Path(path)
        is_local = True
        is_sb = True

        # If path is a huggingface hub.
        if not source.exists():
            is_local = False

        if is_local:
            # Test for HuggingFace model
            if any(File.endswith(".bin") for File in os.listdir(path)):
                is_sb = False
                return is_sb, checkpoint_filename

            # Test for SpeechBrain model and get the filename.
            for File in os.listdir(path):
                if File.endswith(".ckpt"):
                    checkpoint_filename = os.path.join(path, File)
                    is_sb = True
                    return is_sb, checkpoint_filename
        else:
            files = model_info(
                path
            ).siblings  # get the list of files of the Hub

            # Test if it's an HuggingFace model or a SB one
            for File in files:
                if File.rfilename.endswith(".ckpt"):
                    checkpoint_filename = File.rfilename
                    is_sb = True
                    return is_sb, checkpoint_filename

            for File in files:
                if File.rfilename.endswith(".bin"):
                    checkpoint_filename = File.rfilename
                    is_sb = False
                    return is_sb, checkpoint_filename

        err_msg = f"{path} does not contain a .bin or .ckpt checkpoint !"
        raise FileNotFoundError(err_msg)

    def forward(self, wav):
        """Takes an input waveform and return its corresponding wav2vec encoding.

        Arguments
        ---------
        wav : torch.Tensor (signal)
            A batch of audio signals to transform to features.
        """

        # If we freeze, we simply remove all grads and features from the graph.
        if self.freeze:
            with torch.no_grad():
                return self.extract_features(wav).detach()

        return self.extract_features(wav)

    def extract_features(self, wav):
        """Takes an input waveform and return its corresponding wav2vec encoding.

        Arguments
        ---------
        wav : torch.Tensor (signal)
            A batch of audio signals to transform to features.
        """

        if self.normalize_wav:
            wav = F.layer_norm(wav, wav.shape)

        # Extract wav2vec output
        # out = self.model(wav)[0]
        out = self.model(wav,output_hidden_states=True, return_dict=None, output_attentions=None)
        out = out["hidden_states"]
        out = torch.stack(out, dim=3)
        # print(h.shape)

        # We normalize the output if required
        if self.output_norm:
            out = F.layer_norm(out,out.shape)
        return out
    

class Probe(nn.Module):
    """Computes a linear transformation y = wx + b.

    Arguments
    ---------
    n_neurons : int
        It is the number of output neurons (i.e, the dimensionality of the
        output).
    input_shape: tuple
        It is the shape of the input tensor.
    input_size: int
        Size of the input tensor.
    bias : bool
        If True, the additive bias b is adopted.
    combine_dims : bool
        If True and the input is 4D, combine 3rd and 4th dimensions of input.

    Example
    -------
    >>> inputs = torch.rand(10, 50, 40)
    >>> lin_t = Linear(input_shape=(10, 50, 40), n_neurons=100)
    >>> output = lin_t(inputs)
    >>> output.shape
    torch.Size([10, 50, 100])
    """

    def __init__(
        self,
        n_neurons,
        input_shape=None,
        input_size=None,
        bias=True,
        combine_dims=False,
        weight_sum=False
    ):
        
        super().__init__()
        self.combine_dims = combine_dims
        self.weight_sum = weight_sum

        if input_shape is None and input_size is None:
            raise ValueError("Expected one of input_shape or input_size")

        if input_size is None:
            input_size = input_shape[-1]
            if len(input_shape) == 4 and self.combine_dims:
                input_size = input_shape[2] * input_shape[3]
        if input_size == 768:
            self.lw = torch.nn.parameter.Parameter(data=torch.ones(13), requires_grad=True)
        else:
            self.lw = torch.nn.parameter.Parameter(data=torch.ones(25), requires_grad=True)
        # Weights are initialized following pytorch approach
        self.projection = nn.Linear(input_size, n_neurons, bias=bias)
        
    def forward(self, x):
        """Returns the linear transformation of input tensor.

        Arguments
        ---------
        x : torch.Tensor
            Input to transform linearly.
        """
        # print(x.shape)
        # if x.ndim == 4 and self.combine_dims:
        #     x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3])
        weights = torch.softmax(self.lw,dim=0)
        if self.weight_sum:
            x = torch.matmul(x, weights)
        else:
            x = x[:,:,:,-1]
        wx = self.projection(x)
        return wx, w
    
    def get_layer_weight(self):
        if self.weight_sum:
            weights = torch.softmax(self.lw,dim=0)
            weights = weights.detach().cpu().numpy()
        else:
            weights = None
        return weights
