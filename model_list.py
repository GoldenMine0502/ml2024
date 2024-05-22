from dataset.dataset_vctk import DataLoaderVCTK
from dcunet_model import DcunetModel
from utils.audio import Audio

from dcunet import Dcunet

def get_models(hp):
    audio = Audio(hp)

    models = {
        "DCUNET": lambda: DcunetModel(hp, Dcunet(audio, is_attention=False)),
    }

    return models

def get_datasets(hp, model_name):

    datasets = {
        "VCTK16K": lambda: DataLoaderVCTK(hp, model_name),
    }

    return datasets