import torch.utils.data
import torchvision
from .coco import build as build_coco
from .crowdpose import build as build_crowdpose
from .humanart import build as build_humanart
def get_coco_api_from_dataset(dataset):
    for _ in range(10):
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    if isinstance(dataset, torchvision.datasets.CocoDetection):
        return dataset.coco
    # Support custom datasets that expose the COCO api via a `coco` attribute
    if hasattr(dataset, "coco"):
        return dataset.coco

    return None


def build_dataset(image_set, args):
    if args.dataset_file == 'coco':
        return build_coco(image_set, args)
    if args.dataset_file == 'crowdpose':
        return build_crowdpose(image_set, args)
    if args.dataset_file == 'humanart':
        return build_humanart(image_set, args)
    raise ValueError(f'dataset {args.dataset_file} not supported')
