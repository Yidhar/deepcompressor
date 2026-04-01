import os
import random

import datasets
import yaml

__all__ = ["get_dataset"]


def load_dataset_yaml(meta_path: str, max_dataset_size: int = -1, repeat: int = 4) -> dict:
    meta = yaml.safe_load(open(meta_path, "r"))
    meta_dir = os.path.dirname(os.path.abspath(meta_path))
    names = list(meta.keys())
    if max_dataset_size > 0:
        random.Random(0).shuffle(names)
        names = names[:max_dataset_size]
        names = sorted(names)

    has_images = any(isinstance(meta[name], dict) for name in names)
    ret = {"filename": [], "prompt": [], "meta_path": []}
    if has_images:
        ret["image"] = []
        ret["image_path"] = []
    idx = 0
    for name in names:
        entry = meta[name]
        if isinstance(entry, str):
            prompt = entry
            image_path = None
        else:
            prompt = entry["prompt"]
            image_path = entry.get("image_path", entry.get("image"))
            if image_path is None:
                raise ValueError(f"YAML entry '{name}' is missing 'image' or 'image_path'.")
            if "://" not in image_path and not os.path.isabs(image_path):
                image_path = os.path.join(meta_dir, image_path)
        for j in range(repeat):
            ret["filename"].append(f"{name}-{j}")
            ret["prompt"].append(prompt)
            ret["meta_path"].append(meta_path)
            if has_images:
                ret["image"].append(image_path)
                ret["image_path"].append(image_path)
            idx += 1
    return ret


def get_dataset(
    name: str,
    config_name: str | None = None,
    split: str = "train",
    max_dataset_size: int = -1,
    return_gt: bool = False,
    repeat: int = 4,
    chunk_start: int = 0,
    chunk_step: int = 1,
) -> datasets.Dataset:
    prefix = os.path.dirname(__file__)
    kwargs = {
        "name": config_name,
        "split": split,
        "trust_remote_code": True,
        "token": True,
        "max_dataset_size": max_dataset_size,
    }
    if name.endswith((".yaml", ".yml")):
        yaml_data = load_dataset_yaml(name, max_dataset_size=max_dataset_size, repeat=repeat)
        features = {
            "filename": datasets.Value("string"),
            "prompt": datasets.Value("string"),
            "meta_path": datasets.Value("string"),
        }
        if "image" in yaml_data:
            features["image"] = datasets.Image()
            features["image_path"] = datasets.Value("string")
        dataset = datasets.Dataset.from_dict(
            yaml_data,
            features=datasets.Features(features),
        )
    else:
        path = os.path.join(prefix, f"{name}")
        if name == "COCO":
            dataset = datasets.load_dataset(path, return_gt=return_gt, **kwargs)
        elif name == "DCI":
            dataset = datasets.load_dataset(path, return_gt=return_gt, **kwargs)
        elif name == "MJHQ":
            dataset = datasets.load_dataset(path, return_gt=return_gt, **kwargs)
        else:
            raise ValueError(f"Unknown dataset name: {name}")
    assert not hasattr(dataset, "_unchunk_size")
    assert not hasattr(dataset, "_chunk_start")
    assert not hasattr(dataset, "_chunk_step")
    unchunk_size = len(dataset)
    if chunk_step > 1 or chunk_start > 0:
        assert 0 <= chunk_start < chunk_step
        dataset = dataset.select(range(chunk_start, len(dataset), chunk_step))
    else:
        chunk_start, chunk_step = 0, 1
    dataset._unchunk_size = unchunk_size
    dataset._chunk_start = chunk_start
    dataset._chunk_step = chunk_step
    return dataset
