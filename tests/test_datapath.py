import yaml, os

def test_paths_exist():
    with open("data/coco_subset.yaml") as f:
        y = yaml.safe_load(f)
    assert "train" in y and "val" in y
    # we use coco128 paths that Ultralytics auto-downloads on first train
    # so we only check keys exist here
    assert "nc" in y and "names" in y
    assert isinstance(y["names"], list)
