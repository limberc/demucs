import pdb

import hydra
from hydra.utils import to_absolute_path


@hydra.main(config_path="./conf", config_name="config.yaml")
def wrap_hydra_args(args):
    global __file__
    __file__ = to_absolute_path(__file__)
    for attr in ["musdb", "wav", "metadata"]:
        val = getattr(args.dset, attr)
        if val is not None:
            setattr(args.dset, attr, to_absolute_path(val))
    args = hydra.utils.instantiate(args)
    pdb.set_trace()
    return args


if __name__ == '__main__':
    args = wrap_hydra_args()
    print(args)
