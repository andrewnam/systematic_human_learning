from pathlib import Path


class Paths:
    ROOT = Path(__file__).parents[2]        # root of the whole repository
    DATA = ROOT.joinpath('data')            # data directory for this experiment
    SAVE = Path("/data2/pdp/ajhnam/hidden_singles")
    CONFIG = ROOT / 'config.yaml'