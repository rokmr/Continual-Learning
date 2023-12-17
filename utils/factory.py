import torch
from models.slca import SLCA

def get_model(model_name, args): # _train() in trainer.py calls this function
    name = model_name.lower()
    if 'slca' in name:
        return SLCA(args)
    else:
        assert 0
