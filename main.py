import torch.nn as nn
import omegaconf 
from openhands.apis.inference import InferenceModel 

cfg = omegaconf.OmegaConf.load("gcn_base_all_test.yaml")
model = InferenceModel(cfg=cfg)
model.init_from_checkpoint_if_available()
if cfg.data.test_pipeline.dataset.inference_mode:
    model.test_inference()