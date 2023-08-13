import omegaconf 
from openhands.apis.inference import InferenceModel 

cfg = omegaconf.OmegaConf.load("gcn_base_all_test.yaml")
model = InferenceModel(cfg=cfg)
model.init_from_checkpoint_if_available()
if cfg.data.test_pipeline.dataset.inference_mode:
    model.test_inference()
else:
    model.compute_test_accuracy()
    
# import hydra
# from openhands.apis.inference import InferenceModel


# @hydra.main(config_path="./gcn_base_all_test.yaml", config_name="gcn_base_all_test", version_base= None)
# def main(cfg):
#     model = InferenceModel(cfg=cfg)
#     model.init_from_checkpoint_if_available()
#     if cfg.data.test_pipeline.dataset.inference_mode:
#         model.test_inference()
#     else:
#         model.compute_test_accuracy()
