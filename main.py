import omegaconf
from openhands.apis.inference import InferenceModel
from openhands.apis.classification_model import ClassificationModel
from openhands.core.exp_utils import get_trainer

cfg = omegaconf.OmegaConf.load("gcn_adapter_train.yaml")
trainer = get_trainer(cfg)

model = ClassificationModel(cfg=cfg, trainer=trainer)
model.init_from_checkpoint_if_available()
model.fit()

cfg = omegaconf.OmegaConf.load("videos", inference_mode = True)
if cfg.data.test_pipeline.dataset.inference_mode:
    model.test_inference()
else:
    model.compute_test_accuracy()



