from configs.CC import Config


global cfg
cfg = Config.fromfile("m2det320_vgg.py")


print(cfg.model.input_size)
print(cfg.model.m2det_config)