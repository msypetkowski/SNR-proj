import config
from . import perceptron, my_conv, vgg16_pretrained

__all__ = ["perceptron", "my_conv", "vgg16_pretrained"]


def get_model_and_config(args):
    if args.model_type == 'Perceptron':
        conf = config.PerceptronConfig
        Model = models.perceptron.PerceptronModel
    elif args.model_type == 'MyConv':
        conf = config.MyConvConfig
        Model = models.my_conv.MyConvModel
    elif args.model_type == 'VGG16Pretrained':
        conf = config.VGG16PretrainedConfig
        Model = models.vgg16_pretrained.VGG16PretrainedModel
    else:
        raise ValueError("Model type {} not supported".format(args.model_type))
    return Model, conf