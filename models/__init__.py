import config
from . import perceptron, my_conv, vgg16_pretrained


def get_model_and_config(args):
    if args.model_type == 'Perceptron':
        conf = config.PerceptronConfig
        model_cls = perceptron.PerceptronModel
    elif args.model_type == 'MyConv':
        conf = config.MyConvConfig
        model_cls = my_conv.MyConvModel
    elif args.model_type == 'VGG16Pretrained':
        conf = config.VGG16PretrainedConfig
        model_cls = vgg16_pretrained.VGG16PretrainedModel
    else:
        raise ValueError("Model type {} not supported".format(args.model_type))
    return model_cls, conf


__all__ = [get_model_and_config]
