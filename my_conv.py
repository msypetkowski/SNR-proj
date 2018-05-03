import tensorflow as tf
from perceptron import PerceptronModel

class MyConvModel(PerceptronModel):

    def _create_model(self, img_features):
        summary = []
        r = img_features
        for filter_size, strides, depth in self._config.hidden_size:
            r = self._layer_wrapper(tf.layers.conv2d(r, depth, filter_size, strides,
                                                        padding='same', use_bias=False),
                                    activation=self._config.activation_fun)
        r = tf.layers.flatten(r)
        r = tf.layers.dense(r, self._config.model_output_shape[0])
        return r, summary
