import tensorflow as tf

from models.perceptron import PerceptronModel


class MyConvModel(PerceptronModel):
    def _create_model(self, img_features):
        summary = []
        r = img_features
        for filter_size, strides, depth in self._config.hidden_size:
            conv_strides = strides
            if self._config.use_max_pool:
                conv_strides = (1, 1)
            r = self._layer_wrapper(tf.layers.conv2d(r, depth, filter_size, conv_strides,
                                                     padding='same', use_bias=False),
                                    activation=self._config.activation_fun)
            if self._config.use_max_pool:
                r = tf.layers.max_pooling2d(r, (2, 2), strides=strides, padding='same')
        r = tf.layers.flatten(r)
        for dense_size in self._config.dense_hidden_size:
            r = self._layer_wrapper(tf.layers.dense(r, dense_size, use_bias=False),
                                    activation=self._config.activation_fun)
        r = tf.layers.dense(r, self._config.model_output_shape[0])
        return r, summary
