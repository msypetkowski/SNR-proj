from utils import group
import numpy as np

# TODO: move more methods / write abstracts here
class BaseModel:

    def _get_feed_dict(self, images, labels=None, for_training=False):
        feed_dict = {
            self._images: images,
        }
        if labels is not None:
            feed_dict[self._labels] = labels
        if not isinstance(self._is_training, bool):
            feed_dict[self._is_training] = for_training
        return feed_dict

    def eval_op(self):
        return self._prediction

    def eval_fun(self, sess, images):
        # avoid OOM error by grouping into batches
        results = []
        for img in group(images, self._config.batch_size):
            results.extend(sess.run(self.eval_op(), self._get_feed_dict(img)))
        assert len(results) == len(images)
        return results

    def accuracy_op(self):
        return self._accuracy

    def accuracy_fun(self, sess, images, labels):
        # avoid OOM error by grouping into batches
        correct_answers = 0
        sum_length = 0
        assert len(images) == len(labels)
        for img, lbl in zip(group(images, self._config.batch_size),
                            group(labels, self._config.batch_size)):
            correct_answers += sess.run(self.accuracy_op(),self._get_feed_dict(img, lbl)) * len(img)
            sum_length += len(img)
        return correct_answers / sum_length
