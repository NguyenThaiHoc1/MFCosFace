"""
    loss: https://github.com/xiaoboCASIA/SV-X-Softmax/tree/89b536d4001442453f39021ce4b91a5fb43b3354

    https://github.com/tiandunx/loss_function_search/blob/0b17d38da676e1e47423c4983bb8862cf6ca531b/lfs_core/utils/loss.py#L40

    https://github.com/peteryuX/arcface-tf2/blob/dab13506a1b1d25346f0a8b6bf0130e19c3e33b3/modules/models.py#L75
"""
import logging

import tensorflow as tf
from tensorflow.keras.metrics import Mean

from LossFunction.losses import SoftmaxLoss


class Trainer(object):

    def __init__(self, loader, model, current_epochs,
                 max_epochs, steps, learning_rate,
                 logs, save_path, tensorboard_path,
                 loss_type):
        self.loader = loader
        self.model = model
        self.loss_type = loss_type

        self.max_epochs = max_epochs
        self.current_epochs = current_epochs
        self.steps = steps

        self.lr = learning_rate
        self.logs = logs
        self.save_path = save_path
        self.tensorboard_path = tensorboard_path

        # -------- setting up hyper parameter -------
        self.optimizer = None
        self.loss_fn = None

        self.writer_train = self._setup_writer(name_folder='train')

    def _setup_metrics(self, names):
        self.metrics = {name: Mean() for name in names}

    def _setup_optimizer(self):
        self.learning_rate = tf.constant(self.lr)
        optimizer = tf.keras.optimizers.SGD(learning_rate=self.learning_rate, momentum=0.9, nesterov=True)
        return optimizer

    def _setup_loss(self):
        if self.loss_type == 'Softmax':
            loss_fn = SoftmaxLoss()
        else:
            raise ValueError("Pls ! checking loss type.")
        return loss_fn

    def _save_weight(self, path_dir):
        print('[*] save ckpt file!')
        self.model.save_weights(path_dir)

    def _setup_writer(self, name_folder):
        path_writer = self.tensorboard_path / name_folder
        summary_writer = tf.summary.create_file_writer(path_writer)
        return summary_writer

    @tf.function
    def _training_step(self, inputs, labels):

        with tf.GradientTape() as tape:
            logit = self.model(inputs, training=True)
            reg_loss = tf.reduce_sum(self.model.losses)  # regularization_loss
            pred_loss = self.loss_fn(labels, logit)
            total_loss = pred_loss + reg_loss

        grads = tape.gradient(total_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        return total_loss, pred_loss, reg_loss, self.optimizer.lr

    def _train(self, iter_train, show_detail=False):
        loss_sum = []
        format_show = 'Epoch {}/{}: {}/{}, loss={:.2f}, lr={:.4f}'
        for idx in range(self.loader.steps_per_epoch_train):
            inputs, label = next(iter_train)
            total_loss, pred_loss, reg_loss, lr_training = self._training_step(inputs, label)
            loss_sum.append(total_loss.numpy())

            if show_detail:
                print(format_show.format(
                    self.current_epochs,
                    self.max_epochs,
                    self.steps % self.loader.steps_per_epoch_train,
                    self.loader.steps_per_epoch_train,
                    total_loss.numpy(),
                    lr_training,
                ))

            self.steps += 1

        cast_list_loss = tf.cast(loss_sum, dtype=tf.float32)
        mean_loss = tf.reduce_mean(cast_list_loss)
        return mean_loss

    def training(self):
        self.optimizer = self._setup_optimizer()
        self.loss_fn = self._setup_loss()

        train_dataset = iter(self.loader.train)

        while self.current_epochs < self.max_epochs:
            self._setup_metrics(names=['reg_loss', 'pred_loss', 'total_loss'])

            loss_train = self._train(train_dataset, show_detail=True)

            # show information
            verb_str = "* {} Epoch {}/{}: loss={:.2f}"
            print(verb_str.format('TRAIN', self.current_epochs, self.max_epochs, loss_train))

            # saving checkpoint
            if self.current_epochs % 5:
                name_save = 'e_{}_b_{}.ckpt'.format(self.current_epochs,
                                                    self.steps % self.loader.steps_per_epoch,
                                                    loss_train)
                path_save = self.save_path / name_save
                self._save_weight(path_dir=path_save)

            # writing visualization
            with self.writer_train.as_default():
                tf.summary.scalar('loss/total_loss', loss_train, step=self.current_epochs)
                tf.summary.scalar('loss/learning rate', self.optimizer.lr, step=self.current_epochs)

            # updating step and current epoch
            self.current_epochs = self.steps // self.loader.steps_per_epoch_train

            # writen logs
            logging.info(verb_str.format('TRAIN', self.current_epochs, self.max_epochs, loss_train))
