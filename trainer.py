"""
    loss: https://github.com/xiaoboCASIA/SV-X-Softmax/tree/89b536d4001442453f39021ce4b91a5fb43b3354

    https://github.com/tiandunx/loss_function_search/blob/0b17d38da676e1e47423c4983bb8862cf6ca531b/lfs_core/utils/loss.py#L40

    https://github.com/peteryuX/arcface-tf2/blob/dab13506a1b1d25346f0a8b6bf0130e19c3e33b3/modules/models.py#L75
"""
import tensorflow as tf

from LossFunction.losses import SoftmaxLoss


class Trainer(object):

    def __init__(self, loader, model, current_epochs,
                 max_epochs, steps, learning_rate,
                 logs, save_path, loss_type):
        self.loader = loader
        self.model = model
        self.loss_type = loss_type

        self.max_epochs = max_epochs
        self.current_epochs = current_epochs
        self.steps = steps

        self.lr = learning_rate
        self.logs = logs
        self.save_path = save_path

        # -------- setting up hyper parameter -------
        self.optimizer = None
        self.loss_fn = None

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

    def _training_step(self, iter_train):
        inputs, labels = next(iter_train)

        with tf.GradientTape() as tape:
            logist = self.model(inputs, training=True)
            reg_loss = tf.reduce_sum(self.model.losses)  # regularization_loss
            pred_loss = self.loss_fn(labels, logist)
            total_loss = pred_loss + reg_loss

        grads = tape.gradient(total_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        return total_loss, pred_loss, reg_loss, self.optimizer.lr

    def training(self):
        self.optimizer = self._setup_optimizer()
        self.loss_fn = self._setup_loss()

        train_dataset = iter(self.loader.train)

        while self.current_epochs < self.max_epochs:
            total_loss, pred_loss, reg_loss, lr_training = self._training_step(iter_train=train_dataset)

            # if self.steps % 5 == 0:
            verb_str = "Epoch {}/{}: {}/{}, loss={:.2f}, lr={:.4f}"
            print(verb_str.format(self.current_epochs, self.max_epochs,
                                  self.steps % self.loader.steps_per_epoch_train + 1,
                                  self.loader.steps_per_epoch_train,
                                  total_loss.numpy(),
                                  self.learning_rate.numpy()))

            self.steps += 1
            self.current_epochs = self.steps // self.loader.steps_per_epoch_train + 1
