import os

from keras.callbacks import Callback


class SaveModelWeights(Callback):
    def __init__(self, dense_model, embedding_model, weights_directory, checkpoint_frequency):
        super().__init__()
        self.dense_model = dense_model
        self.directory = weights_directory
        self.embedding_model = embedding_model
        self.checkpoint_frequency = checkpoint_frequency

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.checkpoint_frequency == 0:
            self.embedding_model.save_weights(
                os.path.join(self.directory, 'embedding_model_weights_epoch_{}.h5'.format(epoch)))
            self.dense_model.save_weight(
                os.path.join(self.directory, 'dense_model_weights_epoch_{}.h5'.format(epoch)))
