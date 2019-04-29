import os

from keras.callbacks import Callback


class SaveModelWeights(Callback):
    def __init__(self, models, weights_directory, checkpoint_frequency):
        super().__init__()
        self.models_to_save = models
        self.directory = weights_directory
        self.checkpoint_frequency = checkpoint_frequency

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.checkpoint_frequency == 0:
            for name, model in self.models_to_save:
                model.save_weights(os.path.join(self.directory, '{}_model_weights_epoch_{}.h5'.format(name, epoch)))
