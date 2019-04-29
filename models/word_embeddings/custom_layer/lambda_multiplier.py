from keras.engine import Layer


class LambdaScalarMultiplier(Layer):
    def __init__(self, **kwargs):
        self.w = None
        super(LambdaScalarMultiplier, self).__init__(**kwargs)

    def build(self, input_shape):
        self.w = self.add_weight(shape=(1, 1), initializer='one', trainable=True, name='w')
        super(LambdaScalarMultiplier, self).build(input_shape)

    def call(self, x, mask=None):
        return self.w * x

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1]
