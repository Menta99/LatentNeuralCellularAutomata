import tensorflow as tf


class PixelShuffle(tf.keras.layers.Layer):
    def __init__(self, block_size):
        super().__init__()
        self.block_size = block_size

    def call(self, x, **kwargs):
        return tf.nn.depth_to_space(x, self.block_size)


class SimpleGate(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def call(self, x, **kwargs):
        x1, x2 = tf.split(x, 2, axis=-1)
        return x1 * x2


class NAFBlock(tf.keras.models.Model):
    def __init__(self, c, dw_expand=2, ffn_expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * dw_expand
        self.conv1 = tf.keras.layers.Conv2D(dw_channel, 1, 1, 'valid')
        self.conv2 = tf.keras.layers.Conv2D(dw_channel, 3, 1, 'same', groups=dw_channel)
        self.conv3 = tf.keras.layers.Conv2D(c, 1, 1, 'valid')

        # Simplified Channel Attention
        self.sca1 = tf.keras.layers.GlobalAveragePooling2D(keepdims=True)
        self.sca2 = tf.keras.layers.Conv2D(dw_channel // 2, 1, 1, 'valid')

        # Simple Gate
        self.sg = SimpleGate()

        ffn_channel = ffn_expand * c
        self.conv4 = tf.keras.layers.Conv2D(ffn_channel, 1, 1, 'valid')
        self.conv5 = tf.keras.layers.Conv2D(c, 1, 1, 'valid')

        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(drop_out_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_out_rate)

        self.beta = tf.Variable(tf.zeros((1, 1, 1, c)), trainable=True)
        self.gamma = tf.Variable(tf.zeros((1, 1, 1, c)), trainable=True)

    def call(self, inp, **kwargs):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca2(self.sca1(x))
        x = self.conv3(x)

        x = self.dropout1(x, **kwargs)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x, **kwargs)

        return y + x * self.gamma


class NAFNet(tf.keras.models.Model):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.width = self.params['width']
        self.middle_block_num = self.params['middle_block_num']
        self.enc_block_nums = self.params['enc_block_nums']
        self.dec_block_nums = self.params['dec_block_nums']
        self.intro = tf.keras.layers.Conv2D(self.width, 3, 1, 'same')
        self.ending = tf.keras.layers.Conv2D(3, 3, 1, 'same')
        self.encoders = []
        self.decoders = []
        self.middle_blocks = []
        self.ups = []
        self.downs = []

        chan = self.width
        for num in self.enc_block_nums:
            self.encoders.append(tf.keras.Sequential(
                [NAFBlock(chan) for _ in range(num)]
            ))
            self.downs.append(tf.keras.layers.Conv2D(2 * chan, 2, 2, 'valid'))
            chan *= 2

        self.middle_blocks = tf.keras.Sequential(
            [NAFBlock(chan) for _ in range(self.middle_block_num)]
        )

        for num in self.dec_block_nums:
            self.ups.append(tf.keras.Sequential([
                tf.keras.layers.Conv2D(2 * chan, 1, 1, 'valid', use_bias=False),
                PixelShuffle(2)]
            ))
            chan //= 2
            self.decoders.append(tf.keras.Sequential(
                [NAFBlock(chan) for _ in range(num)]
            ))

        self.pad_size = 2 ** len(self.encoders)

    def call(self, inp, **kwargs):
        _, h, w, _ = inp.shape
        inp = self.check_image_size(inp)
        x = self.intro(inp)
        encodings = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x, **kwargs)
            encodings.append(x)
            x = down(x)

        x = self.middle_blocks(x, **kwargs)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encodings[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x, **kwargs)

        x = self.ending(x)
        x = x + inp

        return x[:, :h, :w, :]

    def check_image_size(self, x):
        _, h, w, _ = x.shape
        if h is None and w is None:
            return x
        mod_pad_h = (self.pad_size - h % self.pad_size) % self.pad_size
        mod_pad_w = (self.pad_size - w % self.pad_size) % self.pad_size
        sp = tf.constant([[0, 0], [0, mod_pad_h], [0, mod_pad_w], [0, 0]])
        return tf.pad(x, sp)
