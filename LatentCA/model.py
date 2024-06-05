import tensorflow as tf
from einops import rearrange
from NAFNet.model import SimpleGate
from ViTCA.model import neighbourhood_filters


class NAFCA(tf.keras.models.Model):
    def __init__(self, params, **kwargs):
        super().__init__(**kwargs)
        self.params = params
        self.localized_attention_neighbourhood = self.params['localized_attention_neighbourhood']
        self.attn_filters = neighbourhood_filters(self.localized_attention_neighbourhood)

        self.cell_in_patch_dim = self.params['cell_in_channels']
        self.cell_out_patch_dim = self.params['cell_out_channels']
        self.cell_hidden_channels = self.params['cell_hidden_channels']
        self.cell_update_dim = self.cell_out_patch_dim + self.cell_hidden_channels
        self.cell_dim = self.cell_in_patch_dim + self.cell_out_patch_dim + self.cell_hidden_channels
        self.ffn_channel = self.cell_dim * self.params['ffn_expand']
        self.embed_dim = self.params['embed_dim']

        self.conv1 = tf.keras.layers.Conv2D(self.embed_dim, 1, 1, 'valid')
        self.conv2 = tf.keras.layers.Conv2D(2 * self.embed_dim, 1, 1, 'valid', groups=self.embed_dim)
        self.sg = SimpleGate()
        self.conv_attention = tf.keras.layers.Conv2D(1, 3, 1, 'same')
        self.conv3 = tf.keras.layers.Conv2D(self.cell_dim, 1, 1, 'valid')
        self.conv4 = tf.keras.layers.Conv2D(self.ffn_channel, 1, 1, 'valid')
        self.conv5 = tf.keras.layers.Conv2D(self.cell_dim, 1, 1, 'valid')
        self.conv_out = tf.keras.layers.Conv2D(self.cell_update_dim, 1, 1, 'valid')

        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(self.params['dropout'])
        self.dropout2 = tf.keras.layers.Dropout(self.params['dropout'])
        self.beta = tf.Variable(tf.zeros((1, 1, 1, self.cell_dim)), trainable=True)  # zeros
        self.gamma = tf.Variable(tf.zeros((1, 1, 1, self.cell_dim)), trainable=True)  # zeros

    def call(self, inputs, **kwargs):
        cells, step_n, update_rate = inputs
        for _ in tf.range(step_n[0]):
            cells = self.pass_through(cells, update_rate[0], **kwargs)
        return cells

    def pass_through(self, cells, update_rate, **kwargs):
        b, h, w, _ = cells.shape
        x = cells

        x = self.norm1(x, **kwargs)
        x = self.conv1(x, **kwargs)
        x = self.conv2(x, **kwargs)
        x = self.sg(x, **kwargs)

        pooled = self.conv_attention(x)
        x = x * pooled

        x = self.conv3(x, **kwargs)
        x = self.dropout1(x, **kwargs)
        y = cells + x * self.beta

        x = self.norm2(y, **kwargs)
        x = self.conv4(x, **kwargs)
        x = self.sg(x, **kwargs)
        x = self.conv5(x, **kwargs)
        x = self.dropout2(x, **kwargs)
        x = y + x * self.gamma

        x = self.norm3(x, **kwargs)
        update = self.conv_out(x, **kwargs)

        update_mask = tf.floor(tf.random.uniform([b, h, w, 1]) + update_rate)
        updated = cells[:, :, :, self.cell_in_patch_dim:] + update_mask * update
        return tf.concat([cells[:, :, :, :self.cell_in_patch_dim], updated], -1)

    def localize_attn_fn(self, x, height, width):
        b, h, _, d = x.shape
        y = rearrange(x, 'b h (i j) d -> (b h d) i j 1', i=height, j=width)
        y = tf.nn.conv2d(y, self.attn_filters[:, :, None], strides=1, padding='SAME')
        return rearrange(y, '(b h d) i j filter_n -> b h (i j) filter_n d', b=b, h=h, d=d)

    def seed(self, rgb_in):
        rgb_out_state = tf.zeros((*rgb_in.shape[:3], self.cell_out_patch_dim)) + 0.5
        hidden_state = tf.zeros((*rgb_in.shape[:3], self.cell_hidden_channels))
        return tf.concat([rgb_in, rgb_out_state, hidden_state], axis=-1)

    def get_rgb_out(self, x):
        return x[:, :, :, self.cell_in_patch_dim: self.cell_in_patch_dim + self.cell_out_patch_dim]

    def get_rgb_in(self, x):
        return x[:, :, :, :self.cell_in_patch_dim]

    def get_hidden(self, x):
        return x[:, :, :, self.cell_in_patch_dim + self.cell_out_patch_dim:]


class AdjConv2D(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides, padding, activation, batch_normalization, dropout_rate, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.activation = activation
        self.batch_normalization = batch_normalization
        self.dropout_rate = dropout_rate
        if self.batch_normalization:
            self.layer_sequence = tf.keras.Sequential([
                tf.keras.layers.Conv2D(filters=self.filters, kernel_size=self.kernel_size, strides=self.strides,
                                       padding=self.padding, activation=None),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation(activation=self.activation),
                tf.keras.layers.Dropout(rate=self.dropout_rate)
            ])
        else:
            self.layer_sequence = tf.keras.Sequential([
                tf.keras.layers.Conv2D(filters=self.filters, kernel_size=self.kernel_size, strides=self.strides,
                                       padding=self.padding, activation=self.activation),
                tf.keras.layers.Dropout(rate=self.dropout_rate)
            ])

    def call(self, inputs, **kwargs):
        return self.layer_sequence(inputs, **kwargs)


class AdjConv2DTranspose(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides, padding, activation, batch_normalization, dropout_rate, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.activation = activation
        self.batch_normalization = batch_normalization
        self.dropout_rate = dropout_rate
        if self.batch_normalization:
            self.layer_sequence = tf.keras.Sequential([
                tf.keras.layers.Conv2DTranspose(filters=self.filters, kernel_size=self.kernel_size,
                                                strides=self.strides, padding=self.padding, activation=None),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation(activation=self.activation),
                tf.keras.layers.Dropout(rate=self.dropout_rate)
            ])
        else:
            self.layer_sequence = tf.keras.Sequential([
                tf.keras.layers.Conv2DTranspose(filters=self.filters, kernel_size=self.kernel_size,
                                                strides=self.strides, padding=self.padding, activation=self.activation),
                tf.keras.layers.Dropout(rate=self.dropout_rate)
            ])

    def call(self, inputs, **kwargs):
        return self.layer_sequence(inputs, **kwargs)


class AdjDense(tf.keras.layers.Layer):
    def __init__(self, units, activation, batch_normalization, dropout_rate, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.activation = activation
        self.batch_normalization = batch_normalization
        self.dropout_rate = dropout_rate
        if self.batch_normalization:
            self.layer_sequence = tf.keras.Sequential([
                tf.keras.layers.Dense(units=self.units, activation=None),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation(activation=self.activation),
                tf.keras.layers.Dropout(rate=self.dropout_rate)
            ])
        else:
            self.layer_sequence = tf.keras.Sequential([
                tf.keras.layers.Dense(units=self.units, activation=self.activation),
                tf.keras.layers.Dropout(rate=self.dropout_rate)
            ])

    def call(self, inputs, **kwargs):
        return self.layer_sequence(inputs, **kwargs)


class AutoEncoder(tf.keras.Model):
    def __init__(self, params):
        super(AutoEncoder, self).__init__()
        self.params = params
        self.cell_in_channels = self.params['CAParams']['cell_in_channels']
        self.cell_out_channels = self.params['CAParams']['cell_out_channels']
        self.cell_hidden_channels = self.params['CAParams']['cell_hidden_channels']

        self.encoder = None

        self.latent_input_encoder = tf.keras.layers.Input(shape=self.params['latent_shape'],
                                                          batch_size=self.params['input_shape'][0], dtype=tf.float32)
        self.latent_input_step = tf.keras.layers.Input(shape=(), dtype=tf.int32)
        self.latent_input_update = tf.keras.layers.Input(shape=(), dtype=tf.float32)
        self.latent_input_pool = tf.keras.layers.Input(shape=self.params['pool_shape'][1:],
                                                       batch_size=self.params['input_shape'][0], dtype=tf.float32)
        self.latent_input_sampled = tf.keras.layers.Input(shape=(), dtype=tf.bool)
        self.switch_layer = tf.keras.layers.Lambda(self.switch, name='switch_layer')(
            [self.latent_input_sampled, self.latent_input_encoder, self.latent_input_pool])
        self.latent_ca = self.params['CAParams']['class'](params=self.params['CAParams'], name='latent_layer')(
            [self.switch_layer, self.latent_input_step, self.latent_input_update])
        self.clean_layer = tf.keras.layers.Lambda(self.clean, name='clean_layer')(
            [self.latent_input_step, self.latent_ca])
        self.latent = tf.keras.models.Model(
            inputs=[self.latent_input_encoder, self.latent_input_step, self.latent_input_update,
                    self.latent_input_pool, self.latent_input_sampled], outputs=[self.clean_layer, self.latent_ca])

        self.decoder = None

    def switch(self, args):
        pool_sampled, z, pool = args
        true_fn = lambda: pool
        false_fn = lambda: self.seed(z)
        return tf.cond(pred=pool_sampled[0], true_fn=true_fn, false_fn=false_fn)

    def clean(self, args):
        step_n, cells = args
        true_fn = lambda: cells[:, :, :, :self.cell_in_channels]
        false_fn = lambda: cells[:, :, :, self.cell_in_channels: self.cell_in_channels + self.cell_out_channels]
        return tf.cond(pred=step_n[0] == 0, true_fn=true_fn, false_fn=false_fn)

    def change_mod(self, latent_ca_trainable=False, autoencoder_trainable=True):
        self.encoder.trainable = self.decoder.trainable = autoencoder_trainable
        self.latent.trainable = latent_ca_trainable

    def seed(self, vector_in):
        rgb_out_state = tf.zeros(
            (vector_in.shape[0], vector_in.shape[1], vector_in.shape[2], self.cell_out_channels)) + 0.5
        hidden_state = tf.zeros((vector_in.shape[0], vector_in.shape[1], vector_in.shape[2], self.cell_hidden_channels))
        return tf.concat([vector_in, rgb_out_state, hidden_state], axis=-1)

    def get_channel_out(self, x):
        return x[:, :, :, self.cell_in_channels: self.cell_in_channels + self.cell_out_channels]

    def get_channel_in(self, x):
        return x[:, :, :, :self.cell_in_channels]

    def get_channel_hidden(self, x):
        return x[:, :, :, self.cell_in_channels + self.cell_out_channels:]

    def build_graph(self):
        x = self.encoder_input, self.latent_input_step, self.latent_input_update, self.latent_input_pool, \
            self.latent_input_sampled
        return tf.keras.models.Model(inputs=[x], outputs=self.call(x))

    def embed(self, x, **kwargs):
        return self.encoder(x, **kwargs)

    def pop_out(self, latent_out, skip_out, **kwargs):
        return self.decoder([latent_out, skip_out], **kwargs)

    def ca(self, latent_out, step_n, update_rate, pool, pool_sampled, **kwargs):
        return self.latent([latent_out, step_n, update_rate, pool, pool_sampled], **kwargs)

    def call(self, inputs, **kwargs):
        x, step_n, update_rate, pool, pool_sampled = inputs
        skip_out, latent_out = self.encoder(x, **kwargs)
        clean_out, latent_ca_out = self.latent([latent_out, step_n, update_rate, pool, pool_sampled], **kwargs)
        decoder_out = self.decoder([clean_out, skip_out], **kwargs)
        return decoder_out, latent_out, latent_ca_out, skip_out


class AutoEncoderDown2(AutoEncoder):
    def __init__(self, params):
        super(AutoEncoderDown2, self).__init__(params)
        self.encoder_input = tf.keras.layers.Input(shape=self.params['input_shape'][1:],
                                                   batch_size=self.params['input_shape'][0], dtype=tf.float32)
        self.conv_layer_1 = AdjConv2D(**self.params['Conv2DParams1'], name='conv_layer_1')(self.encoder_input)
        self.pass_through_layer_1 = AdjConv2D(**self.params['PassThroughParams1'], name='pass_through_layer_1')(
            self.encoder_input)
        self.add_layer_encoder = tf.keras.layers.Add(name='add_layer_encoder')(
            [self.conv_layer_1, self.pass_through_layer_1])
        self.conv_layer_2 = AdjConv2D(**self.params['Conv2DParams2'], name='conv_layer_2')(self.add_layer_encoder)
        self.encoder = tf.keras.models.Model(inputs=[self.encoder_input],
                                             outputs=[self.conv_layer_1, self.conv_layer_2])

        self.decoder_input = tf.keras.layers.Input(shape=self.params['latent_shape'],
                                                   batch_size=self.params['input_shape'][0], dtype=tf.float32)
        self.decoder_input_skip = tf.keras.layers.Input(shape=self.encoder.outputs[0].shape[1:],
                                                        batch_size=self.params['input_shape'][0], dtype=tf.float32)
        self.trans_conv_layer_2 = AdjConv2DTranspose(**self.params['Conv2DTransposeParams2'],
                                                     name='trans_conv_layer_2')(self.decoder_input)
        self.add_layer_decoder = tf.keras.layers.Add(name='add_layer_decoder')(
            [self.trans_conv_layer_2, self.decoder_input_skip])
        self.mix_layer = AdjConv2D(**self.params['MixParams'], name='mix_layer_2')(self.add_layer_decoder)
        self.trans_conv_layer_1 = AdjConv2DTranspose(**self.params['Conv2DTransposeParams1'],
                                                     name='trans_conv_layer_1')(self.mix_layer)
        self.decoder = tf.keras.models.Model(inputs=[self.decoder_input, self.decoder_input_skip],
                                             outputs=[self.trans_conv_layer_1])


class AutoEncoderDown3(AutoEncoder):
    def __init__(self, params):
        super(AutoEncoderDown3, self).__init__(params)
        self.encoder_input = tf.keras.layers.Input(shape=self.params['input_shape'][1:],
                                                   batch_size=self.params['input_shape'][0], dtype=tf.float32)
        self.conv_layer_1 = AdjConv2D(**self.params['Conv2DParams1'], name='conv_layer_1')(self.encoder_input)
        self.conv_layer_2 = AdjConv2D(**self.params['Conv2DParams2'], name='conv_layer_2')(self.conv_layer_1)
        self.pass_through_layer_1 = AdjConv2D(**self.params['PassThroughParams1'], name='pass_through_layer_1')(
            self.encoder_input)
        self.pass_through_layer_2 = AdjConv2D(**self.params['PassThroughParams2'], name='pass_through_layer_2')(
            self.pass_through_layer_1)
        self.add_layer_encoder = tf.keras.layers.Add(name='add_layer_encoder')(
            [self.conv_layer_2, self.pass_through_layer_2])
        self.conv_layer_3 = AdjConv2D(**self.params['Conv2DParams3'], name='conv_layer_3')(self.add_layer_encoder)
        self.encoder = tf.keras.models.Model(inputs=[self.encoder_input],
                                             outputs=[self.conv_layer_1, self.conv_layer_3])

        self.decoder_input = tf.keras.layers.Input(shape=self.params['latent_shape'],
                                                   batch_size=self.params['input_shape'][0], dtype=tf.float32)
        self.decoder_input_skip = tf.keras.layers.Input(shape=self.encoder.outputs[0].shape[1:],
                                                        batch_size=self.params['input_shape'][0], dtype=tf.float32)
        self.trans_conv_layer_3 = AdjConv2DTranspose(**self.params['Conv2DTransposeParams3'],
                                                     name='trans_conv_layer_3')(self.decoder_input)
        self.trans_conv_layer_2 = AdjConv2DTranspose(**self.params['Conv2DTransposeParams2'],
                                                     name='trans_conv_layer_2')(self.trans_conv_layer_3)
        self.add_layer_decoder = tf.keras.layers.Add(name='add_layer_decoder')(
            [self.trans_conv_layer_2, self.decoder_input_skip])
        self.mix_layer = AdjConv2D(**self.params['MixParams'], name='mix_layer_2')(self.add_layer_decoder)
        self.trans_conv_layer_1 = AdjConv2DTranspose(**self.params['Conv2DTransposeParams1'],
                                                     name='trans_conv_layer_1')(self.mix_layer)
        self.decoder = tf.keras.models.Model(inputs=[self.decoder_input, self.decoder_input_skip],
                                             outputs=[self.trans_conv_layer_1])
