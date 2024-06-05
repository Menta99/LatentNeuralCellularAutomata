import numpy as np
import tensorflow as tf
from einops import rearrange
from einops.layers.tensorflow import Rearrange
from scipy import signal
import math


class FeedForward(tf.keras.layers.Layer):
    def __init__(self, hidden_dim, out_dim, dropout=0.):
        super().__init__()
        self.net = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_dim, None),
            tf.keras.layers.Activation('gelu'),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(out_dim, None),
            tf.keras.layers.Dropout(dropout)
        ])

    def call(self, x, **kwargs):
        return self.net(x, **kwargs)


class PreNorm(tf.keras.layers.Layer):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = tf.keras.layers.LayerNormalization(dim)
        self.fn = fn

    def call(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class Attention(tf.keras.layers.Layer):
    def __init__(self, dim, heads=8, head_dim=64, dropout=0.):
        super().__init__()
        inner_dim = head_dim * heads
        project_out = not (heads == 1 and head_dim == dim)
        self.heads = heads
        self.scale = head_dim ** -0.5

        self.to_qkv = tf.keras.layers.Dense(inner_dim * 3, None, use_bias=False)
        self.attend = tf.keras.layers.Softmax(axis=-1)

        self.to_out = tf.keras.Sequential([
            tf.keras.layers.Dense(dim, None),
            tf.keras.layers.Dropout(dropout)
        ]) if project_out else tf.keras.layers.Lambda(lambda x: x)

    def call(self, x, localize=None, h=None, w=None, **kwargs):
        q, k, v = tf.split(self.to_qkv(x), 3, axis=-1)
        q = rearrange(q, 'b n (h c) -> b h n c', h=self.heads)
        k = rearrange(k, 'b n (h c) -> b h n c', h=self.heads)
        v = rearrange(v, 'b n (h c) -> b h n c', h=self.heads)

        q = rearrange(q, 'b h n d -> b h n 1 d')
        k = localize(k, h, w)  # b h n (attn_height attn_width) d
        v = localize(v, h, w)  # b h n (attn_height attn_width) d

        dots = tf.matmul(q, tf.transpose(k, [0, 1, 2, 4, 3])) * self.scale
        attn = self.attend(dots)

        out = tf.matmul(attn, v)
        out = rearrange(out, 'b h n 1 d -> b n (h d)')
        return self.to_out(out, **kwargs)


def encode(x, attn, ff, localize_attn_fn=None, h=None, w=None, **kwargs):
    x = attn(x, localize=localize_attn_fn, h=h, w=w, **kwargs) + x
    x = ff(x, **kwargs) + x
    return x


class Transformer(tf.keras.layers.Layer):
    def __init__(self, dim, depth, heads, head_dim, mlp_dim, dropout=0.):
        super().__init__()
        self.layer_list = []
        for _ in range(depth):
            self.layer_list.append([
                PreNorm(-1, Attention(dim, heads, head_dim, dropout)),
                PreNorm(-1, FeedForward(mlp_dim, dim, dropout))
            ])

    def call(self, x, localize_attn_fn=None, h=None, w=None, **kwargs):
        for attn, ff in self.layer_list:
            x = encode(x, attn, ff, localize_attn_fn, h, w, **kwargs)
        return x


def vit_positional_encoding(n, dim):
    position = tf.expand_dims(tf.range(n, dtype=tf.float32), axis=-1)
    div_term_even = tf.sin(position * tf.exp(tf.range(0, dim, 2, dtype=tf.float32) * (-math.log(10000.0) / dim)))
    div_term_odd = tf.cos(position * tf.exp(tf.range(1, dim, 2, dtype=tf.float32) * (-math.log(10000.0) / dim)))
    pe = tf.expand_dims(tf.concat([div_term_even[:, :1], div_term_odd, div_term_even[:, 1:]], axis=-1), axis=1)
    return tf.transpose(pe, perm=[1, 0, 2])


def neighbourhood_filters(neighbourhood_size):
    return tf.stack(
        [signal.unit_impulse((neighbourhood_size[0], neighbourhood_size[1]), idx=(i, j), dtype=np.float32) for i in
         range(neighbourhood_size[0]) for j in range(neighbourhood_size[1])], axis=-1)


class ViTCA(tf.keras.models.Model):
    def __init__(self, params, **kwargs):
        super().__init__(**kwargs)
        self.params = params
        self.localized_attention_neighbourhood = self.params['localized_attention_neighbourhood']
        self.attn_filters = neighbourhood_filters(self.localized_attention_neighbourhood)
        self.embed_cells = self.params['embed_cells']

        self.patch_height = self.patch_width = self.params['patch_size']

        self.cell_pos_encoding_patch_dim = 0
        self.cell_in_patch_dim = self.params['cell_in_channels'] * self.patch_height * self.patch_width
        self.cell_out_patch_dim = self.params['cell_out_channels'] * self.patch_height * self.patch_width
        self.cell_hidden_channels = self.params['cell_hidden_channels']
        self.cell_update_dim = self.cell_out_patch_dim + self.cell_hidden_channels
        self.cell_dim = self.cell_pos_encoding_patch_dim + self.cell_in_patch_dim + self.cell_out_patch_dim + self.cell_hidden_channels
        self.embed_dim = self.params['embed_dim'] if self.params['embed_cells'] else self.cell_dim

        self.rearrange_cells = Rearrange('b h w c -> b (h w) c')
        self.patchify = Rearrange('b (h p1) (w p2) c -> b h w (p1 p2 c)', p1=self.patch_height, p2=self.patch_width)
        self.un_patchify = Rearrange('b h w (p1 p2 c) -> b (h p1) (w p2) c', p1=self.patch_height, p2=self.patch_width)

        self.cell_to_embedding = tf.keras.layers.Dense(self.embed_dim, activation=None) if self.params[
            'embed_cells'] else None
        self.dropout = tf.keras.layers.Dropout(self.params['embed_dropout'])
        self.transformer = Transformer(self.embed_dim, self.params['depth'], self.params['heads'],
                                       self.embed_dim // self.params['heads'],
                                       self.params['mlp_dim'], self.params['dropout'])
        self.mlp_head = tf.keras.Sequential([
            tf.keras.layers.LayerNormalization(axis=-1),
            tf.keras.layers.Dense(self.cell_update_dim, activation=None, kernel_initializer='zeros',
                                  bias_initializer='zeros')
        ])

    def call(self, inputs, **kwargs):
        cells, step_n, update_rate = inputs
        for _ in tf.range(step_n[0]):
            cells = self.pass_through(cells, update_rate[0], **kwargs)
        return cells

    def localize_attn_fn(self, x, height, width):
        b, h, _, d = x.shape
        y = rearrange(x, 'b h (i j) d -> (b h d) i j 1', i=height, j=width)
        y = tf.nn.conv2d(y, self.attn_filters[:, :, None], strides=1, padding='SAME')
        return rearrange(y, '(b h d) i j filter_n -> b h (i j) filter_n d', b=b, h=h, d=d)

    def pass_through(self, cells, update_rate, **kwargs):
        _cells = cells
        x = self.rearrange_cells(_cells)

        if self.embed_cells:
            x = self.cell_to_embedding(x)

        x = x + vit_positional_encoding(x.shape[-2], x.shape[-1])
        x = self.dropout(x, **kwargs)
        x = self.transformer(x, localize_attn_fn=self.localize_attn_fn, h=cells.shape[-3], w=cells.shape[-2], **kwargs)
        b, h, w, _ = cells.shape
        update = rearrange(self.mlp_head(x), 'b (h w) c -> b h w c', h=h, w=w)
        update_mask = tf.floor(tf.random.uniform([b, h, w, 1]) + update_rate)
        updated = cells[:, :, :, self.cell_pos_encoding_patch_dim + self.cell_in_patch_dim:] + update_mask * update
        return tf.concat([cells[:, :, :, :self.cell_pos_encoding_patch_dim + self.cell_in_patch_dim], updated], -1)

    def seed(self, rgb_in):
        size = (rgb_in.shape[1] // self.patch_height, rgb_in.shape[2] // self.patch_width)
        rgb_in_state = self.patchify(rgb_in)

        rgb_out_state = tf.zeros((rgb_in.shape[0], size[0], size[1], self.cell_out_patch_dim)) + 0.5
        hidden_state = tf.zeros((rgb_in.shape[0], size[0], size[1], self.cell_hidden_channels))
        return tf.concat([rgb_in_state, rgb_out_state, hidden_state], axis=-1)

    def get_rgb_out(self, x):
        rgb_patch = x[:, :, :, self.cell_pos_encoding_patch_dim + self.cell_in_patch_dim:
                               self.cell_pos_encoding_patch_dim + self.cell_in_patch_dim + self.cell_out_patch_dim]
        return self.un_patchify(rgb_patch)

    def get_rgb_in(self, x):
        rgb_patch = x[:, :, :, self.cell_pos_encoding_patch_dim:
                               self.cell_pos_encoding_patch_dim + self.cell_in_patch_dim]
        return self.un_patchify(rgb_patch)

    def get_hidden(self, x):
        return x[:, :, :, self.cell_pos_encoding_patch_dim + self.cell_in_patch_dim + self.cell_out_patch_dim:]
