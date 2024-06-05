import tensorflow as tf
from einops import rearrange


class OverlapPatchEmbed(tf.keras.layers.Layer):
    def __init__(self, embed_dim=48, bias=False):
        super().__init__()
        self.proj = tf.keras.layers.Conv2D(embed_dim, 3, 1, 'same', use_bias=bias)

    def call(self, x, **kwargs):
        x = self.proj(x, **kwargs)
        return x


class PixelShuffle(tf.keras.layers.Layer):
    def __init__(self, block_size):
        super().__init__()
        self.block_size = block_size

    def call(self, x, **kwargs):
        return tf.nn.depth_to_space(x, self.block_size)


class PixelUnShuffle(tf.keras.layers.Layer):
    def __init__(self, block_size):
        super().__init__()
        self.block_size = block_size

    def call(self, x, **kwargs):
        return tf.nn.space_to_depth(x, self.block_size)


class DownSample(tf.keras.layers.Layer):
    def __init__(self, n_feat):
        super().__init__()

        self.body = tf.keras.Sequential([
            tf.keras.layers.Conv2D(n_feat // 2, 3, 1, 'same', use_bias=False),
            PixelUnShuffle(2)
        ])

    def call(self, x, **kwargs):
        return self.body(x, **kwargs)


class UpSample(tf.keras.layers.Layer):
    def __init__(self, n_feat):
        super().__init__()

        self.body = tf.keras.Sequential([
            tf.keras.layers.Conv2D(n_feat * 2, 3, 1, 'same', use_bias=False),
            PixelShuffle(2)
        ])

    def call(self, x, **kwargs):
        return self.body(x, **kwargs)


class FeedForward(tf.keras.layers.Layer):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super().__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = tf.keras.layers.Conv2D(hidden_features * 2, 1, use_bias=bias)
        self.dw_conv = tf.keras.layers.Conv2D(hidden_features * 2, 3, 1, 'same', groups=hidden_features * 2,
                                              use_bias=bias)
        self.project_out = tf.keras.layers.Conv2D(dim, 1, use_bias=bias)

    def call(self, x, **kwargs):
        x = self.project_in(x)
        x1, x2 = tf.split(self.dw_conv(x), 2, axis=-1)
        x = tf.keras.activations.gelu(x1) * x2
        x = self.project_out(x)
        return x


class CustomAttention(tf.keras.layers.Layer):
    def __init__(self, dim, num_heads, bias):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = tf.Variable(tf.ones(1, 1, num_heads))
        self.qkv = tf.keras.layers.Conv2D(dim * 3, 1, use_bias=bias)
        self.qkv_dw_conv = tf.keras.layers.Conv2D(dim * 3, 3, 1, 'same', groups=dim * 3, use_bias=bias, )
        self.project_out = tf.keras.layers.Conv2D(dim, 1, use_bias=bias)
        self.norm_q = tf.keras.layers.Normalization(axis=-2)
        self.norm_k = tf.keras.layers.Normalization(axis=-2)

    def call(self, x, **kwargs):
        b, h, w, c = x.shape

        qkv = self.qkv_dw_conv(self.qkv(x))
        q, k, v = tf.split(qkv, 3, axis=-1)

        q = rearrange(q, 'b h w (head c) -> b head (h w) c', head=self.num_heads)
        k = rearrange(k, 'b h w (head c) -> b head (h w) c', head=self.num_heads)
        v = rearrange(v, 'b h w (head c) -> b head (h w) c', head=self.num_heads)

        q = self.norm_q(q)
        k = self.norm_k(k)

        t = tf.transpose(k, [0, 1, 3, 2])
        attn = (t @ q)
        attn *= self.temperature
        attn = tf.nn.softmax(attn, axis=-2)

        out = (v @ attn)

        out = rearrange(out, 'b head (h w) c -> b h w (head c)', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias):
        super().__init__()

        self.norm1 = tf.keras.layers.LayerNormalization()
        self.attn = CustomAttention(dim, num_heads, bias)
        self.norm2 = tf.keras.layers.LayerNormalization()
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def call(self, x, **kwargs):
        x = x + self.attn(self.norm1(x), **kwargs)
        x = x + self.ffn(self.norm2(x), **kwargs)

        return x


class Restormer(tf.keras.models.Model):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.out_channels = self.params['out_channels']
        self.dim = self.params['dim']
        self.num_blocks = self.params['num_blocks']
        self.num_refinement_blocks = self.params['num_refinement_blocks']
        self.heads = self.params['heads']
        self.ffn_expansion_factor = self.params['ffn_expansion_factor']
        self.bias = self.params['bias']

        self.patch_embed = OverlapPatchEmbed(self.dim)

        self.encoder_level1 = tf.keras.Sequential([TransformerBlock(dim=self.dim, num_heads=self.heads[0],
                                                                    ffn_expansion_factor=self.ffn_expansion_factor,
                                                                    bias=self.bias) for _ in range(self.num_blocks[0])])

        self.down1_2 = DownSample(self.dim)
        self.encoder_level2 = tf.keras.Sequential([TransformerBlock(dim=int(self.dim * 2 ** 1), num_heads=self.heads[1],
                                                                    ffn_expansion_factor=self.ffn_expansion_factor,
                                                                    bias=self.bias) for _ in range(self.num_blocks[1])])

        self.down2_3 = DownSample(int(self.dim * 2 ** 1))
        self.encoder_level3 = tf.keras.Sequential([TransformerBlock(dim=int(self.dim * 2 ** 2), num_heads=self.heads[2],
                                                                    ffn_expansion_factor=self.ffn_expansion_factor,
                                                                    bias=self.bias) for _ in range(self.num_blocks[2])])

        self.down3_4 = DownSample(int(self.dim * 2 ** 2))
        self.latent = tf.keras.Sequential([TransformerBlock(dim=int(self.dim * 2 ** 3), num_heads=self.heads[3],
                                                            ffn_expansion_factor=self.ffn_expansion_factor,
                                                            bias=self.bias)
                                           for _ in range(self.num_blocks[3])])

        self.up4_3 = UpSample(int(self.dim * 2 ** 3))
        self.reduce_chan_level3 = tf.keras.layers.Conv2D(int(self.dim * 2 ** 2), 1, use_bias=self.bias)
        self.decoder_level3 = tf.keras.Sequential([TransformerBlock(dim=int(self.dim * 2 ** 2), num_heads=self.heads[2],
                                                                    ffn_expansion_factor=self.ffn_expansion_factor,
                                                                    bias=self.bias) for _ in range(self.num_blocks[2])])

        self.up3_2 = UpSample(int(self.dim * 2 ** 2))
        self.reduce_chan_level2 = tf.keras.layers.Conv2D(int(self.dim * 2 ** 1), 1, use_bias=self.bias)
        self.decoder_level2 = tf.keras.Sequential([TransformerBlock(dim=int(self.dim * 2 ** 1), num_heads=self.heads[1],
                                                                    ffn_expansion_factor=self.ffn_expansion_factor,
                                                                    bias=self.bias) for _ in range(self.num_blocks[1])])

        self.up2_1 = UpSample(int(self.dim * 2 ** 1))

        self.decoder_level1 = tf.keras.Sequential([TransformerBlock(dim=int(self.dim * 2 ** 1), num_heads=self.heads[0],
                                                                    ffn_expansion_factor=self.ffn_expansion_factor,
                                                                    bias=self.bias) for _ in range(self.num_blocks[0])])

        self.refinement = tf.keras.Sequential([TransformerBlock(dim=int(self.dim * 2 ** 1), num_heads=self.heads[0],
                                                                ffn_expansion_factor=self.ffn_expansion_factor,
                                                                bias=self.bias)
                                               for _ in range(self.num_refinement_blocks)])
        self.output_layer = tf.keras.layers.Conv2D(self.out_channels, 3, 1, 'same', use_bias=self.bias)

    def call(self, inp, **kwargs):
        inp_enc_level1 = self.patch_embed(inp, **kwargs)
        out_enc_level1 = self.encoder_level1(inp_enc_level1, **kwargs)

        inp_enc_level2 = self.down1_2(out_enc_level1, **kwargs)
        out_enc_level2 = self.encoder_level2(inp_enc_level2, **kwargs)

        inp_enc_level3 = self.down2_3(out_enc_level2, **kwargs)
        out_enc_level3 = self.encoder_level3(inp_enc_level3, **kwargs)

        inp_enc_level4 = self.down3_4(out_enc_level3, **kwargs)
        latent = self.latent(inp_enc_level4, **kwargs)

        inp_dec_level3 = self.up4_3(latent, **kwargs)
        inp_dec_level3 = tf.concat([inp_dec_level3, out_enc_level3], -1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3, **kwargs)
        out_dec_level3 = self.decoder_level3(inp_dec_level3, **kwargs)

        inp_dec_level2 = self.up3_2(out_dec_level3, **kwargs)
        inp_dec_level2 = tf.concat([inp_dec_level2, out_enc_level2], -1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2, **kwargs)
        out_dec_level2 = self.decoder_level2(inp_dec_level2, **kwargs)

        inp_dec_level1 = self.up2_1(out_dec_level2, **kwargs)
        inp_dec_level1 = tf.concat([inp_dec_level1, out_enc_level1], -1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1, **kwargs)

        out_dec_level1 = self.refinement(out_dec_level1, **kwargs)

        return self.output_layer(out_dec_level1, **kwargs) + inp
