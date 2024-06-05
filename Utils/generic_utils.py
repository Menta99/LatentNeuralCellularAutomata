import os
import requests
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorboard.plugins import projector
from tqdm import tqdm


def send_to_telegram(message, token, chat_id):
    api_url = f'https://api.telegram.org/bot{token}/sendMessage'
    try:
        requests.post(api_url, json={'chat_id': chat_id, 'text': message})
    except Exception as e:
        print(e)


def get_reduced_name_list(path, reduction, test_split, val_split, real_dataset):
    if real_dataset:
        path = os.path.join(path, 'GT')
    train = os.listdir(path)
    train, test = train[:int(len(train) * test_split)], train[int(len(train) * test_split):]
    train, val = train[:int(len(train) * val_split)], train[int(len(train) * val_split):]
    train = train[:int(len(train) * reduction)]
    val = val[:int(len(val) * reduction)]
    test = test[:int(len(test) * reduction)]
    print("Data Split:\nTrain: {}\nValidation: {}\nTest: {}".
          format(len(train), len(val), len(test)))
    return train, val, test


def create_sprite(img_list, shape, path):
    one_square_size = int(np.ceil(np.sqrt(len(img_list))))
    master_width = shape * one_square_size
    master_height = shape * one_square_size
    sprite_image = Image.new(
        mode='RGBA',
        size=(master_width, master_height),
        color=(0, 0, 0, 0)
    )
    for count, image in tqdm(enumerate(img_list)):
        div, mod = divmod(count, one_square_size)
        h_loc = shape * div
        w_loc = shape * mod
        sprite_image.paste(image, (w_loc, h_loc))
    sprite_image.convert('RGB').save(path, transparency=0)


def plot_to_projector(model, generator, path, num_sample=7, noise=True, min_value=0., max_value=1.):
    max_size = 8192  # default value
    cut_pos = (max_size // generator.img_size[0]) ** 2 // (num_sample + 1)
    img_list = []
    labels = []
    embedding_list = []
    pos = 0
    for anchor, _, _ in tqdm(generator):
        embedding_anchor = model.embed(anchor, training=False)[1]
        embedding_anchor = tf.reshape(tensor=embedding_anchor,
                                      shape=(embedding_anchor.shape[0], np.prod(embedding_anchor.shape[1:])))
        if noise:
            positive_list = [generator.create_mask_noise(min_value + ((i / num_sample) * (max_value - min_value)),
                                                         anchor) for i in range(num_sample)]
        else:
            positive_list = [generator.create_mask_blur(min_value + ((i / num_sample) * (max_value - min_value)),
                                                        anchor) for i in range(num_sample)]
        embedding_positive_list = [
            tf.reshape(tensor=model.embed(positive, training=False)[1], shape=embedding_anchor.shape) for positive
            in positive_list]
        for index, anc in enumerate(anchor):
            img_list.append(Image.fromarray(np.array(anc * 255.0).astype(np.uint8)))
            embedding_list.append(embedding_anchor[index])
            labels.append('clean_{}'.format(pos))
            for i in range(num_sample):
                img_list.append(Image.fromarray(np.array(positive_list[i][index] * 255.0).astype(np.uint8)))
                embedding_list.append(embedding_positive_list[i][index])
                labels.append('{}_{}_{}'.format('noise' if noise else 'blur', pos, i))
            pos += 1
        if pos >= cut_pos:
            break

    create_sprite(img_list, generator.img_size[0], os.path.join(path, 'sprite.jpg'))
    with open(os.path.join(path, 'metadata.tsv'), 'w') as f:
        for lab in labels:
            f.write('{}\n'.format(lab))

    embedding_vector = tf.Variable(tf.convert_to_tensor(np.array(embedding_list), dtype=tf.float32))
    checkpoint = tf.train.Checkpoint(embedding=embedding_vector)
    checkpoint.save(os.path.join(path, 'embeddings_data.ckt'))

    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name = 'embedding/.ATTRIBUTES/VARIABLE_VALUE'
    embedding.metadata_path = 'metadata.tsv'
    embedding.sprite.image_path = 'sprite.jpg'
    embedding.sprite.single_image_dim.extend((generator.img_size[0], generator.img_size[0]))
    projector.visualize_embeddings(logdir=path, config=config)


def tensor_to_images(tensor):
    return [Image.fromarray(np.array(i * 255.).astype(np.uint8)) for i in tensor]


def merge_metrics(metrics):
    return [dict(map(lambda x: (x[0], x[1].numpy()), d.items())) for d in metrics]

