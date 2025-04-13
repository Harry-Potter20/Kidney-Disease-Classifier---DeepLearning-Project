import tensorflow as tf
import numpy as np
import random

def mixup_generator(images, labels, alpha=0.2):
    batch_size = tf.shape(images)[0]
    l = np.random.beta(alpha, alpha)
    index = tf.random.shuffle(tf.range(batch_size))
    
    mixed_images = l * images + (1 - l) * tf.gather(images, index)
    mixed_labels = l * labels + (1 - l) * tf.gather(labels, index)

    return mixed_images, mixed_labels

def cutmix_generator(images, labels, alpha=1.0):
    batch_size = tf.shape(images)[0]
    indices = tf.random.shuffle(tf.range(batch_size))
    shuffled_images = tf.gather(images, indices)
    shuffled_labels = tf.gather(labels, indices)
    
    lam = np.random.beta(alpha, alpha)
    
    img_height, img_width = tf.shape(images)[1], tf.shape(images)[2]
    r_x = tf.cast(img_width * np.random.uniform(), tf.int32)
    r_y = tf.cast(img_height * np.random.uniform(), tf.int32)
    r_w = tf.cast(img_width * np.sqrt(1 - lam), tf.int32)
    r_h = tf.cast(img_height * np.sqrt(1 - lam), tf.int32)

    x1 = tf.clip_by_value(r_x - r_w // 2, 0, img_width)
    y1 = tf.clip_by_value(r_y - r_h // 2, 0, img_height)
    x2 = tf.clip_by_value(r_x + r_w // 2, 0, img_width)
    y2 = tf.clip_by_value(r_y + r_h // 2, 0, img_height)

    new_images = tf.tensor_scatter_nd_update(images, [[i] for i in range(batch_size)], 
                    [tf.tensor_scatter_nd_update(images[i], 
                        [[i] for i in tf.range(y1, y2)], shuffled_images[i][y1:y2]) 
                        for i in range(batch_size)])
    
    lam_adjusted = 1 - ((x2 - x1) * (y2 - y1)) / (img_width * img_height)
    new_labels = lam_adjusted * labels + (1 - lam_adjusted) * shuffled_labels
    
    return new_images, new_labels
