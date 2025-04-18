import tensorflow as tf
import tensorflow_probability as tfp


def mixup_tf(images, labels, alpha=0.2):
    """
    Applies MixUp augmentation using TensorFlow operations (for tf.data pipelines).
    
    Args:
        images (Tensor): Batch of images [B, H, W, C].
        labels (Tensor): Batch of labels [B, num_classes].
        alpha (float): Parameter for Beta distribution.

    Returns:
        mixed_images, mixed_labels: Augmented images and labels.
    """
    batch_size = tf.shape(images)[0]

    beta_dist = tfp.distributions.Beta(alpha, alpha)
    l = beta_dist.sample([batch_size])
    l = tf.maximum(l, 1.0 - l)  # make sure lambda is >= 0.5

    l = tf.reshape(l, [batch_size, 1, 1, 1])
    label_l = tf.reshape(l, [batch_size, 1])

    index = tf.random.shuffle(tf.range(batch_size))
    mixed_images = images * l + tf.gather(images, index) * (1 - l)
    mixed_labels = labels * label_l + tf.gather(labels, index) * (1 - label_l)

    return mixed_images, mixed_labels


def cutmix_tf(images, labels, alpha=1.0):
    """
    Applies CutMix augmentation using TensorFlow operations (for tf.data pipelines).
    
    Args:
        images (Tensor): Batch of images [B, H, W, C].
        labels (Tensor): Batch of labels [B, num_classes].
        alpha (float): Parameter for Beta distribution.

    Returns:
        cutmix_images, cutmix_labels: Augmented images and labels.
    """
    batch_size = tf.shape(images)[0]
    img_height = tf.shape(images)[1]
    img_width = tf.shape(images)[2]

    beta_dist = tfp.distributions.Beta(alpha, alpha)
    lam = beta_dist.sample()

    r_x = tf.cast(tf.random.uniform([], 0, tf.cast(img_width, tf.float32)), tf.int32)
    r_y = tf.cast(tf.random.uniform([], 0, tf.cast(img_height, tf.float32)), tf.int32)
    r_w = tf.cast(img_width * tf.math.sqrt(1.0 - lam), tf.int32)
    r_h = tf.cast(img_height * tf.math.sqrt(1.0 - lam), tf.int32)

    x1 = tf.clip_by_value(r_x - r_w // 2, 0, img_width)
    y1 = tf.clip_by_value(r_y - r_h // 2, 0, img_height)
    x2 = tf.clip_by_value(r_x + r_w // 2, 0, img_width)
    y2 = tf.clip_by_value(r_y + r_h // 2, 0, img_height)

    # Shuffle the batch
    indices = tf.random.shuffle(tf.range(batch_size))
    shuffled_images = tf.gather(images, indices)
    shuffled_labels = tf.gather(labels, indices)

    # Create binary mask
    crop_mask = tf.ones([y2 - y1, x2 - x1, tf.shape(images)[-1]])
    pad_left = x1
    pad_top = y1
    pad_right = img_width - x2
    pad_bottom = img_height - y2

    mask = tf.pad(crop_mask, [[pad_top, pad_bottom], [pad_left, pad_right], [0, 0]])
    mask = tf.expand_dims(mask, 0)
    mask = tf.tile(mask, [batch_size, 1, 1, 1])

    cutmix_images = images * (1 - mask) + shuffled_images * mask

    # Adjust lambda based on the cutout area
    lam_adjusted = 1 - tf.cast((x2 - x1) * (y2 - y1), tf.float32) / tf.cast((img_width * img_height), tf.float32)
    cutmix_labels = lam_adjusted * labels + (1 - lam_adjusted) * shuffled_labels

    return cutmix_images, cutmix_labels
