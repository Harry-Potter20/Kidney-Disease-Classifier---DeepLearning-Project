import tensorflow as tf
import tensorflow_probability as tfp


def mixup_tf(images, labels, alpha=0.2):
    """
    Applies MixUp augmentation using TensorFlow operations.

    Args:
        images (Tensor): Batch of images [B, H, W, C].
        labels (Tensor): Batch of labels [B, num_classes].
        alpha (float): Parameter for Beta distribution.

    Returns:
        mixed_images, mixed_labels
    """
    batch_size = tf.shape(images)[0]
    beta_dist = tfp.distributions.Beta(concentration1=alpha, concentration0=alpha)

    # Sample lambda values with defined shape
    l = beta_dist.sample(sample_shape=(batch_size,))
    l = tf.cast(tf.maximum(l, 1.0 - l), tf.float32)  # Ensure lambda >= 0.5

    l = tf.reshape(l, [batch_size, 1, 1, 1])  # For broadcasting over image
    label_l = tf.reshape(l, [batch_size, 1])  # For broadcasting over label

    # Shuffle
    index = tf.random.shuffle(tf.range(batch_size))
    images_shuffled = tf.gather(images, index)
    labels_shuffled = tf.gather(labels, index)

    # Mix
    mixed_images = images * l + images_shuffled * (1 - l)
    mixed_labels = labels * label_l + labels_shuffled * (1 - label_l)

    return mixed_images, mixed_labels


def cutmix_tf(images, labels, alpha=1.0):
    """
    Applies CutMix augmentation using TensorFlow operations.

    Args:
        images (Tensor): Batch of images [B, H, W, C].
        labels (Tensor): Batch of labels [B, num_classes].
        alpha (float): Parameter for Beta distribution.

    Returns:
        cutmix_images, cutmix_labels
    """
    batch_size = tf.shape(images)[0]
    img_height = tf.shape(images)[1]
    img_width = tf.shape(images)[2]

    beta_dist = tfp.distributions.Beta(concentration1=alpha, concentration0=alpha)

    lam = tf.cast(beta_dist.sample([]), tf.float32)  # Scalar lambda

    # Bounding box coordinates
    cut_rat = tf.math.sqrt(1.0 - lam)
    cut_w = tf.cast(cut_rat * tf.cast(img_width, tf.float32), tf.int32)
    cut_h = tf.cast(cut_rat * tf.cast(img_height, tf.float32), tf.int32)

    cx = tf.random.uniform([], 0, img_width, dtype=tf.int32)
    cy = tf.random.uniform([], 0, img_height, dtype=tf.int32)

    x1 = tf.clip_by_value(cx - cut_w // 2, 0, img_width)
    y1 = tf.clip_by_value(cy - cut_h // 2, 0, img_height)
    x2 = tf.clip_by_value(cx + cut_w // 2, 0, img_width)
    y2 = tf.clip_by_value(cy + cut_h // 2, 0, img_height)

    # Shuffle images and labels
    index = tf.random.shuffle(tf.range(batch_size))
    shuffled_images = tf.gather(images, index)
    shuffled_labels = tf.gather(labels, index)

    # Apply CutMix mask
    mask = tf.ones((y2 - y1, x2 - x1, tf.shape(images)[-1]), dtype=images.dtype)
    pad_top = y1
    pad_bottom = img_height - y2
    pad_left = x1
    pad_right = img_width - x2

    mask = tf.pad(mask, [[pad_top, pad_bottom], [pad_left, pad_right], [0, 0]])
    mask = tf.expand_dims(mask, 0)  # [1, H, W, C]
    mask = tf.tile(mask, [batch_size, 1, 1, 1])  # [B, H, W, C]

    cutmix_images = images * (1 - mask) + shuffled_images * mask

    # Adjust lambda based on cutout area
    box_area = tf.cast((x2 - x1) * (y2 - y1), tf.float32)
    lam_adjusted = 1.0 - (box_area / tf.cast(img_width * img_height, tf.float32))
    cutmix_labels = lam_adjusted * labels + (1.0 - lam_adjusted) * shuffled_labels

    return cutmix_images, cutmix_labels
