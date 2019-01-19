
import tensorflow as tf
slim = tf.contrib.slim
def scale_dimension(dim, scale):
  """Scales the input dimension.

  Args:
    dim: Input dimension (a scalar or a scalar Tensor).
    scale: The amount of scaling applied to the input.

  Returns:
    Scaled dimension.
  """
  if isinstance(dim, tf.Tensor):
    return tf.cast((tf.to_float(dim) - 1.0) * scale + 1.0, dtype=tf.int32)
  else:
    return int((float(dim) - 1.0) * scale + 1.0)

def self_attention(features,
                     model_options=None,
                     weight_decay=0.0001,
                     reuse=None,
                     is_training=False,
                     fine_tune_batch_norm=False):
  """Extracts features by the particular model_variant.

  Args:
    images: A tensor of size [batch, height, width, channels].
    model_options: A ModelOptions instance to configure models.
    weight_decay: The weight decay for model variables.
    reuse: Reuse the model variables or not.
    is_training: Is training or not.
    fine_tune_batch_norm: Fine-tune the batch norm parameters or not.

  Returns:
    concat_logits: A tensor of size [batch, feature_height, feature_width,
      feature_channels], where feature_height/feature_width are determined by
      the images height/width and output_stride.
    end_points: A dictionary from components of the network to the corresponding
      activation.
  """

  # if not model_options.aspp_with_batch_norm:
  #   return features
  if not 1:
      return features
  else:
    batch_norm_params = {
        'is_training': is_training and fine_tune_batch_norm,
        'decay': 0.9997,
        'epsilon': 1e-5,
        'scale': True,
    }

    with slim.arg_scope(
        [slim.conv2d, slim.separable_conv2d],
        weights_regularizer=slim.l2_regularizer(weight_decay),
        activation_fn=tf.nn.relu,
        normalizer_fn=slim.batch_norm,
        padding='SAME',
        stride=1,
        reuse=reuse):
      with slim.arg_scope([slim.batch_norm], **batch_norm_params):
          n, h, w, c = tf.shape(features)[0],tf.shape(features)[1],tf.shape(features)[2],tf.shape(features)[3]
          f1 = slim.conv2d(features, 64, 1, activation_fn=None, normalizer_fn=None, scope="sa_proj_key")
          f_t = tf.transpose(f1, [0, 3, 1, 2])
          proj_key = tf.reshape(f_t, [n, 64, -1])
          f2 = slim.conv2d(features, 64, 1, activation_fn=None, normalizer_fn=None, scope="sa_proj_query")
          proj_query = tf.reshape(f2, [n, -1, 64])
          energy = tf.matmul(proj_query, proj_key)

          attention = tf.nn.softmax(energy)
          f3 = slim.conv2d(features, 512, 1, scope="sa_proj_value")
          proj_value = tf.reshape(f3, [n, 512, -1])

          out = tf.matmul(proj_value, tf.transpose(attention, [0, 2, 1]))
          out = tf.reshape(tf.transpose(out, [0,2,1]), [n,h,w,512])
          skip = slim.conv2d(features, 512, 1, scope="skip")
          print(tf.shape(out))
          print(tf.shape(skip))
          f5 = slim.conv2d(tf.concat([out, skip],3), 256, 1, scope="sa_proj")
          return f5
        # depth = 256
        # branch_logits = []
        #
        # if model_options.add_image_level_feature:
        #   if model_options.crop_size is not None:
        #     image_pooling_crop_size = model_options.image_pooling_crop_size
        #     # If image_pooling_crop_size is not specified, use crop_size.
        #     if image_pooling_crop_size is None:
        #       image_pooling_crop_size = model_options.crop_size
        #     pool_height = scale_dimension(image_pooling_crop_size[0],
        #                                   1. / model_options.output_stride)
        #     pool_width = scale_dimension(image_pooling_crop_size[1],
        #                                  1. / model_options.output_stride)
        #     image_feature = slim.avg_pool2d(
        #         features, [pool_height, pool_width], [1, 1], padding='VALID')
        #     resize_height = scale_dimension(model_options.crop_size[0],
        #                                     1. / model_options.output_stride)
        #     resize_width = scale_dimension(model_options.crop_size[1],
        #                                    1. / model_options.output_stride)
        #   else:
        #     # If crop_size is None, we simply do global pooling.
        #     pool_height = tf.shape(features)[1]
        #     pool_width = tf.shape(features)[2]
        #     image_feature = tf.reduce_mean(features, axis=[1, 2])[:, tf.newaxis,
        #                                                           tf.newaxis]
        #     resize_height = pool_height
        #     resize_width = pool_width
        #   image_feature = slim.conv2d(
        #       image_feature, depth, 1, scope=IMAGE_POOLING_SCOPE)
        #   image_feature = tf.image.resize_bilinear(
        #       image_feature, [resize_height, resize_width], align_corners=True)
        #   # Set shape for resize_height/resize_width if they are not Tensor.
        #   if isinstance(resize_height, tf.Tensor):
        #     resize_height = None
        #   if isinstance(resize_width, tf.Tensor):
        #     resize_width = None
        #   image_feature.set_shape([None, resize_height, resize_width, depth])
        #   branch_logits.append(image_feature)
        #
        # # Employ a 1x1 convolution.
        # branch_logits.append(slim.conv2d(features, depth, 1,
        #                                  scope=ASPP_SCOPE + str(0)))
        #
        # if model_options.atrous_rates:
        #   # Employ 3x3 convolutions with different atrous rates.
        #   for i, rate in enumerate(model_options.atrous_rates, 1):
        #     scope = ASPP_SCOPE + str(i)
        #     if model_options.aspp_with_separable_conv:
        #       aspp_features = split_separable_conv2d(
        #           features,
        #           filters=depth,
        #           rate=rate,
        #           weight_decay=weight_decay,
        #           scope=scope)
        #     else:
        #       aspp_features = slim.conv2d(
        #           features, depth, 3, rate=rate, scope=scope)
        #     branch_logits.append(aspp_features)
        #
        # # Merge branch logits.
        # concat_logits = tf.concat(branch_logits, 3)
        # concat_logits = slim.conv2d(
        #     concat_logits, depth, 1, scope=CONCAT_PROJECTION_SCOPE)
        # concat_logits = slim.dropout(
        #     concat_logits,
        #     keep_prob=0.9,
        #     is_training=is_training,
        #     scope=CONCAT_PROJECTION_SCOPE + '_dropout')
        #
        # return concat_logits, end_points


import numpy as np
a = tf.constant(np.arange(1, 3201, dtype=np.float32), shape=[1, 40, 40, 2])
b=self_attention(a)
with tf.Session() as sess:
    c=sess.run(b)