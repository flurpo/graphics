#Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""This module implements trilinear interpolation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow_graphics.util import export_api
from tensorflow_graphics.util import shape


def _get_batch_indexing(dims):
  """Batch-wise indexing for given dimensions.

  Args:
    dims: 1D tensor with size `[A1, ..., An, M]` where M is the number
    of sampling_points.

  Returns:
    A tensor of shape `[M, len([A1, ..., An])]`
  """

  if dims.shape == 1:
    return tf.zeros([dims[0], 0], dtype=tf.int32)

  n_dims = dims.shape[0]-1
  batch_list = []
  for i in range(n_dims-1, -1, -1):
    a = tf.convert_to_tensor(value=[tf.range(0, dims[i], 1)])
    b = tf.ones((1, tf.reduce_prod(input_tensor=dims[i+1:])), dtype=tf.int32)
    c = tf.matmul(tf.transpose(a=a), b)
    c = tf.reshape(c, [-1])
    c = tf.tile(c, [tf.reduce_prod(input_tensor=dims[:i])])
    batch_list.append(c)
  batch_list = batch_list[::-1]
  return tf.transpose(a=tf.convert_to_tensor(value=batch_list))


def interpolate(grid_3d, sampling_points, name=None):
  """Trilinear interpolation on a 3D regular grid.

  Args:
    grid_3d: A tensor with shape `[A1, ..., An, H, W, D, C]` where H, W, D are
      height, width, depth of the grid and C is thenumber of channels.
    sampling_points: A tensor with shape `[A1, ..., An, M, 3]` where M is the
    number of sampling_points. sampling_points outside the grid are projected
    in the grid borders.
    name:  A name for this op that defaults to "trilinear_interpolate".

  Returns:
    A tensor of shape `[A1, ..., An, M, C]`
  """

  with tf.compat.v1.name_scope(name, "trilinear_interpolate",
                               [grid_3d, sampling_points]):
    grid_3d = tf.convert_to_tensor(value=grid_3d)
    sampling_points = tf.convert_to_tensor(value=sampling_points)

    shape.check_static(
        tensor=grid_3d, tensor_name="grid_3d", has_rank_greater_than=3)
    shape.check_static(tensor=sampling_points,
                       tensor_name="sampling_points",
                       has_dim_equals=(-1, 3),
                       has_rank_greater_than=1)
    shape.compare_batch_dimensions(
        tensors=(grid_3d, sampling_points),
        last_axes=(-5, -3),
        tensor_names=("grid_3d", "sampling_points"),
        broadcast_compatible=True)

    height = tf.shape(input=grid_3d)[-4]
    width = tf.shape(input=grid_3d)[-3]
    depth = tf.shape(input=grid_3d)[-2]
    channels = tf.shape(input=grid_3d)[-1]

    batch_dims = tf.shape(input=sampling_points)[:-2]
    n_points = tf.shape(input=sampling_points)[-2]

    sampling_points = tf.reshape(sampling_points, [-1, 3])
    p0 = tf.floor(sampling_points)
    p1 = p0 + 1

    # clip
    clip_val = tf.convert_to_tensor(value=[height-1, width-1, depth-1],
                                    dtype=sampling_points.dtype)
    p0 = tf.clip_by_value(p0, 0, clip_val)
    p1 = tf.clip_by_value(p1, 0, clip_val)

    p0_int = tf.cast(p0, tf.int32)
    p1_int = tf.cast(p1, tf.int32)

    # split
    x, y, z = tf.split(sampling_points, [1, 1, 1], axis=-1)

    x0, y0, z0 = tf.split(p0, [1, 1, 1], axis=-1)
    x1, y1, z1 = tf.split(p1, [1, 1, 1], axis=-1)

    x0_int, y0_int, z0_int = tf.split(p0_int, [1, 1, 1], axis=-1)
    x1_int, y1_int, z1_int = tf.split(p1_int, [1, 1, 1], axis=-1)

    content_x = tf.concat([x0_int, x1_int, x0_int, x1_int,
                           x0_int, x1_int, x0_int, x1_int], axis=0)
    content_y = tf.concat([y0_int, y0_int, y1_int, y1_int,
                           y0_int, y0_int, y1_int, y1_int], axis=0)
    content_z = tf.concat([z0_int, z0_int, z0_int, z0_int,
                           z1_int, z1_int, z1_int, z1_int], axis=0)

    dims = tf.concat([batch_dims, [n_points]], axis=0)
    b = _get_batch_indexing(dims)

    content_batch = tf.concat([b, b, b, b, b, b, b, b], axis=0)

    indices = tf.concat([content_batch, content_x, content_y, content_z],
                        axis=1)
    content = tf.gather_nd(grid_3d, indices)

    total_points = tf.shape(input=x)[0]

    weights_x = tf.concat([(x1-x), (x-x0), (x1-x), (x-x0),
                           (x1-x), (x-x0), (x1-x), (x-x0)], axis=0)

    weights_y = tf.concat([(y1-y), (y1-y), (y-y0), (y-y0),
                           (y1-y), (y1-y), (y-y0), (y-y0)], axis=0)

    weights_z = tf.concat([(z1-z), (z1-z), (z1-z), (z1-z),
                           (z-z0), (z-z0), (z-z0), (z-z0)], axis=0)

    weights = weights_x * weights_y * weights_z
    out = tf.add_n(tf.split(weights*content, [total_points]*8, -2))

    output_dims = tf.concat([batch_dims, [n_points, channels]], axis=0)
    return tf.reshape(out, output_dims)


# API contains all public functions and classes.
__all__ = export_api.get_functions_and_classes()
