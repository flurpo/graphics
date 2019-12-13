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
"""Tests for trilinear interpolation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow_graphics import geometry
from tensorflow_graphics.geometry.representation import grid
from tensorflow_graphics.math.interpolation import trilinear
from tensorflow_graphics.util import test_case


def _sampling_points_from_grid(grid_size, dtype=tf.float64):
  sampling_points = grid.generate((-1.0, -1.0, -1.0),
                                  (1.0, 1.0, 1.0),
                                  grid_size)
  sampling_points = tf.cast(sampling_points, dtype)
  sampling_points = tf.transpose(a=tf.reshape(sampling_points, [-1, 3]))
  return sampling_points


def _sampling_points_in_volume(sampling_points, voxel_size):
  # Convert the sampling points from -1, 1 to [0, dims]
  height, width, depth = voxel_size
  max_x = tf.cast(width - 1, sampling_points.dtype)
  max_y = tf.cast(height - 1, sampling_points.dtype)
  max_z = tf.cast(depth - 1, sampling_points.dtype)

  x, y, z = tf.split(sampling_points, [1, 1, 1], axis=-2)
  x = 0.5 * ((x + 1.0) * max_x)
  y = 0.5 * ((y + 1.0) * max_y)
  z = 0.5 * ((z + 1.0) * max_z)
  sampling_points = tf.concat([x, y, z], axis=-2)

  # Put the points in the format [A1, ... An, M, 3]
  axes = [i for i in range(len(sampling_points.shape))]
  axes[-1], axes[-2] = axes[-2], axes[-1]
  sampling_points = tf.transpose(a=sampling_points, perm=axes)
  return sampling_points


ANGLE_0 = np.array((0.,))
ANGLE_45 = np.array((np.pi / 4.,))
ANGLE_90 = np.array((np.pi / 2.,))
ANGLE_180 = np.array((np.pi,))

AXIS_3D_0 = np.array((0., 0., 0.))
AXIS_3D_X = np.array((1., 0., 0.))
AXIS_3D_Y = np.array((0., 1., 0.))
AXIS_3D_Z = np.array((0., 0., 1.))


def _get_voxel_grid(voxel_size):
  return np.random.uniform(size=voxel_size)


def _get_sampling_points(sampling_points_size):
  return np.random.randint(0, 5, size=sampling_points_size).astype(np.float64)


def _voxels_horizontal_line(voxel_size):
  voxels = np.zeros(voxel_size)
  mid_y = voxel_size[0]//2
  voxels[mid_y, :-1, :-1, :] = 1
  return voxels


def _voxels_vertical_line(voxel_size):
  voxels = np.zeros(voxel_size)
  mid_x = voxel_size[1]//2
  voxels[:-1, mid_x, :-1, :] = 1
  return voxels


VOX_5_1_HOR = _voxels_horizontal_line((5, 5, 5, 1))
VOX_5_1_VER = _voxels_vertical_line((5, 5, 5, 1))

VOX_5_3_HOR = _voxels_horizontal_line((5, 5, 5, 3))
VOX_5_3_VER = _voxels_vertical_line((5, 5, 5, 3))


class TrilinearTest(test_case.TestCase):

  @parameterized.parameters(
      ("must have a rank greater than 3", ((5, 5, 5), (125, 3))),
      ("must have a rank greater than 1", ((2, 5, 5, 5, 1), (3,))),
      ("must have exactly 3 dimensions in axis -1", ((2, 5, 5, 5, 1),
                                                     (2, 125, 4))),
      ("Not all batch dimensions are broadcast-compatible.",
       ((2, 2, 5, 5, 5, 1), (2, 3, 125, 3))),
  )
  def test_interpolate_exception_raised(self, error_msg, shapes):
    """Tests whether exceptions are raised for incompatible shapes."""
    self.assert_exception_is_raised(
        trilinear.interpolate, error_msg, shapes=shapes)

  @parameterized.parameters(
      ((5, 5, 5, 3), (125, 3)),
      ((2, 5, 5, 5, 3), (2, 125, 3)),
      ((2, 2, 5, 5, 5, 3), (2, 2, 15, 3)),
  )
  def test_interpolate_exception_not_raised(self, grid_size,
                                            sampling_points_size):
    """Tests whether exceptions are not raised for compatible shapes."""
    voxel_grid = _get_voxel_grid(grid_size)
    sampling_points = _get_sampling_points(sampling_points_size)

    self.assert_exception_is_not_raised(
        trilinear.interpolate,
        shapes=[],
        grid_3d=voxel_grid,
        sampling_points=sampling_points)

  @parameterized.parameters(
      ((VOX_5_1_HOR, ANGLE_90*AXIS_3D_Z), (VOX_5_1_VER,)),
      ((VOX_5_1_VER, -ANGLE_90*AXIS_3D_Z), (VOX_5_1_HOR,)),
      ((VOX_5_3_HOR, ANGLE_90*AXIS_3D_Z), (VOX_5_3_VER,)),
      ((VOX_5_3_VER, -ANGLE_90*AXIS_3D_Z), (VOX_5_3_HOR,)),
  )
  def test_interpolation_preset(self, test_inputs, test_outputs):
    """Tests whether interpolation results are correct."""
    def func(voxels, euler_angles):
      transf_matrix = \
        geometry.transformation.rotation_matrix_3d.from_euler(euler_angles)

      grid_size = (5, 5, 5)
      sampling_points = _sampling_points_from_grid(grid_size)
      sampling_points = tf.matmul(transf_matrix, sampling_points)
      sampling_points = _sampling_points_in_volume(sampling_points,
                                                   voxels.shape[-4:-1])
      interpolated_points = trilinear.interpolate(voxels, sampling_points)
      interpolated_voxels = tf.reshape(interpolated_points, voxels.shape)
      return interpolated_voxels
    self.assert_output_is_correct(func, test_inputs, test_outputs)

  @parameterized.parameters(
      (1, 4, 4, 4, 1),
      (2, 4, 4, 4, 3),
      (3, 4, 4, 4, 3),
  )
  def test_interpolate_jacobian_random(self,
                                       bsize, height, width, depth, channels):
    """Tests whether jacobian is correct."""
    grid_3d_np = np.random.uniform(size=(bsize, height, width, depth, channels))
    sampling_points_np = np.zeros((bsize, height*width*depth, 3))
    sampling_points_np[:, :, 0] = np.arange(0, height*width*depth)

    # Wrap these in identities because some assert_* ops look at the constant
    # tensor value and mark it as unfeedable.
    grid_3d = tf.identity(tf.convert_to_tensor(value=grid_3d_np))
    sampling_points = tf.identity(
        tf.convert_to_tensor(value=sampling_points_np))

    y = trilinear.interpolate(grid_3d=grid_3d, sampling_points=sampling_points)

    self.assert_jacobian_is_correct(grid_3d, grid_3d_np, y)


if __name__ == "__main__":
  test_case.main()


