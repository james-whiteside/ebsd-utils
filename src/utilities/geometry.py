# -*- coding: utf-8 -*-

from math import pi, sin, acos, atan2, hypot
from enum import Enum

from numba import njit
from numpy import ndarray, array, zeros, dot, transpose, cos as array_cos, sin as array_sin, float64, empty, \
    array_equal
from numpy.linalg import inv, svd


class Axis:
    def __init__(self, name: str, vector: tuple[float, float, float]):
        self.name = name

        length = hypot(*vector)

        if length == 0:
            raise ValueError("Axis vector cannot have zero length.")
        else:
            self._vector = vector[0] / length, vector[1] / length, vector[2] / length
            self.vector = array(self._vector, dtype=float64)

    def __eq__(self, other):
        if not isinstance(other, Axis):
            return False

        return self.name == other.name and self._vector == other._vector

    def __hash__(self):
        return hash(self._vector)

    @classmethod
    def beam(cls, vector: tuple[float, float, float]):
        return Axis(name="BEAM", vector=vector)

Axis.X = Axis(name="X", vector=(1.0, 0.0, 0.0))
Axis.Y = Axis(name="Y", vector=(0.0, 1.0, 0.0))
Axis.Z = Axis(name="Z", vector=(0.0, 0.0, 1.0))
_x_axis_vector = Axis.X.vector
_y_axis_vector = Axis.Y.vector
_z_axis_vector = Axis.Z.vector


class AxisSet(Enum):
    """
    Enumeration of the Euler axis sets, as listed in Tab. 3.1.
    """

    # Tait-Bryan axis sets
    XYZ = (Axis.X, Axis.Y, Axis.Z)
    XZY = (Axis.X, Axis.Z, Axis.Y)
    YXZ = (Axis.Y, Axis.X, Axis.Z)
    YZX = (Axis.Y, Axis.Z, Axis.X)
    ZXY = (Axis.Z, Axis.X, Axis.Y)
    ZYX = (Axis.Z, Axis.Y, Axis.X)
    # Proper Euler axis sets
    XYX = (Axis.X, Axis.Y, Axis.X)
    XZX = (Axis.X, Axis.Z, Axis.X)
    YXY = (Axis.Y, Axis.X, Axis.Y)
    YZY = (Axis.Y, Axis.Z, Axis.Y)
    ZXZ = (Axis.Z, Axis.X, Axis.Z)
    ZYZ = (Axis.Z, Axis.Y, Axis.Z)

    @property
    def vectors(self) -> tuple[ndarray, ndarray, ndarray]:
        return self.value[0].vector, self.value[1].vector, self.value[2].vector


@njit(inline="always")
def single_rotation_matrix(axis_vector: ndarray, angle: ndarray) -> ndarray:
    """
    Computes the 3D rotation matrix for an active right-handed rotation about a single axis.
    Solves Eqns. 3.59 - 3.61. For arbitrary axes, uses: R = I + sin(θ)K + (1 - cos(θ))K²
    :param axis_vector: The axis of rotation.
    :param angle: The angle of rotation about the axis in ``rad``.
    :return: The rotation matrix.
    """

    c = array_cos(angle)
    s = array_sin(angle)
    R = empty((3, 3), dtype=float64)

    if array_equal(axis_vector, _x_axis_vector):
        R[0, 0] = 1.0; R[0, 1] = 0.0; R[0, 2] = 0.0
        R[1, 0] = 0.0; R[1, 1] = c; R[1, 2] = -s
        R[2, 0] = 0.0; R[2, 1] = s; R[2, 2] = c
    elif array_equal(axis_vector, _y_axis_vector):
        R[0, 0] = c; R[0, 1] = 0.0; R[0, 2] = s
        R[1, 0] = 0.0; R[1, 1] = 1.0; R[1, 2] = 0.0
        R[2, 0] = -s; R[2, 1] = 0.0; R[2, 2] = c
    elif array_equal(axis_vector, _z_axis_vector):
        R[0, 0] = c; R[0, 1] = -s; R[0, 2] = 0.0
        R[1, 0] = s; R[1, 1] = c; R[1, 2] = 0.0
        R[2, 0] = 0.0; R[2, 1] = 0.0; R[2, 2] = 1.0
    else:
        ux, uy, uz = axis_vector[0], axis_vector[1], axis_vector[2]
        omc = 1.0 - c

        R[0, 0] = c + ux * ux * omc
        R[0, 1] = ux * uy * omc - uz * s
        R[0, 2] = ux * uz * omc + uy * s

        R[1, 0] = uy * ux * omc + uz * s
        R[1, 1] = c + uy * uy * omc
        R[1, 2] = uy * uz * omc - ux * s

        R[2, 0] = uz * ux * omc - uy * s
        R[2, 1] = uz * uy * omc + ux * s
        R[2, 2] = c + uz * uz * omc

    return R


@njit(inline="always")
def euler_rotation_matrix(axis_set: tuple[ndarray, ndarray, ndarray], angles: ndarray) -> ndarray:
    """
    Computes the 3D rotation matrix ``R`` for a set of Euler angles ``(α, β, γ)``.
    Solves Eqn 3.62.
    :param axis_set: The set of Euler axes.
    :param angles: The Euler angles ``(α, β, γ)`` in ``rad``.
    :return: The Euler rotation matrix ``R``.
    """

    R1 = single_rotation_matrix(axis_set[0], angles[0])
    R2 = single_rotation_matrix(axis_set[1], angles[1])
    R3 = single_rotation_matrix(axis_set[2], angles[2])
    return R3 @ R2 @ R1


@njit(inline="always")
def reduce_vector(vector: ndarray) -> tuple[ndarray, ndarray, ndarray]:
    """
    Reflects a lattice vector ``(u, v, w)`` into the positive-axis region.
    Satisfies the constraint ``0 ≤ u ≤ v ≤ w``.
    :param vector: The vector ``(u, v, w)``.
    :return: The reflected vector.
    """

    u = abs(vector[0])
    v = abs(vector[1])
    w = abs(vector[2])

    if u > v:
        u, v = v, u

    if v > w:
        v, w = w, v

        if u > v:
            u, v = v, u

    return u, v, w


def orthogonalise_matrix(R: ndarray, scaling_tolerance: float = None) -> ndarray:
    """
    Symmetrically orthogonalises a 3D pseudo-rotation matrix ``R`` by singular value decomposition.
    If ``scaling_tolerance`` is specified, all scale factors in the scaling matrix must be between the tolerance and its reciprocal.
    Throws an ``ArithmeticError`` if the scaling matrix includes a scale factor outside the tolerance range.
    :param R: The pseudo-rotation matrix ``R``.
    :param scaling_tolerance:
    :return: The orthogonalised matrix.
    """
    U, S, VT = svd(R)

    if scaling_tolerance is not None:
        factor_bounds = sorted([scaling_tolerance, 1 / scaling_tolerance])

        for factor in S.tolist():
            if not factor_bounds[0] <= factor <= factor_bounds[1]:
                raise ArithmeticError(f"Scale factor {factor} is outside of specified tolerance range: {factor_bounds[0]} - {factor_bounds[1]}")

    return dot(U, VT)


def euler_angles(rotation_matrix: ndarray, axis_set: AxisSet) -> tuple[float, float, float]:
    """
    Computes the Euler angles for a 3D rotation matrix.
    :param rotation_matrix: The rotation matrix.
    :param axis_set: The set of Euler axes.
    :return: The Euler angles in ``rad``.
    """
    angles: list[float] = list()

    if axis_set is AxisSet.ZXZ:
        if rotation_matrix[2][2] == 1:
            angles.append(0.0)
            angles.append(0.0)
            angles.append(atan2(-rotation_matrix[0][1], rotation_matrix[0][0]))
        elif rotation_matrix[2][2] == -1:
            angles.append(0.0)
            angles.append(pi)
            angles.append(-atan2(-rotation_matrix[0][1], rotation_matrix[0][0]))
        else:
            angles.append(atan2(rotation_matrix[2][0], rotation_matrix[2][1]))
            angles.append(acos(rotation_matrix[2][2]))
            angles.append(atan2(rotation_matrix[0][2], -rotation_matrix[1][2]))
    else:
        raise NotImplementedError(f"Extraction of Euler angles from rotation matrix not implemented for Euler axis set: {axis_set.name.lower()}")

    for i, angle in enumerate(angles):
        if angle < 0:
            angles[i] += 2 * pi

    return angles[0], angles[1], angles[2]


def rotation_angle(R: ndarray) -> float:
    """
    Computes the rotation angle ``θ`` of a rotation matrix ``R``.
    Solves Eqn. 4.1.
    :param R: The rotation matrix ``R``.
    :return: The rotation angle ``θ``.
    """

    if 0.5 * (abs(R[0][0]) + abs(R[1][1]) + abs(R[2][2]) - 1) > 1:
        theta = acos(1)
    elif 0.5 * (abs(R[0][0]) + abs(R[1][1]) + abs(R[2][2]) - 1) < -1:
        theta = acos(-1)
    else:
        theta = acos(0.5 * (abs(R[0][0]) + abs(R[1][1]) + abs(R[2][2]) - 1))

    return theta


def misrotation_matrix(R1: ndarray, R2: ndarray) -> ndarray:
    """
    Computes the misrotation matrix ``dR`` between two rotation matrices ``R1`` and ``R2``.
    Solves Eqn. 4.2.
    :param R1: The rotation matrix ``R1``.
    :param R2: The rotation matrix ``R2``.
    :return: The misrotation matrix ``dR``.
    """

    dR = dot(inv(R1), R2)
    return dR


def misrotation_tensor(dR: ndarray, dx: float) -> ndarray:
    """
    Computes an approximation of the misrotation tensor ``dω`` of the misrotation matrix ``dR`` over the finite interval ``dx``.
    Solves Eqn. 6.54.
    :param dR: The misrotation matrix ``dR``.
    :param dx: The finite interval ``dx``.
    :return: The approximate misrotation tensor ``dω``.
    """

    dtheta = rotation_angle(dR)

    if dtheta == 0:
        return zeros((3, 3))
    else:
        return dot((-3 * dtheta) / (dx * sin(dtheta)), transpose(dR))
