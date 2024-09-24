# -*- coding: utf-8 -*-

import math
from enum import Enum
from typing import Self
import numpy
from src.utilities.utilities import classproperty


class Axis:
    def __init__(self, name: str, vector: tuple[float, float, float]):
        self.name = name
        self.vector = vector

    def __eq__(self, other):
        if not isinstance(other, Axis):
            return False

        return self.name == other.name and self.vector == other.vector

    def __hash__(self):
        return hash(self.vector)

    @property
    def value(self) -> str:
        return self.name

    @classproperty
    def X(cls) -> Self:
        return Axis(name="X", vector=(1.0, 0.0, 0.0))

    @classproperty
    def Y(cls) -> Self:
        return Axis(name="Y", vector=(0.0, 1.0, 0.0))

    @classproperty
    def Z(cls) -> Self:
        return Axis(name="Z", vector=(0.0, 0.0, 1.0))

    @classmethod
    def beam(cls, vector: tuple[float, float, float]):
        return Axis(name="BEAM", vector=vector)


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


def single_rotation_matrix(axis: Axis, angle: float) -> numpy.ndarray:
    """
    Computes the 3D rotation matrix for an active right-handed rotation about a single axis.
    Solves Eqns. 3.59 - 3.61.
    :param axis: The axis of rotation.
    :param angle: The angle of rotation about the axis in ``rad``.
    :return: The rotation matrix.
    """

    match axis:
        case Axis.X:
            return numpy.array((
                (1.0, 0.0, 0.0),
                (0.0, math.cos(angle), -math.sin(angle)),
                (0.0, math.sin(angle), math.cos(angle))
            ))
        case Axis.Y:
            return numpy.array((
                (math.cos(angle), 0.0, math.sin(angle)),
                (0.0, 1.0, 0.0),
                (-math.sin(angle), 0.0, math.cos(angle))
            ))
        case Axis.Z:
            return numpy.array((
                (math.cos(angle), -math.sin(angle), 0.0),
                (math.sin(angle), math.cos(angle), 0.0),
                (0.0, 0.0, 1.0)
            ))
        case _:
            raise ValueError("Non-Cartesian axes are not valid for rotation matrices.")


def euler_rotation_matrix(axis_set: AxisSet, angles: tuple[float, float, float]) -> numpy.ndarray:
    """
    Computes the 3D rotation matrix ``R`` for a set of Euler angles ``(α, β, γ)``.
    Solves Eqn 3.62.
    :param axis_set: The set of Euler axes.
    :param angles: The Euler angles ``(α, β, γ)`` in ``rad``.
    :return: The Euler rotation matrix ``R``.
    """

    R = numpy.eye(3)
    R = numpy.dot(single_rotation_matrix(axis_set.value[0], angles[0]), R)
    R = numpy.dot(single_rotation_matrix(axis_set.value[1], angles[1]), R)
    R = numpy.dot(single_rotation_matrix(axis_set.value[2], angles[2]), R)
    return R


def reduce_vector(vector: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Reflects a lattice vector ``(u, v, w)`` into the positive-axis region.
    Satisfies the constraint ``0 ≤ u ≤ v ≤ w``.
    :param vector: The vector ``(u, v, w)``.
    :param symmetry: The crystal symmetry of the Bravais lattice type.
    :return: The reflected vector.
    """

    u, v, w = vector
    u, v, w = sorted((abs(u), abs(v), abs(w)))
    return u, v, w


def orthogonalise_matrix(R: numpy.ndarray, scaling_tolerance: float = None) -> numpy.ndarray:
    """
    Symmetrically orthogonalises a 3D pseudo-rotation matrix ``R`` by singular value decomposition.
    If ``scaling_tolerance`` is specified, all scale factors in the scaling matrix must be between the tolerance and its reciprocal.
    Throws an ``ArithmeticError`` if the scaling matrix includes a scale factor outside the tolerance range.
    :param R: The pseudo-rotation matrix ``R``.
    :param scaling_tolerance:
    :return: The orthogonalised matrix.
    """
    U, S, VT = numpy.linalg.svd(R)

    if scaling_tolerance is not None:
        factor_bounds = sorted([scaling_tolerance, 1 / scaling_tolerance])

        for factor in S.tolist():
            if not factor_bounds[0] <= factor <= factor_bounds[1]:
                raise ArithmeticError(f"Scale factor {factor} is outside of specified tolerance range: {factor_bounds[0]} - {factor_bounds[1]}")

    return numpy.dot(U, VT)


def euler_angles(rotation_matrix: numpy.ndarray, axis_set: AxisSet) -> tuple[float, float, float]:
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
            angles.append(math.atan2(-rotation_matrix[0][1], rotation_matrix[0][0]))
        elif rotation_matrix[2][2] == -1:
            angles.append(0.0)
            angles.append(math.pi)
            angles.append(-math.atan2(-rotation_matrix[0][1], rotation_matrix[0][0]))
        else:
            angles.append(math.atan2(rotation_matrix[2][0], rotation_matrix[2][1]))
            angles.append(math.acos(rotation_matrix[2][2]))
            angles.append(math.atan2(rotation_matrix[0][2], -rotation_matrix[1][2]))
    else:
        raise NotImplementedError()

    for i, angle in enumerate(angles):
        if angle < 0:
            angles[i] += 2 * math.pi

    return angles[0], angles[1], angles[2]


def rotation_angle(R: numpy.ndarray) -> float:
    """
    Computes the rotation angle ``θ`` of a rotation matrix ``R``.
    Solves Eqn. 4.1.
    :param R: The rotation matrix ``R``.
    :return: The rotation angle ``θ``.
    """

    if 0.5 * (abs(R[0][0]) + abs(R[1][1]) + abs(R[2][2]) - 1) > 1:
        theta = math.acos(1)
    elif 0.5 * (abs(R[0][0]) + abs(R[1][1]) + abs(R[2][2]) - 1) < -1:
        theta = math.acos(-1)
    else:
        theta = math.acos(0.5 * (abs(R[0][0]) + abs(R[1][1]) + abs(R[2][2]) - 1))

    return theta


def misrotation_matrix(R1: numpy.ndarray, R2: numpy.ndarray) -> numpy.ndarray:
    """
    Computes the misrotation matrix ``dR`` between two rotation matrices ``R1`` and ``R2``.
    Solves Eqn. 4.2.
    :param R1: The rotation matrix ``R1``.
    :param R2: The rotation matrix ``R2``.
    :return: The misrotation matrix ``dR``.
    """

    dR = numpy.dot(numpy.linalg.inv(R1), R2)
    return dR


def misrotation_tensor(dR: numpy.ndarray, dx: float) -> numpy.ndarray:
    """
    Computes an approximation of the misrotation tensor ``dω`` of the misrotation matrix ``dR`` over the finite interval ``dx``.
    Solves Eqn. 6.54.
    :param dR: The misrotation matrix ``dR``.
    :param dx: The finite interval ``dx``.
    :return: The approximate misrotation tensor ``dω``.
    """

    dtheta = rotation_angle(dR)

    if dtheta == 0:
        return numpy.zeros((3, 3))
    else:
        return numpy.dot((-3 * dtheta) / (dx * math.sin(dtheta)), numpy.transpose(dR))
