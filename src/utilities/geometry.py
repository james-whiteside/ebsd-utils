# -*- coding: utf-8 -*-

import copy
import math
from enum import Enum
import numpy
from src.data_structures.phase import CrystalFamily


class Axis(Enum):
    """
    Enumeration of the three Cartesian axes.
    """

    X = (1, 0, 0)
    Y = (0, 1, 0)
    Z = (0, 0, 1)


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


def reduce_matrix(R: numpy.ndarray, symmetry: CrystalFamily) -> numpy.ndarray:
    """
    Reduces a lattice orientation matrix ``R`` into the fundamental unit triangle of its Bravais lattice by reflection.
    :param R: The lattice orientation matrix ``R``.
    :param symmetry: The crystal symmetry of the Bravais lattice type.
    :return: The reduced matrix.
    """
    reduced_R = copy.deepcopy(R)

    if symmetry is CrystalFamily.NONE:
        pass
    elif symmetry is CrystalFamily.C:
        if reduced_R[2][2] < 0:
            reduced_R = numpy.dot(numpy.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]]), reduced_R)

        if reduced_R[1][2] > 0:
            reduced_R = numpy.dot(numpy.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]]), reduced_R)

        if reduced_R[0][2] < 0:
            reduced_R = numpy.dot(numpy.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]), reduced_R)

        if reduced_R[1][2] > reduced_R[0][2]:
            reduced_R = numpy.dot(numpy.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]), reduced_R)
    else:
        raise NotImplementedError()

    return reduced_R


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


def inverse_pole_figure_coordinates(vector: tuple[float, float, float], symmetry: CrystalFamily) -> tuple[float, float]:
    """
    Computes the inverse pole figure coordinates ``(X, Y)`` of a lattice vector ``(u, v, w)`` for the given lattice symmetry.
    :param vector: The lattice vector ``(u, v, w)``.
    :param symmetry: The crystal symmetry of the Bravais lattice type.
    :return: The inverse pole figure coordinates ``(X, Y)``.
    """

    match symmetry:
        case CrystalFamily.C:
            u, v, w = reduce_vector(vector)
            r = math.sqrt(u ** 2 + v ** 2 + w ** 2)
            theta = math.acos(w / r)
            phi = math.atan2(v, u)
            rho = math.tan(theta / 2)
            X = rho * math.cos(phi)
            Y = rho * math.sin(phi)
        case _:
            raise NotImplementedError()

def inverse_stereographic(X: float, Y: float) -> tuple[float, float, float]:
    """
    Computes the Cartesian projection ``(x, y, z)`` of a point ``(X, Y)`` in stereographic space.
    Solves Eqn. 3.34.
    :param X: The stereographic ``X``-coordinate.
    :param Y: The stereographic ``Y``-coordinate.
    :return: The Cartesian coordinates ``(x, y, z)``.
    """

    x = 2 * X / (X ** 2 + Y ** 2 + 1)
    y = 2 * Y / (X ** 2 + Y ** 2 + 1)
    z = (X ** 2 + Y ** 2 - 1) / (X ** 2 + Y ** 2 + 1)
    return x, y, z


def forward_gnomonic(x: float, y: float, z: float) -> tuple[float, float]:
    """
    Computes the forward gnomonic projection ``(X, Y)`` of a point ``(x, y, z)`` in Cartesian space.
    Solves Eqn. 3.37.
    :param x: The Cartesian ``x``-coordinate.
    :param y: The Cartesian ``y``-coordinate.
    :param z: The Cartesian ``z``-coordinate.
    :return: The gnomonic coordinates ``(X, Y)``.
    """

    X = -x / z
    Y = -y / z
    return X, Y


def inverse_gnomonic(X: float, Y: float) -> tuple[float, float, float]:
    """
    Computes the Cartesian projection ``(x, y, z)`` of a point ``(X, Y)`` in gnomonic space.
    Solves Eqn. 3.37.
    :param X: The gnomonic ``X``-coordinate.
    :param Y: The gnomonic ``Y``-coordinate.
    :return: The Cartesian coordinates ``(x, y, z)``.
    """

    x = -X / math.sqrt(X ** 2 + Y ** 2 + 1)
    y = -Y / math.sqrt(X ** 2 + Y ** 2 + 1)
    z = 1 / math.sqrt(X ** 2 + Y ** 2 + 1)
    return x, y, z


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
