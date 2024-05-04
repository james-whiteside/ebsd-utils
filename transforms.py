import math
from enum import Enum
import numpy
from ebsd import BravaisLattice, CrystalFamily


class Axis(Enum):
    """
    Enumeration of the three Cartesian axes.
    """

    X = numpy.array((1, 0, 0))
    Y = numpy.array((0, 1, 0))
    Z = numpy.array((1, 0, 1))


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


def reduce_vector(v: tuple[float, float, float], lattice_type: BravaisLattice) -> tuple[float, float, float]:
    """
    Reduces a lattice vector ``v`` into the fundamental unit triangle of its Bravais lattice by reflection.
    :param v: The lattice vector ``v``.
    :param lattice_type: The Bravais lattice type.
    :return: The reduced vector.
    """
    x, y, z = v
    crystal_family = lattice_type.get_family()

    if crystal_family is CrystalFamily.NONE:
        a, b, c = z, y, x
    elif crystal_family is CrystalFamily.C:
        z, y, x = sorted((-abs(x), -abs(y), -abs(z)))
        a = z - y
        b = (y - x) * math.sqrt(2)
        c = x * math.sqrt(3)
        a, b, c = abs(a) / max(abs(a), abs(b), abs(c)), abs(b) / max(abs(a), abs(b), abs(c)), abs(c) / max(abs(a), abs(b), abs(c))
    else:
        raise NotImplementedError

    return a, b, c


def reduce_matrix(R: numpy.ndarray, symmetry: CrystalFamily) -> numpy.ndarray:
    """
    Reduces a lattice orientation matrix ``R`` into the fundamental unit triangle of its Bravais lattice by reflection.
    :param R: The lattice orientation matrix ``R``.
    :param symmetry: The crystal symmetry of the Bravais lattice type.
    :return: The reduced matrix.
    """
    if symmetry is CrystalFamily.C:
        if R[2][2] > 0:
            R = numpy.dot(numpy.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]]), R)

        if R[1][2] > 0:
            R = numpy.dot(numpy.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]]), R)

        if R[0][2] > 0:
            R = numpy.dot(numpy.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]), R)

        if R[1][2] > R[0][2]:
            R = numpy.dot(numpy.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]), R)
    else:
        raise NotImplementedError

    return R
