import math
from enum import Enum
import numpy
from material import CrystalFamily, BravaisLattice


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


def euler_angles(rotation_matrix: numpy.ndarray, axis_set: AxisSet) -> tuple[float, float, float]:
    """
    Computes the Euler angles for a 3D rotation matrix.
    :param rotation_matrix: The rotation matrix.
    :param axis_set: The set of Euler axes.
    :return: The Euler angles in ``rad``.
    """

    if axis_set is AxisSet.ZXZ:
        phi1 = math.acos(rotation_matrix[2][1] / math.sqrt(1 - rotation_matrix[2][2] ** 2))
        Phi = math.acos(rotation_matrix[2][2])
        phi2 = math.acos(-rotation_matrix[1][2] / math.sqrt(1 - rotation_matrix[2][2] ** 2))
        return phi1, Phi, phi2
    else:
        raise NotImplementedError


def forward_stereographic(x: float, y: float, z: float) -> tuple[float, float]:
    """
    Computes the forward stereographic projection ``(X, Y)`` of a point ``(x, y, z)`` in Cartesian space.
    Solves Eqn. 3.34.
    :param x: The Cartesian ``x``-coordinate.
    :param y: The Cartesian ``y``-coordinate.
    :param z: The Cartesian ``z``-coordinate.
    :return: The stereographic coordinates ``(X, Y)``.
    """

    X = -x / (1 - z)
    Y = -y / (1 - z)
    return X, Y


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