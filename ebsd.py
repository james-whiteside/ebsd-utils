# -*- coding: utf-8 -*-

import math
import datetime
import copy
import itertools
import sys
import os
import shutil
from enum import Enum

import numpy
from numba import jit, cuda
from PIL import Image as image
import utilities
import fileloader
import channelling
import orientation


GENERIC_PHASE_IDS = (0, 4294967294, 4294967295)


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


class CrystalFamily(Enum):
	"""
	Enumeration of the crystal lattice types, as listed in Tab. A2.1.
	"""

	C = "c"
	T = "t"
	O = "o"
	H = "h"
	M = "m"
	A = "a"
	NONE = "None"


class BravaisLattice(Enum):
	"""
	Enumeration of the Bravais lattice types, as listed in Tab. A2.1.
	"""

	CP = "cP"
	CI = "cI"
	CF = "cF"
	TP = "tP"
	TI = "tI"
	OP = "oP"
	OI = "oI"
	OF = "oF"
	OS = "oS"
	HP = "hP"
	HR = "hR"
	MP = "mP"
	MS = "mS"
	AP = "aP"
	NONE = "None"

	def get_family(self) -> CrystalFamily:
		if self is BravaisLattice.NONE:
			return CrystalFamily.NONE
		else:
			return CrystalFamily(self.value[0])


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

	X = x / (1 - z)
	Y = y / (1 - z)
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


def differential_rotation_tensor(dR: numpy.ndarray, dx: float) -> numpy.ndarray:
	"""
	Computes an approximation of the differential lattice rotation tensor ``dω`` of the lattice misorientation matrix ``dR`` over the finite interval ``dx``.
	Solves Eqn. 6.54.
	:param dR: The lattice misorientation matrix ``dR``.
	:param dx: The finite interval ``dx``.
	:return: The approximate lattice rotation tensor ``dω``.
	"""

	dtheta = rotation_angle(dR)

	if dtheta == 0:
		return numpy.zeros((3, 3))
	else:
		return numpy.dot((-3 * dtheta) / (dx * math.sin(dtheta)), dR)


# Computes the GND density of a scan pixel ``ρ`` with coordinates ``(x, y)``.
def gnd_density(
		width: int,
		height: int,
		phase: list[list[int]],
		R: list[list[numpy.ndarray]],
		x: int,
		y: int,
		w: float,
		ka: float
) -> float:

	def f(dx: int, dy: int):
		dR = misrotation_matrix(R[y][x], R[y + dy][x + dx])
		return differential_rotation_tensor(dR, w)

	f_kernel = [
		[None, f(+1, 0), f(-1, 0)],
		[f(0, +1), None, None],
		[f(0, -1), None, None],
	]

	weights = [
		[None, 0.5, 0.5],
		[0.5, None, None],
		[0.5, None, None],
	]

	if x == 0 or phase[y][x] != phase[y][x-1]:
		weights[0][-1] = 0
		weights[0][1] *= 2

	if x == width - 1 or phase[y][x] != phase[y][x+1]:
		weights[0][-1] *= 2
		weights[0][1] = 0

	if y == 0 or phase[y][x] != phase[y-1][x]:
		weights[-1][0] = 0
		weights[1][0] *= 2

	if y == height - 1 or phase[y][x] != phase[y+1][x]:
		weights[-1][0] *= 2
		weights[1][0] = 0

	S = sum((
		abs(weights[0][-1] * f_kernel[0][-1][0][2] + weights[0][1] * f_kernel[0][1][0][2]) / 2,
		abs(weights[0][-1] * f_kernel[0][-1][0][1] + weights[0][1] * f_kernel[0][1][0][1]) / 2,
		abs(weights[-1][0] * f_kernel[-1][0][1][2] + weights[1][0] * f_kernel[1][0][1][2]) / 2,
		abs(weights[-1][0] * f_kernel[-1][0][1][0] + weights[1][0] * f_kernel[1][0][1][0]) / 2,
		abs(
			weights[-1][0] * f_kernel[-1][0][2][0] + weights[1][0] * f_kernel[1][0][2][0]
			- weights[0][-1] * f_kernel[0][-1][2][1] - weights[0][1] * f_kernel[0][1][2][1]
		) / 2,
	))

	rho = (3.6 / ka) * S ** 2
	return rho


def close_pack_distance(
		lattice_type: BravaisLattice,
		constants: tuple[float, float, float],
		angles: tuple[float, float, float]
) -> float:
	if lattice_type is BravaisLattice.CP:
		return constants[0]
	elif lattice_type is BravaisLattice.CI:
		return math.sqrt(3) * constants[0] / 2
	elif lattice_type is BravaisLattice.CF:
		return math.sqrt(2) * constants[0] / 2
	else:
		raise NotImplementedError


def calR(data):
	R = list()

	for y in range(data['height']):
		R.append(list())

		for x in range(data['width']):
			R[y].append(reduce_matrix(euler_rotation_matrix(AxisSet.ZXZ, data['data']['euler'][y][x]), CrystalFamily.C))

	return R


def calGND(data, dx):
	R = copy.deepcopy(data['data']['R'])

	R.append(list())

	for y in range(data['height']):
		R[y].append(numpy.eye(3))

	for x in range(data['width'] + 1):
		R[data['height']].append(numpy.eye(3))

	GND = list()

	for y in range(data['height']):
		GND.append(list())

		for x in range(data['width']):
			if data['phases'][data['data']['phase'][y][x]]['ID'] in GENERIC_PHASE_IDS:
				GND[y].append(0)
			else:
				lattice_type = data['phases'][data['data']['phase'][y][x]]['type']
				constants = data['phases'][data['data']['phase'][y][x]]['constants']
				angles = data['phases'][data['data']['phase'][y][x]]['angles']
				ka = close_pack_distance(lattice_type, constants, angles)
				rho = gnd_density(data['width'], data['height'], data['data']['phase'], R, x, y, dx, ka)

				if rho == 0:
					GND[y].append(rho)
				else:
					GND[y].append(math.log10(rho))

	return GND


@jit(nopython=True)
def dbscan(phase, R, width, height, n, epsilon):
	
	k = 0
	categories = numpy.zeros((height, width))
	clusters = numpy.zeros((height, width))
	
	for y in range(height):
		for x in range(width):
			if phase[y][x] == 0:
				categories[y][x] = 3
	
	for y0 in range(height):
		for x0 in range(width):
			if categories[y0][x0] == 0:
				m = 0
				
				for y1 in range(height):
					for x1 in range(width):
						if phase[y0][x0] == phase[y1][x1]:
							dR = numpy.dot(numpy.linalg.inv(R[y0][x0]), R[y1][x1])
							
							if 0.5 * (abs(dR[0][0]) + abs(dR[1][1]) + abs(dR[2][2]) - 1) > 1:
								dTheta = math.acos(1)
							elif 0.5 * (abs(dR[0][0]) + abs(dR[1][1]) + abs(dR[2][2]) - 1) < -1:
								dTheta = math.acos(-1)
							else:
								dTheta = math.acos(0.5 * (abs(dR[0][0]) + abs(dR[1][1]) + abs(dR[2][2]) - 1))
							
							distf = math.degrees(dTheta)
							if distf <= epsilon and x0 != x1 and y0 != y1:
								m += 1
							
							if m >= n:
								categories[y0][x0] = 1
								
								break
					
					if m >= n:
						break
	
	for y0 in range(height):
		for x0 in range(width):
			if categories[y0][x0] != 0:
				continue
			
			for y1 in range(height):
				for x1 in range(width):
					if phase[y0][x0] == phase[y1][x1] and categories[y1][x1] == 1:
						dR = numpy.dot(numpy.linalg.inv(R[y0][x0]), R[y1][x1])
						
						if 0.5 * (abs(dR[0][0]) + abs(dR[1][1]) + abs(dR[2][2]) - 1) > 1:
							dTheta = math.acos(1)
						elif 0.5 * (abs(dR[0][0]) + abs(dR[1][1]) + abs(dR[2][2]) - 1) < -1:
							dTheta = math.acos(-1)
						else:
							dTheta = math.acos(0.5 * (abs(dR[0][0]) + abs(dR[1][1]) + abs(dR[2][2]) - 1))
						
						distf = math.degrees(dTheta)
						if distf <= epsilon:
							categories[y0][x0] = 2
							break
				
				if categories[y0][x0] == 2:
					break
	
	for y0 in range(height):
		for x0 in range(width):
			if categories[y0][x0] == 0:
				categories[y0][x0] = 3
	
	for y0 in range(height):
		for x0 in range(width):
			if categories[y0][x0] == 1 and clusters[y0][x0] == 0:
				k += 1
				clusters[y0][x0] = k
				complete = False
				
				while not complete:
					complete = True
					
					for y1 in range(height):#range(y0, height):
						for x1 in range(width):#range(x0, width):
							if clusters[y1][x1] == k:
								for y2 in range(height):#range(y1, height):
									for x2 in range(width):#range(x1, width):
										if phase[y1][x1] == phase[y2][x2] and categories[y2][x2] == 1 and clusters[y2][x2] == 0:
											dR = numpy.dot(numpy.linalg.inv(R[y1][x1]), R[y2][x2])
											
											if 0.5 * (abs(dR[0][0]) + abs(dR[1][1]) + abs(dR[2][2]) - 1) > 1:
												dTheta = math.acos(1)
											elif 0.5 * (abs(dR[0][0]) + abs(dR[1][1]) + abs(dR[2][2]) - 1) < -1:
												dTheta = math.acos(-1)
											else:
												dTheta = math.acos(0.5 * (abs(dR[0][0]) + abs(dR[1][1]) + abs(dR[2][2]) - 1))
											
											distf = math.degrees(dTheta)
											if distf <= epsilon:
												clusters[y2][x2] = k
												complete = False
	
	for y0 in range(height):
		for x0 in range(width):
			if categories[y0][x0] == 2:
				dmin = epsilon
				xmin = 0
				ymin = 0
				
				for y1 in range(height):
					for x1 in range(width):
						if phase[y0][x0] == phase[y1][x1] and categories[y1][x1] == 1:
							dR = numpy.dot(numpy.linalg.inv(R[y0][x0]), R[y1][x1])
							
							if 0.5 * (abs(dR[0][0]) + abs(dR[1][1]) + abs(dR[2][2]) - 1) > 1:
								dTheta = math.acos(1)
							elif 0.5 * (abs(dR[0][0]) + abs(dR[1][1]) + abs(dR[2][2]) - 1) < -1:
								dTheta = math.acos(-1)
							else:
								dTheta = math.acos(0.5 * (abs(dR[0][0]) + abs(dR[1][1]) + abs(dR[2][2]) - 1))
							
							distf = math.degrees(dTheta)
							if  distf <= dmin:
								dmin = distf
								xmin = x1
								ymin = y1
				
				clusters[y0][x0] = clusters[ymin][xmin]
	
	return k, categories, clusters

def cluster(data, n, epsilon):
	
	phase = numpy.zeros((data['height'], data['width']))
	R = numpy.zeros((data['height'], data['width'], 3, 3))
	
	for y in range(data['height']):
		for x in range(data['width']):
			phase[y][x] = data['phases'][data['data']['phase'][y][x]]['ID']
			R[y][x] = data['data']['R'][y][x]
	
	width = data['width']
	height = data['height']
	k, categories, clusters = dbscan(phase, R, width, height, n, epsilon)
	output = dict()
	output['k'] = k
	output['data'] = dict()
	output['data']['category'] = list()
	output['data']['cluster'] = list()
	
	for y in range(data['height']):
		output['data']['category'].append(list())
		output['data']['cluster'].append(list())
		
		for x in range(data['width']):
			output['data']['cluster'][y].append(int(clusters[y][x]))
			
			if categories[y][x] == 1:
				output['data']['category'][y].append('C')
			elif categories[y][x] == 2:
				output['data']['category'][y].append('B')
			elif categories[y][x] == 3:
				output['data']['category'][y].append('N')
			else:
				output['data']['category'][y].append('X')
	
	return output


def mapGND(data, phaseID=None):
	
	GND = list()
	
	for y in range(data['height']):
		GND.append(list())
		
		for x in range(data['width']):
			if data['data']['GND'][y][x] == 0:
				GND[y].append(list((data['maxGND'] - data['minGND'], 0, 0)))
			elif phaseID != None and data['phases'][data['data']['phase'][y][x]]['ID'] != phaseID:
				GND[y].append(list(((data['maxGND'] - data['minGND']) / 2, 0, 0)))
			else:
				rho = data['data']['GND'][y][x]
				GND[y].append(list((rho - data['minGND'], rho - data['minGND'], rho - data['minGND'])))
	
	return GND

def calKAM(data):
	R = copy.deepcopy(data['data']['r'])
	R.append(list())
	
	for y in range(data['height']):
		R[y].append(numpy.eye(3))
	
	for x in range(data['width'] + 1):
		R[data['height']].append(numpy.eye(3))
	
	KAM = list()
	
	for y in range(data['height']):
		KAM.append(list())
		
		for x in range(data['width']):
			if data['phases'][data['data']['phase'][y][x]]['ID'] == 0:
				KAM[y].append(0)
			else:
				xm = 1
				xp = 1
				ym = 1
				yp = 1
				S = 0
				
				if x == 0 or data['data']['phase'][y][x] != data['data']['phase'][y][x-1]:
					xm = 0
				
				if x == data['width'] - 1 or data['data']['phase'][y][x] != data['data']['phase'][y][x+1]:
					xp = 0
				
				if y == 0 or data['data']['phase'][y][x] != data['data']['phase'][y-1][x]:
					ym = 0
				
				if y == data['height'] - 1 or data['data']['phase'][y][x] != data['data']['phase'][y+1][x]:
					yp = 0
				
				S += xm * rotation_angle(misrotation_matrix(R[y][x], R[y][x - 1]))
				S += xp * rotation_angle(misrotation_matrix(R[y][x], R[y][x + 1]))
				S += ym * rotation_angle(misrotation_matrix(R[y][x], R[y - 1][x]))
				S += yp * rotation_angle(misrotation_matrix(R[y][x], R[y + 1][x]))
				
				if xm + xp + ym + yp == 0:
					KAM[y].append(0)
				else:
					KAM[y].append(math.degrees(S) / (xm + xp + ym + yp))
	
	return KAM

def mapKAM(data, phaseID=None):
	
	KAM = list()
	
	for y in range(data['height']):
		KAM.append(list())
		
		for x in range(data['width']):
			if data['data']['KAM'][y][x] == 0:
				KAM[y].append(list((data['maxKAM'], 0, 0)))
			elif phaseID != None and data['phases'][data['data']['phase'][y][x]]['ID'] != phaseID:
				KAM[y].append(list((data['maxKAM'] / 2, 0, 0)))
			else:
				angle = data['data']['KAM'][y][x]
				KAM[y].append(list((angle, angle, angle)))
	
	return KAM

def mapIPF(data, axis, phaseID=None):
	
	IPF = list()
	refV = axis.value
	
	for y in range(data['height']):
		IPF.append(list())
		
		for x in range(data['width']):
			if data['phases'][data['data']['phase'][y][x]]['ID'] == 0:
				IPF[y].append(list((0, 0, 0)))
			elif phaseID != None and data['phases'][data['data']['phase'][y][x]]['ID'] != phaseID:
				IPF[y].append(list((0.5, 0.5, 0.5)))
			else:
				euler = data['data']['euler'][y][x]
				lattice_type = BravaisLattice(data['phases'][data['data']['phase'][y][x]]['type'])
				v = numpy.dot(euler_rotation_matrix(AxisSet.ZXZ, euler), refV).tolist()
				v = reduce_vector(v, lattice_type)
				IPF[y].append(v)
	
	return IPF

def keyIPF(lattice_type, size, guides):
	
	IPF = list()
	
	for Y in range(size):
		IPF.append(list())
		
		for X in range(size):
			x, y, z = inverse_stereographic(2 * X / size - 1, 2 * Y / size - 1)
			
			if math.sqrt((2 * X / size - 1) ** 2 + (2 * Y / size - 1) ** 2) > 1:
				IPF[Y].append(list((0, 0, 0)))
			elif guides and (round(abs(x), 2) == round(abs(y), 2) or round(abs(y), 2) == round(abs(z), 2) or round(abs(z), 2) == round(abs(x), 2) or X == size / 2 or Y == size / 2):
				IPF[Y].append(list((0, 0, 0)))
			else:
				v = (x, y, z)
				v = reduce_vector(v, lattice_type)
				IPF[Y].append(v)
	
	return IPF

def calV(euler, refV):

	R = reduce_matrix(euler_rotation_matrix(AxisSet.ZXZ, euler), CrystalFamily.C)
	vx, vy, vz = numpy.dot(R, refV).tolist()
	vX, vY = forward_stereographic(vx, vy, vz)
	return (-vX, -vY)

def calSGP(data, axis):
	
	SGP = list()
	refV = axis.value
	
	for y in range(data['height']):
		SGP.append(list())
		
		for x in range(data['width']):
			if data['phases'][data['data']['phase'][y][x]]['ID'] == 0:
				SGP[y].append(list((0, 0)))
			else:
				SGP[y].append(calV(data['data']['euler'][y][x], refV))
	
	return SGP

def calCF(data, Z, E, refV):
	
	CF = list()
	critData = dict()
	
	for localID in data['phases']:
		if data['phases'][localID]['ID'] == 0:
			continue
		
		critData[localID] = channelling.loadCritData(Z, data['phases'][localID]['ID'], E)
	
	for y in range(data['height']):
		CF.append(list())
		
		for x in range(data['width']):
			if data['phases'][data['data']['phase'][y][x]]['ID'] == 0:
				CF[y].append(0)
			else:
				vX, vY = calV(data['data']['euler'][y][x], refV)
				vx, vy, vz = inverse_stereographic(vX, vY)
				vtheta = -math.degrees(math.atan(math.sqrt(vx ** 2 + vy ** 2) / vz))
				vphi = 90 - math.degrees(math.atan2(vy, vx))
				CF[y].append(channelling.fraction(vtheta, vphi, critData[data['data']['phase'][y][x]]))
	
	return CF

def mapCF(data, phaseID=None):
	
	CF = list()
	
	for y in range(data['height']):
		CF.append(list())
		
		for x in range(data['width']):
			if data['phases'][data['data']['phase'][y][x]]['ID'] == 0:
				CF[y].append(list((100, 0, 0)))
			elif phaseID != None and data['phases'][data['data']['phase'][y][x]]['ID'] != phaseID:
				CF[y].append(list((50, 0, 0)))
			else:
				fraction = data['data']['CF'][y][x]
				CF[y].append(list((fraction, fraction, fraction)))
	
	return CF

def mapSGP(data, metadata, size, plot, phaseID=None, trim=True):
	
	SGP = list()
	
	if plot == 'template':
		for Y in range(size):
			SGP.append(list())
			
			for X in range(size):
				SGP[Y].append(list((0, 0, 0)))
				
		for Y in range(size):
			for X in range(size):
				vX = math.tan(math.radians(22.5)) * X / size
				vY = math.tan(math.radians(22.5)) * Y / size
				vx, vy, vz = inverse_stereographic(vX, vY)
				
				if (abs(vx) > abs(vy) or abs(vy) > abs(vz) or abs(vz) < abs(vx)):
					SGP[Y][X] = list((1, 1, 1))
	
	if plot == 'scheme':
		for Y in range(size):
			SGP.append(list())
			
			for X in range(size):
				SGP[Y].append(list((0, 0, 0)))
		
		for Y in range(size):
			for X in range(size):
				vX = math.tan(math.radians(22.5)) * X / (size - 1)
				vY = math.tan(math.radians(22.5)) * Y / (size - 1)
				v = inverse_stereographic(vX, vY)
				lattice_type = BravaisLattice.CP
				v = reduce_vector(v, lattice_type)
				try:
					SGP[int(round((size - 1) * vY / math.tan(math.radians(22.5))))][int(round((size - 1) * vX / math.tan(math.radians(22.5))))] = list(v)
				except IndexError:
					continue
	
	if plot == 'colour':
		for Y in range(size):
			SGP.append(list())
			
			for X in range(size):
				SGP[Y].append(list((0, 0, 0)))

		for y in range(data['height']):
			for x in range(data['width']):
				if data['phases'][data['data']['phase'][y][x]]['ID'] == 0:
					continue
				
				if phaseID != None and data['phases'][data['data']['phase'][y][x]]['ID'] != phaseID:
					continue
				
				vX, vY = data['data']['SGP'][y][x]
				v = inverse_stereographic(vX, vY)
				lattice_type = BravaisLattice(data['phases'][data['data']['phase'][y][x]]['type'])
				v = reduce_vector(v, lattice_type)
				try:
					SGP[int(round((size - 1) * vY / math.tan(math.radians(22.5))))][int(round((size - 1) * vX / math.tan(math.radians(22.5))))] = list(v)
				except IndexError:
					continue
	
	if plot == 'count':
		counts = list()
		
		for Y in range(size):
			SGP.append(list())
			counts.append(list())
			
			for X in range(size):
				SGP[Y].append(list((1, 0, 0)))
				counts[Y].append(0)
		
		for y in range(data['height']):
			for x in range(data['width']):
				if data['phases'][data['data']['phase'][y][x]]['ID'] == 0:
					continue
				
				if phaseID != None and data['phases'][data['data']['phase'][y][x]]['ID'] != phaseID:
					continue
				
				vX, vY = data['data']['SGP'][y][x]
				try:
					counts[int(round((size - 1) * vY / math.tan(math.radians(22.5))))][int(round((size - 1) * vX / math.tan(math.radians(22.5))))] += 1
				except IndexError:
					continue
		
		maxCount = max(list(max(row) for row in counts))
		
		if maxCount == 0:
			return SGP
		
		for Y in range(size):
			for X in range(size):
				if counts[Y][X] != 0:
					if maxCount == 1:
						count = 1
					else:
						count = math.log10(counts[Y][X]) / math.log10(maxCount)
					SGP[Y][X] = list((count, count, count))
	
	if plot == 'PQ':
		counts = list()
		totals = list()
		
		for Y in range(size):
			SGP.append(list())
			counts.append(list())
			totals.append(list())
			
			for X in range(size):
				SGP[Y].append(list((1, 0, 0)))
				counts[Y].append(0)
				totals[Y].append(0)
		
		for y in range(data['height']):
			for x in range(data['width']):
				if data['phases'][data['data']['phase'][y][x]]['ID'] == 0:
					continue
				
				if phaseID != None and data['phases'][data['data']['phase'][y][x]]['ID'] != phaseID:
					continue
				
				vX, vY = data['data']['SGP'][y][x]
				try:
					counts[int(round((size - 1) * vY / math.tan(math.radians(22.5))))][int(round((size - 1) * vX / math.tan(math.radians(22.5))))] += 1
					totals[int(round((size - 1) * vY / math.tan(math.radians(22.5))))][int(round((size - 1) * vX / math.tan(math.radians(22.5))))] += data['data']['PQ'][y][x] / 100
				except:
					continue
		
		for Y in range(size):
			for X in range(size):
				if counts[Y][X] != 0:
					average = totals[Y][X] / counts[Y][X]
					SGP[Y][X] = list((average, average, average))
	
	if plot == 'IQ':
		counts = list()
		totals = list()
		
		for Y in range(size):
			SGP.append(list())
			counts.append(list())
			totals.append(list())
			
			for X in range(size):
				SGP[Y].append(list((1, 0, 0)))
				counts[Y].append(0)
				totals[Y].append(0)
		
		for y in range(data['height']):
			for x in range(data['width']):
				if data['phases'][data['data']['phase'][y][x]]['ID'] == 0:
					continue
				
				if phaseID != None and data['phases'][data['data']['phase'][y][x]]['ID'] != phaseID:
					continue
				
				vX, vY = data['data']['SGP'][y][x]
				try:
					counts[int(round((size - 1) * vY / math.tan(math.radians(22.5))))][int(round((size - 1) * vX / math.tan(math.radians(22.5))))] += 1
					totals[int(round((size - 1) * vY / math.tan(math.radians(22.5))))][int(round((size - 1) * vX / math.tan(math.radians(22.5))))] += data['data']['IQ'][y][x] / 100
				except IndexError:
					continue
		
		for Y in range(size):
			for X in range(size):
				if counts[Y][X] != 0:
					average = totals[Y][X] / counts[Y][X]
					SGP[Y][X] = list((average, average, average))
	
	if plot == 'KAM':
		counts = list()
		totals = list()
		
		for Y in range(size):
			SGP.append(list())
			counts.append(list())
			totals.append(list())
			
			for X in range(size):
				SGP[Y].append(list((1, 0, 0)))
				counts[Y].append(0)
				totals[Y].append(0)
		
		for y in range(data['height']):
			for x in range(data['width']):
				if data['phases'][data['data']['phase'][y][x]]['ID'] == 0:
					continue
				
				if phaseID != None and data['phases'][data['data']['phase'][y][x]]['ID'] != phaseID:
					continue
				
				vX, vY = data['data']['SGP'][y][x]
				try:
					counts[int(round((size - 1) * vY / math.tan(math.radians(22.5))))][int(round((size - 1) * vX / math.tan(math.radians(22.5))))] += 1
					totals[int(round((size - 1) * vY / math.tan(math.radians(22.5))))][int(round((size - 1) * vX / math.tan(math.radians(22.5))))] += data['data']['KAM'][y][x] / data['maxKAM']
				except IndexError:
					continue
		
		for Y in range(size):
			for X in range(size):
				if counts[Y][X] != 0:
					average = totals[Y][X] / counts[Y][X]
					SGP[Y][X] = list((average, average, average))
	
	if plot == 'GND':
		counts = list()
		totals = list()
		
		for Y in range(size):
			SGP.append(list())
			counts.append(list())
			totals.append(list())
			
			for X in range(size):
				SGP[Y].append(list((1, 0, 0)))
				counts[Y].append(0)
				totals[Y].append(0)
		
		for y in range(data['height']):
			for x in range(data['width']):
				if data['phases'][data['data']['phase'][y][x]]['ID'] == 0:
					continue
				
				if phaseID != None and data['phases'][data['data']['phase'][y][x]]['ID'] != phaseID:
					continue
				
				vX, vY = data['data']['SGP'][y][x]
				try:
					counts[int(round((size - 1) * vY / math.tan(math.radians(22.5))))][int(round((size - 1) * vX / math.tan(math.radians(22.5))))] += 1
					totals[int(round((size - 1) * vY / math.tan(math.radians(22.5))))][int(round((size - 1) * vX / math.tan(math.radians(22.5))))] += (data['data']['GND'][y][x] - data['minGND']) / (data['maxGND'] - data['minGND'])
				except IndexError:
					continue
		
		for Y in range(size):
			for X in range(size):
				if counts[Y][X] != 0:
					average = totals[Y][X] / counts[Y][X]
					SGP[Y][X] = list((average, average, average))
	
	if plot == 'CF':
		counts = list()
		totals = list()
		
		for Y in range(size):
			SGP.append(list())
			counts.append(list())
			totals.append(list())
			
			for X in range(size):
				SGP[Y].append(list((1, 0, 0)))
				counts[Y].append(0)
				totals[Y].append(0)
		
		for y in range(data['height']):
			for x in range(data['width']):
				if data['phases'][data['data']['phase'][y][x]]['ID'] == 0:
					continue
				
				if phaseID != None and data['phases'][data['data']['phase'][y][x]]['ID'] != phaseID:
					continue
				
				vX, vY = data['data']['SGP'][y][x]
				try:
					counts[int(round((size - 1) * vY / math.tan(math.radians(22.5))))][int(round((size - 1) * vX / math.tan(math.radians(22.5))))] += 1
					totals[int(round((size - 1) * vY / math.tan(math.radians(22.5))))][int(round((size - 1) * vX / math.tan(math.radians(22.5))))] += data['data']['CF'][y][x] / 100
				except IndexError:
					continue
		
		for Y in range(size):
			for X in range(size):
				if counts[Y][X] != 0:
					average = totals[Y][X] / counts[Y][X]
					SGP[Y][X] = list((average, average, average))
	
	if plot == 'phase':
		phaseIDs = sorted(list(phase['ID'] for phase in data['phases'].values() if phase['ID'] != 0))
		pID = list()
		
		for Y in range(size):
			SGP.append(list())
			pID.append(list())
			
			for X in range(size):
				SGP[Y].append(list((0, 0, 0)))
				pID[Y].append(0)
		
		for y in range(data['height']):
			for x in range(data['width']):
				if data['phases'][data['data']['phase'][y][x]]['ID'] == 0:
					continue
				
				if phaseID != None and data['phases'][data['data']['phase'][y][x]]['ID'] != phaseID:
					continue
				
				vX, vY = data['data']['SGP'][y][x]
				
				try:
					if pID[int(round((size - 1) * vY / math.tan(math.radians(22.5))))][int(round((size - 1) * vX / math.tan(math.radians(22.5))))] == 0:
						pID[int(round((size - 1) * vY / math.tan(math.radians(22.5))))][int(round((size - 1) * vX / math.tan(math.radians(22.5))))] = phaseIDs.index(data['phases'][data['data']['phase'][y][x]]['ID']) + 1
					elif pID[int(round((size - 1) * vY / math.tan(math.radians(22.5))))][int(round((size - 1) * vX / math.tan(math.radians(22.5))))] != phaseIDs.index(data['phases'][data['data']['phase'][y][x]]['ID']) + 1:
						pID[int(round((size - 1) * vY / math.tan(math.radians(22.5))))][int(round((size - 1) * vX / math.tan(math.radians(22.5))))] = -1
				except IndexError:
					continue
		
		for Y in range(size):
			for X in range(size):
				if pID[Y][X] == 0:
					SGP[Y][X] = list((0, 0, 0))
				elif pID[Y][X] == -1:
					SGP[Y][X] = list((1, 1, 1))
				else:
					SGP[Y][X] = utilities.colour_wheel(pID[Y][X] - 1, len(phaseIDs))
	
	if plot == 'cluster':
		cID = list()
		
		for Y in range(size):
			SGP.append(list())
			cID.append(list())
			
			for X in range(size):
				SGP[Y].append(list((0, 0, 0)))
				cID[Y].append(0)
		
		for y in range(data['height']):
			for x in range(data['width']):
				if data['phases'][data['data']['phase'][y][x]]['ID'] == 0:
					continue
				
				if phaseID != None and data['phases'][data['data']['phase'][y][x]]['ID'] != phaseID:
					continue
				
				vX, vY = data['data']['SGP'][y][x]
				
				try:
					if cID[int(round((size - 1) * vY / math.tan(math.radians(22.5))))][int(round((size - 1) * vX / math.tan(math.radians(22.5))))] == 0:
						cID[int(round((size - 1) * vY / math.tan(math.radians(22.5))))][int(round((size - 1) * vX / math.tan(math.radians(22.5))))] = data['data']['cID'][y][x]
					elif cID[int(round((size - 1) * vY / math.tan(math.radians(22.5))))][int(round((size - 1) * vX / math.tan(math.radians(22.5))))] != data['data']['cID'][y][x]:
						cID[int(round((size - 1) * vY / math.tan(math.radians(22.5))))][int(round((size - 1) * vX / math.tan(math.radians(22.5))))] = -1
				except IndexError:
					continue
		
		for Y in range(size):
			for X in range(size):
				if cID[Y][X] == 0:
					SGP[Y][X] = list((0, 0, 0))
				elif cID[Y][X] == -1:
					SGP[Y][X] = list((1, 1, 1))
				else:
					SGP[Y][X] = utilities.colour_wheel(cID[Y][X] - 1, metadata[data['fileref']]['k'])
	
	if trim:
		for Y in range(size):
			for X in range(size):
				vX = math.tan(math.radians(22.5)) * X / size
				vY = math.tan(math.radians(22.5)) * Y / size
				vx, vy, vz = inverse_stereographic(vX, vY)
				
				if (abs(vx) > abs(vy) or abs(vy) > abs(vz) or abs(vz) < abs(vx)):
					SGP[Y][X] = list((1, 1, 1))
	
	return SGP

def mapRGB(data, phaseID=None):
	
	RGB = list()
	
	for y in range(data['height']):
		RGB.append(list())
		
		for x in range(data['width']):
			phi1, Phi, phi2 = data['data']['euler'][y][x]
			lSym = data['phases'][data['data']['phase'][y][x]]['type'][0]
			
			if lSym == 'c':
				phi1 /= 2 * math.pi
				Phi /= math.acos(math.sqrt(3)/3)
				phi2 /= 0.5 * math.pi
			elif lSym == 'N':
				phi1 /= 2 * math.pi
				Phi /= 2 * math.pi
				phi2 /= 2 * math.pi
			else:
				print('Unrecognised symmetry!')
				phi1 /= 2 * math.pi
				Phi /= 2 * math.pi
				phi2 /= 2 * math.pi
			
			if phaseID != None and data['phases'][data['data']['phase'][y][x]]['ID'] != phaseID:
				RGB[y].append(list((0.5, 0.5, 0.5)))
			else:
				RGB[y].append((phi1, Phi, phi2))
	
	return RGB

def mapP(data, phaseID=None):
	
	phaseIDs = sorted(list(phase['ID'] for phase in data['phases'].values() if phase['ID'] != 0))
	
	P = list()
	
	for y in range(data['height']):
		P.append(list())
		
		for x in range(data['width']):
			if data['phases'][data['data']['phase'][y][x]]['ID'] == 0:
				P[y].append(list((0, 0, 0)))
			elif phaseID != None and data['phases'][data['data']['phase'][y][x]]['ID'] != phaseID:
				P[y].append(list((0.5, 0.5, 0.5)))
			else:
				P[y].append(utilities.colour_wheel(phaseIDs.index(data['phases'][data['data']['phase'][y][x]]['ID']), len(phaseIDs)))
	
	return P

def mapG(data, k, phaseID=None):
	
	G = list()
	
	for y in range(data['height']):
		G.append(list())
		
		for x in range(data['width']):
			if data['data']['cType'][y][x] == 'N':
				G[y].append(list((0, 0, 0)))
			elif phaseID != None and data['phases'][data['data']['phase'][y][x]]['ID'] != phaseID:
				G[y].append(list((0.5, 0.5, 0.5)))
			else:
				G[y].append(utilities.colour_wheel(data['data']['cID'][y][x] - 1, k))
	
	return G

def keyG(k):
	
	G = list()
	width = min(50 * k, 250)
	height = 50 * (int(math.floor(k / 5)) + 1)
	
	for y in range(height):
		G.append(list())
		
		for x in range(width):
			i = 5 * int(math.floor(y / 50)) + int(math.floor(x / 50))
			
			if i >= k:
				G[y].append(list((0, 0, 0)))
			else:
				G[y].append(utilities.colour_wheel(i, k))
	
	return G

def crunch(data):
	
	if data['width'] % 2 != 0 or data['height'] % 2 != 0:
		input('Cannot crunch data due to odd valued size.')
		sys.exit()
	
	output = dict()
	output['fileref'] = data['fileref']
	output['phases'] = data['phases']
	output['width'] = data['width'] // 2
	output['height'] = data['height'] // 2
	output['data'] = dict()
	output['data']['phase'] = list()
	output['data']['euler'] = list()
	output['data']['IQ'] = list()
	output['data']['PQ'] = list()
	
	for y in range(output['height']):
		output['data']['phase'].append(list())
		output['data']['euler'].append(list())
		output['data']['IQ'].append(list())
		output['data']['PQ'].append(list())
		
		for x in range(output['width']):
			pixels = list(((2 * x, 2 * y), (2 * x + 1, 2 * y), (2 * x, 2 * y + 1), (2 * x + 1, 2 * y + 1)))
			phases = set(data['data']['phase'][pixel[1]][pixel[0]] for pixel in pixels if data['phases'][data['data']['phase'][pixel[1]][pixel[0]]]['ID'] != 0)
			
			if len(phases) != 1:
				output['data']['phase'][y].append(0)
				output['data']['euler'][y].append(list((0.0, 0.0, 0.0)))
				output['data']['IQ'][y].append(0.0)
				output['data']['PQ'][y].append(0.0)
				continue
			
			phase = list(phases)[0]
			count = 0
			R = numpy.zeros((3, 3))
			IQ = 0
			PQ = 0
			
			for pixel in pixels:
				if data['phases'][data['data']['phase'][pixel[1]][pixel[0]]]['ID'] != 0:
					count += 1
					R += reduce_matrix(euler_rotation_matrix(AxisSet.ZXZ, data['data']['euler'][pixel[1]][pixel[0]]), CrystalFamily.C)
					IQ += data['data']['IQ'][pixel[1]][pixel[0]]
					PQ += data['data']['PQ'][pixel[1]][pixel[0]]
			
			U, S, VT = numpy.linalg.svd(R / count)
			R = numpy.dot(U, VT)
			euler = euler_angles(reduce_matrix(R, CrystalFamily.C), AxisSet.ZXZ)
			IQ = IQ / count
			PQ = PQ / 4
			output['data']['phase'][y].append(phase)
			output['data']['euler'][y].append(euler)
			output['data']['IQ'][y].append(IQ)
			output['data']['PQ'][y].append(PQ)
	
	return output

def crunches(data, n):
	
	if n <= 0:
		return data
	else:
		return crunch(crunches(data, n - 1))

def misorientation():
	
	euler1 = list(math.radians(float(angle.strip())) for angle in input('Enter Bunge-Euler angles for first point, separated by commas (deg): ').split(','))
	euler2 = list(math.radians(float(angle.strip())) for angle in input('Enter Bunge-Euler angles for second point, separated by commas (deg): ').split(','))
	R1 = reduce_matrix(euler_rotation_matrix(AxisSet.ZXZ, euler1), CrystalFamily.C)
	R2 = reduce_matrix(euler_rotation_matrix(AxisSet.ZXZ, euler2), CrystalFamily.C)
	dR = misrotation_matrix(R1, R2)
	theta = utilities.format_sig_figs_or_int(math.degrees(rotation_angle(dR)), 6)
	print('Misorientation is: ' + theta + ' deg')
	input('Press ENTER to close: ')
	

def analyse(path):
	
	filepaths = utilities.get_file_paths(directory_path=utilities.get_directory_path(path + '/data'), recursive=True, extension='csv')
	metadata = fileloader.getNyeMetadata(path + '/metadata.csv')
	cNum = -1
	aType = 'q'
	print()
	
	if input('Reduce map resolution? (Y/N): ').lower() == 'y':
		cNum = int(input('Enter resolution reduction factor (power of 2): '))
	
	print()
	
	if input('Perform defect density analysis? (Y/N): ').lower() == 'y':
		aType += 'g'
	
	print()
	
	if input('Perform channelling fraction analysis? (Y/N): ').lower() == 'y':
		aType += 'c'
		Z = int(input('Enter beam species atomic number: '))
		E = float(input('Enter beam energy (eV): '))
		tau = float(input('Enter beam tilt (deg): '))
		refV = numpy.array((0, -math.sin(math.radians(tau)), math.cos(math.radians(tau))))
	
	print()
	
	if input('Perform orientation cluster analysis? (Y/N): ').lower() == 'y':
		aType += 'd'
		n = int(input('Enter core point neighbour threshold: '))
		epsilon = float(input('Enter point neighbourhood radius (deg): '))
		print()
		cuda.api.detect()
	
	print()
	newLine = False
	
	for filepath in filepaths:
		data = fileloader.getNyeData(filepath)
		fileref = data['fileref']
		
		if cNum >= 0:
			fileref += '-' + str(cNum)
		
		if fileref not in metadata:
			newLine = True
			metadata[fileref] = dict()
			metadata[fileref]['aType'] = 'n'
			metadata[fileref]['dx'] = float(input('Input pixel width (μm) for p' + data['fileref'] + ': ')) * 10 ** -6
			
			if cNum >= 0:
				metadata[fileref]['dx'] *= 2 ** cNum
		
		metadata[fileref]['Z'] = 0
		metadata[fileref]['E'] = 0
		metadata[fileref]['tau'] = 0
		metadata[fileref]['n'] = 0
		metadata[fileref]['epsilon'] = 0
		metadata[fileref]['k'] = 0
	
	if newLine:
		print()
	
	for filepath in filepaths:
		data = fileloader.getNyeData(filepath)
		fileref = data['fileref']
		print('Making analysis for p' + data['fileref'] + '.')
		startTime = datetime.datetime.now()
		
		if cNum >= 0:
			fileref += '-' + str(cNum)
			data = crunches(data, cNum)

		data['data']['R'] = calR(data)
		data['data']['SGP'] = calSGP(data, Axis.Z)
		
		if 'g' in aType:
			data['data']['KAM'] = calKAM(data)
			data['data']['GND'] = calGND(data, metadata[fileref]['dx'])
		
		if 'c' in aType:
			data['data']['CF'] = calCF(data, Z, E, refV)
			metadata[fileref]['Z'] = Z
			metadata[fileref]['E'] = E
			metadata[fileref]['tau'] = tau
		
		if 'd' in aType:
			cData = cluster(data, n, epsilon)
			data['data']['cType'] = cData['data']['category']
			data['data']['cID'] = cData['data']['cluster']
			metadata[fileref]['n'] = n
			metadata[fileref]['epsilon'] = epsilon
			metadata[fileref]['k'] = cData['k']
		
		with open(path + '/analyses/q' + fileref + '.csv', 'w') as output:
			output.write('Phases:\n')
			
			for localID in data['phases']:
				output.write(str(localID) + ',' + data['phases'][localID]['name'] + ',' + str(data['phases'][localID]['ID']) + '\n')
			
			output.write('Map Size:\n')
			output.write('X,' + str(data['width']) + '\n')
			output.write('Y,' + str(data['height']) + '\n')
			output.write('Data:\n')
			output.write('X,Y,Phase,Euler1,Euler2,Euler3,Index Quality,Pattern Quality,IPF x-coordinate,IPF y-coordinate')
			
			if 'g' in aType:
				output.write(',Kernel Average Misorientation,GND Density')
			
			if 'c' in aType:
				output.write(',Channelling Fraction')
			
			if 'd' in aType:
				output.write(',Point Category,Point Cluster')
			
			output.write('\n')
			
			for y in range(data['height']):
				for x in range(data['width']):
					output.write(str(x) + ',')
					output.write(str(y) + ',')
					output.write(str(data['data']['phase'][y][x]) + ',')
					output.write(str(math.degrees(data['data']['euler'][y][x][0])) + ',')
					output.write(str(math.degrees(data['data']['euler'][y][x][1])) + ',')
					output.write(str(math.degrees(data['data']['euler'][y][x][2])) + ',')
					output.write(str(data['data']['IQ'][y][x]) + ',')
					output.write(str(data['data']['PQ'][y][x]) + ',')
					output.write(str(data['data']['SGP'][y][x][0]) + ',')
					output.write(str(data['data']['SGP'][y][x][1]))
					
					if 'g' in aType:
						output.write(',' + str(data['data']['KAM'][y][x]))
						output.write(',' + str(data['data']['GND'][y][x]))
					
					if 'c' in aType:
						output.write(',' + str(data['data']['CF'][y][x]))
					
					if 'd' in aType:
						output.write(',' + str(data['data']['cType'][y][x]))
						output.write(',' + str(data['data']['cID'][y][x]))
					
					output.write('\n')
		
		metadata[fileref]['aType'] = aType
		
		with open(path + '/metadata.csv', 'w') as output:
			output.write('Map ID,Analysis type,Pixel width (um),Channelling species,Channelling energy (eV),Channelling tilt (deg),Core point threshold,Point neighbourhood radius (deg),Clusters\n')
			
			for fileref in sorted(metadata):
				output.write(fileref + ',')
				output.write(metadata[fileref]['aType'] + ',')
				output.write(str(metadata[fileref]['dx'] * 10 ** 6) + ',')
				output.write(str(metadata[fileref]['Z']) + ',')
				output.write(str(metadata[fileref]['E']) + ',')
				output.write(str(metadata[fileref]['tau']) + ',')
				output.write(str(metadata[fileref]['n']) + ',')
				output.write(str(metadata[fileref]['epsilon']) + ',')
				output.write(str(metadata[fileref]['k']) + '\n')
		
		print('Analysis completed in ' + utilities.format_time_interval(int(round((datetime.datetime.now() - startTime).total_seconds()))) + '.')
	
	print()
	print('All analyses complete.')
	input('Press ENTER to close: ')

def summarise(path):
	
	filepaths = utilities.get_file_paths(directory_path=utilities.get_directory_path(path + '/analyses'),
                                         recursive=True, extension='csv')
	metadata = fileloader.getNyeMetadata(path + '/metadata.csv')
	print()
	
	for filepath in filepaths:
		data = fileloader.getNyeAnalysis(filepath)
		print('Making summaries for q' + data['fileref'] + '.')
		data['data']['x'] = dict()
		data['data']['y'] = dict()
		data['data']['x']['IQ'] = list()
		data['data']['x']['PQ'] = list()
		data['data']['y']['IQ'] = list()
		data['data']['y']['PQ'] = list()
		
		if 'd' in metadata[data['fileref']]['aType']:
			data['data']['k'] = dict()
			data['data']['k']['phase'] = list()
			data['data']['k']['euler'] = list()
			data['data']['k']['R'] = list()
			data['data']['k']['SGP'] = list()
			data['data']['k']['IQ'] = list()
			data['data']['k']['PQ'] = list()
			kCounts = list()
		
		if 'g' in metadata[data['fileref']]['aType']:
			data['data']['x']['KAM'] = list()
			data['data']['x']['GND'] = list()
			xGND = list()
			data['data']['y']['KAM'] = list()
			data['data']['y']['GND'] = list()
			yGND = list()
			
			if 'd' in metadata[data['fileref']]['aType']:
				data['data']['k']['KAM'] = list()
				data['data']['k']['GND'] = list()
				kGND = list()
		
		if 'c' in metadata[data['fileref']]['aType']:
			data['data']['x']['CF'] = list()
			data['data']['y']['CF'] = list()
			
			if 'd' in metadata[data['fileref']]['aType']:
				data['data']['k']['CF'] = list()
		
		for x in range(data['width']):
			data['data']['x']['IQ'].append(0)
			data['data']['x']['PQ'].append(0)
			
			if 'g' in metadata[data['fileref']]['aType']:
				data['data']['x']['KAM'].append(0)
				xGND.append(0)
			
			if 'c' in metadata[data['fileref']]['aType']:
				data['data']['x']['CF'].append(0)
		
		for y in range(data['height']):
			data['data']['y']['IQ'].append(0)
			data['data']['y']['PQ'].append(0)
			
			if 'g' in metadata[data['fileref']]['aType']:
				data['data']['y']['KAM'].append(0)
				yGND.append(0)
			
			if 'c' in metadata[data['fileref']]['aType']:
				data['data']['y']['CF'].append(0)
		
		if 'd' in metadata[data['fileref']]['aType']:
			for k in range(metadata[data['fileref']]['k'] + 1):
				data['data']['k']['phase'].append(0)
				data['data']['k']['euler'].append(list((0, 0, 0)))
				data['data']['k']['R'].append(numpy.zeros((3, 3)))
				data['data']['k']['SGP'].append(list((0, 0)))
				data['data']['k']['IQ'].append(0)
				data['data']['k']['PQ'].append(0)
				kCounts.append(0)
				
				if 'g' in metadata[data['fileref']]['aType']:
					data['data']['k']['KAM'].append(0)
					kGND.append(0)
				
				if 'c' in metadata[data['fileref']]['aType']:
					data['data']['k']['CF'].append(0)
		
		for x in range(data['width']):
			for y in range(data['height']):
				data['data']['x']['IQ'][x] += data['data']['IQ'][y][x] / data['height']
				data['data']['x']['PQ'][x] += data['data']['PQ'][y][x] / data['height']
				data['data']['y']['IQ'][y] += data['data']['IQ'][y][x] / data['width']
				data['data']['y']['PQ'][y] += data['data']['PQ'][y][x] / data['width']
				
				if 'g' in metadata[data['fileref']]['aType']:
					data['data']['x']['KAM'][x] += data['data']['KAM'][y][x] / data['height']
					xGND[x] += 10 ** data['data']['GND'][y][x] / data['height']
					data['data']['y']['KAM'][y] += data['data']['KAM'][y][x] / data['width']
					yGND[y] += 10 ** data['data']['GND'][y][x] / data['width']
				
				if 'c' in metadata[data['fileref']]['aType']:
					data['data']['x']['CF'][x] += data['data']['CF'][y][x] / data['height']
					data['data']['y']['CF'][y] += data['data']['CF'][y][x] / data['width']
				
				if 'd' in metadata[data['fileref']]['aType']:
					kCounts[data['data']['cID'][y][x]] += 1
					data['data']['k']['phase'][data['data']['cID'][y][x]] = data['data']['phase'][y][x]
					data['data']['k']['R'][data['data']['cID'][y][x]] += reduce_matrix(euler_rotation_matrix(AxisSet.ZXZ, data['data']['euler'][y][x]), CrystalFamily.C)
					data['data']['k']['IQ'][data['data']['cID'][y][x]] += data['data']['IQ'][y][x]
					data['data']['k']['PQ'][data['data']['cID'][y][x]] += data['data']['PQ'][y][x]
					
					if 'g' in metadata[data['fileref']]['aType']:
						data['data']['k']['KAM'][data['data']['cID'][y][x]] += data['data']['KAM'][y][x]
						kGND[data['data']['cID'][y][x]] += 10 ** data['data']['GND'][y][x]
					
					if 'c' in metadata[data['fileref']]['aType']:
						data['data']['k']['CF'][data['data']['cID'][y][x]] += data['data']['CF'][y][x]
		
		if 'd' in metadata[data['fileref']]['aType']:
			data['data']['k']['phase'][0] = 0
            
			for k in range(metadata[data['fileref']]['k'] + 1):
				if k != 0:
					U, S, VT = numpy.linalg.svd(data['data']['k']['R'][k] / kCounts[k])
					data['data']['k']['R'][k] = numpy.dot(U, VT)
					phi1, Phi, phi2 = euler_angles(data['data']['k']['R'][k], AxisSet.ZXZ)
					data['data']['k']['euler'][k] = list((phi1, Phi, phi2))
					vX, vY = calV(data['data']['k']['euler'][k], Axis.Z.value)
					data['data']['k']['SGP'][k] = list((vX, vY))
				
				data['data']['k']['IQ'][k] /= kCounts[k]
				data['data']['k']['PQ'][k] /= kCounts[k]
				
				if 'g' in metadata[data['fileref']]['aType']:
					data['data']['k']['KAM'][k] /= kCounts[k]
					kGND[k] /= kCounts[k]
				
				if 'c' in metadata[data['fileref']]['aType']:
					data['data']['k']['CF'][k] /= kCounts[k]
		
		if 'g' in metadata[data['fileref']]['aType']:
			for x in range(data['width']):
				data['data']['x']['GND'].append(math.log10(xGND[x]))
			
			for y in range(data['height']):
				data['data']['y']['GND'].append(math.log10(yGND[y]))
			
			if 'd' in metadata[data['fileref']]['aType']:
				for k in range(metadata[data['fileref']]['k'] + 1):
					data['data']['k']['GND'].append(math.log10(kGND[k]))
		
		if 'd' in metadata[data['fileref']]['aType']:
			materials = fileloader.get_materials()
			variants = fileloader.getVariantList()
			twins = fileloader.getTwinList()
			matches = list()
			
			for k1 in range(1, metadata[data['fileref']]['k'] + 1):
				for k2 in range(1, metadata[data['fileref']]['k'] + 1):
					if k1 == k2:
						continue
					
					for variant in variants:
						if variants[variant]['lTypes'][0] == data['phases'][data['data']['k']['phase'][k1]]['type'] and variants[variant]['lTypes'][1] == data['phases'][data['data']['k']['phase'][k2]]['type']:
							match = dict()
							match['variant'] = variant
							match['k1'] = k1
							match['k2'] = k2
							params = list((materials[data['phases'][data['data']['k']['phase'][k1]]['ID']].lattice_constants, materials[data['phases'][data['data']['k']['phase'][k2]]['ID']].lattice_constants))
							R1 = data['data']['k']['R'][k1]
							R2 = data['data']['k']['R'][k2]
							polarity = sorted(list(set(list(itertools.permutations(list((1, 1, 1, 1, -1, -1, -1, -1)), 4)))), reverse=True)
							theta = 2 * math.pi
							
							for i in range(16):
								vectors = list((list((list(polarity[i][0] * hkl for hkl in variants[variant]['vectors'][0][0]), list(polarity[i][1] * hkl for hkl in variants[variant]['vectors'][0][1]))), list((list(polarity[i][2] * hkl for hkl in variants[variant]['vectors'][1][0]), list(polarity[i][3] * hkl for hkl in variants[variant]['vectors'][1][1])))))
								J = orientation.get_relationship_matrix(vectors[0][0], vectors[0][1], vectors[1][0], vectors[1][1], params[0], params[1])
								s = math.sqrt(params[0][0] ** 2 + params[0][1] ** 2 + params[0][2] ** 2) / math.sqrt(params[1][0] ** 2 + params[1][1] ** 2 + params[1][2] ** 2)
								RF = numpy.dot(J / s, R1)
								dR = misrotation_matrix(RF, R2)
								theta = min(rotation_angle(dR), theta)
							
							match['dTheta'] = theta
							match['cosine'] = math.cos(theta)
							matches.append(match)
					
					for variant in twins:
						if data['phases'][data['data']['k']['phase'][k1]]['type'] == data['phases'][data['data']['k']['phase'][k2]]['type'] and k1 < k2:
							match = dict()
							match['variant'] = variant
							match['k1'] = k1
							match['k2'] = k2
							R1 = data['data']['k']['R'][k1]
							R2 = data['data']['k']['R'][k2]
							theta = 2 * math.pi
							family = orientation.get_plane_family(twins[variant])
							
							for plane in family:
								J = orientation.get_twin_matrix(plane)
								RF = numpy.dot(J, R1)
								dR = misrotation_matrix(RF, R2)
								theta = min(rotation_angle(dR), theta)
							
							match['dTheta'] = theta
							match['cosine'] = math.cos(theta)
							matches.append(match)
		
		with open(path + '/summaries/x' + data['fileref'] + '.csv', 'w') as output:
			output.write('Phases:\n')
			
			for localID in data['phases']:
				output.write(str(localID) + ',' + data['phases'][localID]['name'] + ',' + str(data['phases'][localID]['ID']) + '\n')
			
			output.write('Map Size:\n')
			output.write('X,' + str(data['width']) + '\n')
			output.write('Y,' + str(data['height']) + '\n')
			output.write('Data:\n')
			output.write('X,Index Quality,Pattern Quality')
			
			if 'g' in metadata[data['fileref']]['aType']:
				output.write(',Kernel Average Misorientation,GND Density')
			
			if 'c' in metadata[data['fileref']]['aType']:
				output.write(',Channelling Fraction')
			
			output.write('\n')
			
			for x in range(data['width']):
				output.write(str(x) + ',')
				output.write(str(data['data']['x']['IQ'][x]) + ',')
				output.write(str(data['data']['x']['PQ'][x]))
				
				if 'g' in metadata[data['fileref']]['aType']:
					output.write(',' + str(data['data']['x']['KAM'][x]))
					output.write(',' + str(data['data']['x']['GND'][x]))
				
				if 'c' in metadata[data['fileref']]['aType']:
					output.write(',' + str(data['data']['x']['CF'][x]))
				
				output.write('\n')
		
		with open(path + '/summaries/y' + data['fileref'] + '.csv', 'w') as output:
			output.write('Phases:\n')
			
			for localID in data['phases']:
				output.write(str(localID) + ',' + data['phases'][localID]['name'] + ',' + str(data['phases'][localID]['ID']) + '\n')
			
			output.write('Map Size:\n')
			output.write('X,' + str(data['width']) + '\n')
			output.write('Y,' + str(data['height']) + '\n')
			output.write('Data:\n')
			output.write('Y,Index Quality,Pattern Quality')
			
			if 'g' in metadata[data['fileref']]['aType']:
				output.write(',Kernel Average Misorientation,GND Density')
			
			if 'c' in metadata[data['fileref']]['aType']:
				output.write(',Channelling Fraction')
			
			output.write('\n')
			
			for y in range(data['height']):
				output.write(str(y) + ',')
				output.write(str(data['data']['y']['IQ'][y]) + ',')
				output.write(str(data['data']['y']['PQ'][y]))
				
				if 'g' in metadata[data['fileref']]['aType']:
					output.write(',' + str(data['data']['y']['KAM'][y]))
					output.write(',' + str(data['data']['y']['GND'][y]))
				
				if 'c' in metadata[data['fileref']]['aType']:
					output.write(',' + str(data['data']['y']['CF'][y]))
				
				output.write('\n')
		
		if 'd' in metadata[data['fileref']]['aType']:
			with open(path + '/summaries/k' + data['fileref'] + '.csv', 'w') as output:
				output.write('Phases:\n')
				
				for localID in data['phases']:
					output.write(str(localID) + ',' + data['phases'][localID]['name'] + ',' + str(data['phases'][localID]['ID']) + '\n')
				
				output.write('Map Size:\n')
				output.write('X,' + str(data['width']) + '\n')
				output.write('Y,' + str(data['height']) + '\n')
				output.write('Data:\n')
				output.write('K,Cluster Size,Phase,Euler1,Euler2,Euler3,Index Quality,Pattern Quality,IPF x-coordinate,IPF y-coordinate')
				
				if 'g' in metadata[data['fileref']]['aType']:
					output.write(',Kernel Average Misorientation,GND Density')
				
				if 'c' in metadata[data['fileref']]['aType']:
					output.write(',Channelling Fraction')
				
				output.write('\n')
				
				for k in range(metadata[data['fileref']]['k'] + 1):
					output.write(str(k) + ',')
					output.write(str(kCounts[k]) + ',')
					output.write(str(data['data']['k']['phase'][k]) + ',')
					output.write(str(math.degrees(data['data']['k']['euler'][k][0])) + ',')
					output.write(str(math.degrees(data['data']['k']['euler'][k][1])) + ',')
					output.write(str(math.degrees(data['data']['k']['euler'][k][2])) + ',')
					output.write(str(data['data']['k']['IQ'][k]) + ',')
					output.write(str(data['data']['k']['PQ'][k]) + ',')
					output.write(str(data['data']['k']['SGP'][k][0]) + ',')
					output.write(str(data['data']['k']['SGP'][k][1]))
					
					if 'g' in metadata[data['fileref']]['aType']:
						output.write(',' + str(data['data']['k']['KAM'][k]))
						output.write(',' + str(data['data']['k']['GND'][k]))
					
					if 'c' in metadata[data['fileref']]['aType']:
						output.write(',' + str(data['data']['k']['CF'][k]))
					
					output.write('\n')
				
				output.write('Orientation Relationships:\n')
				output.write('Variant,K1,K2,Misorientation,Projection\n')
				
				for match in sorted(matches, key=lambda item: math.degrees(item['dTheta'])):
					output.write(match['variant'] + ',')
					output.write(str(match['k1']) + ',')
					output.write(str(match['k2']) + ',')
					output.write(str(math.degrees(match['dTheta'])) + ',')
					output.write(str(match['cosine']) + '\n')
	
	print()
	print('All summaries complete.')
	input('Press ENTER to close: ')

def makeMaps(path, size=None):
	
	filepaths = utilities.get_file_paths(directory_path=utilities.get_directory_path(path + '/analyses'),
                                         recursive=True, extension='csv')
	metadata = fileloader.getNyeMetadata(path + '/metadata.csv')
	sizeOverride = size
	print()
	print('Processing data.')
	
	for filepath in filepaths:
		data = fileloader.getNyeAnalysis(filepath)
		
		minKAM = math.inf
		maxKAM = 0
		minGND = math.inf
		maxGND = 0
			
		if 'g' in metadata[data['fileref']]['aType']:
			KAMs = sorted(sum(list(list(data['data']['KAM'][y][x] for x in range(data['width']) if data['data']['KAM'][y][x] != 0) for y in range(data['height'])), list()))
			GNDs = sorted(sum(list(list(data['data']['GND'][y][x] for x in range(data['width']) if data['data']['GND'][y][x] != 0) for y in range(data['height'])), list()))
			
			if len(KAMs) != 0:
				minKAM = min(minKAM, KAMs[0])
				maxKAM = min(maxKAM, KAMs[-1])
			
			if len(GNDs) != 0:
				minGND = min(minGND, GNDs[0])
				maxGND = max(maxGND, GNDs[-1])
		
		if minKAM == math.inf or maxKAM == 0:
			minKAM = 0
			maxKAM = 1
		
		if minGND == math.inf or maxGND == 0:
			minGND = 0
			maxGND = 1
		
		if 'g' in metadata[data['fileref']]['aType']:
			data['minKAM'] = minKAM
			data['maxKAM'] = maxKAM
			data['minGND'] = minGND
			data['maxGND'] = maxGND
		
		if 'd' in metadata[data['fileref']]['aType']:
			data['keyWidth'] = min(50 * metadata[data['fileref']]['k'], 250)
			data['keyHeight'] = 50 * (int(math.floor(metadata[data['fileref']]['k'] / 5)) + 1)
		
		if sizeOverride == None:
			size = data['width']
		else:
			size = sizeOverride
		
		print('Making maps for q' + data['fileref'] + '.')
		
		try:
			os.mkdir(path + '/maps/' + data['fileref'])
		except FileExistsError:
			pass
		
		utilities.make_image(mapP(data), data['width'], data['height'], 1, 'RGB').save(path + '/maps/' + data['fileref'] + '/P.png')
		utilities.make_image(data['data']['PQ'], data['width'], data['height'], 100, 'L').save(path + '/maps/' + data['fileref'] + '/PQ.png')
		utilities.make_image(data['data']['IQ'], data['width'], data['height'], 100, 'L').save(path + '/maps/' + data['fileref'] + '/IQ.png')
		utilities.make_image(mapRGB(data), data['width'], data['height'], 1, 'RGB').save(path + '/maps/' + data['fileref'] + '/RGB.png')
		utilities.make_image(mapIPF(data, Axis.X), data['width'], data['height'], 1, 'RGB').save(path + '/maps/' + data['fileref'] + '/OX.png')
		utilities.make_image(mapIPF(data, Axis.Y), data['width'], data['height'], 1, 'RGB').save(path + '/maps/' + data['fileref'] + '/OY.png')
		utilities.make_image(mapIPF(data, Axis.Z), data['width'], data['height'], 1, 'RGB').save(path + '/maps/' + data['fileref'] + '/OZ.png')
		# utilities.makeImage(mapIPF(data, 'z', phaseID=60696), data['width'], data['height'], 1, 'RGB').save(path + '/maps/' + data['fileref'] + '/OZ[60696].png') # temporary
		
		if 'g' in metadata[data['fileref']]['aType']:
			utilities.make_image(mapKAM(data), data['width'], data['height'], maxKAM, 'RGB').save(path + '/maps/' + data['fileref'] + '/KAM.png')
			utilities.make_image(mapGND(data), data['width'], data['height'], maxGND - minGND, 'RGB').save(path + '/maps/' + data['fileref'] + '/GND.png')
		
		if 'c' in metadata[data['fileref']]['aType']:
			utilities.make_image(mapCF(data), data['width'], data['height'], 100, 'RGB').save(path + '/maps/' + data['fileref'] + '/CF.png')
		
		if 'd' in metadata[data['fileref']]['aType']:
			utilities.make_image(mapG(data, metadata[data['fileref']]['k']), data['width'], data['height'], 1, 'RGB').save(path + '/maps/' + data['fileref'] + '/C.png')
			# utilities.makeImage(mapG(data, metadata[data['fileref']]['k'], phaseID=60696), data['width'], data['height'], 1, 'RGB').save(path + '/maps/' + data['fileref'] + '/C[60696].png') # temporary
			utilities.make_image(keyG(metadata[data['fileref']]['k']), data['keyWidth'], data['keyHeight'], 1, 'RGB').save(path + '/maps/' + data['fileref'] + '/C-key.png')
		
		utilities.make_image(mapSGP(data, metadata, size, 'colour'), size, size, 1, 'RGB').save(path + '/maps/' + data['fileref'] + '/IPF.png')
		utilities.make_image(mapSGP(data, metadata, size, 'count'), size, size, 1, 'RGB').save(path + '/maps/' + data['fileref'] + '/IPF-Q.png')
		utilities.make_image(mapSGP(data, metadata, size, 'phase'), size, size, 1, 'RGB').save(path + '/maps/' + data['fileref'] + '/IPF-P.png')
		utilities.make_image(mapSGP(data, metadata, size, 'PQ'), size, size, 1, 'RGB').save(path + '/maps/' + data['fileref'] + '/IPF-PQ.png')
		utilities.make_image(mapSGP(data, metadata, size, 'IQ'), size, size, 1, 'RGB').save(path + '/maps/' + data['fileref'] + '/IPF-IQ.png')
		
		if 'g' in metadata[data['fileref']]['aType']:
			utilities.make_image(mapSGP(data, metadata, size, 'KAM'), size, size, 1, 'RGB').save(path + '/maps/' + data['fileref'] + '/IPF-KAM.png')
			utilities.make_image(mapSGP(data, metadata, size, 'GND'), size, size, 1, 'RGB').save(path + '/maps/' + data['fileref'] + '/IPF-GND.png')
		
		if 'c' in metadata[data['fileref']]['aType']:
			utilities.make_image(mapSGP(data, metadata, size, 'CF'), size, size, 1, 'RGB').save(path + '/maps/' + data['fileref'] + '/IPF-CF.png')
		
		if 'd' in metadata[data['fileref']]['aType']:
			utilities.make_image(mapSGP(data, metadata, size, 'cluster'), size, size, 1, 'RGB').save(path + '/maps/' + data['fileref'] + '/IPF-C.png')
		
		for localID in data['phases']:
			if data['phases'][localID]['ID'] == 0:
				continue
			else:
				phaseID = data['phases'][localID]['ID']
				utilities.make_image(mapSGP(data, metadata, size, 'colour', phaseID=phaseID), size, size, 1, 'RGB').save(path + '/maps/' + data['fileref'] + '/IPF[' + str(phaseID) + '].png')
				utilities.make_image(mapSGP(data, metadata, size, 'count', phaseID=phaseID), size, size, 1, 'RGB').save(path + '/maps/' + data['fileref'] + '/IPF-Q[' + str(phaseID) + '].png')
				utilities.make_image(mapSGP(data, metadata, size, 'phase', phaseID=phaseID), size, size, 1, 'RGB').save(path + '/maps/' + data['fileref'] + '/IPF-P[' + str(phaseID) + '].png')
				utilities.make_image(mapSGP(data, metadata, size, 'PQ', phaseID=phaseID), size, size, 1, 'RGB').save(path + '/maps/' + data['fileref'] + '/IPF-PQ[' + str(phaseID) + '].png')
				utilities.make_image(mapSGP(data, metadata, size, 'IQ', phaseID=phaseID), size, size, 1, 'RGB').save(path + '/maps/' + data['fileref'] + '/IPF-IQ[' + str(phaseID) + '].png')
				
				if 'g' in metadata[data['fileref']]['aType']:
					utilities.make_image(mapSGP(data, metadata, size, 'KAM', phaseID=phaseID), size, size, 1, 'RGB').save(path + '/maps/' + data['fileref'] + '/IPF-KAM[' + str(phaseID) + '].png')
					utilities.make_image(mapSGP(data, metadata, size, 'GND', phaseID=phaseID), size, size, 1, 'RGB').save(path + '/maps/' + data['fileref'] + '/IPF-GND[' + str(phaseID) + '].png')
				
				if 'c' in metadata[data['fileref']]['aType']:
					utilities.make_image(mapSGP(data, metadata, size, 'CF', phaseID=phaseID), size, size, 1, 'RGB').save(path + '/maps/' + data['fileref'] + '/IPF-CF[' + str(phaseID) + '].png')
				
				if 'd' in metadata[data['fileref']]['aType']:
					utilities.make_image(mapSGP(data, metadata, size, 'cluster', phaseID=phaseID), size, size, 1, 'RGB').save(path + '/maps/' + data['fileref'] + '/IPF-C[' + str(phaseID) + '].png')
	
	print()
	print('All maps complete.')
	input('Press ENTER to close: ')

# def makeComparison(path, border=10, schemes=None):
#
# 	data1 = fileloader.getNyeAnalysis(
# 		utilities.get_file_path(directory_path=utilities.get_directory_path(path + '/analyses'), recursive=True,
# 								extension='csv', prompt='Select source data for first map set:'))
# 	print()
# 	data2 = fileloader.getNyeAnalysis(
# 		utilities.get_file_path(directory_path=utilities.get_directory_path(path + '/analyses'), recursive=True,
# 								extension='csv', prompt='Select source data for second map set:'))
# 	print()
#
# 	if schemes == None:
# 		schemes = ''
# 		print('Available map types:')
# 		print(' - ID: P,       Description: Phase map')
# 		print(' - ID: PQ,      Description: Pattern quality map')
# 		print(' - ID: IQ,      Description: Index quality map')
# 		print(' - ID: RGB,     Description: Euler angle map')
# 		print(' - ID: OX,      Description: Orientation map (x-axis)')
# 		print(' - ID: OY,      Description: Orientation map (y-axis)')
# 		print(' - ID: OZ,      Description: Orientation map (z-axis)')
# 		print(' - ID: KAM,     Description: Kernel average misorientation map')
# 		print(' - ID: GND,     Description: Geometrically necessary dislocation density map')
# 		print(' - ID: CF,      Description: Channelling fraction map')
# 		print(' - ID: IPF,     Description: Inverse pole figure (colour key)')
# 		print(' - ID: IPF-C,   Description: Inverse pole figure (pixel count density)')
# 		print(' - ID: IPF-PQ,  Description: Inverse pole figure (average PQ)')
# 		print(' - ID: IPF-IQ,  Description: Inverse pole figure (average IQ)')
# 		print(' - ID: IPF-KAM, Description: Inverse pole figure (average KAM)')
# 		print(' - ID: IPF-GND, Description: Inverse pole figure (average GND density)')
# 		print(' - ID: IPF-CF,  Description: Inverse pole figure (average channelling fraction)')
#
# 		while True:
# 			if schemes == '':
# 				schemes += input('Enter ID for first map: ')
# 			else:
# 				item = input('Enter ID for next map or leave blank to stop adding maps: ')
#
# 				if item == '':
# 					break
#
# 				schemes += ',' + item
#
# 		name = input('Enter file name for output: ')
# 		schemes = name + ':' + schemes
#
# 	for scheme in schemes.split(';'):
# 		name = scheme.split(':')[0]
# 		mTypes = scheme.split(':')[1].split(',')
# 		maps = list()
# 		print('Making comparison \'' + name + '.png\'.')
#
# 		for i in range(len(mTypes)):
# 			maps.append(list())
# 			maps[i].append(image.open(path + '/maps/' + data1['fileref'] + '/' + mTypes[i] + '.png'))
# 			maps[i].append(image.open(path + '/maps/' + data2['fileref'] + '/' + mTypes[i] + '.png'))
#
# 		utilities.compileImages(maps, border).save(path + '/comparisons/' + name + '.png')
#
# 	print()
# 	print('All comparisons complete.')
# 	input('Press ENTER to close: ')
