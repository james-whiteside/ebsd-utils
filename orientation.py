# -*- coding: utf-8 -*-

import itertools
import numpy
import utilities


def numpy_cross(a: numpy.ndarray, b: numpy.ndarray) -> numpy.ndarray:
	"""
	Wrapper to work around return type mislabeling bug in NumPy causing IDE errors.
	:param a: First array.
	:param b: Second array.
	:return: Cross product of arrays.
	"""

	return numpy.cross(a, b)


def get_plane_family(indices: tuple[int, int, int]) -> list[tuple[int, int, int]]:
	"""
	For a given set of Miller indices for a plane family {h,k,l}, determines the list of planes (h,k,l) in the family.
	:param indices: Indices of the plane family.
	:return: The list of planes in the family.
	"""
	
	hcf = utilities.highest_common_factor(list(indices))
	
	if hcf != 0:
		reduced_indices = tuple(index // hcf for index in indices)
	else:
		reduced_indices = indices
	
	index_permutations = sorted(list(set(itertools.permutations(reduced_indices))))
	index_parities = sorted(list(set(itertools.permutations((1, 1, 1, -1, -1, -1), 3))))

	planes = set()

	for permutation in index_permutations:
		for parity in index_parities:
			planes.add((parity[0] * permutation[0], parity[1] * permutation[1], parity[2] * permutation[2]))

	plane_family = list()
	
	for plane in sorted(list(planes)):
		if plane in plane_family or -1 * plane in plane_family:
			continue
		
		plane_parity = 0
		
		for i in range(3):
			if plane[i] == 0:
				continue
			elif plane[i] > 0:
				plane_parity += 1
			elif plane[i] < 0:
				plane_parity -= 1
		
		if plane_parity < 0:
			continue

		plane_family.append(plane)
	
	return plane_family


def get_twin_matrix(indices: tuple[int, int, int]) -> numpy.ndarray:
	"""
	Determines the rotation matrix for the homophase cubic orientation relationship described by a reflection in a plane.
	Solves Eqn. 3.30.
	:param indices: The Miller indices (h,k,l) of the reflecting plane.
	:return: The rotation matrix describing the twin.
	"""

	h, k, l = indices

	T = numpy.array((
		(h ** 2 - k ** 2 - l ** 2, 2 * h * k, 2 * l * h),
		(2 * h * k, k ** 2 - l ** 2 - h ** 2, 2 * k * l),
		(2 * l * h, 2 * k * l, l ** 2 - h ** 2 - k ** 2),
	))

	J = - 1 / (h ** 2 + k ** 2 + l ** 2) * T
	return J


def get_relationship_matrix(
		u1A: tuple[int, int, int],
		u1B: tuple[int, int, int],
		u2A: tuple[int, int, int],
		u2B: tuple[int, int, int],
		a: tuple[float, float, float],
		b: tuple[float, float, float],
) -> numpy.ndarray:
	"""
	Determines the rotation matrix for the heterophase orientation relationship described by two pairs of parallel zone axes.
	Parallel axis pairs are u1A || u1B and u2A || u2B for bases A and B.
	Solves Eqn. 3.50.
	:param u1A: Zone axis 1 for basis A.
	:param u1B: Zone axis 1 for basis B.
	:param u2A: Zone axis 2 for basis A.
	:param u2B: Zone axis 2 for basis B.
	:param a: Basis vectors of basis A.
	:param b: Basis vectors of basis B.
	:return: The rotation matrix describing the orientation relationship between bases A and B.
	"""

	u1A = numpy.array(u1A)
	u1B = numpy.array(u1B)
	u2A = numpy.array(u2A)
	u2B = numpy.array(u2B)
	u3A = numpy.array(int(element) for element in numpy_cross(numpy.array(u1A), numpy.array(u2A)))
	u3B = numpy.array(int(element) for element in numpy_cross(numpy.array(u1B), numpy.array(u2B)))

	x = numpy.array([
		(a[0] * numpy.linalg.norm(u1A)) / (b[0] * numpy.linalg.norm(u1B)),
		(a[1] * numpy.linalg.norm(u2A)) / (b[1] * numpy.linalg.norm(u2B)),
		(a[2] * numpy.linalg.norm(u3A)) / (b[2] * numpy.linalg.norm(u3B)),
	])

	uA = numpy.transpose(numpy.array((u1A, u2A, u3A)))
	uB = numpy.transpose(numpy.array((u1B, u2B, u3B)))
	J = numpy.dot(x * uB, numpy.linalg.inv(uA))
	return J
