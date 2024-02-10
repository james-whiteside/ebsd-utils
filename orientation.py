#-*- coding: utf-8 -*-

import math
import copy
import itertools
import numpy
import utilities

def formatIndices(hkl, iType):
	
	if iType == 'zone':
		prefix = '['
		suffix = ']'
	elif iType == 'zones':
		prefix = '<'
		suffix = '>'
	elif iType == 'plane':
		prefix = '('
		suffix = ')'
	elif iType == 'planes':
		prefix = '{'
		suffix = '}'
	
	return prefix + str(hkl[0]) + ' ' + str(hkl[1]) + ' ' + str(hkl[2]) + suffix

def genFamily(hkl):
	
	hcf = utilities.hcf(hkl)
	
	if hcf != 0:
		hkl = list(index // hcf for index in hkl)
	
	perms = sorted(list(set(list(itertools.permutations(hkl)))))
	refs = sorted(list(set(list(itertools.permutations(list((1, 1, 1, -1, -1, -1)), 3)))))
	refperms = sorted(list(list(refperm) for refperm in set((ref[0] * perm[0], ref[1] * perm[1], ref[2] * perm[2]) for ref in refs for perm in perms)), reverse=True)
	output = list()
	
	for hklA in refperms:
		dupe = False
		
		for hklB in output:
			if (hklA[0] == hklB[0] and hklA[1] == hklB[1] and hklA[2] == hklB[2]) or (hklA[0] == -hklB[0] and hklA[1] == -hklB[1] and hklA[2] == -hklB[2]):
				dupe = True
		
		pol = 0
		
		for i in range(3):
			if hklA[i] > 0:
				pol += 1
			if hklA[i] < 0:
				pol -= 1
		
		if not dupe and pol >= 0:
			output.append(hklA)
	
	return output

def genPairs(hklAs, hklBs):
	
	output = list()
	
	for hklA in hklAs:
		for hklB in hklBs:
			output.append(list((hklA, hklB)))
	
	return output

def genSet(planePair, zonePair):
	
	pair1 = list((numpy.array(planePair[0]), numpy.array(planePair[1])))
	pair2 = list((numpy.array(zonePair[0]), numpy.array(zonePair[1])))
	pair3 = list((numpy.cross(numpy.array(planePair[0]), numpy.array(zonePair[0])), numpy.cross(numpy.array(planePair[1]), numpy.array(zonePair[1]))))
	return list((list(list(hkl) for hkl in pair1), list(list(hkl) for hkl in pair2), list(list(hkl) for hkl in pair3)))

def genMatrix(pairs, params):
	
	pairs = copy.deepcopy(pairs)
	pairs.append(list(list(indices) for indices in (numpy.cross(numpy.array(pairs[0][0]), numpy.array(pairs[1][0])), numpy.cross(numpy.array(pairs[0][1]), numpy.array(pairs[1][1])))))
	x = numpy.array(list((params[0][i] * numpy.linalg.norm(numpy.array(pairs[i][0]))) / (params[1][i] * numpy.linalg.norm(numpy.array(pairs[i][1]))) for i in range(3)))
	uA = numpy.transpose(numpy.array(list(pairs[i][0] for i in range(3))))
	uB = numpy.transpose(numpy.array(list(pairs[i][1] for i in range(3))))
	J = numpy.dot(x * uB, numpy.linalg.inv(uA))
	return J

def genTwin(plane):
	
	h, k, l = plane
	T = numpy.array(list((list((h ** 2 - k ** 2 - l ** 2, 2 * h * k, 2 * l * h)), list((2 * h * k, k ** 2 - l ** 2 - h ** 2, 2 * k * l)), list((2 * l * h, 2 * k * l, l ** 2 - h ** 2 - k ** 2)))))
	s = - 1 / (h ** 2 + k ** 2 + l ** 2)
	J = s * T
	return J

def getVariantList(filepath):
	
	output = list()
	
	with open(filepath, 'r') as file:
		for line in file:
			pairs = list(list(list(int(index) for index in indices.split(',')) for indices in pair.split(':')) for pair in line.split(';'))
			pairs.append(list(list(indices) for indices in (numpy.cross(numpy.array(pairs[0][0]), numpy.array(pairs[1][0])), numpy.cross(numpy.array(pairs[0][1]), numpy.array(pairs[1][1])))))
			output.append(pairs)
	
	return output

if __name__ == '__main__':
	family = genFamily(list((0, 0, 1)))
	print(family)
	print(len(family))
	print()
	for plane in family:
		print(plane)
		print(genTwin(plane))
	'''
	print(genMatrix(list((list((list((1, 0, 0)), list((0, -1, 1)))), list((list((0, 1, 0)), list((1, -1, -1)))), list((list((0, 0, 1)), list((2, 1, 1)))))), list((list((4.5241, 5.0883, 6.7416)), list((2.8662, 2.8662, 2.8662))))))
	print(getVariantList('ortest/ksor.txt'))
	'''
	'''
	hklA = list((1, 1, 1))
	hklB = list((1, 1, 0))
	print('||'.join(list((formatIndices(hklA, 'planes'), formatIndices(hklB, 'planes')))) + ', ' + '||'.join(list((formatIndices(hklB, 'zones'), formatIndices(hklA, 'zones')))) + ':')
	planePairs = genPairs(genFamily(hklA), genFamily(hklB))
	zonePairs = genPairs(genFamily(hklB), genFamily(hklA))
	dps = dict()
	count = 0
	
	for planePair in planePairs:
		for zonePair in zonePairs:
			pairs = genSet(planePair, zonePair)
			
			if sorted(list(abs(index) for index in pairs[2][0])) == list((1, 1, 2)) and sorted(list(abs(index) for index in pairs[2][1])) == list((1, 1, 2)):
				pass
			else:
				continue
			
			dp = ', '.join(list(str(numpy.dot(numpy.array(pairs[i][0]), numpy.array(pairs[i + 1][1]))) for i in range(-1, 2)))
			
			if dp not in dps:
				dps[dp] = 0
			
			dps[dp] += 1
			
			print(', '.join(list(' ' + '||'.join(list(formatIndices(hkl, 'zone') for hkl in pair)) for pair in pairs)))
			print(dp)
			count += 1
	
	for dp in dps:
		print(dp + ': ' + str(dps[dp]))
	'''
	'''
	variants = list()
	matrices = list()
	
	for planePair in planePairs:
		for zonePair in zonePairs:
			pairs = genSet(planePair, zonePair)
			
			if sorted(list(abs(index) for index in pairs[2][0])) == list((1, 1, 2)) and sorted(list(abs(index) for index in pairs[2][1])) == list((1, 1, 2)):
				pass
			else:
				continue
			
			params = list((list((1, 1, 1)), list((1, 1, 1))))
			matrix = genMatrix(pairs, params)
			dupe = False
			
			for m in matrices:
				if numpy.array_equal(m, matrix):
					dupe = True
			
			if not dupe:
				variants.append(pairs)
				matrices.append(matrix)
				print(', '.join(list(' ' + '||'.join(list(formatIndices(hkl, 'zone') for hkl in pair)) for pair in pairs)))
				print(matrix)
				count += 1
	'''
	input()
