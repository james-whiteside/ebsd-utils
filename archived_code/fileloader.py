# -*- coding: utf-8 -*-

import os
import math
from src.data_structures.phase import Phase, BravaisLattice


def get_materials(path: str = f"{os.getcwd()}/example_data/materials.csv".replace("\\", "/")) -> dict[int, Phase]:
	materials = dict()
	
	with open(path, "r") as file:
		file.readline()
		
		for line in file:
			args = line.split(',')
			global_id = int(args[0])

			material = Phase(
				global_id=global_id,
				name=args[1],
				atomic_number=float(args[2]),
				atomic_weight=float(args[3]),
				density=float(args[4]),
				vibration_amplitude=float(args[5]),
				lattice_type=BravaisLattice(args[6]),
				lattice_constants=(float(args[7]), float(args[8]), float(args[9])),
				lattice_angles=(math.radians(float(args[10])), math.radians(float(args[11])), math.radians(float(args[12]))),
				has_diamond_structure=args == "Y",
			)

			materials[global_id] = material
	
	return materials


def getVariantList():
	output = dict()
	
	with open('orientation/vars.csv', 'r') as file:
		file.readline()
		
		for line in file:
			variant = line.split(',')
			output[variant[0]] = dict()
			output[variant[0]]['lTypes'] = list((variant[1], variant[2]))
			pairs = list()
			pairs.append(list((list((int(variant[3]), int(variant[4]), int(variant[5]))), list((int(variant[6]), int(variant[7]), int(variant[8]))))))
			pairs.append(list((list((int(variant[9]), int(variant[10]), int(variant[11]))), list((int(variant[12]), int(variant[13]), int(variant[14]))))))
			#pairs.append(list(list(indices) for indices in (numpy.cross(numpy.array(pairs[0][0]), numpy.array(pairs[1][0])), numpy.cross(numpy.array(pairs[0][1]), numpy.array(pairs[1][1])))))
			output[variant[0]]['vectors'] = pairs
	
	return output


def getTwinList():
	output = dict()
	
	with open('orientation/twin.csv', 'r') as file:
		file.readline()
		
		for line in file:
			ID = 'twin-' + line.replace('\n','').replace(',', '')
			indices = list(int(index) for index in line.split(','))
			output[ID] = indices
	
	return output
