# -*- coding: utf-8 -*-

import sys
import math
import numpy
from PIL import Image as image
from ebsd import BravaisLattice


class Material:
	def __init__(
		self,
		global_id: int,
		name: str,
		atomic_number: float,
		atomic_weight: float,
		density: float,
		vibration_amplitude: float,
		lattice_type: BravaisLattice,
		lattice_constants: tuple[float, float, float],
		lattice_angles: tuple[float, float, float],
		has_diamond_structure: bool,
	):
		self.global_id = global_id
		self.name = name
		self.atomic_number = atomic_number
		self.atomic_weight = atomic_weight
		self.density = density
		self.vibration_amplitude = vibration_amplitude
		self.lattice_type = lattice_type
		self.lattice_constants = lattice_constants
		self.lattice_angles = lattice_angles
		self.has_diamond_structure = has_diamond_structure

	def __eq__(self, other):
		if not isinstance(other, Material):
			return False

		if self.global_id == other.global_id:
			return True
		else:
			return False

	@property
	def close_pack_distance(self) -> float:
		if self.lattice_type is BravaisLattice.CP:
			return self.lattice_constants[0]
		elif self.lattice_type is BravaisLattice.CI:
			return math.sqrt(3) * self.lattice_constants[0] / 2
		elif self.lattice_type is BravaisLattice.CF:
			return math.sqrt(2) * self.lattice_constants[0] / 2
		else:
			raise NotImplementedError

	def as_dict(self):
		return {
			"global_id": self.global_id,
			"name": self.name,
			"Z": self.atomic_number,
			"A": self.atomic_weight,
			"density": self.density,
			"vibration": self.vibration_amplitude,
			"type": self.lattice_type,
			"constants": self.lattice_constants,
			"angles": self.lattice_angles,
			"diamond": self.has_diamond_structure,
		}


def get_materials(path: str = "materials/materials.csv") -> dict[int, Material]:
	materials = dict()
	
	with open(path, "r") as file:
		file.readline()
		
		for line in file:
			args = line.split(',')
			material_id = int(args[0])

			material = Material(
				global_id=material_id,
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

			materials[material_id] = material
	
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

def getNyeMetadata(filepath):
	
	output = dict()
	
	with open(filepath, 'r') as file:
		file.readline()
		
		for line in file:
			metadata = line.split(',')
			output[metadata[0]] = dict()
			output[metadata[0]]['aType'] = metadata[1]
			output[metadata[0]]['dx'] = float(metadata[2]) * 10 ** -6
			output[metadata[0]]['Z'] = int(metadata[3])
			output[metadata[0]]['E'] = float(metadata[4])
			output[metadata[0]]['tau'] = float(metadata[5])
			output[metadata[0]]['n'] = int(metadata[6])
			output[metadata[0]]['epsilon'] = float(metadata[7])
			output[metadata[0]]['k'] = int(metadata[8])
	
	return output

def getNyeData(filepath):
	
	output = dict()
	
	with open(filepath, 'r') as file:
		output['fileref'] = filepath.split('/')[-1].split('.')[0].lstrip('p')
		output['phases'] = dict()
		materials = get_materials()
		file.readline()
		
		while True:
			line = file.readline().rstrip('\n').split(',')
			
			if line[0] == 'Map Size:':
				break
			
			localID = int(line[0])
			output['phases'][localID] = dict()
			output['phases'][localID]['ID'] = int(line[2])
			
			try:
				output['phases'][localID]['name'] = materials[output['phases'][localID]['ID']].name
			except KeyError:
				input('No material with ID ' + str(output['phases'][localID]['ID']) + '.')
				sys.exit()
			
			output['phases'][localID]['Z'] = materials[output['phases'][localID]['ID']].atomic_number
			output['phases'][localID]['A'] = materials[output['phases'][localID]['ID']].atomic_weight
			output['phases'][localID]['density'] = materials[output['phases'][localID]['ID']].density
			output['phases'][localID]['type'] = materials[output['phases'][localID]['ID']].lattice_type
			output['phases'][localID]['constants'] = materials[output['phases'][localID]['ID']].lattice_constants
			output['phases'][localID]['angles'] = materials[output['phases'][localID]['ID']].lattice_angles
		
		output['width'] = int(file.readline().rstrip('\n').split(',')[1])
		output['height'] = int(file.readline().rstrip('\n').split(',')[1])
		output['data'] = dict()
		output['data']['phase'] = list()
		output['data']['euler'] = list()
		output['data']['IQ'] = list()
		output['data']['PQ'] = list()
		file.readline()
		file.readline()
		
		for y in range(output['height']):
			output['data']['phase'].append(list())
			output['data']['euler'].append(list())
			output['data']['IQ'].append(list())
			output['data']['PQ'].append(list())
			
			for x in range(output['width']):
				line = file.readline().rstrip('\n').split(',')
				output['data']['phase'][y].append(int(line[2]))
				output['data']['euler'][y].append(list((math.radians(float(line[3])), math.radians(float(line[4])), math.radians(float(line[5])))))
				output['data']['IQ'][y].append(float(line[6]))
				output['data']['PQ'][y].append(float(line[7]))
	
	return output

def getNyeAnalysis(filepath):
	
	output = dict()
	metadata = getNyeMetadata('/'.join(filepath.split('/')[:-2]) + '/metadata.csv')
	
	with open(filepath, 'r') as file:
		output['fileref'] = filepath.split('/')[-1].split('.')[0].lstrip('q')
		output['phases'] = dict()
		materials = get_materials()
		file.readline()
		
		while True:
			line = file.readline().rstrip('\n').split(',')
			
			if line[0] == 'Map Size:':
				break
			
			localID = int(line[0])
			output['phases'][localID] = dict()
			output['phases'][localID]['ID'] = int(line[2])
			output['phases'][localID]['name'] = materials[output['phases'][localID]['ID']].name
			output['phases'][localID]['Z'] = materials[output['phases'][localID]['ID']].atomic_number
			output['phases'][localID]['A'] = materials[output['phases'][localID]['ID']].atomic_weight
			output['phases'][localID]['density'] = materials[output['phases'][localID]['ID']].density
			output['phases'][localID]['type'] = materials[output['phases'][localID]['ID']].lattice_type
			output['phases'][localID]['constants'] = materials[output['phases'][localID]['ID']].lattice_constants
			output['phases'][localID]['angles'] = materials[output['phases'][localID]['ID']].lattice_angles
		
		output['width'] = int(file.readline().rstrip('\n').split(',')[1])
		output['height'] = int(file.readline().rstrip('\n').split(',')[1])
		output['data'] = dict()
		output['data']['phase'] = list()
		output['data']['euler'] = list()
		output['data']['IQ'] = list()
		output['data']['PQ'] = list()
		output['data']['SGP'] = list()
		
		if 'g' in metadata[output['fileref']]['aType']:
			output['data']['KAM'] = list()
			output['data']['GND'] = list()
		
		if 'c' in metadata[output['fileref']]['aType']:
			output['data']['CF'] = list()
		
		if 'd' in metadata[output['fileref']]['aType']:
			output['data']['cType'] = list()
			output['data']['cID'] = list()
		
		file.readline()
		file.readline()
		
		for y in range(output['height']):
			output['data']['phase'].append(list())
			output['data']['euler'].append(list())
			output['data']['IQ'].append(list())
			output['data']['PQ'].append(list())
			output['data']['SGP'].append(list())
			
			if 'g' in metadata[output['fileref']]['aType']:
				output['data']['KAM'].append(list())
				output['data']['GND'].append(list())
			
			if 'c' in metadata[output['fileref']]['aType']:
				output['data']['CF'].append(list())
			
			if 'd' in metadata[output['fileref']]['aType']:
				output['data']['cType'].append(list())
				output['data']['cID'].append(list())
			
			for x in range(output['width']):
				line = file.readline().rstrip('\n').split(',')
				output['data']['phase'][y].append(int(line[2]))
				output['data']['euler'][y].append(list((math.radians(float(line[3])), math.radians(float(line[4])), math.radians(float(line[5])))))
				output['data']['IQ'][y].append(float(line[6]))
				output['data']['PQ'][y].append(float(line[7]))
				output['data']['SGP'][y].append(list((float(line[8]), float(line[9]))))
				
				linePos = 10
				
				if 'g' in metadata[output['fileref']]['aType']:
					output['data']['KAM'][y].append(float(line[linePos]))
					output['data']['GND'][y].append(float(line[linePos + 1]))
					linePos += 2
				
				if 'c' in metadata[output['fileref']]['aType']:
					output['data']['CF'][y].append(float(line[linePos]))
					linePos += 1
				
				if 'd' in metadata[output['fileref']]['aType']:
					output['data']['cType'][y].append(line[linePos])
					output['data']['cID'][y].append(int(line[linePos + 1]))
					linePos += 2
	
	return output

def getImage(filepath):
	
	output = dict()
	
	with image.open(filepath, 'r').convert('L') as img:
		output['width'], output['height'] = img.size
		output['pixels'] = numpy.array(list(img.getdata())).reshape((output['height'], output['width']))
	
	return output

if __name__ == '__main__':
	print('This is a support module.')
	input('Press ENTER to close: ')
