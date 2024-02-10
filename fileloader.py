#-*- coding: utf-8 -*-

import sys
import math
import numpy
from PIL import Image as image

def getMaterials():
	
	output = dict()
	
	with open('materials/materials.csv', 'r') as file:
		file.readline()
		
		for line in file:
			material = line.split(',')
			output[int(material[0])] = dict()
			output[int(material[0])]['name'] = material[1]
			output[int(material[0])]['Z'] = float(material[2])
			output[int(material[0])]['A'] = float(material[3])
			output[int(material[0])]['density'] = float(material[4])
			output[int(material[0])]['vibration'] = float(material[5]) # For some values at RT see D.S.Gemmell, Rev. Mod. Phys. 46 (1974) 129.
			output[int(material[0])]['type'] = material[6]
			output[int(material[0])]['constants'] = list((float(material[7]), float(material[8]), float(material[9])))
			output[int(material[0])]['angles'] = list((math.radians(float(material[10])), math.radians(float(material[11])), math.radians(float(material[12]))))
			output[int(material[0])]['diamond'] = material[13] == 'Y'
	
	return output

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

def getElectronRun(filepath):
	
	output = dict()
	
	with open(filepath, 'r') as file:
		output['fileref'] = filepath.split('/')[-1].split('.')[0].lstrip('r')
		output['material'] = int(file.readline().rstrip('\n').split(',')[1])
		output['energy'] = float(file.readline().rstrip('\n').split(',')[1])
		output['tilt'] = float(file.readline().rstrip('\n').split(',')[1])
		output['limit'] = float(file.readline().rstrip('\n').split(',')[1])
		output['tests'] = int(file.readline().rstrip('\n').split(',')[1])
		output['data'] = dict()
		output['data']['x'] = list()
		output['data']['y'] = list()
		output['data']['z'] = list()
		output['data']['r'] = list()
		output['data']['E'] = list()
		output['data']['d'] = list()
		file.readline()
		file.readline()
		
		for i in range(output['tests']):
			line = file.readline().rstrip('\n').split(',')
			output['data']['x'].append(float(line[1]))
			output['data']['y'].append(float(line[2]))
			output['data']['z'].append(float(line[3]))
			output['data']['r'].append(float(line[4]))
			output['data']['E'].append(float(line[5]))
			output['data']['d'].append(float(line[6]))
	
	return output

def getElectronAnalysis(filepath):
	
	output = dict()
	
	with open(filepath, 'r') as file:
		output['fileref'] = filepath.split('/')[-1].split('.')[0].lstrip('a')
		output['material'] = int(file.readline().rstrip('\n').split(',')[1])
		output['energy'] = float(file.readline().rstrip('\n').split(',')[1])
		output['tilt'] = float(file.readline().rstrip('\n').split(',')[1])
		output['limit'] = float(file.readline().rstrip('\n').split(',')[1])
		output['tests'] = int(file.readline().rstrip('\n').split(',')[1])
		output['radius'] = float(file.readline().rstrip('\n').split(',')[1])
		output['coefficient'] = float(file.readline().rstrip('\n').split(',')[1])
		output['spectrum'] = dict()
		file.readline()
		output['spectrum']['n'] = int(file.readline().rstrip('\n').split(',')[1])
		output['spectrum']['R2'] = float(file.readline().rstrip('\n').split(',')[1])
		output['spectrum']['fit'] = list()
		file.readline()
		
		for i in range(output['spectrum']['n'] + 1):
			output['spectrum']['fit'].append(float(file.readline().rstrip('\n').split(',')[1]))
		
		output['spectrum']['plot'] = dict()
		output['spectrum']['plot']['P'] = list()
		output['spectrum']['plot']['E'] = list()
		output['spectrum']['plot']['F'] = list()
		file.readline()
		file.readline()
		
		for i in range(int(output['tests'] * output['coefficient']) + 2):
			line = file.readline().rstrip('\n').split(',')
			output['spectrum']['plot']['P'].append(float(line[0]))
			output['spectrum']['plot']['E'].append(float(line[1]))
			output['spectrum']['plot']['F'].append(float(line[2]))
	
	return output

def getElectronDiffraction(filepath):
	
	output = dict()
	
	with open(filepath, 'r') as file:
		output['fileref'] = filepath.split('/')[-1].split('.')[0].lstrip('d')
		output['material'] = int(file.readline().rstrip('\n').split(',')[1])
		output['energy'] = float(file.readline().rstrip('\n').split(',')[1])
		output['tilt'] = float(file.readline().rstrip('\n').split(',')[1])
		output['limit'] = float(file.readline().rstrip('\n').split(',')[1])
		output['tests'] = int(file.readline().rstrip('\n').split(',')[1])
		output['radius'] = float(file.readline().rstrip('\n').split(',')[1])
		output['coefficient'] = float(file.readline().rstrip('\n').split(',')[1])
		output['spectrum'] = dict()
		file.readline()
		output['spectrum']['n'] = int(file.readline().rstrip('\n').split(',')[1])
		output['spectrum']['R2'] = float(file.readline().rstrip('\n').split(',')[1])
		output['pattern'] = dict()
		file.readline()
		output['pattern']['hp'] = float(file.readline().rstrip('\n').split(',')[1])
		output['pattern']['lp'] = float(file.readline().rstrip('\n').split(',')[1])
		output['pattern']['hpP'] = float(file.readline().rstrip('\n').split(',')[1])
		output['pattern']['lpP'] = float(file.readline().rstrip('\n').split(',')[1])
		output['pattern']['coefficient'] = float(file.readline().rstrip('\n').split(',')[1])
		output['pattern']['m'] = int(file.readline().rstrip('\n').split(',')[1])
		output['pattern']['n'] = int(file.readline().rstrip('\n').split(',')[1])
		output['pattern']['contrast'] = float(file.readline().rstrip('\n').split(',')[1])
		output['pattern']['S'] = float(file.readline().rstrip('\n').split(',')[1])
		output['pattern']['mS'] = float(file.readline().rstrip('\n').split(',')[1])
		output['pattern']['hkl'] = list()
		file.readline()
		output['pattern']['hkl'].append(int(file.readline().rstrip('\n').split(',')[1]))
		output['pattern']['hkl'].append(int(file.readline().rstrip('\n').split(',')[1]))
		output['pattern']['hkl'].append(int(file.readline().rstrip('\n').split(',')[1]))
		output['pattern']['profile'] = dict()
		output['pattern']['profile']['psi'] = list()
		output['pattern']['profile']['I'] = list()
		file.readline()
		file.readline()
		
		for i in range(output['pattern']['m'] + 1):
			line = file.readline().rstrip('\n').split(',')
			output['pattern']['profile']['psi'].append(float(line[0]))
			output['pattern']['profile']['I'].append(float(line[1]))
	
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
		materials = getMaterials()
		file.readline()
		
		while True:
			line = file.readline().rstrip('\n').split(',')
			
			if line[0] == 'Map Size:':
				break
			
			localID = int(line[0])
			output['phases'][localID] = dict()
			output['phases'][localID]['ID'] = int(line[2])
			
			try:
				output['phases'][localID]['name'] = materials[output['phases'][localID]['ID']]['name']
			except KeyError:
				input('No material with ID ' + str(output['phases'][localID]['ID']) + '.')
				sys.exit()
			
			output['phases'][localID]['Z'] = materials[output['phases'][localID]['ID']]['Z']
			output['phases'][localID]['A'] = materials[output['phases'][localID]['ID']]['A']
			output['phases'][localID]['density'] = materials[output['phases'][localID]['ID']]['density']
			output['phases'][localID]['type'] = materials[output['phases'][localID]['ID']]['type']
			output['phases'][localID]['constants'] = materials[output['phases'][localID]['ID']]['constants']
			output['phases'][localID]['angles'] = materials[output['phases'][localID]['ID']]['angles']
		
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
		materials = getMaterials()
		file.readline()
		
		while True:
			line = file.readline().rstrip('\n').split(',')
			
			if line[0] == 'Map Size:':
				break
			
			localID = int(line[0])
			output['phases'][localID] = dict()
			output['phases'][localID]['ID'] = int(line[2])
			output['phases'][localID]['name'] = materials[output['phases'][localID]['ID']]['name']
			output['phases'][localID]['Z'] = materials[output['phases'][localID]['ID']]['Z']
			output['phases'][localID]['A'] = materials[output['phases'][localID]['ID']]['A']
			output['phases'][localID]['density'] = materials[output['phases'][localID]['ID']]['density']
			output['phases'][localID]['type'] = materials[output['phases'][localID]['ID']]['type']
			output['phases'][localID]['constants'] = materials[output['phases'][localID]['ID']]['constants']
			output['phases'][localID]['angles'] = materials[output['phases'][localID]['ID']]['angles']
		
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
