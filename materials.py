#-*- coding: utf-8 -*-

import math
import utilities
import fileloader

def decodeType(ID):
	
	if ID == 1:
		return 'cP'
	elif ID == 2:
		return 'cI'
	elif ID == 3:
		return 'cF'
	elif ID == 4:
		return 'tP'
	elif ID == 5:
		return 'tI'
	elif ID == 6:
		return 'oP'
	elif ID == 7:
		return 'oI'
	elif ID == 8:
		return 'oS'
	elif ID == 9:
		return 'oF'
	elif ID == 10:
		return 'hR'
	elif ID == 11:
		return 'hP'
	elif ID == 12:
		return 'mP'
	elif ID == 13:
		return 'mS'
	elif ID == 14:
		return 'aP'

def add(database):
	
	print()
	IDs = utilities.parse_ids(input('Enter material IDs to add separated by commas/hyphens: '))
	materials = fileloader.getMaterials()
	
	for ID in IDs:
		print()
		
		with open(database, 'r') as pfDatabase:
			while True:
				line = pfDatabase.readline()
				
				if line == '    <CrystalID>' + str(ID) + '</CrystalID>\n':
					try:
						del materials[ID]
					except KeyError:
						pass
					
					name = pfDatabase.readline()[17:-15]
					pfDatabase.readline()
					pfDatabase.readline()
					lType = decodeType(int(pfDatabase.readline()[22:-20]))
					a = float(pfDatabase.readline()[12:-10]) / 10
					b = float(pfDatabase.readline()[12:-10]) / 10
					c = float(pfDatabase.readline()[12:-10]) / 10
					alpha = float(pfDatabase.readline()[16:-14])
					beta = float(pfDatabase.readline()[15:-13])
					gamma = float(pfDatabase.readline()[16:-14])
					print('Material ' + str(ID) + ':')
					print('Name: ' + name)
					print('Lattice type: ' + lType)
					print('Lattice constants: ' + str(a) + ' nm, ' + str(b) + ' nm, ' + str(c) + ' nm')
					print('Lattice angles: ' + str(alpha) + ' deg, ' + str(beta) + ' deg, ' + str(gamma) + ' deg')
					Z = float(input('Enter average atomic number: '))
					A = float(input('Enter average atomic weight (g/mol): '))
					rho = float(input('Enter density (g/cm³): '))
					xrms = float(input('Enter thermal vibration amplitude (nm): '))
					diamond = False
					
					if lType == 'cF':
						diamond = input('Does crystal have diamond structure? (Y/N): ').lower() == 'y'
					
					materials[ID] = dict()
					materials[ID]['name'] = name
					materials[ID]['Z'] = Z
					materials[ID]['A'] = A
					materials[ID]['density'] = rho
					materials[ID]['vibration'] = xrms
					materials[ID]['type'] = lType
					materials[ID]['constants'] = list((a, b, c))
					materials[ID]['angles'] = list((math.radians(alpha), math.radians(beta), math.radians(gamma)))
					materials[ID]['diamond'] = diamond
					break
				
				if line == '':
					print('No material found with ID ' + str(ID) + '.')
					break
	
	with open('materials/materials.csv', 'w') as output:
		output.write('ID,Name,Z,A (g/mol), Density (g/cm3),Thermal vibration (nm),Type,a (nm),b (nm),c (nm),alpha (deg),beta (deg),gamma (deg),Diamond structure\n')
		
		for ID, material in sorted(materials.items()):
			output.write(str(ID) + ',')
			output.write(str(material['name']) + ',')
			output.write(str(material['Z']) + ',')
			output.write(str(material['A']) + ',')
			output.write(str(material['density']) + ',')
			output.write(str(material['vibration']) + ',')
			output.write(material['type'] + ',')
			output.write(str(material['constants'][0]) + ',')
			output.write(str(material['constants'][1]) + ',')
			output.write(str(material['constants'][2]) + ',')
			output.write(str(math.degrees(material['angles'][0])) + ',')
			output.write(str(math.degrees(material['angles'][1])) + ',')
			output.write(str(math.degrees(material['angles'][2])) + ',')
			
			if diamond:
				output.write('Y\n')
			else:
				output.write('N\n')
	
	print()
	print('All materials added.')
	input('Press ENTER to close: ')

if __name__ == '__main__':
	print('This is a support module.')
	input('Press ENTER to close: ')
