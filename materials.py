#-*- coding: utf-8 -*-

import math
import utilities
import fileloader
from ebsd import BravaisLattice


def decode_lattice_type(value):
	match value:
		case 1:
			return BravaisLattice.CP
		case 2:
			return BravaisLattice.CI
		case 3:
			return BravaisLattice.CF
		case 4:
			return BravaisLattice.TP
		case 5:
			return BravaisLattice.TI
		case 6:
			return BravaisLattice.OP
		case 7:
			return BravaisLattice.OI
		case 8:
			return BravaisLattice.OS
		case 9:
			return BravaisLattice.OF
		case 10:
			return BravaisLattice.HR
		case 11:
			return BravaisLattice.HP
		case 12:
			return BravaisLattice.MP
		case 13:
			return BravaisLattice.MS
		case 14:
			return BravaisLattice.AP
		case _:
			raise ValueError("Forbidden lattice type encoding.")


def add(database):
	
	print()
	IDs = utilities.parse_ids(input('Enter material IDs to add separated by commas/hyphens: '))
	materials = fileloader.get_materials()
	
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
					lType = decode_lattice_type(int(pfDatabase.readline()[22:-20]))
					a = float(pfDatabase.readline()[12:-10]) / 10
					b = float(pfDatabase.readline()[12:-10]) / 10
					c = float(pfDatabase.readline()[12:-10]) / 10
					alpha = float(pfDatabase.readline()[16:-14])
					beta = float(pfDatabase.readline()[15:-13])
					gamma = float(pfDatabase.readline()[16:-14])
					print('Material ' + str(ID) + ':')
					print('Name: ' + name)
					print('Lattice type: ' + lType.value)
					print('Lattice constants: ' + str(a) + ' nm, ' + str(b) + ' nm, ' + str(c) + ' nm')
					print('Lattice angles: ' + str(alpha) + ' deg, ' + str(beta) + ' deg, ' + str(gamma) + ' deg')
					Z = float(input('Enter average atomic number: '))
					A = float(input('Enter average atomic weight (g/mol): '))
					rho = float(input('Enter density (g/cm³): '))
					xrms = float(input('Enter thermal vibration amplitude (nm): '))
					diamond = False
					
					if lType.value == 'cF':
						diamond = input('Does crystal have diamond structure? (Y/N): ').lower() == 'y'

					material = fileloader.Material(
						name=name,
						atomic_number=Z,
						atomic_weight=A,
						density=rho,
						vibration_amplitude=xrms,
						lattice_type=lType,
						lattice_constants=(a, b, c),
						lattice_angles=(math.radians(alpha), math.radians(beta), math.radians(gamma)),
						has_diamond_structure=diamond
					)
					
					materials[ID] = material
					break
				
				if line == '':
					print('No material found with ID ' + str(ID) + '.')
					break
	
	with open('materials/materials.csv', 'w') as output:
		output.write('ID,Name,Z,A (g/mol), Density (g/cm3),Thermal vibration (nm),Type,a (nm),b (nm),c (nm),alpha (deg),beta (deg),gamma (deg),Diamond structure\n')
		
		for ID, material in sorted(materials.items()):
			output.write(str(ID) + ',')
			output.write(str(material.name) + ',')
			output.write(str(material.atomic_number) + ',')
			output.write(str(material.atomic_weight) + ',')
			output.write(str(material.density) + ',')
			output.write(str(material.vibration_amplitude) + ',')
			output.write(material.lattice_type.value + ',')
			output.write(str(material.lattice_constants[0]) + ',')
			output.write(str(material.lattice_constants[1]) + ',')
			output.write(str(material.lattice_constants[2]) + ',')
			output.write(str(math.degrees(material.lattice_angles[0])) + ',')
			output.write(str(math.degrees(material.lattice_angles[1])) + ',')
			output.write(str(math.degrees(material.lattice_angles[2])) + ',')
			
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
