#-*- coding: utf-8 -*-

import channelling

# hobler.run(beamZ, targetID, E, maxRange=10, maxIndex=10)
# beamZ:    Atomic number of beam species
# targetID: As in materials/materials.csv
# E:        Beam energy (eV)
# maxRange: Maximum range from origin where rows are to be considered (Å)
# maxIndex: Maximum Miller index to be considered

if __name__ == '__main__':
	beamZ = 31
	#targetID = 60696
	targetID = 310619
	E = 30000
	channelling.run(beamZ, targetID, E)
