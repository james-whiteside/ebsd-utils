#-*- coding: utf-8 -*-

import materials

# materials.add(database)
# database: Location of Pathfinder crystal database

if __name__ == '__main__':
	database = 'C:/ProgramData/Thermo Scientific/NSS/NSS Libraries/EBSD/DefaultCrystalDatabase.XML'
	materials.add(database)
