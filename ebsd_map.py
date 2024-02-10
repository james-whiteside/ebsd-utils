#-*- coding: utf-8 -*-

import ebsd

# ebsd.makeMaps(path, size=None)
# path: Working directory for code
# size: Override size for non-spatial maps (px), otherwise defaults to map width

if __name__ == '__main__':
	path = 'ebsd'
	#path = 'ebsd/simulations'
	size = None
	ebsd.makeMaps(path, size)
