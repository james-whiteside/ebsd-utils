#-*- coding: utf-8 -*-

import glob
import sys
import os
import math
import datetime
import copy
import numpy
from PIL import Image as image

def hcf(nums):
	
	nums = copy.deepcopy(nums)
	
	if len(nums) == 2:
		x = nums[0]
		y = nums[1]
		
		while y:
			x, y = y, x % y
		
		return x
	else:
		z = nums.pop()
		return hcf(list((z, hcf(nums))))

def sigFig(n, sf):

	if float(n) == 0.0:
		return 0.0
	else:
		return round(float(n), -int(math.floor(math.log10(abs(float(n)))) - sf + 1))

def intSigFig(n, sf):

	if abs(n) >= 10 ** (sf - 1):
		return int(round(n))
	else:
		return sigFig(n, sf)

def formatError(n, e):

	return str(round(n, -int(math.floor(math.log10(e))))) + '(' + str(format(sigFig(e, 1), 'f')).lstrip('.0').rstrip('0').rstrip('.') +')'

def formatSymbols(string):

	return string.replace('#a', 'α').replace('#g', 'γ').replace('#l', 'λ').replace('#m', 'μ').replace('#t', 'τ').replace('#-', '⁻').replace('#1', '¹').replace('#2', '²').replace('#>', '→')

def formatSubs(string):

	return string.replace('+-', '-')

def formatFileSize(size):

	size = float(size)
	
	if size < 1024.0:
		return str(int(size)) + ' B'
	
	size /= 1024.0
	
	if size < 1024.0:
		return str(sigFig(size, 3)) + ' kB'
	
	size /= 1024.0
	
	if size < 1024.0:
		return str(sigFig(size, 3)) + ' MB'
	
	size /= 1024.0
	return str(sigFig(size, 3)) + ' GB'

def formatTime(time):
	
	seconds = time % 60
	time = int(round((time - seconds) / 60))
	minutes = time % 60
	hours = int(round((time - minutes) / 60))

def format_time(time_seconds: int | float) -> str:
	if type(time_seconds) is float:
		time_seconds = int(round(time_seconds))

	seconds = time_seconds % 60
	time_minutes = int(round((time_seconds - seconds) / 60))
	minutes = time_minutes % 60
	hours = int(round((time_seconds - minutes) / 60))
	return str(hours).zfill(2) + ':' + str(minutes).zfill(2) + ':' + str(seconds).zfill(2)


def decodeIDs(string):

	items = set()
	domains = string.split(',')
	
	for domain in domains:
		limits = domain.split('-')
		
		if len(limits) == 1:
			items.add(int(limits[0]))
		elif len(limits) == 2:
			for i in range(int(limits[0]), int(limits[1]) + 1):
				items.add(i)
		else:
			for i in range(int(limits[0]), int(limits[1]) + 1, int(limits[2])):
				items.add(i)
	
	return sorted(list(items))

def getDir(subDir=None):
	
	if subDir == None:
		path = os.getcwd().replace('\\', '/') + '/'
		
		while True:
			subDirs = list()
			print('Current directory: \'' + path + '\'')
			print('Subdirectories found: ')
			subDirs.append('../')
			print(' - ID: 0, Name: \'../\'')
			
			for subDir in list(subDir.replace('\\', '/').replace(path, '') for subDir in sorted(glob.glob(path + '*/'))):
				subDirs.append(subDir)
				print(' - ID: ' + str(len(subDirs)-1) + ', Name: \'' + subDir + '\'')
			
			subDirID = input('Enter ID to change directory or leave blank to use current: ')
			
			if subDirID == '':
				break
			elif int(subDirID) == 0:
				path = '/'.join(path.split('/')[:-2]) + '/'
			else:
				path += subDirs[int(subDirID)]
		
		path = path[:-1]
		
	else:
		path = os.getcwd().replace('\\', '/') + '/' + subDir
	
	return path

def getFile(path, extension, getSubDirs=False, getMany=False, prompt='Files found:', exclusions=list()):

	path += '/**'
	files = list()
	subDirs = list(subDir.replace('\\', '/') for subDir in glob.glob(path + '/', recursive=getSubDirs))
	print(prompt)
	
	for file in list(file.replace('\\', '/') for file in sorted(glob.glob(path, recursive=getSubDirs))):
		if file[-1] == '/' or file + '/' in subDirs or (extension != '' and file.split('/')[-1].split('.')[-1] != extension) or file.split('/')[-1] in list(exclusion + '.' + extension for exclusion in exclusions):
			continue
		
		files.append(file)
		print(' - ID: ' + str(len(files)-1) + ', Name: \'' + file.split('/')[-1] + '\', Size: ' + formatFileSize(os.path.getsize(file)))
	
	if len(files) == 0:
		print(' None')
		input('Press ENTER to exit: ')
		sys.exit()
	
	if not getMany:
		return files[int(input('Enter file ID to read from: '))]
	
	return list(files[fileID] for fileID in decodeIDs(input('Enter file IDs to read from separated by commas/hyphens: ')))

def colourWheel(i, n):
	
	theta = 360 * i / n
	
	if 0 <= theta < 60:
		R = 1
		G = theta / 60
		B = 0
	elif 60 <= theta < 120:
		R = (120 - theta) / 60
		G = 1
		B = 0
	elif 120 <= theta < 180:
		R = 0
		G = 1
		B = (theta - 120) / 60
	elif 180 <= theta < 240:
		R = 0
		G = (240 - theta) / 60
		B = 1
	elif 240 <= theta < 300:
		R = (theta - 240) / 60
		G = 0
		B = 1
	elif 300 <= theta < 360:
		R = 1
		G = 0
		B = (360 - theta) / 60
	
	return list((R, G, B))

def loadImage(path):
	
	img = image.open(path)
	output = dict()
	output['mode'] = img.mode
	output['width'] = img.width
	output['height'] = img.height
	output['pixels'] = list()
	
	for y in range(output['height']):
		output['pixels'].append(list())
		
		for x in range(output['width']):
			output['pixels'][y].append(img.getpixel((x, y)))
	
	
	
	return output

def makeImage(data, width, height, lim, mode):
	
	output = image.new(mode, (width, height))
	
	for y in range(height):
		for x in range(width):
			if mode == '1':
				pixel = int(bool(data[y][x]))
			
			if mode == 'L':
				pixel = int(round(255 * data[y][x] / lim))
			
			if mode == 'RGB':
				pixel = tuple(int(round(255 * data[y][x][i] / lim)) for i in range(3))
			
			if mode == 'RGBA':
				pixel = tuple(int(round(255 * data[y][x][i] / lim)) for i in range(4))
			
			output.putpixel((x, y), pixel)
	
	return output

def compileImages(imgs, border=0, bg=(255, 255, 255)):
	
	rows = len(imgs)
	columns = len(imgs[0])
	maxWidths = list()
	maxHeights = list()
	
	for x in range(columns):
		maxWidth = 0
		
		for y in range(rows):
			maxWidth = max(maxWidth, imgs[y][x].width)
		
		maxWidths.append(maxWidth)
	
	for y in range(rows):
		maxHeight = 0
		
		for x in range(columns):
			maxHeight = max(maxHeight, imgs[y][x].height)
		
		maxHeights.append(maxHeight)
	
	outputWidth = sum(maxWidths) + border * (columns + 1)
	outputHeight = sum(maxHeights) + border * (rows + 1)
	output = image.new('RGB', (outputWidth, outputHeight), color=bg)
	
	for y in range(rows):
		for x in range(columns):
			xpos = sum(maxWidths[0:x]) + border * (x + 1) + math.floor((maxWidths[x] - imgs[y][x].width) / 2)
			ypos = sum(maxHeights[0:y]) + border * (y + 1) + math.floor((maxHeights[y] - imgs[y][x].height) / 2)
			output.paste(imgs[y][x], (xpos, ypos))
	
	return output

def polyReg(x, y, m):
	
	X = list()
	
	for xi in x:
		X.append(list())
		
		for j in range(m + 1):
			X[-1].append(xi ** j)
	
	return numpy.dot(numpy.dot(numpy.linalg.inv(numpy.dot(numpy.transpose(numpy.array(X)), numpy.array(X))), numpy.transpose(numpy.array(X))), numpy.array(y)).tolist()

def linReg(x, y):
	
	return polyReg(x, y, 1)

def linPol(x1, x2, y1, y2, x):
	
	poly = linReg(list((x1, x2)), list((y1, y2)))
	return poly[0] + poly[1] * x

def coefDet(y, f):
	
	Sr = sum(list((yi - fi) ** 2 for yi, fi in zip(y, f)))
	St = sum(list((yi - sum(y) / len(y)) ** 2 for yi in y))
	return 1 - Sr / St

def polyFit(x, y, R2min, mmin=0, mmax=math.inf):
	
	m = mmin
	polys = list()
	R2s = list()
	
	while m < min(len(y), mmax + 1):
		poly = polyReg(x, y, m)
		f = list(sum(list(poly[i] * xi ** i for i in range(len(poly)))) for xi in x)
		R2 = coefDet(y, f)
		polys.append(poly)
		R2s.append(R2)
		
		if (R2 < 0 and m > 0) or R2 > R2min:
			break
		
		m += 1
	
	poly = polys[R2s.index(max(R2s))]
	R2 = max(R2s)
	return poly, R2

def polyPoint(poly, x):
	
	return sum(list(poly[i] * x ** i for i in range(len(poly))))

def polyRoots(poly):
	
	return list(i.real for i in numpy.polynomial.polynomial.polyroots(numpy.array(poly)).tolist() if i.imag == 0)

def polyDif(poly):
	
	if len(poly) == 1:
		return list((0,))
	
	return list(poly[i] * i for i in range(1, len(poly)))

def polyInt(poly, intercept):
	
	output = list(poly[i] / (i + 1) for i in range (len(poly)))
	output.insert(0, intercept)
	return output

if __name__ == '__main__':
	print('This is a support module.')
	input('Press ENTER to close: ')
