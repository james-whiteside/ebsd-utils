# -*- coding: utf-8 -*-

import math
import itertools
import sys
import os
import numpy
import utilities
import fileloader
import orientation
from phase import CrystalFamily, BravaisLattice
from geometry import Axis, AxisSet, euler_rotation_matrix, reduce_matrix, euler_angles, inverse_stereographic, forward_stereographic, rotation_angle, misrotation_matrix


def inverse_pole_figure_colour(v: tuple[float, float, float], lattice_type: BravaisLattice) -> tuple[float, float, float]:
    """
    This docstring is out of date.
    Reduces a lattice vector ``v`` into the fundamental unit triangle of its Bravais lattice by reflection.
    :param v: The lattice vector ``v``.
    :param lattice_type: The Bravais lattice type.
    :return: The reduced vector.
    """

    x, y, z = v
    crystal_family = lattice_type.family

    if crystal_family is CrystalFamily.NONE:
        a, b, c = z, y, x
    elif crystal_family is CrystalFamily.C:
        z, y, x = sorted((-abs(x), -abs(y), -abs(z)))
        a = z - y
        b = (y - x) * math.sqrt(2)
        c = x * math.sqrt(3)
        a, b, c = abs(a) / max(abs(a), abs(b), abs(c)), abs(b) / max(abs(a), abs(b), abs(c)), abs(c) / max(abs(a), abs(b), abs(c))
    else:
        raise NotImplementedError()

    return a, b, c


def keyIPF(lattice_type, size, guides):
    IPF = list()
    
    for Y in range(size):
        IPF.append(list())
        
        for X in range(size):
            x, y, z = inverse_stereographic(2 * X / size - 1, 2 * Y / size - 1)
            
            if math.sqrt((2 * X / size - 1) ** 2 + (2 * Y / size - 1) ** 2) > 1:
                IPF[Y].append(list((0, 0, 0)))
            elif guides and (round(abs(x), 2) == round(abs(y), 2) or round(abs(y), 2) == round(abs(z), 2) or round(abs(z), 2) == round(abs(x), 2) or X == size / 2 or Y == size / 2):
                IPF[Y].append(list((0, 0, 0)))
            else:
                v = (x, y, z)
                v = inverse_pole_figure_colour(v, lattice_type)
                IPF[Y].append(v)
    
    return IPF


def calV(R: numpy.ndarray, axis: Axis) -> tuple[float, float]:
    x, y, z = numpy.dot(R, numpy.array(axis.value)).tolist()
    X, Y = forward_stereographic(x, y, z)
    return X, Y


def mapSGP(data, metadata, size, plot, phaseID=None, trim=True):
    SGP = list()
    
    if plot == 'template':
        for Y in range(size):
            SGP.append(list())
            
            for X in range(size):
                SGP[Y].append(list((0, 0, 0)))
                
        for Y in range(size):
            for X in range(size):
                vX = math.tan(math.radians(22.5)) * X / size
                vY = math.tan(math.radians(22.5)) * Y / size
                vx, vy, vz = inverse_stereographic(vX, vY)
                
                if (abs(vx) > abs(vy) or abs(vy) > abs(vz) or abs(vz) < abs(vx)):
                    SGP[Y][X] = list((1, 1, 1))
    
    if plot == 'scheme':
        for Y in range(size):
            SGP.append(list())
            
            for X in range(size):
                SGP[Y].append(list((0, 0, 0)))
        
        for Y in range(size):
            for X in range(size):
                vX = math.tan(math.radians(22.5)) * X / (size - 1)
                vY = math.tan(math.radians(22.5)) * Y / (size - 1)
                v = inverse_stereographic(vX, vY)
                lattice_type = BravaisLattice.CP
                v = inverse_pole_figure_colour(v, lattice_type)
                try:
                    SGP[int(round((size - 1) * vY / math.tan(math.radians(22.5))))][int(round((size - 1) * vX / math.tan(math.radians(22.5))))] = list(v)
                except IndexError:
                    continue
    
    if plot == 'colour':
        for Y in range(size):
            SGP.append(list())
            
            for X in range(size):
                SGP[Y].append(list((0, 0, 0)))

        for y in range(data['height']):
            for x in range(data['width']):
                if data['phases'][data['data']['phase'][y][x]]['ID'] == 0:
                    continue
                
                if phaseID != None and data['phases'][data['data']['phase'][y][x]]['ID'] != phaseID:
                    continue
                
                vX, vY = data['data']['SGP'][y][x]
                v = inverse_stereographic(vX, vY)
                lattice_type = BravaisLattice(data['phases'][data['data']['phase'][y][x]]['type'])
                v = inverse_pole_figure_colour(v, lattice_type)
                try:
                    SGP[int(round((size - 1) * vY / math.tan(math.radians(22.5))))][int(round((size - 1) * vX / math.tan(math.radians(22.5))))] = list(v)
                except IndexError:
                    continue
    
    if plot == 'count':
        counts = list()
        
        for Y in range(size):
            SGP.append(list())
            counts.append(list())
            
            for X in range(size):
                SGP[Y].append(list((1, 0, 0)))
                counts[Y].append(0)
        
        for y in range(data['height']):
            for x in range(data['width']):
                if data['phases'][data['data']['phase'][y][x]]['ID'] == 0:
                    continue
                
                if phaseID != None and data['phases'][data['data']['phase'][y][x]]['ID'] != phaseID:
                    continue
                
                vX, vY = data['data']['SGP'][y][x]
                try:
                    counts[int(round((size - 1) * vY / math.tan(math.radians(22.5))))][int(round((size - 1) * vX / math.tan(math.radians(22.5))))] += 1
                except IndexError:
                    continue
        
        maxCount = max(list(max(row) for row in counts))
        
        if maxCount == 0:
            return SGP
        
        for Y in range(size):
            for X in range(size):
                if counts[Y][X] != 0:
                    if maxCount == 1:
                        count = 1
                    else:
                        count = math.log10(counts[Y][X]) / math.log10(maxCount)
                    SGP[Y][X] = list((count, count, count))
    
    if plot == 'PQ':
        counts = list()
        totals = list()
        
        for Y in range(size):
            SGP.append(list())
            counts.append(list())
            totals.append(list())
            
            for X in range(size):
                SGP[Y].append(list((1, 0, 0)))
                counts[Y].append(0)
                totals[Y].append(0)
        
        for y in range(data['height']):
            for x in range(data['width']):
                if data['phases'][data['data']['phase'][y][x]]['ID'] == 0:
                    continue
                
                if phaseID != None and data['phases'][data['data']['phase'][y][x]]['ID'] != phaseID:
                    continue
                
                vX, vY = data['data']['SGP'][y][x]
                try:
                    counts[int(round((size - 1) * vY / math.tan(math.radians(22.5))))][int(round((size - 1) * vX / math.tan(math.radians(22.5))))] += 1
                    totals[int(round((size - 1) * vY / math.tan(math.radians(22.5))))][int(round((size - 1) * vX / math.tan(math.radians(22.5))))] += data['data']['PQ'][y][x] / 100
                except:
                    continue
        
        for Y in range(size):
            for X in range(size):
                if counts[Y][X] != 0:
                    average = totals[Y][X] / counts[Y][X]
                    SGP[Y][X] = list((average, average, average))
    
    if plot == 'IQ':
        counts = list()
        totals = list()
        
        for Y in range(size):
            SGP.append(list())
            counts.append(list())
            totals.append(list())
            
            for X in range(size):
                SGP[Y].append(list((1, 0, 0)))
                counts[Y].append(0)
                totals[Y].append(0)
        
        for y in range(data['height']):
            for x in range(data['width']):
                if data['phases'][data['data']['phase'][y][x]]['ID'] == 0:
                    continue
                
                if phaseID != None and data['phases'][data['data']['phase'][y][x]]['ID'] != phaseID:
                    continue
                
                vX, vY = data['data']['SGP'][y][x]
                try:
                    counts[int(round((size - 1) * vY / math.tan(math.radians(22.5))))][int(round((size - 1) * vX / math.tan(math.radians(22.5))))] += 1
                    totals[int(round((size - 1) * vY / math.tan(math.radians(22.5))))][int(round((size - 1) * vX / math.tan(math.radians(22.5))))] += data['data']['IQ'][y][x] / 100
                except IndexError:
                    continue
        
        for Y in range(size):
            for X in range(size):
                if counts[Y][X] != 0:
                    average = totals[Y][X] / counts[Y][X]
                    SGP[Y][X] = list((average, average, average))
    
    if plot == 'KAM':
        counts = list()
        totals = list()
        
        for Y in range(size):
            SGP.append(list())
            counts.append(list())
            totals.append(list())
            
            for X in range(size):
                SGP[Y].append(list((1, 0, 0)))
                counts[Y].append(0)
                totals[Y].append(0)
        
        for y in range(data['height']):
            for x in range(data['width']):
                if data['phases'][data['data']['phase'][y][x]]['ID'] == 0:
                    continue
                
                if phaseID != None and data['phases'][data['data']['phase'][y][x]]['ID'] != phaseID:
                    continue
                
                vX, vY = data['data']['SGP'][y][x]
                try:
                    counts[int(round((size - 1) * vY / math.tan(math.radians(22.5))))][int(round((size - 1) * vX / math.tan(math.radians(22.5))))] += 1
                    totals[int(round((size - 1) * vY / math.tan(math.radians(22.5))))][int(round((size - 1) * vX / math.tan(math.radians(22.5))))] += data['data']['KAM'][y][x] / data['maxKAM']
                except IndexError:
                    continue
        
        for Y in range(size):
            for X in range(size):
                if counts[Y][X] != 0:
                    average = totals[Y][X] / counts[Y][X]
                    SGP[Y][X] = list((average, average, average))
    
    if plot == 'GND':
        counts = list()
        totals = list()
        
        for Y in range(size):
            SGP.append(list())
            counts.append(list())
            totals.append(list())
            
            for X in range(size):
                SGP[Y].append(list((1, 0, 0)))
                counts[Y].append(0)
                totals[Y].append(0)
        
        for y in range(data['height']):
            for x in range(data['width']):
                if data['phases'][data['data']['phase'][y][x]]['ID'] == 0:
                    continue
                
                if phaseID != None and data['phases'][data['data']['phase'][y][x]]['ID'] != phaseID:
                    continue
                
                vX, vY = data['data']['SGP'][y][x]
                try:
                    counts[int(round((size - 1) * vY / math.tan(math.radians(22.5))))][int(round((size - 1) * vX / math.tan(math.radians(22.5))))] += 1
                    totals[int(round((size - 1) * vY / math.tan(math.radians(22.5))))][int(round((size - 1) * vX / math.tan(math.radians(22.5))))] += (data['data']['GND'][y][x] - data['minGND']) / (data['maxGND'] - data['minGND'])
                except IndexError:
                    continue
        
        for Y in range(size):
            for X in range(size):
                if counts[Y][X] != 0:
                    average = totals[Y][X] / counts[Y][X]
                    SGP[Y][X] = list((average, average, average))
    
    if plot == 'CF':
        counts = list()
        totals = list()
        
        for Y in range(size):
            SGP.append(list())
            counts.append(list())
            totals.append(list())
            
            for X in range(size):
                SGP[Y].append(list((1, 0, 0)))
                counts[Y].append(0)
                totals[Y].append(0)
        
        for y in range(data['height']):
            for x in range(data['width']):
                if data['phases'][data['data']['phase'][y][x]]['ID'] == 0:
                    continue
                
                if phaseID != None and data['phases'][data['data']['phase'][y][x]]['ID'] != phaseID:
                    continue
                
                vX, vY = data['data']['SGP'][y][x]
                try:
                    counts[int(round((size - 1) * vY / math.tan(math.radians(22.5))))][int(round((size - 1) * vX / math.tan(math.radians(22.5))))] += 1
                    totals[int(round((size - 1) * vY / math.tan(math.radians(22.5))))][int(round((size - 1) * vX / math.tan(math.radians(22.5))))] += data['data']['CF'][y][x] / 100
                except IndexError:
                    continue
        
        for Y in range(size):
            for X in range(size):
                if counts[Y][X] != 0:
                    average = totals[Y][X] / counts[Y][X]
                    SGP[Y][X] = list((average, average, average))
    
    if plot == 'phase':
        phaseIDs = sorted(list(phase['ID'] for phase in data['phases'].values() if phase['ID'] != 0))
        pID = list()
        
        for Y in range(size):
            SGP.append(list())
            pID.append(list())
            
            for X in range(size):
                SGP[Y].append(list((0, 0, 0)))
                pID[Y].append(0)
        
        for y in range(data['height']):
            for x in range(data['width']):
                if data['phases'][data['data']['phase'][y][x]]['ID'] == 0:
                    continue
                
                if phaseID != None and data['phases'][data['data']['phase'][y][x]]['ID'] != phaseID:
                    continue
                
                vX, vY = data['data']['SGP'][y][x]
                
                try:
                    if pID[int(round((size - 1) * vY / math.tan(math.radians(22.5))))][int(round((size - 1) * vX / math.tan(math.radians(22.5))))] == 0:
                        pID[int(round((size - 1) * vY / math.tan(math.radians(22.5))))][int(round((size - 1) * vX / math.tan(math.radians(22.5))))] = phaseIDs.index(data['phases'][data['data']['phase'][y][x]]['ID']) + 1
                    elif pID[int(round((size - 1) * vY / math.tan(math.radians(22.5))))][int(round((size - 1) * vX / math.tan(math.radians(22.5))))] != phaseIDs.index(data['phases'][data['data']['phase'][y][x]]['ID']) + 1:
                        pID[int(round((size - 1) * vY / math.tan(math.radians(22.5))))][int(round((size - 1) * vX / math.tan(math.radians(22.5))))] = -1
                except IndexError:
                    continue
        
        for Y in range(size):
            for X in range(size):
                if pID[Y][X] == 0:
                    SGP[Y][X] = list((0, 0, 0))
                elif pID[Y][X] == -1:
                    SGP[Y][X] = list((1, 1, 1))
                else:
                    SGP[Y][X] = utilities.colour_wheel(pID[Y][X] - 1, len(phaseIDs))
    
    if plot == 'cluster':
        cID = list()
        
        for Y in range(size):
            SGP.append(list())
            cID.append(list())
            
            for X in range(size):
                SGP[Y].append(list((0, 0, 0)))
                cID[Y].append(0)
        
        for y in range(data['height']):
            for x in range(data['width']):
                if data['phases'][data['data']['phase'][y][x]]['ID'] == 0:
                    continue
                
                if phaseID != None and data['phases'][data['data']['phase'][y][x]]['ID'] != phaseID:
                    continue
                
                vX, vY = data['data']['SGP'][y][x]
                
                try:
                    if cID[int(round((size - 1) * vY / math.tan(math.radians(22.5))))][int(round((size - 1) * vX / math.tan(math.radians(22.5))))] == 0:
                        cID[int(round((size - 1) * vY / math.tan(math.radians(22.5))))][int(round((size - 1) * vX / math.tan(math.radians(22.5))))] = data['data']['cID'][y][x]
                    elif cID[int(round((size - 1) * vY / math.tan(math.radians(22.5))))][int(round((size - 1) * vX / math.tan(math.radians(22.5))))] != data['data']['cID'][y][x]:
                        cID[int(round((size - 1) * vY / math.tan(math.radians(22.5))))][int(round((size - 1) * vX / math.tan(math.radians(22.5))))] = -1
                except IndexError:
                    continue
        
        for Y in range(size):
            for X in range(size):
                if cID[Y][X] == 0:
                    SGP[Y][X] = list((0, 0, 0))
                elif cID[Y][X] == -1:
                    SGP[Y][X] = list((1, 1, 1))
                else:
                    SGP[Y][X] = utilities.colour_wheel(cID[Y][X] - 1, metadata[data['fileref']]['k'])
    
    if trim:
        for Y in range(size):
            for X in range(size):
                vX = math.tan(math.radians(22.5)) * X / size
                vY = math.tan(math.radians(22.5)) * Y / size
                vx, vy, vz = inverse_stereographic(vX, vY)
                
                if (abs(vx) > abs(vy) or abs(vy) > abs(vz) or abs(vz) < abs(vx)):
                    SGP[Y][X] = list((1, 1, 1))
    
    return SGP


def crunch(data):
    if data['width'] % 2 != 0 or data['height'] % 2 != 0:
        input('Cannot crunch data due to odd valued size.')
        sys.exit()
    
    output = dict()
    output['fileref'] = data['fileref']
    output['phases'] = data['phases']
    output['width'] = data['width'] // 2
    output['height'] = data['height'] // 2
    output['data'] = dict()
    output['data']['phase'] = list()
    output['data']['euler'] = list()
    output['data']['IQ'] = list()
    output['data']['PQ'] = list()
    
    for y in range(output['height']):
        output['data']['phase'].append(list())
        output['data']['euler'].append(list())
        output['data']['IQ'].append(list())
        output['data']['PQ'].append(list())
        
        for x in range(output['width']):
            pixels = list(((2 * x, 2 * y), (2 * x + 1, 2 * y), (2 * x, 2 * y + 1), (2 * x + 1, 2 * y + 1)))
            phases = set(data['data']['phase'][pixel[1]][pixel[0]] for pixel in pixels if data['phases'][data['data']['phase'][pixel[1]][pixel[0]]]['ID'] != 0)
            
            if len(phases) != 1:
                output['data']['phase'][y].append(0)
                output['data']['euler'][y].append(list((0.0, 0.0, 0.0)))
                output['data']['IQ'][y].append(0.0)
                output['data']['PQ'][y].append(0.0)
                continue
            
            phase = list(phases)[0]
            count = 0
            R = numpy.zeros((3, 3))
            IQ = 0
            PQ = 0
            
            for pixel in pixels:
                if data['phases'][data['data']['phase'][pixel[1]][pixel[0]]]['ID'] != 0:
                    count += 1
                    R += reduce_matrix(euler_rotation_matrix(AxisSet.ZXZ, data['data']['euler'][pixel[1]][pixel[0]]), CrystalFamily.C)
                    IQ += data['data']['IQ'][pixel[1]][pixel[0]]
                    PQ += data['data']['PQ'][pixel[1]][pixel[0]]
            
            U, S, VT = numpy.linalg.svd(R / count)
            R = numpy.dot(U, VT)
            euler = euler_angles(reduce_matrix(R, CrystalFamily.C), AxisSet.ZXZ)
            IQ = IQ / count
            PQ = PQ / 4
            output['data']['phase'][y].append(phase)
            output['data']['euler'][y].append(euler)
            output['data']['IQ'][y].append(IQ)
            output['data']['PQ'][y].append(PQ)
    
    return output


def crunches(data, n):
    if n <= 0:
        return data
    else:
        return crunch(crunches(data, n - 1))


def summarise(path):
    filepaths = utilities.get_file_paths(directory_path=utilities.get_directory_path(path + '/analyses'), recursive=True, extension='csv')
    metadata = fileloader.getNyeMetadata(path + '/metadata.csv')
    print()
    
    for filepath in filepaths:
        data = fileloader.getNyeAnalysis(filepath)
        print('Making summaries for q' + data['fileref'] + '.')
        data['data']['x'] = dict()
        data['data']['y'] = dict()
        data['data']['x']['IQ'] = list()
        data['data']['x']['PQ'] = list()
        data['data']['y']['IQ'] = list()
        data['data']['y']['PQ'] = list()
        
        if 'd' in metadata[data['fileref']]['aType']:
            data['data']['k'] = dict()
            data['data']['k']['phase'] = list()
            data['data']['k']['euler'] = list()
            data['data']['k']['R'] = list()
            data['data']['k']['SGP'] = list()
            data['data']['k']['IQ'] = list()
            data['data']['k']['PQ'] = list()
            kCounts = list()
        
        if 'g' in metadata[data['fileref']]['aType']:
            data['data']['x']['KAM'] = list()
            data['data']['x']['GND'] = list()
            xGND = list()
            data['data']['y']['KAM'] = list()
            data['data']['y']['GND'] = list()
            yGND = list()
            
            if 'd' in metadata[data['fileref']]['aType']:
                data['data']['k']['KAM'] = list()
                data['data']['k']['GND'] = list()
                kGND = list()
        
        if 'c' in metadata[data['fileref']]['aType']:
            data['data']['x']['CF'] = list()
            data['data']['y']['CF'] = list()
            
            if 'd' in metadata[data['fileref']]['aType']:
                data['data']['k']['CF'] = list()
        
        for x in range(data['width']):
            data['data']['x']['IQ'].append(0)
            data['data']['x']['PQ'].append(0)
            
            if 'g' in metadata[data['fileref']]['aType']:
                data['data']['x']['KAM'].append(0)
                xGND.append(0)
            
            if 'c' in metadata[data['fileref']]['aType']:
                data['data']['x']['CF'].append(0)
        
        for y in range(data['height']):
            data['data']['y']['IQ'].append(0)
            data['data']['y']['PQ'].append(0)
            
            if 'g' in metadata[data['fileref']]['aType']:
                data['data']['y']['KAM'].append(0)
                yGND.append(0)
            
            if 'c' in metadata[data['fileref']]['aType']:
                data['data']['y']['CF'].append(0)
        
        if 'd' in metadata[data['fileref']]['aType']:
            for k in range(metadata[data['fileref']]['k'] + 1):
                data['data']['k']['phase'].append(0)
                data['data']['k']['euler'].append(list((0, 0, 0)))
                data['data']['k']['R'].append(numpy.zeros((3, 3)))
                data['data']['k']['SGP'].append(list((0, 0)))
                data['data']['k']['IQ'].append(0)
                data['data']['k']['PQ'].append(0)
                kCounts.append(0)
                
                if 'g' in metadata[data['fileref']]['aType']:
                    data['data']['k']['KAM'].append(0)
                    kGND.append(0)
                
                if 'c' in metadata[data['fileref']]['aType']:
                    data['data']['k']['CF'].append(0)
        
        for x in range(data['width']):
            for y in range(data['height']):
                data['data']['x']['IQ'][x] += data['data']['IQ'][y][x] / data['height']
                data['data']['x']['PQ'][x] += data['data']['PQ'][y][x] / data['height']
                data['data']['y']['IQ'][y] += data['data']['IQ'][y][x] / data['width']
                data['data']['y']['PQ'][y] += data['data']['PQ'][y][x] / data['width']
                
                if 'g' in metadata[data['fileref']]['aType']:
                    data['data']['x']['KAM'][x] += data['data']['KAM'][y][x] / data['height']
                    xGND[x] += 10 ** data['data']['GND'][y][x] / data['height']
                    data['data']['y']['KAM'][y] += data['data']['KAM'][y][x] / data['width']
                    yGND[y] += 10 ** data['data']['GND'][y][x] / data['width']
                
                if 'c' in metadata[data['fileref']]['aType']:
                    data['data']['x']['CF'][x] += data['data']['CF'][y][x] / data['height']
                    data['data']['y']['CF'][y] += data['data']['CF'][y][x] / data['width']
                
                if 'd' in metadata[data['fileref']]['aType']:
                    kCounts[data['data']['cID'][y][x]] += 1
                    data['data']['k']['phase'][data['data']['cID'][y][x]] = data['data']['phase'][y][x]
                    data['data']['k']['R'][data['data']['cID'][y][x]] += reduce_matrix(euler_rotation_matrix(
                        AxisSet.ZXZ, data['data']['euler'][y][x]), CrystalFamily.C)
                    data['data']['k']['IQ'][data['data']['cID'][y][x]] += data['data']['IQ'][y][x]
                    data['data']['k']['PQ'][data['data']['cID'][y][x]] += data['data']['PQ'][y][x]
                    
                    if 'g' in metadata[data['fileref']]['aType']:
                        data['data']['k']['KAM'][data['data']['cID'][y][x]] += data['data']['KAM'][y][x]
                        kGND[data['data']['cID'][y][x]] += 10 ** data['data']['GND'][y][x]
                    
                    if 'c' in metadata[data['fileref']]['aType']:
                        data['data']['k']['CF'][data['data']['cID'][y][x]] += data['data']['CF'][y][x]
        
        if 'd' in metadata[data['fileref']]['aType']:
            data['data']['k']['phase'][0] = 0
            
            for k in range(metadata[data['fileref']]['k'] + 1):
                if k != 0:
                    U, S, VT = numpy.linalg.svd(data['data']['k']['R'][k] / kCounts[k])
                    data['data']['k']['R'][k] = numpy.dot(U, VT)
                    phi1, Phi, phi2 = euler_angles(data['data']['k']['R'][k], AxisSet.ZXZ)
                    data['data']['k']['euler'][k] = list((phi1, Phi, phi2))
                    R = reduce_matrix(euler_rotation_matrix(AxisSet.ZXZ, data['data']['k']['euler'][k]), CrystalFamily.C)
                    vX, vY = calV(R, Axis.Z)
                    data['data']['k']['SGP'][k] = list((vX, vY))
                
                data['data']['k']['IQ'][k] /= kCounts[k]
                data['data']['k']['PQ'][k] /= kCounts[k]
                
                if 'g' in metadata[data['fileref']]['aType']:
                    data['data']['k']['KAM'][k] /= kCounts[k]
                    kGND[k] /= kCounts[k]
                
                if 'c' in metadata[data['fileref']]['aType']:
                    data['data']['k']['CF'][k] /= kCounts[k]
        
        if 'g' in metadata[data['fileref']]['aType']:
            for x in range(data['width']):
                data['data']['x']['GND'].append(math.log10(xGND[x]))
            
            for y in range(data['height']):
                data['data']['y']['GND'].append(math.log10(yGND[y]))
            
            if 'd' in metadata[data['fileref']]['aType']:
                for k in range(metadata[data['fileref']]['k'] + 1):
                    data['data']['k']['GND'].append(math.log10(kGND[k]))
        
        if 'd' in metadata[data['fileref']]['aType']:
            materials = fileloader.get_materials()
            variants = fileloader.getVariantList()
            twins = fileloader.getTwinList()
            matches = list()
            
            for k1 in range(1, metadata[data['fileref']]['k'] + 1):
                for k2 in range(1, metadata[data['fileref']]['k'] + 1):
                    if k1 == k2:
                        continue
                    
                    for variant in variants:
                        if variants[variant]['lTypes'][0] == data['phases'][data['data']['k']['phase'][k1]]['type'] and variants[variant]['lTypes'][1] == data['phases'][data['data']['k']['phase'][k2]]['type']:
                            match = dict()
                            match['variant'] = variant
                            match['k1'] = k1
                            match['k2'] = k2
                            params = list((materials[data['phases'][data['data']['k']['phase'][k1]]['ID']].lattice_constants, materials[data['phases'][data['data']['k']['phase'][k2]]['ID']].lattice_constants))
                            R1 = data['data']['k']['R'][k1]
                            R2 = data['data']['k']['R'][k2]
                            polarity = sorted(list(set(list(itertools.permutations(list((1, 1, 1, 1, -1, -1, -1, -1)), 4)))), reverse=True)
                            theta = 2 * math.pi
                            
                            for i in range(16):
                                vectors = list((list((list(polarity[i][0] * hkl for hkl in variants[variant]['vectors'][0][0]), list(polarity[i][1] * hkl for hkl in variants[variant]['vectors'][0][1]))), list((list(polarity[i][2] * hkl for hkl in variants[variant]['vectors'][1][0]), list(polarity[i][3] * hkl for hkl in variants[variant]['vectors'][1][1])))))
                                J = orientation.get_relationship_matrix(vectors[0][0], vectors[0][1], vectors[1][0], vectors[1][1], params[0], params[1])
                                s = math.sqrt(params[0][0] ** 2 + params[0][1] ** 2 + params[0][2] ** 2) / math.sqrt(params[1][0] ** 2 + params[1][1] ** 2 + params[1][2] ** 2)
                                RF = numpy.dot(J / s, R1)
                                dR = misrotation_matrix(RF, R2)
                                theta = min(rotation_angle(dR), theta)
                            
                            match['dTheta'] = theta
                            match['cosine'] = math.cos(theta)
                            matches.append(match)
                    
                    for variant in twins:
                        if data['phases'][data['data']['k']['phase'][k1]]['type'] == data['phases'][data['data']['k']['phase'][k2]]['type'] and k1 < k2:
                            match = dict()
                            match['variant'] = variant
                            match['k1'] = k1
                            match['k2'] = k2
                            R1 = data['data']['k']['R'][k1]
                            R2 = data['data']['k']['R'][k2]
                            theta = 2 * math.pi
                            family = orientation.get_plane_family(twins[variant])
                            
                            for plane in family:
                                J = orientation.get_twin_matrix(plane)
                                RF = numpy.dot(J, R1)
                                dR = misrotation_matrix(RF, R2)
                                theta = min(rotation_angle(dR), theta)
                            
                            match['dTheta'] = theta
                            match['cosine'] = math.cos(theta)
                            matches.append(match)
        
        with open(path + '/summaries/x' + data['fileref'] + '.csv', 'w') as output:
            output.write('Phases:\n')
            
            for localID in data['phases']:
                output.write(str(localID) + ',' + data['phases'][localID]['name'] + ',' + str(data['phases'][localID]['ID']) + '\n')
            
            output.write('Map Size:\n')
            output.write('X,' + str(data['width']) + '\n')
            output.write('Y,' + str(data['height']) + '\n')
            output.write('Data:\n')
            output.write('X,Index Quality,Pattern Quality')
            
            if 'g' in metadata[data['fileref']]['aType']:
                output.write(',Kernel Average Misorientation,GND Density')
            
            if 'c' in metadata[data['fileref']]['aType']:
                output.write(',Channelling Fraction')
            
            output.write('\n')
            
            for x in range(data['width']):
                output.write(str(x) + ',')
                output.write(str(data['data']['x']['IQ'][x]) + ',')
                output.write(str(data['data']['x']['PQ'][x]))
                
                if 'g' in metadata[data['fileref']]['aType']:
                    output.write(',' + str(data['data']['x']['KAM'][x]))
                    output.write(',' + str(data['data']['x']['GND'][x]))
                
                if 'c' in metadata[data['fileref']]['aType']:
                    output.write(',' + str(data['data']['x']['CF'][x]))
                
                output.write('\n')
        
        with open(path + '/summaries/y' + data['fileref'] + '.csv', 'w') as output:
            output.write('Phases:\n')
            
            for localID in data['phases']:
                output.write(str(localID) + ',' + data['phases'][localID]['name'] + ',' + str(data['phases'][localID]['ID']) + '\n')
            
            output.write('Map Size:\n')
            output.write('X,' + str(data['width']) + '\n')
            output.write('Y,' + str(data['height']) + '\n')
            output.write('Data:\n')
            output.write('Y,Index Quality,Pattern Quality')
            
            if 'g' in metadata[data['fileref']]['aType']:
                output.write(',Kernel Average Misorientation,GND Density')
            
            if 'c' in metadata[data['fileref']]['aType']:
                output.write(',Channelling Fraction')
            
            output.write('\n')
            
            for y in range(data['height']):
                output.write(str(y) + ',')
                output.write(str(data['data']['y']['IQ'][y]) + ',')
                output.write(str(data['data']['y']['PQ'][y]))
                
                if 'g' in metadata[data['fileref']]['aType']:
                    output.write(',' + str(data['data']['y']['KAM'][y]))
                    output.write(',' + str(data['data']['y']['GND'][y]))
                
                if 'c' in metadata[data['fileref']]['aType']:
                    output.write(',' + str(data['data']['y']['CF'][y]))
                
                output.write('\n')
        
        if 'd' in metadata[data['fileref']]['aType']:
            with open(path + '/summaries/k' + data['fileref'] + '.csv', 'w') as output:
                output.write('Phases:\n')
                
                for localID in data['phases']:
                    output.write(str(localID) + ',' + data['phases'][localID]['name'] + ',' + str(data['phases'][localID]['ID']) + '\n')
                
                output.write('Map Size:\n')
                output.write('X,' + str(data['width']) + '\n')
                output.write('Y,' + str(data['height']) + '\n')
                output.write('Data:\n')
                output.write('K,Cluster Size,Phase,Euler1,Euler2,Euler3,Index Quality,Pattern Quality,IPF x-coordinate,IPF y-coordinate')
                
                if 'g' in metadata[data['fileref']]['aType']:
                    output.write(',Kernel Average Misorientation,GND Density')
                
                if 'c' in metadata[data['fileref']]['aType']:
                    output.write(',Channelling Fraction')
                
                output.write('\n')
                
                for k in range(metadata[data['fileref']]['k'] + 1):
                    output.write(str(k) + ',')
                    output.write(str(kCounts[k]) + ',')
                    output.write(str(data['data']['k']['phase'][k]) + ',')
                    output.write(str(math.degrees(data['data']['k']['euler'][k][0])) + ',')
                    output.write(str(math.degrees(data['data']['k']['euler'][k][1])) + ',')
                    output.write(str(math.degrees(data['data']['k']['euler'][k][2])) + ',')
                    output.write(str(data['data']['k']['IQ'][k]) + ',')
                    output.write(str(data['data']['k']['PQ'][k]) + ',')
                    output.write(str(data['data']['k']['SGP'][k][0]) + ',')
                    output.write(str(data['data']['k']['SGP'][k][1]))
                    
                    if 'g' in metadata[data['fileref']]['aType']:
                        output.write(',' + str(data['data']['k']['KAM'][k]))
                        output.write(',' + str(data['data']['k']['GND'][k]))
                    
                    if 'c' in metadata[data['fileref']]['aType']:
                        output.write(',' + str(data['data']['k']['CF'][k]))
                    
                    output.write('\n')
                
                output.write('Orientation Relationships:\n')
                output.write('Variant,K1,K2,Misorientation,Projection\n')
                
                for match in sorted(matches, key=lambda item: math.degrees(item['dTheta'])):
                    output.write(match['variant'] + ',')
                    output.write(str(match['k1']) + ',')
                    output.write(str(match['k2']) + ',')
                    output.write(str(math.degrees(match['dTheta'])) + ',')
                    output.write(str(match['cosine']) + '\n')
    
    print()
    print('All summaries complete.')
    input('Press ENTER to close: ')


def makeMaps(path, size=None):
    filepaths = utilities.get_file_paths(directory_path=utilities.get_directory_path(path + '/analyses'), recursive=True, extension='csv')
    metadata = fileloader.getNyeMetadata(path + '/metadata.csv')
    sizeOverride = size
    print()
    print('Processing data.')
    
    for filepath in filepaths:
        data = fileloader.getNyeAnalysis(filepath)
        
        minKAM = math.inf
        maxKAM = 0
        minGND = math.inf
        maxGND = 0
            
        if 'g' in metadata[data['fileref']]['aType']:
            KAMs = sorted(sum(list(list(data['data']['KAM'][y][x] for x in range(data['width']) if data['data']['KAM'][y][x] != 0) for y in range(data['height'])), list()))
            GNDs = sorted(sum(list(list(data['data']['GND'][y][x] for x in range(data['width']) if data['data']['GND'][y][x] != 0) for y in range(data['height'])), list()))
            
            if len(KAMs) != 0:
                minKAM = min(minKAM, KAMs[0])
                maxKAM = min(maxKAM, KAMs[-1])
            
            if len(GNDs) != 0:
                minGND = min(minGND, GNDs[0])
                maxGND = max(maxGND, GNDs[-1])
        
        if minKAM == math.inf or maxKAM == 0:
            minKAM = 0
            maxKAM = 1
        
        if minGND == math.inf or maxGND == 0:
            minGND = 0
            maxGND = 1
        
        if 'g' in metadata[data['fileref']]['aType']:
            data['minKAM'] = minKAM
            data['maxKAM'] = maxKAM
            data['minGND'] = minGND
            data['maxGND'] = maxGND
        
        if 'd' in metadata[data['fileref']]['aType']:
            data['keyWidth'] = min(50 * metadata[data['fileref']]['k'], 250)
            data['keyHeight'] = 50 * (int(math.floor(metadata[data['fileref']]['k'] / 5)) + 1)
        
        if sizeOverride == None:
            size = data['width']
        else:
            size = sizeOverride
        
        print('Making maps for q' + data['fileref'] + '.')
        
        try:
            os.mkdir(path + '/maps/' + data['fileref'])
        except FileExistsError:
            pass
        
        utilities.make_image(mapSGP(data, metadata, size, 'colour'), size, size, 1, 'RGB').save(path + '/maps/' + data['fileref'] + '/IPF.png')
        utilities.make_image(mapSGP(data, metadata, size, 'count'), size, size, 1, 'RGB').save(path + '/maps/' + data['fileref'] + '/IPF-Q.png')
        utilities.make_image(mapSGP(data, metadata, size, 'phase'), size, size, 1, 'RGB').save(path + '/maps/' + data['fileref'] + '/IPF-P.png')
        utilities.make_image(mapSGP(data, metadata, size, 'PQ'), size, size, 1, 'RGB').save(path + '/maps/' + data['fileref'] + '/IPF-PQ.png')
        utilities.make_image(mapSGP(data, metadata, size, 'IQ'), size, size, 1, 'RGB').save(path + '/maps/' + data['fileref'] + '/IPF-IQ.png')
        
        if 'g' in metadata[data['fileref']]['aType']:
            utilities.make_image(mapSGP(data, metadata, size, 'KAM'), size, size, 1, 'RGB').save(path + '/maps/' + data['fileref'] + '/IPF-KAM.png')
            utilities.make_image(mapSGP(data, metadata, size, 'GND'), size, size, 1, 'RGB').save(path + '/maps/' + data['fileref'] + '/IPF-GND.png')
        
        if 'c' in metadata[data['fileref']]['aType']:
            utilities.make_image(mapSGP(data, metadata, size, 'CF'), size, size, 1, 'RGB').save(path + '/maps/' + data['fileref'] + '/IPF-CF.png')
        
        if 'd' in metadata[data['fileref']]['aType']:
            utilities.make_image(mapSGP(data, metadata, size, 'cluster'), size, size, 1, 'RGB').save(path + '/maps/' + data['fileref'] + '/IPF-C.png')
        
        for localID in data['phases']:
            if data['phases'][localID]['ID'] == 0:
                continue
            else:
                phaseID = data['phases'][localID]['ID']
                utilities.make_image(mapSGP(data, metadata, size, 'colour', phaseID=phaseID), size, size, 1, 'RGB').save(path + '/maps/' + data['fileref'] + '/IPF[' + str(phaseID) + '].png')
                utilities.make_image(mapSGP(data, metadata, size, 'count', phaseID=phaseID), size, size, 1, 'RGB').save(path + '/maps/' + data['fileref'] + '/IPF-Q[' + str(phaseID) + '].png')
                utilities.make_image(mapSGP(data, metadata, size, 'phase', phaseID=phaseID), size, size, 1, 'RGB').save(path + '/maps/' + data['fileref'] + '/IPF-P[' + str(phaseID) + '].png')
                utilities.make_image(mapSGP(data, metadata, size, 'PQ', phaseID=phaseID), size, size, 1, 'RGB').save(path + '/maps/' + data['fileref'] + '/IPF-PQ[' + str(phaseID) + '].png')
                utilities.make_image(mapSGP(data, metadata, size, 'IQ', phaseID=phaseID), size, size, 1, 'RGB').save(path + '/maps/' + data['fileref'] + '/IPF-IQ[' + str(phaseID) + '].png')
                
                if 'g' in metadata[data['fileref']]['aType']:
                    utilities.make_image(mapSGP(data, metadata, size, 'KAM', phaseID=phaseID), size, size, 1, 'RGB').save(path + '/maps/' + data['fileref'] + '/IPF-KAM[' + str(phaseID) + '].png')
                    utilities.make_image(mapSGP(data, metadata, size, 'GND', phaseID=phaseID), size, size, 1, 'RGB').save(path + '/maps/' + data['fileref'] + '/IPF-GND[' + str(phaseID) + '].png')
                
                if 'c' in metadata[data['fileref']]['aType']:
                    utilities.make_image(mapSGP(data, metadata, size, 'CF', phaseID=phaseID), size, size, 1, 'RGB').save(path + '/maps/' + data['fileref'] + '/IPF-CF[' + str(phaseID) + '].png')
                
                if 'd' in metadata[data['fileref']]['aType']:
                    utilities.make_image(mapSGP(data, metadata, size, 'cluster', phaseID=phaseID), size, size, 1, 'RGB').save(path + '/maps/' + data['fileref'] + '/IPF-C[' + str(phaseID) + '].png')
    
    print()
    print('All maps complete.')
    input('Press ENTER to close: ')
