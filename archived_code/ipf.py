# -*- coding: utf-8 -*-

import math
import os
import fileloader
from src.data_structures.phase import CrystalFamily, BravaisLattice
from src.utilities.geometry import inverse_stereographic
from src.utilities.utilities import colour_wheel, get_directory_path, get_file_paths, make_image


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
                    SGP[Y][X] = colour_wheel(pID[Y][X] - 1, len(phaseIDs))
    
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
                    SGP[Y][X] = colour_wheel(cID[Y][X] - 1, metadata[data['fileref']]['k'])
    
    if trim:
        for Y in range(size):
            for X in range(size):
                vX = math.tan(math.radians(22.5)) * X / size
                vY = math.tan(math.radians(22.5)) * Y / size
                vx, vy, vz = inverse_stereographic(vX, vY)
                
                if (abs(vx) > abs(vy) or abs(vy) > abs(vz) or abs(vz) < abs(vx)):
                    SGP[Y][X] = list((1, 1, 1))
    
    return SGP


def makeMaps(path, size=None):
    filepaths = get_file_paths(directory_path=get_directory_path(path + '/analyses'), recursive=True, extension='csv')
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
        
        make_image(mapSGP(data, metadata, size, 'colour'), size, size, 1, 'RGB').save(path + '/maps/' + data['fileref'] + '/IPF.png')
        make_image(mapSGP(data, metadata, size, 'count'), size, size, 1, 'RGB').save(path + '/maps/' + data['fileref'] + '/IPF-Q.png')
        make_image(mapSGP(data, metadata, size, 'phase'), size, size, 1, 'RGB').save(path + '/maps/' + data['fileref'] + '/IPF-P.png')
        make_image(mapSGP(data, metadata, size, 'PQ'), size, size, 1, 'RGB').save(path + '/maps/' + data['fileref'] + '/IPF-PQ.png')
        make_image(mapSGP(data, metadata, size, 'IQ'), size, size, 1, 'RGB').save(path + '/maps/' + data['fileref'] + '/IPF-IQ.png')
        
        if 'g' in metadata[data['fileref']]['aType']:
            make_image(mapSGP(data, metadata, size, 'KAM'), size, size, 1, 'RGB').save(path + '/maps/' + data['fileref'] + '/IPF-KAM.png')
            make_image(mapSGP(data, metadata, size, 'GND'), size, size, 1, 'RGB').save(path + '/maps/' + data['fileref'] + '/IPF-GND.png')
        
        if 'c' in metadata[data['fileref']]['aType']:
            make_image(mapSGP(data, metadata, size, 'CF'), size, size, 1, 'RGB').save(path + '/maps/' + data['fileref'] + '/IPF-CF.png')
        
        if 'd' in metadata[data['fileref']]['aType']:
            make_image(mapSGP(data, metadata, size, 'cluster'), size, size, 1, 'RGB').save(path + '/maps/' + data['fileref'] + '/IPF-C.png')
        
        for localID in data['phases']:
            if data['phases'][localID]['ID'] == 0:
                continue
            else:
                phaseID = data['phases'][localID]['ID']
                make_image(mapSGP(data, metadata, size, 'colour', phaseID=phaseID), size, size, 1, 'RGB').save(path + '/maps/' + data['fileref'] + '/IPF[' + str(phaseID) + '].png')
                make_image(mapSGP(data, metadata, size, 'count', phaseID=phaseID), size, size, 1, 'RGB').save(path + '/maps/' + data['fileref'] + '/IPF-Q[' + str(phaseID) + '].png')
                make_image(mapSGP(data, metadata, size, 'phase', phaseID=phaseID), size, size, 1, 'RGB').save(path + '/maps/' + data['fileref'] + '/IPF-P[' + str(phaseID) + '].png')
                make_image(mapSGP(data, metadata, size, 'PQ', phaseID=phaseID), size, size, 1, 'RGB').save(path + '/maps/' + data['fileref'] + '/IPF-PQ[' + str(phaseID) + '].png')
                make_image(mapSGP(data, metadata, size, 'IQ', phaseID=phaseID), size, size, 1, 'RGB').save(path + '/maps/' + data['fileref'] + '/IPF-IQ[' + str(phaseID) + '].png')
                
                if 'g' in metadata[data['fileref']]['aType']:
                    make_image(mapSGP(data, metadata, size, 'KAM', phaseID=phaseID), size, size, 1, 'RGB').save(path + '/maps/' + data['fileref'] + '/IPF-KAM[' + str(phaseID) + '].png')
                    make_image(mapSGP(data, metadata, size, 'GND', phaseID=phaseID), size, size, 1, 'RGB').save(path + '/maps/' + data['fileref'] + '/IPF-GND[' + str(phaseID) + '].png')
                
                if 'c' in metadata[data['fileref']]['aType']:
                    make_image(mapSGP(data, metadata, size, 'CF', phaseID=phaseID), size, size, 1, 'RGB').save(path + '/maps/' + data['fileref'] + '/IPF-CF[' + str(phaseID) + '].png')
                
                if 'd' in metadata[data['fileref']]['aType']:
                    make_image(mapSGP(data, metadata, size, 'cluster', phaseID=phaseID), size, size, 1, 'RGB').save(path + '/maps/' + data['fileref'] + '/IPF-C[' + str(phaseID) + '].png')
    
    print()
    print('All maps complete.')
    input('Press ENTER to close: ')
