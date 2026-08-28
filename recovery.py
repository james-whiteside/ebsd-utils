import copy
import itertools
import math
from typing import Any

import numpy


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
            output[int(material[0])]['vibration'] = float(
                material[5])  # For some values at RT see D.S.Gemmell, Rev. Mod. Phys. 46 (1974) 129.
            output[int(material[0])]['type'] = material[6]
            output[int(material[0])]['constants'] = list((float(material[7]), float(material[8]), float(material[9])))
            output[int(material[0])]['angles'] = list(
                (math.radians(float(material[10])), math.radians(float(material[11])),
                 math.radians(float(material[12]))))
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
            pairs.append(list((list((int(variant[3]), int(variant[4]), int(variant[5]))),
                               list((int(variant[6]), int(variant[7]), int(variant[8]))))))
            pairs.append(list((list((int(variant[9]), int(variant[10]), int(variant[11]))),
                               list((int(variant[12]), int(variant[13]), int(variant[14]))))))
            # pairs.append(list(list(indices) for indices in (numpy.cross(numpy.array(pairs[0][0]), numpy.array(pairs[1][0])), numpy.cross(numpy.array(pairs[0][1]), numpy.array(pairs[1][1])))))
            output[variant[0]]['vectors'] = pairs

    return output


def getTwinList():
    output = dict()

    with open('orientation/twin.csv', 'r') as file:
        file.readline()

        for line in file:
            ID = 'twin-' + line.replace('\n', '').replace(',', '')
            indices = list(int(index) for index in line.split(','))
            output[ID] = indices

    return output


def dTheta(dR):
    if 0.5 * (abs(dR[0][0]) + abs(dR[1][1]) + abs(dR[2][2]) - 1) > 1:
        return math.acos(1)
    elif 0.5 * (abs(dR[0][0]) + abs(dR[1][1]) + abs(dR[2][2]) - 1) < -1:
        return math.acos(-1)
    else:
        return math.acos(0.5 * (abs(dR[0][0]) + abs(dR[1][1]) + abs(dR[2][2]) - 1))


def formatIndices(hkl, iType):
    if iType == 'zone':
        prefix = '['
        suffix = ']'
    elif iType == 'zones':
        prefix = '<'
        suffix = '>'
    elif iType == 'plane':
        prefix = '('
        suffix = ')'
    elif iType == 'planes':
        prefix = '{'
        suffix = '}'

    return prefix + str(hkl[0]) + ' ' + str(hkl[1]) + ' ' + str(hkl[2]) + suffix


def genFamily(hkl):
    if hcf(hkl) != 0:
        hkl = list(index // hcf(hkl) for index in hkl)

    perms = sorted(list(set(list(itertools.permutations(hkl)))))
    refs = sorted(list(set(list(itertools.permutations(list((1, 1, 1, -1, -1, -1)), 3)))))
    refperms = sorted(list(list(refperm) for refperm in set(
        (ref[0] * perm[0], ref[1] * perm[1], ref[2] * perm[2]) for ref in refs for perm in perms)), reverse=True)
    output = list()

    for hklA in refperms:
        dupe = False

        for hklB in output:
            if (hklA[0] == hklB[0] and hklA[1] == hklB[1] and hklA[2] == hklB[2]) or (
                    hklA[0] == -hklB[0] and hklA[1] == -hklB[1] and hklA[2] == -hklB[2]):
                dupe = True

        pol = 0

        for i in range(3):
            if hklA[i] > 0:
                pol += 1
            if hklA[i] < 0:
                pol -= 1

        if not dupe and pol >= 0:
            output.append(hklA)

    return output


def genPairs(hklAs, hklBs):
    output = list()

    for hklA in hklAs:
        for hklB in hklBs:
            output.append(list((hklA, hklB)))

    return output


def genSet(planePair, zonePair):
    pair1 = list((numpy.array(planePair[0]), numpy.array(planePair[1])))
    pair2 = list((numpy.array(zonePair[0]), numpy.array(zonePair[1])))
    pair3 = list((numpy.cross(numpy.array(planePair[0]), numpy.array(zonePair[0])),
                  numpy.cross(numpy.array(planePair[1]), numpy.array(zonePair[1]))))
    return list((list(list(hkl) for hkl in pair1), list(list(hkl) for hkl in pair2), list(list(hkl) for hkl in pair3)))


def genMatrix(pairs, params):
    pairs = copy.deepcopy(pairs)
    pairs.append(list(list(indices) for indices in (numpy.cross(numpy.array(pairs[0][0]), numpy.array(pairs[1][0])),
                                                    numpy.cross(numpy.array(pairs[0][1]), numpy.array(pairs[1][1])))))
    x = numpy.array(list((params[0][i] * numpy.linalg.norm(numpy.array(pairs[i][0]))) / (
                params[1][i] * numpy.linalg.norm(numpy.array(pairs[i][1]))) for i in range(3)))
    uA = numpy.transpose(numpy.array(list(pairs[i][0] for i in range(3))))
    uB = numpy.transpose(numpy.array(list(pairs[i][1] for i in range(3))))
    J = numpy.dot(x * uB, numpy.linalg.inv(uA))
    return J


def genTwin(plane):
    h, k, l = plane
    T = numpy.array(list(
        (list((h ** 2 - k ** 2 - l ** 2, 2 * h * k, 2 * l * h)), list((2 * h * k, k ** 2 - l ** 2 - h ** 2, 2 * k * l)),
         list((2 * l * h, 2 * k * l, l ** 2 - h ** 2 - k ** 2)))))
    s = - 1 / (h ** 2 + k ** 2 + l ** 2)
    J = s * T
    return J


def getVariantList(filepath):
    output = list()

    with open(filepath, 'r') as file:
        for line in file:
            pairs = list(
                list(list(int(index) for index in indices.split(',')) for indices in pair.split(':')) for pair in
                line.split(';'))
            pairs.append(list(list(indices) for indices in
                              (numpy.cross(numpy.array(pairs[0][0]), numpy.array(pairs[1][0])),
                               numpy.cross(numpy.array(pairs[0][1]), numpy.array(pairs[1][1])))))
            output.append(pairs)

    return output


def print_matches(data: dict[str, Any], metadata: dict[str, Any]) -> None:
    if 'd' in metadata[data['fileref']]['aType']:
        materials = getMaterials()
        variants = getVariantList()
        twins = getTwinList()
        matches = list()

        for k1 in range(1, metadata[data['fileref']]['k'] + 1):
            for k2 in range(1, metadata[data['fileref']]['k'] + 1):
                if k1 == k2:
                    continue

                for variant in variants:
                    if variants[variant]['lTypes'][0] == data['phases'][data['data']['k']['phase'][k1]]['type'] and \
                            variants[variant]['lTypes'][1] == data['phases'][data['data']['k']['phase'][k2]]['type']:
                        match = dict()
                        match['variant'] = variant
                        match['k1'] = k1
                        match['k2'] = k2
                        params = list((materials[data['phases'][data['data']['k']['phase'][k1]]['ID']]['constants'],
                                       materials[data['phases'][data['data']['k']['phase'][k1]]['ID']]['constants']))
                        R1 = data['data']['k']['R'][k1]
                        R2 = data['data']['k']['R'][k2]
                        polarity = sorted(list(set(list(itertools.permutations(list((1, 1, 1, 1, -1, -1, -1, -1)), 4)))),
                                          reverse=True)
                        theta = 2 * math.pi

                        for i in range(16):
                            vectors = list((list((list(polarity[i][0] * hkl for hkl in variants[variant]['vectors'][0][0]),
                                                  list(
                                                      polarity[i][1] * hkl for hkl in variants[variant]['vectors'][0][1]))),
                                            list((list(polarity[i][2] * hkl for hkl in variants[variant]['vectors'][1][0]),
                                                  list(polarity[i][3] * hkl for hkl in
                                                       variants[variant]['vectors'][1][1])))))
                            J = genMatrix(vectors, params)
                            s = math.sqrt(params[0][0] ** 2 + params[0][1] ** 2 + params[0][2] ** 2) / math.sqrt(
                                params[1][0] ** 2 + params[1][1] ** 2 + params[1][2] ** 2)
                            RF = numpy.dot(J / s, R1)
                            dR = numpy.dot(numpy.linalg.inv(RF), R2)
                            theta = min(dTheta(dR), theta)

                        match['dTheta'] = theta
                        match['cosine'] = math.cos(theta)
                        matches.append(match)

                for variant in twins:
                    if data['phases'][data['data']['k']['phase'][k1]]['type'] == \
                            data['phases'][data['data']['k']['phase'][k2]]['type'] and k1 < k2:
                        match = dict()
                        match['variant'] = variant
                        match['k1'] = k1
                        match['k2'] = k2
                        R1 = data['data']['k']['R'][k1]
                        R2 = data['data']['k']['R'][k2]
                        theta = 2 * math.pi
                        family = genFamily(twins[variant])

                        for plane in family:
                            J = genTwin(plane)
                            RF = numpy.dot(J, R1)
                            dR = numpy.dot(numpy.linalg.inv(RF), R2)
                            theta = min(dTheta(dR), theta)

                        match['dTheta'] = theta
                        match['cosine'] = math.cos(theta)
                        matches.append(match)

    for match in sorted(matches, key=lambda item: math.degrees(item['dTheta'])):
        output = ""
        output += match['variant'] + ','
        output += str(match['k1']) + ','
        output += str(match['k2']) + ','
        output += str(math.degrees(match['dTheta'])) + ','
        output += str(match['cosine']) + '\n'
        print(output)
