# -*- coding: utf-8 -*-

# Modified version of code written by Dr Gerhard Hobler (gerhard.hobler@tuwien.ac.at).
# Original code was used to generate figures in his paper DOI: 10.1103/PhysRevB.94.214109

import math
import copy
import os
from itertools import permutations
from random import Random
from shutil import rmtree
import numpy
from scipy import special, constants, optimize
from src.data_structures.phase import Phase
from src.utilities.utils import ProgressBar


def get_base(lattice):
	if lattice == 'diamond':
		xbase = (0, 2, 2, 0, 1, 3, 3, 1)
		ybase = (0, 2, 0, 2, 1, 3, 1, 3)
		zbase = (0, 0, 2, 2, 1, 1, 3, 3)
		a = 4
	elif lattice == 'fcc':
		xbase = (0, 1, 1, 0)
		ybase = (0, 1, 0, 1)
		zbase = (0, 0, 1, 1)
		a = 2
	elif lattice == 'bcc':
		xbase = (0, 1)
		ybase = (0, 1)
		zbase = (0, 1)
		a = 2
	elif lattice == 'sc':
		xbase = (0,)
		ybase = (0,)
		zbase = (0,)
		a = 1
	
	return xbase, ybase, zbase, a


def transform(x, y, z, miller):
	i, j, k = miller
	xi_scale = 1. / numpy.sqrt((i ** 2 + j ** 2) ** 2 + (i * k) ** 2 + (j * k) ** 2)
	eta_scale = 1. / numpy.sqrt(i ** 2 + j ** 2)
	zeta_scale = 1. / numpy.sqrt(i ** 2 + j ** 2 + k ** 2)
	xi = - i * k * x - j * k * y + (i ** 2 + j ** 2) * z
	eta = j * x - i * y
	zeta = i * x + j * y + k * z
	return xi, eta, zeta, xi_scale, eta_scale, zeta_scale


def get_crystal(base, miller, max_range):
	xbase, ybase, zbase, a = base
	i, j, k = miller
	rmax = max(numpy.ceil(max_range), i) * a
	lattice_x = numpy.arange(-rmax, rmax, a)
	dx, dy, dz = numpy.meshgrid(lattice_x, lattice_x, lattice_x)
	xs = dx.flatten() + xbase[0]
	ys = dy.flatten() + ybase[0]
	zs = dz.flatten() + zbase[0]
	
	for ibase in range(1, len(xbase)):
		xs = numpy.concatenate((xs, dx.flatten() + xbase[ibase]))
		ys = numpy.concatenate((ys, dy.flatten() + ybase[ibase]))
		zs = numpy.concatenate((zs, dz.flatten() + zbase[ibase]))
	
	xis, etas, zetas, xi_scale, eta_scale, zeta_scale = transform(xs, ys, zs, miller)
	xi_scale /= a
	eta_scale /= a
	zeta_scale /= a
	return xis, etas, zetas, xi_scale, eta_scale, zeta_scale


def get_rows(xis, etas, zetas, xi_scale, eta_scale, zeta_scale, max_range):
	dist = dict()
	
	for xi, eta, zeta in zip(xis, etas, zetas):
		if (xi, eta) in dist:
			dist[(xi, eta)].append(zeta)
		else:
			dist[(xi, eta)] = [zeta]
	
	keys = copy.deepcopy(dist).keys()
	xi_rows = list()
	eta_rows = list()
	
	for xi, eta in keys:
		xi_row = xi * xi_scale
		eta_row = eta * eta_scale
		
		if xi_row ** 2 + eta_row ** 2 > max_range ** 2:
			del dist[(xi,eta)]
		else:
			xi_rows.append(xi_row)
			eta_rows.append(eta_row)
	
	d = numpy.array(sorted(dist[(0, 0)])) * zeta_scale
	d = numpy.diff(d)
	dmax = numpy.max(d)
	dmin = numpy.min(d)
	dmean = 0.5 * (dmax + dmin)
	return numpy.array(xi_rows), numpy.array(eta_rows), dmean, dmax


def get_planes(xis, etas, zetas, zeta_scale, max_range):
	pos = dict()
	for xi, eta, zeta in zip(xis, etas, zetas):
		if zeta in pos:
			pos[zeta].append((xi, eta))
		else:
			pos[zeta] = [(xi, eta)]
	
	keys = copy.deepcopy(pos).keys()
	for zeta in keys:
		zeta_plane = zeta * zeta_scale
		if abs(zeta_plane) > max_range:
			del pos[zeta]
	
	zeta_sorted = sorted(pos.keys())
	index = numpy.where(numpy.array(zeta_sorted) == 0)[0][0]
	
	if abs(zeta_sorted[index-1]) > zeta_sorted[index+1]:
		zeta_ch = 0.5 * zeta_sorted[index-1] * zeta_scale
		opposing = (0, 0) in pos[zeta_sorted[index-1]]
	else:
		zeta_ch = 0.5 * zeta_sorted[index+1] * zeta_scale
		opposing = (0, 0) in pos[zeta_sorted[index+1]]

	zeta_planes = numpy.array(zeta_sorted) * zeta_scale
	d = numpy.diff(zeta_planes)
	dmax = numpy.max(d)
	dmin = numpy.min(d)
	dmean = 0.5 * (dmax + dmin)
	return zeta_planes, zeta_ch, opposing, dmean


def fr1(r):
	a = numpy.array((0.1818, 0.50983, 0.2802, 0.02817))
	b = numpy.array((3.2, 0.9423, 0.4028, 0.2016))
	f = numpy.zeros_like(r)
	f1 = numpy.zeros_like(r)
	f2 = numpy.zeros_like(r)
	
	for i in range(4):
		br = b[i] * r
		k0_val = special.k0(br)
		k1_val = special.k1(br)
		f += a[i] * k0_val
		f1 += -a[i] * b[i] * k1_val
		f2 += a[i] * b[i] ** 2 * (k0_val + k1_val/br)
	
	return f, f1, f2


def ur1(r, Z1, Z2, dmean):
	e = constants.physical_constants['elementary charge'][0]
	eps0 = constants.physical_constants['electric constant'][0]
	q = e / (4 * numpy.pi * eps0)
	aBohr = constants.physical_constants['Bohr radius'][0] * 1e10
	ufac = 2 * Z1 * Z2 * q / (dmean * 1e-10)
	aZBL = 0.8853 * aBohr / (Z1 ** 0.23 + Z2 ** 0.23)
	f, f1, f2 = fr1(r / aZBL)
	u = ufac * f
	u1 = ufac * f1 / aZBL
	u2 = ufac * f2 / aZBL ** 2
	return u, u1, u2


def ur(pos, xi_rows, eta_rows, Z1, Z2, dmean):
	x, y = pos
	r = numpy.hypot(x-xi_rows, y-eta_rows)
	u1, du1_dr, dummy = ur1(r, Z1, Z2, dmean)
	u = numpy.sum(u1)
	dudx = numpy.sum(du1_dr * (x-xi_rows) / r)
	dudy = numpy.sum(du1_dr * (y-eta_rows) / r)
	return u, numpy.array((dudx, dudy))


def fp1(r):
	a = numpy.array((0.1818, 0.50983, 0.2802, 0.02817))
	b = numpy.array((3.2, 0.9423, 0.4028, 0.2016))
	f = numpy.sum(a / b * numpy.exp(-b * r))
	return f


def up1(r, Z1, Z2, d2):
	e = constants.physical_constants['elementary charge'][0]
	eps0 = constants.physical_constants['electric constant'][0]
	q = e / (4 * numpy.pi * eps0)
	aBohr = constants.physical_constants['Bohr radius'][0] * 1e10
	aZBL = 0.8853 * aBohr / (Z1 ** 0.23 + Z2 ** 0.23)
	ufac = 2 * numpy.pi * Z1 * Z2 * q * aZBL / (d2 ** 2 * 1e-10)
	f = fp1(r / aZBL)
	u = ufac * f
	return u


def up(zeta, zeta_planes, Z1, Z2, d2):
	u = 0.
	
	for zeta_plane in zeta_planes:
		r = abs(zeta - zeta_plane)
		u1 = up1(r, Z1, Z2, d2)
		u += u1
	
	return u


def fun1(r, Z1, Z2, dmean, dmax, vbcorr, e):
	u, u1, u2 = ur1(r, Z1, Z2, dmean)
	ee = (dmax * u2 / (2 * vbcorr * u1)) ** 2 * u
	return ee - e

def fun2(r, Z1, Z2, opposing, rch, d2, e):
	u1, du1_dr, dummy = ur1(r, Z1, Z2, 1.)
	
	if opposing:
		u2, du2_dr, dummy = ur1(2 * rch-r, Z1, Z2, 1.)
		ee = -2 * (du1_dr-du2_dr) * (r / d2) ** (-1.5)
	else:
		ee = -2 * du1_dr * (r / d2) ** (-1.5)
	
	return ee - e


def gen_crit_data(
	beam_atomic_number: int,
	target: Phase,
	beam_energy: float,
	max_range: float,
	max_index: int,
	random_source: Random,
	cache_dir: str,
):
	e = beam_energy
	Z1 = beam_atomic_number
	Z2 = target.atomic_number
	lType = target.lattice_type.value
	diamond = target.diamond_structure
	
	if lType == 'cP':
		lattice = 'sc'
	elif lType == 'cI':
		lattice = 'bcc'
	elif lType == 'cF' and diamond:
		lattice = 'diamond'
	elif lType == 'cF':
		lattice = 'fcc'
	else:
		raise NotImplementedError()
	
	alat = 10 * target.lattice_constants_nm[0]
	xrms = 10 * target.vibration_amplitude_nm
	base = get_base(lattice)
	fileref = '[' + str(target.global_id) + '][' + str(beam_atomic_number) + '][' + str(beam_energy) + ']'
	os.makedirs(cache_dir, exist_ok=True)
	file_emin_a = open(cache_dir + "/" + fileref + 'emin-a.txt', 'w')
	file_emin_p = open(cache_dir + "/" + fileref + 'emin-p.txt', 'w')
	file_psicrit_a = open(cache_dir + "/" + fileref + 'psicrit-a.txt', 'w')
	file_psicrit_p = open(cache_dir + "/" + fileref + 'psicrit-p.txt', 'w')
	file_eperpcrit_a = open(cache_dir + "/" + fileref + 'eperpcrit-a.txt', 'w')
	file_eperpcrit_p = open(cache_dir + "/" + fileref + 'eperpcrit-p.txt', 'w')
	file_uper_a = open(cache_dir + "/" + fileref + 'uper-a.txt', 'w')
	file_uper_p = open(cache_dir + "/" + fileref + 'uper-p.txt', 'w')
	file_emin_a.write('# h  k  l  E_min\n')
	file_emin_p.write('# h  k  l  E_min\n')
	file_psicrit_a.write('# h  k  l  psi_crit\n')
	file_psicrit_p.write('# h  k  l  psi_crit\n')
	file_eperpcrit_a.write('# h  k  l  E_perp_crit\n')
	file_eperpcrit_p.write('# h  k  l  E_perp_crit\n')
	file_uper_a.write('# h  k  l  U_percentiles\n')
	file_uper_p.write('# h  k  l  U_percentiles\n')
	xrand = numpy.array([random_source.random() for _ in range(10000)])
	yrand = numpy.array([random_source.random() for _ in range(10000)])
	zrand = numpy.array([random_source.random() for _ in range(10000)])
	millers = list()
	emins = list()
	total = 0
	
	for i in range(max_index + 1):
		for j in range(i + 1):
			for k in range(j + 1):
				total += 1

	progress_bar = ProgressBar(total)
	progress_bar.print()
	
	for i in range(max_index + 1):
		for j in range(i + 1):
			ggT = math.gcd(i, j)
			
			for k in range(j + 1):
				if math.gcd(ggT, k) != 1:
					progress_bar.increment_print()
					continue
				
				miller = (i, j, k)
				xis, etas, zetas, xi_scale, eta_scale, zeta_scale = get_crystal(base, miller, max_range / alat)
				xi_rows, eta_rows, dmean, dmax = get_rows(xis, etas, zetas, xi_scale, eta_scale, zeta_scale, max_range / alat)
				xi_rows *= alat
				eta_rows *= alat
				dmean *= alat
				dmax *= alat
				rs = numpy.hypot(xi_rows, eta_rows)
				inear = numpy.argmin(rs[rs != 0])
				xi_near = xi_rows[rs != 0][inear]
				eta_near = eta_rows[rs != 0][inear]
				xstarts = list()
				ystarts = list()
				xstarts.append(xi_near / 2 + eta_near / 10)
				ystarts.append(eta_near / 2 - xi_near / 10)
				xstarts.append(-xstarts[0])
				ystarts.append(-ystarts[0])
				xstarts.append(ystarts[0])
				ystarts.append(-xstarts[0])
				xstarts.append(-ystarts[0])
				ystarts.append(xstarts[0])
				rstart = numpy.hypot(xstarts[0], ystarts[0])
				urmins = list()
				xchs = list()
				ychs = list()
				
				for start in zip(xstarts, ystarts):
					res = optimize.minimize(ur, numpy.array(start), (xi_rows, eta_rows, Z1, Z2, dmean), method='TNC', jac=True, tol=1e-5, options={'stepmx':rstart/100, 'disp':False})
					
					if res['success']:
						urmins.append(res['fun'])
						xchs.append(res['x'][0])
						ychs.append(res['x'][1])
				
				if len(xchs) != 0:
					xchs = numpy.array(xchs)
					ychs = numpy.array(ychs)
					urmins = numpy.array(urmins)
					rchs = numpy.hypot(xchs, ychs)
					valid = numpy.ones_like(rchs, dtype=bool)
					select = numpy.zeros_like(rchs, dtype=bool)
					valid = (rchs < max_range)
					valid = numpy.logical_and(valid, urmins < 1.0001 * numpy.min(urmins[valid]))
					valid = numpy.logical_and(valid, rchs == numpy.min(rchs[valid]))
					index = numpy.argmax(valid)
					xch = xchs[index]
					ych = ychs[index]
					rch = rchs[index]
					urmin = urmins[index]
					index = numpy.argmin(rs)
					rchs = numpy.hypot(xch - xi_rows, ych - eta_rows)
					
					if any(rchs < 0.99 * rch):
						index = numpy.argmin(rchs)
					
					xi = xi_rows[index]
					eta = eta_rows[index]
					rch = rchs[index]
					
					vbcorr = 1.15
					rmax = numpy.sqrt(rch ** 2 - (2 * xrms) ** 2)
					u, u1, u2 = ur1(rmax, Z1, Z2, dmean)
					emin = (dmax * u2 / (2 * vbcorr * u1)) ** 2 * u
					file_emin_a.write(' %2d %2d %2d  %0.2f\n' % (i, j, k, emin))
					
					if e > emin:
						rcrit0 = optimize.brentq(fun1, 1e-6, rmax, args=(Z1, Z2, dmean, dmax, vbcorr, e))
						rcrit = numpy.sqrt(rcrit0 ** 2 + (2 * xrms) ** 2)
						dirx = (xch - xi) / rch
						diry = (ych - eta) / rch
						x = xi + dirx * rcrit
						y = eta + diry * rcrit
						u, dummy = ur((x, y), xi_rows, eta_rows, Z1, Z2, dmean)
						psicrit = numpy.degrees(numpy.sqrt((u - urmin) / e))
						file_eperpcrit_a.write(' %2d %2d %2d  %0.2f\n' % (i, j, k, u))
						file_psicrit_a.write(' %2d %2d %2d  %0.2f\n' % (i, j, k, psicrit))
						xi_rand, eta_rand, zeta_rand, xi_scale_rand, eta_scale_rand, zeta_scale_rand = transform(xrand, yrand, zrand, miller)
						xi_rand *= alat * xi_scale_rand
						eta_rand *= alat * eta_scale_rand
						zeta_rand *= alat * zeta_scale_rand
						u = list()
						
						for xi, eta in zip(xi_rand, eta_rand):
							u.append(ur((xi, eta), xi_rows, eta_rows, Z1, Z2, dmean)[0])
						
						u_percentiles = numpy.percentile(u, range(100))
						line = ' %2d %2d %2d ' % (i, j, k)
						
						for u_percentile in u_percentiles:
							line += ' %0.2f' % u_percentile
						
						line += '\n'
						file_uper_a.write(line)
				
				zeta_planes, zeta_ch, opposing, dmean = get_planes(xis, etas, zetas, zeta_scale, max_range / alat)
				zeta_planes *= alat
				zeta_ch *= alat
				dmean *= alat
				dens = len(base[0]) / alat ** 3
				d2 = 1 / numpy.sqrt(dens * dmean)
				rch = abs(zeta_ch)
				
				if rch <= 2 * xrms:
					emin = numpy.inf
				else:
					rmax = numpy.sqrt(rch ** 2 - (2 * xrms) ** 2)
					u1, du1_dr, dummy = ur1(rmax, Z1, Z2, 1.)
					
					if opposing:
						u2, du2_dr, dummy = ur1(2 * rch-rmax, Z1, Z2, 1.)
						emin = -2 * (du1_dr-du2_dr) * (rmax / d2) ** (-1.5)
					else:
						emin = -2 * du1_dr * (rmax / d2) ** (-1.5)
				
				file_emin_p.write(' %2d %2d %2d  %0.2f\n' % (i, j, k, emin))
				
				if e > emin:
					rcrit0 = optimize.brentq(fun2, 1e-6, rmax, args=(Z1, Z2, opposing, rch, d2, e))
					rcrit = numpy.sqrt(rcrit0 ** 2 + (2 * xrms) ** 2)
					upmin = up(zeta_ch, zeta_planes, Z1, Z2, d2)
					u = up(numpy.copysign(rcrit, zeta_ch), zeta_planes, Z1, Z2, d2)
					psicrit = numpy.degrees(numpy.sqrt((u - upmin) / e))
					file_eperpcrit_p.write(' %2d %2d %2d  %0.2f\n' % (i, j, k, u))
					file_psicrit_p.write(' %2d %2d %2d  %0.2f\n' % (i, j, k, psicrit))
					u = list()
					
					for zeta in zeta_rand:
						u.append(up(zeta, zeta_planes, Z1, Z2, d2))
					
					u_percentiles = numpy.percentile(u, range(100))
					line = ' %2d %2d %2d ' % (i, j, k)
					
					for u_percentile in u_percentiles:
						line += ' %0.2f' % u_percentile
					
					line += '\n'
					file_uper_p.write(line)

				progress_bar.increment_print()

	progress_bar.terminate_print()
	file_emin_a.close()
	file_emin_p.close()
	file_psicrit_a.close()
	file_psicrit_p.close()
	file_eperpcrit_a.close()
	file_eperpcrit_p.close()
	file_uper_a.close()
	file_uper_p.close()


def load_crit_data(
	beam_atomic_number: int,
	target: Phase,
	beam_energy: float,
	random_source: Random,
	use_cache: bool,
	cache_dir: str,
) -> dict:
	if not use_cache:
		cache_dir = f"{cache_dir}/temp"

	try:
		fileref = '[' + str(target.global_id) + '][' + str(beam_atomic_number) + '][' + str(beam_energy) + ']'

		try:
			file_eperpcrit_a = open(cache_dir + "/" + fileref + 'eperpcrit-a.txt', 'r')
			file_eperpcrit_p = open(cache_dir + "/" + fileref + 'eperpcrit-p.txt', 'r')
			file_uper_a = open(cache_dir + "/" + fileref + 'uper-a.txt', 'r')
			file_uper_p = open(cache_dir + "/" + fileref + 'uper-p.txt', 'r')
			file_eperpcrit_a.close()
			file_eperpcrit_p.close()
			file_uper_a.close()
			file_uper_p.close()
		except FileNotFoundError:
			max_range = 10  # Maximum range from origin where rows are to be considered (Å)
			max_index = 10  # Maximum Miller index to be considered
			print('Generating channelling fraction data for phase ' + str(target.global_id) + '.')

			gen_crit_data(
				beam_atomic_number=beam_atomic_number,
				target=target,
				beam_energy=beam_energy,
				max_range=max_range,
				max_index=max_index,
				random_source=random_source,
				cache_dir=cache_dir,
			)

		try:
			has, kas, las, eperpcrit_a = numpy.loadtxt(cache_dir + "/" + fileref + 'eperpcrit-a.txt', unpack=True)
			line_tuples = numpy.loadtxt(cache_dir + "/" + fileref + 'uper-a.txt')
			has_u = line_tuples[:,0]
			kas_u = line_tuples[:,1]
			las_u = line_tuples[:,2]

			if not (all(has_u == has) and all(kas_u == kas) and all(las_u == las)):
				exit('Inconsistent files ' + cache_dir + "/" + fileref + 'eperpcrit-a.txt\' and ' + cache_dir + "/" + fileref + 'uper-a.txt\'')

			u_percentiles_a = line_tuples[:,3:]
			axial = True
		except IndexError:
			u_percentiles_a = 0
			axial = False

		try:
			hps, kps, lps, eperpcrit_p = numpy.loadtxt(cache_dir + "/" + fileref + 'eperpcrit-p.txt', unpack=True)
			line_tuples = numpy.loadtxt(cache_dir + "/" + fileref + 'uper-p.txt')
			hps_u = line_tuples[:,0]
			kps_u = line_tuples[:,1]
			lps_u = line_tuples[:,2]

			if not (all(hps_u == hps) and all(kps_u == kps) and all(lps_u == lps)):
				exit('Inconsistent files ' + cache_dir + "/" + fileref + 'eperpcrit-p.txt\' and ' + cache_dir + "/" + fileref + 'uper-p.txt\'')

			u_percentiles_p = line_tuples[:,3:]
			planar = True
		except IndexError:
			u_percentiles_p = 0
			planar = False

		output = dict()
		output['beam_z'] = beam_atomic_number
		output['target_id'] = target.global_id
		output['energy'] = beam_energy
		output['data'] = dict()
		output['data']['axial'] = axial
		output['data']['has'] = has
		output['data']['kas'] = kas
		output['data']['las'] = las
		output['data']['eperpcrit_a'] = eperpcrit_a
		output['data']['u_percentiles_a'] = u_percentiles_a
		output['data']['planar'] = planar
		output['data']['hps'] = hps
		output['data']['kps'] = kps
		output['data']['lps'] = lps
		output['data']['eperpcrit_p'] = eperpcrit_p
		output['data']['u_percentiles_p'] = u_percentiles_p
		return output
	finally:
		if not use_cache:
			rmtree(cache_dir)


def fraction(effective_beam_vector: tuple[float, float, float], crit_data: dict) -> float:
	vx, vy, vz = -effective_beam_vector[0], -effective_beam_vector[1], effective_beam_vector[2]
	theta = -math.atan(math.sqrt(vx ** 2 + vy ** 2) / vz)
	phi = (math.pi / 2) - math.atan2(vy, vx)
	e = crit_data['energy']
	axial = crit_data['data']['axial']
	has = crit_data['data']['has']
	kas = crit_data['data']['kas']
	las = crit_data['data']['las']
	eperpcrit_a = crit_data['data']['eperpcrit_a']
	u_percentiles_a = crit_data['data']['u_percentiles_a']
	planar = crit_data['data']['planar']
	hps = crit_data['data']['hps']
	kps = crit_data['data']['kps']
	lps = crit_data['data']['lps']
	eperpcrit_p = crit_data['data']['eperpcrit_p']
	u_percentiles_p = crit_data['data']['u_percentiles_p']
	
	output = 0.0
	dirx = numpy.cos(theta)
	diry = numpy.sin(theta) * numpy.cos(phi)
	dirz = numpy.sin(theta) * numpy.sin(phi)
	
	if axial:
		for h, k, l, eperpcrit, u_percentiles in zip(has, kas, las, eperpcrit_a, u_percentiles_a):
			dirm = numpy.sqrt(h ** 2 + k ** 2 + l ** 2)
			dirmx = h / dirm
			dirmy = k / dirm
			dirmz = l / dirm
			psi = numpy.arccos(dirx * dirmx + diry * dirmy + dirz * dirmz)
			ulim = eperpcrit - e * psi ** 2
			prob = numpy.interp(ulim, u_percentiles, range(100), left=0, right=100)
			output = max(output, float(prob))
	
	if planar:
		for h, k, l, eperpcrit, u_percentiles in zip(hps, kps, lps, eperpcrit_p, u_percentiles_p):
			dirm = numpy.sqrt(h ** 2 + k ** 2 + l ** 2)
			
			for hh, kk, ll in permutations((h, k, l)):
				for hsign in (1, -1):
					for ksign in (1, -1):
						for lsign in (1, -1):
							hhh = hsign * hh
							kkk = ksign * kk
							lll = lsign * ll
							dirmx = hhh / dirm
							dirmy = kkk / dirm
							dirmz = lll / dirm
							psi = abs(numpy.arcsin(dirx * dirmx + diry * dirmy + dirz * dirmz))
							ulim = eperpcrit - e * psi ** 2
							prob = numpy.interp(ulim, u_percentiles, range(100), left=0, right=100)
							output = max(output, float(prob))

	return output
