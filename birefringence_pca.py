import json
import math
from pathlib import Path

import numpy as np
import sympy as sp
import camb
import camb.symbolic as cs

OUT = Path('outputs')
OUT.mkdir(exist_ok=True)

# Frozen before looking at CAMB result.
FROZEN = {
    'prediction': 'Birefringence Temporal-Eigenmode Prediction 01',
    'N_knee': 5,
    'tolerance': 1,
    'support_window': [4, 6],
    'expected_most_likely_after_mode': 5,
    'support_scale': {'modes_4_5_relative_lambda': 'about >=1e-3', 'mode_6': 'substantial decline expected'},
    'falsification_examples': ['first major knee <=3', 'no strong decline through modes 7-8+'],
}

LMAX = 2500
pars = camb.CAMBparams()
pars.set_cosmology(H0=67.36, ombh2=0.02237, omch2=0.1200, mnu=0.06, omk=0.0, tau=0.0544)
pars.InitPower.set_params(As=2.10e-9, ns=0.9649)
pars.set_for_lmax(LMAX, lens_potential_accuracy=1)
pars.WantTensors = False
pars.Want_cl_2D_array = True
pars.Accuracy.AccuratePolarization = True
pars.Accuracy.AccurateReionization = True
pars.Accuracy.AccuracyBoost = 1.2
pars.Accuracy.lAccuracyBoost = 1.2
pars.Accuracy.lSampleBoost = 1.2

# Primary physical basis: beta(z), in radians, is constant inside each window.
reion_edges = [4.0, 5.5, 6.5, 7.5, 8.5, 10.0, 15.0, 30.0]
recomb_edges = [700.0, 850.0, 950.0, 1025.0, 1075.0, 1125.0, 1175.0, 1225.0, 1300.0, 1450.0, 1700.0]
windows = []
for family, edges in [('rei', reion_edges), ('rec', recomb_edges)]:
    for j in range(len(edges)-1):
        windows.append((family, edges[j], edges[j+1]))

zsym = 1/cs.a - 1
custom = []
names = []
for i, (family, zlo, zhi) in enumerate(windows):
    win = sp.Piecewise((1, sp.And(zsym >= zlo, zsym < zhi)), (0, True))
    custom.append(cs.scalar_E_source * win)
    names.append(f'B{i:02d}')

# n=2 gives the same spin-2 ell scaling as scalar E polarization.
pars.set_custom_scalar_sources(custom, source_names=names, source_ell_scales=[2]*len(names))

print('CAMB version:', getattr(camb, '__version__', 'unknown'))
print('Custom beta windows:', len(windows))
for i,w in enumerate(windows):
    print(i, names[i], w)

results = camb.get_results(pars)
arr = results.get_cmb_unlensed_scalar_array_dict(lmax=LMAX, CMB_unit='muK', raw_cl=True)
lensed = results.get_lensed_scalar_cls(lmax=LMAX, CMB_unit='muK', raw_cl=True)
unlensed = results.get_unlensed_scalar_cls(lmax=LMAX, CMB_unit='muK', raw_cl=True)

ells = np.arange(LMAX+1)
CEE_lensed = lensed[:,1]
CBB_lensed = lensed[:,2]
CEE_unlensed = unlensed[:,1]

# Small-angle response:
# Delta_B_i = 2 beta_i Delta_E_i, so d C_l^EB / d beta_i = 2 C_l^(E,E_i).
D = np.zeros((LMAX+1, len(names)))
for i,name in enumerate(names):
    key = f'Ex{name}'
    if key not in arr:
        key = f'{name}xE'
    if key not in arr:
        raise KeyError(f'Missing cross spectrum for {name}; available examples: {list(arr)[:30]}')
    D[:,i] = 2.0 * arr[key]

Ssum = np.zeros(LMAX+1)
for i,name in enumerate(names):
    key = f'Ex{name}' if f'Ex{name}' in arr else f'{name}xE'
    Ssum += arr[key]
mask_check = (ells >= 2) & (ells <= 2000) & (CEE_unlensed > 0)
coverage_ratio = np.median(Ssum[mask_check] / CEE_unlensed[mask_check])
print('Median E-source coverage ratio:', coverage_ratio)

def noise_cl(delta_p_uK_arcmin, fwhm_arcmin):
    delta = delta_p_uK_arcmin * np.pi / (180.0*60.0)
    sigma_b = (fwhm_arcmin * np.pi / (180.0*60.0)) / np.sqrt(8.0*np.log(2.0))
    return delta**2 * np.exp(ells*(ells+1.0)*sigma_b**2)

experiments = {
    'CVL_fullsky': {'segments': [(2, 2500, 1.0, 0.0, 0.0)]},
    'SO_like': {'segments': [(30, 2500, 0.40, 6.0, 1.4)]},
    'LiteBIRD_like': {'segments': [(2, 200, 0.70, 2.5, 30.0)]},
    'LiteBIRD_plus_SO_like': {
        'segments': [(2, 29, 0.70, 2.5, 30.0), (30, 2500, 0.40, 6.0, 1.4)]
    },
}

def fisher_for_experiment(spec, calibration_sigma_deg=None):
    w = np.zeros(LMAX+1)
    for lmin,lmax,fsky,dp,beam in spec['segments']:
        N = np.zeros(LMAX+1) if dp == 0.0 else noise_cl(dp, beam)
        use = (ells >= lmin) & (ells <= lmax)
        var = ((CEE_lensed + N) * (CBB_lensed + N)) / np.maximum((2*ells+1.0)*fsky, 1.0)
        good = use & np.isfinite(var) & (var > 0)
        w[good] += 1.0/var[good]

    F = D.T @ (w[:,None] * D)
    F = 0.5*(F+F.T)

    if calibration_sigma_deg is not None:
        dcal = 2.0*(CEE_lensed - CBB_lensed)
        fb = D.T @ (w*dcal)
        faa = np.sum(w*dcal*dcal) + 1.0/np.deg2rad(calibration_sigma_deg)**2
        F = F - np.outer(fb,fb)/faa
        F = 0.5*(F+F.T)

    vals, vecs = np.linalg.eigh(F)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:,order]
    vals[vals < 0] = 0.0
    rel = vals/vals[0] if vals[0] > 0 else vals
    return F, vals, rel, vecs, w

def knee_metrics(rel):
    eps=1e-300
    drops = rel[:-1]/np.maximum(rel[1:],eps)
    major = np.where(drops >= 10.0)[0]
    first_major_after = int(major[0]+1) if len(major) else None
    n_1e3 = int(np.sum(rel >= 1e-3))
    n_1e4 = int(np.sum(rel >= 1e-4))
    return {'adjacent_drop_factors': drops.tolist(),
            'first_drop_ge_10_after_mode': first_major_after,
            'n_rel_ge_1e-3': n_1e3,
            'n_rel_ge_1e-4': n_1e4}

report = {
    'frozen_prediction': FROZEN,
    'camb_version': getattr(camb, '__version__', 'unknown'),
    'lmax': LMAX,
    'basis': [{'name':names[i], 'family':w[0], 'zlo':w[1], 'zhi':w[2]} for i,w in enumerate(windows)],
    'coverage_ratio_median_ell2_2000': float(coverage_ratio),
    'experiments': {},
}

rows=[]
for exp_name,spec in experiments.items():
    for cal in [None, 0.11]:
        label = exp_name + ('_calfree' if cal is None else '_cal0p11deg')
        F,vals,rel,vecs,w = fisher_for_experiment(spec, cal)
        km = knee_metrics(rel)
        report['experiments'][label] = {
            'eigenvalues': vals.tolist(),
            'relative_eigenvalues': rel.tolist(),
            'knee_metrics': km,
        }
        np.savetxt(OUT/f'{label}_fisher.csv', F, delimiter=',')
        np.savetxt(OUT/f'{label}_eigenvectors.csv', vecs, delimiter=',')
        for m,(v,r) in enumerate(zip(vals,rel),start=1):
            rows.append((label,m,v,r))
        print('\n',label)
        print('relative eigenvalues:', np.array2string(rel, precision=5, suppress_small=False))
        print('knee:', km)

# Column-normalized response geometry is a robustness diagnostic only,
# not the primary frozen-prediction test.
_,_,_,_,wcomb = fisher_for_experiment(experiments['LiteBIRD_plus_SO_like'], None)
colnorm = np.sqrt(np.sum(wcomb[:,None]*D*D, axis=0))
valid = colnorm > 0
Dn = D[:,valid]/colnorm[valid]
G = Dn.T @ (wcomb[:,None]*Dn)
gvals = np.linalg.eigvalsh(0.5*(G+G.T))[::-1]
grel = gvals/gvals[0]
report['response_geometry_column_normalized'] = {
    'valid_basis_names':[n for n,v in zip(names,valid) if v],
    'relative_eigenvalues':grel.tolist(),
    'knee_metrics':knee_metrics(grel),
}

with open(OUT/'report.json','w') as f:
    json.dump(report,f,indent=2)

import csv
with open(OUT/'eigenvalues.csv','w',newline='') as f:
    wr=csv.writer(f)
    wr.writerow(['experiment','mode','eigenvalue','relative_eigenvalue'])
    wr.writerows(rows)

with open(OUT/'basis.csv','w',newline='') as f:
    wr=csv.writer(f)
    wr.writerow(['index','name','family','zlo','zhi'])
    for i,(name,w) in enumerate(zip(names,windows)):
        wr.writerow([i,name,w[0],w[1],w[2]])

print('\nWrote outputs/report.json and CSV matrices.')
