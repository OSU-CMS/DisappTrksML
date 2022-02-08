import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys

ele = np.load('output_electrons.npz', allow_pickle=True, encoding='bytes')
muons = np.load('output_muons.npz', allow_pickle=True, encoding='bytes')
higgsino_700_10 = np.load('output_higgsino_700_10.npz', allow_pickle=True, encoding='bytes')
higgsino_700_100 = np.load('output_higgsino_700_100.npz', allow_pickle=True, encoding='bytes')
higgsino_700_1000 = np.load('output_higgsino_700_1000.npz', allow_pickle=True, encoding='bytes')
higgsino_700_10000 = np.load('output_higgsino_700_10000.npz', allow_pickle=True, encoding='bytes')

# muons
model_idx = 2
disc_ele = ele['results'][:, model_idx]
disc_muons = muons['results'][:, model_idx]
disc_higgsino_10 = higgsino_700_10['results'][:, model_idx]
disc_higgsino_100 = higgsino_700_100['results'][:, model_idx]
disc_higgsino_1000 = higgsino_700_1000['results'][:, model_idx]
disc_higgsino_10000 = higgsino_700_10000['results'][:, model_idx]

#for x in [disc_ele[disc_ele > -1], disc_muons[disc_muons > -1]]:
#	print(x.shape)

for file in [muons, higgsino_700_10, higgsino_700_100, higgsino_700_1000, higgsino_700_10000]:
	
	events_fidMap = file['results'][:,-1]
	disc = file['results'][:,0]
	events_fidMap = events_fidMap[disc > -1]
	n_before = events_fidMap.shape[0]
	events_fidMap = events_fidMap[events_fidMap < 2.0]
	n_after = events_fidMap.shape[0]
	eff_fidMap = n_after*1.0 / n_before if n_before > 0 else 0

	print(eff_fidMap)
	# print n_before, n_after

	# events_clf = file['results'][:,3]
	# events_clf = events_clf[disc > -1]
	# n_before = events_clf.shape[0]
	# events_clf = events_clf[events_clf < 0.5]
	# n_after = events_clf.shape[0]
	# eff_clf = n_after*1.0 / n_before if n_before > 0 else 0

	# print n_before, n_after

	# print round((eff_clf/eff_fidMap)*100, 2)
	# print

# events_fidMap = muons['results'][:,-1]
# disc = muons['results'][:,3]
# events_fidMap = events_fidMap[disc > -1]
# n_before = events_fidMap.shape[0]
# events_fidMap = events_fidMap[events_fidMap < 2.0]
# n_after = events_fidMap.shape[0]
# eff_fidMap = n_after*1.0 / n_before if n_before > 0 else 0

# sys.exit(0)

# for i, wp in enumerate(['ele_14', 'ele_17', 'ele_19']):
# 	n = ele['results'][:, i]
# 	n = n[n > -1]
# 	n = n[n <= 0.5]

# 	print n[n > 0.5].shape[0] / n[n > -1].shape[0]
# 	print(wp, '=', n.shape[0])

# 	for i2, s in enumerate([higgsino_700_10, higgsino_700_100, higgsino_700_1000, higgsino_700_10000]):
# 		ns = s['results'][:, i]
# 		ns = ns[ns > -1]
# 		ns = ns[ns <= 0.5]
# 		print([10, 100, 1000, 10000][i2], wp, '=', ns.shape[0])

# 		print ns.shape[0]/np.sqrt(n.shape[0])

fig = plt.figure(figsize=(7,5))
ax = fig.add_subplot(1,1,1)

ax.hist(disc_ele[disc_ele > -1], bins=np.linspace(0,1,50), label='T&P Electrons', histtype='step', density=True)
# ax.hist(disc_muons[disc_muons > -1], bins=np.linspace(0,1,50), label='T&P Muons', histtype='step', density=True)
ax.hist(disc_higgsino_10, bins=np.linspace(0,1,50), label='Higgsino 700GeV 10cm MC', histtype='step', density=True)
# ax.hist(disc_higgsino_100, bins=np.linspace(0,1,50), label='Higgsino 700GeV 100cm MC', histtype='step', density=True)
ax.hist(disc_higgsino_1000, bins=np.linspace(0,1,50), label='Higgsino 700GeV 1000cm MC', histtype='step', density=True)
ax.hist(disc_higgsino_10000, bins=np.linspace(0,1,50), label='Higgsino 700GeV 10000cm MC', histtype='step', density=True)

ax.set_xlabel("Discriminant value")
ax.set_ylabel("Occurrence")
ax.legend()
fig.tight_layout()
ax.set_yscale('log')
fig.savefig("electronModel_preds.pdf", bbox_inches='tight')

