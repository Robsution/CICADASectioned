import numpy as np
from ROOT import TFile
from ROOT import TCanvas
from ROOT import TString
from ROOT import TString
import ROOT
import h5py
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")
data=[]

for i in tqdm(range(1,691)):
	if (i==448 or i==543 or i==645 or i==656): continue
	f=ROOT.TFile(f"/afs/hep.wisc.edu/home/andrewji/public/CICADASectionedData/histos/A_histos_{i}.root", "READ")
	nEvents=f.GetNkeys()
	datatemp=np.zeros((nEvents,14,18))
	for n in range(nEvents):
		h=f.Get(f"h;{i}")
		for j in range(14):
			for k in range(18):
				datatemp[n,j,k] = h.GetBinContent(h.GetBin(j+1,k+1))
	f.Close()
	data.append(datatemp)

data = np.concatenate(data)
data = np.swapaxes(data, 1, 2)
fwrite=h5py.File("/afs/hep.wisc.edu/home/andrewji/public/CICADASectionedData/h5/B_phi_sliced.h5", "a")
for n in range(3):
	if f"CaloRegions{n+1}" in str(fwrite.keys()): del fwrite[f"CaloRegions{n+1}"]
	dstemp=fwrite.create_dataset(f"CaloRegions{n+1}", data=data[:,n*6:n*6+6,:], maxshape=(data.shape[0],6,14), chunks=True)
fwrite.close()
