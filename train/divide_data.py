import numpy as np
from ROOT import TFile
import ROOT
import h5py
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# define files
f = ROOT.TFile("histos.root", "READ")
fwrite = h5py.File("phi_sliced.hdf5", "a")

# record data from histos to np array
nEvents = 4777
data = np.zeros((nEvents,14,18))

# delete dataset if it already exists
if "phi0_5" in str(fwrite.keys()): del fwrite["phi0_5"]
if "phi6_11" in str(fwrite.keys()): del fwrite["phi6_11"]
if "phi12_17" in str(fwrite.keys()): del fwrite["phi12_17"]

# record data
print("Writing to HDF5...")
for n in tqdm(range(nEvents)):
	h = f.Get(f"myHisto_{n+1};1")
	for i in range(14):
		for j in range(18):
			data[n,i,j] = h.GetBinContent(h.GetBin(i+1,j+1))

# write data into datasets
ds1 = fwrite.create_dataset("phi0_5", data = data[:,:,0:6])
ds2 = fwrite.create_dataset("phi6_11", data = data[:,:,6:12])
ds3 = fwrite.create_dataset("phi12_17", data = data[:,:,12:18])

f.Close()
fwrite.close()

# check if I extracted the right values
fwrite = ROOT.TFile("histos_phi_sliced.root", "RECREATE")
f = h5py.File("phi_sliced.hdf5", "r")
ds1 = f["phi0_5"]
ds2 = f["phi6_11"]
ds3 = f["phi12_17"]

print("Creating histos...")
for i in tqdm(range(nEvents)):
	h1 = ROOT.TH2F("hpartial1", "title;eta;phi",14,0,14,6,0,6)
	h2 = ROOT.TH2F("hpartial2", "title;eta;phi",14,0,14,6,0,6)
	h3 = ROOT.TH2F("hpartial3", "title;eta;phi",14,0,14,6,0,6)
	for j in range(14):
		for k in range(6):
			h1.SetBinContent(j+1,k+1,ds1[i,j,k])
			h2.SetBinContent(j+1,k+1,ds2[i,j,k])
			h3.SetBinContent(j+1,k+1,ds3[i,j,k])
	h1.Write("hpartial1")
	h2.Write("hpartial2")
	h3.Write("hpartial3")

fwrite.Close()
f.close()
