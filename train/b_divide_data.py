import numpy as np
from ROOT import TFile
from ROOT import TCanvas
from ROOT import TString
import ROOT
import h5py
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# define files
f = ROOT.TFile("A_histos.root", "READ")
fwrite = h5py.File("phi_sliced.h5", "a")

# record data from histos to np array
nEvents = 4777
data = np.zeros((nEvents,14,18))

# delete dataset if it already exists
if "CaloRegions1" in str(fwrite.keys()): del fwrite["CaloRegions1"]
if "CaloRegions2" in str(fwrite.keys()): del fwrite["CaloRegions2"]
if "CaloRegions3" in str(fwrite.keys()): del fwrite["CaloRegions3"]

# record data
print("Writing to h5...")
for n in tqdm(range(nEvents)):
	h = f.Get(f"myHisto_{n+1};1")
	for i in range(14):
		for j in range(18):
			data[n,i,j] = h.GetBinContent(h.GetBin(i+1,j+1))

# write data into datasets
for i in range(3):
	dstemp = fwrite.create_dataset(f"CaloRegions{i+1}", data = data[:,i*6:i*6+6,:], maxshape=(nEvents,6,14), chunks=True)
#ds1 = fwrite.create_dataset("phi0_5", data = data[:,:,0:6])
#ds2 = fwrite.create_dataset("phi6_11", data = data[:,:,6:12])
#ds3 = fwrite.create_dataset("phi12_17", data = data[:,:,12:18])

f.Close()
fwrite.close()

# check if right values were extracted
#fwrite = ROOT.TFile("histos_phi_sliced.root", "RECREATE")
#f = h5py.File("phi_sliced.hdf5", "r")
#ds1 = f["CaloRegions1"]
#ds2 = f["CaloRegions2"]
#ds3 = f["CaloRegions3"]

#print("Creating histos...")
#c = TCanvas()
#s = TString()
#for i in tqdm(range(nEvents)):
#	h1 = ROOT.TH2F("h1", "title;eta;phi",14,0,14,6,0,6)
#	h2 = ROOT.TH2F("h2", "title;eta;phi",14,0,14,6,0,6)
#	h3 = ROOT.TH2F("h3", "title;eta;phi",14,0,14,6,0,6)
#	for j in range(14):
#		for k in range(6):
#			h1.SetBinContent(j+1,k+1,ds1[i,j,k])
#			h2.SetBinContent(j+1,k+1,ds2[i,j,k])
#			h3.SetBinContent(j+1,k+1,ds3[i,j,k])
#	h1.Write("h1")
#	h1.Draw()
#	s = f"B_histos_png/h1_{i}.png"
#	c.Print(s)
#	h2.Write("h2")
#	h2.Draw()
#	s = f"B_histos_png/h2_{i}.png"
#	c.Print(s)
#	h3.Write("h3")
#	h3.Draw()
#	s = f"B_histos_png/h3_{i}.png"
#	c.Print(s)

#fwrite.Close()
#f.close()
