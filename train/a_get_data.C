#include <vector>
#include <ROOT/RVec.hxx>
#include <iostream>
#include <typeinfo>
#include <TTreeReaderArray.h>
#include <string>

void unpack(ROOT::RVec<UShort_t> column) {
    TH2F h("h", "title;eta;phi", 14, 0, 14, 18, 0, 18);
    for (int i = 0; i <= 14; ++i){
        for (int j = 0; j <= 18; ++j){
	     h.SetBinContent(i,j,column[i*14+j]);
        }
    }
    h.Write("h");
}

int a_test(){
	gROOT->ProcessLine("gErrorIgnoreLevel = 2001;");
	for (int i = 1; i < 691; i++){
		if (i==448 || i==543 || i==645 || i==656) {
			continue;
		}
		string temp("/afs/hep.wisc.edu/home/andrewji/public/CICADASectionedData/histos/A_histos_" + std::to_string(i) + ".root");
		const char* tempname = temp.c_str();
		TFile f(tempname, "RECREATE");
		ROOT::RDataFrame df("CICADAv2p1p2Ntuplizer/L1TCaloSummaryOutput", 
			"/hdfs/store/user/aloelige/ZeroBias/Paper_Ntuples_27Mar2024/240327_152836/0000/output_" + std::to_string(i) + ".root");
		df.Foreach(unpack, {"modelInput"});
		f.Close();
	}
	return 0;
}
