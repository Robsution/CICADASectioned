#include <vector>
#include <ROOT/RVec.hxx>
#include <iostream>
#include <typeinfo>
#include <TTreeReaderArray.h>

int iter = 0;

void unpack(ROOT::RVec<UShort_t> column) {
    TH2F h("h", "title;eta;phi", 14, 0, 14, 18, 0, 18);
    for (int i = 0; i <= 14; ++i){
        for (int j = 0; j <= 18; ++j){
	    h.SetBinContent(i,j,column[i*14+j]);
        }
    }
    h.Write("h");
}

int a_get_data(){
	TFile f("A_histos.root", "RECREATE");
	ROOT::RDataFrame df("CICADAv2p1p2Ntuplizer/L1TCaloSummaryOutput", "output_1.root");
	df.Foreach(unpack, {"modelInput"});
	f.Close();
	return 0;
}
