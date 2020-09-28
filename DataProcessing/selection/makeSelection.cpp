#include "DisappTrksML/TreeMaker/interface/Infos.h"

TString string_tstring(string thing){
	TString thingT = thing;
	return thingT;
}

TString int_tstring(int num){
	return string_tstring(to_string(num));
}

bool passesSelection(TrackInfo track){

	ROOT::Math::XYZVector momentum = ROOT::Math::XYZVector(track.px, track.py, track.pz);
	double eta = momentum.Eta();
	double pt = sqrt(momentum.Perp2());

	if(!(abs(eta) < 2.4)) return false;
	if(track.inGap) return false;
	if(!(abs(track.dRMinJet) > 0.5)) return false;
	return true;
}

void makeSelection(int file = 0, TString dataDir = "", TString outDir = ""){

	dataDir = "/store/user/bfrancis/images_DYJetsToLL_v3/";

	TString oldFileName = dataDir+"hist_"+int_tstring(file)+".root";
	TFile * oldFile = TFile::Open(oldFileName, "read");
	if(oldFile == nullptr) return;
	TTree * oldTree = (TTree*)oldFile->Get("trackImageProducer/tree");

	std::vector<TrackInfo> * v_tracks = new std::vector<TrackInfo>();
	std::vector<RecHitInfo> * v_recHits = new std::vector<RecHitInfo>(); 
	std::vector<GenParticleInfo> * v_genParticles = new std::vector<GenParticleInfo>(); 
	int nPV;

	oldTree->SetBranchAddress("tracks", &v_tracks);
	oldTree->SetBranchAddress("recHits", &v_recHits);
	oldTree->SetBranchAddress("genParticles", &v_genParticles);
	oldTree->SetBranchAddress("nPV", &nPV);

	TFile * newFile = new TFile(outDir+"hist_"+int_tstring(file)+".root", "recreate");
	TTree * newTree = oldTree->CloneTree(0);

	// debug
	cout << oldFileName << endl;
	cout << oldTree->GetEntries() << endl;

	for(int iE = 0; iE < oldTree->GetEntries(); iE++){
		oldTree->GetEntry(iE);
		if(iE > 50) break; // debug

		cout << (*v_tracks).size() << endl;

		// make selections
		// for(int iT = 0; iT < (*v_tracks).size(); iT++){
		// 	if(!passesSelection((*v_tracks)[iT])) continue;
		// }
		newTree->Fill();
	}

	newFile->Write();
}