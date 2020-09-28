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

bool isReconstructed(TrackInfo track, string flavor){
	if(flavor == "electron") return abs(track.deltaRToClosestElectron) < 0.15;
	else if(flavor == "muon")return abs(track.deltaRToClosestMuon) < 0.15;
	else if(flavor == "tau")return abs(track.deltaRToClosestTauHad) < 0.15;
	else return false;
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

	TString newFileName = outDir+"hist_"+int_tstring(file)+".root";
	TFile * newFile = new TFile(newFileName, "recreate");
	TTree * newTree = oldTree->CloneTree(0);

	cout << "Running over " << oldFileName << " with " << to_string(oldTree->GetEntries()) << " events." << endl;

	for(int iE = 0; iE < oldTree->GetEntries(); iE++){
		oldTree->GetEntry(iE);

		cout << (*v_tracks).size() << endl;

		// make selections
		for(int iT = 0; iT < (*v_tracks).size(); iT++){

			if(!passesSelection((*v_tracks)[iT])) continue;

			if(isReconstructed(track,"muon")) continue;
			if(isReconstructed(track,"tau")) continue;

		}
		newTree->Fill();
	}

	cout << "Saving file " << newFileName << " with " << newTree->GetEntries() << " events."; 
	newFile->Write();
}