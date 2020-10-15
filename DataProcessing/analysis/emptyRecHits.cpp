#include <string>
#include <iostream>
#include <fstream> 
using namespace std;

#include "Math/Vector3D.h"
#include "TString.h"
#include "TMath.h"
#include "TFile.h"
#include "TTree.h"
#include "TChain.h"
#include "Infos.h"

TString dataDir = "/store/user/mcarrigan/disappearingTracks/images_singleEle2017F_V2/";

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
	if(track.nValidPixelHits < 4 || track.nLayersWithMeasurement < 4) return false;
	if (!(pt > 30)) return false;
	if (!(track.nValidPixelHits >= 4)) return false;
	if (!(track.nValidHits >= 4)) return false;
	if (!(track.missingInnerHits == 0)) return false;
	if (!(track.missingMiddleHits == 0)) return false;
	if (!(track.trackIso / pt < 0.05)) return false;
	if (!(abs(track.d0) < 0.02)) return false;
	if (!(abs(track.dz) < 0.5)) return false;
	if (!(track.isTagProbeElectron)) return false;
	if (!(track.passesProbeSelection)) return false;
	return true;
}


bool isReconstructed(TrackInfo track, string flavor){
	if(flavor == "electron") return abs(track.deltaRToClosestElectron) < 0.15;
	else if(flavor == "muon") return abs(track.deltaRToClosestMuon) < 0.15;
	else if(flavor == "tau") return abs(track.deltaRToClosestTauHad) < 0.15;
	else return false;
}

void emptyRecHits(){

	TChain myChain("chain");
	for(int i=0; i < 2500; i++){
		TString filename = dataDir + "hist_"+int_tstring(i)+".root/trackImageProducer/tree";
		myChain.Add(filename);
	}

	// parameters
	bool Zee = false;

	//TTree * oldTree = (TTree*)myChain.GetTree();

	vector<TrackInfo> * v_tracks = new vector<TrackInfo>();
	vector<RecHitInfo> * v_recHits = new vector<RecHitInfo>(); 
	int nPV;
	unsigned long long eventNumber;
	unsigned int lumiBlockNumber;
	unsigned int runNumber;

	myChain.SetBranchAddress("tracks", &v_tracks);
	myChain.SetBranchAddress("recHits", &v_recHits);
	myChain.SetBranchAddress("nPV", &nPV);
	myChain.SetBranchAddress("eventNumber", &eventNumber);
	myChain.SetBranchAddress("lumiBlockNumber", &lumiBlockNumber);
	myChain.SetBranchAddress("runNumber", &runNumber);

	TString newFileName = "emptyRecHits.root";
	TFile * newFile = new TFile(newFileName, "recreate");
	TTree * eTree = new TTree("eTree","eTree");
	vector<TrackInfo> * v_tracks_e = new vector<TrackInfo>();
	eTree->Branch("nPV",&nPV);
	eTree->Branch("recHits",&v_recHits);
	eTree->Branch("tracks",&v_tracks_e);
	eTree->Branch("eventNumber", &eventNumber);
	eTree->Branch("lumiBlockNumber", &lumiBlockNumber);
	eTree->Branch("runNumber", &runNumber);

	//cout << "Running over " << oldFileName << " with " << to_string(oldTree->GetEntries()) << " events." << endl;
	
	for(int iE = 0; iE < myChain.GetEntries(); iE++){
		//if(iE%1000 ==0) cout << "Working on Event: " << iE << endl;
		v_tracks_e->clear();
		bool Debug = false;
		myChain.GetEntry(iE);
		/*if(eventNumber == 28721279 && lumiBlockNumber == 20 && runNumber ==305044){
			cout << "Found event" << endl;
			Debug = true;
		}*/
		// make selections
		for(const auto &track : *v_tracks){

			float hitEnergy = 0;
			// selections
			if(!passesSelection(track)) continue;
			//if(isReconstructed(track,"electron")) continue;
			if(isReconstructed(track,"muon")) continue;
			if(isReconstructed(track,"tau")) continue;
			//if(Debug) cout << "Passes reconstruction" << "Track eta: " << track.eta << " track phi: " << track.phi << endl;
			int rec_hitCount = 0;
			for(const auto &recHits : *v_recHits){
				if(recHits.detType != 1 && recHits.detType != 2) continue;
				if(abs(recHits.eta - track.eta) > 0.25) continue;
				float dPhi = abs(recHits.phi - track.phi);
				if(dPhi > M_PI) {
					dPhi -= round(dPhi / (2.*M_PI))*2*M_PI;
				}
				if(abs(dPhi) > 0.25) continue;
				hitEnergy += recHits.energy;
			}
			//if(Debug) cout << "Hit Energy: " << hitEnergy << " Rec Hit Count: " << rec_hitCount << endl;
			if(hitEnergy == 0) cout << "Event number: " << eventNumber << " lumi: " << lumiBlockNumber << " run: " << runNumber << endl;
			if(hitEnergy ==0) v_tracks_e->push_back(track);
		}

		if(Zee){
			if(v_tracks_e->size() < 2) continue;
		}
		if(v_tracks_e->size() > 0) eTree->Fill();
	}

	cout << "Saving file " << newFileName << " with" << endl;
	cout << eTree->GetEntries() << " electron tracks and" << endl;

	newFile->Write();
}
