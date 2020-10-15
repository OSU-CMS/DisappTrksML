#include <string>
#include <iostream>
#include <fstream> 
using namespace std;

#include "Math/Vector3D.h"
#include "TString.h"
#include "TMath.h"
#include "TFile.h"
#include "TTree.h"

#include "Infos.h"

TString string_tstring(string thing){
	TString thingT = thing;
	return thingT;
}

TString int_tstring(int num){
	return string_tstring(to_string(num));
}

template <class T1, class T2, class T3, class T4>
inline
T1 deltaR (T1 eta1, T2 phi1, T3 eta2, T4 phi2) {
	T1 deta = eta1 - eta2;
	T1 dphi = abs(phi1-phi2); 
	if (dphi>T1(M_PI)) dphi-=T1(2*M_PI);  
	return deta*deta + dphi*dphi;
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
        else if(flavor == "muon") return abs(track.deltaRToClosestMuon) < 0.15;
        else if(flavor == "tau") return abs(track.deltaRToClosestTauHad) < 0.15;
        else return false;
}

bool passesSelectionReal(TrackInfo track){
        
	ROOT::Math::XYZVector momentum = ROOT::Math::XYZVector(track.px, track.py, track.pz);
        double eta = momentum.Eta();
	double pt = sqrt(momentum.Perp2());
        if(!(abs(eta) < 2.4)) return false;
        if (track.inGap) return false;
        if (!(abs(track.dRMinJet) > 0.5)) return false;
        if (!(pt > 30)) return false;
        if (!(track.nValidPixelHits >= 4)) return false;
        if (!(track.nValidHits >= 4)) return false;
        if (!(track.missingInnerHits == 0)) return false;
        if (!(track.missingMiddleHits == 0)) return false;
        if (!(track.trackIso / pt < 0.05)) return false;
        if (!(abs(track.d0) < 0.02)) return false;
        if (!(abs(track.dz) < 0.5)) return false;
	if (isReconstructed(track, "muon")) return false;
        if (isReconstructed(track, "tau")) return false;
	//Selections for AMSB
	if (isReconstructed(track, "electron")) return false;
        //Selections for electron events
	//if (!track.passesProbeSelection) return false;
        //if (!track.isTagProbeElectron) return false;
        return true;
}


void makeSelectionReal(int file = 0, TString dataDir = "/data/users/mcarrigan/condor/AMSB/images_chargino_700GeV_1000cm_step3/", TString filelist = ""){


	//int file = std::stoi(argv[1], nullptr, 10);
	//TString dataDir = string_tstring(argv[2]);
	//TString filelist = string_tstring(argv[3]);

	// parameters
	bool Zee = false;
	cout << "In Make Selection" << endl;

	const double minGenParticlePt_ = 10;

	if(filelist.Length()>0){
		string line;
		ifstream infile(filelist);
		if (infile.is_open()) {
			int iLine = 0;
			while(getline(infile,line)) {
				if(iLine == file) {
					file = stod(line);
					break;
				}
		  		iLine += 1;
			}
		infile.close();
  		}
	}

	TString oldFileName = dataDir+"hist_"+int_tstring(file)+".root";
	TFile * oldFile = TFile::Open(oldFileName, "read");
	if(oldFile == nullptr) return;
	TTree * oldTree = (TTree*)oldFile->Get("trackImageProducer/tree");

	vector<TrackInfo> * v_tracks = new vector<TrackInfo>();
	vector<RecHitInfo> * v_recHits = new vector<RecHitInfo>(); 
	vector<GenParticleInfo> * v_genParticles = new vector<GenParticleInfo>(); 
	int nPV;
	unsigned long long eventNumber;
	unsigned int lumiBlockNumber;
	unsigned int runNumber;

	oldTree->SetBranchAddress("tracks", &v_tracks);
	oldTree->SetBranchAddress("recHits", &v_recHits);
	oldTree->SetBranchAddress("genParticles", &v_genParticles);
	oldTree->SetBranchAddress("nPV", &nPV);
	oldTree->SetBranchAddress("eventNumber", &eventNumber);
	oldTree->SetBranchAddress("lumiBlockNumber", &lumiBlockNumber);
	oldTree->SetBranchAddress("runNumber", &runNumber);

	TString newFileName = "hist_"+int_tstring(file)+".root";
	TFile * newFile = new TFile(newFileName, "recreate");
	TTree * eTree = new TTree("eTree","eTree");
	//TTree * eRecoTree = new TTree("eRecoTree","eRecoTree");
        TTree * bTree = new TTree("bTree","bTree");
	vector<TrackInfo> * v_tracks_e = new vector<TrackInfo>();
	vector<TrackInfo> * v_tracks_eReco = new vector<TrackInfo>();
	eTree->Branch("nPV",&nPV);
	eTree->Branch("recHits",&v_recHits);
	eTree->Branch("tracks",&v_tracks_e);
	eTree->Branch("eventNumber", &eventNumber);
	eTree->Branch("lumiBlockNumber", &lumiBlockNumber);
	eTree->Branch("runNumber", &runNumber);
	/*eRecoTree->Branch("nPV",&nPV);
	eRecoTree->Branch("recHits",&v_recHits);
	eRecoTree->Branch("tracks",&v_tracks_eReco);
	eRecoTree->Branch("eventNumber", &eventNumber);
	eRecoTree->Branch("lumiBlockNumber", &lumiBlockNumber);
	eRecoTree->Branch("runNumber", &runNumber);*/
	bTree->Branch("nPV",&nPV);
        bTree->Branch("recHits",&v_recHits);
        bTree->Branch("tracks",&v_tracks_eReco);
        bTree->Branch("eventNumber", &eventNumber);
        bTree->Branch("lumiBlockNumber", &lumiBlockNumber);
        bTree->Branch("runNumber", &runNumber);
	cout << "Running over " << oldFileName << " with " << to_string(oldTree->GetEntries()) << " events." << endl;

	for(int iE = 0; iE < oldTree->GetEntries(); iE++){

		v_tracks_eReco->clear();
		v_tracks_e->clear();

		oldTree->GetEntry(iE);

		// make selections
		for(const auto &track : *v_tracks){

			// selections
			if(!passesSelectionReal(track)) continue;


			// gen matching
			int genMatchedID = 0;
			double genMatchedDR(-1), genMatchedPt(-1);
			int genMatchedID_promptFinalState = 0;
			double genMatchedDR_promptFinalState(-1), genMatchedPt_promptFinalState(-1);
			for(const auto &genParticle : *v_genParticles) {
				if(genParticle.pt < minGenParticlePt_) continue;
				double thisDR = deltaR(genParticle.eta,genParticle.phi,track.eta,track.phi);
				if(genMatchedDR < 0 || thisDR < genMatchedDR) {
					genMatchedDR = thisDR;
					genMatchedID = genParticle.pdgId;
					genMatchedPt = genParticle.pt;
					if(genParticle.isPromptFinalState || genParticle.isDirectPromptTauDecayProductFinalState) {
						genMatchedDR_promptFinalState = thisDR;
						genMatchedID_promptFinalState = genParticle.pdgId;
						genMatchedPt_promptFinalState = genParticle.pt;
					}
				}
			}
			//if(isReconstructed(track, "electron")) v_tracks_eReco->push_back(track);
			//else v_tracks_e->push_back(track);
			if(abs(genMatchedID) == 1000024 || abs(genMatchedID) == 1000022){
 				if(abs(genMatchedDR) < 0.1) {
					if(track.isTagProbeElectron) v_tracks_e->push_back(track);
					else v_tracks_eReco->push_back(track);
				}
			}
		}

		if(v_tracks_eReco->size() > 0) bTree->Fill();
		if(v_tracks_e->size() > 0) eTree->Fill();
	}

	cout << "Saving file " << newFileName << " with" << endl;
	cout << bTree->GetEntries() << " electron reco tracks and" << endl;
	cout << eTree->GetEntries() << " electron tracks." << endl;

	newFile->Write();
}
