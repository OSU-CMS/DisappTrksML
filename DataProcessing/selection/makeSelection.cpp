#include "DisappTrksML/TreeMaker/interface/Infos.h"
#include "DataFormats/Math/interface/deltaR.h"

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
	else if(flavor == "muon") return abs(track.deltaRToClosestMuon) < 0.15;
	else if(flavor == "tau") return abs(track.deltaRToClosestTauHad) < 0.15;
	else return false;
}

void makeSelection(int file = 0, TString dataDir = "", TString filelist = "test.txt"){

	// parameters
	dataDir = "/store/user/bfrancis/images_DYJetsToLL_v3/";
	bool Zee = false;

	const double minGenParticlePt_ = 10;

	if(filelist.Length()>0){
		string line;
		ifstream infile(filelist);
		if (infile.is_open()) {
			int iLine = 0;
			while(getline(infile,line)) {
				if(iLine == file) file = std::stoi(line);
		  		iLine += 1;
			}
		infile.close();
  		}
	}

	TString oldFileName = dataDir+"hist_"+int_tstring(file)+".root";
	TFile * oldFile = TFile::Open(oldFileName, "read");
	if(oldFile == nullptr) return;
	TTree * oldTree = (TTree*)oldFile->Get("trackImageProducer/tree");

	std::vector<TrackInfo> * v_tracks = new std::vector<TrackInfo>();
	std::vector<RecHitInfo> * v_recHits = new std::vector<RecHitInfo>(); 
	std::vector<GenParticleInfo> * v_genParticles = new std::vector<GenParticleInfo>(); 
	int nPV;
	unsigned long long eventNumber;
	unsigned int lumiBlockNumber;
	unsigned int runNumber;

	oldTree->SetBranchAddress("tracks", &v_tracks);
	oldTree->SetBranchAddress("recHits", &v_recHits);
	oldTree->SetBranchAddress("genParticles", &v_genParticles);
	oldTree->SetBranchAddress("nPV", &nPV);
	// oldTree->SetBranchAddress("eventNumber", &eventNumber);
	// oldTree->SetBranchAddress("lumiBlockNumber", &lumiBlockNumber);
	// oldTree->SetBranchAddress("runNumber", &runNumber);

	TString newFileName = "hist_"+int_tstring(file)+".root";
	TFile * newFile = new TFile(newFileName, "recreate");
	TTree * eTree = new TTree("eTree","eTree");
	TTree * bTree = new TTree("bTree","bTree");
	std::vector<TrackInfo> * v_tracks_e = new std::vector<TrackInfo>();
	std::vector<TrackInfo> * v_tracks_b = new std::vector<TrackInfo>();
	eTree->Branch("nPV",&nPV);
	eTree->Branch("recHits",&v_recHits);
	eTree->Branch("tracks",&v_tracks_e);
	// eTree->Branch("eventNumber", &eventNumber);
	// eTree->Branch("lumiBlockNumber", &lumiBlockNumber);
	// eTree->Branch("runNumber", &runNumber);
	bTree->Branch("nPV",&nPV);
	bTree->Branch("recHits",&v_recHits);
	bTree->Branch("tracks",&v_tracks_b);
	// bTree->Branch("eventNumber", &eventNumber);
	// bTree->Branch("lumiBlockNumber", &lumiBlockNumber);
	// bTree->Branch("runNumber", &runNumber);

	cout << "Running over " << oldFileName << " with " << to_string(oldTree->GetEntries()) << " events." << endl;

	for(int iE = 0; iE < oldTree->GetEntries(); iE++){

		v_tracks_b->clear();
		v_tracks_e->clear();

		oldTree->GetEntry(iE);

		// make selections
		for(const auto &track : *v_tracks){

			// selections
			if(!passesSelection(track)) continue;

			if(isReconstructed(track,"muon")) continue;
			if(isReconstructed(track,"tau")) continue;

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
			if(abs(genMatchedID) == 11 and abs(genMatchedDR) < 0.1) v_tracks_e->push_back(track);
			else v_tracks_b->push_back(track);
		}

		if(Zee){
			if(v_tracks_e->size() < 2) continue;
		}
		if(v_tracks_e->size() > 0) eTree->Fill();
		if(v_tracks_b->size() > 0) bTree->Fill();
	}

	cout << "Saving file " << newFileName << " with" << endl;
	cout << eTree->GetEntries() << " electron tracks and" << endl;
	cout << bTree->GetEntries() << " background tracks." << endl;

	newFile->Write();
}