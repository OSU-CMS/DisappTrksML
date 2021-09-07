#include <string>
#include <iostream>
#include <fstream> 

#include "Math/Vector3D.h"
#include "TString.h"
#include "TMath.h"
#include "TFile.h"
#include "TTree.h"

#include "Infos.h"


using namespace std;

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
	return sqrt(deta*deta + dphi*dphi);
}

bool trackSelection(TrackInfo track){

    //isolated track selection
    if(!(abs(track.eta) < 2.4)) return false;
    if(!(track.pt > 55)) return false;
    if(track.inGap) return false;
    if(!(track.nValidPixelHits >= 4)) return false;
    if(!(track.nValidHits >= 4)) return false;
    if(!(track.missingInnerHits == 0)) return false;
    if(!(track.missingMiddleHits == 0)) return false;
    if(!(track.trackIso /track.pt < 0.05)) return false;
    //if(!(abs(track.d0) < 0.02)) return false;
    if(!(abs(track.dz) < 0.5)) return false;
    if(!(abs(track.dRMinJet) > 0.5)) return false;

    //candidate track selection
    if(!(abs(track.deltaRToClosestElectron) > 0.15)) return false;
    //if(!(abs(track.deltaRToClosestMuon) > 0.15)) return false;
    if(!(abs(track.deltaRToClosestTauHad) > 0.15)) return false;   
    
    //disappearing track selection
    if(!(track.missingOuterHits >= 3)) return false;
    if(!(track.ecalo < 10)) return false;
    return true;

}


double pileupMatching(TrackInfo track, vector<double> pileupZPosition){
    double min_dz = 10e6;
    for(int i = 0; i < pileupZPosition.size(); i++){
        double dZ = fabs(track.vz - pileupZPosition[i]);
        if(dZ < min_dz) min_dz = dZ;
    }
    cout << "DZ Matched " << min_dz << endl;
    return min_dz;
}


//Test files
// ZeroBias  /store/user/mcarrigan/Images_v8_ZeroBias_2017/

void selectDataReal(int fileNum = 1, TString dataDir = "/store/user/mcarrigan/Images-v8-singleMuon2017F/", TString filelist = ""){
    
    if(filelist.Length()>0){
        string line;
        ifstream infile(filelist);
        if (infile.is_open()) {
            int iLine = 0;
            while(getline(infile,line)) {
                 if(iLine == fileNum) {
                     fileNum = stod(line);
                     break;
                 }
                 iLine += 1;
            }
            infile.close();
        }
    }



    TString filename = dataDir + "images_" + int_tstring(fileNum) + ".root";
    TFile* myFile = TFile::Open(filename, "read");
    if(myFile == nullptr) return;
    TTree * myTree = (TTree*)myFile->Get("trackImageProducer/tree");

    // boolean to select only Z->ll tracks
    bool ZtoMuMuSelection = true;

    //boolean to apply basic selection 
    bool basicSelection = false;

    //boolean to apply track selection
    bool selectTracks = true;

    vector<TrackInfo> * v_tracks = new vector<TrackInfo>();
    vector<RecHitInfo> * v_recHits = new vector<RecHitInfo>(); 
    vector<GenParticleInfo> * v_genParticles = new vector<GenParticleInfo>(); 
    vector<VertexInfo> * v_vertexInfos = new vector<VertexInfo>(); 
    int nPV;
    int numTruePV;
    unsigned long long eventNumber;
    unsigned int lumiBlockNumber;
    unsigned int runNumber;
    bool passMETFilters;
    int numGoodPVs;
    double metNoMu;
    double numGoodJets;
    double dijetDeltaPhiMax;
    double leadingJetMetPhi;

    myTree->SetBranchAddress("tracks", &v_tracks);
    myTree->SetBranchAddress("recHits", &v_recHits);
    myTree->SetBranchAddress("genParticles", &v_genParticles);
    myTree->SetBranchAddress("nPV", &nPV);
    myTree->SetBranchAddress("numTruePV", &numTruePV);
    myTree->SetBranchAddress("eventNumber", &eventNumber);
    myTree->SetBranchAddress("lumiBlockNumber", &lumiBlockNumber);
    myTree->SetBranchAddress("runNumber", &runNumber);
    myTree->SetBranchAddress("passMETFilters", &passMETFilters);
    myTree->SetBranchAddress("numGoodPVs", &numGoodPVs);
    myTree->SetBranchAddress("metNoMu", &metNoMu);
    myTree->SetBranchAddress("numGoodJets", &numGoodJets);
    myTree->SetBranchAddress("dijetDeltaPhiMax", &dijetDeltaPhiMax);
    myTree->SetBranchAddress("leadingJetMetPhi", &leadingJetMetPhi);
    myTree->SetBranchAddress("vertexInfos", &v_vertexInfos);

    TString newFileName = "hist_" + int_tstring(fileNum) + ".root";
    TFile * newFile = new TFile(newFileName, "recreate");
    TTree * fakeTree = new TTree("fakeTree","fakeTree");
    TTree * realTree = new TTree("realTree","realTree");
    TTree * pileupTree = new TTree("pileupTree","pileupTree");
    vector<TrackInfo> * v_tracks_fake = new vector<TrackInfo>();
    vector<TrackInfo> * v_tracks_real = new vector<TrackInfo>();
    vector<TrackInfo> * v_tracks_pileup = new vector<TrackInfo>();
    fakeTree->Branch("nPV",&nPV);
    fakeTree->Branch("numTruePV", &numTruePV);
    fakeTree->Branch("recHits",&v_recHits);
    fakeTree->Branch("genParticles", &v_genParticles);
    fakeTree->Branch("tracks",&v_tracks_fake);
    fakeTree->Branch("eventNumber", &eventNumber);
    fakeTree->Branch("lumiBlockNumber", &lumiBlockNumber);
    fakeTree->Branch("runNumber", &runNumber);
    fakeTree->Branch("vertexInfos", &v_vertexInfos);

    realTree->Branch("nPV",&nPV);
    fakeTree->Branch("numTruePV", &numTruePV);
    realTree->Branch("recHits",&v_recHits);
    realTree->Branch("tracks",&v_tracks_real);
    realTree->Branch("genParticles", &v_genParticles);
    realTree->Branch("eventNumber", &eventNumber);
    realTree->Branch("lumiBlockNumber", &lumiBlockNumber);
    realTree->Branch("runNumber", &runNumber);
    realTree->Branch("vertexInfos", &v_vertexInfos);

    pileupTree->Branch("nPV",&nPV);
    pileupTree->Branch("numTruePV", &numTruePV);
    pileupTree->Branch("recHits",&v_recHits);
    pileupTree->Branch("tracks",&v_tracks_real);
    pileupTree->Branch("genParticles", &v_genParticles);
    pileupTree->Branch("eventNumber", &eventNumber);
    pileupTree->Branch("lumiBlockNumber", &lumiBlockNumber);
    pileupTree->Branch("runNumber", &runNumber);
    pileupTree->Branch("vertexInfos", &v_vertexInfos);

    cout << "Running over " << filename << " with " << myTree->GetEntries() << " events." << endl;

    //loop over events in file
    for(int ievent = 0; ievent < myTree->GetEntries(); ievent++){
        
        v_tracks_fake->clear();
        v_tracks_real->clear();

        myTree->GetEvent(ievent);
        
        for(const auto &track : *v_tracks){
            
            //look to see if track passes general selections
            if(selectTracks && !trackSelection(track)) continue;
            if(ZtoMuMuSelection){
                if(!track.isTagProbeMuon) continue;
            }
            if(basicSelection){
                if(!passMETFilters) continue;
                if(!(numGoodPVs >= 1)) continue;
                if(!(metNoMu > 120)) continue;
                if(!(numGoodJets >= 1)) continue;
                if(!(dijetDeltaPhiMax < 2.5)) continue;
                if(!(leadingJetMetPhi > 0.5)) continue;
            }
            v_tracks_real->push_back(track);            

        }//end of tracks loop
    
    if(v_tracks_real->size() > 0) realTree->Fill();

    }//end of event loop


    cout << "Saving file " << newFileName << " with" << endl;
    cout << realTree->GetEntries() << " real tracks." << endl;
    newFile->Write();

}//end of selectData function
