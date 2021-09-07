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

    if(!(abs(track.eta) < 2.4)) return false;
    if(!(track.pt > 55)) return false;
    if(track.inGap) return false;
    if(!(track.nValidPixelHits >= 4)) return false;
    if(!(track.nValidHits >= 4)) return false;
    if(!(track.missingInnerHits == 0)) return false;
    if(!(track.missingMiddleHits == 0)) return false;
    
    return true;

}

bool signalSelection(TrackInfo track){

    if(!(track.trackIso /track.pt < 0.05)) return false;
    if(!(abs(track.d0) < 0.02)) return false;
    if(!(abs(track.dz) < 0.5)) return false;
    if(!(abs(track.dRMinJet) > 0.5)) return false;

    //candidate track selection
    if(!(abs(track.deltaRToClosestElectron) > 0.15)) return false;
    if(!(abs(track.deltaRToClosestMuon) > 0.15)) return false;
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
    return min_dz;
}


void selectData(int fileNum = 1, TString dataDir = "/store/user/mcarrigan/Images-v8-NeutrinoGun-MC2017-ext/", TString filelist = ""){
    
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

    // Booleans for creating signal MC, taking full selection on Training data, or creating a 3rd class for pileup
    bool signalMC = false;
    bool fullSelection = false;
    double PU_cut = 0.1;
   
    //TString filename = "images.root";
    //TString filename = dataDir + "images_" + int_tstring(fileNum) + ".root";
    TString filename = dataDir + "hist_" + int_tstring(fileNum) + ".root";
    TFile* myFile = TFile::Open(filename, "read");
    if(myFile == nullptr) return;
    TTree * myTree = (TTree*)myFile->Get("trackImageProducer/tree");

    vector<TrackInfo> * v_tracks = new vector<TrackInfo>();
    vector<RecHitInfo> * v_recHits = new vector<RecHitInfo>(); 
    vector<GenParticleInfo> * v_genParticles = new vector<GenParticleInfo>(); 
    vector<VertexInfo> * v_vertexInfos = new vector<VertexInfo>();
    int nPV;
    unsigned int numTruePV;
    unsigned long long eventNumber;
    unsigned int lumiBlockNumber;
    unsigned int runNumber;
    vector<double> * v_pileupZPosition = new vector<double>();

    myTree->SetBranchAddress("tracks", &v_tracks);
    myTree->SetBranchAddress("recHits", &v_recHits);
    myTree->SetBranchAddress("genParticles", &v_genParticles);
    myTree->SetBranchAddress("nPV", &nPV);
    myTree->SetBranchAddress("numTruePV", &numTruePV);
    myTree->SetBranchAddress("eventNumber", &eventNumber);
    myTree->SetBranchAddress("lumiBlockNumber", &lumiBlockNumber);
    myTree->SetBranchAddress("runNumber", &runNumber);
    myTree->SetBranchAddress("pileupZPosition", &v_pileupZPosition);
    myTree->SetBranchAddress("vertexInfos", &v_vertexInfos);

    TString newFileName = "hist_" + int_tstring(fileNum) + ".root";
    TFile * newFile = new TFile(newFileName, "recreate");
    TTree * fakeTree = new TTree("fakeTree","fakeTree");
    TTree * realTree = new TTree("realTree","realTree");
    TTree * pileupTree = new TTree("pileupTree", "pileupTree");
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
    fakeTree->Branch("pileupZPosition", &v_pileupZPosition);
    fakeTree->Branch("vertexInfos", &v_vertexInfos);

    realTree->Branch("nPV",&nPV);
    realTree->Branch("numTruePV", &numTruePV);
    realTree->Branch("recHits",&v_recHits);
    realTree->Branch("tracks",&v_tracks_real);
    realTree->Branch("genParticles", &v_genParticles);
    realTree->Branch("eventNumber", &eventNumber);
    realTree->Branch("lumiBlockNumber", &lumiBlockNumber);
    realTree->Branch("runNumber", &runNumber);
    realTree->Branch("pileupZPosition", &v_pileupZPosition);
    realTree->Branch("vertexInfos", &v_vertexInfos);

    pileupTree->Branch("nPV",&nPV);
    pileupTree->Branch("numTruePV", &numTruePV);
    pileupTree->Branch("recHits",&v_recHits);
    pileupTree->Branch("genParticles", &v_genParticles);
    pileupTree->Branch("tracks",&v_tracks_pileup);
    pileupTree->Branch("eventNumber", &eventNumber);
    pileupTree->Branch("lumiBlockNumber", &lumiBlockNumber);
    pileupTree->Branch("runNumber", &runNumber);
    pileupTree->Branch("pileupZPosition", &v_pileupZPosition);
    pileupTree->Branch("vertexInfos", &v_vertexInfos);

    cout << "Running over " << filename << " with " << myTree->GetEntries() << " events." << endl;

    //loop over events in file
    for(int ievent = 0; ievent < myTree->GetEntries(); ievent++){
        
        v_tracks_fake->clear();
        v_tracks_real->clear();
        v_tracks_pileup->clear();

        myTree->GetEvent(ievent);
        
        for(const auto &track : *v_tracks){
            
            //look to see if track passes general selections
            if(!trackSelection(track)) continue;
            if(signalMC) if(!signalSelection(track)) continue;
            if(fullSelection) if(!signalSelection(track)) continue;
             
            float genMatchedDR(-1);
            int genMatchedId(0);
            //check to see if track is gen matched
            for(const auto &genParticle : *v_genParticles){
            
                if(genParticle.pt < 10) continue;
                float thisDR = deltaR(genParticle.eta, genParticle.phi, track.eta, track.phi);
                if(genMatchedDR == -1 || thisDR < genMatchedDR){
                    genMatchedDR = thisDR;
                    genMatchedId = genParticle.pdgId;
                }

            }//end of gen particle loop
            
            double pu_dz = pileupMatching(track, *v_pileupZPosition);

            if(signalMC) if(genMatchedId != 1000024 && genMatchedId != 1000022) continue;
            if(genMatchedDR > 0.1){
                //cout << genMatchedDR << " " << pu_dz << endl;
                if(pu_dz > PU_cut) v_tracks_fake->push_back(track);
                else if(pu_dz <= PU_cut) v_tracks_pileup->push_back(track);
            }
            //else if(genMatchedDR > 0.1 && pu_dz <= PU_cut) v_tracks_pileup->push_back(track);
            else v_tracks_real->push_back(track);       
            

            //if(genMatchedDR > 0.1) v_tracks_fake->push_back(track);
            //else v_tracks_real->push_back(track);     

        }//end of tracks loop
    
    if(v_tracks_fake->size() > 0) fakeTree->Fill();
    if(v_tracks_real->size() > 0) realTree->Fill();
    if(v_tracks_pileup->size() > 0) pileupTree->Fill();

    }//end of event loop


    cout << "Saving file " << newFileName << " with" << endl;
    cout << fakeTree->GetEntries() << " fake tracks and" << endl;
    cout << realTree->GetEntries() << " real tracks and" << endl;
    cout << pileupTree->GetEntries() << " pileup tracks." << endl;
    newFile->Write();

}//end of selectData function
