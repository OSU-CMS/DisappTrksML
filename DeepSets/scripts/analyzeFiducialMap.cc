#include <string>
#include <iostream>
#include <fstream>

#include "Math/Vector3D.h"
#include "TString.h"
#include "TMath.h"
#include "TFile.h"
#include "TTree.h"
#include "TChain.h"


#include "../../TreeMaker/interface/Infos.h"


using namespace std;

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
    //if(!(track.missingOuterHits >= 3)) return false;
    //if(!(track.ecalo < 10)) return false;
    return true;
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

bool fiducialSelection(TrackInfo track){
 
    //fiducial map veto regions for singleEle 2017F
    vector<vector<float>> vetoRegions = {{-2.05, -2.55}, {-1.95, -2.15}, {-1.85, -1.95}, {-1.45, 0.75}, {-1.45, 3.05}, {-1.35, -0.15}, {-1.15, -2.05}, {-1.15, -1.95}, 
                                {-1.15, -1.15}, {1.05, 2.85}, {1.05, 2.95}, {1.05, 3.05}, {1.15, -3.15}, {1.35, -1.85}, {1.45, -1.55}, {1.45, 0.55}, 
                                {1.45, 1.45}, {1.45, 2.95}, {1.95, -3.15}, {2.05, 2.05}, {2.05, 2.45}};

    for(int i=0; i<vetoRegions.size(); i++){
        if(abs(track.eta - vetoRegions[i][0]) <= 0.05) return false;
  	if(abs(track.phi - vetoRegions[i][1]) <= 0.05) return false;
    }

    return true;

}

int analyzeFiducialMap(){

    //TString fileDir = "/store/user/bfrancis/images_SingleEle2017F/";
    TString fileDir = "/store/user/mcarrigan/Images-v9-DYJets-MC2017_aMCNLO_ext/0000/";

    TChain* mychain = new TChain("trackImageProducer/tree");

    for(int i=0; i<100; ++i){
        if(i%20==0) cout << "Adding file number: " << i << endl;
        //TString filename = "hist_" + to_string(i) + ".root";
        TString filename = "images_" + to_string(i) + ".root";
        if(access(fileDir + filename, F_OK) != 0) continue;
        mychain->Add(fileDir + filename);
    }
    int numEvents = mychain->GetEntries();

    cout << "Number of events: " << numEvents << endl;

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

    mychain->SetBranchAddress("tracks", &v_tracks);
    mychain->SetBranchAddress("recHits", &v_recHits);
    //mychain->SetBranchAddress("genParticles", &v_genParticles);
    mychain->SetBranchAddress("nPV", &nPV);
    //mychain->SetBranchAddress("numTruePV", &numTruePV);
    //mychain->SetBranchAddress("eventNumber", &eventNumber);
    //mychain->SetBranchAddress("lumiBlockNumber", &lumiBlockNumber);
    //mychain->SetBranchAddress("runNumber", &runNumber);
    //mychain->SetBranchAddress("pileupZPosition", &v_pileupZPosition);
    //mychain->SetBranchAddress("vertexInfos", &v_vertexInfos);

    int passingEle = 0;
    int totalEle = 0;
    int passingTrack(0), totalTrack(0);
    //int pass1(0), pass2(0), pass3(0), pass4(0), pass5(0), pass6(0), pass7(0);

    for(int ievent = 0; ievent < numEvents; ievent++){
     
        if(ievent%1000 == 0) cout << "Working on event: " << ievent << endl;
        mychain->GetEvent(ievent);

        for(const auto &track : *v_tracks){

            if(!trackSelection(track)) continue;
            //if(!signalSelection(track)) continue;

            if(track.isTagProbeElectron) {
                totalEle++;
                if(fiducialSelection(track)) passingEle++;
            }
            else {
                totalTrack++;
                if(fiducialSelection(track)) passingTrack++;
            }
            
            /*if(!(abs(track.eta) < 2.4)){
                continue;    
            }
            pass1++;
            if(!(track.pt > 55)){
                continue;
            }
            pass2++;
            if(track.inGap){
                continue;
            }
            pass3++;
            if(!(track.nValidPixelHits >= 4)){
                continue;
            }
            pass4++;
            if(!(track.nValidHits >= 4)){
                continue;
            }
            pass5++;
            if(!(track.missingInnerHits == 0)){
                continue;
            }
            pass6++;
            if(!(track.missingMiddleHits == 0)){
                continue;
            }
            pass7++;*/

        } //end tracks loop
    } //end event loop

    //cout << pass1 << " " << pass2 << " " << pass3 << " " << pass4 << " " << pass5 << " " << pass6 << " " << pass7 << endl;
    cout << "Total T&P electrons: " << totalEle << ", Number of passing electrons: " << passingEle << endl;

    return 0;

}
