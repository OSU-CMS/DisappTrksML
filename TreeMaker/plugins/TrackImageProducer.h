#ifndef TRACK_IMAGE_ANALYZER

#define TRACK_IMAGE_ANALYZER

#include <map>
#include <string>

#include "TTree.h"

#include "TH1D.h"
#include "TH2D.h"
#include "TVector3.h"
#include "TLorentzVector.h"

#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/TauReco/interface/PFTau.h"
#include "DataFormats/TrackReco/interface/Track.h"

#include "DataFormats/TauReco/interface/PFTauDiscriminator.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"

#include "DataFormats/Math/interface/deltaPhi.h"

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/InputTag.h"

using namespace std;

enum DetType { None, ECAL, HCAL, Muon };

class TrackImageProducer : public edm::EDAnalyzer {
   public:
      explicit TrackImageProducer(const edm::ParameterSet &);
      ~TrackImageProducer();

   private:
      void analyze(const edm::Event &, const edm::EventSetup &);

      const double getTrackIsolation(const reco::Track &, const vector<reco::Track> &) const;
      void getImage(const reco::Track &, const EBRecHitCollection &, const EERecHitCollection &, const HBHERecHitCollection &);
      const math::XYZVector getPosition(const DetId &) const;

      edm::InputTag tracks_;
      edm::InputTag genParticles_;
      edm::InputTag electrons_, muons_, taus_;
      edm::InputTag pfCandidates_;
      edm::InputTag EBRecHits_;
      edm::InputTag EERecHits_;
      edm::InputTag HBHERecHits_;
      edm::InputTag tauDecayModeFinding_, tauElectronDiscriminator_, tauMuonDiscriminator_;

      const double minGenParticlePt_;
      const double minTrackPt_;
      const double maxRelTrackIso_;

      const double maxDEtaTrackRecHit_;
      const double maxDPhiTrackRecHit_;

      edm::EDGetTokenT<vector<reco::Track> >       tracksToken_;
      edm::EDGetTokenT<vector<reco::GenParticle> > genParticlesToken_;
      edm::EDGetTokenT<vector<reco::GsfElectron> > electronsToken_;
      edm::EDGetTokenT<vector<reco::Muon> >        muonsToken_;
      edm::EDGetTokenT<vector<reco::PFTau> >       tausToken_;
      edm::EDGetTokenT<reco::PFTauDiscriminator>   tauDecayModeFindingToken_, tauElectronDiscriminatorToken_, tauMuonDiscriminatorToken_;
      edm::EDGetTokenT<vector<reco::PFCandidate> > pfCandidatesToken_;
      edm::EDGetTokenT<EBRecHitCollection>         EBRecHitsToken_;
      edm::EDGetTokenT<EERecHitCollection>         EERecHitsToken_;
      edm::EDGetTokenT<HBHERecHitCollection>       HBHERecHitsToken_;

      edm::ESHandle<CaloGeometry> caloGeometry_;

      edm::Service<TFileService> fs_;
      TTree * tree_;

      vector<double> recHits_dEta, recHits_dPhi, recHits_energy;
      vector<int> recHits_detType;

      int genMatchedID;
      double genMatchedDR;
      double deltaRToClosestElectron, deltaRToClosestMuon, deltaRToClosestTauHad;
      double trackEta, trackPhi, trackPt, trackIso;
      // dedx
      // t&p info...

};

#endif
