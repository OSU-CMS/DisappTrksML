#ifndef TRACK_IMAGE_ANALYZER

#define TRACK_IMAGE_ANALYZER

#define M_PI_2 1.57079632679489661923

#include "DisappTrksML/TreeMaker/interface/Infos.h"

#include <map>
#include <string>

#include "TTree.h"

#include "TH1D.h"
#include "TH2D.h"
#include "TVector3.h"
#include "TLorentzVector.h"

#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/TauReco/interface/PFTau.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/JetReco/interface/PFJet.h"

#include "DataFormats/TauReco/interface/PFTauDiscriminator.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"

#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"

#include "CondFormats/EcalObjects/interface/EcalChannelStatus.h"
#include "CondFormats/DataRecord/interface/EcalChannelStatusRcd.h"

#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"

//include Cscs
#include "DataFormats/CSCRecHit/interface/CSCSegment.h"
#include "DataFormats/CSCRecHit/interface/CSCSegmentCollection.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/CSCGeometry/interface/CSCGeometry.h"
#include "Geometry/CSCGeometry/interface/CSCChamber.h"
#include "Geometry/CSCGeometry/interface/CSCLayer.h"
#include "Geometry/CSCGeometry/interface/CSCLayerGeometry.h"
//include Dts
#include "DataFormats/DTRecHit/interface/DTRecHit1D.h"
#include "DataFormats/DTRecHit/interface/DTRecSegment4D.h"
#include "DataFormats/DTRecHit/interface/DTRecSegment2D.h"
#include "DataFormats/DTRecHit/interface/DTRecHitCollection.h"
#include "DataFormats/DTRecHit/interface/DTRecSegment2DCollection.h"
#include "DataFormats/DTRecHit/interface/DTRecSegment4DCollection.h"
#include "Geometry/DTGeometry/interface/DTChamber.h"
#include "Geometry/DTGeometry/interface/DTLayer.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "DataFormats/MuonDetId/interface/DTLayerId.h"
#include "DataFormats/MuonDetId/interface/DTWireId.h"
//jets
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
// RPC hits
#include "DataFormats/RPCRecHit/interface/RPCRecHit.h"
#include "DataFormats/RPCRecHit/interface/RPCRecHitCollection.h"
#include "Geometry/RPCGeometry/interface/RPCGeometry.h"
#include "Geometry/RPCGeometry/interface/RPCChamber.h"

#include "DataFormats/Math/interface/deltaPhi.h"

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/InputTag.h"

using namespace std;

enum DetType { None, EB, EE, ES, HCAL, CSC, DT, RPC };

class TrackImageProducer : public edm::EDAnalyzer {
   public:
      explicit TrackImageProducer(const edm::ParameterSet &);
      ~TrackImageProducer();

   private:
      void analyze(const edm::Event &, const edm::EventSetup &);

      void getGeometries(const edm::EventSetup &);

      const TrackInfo getTrackInfo(const reco::Track &, const vector<reco::Track> &, const reco::Vertex &, const vector<reco::PFJet> &) const;
      void getRecHits(const edm::Event &);
      void getGenParticles(const reco::CandidateView &);

      const bool isProbeTrack(const TrackInfo) const;

      const math::XYZVector getPosition(const DetId &) const;
      const math::XYZVector getPosition(const CSCSegment&) const;
      const math::XYZVector getPosition(const DTRecSegment4D&) const;
      const math::XYZVector getPosition(const RPCRecHit&) const;

      const double minDRBadEcalChannel(const reco::Track &) const;
      void getChannelStatusMaps();

      edm::InputTag tracks_;
      edm::InputTag genParticles_;
      edm::InputTag electrons_, muons_, taus_;
      edm::InputTag pfCandidates_;
      edm::InputTag vertices_;
      edm::InputTag jets_;
      edm::InputTag EBRecHits_, EERecHits_, ESRecHits_;
      edm::InputTag HBHERecHits_;
      edm::InputTag cscSegments_, dtRecSegments_, rpcRecHits_;
      edm::InputTag tauDecayModeFinding_, tauElectronDiscriminator_, tauMuonDiscriminator_;

      const double minGenParticlePt_;
      const double minTrackPt_;
      const double maxRelTrackIso_;

      edm::EDGetTokenT<vector<reco::Track> >       tracksToken_;
      edm::EDGetTokenT<reco::CandidateView>        genParticlesToken_;
      edm::EDGetTokenT<vector<reco::GsfElectron> > electronsToken_;
      edm::EDGetTokenT<vector<reco::Muon> >        muonsToken_;
      edm::EDGetTokenT<vector<reco::PFTau> >       tausToken_;
      edm::EDGetTokenT<reco::PFTauDiscriminator>   tauDecayModeFindingToken_, tauElectronDiscriminatorToken_, tauMuonDiscriminatorToken_;
      edm::EDGetTokenT<vector<reco::PFCandidate> > pfCandidatesToken_;
      edm::EDGetTokenT<vector<reco::Vertex> >      verticesToken_;
      edm::EDGetTokenT<vector<reco::PFJet> >       jetsToken_;
      edm::EDGetTokenT<EBRecHitCollection>         EBRecHitsToken_;
      edm::EDGetTokenT<EERecHitCollection>         EERecHitsToken_;
      edm::EDGetTokenT<ESRecHitCollection>         ESRecHitsToken_;
      edm::EDGetTokenT<HBHERecHitCollection>       HBHERecHitsToken_;
      edm::EDGetTokenT<CSCSegmentCollection>       CSCSegmentsToken_;
      edm::EDGetTokenT<DTRecSegment4DCollection>   DTRecSegmentsToken_;
      edm::EDGetTokenT<RPCRecHitCollection>        RPCRecHitsToken_;

      edm::ESHandle<CaloGeometry> caloGeometry_;
      edm::ESHandle<CSCGeometry>  cscGeometry_;
      edm::ESHandle<DTGeometry>   dtGeometry_;
      edm::ESHandle<RPCGeometry>  rpcGeometry_;

      edm::ESHandle<EcalChannelStatus> ecalStatus_;

      edm::Service<TFileService> fs_;
      TTree * tree_;

      vector<TrackInfo> trackInfos_;
      vector<RecHitInfo> recHitInfos_;
      vector<GenParticleInfo> genParticleInfos_;
      int nPV_;

      map<DetId, vector<double> > EcalAllDeadChannelsValMap_;
      map<DetId, vector<int> >    EcalAllDeadChannelsBitMap_;
};

#endif
