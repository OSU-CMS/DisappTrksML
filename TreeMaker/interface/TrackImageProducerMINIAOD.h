#ifndef TRACK_IMAGE_PRODUCER_MINIAOD
#define TRACK_IMAGE_PRODUCER_MINIAOD

#define M_PI_2 1.57079632679489661923

#include <map>

#include "DisappTrksML/TreeMaker/interface/Infos.h"

#include "DisappTrks/CandidateTrackProducer/interface/CandidateTrack.h"

// MINIAOD
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/PatCandidates/interface/Electron.h"
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/PatCandidates/interface/MET.h"
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/PatCandidates/interface/PackedGenParticle.h"
#include "DataFormats/PatCandidates/interface/Tau.h"
#include "DataFormats/PatCandidates/interface/TriggerObjectStandAlone.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "FWCore/Common/interface/TriggerNames.h"

// recHits
#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"

#include "CondFormats/EcalObjects/interface/EcalChannelStatus.h"
#include "CondFormats/DataRecord/interface/EcalChannelStatusRcd.h"

#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"

#include "DataFormats/CSCRecHit/interface/CSCSegment.h"
#include "DataFormats/CSCRecHit/interface/CSCSegmentCollection.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/CSCGeometry/interface/CSCGeometry.h"
#include "Geometry/CSCGeometry/interface/CSCChamber.h"
#include "Geometry/CSCGeometry/interface/CSCLayer.h"
#include "Geometry/CSCGeometry/interface/CSCLayerGeometry.h"

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

#include "DataFormats/RPCRecHit/interface/RPCRecHit.h"
#include "DataFormats/RPCRecHit/interface/RPCRecHitCollection.h"
#include "Geometry/RPCGeometry/interface/RPCGeometry.h"
#include "Geometry/RPCGeometry/interface/RPCChamber.h"

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "TTree.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "DataFormats/Math/interface/deltaPhi.h"
#include "DataFormats/TrackReco/interface/DeDxHitInfo.h"
#include "RecoTracker/DeDx/interface/DeDxTools.h"
#include "DataFormats/TrackReco/interface/DeDxData.h"
#include "DataFormats/PatCandidates/interface/IsolatedTrack.h"

using namespace std;

enum DetType { None, EB, EE, ES, HCAL, CSC, DT, RPC };

class TrackImageProducerMINIAOD : public edm::EDAnalyzer {
   public:
      explicit TrackImageProducerMINIAOD(const edm::ParameterSet &);
      ~TrackImageProducerMINIAOD();

   private:
      void analyze(const edm::Event &, const edm::EventSetup &);

      void getGeometries(const edm::EventSetup &);
      void findMatchedIsolatedTrack (const edm::Handle<vector<pat::IsolatedTrack> >&isolatedTracks, 
                                     edm::Ref<vector<pat::IsolatedTrack> >&matchedIsolatedTrack, 
                                     double &dRToMatchedIsolatedTrack, const CandidateTrack &track) const;
      void findMatchedGenTrack (const edm::Handle<vector<reco::Track> >&genTracks,
                                     edm::Ref<vector<reco::Track> >&matchedGenTrack,
                                     double &dRToMatchedGenTrack, const CandidateTrack &track) const;
      int countGoodPrimaryVertices(const vector<reco::Vertex> &) const;
      int countGoodJets(const vector<pat::Jet> &) const;
      double getMaxDijetDeltaPhi(const vector<pat::Jet> &) const;
      double getLeadingJetMetPhi(const vector<pat::Jet> &, const pat::MET &) const;

      void getTracks(const edm::Handle<vector<CandidateTrack> > tracks, 
                     const reco::Vertex &, 
                     const vector<pat::Jet> &,
                     const vector<pat::Electron> &,
                     const vector<pat::Muon> &,
                     const vector<pat::Tau> &,
                     const vector<pat::Electron> &tagElectrons,
                     const vector<pat::Muon> &tagMuons,
                     const pat::MET &, 
                     const edm::Handle<vector<pat::IsolatedTrack> >, 
		     const edm::Handle<reco::DeDxHitInfoAss>,
                     const edm::Handle<edm::ValueMap<reco::DeDxData> > dEdxStrip,
                     const edm::Handle<edm::ValueMap<reco::DeDxData> > dEdxPixel,
                     const edm::Handle<vector<reco::Track> > genTracks);

      void getRecHits(const edm::Event &);
      void getGenParticles(const reco::CandidateView &);

      const math::XYZVector getPosition(const DetId &) const;
      const math::XYZVector getPosition(const CSCSegment&) const;
      const math::XYZVector getPosition(const CSCRecHit2D&) const;
      const math::XYZVector getPosition(const DTRecSegment4D&) const;
      const math::XYZVector getPosition(const DTRecHit1D&) const;
      const math::XYZVector getPosition(const RPCRecHit&) const;

      vector<pat::Electron> getTagElectrons(const edm::Event &,
                                            const edm::TriggerResults &,
                                            const vector<pat::TriggerObjectStandAlone> &,
                                            const reco::Vertex &,
                                            const vector<pat::Electron> &);

      vector<pat::Muon> getTagMuons(const edm::Event &,
                                    const edm::TriggerResults &,
                                    const vector<pat::TriggerObjectStandAlone> &,
                                    const reco::Vertex &,
                                    const vector<pat::Muon> &);

      const bool isProbeTrack(const TrackInfo) const;

      // bit values reflect tag+probe status and charge products:
      //     0b<in same-sign pair><in opposite-sign pair>
      //     So 0 = 0b00 : not in any TP pair
      //        1 = 0b01 : in OS TP pair
      //        2 = 0b10 : in SS TP pair
      //        3 = 0b11 : in both an OS and SS pair
      const unsigned int isTagProbeElePair(const CandidateTrack &, const pat::Electron &) const;
      const unsigned int isTagProbeMuonPair(const CandidateTrack &, const pat::Muon &) const;
      const unsigned int isTagProbeTauToElePair(const CandidateTrack &, const pat::Electron &, const pat::MET &) const;
      const unsigned int isTagProbeTauToMuonPair(const CandidateTrack &, const pat::Muon &, const pat::MET &) const;

      const double minDRBadEcalChannel(const CandidateTrack &) const;
      void getChannelStatusMaps();

      edm::InputTag triggers_;
      edm::InputTag trigObjs_;
      edm::InputTag tracks_;
      edm::InputTag genParticles_;
      edm::InputTag met_;
      edm::InputTag electrons_, muons_, taus_;
      edm::InputTag pfCandidates_;
      edm::InputTag vertices_;
      edm::InputTag jets_;
      edm::InputTag rhoCentralCalo_;
      edm::InputTag EBRecHits_, EERecHits_, ESRecHits_;
      edm::InputTag HBHERecHits_;
      edm::InputTag cscSegments_, dtRecSegments_, rpcRecHits_;

      edm::InputTag dEdxPixel_;
      edm::InputTag dEdxStrip_;
      edm::InputTag isoTrk2dedxHitInfo_;     
      edm::InputTag isoTracks_;
      edm::InputTag genTracks_;

      edm::EDGetTokenT<edm::TriggerResults>                   triggersToken_;
      edm::EDGetTokenT<vector<pat::TriggerObjectStandAlone> > trigObjsToken_;
      edm::EDGetTokenT<vector<CandidateTrack> >               tracksToken_;
      edm::EDGetTokenT<reco::CandidateView>                   genParticlesToken_;
      edm::EDGetTokenT<vector<pat::MET> >                     metToken_;
      edm::EDGetTokenT<vector<pat::Electron> >                electronsToken_;
      edm::EDGetTokenT<vector<pat::Muon> >                    muonsToken_;
      edm::EDGetTokenT<vector<pat::Tau> >                     tausToken_;
      edm::EDGetTokenT<vector<pat::PackedCandidate> >         pfCandidatesToken_;
      edm::EDGetTokenT<vector<reco::Vertex> >                 verticesToken_;
      edm::EDGetTokenT<vector<pat::Jet> >                     jetsToken_;

      edm::EDGetTokenT<double> rhoCentralCaloToken_;

      edm::EDGetTokenT<EBRecHitCollection>         EBRecHitsToken_;
      edm::EDGetTokenT<EERecHitCollection>         EERecHitsToken_;
      edm::EDGetTokenT<ESRecHitCollection>         ESRecHitsToken_;
      edm::EDGetTokenT<HBHERecHitCollection>       HBHERecHitsToken_;
      edm::EDGetTokenT<CSCSegmentCollection>       CSCSegmentsToken_;
      edm::EDGetTokenT<DTRecSegment4DCollection>   DTRecSegmentsToken_;
      edm::EDGetTokenT<RPCRecHitCollection>        RPCRecHitsToken_;

      edm::EDGetTokenT<edm::ValueMap<reco::DeDxData> > dEdxPixelToken_;
      edm::EDGetTokenT<edm::ValueMap<reco::DeDxData> > dEdxStripToken_;
      edm::EDGetTokenT<reco::DeDxHitInfoAss> isoTrk2dedxHitInfoToken_;
      edm::EDGetTokenT<vector<pat::IsolatedTrack> >isoTrackToken_;
      edm::EDGetTokenT<vector<reco::Track> > genTracksToken_;
      
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
      
      const double minGenParticlePt_;
      const double minTrackPt_;
      const double maxRelTrackIso_;

      const string dataTakingPeriod_;
 
      // event information  
      int nPV_;
      unsigned long long eventNumber_;
      unsigned int lumiBlockNumber_;
      unsigned int runNumber_;

      // event-wide cut results
      bool firesGrandOrTrigger_;
      bool passMETFilters_;
      int numGoodPVs_;
      double metNoMu_;
      double numGoodJets_;
      double dijetDeltaPhiMax_;
      double leadingJetMetPhi_;

      bool is2017_;

      map<DetId, vector<double> > EcalAllDeadChannelsValMap_;
      map<DetId, vector<int> >    EcalAllDeadChannelsBitMap_;

      vector<string> signalTriggerNames, metFilterNames;
};

#endif
