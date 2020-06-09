#include "DisappTrksML/TreeMaker/interface/TrackImageProducer.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"

TrackImageProducer::TrackImageProducer(const edm::ParameterSet &cfg) :
  tracks_       (cfg.getParameter<edm::InputTag> ("tracks")),
  genParticles_ (cfg.getParameter<edm::InputTag> ("genParticles")),
  electrons_    (cfg.getParameter<edm::InputTag> ("electrons")),
  muons_        (cfg.getParameter<edm::InputTag> ("muons")),
  taus_         (cfg.getParameter<edm::InputTag> ("taus")),
  pfCandidates_ (cfg.getParameter<edm::InputTag> ("pfCandidates")),
  vertices_     (cfg.getParameter<edm::InputTag> ("vertices")),
  jets_         (cfg.getParameter<edm::InputTag> ("jets")),

  EBRecHits_     (cfg.getParameter<edm::InputTag> ("EBRecHits")),
  EERecHits_     (cfg.getParameter<edm::InputTag> ("EERecHits")),
  ESRecHits_     (cfg.getParameter<edm::InputTag> ("ESRecHits")),
  HBHERecHits_   (cfg.getParameter<edm::InputTag> ("HBHERecHits")),
  cscSegments_   (cfg.getParameter<edm::InputTag> ("CSCSegments")),
  dtRecSegments_ (cfg.getParameter<edm::InputTag> ("DTRecSegments")),
  rpcRecHits_    (cfg.getParameter<edm::InputTag> ("RPCRecHits")),

  tauDecayModeFinding_      (cfg.getParameter<edm::InputTag> ("tauDecayModeFinding")),
  tauElectronDiscriminator_ (cfg.getParameter<edm::InputTag> ("tauElectronDiscriminator")),
  tauMuonDiscriminator_     (cfg.getParameter<edm::InputTag> ("tauMuonDiscriminator")),

  minGenParticlePt_   (cfg.getParameter<double> ("minGenParticlePt")),
  minTrackPt_         (cfg.getParameter<double> ("minTrackPt")),
  maxRelTrackIso_     (cfg.getParameter<double> ("maxRelTrackIso"))
{
  tracksToken_       = consumes<vector<reco::Track> >       (tracks_);
  genParticlesToken_ = consumes<vector<reco::GenParticle> > (genParticles_);
  electronsToken_    = consumes<vector<reco::GsfElectron> > (electrons_);
  muonsToken_        = consumes<vector<reco::Muon> >        (muons_);
  tausToken_         = consumes<vector<reco::PFTau> >       (taus_);
  pfCandidatesToken_ = consumes<vector<reco::PFCandidate> > (pfCandidates_);
  verticesToken_     = consumes<vector<reco::Vertex> >      (vertices_);
  jetsToken_         = consumes<vector<reco::PFJet> >       (jets_);

  EBRecHitsToken_     = consumes<EBRecHitCollection>       (EBRecHits_);
  EERecHitsToken_     = consumes<EERecHitCollection>       (EERecHits_);
  ESRecHitsToken_     = consumes<ESRecHitCollection>       (ESRecHits_);
  HBHERecHitsToken_   = consumes<HBHERecHitCollection>     (HBHERecHits_);
  CSCSegmentsToken_   = consumes<CSCSegmentCollection>     (cscSegments_);
  DTRecSegmentsToken_ = consumes<DTRecSegment4DCollection> (dtRecSegments_);
  RPCRecHitsToken_    = consumes<RPCRecHitCollection>      (rpcRecHits_);

  tauDecayModeFindingToken_      = consumes<reco::PFTauDiscriminator> (tauDecayModeFinding_);
  tauElectronDiscriminatorToken_ = consumes<reco::PFTauDiscriminator> (tauElectronDiscriminator_);
  tauMuonDiscriminatorToken_     = consumes<reco::PFTauDiscriminator> (tauMuonDiscriminator_);

  trackInfos_.clear();
  recHitInfos_.clear();

  tree_ = fs_->make<TTree>("tree", "tree");
  tree_->Branch("tracks", &trackInfos_);
  tree_->Branch("recHits", &recHitInfos_);
}

TrackImageProducer::~TrackImageProducer()
{
}

void
TrackImageProducer::analyze(const edm::Event &event, const edm::EventSetup &setup)
{
  // get reco objects

  edm::Handle<vector<reco::Track> > tracks;
  event.getByToken(tracksToken_, tracks);

  edm::Handle<vector<reco::GenParticle> > genParticles;
  event.getByToken(genParticlesToken_, genParticles);

  edm::Handle<vector<reco::GsfElectron> > electrons;
  event.getByToken(electronsToken_, electrons);

  edm::Handle<vector<reco::Muon> > muons;
  event.getByToken(muonsToken_, muons);

  edm::Handle<vector<reco::PFTau> > taus;
  event.getByToken(tausToken_, taus);

  edm::Handle<vector<reco::PFCandidate> > pfCandidates;
  event.getByToken(pfCandidatesToken_, pfCandidates);

  edm::Handle<reco::PFTauDiscriminator> tauDecayModeFinding;
  event.getByToken(tauDecayModeFindingToken_, tauDecayModeFinding);

  edm::Handle<reco::PFTauDiscriminator> tauElectronDiscriminator;
  event.getByToken(tauElectronDiscriminatorToken_, tauElectronDiscriminator);

  edm::Handle<reco::PFTauDiscriminator> tauMuonDiscriminator;
  event.getByToken(tauMuonDiscriminatorToken_, tauMuonDiscriminator);

  edm::Handle<vector<reco::Vertex> > vertices;
  event.getByToken(verticesToken_, vertices);
  const reco::Vertex &pv = vertices->at(0);

  edm::Handle<vector<reco::PFJet> > jets;
  event.getByToken(jetsToken_, jets);

  getGeometries(setup);

  trackInfos_.clear();

  for(const auto &track : *tracks) {

    TrackInfo info = getTrackInfo(track, *tracks, pv, *jets);

    if(minTrackPt_ > 0 && track.pt() < minTrackPt_) continue;
    if(maxRelTrackIso_ > 0 && info.trackIso / track.pt() >= maxRelTrackIso_) continue;

    info.genMatchedID = 0;
    info.genMatchedDR = -1;
    info.genMatchedPt = -1;
    if(genParticles.isValid()) {
      for(const auto &genParticle : *genParticles) {
        if(genParticle.pt() < minGenParticlePt_) continue;
        if(!genParticle.isPromptFinalState() && !genParticle.isDirectPromptTauDecayProductFinalState()) continue;

        double thisDR = deltaR(genParticle, track);
        if(info.genMatchedDR < 0 || thisDR < info.genMatchedDR) {
          info.genMatchedDR = thisDR;
          info.genMatchedID = genParticle.pdgId();
          info.genMatchedPt = genParticle.pt();
        }
      }
    }

    info.deltaRToClosestElectron = -1;
    info.deltaRToClosestMuon = -1;
    info.deltaRToClosestTauHad = -1;

    for(const auto &electron : *electrons) {
      double thisDR = deltaR(electron, track);
      if(info.deltaRToClosestElectron < 0 || thisDR < info.deltaRToClosestElectron) info.deltaRToClosestElectron = thisDR;
    }

    for(const auto &muon : *muons) {
      double thisDR = deltaR(muon, track);
      if(info.deltaRToClosestMuon < 0 || thisDR < info.deltaRToClosestMuon) info.deltaRToClosestMuon = thisDR;
    }

    for(unsigned int i = 0; i < taus->size(); i++) {
      const edm::Ref<vector<reco::PFTau> > tauRef(taus, i);
      
      if((*tauDecayModeFinding)[tauRef] <= 0.5) continue;
      if((*tauElectronDiscriminator)[tauRef] <= 0.5) continue;
      if((*tauMuonDiscriminator)[tauRef] <= 0.5) continue;

      double thisDR = deltaR(*tauRef, track);
      if(info.deltaRToClosestTauHad < 0 || thisDR < info.deltaRToClosestTauHad) info.deltaRToClosestTauHad = thisDR;
    }

    trackInfos_.push_back(info);
  }

  recHitInfos_.clear();
  getRecHits(event);

  tree_->Fill();

}

void TrackImageProducer::getGeometries(const edm::EventSetup &setup) {
  setup.get<CaloGeometryRecord>().get(caloGeometry_);
  if(!caloGeometry_.isValid())
    throw cms::Exception("FatalError") << "Unable to find CaloGeometryRecord in event!\n";

  setup.get<MuonGeometryRecord>().get(cscGeometry_);
  if(!cscGeometry_.isValid())
    throw cms::Exception("FatalError") << "Unable to find MuonGeometryRecord (CSC) in event!\n";

  setup.get<MuonGeometryRecord>().get(dtGeometry_);
  if(!dtGeometry_.isValid())
    throw cms::Exception("FatalError") << "Unable to find MuonGeometryRecord (DT) in event!\n";

  setup.get<MuonGeometryRecord>().get(rpcGeometry_);
  if(!rpcGeometry_.isValid())
    throw cms::Exception("FatalError") << "Unable to find MuonGeometryRecord (RPC) in event!\n";
}

const TrackInfo
TrackImageProducer::getTrackInfo(const reco::Track &track, const vector<reco::Track> &tracks, const reco::Vertex &pv, const vector<reco::PFJet> &jets) const
{

  TrackInfo info;

  info.px = track.px();
  info.py = track.py();
  info.pz = track.pz();

  info.dRMinJet = -1;
  for(const auto &jet : jets) {
    if(jet.pt() > 30 &&
       fabs(jet.eta()) < 4.5 &&
       (((jet.neutralHadronEnergyFraction()<0.90 && jet.neutralEmEnergyFraction()<0.90 && (jet.chargedMultiplicity() + jet.neutralMultiplicity())>1 && jet.muonEnergyFraction()<0.8) && ((fabs(jet.eta())<=2.4 && jet.chargedHadronEnergyFraction()>0 && jet.chargedMultiplicity()>0 && jet.chargedEmEnergyFraction()<0.90) || fabs(jet.eta())>2.4) && fabs(jet.eta())<=3.0)
          || (jet.neutralEmEnergyFraction()<0.90 && jet.neutralMultiplicity()>10 && fabs(jet.eta())>3.0))) {
      double dR = deltaR(track, jet);
      if(info.dRMinJet < 0 || dR < info.dRMinJet) info.dRMinJet = dR;
    }
  }

  bool inTOBCrack = (fabs(track.dz()) < 0.5 && fabs(M_PI_2 - track.theta()) < 1.0e-3);
  bool inECALCrack = (fabs(track.eta()) >= 1.42 && fabs(track.eta()) <= 1.65);
  bool inDTWheelGap = (fabs(track.eta()) >= 0.15 && fabs(track.eta()) <= 0.35);
  bool inCSCTransitionRegion = (fabs(track.eta()) >= 1.55 && fabs(track.eta()) <= 1.85);
  info.inGap = (inTOBCrack || inECALCrack || inDTWheelGap || inCSCTransitionRegion);

  info.trackIso = 0.0;
  for(const auto &t : tracks) {
    if(fabs(track.dz(t.vertex())) > 3.0 * hypot(track.dzError(), t.dzError())) continue;
    double dR = deltaR(track, t);
    if(dR < 0.3 && dR > 1.0e-12) info.trackIso += t.pt();
  }

  info.nValidPixelHits = track.hitPattern().numberOfValidPixelHits();
  info.nValidHits = track.hitPattern().numberOfValidHits();
  info.missingInnerHits = track.hitPattern().trackerLayersWithoutMeasurement(reco::HitPattern::MISSING_INNER_HITS);
  info.missingMiddleHits = track.hitPattern().trackerLayersWithoutMeasurement(reco::HitPattern::TRACK_HITS);
  info.missingOuterHits = track.hitPattern().trackerLayersWithoutMeasurement(reco::HitPattern::MISSING_OUTER_HITS);
  info.nLayersWithMeasurement = track.hitPattern().trackerLayersWithMeasurement();

  // d0 wrt pv (2d) = (vertex - pv) cross p / |p|
  info.d0 = ((track.vx() - pv.x()) * track.py() - (track.vy() - pv.y()) * track.px()) / track.pt(); 
  
  // dz wrt pv (2d) = (v_z - pv_z) - p_z * [(vertex - pv) dot p / |p|^2]
  info.dz = track.vz() - pv.z() -
    ((track.vx() - pv.x()) * track.px() + (track.vy() - pv.y()) * track.py()) * track.pz() / track.pt() / track.pt();

  return info;
}

void 
TrackImageProducer::getRecHits(const edm::Event &event)
{
  edm::Handle<EBRecHitCollection> EBRecHits;
  event.getByToken(EBRecHitsToken_, EBRecHits);
  for(const auto &hit : *EBRecHits) {
    math::XYZVector pos = getPosition(hit.detid());
    recHitInfos_.push_back(RecHitInfo(pos.eta(), pos.phi(), hit.energy(), DetType::EB));
  }

  edm::Handle<EERecHitCollection> EERecHits;
  event.getByToken(EERecHitsToken_, EERecHits);
  for(const auto &hit : *EERecHits) {
    math::XYZVector pos = getPosition(hit.detid());
    recHitInfos_.push_back(RecHitInfo(pos.eta(), pos.phi(), hit.energy(), DetType::EE));
  }

  edm::Handle<ESRecHitCollection> ESRecHits;
  event.getByToken(ESRecHitsToken_, ESRecHits);
  for(const auto &hit : *ESRecHits) {
    math::XYZVector pos = getPosition(hit.detid());
    recHitInfos_.push_back(RecHitInfo(pos.eta(), pos.phi(), hit.energy(), DetType::ES));
  }

  edm::Handle<HBHERecHitCollection> HBHERecHits;
  event.getByToken(HBHERecHitsToken_, HBHERecHits);
  for(const auto &hit : *HBHERecHits) {
    math::XYZVector pos = getPosition(hit.detid());
    recHitInfos_.push_back(RecHitInfo(pos.eta(), pos.phi(), hit.energy(), DetType::HCAL));
  }

  edm::Handle<CSCSegmentCollection> CSCSegments;
  event.getByToken(CSCSegmentsToken_, CSCSegments);
  for(const auto &seg : *CSCSegments) {
    math::XYZVector pos = getPosition(seg);
    recHitInfos_.push_back(RecHitInfo(pos.eta(), pos.phi(), -1, DetType::CSC));
  }

  edm::Handle<DTRecSegment4DCollection> DTRecSegments;
  event.getByToken(DTRecSegmentsToken_, DTRecSegments);
  for(const auto &seg : *DTRecSegments) {
    math::XYZVector pos = getPosition(seg);
    recHitInfos_.push_back(RecHitInfo(pos.eta(), pos.phi(), -1, DetType::DT));
  }

  edm::Handle<RPCRecHitCollection> RPCRecHits;
  event.getByToken(RPCRecHitsToken_, RPCRecHits);
  for(const auto &hit : *RPCRecHits) {
    math::XYZVector pos = getPosition(hit);
    recHitInfos_.push_back(RecHitInfo(pos.eta(), pos.phi(), -1, DetType::RPC));
  }

}

const math::XYZVector 
TrackImageProducer::getPosition(const DetId& id) const
{
   if(!caloGeometry_->getSubdetectorGeometry(id) || !caloGeometry_->getSubdetectorGeometry(id)->getGeometry(id) ) {
      throw cms::Exception("FatalError") << "Failed to access geometry for DetId: " << id.rawId();
      return math::XYZVector(0,0,0);
   }
   const GlobalPoint idPosition = caloGeometry_->getSubdetectorGeometry(id)->getGeometry(id)->getPosition();
   math::XYZVector idPositionRoot(idPosition.x(), idPosition.y(), idPosition.z());
   return idPositionRoot;
}

const math::XYZVector
TrackImageProducer::getPosition(const CSCSegment& seg) const
{
  const LocalPoint localPos = seg.localPosition();
  const CSCDetId id = seg.cscDetId();

  const GlobalPoint idPosition = cscGeometry_->chamber(id)->toGlobal(localPos);
  math::XYZVector idPositionRoot(idPosition.x(), idPosition.y(), idPosition.z());
  return idPositionRoot;
}

const math::XYZVector
TrackImageProducer::getPosition(const DTRecSegment4D& seg) const
{
  const LocalPoint segmentLocal = seg.localPosition();
  const GlobalPoint idPosition = dtGeometry_->idToDet(seg.geographicalId())->surface().toGlobal(segmentLocal);
  math::XYZVector idPositionRoot(idPosition.x(), idPosition.y(), idPosition.z());
  return idPositionRoot;
}

const math::XYZVector
TrackImageProducer::getPosition(const RPCRecHit& seg) const
{
  const RPCDetId detId = static_cast<const RPCDetId>(seg.rpcId());
  const RPCRoll * roll = dynamic_cast<const RPCRoll*>(rpcGeometry_->roll(detId));
  const GlobalPoint idPosition = roll->toGlobal(seg.localPosition());
  math::XYZVector idPositionRoot(idPosition.x(), idPosition.y(), idPosition.z());
  return idPositionRoot;
}

DEFINE_FWK_MODULE(TrackImageProducer);
