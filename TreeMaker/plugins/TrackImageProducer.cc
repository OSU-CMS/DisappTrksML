#include "DisappTrksML/TreeMaker/plugins/TrackImageProducer.h"

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

  clearVectors();

  tree_ = fs_->make<TTree>("tree", "tree");
  tree_->Branch("recHits_eta", &recHits_eta);
  tree_->Branch("recHits_phi", &recHits_phi);
  tree_->Branch("recHits_energy", &recHits_energy);
  tree_->Branch("recHits_detType", &recHits_detType);

  tree_->Branch("track_genMatchedID", &track_genMatchedID);
  tree_->Branch("track_genMatchedDR", &track_genMatchedDR);
  tree_->Branch("track_genMatchedPt", &track_genMatchedPt);
  tree_->Branch("track_deltaRToClosestElectron", &track_deltaRToClosestElectron);
  tree_->Branch("track_deltaRToClosestMuon", &track_deltaRToClosestMuon);
  tree_->Branch("track_deltaRToClosestTauHad", &track_deltaRToClosestTauHad);
  tree_->Branch("track_eta", &track_eta);
  tree_->Branch("track_phi", &track_phi);
  tree_->Branch("track_pt", &track_pt);
  tree_->Branch("track_trackIso", &track_trackIso);
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

  getGeometries(setup);

  clearVectors();

  for(const auto &track : *tracks) {

    if(minTrackPt_ > 0 && track.pt() < minTrackPt_) continue;
    double trackIso = getTrackIsolation(track, *tracks);
    if(maxRelTrackIso_ > 0 && trackIso / track.pt() >= maxRelTrackIso_) continue;

    track_trackIso.push_back(trackIso);

    int genMatchedID = 0;
    double genMatchedDR = -1, genMatchedPt = -1;
    if(genParticles.isValid()) {
      for(const auto &genParticle : *genParticles) {
        if(genParticle.pt() < minGenParticlePt_) continue;
        if(!genParticle.isPromptFinalState() && !genParticle.isDirectPromptTauDecayProductFinalState()) continue;

        double thisDR = deltaR(genParticle, track);
        if(genMatchedDR < 0 || thisDR < genMatchedDR) {
          genMatchedDR = thisDR;
          genMatchedID = genParticle.pdgId();
          genMatchedPt = genParticle.pt();
        }
      }
    }
    track_genMatchedID.push_back(genMatchedID);
    track_genMatchedDR.push_back(genMatchedDR);
    track_genMatchedPt.push_back(genMatchedPt);

    track_eta.push_back(track.eta());
    track_phi.push_back(track.phi());
    track_pt.push_back(track.pt());

    double deltaRToClosestElectron = -1;
    double deltaRToClosestMuon = -1;
    double deltaRToClosestTauHad = -1;

    for(const auto &electron : *electrons) {
      double thisDR = deltaR(electron, track);
      if(deltaRToClosestElectron < 0 || thisDR < deltaRToClosestElectron) deltaRToClosestElectron = thisDR;
    }

    for(const auto &muon : *muons) {
      double thisDR = deltaR(muon, track);
      if(deltaRToClosestMuon < 0 || thisDR < deltaRToClosestMuon) deltaRToClosestMuon = thisDR;
    }

    for(unsigned int i = 0; i < taus->size(); i++) {
      const edm::Ref<vector<reco::PFTau> > tauRef(taus, i);
      
      if((*tauDecayModeFinding)[tauRef] <= 0.5) continue;
      if((*tauElectronDiscriminator)[tauRef] <= 0.5) continue;
      if((*tauMuonDiscriminator)[tauRef] <= 0.5) continue;

      double thisDR = deltaR(*tauRef, track);
      if(deltaRToClosestTauHad < 0 || thisDR < deltaRToClosestTauHad) deltaRToClosestTauHad = thisDR;
    }

    track_deltaRToClosestElectron.push_back(deltaRToClosestElectron);
    track_deltaRToClosestMuon.push_back(deltaRToClosestMuon);
    track_deltaRToClosestTauHad.push_back(deltaRToClosestTauHad);

  }
  getImage(event);

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

const double
TrackImageProducer::getTrackIsolation(const reco::Track &track, const vector<reco::Track> &tracks) const
{
  double sumPt = 0.0;

  for(const auto &t : tracks) {
    if(fabs(track.dz(t.vertex())) > 3.0 * hypot(track.dzError(), t.dzError())) continue;
    double dR = deltaR(track, t);
    if(dR < 0.3 && dR > 1.0e-12) sumPt += t.pt();
  }

  return sumPt;
}

void 
TrackImageProducer::getImage(
  const edm::Event &event)
{

  edm::Handle<EBRecHitCollection> EBRecHits;
  event.getByToken(EBRecHitsToken_, EBRecHits);
  for(const auto &hit : *EBRecHits) {
    math::XYZVector pos = getPosition(hit.detid());
    recHits_eta.push_back(pos.eta());
    recHits_phi.push_back(pos.phi());
    recHits_energy.push_back(hit.energy());
    recHits_detType.push_back(DetType::EB);
  }

  edm::Handle<EERecHitCollection> EERecHits;
  event.getByToken(EERecHitsToken_, EERecHits);
  for(const auto &hit : *EERecHits) {
    math::XYZVector pos = getPosition(hit.detid());
    recHits_eta.push_back(pos.eta());
    recHits_phi.push_back(pos.phi());
    recHits_energy.push_back(hit.energy());
    recHits_detType.push_back(DetType::EE);
  }

  edm::Handle<ESRecHitCollection> ESRecHits;
  event.getByToken(ESRecHitsToken_, ESRecHits);
  for(const auto &hit : *ESRecHits) {
    math::XYZVector pos = getPosition(hit.detid());
    recHits_eta.push_back(pos.eta());
    recHits_phi.push_back(pos.phi());
    recHits_energy.push_back(hit.energy());
    recHits_detType.push_back(DetType::ES);
  }

  edm::Handle<HBHERecHitCollection> HBHERecHits;
  event.getByToken(HBHERecHitsToken_, HBHERecHits);
  for(const auto &hit : *HBHERecHits) {
    math::XYZVector pos = getPosition(hit.detid());
    recHits_eta.push_back(pos.eta());
    recHits_phi.push_back(pos.phi());
    recHits_energy.push_back(hit.energy());
    recHits_detType.push_back(DetType::HCAL);
  }

  edm::Handle<CSCSegmentCollection> CSCSegments;
  event.getByToken(CSCSegmentsToken_, CSCSegments);
  for(const auto &seg : *CSCSegments) {
    math::XYZVector pos = getPosition(seg);
    recHits_eta.push_back(pos.eta());
    recHits_phi.push_back(pos.phi());
    recHits_energy.push_back(-1);
    recHits_detType.push_back(DetType::CSC);
  }

  edm::Handle<DTRecSegment4DCollection> DTRecSegments;
  event.getByToken(DTRecSegmentsToken_, DTRecSegments);
  for(const auto &seg : *DTRecSegments) {
    math::XYZVector pos = getPosition(seg);
    recHits_eta.push_back(pos.eta());
    recHits_phi.push_back(pos.phi());
    recHits_energy.push_back(-1);
    recHits_detType.push_back(DetType::DT);
  }

  edm::Handle<RPCRecHitCollection> RPCRecHits;
  event.getByToken(RPCRecHitsToken_, RPCRecHits);
  for(const auto &hit : *RPCRecHits) {
    math::XYZVector pos = getPosition(hit);
    recHits_eta.push_back(pos.eta());
    recHits_phi.push_back(pos.phi());
    recHits_energy.push_back(-1);
    recHits_detType.push_back(DetType::CSC);
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
