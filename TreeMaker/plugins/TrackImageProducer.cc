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
  genParticlesToken_ = consumes<reco::CandidateView>        (genParticles_);
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
  genParticleInfos_.clear();

  tree_ = fs_->make<TTree>("tree", "tree");
  tree_->Branch("tracks", &trackInfos_);
  tree_->Branch("recHits", &recHitInfos_);
  tree_->Branch("genParticles", &genParticleInfos_);
  
  tree_->Branch("nPV", &nPV_);
  tree_->Branch("eventNumber", &eventNumber_);
  tree_->Branch("lumiBlockNumber", &lumiBlockNumber_);
  tree_->Branch("runNumber", &runNumber_);
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

  edm::Handle<reco::CandidateView> genParticles;
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
  nPV_ = vertices->size();

  eventNumber_ = event.id().event();
  lumiBlockNumber_ = event.id().luminosityBlock();
  runNumber_ = event.id().run();

  edm::Handle<vector<reco::PFJet> > jets;
  event.getByToken(jetsToken_, jets);

  getGeometries(setup);
  getChannelStatusMaps();

  trackInfos_.clear();

  for(const auto &track : *tracks) {
    if(minTrackPt_ > 0 && track.pt() < minTrackPt_) continue;

    TrackInfo info = getTrackInfo(track, *tracks, pv, *jets);

    if(maxRelTrackIso_ > 0 && info.trackIso / info.pt >= maxRelTrackIso_) continue;

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

  if(trackInfos_.size() == 0) return; // only fill tree with passing tracks

  recHitInfos_.clear();
  getRecHits(event);

  genParticleInfos_.clear();
  getGenParticles(*genParticles);

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

  setup.get<EcalChannelStatusRcd>().get(ecalStatus_);
}

const TrackInfo
TrackImageProducer::getTrackInfo(const reco::Track &track, const vector<reco::Track> &tracks, const reco::Vertex &pv, const vector<reco::PFJet> &jets) const
{

  TrackInfo info;

  info.px = track.px();
  info.py = track.py();
  info.pz = track.pz();
  info.eta = track.eta();
  info.pt = track.pt();
  info.phi = track.phi();

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

  bool inTOBCrack            = (fabs(track.dz()) < 0.5 && fabs(M_PI_2 - track.theta()) < 1.0e-3);
  bool inECALCrack           = (fabs(info.eta) >= 1.42 && fabs(info.eta) <= 1.65);
  bool inDTWheelGap          = (fabs(info.eta) >= 0.15 && fabs(info.eta) <= 0.35);
  bool inCSCTransitionRegion = (fabs(info.eta) >= 1.55 && fabs(info.eta) <= 1.85);
  info.inGap = (inTOBCrack || inECALCrack || inDTWheelGap || inCSCTransitionRegion);

  info.dRMinBadEcalChannel = minDRBadEcalChannel(track);

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
  
  info.passesProbeSelection = isProbeTrack(info);

  info.isTagProbeElectron = false;
  info.isTagProbeTauToElectron = false;


  return info;
}

const bool
TrackImageProducer::isProbeTrack(const TrackInfo info) const
{
  if(info.pt <= 30 ||
     fabs(info.eta) >= 2.1 ||
     // skip fiducial selections
     info.nValidPixelHits < 4 ||
     info.nValidHits < 4 ||
     info.missingInnerHits != 0 ||
     info.missingMiddleHits != 0 ||
     info.trackIso / info.pt >= 0.05 ||
     fabs(info.d0) >= 0.02 ||
     fabs(info.dz) >= 0.5 ||
     // skip lepton vetoes
     fabs(info.dRMinJet) <= 0.5) {
    return false;
  }

  return true;
}

void 
TrackImageProducer::getRecHits(const edm::Event &event)
{
  edm::Handle<EBRecHitCollection> EBRecHits;
  event.getByToken(EBRecHitsToken_, EBRecHits);
  for(const auto &hit : *EBRecHits) {
    math::XYZVector pos = getPosition(hit.detid());
    recHitInfos_.push_back(RecHitInfo(pos.eta(), pos.phi(), hit.energy(), -999., DetType::EB));
  }

  edm::Handle<EERecHitCollection> EERecHits;
  event.getByToken(EERecHitsToken_, EERecHits);
  for(const auto &hit : *EERecHits) {
    math::XYZVector pos = getPosition(hit.detid());
    recHitInfos_.push_back(RecHitInfo(pos.eta(), pos.phi(), hit.energy(), -999., DetType::EE));
  }

  edm::Handle<ESRecHitCollection> ESRecHits;
  event.getByToken(ESRecHitsToken_, ESRecHits);
  for(const auto &hit : *ESRecHits) {
    math::XYZVector pos = getPosition(hit.detid());
    recHitInfos_.push_back(RecHitInfo(pos.eta(), pos.phi(), hit.energy(), -999., DetType::ES));
  }

  edm::Handle<HBHERecHitCollection> HBHERecHits;
  event.getByToken(HBHERecHitsToken_, HBHERecHits);
  for(const auto &hit : *HBHERecHits) {
    math::XYZVector pos = getPosition(hit.detid());
    recHitInfos_.push_back(RecHitInfo(pos.eta(), pos.phi(), hit.energy(), -999., DetType::HCAL));
  }

  edm::Handle<CSCSegmentCollection> CSCSegments;
  event.getByToken(CSCSegmentsToken_, CSCSegments);
  for(const auto &seg : *CSCSegments) {
    math::XYZVector pos = getPosition(seg);
    recHitInfos_.push_back(RecHitInfo(pos.eta(), pos.phi(), -1, seg.time(), DetType::CSC));
  }

  edm::Handle<DTRecSegment4DCollection> DTRecSegments;
  event.getByToken(DTRecSegmentsToken_, DTRecSegments);
  for(const auto &seg : *DTRecSegments) {
    double time = -999.;
    if(seg.hasPhi()) {
      time = seg.phiSegment()->t0();
    }
    if(seg.hasZed()) {
      time = seg.zSegment()->t0();
    }
    math::XYZVector pos = getPosition(seg);
    recHitInfos_.push_back(RecHitInfo(pos.eta(), pos.phi(), -1, time, DetType::DT));
  }

  edm::Handle<RPCRecHitCollection> RPCRecHits;
  event.getByToken(RPCRecHitsToken_, RPCRecHits);
  for(const auto &hit : *RPCRecHits) {
    math::XYZVector pos = getPosition(hit);
    recHitInfos_.push_back(RecHitInfo(pos.eta(), pos.phi(), -1, -999., DetType::RPC));
  }

}

void
TrackImageProducer::getGenParticles(const reco::CandidateView &genParticles) {
  
  vector<const reco::Candidate*> cands;
  vector<const reco::Candidate*>::const_iterator found = cands.begin();
  for(reco::CandidateView::const_iterator p = genParticles.begin(); p != genParticles.end(); ++p) cands.push_back(&*p);

  for(reco::CandidateView::const_iterator p = genParticles.begin(); p != genParticles.end(); p++) {
    GenParticleInfo info;
    info.px = p->px();
    info.py = p->py();
    info.pz = p->pz();
    info.e  = p->energy();

    info.eta = p->eta();
    info.phi = p->phi();
    info.pt  = p->pt();

    info.pdgId = p->pdgId();
    info.status = p->status();

    info.mother1_index = -1;
    info.mother2_index = -1;

    info.daughter1_index = -1;
    info.daughter2_index = -1;

    info.nMothers = p->numberOfMothers();
    info.nDaughters = p->numberOfDaughters();

    found = find(cands.begin(), cands.end(), p->mother(0));
    if(found != cands.end()) info.mother1_index = found - cands.begin();

    found = find(cands.begin(), cands.end(), p->mother(p->numberOfMothers() - 1));
    if(found != cands.end()) info.mother2_index = found - cands.begin();

    found = find(cands.begin(), cands.end(), p->daughter(0));
    if(found != cands.end()) info.daughter1_index = found - cands.begin();

    found = find(cands.begin(), cands.end(), p->daughter(p->numberOfDaughters() - 1));
    if(found != cands.end()) info.daughter2_index = found - cands.begin();

    const reco::GenParticle* gp = dynamic_cast<const reco::GenParticle*>(&*p);

    info.isPromptFinalState = gp->isPromptFinalState();
    info.isDirectPromptTauDecayProductFinalState = gp->isDirectPromptTauDecayProductFinalState();
    info.isHardProcess = gp->isHardProcess();
    info.fromHardProcessFinalState = gp->fromHardProcessFinalState();
    info.fromHardProcessBeforeFSR = gp->fromHardProcessBeforeFSR();
    info.isFirstCopy = gp->statusFlags().isFirstCopy();
    info.isLastCopy = gp->isLastCopy();
    info.isLastCopyBeforeFSR = gp->isLastCopyBeforeFSR();

    genParticleInfos_.push_back(info);
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

const double
TrackImageProducer::minDRBadEcalChannel(const reco::Track &track) const
{
   double trackEta = track.eta(), trackPhi = track.phi();

   double min_dist = -1;
   DetId min_detId;

   map<DetId, vector<int> >::const_iterator bitItor;
   for(bitItor = EcalAllDeadChannelsBitMap_.begin(); bitItor != EcalAllDeadChannelsBitMap_.end(); bitItor++) {
      DetId maskedDetId = bitItor->first;
      map<DetId, std::vector<double> >::const_iterator valItor = EcalAllDeadChannelsValMap_.find(maskedDetId);
      if(valItor == EcalAllDeadChannelsValMap_.end()){ 
        cout << "Error cannot find maskedDetId in EcalAllDeadChannelsValMap_ ?!" << endl;
        continue;
      }

      double eta = (valItor->second)[0], phi = (valItor->second)[1];
      double dist = reco::deltaR(eta, phi, trackEta, trackPhi);

      if(min_dist > dist || min_dist < 0) {
        min_dist = dist;
        min_detId = maskedDetId;
      }
   }

   return min_dist;
}

void
TrackImageProducer::getChannelStatusMaps()
{
  EcalAllDeadChannelsValMap_.clear();
  EcalAllDeadChannelsBitMap_.clear();

  // Loop over EB ...
  for(int ieta = -85; ieta <= 85; ieta++) {
    for(int iphi = 0; iphi <= 360; iphi++) {
      if(!EBDetId::validDetId(ieta, iphi)) continue;

      const EBDetId detid = EBDetId(ieta, iphi, EBDetId::ETAPHIMODE);
      EcalChannelStatus::const_iterator chit = ecalStatus_->find(detid);
      // refer https://twiki.cern.ch/twiki/bin/viewauth/CMS/EcalChannelStatus
      int status = (chit != ecalStatus_->end()) ? chit->getStatusCode() & 0x1F : -1;

      const CaloSubdetectorGeometry * subGeom = caloGeometry_->getSubdetectorGeometry(detid);
      auto cellGeom = subGeom->getGeometry(detid);
      double eta = cellGeom->getPosition().eta();
      double phi = cellGeom->getPosition().phi();
      double theta = cellGeom->getPosition().theta();

      if(status >= 3) { // maskedEcalChannelStatusThreshold_
        vector<double> valVec;
        vector<int> bitVec;
        
        valVec.push_back(eta);
        valVec.push_back(phi);
        valVec.push_back(theta);
        
        bitVec.push_back(1);
        bitVec.push_back(ieta);
        bitVec.push_back(iphi);
        bitVec.push_back(status);
        
        EcalAllDeadChannelsValMap_.insert(make_pair(detid, valVec));
        EcalAllDeadChannelsBitMap_.insert(make_pair(detid, bitVec));
      }
    } // end loop iphi
  } // end loop ieta

  // Loop over EE detid
  for(int ix = 0; ix <= 100; ix++) {
    for(int iy = 0; iy <= 100; iy++) {
      for(int iz = -1; iz <= 1; iz++) {
        if(iz == 0) continue;
        if(!EEDetId::validDetId(ix, iy, iz)) continue;

        const EEDetId detid = EEDetId(ix, iy, iz, EEDetId::XYMODE);
        EcalChannelStatus::const_iterator chit = ecalStatus_->find(detid);
        int status = (chit != ecalStatus_->end()) ? chit->getStatusCode() & 0x1F : -1;

        const CaloSubdetectorGeometry * subGeom = caloGeometry_->getSubdetectorGeometry(detid);
        auto cellGeom = subGeom->getGeometry(detid);
        double eta = cellGeom->getPosition().eta();
        double phi = cellGeom->getPosition().phi();
        double theta = cellGeom->getPosition().theta();

        if(status >= 3) { // maskedEcalChannelStatusThreshold_
          vector<double> valVec;
          vector<int> bitVec;
          
          valVec.push_back(eta);
          valVec.push_back(phi);
          valVec.push_back(theta);
          
          bitVec.push_back(2);
          bitVec.push_back(ix);
          bitVec.push_back(iy);
          bitVec.push_back(iz);
          bitVec.push_back(status);

          EcalAllDeadChannelsValMap_.insert(make_pair(detid, valVec));
          EcalAllDeadChannelsBitMap_.insert(make_pair(detid, bitVec));
        }
      } // end loop iz
    } // end loop iy
  } // end loop ix
}

DEFINE_FWK_MODULE(TrackImageProducer);
