#include "DisappTrksML/TreeMaker/interface/TrackImageProducerMINIAOD.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"

#include "OSUT3Analysis/AnaTools/interface/CommonUtils.h"

#include "TLorentzVector.h"

TrackImageProducerMINIAOD::TrackImageProducerMINIAOD(const edm::ParameterSet &cfg) :
  triggers_     (cfg.getParameter<edm::InputTag> ("triggers")),
  trigObjs_     (cfg.getParameter<edm::InputTag> ("triggerObjects")),
  tracks_       (cfg.getParameter<edm::InputTag> ("tracks")),
  genParticles_ (cfg.getParameter<edm::InputTag> ("genParticles")),
  met_          (cfg.getParameter<edm::InputTag> ("met")),
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

  minGenParticlePt_   (cfg.getParameter<double> ("minGenParticlePt")),
  minTrackPt_         (cfg.getParameter<double> ("minTrackPt")),
  maxRelTrackIso_     (cfg.getParameter<double> ("maxRelTrackIso")),

  dataTakingPeriod_ (cfg.getParameter<string> ("dataTakingPeriod"))
{
  assert(dataTakingPeriod_ == "2017" || dataTakingPeriod_ == "2018");
  is2017_ = (dataTakingPeriod_ == "2017");

  triggersToken_     = consumes<edm::TriggerResults>           (triggers_);
  trigObjsToken_     = consumes<vector<pat::TriggerObjectStandAlone> > (trigObjs_);
  tracksToken_       = consumes<vector<CandidateTrack> >       (tracks_);
  genParticlesToken_ = consumes<vector<reco::GenParticle> >    (genParticles_);
  metToken_          = consumes<vector<pat::MET> >             (met_);
  electronsToken_    = consumes<vector<pat::Electron> >        (electrons_);
  muonsToken_        = consumes<vector<pat::Muon> >            (muons_);
  tausToken_         = consumes<vector<pat::Tau> >             (taus_);
  pfCandidatesToken_ = consumes<vector<pat::PackedCandidate> > (pfCandidates_);
  verticesToken_     = consumes<vector<reco::Vertex> >         (vertices_);
  jetsToken_         = consumes<vector<pat::Jet> >             (jets_);

  EBRecHitsToken_     = consumes<EBRecHitCollection>       (EBRecHits_);
  EERecHitsToken_     = consumes<EERecHitCollection>       (EERecHits_);
  ESRecHitsToken_     = consumes<ESRecHitCollection>       (ESRecHits_);
  HBHERecHitsToken_   = consumes<HBHERecHitCollection>     (HBHERecHits_);
  CSCSegmentsToken_   = consumes<CSCSegmentCollection>     (cscSegments_);
  DTRecSegmentsToken_ = consumes<DTRecSegment4DCollection> (dtRecSegments_);
  RPCRecHitsToken_    = consumes<RPCRecHitCollection>      (rpcRecHits_);

  trackInfos_.clear();
  recHitInfos_.clear();

  tree_ = fs_->make<TTree>("tree", "tree");
  tree_->Branch("tracks", &trackInfos_);
  tree_->Branch("recHits", &recHitInfos_);
  
  tree_->Branch("nPV", &nPV_);
  tree_->Branch("eventNumber", &eventNumber_);
  tree_->Branch("lumiBlockNumber", &lumiBlockNumber_);
  tree_->Branch("runNumber", &runNumber_);
}

TrackImageProducerMINIAOD::~TrackImageProducerMINIAOD()
{
}

void
TrackImageProducerMINIAOD::analyze(const edm::Event &event, const edm::EventSetup &setup)
{
  // get reco objects

  edm::Handle<edm::TriggerResults> triggers;
  event.getByToken (triggersToken_, triggers);

  edm::Handle<vector<pat::TriggerObjectStandAlone> > trigObjs;
  event.getByToken (trigObjsToken_, trigObjs);

  edm::Handle<vector<CandidateTrack> > tracks;
  event.getByToken(tracksToken_, tracks);

  edm::Handle<vector<reco::GenParticle> > genParticles;
  event.getByToken(genParticlesToken_, genParticles);

  edm::Handle<vector<pat::MET> > met;
  event.getByToken (metToken_, met);

  edm::Handle<vector<pat::Electron> > electrons;
  event.getByToken(electronsToken_, electrons);

  edm::Handle<vector<pat::Muon> > muons;
  event.getByToken(muonsToken_, muons);

  edm::Handle<vector<pat::Tau> > taus;
  event.getByToken(tausToken_, taus);

  edm::Handle<vector<pat::PackedCandidate> > pfCandidates;
  event.getByToken(pfCandidatesToken_, pfCandidates);

  edm::Handle<vector<reco::Vertex> > vertices;
  event.getByToken(verticesToken_, vertices);
  const reco::Vertex &pv = vertices->at(0);
  nPV_ = vertices->size();

  eventNumber_ = event.id().event();
  lumiBlockNumber_ = event.id().luminosityBlock();
  runNumber_ = event.id().run();

  edm::Handle<vector<pat::Jet> > jets;
  event.getByToken(jetsToken_, jets);

  //

  getGeometries(setup);
  getChannelStatusMaps();

  vector<pat::Electron> tagElectrons = getTagElectrons(event, *triggers, *trigObjs, pv, *electrons);
  vector<pat::Muon> tagMuons = getTagMuons(event, *triggers, *trigObjs, pv, *muons);

  trackInfos_.clear();

  for(const auto &track : *tracks) {
    if(minTrackPt_ > 0 && track.pt() <= minTrackPt_) continue;

    TrackInfo info = getTrackInfo(track, *tracks, pv, *jets, *electrons, *muons, *taus, genParticles);

    if(maxRelTrackIso_ > 0 && info.trackIso / info.pt >= maxRelTrackIso_) continue;

    if(info.passesProbeSelection) {
      for(const auto tag : tagElectrons) {
        double thisDR = deltaR(tag, track);
        if(info.deltaRToClosestTagElectron < 0 || thisDR < info.deltaRToClosestTagElectron) {
          info.deltaRToClosestTagElectron = thisDR;
        }
        if(isTagProbeElePair(track, tag)) info.isTagProbeElectron = true;
        if(isTagProbeTauToElePair(track, tag, met->at(0))) info.isTagProbeTauToElectron = true;
      }

      for(const auto tag : tagMuons) {
        double thisDR = deltaR(tag, track);
        if(info.deltaRToClosestTagMuon < 0 || thisDR < info.deltaRToClosestTagMuon) {
          info.deltaRToClosestTagMuon = thisDR;
        }
        if(isTagProbeMuonPair(track, tag)) info.isTagProbeMuon = true;
        if(isTagProbeTauToMuonPair(track, tag, met->at(0))) info.isTagProbeTauToMuon = true;
      }
    }

    trackInfos_.push_back(info);
  }

  if(trackInfos_.size() == 0) return; // only fill tree with passing tracks

  recHitInfos_.clear();
  getRecHits(event);

  tree_->Fill();

}

void TrackImageProducerMINIAOD::getGeometries(const edm::EventSetup &setup) {
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
TrackImageProducerMINIAOD::getTrackInfo(const CandidateTrack &track, 
                                        const vector<CandidateTrack> &tracks, 
                                        const reco::Vertex &pv, 
                                        const vector<pat::Jet> &jets,
                                        const vector<pat::Electron> &electrons,
                                        const vector<pat::Muon> &muons,
                                        const vector<pat::Tau> &taus,
                                        const edm::Handle<vector<reco::GenParticle> > genParticles) const
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

  bool inTOBCrack = (fabs(track.dz()) < 0.5 && fabs(M_PI_2 - track.theta()) < 1.0e-3);
  bool inECALCrack = (fabs(track.eta()) >= 1.42 && fabs(track.eta()) <= 1.65);
  bool inDTWheelGap = (fabs(track.eta()) >= 0.15 && fabs(track.eta()) <= 0.35);
  bool inCSCTransitionRegion = (fabs(track.eta()) >= 1.55 && fabs(track.eta()) <= 1.85);
  info.inGap = (inTOBCrack || inECALCrack || inDTWheelGap || inCSCTransitionRegion);

  info.dRMinBadEcalChannel = minDRBadEcalChannel(track);

  info.trackIso = 0.0;
  for(const auto &t : tracks) {
    if(fabs(track.dz(t.vertex())) > 3.0 * hypot(track.dzError(), t.dzError())) continue;
    double dR = deltaR(track, t);
    if(dR < 0.3 && dR > 1.0e-12) info.trackIso += t.pt();
  }

  info.nValidPixelHits        = track.hitPattern().numberOfValidPixelHits();
  info.nValidHits             = track.hitPattern().numberOfValidHits();
  info.missingInnerHits       = track.hitPattern().trackerLayersWithoutMeasurement(reco::HitPattern::MISSING_INNER_HITS);
  info.missingMiddleHits      = track.hitPattern().trackerLayersWithoutMeasurement(reco::HitPattern::TRACK_HITS);
  info.missingOuterHits       = track.hitPattern().trackerLayersWithoutMeasurement(reco::HitPattern::MISSING_OUTER_HITS);
  info.nLayersWithMeasurement = track.hitPattern().trackerLayersWithMeasurement();

  // d0 wrt pv (2d) = (vertex - pv) cross p / |p|
  info.d0 = ((track.vx() - pv.x()) * track.py() - (track.vy() - pv.y()) * track.px()) / track.pt(); 
  
  // dz wrt pv (2d) = (v_z - pv_z) - p_z * [(vertex - pv) dot p / |p|^2]
  info.dz = track.vz() - pv.z() -
    ((track.vx() - pv.x()) * track.px() + (track.vy() - pv.y()) * track.py()) * track.pz() / track.pt() / track.pt();

  info.deltaRToClosestElectron = -1;
  for(const auto &electron : electrons) {
    double thisDR = deltaR(electron, track);
    if(info.deltaRToClosestElectron < 0 || thisDR < info.deltaRToClosestElectron) info.deltaRToClosestElectron = thisDR;
  }

  info.deltaRToClosestMuon = -1;
  for(const auto &muon : muons) {
    double thisDR = deltaR(muon, track);
    if(info.deltaRToClosestMuon < 0 || thisDR < info.deltaRToClosestMuon) info.deltaRToClosestMuon = thisDR;
  }

  info.deltaRToClosestTauHad = -1;
  for(const auto &tau : taus) {
    if(tau.isTauIDAvailable("againstElectronLooseMVA5")) {
      if(tau.tauID("decayModeFinding") <= 0.5 ||
         tau.tauID("againstElectronLooseMVA5") <= 0.5 ||
         tau.tauID("againstMuonLoose3") <= 0.5) {
        continue;
      }
    }
    else if(tau.isTauIDAvailable("againstElectronLooseMVA6")) {
      if(tau.tauID("decayModeFinding") <= 0.5 ||
         tau.tauID("againstElectronLooseMVA6") <= 0.5 ||
         tau.tauID("againstMuonLoose3") <= 0.5) {
        continue;
      }
    }
    else {
      continue;
    }

    double thisDR = deltaR(tau, track);
    if(info.deltaRToClosestTauHad < 0 || thisDR < info.deltaRToClosestTauHad) info.deltaRToClosestTauHad = thisDR;
  }

  info.passesProbeSelection = isProbeTrack(info);

  info.deltaRToClosestTagElectron = -1;
  info.deltaRToClosestTagMuon = -1;

  info.isTagProbeElectron = false;
  info.isTagProbeTauToElectron = false;

  info.isTagProbeMuon = false;
  info.isTagProbeTauToMuon = false;

  return info;
}

void 
TrackImageProducerMINIAOD::getRecHits(const edm::Event &event)
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
TrackImageProducerMINIAOD::getPosition(const DetId& id) const
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
TrackImageProducerMINIAOD::getPosition(const CSCSegment& seg) const
{
  const LocalPoint localPos = seg.localPosition();
  const CSCDetId id = seg.cscDetId();

  const GlobalPoint idPosition = cscGeometry_->chamber(id)->toGlobal(localPos);
  math::XYZVector idPositionRoot(idPosition.x(), idPosition.y(), idPosition.z());
  return idPositionRoot;
}

const math::XYZVector
TrackImageProducerMINIAOD::getPosition(const DTRecSegment4D& seg) const
{
  const LocalPoint segmentLocal = seg.localPosition();
  const GlobalPoint idPosition = dtGeometry_->idToDet(seg.geographicalId())->surface().toGlobal(segmentLocal);
  math::XYZVector idPositionRoot(idPosition.x(), idPosition.y(), idPosition.z());
  return idPositionRoot;
}

const math::XYZVector
TrackImageProducerMINIAOD::getPosition(const RPCRecHit& seg) const
{
  const RPCDetId detId = static_cast<const RPCDetId>(seg.rpcId());
  const RPCRoll * roll = dynamic_cast<const RPCRoll*>(rpcGeometry_->roll(detId));
  const GlobalPoint idPosition = roll->toGlobal(seg.localPosition());
  math::XYZVector idPositionRoot(idPosition.x(), idPosition.y(), idPosition.z());
  return idPositionRoot;
}

vector<pat::Electron>
TrackImageProducerMINIAOD::getTagElectrons(const edm::Event &event,
                                           const edm::TriggerResults &triggers,
                                           const vector<pat::TriggerObjectStandAlone> &trigObjs,
                                           const reco::Vertex &vertex,
                                           const vector<pat::Electron> &electrons)
{
  vector<pat::Electron> tagElectrons;

  for(const auto &electron : electrons) {
    if(electron.pt() <= (is2017_ ? 35 : 32)) continue;

    if(!anatools::isMatchedToTriggerObject(event,
                                           triggers,
                                           electron,
                                           trigObjs,
                                           "hltEgammaCandidates::HLT", 
                                           (is2017_ ? "hltEle35noerWPTightGsfTrackIsoFilter" : "hltEle32WPTightGsfTrackIsoFilter"))) {
      continue; // cutElectronMatchToTrigObj
    }

    if(fabs(electron.eta()) >= 2.1) continue;
    if(!electron.electronID(is2017_ ? "cutBasedElectronID-Fall17-94X-V1-tight" : "cutBasedElectronID-Fall17-94X-V2-tight")) continue;
    
    if(fabs(electron.superCluster()->eta()) <= 1.479) {
      if(fabs(electron.gsfTrack()->dxy(vertex.position())) >= 0.05) continue;
      if(fabs(electron.gsfTrack()->dz(vertex.position())) >= 0.10) continue;
    }
    else {
      if(fabs(electron.gsfTrack()->dxy(vertex.position())) >= 0.10) continue;
      if(fabs(electron.gsfTrack()->dz(vertex.position())) >= 0.20) continue;
    }

    tagElectrons.push_back(electron);
  }

  return tagElectrons;
}

vector<pat::Muon>
TrackImageProducerMINIAOD::getTagMuons(const edm::Event &event,
                                       const edm::TriggerResults &triggers,
                                       const vector<pat::TriggerObjectStandAlone> &trigObjs,
                                       const reco::Vertex &vertex,
                                       const vector<pat::Muon> &muons)
{
  vector<pat::Muon> tagMuons;

  for(const auto &muon : muons) {
    if(muon.pt() <= (is2017_ ? 29 : 26)) continue;
    if(fabs(muon.eta()) >= 2.1) continue;
    if(!muon.isTightMuon(vertex)) continue;

    double iso = muon.pfIsolationR04().sumNeutralHadronEt +
                 muon.pfIsolationR04().sumPhotonEt +
                 -0.5 * muon.pfIsolationR04().sumPUPt;
    iso = muon.pfIsolationR04().sumChargedHadronPt + max(0.0, iso);
    if(iso / muon.pt() >= 0.15) continue;

    if(!anatools::isMatchedToTriggerObject(event,
                                           triggers,
                                           muon,
                                           trigObjs,
                                           (is2017_ ? "hltIterL3MuonCandidates::HLT" : "hltHighPtTkMuonCands::HLT"),
                                           (is2017_ ? "hltL3crIsoL1sMu22Or25L1f0L2f10QL3f27QL3trkIsoFiltered0p07" : "hltL3crIsoL1sMu22Or25L1f0L2f10QL3f27QL3trkIsoFiltered0p07"))) {
      continue; // cutMuonMatchToTrigObj
    }
    
    tagMuons.push_back(muon);
  }

  return tagMuons;
}

const bool
TrackImageProducerMINIAOD::isProbeTrack(const TrackInfo info) const
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

const bool
TrackImageProducerMINIAOD::isTagProbeElePair(const CandidateTrack &probe, const pat::Electron &tag) const 
{
  TLorentzVector t(tag.px(), tag.py(), tag.pz(), tag.energy());
  TLorentzVector p(probe.px(), 
                   probe.py(), 
                   probe.pz(), 
                   sqrt(probe.px() * probe.px() + 
                        probe.py() * probe.py() + 
                        probe.pz() * probe.pz() + 
                        0.000510998928 * 0.000510998928)); // energyOfElectron()

  if(fabs((t + p).M() - 91.1876) >= 10.0 || tag.charge() * probe.charge() >= 0) return false;

  return true;
}

const bool
TrackImageProducerMINIAOD::isTagProbeTauToElePair(const CandidateTrack &probe, 
                                                  const pat::Electron &tag, 
                                                  const pat::MET &met) const 
{
  double dPhi = deltaPhi(tag.phi(), probe.phi());
  if(sqrt(2.0 * tag.pt() * probe.pt() * (1 - cos(dPhi))) >= 40) return false; // cutElectronLowMT

  TLorentzVector t(tag.px(), tag.py(), tag.pz(), tag.energy());
  TLorentzVector p(probe.px(), 
                   probe.py(), 
                   probe.pz(), 
                   sqrt(probe.px() * probe.px() + 
                        probe.py() * probe.py() + 
                        probe.pz() * probe.pz() + 
                        0.000510998928 * 0.000510998928)); // energyOfElectron()

  double invMass = (t + p).M();
  if(invMass <= 91.1876 - 50 || invMass >= 91.1876 - 15 || tag.charge() * probe.charge() >= 0) return false;

  return true;
}

const bool
TrackImageProducerMINIAOD::isTagProbeMuonPair(const CandidateTrack &probe, const pat::Muon &tag) const 
{
  TLorentzVector t(tag.px(), tag.py(), tag.pz(), tag.energy());
  TLorentzVector p(probe.px(), 
                   probe.py(), 
                   probe.pz(), 
                   sqrt(probe.px() * probe.px() + 
                        probe.py() * probe.py() + 
                        probe.pz() * probe.pz() + 
                        0.1056583715 * 0.1056583715)); // energyOfMuon()

  if(fabs((t + p).M() - 91.1876) >= 10.0 || tag.charge() * probe.charge() >= 0) return false;

  return true;
}

const bool
TrackImageProducerMINIAOD::isTagProbeTauToMuonPair(const CandidateTrack &probe, 
                                                   const pat::Muon &tag, 
                                                   const pat::MET &met) const 
{
  double dPhi = deltaPhi(tag.phi(), probe.phi());
  if(sqrt(2.0 * tag.pt() * probe.pt() * (1 - cos(dPhi))) >= 40) return false; // cutMuonLowMT

  TLorentzVector t(tag.px(), tag.py(), tag.pz(), tag.energy());
  TLorentzVector p(probe.px(), 
                   probe.py(), 
                   probe.pz(), 
                   sqrt(probe.px() * probe.px() + 
                        probe.py() * probe.py() + 
                        probe.pz() * probe.pz() + 
                        0.1056583715 * 0.1056583715)); // energyOfMuon()

  double invMass = (t + p).M();
  if(invMass <= 91.1876 - 50 || invMass >= 91.1876 - 15 || tag.charge() * probe.charge() >= 0) return false;

  return true;
}

const double
TrackImageProducerMINIAOD::minDRBadEcalChannel(const CandidateTrack &track) const
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
TrackImageProducerMINIAOD::getChannelStatusMaps()
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

DEFINE_FWK_MODULE(TrackImageProducerMINIAOD);
