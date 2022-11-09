#include "DisappTrksML/TreeMaker/interface/TrackImageProducerMINIAOD.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"

#include "OSUT3Analysis/AnaTools/interface/CommonUtils.h"
#include "RecoTracker/DeDx/interface/DeDxTools.h"

#include "TLorentzVector.h"

TrackImageProducerMINIAOD::TrackImageProducerMINIAOD(const edm::ParameterSet &cfg) :
  triggers_     (cfg.getParameter<edm::InputTag> ("triggers")),
  trigObjs_     (cfg.getParameter<edm::InputTag> ("triggerObjects")),
  //tracks_       (cfg.getParameter<edm::InputTag> ("tracks")),
  genParticles_ (cfg.getParameter<edm::InputTag> ("genParticles")),
  met_          (cfg.getParameter<edm::InputTag> ("met")),
  electrons_    (cfg.getParameter<edm::InputTag> ("electrons")),
  muons_        (cfg.getParameter<edm::InputTag> ("muons")),
  taus_         (cfg.getParameter<edm::InputTag> ("taus")),
  pfCandidates_ (cfg.getParameter<edm::InputTag> ("pfCandidates")),
  vertices_     (cfg.getParameter<edm::InputTag> ("vertices")),
  jets_         (cfg.getParameter<edm::InputTag> ("jets")),

  rhoCentralCalo_ (cfg.getParameter<edm::InputTag> ("rhoCentralCalo")),

  EBRecHits_     (cfg.getParameter<edm::InputTag> ("EBRecHits")),
  EERecHits_     (cfg.getParameter<edm::InputTag> ("EERecHits")),
  ESRecHits_     (cfg.getParameter<edm::InputTag> ("ESRecHits")),
  HBHERecHits_   (cfg.getParameter<edm::InputTag> ("HBHERecHits")),
  //cscSegments_   (cfg.getParameter<edm::InputTag> ("CSCSegments")),
  //dtRecSegments_ (cfg.getParameter<edm::InputTag> ("DTRecSegments")),
  //rpcRecHits_    (cfg.getParameter<edm::InputTag> ("RPCRecHits")),

  dEdxPixel_ (cfg.getParameter<edm::InputTag> ("dEdxPixel")),
  dEdxStrip_ (cfg.getParameter<edm::InputTag> ("dEdxStrip")),
  isoTrk2dedxHitInfo_ (cfg.getParameter<edm::InputTag> ("isoTrk2dedxHitInfo")),
  isoTracks_ (cfg.getParameter<edm::InputTag> ("isolatedTracks")),  
  //genTracks_(cfg.getParameter<edm::InputTag> ("genTracks")),

  pileupInfo_   (cfg.getParameter<edm::InputTag>("pileupInfo")),

  minGenParticlePt_   (cfg.getParameter<double> ("minGenParticlePt")),
  minTrackPt_         (cfg.getParameter<double> ("minTrackPt")),
  maxRelTrackIso_     (cfg.getParameter<double> ("maxRelTrackIso")),
  maxTrackEta_        (cfg.getParameter<double> ("maxTrackEta")),
  
  dataTakingPeriod_ (cfg.getParameter<string> ("dataTakingPeriod"))
{
  assert(dataTakingPeriod_ == "2017" || dataTakingPeriod_ == "2018" || dataTakingPeriod_ == "2022");
  is2017_ = (dataTakingPeriod_ == "2017");

  triggersToken_     = consumes<edm::TriggerResults>           (triggers_);
  trigObjsToken_     = consumes<vector<pat::TriggerObjectStandAlone> > (trigObjs_);
  //tracksToken_       = consumes<vector<CandidateTrack> >       (tracks_);
  genParticlesToken_ = consumes<reco::CandidateView>           (genParticles_);
  metToken_          = consumes<vector<pat::MET> >             (met_);
  electronsToken_    = consumes<vector<pat::Electron> >        (electrons_);
  muonsToken_        = consumes<vector<pat::Muon> >            (muons_);
  tausToken_         = consumes<vector<pat::Tau> >             (taus_);
  pfCandidatesToken_ = consumes<vector<pat::PackedCandidate> > (pfCandidates_);
  verticesToken_     = consumes<vector<reco::Vertex> >         (vertices_);
  jetsToken_         = consumes<vector<pat::Jet> >             (jets_);

  rhoCentralCaloToken_ = consumes<double> (rhoCentralCalo_);

  EBRecHitsToken_     = consumes<EBRecHitCollection>       (EBRecHits_);
  EERecHitsToken_     = consumes<EERecHitCollection>       (EERecHits_);
  ESRecHitsToken_     = consumes<ESRecHitCollection>       (ESRecHits_);
  HBHERecHitsToken_   = consumes<HBHERecHitCollection>     (HBHERecHits_);
  //CSCSegmentsToken_   = consumes<CSCSegmentCollection>     (cscSegments_);
  //DTRecSegmentsToken_ = consumes<DTRecSegment4DCollection> (dtRecSegments_);
  //RPCRecHitsToken_    = consumes<RPCRecHitCollection>      (rpcRecHits_);
  
  dEdxPixelToken_ = consumes<edm::ValueMap<reco::DeDxData> > (dEdxPixel_);
  dEdxStripToken_ = consumes<edm::ValueMap<reco::DeDxData> > (dEdxStrip_);
  isoTrk2dedxHitInfoToken_ = consumes<reco::DeDxHitInfoAss> (isoTrk2dedxHitInfo_);
  isoTrackToken_ = consumes<vector<pat::IsolatedTrack> > (isoTracks_);
  //genTracksToken_ = consumes<vector<reco::Track> > (genTracks_);
  pileupInfoToken_ = consumes<edm::View<PileupSummaryInfo> > (pileupInfo_);

  caloGeometryToken_    = esConsumes<CaloGeometry, CaloGeometryRecord>();
  //cscGeometryToken_        = esConsumes<CSCGeometry, MuonGeometryRecord>();
  //dtGeometryToken_        = esConsumes<DTGeometry, MuonGeometryRecord>();
  //rpcGeometryToken_        = esConsumes<RPCGeometry, MuonGeometryRecord>();
  ecalStatusToken_         = esConsumes<EcalChannelStatus, EcalChannelStatusRcd>();
  trackerTopologyToken_    = esConsumes<TrackerTopology, TrackerTopologyRcd>();

  signalTriggerNames = cfg.getParameter<vector<string> >("signalTriggerNames");
  metFilterNames = cfg.getParameter<vector<string> >("metFilterNames");

  trackInfos_.clear();
  recHitInfos_.clear();
  genParticleInfos_.clear();
  pileupZPosition_.clear();
  vertexInfos_.clear();

  tree_ = fs_->make<TTree>("tree", "tree");
  tree_->Branch("tracks", &trackInfos_);
  tree_->Branch("recHits", &recHitInfos_);
  tree_->Branch("genParticles", &genParticleInfos_);
  tree_->Branch("vertexInfos", &vertexInfos_);
  
  tree_->Branch("nPV", &nPV_);
  tree_->Branch("eventNumber", &eventNumber_);
  tree_->Branch("lumiBlockNumber", &lumiBlockNumber_);
  tree_->Branch("runNumber", &runNumber_);
  tree_->Branch("pileupZPosition", &pileupZPosition_);
  tree_->Branch("numTruePV", &numTruePV_);

  tree_->Branch("firesGrandOrTrigger", &firesGrandOrTrigger_);
  tree_->Branch("passMETFilters", &passMETFilters_);
  tree_->Branch("numGoodPVs", &numGoodPVs_);
  tree_->Branch("metNoMu", &metNoMu_);
  tree_->Branch("numGoodJets", &numGoodJets_);
  tree_->Branch("dijetDeltaPhiMax", &dijetDeltaPhiMax_);
  tree_->Branch("leadingJetMetPhi", &leadingJetMetPhi_);
}

TrackImageProducerMINIAOD::~TrackImageProducerMINIAOD()
{
}

void
TrackImageProducerMINIAOD::analyze(const edm::Event &event, const edm::EventSetup &setup)
{
  // get collections, setup objects

  edm::Handle<edm::TriggerResults> triggers;
  event.getByToken (triggersToken_, triggers);

  edm::Handle<vector<pat::TriggerObjectStandAlone> > trigObjs;
  event.getByToken (trigObjsToken_, trigObjs);

  //edm::Handle<vector<CandidateTrack> > tracks;
  //event.getByToken(tracksToken_, tracks);

  edm::Handle<reco::CandidateView> genParticles;
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

  edm::Handle<vector<pat::Jet> > jets;
  event.getByToken(jetsToken_, jets);

  edm::Handle<double> rhoCentralCalo;
  event.getByToken(rhoCentralCaloToken_, rhoCentralCalo);

  edm::Handle<edm::ValueMap<reco::DeDxData> > dEdxStrip;
  event.getByToken (dEdxStripToken_, dEdxStrip);

  edm::Handle<edm::ValueMap<reco::DeDxData> > dEdxPixel;
  event.getByToken (dEdxPixelToken_, dEdxPixel);

  edm::Handle<reco::DeDxHitInfoAss> isoTrk2dedxHitInfo;
  event.getByToken(isoTrk2dedxHitInfoToken_, isoTrk2dedxHitInfo);

  edm::Handle<vector<pat::IsolatedTrack> > isoTracks;
  event.getByToken (isoTrackToken_, isoTracks);

  //edm::Handle<vector<reco::Track> > genTracks;
  //event.getByToken (genTracksToken_, genTracks);

  edm::Handle<edm::View<PileupSummaryInfo> > pileupInfos;
  event.getByToken(pileupInfoToken_, pileupInfos);
 
  const edm::TriggerNames &allTriggerNames = event.triggerNames(*triggers);

  getGeometries(setup);
  getChannelStatusMaps();

  //

  TVector2 metNoMuVec(met->at(0).px(), met->at(0).py());
  for(const auto &pfCandidate : *pfCandidates) {
    if(abs(pfCandidate.pdgId()) != 13) continue;
    TVector2 muon(pfCandidate.px(), pfCandidate.py());
    metNoMuVec += muon;
  }

  metNoMu_ = metNoMuVec.Mod();

  firesGrandOrTrigger_ = false;
  passMETFilters_ = true;

  for(unsigned i = 0; i < allTriggerNames.size(); i++) {
    string thisName = allTriggerNames.triggerName(i);

    for(auto name : signalTriggerNames) {
      if(thisName.find(name) == 0 && triggers->accept(i)) {
        firesGrandOrTrigger_ = true;
        break;
      }
    }

    for(auto name : metFilterNames) {
      if(thisName.find(name) == 0 && !triggers->accept(i)) {
        passMETFilters_ = false;
        break;
      }
    }

    if(firesGrandOrTrigger_ && !passMETFilters_) break;
  }

  const reco::Vertex &pv = vertices->at(0);
  nPV_ = vertices->size();
  numGoodPVs_ = countGoodPrimaryVertices(*vertices);

  vertexInfos_.clear();
  for(auto vertex : *vertices){
      VertexInfo info;
      
      TLorentzVector vertex_pos(vertex.x(), vertex.y(), vertex.z(), vertex.t());
      TLorentzVector vertex_err(vertex.xError(), vertex.yError(), vertex.zError(), vertex.tError());
      info.vertex = vertex_pos;
      info.vertex_error = vertex_err;
      info.chi2 = vertex.chi2();
      info.ndof = vertex.ndof();
      info.isValid = vertex.isValid();     

      vertexInfos_.push_back(info);
  }

  eventNumber_ = event.id().event();
  lumiBlockNumber_ = event.id().luminosityBlock();
  runNumber_ = event.id().run();

  numGoodJets_      = countGoodJets(*jets);
  dijetDeltaPhiMax_ = getMaxDijetDeltaPhi(*jets);
  leadingJetMetPhi_ = getLeadingJetMetPhi(*jets, met->at(0));

  vector<pat::Electron> tagElectrons = getTagElectrons(event, *triggers, *trigObjs, pv, *electrons);
  vector<pat::Muon> tagMuons = getTagMuons(event, *triggers, *trigObjs, pv, *muons);

  getTracks(pv, *jets, *electrons, *muons, *taus, tagElectrons, tagMuons, met->at(0), isoTracks, isoTrk2dedxHitInfo, dEdxStrip, dEdxPixel);

  if(trackInfos_.size() == 0) return; // only fill tree with passing tracks

  getRecHits(event);
  if(genParticles.isValid()) getGenParticles(*genParticles);

  // account for pileup in track ecalo
  double caloCorr = (*rhoCentralCalo) * 2. * M_PI_2 * 0.5 * 0.5;
  for(auto &info : trackInfos_) {
    info.ecalo -= caloCorr;
    if(info.ecalo < 0) info.ecalo = 0;
  }

  // Get pileup vertex z positions
  if(pileupInfos.isValid()) {
    edm::View<PileupSummaryInfo>::const_iterator iterPU;
    for(edm::View<PileupSummaryInfo>::const_iterator iterPU = pileupInfos->begin(); iterPU != pileupInfos->end(); iterPU++) {
      // Out of time pileup is also saved -> need to require 0th bunch crossing (in time bunch crossing)
      if(iterPU->getBunchCrossing() == 0){ 
        pileupZPosition_ = iterPU->getPU_zpositions();
        numTruePV_ = iterPU->getTrueNumInteractions();
      }
    }
  }

  tree_->Fill();

}

/*void
TrackImageProducerMINIAOD::findMatchedIsolatedTrack (const edm::Handle<vector<pat::IsolatedTrack> > &isolatedTracks, edm::Ref<vector<pat::IsolatedTrack> > &matchedIsolatedTrack, double &dRToMatchedIsolatedTrack, const CandidateTrack &track) const
{
  dRToMatchedIsolatedTrack = INVALID_VALUE;
  double maxDeltaR_isolatedTrackMatching_ = 0.01;
  for(vector<pat::IsolatedTrack>::const_iterator isoTrack = isolatedTracks->begin(); isoTrack != isolatedTracks->end(); isoTrack++) {
    double dR = deltaR(*isoTrack, track);
    if(maxDeltaR_isolatedTrackMatching_ >= 0.0 && dR > maxDeltaR_isolatedTrackMatching_) continue;
    if(dR < dRToMatchedIsolatedTrack || dRToMatchedIsolatedTrack < 0.0) {
      dRToMatchedIsolatedTrack = dR;
      matchedIsolatedTrack = edm::Ref<vector<pat::IsolatedTrack> >(isolatedTracks, isoTrack - isolatedTracks->begin());
    }
  }
  return;
}*/


/*void
TrackImageProducerMINIAOD::findMatchedGenTrack (const edm::Handle<vector<reco::Track> > &genTracks, edm::Ref<vector<reco::Track> > &matchedGenTrack, double &dRToMatchedGenTrack, const CandidateTrack &track) const
{
  dRToMatchedGenTrack = INVALID_VALUE;
  double maxDeltaR_genTrackMatching_ = 0.01;
  for(vector<reco::Track>::const_iterator genTrack = genTracks->begin(); genTrack != genTracks->end(); genTrack++) {
    double dR = deltaR(*genTrack, track);
    if(maxDeltaR_genTrackMatching_ >= 0.0 && dR > maxDeltaR_genTrackMatching_) continue;
    if(dR < dRToMatchedGenTrack || dRToMatchedGenTrack < 0.0) {
      dRToMatchedGenTrack = dR;
      matchedGenTrack = edm::Ref<vector<reco::Track> >(genTracks, genTrack - genTracks->begin());
    }
  }
  return;
}*/

void
TrackImageProducerMINIAOD::getGeometries(const edm::EventSetup &setup) {
  caloGeometry_ = setup.getHandle(caloGeometryToken_);
  if(!caloGeometry_.isValid())
    throw cms::Exception("FatalError") << "Unable to find CaloGeometryRecord in event!\n";

  /*cscGeometry_ = setup.getHandle(cscGeometryToken_);
  if(!cscGeometry_.isValid())
    throw cms::Exception("FatalError") << "Unable to find MuonGeometryRecord (CSC) in event!\n";

  dtGeometry_ = setup.getHandle(dtGeometryToken_);
  if(!dtGeometry_.isValid())
    throw cms::Exception("FatalError") << "Unable to find MuonGeometryRecord (DT) in event!\n";

  rpcGeometry_ = setup.getHandle(rpcGeometryToken_);
  if(!rpcGeometry_.isValid())
    throw cms::Exception("FatalError") << "Unable to find MuonGeometryRecord (RPC) in event!\n";*/

  ecalStatus_ = setup.getHandle(ecalStatusToken_);
  trackerTopology_ = setup.getHandle(trackerTopologyToken_);
  //setup.get<EcalChannelStatusRcd>().get(ecalStatus_);
  //setup.get<TrackerTopologyRcd>().get(trackerTopology_);
}

int
TrackImageProducerMINIAOD::countGoodPrimaryVertices(const vector<reco::Vertex> &vertices) const
{
  int n = 0;
  for(const auto &v : vertices) {
    if(v.isValid() && v.ndof() >= 4 && fabs(v.z()) < 24.0 && hypot(v.x(), v.y()) < 2.0) n++;
  }
  return n;
}

int
TrackImageProducerMINIAOD::countGoodJets(const vector<pat::Jet> &jets) const
{
  int n = 0;
  vector<double> goodJetPhis;

  for(const auto &jet : jets) {
    // let's not deal with jet smearing for now...
    if(jet.pt() <= 110) continue;
    if(fabs(jet.eta()) >= 2.4) continue;
    if(!anatools::jetPassesTightLepVeto(jet)) continue;

    n++;
  }
  return n;
}

double
TrackImageProducerMINIAOD::getMaxDijetDeltaPhi(const vector<pat::Jet> &jets) const
{
  vector<double> goodJetPhis;
  for(const auto &jet : jets) {
    if(jet.pt() > 30 && fabs(jet.eta()) < 4.5 && anatools::jetPassesTightLepVeto(jet)) goodJetPhis.push_back(jet.phi());
  }

  double maxDeltaPhi = -999.;
  for(unsigned int i = 0; i < goodJetPhis.size(); i++) {
    for(unsigned int j = 0; j < goodJetPhis.size(); j++) {
      if(i == j) continue;
      double dPhi = fabs(deltaPhi(goodJetPhis[i], goodJetPhis[j]));
      if(dPhi > maxDeltaPhi) maxDeltaPhi = dPhi;
    }
  }

  return maxDeltaPhi;
}

double
TrackImageProducerMINIAOD::getLeadingJetMetPhi(const vector<pat::Jet> &jets, const pat::MET &met) const
{
  double deltaPhiMetJetLeading = 999.;
  double ptJetLeading = -1;

  for(const auto &jet : jets) {
    if(jet.pt() > 30 && 
       fabs(jet.eta()) < 4.5 && 
       anatools::jetPassesTightLepVeto(jet)) {
      if(jet.pt() > ptJetLeading) {
        ptJetLeading = jet.pt();
        deltaPhiMetJetLeading = fabs(deltaPhi(jet, met));
      }
    }
  }

  return deltaPhiMetJetLeading;
}

void
TrackImageProducerMINIAOD::getTracks(const reco::Vertex &pv, 
                                     const vector<pat::Jet> &jets,
                                     const vector<pat::Electron> &electrons,
                                     const vector<pat::Muon> &muons,
                                     const vector<pat::Tau> &taus,
                                     const vector<pat::Electron> &tagElectrons,
                                     const vector<pat::Muon> &tagMuons,
                                     const pat::MET &met,
                                     const edm::Handle<vector<pat::IsolatedTrack> > tracks, 
                                     const edm::Handle<reco::DeDxHitInfoAss> isoTrk2dedxHitInfo,
                                     const edm::Handle<edm::ValueMap<reco::DeDxData> > dEdxStrip,
                                     const edm::Handle<edm::ValueMap<reco::DeDxData> > dEdxPixel)
{
  trackInfos_.clear();


  //for(const auto &track : *tracks) {
  //for(auto track = std::begin(tracks); track != std::endl(tracks); track++) {
  for(vector<pat::IsolatedTrack>::const_iterator it_track = tracks->begin(); it_track != tracks->end(); it_track++) {

    
    TrackInfo info;
    pat::IsolatedTrack track = *it_track;
    
    //apply track pt cut
    if(minTrackPt_ > 0 && track.pt() <= minTrackPt_) continue;
    if(maxTrackEta_ > 0 && abs(track.eta()) > maxTrackEta_) continue;

    info.trackIso = 0.0;
    for(const auto &t : *tracks) {
      const auto theptinv2 = 1 / pow(track.pt(),2);
      float dz_track = (track.vz() - t.vz()) - ((track.vx() - t.vx()) * track.px() + (track.vy() - t.vy()) * track.py()) * track.pz() * theptinv2;
      if(fabs(dz_track) > 3.0 * hypot(track.dzError(), t.dzError())) continue;
      double dR = deltaR(track, t);
      if(dR < 0.3 && dR > 1.0e-12) info.trackIso += t.pt();
    }

    // apply relative track isolation cut
    if(maxRelTrackIso_ > 0 && info.trackIso / track.pt() >= maxRelTrackIso_) continue;

    info.px = track.px();
    info.py = track.py();
    info.pz = track.pz();
    info.vx = track.vx();
    info.vy = track.vy();
    info.vz = track.vz();
    std::cout << "Track vertex: " << track.vx() << ", " << track.vy() <<", " << track.vz() << std::endl;
    info.eta = track.eta();
    info.pt = track.pt();
    //info.ptError = track.ptError();
    info.phi = track.phi();
    info.charge = track.charge();

    info.dEdxInfo.clear();

    edm::Ref<vector<pat::IsolatedTrack> > matchedIsolatedTrack = edm::Ref<vector<pat::IsolatedTrack> >(tracks, it_track - tracks->begin());

    //double dRToMatchedIsolatedTrack;
    //findMatchedIsolatedTrack(isoTracks, matchedIsolatedTrack, dRToMatchedIsolatedTrack, track);

    //if(dRToMatchedIsolatedTrack == INVALID_VALUE) {
    //  info.dEdxInfo.push_back(TrackDeDxInfo());
    //}      
    

    if(isoTrk2dedxHitInfo->contains(matchedIsolatedTrack.id())) {
      const reco::DeDxHitInfo * hitInfo = (*isoTrk2dedxHitInfo)[matchedIsolatedTrack].get();
      if(hitInfo == nullptr) {
        //edm::LogWarning ("disappTrks_DeDxHitInfoVarProducer") << "Encountered a null DeDxHitInfo object from a pat::IsolatedTrack? Skipping this track...";
        continue;
      }

      for(unsigned int iHit = 0; iHit < hitInfo->size(); iHit++) {
        bool isPixel = (hitInfo->pixelCluster(iHit) != nullptr);
        bool isStrip = (hitInfo->stripCluster(iHit) != nullptr);
        if(!isPixel && !isStrip) continue; // probably shouldn't happen
        if(isPixel && isStrip) continue;
        
        //subdet Id = {1, pbx}, {2, pxf}, {3, tib}, {4, tid}, {5, tob}, {6, tec}
        int subDet = hitInfo->detId(iHit).subdetId();
        if(subDet == PixelSubdetector::PixelBarrel) subDet = 1;
        else if (subDet == PixelSubdetector::PixelEndcap) subDet = 2;
        else if(subDet == StripSubdetector::TIB) subDet = 3;  //N.B. in CMSSW_11 StripSubdetector -> SiStripSubdetector
        else if (subDet == StripSubdetector::TID) subDet = 4;
        else if (subDet == StripSubdetector::TOB) subDet = 5;
        else if (subDet == StripSubdetector::TEC) subDet = 6;

        float norm = isPixel ? 3.61e-06 : 3.61e-06 * 265;

        info.dEdxInfo.push_back(
          TrackDeDxInfo(subDet,
                        norm * hitInfo->charge(iHit) / hitInfo->pathlength(iHit),
                        isPixel ? hitInfo->pixelCluster(iHit)->size()  : -1,
                        isPixel ? hitInfo->pixelCluster(iHit)->sizeX() : -1,
                        isPixel ? hitInfo->pixelCluster(iHit)->sizeY() : -1,
#if CMSSW_VERSION_CODE >= CMSSW_VERSION(12,4,0)
                        isStrip ? deDxTools::shapeSelection(*(hitInfo->stripCluster(iHit))) : false,
#else
                        isStrip ? deDxTools::shapeSelection(*(hitInfo->stripCluster(iHit))) : false,
#endif
                        hitInfo->pos(iHit).x(),
                        hitInfo->pos(iHit).y(),
                        hitInfo->pos(iHit).z(),
                        trackerTopology_->layer(hitInfo->detId(iHit)))); // gives layer within sub detector
      }
    } // if isoTrk in association map
    else {
      info.dEdxInfo.push_back(TrackDeDxInfo()); // if somehow the matched isoTrk isn't in the hitInfo?
    }

    /*edm::Ref<vector<reco::Track> > matchedGenTrack;
    double dRToMatchedGenTrack;
    findMatchedGenTrack(genTracks, matchedGenTrack, dRToMatchedGenTrack, track);
    if(dRToMatchedGenTrack == INVALID_VALUE){
      info.dEdxPixel = -10;
      info.numMeasurementsPixel = -10;
      info.numSatMeasurementsPixel = -10;
      info.dEdxStrip = -10;
      info.numMeasurementsStrip = -10;
      info.numSatMeasurementsStrip = -10;
    }
    else{
      const reco::DeDxData &dEdxDataPixel = (*dEdxPixel)[matchedGenTrack];
      const reco::DeDxData &dEdxDataStrip = (*dEdxStrip)[matchedGenTrack];

      info.dEdxPixel = dEdxDataPixel.dEdx();
      info.numMeasurementsPixel = dEdxDataPixel.numberOfMeasurements();
      info.numSatMeasurementsPixel = dEdxDataPixel.numberOfSaturatedMeasurements();
      info.dEdxStrip = dEdxDataStrip.dEdx();
      info.numMeasurementsStrip = dEdxDataStrip.numberOfMeasurements();
      info.numSatMeasurementsStrip = dEdxDataStrip.numberOfSaturatedMeasurements();
    }*/

    info.dEdxPixel = track.dEdxPixel();
    info.dEdxStrip = track.dEdxStrip();

    //FIXME get these values from HitMap
    info.numMeasurementsPixel = -10;
    info.numSatMeasurementsPixel = -10;
    info.numMeasurementsStrip = -10;
    info.numSatMeasurementsStrip = -10;


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

    info.nValidPixelHits        = track.hitPattern().numberOfValidPixelHits();
    info.nValidHits             = track.hitPattern().numberOfValidHits();
    info.numberOfValidMuonHits  = track.hitPattern().numberOfValidMuonHits();
    info.missingInnerHits       = track.hitPattern().trackerLayersWithoutMeasurement(reco::HitPattern::MISSING_INNER_HITS);
    info.missingMiddleHits      = track.hitPattern().trackerLayersWithoutMeasurement(reco::HitPattern::TRACK_HITS);
    info.missingOuterHits       = track.hitPattern().trackerLayersWithoutMeasurement(reco::HitPattern::MISSING_OUTER_HITS);
    info.nLayersWithMeasurement = track.hitPattern().trackerLayersWithMeasurement();
    info.pixelLayersWithMeasurement = track.hitPattern().pixelLayersWithMeasurement();

    // d0 wrt pv (2d) = (vertex - pv) cross p / |p|
    info.d0 = ((track.vx() - pv.x()) * track.py() - (track.vy() - pv.y()) * track.px()) / track.pt(); 
  
    // dz wrt pv (2d) = (v_z - pv_z) - p_z * [(vertex - pv) dot p / |p|^2]
    info.dz = track.vz() - pv.z() -
      ((track.vx() - pv.x()) * track.px() + (track.vy() - pv.y()) * track.py()) * track.pz() / track.pt() / track.pt();

    //info.normalizedChi2 = track.normalizedChi2();
    info.highPurityFlag = track.isHighPurityTrack();

    info.deltaRToClosestElectron = -1;
    std::cout << "Number of electrons: " << electrons.size() << ", number of muons: " << muons.size() << std::endl;
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

    info.isTagProbeElectron = 0;
    info.isTagProbeTauToElectron = 0;

    info.isTagProbeMuon = 0;
    info.isTagProbeTauToMuon = 0;

    if(info.passesProbeSelection) {
      for(const auto tag : tagElectrons) {
        double thisDR = deltaR(tag, track);
        if(info.deltaRToClosestTagElectron < 0 || thisDR < info.deltaRToClosestTagElectron) {
          info.deltaRToClosestTagElectron = thisDR;
        }
        info.isTagProbeElectron |= isTagProbeElePair(track, tag);
        info.isTagProbeTauToElectron |= isTagProbeTauToElePair(track, tag, met);
      }

      for(const auto tag : tagMuons) {
        double thisDR = deltaR(tag, track);
        if(info.deltaRToClosestTagMuon < 0 || thisDR < info.deltaRToClosestTagMuon) {
          info.deltaRToClosestTagMuon = thisDR;
        }
        info.isTagProbeMuon |= isTagProbeMuonPair(track, tag);
        info.isTagProbeTauToMuon |= isTagProbeTauToMuonPair(track, tag, met);
      }
    }

    info.ecalo = 0; // calculated in getRecHits

    trackInfos_.push_back(info);
  } //end of track loop

  return;
}

void 
TrackImageProducerMINIAOD::getRecHits(const edm::Event &event)
{
  recHitInfos_.clear();

  edm::Handle<EBRecHitCollection> EBRecHits;
  event.getByToken(EBRecHitsToken_, EBRecHits);
  for(const auto &hit : *EBRecHits) {
    math::XYZVector pos = getPosition(hit.detid());
    recHitInfos_.push_back(RecHitInfo(pos.eta(), pos.phi(), hit.energy(), -999., DetType::EB));

    for(auto &info : trackInfos_) {
      double dR2 = deltaR2(pos.eta(), pos.phi(), info.eta, info.phi);
      if(dR2 < 0.5*0.5) info.ecalo += hit.energy();
    }
  }

  edm::Handle<EERecHitCollection> EERecHits;
  event.getByToken(EERecHitsToken_, EERecHits);
  for(const auto &hit : *EERecHits) {
    math::XYZVector pos = getPosition(hit.detid());
    recHitInfos_.push_back(RecHitInfo(pos.eta(), pos.phi(), hit.energy(), -999., DetType::EE));

    for(auto &info : trackInfos_) {
      double dR2 = deltaR2(pos.eta(), pos.phi(), info.eta, info.phi);
      if(dR2 < 0.5*0.5) info.ecalo += hit.energy();
    }
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

    for(auto &info : trackInfos_) {
      double dR2 = deltaR2(pos.eta(), pos.phi(), info.eta, info.phi);
      if(dR2 < 0.5*0.5) info.ecalo += hit.energy();
    }
  }

  /*edm::Handle<CSCSegmentCollection> CSCSegments;
  event.getByToken(CSCSegmentsToken_, CSCSegments);
  for(const auto &seg : *CSCSegments) {
    vector<CSCRecHitInfo> CSCRecHits;
    vector<DTRecHitInfo> DTRecHits;
    for(const auto &recHit : seg.specificRecHits()) {
      int iLayer   = recHit.cscDetId().layer();
      int iChamber = recHit.cscDetId().chamber();
      int iRing    = recHit.cscDetId().ring();
      int iStation = recHit.cscDetId().station();
      int iEndcap  = recHit.cscDetId().endcap(); 
      math::XYZVector recHitPos = getPosition(recHit);
      CSCRecHits.push_back(
        CSCRecHitInfo(recHitPos.eta(),recHitPos.phi(),recHitPos.x(),recHitPos.y(),recHitPos.z(), recHit.tpeak(),
                      iLayer,iChamber,iRing,iStation,iEndcap)
      );
    }
    math::XYZVector pos = getPosition(seg);
    recHitInfos_.push_back(RecHitInfo(pos.eta(), pos.phi(), -1, seg.time(), CSCRecHits, DetType::CSC));
  }

  edm::Handle<DTRecSegment4DCollection> DTRecSegments;
  event.getByToken(DTRecSegmentsToken_, DTRecSegments);
  for(const auto &seg : *DTRecSegments) {
    vector<CSCRecHitInfo> CSCRecHits;
    vector<DTRecHitInfo> DTRecHits;
    double time = -999.;
    DTChamberId chamber = seg.chamberId();
    if(seg.hasPhi()) {
      time = seg.phiSegment()->t0();
      for(const auto recHit : seg.phiSegment()->specificRecHits()){
        DTWireId wire = recHit.wireId();
        DTLayerId layer = wire.layerId();
        DTSuperLayerId superlayer = layer.superlayerId();
        math::XYZVector recHitPos = getPosition(recHit);
        DTRecHits.push_back( DTRecHitInfo(recHitPos.eta(),recHitPos.phi(),recHitPos.x(),recHitPos.y(),recHitPos.z(), recHit.digiTime(),
                             wire.wire(), layer.layer(), superlayer.superlayer(), chamber.wheel(), chamber.station(), chamber.sector())
        ); 
      }
    }
    if(seg.hasZed()) {
      time = seg.zSegment()->t0();
      for(const auto recHit : seg.zSegment()->specificRecHits()){
        DTWireId wire = recHit.wireId();
        DTLayerId layer = wire.layerId();
        DTSuperLayerId superlayer = layer.superlayerId();
        math::XYZVector recHitPos = getPosition(recHit);
        DTRecHits.push_back( DTRecHitInfo(recHitPos.eta(),recHitPos.phi(),recHitPos.x(),recHitPos.y(),recHitPos.z(), recHit.digiTime(),
                             wire.wire(), layer.layer(), superlayer.superlayer(), chamber.wheel(), chamber.station(), chamber.sector())
        );
      }
    }
    math::XYZVector pos = getPosition(seg);
    recHitInfos_.push_back(RecHitInfo(pos.eta(), pos.phi(), -1, time, DTRecHits, DetType::DT));
  }

  edm::Handle<RPCRecHitCollection> RPCRecHits;
  event.getByToken(RPCRecHitsToken_, RPCRecHits);
  for(const auto &hit : *RPCRecHits) {
    math::XYZVector pos = getPosition(hit);
    recHitInfos_.push_back(RecHitInfo(pos.eta(), pos.phi(), -1, -999., DetType::RPC));
  }*/

}

void
TrackImageProducerMINIAOD::getGenParticles(const reco::CandidateView &genParticles){
                                          
  
  genParticleInfos_.clear();

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

    info.vx = p->vx();
    info.vy = p->vy();
    info.vz = p->vz();

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

/*const math::XYZVector
TrackImageProducerMINIAOD::getPosition(const CSCSegment& seg) const
{
  const LocalPoint localPos = seg.localPosition();
  const CSCDetId id = seg.cscDetId();

  const GlobalPoint idPosition = cscGeometry_->chamber(id)->toGlobal(localPos);
  math::XYZVector idPositionRoot(idPosition.x(), idPosition.y(), idPosition.z());
  return idPositionRoot;
}

const math::XYZVector
TrackImageProducerMINIAOD::getPosition(const CSCRecHit2D& recHit) const
{
  const LocalPoint localPos = recHit.localPosition();
  const CSCDetId id = recHit.cscDetId();

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
TrackImageProducerMINIAOD::getPosition(const DTRecHit1D& recHit) const
{
  const LocalPoint localPos = recHit.localPosition();
  const GlobalPoint idPosition = dtGeometry_->idToDet(recHit.geographicalId())->surface().toGlobal(localPos);
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
}*/

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

const unsigned int
TrackImageProducerMINIAOD::isTagProbeElePair(const pat::IsolatedTrack &probe, const pat::Electron &tag) const 
{
  TLorentzVector t(tag.px(), tag.py(), tag.pz(), tag.energy());
  TLorentzVector p(probe.px(), 
                   probe.py(), 
                   probe.pz(), 
                   sqrt(probe.px() * probe.px() + 
                        probe.py() * probe.py() + 
                        probe.pz() * probe.pz() + 
                        0.000510998928 * 0.000510998928)); // energyOfElectron()

  if(fabs((t + p).M() - 91.1876) >= 10.0) return 0b00;
  return (tag.charge() * probe.charge() < 0) ? 0b01 : 0b10;
}

const unsigned int
TrackImageProducerMINIAOD::isTagProbeTauToElePair(const pat::IsolatedTrack &probe, 
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

  if(invMass <= 91.1876 - 50 || invMass >= 91.1876 - 15) return 0b00;
  return (tag.charge() * probe.charge() < 0) ? 0b01 : 0b10;
}

const unsigned int
TrackImageProducerMINIAOD::isTagProbeMuonPair(const pat::IsolatedTrack &probe, const pat::Muon &tag) const 
{
  TLorentzVector t(tag.px(), tag.py(), tag.pz(), tag.energy());
  TLorentzVector p(probe.px(), 
                   probe.py(), 
                   probe.pz(), 
                   sqrt(probe.px() * probe.px() + 
                        probe.py() * probe.py() + 
                        probe.pz() * probe.pz() + 
                        0.1056583715 * 0.1056583715)); // energyOfMuon()

  if(fabs((t + p).M() - 91.1876) >= 10.0) return 0b00;
  return (tag.charge() * probe.charge() < 0) ? 0b01 : 0b10;
}

const unsigned int
TrackImageProducerMINIAOD::isTagProbeTauToMuonPair(const pat::IsolatedTrack &probe, 
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

  if(invMass <= 91.1876 - 50 || invMass >= 91.1876 - 15) return 0b00;
  return (tag.charge() * probe.charge() < 0) ? 0b01 : 0b10;
}

const double
TrackImageProducerMINIAOD::minDRBadEcalChannel(const pat::IsolatedTrack &track) const
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
