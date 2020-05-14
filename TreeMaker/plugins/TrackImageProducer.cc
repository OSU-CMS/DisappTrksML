#include "DisappTrks/SignalMC/plugins/TrackImageProducer.h"

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

  EBRecHits_    (cfg.getParameter<edm::InputTag> ("EBRecHits")),
  EERecHits_    (cfg.getParameter<edm::InputTag> ("EERecHits")),
  HBHERecHits_  (cfg.getParameter<edm::InputTag> ("HBHERecHits")),

  tauDecayModeFinding_      (cfg.getParameter<edm::InputTag> ("tauDecayModeFinding")),
  tauElectronDiscriminator_ (cfg.getParameter<edm::InputTag> ("tauElectronDiscriminator")),
  tauMuonDiscriminator_     (cfg.getParameter<edm::InputTag> ("tauMuonDiscriminator")),

  minGenParticlePt_   (cfg.getParameter<double> ("minGenParticlePt")),
  minTrackPt_         (cfg.getParameter<double> ("minTrackPt")),
  maxRelTrackIso_     (cfg.getParameter<double> ("maxRelTrackIso")),
  maxDEtaTrackRecHit_ (cfg.getParameter<double> ("maxDEtaTrackRecHit")),
  maxDPhiTrackRecHit_ (cfg.getParameter<double> ("maxDPhiTrackRecHit"))
{
  tracksToken_       = consumes<vector<reco::Track> >       (tracks_);
  genParticlesToken_ = consumes<vector<reco::GenParticle> > (genParticles_);
  electronsToken_    = consumes<vector<reco::GsfElectron> > (electrons_);
  muonsToken_        = consumes<vector<reco::Muon> >        (muons_);
  tausToken_         = consumes<vector<reco::PFTau> >       (taus_);
  pfCandidatesToken_ = consumes<vector<reco::PFCandidate> > (pfCandidates_);
  EBRecHitsToken_    = consumes<EBRecHitCollection>         (EBRecHits_);
  EERecHitsToken_    = consumes<EERecHitCollection>         (EERecHits_);
  HBHERecHitsToken_  = consumes<HBHERecHitCollection>       (HBHERecHits_);

  tauDecayModeFindingToken_      = consumes<reco::PFTauDiscriminator> (tauDecayModeFinding_);
  tauElectronDiscriminatorToken_ = consumes<reco::PFTauDiscriminator> (tauElectronDiscriminator_);
  tauMuonDiscriminatorToken_     = consumes<reco::PFTauDiscriminator> (tauMuonDiscriminator_);

  recHits_dEta.clear();
  recHits_dPhi.clear();
  recHits_energy.clear();
  recHits_detType.clear();

  tree_ = fs_->make<TTree>("tree", "tree");
  tree_->Branch("recHits_dEta", &recHits_dEta);
  tree_->Branch("recHits_dPhi", &recHits_dPhi);
  tree_->Branch("recHits_energy", &recHits_energy);
  tree_->Branch("recHits_detType", &recHits_detType);

  tree_->Branch("genMatchedID", &genMatchedID);
  tree_->Branch("genMatchedDR", &genMatchedDR);
  tree_->Branch("deltaRToClosestElectron", &deltaRToClosestElectron);
  tree_->Branch("deltaRToClosestMuon", &deltaRToClosestMuon);
  tree_->Branch("deltaRToClosestTauHad", &deltaRToClosestTauHad);
  tree_->Branch("eta", &trackEta);
  tree_->Branch("phi", &trackPhi);
  tree_->Branch("pt", &trackPt);
  tree_->Branch("trackIso", &trackIso);

}

TrackImageProducer::~TrackImageProducer()
{
}

void
TrackImageProducer::analyze(const edm::Event &event, const edm::EventSetup &setup)
{
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

  edm::Handle<EBRecHitCollection> EBRecHits;
  event.getByToken(EBRecHitsToken_, EBRecHits);
  
  edm::Handle<EERecHitCollection> EERecHits;
  event.getByToken(EERecHitsToken_, EERecHits);
  
  edm::Handle<HBHERecHitCollection> HBHERecHits;
  event.getByToken(HBHERecHitsToken_, HBHERecHits);

  edm::Handle<reco::PFTauDiscriminator> tauDecayModeFinding;
  event.getByToken(tauDecayModeFindingToken_, tauDecayModeFinding);

  edm::Handle<reco::PFTauDiscriminator> tauElectronDiscriminator;
  event.getByToken(tauElectronDiscriminatorToken_, tauElectronDiscriminator);

  edm::Handle<reco::PFTauDiscriminator> tauMuonDiscriminator;
  event.getByToken(tauMuonDiscriminatorToken_, tauMuonDiscriminator);

  setup.get<CaloGeometryRecord>().get(caloGeometry_);
  if (!caloGeometry_.isValid())
    throw cms::Exception("FatalError") << "Unable to find CaloGeometryRecord in event!\n";

  for(const auto &track : *tracks) {

    if(minTrackPt_ > 0 && track.pt() < minTrackPt_) continue;
    trackIso = getTrackIsolation(track, *tracks);
    if(maxRelTrackIso_ > 0 && trackIso / track.pt() >= maxRelTrackIso_) continue;

    genMatchedID = 0;
    genMatchedDR = -1;
    if(genParticles.isValid()) {
      for(const auto &genParticle : *genParticles) {
        if(genParticle.pt() < minGenParticlePt_) continue;
        if(!genParticle.isPromptFinalState() && !genParticle.isDirectPromptTauDecayProductFinalState()) continue;

        double thisDR = deltaR(genParticle, track);
        if(genMatchedDR < 0 || thisDR < genMatchedDR) {
          genMatchedDR = thisDR;
          genMatchedID = genParticle.pdgId();
        }
      }
    }

    trackEta = track.eta();
    trackPhi = track.phi();
    trackPt = track.pt();

    deltaRToClosestElectron = -1;
    deltaRToClosestMuon = -1;
    deltaRToClosestTauHad = -1;

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

    getImage(track, *EBRecHits, *EERecHits, *HBHERecHits);

    tree_->Fill();
  }

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
  const reco::Track &track,
  const EBRecHitCollection &EBRecHits,
  const EERecHitCollection &EERecHits,
  const HBHERecHitCollection &HBHERecHits)
{

  recHits_dEta.clear();
  recHits_dPhi.clear();
  recHits_energy.clear();
  recHits_detType.clear();

  for(const auto &hit : EBRecHits) {
    math::XYZVector pos = getPosition(hit.detid());

    if(maxDEtaTrackRecHit_ > 0 && fabs(track.eta() - pos.eta()) > maxDEtaTrackRecHit_) continue;
    if(maxDPhiTrackRecHit_ > 0 && fabs(deltaPhi(track, pos)) > maxDPhiTrackRecHit_) continue;

    recHits_dEta.push_back(track.eta() - pos.eta());
    recHits_dPhi.push_back(deltaPhi(track, pos));
    recHits_energy.push_back(hit.energy());
    recHits_detType.push_back(DetType::ECAL);
  }

  for(const auto &hit : EERecHits) {
    math::XYZVector pos = getPosition(hit.detid());

    if(maxDEtaTrackRecHit_ > 0 && fabs(track.eta() - pos.eta()) > maxDEtaTrackRecHit_) continue;
    if(maxDPhiTrackRecHit_ > 0 && deltaPhi(track, pos) > maxDPhiTrackRecHit_) continue;

    recHits_dEta.push_back(track.eta() - pos.eta());
    recHits_dPhi.push_back(deltaPhi(track, pos));
    recHits_energy.push_back(hit.energy());
    recHits_detType.push_back(DetType::ECAL);
  }

  for(const auto &hit : HBHERecHits) {
    math::XYZVector pos = getPosition(hit.detid());

    if(maxDEtaTrackRecHit_ > 0 && fabs(track.eta() - pos.eta()) > maxDEtaTrackRecHit_) continue;
    if(maxDPhiTrackRecHit_ > 0 && deltaPhi(track, pos) > maxDPhiTrackRecHit_) continue;

    recHits_dEta.push_back(track.eta() - pos.eta());
    recHits_dPhi.push_back(deltaPhi(track, pos));
    recHits_energy.push_back(hit.energy());
    recHits_detType.push_back(DetType::HCAL);
  }

}

const math::XYZVector 
TrackImageProducer::getPosition(const DetId& id) const
{
   if ( ! caloGeometry_.isValid() ||
        ! caloGeometry_->getSubdetectorGeometry(id) ||
        ! caloGeometry_->getSubdetectorGeometry(id)->getGeometry(id) ) {
      throw cms::Exception("FatalError") << "Failed to access geometry for DetId: " << id.rawId();
      return math::XYZVector(0,0,0);
   }
   const GlobalPoint idPosition = caloGeometry_->getSubdetectorGeometry(id)->getGeometry(id)->getPosition();
   math::XYZVector idPositionRoot(idPosition.x(), idPosition.y(), idPosition.z());
   return idPositionRoot;
}

DEFINE_FWK_MODULE(TrackImageProducer);
