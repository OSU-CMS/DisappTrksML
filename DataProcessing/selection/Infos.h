#ifndef INFO_STRUCTS
#define INFO_STRUCTS

struct TrackInfo {
  double deltaRToClosestElectron, deltaRToClosestMuon, deltaRToClosestTauHad;
  double dRMinJet;
  double dRMinBadEcalChannel;

  double trackIso;
  double px, py, pz, pt;
  double eta, phi;
  int nValidPixelHits, nValidHits;
  int missingInnerHits, missingMiddleHits, missingOuterHits, nLayersWithMeasurement;
  double d0, dz;

  bool inGap;

  bool passesProbeSelection;
  double deltaRToClosestTagElectron, deltaRToClosestTagMuon;
  bool isTagProbeElectron, isTagProbeMuon;
  bool isTagProbeTauToElectron, isTagProbeTauToMuon;
};

struct RecHitInfo {
  double eta, phi, energy;
  int detType;

  RecHitInfo() {}

  RecHitInfo(double eta_, double phi_, double energy_, int detType_) :
    eta(eta_),
    phi(phi_),
    energy(energy_),
    detType(detType_) {}
};

struct GenParticleInfo {
  double px, py, pz, e;
  double eta, phi, pt;

  int pdgId, status;

  // gives range for more than 2
  int mother1_index, mother2_index;
  int daughter1_index, daughter2_index;
  int nMothers, nDaughters;

  bool isPromptFinalState,
       isDirectPromptTauDecayProductFinalState,
       isHardProcess,
       fromHardProcessFinalState,
       fromHardProcessBeforeFSR,
       isFirstCopy,
       isLastCopy,
       isLastCopyBeforeFSR;
};

#endif
