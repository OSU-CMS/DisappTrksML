#ifndef INFO_STRUCTS
#define INFO_STRUCTS

struct TrackInfo {
      int genMatchedID;
      double genMatchedDR, genMatchedPt;
      double deltaRToClosestElectron, deltaRToClosestMuon, deltaRToClosestTauHad;
      double dRMinJet;
      double dRMinBadEcalChannel;

      double trackIso;
      double px, py, pz;
      int nValidPixelHits, nValidHits;
      int missingInnerHits, missingMiddleHits, missingOuterHits, nLayersWithMeasurement;
      double d0, dz;

      bool inGap;

      bool passesProbeSelection;
      bool isTagProbeElectron;
      bool isTagProbeTauToElectron;
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

#endif
