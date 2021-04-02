#ifndef INFO_STRUCTS
#define INFO_STRUCTS

#include <vector>
#include "TLorentzVector.h"

struct TrackDeDxInfo {
  int   subDet;
  float charge;
  int   pixelHitSize, pixelHitSizeX, pixelHitSizeY;
  bool  stripShapeSelection;
  float hitPosX, hitPosY, hitPosZ;
  int   hitLayerId;

  TrackDeDxInfo() {
    subDet = -10;
    charge  = -10;
    pixelHitSize  = -10;
    pixelHitSizeX = -10;
    pixelHitSizeY = -10;
    stripShapeSelection = false;
    hitPosX = -50;
    hitPosY = -50;
    hitPosZ = -50;
    hitLayerId = -10;
  }

  TrackDeDxInfo(int subDet_,
                float charge_,
                int pixelHitSize_,
                int pixelHitSizeX_,
                int pixelHitSizeY_,
                bool stripShapeSelection_,
                float hitPosX_,
                float hitPosY_,
                float hitPosZ_,
                int hitLayerId_) {
    subDet = subDet_;
    charge = charge_;
    pixelHitSize = pixelHitSize_;
    pixelHitSizeX = pixelHitSizeX_;
    pixelHitSizeY = pixelHitSizeY_;
    stripShapeSelection = stripShapeSelection_;
    hitPosX = hitPosX_;
    hitPosY = hitPosY_;
    hitPosZ = hitPosZ_;
    hitLayerId = hitLayerId_;
  }

};

struct TrackInfo {
  double deltaRToClosestElectron, deltaRToClosestMuon, deltaRToClosestTauHad;
  double dRMinJet;
  double dRMinBadEcalChannel;

  double ecalo;

  double trackIso;
  double px, py, pz, pt;
  double vx, vy, vz;
  double ptError;
  double eta, phi;
  int nValidPixelHits, nValidHits, numberOfValidMuonHits;
  int missingInnerHits, missingMiddleHits, missingOuterHits, nLayersWithMeasurement, pixelLayersWithMeasurement;
  double d0, dz;
  double normalizedChi2;
  bool highPurityFlag;
  int charge;

  bool inGap;

  bool passesProbeSelection;
  double deltaRToClosestTagElectron, deltaRToClosestTagMuon;

  // bit values reflect tag+probe status and charge products:
  //     0b<in same-sign pair><in opposite-sign pair>
  //     So 0 = 0b00 : not in any TP pair
  //        1 = 0b01 : in OS TP pair
  //        2 = 0b10 : in SS TP pair
  //        3 = 0b11 : in both an OS and SS pair
  unsigned int isTagProbeElectron, isTagProbeMuon;
  unsigned int isTagProbeTauToElectron, isTagProbeTauToMuon;

  double dEdxPixel, dEdxStrip;
  int numMeasurementsPixel, numMeasurementsStrip;
  int numSatMeasurementsPixel, numSatMeasurementsStrip;

  std::vector<TrackDeDxInfo> dEdxInfo;

  TrackInfo() { dEdxInfo.clear(); }
};

struct CSCRecHitInfo {
  double eta, phi, x, y, z, tpeak;
  int layer, chamber, ring, station, endcap;

  CSCRecHitInfo() {}

  CSCRecHitInfo(double eta_, double  phi_, double x_, double y_, double z_, double tpeak_,
                int layer_, int chamber_, int ring_, int station_, int endcap_):
    eta(eta_),
    phi(phi_),
    x(x_),
    y(y_),
    z(z_),
    tpeak(tpeak_),
    layer(layer_),
    chamber(chamber_),
    ring(ring_),
    station(station_),
    endcap(endcap_) {}
};

struct DTRecHitInfo {
  double eta, phi, x, y, z, digitime;
  int wire, layer, superlayer, wheel, station, sector;

  DTRecHitInfo() {}

  DTRecHitInfo(double eta_, double  phi_, double x_, double y_, double z_, double digitime_,
               int wire_, int layer_, int superlayer_, int wheel_, int station_, int sector_):
    eta(eta_),
    phi(phi_),
    x(x_),
    y(y_),
    z(z_),
    digitime(digitime_),
    wire(wire_),
    layer(layer_),
    superlayer(superlayer_),
    wheel(wheel_), 
    station(station_),
    sector(sector_) {}
};

struct RecHitInfo {
  double eta, phi, energy, time;
  int detType;
  std::vector<CSCRecHitInfo> cscRecHits;
  std::vector<DTRecHitInfo> dtRecHits;

  RecHitInfo() {}

  RecHitInfo(double eta_, double phi_, double energy_, double time_, int detType_) :
    eta(eta_),
    phi(phi_),
    energy(energy_),
    time(time_),
    detType(detType_) {}

  RecHitInfo(double eta_, double phi_, double energy_, double time_, std::vector<CSCRecHitInfo> cscRecHits_, int detType_) :
    eta(eta_),
    phi(phi_),
    energy(energy_),
    time(time_),
    detType(detType_) {
      cscRecHits = cscRecHits_;
    }

  RecHitInfo(double eta_, double phi_, double energy_, double time_, std::vector<DTRecHitInfo> dtRecHits_, int detType_) :
    eta(eta_),
    phi(phi_),
    energy(energy_),
    time(time_),
    detType(detType_) {
      dtRecHits = dtRecHits_;
    }
};

struct GenParticleInfo {
  double px, py, pz, e;
  double eta, phi, pt;
  double vx, vy, vz;

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

struct VertexInfo {
  TLorentzVector vertex, vertex_error;
  double chi2;
  int ndof;
  bool isValid;

  VertexInfo() {}

  VertexInfo(TLorentzVector vertex_, TLorentzVector vertex_error_, double chi2_, int ndof_, bool isValid_){
    vertex = vertex_;
    vertex_error = vertex_error_; 
    chi2 = chi2_;
    ndof = ndof_;
    isValid = isValid_;
  }                   

};

#endif
