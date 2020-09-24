#include "DisappTrksML/TreeMaker/interface/Infos.h"

#include <vector>

namespace {
  struct DisappTrksML_TreeMaker {
    TrackInfo trackInfo0;
    std::vector<TrackInfo> trackInfo1;

    RecHitInfo recHitInfo0;
    std::vector<RecHitInfo> recHitInfo1;

    GenParticleInfo genParticleInfo0;
    std::vector<GenParticleInfo> genParticleInfo1;
  };
}
