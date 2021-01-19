#include "DisappTrksML/TreeMaker/interface/Infos.h"

#include <vector>

namespace {
  struct DisappTrksML_TreeMaker {

  	TrackDeDxInfo trackDeDxInfo0;
  	std::vector<TrackDeDxInfo> trackDeDxInfo1;

    TrackInfo trackInfo0;
    std::vector<TrackInfo> trackInfo1;

	CSCRecHitInfo cscRecHitInfo0;
	std::vector<CSCRecHitInfo> cscRecHitInfo1;

	DTRecHitInfo dtRecHitInfo0;
	std::vector<DTRecHitInfo> dtRecHitInfo1;

    RecHitInfo recHitInfo0;
    std::vector<RecHitInfo> recHitInfo1;

    GenParticleInfo genParticleInfo0;
    std::vector<GenParticleInfo> genParticleInfo1;
  };
}
