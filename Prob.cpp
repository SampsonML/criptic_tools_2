// This file provides a problem setup for turbulent diffusion of
// streaming cosmic rays

#include "../Prob.H"
#include "../../Gas/CartesianTimeInterpFLASH.H"
#include "../../Propagation/PropPowerlaw.H"
#include "../../Utils/Constants.H"
#include "../../Utils/PartTypes.H"
#include "../../Utils/SR.H"
#include "../../Utils/Units.H"

using namespace criptic;
using namespace criptic::gas;
using namespace criptic::propagation;
using namespace std;

// Set up the background gas; this problem uses the
// CartesianTimeInterpFLASH setup
Gas *criptic::initGas(const ParmParser &pp,
		      const Geometry &geom) {
  return new CartesianTimeInterpFLASH(pp, geom);
}

// Set up the initial packets -- trivial since there are none
void criptic::initPackets(const ParmParser &pp,
			  vector<RealVec>& x,
			  vector<CRPacket>& packets,
			  RngThread &rng) { }

// Set up the CR propagation model; this test uses the simple
// PropPowerlaw model
Propagation *criptic::initProp(const ParmParser &pp) {
  return new PropPowerlaw(pp);
}

// Set up a single source at the origin
void criptic::initSources(const ParmParser &pp,
			  vector<RealVec>& x,
			  vector<CRSource>& sources,
			  RngThread &rng) {

  // Only one MPI rank does this
  if (MPIUtil::IOProc) {
  

    // Grab upper and lower bounds
    //pp.get("geometry.prob_lo", bound_low);
    //pp.get("geometry.prob_hi", bound_hi);
    Real bound_hi = 3.09e19;

    // Edit for  multi chi per source
    Real n_chi =  6;

    // Create sources
    Real numSrc = n_chi * 81;
    x.resize(numSrc);
    sources.resize(numSrc);
    Real Domain = (bound_hi) / 1.2 ;
    Real Count = -1;
    for (IdxType i=0; i<numSrc; i++) {

      if ((i % 9) == 0) {
         Count = Count + 1;
      }

      // Reset count when moving to next chi batch
      if (Count > 9) {
          Count = 0;
      }

      // Matt's multi-source code
      x[i][2] = -bound_hi;                        // z
      x[i][1] = -Domain +  (i % 9) * Domain/4;    // y
      x[i][0] = -Domain +  Count * Domain/4;      // x


      // sources info
      sources[i].type = partTypes::proton;
      sources[i].p0 = sources[i].p1 =
      sr::p_from_T(partTypes::proton,
  	     units::GeV / constants::mp_c2);
      sources[i].setLum(1);

    }
  }
}

// Empty userWork function
void criptic::userWork(const Real t,
		       Real& dt,
		       Gas& gasBG,
		       Propagation& prop,
		       CRTree& tree) { }
