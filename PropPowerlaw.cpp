// Implementation of the PropPowerlaw class

#include "PropPowerlaw.H"
#include <cmath>

using namespace criptic;
using namespace criptic::propagation;

// Constructor
PropPowerlaw::PropPowerlaw(const ParmParser& pp) {

  // Read parameters from input file
  if (!pp.query("cr.kPar0", kPar0)) kPar0 = 0;
  if (!pp.query("cr.kParIdx", kParIdx)) kParIdx = 0;
  if (!pp.query("cr.kPerp0", kPerp0)) kPerp0 = 0;
  if (!pp.query("cr.kPerpIdx", kPerpIdx)) kPerpIdx = 0;
  if (!pp.query("cr.kPP0", kPP0)) kPP0 = 0;
  if (!pp.query("cr.kPPIdx", kPPIdx)) kPPIdx = 0;
  if (!pp.query("cr.vStr0", vStr0)) vStr0 = 0;
  if (!pp.query("cr.vStrIdx", vStrIdx)) vStrIdx = 0;

  // Adjust kPP0 from physical units to code units; the physical unit
  // convention in criptic is the momenta are measured in units of
  // GeV/c, and kPP0 has units of momentum^2 / time, so the input
  // quantity we have been given is in units of (GeV / c)^2 / s,
  // whereas the code works in internal units of (m_p c)^2 / s.
  kPP0 *= (units::GeV / constants::mp_c2) * (units::GeV / constants::mp_c2);

  // Get streaming direction and whether vStr0 is absolute or relative
  // to Alfven speed
  if (vStr0 == 0) {
    varStreamDir = false;
    vAStream = false;
  } else {
    int varStreamDir_;
    if (!pp.query("cr.varStreamDir", varStreamDir_)) varStreamDir = true;
    else varStreamDir = varStreamDir_;
    int vAStream_;
    if (!pp.query("cr.vAStream", vAStream_)) vAStream = true;
    else vAStream = vAStream_;
  }
}


// Propagation coefficient calculation
inline PropagationData
PropPowerlaw::operator()(const RealVec& x,
			 const Real t,
			 const gas::GasData& gd,
			 const CRPacket& packet,
			 const FieldQty& qty,
			 const FieldQtyGrad& qtyGrad) const {

  // Fill propagation data
  PropagationData pd;

  // Diffusion coefficients
  pd.kPar = kPar0 * pow(packet.p, kParIdx);
  pd.kPerp = kPerp0 * pow(packet.p, kPerpIdx);
  pd.kPP = kPP0 * pow(packet.p, kPPIdx);
  pd.kParGrad = zeroVec;
  pd.kPerpGrad = zeroVec;
  pd.dkPP_dp = kPPIdx * pd.kPP/packet.p;

  // Streaming; action depends on whether streaming is relative to
  // Alfven speed or absolute
  if (!vAStream && !varStreamDir) {

    // Streaming is absolute and one-directional
    pd.vStr = vStr0 * pow(packet.p, vStrIdx);
    pd.vStrGrad = zeroVec;
    pd.dvStr_dp = vStrIdx * pd.vStr/packet.p;
    
  } else {
    
    // If streaming is relative to pressure gradient, get the direction
    // of the pressure gradient relative to B field vector
    int dir = 1;
    if (varStreamDir) {
      Real dot = qtyGrad[presIdx].dot(gd.B);
      if (dot > 0) dir = -1;
      else if (dot == 0) dir = 0;
      //std::cout << "dir = " << dir << std::endl;
    }

    // Set absolute or relative streaming speed
    if (!vAStream) {
  
      // Streaming speed is absolute
      pd.vStr = dir * vStr0 * pow(packet.p, vStrIdx);
      pd.vStrGrad = zeroVec;
      
    } else {
      
      // Streaming speed is relative to ion Alfven speed; note that,
      // to compute the gradient, we use the fact that vStr ~ B /
      // sqrt(rho_ion) to write out the derivative as dvStr / dx_i =
      // vStr [ (1/|B|) d|B|/dx_i - (1/2) (1/rho_ion) d(rho_ion)/dx_i]

      //------------------------------------------------//
      // Matts edit for multi-chi
      // Init new real val
      
      Real ion_den_new = gd.ionDen;       // rename as cannot alter gd.ionDen directly
      RealVec ionGrad = gd.ionDenGrad;    // dummy var for ionDensity gradient
      Real src_batch = 81;                // number of sources per chi value

      // Now check to see the ion density needed for source batch
      if (packet.src > src_batch) {
          ion_den_new = gd.ionDen * 0.1; //  chi = 1
          ionGrad = gd.ionDenGrad * 0.1;
      }

      if (packet.src > 2*src_batch) {
          ion_den_new = gd.ionDen * 0.01; //  chi = 2
          ionGrad = gd.ionDenGrad * 0.01;
      }

      if (packet.src > 3*src_batch) {
          ion_den_new = gd.ionDen * 0.001; //  chi = 3
          ionGrad = gd.ionDenGrad * 0.001;
      }

      if (packet.src > 4*src_batch) {
          ion_den_new = gd.ionDen * 0.0001; //  chi = 4
          ionGrad = gd.ionDenGrad * 0.0001;
      }

      if (packet.src > 5*src_batch) {
          ion_den_new = gd.ionDen * 0.00001; //  chi = 5
          ionGrad = gd.ionDenGrad * 0.00001;
      }
      //--------------------------------------------------//      

      Real Bmag = gd.B.mag();
      RealVec gradBmag = gd.BGrad.contract2(gd.B) / Bmag;
#ifdef CRIPTIC_UNITS_CGS
      Real vAi = Bmag / sqrt(4 * M_PI * ion_den_new);
#else
      Real vAi = Bmag / sqrt(constants::mu0 * ion_den_new);
#endif
      pd.vStr = dir * vAi * vStr0 * pow(packet.p, vStrIdx);
      pd.vStrGrad = pd.vStr *
	(gradBmag / Bmag - ionGrad / (2*ion_den_new));
    }
  }

  // Derivative of vStr with respect to momentum is the same in any
  // case
  pd.dvStr_dp = vStrIdx * pd.vStr/packet.p;

  // Return
  return pd;
}
