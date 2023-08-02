/* GRChombo
 * Copyright 2012 The GRChombo collaboration.
 * Please refer to LICENSE in GRChombo's root directory.
 */

#include "KerrBH4dSTLevel.hpp"
#include "BoxLoops.hpp"
#include "CCZ4RHS.hpp"
#include "ChiTaggingCriterion.hpp"
#include "ComputePack.hpp"
#include "ModifiedCCZ4RHS.hpp"
#include "NanCheck.hpp"
#include "ModifiedGravityConstraints.hpp"
#include "PositiveChiAndAlpha.hpp"
#include "SetValue.hpp"
#include "SixthOrderDerivatives.hpp"
#include "TraceARemoval.hpp"

// Initial data
#include "GammaCalculator.hpp"
#include "KerrBH.hpp"

void KerrBH4dSTLevel::specificAdvance()
{
    // Enforce the trace free A_ij condition and positive chi and alpha
    BoxLoops::loop(make_compute_pack(TraceARemoval(), PositiveChiAndAlpha()),
                   m_state_new, m_state_new, INCLUDE_GHOST_CELLS);

    // Check for nan's
    if (m_p.nan_check)
        BoxLoops::loop(
            NanCheck(), m_state_new, m_state_new, 
	    EXCLUDE_GHOST_CELLS, disable_simd());
}

void KerrBH4dSTLevel::initialData()
{
    CH_TIME("KerrBH4dSTLevel::initialData");
    if (m_verbosity)
        pout() << "KerrBH4dSTLevel::initialData " << m_level << endl;

    // First set everything to zero then calculate initial data  Get the Kerr
    // solution in the variables, then calculate the \tilde\Gamma^i numerically
    // as these are non zero and not calculated in the Kerr ICs
    BoxLoops::loop(
        make_compute_pack(SetValue(0.), KerrBH(m_p.kerr_params, m_dx)),
        m_state_new, m_state_new, INCLUDE_GHOST_CELLS);

    fillAllGhosts();
    BoxLoops::loop(GammaCalculator(m_dx), m_state_new, m_state_new,
                   EXCLUDE_GHOST_CELLS);
}

#ifdef CH_USE_HDF5
void KerrBH4dSTLevel::prePlotLevel()
{
    CouplingAndPotential coupling_and_potential(m_p.coupling_and_potential_params);
    FourDerivScalarTensorWithCouplingAndPotential fdst(coupling_and_potential);
    fillAllGhosts();
    BoxLoops::loop(ModifiedGravityConstraints<FourDerivScalarTensorWithCouplingAndPotential>(
                       fdst, m_dx, m_p.G_Newton, c_Ham, Interval(c_Mom1, c_Mom3)),
                   m_state_new, m_state_diagnostics, EXCLUDE_GHOST_CELLS);
}
#endif /* CH_USE_HDF5 */

void KerrBH4dSTLevel::specificEvalRHS(GRLevelData &a_soln, GRLevelData &a_rhs,
                                      const double a_time)
{
    // Enforce the trace free A_ij condition and positive chi and alpha
    BoxLoops::loop(make_compute_pack(TraceARemoval(), PositiveChiAndAlpha()),
                   a_soln, a_soln, INCLUDE_GHOST_CELLS);

    // Calculate ModifiedCCZ4 right hand side with matter_t = FourDerivScalarTensor
    CouplingAndPotential coupling_and_potential(m_p.coupling_and_potential_params);
    FourDerivScalarTensorWithCouplingAndPotential fdst(coupling_and_potential);
    ModifiedGauge modified_gauge(m_p.modified_gauge_params);
    if (m_p.max_spatial_derivative_order == 4)
    {
        ModifiedCCZ4RHS<FourDerivScalarTensorWithCouplingAndPotential, MovingPunctureGauge,
                         FourthOrderDerivatives, ModifiedGauge>
            my_modified_ccz4(fdst, m_p.ccz4_params, modified_gauge, m_dx, m_p.sigma,
                              m_p.formulation);
        BoxLoops::loop(my_modified_ccz4, a_soln, a_rhs, EXCLUDE_GHOST_CELLS);
    }
    else if (m_p.max_spatial_derivative_order == 6)
    {
        ModifiedCCZ4RHS<FourDerivScalarTensorWithCouplingAndPotential, MovingPunctureGauge,
                         SixthOrderDerivatives, ModifiedGauge>
            my_modified_ccz4(fdst, m_p.ccz4_params, modified_gauge, m_dx, m_p.sigma,
                              m_p.formulation);
        BoxLoops::loop(my_modified_ccz4, a_soln, a_rhs, EXCLUDE_GHOST_CELLS);
    }
}

void KerrBH4dSTLevel::specificUpdateODE(GRLevelData &a_soln,
                                        const GRLevelData &a_rhs, Real a_dt)
{
    // Enforce the trace free A_ij condition
    BoxLoops::loop(TraceARemoval(), a_soln, a_soln, INCLUDE_GHOST_CELLS);
}

void KerrBH4dSTLevel::preTagCells()
{
    // We only use chi in the tagging criterion so only fill the ghosts for chi
    fillAllGhosts(VariableType::evolution, Interval(c_chi, c_chi));
}

void KerrBH4dSTLevel::computeTaggingCriterion(FArrayBox &tagging_criterion,
                                              const FArrayBox &current_state)
{
    BoxLoops::loop(ChiTaggingCriterion(m_dx), current_state, tagging_criterion);
}
