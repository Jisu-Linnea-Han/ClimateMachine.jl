abstract type SGStkeBC end

"""
    NoSGStkeFlux() :: SGStkeBC

No energy flux across boundary
"""
struct NoSGStkeFlux <: SGStkeBC end
function atmos_sgstke_boundary_state!(
    nf,
    bc_sgstke::NoSGStkeFlux,
    atmos,
    _...,
)
    nothing
end
function atmos_sgstke_normal_boundary_flux_second_order!(
    nf,
    bc_sgstke::NoSGStkeFlux,
    atmos,
    _...,
)
    nothing
end


"""
    PrescribedSGStke(sgs) :: SGStkeBC

Dirichlet boundary condition
"""
struct PrescribedSGStke{SGS} <: SGStkeBC
    e_SGS::SGS
end
function atmos_sgstke_boundary_state!(
    nf,
    bc_sgstke::PrescribedSGStke,
    atmos,
    state⁺,
    args,
)   
    @unpack aux⁻, state⁻, t = args  
    state⁺.sgstke.ρe_SGS = state⁺.ρ * bs_sgstke.e_SGS(state⁻, aux⁻, t)
end
function atmos_sgstke_normal_boundary_flux_second_order!(
    nf,
    bc_sgstke::PrescribedSGStke,
    atmos,
    fluxᵀn,
    args,
)
    @unpack state⁻, aux⁻, diffusive⁻, hyperdiff⁻, t, n⁻ = args
    tend_type = Flux{SecondOrder}()
    _args⁻ = (;
        state = state⁻,
        aux = aux⁻,
        t,
        diffusive = diffusive⁻,
        hyperdiffusive = hyperdiff⁻,
    )
    pargs = merge(_args⁻, (precomputed = precompute(atmos, _args⁻, tend_type),))
    total_flux =
        Σfluxes(SGSTKE(), eq_tends(SGSTKE(), atmos, tend_type), atmos, pargs)
    nd_ρe_SGS = dot(n⁻, total_flux)
    # both sides involve projections of normals, so signs are consistent
    fluxᵀn.sgstke.ρe_SGS += nd_ρe_SGS
end

"""
    PrescribedSGStkeFlux(f_sgs) :: SGStkeBC
Prescribe the net inward SGS tke flux across the boundary by `f_sgs`, a function
with signature `f_sgs(state, aux, t)`, returning the flux (in W/m^2).
"""
struct PrescribedSGStkeFlux{FSGS} <: SGStkeBC
    f_sgs::FSGS
end
function atmos_sgstke_boundary_state!(
    nf,
    bc_sgstke::PrescribedSGStkeFlux,
    atmos,
    _...,
) end
function atmos_sgstke_normal_boundary_flux_second_order!(
    nf,
    bc_sgstke::PrescribedSGStkeFlux,
    atmos,
    fluxᵀn,
    args,
)
    @unpack state⁻, aux⁻, t = args

    # DG normal is defined in the outward direction
    # we want to prescribe the inward flux
    fluxᵀn.sgstke.ρe_SGS -= bc_sgstke.f_sgs(state⁻, aux⁻, t)
end
