export AbstractSGStkeModel, NoSGStke, SGStkeModel

abstract type AbstractSGStkeModel end

struct NoSGStke <: AbstractSGStkeModel end
struct SGStkeModel <: AbstractSGStkeModel end

###########################
## default option: NoSGStke
###########################

vars_state(::AbstractSGStkeModel, ::AbstractStateType, FT) = @vars()

precompute(::AbstractSGStkeModel, atmos::AtmosModel, args, ts, ::Source) = NamedTuple()

function atmos_init_aux!(
    ::AbstractSGStkeModel,
    ::AtmosModel,
    aux::Vars,
    geom::LocalGeometry,
)
    nothing
end

function atmos_nodal_update_auxiliary_state!(
    ::AbstractSGStkeModel,
    m::AtmosModel,
    state::Vars,
    aux::Vars,
    t::Real,
)
    nothing
end

function compute_gradient_flux!(
    ::AbstractSGStkeModel,
    diffusive::Vars,
    ∇transform::Grad,
    state::Vars,
    aux::Vars,
    t::Real,
)
    nothing
end

function compute_gradient_argument!(
    ::AbstractSGStkeModel,
    atmos::AtmosModel,
    transform,
    state,
    aux,
    t,
)
    nothing
end

function wavespeed_sgstke!(
    ::AbstractSGStkeModel,
    wavespeed::Vars,
    nM,
    state::Vars,
    aux::Vars,
    t::Real,
)
    nothing
end

###########################
## SGStkeModel
###########################

vars_state(sgstke::SGStkeModel, ::Prognostic, FT) = @vars(ρe_SGS::FT)
vars_state(sgstke::SGStkeModel, ::Gradient, FT) = @vars(e_SGS::FT, θ_liq_ice::FT)
vars_state(sgstke::SGStkeModel, ::GradientFlux, FT) = 
    @vars(∇e_SGS::SVector{3, FT}, ∇θ_liq_ice::SVector{3, FT}, ∇u::SMatrix{3, 3, FT, 9})
vars_state(sgstke::SGStkeModel, ::Auxiliary, FT) = @vars(Δ::FT)

function precompute(::SGStkeModel, atmos::AtmosModel, args, ts, ::Source) 
    @unpack state, aux, diffusive, t = args
    ν, D_t, τ = turbulence_tensors(atmos, state, diffusive, aux, t)
    turbulence = (ν = ν, D_t = D_t, τ = τ)
    return turbulence
end

function atmos_init_aux!(
    ::SGStkeModel,
    ::AtmosModel,
    aux::Vars,
    geom::LocalGeometry,
)
    aux.sgstke.Δ = lengthscale(geom)
end

"""
Set the variable to take the gradient of.
"""
function compute_gradient_argument!(
    e_SGS::SGStkeModel,
    atmos::AtmosModel,
    transform,
    state,
    aux,
    t,
)
    transform.sgstke.e_SGS = state.sgstke.ρe_SGS * (1 / state.ρ)
    ts = recover_thermo_state(atmos, state, aux)
    transform.sgstke.θ_liq_ice = liquid_ice_pottemp(ts)
end

compute_gradient_flux!(::AbstractSGStkeModel, _...) = nothing
"""
Compute the gradient of the variable.
"""
function compute_gradient_flux!(
    ::SGStkeModel,
    diffusive, #auxDG
    ∇transform,#gradvars
    state,
    aux,
    t,
)
    diffusive.sgstke.∇e_SGS = ∇transform.sgstke.e_SGS
    diffusive.sgstke.∇θ_liq_ice = ∇transform.sgstke.θ_liq_ice
    diffusive.sgstke.∇u = ∇transform.u
end

# AtmosLESConfigType's default numerical_flux_first_order is RusanovNumericalFlux(), 
# and it requires a `flux_first_order!` and `wavespeed` method for the balance law.

function wavespeed_sgstke!(
    ::SGStkeModel,
    wavespeed::Vars,
    nM,
    state::Vars,
    aux::Vars,
    t::Real,
)
    ρinv = 1 / state.ρ
    u = ρinv * state.ρu
    uN = abs(dot(nM, u))

    wavespeed.sgstke.ρe_SGS = uN

    return nothing
end


#####
##### Tendency specification
#####
eq_tends(pv::SGSTKE, m::SGStkeModel, tt::Flux{SecondOrder}) =
    (Diffusion(),)

eq_tends(pv::SGSTKE, m::SGStkeModel, tt::Source) =
    (ShearProduction(), BuoyancyProduction(), Dissipation(),)
#eq_tends(pv::SGSTKE, m::SGStkeModel, tt::Flux{SecondOrder}) = nothing
