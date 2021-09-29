export AbstractSGStkeModel, NoSGStke, SGStkeModel

abstract type AbstractSGStkeModel end

struct NoSGStke <: AbstractSGStkeModel end
struct SGStkeModel <: AbstractSGStkeModel end

vars_state(::AbstractSGStkeModel, ::AbstractStateType, FT) = @vars()

vars_state(sgstke::SGStkeModel, ::Prognostic, FT) = @vars(ρe_SGS::FT)
vars_state(sgstke::SGStkeModel, ::Gradient, FT) = @vars(e_SGS::FT)
vars_state(sgstke::SGStkeModel, ::GradientFlux, FT) =
    @vars(∇e_SGS::SVector{3, FT})
vars_state(sgstke::SGStkeModel, ::Auxiliary, FT) = @vars(Δ::FT)

function atmos_init_aux!(
    ::AbstractSGStkeModel,
    ::BalanceLaw,
    aux::Vars,
    geom::LocalGeometry,
)
    nothing
end

function atmos_init_aux!(
    ::SGStkeModel,
    ::BalanceLaw,
    aux::Vars,
    geom::LocalGeometry,
)
    aux.sgstke.Δ = lengthscale(geom)
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
end

# AtmosLESConfigType's default numerical_flux_first_order is RusanovNumericalFlux(), 
# and it requires a `flux_first_order!` and `wavespeed` method for the balance law.
function wavespeed_tracers!(
    ::AbstractSGStkeModel,
    wavespeed::Vars,
    nM,
    state::Vars,
    aux::Vars,
    t::Real,
)
    nothing
end

function wavespeed_tracers!(
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

    wavespeed.sgstke.ρe_SGS = fill(uN, SVector{N, FT})

    return nothing
end


#####
##### Tendency specification
#####
eq_tends(pv::SGSTKE, m::SGStkeModel, tt::Flux{SecondOrder}) =
    (Diffusion(),)

#eq_tends(pv::SGSTKE, m::SGStkeModel, tt::Flux{SecondOrder}) = nothing

