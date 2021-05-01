#!/usr/bin/env julia --project
include("../interface/utilities/boilerplate.jl")

########
# Set up parameters
########
parameters = (
    a    = 6.371229e6,
    Ω    = 7.292e-5,
    g    = 9.80616,
    H    = 30e3,
    pₒ   = 1.0e5,
    k    = 3.0,
    Γ    = 0.005,
    T_E  = 310.0,
    T_P  = 240.0,
    b    = 2.0,
    z_t  = 15e3,
    λ_c  = π / 9,
    ϕ_c  = 2 * π / 9,
    V_p  = 1.0,
    κ    = 2/7,
    R_d = 8.3144598 / 28.97e-3,
    κ_d = 2/7,
    cp_d = (8.3144598 / 28.97e-3) / (2/7),
    cv_d = (8.3144598 / 28.97e-3) / (2/7) - 1 / (2/7),
    day = 86400,
    grav = 9.8,
    p0 = 1e5,
    T_ref = 255,
)

########
# Set up domain
########
domain = SphericalShell(
    radius = planet_radius(param_set), 
    height = 30e3,
)
grid = DiscretizedDomain(
    domain;
    elements = (vertical = 8, horizontal = 30),
    polynomial_order = (vertical = 3, horizontal = 3),
    overintegration_order = (vertical = 0, horizontal = 0),
)

# ########
# # Set up inital condition
# ########
# # additional initial condition parameters
# T_0(𝒫)  = 0.5 * (𝒫.T_E + 𝒫.T_P) 
# A(𝒫)    = 1.0 / 𝒫.Γ
# B(𝒫)    = (T_0(𝒫) - 𝒫.T_P) / T_0(𝒫) / 𝒫.T_P
# C(𝒫)    = 0.5 * (𝒫.k + 2) * (𝒫.T_E - 𝒫.T_P) / 𝒫.T_E / 𝒫.T_P
# H(𝒫)    = 𝒫.R_d * T_0(𝒫) / 𝒫.g
# d_0(𝒫)  = 𝒫.a / 6

# # convenience functions that only depend on height
# τ_z_1(𝒫,r)   = exp(𝒫.Γ * (r - 𝒫.a) / T_0(𝒫))
# τ_z_2(𝒫,r)   = 1 - 2 * ((r - 𝒫.a) / 𝒫.b / H(𝒫))^2
# τ_z_3(𝒫,r)   = exp(-((r - 𝒫.a) / 𝒫.b / H(𝒫))^2)
# τ_1(𝒫,r)     = 1 / T_0(𝒫) * τ_z_1(𝒫,r) + B(𝒫) * τ_z_2(𝒫,r) * τ_z_3(𝒫,r)
# τ_2(𝒫,r)     = C(𝒫) * τ_z_2(𝒫,r) * τ_z_3(𝒫,r)
# τ_int_1(𝒫,r) = A(𝒫) * (τ_z_1(𝒫,r) - 1) + B(𝒫) * (r - 𝒫.a) * τ_z_3(𝒫,r)
# τ_int_2(𝒫,r) = C(𝒫) * (r - 𝒫.a) * τ_z_3(𝒫,r)
# F_z(𝒫,r)     = (1 - 3 * ((r - 𝒫.a) / 𝒫.z_t)^2 + 2 * ((r - 𝒫.a) / 𝒫.z_t)^3) * ((r - 𝒫.a) ≤ 𝒫.z_t)

# # convenience functions that only depend on longitude and latitude
# d(𝒫,λ,ϕ)     = 𝒫.a * acos(sin(ϕ) * sin(𝒫.ϕ_c) + cos(ϕ) * cos(𝒫.ϕ_c) * cos(λ - 𝒫.λ_c))
# c3(𝒫,λ,ϕ)    = cos(π * d(𝒫,λ,ϕ) / 2 / d_0(𝒫))^3
# s1(𝒫,λ,ϕ)    = sin(π * d(𝒫,λ,ϕ) / 2 / d_0(𝒫))
# cond(𝒫,λ,ϕ)  = (0 < d(𝒫,λ,ϕ) < d_0(𝒫)) * (d(𝒫,λ,ϕ) != 𝒫.a * π)

# # base-state thermodynamic variables
# I_T(𝒫,ϕ,r)   = (cos(ϕ) * r / 𝒫.a)^𝒫.k - 𝒫.k / (𝒫.k + 2) * (cos(ϕ) * r / 𝒫.a)^(𝒫.k + 2)
# T(𝒫,ϕ,r)     = (τ_1(𝒫,r) - τ_2(𝒫,r) * I_T(𝒫,ϕ,r))^(-1) * (𝒫.a/r)^2
# p(𝒫,ϕ,r)     = 𝒫.pₒ * exp(-𝒫.g / 𝒫.R_d * (τ_int_1(𝒫,r) - τ_int_2(𝒫,r) * I_T(𝒫,ϕ,r)))
# θ(𝒫,ϕ,r)     = T(𝒫,ϕ,r) * (𝒫.pₒ / p(𝒫,ϕ,r))^𝒫.κ

# # base-state velocity variables
# U(𝒫,ϕ,r)  = 𝒫.g * 𝒫.k / 𝒫.a * τ_int_2(𝒫,r) * T(𝒫,ϕ,r) * ((cos(ϕ) * r / 𝒫.a)^(𝒫.k - 1) - (cos(ϕ) * r / 𝒫.a)^(𝒫.k + 1))
# u(𝒫,ϕ,r)  = -𝒫.Ω * r * cos(ϕ) + sqrt((𝒫.Ω * r * cos(ϕ))^2 + r * cos(ϕ) * U(𝒫,ϕ,r))
# v(𝒫,ϕ,r)  = 0.0
# w(𝒫,ϕ,r)  = 0.0

# # velocity perturbations
# δu(𝒫,λ,ϕ,r)  = -16 * 𝒫.V_p / 3 / sqrt(3) * F_z(𝒫,r) * c3(𝒫,λ,ϕ) * s1(𝒫,λ,ϕ) * (-sin(𝒫.ϕ_c) * cos(ϕ) + cos(𝒫.ϕ_c) * sin(ϕ) * cos(λ - 𝒫.λ_c)) / sin(d(𝒫,λ,ϕ) / 𝒫.a) * cond(𝒫,λ,ϕ)
# δv(𝒫,λ,ϕ,r)  = 16 * 𝒫.V_p / 3 / sqrt(3) * F_z(𝒫,r) * c3(𝒫,λ,ϕ) * s1(𝒫,λ,ϕ) * cos(𝒫.ϕ_c) * sin(λ - 𝒫.λ_c) / sin(d(𝒫,λ,ϕ) / 𝒫.a) * cond(𝒫,λ,ϕ)
# δw(𝒫,λ,ϕ,r)  = 0.0

# # CliMA prognostic variables
# # compute the total energy
# uˡᵒⁿ(𝒫,λ,ϕ,r)   = u(𝒫,ϕ,r) + δu(𝒫,λ,ϕ,r)
# uˡᵃᵗ(𝒫,λ,ϕ,r)   = v(𝒫,ϕ,r) + δv(𝒫,λ,ϕ,r)
# uʳᵃᵈ(𝒫,λ,ϕ,r)   = w(𝒫,ϕ,r) + δw(𝒫,λ,ϕ,r)

# e_int(𝒫,λ,ϕ,r)  = (𝒫.R_d / 𝒫.κ - 𝒫.R_d) * T(𝒫,ϕ,r)
# e_kin(𝒫,λ,ϕ,r)  = 0.5 * ( uˡᵒⁿ(𝒫,λ,ϕ,r)^2 + uˡᵃᵗ(𝒫,λ,ϕ,r)^2 + uʳᵃᵈ(𝒫,λ,ϕ,r)^2 )
# e_pot(𝒫,λ,ϕ,r)  = 𝒫.g * r

# ρ₀(𝒫,λ,ϕ,r)    = p(𝒫,ϕ,r) / 𝒫.R_d / T(𝒫,ϕ,r)
# ρuˡᵒⁿ(𝒫,λ,ϕ,r) = ρ₀(𝒫,λ,ϕ,r) * uˡᵒⁿ(𝒫,λ,ϕ,r)
# ρuˡᵃᵗ(𝒫,λ,ϕ,r) = ρ₀(𝒫,λ,ϕ,r) * uˡᵃᵗ(𝒫,λ,ϕ,r)
# ρuʳᵃᵈ(𝒫,λ,ϕ,r) = ρ₀(𝒫,λ,ϕ,r) * uʳᵃᵈ(𝒫,λ,ϕ,r)
# if total_energy
#     ρe(𝒫,λ,ϕ,r) = ρ₀(𝒫,λ,ϕ,r) * (e_int(𝒫,λ,ϕ,r) + e_kin(𝒫,λ,ϕ,r) + e_pot(𝒫,λ,ϕ,r))
# else
#     ρe(𝒫,λ,ϕ,r) = ρ₀(𝒫,λ,ϕ,r) * (e_int(𝒫,λ,ϕ,r) + e_kin(𝒫,λ,ϕ,r))
# end

# Cartesian Representation (boiler plate really)
ρ₀ᶜᵃʳᵗ(𝒫, x...)  = ρ₀(𝒫, lon(x...), lat(x...), rad(x...))
ρu⃗₀ᶜᵃʳᵗ(𝒫, x...) = (   ρuʳᵃᵈ(𝒫, lon(x...), lat(x...), rad(x...)) * r̂(x...) 
                     + ρuˡᵃᵗ(𝒫, lon(x...), lat(x...), rad(x...)) * ϕ̂(x...)
                     + ρuˡᵒⁿ(𝒫, lon(x...), lat(x...), rad(x...)) * λ̂(x...) ) 
ρeᶜᵃʳᵗ(𝒫, x...) = ρe(𝒫, lon(x...), lat(x...), rad(x...))

########
# Set up model physics
########
FT = Float64

ref_state = DryReferenceState(DecayingTemperatureProfile{FT}(param_set, FT(290), FT(220), FT(8e3)))

# # Non-total energy
# eos     = DryEuler(γ = 1 / (1 - parameters.κ))
# physics = Physics(
#     orientation = SphericalOrientation(),
#     ref_state   = ref_state,
#     eos         = eos,
#     lhs         = (
#         ESDGNonLinearAdvection(eos = eos),
#         PressureDivergence(eos = eos),
#     ),
#     sources     = sources = (
#         DeepShellCoriolis{FT}(Ω = parameters.Ω),
#         ThinShellGravityFromPotential(),
#         TotalEnergyGravityFromPotential(),
#     ),

# )
# linear_eos = linearize(physics.eos)
# linear_physics = Physics(
#     orientation = physics.orientation,
#     ref_state   = physics.ref_state,
#     eos         = linear_eos,
#     lhs         = (
#         ESDGLinearAdvection(),
#         PressureDivergence(eos = linear_eos),
#     ),
#     sources     = (
#         ThinShellGravityFromPotential(),
#         TotalEnergyGravityFromPotential(),
#     ),
# )

#####
# Held-Suarez Forcing
#####
struct HeldSuarezForcing{S} <: AbstractPhysicsComponent
    parameters::S
end

FT = Float64
day = 86400
held_suarez_parameters = (;
    k_a = FT(1 / (40 * day)),
    k_f = FT(1 / day),
    k_s = FT(1 / (4 * day)),
    ΔT_y = FT(60),
    Δθ_z = FT(10),
    T_equator = FT(315),
    T_min = FT(200),
    σ_b = FT(7 / 10),
)

######
# Modified Held-Suarez Forcing
######
function source!(
    m::DryAtmosModel,
    hsf::HeldSuarezForcing,
    source,
    state,
    aux,
)
    FT = eltype(state)
    
    _R_d  = m.parameters.R_d
    _day  = m.parameters.day
    _grav = m.parameters.grav
    _cp_d = m.parameters.cp_d
    _cv_d = m.parameters.cv_d
    _p0   = m.parameters.MSLP  

    # Parameters
    T_ref = FT(255)

    # Extract the state
    ρ = state.ρ
    ρu = state.ρu
    ρe = state.ρe
    Φ = aux.Φ

    coord = aux.coord
    
    p = pressure(ρ, ρu, ρe, Φ)
    T = p / (ρ * _R_d)

    # Held-Suarez parameters
    k_a  = hsf.parameters.k_a
    k_f  = hsf.parameters.k_f
    k_s  = hsf.parameters.k_s
    ΔT_y = hsf.parameters.ΔT_y
    Δθ_z = hsf.parameters.Δθ_z
    T_equator = hsf.parameters.T_equator
    T_min = hsf.parameters.T_min
    σ_b = hsf.parameters.σ_b

    # Held-Suarez forcing
    φ = @inbounds asin(coord[3] / norm(coord, 2))

    #TODO: replace _p0 with dynamic surfce pressure in Δσ calculations to account
    #for topography, but leave unchanged for calculations of σ involved in T_equil
    σ = p / _p0
    exner_p = σ^(_R_d / _cp_d)
    Δσ = (σ - σ_b) / (1 - σ_b)
    height_factor = max(0, Δσ)
    T_equil = (T_equator - ΔT_y * sin(φ)^2 - Δθ_z * log(σ) * cos(φ)^2) * exner_p
    T_equil = max(T_min, T_equil)
    k_T = k_a + (k_s - k_a) * height_factor * cos(φ)^4
    k_v = k_f * height_factor

    # horizontal projection
    k = coord / norm(coord)
    P = I - k * k'

    # Apply Held-Suarez forcing
    source.ρu -= k_v * P * ρu
    source.ρe -= k_T * ρ * _cv_d * (T - T_equil)
    return nothing
end

# total energy
eos     = TotalEnergy(γ = 1 / (1 - parameters.κ))
physics = Physics(
    orientation = SphericalOrientation(),
    ref_state   = ref_state,
    eos         = eos,
    lhs         = (
        ESDGNonLinearAdvection(eos = eos),
        PressureDivergence(eos = eos),
    ),
    sources     = sources = (
        DeepShellCoriolis{FT}(Ω = parameters.Ω),
        HeldSuarezForcing(held_suarez_parameters),
    ),
)
linear_eos = linearize(physics.eos)
linear_physics = Physics(
    orientation = physics.orientation,
    ref_state   = physics.ref_state,
    eos         = linear_eos,
    lhs         = (
        ESDGLinearAdvection(),
        PressureDivergence(eos = linear_eos),
    ),
    sources     = (
        ThinShellGravityFromPotential(),
    ),
)

########
# Set up model
########
model = DryAtmosModel(
    physics = physics,
    boundary_conditions = (5, 6),
    initial_conditions = (ρ = ρ₀ᶜᵃʳᵗ, ρu = ρu⃗₀ᶜᵃʳᵗ, ρe = ρeᶜᵃʳᵗ),
    numerics = (
        flux = RusanovNumericalFlux(),
    ),
    parameters = parameters,
)

linear_model = DryAtmosLinearModel(
    physics = linear_physics,
    boundary_conditions = model.boundary_conditions,
    initial_conditions = nothing,
    numerics = (
        flux = model.numerics.flux,
        direction = VerticalDirection()
    ),
    parameters = model.parameters,
)

########
# Set up time steppers (could be done automatically in simulation)
########
# determine the time step construction
# element_size = (domain_height / numelem_vert)
# acoustic_speed = soundspeed_air(param_set, FT(330))
dx = min_node_distance(grid.numerical)
cfl = 3
Δt = cfl * dx / 330.0
start_time = 0
end_time = 30 * 24 * 3600
method = ARK2GiraldoKellyConstantinescu
callbacks = (
  Info(), 
  CFL(), 
  VTKState(
    iteration = Int(floor(4*3600/Δt)), 
    filepath = "/central/scratch/bischtob/benchmark_baroclinic_wave"),
  )

########
# Set up simulation
########
simulation = Simulation(
    (model, linear_model,);
    grid = grid,
    timestepper = (method = method, timestep = Δt),
    time        = (start = start_time, finish = end_time),
    callbacks   = callbacks,
)

########
# Run the simulation
########
# initialize!(simulation)
# evolve!(simulation)

nothing