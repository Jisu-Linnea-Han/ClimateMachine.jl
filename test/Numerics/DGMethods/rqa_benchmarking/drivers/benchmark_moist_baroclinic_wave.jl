#!/usr/bin/env julia --project
include("../interface/utilities/boilerplate.jl")
include("../interface/numerics/timestepper_abstractions.jl")

########
# Set up parameters
########
parameters = (
    a        = get_planet_parameter(:planet_radius),
    Ω        = get_planet_parameter(:Omega),
    g        = get_planet_parameter(:grav),
    κ        = get_planet_parameter(:kappa_d),
    R_d      = get_planet_parameter(:R_d),
    R_v      = get_planet_parameter(:R_v),
    cp_d     = get_planet_parameter(:cp_d),
    cp_v     = get_planet_parameter(:cp_v),
    cp_l     = get_planet_parameter(:cp_l),
    cp_i     = get_planet_parameter(:cp_i),
    cv_d     = get_planet_parameter(:cv_d),
    cv_v     = get_planet_parameter(:cv_v),
    cv_l     = get_planet_parameter(:cv_l),
    cv_i     = get_planet_parameter(:cv_i),
    γ        = get_planet_parameter(:cp_d)/get_planet_parameter(:cv_d),
    molmass_ratio = get_planet_parameter(:molmass_dryair)/get_planet_parameter(:molmass_water),
    pₒ       = get_planet_parameter(:MSLP),
    pₜᵣ      = get_planet_parameter(:press_triple),
    Tₜᵣ      = get_planet_parameter(:T_triple),
    T_0      = get_planet_parameter(:T_0),
    LH_v0    = get_planet_parameter(:LH_v0),
    e_int_v0 = get_planet_parameter(:e_int_v0),
    e_int_i0 = get_planet_parameter(:e_int_i0),
    H        = 30e3,
    k        = 3.0,
    Γ        = 0.005,
    T_E      = 310.0,
    T_P      = 240.0,
    b        = 2.0,
    z_t      = 15e3,
    λ_c      = π / 9,
    ϕ_c      = 2 * π / 9,
    V_p      = 1.0,
    ϕ_w      = 2*π/9,
    p_w      = 3.4e4,
    q₀       = 0.018,
    qₜ       = 1e-12,
    τ_precip = 28.409,
    Mᵥ       = 0.608,
)

########
# Set up domain
########
domain = SphericalShell(
    radius = parameters.a,
    height = parameters.H,
)
grid = DiscretizedDomain(
    domain;
    elements = (vertical = 8, horizontal = 16),
    polynomial_order = (vertical = 2, horizontal = 2),
    overintegration_order = (vertical = 0, horizontal = 0),
   )

########
# Set up inital condition
########
# additional initial condition parameters
T₀(𝒫)   = 0.5 * (𝒫.T_E + 𝒫.T_P)
A(𝒫)    = 1.0 / 𝒫.Γ
B(𝒫)    = (T₀(𝒫) - 𝒫.T_P) / T₀(𝒫) / 𝒫.T_P
C(𝒫)    = 0.5 * (𝒫.k + 2) * (𝒫.T_E - 𝒫.T_P) / 𝒫.T_E / 𝒫.T_P
H(𝒫)    = 𝒫.R_d * T₀(𝒫) / 𝒫.g
d_0(𝒫)  = 𝒫.a / 6

# convenience functions that only depend on height
τ_z_1(𝒫,r)   = exp(𝒫.Γ * (r - 𝒫.a) / T₀(𝒫))
τ_z_2(𝒫,r)   = 1 - 2 * ((r - 𝒫.a) / 𝒫.b / H(𝒫))^2
τ_z_3(𝒫,r)   = exp(-((r - 𝒫.a) / 𝒫.b / H(𝒫))^2)
τ_1(𝒫,r)     = 1 / T₀(𝒫) * τ_z_1(𝒫,r) + B(𝒫) * τ_z_2(𝒫,r) * τ_z_3(𝒫,r)
τ_2(𝒫,r)     = C(𝒫) * τ_z_2(𝒫,r) * τ_z_3(𝒫,r)
τ_int_1(𝒫,r) = A(𝒫) * (τ_z_1(𝒫,r) - 1) + B(𝒫) * (r - 𝒫.a) * τ_z_3(𝒫,r)
τ_int_2(𝒫,r) = C(𝒫) * (r - 𝒫.a) * τ_z_3(𝒫,r)
F_z(𝒫,r)     = (1 - 3 * ((r - 𝒫.a) / 𝒫.z_t)^2 + 2 * ((r - 𝒫.a) / 𝒫.z_t)^3) * ((r - 𝒫.a) ≤ 𝒫.z_t)

# convenience functions that only depend on longitude and latitude
d(𝒫,λ,ϕ)     = 𝒫.a * acos(sin(ϕ) * sin(𝒫.ϕ_c) + cos(ϕ) * cos(𝒫.ϕ_c) * cos(λ - 𝒫.λ_c))
c3(𝒫,λ,ϕ)    = cos(π * d(𝒫,λ,ϕ) / 2 / d_0(𝒫))^3
s1(𝒫,λ,ϕ)    = sin(π * d(𝒫,λ,ϕ) / 2 / d_0(𝒫))
cond(𝒫,λ,ϕ)  = (0 < d(𝒫,λ,ϕ) < d_0(𝒫)) * (d(𝒫,λ,ϕ) != 𝒫.a * π)

# base-state thermodynamic variables
I_T(𝒫,ϕ,r)   = (cos(ϕ) * r / 𝒫.a)^𝒫.k - 𝒫.k / (𝒫.k + 2) * (cos(ϕ) * r / 𝒫.a)^(𝒫.k + 2)
Tᵥ(𝒫,ϕ,r)    = (τ_1(𝒫,r) - τ_2(𝒫,r) * I_T(𝒫,ϕ,r))^(-1) * (𝒫.a/r)^2
p(𝒫,ϕ,r)     = 𝒫.pₒ * exp(-𝒫.g / 𝒫.R_d * (τ_int_1(𝒫,r) - τ_int_2(𝒫,r) * I_T(𝒫,ϕ,r)))
q(𝒫,ϕ,r)     = (p(𝒫,ϕ,r) > 𝒫.p_w) ? 𝒫.q₀ * exp(-(ϕ / 𝒫.ϕ_w)^4) * exp(-((p(𝒫,ϕ,r) - 𝒫.pₒ) / 𝒫.p_w)^2) : 𝒫.qₜ

# base-state velocity variables
U(𝒫,ϕ,r)  = 𝒫.g * 𝒫.k / 𝒫.a * τ_int_2(𝒫,r) * Tᵥ(𝒫,ϕ,r) * ((cos(ϕ) * r / 𝒫.a)^(𝒫.k - 1) - (cos(ϕ) * r / 𝒫.a)^(𝒫.k + 1))
u(𝒫,ϕ,r)  = -𝒫.Ω * r * cos(ϕ) + sqrt((𝒫.Ω * r * cos(ϕ))^2 + r * cos(ϕ) * U(𝒫,ϕ,r))
v(𝒫,ϕ,r)  = 0.0
w(𝒫,ϕ,r)  = 0.0

# velocity perturbations
δu(𝒫,λ,ϕ,r)  = -16 * 𝒫.V_p / 3 / sqrt(3) * F_z(𝒫,r) * c3(𝒫,λ,ϕ) * s1(𝒫,λ,ϕ) * (-sin(𝒫.ϕ_c) * cos(ϕ) + cos(𝒫.ϕ_c) * sin(ϕ) * cos(λ - 𝒫.λ_c)) / sin(d(𝒫,λ,ϕ) / 𝒫.a) * cond(𝒫,λ,ϕ)
δv(𝒫,λ,ϕ,r)  = 16 * 𝒫.V_p / 3 / sqrt(3) * F_z(𝒫,r) * c3(𝒫,λ,ϕ) * s1(𝒫,λ,ϕ) * cos(𝒫.ϕ_c) * sin(λ - 𝒫.λ_c) / sin(d(𝒫,λ,ϕ) / 𝒫.a) * cond(𝒫,λ,ϕ)
δw(𝒫,λ,ϕ,r)  = 0.0

# CliMA prognostic variables
# compute the total energy
uˡᵒⁿ(𝒫,λ,ϕ,r)   = u(𝒫,ϕ,r) + δu(𝒫,λ,ϕ,r)
uˡᵃᵗ(𝒫,λ,ϕ,r)   = v(𝒫,ϕ,r) + δv(𝒫,λ,ϕ,r)
uʳᵃᵈ(𝒫,λ,ϕ,r)   = w(𝒫,ϕ,r) + δw(𝒫,λ,ϕ,r)

# cv_m and R_m for moist experiment
cv_m(𝒫,ϕ,r)  = 𝒫.cv_d + (𝒫.cv_v - 𝒫.cv_d) * q(𝒫,ϕ,r)
R_m(𝒫,ϕ,r) = 𝒫.R_d * (1 + (𝒫.molmass_ratio - 1) * q(𝒫,ϕ,r))

T(𝒫,ϕ,r) = Tᵥ(𝒫,ϕ,r) / (1 + 𝒫.Mᵥ * q(𝒫,ϕ,r)) 
e_int(𝒫,λ,ϕ,r)  = cv_m(𝒫,ϕ,r) * (T(𝒫,ϕ,r) - 𝒫.T_0) + q(𝒫,ϕ,r) * 𝒫.e_int_v0
e_kin(𝒫,λ,ϕ,r)  = 0.5 * ( uˡᵒⁿ(𝒫,λ,ϕ,r)^2 + uˡᵃᵗ(𝒫,λ,ϕ,r)^2 + uʳᵃᵈ(𝒫,λ,ϕ,r)^2 )
e_pot(𝒫,λ,ϕ,r)  = 𝒫.g * r

ρ₀(𝒫,λ,ϕ,r)    = p(𝒫,ϕ,r) / R_m(𝒫,ϕ,r) / T(𝒫,ϕ,r)
ρuˡᵒⁿ(𝒫,λ,ϕ,r) = ρ₀(𝒫,λ,ϕ,r) * uˡᵒⁿ(𝒫,λ,ϕ,r)
ρuˡᵃᵗ(𝒫,λ,ϕ,r) = ρ₀(𝒫,λ,ϕ,r) * uˡᵃᵗ(𝒫,λ,ϕ,r)
ρuʳᵃᵈ(𝒫,λ,ϕ,r) = ρ₀(𝒫,λ,ϕ,r) * uʳᵃᵈ(𝒫,λ,ϕ,r)
ρe(𝒫,λ,ϕ,r) = ρ₀(𝒫,λ,ϕ,r) * (e_int(𝒫,λ,ϕ,r) + e_kin(𝒫,λ,ϕ,r) + e_pot(𝒫,λ,ϕ,r))
ρq(𝒫,λ,ϕ,r) = ρ₀(𝒫,λ,ϕ,r) * q(𝒫,ϕ,r)

# Cartesian Representation (boiler plate really)
ρ₀ᶜᵃʳᵗ(𝒫, x...)  = ρ₀(𝒫, lon(x...), lat(x...), rad(x...))
ρu⃗₀ᶜᵃʳᵗ(𝒫, x...) = (   ρuʳᵃᵈ(𝒫, lon(x...), lat(x...), rad(x...)) * r̂(x...)
                     + ρuˡᵃᵗ(𝒫, lon(x...), lat(x...), rad(x...)) * ϕ̂(x...)
                     + ρuˡᵒⁿ(𝒫, lon(x...), lat(x...), rad(x...)) * λ̂(x...) )
ρeᶜᵃʳᵗ(𝒫, x...) = ρe(𝒫, lon(x...), lat(x...), rad(x...))
ρqᶜᵃʳᵗ(𝒫, x...) = ρq(𝒫, lon(x...), lat(x...), rad(x...))

########
# Set up model physics
########
FT = Float64

ref_state = DryReferenceState(
  DecayingTemperatureProfile{FT}(parameters, FT(290), FT(220), FT(8e3))
)
physics = Physics(
    orientation = SphericalOrientation(),
    ref_state   = ref_state,
    eos         = MoistIdealGas(),
    lhs         = (
        NonlinearAdvection{(:ρ, :ρu, :ρe)}(),
        PressureDivergence(),
    ),
    sources     = (
        DeepShellCoriolis(),
        FluctuationGravity(),
        ZeroMomentMicrophysics(),
    ),
    parameters = parameters,
)

linear_physics = Physics(
    orientation = physics.orientation,
    ref_state   = physics.ref_state,
    eos         = physics.eos,
    lhs         = (
        LinearAdvection{(:ρ, :ρu, :ρe)}(),
        LinearPressureDivergence(),
    ),
    sources     = (
        FluctuationGravity(),
    ),
    parameters = parameters,
)

########
# Set up model
########
model = DryAtmosModel(
    physics = physics,
    boundary_conditions = (DefaultBC(), DefaultBC()),
    initial_conditions = (ρ = ρ₀ᶜᵃʳᵗ, ρu = ρu⃗₀ᶜᵃʳᵗ, ρe = ρeᶜᵃʳᵗ, ρq = ρqᶜᵃʳᵗ),
    numerics = (
      flux = LMARSNumericalFlux(),
    ),
)

linear_model = DryAtmosModel(
    physics = linear_physics,
    boundary_conditions = (DefaultBC(), DefaultBC()),
    initial_conditions = (ρ = ρ₀ᶜᵃʳᵗ, ρu = ρu⃗₀ᶜᵃʳᵗ, ρe = ρeᶜᵃʳᵗ, ρq = ρqᶜᵃʳᵗ),
    numerics = (
        flux = RefanovFlux(),
    ),

)

########
# Set up time steppers (could be done automatically in simulation)
########
dx = min_node_distance(grid.numerical)
cfl = 5 # 13 for 10 days, 7.5 for 200+ days
Δt = cfl * dx / 330.0
start_time = 0
end_time = 30 * 24 * 3600
method = IMEX() 
callbacks = (
  Info(),
  CFL(),
  VTKState(
    iteration = Int(floor(6*3600/Δt)), 
    filepath = "./out_esdg/moist_baroclinic_wave/"),
  TMARCallback(),
)

########
# Set up simulation
########
simulation = Simulation(
    (Explicit(model), Implicit(linear_model),);
    grid = grid,
    timestepper = (method = method, timestep = Δt),
    time        = (start = start_time, finish = end_time),
    callbacks   = callbacks,
);

evolve!(simulation)

nothing