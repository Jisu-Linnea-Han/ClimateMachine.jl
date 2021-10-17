using Distributions
using Random
using StaticArrays
using Test
using UnPack
using LinearAlgebra
using ArgParse
using Printf

using ClimateMachine
using ClimateMachine.Atmos # add include("AtmosCanopyModel.jl")
using ClimateMachine.Orientations
using ClimateMachine.ConfigTypes
using ClimateMachine.Diagnostics
using ClimateMachine.GenericCallbacks
using ClimateMachine.Mesh.Grids
using ClimateMachine.Mesh.Filters
using Thermodynamics.TemperatureProfiles # ClimateMachine.TemperatureProfiles is deprecated!
#using ClimateMachine.SystemSolvers
using ClimateMachine.ODESolvers
#using Thermodynamics, instead use
using ClimateMachine.Thermodynamics
using ClimateMachine.Thermodynamics: relative_humidity, air_density, internal_energy, total_energy, PhaseDry, PhaseEquil_pθq
using ClimateMachine.TurbulenceClosures
using ClimateMachine.TurbulenceConvection
using ClimateMachine.VariableTemplates
#import ClimateMachine.Atmos: atmos_source!
using ClimateMachine.Writers

using CLIMAParameters
using CLIMAParameters.Planet: MSLP, R_d, day, grav, cp_d, cv_d, planet_radius, LH_v0
using CLIMAParameters.Atmos.SubgridScale: C_smag

using ClimateMachine.BalanceLaws
import ClimateMachine.BalanceLaws: source, prognostic_vars
using ClimateMachine.Atmos: altitude

struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet()
const n_tracer = Int(1)

struct CanopyAerodynamics{FT} <: TendencyDef{Source} # <: AbstractTendencyType, under BalanceLaw.
    "drag coefficient"
    c_d::FT
    "canopy height [m]"
    h_c::FT
    "Leaf Area Index LAI"
    LAI::FT
    #"Leaf Area Density LAD"
    #LAD::FT
    Δz::FT
end
prognostic_vars(::CanopyAerodynamics) = (Momentum(), SGSTKE(),)

function source(::Momentum, c::CanopyAerodynamics, m ,args)
    # currently only regular meshing is possible.
    # args include below:
    #_args = (; state, aux, t, direction, diffusive)
    #args = merge(_args, (precomputed = precompute(bl, _args, tend),))

    @unpack state, aux = args
    @unpack c_d, h_c, LAI = c
    LAD = LAI/h_c

    z = altitude(m, aux)
    if z <= h_c
        ρu = state.ρu
        ρ = state.ρ
        u_mag = norm(ρu/ρ)
        # see eq (9-10) in Patton et al 2016
        return -c_d * LAD * ρu * u_mag #SVector{3, FT}
    else
        FT = eltype(state)
        return SVector{3, FT}(0,0,0)
    end
end

function source(::SGSTKE, c::CanopyAerodynamics, m ,args)
    # currently only regular meshing is possible.
    @unpack state, aux = args
    @unpack c_d, h_c, LAI = c

    #turbulence_model(m)
    LAD = LAI/h_c
    FT = eltype(state)
    z = altitude(m, aux)
    if z <= h_c
        ρu = state.ρu
        ρ = state.ρ
        u_mag = norm(ρu/ρ)
        ρe = state.sgstke.ρe_SGS
        # see eq (9-10) in Patton et al 2016
        return -FT(8/3) * c_d * LAD * ρe * u_mag
    else
        return FT(0)
    end
end

struct CanopyAtmoInteraction{FT} <: TendencyDef{Source} # <: AbstractTendencyType, under BalanceLaw.
    "canopy height [m]"
    h_c::FT
    "Leaf Area Index LAI"
    LAI::FT
    #"Leaf Area Density LAD"
    #LAD::FT
    Δz::FT
    "vegetation canopy conductance"
    G_veg::FT #[m/s]
    "Gross Primary Production"
    GPP::FT
end

prognostic_vars(::CanopyAtmoInteraction) = (TotalMoisture(), Tracers{n_tracer}(),)
# Later it should include energy exchange due to Latent & sensible heat exchange

function source(::TotalMoisture, c::CanopyAtmoInteraction, m, args)
    @unpack state, aux = args
    @unpack h_c, LAI, Δz, G_veg = c
    @unpack ts = args.precomputed
    #@unpcak = node

    # incomplete! need to account for the gas moisture content...
    # First assuming T_l = T_air
    LAD = LAI/h_c
    FT = eltype(state)
    z = altitude(m, aux)
    if z <= h_c
        VPD = (FT(1) - relative_humidity(ts))*air_density(ts) # in density unit [kg/m3]
        return G_veg*LAD*VPD
    else
        return FT(0) 
    end
end

function source(::Tracers{N}, c::CanopyAtmoInteraction, m, args) where {N}
    @unpack state, aux = args
    @unpack h_c, LAI, Δz, GPP = c
    #SPAC_Flux = spac_model(m)
    #node = SPAC_Flux.node
    #flux = SPAC_Flux.flux
    ##@unpcak = node 
    LAD = LAI / h_c
    z = altitude(m, aux)
    FT = eltype(state)
    if z <= h_c
        return SVector{N, FT}(- GPP * LAD)
    else
        return SVector{N, FT}(0)
    end
end

function init_canopy_dry!(problem, bl, state, aux, localgeo, time)
    # here the first argument, `problem` is AtmosProblem, plugged in `init_state_prognostic`
    # ClimateMachine.Atmos.atmos_problem_init_state_auxiliary
    # orientation = FlatOrientation()
    FT = eltype(state)
    param_set = parameter_set(bl)
    (x, y, z) = localgeo.coord

    # These constants are those used by Stevens et al. (2005)
    #PhaseDry(param_set, e_int, state.e_int=19284.033384, ρ=1.160988, q_tot=1.000000, T = 404.512170, maxiter=3, tol=71.75060)
    # Patton et al 2015
    #qref = FT(0.0)  #FT(9.0e-3)
    
    R_gas = FT(R_d(param_set)) # for dry air
    c_v::FT = cv_d(param_set)
    c_p::FT = cp_d(param_set)
    p0::FT = MSLP(param_set)
    _grav = FT(grav(param_set))
    _LH_v0 = LH_v0(param_set) # Latent heat of vaporization

    # Use this if wish to follow the reference profile stated in the AtmosModel
    #ref_state = reference_state(bl)
    #θ, p = ref_state.virtual_temperature_profile(param_set, z)

    # Below is artifical profile, following Patton et al. 2016
    # Potential Temperature profile θ=300K for z<2h_c, and apply lapse rate of 3K/km.
    Γ = FT(3e-3) # [K/m], artificial lapse rate 
    T_surf = FT(300) # This should be removed later
    h_c = FT(20) # Look up z_canopy in the main()
    if z <= 2*h_c
        θ = T_surf
    else
        θ = T_surf - Γ*(z - 2*h_c)
    end

    # initial velocity: 2-D geostrophic flow
    # MU case in Patton et al 2015
    ugeo = FT(5)
    vgeo = FT(0)
    u, v, w = ugeo, vgeo, FT(0)

    # Density, Temperature, ρ
    # Moisture: DryModel
    π_exner = FT(1) - _grav / (c_p * θ) * z
    ρ = p0 / (R_gas * θ) * (π_exner)^(c_v / R_gas)
    T = θ * π_exner
    e_int = internal_energy(param_set, T)
    ts = PhaseDry(param_set, e_int, ρ)
    # Below are common
    e_kin = FT(0.5) * FT((u^2 + v^2 + w^2))
    e_pot = gravitational_potential(bl.orientation, aux) 
    ρe_tot = ρ * total_energy(e_kin, e_pot, ts)

    state.ρ = ρ
    state.ρu = SVector(ρ * u, ρ * v, ρ * w)
    state.energy.ρe = ρe_tot
    if !(sgstke_model(bl) isa NoSGStke)
        state.sgstke.ρe_SGS = ( FT(0.2) + FT(rand(Uniform(-0.01, 0.01))) ) * e_kin
    end
    if !(tracer_model(bl) isa NoTracers)
        # tracer we would like to track is CO2--in ppm unit.
        # intial field of 340 ppm
        ρχ = SVector{n_tracer, FT}(FT(3.4e-4)*ρ)
        state.tracers.ρχ = ρχ
    end

    # initialize spac? do we need this?
    return nothing
end

function init_canopy_moist!(problem, bl, state, aux, localgeo, time)
    # here the first argument, `problem` is AtmosProblem, plugged in `init_state_prognostic`
    # ClimateMachine.Atmos.atmos_problem_init_state_auxiliary
    # orientation = FlatOrientation()
    FT = eltype(state)
    param_set = parameter_set(bl)
    (x, y, z) = localgeo.coord

    # These constants are those used by Stevens et al. (2005)
    #PhaseDry(param_set, e_int, state.e_int=19284.033384, ρ=1.160988, q_tot=1.000000, T = 404.512170, maxiter=3, tol=71.75060)
    # Patton et al 2015
    #qref = FT(0.0)  #FT(9.0e-3)
    
    q_tot = FT(1e-3) # kg/kg
    q_liq = FT(0)   # Assuming only water vapor state
    q_ice = FT(0)
    q_pt_sfc = PhasePartition(q_tot)
    Rm_sfc = gas_constant_air(param_set, q_pt_sfc)
    c_v::FT = cv_d(param_set)
    c_p::FT = cp_d(param_set)
    p0::FT = MSLP(param_set)
    _grav = FT(grav(param_set))
    _LH_v0 = LH_v0(param_set) # Latent heat of vaporization

    # Use this if wish to follow the reference profile stated in the AtmosModel
    #ref_state = reference_state(bl)
    #θ, p = ref_state.virtual_temperature_profile(param_set, z)

    # Below is artifical profile, following Patton et al. 2016
    # Potential Temperature profile θ=300K for z<2h_c, and apply lapse rate of 3K/km.
    Γ = FT(3e-3) # [K/m], artificial lapse rate 
    T_surf = 300 # This should be removed later
    h_c = FT(20) # Look up z_canopy in the main()
    if z <= 2*h_c
        θ = T_surf
    else
        θ = T_surf - Γ*(z - 2*h_c)
    end

    # initial velocity: 2-D geostrophic flow
    # MU case in Patton et al 2015
    ugeo = FT(5)
    vgeo = FT(0)
    u, v, w = ugeo, vgeo, FT(0)

    # Density, Temperature, ρ
    # Moisture: EquilMoist
    H = Rm_sfc * T_surf / _grav
    p = p0 * exp(-z / H) # hydrostatic ressure
    θ_liq_ice = θ - (_LH_v0/c_p)*q_liq  # liquid-ice potential temperature, θ_liq = θ for q_liq = 0
    ts = PhaseEquil_pθq(param_set, p,θ_liq_ice, q_tot) 
    ρ = air_density(ts)
    # Below are common
    e_kin = FT(0.5) * FT((u^2 + v^2 + w^2))
    e_pot = gravitational_potential(bl.orientation, aux) 
    ρe_tot = ρ * total_energy(e_kin, e_pot, ts)

    state.ρ = ρ
    state.ρu = SVector(ρ * u, ρ * v, ρ * w)
    state.energy.ρe = ρe_tot
    # activate only for non-dry model!
    state.moisture.ρq_tot = ρ * q_tot
    if !(sgstke_model(bl) isa NoSGStke)
        state.sgstke.ρe_SGS =  ( FT(0.2) + FT(rand(Uniform(-0.01, 0.01))) ) * e_kin
    end
    if !(tracer_model(bl) isa NoTracers) 
        # tracer we would like to track is CO2--in ppm unit.
        # intial field of 340 ppm
        ρχ = SVector{n_tracer, FT}(FT(3.4e-4)*ρ)
        state.tracers.ρχ = ρχ
    end

    # SPAC will be initialized separately
    return nothing
end

ClimateMachine.init()

function canopy_model(
    ::Type{FT},
    c_aero::CanopyAerodynamics{FT},
    c_atmo::CanopyAtmoInteraction{FT};
    config_type = AtmosLESConfigType,
    turbconv = NoTurbConv(),
    moisture_model = "dry", #"dry", "equilibrium"
    sgstke = NoSGStke(),
    ) where {FT}
    #@unpack  = node;

    if n_tracer > 0
        δ_χ = @SVector [FT(1.6e-5) for ii in 1:n_tracer] # Diffusion coefficient for CO2
        tracers = NTracers{n_tracer, FT}(δ_χ)
    else
        tracers = NoTracers()
    end

    T_surf = FT(300)  # from Patton et al 2016
    T_min_ref = FT(0)
    #T_virt = FT(300)
    temp_profile_ref = DryAdiabaticProfile{FT}(param_set, T_surf, T_min_ref) #or use IsothermalProfile(param_set, T_virt)
    # Note that IsothermalProfile = DecayingTemperatureProfile{FT}(param_set, T_virt, T_virt)
    ref_state = HydrostaticState(temp_profile_ref)

    C_drag = FT(0.0011)
    #_C_smag = FT(C_smag(param_set))
    LHF = FT(50) # Latent heat flux
    SHF = FT(15) # Sensible heat flux
    moisture_flux = LHF / FT(LH_v0(param_set))

    # Assemble source components
    source_sgstke = (ShearProduction(), BuoyancyProduction(), Dissipation(),)
    source_default = (Gravity(), c_aero, source_sgstke...)
    if moisture_model == "dry"
        source = source_default
        moisture = DryModel()
        moisture_bcs = ()
        init_state_prognostic = init_canopy_dry!
        #tracer_bcs = ()
    elseif moisture_model == "equilibrium"
        source = (source_default..., c_atmo)
        moisture = EquilMoist(; maxiter = 5, tolerance = FT(0.1))
        moisture_bcs = (; moisture = PrescribedMoistureFlux((state, aux, t) -> moisture_flux)) # No-flux from the ground, modify `LHF`
        init_state_prognostic = init_canopy_moist!
        #tracer_bcs = (tracer = ImpermeableTracer())
    #elseif moisture_model == "nonequilibrium"
    #    source = (source_default..., CreateClouds())
    #    moisture = NonEquilMoist()
    else
        @warn @sprintf(
            """
%s: unrecognized moisture_model in source terms, using the defaults""",
            moisture_model,
        )
        source = source_default
        moisture_bcs = ()
        #tracer_bcs = ()
    end
    
    physics = AtmosPhysics{FT}(
        param_set;
        ref_state = ref_state,
        turbulence = Deardorff(), #SmagorinskyLilly(_C_smag),
        moisture = moisture,
        tracers = tracers,
        turbconv = turbconv,
        sgstke = sgstke,
    )

    problem = AtmosProblem(
        boundaryconditions = (
            AtmosBC(
                physics;
                momentum = Impenetrable(DragLaw(
                    (state, aux, t, normPu) -> C_drag,
                )),#default is Impenetrable{FreeSlip}
                ## Latent + Sensible heat flux from the ground?
                energy = PrescribedEnergyFlux((state, aux, t) -> LHF + SHF),
                moisture_bcs...,
                #tracer_bcs...,
                turbconv = turbconv_bcs(turbconv)[1],
                #sgstke = NoSGStkeFlux(),
            ),
            AtmosBC(physics; turbconv = turbconv_bcs(turbconv)[2]),          #it's not wrong, dont' worry!
            # default setting of AtmosBC: { Insulating, ClimateMachine.TurbulenceConvection.NoTurbConvBC}
        ),
        init_state_prognostic = init_state_prognostic, # you can either put it here or in `model`
        # ClimateMachine.Atmos.atmos_problem_init_state_auxiliary
    )

    #struct AtmosModel{FT, PH, PR, O, S, DC} <: BalanceLaw
    model = AtmosModel{FT}(
        config_type,                            # Flow in a box, requires the AtmosLESConfigType
        physics;       
        problem = problem,
        source = source,
        # default settings: hyperdiffusion = NoHyperDiffusion(), radiation = NoRadiation(), precipitation = NoPrecipitation(), moisture = EquilMoist{Float64}(3, 0.1)
        # we can set hyperdiffusion by (τ_hyper = FT(4 * 3600); # hyperdiffusion time scale hyperdiffusion_model = DryBiharmonic(FT(4 * 3600));)
    )

    return model
end

function main()
    # add a command line argument to specify the kind of surface flux
    # TODO: this will move to the future namelist functionality
    canopy_args = ArgParseSettings(autofix_names = true)
    add_arg_group!(canopy_args, "ATMO-LAND")
    @add_arg_table! canopy_args begin
        "--surface-flux"
        help = "specify surface flux for energy and moisture"
        metavar = "prescribed|bulk"
        arg_type = String
        default = "prescribed"
    end

    cl_args =
        ClimateMachine.init(parse_clargs = true, custom_clargs = canopy_args)
    ClimateMachine.init(parse_clargs = true) # every 5 simulation minutes
    surface_flux = cl_args["surface_flux"]
    
    FT = Float64

    # DG polynomial order
    N = 4

    # Domain resolution and size
    Δh = FT(10)
    Δv = FT(5)
    resolution = (Δh, Δh, Δv)

    xmax = FT(100)
    ymax = FT(100)
    zmax = FT(100)

    t0 = FT(0)
    timeend = FT(600)
    CFLmax = FT(0.1)     # use this for single-rate explicit LSRK144

    # constants for struct Canopy
    c_d = FT(0.2)
    LAI = FT(2.0)       # Later improve this to LAD profile
    Δz = resolution[3] # This approach only works for uniform gridding
    G_veg = FT(0.2)
    h_c = FT(20)
    GPP = FT(1)

    ode_solver_type = ClimateMachine.ExplicitSolverType(
        solver_method = LSRK144NiegemannDiehlBusch,)
    
    model = canopy_model(
        FT, 
        CanopyAerodynamics(c_d, h_c, LAI, Δz), 
        CanopyAtmoInteraction(h_c, LAI, Δz, G_veg, GPP);
	moisture_model = "equilibrium",
	sgstke = SGStkeModel(),
        )
    ics = model.problem.init_state_prognostic


    # need to tweak the configuration to add land module flux simulation
    driver_config = ClimateMachine.AtmosLESConfiguration(
        "CanopyFlow",
        N,
        resolution,
        xmax,
        ymax,
        zmax,
        param_set,
        ics,
        model = model,
    )
    solver_config = ClimateMachine.SolverConfiguration(
        t0,
        timeend,
        driver_config,
        ode_solver_type = ode_solver_type,
        init_on_cpu = true,
        Courant_number = CFLmax,
    )
    
    # The `user_callbacks` are passed to the ODE solver as callback functions;
    # see [`solve!`](@ref ODESolvers.solve!).

    # GenericCallbacks.init!(callbacks, solver, Q, param, t0)
    #=
    cbtmarfilter = GenericCallbacks.EveryXSimulationSteps(1) do
        Filters.apply!(
            solver_config.Q,
            ("moisture.ρq_tot", "tracers.ρχ",),
            solver_config.dg.grid,
            TMARFilter(),
        )
        nothing
    end
    =#
    check_cons = (
        ClimateMachine.ConservationCheck("ρ", "1smins", FT(0.0001)),
        ClimateMachine.ConservationCheck("energy.ρe", "1smins", FT(0.0025)),
    )

    result = ClimateMachine.invoke!(
        solver_config;
        check_cons = check_cons,
        #user_callbacks = (cbtmarfilter, ),
        check_euclidean_distance = true,
    )

end

main()
