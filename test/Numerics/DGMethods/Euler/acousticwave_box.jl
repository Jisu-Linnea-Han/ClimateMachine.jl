using ClimateMachine
using ClimateMachine.ConfigTypes
using ClimateMachine.Mesh.Topologies
using ClimateMachine.Mesh.Grids
using ClimateMachine.VariableTemplates
using ClimateMachine.Mesh.Filters
using ClimateMachine.DGMethods: DGModel, init_ode_state, remainder_DGModel
using ClimateMachine.DGMethods.NumericalFluxes
using ClimateMachine.Mesh.Geometry
using ClimateMachine.ODESolvers
using ClimateMachine.SystemSolvers
using ClimateMachine.VTK: writevtk, writepvtu
using ClimateMachine.GenericCallbacks:
    EveryXWallTimeSeconds, EveryXSimulationSteps
using ClimateMachine.Thermodynamics
using ClimateMachine.TemperatureProfiles: IsothermalProfile
using ClimateMachine.Atmos
using ClimateMachine.TurbulenceClosures
using ClimateMachine.Orientations
using ClimateMachine.BalanceLaws: Prognostic, Auxiliary, vars_state
    
using ClimateMachine.VariableTemplates: flattenednames

using CLIMAParameters
using CLIMAParameters.Planet: planet_radius, R_d, grav, MSLP, cp_d, cv_d
import CLIMAParameters
struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet()

CLIMAParameters.Planet.T_0(::EarthParameterSet) = 0
#CLIMAParameters.Planet.MSLP(::EarthParameterSet) = 100

using MPI, Logging, StaticArrays, LinearAlgebra, Printf, Dates, Test

const output_vtk = false

function main()
    ClimateMachine.init()
    ArrayType = ClimateMachine.array_type()

    mpicomm = MPI.COMM_WORLD

    polynomialorder = 5
    numelem_horz = 10
    numelem_vert = 10

    timeend = 100
    outputtime = 1

    FT = Float64

    split_explicit_implicit = false

    test_run(
        mpicomm,
        polynomialorder,
        numelem_horz,
        numelem_vert,
        timeend,
        outputtime,
        ArrayType,
        FT,
        split_explicit_implicit,
    )
end

function test_run(
    mpicomm,
    polynomialorder,
    numelem_horz,
    numelem_vert,
    timeend,
    outputtime,
    ArrayType,
    FT,
    split_explicit_implicit,
)
    setup = AcousticBoxSetup{FT}()

    horz_range = range(FT(0), stop = setup.domain_length, length = numelem_horz+1)
    vert_range = range(FT(0), stop = setup.domain_height, length = numelem_vert+1)
    brickrange = (horz_range, horz_range, vert_range)
   
    periodicity=(true, true, true)

    topology = StackedBrickTopology(mpicomm, brickrange, periodicity=periodicity)

    grid = DiscontinuousSpectralElementGrid(
        topology,
        FloatType = FT,
        DeviceArray = ArrayType,
        polynomialorder = polynomialorder,
    )
  
    zero_ref_state_velocity = true
    if zero_ref_state_velocity
      ref_state = AcousticBoxReferenceState(setup.T0, FT(0))
    else
      ref_state = AcousticBoxReferenceState(setup.T0, setup.u0)
    end

    model = AtmosModel{FT}(
        AtmosLESConfigType,
        param_set;
        init_state_prognostic = setup,
        orientation = NoOrientation(),
        ref_state = ref_state,
        turbulence = ConstantDynamicViscosity(FT(0)),
        moisture = DryModel(),
        source = (),
    )
    
    linearmodel = AtmosAcousticLinearModel(model)

    dg = DGModel(
        model,
        grid,
        RusanovNumericalFlux(),
        CentralNumericalFluxSecondOrder(),
        CentralNumericalFluxGradient(),
    )

    lineardg = DGModel(
        linearmodel,
        grid,
        RusanovNumericalFlux(),
        CentralNumericalFluxSecondOrder(),
        CentralNumericalFluxGradient();
        direction = VerticalDirection(),
        state_auxiliary = dg.state_auxiliary,
    )

    # determine the time step
    acoustic_speed = soundspeed_air(model.param_set, FT(setup.T0))

    dx = min_node_distance(grid, HorizontalDirection())
    dz = min_node_distance(grid, VerticalDirection())
    aspect_ratio = dx / dz
   

    Q = init_ode_state(dg, FT(0))

    # LU doesn't work with periodic bcs
    linearsolver = GeneralizedMinimalResidual(
        Q,
        M = 50,
        rtol = sqrt(eps(FT)) / 100,
        atol = sqrt(eps(FT)) / 100,
    )

    if split_explicit_implicit
        rem_dg = remainder_DGModel(
            dg,
            (lineardg_vert,)
        )
    end

    # IMEX
    #cfl = FT(60 // 100)
    #dt = cfl * dx / acoustic_speed
    #odesolver = ARK2GiraldoKellyConstantinescu(
    #    split_explicit_implicit ? rem_dg : dg,
    #    lineardg,
    #    LinearBackwardEulerSolver(
    #        linearsolver;
    #        isadjustable = true,
    #        preconditioner_update_freq = -1,
    #    ),
    #    Q;
    #    dt = dt,
    #    t0 = 0,
    #    split_explicit_implicit = split_explicit_implicit,
    #    variant=NaiveVariant(),
    #)
    
    # Explicit
    cfl = FT(60 // 100)
    dt = cfl * dz / acoustic_speed
    odesolver = LSRK54CarpenterKennedy(dg, Q; dt = dt, t0 = 0)
    
    nsteps = ceil(Int, timeend / dt)

    filterorder = 18
    filter = ExponentialFilter(grid, 0, filterorder)
    cbfilter = EveryXSimulationSteps(1) do
        Filters.apply!(
            Q,
            :,
            #AtmosFilterPerturbations(model),
            grid,
            filter,
            #state_auxiliary = dg.state_auxiliary,
        )
        nothing
    end
    
    cbcheck = EveryXSimulationSteps(10) do
        ρ = Array(Q.data[:, 1, :])
        ρu = Array(Q.data[:, 2, :])
        ρv = Array(Q.data[:, 3, :])
        ρw = Array(Q.data[:, 4, :])

        u = ρu ./ ρ
        v = ρv ./ ρ
        w = ρw ./ ρ

        @info "u = $(extrema(u))"
        @info "v = $(extrema(v))"
        @info "w = $(extrema(w))"
        nothing
    end

    eng0 = norm(Q)
    @info @sprintf """Starting
                      ArrayType       = %s
                      FT              = %s
                      polynomialorder = %d
                      numelem_horz    = %d
                      numelem_vert    = %d
                      dt              = %.16e
                      norm(Q₀)        = %.16e
                      """ "$ArrayType" "$FT" polynomialorder numelem_horz numelem_vert dt eng0

    # Set up the information callback
    starttime = Ref(now())
    cbinfo = EveryXWallTimeSeconds(60, mpicomm) do (s = false)
        if s
            starttime[] = now()
        else
            energy = norm(Q)
            runtime = Dates.format(
                convert(DateTime, now() - starttime[]),
                dateformat"HH:MM:SS",
            )
            @info @sprintf """Update
                              simtime = %.16e
                              runtime = %s
                              norm(Q) = %.16e
                              """ gettime(odesolver) runtime energy
        end
    end
    callbacks = (cbinfo, cbfilter, cbcheck)

    if output_vtk
        # create vtk dir
        vtkdir =
            "vtk_acousticwave_box" *
            "_poly$(polynomialorder)_horz$(numelem_horz)_vert$(numelem_vert)" *
            "_$(ArrayType)_$(FT)"
        mkpath(vtkdir)

        vtkstep = 0
        # output initial step
        do_output(mpicomm, vtkdir, vtkstep, dg, Q, Q, model)

        # setup the output callback
        cbvtk = EveryXSimulationSteps(floor(outputtime / dt)) do
            vtkstep += 1
            Qe = init_ode_state(dg, gettime(odesolver))
            do_output(mpicomm, vtkdir, vtkstep, dg, Q, Qe, model)
        end
        callbacks = (callbacks..., cbvtk)
    end

    solve!(
        Q,
        odesolver;
        numberofsteps = nsteps,
        adjustfinalstep = false,
        callbacks = callbacks,
    )
    Qe = init_ode_state(dg, gettime(odesolver))

    @show norm(Q .- Qe, dims = (1, 3))

    # final statistics
    engf = norm(Q)
    @info @sprintf """Finished
    norm(Q)                 = %.16e
    norm(Q) / norm(Q₀)      = %.16e
    norm(Q) - norm(Q₀)      = %.16e
    """ engf engf / eng0 engf - eng0
    engf
end

Base.@kwdef struct AcousticBoxSetup{FT}
    domain_length::FT = planet_radius(param_set)
    domain_height::FT = 30e3
    T0::FT = 300
    u0::FT = 20
end

function (setup::AcousticBoxSetup)(problem, atmos, state, aux, localgeo, t)
    # callable to set initial conditions
    FT = eltype(state)
    
    _R_d::FT = R_d(param_set)
    _cp_d::FT = cp_d(param_set)
    _cv_d::FT = cv_d(param_set)
    _γ = _cp_d / _cv_d

    domain_length = setup.domain_length
    domain_height = setup.domain_height

    (x, y, z) = localgeo.coord

    u0 = setup.u0
    T0 = setup.T0
    c = soundspeed_air(param_set, FT(setup.T0))

    p_ref = aux.ref_state.p
    ρ_ref = aux.ref_state.ρ
    ρu_ref = aux.ref_state.ρu
    ρe_ref = aux.ref_state.ρe

    #pressure perturbation
    kz = 1
    δp_mag = 1
    δp = δp_mag * cos(2π * kz * (z + c * t) / domain_height)
   
    ρu_ref = ρ_ref * SVector(u0, 0, 0)
    ρe_ref += ρu_ref' * ρu_ref / (2 * ρ_ref)

    p = p_ref + δp
    δρ = δp / c ^ 2
    ρ = ρ_ref + δρ
    
    δρu = SVector(0, 0, -δp / c) + δρ * ρu_ref / ρ_ref
    
    e_int = internal_energy(param_set, T0)
    e_kin = u0 ^ 2 / 2

    δρe = δp / (_γ - 1) + ρu_ref' * δρu / ρ_ref - ρu_ref' * ρu_ref / (2 * ρ_ref ^ 2) * δρ
   
    state.ρ = ρ
    state.ρu = ρ_ref * SVector(u0, 0, 0) + δρu
    state.ρe = ρ_ref * (e_int + e_kin) + δρe
end

function do_output(
    mpicomm,
    vtkdir,
    vtkstep,
    dg,
    Q,
    Qe,
    model,
    testname = "acousticwave_box",
)
    ## name of the file that this MPI rank will write
    filename = @sprintf(
        "%s/%s_mpirank%04d_step%04d",
        vtkdir,
        testname,
        MPI.Comm_rank(mpicomm),
        vtkstep
    )

    statenames = flattenednames(vars_state(model, Prognostic(), eltype(Q)))
    exactnames = statenames .* "_exact"
    auxnames = flattenednames(vars_state(model, Auxiliary(), eltype(Q)))
    #writevtk(filename, Q, dg, statenames, dg.state_auxiliary, auxnames)
    writevtk(filename, Q, dg, statenames, Qe, exactnames)

    ## Generate the pvtu file for these vtk files
    if MPI.Comm_rank(mpicomm) == 0
        ## name of the pvtu file
        pvtuprefix = @sprintf("%s/%s_step%04d", vtkdir, testname, vtkstep)

        ## name of each of the ranks vtk files
        prefixes = ntuple(MPI.Comm_size(mpicomm)) do i
            @sprintf("%s_mpirank%04d_step%04d", testname, i - 1, vtkstep)
        end

        #writepvtu(pvtuprefix, prefixes, (statenames..., auxnames...), eltype(Q))
        writepvtu(pvtuprefix, prefixes, (statenames..., exactnames...), eltype(Q))

        @info "Done writing VTK: $pvtuprefix"
    end
end

import ClimateMachine.Atmos: atmos_init_aux!, vars_state

struct AcousticBoxReferenceState{FT} <: ReferenceState
    T0::FT
    u0::FT
end

vars_state(::AcousticBoxReferenceState, ::Auxiliary, FT) =
  @vars(ρ::FT, ρu::SVector{3, FT}, ρe::FT, p::FT, T::FT)
function atmos_init_aux!(
    refstate::AcousticBoxReferenceState,
    atmos::AtmosModel,
    aux::Vars,
    tmp::Vars,
    geom::LocalGeometry,
)
    FT = eltype(aux)
    (x, y, z) = geom.coord
    param_set = atmos.param_set
    _MSLP::FT = MSLP(param_set)
    _R_d::FT = R_d(param_set)
    _grav::FT = grav(param_set)
    
    T0 = refstate.T0
    u0 = refstate.u0
    
    e_int = internal_energy(param_set, T0)
    e_kin = u0 ^ 2 / 2
    
    p = _MSLP
    ρ = p / (_R_d * T0)

    aux.ref_state.ρ = ρ
    aux.ref_state.ρu = ρ * SVector(u0, 0, 0)
    aux.ref_state.ρe = ρ * (e_int + e_kin)
    aux.ref_state.p = p
    aux.ref_state.T = T0
end

main()