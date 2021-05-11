# Test the branching of the SoilWaterParameterizations functions,
# test the instantiation of them works as expected, and that they
# return what we expect.
using MPI
using OrderedCollections
using StaticArrays
using Test

using CLIMAParameters
struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet()

using ClimateMachine
using ClimateMachine.Land
using ClimateMachine.Land.SoilWaterParameterizations

@testset "Land water parameterizations" begin
    FT = Float64
    test_array = [0.5, 1.0]
    vg_model = vanGenuchten(FT;)
    mm = MoistureDependent{FT}()
    bc_model = BrooksCorey(FT;)
    hk_model = Haverkamp(FT;)

    #Use an array to confirm that extra arguments are unused.
    @test viscosity_factor.(Ref(ConstantViscosity{FT}()), test_array) ≈
          [1.0, 1.0]
    @test impedance_factor.(Ref(NoImpedance{FT}()), test_array) ≈ [1.0, 1.0]
    @test moisture_factor.(
        Ref(MoistureIndependent{FT}()),
        Ref(vg_model),
        test_array,
    ) ≈ [1.0, 1.0]

    viscosity_model = TemperatureDependentViscosity{FT}(; T_ref = FT(1.0))
    @test viscosity_factor(viscosity_model, FT(1.0)) == 1
    impedance_model = IceImpedance{FT}(; Ω = 2.0)
    @test impedance_factor(impedance_model, 0.5) == FT(0.1)


    imp_f = impedance_factor(impedance_model, 0.5)
    vis_f = viscosity_factor(viscosity_model, 1.0)
    m_f = moisture_factor(MoistureDependent{FT}(), vg_model, 1.0)
    Ksat = 1.0
    @test hydraulic_conductivity(Ksat, imp_f, vis_f, m_f) == FT(0.1)

    @test_throws Exception effective_saturation(0.5, -1.0, 0.0)
    @test effective_saturation(0.5, 0.25, 0.05) - 4.0 / 9.0 < eps(FT)

    n = vg_model.n
    m = 1.0 - 1.0 / n
    α = vg_model.α
    S_s = 0.001
    θ_r = 0.0
    ν = 1.0
    @test moisture_factor.(Ref(mm), Ref(vg_model), test_array) ==
          sqrt.(test_array) .*
          (FT(1) .- (FT(1) .- test_array .^ (FT(1) / m)) .^ m) .^ FT(2)
    @test moisture_factor.(Ref(mm), Ref(bc_model), test_array) ==
          test_array .^ (FT(2) * bc_model.m + FT(3))
    ψ =
        -(
            (test_array .^ (-FT(1) / hk_model.m) .- FT(1)) .*
            hk_model.α^(-hk_model.n)
        ) .^ (FT(1) / hk_model.n)
    @test moisture_factor.(Ref(mm), Ref(hk_model), test_array) ==
          hk_model.A ./ (hk_model.A .+ abs.(ψ) .^ hk_model.k)

    @test pressure_head.(
        Ref(vg_model),
        Ref(ν),
        Ref(S_s),
        Ref(θ_r),
        test_array,
        Ref(0.0),
    ) ≈ .-((-1 .+ test_array .^ (-1 / m)) .* α^(-n)) .^ (1 / n)
    #test branching in pressure head
    @test pressure_head(vg_model, ν, S_s, θ_r, 1.5, 0.0) == 500
    @test pressure_head.(
        Ref(hk_model),
        Ref(ν),
        Ref(S_s),
        Ref(θ_r),
        test_array,
        Ref(0.0),
    ) ≈ .-((-1 .+ test_array .^ (-1 / m)) .* α^(-n)) .^ (1 / n)
    m = FT(0.5)
    ψb = FT(0.1656)
    @test pressure_head(bc_model, ν, S_s, θ_r, 0.5, 0.0) ≈ -ψb * 0.5^(-m)
    @test volumetric_liquid_fraction.([0.5, 1.5], Ref(0.75)) ≈ [0.5, 0.75]

    @test inverse_matric_potential(
        bc_model,
        matric_potential(bc_model, FT(0.5)),
    ) - FT(0.5) < eps(FT)
    @test inverse_matric_potential(
        vg_model,
        matric_potential(vg_model, FT(0.5)),
    ) == FT(0.5)
    @test inverse_matric_potential(
        hk_model,
        matric_potential(hk_model, FT(0.5)),
    ) == FT(0.5)

    @test_throws Exception inverse_matric_potential(bc_model, 1)
    @test_throws Exception inverse_matric_potential(hk_model, 1)
    @test_throws Exception inverse_matric_potential(vg_model, 1)


    # Test that spatial dependence works properly
    vg_spatial = vanGenuchten(FT; n = (x) -> 2 * x, α = (x) -> x^2)
    @test supertype(typeof(vg_spatial.n)) == Function
    vg_point = vg_spatial(FT(2.0))
    @test typeof(vg_point.n) == FT
    @test vg_point.n == 4.0
    @test vg_point.α == 4.0
    @test vg_point.m == 0.75
end
