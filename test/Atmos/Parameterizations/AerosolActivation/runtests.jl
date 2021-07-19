
using SpecialFunctions
using Test

using Thermodynamics

using ClimateMachine.AerosolModel: mode, aerosol_model
using ClimateMachine.AerosolActivation

using ClimateMachine.Microphysics: G_func

using CLIMAParameters
using CLIMAParameters: gas_constant
using CLIMAParameters.Planet: molmass_water, ρ_cloud_liq, grav, cp_d, molmass_dryair
using CLIMAParameters.Atmos.Microphysics

include("/home/skadakia/clones/ClimateMachine.jl/src/Atmos/Parameterizations/CloudPhysics/Mode_creation.jl")
struct EarthParameterSet <: AbstractEarthParameterSet end
const EPS = EarthParameterSet
const param_set = EarthParameterSet()

T = 283.15     # air temperature
p = 100000.0   # air pressure
w = 5.0        # vertical velocity

# TODO - move areosol properties to CLIMAParameters

function tp_coeff_of_curve(param_set::EPS, T::FT) where {FT <: Real}

    _molmass_water::FT = molmass_water(param_set)
    _gas_const::FT = gas_constant()
    _ρ_cloud_liq::FT = ρ_cloud_liq(param_set)

    _surface_tension::FT  = 0.072   # TODO - use from CLIMAParameters

    return 2 * _surface_tension * _molmass_water / _ρ_cloud_liq / _gas_const / T
end

function tp_mean_hygroscopicity(param_set::EPS, am::aerosol_model)

    _molmass_water = molmass_water(param_set)
    _ρ_cloud_liq = ρ_cloud_liq(param_set)

    return ntuple(am.N) do i
        mode_i = am.modes[i]
        num_of_comp = mode_i.n_components
        numerator = sum(num_of_comp) do j
            mode_i.osmotic_coeff[j] *
            mode_i.mass_mix_ratio[j] *
            mode_i.dissoc[j] *
            mode_i.soluble_mass_frac[j] *
            1 / mode_i.molar_mass[j]
        end
        denominator = sum(num_of_comp) do j
            mode_i.mass_mix_ratio[j] / mode_i.aerosol_density[j]
        end
        (numerator / denominator) * (_molmass_water / _ρ_cloud_liq)
    end
end

function α(param_set::EPS, T::FT, aerosol_mass::FT) where {FT <: Real}

    _molmass_water::FT = molmass_water(param_set)
    _grav::FT = grav(param_set)
    _gas_constant::FT = gas_constant()
    _cp_d::FT = cp_d(param_set)
    _molmass_dryair::FT = molmass_dryair(param_set)
    L::FT = latent_heat_vapor(param_set, T)

    return _grav * _molmass_water * L / (_cp_d * _gas_constant * T^2) -
           _grav * _molmass_dryair / (_gas_constant * T)
end

function γ(
    param_set::EPS,
    T::FT,
    aerosol_mass::FT,
    press::FT,
) where {FT <: Real}

    _molmass_water::FT = molmass_water(param_set)
    _gas_constant::FT = gas_constant()
    _molmass_dryair::FT = molmass_dryair(param_set)
    _cp_d::FT = cp_d(param_set)

    L::FT = latent_heat_vapor(param_set, T)
    p_vs::FT = saturation_vapor_pressure(param_set, T, Liquid())
    # 1227.623212237539 # saturation_vapor_pressure(param_set, T, Liquid())
    return _gas_constant * T / (p_vs * _molmass_water) +
           _molmass_water * L^2  / (_cp_d * press * _molmass_dryair * T)
end

function ζ(
    param_set::EPS,
    T::FT,
    aerosol_mass::FT,
    updraft_velocity::FT,
    G_diff::FT,
) where {FT <: Real}
    α_var = α(param_set, T, aerosol_mass)
    println("this is α test: ", α_var)
    return 2 * tp_coeff_of_curve(param_set, T) / 3 *
           (α_var *= updraft_velocity / G_diff
    )^(1 / 2)
end

function η(
    param_set::EPS,
    temp::Float64,
    aerosol_mass::Float64,
    number_concentration::Float64,
    G_diff::Float64,
    updraft_velocity::Float64,
    press::Float64,
)

    _ρ_cloud_liq = ρ_cloud_liq(param_set)
    α_var = α(param_set, temp, aerosol_mass)
    γ_var = γ(param_set, temp, aerosol_mass, press)
    # println("this is α")
    # println(α_var)
    println("this is γ test: ", γ_var)
    return (α_var * updraft_velocity /
           G_diff)^(3 / 2) / (
        2 *
        pi *
        _ρ_cloud_liq * γ_var
         *
        number_concentration
    )
end

function tp_max_super_sat(
    param_set::EPS,
    am::aerosol_model,
    temp::FT,
    updraft_velocity::FT,
    press::FT) where {FT <: Real}

    _grav::FT = grav(param_set)
    _molmass_water::FT = molmass_water(param_set)
    _molmass_dryair::FT = molmass_dryair(param_set)
    _gas_constant::FT = gas_constant()
    _cp_d::FT = cp_d(param_set)
    _ρ_cloud_liq::FT = ρ_cloud_liq(param_set)

    L::FT = latent_heat_vapor(param_set, T)
    p_vs::FT = saturation_vapor_pressure(param_set, T, Liquid())
    G_diff::FT = G_func(param_set, T, Liquid())

    mean_hygro = tp_mean_hygroscopicity(param_set, am)
    return ntuple(am.N) do i
        mode_i = am.modes[i]
        num_of_comp = mode_i.n_components
        a = sum(num_of_comp) do j
            f = 0.5 * exp(2.5 * (log(mode_i.stdev))^2)
            println("this is f test: ", f)
            g = 1 + 0.25 * log(mode_i.stdev)
            println("this is g test: ", g)
            coeff_of_curve = tp_coeff_of_curve(param_set, temp)
            println("this is A test: ", coeff_of_curve)
            surface_tension_effects = ζ(param_set,
                                           temp,
                                           mode_i.molar_mass[j],
                                           updraft_velocity,
                                           G_diff)
                                
            println("this is ζ test: ", surface_tension_effects)
            critsat =
                2 / sqrt(mean_hygro[i]) *
                (coeff_of_curve / (3 * mode_i.r_dry))^(3 / 2) # FILL
            println("this is critsat test: ", critsat)
            η_value = η(param_set, temp, mode_i.molar_mass[j], mode_i.N, G_diff, updraft_velocity, press)
            println("this is η_value test: ", η_value)
            1 / (critsat^2) * (
                f * (surface_tension_effects / η_value)^(3 / 2) +
                g * (critsat^2 /
                (η_value + 3 * surface_tension_effects))^(3 / 4)
            )

        end
        a^(1 / 2)

    end
end

function tp_critical_supersaturation(
    param_set::EPS,
    am::aerosol_model,
    temp::Float64,
)
    mean_hygro = tp_mean_hygroscopicity(param_set, am)
    return ntuple(am.N) do i
        mode_i = am.modes[i]
        num_of_comp = mode_i.n_components
        a = sum(num_of_comp) do j
            2 /
            sqrt(mean_hygro[i]) *
            (
                tp_coeff_of_curve(param_set, temp) / (3 * mode_i.r_dry)
            )^(3 / 2)
        end
        a
    end

end

function tp_total_n_act(
    param_set::EPS,
    am::aerosol_model,
    temp::Float64,
    updraft_velocity::Float64,
    G_diff::Float64,
    press::Float64,
)
    critical_supersaturation = tp_critical_supersaturation(param_set, am, temp)
    max_supersat =
        tp_max_super_sat(param_set, am, temp, updraft_velocity, press)
    values = ntuple(am.N) do i
        mode_i = am.modes[i]

        sigma = mode_i.stdev
        u_top = 2 * log(critical_supersaturation[i] / max_supersat[i])
        u_bottom = 3 * sqrt(2) * log(sigma)
        u = u_top / u_bottom
        mode_i.N *
        1 / 2 * (1 - erf(u))
    end
    summation = 0.0
    for i in range(1, length = length(values))
        summation += values[i]
    end
    return summation
end


function tp_max_super_sat_prac(param_set::EPS,
                               am::aerosol_model,
                               temp::Float64,
                               updraft_velocity::Float64,
                               G_diff::Float64,
                               press::Float64)
    mean_hygro = 2 # tp_mean_hygroscopicity(param_set, am)
    return ntuple(am.N) do i
        mode_i = am.modes[i]
        num_of_comp = mode_i.n_components
        a = sum(num_of_comp) do j
            f = 2 # 0.5 * exp(2.5 * log(mode_i.stdev)^2)
            g = 2 # 1 + 0.25 * log(mode_i.stdev)
            coeff_of_curve = 2 # tp_coeff_of_curve(param_set, temp)
            surface_tension_effects = 2 
            critsat = 3
            η_value = 3
            println(j)
            1 / (critsat^2) * (f * (surface_tension_effects / η_value)^(3 / 2) + g * (critsat^2 /(η_value + 3 * surface_tension_effects))^(3 / 4))
            
        end
        a^(1 / 2)

    end
end




@testset "mean_hygroscopicity" begin

    println("----------")
    println("mean_hygroscopicity: ")
    println(tp_mean_hygroscopicity(param_set, AM_1))
    println(mean_hygroscopicity(param_set, AM_1))

    for AM in AM_test_cases
        @test all(
            tp_mean_hygroscopicity(param_set, AM) .≈
            mean_hygroscopicity(param_set, AM)
        )
    end
    println(" ")
end

@testset "max_supersaturation" begin

    println("----------")
    println("max_supersaturation: ")
    println(tp_max_super_sat(param_set, AM_1,  T, p, w))
    println(max_supersaturation(param_set, AM_1, T, p, w))

    # TODO
    #for AM in AM_test_cases
    #    @test all(
    #        tp_max_super_sat(param_set, AM, 2.0, 3.0, 4.0, 1.0) .≈
    #        max_supersaturation(param_set, AM, T, p, w)
    #    )
    #end

    println(" ")
end

@testset "total_n_act" begin

    println("----------")
    println("total_N_act: ")
    println(tp_total_n_act(param_set, AM_1, 2.0, 3.0, 4.0, 1.0))
    println(total_N_activated(param_set, AM_1, T, p, w))

    # TODO
    #for AM in AM_test_cases
    #    @test all(
    #        tp_total_n_act(param_set, AM, 2.0, 3.0, 4.0, 1.0) .≈
    #        total_N_activated(param_set, AM, T, p, w)
    #    )
    #end

    println(" ")
end


# println(tp_max_super_sat_prac(param_set, AM_5, 2.0, 3.0, 4.0, 1.0,))
# println(max_supersaturation(param_set, AM_1, T, p, w))