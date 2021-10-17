##### SGS TKE tendencies
export ShearProduction,
    BuoyancyProduction,
    Dissipation

#####
##### First order flux (Advective)
#####

function flux(::SGSTKE, ::Advect, atmos, args)
    @unpack state = args
    u = state.ρu / state.ρ
    return u * state.sgstke.ρe_SGS
end

#####
##### Second order flux (Diffusive)
#####

#struct SGSDiffFlux <: TendencyDef{Flux{SecondOrder}} end
function flux(::SGSTKE, ::Diffusion, atmos, args)
    @unpack state, diffusive = args
    @unpack ν, D_t = args.precomputed.turbulence

    FT = eltype(state)
    K_m = (ν isa SDiagonal) ? diag(ν) : ν
    d_e_SGS = -FT(2) * K_m .* diffusive.sgstke.∇e_SGS
    return d_e_SGS * state.ρ
    # technically K_m instead of ν. Need to include this into TurbulenceClosures.
end

#####
##### Sources (Productions and dissipation)
#####

struct ShearProduction <: TendencyDef{Source} end
prognostic_vars(::ShearProduction) = (SGSTKE(),)
struct BuoyancyProduction <: TendencyDef{Source} end
prognostic_vars(::BuoyancyProduction) = (SGSTKE(),)

function source(::SGSTKE, ::ShearProduction, atmos, args)
    @unpack state, diffusive = args
    @unpack τ = args.precomputed.turbulence

    return - state.ρ * sum(τ .* diffusive.sgstke.∇u)
end

function source(::SGSTKE, ::BuoyancyProduction, atmos, args)
    @unpack state, aux, diffusive = args
    @unpack D_t = args.precomputed.turbulence
    param_set = parameter_set(atmos)

    FT = eltype(state)
    g = FT(grav(param_set))

    K_h = D_t # could be FT or SVector{3, FT}
    flux_θ_liq_ice = state.ρ * (-K_h) .* diffusive.sgstke.∇θ_liq_ice
    flux_q_t = state.ρ * (-K_h) .* diffusive.moisture.∇q_tot

    # Deardorff 1980, for unsaturated air
    A = FT(1) + FT(0.61) * state.moisture.ρq_tot / state.ρ
    B = FT(0.61)
    flux_θ_vir = A * flux_θ_liq_ice[3] + B * flux_q_t[3]
    return (g / aux.ref_state.T ) * flux_θ_vir
end

# The effective geometric mean grid resolution at the point
struct Dissipation <: TendencyDef{Source} end
prognostic_vars(::Dissipation) = (SGSTKE(),)

function source(::SGSTKE, ::Dissipation, atmos, args)
    @unpack state, aux, diffusive = args
    @unpack D_t = args.precomputed.turbulence
    
    #turbulence_model(atmos) isa ConstantViscosity ? error("ConstantViscosity not supported") : nothing
        
    FT = eltype(state)

    param_set = parameter_set(atmos)
    g = FT(grav(param_set))
    
    Δs = aux.turbulence.Δ
    e = state.sgstke.ρe_SGS / state.ρ

    l_s = FT(0.76) * sqrt(e) / sqrt(abs((g / aux.ref_state.T ) * diffusive.sgstke.∇θ_liq_ice[3]))
    l = (l_s < Δs) ? l_s : Δs
    C = FT(0.19) + FT(0.51) * (l / Δs)
    return - state.ρ * C * e^FT(1.5) / l
end



