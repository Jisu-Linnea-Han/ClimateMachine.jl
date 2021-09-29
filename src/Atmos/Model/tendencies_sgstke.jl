##### SGS TKE tendencies

#####
##### First order flux (Advective)
#####

function flux(::SGSTKE, ::Advect, atmos, args)
    @unpack state = args

    return state.ρu * state.sgstke.e_SGS
end

#####
##### Second order flux (Diffusive)
#####

#struct SGSDiffFlux <: TendencyDef{Flux{SecondOrder}} end
function flux(::SGSTKE, ::Diffusion, atmos, args)
    @unpack state, diffusive = args
    @unpack ν, D_t = args.precomputed.turbulence

    FT = eltype(state)
    return  ( -FT(2) * ν * state.ρ ) .* diffusive.sgstke.∇e_SGS
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

    return - state.ρ * sum(τ .* diffusive.turbulence.∇u)
end

function source(::SGSTKE, ::BuoyancyProduction, atmos, args)
    @unpack state, aux, diffusive = args
    @unpack D_t = args.precomputed.turbulence

    FT = eltype(state)
    flux_θ_liq_ice = state.ρ * (-D_t) .* diffusive.energy.∇θ_liq_ice
    flux_q_t = state.ρ * (-D_t) .* diffusive.moisture.∇θq_tot
    
    # Deardorff 1980, for unsaturated air
    A = FT(1) + FT(0.61) * state.moisture.q_tot
    B = FT(0.61)
    flux_θ_vir = A * flux_θ_liq_ice[3] + B * flux_q_t[3]
    return (aux.orientation.∇Φ / aux.ref_state.T ) * flux_θ_vir
end

# The effective geometric mean grid resolution at the point
struct Dissipation <: TendencyDef{Source} end
prognostic_vars(::Dissipation) = (SGSTKE(),)

function source(::SGSTKE, ::Dissipation, atmos, args)
    @unpack state, aux, diffusive = args
    
    #turbulence_model(atmos) isa ConstantViscosity ? error("ConstantViscosity not supported") : nothing
        
    FT = eltype(state)

    Δs = aux.turbulence.Δ
    e = state.sgstke.ρe_SGS / state.ρ

    flux_θ_liq_ice = state.ρ * (-D_t) .* diffusive.energy.∇θ_liq_ice
    l_s = FT(0.76) * sqrt(e) / sqrt((aux.orientation.∇Φ / aux.ref_state.T ) * flux_θ_liq_ice)
    l = (l_s < Δs) ? l_s : Δs
    C = FT(0.19) + FT(0.51) * (l / Δs)
    return - state.ρ * C * e^FT(1.5) / l
end



