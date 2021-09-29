##### Prognostic variable

export Mass, Momentum, Energy, ρθ_liq_ice
export AbstractMoistureVariable, TotalMoisture, LiquidMoisture, IceMoisture
export AbstractPrecipitationVariable, Rain, Snow
export Tracers
export AbstractSGSTKEVariable, SGSTKE

struct Mass <: AbstractPrognosticVariable end
struct Momentum <: AbstractMomentumVariable end

struct Energy <: AbstractEnergyVariable end
struct ρθ_liq_ice <: AbstractEnergyVariable end

struct TotalMoisture <: AbstractMoistureVariable end
struct LiquidMoisture <: AbstractMoistureVariable end
struct IceMoisture <: AbstractMoistureVariable end

struct Rain <: AbstractPrecipitationVariable end
struct Snow <: AbstractPrecipitationVariable end

struct Tracers{N} <: AbstractTracersVariable{N} end

abstract type AbstractSGSTKEVariable <: AbstractPrognosticVariable end
struct SGSTKE <: AbstractSGSTKEVariable end
