#### Create states

function create_conservative_state(
    ::Type{MPIStateArray},
    balance_law::BalanceLaw,
    grid,
)
    topology = grid.topology
    # FIXME: Remove after updating CUDA
    h_vgeo = Array(grid.vgeo)
    FT = eltype(h_vgeo)
    Np = dofs_per_element(grid)
    DA = arraytype(grid)

    weights = view(h_vgeo, :, grid.Mid, :)
    weights = reshape(weights, size(weights, 1), 1, size(weights, 2))

    V = vars_state_conservative(balance_law, FT)
    state_conservative = MPIStateArray{FT, V}(
        topology.mpicomm,
        DA,
        Np,
        number_state_conservative(balance_law, FT),
        length(topology.elems),
        realelems = topology.realelems,
        ghostelems = topology.ghostelems,
        vmaprecv = grid.vmaprecv,
        vmapsend = grid.vmapsend,
        nabrtorank = topology.nabrtorank,
        nabrtovmaprecv = grid.nabrtovmaprecv,
        nabrtovmapsend = grid.nabrtovmapsend,
        weights = weights,
    )
    return state_conservative
end
function create_conservative_state(::Type{Array}, balance_law::BalanceLaw, grid)
    topology = grid.topology
    FT = eltype(grid.vgeo)
    Np = dofs_per_element(grid)
    return Array{FT}(
        undef,
        (
            Np,
            number_state_conservative(balance_law, FT),
            length(topology.elems),
        ),
    )
end

function create_auxiliary_state(::Type{MPIStateArray}, balance_law, grid)
    topology = grid.topology
    Np = dofs_per_element(grid)

    h_vgeo = Array(grid.vgeo)
    FT = eltype(h_vgeo)
    DA = arraytype(grid)

    weights = view(h_vgeo, :, grid.Mid, :)
    weights = reshape(weights, size(weights, 1), 1, size(weights, 2))

    V = vars_state_auxiliary(balance_law, FT)
    state_auxiliary = MPIStateArray{FT, V}(
        topology.mpicomm,
        DA,
        Np,
        number_state_auxiliary(balance_law, FT),
        length(topology.elems),
        realelems = topology.realelems,
        ghostelems = topology.ghostelems,
        vmaprecv = grid.vmaprecv,
        vmapsend = grid.vmapsend,
        nabrtorank = topology.nabrtorank,
        nabrtovmaprecv = grid.nabrtovmaprecv,
        nabrtovmapsend = grid.nabrtovmapsend,
        weights = weights,
    )
    init_auxiliary_state!(state_auxiliary, balance_law, grid)
    return state_auxiliary
end
function create_auxiliary_state(::Type{Array}, balance_law, grid)
    topology = grid.topology
    FT = eltype(grid.vgeo)
    Np = dofs_per_element(grid)
    state_auxiliary = Array{FT}(
        undef,
        (Np, number_state_auxiliary(balance_law, FT), length(topology.elems)),
    )
    init_auxiliary_state!(state_auxiliary, balance_law, grid)
    return state_auxiliary
end

function init_auxiliary_state!(state_auxiliary, balance_law, grid)
    topology = grid.topology
    dim = dimensionality(grid)
    polyorder = polynomialorder(grid)
    vgeo = grid.vgeo
    device = array_device(state_auxiliary)
    Np = dofs_per_element(grid)
    nrealelem = length(topology.realelems)
    event = Event(device)
    event = kernel_init_state_auxiliary!(device, min(Np, 1024), Np * nrealelem)(
        balance_law,
        Val(dim),
        Val(polyorder),
        data(state_auxiliary),
        vgeo,
        topology.realelems,
        dependencies = (event,),
    )
    event = MPIStateArrays.begin_ghost_exchange!(
        state_auxiliary;
        dependencies = event,
    )
    event = MPIStateArrays.end_ghost_exchange!(
        state_auxiliary;
        dependencies = event,
    )
    wait(device, event)

    return state_auxiliary
end

function create_gradient_state(::Type{MPIStateArray}, balance_law, grid)
    topology = grid.topology
    Np = dofs_per_element(grid)

    h_vgeo = Array(grid.vgeo)
    FT = eltype(h_vgeo)
    DA = arraytype(grid)

    weights = view(h_vgeo, :, grid.Mid, :)
    weights = reshape(weights, size(weights, 1), 1, size(weights, 2))

    # TODO: Clean up this MPIStateArray interface...
    V = vars_state_gradient_flux(balance_law, FT)
    state_gradient_flux = MPIStateArray{FT, V}(
        topology.mpicomm,
        DA,
        Np,
        number_state_gradient_flux(balance_law, FT),
        length(topology.elems),
        realelems = topology.realelems,
        ghostelems = topology.ghostelems,
        vmaprecv = grid.vmaprecv,
        vmapsend = grid.vmapsend,
        nabrtorank = topology.nabrtorank,
        nabrtovmaprecv = grid.nabrtovmaprecv,
        nabrtovmapsend = grid.nabrtovmapsend,
        weights = weights,
    )

    return state_gradient_flux
end

function create_gradient_state(::Type{Array}, balance_law, grid)
    topology = grid.topology
    Np = dofs_per_element(grid)
    FT = eltype(grid.vgeo)
    state_gradient_flux = Array{FT}(
        undef,
        (
            Np,
            number_state_gradient_flux(balance_law, FT),
            length(topology.elems),
        ),
    )
    return state_gradient_flux
end

function create_higher_order_states(::Type{MPIStateArray}, balance_law, grid)
    topology = grid.topology
    Np = dofs_per_element(grid)

    h_vgeo = Array(grid.vgeo)
    FT = eltype(h_vgeo)
    DA = arraytype(grid)

    weights = view(h_vgeo, :, grid.Mid, :)
    weights = reshape(weights, size(weights, 1), 1, size(weights, 2))

    ngradlapstate = num_gradient_laplacian(balance_law, FT)
    # TODO: Clean up this MPIStateArray interface...
    V = vars_gradient_laplacian(balance_law, FT)
    Qhypervisc_grad = MPIStateArray{FT, V}(
        topology.mpicomm,
        DA,
        Np,
        3ngradlapstate,
        length(topology.elems),
        realelems = topology.realelems,
        ghostelems = topology.ghostelems,
        vmaprecv = grid.vmaprecv,
        vmapsend = grid.vmapsend,
        nabrtorank = topology.nabrtorank,
        nabrtovmaprecv = grid.nabrtovmaprecv,
        nabrtovmapsend = grid.nabrtovmapsend,
        weights = weights,
    )

    V = vars_hyperdiffusive(balance_law, FT)
    Qhypervisc_div = MPIStateArray{FT, V}(
        topology.mpicomm,
        DA,
        Np,
        ngradlapstate,
        length(topology.elems),
        realelems = topology.realelems,
        ghostelems = topology.ghostelems,
        vmaprecv = grid.vmaprecv,
        vmapsend = grid.vmapsend,
        nabrtorank = topology.nabrtorank,
        nabrtovmaprecv = grid.nabrtovmaprecv,
        nabrtovmapsend = grid.nabrtovmapsend,
        weights = weights,
    )
    return Qhypervisc_grad, Qhypervisc_div
end

function create_higher_order_states(::Type{Array}, balance_law, grid)
    topology = grid.topology
    Np = dofs_per_element(grid)
    FT = eltype(grid.vgeo)
    ngradlapstate = num_gradient_laplacian(balance_law, FT)
    Qhypervisc_grad =
        Array{FT}(undef, (Np, 3ngradlapstate, length(topology.elems)))
    Qhypervisc_div =
        Array{FT}(undef, (Np, ngradlapstate, length(topology.elems)))
    return Qhypervisc_grad, Qhypervisc_div
end
