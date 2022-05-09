module EffortLCDM

using LoopVectorization
using Flux
using Base: @kwdef

abstract type AbstractComponentEmulators end
abstract type AbstractP_lEmulators end

@kwdef mutable struct EmulatorP11_0 <: AbstractComponentEmulators
    NN
    InMinMax::Matrix{Float64} = zeros(7,2)
    OutMinMax::Array{Float64} = zeros(3,2)
end

@kwdef mutable struct EmulatorP11_2 <: AbstractComponentEmulators
    NN
    InMinMax::Matrix{Float64} = zeros(7,2)
    OutMinMax::Array{Float64} = zeros(3,2)
end

@kwdef mutable struct EmulatorPloop_0 <: AbstractComponentEmulators
    NN
    InMinMax::Matrix{Float64} = zeros(7,2)
    OutMinMax::Array{Float64} = zeros(12,2)
end

@kwdef mutable struct EmulatorPloop_2 <: AbstractComponentEmulators
    NN
    InMinMax::Matrix{Float64} = zeros(7,2)
    OutMinMax::Array{Float64} = zeros(12,2)
end

@kwdef mutable struct EmulatorPct_0 <: AbstractComponentEmulators
    NN
    InMinMax::Matrix{Float64} = zeros(7,2)
    OutMinMax::Array{Float64} = zeros(12,2)
end

@kwdef mutable struct EmulatorPct_2 <: AbstractComponentEmulators
    NN
    InMinMax::Matrix{Float64} = zeros(7,2)
    OutMinMax::Array{Float64} = zeros(12,2)
end

@kwdef mutable struct EmulatorP_0 <: AbstractP_lEmulators
    P11_0::EmulatorP11_0
    Ploop_0::EmulatorPloop_0
    Pct_0::EmulatorPct_0
end

@kwdef mutable struct EmulatorP_2 <: AbstractP_lEmulators
    P11_2::EmulatorP11_2
    Ploop_2::EmulatorPloop_2
    Pct_2::EmulatorPct_2
end

function SetkGrid!(cosmoemu::AbstractComponentEmulators, kgrid::Array{T}) where T
    cosmoemu.InputParams = zeros(6, length(kgrid))
end

function ComputeP_ℓ(cosmoemu::EmulatorP_0, kgrid::Array{T}, bs, f, cosmology) where T
    input_NN = SetInputNN(cosmoemu.P11_0, kgrid, cosmology)

    P11_comp_array = ComputePk(cosmoemu.P11_0, input_NN)
    Ploop_comp_array = ComputePk(cosmoemu.Ploop_0, input_NN)
    Pct_comp_array = ComputePk(cosmoemu.Pct_0, input_NN)

    return SumMultipoleComponents(P11_comp_array, Ploop_comp_array, Pct_comp_array, bs, f)
end

function ComputeP_ℓ(cosmoemu::EmulatorP_2, kgrid::Array{T}, bs, f, cosmology) where T
    input_NN = SetInputNN(cosmoemu.P11_2, kgrid, cosmology)

    P11_comp_array = ComputePk(cosmoemu.P11_2, input_NN)
    Ploop_comp_array = ComputePk(cosmoemu.Ploop_2, input_NN)
    Pct_comp_array = ComputePk(cosmoemu.Pct_2, input_NN)

    return SumMultipoleComponents(P11_comp_array, Ploop_comp_array, Pct_comp_array, bs, f)
end

function SumMultipoleComponents(P11_comp_array, Ploop_comp_array, Pct_comp_array,
    bs::Array{T}, f) where T
    b1, b2, b3, b4, b5, b6, b7 = bs

    b11 = Array([ b1^2, 2*b1*f, f^2])
    bloop = Array([ 1., b1, b2, b3, b4, b1*b1, b1*b2, b1*b3, b1*b4, b2*b2, b2*b4, b4*b4 ])
    bct = Array([ 2*b1*b5, 2*b1*b6, 2*b1*b7, 2*f*b5, 2*f*b6, 2*f*b7 ])

    P11_array = Array{T}(zeros(length(P11_comp_array[1,:])))
    Ploop_array = Array{T}(zeros(length(P11_comp_array[1,:])))
    Pct_array = Array{T}(zeros(length(P11_comp_array[1,:])))

    EvaluateP11!(P11_array, b11, P11_comp_array)
    EvaluatePloop!(Ploop_array, bloop, Ploop_comp_array)
    EvaluatePct!(Pct_array, bct, Pct_comp_array)
    P_0 = P11_array .+ Ploop_array .+ Pct_array

    return P_0
end

function SetInputNN(cosmoemu, kgrid::Array{T}, cosmology::Array{R}) where {T, R}
    # assume the parameters are order as ln10A_s, ns, h, omb, omc
    input_params = Array{R}(zeros(6, length(kgrid)))

    input_params[1,:] .= (cosmology[1] - cosmoemu.InMinMax[1,1])/
    (cosmoemu.InMinMax[1,2]-cosmoemu.InMinMax[1,1])
    input_params[2,:] .= (cosmology[2] - cosmoemu.InMinMax[2,1])/
    (cosmoemu.InMinMax[2,2]-cosmoemu.InMinMax[2,1])
    input_params[3,:] .= (cosmology[3] - cosmoemu.InMinMax[3,1])/
    (cosmoemu.InMinMax[3,2]-cosmoemu.InMinMax[3,1])
    input_params[4,:] .= (cosmology[4] - cosmoemu.InMinMax[4,1])/
    (cosmoemu.InMinMax[4,2]-cosmoemu.InMinMax[4,1])
    input_params[5,:] .= (cosmology[5] - cosmoemu.InMinMax[5,1])/
    (cosmoemu.InMinMax[5,2]-cosmoemu.InMinMax[5,1])
    input_params[6,:]  = (log10.(kgrid) .-cosmoemu.InMinMax[6,1]) ./
    (cosmoemu.InMinMax[6,2]-cosmoemu.InMinMax[6,1])

    return input_params
end

function ComputePk(cosmoemulator::EmulatorP11_0, input_NN)
    y = cosmoemulator.NN(input_NN)
    n_k = length(input_NN[1,:])

    @avx for i in 1:length(y[:,1])
        for l in 1:n_k
            y[i,l] *= (cosmoemulator.OutMinMax[i,2]-cosmoemulator.OutMinMax[i,1])
            y[i,l] += (cosmoemulator.OutMinMax[i,1])
            y[i,l] = 10^y[i,l]
        end
    end

    return y
end

function ComputePk(cosmoemulator::EmulatorP11_2, input_NN)
    y = cosmoemulator.NN(input_NN)
    n_k = length(input_NN[1,:])

    @avx for i in 1:length(y[:,1])
        for l in 1:n_k
            y[i,l] *= (cosmoemulator.OutMinMax[i,2]-cosmoemulator.OutMinMax[i,1])
            y[i,l] += (cosmoemulator.OutMinMax[i,1])
        end
    end

    @avx for l in 1:n_k
        y[2,l] = 10^y[2,l]
        y[3,l] = 10^y[3,l]
    end
    return y
end

function EvaluateP11!(input_array, b11, Pk_in)
    @avx for b in 1:3
        for k in 1:length(input_array)
            input_array[k] += b11[b]*Pk_in[b,k]
        end
    end
end

function ComputePk(cosmoemulator::EmulatorPloop_0, input_NN)
    y = cosmoemulator.NN(input_NN)
    n_k = length(input_NN[1,:])

    @avx for i in 1:length(y[:,1])
        for l in 1:n_k
            y[i,l] *= (cosmoemulator.OutMinMax[i,2]-cosmoemulator.OutMinMax[i,1])
            y[i,l] += (cosmoemulator.OutMinMax[i,1])
        end
    end

    return y
end

function ComputePk(cosmoemulator::EmulatorPloop_2, input_NN)
    y = cosmoemulator.NN(input_NN)
    n_k = length(input_NN[1,:])

    @avx for i in 1:length(y[:,1])
        for l in 1:n_k
            y[i,l] *= (cosmoemulator.OutMinMax[i,2]-cosmoemulator.OutMinMax[i,1])
            y[i,l] += (cosmoemulator.OutMinMax[i,1])
        end
    end

    return y
end


function EvaluatePloop!(input_array, bloop, Pk_in)
    @avx for b in 1:12
        for k in 1:length(input_array)
            input_array[k] += bloop[b]*Pk_in[b,k]
        end
    end

end

function ComputePk(cosmoemulator::EmulatorPct_0, input_NN)
    y = cosmoemulator.NN(input_NN)
    n_k = length(input_NN[1,:])

    @avx for i in 1:length(y[:,1])
        for l in 1:n_k
            y[i,l] *= (cosmoemulator.OutMinMax[i,2]-cosmoemulator.OutMinMax[i,1])
            y[i,l] += (cosmoemulator.OutMinMax[i,1])
            y[i,l] = 10^y[i,l]
        end
    end

    return y
end

function ComputePk(cosmoemulator::EmulatorPct_2, input_NN)
    y = cosmoemulator.NN(input_NN)
    n_k = length(input_NN[1,:])

    @avx for i in 1:length(y[:,1])
        for l in 1:n_k
            y[i,l] *= (cosmoemulator.OutMinMax[i,2]-cosmoemulator.OutMinMax[i,1])
            y[i,l] += (cosmoemulator.OutMinMax[i,1])
        end
    end

    return y
end

function EvaluatePct!(input_array, bloop, Pk_in)
    @avx for b in 1:6
        for k in 1:length(input_array)
            input_array[k] += bloop[b]*Pk_in[b,k]
        end
    end

end


end # module
