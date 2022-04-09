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
    InputParams::Matrix{Float64} = zeros(7,3)
end

@kwdef mutable struct EmulatorP11_2 <: AbstractComponentEmulators
    NN
    InMinMax::Matrix{Float64} = zeros(7,2)
    OutMinMax::Array{Float64} = zeros(3,2)
    InputParams::Matrix{Float64} = zeros(7,3)
end

@kwdef mutable struct EmulatorPloop_0 <: AbstractComponentEmulators
    NN
    InMinMax::Matrix{Float64} = zeros(7,2)
    OutMinMax::Array{Float64} = zeros(12,2)
    InputParams::Matrix{Float64} = zeros(7,12)
end

@kwdef mutable struct EmulatorPloop_2 <: AbstractComponentEmulators
    NN
    InMinMax::Matrix{Float64} = zeros(7,2)
    OutMinMax::Array{Float64} = zeros(12,2)
    InputParams::Matrix{Float64} = zeros(7,12)
end

@kwdef mutable struct EmulatorPct_0 <: AbstractComponentEmulators
    NN
    InMinMax::Matrix{Float64} = zeros(7,2)
    OutMinMax::Array{Float64} = zeros(12,2)
    InputParams::Matrix{Float64} = zeros(7,6)
end

@kwdef mutable struct EmulatorPct_2 <: AbstractComponentEmulators
    NN
    InMinMax::Matrix{Float64} = zeros(7,2)
    OutMinMax::Array{Float64} = zeros(12,2)
    InputParams::Matrix{Float64} = zeros(7,6)
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

    P11_comp_array = ComputePk(cosmoemu.P11_0, cosmology, kgrid)
    Ploop_comp_array = ComputePk(cosmoemu.Ploop_0, cosmology, kgrid)
    Pct_comp_array = ComputePk(cosmoemu.Pct_0, cosmology, kgrid)
    
    b1, b2, b3, b4, b5, b6, b7 = bs
    b11 = Array([ b1^2, 2*b1*f, f^2 ])
    bloop = Array([ 1., b1, b2, b3, b4, b1*b1, b1*b2, b1*b3, b1*b4, b2*b2, b2*b4, b4*b4 ])
    bct = Array([ 2*b1*b5, 2*b1*b6, 2*b1*b7, 2*f*b5, 2*f*b6, 2*f*b7 ])
    
    P11_array = zeros(length(kgrid))
    Ploop_array = zeros(length(kgrid))
    Pct_array = zeros(length(kgrid))
    
    return SumMultipoleComponents(P11_comp_array, Ploop_comp_array, Pct_comp_array, bs, f)
end



function ComputeP_ℓ(cosmoemu::EmulatorP_2, kgrid::Array{T}, bs, f, cosmology) where T

    P11_comp_array = ComputePk(cosmoemu.P11_2, cosmology, kgrid)
    Ploop_comp_array = ComputePk(cosmoemu.Ploop_2, cosmology, kgrid)
    Pct_comp_array = ComputePk(cosmoemu.Pct_2, cosmology, kgrid)
    
    b1, b2, b3, b4, b5, b6, b7 = bs
    b11 = Array([ b1^2, 2*b1*f, f^2 ])
    bloop = Array([ 1., b1, b2, b3, b4, b1*b1, b1*b2, b1*b3, b1*b4, b2*b2, b2*b4, b4*b4 ])
    bct = Array([ 2*b1*b5, 2*b1*b6, 2*b1*b7, 2*f*b5, 2*f*b6, 2*f*b7 ])
    
    P11_array = zeros(length(kgrid))
    Ploop_array = zeros(length(kgrid))
    Pct_array = zeros(length(kgrid))
    
    return SumMultipoleComponents(P11_comp_array, Ploop_comp_array, Pct_comp_array, bs, f)
end

function SumMultipoleComponents(P11_comp_array, Ploop_comp_array, Pct_comp_array, bs, f)
    b1, b2, b3, b4, b5, b6, b7 = bs
    
    b11 = Array([ b1^2, 2*b1*f, f^2])
    bloop = Array([ 1., b1, b2, b3, b4, b1*b1, b1*b2, b1*b3, b1*b4, b2*b2, b2*b4, b4*b4 ])
    bct = Array([ 2*b1*b5, 2*b1*b6, 2*b1*b7, 2*f*b5, 2*f*b6, 2*f*b7 ])
    
    P11_array = zeros(length(P11_comp_array[1,:]))
    Ploop_array = zeros(length(P11_comp_array[1,:]))
    Pct_array = zeros(length(P11_comp_array[1,:]))
    
    EvaluateP11_0!(P11_array, b11, P11_comp_array)
    EvaluatePloop_0!(Ploop_array, bloop, Ploop_comp_array)
    EvaluatePct_0!(Pct_array, bct, Pct_comp_array)
    P_0 = P11_array .+ Ploop_array .+ Pct_array
    
    return P_0
end

function SetInputNN!(cosmoemu, kgrid::Array{T}, cosmology) where T
    cosmoemu.InputParams[1,:] .= (cosmology["ln10A_s"] - cosmoemu.InMinMax[1,1])/
    (cosmoemu.InMinMax[1,2]-cosmoemu.InMinMax[1,1])
    cosmoemu.InputParams[2,:] .= (cosmology["n_s"] - cosmoemu.InMinMax[2,1])/
    (cosmoemu.InMinMax[2,2]-cosmoemu.InMinMax[2,1])
    cosmoemu.InputParams[3,:] .= (cosmology["h"] - cosmoemu.InMinMax[3,1])/
    (cosmoemu.InMinMax[3,2]-cosmoemu.InMinMax[3,1])
    cosmoemu.InputParams[4,:] .= (cosmology["omega_b"] - cosmoemu.InMinMax[4,1])/
    (cosmoemu.InMinMax[4,2]-cosmoemu.InMinMax[4,1])
    cosmoemu.InputParams[5,:] .= (cosmology["omega_cdm"] - cosmoemu.InMinMax[5,1])/
    (cosmoemu.InMinMax[5,2]-cosmoemu.InMinMax[5,1])
    cosmoemu.InputParams[6,:]  = (log10.(kgrid) .-cosmoemu.InMinMax[6,1]) ./
    (cosmoemu.InMinMax[6,2]-cosmoemu.InMinMax[6,1])
end

function ComputePk(cosmoemulator::EmulatorP11_0, cosmology, kgrid::Array{Float64})
    SetkGrid!(cosmoemulator, kgrid)
    SetInputNN!(cosmoemulator, kgrid, cosmology)
    y = cosmoemulator.NN(cosmoemulator.InputParams)
    n_k = length(kgrid)

    @avx for i in 1:length(y[:,1])
        for l in 1:n_k
            y[i,l] *= (cosmoemulator.OutMinMax[i,2]-cosmoemulator.OutMinMax[i,1])
            y[i,l] += (cosmoemulator.OutMinMax[i,1])
            y[i,l] = 10^y[i,l]
        end
    end
    
    return y
end

function ComputePk(cosmoemulator::EmulatorP11_2, cosmology, kgrid::Array{Float64})
    SetkGrid!(cosmoemulator, kgrid)
    SetInputNN!(cosmoemulator, kgrid, cosmology)
    y = cosmoemulator.NN(cosmoemulator.InputParams)
    n_k = length(kgrid)
    @avx for i in 1:length(y[:,1])
        for l in 1:n_k
            y[i,l] *= (cosmoemulator.OutMinMax[i,2]-cosmoemulator.OutMinMax[i,1])
            y[i,l] += (cosmoemulator.OutMinMax[i,1])
        end
    end
    for l in 1:n_k
        y[2,l] = 10^y[2,l]
        y[3,l] = 10^y[3,l]
    end
    return y
end

function EvaluateP11_0!(input_array, b11, Pk_in)
    @avx for b in 1:3
        for k in 1:length(input_array)
            input_array[k] += b11[b]*Pk_in[b,k]
        end
    end
end

function ComputePk(cosmoemulator::EmulatorPloop_0, cosmology, kgrid::Array{Float64})
    SetkGrid!(cosmoemulator, kgrid)
    SetInputNN!(cosmoemulator, kgrid, cosmology)
    y = cosmoemulator.NN(cosmoemulator.InputParams)
    n_k = length(kgrid)
    @avx for i in 1:length(y[:,1])
        for l in 1:n_k
            y[i,l] *= (cosmoemulator.OutMinMax[i,2]-cosmoemulator.OutMinMax[i,1])
            y[i,l] += (cosmoemulator.OutMinMax[i,1])
        end
    end
    
    return y
end

function ComputePk(cosmoemulator::EmulatorPloop_2, cosmology, kgrid::Array{Float64})
    SetkGrid!(cosmoemulator, kgrid)
    SetInputNN!(cosmoemulator, kgrid, cosmology)
    y = cosmoemulator.NN(cosmoemulator.InputParams)
    n_k = length(kgrid)
    @avx for i in 1:length(y[:,1])
        for l in 1:n_k
            y[i,l] *= (cosmoemulator.OutMinMax[i,2]-cosmoemulator.OutMinMax[i,1])
            y[i,l] += (cosmoemulator.OutMinMax[i,1])
        end
    end
    
    return y
end


function EvaluatePloop_0!(input_array, bloop, Pk_in)
    @avx for b in 1:12
        for k in 1:length(input_array)
            input_array[k] += bloop[b]*Pk_in[b,k]
        end
    end
    
end

function ComputePk(cosmoemulator::EmulatorPct_0, cosmology, kgrid::Array{Float64})
    SetkGrid!(cosmoemulator, kgrid)
    SetInputNN!(cosmoemulator, kgrid, cosmology)
    y = cosmoemulator.NN(cosmoemulator.InputParams)
    n_k = length(kgrid)
    @avx for i in 1:length(y[:,1])
        for l in 1:n_k
            y[i,l] *= (cosmoemulator.OutMinMax[i,2]-cosmoemulator.OutMinMax[i,1])
            y[i,l] += (cosmoemulator.OutMinMax[i,1])
            y[i,l] = 10^y[i,l]
        end
    end
    
    return y
end

function ComputePk(cosmoemulator::EmulatorPct_2, cosmology, kgrid::Array{Float64})
    SetkGrid!(cosmoemulator, kgrid)
    SetInputNN!(cosmoemulator, kgrid, cosmology)
    y = cosmoemulator.NN(cosmoemulator.InputParams)
    n_k = length(kgrid)
    @avx for i in 1:length(y[:,1])
        for l in 1:n_k
            y[i,l] *= (cosmoemulator.OutMinMax[i,2]-cosmoemulator.OutMinMax[i,1])
            y[i,l] += (cosmoemulator.OutMinMax[i,1])
        end
    end
    
    return y
end

function EvaluatePct_0!(input_array, bloop, Pk_in)
    @avx for b in 1:6
        for k in 1:length(input_array)
            input_array[k] += bloop[b]*Pk_in[b,k]
        end
    end
    
end


end # module
