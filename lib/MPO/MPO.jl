###############################################################################
#
#  Density Matrix Renormalization Group (and other methods) in julia (DMRjulia)
#                               v1.0
#
###############################################################################
# Made by Thomas E. Baker and « les qubits volants » (2024)
# See accompanying license with this program
# This code is native to the julia programming language (v1.10.0+)
#

"""
    mpo = MPO(H[,regtens=false])

constructor for MPO with tensors `H` either a vector of tensors or the MPO. `regtens` outputs with the julia Array type
"""
function MPO(H::Array{W,1};regtens::Bool=false) where W <: TensType
  T = typeof(prod(a->eltype(H[a])(1),1:length(H)))
  if !regtens && (typeof(H[1]) <: Array)
    M = [tens{T}(H[a]) for a = 1:length(H)]
  else
    M = H
  end
  return MPO(T,M,regtens=regtens)
end

"""
    mpo = MPO(T,H[,regtens=false])

constructor for MPO with tensors `H` either a vector of tensors or the `MPO`; can request an element type `T` for the tensors. `regtens` outputs with the julia Array type
"""
function MPO(T::DataType,H::Array{W,1};regtens::Bool=false) where W <: TensType
  if W <: AbstractArray
    newH = Array{Array{eltype(H[1]),4},1}(undef,size(H,1))
  else
    newH = Array{W,1}(undef,size(H,1))
  end

  for a = 1:size(H,1)
    if ndims(H[a]) == 2
      rP = reshape(H[a],size(H[a],1),size(H[a],2),1,1)
      newH[a] = permutedims(rP,[4,1,2,3])
    else
      newH[a] = H[a]
    end
  end

  if W <: densTensType && !regtens
    finalH = Array{tens{T}}(undef,length(newH))
    for a = 1:length(newH)
      finalH[a] = tens{T}(newH[a])
    end
  else
    finalH = newH
  end
  
  return matrixproductoperator{eltype(finalH)}(network(finalH))
end

function MPO(T::DataType,H::network;regtens::Bool=false)
  return MPO(T,H.net,regtens=regtens)
end

function MPO(T::DataType,mpo::MPO;regtens::Bool=false)
  return MPO(T,mpo.H,regtens=regtens)
end

function MPO(mpo::MPO;regtens::Bool=false)
  return MPO(mpo.H,regtens=regtens)
end

function MPO(mpo::network;regtens::Bool=false)
  return MPO(mpo.net,regtens=regtens)
end

function MPO(Qlabels::Array{Array{Q,1},1},mpo::MPO,arrows::Array{Bool,1}...;infinite::Bool=false,unitcell::Integer=1,newnorm::Bool=true,setflux::Bool=false,flux::Q=Q(),randomize::Bool=true,override::Bool=true,lastfluxzero::Bool=false) where Q <: Qnum
  qmpo = makeqMPO(Qlabels,mpo,arrows...,infinite=infinite,unitcell=unitcell)
  return qmpo
end




#ease of use
function MPO(Qlabels::Array{Q,1},mpo::MPO,mps::MPS,arrows::Array{Bool,1}...;infinite::Bool=false,unitcell::Integer=1,newnorm::Bool=true,setflux::Bool=false,flux::Q=Q(),randomize::Bool=true,override::Bool=true,lastfluxzero::Bool=false) where Q <: Qnum
  qmpo = makeqMPO([Qlabels],mpo,arrows...,infinite=infinite,unitcell=unitcell)
  qpsi = makeqMPS([Qlabels],mps,arrows...,newnorm=newnorm,setflux=setflux,flux=flux,randomize=randomize,override=override,lastfluxzero=lastfluxzero)
  return qmpo,qpsi
end



function MPO(Qlabels::Array{Q,1},mpo::MPO,arrows::Array{Bool,1}...;infinite::Bool=false,unitcell::Integer=1,newnorm::Bool=true,setflux::Bool=false,flux::Q=Q(),randomize::Bool=true,override::Bool=true,lastfluxzero::Bool=false) where Q <: Qnum
  qmpo = makeqMPO([Qlabels],mpo,arrows...,infinite=infinite,unitcell=unitcell)
  return qmpo
end







function MPO(Qlabels::Array{Array{Q,1},1},mpo::MPO,mps::MPS,arrows::Array{Bool,1}...;infinite::Bool=false,unitcell::Integer=1,newnorm::Bool=true,setflux::Bool=false,flux::Q=Q(),randomize::Bool=true,override::Bool=true,lastfluxzero::Bool=false) where Q <: Qnum
  qmpo = makeqMPO(Qlabels,mpo,arrows...,infinite=infinite,unitcell=unitcell)
  qpsi = makeqMPS(Qlabels,mps,arrows...,newnorm=newnorm,setflux=setflux,flux=flux,randomize=randomize,override=override,lastfluxzero=lastfluxzero)
  return qmpo,qpsi
end

