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
    psi = MPS(A[,regtens=false,oc=1])

Constructs `psi` for MPS with tensors `A` (Array of tensors or MPS) with orthogonality center `oc`. `regtens` avoids conversion to `denstens` type (defaulted to perform the conversion for efficiency)
"""
function MPS(psi::Array{W,1};regtens::Bool=false,oc::Integer=1,type::DataType=eltype(psi[1])) where W <: TensType
  if W <: densTensType
    if eltype(psi[1]) != type && !regtens
      MPSvec = network([tens(type, copy(psi[i])) for i = 1:length(psi)])
      out = matrixproductstate{tens{type}}(MPSvec,oc)
    elseif !regtens
      MPSvec = network([tens(copy(psi[i])) for i = 1:length(psi)])
      out = matrixproductstate{tens{type}}(MPSvec,oc)
    else
      MPSvec = network(psi)
      out = matrixproductstate(MPSvec,oc)
    end
  elseif eltype(psi[1]) != type && W <: qarray
    MPSvec = network([Qtens(type, copy(psi[i])) for i = 1:length(psi)])
    out = matrixproductstate{Qtens{type,typeof(psi[1].flux)}}(MPSvec,oc)
  else
    MPSvec = network(psi)
    out = matrixproductstate{W}(MPSvec,oc)
  end
  return out
end

"""
    psi = MPS(A[,regtens=false,oc=1])

Constructs `psi` for MPS with tensors `A` (MPS format) with orthogonality center `oc`. `regtens` avoids conversion to `denstens` type (defaulted to perform the conversion for efficiency)
"""
function MPS(psi::MPS;regtens::Bool=false,oc::Integer=psi.oc,type::DataType=eltype(psi[1]))
  return MPS(psi.A,regtens=regtens,oc=oc,type=type)
end

"""
    psi = MPS(A[,regtens=false,oc=1])

Constructs `psi` for MPS with tensors `A` (`network`) with orthogonality center `oc`. `regtens` avoids conversion to `denstens` type (defaulted to perform the conversion for efficiency)
"""
function MPS(psi::network;regtens::Bool=false,oc::Integer=psi.oc,type::DataType=eltype(psi[1]))
  return MPS(psi.net,regtens=regtens,oc=oc,type=type)
end

#=
"""
  psi = MPS(A[,regtens=false,oc=1,type=eltype(A[1])])

Constructs `psi` for MPS with tensors `A` (Array of tensors or MPS) with orthogonality center `oc`. `regtens` avoids conversion to `denstens` type (defaulted to perform the conversion for efficiency)
"""
function MPS(B::Array{W,1};regtens::Bool=false,oc::Integer=1,type::DataType=eltype(B[1])) where W <: Array
  if !regtens
    MPSvec = [tens(convert(Array{type,ndims(B[i])},copy(B[i]))) for i = 1:size(B,1)]
  else
    MPSvec = [convert(Array{type,ndims(B[i])},copy(B[i])) for i = 1:size(B,1)]
  end
  return matrixproductstate{eltype(MPSvec)}(MPSvec,oc)
end
=#

"""
    psi = MPS(T,A[,regtens=false,oc=1])

Constructs `psi` for MPS of tensor type `T` with tensors `A` (Array of tensors or MPS) with orthogonality center `oc`. `regtens` avoids conversion to `denstens` type (defaulted to perform the conversion for efficiency)
"""
function MPS(type::DataType,B::Union{MPS,Array{W,1}};regtens::Bool=false,oc::Integer=1) where W <: Union{densTensType,Integer}
  return MPS(B,regtens=regtens,oc=oc,type=type)
end

"""
    psi = MPS(physindvec[,regtens=false,oc=1,type=Float64])

Constructs `psi` for MPS of tensor type `type` by making empty tensors of size (1,`physindvec`[w],1) for w indexing `physindvec` with othrogonality center `oc`. `regtens` avoids conversion to `denstens` type (defaulted to perform the conversion for efficiency)
"""
function MPS(physindvec::Array{W,1};regtens::Bool=false,oc::Integer=1,type::DataType=Float64) where W <: Integer
  Ns = length(physindvec)
  if regtens
    vec = Array{Array{type,3},1}(undef,Ns)
    for w = 1:Ns
      vec[w] = zeros(type,1,physindvec[w],1)
      vec[w][1,1,1] = 1
    end
  else
    vec = Array{tens{type},1}(undef,Ns)
    for w = 1:Ns
      temp = zeros(type,1,physindvec[w],1)
      temp[1,1,1] = 1
      vec[w] = tens(temp)
    end
  end
  return MPS(vec,oc=oc)
end

"""
    psi = MPS(physindvec,Ns[,regtens=false,oc=1])

Constructs `psi` for MPS of tensor type `T` by making empty tensors of size (1,`physindvec`[w],1) for w indexing 1:`Ns` (repeats on `physindvec`) with othrogonality center `oc`. `regtens` avoids conversion to `denstens` type (defaulted to perform the conversion for efficiency)
"""
function MPS(physindvec::Array{W,1},Ns::Integer;regtens::Bool=false,oc::Integer=1,type::DataType=Float64) where W <: Integer
  physindvecfull = physindvec[(w-1) % length(physindvec) + 1 for w = 1:Ns]
  return MPS(physindvecfull,regtens=regtens,oc=oc,type=type)
end

"""
    psi = MPS(physindsize,Ns[,regtens=false,oc=1,type=Float64])

Constructs `psi` for MPS of tensor type Float64 by making empty tensors of size (1,`physindsize`,1) for w indexing 1:`Ns` (repeats on `physindvec`) with othrogonality center `oc`. `regtens` avoids conversion to `denstens` type (defaulted to perform the conversion for efficiency)
"""
function MPS(physindsize::Integer,Ns::Integer;regtens::Bool=false,oc::Integer=1,type::DataType=Float64)
  return MPS([physindsize for w = 1:Ns],oc=oc,type=type)
end

"""
    psi = MPS(type,physindsize,Ns[,regtens=false,oc=1])

Constructs `psi` for MPS of tensor type Float64 by making empty tensors of size (1,`physindsize`,1) for w indexing 1:`Ns` (repeats on `physindvec`) with othrogonality center `oc`. `regtens` avoids conversion to `denstens` type (defaulted to perform the conversion for efficiency)
"""
function MPS(type::DataType,physindsize::Integer,Ns::Integer;regtens::Bool=false,oc::Integer=1)
  return MPS([physindsize for w = 1:Ns],oc=oc,type=type)
end

"""
    psi = MPS(T,physindvec,Ns[,regtens=false,oc=1])

Constructs `psi` for MPS of tensor type `T` by making empty tensors of size (1,`physindvec`[w],1) for w indexing 1:`Ns` (repeats on `physindvec`) with othrogonality center `oc`. `regtens` avoids conversion to `denstens` type (defaulted to perform the conversion for efficiency)
"""
function MPS(type::DataType,physindvec::Array{W,1},Ns::Integer;regtens::Bool=false,oc::Integer=1) where W <: Integer
  return MPS(physindvec,Ns,regtens=regtens,oc=oc,type=type)
end









function MPS(Qlabels::Array{Q,1},Ns::Integer;oc::Integer=1,type::DataType=Float64) where Q <: Qnum
  newQlabels = [Qlabels for w = 1:Ns]
  return MPS(newQlabels,Ns,oc=oc,type=type)
end


function MPS(Qlabels::Array{Array{Q,1},1},Ns::Integer;oc::Integer=1,type::DataType=Float64) where Q <: Qnum
  physindvec = [length(Qlabels[(i-1) % length(Qlabels) + 1]) for i = 1:Ns]
  psi = MPS(physindvec,oc=oc,type=type)
  qpsi = makeqMPS(Qlabels,psi)
  return qpsi
end


function MPS(Qlabels::Array{Array{Q,1},1};oc::Integer=1,type::DataType=Float64) where Q <: Qnum
  return MPS(Qlabels,length(Qlabels),oc=oc,type=type)
end


function MPS(Tarray::Array{W,1},Qlabels::Array{Array{Q,1},1},arrows::Array{Bool,1}...;newnorm::Bool=true,setflux::Bool=false,flux::Q=Q(),randomize::Bool=true,override::Bool=true,lastfluxzero::Bool=false)::MPS where {Q <: Qnum, W <: densTensType}

  A = MPS(Tarray,arrows...)
  return makeqMPS(Qlabels,A)
end

"""
    assignflux!(i,mps,QnumMat,storeVal)

Assigns flux to the right link index on an MPS tensor

# Arguments
+ `i::intType`: current position
+ `mps::MPS`: MPS
+ `QnumMat::Array{Array{Qnum,1},1}`: quantum number matrix for the physical index
+ `storeVal::Array{T,1}`: maximum value found in MPS tensor, determine quantum number
"""
function assignflux!(i::Integer,mps::MPS,QnumMat::Array{Array{Q,1},1},storeVal::Array{Float64,1}) where Q <: Qnum
  for a = 1:size(mps[i],1)
    for b = 1:size(mps[i],2)
      @inbounds for c = 1:size(mps[i],3)
        absval = abs(mps[i][a,b,c])
        if absval > storeVal[c]
          storeVal[c] = absval
          QnumMat[3][c] = -(QnumMat[1][a]+QnumMat[2][b])
        end
      end
    end
  end
  nothing
end







function MPS(Qlabels::Array{Q,1},mps::MPS,mpo::MPO,arrows::Array{Bool,1}...;infinite::Bool=false,unitcell::Integer=1,newnorm::Bool=true,setflux::Bool=false,flux::Q=Q(),randomize::Bool=true,override::Bool=true,lastfluxzero::Bool=false) where Q <: Qnum
  qmpo = makeqMPO([Qlabels],mpo,arrows...,infinite=infinite,unitcell=unitcell)
  qpsi = makeqMPS([Qlabels],mps,arrows...,newnorm=newnorm,setflux=setflux,flux=flux,randomize=randomize,override=override,lastfluxzero=lastfluxzero)
  return qpsi,qmpo
end

function MPS(Qlabels::Array{Q,1},mps::MPS,arrows::Array{Bool,1}...;infinite::Bool=false,unitcell::Integer=1,newnorm::Bool=true,setflux::Bool=false,flux::Q=Q(),randomize::Bool=true,override::Bool=true,lastfluxzero::Bool=false) where Q <: Qnum
  qpsi = makeqMPS([Qlabels],mps,arrows...,newnorm=newnorm,setflux=setflux,flux=flux,randomize=randomize,override=override,lastfluxzero=lastfluxzero)
  return qpsi
end


function MPS(Qlabels::Array{Array{Q,1},1},mps::MPS,mpo::MPO,arrows::Array{Bool,1}...;infinite::Bool=false,unitcell::Integer=1,newnorm::Bool=true,setflux::Bool=false,flux::Q=Q(),randomize::Bool=true,override::Bool=true,lastfluxzero::Bool=false) where Q <: Qnum
  qmpo = makeqMPO(Qlabels,mpo,arrows...,infinite=infinite,unitcell=unitcell)
  qpsi = makeqMPS(Qlabels,mps,arrows...,newnorm=newnorm,setflux=setflux,flux=flux,randomize=randomize,override=override,lastfluxzero=lastfluxzero)
  return qpsi,qmpo
end





function MPS(Qlabels::Array{Array{Q,1},1},mps::MPS,arrows::Array{Bool,1}...;infinite::Bool=false,unitcell::Integer=1,newnorm::Bool=true,setflux::Bool=false,flux::Q=Q(),randomize::Bool=true,override::Bool=true,lastfluxzero::Bool=false) where Q <: Qnum
  qpsi = makeqMPS(Qlabels,mps,arrows...,newnorm=newnorm,setflux=setflux,flux=flux,randomize=randomize,override=override,lastfluxzero=lastfluxzero)
  return qpsi
end