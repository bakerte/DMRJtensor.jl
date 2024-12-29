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
    psi = MPS(A[,regtens=false,oc=psi.oc,type=...])

Constructs `psi` for MPS with tensors `A` (Array of tensors or MPS) with orthogonality center `oc`. `regtens` avoids conversion to `denstens` type (defaulted to perform the conversion for efficiency); `type` defaults to the input type of the tensors in `psi`
"""
function MPS(psi::Union{Array{W,1},Memory{W}};regtens::Bool=false,oc::Integer=1,type::DataType=eltype(psi[1])) where W <: Union{TensType,TNobj}
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
    psi = MPS(A[,regtens=false,oc=psi.oc,type=...])

Constructs `psi` for MPS with tensors `A` (MPS format) with orthogonality center `oc`. `regtens` avoids conversion to `denstens` type (defaulted to perform the conversion for efficiency); `type` defaults to the input type of the tensors in `psi`
"""
function MPS(psi::MPS;regtens::Bool=false,oc::Integer=psi.oc,type::DataType=eltype(psi[1]))
  return MPS(psi.A,regtens=regtens,oc=oc,type=type)
end

"""
    psi = MPS(A[,regtens=false,oc=psi.oc,type=...])

Constructs `psi` for MPS with tensors `A` (`network`) with orthogonality center `oc`. `regtens` avoids conversion to `denstens` type (defaulted to perform the conversion for efficiency); `type` defaults to the input type of the tensors in `psi`
"""
function MPS(psi::network;regtens::Bool=false,oc::Integer=psi.oc,type::DataType=eltype(psi[1]))
  return MPS(psi.net,regtens=regtens,oc=oc,type=type)
end

"""
    psi = MPS(T,A[,regtens=false,oc=1])

Constructs `psi` for MPS of tensor type `T` with tensors `A` (Array of tensors or MPS) with orthogonality center `oc`. `regtens` avoids conversion to `denstens` type (defaulted to perform the conversion for efficiency)
"""
function MPS(type::DataType,B::Union{MPS,Array{W,1}};regtens::Bool=false,oc::Integer=1) where W <: Union{densTensType,Integer}
  return MPS(B,regtens=regtens,oc=oc,type=type)
end

"""
    cpsi,cmpo = MPS(T,psi,mpo[,oc=1])

Converts `psi` (MPS) and `mpo` (MPO) to type given by `T`
"""
function MPS(type::DataType,psi::MPS,mpo::MPO;regtens::Bool=false,oc::Integer=1)
  return MPS(psi,regtens=regtens,oc=oc,type=type),MPO(type,mpo,regtens=regtens)
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








"""
    qmps = MPS(Qlabels[,oc=1,type=Float64])

Creates a quantum number MPS `qmps` with quantum numbers assigned from an array of `qnum`s in `Qlabels` with orthogonality center `oc` and data type `type`; `Ns` is the number of sites in the system
"""
function MPS(Qlabels::Array{Q,1},Ns::Integer;oc::Integer=1,type::DataType=Float64) where Q <: Qnum
  newQlabels = [Qlabels for w = 1:Ns]
  return MPS(newQlabels,Ns,oc=oc,type=type)
end

"""
    qmps = MPS(Qlabels[,oc=1,type=Float64])

Creates a quantum number MPS `qmps` with quantum numbers assigned from an array of an array of `qnum` (non-uniform and can be shorter than `Ns`, which will cause the program to repeat a shorter number) `Qlabels` with orthogonality center `oc` and data type `type`; `Ns` is the number of sites in the system
"""
function MPS(Qlabels::Array{Array{Q,1},1},Ns::Integer;oc::Integer=1,type::DataType=Float64) where Q <: Qnum
  physindvec = [length(Qlabels[(i-1) % length(Qlabels) + 1]) for i = 1:Ns]
  psi = MPS(physindvec,oc=oc,type=type)
  qpsi = makeqMPS(Qlabels,psi)
  return qpsi
end

"""
    qmps = MPS(Qlabels[,oc=1,type=Float64])

Creates a quantum number MPS `qmps` with quantum numbers assigned from an array of `qnum` vectors `Qlabels` with orthogonality center `oc` and data type `type`
"""
function MPS(Qlabels::Array{Array{Q,1},1};oc::Integer=1,type::DataType=Float64) where Q <: Qnum
  return MPS(Qlabels,length(Qlabels),oc=oc,type=type)
end

"""
    qmps = MPS(Qlabels,arr[,oc=1,newnorm=true,setflux=false,flux=...,randomize=true,override=true,lastfluxzero=false])

Converts an array `arr` into a quantum number MPS `qmps` with quantum numbers assigned from an array of `qnum` vectors `Qlabels`

# Optional arguments
+ `mps::MPS`: dense MPS
+ `Qlabels::Array{Array{Qnum,1},1}`: quantum number labels on each physical index (assigns physical index labels mod size of this vector)
+ `oc::Integer`: Integer
+ `newnorm::Bool`: set new norm of the MPS tensor
+ `setflux::Bool`: toggle to force this to be the total flux of the MPS tensor
+ `flux::Qnum`: quantum number to force total MPS flux to be if setflux=true
+ `randomize::Bool`: randomize last tensor if flux forces a zero tensor
+ `lastfluxzero::Bool`: determines whether the rightmost flux should be zero or not
"""
function MPS(Qlabels::Array{Array{Q,1},1},Tarray::Array{W,1},;oc::Integer=1,newnorm::Bool=true,setflux::Bool=false,flux::Q=Q(),randomize::Bool=true,override::Bool=true,lastfluxzero::Bool=false)::MPS where {Q <: Qnum, W <: densTensType}

  A = MPS(Tarray,oc=oc)
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






"""
    MPS(Qlabels,mps[,newnorm=true,setflux=false,flux=...,randomize=true,override=true,lastfluxzero=false])

Converts an MPS `mps` into a quantum number version with `Qlabels` assigned uniformly to every site

# Optional arguments
+ `mps::MPS`: dense MPS
+ `Qlabels::Array{Array{Qnum,1},1}`: quantum number labels on each physical index (assigns physical index labels mod size of this vector)
+ `newnorm::Bool`: set new norm of the MPS tensor
+ `setflux::Bool`: toggle to force this to be the total flux of the MPS tensor
+ `flux::Qnum`: quantum number to force total MPS flux to be if setflux=true
+ `randomize::Bool`: randomize last tensor if flux forces a zero tensor
+ `lastfluxzero::Bool`: determines whether the rightmost flux should be zero or not
"""
function MPS(Qlabels::Array{Q,1},mps::MPS;newnorm::Bool=true,setflux::Bool=false,flux::Q=Q(),randomize::Bool=true,override::Bool=true,lastfluxzero::Bool=false) where Q <: Qnum
  qpsi = makeqMPS([Qlabels],mps,newnorm=newnorm,setflux=setflux,flux=flux,randomize=randomize,override=override,lastfluxzero=lastfluxzero)
  return qpsi
end

"""
    MPS(Qlabels,mps,mpo[,newnorm=true,setflux=false,flux=...,randomize=true,override=true,lastfluxzero=false])

Converts an MPS `mps` and MPO `mpo` into a an array of quantum number arrays with quantum numbers assigned from an array of `qnum` vectors `Qlabels`

# Optional arguments
+ `mps::MPS`: dense MPS
+ `Qlabels::Array{Array{Qnum,1},1}`: quantum number labels on each physical index (assigns physical index labels mod size of this vector)
+ `newnorm::Bool`: set new norm of the MPS tensor
+ `setflux::Bool`: toggle to force this to be the total flux of the MPS tensor
+ `flux::Qnum`: quantum number to force total MPS flux to be if setflux=true
+ `randomize::Bool`: randomize last tensor if flux forces a zero tensor
+ `lastfluxzero::Bool`: determines whether the rightmost flux should be zero or not
"""
function MPS(Qlabels::Array{Array{Q,1},1},mps::MPS,mpo::MPO;newnorm::Bool=true,setflux::Bool=false,flux::Q=Q(),randomize::Bool=true,override::Bool=true,lastfluxzero::Bool=false) where Q <: Qnum
  qmpo = makeqMPO(Qlabels,mpo)
  qpsi = makeqMPS(Qlabels,mps,newnorm=newnorm,setflux=setflux,flux=flux,randomize=randomize,override=override,lastfluxzero=lastfluxzero)
  return qpsi,qmpo
end




"""
    MPS(Qlabels,mps[,newnorm=true,setflux=false,flux=...,randomize=true,override=true,lastfluxzero=false])

Converts an MPS `mps` into a an array of quantum number arrays with quantum numbers assigned from an array of `qnum` vectors `Qlabels`

# Optional arguments
+ `mps::MPS`: dense MPS
+ `Qlabels::Array{Array{Qnum,1},1}`: quantum number labels on each physical index (assigns physical index labels mod size of this vector)
+ `newnorm::Bool`: set new norm of the MPS tensor
+ `setflux::Bool`: toggle to force this to be the total flux of the MPS tensor
+ `flux::Qnum`: quantum number to force total MPS flux to be if setflux=true
+ `randomize::Bool`: randomize last tensor if flux forces a zero tensor
+ `lastfluxzero::Bool`: determines whether the rightmost flux should be zero or not
"""
function MPS(Qlabels::Array{Array{Q,1},1},mps::MPS;newnorm::Bool=true,setflux::Bool=false,flux::Q=Q(),randomize::Bool=true,override::Bool=true,lastfluxzero::Bool=false) where Q <: Qnum
  qpsi = makeqMPS(Qlabels,mps,newnorm=newnorm,setflux=setflux,flux=flux,randomize=randomize,override=override,lastfluxzero=lastfluxzero)
  return qpsi
end

"""
    MPS(Qlabels,mps,mpo[,newnorm=true,setflux=false,flux=...,randomize=true,override=true,lastfluxzero=false])

Converts an MPS `mps` and MPO `mpo` into a quantum number version with `Qlabels` assigned uniformly to every site

# Optional arguments
+ `mps::MPS`: dense MPS
+ `Qlabels::Array{Array{Qnum,1},1}`: quantum number labels on each physical index (assigns physical index labels mod size of this vector)
+ `newnorm::Bool`: set new norm of the MPS tensor
+ `setflux::Bool`: toggle to force this to be the total flux of the MPS tensor
+ `flux::Qnum`: quantum number to force total MPS flux to be if setflux=true
+ `randomize::Bool`: randomize last tensor if flux forces a zero tensor
+ `lastfluxzero::Bool`: determines whether the rightmost flux should be zero or not
"""
function MPS(Qlabels::Array{Q,1},mps::MPS,mpo::MPO;newnorm::Bool=true,setflux::Bool=false,flux::Q=Q(),randomize::Bool=true,override::Bool=true,lastfluxzero::Bool=false) where Q <: Qnum
  qmpo = makeqMPO([Qlabels],mpo)
  qpsi = makeqMPS([Qlabels],mps,newnorm=newnorm,setflux=setflux,flux=flux,randomize=randomize,override=override,lastfluxzero=lastfluxzero)
  return qpsi,qmpo
end
