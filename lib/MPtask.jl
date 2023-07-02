#########################################################################
#
#  Density Matrix Renormalization Group (and other methods) in julia (DMRjulia)
#                              v0.8
#
#########################################################################
# Made by Thomas E. Baker (2020)
# See accompanying license with this program
# This code is native to the julia programming language (v1.5.4+)
#

"""
  Module: MPtask

Functions to generate MPSs and MPOs
"""
#=
module MPmaker
using ..tensor
using ..QN
using ..Qtensor
using ..Qtask
using ..contractions
using ..decompositions
using ..MPutil
=#

"""
  MPS

Abstract types for MPS
"""
abstract type MPS end
export MPS

"""
  MPO
  
Abstract types for MPO
"""
abstract type MPO end
export MPO

"""
  MPS

Abstract types for MPS
"""
abstract type regMPS <: MPS end
export regMPS

"""
  MPO
  
Abstract types for MPO
"""
abstract type regMPO <: MPO end
export regMPO

"""
  envType

Vector that holds environments (rank N)
"""
abstract type envType end
export envType

abstract type regEnv <: envType end
export regEnv

"""
  Env

A global super-type for the environment defined either with `envType` or `AbstractArray`
"""
const Env = Union{Array,envType}
export Env

"""
  `matrixproductstate` 

Contruct this through `MPS`. struct to hold regMPS tensors and orthogonality center

# Fields:
+ `A::Array{TensType,1}`: vector of MPS tensors
+ `oc::Integer`: orthogonality center

See also: [`MPS`](@ref)
"""
mutable struct matrixproductstate{W} <: regMPS where W <: TensType
  A::Array{W,1}
  oc::Integer
end

"""
  `matrixproductoperator` 
  
Contruct this through `MPO`. Struct to hold MPO tensors

# Fields:
+ `H::Array{TensType,1}`: vector of MPO tensors

See also: [`MPO`](@ref)
"""
struct matrixproductoperator{W} <: regMPO where W <: TensType
  H::Array{W,1}
end

"""
  `environment`

Construct this object through `Env`. Array that holds environment tensors

# Fields:
+ `V::Array{TensType,1}`: vector of environment tensors

See also: [`Env`](@ref)
"""
struct environment{W} <: regEnv where W <: TensType
  V::Array{W,1}
end

"""
    nameMPS(A)

Assigns names to MPS `A`

See also: [`nameMPO`](@ref)
"""
function nameMPS(psi::MPS)
  TNmps = Array{TNobj,1}(undef,length(psi))
  for i = 1:length(TNmps)
    TNmps[i] = nametens(psi[i],["l$(i-1)","p$i","l$i"])
  end
  return network(TNmps)
end
export nameMPS

"""
    nameMPO(A)

Assigns names to MPO `A`

See also: [`nameMPS`](@ref)
"""
function nameMPO(mpo::MPO)
  TNmpo = Array{TNobj,1}(undef,length(mpo))
  for i = 1:length(mpo)
    TNmpo[i] = nametens(mpo[i],["l$(i-1)","p$i","d$i","l$i"])
  end
  return network(TNmpo)
end
export nameMPO

"""
  psi = MPS(A[,regtens=false,oc=1])

Constructs `psi` for MPS with tensors `A` (Array of tensors or MPS) with orthogonality center `oc`. `regtens` avoids conversion to `denstens` type (defaulted to perform the conversion for efficiency)
"""
function MPS(psi::Array{W,1};regtens::Bool=false,oc::Integer=1,type::DataType=eltype(psi[1])) where W <: TensType
  if W <: densTensType
    if eltype(psi[1]) != type && !regtens
      MPSvec = [tens(convertTens(type, copy(psi[i]))) for i = 1:length(psi)]
      out = matrixproductstate{tens{type}}(MPSvec,oc)
    elseif !regtens
      MPSvec = [tens(copy(psi[i])) for i = 1:length(psi)]
      out = matrixproductstate{tens{type}}(MPSvec,oc)
    else
      MPSvec = psi
      out = matrixproductstate{W}(MPSvec,oc)
    end
  elseif eltype(psi[1]) != type && W <: qarray
    MPSvec = [convertTens(type, copy(psi[i])) for i = 1:length(psi)]
    out = matrixproductstate{Qtens{type,typeof(psi[1].flux)}}(MPSvec,oc)
  else
    MPSvec = psi
    out = matrixproductstate{W}(MPSvec,oc)
  end
  return out
end

function MPS(psi::MPS;regtens::Bool=false,oc::Integer=psi.oc,type::DataType=eltype(psi[1]))
  return MPS(psi.A,regtens=regtens,oc=oc,type=type)
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

"""
  psi = randMPS(T,physindsize,Ns[,oc=1,m=1])

Generates MPS with data type `T`, uniform physical index size `physindsize`, with `Ns` sites, orthogonality center `oc`, and bond dimension `m`.
"""
function randMPS(T::DataType,physindsize::Integer,Ns::Integer;oc::Integer=1,m::Integer=1)
  physindvec = [physindsize for i = 1:Ns]
  return randMPS(T,physindvec,oc=oc,m=m)
end

"""
  psi = randMPS(T,physindvec,Ns[,oc=1,m=1])

Generates MPS with data type `T`, physical index size vector `physindvec` (repeating over `Ns` sites), with `Ns` sites, orthogonality center `oc`, and bond dimension `m`.
"""
function randMPS(T::DataType,physindvec::Array{W,1};oc::Integer=1,m::Integer=1) where W <: Integer
  Ns = length(physindvec)
  vec = Array{Array{T,3},1}(undef,Ns)
  if m == 1
    for w = 1:Ns
      vec[w] = zeros(1,physindvec[w],1)
      state = rand(1:physindvec[w],1)[1]
      vec[w][1,state,1] = 1
    end
    psi = MPS(vec,oc=oc)
  else
    Lsize,Rsize = 1,prod(w->physindvec[w],2:length(physindvec))
    currLsize = 1
    for w = 1:Ns
      physindsize = physindvec[w]
      currRsize = min(Rsize,m)
      vec[w] = rand(T,currLsize,physindsize,currRsize)
      vec[w] /= norm(vec[w])
      currLsize = currRsize
      Rsize = cld(Rsize,physindsize)
    end
    psi = MPS(vec,oc=oc)
    move!(psi,1)
    move!(psi,Ns)
    move!(psi,1)
    psi[oc] /= expect(psi)
  end
  return psi
end

"""
  psi = randMPS(physindsize,Ns[,oc=1,m=1])

Generates MPS with data type Float64, uniform physical index size `physindsize`, with `Ns` sites, orthogonality center `oc`, and bond dimension `m`.
"""
function randMPS(physindsize::Integer,Ns::Integer;oc::Integer=1,m::Integer=1,datatype::DataType=Float64)
  return randMPS(datatype,physindsize,Ns,oc=oc,m=m)
end

"""
  psi = randMPS(T,physindvec,Ns[,oc=1,m=1])

Generates MPS with data type Float64, physical index size vector `physindvec` (repeating over `Ns` sites), with `Ns` sites, orthogonality center `oc`, and bond dimension `m`.
"""
function randMPS(physindvec::Array{W,1};oc::Integer=1,m::Integer=1,datatype::DataType=Float64) where W <: Integer
  return randMPS(datatype,physindvec,oc=oc,m=m)
end

function randMPS(physindvec::Array{W,1},Ns::Integer;oc::Integer=1,m::Integer=1,datatype::DataType=Float64) where W <: Integer
  newphysindvec = [physindvec[(w-1) % length(physindvec) + 1] for w = 1:Ns]
  return randMPS(datatype,newphysindvec,oc=oc,m=m)
end

function randMPS(psi::MPS;oc::Integer=psi.oc,m::Integer=maximum([size(psi[w],3) for w = 1:length(psi)]),datatype::DataType=eltype(psi),physind::Union{intType,Array{intType,1}} = [size(psi[w],2) for w = 1:length(psi)])
  if typeof(psi[1]) <: qarray
    Ns = length(psi)
    Qlabels = [[getQnum(2,w,psi[i]) for w = 1:size(psi[i],2)] for i = 1:Ns]
    return randMPS(Qlabels,datatype=datatype,physindvec,oc=oc,m=m)
  else
    Ns = length(psi)
    physindvec = [physind[(i-1) % length(physind) + 1] for i = 1:Ns]
    return randMPS(datatype,physindvec,oc=oc,m=m)
  end
end

function randMPS(mpo::MPO;oc::Integer=1,m::Integer=1,datatype::DataType=eltype(mpo),physind::Union{intType,Array{intType,1}} = [size(mpo[w],2) for w = 1:length(mpo)])
  if typeof(mpo[1]) <: qarray
    Ns = length(mpo)
    Qlabels = [[getQnum(3,w,mpo[i]) for w = 1:size(mpo[i],3)] for i = 1:Ns]
    return randMPS(Qlabels,datatype=datatype,physindvec,oc=oc,m=m)
  else
    Ns = length(mpo)
    physindvec = [physind[(i-1) % length(physind) + 1] for i = 1:Ns]
    return randMPS(datatype,physindvec,oc=oc,m=m)
  end
end
export randMPS

"""
  mpo = MPO(H[,regtens=false])

constructor for MPO with tensors `H` either a vector of tensors or the MPO. `regtens` outputs with the julia Array type
"""
function MPO(H::Array{W,1};regtens::Bool=false) where W <: TensType
  T = typeof(prod(a->eltype(H[a])(1),1:size(H,1)))
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
      newH[a] = permutedims(rP,[4,2,1,3])
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
  
  return matrixproductoperator{eltype(finalH)}(finalH)
end

function MPO(T::DataType,mpo::MPO;regtens::Bool=false)
  return MPO(T,mpo.H,regtens=regtens)
end

function MPO(mpo::MPO;regtens::Bool=false)
  return MPO(mpo.H,regtens=regtens)
end





"""
  V = environment(T...)

Inputs tensors `T` into environment `V`
"""
function environment(T::G...) where G <: TensType
  return environment{G}([T...])
end
export environment

"""
  V = environment(W,Ns)

Creates a blank environment of type `W` with entries for `Ns` tensors
"""
function environment(T::W,Ns::Integer) where W <: TensType
  return environment{W}(Array{W,1}(undef,Ns))
end

"""
  V = environment(P...)

Inputs tensors `P` representing an `MPS` or an `MPO` into environment `V`

See also: [`MPS`](@ref) [`MPO`](@ref)
"""
function environment(network::G) where G <: Union{MPS,MPO}
  Ns = length(network)
  return environment(network[1],Ns)
end


"""
  /(psi,c)

makes copy of input MPS `psi` and divides orthogonality center by a number `c`. Some number types require conversaion of `psi`

See also: [`convertTensor`](@ref) [`MPS`](@ref)
"""
function /(psi::MPS,num::Number)
  return div!(copy(psi),num)
end

"""
  div!(psi,c)

takes input MPS `psi` and divides orthogonality center by a number `c`. Some number types require conversaion of `psi`

See also: [`convertTensor`](@ref) [`MPS`](@ref)
"""
function div!(psi::MPS,num::Number)
  psi[psi.oc] = div!(psi[psi.oc],num)
  return psi
end

"""
  *(psi,c)

makes copy of input MPS `psi` and multiplies orthogonality center by a number `c`. Some number types require conversaion of `psi`

See also: [`convertTensor`](@ref) [`MPS`](@ref)
"""
function *(psi::MPS,num::Number)
  return mult!(copy(psi),num)
end

"""
  *(c,psi)

makes copy of input MPS `psi` and multiplies orthogonality center by a number `c`. Some number types require conversaion of `psi`

See also: [`convertTensor`](@ref) [`MPS`](@ref)
"""
function *(num::Number,psi::MPS)
  return *(psi,num)
end

"""
  mult!(psi,c)

takes input MPS `psi` and multiplies orthogonality center by a number `c`. Some number types require conversaion of `psi`

See also: [`convertTensor`](@ref) [`MPS`](@ref)
"""
function mult!(psi::MPS,num::Number)
  psi[psi.oc] = mult!(psi[psi.oc],num)
  return psi
end

"""
  mult!(c,psi)

takes input MPS `psi` and multiplies orthogonality center by a number `c`. Some number types require conversaion of `psi`

See also: [`convertTensor`](@ref) [`MPS`](@ref)
"""
function mult!(num::Number,psi::MPS)
  return mult!(psi,num)
end
#=
"""
  sub!(psi,altpsi)

Subtracts

See also: [`convertTensor`](@ref) [`MPS`](@ref)
"""
function sub!(X::MPS,Y::MPS)
  move!(X,Y.oc)
  @inbounds for i = 1:length(X)
    X[i] -= Y[i]
  end
  return X
end

function -(X::MPS,Y::MPS)
  return sub!(copy(X),Y)
end

function add!(X::MPS,Y::MPS)
  move!(X,Y.oc)
  @inbounds for i = 1:length(X)
    X[i] += Y[i]
  end
  return X
end

function +(X::MPS,Y::MPS)
  return add!(copy(X),Y)
end
=#


#  import .tensor.elnumtype
"""
  G = elnumtype(op...)

Gives type `G` of input containers `op`, a tuple containing `MPS`, `MPO`, or `Env` types
"""
function elnumtype(op...)
  opnum = eltype(op[1])(1)
  for b = 2:length(op)
    opnum *= eltype(op[b])(1)
  end
  return typeof(opnum)
end

#import Base.size
"""
  G = size(H)

`size` prints out the size tuple `G` of the tensor field of a `Env`, `MPS`, or `MPO`; this is effectively the number of sites
"""
@inline function size(H::MPO)
  return size(H.H)
end

"""
  G = size(H,i)

`size` prints out the size tuple `G` of the tensor field of a `Env`, `MPS`, or `MPO`; this is effectively the number of sites
"""
@inline function size(H::MPO,i::Integer)
  return size(H.H[i])
end

@inline function size(psi::MPS)
  return size(psi.A)
end

@inline function size(psi::MPS,i::Integer)
  return size(psi.A[i])
end

@inline function size(G::regEnv)
  return size(G.V)
end

@inline function size(G::regEnv,i::Integer)
  return size(G.V[i])
end

#import Base.length
"""
  G = length(H)

`length` prints out the number of entries of the input `H` being a `Env`, `MPS`, or `MPO`; this is effecitvely the number of sites
"""
@inline function length(H::MPO)
  return length(H.H)
end

@inline function length(psi::MPS)
  return length(psi.A)
end

@inline function length(G::regEnv)
  return length(G.V)
end

#import Base.eltype
"""
  G = eltype(Y)

`eltype` gets element type `G` of the `Env`, `MPS`, or `MPO` tensor fields
"""
@inline function eltype(Y::regMPS)
  return eltype(Y.A[1])
end

@inline function eltype(H::regMPO)
  return eltype(H.H[1])
end

@inline function eltype(G::regEnv)
  return eltype(G.V[1])
end

#import Base.getindex
"""
  G = getindex(A,i)

`getindex` allows to retrieve tensor `G` at position `i` from `Env`, `MPS` or `MPO`
"""
@inline function getindex(A::regMPS,i::Integer)
  return A.A[i]
end

"""
  G = getindex(A,r)

`getindex` allows to retrieve a range `r` (ex: r = 2:6) tensor `G` from `Env`, `MPS` or `MPO`
"""
function getindex(A::regMPS,r::UnitRange{W}) where W <: Integer
  if A.oc in r
    newoc = findfirst(w->w == A.oc,r)
  else
    newoc = 0
  end
  return MPS(A.A[r],oc=newoc)
end

@inline function getindex(H::regMPO,i::Integer)
  return H.H[i]
end

function getindex(H::regMPO,r::UnitRange{W}) where W <: Integer
  return MPO(H.H[r])
end

@inline function getindex(G::regEnv,i::Integer)
  return G.V[i]
end

function getindex(G::regEnv,r::UnitRange{W}) where W <: Integer
  return environment(G.V[r])
end


#import Base.lastindex
"""
  B = psi[end]

`lastindex!` allows to get the end element of an `Env`, `MPS`, or `MPO`
"""
@inline function lastindex(A::regMPS)
  return lastindex(A.A)
end

@inline function lastindex(H::regMPO)
  return lastindex(H.H)
end

@inline function lastindex(G::regEnv)
  return lastindex(G.V)
end


#import Base.setindex!
"""
  psi[i] = G

setindex! allows to assign elements `G` to an `Env`, `MPS`, or `MPO` at element `i`
"""
@inline function setindex!(H::regMPO,A::TensType,i::intType)
  H.H[i] = A
  nothing
end

@inline function setindex!(H::regMPS,A::TensType,i::intType)
  H.A[i] = A
  nothing
end

@inline function setindex!(G::regEnv,A::TensType,i::intType)
  G.V[i] = A
  nothing
end





function emptyTensor(A::Array{Qtens{W,Q},1}) where {W <: Number, Q <: Qnum}
  return Qtens{W,Q}()
end

function emptyTensor(A::Array{tens{W},1}) where W <: Number
  return tens{W}()
end

function emptyTensor(A::Array{Array{W,1},1}) where W <: Number
  return undefMat(W,0,0)
end

#import Base.copy
"""
  newpsi = copy(psi)

Copies an `MPS`, `MPO`, or `regEnv` to new output container of the same type; type stable (where deepcopy is type-unstable inherently)
"""
#=
function copy(mps::matrixproductstate{W}) where W <: TensType
  return matrixproductstate{W}([copy(mps.A[i]) for i = 1:length(mps)],copy(mps.oc))
end

function copy(mpo::matrixproductoperator{W}) where W <: TensType
  return matrixproductoperator{W}([copy(mpo.H[i]) for i = 1:length(mpo)])
end
=#
function copy(mps::regMPS)
  out = [copy(mps.A[i]) for i = 1:length(mps)]
  return MPS(out,oc=copy(mps.oc))
end

function copy(mpo::regMPO)
  out = [copy(mpo.H[i]) for i = 1:length(mpo)]
  return MPO(out)
end


function copy(G::regEnv)
  out = Array{typeof(emptyTensor(G.V)),1}(undef,length(G))
  for i = 1:length(G)
    try
      out[i] = copy(G.V[i])
    catch
      out[i] = emptyTensor(G.V)
    end
  end
  return environment(out)
end

#import Base.conj!
"""
  psi = conj!(psi)

Conjugates all elements in an `MPS` in-place; outputs `psi` which was input

See also [`conj`](@ref)
"""
function conj!(A::regMPS)
  conj!.(A.A)
  return A
end

#import Base.conj
"""
  A = conj(psi)

Conjugates all elements in an `MPS` and makes a copy `A`

See also [`conj!`](@ref)
"""
function conj(A::regMPS)
  B = copy(A)
  conj!.(B.A)
  return B
end


#       +---------------------------------------+
#>------+       Constructing MPO operators      +---------<
#       +---------------------------------------+

  #converts an array to an MPO so that it is instead of being represented in by an array,
  #it is represented by a tensor diagrammatically as
  #
  #       s2
  #       |
  # a1 -- W -- a2       =    W[a1,s1,s2,a2]
  #       |
  #       s1
  #
  #The original Hamiltonian matrix H in the DMRjulia.jl file is of the form
  #
  # H = [ W_11^s1s2  W_12^s1s2 W_13^s1s2 ....
  #       W_21^s1s2  W_22^s1s2 W_23^s1s2 ....
  #       W_31^s1s2  W_32^s1s2 W_33^s1s2 ....
  #       W_41^s1s2  W_42^s1s2 W_43^s1s2 ....]
  #where each W occupies the equivalent of (vars.qstates X vars.qstates) sub-matrices in
  #of the H matrix as recorded in each s1s2 pair.  These are each of the operators in H.

"""
    makeMPO(H,physSize,Ns[,infinite=,lower=])

Converts function or vector (`H`) to each of `Ns` MPO tensors; `physSize` can be a vector (one element for the physical index size on each site) or a number (uniform sites); `lower` signifies the input is the lower triangular form (default)

# Example:

```julia
spinmag = 0.5;Ns = 10
Sx,Sy,Sz,Sp,Sm,O,Id = spinOps(spinmag)
function H(i::Integer)
    return [Id O;
            Sz Id]
end
isingmpo = makeMPO(H,size(Id,1),Ns)
```
"""
function makeMPO(H::Array{Array{X,2},1},physSize::Array{Y,1};
                      lower::Bool=true,regtens::Bool=false) where {X <: Number, Y <: Integer}
  Ns = length(H)
  retType = typeof(prod(a->eltype(H[a])(1),1:Ns))
  finalMPO = Array{Array{retType,4},1}(undef,Ns)
  for i = 1:Ns
    thisH = lower ? H[i] : transpose(H[i])
    states = physSize[(i-1) % size(physSize,1) + 1]
    a1size = div(size(thisH,1),states) #represented in LEFT link indices
    a2size = div(size(thisH,2),states) #represented in RIGHT link indices

    G = Array{retType,4}(undef,a1size,states,states,a2size)
    
    for m = 1:a2size
      for k = 1:states
        for j = 1:states
          @inbounds @simd for l = 1:a1size
            G[l,j,k,m] = thisH[j + (l-1)*states, k + (m-1)*states]
          end
        end
      end
    end
    
    finalMPO[i] = G
  end
  if lower
    finalMPO[1] = finalMPO[1][end:end,:,:,:]
    finalMPO[end] = finalMPO[end][:,:,:,1:1]
  else
    finalMPO[1] = finalMPO[1][1:1,:,:,:]
    finalMPO[end] = finalMPO[end][:,:,:,end:end]
  end
  return MPO(retType,finalMPO,regtens=regtens)
end

function makeMPO(H::Array{X,2},physSize::Array{Y,1},Ns::Integer;
                      lower::Bool=true,regtens::Bool=false) where {X <: Number, Y <: Integer}
  return makeMPO([H for i = 1:Ns],physSize)
end

function makeMPO(H::Array{X,2},physSize::Y,Ns::Integer;
                      lower::Bool=true,regtens::Bool=false) where {X <: Number, Y <: Integer}
  return makeMPO([H for i = 1:Ns],[physSize])
end

function makeMPO(H::Array{X,1},physSize::Y,Ns::Integer;
                      lower::Bool=true,regtens::Bool=false) where {X <: Array, Y <: Integer}
  return makeMPO(H,[physSize],lower=lower,regtens=regtens)
end

function makeMPO(H::Array{X,1},physSize::Y;
                      lower::Bool=true,regtens::Bool=false) where {X <: Array, Y <: Integer}
  return makeMPO(H,[physSize],lower=lower,regtens=regtens)
end

function makeMPO(H::Function,physSize::Array{X,1},Ns::Integer;
                      lower::Bool=true,regtens::Bool=false) where X <: Integer
  thisvec = [H(i) for i = 1:Ns]
  return makeMPO(thisvec,physSize,lower=lower,regtens=regtens)
end

function makeMPO(H::Function,physSize::Integer,Ns::Integer;
                      lower::Bool=true,regtens::Bool=false)
  return makeMPO(H,[physSize],Ns,lower=lower,regtens=regtens)
end
export makeMPO

"""
    makeMPS(vect,physInd[,Ns=,oc=])

generates an MPS from a single vector (i.e., from exact diagonalization) for `Ns` sites and `physInd` size physical index at orthogonality center `oc`
"""
function makeMPS(vect::Array{W,1},inputphysInd::Array{P,1};Ns::Integer=length(physInd),left2right::Bool=true,oc::Integer=left2right ? Ns : 1,regtens::Bool=false) where {W <: Number, P <: Integer}


  hilbertspacesize = prod(inputphysInd)
  if hilbertspacesize != length(vect)
    prodtrack = 1
    Ns = 0
    while prodtrack != length(vect)
      Ns += 1
      prodtrack *= inputphysInd[(Ns-1) % length(inputphysInd) + 1]
    end
    physInd = [inputphysInd[(w-1) % length(inputphysInd) + 1] for w = 1:Ns]
  else
    physInd = inputphysInd
  end

  mps = Array{Array{W,3},1}(undef,Ns)
  # MPS building loop
  if left2right
    M = reshape(vect, physInd[1], div(length(vect),physInd[1]))
    Lindsize = 1 #current size of the left index
    for i=1:Ns-1
      U,DV = qr(M)
      mps[i] = reshape(U,Lindsize,physInd[i],size(DV,1))

      Lindsize = size(DV,1)
      if i == Ns-1
        mps[Ns] = unreshape(DV,Lindsize,physInd[i+1],1)
      else
        Rsize = cld(size(M,2),physInd[i+1]) #integer division, round up
        M = unreshape(DV,size(DV,1)*physInd[i+1],Rsize)
      end
    end
    finalmps = MPS(mps,oc=Ns,regtens=regtens)
  else
    M = reshape(vect, div(length(vect),physInd[end]), physInd[end])
    Rindsize = 1 #current size of the right index
    for i=Ns:-1:2
      UD,V = lq(M)
      mps[i] = reshape(V,size(UD,2),physInd[i],Rindsize)
      Rindsize = size(UD,2)
      if i == 2
        mps[1] = unreshape(UD,1,physInd[i-1],Rindsize)
      else
        Rsize = cld(size(M,1),physInd[i-1]) #integer division, round up
        M = unreshape(UD,Rsize,size(UD,2)*physInd[i-1])
      end
    end
    finalmps = MPS(mps,oc=1,regtens=regtens)
  end
  move!(finalmps,oc)
  return finalmps
end

function makeMPS(vect::denstens,physInd::Array{P,1};Ns::Integer=length(physInd),
                  left2right::Bool=true,oc::Integer=left2right ? Ns : 1,regtens::Bool=false) where P <: Integer
  newvect = copy(vect.T)
  return makeMPS(newvect,physInd;Ns=Ns,oc=oc,left2right=left2right,regtens=regtens)
end

function makeMPS(vect::Array{W,1},physInd::Integer;Ns::Integer=convert(Int64,log(physInd,length(vect))),
                  left2right::Bool=true,oc::Integer=left2right ? Ns : 1,regtens::Bool=false) where W <: Union{denstens,Number}
  vecPhysInd = [physInd for i = 1:Ns]
  return makeMPS(vect,vecPhysInd;Ns=Ns,oc=oc,left2right=left2right,regtens=regtens)
end

function makeMPS(vect::tens{W},physInd::Integer;Ns::Integer=convert(Int64,log(physInd,length(vect))),
                  left2right::Bool=true,oc::Integer=left2right ? Ns : 1,regtens::Bool=false) where W <: Union{denstens,Number}
  vecPhysInd = [physInd for i = 1:Ns]
  return makeMPS(vect,vecPhysInd;Ns=Ns,oc=oc,left2right=left2right,regtens=regtens)
end
export makeMPS

"""
    fullH(mpo)

Generates the full Hamiltonian from an MPO (memory providing); assumes lower left triagular form
"""
function fullH(mpo::MPO)
  Ns = length(mpo)
  fullH = mpo[1]
  for p = 2:Ns
    fullH = contract(fullH,ndims(fullH),mpo[p],1)
  end
  dualinds = [i+1 for i = 2:2:2Ns]
  ketinds = [i+1 for i = 1:2:2Ns]
  finalinds = vcat([1],ketinds,dualinds,[ndims(fullH)])
  pfullH = permutedims(fullH,finalinds)

  size1 = size(pfullH,1)
  size2 = prod(a->size(fullH,a),ketinds)
  size3 = prod(a->size(fullH,a),dualinds)
  size4 = size(pfullH,ndims(pfullH))

  rpfullH = reshape!(pfullH,size1,size2,size3,size4)
  return rpfullH[size(rpfullH,1),:,:,1]
end
export fullH

"""
  eigen(H)

Computes eigenvalue decomposition of an input MPO `H` that is contracted into the Hamiltonian tensor (will give a fault if the resulting array is too large)
"""
function eigen(A::MPO)
  return eigen(fullH(A))
end

"""
    fullpsi(psi)

Generates the full wavefunction from an MPS (memory providing)
"""
function fullpsi(psi::MPS)
  Ns = length(psi)
  fullpsi = psi[1]
  for p = 2:Ns
    fullpsi = contract(fullpsi,ndims(fullpsi),psi[p],1)
  end
  return reshape!(fullpsi,prod(size(fullpsi)))
end
export fullpsi







#       +---------------------------------------+
#>------+           convert to qMPS             +---------<
#       +---------------------------------------+

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


function randMPS(Qlabel::Array{Q,1},Ns::Integer;m::Integer=2,type::DataType=Float64,flux::Q=Q()) where Q <: Qnum
  Qlabels = [Qlabel for i = 1:Ns]
  return randMPS(Qlabels,Ns,m=m,type=type,flux=flux)
end



function randMPS(Qlabels::Array{Array{Q,1},1},Ns::Integer;m::Integer=2,type::DataType=Float64,flux::Q=Q()) where Q <: Qnum
  A = Array{tens{type},1}(undef,Ns)
  for i = 1:Ns
    lsize = i == 1 ? 1 : m
    rsize = i == Ns ? 1 : m
    A[i] = tens(rand(lsize,length(Qlabels[(i-1) % Ns + 1]),rsize))
  end

  mps = makeqMPS(Qlabels,A,flux=flux)

  move!(mps,Ns)
  move!(mps,1)
#  move!(mps,oc)

  mps[mps.oc] /= norm(mps[mps.oc])

  return mps
end

function MPS(Tarray::Array{W,1},Qlabels::Array{Array{Q,1},1},arrows::Array{Bool,1}...;newnorm::Bool=true,setflux::Bool=false,flux::Q=Q(),randomize::Bool=true,override::Bool=true,lastfluxzero::Bool=false)::MPS where {Q <: Qnum, W <: densTensType}

  A = MPS(Tarray,arrows...)
  return makeqMPS(Qlabels,A)
end

#=
function possibleQNs(QNsummary::Array{Q,1},w::Integer,physinds::Array{Array{Q,1},1},flux::Q,m::Integer) where Q <: Qnum
  maxQNrange = [QNsummary[q] for q = 1:length(QNsummary)]
  minQNrange = [QNsummary[q] for q = 1:length(QNsummary)]

  minQN = Q()
  maxQN = Q()
  for i = w+1:length(physinds)
    minQN += minimum(physinds[i])
    maxQN += maximum(physinds[i])
  end
  possibleQN = Array{Bool,1}(undef,length(QNsummary))
  for q = 1:length(QNsummary)
    possibleQN[q] = QNsummary[q] + minQN <= flux <= QNsummary[q] + maxQN
  end
  QNsummary = QNsummary[possibleQN]
  if length(QNsummary) > m
    QNsummary = rand(QNsummary,m)
  end
  return QNsummary
end


function randMPS(Qlabels::Array{Q,1},Ns::Integer;oc::Integer=1,m::Integer=1,type::DataType=Float64,flux::Q=Q()) where Q <: Qnum
  return randMPS([Qlabels],Ns,oc=oc,m=m,type=type,flux=flux)
end

function randMPS(Qlabels::Array{Array{Q,1},1};oc::Integer=1,m::Integer=1,type::DataType=Float64,flux::Q=Q()) where Q <: Qnum
  return randMPS(Qlabels,length(Qlabels),oc=oc,m=m,type=type,flux=flux)
end

function randMPS(Qlabels::Array{Array{Q,1},1},Ns::Integer;oc::Integer=1,m::Integer=1,type::DataType=Float64,flux::Q=Q(),maxiter::Integer=10000) where Q <: Qnum
  physinds = [Qlabels[(w-1) % length(Qlabels) + 1] for w = 1:Ns]

  storeQNs = Array{Array{Array{Q,1},1},1}(undef,Ns)
  nonzerointersect = true

  lastQNs = [inv(flux)]
  QNsummary2 = multi_indexsummary([physinds[end],lastQNs],[1,2])

  counter = 0
  while nonzerointersect && counter < maxiter
    counter += 1
    currQNs = [Q()]
    for w = 1:Ns-1
      QNsummary = multi_indexsummary([physinds[w],currQNs],[1,2])
      QNsummary = possibleQNs(QNsummary,w,physinds,flux,m)

      newQNs = inv.(QNsummary)
      temp = [currQNs,physinds[w],newQNs]
      storeQNs[w] = temp

      currQNs = QNsummary
    end

    QNsummary = intersect(inv.(currQNs),QNsummary2)
    QNsummary = possibleQNs(QNsummary,Ns,physinds,flux,m)

    newQNs = inv.(QNsummary)
    storeQNs[end] = [newQNs,physinds[end],[Q()]]
    nonzerointersect = length(newQNs) == 0
  end

  tensvec = Array{Qtens{type,Q},1}(undef,Ns)
  for w = 1:Ns
    thisblock = w <= oc ? [[1,2],[3]] : [[1],[2,3]]
    tensvec[w] = rand(storeQNs[w],currblock = thisblock, datatype = type, flux = w < Ns ? Q() : flux)
  end
  mps = MPS(tensvec,oc=oc)

  move!(mps,Ns)
  move!(mps,1)
  move!(mps,oc)

  mps[1] /= norm(mps[1])
  return mps
end
=#
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
    makeqMPS(mps,Qlabels[,arrows,newnorm=,setflux=,flux=,randomize=,override=])

creates quantum number MPS from regular MPS according to `Qlabels`

# Arguments
+ `mps::MPS`: dense MPS
+ `Qlabels::Array{Array{Qnum,1},1}`: quantum number labels on each physical index (assigns physical index labels mod size of this vector)
+ `arrows::Array{Bool,1}`: first entry of this tuple is the arrow convention on MPS tensors (default: [false,true,true])
+ `newnorm::Bool`: set new norm of the MPS tensor
+ `setflux::Bool`: toggle to force this to be the total flux of the MPS tensor
+ `flux::Qnum`: quantum number to force total MPS flux to be if setflux=true
+ `randomize::Bool`: randomize last tensor if flux forces a zero tensor
"""
function makeqMPS(Qlabels::Array{Array{Q,1},1},mps::MPS,arrows::Array{Bool,1}...;newnorm::Bool=true,setflux::Bool=false,
                  flux::Q=Q(),randomize::Bool=true,override::Bool=true,lastfluxzero::Bool=false)::MPS where Q <: Qnum
  if newnorm
    if mps.oc < length(mps)
      move!(mps,mps.oc+1)
      move!(mps,mps.oc-1)
    else
      move!(mps,mps.oc-1)
      move!(mps,mps.oc+1)
    end
    start_norm = expect(mps)
  end

  W = elnumtype(mps)
  QtensVec = Array{Qtens{W,Q},1}(undef, size(mps.A,1))

  Ns = length(mps)
  storeQnumMat = [Q()]
  theseArrows = length(arrows) == 0 ? Bool[false,true,true] : arrows[1]
  @inbounds for i = 1:Ns
    currSize = size(mps[i])
    QnumMat = Array{Q,1}[Array{Q,1}(undef,currSize[a]) for a = 1:ndims(mps[i])]

    QnumMat[1] = inv.(storeQnumMat)
    QnumMat[2] = Qlabels[(i-1) % size(Qlabels,1) + 1] 
    storeVal = zeros(Float64,size(mps[i],3))

    if i < Ns
      assignflux!(i,mps,QnumMat,storeVal)
    else
      if setflux
        QnumMat[3][1] = flux
      else
        assignflux!(i,mps,QnumMat,storeVal)
      end
    end
    storeQnumMat = QnumMat[3]
    optblocks = i <= mps.oc ? [[1,2],[3]] : [[1],[2,3]]

    QtensVec[i] = Qtens(mps[i],QnumMat,currblock=optblocks)

    if size(QtensVec[i].T,1) == 0 && randomize
      QtensVec[i] = rand(QtensVec[i])
      if size(QtensVec[i].T,1) == 0 && !override
        error("specified bad quantum number when making QN MPS...try a different quantum number")
      end
    end
  end
  finalMPS = matrixproductstate{Qtens{W,Q}}(QtensVec,mps.oc)

  thisnorm = expect(finalMPS)

  if newnorm
    finalMPS[mps.oc] *= sqrt(start_norm)/sqrt(thisnorm)
  end
  if lastfluxzero
    @inbounds for q = 1:length(finalMPS[end].Qblocksum)
      finalMPS[end].Qblocksum[q][2] = -(finalMPS[end].flux)
    end
  else
    Qnumber = finalMPS[end].QnumMat[3][1]

    finalMPS[end].flux,newQnum = -(finalMPS[end].QnumSum[3][Qnumber]),-(finalMPS[end].flux)
    finalMPS[end].QnumSum[3][1] = newQnum
  end

  @inbounds for q = 1:length(finalMPS[end].Qblocksum)
    index = finalMPS[end].ind[q][2][:,1] .+ 1 #[x]
    pos = finalMPS[end].currblock[2]
    newQsum = Q()
    @inbounds for y = 1:length(pos)
      newQsum += getQnum(pos[y],index[y],finalMPS[end])
    end
    finalMPS[end].Qblocksum[q] = (finalMPS[end].Qblocksum[q][1],newQsum)
  end

  return finalMPS #sets initial orthogonality center at site 1 (hence the arrow definition above)
end

"""
    makeqMPS(Qlabels,mps[,arrows,newnorm=,setflux=,flux=,randomize=,override=])

creates quantum number MPS from regular MPS according to `Qlabels`

# Arguments
+ `mps::MPS`: dense MPS
+ `Qlabels::Array{Qnum,1}`: quantum number labels on each physical index (uniform physical indices)
+ `arrows::Array{Bool,1}`: first entry of this tuple is the arrow convention on MPS tensors (default: [false,true,true])
+ `newnorm::Bool`: set new norm of the MPS tensor
+ `setflux::Bool`: toggle to force this to be the total flux of the MPS tensor
+ `flux::Qnum`: quantum number to force total MPS flux to be if setflux=true
+ `randomize::Bool`: randomize last tensor if flux forces a zero tensor
"""
function makeqMPS(Qlabels::Array{Q,1},mps::MPS,arrows::Array{Bool,1}...;newnorm::Bool=true,setflux::Bool=false,
  flux::Q=Q(),randomize::Bool=true,override::Bool=true,lastfluxzero::Bool=false)::MPS where Q <: Qnum
  return makeqMPS([Qlabels],mps,arrows...,newnorm=newnorm,setflux=setflux,flux=flux,randomize=randomize,override=override,lastfluxzero=lastfluxzero)
end

function makeqMPS(Qlabels::W,arr::Array,arrows::Array{Bool,1}...;oc::Integer=1,newnorm::Bool=true,setflux::Bool=false,
                  flux::Q=Q(),randomize::Bool=true,override::Bool=true,lastfluxzero::Bool=false)::MPS where W <: Union{Array{Array{Q,1},1},Array{Q,1}} where Q <: Qnum
  mps = MPS(arr,oc=oc)
  makeqMPS(Qlabels,mps,arrows...,newnorm=newnorm,setflux=setflux,flux=flux,randomize=randomize,override=override,lastfluxzero=lastfluxzero)
end
export makeqMPS

#       +---------------------------------------+
#>------+           convert to qMPO             +---------<
#       +---------------------------------------+

"""
    makeqMPO(Qlabels,mpo[,arrows])

Generates quantum number MPO from a dense Hamiltonian based on `Qlabels`

# Arguments:
+ `mpo::MPO`: dense MPO
+ `Qlabels::Array{Array{Qnum,1},1}`: quantum numbers for physical indices (modulus size of vector)
+ `arrows::Array{Bool,1}`: arrow convention for quantum numbers (default: [false,false,true,true])
"""
function makeqMPO(Qlabels::Array{Array{Q,1},1},mpo::MPO,arrows::Array{Bool,1}...;infinite::Bool=false,unitcell::Integer=1)::MPO where Q <: Qnum
  Ns = infinite ? 3*unitcell*length(mpo) : length(mpo)
  W = elnumtype(mpo)
  QtensVec = Array{Qtens{W,Q},1}(undef, Ns)

  storeQnumMat = [Q()]
  theseArrows = length(arrows) == 0 ? Bool[false,false,true,true] : arrows[1]
  @inbounds for w = 1:Ns
    i = (w-1) % length(mpo) + 1
    QnumMat = Array{Q,1}[Array{Q,1}(undef,size(mpo[i],a)) for a = 1:ndims(mpo[i])]

    QnumMat[1] = inv.(storeQnumMat)
    theseQN = Qlabels[(i-1) % size(Qlabels,1) + 1]
    QnumMat[2] = inv.(theseQN)
    QnumMat[3] = theseQN
    storeVal = -ones(Float64,size(mpo[i],4))
    for a = 1:size(mpo[i],1)
      for b = 1:size(mpo[i],2)
        for c = 1:size(mpo[i],3)
          @inbounds for d = 1:size(mpo[i],4)
            absval = abs(mpo[i][a,b,c,d])
            if absval > storeVal[d]
              storeVal[d] = absval
              tempQN = QnumMat[1][a] + QnumMat[2][b] + QnumMat[3][c]
              QnumMat[4][d] = -(tempQN)
            end
          end
        end
      end
    end
    storeQnumMat = QnumMat[4]
    baseQtens = Qtens(QnumMat,currblock=[[1,2],[3,4]])
    QtensVec[i] = Qtens(mpo[i],baseQtens)
  end
  T = prod(a->eltype(mpo[a])(1),1:length(mpo))
  if infinite
    finalQtensVec = QtensVec[unitcell*length(mpo)+1:2*unitcell*length(mpo)]
  else
    finalQtensVec = QtensVec
  end
  finalMPO = MPO(typeof(T),finalQtensVec)
  return finalMPO
end

"""
    makeqMPO(Qlabels,mpo[,arrows])

Generates quantum number MPO from a dense Hamiltonian based on `Qlabels`

# Arguments:
+ `mpo::MPO`: dense MPO
+ `Qlabels::Array{Qnum,1}`: quantum numbers for physical indices (uniform)
+ `arrows::Array{Bool,1}`: arrow convention for quantum numbers (default: [false,false,true,true])
"""
function makeqMPO(Qlabels::Array{Q,1},mpo::MPO,arrows::Array{Bool,1}...;infinite::Bool=false,unitcell::Integer=1)::MPO where Q <: Qnum
  return makeqMPO([Qlabels],mpo,arrows...,infinite=infinite,unitcell=unitcell)
end
export makeqMPO



















#ease of use
function MPO(Qlabels::Array{Q,1},mpo::MPO,mps::MPS,arrows::Array{Bool,1}...;infinite::Bool=false,unitcell::Integer=1,newnorm::Bool=true,setflux::Bool=false,flux::Q=Q(),randomize::Bool=true,override::Bool=true,lastfluxzero::Bool=false) where Q <: Qnum
  qmpo = makeqMPO([Qlabels],mpo,arrows...,infinite=infinite,unitcell=unitcell)
  qpsi = makeqMPS([Qlabels],mps,arrows...,newnorm=newnorm,setflux=setflux,flux=flux,randomize=randomize,override=override,lastfluxzero=lastfluxzero)
  return qmpo,qpsi
end

function MPS(Qlabels::Array{Q,1},mps::MPS,mpo::MPO,arrows::Array{Bool,1}...;infinite::Bool=false,unitcell::Integer=1,newnorm::Bool=true,setflux::Bool=false,flux::Q=Q(),randomize::Bool=true,override::Bool=true,lastfluxzero::Bool=false) where Q <: Qnum
  qmpo = makeqMPO([Qlabels],mpo,arrows...,infinite=infinite,unitcell=unitcell)
  qpsi = makeqMPS([Qlabels],mps,arrows...,newnorm=newnorm,setflux=setflux,flux=flux,randomize=randomize,override=override,lastfluxzero=lastfluxzero)
  return qpsi,qmpo
end

function MPO(Qlabels::Array{Q,1},mpo::MPO,arrows::Array{Bool,1}...;infinite::Bool=false,unitcell::Integer=1,newnorm::Bool=true,setflux::Bool=false,flux::Q=Q(),randomize::Bool=true,override::Bool=true,lastfluxzero::Bool=false) where Q <: Qnum
  qmpo = makeqMPO([Qlabels],mpo,arrows...,infinite=infinite,unitcell=unitcell)
  return qmpo
end

function MPS(Qlabels::Array{Q,1},mps::MPS,arrows::Array{Bool,1}...;infinite::Bool=false,unitcell::Integer=1,newnorm::Bool=true,setflux::Bool=false,flux::Q=Q(),randomize::Bool=true,override::Bool=true,lastfluxzero::Bool=false) where Q <: Qnum
  qpsi = makeqMPS([Qlabels],mps,arrows...,newnorm=newnorm,setflux=setflux,flux=flux,randomize=randomize,override=override,lastfluxzero=lastfluxzero)
  return qpsi
end





function MPO(Qlabels::Array{Array{Q,1},1},mpo::MPO,mps::MPS,arrows::Array{Bool,1}...;infinite::Bool=false,unitcell::Integer=1,newnorm::Bool=true,setflux::Bool=false,flux::Q=Q(),randomize::Bool=true,override::Bool=true,lastfluxzero::Bool=false) where Q <: Qnum
  qmpo = makeqMPO(Qlabels,mpo,arrows...,infinite=infinite,unitcell=unitcell)
  qpsi = makeqMPS(Qlabels,mps,arrows...,newnorm=newnorm,setflux=setflux,flux=flux,randomize=randomize,override=override,lastfluxzero=lastfluxzero)
  return qmpo,qpsi
end

function MPS(Qlabels::Array{Array{Q,1},1},mps::MPS,mpo::MPO,arrows::Array{Bool,1}...;infinite::Bool=false,unitcell::Integer=1,newnorm::Bool=true,setflux::Bool=false,flux::Q=Q(),randomize::Bool=true,override::Bool=true,lastfluxzero::Bool=false) where Q <: Qnum
  qmpo = makeqMPO(Qlabels,mpo,arrows...,infinite=infinite,unitcell=unitcell)
  qpsi = makeqMPS(Qlabels,mps,arrows...,newnorm=newnorm,setflux=setflux,flux=flux,randomize=randomize,override=override,lastfluxzero=lastfluxzero)
  return qpsi,qmpo
end

function MPO(Qlabels::Array{Array{Q,1},1},mpo::MPO,arrows::Array{Bool,1}...;infinite::Bool=false,unitcell::Integer=1,newnorm::Bool=true,setflux::Bool=false,flux::Q=Q(),randomize::Bool=true,override::Bool=true,lastfluxzero::Bool=false) where Q <: Qnum
  qmpo = makeqMPO(Qlabels,mpo,arrows...,infinite=infinite,unitcell=unitcell)
  return qmpo
end

function MPS(Qlabels::Array{Array{Q,1},1},mps::MPS,arrows::Array{Bool,1}...;infinite::Bool=false,unitcell::Integer=1,newnorm::Bool=true,setflux::Bool=false,flux::Q=Q(),randomize::Bool=true,override::Bool=true,lastfluxzero::Bool=false) where Q <: Qnum
  qpsi = makeqMPS(Qlabels,mps,arrows...,newnorm=newnorm,setflux=setflux,flux=flux,randomize=randomize,override=override,lastfluxzero=lastfluxzero)
  return qpsi
end
