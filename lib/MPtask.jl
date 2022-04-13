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
mutable struct matrixproductstate{W} <: regMPS where W <: Array{TensType,1}
  A::W
  oc::Integer
end

"""
  `matrixproductoperator` 
  
Contruct this through `MPO`. Struct to hold MPO tensors

# Fields:
+ `H::Array{TensType,1}`: vector of MPO tensors

See also: [`MPO`](@ref)
"""
struct matrixproductoperator{W} <: regMPO where W <: Array{TensType,1}
  H::W
end

"""
  `environment`

Construct this object through `Env`. Array that holds environment tensors

# Fields:
+ `V::Array{TensType,1}`: vector of environment tensors

See also: [`Env`](@ref)
"""
struct environment{W} <: regEnv where W <: Array{TensType,1}
  V::Array{W,1}
end

"""
  V = environment(T...)

Inputs tensors `T` into environment `V`
"""
function environment(T::G...) where G <: TensType
  return environment([T...])
end
export environment

"""
  V = environment(W,Ns)

Creates a blank environment of type `W` with entries for `Ns` tensors
"""
function environment(T::W,Ns::Integer) where W <: TensType
  return environment(Array{W,1}(undef,Ns))
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
  psi = MPS(A[,regtens=false,oc=1])

Constructs `psi` for MPS with tensors `A` (Array of tensors or MPS) with orthogonality center `oc`. `regtens` avoids conversion to `denstens` type (defaulted to perform the conversion for efficiency)
"""
function MPS(psi::Array{W,1};regtens::Bool=false,oc::Integer=1,type::DataType=eltype(psi[1])) where W <: TensType
  if eltype(psi[1]) != type && !regtens
    MPSvec = [convertTens(type, copy(psi[i])) for i = 1:length(psi)]
  else
    MPSvec = psi
  end
  return matrixproductstate(MPSvec,oc) #MPS(psi,regtens=regtens,oc=oc,type=type)
end

function MPS(psi::MPS;regtens::Bool=false,oc::Integer=psi.oc,type::DataType=eltype(psi[1]))
  return MPS(psi.A,regtens=regtens,oc=oc,type=type)
end

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
  return matrixproductstate{typeof(MPSvec)}(MPSvec,oc)
end



"""
  psi = MPS(T,A[,regtens=false,oc=1])

Constructs `psi` for MPS of tensor type `T` with tensors `A` (Array of tensors or MPS) with orthogonality center `oc`. `regtens` avoids conversion to `denstens` type (defaulted to perform the conversion for efficiency)
"""
function MPS(type::DataType,B::Union{MPS,Array{W,1}};regtens::Bool=false,oc::Integer=1) where W <: Union{Array,Integer}
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

function /(psi::MPS,num::Number)
  return div!(copy(psi),num)
end

function div!(psi::MPS,num::Number)
  psi[psi.oc] = div!(psi[psi.oc],num)
  return psi
end

function *(psi::MPS,num::Number)
  return mult!(copy(psi),num)
end

function *(num::Number,psi::MPS)
  return *(psi,num)
end

function mult!(psi::MPS,num::Number)
  psi[psi.oc] = mult!(psi[psi.oc],num)
  return psi
end

function mult!(num::Number,psi::MPS)
  return mult!(psi,num)
end

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
  newphysindvec = [physindvec[(w-1) % length(physindvec)] for w = 1:Ns]
  return randMPS(datatype,newphysindvec,oc=oc,m=m)
end

function randMPS(psi::MPS;oc::Integer=psi.oc,m::Integer=maximum([size(psi[w],3) for w = 1:length(psi)]),datatype::DataType=eltype(psi),physind::Union{intType,Array{intType,1}} = [size(psi[w],2) for w = 1:length(psi)]) where W <: Integer
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

function randMPS(mpo::MPO;oc::Integer=1,m::Integer=1,datatype::DataType=eltype(mpo),physind::Union{intType,Array{intType,1}} = [size(mpo[w],2) for w = 1:length(mpo)]) where W <: Integer
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
  T = prod(a->eltype(H[a])(1),1:size(H,1))
  if !regtens && (typeof(H[1]) <: Array)
    M = [tens(H[a]) for a = 1:length(H)]
  else
    M = H
  end
  return MPO(typeof(T),M,regtens=regtens)
end

"""
  mpo = MPO(T,H[,regtens=false])

constructor for MPO with tensors `H` either a vector of tensors or the `MPO`; can request an element type `T` for the tensors. `regtens` outputs with the julia Array type
"""
function MPO(T::DataType,H::Array{W,1};regtens::Bool=false) where W <: TensType
  if W <: AbstractArray
#    if regtens
      newH = Array{Array{eltype(H[1]),4},1}(undef,size(H,1))
      #=
    else
      newH = Array{tens{eltype(H[1])},1}(undef,size(H,1))
    end=#
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

  if !regtens && (typeof(newH[1]) <: AbstractArray)
    finalH = [tens(newH[a]) for a = 1:length(newH)]
  else
    finalH = newH
  end
  if elnumtype(finalH...) != T
    finalH = [convertTens(T,finalH[w]) for w = 1:length(finalH)]
  end
  return matrixproductoperator(finalH)
end

function MPO(T::DataType,mpo::MPO;regtens::Bool=false)
  return MPO(T,mpo.H,regtens=regtens)
end

function MPO(mpo::MPO;regtens::Bool=false)
  return MPO(mpo.H,regtens=regtens)
end

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
  return size(H.H,i)
end

@inline function size(psi::MPS)
  return size(psi.A)
end

@inline function size(psi::MPS,i::Integer)
  return size(psi.A,i)
end

@inline function size(G::regEnv)
  return size(G.V)
end

@inline function size(G::regEnv,i::Integer)
  return size(G.V,i)
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


#import Base.copy
"""
  newpsi = copy(psi)

Copies an `MPS`, `MPO`, or `regEnv` to new output container of the same type; type stable (where deepcopy is type-unstable inherently)
"""
function copy(mps::matrixproductstate{W}) where W <: TensType
  return matrixproductstate{W}([copy(mps.A[i]) for i = 1:length(mps)],copy(mps.oc))
end

function copy(mpo::matrixproductoperator{W}) where W <: TensType
  return matrixproductoperator{W}([copy(mpo.H[i]) for i = 1:length(mpo)])
end

function copy(mps::regMPS)
  return MPS([copy(mps.A[i]) for i = 1:length(mps)],oc=copy(mps.oc))
end

function copy(mpo::regMPO)
  return MPO([copy(mpo.H[i]) for i = 1:length(mpo)])
end

function copy(G::regEnv)
  T = eltype(G[1])
  return envVec{T}([copy(G.V[i]) for i = 1:length(G)])
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
#>------+  move! orthogonality center in MPS    +---------<
#       +---------------------------------------+

"""
  D,truncerr = moveR(Lpsi,Rpsi[,cutoff=,m=,minm=,condition=])

moves `psi`'s orthogonality center right one space, with a maximum bond dimenion `m`, cutoff `cutoff`, minimum bond dimension `minm`; toggle truncation of tensor during movement with `condition`; returns psi[iL],psi[iR],D from the SVD, and the truncation error

See also: [`moveR!`](@ref)
"""
@inline function moveR(Lpsi::TensType,Rpsi::TensType;cutoff::Float64=0.,m::Integer=0,minm::Integer=0,condition::Bool=false,mag::Number=0.,
                fast::Bool=false)
  if (size(Lpsi,3) <= m) && !isapprox(cutoff,0.) && fast
    Ltens,modV = qr!(Lpsi,[[1,2],[3]])

    DV = (condition ? getindex!(modV,:,1:size(Rpsi,1)) : modV)
    D = DV
    truncerr = 0.
  else
    Ltens,D,V,truncerr,sumD = svd!(Lpsi,[[1,2],[3]],cutoff=cutoff,m=m,minm=minm,mag=mag)      
    modV = (condition ? getindex!(V,:,1:size(Rpsi,1)) : V)
    DV = rmul!(D,modV) #contract(D,(2,),modV,(1,))
  end
  Rtens = contract(DV,2,Rpsi,1)
  return Ltens,Rtens,D,truncerr
end
export moveR

"""
  D,truncerr = moveR!(psi[,cutoff=,m=,minm=,condition=])

acts in-place to move `psi`'s orthogonality center right one space, with a maximum bond dimenion `m`, cutoff `cutoff`, and minimum bond dimension `minm`; toggle truncation of tensor during movement with `condition`

See also: [`moveR`](@ref)
"""
@inline function moveR!(psi::MPS;cutoff::Float64=0.,m::Integer=0,minm::Integer=0,condition::Bool=false,
                  mag::Number=0.)
  iL = psi.oc
  psi[iL],psi[iL+1],D,truncerr = moveR(psi[iL],psi[iL+1],cutoff=cutoff,m=m,minm=minm,condition=condition,mag=mag)
  psi.oc += 1
  return D,truncerr
end
export moveR!

"""
  D,truncerr = moveL(Lpsi,Rpsi[,cutoff=,m=,minm=,condition=])

moves `psi`'s orthogonality center left one space, with a maximum bond dimenion `m`, cutoff `cutoff`, minimum bond dimension `minm`; toggle truncation of tensor during movement with `condition`; returns psi[iL],psi[iR],D from the SVD, and the truncation error

See also: [`moveL!`](@ref)
"""
@inline function moveL(Lpsi::TensType,Rpsi::TensType;cutoff::Float64=0.,m::Integer=0,minm::Integer=0,condition::Bool=false,mag::Number=0.,
                fast::Bool=false)
  if (size(Rpsi,1) <= m) && !isapprox(cutoff,0.) && fast
    modU,Rtens = lq!(Rpsi,[[1],[2,3]])

    UD = (condition ? getindex!(modU,1:size(Lpsi,3),:) : modU)
    D = UD
    truncerr = 0.
  else
    U,D,Rtens,truncerr,sumD = svd!(Rpsi,[[1],[2,3]],cutoff=cutoff,m=m,minm=minm,mag=mag)
    modU = (condition ? getindex!(U,1:size(Lpsi,3),:) : U)
    UD = lmul!(modU,D) #contract(modU,(2,),D,(1,))
  end
  Ltens = contract(Lpsi,3,UD,1)
  return Ltens,Rtens,D,truncerr
end
export moveL

"""
    moveL!(psi[,cutoff=,m=,minm=,condition=])

acts in-place to move `psi`'s orthogonality center left one space, with a maximum bond dimenion `m`, cutoff `cutoff`, and minimum bond dimension `minm`; toggle truncation of tensor during movement with `condition`

See also: [`moveL`](@ref)
"""
@inline function moveL!(psi::MPS;cutoff::Float64=0.,m::Integer=0,minm::Integer=0,condition::Bool=false,
                  mag::Number=0.)
  iR = psi.oc
  psi[iR-1],psi[iR],D,truncerr = moveL(psi[iR-1],psi[iR],cutoff=cutoff,m=m,minm=minm,condition=condition,mag=mag)
  psi.oc -= 1
  return D,truncerr
end
export moveL!

"""
    movecenter!(psi,newoc[,m=,cutoff=,minm=,Lfct=,Rfct=])

movement function to move `psi` to a new site, `newoc` with `Lfct` and `Rfct`, with a maximum bond dimenion `m`, cutoff `cutoff`, and minimum bond dimension `minm`; toggle truncation of tensor during movement with `condition`

See also: [`move!`](@ref) [`move`](@ref)
"""
@inline function movecenter!(psi::MPS,pos::Integer;cutoff::Float64=1E-14,m::Integer=0,minm::Integer=0,
                      Lfct::Function=moveR,Rfct::Function=moveL)
  if m == 0
    m = maximum([maximum(size(psi[i])) for i = 1:size(psi,1)])
  end
  while psi.oc != pos
    if psi.oc < pos
      iL = psi.oc
      iR = psi.oc+1
      psi[iL],psi[iR],D,truncerr = Lfct(psi[iL],psi[iR],cutoff=cutoff,m=m,minm=minm,fast=true)
      psi.oc = iR
    else
      iL = psi.oc-1
      iR = psi.oc
      psi[iL],psi[iR],D,truncerr = Rfct(psi[iL],psi[iR],cutoff=cutoff,m=m,minm=minm,fast=true)
      psi.oc = iL
    end
  end
  nothing
end

"""
    move!(psi,newoc[,m=,cutoff=,minm=])

in-place move orthgononality center of `psi` to a new site, `newoc`, with a maximum bond dimenion `m`, cutoff `cutoff`, and minimum bond dimension `minm`; toggle truncation of tensor during movement with `condition`

See also: [`move`](@ref)
"""
@inline function move!(mps::MPS,pos::Integer;m::Integer=0,cutoff::Float64=1E-14,minm::Integer=0)
  movecenter!(mps,pos,cutoff=cutoff,m=m,minm=minm)
  nothing
end
export move!

"""
    move(psi,newoc[,m=,cutoff=,minm=])

same as `move!` but makes a copy of `psi`

See also: [`move!`](@ref)
"""
@inline function move(mps::MPS,pos::Integer;m::Integer=0,cutoff::Float64=1E-14,minm::Integer=0)
  newmps = copy(mps)
  movecenter!(newmps,pos,cutoff=cutoff,m=m,minm=minm)
  return newmps
end
export move

"""
  newpsi,D,V = leftnormalize(psi)

Creates a left-normalized MPS `newpsi` from `psi` and returns the external tensors `D` and `V`
"""
function leftnormalize(psi::MPS)
  newpsi = move(psi,length(psi))
  U,D,V = svd(psi[end],[[1,2],[3]])
  newpsi[end] = U
  newpsi.oc = 0
  return newpsi,D,V
end
export leftnormalize

"""
  psi,D,V = leftnormalize!(psi)

Creates a left-normalized MPS in-place from `psi` and returns the external tensors `D` and `V`
"""
function leftnormalize!(psi::MPS)
  move!(psi,length(psi))
  U,D,V = svd(psi[end],[[1,2],[3]])
  psi[end] = U
  psi.oc = 0
  return psi,D,V
end
export leftnormalize!

"""
  U,D,newpsi = rightnormalize(psi)

Creates a right-normalized MPS `newpsi` from `psi` and returns the external tensors `U` and `D`
"""
function rightnormalize(psi::MPS)
  newpsi = move(psi,1)
  U,D,V = svd(psi[1],[[1],[2,3]])
  newpsi[1] = V
  newpsi.oc = 0
  return U,D,newpsi
end
export rightnormalize

"""
  U,D,psi = rightnormalize!(psi)

Creates a right-normalized MPS in-place from `psi` and returns the external tensors `U` and `D`
"""
function rightnormalize!(psi::MPS)
  psi = move!(psi,1)
  U,D,V = svd(psi[1],[[1],[2,3]])
  psi[1] = V
  psi.oc = 0
  return U,D,psi
end
export rightnormalize!

"""
    Lupdate(Lenv,dualpsi,psi,mpo)

Updates left environment tensor `Lenv` with dual MPS `dualpsi`, MPS `psi`, and MPO `mpo`
"""
@inline function  Lupdate(Lenv::TensType,dualpsi::TensType,psi::TensType,mpo::TensType...)
  nMPOs = length(mpo)
  nLsize = nMPOs+2
  tempLenv = contractc(Lenv,1,dualpsi,1)
  for j = 1:nMPOs
    tempLenv = contract(tempLenv,(1,nLsize),mpo[j],(1,3))
  end
  return contract(tempLenv,(1,nLsize),psi,(1,2))
end
export Lupdate

"""
    Rupdate(Renv,dualpsi,psi,mpo)

Updates right environment tensor `Renv` with dual MPS `dualpsi`, MPS `psi`, and MPO `mpo`
"""
@inline function  Rupdate(Renv::TensType,dualpsi::TensType,psi::TensType,mpo::TensType...)
  nMPOs = length(mpo)
  nLsize = nMPOs+2
  nRsize = nLsize+1
  tempRenv = ccontract(dualpsi,3,Renv,nLsize)
  for j = 1:nMPOs
    tempRenv = contract(mpo[j],(3,4),tempRenv,(2,nRsize))
  end
  return contract(psi,(2,3),tempRenv,(2,nRsize))
end
export Rupdate

"""
    Lupdate!(i,Lenv,psi,dualpsi,mpo)

Updates left environment's (`Lenv`) `i`+1 site from info on the site `i` based on dual MPS (`dualpsi`), MPS (`psi), and MPO (`mpo`)
"""
@inline function Lupdate!(i::Integer,Lenv::Env,dualpsi::MPS,psi::MPS,mpo::MPO...)
  Lenv[i+1] = Lupdate(Lenv[i],dualpsi[i],psi[i],[mpo[a][i] for a = 1:length(mpo)]...)
  nothing
end

"""
    Lupdate!(i,Lenv,psi,mpo)

Updates left environment's (`Lenv`) `i`+1 site from info on the site `i` based on MPS (`psi) and MPO (`mpo`)
"""
@inline function Lupdate!(i::Integer,Lenv::Env,psi::MPS,mpo::MPO...)
  Lupdate!(i,Lenv,psi,psi,mpo...)
end
export Lupdate!

"""
    Rupdate!(i,Renv,dualpsi,psi,mpo)

Updates right environment's (`Renv`) `i`-1 site from info on the site `i` based on dual MPS (`dualpsi`), MPS (`psi), and MPO (`mpo`)
"""
@inline function Rupdate!(i::Integer,Renv::Env,dualpsi::MPS,psi::MPS,mpo::MPO...)
  Renv[i-1] = Rupdate(Renv[i],dualpsi[i],psi[i],[mpo[a][i] for a = 1:length(mpo)]...)
  nothing
end

"""
    Rupdate!(i,Renv,psi,mpo)

Updates right environment's (`Renv`) `i`-1 site from info on the site `i` based on MPS (`psi) and MPO (`mpo`)
"""
@inline function Rupdate!(i::Integer,Renv::Env,psi::MPS,mpo::MPO...)
  Rupdate!(i,Renv,psi,psi,mpo...)
end
export Rupdate!

"""
    boundaryMove!(psi,i,mpo,Lenv,Renv)

Move orthogonality center of `psi` to site `i` and updates left and right environments `Lenv` and `Renv` using MPO `mpo`

See also: [`move!`](@ref)
"""
@inline function boundaryMove!(psi::MPS,i::Integer,Lenv::Env,
                        Renv::Env,mpo::MPO...;mover::Function=move!)
  origoc = psi.oc
  if origoc < i
    mover(psi,i)
    for w = origoc:i-1
      Lupdate!(w,Lenv,psi,mpo...)
    end
  elseif origoc > i
    mover(psi,i)
    for w = origoc:-1:i+1
      Rupdate!(w,Renv,psi,mpo...)
    end
  end
  nothing
end

"""
    boundaryMove!(dualpsi,psi,i,mpo,Lenv,Renv)

Move orthogonality center of `psi` and `dualpsi` to site `i` and updates left and right environments `Lenv` and `Renv` using MPO `mpo`

See also: [`move!`](@ref)
"""
@inline function boundaryMove!(dualpsi::MPS,psi::MPS,i::Integer,Lenv::Env,
                        Renv::Env,mpo::MPO...;mover::Function=move!)
  origoc = psi.oc
  if origoc < i
    mover(psi,i)
    mover(dualpsi,i)
    for w = origoc:i-1
      Lenv[w+1] = Lupdate(Lenv[w],dualpsi[w],psi[w],[mpo[a][w] for a = 1:length(mpo)]...)
    end
  elseif origoc > i
    mover(psi,i)
    mover(dualpsi,i)
    for w = origoc:-1:i+1
      Renv[w-1] = Rupdate(Renv[w],dualpsi[w],psi[w],[mpo[a][w] for a = 1:length(mpo)]...)
    end
  end
  nothing
end
export boundaryMove!

"""
    boundaryMove(dualpsi,psi,i,mpo,Lenv,Renv)

Copies `psi` and moves orthogonality center of `psi` and `dualpsi` to site `i` and updates left and right environments `Lenv` and `Renv` using MPO `mpo`

See also: [`move!`](@ref)
"""
@inline function boundaryMove(dualpsi::MPS,psi::MPS,i::Integer,mpo::MPO,Lenv::Env,Renv::Env)
  newpsi = copy(psi)
  newdualpsi = copy(dualpsi)
  newLenv = copy(Lenv)
  newRenv = copy(Renv)
  boundaryMove!(newdualpsi,newpsi,i,mpo,newLenv,newRenv)
  return newdualpsi,newpsi,newLenv,newRenv
end

"""
    boundaryMove(psi,i,mpo,Lenv,Renv)

Copies `psi` and moves orthogonality center of `psi` to site `i` and updates left and right environments `Lenv` and `Renv` using MPO `mpo`

See also: [`move!`](@ref)
"""
@inline function boundaryMove(psi::MPS,i::Integer,mpo::MPO,Lenv::Env,Renv::Env)
  newpsi = copy(psi)
  newLenv = copy(Lenv)
  newRenv = copy(Renv)
  boundaryMove!(newpsi,newpsi,i,mpo,newLenv,newRenv)
  return newpsi,newLenv,newRenv
end
export boundaryMove



#       +---------------------------------------+
#>------+    Construction of boundary tensors   +---------<
#       +---------------------------------------+

#
#Current environment convention is
#     LEFT              RIGHT
#   +--<-- 1          3 ---<--+
#   |                         |
#   |                         |
#   +-->-- 2          2 --->--+
#   |                         |
#   |                         |
#   +-->-- 3          1 --->--+
# any MPOs in between have the same arrow conventions as 2

"""
    makeBoundary(qind,newArrows[,retType=])

makes a boundary tensor for an input from the quantum numbers `qind` and arrows `newArrows`; can also define type of resulting Qtensor `retType` (default `Float64`)

#Note:
+dense tensors are just ones(1,1,1,...)

See also: [`makeEnds`](@ref)
"""
function makeBoundary(dualpsi::MPS,psi::MPS,mpovec::MPO...;left::Bool=true,rightind::Integer=3)
  retType = elnumtype(dualpsi,psi,mpovec...)
  nrank = 2 + length(mpovec)
  boundary = ones(retType,ones(intType,nrank)...)
  if typeof(psi[1]) <: qarray

    Q = typeof(psi[1].flux)

    qind = Array{Q,1}(undef,nrank)
    Ns = length(psi)
    site = left ? 1 : Ns
    index = left ? 1 : rightind
    qind[1] = -(getQnum(index,1,dualpsi[site]))
    qind[end] = getQnum(index,1,psi[site])
    for i = 1:length(mpovec)
      index = left ? 1 : ndims(mpovec[i][Ns])
      qind[i+1] = getQnum(index,1,mpovec[i][site])
    end

    thisQnumMat = Array{Array{Q,1},1}(undef,nrank)
    for j = 1:nrank
      qn = qind[j]
      thisQnumMat[j] = Q[qn]
    end
    return Qtens(boundary,thisQnumMat)
  else
    if typeof(psi[1]) <: denstens
      return tens(boundary)
    else
      return boundary
    end
  end
end
export makeBoundary





function makeEdgeEnv(dualpsi::MPS,psi::MPS,mpovec::MPO...;boundary::TensType=typeof(psi[1])(),left::Bool=true)
  expsize = 2+length(mpovec)
  Lnorm = norm(boundary)
  if ndims(boundary) != expsize || isapprox(Lnorm,0) || isnan(Lnorm) || isinf(Lnorm)
    Lout = makeBoundary(dualpsi,psi,mpovec...,left=left)
  else
    Lout = copy(boundary)
  end
  return Lout
end

"""
    makeEnds(dualpsi,psi[,mpovec,Lbound=,Rbound=])

Generates first and last environments for a given system of variable MPOs

# Arguments:
+ `dualpsi::MPS`: dual MPS
+ `psi::MPS`: MPS
+ `mpovec::MPO`: MPOs
+ `Lbound::TensType`: left boundary
+ `Rbound::TensType`: right boundary
"""
function makeEnds(dualpsi::MPS,psi::MPS,mpovec::MPO...;Lbound::TensType=typeof(psi[1])(),Rbound::TensType=typeof(psi[end])())
  return makeEdgeEnv(dualpsi,psi,mpovec...,boundary=Lbound),makeEdgeEnv(dualpsi,psi,mpovec...,boundary=Rbound,left=false)
end

"""
    makeEnds(psi[,mpovec,Lbound=,Rbound=])

Generates first and last environment tensors for a given system of variable MPOs.  Same as other implementation but `dualpsi`=`psi`

# Arguments:
+ `psi::MPS`: MPS
+ `mpovec::MPO`: MPOs
+ `Lbound::TensType`: left boundary
+ `Rbound::TensType`: right boundary
"""
function makeEnds(psi::MPS,mpovec::MPO...;Lbound::TensType=typeof(psi[1])(),Rbound::TensType=typeof(psi[1])())
  return makeEnds(psi,psi,mpovec...,Lbound=Lbound,Rbound=Rbound)
end
export makeEnds



function genEnv(dualpsi::MPS,psi::MPS,mpo::MPO...;bound::TensType=typeof(psi[1])())

end




"""
    makeEnv(dualpsi,psi,mpo[,Lbound=,Rbound=])

Generates environment tensors for a MPS (`psi` and its dual `dualpsi`) with boundaries `Lbound` and `Rbound`
"""
function makeEnv(dualpsi::MPS,psi::MPS,mpo::MPO...;Lbound::TensType=typeof(psi[1])(),Rbound::TensType=typeof(psi[1])())
  Ns = length(psi)
  numtype = elnumtype(dualpsi,psi,mpo...)
  C = psi[1]

  if typeof(psi) <: largeMPS || typeof(mpo) <: largeMPO
    Lenv,Renv = largeEnv(numtype,Ns)
  else
    Lenv = environment(psi)
    Renv = environment(psi)
  end
  Lenv[1],Renv[Ns] = makeEnds(dualpsi,psi,mpo...;Lbound=Lbound,Rbound=Rbound)
  for i = 1:psi.oc-1
    Lupdate!(i,Lenv,dualpsi,psi,mpo...)
  end

  for i = Ns:-1:psi.oc+1
    Rupdate!(i,Renv,dualpsi,psi,mpo...)
  end
  return Lenv,Renv
end

"""
    makeEnv(psi,mpo[,Lbound=,Rbound=])

Generates environment tensors for a MPS (`psi`) with boundaries `Lbound` and `Rbound`
"""
function makeEnv(psi::MPS,mpo::MPO;Lbound::TensType=[0],Rbound::TensType=[0])
  return makeEnv(psi,psi,mpo,Lbound=Lbound,Rbound=Rbound)
end
export makeEnv





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
function makeMPO(H::Array{X,1},physSize::Array{Y,1},Ns::Integer;
                      lower::Bool=true,regtens::Bool=false) where {X <: Array, Y <: Integer}
  retType = typeof(prod(a->eltype(H[a])(1),1:Ns))
  finalMPO = Array{Array{retType,4},1}(undef,Ns)
  for i = 1:Ns
    thisH = lower ? H[i] : transpose(H[i])
    states = physSize[(i-1)%size(physSize,1) + 1]
    a1size = div(size(thisH,1),states) #represented in LEFT link indices
    a2size = div(size(thisH,2),states) #represented in RIGHT link indices
    P = eltype(thisH)

    currsize = [a1size,states,states,a2size]
    G = Array{P,4}(undef,currsize...)
    
    for l = 1:a1size
      for j = 1:states
        for k = 1:states
          @inbounds @simd for m = 1:a2size
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
  return MPO(finalMPO,regtens=regtens)
end

function makeMPO(H::Array{X,2},physSize::Array{Y,1},Ns::Integer;
                      lower::Bool=true,regtens::Bool=false) where {X <: Number, Y <: Integer}
  return makeMPO([H for i = 1:Ns],physSize,Ns)
end

function makeMPO(H::Array{X,2},physSize::Y,Ns::Integer;
                      lower::Bool=true,regtens::Bool=false) where {X <: Number, Y <: Integer}
  return makeMPO([H for i = 1:Ns],[physSize],Ns)
end

function makeMPO(H::Array{X,1},physSize::Y,Ns::Integer;
                      lower::Bool=true,regtens::Bool=false) where {X <: Array, Y <: Integer}
  return makeMPO(H,[physSize],Ns,lower=lower,regtens=regtens)
end

function makeMPO(H::Array{X,1},physSize::Y;
                      lower::Bool=true,regtens::Bool=false) where {X <: Array, Y <: Integer}
  return makeMPO(H,[physSize],length(H),lower=lower,regtens=regtens)
end

function makeMPO(H::Function,physSize::Array{X,1},Ns::Integer;
                      lower::Bool=true,regtens::Bool=false) where X <: Integer
  thisvec = [H(i) for i = 1:Ns]
  return makeMPO(thisvec,physSize,Ns,lower=lower,regtens=regtens)
end

function makeMPO(H::Function,physSize::Integer,Ns::Integer;
                      lower::Bool=true,regtens::Bool=false)
  return makeMPO(H,[physSize],Ns,lower=lower,regtens=regtens)
end
export makeMPO

"""
    makeMPS(vect,physInd,Ns[,oc=])

generates an MPS from a single vector (i.e., from exact diagonalization) for `Ns` sites and `physInd` size physical index at orthogonality center `oc`
"""
function makeMPS(vect::Array{W,1},physInd::Array{P,1};Ns::Integer=length(physInd),left2right::Bool=true,oc::Integer=left2right ? Ns : 1,regtens::Bool=false) where {W <: Number, P <: Integer}
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
  newQlabels = [Qlabels[w] for w = 1:Ns]
  return MPS(T,newQlabels,Ns,oc=oc)
end

function MPS(Qlabels::Array{Array{Q,1},1},Ns::Integer;oc::Integer=1,type::DataType=Float64) where Q <: Qnum
  physindvec = [length(Qlabels[(x-1) % length(Qlabels) + 1]) for i = 1:Ns]
  psi = MPS(physindvec,oc=oc,type=type)
  qpsi = makeqMPS(psi,Qlabels)
  return qpsi
end

function MPS(Qlabels::Array{Array{Q,1},1};oc::Integer=1,type::DataType=Float64) where Q <: Qnum
  return MPS(type,Qlabels,oc=oc)
end

function possibleQNs(QNsummary::Array{Q,1},w::Integer,physinds::Array{Array{Q,1},1},flux::Q,m::Integer) where Q <: Qnum
  maxQNrange = [QNsummary[q] for q = 1:length(QNsummary)]
  minQNrange = [QNsummary[q] for q = 1:length(QNsummary)]

  minQN = Q()
  maxQN = Q()
  for i = w+1:length(physinds)
    minQN += minimum(physinds[i])
    maxQN += maximum(physinds[i])
  end
  possibleQN = Array{Q,1}(undef,length(QNsummary))
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

function randMPS(Qlabels::Array{Array{Q,1},1},Ns::Integer;oc::Integer=1,m::Integer=1,type::DataType=Float64,flux::Q=Q()) where Q <: Qnum
  physinds = [Qlabels[(w-1) % length(Qlabels) + 1] for w = 1:Ns]

  storeQNs = Array{Array{Array{Q,1},1},1}(undef,Ns)
  nonzerointersect = true

  lastQNs = [inv(flux)]
  QNsummary2 = multi_indexsummary([physinds[end],lastQNs],[1,2])

  while nonzerointersect
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

"""
    assignflux!(i,mps,QnumMat,storeVal)

Assigns flux to the right link index on an MPS tensor

# Arguments
+ `i::intType`: current position
+ `mps::MPS`: MPS
+ `QnumMat::Array{Array{Qnum,1},1}`: quantum number matrix for the physical index
+ `storeVal::Array{T,1}`: maximum value found in MPS tensor, determine quantum number
"""
function assignflux!(i::Integer,mps::MPS,QnumMat::Array{Array{Q,1},1},storeVal::Array{Float64,1}) where {Q <: Qnum, W <: Number}
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
+ `warning::Bool`: Toggle warning message if last tensor has no values or is zero
"""
function makeqMPS(mps::MPS,Qlabels::Array{Array{Q,1},1},arrows::Array{Bool,1}...;newnorm::Bool=true,setflux::Bool=false,
                  flux::Q=Q(),randomize::Bool=true,override::Bool=true,silent::Bool=false,lastfluxzero::Bool=false)::MPS where Q <: Qnum
  if newnorm
    if !silent
      println("(makeqMPS: If converting from a non-QN MPS to a QN MPS, then beware that applying a 3S or adding noise in general to the MPS is not considered when calling makeqMPS)")
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
  finalMPS = matrixproductstate{typeof(QtensVec)}(QtensVec,mps.oc)

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
    makeqMPS(mps,Qlabels[,arrows,newnorm=,setflux=,flux=,randomize=,override=])

creates quantum number MPS from regular MPS according to `Qlabels`

# Arguments
+ `mps::MPS`: dense MPS
+ `Qlabels::Array{Qnum,1}`: quantum number labels on each physical index (uniform physical indices)
+ `arrows::Array{Bool,1}`: first entry of this tuple is the arrow convention on MPS tensors (default: [false,true,true])
+ `newnorm::Bool`: set new norm of the MPS tensor
+ `setflux::Bool`: toggle to force this to be the total flux of the MPS tensor
+ `flux::Qnum`: quantum number to force total MPS flux to be if setflux=true
+ `randomize::Bool`: randomize last tensor if flux forces a zero tensor
+ `silent::Bool`: Toggle warning message if last tensor has no values or is zero
"""
function makeqMPS(mps::MPS,Qlabels::Array{Q,1},arrows::Array{Bool,1}...;newnorm::Bool=true,setflux::Bool=false,
  flux::Q=Q(),randomize::Bool=true,override::Bool=true,lastfluxzero::Bool=false)::MPS where Q <: Qnum
  return makeqMPS(mps,[Qlabels],arrows...,newnorm=newnorm,setflux=setflux,flux=flux,randomize=randomize,override=override,lastfluxzero=lastfluxzero)
end

function makeqMPS(arr::Array,Qlabels::W,arrows::Array{Bool,1}...;oc::Integer=1,newnorm::Bool=true,setflux::Bool=false,
                  flux::Q=Q(),randomize::Bool=true,override::Bool=true,warning::Bool=true,lastfluxzero::Bool=false)::MPS where W <: Union{Array{Array{Q,1},1},Array{Q,1}} where Q <: Qnum
  mps = MPS(arr,oc)
  makeqMPS(mps,Qlabels,arrows...,newnorm=newnorm,setflux=setflux,flux=flux,randomize=randomize,override=override,warning=warning,lastfluxzero=lastfluxzero)
end
export makeqMPS

#       +---------------------------------------+
#>------+           convert to qMPO             +---------<
#       +---------------------------------------+

"""
    makeqMPO(mpo,Qlabels[,arrows])

Generates quantum number MPO from a dense Hamiltonian based on `Qlabels`

# Arguments:
+ `mpo::MPO`: dense MPO
+ `Qlabels::Array{Array{Qnum,1},1}`: quantum numbers for physical indices (modulus size of vector)
+ `arrows::Array{Bool,1}`: arrow convention for quantum numbers (default: [false,false,true,true])
"""
function makeqMPO(mpo::MPO,Qlabels::Array{Array{Q,1},1},arrows::Array{Bool,1}...;infinite::Bool=false,unitcell::Integer=1)::MPO where Q <: Qnum
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
    makeqMPO(mpo,Qlabels[,arrows])

Generates quantum number MPO from a dense Hamiltonian based on `Qlabels`

# Arguments:
+ `mpo::MPO`: dense MPO
+ `Qlabels::Array{Qnum,1}`: quantum numbers for physical indices (uniform)
+ `arrows::Array{Bool,1}`: arrow convention for quantum numbers (default: [false,false,true,true])
"""
function makeqMPO(mpo::MPO,Qlabels::Array{Q,1},arrows::Array{Bool,1}...;infinite::Bool=false,unitcell::Integer=1)::MPO where Q <: Qnum
  return makeqMPO(mpo,[Qlabels],arrows...,infinite=infinite,unitcell=unitcell)
end

function makeqMPO(arr::Array,Qlabels::W,arrows::Array{Bool,1}...;infinite::Bool=false,unitcell::Integer=1)::MPO where W <: Union{Array{Array{Q,1},1},Array{Q,1}} where Q <: Qnum
  mpo = makeMPO(arr,infinite=infinite)
  return makeqMPO(mpo,Qlabels,arrows...,infinite=infinite,unitcell=unitcell)
end
export makeqMPO
