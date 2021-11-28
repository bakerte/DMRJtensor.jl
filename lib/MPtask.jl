#########################################################################
#
#  Density Matrix Renormalization Group (and other methods) in julia (DMRjulia)
#                              v0.8
#
#########################################################################
# Made by Thomas E. Baker and M.P. Thompson (2020)
# See accompanying license with this program
# This code is native to the julia programming language (v1.5.4+)
#

"""
  Module: MPmaker

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
const Env = Union{AbstractArray,envType}
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
mutable struct matrixproductoperator{W} <: regMPO where W <: Array{TensType,1}
  H::W
end

"""
  `environment`

Construct this object through `Env`. Array that holds environment tensors

# Fields:
+ `V::Array{TensType,1}`: vector of environment tensors

See also: [`Env`](@ref)
"""
mutable struct environment{W} <: regEnv where W <: Array{TensType,1}
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
function environment(network::G...) where G <: Union{MPS,MPO}
  Ns = length(network[1])
  return environment(typeof(network[1]),Ns)
end

"""
  psi = MPS(A[,regtens=false,oc=1])

Constructs `psi` for MPS with tensors `A` (Array of tensors or MPS) with orthogonality center `oc`. `regtens` avoids conversion to `denstens` type (defaulted to perform the conversion for efficiency)
"""
function MPS(psi::Array{W,1};regtens::Bool=false,oc::Integer=1,type::DataType=eltype(psi[1])) where W <: densTensType
  if eltype(psi[1]) != type && !regtens
    MPSvec = [convertTens(type, copy(B[i])) for i = 1:size(B,1)]
  else
    MPSvec = psi
  end
  return matrixproductstate(MPSvec,oc) #MPS(psi,regtens=regtens,oc=oc,type=type)
end

function MPS(psi::MPS;regtens::Bool=false,oc::Integer=psi.oc,type::DataType=eltype(psi))
  return MPS(psi.A,regtens=regtens,oc=oc,type=type)
end

"""
  psi = MPS(A[,regtens=false,oc=1])

Constructs `psi` for MPS with tensors `A` (Array of tensors or MPS) with orthogonality center `oc`. `regtens` avoids conversion to `denstens` type (defaulted to perform the conversion for efficiency)
"""
function MPS(B::Array{W,1};regtens::Bool=false,oc::Integer=1,type::DataType=eltype(B[1])) where W <: AbstractArray
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
function MPS(type::DataType,B::Union{MPS,Array{W,1}};regtens::Bool=false,oc::Integer=1) where W <: AbstractArray
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
  psi = MPS(physindsize,Ns[,regtens=false,oc=1])

Constructs `psi` for MPS of tensor type Float64 by making empty tensors of size (1,`physindsize`,1) for w indexing 1:`Ns` (repeats on `physindvec`) with othrogonality center `oc`. `regtens` avoids conversion to `denstens` type (defaulted to perform the conversion for efficiency)
"""
function MPS(physindsize::Integer,Ns::Integer;regtens::Bool=false,oc::Integer=1,type::DataType=Float64)
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
  psi = applyOps!(psi,sites,Op[,trail=ones(1,1)])

Applies operator `Op` (any `TensType`) in-place to the MPS `psi` at sites `sites`, a vector of integers. A trailing operator `trail` can be applied if not the default.
"""
function applyOps!(psi::MPS,sites::Array{W,1},Op::TensType;trail::TensType=ones(1,1)) where W <: Integer
  for i = 1:length(sites)
    site = sites[i]
    p = site
    psi[p] = contract([2,1,3],Op,2,psi[p],2)
    if trail != ones(1,1)
      for j = 1:p-1
        psi[j] = contract([2,1,3],trail,2,psi[j],2)
      end
    end
  end
  return psi
end
export applyOps!

"""
  newpsi = applyOps(psi,sites,Op[,trail=ones(1,1)])

Applies operator `Op` (any `TensType`) to the MPS `psi` at sites `sites`, a vector of integers. A trailing operator `trail` can be applied if not the default.
"""
function applyOps(psi::MPS,sites::Array{W,1},Op::TensType;trail::TensType=ones(1,1)) where W <: Integer
  cpsi = copy(psi)
  return applyOps!(cpsi,sites,Op,trail=trail)
end
export applyOps

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
    psi = MPS(vec,oc)
  else
    Lsize,Rsize = 1,prod(w->physindvec[w],2:length(physindvec))
    accumsize = physindvec[1]
    for w = 1:Ns
      physindsize = physindvec[w]
      currRsize = Rsize < accumsize ? min(Rsize,m) : min(accumsize,m)
      vec[w] = rand(T,Lsize,physindsize,currRsize)
      vec[w] /= norm(vec[w])
      Lsize = currRsize
      Rsize = cld(Rsize,physindsize)
      accumsize *= physindsize
    end
    psi = MPS(vec,oc)
    psi[oc] /= expect(psi)
  end
  return psi
end

"""
  psi = randMPS(physindsize,Ns[,oc=1,m=1])

Generates MPS with data type Float64, uniform physical index size `physindsize`, with `Ns` sites, orthogonality center `oc`, and bond dimension `m`.
"""
function randMPS(physindsize::Integer,Ns::Integer;oc::Integer=1,m::Integer=1)
  return randMPS(Float64,physindsize,Ns,oc=oc,m=m)
end

"""
  psi = randMPS(T,physindvec,Ns[,oc=1,m=1])

Generates MPS with data type Float64, physical index size vector `physindvec` (repeating over `Ns` sites), with `Ns` sites, orthogonality center `oc`, and bond dimension `m`.
"""
function randMPS(physindvec::Array{W,1};oc::Integer=1,m::Integer=1) where W <: Integer
  return randMPS(Float64,physindvec,oc=oc,m=m)
end
export randMPS

"""
  mpo = MPO(H[,regtens=false])

constructor for MPO with tensors `H` either a vector of tensors or the MPO. `regtens` outputs with the julia Array type
"""
function MPO(H::Array{W,1};regtens::Bool=false) where W <: TensType
  T = prod(a->eltype(H[a])(1),1:size(H,1))
  if !regtens && (typeof(H[1]) <: AbstractArray)
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
  newH = Array{W,1}(undef,size(H,1))

  for a = 1:size(H,1)
    if ndims(H[a]) == 2
      rP = reshape(H[a],size(H[a],1),size(H[a],2),1,1)
      newH[a] = permutedims!(rP,[4,1,2,3])
    else
      newH[a] = H[a]
    end
  end

  if !regtens && (typeof(newH[1]) <: AbstractArray)
    finalH = [tens(newH[a]) for a = 1:length(newH)]
  else
    finalH = newH
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
function size(H::MPO)
  return size(H.H)
end

"""
  G = size(H,i)

`size` prints out the size tuple `G` of the tensor field of a `Env`, `MPS`, or `MPO`; this is effectively the number of sites
"""
function size(H::MPO,i::Integer)
  return size(H.H,i)
end

function size(psi::MPS)
  return size(psi.A)
end

function size(psi::MPS,i::Integer)
  return size(psi.A,i)
end

function size(G::regEnv)
  return size(G.V)
end

function size(G::regEnv,i::Integer)
  return size(G.V,i)
end

#import Base.length
"""
  G = length(H)

`length` prints out the number of entries of the input `H` being a `Env`, `MPS`, or `MPO`; this is effecitvely the number of sites
"""
function length(H::MPO)
  return length(H.H)
end

function length(psi::MPS)
  return length(psi.A)
end

function length(G::regEnv)
  return length(G.V)
end

#import Base.eltype
"""
  G = eltype(Y)

`eltype` gets element type `G` of the `Env`, `MPS`, or `MPO` tensor fields
"""
function eltype(Y::regMPS)
  return eltype(Y.A[1])
end

function eltype(H::regMPO)
  return eltype(H.H[1])
end

function eltype(G::regEnv)
  return eltype(G.V[1])
end

#import Base.getindex
"""
  G = getindex(A,i)

`getindex` allows to retrieve tensor `G` at position `i` from `Env`, `MPS` or `MPO`
"""
function getindex(A::regMPS,i::Integer)
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

function getindex(H::regMPO,i::Integer)
  return H.H[i]
end

function getindex(H::regMPO,r::UnitRange{W}) where W <: Integer
  return MPO(H.H[r])
end

function getindex(G::regEnv,i::Integer)
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
function lastindex(A::regMPS)
  return lastindex(A.A)
end

function lastindex(H::regMPO)
  return lastindex(H.H)
end

function lastindex(G::regEnv)
  return lastindex(G.V)
end


#import Base.setindex!
"""
  psi[i] = G

setindex! allows to assign elements `G` to an `Env`, `MPS`, or `MPO` at element `i`
"""
function setindex!(H::regMPO,A::TensType,i::intType)
  H.H[i] = A
  nothing
end
function setindex!(H::regMPS,A::TensType,i::intType)
  H.A[i] = A
  nothing
end

function setindex!(G::regEnv,A::TensType,i::intType)
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
  return MPS([copy(mps.A[i]) for i = 1:length(mps)],copy(mps.oc))
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

#       +------------------+
#>------|  Large MPS/MPO   |----------<
#       +------------------+

"""
  file_extension

A default file extension can be specified for the large types.

See also: [`largeMPS`](@ref) [`largeMPO`](@ref) [`largeEnv`](@ref)
"""
const file_extension = ".dmrjulia"

"""
  largeMPS

Abstract types for `largeMPS`; saves tensors to disk on setting values and retrieves them on calling an element; slower than other `regMPS` form

See also: [`largeMPO`](@ref) [`largeEnv`](@ref)
"""
abstract type largeMPS <: MPS end
export largeMPS

"""
  largeMPO
  
Abstract types for `largeMPO`; saves tensors to disk on setting values and retrieves them on calling an element; slower than other `regMPO` form

See also: [`largeEnv`](@ref) [`largeMPO`](@ref)
"""
abstract type largeMPO <: MPO end
export largeMPO

"""
  largeEnv
  
Abstract types for `largeEnv`; saves tensors to disk on setting values and retrieves them on calling an element; slower than other `regEnv` form

See also: [`largeMPS`](@ref) [`largeMPO`](@ref)
"""
abstract type largeEnv <: envType end
export largeEnv

"""
  largeType
  
A union of the types `largeMPS`, `largeMPO`, and `largeEnv`

See also: [`largeMPS`](@ref) [`largeMPO`](@ref) [`largeEnv`](@ref)
"""
const largeType = Union{largeMPS,largeMPO,largeEnv}
export largeType

"""
  `largematrixproductstate` 
  
Construct this container with `largeMPS`. struct to hold `largeMPS` tensors and orthogonality center

# Fields:
+ `A::Array{String,1}`: filenames where the tensors are stored on disk
+ `oc::Integer`: orthogonality center
+ `type::DataType`: DataType of the stored tensors

See also: [`largeMPS`](@ref)
"""
mutable struct largematrixproductstate <: largeMPS
  A::Array{String,1}
  oc::Integer
  type::DataType
end

"""
  `largematrixproductoperator` 
  
Construct this container with `largeMPO`. struct to hold `largeMPO` tensors

# Fields:
+ `H::Array{String,1}`: filenames where the tensors are stored on disk
+ `type::DataType`: DataType of the stored tensors

See also: [`largeMPO`](@ref)
"""
mutable struct largematrixproductoperator <: largeMPO
  H::Array{String,1}
  type::DataType
end

"""
  `largeenvironment` 
  
Construct this container with `largeEnv`. struct to hold `largeEnv` tensors

# Fields:
+ `V::Array{String,1}`: filenames where the tensors are stored on disk
+ `type::DataType`: DataType of the stored tensors

See also: [`largeEnv`](@ref)
"""
mutable struct largeenvironment <: largeEnv
  V::Array{String,1}
  type::DataType
end

"""
  tensor2disc(name,tensor[,ext=".dmrjulia"])

Writes `tensor` to disc with the Serialization package and filename `name`*`ext`

See also: [`tensorfromdisc`](@ref)
"""
function tensor2disc(name::String,tensor::TensType;ext::String=file_extension)
  Serialization.serialize(name*ext,tensor)
  nothing
end

"""
  A = tensorfromdisc(name[,ext=".dmrjulia"])

Reads tensor `A` from disc with the Serialization package and filename `name`*`ext`

See also: [`tensor2disc`](@ref)
"""
function tensorfromdisc(name::String;ext::String=file_extension)
  return Serialization.deserialize(name*ext)
end

"""
  G = largeMPS(psi[,label="mps_",names=[label*"i" for i = 1:length(psi)],ext=".dmrjulia"])

Writes tensors from `MPS` `psi` to hard disk as retrieved through `G` according to filenames specified in `names` (default file extension: `.dmrjulia`)

See also: [`largeMPO`](@ref) [`largeLenv`](@ref) [`largeRenv`](@ref) [`largeEnv`](@ref)
"""
function largeMPS(psi::MPS;label::String="mps_",names::Array{String,1}=[label*"$i" for i = 1:length(psi)],ext::String=file_extension)
  lastnum = 1
  for b = 1:length(psi)
    C = psi[b]
    tensor2disc(names[b],C,ext=ext)
    lastnum *= eltype(C)(1)
  end
  return largematrixproductstate(names,psi.oc,typeof(lastnum))
end

"""
  G = largeMPS(Ns[,label="mps_",names=[label*"i" for i = 1:length(psi)],ext=".dmrjulia"])

Creates a large MPS type (stored on disk) with `Ns` tensors (element type: Float64) but does not write anything to disk initially. Filenames specified in `names` (default file extension: `.dmrjulia`)

See also: [`largeMPO`](@ref) [`largeLenv`](@ref) [`largeRenv`](@ref) [`largeEnv`](@ref)
"""
function largeMPS(Ns::Integer;label::String="mps_",names::Array{String,1}=[label*"$i" for i = 1:Ns],ext::String=file_extension,thisoc::Integer=1,type::DataType=Float64)
  return largematrixproductstate(names,thisoc,type)
end

"""
  G = largeMPS(T,Ns[,label="mps_",names=[label*"i" for i = 1:length(psi)],ext=".dmrjulia"])

Creates a large MPS type (stored on disk) of element type `T` with `Ns` tensors but does not write anything to disk initially. Filenames specified in `names` (default file extension: `.dmrjulia`)

See also: [`largeMPO`](@ref) [`largeLenv`](@ref) [`largeRenv`](@ref) [`largeEnv`](@ref)
"""
function largeMPS(type::DataType,Ns::Integer;label::String="mps_",names::Array{String,1}=[label*"$i" for i = 1:Ns],ext::String=file_extension,thisoc::Integer=1)
  return largeMPS(Ns,label=label,names=names,ext=ext,thisoc=thisoc,type=type)
end

"""
  G = largeMPO(mpo[,label="mpo_",names=[label*"i" for i = 1:length(mpo)],ext=".dmrjulia"])

Writes tensors from `MPO` `mpo` to hard disk as retrieved through `G` according to filenames specified in `names` (default file extension: `.dmrjulia`)

See also: [`largeMPS`](@ref) [`largeLenv`](@ref) [`largeRenv`](@ref) [`largeEnv`](@ref)
"""
function largeMPO(mpo::P;label::String="mpo_",names::Array{String,1}=[label*"$i" for i = 1:length(mpo)],ext::String=file_extension) where P <: Union{AbstractArray,MPO}
  lastnum = 1
  for b = 1:length(mpo)
    C = mpo[b]
    tensor2disc(names[b],C,ext=ext)
    lastnum *= eltype(C)(1)
  end
  return largematrixproductoperator(names,typeof(lastnum))
end

"""
  G = largeMPO(Ns[,label="mpo_",names=[label*"i" for i = 1:length(mpo)],ext=".dmrjulia"])

Creates a large MPO type (stored on disk) with `Ns` tensors (element type: Float64) but does not write anything to disk initially. Filenames specified in `names` (default file extension: `.dmrjulia`)

See also: [`largeMPS`](@ref) [`largeLenv`](@ref) [`largeRenv`](@ref) [`largeEnv`](@ref)
"""
function largeMPO(Ns::Integer;label::String="mpo_",names::Array{String,1}=[label*"$i" for i = 1:Ns],ext::String=file_extension,type::DataType=Float64)
  return largematrixproductoperator(names,type)
end

"""
  G = largeMPO(T,Ns[,label="mpo_",names=[label*"i" for i = 1:length(mpo)],ext=".dmrjulia"])

Creates a large MPO type (stored on disk) of element type `T` with `Ns` tensors but does not write anything to disk initially. Filenames specified in `names` (default file extension: `.dmrjulia`)

See also: [`largeMPS`](@ref) [`largeLenv`](@ref) [`largeRenv`](@ref) [`largeEnv`](@ref)
"""
function largeMPO(type::DataType,Ns::Integer;label::String="mpo_",names::Array{String,1}=[label*"$i" for i = 1:Ns],ext::String=file_extension)
  return largeMPO(Ns,label=label,names=names,ext=ext,type=type)
end

"""
  G = largeLenv(lenv[,label="Lenv_",names=[label*"i" for i = 1:length(lenv)],ext=".dmrjulia"])

Writes tensors from environment `lenv` to hard disk as retrieved through `G` according to filenames specified in `names` (default file extension: `.dmrjulia`)

See also: [`largeMPS`](@ref) [`largeMPO`](@ref) [`largeRenv`](@ref) [`largeEnv`](@ref)
"""
function largeLenv(lenv::P;label::String="Lenv_",names::Array{String,1}=[label*"$i" for i = 1:length(lenv)],ext::String=file_extension) where P <: Union{AbstractArray,MPO}
  lastnum = 1
  for b = 1:length(lenv)
    C = lenv[b]
    tensor2disc(names[b],C,ext=ext)
    lastnum *= eltype(C)(1)
  end
  return largeenvironment(names,typeof(lastnum))
end

"""
  G = largeLenv(Ns[,label="Lenv_",names=[label*"i" for i = 1:Ns],ext=".dmrjulia"])

Creates a large environment type (stored on disk) with `Ns` tensors (element type: Float64) but does not write anything to disk initially. Filenames specified in `names` (default file extension: `.dmrjulia`)

See also: [`largeMPS`](@ref) [`largeMPO`](@ref) [`largeRenv`](@ref) [`largeEnv`](@ref)
"""
function largeLenv(Ns::Integer;label::String="Lenv_",names::Array{String,1}=[label*"$i" for i = 1:Ns],ext::String=file_extension,type::DataType=Float64)
  return largeenvironment(names,type)
end

"""
  G = largeLenv(T,Ns[,label="Lenv_",names=[label*"i" for i = 1:Ns],ext=".dmrjulia"])

Creates a large environment type (stored on disk) of element type `T` with `Ns` tensors but does not write anything to disk initially. Filenames specified in `names` (default file extension: `.dmrjulia`)

See also: [`largeMPS`](@ref) [`largeMPO`](@ref) [`largeRenv`](@ref) [`largeEnv`](@ref)
"""
function largeLenv(type::DataType,Ns::Integer;label::String="Lenv_",names::Array{String,1}=[label*"$i" for i = 1:Ns],ext::String=file_extension)
  return largeLenv(Ns,label=label,names=names,ext=ext,type=type)
end
export largeLenv

"""
  G = largeRenv(renv[,label="Renv_",names=[label*"i" for i = 1:length(renv)],ext=".dmrjulia"])

Writes tensors from environment `renv` to hard disk as retrieved through `G` according to filenames specified in `names` (default file extension: `.dmrjulia`)

See also: [`largeMPS`](@ref) [`largeMPO`](@ref) [`largeLenv`](@ref) [`largeEnv`](@ref)
"""
function largeRenv(renv::P;label::String="Renv_",names::Array{String,1}=[label*"$i" for i = 1:length(renv)],ext::String=file_extension) where P <: Union{AbstractArray,MPO}
  return largeLenv(renv,label=label,names=names,ext=ext)
end

"""
  G = largeRenv(Ns[,label="Renv_",names=[label*"i" for i = 1:Ns],ext=".dmrjulia"])

Creates a large environment type (stored on disk) with `Ns` tensors (element type: Float64) but does not write anything to disk initially. Filenames specified in `names` (default file extension: `.dmrjulia`)

See also: [`largeMPS`](@ref) [`largeMPO`](@ref) [`largeLenv`](@ref) [`largeEnv`](@ref)
"""
function largeRenv(Ns::Integer;label::String="Renv_",names::Array{String,1}=[label*"$i" for i = 1:Ns],ext::String=file_extension,type::DataType=Float64)
  return largeenvironment(names,type)
end

"""
  G = largeRenv(T,Ns[,label="Renv_",names=[label*"i" for i = 1:Ns],ext=".dmrjulia"])

Creates a large environment type (stored on disk) of element type `T` with `Ns` tensors but does not write anything to disk initially. Filenames specified in `names` (default file extension: `.dmrjulia`)

See also: [`largeMPS`](@ref) [`largeMPO`](@ref) [`largeLenv`](@ref) [`largeEnv`](@ref)
"""
function largeRenv(type::DataType,Ns::Integer;label::String="Renv_",names::Array{String,1}=[label*"$i" for i = 1:Ns],ext::String=file_extension)
  return largeRenv(Ns,label=label,names=names,ext=ext,type=type)
end
export largeRenv

"""
  G,K = largeEnv(lenv,renv[,Llabel="Lenv_",Lnames=[label*"i" for i = 1:length(psi)],Rlabel="Renv_",Rnames=[label*"i" for i = 1:length(renv)],ext=".dmrjulia",type=Float64])

Writes tensors from environments `lenv` and `renv` to hard disk as retrieved through `G` and `K` respectively according to filenames specified in `Lnames` and `Rnames` (default file extension: `.dmrjulia`)

See also: [`largeMPS`](@ref) [`largeMPO`](@ref) [`largeLenv`](@ref) [`largeRenv`](@ref)
"""
function largeEnv(lenv::P,renv::P;Llabel::String="Lenv_",Lnames::Array{String,1}=[Llabel*"$i" for i = 1:Ns],Rlabel::String="Renv_",Rnames::Array{String,1}=[Rlabel*"$i" for i = 1:Ns],ext::String=file_extension,type::DataType=Float64) where P <: Union{AbstractArray,MPO}
  return largeLenv(lenv,names=Lnames,type=type),largeRenv(renv,names=Rnames,type=type)
end

"""
  G,K = largeEnv(Ns[,Llabel="Lenv_",Lnames=[label*"i" for i = 1:Ns],Rlabel="Renv_",Rnames=[label*"i" for i = 1:Ns],ext=".dmrjulia",type=Float64])

Creates large environments with `Ns` tensors and retrieved through `G` and `K` respectively according to filenames specified in `Lnames` and `Rnames` (default file extension: `.dmrjulia`)

See also: [`largeMPS`](@ref) [`largeMPO`](@ref) [`largeLenv`](@ref) [`largeRenv`](@ref)
"""
function largeEnv(Ns::Integer;Llabel::String="Lenv_",Lnames::Array{String,1}=[Llabel*"$i" for i = 1:Ns],Rlabel::String="Renv_",Rnames::Array{String,1}=[Rlabel*"$i" for i = 1:Ns],ext::String=file_extension,type::DataType=Float64)
  return largeenvironment(Lnames,type),largeenvironment(Rnames,type)
end

"""
  G,K = largeEnv(T,Ns[,Llabel="Lenv_",Lnames=[label*"i" for i = 1:Ns],Rlabel="Renv_",Rnames=[label*"i" for i = 1:Ns],ext=".dmrjulia"])

Creates large environments with `Ns` tensors of type `T` and retrieved through `G` and `K` respectively according to filenames specified in `Lnames` and `Rnames` (default file extension: `.dmrjulia`)

See also: [`largeMPS`](@ref) [`largeMPO`](@ref) [`largeLenv`](@ref) [`largeRenv`](@ref)
"""
function largeEnv(type::DataType,Ns::Integer;Llabel::String="Lenv_",Lnames::Array{String,1}=[Llabel*"$i" for i = 1:Ns],Rlabel::String="Renv_",Rnames::Array{String,1}=[Rlabel*"$i" for i = 1:Ns],ext::String=file_extension)
  return largeEnv(Ns,Llabel=Llabel,Lnames=Lnames,Rlabel=Rlabel,Rnames=Rnames,ext=ext,type=type)
end
export largeEnv

#import Base.getindex
function getindex(A::largeMPS,i::Integer)
  return tensorfromdisc(A.A[i])
end

function getindex(A::largeMPO,i::Integer)
  return tensorfromdisc(A.H[i])
end

function getindex(A::largeEnv,i::Integer)
  return tensorfromdisc(A.V[i])
end

#import Base.setindex!
function setindex!(H::largeMPS,A::TensType,i::intType;ext::String=file_extension)
  tensor2disc(H.A[i],A,ext=ext)
  nothing
end

function setindex!(H::largeMPO,A::TensType,i::intType;ext::String=file_extension)
  tensor2disc(H.H[i],A,ext=ext)
  nothing
end

function setindex!(H::largeEnv,A::TensType,i::intType;ext::String=file_extension)
  tensor2disc(H.V[i],A,ext=ext)
  nothing
end

#import Base.lastindex
function lastindex(A::largeMPS;ext::String=file_extension)
  return tensorfromdisc(A.A[end],ext=ext)
end

function lastindex(H::largeMPO;ext::String=file_extension)
  return tensorfromdisc(H.H[end],ext=ext)
end

function lastindex(H::largeEnv;ext::String=file_extension)
  return tensorfromdisc(H.V[end],ext=ext)
end

function length(A::largeMPS)
  return length(A.A)
end

function length(H::largeMPO)
  return length(H.H)
end

function length(H::largeEnv)
  return length(H.V)
end

function eltype(op::largeType)
  return op.type
end

"""
  G = loadMPS(Ns[,label="mps_",names=[label*"i" for i = 1:Ns],ext=".dmrjulia"])

If MPS tensors are stored on hard disk, then they can be retrieved by using `loadMPS`
"""
function loadMPS(Ns::Integer;label::String="mps_",names::Array{String,1}=[label*"$i" for i = 1:Ns],ext::String=file_extension,oc::Integer=0)
  lastnum = 1
  storeoc = [1.]
  for i = 1:Ns
    name = names[i]
    A = tensorfromdisc(name,ext=ext)
    lastnum *= eltype(A)(1)
    if isapprox(norm(A),1.)
      storeoc[1] = i
    end
  end
  thistype = typeof(lastnum)
  if oc == 0
    thisoc = storeoc[1]
  else
#    @assert(storeoc[1] == oc)
    thisoc = oc
  end
  return largematrixproductstate(names,thisoc,thistype)
end
export loadMPS

"""
  G = loadMPO(Ns[,label="mpo_",names=[label*"i" for i = 1:Ns],ext=".dmrjulia"])

If MPO tensors are stored on hard disk, then they can be retrieved by using `loadMPO`
"""
function loadMPO(Ns::Integer;label::String="mpo_",names::Array{String,1}=[label*"$i" for i = 1:Ns],ext::String=file_extension,oc::Integer=0)
  lastnum = 1
  for i = 1:Ns
    name = names[i]
    A = tensorfromdisc(name,ext=ext)
    lastnum *= eltype(A)(1)
  end
  thistype = typeof(lastnum)
  return largematrixproductoperator(names,thistype)
end
export loadMPO

"""
  G = loadLenv(Ns[,label="Lenv_",names=[label*"i" for i = 1:Ns],ext=".dmrjulia"])

If left environment tensors are stored on hard disk, then they can be retrieved by using `loadLenv`
"""
function loadLenv(Ns::Integer;label::String="Lenv_",names::Array{String,1}=[label*"$i" for i = 1:Ns],ext::String=file_extension,oc::Integer=0)
  lastnum = 1
  for i = 1:Ns
    name = names[i]
    A = tensorfromdisc(name,ext=ext)
    lastnum *= eltype(A)(1)
  end
  thistype = typeof(lastnum)
  return largeenvironment(names,thistype)
end
export loadLenv

"""
  G = loadRenv(Ns[,label="Renv_",names=[label*"i" for i = 1:Ns],ext=".dmrjulia"])

If right environment tensors are stored on hard disk, then they can be retrieved by using `loadRenv`
"""
function loadRenv(Ns::Integer;label::String="Renv_",names::Array{String,1}=[label*"$i" for i = 1:Ns],ext::String=file_extension,oc::Integer=0)
  lastnum = 1
  for i = 1:Ns
    name = names[i]
    A = tensorfromdisc(name,ext=ext)
    lastnum *= eltype(A)(1)
  end
  thistype = typeof(lastnum)
  return largeenvironment(names,thistype)
end
export loadRenv

"""
  G,K = loadEnv(Ns[,Llabel="Lenv_",Lnames=[label*"i" for i = 1:Ns],Rlabel="Renv_",Rnames=[label*"i" for i = 1:Ns],ext=".dmrjulia"])

If environment tensors are stored on hard disk, then they can be retrieved by using `loadEnv`
"""
function loadEnv(Ns::Integer;Llabel::String="Lenv_",Lnames::Array{String,1}=[Llabel*"$i" for i = 1:Ns],Rlabel::String="Renv_",Rnames::Array{String,1}=[Rlabel*"$i" for i = 1:Ns],ext::String=file_extension,oc::Integer=0)
  lastnumL = 1
  lastnumR = 1
  for i = 1:Ns
    name = Lnames[i]
    A = tensorfromdisc(name,ext=ext)
    lastnumL *= eltype(A)(1)

    B = tensorfromdisc(name,ext=ext)
    lastnumR *= eltype(B)(1)
  end
  thistypeL = typeof(lastnumL)
  thistypeR = typeof(lastnumR)
  return largeenvironment(Lnames,thistypeL),largeenvironment(Rnames,thistypeR)
end
export loadEnv

"""
  G = copy(names,X[,ext=".dmrjulia",copyext=ext])

Copies tensors from `largeMPS` input `X` to a new tensor with a vector of strings `names` representing the new filenames
"""
function copy(names::Array{String,1},X::largeMPS;ext::String=file_extension,copyext::String=ext)
  newObj = copy(X)
  newObj.A = names
  for i = 1:length(X)
    Y = tensorfromdisc(names[i],ext=ext)
    tensor2disc(X.A[i],Y,ext=copyext)
  end
  return newObj
end

"""
  G = copy(names,X[,ext=".dmrjulia",copyext=ext])

Copies tensors from `largeMPO` input `X` to a new tensor with a vector of strings `names` representing the new filenames
"""
function copy(names::Array{String,1},X::largeMPO;ext::String=file_extension,copyext::String=ext)
  newObj = copy(X)
  newObj.H = names
  for i = 1:length(X)
    Y = tensorfromdisc(names[i],ext=ext)
    tensor2disc(X.H[i],Y,ext=copyext)
  end
  return newObj
end

"""
  G = copy(names,X[,ext=".dmrjulia",copyext=ext])

Copies tensors from `largeEnv` input `X` to a new tensor with a vector of strings `names` representing the new filenames
"""
function copy(names::Array{String,1},X::largeEnv;ext::String=file_extension,copyext::String=ext)
  newObj = copy(X)
  newObj.V = names
  for i = 1:length(X)
    Y = tensorfromdisc(names[i],ext=ext)
    tensor2disc(X.V[i],Y,ext=copyext)
  end
  return newObj
end


#       +---------------------------------------+
#>------+  move! orthogonality center in MPS    +---------<
#       +---------------------------------------+

"""
  D,truncerr = moveR(Lpsi,Rpsi[,cutoff=,m=,minm=,condition=])

moves `psi`'s orthogonality center from `Lpsi` to `Rpsi`, with a maximum bond dimenion `m`, cutoff `cutoff`, minimum bond dimension `minm`; toggle truncation of tensor during movement with `condition`; returns psi[iL],psi[iR],D from the SVD, and the truncation error

See also: [`moveR!`](@ref)
"""
function moveR(Lpsi::TensType,Rpsi::TensType;cutoff::Float64=0.,m::Integer=0,minm::Integer=0,condition::Bool=false,mag::Number=0.,
                recursive::Bool=false,fast::Bool=false)
  if (size(Lpsi,3) <= m) && !isapprox(cutoff,0.) && fast
    Ltens,modV = qr(Lpsi,[[1,2],[3]])

    DV = (condition ? getindex!(modV,:,1:size(Rpsi,1)) : modV)
    D = DV
    truncerr = 0.
  else
    Ltens,D,V,truncerr,sumD = svd(Lpsi,[[1,2],[3]],cutoff=cutoff,m=m,minm=minm,recursive=recursive,mag=mag)      
    modV = (condition ? getindex!(V,:,1:size(Rpsi,1)) : V)
    DV = contract(D,[2],modV,[1])
  end
  Rtens = contract(DV,2,Rpsi,1)
  return Ltens,Rtens,D,truncerr
end
export moveR

"""
  D,truncerr = moveR!(psi[,cutoff=,m=,minm=,condition=])

acts in-place to move `psi`'s orthogonality center from `psi.oc` to `psi.oc+1`, with a maximum bond dimenion `m`, cutoff `cutoff`, and minimum bond dimension `minm`; toggle truncation of tensor during movement with `condition`

See also: [`moveR`](@ref)
"""
function moveR!(psi::MPS;cutoff::Float64=0.,m::Integer=0,minm::Integer=0,condition::Bool=false,
                  recursive::Bool=false,mag::Number=0.)
  iL = psi.oc
  psi[iL],psi[iL+1],D,truncerr = moveR(psi[iL],psi[iL+1],cutoff=cutoff,m=m,minm=minm,condition=condition,recursive=recursive,mag=mag)
  psi.oc += 1
  return D,truncerr
end
export moveR!

"""
  D,truncerr = moveL(Lpsi,Rpsi[,cutoff=,m=,minm=,condition=])

moves `psi`'s orthogonality center from `psi.oc` to `psi.oc-1`, with a maximum bond dimenion `m`, cutoff `cutoff`, minimum bond dimension `minm`; toggle truncation of tensor during movement with `condition`; returns psi[iL],psi[iR],D from the SVD, and the truncation error

See also: [`moveL!`](@ref)
"""
function moveL(Lpsi::TensType,Rpsi::TensType;cutoff::Float64=0.,m::Integer=0,minm::Integer=0,condition::Bool=false,mag::Number=0.,
                recursive::Bool=false,fast::Bool=false)
  if (size(Rpsi,1) <= m) && !isapprox(cutoff,0.) && fast
    modU,Rtens = lq(Rpsi,[[1],[2,3]])

    UD = (condition ? getindex!(modU,1:size(Lpsi,3),:) : modU)
    D = UD
    truncerr = 0.
  else
    U,D,Rtens,truncerr,sumD = svd(Rpsi,[[1],[2,3]],cutoff=cutoff,m=m,minm=minm,recursive=recursive,mag=mag)
    modU = (condition ? getindex!(U,1:size(Lpsi,3),:) : U)
    UD = contract(modU,[2],D,[1])
  end
  Ltens = contract(Lpsi,3,UD,1)
  return Ltens,Rtens,D,truncerr
end
export moveL

"""
    moveL!(psi[,cutoff=,m=,minm=,condition=])

acts in-place to move `psi`'s orthogonality center from `iR` to `iL`, with a maximum bond dimenion `m`, cutoff `cutoff`, and minimum bond dimension `minm`; toggle truncation of tensor during movement with `condition`

See also: [`moveL`](@ref)
"""
function moveL!(psi::MPS;cutoff::Float64=0.,m::Integer=0,minm::Integer=0,condition::Bool=false,
                  recursive::Bool=false,mag::Number=0.)
  iR = psi.oc
  psi[iR-1],psi[iR],D,truncerr = moveL(psi[iR-1],psi[iR],cutoff=cutoff,m=m,minm=minm,condition=condition,recursive=recursive,mag=mag)
  psi.oc -= 1
  return D,truncerr
end
export moveL!

"""
    movecenter!(psi,newoc[,m=,cutoff=,minm=,Lfct=,Rfct=])

movement function to move `psi` to a new site, `newoc` with `Lfct` and `Rfct`, with a maximum bond dimenion `m`, cutoff `cutoff`, and minimum bond dimension `minm`; toggle truncation of tensor during movement with `condition`

See also: [`move!`](@ref) [`move`](@ref)
"""
function movecenter!(psi::MPS,pos::Integer;cutoff::Float64=1E-14,m::Integer=0,minm::Integer=0,
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
function move!(mps::MPS,pos::Integer;m::Integer=0,cutoff::Float64=1E-14,minm::Integer=0)
  movecenter!(mps,pos,cutoff=cutoff,m=m,minm=minm)
  nothing
end
export move!

"""
    move(psi,newoc[,m=,cutoff=,minm=])

same as `move!` but makes a copy of `psi`

See also: [`move!`](@ref)
"""
function move(mps::MPS,pos::Integer;m::Integer=0,cutoff::Float64=1E-14,minm::Integer=0)
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
function  Lupdate(Lenv::TensType,dualpsi::TensType,psi::TensType,mpo::TensType...)
  nMPOs = length(mpo)
  nLsize = nMPOs+2
  tempLenv = contractc(Lenv,1,dualpsi,1)
  for j = 1:nMPOs
    tempLenv = contract(tempLenv,[1,nLsize],mpo[j],[1,3])
  end
  return contract(tempLenv,[1,nLsize],psi,[1,2])
end
export Lupdate

"""
    Rupdate(Renv,dualpsi,psi,mpo)

Updates right environment tensor `Renv` with dual MPS `dualpsi`, MPS `psi`, and MPO `mpo`
"""
function  Rupdate(Renv::TensType,dualpsi::TensType,psi::TensType,mpo::TensType...)
  nMPOs = length(mpo)
  nRsize = nMPOs+2
  tempRenv = contract(Renv,1,psi,3)
  for j = 1:nMPOs
    tempRenv = contract(tempRenv,[nRsize+1,1],mpo[j],[2,4])
  end
  return contractc(tempRenv,[nRsize+1,1],dualpsi,[2,3])
end
export Rupdate

"""
    Lupdate!(i,Lenv,psi,dualpsi,mpo)

Updates left environment's (`Lenv`) `i`+1 site from info on the site `i` based on dual MPS (`dualpsi`), MPS (`psi), and MPO (`mpo`)
"""
function Lupdate!(i::Integer,Lenv::Env,dualpsi::MPS,psi::MPS,mpo::MPO...)
  Lenv[i+1] = Lupdate(Lenv[i],dualpsi[i],psi[i],[mpo[a][i] for a = 1:length(mpo)]...)
  nothing
end

"""
    Lupdate!(i,Lenv,psi,mpo)

Updates left environment's (`Lenv`) `i`+1 site from info on the site `i` based on MPS (`psi) and MPO (`mpo`)
"""
function Lupdate!(i::Integer,Lenv::Env,psi::MPS,mpo::MPO...)
  Lupdate!(i,Lenv,psi,psi,mpo...)
end
export Lupdate!

"""
    Rupdate!(i,Renv,dualpsi,psi,mpo)

Updates right environment's (`Renv`) `i`-1 site from info on the site `i` based on dual MPS (`dualpsi`), MPS (`psi), and MPO (`mpo`)
"""
function Rupdate!(i::Integer,Renv::Env,dualpsi::MPS,psi::MPS,mpo::MPO...)
  Renv[i-1] = Rupdate(Renv[i],dualpsi[i],psi[i],[mpo[a][i] for a = 1:length(mpo)]...)
  nothing
end

"""
    Rupdate!(i,Renv,psi,mpo)

Updates right environment's (`Renv`) `i`-1 site from info on the site `i` based on MPS (`psi) and MPO (`mpo`)
"""
function Rupdate!(i::Integer,Renv::Env,psi::MPS,mpo::MPO...)
  Rupdate!(i,Renv,psi,psi,mpo...)
end
export Rupdate!

"""
    boundaryMove!(psi,i,mpo,Lenv,Renv)

Move orthogonality center of `psi` to site `i` and updates left and right environments `Lenv` and `Renv` using MPO `mpo`

See also: [`move!`](@ref)
"""
function boundaryMove!(psi::MPS,i::Integer,Lenv::Env,
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
function boundaryMove!(dualpsi::MPS,psi::MPS,i::Integer,Lenv::Env,
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
function boundaryMove(dualpsi::MPS,psi::MPS,i::Integer,mpo::MPO,Lenv::Env,Renv::Env)
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
function boundaryMove(psi::MPS,i::Integer,mpo::MPO,Lenv::Env,Renv::Env)
  newpsi = copy(psi)
  newLenv = copy(Lenv)
  newRenv = copy(Renv)
  boundaryMove!(newpsi,newpsi,i,mpo,newLenv,newRenv)
  return newpsi,newLenv,newRenv
end
export boundaryMove

#       +---------------------------------------+
#>------+       measurement operations          +---------<
#       +---------------------------------------+


"""
    applyMPO(psi,H[,m=,cutoff=])

Applies MPO (`H`) to an MPS (`psi`) and truncates to maximum bond dimension `m` and cutoff `cutoff`
"""
function applyMPO(psi::MPS,H::MPO;m::Integer=0,cutoff::Float64=0.)::MPS
  if m == 0
    m = maximum([size(psi[i],ndims(psi[i])) for i = 1:size(psi.A,1)])
  end

  thissize = size(psi,1)
  newpsi = [contract([1,3,4,2,5],psi[i],2,H[i],2) for i = 1:thissize]

  finalpsi = Array{typeof(psi[1]),1}(undef,thissize)
  finalpsi[thissize] = reshape!(newpsi[thissize],[[1],[2],[3],[4,5]],merge=true)

  for i = thissize:-1:2
    currTens = finalpsi[i]
    newsize = size(currTens)
    
    temp = reshape!(currTens,[[1,2],[3,4]])
    U,D,V = svd(temp,m = m,cutoff=cutoff)
    finalpsi[i] = reshape!(V,size(D,1),newsize[3],newsize[4],merge=true)
    tempT = contract(U,2,D,1)
    
    finalpsi[i-1] = contract(newpsi[i-1],[4,5],tempT,1)
  end
  finalpsi[1] = reshape!(finalpsi[1],[[1,2],[3],[4]],merge=true)
  return MPS(finalpsi,1)
end
export applyMPO


"""
    expect(dualpsi,psi,H[,Lbound=,Rbound=,order=])

evaluate <`dualpsi`|`H`|`psi`> for any number of MPOs `H`; can specifcy left and right boundaries (`Lbound` and `Rbound`); `dualpsi` is conjugated inside of the algorithm

See also: [`overlap`](@ref)
"""
function expect(dualpsi::MPS,psi::MPS,H::MPO...;Lbound::TensType=typeof(psi[1])(),Rbound::TensType=Lbound,order::intvecType=intType[])
  Ns = size(psi,1)
  nMPOs = size(H,1)
  nLsize = nMPOs+2
  Lenv,Renv = makeEnds(dualpsi,psi,H...,Lbound=Lbound,Rbound=Rbound)

  for i = 1:length(psi)
    Lenv = contractc(Lenv,1,dualpsi[i],1)
    for j = 1:nMPOs
      Lenv = contract(Lenv,(1,nLsize),H[j][i],(1,3))
    end
    Lenv = contract(Lenv,(1,nLsize),psi[i],(1,2))
  end

  if order == intType[]
    permvec = vcat([ndims(Renv)],[i for i = 2:ndims(Renv)-1],[1])
    modRenv = permutedims(Renv,permvec)
  else
    modRenv = Renv
  end
  
  return contract(Lenv,modRenv)
end

"""
    expect(psi,H[,Lbound=,Rbound=,order=])

evaluate <`psi`|`H`|`psi`> for any number of MPOs `H`; can specifcy left and right boundaries (`Lbound` and `Rbound`)

See also: [`overlap`](@ref)
"""
function expect(psi::MPS,H::MPO...;Lbound::TensType=typeof(psi[1])(),Rbound::TensType=Lbound,order::intvecType=intType[])
  return expect(psi,psi,H...,Lbound=Lbound,Rbound=Rbound,order=order)
end
export expect

"""
    correlationmatrix(dualpsi,psi,Cc,Ca[,F,silent=])

Compute the correlation funciton (example, <`dualpsi`|`Cc`_i . `Ca`_j|`psi`>) on all sites; can also specify a trail `F` for Fermion strings; `silent` toggles output of every line

# Note:
+ More efficient than using `mpoterm`s
+ Use `mpoterm` and `applyMPO` for higher order correlation functions or write a new function
"""
function correlationmatrix(dualpsi::MPS, psi::MPS, mCc::TensType, mCa::TensType, F::TensType...;silent::Bool = true)
  if typeof(mCc) <: Array && eltype(psi) <: denstens
    Cc = tens(mCc)
  else
    Cc = mCc
  end
  if typeof(mCa) <: Array && eltype(psi) <: denstens
    Ca = tens(mCa)
  else
    Ca = mCa
  end
  rho = Array{eltype(psi[1]),2}(undef,size(psi,1),size(psi,1))
  onsite = contract(Cc,2,Ca,1)
  if size(F,1) != 0
    FCc = contract(Cc,2,F[1],1)
  else
    FCc = Cc
  end
  diffTensors = !(psi == dualpsi)
  for i = 1:size(psi,1)
    move!(psi,i)
    if diffTensors
      move!(dualpsi,i)
    end
    TopTerm = contract([1,3,2],psi[i],[2],onsite,[1])
    rho[i,i] = contractc(TopTerm,dualpsi[i])
  end
  for i = 1:size(psi,1)-1
    move!(psi,i)
    if diffTensors
      move!(dualpsi,i)
    end
    TopTerm = contract(psi[i],[2],FCc,[1])
    Lenv = contractc(TopTerm,[1,3],dualpsi[i],[1,2])
    for j = i+1:size(psi,1)
      Renv = contract(psi[j],[2],Ca,[1])
      Renv = contractc(Renv,[2,3],dualpsi[j],[3,2])
      DMElement = contract(Lenv,Renv)
      if j < size(psi,1)
        if size(F,1) != 0
          Lenv = contract(Lenv,1,psi[j],1)
          Lenv = contract(Lenv,2,F[1],1)
          Lenv = contractc(Lenv,[1,3],dualpsi[j],[1,2])
        else
          Lenv = contract(Lenv, 1, psi[j], 1)
          Lenv = contractc(Lenv, [1,2], dualpsi[j], [1,2])
        end
      end
      rho[i,j] = DMElement
      rho[j,i] = conj(DMElement)
      if !silent
        println("Printing element: ",i," , ",j," ",DMElement)
      end
    end
  end
  return rho
end

"""
    correlationmatrix(psi,Cc,Ca[,F,silent=])

Compute the correlation funciton (example, <`psi`|`Cc`_i . `Ca`_j|`psi`>) on all sites; can also specify a trail `F` for Fermion strings; `silent` toggles output of every line

# Example:
```julia
Cup, Cdn, Nup, Ndn, Ndens, F, O, Id = fermionOps()
rho = correlationmatrix(psi,Cup',Cup,F) #density matrix
```
"""
function correlationmatrix(psi::MPS, Cc::TensType, Ca::TensType, F::TensType...;silent::Bool = true)
  return correlationmatrix(psi,psi,Cc,Ca,F...,silent=silent)
end
export correlationmatrix



























































"""
  operator_in_order!(pos,sizes)

Increments elements of input vector `pos` with sizes of a tensor `sizes` such that `pos` all elements are ordered least to greatest.  For use in `correlation` function.
"""
function operator_in_order!(pos::Array{G,1},sizes::intvecType) where G <: Integer
  w = length(pos)
  pos[w] += 1
  while w > 1 && pos[w] > sizes[w]
    w -= 1
    @inbounds pos[w] += 1
    @simd for x = w:length(pos)-1
      @inbounds pos[x+1] = pos[x]
    end
  end
  nothing
end

#heap algorithm for permutations (non-recursive)...
"""
  G = permutations(nelem)

Heap algorithm for finding permutations of `nelem` elements. Output `G` is an Vector of all permutations stored as vectors.  For example, a permutation of [1,2,3,4] is [2,1,3,4].
"""
function permutations(nelem::Integer)
  vec = [i for i = 1:nelem]
  numvecs = factorial(nelem)
  storevecs = Array{Array{intType,1},1}(undef,numvecs)
  saveind = zeros(intType,nelem)
  i = 0
  counter = 1
  storevecs[1] = copy(vec)
  while i < nelem
    if saveind[i+1] < i
      if i % 2 == 0
        a,b = 0,i
      else
        a,b = saveind[i+1],i
      end
      vec[a+1],vec[b+1] = vec[b+1],vec[a+1]
      
      counter += 1
      storevecs[counter] = copy(vec)

      saveind[i+1] += 1
      i = 0
    else
      saveind[i+1] = 0
      i += 1
    end
  end
  return storevecs
end

"""
  correlation(dualpsi,psi,inputoperators...[,sites=ntuple(i->1,length(inputoperators)),trail=()])

Computes the general correlation on the input MPS `psi` with operators defined in the tuple `inputoperators`. The `sites` command specifies which sites to allow for the operators to be applied (i.e., some operators can skip some sites) and the `trail` option allows for a trailing operator like the Fermion sign operator.

See also: [`expect`](@ref) [`correlationmatrix`](@ref)
"""
function correlation(dualpsi::MPS, psi::MPS, inputoperators::S...;
	sites::intvecType=ntuple(i->1:length(psi),length(inputoperators)),
	trail::Tuple=()) where S <: Union{Array{TensType,1},TensType}

  operators = Array{Array{typeof(psi[1]),1},1}(undef,length(inputoperators))
  lengthops = Array{intType,1}(undef,length(operators))
  for k = 1:length(operators)
    if eltype(inputoperators[k]) <: TensType
      operators[k] = inputoperators[k]
      lengthops[k] = length(operators[k])
    else
      operators[k] = [inputoperators[k]]
      lengthops[k] = 1
    end
  end

  Ns = length(psi)
  maxOplength = maximum(lengthops)
  retType = typeof(eltype(dualpsi[1])(1) * eltype(psi[1])(1) * prod(w->prod(a->eltype(operators[w][a])(1),1:lengthops[w]),1:length(operators)))

  base_sizes = [length(sites[i]) - (lengthops[i] - 1) for i = 1:length(sites)]

  omega = zeros(retType,length(operators)) #Array{retType,length(operators)}(undef,base_sizes...)

  perm = permutations(length(operators))

  move!(psi,1)
  move!(dualpsi,1)

  Lenv,Renv = makeEnv(dualpsi,psi)
  for b = 1:length(Renv)
    Renv[b] = permutedims(Renv[b],[2,1])
  end

  isId = [true for r = 1:length(operators)]
  if length(trail) > 0
    for r = 1:length(isId)
      index = 0
      while isId[r]
        index += 1
        isId[r] = searchindex(bigtrail,index,index) == 1
      end
    end
  end

  for i = 1:length(perm)

    order = perm[i]

    base_pos = ones(intType,length(operators))

    pos = [sites[1][1] for i = 1:length(operators)]
    prevpos = [sites[1][1] for i = 1:length(operators)]

    while sum(base_sizes - pos) >= 0
      startsite = 1
      while startsite < length(pos) && pos[startsite] == prevpos[startsite]
        startsite += 1
      end

      while startsite > 1 && pos[startsite-1] == prevpos[startsite]
        startsite -= 1
      end

      beginsite = prevpos[startsite]
      finalsite = pos[end]

      thisLenv = Lenv[beginsite]

      for w = beginsite:finalsite
        newpsi = psi[w]
        for g = 1:length(pos)
          opdist = w - pos[g]
          if 0 <= opdist < lengthops[g]
            newpsi = contract([2,1,3],operators[order[g]][opdist + 1],2,newpsi,2)
          end
        end
        for r = 1:length(pos)
          if  w < pos[r] && !isId[r]
            newpsi = contract([2,1,3],trail[r],2,newpsi,2)
          end
        end
        thisLenv = Lupdate(thisLenv,dualpsi[w],newpsi)
        if w < Ns
          Lenv[w+1] = thisLenv
        end
      end

      thisRenv = Renv[finalsite]
      res = contract(thisLenv,thisRenv)
      finalpos = pos[order]
      
      @inbounds omega[finalpos...] = res

      @simd for b = 1:length(pos)
        @inbounds prevpos[b] = pos[b]
      end
      operator_in_order!(base_pos,base_sizes)
      @simd for b = 1:length(pos)
        @inbounds pos[b] = sites[b][base_pos[b]]
      end
    end

  end
  return omega
end
export correlation

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
    qind[1] = inv(getQnum(index,1,dualpsi[site]))
    qind[end] = copy(getQnum(index,1,psi[site]))
    for i = 1:length(mpovec)
      index = left ? 1 : ndims(mpovec[i][Ns])
      qind[i+1] = copy(getQnum(index,1,mpovec[i][site]))
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
  if abs(norm(Lbound)) <= 1E-13
    Lout = makeBoundary(dualpsi,psi,mpovec...)
  else
    Lout = copy(Lbound)
  end
  if abs(norm(Rbound)) <= 1E-13
    Rout = makeBoundary(dualpsi,psi,mpovec...,left=false)
  else
    Rout = copy(Rbound)
  end
  return Lout,Rout
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

"""
    makeEnv(dualpsi,psi,mpo[,Lbound=,Rbound=])

Generates environment tensors for a MPS (`psi` and its dual `dualpsi`) with boundaries `Lbound` and `Rbound`
"""
function makeEnv(dualpsi::MPS,psi::MPS,mpo::MPO...;Lbound::TensType=typeof(psi[1])(),
          Rbound::TensType=typeof(psi[1])())
  Ns = length(psi)
  numtype = elnumtype(dualpsi,psi,mpo...)
  C = psi[1]

  if typeof(psi) <: largeMPS || typeof(mpo) <: largeMPO
    Lenv,Renv = largeEnv(numtype,Ns)
  else
    Lenv = environment(psi[1],Ns)
    Renv = environment(psi[1],Ns)
  end
  Lenv[1],Renv[Ns] = makeEnds(dualpsi,psi,mpo...,Lbound=Lbound,Rbound=Rbound)

  for i = Ns:-1:psi.oc+1
    Rupdate!(i,Renv,dualpsi,psi,mpo...)
  end
  for i = 1:psi.oc-1
    Lupdate!(i,Lenv,dualpsi,psi,mpo...)
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
                      lower::Bool=true,regtens::Bool=false) where {X <: AbstractArray, Y <: Integer}
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
          @simd for m = 1:a2size
            @inbounds G[l,j,k,m] = thisH[j + (l-1)*states, k + (m-1)*states]
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
                      lower::Bool=true,regtens::Bool=false) where {X <: AbstractArray, Y <: Integer}
  return makeMPO(H,[physSize],Ns,lower=lower,regtens=regtens)
end

function makeMPO(H::Array{X,1},physSize::Y;
                      lower::Bool=true,regtens::Bool=false) where {X <: AbstractArray, Y <: Integer}
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
      temp = reshape(U,Lindsize,physInd[i],size(DV,1))
      mps[i] = temp

      Lindsize = size(DV,1)
      if i == Ns-1
        temp = unreshape(DV,Lindsize,physInd[i+1],1)
        mps[Ns] = temp
      else
        Rsize = cld(size(M,2),physInd[i+1]) #integer division, round up
        M = unreshape(DV,size(DV,1)*physInd[i+1],Rsize)
      end
    end
    finalmps = MPS(mps,Ns,regtens=regtens)
  else
    M = reshape(vect, div(length(vect),physInd[end]), physInd[end])
    Rindsize = 1 #current size of the right index
    for i=Ns:-1:2
      UD,V = lq(M)
      temp = reshape(V,size(UD,2),physInd[i],Rindsize)
      mps[i] = temp
      Rindsize = size(UD,2)
      if i == 2
        temp = unreshape(UD,1,physInd[i-1],Rindsize)
        mps[1] = temp
      else
        Rsize = cld(size(M,1),physInd[i-1]) #integer division, round up
        M = unreshape(UD,Rsize,size(UD,2)*physInd[i-1])
      end
    end
    finalmps = MPS(mps,1,regtens=regtens)
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






















#       +--------------------------------+
#>------+    Methods for excitations     +---------<
#       +--------------------------------+

"""
    penalty!(mpo,lambda,psi[,compress=])

Adds penalty to Hamiltonian (`mpo`), H0, of the form H0 + `lambda` * |`psi`><`psi`|; toggle to compress resulting wavefunction

See also: [`penalty`](@ref)
"""
function penalty!(mpo::MPO,lambda::Float64,psi::MPS;compress::Bool=true)
  for i = 1:length(psi)
    QS = size(psi[i],2)
    R = eltype(mpo[i])
    temp_psi = reshape(psi[i],size(psi[i])...,1)
    if i == psi.oc
      term = contractc(temp_psi,4,temp_psi,4,alpha=lambda)
    else
      term = contractc(temp_psi,4,temp_psi,4)
    end
    bigrho = permutedims(term,[1,4,5,2,3,6])
    rho = reshape!(bigrho,size(bigrho,1)*size(bigrho,2),QS,QS,size(bigrho,5)*size(bigrho,6),merge=true)
    if i == 1
      mpo[i] = joinindex!(4,mpo[i],rho)
    elseif i == length(psi)
      mpo[i] = joinindex!(1,mpo[i],rho)
    else
      mpo[i] = joinindex!([1,4],mpo[i],rho)
    end
  end
  return compress ? compressMPO!(mpo) : mpo
end
export penalty!

"""
    penalty!(mpo,lambda,psi[,compress=])

Same as `penalty!` but makes a copy of `mpo`

See also: [`penalty!`](@ref)
  """
function penalty(mpo::MPO,lambda::Float64,psi::MPS;compress::Bool=true)
  newmpo = copy(mpo)
  return penalty!(newmpo,lambda,psi,compress=compress)
end
export penalty

"""
  transfermatrix([dualpsi,]psi,i,j[,transfermat=])

Forms the transfer matrix (an MPS tensor and its dual contracted along the physical index) between sites `i` and `j` (inclusive). If not specified, the `transfermat` field will initialize to the transfer matrix from the `i-1` site.  If not set otherwise, `dualpsi = psi`.

The form of the transfer matrix must be is as follows (dual wavefunction tensor on top, conjugated)

1 ------------ 3
        |
        |
        |
        |
2 ------------ 4

There is no in-place version of this function

"""
function transfermatrix(dualpsi::MPS,psi::MPS,i::Integer,j::Integer;startsite::Integer=i,transfermat::TensType=ccontract([1,3,2,4],dualpsi[startsite],2,psi[startsite],2))
  for k = startsite+1:j
    transfermat = contractc(transfermat,3,dualpsi[k],1)
    transfermat = contract(transfermat,[3,4],psi[k],[1,2])
  end
  return transfermat
end

function transfermatrix(psi::MPS,i::Integer,j::Integer;startsite::Integer=i,transfermat::TensType=ccontract([1,3,2,4],psi[startsite],2,psi[startsite],2))
  return transfermatrix(psi,psi,i,j,transfermat=transfermat)
end
























































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

function MPS(vec::Array{W,1};regtens::Bool=false,oc::Integer=1,type::DataType=Float64) where W <: qarray
  return matrixproductstate(vec,oc)
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
function assignflux!(i::Integer,mps::MPS,QnumMat::Array{Array{Q,1},1},storeVal::Array{W,1}) where {Q <: Qnum, W <: Number}
  for a = 1:size(mps[i],1)
    for b = 1:size(mps[i],2)
      for c = 1:size(mps[i],3)
        absval = abs(mps[i][a,b,c])
        if absval > storeVal[c]
          storeVal[c] = absval
          QnumMat[3][c] = inv(QnumMat[1][a]+QnumMat[2][b])
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
                  flux::Q=Q(),randomize::Bool=true,override::Bool=true,warning::Bool=true,lastfluxzero::Bool=false)::MPS where Q <: Qnum
  if newnorm
    if warning
      println("(makeqMPS: If converting from a non-QN MPS to a QN MPS, then beware that applying a 3S or adding noise in general to the MPS is not considered when calling makeqMPS)")
    end
    start_norm = expect(mps)
  end
  W = elnumtype(mps)
  QtensVec = Array{Qtens{W,Q},1}(undef, size(mps.A,1))
  thisflux  = Q()
  zeroQN = copy(thisflux)
  Ns = length(mps)
  storeQnumMat = Q[zeroQN]
  theseArrows = length(arrows) == 0 ? Bool[false,true,true] : arrows[1]
  for i = 1:Ns
    currSize = size(mps[i])
    QnumMat = Array{Q,1}[Array{Q,1}(undef,currSize[a]) for a = 1:ndims(mps[i])]

    QnumMat[1] = inv.(storeQnumMat)
    QnumMat[2] = copy(Qlabels[(i-1) % size(Qlabels,1) + 1])
    storeVal = zeros(eltype(mps[i]),size(mps[i],3))
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
#      @assert(!isapprox(thisnorm,0.))
    finalMPS[mps.oc] *= sqrt(start_norm)/sqrt(thisnorm)
  end
  if lastfluxzero
    for q = 1:length(finalMPS[end].Qblocksum)
      finalMPS[end].Qblocksum[q][2] = inv(finalMPS[end].flux)
    end
  else
    Qnumber = finalMPS[end].QnumMat[3][1]
    finalMPS[end].flux,newQnum = inv(finalMPS[end].QnumSum[3][Qnumber]),inv(finalMPS[end].flux)
    finalMPS[end].QnumSum[3][1] = newQnum
  end

  for q = 1:length(finalMPS[end].Qblocksum)
    newQsum = Q()
    index = finalMPS[end].ind[q][2][:,1] .+ 1 #[x]
    pos = finalMPS[end].currblock[2]
    for y = 1:length(pos)
      Qnumber = finalMPS[end].QnumMat[pos[y]][index[y]]
      add!(newQsum,finalMPS[end].QnumSum[pos[y]][Qnumber])
    end
    copy!(finalMPS[end].Qblocksum[q][2],newQsum)
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
+ `warning::Bool`: Toggle warning message if last tensor has no values or is zero
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
  zeroQN = Q()
  Ns = infinite ? 3*unitcell*length(mpo) : length(mpo)
  W = elnumtype(mpo)
  QtensVec = Array{Qtens{W,Q},1}(undef, Ns)
  storeQnumMat = Q[zeroQN]
  theseArrows = length(arrows) == 0 ? Bool[false,false,true,true] : arrows[1]
  zeroQN = Q()
  tempQN = Q()
  for w = 1:Ns
    i = (w-1) % length(mpo) + 1
    QnumMat = Array{Q,1}[Array{Q,1}(undef,size(mpo[i],a)) for a = 1:ndims(mpo[i])]

    QnumMat[1] = inv.(storeQnumMat)
    theseQN = Qlabels[(i-1) % size(Qlabels,1) + 1]
    QnumMat[2] = inv.(theseQN)
    QnumMat[3] = theseQN
    storeVal = -ones(Float64,size(mpo[i],4))
    for  a = 1:size(mpo[i],1)
      for b = 1:size(mpo[i],2)
        for c = 1:size(mpo[i],3)
          for d = 1:size(mpo[i],4)
            absval = abs(mpo[i][a,b,c,d])
            if absval > storeVal[d]
              storeVal[d] = absval
              copy!(tempQN,zeroQN)
              add!(tempQN,QnumMat[1][a])
              add!(tempQN,QnumMat[2][b])
              add!(tempQN,QnumMat[3][c])
              QnumMat[4][d] = inv(tempQN)
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



















#       +---------------------------------------+
#>------+    Automatic determination of MPO     +---------<
#       +---------------------------------------+

"""
    reorder!(C[,Ncols=])

Reorders the `Ncols` columns of `C` according to the Fiedler vector reordering in place if site is not 0

See also: [`reorder`](@ref)
"""
function reorder!(C::Array{W,2};Ncols::Integer=2) where W <: Number
    sitevec = vcat(C[:,1],C[:,2])
    for w = 3:Ncols
      sitevec = vcat(sitevec,C[:,w])
    end
    Ns = maximum(sitevec)
    A = zeros(Int64,Ns,Ns) #adjacency matrix = neighbor table
    D = zeros(Int64,Ns) #degree matrix
    for i = 1:size(C,1)
      for x = 1:Ncols
        xpos = C[i,x]
        for w = x+1:Ncols
          ypos = C[i,w]
          if xpos != 0 && ypos != 0
            A[xpos,ypos] = 1
            D[xpos] += 1
            D[ypos] += 1
          end
        end
      end
    end
    L = D - A
    D,U = LinearAlgebra.eigen(L)
    fiedlervec = sortperm(U[:,2]) #lowest is all ones, so this is the first non-trivial one
    for i = 1:size(C,1)
      for w = 1:Ncols
        if C[i,w] != 0
          C[i,w] = fiedlervec[C[i,w]]
        end
      end
    end
    return C,fiedlervec #second eigenvector is the Fiedler vector
  end

  """
      reorder(C[,Ncols=])

  Reorders the `Ncols` columns of `C` according to the Fiedler vector reordering if site is not 0

  See also: [`reorder!`](@ref)
  """
  function reorder(C::Array{W,2};Ncols::Integer=2) where W <: Number
    P = copy(C)
    return reorder!(P,Ncols=Ncols)
  end

"""
    mpoterm(val,operator,ind,base,trail...)

Creates an MPO from operators (`operator`) with a prefactor `val` on sites `ind`.  Must also provide a `base` function which are identity operators for each site.  `trail` operators can be defined (example: fermion string operators)

# Example:
```julia
Cup, Cdn, Nup, Ndn, Ndens, F, O, Id = fermionOps()
base = [Id for i = 1:Ns]; #identity operators of the appropriate physical index size for each site
CupCupdag = mpoterm(-1.,[Cup,Cup'],[1,2],base,F); #Cup_1 Cup^dag_2
newpsi = applyMPO(psi,CupCupdag); #applies <Cup_1 Cup^dag_2> to psi
expect(psi,newpsi);
```
"""
function mpoterm(val::Number,operator::Array{W,1},ind::Array{P,1},base::Array{X,1},trail::Y...)::MPO where {W <: AbstractArray, X <: AbstractArray, Y <: AbstractArray, P <: Integer}
  opString = copy(base)
  for a = 1:size(ind,1)
    opString[ind[a]] = (a == 1 ? val : 1.)*copy(operator[a])
    if length(trail) > 0
      for b = 1:ind[a]-1
        opString[b] = contract(trail[1],2,opString[b],1)
      end
    end
  end
  return MPO(opString)
end

function mpoterm(operator::AbstractArray,ind::Array{P,1},base::AbstractArray,trail::AbstractArray...)::MPO where P <: Integer
  return mpoterm(1.,operator,ind,base,trail...)
end
export mpoterm

"""
    mpoterm(Qlabels,val,operator,ind,base,trail...)

Same as `mpoterm` but converts to quantum number MPO with `Qlabels` on the physical indices

See also: [`mpoterm`](@ref)
"""
function mpoterm(Qlabels::Array{Array{Q,1},1},val::Number,operator::AbstractArray,ind::Array{P,1},base::AbstractArray,trail::AbstractArray...)::MPO where {Q <: Qnum, P <: Integer}
  return makeqMPO(mpoterm(val,operator,ind,base,trail...),Qlabels)
end

function mpoterm(Qlabels::Array{Array{Q,1},1},operator::AbstractArray,ind::Array{P,1},base::AbstractArray,trail::AbstractArray...)::MPO where {Q <: Qnum, P <: Integer}
  return mpoterm(Qlabels,1.,operator,ind,base,trail...)
end
export mpoterm

#import Base.*
"""
    mpo * mpo

functionality for multiplying MPOs together; contracts physical indices together
"""
function *(X::MPO,Y::MPO)
  return mult!(copy(X),Y)
end

#  import .Qtensor.mult!
function mult!(X::MPO,Y::MPO)
  for i = 1:size(X,1)
    temp = contract([4,1,2,5,6,3],X[i],3,Y[i],2)
    X[i] = reshape(temp,size(temp,1)*size(temp,2),size(temp,3),size(temp,4),size(temp,5)*size(temp,6))
  end
  return deparallelize!(X)
end

#import Base.+
"""
    A + B

functionality for adding (similar to direct sum) of MPOs together; uses joinindex function to make a combined MPO

note: deparallelizes after every addition

See also: [`deparallelization`](@ref) [`add!`](@ref)
"""
function +(X::MPO,Y::MPO)
  checktype = typeof(eltype(X[1])(1) * eltype(Y[1])(1))
  if checktype != eltype(X[1])
    S = MPO(checktype,copy(X))
  else
    S = copy(X)
  end
  return add!(S,Y)
end

"""
    H + c

Adds a constant `c` to a Hamiltonian `H` (commutative)
"""
function +(H::MPO,c::Number;pos::Integer=1)
  const_term = MPO([i == pos ? mult!(c,makeId(H[i],[2,3])) : makeId(H[i],[2,3]) for i = 1:length(H)])
  return copy(H) + const_term
end

function +(c::Number,H::MPO;pos::Integer=1)
  return +(H,c,pos=pos)
end

#import Base.-
"""
    H - c

Adds a constant `c` to a Hamiltonian `H`
"""
function -(H::MPO,c::Number;pos::Integer=1)
  return +(H,-c,pos=pos)
end

#  import .QN.add!
"""
    add!(A,B)

functionality for adding (similar to direct sum) of MPOs together and replacing `A`; uses joinindex function to make a combined MPO

note: deparallelizes after every addition

See also: [`deparallelization`](@ref) [`+`](@ref)
"""
function add!(A::MPO,B::MPO;finiteBC::Bool=true)
  if finiteBC
    A[1] = joinindex!(4,A[1],B[1])
    for a = 2:size(A,1)-1
      A[a] = joinindex!([1,4],A[a],B[a])
    end
    A[end] = joinindex!(1,A[size(A,1)],B[size(A,1)])
  else
    for a = 1:size(A,1)
      A[a] = joinindex!([1,4],A[a],B[a])
    end
  end
  return deparallelize!(A)
end

"""
    makeInvD(D)

Finds nearest factor of 2 to the magnitude of `D`
"""
function makeInvD(D::TensType)
  avgD = sum(D)
  avgD /= size(D,1)
  finaltwo = 2^(max(0,convert(intType,floor(log(2,avgD))))+1)
  return finaltwo
end

function pullvec(M::TensType,j::Integer,left::Bool)
  if typeof(M) <: qarray
    if left
      temp = M[:,:,:,j]
      firstvec = reshape(temp,size(M,1),size(M,2),size(M,3),1)

      newqn = [inv(firstvec.flux)]
      firstvec.QnumMat[4] = newqn
      firstvec.QnumSum[4] = newqn
      firstvec.flux = typeof(M.flux)()
    else
      temp = M[j,:,:,:]
      firstvec = reshape(temp,size(M,2),size(M,3),size(M,4),1)

      newqn = [inv(firstvec.flux)]
      firstvec.QnumMat[4] = newqn
      firstvec.QnumSum[4] = newqn
      firstvec.flux = typeof(M.flux)()
      firstvec = permutedims!(firstvec,[4,1,2,3])
    end
  else
    if left
      firstvec = reshape!(M[:,:,:,j],size(M,1),size(M,2),size(M,3),1)
    else
      firstvec = reshape!(M[j,:,:,:],1,size(M,2),size(M,3),size(M,4))
    end
  end
  return firstvec
end

"""
    deparallelize!(M[,left=])

Deparallelizes a matrix-equivalent of a rank-4 tensor `M`; toggle the decomposition into the `left` or `right`
"""
function deparallelize!(M::TensType;left::Bool=true,zero::Float64=0.) where W <: Number
  if left
    rsize = size(M,4)
    lsize = max(prod(a->size(M,a),1:3),rsize)
  else
    lsize = size(M,1)
    rsize = max(lsize,prod(a->size(M,a),2:4))
  end
  T = zeros(eltype(M),lsize,rsize) #maximum size for either left or right
  firstvec = pullvec(M,1,left)

  K = typeof(M)[firstvec]
  normK = eltype(M)[norm(K[1])]

  b = left ? 4 : 1
  for j = 1:size(M,b)
    thisvec = pullvec(M,j,left)

    mag_thisvec = norm(thisvec) # |A|

    condition = true
    i = 0
    while condition  && i < size(K,1)
      i += 1
      if left
        dot = ccontract(K[i],thisvec)#,left,!left)
      else
        dot = contractc(K[i],thisvec)
      end
#        dot = searchindex(temp,1,1)
      if isapprox(real(dot),mag_thisvec * normK[i]) && !isapprox(normK[i],0) #not sure why it would be zero...
        normres = mag_thisvec/normK[i]
        if left
          T[i,j] = normres
        else
          T[j,i] = normres
        end
        condition = false
      end
    end

    if condition && !(isapprox(mag_thisvec,0.))

      push!(K,thisvec)
      push!(normK,norm(K[end]))

      if left
        T[length(K),j] = 1.
      else
        T[j,length(K)] = 1.
      end
    end
  end

  QNflag = typeof(M) <: qarray
  if left
    newT = T[1:size(K,1),:]
    if size(K,1) == 1
      finalT = reshape(newT,1,size(T,2))
    else
      finalT = newT
    end

    newK = K[1]
    for a = 2:size(K,1)
      newK = joinindex!(newK,K[a],4)
    end

    if QNflag
      finalT = Qtens(finalT,[inv.(newK.QnumMat[4]),M.QnumMat[4]])
    end

    return newK,finalT
  else
    newT = T[:,1:size(K,1)]
    if size(K,1) == 1
      finalT = reshape(newT,size(T,1),1)
    else
      finalT = newT
    end

    newK = K[1]
    for a = 2:size(K,1)
      newK = joinindex!(newK,K[a],1)
    end

    if QNflag
      finalT = Qtens(finalT,[M.QnumMat[1],inv.(newK.QnumMat[1])])
    end

    return finalT,newK
  end
end

function deparallelize!(M::tens{W};left::Bool=true) where W <: Number
  X = reshape(M.T,size(M)...)
  out = deparallelize!(X,left=left)
  return tens(out[1]),tens(out[2])
end

"""
    deparallelize!(W[,sweeps=])

Applies `sweeps` to MPO (`W`) to compress the bond dimension
"""
function deparallelize!(W::MPO;sweeps::Integer=1)
  for n = 1:sweeps
    for i = 1:length(W)-1
      W[i],T = deparallelize!(W[i],left=true)
      W[i+1] = contract(T,2,W[i+1],1)
    end
    for i = length(W):-1:2
      T,W[i] = deparallelize!(W[i],left=false)
      W[i-1] = contract(W[i-1],4,T,1)
    end
  end
  return W
end
export deparallelize!

"""
    deparallelize!(W[,sweeps=])

Deparallelize an array of MPOs (`W`) for `sweeps` passes; compressed MPO appears in first entry
"""
function deparallelize!(W::Array{MPO,1};sweeps::Integer=1)
  nlevels = floor(intType,log(2,size(W,1)))
  active = Bool[true for i = 1:size(W,1)]
  if size(W,1) > 2
    for j = 1:nlevels
      currsize = fld(length(W),2^(j-1))
#        let currsize = currsize, W = W, j = j
        #=Threads.@threads=# for i = 1:2^j:currsize
          iL = i
          iR = iL + 2^(j-1)
          if iR < currsize
            add!(W[iL],W[iR])
            W[iL] = deparallelize!(W[iL],sweeps=sweeps)
            active[iR] = false
          end
        end
#        end
    end
    if sum(active) > 1
      deparallelize!(W[active],sweeps=sweeps)
    end
  end
  if size(W,1) == 2
    W[1] = add!(W[1],W[2])
  end
  return deparallelize!(W[1],sweeps=sweeps)
end

"""
    deparallelize(W[,sweeps=])

makes copy of W while deparallelizing

See also: [`deparallelize!`](@ref)
"""
function deparallelize(W::MPO;sweeps::Integer=1)
  return deparallelize!(copy(W),sweeps=sweeps)
end

function deparallelize(W::T;sweeps::Integer=1) where T <: Union{AbstractArray,qarray}
  return deparallelize!(copy(W),sweeps=sweeps)
end
export deparallelize

"""
    compressMPO!(W[,sweeps=,cutoff=,deltam=,minsweep=,nozeros=])

compresses MPO (`W`; or several `M`) with SVD compression for `sweeps` sweeps, `cutoff` applied to the SVD, `deltam` target for teh bond dimension compression, and `nozeros` defaulted to true to eliminate all zeros in the SVD
"""
const forwardshape = Array{intType,1}[intType[1,2,3],intType[4]]
const backwardshape = Array{intType,1}[intType[1],intType[2,3,4]]
function compressMPO!(W::MPO,M::MPO...;sweeps::Integer=1000,cutoff::Float64=1E-16,
                      deltam::Integer=0,minsweep::Integer=1,nozeros::Bool=false)
  for a = 1:length(M)
    W = add!(W,M[a])
  end
  n = 0
  mchange = 1000
  lastmdiff = [size(W[i],4) for i = 1:size(W,1)-1]
  while (n < sweeps && mchange > deltam) || (n < minsweep)
    n += 1
    for i = 1:size(W,1)-1
      U,D,V = svd(W[i],forwardshape,cutoff=cutoff,nozeros=nozeros)
      scaleD = makeInvD(D)

      U = mult!(U,scaleD)
      W[i] = U

      scaleDV = contract(D,2,V,1,alpha=1/scaleD)
      W[i+1] = contract(scaleDV,2,W[i+1],1)
    end
    for i = size(W,1):-1:2
      U,D,V = svd(W[i],backwardshape,cutoff=cutoff,nozeros=nozeros)
      scaleD = makeInvD(D)
      
      V = mult!(V,scaleD)
      W[i] = V

      scaleUD = contract(U,2,D,1,alpha=1/scaleD)
      W[i-1] = contract(W[i-1],4,scaleUD,1)
    end
    thismdiff = intType[size(W[i],4) for i = 1:size(W,1)-1]
    mchange = sum(a->lastmdiff[a]-thismdiff[a],1:size(thismdiff,1))
    lastmdiff = copy(thismdiff)
  end
  return W
end

"""
    compressMPO!(W[,sweeps=,cutoff=,deltam=,minsweep=,nozeros=])

compresses an array of MPOs (`W`) in parallel with SVD compression for `sweeps` sweeps, `cutoff` applied to the SVD, `deltam` target for teh bond dimension compression, and `nozeros` defaulted to true to eliminate all zeros in the SVD
"""
function compressMPO!(W::Array{MPO,1};sweeps::Integer=1000,cutoff::Float64=1E-16,
        deltam::Integer=0,minsweep::Integer=1,nozeros::Bool=true) # where V <: Union{Array{MPO,1},SubArray{MPO,1,Array{MPO,1},Tuple{Array{intType,1}}}}
  nlevels = floor(intType,log(2,size(W,1)))
  active = Bool[true for i = 1:size(W,1)]
  if size(W,1) > 2
    for j = 1:nlevels
      currsize = fld(length(W),2^(j-1))
      let currsize = currsize, W = W, sweeps = sweeps, cutoff = cutoff, j = j, nozeros=nozeros
        #=Threads.@threads=# for i = 1:2^j:currsize
          iL = i
          iR = iL + 2^(j-1)
          if iR < currsize
            W[iL] = compressMPO!(W[iL],W[iR],sweeps=sweeps,cutoff=cutoff,nozeros=nozeros)
            active[iR] = false
          end
        end
      end
    end
    if sum(active) > 1
      compressMPO!(W[active],sweeps=sweeps,cutoff=cutoff,nozeros=nozeros)
    end
  end
  return size(W,1) == 2 ? compressMPO!(W[1],W[2],nozeros=nozeros) : compressMPO!(W[1],nozeros=nozeros)
end
export compressMPO!

"""
    compressMPO(W,sweeps=,cutoff=,deltam=,minsweep=,nozeros=])

Same as `compressMPO!` but a copy is made of the original vector of MPOs

See also: [`compressMPO!`](@ref)
"""
function compressMPO(W::Array{MPO,1};sweeps::Integer=1000,cutoff::Float64=1E-16,deltam::Integer=0,minsweep::Integer=1,nozeros::Bool=true)
  M = copy(W)
  return compressMPO!(M;sweeps=sweeps,cutoff=cutoff,deltam=deltam,minsweep=minsweep,nozeros=nozeros)
end
export compressMPO

#end