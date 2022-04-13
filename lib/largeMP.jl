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
struct largematrixproductoperator <: largeMPO
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
struct largeenvironment <: largeEnv
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
function largeMPO(mpo::P;label::String="mpo_",names::Array{String,1}=[label*"$i" for i = 1:length(mpo)],ext::String=file_extension) where P <: Union{Array,MPO}
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
function largeLenv(lenv::P;label::String="Lenv_",names::Array{String,1}=[label*"$i" for i = 1:length(lenv)],ext::String=file_extension) where P <: Union{Array,MPO}
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
function largeRenv(renv::P;label::String="Renv_",names::Array{String,1}=[label*"$i" for i = 1:length(renv)],ext::String=file_extension) where P <: Union{Array,MPO}
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
function largeEnv(lenv::P,renv::P;Llabel::String="Lenv_",Lnames::Array{String,1}=[Llabel*"$i" for i = 1:Ns],Rlabel::String="Renv_",Rnames::Array{String,1}=[Rlabel*"$i" for i = 1:Ns],ext::String=file_extension,type::DataType=Float64) where P <: Union{Array,MPO}
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