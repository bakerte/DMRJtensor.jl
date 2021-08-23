#########################################################################
#
#  Density Matrix Renormalization Group (and other methods) in julia (DMRjulia)
#                              v0.8
#
#########################################################################
# Made by Thomas E. Baker (2018)
# See accompanying license with this program
# This code is native to the julia programming language (v1.1.1) or (v1.5)
#


"""
    Module: MPutil

Functions and definitions for MPS and MPO
"""
#=
module MPutil
using ..tensor
using ..contractions
using ..decompositions
=#
#       +---------------------------------------+
#>------+         MPS and qMPS types            +---------<
#       +---------------------------------------+

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
      
  struct to hold regMPS tensors and orthogonality center

  ```
  A::Array{Array{Number,3},N} (vector of regMPS tensors)
  oc::Int64 (orthogonality center)
  ```
  # Note:
  + Contruct this through [`regMPS`](@ref)
  """
  mutable struct matrixproductstate{W} <: regMPS where W <: Array{TensType,1}
    A::W
    oc::Integer
  end
  
  """
      `matrixproductoperator` 
      
  struct to hold MPO tensors

  ```
  H::Array{Array{Number,4},N} (vector of MPO tensors)
  ```

  # Note:
  + Contruct this through [`MPO`](@ref)
  """
  mutable struct matrixproductoperator{W} <: regMPO where W <: Array{TensType,1}
    H::W
  end

  """
      `envVec`

  Array that holds environment tensors
  """
  mutable struct environment{W} <: regEnv where W <: Array{TensType,1}
    V::Array{W,1}
  end

  function environment(T::G...) where G <: TensType
    return environment([T...])
  end
  export environment
  

  function environment(T::W,Ns::Integer) where W <: TensType
    return environment(Array{W,1}(undef,Ns))
  end

  function environment(network::G...) where G <: Union{MPS,MPO}
    Ns = length(network[1])
    return environment(typeof(network[1]),Ns)
  end

  #=
  """
      `envVec`

  Array that holds environment tensors
  """
  mutable struct vecenvironment{W} <: regEnv where W <: Array{TensType,1} #Union{Array{Array{T,N},1},Array{tens{T},1},Array{Qtens{T,Q},1}} where {N,T <: Number,Q <: Qnum}
    V::Array{Array{W,1}}
  end
  export vecenvironment
  =#

  """
    makeoc(Ns[,oc])

  processes `oc` to get location of the orthogonality center (first entry, default 1) for a system with `Ns` sites
  """
  function makeoc(Ns::Integer,oc::Integer...)
    return length(oc) > 0  && 0 < oc[1] <= Ns ? oc[1] : 1
  end

  """
      MPS([T,]A[,oc])

  constructor for MPS with tensors `A` and orhtogonailty center `oc`; can optinally request an element type `T` for the tensors
  """
  function MPS(psi::Array{W,1};regtens::Bool=false) where W <: TensType
    return MPS(psi,1,regtens=regtens)
  end

  function MPS(psi::MPS;regtens::Bool=false) where W <: TensType
    return MPS(psi.A,1,regtens=regtens)
  end

  function MPS(psi::MPS,oc::Integer;regtens::Bool=false) where W <: TensType
    return MPS(psi.A,oc,regtens=regtens)
  end

  function MPS(psi::Array{W,1},oc::Integer;regtens::Bool=false) where W <: TensType
    if !regtens && typeof(psi[1]) <: AbstractArray
      tenspsi = [tens(psi[a]) for a = 1:length(psi)]
    else
      tenspsi = copy(psi)
    end
    return matrixproductstate(tenspsi,oc)
  end

  function MPS(thistype::DataType,B::Array{W,1},oc::Integer...;regtens::Bool=false) where W <: AbstractArray
    if !regtens
      MPSvec = [tens(convert(Array{thistype,ndims(B[i])},copy(B[i]))) for i = 1:size(B,1)]
    else
      MPSvec = [convert(Array{thistype,ndims(B[i])},copy(B[i])) for i = 1:size(B,1)]
    end
    return MPS(MPSvec,makeoc(length(B),oc...))
  end

  function MPS(thistype::DataType,B::Array{W,1},oc::Integer...;regtens::Bool=false) where W <: Union{denstens,qarray}
    MPSvec = [convertTens(thistype, copy(B[i])) for i = 1:size(B,1)]
    return MPS(MPSvec,makeoc(length(B),oc...))
  end

  function MPS(T::Type,mps::P) where P <: regMPS
    return MPS(T,mps.A,mps.oc)
  end

  function MPS(thistype::DataType,psi::W;regtens::Bool=false) where W <: TensType
    return MPS(thistype,psi,1,regtens=regtens)
  end

  """
      MPO([T,]H)

  constructor for MPO with tensors `H`; can optinally request an element type `T` for the tensors
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

  function MPO(T::Type,mpo::P,regtens::Bool=false) where P <: MPO
    return MPO(T,mpo.H,regtens=regtens)
  end

  function MPO(mpo::P,regtens::Bool=false) where P <: MPO
    return MPO(mpo.H,regtens=regtens)
  end

  const temptype = Union{regMPS,regMPO,regEnv}

#  import .tensor.elnumtype
  function elnumtype(op...)
    opnum = eltype(op[1])(1)
    for b = 2:length(op)
      opnum *= eltype(op[b])(1)
    end
    return typeof(opnum)
  end
  
  import Base.size
  """
      size(H[,i])
  
  size prints out the size of the tensor field of an regEnv, regMPS, or MPO; this is effectively the number of sites
  """
  function size(H::P) where P <: MPO
    return size(H.H)
  end
  function size(H::P,i::Integer) where P <: MPO
    return size(H.H,i)
  end
  function size(psi::P) where P <: MPS
    return size(psi.A)
  end
  function size(psi::P,i::Integer) where P <: MPS
    return size(psi.A,i)
  end
  
  function size(G::P) where P <: regEnv
    return size(G.V)
  end
  function size(G::P,i::Integer) where P <: regEnv
    return size(G.V,i)
  end

  import Base.length
  function length(H::P) where P <: MPO
    return length(H.H)
  end
  function length(psi::P) where P <: MPS
    return length(psi.A)
  end
  function length(G::P) where P <: regEnv
    return length(G.V)
  end

  import Base.eltype
  """
      eltype(Y)

  eltype gets element type of the regEnv, MPS, or MPO tensor fields
  """
  function eltype(Y::P) where P <: regMPS
    return eltype(Y.A[1])
  end

  function eltype(H::P) where P <: regMPO
    return eltype(H.H[1])
  end

  function eltype(G::P) where P <: regEnv
    return eltype(G.V[1])
  end

  import Base.getindex
  """
      getindex(A,i...)

  getindex allows to retrieve elements to an regEnv, MPS or MPO (ex: W = psi[1])
  """
  function getindex(A::P,i::Integer) where P <: regMPS
    return A.A[i]
  end
  function getindex(A::P,r::UnitRange{W}) where W <: Integer where P <: regMPS
    if A.oc in r
      newoc = findfirst(w->w == A.oc,r)
    else
      newoc = 0
    end
    return P(A.A[r],newoc)
  end
  function getindex(H::P,i::Integer) where P <: regMPO
    return H.H[i]
  end
  function getindex(H::P,r::UnitRange{W}) where W <: Integer where P <: regMPO
    return P(H.H[r])
  end
  
  function getindex(G::P,i::Integer) where P <: regEnv
    return G.V[i]
  end
  function getindex(G::P,r::UnitRange{W}) where W <: Integer where P <: regEnv
    return P(G.V[r])
  end
  

  import Base.lastindex
  """
      psi[end]

  lastindex! allows to get the end element of an regEnv, MPS, or MPO
  """
  function lastindex(A::P) where P <: regMPS
    return lastindex(A.A)
  end

  function lastindex(H::P) where P <: regMPO
    return lastindex(H.H)
  end
  
  function lastindex(G::P) where P <: regEnv
    return lastindex(G.V)
  end
  

  import Base.setindex!
  """
      psi[1] = W

  setindex! allows to assign elements to an regEnv, MPS, or MPO (ex: psi[1] = W)
  """
  function setindex!(H::P,A::G,i::Int64) where P <: regMPO where G <: TensType
    H.H[i] = A
    nothing
  end
  function setindex!(H::P,A::G,i::Int64) where P <: regMPS where G <: TensType
    H.A[i] = A
    nothing
  end
  
  function setindex!(G::P,A::K,i::Int64) where P <: regEnv where K <: TensType
    G.V[i] = A
    nothing
  end
  

  import Base.copy
  """
      copy(psi)

  Copies an MPS; type stable (where deepcopy is type-unstable inherently)
  """
  function copy(mps::matrixproductstate{W}) where W <: TensType
    return matrixproductstate{W}([copy(mps.A[i]) for i = 1:length(mps)],copy(mps.oc))
  end

  function copy(mpo::matrixproductoperator{W}) where W <: TensType
    return matrixproductoperator{W}([copy(mpo.H[i]) for i = 1:length(mpo)])
  end

  function copy(mps::P) where P <: regMPS
    return MPS([copy(mps.A[i]) for i = 1:length(mps)],copy(mps.oc))
  end

  function copy(mpo::P) where P <: regMPO
    return MPO([copy(mpo.H[i]) for i = 1:length(mpo)])
  end

  function copy(G::P) where P <: regEnv
    return envVec{eltype(G[1])}([copy(G.V[i]) for i = 1:length(G)])
  end

  import Base.conj!
  """
      conj!(psi)

  Conjugates all elements in an MPS in-place

  See also [`conj`](@ref)
  """
  function conj!(A::P) where P <: regMPS
    conj!.(A.A)
    return A
  end

  import Base.conj
  """
      A = conj(psi)
  
  Conjugates all elements in an MPS and makes a copy

  See also [`conj!`](@ref)
  """
  function conj(A::P) where P <: regMPS
    B = copy(A)
    conj!.(B.A)
    return B
  end

  #       +------------------+
  #>------|  Large MPS/MPO   |----------<
  #       +------------------+


  const file_extension = ".dmrjulia"

  """
      largeMPS

  Abstract types for MPS; saves tensors to disk on setting values and retrieves them on calling an element; slower than other `MPS` form

  See also: [`largeMPO`](@ref) [`MPS`](@ref)
  """
  abstract type largeMPS <: MPS end
  export largeMPS

  """
      largeMPO
      
  Abstract types for largeMPO; saves tensors to disk on setting values and retrieves them on calling an element; slower than other `MPO` form

  See also: [`largeMPS`](@ref) [`MPO`](@ref)
  """
  abstract type largeMPO <: MPO end
  export largeMPO


  abstract type largeEnv <: envType end
  export largeEnv

  """
      `largematrixproductstate` struct to hold MPS tensors and orthogonality center
  ```
  A::Array{String,1} (vector of file names where MPS tensors are saved)
  oc::Int64 (orthogonality center)
  ```
  # Note:
  + Contruct this through [`largeMPS`](@ref)
  """
  mutable struct largematrixproductstate <: largeMPS
    A::Array{String,1}
    oc::Integer
    type::DataType
  end
  
  """
      `matrixproductoperator` struct to hold MPO tensors

  ```
  H::Array{Array{Number,4},N} (vector of MPO tensors)
  ```

  # Note:
  + Contruct this through [`MPO`](@ref)
  """
  mutable struct largematrixproductoperator <: largeMPO
    H::Array{String,1}
    type::DataType
  end

  mutable struct largeenvironment <: largeEnv
    V::Array{String,1}
    type::DataType
  end

  function tensor2disc(name::String,tensor::P;ext::String=file_extension) where P <: TensType
    Serialization.serialize(name*ext,tensor)
    nothing
  end

  function tensorfromdisc(name::String;ext::String=file_extension)
    return Serialization.deserialize(name*ext)
  end

  function largeMPS(psi::P;label::String="mps_",names::Array{String,1}=[label*"$i" for i = 1:length(psi)],ext::String=file_extension) where P <: MPS
    lastnum = 1
    for b = 1:length(psi)
      C = psi[b]
      tensor2disc(names[b],C,ext=ext)
      lastnum *= eltype(C)(1)
    end
    return largematrixproductstate(names,psi.oc,typeof(lastnum))
  end
#=
  function largeMPO(mpo::MPO;label::String="mpo_",names::Array{String,1}=[label*"$i" for i = 1:length(mpo)])
    lastnum = 1
    for b = 1:length(mpo)
      C = mpo[b]
      tensor2disc(names[b],C)
      lastnum *= eltype(C)(1)
    end
    return largematrixproductoperator(names,typeof(lastnum))
  end
=#
  function largeMPO(mpo::P;label::String="mpo_",names::Array{String,1}=[label*"$i" for i = 1:length(mpo)],ext::String=file_extension) where P <: Union{AbstractArray,MPO}
    lastnum = 1
    for b = 1:length(mpo)
      C = mpo[b]
      tensor2disc(names[b],C,ext=ext)
      lastnum *= eltype(C)(1)
    end
    return largematrixproductoperator(names,typeof(lastnum))
  end

  function largematrixproductoperator(mpo::G;label::String="mpo_",names::Array{String,1}=[label*"$i" for i = 1:length(mpo)],ext::String=file_extension) where G <: Union{AbstractArray,regMPO}
    lastnum = 1
    for b = 1:length(mpo)
      C = mpo[b]
      tensor2disc(names[b],C,ext=ext)
      lastnum *= eltype(C)(1)
    end
    return largematrixproductoperator(names,typeof(lastnum))
  end

  function largeenvironment(Lenv::P,Renv::P;Ltag::String = "Lenv_",Lnames::Array{String,1}=[Ltag*"$i" for i = 1:length(Lenv)],
                            Rtag::String = "Renv_",Rnames::Array{String,1}=[Rtag*"$i" for i = 1:length(Renv)],ext::String=file_extension) where P <: Env
    lastnum = 1
    for b = 1:length(Lenv)
      L = Lenv[b]
      tensor2disc(Lnames[b],L,ext=ext)
      lastnum *= eltype(L)(1)

      R = Renv[b]
      tensor2disc(Rnames[b],R,ext=ext)
      lastnum *= eltype(R)(1)
    end
    return largeenvironment(Lnames,typeof(lastnum)),largeenvironment(Rnames,typeof(lastnum))
  end
  export largeenvironment

  import Base.getindex
  function getindex(A::P,i::Integer) where P <: largeMPS
    return tensorfromdisc(A.A[i])
  end

  function getindex(A::P,i::Integer) where P <: largeMPO
    return tensorfromdisc(A.H[i])
  end

  function getindex(A::P,i::Integer) where P <: largeEnv
    return tensorfromdisc(A.V[i])
  end

  import Base.setindex!
  function setindex!(H::G,A::P,i::Int64;ext::String=file_extension) where P <: TensType where G <: largeMPS
    tensor2disc(H.A[i],A,ext=ext)
    nothing
  end

  function setindex!(H::G,A::P,i::Int64;ext::String=file_extension) where P <: TensType where G <: largeMPO
    tensor2disc(H.H[i],A,ext=ext)
    nothing
  end

  function setindex!(H::G,A::P,i::Int64;ext::String=file_extension) where P <: TensType where G <: largeEnv
    tensor2disc(H.V[i],A,ext=ext)
    nothing
  end

  import Base.lastindex
  function lastindex(A::G;ext::String=file_extension) where G <: largeMPS
    return tensorfromdisc(A.A[end],ext=ext)
  end

  function lastindex(H::G;ext::String=file_extension) where G <: largeMPO
    return tensorfromdisc(H.H[end],ext=ext)
  end

  function lastindex(H::G;ext::String=file_extension) where G <: largeEnv
    return tensorfromdisc(H.V[end],ext=ext)
  end

  function length(A::G) where G <: largeMPS
    return length(A.A)
  end

  function length(H::G) where G <: largeMPO
    return length(H.H)
  end

  function length(H::G) where G <: largeEnv
    return length(H.V)
  end

  function eltype(op::R) where R <: Union{largeMPS,largeMPO,largeEnv}
    return op.type
  end

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
      @assert(storeoc[1] == oc)
      thisoc = oc
    end
    return largematrixproductstate(names,thisoc,thistype)
  end
  export loadMPS

  function loadMPO(Ns::Integer;label::String="mpo_",names::Array{String,1}=[label*"$i" for i = 1:Ns],ext::String=file_extension,oc::Integer=0)
    lastnum = 1
    for i = 1:Ns
      name = names[i]
      A = tensorfromdisc(name,ext=ext)
      lastnum *= eltype(A)(1)
    end
    thistype = typeof(lastnum)
    return largematrixproductstate(names,thistype)
  end
  export loadMPO

  function loadLenv(Ns::Integer;label::String="Lenv_",names::Array{String,1}=[label*"$i" for i = 1:Ns],ext::String=file_extension,oc::Integer=0)
    lastnum = 1
    for i = 1:Ns
      name = names[i]
      A = tensorfromdisc(name,ext=ext)
      lastnum *= eltype(A)(1)
    end
    thistype = typeof(lastnum)
    return largematrixproductstate(names,thistype)
  end
  export loadLenv

  function loadRenv(Ns::Integer;label::String="Renv_",names::Array{String,1}=[label*"$i" for i = 1:Ns],ext::String=file_extension,oc::Integer=0)
    lastnum = 1
    for i = 1:Ns
      name = names[i]
      A = tensorfromdisc(name,ext=ext)
      lastnum *= eltype(A)(1)
    end
    thistype = typeof(lastnum)
    return largematrixproductstate(names,thistype)
  end
  export loadRenv

  function copy(names::Array{String,1},X::largeMPS;ext::String=file_extension,copyext::String=ext)
    newObj = deepcopy(X)
    newObj.A = names
    for i = 1:length(X)
      Y = tensorfromdisc(names[i],ext=ext)
      tensor2disc(X.A[i],Y,ext=copyext)
    end
    return newObj
  end

  function copy(names::Array{String,1},X::G;ext::String=file_extension,copyext::String=ext) where G <: largeMPO
    newObj = deepcopy(X)
    newObj.H = names
    for i = 1:length(X)
      Y = tensorfromdisc(names[i],ext=ext)
      tensor2disc(X.H[i],Y,ext=copyext)
    end
    return newObj
  end

  function copy(names::Array{String,1},X::largeEnv;ext::String=file_extension,copyext::String=ext)
    newObj = deepcopy(X)
    newObj.V = names
    for i = 1:length(X)
      Y = tensorfromdisc(names[i],ext=ext)
      tensor2disc(X.V[i],Y,ext=copyext)
    end
    return newObj
  end

  function largeMPS(Ns::Integer;label::String="mps_",names::Array{String,1}=[label*"$i" for i = 1:Ns],ext::String=file_extension,thisoc::Integer=1,type::DataType=Float64)
    return largematrixproductstate(names,thisoc,type)
  end

  function largeMPS(type::DataType,Ns::Integer;label::String="mps_",names::Array{String,1}=[label*"$i" for i = 1:Ns],ext::String=file_extension,thisoc::Integer=1)
    return largeMPS(Ns,label=label,names=names,ext=ext,thisoc=thisoc,type=type)
  end

  function largeMPO(Ns::Integer;label::String="mpo_",names::Array{String,1}=[label*"$i" for i = 1:Ns],ext::String=file_extension,type::DataType=Float64)
    return largematrixproductoperator(names,type)
  end

  function largeMPO(type::DataType,Ns::Integer;label::String="mpo_",names::Array{String,1}=[label*"$i" for i = 1:Ns],ext::String=file_extension)
    return largeMPO(Ns,label=label,names=names,ext=ext,type=type)
  end



  function largeLenv(Ns::Integer;label::String="Lenv_",names::Array{String,1}=[label*"$i" for i = 1:Ns],ext::String=file_extension,type::DataType=Float64)
    return largeenvironment(names,type)
  end

  function largeLenv(type::DataType,Ns::Integer;label::String="Lenv_",names::Array{String,1}=[label*"$i" for i = 1:Ns],ext::String=file_extension)
    return largeLenv(Ns,label=label,names=names,ext=ext,type=type)
  end
  export largeLenv

  function largeRenv(Ns::Integer;label::String="Renv_",names::Array{String,1}=[label*"$i" for i = 1:Ns],ext::String=file_extension,type::DataType=Float64)
    return largeenvironment(names,type)
  end

  function largeRenv(type::DataType,Ns::Integer;label::String="Renv_",names::Array{String,1}=[label*"$i" for i = 1:Ns],ext::String=file_extension)
    return largeRenv(Ns,label=label,names=names,ext=ext,type=type)
  end
  export largeRenv

  function largeLRenv(Ns::Integer;Llabel::String="Lenv_",Lnames::Array{String,1}=[Llabel*"$i" for i = 1:Ns],Rlabel::String="Renv_",Rnames::Array{String,1}=[Rlabel*"$i" for i = 1:Ns],ext::String=file_extension,type::DataType=Float64)
    return largeenvironment(Lnames,type),largeenvironment(Rnames,type)
  end

  function largeLRenv(type::DataType,Ns::Integer;Llabel::String="Lenv_",Lnames::Array{String,1}=[Llabel*"$i" for i = 1:Ns],Rlabel::String="Renv_",Rnames::Array{String,1}=[Rlabel*"$i" for i = 1:Ns],ext::String=file_extension)
    return largeLRenv(Ns,Llabel=Llabel,Lnames=Lnames,Rlabel=Rlabel,Rnames=Rnames,ext=ext,type=type)
  end
  export largeLRenv

#end
