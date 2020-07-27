#########################################################################
#
#  Density Matrix Renormalization Group (and other methods) in julia (DMRjulia)
#                              v0.1
#
#########################################################################
# Made by Thomas E. Baker (2018)
# See accompanying license with this program
# This code is native to the julia programming language (v1.1.0) or (v1.5)
#

"""
    Module: MPutil

Functions and definitions for MPS and MPO
"""
module MPutil
using ..tensor
using ..QN
using ..Qtensor
using ..Qtask
using ..contractions
using ..decompositions

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
#=
  """
      envType

  Vector that holds environments (rank N)
  """
  abstract type envType end
  export envType
=#
  """
      `matrixproductstate` 
      
  struct to hold MPS tensors and orthogonality center

  ```
  A::Array{Array{Number,3},N} (vector of MPS tensors)
  oc::Int64 (orthogonality center)
  ```
  # Note:
  + Contruct this through [`MPS`](@ref)
  """
  mutable struct matrixproductstate{W} <: MPS where W <: Union{Array{Array{T,N},1},Array{tens{T},1},Array{Qtens{T,Q},1}} where {N,T <: Number,Q <: Qnum}
    A::W
    oc::intType
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
  mutable struct matrixproductoperator{W} <: MPO where W <: Union{Array{Array{T,N},1},Array{tens{T},1},Array{Qtens{T,Q},1}} where {N,T <: Number,Q <: Qnum}
    H::W
  end
#=
  """
      `envVec`

  Array that holds environment tensors
  """
  mutable struct envVec{W} <: envType where W <: Union{Array{Array{T,N},1},Array{tens{T},1},Array{Qtens{T,Q},1}} where {N,T <: Number,Q <: Qnum}
    V::Array{W,1}
  end

  function envVec(T::TensType...)
    return envVec([T...])
  end
  export envVec
=#
  """
    makeoc(Ns[,oc])

  processes `oc` to get location of the orthogonality center (first entry, default 1) for a system with `Ns` sites
  """
  function makeoc(Ns::Integer,oc::intType...)
    if length(oc) > 0
      @assert(length(oc) == 1)
      @assert(0 < oc[1] <= Ns)
      return oc[1]
    else
      return 1
    end
  end

  """
      MPS([T,]A[,oc])

  constructor for MPS with tensors `A` and orhtogonailty center `oc`; can optinally request an element type `T` for the tensors
  """
  function MPS(psi::Array{W,1},oc::intType...;regtens::Bool=false)::MPS where W <: TensType
    if !regtens && typeof(psi[1]) <: AbstractArray
      tenspsi = [tens(psi[a]) for a = 1:length(psi)]
    else
      tenspsi = copy(psi)
    end
    return matrixproductstate(tenspsi,makeoc(length(psi),oc...))
  end

  function MPS(thistype::DataType,B::Array{W,1},oc::intType...;regtens::Bool=false)::MPS where W <: AbstractArray # where T <: Number
    if !regtens #&& typeof(psi[1]) <: AbstractArray
      MPSvec = [tens(convert(Array{thistype,ndims(B[i])},copy(B[i]))) for i = 1:size(B,1)]
    else
      MPSvec = [convert(Array{thistype,ndims(B[i])},copy(B[i])) for i = 1:size(B,1)]
    end
    return MPS(MPSvec,makeoc(length(B),oc...))
  end

  function MPS(thistype::DataType,B::Array{W,1},oc::intType...;regtens::Bool=false)::MPS where W <: qarray # where T <: Number
#    if !regtens #&& typeof(psi[1]) <: AbstractArray
      MPSvec = [convertQtens(thistype,copy(B[i])) for i = 1:size(B,1)]
#    else
#      MPSvec = [convertQtens(thistype,B[i]) for i = 1:size(B,1)]
#    end
    return MPS(MPSvec,makeoc(length(B),oc...))
  end

  function MPS(thistype::DataType,B::Array{W,1},oc::intType...;regtens::Bool=false)::MPS where W <: denstens # where T <: Number
#    if !regtens #&& typeof(psi[1]) <: AbstractArray
      MPSvec = [convertTens(thistype, copy(B[i])) for i = 1:size(B,1)]
#    else
#      MPSvec = [convert(Array{thistype,ndims(B[i])},copy(B[i])) for i = 1:size(B,1)]
#    end
    return MPS(MPSvec,makeoc(length(B),oc...))
  end

  function MPS(T::Type,mps::MPS)
    return MPS(T,mps.A,mps.oc)
  end

  """
      MPO([T,]H)

  constructor for MPO with tensors `H`; can optinally request an element type `T` for the tensors
  """
  function MPO(H::Array{W,1};regtens::Bool=false)::MPO where W <: TensType
    T = prod(a->eltype(H[a])(1),1:size(H,1))
    if !regtens && (typeof(H[1]) <: AbstractArray)
      M = [tens(H[a]) for a = 1:length(H)]
    else
      M = H
    end
    return MPO(typeof(T),M,regtens=regtens)
  end

  function MPO(T::DataType,H::Array{W,1};regtens::Bool=false)::MPO where W <: TensType
    QN = typeof(H[1]) <: qarray
    if QN
      newH = Array{qarray,1}(undef,size(H,1))
      for a = 1:size(H,1)
        if ndims(H[a]) == 2
          rP = reshape(H[a],size(H[a],1),size(H[a],2),1,1)
          newH[a] = permutedims!(rP,[4,1,2,3])
        else
          newH[a] = H[a]
        end
      end
    else
      newH = W <: AbstractArray ? Array{Array{T,4},1}(undef,size(H,1)) : Array{denstens,1}(undef,size(H,1))
      for a = 1:size(H,1)
        if ndims(H[a]) == 2
          rP = reshape(H[a],size(H[a],1),size(H[a],2),1,1)
          newH[a] = permutedims!(rP,[4,1,2,3])
        else
          newH[a] = H[a]
        end
      end
    end
    if !regtens && (typeof(newH[1]) <: AbstractArray)
      newH = [tens(newH[a]) for a = 1:length(newH)]
    else
      newH = newH
    end
    return matrixproductoperator(newH)
  end

  function MPO(T::Type,mpo::MPO,regtens::Bool=false)
    return MPO(T,mpo.H,regtens=regtens)
  end
  
  import Base.size
  """
      size(H[,i])
  
  size prints out the size of the tensor field of an envType, MPS, or MPO; this is effectively the number of sites
  """
  function size(H::MPO)
    return size(H.H)
  end
  function size(H::MPO,i::intType)
    return size(H.H,i)
  end
  function size(psi::MPS)
    return size(psi.A)
  end
  function size(psi::MPS,i::intType)
    return size(psi.A,i)
  end
  #=
  function size(G::envType)
    return size(G.V)
  end
  function size(G::envType,i::intType)
    return size(G.V,i)
  end=#
  import Base.length
  function length(H::MPO)
    return length(H.H)
  end
  function length(psi::MPS)
    return length(psi.A)
  end
#=  function length(G::envType)
    return length(G.V)
  end=#

  import Base.eltype
  """
      eltype(Y)

  eltype gets element type of the envType, MPS, or MPO tensor fields
  """
  function eltype(Y::MPS)
    return eltype(Y.A)
  end
  function eltype(H::MPO)
    return eltype(H.H)
  end
#=  function eltype(G::envType)
    return eltype(G.V)
  end=#

  import Base.getindex
  """
      getindex(A,i...)

  getindex allows to retrieve elements to an envType, MPS or MPO (ex: W = psi[1])
  """
  function getindex(A::MPS,i::intType)::TensType
    return A.A[i]
  end
  function getindex(A::MPS,r::UnitRange{W})::TensType where W <: Integer
    return A.A[r]
  end
  function getindex(H::MPO,i::intType)::TensType
    return H.H[i]
  end
  function getindex(H::MPO,r::UnitRange{W})::TensType where W <: Integer
    return H.H[r]
  end
  #=
  function getindex(G::envType,i::intType)
    return G.V[i]
  end
  function getindex(G::envType,r::UnitRange{W}) where W <: Integer
    return G.V[r]
  end
  =#

  import Base.lastindex
  """
      psi[end]

  lastindex! allows to get the end element of an envType, MPS, or MPO
  """
  function lastindex(A::MPS)
    return lastindex(A.A)
  end
  function lastindex(H::MPO)
    return lastindex(H.H)
  end
  #=
  function lastindex(G::envType)
    return lastindex(G.V)
  end
  =#

  import Base.setindex!
  """
      psi[1] = W

  setindex! allows to assign elements to an envType, MPS, or MPO (ex: psi[1] = W)
  """
  function setindex!(H::MPO,A::TensType,i::Int64)
    H.H[i] = A
    nothing
  end
  function setindex!(H::MPS,A::TensType,i::Int64)
    H.A[i] = A
    nothing
  end
  #=
  function setindex!(G::envType,A::TensType,i::Int64)
    G.V[i] = A
    nothing
  end
  =#

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

  function copy(mps::MPS)::MPS
    return MPS([copy(mps.A[i]) for i = 1:length(mps)],copy(mps.oc))
  end

  function copy(mpo::MPO)::MPO
    return MPO([copy(mpo.H[i]) for i = 1:length(mpo)])
  end
#=
  function copy(G::envType)::envType
    return envVec{eltype(G[1])}([copy(G.V[i]) for i = 1:length(G)])
  end
=#
  import Base.conj!
  """
      conj!(psi)

  Conjugates all elements in an MPS in-place

  See also [`conj`](@ref)
  """
  function conj!(A::MPS)
    conj!.(A.A)
    nothing
  end

  import Base.conj
  """
      A = conj(psi)
  
  Conjugates all elements in an MPS and makes a copy

  See also [`conj!`](@ref)
  """
  function conj(A::MPS)
    B = copy(A)
    conj!.(B.A)
    return B
  end

#       +---------------------------------------+
#>------+  move! orthogonality center in MPS    +---------<
#       +---------------------------------------+

  """
      moveR(Lpsi,Rpsi[,cutoff=,m=,minm=,condition=])

  moves `psi`'s orthogonality center from `Lpsi` to `Rpsi`, with a maximum bond dimenion `m`, cutoff `cutoff`, minimum bond dimension `minm`; toggle truncation of tensor during movement with `condition`; returns psi[iL],psi[iR],D from the SVD, and the truncation error

  See also: [`moveR!`](@ref)
  """
  function moveR(Lpsi::P,Rpsi::P;cutoff::Float64=0.,m::Integer=0,minm::Integer=0,condition::Bool=false,
                  recursive::Bool=false) where P <: Union{Qtens{R,Z},Array{R,3},tens{R}} where {R <: Number, Z <: Qnum} #::Tuple{Union{Qtens{W,Q},Array{W,N}},Union{Qtens{W,Q},Array{W,N}},Union{Qtens{W,Q},Array{W,2}},P} where {W <: Number, Q <: Qnum, P <: Number, N})
#    sizes = size(Lpsi)
#    rLpsi = reshape!(Lpsi,[[1,2],[3]])
    Ltens,D,V,truncerr,sumD = svd(Lpsi,[[1,2],[3]],cutoff=cutoff,m=m,minm=minm,recursive=recursive)

#    Ltens = unreshape!(U,sizes[1],sizes[2],size(D,1))

    modV = (condition ? getindex!(V,:,1:size(Rpsi,1)) : V) #::Union{Qtens{R,Z},Array{R,2},tens{R}} where Z <: Qnum
    DV = contract(D,[2],modV,[1]) #::Union{Qtens{R,Z},Array{R,2},tens{R}} where Z <: Qnum
    Rtens = contract(DV,2,Rpsi,1) #::Union{Qtens{R,Z},Array{R,3},tens{R}} where Z <: Qnum
    return Ltens,Rtens,D,truncerr
  end
  export moveR

  """
      moveL(Lpsi,Rpsi[,cutoff=,m=,minm=,condition=])

  moves `psi`'s orthogonality center from `Rpsi` to `Lpsi`, with a maximum bond dimenion `m`, cutoff `cutoff`, minimum bond dimension `minm`; toggle truncation of tensor during movement with `condition`; returns psi[iL],psi[iR],D from the SVD, and the truncation error

  See also: [`moveL!`](@ref)
  """
  function moveL(Lpsi::P,Rpsi::P;cutoff::Float64=0.,m::Integer=0,minm::Integer=0,condition::Bool=false,
                  recursive::Bool=false) where P <: Union{Qtens{R,Z},Array{R,3},tens{R}}  where {R <: Number, Z <: Qnum} #::Tuple{Union{Qtens{W,Q},Array{W,N}},Union{Qtens{W,Q},Array{W,N}},Union{Qtens{W,Q},Array{W,2}},P} where {W <: Number, Q <: Qnum, P <: Number, N})
#    sizes = size(Rpsi)
#    rRpsi = reshape!(Rpsi,[[1],[2,3]])

    U,D,Rtens,truncerr,sumD = svd(Rpsi,[[1],[2,3]],cutoff=cutoff,m=m,minm=minm,recursive=recursive)

#    Rtens = unreshape!(V,size(D,2),sizes[2],sizes[3])

    modU = (condition ? getindex!(U,1:size(Lpsi,ndims(Lpsi)),:) : U) #::Union{Qtens{R,Z},Array{R,2},tens{R}} where Z <: Qnum
    UD = contract(modU,[2],D,[1]) #::Union{Qtens{R,Z},Array{R,2},tens{R}} where Z <: Qnum
    Ltens = contract(Lpsi,ndims(Lpsi),UD,1) #::Union{Qtens{R,Z},Array{R,3},tens{R}} where Z <: Qnum
    return Ltens,Rtens,D,truncerr
  end
  export moveL

  """
      moveR!(psi,iL,iR[,cutoff=,m=,minm=,condition=])
  
  acts in-place to move `psi`'s orthogonality center from `iL` to `iR`, with a maximum bond dimenion `m`, cutoff `cutoff`, and minimum bond dimension `minm`; toggle truncation of tensor during movement with `condition`

  See also: [`moveR`](@ref)
  """
  function moveR!(psi::MPS,iL::Integer,iR::Integer;cutoff::Float64=0.,m::Integer=0,minm::Integer=0,condition::Bool=false,
                   recursive::Bool=false) #::Tuple{T,Q} where {T <: TensType, Q <: Number})
    psi[iL],psi[iR],D,truncerr = moveR(psi[iL],psi[iR],cutoff=cutoff,m=m,minm=minm,condition=condition,recursive=recursive)
    return D,truncerr
  end
  export moveR!

  """
      moveL!(psi,iL,iR[,cutoff=,m=,minm=,condition=])

  acts in-place to move `psi`'s orthogonality center from `iR` to `iL`, with a maximum bond dimenion `m`, cutoff `cutoff`, and minimum bond dimension `minm`; toggle truncation of tensor during movement with `condition`

  See also: [`moveL`](@ref)
  """
  function moveL!(psi::MPS,iL::Integer,iR::Integer;cutoff::Float64=0.,m::Integer=0,minm::Integer=0,condition::Bool=false,
                   recursive::Bool=false) #::Tuple{T,Q} where {T <: TensType, Q <: Number})
    psi[iL],psi[iR],D,truncerr = moveL(psi[iL],psi[iR],cutoff=cutoff,m=m,minm=minm,condition=condition,recursive=recursive)
    return D,truncerr
  end
  export moveL!

  """
      mainmove!(psi,newoc[,m=,cutoff=,minm=,Lfct=,Rfct=])

  movement function to move `psi` to a new site, `newoc` with `Lfct` and `Rfct`, with a maximum bond dimenion `m`, cutoff `cutoff`, and minimum bond dimension `minm`; toggle truncation of tensor during movement with `condition`

  See also: [`move!`](@ref) [`move`](@ref)
  """
  function mainmove!(psi::MPS,pos::Integer;cutoff::Float64=1E-14,m::Integer=0,minm::Integer=0,
                        Lfct::Function=moveR,Rfct::Function=moveL)
    if m == 0
      m = maximum([maximum(size(psi[i])) for i = 1:size(psi,1)])
    end
    while psi.oc != pos
      if psi.oc < pos
        iL = psi.oc
        iR = psi.oc+1
        psi[iL],psi[iR],D,truncerr = Lfct(psi[iL],psi[iR],cutoff=cutoff,m=m,minm=minm)
        psi.oc = iR
      else
        iL = psi.oc-1
        iR = psi.oc
        psi[iL],psi[iR],D,truncerr = Rfct(psi[iL],psi[iR],cutoff=cutoff,m=m,minm=minm)
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
  function move!(mps::MPS,pos::intType;m::Integer=0,cutoff::Float64=1E-14,minm::Integer=0)
    mainmove!(mps,pos,cutoff=cutoff,m=m,minm=minm)
    nothing
  end
  export move!

  """
      move(psi,newoc[,m=,cutoff=,minm=])

  same as `move!` but makes a copy of `psi`

  See also: [`move!`](@ref)
  """
  function move(mps::MPS,pos::intType;m::Integer=0,cutoff::Float64=1E-14,minm::Integer=0)
    newmps = copy(mps)
    mainmove!(newmps,pos,cutoff=cutoff,m=m,minm=minm)
    return newmps
  end
  export move

  """
      boundaryMove!([dualpsi,]psi,i,mpo,Lenv,Renv)

  Move orthogonality center of `psi` to site `i` and updates left and right environments `Lenv` and `Renv` using MPO `mpo`

  See also: [`move!`](@ref)
  """
  function boundaryMove!(psi::MPS,i::Integer,Lenv::Array{P,1},
                          Renv::Array{P,1},mpo::MPO...;mover::Function=move!) where P <: TensType
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

  function boundaryMove!(dualpsi::MPS,psi::MPS,i::Integer,Lenv::Array{P,1},
                          Renv::Array{P,1},mpo::MPO...;mover::Function=move!) where P <: TensType
    origoc = psi.oc
    if origoc < i
      mover(psi,i)
      mover(dualpsi,i)
      for w = origoc:i-1
        Lenv[w+1] = Lupdate(Lenv[w],dualpsi[w],psi[w],[mpo[a][w] for a = 1:length(mpo)]...)
#        Lupdate!(w,Lenv,dualpsi,psi,mpo...)
      end
    elseif origoc > i
      mover(psi,i)
      mover(dualpsi,i)
      for w = origoc:-1:i+1
        Renv[w-1] = Rupdate(Renv[w],dualpsi[w],psi[w],[mpo[a][w] for a = 1:length(mpo)]...)
#        Rupdate!(w,Renv,dualpsi,psi,mpo...)
      end
    end
    nothing
  end
  export boundaryMove!

  """
      boundaryMove([dualpsi,]psi,i,mpo,Lenv,Renv)

  Copies `psi` and moves orthogonality center of `psi` to site `i` and updates left and right environments `Lenv` and `Renv` using MPO `mpo`

  See also: [`move!`](@ref)
  """
  function boundaryMove(dualpsi,psi::MPS,i::Integer,mpo::MPO,Lenv::Array{P,1},
                          Renv::Array{P,1}) where P <: TensType
    newpsi = copy(psi)
    newdualpsi = copy(dualpsi)
    newLenv = copy(Lenv)
    newRenv = copy(Renv)
    boundaryMove!(newdualpsi,newpsi,i,mpo,newLenv,newRenv)
    return newdualpsi,newpsi,newLenv,newRenv
  end

  function boundaryMove(psi::MPS,i::Integer,mpo::MPO,Lenv::Array{P,1},
                          Renv::Array{P,1}) where P <: TensType
    newdualpsi,newpsi,newLenv,newRenv = boundaryMove(psi,psi,i,mpo,Lenv,Renv)
    return newpsi,newLenv,newRenv
  end
  export boundaryMove

  """
      applyMPO(psi,H[,m=,cutoff=])
  
  Applies MPO (`H`) to an MPS (`psi`) and truncates to maximum bond dimension `m` and cutoff `cutoff`
  """
  function applyMPO(psi::MPS,H::MPO;m::intType=0,cutoff::Float64=0.)::MPS
    if m == 0
      m = maximum([size(psi[i],ndims(psi[i])) for i = 1:size(psi.A,1)])
    end

    thissize = size(psi,1)
    newpsi = [contract([1,3,4,2,5],psi[i],2,H[i],2) for i = 1:thissize]

    if typeof(psi[1]) <: qarray
      finalpsi = Array{qarray,1}(undef,thissize)
    elseif typeof(psi[1]) <: denstens
      finalpsi = Array{denstens,1}(undef,thissize)
    else
      finalpsi = Array{AbstractArray,1}(undef,thissize)
    end
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

#       +---------------------------------------+
#>------+       measurement operations          +---------<
#       +---------------------------------------+

  """
      expect(dualpsi,psi,H[,Lbound=,Rbound=])

  evaluate <`dualpsi`|`H`|`psi`> for any number of MPOs `H`; can specifcy left and right boundaries (`Lbound` and `Rbound`); `dualpsi` is conjugated inside of the algorithm

  See also: [`overlap`](@ref)
  """
  function expect(dualpsi::MPS,psi::MPS,H::MPO...;Lbound::TensType=[0],Rbound::TensType=[0],order::Array{Int64,1}=Int64[])::Number
    Ns = size(psi,1)
    nMPOs = size(H,1)
    nLsize = nMPOs+2
    if ndims(psi[1]) == 2
      Lenv = psi[1]
      for j = 1:nMPOs
        Lenv = contract(H[j][1],1,Lenv,1)
      end
      Lenv = ccontract(dualpsi[1],1,Lenv,1)

      for i = 2:size(psi,1)-1
        Lenv = contractc(Lenv,1,dualpsi[i],1)
        for j = 1:nMPOs
          Lenv = contract(Lenv,[1,nLsize],H[j][i],[1,3])
        end
        Lenv = contract(Lenv,[1,nLsize],psi[i],[1,2])
      end
      Lenv = contract(Lenv,nLsize,psi[Ns],1)
      for j = 1:nMPOs
        Lenv = contract(Lenv,[nLsize-j,nLsize-(j-1)],H[j][Ns],[1,2])
      end
      return contractc(Lenv,dualpsi[Ns])
    else
      Lenv,Renv = makeEnds(dualpsi,psi,H...,Lbound=Lbound,Rbound=Rbound)

      for i = 1:size(psi,1)
        Lenv = contractc(Lenv,1,dualpsi[i],1)
        for j = 1:nMPOs
          Lenv = contract(Lenv,[1,nLsize],H[j][i],[1,3])
        end
        Lenv = contract(Lenv,[1,nLsize],psi[i],[1,2])
      end

      if order == Int64[]
        permvec = vcat([ndims(Renv)],[i for i = 2:ndims(Renv)-1],[1])
        modRenv = permutedims(Renv,permvec)
      else
        modRenv = Renv
      end
      
      return contract(Lenv,modRenv)
    end
  end

  """
      expect(psi,H[,Lbound=,Rbound=])

  evaluate <`psi`|`H`|`psi`> for any number of MPOs `H`; can specifcy left and right boundaries (`Lbound` and `Rbound`)

  See also: [`overlap`](@ref)
  """
  function expect(psi::MPS,H::MPO...;Lbound::TensType=[0],Rbound::TensType=[0])
    return expect(psi,psi,H...,Lbound=Lbound,Rbound=Rbound)
  end
  export expect

  """
      overlap(dualpsi,psi[,Lbound=,Rbound=])

  evaluate <`dualpsi`|`psi`>; can specifcy left and right boundaries (`Lbound` and `Rbound`)

  See also: [`expect`](@ref)
  """
  function overlap(dualpsi::MPS,psi::MPS;Lbound::TensType=[0],Rbound::TensType=[0])::Number
    return expect(dualpsi,psi,Lbound=Lbound,Rbound=Rbound)
  end
  export overlap

  """
      correlation(dualpsi,psi,Cc,Ca[,F,silent=])

  Compute the correlation funciton (example, <`dualpsi`|`Cc`_i . `Ca`_j|`psi`>) on all sites; can also specify a trail `F` for Fermion strings; `silent` toggles output of every line

  # Note:
  + More efficient than using `mpoterm`s
  + Use `mpoterm` and `applyMPO` for higher order correlation functions or write a new function
  """
  function correlation(dualpsi::MPS, psi::MPS, mCc::Q, mCa::R, F::S...;silent::Bool = true) where {Q <: TensType,R <: TensType,S <: TensType}
#    @assert(size(psi,1) == size(dualpsi,1))
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
    Lenv,Renv = makeEnds(dualpsi,psi)
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
        rho[i,j] += DMElement
        rho[j,i] += conj(DMElement)
        if !silent
          println("Printing element: ",i," , ",j," ",DMElement)
        end
      end
    end
    return rho
  end

  """
      correlation(psi,Cc,Ca[,F,silent=])

  Compute the correlation funciton (example, <`psi`|`Cc`_i . `Ca`_j|`psi`>) on all sites; can also specify a trail `F` for Fermion strings; `silent` toggles output of every line

  # Example:
  ```julia
  Cup, Cdn, Nup, Ndn, Ndens, F, O, Id = fermionOps()
  rho = correlation(psi,Cup',Cup,F) #density matrix
  ```
  """
  function correlation(psi::MPS, Cc::Q, Ca::R, F::S...;silent::Bool = true) where {Q <: TensType,R <: TensType,S <: TensType}
    return correlation(psi,psi,Cc,Ca,F...,silent=silent)
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
      makeQstore(numlegs,dualpsi,psi[,mpovec...,left=,rightind=])

  Generates quantum number for an environment tensor (helper function for making environments with Qtensors)
  
  # Arguments:
  + `numlegs::intType`: number of legs of the environment created
  + `dualpsi::MPS`: dual MPS
  + `psi::MPS`: MPS
  + `mpovec::MPO`: MPO tensor
  + `left::Bool`: toggle for left or right to get correct arrow convention
  + `rightind::intType`: number of the right link index
  """
  function makeQstore(numlegs::intType,dualpsi::MPS,psi::MPS,mpovec::MPO...;left::Bool=true,rightind::intType=0)
    LQNs = Array{typeof(psi[1].flux),1}(undef,numlegs)
    Ns = size(psi,1)
    site = left ? 1 : Ns
    for i = 1:numlegs
      if i == 1
        index = left ? 1 : ndims(dualpsi[Ns])
        LQNs[i] = copy(dualpsi[site].QnumMat[index][1])
      elseif i==numlegs
        if rightind == 0
          index = left ? 1 : ndims(dualpsi[Ns])
        else
          index = left ? 1 : rightind
        end
        LQNs[i] = copy(psi[site].QnumMat[index][1])
      else
        index = left ? 1 : ndims(mpovec[i-1][Ns])
        LQNs[i] = copy(mpovec[i-1][site].QnumMat[index][1])
      end
    end
    return LQNs
  end
  export Qstore

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
  function makeEnds(dualpsi::MPS,psi::MPS,mpovec::MPO...;Lbound::TensType=[0],Rbound::TensType=[0])
#    @assert(size(psi,1) == size(dualpsi,1))
    numlegs = size(mpovec,1)+2
    retType = typeof(eltype(psi[1])(1)*eltype(psi[2])(1))
    if typeof(psi[1]) <: qarray
      if Lbound == [0]
        Larrows = [i==1 ? false : true for i = 1:numlegs]
        LQNs = makeQstore(numlegs,dualpsi,psi,mpovec...)
        Lout = makeBoundary(LQNs,Larrows,retType=retType)
      else
        Lout = copy(Lbound) #?????????
      end
      if Rbound == [0]
        Rarrows = [i==numlegs ? true : false for i = 1:numlegs]
        RQNs = makeQstore(numlegs,dualpsi,psi,mpovec...,left=false)
        Rout = makeBoundary(RQNs,Rarrows,retType=retType)
      else
        Rout = copy(Rbound) #?????????
      end
    else
      if Lbound == [0] || Rbound ==[0]
        boundary = ones(retType,ones(intType,numlegs)...)
        if typeof(psi[1]) <: denstens
          boundary = tens(boundary)
        end
      end
      if Lbound == [0]
        Lout = boundary
      else
        Lout = copy(Lbound)
      end
      if Rbound == [0]
        Rout = boundary
      else
        Rout = copy(Rbound)
      end
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
  function makeEnds(psi::MPS,mpovec::MPO...;Lbound::TensType=[0],Rbound::TensType=[0])
    return makeEnds(psi,psi,mpovec...,Lbound=Lbound,Rbound=Rbound)
  end
  export makeEnds

  """
      makeEnv(dualpsi,psi,mpo[,Lbound=,Rbound=])

  Generates environment tensors for a MPS (`psi` and its dual `dualpsi`) with boundaries `Lbound` and `Rbound`
  """
  function makeEnv(dualpsi::MPS,psi::MPS,mpo::MPO...;Lbound::TensType=[0],
            Rbound::TensType=[0]) where B <: TensType
    @assert(length(mpo) < 2)
    Ns = size(psi,1)
    if length(mpo) == 1
      numtype = typeof(eltype(psi[1])(1)*eltype(mpo[1][1])(1))
    else
      numtype = eltype(psi[1])
    end
    if typeof(psi[1]) <: qarray
      thistype = Qtens
    elseif typeof(psi[1]) <: denstens
      thistype = tens
    else
      thistype = Array{numtype,3}
    end
    Lenv = Array{thistype,1}(undef,Ns)
    Renv = Array{thistype,1}(undef,Ns)
    Lenv[1],Renv[Ns] = makeEnds(dualpsi,psi,mpo...,Lbound=Lbound,Rbound=Rbound)
    if length(mpo) == 1
      for i = Ns:-1:psi.oc+1
        Rupdate!(i,Renv,dualpsi,psi,mpo[1])
      end
    else
      for i = Ns:-1:psi.oc+1
        Renv[i-1] = Rupdate(Renv[i],dualpsi[i],psi[i])
      end
    end
    for i = psi.oc-1:-1:1
      thisind = i + 1
      T = typeof(Renv[thisind])
      if T <: qarray
        Renv[i] = T() #Tenstype() #fill with empty object
      elseif typeof(psi[1]) <: denstens
        Renv[i] = tens(type=eltype(psi[1]))
      else
        Renv[i] = T(undef,zeros(Int64,length(size(Renv[thisind])))...)
      end
    end
    if length(mpo) == 1
      for i = 1:psi.oc-1
        Lupdate!(i,Lenv,dualpsi,psi,mpo[1])
      end
    else
      for i = 1:psi.oc-1
        Lenv[i+1] = Lupdate!(Lenv[i],dualpsi[i],psi[i])
      end
    end
    for i = psi.oc+1:Ns
      thisind = i - 1
      T = typeof(Lenv[thisind])
      if T <: AbstractArray
        Lenv[i] = T(undef,zeros(Int64,length(size(Lenv[thisind])))...)
      elseif T <: qarray
        Lenv[i] = T() #Tenstype() #fill with empty object
      elseif T <: tens
        Lenv[i] = tens(type=eltype(Lenv[thisind])) #Tenstype() #fill with empty object
      end
    end
    return Lenv,Renv
  end

  """
      makeEnv(psi,mpo[,Lbound=,Rbound=])

  Generates environment tensors for a MPS (`psi`) with boundaries `Lbound` and `Rbound`
  """
  function makeEnv(psi::MPS,mpo::MPO;Lbound::TensType=[0],Rbound::TensType=[0]) where B <: TensType
    return makeEnv(psi,psi,mpo,Lbound=Lbound,Rbound=Rbound)
  end
  export makeEnv

  """
      Lupdate(Lenv,dualpsi,psi,mpo)

  Updates left environment tensor `Lenv` with dual MPS `dualpsi`, MPS `psi`, and MPO `mpo`
  """
  function  Lupdate(Lenv::X,dualpsi::Y,psi::Y,mpo::Z...) where {X <: Union{Qtens{A,Q},Array{A,3},tens{A}}, Y <: Union{Qtens{B,Q},Array{B,3},tens{B}},Z <: Union{Qtens{W,Q},Array{W,4},tens{W}}} where {A <: Number, B <: Number, W <: Number, Q <: Qnum}
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
  function  Rupdate(Renv::X,dualpsi::Y,psi::Y,mpo::Z...) where {X <: Union{Qtens{A,Q},Array{A,3},tens{A}}, Y <: Union{Qtens{B,Q},Array{B,3},tens{B}},Z <: Union{Qtens{W,Q},Array{W,4},tens{W}}} where {A <: Number, B <: Number, W <: Number, Q <: Qnum}
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
  function Lupdate!(i::Integer,Lenv::AbstractArray,dualpsi::MPS,psi::MPS,mpo::MPO) where B <: TensType
    Lenv[i+1] = Lupdate(Lenv[i],dualpsi[i],psi[i],mpo[i])
    nothing
  end

  """
      Lupdate!(i,Lenv,psi,mpo)

  Updates left environment's (`Lenv`) `i`+1 site from info on the site `i` based on MPS (`psi) and MPO (`mpo`)
  """
  function Lupdate!(i::Integer,Lenv::AbstractArray,psi::MPS,mpo::MPO) where B <: TensType
    Lupdate!(i,Lenv,psi,psi,mpo)
  end
  export Lupdate!

  """
      Rupdate!(i,Renv,dualpsi,psi,mpo)

  Updates right environment's (`Renv`) `i`-1 site from info on the site `i` based on dual MPS (`dualpsi`), MPS (`psi), and MPO (`mpo`)
  """
  function Rupdate!(i::Integer,Renv::AbstractArray,dualpsi::MPS,psi::MPS,mpo::MPO) where B <: TensType
    Renv[i-1] = Rupdate(Renv[i],dualpsi[i],psi[i],mpo[i])
    nothing
  end

  """
      Rupdate!(i,Renv,psi,mpo)

  Updates right environment's (`Renv`) `i`-1 site from info on the site `i` based on MPS (`psi) and MPO (`mpo`)
  """
  function Rupdate!(i::Integer,Renv::AbstractArray,psi::MPS,mpo::MPO) where B <: TensType
    Rupdate!(i,Renv,psi,psi,mpo)
  end
  export Rupdate!






















  """
      Lupdate(Lenv,dualpsi,psi,mpo)

  Updates left environment tensor `Lenv` with MPS `psi` and MPO `mpo`
  """
  #can not resolve if it is this one or the previous one
#  function  Lupdate(Lenv::X,psi::Y,mpo::Z...) where {X <: Union{Qtens{A,Q},Array{A,3},tens{A}}, Y <: Union{Qtens{B,Q},Array{B,3},tens{B}},Z <: Union{Qtens{W,Q},Array{W,4},tens{W}}} where {A <: Number, B <: Number, W <: Number, Q <: Qnum}
#    return Lupdate(Lenv,psi,psi,mpo...)
#  end

  """
      Rupdate(Renv,psi,mpo)

  Updates right environment tensor `Renv` with MPS `psi` and MPO `mpo`
  """
#  function  Rupdate(Renv::X,psi::Y,mpo::Z...) where {X <: Union{Qtens{A,Q},Array{A,3},tens{A}}, Y <: Union{Qtens{B,Q},Array{B,3},tens{B}},Z <: Union{Qtens{W,Q},Array{W,4},tens{W}}} where {A <: Number, B <: Number, W <: Number, Q <: Qnum}
#    return Rupdate(Renv,psi,psi,mpo...)
#  end
end
