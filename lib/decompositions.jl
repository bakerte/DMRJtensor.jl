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
  Module: decompositions

Decompose a tensor

See also: [`contractions`](@ref)
"""
#=
module decompositions
using ..tensor
using ..contractions
import LinearAlgebra
=#


#       +---------------------------------------+
#       |                                       |
#>------+            SVD of a tensor            +---------<
#       |                                       |
#       +---------------------------------------+
"""
    U,D,V = libsvd(X)

Chooses the best svd function for tensor decomposition with the standard three tensor output for the SVD
"""
@inline function libsvd(X::Array{W,2})::Tuple{Matrix{W},Vector{Float64},Matrix{W}} where W <: Number
  F = LinearAlgebra.svd(X)
  return F.U,F.S,Array(F.Vt)
end

"""
    D,U = libeigen(AA[,B])

Chooses the best eigenvalue decomposition function for tensor `AA` with the standard three tensor output. Can include an overlap matrix for generalized eigenvalue decompositions `B`
"""
@inline function libeigen(AA::Array{W,2},B::Array{W,2}...) where W <: Number
  if eltype(AA) <: Complex
    M = LinearAlgebra.Hermitian((AA+AA')/2)
  else
    M = LinearAlgebra.Symmetric((AA+AA')/2)
  end
  return length(B) == 0 ? LinearAlgebra.eigen(M) : LinearAlgebra.eigen(M,B[1])
end

"""
    Q,R = libqr(X[,decomposer=LinearAlgebra.qr])

Decomposes `X` with a QR decomposition.
"""
@inline function libqr(R::Array{W,2};decomposer::Function=LinearAlgebra.qr) where W <: Number
  P,Q = decomposer(R)
  X,Y = Array(P),Array(Q)
  if size(P,2) > size(Q,1)
    U = X[:,1:size(Y,1)]
    V = Y
  elseif size(P,2) < size(Q,1)
    U = X
    V = Y[1:size(X,2),:]
  else
    U,V = X,Y
  end
  return U,V
end

"""
  L,Q = libqr(X[,decomposer=LinearAlgebra.lq])

Decomposes `X` with a LQ decomposition.
"""
@inline function liblq(R::Array{W,2};decomposer::Function=LinearAlgebra.lq) where W <: Number
  return libqr(R,decomposer=decomposer)
end

"""
  newm,sizeD,truncerr,sumD = findnewm(D,m,minm,mag,cutoff,effZero,nozeros,power,keepdeg)

Determines the maximum bond dimension to keep for SVD and eigenvalue decompositions based on:
  + The input diagonal matrix `D` (given as a vector here)
  + The maximum bond dimension `m`
  + The minimal bond dimension `minm`
  + The magnitude of the original tensor `mag`
  + The cutoff value `cutoff`
  + `effZero` contains the effective value of zero (default: 1E-16)
  + `nozeros` will remove all values that are approximately zero
  + `power` is the power to which the values of `D` are raised to
  + `keepdeg` is the boolean parameter whether to keep degenerate values or not

The outputs are the size of the new bond (`newm`), size of the input `D` tensor (`sizeD`), truncation error (`truncerr`), and magnitude of the original tensor (`sumD` which is skipped if the input is not 0.)

All parameters can be set in `svd` or `eigen` or similar.
"""
function findnewm(D::Array{W,1},m::Integer,minm::Integer,mag::Float64,cutoff::Real,effZero::Real,nozeros::Bool,power::Number,keepdeg::Bool) where {W <: Number}
  sizeD = size(D,1)
  p = sizeD
  if mag == 0.
    sumD = 0.
    @inbounds @simd for a = 1:size(D,1)
      sumD += abs(D[a])^2
    end
  else
    sumD = mag
  end

  truncerr = 0.

  if cutoff > 0.
    modcutoff = sumD*cutoff
    @inbounds truncadd = abs(D[p])^power
    @inbounds while p > 0 && ((truncerr + truncadd < modcutoff) || (nozeros && abs(D[p]) < effZero))
      truncerr += truncadd
      p -= 1
      truncadd = abs(D[p])^power
    end
    if keepdeg
      while p < length(D) && isapprox(D[p],D[p+1])
        p += 1
      end
    end
    thism = m == 0 ? max(min(p,sizeD),minm) : max(min(m,p,sizeD),minm)
  else
    thism = m == 0 ? max(sizeD,minm) : max(min(m,sizeD),minm)
  end

  return thism,sizeD,truncerr,sumD
end

# 
"""
  U,D,V = recursive_SVD(A,tol)

Iterative SVD for increased precision of matrix `A` to a tolerance `tol`. Adapted from Phys. Rev. B 87, 155137 (2013)

WARNING: This function is not known to be practically useful except for finding more accurate singular values. The use in algorithms has not been tested by the developers of DMRjulia to find a good use.
"""
function recursive_SVD(AA::Array{W,2},tolerance::Float64=1E-4)::Tuple{Matrix{W},Vector{Float64},Matrix{W}} where W <: Number
  U,D,V = safesvd(AA)
  counter = 2
  anchor = 1
  while counter <= size(D,1)
    if D[counter]/D[anchor] < tolerance
      Lpsi = U[:,counter:size(D,1)]
      Rpsi = V[counter:size(D,1),:]
      X = LinearAlgebra.dot(Lpsi,AA,Rpsi)
      Up,Dp,Vp = safesvd(X)
      for a = counter:size(D,1)
        U[:,a] = sum(p->U[:,p]*Up[p-counter+1,a-counter+1],counter:size(D,1))
        V[a,:] = sum(p->V[p,:]*Vp[p-counter+1,a-counter+1],counter:size(D,1))
      end
      D[counter:size(D,1)] = Dp
      anchor = counter
    end
    counter += 1
  end
  newD = Array(LinearAlgebra.Diagonal(D))
  return U,newD,V
end

"""
    U,D,V = safesvd(AA)

Evaluates SVD with checks on the matrix-equivalent of a tensor

# Explanation (v1.1.1-v1.5.3 and beyond):

For some highly degenerate matrices, julia's svd function will throw an error (interface issue with LAPACK).
This try/catch function will test the svd output and use a modified algorithm with lower preicision (fast),
a high precision algorithm (slow), and then throw an error. 
Messages and the offending matrix are printed when this occurs so it is known when this occurs.
The problem often self-corrects as the computation goes along in a standard DMRG computation for affected models.
Typically, this is just a warning and not a cause for concern.
"""
@inline function (safesvd(AA::Array{W,2})) where W <: Number
  try
    libsvd(AA)
  catch
    try
      println("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
      println("!!!! Ill-conditioned matrix !!!!!")
      println("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
      println("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
      println("!!!!! SQRT SVD ACTIVATED !!!!!")
      println("(loss of precision for this SVD...will be restored later)")
      println("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
      if size(AA,1) < size(AA,2)
        if eltype(AA) <: Complex
          M = LinearAlgebra.Hermitian(AA*Array(AA'))
        else
          M = LinearAlgebra.Symmetric(AA*Array(AA'))
        end
        D,U = LinearAlgebra.eigen(M)
        sqrtD = sqrt.(D)
        invsqrtD = Array(LinearAlgebra.Diagonal(Float64[1/sqrtD[i] for i = 1:size(D,1)]))
        Vt = invsqrtD * Array(U') * AA
      else
        if eltype(AA) <: Complex
          M = LinearAlgebra.Hermitian(AA'*Array(AA))
        else
          M = LinearAlgebra.Symmetric(AA'*Array(AA))
        end
        D,V = LinearAlgebra.eigen(M)
        sqrtD = sqrt.(D)
        invsqrtD = Array(LinearAlgebra.Diagonal(Float64[1/sqrtD[i] for i = 1:size(D,1)]))
        U = AA * V * invsqrtD
        Vt = Array(V')
      end
      return U,sqrtD,Vt
      catch
      try
        println("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        println("!!!! Ill-conditioned matrix !!!!!")
        println("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        println("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        println("!!!!! SPECIAL SVD ACTIVATED !!!!!")
        println("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        Mtype = eltype(AA)
        AAdag = Array(AA')
        M = LinearAlgebra.Hermitian(Mtype[zeros(Mtype,size(AA,2),size(AA,2)) AAdag;
                        AA zeros(Mtype,size(AA,1),size(AA,1))])
        D,U = LinearAlgebra.eigen(M)
        condition = size(AA,1) < size(AA,2)
        if size(AA,1) < size(AA,2)
          sortvec = sortperm(D,rev=true)[1:size(AA,1)]
        else
          sortvec = sortperm(D,rev=true)[1:size(AA,2)]
        end
        newU = sqrt(2)*U[size(AA,2)+1:size(U,1),sortvec]
        newD = D[sortvec]
        newV = Array(sqrt(2)*(U[1:size(AA,2),sortvec])')
        return newU,newD,newV
      catch
        println("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        println("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        println("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        println("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        println("!!!!!!!!!!SVD BADNESS!!!!!!!!!!!!")
        println("!!!!!!!!!!~~~EVIL~~~!!!!!!!!!!!!!")
        println("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        println("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        println("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        println("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        println(map(x -> Printf.@sprintf("%.16f",x), AA))
        println()
        println()
        error("LAPACK functions are not working!")
        return Array{W,2}(undef,0,0),Array{Float64,1}(undef,0),Array{W,2}(undef,0,0)
      end
    end
  end
end

"""
    U,D,V,truncerr,newmag = svd(AA[,cutoff=0.,m=0,mag=0.,minm=2,nozeros=false,recursive=false])

SVD routine with truncation; accepts Qtensors

# Arguments
- `cutoff::Float64`: the maximum sum of singular values squared that can be discarded
- `m::Int64`: maximum bond dimension (does not truncate if 0)
- `mag::Int64`: magnitude of the tensor (calculated here if 0)
- `nozeros::Bool`: toggles whether singular values that are zero are discarded (false by default)

# Warning:

We recommend not defining `using LinearAlgebra` to avoid conflicts.  Instead, define
```
import LinearAlgebra
```
and define functions as `LinearAlgebra.svd` to use functions from that package.

"""
function svd(AA::Array{W,2};power::Number=2,cutoff::Float64 = 0.,m::Integer = 0,mag::Float64=0.,minm::Integer=2,
              nozeros::Bool=false,recursive::Bool = false,effZero::Float64 = 1E-16,keepdeg::Bool=false) where W <: Number
    U,D,Vt = recursive ? recursive_SVD(AA) : safesvd(AA)
    thism,sizeD,truncerr,sumD = findnewm(D,m,minm,mag,cutoff,effZero,nozeros,power,keepdeg)
    if sizeD > thism
      @inbounds Utrunc = U[:,1:thism]
      @inbounds Dtrunc = D[1:thism]
      @inbounds Vtrunc = Vt[1:thism,:]
    elseif sizeD < thism
      Utrunc = zeros(eltype(U),size(U,1),thism)
      Dtrunc = zeros(eltype(D),thism)
      Vtrunc = zeros(eltype(Vt),thism,size(Vt,2))
      @inbounds Utrunc[:,1:size(U,2)] = U
      @inbounds Dtrunc[1:size(D,1)] = D
      @inbounds Vtrunc[1:size(Vt,1),:] = Vt
    else
      Utrunc = U
      Dtrunc = D
      Vtrunc = Vt
    end
    Darray = Array(LinearAlgebra.Diagonal(Dtrunc))::Array{Float64,2}
    return Utrunc,Darray,Vtrunc,truncerr,sumD
end
export svd

function svd(AA::tens{W};power::Number=2,cutoff::Float64 = 0.,
          m::Integer = 0,mag::Float64=0.,minm::Integer=2,nozeros::Bool=false,
          recursive::Bool = false,effZero::Number = 1E-16,keepdeg::Bool=false) where W <: Number
  rAA = reshape(AA.T,size(AA)...)
  U,D,V,truncerr,sumD = svd(rAA,power=power,cutoff=cutoff,m=m,mag=mag,minm=minm,nozeros=nozeros,recursive=recursive,effZero=effZero,keepdeg=keepdeg)
  tensU = tens(W,U)
  tensD = tens(Float64,D)
  tensV = tens(W,V)
  return tensU,tensD,tensV,truncerr,sumD
end

"""
    D,U = eigen(AA[,B,cutoff=,m=,mag=,minm=,nozeros=])

Eigensolver routine with truncation (output to `D` and `U` just as `LinearAlgebra`'s function); accepts Qtensors; arguments similar to `svd`; can perform generalized eigenvalue problem with `B` overlap matrix

See also: [`svd`](@ref)
"""
function eigen(AA::Array{W,2},B::Array{W,2}...;cutoff::Float64 = 0.,m::Integer = 0,mag::Float64=0.,
                minm::Integer=1,nozeros::Bool=false,power::Number=1,effZero::Float64 = 1E-16,keepdeg::Bool=false) where {W <: Number}

  Dsq,U = libeigen(AA,B...)

  thism,sizeD,truncerr,sumD = findnewm(Dsq,m,minm,mag,cutoff,effZero,nozeros,power,keepdeg)
  if sizeD > thism
    @inbounds Utrunc = U[:,sizeD-thism+1:sizeD]
    @inbounds Dtrunc = Dsq[sizeD-thism+1:sizeD]
  elseif sizeD < minm
    Utrunc = zeros(eltype(U),size(U,1),minm)
    Dtrunc = zeros(eltype(Dsq),minm)
    @inbounds Utrunc[:,1:thism] = U
    @inbounds Dtrunc[1:thism] = Dsq
  else
    Utrunc = U
    Dtrunc = Dsq
  end
  finalD = Array(LinearAlgebra.Diagonal(Dtrunc))::Array{Float64,2}
  return finalD,Utrunc,truncerr,sumD
end
export eigen

function eigen(AA::tens{W},B::tens{R}...;cutoff::Float64 = 0.,
              m::Integer = 0,mag::Float64=0.,minm::Integer=2,
              nozeros::Bool=false,recursive::Bool = false,keepdeg::Bool=false) where {W <: Number,R <: Number}
  X = reshape(AA.T,size(AA)...)
  D,U,truncerr,sumD = eigen(X,B...,cutoff=cutoff,m=m,mag=mag,minm=minm,nozeros=nozeros,keepdeg=keepdeg)
  tensD = tens(Float64,D)
  tensU = tens(T,U)
  return tensD,tensU,truncerr,mag
end

import .LinearAlgebra.eigvals
"""
  vec = eigvals(A)

Eigenvalues of input `A` (allowing julia's arrays and `denstens` types) output to a vector `vec`

See also: [`svd`](@ref)
"""
function eigvals(A::Array{W,2}) where {W <: Number, N}
  return LinearAlgebra.eigvals(A)
end

function eigvals(A::tens{W}) where W <: Number
  thisarray = makeArray(A)
  vals = eigvals(thisarray)
  return tens(vals)
end

"""
  rAA,Lsizes,Rsizes = getorder(AA,vec)

Obtains the present state of an input tensor `AA` grouped into two groups `vec` (i.e., [[1,2],[3,4,5]] or any order). The output is the reshaped tensor and sizes of each dimension for rows and columns.
"""
function getorder(AA::G,vecA::Array{Array{W,1},1}) where {G <: TensType, W <: Integer}
  order = Array{W,1}(undef,sum(a->length(vecA[a]),1:length(vecA)))
  counter = 0
  for b = 1:length(vecA)
    @inbounds @simd for j = 1:length(vecA[b])
      counter += 1
      order[counter] = vecA[b][j]
    end
  end

  if !(issorted(order))
    AB = permutedims(AA,order)
    Lvec = [i for i = 1:length(vecA[1])]
    Rvec = [i + length(vecA[1]) for i = 1:length(vecA[2])]
    rAA = reshape(AB,[Lvec,Rvec]) #should always be rank-2 here
  else
    rAA = reshape(AA,vecA) #should always be rank-2 here
  end

  Lsizes = [size(AA,vecA[1][i]) for i = 1:length(vecA[1])]
  Rsizes = [size(AA,vecA[2][i]) for i = 1:length(vecA[2])]
  return rAA,Lsizes,Rsizes
end

"""
    U,D,V,truncerr,mag = svd(AA,vecA[,cutoff=,m=,mag=,minm=,nozeros=])

Reshapes `AA` for `svd` and then unreshapes U and V matrices on return; `vecA` is of the form [[1,2],[3,4,5]] and must be length 2 with the elements representing the grouped indices for the left and right sets of the SVD for use in unreshaping later
"""
function svd(AA::TensType,vecA::Array{Array{W,1},1};cutoff::Float64 = 0.,
            m::Integer = 0,mag::Float64=0.,minm::Integer=1,nozeros::Bool=false,
            recursive::Bool = false,power::Number=2,keepdeg::Bool=false) where {W <: Integer}
  AB,Lsizes,Rsizes = getorder(AA,vecA)
  U,D,V,truncerr,newmag = svd(AB,power = power,cutoff=cutoff,m=m,mag=mag,minm=minm,nozeros=nozeros,recursive=recursive,keepdeg=keepdeg)

  outU = unreshape!(U,Lsizes...,size(D,1))
  outV = unreshape!(V,size(D,2),Rsizes...)
  return outU,D,outV,truncerr,newmag
end

"""
    D,U,truncerr,mag = eigen(AA,vecA[,B,cutoff=,m=,mag=,minm=,nozeros=])

reshapes `AA` for `eigen` and then unreshapes U matrix on return; `vecA` is of the form [[1,2],[3,4,5]] and must be length 2
"""
function eigen(AA::TensType,vecA::Array{Array{W,1},1},
                B::TensType...;cutoff::Float64 = 0.,m::Integer = 0,mag::Float64=0.,
                minm::Integer=1,nozeros::Bool=false,keepdeg::Bool=false) where W <: Integer
  AB,Lsizes,Rsizes = getorder(AA,vecA)
  D,U,truncerr,newmag = eigen(AB,B...,cutoff=cutoff,m=m,mag=mag,minm=minm,nozeros=nozeros,keepdeg=keepdeg)
  outU = unreshape!(U,Lsizes...,size(D,1))
  return D,outU,truncerr,newmag
end   

"""
    Q,R,0.,1. = qr(AA,vecA[,cutoff=,m=,mag=,minm=,nozeros=])

Reshapes `AA` for `svd` and then unreshapes U and V matrices on return; `vecA` is of the form [[1,2],[3,4,5]] and must be length 2
"""
function qr(AA::TensType,vecA::Array{Array{W,1},1}) where W <: Integer
  AB,Lsizes,Rsizes = getorder(AA,vecA)
  Qmat,Rmat,truncerr,newmag = qr(AB)

  innerdim = size(Qmat,2)

  outU = unreshape!(Qmat,Lsizes...,innerdim)
  outV = unreshape!(Rmat,innerdim,Rsizes...)
  return outU,outV,truncerr,newmag
end

function qr(AA::Array{W,2};decomposer::Function=LinearAlgebra.qr) where W <: Number
  return libqr(AA,decomposer=decomposer)
end

function qr(AA::denstens;decomposer::Function=LinearAlgebra.qr)
  rAA = makeArray(AA)
  Q,R = libqr(rAA,decomposer=decomposer)
  return tens(Q),tens(R),0.,1.
end

"""
    L,Q,0.,1. = lq(AA,vecA[,cutoff=,m=,mag=,minm=,nozeros=])

Reshapes `AA` for `svd` and then unreshapes U and V matrices on return; `vecA` is of the form [[1,2],[3,4,5]] and must be length 2
"""
function lq(AA::TensType,vecA::Array{Array{W,1},1}) where W <: Integer
  AB,Lsizes,Rsizes = getorder(AA,vecA)
  Lmat,Qmat,truncerr,newmag = lq(AB)

  innerdim = size(Qmat,1)

  outU = unreshape!(Lmat,Lsizes...,innerdim)
  outV = unreshape!(Qmat,innerdim,Rsizes...)
  return outU,outV,truncerr,newmag
end

function lq(AA::Array{W,2};decomposer::Function=LinearAlgebra.lq) where W <: Number
  return libqr(AA,decomposer=decomposer)
end

function lq(AA::denstens;decomposer::Function=LinearAlgebra.lq)
  rAA = makeArray(AA)
  L,Q = libqr(rAA,decomposer=decomposer)
  return tens(L),tens(Q),0.,1.
end

"""
    LTens,RTens,D,truncerr,newmag = polar(A,group=[,*])

*-options from `svd`

Performs a polar decomposition on tensor `A` with grouping `group` (default: [[1,2],[3]] for an MPS); if `left` (default), this returns U*D*U',U'*V else functoin returns U*V',V*D*V' from an `svd`

See also: [`svd`](@ref)
"""
function polar(AA::TensType,group::Array{Array{W,1},1};
                right::Bool=true,cutoff::Float64 = 0.,m::Integer = 0,mag::Float64 = 0.,
                minm::Integer=1,nozeros::Bool=false,recursive::Bool=false,outermaxm::Integer=0,keepdeg::Bool=false) where W <: Integer
  AB,Lsizes,Rsizes = getorder(AA,group)
  U,D,V,truncerr,newmag = svd(AB,cutoff=cutoff,m=m,mag=mag,minm=minm,nozeros=nozeros,keepdeg=keepdeg)
  U = unreshape!(U,Lsizes...,size(D,1))
  V = unreshape!(V,size(D,2),Rsizes...)
  #polar decomposition
  if right
    if outermaxm > 0
      outermtrunc = m > 0 ? min(m,outermaxm,size(V,2)) : min(outermaxm,size(V,2))
      truncV = getindex(V,:,1:outermtrunc)
      modV = V
    else
      truncV = V
      modV = V
    end
    DV = contract(D,2,truncV,1)
    rR = ccontract(modV,1,DV,1)
    rU = contract(U,3,modV,1)
    leftTensor,rightTensor = rU,rR
  else
    if outermaxm > 0
      outmtrunc = m > 0 ? min(m,size(U,1),outermaxm) : min(outermaxm,size(U,1))
      truncU = getindex(U,1:outmtrunc,:)
      modU = U
    else
      truncU = U
      modU = U
    end   
    UD = contract(truncU,2,D,1)
    lR = contractc(UD,2,U,2)
    lU = contract(U,2,V,1)
    leftTensor,rightTensor = lR,lU
  end
  return leftTensor,rightTensor,D,truncerr,newmag
end
export polar






#       +---------------------------------------+
#       |                                       |
#>------+  Quantum Number conserving operation  +---------<
#       |                                       |
#       +---------------------------------------+
#using ..QN
#using ..Qtensor

function makeU(nQN::Integer,keepq::Array{Bool,1},outU::Array{Array{W,2},1},QtensA::Qtens{W,Q},
                finalinds::Array{Array{P,2},1},newqindexL::Array{Array{P,1},1},newqindexLsum::Array{Array{Q,1},1},
                leftflux::Bool,Linds::Array{P,1},thism::Integer) where {W <: Number, Q <: Qnum, P <: Integer}
  finalnQN = sum(keepq)
  finalUinds = Array{NTuple{2,Array{P,2}},1}(undef,finalnQN)
  newQblocksum = Array{NTuple{2,Q},1}(undef,finalnQN)
  counter = 0
  for q = 1:nQN
    if keepq[q]
      counter += 1
      left = QtensA.ind[q][1]
      right = finalinds[q]
      finalUinds[counter] = (left,right)
      
      newQblocksum[counter] = (QtensA.Qblocksum[q][1],newqindexLsum[1][q])
    end
  end
  finalUQnumMat = vcat(QtensA.QnumMat[Linds],newqindexL)
  Uflux = leftflux ? QtensA.flux : Q()

  newUQsize = [[i for i = 1:length(Linds)],[length(Linds) + 1]]
  newUblocks = (newUQsize...,)
  finalUQnumSum = vcat(QtensA.QnumSum[Linds],newqindexLsum)

  return Qtens{W,Q}(newUQsize,outU,finalUinds,newUblocks,newQblocksum,finalUQnumMat,finalUQnumSum,Uflux)
end

function makeV(nQN::Integer,keepq::Array{Bool,1},outV::Array{Array{W,2},1},QtensA::Qtens{W,Q},
                finalinds::Array{Array{P,2},1},newqindexR::Array{Array{P,1},1},newqindexRsum::Array{Array{Q,1},1},
                leftflux::Bool,Rinds::Array{P,1},thism::Integer) where {W <: Number, Q <: Qnum, P <: Integer}
  finalnQN = sum(keepq)
  finalVinds = Array{NTuple{2,Array{P,2}},1}(undef,finalnQN)
  newQblocksum = Array{NTuple{2,Q},1}(undef,finalnQN)
  counter = 0
  for q = 1:nQN
    if keepq[q]
      counter += 1
      left = finalinds[q]
      right = QtensA.ind[q][2]
      finalVinds[counter] = (left,right)

      newQblocksum[counter] = (newqindexRsum[1][q],QtensA.Qblocksum[q][2])
    end
  end
  finalVQnumMat = vcat(newqindexR,QtensA.QnumMat[Rinds])
  Vflux = !leftflux ? QtensA.flux : Q()

  newVQsize = [[1],[i+1 for i = 1:length(Rinds)]]
  newVblocks = (newVQsize...,)

  finalVQnumSum = vcat(newqindexRsum,QtensA.QnumSum[Rinds])

  return Qtens{W,Q}(newVQsize,outV,finalVinds,newVblocks,newQblocksum,finalVQnumMat,finalVQnumSum,Vflux)
end




function makeD(nQN::Integer,keepq::Array{Bool,1},outD::Array{Array{W,2},1},QtensA::Qtens{W,Q},
                finalinds::Array{Array{P,2},1},newqindexL::Array{Array{P,1},1},newqindexR::Array{Array{P,1},1},
                newqindexRsum::Array{Array{Q,1},1},newqindexLsum::Array{Array{Q,1},1},thism::Integer) where {W <: Number, Q <: Qnum, P <: Integer}
  finalnQN = sum(keepq)
  finalDinds = Array{NTuple{2,Array{P,2}},1}(undef,finalnQN)
  newQblocksum = Array{NTuple{2,Q},1}(undef,finalnQN)
  counter = 0
  for q = 1:nQN
    if keepq[q]
      counter += 1
      left = finalinds[q]
      right = finalinds[q]
      finalDinds[counter] = (left,right)

      newQblocksum[counter] = (newqindexRsum[1][q],newqindexLsum[1][q])
    end
  end
  finalDQnumMat = vcat(newqindexR,newqindexL)

  Dflux = Q()
  newDQsize = [[1],[2]]
  newDblocks = (newDQsize...,)
  finalDQnumSum = vcat(newqindexRsum,newqindexLsum)

  return Qtens{W,Q}(newDQsize,outD,finalDinds,newDblocks,newQblocksum,finalDQnumMat,finalDQnumSum,Dflux)
end

function truncate(sizeinnerm::Integer,newD::Array{Array{W,1},1},power::Number,
                  nozeros::Bool,cutoff::Float64,m::Integer,nQN::Integer,mag::Float64,keepdeg::Bool) where W <: Number

  rhodiag = Array{Float64,1}(undef,sizeinnerm)
  counter = 0

  @inbounds #=Threads.@threads=# for q = 1:nQN
    offset = q == 1 ? 0 : sum(w->length(newD[w]),1:q-1)
    newvals = newD[q]
    for x = 1:length(newvals)
      rhodiag[x + offset] = abs(newvals[x])^power
    end
  end

  if isapprox(mag,0.)
    sumD = sum(x->rhodiag[x],1:length(rhodiag))
  else
    sumD = mag
  end
  modcutoff = cutoff * sumD

  order = sortperm(rhodiag,rev=true)

  truncerr = 0.
  y = length(order) #m == 0 ? length(order) : min(m,length(order))

  truncadd = rhodiag[order[y]]
  while y > 0 && ((truncerr + truncadd < modcutoff) || (nozeros && isapprox(truncadd,0.)))
    truncerr += truncadd
    y -= 1
    truncadd = rhodiag[order[y]]
  end
  if keepdeg
    while y < length(rhodiag) && isapprox(rhodiag[order[y]],rhodiag[order[y+1]])
      y += 1
    end
  end
  thism = max(m == 0 ? y : min(m,y),1)

  qstarts = vcat([0],[sum(w->length(newD[w]),1:q-1) for q = 2:nQN],[sizeinnerm])
  qranges = [UnitRange(qstarts[q]+1,qstarts[q+1]) for q = 1:nQN]  
  keepers = [Array{intType,1}(undef,thism) for q = 1:nQN]
  sizekeepers = zeros(intType,nQN)
  #=Threads.@threads=# for q = 1:nQN
    for x = 1:thism
      if order[x] in qranges[q]
        sizekeepers[q] += 1
        keepers[q][sizekeepers[q]] = order[x] - qstarts[q]
      end
    end
  end
  keepers = [keepers[q][1:sizekeepers[q]] for q = 1:nQN]
  finalinds = Array{Array{intType,2},1}(undef,nQN)
  #=Threads.@threads=# for q = 1:nQN
    offset = q > 1 ? sum(w->length(keepers[w]),1:(q-1)) : 0
    tempvec = [i + offset - 1 for i = 1:length(keepers[q])]
    finalinds[q] = reshape(tempvec,1,length(keepers[q]))
  end
  return finalinds,thism,qstarts,qranges,keepers,truncerr,sumD
end

function threeterm(arr::Array{Array{W,2},1};decomposer::Function=safesvd) where W <: Number
  nQN = length(arr)
  newU = Array{Array{W,2},1}(undef,nQN)
  newD = Array{Array{W,1},1}(undef,nQN)
  newV = Array{Array{W,2},1}(undef,nQN)
  #=Threads.@threads=# for q = 1:nQN
    newU[q],newD[q],newV[q] = decomposer(arr[q])
  end
  return newU,newD,newV
end

function svd(QtensA::Qtens{W,Q};cutoff::Float64 = 0.,m::Integer = 0,
              minm::Integer=1,nozeros::Bool=true,recursive::Bool=false,
              power::Number=2,leftflux::Bool=false,mag::Float64=0.,
              effZero::Real=1E-16,keepdeg::Bool=false) where {W <: Number, Q <: Qnum}


  Tsize = QtensA.size
  Linds = Tsize[1]
  Rinds = Tsize[2]

  A = changeblock(QtensA,Linds,Rinds)
  nQN = length(A.T)

  LRinds = 1
  QNsummary = [inv(A.Qblocksum[q][LRinds]) for q = 1:nQN]

  newU,newD,newV = threeterm(A.T)
  
  sizeinnerm = sum(q->size(newD[q],1),1:nQN)
    
  finalinds,thism,qstarts,qranges,keepers,truncerr,sumD = truncate(sizeinnerm,newD,power,nozeros,cutoff,m,nQN,mag,keepdeg)

  newqindexL = [Array{intType,1}(undef,thism)]
  keepq = Array{Bool,1}(undef,nQN)

  tempD = Array{Array{W,2},1}(undef,nQN)
  @inbounds for q = 1:nQN
    keepq[q] = length(keepers[q]) > 0
    if keepq[q]
      newU[q] = newU[q][:,keepers[q]]

      dimD = length(keepers[q])
      inputD = LinearAlgebra.Diagonal(newD[q][1:dimD])
      tempD[q] = Array(inputD)
      newV[q] = newV[q][keepers[q],:]
      offset = q == 1 ? 0 : sum(w->length(finalinds[w]),1:q-1)
      @simd for i = 1:length(finalinds[q])
        newqindexL[1][i + offset] = q
      end
    end
  end

  newqindexLsum = [QNsummary]
  newqindexRsum = [inv.(QNsummary)]
  newqindexR = newqindexL


  if sum(keepq) < nQN
    outU = newU[keepq]
    outD = tempD[keepq]
    outV = newV[keepq]
  else
    outU = newU
    outD = tempD
    outV = newV
  end


  U = makeU(nQN,keepq,outU,A,finalinds,newqindexL,newqindexLsum,leftflux,Linds,thism)
  D = makeD(nQN,keepq,outD,A,finalinds,newqindexL,newqindexR,newqindexRsum,newqindexLsum,thism)
  V = makeV(nQN,keepq,outV,A,finalinds,newqindexR,newqindexRsum,leftflux,Rinds,thism)
  return U,D,V,truncerr,sumD
end







function twoterm(arr::Array{Array{W,2},1};decomposer::Function=libeigen,centerRank::Integer=1) where W <: Number
  nQN = length(arr)
  newU = Array{Array{W,2},1}(undef,nQN)
  newD = Array{Array{W,centerRank},1}(undef,nQN)
  #=Threads.@threads=# for q = 1:nQN
    newD[q],newU[q] = decomposer(arr[q])
  end
  return newD,newU
end

function eigen(QtensA::Qtens{W,Q};cutoff::Float64 = 0.,m::Integer = 0,
              minm::Integer=1,nozeros::Bool=false,recursive::Bool=false,
              power::Number=1,leftflux::Bool=false,mag::Float64=0.,decomposer::Function=libeigen,keepdeg::Bool=false) where {W <: Number, Q <: Qnum}

  Tsize = QtensA.size
  Linds = Tsize[1]
  Rinds = Tsize[2]

  A = changeblock(QtensA,Linds,Rinds)
  nQN = length(A.T)

  LRinds = 1
  QNsummary = [inv(A.Qblocksum[q][LRinds]) for q = 1:nQN]

  

  newD,newU = twoterm(A.T,decomposer=decomposer)

  tempD = Array{Array{W,2},1}(undef,nQN)

  sizeinnerm = sum(q->size(newD[q],1),1:nQN)
    
  finalinds,thism,qstarts,qranges,keepers,truncerr,sumD = truncate(sizeinnerm,newD,power,nozeros,cutoff,m,nQN,mag,keepdeg)

  newqindexL = [Array{intType,1}(undef,thism)]
  keepq = Array{Bool,1}(undef,nQN)

  @inbounds for q = 1:nQN
    keepq[q] = length(keepers[q]) > 0
    if keepq[q]
      newU[q] = newU[q][:,keepers[q]]
      dimD = length(keepers[q])
 
      inputD = LinearAlgebra.Diagonal(newD[q][1:dimD])
      tempD[q] = Array(inputD)
      offset = q == 1 ? 0 : sum(w->length(finalinds[w]),1:q-1)
      @simd for i = 1:length(finalinds[q])
        newqindexL[1][i + offset] = q
      end
    end
  end

  newqindexLsum = [QNsummary]
  newqindexRsum = [inv.(QNsummary)]
  newqindexR = newqindexL


  if sum(keepq) < nQN
    outU = newU[keepq]
    outD = tempD[keepq]
  else
    outU = newU
    outD = tempD
  end

  U = makeU(nQN,keepq,outU,A,finalinds,newqindexL,newqindexLsum,leftflux,Linds,thism)
  D = makeD(nQN,keepq,outD,A,finalinds,newqindexL,newqindexR,newqindexRsum,newqindexLsum,thism)

  return D,U,truncerr,sumD
end

function qr(QtensA::Qtens{W,Q};leftflux::Bool=false,decomposer::Function=libqr,mag::Number=1.) where {W <: Number, Q <: Qnum}

  Rsize = QtensA.size
  Linds = Rsize[1]
  Rinds = Rsize[2]

  A = changeblock(QtensA,Linds,Rinds)

  nQN = length(A.T)

  LRinds = 1
  QNsummary = [inv(A.Qblocksum[q][LRinds]) for q = 1:nQN]

  newU,newV = twoterm(A.T,decomposer=decomposer,centerRank=2)

  sizeinnerm = sum(q->size(newU[q],2),1:nQN)

  thism = sizeinnerm
  sumD = mag

  outU,outV = newU,newV

  qstarts = vcat([0],[sum(w->size(newU[w],2),1:q-1) for q = 2:nQN],[thism])
  qranges = [UnitRange(qstarts[q]+1,qstarts[q+1]) for q = 1:nQN]

  truncerr = 0.

  newqindexL = [Array{intType,1}(undef,thism)]
  finalinds = [reshape([i-1 for i = qranges[q]],1,length(qranges[q])) for q = 1:length(QNsummary)]
  @inbounds for q = 1:nQN
    offset = q == 1 ? 0 : sum(w->length(finalinds[w]),1:q-1)
    @simd for i = 1:length(finalinds[q])
      newqindexL[1][i + offset] = q
    end
  end

  newqindexLsum = [QNsummary]
  newqindexRsum = [inv.(QNsummary)]
  newqindexR = newqindexL

  keepq = [true for q = 1:nQN]

  U = makeU(nQN,keepq,outU,A,finalinds,newqindexL,newqindexLsum,leftflux,Linds,thism)
  V = makeV(nQN,keepq,outV,A,finalinds,newqindexR,newqindexRsum,leftflux,Rinds,thism)

  return U,V,truncerr,mag
end
export qr

function lq(QtensA::Qtens{W,Q};leftflux::Bool=false) where {W <: Number, Q <: Qnum}
  return qr(QtensA,leftflux=leftflux,decomposer=liblq)
end
export lq



















@inline function svdvals(A::AbstractVecOrMat)
  return LinearAlgebra.svdvals(A)
end

function svdvals(A::denstens)
  B = makeArray(A) 
  return LinearAlgebra.svdvals(B)
end

function makeDvec(nQN::Integer,keepq::Array{Bool,1},outD::Array{Array{W,2},1},QtensA::Qtens{W,Q},
                finalinds::Array{Array{P,2},1},newqindexL::Array{Array{P,1},1},newqindexR::Array{Array{P,1},1},
                newqindexRsum::Array{Array{Q,1},1},newqindexLsum::Array{Array{Q,1},1},thism::Integer) where {W <: Number, Q <: Qnum, P <: Integer}
  finalnQN = sum(keepq)
  finalDinds = Array{NTuple{2,Array{P,2}},1}(undef,finalnQN)
  newQblocksum = Array{NTuple{2,Q},1}(undef,finalnQN)
  counter = 0
  for q = 1:nQN
    if keepq[q]
      counter += 1
      left = finalinds[q]
      right = [1]
      finalDinds[counter] = (left,right)

      newQblocksum[counter] = (newqindexRsum[1][q],newqindexLsum[1][q])
    end
  end
  finalDQnumMat = vcat(newqindexR,newqindexL)

  Dflux = Q()
  newDQsize = [[1],intType[]]
  newDblocks = (newDQsize...,)
  finalDQnumSum = vcat(newqindexRsum,Q)

  return Qtens{W,Q}(newDQsize,outD,finalDinds,newDblocks,newQblocksum,finalDQnumMat,finalDQnumSum,Dflux)
end

function svdvals(QtensA::Qtens{W,Q};mag::Number=1.) where {W <: Number, Q <: Qnum}
  Rsize = QtensA.size
  Linds = Rsize[1]
  Rinds = Rsize[2]

  A = changeblock(QtensA,Linds,Rinds)

  nQN = length(A.T)
  newD = [svdvals(A.T[q]) for q = 1:nQN]
  return vcat(newD...)
end

function nullspace(A::TensType; left::Bool=false)#atol::Real = 0.0, rtol::Real = (min(size(A, 1), size(A, 2))*eps(real(float(one(eltype(A))))))*iszero(atol))
  #  m, n = size(A, 1), size(A, 2)
  #  (m == 0 || n == 0) && return Matrix{eigtype(eltype(A))}(I, n, n)
  U,D,V = svd(A)
  Dvals = [searchindex(D,i,i) for i = 1:size(D,1)]
  tol = max(atol, Dvals[1]*rtol)
  indstart = sum(s -> s .> tol, Dvals) + 1

  minval = minimum(abs.(Dvals))
  ipos = findall(w->isapprox(abs(Dvals[w]),minval),1:length(Dvals))

  if length(ipos) > 1
    g = rand(length(ipos))
    minpos = ipos[g]
  else
    minpos = ipos[1]
  end
  outTens = left ? U[:,minpos:minpos] : V[minpos:minpos,:]
  return outTens
end


#end
