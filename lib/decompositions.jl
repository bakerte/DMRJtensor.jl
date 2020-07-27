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
    Module: decompositions

Decompose a tensor

See also: [`contractions`](@ref)
"""
module decompositions
using ..tensor
using ..QN
using ..Qtensor
using ..Qtask
using ..contractions
import LinearAlgebra

using Printf

const effZero = 1E-16

#       +---------------------------------------+
#       |                                       |
#>------+            SVD of a tensor            +---------<
#       |                                       |
#       +---------------------------------------+

# iterative SVD for increased precision
  function recursive_SVD(AA::AbstractArray,tolerance::Float64=1E-4)
    U,D,V = safesvd(AA) #LinearAlgebra.svd(AA)
    counter = 2
    anchor = 1
    while counter <= size(D,1)
      if D[counter]/D[anchor] < tolerance
        X = U[:,counter:size(D,1)]' * AA * V[counter:size(D,1),:]'
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
  export recursive_SVD

  """
      safesvd(AA)
  Evaluates SVD with checks on the matrix-equivalent of a tensor

  # Explanation:
  
  For some highly degenerate matrices, julia's svd function will throw an error (interface issue with LAPACK).
  This try/catch function will test the svd output and use a modified algorithm with lower preicision (fast),
  a high precision algorithm (slow), and then throw an error. 
  Messages and the offending matrix are printed when this occurs so it is known when this occurs.
  """
  function (safesvd(AA::Array{T,2})) where T <: Number
    try
      F = LinearAlgebra.svd(AA)
      F.U,F.S,Array(F.Vt)
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
        U,sqrtD,Vt
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
          newU,newD,newV
        catch
          println("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
          println("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
          println("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
          println("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
          println("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
          println("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
          println("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
          println("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
          println()
          println()
          error("LAPACK functions are not working!")
          Array{Float64,2}(undef,0,0),Array{Float64,1}(undef,0),Array{Float64,2}(undef,0,0)
        end
      end
    end
  end

  """
      svd(AA[,cutoff=,m=,mag=,minm=,nozeros=])
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
  function svd(AA::Array{T,2};method::String="square",cutoff::Float64 = 0.,m::intType = 0,mag::Float64=0.,minm::intType=2,
               nozeros::Bool=false,recursive::Bool = false) where T <: Number #::Tuple{Array{W,2},Array{Float64,2},Array{W,2},Float64,Float64} where {W <: Number}) where T <: Number
      U,D,Vt = recursive ? recursive_SVD(AA) : safesvd(AA)
      sizeD = size(D,1)
      p = sizeD
      if mag == 0.
        sumD = 0.
        @simd for a = 1:size(D,1)
          sumD += D[a]^2
        end
      else
        sumD = mag
      end
      modcutoff = sumD*cutoff
      truncerr = 0.
      if method == "square"
        while p > 0 && ((truncerr + D[p]^2 < modcutoff) || (nozeros && abs(D[p]) < effZero))
          truncerr += D[p]^2
          p -= 1
        end
      elseif method == "sqrt"
        while p > 0 && ((truncerr + sqrt(D[p]) < modcutoff) || (nozeros && abs(D[p]) < effZero))
          truncerr += sqrt(D[p])
          p -= 1
        end
      else #method == "sqrt"
        while p > 0 && ((truncerr + D[p] < modcutoff) || (nozeros && abs(D[p]) < effZero))
          truncerr += D[p]
          p -= 1
        end
      end
      thism = m == 0 ? max(min(p,sizeD),minm) : max(min(m,p,sizeD),minm)
      if sizeD > thism
        Utrunc = U[:,1:thism]::Array{W,2} where W <: Number
        Dtrunc = D[1:thism]
        Vtrunc = Vt[1:thism,:]::Array{W,2} where W <: Number
      elseif sizeD < thism
        Utrunc = zeros(eltype(U),size(U,1),thism)
        Dtrunc = zeros(eltype(D),thism)
        Vtrunc = zeros(eltype(Vt),thism,size(Vt,2))
        Utrunc[:,1:size(U,2)] = U::Array{W,2} where W <: Number
        Dtrunc[1:size(D,1)] = D
        Vtrunc[1:size(Vt,1),:] = Vt::Array{W,2} where W <: Number
      else
        Utrunc = U::Array{W,2} where W <: Number
        Dtrunc = D
        Vtrunc = Vt::Array{W,2} where W <: Number
      end
      Darray = Array(LinearAlgebra.Diagonal(Dtrunc))::Array{Float64,2}
      finaltrunc = (thism != p ? 1 - sum(a->Dtrunc[a]^2,1:thism)/sumD : truncerr)::Float64
      return Utrunc,Darray,Vtrunc,finaltrunc,sumD
  end
  export svd

#         +-----------------+
#>--------|  Decompositions |------<
#         +-----------------+

  function svd(AA::tens{T};method::String="square",cutoff::Float64 = 0.,
            m::intType = 0,mag::Float64=0.,minm::intType=2,nozeros::Bool=false,
            recursive::Bool = false) where T <: Number
    rAA = reshape(AA.T,AA.size...)
    U,D,V,truncerr,sumD = svd(rAA,method=method,cutoff=cutoff,m=m,mag=mag,minm=minm,nozeros=nozeros,recursive=recursive)
    tensU = tens(T,U)
    tensD = tens(Float64,D)
    tensV = tens(T,V)
    return tensU,tensD,tensV,truncerr,sumD
  end

  """
      eigen(AA[,B,cutoff=,m=,mag=,minm=,nozeros=])
  eigen-solver routine with truncation; accepts Qtensors; arguments similar to `svd`; can perform generalized eigenvalue problem with `B` overlap matrix

  See also: [`svd`](@ref)
  """
  function eigen(AA::Array{T,2},B::Array{T,2}...;cutoff::Float64 = 0.,m::intType = 0,mag::Float64=0.,
                  minm::intType=1,nozeros::Bool=false) where {T <: Number}

    if eltype(AA) <: Complex
      M = LinearAlgebra.Hermitian((AA+AA')/2)
    else
      M = LinearAlgebra.Symmetric((AA+AA')/2)
    end
    Dsq,U = length(B) == 0 ? LinearAlgebra.eigen(M) : LinearAlgebra.eigen(M,B[1])
    if m == 0
      finalD = Array(LinearAlgebra.Diagonal(Dsq))::Array{Float64,2}
      Utrunc = U::Array{T,2}
      finaltrunc,sumD = Float64(0),Float64(0)
    else
      Dsq = abs.(Dsq)
      sizeD = size(Dsq,1)
      truncerr = 0.
      if mag == 0.
        sumD = real(sum(Dsq))::Float64
      else
        sumD = mag::Float64
      end
      modcutoff = cutoff*sumD
        p = 1
        while p < sizeD && ((truncerr + Dsq[p] < modcutoff) || (nozeros && abs(D[p]) < effZero))
          truncerr += Dsq[p]
          p += 1
        end
        thism = min(m < sizeD-p+1 ? sizeD-m+1 : p,sizeD-minm+1)
        if 0 < thism
          Utrunc = U[:,thism:sizeD]::Array{W,2} where W <: Number
          Dtrunc = Dsq[thism:sizeD]
        elseif 0 > thism
          Utrunc = zeros(eltype(U),size(U,1),thism)::Array{W,2} where W <: Number
          Dtrunc = zeros(eltype(Dsq),thism)
          Utrunc[:,thism:size(U,2)] = U
          Dtrunc[thism:size(Dsq,1)] = Dsq
        else
          Utrunc = U::Array{W,2} where W <: Number
          Dtrunc = Dsq
        end
        finaltrunc = (thism != p ? 1 - sum(Dtrunc)/sumD : truncerr)::Float64
        finalD = Array(LinearAlgebra.Diagonal(Dtrunc))::Array{Float64,2}
    end
    return finalD,Utrunc,finaltrunc,sumD
  end
  export eigen

  function eigen(AA::tens{T},B::tens{R}...;cutoff::Float64 = 0.,
                m::intType = 0,mag::Float64=0.,minm::intType=2,
                nozeros::Bool=false,recursive::Bool = false) where {T <: Number, R <: Number}
    X = reshape(AA.T,AA.size...)
    D,U,truncerr,sumD = eigen(X,B...,cutoff=cutoff,m=m,mag=mag,minm=minm,nozeros=nozeros)
    tensD = tens(Float64,D)
    tensU = tens(T,U)
    return tensD,tensU,truncerr,sumD
  end

  function getorder(AA::Union{Qtens{W,Q},AbstractArray,tens{W}},vecA::Array{Array{intType,1},1}) where {W <: Number, Q <: Qnum}
    order = Array{intType,1}(undef,sum(a->length(vecA[a]),1:length(vecA)))
    counter = 0
    for b = 1:length(vecA)
      for j = 1:length(vecA[b])
        counter += 1
        order[counter] = vecA[b][j]
      end
    end

    permuteq = !(issorted(order))
    if permuteq
      AB = permutedims(AA,order)
      Lvec = [i for i = 1:length(vecA[1])]
      Rvec = [i + length(vecA[1]) for i = 1:length(vecA[2])]
      rAA = reshape(AB,[Lvec,Rvec])
    else
      rAA = reshape(AA,vecA)
    end

    Lsizes = [size(AA,vecA[1][i]) for i = 1:length(vecA[1])]
    Rsizes = [size(AA,vecA[2][i]) for i = 1:length(vecA[2])]
    return rAA,Lsizes,Rsizes
  end

  """
      svd(AA,vecA[,cutoff=,m=,mag=,minm=,nozeros=])

  reshapes `AA` for `svd` and then unreshapes U and V matrices on return; `vecA` is of the form [[1,2],[3,4,5]] and must be length 2
  """
  function svd(AA::Union{Qtens{W,Q},AbstractArray,tens{W}},vecA::Array{Array{intType,1},1};cutoff::Float64 = 0.,
              m::intType = 0,mag::Float64=0.,minm::intType=1,nozeros::Bool=false,
              recursive::Bool = false,inplace::Bool=true,
              method::String="square") where {W <: Number,Q <: Qnum}
    AB,Lsizes,Rsizes = getorder(AA,vecA)
    U,D,V,truncerr,newmag = svd(AB,method = method,cutoff=cutoff,m=m,mag=mag,minm=minm,nozeros=nozeros,recursive=recursive)

    outU = unreshape!(U,Lsizes...,size(D,1))
    outV = unreshape!(V,size(D,2),Rsizes...)
    return outU,D,outV,truncerr,newmag
  end

  """
      eigen(AA,vecA[,B,cutoff=,m=,mag=,minm=,nozeros=])

  reshapes `AA` for `eigen` and then unreshapes U matrix on return; `vecA` is of the form [[1,2],[3,4,5]] and must be length 2
  """
  function eigen(AA::Union{Qtens{W,Q},AbstractArray,tens{W}},vecA::Array{Array{intType,1},1},
                  B::Array{W,2}...;cutoff::Float64 = 0.,m::intType = 0,mag::Float64=0.,
                  minm::intType=1,nozeros::Bool=false,inplace::Bool=true) where {W <: Number,Q <: Qnum}
    AB,Lsizes,Rsizes = getorder(AA,vecA)
    D,U,truncerr,newmag = eigen(AB,B...,cutoff=cutoff,m=m,mag=mag,minm=minm,nozeros=nozeros)
    outU = unreshape!(U,Lsizes...,size(D,1))
    return D,outU,truncerr,newmag
  end 

#       +---------------------------------------+
#       |                                       |
#>------+  Quantum Number conserving operation  +---------<
#       |                                       |
#       +---------------------------------------+

  const DQsize = Array{intType,1}[[1],[2]]
  """
      storage(QtensAA,totBlockSizeA)

  Defines storage for an output object of svd or eigen

  # Arguments:
  - `QtensAA::qarray` input Qtensor
  - `totBlockSizeA::Integer` total number of elements in `QtensAA`

  # Output:
  - storage vectors for X position, Y positions, and tensor values
  """
  function storage(QtensAA::qarray,totBlockSizeA::Integer)
    bigUx = Array{intType,1}(undef,totBlockSizeA)
    bigUy = Array{intType,1}(undef,totBlockSizeA)
    bigUT = Array{eltype(QtensAA.T),1}(undef,totBlockSizeA)
    Ucounter = 0
    return bigUx,bigUy,bigUT
  end

  """
      loadU!(curroffset,U,retAALrows,Ucounter,bigUT,bigUx,bigUy)
  Transfers elements from a sub-block `U` into storage vectors `bigUT`, `bigUx`, `bigUy`

  # Arguments:
  + `curroffset::Integer`: current offset on the column of `U`
  + `U::Array{T,2}`: sub-block
  + `retAALrows::Array{Int64,2}`: rows of the large matrix-equivalent for each of the sub-block's rows
  + `Ucounter::Integer`: offset integer for positions in storage vectors
  + `bigUT::Array{T,1}`: stores elements from the sub-blocks for the large matrix-equivalent
  + `bigUx::Array{Int64,1}`: stores elements from the X position the large matrix-equivalent
  + `bigUy::Array{Int64,1}`: stores elements from the Y position for the large matrix-equivalent
  """
  function loadU!(curroffset::Integer,U::Array{T,2},retAALrows::Array{intType,1},Ucounter::Integer,
                  bigUT::Array{T,1},bigUx::Array{intType,1},bigUy::Array{intType,1}) where {T <: Number}
    let Ucounter = Ucounter, retAALrows = retAALrows, bigUT = bigUT, bigUx = bigUx, bigUy = bigUy, U = U, curroffset = curroffset
      Threads.@threads    for j = 1:size(U,2)
        currcol = j + curroffset
        thisind = Ucounter + size(retAALrows,1)*(j-1)
        @simd for i = 1:size(retAALrows,1)
          thisrow = retAALrows[i]
          thisotherind = i + thisind
          bigUT[thisotherind] = U[i,j]
          bigUx[thisotherind] = thisrow
          bigUy[thisotherind] = currcol
        end
      end
    end
    nothing
  end

  """
      loadD!(D,bigD,newQnums,conQnumSumAAL,x,curroffset)
  Places values from sub-block into arrays for a D center

  # Arguments:
  + `D::Array{T,1}`: dense `D` matrix given as a vector
  + `bigD::Array{T,1}`: storage elements of full `D`
  + `newQnums::Array{Qnum,1}`: quantum numbers for created indices
  + `conQnumSumAAL::Array{Qnum,1}`: quantum number summary
  + `x::Integer`: integer of quantum number summary corresponding to the current sub-block
  + `curroffset::Integer`: current storage offset
  """
  function loadD!(D::Array{T,1},bigD::Array{T,1},newQnums::Array{Q,1},conQnumSumAAL::Array{Q,1},x::Integer,curroffset::Integer) where {T <: Number, Q <: Qnum}
    invQN = inv(conQnumSumAAL[x])
    let D = D, bigD = bigD, newQnums = newQnums, invQN = invQN, curroffset = curroffset
      Threads.@threads    for i = 1:size(D,1)
        bigD[i+curroffset] = D[i]
        newQnums[i+curroffset] = invQN
      end
    end
    nothing
  end

  """
  loadV!(curroffset,Vt,retAARcols,Vcounter,bigVT,bigVx,bigVy)
  Transfers elements from a sub-block `Vt` into storage vectors `bigVT`, `bigVx`, `bigVy`

  # Arguments:
  + `curroffset::Integer`: current offset on the column of `V`
  + `V::Array{T,2}`: sub-block
  + `retAARcols::Array{Int64,2}`: columns of the large matrix-equivalent for each of the sub-block's columns
  + `Vcounter::Integer`: offset integer for positions in storage vectors
  + `bigVT::Array{T,1}`: stores elements from the sub-blocks for the large matrix-equivalent
  + `bigVx::Array{Int64,1}`: stores elements from the X position the large matrix-equivalent
  + `bigVy::Array{Int64,1}`: stores elements from the Y position for the large matrix-equivalent
  """
  function loadV!(curroffset::Integer,Vt::Array{T,2},retAARcols::Array{intType,1},Vcounter::Integer,
                  bigVT::Array{T,1},bigVx::Array{intType,1},bigVy::Array{intType,1}) where {T <: Number}
    let retAARcols = retAARcols, Vcounter = Vcounter, Vt = Vt, bigVT = bigVT, bigVx = bigVx, bigVy = bigVy, curroffset = curroffset
      Threads.@threads    for j = 1:size(retAARcols,1)
        thiscol = retAARcols[j]
        thisind = Vcounter + size(Vt,1)*(j-1)
        @simd for i = 1:size(Vt,1)
          thisotherind = i + thisind
          bigVT[thisotherind] = Vt[i,j]
          bigVx[thisotherind] = i + curroffset
          bigVy[thisotherind] = thiscol
        end
      end 
    end
    nothing
  end

  """
      truncation(newQnums,zeroQN,m,bigD,curroffset,Dsize,mag,minm,cutoff[,method=,nozeros=])
  truncates `D` matrix

  # Arguments:
  + `newQnums::Array{Qnum,1}`: quantum numbers on the created index
  + `zeroQN::Qnum`: a quantum number of zero
  + `m::Integer`: maximum bond dimension size
  + `bigD::Array{T,1}`: storage for `D`
  + `curroffset::Integer`: number of elements of `D stored so far`
  + `Dsize::Integer`: size of row of `D`
  + `mag::Number`: magnitude of the original tensor
  + `minm::Integer`: minmum bond dimension
  + `cutoff::Float64`: maximum sum of values that can be truncated
  + `method`: accepts string of "square" to truncate over square of `D` values and "sqrt" to truncate over square root; otherwise truncates over only `D` values
  + `nozeros`: toggle to eliminate zeros in `D`
  """
    function truncation(newQnums::Array{Q,1},zeroQN::Q,m::Integer,bigD::Array{T,1},curroffset::Integer,Dsize::Integer,
                    mag::Number,minm::Integer,cutoff::Number;method::String="identity",nozeros::Bool=false) where {T <: Number, Q <: Qnum}
    let currofset = curroffset, Dsize = Dsize, newQnums = newQnums, zeroQN = zeroQN, bigD = bigD
      Threads.@threads for k = curroffset+1:Dsize
        newQnums[k] = zeroQN
        bigD[k] = 0.
      end
    end

    ranked = sortperm(bigD,rev=true)
    sizeD = Dsize
    truncerr = 0.
    if method == "square"
      sumD = mag != 1. ? sum(a->bigD[a]^2,1:size(bigD,1)) : mag
    elseif method == "sqrt"
      sumD = mag != 1. ? sum(a->sqrt(bigD[a]),1:size(bigD,1)) : mag
    else
      sumD = mag != 1. ? sum(a->bigD[a],1:size(bigD,1)) : mag
    end

    if m == 0 && !nozeros && cutoff == 0.
      thism = sizeD
      p = sizeD
    else
      p = sizeD
      modcutoff = sumD*cutoff
      if method == "square"
        while p > 0 && ((truncerr + bigD[ranked[p]]^2 < modcutoff) || (nozeros && (abs(bigD[ranked[p]]) < effZero)))
          truncerr += bigD[ranked[p]]^2
          p -= 1
        end
      elseif method == "sqrt"
        while p > 0 && ((truncerr + sqrt(bigD[ranked[p]]) < modcutoff) || (nozeros && (abs(bigD[ranked[p]]) < effZero)))
          truncerr += sqrt(bigD[ranked[p]])
          p -= 1
        end
      else
        while p > 0 && ((truncerr + bigD[ranked[p]] < modcutoff) || (nozeros && (abs(bigD[ranked[p]]) < effZero)))
          truncerr += bigD[ranked[p]]
          p -= 1
        end
      end
      if m == 0
        thism = max(min(sizeD,p),minm)
      else
        thism = max(min(m,p,sizeD),minm)
      end
    end
    truncInds = ranked[1:thism]

    sortTruncInds = sort(truncInds)
    truncD = bigD[sortTruncInds]
    truncsize = size(truncD,1)
    primetruncD = bigD[sortTruncInds]
    finalnewQnums = newQnums[sortTruncInds]
    finalnewQnumSum = unique(finalnewQnums)

    return truncerr, truncsize, p, sortTruncInds, primetruncD, finalnewQnums, finalnewQnumSum, truncD, sumD
  end

  """
      returnU(Linds,Lsize,truncsize,QtensAA,bigUT,bigUx,bigUy,sortTruncInds,nrowsAA,finalnewQnums,finalnewQnumSum,zeroQN)

  generates `U` tensor for a quantum number SVD decomposition

  # Arguments:
  + `Linds::Array{intType,1}`: indices field for Qtensor
  + `Lsize::intType`: number of dimensions of `Linds`
  + `truncsize::intType`: size of truncated index
  + `QtensAA::Qtens{T,Q}`: original Qtensor
  + `bigUT::Array{R,1}`: storage for `U` values
  + `bigUx::Array{intType,1}`: storage for X values
  + `bigUy::Array{intType,1}`: storage for Y values
  + `sortTruncInds::Array{intType,1}`: non-truncated values on created index
  + `nrowsAA::intType`: size of the row of the matrix-equivalent of QtensAA
  + `finalnewQnums::Array{Qnum,1}`: new quantum numbers on created index
  + `finalnewQnumSum::Array{Qnum,1}`: summary on created index
  + `zeroQN::Qnum`: zero quantum number
  """
  function returnU(Linds::Array{intType,1},Lsize::intType,truncsize::intType,QtensAA::Qtens{T,Q},bigUT::Array{R,1},bigUx::Array{intType,1},bigUy::Array{intType,1},
                    sortTruncInds::Array{intType,1},nrowsAA::intType,finalnewQnums::Array{Q,1},finalnewQnumSum::Array{Q,1},zeroQN::Q) where {T <: Number, R <: Number, Q <: Qnum}
    currinds = Linds
    vecsize = intType[i > size(currinds,1) ? truncsize : QtensAA.size[currinds[i]] for i = 1:size(currinds,1)+1]
    newUT,newUpos = makeUnitary(bigUT,bigUx,bigUy,sortTruncInds,nrowsAA,true)
    UQnumMat = Array{typeof(QtensAA.flux),1}[a > Lsize ? finalnewQnums : QtensAA.QnumMat[a] for a = 1:Lsize+1]
    UQnumSum = Array{typeof(QtensAA.flux),1}[a > Lsize ? finalnewQnumSum : QtensAA.QnumSum[a] for a = 1:Lsize+1]
    newQsize = Array{intType,1}[currinds,intType[size(currinds,1)+1]]
    Uq = Qtens{R,Q}(vecsize,newQsize,newUT,newUpos,UQnumMat,UQnumSum,zeroQN)
    return Uq
  end

  """
      returnD(truncsize,QtensAA,primetruncD,finalnewQnums,finalnewQnumSum,zeroQN)

  generates `D` tensor for a quantum number SVD decomposition

  # Arguments:
  + `truncsize::Int64`: size of truncated index
  + `QtensAA::qarray`: original Qtensor
  + `primetruncD::Array{T,1}`: values of `D`
  + `finalnewQnums::Array{Qnum,1}`: new quantum numbers on created index
  + `finalnewQnumSum::Array{Qnum,1}`: summary of quantum numbers on created index
  + `zeroQN::Qnum`: zero quantum number
  """
  function returnD(truncsize::intType,QtensAA::Qtens{T,Q},primetruncD::Array{R,1},finalnewQnums::Array{Q,1},finalnewQnumSum::Array{Q,1},zeroQN::Q) where {T <: Number, R <: Float64, Q <: Qnum}
    Dnewinds = intType[i + truncsize*(i-1) for i = 1:truncsize]
    DQnumMat = Array{typeof(QtensAA.flux),1}[inv.(finalnewQnums),finalnewQnums]
    DQnumSum = Array{typeof(QtensAA.flux),1}[inv.(finalnewQnumSum),finalnewQnumSum]
    Dq = Qtens{R,Q}(intType[truncsize,truncsize],DQsize,primetruncD,Dnewinds,DQnumMat,DQnumSum,zeroQN)
    return Dq
  end

  """
      returnV(Rinds,Rsize,truncsize,QtensAA,bigVT,bigVx,bigVy,sortTruncInds,finalnewQnums,finalnewQnumSum,zeroQN)

  generates `V` tensor for a quantum number SVD decomposition

  # Arguments:
  + `Rinds::Array{intType,1}`: indices field for Qtensor
  + `Rsize::intType`: number of dimensions of `Rinds`
  + `truncsize::intType`: size of truncated index
  + `QtensAA::Qtens{T,Q}`: original Qtensor
  + `bigVT::Array{R,1}`: storage for `V` values
  + `bigVx::Array{intType,1}`: storage for X values
  + `bigVy::Array{intType,1}`: storage for Y values
  + `sortTruncInds::Array{intType,1}`: non-truncated values on created index
  + `nrowsAA::intType`: size of the row of the matrix-equivalent of QtensAA
  + `finalnewQnums::Array{Qnum,1}`: new quantum numbers on created index
  + `finalnewQnumSum::Array{Qnum,1}`: summary on created index
  + `zeroQN::Qnum`: zero quantum number
  """
  function returnV(Rinds::Array{intType,1},Rsize::intType,truncsize::intType,QtensAA::Qtens{T,Q},bigVT::Array{R,1},
                    bigVx::Array{intType,1},bigVy::Array{intType,1},sortTruncInds::Array{intType,1},finalnewQnums::Array{Q,1},
                    finalnewQnumSum::Array{Q,1},zeroQN::Q) where {T <: Number, R <: Number, Q <: Qnum}
    currinds = Rinds
    vecsize = intType[i == 1 ? truncsize : QtensAA.size[currinds[i-1]] for i = 1:size(currinds,1)+1]
    newVT,newVpos = makeUnitary(bigVT,bigVy,bigVx,sortTruncInds,truncsize,false)
    VQnumMat = Array{typeof(QtensAA.flux),1}[a == 1 ? inv.(finalnewQnums) : QtensAA.QnumMat[Rinds[a-1]] for a = 1:Rsize+1]
    VQnumSum = Array{typeof(QtensAA.flux),1}[a == 1 ? inv.(finalnewQnumSum) : QtensAA.QnumSum[Rinds[a-1]] for a = 1:Rsize+1]
    newQsize = Array{intType,1}[intType[1],intType[i+1 for i = 1:size(currinds,1)]]
    Vq = Qtens{R,Q}(vecsize,newQsize,newVT,newVpos,VQnumMat,VQnumSum,QtensAA.flux)
    return Vq
  end

  """
      prepareSubmat(QtensAA,x,AALcountsizes,AARcountsizes,AALfinalxy,AARfinalxy,AAxypos,AALfinalQN,AARfinalQN,fullAA)

  Prepares sub-block with relevant elements from the input Qtensor
  + `QtensAA::qarray`: input Qtensor
  + `x::intType`: entry of AALcountsizes (synonymous with quantum number block we rae examining)
  + `AALcountsizes::Array{intType,1}`: left size of quantum number block
  + `AARcountsizes::Array{intType,1}`: right size of quantum number block
  + `AALfinalxy::Array{intType,1}`: row of the large tensor corresponding to sub-block row
  + `AARfinalxy::Array{intType,1}`: column of the large tensor corresponding to sub-block column
  + `AAxypos::Array{intType,2}`: XY positions of tensor elements
  + `AALfinalQN::Array{intType,1}`quantum number converted to number corresponding to row
  + `AARfinalQN::Array{intType,1}`: quantum number converted to number corresponding to column
  + `fullAA::Bool`: toggle whether the Qtenosr is full or not (initialize with zeros or not)
  """
  function prepareSubmat(QtensAA::qarray,x::intType,AALcountsizes::Array{intType,1},AARcountsizes::Array{intType,1},AALfinalxy::Array{intType,1},
                          AARfinalxy::Array{intType,1},AAxypos::Array{intType,2},AALfinalQN::Array{intType,1},AARfinalQN::Array{intType,1},fullAA::Bool)
    if fullAA
      submatAA = Array{eltype(QtensAA.T),2}(undef,AALcountsizes[x],AARcountsizes[x])
    else
      submatAA = zeros(eltype(QtensAA.T),AALcountsizes[x],AARcountsizes[x])
    end
    retAALrows = QNsearch(submatAA,QtensAA,AALfinalxy,AARfinalxy,AAxypos,1,AALfinalQN,x,AALcountsizes)
    retAARcols = makeRetRowCol(AARfinalQN,x,AARcountsizes)
    return submatAA,retAALrows,retAARcols
  end

  function svd(QtensAA::Qtens{W,Q};cutoff::Float64 = 0.,m::intType = 0,mag::Float64=0.,
                minm::intType=1,nozeros::Bool=false,recursive::Bool=false,
                method::String="identity") where {W <: Number, Q <: Qnum}
    Linds,Rinds,Lsize,Rsize,AAxypos,LR,nrowsAA,ncolsAA,currpos,zeroQN,
    AALfinalxy,AALcountsizes,AALfinalQN,conQnumSumAAL,AARfinalxy,AARcountsizes,AARfinalQN,
    conQnumSumAAR, subblocksizes, mindims, Ablocksizes, Bblocksizes, totRetSize, totBlockSizeA, totBlockSizeB,fullAA = initmatter(QtensAA)
 
    Dsize = max(min(nrowsAA,ncolsAA),minm)

    bigUx,bigUy,bigUT = storage(QtensAA,totBlockSizeA)
    bigVx,bigVy,bigVT = storage(QtensAA,totBlockSizeB)

    bigD = Array{Float64,1}(undef,Dsize)
    newQnums = Array{typeof(QtensAA.flux),1}(undef,Dsize)
    let AALcountsizes = AALcountsizes, recursive = recursive, AARcountsizes = AARcountsizes, QtensAA = QtensAA, AALfinalxy = AALfinalxy,AARfinalxy = AARfinalxy,AAxypos = AAxypos,AALfinalQN = AALfinalQN,AARfinalQN = AARfinalQN,fullAA = fullAA, Ablocksizes = Ablocksizes, Bblocksizes = Bblocksizes, mindims = mindims, bigUT = bigUT,bigUx = bigUx,bigUy = bigUy,bigVT = bigVT,bigVx = bigVx,bigVy = bigVy,bigD = bigD, newQnums = newQnums, conQnumSumAAL = conQnumSumAAL
      Threads.@threads  for x  = 1:size(AALcountsizes,1)
        if AALcountsizes[x] != 0 && AARcountsizes[x] != 0

          submatAA,retAALrows,retAARcols = prepareSubmat(QtensAA,x,AALcountsizes,AARcountsizes,AALfinalxy,AARfinalxy,AAxypos,AALfinalQN,AARfinalQN,fullAA)

          U,D,Vt = recursive ? recursive_SVD(AA) : safesvd(submatAA)

          if x == 1
            Usize,Vsize,curroffset = 0,0,0
          else
            Usize,Vsize = sum(a->Ablocksizes[a],1:x-1),sum(a->Bblocksizes[a],1:x-1)
            curroffset = sum(a->mindims[a],1:x-1)
          end

          loadU!(curroffset,U,retAALrows,Usize,bigUT,bigUx,bigUy)
          loadV!(curroffset,Vt,retAARcols,Vsize,bigVT,bigVx,bigVy)
          loadD!(D,bigD,newQnums,conQnumSumAAL,x,curroffset)
        end
      end
    end
    mcurroffset = sum(mindims)

    out = truncation(newQnums,zeroQN,m,bigD,mcurroffset,Dsize,mag,minm,cutoff;method=method,nozeros=nozeros)
    truncerr, truncsize, p, sortTruncInds, primetruncD, finalnewQnums, finalnewQnumSum, truncD, sumD = out

    Uq = returnU(Linds,Lsize,truncsize,QtensAA,bigUT,bigUx,bigUy,sortTruncInds,nrowsAA,finalnewQnums,finalnewQnumSum,zeroQN)
    Vq = returnV(Rinds,Rsize,truncsize,QtensAA,bigVT,bigVx,bigVy,sortTruncInds,finalnewQnums,finalnewQnumSum,zeroQN)
    Dq = returnD(truncsize,QtensAA,primetruncD,finalnewQnums,finalnewQnumSum,zeroQN)

    return Uq,Dq,Vq,m == 0 || m == p ? truncerr : 1-sum(a->truncD[a]^2,1:size(truncD,1))/sumD,sumD
  end

  function eigen(QtensAA::Qtens{W,Q};cutoff::Float64 = 0.,m::intType = 0,mag::Float64=0.,minm::intType=1,
                  nozeros::Bool=false) where {W <: Number, Q <: Qnum}
    Linds,Rinds,Lsize,Rsize,AAxypos,LR,nrowsAA,ncolsAA,currpos,zeroQN,
    AALfinalxy,AALcountsizes,AALfinalQN,conQnumSumAAL,AARfinalxy,AARcountsizes,AARfinalQN,
    conQnumSumAAR, subblocksizes, mindims, Ablocksizes, Bblocksizes, totRetSize, totBlockSizeA, totBlockSizeB,fullAA = initmatter(QtensAA)

    Dsize = max(min(nrowsAA,ncolsAA),minm)

    bigUx,bigUy,bigUT = storage(QtensAA,totBlockSizeA)

    bigD = Array{Float64,1}(undef,Dsize)
    newQnums = Array{typeof(QtensAA.flux),1}(undef,Dsize)
    let AALcountsizes = AALcountsizes, AARcountsizes = AARcountsizes, QtensAA = QtensAA, AALfinalxy = AALfinalxy,AARfinalxy = AARfinalxy,AAxypos = AAxypos,AALfinalQN = AALfinalQN,AARfinalQN = AARfinalQN,fullAA = fullAA, Ablocksizes = Ablocksizes, Bblocksizes = Bblocksizes, mindims = mindims, bigUT = bigUT,bigUx = bigUx,bigUy = bigUy,bigD = bigD, newQnums = newQnums, conQnumSumAAL = conQnumSumAAL
      Threads.@threads  for x  = 1:size(AALcountsizes,1)
        if AALcountsizes[x] != 0 && AARcountsizes[x] != 0
          submatAA,retAALrows,retAARcols = prepareSubmat(QtensAA,x,AALcountsizes,AARcountsizes,AALfinalxy,AARfinalxy,AAxypos,AALfinalQN,AARfinalQN,fullAA)

          if eltype(submatAA) <: Complex
            M = LinearAlgebra.Hermitian((submatAA+submatAA')/2)
          else
            M = LinearAlgebra.Symmetric((submatAA+submatAA')/2)
          end
          D,U = LinearAlgebra.eigen(M)

          if x == 1
            Usize,curroffset = 0,0
          else
            Usize = sum(a->Ablocksizes[a],1:x-1)
            curroffset = sum(a->mindims[a],1:x-1)
          end
          
          loadU!(curroffset,U,retAALrows,Usize,bigUT,bigUx,bigUy)
          loadD!(D,bigD,newQnums,conQnumSumAAL,x,curroffset)
        end
      end
    end
    mcurroffset = sum(mindims)

    out = truncation(newQnums,zeroQN,m,bigD,mcurroffset,Dsize,mag,minm,cutoff;method="square",nozeros=nozeros)
    truncerr, truncsize, p, sortTruncInds, primetruncD, finalnewQnums, finalnewQnumSum, truncD, sumD = out

    Uq = returnU(Linds,Lsize,truncsize,QtensAA,bigUT,bigUx,bigUy,sortTruncInds,nrowsAA,finalnewQnums,finalnewQnumSum,zeroQN)
    Dq = returnD(truncsize,QtensAA,primetruncD,finalnewQnums,finalnewQnumSum,zeroQN) #::Qtens{Float64,Q}

    return Dq,Uq,m == 0 || m == p ? truncerr : 1-sum(a->truncD[a],1:size(truncD,1))/sumD,sumD
  end

  """
      Q,R = qr(A)

  Returns the QR decomposition of `A`; no truncation defined here

  See also: [`LQ`](@ref)
  """
  function qr(A::Array{T,2};decompfct::Function=LinearAlgebra.qr) where T <: Number
    U,V = decompfct(A)
    if size(U,2) > size(V,1)
      U = U[:,1:size(V,1)]
    elseif size(U,2) < size(V,1)
      V = V[1:size(U,2),:]
    end
    return U,V
  end

  function qr(A::denstens;decompfct::Function=LinearAlgebra.qr)
    U,V = qr(to_Array(A),decompfct=decompfct)
    return tens(U),tens(V)
  end

  function qr(QtensAA::Qtens{W,Q};decompfct::Function=LinearAlgebra.qr) where {W <: Number, Q <: Qnum}
    Linds,Rinds,Lsize,Rsize,AAxypos,LR,nrowsAA,ncolsAA,currpos,zeroQN,
    AALfinalxy,AALcountsizes,AALfinalQN,conQnumSumAAL,AARfinalxy,AARcountsizes,AARfinalQN,
    conQnumSumAAR, subblocksizes, mindims, Ablocksizes, Bblocksizes, totRetSize, totBlockSizeA, totBlockSizeB,fullAA = initmatter(QtensAA)

    Dsize = max(min(nrowsAA,ncolsAA),minm)

    bigUx,bigUy,bigUT = storage(QtensAA,totBlockSizeA)
    bigRx,bigRy,bigRT = storage(QtensAA,totBlockSizeB)

    newQnums = Array{typeof(QtensAA.flux),1}(undef,Dsize)
    let AALcountsizes = AALcountsizes, AARcountsizes = AARcountsizes, QtensAA = QtensAA, AALfinalxy = AALfinalxy,AARfinalxy = AARfinalxy,AAxypos = AAxypos,AALfinalQN = AALfinalQN,AARfinalQN = AARfinalQN,fullAA = fullAA, Ablocksizes = Ablocksizes, Bblocksizes = Bblocksizes, mindims = mindims, bigUT = bigUT,bigUx = bigUx,bigUy = bigUy,bigD = bigD, newQnums = newQnums, conQnumSumAAL = conQnumSumAAL
      Threads.@threads  for x  = 1:size(AALcountsizes,1)
        if AALcountsizes[x] != 0 && AARcountsizes[x] != 0
          submatAA,retAALrows,retAARcols = prepareSubmat(QtensAA,x,AALcountsizes,AARcountsizes,AALfinalxy,AARfinalxy,AAxypos,AALfinalQN,AARfinalQN,fullAA)

          qQ,R = decompfct(submatAA)
          if size(qQ,2) > size(R,1)
            qQ = qQ[:,1:size(R,1)]
          elseif size(qQ,2) < size(R,1)
            R = R[1:size(qQ,2),:]
          end

          if x == 1
            Usize,curroffset = 0,0
          else
            Usize = sum(a->Ablocksizes[a],1:x-1)
            curroffset = sum(a->mindims[a],1:x-1)
          end
          
          loadU!(curroffset,qQ,retAALrows,Usize,bigUT,bigUx,bigUy)
          loadV!(curroffset,R,retAARcols,Vsize,bigVT,bigVx,bigVy)
          invQN = inv(conQnumSumAAL[x])
          let newQnums = newQnums, invQN = invQN, curroffset = curroffset
            Threads.@threads    for i = 1:size(R,1)
              newQnums[i+curroffset] = invQN
            end
          end
        end
      end
    end
    mcurroffset = sum(mindims)

    finalnewQnums = newQnums
    finalnewQnumSum = unique(finalnewQnums)
    truncsize = min(Lsize,Rsize)

    Uq = returnU(Linds,Lsize,truncsize,QtensAA,bigUT,bigUx,bigUy,sortTruncInds,nrowsAA,finalnewQnums,finalnewQnumSum,zeroQN)
    Vq = returnV(Rinds,Rsize,truncsize,QtensAA,bigVT,bigVx,bigVy,sortTruncInds,finalnewQnums,finalnewQnumSum,zeroQN)

    return Uq,Vq
  end

  function qr(AA::Union{Qtens{W,Q},AbstractArray,tens{W}},vecA::Array{Array{intType,1},1};decompfct::Function=LinearAlgebra.qr) where {W <: Number,Q <: Qnum}
    AB,Lsizes,Rsizes = getorder(AA,vecA)
    U,V = qr(AB,decompfct=decompfct)

    outU = unreshape!(U,Lsizes...,size(V,1))
    outV = unreshape!(V,size(V,1),Rsizes...)
    return outU,outV
  end
  export qr

  """
      L,Q = lq(A)

  Returns the LQ decomposition of `A`; no truncation defined here

  See also: [`QR`](@ref)
  """
  function lq(A::TensType)
    return qr(A,decompfct=LinearAlgebra.lq)
  end

  function lq(AA::Union{Qtens{W,Q},AbstractArray,tens{W}},vecA::Array{Array{intType,1},1}) where {W <: Number,Q <: Qnum}
    return qr(AA,vecA,decompfct=LinearAlgebra.lq)
  end
  export lq

end
