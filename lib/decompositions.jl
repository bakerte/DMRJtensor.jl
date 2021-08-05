#########################################################################
#
#  Density Matrix Renormalization Group (and other methods) in julia (DMRjulia)
#                              v1.0
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
module decompositions
using ..tensor
using ..contractions
import LinearAlgebra

import Printf

#       +---------------------------------------+
#       |                                       |
#>------+            SVD of a tensor            +---------<
#       |                                       |
#       +---------------------------------------+
  """
      libsvd(C,D[,Z,alpha=,beta=])
  Chooses the best svd function for tensor decomposition
  + Outputs U,D,Vt
  """
  function libsvd(X::Array{T,2}) where T <: Number
    F = LinearAlgebra.svd(X)
    return F.U,F.S,Array(F.Vt)
  end
  export libsvd

  function libqr(R::AbstractArray;decomposer::Function=LinearAlgebra.qr)
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
    return U,V,0.,1.
  end

  function liblq(R::AbstractArray;decomposer::Function=LinearAlgebra.lq)
    return libqr(R,decomposer=decomposer)
  end

# iterative SVD for increased precision
  function recursive_SVD(AA::AbstractArray,tolerance::Float64=1E-4)
    U,D,V = safesvd(AA)
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

  # Explanation (v1.1.1-v1.5.3 and beyond?):
  
  For some highly degenerate matrices, julia's svd function will throw an error (interface issue with LAPACK).
  This try/catch function will test the svd output and use a modified algorithm with lower preicision (fast),
  a high precision algorithm (slow), and then throw an error. 
  Messages and the offending matrix are printed when this occurs so it is known when this occurs.
  The problem often self-corrects as the computation goes along in a standard DMRG computation for affected models.
  Typically, this is just a warning and not a cause for concern.
  """
  function (safesvd(AA::Array{T,2})) where T <: Number
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
          return Array{Float64,2}(undef,0,0),Array{Float64,1}(undef,0),Array{Float64,2}(undef,0,0)
        end
      end
    end
  end

  function findnewm(D::Array{W,1},m::Number,minm::Integer,mag::Float64,cutoff::Real,effZero::Real,nozeros::Bool,power::Number,keepdeg::Bool) where {W <: Number}
    sizeD = size(D,1)
    p = sizeD
    if mag == 0.
      sumD = 0.
      @simd for a = 1:size(D,1)
        sumD += (real(D[a])::Float64)^2
      end
    else
      sumD = mag::Float64
    end
    modcutoff = sumD*cutoff
    truncerr = 0.
    while p > 0 && ((truncerr + D[p]^power < modcutoff) || (nozeros && abs(D[p]) < effZero))
      truncerr += D[p]^power
      p -= 1
    end
    if keepdeg
      while p < length(D) && isapprox(D[p],D[p+1])
        p += 1
      end
    end
    thism = m == 0 ? max(min(p,sizeD),minm) : max(min(m,p,sizeD),minm)
    return thism,p,sizeD,truncerr,sumD
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
  function svd(AA::Array{T,2};power::Number=2,cutoff::Float64 = 0.,m::intType = 0,mag::Float64=0.,minm::intType=2,
               nozeros::Bool=false,recursive::Bool = false,effZero::Float64 = 1E-16,keepdeg::Bool=false) where T <: Number
      U,D,Vt = recursive ? recursive_SVD(AA) : safesvd(AA)
      thism,p,sizeD,truncerr,sumD = findnewm(D,m,minm,mag,cutoff,effZero,nozeros,power,keepdeg)
      if sizeD > thism
        Utrunc = U[:,1:thism]
        Dtrunc = D[1:thism]
        Vtrunc = Vt[1:thism,:]
      elseif sizeD < thism
        Utrunc = zeros(eltype(U),size(U,1),thism)
        Dtrunc = zeros(eltype(D),thism)
        Vtrunc = zeros(eltype(Vt),thism,size(Vt,2))
        Utrunc[:,1:size(U,2)] = U
        Dtrunc[1:size(D,1)] = D
        Vtrunc[1:size(Vt,1),:] = Vt
      else
        Utrunc = U
        Dtrunc = D
        Vtrunc = Vt
      end
      Darray = Array(LinearAlgebra.Diagonal(Dtrunc))::Array{Float64,2}
      finaltrunc = (thism != p ? 1 - sum(a->Dtrunc[a]^2,1:thism)/sumD : truncerr)::Float64
      return Utrunc,Darray,Vtrunc,finaltrunc,sumD
  end
  export svd

#         +-----------------+
#>--------|  Decompositions |------<
#         +-----------------+

  function svd(AA::tens{W};power::Number=2,cutoff::Float64 = 0.,
            m::intType = 0,mag::Float64=0.,minm::intType=2,nozeros::Bool=false,
            recursive::Bool = false,effZero::Number = 1E-16,keepdeg::Bool=false) where W <: Number
    rAA = reshape(AA.T,size(AA)...)
    U,D,V,truncerr,sumD = svd(rAA,power=power,cutoff=cutoff,m=m,mag=mag,minm=minm,nozeros=nozeros,recursive=recursive,effZero=effZero,keepdeg=keepdeg)
    tensU = tens(W,U)
    tensD = tens(Float64,D)
    tensV = tens(W,V)
    return tensU,tensD,tensV,truncerr,sumD
  end

  """
      eigen(AA[,B,cutoff=,m=,mag=,minm=,nozeros=])
  eigen-solver routine with truncation; accepts Qtensors; arguments similar to `svd`; can perform generalized eigenvalue problem with `B` overlap matrix

  See also: [`svd`](@ref)
  """
  function eigen(AA::Array{T,2},B::Array{T,2}...;cutoff::Float64 = 0.,m::intType = 0,mag::Float64=0.,
                  minm::intType=1,nozeros::Bool=false,power::Number=1,effZero::Float64 = 1E-16,keepdeg::Bool=false) where {T <: Number}

    if eltype(AA) <: Complex
      M = LinearAlgebra.Hermitian((AA+AA')/2)
    else
      M = LinearAlgebra.Symmetric((AA+AA')/2)
    end
    Dsq,U = length(B) == 0 ? LinearAlgebra.eigen(M) : LinearAlgebra.eigen(M,B[1])

    thism,p,sizeD,truncerr,sumD = findnewm(Dsq,m,minm,mag,cutoff,effZero,nozeros,power,keepdeg)
      if sizeD > thism
    #=
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
      =#
#    if 0 < thism
      Utrunc = U[:,thism:sizeD]::Array{W,2} where W <: Number
      Dtrunc = Dsq[thism:sizeD]
    elseif sizeD < thism
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
#    end
    return finalD,Utrunc,finaltrunc,sumD
  end
  export eigen

  function eigen(AA::tens{T},B::tens{R}...;cutoff::Float64 = 0.,
                m::intType = 0,mag::Float64=0.,minm::intType=2,
                nozeros::Bool=false,recursive::Bool = false,keepdeg::Bool=false) where {T <: Number, R <: Number}
    X = reshape(AA.T,size(AA)...)
    D,U,truncerr,sumD = eigen(X,B...,cutoff=cutoff,m=m,mag=mag,minm=minm,nozeros=nozeros,keepdeg)#,recursive=recursive)
    tensD = tens(Float64,D)
    tensU = tens(T,U)
    return tensD,tensU,truncerr,mag
  end

  import .LinearAlgebra.eigvals
  function eigvals(A::tens{W}) where W <: Number
    thisarray = makeArray(A)
    vals = LinearAlgebra.eigvals(thisarray)
    return tens(vals)
  end

#         +-------------------------+
#>--------|  Quantum number part   |--------<
#         +-------------------------+



  function getorder(AA::G,vecA::Array{Array{intType,1},1}) where G <: TensType
    order = Array{intType,1}(undef,sum(a->length(vecA[a]),1:length(vecA)))
    counter = 0
    for b = 1:length(vecA)
      for j = 1:length(vecA[b])
        counter += 1
        order[counter] = vecA[b][j]
      end
    end

    if !(issorted(order))
      AB = permutedims(AA,order)
      Lvec = [i for i = 1:length(vecA[1])]
      Rvec = [i + length(vecA[1]) for i = 1:length(vecA[2])]
      rAA = reshape(AB,[Lvec,Rvec])::G
    else
      rAA = reshape(AA,vecA)::G
    end

    Lsizes = [size(AA,vecA[1][i]) for i = 1:length(vecA[1])]
    Rsizes = [size(AA,vecA[2][i]) for i = 1:length(vecA[2])]
    return rAA,Lsizes,Rsizes
  end

  """
      svd(AA,vecA[,cutoff=,m=,mag=,minm=,nozeros=])

  reshapes `AA` for `svd` and then unreshapes U and V matrices on return; `vecA` is of the form [[1,2],[3,4,5]] and must be length 2
  """
  function svd(AA::G,vecA::Array{Array{intType,1},1};cutoff::Float64 = 0.,
              m::intType = 0,mag::Float64=0.,minm::intType=1,nozeros::Bool=false,
              recursive::Bool = false,inplace::Bool=true,power::Number=2,keepdeg::Bool=false) where G <: TensType
    AB,Lsizes,Rsizes = getorder(AA,vecA)
    U,D,V,truncerr,newmag = svd(AB,power = power,cutoff=cutoff,m=m,mag=mag,minm=minm,nozeros=nozeros,recursive=recursive,keepdeg=keepdeg)

    outU = unreshape!(U,Lsizes...,size(D,1))
    outV = unreshape!(V,size(D,2),Rsizes...)
    return outU,D,outV,truncerr,newmag
  end

  """
      eigen(AA,vecA[,B,cutoff=,m=,mag=,minm=,nozeros=])

  reshapes `AA` for `eigen` and then unreshapes U matrix on return; `vecA` is of the form [[1,2],[3,4,5]] and must be length 2
  """
  function eigen(AA::TensType,vecA::Array{Array{intType,1},1},
                  B::TensType...;cutoff::Float64 = 0.,m::intType = 0,mag::Float64=0.,
                  minm::intType=1,nozeros::Bool=false,inplace::Bool=true,keepdeg::Bool=false)
    AB,Lsizes,Rsizes = getorder(AA,vecA)
    D,U,truncerr,newmag = eigen(AB,B...,cutoff=cutoff,m=m,mag=mag,minm=minm,nozeros=nozeros,keepdeg)
    outU = unreshape!(U,Lsizes...,size(D,1))
    return D,outU,truncerr,newmag
  end   
  
  """
      qr(AA,vecA[,cutoff=,m=,mag=,minm=,nozeros=])

  reshapes `AA` for `svd` and then unreshapes U and V matrices on return; `vecA` is of the form [[1,2],[3,4,5]] and must be length 2
  """
  function qr(AA::TensType,vecA::Array{Array{intType,1},1})
    AB,Lsizes,Rsizes = getorder(AA,vecA)
    Qmat,Rmat,truncerr,newmag = qr(AB)

    innerdim = size(Qmat,2)

    outU = unreshape!(Qmat,Lsizes...,innerdim)
    outV = unreshape!(Rmat,innerdim,Rsizes...)
    return outU,outV,truncerr,newmag
  end

  function qr(AA::AbstractArray,decomposer::Function=LinearAlgebra.qr)
    return libqr(AA,decomposer=decomposer)
  end

  function qr(AA::denstens,decomposer::Function=LinearAlgebra.qr)
    rAA = makeArray(AA)
    Q,R,truncerr,mag = libqr(rAA,decomposer=decomposer)
    return tens(Q),tens(R),truncerr,mag
  end

  """
      lq(AA,vecA[,cutoff=,m=,mag=,minm=,nozeros=])

  reshapes `AA` for `svd` and then unreshapes U and V matrices on return; `vecA` is of the form [[1,2],[3,4,5]] and must be length 2
  """
  function lq(AA::TensType,vecA::Array{Array{intType,1},1})
    AB,Lsizes,Rsizes = getorder(AA,vecA)
    Lmat,Qmat,truncerr,newmag = lq(AB)

    innerdim = size(Qmat,1)

    outU = unreshape!(Lmat,Lsizes...,innerdim)
    outV = unreshape!(Qmat,innerdim,Rsizes...)
    return outU,outV,truncerr,newmag
  end

  function lq(AA::AbstractArray,decomposer::Function=LinearAlgebra.lq)
    return libqr(AA,decomposer=decomposer)
  end

  function lq(AA::denstens,decomposer::Function=LinearAlgebra.lq)
    rAA = makeArray(AA)
    L,Q,truncerr,mag = libqr(rAA,decomposer=decomposer)
    return tens(L),tens(Q),truncerr,mag
  end

  """
      polar(A[,group=,*])

  *-options from `svd`

  Performs a polar decomposition on tensor `A` with grouping `group` (default: [[1,2],[3]] for an MPS); if `left` (default), this returns U*D*U',U'*V else functoin returns U*V',V*D*V' from an `svd`

  See also: [`svd`](@ref)
  """
  function polar(AA::TensType,group::Array{Array{intType,1},1};
                  right::Bool=true,cutoff::Float64 = 0.,m::intType = 0,mag::Float64 = 0.,
                  minm::intType=1,nozeros::Bool=false,recursive::Bool=false,outermaxm::intType=0,keepdeg::Bool=false)
    AB,Lsizes,Rsizes = getorder(AA,group)
    U,D,V = svd(AB,cutoff=cutoff,m=m,mag=mag,minm=minm,nozeros=nozeros,keepdeg=keepdeg)
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
    return leftTensor,rightTensor,D,0.
  end
  export polar

#       +---------------------------------------+
#       |                                       |
#>------+  Quantum Number conserving operation  +---------<
#       |                                       |
#       +---------------------------------------+
using ..QN
using ..Qtensor
using ..Qtask

  function makeQblocksum(finalUinds::Array{Array{Array{intType,2},1},1},finalUQnumMat::Array{Array{intType,1},1},finalUQnumSum::Array{Array{Q,1},1}) where Q <: Qnum
    nQN = length(finalUinds)
    newQblocksum = [[Q(),Q()] for q = 1:nQN]
    for q = 1:nQN
      for i = 1:size(finalUinds[q][1],1)
        x = finalUinds[q][1][i,1] + 1
        Qnumber = finalUQnumMat[i][x]
        qnum = finalUQnumSum[i][Qnumber]
        add!(newQblocksum[q][1],qnum)
      end
      for i = 1:size(finalUinds[q][2],1)
        index = i + size(finalUinds[q][1],1)
        y = finalUinds[q][2][i,1] + 1
        Qnumber = finalUQnumMat[index][y]
        qnum = finalUQnumSum[index][Qnumber]
        add!(newQblocksum[q][2],qnum)
      end
    end
    return newQblocksum
  end


  function makeU(nQN::Integer,keepq::Array{Bool,1},outU::Array{tens{W},1},QtensA::Qtens{W,Q},
                 finalinds::Array{Array{intType,2},1},
                 newqindexL::Array{Array{intType,1},1},newqindexLsum::Array{Array{Q,1},1},
                 leftflux::Bool,Linds::Array{intType,1},thism::Integer) where {W <: Number, Q <: Qnum}
    finalnQN = sum(keepq)
    finalUinds = Array{Array{Array{intType,2},1},1}(undef,finalnQN)
    counter = 0
    for q = 1:nQN
      if keepq[q]
        counter += 1
        left = QtensA.ind[q][1]
        right = finalinds[q]
        finalUinds[counter] = [left,right]
      end
    end
    finalUQnumMat = vcat(QtensA.QnumMat[Linds],newqindexL)
    Uflux = leftflux ? QtensA.flux : Q()
#    newUsize = (size(QtensA,1),thism)
    newUQsize = [[i for i = 1:length(Linds)],[length(Linds) + 1]]
    finalUQnumSum = vcat(QtensA.QnumSum[Linds],newqindexLsum)

    newQblocksum = makeQblocksum(finalUinds,finalUQnumMat,finalUQnumSum)
    return Qtens{W,Q}(newUQsize,outU,finalUinds,newUQsize,newQblocksum,finalUQnumMat,finalUQnumSum,Uflux)
  end

  function makeV(nQN::Integer,keepq::Array{Bool,1},outV::Array{tens{W},1},QtensA::Qtens{W,Q},
                 finalinds::Array{Array{intType,2},1},
                 newqindexR::Array{Array{intType,1},1},newqindexRsum::Array{Array{Q,1},1},
                 leftflux::Bool,Rinds::Array{intType,1},thism::Integer) where {W <: Number, Q <: Qnum}
    finalnQN = sum(keepq)
    finalVinds = Array{Array{Array{intType,2},1},1}(undef,finalnQN)
    counter = 0
    for q = 1:nQN
      if keepq[q]
        counter += 1
        left = finalinds[q]
        right = QtensA.ind[q][2]
        finalVinds[counter] = [left,right]
      end
    end
    finalVQnumMat = vcat(newqindexR,QtensA.QnumMat[Rinds])
    Vflux = !leftflux ? copy(QtensA.flux) : Q()
#    newVsize = (thism,size(QtensA,2))
    newVQsize = [[1],[i+1 for i = 1:length(Rinds)]]

    finalVQnumSum = vcat(newqindexRsum,QtensA.QnumSum[Rinds])

    newQblocksum = makeQblocksum(finalVinds,finalVQnumMat,finalVQnumSum)
    return Qtens{W,Q}(newVQsize,outV,finalVinds,newVQsize,newQblocksum,finalVQnumMat,finalVQnumSum,Vflux)
  end

  function makeD(nQN::Integer,keepq::Array{Bool,1},outD::Array{tens{W},1},QtensA::Qtens{W,Q},
                 finalinds::Array{Array{intType,2},1},
                 newqindexL::Array{Array{intType,1},1},newqindexR::Array{Array{intType,1},1},
                 newqindexRsum::Array{Array{Q,1},1},newqindexLsum::Array{Array{Q,1},1},thism::Integer) where {W <: Number, Q <: Qnum}
    finalnQN = sum(keepq)
    finalDinds = Array{Array{Array{intType,2},1},1}(undef,finalnQN)
    counter = 0
    for q = 1:nQN
      if keepq[q]
        counter += 1
        left = finalinds[q]
        right = finalinds[q]
        finalDinds[counter] = [left,right]
      end
    end
    finalDQnumMat = vcat(newqindexR,newqindexL)

    Dflux = Q()
#    newDsize = (thism,thism)
    newDQsize = vcat([[1]],[[2]])
    finalDQnumSum = vcat(newqindexRsum,newqindexLsum)

    newQblocksum = makeQblocksum(finalDinds,finalDQnumMat,finalDQnumSum)
    return Qtens{W,Q}(newDQsize,outD,finalDinds,newDQsize,newQblocksum,finalDQnumMat,finalDQnumSum,Dflux)
  end

  function truncate(sizeinnerm::Integer,newD::Array{Array{W,1},1},power::Number,
                    nozeros::Bool,cutoff::Float64,m::Integer,nQN::Integer,mag::Float64,keepdeg::Bool) where W <: Number

    rhodiag = Array{Float64,1}(undef,sizeinnerm)
    counter = 0
    #=Threads.@threads=# for q = 1:nQN
      offset = q == 1 ? 0 : sum(w->length(newD[w]),1:q-1)
      for x = 1:length(newD[q])
        rhodiag[x + offset] = Real(newD[q][x])^power
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
    y = m == 0 ? length(order) : min(m,length(order))

    truncadd = rhodiag[order[y]]
    while y > 0 && ((truncerr + truncadd < modcutoff) || (nozeros && isapprox(truncadd,0.)))
      truncerr += truncadd
      truncadd = rhodiag[order[y]]
      y -= 1
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


  function newqindexes(thism::Integer,QNsummary::Array{Q,1},qranges::Array{R,1}) where {Q <: Qnum, R <: UnitRange}

    newqindexL = Array{Q,1}(undef,thism)
    counter = 1
    k = 1
    while k <= thism
      if k in qranges[counter]
        newqindexL[k] = copy(QNsummary[counter])
        k += 1
      else
        counter += 1
      end
    end
    newqindexL = [newqindexL]
    newqindexL,newqindexLsum = convertQnumMat(newqindexL)
 
    
    newqindexRsum = [inv.(newqindexLsum[1])]
    newqindexR = newqindexL

    return newqindexL,newqindexLsum,newqindexR,newqindexRsum
  end

  function svd(QtensA::Qtens{W,Q};cutoff::Float64 = 0.,m::intType = 0,
                minm::intType=1,nozeros::Bool=true,recursive::Bool=false,
                power::Number=2,leftflux::Bool=false,mag::Float64=0.,
                effZero::Real=1E-16,keepdeg::Bool=false) where {W <: Number, Q <: Qnum}

    Rsize = QtensA.size
    Linds = Rsize[1]
    Rinds = Rsize[2]

    newcurrblock = [Linds,Rinds]

    A = changeblock(QtensA,Linds,Rinds)
    nQN = length(A.T)

    LRinds = 1
    QNsummary = [inv(A.Qblocksum[q][LRinds]) for q = 1:nQN]

    
  
    newU = Array{Array{W,2},1}(undef,nQN)
    newD = Array{Array{W,1},1}(undef,nQN)
    newV = Array{Array{W,2},1}(undef,nQN)

    Threads.@threads for q = 1:nQN
      arr = makeArray(A.T[q])
      newU[q],newD[q],newV[q] = safesvd(arr)
    end

    sizeinnerm = sum(q->size(newD[q],1),1:nQN)
      
      finalinds,thism,qstarts,qranges,keepers,truncerr,sumD = truncate(sizeinnerm,newD,power,nozeros,cutoff,m,nQN,mag,keepdeg)

      tempU,tempD,tempV = Array{tens{W},1}(undef,nQN),Array{tens{W},1}(undef,nQN),Array{tens{W},1}(undef,nQN)
    
      newqindexL = Array{Q,1}(undef,thism)
      keepq = Array{Bool,1}(undef,nQN)
      #=Threads.@threads=# for q = 1:nQN
        keepq[q] = length(keepers[q]) > 0
        if keepq[q]
          tempU[q] = tens{W}(newU[q][:,keepers[q]])
          squareD = Array{W,2}(undef,length(keepers[q]),length(keepers[q]))
          for y = 1:size(squareD,1)
            for x = 1:size(squareD,1)
              @simd for i = 1:x-1
                @inbounds squareD[x,y] = 0
              end
              @inbounds squareD[x,x] = newD[q][x]
              @simd for j = x+1:size(squareD,1)
                @inbounds squareD[x,y] = 0
              end
            end
          end

          tempD[q] = tens{W}(squareD)
          tempV[q] = tens{W}(newV[q][keepers[q],:])
          for i = 1:length(finalinds[q])
            @inbounds offset = q == 1 ? 0 : sum(w->length(finalinds[w]),1:q-1)
            @inbounds newqindexL[i + offset] = QNsummary[q]
          end
        end
      end

      newqindexL = [newqindexL]
      newqindexL,newqindexLsum = convertQnumMat(newqindexL)
  
      
      newqindexRsum = [inv.(newqindexLsum[1])]
      newqindexR = newqindexL


      if sum(keepq) < nQN
        outU = tempU[keepq]
        outD = tempD[keepq]
        outV = tempV[keepq]
      else
        outU = tempU
        outD = tempD
        outV = tempV
      end

    U = makeU(nQN,keepq,outU,A,finalinds,newqindexL,newqindexLsum,leftflux,Linds,thism)
    D = makeD(nQN,keepq,outD,A,finalinds,newqindexL,newqindexR,newqindexRsum,newqindexLsum,thism)
    V = makeV(nQN,keepq,outV,A,finalinds,newqindexR,newqindexRsum,leftflux,Rinds,thism)

    temp = contract(U,ndims(U),D,1)
    lasty = contract(temp,ndims(temp),V,1)

#    checkflux(U)
#    checkflux(D)
#    checkflux(V)

    return U,D,V,truncerr,sumD
  end

  function symeigen(submatAA::AbstractArray)
    if eltype(submatAA) <: Complex
      M = LinearAlgebra.Hermitian((submatAA+submatAA')/2)
    else
      M = LinearAlgebra.Symmetric((submatAA+submatAA')/2)
    end
    return LinearAlgebra.eigen(M)
  end 






















  function eigen(QtensA::Qtens{W,Q};cutoff::Float64 = 0.,m::intType = 0,
                minm::intType=1,nozeros::Bool=false,recursive::Bool=false,
                power::Number=1,leftflux::Bool=false,mag::Float64=0.,decomposer::Function=symeigen,keepdeg::Bool=false) where {W <: Number, Q <: Qnum}

    Rsize = QtensA.size #recoverShape(QtensA)
    Linds = Rsize[1]
    Rinds = Rsize[2]

    newcurrblock = [Linds,Rinds]

    A = changeblock(QtensA,Linds,Rinds)

    nQN = length(A.T)

    LRinds = 1
    QNsummary = [inv(A.Qblocksum[q][LRinds]) for q = 1:nQN]
  
    newU = Array{Array{W,2},1}(undef,nQN)
    newD = Array{Array{W,1},1}(undef,nQN)
  

#    let A = A, newU = newU, newD = newD
      Threads.@threads for q = 1:nQN
        arr = makeArray(A.T[q])
        newD[q],newU[q] = decomposer(arr)
      end
#    end
 
    sizeinnerm = sum(q->size(newD[q],1),1:nQN)
    if cutoff > 0. || nozeros
      
      finalinds,thism,qstarts,qranges,keepers,truncerr,sumD = truncate(sizeinnerm,newD,power,nozeros,cutoff,m,nQN,mag,keepdeg)


      tempU,tempD = Array{tens{W},1}(undef,nQN),Array{tens{W},1}(undef,nQN)
    
      newqindexL = Array{Q,1}(undef,thism)
      keepq = Array{Bool,1}(undef,nQN)
      for q = 1:nQN
        keepq[q] = length(keepers[q]) > 0
        if keepq[q]
          tempU[q] = tens{W}(newU[q][:,keepers[q]])
          squareD = Array(LinearAlgebra.Diagonal(newD[q][keepers[q]]))
          tempD[q] = tens{W}(squareD)
          for i = 1:length(finalinds[q])
            offset = q == 1 ? 0 : sum(w->length(finalinds[w]),1:q-1)
            newqindexL[i + offset] = copy(QNsummary[q])
          end
        end
      end

      newqindexL = [newqindexL]
      newqindexL,newqindexLsum = convertQnumMat(newqindexL)
  
      newqindexRsum = [inv.(newqindexLsum[1])]
      newqindexR = newqindexL


      if sum(keepq) < nQN
        outU = tempU[keepq]
        outD = tempD[keepq]
      else
        outU = tempU
        outD = tempD
      end

    else

      thism = sizeinnerm
      sumD = mag

      outU,outD = Array{tens{W},1}(undef,nQN),Array{tens{W},1}(undef,nQN)
      for q = 1:nQN
        outU[q] = tens{W}(newU[q])
        sizeD = size(newD[q],1)
        fullD = Array{W,1}(undef,sizeD^2)
        prevDind = 1
        fullD[prevDind] = newD[q][1]
        for i = 2:sizeD
          diagind = i + sizeD * (i-1)
          @simd for j = prevDind+1:diagind-1
            fullD[j] = 0
          end
          prevDind = diagind
          fullD[diagind] = newD[q][i]
        end
        outD[q] = tens{W}((sizeD,sizeD),fullD)
      end

      qstarts = vcat([0],[sum(w->size(newD[w],1),1:q-1) for q = 2:nQN],[thism])
      qranges = [UnitRange(qstarts[q]+1,qstarts[q+1]) for q = 1:nQN]

      truncerr = 0.

      finalinds = [reshape([i-1 for i = qranges[q]],1,length(qranges[q])) for q = 1:length(QNsummary)]
      newqindexL,newqindexLsum,newqindexR,newqindexRsum = newqindexes(thism,QNsummary,qranges)
      keepq = [true for q = 1:nQN]
    end

#    outTens = Array{Qtens{W,Q},1}(undef,2)
#    Threads.@threads for w = 1:2
#      if w == 1
        U = makeU(nQN,keepq,outU,A,finalinds,newqindexL,newqindexLsum,leftflux,Linds,thism)
#      else
        D = makeD(nQN,keepq,outD,A,finalinds,newqindexL,newqindexR,newqindexRsum,newqindexLsum,thism)
#      end
#    end

    return D,U,truncerr,sumD
  end

  function qr(QtensA::Qtens{W,Q};leftflux::Bool=false,decomposer::Function=libqr,mag::Number=1.) where {W <: Number, Q <: Qnum}

    Rsize = QtensA.size #recoverShape(QtensA)
    Linds = Rsize[1]
    Rinds = Rsize[2]

    newcurrblock = [Linds,Rinds]

    A = changeblock(QtensA,Linds,Rinds)

    nQN = length(A.T)

    LRinds = 1
    QNsummary = [inv(A.Qblocksum[q][LRinds]) for q = 1:nQN]
  
    newU = Array{Array{W,2},1}(undef,nQN)
    newV = Array{Array{W,2},1}(undef,nQN)
  

#    let A = A, newU = newU, newV = newV
      Threads.@threads for q = 1:nQN
        arr = makeArray(A.T[q])
        newU[q],newV[q] = decomposer(arr)
      end
#    end
 
    sizeinnerm = sum(q->size(newU[q],2),1:nQN)

    thism = sizeinnerm
    sumD = mag

    outU,outV = Array{tens{W},1}(undef,nQN),Array{tens{W},1}(undef,nQN)
    for q = 1:nQN
      outU[q] = tens{W}(newU[q])
      outV[q] = tens{W}(newV[q])
    end

    qstarts = vcat([0],[sum(w->size(newU[w],2),1:q-1) for q = 2:nQN],[thism])
    qranges = [UnitRange(qstarts[q]+1,qstarts[q+1]) for q = 1:nQN]

    truncerr = 0.

    finalinds = [reshape([i-1 for i = qranges[q]],1,length(qranges[q])) for q = 1:length(QNsummary)]
    newqindexL,newqindexLsum,newqindexR,newqindexRsum = newqindexes(thism,QNsummary,qranges)
    keepq = [true for q = 1:nQN]

#    outTens = Array{Qtens{W,Q},1}(undef,2)
#    Threads.@threads for w = 1:2
#      if w == 1
        U = makeU(nQN,keepq,outU,A,finalinds,newqindexL,newqindexLsum,leftflux,Linds,thism)
#      else
        V = makeV(nQN,keepq,outV,A,finalinds,newqindexR,newqindexRsum,leftflux,Rinds,thism)
#      end
#    end

    return U,V,truncerr,mag
  end
  export qr

  function lq(QtensA::Qtens{W,Q};leftflux::Bool=false) where {W <: Number, Q <: Qnum}
    return qr(QtensA,leftflux=leftflux,decomposer=liblq)
  end
  export lq

end
