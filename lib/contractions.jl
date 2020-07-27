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
    Module: contractions

Contract two tensors together

See also: [`decompositions`](@ref)
"""
module contractions
using ..tensor
using ..QN
using ..Qtensor
using ..Qtask
import LinearAlgebra

#       +------------------------+
#>------|    Matrix multiply     |---------<
#       +------------------------+

  """
      blas_pick(C,D[,Z,alpha=,beta=])
  Chooses the best matrix multiply function for tensor contraction (dense tensor or sub-block)
  + Outputs alpha * C * D (+ beta * Z)
  """
  function blas_pick(C::Array{W,2},D::Array{X,2},tempZ::Array{P,2}...;alpha::Number=1.,beta::Number=1.) where {W <: Number, X <: Number, P <: Number}
    if length(tempZ) == 0
      if (eltype(C) == eltype(D)) && !(eltype(C) <: Integer)
        temp_alpha = typeof(alpha) <: eltype(C) ?  alpha : convert(eltype(C),alpha)
        retMat = LinearAlgebra.BLAS.gemm('N','N',temp_alpha,C,D)
      else
        retMat = alpha * C * D
      end
    else
      if (eltype(C) == eltype(D) == eltype(Z[1])) && !(eltype(C) <: Integer)
        temp_alpha = convert(eltype(C),alpha)
        temp_beta = convert(eltype(C),beta)
        LinearAlgebra.BLAS.gemm!('N','N',temp_alpha,C,D,temp_beta,tempZ[1])
        retMat = tempZ[1]
      else
        retMat = alpha * C * D + beta * tempZ[1]
      end
    end
    return retMat
  end

#=  """
      scalarcontractor(A,B,conjA,conjB[,alpha=])

  Contract to a scalar; accepts conjugation on `A` (`conjA`) and `B` (`conjB`); accepts Qtensors
  """=#
  function scalarcontractor(A::AbstractArray,B::AbstractArray,conjA::Bool,conjB::Bool;alpha::Number=1.)
    thissize = prod(size(A))
    cA = conjA ? conj(A) : A
    cB = conjB ? conj(B) : B
    rA = reshape(cA,thissize)
    rB = reshape(cB,thissize)
    return alpha * sum(i->rA[i]*rB[i],1:thissize)
  end

  function scalarcontractor(A::Number,B::P,conjA::Bool,conjB::Bool;alpha::Number=1.) where P <: Union{Number,TensType}
    mA = conjA ? conj(A) : A
    mB = conjB ? conj(B) : B
    return mA * mB * alpha
  end

  function scalarcontractor(A::TensType,B::Number,conjA::Bool,conjB::Bool;alpha::Number=1.)
    return scalarcontractor(B,A,conjB,conjA;alpha=1.)
  end

  """
      prepareT(A,Lvec,Rev,conjvar)

  Forms the matrix-equivlanet of a tensor defined by `Lvec` indices forming the rows and `Rvec` indices forming the columns; toggle conjugate (`conjvar`)
  """
  function prepareT(A::AbstractArray,Lvec::Array{intType,1},Rvec::Array{intType,1},conjvar::Bool)
    vec = vcat(Lvec,Rvec)
    Lsize = length(Lvec) > 0 ? prod(a->size(A,a),Lvec) : 1
    Rsize = length(Rvec) > 0 ? prod(a->size(A,a),Rvec) : 1

    currTens = conjvar ? conj(A) : A
    temp = !issorted(vec) ? permutedims(currTens,vec) : currTens
    return reshape(temp,Lsize,Rsize)
  end

  function arbTens(AB::Array{W,2},L::intType...) where W <: Number
    return reshape(AB,L...)
  end

  """
      mastercontractor(A,iA,B,iB,conjA,conjB[,Z,alpha=,beta=])

  Primary contraction function.  Contracts `A` along indices `iA` to `B` along `iB`; toggle conjugation (`conjA` and `conjB`)
  """
  function mastercontractor(A::AbstractArray,iA::Array{intType,1},B::AbstractArray,iB::Array{intType,1},
                            conjA::Bool,conjB::Bool,Z::AbstractArray...;alpha::Number=1.,beta::Number=1.)
    notvecA = setdiff([i for i = 1:ndims(A)],iA)
    C = prepareT(A,notvecA,iA,conjA)
    notvecB = setdiff([i for i = 1:ndims(B)],iB)
    D = prepareT(B,iB,notvecB,conjB)

    if length(Z) > 0
      tempZ = reshape(Z[1],size(C,1),size(D,2))
      CD = blas_pick(C,D,tempZ...,alpha=alpha,beta=beta)
    else
      CD = blas_pick(C,D,alpha=alpha,beta=beta)
    end

    notvecAsize = [size(A,w) for w in notvecA]
    notvecBsize = [size(B,w) for w in notvecB]
    outdims = ndims(A) + ndims(B) - length(iA) - length(iB)
    AAsizes = Array{intType,1}(undef,outdims)
    counter = 0
    @simd for a = 1:length(notvecAsize)
      counter += 1
      AAsizes[counter] = notvecAsize[a]
    end
    @simd for a = 1:length(notvecBsize)
      counter += 1
      AAsizes[counter] = notvecBsize[a]
    end
    return arbTens(CD,AAsizes...)
  end

#       +------------------------+
#>------| Quantum number version |---------<
#       +------------------------+

  function scalarcontractor(A::qarray,B::qarray,conjA::Bool,conjB::Bool;alpha::Number=1.)
    quickout = 0.
    if A.T == B.T
      thisA = conjA ? conj(A.T) : A.T
      thisB = conjB ? conj(B.T) : B.T
      @simd for i = 1:size(thisA,1)
        quickout += alpha * thisA[i] * thisB[i]
      end
    else
      sortTensors!(A,B)
      thisA = conjA && !(eltype(A.T) <: Real) ? conj(A.T) : A.T
      thisB = conjB && !(eltype(B.T) <: Real) ? conj(B.T) : B.T
      i,j = 1,1
      while i <= size(A.ind,1) && j <= size(B.ind,1)
        if A.ind[i] < B.ind[j]
          i += 1
        elseif A.ind[i] > B.ind[j]
          j += 1
        else #if A.ind[i] == B.ind[j]
          quickout += alpha * thisA[i] * thisB[j]
          i += 1
          j += 1
        end
      end
    end
    return quickout
  end

  function mastercontractor(QtensA::Qtens{T,Q},vecA::Array{intType,1},QtensB::Qtens{R,Q},vecB::Array{intType,1},
                            conjA::Bool,conjB::Bool,Z::Qtens{S,Q}...;alpha::Number=1.,
                            beta::Number=1.) where {T <: Number, R <: Number, S <: Number, Q <: Qnum}    
      if QtensA === QtensB && (conjA || conjB)
        QtensB = copy(QtensA)
      end


      numthreads = Threads.nthreads()
      firstout = resQtensInfo(QtensA,vecA,QtensB,vecB,numthreads)
      conA,notconA,conB,notconB,sizeAA,nrowsA,ncolsA,nrowsB,ncolsB,currpos = firstout

      Axypos = makeXYpos(QtensA,notconA,conA,currpos)
      Bxypos = makeXYpos(QtensB,conB,notconB,currpos)

      zeroQN = Q()

      sumALsizes = length(notconA) > 0 ? prod(a->length(QtensA.QnumSum[a]),notconA) : 1
      sumBRsizes = length(notconB) > 0 ? prod(a->length(QtensB.QnumSum[a]),notconB) : 1
      twoLR = sumALsizes < sumBRsizes
      
      if twoLR
        out = sumBlockInfo(QtensA,conA,notconA,conjA,sumALsizes,QtensB,currpos,ncolsA,conjB)
        notconQnumSumA,conQnumSumA,conQnumSumB,notconQnumSumB,innerfinalxy,innercountsizes,qvec = out
      else
        out = sumBlockInfo(QtensB,conB,notconB,conjB,sumBRsizes,QtensA,currpos,nrowsB,conjA)
        notconQnumSumB,conQnumSumB,conQnumSumA,notconQnumSumA,innerfinalxy,innercountsizes,qvec = out
      end

      if size(notconA,1) == 0
        Afinalxy = intType[i for i = 1:size(QtensA.ind, 1)]
        Acountsizes = intType[1]
        AfinalQN = ones(intType, size(QtensA.ind, 1))
        notconQnumSumA = Q[zeroQN]
      else
        Afinalxy,Acountsizes,AfinalQN = QBlock(QtensA,currpos,nrowsA,notconQnumSumA,notconA,QtensA.flux,qvec)
      end
      if size(notconB,1) == 0
        Bfinalxy = intType[i for i = 1:size(QtensB.ind, 1)]
        Bcountsizes = intType[1]
        BfinalQN =  ones(intType, size(QtensB.ind, 1))
        notconQnumSumB = Q[zeroQN]
      else
        Bfinalxy,Bcountsizes,BfinalQN = QBlock(QtensB,currpos,ncolsB,notconQnumSumB,notconB,QtensB.flux,qvec)
      end

      numElA = 0
      numElB = 0
      fullA = true
      fullB = true
      subblocksizes = zeros(intType,length(Acountsizes))
      for x = 1:length(Acountsizes)
        if innercountsizes[x] > 0
          subblocksizes[x] = Acountsizes[x]*Bcountsizes[x]
          numElA += Acountsizes[x]*innercountsizes[x]
          numElB += innercountsizes[x]*Bcountsizes[x]
        else
          fullA = false
          fullB = false
        end
      end
      totRetSize = sum(subblocksizes)

      retType = typeof(T(1)*R(1)) #size(QtensA.T,1) > 0 && size(QtensB.T,1) > 0 ? typeof(QtensA.T[1]*QtensB.T[1]) : Float64
      newT = Array{retType,1}(undef,totRetSize)
      newinds = Array{intType,1}(undef,totRetSize)

      fullA = numElA == length(QtensA.ind) && fullA
      fullB = numElB == length(QtensB.ind) && fullB

      typeA,typeB = eltype(QtensA.T),eltype(QtensB.T)

      if length(Z) != 0
        altmaxindZ = size(Z[1].T,1)
        altZbin = intType[w for w = 1:altmaxindZ]
        Zxypos = makeXYpos(Z[1],[i for i = 1:size(notconA,1)],[i for i = 1:size(notconB,1)],currpos)
        ZLinds = [i for i = 1:size(notconA,1)]
        ZRinds = [i+size(notconA,1) for i = 1:size(notconB,1)]
        conQnumSumZL = notconQnumSumA
        conQnumSumZR = notconQnumSumB
        ZLfinalxy,ZLcountsizes,ZLfinalQN = QBlock(Z[1],currpos,nrowsA,conQnumSumZL,ZLinds,zeroQN,qvec)
        ZRfinalxy,ZRcountsizes,ZRfinalQN = QBlock(Z[1],currpos,ncolsB,conQnumSumZR,ZRinds,Z[1].flux,qvec)
      end




      newflux = conjA ? inv(QtensA.flux) : QtensA.flux
      newflux += conjB ? inv(QtensB.flux) : QtensB.flux
      newQtens = makenewQtens(retType[],intType[],QtensA,QtensB,conA,notconA,conB,notconB,sizeAA,newflux,conjA,conjB)


      let fullA = fullA, typeA = typeA, fullB = fullB, typeB = typeB, numthreads = numthreads, Acountsizes = Acountsizes, Bcountsizes = Bcountsizes, 
          innercountsizes = innercountsizes, QtensA = QtensA, QtensB = QtensB, Axypos = Axypos, Bxypos = Bxypos, AfinalQN = AfinalQN, BfinalQN = BfinalQN, 
          nrowsA = nrowsA, newT = newT, newinds = newinds, alpha = alpha, beta = beta
        #moderate memory increase when using parallelization like this...replace the "#=Threads.@threads=#" with "#=#=Threads.@threads=#=#" if this is an issue
        Threads.@threads for x = 1:size(Acountsizes,1)
          if Acountsizes[x] != 0 && innercountsizes[x] != 0 && Bcountsizes[x] != 0

            submatA,retArows = loadSubmat(Acountsizes,Acountsizes,innercountsizes,numthreads,QtensA,Afinalxy,innerfinalxy,Axypos,AfinalQN,x,1,fullA,typeA)
            submatB,retBcols = loadSubmat(Bcountsizes,innercountsizes,Bcountsizes,numthreads,QtensB,innerfinalxy,Bfinalxy,Bxypos,BfinalQN,x,2,fullB,typeB)

            if conjA
              conj!(submatA)
            end
            if conjB
              conj!(submatB)
            end

            if length(Z) == 0
              newmat = blas_pick(submatA,submatB,alpha=alpha)
            else
              submatZ = zeros(eltype(QtensZ.T),Acountsizes[x],Bcountsizes[x])
              retZrows = QNsearch(submatZ,Z[1].T,ZLfinalxy,ZRfinalxy,Zxypos,1,ZLfinalQN,x,ZLcountsizes)
              newmat = blas_pick(submatA,submatB,submatZ,alpha=alpha,beta=beta)
            end

            subcounter = x == 1 ? 0 : sum(y->subblocksizes[y],1:x-1)
            retsubBlock!(nrowsA,retArows,retBcols,newT,newinds,newmat,subcounter)
          end
        end
      end

      newQtens.T = newT
      newQtens.ind = newinds


      return newQtens
  end

#       +----------------------------+
#>------| Contraction function calls |---------<
#       +----------------------------+

  const TensNum = Union{Number,TensType}

  """
      contract(A[,B,alpha=])
  Contracts to (alpha * A * B and returns a scalar output

  See also: [`ccontract`](@ref) [`contractc`](@ref) [`ccontractc`](@ref)
  """
  function contract(A::Q,B::R;alpha::Number=1.) where {Q <: TensNum,R <: TensNum}
    mA,mB = checkType(A,B)
    return scalarcontractor(mA,mB,false,false,alpha=alpha)
  end

  """
      ccontract(A[,B,alpha=])
  Similar to contract but 'A' is conjugated

  See also: ['ccontract'](@ref) ['contractc'](@ref) ['ccontractc'](@ref)
  """
  function ccontract(A::Q,B::R;alpha::Number=1.) where {Q <: TensNum,R <: TensNum}
    mA,mB = checkType(A,B)
    return scalarcontractor(mA,mB,true,false,alpha=alpha)
  end

  """
      contractc(A[,B,alpha=])
  Similar to contract but 'B' is conjugated

  See also: ['ccontract'](@ref) ['contractc'](@ref) ['ccontractc'](@ref)
  """
  function contractc(A::Q,B::R;alpha::Number=1.) where {Q <: TensNum,R <: TensNum}
    mA,mB = checkType(A,B)
    return scalarcontractor(mA,mB,false,true,alpha=alpha)
  end

  """
      ccontractc(A[,B,alpha=])
  Similar to contract but 'A' and 'B' are conjugated

  See also: ['ccontract'](@ref) ['contractc'](@ref) ['ccontractc'](@ref)
  """
  function ccontractc(A::Q,B::R;alpha::Number=1.) where {Q <: TensNum,R <: TensNum}
    mA,mB = checkType(A,B)
    return scalarcontractor(mA,mB,true,true,alpha=alpha)
  end



  function contract(A::Q;alpha::Number=1.) where {Q <: TensNum}
    return contract(A,A,alpha=alpha)
  end

  function ccontract(A::Q;alpha::Number=1.) where {Q <: TensNum}
    return ccontract(A,A,alpha=alpha)
  end

  function contractc(A::Q;alpha::Number=1.) where {Q <: TensNum}
    return contractc(A,A,alpha=alpha)
  end

  function ccontractc(A::Q;alpha::Number=1.) where {Q <: TensNum}
    return ccontractc(A,A,alpha=alpha)
  end

  """
      contract(A,iA,B,iB[,Z,alpha=,beta=])
  Contracts to (alpha * A * B + beta * Z) on input indices `iA` and `iB`; accepts different formats, ex: 1,[1,2],[1 2]); accepts any rank and also Qtensors

  # Example:

  ```julia
  julia> A = [1 0;0 -1];B = [-1 0;0 1];Z=[1 0;0 0];
  julia> contract(A,2,B,1,Z,alpha=2.,beta=5)
  2×2 Array{Float64,2}:
  3.0   0.0
  0.0  -2.0
  ```

  See also: [`ccontract`](@ref) [`contractc`](@ref) [`ccontractc`](@ref)
  """
  function contract(A::X,iA::O,B::Y,iB::P,Z::W...;alpha::Number=1.,beta::Number=1.) where {X <: TensType, Y <: TensType, W <: TensType, O <: intvecType, P <: intvecType}
    mA,mB = checkType(A,B)
    return mastercontractor(mA,convIn(iA),mB,convIn(iB),false,false,Z...,alpha=alpha,beta=beta)
  end

  """
      ccontract(A,iA,B,iB[,Z,alpha=,beta=])
  Similar to contract but 'A' is conjugated

  See also: ['ccontract'](@ref) ['contractc'](@ref) ['ccontractc'](@ref)
  """
  function ccontract(A::X,iA::O,B::Y,iB::P,Z::W...;alpha::Number=1.,beta::Number=1.) where {X <: TensType, Y <: TensType, W <: TensType, O <: intvecType, P <: intvecType}
    mA,mB = checkType(A,B)
    return mastercontractor(mA,convIn(iA),mB,convIn(iB),true,false,Z...,alpha=alpha,beta=beta)
  end

  """
      contractc(A,iA,B,iB[,Z,alpha=,beta=])
  Similar to contract but 'B' is conjugated

  See also: ['ccontract'](@ref) ['contractc'](@ref) ['ccontractc'](@ref)
  """
  function contractc(A::X,iA::O,B::Y,iB::P,Z::W...;alpha::Number=1.,beta::Number=1.) where {X <: TensType, Y <: TensType, W <: TensType, O <: intvecType, P <: intvecType}
    mA,mB = checkType(A,B)
    return mastercontractor(mA,convIn(iA),mB,convIn(iB),false,true,Z...,alpha=alpha,beta=beta)
  end

  """
      ccontractc(A,iA,B,iB[,Z,alpha=,beta=])
  Similar to contract but 'A' and 'B' are conjugated

  See also: ['ccontract'](@ref) ['contractc'](@ref) ['ccontractc'](@ref)
  """
  function ccontractc(A::X,iA::O,B::Y,iB::P,Z::W...;alpha::Number=1.,beta::Number=1.) where {X <: TensType, Y <: TensType, W <: TensType, O <: intvecType, P <: intvecType}
    mA,mB = checkType(A,B)
    return mastercontractor(mA,convIn(iA),mB,convIn(iB),true,true,Z...,alpha=alpha,beta=beta)
  end

  """
      contract(order,A,iA,B,iB[,Z,alpha=,beta=])
  Permutes result of [`contract`](@ref) to `order`
  """
  function contract(order::L,A::X,iA::O,B::Y,iB::P,Z::W...;alpha::Number=1.,beta::Number=1.) where {X <: TensType, Y <: TensType, W <: TensType, O <: intvecType, P <: intvecType, L <: intvecType}
    newT = contract(A,iA,B,iB,Z...,alpha=alpha,beta=beta)
    return permutedims!(newT,convIn(order))
  end

  """
      ccontract(order,A,iA,B,iB[,Z,alpha=,beta=])
  Permutes result of [`ccontract`](@ref) to `order`
  """
  function ccontract(order::L,A::X,iA::O,B::Y,iB::P,Z::W...;alpha::Number=1.,beta::Number=1.) where {X <: TensType, Y <: TensType, W <: TensType, O <: intvecType, P <: intvecType, L <: intvecType}
    mA,mB = checkType(A,B)
    newT = ccontract(mA,iA,mB,iB,Z...,alpha=alpha,beta=beta)
    return permutedims!(newT,convIn(order))
  end

  """
      contractc(order,A,iA,B,iB[,Z,alpha=,beta=])
  Permutes result of [`contractc`](@ref) to `order`
  """
  function contractc(order::L,A::X,iA::O,B::Y,iB::P,Z::W...;alpha::Number=1.,beta::Number=1.) where {X <: TensType, Y <: TensType, W <: TensType, O <: intvecType, P <: intvecType, L <: intvecType}
    mA,mB = checkType(A,B)
    newT = contractc(mA,iA,mB,iB,Z...,alpha=alpha,beta=beta)
    return permutedims!(newT,convIn(order))
  end

  """
      ccontractc(order,A,iA,B,iB[,Z,alpha=,beta=])
  Permutes result of [`ccontractc`](@ref) to `order`
  """
  function ccontractc(order::L,A::X,iA::O,B::Y,iB::P,Z::W...;alpha::Number=1.,beta::Number=1.) where {X <: TensType, Y <: TensType, W <: TensType, O <: intvecType, P <: intvecType, L <: intvecType}
    mA,mB = checkType(A,B)
    newT = ccontractc(mA,iA,mB,iB,Z...,alpha=alpha,beta=beta)
    return permutedims!(newT,convIn(order))
  end

  export contract,ccontract,contractc,ccontractc

  """
      trace(A,iA)

  Computes trace of `A` over indices `iA` by first making an identity tensor among those indices; `iA` is a vector of numbers (ex: [1,3]) or a vector of 2-element vectors ([[1,2],[3,4],[5,6]])

  # Example:

  ```julia
  julia> A = [1 0;0 -1];
  julia> trace(A,[1,2])
  0-dimensional Array{Float64,0}:
  0.0

  julia> A = ones(10,20,10,20);
  julia> trace(A,[[1,3],[2,4]])
  0-dimensional Array{Float64,0}:
  200.0
  ```
  """
  function trace(A::TensType,iA::W) where W <: Union{intvecType,Array{Array{intType,1},1}}
    if typeof(iA) <: intvecType
      Id = makeId(A,iA)
      conA = iA
    else
      Id = makeId(A,iA)
      conL = [iA[w][1] for w = 1:length(iA)]
      conR = [iA[w][2] for w = 1:length(iA)]
      conA = vcat(conL,conR)
    end
    return contract(A,conA,Id,[i for i = 1:ndims(Id)])
  end
  export trace



































#         +-----------------+
#>--------|  Contractions   |------<
#         +-----------------+

  import ..contractions.scalarcontractor
  """
    scalarcontractor(A,B,conjA,conjB[,alpha=])

  Contract to a scalar; accepts conjugation on `A` (`conjA`) and `B` (`conjB`); accepts Qtensors
  """
  function scalarcontractor(A::denstens,B::denstens,conjA::Bool,conjB::Bool;alpha::Number=1.)
    if conjA && conjB
      P = conj(sum(i->A.T[i]*B.T[i],1:length(A.T)))
    elseif conjA
      P = sum(i->conj(A.T[i])*B.T[i],1:length(A.T))
    elseif conjB
      P = sum(i->A.T[i]*conj(B.T[i]),1:length(A.T))
    else
      P = sum(i->A.T[i]*B.T[i],1:length(A.T))
    end
    return alpha * P
  end

  import ..contractions.prepareT
  """
      prepareT(A,Lvec,Rev,conjvar)

  Forms the matrix-equivlanet of a tensor defined by `Lvec` indices forming the rows and `Rvec` indices forming the columns; toggle conjugate (`conjvar`)
  """
  function prepareT(A::denstens,Lvec::Array{intType,1},Rvec::Array{intType,1},conjvar::Bool)
    vec = vcat(Lvec,Rvec)
    if issorted(vec)
      if conjvar
        X = conj(A)
      else
        X = A
      end
    else
      X = permutedims(A,vec)
      if conjvar
        conj!(X)
      end
    end
    Lsize = length(Lvec) > 0 ? prod(a->size(A,a),Lvec) : 1
    Rsize = length(Rvec) > 0 ? prod(a->size(A,a),Rvec) : 1
    return reshape(X.T,Lsize,Rsize)
  end

  import ..contractions.blas_pick
  """
      mastercontractor(A,iA,B,iB,conjA,conjB[,Z,alpha=,beta=])

  Primary contraction function.  Contracts `A` along indices `iA` to `B` along `iB`; toggle conjugation (`conjA` and `conjB`)
  """
  function mastercontractor(A::tens{X},iA::Array{intType,1},B::tens{Y},iB::Array{intType,1},
                          conjA::Bool,conjB::Bool,Z::tens{W}...;alpha::Number=1.,
                          beta::Number=1.) where {X <: Number, Y <: Number, W <: Number}
    notvecA = setdiff([i for i = 1:ndims(A)],iA)
    C = prepareT(A,notvecA,iA,conjA)
    notvecB = setdiff([i for i = 1:ndims(B)],iB)
    D = prepareT(B,iB,notvecB,conjB)

    if length(Z) > 0
      tempZ = reshape(Z[1],size(C,1),size(D,2))
      CD = blas_pick(C,D,tempZ...,alpha=alpha,beta=beta)
    else
      CD = blas_pick(C,D,alpha=alpha,beta=beta)
    end

    notvecAsize = [size(A,w) for w in notvecA]
    notvecBsize = [size(B,w) for w in notvecB]
    AAsizes = vcat(notvecAsize,notvecBsize)
    outType = typeof(X(1)*Y(1))
    nelem = prod(size(CD))
    return tens{outType}(AAsizes,reshape(CD,nelem))
  end


  function checkcontract(A::X,iA::O,B::Y,iB::P,Z::W...;alpha::Number=1.,beta::Number=1.) where {X <: TensType, Y <: TensType, W <: TensType, O <: intvecType, P <: intvecType}
    mA,mB = checkType(A,B)

    if size(mA)[iA] == size(mB)[iB]
      println("contracting over indices with equal sizes")
    else
      error("some indices in A or B are not equal size; A->",size(mA)[iA],", B->",size(mB)[iB])
    end
    if typeof(mA) <: qarray
      println("checking flux:")
      checkflux(mA)
      checkflux(mB)
      for a = 1:length(iA)
        println("contracted index $a")
        if mA.QnumMat[iA[a]] == mB.QnumMat[iB[a]]
          println("equal quantum numbers on indices on index ",iA[a]," of A and index ",iB[a]," of B")
        else
          error("unequal quantum numbers on indices on index ",iA[a]," of A and index ",iB[a]," of B")
        end
      end
    end
    nothing
  end
end
