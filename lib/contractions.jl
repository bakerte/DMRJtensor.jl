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
    Module: contractions

Contract two tensors together

See also: [`decompositions`](@ref)
"""
#=
module contractions
using ..tensor
import LinearAlgebra
=#
  """
      libmult(C,D[,Z,alpha=,beta=])
  Chooses the best matrix multiply function for tensor contraction (dense tensor or sub-block)
  + Outputs alpha * C * D (+ beta * Z)
  """
  function libmult(C::Array{W,2},D::Array{X,2},tempZ::Array{P,2}...;alpha::Number=1.,beta::Number=1.) where {W <: Number, X <: Number, P <: Number}
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
  export libmult

#       +------------------------+
#>------|    Matrix multiply     |---------<
#       +------------------------+

  """
      matrixequiv(X,Lsize,Rsize)

  Reshapes tensor `X` into an `Lsize` x `Rsize` matrix
  """
  function matrixequiv(X::W,Lsize::Integer,Rsize::Integer) where W <: Union{AbstractArray,denstens}
    return matrixequiv!(copy(X),Lsize,Rsize)
  end

  """
      matrixequiv!(X,Lsize,Rsize)

  Reshapes tensor `X` into an `Lsize` x `Rsize` matrix in-place
  """
  function matrixequiv!(X::AbstractArray,Lsize::Integer,Rsize::Integer)
    return reshape!(X,Lsize,Rsize)
  end

  function matrixequiv!(X::denstens,Lsize::Integer,Rsize::Integer)
    return reshape!(X.T,Lsize,Rsize)
  end

  function permutedims_2matrix!(X::AbstractArray,vec::Tuple,Lsize::Integer,Rsize::Integer)
    xM = permutedims(X, vec)
    return reshape!(xM,Lsize,Rsize)
  end

  function permutedims_2matrix!(X::denstens,vec::Tuple,Lsize::Integer,Rsize::Integer)
    return permutedims_2matrix!(makeArray(X),vec,Lsize,Rsize)
  end

  """
      prepareT(A,Lvec,Rev,conjvar)

  Forms the matrix-equivlanet of a tensor defined by `Lvec` indices forming the rows and `Rvec` indices forming the columns; toggle conjugate (`conjvar`)
  """
  function prepareT(A::P,Lvec::Tuple,Rvec::Tuple,conjvar::Bool) where P <: Union{AbstractArray,denstens}
    vec = (Lvec...,Rvec...)
    Lsize = length(Lvec) > 0 ? prod(a->size(A,a),Lvec) : 1
    Rsize = length(Rvec) > 0 ? prod(a->size(A,a),Rvec) : 1

    issort = true
    a = 1
    while issort && a < length(Lvec)
      a += 1
      issort = Lvec[a-1] < Lvec[a]
    end
    if length(Rvec) > 0 && length(Lvec) > 0
      issort = issort && Lvec[end] < Rvec[1]
      a = 1
    end
    while issort && a < length(Rvec)
      a += 1
      issort = Rvec[a-1] < Rvec[a]
    end

    if issort
      if eltype(A) <: Complex && conjvar
        out = matrixequiv(A,Lsize,Rsize)
      else
        out = matrixequiv!(A,Lsize,Rsize)
      end
    else
      out = permutedims_2matrix!(A,vec,Lsize,Rsize)
    end
    if conjvar
      conj!(out)
    end
    return out
  end

  """
    corecontractor(A,iA,B,iB,conjA,conjB,Z...[,alpha=,beta=])

  Primary contraction function.  Contracts `A` along indices `iA` to `B` along `iB`; toggle conjugation (`conjA` and `conjB`)
  """
  function corecontractor(conjA::Bool,conjB::Bool,A::P,iA::X,B::R,iB::Y,
                          Z::S...;alpha::Number=1.,beta::Number=1.) where {X <: intvecType,Y <: intvecType,P <: Union{AbstractArray,denstens}, R <: Union{AbstractArray,denstens}, S <: Union{AbstractArray,denstens}}
    notvecA = findnotcons(ndims(A),iA)
    C = prepareT(A,notvecA,iA,conjA)
    notvecB = findnotcons(ndims(B),iB) 
    D = prepareT(B,iB,notvecB,conjB)

    if length(Z) > 0
      tempZ = reshape(Z[1],size(C,1),size(D,2))
      CD = libmult(C,D,tempZ...,alpha=alpha,beta=beta)
    else
      CD = libmult(C,D,alpha=alpha,beta=beta)
    end

    AAsizes = ()
    for w = 1:length(notvecA)
      AAsizes = (AAsizes...,size(A,notvecA[w]))
    end
    for w = 1:length(notvecB)
      AAsizes = (AAsizes...,size(B,notvecB[w]))
    end
    return CD,AAsizes
  end

  """
      maincontractor(A,iA,B,iB,conjA,conjB[,Z,alpha=,beta=])

  Primary contraction function.  Contracts `A` along indices `iA` to `B` along `iB`; toggle conjugation (`conjA` and `conjB`)
  """
  function maincontractor(conjA::Bool,conjB::Bool,A::AbstractArray,iA::X,B::AbstractArray,iB::Y,
                            Z::AbstractArray...;alpha::Number=1.,beta::Number=1.) where {X <: intvecType,Y <: intvecType}
    CD,AAsizes = corecontractor(conjA,conjB,A,iA,B,iB,Z...,alpha=alpha,beta=beta)
    return reshape(CD,AAsizes)
  end

  """
      maincontractor(A,iA,B,iB,conjA,conjB[,Z,alpha=,beta=])

  Primary contraction function.  Contracts `A` along indices `iA` to `B` along `iB`; toggle conjugation (`conjA` and `conjB`)
  """
  function maincontractor(conjA::Bool,conjB::Bool,A::tens{X},iA::Tuple,B::tens{Y},iB::Tuple,
                          Z::tens{W}...;alpha::Number=1.,
                          beta::Number=1.) where {X <: Number, Y <: Number, W <: Number}
    CD,AAsizes = corecontractor(conjA,conjB,A,iA,B,iB,Z...,alpha=alpha,beta=beta)
    outType = typeof(X(1)*Y(1))
    nelem = prod(size(CD))
    return tens{outType}(AAsizes,reshape(CD,nelem))
  end
  export maincontractor

#       +----------------------------+
#>------| Contraction function calls |---------<
#       +----------------------------+

  """
      contract(A[,B,alpha=])
  Contracts to (alpha * A * B and returns a scalar output...if only `A` is specified, then the norm is evaluated

  See also: [`ccontract`](@ref) [`contractc`](@ref) [`ccontractc`](@ref)
  """
  function contract(A::Q,B::R;alpha::Number=1.) where {Q <: TensType,R <: TensType}
    mA,mB = checkType(A,B)
    vec_in = ntuple(i->i,ndims(mA)) #[i for i = 1:ndims(mA)]
    C = contract(mA,vec_in,mB,vec_in,alpha=alpha)
    return searchindex(C,1,1)
  end

  """
      ccontract(A[,B,alpha=])
  Similar to contract but 'A' is conjugated

  See also: ['ccontract'](@ref) ['contractc'](@ref) ['ccontractc'](@ref)
  """
  function ccontract(A::Q,B::R;alpha::Number=1.) where {Q <: TensType,R <: TensType}
    mA,mB = checkType(A,B)
    vec_in = ntuple(i->i,ndims(mA)) #[i for i = 1:ndims(mA)]
    C = ccontract(mA,vec_in,mB,vec_in,alpha=alpha)
    return searchindex(C,1,1)
  end

  """
      contractc(A[,B,alpha=])
  Similar to contract but 'B' is conjugated

  See also: ['ccontract'](@ref) ['contractc'](@ref) ['ccontractc'](@ref)
  """
  function contractc(A::Q,B::R;alpha::Number=1.) where {Q <: TensType,R <: TensType}
    mA,mB = checkType(A,B)
    vec_in = ntuple(i->i,ndims(mA)) #[i for i = 1:ndims(mA)]
    C = contractc(mA,vec_in,mB,vec_in,alpha=alpha)
    return searchindex(C,1,1)
  end

  """
      ccontractc(A[,B,alpha=])
  Similar to contract but 'A' and 'B' are conjugated

  See also: ['ccontract'](@ref) ['contractc'](@ref) ['ccontractc'](@ref)
  """
  function ccontractc(A::Q,B::R;alpha::Number=1.) where {Q <: TensType,R <: TensType}
    mA,mB = checkType(A,B)
    vec_in = ntuple(i->i,ndims(mA)) #[i for i = 1:ndims(mA)]
    C = ccontractc(mA,vec_in,mB,vec_in,alpha=alpha)
    return searchindex(C,1,1)
  end



  function contract(A::Q;alpha::Number=1.) where {Q <: TensType}
    return contract(A,A,alpha=alpha)
  end

  function ccontract(A::Q;alpha::Number=1.) where {Q <: TensType}
    return ccontract(A,A,alpha=alpha)
  end

  function contractc(A::Q;alpha::Number=1.) where {Q <: TensType}
    return contractc(A,A,alpha=alpha)
  end

  function ccontractc(A::Q;alpha::Number=1.) where {Q <: TensType}
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
  function contract(A::TensType,iA::O,B::TensType,iB::P,Z::TensType...;alpha::Number=1.,beta::Number=1.) where {O <: intvecType, P <: intvecType}
    mA,mB = checkType(A,B)
    return maincontractor(false,false,mA,convIn(iA),mB,convIn(iB),Z...,alpha=alpha,beta=beta)
  end

  """
      ccontract(A,iA,B,iB[,Z,alpha=,beta=])
  Similar to contract but 'A' is conjugated

  See also: ['ccontract'](@ref) ['contractc'](@ref) ['ccontractc'](@ref)
  """
  function ccontract(A::TensType,iA::O,B::TensType,iB::P,Z::TensType...;alpha::Number=1.,beta::Number=1.) where {O <: intvecType, P <: intvecType}
    mA,mB = checkType(A,B)
    return maincontractor(true,false,mA,convIn(iA),mB,convIn(iB),Z...,alpha=alpha,beta=beta)
  end

  """
      contractc(A,iA,B,iB[,Z,alpha=,beta=])
  Similar to contract but 'B' is conjugated

  See also: ['ccontract'](@ref) ['contractc'](@ref) ['ccontractc'](@ref)
  """
  function contractc(A::TensType,iA::O,B::TensType,iB::P,Z::TensType...;alpha::Number=1.,beta::Number=1.) where {O <: intvecType, P <: intvecType}
    mA,mB = checkType(A,B)
    return maincontractor(false,true,mA,convIn(iA),mB,convIn(iB),Z...,alpha=alpha,beta=beta)
  end

  """
      ccontractc(A,iA,B,iB[,Z,alpha=,beta=])
  Similar to contract but 'A' and 'B' are conjugated

  See also: ['ccontract'](@ref) ['contractc'](@ref) ['ccontractc'](@ref)
  """
  function ccontractc(A::TensType,iA::O,B::TensType,iB::P,Z::TensType...;alpha::Number=1.,beta::Number=1.) where {O <: intvecType, P <: intvecType}
    mA,mB = checkType(A,B)
    return maincontractor(true,true,mA,convIn(iA),mB,convIn(iB),Z...,alpha=alpha,beta=beta)
  end

  """
      contract(order,A,iA,B,iB[,Z,alpha=,beta=])
  Permutes result of [`contract`](@ref) to `order`
  """
  function contract(order::L,A::TensType,iA::O,B::TensType,iB::P,Z::TensType...;alpha::Number=1.,beta::Number=1.) where {O <: intvecType, P <: intvecType, L <: intvecType}
    mA,mB = checkType(A,B)
    newT = contract(mA,iA,mB,iB,Z...,alpha=alpha,beta=beta)
    return permutedims!(newT,convIn(order))
  end

  """
      ccontract(order,A,iA,B,iB[,Z,alpha=,beta=])
  Permutes result of [`ccontract`](@ref) to `order`
  """
  function ccontract(order::L,A::TensType,iA::O,B::TensType,iB::P,Z::TensType...;alpha::Number=1.,beta::Number=1.) where {O <: intvecType, P <: intvecType, L <: intvecType}
    mA,mB = checkType(A,B)
    newT = ccontract(mA,iA,mB,iB,Z...,alpha=alpha,beta=beta)
    return permutedims!(newT,convIn(order))
  end

  """
      contractc(order,A,iA,B,iB[,Z,alpha=,beta=])
  Permutes result of [`contractc`](@ref) to `order`
  """
  function contractc(order::L,A::TensType,iA::O,B::TensType,iB::P,Z::TensType...;alpha::Number=1.,beta::Number=1.) where {O <: intvecType, P <: intvecType, L <: intvecType}
    mA,mB = checkType(A,B)
    newT = contractc(mA,iA,mB,iB,Z...,alpha=alpha,beta=beta)
    return permutedims!(newT,convIn(order))
  end

  """
      ccontractc(order,A,iA,B,iB[,Z,alpha=,beta=])
  Permutes result of [`ccontractc`](@ref) to `order`
  """
  function ccontractc(order::L,A::TensType,iA::O,B::TensType,iB::P,Z::TensType...;alpha::Number=1.,beta::Number=1.) where {O <: intvecType, P <: intvecType, L <: intvecType}
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
  function trace(A::TensType,iA::W) where W <: Union{intvecType,AbstractArray}
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



  #       +------------------------+
  #>------| Quantum number version |---------<
  #       +------------------------+


#  using ..QN
#  using ..Qtensor
#  using ..Qtask

  function remQN(Qvec::Array{Array{Q,1},1},vec::intvecType,conjvar::Bool) where Q <: Qnum
    remAQ = Array{Array{Q,1},1}(undef,length(vec))
    for q = 1:length(vec)
      #=@inbounds=# remAQ[q] = Q[copy(Qvec[vec[q]][j]) for j = 1:length(Qvec[vec[q]])]
    end
    if conjvar
      for q = 1:length(remAQ)
        for j = 1:length(remAQ[q])
          #=@inbounds=# inv!(remAQ[q][j])
        end
      end
    end
    return remAQ
  end

  function computeQsum(A::Qtens{W,Q},Ablocks::Array{P,1},LR::P,Aqind::P,conjvar::Bool) where {W <: Number, Q <: Qnum, P <: Integer}
    Asum = Q()
    for w = 1:length(Ablocks)
      @inbounds pos = Ablocks[w]
      @inbounds index = A.ind[Aqind][LR][w,1] + 1
      @inbounds Qnumber = A.QnumMat[pos][index]
      @inbounds add!(Asum,getQnum(pos,index,A.QnumMat,A.QnumSum))
    end
    if conjvar
      inv!(Asum)
    end
    return Asum
  end


  #  import Base.Threads.@spawn

  const one = (1,)
  const two = (2,)

  function maincontractor(conjA::Bool,conjB::Bool,QtensA::Qtens{W,Q},vecA::Tuple,QtensB::Qtens{R,Q},vecB::Tuple,Z::Qtens{S,Q}...;alpha::Number=1.,beta::Number=1.) where {W <: Number, R <: Number, S <: Number, Q <: Qnum}
      if QtensA === QtensB && ((conjA && W <: Complex) || (conjB && R <: Complex))
        QtensB = copy(QtensA)
      end
      

      conA,notconA = getinds(QtensA,vecA)
      conB,notconB = getinds(QtensB,vecB)

      taskA = Array{qarray,1}(undef,2)
      #=Threads.@threads=# for w = 1:2
        if w == 1
          taskA[w] = changeblock(QtensA,notconA,conA,leftflux=true)
        else
          taskA[w] = changeblock(QtensB,conB,notconB,leftflux=false)
        end
      end

      A = taskA[1]
      B = taskA[2]


      commonblocks = matchblocks((conjA,conjB),A,B)


      if length(Z) > 0
        Zone = [i for i = 1:length(notconA)]
        Ztwo = [i + length(notconA) for i = 1:length(notconB)]
        Zed = changeblock(Z[1],Zone,Ztwo)
        Zcommonblocks = matchblocks((conjA,false),A,Zed)
      else
        Zed = Z
      end

      outType = typeof(W(1) * R(1))

      numQNs = length(commonblocks)
      outTens = Array{tens{outType},1}(undef,numQNs)

      newrowcols = Array{Array{Array{intType,2},1},1}(undef,numQNs)
      newQblocksum = Array{Array{Q,1},1}(undef,numQNs)

      Ablocks = A.currblock[1]
      Bblocks = B.currblock[2]

#      let A = A, B = B, outTens = outTens, vecA = vecA, vecB = vecB, numQNs = numQNs, conjA = conjA, conjB = conjB, commonblocks = commonblocks, alpha = alpha, beta = beta, Zed = Zed
        Threads.@threads for q = 1:numQNs
          Aqind = commonblocks[q][1]
          Bqind = commonblocks[q][2]

          if length(Z) > 0
            Zqind = Zcommonblocks[q][2]
            resT = maincontractor(conjA,conjB,A.T[Aqind],two,B.T[Bqind],one,alpha=alpha,beta=beta,Zed.T[Zqind]...)
          else
            resT = maincontractor(conjA,conjB,A.T[Aqind],two,B.T[Bqind],one,alpha=alpha)
          end

          outTens[q] = tens{outType}(resT)

          newrowcols[q] = [A.ind[Aqind][1],B.ind[Bqind][2]]

          Asum = computeQsum(A,Ablocks,1,Aqind,conjA)
          Bsum = computeQsum(B,Bblocks,2,Bqind,conjB)
          newQblocksum[q] = [Asum,Bsum]
        end
#      end

      remAQ = A.QnumMat[notconA] #remQN(A.QnumMat,notconA,conjA)
      remBQ = B.QnumMat[notconB] #remQN(B.QnumMat,notconB,conjB)
      newQnumMat = vcat(remAQ,remBQ)

      sumAQ = remQN(A.QnumSum,notconA,conjA)
      sumBQ = remQN(B.QnumSum,notconB,conjB)
      newQnumSum = vcat(sumAQ,sumBQ)

      notvecA = findnotcons(ndims(A),vecA)
      notvecB = findnotcons(ndims(B),vecB)

#      println(QtensA.size)

#      println(QtensA.size[notvecA])
#      println(QtensB.size[notvecB])
      newsize = [Array{intType,1}(undef,length(QtensA.size[notvecA[w]])) for w = 1:length(notvecA)]
      newsize = vcat(newsize,[Array{intType,1}(undef,length(QtensB.size[notvecB[w]])) for w = 1:length(notvecB)])
      counter = 0
      for w = 1:length(newsize)
        for a = 1:length(newsize[w])
          counter += 1
          newsize[w][a] = counter
        end
      end

      keepers = [size(outTens[q],1) > 0 && size(outTens[q],2) > 0 for q = 1:length(outTens)]

      newflux = conjA ? inv(QtensA.flux) : copy(QtensA.flux)
      newflux += conjB ? inv(QtensB.flux) : copy(QtensB.flux)

      newcurrblocks = [[i for i = 1:length(notconA)],[i + length(notconA) for i = 1:length(notconB)]]

      if sum(keepers) < length(keepers)
        newT = outTens[keepers]
        newinds = newrowcols[keepers]
        newQblocks = newQblocksum[keepers]
      else
        newT = outTens
        newinds = newrowcols
        newQblocks = newQblocksum
      end

      newQtensOutput = Qtens{outType,Q}(newsize,newT,newinds,newcurrblocks,newQblocks,newQnumMat,newQnumSum,newflux)

      return newQtensOutput
  end


#         +-----------------+
#>--------|  Check contract |------<
#         +-----------------+


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
        AQNs = recoverQNs(iA[a],A)
        BQNs = recoverQNs(iB[a],B)
        println("contracted index $a")
        if AQNs[iA[a]] == inv.(BQNs[iB[a]])
          println("matching quantum numbers on indices on index ",iA[a]," of A and index ",iB[a]," of B")
        else
          error("unmatching quantum numbers on indices on index ",iA[a]," of A and index ",iB[a]," of B")
        end
      end
    end
    nothing
  end
  export checkcontract
#end
