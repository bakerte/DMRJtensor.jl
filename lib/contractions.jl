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
  Module: contractions

Contract two tensors together

See also: [`decompositions`](@ref)
"""
#=
module contractions
using ..tensor
import LinearAlgebra
=#

#       +------------------------+
#>------|    Matrix multiply     |---------<
#       +------------------------+
"""
  C = libmult([alpha,]C,D[,beta,Z])

Chooses the best matrix multiply function for tensor contraction (dense tensor or sub-block) withoutput matrix `C`

+ Outputs `C` * `D` if `alpha`=1
+ Outputs `alpha` * `C` * `D` if `Z` is not input
+ Outputs `alpha` * `C` * `D` + `beta` * `Z` if `Z` is input
"""
function libmult! end

"""
    libmult(tA, tB, alpha, A, B)
Return `alpha*A*B` or the other three variants according to [`tA`](@ref stdlib-blas-trans) and `tB`.
"""
libmult(tA, tB, alpha, A, B)

"""
    libmult(tA, tB, alpha, A, B, beta, C)
Return `alpha*A*B+beta*C` or the other three variants according to [`tA`](@ref stdlib-blas-trans) and `tB`.
"""
libmult(tA, tB, alpha, A, B,beta,C)

"""
    libmult(tA, tB, A, B)
Return `A*B` or the other three variants according to [`tA`](@ref stdlib-blas-trans) and `tB`.
"""
libmult(tA, tB, A, B)

"""
  permq(A,iA)

Answers the question of whether to permute the tensor `A` with contracted indices `iA`
"""
function permq(A::densTensType,iA::Union{Array{intType,1},NTuple{K,intType}}) where K
  nopermL = true
  w = 0
  while nopermL && w < length(iA)
    w += 1
    nopermL = iA[w] == w
  end
  nopermR = !nopermL
  if nopermR
    w = length(iA)
    end_dim = ndims(A)
    while nopermR && w > 0
      nopermR = iA[w] == end_dim
      end_dim -= 1
      w -= 1
    end
  end
  return nopermL,nopermR
end

"""
  willperm(conjA,WA,AnopermL,AnopermR)

Determines LAPACK flag based on output of `permq` and whether to conjugate `conjA` and what type `WA`
"""
function willperm(conjA::Bool,WA::DataType,AnopermL::Bool,AnopermR::Bool)
  if AnopermL
    transA = conjA && WA <: Complex ? 'C' : 'T'
    Aperm = true
  else
    transA = 'N'
    Aperm = AnopermR
  end
  return Aperm,transA
end

"""
  prepareT(A,Lvec,Rvec,conjvar)

Converts input tensor `A` to its matrix equivalent with left indices contained in `Lvec` and right indices contained in `Rvec` and whether to conjugate (`conjvar`)
"""
function prepareT(A::densTensType,Lvec::Union{Array{intType,1},NTuple{K,intType}},Rvec::Union{Array{intType,1},NTuple{P,intType}},conjvar::Bool) where {K,P}
  newdimsA = Array{intType,1}(undef,ndims(A))
  counter = 0
  @inbounds @simd for w = 1:length(Lvec)
    counter += 1
    newdimsA[counter] = Lvec[w]
  end
  @inbounds @simd for w = 1:length(Rvec)
    counter += 1
    newdimsA[counter] = Rvec[w]
  end
  pA = permutedims(A,newdimsA)
  if conjvar
    conj!(pA)
  end
  return pA
end

"""
  getsizes(A,iA,AAsizes,counter)

Finds sizes of the matrix equivalents for tensor `A`, contracted indices `iA`, sizes of the new tensor `AAsizes`, and a `counter` which increments between the two tensors being contracted over
"""
function getsizes(A::TensType,iA::intvecType,AAsizes::Array{intType,1},counter::intType)
  Lsize = innersizeL = 1
  @inbounds for w = 1:ndims(A)
    if !(w in iA)
      counter += 1
      AAsizes[counter] = size(A,w)
      Lsize *= size(A,w)
    else
      innersizeL *= size(A,w)
    end
  end
  return Lsize,innersizeL
end

"""
  maincontractor(conjA,conjB,A,iA,B,iB,Z...;alpha=1,beta=1)

Contraction function (`alpha`*`A`*`B`+`beta`*`Z`) and can conjugate A (`conjA`) or B (`conjB`)
"""
function maincontractor(conjA::Bool,conjB::Bool,A::densTensType,iA::intvecType,B::densTensType,iB::intvecType,Z::TensType...;alpha::Number=1,beta::Number=1)
  AnopermL,AnopermR = permq(A,iA)
  BnopermL,BnopermR = permq(B,iB)

  Aperm,transA = willperm(conjA,eltype(A),AnopermL,AnopermR)
  Bperm,transB = willperm(conjB,eltype(B),BnopermR,BnopermL)

  Aremain = ndims(A)-length(iA)
  Bremain = ndims(B)-length(iB)
  AAsizes = Array{intType,1}(undef,Aremain+Bremain)
  Lsize,innersizeL = getsizes(A,iA,AAsizes,0)
  Rsize,innersizeR = getsizes(B,iB,AAsizes,Aremain)

  if Aperm && Bperm
    mulA = conjA && eltype(A) <: Complex && !AnopermL ? conj(A) : A
    mulB = conjB && eltype(B) <: Complex && !BnopermR ? conj(B) : B
  elseif Aperm
    mulA = conjA && eltype(A) <: Complex && !AnopermL ? conj(A) : A
    notvecB = findnotcons(ndims(B),iB)
    mulB = prepareT(B,iB,notvecB,conjB)
  elseif Bperm
    notvecA = findnotcons(ndims(A),iA)
    mulA = prepareT(A,notvecA,iA,conjA)
    mulB = conjB && eltype(B) <: Complex && !BnopermR ? conj(B) : B
  else
    notvecA = findnotcons(ndims(A),iA)
    mulA = prepareT(A,notvecA,iA,conjA)
    notvecB = findnotcons(ndims(B),iB)
    mulB = prepareT(B,iB,notvecB,conjB)
  end

  if length(Z) > 0
    outType = typeof(eltype(A)(1)*eltype(B)(1)*eltype(Z[1])(1)*typeof(alpha)(1)*typeof(beta)(1))
    type_alpha = convert(outType,alpha)
    type_beta = convert(outType,beta)
    out = libmult(transA,transB,type_alpha,mulA,mulB,type_beta,Z[1],Lsize,innersizeL,innersizeR,Rsize)
  elseif isapprox(alpha,1)
    out = libmult(transA,transB,mulA,mulB,Lsize,innersizeL,innersizeR,Rsize)
  else
    outType = typeof(eltype(A)(1)*eltype(B)(1)*typeof(alpha)(1))
    type_alpha = convert(outType,alpha)
    out = libmult(transA,transB,type_alpha,mulA,mulB,Lsize,innersizeL,innersizeR,Rsize)
  end

  if typeof(A) <: denstens || typeof(B) <: denstens
    outTens = tens{eltype(out)}(AAsizes,out)
  else
    outTens = reshape(out,AAsizes)
  end
  return outTens
end

#       +----------------------------+
#>------| Contraction function calls |---------<
#       +----------------------------+


"""
  dot(A,B;Lfct=adjoint,Rfct=identity)

takes `identity` or `adjoint` (or equivalently `conj`) for the `Lfct` and `Rfct`

See also: [`identity`](@ref) [`adjoint`](@ref) [`conj`](@ref)
"""
function dot(inA::densTensType,inB::densTensType;Lfct::Function=adjoint,Rfct::Function=identity,transA::Bool=true)
  A = typeof(inA) <: denstens ? inA.T : inA
  B = typeof(inB) <: denstens ? inB.T : inB

  newtypeAB = typeof(eltype(A)(1) * eltype(B)(1))
  val = newtypeAB(0)
  dim1 = length(A) #size(inA,transA ? 1 : 2)
  @inbounds @simd for j = 1:dim1
    val += Lfct(A[j]) * Rfct(B[j])
  end
  return val
end

function dot(C::Qtens{W,Q},D::Qtens{R,Q};Lfct::Function=adjoint,Rfct::Function=identity) where {W <: Number, R <: Number, Q <: Qnum}
  newtype = typeof(W(1)*R(1))

  A = changeblock(C,intType[],intType[i for i = 1:length(C.QnumMat)])
  B = changeblock(D,intType[i for i = 1:length(D.QnumMat)],intType[])
  conjA = Lfct != identity
  conjB = Rfct != identity
  commonblocks = matchblocks((conjA,conjB),A,B,ind=(2,1))

  val = newtype(0)
  @inbounds for q = 1:length(commonblocks)
    Aqind = commonblocks[q][1]
    Bqind = commonblocks[q][2]
    val += dot(A.T[Aqind],B.T[Bqind],Lfct=Lfct,Rfct=Rfct)
  end
  return val
end

"""
  

The function will admit any dimension or input element type in terms of the arrays `A`, `H`, and `B`. However, the function will only work properly when the total elements of `A` times those in `B` equal the elements in `H`

If more operators `H` should be contracted between `A` and `B`, then it is advised here to contract them first before using this function

"""
function dot(inA::densTensType,inH::densTensType,inB::densTensType;Lfct::Function=adjoint,Rfct::Function=identity,transA::Bool=true)

  A = typeof(inA) <: denstens ? inA.T : inA
  H = typeof(inH) <: denstens ? inH.T : inH
  B = typeof(inB) <: denstens ? inB.T : inB

  newtypeAHB = typeof(eltype(A)(1) * eltype(H)(1) * eltype(B)(1))
  val = newtypeAHB(0)

  dim1 = length(A) #size(inA,transA ? 1 : 2)
  dim2 = length(B) #size(inB,1)
  newtypeAH = typeof(eltype(A)(1) * eltype(H)(1))
  @inbounds for j = 1:dim2
    ival = newtypeAH(0)
    savedim = dim1*(j-1)
    @inbounds @simd for i = 1:dim1
      ival += Lfct(A[i]) * H[i + savedim]
    end
    val += ival * Rfct(B[j])
  end
  return val
end

#not sure how to assign matrix blocks...or how to get information from that to A and B vectors
#must be set up correctly with changeblock
function dot(A::Qtens{W,Q},H::Qtens{Y,Q},B::Qtens{R,Q};Lfct::Function=adjoint,Rfct::Function=identity) where {W <: Number, Y <: Number, R <: Number, Q <: Qnum}
  Acommonblocks = matchblocks((conjA,false),A,H,ind=(2,1))
  Bcommonblocks = matchblocks((conjA,conjB),A,B,ind=(2,1))
  newtype = typeof(W(1)*R(1))
  val = newtype(0)
  @inbounds for q = 1:length(commonblocks)
    Aqind = Acommonblocks[q][1]
    Hqind = Acommonblocks[q][2]
    Bqind = Bcommonblocks[q][2]
    if length(A.T[Aqind]) * length(B.T[Bqind]) != length(H.T[Hqind])
      error("unequal sizes in dot for quantum number tensors for A block: $Aqind, H block: $Hqind, and B block: $Bqind")
    end
    if Aqind != 0 && Hqind != 0 && Bqind != 0
      val += dot(A.T[Aqind],H.T[Hqind],B.T[Bqind],Lfct=Lfct,Rfct=Rfct)
    end
  end
  return val
end
export dot


function *(X::TensType,Y::TensType)
  if ndims(X) == 1 && ndims(Y) == 2
    X = reshape(X,size(X,1),1)
  end
  return contract(X,ndims(X),Y,1)
end

function *(X::LinearAlgebra.Diagonal{R, Vector{R}},Y::tens{W}) where {R <: Number, W <: Number}
  outType = typeof(R(1)*W(1))
  Z = Array{outType,1}(undef,length(Y))
  longdim = cld(length(Y.T),size(X,1))
  for y = 1:longdim
    tempind = size(X,1)*(y-1)
    @inbounds @simd for x = 1:size(X,1)
      Z[x + tempind] = X[x,x]*Y.T[x + tempind]
    end
  end
  return tens{outType}(Y.size,Z)
end

function *(Y::tens{W},X::LinearAlgebra.Diagonal{R, Vector{R}}) where {R <: Number, W <: Number}
  outType = typeof(R(1)*W(1))
  Z = Array{outType,1}(undef,length(Y))
  longdim = cld(length(Y.T),size(X,1))
  for x = 1:longdim
    tempind = x - longdim
    @inbounds @simd for y = 1:size(X,1)
      zval = tempind + longdim*y
      Z[zval] = Y.T[zval]*X[y,y]
    end
  end
  return tens{outType}(Y.size,Z)
end



#=

function *(X::LinearAlgebra.UpperTriangular{R, Matrix{R}},Y::tens{W}) where {R <: Number, W <: Number}
  outType = typeof(R(1)*W(1))
  Z = Array{outType,1}(undef,length(Y))
  longdim = cld(length(Y.T),size(X,1))
  for y = 1:longdim
    tempind = size(X,1)*(y-1)
    @inbounds for x = 1:size(X,1)
      val = outType(0)
      @inbounds @simd for k = x:size(X,2)
        val += X[x,k]*Y.T[k + tempind]
      end
      Z[x + tempind] = val
    end
  end
  return tens{outType}(Y.size,Z)
end

function *(Y::tens{W},X::LinearAlgebra.UpperTriangular{R, Matrix{R}}) where {R <: Number, W <: Number}
  outType = typeof(R(1)*W(1))
  Z = Array{outType,1}(undef,length(Y))
  longdim = cld(length(Y.T),size(X,1))
  for x = 1:size(X,1)
    tempind = x - size(X,1)
    @inbounds for y = 1:longdim
      val = outType(0)
      @inbounds @simd for k = 1:x
        val += Y.T[tempind + size(X,1)*k]*X[k,y]
      end
      zval = tempind + size(X,1)*y
      Z[zval] = val
    end
  end
  return tens{outType}(Y.size,Z)
end



function *(X::LinearAlgebra.LowerTriangular{R, Matrix{R}},Y::tens{W}) where {R <: Number, W <: Number}
  outType = typeof(R(1)*W(1))
  Z = Array{outType,1}(undef,length(Y))
  longdim = cld(length(Y.T),size(X,1))
  for y = 1:longdim
    tempind = size(X,1)*(y-1)
    @inbounds for x = 1:size(X,1)
      val = outType(0)
      @inbounds @simd for k = 1:x
        val += X[x,k]*Y.T[k + tempind]
      end
      Z[x + tempind] = val
    end
  end
  return tens{outType}(Y.size,Z)
end

function *(Y::tens{W},X::LinearAlgebra.LowerTriangular{R, Matrix{R}}) where {R <: Number, W <: Number}
  outType = typeof(R(1)*W(1))
  Z = Array{outType,1}(undef,length(Y))
  longdim = cld(length(Y.T),size(X,1))
  for x = 1:size(X,1)
    tempind = x - size(X,1)
    @inbounds for y = 1:longdim
      val = outType(0)
      @inbounds @simd for k = x:size(X,2)
        val += Y.T[tempind + size(X,1)*k]*X[k,y]
      end
      zval = tempind + size(X,1)*y
      Z[zval] = val
    end
  end
  return tens{outType}(Y.size,Z)
end
=#



#import LinearAlgebra.rmul!
function dmul!(Y::tens{R},X::LinearAlgebra.Diagonal{W, Vector{W}}) where {R <: Number, W <: Number}
  longdim = cld(length(Y.T),size(X,1))
  for y = 1:size(X,1)
    zval = longdim*(y-1)
    val = X[y,y]
    @inbounds @simd for x = 1:longdim
      Y.T[x + zval] *= val
    end
  end
  return Y
end

function dmul!(Y::Array{R,N},X::LinearAlgebra.Diagonal{W, Vector{W}}) where {R <: Number, N, W <: Number}
  longdim = cld(length(Y),size(X,1))
  for y = 1:size(X,1)
    zval = longdim*(y-1)
    val = X[y,y]
    @inbounds @simd for x = 1:longdim
      Y[x + zval] *= val
    end
  end
  return Y
end
#=
function rmul!(Y::AbstractArray{W,N},X::LinearAlgebra.Diagonal{W, Vector{W}}) where {R <: Number, W <: Number, N}
  return LinearAlgebra.rmul!(Y,X)
end
=#
function dmul!(X::R,Y::tens{W}) where {R <: Number, W <: Number}
  return tensorcombination!((X,),Y)
end
function dmul!(Y::tens{W},X::R) where {R <: Number, W <: Number}
  return dmul!(X,Y)
end
#export rmul!

#import LinearAlgebra.lmul!
function dmul!(X::LinearAlgebra.Diagonal{R, Vector{R}},Y::tens{W}) where {R <: Number, W <: Number}
  longdim = cld(length(Y.T),size(X,1))
  for y = 1:longdim
    tempind = size(X,1)*(y-1)
    @inbounds @simd for x = 1:size(X,1)
      Y.T[x + tempind] *= X[x,x]
    end
  end
  return Y
end

function dmul!(X::LinearAlgebra.Diagonal{R, Vector{R}},Y::Array{W,N}) where {R <: Number, W <: Number, N}
  longdim = cld(length(Y),size(X,1))
  for y = 1:longdim
    tempind = size(X,1)*(y-1)
    @inbounds @simd for x = 1:size(X,1)
      Y[x + tempind] *= X[x,x]
    end
  end
  return Y
end

function dmul(X::LinearAlgebra.Diagonal{R, Vector{R}},Y::densTensType) where {R <: Number}
  return dmul!(X,copy(Y))
end

function dmul(X::densTensType,Y::LinearAlgebra.Diagonal{R, Vector{R}}) where {R <: Number}
  return dmul!(copy(X),Y)
end
#=
function lmul!(X::LinearAlgebra.Diagonal{R, Vector{R}},Y::AbstractArray{W,N}) where {R <: Number, W <: Number, N}
  return LinearAlgebra.lmul!(Y,X)
end
=#
#=
function dmul!(Y::TensType,X::R) where {R <: Number}
  return tensorcombination!((X,),Y)
end
function dmul!(X::R,Y::tens{W}) where {R <: Number, W <: Number}
  return dmul!(Y,X)
end
=#
export dmul!


function contract(A::LinearAlgebra.Diagonal{W,Vector{W}},B::densTensType;alpha::Number=eltype(A)(1)) where W <: Number
  return trace(isapprox(alpha,1) ? A*B : alpha*A*B)
end

function contract!(A::LinearAlgebra.Diagonal{W,Vector{W}},B::densTensType;alpha::Number=eltype(A)(1)) where W <: Number
  out = dot(A,B,Lfct=identity,Rfct=identity)
  return  alpha*out
end
#=
function contract(A::Union{TensType,LinearAlgebra.Diagonal{W, Vector{W}}},iA::intvecType,B::Union{TensType,LinearAlgebra.Diagonal{W, Vector{W}}},iB::intvecType,Z::TensType...;alpha::Number=eltype(A)(1),beta::Number=eltype(A)(1)) where W <: Number
  return diagcontract(A,convIn(iA),B,convIn(iB),Z...,alpha=alpha,beta=beta)
end
=#

function contract(A::LinearAlgebra.Diagonal{W, Vector{W}},iA::intvecType,B::densTensType,iB::intvecType,Z::densTensType...;alpha::Number=eltype(A)(1),beta::Number=eltype(A)(1)) where W <: Number
  return diagcontract!(A,convIn(iA),B,convIn(iB),Z...,alpha=alpha,beta=beta,inplace=false)
end

function contract(A::densTensType,iA::intvecType,B::LinearAlgebra.Diagonal{W, Vector{W}},iB::intvecType,Z::densTensType...;alpha::Number=eltype(A)(1),beta::Number=eltype(A)(1)) where W <: Number
  return diagcontract!(A,convIn(iA),B,convIn(iB),Z...,alpha=alpha,beta=beta,inplace=false)
end

function diagcontract!(A::LinearAlgebra.Diagonal{W, Vector{W}},iA::intvecType,B::densTensType,iB::intvecType,Z::densTensType...;alpha::Number=eltype(A)(1),beta::Number=eltype(A)(1),inplace::Bool=true) where W <: Number

  Lperm = true
  w = 0
  while Lperm && w < length(iB)
    w += 1
    Lperm = iB[w] == w
  end

  Rperm = true
  w = length(iB)
  while Rperm && w > 0
    Rperm = iB[w] == w
    w -= 1
  end

  if Lperm
    C = inplace ? dmul!(A,B) : dmul(A,B)
  elseif Rperm
    C = inplace ? dmul!(B,A) : dmul(B,A)
  else
    notvecB = findnotcons(ndims(B),iB)
    mB = prepareT(B,iB,notvecB,false)
    C = inplace ? dmul!(A,mB) : dmul(A,mB)
  end
  if length(Z) == 0
    if isapprox(alpha,1)
      out = C
    else
      out = dmul!(alpha,C)
    end
  else
    out = tensorcombination!((alpha,beta),C,Z[1])
  end
  return out
end

function diagcontract!(A::densTensType,iA::intvecType,B::LinearAlgebra.Diagonal{W, Vector{W}},iB::intvecType,Z::densTensType...;alpha::Number=eltype(A)(1),beta::Number=eltype(A)(1),inplace::Bool=true) where W <: Number

  Lperm = true
  w = 0
  while Lperm && w < length(iA)
    w += 1
    Lperm = iA[w] == w
  end

  Rperm = true
  w = length(iA)
  while Rperm && w > 0
    Rperm = iA[w] == w
    w -= 1
  end

  if Lperm
    C = inplace ? dmul!(B,A) : dmul(B,A)
  elseif Rperm
    C = inplace ? dmul!(A,B) : dmul(A,B)
  else
    notvecA = findnotcons(ndims(A),iA)
    mA = prepareT(A,notvecA,iA,false)
    C = inplace ? dmul!(mA,B) : dmul(mA,B)
  end
  if length(Z) == 0
    if isapprox(alpha,1)
      out = C
    else
      out = dmul!(alpha,C)
    end
  else
    out = tensorcombination!((alpha,beta),C,Z[1])
  end
  return out
end
export diagcontract!

"""
  C = contract(A,B[,alpha=])

Contracts to (alpha * A * B and returns a scalar output...if only `A` is specified, then the norm is evaluated

See also: [`ccontract`](@ref) [`contractc`](@ref) [`ccontractc`](@ref)
"""
function contract(A::TensType,B::TensType;alpha::Number=eltype(A)(1))
  mA,mB = checkType(A,B)
  out = dot(mA,mB,Lfct=identity,Rfct=identity)
  return  alpha*out
end

"""
  ccontract(A,B[,alpha=])

Similar to contract but 'A' is conjugated

See also: ['ccontract'](@ref) ['contractc'](@ref) ['ccontractc'](@ref)
"""
function ccontract(A::TensType,B::TensType;alpha::Number=eltype(A)(1))
  mA,mB = checkType(A,B)
  out = dot(mA,mB,Lfct=adjoint,Rfct=identity)
  return  alpha*out
end

"""
  contractc(A,B[,alpha=])

Similar to contract but 'B' is conjugated

See also: ['ccontract'](@ref) ['contractc'](@ref) ['ccontractc'](@ref)
"""
function contractc(A::TensType,B::TensType;alpha::Number=eltype(A)(1))
  mA,mB = checkType(A,B)
  out = dot(mA,mB,Lfct=identity,Rfct=adjoint)
  return  alpha*out
end

"""
  ccontractc(A,B[,alpha=])

Similar to contract but 'A' and 'B' are conjugated

See also: ['ccontract'](@ref) ['contractc'](@ref) ['ccontractc'](@ref)
"""
function ccontractc(A::TensType,B::TensType;alpha::Number=eltype(A)(1))
  mA,mB = checkType(A,B)
  out = dot(mA,mB,Lfct=adjoint,Rfct=adjoint)
  return  alpha*out
end

"""
    contract(A[,alpha=])

Self-contraction of input tensor `A` to a scalar value of `A`.`A`

See also: ['ccontract'](@ref) ['contractc'](@ref) ['ccontractc'](@ref)
"""
function contract(A::TensType;alpha::Number=eltype(A)(1))
  out = dot(A,A,Lfct=identity,Rfct=identity)
  return alpha*out
end

"""
    ccontract(A[,alpha=])

Self-contraction of input tensor `A` to a scalar value of `A`.`A` with one of the inputs conjugated

See also: ['ccontract'](@ref) ['contractc'](@ref) ['ccontractc'](@ref)
"""
function ccontract(A::TensType;alpha::Number=eltype(A)(1))
  out = dot(A,A,Lfct=adjoint,Rfct=identity)
  return  alpha*out
end

"""
    contractc(A[,alpha=])

Self-contraction of input tensor `A` to a scalar value of `A`.`A` with one of the inputs conjugated

See also: ['ccontract'](@ref) ['contractc'](@ref) ['ccontractc'](@ref)
"""
function contractc(A::TensType;alpha::Number=eltype(A)(1))
  out = dot(A,A,Lfct=identity,Rfct=adjoint)
  return  alpha*out
end

"""
    ccontractc(A[,alpha=])

Self-contraction of input tensor `A` to a scalar value of `A`.`A` with both inputs conjugated

See also: ['ccontract'](@ref) ['contractc'](@ref) ['contract'](@ref)
"""
function ccontractc(A::TensType;alpha::Number=eltype(A)(1))
  out = dot(A,A,Lfct=adjoint,Rfct=adjoint)
  return  alpha*out
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
function contract(A::TensType,iA::intvecType,B::TensType,iB::intvecType,Z::TensType...;alpha::Number=eltype(A)(1),beta::Number=eltype(A)(1))
  mA,mB = checkType(A,B)
  return maincontractor(false,false,mA,convIn(iA),mB,convIn(iB),Z...,alpha=alpha,beta=beta)
end

"""
  contract(A,iA,B[,Z,alpha=,beta=])

same as `contract(A,iA,B,iB)` but `iB` is over all indices.

See also: [`ccontract`](@ref) [`contractc`](@ref) [`ccontractc`](@ref)
"""
function contract(A::TensType,iA::intvecType,B::TensType,Z::TensType...;alpha::Number=eltype(A)(1),beta::Number=eltype(A)(1))
  mA,mB = checkType(A,B)
  iB = ntuple(w->w,ndims(B))
  return maincontractor(false,false,mA,convIn(iA),mB,iB,Z...,alpha=alpha,beta=beta)
end

"""
  contract(A,B,iB[,Z,alpha=,beta=])

same as `contract(A,iA,B,iB)` but `iB` is over all indices.

See also: [`ccontract`](@ref) [`contractc`](@ref) [`ccontractc`](@ref)
"""
function contract(A::TensType,B::TensType,iB::intvecType,Z::TensType...;alpha::Number=eltype(A)(1),beta::Number=eltype(A)(1))
  mA,mB = checkType(A,B)
  iA = ntuple(w->w,ndims(A))
  return maincontractor(false,false,mA,iA,mB,convIn(iB),Z...,alpha=alpha,beta=beta)
end

"""
    ccontract(A,iA,B,iB[,Z,alpha=,beta=])

Similar to contract but 'A' is conjugated

See also: ['ccontract'](@ref) ['contractc'](@ref) ['ccontractc'](@ref)
"""
function ccontract(A::TensType,iA::intvecType,B::TensType,iB::intvecType,Z::TensType...;alpha::Number=eltype(A)(1),beta::Number=eltype(A)(1))
  mA,mB = checkType(A,B)
  return maincontractor(true,false,mA,convIn(iA),mB,convIn(iB),Z...,alpha=alpha,beta=beta)
end

"""
  ccontract(A,B,iB[,Z,alpha=,beta=])

same as `contract(A,iA,B,iB)` but `iB` is over all indices.

See also: [`ccontract`](@ref) [`contractc`](@ref) [`ccontractc`](@ref)
"""
function ccontract(A::TensType,iA::intvecType,B::TensType,Z::TensType...;alpha::Number=eltype(A)(1),beta::Number=eltype(A)(1))
  mA,mB = checkType(A,B)
  iB = ntuple(w->w,ndims(B))
  return maincontractor(true,false,mA,convIn(iA),mB,iB,Z...,alpha=alpha,beta=beta)
end

"""
  ccontract(A,iA,B[,Z,alpha=,beta=])

same as `contract(A,iA,B,iB)` but `iB` is over all indices.

See also: [`ccontract`](@ref) [`contractc`](@ref) [`ccontractc`](@ref)
"""
function ccontract(A::TensType,B::TensType,iB::intvecType,Z::TensType...;alpha::Number=eltype(A)(1),beta::Number=eltype(A)(1))
  mA,mB = checkType(A,B)
  iA = ntuple(w->w,ndims(A))
  return maincontractor(true,false,mA,iA,mB,convIn(iB),Z...,alpha=alpha,beta=beta)
end

"""
    contractc(A,iA,B,iB[,Z,alpha=,beta=])

Similar to contract but 'B' is conjugated

See also: ['ccontract'](@ref) ['contractc'](@ref) ['ccontractc'](@ref)
"""
function contractc(A::TensType,iA::intvecType,B::TensType,iB::intvecType,Z::TensType...;alpha::Number=eltype(A)(1),beta::Number=eltype(A)(1))
  mA,mB = checkType(A,B)
  return maincontractor(false,true,mA,convIn(iA),mB,convIn(iB),Z...,alpha=alpha,beta=beta)
end

"""
  contractc(A,iA,B[,Z,alpha=,beta=])

same as `contract(A,iA,B,iB)` but `iB` is over all indices.

See also: [`ccontract`](@ref) [`contractc`](@ref) [`ccontractc`](@ref)
"""
function contractc(A::TensType,iA::intvecType,B::TensType,Z::TensType...;alpha::Number=eltype(A)(1),beta::Number=eltype(A)(1))
  mA,mB = checkType(A,B)
  iB = ntuple(w->w,ndims(B))
  return maincontractor(false,true,mA,convIn(iA),mB,iB,Z...,alpha=alpha,beta=beta)
end

"""
  contractc(A,B,iB[,Z,alpha=,beta=])

same as `contract(A,iA,B,iB)` but `iA` is over all indices.

See also: [`ccontract`](@ref) [`contractc`](@ref) [`ccontractc`](@ref)
"""
function contractc(A::TensType,B::TensType,iB::intvecType,Z::TensType...;alpha::Number=eltype(A)(1),beta::Number=eltype(A)(1))
  mA,mB = checkType(A,B)
  iA = ntuple(w->w,ndims(A))
  return maincontractor(false,true,mA,iA,mB,convIn(iB),Z...,alpha=alpha,beta=beta)
end

"""
    ccontractc(A,iA,B,iB[,Z,alpha=,beta=])

Similar to contract but 'A' and 'B' are conjugated

See also: ['ccontract'](@ref) ['contractc'](@ref) ['ccontractc'](@ref)
"""
function ccontractc(A::TensType,iA::intvecType,B::TensType,iB::intvecType,Z::TensType...;alpha::Number=eltype(A)(1),beta::Number=eltype(A)(1))
  mA,mB = checkType(A,B)
  return maincontractor(true,true,mA,convIn(iA),mB,convIn(iB),Z...,alpha=alpha,beta=beta)
end
export contract,ccontract,contractc,ccontractc

"""
  ccontractc(A,iA,B[,Z,alpha=,beta=])

same as `contract(A,iA,B,iB)` but `iB` is over all indices.

See also: [`ccontract`](@ref) [`contractc`](@ref) [`ccontractc`](@ref)
"""
function ccontractc(A::TensType,iA::intvecType,B::TensType,Z::TensType...;alpha::Number=eltype(A)(1),beta::Number=eltype(A)(1))
  mA,mB = checkType(A,B)
  iB = ntuple(w->w,ndims(B))
  return maincontractor(true,true,mA,convIn(iA),mB,iB,Z...,alpha=alpha,beta=beta)
end

"""
  ccontractc(A,B,iB[,Z,alpha=,beta=])

same as `contract(A,iA,B,iB)` but `iB` is over all indices.

See also: [`ccontract`](@ref) [`contractc`](@ref) [`ccontractc`](@ref)
"""
function ccontractc(A::TensType,B::TensType,iB::intvecType,Z::TensType...;alpha::Number=eltype(A)(1),beta::Number=eltype(A)(1))
  mA,mB = checkType(A,B)
  iA = ntuple(w->w,ndims(A))
  return maincontractor(true,true,mA,iA,mB,convIn(iB),Z...,alpha=alpha,beta=beta)
end

"""
    contract(order,A,iA,B,iB[,Z,alpha=,beta=])

Permutes result of [`contract`](@ref) to `order`
"""
function contract(order::intvecType,A::TensType,iA::intvecType,B::TensType,iB::intvecType,Z::TensType...;alpha::Number=eltype(A)(1),beta::Number=eltype(A)(1))
  mA,mB = checkType(A,B)
  newT = contract(mA,convIn(iA),mB,convIn(iB),Z...,alpha=alpha,beta=beta)
  return permutedims!(newT,convIn(order))
end

"""
    contract(order,A,iA,B,iB[,Z,alpha=,beta=])

Permutes result of [`contract`](@ref) to `order`
"""
function contract(order::intvecType,A::TensType,iA::intvecType,B::TensType,Z::TensType...;alpha::Number=eltype(A)(1),beta::Number=eltype(A)(1))
  mA,mB = checkType(A,B)
  newT = contract(mA,iA,mB,Z...,alpha=alpha,beta=beta)
  return permutedims!(newT,convIn(order))
end

"""
    contract(order,A,iA,B,iB[,Z,alpha=,beta=])

Permutes result of [`contract`](@ref) to `order`
"""
function contract(order::intvecType,A::TensType,B::TensType,iB::intvecType,Z::TensType...;alpha::Number=eltype(A)(1),beta::Number=eltype(A)(1))
  mA,mB = checkType(A,B)
  newT = contract(mA,mB,iB,Z...,alpha=alpha,beta=beta)
  return permutedims!(newT,convIn(order))
end

"""
    ccontract(order,A,iA,B,iB[,Z,alpha=,beta=])

Permutes result of [`ccontract`](@ref) to `order`
"""
function ccontract(order::intvecType,A::TensType,iA::intvecType,B::TensType,iB::intvecType,Z::TensType...;alpha::Number=eltype(A)(1),beta::Number=eltype(A)(1))
  mA,mB = checkType(A,B)
  newT = ccontract(mA,convIn(iA),mB,convIn(iB),Z...,alpha=alpha,beta=beta)
  return permutedims!(newT,convIn(order))
end

"""
    ccontract(order,A,iA,B,iB[,Z,alpha=,beta=])

Permutes result of [`contract`](@ref) to `order`
"""
function ccontract(order::intvecType,A::TensType,iA::intvecType,B::TensType,Z::TensType...;alpha::Number=eltype(A)(1),beta::Number=eltype(A)(1))
  mA,mB = checkType(A,B)
  newT = contract(mA,convIn(iA),mB,Z...,alpha=alpha,beta=beta)
  return permutedims!(newT,convIn(order))
end

"""
    ccontract(order,A,iA,B,iB[,Z,alpha=,beta=])

Permutes result of [`contract`](@ref) to `order`
"""
function ccontract(order::intvecType,A::TensType,B::TensType,iB::intvecType,Z::TensType...;alpha::Number=eltype(A)(1),beta::Number=eltype(A)(1))
  mA,mB = checkType(A,B)
  newT = contract(mA,mB,convIn(iB),Z...,alpha=alpha,beta=beta)
  return permutedims!(newT,convIn(order))
end

"""
    contractc(order,A,iA,B,iB[,Z,alpha=,beta=])

Permutes result of [`contractc`](@ref) to `order`
"""
function contractc(order::intvecType,A::TensType,iA::intvecType,B::TensType,iB::intvecType,Z::TensType...;alpha::Number=eltype(A)(1),beta::Number=eltype(A)(1))
  mA,mB = checkType(A,B)
  newT = contractc(mA,convIn(iA),mB,convIn(iB),Z...,alpha=alpha,beta=beta)
  return permutedims!(newT,convIn(order))
end

"""
    contractc(order,A,iA,B,iB[,Z,alpha=,beta=])

Permutes result of [`contract`](@ref) to `order`
"""
function contractc(order::intvecType,A::TensType,iA::intvecType,B::TensType,Z::TensType...;alpha::Number=eltype(A)(1),beta::Number=eltype(A)(1))
  mA,mB = checkType(A,B)
  newT = contract(mA,iA,mB,Z...,alpha=alpha,beta=beta)
  return permutedims!(newT,convIn(order))
end

"""
    contractc(order,A,iA,B,iB[,Z,alpha=,beta=])

Permutes result of [`contract`](@ref) to `order`
"""
function contractc(order::intvecType,A::TensType,B::TensType,iB::intvecType,Z::TensType...;alpha::Number=eltype(A)(1),beta::Number=eltype(A)(1))
  mA,mB = checkType(A,B)
  newT = contract(mA,mB,iB,Z...,alpha=alpha,beta=beta)
  return permutedims!(newT,convIn(order))
end

"""
    ccontractc(order,A,iA,B,iB[,Z,alpha=,beta=])

Permutes result of [`ccontractc`](@ref) to `order`
"""
function ccontractc(order::intvecType,A::TensType,iA::intvecType,B::TensType,iB::intvecType,Z::TensType...;alpha::Number=eltype(A)(1),beta::Number=eltype(A)(1))
  mA,mB = checkType(A,B)
  newT = ccontractc(mA,iA,mB,iB,Z...,alpha=alpha,beta=beta)
  return permutedims!(newT,convIn(order))
end

"""
    ccontractc(order,A,iA,B,iB[,Z,alpha=,beta=])

Permutes result of [`contract`](@ref) to `order`
"""
function ccontractc(order::intvecType,A::TensType,iA::intvecType,B::TensType,Z::TensType...;alpha::Number=eltype(A)(1),beta::Number=eltype(A)(1))
  mA,mB = checkType(A,B)
  newT = ccontractc(mA,convIn(iA),mB,Z...,alpha=alpha,beta=beta)
  return permutedims!(newT,convIn(order))
end

"""
    ccontractc(order,A,iA,B,iB[,Z,alpha=,beta=])

Permutes result of [`contract`](@ref) to `order`
"""
function ccontractc(order::intvecType,A::TensType,B::TensType,iB::intvecType,Z::TensType...;alpha::Number=eltype(A)(1),beta::Number=eltype(A)(1))
  mA,mB = checkType(A,B)
  newT = ccontractc(mA,mB,convIn(iB),Z...,alpha=alpha,beta=beta)
  return permutedims!(newT,convIn(order))
end

"""
    trace!(A,iA)

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
function trace!(A::TensType,iA::Array{NTuple{2,P},1}) where P <: Integer
  Id = makeId(A,iA)
  conA = (iA[1]...,)
  for w = 2:length(iA)
    conA = (conA...,iA[w]...)
  end

  conId = ntuple(w->w,2*length(iA))

#  permutedims!(A,conA)
  return contract(A,conA,Id,conId)
end

function trace!(A::TensType,iA::Array{Array{P,1},1}) where P <: Integer
  return trace!(A,[convIn(iA[w]) for w = 1:length(iA)])
end

function trace!(A::TensType,iA::Array{P,1}...) where P <: Integer
  return trace!(A,[convIn(iA[w]) for w = 1:length(iA)])
end

"""
    trace!(A)

Find trace of a matrix `A`
"""
function trace!(A::TensType)
  return sum(w->searchindex(A,w,w),1:size(A,1))
end
export trace!

"""
    trace(A,iA)

Computes trace of `A` (copying `A`) over indices `iA` by first making an identity tensor among those indices; `iA` is a vector of numbers (ex: [1,3]) or a vector of 2-element vectors ([[1,2],[3,4],[5,6]])

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
function trace(A::TensType,iA::R...) where R <: Union{Array{P,1},Array{Array{P,1},1},Array{NTuple{2,P},1}} where P <: Integer
  return trace!(copy(A),iA...)
end
export trace


#       +------------------------+
#>------| Quantum number version |---------<
#       +------------------------+

function checkblocks(Aqind::intType,Bqind::intType,A::Qtens{W,Q},B::Qtens{R,Q},inputA::Union{LinearAlgebra.Diagonal{W,Array{W,1}},Array{W,2}},inputB::Union{LinearAlgebra.Diagonal{R,Array{R,1}},Array{R,2}};Aind::intType=2,Bind::intType=1) where {W <: Number, R <: Number, Q <: Qnum}
  checksum = 0
  minAB = min(length(A.ind[Aqind][Aind]),length(B.ind[Bqind][Bind]))
  w = 0
  @inbounds while w < minAB && checksum == 0
    w += 1
    checksum += A.ind[Aqind][Aind][w]-B.ind[Bqind][Bind][w]
  end

  mulblockA = inputA
  mulblockB = inputB

  if !(length(A.ind[Aqind][Aind]) == length(B.ind[Bqind][Bind]) && checksum == 0)

    blocksizes = ntuple(n->length(A.QnumMat[A.currblock[Aind][n]]),length(A.currblock[Aind]))

    indsA = A.ind[Aqind][Aind]
    indsB = B.ind[Bqind][Bind]

    Lrowcols = Array{intType,1}(undef,size(indsA,2))
    Rrowcols = Array{intType,1}(undef,size(indsB,2))

    for p = 1:2
      if p == 1
        G = Lrowcols
        K = indsA
      else
        G = Rrowcols
        K = indsB
      end
      for x = 1:length(G)
        z = K[end,x]
        @inbounds @simd for y = length(blocksizes)-1:-1:1
          z *= blocksizes[y]
          z += K[y,x]
        end
        G[x] = z+1
      end
    end

    equalinds = length(Lrowcols) == length(Rrowcols)
    if equalinds
      k = 0
      while equalinds && w < length(Lrowcols)
        k += 1
        equalinds = Lrowcols[k] == Rrowcols[k]
      end
    end
    if !equalinds
      commoninds = intersect(Lrowcols,Rrowcols)
      if !issorted(commoninds)
        sort!(commoninds)
      end
      orderL = sortperm(Lrowcols)
      orderR = sortperm(Rrowcols)

      keepL = Array{intType,1}(undef,length(commoninds))
      keepR = Array{intType,1}(undef,length(commoninds))

      for p = 1:length(commoninds)
        b = 1
        @inbounds while b < length(orderR) && Lrowcols[orderL[p]] != Rrowcols[orderR[b]]
          b += 1
        end
        keepL[p] = orderL[p]
        keepR[p] = orderR[b]
      end

      mulblockA = mulblockA[:,keepL]
      mulblockB = mulblockB[keepR,:]
    end
  end
  return mulblockA,mulblockB
end

@inline function genblockinds(offset::intType,firstblock::Array{intType,1})
  @inbounds @simd for w = 1:length(firstblock)
    firstblock[w] = offset + w
  end
  nothing
end

@inline function loadnewsize(newsize::Array{Array{intType,1},1})
  counter = 0
  for w = 1:length(newsize)
    @inbounds @simd for a = 1:length(newsize[w])
      counter += 1
      newsize[w][a] = counter
    end
  end
  nothing
end

@inline function loadarraynewsize(newsize::Array{Array{intType,1},1},offset::intType,notconA::Array{intType,1},QtensA::Qtens{W,Q}) where {W <: Number, Q <: Qnum}
  @inbounds for w = 1:length(notconA)
    newsize[w+offset] = Array{intType,1}(undef,length(QtensA.size[notconA[w]]))
  end
  nothing
end

@inline function loadnewQnumMat(newQnumMat::Array{Array{intType,1},1},offset::Integer,notconA::Array{intType,1},A::Qtens{W,Q}) where {W <: Number, Q <: Qnum}
  @inbounds @simd for q = 1:length(notconA)
    newQnumMat[q+offset] = A.QnumMat[notconA[q]]
  end
  nothing
end

@inline function loadnewQnumSum(newQnumMat::Array{Array{Q,1},1},offset::Integer,notconA::Array{intType,1},A::Qtens{W,Q}) where {W <: Number, Q <: Qnum}
  @inbounds @simd for q = 1:length(notconA)
    newQnumMat[q+offset] = A.QnumSum[notconA[q]]
  end
  nothing
end

@inline function loadnewQnumSum_inv(newQnumMat::Array{Array{Q,1},1},offset::Integer,notconA::Array{intType,1},A::Qtens{W,Q}) where {W <: Number, Q <: Qnum}
  @inbounds @simd for q = 1:length(notconA)
    newQnumMat[q+offset] = inv.(A.QnumSum[notconA[q]])
  end
  nothing
end

function permq(A::Qtens{W,Q},iA::Array{intType,1}) where {W <: Number, Q <: Qnum} #K
  nopermL = length(iA) == length(A.currblock[1]) && issorted(A.currblock[1])
  w = 0
  @inbounds while nopermL && w < length(iA)
    w += 1
    nopermL = w == A.currblock[1][w] && w == iA[w]
  end

  nopermR = !nopermL && length(iA) == length(A.currblock[2]) && issorted(A.currblock[2])
  #println(nopermR)
  if nopermR
    w =length(iA)
    end_dim = length(A.QnumMat)
    @inbounds while nopermR && w > 0
      nopermR = A.currblock[2][w] == end_dim && iA[w] == end_dim
      end_dim -= 1
      w -= 1
    end
  end
  return nopermL,nopermR
end

function dmul!(X::Qtens{R,Q},Y::Qtens{W,Q}) where {R <: Number, W <: Number, Q <: Qnum}
  #assert diagonal types in one of hte matrices here
  return maincontractor(false,false,X,(ndims(X),),Y,(1,),inplace=true)
end

function dmul!(X::Qtens{R,Q},Y::Qtens{W,Q}) where {R <: Number, W <: Number, Q <: Qnum}
  return maincontractor(false,false,X,(ndims(X),),Y,(1,),inplace=true)
end

function maincontractor(conjA::Bool,conjB::Bool,QtensA::Qtens{W,Q},vecA::Tuple,QtensB::Qtens{R,Q},vecB::Tuple,Z::Qtens{S,Q}...;alpha::Number=W(1),beta::Number=W(1),inplace::Bool=false) where {W <: Number, R <: Number, S <: Number, Q <: Qnum}
  if QtensA === QtensB && ((conjA && W <: Complex) || (conjB && R <: Complex))
    QtensB = copy(QtensA)
  end

  conA,notconA = getinds(QtensA,vecA)
  conB,notconB = getinds(QtensB,vecB)




  AnopermL,AnopermR = permq(QtensA,conA)
  BnopermL,BnopermR = permq(QtensB,conB)

  Aperm,transA = willperm(conjA,eltype(QtensA),AnopermL,AnopermR)
  Bperm,transB = willperm(conjB,eltype(QtensB),BnopermR,BnopermL)

  if Aperm
    A = QtensA
  else
    A = changeblock(QtensA,notconA,conA)
    transA = 'N'
  end
  if Bperm
    B = QtensB
  else
    B = changeblock(QtensB,conB,notconB)
    transB = 'N'
  end

  Aretind,notAretind = transA == 'N' ? (2,1) : (1,2)
  Bretind,notBretind = transB == 'N' ? (1,2) : (2,1)
  commonblocks = matchblocks((conjA,conjB),A,B,ind=(Aretind,Bretind))


  outType = W == R ? W : typeof(W(1) * R(1))

  usealpha = !isapprox(alpha,1)

  numQNs = length(commonblocks)

  useZ = length(Z) > 0
  if useZ
    Zone = [i for i = 1:length(notconA)]
    Ztwo = [i + length(notconA) for i = 1:length(notconB)]
    Z = changeblock(Zed,Zone,Ztwo)
    Zcommonblocks = matchblocks((conjA,false),A,Zed,ind=(2,1))
    type_beta = eltype(beta) == outType && !isapprox(beta,1) ? beta : convert(outType,beta)
    type_alpha = typeof(alpha) == outType ? alpha : convert(outType,alpha)
  elseif usealpha
    type_alpha = typeof(alpha) == outType ? alpha : convert(outType,alpha)
  end


  ############

  newrowcols = Array{NTuple{2,Array{intType,2}},1}(undef,numQNs)
  newQblocksum = Array{NTuple{2,Q},1}(undef,numQNs)
  outTens = Array{Array{outType,2},1}(undef,numQNs)

  #optional types like alpha input into functions from upstream calls create perhaps a type instability? Disabling alpha saves one allocation

  @inbounds for q = 1:numQNs
    Aqind = commonblocks[q][1]
    Bqind = commonblocks[q][2]

    if conjA && W <: Complex && transA == 'N'
      inputA = conj(A.T[Aqind])
    else
      inputA = A.T[Aqind]
    end

    if conjB && R <: Complex && transB == 'N'
      inputB = conj(B.T[Bqind])
    else
      inputB = B.T[Bqind]
    end

    mulblockA,mulblockB = checkblocks(Aqind,Bqind,A,B,inputA,inputB,Aind=Aretind,Bind=Bretind)

    Lsize,innersizeL = size(mulblockA,notAretind),size(mulblockA,Aretind)
    Rsize,innersizeR = size(mulblockB,notBretind),size(mulblockB,Bretind)

    Adiag = typeof(mulblockA) <: LinearAlgebra.Diagonal
    if Adiag || typeof(mulblockB) <: LinearAlgebra.Diagonal
#      println("IN HERE? ",transA," ",transB)
      if transA == 'N' && transB == 'N'
        if inplace
          outTens[q] = dmul!(mulblockA,mulblockB)
        else
          outTens[q] = mulblockA*mulblockB
        end
      else
        if Adiag
          if eltype(mulblockA) <: Complex
            mulblockA = conj(mulblockA)
          end
        else
          if eltype(mulblockB) <: Complex
            mulblockB = conj(mulblockB)
          end
        end
        if inplace
          outTens[q] = dmul!(mulblockB,mulblockA)
        else
          outTens[q] = mulblockB*mulblockA
        end
        outTens[q] = transpose(outTens[q])
      end
    else
      if useZ
        Zqind = Zcommonblocks[q][2]
        outTens[q] = libmult(transA,transB,type_alpha,mulblockA,mulblockB,type_beta,inputZed,Lsize,innersizeL,innersizeR,Rsize)
      elseif usealpha
        outTens[q] = libmult(transA,transB,type_alpha,mulblockA,mulblockB,Lsize,innersizeL,innersizeR,Rsize)
      else
        outTens[q] = libmult(transA,transB,mulblockA,mulblockB,Lsize,innersizeL,innersizeR,Rsize)
      end
    end

    LQN = conjA ? inv(A.Qblocksum[Aqind][notAretind]) : A.Qblocksum[Aqind][notAretind]
    RQN = conjB ? inv(B.Qblocksum[Bqind][notBretind]) : B.Qblocksum[Bqind][notBretind]
    newQblocksum[q] = (LQN,RQN)

    newrowcols[q] = (A.ind[Aqind][notAretind],B.ind[Bqind][notBretind])
  end
  ############

  newQnumMat = Array{Array{intType,1},1}(undef,length(notconA)+length(notconB))
  loadnewQnumMat(newQnumMat,0,notconA,A)
  loadnewQnumMat(newQnumMat,length(notconA),notconB,B)

  newQnumSum = Array{Array{Q,1},1}(undef,length(notconA)+length(notconB))
  if conjA
    loadnewQnumSum_inv(newQnumSum,0,notconA,A)
  else
    loadnewQnumSum(newQnumSum,0,notconA,A)
  end
  if conjB
    loadnewQnumSum_inv(newQnumSum,length(notconA),notconB,B)
  else
    loadnewQnumSum(newQnumSum,length(notconA),notconB,B)
  end

  newsize = Array{Array{intType,1},1}(undef,length(notconA)+length(notconB))
  loadarraynewsize(newsize,0,notconA,QtensA)
  loadarraynewsize(newsize,length(notconA),notconB,QtensB)

  loadnewsize(newsize)

  keepers = Bool[size(outTens[q],1) > 0 && size(outTens[q],2) > 0 for q = 1:length(outTens)]

  if !conjA && !conjB
    newflux = QtensA.flux + QtensB.flux
  elseif conjA && !conjB
    newflux = QtensB.flux - QtensA.flux
  elseif !conjA && conjB
    newflux = QtensA.flux - QtensB.flux
  elseif conjA && conjB
    newflux = -(QtensA.flux,QtensB.flux)
  end
  firstblock = Array{intType,1}(undef,length(notconA))
  genblockinds(0,firstblock)

  secondblock = Array{intType,1}(undef,length(notconB))
  genblockinds(length(notconA),secondblock)
  newcurrblocks = (firstblock,secondblock)

  if sum(keepers) < length(keepers)
    newT = outTens[keepers]
    newinds = newrowcols[keepers]
    newQblocks = newQblocksum[keepers]
  else
    newT = outTens
    newinds = newrowcols
    newQblocks = newQblocksum
  end
  return Qtens{outType,Q}(newsize,newT,newinds,newcurrblocks,newQblocks,newQnumMat,newQnumSum,newflux)
end


#         +-----------------+
#>--------|  Check contract |------<
#         +-----------------+


function checkcontract(A::TensType,iA::intvecType,B::TensType,iB::intvecType,Z::TensType...;alpha::Number=eltype(A)(1),beta::Number=eltype(A)(1))
  mA,mB = checkType(A,B)

  iA = convIn(iA)
  iB = convIn(iB)

  if size(mA)[[iA...]] == size(mB)[[iB...]]
    println("contracting over indices with equal sizes")
  else
    error("some indices in A or B are not equal size; A->",size(mA)[[iA...]],", B->",size(mB)[[iB...]])
  end
  if typeof(mA) <: qarray
    println("checking flux left:")
    checkflux(mA)
    println("checking flux right:")
    checkflux(mB)
    for a = 1:length(iA)
      AQNs = recoverQNs(iA[a],A)
      BQNs = recoverQNs(iB[a],B)
      println("contracted index $a (A's index: ",iA[a],", B's index: ",iB[a],")")
      if length(AQNs) == length(BQNs)
        for w = 1:length(AQNs)
          if AQNs[w] != inv(BQNs[w])
            error("non-matching quantum numbers on indices on index ",iA[a]," of A and index ",iB[a]," of B: values ")
            println(A.QnumSum[iA[a]])
            println(B.QnumSum[iB[a]])
          end
        end
      else
        error("unmatching quantum number vector lengths")
      end
      println("matching quantum numbers on both indices")
      println("FULL PASS")
    end
  end
  nothing
end
export checkcontract
#end
 