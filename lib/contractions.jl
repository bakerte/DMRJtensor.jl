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
"""
  C = libmult([alpha,]C,D[,beta,Z])

Chooses the best matrix multiply function for tensor contraction (dense tensor or sub-block) withoutput matrix `C`

+ Outputs `C` * `D` if `alpha`=1
+ Outputs `alpha` * `C` * `D` if `Z` is not input
+ Outputs `alpha` * `C` * `D` + `beta` * `Z` if `Z` is input
"""
@inline function libmult(C::Array{W,2},D::Array{X,2},alpha::Y) where {W <: Number, X <: Number, P <: Number, Y <: Number, Z <: Number}
  if (W == X)
    temp_alpha = Y == W ?  alpha : convert(W,alpha)
    retMat = LinearAlgebra.BLAS.gemm('N','N',temp_alpha,C,D)
  else
    retMat = alpha * C * D
  end
  return retMat
end

@inline function libmult(C::Array{W,2},D::Array{X,2}) where {W <: Number, X <: Number, P <: Number, Y <: Number, Z <: Number}
  if (W == X)
    retMat = LinearAlgebra.BLAS.gemm('N','N',C,D)
  else
    retMat = C * D
  end
  return retMat
end

@inline function libmult(C::Array{W,2},D::Array{X,2},alpha::Y,beta::Z,tempZ::Array{P,2}) where {W <: Number, X <: Number, P <: Number, Y <: Number, Z <: Number}
  if (W == X == P)
    temp_alpha = Y == P ?  alpha : convert(P,alpha)
    temp_beta = Z == P ?  beta : convert(P,beta)
    LinearAlgebra.BLAS.gemm!('N','N',temp_alpha,C,D,temp_beta,tempZ)
    retMat = tempZ
  else
    retMat = alpha * C * D + beta * tempZ[1]
  end
  return retMat
end
export libmult

#       +------------------------+
#>------|    Matrix multiply     |---------<
#       +------------------------+


import LinearAlgebra.BLAS: BlasReal, BlasComplex, BlasFloat, BlasInt, DimensionMismatch, checksquare, axpy!, @blasfunc, libblastrampoline

function libmult! end

for (gemm, elty) in
        ((:dgemm_,:Float64),
         (:sgemm_,:Float32),
         (:zgemm_,:ComplexF64),
         (:cgemm_,:ComplexF32))
    @eval begin
             # SUBROUTINE DGEMM(TRANSA,TRANSB,M,N,K,ALPHA,A,LDA,B,LDB,BETA,C,LDC)
             # *     .. Scalar Arguments ..
             #       DOUBLE PRECISION ALPHA,BETA
             #       INTEGER K,LDA,LDB,LDC,M,N
             #       CHARACTER TRANSA,TRANSB
             # *     .. Array Arguments ..
             #       DOUBLE PRECISION A(LDA,*),B(LDB,*),C(LDC,*)
    function libmult!(transA::AbstractChar, transB::AbstractChar,
                  alpha::Union{($elty), Bool},
                  A::AbstractArray{$elty,N},
                  B::AbstractArray{$elty,M},
                  beta::Union{($elty), Bool},
                  C::AbstractArray{$elty,G},
                  m::Integer,ka::Integer,kb::Integer,n::Integer) where {N,M,G}
        lda = max(1,transA == 'N' ? m : ka)
        ldb = max(1,transB == 'N' ? ka : n)
        ldc = max(1,m)
        ccall((@blasfunc($gemm), libblastrampoline), Cvoid,
            (Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt},
            Ref{BlasInt}, Ref{$elty}, Ptr{$elty}, Ref{BlasInt},
            Ptr{$elty}, Ref{BlasInt}, Ref{$elty}, Ptr{$elty},
            Ref{BlasInt}, Clong, Clong),
            transA, transB, m, n,
            ka, alpha, A,  lda,
            B, ldb, beta, C,
            ldc, 1, 1)
        C
    end
    function libmult(transA::AbstractChar, transB::AbstractChar, alpha::($elty), A::AbstractArray{$elty,N},B::AbstractArray{$elty,M},m::Integer,ka::Integer,kb::Integer,n::Integer) where {N,M}
        C = Array{($elty),2}(undef,m,n)
        libmult!(transA, transB, alpha, A, B, zero($elty), C,m,ka,kb,n)
        return C
    end
    function libmult(transA::AbstractChar, transB::AbstractChar, A::Union{AbstractArray{$elty,N},tens{$elty}},B::Union{AbstractArray{$elty,M},tens{$elty}},m::Integer,ka::Integer,kb::Integer,n::Integer) where {N,M}
        libmult(transA, transB, one($elty), A, B,m,ka,kb,n)
    end

    function libmult(transA::AbstractChar, transB::AbstractChar, alpha::($elty), A::tens{$elty},B::tens{$elty},m::Integer,ka::Integer,kb::Integer,n::Integer) where {N,M}
      C = Array{($elty),1}(undef,m*n)
      libmult!(transA, transB, alpha, A.T, B.T, zero($elty), C,m,ka,kb,n)
      return C
    end
    end
end

"""
    libmult(tA, tB, alpha, A, B)
Return `alpha*A*B` or the other three variants according to [`tA`](@ref stdlib-blas-trans) and `tB`.
"""
libmult(tA, tB, alpha, A, B)

"""
    libmult(tA, tB, A, B)
Return `A*B` or the other three variants according to [`tA`](@ref stdlib-blas-trans) and `tB`.
"""
libmult(tA, tB, A, B)


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

function maincontractor(conjA::Bool,conjB::Bool,A::TensType,iA::intvecType,B::TensType,iB::intvecType,Z::TensType...;alpha::Number=1,beta::Number=1)
  AnopermL,AnopermR = permq(A,iA)
  BnopermL,BnopermR = permq(B,iB)

  Aperm,transA = willperm(conjA,eltype(A),AnopermL,AnopermR)
  Bperm,transB = willperm(conjB,eltype(B),BnopermR,BnopermL)

  Aremain = ndims(A)-length(iA)
  Bremain = ndims(B)-length(iB)
  AAsizes = Array{intType,1}(undef,Aremain+Bremain)
  Lsize,innersizeL = getsizes(A,iA,AAsizes,0)
  Rsize,innersizeR = getsizes(B,iB,AAsizes,Aremain)

  m,n,ka,kb = Lsize,Rsize,innersizeL,innersizeR
  if ka != kb #|| m != size(C,1) || n != size(C,2)
    throw(DimensionMismatch("A has size ($m,$ka), B has size ($kb,$n), C has size ($m,$n)"))
  end

  if Aperm && Bperm
    newA = conjA && eltype(A) <: Complex && !AnopermL ? conj(A) : A
    newB = conjB && eltype(B) <: Complex && !BnopermR ? conj(B) : B
  elseif Aperm
    newA = conjA && eltype(A) <: Complex && !AnopermL ? conj(A) : A
    notvecB = findnotcons(ndims(B),iB)
    newB = prepareT(B,iB,notvecB,conjB)
  elseif Bperm
    notvecA = findnotcons(ndims(A),iA)
    newA = prepareT(A,notvecA,iA,conjA)
    newB = conjB && eltype(B) <: Complex && !BnopermR ? conj(B) : B
  else
    notvecA = findnotcons(ndims(A),iA)
    newA = prepareT(A,notvecA,iA,conjA)
    notvecB = findnotcons(ndims(B),iB)
    newB = prepareT(B,iB,notvecB,conjB)
  end

  mulA = typeof(newA) <: denstens ? newA.T : newA
  mulB = typeof(newB) <: denstens ? newB.T : newB

  if length(Z) > 0
    outType = typeof(eltype(A)(1)*eltype(B)(1)*eltype(Z)(1)*typeof(alpha)(1)*typeof(beta)(1))
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
    outTens = tens(out)
    outTens.size = AAsizes
  else
    outTens = reshape(out,AAsizes)
  end
  return outTens
end

#       +----------------------------+
#>------| Contraction function calls |---------<
#       +----------------------------+


"""


takes `identity` or `adjoint` (or equivalently `conj`) for the `Afct` and `Bfct`

See also: [`identity`](@ref) [`adjoint`](@ref) [`conj`](@ref)
"""
function dot(inA::densTensType,inB::densTensType;Afct::Function=adjoint,Bfct::Function=identity,transA::Bool=true)
  A = typeof(inA) <: denstens ? inA.T : inA
  B = typeof(inB) <: denstens ? inB.T : inB

  newtypeAB = typeof(eltype(A)(1) * eltype(B)(1))
  val = newtypeAB(0)
  dim1 = length(A) #size(inA,transA ? 1 : 2)
  @inbounds @simd for j = 1:dim1
    val += Afct(A[j]) * Bfct(B[j])
  end
  return val
end

function dot(C::Qtens{W,Q},D::Qtens{R,Q};Afct::Function=adjoint,Bfct::Function=identity) where {W <: Number, R <: Number, Q <: Qnum}
  newtype = typeof(W(1)*R(1))

  A = changeblock(C,intType[],intType[i for i = 1:length(C.QnumMat)])
  B = changeblock(D,intType[i for i = 1:length(D.QnumMat)],intType[])
  conjA = Afct != identity
  conjB = Bfct != identity
  commonblocks = matchblocks((conjA,conjB),A,B,ind=(2,1))

  val = newtype(0)
  @inbounds for q = 1:length(commonblocks)
    Aqind = commonblocks[q][1]
    Bqind = commonblocks[q][2]
    val += dot(A.T[Aqind],B.T[Bqind],Afct=Afct,Bfct=Bfct)
  end
  return val
end

"""
  

The function will admit any dimension or input element type in terms of the arrays `A`, `H`, and `B`. However, the function will only work properly when the total elements of `A` times those in `B` equal the elements in `H`

If more operators `H` should be contracted between `A` and `B`, then it is advised here to contract them first before using this function

"""
function dot(inA::densTensType,inH::densTensType,inB::densTensType;Afct::Function=adjoint,Bfct::Function=identity,transA::Bool=true)

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
      ival += Afct(A[i]) * H[i + savedim]
    end
    val += ival * Bfct(B[j])
  end
  return val
end
#=
function dot(A::denstens,H::denstens,B::denstens;Afct::Function=adjoint,Bfct::Function=identity)
  return dot(A.T,H.T,B.T,Afct=Afct,Bfct=Bfct)
end
=#
#not sure how to assign matrix blocks...or how to get information from that to A and B vectors
#must be set up correctly with changeblock
function dot(A::Qtens{W,Q},H::Qtens{Y,Q},B::Qtens{R,Q};Afct::Function=adjoint,Bfct::Function=identity) where {W <: Number, Y <: Number, R <: Number, Q <: Qnum}
#  conjA = Afct != identity
#  conjB = Bfct != identity
#  ival = maincontractor(C,G,)
#  A = changeblock(C,intType[],intType[i for i = 1:length(C.QnumMat)])
#  B = changeblock(D,intType[i for i = 1:length(D.QnumMat)],intType[])
#  H = changeblock(G,G.currblock[1],G.currblock[2])
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
      val += dot(A.T[Aqind],H.T[Hqind],B.T[Bqind],Afct=Afct,Bfct=Bfct)
    end
  end
  return val
end
export dot



function *(X::TensType,Y::TensType)
  return contract(X,2,Y,1)
end



"""
  C = contract(A,B[,alpha=])

Contracts to (alpha * A * B and returns a scalar output...if only `A` is specified, then the norm is evaluated

See also: [`ccontract`](@ref) [`contractc`](@ref) [`ccontractc`](@ref)
"""
function contract(A::TensType,B::TensType;alpha::Number=eltype(A)(1))
  mA,mB = checkType(A,B)
  return alpha * dot(mA,mB,Afct=identity,Bfct=identity)
end

"""
  ccontract(A,B[,alpha=])

Similar to contract but 'A' is conjugated

See also: ['ccontract'](@ref) ['contractc'](@ref) ['ccontractc'](@ref)
"""
function ccontract(A::TensType,B::TensType;alpha::Number=eltype(A)(1))
  mA,mB = checkType(A,B)
  return alpha * dot(mA,mB,Afct=adjoint,Bfct=identity)
end

"""
  contractc(A,B[,alpha=])

Similar to contract but 'B' is conjugated

See also: ['ccontract'](@ref) ['contractc'](@ref) ['ccontractc'](@ref)
"""
function contractc(A::TensType,B::TensType;alpha::Number=eltype(A)(1))
  mA,mB = checkType(A,B)
  return alpha * dot(mA,mB,Afct=identity,Bfct=adjoint)
end

"""
  ccontractc(A,B[,alpha=])

Similar to contract but 'A' and 'B' are conjugated

See also: ['ccontract'](@ref) ['contractc'](@ref) ['ccontractc'](@ref)
"""
function ccontractc(A::TensType,B::TensType;alpha::Number=eltype(A)(1))
  mA,mB = checkType(A,B)
  return alpha * dot(mA,mB,Afct=adjoint,Bfct=adjoint)
end

"""
    contract(A[,alpha=])

Self-contraction of input tensor `A` to a scalar value of `A`.`A`

See also: ['ccontract'](@ref) ['contractc'](@ref) ['ccontractc'](@ref)
"""
function contract(A::TensType;alpha::Number=eltype(A)(1))
  return alpha * dot(A,A,Afct=identity,Bfct=identity)
end

"""
    ccontract(A[,alpha=])

Self-contraction of input tensor `A` to a scalar value of `A`.`A` with one of the inputs conjugated

See also: ['ccontract'](@ref) ['contractc'](@ref) ['ccontractc'](@ref)
"""
function ccontract(A::TensType;alpha::Number=eltype(A)(1))
  return alpha * dot(A,A,Afct=adjoint,Bfct=identity)
end

"""
    contractc(A[,alpha=])

Self-contraction of input tensor `A` to a scalar value of `A`.`A` with one of the inputs conjugated

See also: ['ccontract'](@ref) ['contractc'](@ref) ['ccontractc'](@ref)
"""
function contractc(A::TensType;alpha::Number=eltype(A)(1))
  return alpha * dot(A,A,Afct=identity,Bfct=adjoint)
end

"""
    ccontractc(A[,alpha=])

Self-contraction of input tensor `A` to a scalar value of `A`.`A` with both inputs conjugated

See also: ['ccontract'](@ref) ['contractc'](@ref) ['contract'](@ref)
"""
function ccontractc(A::TensType;alpha::Number=eltype(A)(1))
  return alpha * dot(A,A,Afct=adjoint,Bfct=adjoint)
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
    ccontract(A,iA,B,iB[,Z,alpha=,beta=])

Similar to contract but 'A' is conjugated

See also: ['ccontract'](@ref) ['contractc'](@ref) ['ccontractc'](@ref)
"""
function ccontract(A::TensType,iA::intvecType,B::TensType,iB::intvecType,Z::TensType...;alpha::Number=eltype(A)(1),beta::Number=eltype(A)(1))
  mA,mB = checkType(A,B)
  return maincontractor(true,false,mA,convIn(iA),mB,convIn(iB),Z...,alpha=alpha,beta=beta)
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
    contract(order,A,iA,B,iB[,Z,alpha=,beta=])

Permutes result of [`contract`](@ref) to `order`
"""
function contract(order::intvecType,A::TensType,iA::intvecType,B::TensType,iB::intvecType,Z::TensType...;alpha::Number=eltype(A)(1),beta::Number=eltype(A)(1))
  mA,mB = checkType(A,B)
  newT = contract(mA,iA,mB,iB,Z...,alpha=alpha,beta=beta)
  return permutedims!(newT,convIn(order))
end

"""
    ccontract(order,A,iA,B,iB[,Z,alpha=,beta=])

Permutes result of [`ccontract`](@ref) to `order`
"""
function ccontract(order::intvecType,A::TensType,iA::intvecType,B::TensType,iB::intvecType,Z::TensType...;alpha::Number=eltype(A)(1),beta::Number=eltype(A)(1))
  mA,mB = checkType(A,B)
  newT = ccontract(mA,iA,mB,iB,Z...,alpha=alpha,beta=beta)
  return permutedims!(newT,convIn(order))
end

"""
    contractc(order,A,iA,B,iB[,Z,alpha=,beta=])

Permutes result of [`contractc`](@ref) to `order`
"""
function contractc(order::intvecType,A::TensType,iA::intvecType,B::TensType,iB::intvecType,Z::TensType...;alpha::Number=eltype(A)(1),beta::Number=eltype(A)(1))
  mA,mB = checkType(A,B)
  newT = contractc(mA,iA,mB,iB,Z...,alpha=alpha,beta=beta)
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
function trace!(A::TensType,iA::Array{NTuple{2,P},1}) where P <: Integer
  Id = makeId(A,iA)
  conL = ntuple(w->iA[w][1],length(iA))
  conR = ntuple(w->iA[w][2],length(iA))
  conA = (conL...,conR...)
  permutedims!(A,conA)
  return contract(A,Id)
end

function trace!(A::TensType,iA::Array{Array{P,1},1}) where P <: Integer
  return trace!(A,[convIn(iA[w]) for w = 1:length(iA)])
end

function trace!(A::TensType,iA::Array{P,1}...) where P <: Integer
  return trace!(A,[convIn(iA[w]) for w = 1:length(iA)])
end

function trace!(A::TensType)
  return sum(w->searchindex(A,w,w),1:size(A,1))
end
export trace!

function trace(A::TensType,iA::R...) where R <: Union{Array{P,1},Array{Array{P,1},1},Array{NTuple{2,P},1}} where P <: Integer
  return trace!(A,iA...)
end
export trace




#       +------------------------+
#>------| Quantum number version |---------<
#       +------------------------+
#=
#=@inline=# function contractloopZ(outType::DataType,numQNs::Integer,conjA::Bool,conjB::Bool,commonblocks::Array{NTuple{2,intType},1},A::Qtens{W,Q},B::Qtens{R,Q},
                                    Zed::Qtens{S,Q};alpha::Number=eltype(A)(1),beta::Number=eltype(Z[1])(1)) where {W <: Number, R <: Number, S <: Number, Q <: Qnum}

  Zone = [i for i = 1:length(notconA)]
  Ztwo = [i + length(notconA) for i = 1:length(notconB)]
  Z = changeblock(Zed,Zone,Ztwo)
  Zcommonblocks = matchblocks((conjA,false),A,Zed,ind=(2,1))

  Ablocks = A.currblock[1]
  Bblocks = B.currblock[2]

  newrowcols = Array{NTuple{2,Array{intType,2}},1}(undef,numQNs)
  newQblocksum = Array{NTuple{2,Q},1}(undef,numQNs)
  outTens = Array{Array{outType,2},1}(undef,numQNs)

  @inbounds for q = 1:numQNs
    Aqind = commonblocks[q][1]
    Bqind = commonblocks[q][2]

    if conjA && !(eltype(A.T[Aqind]) <: Real)
      inputA = conj(A.T[Aqind])
    else
      inputA = A.T[Aqind]
    end

    if conjB && !(eltype(B.T[Bqind]) <: Real)
      inputB = conj(B.T[Bqind])
    else
      inputB = B.T[Bqind]
    end

    mulblockA,mulblockB = checkblocks(Aqind,Bqind,A,B,inputA,inputB)

    Zqind = Zcommonblocks[q][2]
    if Zqind != 0
      inputZed = Z.T[Zqind]
      outTens[q] = libmult(mulblockA,mulblockB,alpha,beta,inputZed)
    elseif alpha != W(1)
      outTens[q] = libmult(mulblockA,mulblockB,alpha)
    else
      outTens[q] = libmult(mulblockA,mulblockB)
    end

    newrowcols[q] = [A.ind[Aqind][1],B.ind[Bqind][2]]
    newQblocksum[q] = [A.Qblocksum[Aqind][1],B.Qblocksum[Bqind][2]]
  end
  return outTens,newrowcols,newQblocksum
end
=#
function checkblocks(Aqind::intType,Bqind::intType,A::Qtens{W,Q},B::Qtens{R,Q},inputA::Array{W,2},inputB::Array{R,2};Aind::intType=2,Bind::intType=1) where {W <: Number, R <: Number, Q <: Qnum}
  checksum = 0
  minAB = min(length(A.ind[Aqind][Aind]),length(B.ind[Bqind][Bind]))
  @inbounds @simd for w = 1:minAB
    checksum += A.ind[Aqind][Aind][w]-B.ind[Bqind][Bind][w]
  end
  if length(A.ind[Aqind][Aind]) == length(B.ind[Bqind][Bind]) && isapprox(checksum,0)
    mulblockA = inputA
    mulblockB = inputB
  else
    keepA = [false for w = 1:size(A.ind[Aqind][Aind],2)]
    Aperm = Array{intType,1}(undef,size(A.ind[Aqind][Aind],2))
    keepB = [false for w = 1:size(B.ind[Bqind][Bind],2)]
    Bperm = Array{intType,1}(undef,size(B.ind[Bqind][Bind],2))
    counter = 0
    @inbounds for a = 1:length(keepA)
      b = 0
      while b < length(keepB) && !keepA[a]
        b += 1
        verdict = true
        r = 0
        while r < length(A.currblock[2]) && !keepA[a] && verdict
          r += 1
          verdict = verdict && A.ind[Aqind][Aind][r,a] == B.ind[Bqind][Bind][r,b]
        end
        if verdict
          counter += 1
          Aperm[a] = counter
          keepA[a] = true
          Bperm[b] = counter
          keepB[b] = true
        end
      end
    end
#    else
#    end

    mulblockA = (inputA[:,keepA])[:,Aperm[keepA]]
    mulblockB = (inputB[keepB,:])[Bperm[keepB],:]
  end
  return mulblockA,mulblockB
end

#=
#=@inline=# function contractloop(outType::DataType,numQNs::Integer,conjA::Bool,conjB::Bool,commonblocks::Array{NTuple{2,intType},1},
                                  A::Qtens{W,Q},B::Qtens{R,Q},alpha::Number) where {W <: Number, R <: Number, S <: Number, Q <: Qnum}

  Ablocks = A.currblock[1]
  Bblocks = B.currblock[2]

  newrowcols = Array{NTuple{2,Array{intType,2}},1}(undef,numQNs)
  newQblocksum = Array{NTuple{2,Q},1}(undef,numQNs)
  outTens = Array{Array{outType,2},1}(undef,numQNs)

    #optional types like alpha input into functions from upstream calls create perhaps a type instability? Disabling alpha saves one allocation
    usealpha = alpha != W(1)

  for q = 1:numQNs
    Aqind = commonblocks[q][1]
    Bqind = commonblocks[q][2]

    if conjA && !(eltype(A.T[Aqind]) <: Real)
      inputA = conj(A.T[Aqind])
    else
      inputA = A.T[Aqind]
    end

    if conjB && !(eltype(B.T[Bqind]) <: Real)
      inputB = conj(B.T[Bqind])
    else
      inputB = B.T[Bqind]
    end


    mulblockA,mulblockB = checkblocks(Aqind,Bqind,A,B,inputA,inputB)


  if usealpha
    outTens[q] = libmult(mulblockA,mulblockB,alpha)
  else
    outTens[q] = libmult(mulblockA,mulblockB)
  end

    newrowcols[q] = (A.ind[Aqind][1],B.ind[Bqind][2])
    if conjA
      LQN = inv(A.Qblocksum[Aqind][1])
    else
      LQN = A.Qblocksum[Aqind][1]
    end
    if conjB
      RQN = inv(B.Qblocksum[Bqind][2])
    else
      RQN = B.Qblocksum[Bqind][2]
    end
    newQblocksum[q] = (LQN,RQN)
  end
  return outTens,newrowcols,newQblocksum
end
=#

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

function permq(A::qarray,iA::Union{Array{intType,1},NTuple{K,intType}}) where K
  nopermL = length(iA) == length(A.currblock[1]) && issorted(A.currblock[1])
  w = 0
  while nopermL && w < length(iA)
    w += 1
    nopermL = w == A.currblock[1][w] && w == iA[w]
  end
  nopermR = !nopermL && length(iA) == length(A.currblock[2]) && issorted(A.currblock[2])
  if nopermR
    w =length(iA)
    end_dim = length(A.QnumMat)
    while nopermR && w > 0
      nopermR = A.currblock[2][w] == end_dim && iA[w] == end_dim
      end_dim -= 1
      w -= 1
    end
  end
  return nopermL,nopermR
end

function maincontractor(conjA::Bool,conjB::Bool,QtensA::Qtens{W,Q},vecA::Tuple,QtensB::Qtens{R,Q},vecB::Tuple,Z::Qtens{S,Q}...;alpha::Number=W(1),beta::Number=W(1)) where {W <: Number, R <: Number, S <: Number, Q <: Qnum}
  if QtensA === QtensB && ((conjA && W <: Complex) || (conjB && R <: Complex))
    QtensB = copy(QtensA)
  end

  conA,notconA = getinds(QtensA,vecA)
  conB,notconB = getinds(QtensB,vecB)




  AnopermL,AnopermR = permq(QtensA,conA)
  BnopermL,BnopermR = permq(QtensB,conB)

  Aperm,transA = willperm(conjA,eltype(QtensA),AnopermL,AnopermR)
  Bperm,transB = willperm(conjB,eltype(QtensB),BnopermR,BnopermL)

  if !Aperm
    A = changeblock(QtensA,notconA,conA)
    transA = 'N'
  else
    A = QtensA
  end
  if !Bperm
    B = changeblock(QtensB,conB,notconB)
    transB = 'N'
  else
    B = QtensB
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
  Ablocks = A.currblock[1]
  Bblocks = B.currblock[2]

  newrowcols = Array{NTuple{2,Array{intType,2}},1}(undef,numQNs)
  newQblocksum = Array{NTuple{2,Q},1}(undef,numQNs)
  outTens = Array{Array{outType,2},1}(undef,numQNs)

  #optional types like alpha input into functions from upstream calls create perhaps a type instability? Disabling alpha saves one allocation

  for q = 1:numQNs
    Aqind = commonblocks[q][1]
    Bqind = commonblocks[q][2]

    if conjA && !(eltype(A.T[Aqind]) <: Real) && transA == 'N'
      inputA = conj(A.T[Aqind])
    else
      inputA = A.T[Aqind]
    end

    if conjB && !(eltype(B.T[Bqind]) <: Real) && transB == 'N'
      inputB = conj(B.T[Bqind])
    else
      inputB = B.T[Bqind]
    end

    mulblockA,mulblockB = checkblocks(Aqind,Bqind,A,B,inputA,inputB,Aind=Aretind,Bind=Bretind)

    Lsize,innersizeL = size(mulblockA,notAretind),size(mulblockA,Aretind)
    Rsize,innersizeR = size(mulblockB,notBretind),size(mulblockB,Bretind)

    if useZ
      Zqind = Zcommonblocks[q][2]
      outTens[q] = libmult(transA,transB,type_alpha,mulblockA,mulblockB,type_beta,inputZed,Lsize,innersizeL,innersizeR,Rsize)
    elseif usealpha
      outTens[q] = libmult(transA,transB,type_alpha,mulblockA,mulblockB,Lsize,innersizeL,innersizeR,Rsize)
    else
      outTens[q] = libmult(transA,transB,mulblockA,mulblockB,Lsize,innersizeL,innersizeR,Rsize)
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

  if size(mA)[iA] == size(mB)[iB]
    println("contracting over indices with equal sizes")
  else
    error("some indices in A or B are not equal size; A->",size(mA)[iA],", B->",size(mB)[iB])
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
            error("unmatching quantum numbers on indices on index ",iA[a]," of A and index ",iB[a]," of B: value ",w)
          end
        end
      else
        error("unmatching quantum number vector lengths")
      end
      println("maching quantum numbers on both indices")
    end
  end
  nothing
end
export checkcontract
#end
