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
    Module: tensor

Stores functions for the dense tensor (tens) type
"""
#=
module tensor
import LinearAlgebra
=#

"""
    denstens

Abstract type for either dense tensors

See also: [`tens`](@ref)
"""
abstract type denstens end
export denstens

"""
    qarray

Abstract type for either Qtensors

See also: [`Qtens`](@ref)
"""
abstract type qarray end
export qarray

"""
    TensType

Abstract type for either Qtensors or AbstractArrays or dense tensors

See also: [`Qtens`](@ref) [`denstens`](@ref)
"""
const TensType = Union{qarray,denstens,AbstractArray}
export TensType

"""
    densTensType

Abstract type for either AbstractArrays or dense tensors

See also: [`denstens`](@ref)
"""
const densTensType = Union{AbstractArray,denstens}
export densTensType

"""
    tenstype

Duplicate of `TensType`

See also: [`TensType`](@ref)
"""
const tenstype = TensType
export tenstype

"""
    intType

intType` = Int64
"""
const intType = Int64
export intType

"""
    intvectype

equivalent to `Union{intType,Array{intType,1},Array{intType,2}}`

See also: [`intType`](@ref) [`convIn`](@ref)
"""
const intvecType = Union{P,Array{P,1},Array{P,2},Tuple} where P <: Integer
export intvecType

"""
    genColType

All types input into `getindex!` (UnitRange,intType,Array{intType,1},Colon)

See also: [`getindex!`](@ref)
"""
const genColType = Union{UnitRange,Integer,Array{P,1},Colon,StepRange,Tuple} where P <: Integer
export genColType

"""
    tens{Z}

regular tensor type; defined as `tens{W}` for W <: Number

# Fields:
+ `size::Array{intType,1}`: size of base tensor (unreshaped)
+ `T::Array{W,1}`: tensor reshaped into a vector
"""
mutable struct tens{W <: Number} <: denstens
  size::Tuple
  T::Array{W,1}
end

"""
    tens([type=Float64])

Initializes an empty tensor with no indices
"""
function tens(;type::DataType=Float64)
  return tens{type}((),type[])
end

"""
    tens(type)

Initializes an empty tensor with no indices of data-type `type`
"""
function tens(type::DataType)
  return tens{type}((),type[])
end

"""
    tens{T}()

Initializes an empty tensor with no indices of data-type `T`
"""
function tens{T}() where T <: Number
  return tens(type=T)
end

"""
    tens(T,P)

Converts tensor `P` into a `denstens` and converts to type `T`

See also: [`denstens`](@ref)
"""
function tens(G::DataType,P::Array{W,N}) where W <: Number where N
  sizeP = size(P) #size of P to a vector
  vecP = reshape(P,prod(sizeP))
  if G != W #converts types if they do not match
    rP = convert(Array{G,1},vecP)
  else
    rP = vecP
  end
  return tens{G}(sizeP,rP)
end

"""
    tens(P)

Converts tensor `P` into a `denstens` of the same type

See also: [`denstens`](@ref)
"""
function tens(P::Array{W,N}) where W <: Number where N
  return tens(W,P)
end

"""
    tens{W}(P)

Converts array `P` into a `denstens` and converts to type `W`

See also: [`denstens`](@ref) [`Array`](@ref)
"""
function tens{W}(P::Array{G,N}) where {W <: Number, G <: Number} where N
  return tens(W,P)
end

"""
    tens{W}(P)

Converts tensor `P` into the `denstens` and converts to type `W`

See also: [`denstens`](@ref)
"""
function tens{W}(P::tens{Z}) where {W <: Number, Z <: Number}
  newtens = tens(W)
  newtens.size = P.size
  newtens.T = convert(Array{W,1},P.T)
  return newtens
end
export tens

import Base.rand
"""
    rand(A)

generates a random tensor from inputting another tensor (rank-2) or from quantum number labels; can assign `arrows` for Qtensors if alternate arrow convention is requested
"""
function rand(rho::tens{W}) where W <: Number
  return tens{W}(rand(W, size(rho)))
end

import Base.zero
"""
  zero(A)

creates a new tensor of the same size as `A` but filled with zeros (same as internal julia function)
"""
function zero(M::tens{W}) where W <: Number
  return tens{W}(zeros(W,M.size))
end

"""
    makeId(W,ldim[,addone=false,addRightDim=false])

Generates an identity tensor (`denstens` of output type `W`) that contracts to the identity operator (rank-2) when applied on another tensor traces out that a pair of indices of equal dimension. Parameter `ldim` denotes the size of the index to contract over, `addone` if `true` leaves two indices of size 1 on indices 1 and 4 (rank-4 tensor). Option `addRightDim` adds one index of size 1 on the third index (rank-3).

See also: [`denstens`](@ref)
"""
function makeId(W::DataType,ldim::Integer;addone::Bool=false,addRightDim::Bool=false)
  oneval = W(1)
  if addone
    if addRightDim
      newsize = (ldim,ldim,1)
    else
      newsize = (1,ldim,ldim,1)
    end
  else
    newsize = (ldim,ldim)
  end
  Id = zeros(W,ldim^2)
  @simd for i = 1:ldim
    @inbounds Id[i + ldim*(i-1)] = oneval
  end
  return tens{W}(newsize,Id)
end

"""
    makeId(A,iA)

Similar to other `makeId`, but inputs tensor `A` (`denstens` or `AbstractArray`) with a pair of indices to trace over. Indices are selected by vector `iA` that identifies which indices from which to generate the identity matrix.

Example: `makeId(A,[[1,2],[3,4],[5,6]])`

See also: [`denstens`](@ref)
"""
function makeId(A::tens{W},iA::R) where R <: intvecType where W <: Number
  Id = makeId(W,size(A,iA[1][1]),addone=true,addRightDim=true)
  for g = 2:length(iA)
    addId = makeId(W,size(A,iA[g][1]),addone=true,addRightDim=false)
    Id = contract(Id,ndims(Id),addId,1)
  end
  newsize = ntuple(a->size(Id,a),ndims(Id)-1)
  return reshape!(Id,newsize...)
end

function makeId(A::Array{W,N},iA::R) where N where {R <: intvecType, W <: Number}
  densA = tens(A)
  Id = makeId(densA,iA)
  return makedens(Id)
end
export makeId

"""
    convertTens(G,M)

Convert `denstens` from type `M` to type `G`

See also: [`denstens`](@ref)
"""
function convertTens(W::DataType, M::denstens)
  return tens{W}(M.size,convert(Array{W,1}, M.T))
end
export convertTens



"""
    makedens(A)

converts `denstens` (`A`) to dense array

See also: [`denstens`](@ref)
"""
function makedens(M::denstens)
  return M
end

function makedens(M::AbstractArray)
  return tens(M)
end
export makedens

"""
  makeArray(M)

Convert `denstens` `M` to julia `Array`

See: [`makedens`](@ref) [`denstens`](@ref) [`Array`](@ref)
"""
function makeArray(M::denstens)
  return reshape(M.T, size(M))
end

function makeArray(M::AbstractArray)
  return M
end
export makeArray
















"""
    convIn(iA)

Convert `iA` of type Int64 (ex: 1), Array{Int64,1} ([1,2,3]), or Array{Int64,2}* ([1 2 3]) to Array{Int64,1}

*- two-dimensional arrays must be size "m x 1"
"""
function convIn(iA::Union{Array{P,1},Array{P,2}}) where P <: Integer
  return ntuple(i->iA[i],length(iA))
end

function convIn(iA::Integer)
  return (iA,)
end

function convIn(iA::Tuple)
  return iA
end
export convIn


"""
  findnotcons(nA,iA)

Generates the complement set of an input `iA` for a total number of elements `nA`.  Used for contractions and other functions.
"""
function findnotcons(nA::Integer,iA::Tuple)
  notcon = ()
  for i = 1:nA
    k = 0
    notmatchinginds = true
    while k < length(iA) && notmatchinginds
      k += 1
      notmatchinginds = iA[k] != i
    end
    if notmatchinginds
      notcon = (notcon...,i)
    end
  end
  return notcon
end
export findnotcons



"""
    checkType(A,B)

converts both of `A` and `B` to `denstens` if they are not already and the types are mixed between AbstractArrays and `denstens`
"""
function checkType(A::R,B::S) where {R <: AbstractArray, S <: denstens}
  return tens(A),B
end

function checkType(A::S,B::R) where {R <: AbstractArray, S <: denstens}
  return A,tens(B)
end

function checkType(A::AbstractArray,B::AbstractArray)
  return A,B
end

function checkType(A::W,B::W) where {W <: Union{denstens,qarray}}
  return A,B
end
export checkType

"""
    ind2pos!(currpos,k,x,index,S)

converts `x[index]` to a position stored in `currpos` (parallelized) with tensor size `S`

#Arguments:
+`currpos::Array{Z,1}`: input position
+`k::Integer`: current thread for `currpos`
+`x::Array{Y,1}`: vector of intput integers to convert
+`index::Integer`: index of input position of `x`
+`S::Array{W,1}`: size of tensor to convert from

See also: [`pos2ind`](@ref)
"""
@inline function ind2pos!(currpos::Array{Array{X,1},1},k::X,x::Array{X,1},index::X,S::Array{X,1}) where X <: Integer
  @inbounds currpos[k][1] = x[index]-1
  @simd for j = 1:size(S,1)-1
    @inbounds currpos[k][j+1] = fld(currpos[k][j],S[j])
    @inbounds currpos[k][j] = (currpos[k][j]) % S[j] + 1
  end
  @inbounds currpos[k][size(S,1)] = currpos[k][size(S,1)] % S[size(S,1)] + 1
  nothing
end


"""
    pos2ind(currpos,S)

generates a index from an input position `currpos` (tuple) with tensor size `S` (tuple)
"""
@inline function pos2ind(currpos::NTuple{N,P},S::NTuple{N,P}) where {N, P <: Integer}
  x = 0
  @simd for i = length(S):-1:2
    @inbounds x += currpos[i]-1
    @inbounds x *= S[i-1]
  end
  @inbounds x += currpos[1]
  return x
end
export pos2ind


@inline function pos2ind!(x::Array{X,1},j::Integer,currpos::Tuple,S::Union{Array{X,1},Tuple}) where X <: Integer
  @inbounds x[j] = currpos[end]
  @simd for i = length(currpos)-1:-1:1
    @inbounds x[j] -= 1
    @inbounds x[j] *= S[i]
    @inbounds x[j] += currpos[i]
  end
  nothing
end

@inline function pos2ind!(x::Array{X,1},j::Integer,currpos::Union{Array{X,1},Tuple},S::Tuple) where X <: Integer
  @inbounds x[j] = currpos[end]
  @simd for i = length(S)-1:-1:1
    @inbounds x[j] -= 1
    @inbounds x[j] *= S[i]
    @inbounds x[j] += currpos[i]
  end
  nothing
end
export pos2ind!


"""
    get_ranges(C,a...)

Converts `a` of type `genColType` to Arrays for use in `getindex`
"""
function get_ranges(C::Tuple,a::genColType...)
  ap = Array{genColType,1}(undef,length(a))
  for y = 1:length(ap)
    if typeof(a[y]) <: Colon
      ap[y] = 1:C[y]
    elseif typeof(a[y]) <: Integer
      ap[y] = a[y]:a[y]
    elseif typeof(a[y]) <: AbstractArray
      ap[y] = 1:length(a[y])
    else
      ap[y] = a[y]
    end
  end
  return ap
end

"""
    makepos(ninds[,zero=])

Generates a 1-indexed vector of length `nind` (0-indexed if `zero=-1`) with first entry 0 and the rest 1.

See also: [`position_incrementer!`](@ref)
"""
function makepos(ninds::P;zero::P=0) where P <: Integer
  pos = Array{P,1}(undef,ninds)
  pos[1] = zero
  one = zero + 1
  for g = 2:ninds
    pos[g] = one
  end
  return pos
end
export makepos

"""
    makepos(pos,sizes)

Increments a vector (but no entry over `sizes`) by one step.  Will change contents of `pos`.
"""
@inline function position_incrementer!(pos::Array{G,1},sizes::Union{Array{G,1},Tuple}) where G <: intType
  w = 1
  @inbounds pos[w] += 1
  @inbounds while w < length(sizes) && pos[w] > sizes[w]
    @inbounds pos[w] = 1
    w += 1
    @inbounds pos[w] += 1
  end
  nothing
end
export position_incrementer!











import Base.copy
"""
    copy(M)

Copies a `denstens`; `deepcopy` is inherently not type stable, so this function should be used instead

See: [`denstens`](@ref) [`deepcopy`](@ref)
"""
function copy(A::tens{W}) where {W <: Number}
  return tens{W}(A.size,copy(A.T))
end

import Base.length
"""
  length(M)

Returns number of elements total in `denstens` `M`

See: [`denstens`](@ref)
"""
function length(M::denstens)
  return length(M.T)
end

import LinearAlgebra.size
import Base.size
"""
    size(A)

gets the size of a `denstens` `A` (identical usage to `Array` `size` call)

See also: [`denstens`](@ref) [`Array`](@ref)
"""
function size(A::denstens)
  return A.size
end

"""
    size(A,i)

gets the size of index `i` of a `denstens` `A` (identical usage to `Array` `size` call)

See also: [`denstens`](@ref) [`Array`](@ref)
"""
function size(A::denstens,i::Integer)
  return A.size[i]
end

import Base.sum
"""
    sum(A)

sum elements of a `denstens` `A`

See: [`denstens`](@ref)
"""
function sum(A::denstens)
  return sum(A.T)
end

import LinearAlgebra.norm
"""
    norm(A)

Froebenius norm of a `denstens` `A

See: [`denstens`](@ref)`
"""
function norm(A::denstens)
  return norm(A.T)
end
export norm

import LinearAlgebra.conj!
"""
    conj!(A)

conjugates a `denstens` `A` in-place (creates no copy)

See also: [`conj`](@ref) [`denstens`](@ref)`
"""
function conj!(currtens::tens{W}) where W <: Number
  if !(W <: Real)
    LinearAlgebra.conj!.(currtens.T)
  end
  return currtens
end

import LinearAlgebra.conj
"""
    conj(A)

conjugates a `denstens` by creating a copy

See also: [`conj!`](@ref) [`denstens`](@ref)`
"""
function conj(M::tens{W}) where W <: Number
  newT = LinearAlgebra.conj(M.T)
  return tens{W}(M.size,newT)
end

import LinearAlgebra.ndims
"""
    ndims(A)

Number of dimensions of a `denstens` (identical usage to `Array` `ndims` call)

See also: [`denstens`](@ref) [`Array`](@ref)
"""
function ndims(A::denstens)
  return length(A.size)
end

import Base.lastindex
"""
    lastindex(M,i)

Same as julia's `Array` `lastindex` but for `denstens`

See also: [`lastindex`](@ref) [`denstens`](@ref)
"""
function lastindex(M::denstens, i::Integer)
  return M.size[i]
end


import Base.eltype
"""
    eltype(A)

Returns the element type contained in the `T` field of `denstens` `A`
"""
function eltype(A::tens{W}) where W <: Number
  return W
end

"""
    elnumtype(A)

Returns the element type contained in the `T` field of `denstens` `A`.  Same behavior as `eltype` for `denstens`.

See also: [`eltype`](@ref) [`denstens`](@ref)
"""
function elnumtype(A::tens{W}) where W <: Number
  return eltype(A)
end
export elnumtype








import Base.getindex
"""
    A[[genColType]]

For example, `A[:,3:6,2,[1,2,4,8]]`

Finds selected elements of a `denstens` similar to julia's form for `denstens`

See also: [`searchindex`](@ref) [`getindex!`](@ref) [`denstens`](@ref) [`genColType`](@ref)
"""
function getindex(M::denstens, a::genColType...)
  return getindex!(M,a...)
end

function getindex(C::tens{W}, a::G...)where W <: Number where G <: Array{Bool,1}
  M = makeArray(C)
  return tens{W}(M[a...])
end

function getindex!(C::AbstractArray, a::genColType...)
  return getindex(C, a...)
end

"""
  getindex!(A,genColType...)

For example, `getindex!(A,[:,3:6,2,[1,2,4,8]])`

Finds selected elements of a `denstens` similar to julia's form for `denstens`

See also: [`searchindex`](@ref) [`getindex`](@ref) [`denstens`](@ref) [`genColType`](@ref)
"""
function getindex!(C::tens{W}, a::genColType...) where W <: Number
  i = 0
  allintegers = true
  while allintegers && i < length(a)
    i += 1
    allintegers = typeof(a[i]) <: Integer
  end
  if allintegers
    val = pos2ind(a,C.size)
    return C.T[val]
  else
    dC = makeArray(C)[a...]
    return tens{W}(dC)
  end
end
export getindex!



"""
    searchindex(C,a...)

Find element of `C` that corresponds to positions `a`
"""
function searchindex(C::denstens,a::Integer...)
  if length(C.T) == 0
    outnum = 0
  elseif length(C.size) == 0
    outnum = C.T[1]
  else
    veca = ntuple(i->a[i],length(a))
    w = pos2ind(veca,C.size)
    outnum = C.T[w]
  end
  return outnum
end
export searchindex


import Base.setindex!
"""
    setindex!(B,A,i...)

puts `A` into the indices of `B` specified by `i` (functionality for tensor class)
"""
function setindex!(B::denstens,A::G,a::genColType...) where G <: Union{denstens,AbstractArray}
  indexranges = get_ranges(B.size,a...)
  nterms = prod(newsize)

  Asize = size(A)
  Bsize = size(B)
  for z = 1:nterms
    ind2pos!(pos,z,Asize)
    @simd for i = 1:length(indexranges)
      @inbounds s = pos[i]
      @inbounds pos[i] = indexranges[i][s]
    end
    x = pos2ind(pos,Bsize)
    @inbounds B.T[x] = A.T[z]
  end
  nothing
end

function setindex!(B::tens{W},A::W,a::Integer...) where W <: Number
  @inbounds index = a[end]-1
  @simd for q = length(a)-1:-1:1
    @inbounds index *= size(B,q)
    @inbounds index += a[q]-1
  end
  @inbounds B.T[index+1] = A
  nothing
end









"""
    tensorcombination(M...[,alpha=,fct=])

Performs a linear combination of the input tensors `M` with coefficients `alpha`.  For example, A*2 + B*3 is tensorcombinaton(A,B,alpha=(2,3)).

The input function `fct` can be altered. For example,  A/2 + B/3 is tensorcombinaton(A,B,alpha=(2,3),fct=/).
"""
function tensorcombination(M::denstens...;alpha::Tuple=ntuple(i->eltype(M[1])(1),length(M)),fct::Function=*)
  alphamult = prod(alpha)
  Mmult = prod(i->M[i].T[1],1:length(M))
  outType = typeof(alphamult * Mmult)
  newTensor = tens{outType}(M[1].size,zeros(outType,length(M[1])))
  for i = 1:length(M[1])
    @inbounds newTensor.T[i] = 0
    @simd for k = 1:length(M)
      @inbounds newTensor.T[i] += fct(M[k].T[i],alpha[k])
    end
  end
  return newTensor
end

function tensorcombination(M::AbstractArray...;alpha::Tuple=ntuple(i->eltype(M[1])(1),length(M)),fct::Function=*)
  return sum(k->fct(M[k],alpha[k]),1:length(M))
end

function tensorcombination(alpha::Tuple,M::P...;fct::Function=*) where P <: TensType
  return tensorcombination(M...,alpha=alpha,fct=fct)
end
export tensorcombination

"""
    tensorcombination!(M...[,alpha=,fct=])

Same as `tensorcombination` but alters the contents of the first input tensor in `M`

See also: [`tensorcombination`](@ref)
"""
function tensorcombination!(M::denstens...;alpha::Tuple=ntuple(i->eltype(M[1])(1),length(M)),fct::Function=*) where W <: Number
  for i = 1:length(M[1])
    x = 0
    @simd for k = 1:length(M)
      @inbounds x += fct(M[k].T[i],alpha[k])
    end
    M[1].T[i] = x
  end
  return M[1]
end

function loadM!(output::Array{W,2},input::Array{W,2}) where W <: Number
  for x = 1:size(output,1)
    @simd for y = 1:size(output,2)
      @inbounds output[x,y] = input[x,y]
    end
  end
  nothing
end

function tensorcombination!(M::AbstractArray...;alpha::Tuple=ntuple(i->eltype(M[1])(1),length(M)),fct::Function=*)
  out = tensorcombination(M...,alpha=alpha,fct=fct)
  loadM!(M[1],out)
  return out
end

function tensorcombination!(alpha::Tuple,M::P...;fct::Function=*) where P <: TensType
  return tensorcombination!(M...,alpha=alpha,fct=fct)
end
export tensorcombination!



"""
    mult!(A,x)

Multiplies `x*A` (commutative) for dense or quantum tensors

See also: [`*`](@ref) [`add!`](@ref) [`sub!`](@ref) [`div!`](@ref)
"""
function mult!(M::W, num::Number) where W <: TensType
  return tensorcombination!(M,alpha=(num,))
end

function mult!(num::Number,M::W) where W <: TensType
  return mult!(M,num)
end
export mult!

"""
    add!(A,B,x)

Adds `A + x*B` for dense or quantum tensors

See also: [`+`](@ref) [`sub!`](@ref) [`mult!`](@ref) [`div!`](@ref)
"""
function add!(A::W, B::W, num::Number) where W <: TensType
  return tensorcombination!((eltype(A)(1),num),A,B)
end

"""
    add!(A,B)

Adds `A + B` for dense or quantum tensors

See also: [`+`](@ref) [`sub!`](@ref) [`mult!`](@ref) [`div!`](@ref)
"""
function add!(A::W, B::W) where W <: TensType
  return add!(A,B,eltype(B)(1))
end
export add!

"""
    sub!(A,B,x)

Subtracts `A - x*B` for dense or quantum tensors

See also: [`-`](@ref) [`add!`](@ref) [`sub!`](@ref) [`mult!`](@ref) [`div!`](@ref)
"""
function sub!(A::W,B::W,mult::Number) where W <: TensType
  return add!(A,B,-mult)
end

"""
    sub!(A,B)

Subtracts `A - B` for dense or quantum tensors

See also: [`-`](@ref) [`add!`](@ref) [`sub!`](@ref) [`mult!`](@ref) [`div!`](@ref)
"""
function sub!(A::W,B::W) where W <: TensType
  return add!(A,B,eltype(A)(-1))
end
export sub!

"""
    div!(A,x)

Division by a scalar `A/x` (default x = 1) for dense or quantum tensors

See also: [`/`](@ref) [`add!`](@ref) [`sub!`](@ref) [`mult!`](@ref)
"""
function div!(M::TensType, num::Number)
  return tensorcombination!(M,alpha=(num,),fct=/)
end
export div!

import LinearAlgebra.+
"""
    +(A,B)

adds two tensors `A` and `B` together

See also: [`add!`](@ref)
"""
function +(A::TensType, B::TensType)
  return tensorcombination(A,B)
end

import LinearAlgebra.-
"""
    -(A,B)

subtracts two tensors `A` and `B` (`A`-`B`)

See also: [`sub!`](@ref)
"""
function -(A::TensType, B::TensType)
  return tensorcombination((eltype(A)(1),eltype(B)(-1)),A,B)
end

import Base.*
"""
    *(A,num)

mutiplies a tensor `A` by a number `num`

See also: [`mult!`](@ref)
"""
function *(num::Number, M::TensType)
  return tensorcombination(M,alpha=(num,))
end

"""
    *(num,A)

mutiplies a tensor `A` by a number `num`. Ensures commutativity of the operation

See also: [`mult!`](@ref)
"""
function *(M::TensType, num::Number)
  return num * M
end

import LinearAlgebra./
"""
    /(A,num)

divides a tensor `A` by a number `num`

See also: [`div!`](@ref)
"""
function /(M::TensType, num::Number)
  return tensorcombination(M,alpha=(num,),fct=/)
end

"""
    sqrt!(M)

Takes the square root of a dense tensor (new tensor created) or Qtensor (in-place)
"""
function sqrt!(M::TensType)
  return tensorcombination!(M,alpha=(0.5,),fct=^)
end
export sqrt!

import Base.sqrt
"""
    sqrt(M)

Takes the square root of a tensor

See also: [`sqrt`](@ref)
"""
function sqrt(M::TensType)
  return tensorcombination(M,alpha=(0.5,),fct=^)
end

"""
    inverse_element(x,zero)

Computes the inverse of the input element `x` up to some numerical zero value `zero`.  Below this value, zero is returned to avoid NaNs.
"""
function inverse_element(x::Number,zeronum::Real)
  return abs(x) > zeronum ? 1/x : 0.
end

"""
    invmat!(M[,zero=])

Creates inverse of a diagonal matrix in place (dense matrices are copied);
if value is below `zero`, the inverse is set to zero

See also: [`invmat!`](@ref)
"""
function invmat!(M::denstens;zeronum::Float64=1E-16)
  return tensorcombination!(M,alpha=(zeronum,),fct=inverse_element)
end

function invmat!(M::AbstractArray;zeronum::Float64=1E-16)
  for a = 1:length(M)
    @inbounds M[a,a] = abs(M[a,a]) > zeronum ? 1/M[a,a] : 0.
  end
  return M
end
export invmat!


"""
    invmat(M[,zero=])

Creates inverse of a diagonal matrix; if value is below `zero`, the inverse is set to zero

See also: [`invmat!`](@ref)
"""
function invmat(M::denstens;zeronum::Float64=1E-16)
  return tensorcombination(M,alpha=(zeronum,),fct=inverse_element)
end

function invmat(M::AbstractArray;zeronum::Float64=1E-16)
  outM = zeros(eltype(M),size(M))
  for a = 1:length(M)
    outM = abs(M[a,a]) > zeronum ? 1/M[a,a] : 0.
  end
  return outM
end
export invmat

import Base.exp
"""
    exp(A)

Exponentiate a matrix `A` from the `denstens` type
"""
function exp(A::tens{W}) where W <: Number
  X = reshape(A.T,A.size)
  return tens(W,exp(X))
end


















"""
    reshape!(M,a...[,merge=])

In-place reshape for dense tensors (otherwise makes a copy); can also make Qtensor unreshapable with `merge`, joining all grouped indices together

See also: [`reshape`](@ref)
"""
function reshape!(M::tens{W}, S::Tuple;merge::Bool=false) where W <: Number
  M.size = S
  return M
end

function reshape!(M::tens{W}, S::Integer...;merge::Bool=false) where W <: Number
  M.size = S
  return M
end

function reshape!(M::tens{W}, S::Array{Array{P,1},1};merge::Bool=false) where {W <: Number, P <: Integer}
  newsize = ntuple(a->prod(b->size(M,b),S[a]),length(S))
  order = vcat(S...)
  pM = issorted(order) ? M : permutedims!(M,order)
  return reshape!(M,newsize)
end
export reshape!

import Base.reshape
function reshape(M::tens{W}, S::Integer...;merge::Bool=false) where W <: Number
  newMsize = S
  newM = tens{W}(newMsize,M.T)
  return reshape!(newM,S)
end

function reshape(M::tens{W}, S::Array{Array{P,1},1};merge::Bool=false) where {W <: Number, P <: Integer}
  return reshape!(copy(M),S)
end


"""
    unreshape!(M)

Same as `reshape!` but used for ease of reading code and also has new context with quantum numbers
"""
function unreshape!(M::AbstractArray, S::Integer...;merge::Bool=false) where W <: Number
  return reshape(M,S...)
end

function unreshape!(M::tens{W}, S::Integer...;merge::Bool=false) where W <: Number
  M.size = S
  return M
end
export unreshape!

function unreshape(M::AbstractArray, S::Integer...;merge::Bool=false) where W <: Number
  return reshape(M,S...)
end

function unreshape(M::tens{W}, S::Integer...;merge::Bool=false) where W <: Number
  newM = tens{W}(M.size,M.T)
  return unreshape!(newM,S...)
end
export unreshape

import Base.permutedims!
"""
    permutedims!(A,[1,3,2,...])

permute dimensions of a Qtensor in-place (no equivalent, but returns value so it can be used identically to `permutedims` on a dense tensor)

See also: [`permutedims`](@ref)
"""
function permutedims!(M::tens{W}, vec::Array{P,1}) where {W <: Number, P <: Integer}
  return permutedims!(M,(vec...,),M.size...)
end

function permutedims!(M::tens{W}, vec::Tuple) where W <: Number
  return permutedims!(M,vec,M.size...)
end

function permutedims!(M::AbstractArray, vec::Tuple) where W <: Number
  return permutedims(M,vec)
end

"""
    permutedims!(M,vec,x...)

Permutes `M` according to `vec`, but the tensor `M` is reshaped to `x`.  The size of `x` must match `vec`'s size.
"""
function permutedims!(M::tens{W}, vec::Tuple,x::Integer...) where W <: Number
  rM = reshape(M.T,x)
  xM = permutedims(rM, vec)
  out = tens(W,xM)
  return out
end

import Base.permutedims
"""
    permutedims(A,[1,3,2,...])

permutes dimensions of `A`  (identical usage to dense `size` call)

See also: [`permutedims!`](@ref)
"""
function permutedims(M::tens{W}, vec::Array{P,1}) where {W <: Number, P <: Integer}
  return permutedims!(M,(vec...,))
end
































"""
    joinindex!(vec,A,B)

in-place joinindexenatation of tensors `A` (replaced for Qtensors only) and `B` along indices specified in `vec`
"""
function joinindex!(bareinds::intvecType,A::Array{R,N},B::Array{S,N})::Array{typeof(R(1)*S(1)),N} where {R <: Number, S <: Number, N}
  inds = convIn(bareinds)
  
  nA = ndims(A)
  nB = nA
  axesA = [1:size(A,a) for a = 1:nA]
  finalsize = [size(A,i) for i = 1:nA]
  @simd for i = 1:length(inds)
    @inbounds a = inds[i]
    @inbounds finalsize[a] += size(B,a)
  end
  thistype = typeof(S(1)*R(1))
  if length(inds) > 1
    newTensor = zeros(thistype,finalsize...)
  else
    dim = length(finalsize)
    newTensor = Array{thistype,dim}(undef,finalsize...)
  end

  notinds = findnotcons(ndims(A),(inds...,))
  axesout = Array{UnitRange{intType},1}(undef,nA)
  @simd for i = 1:length(notinds)
    @inbounds a = notinds[i]
    @inbounds axesout[a] = 1:finalsize[a] 
  end
  @simd for i = 1:length(inds)
    @inbounds b = inds[i]
    @inbounds start = size(A,b)+1
    @inbounds stop = finalsize[b]
    @inbounds axesout[b] = start:stop
  end

  @inbounds newTensor[axesA...] = A
  @inbounds newTensor[axesout...] = B

  return newTensor
end

function joinindex!(bareinds::intvecType,A::tens{S},B::tens{W}) where {W <: Number, S <: Number}
  rA = reshape!(A.T,A.size...)
  rB = reshape(B.T,B.size...)
  C = joinindex!(bareinds,rA,rB)
  retType = typeof(S(1)*W(1))
  return tens(retType,C)
end

"""
    joinindex(vec,A,B...)

joinindexenatation of tensors `A` and any number of `B` along indices specified in `vec`
"""
function joinindex(inds::intvecType,A::W,B::R...) where {W <: TensType, R <: TensType}
  if typeof(A) <: densTensType
    C = A
  else
    C = copy(A)
  end
  return joinindex!(inds,C,B...)
end

"""
    joinindex(A,B,vec)

joinindexenatation of tensors `A` and any number of `B` along indices specified in `vec`
"""
function joinindex(A::W,B::R,inds::intvecType) where {W <: TensType, R <: TensType}
  mA,mB = checkType(A,B)
  return joinindex(inds,mA,mB)
end
export joinindex

"""
    joinindex!(A,B,vec)

In-place joinindexenatation of tensors `A` and any number of `B` along indices specified in `vec`
"""
function joinindex!(A::W,B::R,inds::intvecType) where {W <: TensType, R <: TensType}
  mA,mB = checkType(A,B)
  return joinindex!(inds,mA,mB)
end
export joinindex!














"""
    showQtens(Qt[,show=])

Prints fields of the Qtensor (`Qt`) out to a number of elements `show`; can also be called with `print` or `println`

See also: [`Qtens`](@ref) [`print`](@ref) [`println`](@ref)
"""
function showTens(M::denstens;show::Integer = 4)
  println("printing regular tensor of type: ", typeof(M))
  println("size = ", M.size)
  maxshow = min(show, size(M.T, 1))
  maxBool = show < size(M.T, 1)
  println("T = ", M.T[1:maxshow], maxBool ? "..." : "")
  nothing
end
export showTens

import Base.print
"""
    print(A[,show=])

Idential to `showQtens`

See also: [`showQtens`](@ref) [`println`](@ref)
"""
function print(A::denstens;show::Integer = 4)
  showTens(A, show = show)
  nothing
end

import Base.println
"""
    println(A[,show=])

Idential to `showQtens`

See also: [`showQtens`](@ref) [`print`](@ref)
"""
function println(A::denstens;show::Integer = 4)
  showTens(A, show = show)
  print("\n")
  nothing
end

#end
