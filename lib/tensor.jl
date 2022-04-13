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
const intType = typeof(1)
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
const genColType = Union{UnitRange{intType},intType,Array{intType,1},Colon,StepRange{intType},Tuple{intType,Vararg{intType}}}
export genColType

"""
    tens{Z}

Regular tensor type; defined as `tens{W}` for W <: Number

# Fields:
+ `size::NTuple{N,intType}`: size of base tensor (unreshaped)
+ `T::Array{W,1}`: tensor reshaped into a vector
"""
mutable struct tens{W <: Number} <: denstens
  size::Array{intType,1} #Tuple #Tuple{intType,Vararg{intType}}
  T::Array{W,1}
end

"""
    G = tens([type=Float64])

Initializes an empty tensor `G` with no indices
"""
function tens(;type::DataType=Float64)
  return tens{type}(intType[],type[])
end

"""
    G = tens(type)

Initializes an empty tensor `G` with no indices of data-type `type`
"""
function tens(type::DataType)
  return tens{type}(intType[],type[])
end

"""
    G = tens{W}()

Initializes an empty tensor `G` with no indices of data-type `W`
"""
function tens{T}() where T <: Number
  return tens(type=T)
end

"""
    A = tens(A)

Trivial convertion of denstens `A` to itself
"""
function tens(A::denstens)
  return A
end

"""
    G = tens(W,P)

Converts tensor `P` into a `denstens` and converts to type `W` for output tensor `G`

See also: [`denstens`](@ref)
"""
function tens(G::DataType,P::AbstractArray{W,N}) where W <: Number where N
  return tens(G,Array(rP))
end

function tens(G::DataType,P::Array{W,N}) where W <: Number where N
  sizeP = [size(P,w) for w = 1:N] #size of P to a vector
  vecP = reshape(P,prod(sizeP))
  if G != W #converts types if they do not match
    rP = convert(Array{G,1},vecP)
  else
    rP = vecP
  end
  return tens{G}(sizeP,rP)
end
#=
function tens{W}(newsize::Array{intType,1},P::Array{W,N}) where W <: Number where N
  vecP = reshape(P,prod(newsize))
  return tens{W}(newsize,vecP)
end
=#

"""
    G = tens(P)

Converts tensor `P` into a `denstens` (`G`) of the same type

See also: [`denstens`](@ref)
"""
function tens(P::Array{W,N}) where {W <: Number, N}
  return tens(W,P)
end

function tens(P::AbstractArray{W,N}) where {W <: Number, N}
  return tens(eltype(P),Array(P))
end

"""
    G = tens{W}(P)

Converts array `P` into a `denstens` (`G`) and converts to type `W`

See also: [`denstens`](@ref) [`Array`](@ref)
"""
function tens{W}(P::Array{G,N}) where {W <: Number, G <: Number} where N
  return tens(W,P)
end

"""
    G = tens{W}(P)

Converts tensor `P` into the `denstens` (`G`) and converts to type `W`

See also: [`denstens`](@ref)
"""
function tens{W}(P::tens{Z}) where {W <: Number, Z <: Number}
  if W != Z
    newtens = tens(W)
    newtens.size = P.size
    newtens.T = convert(Array{W,1},P.T)
  else
    newtens = P
  end
  return newtens
end
export tens

import Base.rand
"""
    G = rand(A)

Generates a random tensor `G` from inputting another tensor (rank-2) or from quantum number labels; can assign `arrows` for Qtensors if alternate arrow convention is requested
"""
function rand(rho::tens{W}) where W <: Number
  return tens{W}(rand(W, size(rho)))
end

function rand(rho::AbstractArray{W,N}) where {W <: Number, N}
  return rand(W, size(rho))
end

import Base.zeros
function zeros(A::AbstractArray{W,N}) where {W <: Number, N}
  return zeros(W,size(A))
end

function zeros(A::tens{W}) where W <: Number
  return tens{W}(zeros(W,size(A)))
end

import Base.zero
"""
  G = zero(A)

Creates a new tensor `G` of the same size as `A` but filled with zeros (same as internal julia function)
"""
function zero(M::tens{W}) where W <: Number
  return tens{W}(zeros(W,size(M)))
end





function makeIdarray(W::DataType,ldim::Integer;addone::Bool=false,addRightDim::Bool=false,loadleft::Bool=true)
  return makeId(W,ldim,ldim;addone=addone,addRightDim=addRightDim)
end

function makeIdarray(W::DataType,ldim::Integer,rdim::Integer;addone::Bool=false,addRightDim::Bool=false,loadleft::Bool=true)
  oneval = W(1)
  if addone
    if addRightDim
      newsize = (ldim,rdim,1)
    else
      newsize = (1,ldim,rdim,1)
    end
  else
    newsize = (ldim,rdim)
  end
  Id = zeros(W,prod(newsize))
  stop = loadleft ? ldim : rdim
  @inbounds @simd for i = 1:stop
    Id[i + ldim*(i-1)] = oneval
  end
  return reshape(Id,newsize)
end

"""
    G = makeId(W,ldim[,addone=false,addRightDim=false])

Generates an identity tensor (`denstens` of output type `W`, `G`) that contracts to the identity operator (rank-2) when applied on another tensor traces out that a pair of indices of equal dimension. Parameter `ldim` denotes the size of the index to contract over, `addone` if `true` leaves two indices of size 1 on indices 1 and 4 (rank-4 tensor). Option `addRightDim` adds one index of size 1 on the third index (rank-3).

See also: [`denstens`](@ref)
"""
function makeId(W::DataType,ldim::Integer;addone::Bool=false,addRightDim::Bool=false,loadleft::Bool=true)
  return tens(makeIdarray(W,ldim,ldim;addone=addone,addRightDim=addRightDim))
end

function makeId(W::DataType,ldim::Integer,rdim::Integer;addone::Bool=false,addRightDim::Bool=false,loadleft::Bool=true)
  return tens(makeIdarray(W,ldim,rdim;addone=addone,addRightDim=addRightDim))
end

"""
    G = makeId(A,iA)

Similar to other `makeId`, but inputs tensor `A` (`denstens` or `Array`) with a pair of indices to trace over. Indices are selected by vector `iA` that identifies which indices from which to generate the identity matrix `G`.

Example: `makeId(A,[[1,2],[3,4],[5,6]])`

See also: [`denstens`](@ref)
"""
function makeId(A::tens{W},iA::Array{P,1}) where {W <: Number, P <: Union{Integer,Tuple}}
  Id = makeId(W,size(A,iA[1][1]),addone=true,addRightDim=true)
  for g = 2:length(iA)
    addId = makeId(W,size(A,iA[g][1]),addone=true,addRightDim=false)
    Id = contract(Id,ndims(Id),addId,1)
  end
  newsize = length(iA) > 1 ? size(Id)[1:ndims(Id)-1] : size(Id)
  return reshape!(Id,newsize...)
end

function makeId(A::tens{W},iA::Union{Integer,Tuple}) where {W <: Number, P <: Integer}
  return makeId(A,[iA])
end

function makeId(A::Array{W,N},iA::Array{P,1}) where {N, W <: Number,P <: Union{Integer,Tuple}}
  densA = tens(A)
  Id = makeId(densA,iA)
  return makedens(Id)
end
export makeId

"""
    K = convertTens(G,M)

Convert `denstens` from type `M` to type `G` (`K`)

See also: [`denstens`](@ref)
"""
function convertTens(W::DataType, M::denstens)
  return tens{W}(M.size,convert(Array{W,1}, M.T))
end

function convertTens(T::DataType, M::Array{W,N}) where {W <: Number, N}
  return convert(Array{T,N},M)
end
export convertTens

"""
    G = makedens(A)

converts `denstens` (`A`) to dense array `G`

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
  G = makeArray(M)

Convert `denstens` `M` to julia `Array` (`G`)

See: [`makedens`](@ref) [`denstens`](@ref) [`Array`](@ref)
"""
function makeArray(M::tens{W}) where W <: Number
  return reshape!(M.T, size(M))
end

function makeArray(M::AbstractArray)
  return Array(M)
end
export makeArray

"""
    G = convIn(iA)

Convert `iA` of type Int64 (ex: 1), Array{Int64,1} ([1,2,3]), or Array{Int64,2}* ([1 2 3]) to Array{Int64,1} (`G`)

*- two-dimensional arrays must be size "m x 1"
"""
function convIn(iA::Union{Array{P,1},Array{P,2}}) where P <: Integer
  return ntuple(i->iA[i],length(iA))
end

function convIn(iA::Integer)
  return (iA,)
end

function convIn(iA::NTuple{N,intType}) where N
  return iA
end
export convIn

"""
  G = findnotcons(nA,iA)

Generates the complement set of an input `iA` (`G`) for a total number of elements `nA`.  Used for contractions and other functions.
"""
function findnotcons(nA::Integer,iA::NTuple{N,intType}) where N
  notconvec = Array{intType,1}(undef,nA-length(iA))
  counter = 0
  @inbounds for w = 1:nA
    k = 0
    notmatchinginds = true
    @inbounds while k < length(iA) && notmatchinginds
      k += 1
      notmatchinginds = iA[k] != w
    end
    if notmatchinginds
      counter += 1
      notconvec[counter] = w
    end
  end
  return notconvec #(notconvec...,) #ntuple(w->notconvec[w],length(notconvec))
end
export findnotcons

"""
    G,K = checkType(A,B)

converts both of `A` and `B` to `denstens` (`G` and `K`) if they are not already and the types are mixed between AbstractArrays and `denstens`
"""
function checkType(A::R,B::S) where {R <: AbstractArray, S <: denstens}
  outType = typeof(eltype(A)(1)*eltype(B)(1))
  return tens{outType}(checkType(A)),tens{outType}(B)
end

function checkType(A::S,B::R) where {R <: AbstractArray, S <: denstens}
  outType = typeof(eltype(A)(1)*eltype(B)(1))
  return tens{outType}(A),tens{outType}(checkType(B))
end

#for abstract arrays, we only care about the element type
function checkType(A::AbstractArray{R,N},B::AbstractArray{S,M}) where {R <: Number, S <: Number, N, M}
  outType = typeof(eltype(A)(1)*eltype(B)(1))
  return convert(Array{outType,N},checkType(A)),convert(Array{outType,M},checkType(B))
end

function checkType(A::AbstractArray)
  if typeof(A) <: Array
    out = A
  else
    out = Array(A)
  end
  return out
end

function checkType(A::denstens)
  return A
end

function checkType(A::tens{W},B::tens{G}) where {W <: Number, G <: Number}
  outType = typeof(eltype(A)(1)*eltype(B)(1))
  return tens{outType}(A),tens{outType}(B)
end
export checkType

"""
    ind2pos!(currpos,k,x,index,S)

Converts `x[index]` to a position stored in `currpos` (parallelized) with tensor size `S`

#Arguments:
+`currpos::Array{Z,1}`: input position
+`k::Integer`: current thread for `currpos`
+`x::Array{Y,1}`: vector of intput integers to convert
+`index::Integer`: index of input position of `x`
+`S::Array{W,1}`: size of tensor to convert from

See also: [`pos2ind`](@ref) [`pos2ind!`](@ref)
"""
@inline function ind2pos!(currpos::Array{Array{X,1},1},k::X,x::Array{X,1},index::X,S::Array{X,1}) where X <: Integer
  currpos[k][1] = x[index]-1
  @inbounds @simd for j = 1:size(S,1)-1
    val = currpos[k][j]
    currpos[k][j+1] = fld(val,S[j])
    currpos[k][j] = val % S[j] + 1
  end
  @inbounds currpos[k][size(S,1)] = currpos[k][size(S,1)] % S[size(S,1)] + 1
  nothing
end

"""
    G = pos2ind(currpos,S)

Generates an index `G` from an input position `currpos` (tuple) with tensor size `S` (tuple)

See also: [`pos2ind!`](@ref)
"""
@inline function pos2ind(currpos::Union{NTuple{N,P},Array{P,1}},S::NTuple{N,P}) where {N, P <: Integer}
  x = 0
  @inbounds @simd for i = N:-1:2
    x += currpos[i]-1
    x *= S[i-1]
  end
  @inbounds x += currpos[1]
  return x
end


@inline function pos2ind(currpos::NTuple{N,P},S::Array{P,1}) where {N, P <: Integer}
  x = 0
  @inbounds @simd for i = N:-1:2
    x += currpos[i]-1
    x *= S[i-1]
  end
  @inbounds x += currpos[1]
  return x
end
#=
@inline function pos2ind(currpos::Array{P,1},S::Array{P,1}) where {N, P <: Integer,G}
  return pos2ind(currpos,S,length(S))
end

@inline function pos2ind(currpos::Array{P,1},S::Array{P,1},G::intType) where {N, P <: Integer}
  x = 0
  @inbounds @simd for i = G:-1:2
    x += currpos[i]-1
    x *= S[i-1]
  end
  @inbounds x += currpos[1]
  return x
end
=#
export pos2ind

"""
  pos2ind!(currpos,S)

Generates an index in element `j` of input storage array `x` from an input position `currpos` (tuple or vector) with tensor size `S` (tuple)

See also: [`pos2ind`](@ref)
"""
@inline function pos2ind!(x::Array{X,1},j::Integer,currpos::Union{Array{X,1},NTuple{N,intType}},S::NTuple{N,intType}) where {N, X <: Integer}
  @inbounds val = currpos[end]
  @inbounds @simd for i = length(S)-1:-1:1
    val -= 1
    val *= S[i]
    val += currpos[i]
  end
  @inbounds x[j] = val
  nothing
end
#=
"""
  pos2ind!(currpos,S)

Generates an index in element `j` of input storage array `x` from an input position `currpos` (tuple or vector) with tensor size `S` (tuple)

See also: [`pos2ind`](@ref)
"""
@inline function pos2ind!(x::Array{X,1},j::Integer,currpos::Union{Array{X,1},NTuple{N,intType}},order::Array{intType,1},S::NTuple{N,intType}) where {N, X <: Integer}
  @inbounds val = currpos[order[end]]
  @inbounds @simd for i = length(S)-1:-1:1
    val -= 1
    val *= S[order[i]]
    val += currpos[order[i]]
  end
  @inbounds x[j] = val
  nothing
end
=#
export pos2ind!


function get_denseranges(sizes::NTuple{G,intType},a::genColType...) where G
  unitranges = Array{genColType,1}(undef,length(a))
  @inbounds for i = 1:length(a)
    if typeof(a[i]) <: Colon
      unitranges[i] = 1:sizes[i] #[w for w = 1:sizes[i]]
    elseif typeof(a[i]) <: Array{intType,1} || typeof(a[i]) <: Tuple{intType,Vararg{intType}}
      unitranges[i] = [a[i][w] for w = 1:length(a[i])]
    elseif typeof(a[i]) <: UnitRange{intType}
      unitranges[i] = a[i] #UnitRange(a[i][1],a[i][end])
    elseif typeof(a[i]) <: StepRange{intType}
      unitranges[i] = a[i] #UnitRange(a[i][1],a[i][end])
    elseif typeof(a[i]) <: Integer
      unitranges[i] = a[i]
    end
  end
  return unitranges
end
#=
function get_denseranges(sizes::NTuple{G,intType},a::genColType...) where G
  unitranges = Array{Array{intType,1},1}(undef,length(a))
  @inbounds for i = 1:length(a)
    if typeof(a[i]) <: Colon
      unitranges[i] = [w for w = 1:sizes[i]]
    elseif typeof(a[i]) <: Array{intType,1} || typeof(a[i]) <: Tuple{intType,Vararg{intType}}
      unitranges[i] = [a[i][w] for w = 1:length(a[i])]
    elseif typeof(a[i]) <: UnitRange{intType}
      unitranges[i] = [w for w = a[i]]
    elseif typeof(a[i]) <: StepRange{intType}
      unitranges[i] = [w for w = a[i]]
    elseif typeof(a[i]) <: Integer
      unitranges[i] = [a[i]]
    end
  end
  return unitranges
end
=#
#=
"""
  G = get_ranges(C,a...)

Converts `a` of type `genColType` to Arrays for use in `getindex` as an output tuple `G` with as many elements as `a`
"""
function get_ranges(C::Tuple,a::genColType...)
  ap = Array{genColType,1}(undef,length(a))
  @inbounds for y = 1:length(ap)
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
=#
"""
  G = makepos!(pos)

Generates a 1-indexed vector `G` of length `nind` (0-indexed if `zero=-1`) with first entry 0 and the rest 1.

See also: [`position_incrementer!`](@ref)
"""
function makepos!(pos::Array{intType,1})
  if length(pos) > 0
    pos[1] = 0
    @inbounds @simd for g = 2:length(pos)
      pos[g] = 1
    end
  end
  return pos
end
export makepos!

"""
  G = makepos(ninds)

Generates a 1-indexed vector `G` of length `nind` (0-indexed if `zero=-1`) with first entry 0 and the rest 1.

See also: [`position_incrementer!`](@ref)
"""
function makepos(ninds::intType)
  pos = Array{intType,1}(undef,ninds)
  return makepos!(pos)
end
export makepos

"""
  position_incrementer!(pos,sizes)

Increments a vector (but no entry over `sizes`) by one step.  Will change contents of `pos`.
"""
@inline function position_incrementer!(pos::Array{G,1},sizes::Union{Array{G,1},Tuple{G,Vararg{G}}}) where G <: intType
  w = 1
  @inbounds pos[w] += 1
  @inbounds while w < length(sizes) && pos[w] > sizes[w]
    pos[w] = 1
    w += 1
    pos[w] += 1
  end
  nothing
end
export position_incrementer!

import Base.adjoint
function adjoint(M::TensType)
  pM = ndims(M) == 1 ? reshape(M,size(M)...,1) : M
  pM = permutedims(pM,[2,1])
  conj!(pM)
  return pM
end

import Base.copy
"""
  G = copy(M)

Copies a `denstens` (output `G`); `deepcopy` is inherently not type stable, so this function should be used instead

See: [`denstens`](@ref) [`deepcopy`](@ref)
"""
function copy(A::tens{W}) where {W <: Number}
  return tens{W}(A.size,copy(A.T))
end

import Base.length
"""
  G = length(M)

Returns number of elements `G` (integer) total in `denstens` `M`

See: [`denstens`](@ref)
"""
function length(M::denstens)
  return length(M.T)
end

import LinearAlgebra.size
import Base.size
"""
  G = size(A)

Outputs tuple `G` representing the size of a `denstens` `A` (identical usage to `Array` `size` call)

See also: [`denstens`](@ref) [`Array`](@ref)
"""
function size(A::denstens)
  return (A.size...,)
end

"""
  G = size(A,i)

Gets the size of index `i` of a `denstens` `A` (identical usage to `Array` `size` call) as output `G`

See also: [`denstens`](@ref) [`Array`](@ref)
"""
function size(A::denstens,i::Integer)::intType
  return A.size[i]
end

import Base.sum
"""
  G = sum(A)

Sum elements of a `denstens` `A` (ouptut `G`)

See: [`denstens`](@ref)
"""
function sum(A::denstens)
  return sum(A.T)
end

import LinearAlgebra.norm
"""
  G = norm(A)

Froebenius norm of a `denstens` `A` as output `G`

See: [`denstens`](@ref)`
"""
function norm(A::denstens)
  return real(norm(A.T))
end
export norm

import LinearAlgebra.conj!
"""
  G = conj!(A)

Conjugates a `denstens` `A` in-place (creates no copy) but also has output `G`

See also: [`conj`](@ref) [`denstens`](@ref)`
"""
function conj!(currtens::tens{W}) where W <: Number
  if !(W <: Real)
    LinearAlgebra.conj!(currtens.T)
  end
  return currtens
end

import LinearAlgebra.conj
"""
  G = conj(A)

Conjugates a `denstens` by creating a copy `G`

See also: [`conj!`](@ref) [`denstens`](@ref)`
"""
function conj(M::tens{W}) where W <: Number
  newT = LinearAlgebra.conj(M.T)
  return tens{W}(M.size,newT)
end

import LinearAlgebra.ndims
"""
  G = ndims(A)

Number of dimensions (rank) `G` of a `denstens` (identical usage to `Array` `ndims` call)

See also: [`denstens`](@ref) [`Array`](@ref)
"""
function ndims(A::denstens)
  return length(A.size)
end

import Base.lastindex
"""
  G = lastindex(M,i)

Same as julia's `Array` `lastindex` but for `denstens` with output `G`

See also: [`lastindex`](@ref) [`denstens`](@ref)
"""
function lastindex(M::denstens, i::Integer)
  return M.size[i]
end

import Base.eltype
"""
  G = eltype(A)

Returns the element type `G` contained in the `T` field of `denstens` `A`
"""
function eltype(A::tens{W}) where W <: Number
  return W
end

"""
  G = elnumtype(A)

Returns the element type `G` contained in the `T` field of `denstens` `A`.  Same behavior as `eltype` for `denstens`.

See also: [`eltype`](@ref) [`denstens`](@ref)
"""
function elnumtype(A::tens{W}) where W <: Number
  return eltype(A)
end
export elnumtype

import Base.getindex
"""
  G = getindex(M,a...)

Selects subset or single element `G` of input tensor `M` with elements of `a` being of `genColType` (ranges or values in tensor)

For example, `A[:,3:6,2,[1,2,4,8]]`

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
  G = getindex!(A,genColType...)

For example, `getindex!(A,[:,3:6,2,[1,2,4,8]])`

Finds selected elements of a `denstens` similar to julia's form for `denstens`. Performs operation in-place but outputs tensor to `G`

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
    return searchindex(C,a...)
  else
    dC = makeArray(C)[a...]
    return tens{W}(dC)
  end
end
export getindex!

"""
  G = searchindex(C,a...)

Find element of `C` that corresponds to positions `a` and outputs value `G`
"""
function searchindex(C::denstens,a::Integer...)
  if length(C.T) == 0
    outnum = eltype(C)(0)
  elseif length(C.size) == 0
    outnum = C.T[1]
  else
#    veca = ntuple(i->a[i],length(a))
    w = pos2ind(a,C.size)
    outnum = C.T[w]
  end
  return outnum
end

function searchindex(C::Array,a::Integer...)
  return searchindex(C,a)
end

function searchindex(C::Array{W,N},a::NTuple{N,intType}) where {W <: Number, N}
  return C[a...]
end
export searchindex


import Base.setindex!
"""
  setindex!(B,A,i...)

Puts `A` into the indices of `B` specified by `i` (functionality for tensor class)
"""
function setindex!(B::tens{W},A::tens{W},a::genColType...) where W <: Number #where G <: Union{denstens,AbstractArray}
  G = makeArray(B)
  G[a...] = A
  B.T = reshape(G,prod(size(G))...)


  #=
  Asize = size(A)
  Bsize = size(B)

  indexranges = get_denseranges(Bsize,a...)
  indexsizes = ntuple(w->length(indexranges[w]),length(indexranges))
  nterms = 1
  @simd for w = 1:length(Asize)
    nterms *= Asize[w]
  end

  indexpos = makepos(length(indexranges))
  pos = makepos(length(Asize))

  @inbounds for z = 1:nterms
    position_incrementer!(indexpos,indexsizes)
    transfer2vec!(pos,indexranges,indexpos)
    x = pos2ind(pos,Bsize)
    B.T[x] = A.T[z]
  end
  =#
  nothing
end

function setindex!(B::tens{W},A::Array{W,N},a::genColType...) where {W <: Number, N}#where G <: Union{denstens,AbstractArray}
  G = makeArray(B)
  G[a...] = A
  B.T = reshape(G,prod(size(G))...)
  #=
  Asize = size(A)
  Bsize = size(B)

  indexranges = get_denseranges(Bsize,a...)
  indexsizes = ntuple(w->length(indexranges[w]),length(indexranges))
  nterms = 1
  @simd for w = 1:length(Asize)
    nterms *= Asize[w]
  end

  indexpos = makepos(length(indexranges))
  pos = makepos(length(Asize))

  @inbounds for z = 1:nterms
    position_incrementer!(indexpos,indexsizes)
    transfer2vec!(pos,indexranges,indexpos)
    x = pos2ind(pos,Bsize)
    B.T[x] = A[z]
  end
  =#
  nothing
end

@inline function transfer2vec!(pos::Array{intType,1},indexranges::Array{G,1},indexpos::Array{intType,1}) where G <: genColType
  @inbounds @simd for i = 1:length(indexranges)
    s = indexpos[i]
    pos[i] = indexranges[i][s]
  end
  nothing
end

function setindex!(B::Array{W,N},A::tens{W},a::genColType...) where {W <: Number, N}

#  println("in HERE")

  G = makeArray(A)
  B[a...] = G
#=
  @time Asize = size(A)
  @time Bsize = size(B)

  @time indexranges = get_denseranges(Bsize,a...)
  @time indexsizes = ntuple(w->length(indexranges[w]),length(indexranges))
  nterms = 1
  @time @simd for w = 1:length(Asize)
    nterms *= Asize[w]
  end

  @time indexpos = makepos(length(indexranges))
  @time pos = makepos(length(Asize))

  @time @inbounds for z = 1:nterms
    @time position_incrementer!(indexpos,indexsizes)
    @time transfer2vec!(pos,indexranges,indexpos)
    @time x = pos2ind(pos,Bsize)
#    subeq(B,x,A,z)
    @time B[x] = A.T[z]
    println()
  end
  println()
  =#
  nothing
end

@inline function subeq(B::Array{W,N},x::intType,A::tens{W},z::intType) where {W <: Number, N}
  B[x] = A.T[z]
  nothing
end

function setindex!(B::tens{W},A::W,a::Integer...) where W <: Number
  @inbounds index = a[end]-1
  @inbounds @simd for q = length(a)-1:-1:1
    index *= size(B,q)
    index += a[q]-1
  end
  @inbounds B.T[index+1] = A
  nothing
end


#=
function loadM!(output::tens{W},input::Array{W,N}) where {N, W <: Number}
  @inbounds @simd for x = 1:length(input)
    output.T[x] = input[x]
  end
  nothing
end

function loadM!(output::Array{W,N},input::tens{W}) where {N, W <: Number}
  @inbounds @simd for x = 1:length(input.T)
    output[x] = input.T[x]
  end
  nothing
end
=#
"""
  loadM!(out,in)

Simple copy operation from `in` matrix to `out` matrix. Assumes same element type and useful for ensuring compiler efficiency.
"""
function loadM!(output::Array{W,N},input::Array{W,N}) where {N, W <: Number}
  @inbounds @simd for x = 1:length(input)
    output[x] = input[x]
  end
  nothing
end

"""
  G = tensorcombination(M...[,alpha=,fct=])

Performs a linear combination of the input tensors `M` with coefficients `alpha` to output tensor `G`.  For example, A*2 + B*3 is tensorcombinaton(A,B,alpha=(2,3)).

The input function `fct` can be altered. For example,  A/2 + B/3 is tensorcombinaton(A,B,alpha=(2,3),fct=/).
"""
function tensorcombination(M::tens{W}...;alpha::Tuple=ntuple(i->W(1),length(M)),fct::Function=*) where W <: Number
  sizeM = Array{intType,1}(undef,ndims(M[1]))
  @inbounds @simd for w = 1:length(sizeM)
    sizeM[w] = size(M[1],w)
  end
  newTensor = tens{W}(sizeM,Array{W,1}(undef,length(M[1])))
  nterms = min(length(M),length(alpha))
  @inbounds @simd for i = 1:length(M[1])
    newTensor.T[i] = fct(M[1].T[i],alpha[1])
  end
  @inbounds for k = 2:nterms
    @inbounds @simd for i = 1:length(M[1])
      newTensor.T[i] += fct(M[k].T[i],alpha[k])
    end
  end
  return newTensor
end

function tensorcombination(M::Array{W,N}...;alpha::NTuple{G,W}=ntuple(i->W(1),length(M)),fct::Function=*) where {W <: Number, N, G}
  #  if fct == *
  #    return lineartensorcombination!(M...,alpha=alpha)
  #  else
      nterms = min(length(M),length(alpha))
      newTensor = Array{W,N}(undef,size(M[1])...)
      @inbounds @simd for i = 1:length(M[1])
        newTensor[i] = fct(M[1][i],alpha[1])
      end
      @inbounds for k = 2:nterms
        @inbounds @simd for i = 1:length(M[1])
          newTensor[i] += fct(M[k][i],alpha[k])
        end
      end
      return newTensor
  #=
      @inbounds for w = 1:length(outT)
        x = W(0)
        @inbounds @simd for k = 1:nterms
          x = fct(M[k][w],alpha[k])
        end
        outT[w] = x
      end
      return outT
      =#
  #  end
  end

function tensorcombination(alpha::Tuple,M::P...;fct::Function=*) where P <: densTensType
  return tensorcombination(M...,alpha=alpha,fct=fct)
end
export tensorcombination

"""
  G = tensorcombination!(M...[,alpha=,fct=])

Same as `tensorcombination` but alters the contents of the first input tensor in `M` and still outputs tensor `G`

See also: [`tensorcombination`](@ref)
"""
function tensorcombination!(M::tens{W}...;alpha::NTuple{N,W}=ntuple(i->W(1),length(M)),fct::Function=*) where {W <: Number, N}
  #  if fct == *
  #    return lineartensorcombination!(M...,alpha=alpha)
  #  else
    nterms = min(length(M),length(alpha))
    @inbounds @simd for i = 1:length(M[1])
      M[1].T[i] = fct(M[1].T[i],alpha[1])
    end
    @inbounds for k = 2:nterms
      @inbounds @simd for i = 1:length(M[1])
        M[1].T[i] += fct(M[k].T[i],alpha[k])
      end
    end
  
  
  
  #=
      nterms = min(length(M),length(alpha))
      @inbounds for i = 1:length(M[1])
        x = W(0)
        @inbounds @simd for k = 1:nterms
          x += fct(M[k].T[w],alpha[k])
        end
        M[1].T[i] = x
      end
      =#
      return M[1]
  #  end
  end
  
  function tensorcombination!(M::Array{W,N}...;alpha::NTuple{G,W}=ntuple(i->W(1),length(M)),fct::Function=*) where {W <: Number, N, G}
  #  if fct == *
  #    return lineartensorcombination!(M...,alpha=alpha)
  #  else
  
    nterms = min(length(M),length(alpha))
    @inbounds @simd for i = 1:length(M[1])
      M[1][i] = fct(M[1][i],alpha[1])
    end
    @inbounds for k = 2:nterms
      @inbounds @simd for i = 1:length(M[1])
        M[1][i] += fct(M[k][i],alpha[k])
      end
    end
  
  #=
      nterms = min(length(M),length(alpha))
      @inbounds for w = 1:length(M[1])
        x = W(0)
        @inbounds @simd for k = 1:nterms
          x += fct(M[k][w],alpha[k])
        end
        M[1][w] = x
      end
  
      =#
      return M[1]
  #  end
  end

function tensorcombination!(alpha::NTuple{N,W},M::P...;fct::Function=*) where {P <: TensType, W <: Number, N}
  return tensorcombination!(M...,alpha=alpha,fct=fct)
end
export tensorcombination!

"""
  G = mult!(A,x)

Multiplies `x*A` (commutative) for dense or quantum tensors with output `G`

See also: [`*`](@ref) [`add!`](@ref) [`sub!`](@ref) [`div!`](@ref)
"""
function mult!(M::W, num::Number) where W <: TensType
  cnum = typeof(num) == eltype(M) ? num : convert(eltype(M),num)
  return tensorcombination!(M,alpha=(cnum,))
end

function mult!(num::Number,M::W) where W <: TensType
  return mult!(M,num)
end
export mult!

"""
  G = add!(A,B,x)

Adds `A + x*B` for dense or quantum tensors with output `G`

See also: [`+`](@ref) [`sub!`](@ref) [`mult!`](@ref) [`div!`](@ref)
"""
function add!(A::W, B::W, num::Number) where W <: TensType
  cnum = typeof(num) == eltype(A) ? num : convert(eltype(A),num)
  return tensorcombination!((eltype(A)(1),cnum),A,B)
end

"""
  G = add!(A,B)

Adds `A + B` for dense or quantum tensors with output `G`

See also: [`+`](@ref) [`sub!`](@ref) [`mult!`](@ref) [`div!`](@ref)
"""
function add!(A::W, B::W) where W <: TensType
  return add!(A,B,eltype(B)(1))
end
export add!

"""
  G = sub!(A,B,x)

Subtracts `A - x*B` for dense or quantum tensors with output `G`

See also: [`-`](@ref) [`add!`](@ref) [`sub!`](@ref) [`mult!`](@ref) [`div!`](@ref)
"""
function sub!(A::W,B::W,mult::Number) where W <: TensType
  cnum = typeof(mult) == eltype(A) ? mult : convert(eltype(A),mult)
  return add!(A,B,-cnum)
end

"""
  G = sub!(A,B)

Subtracts `A - B` for dense or quantum tensors with output `G`

See also: [`-`](@ref) [`add!`](@ref) [`sub!`](@ref) [`mult!`](@ref) [`div!`](@ref)
"""
function sub!(A::W,B::W) where W <: TensType
  return add!(A,B,eltype(A)(-1))
end
export sub!

"""
  G = div!(A,x)

Division by a scalar `A/x` (default x = 1) for dense or quantum tensors with output `G`

See also: [`/`](@ref) [`add!`](@ref) [`sub!`](@ref) [`mult!`](@ref)
"""
function div!(M::TensType, num::Number)
  cnum = typeof(num) == eltype(M) ? num : convert(eltype(M),num)
  return tensorcombination!(M,alpha=(cnum,),fct=/)
end
export div!

function norm!(M::TensType)
  return div!(M,norm(M))
end
#=
function norm!(A::TensType,M::TensType...)
  return (norm!(A),ntuple(i->norm!(M[i]),length(M))...)
end
=#
export norm!

import LinearAlgebra.+
"""
  G = +(A,B)

Adds two tensors `A` and `B` together with output `G`

See also: [`add!`](@ref)
"""
function +(A::TensType, B::TensType)
  mA,mB = checkType(A,B)
  return tensorcombination(mA,mB)
end

import LinearAlgebra.-
"""
  G = -(A,B)

Subtracts two tensors `A` and `B` (`A`-`B`) with output `G`

See also: [`sub!`](@ref)
"""
function -(A::TensType, B::TensType)
  return tensorcombination((eltype(A)(1),eltype(B)(-1)),A,B)
end

import Base.*
"""
  G = *(A,num)

Mutiplies a tensor `A` by a number `num` with output `G`

See also: [`mult!`](@ref)
"""
function *(num::Number, M::TensType)
  return tensorcombination(M,alpha=(num,))
end

"""
  G = *(num,A)

Mutiplies a tensor `A` by a number `num` with output `G`. Ensures commutativity of the operation

See also: [`mult!`](@ref)
"""
function *(M::TensType, num::Number)
  return num * M
end

import LinearAlgebra./
"""
  G = /(A,num)

Divides a tensor `A` by a number `num` with output `G`

See also: [`div!`](@ref)
"""
function /(M::TensType, num::Number)
  return tensorcombination(M,alpha=(num,),fct=/)
end

"""
  G = sqrt!(M)

Takes the square root of a dense tensor (new tensor created) or Qtensor (in-place) with output `G`
"""
function sqrt!(M::TensType)
  return tensorcombination!(M,alpha=(0.5,),fct=^)
end
export sqrt!

import Base.sqrt
"""
  G = sqrt(M)

Takes the square root of a tensor with output `G`

See also: [`sqrt`](@ref)
"""
function sqrt(M::TensType)
  return tensorcombination(M,alpha=(0.5,),fct=^)
end

"""
  G = inverse_element(x,zero)

Computes the inverse of the input element `x` up to some numerical zero value `zero` with output value `G`.  Below this value, zero is returned to avoid NaNs.
"""
function inverse_element(x::Number,zeronum::Real)
  return abs(x) > zeronum ? 1/x : 0.
end

"""
  G = invmat!(M[,zero=])

Creates inverse of a diagonal matrix in-place (dense matrices are copied) with output `G`; if value is below `zero`, the inverse is set to zero

See also: [`invmat!`](@ref)
"""
function invmat!(M::tens{W};zeronum::Float64=1E-16) where W <: Number
  rM = reshape(M.T,M.size)
  return tens{W}(inv(rM))
#  return tensorcombination!(M,alpha=(zeronum,),fct=inverse_element)
end

function invmat!(M::Array{W,2};zeronum::Float64=1E-16) where  W <: Number
#  rM = reshape(M.T,M.size)
  #=
  @inbounds for a = 1:length(M)
    M[a,a] = abs(M[a,a]) > zeronum ? 1/M[a,a] : 0.
  end=#
  return inv(M)
end
export invmat!

import Base.inv
"""
  G = invmat(M[,zero=])

Creates inverse of a diagonal matrix with output `G`; if value is below `zero`, the inverse is set to zero

See also: [`invmat!`](@ref)
"""
function invmat(M::TensType;zeronum::Float64=1E-16) 
  return invmat!(copy(M),zeronum=zeronum)#tensorcombination(M,alpha=(zeronum,),fct=inverse_element)
end
#=
function invmat(M::AbstractArray;zeronum::Float64=1E-16)
  outM = zeros(eltype(M),size(M))
  for a = 1:length(M)
    outM = abs(M[a,a]) > zeronum ? 1/M[a,a] : 0.
  end
  return invmat(copy(M))
end=#
export invmat

function exp!(A::Array{W,2},prefactor::Number) where W <: Number
  if !isapprox(prefactor,1)
    if W == typeof(prefactor)
      for x = 1:size(A,1)
        @inbounds @simd for y = 1:size(A,2)
          A[x,y] *= prefactor
        end
      end
    else
      A = A*prefactor
    end
  end
  #=
  maxA = maximum(A)
  if isnan(exp(A)) || isapprox(maxA,-Inf)
    expA = zeros(W,size(A)) + LinearAlgebra.I + A
  else =#
    expA = exp(A)
#  end
  return expA
end

function exp!(A::tens{W},prefactor::Number) where W <: Number
  X = reshape(A.T,A.size)
  expX = exp!(X,prefactor)
  newtype = typeof(W(1)*prefactor)
  if newtype == W
    A.T = reshape(expX,prod(size(A)))
  else
    A = tens(exp(expX,prefactor))
  end
  return A
end

function exp!(A::Array{W,2}) where W <: Number
  return exp!(A,W(1))
end

function exp!(A::tens{W}) where W <: Number
  return exp!(A,W(1))
end
export exp!

import Base.exp
function exp(A::Array{W,2},prefactor::Number) where W <: Number
  return exp!(copy(A),prefactor)
end

"""
  G = exp(A)

Exponentiate a matrix `A` from the `denstens` type with output `G`
"""
function exp(A::tens{W},prefactor::Number) where W <: Number
  X = reshape(A.T,A.size)
  newtype = typeof(W(1)*prefactor)
  return tens(newtype,exp(X,prefactor))
end

"""
  G = exp(alpha,beta)

Exponentiate a tridiagonal matrix from the two lists `alpha` (diagonal elements) and `beta` (off-diagonal elements) type with output `G`
"""
function exp(alpha::Array{W,1},beta::Array{Y,1},prefactor::Number) where {W <: Number, Y <: Number}
  d = length(alpha)
  if Y <: Complex || W <: Complex
    G = zeros(typeof(Y(1)*W(1)),length(alpha),length(alpha))
    @inbounds @simd for i = 1:d
      G[i,i] = alpha[i]
    end
    @inbounds @simd for i = 1:d-1
      G[i,i+1] = beta[i]
      G[i+1,i] = conj(beta[i])
    end
#    G = LinearAlgebra.Hermitian(C)
  else  
    G = LinearAlgebra.SymTridiagonal(alpha,beta)
  end
  return exp(G,prefactor)
end

#best implementation
"""
  G = exp(A)

Exponentiate a symmetric, tridiagonal matrix `A` with output `G`
"""
function exp(G::LinearAlgebra.SymTridiagonal{W, Vector{W}},prefactor::Number) where W <: Number
  D,U = LinearAlgebra.eigen(G)
  if typeof(prefactor) == W
    @inbounds for i = 1:length(D)
      D[i] = exp(prefactor * D[i])
    end
  else
    D = [exp(D[i]*prefactor) for i = 1:length(D)]
  end
  return U*LinearAlgebra.Diagonal(D)*U'
end

function exp(A::tens{W}) where W <: Number
  return exp(A,W(1))
end

function exp(alpha::Array{W,1},beta::Array{Y,1}) where {W <: Number, Y <: Number}
  return exp(alpha,beta,W(1))
end

function exp(G::LinearAlgebra.SymTridiagonal{W, Vector{W}}) where W <: Number
  return exp(G,W(1))
end





"""
  G = reshape!(M,a...[,merge=])

In-place reshape for dense tensors (otherwise makes a copy) with output `G`; can also make Qtensor unreshapable with `merge`, joining all grouped indices together

See also: [`reshape`](@ref)
"""
function reshape!(M::tens{W}, S::NTuple{N,intType};merge::Bool=false) where {N, W <: Number}
  newsize = Array{intType,1}(undef,N)
  @inbounds @simd for w = 1:N
    newsize[w] = S[w]
  end
  M.size = newsize
  return M #tens{W}(newsize,M.T)
end

function reshape!(M::tens{W}, S::intType...;merge::Bool=false) where W <: Number
  return reshape!(M,S)
end

function reshape!(M::tens{W}, S::Array{intType,1};merge::Bool=false) where W <: Number
  return reshape!(M,S...)
end

function reshape!(M::Array{W,P}, S::NTuple{N,intType};merge::Bool=false) where {N,P, W <: Number}
  return reshape(M,S)
end

function reshape!(M::Array{W,P}, S::intType...;merge::Bool=false) where {P,W <: Number}
  return reshape(M,S)
end

function reshape!(M::Array{W,P}, S::Array{intType,1};merge::Bool=false) where {P,W <: Number}
  return reshape!(M,S...)
end

"""
  G = reshape!(M,a[,merge=])

Similar to `reshape!` with an input tuple for new sizes of the tensor `G`, but can also reshape indices explicitly by specifying them in a vector of vectors.

# Example:

```julia
julia> A = rand(20,40,30);
julia> reshape!(A,800,30); #reshapes first indices together
julia> reshape!(A,[[1,2],[3]]); #same as above
```

See also: [`reshape`](@ref)
"""
function reshape!(M::tens{W}, S::Union{Array{Array{P,1},1},Tuple};merge::Bool=false) where {W <: Number, P <: Integer}
  newsize = ntuple(a->prod(b->size(M,b),S[a]),length(S))
  order = vcat(S...)
  pM = issorted(order) ? M : permutedims!(M,order)
  return reshape!(M,newsize)
end
export reshape!

import Base.reshape
"""
  G = reshape(M,a...[,merge=])

Reshape for dense tensors (other types make a copy) with output `G`; can also make Qtensor unreshapable with `merge`, joining all grouped indices together

See also: [`reshape!`](@ref)
"""
function reshape(M::tens{W}, S::intType...;merge::Bool=false) where W <: Number
  return reshape!(copy(M),S)
end

function reshape(M::tens{W}, S::Array{intType,1};merge::Bool=false) where W <: Number
  return reshape(copy(M),S...)
end

function reshape(M::Array{W,N}, S::Array{intType,1};merge::Bool=false) where {W <: Number, N}
  return reshape(copy(M),S...)
end

"""
  G = reshape(M,a[,merge=])

Similar to `reshape` with an input tuple for new sizes of the tensor `G`, but can also reshape indices explicitly by specifying them in a vector of vectors.

# Example:

```julia
julia> A = rand(20,40,30);
julia> reshape!(A,800,30); #reshapes first indices together
julia> reshape!(A,[[1,2],[3]]); #same as above
```

See also: [`reshape!`](@ref)
"""
function reshape(M::tens{W}, S::Array{Array{P,1},1};merge::Bool=false) where {W <: Number, P <: intType}
  return reshape!(copy(M),S)
end


"""
  G = unreshape!(M,S)

Same as `reshape!` but used for ease of reading code and also has new context with quantum numbers

See also: [`reshape!`](@ref)
"""
function unreshape!(M::AbstractArray, S::intType...;merge::Bool=false) where W <: Number
  return reshape(M,S...)
end

function unreshape!(M::tens{W}, S::intType...;merge::Bool=false) where W <: Number
  return reshape!(M,S)
end
export unreshape!

"""
  G = unreshape(M,S)

Same as `reshape` but used for ease of reading code and also has new context with quantum numbers

See also: [`reshape`](@ref)
"""
function unreshape(M::AbstractArray, S::intType...;merge::Bool=false) where W <: Number
  return reshape(M,S...)
end

function unreshape(M::tens{W}, S::intType...;merge::Bool=false) where W <: Number
  newM = tens{W}(M.size,M.T)
  return unreshape!(newM,S...)
end
export unreshape

import Base.permutedims!
"""
  G = permutedims!(A,[1,3,2,...])

Permute dimensions of a Qtensor in-place (no equivalent, but returns value so it can be used identically to `permutedims` on a dense tensor)

See also: [`permutedims`](@ref)
"""
function permutedims!(M::tens{W}, vec::Array{P,1}) where {W <: Number, P <: intType}
  return permutedims(M,(vec...,))
end

function permutedims!(M::tens{W}, vec::NTuple{N,intType}) where {N, W <: Number}
  return permutedims(M,vec)
end

function permutedims!(M::AbstractArray, vec::NTuple{N,intType}) where {N, W <: Number}
  return permutedims(M,vec)
end

function permutedims!(M::AbstractArray, vec::Array{intType,1}) where {N, W <: Number}
  return permutedims(M,vec)
end

import Base.permutedims
"""
  G = permutedims(A,[1,3,2,...])

Permutes dimensions of `A` (identical usage to dense `size` call)

See also: [`permutedims!`](@ref)
"""
function permutedims(M::Union{Array{W,G},tens{W}}, vec::Array{P,1}) where {W <: Number, P <: intType, G}
  return permutedims(M,(vec...,))
end

function permutedims(A::Array{W,G},iA::NTuple{G,intType}) where {W <: Number, G}
  if issorted(iA)
    out = A
  else
    Asizes = ntuple(w->size(A,w),G)
    newsizes = ntuple(w->Asizes[iA[w]],G)

    P = Array{W,G}(undef,newsizes...)

    permutedims!(P,A,iA,Asizes,newsizes)
  end
  return P
end

function permutedims!(P::Array{W,R},A::Union{Array{W,R},tens{W}},iA::NTuple{G,intType},Asizes::NTuple{G,intType},newsizes::NTuple{G,intType}) where {W <: Number, G, R}
  startind = 0
  @inbounds while startind < G && iA[startind+1] == startind+1
    startind += 1
  end

  permutevec = makepos(G)
  if startind > 0
    startindex = 1
    @inbounds @simd for w = 1:startind
      startindex *= newsizes[w]
      permutevec[w] = newsizes[w]
    end
    @inbounds @simd for w = 1:startindex
      P[w] = A[w]
    end
  else
    startindex = 0
  end

  d1 = 1
  @inbounds while iA[d1] != 1
    d1 += 1
  end
  
  notfirst = d1 > 1

  i = startindex
  @inbounds while i < length(A) 
    i += 1
    position_incrementer!(permutevec,Asizes)
    newvec = ntuple(w->permutevec[iA[w]],G)

    backZ = newvec[G]
    @inbounds @simd for w = G-1:-1:d1
      backZ -= 1
      backZ *= newsizes[w]
      backZ += newvec[w]
    end

    foreZ = notfirst ? newvec[d1-1] : 0
    @inbounds @simd for w = d1-2:-1:1
      foreZ -= 1
      foreZ *= newsizes[w]
      foreZ += newvec[w]
    end
    
    factor = 1
    @inbounds @simd for h = d1-1:-1:1
      factor *= newsizes[h]
    end

    if notfirst
      @inbounds for x = 0:Asizes[1]-1
        z = x + backZ
        z -= 1
        z *= factor
        z += foreZ

        P[z] = A[i+x]
      end
    else
      @inbounds for x = 0:Asizes[1]-1
        z = x + backZ
        P[z] = A[i+x]
      end
    end
    permutevec[1] = Asizes[1]
    i += Asizes[1]-1
  end
  nothing
end

function permutedims(A::tens{W},iA::NTuple{G,intType}) where {W <: Number, G}
  if issorted(iA)
    out = A
  else
    Asizes = ntuple(w->size(A,w),G)
    newsizes = ntuple(w->Asizes[iA[w]],G)

    psize = 1
    @inbounds @simd for w = 1:G
      psize *= Asizes[w]
    end
    P = Array{W,1}(undef,psize)

    permutedims!(P,A.T,iA,Asizes,newsizes)

    vecnewsizes = [newsizes[w] for w = 1:G]
    out = tens{W}(vecnewsizes,P)
  end
  return out
end

"""
  G = joinindex!(vec,A,B)

In-place joinindexenatation of tensors `A` (replaced for Qtensors only) and `B` along indices specified in `vec`
"""

function joinindex!(bareinds::intvecType,A::Union{tens{W},Array{W,N}},B::Union{tens{W},Array{W,N}}) where {W <: Number, N}
  inds = convIn(bareinds)

  nA = ndims(A)
  if typeof(A) <: denstens || typeof(B) <: denstens
    finalsize = Array{intType,1}(undef,nA)
    @inbounds @simd for w = 1:nA
      finalsize[w] = size(A,w)
    end
    for w = 1:length(inds)
      finalsize[inds[w]] += size(B,inds[w])
    end
  else
    finalsize = ntuple(w-> w in inds ? size(A,w) + size(B,w) : size(A,w),nA)
  end

  Asize = size(A)
  Bsize = size(B)

  

  if length(inds) > 1
    if typeof(A) <: denstens || typeof(B) <: denstens
      Csize = 1
      for w = 1:nA
        Csize *= finalsize[w]
      end
      C = zeros(W,Csize)
    else
      C = zeros(W,finalsize)
    end
  else
    if typeof(A) <: denstens || typeof(B) <: denstens
      Csize = 1
      for w = 1:nA
        Csize *= finalsize[w]
      end
      C = Array{W,1}(undef,Csize)
    else
      C = Array{W,nA}(undef,finalsize)
    end
  end
  Aloop!(C,A,finalsize,Asize)
  Bloop!(C,B,finalsize,Bsize,inds,Asize)

  if typeof(A) <: denstens || typeof(B) <: denstens
    C = tens{W}(finalsize,C)
  end
  return C
end

function Aloop!(C::Union{Array{W,nA},Array{W,1}},A::Union{tens{W},Array{W,nA}},finalsize::Union{NTuple{nA,intType},Array{intType,1}},Asize::NTuple{nA,intType}) where {nA, W <: Number}
  pos = makepos(nA)
  d1 = 1
#  @inbounds while d1 < nA && Asize[d1] == 1
#    d1 += 1
#  end
  dimA = size(A,d1)

  Aysize = 1
  @inbounds @simd for w = d1+1:nA
    Aysize *= size(A,w)
  end

  p = 0
  for y = 1:Aysize
    position_incrementer!(pos,Asize)
    backZ = pos[nA]
    @inbounds @simd for w = nA-1:-1:d1
      backZ -= 1
      backZ *= finalsize[w]
      backZ += pos[w]
    end
    @inbounds @simd for x = 0:dimA-1
      p += 1
      C[x + backZ] = A[p]
    end
    pos[d1] += dimA
  end
  nothing
end

function Bloop!(C::Union{Array{W,nA},Array{W,1}},B::Union{tens{W},Array{W,nA}},finalsize::Union{NTuple{nA,intType},Array{intType,1}},Bsize::NTuple{nA,intType},inds::NTuple{G,intType},Asize::NTuple{nA,intType}) where {nA, G, W <: Number}
  pos = makepos(nA) #makepos!(pos)
  Bpos = makepos(nA)

  d1 = 1
#  @inbounds while d1 < nA && Bsize[d1] == 1
#    d1 += 1
#  end
  dimB = size(B,d1)

  Bysize = 1
  @inbounds @simd for w = d1+1:nA
    Bysize *= size(B,w)
  end
  p = 0
  for y = 1:Bysize

    position_incrementer!(Bpos,Bsize)

    @inbounds @simd for w = 1:nA
      pos[w] = Bpos[w]
    end

    @inbounds @simd for w = 1:length(inds)
      pos[inds[w]] += Asize[inds[w]]
    end

    backZ = pos[nA]
    @inbounds @simd for w = nA-1:-1:d1
      backZ -= 1
      backZ *= finalsize[w]
      backZ += pos[w]
    end
    @inbounds @simd for x = 0:dimB-1
      p += 1
      C[x + backZ] = B[p]
    end
    Bpos[d1] += dimB
  end
  nothing
end
#=

function joinindex!(bareinds::intvecType,A::Union{tens{W},Array{W,N}},B::Union{tens{W},Array{W,N}}) where {W <: Number, N}

  inds = convIn(bareinds)
  
  nA = ndims(A)
  nB = nA
  finalsize = Array{intType,1}(undef,nA) #[size(A,i) for i = 1:nA]
  axesA = Array{UnitRange{intType},1}(undef,nA)
  axesout = Array{UnitRange{intType},1}(undef,nA)
  @inbounds @simd for i = 1:nA
    finalsize[i] = size(A,i)
    axesA[i] = 1:finalsize[i]
  end
  @inbounds @simd for i = 1:length(inds)
    a = inds[i]
    finalsize[a] += size(B,a)
  end
#  tupsize = ntuple(w->finalsize[w],nA)
#  if length(inds) > 1
    newTensor = zeros(W,finalsize...) #tupsize)
#  else
#    newTensor = Array{W,nA}(undef,tupsize)
#  end

  notinds = findnotcons(nA,inds)
  @inbounds @simd for i = 1:length(notinds)
    a = notinds[i]
    axesout[a] = 1:finalsize[a]
  end
  @inbounds @simd for i = 1:length(inds)
    a = inds[i]
    start = size(A,a)+1
    stop = finalsize[a]
    axesout[a] = start:stop
  end

  #better way to load tensors?
  #=@time=# @inbounds newTensor[axesA...] = A
  #=@time=# @inbounds newTensor[axesout...] = B

  if typeof(A) <: denstens || typeof(B) <: denstens
    newTensor = tens{W}(newTensor)
  end

  return newTensor
end
=#




function joinindex!(A::Array{S,N},B::Array{W,N}) where {W <: Number, S <: Number, N}
  return joinindex!(A,B,[i for i = 1:N])
end

function joinindex!(A::tens{S},B::tens{W}) where {W <: Number, S <: Number}
  return joinindex!(A,B,[i for i = 1:ndims(A)])
end

"""
  G = joinindex(vec,A,B...)

Concatenatation of tensors `A` and any number of `B` along indices specified in `vec`
"""
function joinindex(inds::intvecType,A::W,B::R) where {W <: TensType, R <: TensType}
  if typeof(A) <: densTensType
    C = A
  else
    C = copy(A)
  end
  return joinindex!(inds,C,B)
end

"""
  G = joinindex(A,B,vec)

Concatenatation of tensors `A` and any number of `B` along indices specified in `vec` to output `G`
"""
function joinindex(A::W,B::R,inds::intvecType) where {W <: TensType, R <: TensType}
  mA,mB = checkType(A,B)
  return joinindex(inds,mA,mB)
end
export joinindex

"""
  G = joinindex!(A,B,vec)

In-place joinindexenatation of tensors `A` and any number of `B` along indices specified in `vec` to output `G`
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
