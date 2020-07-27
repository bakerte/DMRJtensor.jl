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
    Module: tensor

Stores functions for the dense tensor (tens) type
"""
module tensor
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
  const intvecType = Union{intType,Array{intType,1},Array{intType,2},Tuple}
  export intvecType

  """
      denstens

  Abstract type for either dense tensors

  See also: [`tens`](@ref)
  """
  abstract type denstens end
  export denstens

  """
      tens{Z}

  regular tensor type; defined as `tens{Z}` for Z <: Number

  # Fields:
  + `size::Array{intType,1}`: size of base tensor (unreshaped)
  + `T::Array{Z,1}`: tensor reshaped into a vector

  See also: [`Qnum`](@ref) [`convertTensor`](@ref) [`checkflux`](@ref)
  """
  mutable struct tens{Z <: Number} <: denstens
    size::Array{intType,1} #the size of the tensor if it were represented densely
    T::Array{Z,1}
  end
  export tens

  function tens(;type::DataType=Float64)
    return tens{type}(intType[0],type[])
  end

  function tens(type::DataType)
    return tens{type}(intType[0],type[])
  end

  function tens{T}() where T <: DataType
    return tens(type=T)
  end

  function tens(T::DataType,P::Array{W,N}) where W <: Number where N
    sizeP = [size(P)...]
    if T != W
      rP = convert(Array{T,1},reshape(P,prod(sizeP)))
    else
      rP = reshape(P,prod(sizeP))
    end
    return tens{T}(sizeP,rP)
  end

  function tens(P::Array{W,N}) where W <: Number where N
    return tens(W,P)
  end

  function tens{W}(P::Array{W,N}) where W <: Number where N
    return tens(W,P)
  end

  import Base.rand
  """
      rand(A)

  generates a random tensor from inputting another tensor (rank-2) or from quantum number labels; can assign `arrows` for Qtensors if alternate arrow convention is requested
  """
  function rand(rho::denstens)
    return rand(eltype(rho), size(rho))
  end

  """
      convertQtens(T,Qt)

  Convert Qtensor `Qt` to type `T`
  """
  function convertTens(T::DataType, M::denstens)
    return tens{T}(copy(M.size),convert(Array{T,1}, M.T))
  end
  export convertTens

  """
      convertTensor(Qt)

  converts Qtensor (`Qt`) to dense array
  """
  function convertTensor(M::denstens)
    return reshape(M.T, size(M)...)
  end
  export convertTensor

  """
    to_Array(Qt)

  See: [`convertTensor`](@ref)
  """
  function to_Array(M::denstens)
    return reshape(M.T, size(M)...)
  end
  export to_Array

  import Base.copy
  """
      copy(Qt)

  Copies a Qtensor; `deepcopy` is inherently not type stable, so this function should be used instead
  """
  function copy(Qt::tens{T}) where {T <: Number}
    return tens{T}(copy(Qt.size),copy(Qt.T))
  end

  """
      showQtens(Qt[,show=])

  Prints fields of the Qtensor (`Qt`) out to a number of elements `show`; can also be called with `print` or `println`

  See also: [`Qtens`](@ref) [`print`](@ref) [`println`](@ref)
  """
  function showTens(M::denstens;show::intType = 4)
    println("printing regular tensor of type: ", typeof(M))
    println("size = ", convert(Array{intType,1}, M.size))
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
  function print(A::denstens...;show::intType = 4)
    showTens(A, show = show)
    nothing
  end

  import Base.println
  """
      println(A[,show=])

  Idential to `showQtens`

  See also: [`showQtens`](@ref) [`print`](@ref)
  """
  function println(A::denstens;show::intType = 4)
    showTens(A, show = show)
    print("\n")
    nothing
  end

  """
      genColType

  All types input into `getindex!` (UnitRange,intType,Array{intType,1},Colon)

  See also: [`getindex!`](@ref)
  """
  const genColType = Union{UnitRange,intType,Array{intType,1},Colon}
  export genColType

  """
      ind2pos(x,S)

  Converts a position vector `currpos` to a one-indexed index based on size `S`

  See also: [`ind2zeropos`](@ref) [`ind2pos`](@ref) [`ind2pos!`](@ref) [`zeropos2ind`](@ref) [`pos2ind`](@ref) [`pos2ind!`](@ref)
  """
  @inline function pos2ind(currpos::R,S::Array{Y,1}) where R <: Union{Array{X,1},NTuple{N,X} where N} where X <: Integer where Y <: Integer
    x = 0
    @simd for i = length(S):-1:2
      x += currpos[i]-1
      x *= S[i-1]
    end
    return currpos[1]+x
  end
  export pos2ind

  import Base.getindex
  """
      A[:,3:6,2,[1,2,4,8]]

  Finds selected elements of a Qtensor or dense tensor;
  
  #Note:
  + Any modification to the output of this function can make the orignal tensor invalid.
    If the orignal tensor is needed and you need to operate on the resulting tensor of this function, 
    do a copy of one of the two before hand. This will decouple them.
  + (For Qtensors): Always returns a Qtensor.  If you want one element, use the searchindex function below

  See also: [`searchindex`](@ref)
  """
  function getindex(M::denstens, a::genColType...)
    X = copy(M)
    return getindex!(X,a...)
  end

  function getindex!(C::tens{T}, a::genColType...) where {T <: Number}
    X = reshape(C.T,C.size...)
    Y = getindex!(X, a...)
    if typeof(Y) <: Number
      M = Y
    else
      Msize = [size(Y,a) for a = 1:ndims(Y)]
      MT = reshape(Y,prod(Msize))
      M = tens{eltype(MT)}(Msize,MT)
    end
    return M
  end
  export getindex!

  import Base.setindex!
  """
      setindex!(B,A,i...)

  puts `A` into the indices of `B` specified by `i` (functionality for tensor class)
  """
  function setindex!(B::denstens,A::denstens,a::genColType...)
    mB = to_Array(B)
    mA = to_Array(A)
    mB[a...] = mA
    B.T = reshape(mB,prod(size(mB)))
    nothing
  end

  """
      searchindex(C,a...)

  Find element of `C` that corresponds to positions `a`
  """
  function searchindex(C::denstens,a::intType...)::Number
    veca = [a[i] for i = 1:length(a)]
    w = pos2ind(veca,C.size)
    return C.T[w]
  end
  export searchindex

  function ArrayTens(A::U...) where U <: Union{denstens,Array}
    W = Array{denstens,1}(undef,length(A))
    for i = 1:length(A)
      if typeof(A[i]) <: Array
        W[i] = tens(A[i])
      else
        W[i] = A[i]
      end
    end
    return (W...,)
  end
  export ArrayTens

  #get the last index of a Qtensor (ex: A[:,1:end]...defines "end")
  import Base.lastindex
  """
      lastindex(Qtens,i)

  get the last index of a Qtensor (ex: A[:,1:end]...defines "end")
  """
  function lastindex(M::denstens, i::intType) #where Q <: qarray
    return M.size[i]
  end

  function checktensTypes(M::tens{A}, num::B) where {A <: Number, B <: Number}
    if A != B
      if B <: Complex && !(A <: Complex)
        X = convertTens(B,M)
        Y = num
      else
        X = M
        Y = convert(A,num)
      end
    else
      X = M
      Y = num
    end
    return X,Y
  end
  export checktensTypes

  """
      mult!(A,x)

  Multiplies `x*A` (commutative) for dense or quantum tensors

  See also: [`*`](@ref) [`add!`](@ref) [`sub!`](@ref) [`div!`](@ref)
  """
  function mult!(M::denstens, num::Number)::denstens
    newM,newnum = checktensTypes(M,num)
    @simd for i = 1:size(newM.T, 1)
      newM.T[i] *= newnum
    end
    return newM
  end
  export mult!

  """
      add!(A,B[,x])

  Adds `A + x*B` (default x = 1) for dense or quantum tensors

  See also: [`+`](@ref) [`sub!`](@ref) [`mult!`](@ref) [`div!`](@ref)
  """
  function add!(A::denstens, B::denstens, mult::Number)::denstens
    mA,mB = ArrayTens(A,B)
    X,newnum = checktensTypes(mA,mult)
    @simd for i = 1:length(X.T)
      X.T[i] += newnum*mB.T[i]
    end
    return X
  end
  export add!

  """
      sub!(A,B[,x])

  Subtracts `A - x*B` (default x = 1) for dense or quantum tensors

  See also: [`-`](@ref) [`add!`](@ref) [`sub!`](@ref) [`mult!`](@ref) [`div!`](@ref)
  """
  function sub!(A::denstens,B::denstens,mult::Number)
    return add!(A,B,-mult)
  end

  function sub!(A::denstens,B::denstens)
    return add!(A,B,-1.)
  end
  export sub!

  """
      div!(A,x)

  Division by a scalar `A/x` (default x = 1) for dense or quantum tensors

  See also: [`/`](@ref) [`add!`](@ref) [`sub!`](@ref) [`mult!`](@ref)
  """
  function div!(A::denstens, num::Number)::denstens
    X,newnum = checktensTypes(A,num)
    @simd for w = 1:size(X.T, 1)
      X.T[w] /= newnum
    end
    return X
  end
  export div!

  import LinearAlgebra.+
  """
      +(A,B)

  adds two tensors `A` and `B` together

  See also: [`add!`](@ref)
  """
  function +(A::denstens, B::denstens)::denstens
    C = copy(A)
    return add!(C, B)
  end

  import LinearAlgebra.-
  """
      -(A,B)

  subtracts two tensors `A` and `B` (`A`-`B`)

  See also: [`sub!`](@ref)
  """
  function -(A::denstens, B::denstens)::denstens
    C = copy(A)
    return sub!(C, B)
  end

  import Base.*
  """
      *(A,num)

  mutiplies a tensor `A` by a number `num`

  See also: [`mult!`](@ref)
  """
  function *(num::Number, M::denstens)::denstens
    return mult!(copy(M), num)
  end

  function *(M::denstens, num::Number)::denstens
    return num * M
  end

  import LinearAlgebra./
  """
      /(A,num)

  divides a tensor `A` by a number `num`

  See also: [`div!`](@ref)
  """
  function /(A::denstens, num::Number)::denstens
    R = copy(A)
    return div!(R, num)
  end

  import Base.sqrt
  """
      sqrt(M)

  Takes the square root of a tensor

  See also: [`sqrt`](@ref)
  """
  function sqrt(M::denstens)::denstens
    newM = copy(M)
    return sqrt!(newM)
  end

  """
      sqrt!(M)

  Takes the square root of a dense tensor (new tensor created) or Qtensor (in-place)
  """
  function sqrt!(M::denstens)::denstens
    @simd for a = 1:length(M.T)
      M.T[a] = sqrt.(M.T[a])
    end
    return M
  end
  export sqrt!

  function checkflux(M::denstens)
    nothing
  end

  """
      invmat!(Qt[,zero=])

  Creates inverse of a diagonal matrix in place (dense matrices are copied anyway);
  if value is below `zero`, the inverse is set to zero

  See also: [`invmat`](@ref)
  """
  function invmat!(M::denstens;zero::Float64=1E-16)
#    @assert(size(Qt.QnumMat,1) == 2)
    for a = 1:size(M.T,1)
      M.T[a] = abs(M.T[a]) > zero ? 1/M.T[a] : 0.
    end
    return M
  end
  export invmat!

  import Base.sum
  """
      sum(A)

  sum elements of a Qtensor
  """
  function sum(A::denstens)::Number
    return sum(A.T)
  end

  import LinearAlgebra.norm
  """
      norm(A)

  Froebenius norm of a Qtensor
  """
  function norm(A::denstens)::Number
    return norm(A.T)
  end
  export norm

  import Base.eltype
  """
      eltype(A)

  element type of a Qtensor (i.e., `T` field)

  See also: [`Qtens`](@ref)
  """
  function eltype(A::tens{T}) where {T <: Number}
    return T
  end

  import LinearAlgebra.conj
  """
      conj(A)

  conjugates a Qtensor by creating a copy

  See also: [`conj!`](@ref)
  """
  function conj(currQtens::denstens)::denstens
    newQtens = copy(currQtens)
    conj!(newQtens)
    return newQtens
  end

  import LinearAlgebra.conj!
  """
      conj!(A)

  conjugates a Qtensor in place

  See also: [`conj`](@ref)
  """
  function conj!(currQtens::denstens)
    conj!(currQtens.T)
    return currQtens
  end

  function conj(currQtens::denstens)
    newtens = copy(currQtens)
    conj!(newtens)
    return newtens
  end

  import LinearAlgebra.size
  import Base.size
  """
      size(A[,i])

  gets the size of a Qtensor (identical usage to dense `size` call)
  """
  function size(A::denstens)
    return A.size
  end

  function size(A::denstens,i::Integer)
    return A.size[i]
  end

  import LinearAlgebra.ndims
  """
      ndims(A)

  number of dimensions of a Qtensor (identical usage to dense `size` call)
  """
  function ndims(A::denstens)
    return length(A.size)
  end

  import Base.permutedims
  """
      permutedims(A,[1,3,2,...])

  permutes dimensions of `A`  (identical usage to dense `size` call)

  See also: [`permutedims!`](@ref)
  """
  function permutedims(M::tens{W}, vec::Array{intType,1}) where W <: Number #::denstens
    return permutedims!(copy(M),vec)
  end

  import Base.permutedims!
  """
      permutedims!(A,[1,3,2,...])

  permute dimensions of a Qtensor in-place (no equivalent, but returns value so it can be used identically to `permutedims` on a dense tensor)

  See also: [`permutedims`](@ref)
  """
  function permutedims!(M::tens{W}, vec::Array{intType,1}) where W <: Number #::denstens
    rM = reshape(M.T,M.size...)
    xM = permutedims(rM, vec)
    return tens(W,xM)
  end

  import Base.zero
    """Like the default function zero(t::Array), return an object with the same properties containing only zeros."""
  function zero(M::tens{T}) where T <: Number
    return tens(zeros(T,M.size...))
  end

  """
      reshape!(M,a...[,merge=])

  In-place reshape for dense tensors (otherwise makes a copy); can also make Qtensor unreshapable with `merge`, joining all grouped indices together

  See also: [`reshape`](@ref)
  """
  function reshape!(M::tens{W}, S::intType...;merge::Bool=false) where W <: Number
    M.size = [S[i] for i = 1:length(S)]
    return M
  end
  export reshape!

  import Base.reshape
  function reshape(M::tens{W}, S::intType...;merge::Bool=false) where W <: Number
    return reshape!(copy(M),S...)
  end

  function reshape!(M::tens{W}, S::Array{Array{Int64,1},1};merge::Bool=false) where W <: Number
    newsize = [prod(b->size(M,b),S[a]) for a = 1:length(S)]
    return reshape!(copy(M),newsize...)
  end

  function reshape(M::tens{W}, S::Array{Array{Int64,1},1};merge::Bool=false) where W <: Number
    return reshape!(copy(M),S)
  end

  function unreshape!(M::tens{W}, S::intType...;merge::Bool=false) where W <: Number
    M.size = intType[S[a] for a = 1:length(S)]
    return M
  end
  export unreshape!

  function unreshape(M::tens{W}, S::intType...;merge::Bool=false) where W <: Number
    return unreshape!(copy(M),S...)
  end
  export unreshape

  import Base.exp
  function exp(A::tens{W}) where W <: Number
    X = reshape(A.T,A.size...)
    return tens(W,exp(X))
  end
end
