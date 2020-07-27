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
    Module: Qtensor

Stores functions for the quantum number tensor (Qtensor) type

See also: [`Qtask`](@ref)
"""
module Qtensor
using ..tensor
using ..QN
using SparseArrays
import LinearAlgebra

  """
      qarray

  Abstract type for either Qtensors

  See also: [`Qtens`](@ref)
  """
  abstract type qarray end
  export qarray

  """
      TensType

  Abstract type for either Qtensors or AbstractArrays (dense tensors)

  See also: [`Qtens`](@ref)
  """
  const TensType = Union{qarray,denstens,AbstractArray}
  export TensType

  """
      checkType(A,B)
  
  Checks types between `A` and `B` to make uniform (ex: Array and denstens converts to two denstens)
  """
  function checkType(A::R,B::S) where {R <: Union{Number,TensType}, S <: Union{Number,TensType}}
    if typeof(A) <: denstens || typeof(B) <: denstens
      if typeof(A) <: denstens && !(typeof(B) <: denstens)
        mA = A
        mB = tens(typeof(B) <: Union{qarray,Array} ? B : Array(B))
      elseif !(typeof(A) <: denstens) && typeof(B) <: denstens
        mA = tens(typeof(A) <: Union{qarray,Array} ? A : Array(A))
        mB = B
      else
        mA = A
        mB = B
      end
    else
      mA = typeof(A) <: Union{qarray,Array} ? A : Array(A)
      mB = typeof(B) <: Union{qarray,Array} ? B : Array(B)
    end
    return mA,mB
  end
  export checkType

  """
      Qtens{T,Q}

  Qtensor; defined as `Qtens{T,Q}` for T <: Number and Q <: Qnum

  # Fields:
  + `size::Array{intType,1}`: size of base tensor (unreshaped)
  + `Qsize::Array{Array{intType,1},1}`: stores original indices of the vector as a vector of vectors, ex: [[1 2],[3]] means that 1 and 2 were combined
  + `T::Array{Z,1}`: Array containing non-zero blocks' values of the tensor
  + `ind::Array{intType,1}`: indices of the stored values
  + `QnumMat::Array{Array{Q,1},1}`: quantum numbers on each index
  + `QnumSum::Array{Array{Q,1},1}`: summary of quantum numbers on each index
  + `flux::Q`: total quantum number flux on tensor
  + `conjugated::Bool`: toggle to conjugate tensor without flipping all fluxes

  See also: [`Qnum`](@ref) [`convertTensor`](@ref) [`checkflux`](@ref)
  """
  mutable struct Qtens{Z <: Number,Q <: Qnum} <: qarray
    size::Array{intType,1} #the size of the tensor if it were represented densely
    Qsize::Array{Array{intType,1},1} #stores original indices of the vector as a vector of vectors, ex: [[1 2],[3]] means that 1 and 2 were combined
    #^This is an array since it can change on reshape
    T::Array{Z,1}
    ind::Array{intType,1}
    QnumMat::Array{Array{Q,1},1} #quantum numbers on each index
    QnumSum::Array{Array{Q,1},1} #summary of indices on each index
    flux::Q #sum of QNs on all other indices, acts like an extra index with an inward arrow (that is never contractd over)
  end
  export Qtens

  """
      Qtens{Z,Q}()

  Default initializer to an empty tensor with type `Z` and quantum number `Q`
  """
  function Qtens{Z,Q}() where {Z <: Number,Q <: Qnum}
    return Qtens([[Q()]])
  end
 
  """
      Qtens(Qlabels[,arrows,Type=])

  Creates empty `Qtens` with array type `Type` (default Float64), quantum number labels given by `Qlabels` and arrow convention `arrows`; can be conjugated (`conjugated`)
  """
  function Qtens(Qlabels::Array{Array{W,1},1}, arrows::U...;Type::DataType=Float64)::qarray where {W <: Qnum, U <: Union{Bool,Array{Bool,1}}}
    if length(arrows) > 0
      QnumMat = [arrows[1][a] ? Qlabels[a] : inv.(Qlabels[a]) for a = 1:length(arrows[1])]
    else
      QnumMat = Qlabels
    end

    T = Type[]
    ind = intType[]
    flux = typeof(QnumMat[1][1])()
    return Qtens(T, ind, QnumMat, flux)
  end

  """
      Qtens(T,ind,QnumMat[,arrows])

  constructor for Qtensor with non-zero values `T`, indices `ind`, quantum numbers `QnumMat`; optionally assign `arrows` for indices or conjugate the tensor (`conjugated`)
  """
  function Qtens(T::Array{Z,1}, ind::Array{intType,1}, QnumMat::Array{Array{Q,1},1},
                  arrows::U...;conjugated::Bool=false)::qarray where {Z <: Number,Q <: Qnum, U <: Union{Bool,Array{Bool,1}}}
    if length(ind) > 0
      currpos = ind2pos(ind[1])
      if length(arrows) == 0
        thisQN = [QnumMat[a][currpos[a]] for a = 1:length(QnumMat)]
      else
        thisQN = [arrows[1][a] ? QnumMat[a][currpos[a]] : inv(QnumMat[a][currpos[a]]) for a = 1:length(QnumMat)]
      end
      flux = sum(thisQN)
    else
      flux = Q()
    end
    newsize = [length(QnumMat[a]) for a = 1:length(QnumMat)]
    return Qtens{Z,Q}(newsize,[[i] for i = 1:size(QnumMat, 1)], T, ind, QnumMat, unique.(QnumMat), flux)
  end

  """
      Qtens(T,ind,QnumMat,flux[,arrows])

  constructor for Qtensor with non-zero values `T`, indices `ind`, total flux `flux`, quantum numbers `QnumMat`; optionally assign `arrows` for indices or conjugate the tensor (`conjugated`)
  """
  function Qtens(T::Array{Z,1}, ind::Array{intType,1}, QnumMat::Array{Array{Q,1},1},
                  flux::Q,arrows::U...)::qarray where {Z <: Number,Q <: Qnum, U <: Union{Bool,Array{Bool,1}}}
    if length(arrows) == 0
      newQnumMat = QnumMat
    else
      newQnumMat = [arrows[1][a] ? QnumMat[a] : inv.(QnumMat[a]) for a = 1:length(QnumMat)]
    end
    newsize = [length(QnumMat[a]) for a = 1:length(QnumMat)]
    return Qtens{Z,Q}(newsize,[[i] for i = 1:size(QnumMat, 1)], T, ind, newQnumMat, unique.(newQnumMat), flux)
  end

  """
      Qtens(operator,QnumMat[,Arrows,zero=])

  Creates a dense `operator` as a Qtensor with quantum numbers `QnumMat` on each index (`Array{Array{Q,1},1}`) according to arrow convention `Arrows` and all elements equal to `zero` are not included (default 0)
  """
  function Qtens(operator::AbstractArray,Qlabels::Array{Array{Q,1},1},Arrows::U...;zero::Number=0.) where {Q <: Qnum, U <: Union{Bool,Array{Bool,1}}}
    if length(Arrows) > 0
      QnumMat = [Arrows[1][a] ? Qlabels[a] : inv.(Qlabels[a]) for a = 1:length(Qlabels)]
    else
      QnumMat = Qlabels
    end
    coords = findall(x->abs(x)>zero,operator)
    thisrank = ndims(operator)
    thisflux = sum(a->QnumMat[a][coords[1][a]],1:thisrank)
    newsize = intType[size(operator,a) for a = 1:thisrank]
    newT = Array{eltype(operator),1}(undef,length(coords))
    newpos = Array{intType,1}(undef,length(coords))
    let coords = coords, thisrank=thisrank, operator = operator,newsize = newsize, newT = newT, newpos = newpos
      Threads.@threads  for w = 1:length(coords)
        currpos = [coords[w][p] for p = 1:thisrank]
        newT[w] = operator[currpos...]
        newpos[w] = pos2ind(currpos,newsize)
      end
    end
    newQtens = Qtens(newT,newpos,QnumMat,thisflux)
    return newQtens
  end

  """
      Qtens(newsize,T,ind,QnumMat,flux[,arrows])

  constructor for Qtensor with unreshaped size of `newsize`, non-zero values `T`, indices `ind`, total flux `flux`, quantum numbers `QnumMat`; optionally assign `arrows` for indices or conjugate the tensor (`conjugated`)
  """
  function Qtens(newsize::Array{intType,1},T::Array{Z,1}, ind::Array{intType,1}, QnumMat::Array{Array{Q,1},1},
    flux::Q,arrows::U...)::qarray where {Z <: Number,Q <: Qnum, U <: Union{Bool,Array{Bool,1}}} #, Arrows::Array{Bool,1})::qarray where {Z <: Number,Q <: Qnum}
    if length(arrows) == 0
      newQnumMat = QnumMat
    else
      newQnumMat = [arrows[1][a] ? QnumMat[a] : inv.(QnumMat[a]) for a = 1:length(QnumMat)]
    end
  return Qtens{Z,Q}(newsize,[[i] for i = 1:size(QnumMat, 1)], T, ind, newQnumMat, unique.(newQnumMat), flux)#, Arrows)
  end

  """
      Qtens(operator,Qlabels[,Arrows,zero=])

  Creates a dense `operator` as a Qtensor with quantum numbers `Qlabels` common to all indices (`Array{Q,1}`) according to arrow convention `Arrows` and all elements equal to `zero` are not included (default 0)
  """
  function Qtens(operator::AbstractArray,Qlabels::Array{Q,1},Arrows::U...;zero::Number=0.) where {Q <: Qnum, U <: Union{Bool,Array{Bool,1}}}
    return Qtens(operator,[Qlabels for a = 1:ndims(operator)],Arrows...,zero=zero)
  end

  import Base.rand
  """
      rand(A[,arrows])

  generates a random tensor from inputting another tensor (rank-2) or from quantum number labels; can assign `arrows` for Qtensors if alternate arrow convention is requested
  """
  function rand(rho::Array)
    return rand(eltype(rho), size(rho))
  end

  function rand(Qlabels::Array{Array{Q,1},1}, arrows::Array{Bool,1})::qarray where Q <: Qnum
    newQlabels = Array{W,1}[arrows[a] ? Qlabels[a] : inv.(Qlabels[a]) for a = 1:length(arrows)]
    return rand(newQlabels)
  end

  function rand(Qlabels::Array{Array{Q,1},1})::qarray where Q <: Qnum
    return Qtens(Qtens(Qlabels))
  end

  function rand(currQtens::qarray)::qarray where W <: Qnum
    return Qtens(currQtens)
  end

  function rand(A::AbstractArray)
    return rand(size(A)...)
  end

  function rand(A::denstens)
    return tens(rand(size(A)...))
  end

  """
      convertQtens(T,Qt)

  Convert Qtensor `Qt` to type `T`
  """
  function convertQtens(T::DataType, Qt::Qtens{Z,Q})::qarray where {Z <: Number, Q <: Qnum}
    return Qtens{T,Q}(copy(Qt.size),copy(Qt.Qsize),convert(Array{T,1}, Qt.T),copy(Qt.ind),
                      copy(Qt.QnumMat),copy(Qt.QnumSum),copy(Qt.flux))
   end
  export convertQtens

import ..tensor.checkflux
"""
    checkflux(Qt[,silent=])

Debug tool: checks all non-zero elements obey flux conditions in Qtensor (`Qt`); print element by element with `silent`
"""
function checkflux(Qt::Qtens;silent::Bool = true)
  totflux = Qt.flux
  condition = true
  totsize = prod(Qt.size)
  uniques = unique(Qt.ind)
  if size(uniques,1) != size(Qt.ind,1)
    error("duplicate integers")
  end
  for i = 1:size(Qt.ind, 1)
    if isnan(Qt.ind[i])
      error("index $i is not a number")
    end
    if Qt.ind[i] == Inf
      error("index $i is infinity")
    end
    thispos = ind2pos(Qt.ind[i], Qt.size)

    checkQN = sum(a->Qt.QnumMat[a][thispos[a]], 1:size(thispos, 1))
    if !silent
      theseQNs = [Qt.QnumMat[a][thispos[a]] for a = 1:size(thispos, 1)]
      println("showing: ",i," ", Qt.ind[i], " ", thispos, " ", theseQNs," sum = ", checkQN, " for a flux of ", Qt.flux)
    end
    if totflux != checkQN
      theseQNs = [Qt.QnumMat[a][thispos[a]] for a = 1:size(thispos, 1)]
      println("problem! ",i," ", Qt.ind[i], " ", thispos, " ", theseQNs, " ", checkQN, " for a flux of ", Qt.flux)
      condition = false
    end
  end
  if condition
    checkSum = unique.(Qt.QnumSum)
    for i = 1:size(checkSum,1)
      if size(checkSum[i],1) != size(Qt.QnumSum[i],1)
        error("bad summary (inequivalent elements...perhaps not a full summary) on index $i !")
      end
      unique(Qt.QnumSum[i]) == checkSum[i]
      if size(checkSum[i],1) != size(Qt.QnumSum[i],1)
        error("non-matching summary on index $i !")
      end
    end
      println("PASSED \n")
    else
      error("problems \n")
    end
    nothing
  end
  export checkflux

  function checkflux(Qt::AbstractArray;silent::Bool = true)
    nothing
  end

  import .tensor.convertTensor
  """
      convertTensor(Qt)

  converts Qtensor (`Qt`) to dense array
  """
  function convertTensor(Qt::qarray)::AbstractArray
    A = zeros(eltype(Qt.T), Qt.size...)
    numthreads = Threads.nthreads()
    currpos = Array{Int64,1}[Array{intType,1}(undef,length(Qt.size)) for i = 1:numthreads]
    let A = A, Qt = Qt, currpos = currpos
      Threads.@threads for i = 1:size(Qt.ind, 1)
        thisthread = Threads.threadid()
        ind2pos!(currpos,thisthread,Qt.ind,i, Qt.size)
        A[currpos[thisthread]...] = Qt.T[i]
      end
    end
    return reshape(A, size(Qt)...)
  end

  function convertTensor(Qt::AbstractArray)
    return Qt
  end

  import .tensor.to_Array
  """
    to_Array(Qt)

  See: [`convertTensor`](@ref)
  """
  function to_Array(T::Qtens)
    return convertTensor(T)
  end

  import Base.copy
  """
      copy(Qt)

  Copies a Qtensor; `deepcopy` is inherently not type stable, so this function should be used instead
  """
  function copy(Qt::Qtens{T,Q}) where {T <: Number, Q <: Qnum}
    copyQtQsize = Array{intType,1}[copy(Qt.Qsize[a]) for a = 1:length(Qt.Qsize)]
    return Qtens{T,Q}(copy(Qt.size),copyQtQsize,copy(Qt.T),copy(Qt.ind),
                      copy(Qt.QnumMat),copy(Qt.QnumSum),copy(Qt.flux))
  end

  """
      showQtens(Qt[,show=])

  Prints fields of the Qtensor (`Qt`) out to a number of elements `show`; can also be called with `print` or `println`

  See also: [`Qtens`](@ref) [`print`](@ref) [`println`](@ref)
  """
  function showQtens(Qtens::qarray;show::intType = 4)
    println("printing Qtens of type: ", typeof(Qtens))
    println("size = ", convert(Array{intType,1}, Qtens.size))
    println("Qsize = ", convert(Array{Array{intType,1},1}, Qtens.Qsize))
    maxshow = min(show, size(Qtens.T, 1))
    maxBool = show < size(Qtens.T, 1)
    println("T = ", Qtens.T[1:maxshow], maxBool ? "..." : "")
    println("inds = ", convert(Array{intType,1}, Qtens.ind)[1:maxshow], maxBool ? "..." : "")
    println("QnumMat = ")
    for i = 1:size(Qtens.QnumMat, 1)
      maxshow = min(show, size(Qtens.QnumMat[i], 1))
      maxBool = show < size(Qtens.QnumMat[i], 1)
      println(i, ": ", Qtens.QnumMat[i][1:maxshow], maxBool ? "..." : "")
    end
    println("QnumSum = ")
    for i = 1:size(Qtens.QnumSum, 1)
      maxshow = min(show, size(Qtens.QnumSum[i], 1))
      maxBool = show < size(Qtens.QnumSum[i], 1)
      println(i, ": ", Qtens.QnumSum[i][1:maxshow], maxBool ? "..." : "")
    end
    println("flux = ", Qtens.flux)
    nothing
  end
  export showQtens

  import Base.print
  """
      print(A[,show=])

  Idential to `showQtens`

  See also: [`showQtens`](@ref) [`println`](@ref)
  """
  function print(A::qarray...;show::intType = 4)
    showQtens(A, show = show)
    nothing
  end

  import Base.println
  """
      println(A[,show=])

  Idential to `showQtens`

  See also: [`showQtens`](@ref) [`print`](@ref)
  """
  function println(A::qarray;show::intType = 4)
    showQtens(A, show = show)
    print("\n")
    nothing
  end

###################################################
###################################################
###################################################

  """
      ind2zeropos(x,S)

  Converts an index `x` to a zero-indexed position vector with total size `S`

  See also: [`ind2zeropos`](@ref) [`ind2pos`](@ref) [`ind2pos!`](@ref) [`zeropos2ind`](@ref) [`pos2ind`](@ref) [`pos2ind!`](@ref)
  """
  @inline function ind2zeropos(x::intType,S::Array{X,1}) where X <: Integer
    currpos = Array{intType,1}(undef, size(S,1))
    currpos[1] = x-1
    @simd for j = 1:length(S)-1
      currpos[j+1] = fld(currpos[j],S[j])
      currpos[j] %= S[j]
    end
    currpos[size(S,1)] %= S[size(S,1)]
    return currpos
  end
  export ind2zeropos

  """
      ind2zeropos!(currpos,thisthread,x,i,S)

  in-place converts an index `x[i]` to a zero-indexed position vector with total size `S` stored in `currpos` (parallelized to `thisthread` worker)

  See also: [`ind2zeropos`](@ref) [`ind2pos`](@ref) [`ind2pos!`](@ref) [`zeropos2ind`](@ref) [`pos2ind`](@ref) [`pos2ind!`](@ref)
  """
  @inline function ind2zeropos!(currpos::Array{Array{Int64,1},1},thisthread::Integer,x::Array{intType,1},i::Integer,S::Array{Int64,1})
    currpos[thisthread][1] = x[i] - 1
    @simd for k = 1:size(S, 1) - 1
      currpos[thisthread][k + 1] = fld(currpos[thisthread][k], S[k])
      currpos[thisthread][k] %= S[k]
    end
    currpos[thisthread][size(S, 1)] %= S[size(S, 1)]
    nothing
  end
  export ind2zeropos!

  """
      ind2pos(x,S)

  Converts an index `x` to a one-indexed position vector with total size `S`

  See also: [`ind2zeropos`](@ref) [`ind2pos`](@ref) [`ind2pos!`](@ref) [`zeropos2ind`](@ref) [`pos2ind`](@ref) [`pos2ind!`](@ref)
  """
  @inline function ind2pos(x::intType,S::Array{X,1}) where X <: Integer
    return ind2zeropos(x,S) .+ 1
  end
  export ind2pos

  import ..tensor.pos2ind
  """
      pos2ind(reorder,x,S)

  Reorders (according to `reorder`) and converts a position vector `currpos` to a one-indexed index based on size `S`

  See also: [`ind2zeropos`](@ref) [`ind2pos`](@ref) [`ind2pos!`](@ref) [`zeropos2ind`](@ref) [`pos2ind`](@ref) [`pos2ind!`](@ref)
  """
  @inline function pos2ind(reorder::Array{X,1},currpos::R,S::Array{Y,1}) where R <: Union{Array{X,1},NTuple{N,X} where N} where X <: Integer where Y <: Integer
    x = 0
    @simd for i = length(S):-1:2
      x += currpos[reorder[i]]-1
      x *= S[reorder[i-1]]
    end
    return currpos[reorder[1]]+x
  end

  """
      zeropos2ind(x,S)

  Converts a position vector `currpos` to a one-indexed index based on size `S`

  See also: [`ind2zeropos`](@ref) [`ind2pos`](@ref) [`ind2pos!`](@ref) [`zeropos2ind`](@ref) [`pos2ind`](@ref) [`pos2ind!`](@ref)
  """
  @inline function zeropos2ind(currpos::R,S::Array{Y,1}) where R <: Union{Array{X,1},NTuple{N,X} where N} where X <: Integer where Y <: Integer
    x = 0
    @simd for i = length(S):-1:2
      x += currpos[i]
      x *= S[i-1]
    end
    return currpos[1]+x+1
  end
  export zeropos2ind

  """
    zeropos2ind(reorder,x,S)

  Reorders (according to `reorder`) and converts a position vector `currpos` to a zero-indexed index based on size `S`

  See also: [`ind2zeropos`](@ref) [`ind2pos`](@ref) [`ind2pos!`](@ref) [`zeropos2ind`](@ref) [`pos2ind`](@ref) [`pos2ind!`](@ref)
  """
  @inline function zeropos2ind(vec::Array{Y,1},currpos::R,S::Array{Y,1}) where R <: Union{Array{X,1},NTuple{N,X} where N} where X <: Integer where Y <: Integer
    x = 0
    @simd for i = length(S):-1:2
      x += currpos[reorder[i]]
      x *= S[reorder[i-1]]
    end
    return currpos[reorder[1]]+x+1
  end
  export zeropos2ind

  """
      ind2pos!(currpos,index,S)

  converts `index` to a position stored in `currpos` with tensor size `S`

  #Arguments:
  +`currpos::Array{Z,1}`: input position
  +`index::intType`: index to convert
  +`S::Array{W,1}`: size of tensor to convert from

  See also: [`ind2zeropos`](@ref) [`ind2pos`](@ref) [`ind2pos!`](@ref) [`zeropos2ind`](@ref) [`pos2ind`](@ref) [`pos2ind!`](@ref)
  """
  @inline function ind2pos!(currpos::Array{X,1},index::intType,S::Array{Z,1}) where X <: Integer where Y <: Integer where Z <: Integer
    currpos[1] = index-1
    @simd for j = 1:size(S,1)-1
      currpos[j+1] = fld(currpos[j],S[j])
      currpos[j] = (currpos[j]) % S[j] + 1
    end
    currpos[size(S,1)] = currpos[size(S,1)] % S[size(S,1)] + 1
    nothing
  end

  """
      ind2pos!(currpos,x,index,S)

  converts `x[index]` to a position stored in `currpos` with tensor size `S`

  #Arguments:
  +`currpos::Array{Z,1}`: input position
  +`x::Array{Y,1}`: vector of intput integers to convert
  +`index::intType`: index of input position of `x`
  +`S::Array{W,1}`: size of tensor to convert from

  See also: [`ind2zeropos`](@ref) [`ind2pos`](@ref) [`ind2pos!`](@ref) [`zeropos2ind`](@ref) [`pos2ind`](@ref) [`pos2ind!`](@ref)
  """
  @inline function ind2pos!(currpos::Array{X,1},x::Array{Y,1},index::intType,S::Array{Z,1}) where X <: Integer where Y <: Integer where Z <: Integer
    currpos[1] = x[index]-1
    @simd for j = 1:size(S,1)-1
      currpos[j+1] = fld(currpos[j],S[j])
      currpos[j] = (currpos[j]) % S[j] + 1
    end
    currpos[size(S,1)] = currpos[size(S,1)] % S[size(S,1)] + 1
    nothing
  end

  """
      ind2pos!(currpos,k,x,index,S)

  converts `x[index]` to a position stored in `currpos` (parallelized) with tensor size `S`

  #Arguments:
  +`currpos::Array{Z,1}`: input position
  +`k::intType`: current thread for `currpos`
  +`x::Array{Y,1}`: vector of intput integers to convert
  +`index::intType`: index of input position of `x`
  +`S::Array{W,1}`: size of tensor to convert from

  See also: [`ind2zeropos`](@ref) [`ind2pos`](@ref) [`ind2pos!`](@ref) [`zeropos2ind`](@ref) [`pos2ind`](@ref) [`pos2ind!`](@ref)
  """
  @inline function ind2pos!(currpos::Array{Array{X,1},1},k::intType,x::Array{Y,1},index::intType,S::Array{Z,1}) where X <: Integer where Y <: Integer where Z <: Integer
    currpos[k][1] = x[index]-1
    @simd for j = 1:size(S,1)-1
      currpos[k][j+1] = fld(currpos[k][j],S[j])
      currpos[k][j] = (currpos[k][j]) % S[j] + 1
    end
    currpos[k][size(S,1)] = currpos[k][size(S,1)] % S[size(S,1)] + 1
  #  ind2pos!(currpos[k],x[index],S)
    nothing
  end

  """
      ind2pos!(currpos,k,index,S)

  converts `index` to a position stored in `currpos` (parallelized) with tensor size `S`

  #Arguments:
  +`currpos::Array{Z,1}`: input position
  +`k::intType`: current thread for `currpos`
  +`index::intType`: input position
  +`S::Array{W,1}`: size of tensor to convert from

  See also: [`ind2zeropos`](@ref) [`ind2pos`](@ref) [`ind2pos!`](@ref) [`zeropos2ind`](@ref) [`pos2ind`](@ref) [`pos2ind!`](@ref)
  """
  @inline function ind2pos!(currpos::Array{Array{X,1},1},k::intType,index::intType,S::Array{Z,1}) where X <: Integer where Y <: Integer where Z <: Integer
    currpos[k][1] = index-1
    @simd for j = 1:size(S,1)-1
      currpos[k][j+1] = fld(currpos[k][j],S[j])
      currpos[k][j] = (currpos[k][j]) % S[j] + 1
    end
    currpos[k][size(S,1)] = currpos[k][size(S,1)] % S[size(S,1)] + 1
    nothing
  end
  export ind2pos!

  """
      pos2ind!(currpos,x,j,S)

  converts `currpos` with tensor size `S` to integer to be stored in `x[j]`

  #Arguments:
  +`currpos::Array{Z,1}`: input position
  +`x::Array{Y,1}`: output vector
  +`j::intType`: element of output vector to store resulting integer
  +`S::Array{W,1}`: size of tensor to convert from

  See also: [`ind2zeropos`](@ref) [`ind2pos`](@ref) [`ind2pos!`](@ref) [`zeropos2ind`](@ref) [`pos2ind`](@ref) [`pos2ind!`](@ref)
  """
  @inline function pos2ind!(currpos::Array{Y,1},x::Array{X,1},j::intType,S::Array{Z,1}) where X <: Integer where Y <: Integer where Z <: Integer
    x[j] = currpos[size(S,1)]-1
    @simd for i = size(S,1)-1:-1:2
      x[j] *= S[i]
      x[j] += currpos[i]-1
    end
    x[j] *= S[1]
    x[j] += currpos[1]
    nothing
  end

  """
      pos2ind!(currpos,k,x,j,S)

  converts `currpos` (parallelized) with tensor size `S` to integer to be stored in `x[j]`

  #Arguments:
  +`currpos::Array{Z,1}`: input position
  +`k::intType`: current thread for `currpos`
  +`x::Array{Y,1}`: output vector
  +`j::intType`: element of output vector to store resulting integer
  +`S::Array{W,1}`: size of tensor to convert from

  See also: [`ind2zeropos`](@ref) [`ind2pos`](@ref) [`ind2pos!`](@ref) [`zeropos2ind`](@ref) [`pos2ind`](@ref) [`pos2ind!`](@ref)
  """
  @inline function pos2ind!(currpos::Array{Array{Y,1},1},k::intType,x::Array{X,1},j::intType,S::Array{Z,1}) where X <: Integer where Y <: Integer where Z <: Integer
    x[j] = pos2ind(currpos[k],S)
    nothing
  end

  """
      pos2ind!(order,currpos,x,j,S)

  converts `currpos` with tensor size `S` to integer to be stored in `x[j]` and also reorders according to `order`

  #Arguments:
  +`order::Array{X,1}`: reorders input vector
  +`currpos::Array{Z,1}`: input position
  +`x::Array{Y,1}`: output vector
  +`j::intType`: element of output vector to store resulting integer
  +`S::Array{W,1}`: size of tensor to convert from

  See also: [`ind2zeropos`](@ref) [`ind2pos`](@ref) [`ind2pos!`](@ref) [`zeropos2ind`](@ref) [`pos2ind`](@ref) [`pos2ind!`](@ref)
  """
  @inline function pos2ind!(order::Array{X,1},currpos::Array{Z,1},x::Array{Y,1},j::intType,S::Array{W,1}) where W <: Integer where X <: Integer where Y <: Integer where Z <: Integer
    x[j] = pos2ind(order,currpos,S)
    nothing
  end
  export pos2ind!

  """
    zeropos2ind!(reorder,currpos,thisthread,x,i,,S)

  Reorders (according to `reorder`) and converts a position vector `currpos` to a zero-indexed index `x[i]` based on size `S` (all in-place)

  See also: [`ind2zeropos`](@ref) [`ind2pos`](@ref) [`ind2pos!`](@ref) [`zeropos2ind`](@ref) [`pos2ind`](@ref) [`pos2ind!`](@ref)
  """
  @inline function zeropos2ind!(order::Array{W,1},currpos::Array{Array{W,1},1},
              thisthread::Integer,x::Array{W,1},i::Integer,S::Array{W,1}) where W <: Integer
    x[i] = currpos[thisthread][order[size(S,1)]]
    @simd for j = size(S,1)-1:-1:2
      x[i] *= S[order[j]]
      x[i] += currpos[thisthread][order[j]]
    end
    x[i] *= S[order[1]]
    x[i] += currpos[thisthread][order[1]]+1
    nothing
  end
  export zeropos2ind!

  """
      quicksort!(T)

  Sorts `T` with a quicksort algorithm

  See also: [`sort!`](@ref)
  """
  function quicksort!(T::Array{W,1}) where W <: Real
    sort!(T)
    nothing
  end

  """
      quicksort!(lo,high,T)

  Sorts `T` between elements `lo` and `high` with a quicksort algorithm

  See also: [`sort!`](@ref)
  """
  function quicksort!(lo::intType,high::intType,T::AbstractArray)
    sort!(T,lo,high,Base.Sort.QuickSort,Base.Forward)
    nothing
  end

  """
      quicksort!(T,lo,high)

  Sorts `T` between elements `lo` and `high` with a quicksort algorithm

  See also: [`partition`](@ref)
  """
  function quicksort!(T::Array{X,1},lo::intType,high::intType) where {X<: Number}
    quicksort!(T,lo,high)
    nothing
  end

  """
      quicksort!(T,pos)

  Sorts `pos` and orders `T` accordinly

  See also: [`partition`](@ref)
  """
  function quicksort!(T::Array{W,1},pos::Array{X,1}) where {W <: Number,X<: Integer}
    lo = 1
    high = size(pos,1)
    if (lo < high)
      p = partition(T,pos,lo,high)
      quicksort!(T,pos,lo,p)
      quicksort!(T,pos,p+1,high)
    end
    nothing
  end

  """
      quicksort!(T,pos,lo,high)

  Sorts `pos` between `lo` and `high` and orders `T` accordingly

  See also: [`partition`](@ref)
  """
  function quicksort!(T::Array{W,1},pos::Array{X,1},lo::intType,high::intType) where {W <: Number,X <: Integer}
    if (lo < high)
      p = partition(T,pos,lo,high)
      quicksort!(T,pos,lo,p)
      quicksort!(T,pos,p+1,high)
    end
    nothing
  end

  """
      partition!(T,pos,lo,high)

  Exchanges values beween `lo` and `high` in `pos` and orders `T` accordingly

  See also: [`quicksort!`](@ref)
  """
  function partition(T::Array{W,1},pos::Array{X,1},lo::intType,high::intType) where {W <: Number,X <: Integer}
    pivot = pos[cld(lo+high,2)]
    i = lo
    j = high

    while i <= j
      while (pos[i] < pivot)
        i += 1
      end
      while (pos[j] > pivot)
        j -= 1
      end
      if (i <= j)
        pos[i],pos[j] = pos[j],pos[i]
        T[i],T[j] = T[j],T[i]
        i += 1
        j -= 1
      end
    end
    return j
  end

  """
      quicksort!(T,xpos,ypos,lo,high)

  Sorts `ypos` between `lo` and `high` and orders `T` and `xpos` accordingly

  See also: [`partition`](@ref)
  """
  function quicksort!(T::Array{W,1},xpos::Array{X,1},ypos::Array{X,1},lo::intType,high::intType) where {W <: Number,X <: Integer}
    if (lo < high)
      p = partition(T,xpos,ypos,lo,high)
      quicksort!(T,xpos,ypos,lo,p)
      quicksort!(T,xpos,ypos,p+1,high)
    end
    nothing
  end

  """
      partition!(T,xpos,ypos,lo,high)

  Exchanges values beween `lo` and `high` in `ypos` and orders `T` and `xpos` accordingly

  See also: [`quicksort!`](@ref)
  """
  function partition(T::Array{W,1},xpos::Array{X,1},ypos::Array{X,1},lo::intType,high::intType) where {W <: Number,X <: Integer}
    pivot = ypos[cld(lo+high,2)]
    i = lo
    j = high

    while i <= j
      while (ypos[i] < pivot)
          i += 1
      end
      while (ypos[j] > pivot)
          j -= 1
      end
      if (i <= j)
        xpos[i],xpos[j] = xpos[j],xpos[i]
        T[i],T[j] = T[j],T[i]
        ypos[i],ypos[j] = ypos[j],ypos[i]
        i += 1
        j -= 1
      end
    end
    return j
  end


  """
      quicksort!(T,inds,xpos,ypos,lo,high)

  Sorts `ypos` between `lo` and `high` and orders `T`, `inds`, and `xpos` accordingly

  See also: [`partition`](@ref)
  """
  function quicksort!(T::Array{W,1},inds::Array{intType,1},xpos::Array{X,1},ypos::Array{X,1},lo::intType,high::intType) where {W <: Number,X <: Integer}
    if (lo < high)
      p = partition(T,inds,xpos,ypos,lo,high)
      quicksort!(T,inds,xpos,ypos,lo,p)
      quicksort!(T,inds,xpos,ypos,p+1,high)
    end
    nothing
  end

  """
      partition!(T,inds,xpos,ypos,lo,high)

  Exchanges values beween `lo` and `high` in `ypos` and orders `T`, `inds`, and `xpos` accordingly

  See also: [`quicksort!`](@ref)
  """
  function partition(T::Array{W,1},inds::Array{intType,1},xpos::Array{X,1},ypos::Array{X,1},lo::intType,high::intType) where {W <: Number,X <: Integer}
    pivot = ypos[cld(lo+high,2)]
    i = lo
    j = high

    while i <= j
      while (ypos[i] < pivot)
          i += 1
      end
      while (ypos[j] > pivot)
          j -= 1
      end
      if (i <= j)
        xpos[i],xpos[j] = xpos[j],xpos[i]
        T[i],T[j] = T[j],T[i]
        inds[i],inds[j] = inds[j],inds[i]
        ypos[i],ypos[j] = ypos[j],ypos[i]
        i += 1
        j -= 1
      end
    end
    return j
  end


  """
      quicksort!(T,inds,xpos,ypos,QNs,lo,high)

  Sorts `ypos` between `lo` and `high` and orders `T`, `inds`, and `xpos` accordingly

  See also: [`partition`](@ref)
  """
  function quicksort!(T::Array{W,1},inds::Array{intType,1},xpos::Array{X,1},ypos::Array{X,1},
                      QNs::Array{intType,1},lo::intType,high::intType) where {W <: Number,X <: Integer}
    if (lo < high)
      p = partition(T,inds,xpos,ypos,QNs,lo,high)
      quicksort!(T,inds,xpos,ypos,QNs,lo,p)
      quicksort!(T,inds,xpos,ypos,QNs,p+1,high)
    end
    nothing
  end

  """
      partition!(T,inds,xpos,ypos,lo,high)

  Exchanges values beween `lo` and `high` in `ypos` and orders `T`, `inds`, and `xpos` accordingly

  See also: [`quicksort!`](@ref)
  """
  function partition(T::Array{W,1},inds::Array{intType,1},xpos::Array{X,1},ypos::Array{X,1},
                      QNs::Array{intType,1},lo::intType,high::intType) where {W <: Number,X <: Integer}
    pivot = QNs[cld(lo+high,2)]
    i = lo
    j = high

    while i <= j
      while (QNs[i] < pivot)
          i += 1
      end
      while (QNs[j] > pivot)
          j -= 1
      end
      if (i <= j)
        xpos[i],xpos[j] = xpos[j],xpos[i]
        T[i],T[j] = T[j],T[i]
        inds[i],inds[j] = inds[j],inds[i]
        ypos[i],ypos[j] = ypos[j],ypos[i]
        QNs[i],QNs[j] = QNs[j],QNs[i]
        i += 1
        j -= 1
      end
    end
    return j
  end

  export quicksort!

  import Base.unique!
  """
      unique!(B)

  finds unique elements but reorders input vector

  See also: [`unique`](@ref)
  """
  function unique!(B::Array{W,1}#=,start::intType,final::intType=#) where W <: Qnum
    quicksort!(1,length(B),B)
    uniqueQNs = W[copy(B[1])] #W[copy(B[start])]
    for a = 2:length(B) #start+1:final
      if B[a-1] != B[a]
        push!(uniqueQNs,copy(B[a]))
      end
    end
    return uniqueQNs
  end

  import Base.unique
  """
      unique!(B)

  finds unique elements without generating a new array to be sorted over (i.e., does not reorder to input vector)

  See also: [`unique!`](@ref)
  """
  function unique(T::Array{W,1}) where W <: Qnum
    B = copy(T)
    return unique!(B)
  end

####################################################
####################################################
####################################################

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
  function getindex(C::Q, a::genColType...) where {Q <: qarray}
    return getindex!(C, a...)
  end

  import ..tensor.getindex!
  function getindex!(C::AbstractArray, a::genColType...) #::AbstractArray
    return getindex(C, a...)
  end

  function getindex!(C::Q, a::genColType...) where {Q <: qarray}
    condition = true
    for p = 1:length(a)
      condition = condition && (typeof(a[p]) <: Colon)
      condition = condition && (typeof(a[p]) <: UnitRange && length(a[p]) == size(C,p))
    end
    if condition
      return C
    end

    rangeArgs = intType[]
    integerArgs = intType[]
    for i = 1:size(a, 1)
      if typeof(a[i]) <: Integer
        push!(integerArgs, i)
      else
        push!(rangeArgs,i)
      end
    end
    # transform the colons in actual ranges
    # because the "in" operator and length function cannot be used on raw colons
    ap = []
    for (dim,val) in enumerate(a)
      if typeof(a[dim]) <: Colon
        push!(ap, 1:C.size[dim])
      else
        push!(ap,a[dim])
      end
    end
    checksort = Array{Bool,1}(undef,size(a,1))
    for i = 1:size(a,1)
      if typeof(a[i]) <: Array || (typeof(a[i]) <: UnitRange && a[i][1] != 1)
        checksort[i] = false
      else
        checksort[i] = true
      end
    end
    numthreads = Threads.nthreads()
    ##identify the elements of T and inds that must be kept.
    #can't parallelize super easily as there is a push!
    keep = Array{Bool,1}(undef,length(C.ind))
    pos = Array{intType,1}[Array{intType,1}(undef,size(C.size,1)) for i = 1:numthreads]
    let C = C, keep = keep, pos = pos, ap = ap
      Threads.@threads    for i = 1:size(C.ind,1)
        ind = C.ind[i]
        keep[i] = true
        thisthread = Threads.threadid()
        ind2pos!(pos,thisthread,ind,C.size)
    
        holdint = 0
        while keep[i] && holdint < size(pos[thisthread],1)
          holdint += 1
          keep[i] = keep[i] && (pos[thisthread][holdint] in ap[holdint])
        end
      end
    end
    newT = C.T[keep]
    newinds = C.ind[keep]
    ##
    aprange = ap[rangeArgs]
    newsize = intType[length(ranges) for ranges in aprange]
    ##index must be recomputed. the tensor changed size. this means the result of this function cannot be modified without
    # modifying the original tensor
    # solving this difficulty must involve making the qtensor aware of shared ressources...
    o_pos = Array{intType,1}[Array{intType,1}(undef,size(C.size,1)) for i = 1:numthreads]
    t_pos = Array{intType,1}[Array{intType,1}(undef,size(rangeArgs,1)) for i = 1:numthreads]
    let newinds = newinds, o_pos = o_pos, t_pos = t_pos, rangeArgs = rangeArgs, checksort = checksort, newsize = newsize, ap = ap
      Threads.@threads    for b = 1:size(newinds,1)
        ind = newinds[b]
        thisthread = Threads.threadid()
        ind2pos!(o_pos,thisthread,ind,C.size)
        @simd for k = 1:size(rangeArgs,1)
          thisind = rangeArgs[k]
          t_pos[thisthread][k] = o_pos[thisthread][thisind]
        end
        for i = 1:size(t_pos[thisthread],1)
          if !checksort[i]
            t_pos[thisthread][i] = findfirst(x -> x==t_pos[thisthread][i],(aprange)[i])[1]
          end
        end
        newinds[b] = pos2ind(t_pos[thisthread],newsize)
      end
    end
    ##
    newQsize = Array{intType,1}[intType[i] for i = 1:size(newsize, 1)]
    newQnumMat = Array{typeof(C.flux),1}[Qnlist[arange] for (Qnlist, arange) in zip(C.QnumMat[rangeArgs], a[rangeArgs])]
    newflux = C.flux
    if size(integerArgs,1) > 0
      for k in integerArgs
          newflux += inv(C.QnumMat[k][a[k]])
      end
    end
    newQnumSum = unique.(newQnumMat)
    return Qtens{eltype(newT),typeof(newflux)}(newsize, newQsize, newT, newinds, newQnumMat, newQnumSum, newflux)
  end
  export getindex!

  import ..tensor.searchindex
  """
      searchindex(C,a...)

  Find element of `C` that corresponds to positions `a`
  """
  function searchindex(C::qarray,a::intType...)::Number
    thispos = intType[a[i] for i = 1:length(a)]
    #all the index are integers
    target_ind = pos2ind(thispos,C.size)
    j = 0
    retval = 1E-42
    while j < size(C.ind, 1) && retval == 1E-42
      j += 1
      if C.ind[j] == target_ind
        retval = C.T[j]
      end
    end
    return retval
  end

  function searchindex(C::AbstractArray,a::intType...)::Number
    return C[a...]
  end

  #get the last index of a Qtensor (ex: A[:,1:end]...defines "end")
  import Base.lastindex
  """
      lastindex(Qtens,i)

  get the last index of a Qtensor (ex: A[:,1:end]...defines "end")
  """
  function lastindex(Qtens::qarray, i::intType) #where Q <: qarray
    return Qtens.size[i]
  end

  """
      sortTensors!(A...)

  Sort any number of Qtensors `A` accoring to their `ind` field

  See also: [`sortTensors!`](@ref) [`Qtens`](@ref)
  """
  function sortTensors!(A::qarray...)
    for i = 1:size(A, 1)
      if !issorted(A[i].ind)
        quicksort!(A[i].T, A[i].ind)
      end
    end
    nothing
  end
  export sortTensors!

  """
      sortTensor!(A...)

  Sort Qtensor `A` accoring to their `ind` field

  See also: [`sortTensors!`](@ref) [`Qtens`](@ref)
  """
  function sortTensor!(A::qarray)
    sortTensors!(A)
  end
  export sortTensor!

  import .tensor.mult!
  """
      mult!(A,x)

  Multiplies `x*A` (commutative) for dense or quantum tensors

  See also: [`*`](@ref) [`add!`](@ref) [`sub!`](@ref) [`div!`](@ref)
  """
  function mult!(Qt::qarray, num::Number)::qarray
    @simd for i = 1:size(Qt.T, 1)
      Qt.T[i] *= num
    end
    return Qt
  end

  function mult!(Qt::AbstractArray, num::Number) #::AbstractArray
    return Qt * num
  end

  function mult!(num::Number, Qt::X)::TensType where X <: TensType
    return mult!(Qt, num)
  end

  import .tensor.add!
  """
      add!(A,B[,x])

  Adds `A + x*B` (default x = 1) for dense or quantum tensors

  See also: [`+`](@ref) [`sub!`](@ref) [`mult!`](@ref) [`div!`](@ref)
  """
  function add!(A::qarray, B::qarray, mult::Number)::qarray
    sortTensors!(A, B)
    i, j = 1, 1
    iniSizeA = size(A.ind, 1) # size A.ind can change during the loop. we must stop at the current size anyway.
    while i <= iniSizeA && j <= size(B.ind, 1)
      if A.ind[i] < B.ind[j]
        i += 1
      elseif A.ind[i] > B.ind[j]
        push!(A.T, mult * B.T[j])
        push!(A.ind, B.ind[j])
        j += 1
      else
        A.T[i] += mult * B.T[j]
        i += 1
        j += 1
      end
    end
    for k = j:size(B.ind, 1)
      push!(A.T, mult * B.T[k])
      push!(A.ind, B.ind[k])
    end
    return A
  end

  function add!(A::AbstractArray, B::AbstractArray, mult::Number)::AbstractArray
    return A + mult!(mult,B)
  end

  function add!(A::X, B::Y)::TensType where {X <: TensType,Y <: TensType}
    mA,mB = checkType(A,B)
    return add!(mA, mB, 1.)
  end
  
  function add!(A::X, B::Y, mult::Number)::TensType where {X <: TensType,Y <: TensType}
    mA,mB = checkType(A,B)
    return add!(mA, mB, mult)
  end

  import ..tensor.sub!
  """
      sub!(A,B[,x])

  Subtracts `A - x*B` (default x = 1) for dense or quantum tensors

  See also: [`-`](@ref) [`add!`](@ref) [`sub!`](@ref) [`mult!`](@ref) [`div!`](@ref)
  """
  function sub!(A::X, B::Y)::TensType where {X <: TensType,Y <: TensType}
    return add!(A, B, -1.)
  end

  function sub!(A::X, B::Y, mult::Number)::TensType where {X <: TensType,Y <: TensType}
    return add!(A, B, -mult)
  end


  import .tensor.div!
  """
      div!(A,x)

  Division by a scalar `A/x` (default x = 1) for dense or quantum tensors

  See also: [`/`](@ref) [`add!`](@ref) [`sub!`](@ref) [`mult!`](@ref)
  """
  function div!(A::qarray, num::Number)::qarray
    @simd for w = 1:size(A.T, 1)
      A.T[w] /= num
    end
    return A
  end

  function div!(A::AbstractArray, num::Number)::AbstractArray
    return A / num
  end
  export div!

  import LinearAlgebra.+
  """
      +(A,B)

  adds two tensors `A` and `B` together

  See also: [`add!`](@ref)
  """
  function +(A::qarray, B::qarray)::qarray
    C = copy(A)
    return add!(C, B)
  end

  import LinearAlgebra.-
  """
      -(A,B)

  subtracts two tensors `A` and `B` (`A`-`B`)

  See also: [`sub!`](@ref)
  """
  function -(A::qarray, B::qarray)::qarray
    C = copy(A)
    return sub!(C, B)
  end

  import Base.*
  """
      *(A,num)

  mutiplies a tensor `A` by a number `num`

  See also: [`mult!`](@ref)
  """
  function *(num::Number, Qt::qarray)::qarray
    return mult!(copy(Qt), num)
  end

  function *(Qt::qarray, num::Number)::qarray
    return mult!(copy(Qt), num)
  end

  import LinearAlgebra./
  """
      /(A,num)

  divides a tensor `A` by a number `num`

  See also: [`div!`](@ref)
  """
  function /(A::qarray, num::Number)::qarray
    R = copy(A)
    return div!(R, num)
  end

  import Base.sqrt
  """
      sqrt(Qt)

  Takes the square root of a tensor

  See also: [`sqrt`](@ref)
  """
  function sqrt(Qt::Qtens)::qarray
    newQt = copy(Qt)
    return sqrt!(newQt)
  end

  import .tensor.sqrt!
  """
      sqrt!(Qt)

  Takes the square root of a dense tensor (new tensor created) or Qtensor (in-place)
  """
  function sqrt!(Qt::Qtens)::qarray
    @simd for a = 1:length(Qt.T)
      Qt.T[a] = sqrt.(Qt.T[a])
    end
    return Qt
  end

  function sqrt!(Qt::AbstractArray) #::AbstractArray
    return sqrt(Qt)
  end
  export sqrt!

  import .tensor.invmat!
  """
      invmat!(Qt[,zero=])

  Creates inverse of a diagonal matrix in place (dense matrices are copied anyway);
  if value is below `zero`, the inverse is set to zero

  See also: [`invmat`](@ref)
  """
  function invmat!(Qt::qarray;zero::Float64=1E-16)
#    @assert(size(Qt.QnumMat,1) == 2)
    for a = 1:size(Qt.T,1)
      Qt.T[a] = abs(Qt.T[a]) > zero ? 1/Qt.T[a] : 0.
    end
    return Qt
  end

  function invmat!(Qt::AbstractArray;zero::Float64=1E-16)
    for a = 1:size(Qt,1)
      Qt[a,a] = abs(Qt[a,a]) > zero ? 1/Qt[a,a] : 0.
    end
    return Qt
  end
  export invmat!

  """
      invmat(Qt)

  Creates inverse of a diagonal matrix by making a new copy

  See also: [`invmat!`](@ref)
  """
  function invmat(Qt::TensType)
    return invmat!(copy(Qt))
  end
  export invmat

  import Base.sum
  """
      sum(A)

  sum elements of a Qtensor
  """
  function sum(A::qarray)::Number
    return sum(A.T)
  end

  import LinearAlgebra.norm
  """
      norm(A)

  Froebenius norm of a Qtensor
  """
  function norm(A::qarray)::Number
    return norm(A.T)
  end
  export norm

  import Base.eltype
  """
      eltype(A)

  element type of a Qtensor (i.e., `T` field)

  See also: [`Qtens`](@ref)
  """
  function eltype(A::Qtens{T,Q}) where {T <: Number, Q <: Qnum}
    return T
  end

  import LinearAlgebra.conj
  """
      conj(A)

  conjugates a Qtensor by creating a copy

  See also: [`conj!`](@ref)
  """
  function conj(currQtens::qarray)::qarray
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
  function conj!(currQtens::qarray)
    currQtens.T = conj!(currQtens.T)
    currQtens.QnumMat = [inv!.(currQtens.QnumMat[w]) for w = 1:length(currQtens.QnumMat)]
    currQtens.QnumSum = [unique(currQtens.QnumMat[w]) for w = 1:length(currQtens.QnumMat)]
    currQtens.flux = inv(currQtens.flux)
    return currQtens
  end

  import LinearAlgebra.size
  import Base.size
  """
      size(A[,i])

  gets the size of a Qtensor (identical usage to dense `size` call)
  """
  function size(A::qarray, i::intType)
    return prod(a->A.size[a], A.Qsize[i])
  end

  function size(A::qarray)
    newsize = intType[size(A,i) for i = 1:size(A.Qsize, 1)]
    return newsize
  end

  import LinearAlgebra.ndims
  """
      ndims(A)

  number of dimensions of a Qtensor (identical usage to dense `size` call)
  """
  function ndims(A::qarray)
    return length(A.Qsize)
  end

  """
      permind!(order,x,S)
  
  Permutes indices in position vector `x` (with tensor size `S`) according to reordering `order`
  
  See also: [`permutedims`](@ref) [`permutedims!`](@ref)
  """
  function permind!(order::Array{X,1},x::Array{Y,1},S::Array{Z,1}) where X <: Integer where Y <: Integer where Z <: Integer
    numthreads = Threads.nthreads()
    currpos = Array{intType,1}[Array{intType,1}(undef,size(S,1)) for i = 1:numthreads]
    let currpos = currpos, S = S, x = x, order = order
      Threads.@threads for i = 1:size(x,1)
        thisthread = Threads.threadid()
        ind2zeropos!(currpos,thisthread,x,i,S)
        zeropos2ind!(order,currpos,thisthread,x,i,S)
      end
    end
    nothing
  end
  export permind!

  import Base.permutedims
  """
      permutedims(A,[1,3,2,...])

  permutes dimensions of `A`  (identical usage to dense `size` call)

  See also: [`permutedims!`](@ref)
  """
  function permutedims(currQtens::Qtens{W,Q}, vec::Array{intType,1}) where {W <: Number, Q <: Qnum}
    newQtens = copy(currQtens)
    permutedims!(newQtens, vec)
    return newQtens
  end

  import Base.permutedims!
  """
      permutedims!(A,[1,3,2,...])

  permute dimensions of a Qtensor in-place (no equivalent, but returns value so it can be used identically to `permutedims` on a dense tensor)

  See also: [`permutedims`](@ref)
  """
  function permutedims!(currQtens::Qtens{W,Q}, vec::Array{intType,1}) where {W <: Number, Q <: Qnum}
    if size(currQtens.size, 1) == size(vec, 1)
      order = vec
    else
      order = eltype(vec)[]
      for i = 1:size(vec, 1)
        for j = 1:size(currQtens.Qsize[vec[i]], 1)
          push!(order, currQtens.Qsize[vec[i]][j])
        end
      end
    end
    currQtens.Qsize = currQtens.Qsize[vec]
    counter = intType[0]
    for k = 1:length(currQtens.Qsize)
      for m = 1:length(currQtens.Qsize[k])
        counter[1] += 1
        currQtens.Qsize[k][m] = counter[1]
      end
    end
    thispos = Array{eltype(currQtens.ind),1}(undef, size(order, 1))
    permind!(order, currQtens.ind, currQtens.size)
    currQtens.size = currQtens.size[order]
    currQtens.QnumMat = currQtens.QnumMat[order]
    currQtens.QnumSum = currQtens.QnumSum[order]
    return currQtens
  end

  function permutedims!(A::AbstractArray, order::Array{intType,1}) #where W <: Integer
    return permutedims(A, order)
  end

  """
      convIn(iA)

  Convert `iA` of type Int64 (ex: 1), Array{Int64,1} ([1,2,3]), or Array{Int64,2}* ([1 2 3]) to Array{Int64,1}

  *- two-dimensional arrays must be size "m x 1"
  """
  function convIn(iA::O)::Array{intType,1} where O <: intvecType
    if O == Array{intType,1}
      return iA
    elseif O <: Integer
      return intType[iA]
    elseif O <: Array{Integer,2}
#      @assert(ndims(iA) == 2)
      return iA[:]
    elseif O <: Tuple
      return intType[O[i] for i = 1:length(O)]
    end
  end
  export convIn

  import Base.zero
    """Like the default function zero(t::Array), return an object with the same properties containing only zeros."""
  function zero(t::Qtens)
    return Qtens(copy(t.size),copy(t.Qsize),eltype(t.T)[],eltype(t.ind)[],copy(t.QnumMat),copy(t.QnumSum), copy(t.flux)) 
  end

  """
      Idhelper(A,iA)

  generates the size of matrix equivalent of an identity matrix from tensor `A` with indices `iA`

  #Output:
  +`lsize::Int64`: size of matrix-equivalent of identity operator
  +`finalsizes::Int64`: size of identity operator

  See also: [`makeId`](@ref) [`trace`](@ref)
  """
  function Idhelper(A::TensType,iA::W) where W <: Union{intvecType,Array{Array{intType,1},1}}
    if typeof(iA) <: intvecType
      vA = convIn(iA)
      lsize = size(A,vA[1])
      finalsizes = [lsize,lsize]
    else
      lsize = prod(w->size(A,iA[w][1]),1:length(iA))
      leftsizes = [size(A,iA[w][1]) for w = 1:length(iA)]
      rightsizes = [size(A,iA[w][2]) for w = 1:length(iA)]
      finalsizes = vcat(leftsizes,rightsizes)
    end
    return lsize,finalsizes
  end

  
  """
      makeId(A,iA)

  generates an identity matrix from tensor `A` with indices `iA`

  See also: [`trace`](@ref)
  """
  function makeId(A::Qtens,iA::W) where W <: Union{intvecType,Array{Array{intType,1},1}}
    lsize,finalsizes = Idhelper(A,iA)
    newQnumMat = A.QnumMat[iA]
    typeA = eltype(A)
    Id = Qtens(newQnumMat,Type=typeA)
    Id.ind = intType[i+lsize*(i-1) for i = 1:lsize]
    Id.T = ones(typeA,length(Id.ind))

    Id.size = finalsizes
    Id.Qsize = [[i] for i = 1:length(Id.size)]
    return Id
  end

  function makeId(A::Array,iA::W) where W <: Union{intvecType,Array{Array{intType,1},1}}
    lsize,finalsizes = Idhelper(A,iA)
    Id = zeros(eltype(A),lsize,lsize) + LinearAlgebra.I
    return reshape(Id,finalsizes...)
  end

  """
      makeId(A,iA)

  generates an identity matrix from tensor `A` with indices `iA`

  See also: [`trace`](@ref)
  """
  function makeId(A::denstens,iA::W) where W <: Union{intvecType,Array{Array{intType,1},1}}
    lsize,finalsizes = Idhelper(A,iA)
    Id = zeros(eltype(A),lsize,lsize) + LinearAlgebra.I
    return reshape(Id,finalsizes...)
  end
  export makeId

  """
      swapgate(A,iA,B,iB)
  
  generates a swap gate (order of indices: in index for `A`, in index for `B`, out index for `A`, out index for `B`) for `A` and `B`'s indices `iA` and `iB`
  """
  function swapgate(A::TensType,iA::W,B::TensType,iB::R) where W <: Union{intvecType,Array{Array{intType,1},1}} where R <: Union{intvecType,Array{Array{intType,1},1}}
    LId = makeId(A,iA)
    RId = makeId(B,iB)
    if typeof(LId) <: denstens || typeof(LId) <: qarray
      push!(LId.size,1)
    else
      LId = reshape(LId,size(LId)...,1)
    end
    if typeof(RId) <: denstens || typeof(RId) <: qarray
      push!(RId.size,1)
    else
      RId = reshape(RId,size(RId)...,1)
    end
    fullId = contract(LId,4,RId,4)
    return permute(fullId,[1,3,2,4])
  end

  import Base.isapprox
  """
      isapprox(A,B)

  Checks if `A` is approximately `B`
  """
  function isapprox(A::Qtens{Z,Q},B::Qtens{W,Q})::Bool where {Z <: Number, W <: Number, Q <: Qnum}
    test = length(A.T) == length(B.T)
    if test
      return isapprox(norm(A),norm(B))
    else
      return false
    end
  end

end
