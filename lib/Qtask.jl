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
    Module: Qtask

Functions that are applied to Qtensors that are used in contractions, decompositions, etc.

See also: [`Qtensor`](@ref) [`contractions`](@ref) [`decompositions`](@ref)
"""
module Qtask
using ..tensor
using ..Qtensor
using ..QN

  """
      genarray

  `genarray` = Array{intType,1}

  See also: [`intType`](@ref)
  """
  const genarray = Array{intType,1}

  """
      genarraytwo

  `genarraytwo` = Array{intType,2}

  See also: [`intType`](@ref)
  """
  const genarraytwo = Array{intType,2}

  """
    currpos_type

  `currpos_type` =  Array{Array{intType,1},1}

  See also: [`intType`](@ref)
  """
  const currpos_type = Array{Array{intType,1},1}

  import ..Qtensor.Qtens
  """
      Qtens(QtensAA,Op...)

  Generates Qtensors from operators `Op` provided (`Arrays` or `denstens`) that a Qtensor `QtensAA` is provided with suitable quantum numbers and other information
  """
  function Qtens(QtensAA::qarray,Op::Union{AbstractArray,denstens}...)::qarray
    thisbool = size(QtensAA.Qsize,1) != 2
    if thisbool
      storeQsize = copy(QtensAA.Qsize)
      center = cld(size(QtensAA.size, 1), 2)
      QtensAA.Qsize = Array{intType,1}[intType[i for i = 1:center],[i for i = center+1:size(QtensAA.size, 1)]]
    end
    Linds, Rinds, Lsize, Rsize, AAxypos, LR, nrowsAA, ncolsAA, currpos, zeroQN,
    AALfinalxy, AALcountsizes, AALfinalQN, conQnumSumAAL, AARfinalxy, AARcountsizes, AARfinalQN,
    conQnumSumAAR,subblocksizes, mindims, Ablocksizes, Bblocksizes, totRetSize, totBlockSizeA, totBlockSizeB, 
    fullAA = initmatter(QtensAA)

    notrandom = length(Op) > 0

    if notrandom
      if typeof(Op[1]) <: denstens
        thisOp = reshape(Op[1].T,Op[1].size...)
      else
        thisOp = Op[1]
      end
      newinds = Array{intType,1}(undef,totRetSize)
      newT = Array{eltype(thisOp),1}(undef,totRetSize)
      newQt = Qtens(QtensAA.size,QtensAA.Qsize,eltype(newT)[],eltype(newinds)[],QtensAA.QnumMat,QtensAA.QnumSum,QtensAA.flux)
      thispos = Array{intType,1}(undef,ndims(thisOp))
      vec = intType[i for i = 1:ndims(thisOp)]
      startpos!(thispos,QtensAA.size)
      for i = 1:prod(size(thisOp))
        nextpos!(thispos,QtensAA.size)
        currQN = sum(a->QtensAA.QnumMat[a][thispos[a]],1:size(thispos,1))
        if QtensAA.flux == currQN
          push!(newQt.T,searchindex(thisOp,thispos...))
          push!(newQt.ind,i)
        end
      end
      AAxypos = makeXYpos(newQt,Linds,Rinds,currpos)
    else
      newinds = Array{intType,1}(undef,totRetSize)
      newT = Array{eltype(QtensAA.T),1}(undef,totRetSize)
      newQt = Qtens(QtensAA.size,QtensAA.Qsize,eltype(newT)[],eltype(newinds)[],QtensAA.QnumMat,QtensAA.QnumSum,QtensAA.flux)
    end
    if notrandom
      let AALcountsizes = AALcountsizes, AARcountsizes = AARcountsizes, notrandom = notrandom, AALfinalQN = AALfinalQN, AALfinalxy = AALfinalxy, newQt = newQt, subblocksizes = subblocksizes, nrowsAA = nrowsAA, newT = newT, newinds = newinds
        Threads.@threads  for x  = 1:size(AALcountsizes, 1)
          U = zeros(eltype(thisOp),AALcountsizes[x], AARcountsizes[x])
          retAALrows = QNsearch(U,newQt,AALfinalxy,AARfinalxy,AAxypos,1,AALfinalQN,x,AALcountsizes)  
          retAARcols = makeRetRowCol(AARfinalQN, x, AARcountsizes)

          subcounter = x == 1 ? 0 : sum(y->subblocksizes[y],1:x-1)
          retsubBlock!(nrowsAA,retAALrows,retAARcols,newT,newinds,U,subcounter)
        end
      end
    else
      for x  = 1:size(AALcountsizes, 1)
        U = rand(eltype(QtensAA), AALcountsizes[x], AARcountsizes[x])
        retAALrows = makeRetRowCol(AALfinalQN, x, AALcountsizes)
        retAARcols = makeRetRowCol(AARfinalQN, x, AARcountsizes)

        subcounter = x == 1 ? 0 : sum(y->subblocksizes[y],1:x-1)
        retsubBlock!(nrowsAA,retAALrows,retAARcols,newT,newinds,U,subcounter)
      end
    end

    newQt.T = newT
    newQt.ind = newinds
    
    if thisbool
      newQt.Qsize = storeQsize
    end
    return newQt
  end
#=
  """
      Qtens(QtensAA,Op...)

  Generates Qtensors from type dense tensor `Op` provided that a Qtensor `QtensAA` is provided with suitable quantum numbers and other information
  """
  function Qtens(QtensAA::qarray,Op::denstens...)::qarray
    X = reshape(Op[1].T,Op[1].size...)
    return Qtens(QtensAA,X)
  end
=#
  """
      Qtens(QnumMat,Op...)

  Generates Qtensors from type dense tensor `Op` provided that a Qtensor `QtensAA` is provided with suitable quantum numbers and other information
  """
  function Qtens(QnumMat::Array{Array{Q,1},1},Op::B...)::qarray where Q <: Qnum where B <: Union{AbstractArray,denstens}
    Qt = Qtens(QnumMat)
    return Qtens(Qt,Op...)
  end

  import ..tensor.reshape!
  """
      reshape!(M,a...[,merge=])

  In-place reshape for Qtensors (otherwise makes a copy); can also make Qtensor unreshapable with `merge`, joining all grouped indices together

  # Warning
  If the Qtensor size is (10,1,2) and we want to reshape in to a (10,2) tensor, this algorithm will reshape to a 
  [[1,2],[3]] be default instead of [[1],[2,3]], so beware.

  See also: [`reshape`](@ref)
  """
  function reshape!(Qt::qarray, S::intType...;merge::Bool=false)::qarray#where W <: Integer
    newQsize = Array{eltype(S),1}[]
    k = 0
    for a = 1:size(S, 1)
      currsize = S[a]
      if a > length(Qt.size) #adding size one to the end of a tensor is free
        push!(Qt.size, S[a])
        push!(newQsize, intType[a])
        zeroQN = typeof(Qt.flux)()
        newQNs = [zeroQN for a = 1:S[a]]
        push!(Qt.QnumMat,newQNs)
        push!(Qt.QnumSum,[zeroQN])
      else
        if k <= size(Qt.size, 1)
          k += 1
          thissize = eltype(S)[k]
          indsize = Qt.size[k]
          while indsize < currsize
            k += 1
            push!(thissize, k)
            indsize *= Qt.size[k]
          end
          if indsize == currsize && a == size(S, 1)
            k += 1
            for p = k:size(Qt.size, 1)
              push!(thissize, p)
            end
          end
          push!(newQsize, thissize)
        end
      end
    end
    Qt.Qsize = newQsize
    if merge
      mergereshape!(Qt)
    end
    return Qt
  end

  function (reshape!(Qt::Array{W,N},S::intType...;merge::Bool=false)::Array{W,length(S)}) where W <: Number where N
    return reshape(Qt,S...)
  end

  function reshape!(Qt::qarray, newQsize::Array{Array{intType,1},1};merge::Bool=false)::qarray
    return reshape!(Qt,newQsize...,merge=merge)
  end

  function reshape!(Qt::qarray, newQsize::Array{intType,1}...;merge::Bool=false)::qarray
    Qt.Qsize = Array{intType,1}[newQsize[k] for k = 1:size(newQsize, 1)]
    if merge
      mergereshape!(Qt)
    end
    return Qt
  end
    
  function (reshape!(Qt::Array{W,N}, newQsize::Array{Array{intType,1},1};merge::Bool=false)::Array{W,length(newQsize)}) where W <: Number where N
    return reshape!(Qt,intType[prod(a->size(Qt, a), newQsize[i]) for i = 1:size(newQsize, 1)]...,merge=merge) #reshape(Qt, intType[prod(a->size(Qt, a), newQsize[i]) for i = 1:size(newQsize, 1)]...)
  end
  
  function (reshape!(Qt::Array{W,N}, newQsize::Array{intType,1}...;merge::Bool=false)::Array{W,length(newQsize)}) where W <: Number where N
    return reshape(Qt, intType[newQsize...])
  end

  import Base.reshape
  """
      reshape!(M,a...[,merge=])

  Reshape for Qtensors (makes a copy); can also make Qtensor unreshapable with `merge`, joining all grouped indices together

  # Warning
  If the Qtensor size is (10,1,2) and we want to reshape in to a (10,2) tensor, this algorithm will reshape to a 
  [[1,2],[3]] be default instead of [[1],[2,3]], so beware.

  See also: [`reshape!`](@ref)
  """
  function reshape(Qt::qarray, S::intType...;merge::Bool=false)::qarray
    return reshape!(copy(Qt),S...,merge=merge)
  end

  function reshape(Qt::qarray, newQsize::Array{Array{intType,1},1};merge::Bool=false)
    return reshape!(copy(Qt), newQsize...,merge=merge)
  end
  
  function reshape(Qt::qarray, newQsize::Array{intType,1}...;merge::Bool=false)
    return reshape!(copy(Qt), newQsize...,merge=merge)
  end

  function (reshape(Qt::Array{W,N}, newQsize::Array{Array{intType,1},1};merge::Bool=false)::Array{W,length(newQsize)}) where W <: Number where N
    return reshape(Qt, [prod(b->size(Qt,b),newQsize[a]) for a = 1:length(newQsize)]...)
  end

  """
      mergereshape!(M)

  Groups all joined indices together to make one index that is unreshapable.  Dense tensors are unaffected.

  See also: [`reshape!`](@ref)
  """
  function mergereshape!(Qt::W) where W <: TensType
#    if W <: qarray
      Q = typeof(Qt.flux)
      newdim = length(Qt.Qsize)
      newQnumMat = Array{Array{Q,1},1}(undef,newdim)
      newsizes = intType[size(Qt,a) for a = 1:newdim]
      zeroQN = Q()
      for a = 1:size(Qt.Qsize,1)
        if size(Qt.Qsize[a],1) > 1
          thissize = newsizes[a]
          QnumVec = Q[Q() for i = 1:thissize]
          vec = Qt.Qsize[a]
          numthreads = Threads.nthreads()
          pos = Array{intType,1}[Array{intType,1}(undef,size(vec,1)) for a = 1:numthreads]
          sizes = Qt.size[Qt.Qsize[a]]
          thisflux = a == size(Qt.Qsize,1) ? Qt.flux : zeroQN

          qaddition!(thissize,vec,sizes,QnumVec,pos,Qt.QnumMat)
          newQnumMat[a] = QnumVec
        else
          newQnumMat[a] = Qt.QnumMat[Qt.Qsize[a][1]]
        end
      end
      Qt.size = newsizes
      Qt.Qsize = [[i] for i = 1:newdim]
      Qt.QnumMat = newQnumMat
      Qt.QnumSum = unique.(newQnumMat)
#    end
    return Qt
  end
  export mergereshape!

  import ..tensor.unreshape!
  """
      unreshape!(Qt,a...)

  In-place, unambiguous unreshaping for Qtensors.  Works identically to reshape for dense tensors

  See also: [`reshape!`](@ref)
  """
  function unreshape!(Qt::AbstractArray,sizes::W...) where W <: Integer
    return reshape!(Qt,sizes...)
  end

  function unreshape!(Qt::qarray,sizes::W...) where W <: Integer
    Qt.Qsize = Array{intType,1}[intType[i] for i = 1:length(Qt.size)]
    return Qt
  end

  import ..tensor.unreshape
  """
      unreshape(Qt,a...)

  Unambiguous unreshaping for Qtensors.  Works identically to reshape for dense tensors
  """
  function unreshape(Qt::qarray)
    return unreshape(Qt,Qt.size)
  end

  function unreshape(Qt::T,sizes::Array{W,1}) where {W <: Integer, T<: Union{qarray,AbstractArray}}
    return unreshape(Qt,sizes...)
  end

  function unreshape(Qt::AbstractArray,sizes::W...) where W <: Integer
    return reshape(Qt,sizes...)
  end

  function unreshape(Qt::qarray,sizes::W...) where W <: Integer
    return unreshape!(copy(Qt),sizes...)
  end

  """
      getinds(A,iA)

  Sub-function for quantum number contraction.  A Qtensor `A` with indices to contract `iA` generates all contracted indices (if, for example, a joined index was called by a single index number), and also the un-contracted indices
  """
  function getinds(currQtens::qarray, vec::Array{intType,1}) #where X <: Integer
    consize = sum(a->size(currQtens.Qsize[a],1),vec)
    con = Array{intType,1}(undef,consize)  
    notcon = Array{intType,1}(undef,size(currQtens.size,1)-consize)
    counter,altcounter = 0,0

    for j = 1:size(vec, 1)
      @simd for p in currQtens.Qsize[vec[j]]
        counter += 1
        con[counter] = p
      end
    end

    for j = 1:size(currQtens.Qsize, 1)
      condition = true
      k = 0
      while k < size(vec,1) && condition
        k += 1
        condition = !(j == vec[k])
      end
      if condition
        @simd for p in currQtens.Qsize[j]
          altcounter += 1
          notcon[altcounter] = p
        end
      end
    end

    return con, notcon
  end
  export getinds

  """
      rowcolSize(sizes,inds)

  Obtains the size of the row or column of the matrix-equiavlent of a tensor with size `sizes` for indices `inds` composing the side of the matrix-equivalent of the Qtensor
  """
  function rowcolSize(sizes::Array{intType,1}, inds::Array{intType,1})
    return size(inds, 1) == 0 ? 1 : prod(a->sizes[a], inds)
  end
  export rowcolSize

  """
      resQtensInfo(QtensA, vecA, QtensB,vecB,numthreads)

  Helper function for quantum number contraction

  #Arguments:
  +`QtensA::Qtens{T,Q}`: quantum number tensor on the left of the contraction
  +`vecA::Array{X,1}`: indices to contract on `QtensA`
  +`QtensB::Qtens{R,Q}`: quantum number tensor on the right of the contraction
  +`vecB::Array{X,1}`: indices to contract on `QtensB`
  +`numthreads::X`: number of threads given to program

  #Output:
  +`conA`: indices that are contracted on `QtensA`
  +`notconA`: indices that are not contracted on `QtensA`
  +`conB`: indices that are contracted on `QtensB`
  +`notconB`: indices that are not contracted on `QtensB`
  +`sizeAA`: size of contracted tensor
  +`nrowsA`: number of rows of the matrix-equivalent of `A`
  +`ncolsA`: number of columns of the matrix-equivalent of `A`
  +`nrowsB`: number of rows of the matrix-equivalent of `B`
  +`ncolsB`: number of columns of the matrix-equivalent of `B`
  +`currpos`: one position vector for each parallel thread
  """
  function resQtensInfo(QtensA::Qtens{T,Q}, vecA::Array{X,1}, QtensB::Qtens{R,Q}, 
                        vecB::Array{X,1},numthreads::X)::Tuple{genarray,genarray,genarray,genarray,genarray,intType,intType,intType,intType,currpos_type} where X <: Integer where {T <: Number, R <: Number, Q <: Qnum}

    conA, notconA = getinds(QtensA, vecA)
    conB, notconB = getinds(QtensB, vecB)

    nrowsA = rowcolSize(QtensA.size, notconA)
    ncolsA = rowcolSize(QtensA.size, conA)
    nrowsB = rowcolSize(QtensB.size, conB)
    ncolsB = rowcolSize(QtensB.size, notconB)

    sizenotconA = size(notconA, 1)
    sizeAA = Array{intType,1}(undef, sizenotconA + size(notconB, 1))
    @simd for i = 1:sizenotconA
      sizeAA[i] = QtensA.size[notconA[i]]
    end
    @simd for i = 1:size(notconB,1)
      sizeAA[sizenotconA + i] = QtensB.size[notconB[i]]
    end

    maxsize = max(size(QtensA.size, 1), size(QtensB.size, 1))
    currpos = Array{intType,1}[Array{intType,1}(undef,maxsize) for i = 1:numthreads]

    return conA, notconA, conB, notconB, sizeAA, nrowsA, ncolsA, nrowsB, ncolsB, currpos
  end
  export resQtensInfo

  """
      makenewQtens(retQsize,newT,newinds,QtensA,vecA,QtensB,vecB,conA,notconA,conB,notconB,sizeAA,thisflux)
  
  Creates return Qtensor on contraction

  #Arguments:
  +`retQsize::intType`: return `Qsize` of the final Qtensor
  +`newT::Array{Number,1}`: non-zero elements of the final Qtensor
  +`newinds::Array{intType,1}`: indices of non-zero elements of the final Qtensor
  +`QtensA::Qtens{T,Q}`: left Qtensor in contraction
  +`vecA::Array{intType,1}`: original input for contracted indices on `QtensA`
  +`QtensB::Qtens{R,Q}`: right Qtensor in contraction
  +`vecB::Array{intType,1}`: original input for contracted indices on `QtensB`
  +`conA::Array{intType,1}`: contracted indices of `QtensA`
  +`notconA::Array{intType,1}`: uncontracted indices of `QtensA`
  +`conB::Array{intType,1}`: contracted indices of `QtensB`
  +`notconB::Array{intType,1}`: uncontracted indices of `QtensB`
  +`sizeAA::Array{intType,1}`: size of final Qtensor
  +`thisflux::Q`: flux of final Qtensor
  """
  function makenewQtens(newT::Array{S,1}, newinds::Array{intType,1}, 
              QtensA::Qtens{T,Q}, QtensB::Qtens{R,Q}, conA::Array{intType,1}, notconA::Array{intType,1},
              conB::Array{intType,1}, notconB::Array{intType,1}, sizeAA::Array{intType,1},
              thisflux::Q,conjA::Bool,conjB::Bool) where {T <: Number, R <: Number, Q <: Qnum, S <: Number}
    newQsize = Array{Array{intType,1},1}(undef, length(sizeAA))
    counter = 1
    veccounter = 0
    for i = 1:2
      currQtens = i == 1 ? QtensA : QtensB
      currvec = i == 1 ? conA : conB
      for j = 1:size(currQtens.Qsize, 1)
        condition = true
        for k in currvec
          if j == k
            condition = false
          end
        end
        if condition
          veccounter += 1
          if (size(currQtens.Qsize[j], 1) == 1)
            newQsize[veccounter] = intType[counter]
            counter += 1
          else
            newQsize[veccounter] = intType[counter + (z - 1) for z = 1:size(currQtens.Qsize[j], 1)]
            counter += size(currQtens.Qsize[j], 1)
          end
        end
      end
    end

    newQnumMat = Array{Array{Q,1},1}(undef, size(sizeAA, 1))
    newQnumSum = Array{Array{Q,1},1}(undef, size(sizeAA, 1))
    sizenotconA = size(notconA, 1)
    if conjA
      for a = 1:sizenotconA
        newQnumMat[a] = inv.(QtensA.QnumMat[notconA[a]])
        newQnumSum[a] = inv.(QtensA.QnumSum[notconA[a]])
      end
    else
      for a = 1:sizenotconA
        newQnumMat[a] = copy(QtensA.QnumMat[notconA[a]])
        newQnumSum[a] = copy(QtensA.QnumSum[notconA[a]])
      end
    end
    if conjB
      for b = 1:size(notconB, 1)
        newQnumMat[b + sizenotconA] = inv.(QtensB.QnumMat[notconB[b]])
        newQnumSum[b + sizenotconA] = inv.(QtensB.QnumSum[notconB[b]])
      end
    else
      for b = 1:size(notconB, 1)
        newQnumMat[b + sizenotconA] = copy(QtensB.QnumMat[notconB[b]])
        newQnumSum[b + sizenotconA] = copy(QtensB.QnumSum[notconB[b]])
      end
    end

    return Qtens{S,Q}(sizeAA, newQsize, newT, newinds, newQnumMat, newQnumSum, thisflux)
  end
  export makenewQtens

  """
      xyworker_full!(i,num,XYpos,currpos,A,rowvec,thisthread)

  computes the matrix equivalent position (x or y) or a tensor given which indices compose the rows/columns

  #Arguments:
  +`i::intType`: element of `T` field in `Qtens` to work with
  +`num::intType`: 1 or 2 for row or column
  +`XYpos::Array{intType,2}`: matrix holding all XY positions
  +`currpos::Array{Array{intType,1},1}`: position vectors for each thread
  +`A::qarray`: Qtensor
  +`rowvec::Array{intType,1}`: indices for either the row or the column
  +`thisthread::intType`: current thread

  See also: [`xyworker_one!`](@ref)  [`xyworker_zero!`](@ref)
  """
  @inline function xyworker_full!(i::intType, num::intType, XYpos::Array{intType,2}, currpos::Array{Array{intType,1},1},
            A::qarray, rowvec::Array{intType,1}, thisthread::intType) #where {W <: Integer,X <: Integer,Y <: Integer}
      XYpos[i,num] = currpos[thisthread][rowvec[size(rowvec, 1)]] * A.size[rowvec[size(rowvec, 1) - 1]]
      @simd for j = size(rowvec, 1) - 1:-1:2
        XYpos[i,num] += currpos[thisthread][rowvec[j]]
        XYpos[i,num] *= A.size[rowvec[j - 1]]
      end
      XYpos[i,num] += currpos[thisthread][rowvec[1]] + 1
    nothing
  end

  """
      xyworker_one!(i,num,XYpos,currpos,A,rowvec,thisthread)

  computes the matrix equivalent position (x or y) or a tensor given which indices compose the rows/columns if the size of the row or column is 1

  #Arguments:
  +`i::intType`: element of `T` field in `Qtens` to work with
  +`num::intType`: 1 or 2 for row or column
  +`XYpos::Array{intType,2}`: matrix holding all XY positions
  +`currpos::Array{Array{intType,1},1}`: position vectors for each thread
  +`A::qarray`: Qtensor
  +`rowvec::Array{intType,1}`: indices for either the row or the column
  +`thisthread::intType`: current thread

  See also: [`xyworker_full!`](@ref)  [`xyworker_zero!`](@ref)
  """
  @inline function xyworker_one!(i::intType, num::intType, XYpos::Array{intType,2}, currpos::Array{Array{intType,1},1},
           A::qarray, rowvec::Array{intType,1}, thisthread::intType) #where {W <: Integer,X <: Integer,Y <: Integer}
      XYpos[i,num] = currpos[thisthread][rowvec[1]] + 1
    nothing
  end

  """
      xyworker_zero!(i,num,XYpos,currpos,A,rowvec,thisthread)

  computes the matrix equivalent position (x or y) or a tensor given which indices compose the rows/columns if the size of the row or column is 0 (no remaining indices)

  #Arguments:
  +`i::intType`: element of `T` field in `Qtens` to work with
  +`num::intType`: 1 or 2 for row or column
  +`XYpos::Array{intType,2}`: matrix holding all XY positions
  +`currpos::Array{Array{intType,1},1}`: position vectors for each thread
  +`A::qarray`: Qtensor
  +`rowvec::Array{intType,1}`: indices for either the row or the column
  +`thisthread::intType`: current thread

  See also: [`xyworker_full!`](@ref)  [`xyworker_one!`](@ref)
  """
  @inline function xyworker_zero!(i::intType, num::intType, XYpos::Array{intType,2}, currpos::Array{Array{intType,1},1},
           A::qarray, rowvec::Array{intType,1}, thisthread::intType) #where {W <: Integer,X <: Integer,Y <: Integer}
      XYpos[i,num] = 1
    nothing
  end

  """
      findcurrpos!(i,currpos,A,thisthread)

  converts index to xy position in matrix equivalent of a tensor (parallelizable)

  #Arguments:
  +`i::intType`: current index of `A` to process
  +`currpos::Array{intType,2}`: position vector for each thread
  +`A::qarray`: Qtensor
  +`thisthread::intType`: current thread

  See also: [`startpos!`](@ref) [`nextpos!`](@ref)
  """
  @inline function findcurrpos!(i::intType, currpos::Array{Array{intType,1},1}, A::qarray, thisthread::intType) #where X <: Integer
    ind2zeropos!(currpos,thisthread,A.ind,i,A.size)
    nothing
  end

  """
      startpos!(pos,vec)

  initializes a position `pos` with (0,1,1,1,...) out to size of `vec`; alternative to `findcurrpos!`

  See also: [`findcurrpos!`](@ref) [`nextpos!`](@ref)
  """
  function startpos!(pos::Array{intType,1},vec::Array{intType,1}) #where X <: Integer where Y <: Integer
    pos[1] = 0
    @simd for w = 2:size(vec,1)
      pos[w] = 1
    end
    nothing
  end

  """
      nextpos!(pos,sizes)

  Increments position `pos` (used with `startpos!`) to find the next position according to `sizes`

  See also: [`findcurrpos!`](@ref) [`startpos!`](@ref)
  """
  @inline function nextpos!(pos::Array{intType,1},sizes::Array{intType,1}) #where X <: Integer where Y <: Integer where Z <: Integer
    thisone = 1
    pos[thisone] += 1
    while pos[thisone] > sizes[thisone] && thisone < size(sizes,1)
      pos[thisone] = 1
      thisone += 1
      pos[thisone] += 1
    end
    nothing
  end

  """
      choose_xyfct(vec::Array{intType,1})

  chooses correct function to make the XY position of the matrix-equivalent of the tensor

  See also: [`xyworker_full!`](@ref) [`xyworker_one!`](@ref) [`xyworker_zero!`](@ref)
  """
  function choose_xyfct(vec::Array{intType,1})::Function
    if length(vec) > 1
      return xyworker_full!
    end
    if length(vec) == 1
      return xyworker_one!
    end
    if length(vec) == 0
      return xyworker_zero!
    end
  end

  """
      makeXYpos(A,rowvec,colvec,currpos)

  generates a matrix of XY positions for the matrix-equivalent of `A` as determined by indices assigned in `rowvec` and `colvec` for rows and columns; `currpos` contains one position vector for each thread

  See also: [`xyworker_full!`](@ref) [`xyworker_one!`](@ref)
  """
  function makeXYpos(A::qarray, rowvec::Array{intType,1}, colvec::Array{intType,1},currpos::currpos_type)
    XYpos = Array{intType,2}(undef, size(A.T, 1), 2)
    condition = true
    counter = 1
    while condition && counter <= size(rowvec, 1)
      condition = condition && rowvec[counter] == counter
      counter += 1
    end
    nextcounter = 1
    while condition && nextcounter <= size(colvec, 1)
      condition = condition && colvec[nextcounter] == counter
      counter += 1
      nextcounter += 1
    end
    if condition
      Lsize = size(rowvec, 1) == 0 ? 1 : prod(a->A.size[a], rowvec)
      let A = A, XYpos = XYpos, Lsize = Lsize
        Threads.@threads     for i = 1:size(A.ind, 1)
          XYpos[i,1] = (A.ind[i] - 1) % Lsize + 1
          XYpos[i,2] = cld(A.ind[i], Lsize)
        end
      end
    else
      Lfct = choose_xyfct(rowvec)
      Rfct = choose_xyfct(colvec)
      let A = A, XYpos = XYpos, rowvec = rowvec, colvec = colvec,currpos = currpos, Lfct = Lfct, Rfct = Rfct
        Threads.@threads  for i = 1:size(A.ind, 1)
          thisthread = Threads.threadid()
          findcurrpos!(i, currpos, A, thisthread)
          Lfct(i, 1, XYpos, currpos, A, rowvec, thisthread)
          Rfct(i, 2, XYpos, currpos, A, colvec, thisthread)
        end
      end
    end
    return XYpos
  end
  export makeXYpos

  """
      qaddition!(maxind,vec,sizes,QnumVec,currpos,Qnums)

  adds all quantum numbers from `Qnums` into `QnumVec` as chosen by `vec`

  #Arguments:
  +`maxind::intType`: size of `QnumVec`
  +`vec::Array{intType,1}`: elements of `Qnums` to sum
  +`sizes::Array{intType,1}`: size of Qtensor
  +`QnumVec::Array{Q,1}`: output with summed quantum numbers
  +`currpos::Array{Array{intType,1},1}`: memory storage for position vectors generated
  +`Qnums::Array{Array{Q,1},1}`: quantum numbers to sum

  See also: [`qadd!`](@ref)
  """
  function qaddition!(maxind::intType,vec::Array{intType,1},sizes::Array{intType,1},QnumVec::Array{Q,1},
                      currpos::Array{Array{intType,1},1},Qnums::Array{Array{Q,1},1}) where Q <: Qnum
    let maxind = maxind, vec = vec, sizes = sizes, QnumVec = QnumVec, currpos = currpos, Qnums = Qnums #, thisflux = thisflux
      Threads.@threads   for i = 1:maxind
        thisthread = Threads.threadid()
        qadd!(i,currpos,thisthread,vec,sizes,QnumVec,Qnums)
      end
    end
    nothing
  end

  """
      copyadd!(QnumVec,i,Qnums,vec,currpos,thisthread)

  in-place copies and adds `Qnum`s together

  #Arguments:
  +`QnumVec::Array{Q,1}`: vector for return quantum numbers
  +`i::Integer`: index of `QnumVec`
  +`Qnums::Array{Array{Q,1},1}`: quantum numbers (`QnumMat or `QnumSum`)
  +`vec::Array{intType,1}`: elements of `Qnums` to add
  +`currpos::Array{Array{intType,1},1}`: positions for each thread
  +`thisthread::Integer`: current thread

  See also: [`Qnum`](@ref)
  """
  @inline function copyadd!(QnumVec::Array{Q,1},i::Integer,Qnums::Array{Array{Q,1},1},
                    vec::Array{intType,1},currpos::Array{Array{intType,1},1},
                    thisthread::Integer) where Q <: Qnum
    copy!(QnumVec[i],Qnums[vec[1]][currpos[thisthread][1]])
    for a = 2:length(vec)
      add!(QnumVec[i],Qnums[vec[a]][currpos[thisthread][a]])
    end
    nothing
  end

  """
      xcopyadd!(QnumVec,i,Qnums,vec,currpos,thisthread)

  in-place copies and adds `Qnum`s together

  #Arguments:
  +`QnumVec::Array{Q,1}`: vector for return quantum numbers but shorter than QnumVec (one per thread)
  +`i::Integer`: index of `QnumVec`
  +`Qnums::Array{Array{Q,1},1}`: quantum numbers (`QnumMat or `QnumSum`)
  +`vec::Array{intType,1}`: elements of `Qnums` to add
  +`currpos::Array{Array{intType,1},1}`: positions for each thread
  +`thisthread::Integer`: current thread

  See also: [`Qnum`](@ref) [`xcopyadd!`](@ref)
  """
  @inline function xcopyadd!(QnumVec::Array{Q,1},i::Integer,Qnums::Array{Array{Q,1},1},
                              vec::Array{intType,1},thiscurrpos::Array{intType,1},
                              thisthread::Integer) where Q <: Qnum
    copy!(QnumVec[thisthread],Qnums[vec[1]][thiscurrpos[1]])
    for a = 2:length(vec)
      add!(QnumVec[thisthread],Qnums[vec[a]][thiscurrpos[a]])
    end
    nothing
  end

  """
      qadd!(i,currpos,thisthread,vec,sizes,QnumVec,Qnums)

  adds all quantum numbers indexed by `vec` from `Qnums`into `QnumVec` (element `i`)

  #Arguments:
  +`i::intType`: element of `QnumVec` to access
  +`currpos::Array{Array{intType,1},1}`: position vectors for each thread
  +`thisthread::intType`: current thread
  +`vec::Array{intType,1}`: vector to pull quantum numbers from in `Qnums`
  +`sizes::Array{intType,1}`: sizes of Qtensor
  +`QnumVec::Array{Q,1}`: storage vector for quantum numbers
  +`Qnums::Array{Array{Q,1},1}`: 

  See also: [`qaddition!`](@ref)
  """
  @inline function qadd!(i::intType,currpos::Array{Array{intType,1},1},thisthread::intType,vec::Array{intType,1},
                  sizes::Array{intType,1},QnumVec::Array{Q,1},Qnums::Array{Array{Q,1},1}) where Q <: Qnum
    ind2pos!(currpos,thisthread,i,sizes)
    copyadd!(QnumVec,i,Qnums,vec,currpos,thisthread)
  end

  """
      revSum(summary,flux)

  Generates inverse of a summary of quantum numbers along minimal side of a Qtensor

  See also: [`makeSummary`](@ref)
  """
  function revSum(summary::Array{Q,1},flux::Q) where Q <: Qnum
    oppsum = inv.(summary)
    @simd for a = 1:length(oppsum)
      add!(oppsum[a],flux)
    end
    return oppsum
  end
  export revSum

  """
      makeSummary(thisQtens,vec,summaxind,LR,currpos,QnumVec[,thisflux=])

  makes summary on shorter side of matrix-equivalent of `thisQtens`

  #Arguments:
  +`thisQtens::Qtens{R,Qnum}`: Qtensor
  +`vec::Array{intType,1}`: contracted indices on `thisQtens`
  +`summaxind::intType`: size of row or column
  +`LR::Bool`: toggles which side to compute (true if shorter side is the row)
  +`currpos::Array{Array{intType,1},1}`: position vectors for each thread
  +`QnumVec::Array{Qnum,1}`: resulting quantum number vector
  +`thisflux::Qnum=thisQtens.flux`: tensor flux (default, flux of `thisQtens`)

  #Output:
  +summary for rows, summary for columns
  """
  function makeSummary(thisQtens::Qtens{R,Q},conjvar::Bool,vec::Array{intType,1},summaxind::intType,LR::Bool,
                      currpos::Array{Array{intType,1},1},QnumVec::Array{Q,1};thisflux::Q=thisQtens.flux) where {R <: Number, Q <: Qnum}
    if size(vec,1) == 0
      summary = Q[conjvar ? inv(thisflux) : thisflux]
    else
      sumsizes = intType[size(thisQtens.QnumSum[a], 1) for a in vec]
      qaddition!(summaxind,vec,sumsizes,QnumVec,currpos,thisQtens.QnumSum)
      summary = unique!(QnumVec)
    end
    oppsum = revSum(summary,thisQtens.flux)
    if LR
      return summary,oppsum
    else
      return oppsum,summary
    end
  end
  export makeSummary

  """
      QBlock(thisQtens,currpos,maxind,summary,vec,thisflux,QnumVec)

  Identifies sizes of quantum number blocks on the outer index by analyzing only the row or column of the Qtensor

  #Arguments:
  +`thisQtens::Qtens{T,Q}`: Qtensor
  +`currpos::Array{Array{intType,1},1}`: position for every thread
  +`maxind::intType`: maximum number of non-zero indices in `thisQtens`
  +`summary::Array{Qnum,1}`: summary of quantum numbers
  +`vec::Array{intType,1}`: contracted indices
  +`thisflux::Qnum`: flux of `thisQtens`
  +`QnumVec::Array{Qnum,1}`: quantum number vector summed on smaller of row or column of matrix-equivalent

  #Outputs:
  +`finalxy`: row or column of sub-matrix in a quantum block sector
  +`countsizes`: number of rows or columns in a sub-block
  +`finalQN`: quantum number on each element of row or column

  See also: [`innerQBlock`](@ref)
  """
  function QBlock(thisQtens::Qtens{T,Q}, currpos::Array{Array{intType,1},1}, maxind::intType,
                    summary::Array{Q,1}, vec::Array{intType,1}, thisflux::Q, QnumVec::Array{Q,1}) where {T <: Number,Q <: Qnum}

    finalxy = Array{intType,1}(undef,maxind)
    countsizes = zeros(intType, size(summary,1))

    sizes = intType[thisQtens.size[a] for a in vec]

    Qnums = thisQtens.QnumMat

    finalQN = zeros(intType, maxind)
    thisthread = Threads.threadid()

    thiscurrpos = currpos[1]
    startpos!(thiscurrpos,sizes)

    for i = 1:maxind
      nextpos!(thiscurrpos,sizes)
      xcopyadd!(QnumVec,i,Qnums,vec,thiscurrpos,thisthread)

      a = 0
      condition = true
      while a < size(summary,1) && condition
        a += 1
        if QnumVec[thisthread] == summary[a]
          finalQN[i] = a
          countsizes[a] += 1
          finalxy[i] = countsizes[a]
          condition = false
        end
      end
    end
    return finalxy, countsizes, finalQN
  end
  export QBlock

  """
      innerQBlock(thisQtens,currpos,maxind,summary,vec,thisflux,QnumVec)

  Identifies sizes of quantum number blocks on the outer index by analyzing only the row or column of the Qtensor

  #Arguments:
  +`thisQtens::Qtens{T,Q}`: Qtensor
  +`currpos::Array{Array{intType,1},1}`: position for every thread
  +`maxind::intType`: maximum number of non-zero indices in `thisQtens`
  +`summary::Array{Qnum,1}`: summary of quantum numbers
  +`vec::Array{intType,1}`: contracted indices
  +`thisflux::Qnum`: flux of `thisQtens`
  +`QnumVec::Array{Qnum,1}`: quantum number vector summed on smaller of row or column of matrix-equivalent

  #Outputs:
  +`finalxy`: row or column of sub-matrix in a quantum block sector
  +`countsizes`: number of rows or columns in a sub-block
  +`finalQN`: quantum number on each element of row or column

  See also: [`QBlock`](@ref)
  """
  function innerQBlock(thisQtens::Qtens{T,Q}, currpos::Array{Array{intType,1},1}, maxind::intType,
                    summary::Array{Q,1}, vec::Array{intType,1}, thisflux::Q, QnumVec::Array{Q,1}) where {T <: Number,Q <: Qnum}

    finalxy = Array{intType,1}(undef,maxind)
    countsizes = zeros(intType, size(summary,1))

    sizes = intType[thisQtens.size[a] for a in vec]

    Qnums = thisQtens.QnumMat

    finalQN = zeros(intType, maxind)
    thisthread = Threads.threadid()

    thiscurrpos = currpos[1]

    startpos!(thiscurrpos,sizes)

    for i = 1:maxind
      nextpos!(thiscurrpos,sizes)
      xcopyadd!(QnumVec,i,Qnums,vec,thiscurrpos,thisthread)

      a = 0
      condition = true
      while a < size(summary,1) && condition
        a += 1
        if QnumVec[thisthread] == summary[a]
          countsizes[a] += 1
          finalxy[i] = countsizes[a]
          condition = false
        end
      end
    end

    return finalxy, countsizes
  end
  export innerQBlock

  """
      sumBlockInfo(QtensA,conA,notconA,sumALsizes,QtensB,currpos,ncolsA)

  Returns information on the sub-blocks of `QtensA`

  #Arguments:
  +`QtensA::Qtens{T,Q}`: left Qtensor in contraction
  +`conA::Array{Int64,1}`: indices to contract
  +`notconA::Array{Int64,1}`: indices not to contract
  +`sumALsizes::Int64`: size of the summary on the left
  +`QtensB::qarray`: right Qtensor in contraction
  +`currpos::Array{Array{intType,1},1}`: position vector for each thread
  +`ncolsA::intType`: number of columns of `QtensA`

  #Outputs:
  +`notconQnumSumA`: summary of quantum numbers on non-contracted indices of `A`
  +`conQnumSumA`: summary of quantum numbers on contracted indices of `A`
  +`conQnumSumB`: summary of quantum numbers on contracted indices of `B`
  +`notconQnumSumB`: summary of quantum numbers on non-contracted indices of `B`
  +`innerfinalxy`: XY positions of sub-block on contracted indices
  +`innercountsizes`: size of quantum number sectors on contracted indices
  +`qvec`: vector of quantum numbers for shortest side between both tensors
  """
  function sumBlockInfo(QtensA::Qtens{T,Q},conA::Array{Int64,1},notconA::Array{Int64,1},conjA::Bool,
                        sumALsizes::Int64,QtensB::qarray,currpos::Array{Array{intType,1},1},ncolsA::intType,
                        conjB::Bool) where {T <: Number, Q <: Qnum}
    sumARsizes = prod(a->length(QtensA.QnumSum[a]),conA)
    LR = sumALsizes < sumARsizes
    vect = LR ? notconA : conA
    maxvecsize = LR ? sumALsizes : sumARsizes

    numthreads = Threads.nthreads()
    qvec = Q[Q() for i = 1:max(maxvecsize,numthreads,3)] #3 for each other side we need to calculate in parallel

    notconQnumSumA,conQnumSumA = makeSummary(QtensA,conjA,vect,maxvecsize,LR,currpos,qvec)
    if conjA != conjB
      condition = conjA ? !conjB : conjB
    else
      condition = false
    end
    conQnumSumB = condition ? copy(conQnumSumA) : inv.(conQnumSumA)
    notconQnumSumB = revSum(conQnumSumB,QtensB.flux)

    innerfinalxy,innercountsizes = innerQBlock(QtensA,currpos,ncolsA,conQnumSumA,conA,QtensA.flux,qvec)
    return notconQnumSumA,conQnumSumA,conQnumSumB,notconQnumSumB,innerfinalxy,innercountsizes,qvec
  end
  export sumBlockInfo

  """
      QNsearch(submat,A,rowpos,colpos,XYpos,oneOrTwo,AfinalQN,x,maxfield)

  (parallelized) finds which elements of a tensor belong with which quantum number

  #Arguments:
  +`submat::Array{T,2}`: sub-block
  +`A::Qtens{T,Q}`: Qtensor
  +`rowpos::Array{intType,1}`: row position in the sub-block for the full row
  +`colpos::Array{intType,1}`: column positions in the sub-block for the full column
  +`XYpos::Array{intType,2}`: XY positions for Qtensor
  +`oneOrTwo::intType`: evaluate left or right of matrix-equivalent
  +`AfinalQN::Array{intType,1}`: quantum number on each element of the row or column
  +`x::intType`: quantum number position in summary
  +`maxfield::Array{intType,1}`: maximum size of row or column

  See also: [`altQNsearch`](@ref)
  """
  function QNsearch(submat::Array{T,2}, A::Qtens{R,Q}, rowpos::Array{intType,1}, colpos::Array{intType,1},XYpos::Array{intType,2}, oneOrTwo::intType, 
          AfinalQN::Array{intType,1}, x::intType, maxfield::Array{intType,1}) where {T <: Number, R <: Number, Q <: Qnum}
    let XYpos = XYpos, AfinalQN = AfinalQN, submat = submat, x = x, A = A, rowpos = rowpos, colpos = colpos, oneOrTwo = oneOrTwo
      Threads.@threads   for w = 1:size(XYpos, 1)
        XYind = XYpos[w,oneOrTwo]
        if AfinalQN[XYind] == x
          xpos = rowpos[XYpos[w,1]]
          ypos = colpos[XYpos[w,2]]
          submat[xpos,ypos] = A.T[w]
        end
      end
    end
    return makeRetRowCol(AfinalQN, x, maxfield)
  end
  export QNsearch

  """
      altQNsearch(submat,A,rowpos,colpos,XYpos,oneOrTwo,AfinalQN,x,maxfield,countElem)

  finds which elements of a tensor belong with which quantum number

  #Arguments:
  +`submat::Array{T,2}`: sub-block
  +`A::Qtens{T,Q}`: Qtensor
  +`rowpos::Array{intType,1}`: row position in the sub-block for the full row
  +`colpos::Array{intType,1}`: column positions in the sub-block for the full column
  +`XYpos::Array{intType,2}`: XY positions for Qtensor
  +`oneOrTwo::intType`: evaluate left or right of matrix-equivalent
  +`AfinalQN::Array{intType,1}`: quantum number on each element of the row or column
  +`x::intType`: quantum number position in summary
  +`maxfield::Array{intType,1}`: maximum size of row or column
  +`countElem::Array{intType,1}`: count elements found to check against total size later (determines if a zeros or Array can initialize the sub-block and allows for recomputation in `contractions`)

  See also: [`QNsearch`](@ref)
  """
  function altQNsearch(submat::Array{T,2}, A::Qtens{R,Q}, rowpos::Array{intType,1}, colpos::Array{intType,1},XYpos::Array{intType,2}, oneOrTwo::intType, 
    AfinalQN::Array{intType,1}, x::intType, maxfield::Array{intType,1}, countElem::Array{intType,1}) where {T <: Number, R <: Number, Q <: Qnum}
  let countElem = countElem, XYpos = XYpos, AfinalQN = AfinalQN, submat = submat, x = x, A = A, rowpos = rowpos, colpos = colpos, oneOrTwo = oneOrTwo
    Threads.@threads     for w = 1:size(XYpos, 1)
      XYind = XYpos[w,oneOrTwo]
      if AfinalQN[XYind] == x
        countElem[Threads.threadid()] += 1
        xpos = rowpos[XYpos[w,1]]
        ypos = colpos[XYpos[w,2]]
        submat[xpos,ypos] = A.T[w]
      end
    end
  end
  return makeRetRowCol(AfinalQN, x, maxfield)
  end
  export altQNsearch

  """
      loadSubmat(Acountsizes,innercountsizes,numthreads,QtensA,Afinalxy,innerfinalxy,Axypos,AfinalQN,x,rowcol,fullA,typeA)

  creates a sub-matrix and corresponding dictionary back to the full tensor for a given quantum number

  #Arguments:
  +`searchAcountsizes::Array{intType,1}`: number of the rows of each sub-block
  +`Acountsizes::Array{intType,1}`: same as `searchAcountsizes` but can be interchanged with the next field
  +`innercountsizes::Array{intType,1}`: number of the columns of each sub-block
  +`numthreads::intType`: number of threads
  +`QtensA::qarray`: Qtensor
  +`Afinalxy::Array{intType,1}`: back dictionary for rows of sub-block to identify positions in large tensor
  +`innerfinalxy::Array{intType,1}`:  back dictionary for columns of sub-block to identify positions in large tensor
  +`Axypos::Array{intType,2}`: XY position of Qtensor
  +`AfinalQN::Array{intType,1}`: the quantum number along a given row
  +`x::intType`: index for sub-block (corresponds to quantum number summary)
  +`rowcol::intType`: use row or column of XYposition to check quantum number
  +`fullA::Bool`: whether Qtensor is full and sub-blocks can be initialized with zeros or Arrays (undefined)
  +`typeA::DataType`: type of element in sub-matrix
  """
  function loadSubmat(searchAcountsizes::Array{intType,1},Acountsizes::Array{intType,1},innercountsizes::Array{intType,1},
            numthreads::intType,QtensA::qarray,Afinalxy::Array{intType,1},innerfinalxy::Array{intType,1},Axypos::Array{intType,2},
            AfinalQN::Array{intType,1},x::intType,rowcol::intType,fullA::Bool,typeA::DataType)
    if fullA
      submatA = Array{typeA,2}(undef,Acountsizes[x],innercountsizes[x])
      counterA = zeros(intType,numthreads)
      retArows = altQNsearch(submatA,QtensA,Afinalxy,innerfinalxy,Axypos,rowcol,AfinalQN,x,searchAcountsizes,counterA)
      if sum(counterA) != prod(size(submatA))
        submatA = zeros(typeA,Acountsizes[x],innercountsizes[x])
        retArows = QNsearch(submatA,QtensA,Afinalxy,innerfinalxy,Axypos,rowcol,AfinalQN,x,searchAcountsizes)
      end
    else
      submatA = zeros(typeA,Acountsizes[x],innercountsizes[x])
      retArows = QNsearch(submatA,QtensA,Afinalxy,innerfinalxy,Axypos,rowcol,AfinalQN,x,searchAcountsizes)
    end
    return submatA,retArows
  end
  export loadSubmat

  #need to make this a separate loop since we don't want to keep overwriting the value in the return array if we go element by element
  """
      makeRetRowCol(AfinalQN, x, maxfield)

  prepares the position in the final tensor from the quantum number on that row or column `AfinalQN`, the quantum number `x`, and the size of that row or column (the backward map between the sub-block's labeling of rows and columns and the rows and columns of the final tensor)

  See also: [`QBlock`](@ref)
  """
  function makeRetRowCol(AfinalQN::Array{intType,1}, x::intType, maxfield::Array{intType,1})
    p = 1
    counter = 0
    retArows = Array{intType,1}(undef, maxfield[x])
    while counter < maxfield[x] && p <= size(AfinalQN, 1)
      if AfinalQN[p] == x
        counter += 1
        retArows[counter] = p
      end
      p += 1
    end
    return retArows
  end
  export makeRetRowCol

  """
      retsubBlock!(nrowsA,retArows,retBcols,newT,newpos,newmat,counter)

  loads the new Qtensor with the correct values

  #Arguments:
  +`nrowsA::intType`: size of rows of the sub-block
  +`retArows::Array{intType,1}`: rows of sub-block corresponding to these rows in the full matrix-equivalent
  +`retBcols::Array{intType,1}`: columns of sub-block corresponding to these columns in the full matrix-equivalent
  +`newT::Array{T,1}`: non-zero elements in new Qtensor
  +`newpos::Array{intType,1}`: indices of non-zero elements in Qtensor
  +`newmat::Array{T,2}`: sub-block
  +`counter::intType`: offset for single array containing non-zero elements
  """
  function retsubBlock!(nrowsA::intType, retArows::Array{intType,1}, retBcols::Array{intType,1},
              newT::Array{T,1}, newpos::Array{intType,1}, newmat::Array{W,2}, counter::intType) where {T <: Number, W <: Number} #W <: Integer,X <: Integer,T <: Number,Y <: Integer,Z <: Integer}
    let newmat = newmat, nrowsA = nrowsA, retBcols = retBcols, counter = counter, newT = newT, newpos = newpos, retArows = retArows
      Threads.@threads   for j = 1:size(newmat, 2)
        thisind = nrowsA * (retBcols[j] - 1)
        storeind = counter + size(newmat, 1) * (j - 1)
        @simd for i = 1:size(newmat, 1)
          thisotherind = storeind + i
          newT[thisotherind] = newmat[i,j]
          newpos[thisotherind] = retArows[i] + thisind
        end
      end
    end
    nothing
  end
  export retsubBlock!

#        +-------------------------------+
#>-------|  Functions for decomposition  |-----------<
#        +-------------------------------+

  """
      findmatches(Ypos,rankedinds)

  finds elements of a tensor `Ypos` that will not be truncated (i.e., those in `rankedinds`)

  See also: [`makeUnitary`](@ref)
  """
  function findmatches(Ypos::Array{intType,1},rankedinds::Array{intType,1})
    isranked = Array{Bool,1}(undef, size(Ypos, 1))
    let Ypos = Ypos, isranked = isranked, rankedinds = rankedinds
      Threads.@threads   for i = 1:size(Ypos, 1)
        isranked[i] = Ypos[i] in rankedinds
      end
    end
    return findall(isranked)
  end

  """
      makeUnitary!(T,Xpos,Ypos,rankedinds,nrows,LR)

  makes non-zero, not truncated elements and indices of a Qtensor

  #Arguments:
  +`T::Array{Y,1}`: non-zero elements of Qtensor
  +`Xpos::Array{intType,1}`: row position
  +`Ypos::Array{intType,1}`: column position
  +`rankedinds::Array{intType,1}`: order of indices for newly created index (see `findmatches`), keeps only those not truncated
  +`nrows::intType`: size of the first dimension of the row-equivalent of the tensor
  +`LR::Bool`: affects how final index is computed (true for U, false for V)

  See also: [`findmatches`](@ref)
  """
  function makeUnitary(T::Array{Y,1}, Xpos::Array{intType,1}, Ypos::Array{intType,1},
             rankedinds::Array{intType,1}, nrows::intType, LR::Bool) where {X <: Integer, Y <: Number}
    pulled = findmatches(Ypos,rankedinds)
    pulling = Ypos[pulled]

    if !(issorted(pulling))
      sort!(pulling)
    end
    uniqueY = intType[pulling[1]]
    for i = 2:size(pulling,1)
      if pulling[i-1] != pulling[i]
        push!(uniqueY,pulling[i])
      end
    end

    maxEl = maximum(uniqueY)
    newInnerVals = Array{intType,1}(undef, maxEl)
    counter = 0
    @simd for p in uniqueY
      counter += 1 
      newInnerVals[p] = counter
    end
    retType = eltype(T)
    altnewT = Array{retType,1}(undef, size(pulled, 1))
    newinds = Array{intType,1}(undef, size(pulled, 1))

    if LR
      let newinds = newinds, pulled=pulled, T = T, altnewT = altnewT, Xpos = Xpos, Ypos = Ypos, newInnerVals = newInnerVals
        Threads.@threads  for i = 1:size(pulled, 1)
          p = pulled[i]
          altnewT[i] = T[p]
          newinds[i] = Xpos[p] + nrows * (newInnerVals[Ypos[p]] - 1)
        end
      end
    else
      let newinds = newinds, pulled=pulled, T = T, altnewT = altnewT, Xpos = Xpos, Ypos = Ypos, newInnerVals = newInnerVals
        Threads.@threads  for i = 1:size(pulled, 1)
          p = pulled[i]
          altnewT[i] = T[p]
          newinds[i] = newInnerVals[Ypos[p]] + nrows * (Xpos[p] - 1)
        end
      end
    end
    return altnewT,newinds
  end
  export makeUnitary

  """
      initmatter(QtensAA)

  makes initial information for the decomposition available

  #Output:
  +`Linds`: indices on the left
  +`Rinds`: indices on the right
  +`Lsize`: number of indices on the left
  +`Rsize`: number of indices on the right
  +`AAxypos`: XY positions of all indices in Qtensor
  +`LR`: true if left is smaller
  +`nrowsAA`: size of the row of the matrix-equivalent
  +`ncolsAA`: size of the column of the matrix-equivalent
  +`currpos`: position vector for every thread
  +`zeroQN`: zero quantum number
  +`AALfinalxy`: row of the sub-block for each sector
  +`AALcountsizes`: size of rows of each sub-block
  +`AALfinalQN`: which quantum number in `conQnumSumAAL` belongs to this row
  +`conQnumSumAAL`: summary of quantum numbers on the left
  +`AARfinalxy`: column of the sub-block for each sector
  +`AARcountsizes`: size of columns of each sub-block
  +`AARfinalQN`: which quantum number in `conQnumSumAAR` belongs to this row
  +`conQnumSumAAR`: summary of quantum numbers on the right
  +`subblocksizes`: total elements in all sub-blocks
  +`mindims`: minimum dimensions on created index in SVD
  +`Ablocksizes`: total non-zero elements in `U` of an SVD by block
  +`Bblocksizes`: total non-zero elements in `V` of an SVD by block
  +`totRetSize`: total number of elements in all sub-blocks
  +`sum(Ablocksizes)`: total non-zero elements in `U` of an SVD
  +`sum(Bblocksizes)`: total non-zero elements in `V` of an SVD
  +`totRetSize == size(QtensAA.T, 1)`: is the Qtensor full (or is it missing elements)?
  """
  function initmatter(QtensAA::Qtens{Z,Q}) where {Q <: Qnum, Z <: Number}
    Linds = QtensAA.Qsize[1]
    Rinds = QtensAA.Qsize[2]
    Lsize = size(Linds, 1)
    Rsize = size(Rinds, 1)

    LR = Lsize < Rsize
    numthreads = Threads.nthreads()

    currpos = Array{intType,1}[Array{intType,1}(undef, Rsize + Lsize) for i = 1:numthreads]

    AAxypos = makeXYpos(QtensAA, Linds, Rinds, currpos)

    nrowsAA = rowcolSize(QtensAA.size, Linds)
    ncolsAA = rowcolSize(QtensAA.size, Rinds)
    zeroQN = typeof(QtensAA.flux)()

    sumAALsizes = prod(a->length(QtensAA.QnumSum[a]),Linds)
    sumAARsizes = prod(a->length(QtensAA.QnumSum[a]),Rinds)

    LR = sumAALsizes < sumAARsizes
    if LR
      vec = Linds
      maxvecsize = sumAALsizes
      qvec = Q[Q() for i = 1:max(maxvecsize,3)]
      conQnumSumAAL,conQnumSumAAR = makeSummary(QtensAA,false,vec,maxvecsize,LR,currpos,qvec)
    else
      vec = Rinds
      maxvecsize = sumAARsizes
      qvec = Q[Q() for i = 1:max(maxvecsize,3)]
      conQnumSumAAL,conQnumSumAAR = makeSummary(QtensAA,false,vec,maxvecsize,LR,currpos,qvec)
    end

    AALfinalxy, AALcountsizes, AALfinalQN = QBlock(QtensAA, currpos, nrowsAA, conQnumSumAAL, Linds, zeroQN,qvec)
    AARfinalxy, AARcountsizes, AARfinalQN = QBlock(QtensAA, currpos, ncolsAA, conQnumSumAAR, Rinds, QtensAA.flux,qvec)
    
    subblocksizes = Array{Int64,1}(undef,length(AALcountsizes))
    mindims = Array{Int64,1}(undef,length(AALcountsizes))
    Ablocksizes = Array{Int64,1}(undef,length(AALcountsizes))
    Bblocksizes = Array{Int64,1}(undef,length(AALcountsizes))
    @simd for x = 1:size(AALcountsizes, 1)
      subblocksizes[x] = AALcountsizes[x] * AARcountsizes[x]
      mindims[x] = min(AALcountsizes[x], AARcountsizes[x])
      Ablocksizes[x] = AALcountsizes[x] * mindims[x]
      Bblocksizes[x] = mindims[x] * AARcountsizes[x]
    end
    totRetSize = sum(subblocksizes)

    return Linds, Rinds, Lsize, Rsize, AAxypos, LR, nrowsAA, ncolsAA, currpos, zeroQN,
        AALfinalxy, AALcountsizes, AALfinalQN, conQnumSumAAL, AARfinalxy, AARcountsizes, AARfinalQN,
        conQnumSumAAR, subblocksizes, mindims, Ablocksizes, Bblocksizes,totRetSize, 
        sum(Ablocksizes), sum(Bblocksizes),totRetSize == size(QtensAA.T, 1)
  end
  export initmatter

  """
      makeBoundary(qind,newArrows[,retType=])

  makes a boundary tensor for an input from the quantum numbers `qind` and arrows `newArrows`; can also define type of resulting Qtensor `retType` (default `Float64`)

  #Note:
  +dense tensors are just ones(1,1,1,...)

  See also: [`makeEnds`](@ref)
  """
  function makeBoundary(qind::Array{Q,1},newArrows::Array{Bool,1};retType::DataType=Float64) where Q <: Qnum
    qsize = length(qind)
    thisQnumMat = Array{Array{Q,1},1}(undef,qsize)
    flux = Q()
    let qsize = qsize, qind = qind, newArrows = newArrows, thisQnumMat = thisQnumMat, flux = flux
      Threads.@threads for j = 1:qsize
        q = newArrows[j] ? qind[j] : inv(qind[j])
        add!(flux,q)
        thisQnumMat[j] = typeof(q)[q]
      end
    end
    return Qtens(retType[1.],intType[1],thisQnumMat,flux)
  end
  export makeBoundary

  """
      applylocalF!(tens, i)

  (in-place) effectively apply a chain of F (fermion phase) operator by a single manipulation on the specified index.

  See also: [`applylocalF`](@ref)
  """
  function applylocalF!(tens::TT, i) where {TT <: qarray}
#    @assert typeof(tens.flux) <: fermionQnum
    for (j, (t, index)) in enumerate(zip(tens.T, tens.ind))
      pos = ind2pos(index, tens.size)
      p = parity(tens.QnumMat[i][pos[i]])
      tens.T[j] *= (-1)^p
    end
  end

  """
      applylocalF(tens, i)

  effectively apply a chain of F (fermion phase) operator by a single manipulation on the specified index.

  See also: [`applylocalF!`](@ref)
  """
  function applylocalF(tens::T, i) where {T <: qarray}
    W = copy(tens)
    applylocalF!(W, i)
    return W 
  end
  export applylocalF,applylocalF!

  function makeqOp(operator::AbstractArray,Qlabels::Array{Q,1},Arrows::Array{Bool,1}...;conj::Bool=false) where Q <: Qnum
    println("WARNING: DEPRECATED")
    return Qtens(operator,[Qlabels for i = 1:ndims(operator)],arrows...,conj=conj)
  end
  function makeqOp(operator::AbstractArray,Qlabels::Array{Array{Q,1},1},Arrows::Array{Bool,1}...;conj::Bool=false) where Q <: Qnum
    println("WARNING: DEPRECATED")
    return Qtens(operator,Qlabels,arrows...,conj=conj)
  end
  export makeqOp

  """
      concat!(vec,A,B)

  in-place concatenatation of tensors `A` (replaced for Qtensors only) and `B` along indices specified in `vec`
  """
  function concat!(bareinds::intvecType,A::Array{R,N},B::Array{S,N})::Array{typeof(R(1)*S(1)),N} where {R <: Number, S <: Number, N}
    inds = convIn(bareinds)
#=    """
    concatenate two tensors, the index not mentionned retain their original value, for those mentionned, the index of B is augmented by the size of A in the same direction.
    if A and B are matrices
      And both indexes are present then the output is [A 0;0 B], this is the direct sum.
      if only one of the index is present the result is either [A B] or [A ; B]
    """=#
#=    if warning
      @assert(length(inds) > 0)
      @assert(length(size(A))==length(size(B))) #they must have the same rank
      @assert(all([ (i in inds) || a==b  for (i,(a,b)) in enumerate(zip(size(A),size(B)))])) # the direction not mention must have the same length.
    end=#
    shapeout = [a + ( (i in inds) ? b : 0 ) for (i,(a,b)) in enumerate(zip(size(A),size(B)))]
    thistype = typeof(eltype(A)(1)*eltype(B)(1))
    out = zeros(thistype,shapeout...)
    fora = [1:a for a in size(A)]
    forb = [1:b for b in size(B)]
    fora2b = [((i in inds) ? a : 0 )+1:b+((i in inds) ? a : 0 ) for (i,(a,b)) in enumerate(zip(size(A),size(B))) ]
    out[fora...] = A[fora...]
    out[fora2b...] =  B[forb...]
    return out
  end

  function concat!(bareinds::intvecType,A::tens{S},B::tens{W}) where {W <: Number, S <: Number}
    rA = reshape(A.T,A.size...)
    rB = reshape(B.T,B.size...)
    C = concat!(bareinds,rA,rB)
    retType = typeof(S(1)*W(1))
    return tens(retType,C)
  end
  
  function concat!(bareinds::intvecType,A::Qtens{R,Q},B::Qtens{S,Q}) where {R <: Number, S <: Number, Q <: Qnum}
    inds = convIn(bareinds)
#=    function cinv(xx::Qnum,cond::Bool)
      #conditionnal inversion
      if cond
        return inv(xx)
      end
      return xx
    end
    if warning
      @assert(length(inds) > 0)# we don't do element-wise addition here.
      @assert(length(size(A))==length(size(B))) #they must have the same rank
      @assert(cinv(A.flux,A.conjugated) == cinv(B.flux,B.conjugated)) #must have the same flux, can be adjusted externally by playing with the gauge.
      @assert(all([ (i in inds) || a==b for (i,(a,b)) in enumerate(zip(size(A),size(B)))])) # the direction not mention must have the same length.
      @assert(all([ (i in inds) || all(@. cinv(a,A.conjugated) == cinv(b,B.conjugated)) for (i,(a,b)) in enumerate(zip(A.QnumMat,B.QnumMat))])) # the direction not mention must have the same Qnums in the same order.
    end=#
    
    shapeout = copy(A.size)
    @simd for w = 1:length(inds)
      shapeout[inds[w]] += B.size[inds[w]]
    end
    Ttype = typeof(eltype(A.T)(0)*eltype(B.T)(0))
    Tout = Array{Ttype,1}(undef,length(A.T)+length(B.T))
    indsout = Array{intType,1}(undef,length(Tout))
    La = length(A.ind)

    numthreads = Threads.nthreads()
    maxlen = max(length(A.size),length(B.size))
    thispos = [Array{intType,1}(undef,maxlen) for i = 1:numthreads]

    let A = A, B = B, thispos = thispos, shapeout = shapeout, indsout = indsout, inds = inds, La = La
      Threads.@threads  for i = 1:length(A.ind)
        thisthread = Threads.threadid()
        ind2pos!(thispos,thisthread,A.ind,i,A.size)
        indsout[i] = pos2ind(thispos[thisthread],shapeout)
        Tout[i] = A.T[i]
      end
      Threads.@threads  for i = 1:length(B.ind)
        thisthread = Threads.threadid()
        ind2pos!(thispos,thisthread,B.ind,i,B.size)
        @simd for w = 1:length(inds)
          thispos[thisthread][inds[w]] += A.size[inds[w]]
        end
        indsout[La+i] = pos2ind(thispos[thisthread],shapeout)
        Tout[La+i] = B.T[i]
      end
    end
    A.size = shapeout
    A.T = Tout
    A.ind = indsout

    deltaflux = A.flux + inv(B.flux)
    for w = 1:length(inds)
      Aqn = A.QnumMat[inds[w]]
      Bqn = B.QnumMat[inds[w]]
      newQnums = Array{Q,1}(undef,length(Aqn) + length(Bqn))
      let Aqn = Aqn, Bqn = Bqn, newQnums = newQnums, deltaflux = deltaflux
        Threads.@threads  for j = 1:length(Aqn)
          newQnums[j] = Aqn[j]
        end
        Threads.@threads  for j = 1:length(Bqn)
          newQnums[length(Aqn)+j] = Bqn[j] + deltaflux
        end
      end
      A.QnumMat[inds[w]] = newQnums
      A.QnumSum[inds[w]] = unique(newQnums)
    end
    return A::Qtens{R,Q}
  end

  """
      concat(vec,A,B...)

  Concatenatation of tensors `A` and any number of `B` along indices specified in `vec`
  """
  function concat(inds::intvecType,A::W,B::R...) where {W <: TensType, R <: TensType}
    if typeof(A) <: qarray
      C = copy(A)
    else
      C = A
    end
    C = concat!(inds,C,B...)
    return C
  end

  """
      concat(A,B,vec)

  Concatenatation of tensors `A` and any number of `B` along indices specified in `vec`
  """
  function concat(A::W,B::R,inds::intvecType) where {W <: TensType, R <: TensType}
    mA,mB = checkType(A,B)
    return concat(inds,mA,mB)
  end
  export concat

  """
      concat!(A,B,vec)

  In-place concatenatation of tensors `A` and any number of `B` along indices specified in `vec`
  """
  function concat!(A::W,B::R,inds::intvecType) where {W <: TensType, R <: TensType}
    mA,mB = checkType(A,B)
    return concat!(inds,mA,mB)
  end
  export concat!
end