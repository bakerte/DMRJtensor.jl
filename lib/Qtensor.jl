#########################################################################
#
#  Density Matrix Renormalization Group (and other methods) in julia (DMRjulia)
#                              v1.0
#
#########################################################################
# Made by Thomas E. Baker (2020)
# See accompanying license with this program
# This code is native to the julia programming language (v1.1.1+)
#

"""
    Module: Qtensor

Stores functions for the quantum number tensor (Qtensor) type

See also: [`Qtask`](@ref)
"""
module Qtensor
using ..tensor
using ..QN
import LinearAlgebra

  import .tensor.checkType
  """
      checkType(A,B)
  
  Checks types between `A` and `B` to make uniform (ex: Array and denstens converts to two denstens)
  """
  function checkType(A::qarray,B::qarray)
    return A,B
  end
  export checkType

  """
      Qtens{T,Q}

  Qtensor; defined as `Qtens{T,Q}` for T <: Number and Q <: Qnum

  # Fields:
  + `size::Array{intType,1}`: size of base tensor (unreshaped)
  + `T::Array{Z,1}`: Array containing non-zero blocks' values of the tensor
  + `ind::Array{intType,1}`: indices of the stored values
  + `QnumMat::Array{Array{Q,1},1}`: quantum numbers on each index
  + `QnumSum::Array{Array{Q,1},1}`: summary of quantum numbers on each index
  + `flux::Q`: total quantum number flux on tensor
  + `conjugated::Bool`: toggle to conjugate tensor without flipping all fluxes

  See also: [`Qnum`](@ref) [`makedens`](@ref) [`checkflux`](@ref)
  """
  mutable struct Qtens{W <: Number,Q <: Qnum} <: qarray
    size::Array{Array{intType,1},1} #the size of the tensor if it were represented densely
    #^This is an array since it can change on reshape
    T::Array{tens{W},1}
    ind::Array{Array{Array{intType,2},1},1}
    currblock::Array{Array{intType,1},1}
    Qblocksum::Array{Array{Q,1},1}
    QnumMat::Array{Array{intType,1},1} #quantum numbers on each index
    QnumSum::Array{Array{Q,1},1} #summary of indices on each index
    flux::Q #flux = sum of other indices
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
      Qtens(Qlabels[,arrows,datatype=])

  Creates empty `Qtens` with array type `Type` (default Float64), quantum number labels given by `Qlabels` and arrow convention `arrows`; can be conjugated (`conjugated`)
  """
  function Qtens(Qlabels::Array{Array{Q,1},1}, flux::Q, arrows::U...;datatype::DataType=Float64,currblock::Array{Array{intType,1},1}=equalblocks(Qlabels))::qarray where {Q <: Qnum, U <: Union{Bool,Array{Bool,1}}}
    if length(arrows) > 0
      QnumMat = Array{Q,1}[arrows[1][a] ? Qlabels[a] : inv.(Qlabels[a]) for a = 1:length(arrows[1])]
    else
      QnumMat = Qlabels
    end

    T = tens{datatype}[]
    ind = Array{Array{intType,1},1}[]
    return Qtens(T,ind,currblock,QnumMat, flux)
  end

  function Qtens(Qlabels::Array{Array{Q,1},1}, arrows::U...;datatype::DataType=Float64,currblock::Array{Array{intType,1},1}=equalblocks(Qlabels))::qarray where {Q <: Qnum, U <: Union{Bool,Array{Bool,1}}}
    flux = Q()
    return Qtens(Qlabels, flux, arrows...,currblock = currblock,datatype=datatype)
  end

  """
      Qtens(T,ind,currblock,QnumMat[,arrows])

  constructor for Qtensor with non-zero values `T`, indices `ind`, quantum numbers `QnumMat`; optionally assign `arrows` for indices or conjugate the tensor (`conjugated`)

  If `currblock` is not known, provide a set of tensors `T` of size 1x(dimension of the tensor) and set currblock = [[1],[2,3,4,5...]]
  """
  function Qtens(T::Array{tens{Z},1}, ind::Array{Array{Array{intType,1},1},1}, currblock::Array{Array{intType,1},1}, 
                  QnumMat::Array{Array{Q,1},1},arrows::U...;conjugated::Bool=false)::qarray where {Z <: Number,Q <: Qnum, U <: Union{Bool,Array{Bool,1}}}
    if length(ind) > 0
      if length(arrows) == 0
        thisQN = [QnumMat[a][ind[1][a]] for a = 1:length(QnumMat)]
      else
        thisQN = [arrows[1][a] ? QnumMat[a][ind[1][a]] : inv(QnumMat[a][ind[1][a]]) for a = 1:length(QnumMat)]
      end
      flux = sum(thisQN)
    else
      flux = Q()
    end
    newsize = [[i] for i = 1:length(QnumMat)] #ntuple(a->length(QnumMat[a]),length(QnumMat))
    Qblocksum = Array{Q,1}[]
    finalQnumMat, QnumSum = convertQnumMat(QnumMat)
    return Qtens{Z,Q}(newsize, T, ind, currblock, Qblocksum, finalQnumMat, QnumSum, flux)
  end

  """
      Qtens(T,ind,currblock,QnumMat,flux[,arrows])

  constructor for Qtensor with non-zero values `T`, indices `ind`, total flux `flux`, quantum numbers `QnumMat`; optionally assign `arrows` for indices or conjugate the tensor (`conjugated`)

  If `currblock` is not known, provide a set of tensors `T` of size 1x(dimension of the tensor) and set currblock = [[1],[2,3,4,5...]]
  """
  function Qtens(T::Array{tens{Z},1}, ind::Array{Array{Array{intType,1},1},1}, currblock::Array{Array{intType,1},1}, QnumMat::Array{Array{Q,1},1},
                  flux::Q,arrows::U...)::qarray where {Z <: Number,Q <: Qnum, U <: Union{Bool,Array{Bool,1}}}
    if length(arrows) == 0
      newQnumMat = QnumMat
    else
      newQnumMat = [arrows[1][a] ? QnumMat[a] : inv.(QnumMat[a]) for a = 1:length(QnumMat)]
    end
    newsize = [[i] for i = 1:length(newQnumMat)]
    Qblocksum = Array{Q,1}[]

    finalQnumMat,QnumSum = convertQnumMat(newQnumMat)
    return Qtens{Z,Q}(newsize, T, ind, currblock, Qblocksum, finalQnumMat, unique.(QnumSum), flux)
  end

  function convertQnumMat(QnumMat::Array{Array{Q,1},1}) where Q <: Qnum
    QnumSum = unique.(QnumMat)
    return convertQnumMat(QnumMat,QnumSum),QnumSum
  end

  function convertQnumMat(QnumMat::Array{Array{Q,1},1},QnumSum::Array{Array{Q,1},1}) where Q <: Qnum
    finalQnumMat = [Array{intType,1}(undef,length(QnumMat[i])) for i = 1:length(QnumMat)]
    for i = 1:length(QnumMat)
      for w = 1:length(QnumMat[i])
        y = 0
        notmatchingQN = true
        while notmatchingQN
          y += 1
          notmatchingQN = QnumMat[i][w] != QnumSum[i][y]
        end
        finalQnumMat[i][w] = y
      end
    end
    return finalQnumMat
  end
  export convertQnumMat

  """
    equalblocks(A)

  Proposes a block structure that makes the matrix equivalent approximate equal in both number of rows and columns
  """
  function equalblocks(A::Array{Array{Q,1},1}) where Q <: Qnum
    sizes = ntuple(q->length(A[q]),length(A))
    return equalblocks(sizes)
  end

  function equalblocks(sizes::Tuple) where Q <: Qnum
    row = sizes[1]
    column = prod(sizes) ÷ row
    i = 1
    while row < column && i < length(sizes) - 1
      i += 1
      row *= sizes[i]
      column ÷= sizes[i]
    end
    return [[w for w = 1:i],[w for w = i+1:length(sizes)]]
  end

  function equalblocks(A::qarray)
    Rsize = A.size
    sizes = ntuple(q->prod(w->length(A.QnumMat[w]),Rsize[q]),length(Rsize))
    return equalblocks(sizes)
  end

  function recoverQNs(q::Integer,QnumMat::Array{Array{intType,1},1},QnumSum::Array{Array{Q,1},1}) where Q <: Qnum
    finalQnumMat = Array{Q,1}(undef,length(QnumMat[q]))
    for i = 1:length(QnumMat[q])
      finalQnumMat[i] = getQnum(q,i,QnumMat,QnumSum)
    end
    return finalQnumMat
  end

  function recoverQNs(q::Integer,Qt::Qtens{W,Q}) where {W <: Number, Q <: Qnum}
    return recoverQNs(q,Qt.QnumMat,Qt.QnumSum)
  end
  export recoverQNs

  function fullQnumMat(QnumMat::Array{Array{intType,1},1},QnumSum::Array{Array{Q,1},1}) where Q <: Qnum
    finalQnumMat = [recoverQNs(q,QnumMat,QnumSum) for q = 1:length(QnumMat)]
    return finalQnumMat
  end

  function fullQnumMat(Qt::Qtens{W,Q}) where {W <: Number, Q <: Qnum}
    return fullQnumMat(Qt.QnumMat,Qt.QnumSum)
  end
  export fullQnumMat

  """
      Qtens(operator,QnumMat[,Arrows,zero=])

  Creates a dense `operator` as a Qtensor with quantum numbers `QnumMat` on each index (`Array{Array{Q,1},1}`) according to arrow convention `Arrows` and all elements equal to `zero` are not included (default 0)
  """
  function Qtens(optens::denstens,Qtensor::qarray;zero::Number=0.,currblock::Array{Array{intType,1},1}=Qtensor.currblock)
    Op = makeArray(optens)
    finalQnumMat = fullQnumMat(Qtensor.QnumMat,Qtensor.QnumSum)
    return Qtens(Op,finalQnumMat,zero=zero,currblock=currblock)
  end

  function Qtens(optens::AbstractArray,Qtensor::qarray;zero::Number=0.,currblock::Array{Array{intType,1},1}=Qtensor.currblock)
    finalQnumMat = fullQnumMat(Qtensor.QnumMat,Qtensor.QnumSum)
    return Qtens(optens,finalQnumMat,zero=zero,currblock=currblock)
  end

  """
      Qtens(operator,QnumMat[,Arrows,zero=])

  Creates a dense `operator` as a Qtensor with quantum numbers `QnumMat` on each index (`Array{Array{Q,1},1}`) according to arrow convention `Arrows` and all elements equal to `zero` are not included (default 0)
  """
  function Qtens(Op::R,Qlabels::Array{Array{Q,1},1},Arrows::U...;zero::Number=0.,currblock::Array{Array{intType,1},1}=equalblocks(Qlabels),leftflux::Bool=false,datatype::DataType=eltype(Op)) where {Q <: Qnum, W <: Number, R <: Union{denstens,AbstractArray}, U <: Union{Bool,Array{Bool,1}}}
    theseArrows = typeof(Arrows) <: Bool ? Arrows : (Arrows[1]...,)
    newQnumMat = [theseArrows[q] ? Qlabels[q] : inv.(Qlabels[q]) for q = 1:length(Qlabels)]
    return Qtens(Op,newQnumMat;zero=zero,currblock=currblock,leftflux=leftflux,datatype=datatype)
  end


  function Qtens(Op::R,Qlabels::Array{Array{Q,1},1},Arrows::U...;zero::Number=0.,currblock::Array{Array{intType,1},1}=equalblocks(Qlabels),leftflux::Bool=false,datatype::DataType=eltype(Op)) where {Q <: Qnum, W <: Number, R <: Union{denstens,AbstractArray}, U <: Union{Bool,Array{Bool,1}}}
    theseArrows = typeof(Arrows) <: Bool ? Arrows : (Arrows[1]...,)
    newQnumMat = [theseArrows[q] ? Qlabels[q] : inv.(Qlabels[q]) for q = 1:length(Qlabels)]
    return Qtens(Op,newQnumMat;zero=zero,currblock=currblock,leftflux=leftflux,datatype=datatype)
  end

  function Qtens(Qlabels::Array{Array{Q,1},1},Op::R...;zero::Number=0.,currblock::Array{Array{intType,1},1}=equalblocks(Qlabels),leftflux::Bool=false,datatype::DataType=eltype(Op)) where {Q <: Qnum, W <: Number, R <: Union{denstens,AbstractArray}, U <: Union{Bool,Array{Bool,1}}}
    return ntuple(w->Qtens(Op[w],Qlabels,zero=zero,currblock=currblock,leftflux=leftflux,datatype=datatype),length(Op))
  end

  function Qtens(Op::R,Qlabels::Array{Array{Q,1},1};zero::Number=0.,currblock::Array{Array{intType,1},1}=equalblocks(Qlabels),leftflux::Bool=false,datatype::DataType=eltype(Op)) where {Q <: Qnum, W <: Number, R <: Union{denstens,AbstractArray}}
    pLinds = currblock[1]
    pRinds = currblock[2]

#    println(pLinds)
#    println(pRinds)

    sizes = size(Op)

    Op_mat = reshape(makeArray(Op),[pLinds,pRinds])
  
    pos = makepos(length(sizes))
    currval = 0
    saveval = 0
    savepos = Int64[]
    for x = 1:length(Op_mat)
      position_incrementer!(pos,sizes)
      if abs(Op_mat[x]) > currval
        currval = abs(Op_mat[x])
        saveval = x
        savepos = copy(pos)
      end
    end
    x = saveval
    pos = savepos
    invflux = sum(w->Qlabels[w][pos[w]],1:length(pos))

    outtype = eltype(Op_mat)
    Qt = Qtens(Qlabels,datatype=outtype)


    Qt.flux = copy(invflux)
    Qt.currblock = currblock
  
    Lsizes = sizes[pLinds]
    Rsizes = sizes[pRinds]
  
    Lsize = prod(Lsizes)
    Rsize = prod(Rsizes)
    LR = Lsize < Rsize
    if LR
      QNsummary = multi_indexsummary(Qt,pLinds)
      leftSummary = QNsummary
      rightSummary = inv.(QNsummary)
      
      Qt.Qblocksum = [[copy(leftSummary[q]),copy(rightSummary[q])] for q = 1:length(QNsummary)]
    else
      QNsummary = multi_indexsummary(Qt,pRinds)
      leftSummary = inv.(QNsummary)
      rightSummary = QNsummary

      Qt.Qblocksum = [[copy(leftSummary[q]),copy(rightSummary[q])] for q = 1:length(QNsummary)]
    end
   
    leftQNs,Lbigtosub,rows,Lindexes = QnumList(Qt,pLinds,leftSummary)
    rightQNs,Rbigtosub,columns,Rindexes = QnumList(Qt,pRinds,rightSummary)


    newblocks = [Array{outtype,2}(undef,rows[g],columns[g]) for g = 1:length(QNsummary)]

    permindexes = vcat(pLinds,pRinds)
  
    for y = 1:size(Op_mat,2)
      for x = 1:size(Op_mat,1)
        newx,newy = x,y
        if leftQNs[newx] != 0
        end
        if rightQNs[newy] != 0
        end
        if leftQNs[newx] != 0 && rightQNs[newy] != 0 && leftSummary[leftQNs[newx]] + rightSummary[rightQNs[newy]] == Qt.flux
          thisT = newblocks[leftQNs[newx]]
          thisT[Lbigtosub[newx],Rbigtosub[newy]] = Op_mat[x,y]
        end
      end
    end


    commonblocks = matchblocks((false,false),Qt,Qt,ind=(1,2),matchQN=Qt.flux)

    keepers = Array{Bool,1}(undef,length(newblocks))
    for q = 1:length(commonblocks)
      keepq = commonblocks[q][1]
      keepers[keepq] = true && (size(newblocks,1) > 0 && size(newblocks,2) > 0)
    end

    newind = Array{Array{Array{intType,2},1},1}(undef,length(commonblocks))
    newQblocks = Array{Array{Q,1},1}(undef,length(commonblocks))
    for q = 1:length(commonblocks)
      Aqind = commonblocks[q][1]
      Bqind = commonblocks[q][2]

      if size(Lindexes[Aqind],2) > 0 && size(Rindexes[Bqind],2) > 0
        newind[q] = [Lindexes[Aqind],Rindexes[Bqind]]
        #make this the standard and comment out the previous determination of Qblocksum
        if Qt.flux != Q()
          newQblocks[q] = [Q(),Q()]
          for i = 1:2
            offset = i == 1 ? 0 : size(newind[q][i],1)

            for a = 1:size(newind[q][i],1)
              index = newind[q][i][a,1] + 1
              add!(newQblocks[q][i],getQnum(offset + a,index,Qt))
            end
          end
        end
      else
        keepers[q] = false
      end
    end
    if Qt.flux != Q()
      Qt.Qblocksum = newQblocks[keepers]
    else
      Qt.Qblocksum = Qt.Qblocksum[keepers]
    end

    Qt.ind = newind[keepers] #[[Lindexes[w],Rindexes[w]] for w = 1:length(newblocks)]

    temp = newblocks[keepers]
    Qt.T = [tens(temp[g]) for g = 1:length(temp)]
    return Qt
  end

  """
      Qtens(newsize,T,ind,currblock,QnumMat,flux[,arrows])

  constructor for Qtensor with unreshaped size of `newsize`, non-zero values `T`, indices `ind`, total flux `flux`, quantum numbers `QnumMat`; optionally assign `arrows` for indices or conjugate the tensor (`conjugated`)
  """
  function Qtens(newsize::Tuple,T::Array{tens{Z},1}, ind::Array{P,1}, currblock::Array{Array{P,1},1}, QnumMat::Array{Array{Q,1},1},
                 flux::Q,arrows::U...)::qarray where {P <: Integer, Z <: Number,Q <: Qnum, U <: Union{Bool,Array{Bool,1}}}
    if length(arrows) == 0
      newQnumMat = QnumMat
    else
      newQnumMat = [arrows[1][a] ? QnumMat[a] : inv.(QnumMat[a]) for a = 1:length(QnumMat)]
    end
    newQblocksum = Array{Q,1}[]
    finalQnumMat,QnumSum = convertQnumMat(newQnumMat)
    return Qtens{Z,Q}(newsize, T, ind, currblock, newQblocksum, finalQnumMat,QnumSum, flux)
  end

  """
      Qtens(operator,Qlabels[,Arrows,zero=])

  Creates a dense `operator` as a Qtensor with quantum numbers `Qlabels` common to all indices (`Array{Q,1}`) according to arrow convention `Arrows` and all elements equal to `zero` are not included (default 0)
  """
  function Qtens(operator::AbstractArray,Qlabels::Array{Q,1},Arrows::U...;zero::Number=0.) where {Q <: Qnum, U <: Union{Bool,Array{Bool,1}}}
    return Qtens(operator,[Qlabels for a = 1:ndims(operator)],Arrows...,zero=zero)
  end

  """
     Qtens(A)

  `A` is a Qtensor; makes shell of a Qtensor with only meta-data (no blocks, no row reductions); used mainly for copies
  """
  function Qtens(A::Qtens{W,Q}) where {W <: Number, Q <: Qnum}
    return Qtens{W,Q}(A.size, Array{tens{W},1}[],
                      Array{Array{Array{intType,1},1},1}[], Array{intType,1}[], Array{Q,1}[], A.QnumMat, A.QnumSum, A.flux)
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
    newQlabels = Array{Q,1}[arrows[a] ? Qlabels[a] : inv.(Qlabels[a]) for a = 1:length(arrows)]
    return rand(newQlabels)
  end


  function basesize(Qtensor::qarray)
    return basesize(Qtensor.QnumMat)
  end

  function basesize(Qlabels::Array{Array{Q,1},1}) where Q <: Union{Qnum,Integer}
    return ntuple(i->length(Qlabels[i]),length(Qlabels))
  end
  export basesize

  function basedims(Qtensor::qarray)
    return length(Qtensor.QnumMat)
  end



  function rand(Qlabels::Array{Array{Q,1},1};currblock::Array{Array{P,1},1}=equalblocks(Qlabels),datatype::DataType=Float64,leftflux::Bool=true,blockfct::Function=rand)::qarray where {Q <: Qnum, P <: Integer}
    Linds = currblock[1]
    Rinds = currblock[2]

    Qtensor = Qtens(Qlabels)

    truesize = basesize(Qlabels)
    #some repeated code...could combine into one function
  
    Lsizes = truesize[Linds]
    Rsizes = truesize[Rinds]
  
    Lsize = prod(Lsizes)
    Rsize = prod(Rsizes)
    LR = Lsize < Rsize
    if LR
      QNsummary = multi_indexsummary(Qtensor,Linds)
      leftSummary = QNsummary
      rightSummary = inv.(QNsummary)
    else
      QNsummary = multi_indexsummary(Qtensor,Rinds)
      leftSummary = inv.(QNsummary)
      rightSummary = QNsummary
    end

    Qtensor.Qblocksum = [[copy(leftSummary[q]),copy(rightSummary[q])] for q = 1:length(QNsummary)]
    
    leftQNs,Lbigtosub,rows,Lindexes,Rbigtosub,columns,Rindexes = QnumList(Qtensor,LR,Linds,Rinds,leftSummary,rightSummary)

    Qtensor.ind = Array{Array{Array{intType,2},1},1}(undef,length(QNsummary))
    Qtensor.T = Array{tens{datatype},1}(undef,length(QNsummary))
    for q = 1:length(QNsummary)
      Qtensor.ind[q] = [Lindexes[q],Rindexes[q]]

      newblock = blockfct(datatype,rows[q],columns[q])
      Qtensor.T[q] = tens(newblock)
    end
    Qtensor.currblock = currblock
  
    return Qtensor
  end

  function rand(currQtens::qarray)::qarray
    return rand(currQtens.QnumMat)
  end

  function rand(A::AbstractArray)
    return rand(eltype(A),size(A)...)
  end

  function rand(A::tens{W}) where W <: Number
    return tens{W}(rand(W,size(A)...))
  end



  import Base.zeros
  function zeros(Qlabels::Array{Array{Q,1},1};currblock::Array{Array{P,1},1}=equalblocks(Qlabels),datatype::DataType=Float64,leftflux::Bool=true,blockfct::Function=rand)::qarray where {Q <: Qnum, P <: Integer}
    return rand(Qlabels,currblock=currblock,datatype=datatype,leftflux=leftflux,blockfct=zeros)
  end

  function zeros(datatype::DataType,Qlabels::Array{Array{Q,1},1};currblock::Array{Array{P,1},1}=equalblocks(Qlabels),leftflux::Bool=true,blockfct::Function=rand)::qarray where {Q <: Qnum, P <: Integer}
    return rand(Qlabels,currblock=currblock,datatype=datatype,leftflux=leftflux,blockfct=zeros)
  end

  function zeros(currQtens::Qtens{W,Q}) where {W <: Number, Q <: Qnum}
    return rand(currQtens.QnumMat,blockfct=zeros,datatype=W)
  end

  function zeros(A::AbstractArray)
    return zeros(eltype(A),size(A)...)
  end

  function zeros(A::tens{W}) where W <: Number
    return tens{W}(zeros(W,size(A)...))
  end

  import Base.zero
    """Like the default function zero(t::Array), return an object with the same properties containing only zeros."""
  function zero(Qt::qarray)
    return zeros(Qt)
  end

  import .tensor.convertTens
  """
      convertTens(T,Qt)

  Convert Qtensor `Qt` to type `T`
  """
  function convertTens(T::DataType, Qt::Qtens{Z,Q})::qarray where {Z <: Number, Q <: Qnum}
    newsize = [copy(Qt.size[i]) for i = 1:length(Qt.size)]
    newQblocksum = [copy(Qt.Qblocksum[i]) for i = 1:length(Qt.Qblocksum)]
    return Qtens{T,Q}(newsize,tens{T}.(Qt.T),copy(Qt.ind),copy(Qt.currblock),newQblocksum,
                      copy(Qt.QnumMat),copy(Qt.QnumSum),copy(Qt.flux))
   end
  export convertTens

  import Base.copy!
  """
      copy!(Qt)

  Copies a Qtensor; `deepcopy` is inherently not type stable, so this function should be used instead; `copy!` refers to all fields except `Qsize` as pointers
  """
  function copy!(Qt::Qtens{T,Q}) where {T <: Number, Q <: Qnum}
    return Qtens{T,Q}(Qt.size,Qt.T,Qt.ind,Qt.currblock,Qt.Qblocksum,Qt.QnumMat,Qt.QnumSum,Qt.flux)
  end

  import Base.copy
  """
      copy(Qt)

  Copies a Qtensor; `deepcopy` is inherently not type stable, so this function should be used instead
  """
  function copy(Qt::Qtens{W,Q}) where {W <: Number, Q <: Qnum}
    newsize = [copy(Qt.size[i]) for i = 1:length(Qt.size)]
    copyQtT = [copy(Qt.T[q]) for q = 1:length(Qt.T)]
    copyQtind = [[copy(Qt.ind[q][1]),copy(Qt.ind[q][2])] for q = 1:length(Qt.ind)]
    newcurrblock = [copy(Qt.currblock[1]),copy(Qt.currblock[2])]
    newQblocksum = [copy(Qt.Qblocksum[i]) for i = 1:length(Qt.Qblocksum)]
    newQnumSum = [copy(Qt.QnumSum[i]) for i = 1:length(Qt.QnumSum)]
    return Qtens{W,Q}(newsize,copyQtT,copyQtind,newcurrblock,
                      newQblocksum,copy(Qt.QnumMat),newQnumSum,copy(Qt.flux))
  end

####################################################
####################################################

function QnumList(Qt::Qtens{W,Q},LR::Bool,pLinds::Array{P,1},pRinds::Array{P,1},
                  leftSummary::Array{Q,1},rightSummary::Array{Q,1}) where {P <: Integer, W <: Number, Q <: Qnum}
  if LR
    leftQNs,Lbigtosub,rows,Lindexes = QnumList(Qt,pLinds,leftSummary)
    Rbigtosub,columns,Rindexes = smallQnumList(Qt,pRinds,rightSummary)
  else
    Lbigtosub,rows,Lindexes = smallQnumList(Qt,pLinds,leftSummary)
    leftQNs,Rbigtosub,columns,Rindexes = QnumList(Qt,pRinds,rightSummary)
  end
  return leftQNs,Lbigtosub,rows,Lindexes,Rbigtosub,columns,Rindexes
end

function QnumList(Qtens::Qtens{W,Q},vec::Array{Int64,1},QnumSummary::Array{Q,1}) where {W <: Number, Q <: Qnum}

  ninds = length(vec)
  truesize = basesize(Qtens)
  sizes = ntuple(a->truesize[vec[a]],ninds)
  QnumMat = Qtens.QnumMat
  QnumSum = Qtens.QnumSum

  return QnumList_worker(sizes,QnumMat,QnumSum,vec,QnumSummary)
end

function QnumList_worker(sizes::NTuple{G,P},QnumMat::Array{Array{P,1},1},QnumSum::Array{Array{Q,1},1},
                         vec::Array{P,1},QnumSummary::Array{Q,1}) where {Q <: Qnum, G, P <: Integer}
  ninds = length(vec)
  #=
  if ninds == 0
    matchingQNs = [1]
    returnvector = [1]
    QNblocksizes = [1]
    saveindexes = reshape([1],1,1)
  else
    =#
    pos = makepos(ninds)

    numElements = prod(sizes)
    matchingQNs = Array{intType,1}(undef,numElements)

    numQNs = length(QnumSummary)

    keepbools = [[false for i = 1:numElements] for q = 1:numQNs]
    longsaveindexes = Array{intType,2}(undef,length(vec),numElements)

    returnvector = Array{intType,1}(undef,numElements)
    QNblocksizes = zeros(intType,numQNs)

    startQN = Q()
    currQN = Q()
    for y = 1:numElements
      position_incrementer!(pos,sizes)
      copy!(currQN,startQN)
      for b = 1:ninds
        @inbounds add!(currQN,getQnum(vec[b],pos[b],QnumMat,QnumSum))
      end
      notmatchQNs = true
      q = 0
      while (q < numQNs) && notmatchQNs
        q += 1
        @inbounds notmatchQNs = currQN != QnumSummary[q]
      end

      if notmatchQNs
        @inbounds matchingQNs[y] = 0
        @inbounds returnvector[y] = 0
      else

        @inbounds matchingQNs[y] = q

        @inbounds QNblocksizes[q] += 1
        @inbounds returnvector[y] = QNblocksizes[q]

        keepbools[q][y] = true
        @simd for r = 1:length(pos)
          @inbounds longsaveindexes[r,y] = pos[r] - 1
        end

      end
    end

    saveindexes = Array{Array{intType,2},1}(undef,numQNs)
    for q = 1:numQNs
      @inbounds saveindexes[q] = longsaveindexes[:,keepbools[q]]
    end
#  end

  return matchingQNs,returnvector,QNblocksizes,saveindexes
end

function smallQnumList(Qtens::Qtens{W,Q},vec::Array{Int64,1},QnumSummary::Array{Q,1}) where {W <: Number, Q <: Qnum}
  ninds = length(vec)
  truesize = basesize(Qtens)
  sizes = ntuple(a->truesize[vec[a]],ninds)
  QnumMat = Qtens.QnumMat
  QnumSum = Qtens.QnumSum

  return smallQnumList_worker(sizes,QnumMat,QnumSum,vec,QnumSummary)
end

function smallQnumList_worker(sizes::NTuple{G,P},QnumMat::Array{Array{P,1},1},QnumSum::Array{Array{Q,1},1},
                              vec::Array{P,1},QnumSummary::Array{Q,1}) where {Q <: Qnum, G, P <: Integer}
  ninds = length(vec)

  pos = makepos(ninds)

  numElements = prod(sizes)
  matchingQNs = Array{intType,1}(undef,numElements)

  numQNs = length(QnumSummary)

  keepbools = [[false for i = 1:numElements] for q = 1:numQNs]
  longsaveindexes = Array{intType,2}(undef,length(vec),numElements)

  returnvector = Array{intType,1}(undef,numElements)
  QNblocksizes = zeros(intType,numQNs)


  startQN = Q()
  currQN = Q()
  for y = 1:numElements
    position_incrementer!(pos,sizes)
    copy!(currQN,startQN)
    for b = 1:ninds
      @inbounds add!(currQN,getQnum(vec[b],pos[b],QnumMat,QnumSum))
    end
    notmatchQNs = true
    q = 0
    while (q < numQNs) && notmatchQNs
      q += 1
      @inbounds notmatchQNs = currQN != QnumSummary[q]
    end

    if notmatchQNs
      @inbounds returnvector[y] = 0
    else
      @inbounds QNblocksizes[q] += 1
      @inbounds returnvector[y] = QNblocksizes[q]

      @inbounds keepbools[q][y] = true
      @simd for r = 1:length(pos)
        @inbounds longsaveindexes[r,y] = pos[r] - 1
      end
    end
  end


  saveindexes = Array{Array{intType,2},1}(undef,numQNs)
  for q = 1:numQNs
    @inbounds saveindexes[q] = longsaveindexes[:,keepbools[q]]
  end

  return returnvector,QNblocksizes,saveindexes
end

  function multi_indexsummary(Qt::Qtens{W,Q},vec::Array{P,1}) where {W <: Number, Q <: Qnum, N, P <: Integer}
    QnumSum = Qt.QnumSum
    if length(vec) == 1
      Qsumvec = QnumSum[vec[1]]
      out = Qsumvec
      #=
    elseif length(vec) == 0
      Qsumvec = [Q()]
      out = Qsumvec
      =#
    else
      ninds = length(vec)
      QsumSizes = [length(QnumSum[vec[a]]) for a = 1:ninds]
      Qsumel = prod(QsumSizes)
      Qsumvec = Array{Q,1}(undef,Qsumel)

      counter = 0

      pos = makepos(ninds)

      zeroQN = Q()
      currQN = Q()
      for g = 1:Qsumel
        position_incrementer!(pos,QsumSizes)
        copy!(currQN,zeroQN)

        for b = 1:ninds
          @inbounds add!(currQN,QnumSum[vec[b]][pos[b]])
        end
        addQ = true
        w = 0
        while w < counter && addQ
          w += 1
          addQ = currQN != Qsumvec[w]
        end
        if addQ
          counter += 1
          Qsumvec[counter] = copy(currQN)
        end
      end
      out = Qsumvec[1:counter]
    end
    return out
  end
  export multi_indexsummary

  function changeblock(Qt::Qtens{W,Q},Linds::Array{P,1},
                        Rinds::Array{P,1};leftflux::Bool=true) where {W <: Number, Q <: Qnum, P <: Integer}
    return changeblock(Qt,[Linds,Rinds],leftflux=leftflux)
  end

  function findsizes(Qt::Qtens{W,Q},Linds::Array{P,1}) where {W <: Number, Q <: Qnum, P <: Integer}
    nind = length(Linds)
    Lnonzero = nind > 0
    Lsize = Lnonzero ? prod(Linds) : 1
    return Lsize,nind,Lnonzero
  end
  
  function changeblock(Qt::Qtens{W,Q},newblock::Array{Array{P,1},1};
                       leftflux::Bool=true) where {W <: Number, Q <: Qnum, P <: Integer}
    if Qt.currblock == newblock
      newQt = Qt
    else

      xsorted = issorted(Qt.currblock[1])
      if !xsorted
        xorder = sortperm(Qt.currblock[1])
        sort!(Qt.currblock[1])
        for q = 1:length(Qt.ind)
          Qt.ind[q][1] = Qt.ind[q][1][xorder,:]
        end
      end

      ysorted = issorted(Qt.currblock[2])
      if !ysorted
        yorder = sortperm(Qt.currblock[2])
        sort!(Qt.currblock[2])
        for q = 1:length(Qt.ind)
          Qt.ind[q][2] = Qt.ind[q][2][yorder,:]
        end
      end


      Linds = newblock[1]
      Lsize,Lnind,Lnonzero = findsizes(Qt,Linds)
      Rinds = newblock[2]
      Rsize,Rnind,Rnonzero = findsizes(Qt,Rinds)

      ninds = Lnind + Rnind

      if Lnonzero && Rnonzero

        LR = Lsize < Rsize
        if LR
          QNsummary = multi_indexsummary(Qt,Linds)
          leftSummary = QNsummary
          rightSummary = inv.(QNsummary)
          for q = 1:length(rightSummary)
            add!(rightSummary[q],Qt.flux)
          end
        else
          QNsummary = multi_indexsummary(Qt,Rinds)
          leftSummary = inv.(QNsummary)
          rightSummary = QNsummary
          for q = 1:length(leftSummary)
            add!(leftSummary[q],Qt.flux)
          end
        end
        newQblocksum = [[copy(leftSummary[q]),copy(rightSummary[q])] for q = 1:length(QNsummary)]

        leftQNs,Lbigtosub,rows,Lindexes,Rbigtosub,columns,Rindexes = QnumList(Qt,LR,Linds,Rinds,leftSummary,rightSummary)
      else

        QNsummary = [Q()]
        if Rnonzero
          leftSummary = QNsummary
          rightSummary = inv.(QNsummary)
          thisflux = leftflux ? inv(Qt.flux) : Qt.flux
          for q = 1:length(rightSummary)
            add!(rightSummary[q],Qt.flux)
          end
        else
          leftSummary = inv.(QNsummary)
          rightSummary = QNsummary
          for q = 1:length(leftSummary)
            add!(leftSummary[q],Qt.flux)
          end
        end
        newQblocksum = [[copy(leftSummary[q]),copy(rightSummary[q])] for q = 1:length(QNsummary)]

        LR = Rnonzero
        if Lnonzero
          Lbigtosub,rows,Lindexes = smallQnumList(Qt,Linds,leftSummary)
          leftQNs = [1]
          Rbigtosub,columns,Rindexes = [1],[1],[reshape([1],1,1)]
        else
          Lbigtosub,rows,Lindexes = [1],[1],[reshape([1],1,1)]
          leftQNs = [1]
          Rbigtosub,columns,Rindexes = smallQnumList(Qt,Rinds,rightSummary)
        end
      end

      newQt = reblock(Qt,newblock,newQblocksum,LR,QNsummary,leftQNs,ninds,Linds,Rinds,Lindexes,Rindexes,rows,columns,Lbigtosub,Rbigtosub)
    end
    return newQt
  end
  export changeblock



  @inline function AAzeropos2ind(currpos::Array{X,1},S::NTuple{P,X},selector::NTuple{G,X}) where X <: Integer where G where P
    if G > 0
      x = 0
      @simd for i = G:-1:2
        @inbounds x += currpos[selector[i]]
        @inbounds x *= S[selector[i-1]]
      end
      @inbounds x += 1 + currpos[selector[1]]
    else
      x = 1
    end
    return x
  end

  @inline function innerloadpos!(y::P,Rorigsize::P,thispos::Array{P,1},thiscurrblock_two::Array{P,1},thisind_two::Array{P,2}) where P <: Integer
    @simd for i = 1:Rorigsize
      @inbounds thispos[thiscurrblock_two[i]] = thisind_two[i,y]
    end
    nothing
  end

  @inline function innerloop(x::S,y::S,newblocks::Array{Array{W,2},1},inputval::W,leftQNs::Array{S,1},LRpos::Bool,basesize::NTuple{P,S},thispos::Array{S,1},#thisthread::S,
            thisind_one::Array{S,2},thisind_one_sizetwo::S,Linds::NTuple{Z,S},Lbigtosub::Array{S,1},
            Rinds::NTuple{G,S},Rbigtosub::Array{S,1}) where {P, Z, G, W <: Number, S <: Integer}

    smallx = AAzeropos2ind(thispos,basesize,Linds)
    smally = AAzeropos2ind(thispos,basesize,Rinds)
      
    @inbounds newq = leftQNs[LRpos ? smallx : smally]
    @inbounds xval = Lbigtosub[smallx]
    @inbounds yval = Rbigtosub[smally]

    setindex!(newblocks[newq],inputval,xval,yval)
    nothing
  end

  function double_loop(newblocks::Array{Array{W,2},1},thisTens::Array{W,1},leftQNs::Array{S,1},LRpos::Bool,basesize::NTuple{P,S},posvecs::Array{Array{S,1},1},#thisthread::S,
            thisind_one::Array{S,2},thisind_one_sizetwo::S,Lorigsize::S,thiscurrblock_one::Array{S,1},Linds::NTuple{Z,S},Lbigtosub::Array{S,1},
            thisind_two::Array{S,2},thisind_two_sizetwo::S,Rorigsize::S,thiscurrblock_two::Array{S,1},Rinds::NTuple{G,S},Rbigtosub::Array{S,1};minelements::Integer=50) where {P, Z, G, W <: Number, S <: Integer}
  if thisind_two_sizetwo > minelements
    Threads.@threads for y = 1:thisind_two_sizetwo
      thisthread = Threads.threadid()
      thispos = posvecs[thisthread]
      innerloadpos!(y,Rorigsize,thispos,thiscurrblock_two,thisind_two)

      z = thisind_one_sizetwo * (y-1)

      for x = 1:thisind_one_sizetwo
        inputval = thisTens[x + z]

        innerloadpos!(x,Lorigsize,thispos,thiscurrblock_one,thisind_one)
        
        innerloop(x,y,newblocks,inputval,leftQNs,LRpos,basesize,thispos,
                thisind_one,thisind_one_sizetwo,Linds,Lbigtosub,Rinds,Rbigtosub)
      end
    end
  else    
    for y = 1:thisind_two_sizetwo
      thisthread = Threads.threadid()
      thispos = posvecs[thisthread]
      innerloadpos!(y,Rorigsize,thispos,thiscurrblock_two,thisind_two)

      z = thisind_one_sizetwo * (y-1)

      for x = 1:thisind_one_sizetwo
        inputval = thisTens[x + z]

        innerloadpos!(x,Lorigsize,thispos,thiscurrblock_one,thisind_one)
        
        innerloop(x,y,newblocks,inputval,leftQNs,LRpos,basesize,thispos,
                thisind_one,thisind_one_sizetwo,Linds,Lbigtosub,Rinds,Rbigtosub)
      end
    end
  end
    nothing
  end

  function (reblock(Qt::Qtens{W,Q},newcurrblocks::Array{Array{S,1},1},newQblocksum::Array{Array{Q,1},1},
                   LR::Bool,QNsummary::Array{Q,1},leftQNs::Array{S,1},ninds::S,
                   Linds::Array{S,1},Rinds::Array{S,1},Lindexes::Array{Array{S,2},1},
                   Rindexes::Array{Array{S,2},1},rows::Array{S,1},columns::Array{S,1},
                   Lbigtosub::Array{S,1},Rbigtosub::Array{S,1};effZero::Float64=1E-13) where {W <: Number, Q <: Qnum, G <: Union{Array{S,1},Tuple}, P <: Union{Array{S,1},Tuple}}) where S <: Integer

    fulltens = sum(q->length(Qt.T[q]),1:length(Qt.T)) == sum(q->rows[q]*columns[q],1:length(rows))
    newblocks = Array{Array{W,2},1}(undef,length(QNsummary))
    keepblocks = Array{Bool,1}(undef,length(QNsummary))
    for q = 1:length(QNsummary)
      keepblocks[q] = rows[q] > 0 && columns[q] > 0
      if keepblocks[q]
        if fulltens
          newblocks[q] = Array{W,2}(undef,rows[q],columns[q])
        else
          newblocks[q] = zeros(W,rows[q],columns[q])
        end
      end
    end

    nQNs = length(Qt.ind)

    numthreads = Threads.nthreads()
    posvecs = [ones(intType,length(Qt.QnumMat)) for q = 1:numthreads]

    Lorigsize = length(Qt.currblock[1])
    Rorigsize = length(Qt.currblock[2])

    thiscurrblock_one = Qt.currblock[1]
    thiscurrblock_two = Qt.currblock[2]

    tup_Linds = (Linds...,)
    tup_Rinds = (Rinds...,)

    basesize = ntuple(i->length(Qt.QnumMat[i]),length(Qt.QnumMat))

    for q = 1:nQNs

      thisTens = Qt.T[q].T

      thisind = Qt.ind[q]

      thisind_one_sizeone = size(Qt.ind[q][1],1)
      thisind_one_sizetwo = size(Qt.ind[q][1],2)
      thisind_one = Qt.ind[q][1]


      thisind_two_sizeone = size(Qt.ind[q][2],1)
      thisind_two_sizetwo = size(Qt.ind[q][2],2)
      thisind_two = Qt.ind[q][2]

        double_loop(newblocks,thisTens,leftQNs,LR,basesize,posvecs,
                    thisind_one,thisind_one_sizetwo,Lorigsize,thiscurrblock_one,tup_Linds,Lbigtosub,
                    thisind_two,thisind_two_sizetwo,Rorigsize,thiscurrblock_two,tup_Rinds,Rbigtosub)
    end
    outTens = Array{tens{W},1}(undef,sum(keepblocks))
    newrowcols = Array{Array{Array{intType,2},1},1}(undef,length(outTens))


    count = 0
    for q = 1:length(keepblocks)
      if keepblocks[q]
        count += 1
        thisq = count
        outTens[thisq] = tens{W}(newblocks[q])
        newrowcols[thisq] = [Lindexes[q],Rindexes[q]]
      end
    end
    
    finalQblocksum = newQblocksum[keepblocks]
    return Qtens{W,Q}(Qt.size,outTens,newrowcols,newcurrblocks,finalQblocksum,Qt.QnumMat,Qt.QnumSum,Qt.flux)
  end


####################################################
####################################################

  function newindexsizeone!(Qt::Qtens{W,Q},S::Integer...) where {W <: Number, Q <: Qnum}
    size_ones = 0
    for w = 1:length(S)
      size_ones += S[w] == 1
    end
    base_ones = 0
    for w = 1:length(Qt.QnumMat)
      base_ones += length(Qt.QnumMat[w]) == 1
    end

    newindices = size_ones - base_ones

    if  newindices > 0
      for q = 1:length(Qt.ind)
        Qt.ind[q][2] = vcat(Qt.ind[q][2],zeros(intType,newindices,size(Qt.ind[q][2],2)))
      end
      newinds = [length(Qt.QnumMat) + w for w = 1:newindices]
      Qt.currblock[2] = vcat(Qt.currblock[2],newinds)

      newindex = [[Q()] for w = 1:newindices]
      Qt.QnumSum = vcat(Qt.QnumSum,newindex)
      newQnum = [[1] for w = 1:newindices]
      Qt.QnumMat = vcat(Qt.QnumMat,newQnum)
    end
    nothing
  end
  export newindexsizeone!

  import ..tensor.reshape!
  """
      reshape!(M,a...[,merge=])

  In-place reshape for Qtensors (otherwise makes a copy); can also make Qtensor unreshapable with `merge`, joining all grouped indices together

  # Warning
  If the Qtensor size is (10,1,2) and we want to reshape in to a (10,2) tensor, this algorithm will reshape to a 
  [[1,2],[3]] be default instead of [[1],[2,3]], so beware.

  See also: [`reshape`](@ref)
  """
  function reshape!(Qt::Qtens{W,Q}, S::Integer...;merge::Bool=false)::qarray where {W <: Number, Q <: Qnum}
    Rsize = recoverShape(Qt,S...)
    outQt = reshape!(Qt,Rsize,merge=merge)
    newindexsizeone!(outQt,S...)
    return outQt
  end

  function (reshape!(Qt::Array{W,N},S::Integer...;merge::Bool=false)::Array{W,length(S)}) where W <: Number where N
    return reshape(Qt,S...)
  end

  function reshape!(Qt::Qtens{W,Q}, newQsize::Array{Array{intType,1},1};merge::Bool=false)::qarray where {W <: Number, Q <: Qnum}
    Qt.size = newQsize
    if merge
      outQt = mergereshape!(Qt)
    else
      outQt = Qt
    end
    return outQt
  end

  function reshape!(Qt::Qtens{W,Q}, newQsize::Array{P,1}...;merge::Bool=false)::qarray where {W <: Number, Q <: Qnum, P <: Integer}
    return reshape!(Qt,[newQsize[i] for i = 1:length(newQsize)],merge=merge)
  end
    
  function (reshape!(Qt::Array{W,N}, newQsize::Array{Array{P,1},1};merge::Bool=false)::Array{W,length(newQsize)}) where {W <: Number, N, P <: Integer}
    return reshape!(Qt,intType[prod(a->size(Qt, a), newQsize[i]) for i = 1:size(newQsize, 1)]...,merge=merge)
  end
  
  function (reshape!(Qt::Array{W,N}, newQsize::Array{P,1}...;merge::Bool=false)::Array{W,length(newQsize)}) where {W <: Number, N, P <: Integer}
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
  function reshape(Qt::Qtens{W,Q}, S::Integer...;merge::Bool=false) where {W <: Number, Q <: Qnum}
    return reshape!(copy(Qt),S...,merge=merge)
  end

  function reshape(Qt::Qtens{W,Q}, newQsize::Array{Array{P,1},1};merge::Bool=false) where {W <: Number, Q <: Qnum, P <: Integer}
    return reshape!(copy(Qt), newQsize...,merge=merge)
  end
  
  function reshape(Qt::Qtens{W,Q}, newQsize::Array{P,1}...;merge::Bool=false) where {W <: Number, Q <: Qnum, P <: Integer}
    return reshape!(copy(Qt), newQsize...,merge=merge)
  end

  function (reshape(Qt::Array{W,N}, newQsize::Array{Array{P,1},1};merge::Bool=false)::Array{W,length(newQsize)}) where {W <: Number, N, P <: Integer}
#    totdim = sum(w->length(newQsize[w])>0,1:length(newQsize))
    M = Array{intType,1}(undef,length(newQsize))
    counter = 0
    for g = 1:length(newQsize)
      counter += 1
      if length(newQsize[g]) > 0
        M[counter] = prod(b->size(Qt,b),newQsize[g])
      else
        M[counter] = 1
      end
    end
    return reshape(deepcopy(Qt), M...)
  end

  function getQnum(a::Integer,b::Integer,QnumMat::Array{Array{P,1},1},QnumSum::Array{Array{Q,1},1}) where {Q <: Qnum, P <: Integer}
    Qnumber = QnumMat[a][b]
    return QnumSum[a][Qnumber]
  end

  function getQnum(a::Integer,b::Integer,Qt::qarray)
    return getQnum(a,b,Qt.QnumMat,Qt.QnumSum)
  end
  export getQnum


  function makenewindsL(LR::Integer,newQt::qarray,Qt::qarray,Rsize::Array{Array{P,1},1},offset::Integer) where P <: Integer
    newindsL = [Array{intType,2}(undef,length(newQt.currblock[1]),size(newQt.ind[q][LR],2)) for q = 1:length(newQt.T)]
    for q = 1:length(newindsL)
      for x = 1:size(newindsL[q],2)
        for i = 1:size(newindsL[q],1)
          val = 0
          b = i + offset
          @simd for a = length(Rsize[b]):-1:1
            index = Rsize[b][a]
            @inbounds val *= length(Qt.QnumMat[index])
            @inbounds val += newQt.ind[q][LR][index,x]
          end
          @inbounds newindsL[q][i,x] = val
        end
      end
    end
    return newindsL
  end


  function makenewindsR(LR::Integer,newQt::qarray,Qt::qarray,Rsize::Array{Array{P,1},1},offset::Integer) where P <: Integer
    newindsR = [Array{intType,2}(undef,length(newQt.currblock[LR]),size(newQt.ind[q][LR],2)) for q = 1:length(newQt.T)]
    for q = 1:length(newindsR)
      for x = 1:size(newindsR[q],2)
        for i = 1:size(newindsR[q],1)
          val = 0
          @inbounds b = i + offset
          @simd for a = length(Rsize[b]):-1:1
            @inbounds index = Rsize[b][a]
            @inbounds val *= length(Qt.QnumMat[index])
            @inbounds val += newQt.ind[q][LR][a,x]
          end
          @inbounds newindsR[q][i,x] = val
        end
      end
    end
    return newindsR
  end

  """
      mergereshape!(M)

  Groups all joined indices together to make one index that is unreshapable.  Dense tensors are unaffected.

  See also: [`reshape!`](@ref)
  """
  function mergereshape!(Qt::Qtens{W,Q};currblock::Array{Array{P,1},1}=equalblocks(Qt)) where {W <: Number, Q <: Qnum, P <: Integer}
    Rsize = Qt.size
    newdim = length(Rsize)

    newQnumMat = Array{Array{intType,1},1}(undef,newdim)
    newQnumSum = Array{Array{Q,1},1}(undef,newdim)

    newsizes = ntuple(a->prod(w->length(Qt.QnumMat[w]),Rsize[a]),newdim)

    zeroQN = Q()
    currQN = Q()
    truesize = basesize(Qt)
    for a = 1:length(Rsize)
      if size(Rsize[a],1) > 1

        sizes = truesize[Rsize[a]]
        thisflux = a == length(Rsize) ? Qt.flux : zeroQN

        Linds = Rsize[a]
        vec = Linds
        QNsummary = multi_indexsummary(Qt,Linds)


        ninds = length(vec)
        pos = makepos(ninds)
      
        numElements = prod(sizes)
        current_QnumMat = Array{intType,1}(undef,numElements)

        startQN = thisflux
        for y = 1:numElements
          position_incrementer!(pos,sizes)
          copy!(currQN,startQN)
          add!(currQN,getQnum(vec[1],pos[1],Qt))
          for b = 2:ninds
            add!(currQN,getQnum(vec[b],pos[b],Qt))
          end
          b = 0
          findmatch = true
          while findmatch && b < length(QNsummary)
            b += 1
            findmatch = QNsummary[b] != currQN
          end
          current_QnumMat[y] = b
        end
        newQnumMat[a],newQnumSum[a] = current_QnumMat,QNsummary

      else
        thisind = Rsize[a][1]
        newQnumMat[a] = Qt.QnumMat[thisind]
        newQnumSum[a] = Qt.QnumSum[thisind]
      end
    end

    Linds = [i for i = 1:Rsize[end][1]-1]
    Rinds = Rsize[end]

    newQt = changeblock(Qt,Linds,Rinds)
    newQt.size = [[i] for i = 1:length(Rsize)]
    merged_currblock = [[i for i = 1:length(Rsize)-1],[length(Rsize)]]

    newQt.currblock = merged_currblock





    newindsL = makenewindsL(1,newQt,Qt,Rsize,0)
    newindsR = makenewindsR(2,newQt,Qt,Rsize,length(newQt.currblock[1]))

    newQt.ind = [[newindsL[q],newindsR[q]] for q = 1:length(newindsL)]

    newQt.QnumMat = newQnumMat
    newQt.QnumSum = newQnumSum

    return newQt
  end
  export mergereshape!

  function mergereshape(Qt::Qtens{W,Q};currblock::Array{Array{P,1},1}=equalblocks(Qt)) where {W <: Number, Q <: Qnum, P <: Integer}
    cQt = copy(Qt)
    return mergereshape!(cQt,currblock=currblock)
  end
  export mergereshape
#=
  function basereshape!(A::Qtens{W,Q},Qlabels::Array{Array{Q,1},1};currblock::Array{Array{intType,1},1}=equalblocks(Qlabels)) where {W <: Number, Q <: Qnum}

    A.ind = #downconvert

    A.currblock = currblock
    A.size = ntuple(w->length(Qlabels[w]),length(Qlabels))
    A.QnumMat,A.QnumSum = convertQnumMat(Qlabels)
    return A
  end
  export basereshape!

  function basereshape(A::Qtens{W,Q},Qlabels::Array{Array{Q,1},1};currblock::Array{Array{intType,1},1}=equalblocks(Qlabels)) where {W <: Number, Q <: Qnum}
    cA = copy(A)
    return basereshape!(cA,Qlabels,currblock=currblock)
  end
=#
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
    return unreshape(Qt)
  end

  import ..tensor.unreshape
  """
      unreshape(Qt,a...)

  Unambiguous unreshaping for Qtensors.  Works identically to reshape for dense tensors
  """
  function unreshape(Qt::qarray)
    Qt.size = [[i] for i = 1:length(Qt.QnumMat)]
    return Qt
  end

  function unreshape(Qt::T,sizes::Array{W,1}) where {W <: Integer, T<: Union{qarray,AbstractArray}}
    return unreshape(Qt,sizes...)
  end

  function unreshape(Qt::AbstractArray,sizes::W...) where W <: Integer
    return reshape(Qt,sizes...)
  end

  function unreshape(Qt::qarray,sizes::W...) where W <: Integer
    return unreshape!(copy!(Qt),sizes...)
  end


  #get the last index of a Qtensor (ex: A[:,1:end]...defines "end")
  import Base.lastindex
  """
      lastindex(Qtens,i)

  get the last index of a Qtensor (ex: A[:,1:end]...defines "end")
  """
  function lastindex(Qtens::qarray, i::Integer) #where Q <: qarray
    return Qtens.size[i]
  end

  import .tensor.mult!
  """
      mult!(A,x)

  Multiplies `x*A` (commutative) for dense or quantum tensors

  See also: [`*`](@ref) [`add!`](@ref) [`sub!`](@ref) [`div!`](@ref)
  """
  function mult!(A::qarray, num::Number)::qarray
    return tensorcombination!(A,alpha=num)
  end

  function mult!(Qt::AbstractArray, num::Number)
    return Qt * num
  end

  function mult!(num::Number, Qt::X)::TensType where X <: TensType
    return mult!(Qt, num)
  end

  function matchblocks(conjvar::NTuple{G,Bool},operators::qarray...;
                       ind::NTuple{G,P}=ntuple(i->i==1 ? 2 : 1,length(operators)),
                       matchQN::Q=typeof(operators[1].flux)(),nozeros::Bool=true) where {G,Q <: Qnum, P <: Integer}
    zeroQN = Q()
    A = operators[1]
    Aind = ind[1]

    LQNs = [conjvar[1] ? inv(A.Qblocksum[q][Aind]) : A.Qblocksum[q][Aind] for q = 1:length(A.Qblocksum)]
    Aorder = [Array{intType,1}(undef,length(operators)) for q = 1:length(LQNs)]
    for q = 1:length(LQNs)
      Aorder[q] = Array{intType,1}(undef,length(operators))
      Aorder[q][1] = q
    end
    matchBool = Array{Bool,1}(undef,length(LQNs))

    numthreads = Threads.nthreads()
    storeQN = [Q() for i = 1:numthreads]

    for k = 2:length(operators)
      B = operators[k]
      Bind = ind[k]
      RQNs = [conjvar[k] ? inv(B.Qblocksum[q][Bind]) : B.Qblocksum[q][Bind] for q = 1:length(B.Qblocksum)]

      #=Threads.@threads=# for q = 1:length(LQNs)
        thisthread = Threads.threadid()
        @inbounds matchBool[q] = false
        w = 0
        while w < length(RQNs) && !matchBool[q]
          @inbounds thisQN = storeQN[thisthread]
          copy!(thisQN,zeroQN)
          w += 1
          @inbounds add!(thisQN,RQNs[w])
          @inbounds add!(thisQN,LQNs[q])
          @inbounds matchBool[q] = thisQN == matchQN
        end
        @inbounds Aorder[q][k] = matchBool[q] ? w : 0
      end
    end

    if nozeros
      Aorder = Aorder[matchBool]
    end

    return Aorder
  end
  export matchblocks

  import .tensor.tensorcombination!
  function tensorcombination!(M::Qtens{W,Q}...;alpha::Tuple=ntuple(i->1,length(A)),fct::Function=*)::qarray where {Q <: Qnum, W <: Number}
    A = tensorcombination!(M[1],alpha=alpha[1])
    for i = 2:length(M)
      tensorcombination!(A,M[i],alpha=(1,alpha[i]),fct=fct)
    end
    return A
  end

  function tensorcombination!(M::Qtens{W,Q};alpha::Number=1,fct::Function=*)::qarray where {Q <: Qnum, W <: Number}
    if !isapprox(alpha[1],1.)
      #=Threads.@threads=# for q = 1:length(M.T)
        @simd for i = 1:length(M.T[q])
          M.T[q].T[i] = fct(M.T[q].T[i],alpha)
        end
      end
    end
    return M
  end



  function tensorcombination!(A::Qtens{W,Q},QtensB::Qtens{W,Q};alpha::Tuple=(1,1),fct::Function=*)::qarray where {Q <: Qnum, W <: Number}

    A = tensorcombination!(A,alpha=alpha[1],fct=fct)

    mult = alpha[2]

    B = changeblock(QtensB,A.currblock)

    commoninds = matchblocks((false,false),A,B,ind=(1,2),matchQN=A.flux)

    #=Threads.@threads=# for q = 1:length(commoninds)
      Aqind = commoninds[q][1]
      Bqind = commoninds[q][2]
      add!(A.T[Aqind],B.T[Bqind],mult)
    end

    Bcommon = [commoninds[q][2] for q = 1:length(commoninds)]
    Bleftover = setdiff(1:length(B.T),Bcommon)

    if length(Bleftover) > 0
      AQblocks = length(A.T)
      newT = Array{tens{W},1}(undef,AQblocks+length(Bleftover))
      newind = Array{Array{Array{intType,2},1},1}(undef,length(newT))
      newQblocksum = Array{Array{Q,1},1}(undef,length(newT))
      for q = 1:AQblocks
        newT[q] = A.T[q]
        newind[q] = A.ind[q]
        newQblocksum[q] = A.Qblocksum[q]
      end
      #=Threads.@threads=# for q = 1:length(Bleftover)
        addq = Bleftover[q]
        newT[q+AQblocks] = fct(B.T[addq],mult)
        newind[q+AQblocks] = B.ind[addq]
        newQblocksum[q+AQblocks] = B.Qblocksum[addq]
      end
      A.T = newT
      A.ind = newind
      A.Qblocksum = newQblocksum
    end

    return A
  end

  import .tensor.add!
  """
      add!(A,B[,x])

  Adds `A + x*B` (default x = 1) for dense or quantum tensors

  See also: [`+`](@ref) [`sub!`](@ref) [`mult!`](@ref) [`div!`](@ref)
  """
  function add!(A::Qtens{W,Q}, QtensB::qarray, mult::Number)::qarray where {Q <: Qnum, W <: Number}
    return tensorcombination!(A,QtensB,alpha=(1,mult))
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
    return mult!(A,1/num)
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
    return num * Qt
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
  function sqrt!(A::qarray)
    for q = 1:length(A.T)
      sqrt!(A.T[q])
    end
    return A
  end

  function sqrt!(Qt::AbstractArray)
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
  function invmat!(A::qarray)
    for q = 1:length(A.T)
      invmat!(A.T[q])
    end
    return A
  end
  export invmat!

  import .tensor.invmat
  """
      invmat(Qt)

  Creates inverse of a diagonal matrix by making a new copy

  See also: [`invmat!`](@ref)
  """
  function invmat(Qt::qarray)
    return invmat!(copy(Qt))
  end
  export invmat

  """
    metricdistance(A[,power=])

  computes the Forebenius norm of all elements in the tensor...equal to L^power norm
  """
  function metricdistance(D::Qtens{W,Q};power::Number=1) where {W <: Number, Q <: Qnum}
    powersums = Array{W,1}(undef,length(D.T))
    #=Threads.@threads=# for q = 1:length(D.T)
      powersums[q] = sum(w->D.T[q].T[w]^power,1:length(D.T[q].T))
    end
    return sum(powersums)^(1/power)
  end

  import Base.sum
  """
      sum(A)

  sum elements of a Qtensor
  """
  function sum(QtensA::Qtens{W,Q}) where {W <: Number, Q <: Qnum}
    return metricdistance(QtensA,power=1)
  end

  import LinearAlgebra.norm
  """
      norm(A)

  Froebenius norm of a Qtensor
  """
  function norm(QtensA::Qtens{W,Q}) where {W <: Number, Q <: Qnum}
    return metricdistance(QtensA,power=2)
  end
  export norm

  import Base.eltype
  """
      eltype(A)

  element type of a Qtensor (i.e., `T` field)

  See also: [`Qtens`](@ref)
  """
  function eltype(A::Qtens{W,Q}) where {W <: Number, Q <: Qnum}
    return W
  end

  import .tensor.elnumtype
  function elnumtype(A::Qtens{W,Q}) where {W <: Number, Q <: Qnum}
    return eltype(A)
  end

  import LinearAlgebra.conj
  """
      conj(A)

  conjugates a Qtensor by creating a copy

  See also: [`conj!`](@ref)
  """
  function conj(currQtens::qarray)::qarray
    Qtens = copy(currQtens)
    conj!(Qtens)
    return Qtens
  end

  import LinearAlgebra.conj!
  """
      conj!(A)

  conjugates a Qtensor in place

  See also: [`conj`](@ref)
  """
  function conj!(currQtens::qarray)
    for q = 1:length(currQtens.T)
      currQtens.T[q] = conj!(currQtens.T[q])
    end
    currQtens.QnumSum = [inv!.(currQtens.QnumSum[w]) for w = 1:length(currQtens.QnumSum)]
    currQtens.flux = inv(currQtens.flux)
    return currQtens
  end

  import LinearAlgebra.ndims
  """
      ndims(A)

  number of dimensions of a Qtensor (identical usage to dense `size` call)
  """
  function ndims(A::qarray)
    return length(A.size)
  end

  import LinearAlgebra.size
  import Base.size
  """
      size(A[,i])

  gets the size of a Qtensor (identical usage to dense `size` call)
  """
  function size(A::qarray, i::intType)
    return prod(w->length(A.QnumMat[w]),A.size[i])
  end

  function size(A::qarray)
    return ntuple(w->size(A,w),ndims(A))
  end

  function recoverShape(Qt::Qtens{W,Q},S::Integer...) where {W <: Number, Q <: Qnum}
    Rsize = Array{Array{intType,1},1}(undef,length(S))
    count = 1
    wstart = 1
    for a = 1:length(Rsize)
      if count > length(Qt.QnumMat)
        Rsize[a] = [count]
        count += 1
      else
        currdim = length(Qt.QnumMat[count])
        while count < length(Qt.QnumMat) && currdim < S[a]
          count += 1
          currdim *= length(Qt.QnumMat[count])
        end
        while length(Qt.QnumMat) > count && a == length(Rsize) && currdim*length(Qt.QnumMat[count+1]) == S[a] #&& length(Qt.QnumMat[count+1]) == 1
          count += 1
        end
        Rsize[a] = [i for i = wstart:count]
        count += 1
        wstart = count
      end
    end
    return Rsize
  end
  export recoverShape



  import Base.permutedims
  """
      permutedims(A,[1,3,2,...])

  permutes dimensions of `A`  (identical usage to dense `size` call)

  See also: [`permutedims!`](@ref)
  """
  function permutedims(currQtens::Qtens{W,Q}, vec::Array{P,1}) where {W <: Number, Q <: Qnum, P <: Integer}
    Qtens = copy(currQtens)
    permutedims!(Qtens, vec)
    return Qtens
  end

  import Base.permutedims!
  """
      permutedims!(A,[1,3,2,...])

  permute dimensions of a Qtensor in-place (no equivalent, but returns value so it can be used identically to `permutedims` on a dense tensor)

  See also: [`permutedims`](@ref)
  """
  function permutedims!(currQtens::Qtens{W,Q}, vec::Union{NTuple{N,P},Array{P,1}}) where {N, W <: Number, Q <: Qnum, P <: Integer}
    Rsize = currQtens.size

    totalordersize = sum(q->length(Rsize[q]),1:length(Rsize))
    order = Array{intType,1}(undef,totalordersize)
    counter = 0
    for i = 1:size(vec, 1)
      for j = 1:length(Rsize[vec[i]])
        counter += 1
        order[counter] = Rsize[vec[i]][j]
      end
    end

    permorder = Array{intType,1}(undef,length(order))

    newRsize = Rsize[[vec...]]
    counter = intType[0]
    for k = 1:length(Rsize)
      for m = 1:length(Rsize[k])
        counter[1] += 1
        permorder[order[counter[1]]] = counter[1]
      end
    end

    counter = 0
    for w = 1:length(newRsize)
      for a = 1:length(newRsize[w])
        counter += 1
        newRsize[w][a] = counter
      end
    end

    currQtens.size = newRsize #ntuple(i->currQtens.size[order[i]],length(order))
    currQtens.QnumMat = currQtens.QnumMat[[order...]]
    currQtens.QnumSum = currQtens.QnumSum[[order...]]

    for w = 1:2
      theseblocks = currQtens.currblock[w]
      currQtens.currblock[w] = [permorder[theseblocks[i]] for i = 1:length(theseblocks)]
    end

    return currQtens
  end

  function permutedims!(A::AbstractArray, order::Union{NTuple{N,P},Array{P,1}}) where {N, P <: Integer}
    return permutedims(A, order)
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




  """
      showQtens(Qt[,show=])

  Prints fields of the Qtensor (`Qt`) out to a number of elements `show`; can also be called with `print` or `println`

  See also: [`Qtens`](@ref) [`print`](@ref) [`println`](@ref)
  """
  function showQtens(Qtens::qarray;show::Integer = 4)
    println("printing Qtens of type: ", typeof(Qtens))
    println("size = ", Qtens.size)
    maxshow = min(show, size(Qtens.T, 1))
    maxBool = show < size(Qtens.T, 1)
    println("block tensor: ")#Qtens.T[1:maxshow], maxBool ? "..." : "")
    if length(Qtens.T) == 0
      println("<null tensor>")
    else
      for q = 1:length(Qtens.T)
        maxshow = min(show, length(Qtens.T[q].T))
        maxBool = show < length(Qtens.T[q].T)
        println("block $q size = ",Qtens.T[q].size,", ",Qtens.Qblocksum[q],", values = ",Qtens.T[q].T[1:maxshow], maxBool ? "..." : "")
        println("inds: block $q")
        maxshow = min(show, length(Qtens.ind[q][1]))
        maxBool = show < length(Qtens.ind[q][1])
        println("  row: ",Qtens.ind[q][1][1:maxshow], maxBool ? "..." : "")
        maxshow = min(show, length(Qtens.ind[q][2]))
        maxBool = show < length(Qtens.ind[q][2])
        println("  col: ",Qtens.ind[q][2][1:maxshow], maxBool ? "..." : "")
      end
    end
    println("currblock = ",Qtens.currblock)
    println("Qblocksum = ",Qtens.Qblocksum)
    println("QnumMat = ")
    for i = 1:size(Qtens.QnumMat, 1)
      maxshow = min(show, size(Qtens.QnumMat[i], 1))
      maxBool = show < size(Qtens.QnumMat[i], 1)
      println("  ",i, ": ", Qtens.QnumMat[i][1:maxshow], maxBool ? "..." : "")
    end
    println("QnumSum = ")
    for i = 1:size(Qtens.QnumSum, 1)
      maxshow = min(show, size(Qtens.QnumSum[i], 1))
      maxBool = show < size(Qtens.QnumSum[i], 1)
      println("  ",i, ": ", Qtens.QnumSum[i][1:maxshow], maxBool ? "..." : "")
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
  function println(A::qarray;show::Integer = 4)
    showQtens(A, show = show)
    print("\n")
    nothing
  end


end
