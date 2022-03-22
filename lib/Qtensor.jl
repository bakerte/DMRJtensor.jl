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
#=
module Qtensor
using ..tensor
using ..QN
import LinearAlgebra
=#
#  import .tensor.checkType

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

See also: [`Qnum`](@ref) [`makedens`](@ref) [`checkflux`](@ref)
"""
mutable struct Qtens{W <: Number,Q <: Qnum} <: qarray
  size::Array{Array{intType,1},1} #the size of the tensor if it were represented densely
  #^This is an array since it can change on reshape
  T::Array{Array{W,2},1}
  ind::Array{NTuple{2,Array{intType,2}},1}
  currblock::NTuple{2,Array{intType,1}}
  Qblocksum::Array{NTuple{2,Q},1}
  QnumMat::Array{Array{intType,1},1} #quantum numbers on each index
  QnumSum::Array{Array{Q,1},1} #summary of indices on each index
  flux::Q #flux = sum of other indices
end
export Qtens

const currblockTypes = Union{NTuple{2,Array{intType,1}},Array{Array{intType,1},1}}

"""
    Qtens(Qlabels[,arrows,datatype=])

Creates empty `Qtens` with array type `Type` (default Float64), quantum number labels given by `Qlabels` and arrow convention `arrows`
"""
function Qtens(Qlabels::Array{Array{Q,1},1}, arrows::U;datatype::DataType=Float64,currblock::currblockTypes=equalblocks(Qlabels),flux::Q=Q()) where {Q <: Qnum, U <: Union{Bool,Array{Bool,1}}}
  newQlabels = Array{Q,1}[arrows[a] ? Qlabels[a] : inv.(Qlabels[a]) for a = 1:length(arrows)]
  return Qtens(Qlabels,datatype=datatype,currblock=currblock,flux=flux)
end

function Qtens(Qlabels::Array{Array{Q,1},1};datatype::DataType=Float64,currblock::currblockTypes=equalblocks(Qlabels),flux::Q=Q(),blockfct::Function=undefMat) where {Q <: Qnum, U <: Union{Bool,Array{Bool,1}}}

  pLinds = currblock[1]
  pRinds = currblock[2]

  Lsizes = Array{intType,1}(undef,length(pLinds))
  @inbounds for w = 1:length(pLinds)
    Lsizes[w] = length(Qlabels[pLinds[w]])
  end

  Rsizes = Array{intType,1}(undef,length(pRinds))
  @inbounds for w = 1:length(pRinds)
    Rsizes[w] = length(Qlabels[pRinds[w]])
  end

  Lsize = prod(Lsizes)
  Rsize = prod(Rsizes)
  LR = Lsize < Rsize

  QNsummary,leftSummary,rightSummary,newQblocksum = makeQNsummaries(Qlabels,pLinds,pRinds,LR,flux)
  
  finalQnumMat, QnumSum = convertQnumMat(Qlabels)

  leftQNs,Lbigtosub,rows,Lindexes,rightQNs,Rbigtosub,columns,Rindexes = QnumList(Lsizes,Rsizes,finalQnumMat,QnumSum,pLinds,pRinds,leftSummary,rightSummary)

  finalQNs = 0
  @inbounds @simd for q = 1:length(rows)
    finalQNs += rows[q] > 0 && columns[q] > 0
  end

  newblocks = Array{Array{datatype,2},1}(undef,finalQNs)
  newind = Array{NTuple{2,Array{intType,2}},1}(undef,finalQNs)
  finalQblocksum = Array{NTuple{2,Q},1}(undef,finalQNs)
  counter = 0
  @inbounds for q = 1:length(rows)
    if rows[q] > 0 && columns[q] > 0
      counter += 1
      newblocks[counter] = blockfct(datatype,rows[q],columns[q])
      newind[counter] = (Lindexes[q],Rindexes[q])
      finalQblocksum[counter] = newQblocksum[q]
    end
  end

  newsize = Array{Array{intType,1},1}(undef,length(Qlabels))
  @inbounds @simd for i = 1:length(newsize)
    newsize[i] = [i]
  end

  newcurrblock = (currblock[1],currblock[2])
  return Qtens{datatype,Q}(newsize, newblocks, newind, newcurrblock, finalQblocksum, finalQnumMat, QnumSum, flux)
end

function convertQnumMat(QnumMat::Array{Array{Q,1},1}) where Q <: Qnum
  QnumSum = unique.(QnumMat)
  return convertQnumMat(QnumMat,QnumSum),QnumSum
end

function convertQnumMat(QnumMat::Array{Array{Q,1},1},QnumSum::Array{Array{Q,1},1}) where Q <: Qnum
  finalQnumMat = Array{Array{intType,1},1}(undef,length(QnumMat))
  @inbounds for i = 1:length(QnumMat)
    finalQnumMat[i] = Array{intType,1}(undef,length(QnumMat[i]))
    @inbounds for w = 1:length(QnumMat[i])
      y = 0
      notmatchingQN = true
      @inbounds while notmatchingQN
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
@inline function equalblocks(A::Array{Array{Q,1},1}) where Q <: Qnum
  sizes = ntuple(q->length(A[q]),length(A))
  return equalblocks(sizes)
end

@inline function equalblocks(sizes::Tuple) where Q <: Qnum
  row = sizes[1]
  column = prod(sizes) ÷ row
  i = 1
  @inbounds while row < column && i < length(sizes) - 1
    i += 1
    row *= sizes[i]
    column ÷= sizes[i]
  end
  return ([w for w = 1:i],[w for w = i+1:length(sizes)])
end

@inline function equalblocks(A::qarray)
  Rsize = A.size
  sizes = ntuple(q->prod(w->length(A.QnumMat[w]),Rsize[q]),length(Rsize))
  return equalblocks(sizes)
end

@inline function recoverQNs(q::Integer,QnumMat::Array{Array{intType,1},1},QnumSum::Array{Array{Q,1},1}) where Q <: Qnum
  return [getQnum(q,i,QnumMat,QnumSum) for i = 1:length(QnumMat[q])]
end

@inline function recoverQNs(q::Integer,Qt::Qtens{W,Q}) where {W <: Number, Q <: Qnum}
  return recoverQNs(q,Qt.QnumMat,Qt.QnumSum)
end
export recoverQNs

@inline function fullQnumMat(QnumMat::Array{Array{intType,1},1},QnumSum::Array{Array{Q,1},1}) where Q <: Qnum
  finalQnumMat = [recoverQNs(q,QnumMat,QnumSum) for q = 1:length(QnumMat)]
  return finalQnumMat
end

@inline function fullQnumMat(Qt::Qtens{W,Q}) where {W <: Number, Q <: Qnum}
  return fullQnumMat(Qt.QnumMat,Qt.QnumSum)
end
export fullQnumMat

"""
    Qtens(operator,QnumMat[,Arrows,zero=])

Creates a dense `operator` as a Qtensor with quantum numbers `QnumMat` on each index (`Array{Array{Q,1},1}`) according to arrow convention `Arrows` and all elements equal to `zero` are not included (default 0)
"""
function Qtens(optens::denstens,Qtensor::qarray;zero::Number=0.,currblock::currblockTypes=Qtensor.currblock)
  Op = makeArray(optens)
  finalQnumMat = fullQnumMat(Qtensor.QnumMat,Qtensor.QnumSum)
  return Qtens(Op,finalQnumMat,zero=zero,currblock=currblock)
end

function Qtens(optens::Array,Qtensor::qarray;zero::Number=0.,currblock::currblockTypes=Qtensor.currblock)
  finalQnumMat = fullQnumMat(Qtensor.QnumMat,Qtensor.QnumSum)
  return Qtens(optens,finalQnumMat,zero=zero,currblock=currblock)
end

"""
    Qtens(operator,QnumMat[,Arrows,zero=])

Creates a dense `operator` as a Qtensor with quantum numbers `QnumMat` on each index (`Array{Array{Q,1},1}`) according to arrow convention `Arrows` and all elements equal to `zero` are not included (default 0)
"""
@inline function Qtens(Op::R,Qlabels::Array{Array{Q,1},1},Arrows::U...;zero::Number=0.,currblock::currblockTypes=([i for i = 1:ndims(Op)-1],[ndims(Op)]),datatype::DataType=eltype(Op)) where {Q <: Qnum, W <: Number, R <: densTensType, U <: Union{Bool,Array{Bool,1}}}
  theseArrows = typeof(Arrows) <: Bool ? Arrows : (Arrows[1]...,)
  newQnumMat = [theseArrows[q] ? Qlabels[q] : inv.(Qlabels[q]) for q = 1:length(Qlabels)]
  return Qtens(Op,newQnumMat;zero=zero,currblock=currblock,datatype=datatype)
end

@inline function Qtens(Qlabels::Array{Array{Q,1},1},Op#=::R=#...;zero::Number=0.,currblock::currblockTypes=([i for i = 1:ndims(Op[1])-1],[ndims(Op[1])]),datatype::DataType=eltype(Op)) where {Q <: Qnum, W <: Number, R <: densTensType, U <: Union{Bool,Array{Bool,1}}}
  return ntuple(w->Qtens(Op[w],Qlabels,zero=zero,currblock=currblock,datatype=datatype),length(Op))
end

function Qtens(Qlabels::Array{Q,1},Op::R...;zero::Number=0.,currblock::currblockTypes=([i for i = 1:ndims(Op[1])-1],[ndims(Op[1])]),datatype::DataType=eltype(Op)) where {Q <: Qnum, W <: Number, R <: densTensType, U <: Union{Bool,Array{Bool,1}}}
  Qnumvec = [inv.(Qlabels),Qlabels]
  if length(Op) == 1
    return Qtens(Op[1],Qnumvec,zero=zero,currblock=currblock,datatype=datatype)
  else
    return ntuple(w->Qtens(Op[w],Qnumvec,zero=zero,currblock=currblock,datatype=datatype),length(Op))
  end
end

function makeQNsummaries(Qlabels::Array{Array{Q,1},1},Linds::Array{P,1},Rinds::Array{P,1},LR::Bool,flux::Q) where {W <: Number, Q <: Qnum, P <: Integer}
  if LR
    QNsummary = multi_indexsummary(Qlabels,Linds)
    leftSummary,rightSummary = LRsummary_invQ(QNsummary,flux)
  else
    QNsummary = multi_indexsummary(Qlabels,Rinds)
    rightSummary,leftSummary = LRsummary_invQ(QNsummary,flux)
  end
  newQblocksum = Array{NTuple{2,Q},1}(undef,length(QNsummary))
  @inbounds for q = 1:length(QNsummary)
    newQblocksum[q] = (leftSummary[q],rightSummary[q])
  end
  return QNsummary,leftSummary,rightSummary,newQblocksum
end

function makeQNsummaries(Qt::Qtens{W,Q},Linds::Array{P,1},Rinds::Array{P,1},LR::Bool) where {W <: Number, Q <: Qnum, P <: Integer}
  return makeQNsummaries(Qt.QnumSum,Linds,Rinds,LR,Qt.flux)
end

@inline function undefMat(outtype::DataType,x::Integer,y::Integer)
  return Array{outtype,2}(undef,x,y)
end

function Qtens(Op::densTensType,Qlabels::Array{Array{Q,1},1};zero::R=eltype(Op)(0),currblock::currblockTypes=([i for i = 1:ndims(Op)-1],[ndims(Op)]),datatype::DataType=eltype(Op),blockfct::Function=undefMat) where {Q <: Qnum, W <: Number, R <: Number}
  outtype = R
  if isapprox(norm(Op),0)
    return zeros(Qlabels,datatype=outtype,currblock=currblock)
  end

  pLinds = currblock[1]
  pRinds = currblock[2]
  
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

  Qt = Qtens(Qlabels,datatype=outtype,flux=invflux,currblock=currblock)

  Lsizes = sizes[pLinds]
  Rsizes = sizes[pRinds]

  pos = ntuple(i->Array{intType,1}(undef,length(Qt.currblock[i])),2)
  for q = 1:length(Qt.T)
    for b = 1:size(Qt.T[q],2)
      @inbounds for a = 1:size(Qt.T[q],1)
        for i = 1:2
          blockindex = i == 1 ? a : b
          @inbounds for w = 1:length(Qt.currblock[i])
            pos[i][w] = Qt.ind[q][i][w,blockindex] + 1
          end
        end
        x = pos2ind(pos[1],Lsizes)
        y = pos2ind(pos[2],Rsizes)
        Qt.T[q][a,b] = Op_mat[x,y]
      end
    end
  end


  keepers = Array{Bool,1}(undef,length(Qt.T))
  for q = 1:length(keepers)
    allzero = (size(Qt.T[q],1) > 0 && size(Qt.T[q],2) > 0)
    w = 0
    while allzero && w < length(Qt.T[q])
      w += 1
      allzero &= isapprox(Qt.T[q][w],zero)
    end
    keepers[q] = !allzero
  end

  if sum(keepers) < length(Qt.T)
    Qt.T = Qt.T[keepers]
    Qt.ind = Qt.ind[keepers]
    Qt.Qblocksum = Qt.Qblocksum[keepers]
  end
  return Qt
end

"""
    Qtens(operator,Qlabels[,Arrows,zero=])

Creates a dense `operator` as a Qtensor with quantum numbers `Qlabels` common to all indices (`Array{Q,1}`) according to arrow convention `Arrows` and all elements equal to `zero` are not included (default 0)
"""
@inline function Qtens(operator::Array{W,N},Qlabels::Array{Q,1},arrows::U;zero::Number=0.,currblock::currblockTypes=([i for i = 1:ndims(Op)-1],[ndims(Op)]),datatype::DataType=eltype(Op),blockfct::Function=undefMat) where {Q <: Qnum, U <: Union{Bool,Array{Bool,1}}, W <: Number, N}
  newQlabels = Array{Q,1}[arrows[a] ? Qlabels[a] : inv.(Qlabels[a]) for a = 1:length(arrows)]
  return Qtens(operator,newQlabels,zero=zero,currblock=currblock,datatype=datatype,blockfct=blockfct)
end

"""
    Qtens(A)

`A` is a Qtensor; makes shell of a Qtensor with only meta-data (no blocks, no row reductions); used mainly for copies
"""
@inline function Qtens(A::Qtens{W,Q};zero::Number=0.,currblock::currblockTypes=([i for i = 1:ndims(Op)-1],[ndims(Op)]),datatype::DataType=eltype(Op),blockfct::Function=undefMat) where {W <: Number, Q <: Qnum}
  newQlabels = fullQnumMat(A)
  return Qtens(newQlabels,zero=zero,currblock=currblock,datatype=datatype,blockfct=blockfct,flux=A.flux)
end



import Base.rand
"""
    rand(A[,arrows])

generates a random tensor from inputting another tensor (rank-2) or from quantum number labels; can assign `arrows` for Qtensors if alternate arrow convention is requested
"""
@inline function rand(Qlabels::Array{Array{Q,1},1}, arrows::Array{Bool,1};datatype::DataType=Float64,flux::Q=Q()) where Q <: Qnum
  newQlabels = Array{Q,1}[arrows[a] ? Qlabels[a] : inv.(Qlabels[a]) for a = 1:length(arrows)]
  return Qtens(newQlabels,datatype=datatype,flux=flux,blockfct=rand)
end

@inline function rand(datatype::DataType,Qlabels::Array{Array{Q,1},1};flux::Q=Q(),currblock::currblockTypes=equalblocks(Qlabels)) where {Q <: Qnum, P <: Integer}
  return Qtens(Qlabels,datatype=datatype,flux=flux,currblock=currblock,blockfct=rand)
end

@inline function rand(Qlabels::Array{Array{Q,1},1};flux::Q=Q(),currblock::currblockTypes=equalblocks(Qlabels),datatype::DataType=Float64) where {Q <: Qnum, P <: Integer}
  return Qtens(Qlabels,datatype=datatype,flux=flux,currblock=currblock,blockfct=rand)
end

@inline function rand(currQtens::Qtens{W,Q}) where {W <: Number, Q <: Qnum}
  Qlabels = fullQnumMat(currQtens)
  return Qtens(Qlabels,datatype=W,currblock=currQtens.currblock,flux=currQtens.flux,blockfct=rand)
end

@inline function rand(datatype::DataType,currQtens::Qtens{W,Q}) where {W <: Number, Q <: Qnum}
  Qlabels = fullQnumMat(currQtens)
  return Qtens(Qlabels,datatype=datatype,currblock=currQtens.currblock,flux=currQtens.flux,blockfct=rand)
end





@inline function basesize(Qtensor::qarray)
  return ntuple(i->length(Qtensor.QnumMat[i]),length(Qtensor.QnumMat))
end

@inline function basesize(Qlabels::Array{Array{Q,1},1}) where Q <: Union{Qnum,Integer}
  return ntuple(i->length(Qlabels[i]),length(Qlabels))
end
export basesize

@inline function basedims(Qtensor::qarray)
  return length(Qtensor.QnumMat)
end



import Base.zeros
@inline function zeros(Qlabels::Array{Array{Q,1},1}, arrows::Array{Bool,1};datatype::DataType=Float64,flux::Q=Q()) where Q <: Qnum
  newQlabels = Array{Q,1}[arrows[a] ? Qlabels[a] : inv.(Qlabels[a]) for a = 1:length(arrows)]
  return Qtens(newQlabels,datatype=datatype,flux=flux,blockfct=zeros)
end

@inline function zeros(Qlabels::Array{Array{Q,1},1};flux::Q=Q(),currblock::currblockTypes=equalblocks(Qlabels),datatype::DataType=Float64) where {Q <: Qnum, P <: Integer}
  return Qtens(Qlabels,datatype=datatype,flux=flux,currblock=currblock,blockfct=zeros)
end

@inline function zeros(datatype::DataType,Qlabels::Array{Array{Q,1},1};flux::Q=Q(),currblock::currblockTypes=equalblocks(Qlabels)) where {Q <: Qnum, P <: Integer}
  return Qtens(Qlabels,datatype=datatype,flux=flux,currblock=currblock,blockfct=zeros)
end

@inline function zeros(currQtens::Qtens{W,Q}) where {W <: Number, Q <: Qnum}
  Qlabels = fullQnumMat(currQtens)
  return Qtens(Qlabels,datatype=W,currblock=currQtens.currblock,flux=currQtens.flux,blockfct=zeros)
end

@inline function zeros(datatype::DataType,currQtens::Qtens{W,Q}) where {W <: Number, Q <: Qnum}
  Qlabels = fullQnumMat(currQtens)
  return Qtens(Qlabels,datatype=datatype,currblock=currQtens.currblock,flux=currQtens.flux,blockfct=zeros)
end

import Base.zero
  """Like the default function zero(t::Array), return an object with the same properties containing only zeros."""
  @inline function zero(Qt::qarray)
  return zeros(Qt)
end


"""
    Qtens{Z,Q}()

Default initializer to an empty tensor with type `Z` and quantum number `Q`
"""
@inline function Qtens{Z,Q}() where {Z <: Number,Q <: Qnum}
  return zeros([[Q()]])
end







function checkType(A::Qtens{W,Q},B::Qtens{G,Q}) where {W <: Number, G <: Number, Q <: Qnum}
  outType = typeof(eltype(A)(1)*eltype(B)(1))
  return convertTens(outType,A),convertTens(outType,B)
end



#  import .tensor.convertTens
"""
    convertTens(T,Qt)

Convert Qtensor `Qt` to type `T`
"""
function convertTens(T::DataType, Qt::Qtens{Z,Q}) where {Z <: Number, Q <: Qnum}
  if Z == T
    return Qt
  else
    newsize = Qt.size #[copy(Qt.size[i]) for i = 1:length(Qt.size)]
    newQblocksum = Qt.Qblocksum
    newT = [copy(Qt.T[w]) for w = 1:length(Qt.T)]
    newcurrblock = Qt.currblock #(Qt.currblock[1],Qt.currblock[2]) #(copy(Qt.currblock[1]),copy(Qt.currblock[2]))
    return Qtens{T,Q}(newsize,newT,copy(Qt.ind),newcurrblock,newQblocksum,
                      Qt.QnumMat,Qt.QnumSum,Qt.flux)
  end
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
  copyQtind = Qt.ind #[(copy(Qt.ind[q][1]),copy(Qt.ind[q][2])) for q = 1:length(Qt.ind)]
  newcurrblock = (copy(Qt.currblock[1]),copy(Qt.currblock[2]))
  newQblocksum = Qt.Qblocksum
  newQnumSum = Qt.QnumSum
  return Qtens{W,Q}(newsize,copyQtT,copyQtind,newcurrblock,
                    newQblocksum,Qt.QnumMat,newQnumSum,Qt.flux)
end

####################################################
####################################################

function QnumList(Qt::Qtens{W,Q},pLinds::Array{P,1},pRinds::Array{P,1},
                leftSummary::Array{Q,1},rightSummary::Array{Q,1}) where {P <: Integer, W <: Number, Q <: Qnum}

  Lsizes = [length(Qt.QnumMat[pLinds[a]]) for a = 1:length(pLinds)]
  Rsizes = [length(Qt.QnumMat[pRinds[a]]) for a = 1:length(pRinds)]

  return QnumList(Lsizes,Rsizes,Qt.QnumMat,Qt.QnumSum,pLinds,pRinds,leftSummary,rightSummary)
end

function QnumList(Lsizes::Array{P,1},Rsizes::Array{P,1},QnumMat::Array{Array{P,1},1},QnumSum::Array{Array{Q,1},1},pLinds::Array{P,1},pRinds::Array{P,1},
                  leftSummary::Array{Q,1},rightSummary::Array{Q,1}) where {P <: Integer, W <: Number, Q <: Qnum}

  leftQNs,Lbigtosub,rows = QnumList(Lsizes,QnumMat,QnumSum,pLinds,leftSummary)
  rightQNs,Rbigtosub,columns = QnumList(Rsizes,QnumMat,QnumSum,pRinds,rightSummary)

  Lindexes,Rindexes,leftQNs,rightQNs,rows,columns = makeIndexes(pLinds,Lsizes,leftQNs,Lbigtosub,rows,pRinds,Rsizes,rightQNs,Rbigtosub,columns)

  return leftQNs,Lbigtosub,rows,Lindexes,rightQNs,Rbigtosub,columns,Rindexes
end

@inline function loadindex(nonzero_sizes::Array{Bool,1},Linds::intType,Lsizes::Array{intType,1},Lindexes::Array{Array{intType,2},1},
                           leftQNs::Array{intType,1},Lbigtosub::Array{intType,1},rows::Array{intType,1})
  Lpos = makepos(Linds)
  @inbounds for y = 1:length(Lbigtosub)
    position_incrementer!(Lpos,Lsizes)
    q = leftQNs[y]
    if q > 0 && nonzero_sizes[q]
      w = Lbigtosub[y]
      if rows[q] > 0
        @inbounds @simd for r = 1:Linds
          Lindexes[q][r,w] = Lpos[r] - 1
        end
      end
    end
  end
  nothing
end

function QnumList(sizes::Union{Array{P,1},NTuple{G,P}},QnumMat::Array{Array{P,1},1},QnumSum::Array{Array{Q,1},1},
                        vec::Array{P,1},QnumSummary::Array{Q,1}) where {Q <: Qnum, G, P <: Integer}
  ninds = length(sizes)
  if ninds == 0
    matchingQNs,returnvector,QNblocksizes = [1],[1],[1]
  else
    pos = makepos(ninds)

    numElements = 1
    @inbounds @simd for w = 1:length(sizes)
      numElements *= sizes[w]
    end
    matchingQNs = Array{intType,1}(undef,numElements)
    returnvector = Array{intType,1}(undef,numElements)

    numQNs = length(QnumSummary)
    QNblocksizes = zeros(intType,numQNs)

    @inbounds for y = 1:numElements
      position_incrementer!(pos,sizes)
      currQN = Q()
      @inbounds for b = 1:ninds
        currQN += getQnum(vec[b],pos[b],QnumMat,QnumSum)
      end
      notmatchQNs = true
      q = 0
      @inbounds while (q < numQNs) && notmatchQNs
        q += 1
        notmatchQNs = currQN != QnumSummary[q]
      end

      if notmatchQNs
        matchingQNs[y] = 0
        returnvector[y] = 0
      else
        matchingQNs[y] = q

        QNblocksizes[q] += 1
        returnvector[y] = QNblocksizes[q]
      end
    end
  end
  return matchingQNs,returnvector,QNblocksizes
end

function makeIndexes(pLinds::Array{intType,1},Lsizes::Array{intType,1},leftQNs::Array{intType,1},Lbigtosub::Array{intType,1},rows::Array{intType,1},
                     pRinds::Array{intType,1},Rsizes::Array{intType,1},rightQNs::Array{intType,1},Rbigtosub::Array{intType,1},columns::Array{intType,1})
  Linds = length(pLinds)
  Rinds = length(pRinds)
  numQNs = length(rows)
  nonzero_sizes = Array{Bool,1}(undef,numQNs)
  @inbounds @simd for q = 1:numQNs
    nonzero_sizes[q] = rows[q] > 0 && columns[q] > 0
  end

#  finalQNs = sum(nonzero_sizes)

  Lindexes = Array{Array{intType,2},1}(undef,numQNs)
  Rindexes = Array{Array{intType,2},1}(undef,numQNs)
  @inbounds @simd for q = 1:numQNs
    if nonzero_sizes[q]
      Lindexes[q] = Array{intType,2}(undef,Linds,rows[q])
      Rindexes[q] = Array{intType,2}(undef,Rinds,columns[q])
    end
  end

  loadindex(nonzero_sizes,Linds,Lsizes,Lindexes,leftQNs,Lbigtosub,rows)
  loadindex(nonzero_sizes,Rinds,Rsizes,Rindexes,rightQNs,Rbigtosub,columns)
  return Lindexes,Rindexes,leftQNs,rightQNs,rows,columns
end

function multi_indexsummary(QnumSum::Array{Array{Q,1},1},vec::Array{P,1}) where {Q <: Qnum, P <: Integer}
  ninds = length(vec)
  if ninds > 0
    QsumSizes = ntuple(a->length(QnumSum[vec[a]]),ninds)
    Qsumel = 1
    @inbounds @simd for a = 1:ninds
      Qsumel *= QsumSizes[a]
    end
    Qsumvec = Array{Q,1}(undef,Qsumel)

    counter = 0
    pos = makepos(ninds)

    @inbounds for g = 1:Qsumel
      position_incrementer!(pos,QsumSizes)

      currQN = Q()
      @inbounds for b = 1:ninds
        currQN += QnumSum[vec[b]][pos[b]]
      end
      addQ = true
      w = 0
      @inbounds while w < counter && addQ
        w += 1
        addQ = currQN != Qsumvec[w]
      end
      if addQ
        counter += 1
        Qsumvec[counter] = currQN
      end
    end
    outQNsum = Qsumvec[1:counter]
  else
    outQNsum = [Q()]
  end
  return outQNsum
end

function multi_indexsummary(Qt::Qtens{W,Q},vec::Array{P,1}) where {W <: Number, Q <: Qnum, P <: Integer}
  QnumSum = Qt.QnumSum
  return multi_indexsummary(QnumSum,vec)
end
export multi_indexsummary

function findsizes(Qt::Qtens{W,Q},Linds::Array{P,1}) where {W <: Number, Q <: Qnum, P <: Integer}
  nind = length(Linds)
  Lnonzero = nind > 0
  Lsize = Lnonzero ? prod(Linds) : 1
  return Lsize,nind,Lnonzero
end


function checkorder!(Qt::Qtens{W,Q},side::Integer) where {W <: Number, Q <: Qnum}
  xsorted = issorted(Qt.currblock[side])  
  Lindmat = Array{Array{intType,2},1}(undef,length(Qt.ind))
  if !xsorted
    xorder = sortperm(Qt.currblock[side])
    sort!(Qt.currblock[side])
    @inbounds for q = 1:length(Qt.ind)
      Lindmat[q] = Qt.ind[q][side][xorder,:]
    end
  else
    @inbounds for q = 1:length(Qt.ind)
      Lindmat[q] = Qt.ind[q][side]
    end
  end
  return Lindmat
end

function checkorder!(Qt::Qtens{W,Q}) where {W <: Number, Q <: Qnum}
  Lindmat = checkorder!(Qt,1)
  Rindmat = checkorder!(Qt,2)
  @inbounds for q = 1:length(Qt.ind)
    Qt.ind[q] = (Lindmat[q],Rindmat[q])
  end
  nothing
end

function LRsummary_invQ(QNsummary::Array{Q,1},flux::Q) where Q <: Qnum
  leftSummary = QNsummary
  rightSummary = Array{Q,1}(undef,length(leftSummary))
  @inbounds for q = 1:length(rightSummary)
    rightSummary[q] = flux - QNsummary[q]
  end
  return leftSummary,rightSummary
end


function changeblock(Qt::Qtens{W,Q},Linds::Array{P,1},Rinds::Array{P,1};zeroblocks::Bool=false) where {W <: Number, Q <: Qnum, P <: Integer}
  return changeblock(Qt,(Linds,Rinds),zeroblocks=zeroblocks)
end

function changeblock(Qt::Qtens{W,Q},newblock::Array{Array{P,1},1};zeroblocks::Bool=false) where {W <: Number, Q <: Qnum, P <: Integer}
  return changeblock(Qt,(newblock[1],newblock[2]),zeroblocks=removezerblocks)
end

const parallel_trigger = 1000 #set as of DMRjulia v0.8.6

function changeblock(Qt::Qtens{W,Q},newblock::NTuple{2,Array{P,1}};minelements::Integer=parallel_trigger,zeroblocks::Bool=false) where {W <: Number, Q <: Qnum, P <: Integer}
  if Qt.currblock == newblock
    newQt = Qt
  else
    checkorder!(Qt)

    Linds = newblock[1]
    Lsize,Lnind,Lnonzero = findsizes(Qt,Linds)
    Rinds = newblock[2]
    Rsize,Rnind,Rnonzero = findsizes(Qt,Rinds)

    ninds = Lnind + Rnind

    LR = Lsize <= Rsize
    QNsummary,leftSummary,rightSummary,newQblocksum = makeQNsummaries(Qt,Linds,Rinds,LR)
    leftQNs,Lbigtosub,rows,Lindexes,rightQNs,Rbigtosub,columns,Rindexes = QnumList(Qt,Linds,Rinds,leftSummary,rightSummary)

    outQNs = LR ? leftQNs : rightQNs

    Tcheck = 0
    @inbounds @simd for q = 1:length(Qt.T)
      Tcheck += length(Qt.T[q])
    end
    rowcolcheck = 0
    @inbounds @simd for q = 1:length(rows)
      rowcolcheck += rows[q]*columns[q]
    end

    fulltens = Tcheck == rowcolcheck
    newblocks = Array{Array{W,2},1}(undef,length(rows))
    keepblocks = Array{Bool,1}(undef,length(rows))
    Matfct = fulltens ? undefMat : zeros
    @inbounds for q = 1:length(rows)
      keepblocks[q] = rows[q] > 0 && columns[q] > 0
      if keepblocks[q]
        x,y = rows[q],columns[q]
        newblocks[q] = Matfct(W,x,y)
      end
    end

    reblock!(newblocks,Qt,outQNs,LR,Linds,Rinds,Lbigtosub,Rbigtosub,minelements=minelements)

    if !zeroblocks
      checkzeroblocks!(newblocks,keepblocks)
    end

    outTens = Array{Array{W,2},1}(undef,sum(keepblocks))
    newrowcols = Array{NTuple{2,Array{intType,2}},1}(undef,length(outTens))

    count = 0
    @inbounds for q = 1:length(keepblocks)
      if keepblocks[q]
        count += 1
        outTens[count] = newblocks[q]
        newrowcols[count] = (Lindexes[q],Rindexes[q])
      end
    end

    if sum(keepblocks) != length(newQblocksum)
      finalQblocksum = newQblocksum[keepblocks]
    else
      finalQblocksum = newQblocksum
    end
    finalblocks = (newblock[1],newblock[2])

    newQt = Qtens{W,Q}(Qt.size,outTens,newrowcols,finalblocks,finalQblocksum,Qt.QnumMat,Qt.QnumSum,Qt.flux)
  end
  return newQt
end
export changeblock



@inline function AAzeropos2ind(currpos::Array{X,1},S::NTuple{P,X},selector::NTuple{G,X}) where X <: Integer where G where P
  x = 0
  @inbounds @simd for i = G:-1:1
    x *= S[selector[i]]
    x += currpos[selector[i]]
  end
  return x+1
end

@inline function innerloadpos!(y::P,Rorigsize::P,thispos::Array{P,1},thiscurrblock_two::Array{P,1},thisind_two::Array{P,2}) where P <: Integer
  @inbounds @simd for i = 1:Rorigsize
    thispos[thiscurrblock_two[i]] = thisind_two[i,y]
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

@inline function doubleloop_right(newblocks::Array{Array{W,2},1},thisTens::Array{W,2},leftQNs::Array{S,1},LRpos::Bool,basesize::NTuple{P,S},posvecs::Array{Array{S,1},1},
                          thisind_one::Array{S,2},thisind_one_sizetwo::S,Lorigsize::S,thiscurrblock_one::Array{S,1},Linds::NTuple{Z,S},Lbigtosub::Array{S,1},
                          thisind_two::Array{S,2},thisind_two_sizetwo::S,Rorigsize::S,thiscurrblock_two::Array{S,1},Rinds::NTuple{G,S},Rbigtosub::Array{S,1}) where {P, Z, G, W <: Number, S <: Integer}
  Threads.@threads for x = 1:thisind_one_sizetwo
    thisthread = Threads.threadid()
    thispos = posvecs[thisthread]

    innerloadpos!(x,Lorigsize,thispos,thiscurrblock_one,thisind_one)

    @inbounds for y = 1:thisind_two_sizetwo
    inputval = thisTens[x,y]
    innerloadpos!(y,Rorigsize,thispos,thiscurrblock_two,thisind_two)
    
    innerloop(x,y,newblocks,inputval,leftQNs,LRpos,basesize,thispos,
            thisind_one,thisind_one_sizetwo,Linds,Lbigtosub,Rinds,Rbigtosub)
    end
  end
  nothing
end

@inline function doubleloop_left(newblocks::Array{Array{W,2},1},thisTens::Array{W,2},leftQNs::Array{S,1},LRpos::Bool,basesize::NTuple{P,S},posvecs::Array{Array{S,1},1},
                          thisind_one::Array{S,2},thisind_one_sizetwo::S,Lorigsize::S,thiscurrblock_one::Array{S,1},Linds::NTuple{Z,S},Lbigtosub::Array{S,1},
                          thisind_two::Array{S,2},thisind_two_sizetwo::S,Rorigsize::S,thiscurrblock_two::Array{S,1},Rinds::NTuple{G,S},Rbigtosub::Array{S,1}) where {P, Z, G, W <: Number, S <: Integer}
  Threads.@threads for y = 1:thisind_two_sizetwo
    thisthread = Threads.threadid()
    thispos = posvecs[thisthread]

    innerloadpos!(y,Rorigsize,thispos,thiscurrblock_two,thisind_two)

    @inbounds for x = 1:thisind_one_sizetwo
      inputval = thisTens[x,y]
      innerloadpos!(x,Lorigsize,thispos,thiscurrblock_one,thisind_one)
      
      innerloop(x,y,newblocks,inputval,leftQNs,LRpos,basesize,thispos,
              thisind_one,thisind_one_sizetwo,Linds,Lbigtosub,Rinds,Rbigtosub)
    end
  end
  nothing
end

@inline function doubleloop_reg(newblocks::Array{Array{W,2},1},thisTens::Array{W,2},leftQNs::Array{S,1},LRpos::Bool,basesize::NTuple{P,S},posvec::Array{S,1},
                          thisind_one::Array{S,2},thisind_one_sizetwo::S,Lorigsize::S,thiscurrblock_one::Array{S,1},Linds::NTuple{Z,S},Lbigtosub::Array{S,1},
                          thisind_two::Array{S,2},thisind_two_sizetwo::S,Rorigsize::S,thiscurrblock_two::Array{S,1},Rinds::NTuple{G,S},Rbigtosub::Array{S,1}) where {P, Z, G, W <: Number, S <: Integer}
  for x = 1:thisind_one_sizetwo
    innerloadpos!(x,Lorigsize,posvec,thiscurrblock_one,thisind_one)

    @inbounds for y = 1:thisind_two_sizetwo
      inputval = thisTens[x,y]
      innerloadpos!(y,Rorigsize,posvec,thiscurrblock_two,thisind_two)
      
      innerloop(x,y,newblocks,inputval,leftQNs,LRpos,basesize,posvec,
              thisind_one,thisind_one_sizetwo,Linds,Lbigtosub,Rinds,Rbigtosub)
    end
  end
  nothing
end

@inline function reblock!(newblocks::Array{Array{W,2},1},Qt::Qtens{W,Q},leftQNs::Array{S,1},LR::Bool,
                          Linds::Array{S,1},Rinds::Array{S,1},Lbigtosub::Array{S,1},Rbigtosub::Array{S,1};minelements::Integer=parallel_trigger) where {W <: Number, Q <: Qnum, S <: Integer}

  thiscurrblock_one = Qt.currblock[1]
  thiscurrblock_two = Qt.currblock[2]

  Lorigsize = length(Qt.currblock[1])
  Rorigsize = length(Qt.currblock[2])

  posvec = Array{intType,1}(undef,length(Qt.QnumMat))
  numthreads = Threads.nthreads()
  posvecs = Array{Array{intType,1},1}(undef,numthreads)
  @inbounds @simd for q = 1:numthreads
    posvecs[q] = Array{intType,1}(undef,length(Qt.QnumMat)) #copy(posvec)
  end
                        
  basesize = ntuple(i->length(Qt.QnumMat[i]),length(Qt.QnumMat))
  tup_Linds = ntuple(i->Linds[i],length(Linds))
  tup_Rinds = ntuple(i->Rinds[i],length(Rinds))
  nQNs = length(Qt.ind)
  @inbounds for q = 1:nQNs

    thisTens = Qt.T[q]
    thisind = Qt.ind[q]

    thisind_one_sizetwo = size(thisind[1],2)
    thisind_one = thisind[1]

    thisind_two_sizetwo = size(thisind[2],2)
    thisind_two = thisind[2]

    if max(thisind_two_sizetwo,thisind_one_sizetwo) > minelements
      if thisind_two_sizetwo < thisind_one_sizetwo
        doubleloop_right(newblocks,thisTens,leftQNs,LR,basesize,posvecs,
                          thisind_one,thisind_one_sizetwo,Lorigsize,thiscurrblock_one,tup_Linds,Lbigtosub,
                          thisind_two,thisind_two_sizetwo,Rorigsize,thiscurrblock_two,tup_Rinds,Rbigtosub)
      else
        doubleloop_left(newblocks,thisTens,leftQNs,LR,basesize,posvecs,
                        thisind_one,thisind_one_sizetwo,Lorigsize,thiscurrblock_one,tup_Linds,Lbigtosub,
                        thisind_two,thisind_two_sizetwo,Rorigsize,thiscurrblock_two,tup_Rinds,Rbigtosub)
      end
    else
      doubleloop_reg(newblocks,thisTens,leftQNs,LR,basesize,posvec,
                        thisind_one,thisind_one_sizetwo,Lorigsize,thiscurrblock_one,tup_Linds,Lbigtosub,
                        thisind_two,thisind_two_sizetwo,Rorigsize,thiscurrblock_two,tup_Rinds,Rbigtosub)
    end
  end
  nothing
end

@inline function checkzeroblocks!(newblocks::Array{Array{W,2},1},keepblocks::Array{Bool,1}) where W <: Number
  if length(newblocks) > 1
    @inbounds for q = 1:length(keepblocks)
      if keepblocks[q]
        w = 0
        allzero = true
        @inbounds while allzero && w < length(newblocks[q])
          w += 1
          allzero &= isapprox(newblocks[q][w],0)
        end
        keepblocks[q] &= !allzero
      end
    end
  end
  nothing
end

####################################################
####################################################


@inline function countload(A::Array{W,1},B::Array{W,1}) where W 
  loadvec = Array{W,1}(undef,length(A)+length(B))
  counter = 0
  @inbounds @simd for k = 1:length(A)
    counter += 1
    loadvec[counter] = A[k]
  end
  @inbounds @simd for k = 1:length(B)
    counter += 1
    loadvec[counter] = B[k]
  end
  return loadvec
end


####################################################
####################################################

function newindexsizeone!(Qt::Qtens{W,Q},S::Integer...) where {W <: Number, Q <: Qnum}
  size_ones = 0
  @inbounds for w = 1:length(S)
    size_ones += S[w] == 1
  end
  base_ones = 0
  @inbounds for w = 1:length(Qt.QnumMat)
    base_ones += length(Qt.QnumMat[w]) == 1
  end

  newindices = size_ones - base_ones

  if newindices > 0
    @inbounds for q = 1:length(Qt.ind)
      newind = Array{intType,2}(undef,size(Qt.ind[q][2],1)+newindices,size(Qt.ind[q][2],2))  #vcat(Qt.ind[q][2],zeros(intType,newindices,size(Qt.ind[q][2],2)))
      newind[1:size(Qt.ind[q][2],1),:] = Qt.ind[q][2]
      newind[size(Qt.ind[q][2],1)+1:end,:] = zeros(intType,newindices,size(Qt.ind[q][2],2))

      Qt.ind[q] = (Qt.ind[q][1],newind)
    end
    newinds = Array{intType,1}(undef,newindices)
    @inbounds @simd for w = 1:newindices
      newinds[w] = length(Qt.QnumMat) + w
    end

    newcurrblock = countload(Qt.currblock[2],newinds)

    Qt.currblock = (Qt.currblock[1],newcurrblock)

    zeroQN = Q()
    newindex = Array{Array{Q,1},1}(undef,newindices) #[[zeroQN] for w = 1:newindices]    
    @inbounds @simd for q = 1:newindices
      newindex[q] = [zeroQN]
    end

    Qt.QnumSum = countload(Qt.QnumSum,newindex)

    newQnum = Array{Array{intType,1},1}(undef,newindices)
    @inbounds @simd for w = 1:newindices
      newQnum[w] = [1]
    end

    Qt.QnumMat = countload(Qt.QnumMat,newQnum)
  end
  nothing
end
export newindexsizeone!

#  import ..tensor.reshape!
"""
    reshape!(M,a...[,merge=])

In-place reshape for Qtensors (otherwise makes a copy); can also make Qtensor unreshapable with `merge`, joining all grouped indices together

# Warning
If the Qtensor size is (10,1,2) and we want to reshape in to a (10,2) tensor, this algorithm will reshape to a 
[[1,2],[3]] be default instead of [[1],[2,3]], so beware.

See also: [`reshape`](@ref)
"""
function reshape!(Qt::Qtens{W,Q}, S::Integer...;merge::Bool=false) where {W <: Number, Q <: Qnum}
  Rsize = recoverShape(Qt,S...)
  outQt = reshape!(Qt,Rsize,merge=merge)
  newindexsizeone!(outQt,S...)
  return outQt
end

function reshape!(Qt::Qtens{W,Q}, newQsize::Union{Array{Array{P,1},1},Tuple};merge::Bool=false) where {W <: Number, Q <: Qnum, P <: intType}
  order = vcat(newQsize...)
  if !issorted(order)
    permutedims!(Qt,order)
  end
  if typeof(newQsize) <: Array{Array{intType,1},1}
    Qt.size = newQsize
  else
    Qt.size = [[newQsize[q][w] for w = 1:length(newQsize[q])] for q = 1:length(newQsize)]    
  end
  if merge
    outQt = mergereshape!(Qt)
  else
    outQt = Qt
  end
  return outQt
end

function reshape!(Qt::Qtens{W,Q}, newQsize::Array{P,1}...;merge::Bool=false) where {W <: Number, Q <: Qnum, P <: Integer}
  order = vcat(newQsize)
  if !issorted(order)
    permutedims!(Qt,order)
  end
  return reshape!(Qt,[newQsize[i] for i = 1:length(newQsize)],merge=merge)
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
@inline function reshape(Qt::Qtens{W,Q}, S::Integer...;merge::Bool=false) where {W <: Number, Q <: Qnum}
  return reshape!(copy(Qt),S...,merge=merge)
end

@inline function reshape(Qt::Qtens{W,Q}, newQsize::Array{Array{P,1},1};merge::Bool=false) where {W <: Number, Q <: Qnum, P <: Integer}
  return reshape!(copy(Qt), newQsize...,merge=merge)
end

@inline function reshape(Qt::Qtens{W,Q}, newQsize::Array{P,1}...;merge::Bool=false) where {W <: Number, Q <: Qnum, P <: Integer}
  return reshape!(copy(Qt), newQsize...,merge=merge)
end

function (reshape(Qt::Array{W,N}, newQsize::Array{Array{P,1},1};merge::Bool=false)::Array{W,length(newQsize)}) where {W <: Number, N, P <: Integer}
  M = Array{intType,1}(undef,length(newQsize))
  counter = 0
  @inbounds for g = 1:length(newQsize)
    counter += 1
    if length(newQsize[g]) > 0
      M[counter] = prod(b->size(Qt,b),newQsize[g])
    else
      M[counter] = 1
    end
  end
  return reshape(copy(Qt), M...)
end

@inline function getQnum(a::Integer,b::Integer,QnumMat::Array{Array{P,1},1},QnumSum::Array{Array{Q,1},1}) where {Q <: Qnum, P <: intType}
  Qnumber = QnumMat[a][b]
  return QnumSum[a][Qnumber]
end

@inline function getQnum(a::Integer,b::Integer,Qt::Qtens{W,Q}) where {Q <: Qnum, W <: Number}
  return getQnum(a,b,Qt.QnumMat,Qt.QnumSum)
end
export getQnum


function makenewindsL(LR::Integer,newQt::qarray,Qt::qarray,Rsize::Array{Array{P,1},1},offset::Integer) where P <: Integer
  newindsL = Array{Array{intType,2},1}(undef,length(newQt.T)) #[Array{intType,2}(undef,length(newQt.currblock[1]),size(newQt.ind[q][LR],2)) 
  @inbounds for q = 1:length(newQt.T)
    newindsL[q] = Array{intType,2}(undef,length(newQt.currblock[1]),size(newQt.ind[q][LR],2))
  end
  for q = 1:length(newindsL)
    for i = 1:size(newindsL[q],1)
      b = i + offset
      @inbounds for x = 1:size(newindsL[q],2)
        val = 0
        @inbounds @simd for a = length(Rsize[b]):-1:1
          index = Rsize[b][a]
          val *= length(Qt.QnumMat[index])
          val += newQt.ind[q][LR][index,x]
        end
        newindsL[q][i,x] = val
      end
    end
  end
  return newindsL
end

function makenewindsR(LR::Integer,newQt::qarray,Qt::qarray,Rsize::Array{Array{P,1},1},offset::Integer) where P <: Integer
  newindsR = Array{Array{intType,2},1}(undef,length(newQt.T)) #[Array{intType,2}(undef,length(newQt.currblock[1]),size(newQt.ind[q][LR],2)) 
  @inbounds for q = 1:length(newQt.T)
    newindsR[q] = Array{intType,2}(undef,length(newQt.currblock[LR]),size(newQt.ind[q][LR],2))
  end
  for q = 1:length(newindsR)
    for i = 1:size(newindsR[q],1)
      b = i + offset
      @inbounds for x = 1:size(newindsR[q],2)
        val = 0
        @inbounds @simd for a = length(Rsize[b]):-1:1
          index = Rsize[b][a]
          val *= length(Qt.QnumMat[index])
          val += newQt.ind[q][LR][a,x]
        end
        newindsR[q][i,x] = val
      end
    end
  end
  return newindsR
end

function mergeQNloop!(ninds::Integer,numElements::Integer,vec::Array{P,1},pos::Array{P,1},
                      sizes::Tuple,currflux::Q,QNsummary::Array{Q,1},
                      current_QnumMat::Array{P,1},Qt::Qtens{W,Q}) where {W <: Number, Q <: Qnum, P <: Integer}
  @inbounds for y = 1:numElements
    position_incrementer!(pos,sizes)
    currQN = currflux
    @inbounds for b = 1:ninds
      currQN += getQnum(vec[b],pos[b],Qt)
    end
    b = 0
    findmatch = true
    @inbounds while findmatch && b < length(QNsummary)
      b += 1
      findmatch = QNsummary[b] != currQN
    end
    current_QnumMat[y] = b
  end
  nothing
end

"""
    mergereshape!(M)

Groups all joined indices together to make one index that is unreshapable.  Dense tensors are unaffected.

See also: [`reshape!`](@ref)
"""
function mergereshape!(Qt::Qtens{W,Q};currblock::currblockTypes=equalblocks(Qt)) where {W <: Number, Q <: Qnum, P <: Integer}

  Rsize = Qt.size
  newdim = length(Rsize)

  newQnumMat = Array{Array{intType,1},1}(undef,newdim)
  newQnumSum = Array{Array{Q,1},1}(undef,newdim)

  zeroQN = Q()
  truesize = basesize(Qt)

  @inbounds for a = 1:length(Rsize)
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

      mergeQNloop!(ninds,numElements,vec,pos,sizes,thisflux,QNsummary,current_QnumMat,Qt)

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
  merged_currblock = ([i for i = 1:length(Rsize)-1],[length(Rsize)])

  newQt.currblock = merged_currblock

  newindsL = makenewindsL(1,newQt,Qt,Rsize,0)
  newindsR = makenewindsR(2,newQt,Qt,Rsize,length(newQt.currblock[1]))

  newQt.ind = [(newindsL[q],newindsR[q]) for q = 1:length(newindsL)]

  newQt.QnumMat = newQnumMat
  newQt.QnumSum = newQnumSum

  return newQt
end
export mergereshape!

function mergereshape(Qt::Qtens{W,Q};currblock::currblockTypes=equalblocks(Qt)) where {W <: Number, Q <: Qnum, P <: Integer}
  cQt = copy(Qt)
  return mergereshape!(cQt,currblock=currblock)
end
export mergereshape

#  import ..tensor.unreshape!
"""
    unreshape!(Qt,a...)

In-place, unambiguous unreshaping for Qtensors.  Works identically to reshape for dense tensors

See also: [`reshape!`](@ref)
"""
function unreshape!(Qt::densTensType,sizes::W...) where {W <: Integer}
  return reshape!(Qt,sizes...)
end

function unreshape!(Qt::densTensType,sizes::Array{W,1}) where {W <: Integer}
  return reshape!(Qt,sizes...)
end

function unreshape!(Qt::qarray,sizes::W...) where W <: Integer
  return unreshape(Qt)
end

function unreshape!(Qt::qarray,sizes::Array{W,1}) where W <: Integer
  return unreshape(Qt)
end

function unreshape!(Qt::qarray)
  return unreshape(Qt)
end

#  import ..tensor.unreshape
"""
    unreshape(Qt,a...)

Unambiguous unreshaping for Qtensors.  Works identically to reshape for dense tensors
"""
function unreshape(Qt::qarray)
  Qt.size = [[i] for i = 1:length(Qt.QnumMat)]
  return Qt
end

function unreshape(Qt::densTensType,sizes::Array{W,1}) where W <: Integer
  return reshape(Qt,sizes...)
end

function unreshape(Qt::densTensType,sizes::W...) where W <: Integer
  return reshape(Qt,sizes...)
end

function unreshape(Qt::qarray,sizes::W...) where W <: Integer
  return unreshape!(copy!(Qt),sizes...)
end

function unreshape(Qt::qarray,sizes::Array{W,1}) where W <: Integer
  return unreshape!(copy!(Qt),sizes...)
end


#get the last index of a Qtensor (ex: A[:,1:end]...defines "end")
import Base.lastindex
"""
    lastindex(Qtens,i)

get the last index of a Qtensor (ex: A[:,1:end]...defines "end")
"""
function lastindex(Qtens::qarray, i::Integer)
  return Qtens.size[i]
end

@inline function genQNs(conjvar::Bool,A::Qtens{W,Q},Aind::intType) where {W <: Number, Q <: Qnum}
  LQNs = Array{Q,1}(undef,length(A.Qblocksum))
  if conjvar
    @inbounds @simd for q = 1:length(A.Qblocksum)
      LQNs[q] = inv(A.Qblocksum[q][Aind])
    end
  else
    @inbounds @simd for q = 1:length(A.Qblocksum)
      LQNs[q] = A.Qblocksum[q][Aind]
    end
  end
  return LQNs
end

function matchblocks(conjvar::NTuple{2,Bool},A::Qtens{W,Q},B::Qtens{S,Q};
                      ind::NTuple{2,P}=(2,1),matchQN::Q=Q()) where {W <: Number, S <: Number, Q <: Qnum, P <: Integer}

  Aind,Bind = ind
  LQNs = genQNs(conjvar[1],A,Aind)
  RQNs = genQNs(conjvar[2],B,Bind)

  return matchblocks(LQNs,RQNs,matchQN=matchQN)
end

function matchblocks(LQNs::Array{Q,1},RQNs::Array{Q,1};matchQN::Q=Q()) where Q <: Qnum
  Aorder = Array{intType,1}(undef,length(LQNs))
  outmatches = 0
  @inbounds for q = 1:length(LQNs)
    matchBool = false
    w = 0
    @inbounds while w < length(RQNs) && !matchBool
      w += 1
      thisQN = LQNs[q] + RQNs[w]
      matchBool = thisQN == matchQN
    end
    if matchBool
      Aorder[q] = w
      outmatches += 1
    else
      Aorder[q] = 0
    end
  end

  outAorder = Array{NTuple{2,intType},1}(undef,outmatches)
  counter = 0
  k = 0
  @inbounds while counter < outmatches
    k += 1
    if Aorder[k] != 0
      counter += 1
      loadtup!(outAorder,counter,k,Aorder[k])
    end
  end
  return outAorder
end

@inline function loadtup!(Aorder::Array{NTuple{2,intType},1},q::intType,k::intType,newblock::intType)
  Aorder[q] = (k,newblock)
  nothing
end

function findextrablocks(B::Qtens{W,Q},commoninds::Array{NTuple{2,intType},1}) where {W <: Number, Q <: Qnum}
  nB = length(B.T)-length(commoninds)
  Bleftover = Array{intType,1}(undef,nB)

  counter = 0
  z = 0
  while z < length(B.T) && counter < nB
    z += 1
    notfound = true
    q = 0
    @inbounds while notfound && q < length(commoninds)
      q += 1
      notfound = z != commoninds[q][2]
    end
    if notfound
      counter += 1
      Bleftover[counter] = z
    end
  end
  return Bleftover
end



function tensorcombination!(M::Qtens{W,Q}...;alpha::NTuple{N,W}=ntuple(i->eltype(M[1])(1),length(M)),fct::Function=*) where {Q <: Qnum, W <: Number, N}
  if length(M) == 1
    A = tensorcombination!(M[1],alpha=(alpha[1],))
  else
    A = tensorcombination!(M[1],M[2],alpha=alpha)
    nterms = min(length(M),length(alpha))
    @inbounds for i = 3:nterms
      A = tensorcombination!(A,M[i],alpha=(W(1),alpha[i]),fct=fct)
    end
  end
  return A
end

function tensorcombination!(M::Qtens{W,Q};alpha::NTuple{N,W}=(W(1),),fct::Function=*) where {Q <: Qnum, W <: Number, N}
  if !isapprox(alpha[1],W(1))
    @inbounds for q = 1:length(M.T)
      thisM = M.T[q]
      for y = 1:size(thisM,2)
        @inbounds @simd for x = 1:size(thisM,1)
          thisM[x,y] = fct(thisM[x,y],alpha[1])
        end
      end
      M.T[q] = thisM
    end
  end
  return M
end

function tensorcombination!(A::Qtens{W,Q},QtensB::Qtens{W,Q};alpha::NTuple{N,W}=(W(1),W(1)),fct::Function=*) where {Q <: Qnum, W <: Number, N}
  B = changeblock(QtensB,A.currblock)
  commoninds = matchblocks((false,false),A,B,ind=(1,2),matchQN=A.flux)

  @inbounds for q = 1:length(commoninds)
    Aqind = commoninds[q][1]
    Bqind = commoninds[q][2]
    tensorcombination!(A.T[Aqind],B.T[Bqind],alpha=alpha)
  end

  Bleftover = findextrablocks(B,commoninds)
  mult = alpha[2]
  if length(Bleftover) > 0
    AQblocks = length(A.T)
    newT = Array{Array{W,2},1}(undef,AQblocks+length(Bleftover))
    newind = Array{NTuple{2,Array{intType,2}},1}(undef,length(newT))
    newQblocksum = Array{NTuple{2,Q},1}(undef,length(newT))
    @inbounds for q = 1:AQblocks
      newT[q] = A.T[q]
      newind[q] = A.ind[q]
      newQblocksum[q] = A.Qblocksum[q]
    end
    @inbounds for q = 1:length(Bleftover)
      addq = Bleftover[q]
      if isapprox(mult,1)
        Bout = B.T[addq]
      else
        Bout = tensorcombination(B.T[addq],alpha=(mult,))
      end
      c = q+AQblocks
      newT[c] = Bout
      newind[c] = B.ind[addq]
      newQblocksum[c] = B.Qblocksum[addq]
    end
    A.T = newT
    A.ind = newind
    A.Qblocksum = newQblocksum
  end

  return A
end

function tensorcombination(M::Qtens{W,Q}...;alpha::NTuple{N,W}=ntuple(i->eltype(M[1])(1),length(M)),fct::Function=*) where {Q <: Qnum, W <: Number, N}
  A = copy(M[1])
  newtup = (A,Base.tail(M)...)
  return tensorcombination!(newtup...,alpha=alpha,fct=fct)
end

function tensorcombination(alpha::NTuple{N,W},M::Qtens{W,Q}...;fct::Function=*) where {Q <: Qnum, W <: Number, N}
  return tensorcombination(M...,alpha=alpha,fct=fct)
end

#  import .tensor.invmat!
"""
    invmat!(Qt[,zero=])

Creates inverse of a diagonal matrix in place (dense matrices are copied anyway);
if value is below `zero`, the inverse is set to zero

See also: [`invmat`](@ref)
"""
function invmat!(A::qarray)
  @inbounds for q = 1:length(A.T)
    invmat!(A.T[q])
  end
  return A
end
export invmat!

#decomposition of unitaries, so can do block by block
function exp!(C::Qtens{W,Q},prefactor::Number) where {W <: Number, Q <: Qnum}
  if W == typeof(prefactor)
    A = changeblock(C,C.currblock[1],C.currblock[2])
    @inbounds for q = 1:length(A.T)
      A.T[q] = exp(A.T[q],prefactor)
    end
    B = A
  else
    B = exp(C,prefactor)
  end
  return B
end

function exp!(C::Qtens{W,Q}) where {W <: Number, Q <: Qnum}
  return exp!(C,W(1))
end

import Base.exp
function exp(C::Qtens{W,Q},prefactor::Number) where {W <: Number, Q <: Qnum}
  A = changeblock(C,C.currblock)

  newtype = typeof(W(1)*prefactor)
  newT = Array{W,1}(undef,length(A.T))
  @inbounds @simd for q = 1:length(A.T)
    newT[q] = exp(A.T[q],prefactor)
  end

  newsize = Array{intType,1}(undef,length(A.size))
  @inbounds @simd for w = 1:length(newsize)
    newsize[w] = A.size[w] #could be copy(A.size[w]) but this is never changing
  end
  newcurrblock = A.currblock
  newind = A.ind
  newQblocksum = A.Qblocksum
  newMat = A.QnumMat
  newSum = A.QnumSum
  newflux = copy(A.flux)

  return Qtens{newtype,Q}(newsize,newT,newind,newcurrblock,newQblocksum,newMat,newSum,newflux)
end

function exp(C::Qtens{W,Q}) where {W <: Number, Q <: Qnum}
  return exp(C,W(1))
end

"""
  metricdistance(A[,power=])

computes the Forebenius norm of all elements in the tensor...equal to L^power norm
"""
function metricdistance(D::Qtens{W,Q};power::Number=1,fct::Function=identity) where {W <: Number, Q <: Qnum}
  powersums = W(0)
  for q = 1:length(D.T)
    for y = 1:size(D.T[q],2)
      @inbounds @simd for x = 1:size(D.T[q],1)
        powersums += fct(D.T[q][x,y])^power
      end
    end
  end
  return powersums^(1/power)
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
  return real(metricdistance(QtensA,power=2,fct=abs))
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

#  import .tensor.elnumtype
function elnumtype(A::Qtens{W,Q}) where {W <: Number, Q <: Qnum}
  return eltype(A)
end

import LinearAlgebra.conj
"""
    conj(A)

conjugates a Qtensor by creating a copy

See also: [`conj!`](@ref)
"""
function conj(currQtens::qarray)
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
function conj!(currQtens::Qtens{W,Q}) where {W <: Number, Q <: Qnum}
  @inbounds for q = 1:length(currQtens.T)
    currQtens.T[q] = conj!(currQtens.T[q])
  end

  newQnumSum = Array{Array{Q,1},1}(undef,length(currQtens.QnumSum))
  @inbounds for w = 1:length(currQtens.QnumSum)
    newQnumSum[w] = inv.(currQtens.QnumSum[w])
  end
  currQtens.QnumSum = newQnumSum

  newQblocksum = Array{NTuple{2,Q},1}(undef,length(currQtens.Qblocksum))
  @inbounds for q = 1:length(newQblocksum)
    newQblocksum[q] = (inv(currQtens.Qblocksum[q][1]),inv(currQtens.Qblocksum[q][2]))
  end
  currQtens.Qblocksum = newQblocksum

  currQtens.flux = inv(currQtens.flux)
  return currQtens
end

function adjoint(Qt::qarray)
  return conj!(permutedims(Qt,[2,1]))
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
function size(A::qarray, i::intType)::intType
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
  count = 0

  for i = 1:size(vec, 1)
    @inbounds @simd for j = 1:length(Rsize[vec[i]])
      count += 1
      order[count] = Rsize[vec[i]][j]
    end
  end

  permorder = Array{intType,1}(undef,length(order))

  newRsize = Array{Array{intType,1},1}(undef,length(vec))
  @inbounds for q = 1:length(vec)
    newRsize[q] = Rsize[vec[q]]
  end
  counting = 0
  for k = 1:length(Rsize)
    @inbounds @simd for m = 1:length(Rsize[k])
      counting += 1
      permorder[order[counting]] = counting
    end
  end

  counter = 0
  for w = 1:length(newRsize)
    @inbounds @simd for a = 1:length(newRsize[w])
      counter += 1
      newRsize[w][a] = counter
    end
  end

  currQtens.size = newRsize
  currQtens.QnumMat = currQtens.QnumMat[[order...]]
  currQtens.QnumSum = currQtens.QnumSum[[order...]]

  theseblocks = (currQtens.currblock[1],currQtens.currblock[2])
  currQtens.currblock = (permorder[theseblocks[1]],permorder[theseblocks[2]])

  return currQtens
end

import Base.isapprox
"""
    isapprox(A,B)

Checks if `A` is approximately `B`
"""
function isapprox(A::Qtens{Z,Q},B::Qtens{W,Q})::Bool where {Z <: Number, W <: Number, Q <: Qnum}
  test = length(A.T) == length(B.T)
  return test ? false : isapprox(norm(A-B),0)
end

function evaluate_keep(C::qarray,q::Integer,Linds::Array{P,1},ap::Array{K,1},rowcol::Integer) where {P <: Integer, K <: genColType}
  thisindmat = C.ind[q][rowcol]
  keeprows = Array{Bool,1}(undef,size(thisindmat,2))
  rowindexes = size(thisindmat,1)
  @inbounds for x = 1:size(thisindmat,2)
    condition = true
    index = 0
    @inbounds while condition && index < rowindexes
      index += 1
      condition = thisindmat[index,x] in ap[Linds[index]]
    end
    keeprows[x] = condition
  end
  return keeprows
end

@inline function truncate_replace_inds(C::qarray,q::Integer,rowcol::Integer,Lkeepinds::Array{P,1},
                                keepbool::Array{Bool,1},kept_unitranges::Array{K,1},keeprows::Array{Bool,1}) where {P <: Integer, K <: genColType}

  thisindmat = C.ind[q][rowcol][keepbool,keeprows]
  offset = (rowcol-1)*length(C.currblock[1])

  @inbounds for a = 1:size(thisindmat,1)
    theseranges = kept_unitranges[Lkeepinds[a]]
    @inbounds for x = 1:size(thisindmat,2)
      thisval = thisindmat[a,x]

      newval = findfirst(w -> w == thisval,theseranges)[1]
      thisindmat[a,x] = newval-1
    end
  end
  return thisindmat
end

function get_ranges(sizes::NTuple{G,intType},a::genColType...) where G
  unitranges = Array{genColType,1}(undef,length(a))
  @inbounds for i = 1:length(a)
    if typeof(a[i]) <: Colon
      unitranges[i] = 0:sizes[i]-1#[w for w = 1:sizes[i]]
    elseif typeof(a[i]) <: Array{intType,1} || typeof(a[i]) <: Tuple{intType,Vararg{intType}}
      unitranges[i] = [a[i][w]-1 for w = 1:length(a[i])]
    elseif typeof(a[i]) <: UnitRange{intType}
      unitranges[i] = UnitRange(a[i][1]-1,a[i][end]-1)
    elseif typeof(a[i]) <: StepRange{intType}
      unitranges[i] = UnitRange(a[i][1]-1,a[i][end]-1)
    elseif typeof(a[i]) <: Integer
      unitranges[i] = a[i]-1
    end
  end
  return unitranges
end

function isinteger(a::genColType...)
  isinteger = true
  w = 0
  @inbounds while isinteger && w < length(a)
    w += 1
    isinteger = typeof(a[w]) <: Integer
  end
  return isinteger
end
export isinteger


function innerjoinloop(C::Qtens{W,Q},Linds::Array{P,1},unitranges::Array{G,1},Rinds::Array{P,1},
                        keep_one::Array{B,1},keep_two::Array{B,1},Lkeepinds::Array{P,1},Rkeepinds::Array{P,1}) where {W <: Number, Q <: Qnum, P <: Integer, B <: Bool, G <: genColType}

  keepers = [false for i = 1:length(C.T)]
  loadT = Array{Array{W,2},1}(undef,length(C.T))
  loadind_one = Array{Array{intType,2},1}(undef,length(C.T))
  loadind_two = Array{Array{intType,2},1}(undef,length(C.T))

  @inbounds for q = 1:length(C.T)

    keeprows = evaluate_keep(C,q,Linds,unitranges,1)

    if sum(keeprows) > 0
      keepcols = evaluate_keep(C,q,Rinds,unitranges,2)
      if sum(keepcols) > 0

        keepers[q] = true

        loadT[q] = C.T[q][keeprows,keepcols]

        loadind_one[q] = truncate_replace_inds(C,q,1,Lkeepinds,keep_one,unitranges,keeprows)
        loadind_two[q] = truncate_replace_inds(C,q,2,Rkeepinds,keep_two,unitranges,keepcols)
      end
    end

  end
  return keepers,loadT,loadind_one,loadind_two
end


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
function getindex(C::qarray, a::genColType...)
  return getindex!(C, a...)
end

function getindex!(A::Qtens{W,Q}, a::genColType...) where {Q <: Qnum, W <: Number}
  if isinteger(a...)
    return searchindex(A,a...)
  end

  condition = true
  for p = 1:length(a)
    condition = condition && (typeof(a[p]) <: Colon)
    condition = condition && (typeof(a[p]) <: UnitRange && length(a[p]) == size(A,p))
  end
  if condition
    return A
  end

  isjoinedindices = 0
  @inbounds for w = 1:length(A.size)
    isjoinedindices += length(A.size[w]) > 1
  end

  if isjoinedindices > 0
    C = mergereshape(A)
  else
    C = A
  end
  
  unitranges = get_ranges(size(C),a...)

  keepinds = Array{Bool,1}(undef,length(unitranges))
  @inbounds @simd for w = 1:length(keepinds)
    keepinds[w] = !(typeof(unitranges) <: Integer)
  end

  newdim = sum(keepinds)
  newQnumMat = Array{Array{intType,1},1}(undef,newdim)
  newQnumSum = Array{Array{Q,1},1}(undef,newdim)
  newsize = Array{intType,1}(undef,newdim)
  counter = 0
  @inbounds for i = 1:length(keepinds)
    if keepinds[i]
      counter += 1
      newQnumMat[counter] = C.QnumMat[i][a[i]]
      newQnumSum[counter] = C.QnumSum[i]
      newsize[counter] = length(unitranges[i])
    end
  end
  tup_newsize = Array{Array{intType,1},1}(undef,length(newQnumMat))
  @inbounds for i = 1:length(newQnumMat)
    tup_newsize[i] = [i]
  end

  if length(keepinds) != sum(keepinds)
    newflux = -C.flux
    @inbounds for k = 1:length(keepinds)
      if !keepinds[k]
        newflux += getQnum(k,a[k],C)
      end
    end
    newflux = -newflux
  else
    newflux = C.flux
  end
  

  Linds = C.currblock[1]
  keep_one = keepinds[Linds]
  Lkeepinds = Linds[keep_one]

  Rinds = C.currblock[2]
  keep_two = keepinds[Rinds]
  Rkeepinds = Rinds[keep_two]

  keepers,loadT,loadind_one,loadind_two = innerjoinloop(C,Linds,unitranges,Rinds,keep_one,keep_two,Lkeepinds,Rkeepinds)

  keptindices = vcat(A.size[keepinds]...)
  convertInds = Array{intType,1}(undef,length(A.QnumMat))
  count = 0
  @inbounds @simd for i = 1:length(keptindices)
    count += 1
    convertInds[keptindices[i]] = count
  end

  newcurrblock = (Lkeepinds,Rkeepinds)

  for w = 1:2
    @inbounds @simd for r = 1:length(newcurrblock[w])
      g = newcurrblock[w][r]
      newcurrblock[w][r] = convertInds[g]
    end
  end

  newT = loadT[keepers]

  nkeeps = sum(keepers)
  newinds = Array{NTuple{2,Array{intType,2}},1}(undef,nkeeps)

  newQsum = Array{NTuple{2,Q},1}(undef,nkeeps)
  counter = 0
  @inbounds for q = 1:length(loadind_one)
    if keepers[q]
      counter += 1
      newinds[counter] = (loadind_one[q],loadind_two[q])


      startQNs = Array{Q,1}(undef,2) # [Q(),Q()]
      for w = 1:2
        thisQN = Q()
        @inbounds for a = 1:length(newcurrblock[w])
          index = newcurrblock[w][a]
          dimval = newinds[counter][w][a,1] + 1
          thisQN += getQnum(index,dimval,newQnumMat,newQnumSum)
        end
        startQNs[w] = thisQN
      end
      newQsum[counter] = (startQNs...,)    
    end
  end

  return Qtens{W,Q}(tup_newsize, newT, newinds, newcurrblock, newQsum, newQnumMat, newQnumSum, newflux)
end
export getindex!


function findmatch(Lpos::Array{P,1},A::Qtens{W,Q},C::Qtens{W,Q},Aqind::Integer,LR::Integer) where {P <: Integer, W <: Number, Q <: Qnum}
  found = false
  rowindex = 0
  while !found && rowindex < size(A.ind[Aqind][LR],2)
    rowindex += 1
    matchinginds = true
    g = 0
    @inbounds while matchinginds && g < length(C.currblock[LR])
      g += 1
      matchinginds = A.ind[Aqind][LR][g,rowindex] == Lpos[g]
    end
    found = matchinginds
  end
  return found,rowindex
end


function loadpos!(Lpos::Array{P,1},C::Qtens{W,Q},Cqind::Integer,LR::Integer,x::Integer,unitranges::Array{B,1}) where {B <: genColType, P <: Integer, W <: Number, Q <: Qnum}
  @inbounds @simd for w = 1:length(Lpos)
    index = C.currblock[LR][w]
    xpos = C.ind[Cqind][LR][w,x] + 1
    Lpos[w] = unitranges[index][xpos]
  end
  nothing
end




import Base.setindex!
function setindex!(A::Qtens{W,Q},B::Qtens{W,Q},vals::genColType...) where {W <: Number, Q <: Qnum}
  C = changeblock(B,A.currblock)
  Asize = size(A)
  unitranges = get_denseranges(Asize,vals...)

  commoninds = matchblocks((false,false),A,C,ind=(2,1),matchQN=A.flux)

  Lpos = Array{intType,1}(undef,length(C.currblock[1]))
  Rpos = Array{intType,1}(undef,length(C.currblock[2]))

  valvec = [0]

  @inbounds for q = 1:length(commoninds)
    Aqind = commoninds[q][1]
    Cqind = commoninds[q][2]
    for y = 1:size(C.ind[Cqind][2],2)
      loadpos!(Rpos,C,Cqind,2,y,unitranges)
      found2,colindex = findmatch(Rpos,A,C,Aqind,2)
      if found2
        @inbounds for x = 1:size(C.ind[Cqind][1],2)
          loadpos!(Lpos,C,Cqind,1,x,unitranges)
          found,rowindex = findmatch(Lpos,A,C,Aqind,1)
          if found
            num = C.T[Cqind][x,y]
            A.T[Aqind][rowindex,colindex] = num
          end
        end
      end

    end
  end
  nothing
end

function setindex!(A::Array{W,N},B::Qtens{W,Q},vals::genColType...) where {W <: Number, Q <: Qnum, N}
  Asize = size(A)
  unitranges = get_denseranges(Asize,vals...)

  Lpos = Array{intType,1}(undef,length(B.currblock[1]))
  Rpos = Array{intType,1}(undef,length(B.currblock[2]))
  totalpos = Array{intType,1}(undef,ndims(A))

  @inbounds for q = 1:length(B.T)
    @inbounds for y = 1:size(B.ind[q][2],2)
      loadpos!(Rpos,B,q,2,y,unitranges)
      @inbounds @simd for w = 1:length(Rpos)
        totalpos[B.currblock[2][w]] = Rpos[w]
      end
      @inbounds for x = 1:size(B.ind[q][1],2)
        loadpos!(Lpos,B,q,1,x,unitranges)
        num = B.T[q][x,y]
        @inbounds @simd for w = 1:length(Lpos)
          totalpos[B.currblock[1][w]] = Lpos[w]
        end
        A[totalpos...] = num
      end
    end
  end
  nothing
end

function setindex!(A::tens{W},B::Qtens{W,Q},vals::genColType...) where {W <: Number, Q <: Qnum}
  Asize = size(A)
  unitranges = get_denseranges(Asize,vals...)

  Lpos = Array{intType,1}(undef,length(B.currblock[1]))
  Rpos = Array{intType,1}(undef,length(B.currblock[2]))
  totalpos = Array{intType,1}(undef,ndims(A))

  @inbounds for q = 1:length(B.T)
    @inbounds for y = 1:size(B.ind[q][2],2)
      loadpos!(Rpos,B,q,2,y,unitranges)
      @inbounds @simd for w = 1:length(Rpos)
        totalpos[B.currblock[2][w]] = Rpos[w]
      end
      @inbounds for x = 1:size(B.ind[q][1],2)
        loadpos!(Lpos,B,q,1,x,unitranges)
        num = B.T[q][x,y]
        @inbounds @simd for w = 1:length(Lpos)
          totalpos[B.currblock[1][w]] = Lpos[w]
        end
        z = pos2ind(totalpos,Asize)
        A.T[z] = num
      end
    end
  end
  nothing
end


function setindex!(C::Qtens{W,Q},val::W,a::intType...) where {W <: Number, Q <: Qnum}
  if length(C.T) > 0
    q = findqsector(C,a)

    x = scaninds(1,q,C,a)
    y = scaninds(2,q,C,a)

    @inbounds C.T[q][x,y] = val
  end
  nothing
end








function findqsector(C::qarray,a::NTuple{N,intType}) where N

  LR = length(C.currblock[1]) < length(C.currblock[2]) ? 1 : 2

  smallinds = C.currblock[LR]
  if length(smallinds) == 0
    targetQN = C.flux
  else
    targetQN = C.flux
    @inbounds @simd for i = 1:length(smallinds)
      y = smallinds[i]
      targetQN += getQnum(y,a[y],C)
    end
  end

  notmatchingQNs = true
  q = 0
  @inbounds while q < length(C.T) && notmatchingQNs
    q += 1
    currQN = C.Qblocksum[q][LR]
    notmatchingQNs = targetQN != currQN
  end
  return q
end


function scaninds(blockindex::intType,q::intType,C::qarray,a::NTuple{N,intType}) where N
  x = 0
  notmatchingrow = q > 0
  @inbounds while notmatchingrow && x < size(C.ind[q][blockindex],2)
    x += 1
    r = 0
    matchvals = true
    @inbounds while matchvals && r < length(C.currblock[blockindex])
      r += 1
      matchvals = C.ind[q][blockindex][r,x] + 1 == a[C.currblock[blockindex][r]]
    end
    notmatchingrow = !matchvals
  end
  return x
end



#  import ..tensor.searchindex
"""
    searchindex(C,a...)

Find element of `C` that corresponds to positions `a`
"""
function searchindex(C::Qtens{W,Q},a::intType...) where {Q <: Qnum, W <: Number}
  q = findqsector(C,a)
  x = scaninds(1,q,C,a)
  y = scaninds(2,q,C,a)
  
  outnum = q == 0 ? W(0) : C.T[q][x,y]
  return outnum
end




function getblockrows(A::qarray,Aqind::Integer,leftAblock::Array{P,1}) where P <: Integer
  Arows = Array{intType,1}(undef,size(A.ind[Aqind][1],2))
  if size(A.ind[Aqind][1],1) > 0
    @inbounds for i = 1:size(A.ind[Aqind][1],2)
      Arows[i] = A.ind[Aqind][1][1,i]
      @inbounds @simd for j = 2:size(A.ind[Aqind][1],1)
        Arows[i] *= leftAblock[j-1]
        Arows[i] += A.ind[Aqind][1][j,i]
      end
    end
  end
  return Arows
end



@inline function AAind2zeropos(vec::Array{X,1},S::NTuple{P,X}) where X <: Integer where P
  currpos = Array{X,2}(undef,P,length(vec))
  @inbounds for k = 1:length(vec)
    currpos[1,k] = vec[k]-1
    @inbounds @simd for j = 1:P-1
      val = currpos[j,k]
      currpos[j+1,k] = fld(val,S[j])
      currpos[j,k] = val % S[j]
    end
    currpos[P,k] %= S[end]
  end
  return currpos
end


function joinloop!(A::Qtens{W,Q},B::Qtens{R,Q},commonblocks::Array{NTuple{2,intType},1},origAsize::intType...) where {W <: Number, R <: Number, Q <: Qnum, G}
  tup_Aleftsize = ntuple(w->A.currblock[1][w],length(A.currblock[1]))
  leftAblock = Array{intType,1}(undef,length(A.currblock[1])-1)
  @inbounds @simd for i = 1:length(leftAblock)
    leftAblock[i] = length(A.QnumMat[A.currblock[1][i]])
  end
  leftBblock = Array{intType,1}(undef,length(B.currblock[1])-1)
  @inbounds @simd for i = 1:length(leftBblock)
    leftBblock[i] = length(B.QnumMat[B.currblock[1][i]]) 
  end

  @inbounds for q = 1:length(commonblocks)
    Aqind = commonblocks[q][1]
    Bqind = commonblocks[q][2]


    Arows = getblockrows(A,Aqind,leftAblock)
    Brows = getblockrows(B,Bqind,leftBblock)

    if Arows != Brows 

      newrow = Array{intType,1}(undef,length(Arows)+length(Brows)) #union(Arows,Brows)
      @inbounds for w = 1:length(Arows)
        newrow[w] = Arows[w]
      end
      @inbounds for w = 1:length(Brows)
        newrow[w] = Brows[w]
      end

      Anewrows = Array{intType,1}(undef,length(Arows))
      @inbounds for w = 1:length(Arows)
        Anewrows[w] = findfirst(w->newrow[w]==Arows[i],1:length(newrow))
      end
      Bnewrows = Array{intType,1}(undef,length(Brows))
      @inbounds for w = 1:length(Brows)
        Bnewrows[w] = findfirst(w->newrow[w]==Brows[i],1:length(newrow))
      end

      Ttype = typeof(W(1)*R(1))
      newcolsize = size(A.T[Aqind],2) + size(B.T[Bqind],2)
      if size(A.T[Aqind],1) + size(B.T[Bqind],1) == length(newrow)
        newT = Array{Ttype,2}(undef,length(newrow),newcolsize)
      else
        newT = zeros(Ttype,length(newrow),newcolsize)
      end

      newT[Anewrows,1:size(A.T[Aqind],2)] = A.T[Aqind]
      newT[Bnewrows,size(A.T[Bqind],2)+1:end] = B.T[Bqind]

      A.T[Aqind] = newT
      leftinds = AAind2zeropos(newrow,tup_Aleftsize)
    else
      A.T[Aqind] = joinindex!([2],A.T[Aqind],B.T[Bqind])
      leftinds = A.ind[Aqind][1]
    end

    Ainds = A.ind[Aqind][2]
    Binds = B.ind[Bqind][2]

    Asize = size(Ainds,2)
    Bsize = size(Binds,2)

    newlength = Asize + Bsize
    newind = Array{intType,2}(undef,length(origAsize),newlength)
    for g = 1:Asize 
      @inbounds @simd for r = 1:length(origAsize)
        newind[r,g] = Ainds[r,g]
      end
    end
    for g = 1:Bsize
      modg = Asize + g
      @inbounds @simd for r = 1:length(origAsize)
        newind[r,modg] = Binds[r,g] + origAsize[r]
      end
    end
    A.ind[Aqind] = (leftinds,newind)
  end
  nothing
end


function firstloop!(w::Integer,origAsize::Union{Array{P,1},NTuple{G,P}},A::Qtens{W,Q},index::Integer,newQnums::Array{intType,1},newQnumSum::Array{Q,1}) where {W <: Number, Q <: Qnum, P <: Integer, G}
  @inbounds for j = 1:origAsize[w] #better serial on small systems (for sequential memory access?)
    thisQN = getQnum(index,j,A)
    notmatchQN = true
    b = 0
    @inbounds while b < length(newQnumSum) && notmatchQN
      b += 1
      notmatchQN = thisQN != newQnumSum[b]
    end
    newQnums[j] = b
  end
  nothing
end


function matchloop(g::Integer,B::Qtens{W,Q},index::Integer,deltaflux::Q,newQnums::Array{intType,1},newQnumSum::Array{Q,1}) where {W <: Number, Q <: Qnum}
  @inbounds for j = 1:size(B,index)
    g += 1
    thisQN = getQnum(index,j,B) + deltaflux
    notmatchQN = true
    b = 0
    @inbounds while b < length(newQnumSum) && notmatchQN
      b += 1
      notmatchQN = thisQN != newQnumSum[b]
    end
    newQnums[g] = b
  end
  nothing
end

function Bextraloop!(inds::Array{intType,1},A::Qtens{W,Q},B::Qtens{R,Q},Bleftover::Array{intType,1},
                      newT::Array{Array{P,2},1},newindexlist::Array{NTuple{2,Array{intType,2}},1},
                      inputsize::Tuple,newQblocksum::Array{NTuple{2,Q},1}) where {W <: Number, R <: Number, Q <: Qnum, P <: Number}
  @inbounds for q = 1:length(Bleftover)
    addq = Bleftover[q]
    thisind = q + length(A.T)
    newT[thisind] = B.T[addq]
    newindexlist[thisind] = B.ind[addq]
    for i = 1:2
      @inbounds for a = 1:length(A.currblock[i])
        index = A.currblock[i][a]
        if index in inds
          @inbounds @simd for j = 1:size(newindexlist[thisind][i],2)
            newindexlist[thisind][i][a,j] += inputsize[index]
          end
        end
      end
    end
    newQblocksum[thisind] = B.Qblocksum[addq]
  end
  nothing
end

function makerowcol(Lposes::Array{P,1},Lsizes::Tuple,A::Qtens{W,Q},q::Integer,LR::Integer) where {P <: Integer, W <: Number, Q <: Qnum}
  @inbounds rows = Array{intType,1}(undef,size(A.ind[q][LR],2))
  for x = 1:length(rows)
    @inbounds @simd for i = 1:size(A.ind[q][LR],1)
      Lposes[i] = A.ind[q][LR][i,x]
    end
    pos2ind!(rows,x,Lposes,Lsizes)
  end
  return rows
end

function rowcolsort(rows::Array{P,1},A::Qtens{W,Q},LR::Integer) where {P <: Integer, W <: Number, Q <: Qnum}
  rowsort = issorted(rows)
  newroworder = rowsort ? sortperm(rows) : [0] #1:size(A.ind[q][LR],2)
  return rowsort,newroworder
end

function orderloop!(A::Qtens{W,Q},Lsizes::Tuple,Rsizes::Tuple,Lposes::Array{P,1},Rposes::Array{P,1}) where {W <: Number, Q <: Qnum, P <: Integer}
  @inbounds for q = 1:length(A.ind)
    rows = makerowcol(Lposes,Lsizes,A,q,1)
    cols = makerowcol(Rposes,Rsizes,A,q,2)

    rowsort,newroworder = rowcolsort(rows,A,1)
    colsort,newcolorder = rowcolsort(cols,A,2)
    if rowsort && colsort
      A.T[q] = A.T[q][newroworder,newcolorder]
      loadM!(A.ind[q][1],A.ind[q][1][:,newroworder])
      loadM!(A.ind[q][2],A.ind[q][2][:,newcolorder])
    elseif rowsort
      A.T[q] = A.T[q][newroworder,:]
      loadM!(A.ind[q][1],A.ind[q][1][:,newroworder])
    else
      A.T[q] = A.T[q][:,newcolorder]
      loadM!(A.ind[q][2],A.ind[q][2][:,newcolorder])
    end
  end
  nothing
end

function joinindex(bareinds::intvecType,QtensA::Qtens{R,Q},QtensB::Qtens{S,Q};ordered::Bool=false) where {R <: Number, S <: Number, Q <: Qnum}
  return joinindex!(bareinds,copy(QtensA),QtensB,ordered=ordered)
end
export joinindex



function joinindex!(bareinds::intvecType,QtensA::Qtens{R,Q},QtensB::Qtens{S,Q};ordered::Bool=false) where {R <: Number, S <: Number, Q <: Qnum}
  preinds = convIn(bareinds)
  inds = Array{intType,1}(undef,length(preinds))
  @inbounds @simd for i = 1:length(inds)
    inds[i] = preinds[i]
  end

  inputsize = size(QtensA)

  notcommoninds = findnotcons(length(QtensA.QnumMat),preinds)

  A = changeblock(QtensA,notcommoninds,inds)
  B = changeblock(QtensB,notcommoninds,inds)

  origAsize = Array{intType,1}(undef,length(inds))
  @inbounds @simd for w = 1:length(inds)
    origAsize[w] = length(A.QnumMat[inds[w]])
  end
  commonblocks = matchblocks((false,false),A,B,ind=(2,1),matchQN=A.flux)

  Bleftover = findextrablocks(B,commonblocks)

  joinloop!(A,B,commonblocks,origAsize...)

  Ttype = typeof(eltype(QtensA.T[1])(0)*eltype(QtensB.T[1])(0))
  newT = Array{Array{Ttype,2},1}(undef,length(A.T)+length(Bleftover))
  newindexlist = Array{NTuple{2,Array{intType,2}},1}(undef,length(newT))
  newQblocksum = Array{NTuple{2,Q},1}(undef,length(newT))


  newT[1:length(A.T)] = A.T
  newindexlist[1:length(A.T)] = A.ind
  newQblocksum[1:length(A.T)] = A.Qblocksum

  Bextraloop!(inds,A,B,Bleftover,newT,newindexlist,inputsize,newQblocksum)

  A.T = newT
  A.ind = newindexlist
  A.Qblocksum = newQblocksum

  zeroQN = Q()

  deltaflux = A.flux - B.flux
  @inbounds for w = 1:length(inds)
    index = inds[w]

    thisvec = Array{Q,1}(undef,length(A.QnumSum[index])+length(B.QnumSum[index]))
    @inbounds @simd for w = 1:length(A.QnumSum[index])
      thisvec[w] = A.QnumSum[index][w]
    end
    counter = length(A.QnumSum[index])
    if deltaflux == zeroQN || w > 1
      @inbounds @simd for w = 1:length(B.QnumSum[index])
        counter += 1
        thisvec[counter] = B.QnumSum[index][w] + deltaflux
      end
    else
      @inbounds @simd for w = 1:length(B.QnumSum[index])
        counter += 1
        thisvec[counter] = B.QnumSum[index][w]
      end
    end
    newQnumSum = unique!(thisvec)

    newQnums = Array{intType,1}(undef,origAsize[w] + size(B,index))
    firstloop!(w,origAsize,A,index,newQnums,newQnumSum)
    
    g = origAsize[w]
    matchloop(g,B,index,deltaflux,newQnums,newQnumSum)

    A.QnumMat[index] = newQnums
    A.QnumSum[index] = newQnumSum
  end

  if ordered
    Asize = size(A)
    Lsizes = Asize[notcommoninds]
    Rsizes = Asize[inds]

    Lposes = Array{intType,1}(undef,length(Lsizes))
    Rposes = Array{intType,1}(undef,length(Rsizes))

    orderloop!(A,Lsizes,Rsizes,Lposes,Rposes)
  end

  return A
end

"""
    applylocalF!(tens, i)

(in-place) effectively apply a chain of F (fermion phase) operator by a single manipulation on the specified index.

contributed by A. Foley

See also: [`applylocalF`](@ref)
"""
function applylocalF!(M::R, i) where {R <: qarray}
  for (j, (t, index)) in enumerate(zip(M.T, M.ind))
    pos = ind2pos(index, size(M))
    p = parity(getQnum(i,pos[i],tens))
    M.T[j] *= (-1)^p
  end
end

"""
    applylocalF(tens, i)

effectively apply a chain of F (fermion phase) operator by a single manipulation on the specified index.

contributed by A. Foley

See also: [`applylocalF!`](@ref)
"""
function applylocalF(tens::R, i::Integer) where {R <: qarray}
  W = copy(tens)
  applylocalF!(W, i)
  return W 
end
export applylocalF,applylocalF!

function getinds_loop_one(con::Array{intType,1},Rsize::Array{Array{intType,1},1},vec::Union{NTuple{N,intType},Array{intType,1}}) where N
  counter = 0
  for j = 1:size(vec, 1)
    @inbounds @simd for p in Rsize[vec[j]]
      counter += 1
      con[counter] = p
    end
  end
  return counter
end

function getinds_loop_two(notcon::Array{intType,1},Rsize::Array{Array{intType,1},1},vec::Union{NTuple{N,intType},Array{intType,1}}) where N
  altcounter = 0
  for j = 1:length(Rsize)
    condition = true
    k = 0
    @inbounds while k < size(vec,1) && condition
      k += 1
      condition = !(j == vec[k])
    end
    if condition
      @inbounds @simd for p in Rsize[j]
        altcounter += 1
        notcon[altcounter] = p
      end
    end
  end
  nothing
end
"""
    getinds(A,iA)

Sub-function for quantum number contraction.  A Qtensor `A` with indices to contract `iA` generates all contracted indices (if, for example, a joined index was called by a single index number), and also the un-contracted indices
"""
function getinds(currQtens::qarray, vec::Union{Array{intType,1},NTuple{N,intType}}) where N
  Rsize = currQtens.size
  consize = 0
  @inbounds @simd for w = 1:length(vec)
    consize += length(Rsize[w])
  end
  con = Array{intType,1}(undef,consize)  

  getinds_loop_one(con,Rsize,vec)
  
  notcon = Array{intType,1}(undef,length(currQtens.QnumMat)-consize)

  getinds_loop_two(notcon,Rsize,vec)

  return con, notcon
end
export getinds

"""
    Idhelper(A,iA)

generates the size of matrix equivalent of an identity matrix from tensor `A` with indices `iA`

#Output:
+`lsize::Int64`: size of matrix-equivalent of identity operator
+`finalsizes::Int64`: size of identity operator

See also: [`makeId`](@ref) [`trace`](@ref)
"""
function Idhelper(A::TensType,iA::Array{NTuple{2,P},1}) where P <: Integer
  lsize = prod(w->size(A,iA[w][1]),1:length(iA))
  leftsizes = ntuple(w->size(A,iA[w][1]),length(iA))
  rightsizes = ntuple(w->size(A,iA[w][2]),length(iA))
  finalsizes = (leftsizes...,rightsizes...)
  return lsize,finalsizes
end

#  import ..tensor.makeId
"""
    makeId(A,iA)

generates an identity matrix from tensor `A` with indices `iA`

See also: [`trace`](@ref)
"""
function makeId(A::Qtens{W,Q},iA::Array{NTuple{2,P},1}) where {W <: Number, Q <: Qnum, P <: Integer}
  lsize,finalsizes = Idhelper(A,iA)
  newQnumMat = A.QnumMat[iA]
  typeA = eltype(A)
  Id = rand(A,blockfct=makeIdarray)
  return Id
end

"""
    swapgate(A,iA,B,iB)

generates a swap gate (order of indices: in index for `A`, in index for `B`, out index for `A`, out index for `B`) for `A` and `B`'s indices `iA` and `iB`
"""
function (swapgate(A::TensType,iA::W,B::TensType,iB::R) where {W <: Union{intvecType,Array{Array{P,1},1}},R <: Union{intvecType,Array{Array{P,1},1}}}) where P <: Integer
  LId = makeId(A,iA)
  RId = makeId(B,iB)
  if typeof(LId) <: qarray
    push!(LId.size,[ndims(LId)+1])
  else
    LId = reshape(LId,size(LId)...,1)
  end
  if typeof(RId) <: qarray
    push!(RId.size,[ndims(RId)+1])
  else
    RId = reshape(RId,size(RId)...,1)
  end
  fullId = contract(LId,4,RId,4)
  return permute(fullId,[1,3,2,4])
end

#  import .tensor.makedens
"""
    makedens(Qt)

converts Qtensor (`Qt`) to dense array
"""
function makedens(QtensA::Qtens{W,Q}) where {W <: Number, Q <: Qnum}
  truesize = basesize(QtensA)
  Lsizes = truesize[QtensA.currblock[1]]
  Rsizes = truesize[QtensA.currblock[2]]
  Ldim = prod(Lsizes)
  Rdim = prod(Rsizes)
  
  G = zeros(W,basesize(QtensA))

  newpos = Array{intType,1}(undef,length(QtensA.QnumMat))  
  @inbounds for q = 1:length(QtensA.ind)
    thisTens = QtensA.T[q]
    theseinds = QtensA.ind[q]
    for x = 1:size(QtensA.T[q],1)
      @inbounds @simd for m = 1:length(QtensA.currblock[1])
        bb = QtensA.currblock[1][m]
        newpos[bb] = theseinds[1][m,x] + 1
      end
      @inbounds for y = 1:size(QtensA.T[q],2)
        @inbounds for n = 1:length(QtensA.currblock[2])
          rr = QtensA.currblock[2][n]
          newpos[rr] = theseinds[2][n,y] + 1
        end
        G[newpos...] = thisTens[x,y]
      end
    end
  end
  return tens{W}(G)
end

@inline function makedens(Qt::Array)
  return tens(Qt)
end

#  import .tensor.makeArray
"""
  makeArray(Qt)

See: [`makedens`](@ref)
"""
function makeArray(QtensA::Qtens{W,Q}) where {W <: Number, Q <: Qnum}
  return makeArray(makedens(QtensA))
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
      maxshow = min(show, length(Qtens.T[q]))
      maxBool = show < length(Qtens.T[q])
      println("block $q size = ",size(Qtens.T[q]),", ",Qtens.Qblocksum[q],", values = ",Qtens.T[q][1:maxshow], maxBool ? "..." : "")
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

"""
    checkflux(Qt[,silent=])

Debug tool: checks all non-zero elements obey flux conditions in Qtensor (`Qt`); print element by element with `silent`
"""
function checkflux(Qt::Qtens{W,Q},;silent::Bool = true) where {W <: Number, Q <: Qnum}
  condition = true

  if length(Qt.T) == 0
    println("WARNING: zero elements detected in tensor")
  end

  #meta-data checked first
  Rsize = Qt.size
  checksizes = [prod(x->length(Qt.QnumMat[x]),Rsize[w]) for w = 1:length(Rsize)]

  subcondition = sum(w->size(Qt,w) - checksizes[w],1:length(Rsize)) != 0
  condition = condition && !subcondition
  if subcondition
    println("size field does not match QnumMat")
  end

  firstnorm = norm(Qt)
  secondnorm = norm(makeArray(Qt))
  subcondition = !isapprox(firstnorm,secondnorm)
  condition = condition && !subcondition
  if subcondition
    println(firstnorm," ",secondnorm)
    error("ill-defined position (.ind) fields...did not return detectably same tensor on dense conversion")
  end

  subcondition = length(Qt.currblock[1]) + length(Qt.currblock[2]) != sum(w->length(Qt.size[w]),1:length(Qt.size))
  condition = condition && !subcondition
  if subcondition
    println("currblock is not correct for sizes")
  end


  numQNs = length(Qt.T)
  LQNs = Array{Q,1}(undef,numQNs)
  RQNs = Array{Q,1}(undef,numQNs)
  matchingQNs = Array{Bool,1}(undef,numQNs)

  for q = 1:numQNs
    LQNs[q] = Q()
    for w = 1:length(Qt.currblock[1])
      thispos = Qt.currblock[1][w]
      thisdim = Qt.ind[q][1][w,1] + 1
      LQNs[q] += getQnum(thispos,thisdim,Qt)
    end

    RQNs[q] = Q()
    for w = 1:length(Qt.currblock[2])
      thispos = Qt.currblock[2][w]
      thisdim = Qt.ind[q][2][w,1] + 1
      RQNs[q] += getQnum(thispos,thisdim,Qt)
    end
    matchingQNs[q] = LQNs[q] + RQNs[q] == Qt.flux
  end

  subcondition = sum(matchingQNs) != numQNs && numQNs > 0
  condition = condition && !subcondition
  if subcondition
    println("not matching quantum numbers...probably issue in defininig (.ind) field in Qtensor")
  end

  subcondition = !(sort(LQNs) == sort([Qt.Qblocksum[q][1] for q = 1:length(Qt.Qblocksum)]))
  condition = condition && !subcondition
  if subcondition
    println(LQNs)
    println([Qt.Qblocksum[q][1] for q = 1:length(Qt.Qblocksum)])
    println("error in left QN block definitions")
  end

  subcondition = !(sort(RQNs) == sort([Qt.Qblocksum[q][2] for q = 1:length(Qt.Qblocksum)]))
  condition = condition && !subcondition
  if subcondition
    println(RQNs)
    println([Qt.Qblocksum[q][2] for q = 1:length(Qt.Qblocksum)])
    println("error in right QN block definitions")
  end


  totalLcheck = Array{Bool,1}(undef,numQNs)
  totalRcheck = Array{Bool,1}(undef,numQNs)
  for q = 1:numQNs
    Lcheck = [true for w = 1:size(Qt.ind[q][1],2)]
    for w = 1:size(Qt.ind[q][1],2)
      checkLQN = Q()
      for x = 1:size(Qt.ind[q][1],1)
        thisrow = Qt.currblock[1][x]
        thisdim = Qt.ind[q][1][x,w]+1
        checkLQN += getQnum(thisrow,thisdim,Qt)
      end
      Lcheck[w] = checkLQN == LQNs[q]
    end
    totalLcheck[q] = sum(Lcheck) == size(Qt.ind[q][1],2)



    Rcheck = [true for w = 1:size(Qt.ind[q][2],2)]
    for w = 1:size(Qt.ind[q][2],2)
      checkRQN = Q()
      for x = 1:size(Qt.ind[q][2],1)
        thisrow = Qt.currblock[2][x]
        thisdim = Qt.ind[q][2][x,w]+1
        checkRQN += getQnum(thisrow,thisdim,Qt)
      end
      Rcheck[w] = checkRQN == RQNs[q]
    end
    totalRcheck[q] = sum(Rcheck) == size(Qt.ind[q][2],2)
  end

  subcondition = sum(totalLcheck) != numQNs
  condition = condition && !subcondition
  if subcondition
    println("wrong quantum number on some rows; quantum numbers: ",totalLcheck)
  end

  subcondition = sum(totalRcheck) != numQNs
  condition = condition && !subcondition
  if subcondition
    println("wrong quantum number on some columns; quantum numbers: ",totalRcheck)
  end



  for q = 1:numQNs
    subcondition = sum(isnan.(Qt.T[q])) > 0
    condition = condition && !subcondition
    if subcondition
      println("element of q = ",q," is not a number")
    end
  end



  if condition
    println("PASSED \n")
  else
    error("problems \n")
  end
  nothing
end
export checkflux

function checkflux(Qt::densTensType;silent::Bool = true)
  nothing
end



#end
