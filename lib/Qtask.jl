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
    Module: Qtask

Functions that are applied to Qtensors that are used in contractions, decompositions, etc.

See also: [`Qtensor`](@ref) [`contractions`](@ref) [`decompositions`](@ref)
"""
#=
module Qtask
using ..tensor
using ..Qtensor
using ..QN
=#
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



  function evaluate_keep(C::qarray,q::Integer,Linds::Array{P,1},ap::Array{Array{P,1},1},rowcol::Integer) where P <: Integer
    thisindmat = C.ind[q][rowcol]
    keeprows = Array{Bool,1}(undef,size(thisindmat,2))
    rowindexes = size(thisindmat,1)
    for x = 1:size(thisindmat,2)
      condition = true
      index = 0
      while condition && index < rowindexes
        index += 1
        @inbounds condition = thisindmat[index,x] in ap[Linds[index]]
      end
      @inbounds keeprows[x] = condition
    end
    return keeprows
  end

  function truncate_replace_inds(C::qarray,q::Integer,rowcol::Integer,Lkeepinds::Array{P,1},
                                 keepbool::Array{Bool,1},kept_unitranges::Array{Array{P,1},1},keeprows::Array{Bool,1}) where P <: Integer
    thisindmat = C.ind[q][rowcol][keepbool,keeprows]
    offset = (rowcol-1)*length(C.currblock[1])

    for x = 1:size(thisindmat,2)
      for a = 1:size(thisindmat,1)
        @inbounds thisval = thisindmat[a,x]

        @inbounds newval = findfirst(w -> w == thisval,kept_unitranges[Lkeepinds[a]])[1]
        @inbounds thisindmat[a,x] = newval-1
      end
    end
    return thisindmat
  end



  function get_ranges(sizes::NTuple{G,P},a::genColType...) where {G, P <: Integer}
    unitranges = Array{Array{intType,1},1}(undef,length(a))
    keepinds = Array{Bool,1}(undef,length(a))
    for i = 1:length(a)
      if typeof(a[i]) <: Colon
        @inbounds unitranges[i] = [w-1 for w = 1:sizes[i]]
        @inbounds keepinds[i] = true
      elseif typeof(a[i]) <: Array{intType,1} || typeof(a[i]) <: Array{intType,1} <: Tuple
        @inbounds unitranges[i] = a[i] .- 1
        @inbounds keepinds[i] = true
      elseif typeof(a[i]) <: UnitRange{intType}
        @inbounds unitranges[i] = [w-1 for w = a[i]]
        @inbounds keepinds[i] = true
      elseif typeof(a[i]) <: StepRange{intType}
        @inbounds unitranges[i] = [w-1 for w = a[i]]
        @inbounds keepinds[i] = true
      elseif typeof(a[i]) <: Integer
        @inbounds unitranges[i] = [a[i]-1]
        @inbounds keepinds[i] = false
      end
    end
    return unitranges,keepinds
  end





  function isinteger(a::genColType...)
    isinteger = true
    w = 0
    while isinteger && w < length(a)
      w += 1
      isinteger = typeof(a[w]) <: Integer
    end
    return isinteger
  end
  export isinteger
  


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
    condition = true
    for p = 1:length(a)
      condition = condition && (typeof(a[p]) <: Colon)
      condition = condition && (typeof(a[p]) <: UnitRange && length(a[p]) == size(A,p))
    end
    if condition
      return A
    end

    if isinteger(a...)
      return searchindex(A,a...)
    end


    isjoinedindices = sum(w->length(A.size[w]) > 1,1:length(A.size))

    if isjoinedindices > 0
      C = mergereshape(A)
    else
      C = A
    end

#    println(a)
#    println(C)

    unitranges,keepinds = get_ranges(size(C),a...)

    newdim = sum(keepinds)
    newQnumMat = Array{Array{intType,1},1}(undef,newdim)
    newQnumSum = Array{Array{Q,1},1}(undef,newdim)
    newsize = Array{intType,1}(undef,newdim)
    counter = 0
    for i = 1:length(keepinds)
      if keepinds[i]
        counter += 1
        newQnumMat[counter] = C.QnumMat[i][a[i]]
        newQnumSum[counter] = C.QnumSum[i]
        newsize[counter] = length(unitranges[i])
      end
    end
    tup_newsize = [[i] for i = 1:length(newQnumMat)]

    newflux = inv(C.flux)
    for k = 1:length(keepinds)
      if !keepinds[k]
        add!(newflux,getQnum(k,a[k],C.QnumMat,C.QnumSum))
      end
    end


#    println(unitranges)
#    println(keepinds)

#    println(C.currblock)

    Linds = C.currblock[1]
    keep_one = keepinds[Linds]
    Lkeepinds = Linds[keep_one]

    Rinds = C.currblock[2]
    keep_two = keepinds[Rinds]
    Rkeepinds = Rinds[keep_two]

    newcurrblock = [Lkeepinds,Rkeepinds]

    keepers = [false for i = 1:length(C.T)]
    loadT = Array{tens{W},1}(undef,length(C.T))
    loadind_one = Array{Array{intType,2},1}(undef,length(C.T))
    loadind_two = Array{Array{intType,2},1}(undef,length(C.T))

    Threads.@threads for q = 1:length(C.T)

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

    keptindices = vcat(A.size[keepinds]...)
    convertInds = Array{intType,1}(undef,length(A.QnumMat))
    count = 0
    for i = 1:length(keptindices)
      count += 1
      convertInds[keptindices[i]] = count
    end

    for w = 1:2
      for r = 1:length(newcurrblock[w])
        g = newcurrblock[w][r]
        newcurrblock[w][r] = convertInds[g]
      end
    end

    newQsum = C.Qblocksum[keepers]
    newT = loadT[keepers]

    nkeeps = sum(keepers)
    newinds = Array{Array{Array{intType,2},1},1}(undef,nkeeps)
    counter = 0
    for q = 1:length(loadind_one)
      if keepers[q]
        counter += 1
        newinds[counter] = [loadind_one[q],loadind_two[q]]
      end
    end

    out = Qtens{W,Q}(tup_newsize, newT, newinds, newcurrblock, newQsum, newQnumMat, newQnumSum, newflux)

    return out
  end
  export getindex!


  function findmatch(Lpos::Array{P,1},A::Qtens{W,Q},C::Qtens{W,Q},Aqind::Integer,LR::Integer) where {P <: Integer, W <: Number, Q <: Qnum}
    found = false
    rowindex = 0
    while !found
      rowindex += 1
      matchinginds = true
      g = 0
      while matchinginds && g < length(C.currblock[LR])
        g += 1
        @inbounds matchinginds = A.ind[Aqind][LR][g,rowindex] == Lpos[g]
      end
      found = matchinginds
    end
    return found,rowindex
  end


  function loadpos!(Lpos::Array{P,1},C::Qtens{W,Q},Cqind::Integer,LR::Integer,x::Integer,unitranges::Array{Array{B,1},1}) where {B <: Integer, P <: Integer, W <: Number, Q <: Qnum}
    @simd for w = 1:length(Lpos)
      @inbounds index = C.currblock[LR][w]
      @inbounds xpos = C.ind[Cqind][LR][w,x] + 1
      @inbounds Lpos[w] = unitranges[index][xpos]
    end
    nothing
  end

  


  import Base.setindex!
  function setindex!(A::Qtens{W,Q},B::Qtens{W,Q},vals::genColType...) where {W <: Number, Q <: Qnum}
    C = changeblock(B,A.currblock)
    unitranges,keepinds = get_ranges(size(A),vals...)

    commoninds = matchblocks((false,false),A,C,matchQN=A.flux)

    Lpos = Array{intType,1}(undef,length(C.currblock[1]))
    Rpos = Array{intType,1}(undef,length(C.currblock[2]))

    valvec = [0]

    for q = 1:length(commoninds)
      Aqind = commoninds[q][1]
      Cqind = commoninds[q][2]
      for x = 1:size(C.ind[Cqind][1],2)
        loadpos!(Lpos,C,Cqind,1,x,unitranges)
        found,rowindex = findmatch(Lpos,A,C,Aqind,1)
        if found
          for y = 1:size(C.ind[Cqind][2],2)
            loadpos!(Rpos,C,Cqind,2,y,unitranges)
            found2,colindex = findmatch(Rpos,A,C,Aqind,2)
            if found2
              num = getSingleVal!(C.T[Cqind],x,y)
              A.T[Aqind][rowindex,colindex] = num
            end
          end
        end

      end
    end
    nothing
  end


  function setindex!(C::Qtens{W,Q},val::W,a::Integer...) where {W <: Number, Q <: Qnum}
    if length(C.T) > 0
      q = findqsector(C,a...)

      x = scaninds(1,q,C,a...)
      y = scaninds(2,q,C,a...)

      C.T[q][x,y] = val
    end
    nothing
  end








  function findqsector(C::qarray,a::Integer...)
    LR = length(C.currblock[1]) < length(C.currblock[2]) ? 1 : 2

    smallinds = C.currblock[LR]
    if length(smallinds) == 0
      targetQN = C.flux
    else
      x = smallinds[1]
      targetQN = C.flux + getQnum(x,a[x],C)
      for i = 2:length(smallinds)
        y = smallinds[i]
        add!(targetQN,getQnum(y,a[y],C))
      end
    end

    notmatchingQNs = true
    q = 0
    while q < length(C.T) && notmatchingQNs
      q += 1
      currQN = C.Qblocksum[q][LR]
      notmatchingQNs = targetQN != currQN
    end
    return q
  end


  function scaninds(blockindex::Integer,q::Integer,C::qarray,a::Integer...)
    La = a[C.currblock[blockindex]]
    notmatchingrow = true
    x = 0

    while notmatchingrow
      x += 1
      r = 0
      matchvals = true
      while matchvals && r < length(La)
        r += 1
        @inbounds matchvals = C.ind[q][blockindex][r,x] + 1 == La[r]
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
  function searchindex(C::Qtens{W,Q},a::Integer...) where {Q <: Qnum, W <: Number}
    if length(C.T) == 0
      outnum = 0
    else
      q = findqsector(C,a...)

      x = scaninds(1,q,C,a...)
      y = scaninds(2,q,C,a...)

      outnum = C.T[q][x,y]
    end

    return outnum
  end

  function searchindex(C::AbstractArray,a::Integer...)
    return C[a...]
  end




#import ..tensor.joinindex
#import ..tensor.joinindex!
  function joinindex!(bareinds::intvecType,QtensA::Qtens{R,Q},QtensB::Qtens{S,Q};ordered::Bool=false) where {R <: Number, S <: Number, Q <: Qnum}
    inds = convIn(bareinds)

    inds = [inds[i] for i = 1:length(inds)]

    inputsize = size(QtensA)#.size


    

    Ttype = typeof(eltype(QtensA.T[1])(0)*eltype(QtensB.T[1])(0))
    Tout = Array{Ttype,1}(undef,length(QtensA.T)+length(QtensB.T))

    notcommoninds = setdiff(1:length(QtensA.QnumMat),inds)

    undo_bool = QtensA.currblock != [notcommoninds,inds]
    origcurrblock = [copy(QtensA.currblock[1]),copy(QtensA.currblock[2])]

    taskA = Array{qarray,1}(undef,2)
    #=Threads.@threads=# for w = 1:2
      if w == 1
        taskA[w] = changeblock(QtensA,notcommoninds,inds)
      else
        taskA[w] = changeblock(QtensB,notcommoninds,inds)
      end
    end

    A = taskA[1]
    B = taskA[2]

    lengthnewinds = length(inds)

    origAsize = [length(A.QnumMat[inds[w]]) for w = 1:lengthnewinds]
#    A.size = ntuple(w->QtensA.size[w] + (w in inds ? QtensB.size[w] : 0),length(QtensA.size))
    commonblocks = matchblocks((false,false),A,B,matchQN=A.flux)

    Bcommon = [commonblocks[q][2] for q = 1:length(commonblocks)]
    Bleftover = setdiff(1:length(B.T),Bcommon)

    for q = 1:length(commonblocks)
      Aqind = commonblocks[q][1]
      Bqind = commonblocks[q][2]
      A.T[Aqind] = joinindex!([2],A.T[Aqind],B.T[Bqind])

      Ainds = A.ind[Aqind][2]
      Binds = B.ind[Bqind][2]


      Asize = size(Ainds,2)
      Bsize = size(Binds,2)

      newlength = Asize + Bsize
      newind = Array{intType,2}(undef,lengthnewinds,newlength)
      for g = 1:Asize
        @simd for r = 1:lengthnewinds
          @inbounds newind[r,g] = Ainds[r,g]
        end
      end
      for g = 1:Bsize
        modg = Asize + g
        @simd for r = 1:lengthnewinds
          @inbounds newind[r,modg] = Binds[r,g] + origAsize[r]
        end
      end
      A.ind[Aqind][2] = newind
    end

    newT = Array{tens{Ttype},1}(undef,length(A.T)+length(Bleftover))
    newind = Array{Array{Array{intType,2},1},1}(undef,length(newT))
    newQblocksum = Array{Array{Q,1},1}(undef,length(newT))

    for q = 1:length(A.T)
      newT[q] = A.T[q]
      newind[q] = A.ind[q]
      newQblocksum[q] = A.Qblocksum[q]
    end

    #=Threads.@threads=# for q = 1:length(Bleftover)
      addq = Bleftover[q]
      thisind = q + length(A.T)
      newT[thisind] = B.T[addq]
      newind[thisind] = B.ind[addq]
      for i = 1:2
        for a = 1:length(A.currblock[i])
          index = A.currblock[i][a]
          if index in inds
            @simd for j = 1:length(newind[thisind][i])
              @inbounds newind[thisind][i][a,j] += inputsize[index]
            end
          end
        end
      end
      newQblocksum[thisind] = B.Qblocksum[addq]
    end
    A.T = newT
    A.ind = newind
    A.Qblocksum = newQblocksum


    numthreads = Threads.nthreads()
    parallelQNs = [Q() for i = 1:numthreads]

    deltaflux = A.flux + inv(B.flux)
    for w = 1:length(inds)
      index = inds[w]




      Bsums = [B.QnumSum[index][i] + deltaflux for i = 1:length(B.QnumSum[index])]
      newQnumSum = unique(vcat(A.QnumSum[index],Bsums))

      newQnums = Array{intType,1}(undef,origAsize[w] + size(B,index))
#      let Aqn = Aqn, Bqn = Bqn, newQnums = newQnums, deltaflux = deltaflux
      #=Threads.@threads=#  for j = 1:origAsize[w] #better serial on small systems (for sequential memory access?)
        thisthread = Threads.threadid()
        thisQN = parallelQNs[thisthread]
        copy!(thisQN,getQnum(index,j,A))
        notmatchQN = true
        b = 0
        while b < length(newQnumSum) && notmatchQN
          b += 1
          notmatchQN = thisQN != newQnumSum[b]
        end
        newQnums[j] = b
      end
      #=Threads.@threads=#  for j = 1:size(B,index)
        thisthread = Threads.threadid()
        thisQN = parallelQNs[thisthread]
        copy!(thisQN,getQnum(index,j,B))
        add!(thisQN,deltaflux)
        notmatchQN = true
        b = 0
        while b < length(newQnumSum) && notmatchQN
          b += 1
          notmatchQN = thisQN != newQnumSum[b]
        end
        newQnums[origAsize[w]+j] = b
      end

      A.QnumMat[index] = newQnums
      A.QnumSum[index] = newQnumSum

    end



    if ordered
      Lsizes = size(A)[notcommoninds]
      Rsizes = size(A)[inds]

      numthreads = Threads.nthreads()
      Lposes = [Array{intType,1}(undef,length(Lsizes)) for w = 1:numthreads]
      Rposes = [Array{intType,1}(undef,length(Rsizes)) for w = 1:numthreads]

      #=Threads.@threads=# for q = 1:length(A.ind)
        rows = Array{intType,1}(undef,size(A.ind[q][1],2))
        thisthread = Threads.threadid()
        for x = 1:length(rows)
          @simd for i = 1:size(A.ind[q][1],1)
            @inbounds Lposes[thisthread][i] = A.ind[q][1][i,x]
          end
          pos2ind!(rows,x,Lposes,thisthread,Lsizes)
        end

        columns = Array{intType,1}(undef,size(A.ind[q][2],2))
        for y = 1:length(columns)
          @simd for i = 1:size(A.ind[q][2],1)
            @inbounds Rposes[thisthread][i] = A.ind[q][2][i,y]
          end
          pos2ind!(columns,y,Rposes,thisthread,Rsizes)
        end

        rowsort = issorted(rows)
        colsort = issorted(columns)
        if rowsort || colsort
          if rowsort
            newroworder = sortperm(rows)
            @inbounds A.ind[q][1] = A.ind[q][1][:,newroworder]
          else
            @inbounds newroworder = 1:size(A.ind[q][1],2)
          end
          if colsort
            newcolorder = sortperm(columns)
            @inbounds A.ind[q][2] = A.ind[q][2][:,newcolorder]
          else
            @inbounds newcolorder = 1:size(A.ind[q][2],2)
          end
          A.T[q] = A.T[q][newroworder,newcolorder]
          A.ind[q][1] = A.ind[q][1][:,newroworder]
          A.ind[q][2] = A.ind[q][2][:,newcolorder]
        end
      end
    end

#    checkflux(A)

    return A
  end

  """
      applylocalF!(tens, i)

  (in-place) effectively apply a chain of F (fermion phase) operator by a single manipulation on the specified index.

  contributed by A. Foley
  
  See also: [`applylocalF`](@ref)
  """
  function applylocalF!(tens::R, i) where {R <: qarray}
    for (j, (t, index)) in enumerate(zip(tens.T, tens.ind))
      pos = ind2pos(index, tens.size)
      p = parity(getQnum(i,pos[i],tens))
      tens.T[j] *= (-1)^p
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

  """
      getinds(A,iA)

  Sub-function for quantum number contraction.  A Qtensor `A` with indices to contract `iA` generates all contracted indices (if, for example, a joined index was called by a single index number), and also the un-contracted indices
  """
  function getinds(currQtens::qarray, vec::Union{Array{P,1},Tuple}) where P <: Integer
    Rsize = currQtens.size
    consize = sum(a->length(Rsize[a]),vec)
    con = Array{intType,1}(undef,consize)  
    notcon = Array{intType,1}(undef,length(currQtens.QnumMat)-consize)
    counter,altcounter = 0,0


    for j = 1:size(vec, 1)
      @simd for p in Rsize[vec[j]]
        counter += 1
        con[counter] = p
      end
    end

#    println(length(Rsize)," ",Rsize," ",[length(currQtens.QnumMat[w]) for w = 1:length(currQtens.QnumMat)])
#    println(length(notcon))
    for j = 1:length(Rsize)
      condition = true
      k = 0
      while k < size(vec,1) && condition
        k += 1
        condition = !(j == vec[k])
      end
      if condition
        @simd for p in Rsize[j]
          altcounter += 1
          notcon[altcounter] = p
        end
      end
    end

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
  function (Idhelper(A::TensType,iA::W) where W <: Union{intvecType,Array{Array{P,1},1}}) where P <: Integer
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

#  import ..tensor.makeId
  """
      makeId(A,iA)

  generates an identity matrix from tensor `A` with indices `iA`

  See also: [`trace`](@ref)
  """
  function (makeId(A::Qtens{W,Q},iA::X) where X <: Union{intvecType,Array{Array{P,1},1}}) where {W <: Number, Q <: Qnum, P <: Integer}
    lsize,finalsizes = Idhelper(A,iA)
    newQnumMat = A.QnumMat[iA]
    typeA = eltype(A)
    Id = Qtens{W,Q}(newQnumMat,Type=typeA)

    error("error here....needs to make identity for each block")
    Id.ind = intType[i+lsize*(i-1) for i = 1:lsize]
    Id.T = ones(typeA,length(Id.ind))

    Id.size = finalsizes
    return Id
  end

  """
      swapgate(A,iA,B,iB)
  
  generates a swap gate (order of indices: in index for `A`, in index for `B`, out index for `A`, out index for `B`) for `A` and `B`'s indices `iA` and `iB`
  """
  function (swapgate(A::TensType,iA::W,B::TensType,iB::R) where {W <: Union{intvecType,Array{Array{P,1},1}},R <: Union{intvecType,Array{Array{P,1},1}}}) where P <: Integer
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


#  import .tensor.makedens
  """
      makedens(Qt)

  converts Qtensor (`Qt`) to dense array
  """
  function makedens(QtensA::Qtens{W,Q})::denstens where {W <: Number, Q <: Qnum}

    truesize = basesize(QtensA)
    Lsizes = truesize[QtensA.currblock[1]]
    Rsizes = truesize[QtensA.currblock[2]]
    Ldim = prod(Lsizes)
    Rdim = prod(Rsizes)
    
    G = zeros(W,basesize(QtensA)...)

    numthreads = Threads.nthreads()
    newpos = [Array{intType,1}(undef,length(QtensA.QnumMat)) for n = 1:numthreads]
    
    for q = 1:length(QtensA.ind)
      thisTens = makeArray(QtensA.T[q])
      theseinds = QtensA.ind[q]
      for y = 1:size(QtensA.T[q],2)
        thisthread = Threads.threadid()
        for n = 1:length(QtensA.currblock[2])
          rr = QtensA.currblock[2][n]
          inout = theseinds[2][n,y] + 1
          newpos[thisthread][rr] = inout
        end
        for x = 1:size(QtensA.T[q],1)

          for m = 1:length(QtensA.currblock[1])
            bb = QtensA.currblock[1][m]
            newpos[thisthread][bb] = theseinds[1][m,x] + 1
          end

          G[newpos[thisthread]...] = thisTens[x,y]
        end
      end
    end
    return tens{W}(G)
  end

  function makedens(Qt::AbstractArray)
    return Qt
  end

#  import .tensor.makeArray
  """
    makeArray(Qt)

  See: [`makedens`](@ref)
  """
  function makeArray(QtensA::Qtens{W,Q}) where {W <: Number, Q <: Qnum}
    return makeArray(reshape(makedens(QtensA).T,size(QtensA)))
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
        add!(LQNs[q],getQnum(thispos,thisdim,Qt))
      end

      RQNs[q] = Q()
      for w = 1:length(Qt.currblock[2])
        thispos = Qt.currblock[2][w]
        thisdim = Qt.ind[q][2][w,1] + 1
        add!(RQNs[q],getQnum(thispos,thisdim,Qt))
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
#      println(LQNs)
#      println([Qt.Qblocksum[q][1] for q = 1:length(Qt.Qblocksum)])
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
      Lcheck = Array{Bool,1}(undef,size(Qt.ind[q][1],2))
      for w = 1:size(Qt.ind[q][1],2)
        checkLQN = Q()
        for x = 1:size(Qt.ind[q][1],1)
          thisrow = Qt.currblock[1][x]
          thisdim = Qt.ind[q][1][x,w]+1
          add!(checkLQN,getQnum(thisrow,thisdim,Qt))
        end
        Lcheck[w] = checkLQN == LQNs[q]
      end
      totalLcheck[q] = sum(Lcheck) == size(Qt.ind[q][1],2)



      Rcheck = Array{Bool,1}(undef,size(Qt.ind[q][2],2))
      for w = 1:size(Qt.ind[q][2],2)
        checkRQN = Q()
        for x = 1:size(Qt.ind[q][2],1)
          thisrow = Qt.currblock[2][x]
          thisdim = Qt.ind[q][2][x,w]+1
          add!(checkRQN,getQnum(thisrow,thisdim,Qt))
#          println(x," ",checkLQN," ",LQNs[q]," ",Lcheck[x])
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
      subcondition = sum(isnan.(Qt.T[q].T)) > 0
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
  
    function checkflux(Qt::AbstractArray;silent::Bool = true)
      nothing
    end
  
    function checkflux(M::denstens)
      nothing
    end

#end