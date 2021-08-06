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

module tensornetwork
#using ..shuffle
using ..tensor
using ..QN
using ..Qtensor
#using ..Qtask
using ..contractions
using ..decompositions
using ..MPutil

  abstract type TNobj end
  export TNobj

  abstract type TNnetwork end
  export TNnetwork

  """
      nameT{W,B}

  named tensor with tensor of type `W` and type of names `B`

  # Fields:
  + `N::W`: Tensor stored
  + `names::Array{B,1}`: names of all indices
  """
#  + `arrows::Array{Bool,1}`: fluxes on all indices
  mutable struct nameT{W,B} <: TNobj where {W <: Union{qarray,AbstractArray,denstens}, B <: Union{Any,String}}
    N::W
    names::Array{B,1}
#    arrows::Array{Bool,1} #for conjugation...acts like an arrows but also applies to non-quantum number tensors
  end
#=
  """
      nameT(Qt,namez,arrowss)

  constructor for named tensor from a tensor `Qt`, vector of index names `namez`, and vector of fluxes `arrowss`
  """
  function nameT(Qt::T,namez::Array{B,1},arrows::Array{Bool,1})::TNobj where {T <: TensType, B <: Union{Any,String}}
    return nameT{T,B}(Qt,namez,arrows)
  end
  export nameT
=#
  """
      nameT(Qt,namez)

  constructor for named tensor from a tensor `Qt` and vector of index names `namez`
  """
  function nameT(Qt::T,namez::Array{B,1};regTens::Bool=false)::TNobj where {T <: TensType, B <: Union{Any,String}}
    newQt = !regTens && typeof(Qt) <: AbstractArray ? tens(Qt) : Qt
    return nameT{typeof(newQt),B}(newQt,namez)#,[true for i = 1:length(namez)])
  end
  export nameT

  """
      network{N,W}

  Generates a network of TNobjs that stores more than one named tensor

  # Fields:
  + `net::NTuple{N,W}`: A network of named tensors
  """
  mutable struct network{W} <: TNnetwork where W  <: TNobj
    net::Array{W,1}
  end

  """
      network(Qts)

  constructor to generates a network of TNobjs that stores a vector of named tensors `Qts`
  """
  function network(Qts::Array{W,1}) where W  <: TNobj
    return network{W}(Qts)
  end

  """
      network(Qts)

  converts named tensor to a network with a single tensor element
  """
  function network(Qts::nameT{W,S}...) where W  <: TNobj where S <: Union{Any,String}
    return network{W,S}([Qts[i] for i = 1:length(Qts)])
  end
  export network

  import ..Base.getindex
  function getindex(Qts::TNnetwork,i::Integer)
    return Qts.net[i]
  end

  import ..Base.setindex!
  function setindex!(Qts::TNnetwork,newTens::TNobj,i::Integer)
    return Qts.net[i] = newTens
  end

  import ..Base.length
  function length(Qts::TNnetwork)
    return length(Qts.net)
  end

  function contractinds(A::TNobj,B::TNobj)
    vecA = Int64[]
    vecB = Int64[]
    for a = 1:size(A.names,1)
      for b = 1:size(B.names,1)
        if A.names[a] == B.names[b] #&& (A.arrows[a] != B.arrows[b] || !arrows)
          push!(vecA,a)
          push!(vecB,b)
        end
      end
    end
    return vecA,vecB
  end

  import ..Qtensor.*
  """
      *(A,B...)

  Contracts `A` and any number of `B` along common indices; simple algorithm at present for the order
  """
  function *(A::TNobj,B::TNobj)#;arrows::Bool=false)
    vecA,vecB = contractinds(A,B)

    if sort(vecA) == [ndims(A)-i+1 for i = 1:length(vecA)]
      Aplace = "L"
    elseif sort(vecA) == [i for i = 1:length(vecA)]
      Aplace = "R"
    else
      Aplace = "LR"
    end

    if sort(vecB) == [ndims(B)-i+1 for i = 1:length(vecB)]
      Bplace = "L"
    elseif sort(vecB) == [i for i = 1:length(vecB)]
      Bplace = "R"
    else
      Bplace = "LR"
    end

    if Aplace == "L" || Aplace == "LR"
      if Bplace == "L"
        LR = prod(size(A)) > prod(size(B))
      else
        LR = true
      end
    elseif Aplace == "R"
      if Bplace == "R"
        LR = prod(size(A)) > prod(size(B))
      else
        LR = false
      end
    else #if Aplace == Bplace == "LR"
      LR = true
    end

    #println(vecA," ",vecB)

    retindsA = setdiff([i for i = 1:ndims(A)],vecA)
    retindsB = setdiff([i for i = 1:ndims(B)],vecB)

    newnames = setdiff(A.names,B.names) #Any[]
#    newc = Bool[]
    if LR
      newnames = vcat(A.names[retindsA],B.names[retindsB])
#      newc = vcat(A.arrows[retindsA],B.arrows[retindsB])
      newTens = contract(A.N,vecA,B.N,vecB)
    else
      newnames = vcat(B.names[retindsB],A.names[retindsA])
#      newc = vcat(B.arrows[retindsB],A.arrows[retindsA])
      newTens = contract(B.N,vecB,A.N,vecA)
    end

    return nameT(newTens,newnames)#,newc)
  end

  function *(R::TNobj...)#;arrows::Bool=false)
    out = *(R[1],R[2])#,arrows=arrows)
    @simd for b = 3:length(R)
      out = *(out,R[b])#,arrows=arrows)
    end
    return out
  end

  """
      *(a,b)

  concatenates string `a` with integer `b` after converting integer to a string
  """
  function *(a::String,b::Integer)
    return a*string(b)
  end

  import Base.permutedims
  """
      permtuedims(A,order)

  Permutes named tensor `A` according to `order` (ex: [[1,2],[3,4]] or [["a","b"],["c","d"]])

  See also: [`permutedims!`](@ref)
  """
  function permutedims(A::TNobj,order::Array{W,1}) where W <: Union{String,Integer}
    B = copy(A)
    return permutedims!(B,order)
  end

  #differentiate between case for integers (above but wrong code) and for labels
  #get "not labels" for set diff of the labels we know and don't know
  import ..Qtensor.permutedims!
  """
      permtuedims!(A,order)

  Permutes named tensor `A` according to `order` (ex: [[1,2],[3,4]] or [["a","b"],["c","d"]])

  See also: [`permutedims`](@ref)
  """
  function permutedims!(A::TNobj,order::Array{W,1}) where W <: String
    newnumberorder = Int64[]
    for i = 1:size(order,1)
      for j = 1:size(A.names,1)
        if order[i] == A.names[j]
          push!(newnumberorder,j)
          continue
        end
      end
    end
    return permutedims!(A,newnumberorder)
  end

  function permutedims!(A::TNobj,order::Array{W,1}) where W <: Integer
    A.N = permutedims!(A.N,order)
    A.names = A.names[order]
#    A.arrows = A.arrows[order]
    return A
  end

  """
      matchnames(AA,order,q)

  Matches `order` (a length 2 vector of vectors of strings for indices) to the indices in `AA` for the left (right) with `q`=1 (2)
  """
  function matchnames(AA::TNobj,order::Array{B,1}) where B <: Union{Any,String}
    vect = Array{intType,1}(undef,length(order))
    for a = 1:length(order)
      condition = true
      w = 0
      while condition && w < length(AA.names)
        w += 1
        if order[a] == AA.names[w]
          vect[a] = w
          condition = false
        end
      end
    end
    return vect
  end
#=
  """
      findinds(AA,order)

  prepares return indices and tensor `AA` for decomposition
  """
  function findinds(AA::TNobj,order::Array{Array{B,1},1}) where B <: Union{Any,String}
    left = matchnames(AA,order,1)
    right = matchnames(AA,order,2)
    return left,right
  end
=#
  import ..decompositions.svd
  """
      svd(A,order[,mag=,cutoff=,m=,name=,leftadd=,rightadd=])

  Generates SVD of named tensor `A` according to `order`; same output as regular SVD but with named tensors

  # Naming created index:
  + the index to the left of `D` is `vcat(name,leftadd)`
  """
  function svd(AA::TNobj,order::Array{Array{B,1},1};mag::Number = 0.,cutoff::Number = 0.,
                m::Integer = 0,power::Integer=2,name::String="svdind",leftadd::String="L",
                rightadd::String="R",nozeros::Bool=false) where B <: Union{Any,String}

    left = matchnames(AA,order[1])
    right = matchnames(AA,order[2])

    neworder = Array{intType,1}[left,right]
    leftname = name * leftadd
    rightname = name * rightadd

    U,D,V,truncerr,newmag = svd(AA.N,neworder,power=power,mag=mag,cutoff=cutoff,m=m,nozeros=nozeros)

    TNobjU = nameT(U,vcat(AA.names[left],[leftname]))#,vcat(AA.arrows[left],[true]))
    TNobjD = nameT(D,[leftname,rightname])#,[false,true])
    TNobjV = nameT(V,vcat([rightname],AA.names[right]))#,vcat([false],AA.arrows[right]))

    return TNobjU,TNobjD,TNobjV,truncerr,newmag
  end

  """
      symsvd(A,order[,mag=,cutoff=,m=,name=,leftadd=,rightadd=])

  Takes `svd` of `A` according to `order` and returns U*sqrt(D),sqrt(D)*V

  See also: [`svd`](@ref)
  """
  function symsvd(AA::TNobj,order::Array{Array{B,1},1};mag::Number = 0.,power::Integer=2,
                  cutoff::Number = 0.,m::Integer = 0,name::String="svdind",rightadd::String="L",
                  leftadd::String="R") where B <: Union{Any,String}

    U,D,V = svd(AA,order,power=power,mag=mag,cutoff=cutoff,m=m,name=name,leftadd=leftadd,rightadd=rightadd)
    S1 = sqrt(D)
    return U*S1,S1*V
  end
  export symsvd

  import ..decompositions.eigen
  """
      eigen(A,order[,mag=,cutoff=,m=,name=,leftadd=])

  Generates eigenvalue decomposition of named tensor `A` according to `order`; same output as regular eigenvalue but with named tensors

  # Naming created index:
  + the index to the left of `D` is `vcat(name,leftadd)`
  """
  function eigen(AA::TNobj,order::Array{Array{B,1},1};mag::Number = 0.,cutoff::Number = 0.,
                  m::Integer = 0,name::String="eigind",leftadd::String="L") where B <: Union{Any,String}

    left = matchnames(AA,order[1])
    right = matchnames(AA,order[2])
    neworder = [left,right]
    leftname = name*leftadd

    D,U,truncerr,newmag = eigen(AA.N,order,mag=mag,cutoff=cutoff,m=m)

    TNobjD = nameT(D,[leftname,rightname])#,[false,true])
    TNobjU = nameT(U,vcat(AA.names[left],[leftname]))#,vcat(AA.arrows[left],[true]))
    return TNobjD,TNobjU,truncerr,newmag
  end

  """
      nameMPS(A)

  Assigns names to MPS `A`

  See also: [`nameMPO`](@ref)
  """
  function nameMPS(psi::MPS) #::Array{TNobj,1}
    TNmps = Array{TNobj,1}(undef,size(psi,1))
#    TNmps[1] = nameT(psi[1],["p1","l1"])#,[true,true])
#    counter = 0
    for i = 1:size(psi,1)
      TNmps[i] = nameT(psi[i],[join(["l",i-1]),join(["p",i]),join(["l",i])])#,[false,true,true])
    end
#    TNmps[size(psi,1)] = nameT(psi[size(psi,1)],[join(["l",size(psi,1)-1]),join(["p",size(psi,1)])])#,[false,true])
    return network(TNmps)
  end
  export nameMPS

  """
      nameMPO(A)

  Assigns names to MPO `A`

  See also: [`nameMPS`](@ref)
  """
  function nameMPO(mpo::MPO) #::Array{TNobj,1}
    TNmpo = Array{TNobj,1}(undef,size(mpo,1))
#    TNmpo[1] = nameT(psi[1],["p1","p1","lmpo1"])#,[false,true,true])
#    counter = 0
    for i = 1:size(mpo,1)
      TNmpo[i] = nameT(mpo[i],[join(["lmpo",i-1]),join(["p",i]),join(["p",i]),join(["lmpo",i])])#,[false,false,true,true])
    end
#    TNmpo[size(psi,1)] = nameT(psi[size(psi,1)],[join(["lmpo",size(psi,1)-1]),join(["p",size(psi,1)]),join(["p",size(psi,1)])])#,[false,false,true])
    return network(TNmpo)
  end
  export nameMPO

  import Base.copy
  """
      copy(A)

  Returns a copy of named tensor `A`
  """
  function copy(A::nameT{W,B}) where {W <: Union{qarray,AbstractArray,denstens}, B <: Union{Any,String}}
    return nameT{W,B}(copy(A.N),copy(A.names))#,[true for i = 1:length(A.names)])#,copy(A.arrows))
  end

  import Base.println
  """
      println(A[,show=])

  Prints named tensor `A`

  # Outputs:
  + `size`: size of `A`
  + `index names`: current names on `A`
  + `arrowss`: fluxes for each index on `A`
  + `elements`: elements of `A` if reshaped into a vector (out to `show`)
  """
  function println(A::TNobj;show::Integer=10)

    println("size = ",size(A))
    println("index names = ",A.names)
#    println("arrowss = ",A.arrows)
    if typeof(A.N) <: denstens ||  typeof(A.N) <: qarray
      temp = length(A.N.T)
      maxshow = min(show,temp)
      println("elements = ",A.N.T[1:maxshow])
    else #if typeof(A.N) <: AbstractArray
      rAA = reshape(A.N,prod(size(A)))
      temp = length(rAA)
      maxshow = min(show,temp)
      if length(rAA) > maxshow
        println("elements = ",rAA[1:maxshow],"...")
      else
        println("elements = ",rAA[1:maxshow])
      end
    end
    println()
#    print(A,show=show)
    nothing
  end
#=
  import Base.print
  """
      print(A[,show=])

  Prints named tensor `A`

  # Outputs:
  + `size`: size of `A`
  + `index names`: current names on `A`
  + `arrowss`: fluxes for each index on `A`
  + `elements`: elements of `A` if reshaped into a vector (out to `show`)
  """
  function print(A::TNobj;show::intType=10)
    println("size = ",size(A))
    println("index names = ",A.names)
    println("arrowss = ",A.arrows)
    rAA = reshape(A.N,prod(size(A)))
    if length(rAA) > show
      println("elements = ",rAA[1:show],"...")
    else
      println("elements = ",rAA[1:show])
    end
    nothing
  end
=#
  import Base.size
  """
      size(A[,w=])

  Gives the size of named tensor `A` where `w` is an integer or an index label
  """
  function size(A::TNobj)
    return size(A.N)
  end

  function size(A::TNobj,w::Integer)
    return size(A.N,w)
  end

  function size(A::TNobj,w::String)
    condition = true
    p = 0
    while condition && p < ndims(A)
      p += 1
      condition = A.names[p] != w
    end
    return size(A.N,w)
  end

  import ..Qtensor.norm
  """
      norm(A)

  Gives the norm of named tensor `A`
  """
  function norm(A::TNobj)
    return norm(A.N)
  end



  import ..Qtensor.div!
  """
      div!(A,num)

  Gives the division of named tensor `A` by number `num`

  See also: [`/`](@ref)
  """
  function div!(A::TNobj,num::Number)
    A.N = div!(A.N,num)
    return A
  end

  import ..Qtensor./
  """
      /(A,num)

  Gives the division of named tensor `A` by number `num`

  See also: [`div!`](@ref)
  """
  function /(A::TNobj,num::Number)
    return div!(copy(A),num)
  end



  import ..Qtensor.mult!
  """
      mult!(A,num)

  Gives the multiplication of named tensor `A` by number `num`

  See also: [`*`](@ref)
  """
  function mult!(A::TNobj,num::Number)
    A.N = mult!(A.N,num)
    return A
  end

  import ..Qtensor.*
  """
      *(A,num)

  Gives the multiplication of named tensor `A` by number `num` (commutative)

  See also: [`mult!`](@ref)
  """
  function *(A::TNobj,num::Number)
    return mult!(copy(A),num)
  end

  function *(num::Number,A::TNobj)
    return A*num
  end


  import .QN.add!
  #=
  """
      add!(A,B)

  Adds tensors `A` and `B` (changes `A`)

  See also: [`+`](@ref)
  """
  =#
  function add!(A::TNobj,B::TNobj)
    reorder = matchnames(A,B.names)
    if !issorted(reorder)
      C = permutedims(B,reorder)
    else
      C = B
    end
    A.N = add!(A.N,C.N)
    return A
  end

  import ..Qtensor.+
  """
      +(A,B)

  Adds tensors `A` and `B`

  See also: [`add!`](@ref)
  """
  function +(A::TNobj,B::TNobj)
    return add!(copy(A),B)
  end


  import ..Qtensor.sub!
  """
      sub!(A,B)

  Subtracts tensor `A` from `B` (changes `A`)

  See also: [`-`](@ref)
  """
  function add!(A::TNobj,B::TNobj)
    reorder = matchnames(A,B.names)
    if !issorted(reorder)
      C = permutedims(B,reorder)
    else
      C = B
    end
    A.N = sub!(A.N,C.N)
    return A
  end

  import ..Qtensor.-
  """
      -(A,B)

  Subtracts tensor `A` from `B`

  See also: [`sub!`](@ref)
  """
  function -(A::TNobj,B::TNobj)
    return sub!(copy(A),B)
  end

  import Base.sqrt
  """
      sqrt(A)

  Takes the square root of named tensor `A`

  See also: [`sqrt`](@ref)
  """
  function sqrt(A::TNobj)
    B = copy(A)
    return sqrt!(B)
  end

  import .tensor.sqrt!
  """
      sqrt!(A)

  Takes the square root of named tensor `A`

  See also: [`sqrt!`](@ref)
  """
  function sqrt!(A::TNobj)
    A.N = sqrt!(A.N)
    return A
  end

  import Base.ndims
  """
      ndims(A)

  Returns the number of indices of named tensor `A`
  """
  function ndims(A::TNobj)
    return length(A.names)
  end

  import ..Qtensor.conj!
  """
      conj!(A)

  Conjugates named tensor `A` in-place

  See also: [`conj`](@ref)
  """
  function conj!(A::TNobj)
    conj!(A.N)
#    A.arrows = Bool[!A.arrows[i] for i = 1:ndims(A)]
    nothing
  end

  import LinearAlgebra.conj
  """
      conj(A)

  Conjugates named tensor `A`

  See also: [`conj!`](@ref)
  """
  function conj(A::TNobj)::TNobj
    B = copy(A)
    conj!(B)
    return B
  end


  import ..contractions.trace
  """
      trace(A)

  Computes the trace of named tensor `A` over indices with 1) the same name and 2) opposite arrowss
  """
  function trace(A::TNobj)
    vect = Array{intType,1}[]
    for w = 1:length(A.names)
      condition = true
      z = w+1
      while condition && z < length(A.names)
        z += 1
        if A.names[w] == A.names[z] #&& A.arrows[w] && A.arrows[z]
          push!(vect,[w,z])
          condition = false
        end
      end
    end
    return trace(A.N,vect)
  end

  """
      trace(A,inds)

  Computes the trace of named tensor `A` with specified `inds` (integers, symbols, or strings--ex: [[1,2],[3,4],[5,6]])
  """
  function trace(A::TNobj,inds::Array{Array{W,1},1}) where W <: Union{Any,Integer}
    if W <: Integer
      return trace(A.N,inds)
    else
      vect = Array{intType,1}[zeros(intType,2) for i = 1:length(inds)] #Array{intType,1}[Array{intType,1}(undef,2) for i = 1:length(inds)]
      for w = 1:length(A.names)
        matchindex!(A,vect,inds,w,1)
        matchindex!(A,vect,inds,w,2)
      end
      return trace(A.N,vect)
    end
  end

  """
      trace(A,inds)

  Computes the trace of named tensor `A` with specified `inds` (integers, symbols, or strings--ex: [1,2])
  """
  function trace(A::TNobj,inds::Array{W,1}) where W <: Union{Any,Integer}
    return trace(A,[inds])
  end

  """
      matchindex!(A,vect,inds,w,q)

  takes index names from `vect` over indices `inds` (position `w`, index `q`, ex: inds[w][q]) and converts into an integer; both `vect` and `index` are of the form Array{Array{?,1},1}

  See also: [`trace`](@ref)
  """
  function matchindex!(A::TNobj,vect::Array{Array{P,1},1},inds::Array{Array{W,1},1},w::Integer,q::Integer) where {W <: Union{Any,Integer}, P <: Integer}
#    condition = true
#    z = 0
    convInds!(A,inds,vect)
    nothing
  end

  """
      convInds!(A,inds,vect)

  converts named indices in `A` to integers; finds only indices specified in `inds` and returns `vect` with integers; does nothing if its only integers
  """
  function convInds!(A::TNobj,inds::Array{Array{W,1},1},vect::Array{Array{P,1},1}) where {W <: Union{Any,Integer}, P <: Integer}
    if W <: Integer
      return inds
    end
    for a = 1:length(vect)
      for b = 1:length(vect[a])
        saveind = b > 1 ? vect[a][b-1] : 0
        for c = 1:length(A.names)
          if inds[a][b] == A.names[c] && saveind != c
            vect[a][b] = c
            saveind = c
          end
        end
      end
    end
    return vect
  end

  """
      rename!(A,inds)

  replaces named indices in `A` with indices in `inds`; either format [string,[string,arrow]] or [string,string] or [string,[string]] is accepted for `inds`
  """
  function rename!(A::TNobj,inds::Array{Array{W,1},1}) where W <: Any
    for a = 1:length(inds)
      condition = true
      b = 0
      while condition && b < length(A.names)
        b += 1
        if A.names[b] == inds[a][1]
          if typeof(inds[a][2]) <: Array
            A.names[b] = inds[a][2][1]
#            A.arrows[b] = inds[a][2][2]
          else
            A.names[b] = inds[a][2]
          end
        end
      end
    end
    nothing
  end
  #=            one = ["s1",["i1",false]]
            two = ["s2",["i2",false]]
            three = ["s3",["i3",true]]
            four = ["s4",["i4",true]]
            rename!(A1,[one,two,three,four])=#

  """
      rename!(A,currvar,newvar[,arrows])

  replaces a string `currvar` in named indices of `A` with `newvar`; can also set arrows if needed
  """
  function rename!(A::TNobj,currvar::String,newvar::String)#,arrows::Array{Bool,1}...)
    for a = 1:length(A.names)
      loc = findfirst(currvar,A.names[a])
      if !(typeof(loc) <: Nothing)
        first = loc[1] == 1 ? "" : A.names[a][1:loc[1]-1]
        last = loc[end] == length(A.names[a]) ? "" : A.names[a][loc[end]+1]
        newstring = first * newvar * last
        A.names[a] = newstring
      end
#      if length(arrows) > 0
#        A.arrows[a] = arrows[a]
#      end
    end
    nothing
  end
  export rename!

  function rename(A::TNobj,inds::Array{Array{W,1},1}) where W <: Any
    B = copy(A)
    rename!(B,inds)
    return B
  end
  export rename
#=
  function reshape(A,order)
    newnumberorder = Int64[]
    for i = 1:size(order,1)
      for j = 1:size(A.names,1)
        if order[i] == A.names[j]
          push!(newnumberorder,j)
          continue
        end
      end
    end
    W = permutedims(A,newnumberorder)
    counter = 0
    reshapevec = Array{intType,1}[]
    for a = 1:length(newnumberorder)
      temp = [i+counter for i = 1:length(newnumberorder[a])]
      push!(reshapevec,temp)
      counter += length(temp)
    end
    return reshape(W,reshapevec)
  end
=#

  function addindex!(X::TNobj,Y::TNobj)
#    push!(X.arrows,true)
#    push!(Y.arrows,false)
    if typeof(X.N) <: denstens || typeof(X.N) <: qarray
      X.N.size = (size(X.N)...,1) #not super efficient if you're copying beforehand
    else #if typeof(X.N) <: AbstractArray
#      try 
#        X.N = reshape(X.N,size(X.N)...,1) #not super efficient if you're copying beforehand
#      catch
        error("trying to convert an array to the wrong type...switch to using the denstens class")
#      end
#    elseif typeof(X.N) <: qarray
#      X.N = reshape!(X.N,size(X.N)...,1) #not super efficient if you're copying beforehand
    end
    if typeof(Y.N) <: denstens || typeof(Y.N) <: qarray
      Y.N.size = (size(Y.N)...,1) #not super efficient if you're copying beforehand
    elseif typeof(Y.N) <: AbstractArray
#      try
#        Y.N = reshape(Y.N,size(Y.N)...,1) #not super efficient if you're copying beforehand
#      catch
        error("trying to convert an array to the wrong type...switch to using the denstens class")
#      end
#    elseif typeof(Y.N) <: qarray
#      Y.N = reshape!(Y.N,size(Y.N)...,1) #not super efficient if you're copying beforehand
    end
#    push!(X.size,1)
#    push!(Y.size,1)
    push!(X.names,"extra_ones")
    push!(Y.names,"extra_ones")
    nothing
  end
  export addindex!

  function addindex(X::TNobj,Y::TNobj)
    A = copy(X)
    B = copy(Y)
    addindex!(A,B)
    return A,B
  end
  export addindex

#=
  function joinTens!(X::TNobj,Y::TNobj)
    addindex!(X,Y)
    res = X*Y
    if typeof(X.N) <: denstens || typeof(X.N) <: qarray
      X.N.size = X.N.size[1:end-1]
    end
    if typeof(Y.N) <: denstens || typeof(Y.N) <: qarray
      Y.N.size = Y.N.size[1:end-1]
    end
    return res
  end
  export joinTens!
=#
  function joinTens(X::TNobj,Y::TNobj)
    A,B = addindex(X,Y)
    return A*B
  end
  export joinTens




  mutable struct sizeT{W,B} <: TNobj where {W <: Integer,B <: Union{Any,String}}
    size::Array{W,1}
    names::Array{B,1}
  end

  function sizeT(Qt::nameT{W,B})::TNobj where {T <: TNobj, W <: Union{qarray,AbstractArray,denstens}, B <: Union{Any,String}}
    return sizeT{Int64,B}(size(Qt),Qt.names)#,[true for i = 1:length(namez)])
  end

  function size(Qt::sizeT{W,B}) where {W <: Integer,B <: Union{Any,String}}
    return Qt.size
  end

  function size(Qt::sizeT{W,B},i::Integer) where {W <: Integer,B <: Union{Any,String}}
    return Qt.size[i]
  end

  function commoninds(A::TNobj,B::TNobj)
    Ainds = intType[]
    Binds = intType[]
    let A = A, B = B, Ainds = Ainds, Binds = Binds
      #=Threads.@threads=# for i = 1:length(A.names)
        for j = 1:length(B.names)
          if A.names[i] == B.names[j] #&& A.arrows[i] != B.arrows[j]
            push!(Ainds,i)
            push!(Binds,j)
          end
        end
      end
    end
    allA = [i for i = 1:ndims(A)]
    allB = [i for i = 1:ndims(B)]
    notconA = setdiff(allA,Ainds)
    notconB = setdiff(allB,Binds)
    return Ainds,Binds,notconA,notconB
  end

  function contractcost(A::TNobj,B::TNobj)
    conA,conB,notconA,notconB = commoninds(A,B)
    concost = prod(w->size(A,conA[w]),1:length(conA))
    concost *= prod(w->size(A,notconA[w]),1:length(notconA))
    concost *= prod(w->size(B,notconB[w]),1:length(notconB))
    return concost
  end
  export contractcost


  function sizecost(A::TNobj,B::TNobj;alpha::Bool=true)#arxiv:2002.01935
    conA,conB,notconA,notconB = commoninds(A,B)
    cost = prod(w->size(A,w),notconA)*prod(w->size(B,w),notconB)
    if alpha
      cost -= prod(size(A)) + prod(size(B))
    end
    return cost
  end
  export sizecost

  function bgreedy(TNnet::TNnetwork;nsamples::Integer=length(TNnet.net),costfct::Function=contractcost)
    numtensors = length(TNnet.net)
    basetensors = [sizeT(TNnet.net[i]) for i = 1:numtensors]
    savecontractlist = Int64[]
    savecost = 99999999999999999999999
    for i = 1:nsamples
      currTensInd = (i-1) % nsamples + 1
      contractlist = [currTensInd]
      A = basetensors[currTensInd]

      availTens = [i for i = 1:numtensors]
      deleteat!(availTens,currTensInd)
      searchindex = copy(availTens)

      nextTens = rand(1:length(searchindex),1)[1]

      newcost = 0

      while length(availTens) > 1
        B = basetensors[searchindex[nextTens]]
        if isdisjoint(A.names,B.names)
          deleteat!(searchindex,nextTens)
          nextTens = rand(1:length(searchindex),1)[1]
        else
          push!(contractlist,searchindex[nextTens])

          searchindex = setdiff(availTens,contractlist)
          availTens = setdiff(availTens,contractlist[end])
          

          vecA,vecB = contractinds(A,B)
          retindsA = setdiff([i for i = 1:ndims(A)],vecA)
          retindsB = setdiff([i for i = 1:ndims(B)],vecB)
          newnames = vcat(A.names[retindsA],B.names[retindsB])
          newsizes = vcat(size(A)[retindsA],size(B)[retindsB])

          newcost += costfct(A,B)

          A = sizeT(newsizes,newnames)

          nextTens = rand(1:length(searchindex),1)[1]
        end
      end
      push!(contractlist,availTens[end])
      if savecost > newcost
        savecost = newcost
        savecontractlist = contractlist
      end
    end
    return savecontractlist
  end
  export bgreedy

  import .contractions.contract
  function contract(Z::network;method::Function=bgreedy)
    order = method(Z)
    outTensor = Z[order[1]]*Z[order[2]]
    for i = 3:length(order)
      outTensor = outTensor * Z[order[i]]
    end
    return outTensor
  end




#=
  function KHP(Qts::network,invec::Array{R,1};nsamples::Integer=sum(p->length(Qts[p].names),invec),costfct::Function=contractcost) where R <: Integer
    exclude_ind = invec[1]
    numtensors = length(Qts) - length(invec)
    basetensors = [sizeT(Qts[i]) for i = 1:numtensors]
    savecost = 99999999999999999999999

    savecontractlist = Int64[]

    startnames = Qts[invec[1]].names
    for g = 2:length(invec)
      startnames = vcat(startnames,Qts[invec[g]].names)
    end
    startnames = unique(startnames)

    for i = 1:nsamples
      currTensInd = (i-1) % length(startnames) + 1
      contractlist = Int64[]

      availTens = setdiff([i for i = 1:numtensors],invec)
      searchindex = copy(availTens)

      nextTens = rand(1:length(searchindex),1)[1]

      newcost = 0

      A = sizeT([1],[startnames[currTensInd]])



println(availTens)


      while length(availTens) > 1

        println(searchindex)
        println(searchindex[nextTens])

        B = basetensors[searchindex[nextTens]]
        if isdisjoint(A.names,B.names)
          println("FAIL ",searchindex," ",nextTens)
          deleteat!(searchindex,nextTens)
          nextTens = rand(1:length(searchindex),1)[1]
          println(nextTens)
        else

          if length(contractlist) > 0 
            newcost += costfct(A,B)
  end

          push!(contractlist,searchindex[nextTens])

          searchindex = setdiff(availTens,contractlist)
          availTens = setdiff(availTens,contractlist[end])
          

          vecA,vecB = contractinds(A,B)
          retindsA = setdiff([i for i = 1:ndims(A)],vecA)
          retindsB = setdiff([i for i = 1:ndims(B)],vecB)
          newnames = vcat(A.names[retindsA],B.names[retindsB])
          newsizes = vcat(size(A)[retindsA],size(B)[retindsB])

          A = sizeT(newsizes,newnames)

          nextTens = rand(1:length(searchindex),1)[1]
        end
      end

      A = basetensors[currTensInd]


      push!(contractlist,availTens[end])
      if savecost > newcost
        savecost = newcost
        savecontractlist = contractlist
      end
    end














#=
    searching = true
    x = 1
    currnames = startnames
    newcost = 0
    contractorder = []
    while searching && x <= length(currnames)
      firstindex = currnames[x]

      nextTens = rand(1:length(activeTensors),1)[1]

      w = 1
      while isdisjoint(firstindex,Qts[activeTensors[nextTens]].names) && w < length(activeTensors)
        nextTens = rand(1:length(activeTensors),1)[1]
        w += 1
      end
      if w == length(activeTensors)
        x += 1
      else
        push!(contractorder,activeTensors[nextTens])
        deleteat!(activeTensors,nextTens)
        searching = length(contractorder) != (length(Qts)-length(exclude_ind))
      end
    end
    push!(contractorder,activeTensors[1])
    newcost = 0
    println(contractorder)
    println(numtensors)
    A = basetensors[contractorder[end]]
    for m = numtensors:-1:2
      B = basetensors[contractorder[m]]
      newcost += costfct(A,B)

      vecA,vecB = contractinds(A,B)
      retindsA = setdiff([i for i = 1:ndims(A)],vecA)
      retindsB = setdiff([i for i = 1:ndims(B)],vecB)
      newnames = vcat(A.names[retindsA],B.names[retindsB])
      newsizes = vcat(size(A)[retindsA],size(B)[retindsB])

#      newcost += costfct(A,B)

      A = sizeT(newsizes,newnames)
    end
    if newcost < savecost
      savecontractorder = reverse(contractorder)
    end
=#
    return savecontractorder
  end
  export KHP
=#
end




#=
  function LRpermute(conA,A)
    Aleft = sort(conA,rev=true) == [ndims(A) - i + 1 for i = 1:length(conA)]
    Aright = sort(conA) == [i for i = 1:length(conA)]
    if Aleft
      permA = "L"
    elseif Aright
      permA = "R"
    else
      permA = "LR" #signifies permutation
    end
    return permA
  end

  function checkpermute(A::TNobj,conA::Array{intType,1},B::TNobj,
                        conB::Array{intType,1},nelems::Array{intType,1},i::Integer,j::Integer)
    permA = LRpermute(conA,A)
    permB = LRpermute(conB,B)
    nelemA = nelems[i]
    nelemB = nelems[j]
    if permA == permB == "L"
      if nelemA > nelemB
        permB = "R"
      else
        permA = "R"
      end
    elseif permA == permB == "R"
      if nelemA > nelemB
        permB = "L"
      else
        permA = "L"
      end
#    elseif permA == permB == "LR" #|| permA == "LR" || permB == "LR"
    end
    return permA,permB
  end

  function commoninds(A::TNobj,B::TNobj)
    Ainds = intType[]
    Binds = intType[]
    let A = A, B = B, Ainds = Ainds, Binds = Binds
      #=Threads.@threads=# for i = 1:length(A.names)
        for j = 1:length(B.names)
          if A.names[i] == B.names[j] && A.arrows[i] != B.arrows[j]
            push!(Ainds,i)
            push!(Binds,j)
          end
        end
      end
    end
    return Ainds,Binds
  end

  function contractcost(A::TNobj,B::TNobj)
    conA,conB = commoninds(A,B)
    allA = [i for i = 1:ndims(A)]
    allB = [i for i = 1:ndims(B)]
    notconA = setdiff(allA,conA)
    notconB = setdiff(allB,conB)
    permA = permutes(notconA,conA)
    permB = permutes(conB,notconB)
    concost = prod(w->size(A,conA[w]),1:length(conA))
    concost *= prod(w->size(A,notconA[w]),1:length(notconA))
    concost *= prod(w->size(B,notconB[w]),1:length(notconB))
    return concost,permA,permB
  end

  function conorder(T::TNobj...)
    NT = length(T)
    free = fill(true,NT,NT)
    costs = fill(true,NT,NT)
    permutes = fill(true,NT,NT)
    contens = fill(true,NT,NT)
    @simd for i = 1:NT
      free[i,i] = false
      permutes[i,i] = false
      #missing test for consecutive kronecker....contens?
    end
    order = [Array{UInt8,1}(undef,NT) for i = 1:NT, j = 1:NT]
    leftright = Array{String,1}(undef,NT,NT,2)
    nelems = Array{intType,1}(undef,NT)
    @simd for i = 1:NT
      nelems[i] = prod(size(T[i]))
      order[i][j] = i
    end
    for i = 1:NT
      for j = 1:NT
        if free[i,j]
          if concost == 0
            if contens[i,j]
              #kronecker product
              contens[i,j] = false
              leftright[i,j,1],leftright[i,j,2] = "L","R"
            else
              free[i,j] = false
              costs[i,j] = 0
            end
          else
            concost = contractcost(T[i],T[j])
            leftright[i,j,1],leftright[i,j,2] = checkpermute(T[i],T[j],nelems,i,j) #HERE: Must also put in niformation from previous tensors level
            if concost > costs[i,j]
              costs[i,j] = concost
            end
          end
        end
      end
      xy = argmin(costs)
    end
  end


  import .contractions.contract
  function contract(Z::network)
  end
=#
