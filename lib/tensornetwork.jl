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
#=
module tensornetwork
#using ..shuffle
using ..tensor
using ..QN
using ..Qtensor
#using ..Qtask
using ..contractions
using ..decompositions
using ..MPutil
=#
abstract type TNobj end
export TNobj

abstract type TNnetwork end
export TNnetwork

"""
    nametens{W,B}

named tensor with tensor of type `W` and type of names `B`

# Fields:
+ `N::W`: Tensor stored
+ `names::Array{B,1}`: names of all indices
"""
mutable struct nametens{W,B} <: TNobj where {W <: TensType, B <: Union{Any,String}}
  N::W
  names::Array{B,1}
end

"""
    nametens(Qt,namez)

constructor for named tensor from a tensor `Qt` and vector of index names `namez`
"""
function nametens(Qt::TensType,namez::Array{B,1};regTens::Bool=false)::TNobj where B <: Union{Any,String}
  newQt = !regTens && typeof(Qt) <: AbstractArray ? tens(Qt) : Qt
  return nametens{typeof(newQt),B}(newQt,namez)
end

"""
    nametens(Qt,string)

constructor for named tensor from a tensor `Qt` and vector of index names `string` where an integer is added onto `string` to make the full name of the index
"""
function nametens(Qt::T,namez::String;regTens::Bool=false)::TNobj where T <: TensType
  return nametens(Qt,[namez*"$i" for i = 1:basedims(Qt)],regTens=regTens)
end
export nametens



"""
  directedtens{W,B}

named tensor with named tensor of type `W` and vector of Booleans `B`. Creates a directed graph for use in MERA computations

# Fields:
+ `T::W`: named tensor stored
+ `names::Array{B,1}`: arrows of all indices

See also: [`nametens`](@ref)
"""
mutable struct directedtens{W,B} <: TNobj where {W <: nametens, B <: Bool}
  T::W
  arrows::Array{B,1}
  conj::Bool
end



"""
  directedtens(Qt,vecbools)

constructor for named tensor `Qt` and vector of directed arrows `vecbools`
"""
function directedtens(Qt::nametens{W,B},vecbools::Array{Bool,1};conj::Bool=false) where {W <: TensType, B <: Union{Any,String}}
  return directedtens(Qt,vecbools,conj)
end

"""
  directedtens(Qt,namez)

constructor for named tensor from a tensor `Qt` and vector of index names `namez`
"""
function directedtens(Qt::TensType,namez::Array{B,1},vecbools::Array{Bool,1};regTens::Bool=false,conj::Bool=false)::TNobj where B <: Union{Any,String}
  return directedtens(nametens(newQt,namez,regtens=regtens),vecbools,conj=conj)
end

"""
  directedtens(Qt,string)

constructor for named tensor from a tensor `Qt` and vector of index names `string` where an integer is added onto `string` to make the full name of the index
"""
function directedtens(Qt::T,namez::String;regTens::Bool=false)::TNobj where T <: TensType
  return directedtens(nametens(Qt,namez,regtens=regtens),vecbools,conj=conj)
end
export directedtens

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
function network(Qts::W...) where W  <: TNobj #where S <: Union{Any,String}
  return network{W}([Qts[i] for i = 1:length(Qts)])
end

"""
  network(Qt,i)

converts named tensor `Qt` to a network with `i` copied elements not shallow copied
"""
function network(Qts::W,n::Integer) where W  <: TNobj #where S <: Union{Any,String}
return network{W}([copy(Qts) for i = 1:n])
end
export network

import ..Base.getindex
function getindex(Qts::TNnetwork,i::Integer)
  return Qts.net[i]
end

function getindex(Qts::nametens,i::Integer)
  return Qts.N[i]
end

function getindex(Qts::directedtens,i::Integer)
  return getindex(Qts.T,i)
end

import ..Base.setindex!
function setindex!(Qts::TNnetwork,newTens::TNobj,i::Integer)
  return Qts.net[i] = newTens
end

import ..Base.length
function length(Qts::TNnetwork)
  return length(Qts.net)
end

function contractinds(A::nametens,B::nametens;check::Bool=false)
  pairs = Matrix{Bool}(undef,length(A.names),length(B.names))
  counter = 0
  for b = 1:size(pairs,2)
    for a = b:size(pairs,1)
      if A.names[a] == B.names[b]
        counter += 1
        pairs[a,b] = true
        pairs[b,a] = true
      else
        pairs[a,b] = false
        pairs[b,a] = false
      end
    end

    if check
      checkcounter = 0
      @inbounds @simd for x = b:size(pairs,1)
        checkcounter += pairs[x,b]
      end
      if checkcounter > 1
        error("Indices not paired on contraction of named tensors (duplicate index name detected)")
      end
    end
  end

  vecA = Array{intType,1}(undef,counter)
  vecB = Array{intType,1}(undef,counter)

  newcounter = 0
  b = 0
  while newcounter < counter
    b += 1
    a = b-1
    search_bool = true
    for a = b:size(pairs,1)
      if pairs[a,b]
        newcounter += 1
        vecA[newcounter] = a
        vecB[newcounter] = b
        search_bool = false
      end
    end
  end

  return vecA,vecB,pairs
end

function contractinds(A::directedtens,B::directedtens)
  return contractinds(A.T,B.T)
end

#  import ..Qtensor.*
"""
    *(A,B...)

Contracts `A` and any number of `B` along common indices; simple algorithm at present for the order
"""
function *(A::nametens,B::nametens)
  vecA,vecB,pairs = contractinds(A,B)

  xnewnames = 0
  ynewnames = 0
  for y = 1:size(pairs,2)
    counter = 0
  #  x = y
  #  while x < size(pairs,1) && counter == 0
  #    x += 1
    @inbounds @simd for x = y:size(pairs,1)
      counter += pairs[x,y]
    end
    if counter == 0
      xnewnames += 1
    end


    counter = 0
    x = 0
  #  while x < y && counter == 0
  #    x += 1
    @inbounds @simd for x = 1:y
      counter += pairs[x,y]
    end
    if counter == 0
      ynewnames += 1
    end
  end

  newnames = Array{String,1}(undef,xnewnames+ynewnames)

  name_counter = 0
  y = 0
  while name_counter < xnewnames
    y += 1
    counter = 0
    @inbounds @simd for x = y:size(pairs,1)
      counter += pairs[x,y]
    end
    if counter == 0
      name_counter += 1
      newnames[name_counter] = A.names[y]
    end
  end

  y = 0
  while name_counter < xnewnames + ynewnames
    y += 1
    counter = 0
    @inbounds @simd for x = 1:y
      counter += pairs[x,y]
    end
    if counter == 0
      name_counter += 1
      newnames[name_counter] = B.names[y]
    end
  end

  newTens = contract(A.N,vecA,B.N,vecB)

  return nametens(newTens,newnames)
end

function *(A::directedtens,B::directedtens)
  C = A.T * B.T
  newarrows = vcat(A.arrows[retindsA],B.arrows[retindsB])
  return directedtens(C,newarrows,false)
end

function *(R::TNobj...)
  out = *(R[1],R[2])
  @simd for b = 3:length(R)
    out = *(out,R[b])
  end
  return out
end


function sum(R::TNobj)
  return sum(R.N)
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
#  import ..Qtensor.permutedims!
"""
    permtuedims!(A,order)

Permutes named tensor `A` according to `order` (ex: [[1,2],[3,4]] or [["a","b"],["c","d"]])

See also: [`permutedims`](@ref)
"""
function permutedims!(A::TNobj,order::Array{W,1}) where W <: String
  newnumberorder = intType[]
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

function permutedims!(B::TNobj,order::Array{W,1}) where W <: Integer

  A = typeof(B) <: nametens ? B : B.T

  A.N = permutedims!(A.N,order)
  A.names = A.names[order]
#    A.arrows = A.arrows[order]

  if typeof(B) <: directedtens
    B.T = A
  end

  return B
end

"""
    matchnames(AA,order,q)

Matches `order` (a length 2 vector of vectors of strings for indices) to the indices in `AA` for the left (right) with `q`=1 (2)
"""
function matchnames(AA::nametens,order::Array{B,1}) where B <: Union{Any,String}
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
#  import ..decompositions.svd
"""
    svd(A,order[,mag=,cutoff=,m=,name=,leftadd=,rightadd=])

Generates SVD of named tensor `A` according to `order`; same output as regular SVD but with named tensors

# Naming created index:
+ the index to the left of `D` is `vcat(name,leftadd)`
"""
function svd(AA::nametens,order::Array{Array{B,1},1};mag::Number = 0.,cutoff::Number = 0.,
              m::Integer = 0,power::Integer=2,name::String="svdind",leftadd::String="L",
              rightadd::String="R",nozeros::Bool=false) where B <: Union{Any,String}

  left = matchnames(AA,order[1])
  right = matchnames(AA,order[2])

  neworder = Array{intType,1}[left,right]
  leftname = name * leftadd
  rightname = name * rightadd

  U,D,V,truncerr,newmag = svd(AA.N,neworder,power=power,mag=mag,cutoff=cutoff,m=m,nozeros=nozeros)

  TNobjU = nametens(U,vcat(AA.names[left],[leftname]))
  TNobjD = nametens(D,[leftname,rightname])
  TNobjV = nametens(V,vcat([rightname],AA.names[right]))

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

  U,D,V,truncerr,mag = svd(AA,order,power=power,mag=mag,cutoff=cutoff,m=m,name=name,leftadd=leftadd,rightadd=rightadd)
  S1 = sqrt!(D)
  return U*S1,S1*V,truncerr,mag
end
export symsvd

#  import ..decompositions.eigen
"""
    eigen(A,order[,mag=,cutoff=,m=,name=,leftadd=])

Generates eigenvalue decomposition of named tensor `A` according to `order`; same output as regular eigenvalue but with named tensors

# Naming created index:
+ the index to the left of `D` is `vcat(name,leftadd)`
"""
function eigen(AA::nametens,order::Array{Array{B,1},1};mag::Number = 0.,cutoff::Number = 0.,
                m::Integer = 0,name::String="eigind",leftadd::String="L") where B <: Union{Any,String}

  left = matchnames(AA,order[1])
  right = matchnames(AA,order[2])
  neworder = [left,right]
  leftname = name*leftadd

  D,U,truncerr,newmag = eigen(AA.N,order,mag=mag,cutoff=cutoff,m=m)

  TNobjD = nametens(D,[leftname,rightname])
  TNobjU = nametens(U,vcat(AA.names[left],[leftname]))
  return TNobjD,TNobjU,truncerr,newmag
end

"""
    nameMPS(A)

Assigns names to MPS `A`

See also: [`nameMPO`](@ref)
"""
function nameMPS(psi::MPS)
  TNmps = Array{TNobj,1}(undef,length(psi))
  for i = 1:length(TNmps)
    TNmps[i] = nametens(psi[i],["l$(i-1)","p$i","l$i"])
  end
  return network(TNmps)
end
export nameMPS

"""
    nameMPO(A)

Assigns names to MPO `A`

See also: [`nameMPS`](@ref)
"""
function nameMPO(mpo::MPO)
  TNmpo = Array{TNobj,1}(undef,length(mpo))
  for i = 1:length(mpo)
    TNmpo[i] = nametens(mpo[i],["l$(i-1)","p$i","d$i","l$i"])
  end
  return network(TNmpo)
end
export nameMPO

"""
    conj(A)

Conjugates named MPS `A`

See also: [`conj!`](@ref)
"""
function conj(A::TNnetwork)
  return network([conj(A.net[i]) for i = 1:length(A)])
end

import Base.copy
"""
    copy(A)

Returns a copy of named tensor `A`
"""
function copy(A::nametens{W,B}) where {W <: Union{qarray,AbstractArray,denstens}, B <: Union{Any,String}}
  return nametens{W,B}(copy(A.N),copy(A.names))
end

"""
  copy(A)

Returns a copy of network of named tensors `A`
"""
function copy(A::TNnetwork)
  return network([copy(A.net[i]) for i = 1:length(A)])
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
  if typeof(A.N) <: denstens ||  typeof(A.N) <: qarray
    temp = length(A.N.T)
    maxshow = min(show,temp)
    println("elements = ",A.N.T[1:maxshow])
  else
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
  nothing
end

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

"""
    norm(A)

Gives the norm of named tensor `A`
"""
function norm(A::TNobj)
  return norm(A.N)
end

"""
    div!(A,num)

Gives the division of named tensor `A` by number `num`

See also: [`/`](@ref)
"""
function div!(A::TNobj,num::Number)
  A.N = div!(A.N,num)
  return A
end

"""
    /(A,num)

Gives the division of named tensor `A` by number `num`

See also: [`div!`](@ref)
"""
function /(A::TNobj,num::Number)
  return div!(copy(A),num)
end

"""
    mult!(A,num)

Gives the multiplication of named tensor `A` by number `num`

See also: [`*`](@ref)
"""
function mult!(A::TNobj,num::Number)
  A.N = mult!(A.N,num)
  return A
end

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

"""
    +(A,B)

Adds tensors `A` and `B`

See also: [`add!`](@ref)
"""
function +(A::TNobj,B::TNobj)
  return add!(copy(A),B)
end

"""
    sub!(A,B)

Subtracts tensor `A` from `B` (changes `A`)

See also: [`-`](@ref)
"""
function sub!(A::TNobj,B::TNobj)
  reorder = matchnames(A,B.names)
  if !issorted(reorder)
    C = permutedims(B,reorder)
  else
    C = B
  end
  A.N = sub!(A.N,C.N)
  return A
end

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
function sqrt(A::TNobj;root::Number=0.5)
  B = copy(A)
  return sqrt!(B,root=root)
end

#  import ..TENPACK.sqrt!
"""
    sqrt!(A)

Takes the square root of named tensor `A`

See also: [`sqrt!`](@ref)
"""
function sqrt!(A::TNobj;root::Number=0.5)
  A.N = tensorcombination!(A.N,alpha=(root,),fct=^)#sqrt!(A.N,root=root)
  return A
end

import Base.ndims
"""
    ndims(A)

Returns the number of indices of named tensor `A`
"""
function ndims(A::nametens)
  return length(A.names)
end

function ndims(A::directedtens)
  return length(A.arrows)
end

"""
    conj!(A)

Conjugates named tensor `A` in-place

See also: [`conj`](@ref)
"""
function conj!(A::nametens)
  conj!(A.N)
  nothing
end


function conj!(A::directedtens)
  @inbounds @simd for w = 1:ndims(A)
    A.arrows[w] = !A.arrows[w]
  end
  A.conj = !A.conj
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

"""
    trace(A)

Computes the trace of named tensor `A` over indices with 1) the same name and 2) opposite arrowss
"""
function trace(B::TNobj)
  A = typeof(B) <: nametens ? B : B.T
  vect = Array{intType,1}[]
  for w = 1:length(A.names)
    condition = true
    z = w+1
    while condition && z < length(A.names)
      z += 1
      if A.names[w] == A.names[z]
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
function trace(A::nametens,inds::Array{Array{W,1},1}) where W <: Union{Any,Integer}
  if W <: Integer
    return trace(A.N,inds)
  else
    vect = Array{intType,1}[zeros(intType,2) for i = 1:length(inds)]
    for w = 1:length(A.names)
      matchindex!(A,vect,inds,w,1)
      matchindex!(A,vect,inds,w,2)
    end
    return trace(A.N,vect)
  end
end

function trace(A::directedtens,inds::Array{Array{W,1},1}) where W <: Union{Any,Integer}
  B = trace(A.T,inds)
  newinds = vcat(inds...)
  leftoverinds = setdiff([i for i = 1:ndims(A)],newinds)
  newarrows = A.arrows[leftoverinds]
  return directedtens(B,newarrows,A.conj)
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
function matchindex!(A::nametens,vect::Array{Array{P,1},1},inds::Array{Array{W,1},1},w::Integer,q::Integer) where {W <: Union{Any,Integer}, P <: Integer}
  convInds!(A,inds,vect)
  nothing
end

"""
    convInds!(A,inds,vect)

converts named indices in `A` to integers; finds only indices specified in `inds` and returns `vect` with integers; does nothing if its only integers
"""
function convInds!(A::nametens,inds::Array{Array{W,1},1},vect::Array{Array{P,1},1}) where {W <: Union{Any,Integer}, P <: Integer}
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
  swapname!(A,labels)

Finds elements in `labels` (must be length 2) and interchanges the name. For example, `swapname!(A,["a","c"])` will find the label `"a"` and `"c"` and interchange them. A chain of swaps can also be requested (i.e., `swapname!(A,[["a","c"],["b","d"]])`)

Works as a pseudo-permute in many cases. Will not permute if the names are not found.

See also: [`swapnames!`](@ref)
"""
function swapname!(A::nametens,inds::Array{Array{W,1},1}) where W <: Any
  for c = 1:length(inds)
    x = 1
    while x < length(A.names) && A.names[x] != inds[c][1]
      x += 1
    end
    y = 1
    while y < length(A.names) && A.names[y] != inds[c][2]
      y += 1
    end
    if inds[c] == [A.names[x],A.names[y]]
      A.names[x],A.names[y] = A.names[y],A.names[x]
    end
  end
  nothing
end

function swapname!(A::nametens,inds::Array{W,1}) where W <: Any
  swapname!(A,[inds])
end
export swapname!

"""
  swapnames!(A,labels)

Finds elements in `labels` (must be length 2) and interchanges the name. For example, `swapname!(A,["a","c"])` will find the label `"a"` and `"c"` and interchange them. A chain of swaps can also be requested (i.e., `swapname!(A,[["a","c"],["b","d"]])`)

Works as a pseudo-permute in many cases. Will not permute if the names are not found.

See also: [`swapname!`](@ref)
"""
function swapnames!(A::nametens,inds::Array{Array{W,1},1}) where W <: Any
  swapname!(A,inds)
end

function swapnames!(A::nametens,inds::Array{W,1}) where W <: Any
  swapname!(A,[inds])
end
export swapnames!


"""
    rename!(A,inds)

replaces named indices in `A` with indices in `inds`; either format [string,[string,arrow]] or [string,string] or [string,[string]] is accepted for `inds`
"""
function rename!(A::nametens,inds::Array{Array{W,1},1}) where W <: Any
  for a = 1:length(inds)
    condition = true
    b = 0
    while condition && b < length(A.names)
      b += 1
      if A.names[b] == inds[a][1]
        if typeof(inds[a][2]) <: Array
          A.names[b] = inds[a][2][1]
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
function rename!(A::nametens,currvar::String,newvar::String)
  for a = 1:length(A.names)
    loc = findfirst(currvar,A.names[a])
    if !(typeof(loc) <: Nothing)
      first = loc[1] == 1 ? "" : A.names[a][1:loc[1]-1]
      last = loc[end] == length(A.names[a]) ? "" : A.names[a][loc[end]+1]
      newstring = first * newvar * last
      A.names[a] = newstring
    end
  end
  nothing
end
export rename!

function rename(A::nametens,inds::Array{Array{W,1},1}) where W <: Any
  B = copy(A)
  rename!(B,inds)
  return B
end
export rename

function addindex!(X::nametens,Y::nametens)
  if typeof(X.N) <: denstens || typeof(X.N) <: qarray
    X.N.size = (size(X.N)...,1)
  else
      error("trying to convert an array to the wrong type...switch to using the denstens class")
  end
  if typeof(Y.N) <: denstens || typeof(Y.N) <: qarray
    Y.N.size = (size(Y.N)...,1)
  elseif typeof(Y.N) <: AbstractArray
      error("trying to convert an array to the wrong type...switch to using the denstens class")
  end
  push!(X.names,"extra_ones")
  push!(Y.names,"extra_ones")
  nothing
end
export addindex!

function addindex(X::nametens,Y::nametens)
  A = copy(X)
  B = copy(Y)
  addindex!(A,B)
  return A,B
end
export addindex

function joinTens(X::nametens,Y::nametens)
  A,B = addindex(X,Y)
  return A*B
end
export joinTens





































 #
 # Kiana
 # 

  mutable struct sizeT{W,B} <: TNobj where {W <: Integer,B <: Union{Any,String}}
    size::Array{W,1}
    names::Array{B,1}
  end

  function sizeT(Qt::nametens{W,B}) where {W <: Union{qarray,AbstractArray,denstens}, B <: Union{Any,String}}
    return sizeT{Int64,B}(size(Qt),Qt.names)
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
      for i = 1:length(A.names)
        for j = 1:length(B.names)
          if A.names[i] == B.names[j]
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
#=
  function bgreedy(TNnet::TNnetwork;nsamples::Integer=length(TNnet.net),costfct::Function=contractcost)
    numtensors = length(TNnet.net) #number of tensors
    basetensors = [sizeT(TNnet.net[i]) for i = 1:numtensors] #sizes of the tensors
    savecontractlist = Int64[] #a list of possible ways to contract the network
    savecost = 99999999999999999999999 #very high cost to contracting the intial steps
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

#  import .contractions.contract
  function contract(Z::network;method::Function=bgreedy)
    order = method(Z)
    outTensor = Z[order[1]]*Z[order[2]]
    for i = 3:length(order)
      outTensor = outTensor * Z[order[i]]
    end
    return outTensor
  end

=#


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
#end




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


#       +--------------------------+
#>------| Code by Kiana Gallagher  |---------<
#       +--------------------------+

#ContractFunctions.jl

struct Indicies
	names::Vector{String}
	dimensions::Vector{intType}

end


"""
to_ascii()

Converts an ascii code of type UInt16 to a string value.

Parameters:
string: a string to be converted to an ascii value.

Returns:
A vector sotring the ascii equivalent for each letter of the string given.

"""
function to_ascii(string::String)::UInt32
	return Base.codepoint(string)

end


"""
to_string()

Converts a string to an ascii code value.

Parameters:
ascii_val: a vectory of ascii values to be converted to a string.

Returns:
The string equivalent of the ascii valus given.

"""
function to_string(ascii_val::UInt32)::String # this may need to be updated, look for an altnernative
	return Base.transcode(String, UInt32[ascii_val])

end


function remove_tensors(original_network::TNnetwork, to_remove::TNnetwork) #OK
	updated_network = []

	for tensor in original_network
		if !(tensor in to_remove)
			push!(updated_network, tensor)
		end

	end

	return updated_network
end


function find_common_edges(left_edges::Indicies, right_edges::Indicies)::Indicies #OK
	common_edges_names = []
	common_edges_dimensions = []

	for (pos, edge_name) in enumerate(left_edges.names)
		if edge_name in right_edges.names 
			push!(common_edges_names, edge_name)
			push!(common_edges_dimensions, left_edges.dimensions[pos])

		end
	end

	return Indicies(common_edges_names, common_edges_dimensions)

end


function lable_edges(left_edges::Indicies, right_edges::Indicies, common_edges::Indicies)::Indicies #OK
	all_edge_names = vcat(left_edges.names, right_edges.names)
	all_edge_dimensions = vcat(left_edges.dimensions, right_edges.dimensions)

	curr_pos = 0
	num_elements = length(left_edges.names) + length(right_edges.names) - 2*length(common_edges.names)
	new_tensor_names = Vector{String}(undef, num_elements)
	new_tensor_dimensions = Vector{Int64}(undef, num_elements)

	for (pos, edge) in enumerate(all_edge_names)
		if !(edge in common_edges.names)
			new_tensor_names[curr_pos+=1] =  edge
			new_tensor_dimensions[curr_pos] =  all_edge_dimensions[pos]

		end
	end

	return Indicies(new_tensor_names, new_tensor_dimensions)

end


function permute(edges::Indicies, common_edges::Vector{String})::intType # Check if it is possible to specify #OK
	permute_cost = 1
	position = Vector{intType}(undef, length(common_edges))

	for (pos, edge) in enumerate(common_edges)
		index = findfirst(==(edge), edges.names)
		position[pos] = index

	end

	sort!(position)

	if !(length(edges.names) in position) && !(1 in position)
		for edge_dim in edges.dimensions
			permute_cost *= edge_dim
			
		end

		return permute_cost

	else
		for pos in range(2, length(common_edges))
			if (position[pos]-position[pos-1]) != 1

				permute_required = true
				
				for edge_dim in edges.dimensions
					permute_cost *= edge_dim
			
				end

				return permute_cost

			end
		end
	end

	return 0
end


function cost(edges::Indicies)::Int64 #OK
	cost = 1

	for edge_dim in edges.dimensions
	cost *= edge_dim

	end

	return cost

end 


function contract_in_order(string_val::String, dictionary::Dict{Char, nametens{tens{Float64}, String}})::nametens{tens{Float64}, String} #OK 
	match_found = false
	start_string = 1
	end_string = 1 

	#Looks like a do-while loop

	m = match(r"[^\(\)]{2}", string_val, end_string)

	if (m != nothing)

		left_tensor = m.match[1]

		right_tensor = m.match[lastindex(m.match)]

		end_string = length(m.match) + m.offset

		contracted = dictionary[left_tensor]*dictionary[right_tensor]

		result = multiply(string_val, m.match, dictionary, contracted)

		match_found = true

	end

	while (match_found)
		m = match(r"[^\(\)]{2}", string_val, end_string)

		if (m != nothing)

			left_tensor = m.match[1]

			right_tensor = m.match[lastindex(m.match)]

			end_string = length(m.match) + m.offset

			contracted = dictionary[left_tensor]*dictionary[right_tensor]

			result *= multiply(string_val, m.match, dictionary, contracted)

		else
			match_found = false

		end
	end

	return result
end


function multiply(string_val::String, last_string, dictionary::Dict{Char, nametens{tens{Float64}, String}}, contracted::nametens{tens{Float64}, String})::nametens{tens{Float64}, String} #OK
	regex_left = Regex("[^\\(\\)]\\($(last_string)")
	m_left = match(regex_left, string_val)

	if !(m_left == nothing)

		tensor_key = m_left.match[1]

		new_string = "$(tensor_key)\\($(last_string)\\)"
		new_contract = dictionary[tensor_key]*contracted

		result = multiply(string_val, new_string, dictionary, new_contract)


	elseif (m_left == nothing)
	    regex_right = Regex("$(last_string)\\)[^\\(\\)]")
	    m_right = match(regex_right, string_val)

	    if !(m_right == nothing)

			tensor_key = m_right.match[lastindex(m_right.match)]

			new_string = "\\($(last_string)\\)$(tensor_key)"
			new_contract = contracted*dictionary[tensor_key]

			result = multiply(string_val, new_string, dictionary, new_contract)

		else
			result = contracted

		end
	end

	return result
end

############
############
############
# END OF Contract_functions.jl
############
############
############

#Greedy.jl

function common_edges(network)
	shared_edges = Dict()

  Nobjs = length(network)

	for w = 1:Nobjs
    tensor = network[w]
		for edge in tensor.names
			if edge in keys(shared_edges)
				push!(shared_edges[edge], tensor)

			else
				shared_edges[edge] = [tensor]

			end

		end


	end

	return shared_edges
end


function find_common_edges(left_edges::Indicies, right_edges::Indicies)::Indicies #OK
	common_edges_names = []
	common_edges_dimensions = []

	for (pos, edge_name) in enumerate(left_edges.names)
		if edge_name in right_edges.names 
			push!(common_edges_names, edge_name)
			push!(common_edges_dimensions, left_edges.dimensions[pos])

		end
	end

	return Indicies(common_edges_names, common_edges_dimensions)

end


function cost_possible(shared_edges)
	all_costs = Dict()
	cost_vals = []

	for tensors in values(shared_edges)
		for left_tensor in tensors
			for right_tensor in tensors
				if !(left_tensor == right_tensor) && (length(tensors) != 1)

#          println(left_tensor.names)
#          println(left_tensor.N.size)

          leftsize = [left_tensor.N.size[i] for i = 1:length(left_tensor.N.size)]
          rightsize = [right_tensor.N.size[i] for i = 1:length(right_tensor.N.size)]

					left_details = Indicies(left_tensor.names, leftsize)
					right_details = Indicies(right_tensor.names, rightsize)

					common_edges = find_common_edges(left_details, right_details)

					permute_cost = permute(left_details, common_edges.names) + permute(right_details, common_edges.names) 
					cost_tot = (cost(left_details) + cost(right_details))÷cost(common_edges)

					#order = find_order(left_tensor, right_tensor)
					all_costs[cost_tot] = (tensors = (left_tensor, right_tensor), common = common_edges)
					push!(cost_vals, cost_tot)


				end

			end
		end
	end

	return (possible_costs = cost_vals, cost_dict = all_costs)



end

function greedy(network)
  return greedy!(copy(network))
end

function greedy!(thisnetwork)

	shared_edges = common_edges(thisnetwork)
	cost_details = cost_possible(shared_edges)

	while length(cost_details.possible_costs) != 0


		sort!(cost_details.possible_costs) #is it better to sort in place?? to do a comparison when pushing to the dict

		least_expensive = cost_details.cost_dict[cost_details.possible_costs[1]] # do the contraction and repeat the process OR update the current elements

		left_tensor = least_expensive.tensors[1] # the left tensor
		right_tensor = least_expensive.tensors[2] # the right tensor

		result = left_tensor*right_tensor #expensive

    x = 1
    while x < length(thisnetwork) && left_tensor != thisnetwork[x]
      x += 1
    end

    y = 1
    while y < length(thisnetwork) && right_tensor != thisnetwork[y]
      y += 1
    end

    left_location = x
    right_location = y

#		left_location = findfirst(==(left_tensor), network)
#		right_location = findfirst(==(right_tensor), network)

    thisnetwork[left_location] = result

    tempfix = thisnetwork.net

		deleteat!(tempfix, right_location)
    thisnetwork = network(tempfix)

		shared_edges = common_edges(thisnetwork) #expensive
		cost_details = cost_possible(shared_edges) #expensive


end

  return thisnetwork[1]
end

import ..Base.keys
function keys(A::TNnetwork)
  return keys(A.net)
end

#ContractionAlg.jl

struct PseudoTensor
	name::String
	edges::Indicies
	composition::Vector{String}
	cost::intType
end


"""
convert_tensor()

The default tensors are converted into PseudoTensor datatypes.

Parameters:
tensor:
ascii_val:
position:

Returns:
PseudoTensor:

"""
function convert_tensor(tensor::nametens{tens{Float64}, String}, ascii_val::UInt32)::PseudoTensor #OK 

	# Gives the tensor the next possible name
	name = to_string(ascii_val)

	# Creating empty lists for stores the details of each index.
	edge_names = Vector{String}(undef, length(tensor.names))
	edge_dimensions = Vector{Int64}(undef, length(tensor.names))

	for pos = 1:length(tensor.names)
		edge_names[pos] = tensor.names[pos]
		edge_dimensions[pos] = tensor.N.size[pos]

	end

	# The for loop updates the information of the Indicies.
	# @time for pos in range(1, length(tensor.names))
	# 	push!(edge_names, tensor.names[pos])
	# 	push!(edge_dimensions, tensor.N.size[pos])

	# end

	# Stores the details of the Indicies for a tensor.
	edges = Indicies(edge_names, edge_dimensions)

	# Returns a PseudoTensor struct.
	return PseudoTensor(name, edges, [name], 0)

end 


function combine(left_composition, right_composition)
	combined = vcat(left_composition, right_composition)

	return unique!(combined)

end


"""
simple_contract()

Contracts two tensors that are of type PseudoTensor.

"""
function simple_contract(left_tensor::PseudoTensor, right_tensor::PseudoTensor, common_edges::Indicies)::PseudoTensor #GOOD
	name = "("*left_tensor.name*right_tensor.name*")" # A new name is given to the new tensor # alright

	permute_cost = permute(left_tensor.edges, common_edges.names) + permute(right_tensor.edges, common_edges.names) 

	new_edges = lable_edges(left_tensor.edges, right_tensor.edges, common_edges) #HERE
	added_cost = cost(new_edges) + cost(common_edges)
	new_cost = left_tensor.cost + right_tensor.cost + added_cost + permute_cost

	# Updates the composition of the tensors.
	new_composition = combine(left_tensor.composition, right_tensor.composition)
	
	return PseudoTensor(name, new_edges, new_composition, new_cost)

end


function find_same(left_comp::Vector{String}, right_comp::Vector{String})::Bool

for tensor in left_comp
	if tensor in right_comp
		return true

	end

end

return false

end


function expand_table(table::Dict{String, PseudoTensor}, starting_ascii::UInt32, network_size::intType)::Vector{PseudoTensor} #OK
	final_contractions = PseudoTensor[] # Holds the contractions that contract the whole network together.
	#queue = Queue{PseudoTensor}() # Initializes an empty queue.
	queue = PseudoTensor[]

	# Puts all the base case PseudoTensors into a queue.
	for tensor in values(table)
		# enqueue!(queue, tensor)
		push!(queue, tensor)

	end

	while (length(queue) != 0)
		#left_tensor = dequeue!(queue)
		left_tensor = pop!(queue)

		# The current left tensor is compared to all the other tensors in the table
		for right_tensor in values(table)
			# Will not contract on itself.
			if (left_tensor == right_tensor) 
				continue

			# Ensures it will not contract with a tree that already contains the left tensor.
			elseif find_same(left_tensor.composition, right_tensor.composition) # this is expensive, create a new function
				continue

			else
				# Checks if the left and right tensor have an Indicies in common.
				common_edges = find_common_edges(left_tensor.edges, right_tensor.edges) #this seems fairly effcient

				# If they have an edge in common then it is possible to contract.
				if (length(common_edges.names) != 0)

					# A new tensor is produced from contracting the left and right tensor.
					new_tensor = simple_contract(left_tensor, right_tensor, common_edges) #seems decent

					if (length(new_tensor.composition) == network_size)
						push!(final_contractions, new_tensor) #possible way to turn this into a counting problem... so the array can be of fixed size

					else
						table[new_tensor.name] = new_tensor
						push!(queue, new_tensor)
						# enqueue!(queue, new_tensor)

					end
				end
			end
		end
	end
	
	return final_contractions

end


"""
find_min()

Finds the minimum cost of contracting the network and return the last contraction completed
for the minimum cost network.

"""
function find_min(final_product::Vector{PseudoTensor})::PseudoTensor #OK
	answer = final_product[1]
	min_val = final_product[1].cost

	for result in final_product
		if (result.cost<=min_val)
			if (result.name<answer.name)
				answer = result
				min_val = result.cost
			end
		end
	end

	return answer

end


"""
contract()

This function contracts the tensor network given.

"""
function contract(network::TNnetwork; starting_ascii::UInt32=0x000000A1 #=rand(UInt32)=#, remove = intType[]) #where W <: Number #it does not like typehints with kwargs #GOOD
	if length(remove) != 0
		network = remove_tensors(network, remove)

	end

  if true
    contracted_result = greedy(network)
  else
    table::Dict{String, PseudoTensor} = Dict() # A dictionary is used to store the tensors that have been made.
    base_cases::Dict{Char, nametens{tens{Float64}, String}} = Dict() # Holds the basic tensors and their correpsonding names that have been assigned.

    # Creates a PseudoTensor datatype for each nametens datatype.

    g = length(network)
    for w = 1:g #tensor in network 
      tensor = network[w]
      temp_tensor = convert_tensor(tensor, starting_ascii)
      
      table[temp_tensor.name] = temp_tensor
      base_cases[only(temp_tensor.name)] = tensor

      starting_ascii += 0x00000001

    end

    final_result = expand_table(table, starting_ascii, length(network))

    if length(final_result)==0
      error("The tensor network given is disjoint")

    end

    best_order = find_min(final_result)

    contracted_result = contract_in_order(best_order.name, base_cases) 
  end
	return contracted_result 


end


function contract(remove::Array{intType,1},input_network::TNnetwork; starting_ascii::UInt32=0x000000A1 #=rand(UInt32)=#)
  return contract(input_network,starting_ascii=starting_ascii,remove=remove)
end

function contract(remove::Array{intType,1},input_network::TNobj...; starting_ascii::UInt32=0x000000A1 #=rand(UInt32)=#)
  return contract(network(input_network...),starting_ascii=starting_ascii,remove=remove)
end

function contract(input_network::TNobj...; starting_ascii::UInt32=0x000000A1 #=rand(UInt32)=#,remove::Array{intType,1}=intType[])
  return contract(network(input_network...),starting_ascii=starting_ascii,remove=remove)
end

function contract(remove::Array{intType,1},input_network::Array{W,1}; starting_ascii::UInt32=0x000000A1 #=rand(UInt32)=#) where W <: TNobj
  return contract(network(input_network),starting_ascii=starting_ascii,remove=remove)
end

function contract(input_network::Array{W,1}; starting_ascii::UInt32=0x000000A1 #=rand(UInt32)=#,remove::Array{intType,1}=intType[]) where W <: TNobj
  return contract(network(input_network),starting_ascii=starting_ascii,remove=remove)
end

#=
function main(val::Int64) #GOOD
	starting_ascii = 0x000000A1 # This is the first unicode value listed online.
	A::Vector{nametens{tens{Float64}, String}} = [nametens(tens(rand(1,2,2,1)), ["a$(i-1)", "b$i", "c$(i)", "a$i"]) for i in range(1,val)] # 10 is the max number tested so far
	#A::Vector{nametens{tens{Float64}, String}} = [nametens(tens(rand(1,2,1)), ["a$(i-1)", "b$i", "a$i"]) for i in range(1,val)]

	# A = nametens(tens(rand(1,2,1)), ["a0", "b1", "a1"])
	# B = nametens(tens(rand(1,2,1)), ["a1", "b2", "a2"])
	# C = nametens(tens(rand(1,2,1)), ["a2", "b3", "a3"])
	# D = nametens(tens(rand(1,2,1)), ["a3", "b4", "a4"])
	# E = nametens(tens(rand(1,2,1)), ["a4", "b5", "a5"])

	# F = [A,B,C,D,E]

	B::Vector{nametens{tens{Float64}, String}} = []

	middle = 0

	for i in 1:2
		for j in 1:2

		push!(B, nametens(rand(1,1,2,1,1), ["h"*"$(i)"*"$(j-1)", "h"*"$(i)"*"$(j)", "b$(middle)", "v"*"$(i-1)"*"$(j)", "v"*"$(i)"*"$(j)"]))
		middle += 1

		end
	end

	# C = [B[2], B[1], B[4], B[3]]

	# for b in B
	# 	println(b.names)

	# end

	result = contract(A, starting_ascii)



	# @time contract(A, starting_ascii)
	# println()


end

val = parse(Int64, ARGS[1])
#main(val)

@btime main(val)

#@btime contract(T, 0x000000A1)

#@btime (T[1]*T[2])*(T[3]*T[4])

=#






