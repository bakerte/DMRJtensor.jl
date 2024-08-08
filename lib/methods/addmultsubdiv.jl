





#=
struct mpsnetwork{W} <: MPS where W <: directedtens
  A::Array{W,1}
  oc::intType
end

struct mponetwork{W} <: MPO where W <: directedtens
  H::Array{W,1}
end
=#









#=
function PEPSnet()

  #
  # KIANA: Input a neighbor table (see below for example) and output a PEPS in this new system
  #

  #
  # [2 5;
  #  3 6;
  #  ....]
  #
  #would be for a 4x4 lattice
  #

end

"""
    MERAnet(A)

Assigns names to MPO `A`

See also: [`nameMPS`](@ref)
"""
function MERAnet(mera::MERA)
  W = typeof(mera[1])
  TNmera = Array{directedtens{W},1}(undef,length(mera))
  if ndims(mera[1]) % 2 == 0 #bad logic!
    rang = cld(ndims(mera[1]),2)
    for i = 1:length(mera)

      #AARON:
      # RULE: p * "fine lattice position" but the coarse position is the position in the input network (can also use level and range of network as inputs, but find the minimum information required)
      #

      namesleft = ["p$(currlevel*i + rang*(i-1) + w)" for w = 1:cld(rang,2)]
      names = vcat(namesleft,namesleft) #["a$(2*i-1)","a$(2*i-1)","a$(2*i)","a$(2*i)"]
      arrows = Int8[w % 2 == 0 ? 1 : -1 for w = 1:2*range] #[1,-1,1,-1]
      TNmera[i] = directedtens(mera[i],names,arrows)
    end
  else
    rang = ndims(mera[1])-1
    currlevel = cld(mera.A.level,2)
    for i = 1:length(mera)
      upindex = ["p$(currlevel*rang*i)"]
      names = vcat(upindex,["p$(currlevel*rang*i-fld(rang,2)+w)" for w = 0:rang-1])
      arrows = [w == 1 ? 1 : -1 for w = 1:length(names)]
      TNmera[i] = directedtens(mera[i],names,arrows)
    end
  end

#
# KIANA AND AARON: Make a 2D MERA (or higher dimensional MERA) from this
#

  return MERAnet(TNmera)
end
export MERAnet
=#





























































#       +---------------------------------------+
#>------+           convert to qMPS             +---------<
#       +---------------------------------------+
















#=
function possibleQNs(QNsummary::Array{Q,1},w::Integer,physinds::Array{Array{Q,1},1},flux::Q,m::Integer) where Q <: Qnum
  maxQNrange = [QNsummary[q] for q = 1:length(QNsummary)]
  minQNrange = [QNsummary[q] for q = 1:length(QNsummary)]

  minQN = Q()
  maxQN = Q()
  for i = w+1:length(physinds)
    minQN += minimum(physinds[i])
    maxQN += maximum(physinds[i])
  end
  possibleQN = Array{Bool,1}(undef,length(QNsummary))
  for q = 1:length(QNsummary)
    possibleQN[q] = QNsummary[q] + minQN <= flux <= QNsummary[q] + maxQN
  end
  QNsummary = QNsummary[possibleQN]
  if length(QNsummary) > m
    QNsummary = rand(QNsummary,m)
  end
  return QNsummary
end


function randMPS(Qlabels::Array{Q,1},Ns::Integer;oc::Integer=1,m::Integer=1,type::DataType=Float64,flux::Q=Q()) where Q <: Qnum
  return randMPS([Qlabels],Ns,oc=oc,m=m,type=type,flux=flux)
end

function randMPS(Qlabels::Array{Array{Q,1},1};oc::Integer=1,m::Integer=1,type::DataType=Float64,flux::Q=Q()) where Q <: Qnum
  return randMPS(Qlabels,length(Qlabels),oc=oc,m=m,type=type,flux=flux)
end

function randMPS(Qlabels::Array{Array{Q,1},1},Ns::Integer;oc::Integer=1,m::Integer=1,type::DataType=Float64,flux::Q=Q(),maxiter::Integer=10000) where Q <: Qnum
  physinds = [Qlabels[(w-1) % length(Qlabels) + 1] for w = 1:Ns]

  storeQNs = Array{Array{Array{Q,1},1},1}(undef,Ns)
  nonzerointersect = true

  lastQNs = [inv(flux)]
  QNsummary2 = multi_indexsummary([physinds[end],lastQNs],[1,2])

  counter = 0
  while nonzerointersect && counter < maxiter
    counter += 1
    currQNs = [Q()]
    for w = 1:Ns-1
      QNsummary = multi_indexsummary([physinds[w],currQNs],[1,2])
      QNsummary = possibleQNs(QNsummary,w,physinds,flux,m)

      newQNs = inv.(QNsummary)
      temp = [currQNs,physinds[w],newQNs]
      storeQNs[w] = temp

      currQNs = QNsummary
    end

    QNsummary = intersect(inv.(currQNs),QNsummary2)
    QNsummary = possibleQNs(QNsummary,Ns,physinds,flux,m)

    newQNs = inv.(QNsummary)
    storeQNs[end] = [newQNs,physinds[end],[Q()]]
    nonzerointersect = length(newQNs) == 0
  end

  tensvec = Array{Qtens{type,Q},1}(undef,Ns)
  for w = 1:Ns
    thisblock = w <= oc ? [[1,2],[3]] : [[1],[2,3]]
    tensvec[w] = rand(storeQNs[w],currblock = thisblock, datatype = type, flux = w < Ns ? Q() : flux)
  end
  mps = MPS(tensvec,oc=oc)

  move!(mps,Ns)
  move!(mps,1)
  move!(mps,oc)

  mps[1] /= norm(mps[1])
  return mps
end
=#








#       +---------------------------------------+
#>------+           convert to qMPO             +---------<
#       +---------------------------------------+
























#=


#         +-----------------+
#>--------|    MPO viewer   |----------<
#         +-----------------+

#defines the multiplication of two arrays of strings

struct paulistring{W <: String}
  term::Array{W,1}
end

struct Hterm{W}
  vec::Array{paulistring{W},1}
end

import Base.length
function length(X::paulistring)
  return length(X.term)
end

function length(X::Hterm)
  return length(X.vec)
end

import Base.getindex
function getindex(X::paulistring,a::Integer)
  return X.term[a]
end

function getindex(X::Hterm,a::Integer)
  return X.vec[a]
end

import Base.lastindex
function lastindex(X::paulistring)
  return X.term[end] #[length(X.term)]
end

=#




#=
function maketerm(X::paulistring)
  finalterm = ""
  for w = 1:length(X)-1
    finalterm *= X[w]
    finalterm *= "⊗"
  end
  return finalterm * X[length(X)]
end

function maketerm(X::Hterm)
  finalterm = ""
  for w = 1:length(X)-1
    finalterm *= maketerm(X[w])
    finalterm *= " + "
  end
  return finalterm * maketerm(X[length(X)])
end

import Base.display
function display(A::Array{W,2}) where W <: Hterm
  C = Array{String,2}(undef,size(A)...)
  for y = 1:size(C,2)
    for x = 1:size(C,1)
      C[x,y] = maketerm(A[x,y])
    end
  end
  display(C)
end

function display(A::Hterm)
  display(maketerm(A))
end

function format_matrix(H::Array{W,2}) where W <: String
  return [Hterm([paulistring([H[x,y]])]) for x = 1:size(H,1), y = 1:size(H,2)]
end

import Base.*
function *(A::paulistring,B::paulistring)
  return [paulistring(vcat(A.term,B.term))]
end

function *(A::Hterm,B::Hterm)
  C = Array{paulistring{String},1}[]
  for y = 1:length(B)
    for x = 1:length(A)
      push!(C,A.vec[x]*B.vec[y])
    end
  end
  return Hterm(vcat(C...))
end

import Base.+
function +(A::Hterm,B::Hterm)
  Cvec = vcat(A.vec,B.vec)
  keepvec = Array{Bool,1}(undef,length(Cvec))
  for w = 1:length(keepvec)
    finalbool = true
    p = 0
    while finalbool && p < length(Cvec[w])
      p += 1
      finalbool &= Cvec[w][p] != "O"
    end
    keepvec[w] = finalbool
  end

  return sum(keepvec) == 0 ? Hterm([paulistring(["O"])]) : Hterm(Cvec[keepvec])
end

function mult(A::Array{Hterm{String}, 2},B::Array{Hterm{String}, 2})
  C = [Hterm([paulistring(["O"])]) for x = 1:size(A,1), y = 1:size(B,2)]
  for y = 1:size(B,2)
    for x = 1:size(A,1)
      for z = 1:size(A,2)
        out = A[x,z]*B[z,y]
        C[x,y] += out
      end
    end
  end
  return C
end

function mult(A::Array{String,2},B::Array{String,2})
  X = format_matrix(A)
  Y = format_matrix(B)
  return mult(X,Y)
end

function mult(A::Array{Hterm{String}},B::Array{String,2})
  Y = format_matrix(B)
  return mult(A,Y)
end

function mult(A::Array{String,2},B::Array{Hterm{String}})
  X = format_matrix(A)
  return mult(X,B)
end

function *(A::Array{Hterm{String}, 2},B::Array{Hterm{String}, 2})
  return mult(A,B)
end

function *(A::Array{String,2},B::Array{String,2})
  return mult(A,B)
end

function *(A::Array{Hterm{String}},B::Array{String,2})
  return mult(A,B)
end

function *(A::Array{String,2},B::Array{Hterm{String}})
  return mult(A,B)
end

=#

