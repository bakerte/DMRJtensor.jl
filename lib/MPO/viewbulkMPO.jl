###############################################################################
#
#  Density Matrix Renormalization Group (and other methods) in julia (DMRjulia)
#                               v1.0
#
###############################################################################
# Made by Thomas E. Baker and « les qubits volants » (2024)
# See accompanying license with this program
# This code is native to the julia programming language (v1.10.0+)
#

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

const subnumberstrings = [Char(0x2081),Char(0x2082),Char(0x2083),Char(0x2084),Char(0x2085),Char(0x2086),Char(0x2087),Char(0x2088),Char(0x2089),Char(0x2080)]

"""
  half

String code for a fraction of 1/2; useful for bulkMPO

See also: [`bulkMPO`](@ref) [`Omat`](@ref)
"""
const half = Char(0x00BD)
export half

function substringnumber(stringstart::String,w::Integer)
  pstart = ceil(log(10,w) + 1)
  mone = fld(w,10^(pstart-1)) % 10 == 0 #whether to add 1 or not
  pstart = convert(Int64,ceil(log(10,w))) + (mone ? 0 : 1)
  out = ntuple(p->fld(w,10^(pstart-p)) % 10,pstart)

  makestring = stringstart
  for b = 1:length(out)
    pos = (b-1) % length(subnumberstrings) + 1
    rpos = pos == 0 ? 10 : pos
    rout = out[rpos] == 0 ? 10 : out[rpos]
    makestring *= subnumberstrings[rout]
  end
  return makestring
end







"""
  Omat(m)

Generates a bulk matrix product operator with reserved string "O" for zero with a matrix of size `m x m`. Can multiply this object to get bulk MPO for a given system.


# Example:

```julia

H = bulkMPO(5) #["O" for i = 1:5,j = 1:5]

#XXZ model
H[1,1] = H[end,end] = "I"
H[2,1] = "S"*Char(0x207B)
H[3,1] = "S"*Char(0x207A)
H[4,1] = "Sz"
H[end,2] = half*"S"*Char(0x207A)
H[end,3] = half*"S"*Char(0x207B)
H[end,4] = "Sz"
H*H*H #multiply 3 sites together

```

See also: [`bulkMPO`](@ref) [`half`](@ref)
"""
function Omat(m::Integer)
  return ["O" for i = 1:m,j = 1:m]
end
export Omat

"""
  bulkMPO(m)

Generates a bulk matrix product operator with reserved string "O" for zero with a matrix of size `m x m`. Can multiply this object to get bulk MPO for a given system.

# Example:

# Example:

```julia

H = bulkMPO(5) #["O" for i = 1:5,j = 1:5]

#XXZ model
H[1,1] = H[end,end] = "I"
H[2,1] = "S"*Char(0x207B)
H[3,1] = "S"*Char(0x207A)
H[4,1] = "Sz"
H[end,2] = half*"S"*Char(0x207A)
H[end,3] = half*"S"*Char(0x207B)
H[end,4] = "Sz"
H*H*H #multiply 3 sites together

```

See also: [`Omat`](@ref) [`half`](@ref)
"""
function bulkMPO(m::Integer)
  return Omat(m)
end
export bulkMPO


function maketerm(X::paulistring;kron::Bool=false)
  if kron
    finalterm = ""
    for w = 1:length(X)-1
      finalterm *= X[w]
      finalterm *= "⊗" #"⋅"
    end
    return finalterm * X[length(X)]
  else
#    yesprint = [X != "I" && X != "Id" for w = 1:length(X)]
    p = 1
    while p < length(X) && (X[p] == "I" || X[p] == "Id")
      p += 1
    end
    finalterm = substringnumber(X[p],p)
    for w = p+1:length(X)
      if X[w] != "I" && X[w] != "Id"
        finalterm *= "⋅"
        finalterm *= substringnumber(X[w],w) #X[w]*
      end
    end
    return finalterm
  end
end

function maketerm(X::Hterm;kron::Bool=false)
  finalterm = ""
  for w = 1:length(X)-1
    finalterm *= maketerm(X[w],kron=kron)
    finalterm *= " + "
  end
  return finalterm * maketerm(X[length(X)],kron=kron)
end

import Base.display
function display(A::Array{W,2};kron::Bool=false) where W <: Hterm
  C = Array{String,2}(undef,size(A)...)
  for y = 1:size(C,2)
    for x = 1:size(C,1)
      C[x,y] = maketerm(A[x,y],kron=kron)
    end
  end
  display(C)
end

function display(A::Hterm;kron::Bool=false)
  display(maketerm(A,kron=kron))
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