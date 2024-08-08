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

"""
    psi = randMPS(T,physindsize,Ns[,oc=1,m=1])

Generates MPS with data type `T`, uniform physical index size `physindsize`, with `Ns` sites, orthogonality center `oc`, and bond dimension `m`.
"""
function randMPS(T::DataType,physindsize::Integer,Ns::Integer;oc::Integer=1,m::Integer=1)
  physindvec = [physindsize for i = 1:Ns]
  return randMPS(T,physindvec,oc=oc,m=m)
end

"""
    psi = randMPS(T,physindvec,Ns[,oc=1,m=1])

Generates MPS with data type `T`, physical index size vector `physindvec` (repeating over `Ns` sites), with `Ns` sites, orthogonality center `oc`, and bond dimension `m`.
"""
function randMPS(T::DataType,physindvec::Array{W,1};oc::Integer=1,m::Integer=1) where W <: Integer
  Ns = length(physindvec)
  vec = Array{Array{T,3},1}(undef,Ns)
  if m == 1
    for w = 1:Ns
      vec[w] = zeros(1,physindvec[w],1)
      state = rand(1:physindvec[w],1)[1]
      vec[w][1,state,1] = 1
    end
    psi = MPS(vec,oc=oc)
  else
    Lsize,Rsize = 1,prod(w->physindvec[w],2:length(physindvec))
    currLsize = 1
    for w = 1:Ns
      physindsize = physindvec[w]
      currRsize = min(Rsize,m)
      vec[w] = rand(T,currLsize,physindsize,currRsize)
      vec[w] /= norm(vec[w])
      currLsize = currRsize
      Rsize = cld(Rsize,physindsize)
    end
    psi = MPS(vec,oc=oc)
    move!(psi,1)
    move!(psi,Ns)
    move!(psi,1)
    psi[oc] /= expect(psi)
  end
  return psi
end

"""
    psi = randMPS(physindsize,Ns[,oc=1,m=1])

Generates MPS with data type Float64, uniform physical index size `physindsize`, with `Ns` sites, orthogonality center `oc`, and bond dimension `m`.
"""
function randMPS(physindsize::Integer,Ns::Integer;oc::Integer=1,m::Integer=1,datatype::DataType=Float64)
  return randMPS(datatype,physindsize,Ns,oc=oc,m=m)
end

"""
    psi = randMPS(T,physindvec,Ns[,oc=1,m=1])

Generates MPS with data type Float64, physical index size vector `physindvec` (repeating over `Ns` sites), with `Ns` sites, orthogonality center `oc`, and bond dimension `m`.
"""
function randMPS(physindvec::Array{W,1};oc::Integer=1,m::Integer=1,datatype::DataType=Float64) where W <: Integer
  return randMPS(datatype,physindvec,oc=oc,m=m)
end

function randMPS(physindvec::Array{W,1},Ns::Integer;oc::Integer=1,m::Integer=1,datatype::DataType=Float64) where W <: Integer
  newphysindvec = [physindvec[(w-1) % length(physindvec) + 1] for w = 1:Ns]
  return randMPS(datatype,newphysindvec,oc=oc,m=m)
end

function randMPS(psi::MPS;oc::Integer=psi.oc,m::Integer=maximum([size(psi[w],3) for w = 1:length(psi)]),datatype::DataType=eltype(psi),physind::Union{intType,Array{intType,1}} = [size(psi[w],2) for w = 1:length(psi)])
  if typeof(psi[1]) <: qarray
    Ns = length(psi)
    Qlabels = [[getQnum(2,w,psi[i]) for w = 1:size(psi[i],2)] for i = 1:Ns]
    return randMPS(Qlabels,datatype=datatype,physindvec,oc=oc,m=m)
  else
    Ns = length(psi)
    physindvec = [physind[(i-1) % length(physind) + 1] for i = 1:Ns]
    return randMPS(datatype,physindvec,oc=oc,m=m)
  end
end

function randMPS(mpo::MPO;oc::Integer=1,m::Integer=1,datatype::DataType=eltype(mpo),physind::Union{intType,Array{intType,1}} = [size(mpo[w],2) for w = 1:length(mpo)])
  if typeof(mpo[1]) <: qarray
    Ns = length(mpo)
    Qlabels = [[getQnum(3,w,mpo[i]) for w = 1:size(mpo[i],3)] for i = 1:Ns]
    return randMPS(Qlabels,datatype=datatype,physindvec,oc=oc,m=m)
  else
    Ns = length(mpo)
    physindvec = [physind[(i-1) % length(physind) + 1] for i = 1:Ns]
    return randMPS(datatype,physindvec,oc=oc,m=m)
  end
end
export randMPS

function randMPS(Qlabel::Array{Q,1},Ns::Integer;m::Integer=2,type::DataType=Float64,flux::Q=Q()) where Q <: Qnum
  Qlabels = [Qlabel for i = 1:Ns]
  return randMPS(Qlabels,Ns,m=m,type=type,flux=flux)
end

function randMPS(Qlabels::Array{Array{Q,1},1},Ns::Integer;m::Integer=2,type::DataType=Float64,flux::Q=Q()) where Q <: Qnum
  A = Array{tens{type},1}(undef,Ns)
  for i = 1:Ns
    lsize = i == 1 ? 1 : m
    rsize = i == Ns ? 1 : m
    A[i] = tens(rand(lsize,length(Qlabels[(i-1) % Ns + 1]),rsize))
  end

  mps = makeqMPS(Qlabels,A,flux=flux)

  move!(mps,Ns)
  move!(mps,1)
#  move!(mps,oc)

  mps[mps.oc] /= norm(mps[mps.oc])

  return mps
end


