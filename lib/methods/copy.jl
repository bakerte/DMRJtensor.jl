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
    newpsi = copy(psi)

Copies an `MPS` to new output container of the same type; type stable (where deepcopy is type-unstable inherently)
"""
function copy(mps::regMPS)
  out = typeof(mps.A[1])[copy(mps.A[i]) for i = 1:length(mps)]
  return MPS(out,oc=copy(mps.oc))
end
#=
function copy(mps::matrixproductstate{W}) where W <: TensType
  return matrixproductstate{W}([copy(mps.A[i]) for i = 1:length(mps)],copy(mps.oc))
end

function copy(mpo::matrixproductoperator{W}) where W <: TensType
  return matrixproductoperator{W}([copy(mpo.H[i]) for i = 1:length(mpo)])
end
=#

"""
    newmpo = copy(mpo)

Copies an `MPO` to new output container of the same type; type stable (where deepcopy is type-unstable inherently)
"""
function copy(mpo::regMPO)
  out = typeof((mpo.H[1]))[copy(mpo.H[i]) for i = 1:length(mpo)]
  return MPO(out)
end

"""
    newG = copy(G)

Copies an `Env` to new output container of the same type; type stable (where deepcopy is type-unstable inherently)
"""
function copy(G::regEnv)

#  println(G)

  out = Array{eltype(G.V.net),1}(undef,length(G))
  for i = 1:length(G)
#    if isundef(G.V,i)
#      out[i] = copy(G.V[i])
#    else
#      out[i] = emptyTensor(G.V)
#    end
    try
      out[i] = copy(G.V[i])
    catch
      out[i] = emptyTensor(G.V)
    end
  end
  return environment(out)
end


"""
    G = copy(names,X[,ext=".dmrjulia",copyext=ext])

Copies tensors from `largeMPS` input `X` to a new tensor with a vector of strings `names` representing the new filenames with input extension `ext` and the new extension `copyext`
"""
function copy(names::Array{String,1},X::largeMPS;ext::String=file_extension,copyext::String=ext)
  newObj = largematrixproductstate(String[names[(i-1) % length(names) + 1] for i = 1:length(X)],X.oc,X.type)
  for i = 1:length(X)
    Y = tensorfromdisc(X.A[i],ext=ext)
    tensor2disc(newObj.A[i],Y,ext=copyext)
  end
  return newObj
end

"""
    G = copy(names,X[,ext=".dmrjulia",copyext=ext])

Copies tensors from `largeMPS` input `X` to a new tensor with a vector of a single string `names` which is used to generate a vector of names representing the new filenames with input extension `ext` and the new extension `copyext`
"""
function copy(names::String,X::largeMPS;ext::String=file_extension,copyext::String=ext)
  return copy(names .* X.A,X,ext=ext,copyext=copyext)
end

"""
    G = copy(names,X[,ext=".dmrjulia",copyext=ext])

Copies tensors from `largeMPO` input `X` to a new tensor with a vector of strings `names` representing the new filenames with input extension `ext` and the new extension `copyext`
"""
function copy(names::Array{String,1},X::largeMPO;ext::String=file_extension,copyext::String=ext)
  newObj = largematrixproductoperator(String[names[(i-1) % length(names) + 1] for i = 1:length(X)],X.type)
  for i = 1:length(X)
    Y = tensorfromdisc(X.H[i],ext=ext)
    tensor2disc(newObj.H[i],Y,ext=copyext)
  end
  return newObj
end

"""
    G = copy(names,X[,ext=".dmrjulia",copyext=ext])

Copies tensors from `largeMPO` input `X` to a new tensor with a vector of a single string `names` which is used to generate a vector of names representing the new filenames with input extension `ext` and the new extension `copyext`
"""
function copy(names::String,X::largeMPO;ext::String=file_extension,copyext::String=ext)
  return copy(names .* X.H,X,ext=ext,copyext=copyext)
end

"""
    G = copy(names,X[,ext=".dmrjulia",copyext=ext])

Copies tensors from `largeEnv` input `X` to a new tensor with a vector of strings `names` representing the new filenames with input extension `ext` and the new extension `copyext`
"""
function copy(names::Array{String,1},X::largeEnv;ext::String=file_extension,copyext::String=ext)
  newObj = largeenvironment([names[(i-1) % length(names) + 1] for i = 1:length(X)],X.type)
  for i = 1:length(X)
    Y = tensorfromdisc(X.V[i],ext=ext)
    tensor2disc(newObj.V[i],Y,ext=copyext)
  end
  return newObj
end

"""
    G = copy(names,X[,ext=".dmrjulia",copyext=ext])

Copies tensors from `largeEnv` input `X` to a new tensor with a single string `names` which is used to generate a vector of names representing the new filenames with input extension `ext` and the new extension `copyext`
"""
function copy(names::String,X::largeEnv;ext::String=file_extension,copyext::String=ext)
  return copy(names .* X.V,X,ext=ext,copyext=copyext)
end

