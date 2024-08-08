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
    G = largeMPS(psi[,label="mps_",names=[label*"i" for i = 1:length(psi)],ext=".dmrjulia"])

Writes tensors from `MPS` `psi` to hard disk as retrieved through `G` according to filenames specified in `names` (default file extension: `.dmrjulia`)

See also: [`largeMPO`](@ref) [`largeLenv`](@ref) [`largeRenv`](@ref) [`largeEnv`](@ref)
"""
function largeMPS(psi::MPS;label::String="mps_",names::Array{String,1}=[label*"$i" for i = 1:length(psi)],ext::String=file_extension)
  lastnum = 1
  for b = 1:length(psi)
    C = psi[b]
    tensor2disc(names[b],C,ext=ext)
    lastnum *= eltype(C)(1)
  end
  return largematrixproductstate(names,psi.oc,typeof(lastnum))
end

"""
    G = largeMPS(Ns[,label="mps_",names=[label*"i" for i = 1:length(psi)],ext=".dmrjulia"])

Creates a large MPS type (stored on disk) with `Ns` tensors (element type: Float64) but does not write anything to disk initially. Filenames specified in `names` (default file extension: `.dmrjulia`)

See also: [`largeMPO`](@ref) [`largeLenv`](@ref) [`largeRenv`](@ref) [`largeEnv`](@ref)
"""
function largeMPS(Ns::Integer;label::String="mps_",names::Array{String,1}=[label*"$i" for i = 1:Ns],ext::String=file_extension,thisoc::Integer=1,type::DataType=Float64)
  return largematrixproductstate(names,thisoc,type)
end

"""
    G = largeMPS(T,Ns[,label="mps_",names=[label*"i" for i = 1:length(psi)],ext=".dmrjulia"])

Creates a large MPS type (stored on disk) of element type `T` with `Ns` tensors but does not write anything to disk initially. Filenames specified in `names` (default file extension: `.dmrjulia`)

See also: [`largeMPO`](@ref) [`largeLenv`](@ref) [`largeRenv`](@ref) [`largeEnv`](@ref)
"""
function largeMPS(type::DataType,Ns::Integer;label::String="mps_",names::Array{String,1}=[label*"$i" for i = 1:Ns],ext::String=file_extension,thisoc::Integer=1)
  return largeMPS(Ns,label=label,names=names,ext=ext,thisoc=thisoc,type=type)
end





"""
    G = loadMPS(Ns[,label="mps_",names=[label*"i" for i = 1:Ns],ext=".dmrjulia"])

If MPS tensors are stored on hard disk, then they can be retrieved by using `loadMPS`
"""
function loadMPS(Ns::Integer;label::String="mps_",names::Array{String,1}=[label*"$i" for i = 1:Ns],ext::String=file_extension,oc::Integer=0)
  lastnum = 1
  storeoc = [1]
  for i = 1:Ns
    name = names[i]
    A = tensorfromdisc(name,ext=ext)
    lastnum *= eltype(A)(1)
    if isapprox(norm(A),1.)
      storeoc[1] = i
    end
  end
  thistype = typeof(lastnum)
  if oc == 0
    thisoc = storeoc[1]
  else
#    @assert(storeoc[1] == oc)
    thisoc = oc
  end
  return largematrixproductstate(names,thisoc,thistype)
end
export loadMPS

