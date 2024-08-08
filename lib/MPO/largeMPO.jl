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
    G = largeMPO(mpo[,label="mpo_",names=[label*"i" for i = 1:length(mpo)],ext=".dmrjulia"])

Writes tensors from `MPO` `mpo` to hard disk as retrieved through `G` according to filenames specified in `names` (default file extension: `.dmrjulia`)

See also: [`largeMPS`](@ref) [`largeLenv`](@ref) [`largeRenv`](@ref) [`largeEnv`](@ref)
"""
function largeMPO(mpo::P;label::String="mpo_",names::Array{String,1}=[label*"$i" for i = 1:length(mpo)],ext::String=file_extension) where P <: Union{Array,MPO}
  lastnum = 1
  for b = 1:length(mpo)
    C = mpo[b]
    tensor2disc(names[b],C,ext=ext)
    lastnum *= eltype(C)(1)
  end
  return largematrixproductoperator(names,typeof(lastnum))
end

"""
    G = largeMPO(Ns[,label="mpo_",names=[label*"i" for i = 1:length(mpo)],ext=".dmrjulia"])

Creates a large MPO type (stored on disk) with `Ns` tensors (element type: Float64) but does not write anything to disk initially. Filenames specified in `names` (default file extension: `.dmrjulia`)

See also: [`largeMPS`](@ref) [`largeLenv`](@ref) [`largeRenv`](@ref) [`largeEnv`](@ref)
"""
function largeMPO(Ns::Integer;label::String="mpo_",names::Array{String,1}=[label*"$i" for i = 1:Ns],ext::String=file_extension,type::DataType=Float64)
  return largematrixproductoperator(names,type)
end

"""
    G = largeMPO(T,Ns[,label="mpo_",names=[label*"i" for i = 1:length(mpo)],ext=".dmrjulia"])

Creates a large MPO type (stored on disk) of element type `T` with `Ns` tensors but does not write anything to disk initially. Filenames specified in `names` (default file extension: `.dmrjulia`)

See also: [`largeMPS`](@ref) [`largeLenv`](@ref) [`largeRenv`](@ref) [`largeEnv`](@ref)
"""
function largeMPO(type::DataType,Ns::Integer;label::String="mpo_",names::Array{String,1}=[label*"$i" for i = 1:Ns],ext::String=file_extension)
  return largeMPO(Ns,label=label,names=names,ext=ext,type=type)
end



"""
    G = loadMPO(Ns[,label="mpo_",names=[label*"i" for i = 1:Ns],ext=".dmrjulia"])

If MPO tensors are stored on hard disk, then they can be retrieved by using `loadMPO`
"""
function loadMPO(Ns::Integer;label::String="mpo_",names::Array{String,1}=[label*"$i" for i = 1:Ns],ext::String=file_extension,oc::Integer=0)
  lastnum = 1
  for i = 1:Ns
    name = names[i]
    A = tensorfromdisc(name,ext=ext)
    lastnum *= eltype(A)(1)
  end
  thistype = typeof(lastnum)
  return largematrixproductoperator(names,thistype)
end
export loadMPO

