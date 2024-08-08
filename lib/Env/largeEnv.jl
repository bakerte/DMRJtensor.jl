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
    G = largeLenv(lenv[,label="Lenv_",names=[label*"i" for i = 1:length(lenv)],ext=".dmrjulia"])

Writes tensors from environment `lenv` to hard disk as retrieved through `G` according to filenames specified in `names` (default file extension: `.dmrjulia`)

See also: [`largeMPS`](@ref) [`largeMPO`](@ref) [`largeRenv`](@ref) [`largeEnv`](@ref)
"""
function largeLenv(lenv::P;label::String="Lenv_",names::Array{String,1}=[label*"$i" for i = 1:length(lenv)],ext::String=file_extension) where P <: Union{Array,Env}
  lastnum = 1
  for b = 1:length(lenv)
    C = lenv[b]
    tensor2disc(names[b],C,ext=ext)
    lastnum *= eltype(C)(1)
  end
  return largeenvironment(names,typeof(lastnum))
end

"""
    G = largeLenv(Ns[,label="Lenv_",names=[label*"i" for i = 1:Ns],ext=".dmrjulia"])

Creates a large environment type (stored on disk) with `Ns` tensors (element type: Float64) but does not write anything to disk initially. Filenames specified in `names` (default file extension: `.dmrjulia`)

See also: [`largeMPS`](@ref) [`largeMPO`](@ref) [`largeRenv`](@ref) [`largeEnv`](@ref)
"""
function largeLenv(Ns::Integer;label::String="Lenv_",names::Array{String,1}=[label*"$i" for i = 1:Ns],ext::String=file_extension,type::DataType=Float64)
  return largeenvironment(names,type)
end

"""
    G = largeLenv(T,Ns[,label="Lenv_",names=[label*"i" for i = 1:Ns],ext=".dmrjulia"])

Creates a large environment type (stored on disk) of element type `T` with `Ns` tensors but does not write anything to disk initially. Filenames specified in `names` (default file extension: `.dmrjulia`)

See also: [`largeMPS`](@ref) [`largeMPO`](@ref) [`largeRenv`](@ref) [`largeEnv`](@ref)
"""
function largeLenv(type::DataType,Ns::Integer;label::String="Lenv_",names::Array{String,1}=[label*"$i" for i = 1:Ns],ext::String=file_extension)
  return largeLenv(Ns,label=label,names=names,ext=ext,type=type)
end
export largeLenv

"""
    G = largeRenv(renv[,label="Renv_",names=[label*"i" for i = 1:length(renv)],ext=".dmrjulia"])

Writes tensors from environment `renv` to hard disk as retrieved through `G` according to filenames specified in `names` (default file extension: `.dmrjulia`)

See also: [`largeMPS`](@ref) [`largeMPO`](@ref) [`largeLenv`](@ref) [`largeEnv`](@ref)
"""
function largeRenv(renv::P;label::String="Renv_",names::Array{String,1}=[label*"$i" for i = 1:length(renv)],ext::String=file_extension) where P <: Union{Array,Env}
  return largeLenv(renv,label=label,names=names,ext=ext)
end

"""
    G = largeRenv(Ns[,label="Renv_",names=[label*"i" for i = 1:Ns],ext=".dmrjulia"])

Creates a large environment type (stored on disk) with `Ns` tensors (element type: Float64) but does not write anything to disk initially. Filenames specified in `names` (default file extension: `.dmrjulia`)

See also: [`largeMPS`](@ref) [`largeMPO`](@ref) [`largeLenv`](@ref) [`largeEnv`](@ref)
"""
function largeRenv(Ns::Integer;label::String="Renv_",names::Array{String,1}=[label*"$i" for i = 1:Ns],ext::String=file_extension,type::DataType=Float64)
  return largeenvironment(names,type)
end

"""
    G = largeRenv(T,Ns[,label="Renv_",names=[label*"i" for i = 1:Ns],ext=".dmrjulia"])

Creates a large environment type (stored on disk) of element type `T` with `Ns` tensors but does not write anything to disk initially. Filenames specified in `names` (default file extension: `.dmrjulia`)

See also: [`largeMPS`](@ref) [`largeMPO`](@ref) [`largeLenv`](@ref) [`largeEnv`](@ref)
"""
function largeRenv(type::DataType,Ns::Integer;label::String="Renv_",names::Array{String,1}=[label*"$i" for i = 1:Ns],ext::String=file_extension)
  return largeRenv(Ns,label=label,names=names,ext=ext,type=type)
end
export largeRenv

"""
    G,K = largeEnv(lenv,renv[,Llabel="Lenv_",Lnames=[label*"i" for i = 1:length(psi)],Rlabel="Renv_",Rnames=[label*"i" for i = 1:length(renv)],ext=".dmrjulia",type=Float64])

Writes tensors from environments `lenv` and `renv` to hard disk as retrieved through `G` and `K` respectively according to filenames specified in `Lnames` and `Rnames` (default file extension: `.dmrjulia`)

See also: [`largeMPS`](@ref) [`largeMPO`](@ref) [`largeLenv`](@ref) [`largeRenv`](@ref)
"""
function largeEnv(lenv::P,renv::P;Llabel::String="Lenv_",Lnames::Array{String,1}=[Llabel*"$i" for i = 1:length(lenv)],Rlabel::String="Renv_",Rnames::Array{String,1}=[Rlabel*"$i" for i = 1:length(renv)],ext::String=file_extension,type::DataType=Float64) where P <: Union{Array,Env}
  return largeLenv(lenv,names=Lnames),largeRenv(renv,names=Rnames)
end

"""
    G,K = largeEnv(Ns[,Llabel="Lenv_",Lnames=[label*"i" for i = 1:Ns],Rlabel="Renv_",Rnames=[label*"i" for i = 1:Ns],ext=".dmrjulia",type=Float64])

Creates large environments with `Ns` tensors and retrieved through `G` and `K` respectively according to filenames specified in `Lnames` and `Rnames` (default file extension: `.dmrjulia`)

See also: [`largeMPS`](@ref) [`largeMPO`](@ref) [`largeLenv`](@ref) [`largeRenv`](@ref)
"""
function largeEnv(Ns::Integer;Llabel::String="Lenv_",Lnames::Array{String,1}=[Llabel*"$i" for i = 1:Ns],Rlabel::String="Renv_",Rnames::Array{String,1}=[Rlabel*"$i" for i = 1:Ns],ext::String=file_extension,type::DataType=Float64)
  return largeenvironment(Lnames,type),largeenvironment(Rnames,type)
end

"""
    G,K = largeEnv(T,Ns[,Llabel="Lenv_",Lnames=[label*"i" for i = 1:Ns],Rlabel="Renv_",Rnames=[label*"i" for i = 1:Ns],ext=".dmrjulia"])

Creates large environments with `Ns` tensors of type `T` and retrieved through `G` and `K` respectively according to filenames specified in `Lnames` and `Rnames` (default file extension: `.dmrjulia`)

See also: [`largeMPS`](@ref) [`largeMPO`](@ref) [`largeLenv`](@ref) [`largeRenv`](@ref)
"""
function largeEnv(type::DataType,Ns::Integer;Llabel::String="Lenv_",Lnames::Array{String,1}=[Llabel*"$i" for i = 1:Ns],Rlabel::String="Renv_",Rnames::Array{String,1}=[Rlabel*"$i" for i = 1:Ns],ext::String=file_extension)
  return largeEnv(Ns,Llabel=Llabel,Lnames=Lnames,Rlabel=Rlabel,Rnames=Rnames,ext=ext,type=type)
end
export largeEnv





"""
    G = loadLenv(Ns[,label="Lenv_",names=[label*"i" for i = 1:Ns],ext=".dmrjulia"])

If left environment tensors are stored on hard disk, then they can be retrieved by using `loadLenv`
"""
function loadLenv(Ns::Integer;label::String="Lenv_",names::Array{String,1}=[label*"$i" for i = 1:Ns],ext::String=file_extension,oc::Integer=0)
  lastnum = 1
  for i = 1:Ns
    name = names[i]
    A = tensorfromdisc(name,ext=ext)
    lastnum *= eltype(A)(1)
  end
  thistype = typeof(lastnum)
  return largeenvironment(names,thistype)
end
export loadLenv

"""
    G = loadRenv(Ns[,label="Renv_",names=[label*"i" for i = 1:Ns],ext=".dmrjulia"])

If right environment tensors are stored on hard disk, then they can be retrieved by using `loadRenv`
"""
function loadRenv(Ns::Integer;label::String="Renv_",names::Array{String,1}=[label*"$i" for i = 1:Ns],ext::String=file_extension,oc::Integer=0)
  lastnum = 1
  for i = 1:Ns
    name = names[i]
    A = tensorfromdisc(name,ext=ext)
    lastnum *= eltype(A)(1)
  end
  thistype = typeof(lastnum)
  return largeenvironment(names,thistype)
end
export loadRenv

"""
    G,K = loadEnv(Ns[,Llabel="Lenv_",Lnames=[label*"i" for i = 1:Ns],Rlabel="Renv_",Rnames=[label*"i" for i = 1:Ns],ext=".dmrjulia"])

If environment tensors are stored on hard disk, then they can be retrieved by using `loadEnv`
"""
function loadEnv(Ns::Integer;Llabel::String="Lenv_",Lnames::Array{String,1}=[Llabel*"$i" for i = 1:Ns],Rlabel::String="Renv_",Rnames::Array{String,1}=[Rlabel*"$i" for i = 1:Ns],ext::String=file_extension,oc::Integer=0)
  lastnumL = 1
  lastnumR = 1
  for i = 1:Ns
    name = Lnames[i]
    A = tensorfromdisc(name,ext=ext)
    lastnumL *= eltype(A)(1)

    B = tensorfromdisc(name,ext=ext)
    lastnumR *= eltype(B)(1)
  end
  thistypeL = typeof(lastnumL)
  thistypeR = typeof(lastnumR)
  return largeenvironment(Lnames,thistypeL),largeenvironment(Rnames,thistypeR)
end
export loadEnv

