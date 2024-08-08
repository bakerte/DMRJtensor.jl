#########################################################################
#
#  Density Matrix Renormalization Group (and other methods) in julia (DMRjulia)
#                              v1.0
#
#########################################################################
# Made by Thomas E. Baker and « les qubits volants » (2024)
# See accompanying license with this program
# This code is native to the julia programming language (v1.10.4+)
#


"""
    MPS

Abstract types for MPS
"""
abstract type MPS <: TNnetwork end
export MPS

"""
    MPO
  
Abstract types for MPO
"""
abstract type MPO <: TNnetwork end
export MPO

"""
    MERA
  
Abstract types for MERA
"""
abstract type MERA <: TNnetwork end
export MERA

"""
    regMPS

Abstract types for regMPS
"""
abstract type regMPS <: MPS end
export regMPS

"""
    regMPO
  
Abstract types for regMPO
"""
abstract type regMPO <: MPO end
export regMPO

"""
    envType

Vector that holds environments (rank N)
"""
abstract type envType end
export envType

abstract type regEnv <: envType end
export regEnv

"""
    Env

A global super-type for the environment defined either with `envType` or `AbstractArray`
"""
const Env = Union{Array,envType}
export Env


"""
    matrixproductstate

Contruct this through `MPS`. struct to hold regMPS tensors and orthogonality center

# Fields:
+ `A::Array{TensType,1}`: vector of MPS tensors
+ `oc::Integer`: orthogonality center

See also: [`MPS`](@ref)
"""
mutable struct matrixproductstate{W} <: regMPS where W <: TensType
  A::network{W}
  oc::Integer
end

"""
    matrixproductoperator
  
Contruct this through `MPO`. Struct to hold MPO tensors

# Fields:
+ `H::Array{TensType,1}`: vector of MPO tensors

See also: [`MPO`](@ref)
"""
struct matrixproductoperator{W} <: regMPO where W <: TensType
  H::network{W}
end

"""
    environment

Construct this object through `Env`. Array that holds environment tensors

# Fields:
+ `V::Array{TensType,1}`: vector of environment tensors

See also: [`Env`](@ref)
"""
struct environment{W} <: regEnv where W <: TensType
  V::network{W}
end

"""
    file_extension

A default file extension can be specified for the large types.

See also: [`largeMPS`](@ref) [`largeMPO`](@ref) [`largeEnv`](@ref)
"""
const file_extension = ".dmrjulia"

"""
    largeMPS

Abstract types for `largeMPS`; saves tensors to disk on setting values and retrieves them on calling an element; slower than other `regMPS` form

See also: [`largeMPO`](@ref) [`largeEnv`](@ref)
"""
abstract type largeMPS <: MPS end
export largeMPS

"""
    largeMPO
  
Abstract types for `largeMPO`; saves tensors to disk on setting values and retrieves them on calling an element; slower than other `regMPO` form

See also: [`largeEnv`](@ref) [`largeMPO`](@ref)
"""
abstract type largeMPO <: MPO end
export largeMPO

"""
    largeEnv
  
Abstract types for `largeEnv`; saves tensors to disk on setting values and retrieves them on calling an element; slower than other `regEnv` form

See also: [`largeMPS`](@ref) [`largeMPO`](@ref)
"""
abstract type largeEnv <: envType end
export largeEnv

"""
    largeType
  
A union of the types `largeMPS`, `largeMPO`, and `largeEnv`

See also: [`largeMPS`](@ref) [`largeMPO`](@ref) [`largeEnv`](@ref)
"""
const largeType = Union{largeMPS,largeMPO,largeEnv}
export largeType

"""
    largematrixproductstate
 
Construct this container with `largeMPS`. struct to hold `largeMPS` tensors and orthogonality center

# Fields:
+ `A::Array{String,1}`: filenames where the tensors are stored on disk
+ `oc::Integer`: orthogonality center
+ `type::DataType`: DataType of the stored tensors

See also: [`largeMPS`](@ref)
"""
mutable struct largematrixproductstate{W} <: largeMPS where W <: DataType
  A::Array{String,1}
  oc::intType
  type::W
end

"""
    largematrixproductoperator
  
Construct this container with `largeMPO`. struct to hold `largeMPO` tensors

# Fields:
+ `H::Array{String,1}`: filenames where the tensors are stored on disk
+ `type::DataType`: DataType of the stored tensors

See also: [`largeMPO`](@ref)
"""
struct largematrixproductoperator{W} <: largeMPO where W <: DataType
  H::Array{String,1}
  type::W
end

"""
    largeenvironment
  
Construct this container with `largeEnv`. struct to hold `largeEnv` tensors

# Fields:
+ `V::Array{String,1}`: filenames where the tensors are stored on disk
+ `type::DataType`: DataType of the stored tensors

See also: [`largeEnv`](@ref)
"""
struct largeenvironment{W} <: largeEnv where W <: DataType
  V::Array{String,1}
  type::W
end
