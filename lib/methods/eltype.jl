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
    G = eltype(Y)

`eltype` gets element type `G` of the `MPS`
"""
 function eltype(Y::regMPS)
  return eltype(Y.A[1])
end

"""
    G = eltype(Y)

`eltype` gets element type `G` of the `MPO`
"""
function eltype(H::regMPO)
  return eltype(H.H[1])
end

"""
    G = eltype(Y)

`eltype` gets element type `G` of the `Env`
"""
function eltype(G::regEnv)
  return eltype(G.V[1])
end

#  import .tensor.elnumtype
"""
    G = elnumtype(op...)

Gives type `G` of input containers `op`, a tuple containing `MPS`, `MPO`, or `Env` types
"""
function elnumtype(op...)
  opnum = eltype(op[1])(1)
  for b = 2:length(op)
    opnum *= eltype(op[b])(1)
  end
  return typeof(opnum)
end

"""
    G = eltype(Y)

`eltype` gets element type `G` of any `largeType` network

See also: [`largeType`](@ref)
"""
function eltype(op::largeType)
  return op.type
end
