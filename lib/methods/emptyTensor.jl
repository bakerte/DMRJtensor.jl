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
    B = emptyTensor(A)

Generates a tensor with no inputs from an input array of `Qtens`
"""
function emptyTensor(A::Array{Qtens{W,Q},1}) where {W <: Number, Q <: Qnum}
  return Qtens{W,Q}()
end

"""
    B = emptyTensor(A)

Generates a tensor with no inputs from an input array of `tens`
"""
function emptyTensor(A::Array{tens{W},1}) where W <: Number
  return tens{W}()
end

"""
    B = emptyTensor(A)

Generates a tensor with no inputs from an input array of `Array`
"""
function emptyTensor(A::Array{Array{W,1},1}) where W <: Number
  return undefMat(W,0,0)
end

"""
    B = emptyTensor(A)

Generates a tensor with no inputs from an input `network` of `Qtens`
"""
function emptyTensor(A::network{Vector{Qtens{W,Q}}}) where {W <: Number, Q <: Qnum}
  return Qtens{W,Q}()
end

"""
    B = emptyTensor(A)

Generates a tensor with no inputs from an input `network` of `tens`
"""
function emptyTensor(A::network{Vector{tens{W}}}) where W <: Number
  return tens{W}()
end

"""
    B = emptyTensor(A)

Generates a tensor with no inputs from an input `network` of `Array`
"""
function emptyTensor(A::network{W}) where W <: TensType
  outtype = eltype(W) <: Any ? Float64 : eltype(W)
  return undefMat(outtype,0,0)
end