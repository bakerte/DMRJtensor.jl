
"""
    applylocalF!(tens, i)

(in-place) effectively apply a chain of F (fermion phase) operator by a single manipulation on the specified index.

See also: [`applylocalF`](@ref)
"""
function applylocalF!(M::R, i) where {R <: qarray}
  for (j, (t, index)) in enumerate(zip(M.T, M.ind))
    pos = ind2pos(index, size(M))
    p = parity(getQnum(i,pos[i],tens))
    M.T[j] *= (-1)^p
  end
end

"""
    applylocalF(tens, i)

effectively apply a chain of F (fermion phase) operator by a single manipulation on the specified index.

See also: [`applylocalF!`](@ref)
"""
function applylocalF(tens::R, i::Integer) where {R <: qarray}
  W = copy(tens)
  applylocalF!(W, i)
  return W 
end
export applylocalF,applylocalF!