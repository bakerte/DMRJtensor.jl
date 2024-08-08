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


#       +---------------------------------------+
#>------+       Constructing MPO operators      +---------<
#       +---------------------------------------+

  #converts an array to an MPO so that it is instead of being represented in by an array,
  #it is represented by a tensor diagrammatically as
  #
  #       s2
  #       |
  # a1 -- W -- a2       =    W[a1,s1,s2,a2]
  #       |
  #       s1
  #
  #The original Hamiltonian matrix H in the DMRjulia.jl file is of the form
  #
  # H = [ W_11^s1s2  W_12^s1s2 W_13^s1s2 ....
  #       W_21^s1s2  W_22^s1s2 W_23^s1s2 ....
  #       W_31^s1s2  W_32^s1s2 W_33^s1s2 ....
  #       W_41^s1s2  W_42^s1s2 W_43^s1s2 ....]
  #where each W occupies the equivalent of (vars.qstates X vars.qstates) sub-matrices in
  #of the H matrix as recorded in each s1s2 pair.  These are each of the operators in H.

"""
    makeMPO(H,physSize,Ns[,infinite=,lower=])

Converts function or vector (`H`) to each of `Ns` MPO tensors; `physSize` can be a vector (one element for the physical index size on each site) or a number (uniform sites); `lower` signifies the input is the lower triangular form (default)

# Example:

```julia
spinmag = 0.5;Ns = 10
Sx,Sy,Sz,Sp,Sm,O,Id = spinOps(spinmag)
function H(i::Integer)
    return [Id O;
            Sz Id]
end
isingmpo = makeMPO(H,size(Id,1),Ns)
```
"""
function makeMPO(H::Array{Array{X,2},1},physSize::Array{Y,1};
                      lower::Bool=true,regtens::Bool=false) where {X <: Number, Y <: Integer}
  Ns = length(H)
  retType = typeof(prod(a->eltype(H[a])(1),1:Ns))
  finalMPO = Array{Array{retType,4},1}(undef,Ns)
  for i = 1:Ns
    thisH = lower ? H[i] : transpose(H[i])
    states = physSize[(i-1) % size(physSize,1) + 1]
    a1size = div(size(thisH,1),states) #represented in LEFT link indices
    a2size = div(size(thisH,2),states) #represented in RIGHT link indices

    G = Array{retType,4}(undef,a1size,states,states,a2size)
    
    for m = 1:a2size
      for k = 1:states
        for j = 1:states
          @inbounds @simd for l = 1:a1size
            G[l,j,k,m] = thisH[j + (l-1)*states, k + (m-1)*states]
          end
        end
      end
    end
    
    finalMPO[i] = G
  end
  if lower
    finalMPO[1] = finalMPO[1][end:end,:,:,:]
    finalMPO[end] = finalMPO[end][:,:,:,1:1]
  else
    finalMPO[1] = finalMPO[1][1:1,:,:,:]
    finalMPO[end] = finalMPO[end][:,:,:,end:end]
  end
  return MPO(retType,finalMPO,regtens=regtens)
end

function makeMPO(H::Array{X,2},physSize::Array{Y,1},Ns::Integer;
                      lower::Bool=true,regtens::Bool=false) where {X <: Number, Y <: Integer}
  return makeMPO([H for i = 1:Ns],physSize)
end

function makeMPO(H::Array{X,2},physSize::Y,Ns::Integer;
                      lower::Bool=true,regtens::Bool=false) where {X <: Number, Y <: Integer}
  return makeMPO([H for i = 1:Ns],[physSize])
end

function makeMPO(H::Array{X,1},physSize::Y,Ns::Integer;
                      lower::Bool=true,regtens::Bool=false) where {X <: Array, Y <: Integer}
  return makeMPO(H,[physSize],lower=lower,regtens=regtens)
end

function makeMPO(H::Array{X,1},physSize::Y;
                      lower::Bool=true,regtens::Bool=false) where {X <: Array, Y <: Integer}
  return makeMPO(H,[physSize],lower=lower,regtens=regtens)
end

function makeMPO(H::Function,physSize::Array{X,1},Ns::Integer;
                      lower::Bool=true,regtens::Bool=false) where X <: Integer
  thisvec = [H(i) for i = 1:Ns]
  return makeMPO(thisvec,physSize,lower=lower,regtens=regtens)
end

function makeMPO(H::Function,physSize::Integer,Ns::Integer;
                      lower::Bool=true,regtens::Bool=false)
  return makeMPO(H,[physSize],Ns,lower=lower,regtens=regtens)
end
export makeMPO