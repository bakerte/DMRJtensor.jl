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
    net = network(input[,groundlevel=1])

Produces `net` from a set of input types (any number) `input`; `groundlevel` gives the number of the bottom level for MPS-MPO systems

#Examples:
julia> network(dualpsi,psi,mpo) #generates a network of `nametens`; works for any number of MPOs
julia> network(dualpeps,peps,mpdo) #same functionality as for the MPO case
julia> network(mpo,mera) #creates a MERA network on top of an MPO
"""
function network(input::TNnetwork...;groundlevel::intType=1)


  notMERA = true
  w = 0
  while w < length(input) && notMERA
    w += 1 
    notMERA &= !(typeof(input[w]) <: MERA)
  end


  nets = [input[w] for w = 1:length(input)]
  if notMERA
    levels = [get_tensors(nets[w]).level for w = 1:length(nets)]
    sort!(levels,rev=true)

    minlevel = minimum(levels)

    sorted = true
    w = 0
    while w < length(levels) && sorted
      w += 1
      sorted &= levels[w] == w
    end

    if !sorted
      #first MPS/PEPS
      counter = 1
      while counter < length(nets) && (!(typeof(nets[counter]) <: MPS) || minlevel != get_tensors(nets[counter]).level)
        counter += 1
      end
      nets[counter].A.level = groundlevel

      #MPOs/MPDOs
      currlevel = groundlevel
      for w = 1:length(nets)
        if typeof(nets[w]) <: MPO
          currlevel += 1
          nets[w].H.level = currlevel
        end
      end

      #final MPS/PEPS
      for w = 1:length(nets)
        if w != counter && typeof(nets[w]) <: MPS
          nets[w].A.level = currlevel + 1
        end
      end

      for w = 1:length(nets)
        for x = 1:length(nets[w]) #sites
          thislevel = get_tensors(nets[w]).level
          stringlevel = string(get_tensors(nets[w]).level)
          for y = 1:ndims(nets[w][x]) #index
            if nets[w][x].arrows[y] == 0
              nets[w][x].names[y] = stringlevel * nets[w][x].names[y]
            else
              p = thislevel + (nets[w][x].arrows[y] == 1 ? nets[w][x].arrows[y] : 0)
              nets[w][x].names[y] = string(p) * nets[w][x].names[y]
              nets[w][x].names[y] *= string(x)
            end
          end
        end
      end

    end

    newnet = network(vcat([[nametens(nets[w][y]) for y = 1:length(nets[w])] for w = 1:length(nets)]...))

  else

    nMPOs = 0
    for p = 1:length(nets)
      nMPOs += typeof(nets[p]) <: MPO
    end
    if nMPOs > 1
      error("Please input a single MPO into a network with MERAs (you can contract multiple MPOs to form a single MPO or generate a single MPO at the output)")
    else
      p = 0
      while p < length(nets) && typeof(nets[p+1]) != MPO
        p += 1
      end
      if typeof(nets[p]) <: MPO
        nets[p].level = 0
      end
    end


    ntens = sum(w->sum(x->length(nets[w][x]),length(nets[w])),length(nets))

    newnet_tensors = directedtens[]

    for w = 1:length(nets) #network input (i.e., MERA)
      meranet = get_tensors(nets[w])
      for x = 1:length(nets[w]) #layer
        for y = 1:length(nets[w][x]) #tensors
          for p = 1:length(nets[w][x][y].names)
            nets[w][x][y].names[p] = string(abs(meranet[x].level)) * nets[w][x][y].names[p]
          end
          push!(newnet_tensors,nets[w][x][y])
        end
      end
    end

    newnet = network(newnet_tensors)

  end

  return newnet

  #=
  for w = 1:length(nets)
    if typeof(nets[w]) <: MPS
      counter[1] += 1
    elseif typeof(nets[w]) <: MPO
      counter[2] += 1
#    elseif typeof(nets[w]) <: PEPS
#    elseif typeof(nets[w]) <: MERA
    end
  end
  =#
end