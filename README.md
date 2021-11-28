# DMRjulia (DMRJtensor.jl)
DMRjulia is a general purpose tensor network library in the julia programming language

Hello! Welcome to the site of the future full release of DMRjulia, a tensor network library for physics computations written in julia. Right now, ground state calculations on the MPS are available (v0.8.7) for both quantum numbers (mostly documented) and dense tensors (fully documented).

DMRjulia is now official registered in julia's package library! Simply type the following into the julia terminal.

```
julia> ]
pkg> add DMRJtensor
```

Then, typing `using DMRJtensor` will give you access to all the functions in the library.
  
Troubleshooting? Email: thomas.baker@usherbrooke.ca

If you're just starting out, I recommend reading the introduction article at http://arxiv.org/abs/1911.11566 (English starting on page 19, accepted at the Canadian Journal of Physics) and beginning in the /examples folder.

If you want to view documentation in the code, open the julia terminal, type `?` and type the name of the function.  There is a lot of good information that can help you there.

## Papers and documentation:

### Tensor recipes for algorithms in DMRjulia

T.E. Baker, et. al. "Tensor Recipes for entanglement renormalization computations" arxiv:2112.XXXX  (2021)

### Introduction to tensor networks:

[Français] T.E. Baker, S. Desrosiers, M. Tremblay, M. Thompson, "Méthodes de calcul avec réseaux de tenseurs en physique" Can. J. Phys. 99, 4 (2021); https://cdnsciencepub.com/doi/abs/10.1139/cjp-2019-0611

[English] ibid., "Basic tensor network computations in physics" https://arxiv.org/abs/1911.11566 (2019), pp. 19

### Introduction to DMRG: 

T.E. Baker, M.P. Thompson "Build your own tensor network library: DMRjulia I. Basic library for the density matrix renormalization group" https://arxiv.org/abs/2109.03120 (2021)
