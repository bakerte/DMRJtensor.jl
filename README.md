# DMRjulia (DMRJtensor.jl)
DMRjulia is a general purpose tensor network library in the Julia programming language

Hello! Welcome to the site of DMRjulia, a tensor network library for physics computations written in Julia. Right now, ground state calculations on the MPS are available (v0.9.2+) for both quantum numbers (mostly documented) and dense tensors (fully documented).

DMRjulia is official registered in Julia's package library, making installation simple. Just type the following into the Julia terminal.

```
julia> ]
pkg> add DMRJtensor
```

Then, typing `using DMRJtensor` will give you access to all the functions in the library.
  
Troubleshooting? Email: bakerte@uvic.ca

If you're just starting out, I recommend reading the introduction article at http://arxiv.org/abs/1911.11566 (English starting on page 19, accepted at the Canadian Journal of Physics) and beginning in the /examples folder.

If you want to view documentation in the code, open the Julia terminal, type `?` and type the name of the function. For example,
```
julia> ?
help?> DMRJtensor
```
There is a lot of good information that can help you there.

## Papers and documentation:

### Tensor recipes for algorithms in DMRjulia

T.E. Baker, et. al. "Tensor Recipes for entanglement renormalization computations" arxiv:2111.14530  (2021)

### Introduction to tensor networks:

[Français] T.E. Baker, S. Desrosiers, M. Tremblay, M. Thompson, "Méthodes de calcul avec réseaux de tenseurs en physique" Can. J. Phys. 99, 4 (2021); https://cdnsciencepub.com/doi/abs/10.1139/cjp-2019-0611

[English] ibid., "Basic tensor network computations in physics" https://arxiv.org/abs/1911.11566 (2019), pp. 19

### Introduction to DMRG: 

T.E. Baker, M.P. Thompson "Build your own tensor network library: DMRjulia I. Basic library for the density matrix renormalization group" https://arxiv.org/abs/2109.03120 (2021)

## Using DMRjulia in Python

The installation of Julia is straightforwardly done with a simple downloaded file at https://julialang.org/downloads/. Simply select the operating system applicable to you and follow the instructions above to install the `DMRJtensor` package.

Just in case it is needed, the package can be used in Python.  If these details are out of date, there are many useful help articles on this topic and let the authors know if you run into trouble with these instructions.  This will be written for a MacOS user as of 22/03/22.

 + Install Xcode (recommend downloading .xip file from https://developer.apple.com/download/all/?q=Xcode)
 + Apply command line tools with "xcode-select --install" if not already installed
 + Install Homebrew: /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install.sh)"
 + Type this into terminal: export PATH="/usr/local/opt/python/libexec/bin:$PATH"
 + install Python: brew install python

Now open Python and install the necessary `julia` packages.  In the test run, the tester had to install the `PyCall` package in Julia (`] add PyCall`)
 + python3 -m pip install --user julia
 + python3
 + import julia
 + julia.install()
 + import julia.DMRJtensor
