![alt text](https://github.com/bakerte/DMRJtensor.jl/blob/main/dmrjulia.png?raw=true)

# DMRjulia (DMRJtensor.jl)

Hello! Welcome to the site of DMRjulia, a tensor network library for physics computations written in Julia. We are beta-testing the full release of v1.0 which will contain all algorithms needed for cutting edge research. DMRjulia is fast and easy to use for making new algorithms or solving systems.

DMRjulia is official registered in Julia's package library, making installation simple. Just type the following into the Julia terminal.

```
julia> ]
pkg> add DMRJtensor
```

Then, typing `using DMRJtensor` will give you access to all the functions in the library.
  
Troubleshooting? Email: bakerte@uvic.ca

If you're just starting out, I recommend reading the introduction article at listed below (available in French or English) and beginning in the /examples folder.

If you want to view documentation in the code, open the Julia terminal, type `?` and type the name of the function. For example,
```
julia> ?
help?> DMRJtensor
```
There is a lot of good information that can help you there.

## Papers and documentation:

### Introduction to tensor networks:

[Français] T.E. Baker, S. Desrosiers, M. Tremblay, M. Thompson, "Méthodes de calcul avec réseaux de tenseurs en physique" Can. J. Phys. 99, 4 (2021); https://cdnsciencepub.com/doi/abs/10.1139/cjp-2019-0611

[English] ibid., "Basic tensor network computations in physics" https://arxiv.org/abs/1911.11566 (2019), pp. 20

### Introduction to DMRG: 

T.E. Baker, M.P. Thompson "Build your own tensor network library: DMRjulia I. Basic library for the density matrix renormalization group" https://arxiv.org/abs/2109.03120 (2021)

### Tensor recipes for algorithms in DMRjulia

T.E. Baker, et. al. "Tensor Recipes for entanglement renormalization computations" arxiv:2111.14530  (2021)

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
