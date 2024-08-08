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


import LinearAlgebra
import Printf
import Serialization

import Distributed












import .Base.getindex
import .Base.setindex!
import .Base.copy
import .Base.size
import .Base.conj
import .Base.conj!
import .Base.eltype
import .Base.lastindex
import .Base.sum
import .Base.permutedims!
import .LinearAlgebra.norm



#using TensorPACK
using ..TENPACK


#import .TENPACK.TensType

import ..TENPACK.undefMat,..TENPACK.checkType,..TENPACK.tensorcombination,..TENPACK.tensorcombination!
import ..TENPACK.permutedims

import ..TENPACK.invmat!

import ..TENPACK.reshape!,..TENPACK.unreshape!,..TENPACK.unreshape

import ..TENPACK.get_denseranges

import ..TENPACK.sub!,..TENPACK.add!,..TENPACK.mult!,..TENPACK.div!
import ..TENPACK.joinindex!
import ..TENPACK.getindex!
import ..TENPACK.searchindex

import ..TENPACK.libmult,..TENPACK.libsvd,..TENPACK.libsvd!,..TENPACK.libeigen,..TENPACK.libeigen!,..TENPACK.libqr,..TENPACK.liblq,..TENPACK.libqr!,..TENPACK.liblq!



#contracions
import ..TENPACK.contract
import ..TENPACK.ccontract
import ..TENPACK.contractc
import ..TENPACK.ccontractc



import ..TENPACK.dot
import ..TENPACK.libmult
import ..TENPACK.dmul!
import ..TENPACK.diagcontract!


import ..TENPACK.permq
import ..TENPACK.willperm
import ..TENPACK.prepareT
import ..TENPACK.getsizes
import ..TENPACK.maincontractor

import ..TENPACK.trace

#decompositions
import ..TENPACK.libsvd
import ..TENPACK.libsvd!


import ..TENPACK.libeigen
import ..TENPACK.libeigen!

import ..TENPACK.sqrt!

import ..TENPACK.libqr
import ..TENPACK.libqr!

import ..TENPACK.liblq
import ..TENPACK.liblq!


import ..TENPACK.defzero
import ..TENPACK.basedims
import ..TENPACK.makeHpsi


import ..TENPACK.root


import ..TENPACK.truncate,..TENPACK.svd,..TENPACK.svd!,..TENPACK.getorder,..TENPACK.findsize,..TENPACK.svdvals,..TENPACK.eigen,..TENPACK.eigen!,..TENPACK.eigvals,..TENPACK.eigvals!,..TENPACK.qr,..TENPACK.qr!,..TENPACK.lq,..TENPACK.lq!,..TENPACK.polar,...TENPACK.eye


import ..TENPACK.network, ..TENPACK.TNnetwork


import ..TENPACK.largevector, ..TENPACK.tensor2disc, ..TENPACK.tensorfromdisc


import ..TENPACK.stdavg
import ..TENPACK.testfct

#=
import TensorPACK.undefMat,TensorPACK.makeArray,TensorPACK.checkType,TensorPACK.tensorcombination,TensorPACK.tensorcombination!
import TensorPACK.makeId

import TensorPACK.reshape!,TensorPACK.convertTens

import TensorPACK.makedens
import TensorPACK.get_denseranges

import TensorPACK.sub!,TensorPACK.add!,TensorPACK.mult!,TensorPACK.div!
import TensorPACK.joinindex!
import TensorPACK.getindex!
import TensorPACK.searchindex

import TensorPACK.libmult,TensorPACK.libsvd,TensorPACK.libsvd!,TensorPACK.libeigen,TensorPACK.libeigen!,TensorPACK.libqr,TensorPACK.liblq,TensorPACK.libqr!,TensorPACK.liblq!
=#
import Base.*,Base.-,Base./
import Base.length,Base.adjoint

#import .TENPACK.tens
#export tens


import ..TENPACK.convertTens



#import ..TENPACK.sin
#import ..TENPACK.cos
