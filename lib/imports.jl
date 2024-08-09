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
using TensorPACK


#import .TensorPACK.TensType

import TensorPACK.undefMat,TensorPACK.checkType,TensorPACK.tensorcombination,TensorPACK.tensorcombination!
import TensorPACK.permutedims

import TensorPACK.invmat!

import TensorPACK.reshape!,TensorPACK.unreshape!,TensorPACK.unreshape

import TensorPACK.get_denseranges

import TensorPACK.sub!,TensorPACK.add!,TensorPACK.mult!,TensorPACK.div!
import TensorPACK.joinindex!
import TensorPACK.getindex!
import TensorPACK.searchindex

import TensorPACK.libmult,TensorPACK.libsvd,TensorPACK.libsvd!,TensorPACK.libeigen,TensorPACK.libeigen!,TensorPACK.libqr,TensorPACK.liblq,TensorPACK.libqr!,TensorPACK.liblq!



#contracions
import TensorPACK.contract
import TensorPACK.ccontract
import TensorPACK.contractc
import TensorPACK.ccontractc



import TensorPACK.dot
import TensorPACK.libmult
import TensorPACK.dmul!
import TensorPACK.diagcontract!


import TensorPACK.permq
import TensorPACK.willperm
import TensorPACK.prepareT
import TensorPACK.getsizes
import TensorPACK.maincontractor

import TensorPACK.trace

#decompositions
import TensorPACK.libsvd
import TensorPACK.libsvd!


import TensorPACK.libeigen
import TensorPACK.libeigen!

import TensorPACK.sqrt!

import TensorPACK.libqr
import TensorPACK.libqr!

import TensorPACK.liblq
import TensorPACK.liblq!


import TensorPACK.defzero
import TensorPACK.basedims
import TensorPACK.makeHpsi


import TensorPACK.root


import TensorPACK.truncate,TensorPACK.svd,TensorPACK.svd!,TensorPACK.getorder,TensorPACK.findsize,TensorPACK.svdvals,TensorPACK.eigen,TensorPACK.eigen!,TensorPACK.eigvals,TensorPACK.eigvals!,TensorPACK.qr,TensorPACK.qr!,TensorPACK.lq,TensorPACK.lq!,TensorPACK.polar,.TensorPACK.eye


import TensorPACK.network, TensorPACK.TNnetwork


import TensorPACK.largevector, TensorPACK.tensor2disc, TensorPACK.tensorfromdisc


import TensorPACK.stdavg
import TensorPACK.testfct

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

#import .TensorPACK.tens
#export tens


import TensorPACK.convertTens



#import TensorPACK.sin
#import TensorPACK.cos
