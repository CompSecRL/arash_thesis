Ы╩<
я┤
D
AddV2
x"T
y"T
z"T"
Ttype:
2	ђљ
B
AssignVariableOp
resource
value"dtype"
dtypetypeѕ
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
Џ
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
>
Maximum
x"T
y"T
z"T"
Ttype:
2	
Ї
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(ѕ
?
Mul
x"T
y"T
z"T"
Ttype:
2	љ

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetypeѕ
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0ѕ
.
Rsqrt
x"T
y"T"
Ttype:

2
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0ѕ
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
-
Sqrt
x"T
y"T"
Ttype:

2
3
Square
x"T
y"T"
Ttype:
2
	
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
Й
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ѕ
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	
ї
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
ќ
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ѕ"serve*2.6.22v2.6.1-9-gc2363d6d0258─Ј9
`
beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namebeta_1
Y
beta_1/Read/ReadVariableOpReadVariableOpbeta_1*
_output_shapes
: *
dtype0
`
beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namebeta_2
Y
beta_2/Read/ReadVariableOpReadVariableOpbeta_2*
_output_shapes
: *
dtype0
^
decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedecay
W
decay/Read/ReadVariableOpReadVariableOpdecay*
_output_shapes
: *
dtype0
n
learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namelearning_rate
g
!learning_rate/Read/ReadVariableOpReadVariableOplearning_rate*
_output_shapes
: *
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
ї
stream_0_conv_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*'
shared_namestream_0_conv_1/kernel
Ё
*stream_0_conv_1/kernel/Read/ReadVariableOpReadVariableOpstream_0_conv_1/kernel*"
_output_shapes
:@*
dtype0
ђ
stream_0_conv_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_namestream_0_conv_1/bias
y
(stream_0_conv_1/bias/Read/ReadVariableOpReadVariableOpstream_0_conv_1/bias*
_output_shapes
:@*
dtype0
і
batch_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@**
shared_namebatch_normalization/gamma
Ѓ
-batch_normalization/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization/gamma*
_output_shapes
:@*
dtype0
ѕ
batch_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*)
shared_namebatch_normalization/beta
Ђ
,batch_normalization/beta/Read/ReadVariableOpReadVariableOpbatch_normalization/beta*
_output_shapes
:@*
dtype0
Ї
stream_0_conv_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@ђ*'
shared_namestream_0_conv_2/kernel
є
*stream_0_conv_2/kernel/Read/ReadVariableOpReadVariableOpstream_0_conv_2/kernel*#
_output_shapes
:@ђ*
dtype0
Ђ
stream_0_conv_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*%
shared_namestream_0_conv_2/bias
z
(stream_0_conv_2/bias/Read/ReadVariableOpReadVariableOpstream_0_conv_2/bias*
_output_shapes	
:ђ*
dtype0
Ј
batch_normalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*,
shared_namebatch_normalization_1/gamma
ѕ
/batch_normalization_1/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_1/gamma*
_output_shapes	
:ђ*
dtype0
Ї
batch_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*+
shared_namebatch_normalization_1/beta
є
.batch_normalization_1/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_1/beta*
_output_shapes	
:ђ*
dtype0
ј
stream_0_conv_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђђ*'
shared_namestream_0_conv_3/kernel
Є
*stream_0_conv_3/kernel/Read/ReadVariableOpReadVariableOpstream_0_conv_3/kernel*$
_output_shapes
:ђђ*
dtype0
Ђ
stream_0_conv_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*%
shared_namestream_0_conv_3/bias
z
(stream_0_conv_3/bias/Read/ReadVariableOpReadVariableOpstream_0_conv_3/bias*
_output_shapes	
:ђ*
dtype0
Ј
batch_normalization_2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*,
shared_namebatch_normalization_2/gamma
ѕ
/batch_normalization_2/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_2/gamma*
_output_shapes	
:ђ*
dtype0
Ї
batch_normalization_2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*+
shared_namebatch_normalization_2/beta
є
.batch_normalization_2/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_2/beta*
_output_shapes	
:ђ*
dtype0
y
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ђT*
shared_namedense_1/kernel
r
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes
:	ђT*
dtype0
p
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:T*
shared_namedense_1/bias
i
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes
:T*
dtype0
ј
batch_normalization_3/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:T*,
shared_namebatch_normalization_3/gamma
Є
/batch_normalization_3/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_3/gamma*
_output_shapes
:T*
dtype0
ї
batch_normalization_3/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:T*+
shared_namebatch_normalization_3/beta
Ё
.batch_normalization_3/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_3/beta*
_output_shapes
:T*
dtype0
ќ
batch_normalization/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*0
shared_name!batch_normalization/moving_mean
Ј
3batch_normalization/moving_mean/Read/ReadVariableOpReadVariableOpbatch_normalization/moving_mean*
_output_shapes
:@*
dtype0
ъ
#batch_normalization/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#batch_normalization/moving_variance
Ќ
7batch_normalization/moving_variance/Read/ReadVariableOpReadVariableOp#batch_normalization/moving_variance*
_output_shapes
:@*
dtype0
Џ
!batch_normalization_1/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*2
shared_name#!batch_normalization_1/moving_mean
ћ
5batch_normalization_1/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_1/moving_mean*
_output_shapes	
:ђ*
dtype0
Б
%batch_normalization_1/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*6
shared_name'%batch_normalization_1/moving_variance
ю
9batch_normalization_1/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_1/moving_variance*
_output_shapes	
:ђ*
dtype0
Џ
!batch_normalization_2/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*2
shared_name#!batch_normalization_2/moving_mean
ћ
5batch_normalization_2/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_2/moving_mean*
_output_shapes	
:ђ*
dtype0
Б
%batch_normalization_2/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*6
shared_name'%batch_normalization_2/moving_variance
ю
9batch_normalization_2/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_2/moving_variance*
_output_shapes	
:ђ*
dtype0
џ
!batch_normalization_3/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:T*2
shared_name#!batch_normalization_3/moving_mean
Њ
5batch_normalization_3/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_3/moving_mean*
_output_shapes
:T*
dtype0
б
%batch_normalization_3/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:T*6
shared_name'%batch_normalization_3/moving_variance
Џ
9batch_normalization_3/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_3/moving_variance*
_output_shapes
:T*
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
џ
Adam/stream_0_conv_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*.
shared_nameAdam/stream_0_conv_1/kernel/m
Њ
1Adam/stream_0_conv_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/stream_0_conv_1/kernel/m*"
_output_shapes
:@*
dtype0
ј
Adam/stream_0_conv_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_nameAdam/stream_0_conv_1/bias/m
Є
/Adam/stream_0_conv_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/stream_0_conv_1/bias/m*
_output_shapes
:@*
dtype0
ў
 Adam/batch_normalization/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*1
shared_name" Adam/batch_normalization/gamma/m
Љ
4Adam/batch_normalization/gamma/m/Read/ReadVariableOpReadVariableOp Adam/batch_normalization/gamma/m*
_output_shapes
:@*
dtype0
ќ
Adam/batch_normalization/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*0
shared_name!Adam/batch_normalization/beta/m
Ј
3Adam/batch_normalization/beta/m/Read/ReadVariableOpReadVariableOpAdam/batch_normalization/beta/m*
_output_shapes
:@*
dtype0
Џ
Adam/stream_0_conv_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@ђ*.
shared_nameAdam/stream_0_conv_2/kernel/m
ћ
1Adam/stream_0_conv_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/stream_0_conv_2/kernel/m*#
_output_shapes
:@ђ*
dtype0
Ј
Adam/stream_0_conv_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*,
shared_nameAdam/stream_0_conv_2/bias/m
ѕ
/Adam/stream_0_conv_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/stream_0_conv_2/bias/m*
_output_shapes	
:ђ*
dtype0
Ю
"Adam/batch_normalization_1/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*3
shared_name$"Adam/batch_normalization_1/gamma/m
ќ
6Adam/batch_normalization_1/gamma/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_1/gamma/m*
_output_shapes	
:ђ*
dtype0
Џ
!Adam/batch_normalization_1/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*2
shared_name#!Adam/batch_normalization_1/beta/m
ћ
5Adam/batch_normalization_1/beta/m/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_1/beta/m*
_output_shapes	
:ђ*
dtype0
ю
Adam/stream_0_conv_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђђ*.
shared_nameAdam/stream_0_conv_3/kernel/m
Ћ
1Adam/stream_0_conv_3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/stream_0_conv_3/kernel/m*$
_output_shapes
:ђђ*
dtype0
Ј
Adam/stream_0_conv_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*,
shared_nameAdam/stream_0_conv_3/bias/m
ѕ
/Adam/stream_0_conv_3/bias/m/Read/ReadVariableOpReadVariableOpAdam/stream_0_conv_3/bias/m*
_output_shapes	
:ђ*
dtype0
Ю
"Adam/batch_normalization_2/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*3
shared_name$"Adam/batch_normalization_2/gamma/m
ќ
6Adam/batch_normalization_2/gamma/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_2/gamma/m*
_output_shapes	
:ђ*
dtype0
Џ
!Adam/batch_normalization_2/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*2
shared_name#!Adam/batch_normalization_2/beta/m
ћ
5Adam/batch_normalization_2/beta/m/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_2/beta/m*
_output_shapes	
:ђ*
dtype0
Є
Adam/dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ђT*&
shared_nameAdam/dense_1/kernel/m
ђ
)Adam/dense_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/m*
_output_shapes
:	ђT*
dtype0
~
Adam/dense_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:T*$
shared_nameAdam/dense_1/bias/m
w
'Adam/dense_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/m*
_output_shapes
:T*
dtype0
ю
"Adam/batch_normalization_3/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:T*3
shared_name$"Adam/batch_normalization_3/gamma/m
Ћ
6Adam/batch_normalization_3/gamma/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_3/gamma/m*
_output_shapes
:T*
dtype0
џ
!Adam/batch_normalization_3/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:T*2
shared_name#!Adam/batch_normalization_3/beta/m
Њ
5Adam/batch_normalization_3/beta/m/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_3/beta/m*
_output_shapes
:T*
dtype0
џ
Adam/stream_0_conv_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*.
shared_nameAdam/stream_0_conv_1/kernel/v
Њ
1Adam/stream_0_conv_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/stream_0_conv_1/kernel/v*"
_output_shapes
:@*
dtype0
ј
Adam/stream_0_conv_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_nameAdam/stream_0_conv_1/bias/v
Є
/Adam/stream_0_conv_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/stream_0_conv_1/bias/v*
_output_shapes
:@*
dtype0
ў
 Adam/batch_normalization/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*1
shared_name" Adam/batch_normalization/gamma/v
Љ
4Adam/batch_normalization/gamma/v/Read/ReadVariableOpReadVariableOp Adam/batch_normalization/gamma/v*
_output_shapes
:@*
dtype0
ќ
Adam/batch_normalization/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*0
shared_name!Adam/batch_normalization/beta/v
Ј
3Adam/batch_normalization/beta/v/Read/ReadVariableOpReadVariableOpAdam/batch_normalization/beta/v*
_output_shapes
:@*
dtype0
Џ
Adam/stream_0_conv_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@ђ*.
shared_nameAdam/stream_0_conv_2/kernel/v
ћ
1Adam/stream_0_conv_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/stream_0_conv_2/kernel/v*#
_output_shapes
:@ђ*
dtype0
Ј
Adam/stream_0_conv_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*,
shared_nameAdam/stream_0_conv_2/bias/v
ѕ
/Adam/stream_0_conv_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/stream_0_conv_2/bias/v*
_output_shapes	
:ђ*
dtype0
Ю
"Adam/batch_normalization_1/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*3
shared_name$"Adam/batch_normalization_1/gamma/v
ќ
6Adam/batch_normalization_1/gamma/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_1/gamma/v*
_output_shapes	
:ђ*
dtype0
Џ
!Adam/batch_normalization_1/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*2
shared_name#!Adam/batch_normalization_1/beta/v
ћ
5Adam/batch_normalization_1/beta/v/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_1/beta/v*
_output_shapes	
:ђ*
dtype0
ю
Adam/stream_0_conv_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђђ*.
shared_nameAdam/stream_0_conv_3/kernel/v
Ћ
1Adam/stream_0_conv_3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/stream_0_conv_3/kernel/v*$
_output_shapes
:ђђ*
dtype0
Ј
Adam/stream_0_conv_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*,
shared_nameAdam/stream_0_conv_3/bias/v
ѕ
/Adam/stream_0_conv_3/bias/v/Read/ReadVariableOpReadVariableOpAdam/stream_0_conv_3/bias/v*
_output_shapes	
:ђ*
dtype0
Ю
"Adam/batch_normalization_2/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*3
shared_name$"Adam/batch_normalization_2/gamma/v
ќ
6Adam/batch_normalization_2/gamma/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_2/gamma/v*
_output_shapes	
:ђ*
dtype0
Џ
!Adam/batch_normalization_2/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*2
shared_name#!Adam/batch_normalization_2/beta/v
ћ
5Adam/batch_normalization_2/beta/v/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_2/beta/v*
_output_shapes	
:ђ*
dtype0
Є
Adam/dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ђT*&
shared_nameAdam/dense_1/kernel/v
ђ
)Adam/dense_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/v*
_output_shapes
:	ђT*
dtype0
~
Adam/dense_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:T*$
shared_nameAdam/dense_1/bias/v
w
'Adam/dense_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/v*
_output_shapes
:T*
dtype0
ю
"Adam/batch_normalization_3/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:T*3
shared_name$"Adam/batch_normalization_3/gamma/v
Ћ
6Adam/batch_normalization_3/gamma/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_3/gamma/v*
_output_shapes
:T*
dtype0
џ
!Adam/batch_normalization_3/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:T*2
shared_name#!Adam/batch_normalization_3/beta/v
Њ
5Adam/batch_normalization_3/beta/v/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_3/beta/v*
_output_shapes
:T*
dtype0

NoOpNoOp
яz
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*Ўz
valueЈzBїz BЁz
┐
layer-0
layer-1
layer_with_weights-0
layer-2
layer-3
	optimizer
regularization_losses
trainable_variables
	variables
		keras_api


signatures
 
 
б
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer-4
layer-5
layer_with_weights-2
layer-6
layer_with_weights-3
layer-7
layer-8
layer-9
layer_with_weights-4
layer-10
layer_with_weights-5
layer-11
layer-12
layer-13
layer-14
layer-15
layer_with_weights-6
layer-16
layer_with_weights-7
layer-17
layer-18
regularization_losses
trainable_variables
 	variables
!	keras_api
R
"regularization_losses
#trainable_variables
$	variables
%	keras_api
ђ

&beta_1

'beta_2
	(decay
)learning_rate
*iter+m§,m■-m .mђ/mЂ0mѓ1mЃ2mё3mЁ4mє5mЄ6mѕ7mЅ8mі9mІ:mї+vЇ,vј-vЈ.vљ/vЉ0vњ1vЊ2vћ3vЋ4vќ5vЌ6vў7vЎ8vџ9vЏ:vю
 
v
+0
,1
-2
.3
/4
05
16
27
38
49
510
611
712
813
914
:15
Х
+0
,1
-2
.3
;4
<5
/6
07
18
29
=10
>11
312
413
514
615
?16
@17
718
819
920
:21
A22
B23
Г
Clayer_metrics
regularization_losses
Dlayer_regularization_losses
trainable_variables
	variables
Enon_trainable_variables

Flayers
Gmetrics
 
 
R
Hregularization_losses
Itrainable_variables
J	variables
K	keras_api
h

+kernel
,bias
Lregularization_losses
Mtrainable_variables
N	variables
O	keras_api
Ќ
Paxis
	-gamma
.beta
;moving_mean
<moving_variance
Qregularization_losses
Rtrainable_variables
S	variables
T	keras_api
R
Uregularization_losses
Vtrainable_variables
W	variables
X	keras_api
R
Yregularization_losses
Ztrainable_variables
[	variables
\	keras_api
h

/kernel
0bias
]regularization_losses
^trainable_variables
_	variables
`	keras_api
Ќ
aaxis
	1gamma
2beta
=moving_mean
>moving_variance
bregularization_losses
ctrainable_variables
d	variables
e	keras_api
R
fregularization_losses
gtrainable_variables
h	variables
i	keras_api
R
jregularization_losses
ktrainable_variables
l	variables
m	keras_api
h

3kernel
4bias
nregularization_losses
otrainable_variables
p	variables
q	keras_api
Ќ
raxis
	5gamma
6beta
?moving_mean
@moving_variance
sregularization_losses
ttrainable_variables
u	variables
v	keras_api
R
wregularization_losses
xtrainable_variables
y	variables
z	keras_api
R
{regularization_losses
|trainable_variables
}	variables
~	keras_api
U
regularization_losses
ђtrainable_variables
Ђ	variables
ѓ	keras_api
V
Ѓregularization_losses
ёtrainable_variables
Ё	variables
є	keras_api
l

7kernel
8bias
Єregularization_losses
ѕtrainable_variables
Ѕ	variables
і	keras_api
ю
	Іaxis
	9gamma
:beta
Amoving_mean
Bmoving_variance
їregularization_losses
Їtrainable_variables
ј	variables
Ј	keras_api
V
љregularization_losses
Љtrainable_variables
њ	variables
Њ	keras_api
 
v
+0
,1
-2
.3
/4
05
16
27
38
49
510
611
712
813
914
:15
Х
+0
,1
-2
.3
;4
<5
/6
07
18
29
=10
>11
312
413
514
615
?16
@17
718
819
920
:21
A22
B23
▓
ћlayer_metrics
regularization_losses
 Ћlayer_regularization_losses
trainable_variables
 	variables
ќnon_trainable_variables
Ќlayers
ўmetrics
 
 
 
▓
Ўmetrics
џlayer_metrics
"regularization_losses
#trainable_variables
$	variables
Џnon_trainable_variables
юlayers
 Юlayer_regularization_losses
GE
VARIABLE_VALUEbeta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
GE
VARIABLE_VALUEbeta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
EC
VARIABLE_VALUEdecay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUElearning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUEstream_0_conv_1/kernel0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEstream_0_conv_1/bias0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUEbatch_normalization/gamma0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUEbatch_normalization/beta0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUEstream_0_conv_2/kernel0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEstream_0_conv_2/bias0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEbatch_normalization_1/gamma0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUEbatch_normalization_1/beta0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUEstream_0_conv_3/kernel0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEstream_0_conv_3/bias0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUEbatch_normalization_2/gamma1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEbatch_normalization_2/beta1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEdense_1/kernel1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEdense_1/bias1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUEbatch_normalization_3/gamma1trainable_variables/14/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEbatch_normalization_3/beta1trainable_variables/15/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEbatch_normalization/moving_mean&variables/4/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUE#batch_normalization/moving_variance&variables/5/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUE!batch_normalization_1/moving_mean'variables/10/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUE%batch_normalization_1/moving_variance'variables/11/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUE!batch_normalization_2/moving_mean'variables/16/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUE%batch_normalization_2/moving_variance'variables/17/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUE!batch_normalization_3/moving_mean'variables/22/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUE%batch_normalization_3/moving_variance'variables/23/.ATTRIBUTES/VARIABLE_VALUE
 
 
8
;0
<1
=2
>3
?4
@5
A6
B7

0
1
2
3

ъ0
 
 
 
▓
Ъmetrics
аlayer_metrics
Hregularization_losses
Itrainable_variables
J	variables
Аnon_trainable_variables
бlayers
 Бlayer_regularization_losses
 

+0
,1

+0
,1
▓
цmetrics
Цlayer_metrics
Lregularization_losses
Mtrainable_variables
N	variables
дnon_trainable_variables
Дlayers
 еlayer_regularization_losses
 
 

-0
.1

-0
.1
;2
<3
▓
Еmetrics
фlayer_metrics
Qregularization_losses
Rtrainable_variables
S	variables
Фnon_trainable_variables
гlayers
 Гlayer_regularization_losses
 
 
 
▓
«metrics
»layer_metrics
Uregularization_losses
Vtrainable_variables
W	variables
░non_trainable_variables
▒layers
 ▓layer_regularization_losses
 
 
 
▓
│metrics
┤layer_metrics
Yregularization_losses
Ztrainable_variables
[	variables
хnon_trainable_variables
Хlayers
 иlayer_regularization_losses
 

/0
01

/0
01
▓
Иmetrics
╣layer_metrics
]regularization_losses
^trainable_variables
_	variables
║non_trainable_variables
╗layers
 ╝layer_regularization_losses
 
 

10
21

10
21
=2
>3
▓
йmetrics
Йlayer_metrics
bregularization_losses
ctrainable_variables
d	variables
┐non_trainable_variables
└layers
 ┴layer_regularization_losses
 
 
 
▓
┬metrics
├layer_metrics
fregularization_losses
gtrainable_variables
h	variables
─non_trainable_variables
┼layers
 кlayer_regularization_losses
 
 
 
▓
Кmetrics
╚layer_metrics
jregularization_losses
ktrainable_variables
l	variables
╔non_trainable_variables
╩layers
 ╦layer_regularization_losses
 

30
41

30
41
▓
╠metrics
═layer_metrics
nregularization_losses
otrainable_variables
p	variables
╬non_trainable_variables
¤layers
 лlayer_regularization_losses
 
 

50
61

50
61
?2
@3
▓
Лmetrics
мlayer_metrics
sregularization_losses
ttrainable_variables
u	variables
Мnon_trainable_variables
нlayers
 Нlayer_regularization_losses
 
 
 
▓
оmetrics
Оlayer_metrics
wregularization_losses
xtrainable_variables
y	variables
пnon_trainable_variables
┘layers
 ┌layer_regularization_losses
 
 
 
▓
█metrics
▄layer_metrics
{regularization_losses
|trainable_variables
}	variables
Пnon_trainable_variables
яlayers
 ▀layer_regularization_losses
 
 
 
┤
Яmetrics
рlayer_metrics
regularization_losses
ђtrainable_variables
Ђ	variables
Рnon_trainable_variables
сlayers
 Сlayer_regularization_losses
 
 
 
х
тmetrics
Тlayer_metrics
Ѓregularization_losses
ёtrainable_variables
Ё	variables
уnon_trainable_variables
Уlayers
 жlayer_regularization_losses
 

70
81

70
81
х
Жmetrics
вlayer_metrics
Єregularization_losses
ѕtrainable_variables
Ѕ	variables
Вnon_trainable_variables
ьlayers
 Ьlayer_regularization_losses
 
 

90
:1

90
:1
A2
B3
х
№metrics
­layer_metrics
їregularization_losses
Їtrainable_variables
ј	variables
ыnon_trainable_variables
Ыlayers
 зlayer_regularization_losses
 
 
 
х
Зmetrics
шlayer_metrics
љregularization_losses
Љtrainable_variables
њ	variables
Шnon_trainable_variables
эlayers
 Эlayer_regularization_losses
 
 
8
;0
<1
=2
>3
?4
@5
A6
B7
ј
0
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
 
 
 
 
 
 
8

щtotal

Щcount
ч	variables
Ч	keras_api
 
 
 
 
 
 
 
 
 
 
 
 

;0
<1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

=0
>1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

?0
@1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

A0
B1
 
 
 
 
 
 
 
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

щ0
Щ1

ч	variables
}
VARIABLE_VALUEAdam/stream_0_conv_1/kernel/mLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/stream_0_conv_1/bias/mLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
Ѓђ
VARIABLE_VALUE Adam/batch_normalization/gamma/mLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
Ђ
VARIABLE_VALUEAdam/batch_normalization/beta/mLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/stream_0_conv_2/kernel/mLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/stream_0_conv_2/bias/mLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
Ёѓ
VARIABLE_VALUE"Adam/batch_normalization_1/gamma/mLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ёЂ
VARIABLE_VALUE!Adam/batch_normalization_1/beta/mLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/stream_0_conv_3/kernel/mLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/stream_0_conv_3/bias/mLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
єЃ
VARIABLE_VALUE"Adam/batch_normalization_2/gamma/mMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
Ёѓ
VARIABLE_VALUE!Adam/batch_normalization_2/beta/mMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/dense_1/kernel/mMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/dense_1/bias/mMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
єЃ
VARIABLE_VALUE"Adam/batch_normalization_3/gamma/mMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
Ёѓ
VARIABLE_VALUE!Adam/batch_normalization_3/beta/mMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/stream_0_conv_1/kernel/vLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/stream_0_conv_1/bias/vLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
Ѓђ
VARIABLE_VALUE Adam/batch_normalization/gamma/vLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
Ђ
VARIABLE_VALUEAdam/batch_normalization/beta/vLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/stream_0_conv_2/kernel/vLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/stream_0_conv_2/bias/vLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
Ёѓ
VARIABLE_VALUE"Adam/batch_normalization_1/gamma/vLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ёЂ
VARIABLE_VALUE!Adam/batch_normalization_1/beta/vLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/stream_0_conv_3/kernel/vLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/stream_0_conv_3/bias/vLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
єЃ
VARIABLE_VALUE"Adam/batch_normalization_2/gamma/vMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
Ёѓ
VARIABLE_VALUE!Adam/batch_normalization_2/beta/vMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/dense_1/kernel/vMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/dense_1/bias/vMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
єЃ
VARIABLE_VALUE"Adam/batch_normalization_3/gamma/vMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
Ёѓ
VARIABLE_VALUE!Adam/batch_normalization_3/beta/vMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
є
serving_default_left_inputsPlaceholder*+
_output_shapes
:         }*
dtype0* 
shape:         }
Є
serving_default_right_inputsPlaceholder*+
_output_shapes
:         }*
dtype0* 
shape:         }
│
StatefulPartitionedCallStatefulPartitionedCallserving_default_left_inputsserving_default_right_inputsstream_0_conv_1/kernelstream_0_conv_1/bias#batch_normalization/moving_variancebatch_normalization/gammabatch_normalization/moving_meanbatch_normalization/betastream_0_conv_2/kernelstream_0_conv_2/bias%batch_normalization_1/moving_variancebatch_normalization_1/gamma!batch_normalization_1/moving_meanbatch_normalization_1/betastream_0_conv_3/kernelstream_0_conv_3/bias%batch_normalization_2/moving_variancebatch_normalization_2/gamma!batch_normalization_2/moving_meanbatch_normalization_2/betadense_1/kerneldense_1/bias%batch_normalization_3/moving_variancebatch_normalization_3/gamma!batch_normalization_3/moving_meanbatch_normalization_3/beta*%
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *:
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8ѓ *,
f'R%
#__inference_signature_wrapper_26266
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Б
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenamebeta_1/Read/ReadVariableOpbeta_2/Read/ReadVariableOpdecay/Read/ReadVariableOp!learning_rate/Read/ReadVariableOpAdam/iter/Read/ReadVariableOp*stream_0_conv_1/kernel/Read/ReadVariableOp(stream_0_conv_1/bias/Read/ReadVariableOp-batch_normalization/gamma/Read/ReadVariableOp,batch_normalization/beta/Read/ReadVariableOp*stream_0_conv_2/kernel/Read/ReadVariableOp(stream_0_conv_2/bias/Read/ReadVariableOp/batch_normalization_1/gamma/Read/ReadVariableOp.batch_normalization_1/beta/Read/ReadVariableOp*stream_0_conv_3/kernel/Read/ReadVariableOp(stream_0_conv_3/bias/Read/ReadVariableOp/batch_normalization_2/gamma/Read/ReadVariableOp.batch_normalization_2/beta/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOp/batch_normalization_3/gamma/Read/ReadVariableOp.batch_normalization_3/beta/Read/ReadVariableOp3batch_normalization/moving_mean/Read/ReadVariableOp7batch_normalization/moving_variance/Read/ReadVariableOp5batch_normalization_1/moving_mean/Read/ReadVariableOp9batch_normalization_1/moving_variance/Read/ReadVariableOp5batch_normalization_2/moving_mean/Read/ReadVariableOp9batch_normalization_2/moving_variance/Read/ReadVariableOp5batch_normalization_3/moving_mean/Read/ReadVariableOp9batch_normalization_3/moving_variance/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp1Adam/stream_0_conv_1/kernel/m/Read/ReadVariableOp/Adam/stream_0_conv_1/bias/m/Read/ReadVariableOp4Adam/batch_normalization/gamma/m/Read/ReadVariableOp3Adam/batch_normalization/beta/m/Read/ReadVariableOp1Adam/stream_0_conv_2/kernel/m/Read/ReadVariableOp/Adam/stream_0_conv_2/bias/m/Read/ReadVariableOp6Adam/batch_normalization_1/gamma/m/Read/ReadVariableOp5Adam/batch_normalization_1/beta/m/Read/ReadVariableOp1Adam/stream_0_conv_3/kernel/m/Read/ReadVariableOp/Adam/stream_0_conv_3/bias/m/Read/ReadVariableOp6Adam/batch_normalization_2/gamma/m/Read/ReadVariableOp5Adam/batch_normalization_2/beta/m/Read/ReadVariableOp)Adam/dense_1/kernel/m/Read/ReadVariableOp'Adam/dense_1/bias/m/Read/ReadVariableOp6Adam/batch_normalization_3/gamma/m/Read/ReadVariableOp5Adam/batch_normalization_3/beta/m/Read/ReadVariableOp1Adam/stream_0_conv_1/kernel/v/Read/ReadVariableOp/Adam/stream_0_conv_1/bias/v/Read/ReadVariableOp4Adam/batch_normalization/gamma/v/Read/ReadVariableOp3Adam/batch_normalization/beta/v/Read/ReadVariableOp1Adam/stream_0_conv_2/kernel/v/Read/ReadVariableOp/Adam/stream_0_conv_2/bias/v/Read/ReadVariableOp6Adam/batch_normalization_1/gamma/v/Read/ReadVariableOp5Adam/batch_normalization_1/beta/v/Read/ReadVariableOp1Adam/stream_0_conv_3/kernel/v/Read/ReadVariableOp/Adam/stream_0_conv_3/bias/v/Read/ReadVariableOp6Adam/batch_normalization_2/gamma/v/Read/ReadVariableOp5Adam/batch_normalization_2/beta/v/Read/ReadVariableOp)Adam/dense_1/kernel/v/Read/ReadVariableOp'Adam/dense_1/bias/v/Read/ReadVariableOp6Adam/batch_normalization_3/gamma/v/Read/ReadVariableOp5Adam/batch_normalization_3/beta/v/Read/ReadVariableOpConst*L
TinE
C2A	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *'
f"R 
__inference__traced_save_29211
▓
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamebeta_1beta_2decaylearning_rate	Adam/iterstream_0_conv_1/kernelstream_0_conv_1/biasbatch_normalization/gammabatch_normalization/betastream_0_conv_2/kernelstream_0_conv_2/biasbatch_normalization_1/gammabatch_normalization_1/betastream_0_conv_3/kernelstream_0_conv_3/biasbatch_normalization_2/gammabatch_normalization_2/betadense_1/kerneldense_1/biasbatch_normalization_3/gammabatch_normalization_3/betabatch_normalization/moving_mean#batch_normalization/moving_variance!batch_normalization_1/moving_mean%batch_normalization_1/moving_variance!batch_normalization_2/moving_mean%batch_normalization_2/moving_variance!batch_normalization_3/moving_mean%batch_normalization_3/moving_variancetotalcountAdam/stream_0_conv_1/kernel/mAdam/stream_0_conv_1/bias/m Adam/batch_normalization/gamma/mAdam/batch_normalization/beta/mAdam/stream_0_conv_2/kernel/mAdam/stream_0_conv_2/bias/m"Adam/batch_normalization_1/gamma/m!Adam/batch_normalization_1/beta/mAdam/stream_0_conv_3/kernel/mAdam/stream_0_conv_3/bias/m"Adam/batch_normalization_2/gamma/m!Adam/batch_normalization_2/beta/mAdam/dense_1/kernel/mAdam/dense_1/bias/m"Adam/batch_normalization_3/gamma/m!Adam/batch_normalization_3/beta/mAdam/stream_0_conv_1/kernel/vAdam/stream_0_conv_1/bias/v Adam/batch_normalization/gamma/vAdam/batch_normalization/beta/vAdam/stream_0_conv_2/kernel/vAdam/stream_0_conv_2/bias/v"Adam/batch_normalization_1/gamma/v!Adam/batch_normalization_1/beta/vAdam/stream_0_conv_3/kernel/vAdam/stream_0_conv_3/bias/v"Adam/batch_normalization_2/gamma/v!Adam/batch_normalization_2/beta/vAdam/dense_1/kernel/vAdam/dense_1/bias/v"Adam/batch_normalization_3/gamma/v!Adam/batch_normalization_3/beta/v*K
TinD
B2@*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ **
f%R#
!__inference__traced_restore_29410хн6
┤
Г
N__inference_batch_normalization_layer_call_and_return_conditional_losses_28193

inputs/
!batchnorm_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@1
#batchnorm_readvariableop_1_resource:@1
#batchnorm_readvariableop_2_resource:@
identityѕбbatchnorm/ReadVariableOpбbatchnorm/ReadVariableOp_1бbatchnorm/ReadVariableOp_2бbatchnorm/mul/ReadVariableOpњ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2
batchnorm/add/yѕ
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@2
batchnorm/Rsqrtъ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOpЁ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2
batchnorm/mulЃ
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :                  @2
batchnorm/mul_1ў
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp_1Ё
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:@2
batchnorm/mul_2ў
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp_2Ѓ
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2
batchnorm/subњ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :                  @2
batchnorm/add_1{
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :                  @2

Identity┬
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:                  @: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :                  @
 
_user_specified_nameinputs
к
i
J__inference_dense_1_dropout_layer_call_and_return_conditional_losses_24258

inputs
identityѕc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:         ђ2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape┬
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:         ђ*
dtype0*
seedи2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>2
dropout/GreaterEqual/y┐
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         ђ2
dropout/GreaterEqualђ
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         ђ2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:         ђ2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         ђ:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
╣
o
S__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_23694

inputs
identityr
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Mean/reduction_indicesx
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:                  2
Meanj
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:                  2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'                           :e a
=
_output_shapes+
):'                           
 
_user_specified_nameinputs
╚
│
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_23546

inputs0
!batchnorm_readvariableop_resource:	ђ4
%batchnorm_mul_readvariableop_resource:	ђ2
#batchnorm_readvariableop_1_resource:	ђ2
#batchnorm_readvariableop_2_resource:	ђ
identityѕбbatchnorm/ReadVariableOpбbatchnorm/ReadVariableOp_1бbatchnorm/ReadVariableOp_2бbatchnorm/mul/ReadVariableOpЊ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2
batchnorm/add/yЅ
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:ђ2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:ђ2
batchnorm/RsqrtЪ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
batchnorm/mul/ReadVariableOpє
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:ђ2
batchnorm/mulё
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*5
_output_shapes#
!:                  ђ2
batchnorm/mul_1Ў
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype02
batchnorm/ReadVariableOp_1є
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:ђ2
batchnorm/mul_2Ў
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:ђ*
dtype02
batchnorm/ReadVariableOp_2ё
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:ђ2
batchnorm/subЊ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*5
_output_shapes#
!:                  ђ2
batchnorm/add_1|
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*5
_output_shapes#
!:                  ђ2

Identity┬
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):                  ђ: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:] Y
5
_output_shapes#
!:                  ђ
 
_user_specified_nameinputs
М+
ь
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_23606

inputs6
'assignmovingavg_readvariableop_resource:	ђ8
)assignmovingavg_1_readvariableop_resource:	ђ4
%batchnorm_mul_readvariableop_resource:	ђ0
!batchnorm_readvariableop_resource:	ђ
identityѕбAssignMovingAvgбAssignMovingAvg/ReadVariableOpбAssignMovingAvg_1б AssignMovingAvg_1/ReadVariableOpбbatchnorm/ReadVariableOpбbatchnorm/mul/ReadVariableOpЉ
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indicesћ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*#
_output_shapes
:ђ*
	keep_dims(2
moments/meanЂ
moments/StopGradientStopGradientmoments/mean:output:0*
T0*#
_output_shapes
:ђ2
moments/StopGradient▓
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*5
_output_shapes#
!:                  ђ2
moments/SquaredDifferenceЎ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indicesи
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*#
_output_shapes
:ђ*
	keep_dims(2
moments/varianceѓ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:ђ*
squeeze_dims
 2
moments/Squeezeі
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:ђ*
squeeze_dims
 2
moments/Squeeze_1s
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2
AssignMovingAvg/decayЦ
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:ђ*
dtype02 
AssignMovingAvg/ReadVariableOpЎ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:ђ2
AssignMovingAvg/subљ
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:ђ2
AssignMovingAvg/mul┐
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvgw
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2
AssignMovingAvg_1/decayФ
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:ђ*
dtype02"
 AssignMovingAvg_1/ReadVariableOpА
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:ђ2
AssignMovingAvg_1/subў
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:ђ2
AssignMovingAvg_1/mul╔
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1g
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2
batchnorm/add/yЃ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:ђ2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:ђ2
batchnorm/RsqrtЪ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
batchnorm/mul/ReadVariableOpє
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:ђ2
batchnorm/mulё
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*5
_output_shapes#
!:                  ђ2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:ђ2
batchnorm/mul_2Њ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
batchnorm/ReadVariableOpѓ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:ђ2
batchnorm/subЊ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*5
_output_shapes#
!:                  ђ2
batchnorm/add_1|
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*5
_output_shapes#
!:                  ђ2

IdentityЫ
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):                  ђ: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:] Y
5
_output_shapes#
!:                  ђ
 
_user_specified_nameinputs
Л
F
*__inference_activation_layer_call_fn_28286

inputs
identity╩
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         }@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_activation_layer_call_and_return_conditional_losses_239452
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:         }@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         }@:S O
+
_output_shapes
:         }@
 
_user_specified_nameinputs
╚
│
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_28426

inputs0
!batchnorm_readvariableop_resource:	ђ4
%batchnorm_mul_readvariableop_resource:	ђ2
#batchnorm_readvariableop_1_resource:	ђ2
#batchnorm_readvariableop_2_resource:	ђ
identityѕбbatchnorm/ReadVariableOpбbatchnorm/ReadVariableOp_1бbatchnorm/ReadVariableOp_2бbatchnorm/mul/ReadVariableOpЊ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2
batchnorm/add/yЅ
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:ђ2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:ђ2
batchnorm/RsqrtЪ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
batchnorm/mul/ReadVariableOpє
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:ђ2
batchnorm/mulё
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*5
_output_shapes#
!:                  ђ2
batchnorm/mul_1Ў
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype02
batchnorm/ReadVariableOp_1є
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:ђ2
batchnorm/mul_2Ў
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:ђ*
dtype02
batchnorm/ReadVariableOp_2ё
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:ђ2
batchnorm/subЊ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*5
_output_shapes#
!:                  ђ2
batchnorm/add_1|
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*5
_output_shapes#
!:                  ђ2

Identity┬
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):                  ђ: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:] Y
5
_output_shapes#
!:                  ђ
 
_user_specified_nameinputs
┬
h
/__inference_stream_0_drop_2_layer_call_fn_28534

inputs
identityѕбStatefulPartitionedCallУ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         }ђ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *S
fNRL
J__inference_stream_0_drop_2_layer_call_and_return_conditional_losses_243852
StatefulPartitionedCallђ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:         }ђ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         }ђ22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:         }ђ
 
_user_specified_nameinputs
Џ
T
8__inference_global_average_pooling1d_layer_call_fn_28789

inputs
identityП
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:                  * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *\
fWRU
S__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_236942
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:                  2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'                           :e a
=
_output_shapes+
):'                           
 
_user_specified_nameinputs
ж
T
8__inference_global_average_pooling1d_layer_call_fn_28794

inputs
identityН
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *\
fWRU
S__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_240992
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         }ђ:T P
,
_output_shapes
:         }ђ
 
_user_specified_nameinputs
­
m
N__inference_stream_0_input_drop_layer_call_and_return_conditional_losses_28085

inputs
identityѕc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?2
dropout/Constw
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:         }2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/ShapeМ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:         }*
dtype0*
seedи*
seed2и2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>2
dropout/GreaterEqual/y┬
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:         }2
dropout/GreaterEqualЃ
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:         }2
dropout/Cast~
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:         }2
dropout/Mul_1i
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:         }2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         }:S O
+
_output_shapes
:         }
 
_user_specified_nameinputs
ЇБ
ћ
D__inference_basemodel_layer_call_and_return_conditional_losses_28022
inputs_0Q
;stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource:@=
/stream_0_conv_1_biasadd_readvariableop_resource:@I
;batch_normalization_assignmovingavg_readvariableop_resource:@K
=batch_normalization_assignmovingavg_1_readvariableop_resource:@G
9batch_normalization_batchnorm_mul_readvariableop_resource:@C
5batch_normalization_batchnorm_readvariableop_resource:@R
;stream_0_conv_2_conv1d_expanddims_1_readvariableop_resource:@ђ>
/stream_0_conv_2_biasadd_readvariableop_resource:	ђL
=batch_normalization_1_assignmovingavg_readvariableop_resource:	ђN
?batch_normalization_1_assignmovingavg_1_readvariableop_resource:	ђJ
;batch_normalization_1_batchnorm_mul_readvariableop_resource:	ђF
7batch_normalization_1_batchnorm_readvariableop_resource:	ђS
;stream_0_conv_3_conv1d_expanddims_1_readvariableop_resource:ђђ>
/stream_0_conv_3_biasadd_readvariableop_resource:	ђL
=batch_normalization_2_assignmovingavg_readvariableop_resource:	ђN
?batch_normalization_2_assignmovingavg_1_readvariableop_resource:	ђJ
;batch_normalization_2_batchnorm_mul_readvariableop_resource:	ђF
7batch_normalization_2_batchnorm_readvariableop_resource:	ђ9
&dense_1_matmul_readvariableop_resource:	ђT5
'dense_1_biasadd_readvariableop_resource:TK
=batch_normalization_3_assignmovingavg_readvariableop_resource:TM
?batch_normalization_3_assignmovingavg_1_readvariableop_resource:TI
;batch_normalization_3_batchnorm_mul_readvariableop_resource:TE
7batch_normalization_3_batchnorm_readvariableop_resource:T
identityѕб#batch_normalization/AssignMovingAvgб2batch_normalization/AssignMovingAvg/ReadVariableOpб%batch_normalization/AssignMovingAvg_1б4batch_normalization/AssignMovingAvg_1/ReadVariableOpб,batch_normalization/batchnorm/ReadVariableOpб0batch_normalization/batchnorm/mul/ReadVariableOpб%batch_normalization_1/AssignMovingAvgб4batch_normalization_1/AssignMovingAvg/ReadVariableOpб'batch_normalization_1/AssignMovingAvg_1б6batch_normalization_1/AssignMovingAvg_1/ReadVariableOpб.batch_normalization_1/batchnorm/ReadVariableOpб2batch_normalization_1/batchnorm/mul/ReadVariableOpб%batch_normalization_2/AssignMovingAvgб4batch_normalization_2/AssignMovingAvg/ReadVariableOpб'batch_normalization_2/AssignMovingAvg_1б6batch_normalization_2/AssignMovingAvg_1/ReadVariableOpб.batch_normalization_2/batchnorm/ReadVariableOpб2batch_normalization_2/batchnorm/mul/ReadVariableOpб%batch_normalization_3/AssignMovingAvgб4batch_normalization_3/AssignMovingAvg/ReadVariableOpб'batch_normalization_3/AssignMovingAvg_1б6batch_normalization_3/AssignMovingAvg_1/ReadVariableOpб.batch_normalization_3/batchnorm/ReadVariableOpб2batch_normalization_3/batchnorm/mul/ReadVariableOpбdense_1/BiasAdd/ReadVariableOpбdense_1/MatMul/ReadVariableOpб-dense_1/kernel/Regularizer/Abs/ReadVariableOpб&stream_0_conv_1/BiasAdd/ReadVariableOpб2stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOpб5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpб&stream_0_conv_2/BiasAdd/ReadVariableOpб2stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOpб8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOpб&stream_0_conv_3/BiasAdd/ReadVariableOpб2stream_0_conv_3/conv1d/ExpandDims_1/ReadVariableOpб5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOpІ
!stream_0_input_drop/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?2#
!stream_0_input_drop/dropout/Constх
stream_0_input_drop/dropout/MulMulinputs_0*stream_0_input_drop/dropout/Const:output:0*
T0*+
_output_shapes
:         }2!
stream_0_input_drop/dropout/Mul~
!stream_0_input_drop/dropout/ShapeShapeinputs_0*
T0*
_output_shapes
:2#
!stream_0_input_drop/dropout/ShapeЈ
8stream_0_input_drop/dropout/random_uniform/RandomUniformRandomUniform*stream_0_input_drop/dropout/Shape:output:0*
T0*+
_output_shapes
:         }*
dtype0*
seedи*
seed2и2:
8stream_0_input_drop/dropout/random_uniform/RandomUniformЮ
*stream_0_input_drop/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>2,
*stream_0_input_drop/dropout/GreaterEqual/yњ
(stream_0_input_drop/dropout/GreaterEqualGreaterEqualAstream_0_input_drop/dropout/random_uniform/RandomUniform:output:03stream_0_input_drop/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:         }2*
(stream_0_input_drop/dropout/GreaterEqual┐
 stream_0_input_drop/dropout/CastCast,stream_0_input_drop/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:         }2"
 stream_0_input_drop/dropout/Cast╬
!stream_0_input_drop/dropout/Mul_1Mul#stream_0_input_drop/dropout/Mul:z:0$stream_0_input_drop/dropout/Cast:y:0*
T0*+
_output_shapes
:         }2#
!stream_0_input_drop/dropout/Mul_1Ў
%stream_0_conv_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§        2'
%stream_0_conv_1/conv1d/ExpandDims/dimт
!stream_0_conv_1/conv1d/ExpandDims
ExpandDims%stream_0_input_drop/dropout/Mul_1:z:0.stream_0_conv_1/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:         }2#
!stream_0_conv_1/conv1d/ExpandDimsУ
2stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp;stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype024
2stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOpћ
'stream_0_conv_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2)
'stream_0_conv_1/conv1d/ExpandDims_1/dimэ
#stream_0_conv_1/conv1d/ExpandDims_1
ExpandDims:stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp:value:00stream_0_conv_1/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@2%
#stream_0_conv_1/conv1d/ExpandDims_1Ш
stream_0_conv_1/conv1dConv2D*stream_0_conv_1/conv1d/ExpandDims:output:0,stream_0_conv_1/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:         }@*
paddingSAME*
strides
2
stream_0_conv_1/conv1d┬
stream_0_conv_1/conv1d/SqueezeSqueezestream_0_conv_1/conv1d:output:0*
T0*+
_output_shapes
:         }@*
squeeze_dims

§        2 
stream_0_conv_1/conv1d/Squeeze╝
&stream_0_conv_1/BiasAdd/ReadVariableOpReadVariableOp/stream_0_conv_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02(
&stream_0_conv_1/BiasAdd/ReadVariableOp╠
stream_0_conv_1/BiasAddBiasAdd'stream_0_conv_1/conv1d/Squeeze:output:0.stream_0_conv_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         }@2
stream_0_conv_1/BiasAdd╣
2batch_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       24
2batch_normalization/moments/mean/reduction_indicesж
 batch_normalization/moments/meanMean stream_0_conv_1/BiasAdd:output:0;batch_normalization/moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(2"
 batch_normalization/moments/mean╝
(batch_normalization/moments/StopGradientStopGradient)batch_normalization/moments/mean:output:0*
T0*"
_output_shapes
:@2*
(batch_normalization/moments/StopGradient■
-batch_normalization/moments/SquaredDifferenceSquaredDifference stream_0_conv_1/BiasAdd:output:01batch_normalization/moments/StopGradient:output:0*
T0*+
_output_shapes
:         }@2/
-batch_normalization/moments/SquaredDifference┴
6batch_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       28
6batch_normalization/moments/variance/reduction_indicesє
$batch_normalization/moments/varianceMean1batch_normalization/moments/SquaredDifference:z:0?batch_normalization/moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(2&
$batch_normalization/moments/varianceй
#batch_normalization/moments/SqueezeSqueeze)batch_normalization/moments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 2%
#batch_normalization/moments/Squeeze┼
%batch_normalization/moments/Squeeze_1Squeeze-batch_normalization/moments/variance:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 2'
%batch_normalization/moments/Squeeze_1Џ
)batch_normalization/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2+
)batch_normalization/AssignMovingAvg/decayЯ
2batch_normalization/AssignMovingAvg/ReadVariableOpReadVariableOp;batch_normalization_assignmovingavg_readvariableop_resource*
_output_shapes
:@*
dtype024
2batch_normalization/AssignMovingAvg/ReadVariableOpУ
'batch_normalization/AssignMovingAvg/subSub:batch_normalization/AssignMovingAvg/ReadVariableOp:value:0,batch_normalization/moments/Squeeze:output:0*
T0*
_output_shapes
:@2)
'batch_normalization/AssignMovingAvg/sub▀
'batch_normalization/AssignMovingAvg/mulMul+batch_normalization/AssignMovingAvg/sub:z:02batch_normalization/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:@2)
'batch_normalization/AssignMovingAvg/mulБ
#batch_normalization/AssignMovingAvgAssignSubVariableOp;batch_normalization_assignmovingavg_readvariableop_resource+batch_normalization/AssignMovingAvg/mul:z:03^batch_normalization/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02%
#batch_normalization/AssignMovingAvgЪ
+batch_normalization/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2-
+batch_normalization/AssignMovingAvg_1/decayТ
4batch_normalization/AssignMovingAvg_1/ReadVariableOpReadVariableOp=batch_normalization_assignmovingavg_1_readvariableop_resource*
_output_shapes
:@*
dtype026
4batch_normalization/AssignMovingAvg_1/ReadVariableOp­
)batch_normalization/AssignMovingAvg_1/subSub<batch_normalization/AssignMovingAvg_1/ReadVariableOp:value:0.batch_normalization/moments/Squeeze_1:output:0*
T0*
_output_shapes
:@2+
)batch_normalization/AssignMovingAvg_1/subу
)batch_normalization/AssignMovingAvg_1/mulMul-batch_normalization/AssignMovingAvg_1/sub:z:04batch_normalization/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:@2+
)batch_normalization/AssignMovingAvg_1/mulГ
%batch_normalization/AssignMovingAvg_1AssignSubVariableOp=batch_normalization_assignmovingavg_1_readvariableop_resource-batch_normalization/AssignMovingAvg_1/mul:z:05^batch_normalization/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02'
%batch_normalization/AssignMovingAvg_1Ј
#batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2%
#batch_normalization/batchnorm/add/yм
!batch_normalization/batchnorm/addAddV2.batch_normalization/moments/Squeeze_1:output:0,batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes
:@2#
!batch_normalization/batchnorm/addЪ
#batch_normalization/batchnorm/RsqrtRsqrt%batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes
:@2%
#batch_normalization/batchnorm/Rsqrt┌
0batch_normalization/batchnorm/mul/ReadVariableOpReadVariableOp9batch_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype022
0batch_normalization/batchnorm/mul/ReadVariableOpН
!batch_normalization/batchnorm/mulMul'batch_normalization/batchnorm/Rsqrt:y:08batch_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2#
!batch_normalization/batchnorm/mulл
#batch_normalization/batchnorm/mul_1Mul stream_0_conv_1/BiasAdd:output:0%batch_normalization/batchnorm/mul:z:0*
T0*+
_output_shapes
:         }@2%
#batch_normalization/batchnorm/mul_1╦
#batch_normalization/batchnorm/mul_2Mul,batch_normalization/moments/Squeeze:output:0%batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes
:@2%
#batch_normalization/batchnorm/mul_2╬
,batch_normalization/batchnorm/ReadVariableOpReadVariableOp5batch_normalization_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02.
,batch_normalization/batchnorm/ReadVariableOpЛ
!batch_normalization/batchnorm/subSub4batch_normalization/batchnorm/ReadVariableOp:value:0'batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2#
!batch_normalization/batchnorm/sub┘
#batch_normalization/batchnorm/add_1AddV2'batch_normalization/batchnorm/mul_1:z:0%batch_normalization/batchnorm/sub:z:0*
T0*+
_output_shapes
:         }@2%
#batch_normalization/batchnorm/add_1Ѕ
activation/ReluRelu'batch_normalization/batchnorm/add_1:z:0*
T0*+
_output_shapes
:         }@2
activation/ReluЃ
stream_0_drop_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n█Х?2
stream_0_drop_1/dropout/ConstЙ
stream_0_drop_1/dropout/MulMulactivation/Relu:activations:0&stream_0_drop_1/dropout/Const:output:0*
T0*+
_output_shapes
:         }@2
stream_0_drop_1/dropout/MulІ
stream_0_drop_1/dropout/ShapeShapeactivation/Relu:activations:0*
T0*
_output_shapes
:2
stream_0_drop_1/dropout/ShapeЃ
4stream_0_drop_1/dropout/random_uniform/RandomUniformRandomUniform&stream_0_drop_1/dropout/Shape:output:0*
T0*+
_output_shapes
:         }@*
dtype0*
seedи*
seed2и26
4stream_0_drop_1/dropout/random_uniform/RandomUniformЋ
&stream_0_drop_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *џЎЎ>2(
&stream_0_drop_1/dropout/GreaterEqual/yѓ
$stream_0_drop_1/dropout/GreaterEqualGreaterEqual=stream_0_drop_1/dropout/random_uniform/RandomUniform:output:0/stream_0_drop_1/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:         }@2&
$stream_0_drop_1/dropout/GreaterEqual│
stream_0_drop_1/dropout/CastCast(stream_0_drop_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:         }@2
stream_0_drop_1/dropout/CastЙ
stream_0_drop_1/dropout/Mul_1Mulstream_0_drop_1/dropout/Mul:z:0 stream_0_drop_1/dropout/Cast:y:0*
T0*+
_output_shapes
:         }@2
stream_0_drop_1/dropout/Mul_1Ў
%stream_0_conv_2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§        2'
%stream_0_conv_2/conv1d/ExpandDims/dimр
!stream_0_conv_2/conv1d/ExpandDims
ExpandDims!stream_0_drop_1/dropout/Mul_1:z:0.stream_0_conv_2/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:         }@2#
!stream_0_conv_2/conv1d/ExpandDimsж
2stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp;stream_0_conv_2_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:@ђ*
dtype024
2stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOpћ
'stream_0_conv_2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2)
'stream_0_conv_2/conv1d/ExpandDims_1/dimЭ
#stream_0_conv_2/conv1d/ExpandDims_1
ExpandDims:stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOp:value:00stream_0_conv_2/conv1d/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:@ђ2%
#stream_0_conv_2/conv1d/ExpandDims_1э
stream_0_conv_2/conv1dConv2D*stream_0_conv_2/conv1d/ExpandDims:output:0,stream_0_conv_2/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:         }ђ*
paddingSAME*
strides
2
stream_0_conv_2/conv1d├
stream_0_conv_2/conv1d/SqueezeSqueezestream_0_conv_2/conv1d:output:0*
T0*,
_output_shapes
:         }ђ*
squeeze_dims

§        2 
stream_0_conv_2/conv1d/Squeezeй
&stream_0_conv_2/BiasAdd/ReadVariableOpReadVariableOp/stream_0_conv_2_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02(
&stream_0_conv_2/BiasAdd/ReadVariableOp═
stream_0_conv_2/BiasAddBiasAdd'stream_0_conv_2/conv1d/Squeeze:output:0.stream_0_conv_2/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:         }ђ2
stream_0_conv_2/BiasAddй
4batch_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       26
4batch_normalization_1/moments/mean/reduction_indices­
"batch_normalization_1/moments/meanMean stream_0_conv_2/BiasAdd:output:0=batch_normalization_1/moments/mean/reduction_indices:output:0*
T0*#
_output_shapes
:ђ*
	keep_dims(2$
"batch_normalization_1/moments/mean├
*batch_normalization_1/moments/StopGradientStopGradient+batch_normalization_1/moments/mean:output:0*
T0*#
_output_shapes
:ђ2,
*batch_normalization_1/moments/StopGradientЁ
/batch_normalization_1/moments/SquaredDifferenceSquaredDifference stream_0_conv_2/BiasAdd:output:03batch_normalization_1/moments/StopGradient:output:0*
T0*,
_output_shapes
:         }ђ21
/batch_normalization_1/moments/SquaredDifference┼
8batch_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2:
8batch_normalization_1/moments/variance/reduction_indicesЈ
&batch_normalization_1/moments/varianceMean3batch_normalization_1/moments/SquaredDifference:z:0Abatch_normalization_1/moments/variance/reduction_indices:output:0*
T0*#
_output_shapes
:ђ*
	keep_dims(2(
&batch_normalization_1/moments/variance─
%batch_normalization_1/moments/SqueezeSqueeze+batch_normalization_1/moments/mean:output:0*
T0*
_output_shapes	
:ђ*
squeeze_dims
 2'
%batch_normalization_1/moments/Squeeze╠
'batch_normalization_1/moments/Squeeze_1Squeeze/batch_normalization_1/moments/variance:output:0*
T0*
_output_shapes	
:ђ*
squeeze_dims
 2)
'batch_normalization_1/moments/Squeeze_1Ъ
+batch_normalization_1/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2-
+batch_normalization_1/AssignMovingAvg/decayу
4batch_normalization_1/AssignMovingAvg/ReadVariableOpReadVariableOp=batch_normalization_1_assignmovingavg_readvariableop_resource*
_output_shapes	
:ђ*
dtype026
4batch_normalization_1/AssignMovingAvg/ReadVariableOpы
)batch_normalization_1/AssignMovingAvg/subSub<batch_normalization_1/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_1/moments/Squeeze:output:0*
T0*
_output_shapes	
:ђ2+
)batch_normalization_1/AssignMovingAvg/subУ
)batch_normalization_1/AssignMovingAvg/mulMul-batch_normalization_1/AssignMovingAvg/sub:z:04batch_normalization_1/AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:ђ2+
)batch_normalization_1/AssignMovingAvg/mulГ
%batch_normalization_1/AssignMovingAvgAssignSubVariableOp=batch_normalization_1_assignmovingavg_readvariableop_resource-batch_normalization_1/AssignMovingAvg/mul:z:05^batch_normalization_1/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02'
%batch_normalization_1/AssignMovingAvgБ
-batch_normalization_1/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2/
-batch_normalization_1/AssignMovingAvg_1/decayь
6batch_normalization_1/AssignMovingAvg_1/ReadVariableOpReadVariableOp?batch_normalization_1_assignmovingavg_1_readvariableop_resource*
_output_shapes	
:ђ*
dtype028
6batch_normalization_1/AssignMovingAvg_1/ReadVariableOpщ
+batch_normalization_1/AssignMovingAvg_1/subSub>batch_normalization_1/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_1/moments/Squeeze_1:output:0*
T0*
_output_shapes	
:ђ2-
+batch_normalization_1/AssignMovingAvg_1/sub­
+batch_normalization_1/AssignMovingAvg_1/mulMul/batch_normalization_1/AssignMovingAvg_1/sub:z:06batch_normalization_1/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:ђ2-
+batch_normalization_1/AssignMovingAvg_1/mulи
'batch_normalization_1/AssignMovingAvg_1AssignSubVariableOp?batch_normalization_1_assignmovingavg_1_readvariableop_resource/batch_normalization_1/AssignMovingAvg_1/mul:z:07^batch_normalization_1/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02)
'batch_normalization_1/AssignMovingAvg_1Њ
%batch_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2'
%batch_normalization_1/batchnorm/add/y█
#batch_normalization_1/batchnorm/addAddV20batch_normalization_1/moments/Squeeze_1:output:0.batch_normalization_1/batchnorm/add/y:output:0*
T0*
_output_shapes	
:ђ2%
#batch_normalization_1/batchnorm/addд
%batch_normalization_1/batchnorm/RsqrtRsqrt'batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes	
:ђ2'
%batch_normalization_1/batchnorm/Rsqrtр
2batch_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes	
:ђ*
dtype024
2batch_normalization_1/batchnorm/mul/ReadVariableOpя
#batch_normalization_1/batchnorm/mulMul)batch_normalization_1/batchnorm/Rsqrt:y:0:batch_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:ђ2%
#batch_normalization_1/batchnorm/mulО
%batch_normalization_1/batchnorm/mul_1Mul stream_0_conv_2/BiasAdd:output:0'batch_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:         }ђ2'
%batch_normalization_1/batchnorm/mul_1н
%batch_normalization_1/batchnorm/mul_2Mul.batch_normalization_1/moments/Squeeze:output:0'batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes	
:ђ2'
%batch_normalization_1/batchnorm/mul_2Н
.batch_normalization_1/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_1_batchnorm_readvariableop_resource*
_output_shapes	
:ђ*
dtype020
.batch_normalization_1/batchnorm/ReadVariableOp┌
#batch_normalization_1/batchnorm/subSub6batch_normalization_1/batchnorm/ReadVariableOp:value:0)batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:ђ2%
#batch_normalization_1/batchnorm/subР
%batch_normalization_1/batchnorm/add_1AddV2)batch_normalization_1/batchnorm/mul_1:z:0'batch_normalization_1/batchnorm/sub:z:0*
T0*,
_output_shapes
:         }ђ2'
%batch_normalization_1/batchnorm/add_1љ
activation_1/ReluRelu)batch_normalization_1/batchnorm/add_1:z:0*
T0*,
_output_shapes
:         }ђ2
activation_1/ReluЃ
stream_0_drop_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUН?2
stream_0_drop_2/dropout/Const┴
stream_0_drop_2/dropout/MulMulactivation_1/Relu:activations:0&stream_0_drop_2/dropout/Const:output:0*
T0*,
_output_shapes
:         }ђ2
stream_0_drop_2/dropout/MulЇ
stream_0_drop_2/dropout/ShapeShapeactivation_1/Relu:activations:0*
T0*
_output_shapes
:2
stream_0_drop_2/dropout/Shapeё
4stream_0_drop_2/dropout/random_uniform/RandomUniformRandomUniform&stream_0_drop_2/dropout/Shape:output:0*
T0*,
_output_shapes
:         }ђ*
dtype0*
seedи*
seed2и26
4stream_0_drop_2/dropout/random_uniform/RandomUniformЋ
&stream_0_drop_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠>2(
&stream_0_drop_2/dropout/GreaterEqual/yЃ
$stream_0_drop_2/dropout/GreaterEqualGreaterEqual=stream_0_drop_2/dropout/random_uniform/RandomUniform:output:0/stream_0_drop_2/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:         }ђ2&
$stream_0_drop_2/dropout/GreaterEqual┤
stream_0_drop_2/dropout/CastCast(stream_0_drop_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:         }ђ2
stream_0_drop_2/dropout/Cast┐
stream_0_drop_2/dropout/Mul_1Mulstream_0_drop_2/dropout/Mul:z:0 stream_0_drop_2/dropout/Cast:y:0*
T0*,
_output_shapes
:         }ђ2
stream_0_drop_2/dropout/Mul_1Ў
%stream_0_conv_3/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§        2'
%stream_0_conv_3/conv1d/ExpandDims/dimР
!stream_0_conv_3/conv1d/ExpandDims
ExpandDims!stream_0_drop_2/dropout/Mul_1:z:0.stream_0_conv_3/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         }ђ2#
!stream_0_conv_3/conv1d/ExpandDimsЖ
2stream_0_conv_3/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp;stream_0_conv_3_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:ђђ*
dtype024
2stream_0_conv_3/conv1d/ExpandDims_1/ReadVariableOpћ
'stream_0_conv_3/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2)
'stream_0_conv_3/conv1d/ExpandDims_1/dimщ
#stream_0_conv_3/conv1d/ExpandDims_1
ExpandDims:stream_0_conv_3/conv1d/ExpandDims_1/ReadVariableOp:value:00stream_0_conv_3/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:ђђ2%
#stream_0_conv_3/conv1d/ExpandDims_1э
stream_0_conv_3/conv1dConv2D*stream_0_conv_3/conv1d/ExpandDims:output:0,stream_0_conv_3/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:         }ђ*
paddingSAME*
strides
2
stream_0_conv_3/conv1d├
stream_0_conv_3/conv1d/SqueezeSqueezestream_0_conv_3/conv1d:output:0*
T0*,
_output_shapes
:         }ђ*
squeeze_dims

§        2 
stream_0_conv_3/conv1d/Squeezeй
&stream_0_conv_3/BiasAdd/ReadVariableOpReadVariableOp/stream_0_conv_3_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02(
&stream_0_conv_3/BiasAdd/ReadVariableOp═
stream_0_conv_3/BiasAddBiasAdd'stream_0_conv_3/conv1d/Squeeze:output:0.stream_0_conv_3/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:         }ђ2
stream_0_conv_3/BiasAddй
4batch_normalization_2/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       26
4batch_normalization_2/moments/mean/reduction_indices­
"batch_normalization_2/moments/meanMean stream_0_conv_3/BiasAdd:output:0=batch_normalization_2/moments/mean/reduction_indices:output:0*
T0*#
_output_shapes
:ђ*
	keep_dims(2$
"batch_normalization_2/moments/mean├
*batch_normalization_2/moments/StopGradientStopGradient+batch_normalization_2/moments/mean:output:0*
T0*#
_output_shapes
:ђ2,
*batch_normalization_2/moments/StopGradientЁ
/batch_normalization_2/moments/SquaredDifferenceSquaredDifference stream_0_conv_3/BiasAdd:output:03batch_normalization_2/moments/StopGradient:output:0*
T0*,
_output_shapes
:         }ђ21
/batch_normalization_2/moments/SquaredDifference┼
8batch_normalization_2/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2:
8batch_normalization_2/moments/variance/reduction_indicesЈ
&batch_normalization_2/moments/varianceMean3batch_normalization_2/moments/SquaredDifference:z:0Abatch_normalization_2/moments/variance/reduction_indices:output:0*
T0*#
_output_shapes
:ђ*
	keep_dims(2(
&batch_normalization_2/moments/variance─
%batch_normalization_2/moments/SqueezeSqueeze+batch_normalization_2/moments/mean:output:0*
T0*
_output_shapes	
:ђ*
squeeze_dims
 2'
%batch_normalization_2/moments/Squeeze╠
'batch_normalization_2/moments/Squeeze_1Squeeze/batch_normalization_2/moments/variance:output:0*
T0*
_output_shapes	
:ђ*
squeeze_dims
 2)
'batch_normalization_2/moments/Squeeze_1Ъ
+batch_normalization_2/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2-
+batch_normalization_2/AssignMovingAvg/decayу
4batch_normalization_2/AssignMovingAvg/ReadVariableOpReadVariableOp=batch_normalization_2_assignmovingavg_readvariableop_resource*
_output_shapes	
:ђ*
dtype026
4batch_normalization_2/AssignMovingAvg/ReadVariableOpы
)batch_normalization_2/AssignMovingAvg/subSub<batch_normalization_2/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_2/moments/Squeeze:output:0*
T0*
_output_shapes	
:ђ2+
)batch_normalization_2/AssignMovingAvg/subУ
)batch_normalization_2/AssignMovingAvg/mulMul-batch_normalization_2/AssignMovingAvg/sub:z:04batch_normalization_2/AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:ђ2+
)batch_normalization_2/AssignMovingAvg/mulГ
%batch_normalization_2/AssignMovingAvgAssignSubVariableOp=batch_normalization_2_assignmovingavg_readvariableop_resource-batch_normalization_2/AssignMovingAvg/mul:z:05^batch_normalization_2/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02'
%batch_normalization_2/AssignMovingAvgБ
-batch_normalization_2/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2/
-batch_normalization_2/AssignMovingAvg_1/decayь
6batch_normalization_2/AssignMovingAvg_1/ReadVariableOpReadVariableOp?batch_normalization_2_assignmovingavg_1_readvariableop_resource*
_output_shapes	
:ђ*
dtype028
6batch_normalization_2/AssignMovingAvg_1/ReadVariableOpщ
+batch_normalization_2/AssignMovingAvg_1/subSub>batch_normalization_2/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_2/moments/Squeeze_1:output:0*
T0*
_output_shapes	
:ђ2-
+batch_normalization_2/AssignMovingAvg_1/sub­
+batch_normalization_2/AssignMovingAvg_1/mulMul/batch_normalization_2/AssignMovingAvg_1/sub:z:06batch_normalization_2/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:ђ2-
+batch_normalization_2/AssignMovingAvg_1/mulи
'batch_normalization_2/AssignMovingAvg_1AssignSubVariableOp?batch_normalization_2_assignmovingavg_1_readvariableop_resource/batch_normalization_2/AssignMovingAvg_1/mul:z:07^batch_normalization_2/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02)
'batch_normalization_2/AssignMovingAvg_1Њ
%batch_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2'
%batch_normalization_2/batchnorm/add/y█
#batch_normalization_2/batchnorm/addAddV20batch_normalization_2/moments/Squeeze_1:output:0.batch_normalization_2/batchnorm/add/y:output:0*
T0*
_output_shapes	
:ђ2%
#batch_normalization_2/batchnorm/addд
%batch_normalization_2/batchnorm/RsqrtRsqrt'batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes	
:ђ2'
%batch_normalization_2/batchnorm/Rsqrtр
2batch_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes	
:ђ*
dtype024
2batch_normalization_2/batchnorm/mul/ReadVariableOpя
#batch_normalization_2/batchnorm/mulMul)batch_normalization_2/batchnorm/Rsqrt:y:0:batch_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:ђ2%
#batch_normalization_2/batchnorm/mulО
%batch_normalization_2/batchnorm/mul_1Mul stream_0_conv_3/BiasAdd:output:0'batch_normalization_2/batchnorm/mul:z:0*
T0*,
_output_shapes
:         }ђ2'
%batch_normalization_2/batchnorm/mul_1н
%batch_normalization_2/batchnorm/mul_2Mul.batch_normalization_2/moments/Squeeze:output:0'batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes	
:ђ2'
%batch_normalization_2/batchnorm/mul_2Н
.batch_normalization_2/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_2_batchnorm_readvariableop_resource*
_output_shapes	
:ђ*
dtype020
.batch_normalization_2/batchnorm/ReadVariableOp┌
#batch_normalization_2/batchnorm/subSub6batch_normalization_2/batchnorm/ReadVariableOp:value:0)batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:ђ2%
#batch_normalization_2/batchnorm/subР
%batch_normalization_2/batchnorm/add_1AddV2)batch_normalization_2/batchnorm/mul_1:z:0'batch_normalization_2/batchnorm/sub:z:0*
T0*,
_output_shapes
:         }ђ2'
%batch_normalization_2/batchnorm/add_1љ
activation_2/ReluRelu)batch_normalization_2/batchnorm/add_1:z:0*
T0*,
_output_shapes
:         }ђ2
activation_2/ReluЃ
stream_0_drop_3/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
stream_0_drop_3/dropout/Const┴
stream_0_drop_3/dropout/MulMulactivation_2/Relu:activations:0&stream_0_drop_3/dropout/Const:output:0*
T0*,
_output_shapes
:         }ђ2
stream_0_drop_3/dropout/MulЇ
stream_0_drop_3/dropout/ShapeShapeactivation_2/Relu:activations:0*
T0*
_output_shapes
:2
stream_0_drop_3/dropout/Shapeё
4stream_0_drop_3/dropout/random_uniform/RandomUniformRandomUniform&stream_0_drop_3/dropout/Shape:output:0*
T0*,
_output_shapes
:         }ђ*
dtype0*
seedи*
seed2и26
4stream_0_drop_3/dropout/random_uniform/RandomUniformЋ
&stream_0_drop_3/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2(
&stream_0_drop_3/dropout/GreaterEqual/yЃ
$stream_0_drop_3/dropout/GreaterEqualGreaterEqual=stream_0_drop_3/dropout/random_uniform/RandomUniform:output:0/stream_0_drop_3/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:         }ђ2&
$stream_0_drop_3/dropout/GreaterEqual┤
stream_0_drop_3/dropout/CastCast(stream_0_drop_3/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:         }ђ2
stream_0_drop_3/dropout/Cast┐
stream_0_drop_3/dropout/Mul_1Mulstream_0_drop_3/dropout/Mul:z:0 stream_0_drop_3/dropout/Cast:y:0*
T0*,
_output_shapes
:         }ђ2
stream_0_drop_3/dropout/Mul_1ц
/global_average_pooling1d/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :21
/global_average_pooling1d/Mean/reduction_indicesо
global_average_pooling1d/MeanMean!stream_0_drop_3/dropout/Mul_1:z:08global_average_pooling1d/Mean/reduction_indices:output:0*
T0*(
_output_shapes
:         ђ2
global_average_pooling1d/MeanЃ
dense_1_dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?2
dense_1_dropout/dropout/Const─
dense_1_dropout/dropout/MulMul&global_average_pooling1d/Mean:output:0&dense_1_dropout/dropout/Const:output:0*
T0*(
_output_shapes
:         ђ2
dense_1_dropout/dropout/Mulћ
dense_1_dropout/dropout/ShapeShape&global_average_pooling1d/Mean:output:0*
T0*
_output_shapes
:2
dense_1_dropout/dropout/ShapeЫ
4dense_1_dropout/dropout/random_uniform/RandomUniformRandomUniform&dense_1_dropout/dropout/Shape:output:0*
T0*(
_output_shapes
:         ђ*
dtype0*
seedи26
4dense_1_dropout/dropout/random_uniform/RandomUniformЋ
&dense_1_dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>2(
&dense_1_dropout/dropout/GreaterEqual/y 
$dense_1_dropout/dropout/GreaterEqualGreaterEqual=dense_1_dropout/dropout/random_uniform/RandomUniform:output:0/dense_1_dropout/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         ђ2&
$dense_1_dropout/dropout/GreaterEqual░
dense_1_dropout/dropout/CastCast(dense_1_dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         ђ2
dense_1_dropout/dropout/Cast╗
dense_1_dropout/dropout/Mul_1Muldense_1_dropout/dropout/Mul:z:0 dense_1_dropout/dropout/Cast:y:0*
T0*(
_output_shapes
:         ђ2
dense_1_dropout/dropout/Mul_1д
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	ђT*
dtype02
dense_1/MatMul/ReadVariableOpд
dense_1/MatMulMatMul!dense_1_dropout/dropout/Mul_1:z:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         T2
dense_1/MatMulц
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:T*
dtype02 
dense_1/BiasAdd/ReadVariableOpА
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         T2
dense_1/BiasAddХ
4batch_normalization_3/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 26
4batch_normalization_3/moments/mean/reduction_indicesс
"batch_normalization_3/moments/meanMeandense_1/BiasAdd:output:0=batch_normalization_3/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:T*
	keep_dims(2$
"batch_normalization_3/moments/meanЙ
*batch_normalization_3/moments/StopGradientStopGradient+batch_normalization_3/moments/mean:output:0*
T0*
_output_shapes

:T2,
*batch_normalization_3/moments/StopGradientЭ
/batch_normalization_3/moments/SquaredDifferenceSquaredDifferencedense_1/BiasAdd:output:03batch_normalization_3/moments/StopGradient:output:0*
T0*'
_output_shapes
:         T21
/batch_normalization_3/moments/SquaredDifferenceЙ
8batch_normalization_3/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2:
8batch_normalization_3/moments/variance/reduction_indicesі
&batch_normalization_3/moments/varianceMean3batch_normalization_3/moments/SquaredDifference:z:0Abatch_normalization_3/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:T*
	keep_dims(2(
&batch_normalization_3/moments/variance┬
%batch_normalization_3/moments/SqueezeSqueeze+batch_normalization_3/moments/mean:output:0*
T0*
_output_shapes
:T*
squeeze_dims
 2'
%batch_normalization_3/moments/Squeeze╩
'batch_normalization_3/moments/Squeeze_1Squeeze/batch_normalization_3/moments/variance:output:0*
T0*
_output_shapes
:T*
squeeze_dims
 2)
'batch_normalization_3/moments/Squeeze_1Ъ
+batch_normalization_3/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2-
+batch_normalization_3/AssignMovingAvg/decayТ
4batch_normalization_3/AssignMovingAvg/ReadVariableOpReadVariableOp=batch_normalization_3_assignmovingavg_readvariableop_resource*
_output_shapes
:T*
dtype026
4batch_normalization_3/AssignMovingAvg/ReadVariableOp­
)batch_normalization_3/AssignMovingAvg/subSub<batch_normalization_3/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_3/moments/Squeeze:output:0*
T0*
_output_shapes
:T2+
)batch_normalization_3/AssignMovingAvg/subу
)batch_normalization_3/AssignMovingAvg/mulMul-batch_normalization_3/AssignMovingAvg/sub:z:04batch_normalization_3/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:T2+
)batch_normalization_3/AssignMovingAvg/mulГ
%batch_normalization_3/AssignMovingAvgAssignSubVariableOp=batch_normalization_3_assignmovingavg_readvariableop_resource-batch_normalization_3/AssignMovingAvg/mul:z:05^batch_normalization_3/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02'
%batch_normalization_3/AssignMovingAvgБ
-batch_normalization_3/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2/
-batch_normalization_3/AssignMovingAvg_1/decayВ
6batch_normalization_3/AssignMovingAvg_1/ReadVariableOpReadVariableOp?batch_normalization_3_assignmovingavg_1_readvariableop_resource*
_output_shapes
:T*
dtype028
6batch_normalization_3/AssignMovingAvg_1/ReadVariableOpЭ
+batch_normalization_3/AssignMovingAvg_1/subSub>batch_normalization_3/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_3/moments/Squeeze_1:output:0*
T0*
_output_shapes
:T2-
+batch_normalization_3/AssignMovingAvg_1/sub№
+batch_normalization_3/AssignMovingAvg_1/mulMul/batch_normalization_3/AssignMovingAvg_1/sub:z:06batch_normalization_3/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:T2-
+batch_normalization_3/AssignMovingAvg_1/mulи
'batch_normalization_3/AssignMovingAvg_1AssignSubVariableOp?batch_normalization_3_assignmovingavg_1_readvariableop_resource/batch_normalization_3/AssignMovingAvg_1/mul:z:07^batch_normalization_3/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02)
'batch_normalization_3/AssignMovingAvg_1Њ
%batch_normalization_3/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2'
%batch_normalization_3/batchnorm/add/y┌
#batch_normalization_3/batchnorm/addAddV20batch_normalization_3/moments/Squeeze_1:output:0.batch_normalization_3/batchnorm/add/y:output:0*
T0*
_output_shapes
:T2%
#batch_normalization_3/batchnorm/addЦ
%batch_normalization_3/batchnorm/RsqrtRsqrt'batch_normalization_3/batchnorm/add:z:0*
T0*
_output_shapes
:T2'
%batch_normalization_3/batchnorm/RsqrtЯ
2batch_normalization_3/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_3_batchnorm_mul_readvariableop_resource*
_output_shapes
:T*
dtype024
2batch_normalization_3/batchnorm/mul/ReadVariableOpП
#batch_normalization_3/batchnorm/mulMul)batch_normalization_3/batchnorm/Rsqrt:y:0:batch_normalization_3/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:T2%
#batch_normalization_3/batchnorm/mul╩
%batch_normalization_3/batchnorm/mul_1Muldense_1/BiasAdd:output:0'batch_normalization_3/batchnorm/mul:z:0*
T0*'
_output_shapes
:         T2'
%batch_normalization_3/batchnorm/mul_1М
%batch_normalization_3/batchnorm/mul_2Mul.batch_normalization_3/moments/Squeeze:output:0'batch_normalization_3/batchnorm/mul:z:0*
T0*
_output_shapes
:T2'
%batch_normalization_3/batchnorm/mul_2н
.batch_normalization_3/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_3_batchnorm_readvariableop_resource*
_output_shapes
:T*
dtype020
.batch_normalization_3/batchnorm/ReadVariableOp┘
#batch_normalization_3/batchnorm/subSub6batch_normalization_3/batchnorm/ReadVariableOp:value:0)batch_normalization_3/batchnorm/mul_2:z:0*
T0*
_output_shapes
:T2%
#batch_normalization_3/batchnorm/subП
%batch_normalization_3/batchnorm/add_1AddV2)batch_normalization_3/batchnorm/mul_1:z:0'batch_normalization_3/batchnorm/sub:z:0*
T0*'
_output_shapes
:         T2'
%batch_normalization_3/batchnorm/add_1а
dense_activation_1/SigmoidSigmoid)batch_normalization_3/batchnorm/add_1:z:0*
T0*'
_output_shapes
:         T2
dense_activation_1/SigmoidЬ
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp;stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype027
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp├
&stream_0_conv_1/kernel/Regularizer/AbsAbs=stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_0_conv_1/kernel/Regularizer/AbsЕ
(stream_0_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_0_conv_1/kernel/Regularizer/ConstО
&stream_0_conv_1/kernel/Regularizer/SumSum*stream_0_conv_1/kernel/Regularizer/Abs:y:01stream_0_conv_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/SumЎ
(stream_0_conv_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2*
(stream_0_conv_1/kernel/Regularizer/mul/x▄
&stream_0_conv_1/kernel/Regularizer/mulMul1stream_0_conv_1/kernel/Regularizer/mul/x:output:0/stream_0_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/mulш
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;stream_0_conv_2_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:@ђ*
dtype02:
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOpл
)stream_0_conv_2/kernel/Regularizer/SquareSquare@stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*#
_output_shapes
:@ђ2+
)stream_0_conv_2/kernel/Regularizer/SquareЕ
(stream_0_conv_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_0_conv_2/kernel/Regularizer/Const┌
&stream_0_conv_2/kernel/Regularizer/SumSum-stream_0_conv_2/kernel/Regularizer/Square:y:01stream_0_conv_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_2/kernel/Regularizer/SumЎ
(stream_0_conv_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2*
(stream_0_conv_2/kernel/Regularizer/mul/x▄
&stream_0_conv_2/kernel/Regularizer/mulMul1stream_0_conv_2/kernel/Regularizer/mul/x:output:0/stream_0_conv_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_2/kernel/Regularizer/mul­
5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp;stream_0_conv_3_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:ђђ*
dtype027
5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp┼
&stream_0_conv_3/kernel/Regularizer/AbsAbs=stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*$
_output_shapes
:ђђ2(
&stream_0_conv_3/kernel/Regularizer/AbsЕ
(stream_0_conv_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_0_conv_3/kernel/Regularizer/ConstО
&stream_0_conv_3/kernel/Regularizer/SumSum*stream_0_conv_3/kernel/Regularizer/Abs:y:01stream_0_conv_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_3/kernel/Regularizer/SumЎ
(stream_0_conv_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2*
(stream_0_conv_3/kernel/Regularizer/mul/x▄
&stream_0_conv_3/kernel/Regularizer/mulMul1stream_0_conv_3/kernel/Regularizer/mul/x:output:0/stream_0_conv_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_3/kernel/Regularizer/mulк
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	ђT*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOpе
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	ђT2 
dense_1/kernel/Regularizer/AbsЋ
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_1/kernel/Regularizer/Constи
dense_1/kernel/Regularizer/SumSum"dense_1/kernel/Regularizer/Abs:y:0)dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/SumЅ
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2"
 dense_1/kernel/Regularizer/mul/x╝
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/muly
IdentityIdentitydense_activation_1/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:         T2

Identityў
NoOpNoOp$^batch_normalization/AssignMovingAvg3^batch_normalization/AssignMovingAvg/ReadVariableOp&^batch_normalization/AssignMovingAvg_15^batch_normalization/AssignMovingAvg_1/ReadVariableOp-^batch_normalization/batchnorm/ReadVariableOp1^batch_normalization/batchnorm/mul/ReadVariableOp&^batch_normalization_1/AssignMovingAvg5^batch_normalization_1/AssignMovingAvg/ReadVariableOp(^batch_normalization_1/AssignMovingAvg_17^batch_normalization_1/AssignMovingAvg_1/ReadVariableOp/^batch_normalization_1/batchnorm/ReadVariableOp3^batch_normalization_1/batchnorm/mul/ReadVariableOp&^batch_normalization_2/AssignMovingAvg5^batch_normalization_2/AssignMovingAvg/ReadVariableOp(^batch_normalization_2/AssignMovingAvg_17^batch_normalization_2/AssignMovingAvg_1/ReadVariableOp/^batch_normalization_2/batchnorm/ReadVariableOp3^batch_normalization_2/batchnorm/mul/ReadVariableOp&^batch_normalization_3/AssignMovingAvg5^batch_normalization_3/AssignMovingAvg/ReadVariableOp(^batch_normalization_3/AssignMovingAvg_17^batch_normalization_3/AssignMovingAvg_1/ReadVariableOp/^batch_normalization_3/batchnorm/ReadVariableOp3^batch_normalization_3/batchnorm/mul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp.^dense_1/kernel/Regularizer/Abs/ReadVariableOp'^stream_0_conv_1/BiasAdd/ReadVariableOp3^stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp'^stream_0_conv_2/BiasAdd/ReadVariableOp3^stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOp9^stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp'^stream_0_conv_3/BiasAdd/ReadVariableOp3^stream_0_conv_3/conv1d/ExpandDims_1/ReadVariableOp6^stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:         }: : : : : : : : : : : : : : : : : : : : : : : : 2J
#batch_normalization/AssignMovingAvg#batch_normalization/AssignMovingAvg2h
2batch_normalization/AssignMovingAvg/ReadVariableOp2batch_normalization/AssignMovingAvg/ReadVariableOp2N
%batch_normalization/AssignMovingAvg_1%batch_normalization/AssignMovingAvg_12l
4batch_normalization/AssignMovingAvg_1/ReadVariableOp4batch_normalization/AssignMovingAvg_1/ReadVariableOp2\
,batch_normalization/batchnorm/ReadVariableOp,batch_normalization/batchnorm/ReadVariableOp2d
0batch_normalization/batchnorm/mul/ReadVariableOp0batch_normalization/batchnorm/mul/ReadVariableOp2N
%batch_normalization_1/AssignMovingAvg%batch_normalization_1/AssignMovingAvg2l
4batch_normalization_1/AssignMovingAvg/ReadVariableOp4batch_normalization_1/AssignMovingAvg/ReadVariableOp2R
'batch_normalization_1/AssignMovingAvg_1'batch_normalization_1/AssignMovingAvg_12p
6batch_normalization_1/AssignMovingAvg_1/ReadVariableOp6batch_normalization_1/AssignMovingAvg_1/ReadVariableOp2`
.batch_normalization_1/batchnorm/ReadVariableOp.batch_normalization_1/batchnorm/ReadVariableOp2h
2batch_normalization_1/batchnorm/mul/ReadVariableOp2batch_normalization_1/batchnorm/mul/ReadVariableOp2N
%batch_normalization_2/AssignMovingAvg%batch_normalization_2/AssignMovingAvg2l
4batch_normalization_2/AssignMovingAvg/ReadVariableOp4batch_normalization_2/AssignMovingAvg/ReadVariableOp2R
'batch_normalization_2/AssignMovingAvg_1'batch_normalization_2/AssignMovingAvg_12p
6batch_normalization_2/AssignMovingAvg_1/ReadVariableOp6batch_normalization_2/AssignMovingAvg_1/ReadVariableOp2`
.batch_normalization_2/batchnorm/ReadVariableOp.batch_normalization_2/batchnorm/ReadVariableOp2h
2batch_normalization_2/batchnorm/mul/ReadVariableOp2batch_normalization_2/batchnorm/mul/ReadVariableOp2N
%batch_normalization_3/AssignMovingAvg%batch_normalization_3/AssignMovingAvg2l
4batch_normalization_3/AssignMovingAvg/ReadVariableOp4batch_normalization_3/AssignMovingAvg/ReadVariableOp2R
'batch_normalization_3/AssignMovingAvg_1'batch_normalization_3/AssignMovingAvg_12p
6batch_normalization_3/AssignMovingAvg_1/ReadVariableOp6batch_normalization_3/AssignMovingAvg_1/ReadVariableOp2`
.batch_normalization_3/batchnorm/ReadVariableOp.batch_normalization_3/batchnorm/ReadVariableOp2h
2batch_normalization_3/batchnorm/mul/ReadVariableOp2batch_normalization_3/batchnorm/mul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2^
-dense_1/kernel/Regularizer/Abs/ReadVariableOp-dense_1/kernel/Regularizer/Abs/ReadVariableOp2P
&stream_0_conv_1/BiasAdd/ReadVariableOp&stream_0_conv_1/BiasAdd/ReadVariableOp2h
2stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp2stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp2n
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp2P
&stream_0_conv_2/BiasAdd/ReadVariableOp&stream_0_conv_2/BiasAdd/ReadVariableOp2h
2stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOp2stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOp2t
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp2P
&stream_0_conv_3/BiasAdd/ReadVariableOp&stream_0_conv_3/BiasAdd/ReadVariableOp2h
2stream_0_conv_3/conv1d/ExpandDims_1/ReadVariableOp2stream_0_conv_3/conv1d/ExpandDims_1/ReadVariableOp2n
5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp:U Q
+
_output_shapes
:         }
"
_user_specified_name
inputs/0
┘
Л
J__inference_stream_0_conv_1_layer_call_and_return_conditional_losses_23905

inputsA
+conv1d_expanddims_1_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identityѕбBiasAdd/ReadVariableOpб"conv1d/ExpandDims_1/ReadVariableOpб5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§        2
conv1d/ExpandDims/dimќ
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:         }2
conv1d/ExpandDimsИ
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dimи
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@2
conv1d/ExpandDims_1Х
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:         }@*
paddingSAME*
strides
2
conv1dњ
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:         }@*
squeeze_dims

§        2
conv1d/Squeezeї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpї
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         }@2	
BiasAddя
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype027
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp├
&stream_0_conv_1/kernel/Regularizer/AbsAbs=stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_0_conv_1/kernel/Regularizer/AbsЕ
(stream_0_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_0_conv_1/kernel/Regularizer/ConstО
&stream_0_conv_1/kernel/Regularizer/SumSum*stream_0_conv_1/kernel/Regularizer/Abs:y:01stream_0_conv_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/SumЎ
(stream_0_conv_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2*
(stream_0_conv_1/kernel/Regularizer/mul/x▄
&stream_0_conv_1/kernel/Regularizer/mulMul1stream_0_conv_1/kernel/Regularizer/mul/x:output:0/stream_0_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/mulo
IdentityIdentityBiasAdd:output:0^NoOp*
T0*+
_output_shapes
:         }@2

Identity─
NoOpNoOp^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         }: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp2n
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:S O
+
_output_shapes
:         }
 
_user_specified_nameinputs
у
i
M__inference_dense_activation_1_layer_call_and_return_conditional_losses_28954

inputs
identityW
SigmoidSigmoidinputs*
T0*'
_output_shapes
:         T2	
Sigmoid_
IdentityIdentitySigmoid:y:0*
T0*'
_output_shapes
:         T2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         T:O K
'
_output_shapes
:         T
 
_user_specified_nameinputs
џ
│
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_24000

inputs0
!batchnorm_readvariableop_resource:	ђ4
%batchnorm_mul_readvariableop_resource:	ђ2
#batchnorm_readvariableop_1_resource:	ђ2
#batchnorm_readvariableop_2_resource:	ђ
identityѕбbatchnorm/ReadVariableOpбbatchnorm/ReadVariableOp_1бbatchnorm/ReadVariableOp_2бbatchnorm/mul/ReadVariableOpЊ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2
batchnorm/add/yЅ
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:ђ2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:ђ2
batchnorm/RsqrtЪ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
batchnorm/mul/ReadVariableOpє
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:ђ2
batchnorm/mul{
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*,
_output_shapes
:         }ђ2
batchnorm/mul_1Ў
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype02
batchnorm/ReadVariableOp_1є
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:ђ2
batchnorm/mul_2Ў
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:ђ*
dtype02
batchnorm/ReadVariableOp_2ё
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:ђ2
batchnorm/subі
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*,
_output_shapes
:         }ђ2
batchnorm/add_1s
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*,
_output_shapes
:         }ђ2

Identity┬
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         }ђ: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:T P
,
_output_shapes
:         }ђ
 
_user_specified_nameinputs
▒
ч
%__inference_model_layer_call_fn_26320
inputs_0
inputs_1
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:@ 
	unknown_5:@ђ
	unknown_6:	ђ
	unknown_7:	ђ
	unknown_8:	ђ
	unknown_9:	ђ

unknown_10:	ђ"

unknown_11:ђђ

unknown_12:	ђ

unknown_13:	ђ

unknown_14:	ђ

unknown_15:	ђ

unknown_16:	ђ

unknown_17:	ђT

unknown_18:T

unknown_19:T

unknown_20:T

unknown_21:T

unknown_22:T
identityѕбStatefulPartitionedCallФ
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22*%
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *:
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8ѓ *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_252912
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*q
_input_shapes`
^:         }:         }: : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
+
_output_shapes
:         }
"
_user_specified_name
inputs/0:UQ
+
_output_shapes
:         }
"
_user_specified_name
inputs/1
¤H
Г	
@__inference_model_layer_call_and_return_conditional_losses_26076
left_inputs
right_inputs%
basemodel_25976:@
basemodel_25978:@
basemodel_25980:@
basemodel_25982:@
basemodel_25984:@
basemodel_25986:@&
basemodel_25988:@ђ
basemodel_25990:	ђ
basemodel_25992:	ђ
basemodel_25994:	ђ
basemodel_25996:	ђ
basemodel_25998:	ђ'
basemodel_26000:ђђ
basemodel_26002:	ђ
basemodel_26004:	ђ
basemodel_26006:	ђ
basemodel_26008:	ђ
basemodel_26010:	ђ"
basemodel_26012:	ђT
basemodel_26014:T
basemodel_26016:T
basemodel_26018:T
basemodel_26020:T
basemodel_26022:T
identityѕб!basemodel/StatefulPartitionedCallб#basemodel/StatefulPartitionedCall_1б-dense_1/kernel/Regularizer/Abs/ReadVariableOpб5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpб8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOpб5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp└
!basemodel/StatefulPartitionedCallStatefulPartitionedCallleft_inputsbasemodel_25976basemodel_25978basemodel_25980basemodel_25982basemodel_25984basemodel_25986basemodel_25988basemodel_25990basemodel_25992basemodel_25994basemodel_25996basemodel_25998basemodel_26000basemodel_26002basemodel_26004basemodel_26006basemodel_26008basemodel_26010basemodel_26012basemodel_26014basemodel_26016basemodel_26018basemodel_26020basemodel_26022*$
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         T*:
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8ѓ *M
fHRF
D__inference_basemodel_layer_call_and_return_conditional_losses_251772#
!basemodel/StatefulPartitionedCall┼
#basemodel/StatefulPartitionedCall_1StatefulPartitionedCallright_inputsbasemodel_25976basemodel_25978basemodel_25980basemodel_25982basemodel_25984basemodel_25986basemodel_25988basemodel_25990basemodel_25992basemodel_25994basemodel_25996basemodel_25998basemodel_26000basemodel_26002basemodel_26004basemodel_26006basemodel_26008basemodel_26010basemodel_26012basemodel_26014basemodel_26016basemodel_26018basemodel_26020basemodel_26022*$
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         T*:
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8ѓ *M
fHRF
D__inference_basemodel_layer_call_and_return_conditional_losses_251772%
#basemodel/StatefulPartitionedCall_1Е
distance/PartitionedCallPartitionedCall*basemodel/StatefulPartitionedCall:output:0,basemodel/StatefulPartitionedCall_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *L
fGRE
C__inference_distance_layer_call_and_return_conditional_losses_252642
distance/PartitionedCall┬
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpbasemodel_25976*"
_output_shapes
:@*
dtype027
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp├
&stream_0_conv_1/kernel/Regularizer/AbsAbs=stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_0_conv_1/kernel/Regularizer/AbsЕ
(stream_0_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_0_conv_1/kernel/Regularizer/ConstО
&stream_0_conv_1/kernel/Regularizer/SumSum*stream_0_conv_1/kernel/Regularizer/Abs:y:01stream_0_conv_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/SumЎ
(stream_0_conv_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2*
(stream_0_conv_1/kernel/Regularizer/mul/x▄
&stream_0_conv_1/kernel/Regularizer/mulMul1stream_0_conv_1/kernel/Regularizer/mul/x:output:0/stream_0_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/mul╔
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpbasemodel_25988*#
_output_shapes
:@ђ*
dtype02:
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOpл
)stream_0_conv_2/kernel/Regularizer/SquareSquare@stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*#
_output_shapes
:@ђ2+
)stream_0_conv_2/kernel/Regularizer/SquareЕ
(stream_0_conv_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_0_conv_2/kernel/Regularizer/Const┌
&stream_0_conv_2/kernel/Regularizer/SumSum-stream_0_conv_2/kernel/Regularizer/Square:y:01stream_0_conv_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_2/kernel/Regularizer/SumЎ
(stream_0_conv_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2*
(stream_0_conv_2/kernel/Regularizer/mul/x▄
&stream_0_conv_2/kernel/Regularizer/mulMul1stream_0_conv_2/kernel/Regularizer/mul/x:output:0/stream_0_conv_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_2/kernel/Regularizer/mul─
5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpbasemodel_26000*$
_output_shapes
:ђђ*
dtype027
5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp┼
&stream_0_conv_3/kernel/Regularizer/AbsAbs=stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*$
_output_shapes
:ђђ2(
&stream_0_conv_3/kernel/Regularizer/AbsЕ
(stream_0_conv_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_0_conv_3/kernel/Regularizer/ConstО
&stream_0_conv_3/kernel/Regularizer/SumSum*stream_0_conv_3/kernel/Regularizer/Abs:y:01stream_0_conv_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_3/kernel/Regularizer/SumЎ
(stream_0_conv_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2*
(stream_0_conv_3/kernel/Regularizer/mul/x▄
&stream_0_conv_3/kernel/Regularizer/mulMul1stream_0_conv_3/kernel/Regularizer/mul/x:output:0/stream_0_conv_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_3/kernel/Regularizer/mul»
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpbasemodel_26012*
_output_shapes
:	ђT*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOpе
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	ђT2 
dense_1/kernel/Regularizer/AbsЋ
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_1/kernel/Regularizer/Constи
dense_1/kernel/Regularizer/SumSum"dense_1/kernel/Regularizer/Abs:y:0)dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/SumЅ
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2"
 dense_1/kernel/Regularizer/mul/x╝
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/mul|
IdentityIdentity!distance/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         2

Identityз
NoOpNoOp"^basemodel/StatefulPartitionedCall$^basemodel/StatefulPartitionedCall_1.^dense_1/kernel/Regularizer/Abs/ReadVariableOp6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp9^stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp6^stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*q
_input_shapes`
^:         }:         }: : : : : : : : : : : : : : : : : : : : : : : : 2F
!basemodel/StatefulPartitionedCall!basemodel/StatefulPartitionedCall2J
#basemodel/StatefulPartitionedCall_1#basemodel/StatefulPartitionedCall_12^
-dense_1/kernel/Regularizer/Abs/ReadVariableOp-dense_1/kernel/Regularizer/Abs/ReadVariableOp2n
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp2t
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp2n
5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp:X T
+
_output_shapes
:         }
%
_user_specified_nameleft_inputs:YU
+
_output_shapes
:         }
&
_user_specified_nameright_inputs
Ы
ц
B__inference_dense_1_layer_call_and_return_conditional_losses_28864

inputs1
matmul_readvariableop_resource:	ђT-
biasadd_readvariableop_resource:T
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpб-dense_1/kernel/Regularizer/Abs/ReadVariableOpј
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	ђT*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         T2
MatMulї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:T*
dtype02
BiasAdd/ReadVariableOpЂ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         T2	
BiasAddЙ
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	ђT*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOpе
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	ђT2 
dense_1/kernel/Regularizer/AbsЋ
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_1/kernel/Regularizer/Constи
dense_1/kernel/Regularizer/SumSum"dense_1/kernel/Regularizer/Abs:y:0)dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/SumЅ
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2"
 dense_1/kernel/Regularizer/mul/x╝
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/mulk
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         T2

Identity»
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp.^dense_1/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ђ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2^
-dense_1/kernel/Regularizer/Abs/ReadVariableOp-dense_1/kernel/Regularizer/Abs/ReadVariableOp:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
З
»
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_23732

inputs/
!batchnorm_readvariableop_resource:T3
%batchnorm_mul_readvariableop_resource:T1
#batchnorm_readvariableop_1_resource:T1
#batchnorm_readvariableop_2_resource:T
identityѕбbatchnorm/ReadVariableOpбbatchnorm/ReadVariableOp_1бbatchnorm/ReadVariableOp_2бbatchnorm/mul/ReadVariableOpњ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:T*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2
batchnorm/add/yѕ
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:T2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:T2
batchnorm/Rsqrtъ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:T*
dtype02
batchnorm/mul/ReadVariableOpЁ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:T2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:         T2
batchnorm/mul_1ў
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:T*
dtype02
batchnorm/ReadVariableOp_1Ё
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:T2
batchnorm/mul_2ў
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:T*
dtype02
batchnorm/ReadVariableOp_2Ѓ
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:T2
batchnorm/subЁ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:         T2
batchnorm/add_1n
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:         T2

Identity┬
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         T: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:         T
 
_user_specified_nameinputs
ч
Е
__inference_loss_fn_3_28998I
6dense_1_kernel_regularizer_abs_readvariableop_resource:	ђT
identityѕб-dense_1/kernel/Regularizer/Abs/ReadVariableOpо
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp6dense_1_kernel_regularizer_abs_readvariableop_resource*
_output_shapes
:	ђT*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOpе
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	ђT2 
dense_1/kernel/Regularizer/AbsЋ
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_1/kernel/Regularizer/Constи
dense_1/kernel/Regularizer/SumSum"dense_1/kernel/Regularizer/Abs:y:0)dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/SumЅ
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2"
 dense_1/kernel/Regularizer/mul/x╝
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/mull
IdentityIdentity"dense_1/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: 2

Identity~
NoOpNoOp.^dense_1/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2^
-dense_1/kernel/Regularizer/Abs/ReadVariableOp-dense_1/kernel/Regularizer/Abs/ReadVariableOp
З
i
J__inference_stream_0_drop_2_layer_call_and_return_conditional_losses_24385

inputs
identityѕc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUН?2
dropout/Constx
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:         }ђ2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shapeн
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:         }ђ*
dtype0*
seedи*
seed2и2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠>2
dropout/GreaterEqual/y├
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:         }ђ2
dropout/GreaterEqualё
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:         }ђ2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:         }ђ2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:         }ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         }ђ:T P
,
_output_shapes
:         }ђ
 
_user_specified_nameinputs
М+
ь
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_23444

inputs6
'assignmovingavg_readvariableop_resource:	ђ8
)assignmovingavg_1_readvariableop_resource:	ђ4
%batchnorm_mul_readvariableop_resource:	ђ0
!batchnorm_readvariableop_resource:	ђ
identityѕбAssignMovingAvgбAssignMovingAvg/ReadVariableOpбAssignMovingAvg_1б AssignMovingAvg_1/ReadVariableOpбbatchnorm/ReadVariableOpбbatchnorm/mul/ReadVariableOpЉ
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indicesћ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*#
_output_shapes
:ђ*
	keep_dims(2
moments/meanЂ
moments/StopGradientStopGradientmoments/mean:output:0*
T0*#
_output_shapes
:ђ2
moments/StopGradient▓
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*5
_output_shapes#
!:                  ђ2
moments/SquaredDifferenceЎ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indicesи
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*#
_output_shapes
:ђ*
	keep_dims(2
moments/varianceѓ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:ђ*
squeeze_dims
 2
moments/Squeezeі
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:ђ*
squeeze_dims
 2
moments/Squeeze_1s
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2
AssignMovingAvg/decayЦ
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:ђ*
dtype02 
AssignMovingAvg/ReadVariableOpЎ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:ђ2
AssignMovingAvg/subљ
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:ђ2
AssignMovingAvg/mul┐
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvgw
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2
AssignMovingAvg_1/decayФ
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:ђ*
dtype02"
 AssignMovingAvg_1/ReadVariableOpА
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:ђ2
AssignMovingAvg_1/subў
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:ђ2
AssignMovingAvg_1/mul╔
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1g
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2
batchnorm/add/yЃ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:ђ2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:ђ2
batchnorm/RsqrtЪ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
batchnorm/mul/ReadVariableOpє
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:ђ2
batchnorm/mulё
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*5
_output_shapes#
!:                  ђ2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:ђ2
batchnorm/mul_2Њ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
batchnorm/ReadVariableOpѓ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:ђ2
batchnorm/subЊ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*5
_output_shapes#
!:                  ђ2
batchnorm/add_1|
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*5
_output_shapes#
!:                  ђ2

IdentityЫ
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):                  ђ: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:] Y
5
_output_shapes#
!:                  ђ
 
_user_specified_nameinputs
╣
o
S__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_28800

inputs
identityr
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Mean/reduction_indicesx
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:                  2
Meanj
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:                  2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'                           :e a
=
_output_shapes+
):'                           
 
_user_specified_nameinputs
ц
ђ
#__inference_signature_wrapper_26266
left_inputs
right_inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:@ 
	unknown_5:@ђ
	unknown_6:	ђ
	unknown_7:	ђ
	unknown_8:	ђ
	unknown_9:	ђ

unknown_10:	ђ"

unknown_11:ђђ

unknown_12:	ђ

unknown_13:	ђ

unknown_14:	ђ

unknown_15:	ђ

unknown_16:	ђ

unknown_17:	ђT

unknown_18:T

unknown_19:T

unknown_20:T

unknown_21:T

unknown_22:T
identityѕбStatefulPartitionedCallњ
StatefulPartitionedCallStatefulPartitionedCallleft_inputsright_inputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22*%
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *:
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8ѓ *)
f$R"
 __inference__wrapped_model_231982
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*q
_input_shapes`
^:         }:         }: : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
+
_output_shapes
:         }
%
_user_specified_nameleft_inputs:YU
+
_output_shapes
:         }
&
_user_specified_nameright_inputs
ћ
а
/__inference_stream_0_conv_1_layer_call_fn_28100

inputs
unknown:@
	unknown_0:@
identityѕбStatefulPartitionedCallЂ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         }@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *S
fNRL
J__inference_stream_0_conv_1_layer_call_and_return_conditional_losses_239052
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         }@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         }: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:         }
 
_user_specified_nameinputs
х+
у
N__inference_batch_normalization_layer_call_and_return_conditional_losses_23282

inputs5
'assignmovingavg_readvariableop_resource:@7
)assignmovingavg_1_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@/
!batchnorm_readvariableop_resource:@
identityѕбAssignMovingAvgбAssignMovingAvg/ReadVariableOpбAssignMovingAvg_1б AssignMovingAvg_1/ReadVariableOpбbatchnorm/ReadVariableOpбbatchnorm/mul/ReadVariableOpЉ
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indicesЊ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(2
moments/meanђ
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:@2
moments/StopGradient▒
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :                  @2
moments/SquaredDifferenceЎ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indicesХ
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(2
moments/varianceЂ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 2
moments/SqueezeЅ
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 2
moments/Squeeze_1s
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2
AssignMovingAvg/decayц
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOpў
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg/subЈ
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg/mul┐
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvgw
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2
AssignMovingAvg_1/decayф
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOpа
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg_1/subЌ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg_1/mul╔
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1g
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2
batchnorm/add/yѓ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@2
batchnorm/Rsqrtъ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOpЁ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2
batchnorm/mulЃ
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :                  @2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:@2
batchnorm/mul_2њ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOpЂ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2
batchnorm/subњ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :                  @2
batchnorm/add_1{
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :                  @2

IdentityЫ
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:                  @: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :                  @
 
_user_specified_nameinputs
ю+
ь
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_28747

inputs6
'assignmovingavg_readvariableop_resource:	ђ8
)assignmovingavg_1_readvariableop_resource:	ђ4
%batchnorm_mul_readvariableop_resource:	ђ0
!batchnorm_readvariableop_resource:	ђ
identityѕбAssignMovingAvgбAssignMovingAvg/ReadVariableOpбAssignMovingAvg_1б AssignMovingAvg_1/ReadVariableOpбbatchnorm/ReadVariableOpбbatchnorm/mul/ReadVariableOpЉ
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indicesћ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*#
_output_shapes
:ђ*
	keep_dims(2
moments/meanЂ
moments/StopGradientStopGradientmoments/mean:output:0*
T0*#
_output_shapes
:ђ2
moments/StopGradientЕ
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*,
_output_shapes
:         }ђ2
moments/SquaredDifferenceЎ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indicesи
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*#
_output_shapes
:ђ*
	keep_dims(2
moments/varianceѓ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:ђ*
squeeze_dims
 2
moments/Squeezeі
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:ђ*
squeeze_dims
 2
moments/Squeeze_1s
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2
AssignMovingAvg/decayЦ
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:ђ*
dtype02 
AssignMovingAvg/ReadVariableOpЎ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:ђ2
AssignMovingAvg/subљ
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:ђ2
AssignMovingAvg/mul┐
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvgw
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2
AssignMovingAvg_1/decayФ
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:ђ*
dtype02"
 AssignMovingAvg_1/ReadVariableOpА
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:ђ2
AssignMovingAvg_1/subў
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:ђ2
AssignMovingAvg_1/mul╔
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1g
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2
batchnorm/add/yЃ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:ђ2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:ђ2
batchnorm/RsqrtЪ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
batchnorm/mul/ReadVariableOpє
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:ђ2
batchnorm/mul{
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*,
_output_shapes
:         }ђ2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:ђ2
batchnorm/mul_2Њ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
batchnorm/ReadVariableOpѓ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:ђ2
batchnorm/subі
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*,
_output_shapes
:         }ђ2
batchnorm/add_1s
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*,
_output_shapes
:         }ђ2

IdentityЫ
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         }ђ: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:T P
,
_output_shapes
:         }ђ
 
_user_specified_nameinputs
└	
o
C__inference_distance_layer_call_and_return_conditional_losses_28046
inputs_0
inputs_1
identityW
subSubinputs_0inputs_1*
T0*'
_output_shapes
:         T2
subU
SquareSquaresub:z:0*
T0*'
_output_shapes
:         T2
Squarey
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
         2
Sum/reduction_indicesђ
SumSum
Square:y:0Sum/reduction_indices:output:0*
T0*'
_output_shapes
:         *
	keep_dims(2
SumS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
Constm
MaximumMaximumSum:output:0Const:output:0*
T0*'
_output_shapes
:         2	
MaximumS
SqrtSqrtMaximum:z:0*
T0*'
_output_shapes
:         2
Sqrt\
IdentityIdentitySqrt:y:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:         T:         T:Q M
'
_output_shapes
:         T
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:         T
"
_user_specified_name
inputs/1
▓H
ц	
@__inference_model_layer_call_and_return_conditional_losses_25291

inputs
inputs_1%
basemodel_25178:@
basemodel_25180:@
basemodel_25182:@
basemodel_25184:@
basemodel_25186:@
basemodel_25188:@&
basemodel_25190:@ђ
basemodel_25192:	ђ
basemodel_25194:	ђ
basemodel_25196:	ђ
basemodel_25198:	ђ
basemodel_25200:	ђ'
basemodel_25202:ђђ
basemodel_25204:	ђ
basemodel_25206:	ђ
basemodel_25208:	ђ
basemodel_25210:	ђ
basemodel_25212:	ђ"
basemodel_25214:	ђT
basemodel_25216:T
basemodel_25218:T
basemodel_25220:T
basemodel_25222:T
basemodel_25224:T
identityѕб!basemodel/StatefulPartitionedCallб#basemodel/StatefulPartitionedCall_1б-dense_1/kernel/Regularizer/Abs/ReadVariableOpб5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpб8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOpб5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp╗
!basemodel/StatefulPartitionedCallStatefulPartitionedCallinputsbasemodel_25178basemodel_25180basemodel_25182basemodel_25184basemodel_25186basemodel_25188basemodel_25190basemodel_25192basemodel_25194basemodel_25196basemodel_25198basemodel_25200basemodel_25202basemodel_25204basemodel_25206basemodel_25208basemodel_25210basemodel_25212basemodel_25214basemodel_25216basemodel_25218basemodel_25220basemodel_25222basemodel_25224*$
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         T*:
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8ѓ *M
fHRF
D__inference_basemodel_layer_call_and_return_conditional_losses_251772#
!basemodel/StatefulPartitionedCall┴
#basemodel/StatefulPartitionedCall_1StatefulPartitionedCallinputs_1basemodel_25178basemodel_25180basemodel_25182basemodel_25184basemodel_25186basemodel_25188basemodel_25190basemodel_25192basemodel_25194basemodel_25196basemodel_25198basemodel_25200basemodel_25202basemodel_25204basemodel_25206basemodel_25208basemodel_25210basemodel_25212basemodel_25214basemodel_25216basemodel_25218basemodel_25220basemodel_25222basemodel_25224*$
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         T*:
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8ѓ *M
fHRF
D__inference_basemodel_layer_call_and_return_conditional_losses_251772%
#basemodel/StatefulPartitionedCall_1Е
distance/PartitionedCallPartitionedCall*basemodel/StatefulPartitionedCall:output:0,basemodel/StatefulPartitionedCall_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *L
fGRE
C__inference_distance_layer_call_and_return_conditional_losses_252642
distance/PartitionedCall┬
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpbasemodel_25178*"
_output_shapes
:@*
dtype027
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp├
&stream_0_conv_1/kernel/Regularizer/AbsAbs=stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_0_conv_1/kernel/Regularizer/AbsЕ
(stream_0_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_0_conv_1/kernel/Regularizer/ConstО
&stream_0_conv_1/kernel/Regularizer/SumSum*stream_0_conv_1/kernel/Regularizer/Abs:y:01stream_0_conv_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/SumЎ
(stream_0_conv_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2*
(stream_0_conv_1/kernel/Regularizer/mul/x▄
&stream_0_conv_1/kernel/Regularizer/mulMul1stream_0_conv_1/kernel/Regularizer/mul/x:output:0/stream_0_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/mul╔
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpbasemodel_25190*#
_output_shapes
:@ђ*
dtype02:
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOpл
)stream_0_conv_2/kernel/Regularizer/SquareSquare@stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*#
_output_shapes
:@ђ2+
)stream_0_conv_2/kernel/Regularizer/SquareЕ
(stream_0_conv_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_0_conv_2/kernel/Regularizer/Const┌
&stream_0_conv_2/kernel/Regularizer/SumSum-stream_0_conv_2/kernel/Regularizer/Square:y:01stream_0_conv_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_2/kernel/Regularizer/SumЎ
(stream_0_conv_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2*
(stream_0_conv_2/kernel/Regularizer/mul/x▄
&stream_0_conv_2/kernel/Regularizer/mulMul1stream_0_conv_2/kernel/Regularizer/mul/x:output:0/stream_0_conv_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_2/kernel/Regularizer/mul─
5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpbasemodel_25202*$
_output_shapes
:ђђ*
dtype027
5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp┼
&stream_0_conv_3/kernel/Regularizer/AbsAbs=stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*$
_output_shapes
:ђђ2(
&stream_0_conv_3/kernel/Regularizer/AbsЕ
(stream_0_conv_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_0_conv_3/kernel/Regularizer/ConstО
&stream_0_conv_3/kernel/Regularizer/SumSum*stream_0_conv_3/kernel/Regularizer/Abs:y:01stream_0_conv_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_3/kernel/Regularizer/SumЎ
(stream_0_conv_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2*
(stream_0_conv_3/kernel/Regularizer/mul/x▄
&stream_0_conv_3/kernel/Regularizer/mulMul1stream_0_conv_3/kernel/Regularizer/mul/x:output:0/stream_0_conv_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_3/kernel/Regularizer/mul»
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpbasemodel_25214*
_output_shapes
:	ђT*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOpе
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	ђT2 
dense_1/kernel/Regularizer/AbsЋ
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_1/kernel/Regularizer/Constи
dense_1/kernel/Regularizer/SumSum"dense_1/kernel/Regularizer/Abs:y:0)dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/SumЅ
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2"
 dense_1/kernel/Regularizer/mul/x╝
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/mul|
IdentityIdentity!distance/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         2

Identityз
NoOpNoOp"^basemodel/StatefulPartitionedCall$^basemodel/StatefulPartitionedCall_1.^dense_1/kernel/Regularizer/Abs/ReadVariableOp6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp9^stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp6^stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*q
_input_shapes`
^:         }:         }: : : : : : : : : : : : : : : : : : : : : : : : 2F
!basemodel/StatefulPartitionedCall!basemodel/StatefulPartitionedCall2J
#basemodel/StatefulPartitionedCall_1#basemodel/StatefulPartitionedCall_12^
-dense_1/kernel/Regularizer/Abs/ReadVariableOp-dense_1/kernel/Regularizer/Abs/ReadVariableOp2n
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp2t
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp2n
5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp:S O
+
_output_shapes
:         }
 
_user_specified_nameinputs:SO
+
_output_shapes
:         }
 
_user_specified_nameinputs
п
л
5__inference_batch_normalization_3_layer_call_fn_28877

inputs
unknown:T
	unknown_0:T
	unknown_1:T
	unknown_2:T
identityѕбStatefulPartitionedCallЮ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         T*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Y
fTRR
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_237322
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         T2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         T: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         T
 
_user_specified_nameinputs
о
л
5__inference_batch_normalization_3_layer_call_fn_28890

inputs
unknown:T
	unknown_0:T
	unknown_1:T
	unknown_2:T
identityѕбStatefulPartitionedCallЏ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         T*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Y
fTRR
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_237922
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         T2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         T: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         T
 
_user_specified_nameinputs
┼
l
3__inference_stream_0_input_drop_layer_call_fn_28068

inputs
identityѕбStatefulPartitionedCallв
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         }* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *W
fRRP
N__inference_stream_0_input_drop_layer_call_and_return_conditional_losses_245832
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         }2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         }22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:         }
 
_user_specified_nameinputs
х+
у
N__inference_batch_normalization_layer_call_and_return_conditional_losses_28227

inputs5
'assignmovingavg_readvariableop_resource:@7
)assignmovingavg_1_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@/
!batchnorm_readvariableop_resource:@
identityѕбAssignMovingAvgбAssignMovingAvg/ReadVariableOpбAssignMovingAvg_1б AssignMovingAvg_1/ReadVariableOpбbatchnorm/ReadVariableOpбbatchnorm/mul/ReadVariableOpЉ
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indicesЊ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(2
moments/meanђ
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:@2
moments/StopGradient▒
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :                  @2
moments/SquaredDifferenceЎ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indicesХ
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(2
moments/varianceЂ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 2
moments/SqueezeЅ
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 2
moments/Squeeze_1s
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2
AssignMovingAvg/decayц
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOpў
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg/subЈ
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg/mul┐
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvgw
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2
AssignMovingAvg_1/decayф
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOpа
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg_1/subЌ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg_1/mul╔
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1g
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2
batchnorm/add/yѓ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@2
batchnorm/Rsqrtъ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOpЁ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2
batchnorm/mulЃ
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :                  @2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:@2
batchnorm/mul_2њ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOpЂ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2
batchnorm/subњ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :                  @2
batchnorm/add_1{
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :                  @2

IdentityЫ
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:                  @: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :                  @
 
_user_specified_nameinputs
№
н
5__inference_batch_normalization_1_layer_call_fn_28406

inputs
unknown:	ђ
	unknown_0:	ђ
	unknown_1:	ђ
	unknown_2:	ђ
identityѕбStatefulPartitionedCallа
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         }ђ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_244432
StatefulPartitionedCallђ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:         }ђ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         }ђ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:         }ђ
 
_user_specified_nameinputs
ч
h
J__inference_dense_1_dropout_layer_call_and_return_conditional_losses_24106

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:         ђ2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:         ђ2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         ђ:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
к
ѓ
%__inference_model_layer_call_fn_25342
left_inputs
right_inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:@ 
	unknown_5:@ђ
	unknown_6:	ђ
	unknown_7:	ђ
	unknown_8:	ђ
	unknown_9:	ђ

unknown_10:	ђ"

unknown_11:ђђ

unknown_12:	ђ

unknown_13:	ђ

unknown_14:	ђ

unknown_15:	ђ

unknown_16:	ђ

unknown_17:	ђT

unknown_18:T

unknown_19:T

unknown_20:T

unknown_21:T

unknown_22:T
identityѕбStatefulPartitionedCall▓
StatefulPartitionedCallStatefulPartitionedCallleft_inputsright_inputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22*%
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *:
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8ѓ *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_252912
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*q
_input_shapes`
^:         }:         }: : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
+
_output_shapes
:         }
%
_user_specified_nameleft_inputs:YU
+
_output_shapes
:         }
&
_user_specified_nameright_inputs
Ћ	
н
5__inference_batch_normalization_1_layer_call_fn_28367

inputs
unknown:	ђ
	unknown_0:	ђ
	unknown_1:	ђ
	unknown_2:	ђ
identityѕбStatefulPartitionedCallФ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:                  ђ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_233842
StatefulPartitionedCallЅ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:                  ђ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):                  ђ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:                  ђ
 
_user_specified_nameinputs
┘
H
,__inference_activation_2_layer_call_fn_28752

inputs
identity═
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         }ђ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *P
fKRI
G__inference_activation_2_layer_call_and_return_conditional_losses_240852
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:         }ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         }ђ:T P
,
_output_shapes
:         }ђ
 
_user_specified_nameinputs
Њ	
н
5__inference_batch_normalization_2_layer_call_fn_28613

inputs
unknown:	ђ
	unknown_0:	ђ
	unknown_1:	ђ
	unknown_2:	ђ
identityѕбStatefulPartitionedCallЕ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:                  ђ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Y
fTRR
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_236062
StatefulPartitionedCallЅ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:                  ђ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):                  ђ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:                  ђ
 
_user_specified_nameinputs
з
├
__inference_loss_fn_1_28976X
Astream_0_conv_2_kernel_regularizer_square_readvariableop_resource:@ђ
identityѕб8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOpч
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpAstream_0_conv_2_kernel_regularizer_square_readvariableop_resource*#
_output_shapes
:@ђ*
dtype02:
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOpл
)stream_0_conv_2/kernel/Regularizer/SquareSquare@stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*#
_output_shapes
:@ђ2+
)stream_0_conv_2/kernel/Regularizer/SquareЕ
(stream_0_conv_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_0_conv_2/kernel/Regularizer/Const┌
&stream_0_conv_2/kernel/Regularizer/SumSum-stream_0_conv_2/kernel/Regularizer/Square:y:01stream_0_conv_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_2/kernel/Regularizer/SumЎ
(stream_0_conv_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2*
(stream_0_conv_2/kernel/Regularizer/mul/x▄
&stream_0_conv_2/kernel/Regularizer/mulMul1stream_0_conv_2/kernel/Regularizer/mul/x:output:0/stream_0_conv_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_2/kernel/Regularizer/mult
IdentityIdentity*stream_0_conv_2/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: 2

IdentityЅ
NoOpNoOp9^stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2t
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp
ю+
ь
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_24443

inputs6
'assignmovingavg_readvariableop_resource:	ђ8
)assignmovingavg_1_readvariableop_resource:	ђ4
%batchnorm_mul_readvariableop_resource:	ђ0
!batchnorm_readvariableop_resource:	ђ
identityѕбAssignMovingAvgбAssignMovingAvg/ReadVariableOpбAssignMovingAvg_1б AssignMovingAvg_1/ReadVariableOpбbatchnorm/ReadVariableOpбbatchnorm/mul/ReadVariableOpЉ
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indicesћ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*#
_output_shapes
:ђ*
	keep_dims(2
moments/meanЂ
moments/StopGradientStopGradientmoments/mean:output:0*
T0*#
_output_shapes
:ђ2
moments/StopGradientЕ
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*,
_output_shapes
:         }ђ2
moments/SquaredDifferenceЎ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indicesи
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*#
_output_shapes
:ђ*
	keep_dims(2
moments/varianceѓ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:ђ*
squeeze_dims
 2
moments/Squeezeі
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:ђ*
squeeze_dims
 2
moments/Squeeze_1s
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2
AssignMovingAvg/decayЦ
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:ђ*
dtype02 
AssignMovingAvg/ReadVariableOpЎ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:ђ2
AssignMovingAvg/subљ
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:ђ2
AssignMovingAvg/mul┐
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvgw
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2
AssignMovingAvg_1/decayФ
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:ђ*
dtype02"
 AssignMovingAvg_1/ReadVariableOpА
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:ђ2
AssignMovingAvg_1/subў
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:ђ2
AssignMovingAvg_1/mul╔
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1g
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2
batchnorm/add/yЃ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:ђ2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:ђ2
batchnorm/RsqrtЪ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
batchnorm/mul/ReadVariableOpє
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:ђ2
batchnorm/mul{
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*,
_output_shapes
:         }ђ2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:ђ2
batchnorm/mul_2Њ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
batchnorm/ReadVariableOpѓ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:ђ2
batchnorm/subі
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*,
_output_shapes
:         }ђ2
batchnorm/add_1s
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*,
_output_shapes
:         }ђ2

IdentityЫ
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         }ђ: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:T P
,
_output_shapes
:         }ђ
 
_user_specified_nameinputs
ь
a
E__inference_activation_layer_call_and_return_conditional_losses_28291

inputs
identityR
ReluReluinputs*
T0*+
_output_shapes
:         }@2
Reluj
IdentityIdentityRelu:activations:0*
T0*+
_output_shapes
:         }@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         }@:S O
+
_output_shapes
:         }@
 
_user_specified_nameinputs
З
i
J__inference_stream_0_drop_2_layer_call_and_return_conditional_losses_28551

inputs
identityѕc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUН?2
dropout/Constx
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:         }ђ2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shapeн
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:         }ђ*
dtype0*
seedи*
seed2и2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠>2
dropout/GreaterEqual/y├
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:         }ђ2
dropout/GreaterEqualё
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:         }ђ2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:         }ђ2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:         }ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         }ђ:T P
,
_output_shapes
:         }ђ
 
_user_specified_nameinputs
єѓ
є
D__inference_basemodel_layer_call_and_return_conditional_losses_24171

inputs+
stream_0_conv_1_23906:@#
stream_0_conv_1_23908:@'
batch_normalization_23931:@'
batch_normalization_23933:@'
batch_normalization_23935:@'
batch_normalization_23937:@,
stream_0_conv_2_23976:@ђ$
stream_0_conv_2_23978:	ђ*
batch_normalization_1_24001:	ђ*
batch_normalization_1_24003:	ђ*
batch_normalization_1_24005:	ђ*
batch_normalization_1_24007:	ђ-
stream_0_conv_3_24046:ђђ$
stream_0_conv_3_24048:	ђ*
batch_normalization_2_24071:	ђ*
batch_normalization_2_24073:	ђ*
batch_normalization_2_24075:	ђ*
batch_normalization_2_24077:	ђ 
dense_1_24125:	ђT
dense_1_24127:T)
batch_normalization_3_24130:T)
batch_normalization_3_24132:T)
batch_normalization_3_24134:T)
batch_normalization_3_24136:T
identityѕб+batch_normalization/StatefulPartitionedCallб-batch_normalization_1/StatefulPartitionedCallб-batch_normalization_2/StatefulPartitionedCallб-batch_normalization_3/StatefulPartitionedCallбdense_1/StatefulPartitionedCallб-dense_1/kernel/Regularizer/Abs/ReadVariableOpб'stream_0_conv_1/StatefulPartitionedCallб5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpб'stream_0_conv_2/StatefulPartitionedCallб8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOpб'stream_0_conv_3/StatefulPartitionedCallб5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOpч
#stream_0_input_drop/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         }* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *W
fRRP
N__inference_stream_0_input_drop_layer_call_and_return_conditional_losses_238822%
#stream_0_input_drop/PartitionedCallр
'stream_0_conv_1/StatefulPartitionedCallStatefulPartitionedCall,stream_0_input_drop/PartitionedCall:output:0stream_0_conv_1_23906stream_0_conv_1_23908*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         }@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *S
fNRL
J__inference_stream_0_conv_1_layer_call_and_return_conditional_losses_239052)
'stream_0_conv_1/StatefulPartitionedCall│
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall0stream_0_conv_1/StatefulPartitionedCall:output:0batch_normalization_23931batch_normalization_23933batch_normalization_23935batch_normalization_23937*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         }@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_239302-
+batch_normalization/StatefulPartitionedCallј
activation/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         }@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_activation_layer_call_and_return_conditional_losses_239452
activation/PartitionedCallї
stream_0_drop_1/PartitionedCallPartitionedCall#activation/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         }@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *S
fNRL
J__inference_stream_0_drop_1_layer_call_and_return_conditional_losses_239522!
stream_0_drop_1/PartitionedCallя
'stream_0_conv_2/StatefulPartitionedCallStatefulPartitionedCall(stream_0_drop_1/PartitionedCall:output:0stream_0_conv_2_23976stream_0_conv_2_23978*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         }ђ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *S
fNRL
J__inference_stream_0_conv_2_layer_call_and_return_conditional_losses_239752)
'stream_0_conv_2/StatefulPartitionedCall┬
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall0stream_0_conv_2/StatefulPartitionedCall:output:0batch_normalization_1_24001batch_normalization_1_24003batch_normalization_1_24005batch_normalization_1_24007*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         }ђ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_240002/
-batch_normalization_1/StatefulPartitionedCallЌ
activation_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         }ђ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *P
fKRI
G__inference_activation_1_layer_call_and_return_conditional_losses_240152
activation_1/PartitionedCallЈ
stream_0_drop_2/PartitionedCallPartitionedCall%activation_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         }ђ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *S
fNRL
J__inference_stream_0_drop_2_layer_call_and_return_conditional_losses_240222!
stream_0_drop_2/PartitionedCallя
'stream_0_conv_3/StatefulPartitionedCallStatefulPartitionedCall(stream_0_drop_2/PartitionedCall:output:0stream_0_conv_3_24046stream_0_conv_3_24048*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         }ђ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *S
fNRL
J__inference_stream_0_conv_3_layer_call_and_return_conditional_losses_240452)
'stream_0_conv_3/StatefulPartitionedCall┬
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall0stream_0_conv_3/StatefulPartitionedCall:output:0batch_normalization_2_24071batch_normalization_2_24073batch_normalization_2_24075batch_normalization_2_24077*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         }ђ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Y
fTRR
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_240702/
-batch_normalization_2/StatefulPartitionedCallЌ
activation_2/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         }ђ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *P
fKRI
G__inference_activation_2_layer_call_and_return_conditional_losses_240852
activation_2/PartitionedCallЈ
stream_0_drop_3/PartitionedCallPartitionedCall%activation_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         }ђ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *S
fNRL
J__inference_stream_0_drop_3_layer_call_and_return_conditional_losses_240922!
stream_0_drop_3/PartitionedCallЕ
(global_average_pooling1d/PartitionedCallPartitionedCall(stream_0_drop_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *\
fWRU
S__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_240992*
(global_average_pooling1d/PartitionedCallЌ
dense_1_dropout/PartitionedCallPartitionedCall1global_average_pooling1d/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *S
fNRL
J__inference_dense_1_dropout_layer_call_and_return_conditional_losses_241062!
dense_1_dropout/PartitionedCall▒
dense_1/StatefulPartitionedCallStatefulPartitionedCall(dense_1_dropout/PartitionedCall:output:0dense_1_24125dense_1_24127*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         T*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_241242!
dense_1/StatefulPartitionedCallх
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0batch_normalization_3_24130batch_normalization_3_24132batch_normalization_3_24134batch_normalization_3_24136*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         T*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Y
fTRR
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_237322/
-batch_normalization_3/StatefulPartitionedCallц
"dense_activation_1/PartitionedCallPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         T* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *V
fQRO
M__inference_dense_activation_1_layer_call_and_return_conditional_losses_241442$
"dense_activation_1/PartitionedCall╚
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpstream_0_conv_1_23906*"
_output_shapes
:@*
dtype027
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp├
&stream_0_conv_1/kernel/Regularizer/AbsAbs=stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_0_conv_1/kernel/Regularizer/AbsЕ
(stream_0_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_0_conv_1/kernel/Regularizer/ConstО
&stream_0_conv_1/kernel/Regularizer/SumSum*stream_0_conv_1/kernel/Regularizer/Abs:y:01stream_0_conv_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/SumЎ
(stream_0_conv_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2*
(stream_0_conv_1/kernel/Regularizer/mul/x▄
&stream_0_conv_1/kernel/Regularizer/mulMul1stream_0_conv_1/kernel/Regularizer/mul/x:output:0/stream_0_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/mul¤
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpstream_0_conv_2_23976*#
_output_shapes
:@ђ*
dtype02:
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOpл
)stream_0_conv_2/kernel/Regularizer/SquareSquare@stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*#
_output_shapes
:@ђ2+
)stream_0_conv_2/kernel/Regularizer/SquareЕ
(stream_0_conv_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_0_conv_2/kernel/Regularizer/Const┌
&stream_0_conv_2/kernel/Regularizer/SumSum-stream_0_conv_2/kernel/Regularizer/Square:y:01stream_0_conv_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_2/kernel/Regularizer/SumЎ
(stream_0_conv_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2*
(stream_0_conv_2/kernel/Regularizer/mul/x▄
&stream_0_conv_2/kernel/Regularizer/mulMul1stream_0_conv_2/kernel/Regularizer/mul/x:output:0/stream_0_conv_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_2/kernel/Regularizer/mul╩
5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpstream_0_conv_3_24046*$
_output_shapes
:ђђ*
dtype027
5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp┼
&stream_0_conv_3/kernel/Regularizer/AbsAbs=stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*$
_output_shapes
:ђђ2(
&stream_0_conv_3/kernel/Regularizer/AbsЕ
(stream_0_conv_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_0_conv_3/kernel/Regularizer/ConstО
&stream_0_conv_3/kernel/Regularizer/SumSum*stream_0_conv_3/kernel/Regularizer/Abs:y:01stream_0_conv_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_3/kernel/Regularizer/SumЎ
(stream_0_conv_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2*
(stream_0_conv_3/kernel/Regularizer/mul/x▄
&stream_0_conv_3/kernel/Regularizer/mulMul1stream_0_conv_3/kernel/Regularizer/mul/x:output:0/stream_0_conv_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_3/kernel/Regularizer/mulГ
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_1_24125*
_output_shapes
:	ђT*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOpе
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	ђT2 
dense_1/kernel/Regularizer/AbsЋ
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_1/kernel/Regularizer/Constи
dense_1/kernel/Regularizer/SumSum"dense_1/kernel/Regularizer/Abs:y:0)dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/SumЅ
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2"
 dense_1/kernel/Regularizer/mul/x╝
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/mulє
IdentityIdentity+dense_activation_1/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         T2

IdentityЄ
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall.^dense_1/kernel/Regularizer/Abs/ReadVariableOp(^stream_0_conv_1/StatefulPartitionedCall6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp(^stream_0_conv_2/StatefulPartitionedCall9^stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp(^stream_0_conv_3/StatefulPartitionedCall6^stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:         }: : : : : : : : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2^
-dense_1/kernel/Regularizer/Abs/ReadVariableOp-dense_1/kernel/Regularizer/Abs/ReadVariableOp2R
'stream_0_conv_1/StatefulPartitionedCall'stream_0_conv_1/StatefulPartitionedCall2n
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp2R
'stream_0_conv_2/StatefulPartitionedCall'stream_0_conv_2/StatefulPartitionedCall2t
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp2R
'stream_0_conv_3/StatefulPartitionedCall'stream_0_conv_3/StatefulPartitionedCall2n
5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp:S O
+
_output_shapes
:         }
 
_user_specified_nameinputs
ф
ы
)__inference_basemodel_layer_call_fn_24839
inputs_0
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:@ 
	unknown_5:@ђ
	unknown_6:	ђ
	unknown_7:	ђ
	unknown_8:	ђ
	unknown_9:	ђ

unknown_10:	ђ"

unknown_11:ђђ

unknown_12:	ђ

unknown_13:	ђ

unknown_14:	ђ

unknown_15:	ђ

unknown_16:	ђ

unknown_17:	ђT

unknown_18:T

unknown_19:T

unknown_20:T

unknown_21:T

unknown_22:T
identityѕбStatefulPartitionedCallю
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22*$
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         T*2
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *M
fHRF
D__inference_basemodel_layer_call_and_return_conditional_losses_247352
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         T2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:         }: : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
+
_output_shapes
:         }
"
_user_specified_name
inputs_0
¤
K
/__inference_dense_1_dropout_layer_call_fn_28811

inputs
identity╠
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *S
fNRL
J__inference_dense_1_dropout_layer_call_and_return_conditional_losses_241062
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         ђ:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
Й
ѓ
%__inference_model_layer_call_fn_25972
left_inputs
right_inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:@ 
	unknown_5:@ђ
	unknown_6:	ђ
	unknown_7:	ђ
	unknown_8:	ђ
	unknown_9:	ђ

unknown_10:	ђ"

unknown_11:ђђ

unknown_12:	ђ

unknown_13:	ђ

unknown_14:	ђ

unknown_15:	ђ

unknown_16:	ђ

unknown_17:	ђT

unknown_18:T

unknown_19:T

unknown_20:T

unknown_21:T

unknown_22:T
identityѕбStatefulPartitionedCallф
StatefulPartitionedCallStatefulPartitionedCallleft_inputsright_inputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22*%
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *2
_read_only_resource_inputs
	*0
config_proto 

CPU

GPU2*0J 8ѓ *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_258672
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*q
_input_shapes`
^:         }:         }: : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
+
_output_shapes
:         }
%
_user_specified_nameleft_inputs:YU
+
_output_shapes
:         }
&
_user_specified_nameright_inputs
▀
K
/__inference_stream_0_drop_3_layer_call_fn_28762

inputs
identityл
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         }ђ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *S
fNRL
J__inference_stream_0_drop_3_layer_call_and_return_conditional_losses_240922
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:         }ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         }ђ:T P
,
_output_shapes
:         }ђ
 
_user_specified_nameinputs
Ы
ц
B__inference_dense_1_layer_call_and_return_conditional_losses_24124

inputs1
matmul_readvariableop_resource:	ђT-
biasadd_readvariableop_resource:T
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpб-dense_1/kernel/Regularizer/Abs/ReadVariableOpј
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	ђT*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         T2
MatMulї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:T*
dtype02
BiasAdd/ReadVariableOpЂ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         T2	
BiasAddЙ
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	ђT*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOpе
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	ђT2 
dense_1/kernel/Regularizer/AbsЋ
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_1/kernel/Regularizer/Constи
dense_1/kernel/Regularizer/SumSum"dense_1/kernel/Regularizer/Abs:y:0)dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/SumЅ
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2"
 dense_1/kernel/Regularizer/mul/x╝
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/mulk
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         T2

Identity»
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp.^dense_1/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ђ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2^
-dense_1/kernel/Regularizer/Abs/ReadVariableOp-dense_1/kernel/Regularizer/Abs/ReadVariableOp:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
И	
m
C__inference_distance_layer_call_and_return_conditional_losses_25364

inputs
inputs_1
identityU
subSubinputsinputs_1*
T0*'
_output_shapes
:         T2
subU
SquareSquaresub:z:0*
T0*'
_output_shapes
:         T2
Squarey
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
         2
Sum/reduction_indicesђ
SumSum
Square:y:0Sum/reduction_indices:output:0*
T0*'
_output_shapes
:         *
	keep_dims(2
SumS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
Constm
MaximumMaximumSum:output:0Const:output:0*
T0*'
_output_shapes
:         2	
MaximumS
SqrtSqrtMaximum:z:0*
T0*'
_output_shapes
:         2
Sqrt\
IdentityIdentitySqrt:y:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:         T:         T:O K
'
_output_shapes
:         T
 
_user_specified_nameinputs:OK
'
_output_shapes
:         T
 
_user_specified_nameinputs
▒
h
/__inference_dense_1_dropout_layer_call_fn_28816

inputs
identityѕбStatefulPartitionedCallС
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *S
fNRL
J__inference_dense_1_dropout_layer_call_and_return_conditional_losses_242582
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         ђ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         ђ22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
▀
K
/__inference_stream_0_drop_2_layer_call_fn_28529

inputs
identityл
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         }ђ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *S
fNRL
J__inference_stream_0_drop_2_layer_call_and_return_conditional_losses_240222
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:         }ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         }ђ:T P
,
_output_shapes
:         }ђ
 
_user_specified_nameinputs
І
l
N__inference_stream_0_input_drop_layer_call_and_return_conditional_losses_23882

inputs

identity_1^
IdentityIdentityinputs*
T0*+
_output_shapes
:         }2

Identitym

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:         }2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         }:S O
+
_output_shapes
:         }
 
_user_specified_nameinputs
ю+
ь
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_28514

inputs6
'assignmovingavg_readvariableop_resource:	ђ8
)assignmovingavg_1_readvariableop_resource:	ђ4
%batchnorm_mul_readvariableop_resource:	ђ0
!batchnorm_readvariableop_resource:	ђ
identityѕбAssignMovingAvgбAssignMovingAvg/ReadVariableOpбAssignMovingAvg_1б AssignMovingAvg_1/ReadVariableOpбbatchnorm/ReadVariableOpбbatchnorm/mul/ReadVariableOpЉ
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indicesћ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*#
_output_shapes
:ђ*
	keep_dims(2
moments/meanЂ
moments/StopGradientStopGradientmoments/mean:output:0*
T0*#
_output_shapes
:ђ2
moments/StopGradientЕ
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*,
_output_shapes
:         }ђ2
moments/SquaredDifferenceЎ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indicesи
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*#
_output_shapes
:ђ*
	keep_dims(2
moments/varianceѓ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:ђ*
squeeze_dims
 2
moments/Squeezeі
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:ђ*
squeeze_dims
 2
moments/Squeeze_1s
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2
AssignMovingAvg/decayЦ
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:ђ*
dtype02 
AssignMovingAvg/ReadVariableOpЎ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:ђ2
AssignMovingAvg/subљ
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:ђ2
AssignMovingAvg/mul┐
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvgw
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2
AssignMovingAvg_1/decayФ
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:ђ*
dtype02"
 AssignMovingAvg_1/ReadVariableOpА
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:ђ2
AssignMovingAvg_1/subў
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:ђ2
AssignMovingAvg_1/mul╔
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1g
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2
batchnorm/add/yЃ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:ђ2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:ђ2
batchnorm/RsqrtЪ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
batchnorm/mul/ReadVariableOpє
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:ђ2
batchnorm/mul{
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*,
_output_shapes
:         }ђ2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:ђ2
batchnorm/mul_2Њ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
batchnorm/ReadVariableOpѓ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:ђ2
batchnorm/subі
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*,
_output_shapes
:         }ђ2
batchnorm/add_1s
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*,
_output_shapes
:         }ђ2

IdentityЫ
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         }ђ: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:T P
,
_output_shapes
:         }ђ
 
_user_specified_nameinputs
В
н
J__inference_stream_0_conv_3_layer_call_and_return_conditional_losses_24045

inputsC
+conv1d_expanddims_1_readvariableop_resource:ђђ.
biasadd_readvariableop_resource:	ђ
identityѕбBiasAdd/ReadVariableOpб"conv1d/ExpandDims_1/ReadVariableOpб5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§        2
conv1d/ExpandDims/dimЌ
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         }ђ2
conv1d/ExpandDims║
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:ђђ*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim╣
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:ђђ2
conv1d/ExpandDims_1и
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:         }ђ*
paddingSAME*
strides
2
conv1dЊ
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:         }ђ*
squeeze_dims

§        2
conv1d/SqueezeЇ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
BiasAdd/ReadVariableOpЇ
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:         }ђ2	
BiasAddЯ
5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:ђђ*
dtype027
5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp┼
&stream_0_conv_3/kernel/Regularizer/AbsAbs=stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*$
_output_shapes
:ђђ2(
&stream_0_conv_3/kernel/Regularizer/AbsЕ
(stream_0_conv_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_0_conv_3/kernel/Regularizer/ConstО
&stream_0_conv_3/kernel/Regularizer/SumSum*stream_0_conv_3/kernel/Regularizer/Abs:y:01stream_0_conv_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_3/kernel/Regularizer/SumЎ
(stream_0_conv_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2*
(stream_0_conv_3/kernel/Regularizer/mul/x▄
&stream_0_conv_3/kernel/Regularizer/mulMul1stream_0_conv_3/kernel/Regularizer/mul/x:output:0/stream_0_conv_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_3/kernel/Regularizer/mulp
IdentityIdentityBiasAdd:output:0^NoOp*
T0*,
_output_shapes
:         }ђ2

Identity─
NoOpNoOp^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp6^stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         }ђ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp2n
5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp:T P
,
_output_shapes
:         }ђ
 
_user_specified_nameinputs
¤
Й
__inference_loss_fn_2_28987V
>stream_0_conv_3_kernel_regularizer_abs_readvariableop_resource:ђђ
identityѕб5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOpз
5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp>stream_0_conv_3_kernel_regularizer_abs_readvariableop_resource*$
_output_shapes
:ђђ*
dtype027
5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp┼
&stream_0_conv_3/kernel/Regularizer/AbsAbs=stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*$
_output_shapes
:ђђ2(
&stream_0_conv_3/kernel/Regularizer/AbsЕ
(stream_0_conv_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_0_conv_3/kernel/Regularizer/ConstО
&stream_0_conv_3/kernel/Regularizer/SumSum*stream_0_conv_3/kernel/Regularizer/Abs:y:01stream_0_conv_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_3/kernel/Regularizer/SumЎ
(stream_0_conv_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2*
(stream_0_conv_3/kernel/Regularizer/mul/x▄
&stream_0_conv_3/kernel/Regularizer/mulMul1stream_0_conv_3/kernel/Regularizer/mul/x:output:0/stream_0_conv_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_3/kernel/Regularizer/mult
IdentityIdentity*stream_0_conv_3/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: 2

Identityє
NoOpNoOp6^stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2n
5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp
┘
H
,__inference_activation_1_layer_call_fn_28519

inputs
identity═
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         }ђ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *P
fKRI
G__inference_activation_1_layer_call_and_return_conditional_losses_240152
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:         }ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         }ђ:T P
,
_output_shapes
:         }ђ
 
_user_specified_nameinputs
с
O
3__inference_stream_0_input_drop_layer_call_fn_28063

inputs
identityМ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         }* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *W
fRRP
N__inference_stream_0_input_drop_layer_call_and_return_conditional_losses_238822
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:         }2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         }:S O
+
_output_shapes
:         }
 
_user_specified_nameinputs
­
m
N__inference_stream_0_input_drop_layer_call_and_return_conditional_losses_24583

inputs
identityѕc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?2
dropout/Constw
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:         }2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/ShapeМ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:         }*
dtype0*
seedи*
seed2и2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>2
dropout/GreaterEqual/y┬
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:         }2
dropout/GreaterEqualЃ
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:         }2
dropout/Cast~
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:         }2
dropout/Mul_1i
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:         }2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         }:S O
+
_output_shapes
:         }
 
_user_specified_nameinputs
В
н
J__inference_stream_0_conv_3_layer_call_and_return_conditional_losses_28587

inputsC
+conv1d_expanddims_1_readvariableop_resource:ђђ.
biasadd_readvariableop_resource:	ђ
identityѕбBiasAdd/ReadVariableOpб"conv1d/ExpandDims_1/ReadVariableOpб5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§        2
conv1d/ExpandDims/dimЌ
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         }ђ2
conv1d/ExpandDims║
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:ђђ*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim╣
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:ђђ2
conv1d/ExpandDims_1и
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:         }ђ*
paddingSAME*
strides
2
conv1dЊ
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:         }ђ*
squeeze_dims

§        2
conv1d/SqueezeЇ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
BiasAdd/ReadVariableOpЇ
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:         }ђ2	
BiasAddЯ
5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:ђђ*
dtype027
5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp┼
&stream_0_conv_3/kernel/Regularizer/AbsAbs=stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*$
_output_shapes
:ђђ2(
&stream_0_conv_3/kernel/Regularizer/AbsЕ
(stream_0_conv_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_0_conv_3/kernel/Regularizer/ConstО
&stream_0_conv_3/kernel/Regularizer/SumSum*stream_0_conv_3/kernel/Regularizer/Abs:y:01stream_0_conv_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_3/kernel/Regularizer/SumЎ
(stream_0_conv_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2*
(stream_0_conv_3/kernel/Regularizer/mul/x▄
&stream_0_conv_3/kernel/Regularizer/mulMul1stream_0_conv_3/kernel/Regularizer/mul/x:output:0/stream_0_conv_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_3/kernel/Regularizer/mulp
IdentityIdentityBiasAdd:output:0^NoOp*
T0*,
_output_shapes
:         }ђ2

Identity─
NoOpNoOp^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp6^stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         }ђ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp2n
5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp:T P
,
_output_shapes
:         }ђ
 
_user_specified_nameinputs
з
c
G__inference_activation_2_layer_call_and_return_conditional_losses_28757

inputs
identityS
ReluReluinputs*
T0*,
_output_shapes
:         }ђ2
Reluk
IdentityIdentityRelu:activations:0*
T0*,
_output_shapes
:         }ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         }ђ:T P
,
_output_shapes
:         }ђ
 
_user_specified_nameinputs
В
i
J__inference_stream_0_drop_1_layer_call_and_return_conditional_losses_28318

inputs
identityѕc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n█Х?2
dropout/Constw
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:         }@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/ShapeМ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:         }@*
dtype0*
seedи*
seed2и2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *џЎЎ>2
dropout/GreaterEqual/y┬
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:         }@2
dropout/GreaterEqualЃ
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:         }@2
dropout/Cast~
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:         }@2
dropout/Mul_1i
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:         }@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         }@:S O
+
_output_shapes
:         }@
 
_user_specified_nameinputs
ы
н
5__inference_batch_normalization_2_layer_call_fn_28626

inputs
unknown:	ђ
	unknown_0:	ђ
	unknown_1:	ђ
	unknown_2:	ђ
identityѕбStatefulPartitionedCallб
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         }ђ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Y
fTRR
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_240702
StatefulPartitionedCallђ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:         }ђ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         }ђ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:         }ђ
 
_user_specified_nameinputs
џ
│
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_24070

inputs0
!batchnorm_readvariableop_resource:	ђ4
%batchnorm_mul_readvariableop_resource:	ђ2
#batchnorm_readvariableop_1_resource:	ђ2
#batchnorm_readvariableop_2_resource:	ђ
identityѕбbatchnorm/ReadVariableOpбbatchnorm/ReadVariableOp_1бbatchnorm/ReadVariableOp_2бbatchnorm/mul/ReadVariableOpЊ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2
batchnorm/add/yЅ
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:ђ2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:ђ2
batchnorm/RsqrtЪ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
batchnorm/mul/ReadVariableOpє
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:ђ2
batchnorm/mul{
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*,
_output_shapes
:         }ђ2
batchnorm/mul_1Ў
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype02
batchnorm/ReadVariableOp_1є
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:ђ2
batchnorm/mul_2Ў
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:ђ*
dtype02
batchnorm/ReadVariableOp_2ё
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:ђ2
batchnorm/subі
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*,
_output_shapes
:         }ђ2
batchnorm/add_1s
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*,
_output_shapes
:         }ђ2

Identity┬
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         }ђ: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:T P
,
_output_shapes
:         }ђ
 
_user_specified_nameinputs
з
c
G__inference_activation_2_layer_call_and_return_conditional_losses_24085

inputs
identityS
ReluReluinputs*
T0*,
_output_shapes
:         }ђ2
Reluk
IdentityIdentityRelu:activations:0*
T0*,
_output_shapes
:         }ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         }ђ:T P
,
_output_shapes
:         }ђ
 
_user_specified_nameinputs
з
c
G__inference_activation_1_layer_call_and_return_conditional_losses_28524

inputs
identityS
ReluReluinputs*
T0*,
_output_shapes
:         }ђ2
Reluk
IdentityIdentityRelu:activations:0*
T0*,
_output_shapes
:         }ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         }ђ:T P
,
_output_shapes
:         }ђ
 
_user_specified_nameinputs
З
i
J__inference_stream_0_drop_3_layer_call_and_return_conditional_losses_28784

inputs
identityѕc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Constx
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:         }ђ2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shapeн
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:         }ђ*
dtype0*
seedи*
seed2и2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/y├
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:         }ђ2
dropout/GreaterEqualё
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:         }ђ2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:         }ђ2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:         }ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         }ђ:T P
,
_output_shapes
:         }ђ
 
_user_specified_nameinputs
к
i
J__inference_dense_1_dropout_layer_call_and_return_conditional_losses_28833

inputs
identityѕc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:         ђ2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape┬
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:         ђ*
dtype0*
seedи2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>2
dropout/GreaterEqual/y┐
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         ђ2
dropout/GreaterEqualђ
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         ђ2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:         ђ2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         ђ:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
М+
ь
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_28693

inputs6
'assignmovingavg_readvariableop_resource:	ђ8
)assignmovingavg_1_readvariableop_resource:	ђ4
%batchnorm_mul_readvariableop_resource:	ђ0
!batchnorm_readvariableop_resource:	ђ
identityѕбAssignMovingAvgбAssignMovingAvg/ReadVariableOpбAssignMovingAvg_1б AssignMovingAvg_1/ReadVariableOpбbatchnorm/ReadVariableOpбbatchnorm/mul/ReadVariableOpЉ
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indicesћ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*#
_output_shapes
:ђ*
	keep_dims(2
moments/meanЂ
moments/StopGradientStopGradientmoments/mean:output:0*
T0*#
_output_shapes
:ђ2
moments/StopGradient▓
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*5
_output_shapes#
!:                  ђ2
moments/SquaredDifferenceЎ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indicesи
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*#
_output_shapes
:ђ*
	keep_dims(2
moments/varianceѓ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:ђ*
squeeze_dims
 2
moments/Squeezeі
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:ђ*
squeeze_dims
 2
moments/Squeeze_1s
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2
AssignMovingAvg/decayЦ
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:ђ*
dtype02 
AssignMovingAvg/ReadVariableOpЎ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:ђ2
AssignMovingAvg/subљ
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:ђ2
AssignMovingAvg/mul┐
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvgw
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2
AssignMovingAvg_1/decayФ
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:ђ*
dtype02"
 AssignMovingAvg_1/ReadVariableOpА
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:ђ2
AssignMovingAvg_1/subў
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:ђ2
AssignMovingAvg_1/mul╔
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1g
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2
batchnorm/add/yЃ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:ђ2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:ђ2
batchnorm/RsqrtЪ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
batchnorm/mul/ReadVariableOpє
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:ђ2
batchnorm/mulё
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*5
_output_shapes#
!:                  ђ2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:ђ2
batchnorm/mul_2Њ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
batchnorm/ReadVariableOpѓ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:ђ2
batchnorm/subЊ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*5
_output_shapes#
!:                  ђ2
batchnorm/add_1|
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*5
_output_shapes#
!:                  ђ2

IdentityЫ
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):                  ђ: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:] Y
5
_output_shapes#
!:                  ђ
 
_user_specified_nameinputs
Є
o
S__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_28806

inputs
identityr
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Mean/reduction_indicesp
MeanMeaninputsMean/reduction_indices:output:0*
T0*(
_output_shapes
:         ђ2
Meanb
IdentityIdentityMean:output:0*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         }ђ:T P
,
_output_shapes
:         }ђ
 
_user_specified_nameinputs
▓
ы
)__inference_basemodel_layer_call_fn_24222
inputs_0
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:@ 
	unknown_5:@ђ
	unknown_6:	ђ
	unknown_7:	ђ
	unknown_8:	ђ
	unknown_9:	ђ

unknown_10:	ђ"

unknown_11:ђђ

unknown_12:	ђ

unknown_13:	ђ

unknown_14:	ђ

unknown_15:	ђ

unknown_16:	ђ

unknown_17:	ђT

unknown_18:T

unknown_19:T

unknown_20:T

unknown_21:T

unknown_22:T
identityѕбStatefulPartitionedCallц
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22*$
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         T*:
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8ѓ *M
fHRF
D__inference_basemodel_layer_call_and_return_conditional_losses_241712
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         T2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:         }: : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
+
_output_shapes
:         }
"
_user_specified_name
inputs_0
Вж
╔)
 __inference__wrapped_model_23198
left_inputs
right_inputsa
Kmodel_basemodel_stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource:@M
?model_basemodel_stream_0_conv_1_biasadd_readvariableop_resource:@S
Emodel_basemodel_batch_normalization_batchnorm_readvariableop_resource:@W
Imodel_basemodel_batch_normalization_batchnorm_mul_readvariableop_resource:@U
Gmodel_basemodel_batch_normalization_batchnorm_readvariableop_1_resource:@U
Gmodel_basemodel_batch_normalization_batchnorm_readvariableop_2_resource:@b
Kmodel_basemodel_stream_0_conv_2_conv1d_expanddims_1_readvariableop_resource:@ђN
?model_basemodel_stream_0_conv_2_biasadd_readvariableop_resource:	ђV
Gmodel_basemodel_batch_normalization_1_batchnorm_readvariableop_resource:	ђZ
Kmodel_basemodel_batch_normalization_1_batchnorm_mul_readvariableop_resource:	ђX
Imodel_basemodel_batch_normalization_1_batchnorm_readvariableop_1_resource:	ђX
Imodel_basemodel_batch_normalization_1_batchnorm_readvariableop_2_resource:	ђc
Kmodel_basemodel_stream_0_conv_3_conv1d_expanddims_1_readvariableop_resource:ђђN
?model_basemodel_stream_0_conv_3_biasadd_readvariableop_resource:	ђV
Gmodel_basemodel_batch_normalization_2_batchnorm_readvariableop_resource:	ђZ
Kmodel_basemodel_batch_normalization_2_batchnorm_mul_readvariableop_resource:	ђX
Imodel_basemodel_batch_normalization_2_batchnorm_readvariableop_1_resource:	ђX
Imodel_basemodel_batch_normalization_2_batchnorm_readvariableop_2_resource:	ђI
6model_basemodel_dense_1_matmul_readvariableop_resource:	ђTE
7model_basemodel_dense_1_biasadd_readvariableop_resource:TU
Gmodel_basemodel_batch_normalization_3_batchnorm_readvariableop_resource:TY
Kmodel_basemodel_batch_normalization_3_batchnorm_mul_readvariableop_resource:TW
Imodel_basemodel_batch_normalization_3_batchnorm_readvariableop_1_resource:TW
Imodel_basemodel_batch_normalization_3_batchnorm_readvariableop_2_resource:T
identityѕб<model/basemodel/batch_normalization/batchnorm/ReadVariableOpб>model/basemodel/batch_normalization/batchnorm/ReadVariableOp_1б>model/basemodel/batch_normalization/batchnorm/ReadVariableOp_2б@model/basemodel/batch_normalization/batchnorm/mul/ReadVariableOpб>model/basemodel/batch_normalization/batchnorm_1/ReadVariableOpб@model/basemodel/batch_normalization/batchnorm_1/ReadVariableOp_1б@model/basemodel/batch_normalization/batchnorm_1/ReadVariableOp_2бBmodel/basemodel/batch_normalization/batchnorm_1/mul/ReadVariableOpб>model/basemodel/batch_normalization_1/batchnorm/ReadVariableOpб@model/basemodel/batch_normalization_1/batchnorm/ReadVariableOp_1б@model/basemodel/batch_normalization_1/batchnorm/ReadVariableOp_2бBmodel/basemodel/batch_normalization_1/batchnorm/mul/ReadVariableOpб@model/basemodel/batch_normalization_1/batchnorm_1/ReadVariableOpбBmodel/basemodel/batch_normalization_1/batchnorm_1/ReadVariableOp_1бBmodel/basemodel/batch_normalization_1/batchnorm_1/ReadVariableOp_2бDmodel/basemodel/batch_normalization_1/batchnorm_1/mul/ReadVariableOpб>model/basemodel/batch_normalization_2/batchnorm/ReadVariableOpб@model/basemodel/batch_normalization_2/batchnorm/ReadVariableOp_1б@model/basemodel/batch_normalization_2/batchnorm/ReadVariableOp_2бBmodel/basemodel/batch_normalization_2/batchnorm/mul/ReadVariableOpб@model/basemodel/batch_normalization_2/batchnorm_1/ReadVariableOpбBmodel/basemodel/batch_normalization_2/batchnorm_1/ReadVariableOp_1бBmodel/basemodel/batch_normalization_2/batchnorm_1/ReadVariableOp_2бDmodel/basemodel/batch_normalization_2/batchnorm_1/mul/ReadVariableOpб>model/basemodel/batch_normalization_3/batchnorm/ReadVariableOpб@model/basemodel/batch_normalization_3/batchnorm/ReadVariableOp_1б@model/basemodel/batch_normalization_3/batchnorm/ReadVariableOp_2бBmodel/basemodel/batch_normalization_3/batchnorm/mul/ReadVariableOpб@model/basemodel/batch_normalization_3/batchnorm_1/ReadVariableOpбBmodel/basemodel/batch_normalization_3/batchnorm_1/ReadVariableOp_1бBmodel/basemodel/batch_normalization_3/batchnorm_1/ReadVariableOp_2бDmodel/basemodel/batch_normalization_3/batchnorm_1/mul/ReadVariableOpб.model/basemodel/dense_1/BiasAdd/ReadVariableOpб0model/basemodel/dense_1/BiasAdd_1/ReadVariableOpб-model/basemodel/dense_1/MatMul/ReadVariableOpб/model/basemodel/dense_1/MatMul_1/ReadVariableOpб6model/basemodel/stream_0_conv_1/BiasAdd/ReadVariableOpб8model/basemodel/stream_0_conv_1/BiasAdd_1/ReadVariableOpбBmodel/basemodel/stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOpбDmodel/basemodel/stream_0_conv_1/conv1d_1/ExpandDims_1/ReadVariableOpб6model/basemodel/stream_0_conv_2/BiasAdd/ReadVariableOpб8model/basemodel/stream_0_conv_2/BiasAdd_1/ReadVariableOpбBmodel/basemodel/stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOpбDmodel/basemodel/stream_0_conv_2/conv1d_1/ExpandDims_1/ReadVariableOpб6model/basemodel/stream_0_conv_3/BiasAdd/ReadVariableOpб8model/basemodel/stream_0_conv_3/BiasAdd_1/ReadVariableOpбBmodel/basemodel/stream_0_conv_3/conv1d/ExpandDims_1/ReadVariableOpбDmodel/basemodel/stream_0_conv_3/conv1d_1/ExpandDims_1/ReadVariableOpФ
,model/basemodel/stream_0_input_drop/IdentityIdentityleft_inputs*
T0*+
_output_shapes
:         }2.
,model/basemodel/stream_0_input_drop/Identity╣
5model/basemodel/stream_0_conv_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§        27
5model/basemodel/stream_0_conv_1/conv1d/ExpandDims/dimЦ
1model/basemodel/stream_0_conv_1/conv1d/ExpandDims
ExpandDims5model/basemodel/stream_0_input_drop/Identity:output:0>model/basemodel/stream_0_conv_1/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:         }23
1model/basemodel/stream_0_conv_1/conv1d/ExpandDimsў
Bmodel/basemodel/stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpKmodel_basemodel_stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02D
Bmodel/basemodel/stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp┤
7model/basemodel/stream_0_conv_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 29
7model/basemodel/stream_0_conv_1/conv1d/ExpandDims_1/dimи
3model/basemodel/stream_0_conv_1/conv1d/ExpandDims_1
ExpandDimsJmodel/basemodel/stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp:value:0@model/basemodel/stream_0_conv_1/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@25
3model/basemodel/stream_0_conv_1/conv1d/ExpandDims_1Х
&model/basemodel/stream_0_conv_1/conv1dConv2D:model/basemodel/stream_0_conv_1/conv1d/ExpandDims:output:0<model/basemodel/stream_0_conv_1/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:         }@*
paddingSAME*
strides
2(
&model/basemodel/stream_0_conv_1/conv1dЫ
.model/basemodel/stream_0_conv_1/conv1d/SqueezeSqueeze/model/basemodel/stream_0_conv_1/conv1d:output:0*
T0*+
_output_shapes
:         }@*
squeeze_dims

§        20
.model/basemodel/stream_0_conv_1/conv1d/SqueezeВ
6model/basemodel/stream_0_conv_1/BiasAdd/ReadVariableOpReadVariableOp?model_basemodel_stream_0_conv_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype028
6model/basemodel/stream_0_conv_1/BiasAdd/ReadVariableOpї
'model/basemodel/stream_0_conv_1/BiasAddBiasAdd7model/basemodel/stream_0_conv_1/conv1d/Squeeze:output:0>model/basemodel/stream_0_conv_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         }@2)
'model/basemodel/stream_0_conv_1/BiasAdd■
<model/basemodel/batch_normalization/batchnorm/ReadVariableOpReadVariableOpEmodel_basemodel_batch_normalization_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02>
<model/basemodel/batch_normalization/batchnorm/ReadVariableOp»
3model/basemodel/batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:25
3model/basemodel/batch_normalization/batchnorm/add/yў
1model/basemodel/batch_normalization/batchnorm/addAddV2Dmodel/basemodel/batch_normalization/batchnorm/ReadVariableOp:value:0<model/basemodel/batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes
:@23
1model/basemodel/batch_normalization/batchnorm/add¤
3model/basemodel/batch_normalization/batchnorm/RsqrtRsqrt5model/basemodel/batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes
:@25
3model/basemodel/batch_normalization/batchnorm/Rsqrtі
@model/basemodel/batch_normalization/batchnorm/mul/ReadVariableOpReadVariableOpImodel_basemodel_batch_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02B
@model/basemodel/batch_normalization/batchnorm/mul/ReadVariableOpЋ
1model/basemodel/batch_normalization/batchnorm/mulMul7model/basemodel/batch_normalization/batchnorm/Rsqrt:y:0Hmodel/basemodel/batch_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@23
1model/basemodel/batch_normalization/batchnorm/mulљ
3model/basemodel/batch_normalization/batchnorm/mul_1Mul0model/basemodel/stream_0_conv_1/BiasAdd:output:05model/basemodel/batch_normalization/batchnorm/mul:z:0*
T0*+
_output_shapes
:         }@25
3model/basemodel/batch_normalization/batchnorm/mul_1ё
>model/basemodel/batch_normalization/batchnorm/ReadVariableOp_1ReadVariableOpGmodel_basemodel_batch_normalization_batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype02@
>model/basemodel/batch_normalization/batchnorm/ReadVariableOp_1Ћ
3model/basemodel/batch_normalization/batchnorm/mul_2MulFmodel/basemodel/batch_normalization/batchnorm/ReadVariableOp_1:value:05model/basemodel/batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes
:@25
3model/basemodel/batch_normalization/batchnorm/mul_2ё
>model/basemodel/batch_normalization/batchnorm/ReadVariableOp_2ReadVariableOpGmodel_basemodel_batch_normalization_batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype02@
>model/basemodel/batch_normalization/batchnorm/ReadVariableOp_2Њ
1model/basemodel/batch_normalization/batchnorm/subSubFmodel/basemodel/batch_normalization/batchnorm/ReadVariableOp_2:value:07model/basemodel/batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@23
1model/basemodel/batch_normalization/batchnorm/subЎ
3model/basemodel/batch_normalization/batchnorm/add_1AddV27model/basemodel/batch_normalization/batchnorm/mul_1:z:05model/basemodel/batch_normalization/batchnorm/sub:z:0*
T0*+
_output_shapes
:         }@25
3model/basemodel/batch_normalization/batchnorm/add_1╣
model/basemodel/activation/ReluRelu7model/basemodel/batch_normalization/batchnorm/add_1:z:0*
T0*+
_output_shapes
:         }@2!
model/basemodel/activation/Relu┼
(model/basemodel/stream_0_drop_1/IdentityIdentity-model/basemodel/activation/Relu:activations:0*
T0*+
_output_shapes
:         }@2*
(model/basemodel/stream_0_drop_1/Identity╣
5model/basemodel/stream_0_conv_2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§        27
5model/basemodel/stream_0_conv_2/conv1d/ExpandDims/dimА
1model/basemodel/stream_0_conv_2/conv1d/ExpandDims
ExpandDims1model/basemodel/stream_0_drop_1/Identity:output:0>model/basemodel/stream_0_conv_2/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:         }@23
1model/basemodel/stream_0_conv_2/conv1d/ExpandDimsЎ
Bmodel/basemodel/stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpKmodel_basemodel_stream_0_conv_2_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:@ђ*
dtype02D
Bmodel/basemodel/stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOp┤
7model/basemodel/stream_0_conv_2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 29
7model/basemodel/stream_0_conv_2/conv1d/ExpandDims_1/dimИ
3model/basemodel/stream_0_conv_2/conv1d/ExpandDims_1
ExpandDimsJmodel/basemodel/stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOp:value:0@model/basemodel/stream_0_conv_2/conv1d/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:@ђ25
3model/basemodel/stream_0_conv_2/conv1d/ExpandDims_1и
&model/basemodel/stream_0_conv_2/conv1dConv2D:model/basemodel/stream_0_conv_2/conv1d/ExpandDims:output:0<model/basemodel/stream_0_conv_2/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:         }ђ*
paddingSAME*
strides
2(
&model/basemodel/stream_0_conv_2/conv1dз
.model/basemodel/stream_0_conv_2/conv1d/SqueezeSqueeze/model/basemodel/stream_0_conv_2/conv1d:output:0*
T0*,
_output_shapes
:         }ђ*
squeeze_dims

§        20
.model/basemodel/stream_0_conv_2/conv1d/Squeezeь
6model/basemodel/stream_0_conv_2/BiasAdd/ReadVariableOpReadVariableOp?model_basemodel_stream_0_conv_2_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype028
6model/basemodel/stream_0_conv_2/BiasAdd/ReadVariableOpЇ
'model/basemodel/stream_0_conv_2/BiasAddBiasAdd7model/basemodel/stream_0_conv_2/conv1d/Squeeze:output:0>model/basemodel/stream_0_conv_2/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:         }ђ2)
'model/basemodel/stream_0_conv_2/BiasAddЁ
>model/basemodel/batch_normalization_1/batchnorm/ReadVariableOpReadVariableOpGmodel_basemodel_batch_normalization_1_batchnorm_readvariableop_resource*
_output_shapes	
:ђ*
dtype02@
>model/basemodel/batch_normalization_1/batchnorm/ReadVariableOp│
5model/basemodel/batch_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:27
5model/basemodel/batch_normalization_1/batchnorm/add/yА
3model/basemodel/batch_normalization_1/batchnorm/addAddV2Fmodel/basemodel/batch_normalization_1/batchnorm/ReadVariableOp:value:0>model/basemodel/batch_normalization_1/batchnorm/add/y:output:0*
T0*
_output_shapes	
:ђ25
3model/basemodel/batch_normalization_1/batchnorm/addо
5model/basemodel/batch_normalization_1/batchnorm/RsqrtRsqrt7model/basemodel/batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes	
:ђ27
5model/basemodel/batch_normalization_1/batchnorm/RsqrtЉ
Bmodel/basemodel/batch_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOpKmodel_basemodel_batch_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes	
:ђ*
dtype02D
Bmodel/basemodel/batch_normalization_1/batchnorm/mul/ReadVariableOpъ
3model/basemodel/batch_normalization_1/batchnorm/mulMul9model/basemodel/batch_normalization_1/batchnorm/Rsqrt:y:0Jmodel/basemodel/batch_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:ђ25
3model/basemodel/batch_normalization_1/batchnorm/mulЌ
5model/basemodel/batch_normalization_1/batchnorm/mul_1Mul0model/basemodel/stream_0_conv_2/BiasAdd:output:07model/basemodel/batch_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:         }ђ27
5model/basemodel/batch_normalization_1/batchnorm/mul_1І
@model/basemodel/batch_normalization_1/batchnorm/ReadVariableOp_1ReadVariableOpImodel_basemodel_batch_normalization_1_batchnorm_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype02B
@model/basemodel/batch_normalization_1/batchnorm/ReadVariableOp_1ъ
5model/basemodel/batch_normalization_1/batchnorm/mul_2MulHmodel/basemodel/batch_normalization_1/batchnorm/ReadVariableOp_1:value:07model/basemodel/batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes	
:ђ27
5model/basemodel/batch_normalization_1/batchnorm/mul_2І
@model/basemodel/batch_normalization_1/batchnorm/ReadVariableOp_2ReadVariableOpImodel_basemodel_batch_normalization_1_batchnorm_readvariableop_2_resource*
_output_shapes	
:ђ*
dtype02B
@model/basemodel/batch_normalization_1/batchnorm/ReadVariableOp_2ю
3model/basemodel/batch_normalization_1/batchnorm/subSubHmodel/basemodel/batch_normalization_1/batchnorm/ReadVariableOp_2:value:09model/basemodel/batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:ђ25
3model/basemodel/batch_normalization_1/batchnorm/subб
5model/basemodel/batch_normalization_1/batchnorm/add_1AddV29model/basemodel/batch_normalization_1/batchnorm/mul_1:z:07model/basemodel/batch_normalization_1/batchnorm/sub:z:0*
T0*,
_output_shapes
:         }ђ27
5model/basemodel/batch_normalization_1/batchnorm/add_1└
!model/basemodel/activation_1/ReluRelu9model/basemodel/batch_normalization_1/batchnorm/add_1:z:0*
T0*,
_output_shapes
:         }ђ2#
!model/basemodel/activation_1/Relu╚
(model/basemodel/stream_0_drop_2/IdentityIdentity/model/basemodel/activation_1/Relu:activations:0*
T0*,
_output_shapes
:         }ђ2*
(model/basemodel/stream_0_drop_2/Identity╣
5model/basemodel/stream_0_conv_3/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§        27
5model/basemodel/stream_0_conv_3/conv1d/ExpandDims/dimб
1model/basemodel/stream_0_conv_3/conv1d/ExpandDims
ExpandDims1model/basemodel/stream_0_drop_2/Identity:output:0>model/basemodel/stream_0_conv_3/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         }ђ23
1model/basemodel/stream_0_conv_3/conv1d/ExpandDimsџ
Bmodel/basemodel/stream_0_conv_3/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpKmodel_basemodel_stream_0_conv_3_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:ђђ*
dtype02D
Bmodel/basemodel/stream_0_conv_3/conv1d/ExpandDims_1/ReadVariableOp┤
7model/basemodel/stream_0_conv_3/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 29
7model/basemodel/stream_0_conv_3/conv1d/ExpandDims_1/dim╣
3model/basemodel/stream_0_conv_3/conv1d/ExpandDims_1
ExpandDimsJmodel/basemodel/stream_0_conv_3/conv1d/ExpandDims_1/ReadVariableOp:value:0@model/basemodel/stream_0_conv_3/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:ђђ25
3model/basemodel/stream_0_conv_3/conv1d/ExpandDims_1и
&model/basemodel/stream_0_conv_3/conv1dConv2D:model/basemodel/stream_0_conv_3/conv1d/ExpandDims:output:0<model/basemodel/stream_0_conv_3/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:         }ђ*
paddingSAME*
strides
2(
&model/basemodel/stream_0_conv_3/conv1dз
.model/basemodel/stream_0_conv_3/conv1d/SqueezeSqueeze/model/basemodel/stream_0_conv_3/conv1d:output:0*
T0*,
_output_shapes
:         }ђ*
squeeze_dims

§        20
.model/basemodel/stream_0_conv_3/conv1d/Squeezeь
6model/basemodel/stream_0_conv_3/BiasAdd/ReadVariableOpReadVariableOp?model_basemodel_stream_0_conv_3_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype028
6model/basemodel/stream_0_conv_3/BiasAdd/ReadVariableOpЇ
'model/basemodel/stream_0_conv_3/BiasAddBiasAdd7model/basemodel/stream_0_conv_3/conv1d/Squeeze:output:0>model/basemodel/stream_0_conv_3/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:         }ђ2)
'model/basemodel/stream_0_conv_3/BiasAddЁ
>model/basemodel/batch_normalization_2/batchnorm/ReadVariableOpReadVariableOpGmodel_basemodel_batch_normalization_2_batchnorm_readvariableop_resource*
_output_shapes	
:ђ*
dtype02@
>model/basemodel/batch_normalization_2/batchnorm/ReadVariableOp│
5model/basemodel/batch_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:27
5model/basemodel/batch_normalization_2/batchnorm/add/yА
3model/basemodel/batch_normalization_2/batchnorm/addAddV2Fmodel/basemodel/batch_normalization_2/batchnorm/ReadVariableOp:value:0>model/basemodel/batch_normalization_2/batchnorm/add/y:output:0*
T0*
_output_shapes	
:ђ25
3model/basemodel/batch_normalization_2/batchnorm/addо
5model/basemodel/batch_normalization_2/batchnorm/RsqrtRsqrt7model/basemodel/batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes	
:ђ27
5model/basemodel/batch_normalization_2/batchnorm/RsqrtЉ
Bmodel/basemodel/batch_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOpKmodel_basemodel_batch_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes	
:ђ*
dtype02D
Bmodel/basemodel/batch_normalization_2/batchnorm/mul/ReadVariableOpъ
3model/basemodel/batch_normalization_2/batchnorm/mulMul9model/basemodel/batch_normalization_2/batchnorm/Rsqrt:y:0Jmodel/basemodel/batch_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:ђ25
3model/basemodel/batch_normalization_2/batchnorm/mulЌ
5model/basemodel/batch_normalization_2/batchnorm/mul_1Mul0model/basemodel/stream_0_conv_3/BiasAdd:output:07model/basemodel/batch_normalization_2/batchnorm/mul:z:0*
T0*,
_output_shapes
:         }ђ27
5model/basemodel/batch_normalization_2/batchnorm/mul_1І
@model/basemodel/batch_normalization_2/batchnorm/ReadVariableOp_1ReadVariableOpImodel_basemodel_batch_normalization_2_batchnorm_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype02B
@model/basemodel/batch_normalization_2/batchnorm/ReadVariableOp_1ъ
5model/basemodel/batch_normalization_2/batchnorm/mul_2MulHmodel/basemodel/batch_normalization_2/batchnorm/ReadVariableOp_1:value:07model/basemodel/batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes	
:ђ27
5model/basemodel/batch_normalization_2/batchnorm/mul_2І
@model/basemodel/batch_normalization_2/batchnorm/ReadVariableOp_2ReadVariableOpImodel_basemodel_batch_normalization_2_batchnorm_readvariableop_2_resource*
_output_shapes	
:ђ*
dtype02B
@model/basemodel/batch_normalization_2/batchnorm/ReadVariableOp_2ю
3model/basemodel/batch_normalization_2/batchnorm/subSubHmodel/basemodel/batch_normalization_2/batchnorm/ReadVariableOp_2:value:09model/basemodel/batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:ђ25
3model/basemodel/batch_normalization_2/batchnorm/subб
5model/basemodel/batch_normalization_2/batchnorm/add_1AddV29model/basemodel/batch_normalization_2/batchnorm/mul_1:z:07model/basemodel/batch_normalization_2/batchnorm/sub:z:0*
T0*,
_output_shapes
:         }ђ27
5model/basemodel/batch_normalization_2/batchnorm/add_1└
!model/basemodel/activation_2/ReluRelu9model/basemodel/batch_normalization_2/batchnorm/add_1:z:0*
T0*,
_output_shapes
:         }ђ2#
!model/basemodel/activation_2/Relu╚
(model/basemodel/stream_0_drop_3/IdentityIdentity/model/basemodel/activation_2/Relu:activations:0*
T0*,
_output_shapes
:         }ђ2*
(model/basemodel/stream_0_drop_3/Identity─
?model/basemodel/global_average_pooling1d/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2A
?model/basemodel/global_average_pooling1d/Mean/reduction_indicesќ
-model/basemodel/global_average_pooling1d/MeanMean1model/basemodel/stream_0_drop_3/Identity:output:0Hmodel/basemodel/global_average_pooling1d/Mean/reduction_indices:output:0*
T0*(
_output_shapes
:         ђ2/
-model/basemodel/global_average_pooling1d/Mean╦
(model/basemodel/dense_1_dropout/IdentityIdentity6model/basemodel/global_average_pooling1d/Mean:output:0*
T0*(
_output_shapes
:         ђ2*
(model/basemodel/dense_1_dropout/Identityо
-model/basemodel/dense_1/MatMul/ReadVariableOpReadVariableOp6model_basemodel_dense_1_matmul_readvariableop_resource*
_output_shapes
:	ђT*
dtype02/
-model/basemodel/dense_1/MatMul/ReadVariableOpТ
model/basemodel/dense_1/MatMulMatMul1model/basemodel/dense_1_dropout/Identity:output:05model/basemodel/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         T2 
model/basemodel/dense_1/MatMulн
.model/basemodel/dense_1/BiasAdd/ReadVariableOpReadVariableOp7model_basemodel_dense_1_biasadd_readvariableop_resource*
_output_shapes
:T*
dtype020
.model/basemodel/dense_1/BiasAdd/ReadVariableOpр
model/basemodel/dense_1/BiasAddBiasAdd(model/basemodel/dense_1/MatMul:product:06model/basemodel/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         T2!
model/basemodel/dense_1/BiasAddё
>model/basemodel/batch_normalization_3/batchnorm/ReadVariableOpReadVariableOpGmodel_basemodel_batch_normalization_3_batchnorm_readvariableop_resource*
_output_shapes
:T*
dtype02@
>model/basemodel/batch_normalization_3/batchnorm/ReadVariableOp│
5model/basemodel/batch_normalization_3/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:27
5model/basemodel/batch_normalization_3/batchnorm/add/yа
3model/basemodel/batch_normalization_3/batchnorm/addAddV2Fmodel/basemodel/batch_normalization_3/batchnorm/ReadVariableOp:value:0>model/basemodel/batch_normalization_3/batchnorm/add/y:output:0*
T0*
_output_shapes
:T25
3model/basemodel/batch_normalization_3/batchnorm/addН
5model/basemodel/batch_normalization_3/batchnorm/RsqrtRsqrt7model/basemodel/batch_normalization_3/batchnorm/add:z:0*
T0*
_output_shapes
:T27
5model/basemodel/batch_normalization_3/batchnorm/Rsqrtљ
Bmodel/basemodel/batch_normalization_3/batchnorm/mul/ReadVariableOpReadVariableOpKmodel_basemodel_batch_normalization_3_batchnorm_mul_readvariableop_resource*
_output_shapes
:T*
dtype02D
Bmodel/basemodel/batch_normalization_3/batchnorm/mul/ReadVariableOpЮ
3model/basemodel/batch_normalization_3/batchnorm/mulMul9model/basemodel/batch_normalization_3/batchnorm/Rsqrt:y:0Jmodel/basemodel/batch_normalization_3/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:T25
3model/basemodel/batch_normalization_3/batchnorm/mulі
5model/basemodel/batch_normalization_3/batchnorm/mul_1Mul(model/basemodel/dense_1/BiasAdd:output:07model/basemodel/batch_normalization_3/batchnorm/mul:z:0*
T0*'
_output_shapes
:         T27
5model/basemodel/batch_normalization_3/batchnorm/mul_1і
@model/basemodel/batch_normalization_3/batchnorm/ReadVariableOp_1ReadVariableOpImodel_basemodel_batch_normalization_3_batchnorm_readvariableop_1_resource*
_output_shapes
:T*
dtype02B
@model/basemodel/batch_normalization_3/batchnorm/ReadVariableOp_1Ю
5model/basemodel/batch_normalization_3/batchnorm/mul_2MulHmodel/basemodel/batch_normalization_3/batchnorm/ReadVariableOp_1:value:07model/basemodel/batch_normalization_3/batchnorm/mul:z:0*
T0*
_output_shapes
:T27
5model/basemodel/batch_normalization_3/batchnorm/mul_2і
@model/basemodel/batch_normalization_3/batchnorm/ReadVariableOp_2ReadVariableOpImodel_basemodel_batch_normalization_3_batchnorm_readvariableop_2_resource*
_output_shapes
:T*
dtype02B
@model/basemodel/batch_normalization_3/batchnorm/ReadVariableOp_2Џ
3model/basemodel/batch_normalization_3/batchnorm/subSubHmodel/basemodel/batch_normalization_3/batchnorm/ReadVariableOp_2:value:09model/basemodel/batch_normalization_3/batchnorm/mul_2:z:0*
T0*
_output_shapes
:T25
3model/basemodel/batch_normalization_3/batchnorm/subЮ
5model/basemodel/batch_normalization_3/batchnorm/add_1AddV29model/basemodel/batch_normalization_3/batchnorm/mul_1:z:07model/basemodel/batch_normalization_3/batchnorm/sub:z:0*
T0*'
_output_shapes
:         T27
5model/basemodel/batch_normalization_3/batchnorm/add_1л
*model/basemodel/dense_activation_1/SigmoidSigmoid9model/basemodel/batch_normalization_3/batchnorm/add_1:z:0*
T0*'
_output_shapes
:         T2,
*model/basemodel/dense_activation_1/Sigmoid░
.model/basemodel/stream_0_input_drop/Identity_1Identityright_inputs*
T0*+
_output_shapes
:         }20
.model/basemodel/stream_0_input_drop/Identity_1й
7model/basemodel/stream_0_conv_1/conv1d_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§        29
7model/basemodel/stream_0_conv_1/conv1d_1/ExpandDims/dimГ
3model/basemodel/stream_0_conv_1/conv1d_1/ExpandDims
ExpandDims7model/basemodel/stream_0_input_drop/Identity_1:output:0@model/basemodel/stream_0_conv_1/conv1d_1/ExpandDims/dim:output:0*
T0*/
_output_shapes
:         }25
3model/basemodel/stream_0_conv_1/conv1d_1/ExpandDimsю
Dmodel/basemodel/stream_0_conv_1/conv1d_1/ExpandDims_1/ReadVariableOpReadVariableOpKmodel_basemodel_stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02F
Dmodel/basemodel/stream_0_conv_1/conv1d_1/ExpandDims_1/ReadVariableOpИ
9model/basemodel/stream_0_conv_1/conv1d_1/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2;
9model/basemodel/stream_0_conv_1/conv1d_1/ExpandDims_1/dim┐
5model/basemodel/stream_0_conv_1/conv1d_1/ExpandDims_1
ExpandDimsLmodel/basemodel/stream_0_conv_1/conv1d_1/ExpandDims_1/ReadVariableOp:value:0Bmodel/basemodel/stream_0_conv_1/conv1d_1/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@27
5model/basemodel/stream_0_conv_1/conv1d_1/ExpandDims_1Й
(model/basemodel/stream_0_conv_1/conv1d_1Conv2D<model/basemodel/stream_0_conv_1/conv1d_1/ExpandDims:output:0>model/basemodel/stream_0_conv_1/conv1d_1/ExpandDims_1:output:0*
T0*/
_output_shapes
:         }@*
paddingSAME*
strides
2*
(model/basemodel/stream_0_conv_1/conv1d_1Э
0model/basemodel/stream_0_conv_1/conv1d_1/SqueezeSqueeze1model/basemodel/stream_0_conv_1/conv1d_1:output:0*
T0*+
_output_shapes
:         }@*
squeeze_dims

§        22
0model/basemodel/stream_0_conv_1/conv1d_1/Squeeze­
8model/basemodel/stream_0_conv_1/BiasAdd_1/ReadVariableOpReadVariableOp?model_basemodel_stream_0_conv_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02:
8model/basemodel/stream_0_conv_1/BiasAdd_1/ReadVariableOpћ
)model/basemodel/stream_0_conv_1/BiasAdd_1BiasAdd9model/basemodel/stream_0_conv_1/conv1d_1/Squeeze:output:0@model/basemodel/stream_0_conv_1/BiasAdd_1/ReadVariableOp:value:0*
T0*+
_output_shapes
:         }@2+
)model/basemodel/stream_0_conv_1/BiasAdd_1ѓ
>model/basemodel/batch_normalization/batchnorm_1/ReadVariableOpReadVariableOpEmodel_basemodel_batch_normalization_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02@
>model/basemodel/batch_normalization/batchnorm_1/ReadVariableOp│
5model/basemodel/batch_normalization/batchnorm_1/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:27
5model/basemodel/batch_normalization/batchnorm_1/add/yа
3model/basemodel/batch_normalization/batchnorm_1/addAddV2Fmodel/basemodel/batch_normalization/batchnorm_1/ReadVariableOp:value:0>model/basemodel/batch_normalization/batchnorm_1/add/y:output:0*
T0*
_output_shapes
:@25
3model/basemodel/batch_normalization/batchnorm_1/addН
5model/basemodel/batch_normalization/batchnorm_1/RsqrtRsqrt7model/basemodel/batch_normalization/batchnorm_1/add:z:0*
T0*
_output_shapes
:@27
5model/basemodel/batch_normalization/batchnorm_1/Rsqrtј
Bmodel/basemodel/batch_normalization/batchnorm_1/mul/ReadVariableOpReadVariableOpImodel_basemodel_batch_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02D
Bmodel/basemodel/batch_normalization/batchnorm_1/mul/ReadVariableOpЮ
3model/basemodel/batch_normalization/batchnorm_1/mulMul9model/basemodel/batch_normalization/batchnorm_1/Rsqrt:y:0Jmodel/basemodel/batch_normalization/batchnorm_1/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@25
3model/basemodel/batch_normalization/batchnorm_1/mulў
5model/basemodel/batch_normalization/batchnorm_1/mul_1Mul2model/basemodel/stream_0_conv_1/BiasAdd_1:output:07model/basemodel/batch_normalization/batchnorm_1/mul:z:0*
T0*+
_output_shapes
:         }@27
5model/basemodel/batch_normalization/batchnorm_1/mul_1ѕ
@model/basemodel/batch_normalization/batchnorm_1/ReadVariableOp_1ReadVariableOpGmodel_basemodel_batch_normalization_batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype02B
@model/basemodel/batch_normalization/batchnorm_1/ReadVariableOp_1Ю
5model/basemodel/batch_normalization/batchnorm_1/mul_2MulHmodel/basemodel/batch_normalization/batchnorm_1/ReadVariableOp_1:value:07model/basemodel/batch_normalization/batchnorm_1/mul:z:0*
T0*
_output_shapes
:@27
5model/basemodel/batch_normalization/batchnorm_1/mul_2ѕ
@model/basemodel/batch_normalization/batchnorm_1/ReadVariableOp_2ReadVariableOpGmodel_basemodel_batch_normalization_batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype02B
@model/basemodel/batch_normalization/batchnorm_1/ReadVariableOp_2Џ
3model/basemodel/batch_normalization/batchnorm_1/subSubHmodel/basemodel/batch_normalization/batchnorm_1/ReadVariableOp_2:value:09model/basemodel/batch_normalization/batchnorm_1/mul_2:z:0*
T0*
_output_shapes
:@25
3model/basemodel/batch_normalization/batchnorm_1/subА
5model/basemodel/batch_normalization/batchnorm_1/add_1AddV29model/basemodel/batch_normalization/batchnorm_1/mul_1:z:07model/basemodel/batch_normalization/batchnorm_1/sub:z:0*
T0*+
_output_shapes
:         }@27
5model/basemodel/batch_normalization/batchnorm_1/add_1┐
!model/basemodel/activation/Relu_1Relu9model/basemodel/batch_normalization/batchnorm_1/add_1:z:0*
T0*+
_output_shapes
:         }@2#
!model/basemodel/activation/Relu_1╦
*model/basemodel/stream_0_drop_1/Identity_1Identity/model/basemodel/activation/Relu_1:activations:0*
T0*+
_output_shapes
:         }@2,
*model/basemodel/stream_0_drop_1/Identity_1й
7model/basemodel/stream_0_conv_2/conv1d_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§        29
7model/basemodel/stream_0_conv_2/conv1d_1/ExpandDims/dimЕ
3model/basemodel/stream_0_conv_2/conv1d_1/ExpandDims
ExpandDims3model/basemodel/stream_0_drop_1/Identity_1:output:0@model/basemodel/stream_0_conv_2/conv1d_1/ExpandDims/dim:output:0*
T0*/
_output_shapes
:         }@25
3model/basemodel/stream_0_conv_2/conv1d_1/ExpandDimsЮ
Dmodel/basemodel/stream_0_conv_2/conv1d_1/ExpandDims_1/ReadVariableOpReadVariableOpKmodel_basemodel_stream_0_conv_2_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:@ђ*
dtype02F
Dmodel/basemodel/stream_0_conv_2/conv1d_1/ExpandDims_1/ReadVariableOpИ
9model/basemodel/stream_0_conv_2/conv1d_1/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2;
9model/basemodel/stream_0_conv_2/conv1d_1/ExpandDims_1/dim└
5model/basemodel/stream_0_conv_2/conv1d_1/ExpandDims_1
ExpandDimsLmodel/basemodel/stream_0_conv_2/conv1d_1/ExpandDims_1/ReadVariableOp:value:0Bmodel/basemodel/stream_0_conv_2/conv1d_1/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:@ђ27
5model/basemodel/stream_0_conv_2/conv1d_1/ExpandDims_1┐
(model/basemodel/stream_0_conv_2/conv1d_1Conv2D<model/basemodel/stream_0_conv_2/conv1d_1/ExpandDims:output:0>model/basemodel/stream_0_conv_2/conv1d_1/ExpandDims_1:output:0*
T0*0
_output_shapes
:         }ђ*
paddingSAME*
strides
2*
(model/basemodel/stream_0_conv_2/conv1d_1щ
0model/basemodel/stream_0_conv_2/conv1d_1/SqueezeSqueeze1model/basemodel/stream_0_conv_2/conv1d_1:output:0*
T0*,
_output_shapes
:         }ђ*
squeeze_dims

§        22
0model/basemodel/stream_0_conv_2/conv1d_1/Squeezeы
8model/basemodel/stream_0_conv_2/BiasAdd_1/ReadVariableOpReadVariableOp?model_basemodel_stream_0_conv_2_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02:
8model/basemodel/stream_0_conv_2/BiasAdd_1/ReadVariableOpЋ
)model/basemodel/stream_0_conv_2/BiasAdd_1BiasAdd9model/basemodel/stream_0_conv_2/conv1d_1/Squeeze:output:0@model/basemodel/stream_0_conv_2/BiasAdd_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:         }ђ2+
)model/basemodel/stream_0_conv_2/BiasAdd_1Ѕ
@model/basemodel/batch_normalization_1/batchnorm_1/ReadVariableOpReadVariableOpGmodel_basemodel_batch_normalization_1_batchnorm_readvariableop_resource*
_output_shapes	
:ђ*
dtype02B
@model/basemodel/batch_normalization_1/batchnorm_1/ReadVariableOpи
7model/basemodel/batch_normalization_1/batchnorm_1/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:29
7model/basemodel/batch_normalization_1/batchnorm_1/add/yЕ
5model/basemodel/batch_normalization_1/batchnorm_1/addAddV2Hmodel/basemodel/batch_normalization_1/batchnorm_1/ReadVariableOp:value:0@model/basemodel/batch_normalization_1/batchnorm_1/add/y:output:0*
T0*
_output_shapes	
:ђ27
5model/basemodel/batch_normalization_1/batchnorm_1/add▄
7model/basemodel/batch_normalization_1/batchnorm_1/RsqrtRsqrt9model/basemodel/batch_normalization_1/batchnorm_1/add:z:0*
T0*
_output_shapes	
:ђ29
7model/basemodel/batch_normalization_1/batchnorm_1/RsqrtЋ
Dmodel/basemodel/batch_normalization_1/batchnorm_1/mul/ReadVariableOpReadVariableOpKmodel_basemodel_batch_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes	
:ђ*
dtype02F
Dmodel/basemodel/batch_normalization_1/batchnorm_1/mul/ReadVariableOpд
5model/basemodel/batch_normalization_1/batchnorm_1/mulMul;model/basemodel/batch_normalization_1/batchnorm_1/Rsqrt:y:0Lmodel/basemodel/batch_normalization_1/batchnorm_1/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:ђ27
5model/basemodel/batch_normalization_1/batchnorm_1/mulЪ
7model/basemodel/batch_normalization_1/batchnorm_1/mul_1Mul2model/basemodel/stream_0_conv_2/BiasAdd_1:output:09model/basemodel/batch_normalization_1/batchnorm_1/mul:z:0*
T0*,
_output_shapes
:         }ђ29
7model/basemodel/batch_normalization_1/batchnorm_1/mul_1Ј
Bmodel/basemodel/batch_normalization_1/batchnorm_1/ReadVariableOp_1ReadVariableOpImodel_basemodel_batch_normalization_1_batchnorm_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype02D
Bmodel/basemodel/batch_normalization_1/batchnorm_1/ReadVariableOp_1д
7model/basemodel/batch_normalization_1/batchnorm_1/mul_2MulJmodel/basemodel/batch_normalization_1/batchnorm_1/ReadVariableOp_1:value:09model/basemodel/batch_normalization_1/batchnorm_1/mul:z:0*
T0*
_output_shapes	
:ђ29
7model/basemodel/batch_normalization_1/batchnorm_1/mul_2Ј
Bmodel/basemodel/batch_normalization_1/batchnorm_1/ReadVariableOp_2ReadVariableOpImodel_basemodel_batch_normalization_1_batchnorm_readvariableop_2_resource*
_output_shapes	
:ђ*
dtype02D
Bmodel/basemodel/batch_normalization_1/batchnorm_1/ReadVariableOp_2ц
5model/basemodel/batch_normalization_1/batchnorm_1/subSubJmodel/basemodel/batch_normalization_1/batchnorm_1/ReadVariableOp_2:value:0;model/basemodel/batch_normalization_1/batchnorm_1/mul_2:z:0*
T0*
_output_shapes	
:ђ27
5model/basemodel/batch_normalization_1/batchnorm_1/subф
7model/basemodel/batch_normalization_1/batchnorm_1/add_1AddV2;model/basemodel/batch_normalization_1/batchnorm_1/mul_1:z:09model/basemodel/batch_normalization_1/batchnorm_1/sub:z:0*
T0*,
_output_shapes
:         }ђ29
7model/basemodel/batch_normalization_1/batchnorm_1/add_1к
#model/basemodel/activation_1/Relu_1Relu;model/basemodel/batch_normalization_1/batchnorm_1/add_1:z:0*
T0*,
_output_shapes
:         }ђ2%
#model/basemodel/activation_1/Relu_1╬
*model/basemodel/stream_0_drop_2/Identity_1Identity1model/basemodel/activation_1/Relu_1:activations:0*
T0*,
_output_shapes
:         }ђ2,
*model/basemodel/stream_0_drop_2/Identity_1й
7model/basemodel/stream_0_conv_3/conv1d_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§        29
7model/basemodel/stream_0_conv_3/conv1d_1/ExpandDims/dimф
3model/basemodel/stream_0_conv_3/conv1d_1/ExpandDims
ExpandDims3model/basemodel/stream_0_drop_2/Identity_1:output:0@model/basemodel/stream_0_conv_3/conv1d_1/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         }ђ25
3model/basemodel/stream_0_conv_3/conv1d_1/ExpandDimsъ
Dmodel/basemodel/stream_0_conv_3/conv1d_1/ExpandDims_1/ReadVariableOpReadVariableOpKmodel_basemodel_stream_0_conv_3_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:ђђ*
dtype02F
Dmodel/basemodel/stream_0_conv_3/conv1d_1/ExpandDims_1/ReadVariableOpИ
9model/basemodel/stream_0_conv_3/conv1d_1/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2;
9model/basemodel/stream_0_conv_3/conv1d_1/ExpandDims_1/dim┴
5model/basemodel/stream_0_conv_3/conv1d_1/ExpandDims_1
ExpandDimsLmodel/basemodel/stream_0_conv_3/conv1d_1/ExpandDims_1/ReadVariableOp:value:0Bmodel/basemodel/stream_0_conv_3/conv1d_1/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:ђђ27
5model/basemodel/stream_0_conv_3/conv1d_1/ExpandDims_1┐
(model/basemodel/stream_0_conv_3/conv1d_1Conv2D<model/basemodel/stream_0_conv_3/conv1d_1/ExpandDims:output:0>model/basemodel/stream_0_conv_3/conv1d_1/ExpandDims_1:output:0*
T0*0
_output_shapes
:         }ђ*
paddingSAME*
strides
2*
(model/basemodel/stream_0_conv_3/conv1d_1щ
0model/basemodel/stream_0_conv_3/conv1d_1/SqueezeSqueeze1model/basemodel/stream_0_conv_3/conv1d_1:output:0*
T0*,
_output_shapes
:         }ђ*
squeeze_dims

§        22
0model/basemodel/stream_0_conv_3/conv1d_1/Squeezeы
8model/basemodel/stream_0_conv_3/BiasAdd_1/ReadVariableOpReadVariableOp?model_basemodel_stream_0_conv_3_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02:
8model/basemodel/stream_0_conv_3/BiasAdd_1/ReadVariableOpЋ
)model/basemodel/stream_0_conv_3/BiasAdd_1BiasAdd9model/basemodel/stream_0_conv_3/conv1d_1/Squeeze:output:0@model/basemodel/stream_0_conv_3/BiasAdd_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:         }ђ2+
)model/basemodel/stream_0_conv_3/BiasAdd_1Ѕ
@model/basemodel/batch_normalization_2/batchnorm_1/ReadVariableOpReadVariableOpGmodel_basemodel_batch_normalization_2_batchnorm_readvariableop_resource*
_output_shapes	
:ђ*
dtype02B
@model/basemodel/batch_normalization_2/batchnorm_1/ReadVariableOpи
7model/basemodel/batch_normalization_2/batchnorm_1/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:29
7model/basemodel/batch_normalization_2/batchnorm_1/add/yЕ
5model/basemodel/batch_normalization_2/batchnorm_1/addAddV2Hmodel/basemodel/batch_normalization_2/batchnorm_1/ReadVariableOp:value:0@model/basemodel/batch_normalization_2/batchnorm_1/add/y:output:0*
T0*
_output_shapes	
:ђ27
5model/basemodel/batch_normalization_2/batchnorm_1/add▄
7model/basemodel/batch_normalization_2/batchnorm_1/RsqrtRsqrt9model/basemodel/batch_normalization_2/batchnorm_1/add:z:0*
T0*
_output_shapes	
:ђ29
7model/basemodel/batch_normalization_2/batchnorm_1/RsqrtЋ
Dmodel/basemodel/batch_normalization_2/batchnorm_1/mul/ReadVariableOpReadVariableOpKmodel_basemodel_batch_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes	
:ђ*
dtype02F
Dmodel/basemodel/batch_normalization_2/batchnorm_1/mul/ReadVariableOpд
5model/basemodel/batch_normalization_2/batchnorm_1/mulMul;model/basemodel/batch_normalization_2/batchnorm_1/Rsqrt:y:0Lmodel/basemodel/batch_normalization_2/batchnorm_1/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:ђ27
5model/basemodel/batch_normalization_2/batchnorm_1/mulЪ
7model/basemodel/batch_normalization_2/batchnorm_1/mul_1Mul2model/basemodel/stream_0_conv_3/BiasAdd_1:output:09model/basemodel/batch_normalization_2/batchnorm_1/mul:z:0*
T0*,
_output_shapes
:         }ђ29
7model/basemodel/batch_normalization_2/batchnorm_1/mul_1Ј
Bmodel/basemodel/batch_normalization_2/batchnorm_1/ReadVariableOp_1ReadVariableOpImodel_basemodel_batch_normalization_2_batchnorm_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype02D
Bmodel/basemodel/batch_normalization_2/batchnorm_1/ReadVariableOp_1д
7model/basemodel/batch_normalization_2/batchnorm_1/mul_2MulJmodel/basemodel/batch_normalization_2/batchnorm_1/ReadVariableOp_1:value:09model/basemodel/batch_normalization_2/batchnorm_1/mul:z:0*
T0*
_output_shapes	
:ђ29
7model/basemodel/batch_normalization_2/batchnorm_1/mul_2Ј
Bmodel/basemodel/batch_normalization_2/batchnorm_1/ReadVariableOp_2ReadVariableOpImodel_basemodel_batch_normalization_2_batchnorm_readvariableop_2_resource*
_output_shapes	
:ђ*
dtype02D
Bmodel/basemodel/batch_normalization_2/batchnorm_1/ReadVariableOp_2ц
5model/basemodel/batch_normalization_2/batchnorm_1/subSubJmodel/basemodel/batch_normalization_2/batchnorm_1/ReadVariableOp_2:value:0;model/basemodel/batch_normalization_2/batchnorm_1/mul_2:z:0*
T0*
_output_shapes	
:ђ27
5model/basemodel/batch_normalization_2/batchnorm_1/subф
7model/basemodel/batch_normalization_2/batchnorm_1/add_1AddV2;model/basemodel/batch_normalization_2/batchnorm_1/mul_1:z:09model/basemodel/batch_normalization_2/batchnorm_1/sub:z:0*
T0*,
_output_shapes
:         }ђ29
7model/basemodel/batch_normalization_2/batchnorm_1/add_1к
#model/basemodel/activation_2/Relu_1Relu;model/basemodel/batch_normalization_2/batchnorm_1/add_1:z:0*
T0*,
_output_shapes
:         }ђ2%
#model/basemodel/activation_2/Relu_1╬
*model/basemodel/stream_0_drop_3/Identity_1Identity1model/basemodel/activation_2/Relu_1:activations:0*
T0*,
_output_shapes
:         }ђ2,
*model/basemodel/stream_0_drop_3/Identity_1╚
Amodel/basemodel/global_average_pooling1d/Mean_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2C
Amodel/basemodel/global_average_pooling1d/Mean_1/reduction_indicesъ
/model/basemodel/global_average_pooling1d/Mean_1Mean3model/basemodel/stream_0_drop_3/Identity_1:output:0Jmodel/basemodel/global_average_pooling1d/Mean_1/reduction_indices:output:0*
T0*(
_output_shapes
:         ђ21
/model/basemodel/global_average_pooling1d/Mean_1Л
*model/basemodel/dense_1_dropout/Identity_1Identity8model/basemodel/global_average_pooling1d/Mean_1:output:0*
T0*(
_output_shapes
:         ђ2,
*model/basemodel/dense_1_dropout/Identity_1┌
/model/basemodel/dense_1/MatMul_1/ReadVariableOpReadVariableOp6model_basemodel_dense_1_matmul_readvariableop_resource*
_output_shapes
:	ђT*
dtype021
/model/basemodel/dense_1/MatMul_1/ReadVariableOpЬ
 model/basemodel/dense_1/MatMul_1MatMul3model/basemodel/dense_1_dropout/Identity_1:output:07model/basemodel/dense_1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         T2"
 model/basemodel/dense_1/MatMul_1п
0model/basemodel/dense_1/BiasAdd_1/ReadVariableOpReadVariableOp7model_basemodel_dense_1_biasadd_readvariableop_resource*
_output_shapes
:T*
dtype022
0model/basemodel/dense_1/BiasAdd_1/ReadVariableOpж
!model/basemodel/dense_1/BiasAdd_1BiasAdd*model/basemodel/dense_1/MatMul_1:product:08model/basemodel/dense_1/BiasAdd_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         T2#
!model/basemodel/dense_1/BiasAdd_1ѕ
@model/basemodel/batch_normalization_3/batchnorm_1/ReadVariableOpReadVariableOpGmodel_basemodel_batch_normalization_3_batchnorm_readvariableop_resource*
_output_shapes
:T*
dtype02B
@model/basemodel/batch_normalization_3/batchnorm_1/ReadVariableOpи
7model/basemodel/batch_normalization_3/batchnorm_1/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:29
7model/basemodel/batch_normalization_3/batchnorm_1/add/yе
5model/basemodel/batch_normalization_3/batchnorm_1/addAddV2Hmodel/basemodel/batch_normalization_3/batchnorm_1/ReadVariableOp:value:0@model/basemodel/batch_normalization_3/batchnorm_1/add/y:output:0*
T0*
_output_shapes
:T27
5model/basemodel/batch_normalization_3/batchnorm_1/add█
7model/basemodel/batch_normalization_3/batchnorm_1/RsqrtRsqrt9model/basemodel/batch_normalization_3/batchnorm_1/add:z:0*
T0*
_output_shapes
:T29
7model/basemodel/batch_normalization_3/batchnorm_1/Rsqrtћ
Dmodel/basemodel/batch_normalization_3/batchnorm_1/mul/ReadVariableOpReadVariableOpKmodel_basemodel_batch_normalization_3_batchnorm_mul_readvariableop_resource*
_output_shapes
:T*
dtype02F
Dmodel/basemodel/batch_normalization_3/batchnorm_1/mul/ReadVariableOpЦ
5model/basemodel/batch_normalization_3/batchnorm_1/mulMul;model/basemodel/batch_normalization_3/batchnorm_1/Rsqrt:y:0Lmodel/basemodel/batch_normalization_3/batchnorm_1/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:T27
5model/basemodel/batch_normalization_3/batchnorm_1/mulњ
7model/basemodel/batch_normalization_3/batchnorm_1/mul_1Mul*model/basemodel/dense_1/BiasAdd_1:output:09model/basemodel/batch_normalization_3/batchnorm_1/mul:z:0*
T0*'
_output_shapes
:         T29
7model/basemodel/batch_normalization_3/batchnorm_1/mul_1ј
Bmodel/basemodel/batch_normalization_3/batchnorm_1/ReadVariableOp_1ReadVariableOpImodel_basemodel_batch_normalization_3_batchnorm_readvariableop_1_resource*
_output_shapes
:T*
dtype02D
Bmodel/basemodel/batch_normalization_3/batchnorm_1/ReadVariableOp_1Ц
7model/basemodel/batch_normalization_3/batchnorm_1/mul_2MulJmodel/basemodel/batch_normalization_3/batchnorm_1/ReadVariableOp_1:value:09model/basemodel/batch_normalization_3/batchnorm_1/mul:z:0*
T0*
_output_shapes
:T29
7model/basemodel/batch_normalization_3/batchnorm_1/mul_2ј
Bmodel/basemodel/batch_normalization_3/batchnorm_1/ReadVariableOp_2ReadVariableOpImodel_basemodel_batch_normalization_3_batchnorm_readvariableop_2_resource*
_output_shapes
:T*
dtype02D
Bmodel/basemodel/batch_normalization_3/batchnorm_1/ReadVariableOp_2Б
5model/basemodel/batch_normalization_3/batchnorm_1/subSubJmodel/basemodel/batch_normalization_3/batchnorm_1/ReadVariableOp_2:value:0;model/basemodel/batch_normalization_3/batchnorm_1/mul_2:z:0*
T0*
_output_shapes
:T27
5model/basemodel/batch_normalization_3/batchnorm_1/subЦ
7model/basemodel/batch_normalization_3/batchnorm_1/add_1AddV2;model/basemodel/batch_normalization_3/batchnorm_1/mul_1:z:09model/basemodel/batch_normalization_3/batchnorm_1/sub:z:0*
T0*'
_output_shapes
:         T29
7model/basemodel/batch_normalization_3/batchnorm_1/add_1о
,model/basemodel/dense_activation_1/Sigmoid_1Sigmoid;model/basemodel/batch_normalization_3/batchnorm_1/add_1:z:0*
T0*'
_output_shapes
:         T2.
,model/basemodel/dense_activation_1/Sigmoid_1├
model/distance/subSub.model/basemodel/dense_activation_1/Sigmoid:y:00model/basemodel/dense_activation_1/Sigmoid_1:y:0*
T0*'
_output_shapes
:         T2
model/distance/subѓ
model/distance/SquareSquaremodel/distance/sub:z:0*
T0*'
_output_shapes
:         T2
model/distance/SquareЌ
$model/distance/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
         2&
$model/distance/Sum/reduction_indices╝
model/distance/SumSummodel/distance/Square:y:0-model/distance/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:         *
	keep_dims(2
model/distance/Sumq
model/distance/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
model/distance/ConstЕ
model/distance/MaximumMaximummodel/distance/Sum:output:0model/distance/Const:output:0*
T0*'
_output_shapes
:         2
model/distance/Maximumђ
model/distance/SqrtSqrtmodel/distance/Maximum:z:0*
T0*'
_output_shapes
:         2
model/distance/Sqrtr
IdentityIdentitymodel/distance/Sqrt:y:0^NoOp*
T0*'
_output_shapes
:         2

Identityё
NoOpNoOp=^model/basemodel/batch_normalization/batchnorm/ReadVariableOp?^model/basemodel/batch_normalization/batchnorm/ReadVariableOp_1?^model/basemodel/batch_normalization/batchnorm/ReadVariableOp_2A^model/basemodel/batch_normalization/batchnorm/mul/ReadVariableOp?^model/basemodel/batch_normalization/batchnorm_1/ReadVariableOpA^model/basemodel/batch_normalization/batchnorm_1/ReadVariableOp_1A^model/basemodel/batch_normalization/batchnorm_1/ReadVariableOp_2C^model/basemodel/batch_normalization/batchnorm_1/mul/ReadVariableOp?^model/basemodel/batch_normalization_1/batchnorm/ReadVariableOpA^model/basemodel/batch_normalization_1/batchnorm/ReadVariableOp_1A^model/basemodel/batch_normalization_1/batchnorm/ReadVariableOp_2C^model/basemodel/batch_normalization_1/batchnorm/mul/ReadVariableOpA^model/basemodel/batch_normalization_1/batchnorm_1/ReadVariableOpC^model/basemodel/batch_normalization_1/batchnorm_1/ReadVariableOp_1C^model/basemodel/batch_normalization_1/batchnorm_1/ReadVariableOp_2E^model/basemodel/batch_normalization_1/batchnorm_1/mul/ReadVariableOp?^model/basemodel/batch_normalization_2/batchnorm/ReadVariableOpA^model/basemodel/batch_normalization_2/batchnorm/ReadVariableOp_1A^model/basemodel/batch_normalization_2/batchnorm/ReadVariableOp_2C^model/basemodel/batch_normalization_2/batchnorm/mul/ReadVariableOpA^model/basemodel/batch_normalization_2/batchnorm_1/ReadVariableOpC^model/basemodel/batch_normalization_2/batchnorm_1/ReadVariableOp_1C^model/basemodel/batch_normalization_2/batchnorm_1/ReadVariableOp_2E^model/basemodel/batch_normalization_2/batchnorm_1/mul/ReadVariableOp?^model/basemodel/batch_normalization_3/batchnorm/ReadVariableOpA^model/basemodel/batch_normalization_3/batchnorm/ReadVariableOp_1A^model/basemodel/batch_normalization_3/batchnorm/ReadVariableOp_2C^model/basemodel/batch_normalization_3/batchnorm/mul/ReadVariableOpA^model/basemodel/batch_normalization_3/batchnorm_1/ReadVariableOpC^model/basemodel/batch_normalization_3/batchnorm_1/ReadVariableOp_1C^model/basemodel/batch_normalization_3/batchnorm_1/ReadVariableOp_2E^model/basemodel/batch_normalization_3/batchnorm_1/mul/ReadVariableOp/^model/basemodel/dense_1/BiasAdd/ReadVariableOp1^model/basemodel/dense_1/BiasAdd_1/ReadVariableOp.^model/basemodel/dense_1/MatMul/ReadVariableOp0^model/basemodel/dense_1/MatMul_1/ReadVariableOp7^model/basemodel/stream_0_conv_1/BiasAdd/ReadVariableOp9^model/basemodel/stream_0_conv_1/BiasAdd_1/ReadVariableOpC^model/basemodel/stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOpE^model/basemodel/stream_0_conv_1/conv1d_1/ExpandDims_1/ReadVariableOp7^model/basemodel/stream_0_conv_2/BiasAdd/ReadVariableOp9^model/basemodel/stream_0_conv_2/BiasAdd_1/ReadVariableOpC^model/basemodel/stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOpE^model/basemodel/stream_0_conv_2/conv1d_1/ExpandDims_1/ReadVariableOp7^model/basemodel/stream_0_conv_3/BiasAdd/ReadVariableOp9^model/basemodel/stream_0_conv_3/BiasAdd_1/ReadVariableOpC^model/basemodel/stream_0_conv_3/conv1d/ExpandDims_1/ReadVariableOpE^model/basemodel/stream_0_conv_3/conv1d_1/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*q
_input_shapes`
^:         }:         }: : : : : : : : : : : : : : : : : : : : : : : : 2|
<model/basemodel/batch_normalization/batchnorm/ReadVariableOp<model/basemodel/batch_normalization/batchnorm/ReadVariableOp2ђ
>model/basemodel/batch_normalization/batchnorm/ReadVariableOp_1>model/basemodel/batch_normalization/batchnorm/ReadVariableOp_12ђ
>model/basemodel/batch_normalization/batchnorm/ReadVariableOp_2>model/basemodel/batch_normalization/batchnorm/ReadVariableOp_22ё
@model/basemodel/batch_normalization/batchnorm/mul/ReadVariableOp@model/basemodel/batch_normalization/batchnorm/mul/ReadVariableOp2ђ
>model/basemodel/batch_normalization/batchnorm_1/ReadVariableOp>model/basemodel/batch_normalization/batchnorm_1/ReadVariableOp2ё
@model/basemodel/batch_normalization/batchnorm_1/ReadVariableOp_1@model/basemodel/batch_normalization/batchnorm_1/ReadVariableOp_12ё
@model/basemodel/batch_normalization/batchnorm_1/ReadVariableOp_2@model/basemodel/batch_normalization/batchnorm_1/ReadVariableOp_22ѕ
Bmodel/basemodel/batch_normalization/batchnorm_1/mul/ReadVariableOpBmodel/basemodel/batch_normalization/batchnorm_1/mul/ReadVariableOp2ђ
>model/basemodel/batch_normalization_1/batchnorm/ReadVariableOp>model/basemodel/batch_normalization_1/batchnorm/ReadVariableOp2ё
@model/basemodel/batch_normalization_1/batchnorm/ReadVariableOp_1@model/basemodel/batch_normalization_1/batchnorm/ReadVariableOp_12ё
@model/basemodel/batch_normalization_1/batchnorm/ReadVariableOp_2@model/basemodel/batch_normalization_1/batchnorm/ReadVariableOp_22ѕ
Bmodel/basemodel/batch_normalization_1/batchnorm/mul/ReadVariableOpBmodel/basemodel/batch_normalization_1/batchnorm/mul/ReadVariableOp2ё
@model/basemodel/batch_normalization_1/batchnorm_1/ReadVariableOp@model/basemodel/batch_normalization_1/batchnorm_1/ReadVariableOp2ѕ
Bmodel/basemodel/batch_normalization_1/batchnorm_1/ReadVariableOp_1Bmodel/basemodel/batch_normalization_1/batchnorm_1/ReadVariableOp_12ѕ
Bmodel/basemodel/batch_normalization_1/batchnorm_1/ReadVariableOp_2Bmodel/basemodel/batch_normalization_1/batchnorm_1/ReadVariableOp_22ї
Dmodel/basemodel/batch_normalization_1/batchnorm_1/mul/ReadVariableOpDmodel/basemodel/batch_normalization_1/batchnorm_1/mul/ReadVariableOp2ђ
>model/basemodel/batch_normalization_2/batchnorm/ReadVariableOp>model/basemodel/batch_normalization_2/batchnorm/ReadVariableOp2ё
@model/basemodel/batch_normalization_2/batchnorm/ReadVariableOp_1@model/basemodel/batch_normalization_2/batchnorm/ReadVariableOp_12ё
@model/basemodel/batch_normalization_2/batchnorm/ReadVariableOp_2@model/basemodel/batch_normalization_2/batchnorm/ReadVariableOp_22ѕ
Bmodel/basemodel/batch_normalization_2/batchnorm/mul/ReadVariableOpBmodel/basemodel/batch_normalization_2/batchnorm/mul/ReadVariableOp2ё
@model/basemodel/batch_normalization_2/batchnorm_1/ReadVariableOp@model/basemodel/batch_normalization_2/batchnorm_1/ReadVariableOp2ѕ
Bmodel/basemodel/batch_normalization_2/batchnorm_1/ReadVariableOp_1Bmodel/basemodel/batch_normalization_2/batchnorm_1/ReadVariableOp_12ѕ
Bmodel/basemodel/batch_normalization_2/batchnorm_1/ReadVariableOp_2Bmodel/basemodel/batch_normalization_2/batchnorm_1/ReadVariableOp_22ї
Dmodel/basemodel/batch_normalization_2/batchnorm_1/mul/ReadVariableOpDmodel/basemodel/batch_normalization_2/batchnorm_1/mul/ReadVariableOp2ђ
>model/basemodel/batch_normalization_3/batchnorm/ReadVariableOp>model/basemodel/batch_normalization_3/batchnorm/ReadVariableOp2ё
@model/basemodel/batch_normalization_3/batchnorm/ReadVariableOp_1@model/basemodel/batch_normalization_3/batchnorm/ReadVariableOp_12ё
@model/basemodel/batch_normalization_3/batchnorm/ReadVariableOp_2@model/basemodel/batch_normalization_3/batchnorm/ReadVariableOp_22ѕ
Bmodel/basemodel/batch_normalization_3/batchnorm/mul/ReadVariableOpBmodel/basemodel/batch_normalization_3/batchnorm/mul/ReadVariableOp2ё
@model/basemodel/batch_normalization_3/batchnorm_1/ReadVariableOp@model/basemodel/batch_normalization_3/batchnorm_1/ReadVariableOp2ѕ
Bmodel/basemodel/batch_normalization_3/batchnorm_1/ReadVariableOp_1Bmodel/basemodel/batch_normalization_3/batchnorm_1/ReadVariableOp_12ѕ
Bmodel/basemodel/batch_normalization_3/batchnorm_1/ReadVariableOp_2Bmodel/basemodel/batch_normalization_3/batchnorm_1/ReadVariableOp_22ї
Dmodel/basemodel/batch_normalization_3/batchnorm_1/mul/ReadVariableOpDmodel/basemodel/batch_normalization_3/batchnorm_1/mul/ReadVariableOp2`
.model/basemodel/dense_1/BiasAdd/ReadVariableOp.model/basemodel/dense_1/BiasAdd/ReadVariableOp2d
0model/basemodel/dense_1/BiasAdd_1/ReadVariableOp0model/basemodel/dense_1/BiasAdd_1/ReadVariableOp2^
-model/basemodel/dense_1/MatMul/ReadVariableOp-model/basemodel/dense_1/MatMul/ReadVariableOp2b
/model/basemodel/dense_1/MatMul_1/ReadVariableOp/model/basemodel/dense_1/MatMul_1/ReadVariableOp2p
6model/basemodel/stream_0_conv_1/BiasAdd/ReadVariableOp6model/basemodel/stream_0_conv_1/BiasAdd/ReadVariableOp2t
8model/basemodel/stream_0_conv_1/BiasAdd_1/ReadVariableOp8model/basemodel/stream_0_conv_1/BiasAdd_1/ReadVariableOp2ѕ
Bmodel/basemodel/stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOpBmodel/basemodel/stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp2ї
Dmodel/basemodel/stream_0_conv_1/conv1d_1/ExpandDims_1/ReadVariableOpDmodel/basemodel/stream_0_conv_1/conv1d_1/ExpandDims_1/ReadVariableOp2p
6model/basemodel/stream_0_conv_2/BiasAdd/ReadVariableOp6model/basemodel/stream_0_conv_2/BiasAdd/ReadVariableOp2t
8model/basemodel/stream_0_conv_2/BiasAdd_1/ReadVariableOp8model/basemodel/stream_0_conv_2/BiasAdd_1/ReadVariableOp2ѕ
Bmodel/basemodel/stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOpBmodel/basemodel/stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOp2ї
Dmodel/basemodel/stream_0_conv_2/conv1d_1/ExpandDims_1/ReadVariableOpDmodel/basemodel/stream_0_conv_2/conv1d_1/ExpandDims_1/ReadVariableOp2p
6model/basemodel/stream_0_conv_3/BiasAdd/ReadVariableOp6model/basemodel/stream_0_conv_3/BiasAdd/ReadVariableOp2t
8model/basemodel/stream_0_conv_3/BiasAdd_1/ReadVariableOp8model/basemodel/stream_0_conv_3/BiasAdd_1/ReadVariableOp2ѕ
Bmodel/basemodel/stream_0_conv_3/conv1d/ExpandDims_1/ReadVariableOpBmodel/basemodel/stream_0_conv_3/conv1d/ExpandDims_1/ReadVariableOp2ї
Dmodel/basemodel/stream_0_conv_3/conv1d_1/ExpandDims_1/ReadVariableOpDmodel/basemodel/stream_0_conv_3/conv1d_1/ExpandDims_1/ReadVariableOp:X T
+
_output_shapes
:         }
%
_user_specified_nameleft_inputs:YU
+
_output_shapes
:         }
&
_user_specified_nameright_inputs
ю
Б
/__inference_stream_0_conv_3_layer_call_fn_28566

inputs
unknown:ђђ
	unknown_0:	ђ
identityѕбStatefulPartitionedCallѓ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         }ђ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *S
fNRL
J__inference_stream_0_conv_3_layer_call_and_return_conditional_losses_240452
StatefulPartitionedCallђ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:         }ђ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         }ђ: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:         }ђ
 
_user_specified_nameinputs
┤
Г
N__inference_batch_normalization_layer_call_and_return_conditional_losses_23222

inputs/
!batchnorm_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@1
#batchnorm_readvariableop_1_resource:@1
#batchnorm_readvariableop_2_resource:@
identityѕбbatchnorm/ReadVariableOpбbatchnorm/ReadVariableOp_1бbatchnorm/ReadVariableOp_2бbatchnorm/mul/ReadVariableOpњ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2
batchnorm/add/yѕ
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@2
batchnorm/Rsqrtъ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOpЁ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2
batchnorm/mulЃ
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :                  @2
batchnorm/mul_1ў
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp_1Ё
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:@2
batchnorm/mul_2ў
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp_2Ѓ
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2
batchnorm/subњ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :                  @2
batchnorm/add_1{
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :                  @2

Identity┬
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:                  @: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :                  @
 
_user_specified_nameinputs
ЁБ
њ
D__inference_basemodel_layer_call_and_return_conditional_losses_25655

inputsQ
;stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource:@=
/stream_0_conv_1_biasadd_readvariableop_resource:@I
;batch_normalization_assignmovingavg_readvariableop_resource:@K
=batch_normalization_assignmovingavg_1_readvariableop_resource:@G
9batch_normalization_batchnorm_mul_readvariableop_resource:@C
5batch_normalization_batchnorm_readvariableop_resource:@R
;stream_0_conv_2_conv1d_expanddims_1_readvariableop_resource:@ђ>
/stream_0_conv_2_biasadd_readvariableop_resource:	ђL
=batch_normalization_1_assignmovingavg_readvariableop_resource:	ђN
?batch_normalization_1_assignmovingavg_1_readvariableop_resource:	ђJ
;batch_normalization_1_batchnorm_mul_readvariableop_resource:	ђF
7batch_normalization_1_batchnorm_readvariableop_resource:	ђS
;stream_0_conv_3_conv1d_expanddims_1_readvariableop_resource:ђђ>
/stream_0_conv_3_biasadd_readvariableop_resource:	ђL
=batch_normalization_2_assignmovingavg_readvariableop_resource:	ђN
?batch_normalization_2_assignmovingavg_1_readvariableop_resource:	ђJ
;batch_normalization_2_batchnorm_mul_readvariableop_resource:	ђF
7batch_normalization_2_batchnorm_readvariableop_resource:	ђ9
&dense_1_matmul_readvariableop_resource:	ђT5
'dense_1_biasadd_readvariableop_resource:TK
=batch_normalization_3_assignmovingavg_readvariableop_resource:TM
?batch_normalization_3_assignmovingavg_1_readvariableop_resource:TI
;batch_normalization_3_batchnorm_mul_readvariableop_resource:TE
7batch_normalization_3_batchnorm_readvariableop_resource:T
identityѕб#batch_normalization/AssignMovingAvgб2batch_normalization/AssignMovingAvg/ReadVariableOpб%batch_normalization/AssignMovingAvg_1б4batch_normalization/AssignMovingAvg_1/ReadVariableOpб,batch_normalization/batchnorm/ReadVariableOpб0batch_normalization/batchnorm/mul/ReadVariableOpб%batch_normalization_1/AssignMovingAvgб4batch_normalization_1/AssignMovingAvg/ReadVariableOpб'batch_normalization_1/AssignMovingAvg_1б6batch_normalization_1/AssignMovingAvg_1/ReadVariableOpб.batch_normalization_1/batchnorm/ReadVariableOpб2batch_normalization_1/batchnorm/mul/ReadVariableOpб%batch_normalization_2/AssignMovingAvgб4batch_normalization_2/AssignMovingAvg/ReadVariableOpб'batch_normalization_2/AssignMovingAvg_1б6batch_normalization_2/AssignMovingAvg_1/ReadVariableOpб.batch_normalization_2/batchnorm/ReadVariableOpб2batch_normalization_2/batchnorm/mul/ReadVariableOpб%batch_normalization_3/AssignMovingAvgб4batch_normalization_3/AssignMovingAvg/ReadVariableOpб'batch_normalization_3/AssignMovingAvg_1б6batch_normalization_3/AssignMovingAvg_1/ReadVariableOpб.batch_normalization_3/batchnorm/ReadVariableOpб2batch_normalization_3/batchnorm/mul/ReadVariableOpбdense_1/BiasAdd/ReadVariableOpбdense_1/MatMul/ReadVariableOpб-dense_1/kernel/Regularizer/Abs/ReadVariableOpб&stream_0_conv_1/BiasAdd/ReadVariableOpб2stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOpб5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpб&stream_0_conv_2/BiasAdd/ReadVariableOpб2stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOpб8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOpб&stream_0_conv_3/BiasAdd/ReadVariableOpб2stream_0_conv_3/conv1d/ExpandDims_1/ReadVariableOpб5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOpІ
!stream_0_input_drop/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?2#
!stream_0_input_drop/dropout/Const│
stream_0_input_drop/dropout/MulMulinputs*stream_0_input_drop/dropout/Const:output:0*
T0*+
_output_shapes
:         }2!
stream_0_input_drop/dropout/Mul|
!stream_0_input_drop/dropout/ShapeShapeinputs*
T0*
_output_shapes
:2#
!stream_0_input_drop/dropout/ShapeЈ
8stream_0_input_drop/dropout/random_uniform/RandomUniformRandomUniform*stream_0_input_drop/dropout/Shape:output:0*
T0*+
_output_shapes
:         }*
dtype0*
seedи*
seed2и2:
8stream_0_input_drop/dropout/random_uniform/RandomUniformЮ
*stream_0_input_drop/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>2,
*stream_0_input_drop/dropout/GreaterEqual/yњ
(stream_0_input_drop/dropout/GreaterEqualGreaterEqualAstream_0_input_drop/dropout/random_uniform/RandomUniform:output:03stream_0_input_drop/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:         }2*
(stream_0_input_drop/dropout/GreaterEqual┐
 stream_0_input_drop/dropout/CastCast,stream_0_input_drop/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:         }2"
 stream_0_input_drop/dropout/Cast╬
!stream_0_input_drop/dropout/Mul_1Mul#stream_0_input_drop/dropout/Mul:z:0$stream_0_input_drop/dropout/Cast:y:0*
T0*+
_output_shapes
:         }2#
!stream_0_input_drop/dropout/Mul_1Ў
%stream_0_conv_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§        2'
%stream_0_conv_1/conv1d/ExpandDims/dimт
!stream_0_conv_1/conv1d/ExpandDims
ExpandDims%stream_0_input_drop/dropout/Mul_1:z:0.stream_0_conv_1/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:         }2#
!stream_0_conv_1/conv1d/ExpandDimsУ
2stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp;stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype024
2stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOpћ
'stream_0_conv_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2)
'stream_0_conv_1/conv1d/ExpandDims_1/dimэ
#stream_0_conv_1/conv1d/ExpandDims_1
ExpandDims:stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp:value:00stream_0_conv_1/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@2%
#stream_0_conv_1/conv1d/ExpandDims_1Ш
stream_0_conv_1/conv1dConv2D*stream_0_conv_1/conv1d/ExpandDims:output:0,stream_0_conv_1/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:         }@*
paddingSAME*
strides
2
stream_0_conv_1/conv1d┬
stream_0_conv_1/conv1d/SqueezeSqueezestream_0_conv_1/conv1d:output:0*
T0*+
_output_shapes
:         }@*
squeeze_dims

§        2 
stream_0_conv_1/conv1d/Squeeze╝
&stream_0_conv_1/BiasAdd/ReadVariableOpReadVariableOp/stream_0_conv_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02(
&stream_0_conv_1/BiasAdd/ReadVariableOp╠
stream_0_conv_1/BiasAddBiasAdd'stream_0_conv_1/conv1d/Squeeze:output:0.stream_0_conv_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         }@2
stream_0_conv_1/BiasAdd╣
2batch_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       24
2batch_normalization/moments/mean/reduction_indicesж
 batch_normalization/moments/meanMean stream_0_conv_1/BiasAdd:output:0;batch_normalization/moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(2"
 batch_normalization/moments/mean╝
(batch_normalization/moments/StopGradientStopGradient)batch_normalization/moments/mean:output:0*
T0*"
_output_shapes
:@2*
(batch_normalization/moments/StopGradient■
-batch_normalization/moments/SquaredDifferenceSquaredDifference stream_0_conv_1/BiasAdd:output:01batch_normalization/moments/StopGradient:output:0*
T0*+
_output_shapes
:         }@2/
-batch_normalization/moments/SquaredDifference┴
6batch_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       28
6batch_normalization/moments/variance/reduction_indicesє
$batch_normalization/moments/varianceMean1batch_normalization/moments/SquaredDifference:z:0?batch_normalization/moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(2&
$batch_normalization/moments/varianceй
#batch_normalization/moments/SqueezeSqueeze)batch_normalization/moments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 2%
#batch_normalization/moments/Squeeze┼
%batch_normalization/moments/Squeeze_1Squeeze-batch_normalization/moments/variance:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 2'
%batch_normalization/moments/Squeeze_1Џ
)batch_normalization/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2+
)batch_normalization/AssignMovingAvg/decayЯ
2batch_normalization/AssignMovingAvg/ReadVariableOpReadVariableOp;batch_normalization_assignmovingavg_readvariableop_resource*
_output_shapes
:@*
dtype024
2batch_normalization/AssignMovingAvg/ReadVariableOpУ
'batch_normalization/AssignMovingAvg/subSub:batch_normalization/AssignMovingAvg/ReadVariableOp:value:0,batch_normalization/moments/Squeeze:output:0*
T0*
_output_shapes
:@2)
'batch_normalization/AssignMovingAvg/sub▀
'batch_normalization/AssignMovingAvg/mulMul+batch_normalization/AssignMovingAvg/sub:z:02batch_normalization/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:@2)
'batch_normalization/AssignMovingAvg/mulБ
#batch_normalization/AssignMovingAvgAssignSubVariableOp;batch_normalization_assignmovingavg_readvariableop_resource+batch_normalization/AssignMovingAvg/mul:z:03^batch_normalization/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02%
#batch_normalization/AssignMovingAvgЪ
+batch_normalization/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2-
+batch_normalization/AssignMovingAvg_1/decayТ
4batch_normalization/AssignMovingAvg_1/ReadVariableOpReadVariableOp=batch_normalization_assignmovingavg_1_readvariableop_resource*
_output_shapes
:@*
dtype026
4batch_normalization/AssignMovingAvg_1/ReadVariableOp­
)batch_normalization/AssignMovingAvg_1/subSub<batch_normalization/AssignMovingAvg_1/ReadVariableOp:value:0.batch_normalization/moments/Squeeze_1:output:0*
T0*
_output_shapes
:@2+
)batch_normalization/AssignMovingAvg_1/subу
)batch_normalization/AssignMovingAvg_1/mulMul-batch_normalization/AssignMovingAvg_1/sub:z:04batch_normalization/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:@2+
)batch_normalization/AssignMovingAvg_1/mulГ
%batch_normalization/AssignMovingAvg_1AssignSubVariableOp=batch_normalization_assignmovingavg_1_readvariableop_resource-batch_normalization/AssignMovingAvg_1/mul:z:05^batch_normalization/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02'
%batch_normalization/AssignMovingAvg_1Ј
#batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2%
#batch_normalization/batchnorm/add/yм
!batch_normalization/batchnorm/addAddV2.batch_normalization/moments/Squeeze_1:output:0,batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes
:@2#
!batch_normalization/batchnorm/addЪ
#batch_normalization/batchnorm/RsqrtRsqrt%batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes
:@2%
#batch_normalization/batchnorm/Rsqrt┌
0batch_normalization/batchnorm/mul/ReadVariableOpReadVariableOp9batch_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype022
0batch_normalization/batchnorm/mul/ReadVariableOpН
!batch_normalization/batchnorm/mulMul'batch_normalization/batchnorm/Rsqrt:y:08batch_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2#
!batch_normalization/batchnorm/mulл
#batch_normalization/batchnorm/mul_1Mul stream_0_conv_1/BiasAdd:output:0%batch_normalization/batchnorm/mul:z:0*
T0*+
_output_shapes
:         }@2%
#batch_normalization/batchnorm/mul_1╦
#batch_normalization/batchnorm/mul_2Mul,batch_normalization/moments/Squeeze:output:0%batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes
:@2%
#batch_normalization/batchnorm/mul_2╬
,batch_normalization/batchnorm/ReadVariableOpReadVariableOp5batch_normalization_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02.
,batch_normalization/batchnorm/ReadVariableOpЛ
!batch_normalization/batchnorm/subSub4batch_normalization/batchnorm/ReadVariableOp:value:0'batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2#
!batch_normalization/batchnorm/sub┘
#batch_normalization/batchnorm/add_1AddV2'batch_normalization/batchnorm/mul_1:z:0%batch_normalization/batchnorm/sub:z:0*
T0*+
_output_shapes
:         }@2%
#batch_normalization/batchnorm/add_1Ѕ
activation/ReluRelu'batch_normalization/batchnorm/add_1:z:0*
T0*+
_output_shapes
:         }@2
activation/ReluЃ
stream_0_drop_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n█Х?2
stream_0_drop_1/dropout/ConstЙ
stream_0_drop_1/dropout/MulMulactivation/Relu:activations:0&stream_0_drop_1/dropout/Const:output:0*
T0*+
_output_shapes
:         }@2
stream_0_drop_1/dropout/MulІ
stream_0_drop_1/dropout/ShapeShapeactivation/Relu:activations:0*
T0*
_output_shapes
:2
stream_0_drop_1/dropout/ShapeЃ
4stream_0_drop_1/dropout/random_uniform/RandomUniformRandomUniform&stream_0_drop_1/dropout/Shape:output:0*
T0*+
_output_shapes
:         }@*
dtype0*
seedи*
seed2и26
4stream_0_drop_1/dropout/random_uniform/RandomUniformЋ
&stream_0_drop_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *џЎЎ>2(
&stream_0_drop_1/dropout/GreaterEqual/yѓ
$stream_0_drop_1/dropout/GreaterEqualGreaterEqual=stream_0_drop_1/dropout/random_uniform/RandomUniform:output:0/stream_0_drop_1/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:         }@2&
$stream_0_drop_1/dropout/GreaterEqual│
stream_0_drop_1/dropout/CastCast(stream_0_drop_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:         }@2
stream_0_drop_1/dropout/CastЙ
stream_0_drop_1/dropout/Mul_1Mulstream_0_drop_1/dropout/Mul:z:0 stream_0_drop_1/dropout/Cast:y:0*
T0*+
_output_shapes
:         }@2
stream_0_drop_1/dropout/Mul_1Ў
%stream_0_conv_2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§        2'
%stream_0_conv_2/conv1d/ExpandDims/dimр
!stream_0_conv_2/conv1d/ExpandDims
ExpandDims!stream_0_drop_1/dropout/Mul_1:z:0.stream_0_conv_2/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:         }@2#
!stream_0_conv_2/conv1d/ExpandDimsж
2stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp;stream_0_conv_2_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:@ђ*
dtype024
2stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOpћ
'stream_0_conv_2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2)
'stream_0_conv_2/conv1d/ExpandDims_1/dimЭ
#stream_0_conv_2/conv1d/ExpandDims_1
ExpandDims:stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOp:value:00stream_0_conv_2/conv1d/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:@ђ2%
#stream_0_conv_2/conv1d/ExpandDims_1э
stream_0_conv_2/conv1dConv2D*stream_0_conv_2/conv1d/ExpandDims:output:0,stream_0_conv_2/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:         }ђ*
paddingSAME*
strides
2
stream_0_conv_2/conv1d├
stream_0_conv_2/conv1d/SqueezeSqueezestream_0_conv_2/conv1d:output:0*
T0*,
_output_shapes
:         }ђ*
squeeze_dims

§        2 
stream_0_conv_2/conv1d/Squeezeй
&stream_0_conv_2/BiasAdd/ReadVariableOpReadVariableOp/stream_0_conv_2_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02(
&stream_0_conv_2/BiasAdd/ReadVariableOp═
stream_0_conv_2/BiasAddBiasAdd'stream_0_conv_2/conv1d/Squeeze:output:0.stream_0_conv_2/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:         }ђ2
stream_0_conv_2/BiasAddй
4batch_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       26
4batch_normalization_1/moments/mean/reduction_indices­
"batch_normalization_1/moments/meanMean stream_0_conv_2/BiasAdd:output:0=batch_normalization_1/moments/mean/reduction_indices:output:0*
T0*#
_output_shapes
:ђ*
	keep_dims(2$
"batch_normalization_1/moments/mean├
*batch_normalization_1/moments/StopGradientStopGradient+batch_normalization_1/moments/mean:output:0*
T0*#
_output_shapes
:ђ2,
*batch_normalization_1/moments/StopGradientЁ
/batch_normalization_1/moments/SquaredDifferenceSquaredDifference stream_0_conv_2/BiasAdd:output:03batch_normalization_1/moments/StopGradient:output:0*
T0*,
_output_shapes
:         }ђ21
/batch_normalization_1/moments/SquaredDifference┼
8batch_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2:
8batch_normalization_1/moments/variance/reduction_indicesЈ
&batch_normalization_1/moments/varianceMean3batch_normalization_1/moments/SquaredDifference:z:0Abatch_normalization_1/moments/variance/reduction_indices:output:0*
T0*#
_output_shapes
:ђ*
	keep_dims(2(
&batch_normalization_1/moments/variance─
%batch_normalization_1/moments/SqueezeSqueeze+batch_normalization_1/moments/mean:output:0*
T0*
_output_shapes	
:ђ*
squeeze_dims
 2'
%batch_normalization_1/moments/Squeeze╠
'batch_normalization_1/moments/Squeeze_1Squeeze/batch_normalization_1/moments/variance:output:0*
T0*
_output_shapes	
:ђ*
squeeze_dims
 2)
'batch_normalization_1/moments/Squeeze_1Ъ
+batch_normalization_1/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2-
+batch_normalization_1/AssignMovingAvg/decayу
4batch_normalization_1/AssignMovingAvg/ReadVariableOpReadVariableOp=batch_normalization_1_assignmovingavg_readvariableop_resource*
_output_shapes	
:ђ*
dtype026
4batch_normalization_1/AssignMovingAvg/ReadVariableOpы
)batch_normalization_1/AssignMovingAvg/subSub<batch_normalization_1/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_1/moments/Squeeze:output:0*
T0*
_output_shapes	
:ђ2+
)batch_normalization_1/AssignMovingAvg/subУ
)batch_normalization_1/AssignMovingAvg/mulMul-batch_normalization_1/AssignMovingAvg/sub:z:04batch_normalization_1/AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:ђ2+
)batch_normalization_1/AssignMovingAvg/mulГ
%batch_normalization_1/AssignMovingAvgAssignSubVariableOp=batch_normalization_1_assignmovingavg_readvariableop_resource-batch_normalization_1/AssignMovingAvg/mul:z:05^batch_normalization_1/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02'
%batch_normalization_1/AssignMovingAvgБ
-batch_normalization_1/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2/
-batch_normalization_1/AssignMovingAvg_1/decayь
6batch_normalization_1/AssignMovingAvg_1/ReadVariableOpReadVariableOp?batch_normalization_1_assignmovingavg_1_readvariableop_resource*
_output_shapes	
:ђ*
dtype028
6batch_normalization_1/AssignMovingAvg_1/ReadVariableOpщ
+batch_normalization_1/AssignMovingAvg_1/subSub>batch_normalization_1/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_1/moments/Squeeze_1:output:0*
T0*
_output_shapes	
:ђ2-
+batch_normalization_1/AssignMovingAvg_1/sub­
+batch_normalization_1/AssignMovingAvg_1/mulMul/batch_normalization_1/AssignMovingAvg_1/sub:z:06batch_normalization_1/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:ђ2-
+batch_normalization_1/AssignMovingAvg_1/mulи
'batch_normalization_1/AssignMovingAvg_1AssignSubVariableOp?batch_normalization_1_assignmovingavg_1_readvariableop_resource/batch_normalization_1/AssignMovingAvg_1/mul:z:07^batch_normalization_1/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02)
'batch_normalization_1/AssignMovingAvg_1Њ
%batch_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2'
%batch_normalization_1/batchnorm/add/y█
#batch_normalization_1/batchnorm/addAddV20batch_normalization_1/moments/Squeeze_1:output:0.batch_normalization_1/batchnorm/add/y:output:0*
T0*
_output_shapes	
:ђ2%
#batch_normalization_1/batchnorm/addд
%batch_normalization_1/batchnorm/RsqrtRsqrt'batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes	
:ђ2'
%batch_normalization_1/batchnorm/Rsqrtр
2batch_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes	
:ђ*
dtype024
2batch_normalization_1/batchnorm/mul/ReadVariableOpя
#batch_normalization_1/batchnorm/mulMul)batch_normalization_1/batchnorm/Rsqrt:y:0:batch_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:ђ2%
#batch_normalization_1/batchnorm/mulО
%batch_normalization_1/batchnorm/mul_1Mul stream_0_conv_2/BiasAdd:output:0'batch_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:         }ђ2'
%batch_normalization_1/batchnorm/mul_1н
%batch_normalization_1/batchnorm/mul_2Mul.batch_normalization_1/moments/Squeeze:output:0'batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes	
:ђ2'
%batch_normalization_1/batchnorm/mul_2Н
.batch_normalization_1/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_1_batchnorm_readvariableop_resource*
_output_shapes	
:ђ*
dtype020
.batch_normalization_1/batchnorm/ReadVariableOp┌
#batch_normalization_1/batchnorm/subSub6batch_normalization_1/batchnorm/ReadVariableOp:value:0)batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:ђ2%
#batch_normalization_1/batchnorm/subР
%batch_normalization_1/batchnorm/add_1AddV2)batch_normalization_1/batchnorm/mul_1:z:0'batch_normalization_1/batchnorm/sub:z:0*
T0*,
_output_shapes
:         }ђ2'
%batch_normalization_1/batchnorm/add_1љ
activation_1/ReluRelu)batch_normalization_1/batchnorm/add_1:z:0*
T0*,
_output_shapes
:         }ђ2
activation_1/ReluЃ
stream_0_drop_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUН?2
stream_0_drop_2/dropout/Const┴
stream_0_drop_2/dropout/MulMulactivation_1/Relu:activations:0&stream_0_drop_2/dropout/Const:output:0*
T0*,
_output_shapes
:         }ђ2
stream_0_drop_2/dropout/MulЇ
stream_0_drop_2/dropout/ShapeShapeactivation_1/Relu:activations:0*
T0*
_output_shapes
:2
stream_0_drop_2/dropout/Shapeё
4stream_0_drop_2/dropout/random_uniform/RandomUniformRandomUniform&stream_0_drop_2/dropout/Shape:output:0*
T0*,
_output_shapes
:         }ђ*
dtype0*
seedи*
seed2и26
4stream_0_drop_2/dropout/random_uniform/RandomUniformЋ
&stream_0_drop_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠>2(
&stream_0_drop_2/dropout/GreaterEqual/yЃ
$stream_0_drop_2/dropout/GreaterEqualGreaterEqual=stream_0_drop_2/dropout/random_uniform/RandomUniform:output:0/stream_0_drop_2/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:         }ђ2&
$stream_0_drop_2/dropout/GreaterEqual┤
stream_0_drop_2/dropout/CastCast(stream_0_drop_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:         }ђ2
stream_0_drop_2/dropout/Cast┐
stream_0_drop_2/dropout/Mul_1Mulstream_0_drop_2/dropout/Mul:z:0 stream_0_drop_2/dropout/Cast:y:0*
T0*,
_output_shapes
:         }ђ2
stream_0_drop_2/dropout/Mul_1Ў
%stream_0_conv_3/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§        2'
%stream_0_conv_3/conv1d/ExpandDims/dimР
!stream_0_conv_3/conv1d/ExpandDims
ExpandDims!stream_0_drop_2/dropout/Mul_1:z:0.stream_0_conv_3/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         }ђ2#
!stream_0_conv_3/conv1d/ExpandDimsЖ
2stream_0_conv_3/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp;stream_0_conv_3_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:ђђ*
dtype024
2stream_0_conv_3/conv1d/ExpandDims_1/ReadVariableOpћ
'stream_0_conv_3/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2)
'stream_0_conv_3/conv1d/ExpandDims_1/dimщ
#stream_0_conv_3/conv1d/ExpandDims_1
ExpandDims:stream_0_conv_3/conv1d/ExpandDims_1/ReadVariableOp:value:00stream_0_conv_3/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:ђђ2%
#stream_0_conv_3/conv1d/ExpandDims_1э
stream_0_conv_3/conv1dConv2D*stream_0_conv_3/conv1d/ExpandDims:output:0,stream_0_conv_3/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:         }ђ*
paddingSAME*
strides
2
stream_0_conv_3/conv1d├
stream_0_conv_3/conv1d/SqueezeSqueezestream_0_conv_3/conv1d:output:0*
T0*,
_output_shapes
:         }ђ*
squeeze_dims

§        2 
stream_0_conv_3/conv1d/Squeezeй
&stream_0_conv_3/BiasAdd/ReadVariableOpReadVariableOp/stream_0_conv_3_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02(
&stream_0_conv_3/BiasAdd/ReadVariableOp═
stream_0_conv_3/BiasAddBiasAdd'stream_0_conv_3/conv1d/Squeeze:output:0.stream_0_conv_3/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:         }ђ2
stream_0_conv_3/BiasAddй
4batch_normalization_2/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       26
4batch_normalization_2/moments/mean/reduction_indices­
"batch_normalization_2/moments/meanMean stream_0_conv_3/BiasAdd:output:0=batch_normalization_2/moments/mean/reduction_indices:output:0*
T0*#
_output_shapes
:ђ*
	keep_dims(2$
"batch_normalization_2/moments/mean├
*batch_normalization_2/moments/StopGradientStopGradient+batch_normalization_2/moments/mean:output:0*
T0*#
_output_shapes
:ђ2,
*batch_normalization_2/moments/StopGradientЁ
/batch_normalization_2/moments/SquaredDifferenceSquaredDifference stream_0_conv_3/BiasAdd:output:03batch_normalization_2/moments/StopGradient:output:0*
T0*,
_output_shapes
:         }ђ21
/batch_normalization_2/moments/SquaredDifference┼
8batch_normalization_2/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2:
8batch_normalization_2/moments/variance/reduction_indicesЈ
&batch_normalization_2/moments/varianceMean3batch_normalization_2/moments/SquaredDifference:z:0Abatch_normalization_2/moments/variance/reduction_indices:output:0*
T0*#
_output_shapes
:ђ*
	keep_dims(2(
&batch_normalization_2/moments/variance─
%batch_normalization_2/moments/SqueezeSqueeze+batch_normalization_2/moments/mean:output:0*
T0*
_output_shapes	
:ђ*
squeeze_dims
 2'
%batch_normalization_2/moments/Squeeze╠
'batch_normalization_2/moments/Squeeze_1Squeeze/batch_normalization_2/moments/variance:output:0*
T0*
_output_shapes	
:ђ*
squeeze_dims
 2)
'batch_normalization_2/moments/Squeeze_1Ъ
+batch_normalization_2/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2-
+batch_normalization_2/AssignMovingAvg/decayу
4batch_normalization_2/AssignMovingAvg/ReadVariableOpReadVariableOp=batch_normalization_2_assignmovingavg_readvariableop_resource*
_output_shapes	
:ђ*
dtype026
4batch_normalization_2/AssignMovingAvg/ReadVariableOpы
)batch_normalization_2/AssignMovingAvg/subSub<batch_normalization_2/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_2/moments/Squeeze:output:0*
T0*
_output_shapes	
:ђ2+
)batch_normalization_2/AssignMovingAvg/subУ
)batch_normalization_2/AssignMovingAvg/mulMul-batch_normalization_2/AssignMovingAvg/sub:z:04batch_normalization_2/AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:ђ2+
)batch_normalization_2/AssignMovingAvg/mulГ
%batch_normalization_2/AssignMovingAvgAssignSubVariableOp=batch_normalization_2_assignmovingavg_readvariableop_resource-batch_normalization_2/AssignMovingAvg/mul:z:05^batch_normalization_2/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02'
%batch_normalization_2/AssignMovingAvgБ
-batch_normalization_2/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2/
-batch_normalization_2/AssignMovingAvg_1/decayь
6batch_normalization_2/AssignMovingAvg_1/ReadVariableOpReadVariableOp?batch_normalization_2_assignmovingavg_1_readvariableop_resource*
_output_shapes	
:ђ*
dtype028
6batch_normalization_2/AssignMovingAvg_1/ReadVariableOpщ
+batch_normalization_2/AssignMovingAvg_1/subSub>batch_normalization_2/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_2/moments/Squeeze_1:output:0*
T0*
_output_shapes	
:ђ2-
+batch_normalization_2/AssignMovingAvg_1/sub­
+batch_normalization_2/AssignMovingAvg_1/mulMul/batch_normalization_2/AssignMovingAvg_1/sub:z:06batch_normalization_2/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:ђ2-
+batch_normalization_2/AssignMovingAvg_1/mulи
'batch_normalization_2/AssignMovingAvg_1AssignSubVariableOp?batch_normalization_2_assignmovingavg_1_readvariableop_resource/batch_normalization_2/AssignMovingAvg_1/mul:z:07^batch_normalization_2/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02)
'batch_normalization_2/AssignMovingAvg_1Њ
%batch_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2'
%batch_normalization_2/batchnorm/add/y█
#batch_normalization_2/batchnorm/addAddV20batch_normalization_2/moments/Squeeze_1:output:0.batch_normalization_2/batchnorm/add/y:output:0*
T0*
_output_shapes	
:ђ2%
#batch_normalization_2/batchnorm/addд
%batch_normalization_2/batchnorm/RsqrtRsqrt'batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes	
:ђ2'
%batch_normalization_2/batchnorm/Rsqrtр
2batch_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes	
:ђ*
dtype024
2batch_normalization_2/batchnorm/mul/ReadVariableOpя
#batch_normalization_2/batchnorm/mulMul)batch_normalization_2/batchnorm/Rsqrt:y:0:batch_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:ђ2%
#batch_normalization_2/batchnorm/mulО
%batch_normalization_2/batchnorm/mul_1Mul stream_0_conv_3/BiasAdd:output:0'batch_normalization_2/batchnorm/mul:z:0*
T0*,
_output_shapes
:         }ђ2'
%batch_normalization_2/batchnorm/mul_1н
%batch_normalization_2/batchnorm/mul_2Mul.batch_normalization_2/moments/Squeeze:output:0'batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes	
:ђ2'
%batch_normalization_2/batchnorm/mul_2Н
.batch_normalization_2/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_2_batchnorm_readvariableop_resource*
_output_shapes	
:ђ*
dtype020
.batch_normalization_2/batchnorm/ReadVariableOp┌
#batch_normalization_2/batchnorm/subSub6batch_normalization_2/batchnorm/ReadVariableOp:value:0)batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:ђ2%
#batch_normalization_2/batchnorm/subР
%batch_normalization_2/batchnorm/add_1AddV2)batch_normalization_2/batchnorm/mul_1:z:0'batch_normalization_2/batchnorm/sub:z:0*
T0*,
_output_shapes
:         }ђ2'
%batch_normalization_2/batchnorm/add_1љ
activation_2/ReluRelu)batch_normalization_2/batchnorm/add_1:z:0*
T0*,
_output_shapes
:         }ђ2
activation_2/ReluЃ
stream_0_drop_3/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
stream_0_drop_3/dropout/Const┴
stream_0_drop_3/dropout/MulMulactivation_2/Relu:activations:0&stream_0_drop_3/dropout/Const:output:0*
T0*,
_output_shapes
:         }ђ2
stream_0_drop_3/dropout/MulЇ
stream_0_drop_3/dropout/ShapeShapeactivation_2/Relu:activations:0*
T0*
_output_shapes
:2
stream_0_drop_3/dropout/Shapeё
4stream_0_drop_3/dropout/random_uniform/RandomUniformRandomUniform&stream_0_drop_3/dropout/Shape:output:0*
T0*,
_output_shapes
:         }ђ*
dtype0*
seedи*
seed2и26
4stream_0_drop_3/dropout/random_uniform/RandomUniformЋ
&stream_0_drop_3/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2(
&stream_0_drop_3/dropout/GreaterEqual/yЃ
$stream_0_drop_3/dropout/GreaterEqualGreaterEqual=stream_0_drop_3/dropout/random_uniform/RandomUniform:output:0/stream_0_drop_3/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:         }ђ2&
$stream_0_drop_3/dropout/GreaterEqual┤
stream_0_drop_3/dropout/CastCast(stream_0_drop_3/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:         }ђ2
stream_0_drop_3/dropout/Cast┐
stream_0_drop_3/dropout/Mul_1Mulstream_0_drop_3/dropout/Mul:z:0 stream_0_drop_3/dropout/Cast:y:0*
T0*,
_output_shapes
:         }ђ2
stream_0_drop_3/dropout/Mul_1ц
/global_average_pooling1d/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :21
/global_average_pooling1d/Mean/reduction_indicesо
global_average_pooling1d/MeanMean!stream_0_drop_3/dropout/Mul_1:z:08global_average_pooling1d/Mean/reduction_indices:output:0*
T0*(
_output_shapes
:         ђ2
global_average_pooling1d/MeanЃ
dense_1_dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?2
dense_1_dropout/dropout/Const─
dense_1_dropout/dropout/MulMul&global_average_pooling1d/Mean:output:0&dense_1_dropout/dropout/Const:output:0*
T0*(
_output_shapes
:         ђ2
dense_1_dropout/dropout/Mulћ
dense_1_dropout/dropout/ShapeShape&global_average_pooling1d/Mean:output:0*
T0*
_output_shapes
:2
dense_1_dropout/dropout/ShapeЫ
4dense_1_dropout/dropout/random_uniform/RandomUniformRandomUniform&dense_1_dropout/dropout/Shape:output:0*
T0*(
_output_shapes
:         ђ*
dtype0*
seedи26
4dense_1_dropout/dropout/random_uniform/RandomUniformЋ
&dense_1_dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>2(
&dense_1_dropout/dropout/GreaterEqual/y 
$dense_1_dropout/dropout/GreaterEqualGreaterEqual=dense_1_dropout/dropout/random_uniform/RandomUniform:output:0/dense_1_dropout/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         ђ2&
$dense_1_dropout/dropout/GreaterEqual░
dense_1_dropout/dropout/CastCast(dense_1_dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         ђ2
dense_1_dropout/dropout/Cast╗
dense_1_dropout/dropout/Mul_1Muldense_1_dropout/dropout/Mul:z:0 dense_1_dropout/dropout/Cast:y:0*
T0*(
_output_shapes
:         ђ2
dense_1_dropout/dropout/Mul_1д
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	ђT*
dtype02
dense_1/MatMul/ReadVariableOpд
dense_1/MatMulMatMul!dense_1_dropout/dropout/Mul_1:z:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         T2
dense_1/MatMulц
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:T*
dtype02 
dense_1/BiasAdd/ReadVariableOpА
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         T2
dense_1/BiasAddХ
4batch_normalization_3/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 26
4batch_normalization_3/moments/mean/reduction_indicesс
"batch_normalization_3/moments/meanMeandense_1/BiasAdd:output:0=batch_normalization_3/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:T*
	keep_dims(2$
"batch_normalization_3/moments/meanЙ
*batch_normalization_3/moments/StopGradientStopGradient+batch_normalization_3/moments/mean:output:0*
T0*
_output_shapes

:T2,
*batch_normalization_3/moments/StopGradientЭ
/batch_normalization_3/moments/SquaredDifferenceSquaredDifferencedense_1/BiasAdd:output:03batch_normalization_3/moments/StopGradient:output:0*
T0*'
_output_shapes
:         T21
/batch_normalization_3/moments/SquaredDifferenceЙ
8batch_normalization_3/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2:
8batch_normalization_3/moments/variance/reduction_indicesі
&batch_normalization_3/moments/varianceMean3batch_normalization_3/moments/SquaredDifference:z:0Abatch_normalization_3/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:T*
	keep_dims(2(
&batch_normalization_3/moments/variance┬
%batch_normalization_3/moments/SqueezeSqueeze+batch_normalization_3/moments/mean:output:0*
T0*
_output_shapes
:T*
squeeze_dims
 2'
%batch_normalization_3/moments/Squeeze╩
'batch_normalization_3/moments/Squeeze_1Squeeze/batch_normalization_3/moments/variance:output:0*
T0*
_output_shapes
:T*
squeeze_dims
 2)
'batch_normalization_3/moments/Squeeze_1Ъ
+batch_normalization_3/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2-
+batch_normalization_3/AssignMovingAvg/decayТ
4batch_normalization_3/AssignMovingAvg/ReadVariableOpReadVariableOp=batch_normalization_3_assignmovingavg_readvariableop_resource*
_output_shapes
:T*
dtype026
4batch_normalization_3/AssignMovingAvg/ReadVariableOp­
)batch_normalization_3/AssignMovingAvg/subSub<batch_normalization_3/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_3/moments/Squeeze:output:0*
T0*
_output_shapes
:T2+
)batch_normalization_3/AssignMovingAvg/subу
)batch_normalization_3/AssignMovingAvg/mulMul-batch_normalization_3/AssignMovingAvg/sub:z:04batch_normalization_3/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:T2+
)batch_normalization_3/AssignMovingAvg/mulГ
%batch_normalization_3/AssignMovingAvgAssignSubVariableOp=batch_normalization_3_assignmovingavg_readvariableop_resource-batch_normalization_3/AssignMovingAvg/mul:z:05^batch_normalization_3/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02'
%batch_normalization_3/AssignMovingAvgБ
-batch_normalization_3/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2/
-batch_normalization_3/AssignMovingAvg_1/decayВ
6batch_normalization_3/AssignMovingAvg_1/ReadVariableOpReadVariableOp?batch_normalization_3_assignmovingavg_1_readvariableop_resource*
_output_shapes
:T*
dtype028
6batch_normalization_3/AssignMovingAvg_1/ReadVariableOpЭ
+batch_normalization_3/AssignMovingAvg_1/subSub>batch_normalization_3/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_3/moments/Squeeze_1:output:0*
T0*
_output_shapes
:T2-
+batch_normalization_3/AssignMovingAvg_1/sub№
+batch_normalization_3/AssignMovingAvg_1/mulMul/batch_normalization_3/AssignMovingAvg_1/sub:z:06batch_normalization_3/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:T2-
+batch_normalization_3/AssignMovingAvg_1/mulи
'batch_normalization_3/AssignMovingAvg_1AssignSubVariableOp?batch_normalization_3_assignmovingavg_1_readvariableop_resource/batch_normalization_3/AssignMovingAvg_1/mul:z:07^batch_normalization_3/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02)
'batch_normalization_3/AssignMovingAvg_1Њ
%batch_normalization_3/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2'
%batch_normalization_3/batchnorm/add/y┌
#batch_normalization_3/batchnorm/addAddV20batch_normalization_3/moments/Squeeze_1:output:0.batch_normalization_3/batchnorm/add/y:output:0*
T0*
_output_shapes
:T2%
#batch_normalization_3/batchnorm/addЦ
%batch_normalization_3/batchnorm/RsqrtRsqrt'batch_normalization_3/batchnorm/add:z:0*
T0*
_output_shapes
:T2'
%batch_normalization_3/batchnorm/RsqrtЯ
2batch_normalization_3/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_3_batchnorm_mul_readvariableop_resource*
_output_shapes
:T*
dtype024
2batch_normalization_3/batchnorm/mul/ReadVariableOpП
#batch_normalization_3/batchnorm/mulMul)batch_normalization_3/batchnorm/Rsqrt:y:0:batch_normalization_3/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:T2%
#batch_normalization_3/batchnorm/mul╩
%batch_normalization_3/batchnorm/mul_1Muldense_1/BiasAdd:output:0'batch_normalization_3/batchnorm/mul:z:0*
T0*'
_output_shapes
:         T2'
%batch_normalization_3/batchnorm/mul_1М
%batch_normalization_3/batchnorm/mul_2Mul.batch_normalization_3/moments/Squeeze:output:0'batch_normalization_3/batchnorm/mul:z:0*
T0*
_output_shapes
:T2'
%batch_normalization_3/batchnorm/mul_2н
.batch_normalization_3/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_3_batchnorm_readvariableop_resource*
_output_shapes
:T*
dtype020
.batch_normalization_3/batchnorm/ReadVariableOp┘
#batch_normalization_3/batchnorm/subSub6batch_normalization_3/batchnorm/ReadVariableOp:value:0)batch_normalization_3/batchnorm/mul_2:z:0*
T0*
_output_shapes
:T2%
#batch_normalization_3/batchnorm/subП
%batch_normalization_3/batchnorm/add_1AddV2)batch_normalization_3/batchnorm/mul_1:z:0'batch_normalization_3/batchnorm/sub:z:0*
T0*'
_output_shapes
:         T2'
%batch_normalization_3/batchnorm/add_1а
dense_activation_1/SigmoidSigmoid)batch_normalization_3/batchnorm/add_1:z:0*
T0*'
_output_shapes
:         T2
dense_activation_1/SigmoidЬ
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp;stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype027
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp├
&stream_0_conv_1/kernel/Regularizer/AbsAbs=stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_0_conv_1/kernel/Regularizer/AbsЕ
(stream_0_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_0_conv_1/kernel/Regularizer/ConstО
&stream_0_conv_1/kernel/Regularizer/SumSum*stream_0_conv_1/kernel/Regularizer/Abs:y:01stream_0_conv_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/SumЎ
(stream_0_conv_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2*
(stream_0_conv_1/kernel/Regularizer/mul/x▄
&stream_0_conv_1/kernel/Regularizer/mulMul1stream_0_conv_1/kernel/Regularizer/mul/x:output:0/stream_0_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/mulш
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;stream_0_conv_2_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:@ђ*
dtype02:
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOpл
)stream_0_conv_2/kernel/Regularizer/SquareSquare@stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*#
_output_shapes
:@ђ2+
)stream_0_conv_2/kernel/Regularizer/SquareЕ
(stream_0_conv_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_0_conv_2/kernel/Regularizer/Const┌
&stream_0_conv_2/kernel/Regularizer/SumSum-stream_0_conv_2/kernel/Regularizer/Square:y:01stream_0_conv_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_2/kernel/Regularizer/SumЎ
(stream_0_conv_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2*
(stream_0_conv_2/kernel/Regularizer/mul/x▄
&stream_0_conv_2/kernel/Regularizer/mulMul1stream_0_conv_2/kernel/Regularizer/mul/x:output:0/stream_0_conv_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_2/kernel/Regularizer/mul­
5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp;stream_0_conv_3_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:ђђ*
dtype027
5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp┼
&stream_0_conv_3/kernel/Regularizer/AbsAbs=stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*$
_output_shapes
:ђђ2(
&stream_0_conv_3/kernel/Regularizer/AbsЕ
(stream_0_conv_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_0_conv_3/kernel/Regularizer/ConstО
&stream_0_conv_3/kernel/Regularizer/SumSum*stream_0_conv_3/kernel/Regularizer/Abs:y:01stream_0_conv_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_3/kernel/Regularizer/SumЎ
(stream_0_conv_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2*
(stream_0_conv_3/kernel/Regularizer/mul/x▄
&stream_0_conv_3/kernel/Regularizer/mulMul1stream_0_conv_3/kernel/Regularizer/mul/x:output:0/stream_0_conv_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_3/kernel/Regularizer/mulк
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	ђT*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOpе
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	ђT2 
dense_1/kernel/Regularizer/AbsЋ
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_1/kernel/Regularizer/Constи
dense_1/kernel/Regularizer/SumSum"dense_1/kernel/Regularizer/Abs:y:0)dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/SumЅ
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2"
 dense_1/kernel/Regularizer/mul/x╝
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/muly
IdentityIdentitydense_activation_1/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:         T2

Identityў
NoOpNoOp$^batch_normalization/AssignMovingAvg3^batch_normalization/AssignMovingAvg/ReadVariableOp&^batch_normalization/AssignMovingAvg_15^batch_normalization/AssignMovingAvg_1/ReadVariableOp-^batch_normalization/batchnorm/ReadVariableOp1^batch_normalization/batchnorm/mul/ReadVariableOp&^batch_normalization_1/AssignMovingAvg5^batch_normalization_1/AssignMovingAvg/ReadVariableOp(^batch_normalization_1/AssignMovingAvg_17^batch_normalization_1/AssignMovingAvg_1/ReadVariableOp/^batch_normalization_1/batchnorm/ReadVariableOp3^batch_normalization_1/batchnorm/mul/ReadVariableOp&^batch_normalization_2/AssignMovingAvg5^batch_normalization_2/AssignMovingAvg/ReadVariableOp(^batch_normalization_2/AssignMovingAvg_17^batch_normalization_2/AssignMovingAvg_1/ReadVariableOp/^batch_normalization_2/batchnorm/ReadVariableOp3^batch_normalization_2/batchnorm/mul/ReadVariableOp&^batch_normalization_3/AssignMovingAvg5^batch_normalization_3/AssignMovingAvg/ReadVariableOp(^batch_normalization_3/AssignMovingAvg_17^batch_normalization_3/AssignMovingAvg_1/ReadVariableOp/^batch_normalization_3/batchnorm/ReadVariableOp3^batch_normalization_3/batchnorm/mul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp.^dense_1/kernel/Regularizer/Abs/ReadVariableOp'^stream_0_conv_1/BiasAdd/ReadVariableOp3^stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp'^stream_0_conv_2/BiasAdd/ReadVariableOp3^stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOp9^stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp'^stream_0_conv_3/BiasAdd/ReadVariableOp3^stream_0_conv_3/conv1d/ExpandDims_1/ReadVariableOp6^stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:         }: : : : : : : : : : : : : : : : : : : : : : : : 2J
#batch_normalization/AssignMovingAvg#batch_normalization/AssignMovingAvg2h
2batch_normalization/AssignMovingAvg/ReadVariableOp2batch_normalization/AssignMovingAvg/ReadVariableOp2N
%batch_normalization/AssignMovingAvg_1%batch_normalization/AssignMovingAvg_12l
4batch_normalization/AssignMovingAvg_1/ReadVariableOp4batch_normalization/AssignMovingAvg_1/ReadVariableOp2\
,batch_normalization/batchnorm/ReadVariableOp,batch_normalization/batchnorm/ReadVariableOp2d
0batch_normalization/batchnorm/mul/ReadVariableOp0batch_normalization/batchnorm/mul/ReadVariableOp2N
%batch_normalization_1/AssignMovingAvg%batch_normalization_1/AssignMovingAvg2l
4batch_normalization_1/AssignMovingAvg/ReadVariableOp4batch_normalization_1/AssignMovingAvg/ReadVariableOp2R
'batch_normalization_1/AssignMovingAvg_1'batch_normalization_1/AssignMovingAvg_12p
6batch_normalization_1/AssignMovingAvg_1/ReadVariableOp6batch_normalization_1/AssignMovingAvg_1/ReadVariableOp2`
.batch_normalization_1/batchnorm/ReadVariableOp.batch_normalization_1/batchnorm/ReadVariableOp2h
2batch_normalization_1/batchnorm/mul/ReadVariableOp2batch_normalization_1/batchnorm/mul/ReadVariableOp2N
%batch_normalization_2/AssignMovingAvg%batch_normalization_2/AssignMovingAvg2l
4batch_normalization_2/AssignMovingAvg/ReadVariableOp4batch_normalization_2/AssignMovingAvg/ReadVariableOp2R
'batch_normalization_2/AssignMovingAvg_1'batch_normalization_2/AssignMovingAvg_12p
6batch_normalization_2/AssignMovingAvg_1/ReadVariableOp6batch_normalization_2/AssignMovingAvg_1/ReadVariableOp2`
.batch_normalization_2/batchnorm/ReadVariableOp.batch_normalization_2/batchnorm/ReadVariableOp2h
2batch_normalization_2/batchnorm/mul/ReadVariableOp2batch_normalization_2/batchnorm/mul/ReadVariableOp2N
%batch_normalization_3/AssignMovingAvg%batch_normalization_3/AssignMovingAvg2l
4batch_normalization_3/AssignMovingAvg/ReadVariableOp4batch_normalization_3/AssignMovingAvg/ReadVariableOp2R
'batch_normalization_3/AssignMovingAvg_1'batch_normalization_3/AssignMovingAvg_12p
6batch_normalization_3/AssignMovingAvg_1/ReadVariableOp6batch_normalization_3/AssignMovingAvg_1/ReadVariableOp2`
.batch_normalization_3/batchnorm/ReadVariableOp.batch_normalization_3/batchnorm/ReadVariableOp2h
2batch_normalization_3/batchnorm/mul/ReadVariableOp2batch_normalization_3/batchnorm/mul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2^
-dense_1/kernel/Regularizer/Abs/ReadVariableOp-dense_1/kernel/Regularizer/Abs/ReadVariableOp2P
&stream_0_conv_1/BiasAdd/ReadVariableOp&stream_0_conv_1/BiasAdd/ReadVariableOp2h
2stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp2stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp2n
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp2P
&stream_0_conv_2/BiasAdd/ReadVariableOp&stream_0_conv_2/BiasAdd/ReadVariableOp2h
2stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOp2stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOp2t
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp2P
&stream_0_conv_3/BiasAdd/ReadVariableOp&stream_0_conv_3/BiasAdd/ReadVariableOp2h
2stream_0_conv_3/conv1d/ExpandDims_1/ReadVariableOp2stream_0_conv_3/conv1d/ExpandDims_1/ReadVariableOp2n
5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp:S O
+
_output_shapes
:         }
 
_user_specified_nameinputs
сH
Г	
@__inference_model_layer_call_and_return_conditional_losses_26180
left_inputs
right_inputs%
basemodel_26080:@
basemodel_26082:@
basemodel_26084:@
basemodel_26086:@
basemodel_26088:@
basemodel_26090:@&
basemodel_26092:@ђ
basemodel_26094:	ђ
basemodel_26096:	ђ
basemodel_26098:	ђ
basemodel_26100:	ђ
basemodel_26102:	ђ'
basemodel_26104:ђђ
basemodel_26106:	ђ
basemodel_26108:	ђ
basemodel_26110:	ђ
basemodel_26112:	ђ
basemodel_26114:	ђ"
basemodel_26116:	ђT
basemodel_26118:T
basemodel_26120:T
basemodel_26122:T
basemodel_26124:T
basemodel_26126:T
identityѕб!basemodel/StatefulPartitionedCallб#basemodel/StatefulPartitionedCall_1б-dense_1/kernel/Regularizer/Abs/ReadVariableOpб5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpб8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOpб5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOpИ
!basemodel/StatefulPartitionedCallStatefulPartitionedCallleft_inputsbasemodel_26080basemodel_26082basemodel_26084basemodel_26086basemodel_26088basemodel_26090basemodel_26092basemodel_26094basemodel_26096basemodel_26098basemodel_26100basemodel_26102basemodel_26104basemodel_26106basemodel_26108basemodel_26110basemodel_26112basemodel_26114basemodel_26116basemodel_26118basemodel_26120basemodel_26122basemodel_26124basemodel_26126*$
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         T*2
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *M
fHRF
D__inference_basemodel_layer_call_and_return_conditional_losses_256552#
!basemodel/StatefulPartitionedCallр
#basemodel/StatefulPartitionedCall_1StatefulPartitionedCallright_inputsbasemodel_26080basemodel_26082basemodel_26084basemodel_26086basemodel_26088basemodel_26090basemodel_26092basemodel_26094basemodel_26096basemodel_26098basemodel_26100basemodel_26102basemodel_26104basemodel_26106basemodel_26108basemodel_26110basemodel_26112basemodel_26114basemodel_26116basemodel_26118basemodel_26120basemodel_26122basemodel_26124basemodel_26126"^basemodel/StatefulPartitionedCall*$
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         T*2
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *M
fHRF
D__inference_basemodel_layer_call_and_return_conditional_losses_256552%
#basemodel/StatefulPartitionedCall_1Е
distance/PartitionedCallPartitionedCall*basemodel/StatefulPartitionedCall:output:0,basemodel/StatefulPartitionedCall_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *L
fGRE
C__inference_distance_layer_call_and_return_conditional_losses_253642
distance/PartitionedCall┬
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpbasemodel_26080*"
_output_shapes
:@*
dtype027
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp├
&stream_0_conv_1/kernel/Regularizer/AbsAbs=stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_0_conv_1/kernel/Regularizer/AbsЕ
(stream_0_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_0_conv_1/kernel/Regularizer/ConstО
&stream_0_conv_1/kernel/Regularizer/SumSum*stream_0_conv_1/kernel/Regularizer/Abs:y:01stream_0_conv_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/SumЎ
(stream_0_conv_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2*
(stream_0_conv_1/kernel/Regularizer/mul/x▄
&stream_0_conv_1/kernel/Regularizer/mulMul1stream_0_conv_1/kernel/Regularizer/mul/x:output:0/stream_0_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/mul╔
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpbasemodel_26092*#
_output_shapes
:@ђ*
dtype02:
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOpл
)stream_0_conv_2/kernel/Regularizer/SquareSquare@stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*#
_output_shapes
:@ђ2+
)stream_0_conv_2/kernel/Regularizer/SquareЕ
(stream_0_conv_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_0_conv_2/kernel/Regularizer/Const┌
&stream_0_conv_2/kernel/Regularizer/SumSum-stream_0_conv_2/kernel/Regularizer/Square:y:01stream_0_conv_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_2/kernel/Regularizer/SumЎ
(stream_0_conv_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2*
(stream_0_conv_2/kernel/Regularizer/mul/x▄
&stream_0_conv_2/kernel/Regularizer/mulMul1stream_0_conv_2/kernel/Regularizer/mul/x:output:0/stream_0_conv_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_2/kernel/Regularizer/mul─
5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpbasemodel_26104*$
_output_shapes
:ђђ*
dtype027
5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp┼
&stream_0_conv_3/kernel/Regularizer/AbsAbs=stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*$
_output_shapes
:ђђ2(
&stream_0_conv_3/kernel/Regularizer/AbsЕ
(stream_0_conv_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_0_conv_3/kernel/Regularizer/ConstО
&stream_0_conv_3/kernel/Regularizer/SumSum*stream_0_conv_3/kernel/Regularizer/Abs:y:01stream_0_conv_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_3/kernel/Regularizer/SumЎ
(stream_0_conv_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2*
(stream_0_conv_3/kernel/Regularizer/mul/x▄
&stream_0_conv_3/kernel/Regularizer/mulMul1stream_0_conv_3/kernel/Regularizer/mul/x:output:0/stream_0_conv_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_3/kernel/Regularizer/mul»
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpbasemodel_26116*
_output_shapes
:	ђT*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOpе
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	ђT2 
dense_1/kernel/Regularizer/AbsЋ
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_1/kernel/Regularizer/Constи
dense_1/kernel/Regularizer/SumSum"dense_1/kernel/Regularizer/Abs:y:0)dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/SumЅ
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2"
 dense_1/kernel/Regularizer/mul/x╝
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/mul|
IdentityIdentity!distance/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         2

Identityз
NoOpNoOp"^basemodel/StatefulPartitionedCall$^basemodel/StatefulPartitionedCall_1.^dense_1/kernel/Regularizer/Abs/ReadVariableOp6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp9^stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp6^stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*q
_input_shapes`
^:         }:         }: : : : : : : : : : : : : : : : : : : : : : : : 2F
!basemodel/StatefulPartitionedCall!basemodel/StatefulPartitionedCall2J
#basemodel/StatefulPartitionedCall_1#basemodel/StatefulPartitionedCall_12^
-dense_1/kernel/Regularizer/Abs/ReadVariableOp-dense_1/kernel/Regularizer/Abs/ReadVariableOp2n
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp2t
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp2n
5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp:X T
+
_output_shapes
:         }
%
_user_specified_nameleft_inputs:YU
+
_output_shapes
:         }
&
_user_specified_nameright_inputs
№
н
5__inference_batch_normalization_2_layer_call_fn_28639

inputs
unknown:	ђ
	unknown_0:	ђ
	unknown_1:	ђ
	unknown_2:	ђ
identityѕбStatefulPartitionedCallа
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         }ђ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Y
fTRR
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_243442
StatefulPartitionedCallђ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:         }ђ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         }ђ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:         }ђ
 
_user_specified_nameinputs
▓
ы
)__inference_basemodel_layer_call_fn_27219
inputs_0
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:@ 
	unknown_5:@ђ
	unknown_6:	ђ
	unknown_7:	ђ
	unknown_8:	ђ
	unknown_9:	ђ

unknown_10:	ђ"

unknown_11:ђђ

unknown_12:	ђ

unknown_13:	ђ

unknown_14:	ђ

unknown_15:	ђ

unknown_16:	ђ

unknown_17:	ђT

unknown_18:T

unknown_19:T

unknown_20:T

unknown_21:T

unknown_22:T
identityѕбStatefulPartitionedCallц
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22*$
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         T*:
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8ѓ *M
fHRF
D__inference_basemodel_layer_call_and_return_conditional_losses_251772
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         T2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:         }: : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
+
_output_shapes
:         }
"
_user_specified_name
inputs/0
І
h
J__inference_stream_0_drop_3_layer_call_and_return_conditional_losses_24092

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:         }ђ2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:         }ђ2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         }ђ:T P
,
_output_shapes
:         }ђ
 
_user_specified_nameinputs
■*
у
N__inference_batch_normalization_layer_call_and_return_conditional_losses_28281

inputs5
'assignmovingavg_readvariableop_resource:@7
)assignmovingavg_1_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@/
!batchnorm_readvariableop_resource:@
identityѕбAssignMovingAvgбAssignMovingAvg/ReadVariableOpбAssignMovingAvg_1б AssignMovingAvg_1/ReadVariableOpбbatchnorm/ReadVariableOpбbatchnorm/mul/ReadVariableOpЉ
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indicesЊ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(2
moments/meanђ
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:@2
moments/StopGradientе
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*+
_output_shapes
:         }@2
moments/SquaredDifferenceЎ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indicesХ
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(2
moments/varianceЂ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 2
moments/SqueezeЅ
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 2
moments/Squeeze_1s
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2
AssignMovingAvg/decayц
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOpў
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg/subЈ
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg/mul┐
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvgw
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2
AssignMovingAvg_1/decayф
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOpа
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg_1/subЌ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg_1/mul╔
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1g
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2
batchnorm/add/yѓ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@2
batchnorm/Rsqrtъ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOpЁ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2
batchnorm/mulz
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:         }@2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:@2
batchnorm/mul_2њ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOpЂ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2
batchnorm/subЅ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*+
_output_shapes
:         }@2
batchnorm/add_1r
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*+
_output_shapes
:         }@2

IdentityЫ
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         }@: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:S O
+
_output_shapes
:         }@
 
_user_specified_nameinputs
└	
o
C__inference_distance_layer_call_and_return_conditional_losses_28058
inputs_0
inputs_1
identityW
subSubinputs_0inputs_1*
T0*'
_output_shapes
:         T2
subU
SquareSquaresub:z:0*
T0*'
_output_shapes
:         T2
Squarey
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
         2
Sum/reduction_indicesђ
SumSum
Square:y:0Sum/reduction_indices:output:0*
T0*'
_output_shapes
:         *
	keep_dims(2
SumS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
Constm
MaximumMaximumSum:output:0Const:output:0*
T0*'
_output_shapes
:         2	
MaximumS
SqrtSqrtMaximum:z:0*
T0*'
_output_shapes
:         2
Sqrt\
IdentityIdentitySqrt:y:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:         T:         T:Q M
'
_output_shapes
:         T
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:         T
"
_user_specified_name
inputs/1
ЁБ
њ
D__inference_basemodel_layer_call_and_return_conditional_losses_27647

inputsQ
;stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource:@=
/stream_0_conv_1_biasadd_readvariableop_resource:@I
;batch_normalization_assignmovingavg_readvariableop_resource:@K
=batch_normalization_assignmovingavg_1_readvariableop_resource:@G
9batch_normalization_batchnorm_mul_readvariableop_resource:@C
5batch_normalization_batchnorm_readvariableop_resource:@R
;stream_0_conv_2_conv1d_expanddims_1_readvariableop_resource:@ђ>
/stream_0_conv_2_biasadd_readvariableop_resource:	ђL
=batch_normalization_1_assignmovingavg_readvariableop_resource:	ђN
?batch_normalization_1_assignmovingavg_1_readvariableop_resource:	ђJ
;batch_normalization_1_batchnorm_mul_readvariableop_resource:	ђF
7batch_normalization_1_batchnorm_readvariableop_resource:	ђS
;stream_0_conv_3_conv1d_expanddims_1_readvariableop_resource:ђђ>
/stream_0_conv_3_biasadd_readvariableop_resource:	ђL
=batch_normalization_2_assignmovingavg_readvariableop_resource:	ђN
?batch_normalization_2_assignmovingavg_1_readvariableop_resource:	ђJ
;batch_normalization_2_batchnorm_mul_readvariableop_resource:	ђF
7batch_normalization_2_batchnorm_readvariableop_resource:	ђ9
&dense_1_matmul_readvariableop_resource:	ђT5
'dense_1_biasadd_readvariableop_resource:TK
=batch_normalization_3_assignmovingavg_readvariableop_resource:TM
?batch_normalization_3_assignmovingavg_1_readvariableop_resource:TI
;batch_normalization_3_batchnorm_mul_readvariableop_resource:TE
7batch_normalization_3_batchnorm_readvariableop_resource:T
identityѕб#batch_normalization/AssignMovingAvgб2batch_normalization/AssignMovingAvg/ReadVariableOpб%batch_normalization/AssignMovingAvg_1б4batch_normalization/AssignMovingAvg_1/ReadVariableOpб,batch_normalization/batchnorm/ReadVariableOpб0batch_normalization/batchnorm/mul/ReadVariableOpб%batch_normalization_1/AssignMovingAvgб4batch_normalization_1/AssignMovingAvg/ReadVariableOpб'batch_normalization_1/AssignMovingAvg_1б6batch_normalization_1/AssignMovingAvg_1/ReadVariableOpб.batch_normalization_1/batchnorm/ReadVariableOpб2batch_normalization_1/batchnorm/mul/ReadVariableOpб%batch_normalization_2/AssignMovingAvgб4batch_normalization_2/AssignMovingAvg/ReadVariableOpб'batch_normalization_2/AssignMovingAvg_1б6batch_normalization_2/AssignMovingAvg_1/ReadVariableOpб.batch_normalization_2/batchnorm/ReadVariableOpб2batch_normalization_2/batchnorm/mul/ReadVariableOpб%batch_normalization_3/AssignMovingAvgб4batch_normalization_3/AssignMovingAvg/ReadVariableOpб'batch_normalization_3/AssignMovingAvg_1б6batch_normalization_3/AssignMovingAvg_1/ReadVariableOpб.batch_normalization_3/batchnorm/ReadVariableOpб2batch_normalization_3/batchnorm/mul/ReadVariableOpбdense_1/BiasAdd/ReadVariableOpбdense_1/MatMul/ReadVariableOpб-dense_1/kernel/Regularizer/Abs/ReadVariableOpб&stream_0_conv_1/BiasAdd/ReadVariableOpб2stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOpб5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpб&stream_0_conv_2/BiasAdd/ReadVariableOpб2stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOpб8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOpб&stream_0_conv_3/BiasAdd/ReadVariableOpб2stream_0_conv_3/conv1d/ExpandDims_1/ReadVariableOpб5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOpІ
!stream_0_input_drop/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?2#
!stream_0_input_drop/dropout/Const│
stream_0_input_drop/dropout/MulMulinputs*stream_0_input_drop/dropout/Const:output:0*
T0*+
_output_shapes
:         }2!
stream_0_input_drop/dropout/Mul|
!stream_0_input_drop/dropout/ShapeShapeinputs*
T0*
_output_shapes
:2#
!stream_0_input_drop/dropout/ShapeЈ
8stream_0_input_drop/dropout/random_uniform/RandomUniformRandomUniform*stream_0_input_drop/dropout/Shape:output:0*
T0*+
_output_shapes
:         }*
dtype0*
seedи*
seed2и2:
8stream_0_input_drop/dropout/random_uniform/RandomUniformЮ
*stream_0_input_drop/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>2,
*stream_0_input_drop/dropout/GreaterEqual/yњ
(stream_0_input_drop/dropout/GreaterEqualGreaterEqualAstream_0_input_drop/dropout/random_uniform/RandomUniform:output:03stream_0_input_drop/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:         }2*
(stream_0_input_drop/dropout/GreaterEqual┐
 stream_0_input_drop/dropout/CastCast,stream_0_input_drop/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:         }2"
 stream_0_input_drop/dropout/Cast╬
!stream_0_input_drop/dropout/Mul_1Mul#stream_0_input_drop/dropout/Mul:z:0$stream_0_input_drop/dropout/Cast:y:0*
T0*+
_output_shapes
:         }2#
!stream_0_input_drop/dropout/Mul_1Ў
%stream_0_conv_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§        2'
%stream_0_conv_1/conv1d/ExpandDims/dimт
!stream_0_conv_1/conv1d/ExpandDims
ExpandDims%stream_0_input_drop/dropout/Mul_1:z:0.stream_0_conv_1/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:         }2#
!stream_0_conv_1/conv1d/ExpandDimsУ
2stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp;stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype024
2stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOpћ
'stream_0_conv_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2)
'stream_0_conv_1/conv1d/ExpandDims_1/dimэ
#stream_0_conv_1/conv1d/ExpandDims_1
ExpandDims:stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp:value:00stream_0_conv_1/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@2%
#stream_0_conv_1/conv1d/ExpandDims_1Ш
stream_0_conv_1/conv1dConv2D*stream_0_conv_1/conv1d/ExpandDims:output:0,stream_0_conv_1/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:         }@*
paddingSAME*
strides
2
stream_0_conv_1/conv1d┬
stream_0_conv_1/conv1d/SqueezeSqueezestream_0_conv_1/conv1d:output:0*
T0*+
_output_shapes
:         }@*
squeeze_dims

§        2 
stream_0_conv_1/conv1d/Squeeze╝
&stream_0_conv_1/BiasAdd/ReadVariableOpReadVariableOp/stream_0_conv_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02(
&stream_0_conv_1/BiasAdd/ReadVariableOp╠
stream_0_conv_1/BiasAddBiasAdd'stream_0_conv_1/conv1d/Squeeze:output:0.stream_0_conv_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         }@2
stream_0_conv_1/BiasAdd╣
2batch_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       24
2batch_normalization/moments/mean/reduction_indicesж
 batch_normalization/moments/meanMean stream_0_conv_1/BiasAdd:output:0;batch_normalization/moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(2"
 batch_normalization/moments/mean╝
(batch_normalization/moments/StopGradientStopGradient)batch_normalization/moments/mean:output:0*
T0*"
_output_shapes
:@2*
(batch_normalization/moments/StopGradient■
-batch_normalization/moments/SquaredDifferenceSquaredDifference stream_0_conv_1/BiasAdd:output:01batch_normalization/moments/StopGradient:output:0*
T0*+
_output_shapes
:         }@2/
-batch_normalization/moments/SquaredDifference┴
6batch_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       28
6batch_normalization/moments/variance/reduction_indicesє
$batch_normalization/moments/varianceMean1batch_normalization/moments/SquaredDifference:z:0?batch_normalization/moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(2&
$batch_normalization/moments/varianceй
#batch_normalization/moments/SqueezeSqueeze)batch_normalization/moments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 2%
#batch_normalization/moments/Squeeze┼
%batch_normalization/moments/Squeeze_1Squeeze-batch_normalization/moments/variance:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 2'
%batch_normalization/moments/Squeeze_1Џ
)batch_normalization/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2+
)batch_normalization/AssignMovingAvg/decayЯ
2batch_normalization/AssignMovingAvg/ReadVariableOpReadVariableOp;batch_normalization_assignmovingavg_readvariableop_resource*
_output_shapes
:@*
dtype024
2batch_normalization/AssignMovingAvg/ReadVariableOpУ
'batch_normalization/AssignMovingAvg/subSub:batch_normalization/AssignMovingAvg/ReadVariableOp:value:0,batch_normalization/moments/Squeeze:output:0*
T0*
_output_shapes
:@2)
'batch_normalization/AssignMovingAvg/sub▀
'batch_normalization/AssignMovingAvg/mulMul+batch_normalization/AssignMovingAvg/sub:z:02batch_normalization/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:@2)
'batch_normalization/AssignMovingAvg/mulБ
#batch_normalization/AssignMovingAvgAssignSubVariableOp;batch_normalization_assignmovingavg_readvariableop_resource+batch_normalization/AssignMovingAvg/mul:z:03^batch_normalization/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02%
#batch_normalization/AssignMovingAvgЪ
+batch_normalization/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2-
+batch_normalization/AssignMovingAvg_1/decayТ
4batch_normalization/AssignMovingAvg_1/ReadVariableOpReadVariableOp=batch_normalization_assignmovingavg_1_readvariableop_resource*
_output_shapes
:@*
dtype026
4batch_normalization/AssignMovingAvg_1/ReadVariableOp­
)batch_normalization/AssignMovingAvg_1/subSub<batch_normalization/AssignMovingAvg_1/ReadVariableOp:value:0.batch_normalization/moments/Squeeze_1:output:0*
T0*
_output_shapes
:@2+
)batch_normalization/AssignMovingAvg_1/subу
)batch_normalization/AssignMovingAvg_1/mulMul-batch_normalization/AssignMovingAvg_1/sub:z:04batch_normalization/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:@2+
)batch_normalization/AssignMovingAvg_1/mulГ
%batch_normalization/AssignMovingAvg_1AssignSubVariableOp=batch_normalization_assignmovingavg_1_readvariableop_resource-batch_normalization/AssignMovingAvg_1/mul:z:05^batch_normalization/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02'
%batch_normalization/AssignMovingAvg_1Ј
#batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2%
#batch_normalization/batchnorm/add/yм
!batch_normalization/batchnorm/addAddV2.batch_normalization/moments/Squeeze_1:output:0,batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes
:@2#
!batch_normalization/batchnorm/addЪ
#batch_normalization/batchnorm/RsqrtRsqrt%batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes
:@2%
#batch_normalization/batchnorm/Rsqrt┌
0batch_normalization/batchnorm/mul/ReadVariableOpReadVariableOp9batch_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype022
0batch_normalization/batchnorm/mul/ReadVariableOpН
!batch_normalization/batchnorm/mulMul'batch_normalization/batchnorm/Rsqrt:y:08batch_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2#
!batch_normalization/batchnorm/mulл
#batch_normalization/batchnorm/mul_1Mul stream_0_conv_1/BiasAdd:output:0%batch_normalization/batchnorm/mul:z:0*
T0*+
_output_shapes
:         }@2%
#batch_normalization/batchnorm/mul_1╦
#batch_normalization/batchnorm/mul_2Mul,batch_normalization/moments/Squeeze:output:0%batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes
:@2%
#batch_normalization/batchnorm/mul_2╬
,batch_normalization/batchnorm/ReadVariableOpReadVariableOp5batch_normalization_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02.
,batch_normalization/batchnorm/ReadVariableOpЛ
!batch_normalization/batchnorm/subSub4batch_normalization/batchnorm/ReadVariableOp:value:0'batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2#
!batch_normalization/batchnorm/sub┘
#batch_normalization/batchnorm/add_1AddV2'batch_normalization/batchnorm/mul_1:z:0%batch_normalization/batchnorm/sub:z:0*
T0*+
_output_shapes
:         }@2%
#batch_normalization/batchnorm/add_1Ѕ
activation/ReluRelu'batch_normalization/batchnorm/add_1:z:0*
T0*+
_output_shapes
:         }@2
activation/ReluЃ
stream_0_drop_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n█Х?2
stream_0_drop_1/dropout/ConstЙ
stream_0_drop_1/dropout/MulMulactivation/Relu:activations:0&stream_0_drop_1/dropout/Const:output:0*
T0*+
_output_shapes
:         }@2
stream_0_drop_1/dropout/MulІ
stream_0_drop_1/dropout/ShapeShapeactivation/Relu:activations:0*
T0*
_output_shapes
:2
stream_0_drop_1/dropout/ShapeЃ
4stream_0_drop_1/dropout/random_uniform/RandomUniformRandomUniform&stream_0_drop_1/dropout/Shape:output:0*
T0*+
_output_shapes
:         }@*
dtype0*
seedи*
seed2и26
4stream_0_drop_1/dropout/random_uniform/RandomUniformЋ
&stream_0_drop_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *џЎЎ>2(
&stream_0_drop_1/dropout/GreaterEqual/yѓ
$stream_0_drop_1/dropout/GreaterEqualGreaterEqual=stream_0_drop_1/dropout/random_uniform/RandomUniform:output:0/stream_0_drop_1/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:         }@2&
$stream_0_drop_1/dropout/GreaterEqual│
stream_0_drop_1/dropout/CastCast(stream_0_drop_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:         }@2
stream_0_drop_1/dropout/CastЙ
stream_0_drop_1/dropout/Mul_1Mulstream_0_drop_1/dropout/Mul:z:0 stream_0_drop_1/dropout/Cast:y:0*
T0*+
_output_shapes
:         }@2
stream_0_drop_1/dropout/Mul_1Ў
%stream_0_conv_2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§        2'
%stream_0_conv_2/conv1d/ExpandDims/dimр
!stream_0_conv_2/conv1d/ExpandDims
ExpandDims!stream_0_drop_1/dropout/Mul_1:z:0.stream_0_conv_2/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:         }@2#
!stream_0_conv_2/conv1d/ExpandDimsж
2stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp;stream_0_conv_2_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:@ђ*
dtype024
2stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOpћ
'stream_0_conv_2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2)
'stream_0_conv_2/conv1d/ExpandDims_1/dimЭ
#stream_0_conv_2/conv1d/ExpandDims_1
ExpandDims:stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOp:value:00stream_0_conv_2/conv1d/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:@ђ2%
#stream_0_conv_2/conv1d/ExpandDims_1э
stream_0_conv_2/conv1dConv2D*stream_0_conv_2/conv1d/ExpandDims:output:0,stream_0_conv_2/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:         }ђ*
paddingSAME*
strides
2
stream_0_conv_2/conv1d├
stream_0_conv_2/conv1d/SqueezeSqueezestream_0_conv_2/conv1d:output:0*
T0*,
_output_shapes
:         }ђ*
squeeze_dims

§        2 
stream_0_conv_2/conv1d/Squeezeй
&stream_0_conv_2/BiasAdd/ReadVariableOpReadVariableOp/stream_0_conv_2_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02(
&stream_0_conv_2/BiasAdd/ReadVariableOp═
stream_0_conv_2/BiasAddBiasAdd'stream_0_conv_2/conv1d/Squeeze:output:0.stream_0_conv_2/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:         }ђ2
stream_0_conv_2/BiasAddй
4batch_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       26
4batch_normalization_1/moments/mean/reduction_indices­
"batch_normalization_1/moments/meanMean stream_0_conv_2/BiasAdd:output:0=batch_normalization_1/moments/mean/reduction_indices:output:0*
T0*#
_output_shapes
:ђ*
	keep_dims(2$
"batch_normalization_1/moments/mean├
*batch_normalization_1/moments/StopGradientStopGradient+batch_normalization_1/moments/mean:output:0*
T0*#
_output_shapes
:ђ2,
*batch_normalization_1/moments/StopGradientЁ
/batch_normalization_1/moments/SquaredDifferenceSquaredDifference stream_0_conv_2/BiasAdd:output:03batch_normalization_1/moments/StopGradient:output:0*
T0*,
_output_shapes
:         }ђ21
/batch_normalization_1/moments/SquaredDifference┼
8batch_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2:
8batch_normalization_1/moments/variance/reduction_indicesЈ
&batch_normalization_1/moments/varianceMean3batch_normalization_1/moments/SquaredDifference:z:0Abatch_normalization_1/moments/variance/reduction_indices:output:0*
T0*#
_output_shapes
:ђ*
	keep_dims(2(
&batch_normalization_1/moments/variance─
%batch_normalization_1/moments/SqueezeSqueeze+batch_normalization_1/moments/mean:output:0*
T0*
_output_shapes	
:ђ*
squeeze_dims
 2'
%batch_normalization_1/moments/Squeeze╠
'batch_normalization_1/moments/Squeeze_1Squeeze/batch_normalization_1/moments/variance:output:0*
T0*
_output_shapes	
:ђ*
squeeze_dims
 2)
'batch_normalization_1/moments/Squeeze_1Ъ
+batch_normalization_1/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2-
+batch_normalization_1/AssignMovingAvg/decayу
4batch_normalization_1/AssignMovingAvg/ReadVariableOpReadVariableOp=batch_normalization_1_assignmovingavg_readvariableop_resource*
_output_shapes	
:ђ*
dtype026
4batch_normalization_1/AssignMovingAvg/ReadVariableOpы
)batch_normalization_1/AssignMovingAvg/subSub<batch_normalization_1/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_1/moments/Squeeze:output:0*
T0*
_output_shapes	
:ђ2+
)batch_normalization_1/AssignMovingAvg/subУ
)batch_normalization_1/AssignMovingAvg/mulMul-batch_normalization_1/AssignMovingAvg/sub:z:04batch_normalization_1/AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:ђ2+
)batch_normalization_1/AssignMovingAvg/mulГ
%batch_normalization_1/AssignMovingAvgAssignSubVariableOp=batch_normalization_1_assignmovingavg_readvariableop_resource-batch_normalization_1/AssignMovingAvg/mul:z:05^batch_normalization_1/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02'
%batch_normalization_1/AssignMovingAvgБ
-batch_normalization_1/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2/
-batch_normalization_1/AssignMovingAvg_1/decayь
6batch_normalization_1/AssignMovingAvg_1/ReadVariableOpReadVariableOp?batch_normalization_1_assignmovingavg_1_readvariableop_resource*
_output_shapes	
:ђ*
dtype028
6batch_normalization_1/AssignMovingAvg_1/ReadVariableOpщ
+batch_normalization_1/AssignMovingAvg_1/subSub>batch_normalization_1/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_1/moments/Squeeze_1:output:0*
T0*
_output_shapes	
:ђ2-
+batch_normalization_1/AssignMovingAvg_1/sub­
+batch_normalization_1/AssignMovingAvg_1/mulMul/batch_normalization_1/AssignMovingAvg_1/sub:z:06batch_normalization_1/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:ђ2-
+batch_normalization_1/AssignMovingAvg_1/mulи
'batch_normalization_1/AssignMovingAvg_1AssignSubVariableOp?batch_normalization_1_assignmovingavg_1_readvariableop_resource/batch_normalization_1/AssignMovingAvg_1/mul:z:07^batch_normalization_1/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02)
'batch_normalization_1/AssignMovingAvg_1Њ
%batch_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2'
%batch_normalization_1/batchnorm/add/y█
#batch_normalization_1/batchnorm/addAddV20batch_normalization_1/moments/Squeeze_1:output:0.batch_normalization_1/batchnorm/add/y:output:0*
T0*
_output_shapes	
:ђ2%
#batch_normalization_1/batchnorm/addд
%batch_normalization_1/batchnorm/RsqrtRsqrt'batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes	
:ђ2'
%batch_normalization_1/batchnorm/Rsqrtр
2batch_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes	
:ђ*
dtype024
2batch_normalization_1/batchnorm/mul/ReadVariableOpя
#batch_normalization_1/batchnorm/mulMul)batch_normalization_1/batchnorm/Rsqrt:y:0:batch_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:ђ2%
#batch_normalization_1/batchnorm/mulО
%batch_normalization_1/batchnorm/mul_1Mul stream_0_conv_2/BiasAdd:output:0'batch_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:         }ђ2'
%batch_normalization_1/batchnorm/mul_1н
%batch_normalization_1/batchnorm/mul_2Mul.batch_normalization_1/moments/Squeeze:output:0'batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes	
:ђ2'
%batch_normalization_1/batchnorm/mul_2Н
.batch_normalization_1/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_1_batchnorm_readvariableop_resource*
_output_shapes	
:ђ*
dtype020
.batch_normalization_1/batchnorm/ReadVariableOp┌
#batch_normalization_1/batchnorm/subSub6batch_normalization_1/batchnorm/ReadVariableOp:value:0)batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:ђ2%
#batch_normalization_1/batchnorm/subР
%batch_normalization_1/batchnorm/add_1AddV2)batch_normalization_1/batchnorm/mul_1:z:0'batch_normalization_1/batchnorm/sub:z:0*
T0*,
_output_shapes
:         }ђ2'
%batch_normalization_1/batchnorm/add_1љ
activation_1/ReluRelu)batch_normalization_1/batchnorm/add_1:z:0*
T0*,
_output_shapes
:         }ђ2
activation_1/ReluЃ
stream_0_drop_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUН?2
stream_0_drop_2/dropout/Const┴
stream_0_drop_2/dropout/MulMulactivation_1/Relu:activations:0&stream_0_drop_2/dropout/Const:output:0*
T0*,
_output_shapes
:         }ђ2
stream_0_drop_2/dropout/MulЇ
stream_0_drop_2/dropout/ShapeShapeactivation_1/Relu:activations:0*
T0*
_output_shapes
:2
stream_0_drop_2/dropout/Shapeё
4stream_0_drop_2/dropout/random_uniform/RandomUniformRandomUniform&stream_0_drop_2/dropout/Shape:output:0*
T0*,
_output_shapes
:         }ђ*
dtype0*
seedи*
seed2и26
4stream_0_drop_2/dropout/random_uniform/RandomUniformЋ
&stream_0_drop_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠>2(
&stream_0_drop_2/dropout/GreaterEqual/yЃ
$stream_0_drop_2/dropout/GreaterEqualGreaterEqual=stream_0_drop_2/dropout/random_uniform/RandomUniform:output:0/stream_0_drop_2/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:         }ђ2&
$stream_0_drop_2/dropout/GreaterEqual┤
stream_0_drop_2/dropout/CastCast(stream_0_drop_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:         }ђ2
stream_0_drop_2/dropout/Cast┐
stream_0_drop_2/dropout/Mul_1Mulstream_0_drop_2/dropout/Mul:z:0 stream_0_drop_2/dropout/Cast:y:0*
T0*,
_output_shapes
:         }ђ2
stream_0_drop_2/dropout/Mul_1Ў
%stream_0_conv_3/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§        2'
%stream_0_conv_3/conv1d/ExpandDims/dimР
!stream_0_conv_3/conv1d/ExpandDims
ExpandDims!stream_0_drop_2/dropout/Mul_1:z:0.stream_0_conv_3/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         }ђ2#
!stream_0_conv_3/conv1d/ExpandDimsЖ
2stream_0_conv_3/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp;stream_0_conv_3_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:ђђ*
dtype024
2stream_0_conv_3/conv1d/ExpandDims_1/ReadVariableOpћ
'stream_0_conv_3/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2)
'stream_0_conv_3/conv1d/ExpandDims_1/dimщ
#stream_0_conv_3/conv1d/ExpandDims_1
ExpandDims:stream_0_conv_3/conv1d/ExpandDims_1/ReadVariableOp:value:00stream_0_conv_3/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:ђђ2%
#stream_0_conv_3/conv1d/ExpandDims_1э
stream_0_conv_3/conv1dConv2D*stream_0_conv_3/conv1d/ExpandDims:output:0,stream_0_conv_3/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:         }ђ*
paddingSAME*
strides
2
stream_0_conv_3/conv1d├
stream_0_conv_3/conv1d/SqueezeSqueezestream_0_conv_3/conv1d:output:0*
T0*,
_output_shapes
:         }ђ*
squeeze_dims

§        2 
stream_0_conv_3/conv1d/Squeezeй
&stream_0_conv_3/BiasAdd/ReadVariableOpReadVariableOp/stream_0_conv_3_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02(
&stream_0_conv_3/BiasAdd/ReadVariableOp═
stream_0_conv_3/BiasAddBiasAdd'stream_0_conv_3/conv1d/Squeeze:output:0.stream_0_conv_3/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:         }ђ2
stream_0_conv_3/BiasAddй
4batch_normalization_2/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       26
4batch_normalization_2/moments/mean/reduction_indices­
"batch_normalization_2/moments/meanMean stream_0_conv_3/BiasAdd:output:0=batch_normalization_2/moments/mean/reduction_indices:output:0*
T0*#
_output_shapes
:ђ*
	keep_dims(2$
"batch_normalization_2/moments/mean├
*batch_normalization_2/moments/StopGradientStopGradient+batch_normalization_2/moments/mean:output:0*
T0*#
_output_shapes
:ђ2,
*batch_normalization_2/moments/StopGradientЁ
/batch_normalization_2/moments/SquaredDifferenceSquaredDifference stream_0_conv_3/BiasAdd:output:03batch_normalization_2/moments/StopGradient:output:0*
T0*,
_output_shapes
:         }ђ21
/batch_normalization_2/moments/SquaredDifference┼
8batch_normalization_2/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2:
8batch_normalization_2/moments/variance/reduction_indicesЈ
&batch_normalization_2/moments/varianceMean3batch_normalization_2/moments/SquaredDifference:z:0Abatch_normalization_2/moments/variance/reduction_indices:output:0*
T0*#
_output_shapes
:ђ*
	keep_dims(2(
&batch_normalization_2/moments/variance─
%batch_normalization_2/moments/SqueezeSqueeze+batch_normalization_2/moments/mean:output:0*
T0*
_output_shapes	
:ђ*
squeeze_dims
 2'
%batch_normalization_2/moments/Squeeze╠
'batch_normalization_2/moments/Squeeze_1Squeeze/batch_normalization_2/moments/variance:output:0*
T0*
_output_shapes	
:ђ*
squeeze_dims
 2)
'batch_normalization_2/moments/Squeeze_1Ъ
+batch_normalization_2/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2-
+batch_normalization_2/AssignMovingAvg/decayу
4batch_normalization_2/AssignMovingAvg/ReadVariableOpReadVariableOp=batch_normalization_2_assignmovingavg_readvariableop_resource*
_output_shapes	
:ђ*
dtype026
4batch_normalization_2/AssignMovingAvg/ReadVariableOpы
)batch_normalization_2/AssignMovingAvg/subSub<batch_normalization_2/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_2/moments/Squeeze:output:0*
T0*
_output_shapes	
:ђ2+
)batch_normalization_2/AssignMovingAvg/subУ
)batch_normalization_2/AssignMovingAvg/mulMul-batch_normalization_2/AssignMovingAvg/sub:z:04batch_normalization_2/AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:ђ2+
)batch_normalization_2/AssignMovingAvg/mulГ
%batch_normalization_2/AssignMovingAvgAssignSubVariableOp=batch_normalization_2_assignmovingavg_readvariableop_resource-batch_normalization_2/AssignMovingAvg/mul:z:05^batch_normalization_2/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02'
%batch_normalization_2/AssignMovingAvgБ
-batch_normalization_2/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2/
-batch_normalization_2/AssignMovingAvg_1/decayь
6batch_normalization_2/AssignMovingAvg_1/ReadVariableOpReadVariableOp?batch_normalization_2_assignmovingavg_1_readvariableop_resource*
_output_shapes	
:ђ*
dtype028
6batch_normalization_2/AssignMovingAvg_1/ReadVariableOpщ
+batch_normalization_2/AssignMovingAvg_1/subSub>batch_normalization_2/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_2/moments/Squeeze_1:output:0*
T0*
_output_shapes	
:ђ2-
+batch_normalization_2/AssignMovingAvg_1/sub­
+batch_normalization_2/AssignMovingAvg_1/mulMul/batch_normalization_2/AssignMovingAvg_1/sub:z:06batch_normalization_2/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:ђ2-
+batch_normalization_2/AssignMovingAvg_1/mulи
'batch_normalization_2/AssignMovingAvg_1AssignSubVariableOp?batch_normalization_2_assignmovingavg_1_readvariableop_resource/batch_normalization_2/AssignMovingAvg_1/mul:z:07^batch_normalization_2/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02)
'batch_normalization_2/AssignMovingAvg_1Њ
%batch_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2'
%batch_normalization_2/batchnorm/add/y█
#batch_normalization_2/batchnorm/addAddV20batch_normalization_2/moments/Squeeze_1:output:0.batch_normalization_2/batchnorm/add/y:output:0*
T0*
_output_shapes	
:ђ2%
#batch_normalization_2/batchnorm/addд
%batch_normalization_2/batchnorm/RsqrtRsqrt'batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes	
:ђ2'
%batch_normalization_2/batchnorm/Rsqrtр
2batch_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes	
:ђ*
dtype024
2batch_normalization_2/batchnorm/mul/ReadVariableOpя
#batch_normalization_2/batchnorm/mulMul)batch_normalization_2/batchnorm/Rsqrt:y:0:batch_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:ђ2%
#batch_normalization_2/batchnorm/mulО
%batch_normalization_2/batchnorm/mul_1Mul stream_0_conv_3/BiasAdd:output:0'batch_normalization_2/batchnorm/mul:z:0*
T0*,
_output_shapes
:         }ђ2'
%batch_normalization_2/batchnorm/mul_1н
%batch_normalization_2/batchnorm/mul_2Mul.batch_normalization_2/moments/Squeeze:output:0'batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes	
:ђ2'
%batch_normalization_2/batchnorm/mul_2Н
.batch_normalization_2/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_2_batchnorm_readvariableop_resource*
_output_shapes	
:ђ*
dtype020
.batch_normalization_2/batchnorm/ReadVariableOp┌
#batch_normalization_2/batchnorm/subSub6batch_normalization_2/batchnorm/ReadVariableOp:value:0)batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:ђ2%
#batch_normalization_2/batchnorm/subР
%batch_normalization_2/batchnorm/add_1AddV2)batch_normalization_2/batchnorm/mul_1:z:0'batch_normalization_2/batchnorm/sub:z:0*
T0*,
_output_shapes
:         }ђ2'
%batch_normalization_2/batchnorm/add_1љ
activation_2/ReluRelu)batch_normalization_2/batchnorm/add_1:z:0*
T0*,
_output_shapes
:         }ђ2
activation_2/ReluЃ
stream_0_drop_3/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
stream_0_drop_3/dropout/Const┴
stream_0_drop_3/dropout/MulMulactivation_2/Relu:activations:0&stream_0_drop_3/dropout/Const:output:0*
T0*,
_output_shapes
:         }ђ2
stream_0_drop_3/dropout/MulЇ
stream_0_drop_3/dropout/ShapeShapeactivation_2/Relu:activations:0*
T0*
_output_shapes
:2
stream_0_drop_3/dropout/Shapeё
4stream_0_drop_3/dropout/random_uniform/RandomUniformRandomUniform&stream_0_drop_3/dropout/Shape:output:0*
T0*,
_output_shapes
:         }ђ*
dtype0*
seedи*
seed2и26
4stream_0_drop_3/dropout/random_uniform/RandomUniformЋ
&stream_0_drop_3/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2(
&stream_0_drop_3/dropout/GreaterEqual/yЃ
$stream_0_drop_3/dropout/GreaterEqualGreaterEqual=stream_0_drop_3/dropout/random_uniform/RandomUniform:output:0/stream_0_drop_3/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:         }ђ2&
$stream_0_drop_3/dropout/GreaterEqual┤
stream_0_drop_3/dropout/CastCast(stream_0_drop_3/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:         }ђ2
stream_0_drop_3/dropout/Cast┐
stream_0_drop_3/dropout/Mul_1Mulstream_0_drop_3/dropout/Mul:z:0 stream_0_drop_3/dropout/Cast:y:0*
T0*,
_output_shapes
:         }ђ2
stream_0_drop_3/dropout/Mul_1ц
/global_average_pooling1d/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :21
/global_average_pooling1d/Mean/reduction_indicesо
global_average_pooling1d/MeanMean!stream_0_drop_3/dropout/Mul_1:z:08global_average_pooling1d/Mean/reduction_indices:output:0*
T0*(
_output_shapes
:         ђ2
global_average_pooling1d/MeanЃ
dense_1_dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?2
dense_1_dropout/dropout/Const─
dense_1_dropout/dropout/MulMul&global_average_pooling1d/Mean:output:0&dense_1_dropout/dropout/Const:output:0*
T0*(
_output_shapes
:         ђ2
dense_1_dropout/dropout/Mulћ
dense_1_dropout/dropout/ShapeShape&global_average_pooling1d/Mean:output:0*
T0*
_output_shapes
:2
dense_1_dropout/dropout/ShapeЫ
4dense_1_dropout/dropout/random_uniform/RandomUniformRandomUniform&dense_1_dropout/dropout/Shape:output:0*
T0*(
_output_shapes
:         ђ*
dtype0*
seedи26
4dense_1_dropout/dropout/random_uniform/RandomUniformЋ
&dense_1_dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>2(
&dense_1_dropout/dropout/GreaterEqual/y 
$dense_1_dropout/dropout/GreaterEqualGreaterEqual=dense_1_dropout/dropout/random_uniform/RandomUniform:output:0/dense_1_dropout/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         ђ2&
$dense_1_dropout/dropout/GreaterEqual░
dense_1_dropout/dropout/CastCast(dense_1_dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         ђ2
dense_1_dropout/dropout/Cast╗
dense_1_dropout/dropout/Mul_1Muldense_1_dropout/dropout/Mul:z:0 dense_1_dropout/dropout/Cast:y:0*
T0*(
_output_shapes
:         ђ2
dense_1_dropout/dropout/Mul_1д
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	ђT*
dtype02
dense_1/MatMul/ReadVariableOpд
dense_1/MatMulMatMul!dense_1_dropout/dropout/Mul_1:z:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         T2
dense_1/MatMulц
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:T*
dtype02 
dense_1/BiasAdd/ReadVariableOpА
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         T2
dense_1/BiasAddХ
4batch_normalization_3/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 26
4batch_normalization_3/moments/mean/reduction_indicesс
"batch_normalization_3/moments/meanMeandense_1/BiasAdd:output:0=batch_normalization_3/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:T*
	keep_dims(2$
"batch_normalization_3/moments/meanЙ
*batch_normalization_3/moments/StopGradientStopGradient+batch_normalization_3/moments/mean:output:0*
T0*
_output_shapes

:T2,
*batch_normalization_3/moments/StopGradientЭ
/batch_normalization_3/moments/SquaredDifferenceSquaredDifferencedense_1/BiasAdd:output:03batch_normalization_3/moments/StopGradient:output:0*
T0*'
_output_shapes
:         T21
/batch_normalization_3/moments/SquaredDifferenceЙ
8batch_normalization_3/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2:
8batch_normalization_3/moments/variance/reduction_indicesі
&batch_normalization_3/moments/varianceMean3batch_normalization_3/moments/SquaredDifference:z:0Abatch_normalization_3/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:T*
	keep_dims(2(
&batch_normalization_3/moments/variance┬
%batch_normalization_3/moments/SqueezeSqueeze+batch_normalization_3/moments/mean:output:0*
T0*
_output_shapes
:T*
squeeze_dims
 2'
%batch_normalization_3/moments/Squeeze╩
'batch_normalization_3/moments/Squeeze_1Squeeze/batch_normalization_3/moments/variance:output:0*
T0*
_output_shapes
:T*
squeeze_dims
 2)
'batch_normalization_3/moments/Squeeze_1Ъ
+batch_normalization_3/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2-
+batch_normalization_3/AssignMovingAvg/decayТ
4batch_normalization_3/AssignMovingAvg/ReadVariableOpReadVariableOp=batch_normalization_3_assignmovingavg_readvariableop_resource*
_output_shapes
:T*
dtype026
4batch_normalization_3/AssignMovingAvg/ReadVariableOp­
)batch_normalization_3/AssignMovingAvg/subSub<batch_normalization_3/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_3/moments/Squeeze:output:0*
T0*
_output_shapes
:T2+
)batch_normalization_3/AssignMovingAvg/subу
)batch_normalization_3/AssignMovingAvg/mulMul-batch_normalization_3/AssignMovingAvg/sub:z:04batch_normalization_3/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:T2+
)batch_normalization_3/AssignMovingAvg/mulГ
%batch_normalization_3/AssignMovingAvgAssignSubVariableOp=batch_normalization_3_assignmovingavg_readvariableop_resource-batch_normalization_3/AssignMovingAvg/mul:z:05^batch_normalization_3/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02'
%batch_normalization_3/AssignMovingAvgБ
-batch_normalization_3/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2/
-batch_normalization_3/AssignMovingAvg_1/decayВ
6batch_normalization_3/AssignMovingAvg_1/ReadVariableOpReadVariableOp?batch_normalization_3_assignmovingavg_1_readvariableop_resource*
_output_shapes
:T*
dtype028
6batch_normalization_3/AssignMovingAvg_1/ReadVariableOpЭ
+batch_normalization_3/AssignMovingAvg_1/subSub>batch_normalization_3/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_3/moments/Squeeze_1:output:0*
T0*
_output_shapes
:T2-
+batch_normalization_3/AssignMovingAvg_1/sub№
+batch_normalization_3/AssignMovingAvg_1/mulMul/batch_normalization_3/AssignMovingAvg_1/sub:z:06batch_normalization_3/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:T2-
+batch_normalization_3/AssignMovingAvg_1/mulи
'batch_normalization_3/AssignMovingAvg_1AssignSubVariableOp?batch_normalization_3_assignmovingavg_1_readvariableop_resource/batch_normalization_3/AssignMovingAvg_1/mul:z:07^batch_normalization_3/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02)
'batch_normalization_3/AssignMovingAvg_1Њ
%batch_normalization_3/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2'
%batch_normalization_3/batchnorm/add/y┌
#batch_normalization_3/batchnorm/addAddV20batch_normalization_3/moments/Squeeze_1:output:0.batch_normalization_3/batchnorm/add/y:output:0*
T0*
_output_shapes
:T2%
#batch_normalization_3/batchnorm/addЦ
%batch_normalization_3/batchnorm/RsqrtRsqrt'batch_normalization_3/batchnorm/add:z:0*
T0*
_output_shapes
:T2'
%batch_normalization_3/batchnorm/RsqrtЯ
2batch_normalization_3/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_3_batchnorm_mul_readvariableop_resource*
_output_shapes
:T*
dtype024
2batch_normalization_3/batchnorm/mul/ReadVariableOpП
#batch_normalization_3/batchnorm/mulMul)batch_normalization_3/batchnorm/Rsqrt:y:0:batch_normalization_3/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:T2%
#batch_normalization_3/batchnorm/mul╩
%batch_normalization_3/batchnorm/mul_1Muldense_1/BiasAdd:output:0'batch_normalization_3/batchnorm/mul:z:0*
T0*'
_output_shapes
:         T2'
%batch_normalization_3/batchnorm/mul_1М
%batch_normalization_3/batchnorm/mul_2Mul.batch_normalization_3/moments/Squeeze:output:0'batch_normalization_3/batchnorm/mul:z:0*
T0*
_output_shapes
:T2'
%batch_normalization_3/batchnorm/mul_2н
.batch_normalization_3/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_3_batchnorm_readvariableop_resource*
_output_shapes
:T*
dtype020
.batch_normalization_3/batchnorm/ReadVariableOp┘
#batch_normalization_3/batchnorm/subSub6batch_normalization_3/batchnorm/ReadVariableOp:value:0)batch_normalization_3/batchnorm/mul_2:z:0*
T0*
_output_shapes
:T2%
#batch_normalization_3/batchnorm/subП
%batch_normalization_3/batchnorm/add_1AddV2)batch_normalization_3/batchnorm/mul_1:z:0'batch_normalization_3/batchnorm/sub:z:0*
T0*'
_output_shapes
:         T2'
%batch_normalization_3/batchnorm/add_1а
dense_activation_1/SigmoidSigmoid)batch_normalization_3/batchnorm/add_1:z:0*
T0*'
_output_shapes
:         T2
dense_activation_1/SigmoidЬ
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp;stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype027
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp├
&stream_0_conv_1/kernel/Regularizer/AbsAbs=stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_0_conv_1/kernel/Regularizer/AbsЕ
(stream_0_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_0_conv_1/kernel/Regularizer/ConstО
&stream_0_conv_1/kernel/Regularizer/SumSum*stream_0_conv_1/kernel/Regularizer/Abs:y:01stream_0_conv_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/SumЎ
(stream_0_conv_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2*
(stream_0_conv_1/kernel/Regularizer/mul/x▄
&stream_0_conv_1/kernel/Regularizer/mulMul1stream_0_conv_1/kernel/Regularizer/mul/x:output:0/stream_0_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/mulш
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;stream_0_conv_2_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:@ђ*
dtype02:
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOpл
)stream_0_conv_2/kernel/Regularizer/SquareSquare@stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*#
_output_shapes
:@ђ2+
)stream_0_conv_2/kernel/Regularizer/SquareЕ
(stream_0_conv_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_0_conv_2/kernel/Regularizer/Const┌
&stream_0_conv_2/kernel/Regularizer/SumSum-stream_0_conv_2/kernel/Regularizer/Square:y:01stream_0_conv_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_2/kernel/Regularizer/SumЎ
(stream_0_conv_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2*
(stream_0_conv_2/kernel/Regularizer/mul/x▄
&stream_0_conv_2/kernel/Regularizer/mulMul1stream_0_conv_2/kernel/Regularizer/mul/x:output:0/stream_0_conv_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_2/kernel/Regularizer/mul­
5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp;stream_0_conv_3_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:ђђ*
dtype027
5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp┼
&stream_0_conv_3/kernel/Regularizer/AbsAbs=stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*$
_output_shapes
:ђђ2(
&stream_0_conv_3/kernel/Regularizer/AbsЕ
(stream_0_conv_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_0_conv_3/kernel/Regularizer/ConstО
&stream_0_conv_3/kernel/Regularizer/SumSum*stream_0_conv_3/kernel/Regularizer/Abs:y:01stream_0_conv_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_3/kernel/Regularizer/SumЎ
(stream_0_conv_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2*
(stream_0_conv_3/kernel/Regularizer/mul/x▄
&stream_0_conv_3/kernel/Regularizer/mulMul1stream_0_conv_3/kernel/Regularizer/mul/x:output:0/stream_0_conv_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_3/kernel/Regularizer/mulк
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	ђT*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOpе
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	ђT2 
dense_1/kernel/Regularizer/AbsЋ
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_1/kernel/Regularizer/Constи
dense_1/kernel/Regularizer/SumSum"dense_1/kernel/Regularizer/Abs:y:0)dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/SumЅ
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2"
 dense_1/kernel/Regularizer/mul/x╝
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/muly
IdentityIdentitydense_activation_1/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:         T2

Identityў
NoOpNoOp$^batch_normalization/AssignMovingAvg3^batch_normalization/AssignMovingAvg/ReadVariableOp&^batch_normalization/AssignMovingAvg_15^batch_normalization/AssignMovingAvg_1/ReadVariableOp-^batch_normalization/batchnorm/ReadVariableOp1^batch_normalization/batchnorm/mul/ReadVariableOp&^batch_normalization_1/AssignMovingAvg5^batch_normalization_1/AssignMovingAvg/ReadVariableOp(^batch_normalization_1/AssignMovingAvg_17^batch_normalization_1/AssignMovingAvg_1/ReadVariableOp/^batch_normalization_1/batchnorm/ReadVariableOp3^batch_normalization_1/batchnorm/mul/ReadVariableOp&^batch_normalization_2/AssignMovingAvg5^batch_normalization_2/AssignMovingAvg/ReadVariableOp(^batch_normalization_2/AssignMovingAvg_17^batch_normalization_2/AssignMovingAvg_1/ReadVariableOp/^batch_normalization_2/batchnorm/ReadVariableOp3^batch_normalization_2/batchnorm/mul/ReadVariableOp&^batch_normalization_3/AssignMovingAvg5^batch_normalization_3/AssignMovingAvg/ReadVariableOp(^batch_normalization_3/AssignMovingAvg_17^batch_normalization_3/AssignMovingAvg_1/ReadVariableOp/^batch_normalization_3/batchnorm/ReadVariableOp3^batch_normalization_3/batchnorm/mul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp.^dense_1/kernel/Regularizer/Abs/ReadVariableOp'^stream_0_conv_1/BiasAdd/ReadVariableOp3^stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp'^stream_0_conv_2/BiasAdd/ReadVariableOp3^stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOp9^stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp'^stream_0_conv_3/BiasAdd/ReadVariableOp3^stream_0_conv_3/conv1d/ExpandDims_1/ReadVariableOp6^stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:         }: : : : : : : : : : : : : : : : : : : : : : : : 2J
#batch_normalization/AssignMovingAvg#batch_normalization/AssignMovingAvg2h
2batch_normalization/AssignMovingAvg/ReadVariableOp2batch_normalization/AssignMovingAvg/ReadVariableOp2N
%batch_normalization/AssignMovingAvg_1%batch_normalization/AssignMovingAvg_12l
4batch_normalization/AssignMovingAvg_1/ReadVariableOp4batch_normalization/AssignMovingAvg_1/ReadVariableOp2\
,batch_normalization/batchnorm/ReadVariableOp,batch_normalization/batchnorm/ReadVariableOp2d
0batch_normalization/batchnorm/mul/ReadVariableOp0batch_normalization/batchnorm/mul/ReadVariableOp2N
%batch_normalization_1/AssignMovingAvg%batch_normalization_1/AssignMovingAvg2l
4batch_normalization_1/AssignMovingAvg/ReadVariableOp4batch_normalization_1/AssignMovingAvg/ReadVariableOp2R
'batch_normalization_1/AssignMovingAvg_1'batch_normalization_1/AssignMovingAvg_12p
6batch_normalization_1/AssignMovingAvg_1/ReadVariableOp6batch_normalization_1/AssignMovingAvg_1/ReadVariableOp2`
.batch_normalization_1/batchnorm/ReadVariableOp.batch_normalization_1/batchnorm/ReadVariableOp2h
2batch_normalization_1/batchnorm/mul/ReadVariableOp2batch_normalization_1/batchnorm/mul/ReadVariableOp2N
%batch_normalization_2/AssignMovingAvg%batch_normalization_2/AssignMovingAvg2l
4batch_normalization_2/AssignMovingAvg/ReadVariableOp4batch_normalization_2/AssignMovingAvg/ReadVariableOp2R
'batch_normalization_2/AssignMovingAvg_1'batch_normalization_2/AssignMovingAvg_12p
6batch_normalization_2/AssignMovingAvg_1/ReadVariableOp6batch_normalization_2/AssignMovingAvg_1/ReadVariableOp2`
.batch_normalization_2/batchnorm/ReadVariableOp.batch_normalization_2/batchnorm/ReadVariableOp2h
2batch_normalization_2/batchnorm/mul/ReadVariableOp2batch_normalization_2/batchnorm/mul/ReadVariableOp2N
%batch_normalization_3/AssignMovingAvg%batch_normalization_3/AssignMovingAvg2l
4batch_normalization_3/AssignMovingAvg/ReadVariableOp4batch_normalization_3/AssignMovingAvg/ReadVariableOp2R
'batch_normalization_3/AssignMovingAvg_1'batch_normalization_3/AssignMovingAvg_12p
6batch_normalization_3/AssignMovingAvg_1/ReadVariableOp6batch_normalization_3/AssignMovingAvg_1/ReadVariableOp2`
.batch_normalization_3/batchnorm/ReadVariableOp.batch_normalization_3/batchnorm/ReadVariableOp2h
2batch_normalization_3/batchnorm/mul/ReadVariableOp2batch_normalization_3/batchnorm/mul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2^
-dense_1/kernel/Regularizer/Abs/ReadVariableOp-dense_1/kernel/Regularizer/Abs/ReadVariableOp2P
&stream_0_conv_1/BiasAdd/ReadVariableOp&stream_0_conv_1/BiasAdd/ReadVariableOp2h
2stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp2stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp2n
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp2P
&stream_0_conv_2/BiasAdd/ReadVariableOp&stream_0_conv_2/BiasAdd/ReadVariableOp2h
2stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOp2stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOp2t
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp2P
&stream_0_conv_3/BiasAdd/ReadVariableOp&stream_0_conv_3/BiasAdd/ReadVariableOp2h
2stream_0_conv_3/conv1d/ExpandDims_1/ReadVariableOp2stream_0_conv_3/conv1d/ExpandDims_1/ReadVariableOp2n
5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp:S O
+
_output_shapes
:         }
 
_user_specified_nameinputs
┬
T
(__inference_distance_layer_call_fn_28028
inputs_0
inputs_1
identityЛ
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *L
fGRE
C__inference_distance_layer_call_and_return_conditional_losses_252642
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:         T:         T:Q M
'
_output_shapes
:         T
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:         T
"
_user_specified_name
inputs/1
╦*
ж
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_28944

inputs5
'assignmovingavg_readvariableop_resource:T7
)assignmovingavg_1_readvariableop_resource:T3
%batchnorm_mul_readvariableop_resource:T/
!batchnorm_readvariableop_resource:T
identityѕбAssignMovingAvgбAssignMovingAvg/ReadVariableOpбAssignMovingAvg_1б AssignMovingAvg_1/ReadVariableOpбbatchnorm/ReadVariableOpбbatchnorm/mul/ReadVariableOpі
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indicesЈ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:T*
	keep_dims(2
moments/mean|
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:T2
moments/StopGradientц
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:         T2
moments/SquaredDifferenceњ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices▓
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:T*
	keep_dims(2
moments/varianceђ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:T*
squeeze_dims
 2
moments/Squeezeѕ
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:T*
squeeze_dims
 2
moments/Squeeze_1s
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2
AssignMovingAvg/decayц
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:T*
dtype02 
AssignMovingAvg/ReadVariableOpў
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:T2
AssignMovingAvg/subЈ
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:T2
AssignMovingAvg/mul┐
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvgw
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2
AssignMovingAvg_1/decayф
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:T*
dtype02"
 AssignMovingAvg_1/ReadVariableOpа
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:T2
AssignMovingAvg_1/subЌ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:T2
AssignMovingAvg_1/mul╔
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1g
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2
batchnorm/add/yѓ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:T2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:T2
batchnorm/Rsqrtъ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:T*
dtype02
batchnorm/mul/ReadVariableOpЁ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:T2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:         T2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:T2
batchnorm/mul_2њ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:T*
dtype02
batchnorm/ReadVariableOpЂ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:T2
batchnorm/subЁ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:         T2
batchnorm/add_1n
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:         T2

IdentityЫ
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         T: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:         T
 
_user_specified_nameinputs
І
h
J__inference_stream_0_drop_2_layer_call_and_return_conditional_losses_24022

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:         }ђ2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:         }ђ2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         }ђ:T P
,
_output_shapes
:         }ђ
 
_user_specified_nameinputs
Ћ	
н
5__inference_batch_normalization_2_layer_call_fn_28600

inputs
unknown:	ђ
	unknown_0:	ђ
	unknown_1:	ђ
	unknown_2:	ђ
identityѕбStatefulPartitionedCallФ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:                  ђ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Y
fTRR
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_235462
StatefulPartitionedCallЅ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:                  ђ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):                  ђ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:                  ђ
 
_user_specified_nameinputs
еІ
я
D__inference_basemodel_layer_call_and_return_conditional_losses_25027
inputs_0+
stream_0_conv_1_24937:@#
stream_0_conv_1_24939:@'
batch_normalization_24942:@'
batch_normalization_24944:@'
batch_normalization_24946:@'
batch_normalization_24948:@,
stream_0_conv_2_24953:@ђ$
stream_0_conv_2_24955:	ђ*
batch_normalization_1_24958:	ђ*
batch_normalization_1_24960:	ђ*
batch_normalization_1_24962:	ђ*
batch_normalization_1_24964:	ђ-
stream_0_conv_3_24969:ђђ$
stream_0_conv_3_24971:	ђ*
batch_normalization_2_24974:	ђ*
batch_normalization_2_24976:	ђ*
batch_normalization_2_24978:	ђ*
batch_normalization_2_24980:	ђ 
dense_1_24987:	ђT
dense_1_24989:T)
batch_normalization_3_24992:T)
batch_normalization_3_24994:T)
batch_normalization_3_24996:T)
batch_normalization_3_24998:T
identityѕб+batch_normalization/StatefulPartitionedCallб-batch_normalization_1/StatefulPartitionedCallб-batch_normalization_2/StatefulPartitionedCallб-batch_normalization_3/StatefulPartitionedCallбdense_1/StatefulPartitionedCallб-dense_1/kernel/Regularizer/Abs/ReadVariableOpб'dense_1_dropout/StatefulPartitionedCallб'stream_0_conv_1/StatefulPartitionedCallб5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpб'stream_0_conv_2/StatefulPartitionedCallб8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOpб'stream_0_conv_3/StatefulPartitionedCallб5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOpб'stream_0_drop_1/StatefulPartitionedCallб'stream_0_drop_2/StatefulPartitionedCallб'stream_0_drop_3/StatefulPartitionedCallб+stream_0_input_drop/StatefulPartitionedCallЋ
+stream_0_input_drop/StatefulPartitionedCallStatefulPartitionedCallinputs_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         }* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *W
fRRP
N__inference_stream_0_input_drop_layer_call_and_return_conditional_losses_245832-
+stream_0_input_drop/StatefulPartitionedCallж
'stream_0_conv_1/StatefulPartitionedCallStatefulPartitionedCall4stream_0_input_drop/StatefulPartitionedCall:output:0stream_0_conv_1_24937stream_0_conv_1_24939*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         }@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *S
fNRL
J__inference_stream_0_conv_1_layer_call_and_return_conditional_losses_239052)
'stream_0_conv_1/StatefulPartitionedCall▒
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall0stream_0_conv_1/StatefulPartitionedCall:output:0batch_normalization_24942batch_normalization_24944batch_normalization_24946batch_normalization_24948*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         }@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_245422-
+batch_normalization/StatefulPartitionedCallј
activation/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         }@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_activation_layer_call_and_return_conditional_losses_239452
activation/PartitionedCallм
'stream_0_drop_1/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0,^stream_0_input_drop/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         }@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *S
fNRL
J__inference_stream_0_drop_1_layer_call_and_return_conditional_losses_244842)
'stream_0_drop_1/StatefulPartitionedCallТ
'stream_0_conv_2/StatefulPartitionedCallStatefulPartitionedCall0stream_0_drop_1/StatefulPartitionedCall:output:0stream_0_conv_2_24953stream_0_conv_2_24955*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         }ђ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *S
fNRL
J__inference_stream_0_conv_2_layer_call_and_return_conditional_losses_239752)
'stream_0_conv_2/StatefulPartitionedCall└
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall0stream_0_conv_2/StatefulPartitionedCall:output:0batch_normalization_1_24958batch_normalization_1_24960batch_normalization_1_24962batch_normalization_1_24964*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         }ђ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_244432/
-batch_normalization_1/StatefulPartitionedCallЌ
activation_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         }ђ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *P
fKRI
G__inference_activation_1_layer_call_and_return_conditional_losses_240152
activation_1/PartitionedCallЛ
'stream_0_drop_2/StatefulPartitionedCallStatefulPartitionedCall%activation_1/PartitionedCall:output:0(^stream_0_drop_1/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         }ђ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *S
fNRL
J__inference_stream_0_drop_2_layer_call_and_return_conditional_losses_243852)
'stream_0_drop_2/StatefulPartitionedCallТ
'stream_0_conv_3/StatefulPartitionedCallStatefulPartitionedCall0stream_0_drop_2/StatefulPartitionedCall:output:0stream_0_conv_3_24969stream_0_conv_3_24971*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         }ђ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *S
fNRL
J__inference_stream_0_conv_3_layer_call_and_return_conditional_losses_240452)
'stream_0_conv_3/StatefulPartitionedCall└
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall0stream_0_conv_3/StatefulPartitionedCall:output:0batch_normalization_2_24974batch_normalization_2_24976batch_normalization_2_24978batch_normalization_2_24980*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         }ђ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Y
fTRR
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_243442/
-batch_normalization_2/StatefulPartitionedCallЌ
activation_2/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         }ђ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *P
fKRI
G__inference_activation_2_layer_call_and_return_conditional_losses_240852
activation_2/PartitionedCallЛ
'stream_0_drop_3/StatefulPartitionedCallStatefulPartitionedCall%activation_2/PartitionedCall:output:0(^stream_0_drop_2/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         }ђ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *S
fNRL
J__inference_stream_0_drop_3_layer_call_and_return_conditional_losses_242862)
'stream_0_drop_3/StatefulPartitionedCall▒
(global_average_pooling1d/PartitionedCallPartitionedCall0stream_0_drop_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *\
fWRU
S__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_240992*
(global_average_pooling1d/PartitionedCall┘
'dense_1_dropout/StatefulPartitionedCallStatefulPartitionedCall1global_average_pooling1d/PartitionedCall:output:0(^stream_0_drop_3/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *S
fNRL
J__inference_dense_1_dropout_layer_call_and_return_conditional_losses_242582)
'dense_1_dropout/StatefulPartitionedCall╣
dense_1/StatefulPartitionedCallStatefulPartitionedCall0dense_1_dropout/StatefulPartitionedCall:output:0dense_1_24987dense_1_24989*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         T*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_241242!
dense_1/StatefulPartitionedCall│
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0batch_normalization_3_24992batch_normalization_3_24994batch_normalization_3_24996batch_normalization_3_24998*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         T*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Y
fTRR
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_237922/
-batch_normalization_3/StatefulPartitionedCallц
"dense_activation_1/PartitionedCallPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         T* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *V
fQRO
M__inference_dense_activation_1_layer_call_and_return_conditional_losses_241442$
"dense_activation_1/PartitionedCall╚
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpstream_0_conv_1_24937*"
_output_shapes
:@*
dtype027
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp├
&stream_0_conv_1/kernel/Regularizer/AbsAbs=stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_0_conv_1/kernel/Regularizer/AbsЕ
(stream_0_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_0_conv_1/kernel/Regularizer/ConstО
&stream_0_conv_1/kernel/Regularizer/SumSum*stream_0_conv_1/kernel/Regularizer/Abs:y:01stream_0_conv_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/SumЎ
(stream_0_conv_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2*
(stream_0_conv_1/kernel/Regularizer/mul/x▄
&stream_0_conv_1/kernel/Regularizer/mulMul1stream_0_conv_1/kernel/Regularizer/mul/x:output:0/stream_0_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/mul¤
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpstream_0_conv_2_24953*#
_output_shapes
:@ђ*
dtype02:
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOpл
)stream_0_conv_2/kernel/Regularizer/SquareSquare@stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*#
_output_shapes
:@ђ2+
)stream_0_conv_2/kernel/Regularizer/SquareЕ
(stream_0_conv_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_0_conv_2/kernel/Regularizer/Const┌
&stream_0_conv_2/kernel/Regularizer/SumSum-stream_0_conv_2/kernel/Regularizer/Square:y:01stream_0_conv_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_2/kernel/Regularizer/SumЎ
(stream_0_conv_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2*
(stream_0_conv_2/kernel/Regularizer/mul/x▄
&stream_0_conv_2/kernel/Regularizer/mulMul1stream_0_conv_2/kernel/Regularizer/mul/x:output:0/stream_0_conv_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_2/kernel/Regularizer/mul╩
5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpstream_0_conv_3_24969*$
_output_shapes
:ђђ*
dtype027
5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp┼
&stream_0_conv_3/kernel/Regularizer/AbsAbs=stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*$
_output_shapes
:ђђ2(
&stream_0_conv_3/kernel/Regularizer/AbsЕ
(stream_0_conv_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_0_conv_3/kernel/Regularizer/ConstО
&stream_0_conv_3/kernel/Regularizer/SumSum*stream_0_conv_3/kernel/Regularizer/Abs:y:01stream_0_conv_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_3/kernel/Regularizer/SumЎ
(stream_0_conv_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2*
(stream_0_conv_3/kernel/Regularizer/mul/x▄
&stream_0_conv_3/kernel/Regularizer/mulMul1stream_0_conv_3/kernel/Regularizer/mul/x:output:0/stream_0_conv_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_3/kernel/Regularizer/mulГ
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_1_24987*
_output_shapes
:	ђT*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOpе
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	ђT2 
dense_1/kernel/Regularizer/AbsЋ
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_1/kernel/Regularizer/Constи
dense_1/kernel/Regularizer/SumSum"dense_1/kernel/Regularizer/Abs:y:0)dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/SumЅ
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2"
 dense_1/kernel/Regularizer/mul/x╝
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/mulє
IdentityIdentity+dense_activation_1/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         T2

IdentityП
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall.^dense_1/kernel/Regularizer/Abs/ReadVariableOp(^dense_1_dropout/StatefulPartitionedCall(^stream_0_conv_1/StatefulPartitionedCall6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp(^stream_0_conv_2/StatefulPartitionedCall9^stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp(^stream_0_conv_3/StatefulPartitionedCall6^stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp(^stream_0_drop_1/StatefulPartitionedCall(^stream_0_drop_2/StatefulPartitionedCall(^stream_0_drop_3/StatefulPartitionedCall,^stream_0_input_drop/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:         }: : : : : : : : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2^
-dense_1/kernel/Regularizer/Abs/ReadVariableOp-dense_1/kernel/Regularizer/Abs/ReadVariableOp2R
'dense_1_dropout/StatefulPartitionedCall'dense_1_dropout/StatefulPartitionedCall2R
'stream_0_conv_1/StatefulPartitionedCall'stream_0_conv_1/StatefulPartitionedCall2n
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp2R
'stream_0_conv_2/StatefulPartitionedCall'stream_0_conv_2/StatefulPartitionedCall2t
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp2R
'stream_0_conv_3/StatefulPartitionedCall'stream_0_conv_3/StatefulPartitionedCall2n
5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp2R
'stream_0_drop_1/StatefulPartitionedCall'stream_0_drop_1/StatefulPartitionedCall2R
'stream_0_drop_2/StatefulPartitionedCall'stream_0_drop_2/StatefulPartitionedCall2R
'stream_0_drop_3/StatefulPartitionedCall'stream_0_drop_3/StatefulPartitionedCall2Z
+stream_0_input_drop/StatefulPartitionedCall+stream_0_input_drop/StatefulPartitionedCall:U Q
+
_output_shapes
:         }
"
_user_specified_name
inputs_0
џ
│
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_28713

inputs0
!batchnorm_readvariableop_resource:	ђ4
%batchnorm_mul_readvariableop_resource:	ђ2
#batchnorm_readvariableop_1_resource:	ђ2
#batchnorm_readvariableop_2_resource:	ђ
identityѕбbatchnorm/ReadVariableOpбbatchnorm/ReadVariableOp_1бbatchnorm/ReadVariableOp_2бbatchnorm/mul/ReadVariableOpЊ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2
batchnorm/add/yЅ
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:ђ2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:ђ2
batchnorm/RsqrtЪ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
batchnorm/mul/ReadVariableOpє
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:ђ2
batchnorm/mul{
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*,
_output_shapes
:         }ђ2
batchnorm/mul_1Ў
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype02
batchnorm/ReadVariableOp_1є
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:ђ2
batchnorm/mul_2Ў
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:ђ*
dtype02
batchnorm/ReadVariableOp_2ё
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:ђ2
batchnorm/subі
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*,
_output_shapes
:         }ђ2
batchnorm/add_1s
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*,
_output_shapes
:         }ђ2

Identity┬
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         }ђ: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:T P
,
_output_shapes
:         }ђ
 
_user_specified_nameinputs
І
l
N__inference_stream_0_input_drop_layer_call_and_return_conditional_losses_28073

inputs

identity_1^
IdentityIdentityinputs*
T0*+
_output_shapes
:         }2

Identitym

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:         }2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         }:S O
+
_output_shapes
:         }
 
_user_specified_nameinputs
ф
ы
)__inference_basemodel_layer_call_fn_27272
inputs_0
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:@ 
	unknown_5:@ђ
	unknown_6:	ђ
	unknown_7:	ђ
	unknown_8:	ђ
	unknown_9:	ђ

unknown_10:	ђ"

unknown_11:ђђ

unknown_12:	ђ

unknown_13:	ђ

unknown_14:	ђ

unknown_15:	ђ

unknown_16:	ђ

unknown_17:	ђT

unknown_18:T

unknown_19:T

unknown_20:T

unknown_21:T

unknown_22:T
identityѕбStatefulPartitionedCallю
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22*$
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         T*2
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *M
fHRF
D__inference_basemodel_layer_call_and_return_conditional_losses_256552
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         T2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:         }: : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
+
_output_shapes
:         }
"
_user_specified_name
inputs/0
ч
h
J__inference_dense_1_dropout_layer_call_and_return_conditional_losses_28821

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:         ђ2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:         ђ2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         ђ:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
§Ь
■
D__inference_basemodel_layer_call_and_return_conditional_losses_25177

inputsQ
;stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource:@=
/stream_0_conv_1_biasadd_readvariableop_resource:@C
5batch_normalization_batchnorm_readvariableop_resource:@G
9batch_normalization_batchnorm_mul_readvariableop_resource:@E
7batch_normalization_batchnorm_readvariableop_1_resource:@E
7batch_normalization_batchnorm_readvariableop_2_resource:@R
;stream_0_conv_2_conv1d_expanddims_1_readvariableop_resource:@ђ>
/stream_0_conv_2_biasadd_readvariableop_resource:	ђF
7batch_normalization_1_batchnorm_readvariableop_resource:	ђJ
;batch_normalization_1_batchnorm_mul_readvariableop_resource:	ђH
9batch_normalization_1_batchnorm_readvariableop_1_resource:	ђH
9batch_normalization_1_batchnorm_readvariableop_2_resource:	ђS
;stream_0_conv_3_conv1d_expanddims_1_readvariableop_resource:ђђ>
/stream_0_conv_3_biasadd_readvariableop_resource:	ђF
7batch_normalization_2_batchnorm_readvariableop_resource:	ђJ
;batch_normalization_2_batchnorm_mul_readvariableop_resource:	ђH
9batch_normalization_2_batchnorm_readvariableop_1_resource:	ђH
9batch_normalization_2_batchnorm_readvariableop_2_resource:	ђ9
&dense_1_matmul_readvariableop_resource:	ђT5
'dense_1_biasadd_readvariableop_resource:TE
7batch_normalization_3_batchnorm_readvariableop_resource:TI
;batch_normalization_3_batchnorm_mul_readvariableop_resource:TG
9batch_normalization_3_batchnorm_readvariableop_1_resource:TG
9batch_normalization_3_batchnorm_readvariableop_2_resource:T
identityѕб,batch_normalization/batchnorm/ReadVariableOpб.batch_normalization/batchnorm/ReadVariableOp_1б.batch_normalization/batchnorm/ReadVariableOp_2б0batch_normalization/batchnorm/mul/ReadVariableOpб.batch_normalization_1/batchnorm/ReadVariableOpб0batch_normalization_1/batchnorm/ReadVariableOp_1б0batch_normalization_1/batchnorm/ReadVariableOp_2б2batch_normalization_1/batchnorm/mul/ReadVariableOpб.batch_normalization_2/batchnorm/ReadVariableOpб0batch_normalization_2/batchnorm/ReadVariableOp_1б0batch_normalization_2/batchnorm/ReadVariableOp_2б2batch_normalization_2/batchnorm/mul/ReadVariableOpб.batch_normalization_3/batchnorm/ReadVariableOpб0batch_normalization_3/batchnorm/ReadVariableOp_1б0batch_normalization_3/batchnorm/ReadVariableOp_2б2batch_normalization_3/batchnorm/mul/ReadVariableOpбdense_1/BiasAdd/ReadVariableOpбdense_1/MatMul/ReadVariableOpб-dense_1/kernel/Regularizer/Abs/ReadVariableOpб&stream_0_conv_1/BiasAdd/ReadVariableOpб2stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOpб5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpб&stream_0_conv_2/BiasAdd/ReadVariableOpб2stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOpб8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOpб&stream_0_conv_3/BiasAdd/ReadVariableOpб2stream_0_conv_3/conv1d/ExpandDims_1/ReadVariableOpб5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOpє
stream_0_input_drop/IdentityIdentityinputs*
T0*+
_output_shapes
:         }2
stream_0_input_drop/IdentityЎ
%stream_0_conv_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§        2'
%stream_0_conv_1/conv1d/ExpandDims/dimт
!stream_0_conv_1/conv1d/ExpandDims
ExpandDims%stream_0_input_drop/Identity:output:0.stream_0_conv_1/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:         }2#
!stream_0_conv_1/conv1d/ExpandDimsУ
2stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp;stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype024
2stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOpћ
'stream_0_conv_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2)
'stream_0_conv_1/conv1d/ExpandDims_1/dimэ
#stream_0_conv_1/conv1d/ExpandDims_1
ExpandDims:stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp:value:00stream_0_conv_1/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@2%
#stream_0_conv_1/conv1d/ExpandDims_1Ш
stream_0_conv_1/conv1dConv2D*stream_0_conv_1/conv1d/ExpandDims:output:0,stream_0_conv_1/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:         }@*
paddingSAME*
strides
2
stream_0_conv_1/conv1d┬
stream_0_conv_1/conv1d/SqueezeSqueezestream_0_conv_1/conv1d:output:0*
T0*+
_output_shapes
:         }@*
squeeze_dims

§        2 
stream_0_conv_1/conv1d/Squeeze╝
&stream_0_conv_1/BiasAdd/ReadVariableOpReadVariableOp/stream_0_conv_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02(
&stream_0_conv_1/BiasAdd/ReadVariableOp╠
stream_0_conv_1/BiasAddBiasAdd'stream_0_conv_1/conv1d/Squeeze:output:0.stream_0_conv_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         }@2
stream_0_conv_1/BiasAdd╬
,batch_normalization/batchnorm/ReadVariableOpReadVariableOp5batch_normalization_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02.
,batch_normalization/batchnorm/ReadVariableOpЈ
#batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2%
#batch_normalization/batchnorm/add/yп
!batch_normalization/batchnorm/addAddV24batch_normalization/batchnorm/ReadVariableOp:value:0,batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes
:@2#
!batch_normalization/batchnorm/addЪ
#batch_normalization/batchnorm/RsqrtRsqrt%batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes
:@2%
#batch_normalization/batchnorm/Rsqrt┌
0batch_normalization/batchnorm/mul/ReadVariableOpReadVariableOp9batch_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype022
0batch_normalization/batchnorm/mul/ReadVariableOpН
!batch_normalization/batchnorm/mulMul'batch_normalization/batchnorm/Rsqrt:y:08batch_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2#
!batch_normalization/batchnorm/mulл
#batch_normalization/batchnorm/mul_1Mul stream_0_conv_1/BiasAdd:output:0%batch_normalization/batchnorm/mul:z:0*
T0*+
_output_shapes
:         }@2%
#batch_normalization/batchnorm/mul_1н
.batch_normalization/batchnorm/ReadVariableOp_1ReadVariableOp7batch_normalization_batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype020
.batch_normalization/batchnorm/ReadVariableOp_1Н
#batch_normalization/batchnorm/mul_2Mul6batch_normalization/batchnorm/ReadVariableOp_1:value:0%batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes
:@2%
#batch_normalization/batchnorm/mul_2н
.batch_normalization/batchnorm/ReadVariableOp_2ReadVariableOp7batch_normalization_batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype020
.batch_normalization/batchnorm/ReadVariableOp_2М
!batch_normalization/batchnorm/subSub6batch_normalization/batchnorm/ReadVariableOp_2:value:0'batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2#
!batch_normalization/batchnorm/sub┘
#batch_normalization/batchnorm/add_1AddV2'batch_normalization/batchnorm/mul_1:z:0%batch_normalization/batchnorm/sub:z:0*
T0*+
_output_shapes
:         }@2%
#batch_normalization/batchnorm/add_1Ѕ
activation/ReluRelu'batch_normalization/batchnorm/add_1:z:0*
T0*+
_output_shapes
:         }@2
activation/ReluЋ
stream_0_drop_1/IdentityIdentityactivation/Relu:activations:0*
T0*+
_output_shapes
:         }@2
stream_0_drop_1/IdentityЎ
%stream_0_conv_2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§        2'
%stream_0_conv_2/conv1d/ExpandDims/dimр
!stream_0_conv_2/conv1d/ExpandDims
ExpandDims!stream_0_drop_1/Identity:output:0.stream_0_conv_2/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:         }@2#
!stream_0_conv_2/conv1d/ExpandDimsж
2stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp;stream_0_conv_2_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:@ђ*
dtype024
2stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOpћ
'stream_0_conv_2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2)
'stream_0_conv_2/conv1d/ExpandDims_1/dimЭ
#stream_0_conv_2/conv1d/ExpandDims_1
ExpandDims:stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOp:value:00stream_0_conv_2/conv1d/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:@ђ2%
#stream_0_conv_2/conv1d/ExpandDims_1э
stream_0_conv_2/conv1dConv2D*stream_0_conv_2/conv1d/ExpandDims:output:0,stream_0_conv_2/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:         }ђ*
paddingSAME*
strides
2
stream_0_conv_2/conv1d├
stream_0_conv_2/conv1d/SqueezeSqueezestream_0_conv_2/conv1d:output:0*
T0*,
_output_shapes
:         }ђ*
squeeze_dims

§        2 
stream_0_conv_2/conv1d/Squeezeй
&stream_0_conv_2/BiasAdd/ReadVariableOpReadVariableOp/stream_0_conv_2_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02(
&stream_0_conv_2/BiasAdd/ReadVariableOp═
stream_0_conv_2/BiasAddBiasAdd'stream_0_conv_2/conv1d/Squeeze:output:0.stream_0_conv_2/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:         }ђ2
stream_0_conv_2/BiasAddН
.batch_normalization_1/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_1_batchnorm_readvariableop_resource*
_output_shapes	
:ђ*
dtype020
.batch_normalization_1/batchnorm/ReadVariableOpЊ
%batch_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2'
%batch_normalization_1/batchnorm/add/yр
#batch_normalization_1/batchnorm/addAddV26batch_normalization_1/batchnorm/ReadVariableOp:value:0.batch_normalization_1/batchnorm/add/y:output:0*
T0*
_output_shapes	
:ђ2%
#batch_normalization_1/batchnorm/addд
%batch_normalization_1/batchnorm/RsqrtRsqrt'batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes	
:ђ2'
%batch_normalization_1/batchnorm/Rsqrtр
2batch_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes	
:ђ*
dtype024
2batch_normalization_1/batchnorm/mul/ReadVariableOpя
#batch_normalization_1/batchnorm/mulMul)batch_normalization_1/batchnorm/Rsqrt:y:0:batch_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:ђ2%
#batch_normalization_1/batchnorm/mulО
%batch_normalization_1/batchnorm/mul_1Mul stream_0_conv_2/BiasAdd:output:0'batch_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:         }ђ2'
%batch_normalization_1/batchnorm/mul_1█
0batch_normalization_1/batchnorm/ReadVariableOp_1ReadVariableOp9batch_normalization_1_batchnorm_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype022
0batch_normalization_1/batchnorm/ReadVariableOp_1я
%batch_normalization_1/batchnorm/mul_2Mul8batch_normalization_1/batchnorm/ReadVariableOp_1:value:0'batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes	
:ђ2'
%batch_normalization_1/batchnorm/mul_2█
0batch_normalization_1/batchnorm/ReadVariableOp_2ReadVariableOp9batch_normalization_1_batchnorm_readvariableop_2_resource*
_output_shapes	
:ђ*
dtype022
0batch_normalization_1/batchnorm/ReadVariableOp_2▄
#batch_normalization_1/batchnorm/subSub8batch_normalization_1/batchnorm/ReadVariableOp_2:value:0)batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:ђ2%
#batch_normalization_1/batchnorm/subР
%batch_normalization_1/batchnorm/add_1AddV2)batch_normalization_1/batchnorm/mul_1:z:0'batch_normalization_1/batchnorm/sub:z:0*
T0*,
_output_shapes
:         }ђ2'
%batch_normalization_1/batchnorm/add_1љ
activation_1/ReluRelu)batch_normalization_1/batchnorm/add_1:z:0*
T0*,
_output_shapes
:         }ђ2
activation_1/Reluў
stream_0_drop_2/IdentityIdentityactivation_1/Relu:activations:0*
T0*,
_output_shapes
:         }ђ2
stream_0_drop_2/IdentityЎ
%stream_0_conv_3/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§        2'
%stream_0_conv_3/conv1d/ExpandDims/dimР
!stream_0_conv_3/conv1d/ExpandDims
ExpandDims!stream_0_drop_2/Identity:output:0.stream_0_conv_3/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         }ђ2#
!stream_0_conv_3/conv1d/ExpandDimsЖ
2stream_0_conv_3/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp;stream_0_conv_3_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:ђђ*
dtype024
2stream_0_conv_3/conv1d/ExpandDims_1/ReadVariableOpћ
'stream_0_conv_3/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2)
'stream_0_conv_3/conv1d/ExpandDims_1/dimщ
#stream_0_conv_3/conv1d/ExpandDims_1
ExpandDims:stream_0_conv_3/conv1d/ExpandDims_1/ReadVariableOp:value:00stream_0_conv_3/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:ђђ2%
#stream_0_conv_3/conv1d/ExpandDims_1э
stream_0_conv_3/conv1dConv2D*stream_0_conv_3/conv1d/ExpandDims:output:0,stream_0_conv_3/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:         }ђ*
paddingSAME*
strides
2
stream_0_conv_3/conv1d├
stream_0_conv_3/conv1d/SqueezeSqueezestream_0_conv_3/conv1d:output:0*
T0*,
_output_shapes
:         }ђ*
squeeze_dims

§        2 
stream_0_conv_3/conv1d/Squeezeй
&stream_0_conv_3/BiasAdd/ReadVariableOpReadVariableOp/stream_0_conv_3_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02(
&stream_0_conv_3/BiasAdd/ReadVariableOp═
stream_0_conv_3/BiasAddBiasAdd'stream_0_conv_3/conv1d/Squeeze:output:0.stream_0_conv_3/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:         }ђ2
stream_0_conv_3/BiasAddН
.batch_normalization_2/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_2_batchnorm_readvariableop_resource*
_output_shapes	
:ђ*
dtype020
.batch_normalization_2/batchnorm/ReadVariableOpЊ
%batch_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2'
%batch_normalization_2/batchnorm/add/yр
#batch_normalization_2/batchnorm/addAddV26batch_normalization_2/batchnorm/ReadVariableOp:value:0.batch_normalization_2/batchnorm/add/y:output:0*
T0*
_output_shapes	
:ђ2%
#batch_normalization_2/batchnorm/addд
%batch_normalization_2/batchnorm/RsqrtRsqrt'batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes	
:ђ2'
%batch_normalization_2/batchnorm/Rsqrtр
2batch_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes	
:ђ*
dtype024
2batch_normalization_2/batchnorm/mul/ReadVariableOpя
#batch_normalization_2/batchnorm/mulMul)batch_normalization_2/batchnorm/Rsqrt:y:0:batch_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:ђ2%
#batch_normalization_2/batchnorm/mulО
%batch_normalization_2/batchnorm/mul_1Mul stream_0_conv_3/BiasAdd:output:0'batch_normalization_2/batchnorm/mul:z:0*
T0*,
_output_shapes
:         }ђ2'
%batch_normalization_2/batchnorm/mul_1█
0batch_normalization_2/batchnorm/ReadVariableOp_1ReadVariableOp9batch_normalization_2_batchnorm_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype022
0batch_normalization_2/batchnorm/ReadVariableOp_1я
%batch_normalization_2/batchnorm/mul_2Mul8batch_normalization_2/batchnorm/ReadVariableOp_1:value:0'batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes	
:ђ2'
%batch_normalization_2/batchnorm/mul_2█
0batch_normalization_2/batchnorm/ReadVariableOp_2ReadVariableOp9batch_normalization_2_batchnorm_readvariableop_2_resource*
_output_shapes	
:ђ*
dtype022
0batch_normalization_2/batchnorm/ReadVariableOp_2▄
#batch_normalization_2/batchnorm/subSub8batch_normalization_2/batchnorm/ReadVariableOp_2:value:0)batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:ђ2%
#batch_normalization_2/batchnorm/subР
%batch_normalization_2/batchnorm/add_1AddV2)batch_normalization_2/batchnorm/mul_1:z:0'batch_normalization_2/batchnorm/sub:z:0*
T0*,
_output_shapes
:         }ђ2'
%batch_normalization_2/batchnorm/add_1љ
activation_2/ReluRelu)batch_normalization_2/batchnorm/add_1:z:0*
T0*,
_output_shapes
:         }ђ2
activation_2/Reluў
stream_0_drop_3/IdentityIdentityactivation_2/Relu:activations:0*
T0*,
_output_shapes
:         }ђ2
stream_0_drop_3/Identityц
/global_average_pooling1d/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :21
/global_average_pooling1d/Mean/reduction_indicesо
global_average_pooling1d/MeanMean!stream_0_drop_3/Identity:output:08global_average_pooling1d/Mean/reduction_indices:output:0*
T0*(
_output_shapes
:         ђ2
global_average_pooling1d/MeanЏ
dense_1_dropout/IdentityIdentity&global_average_pooling1d/Mean:output:0*
T0*(
_output_shapes
:         ђ2
dense_1_dropout/Identityд
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	ђT*
dtype02
dense_1/MatMul/ReadVariableOpд
dense_1/MatMulMatMul!dense_1_dropout/Identity:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         T2
dense_1/MatMulц
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:T*
dtype02 
dense_1/BiasAdd/ReadVariableOpА
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         T2
dense_1/BiasAddн
.batch_normalization_3/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_3_batchnorm_readvariableop_resource*
_output_shapes
:T*
dtype020
.batch_normalization_3/batchnorm/ReadVariableOpЊ
%batch_normalization_3/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2'
%batch_normalization_3/batchnorm/add/yЯ
#batch_normalization_3/batchnorm/addAddV26batch_normalization_3/batchnorm/ReadVariableOp:value:0.batch_normalization_3/batchnorm/add/y:output:0*
T0*
_output_shapes
:T2%
#batch_normalization_3/batchnorm/addЦ
%batch_normalization_3/batchnorm/RsqrtRsqrt'batch_normalization_3/batchnorm/add:z:0*
T0*
_output_shapes
:T2'
%batch_normalization_3/batchnorm/RsqrtЯ
2batch_normalization_3/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_3_batchnorm_mul_readvariableop_resource*
_output_shapes
:T*
dtype024
2batch_normalization_3/batchnorm/mul/ReadVariableOpП
#batch_normalization_3/batchnorm/mulMul)batch_normalization_3/batchnorm/Rsqrt:y:0:batch_normalization_3/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:T2%
#batch_normalization_3/batchnorm/mul╩
%batch_normalization_3/batchnorm/mul_1Muldense_1/BiasAdd:output:0'batch_normalization_3/batchnorm/mul:z:0*
T0*'
_output_shapes
:         T2'
%batch_normalization_3/batchnorm/mul_1┌
0batch_normalization_3/batchnorm/ReadVariableOp_1ReadVariableOp9batch_normalization_3_batchnorm_readvariableop_1_resource*
_output_shapes
:T*
dtype022
0batch_normalization_3/batchnorm/ReadVariableOp_1П
%batch_normalization_3/batchnorm/mul_2Mul8batch_normalization_3/batchnorm/ReadVariableOp_1:value:0'batch_normalization_3/batchnorm/mul:z:0*
T0*
_output_shapes
:T2'
%batch_normalization_3/batchnorm/mul_2┌
0batch_normalization_3/batchnorm/ReadVariableOp_2ReadVariableOp9batch_normalization_3_batchnorm_readvariableop_2_resource*
_output_shapes
:T*
dtype022
0batch_normalization_3/batchnorm/ReadVariableOp_2█
#batch_normalization_3/batchnorm/subSub8batch_normalization_3/batchnorm/ReadVariableOp_2:value:0)batch_normalization_3/batchnorm/mul_2:z:0*
T0*
_output_shapes
:T2%
#batch_normalization_3/batchnorm/subП
%batch_normalization_3/batchnorm/add_1AddV2)batch_normalization_3/batchnorm/mul_1:z:0'batch_normalization_3/batchnorm/sub:z:0*
T0*'
_output_shapes
:         T2'
%batch_normalization_3/batchnorm/add_1а
dense_activation_1/SigmoidSigmoid)batch_normalization_3/batchnorm/add_1:z:0*
T0*'
_output_shapes
:         T2
dense_activation_1/SigmoidЬ
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp;stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype027
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp├
&stream_0_conv_1/kernel/Regularizer/AbsAbs=stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_0_conv_1/kernel/Regularizer/AbsЕ
(stream_0_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_0_conv_1/kernel/Regularizer/ConstО
&stream_0_conv_1/kernel/Regularizer/SumSum*stream_0_conv_1/kernel/Regularizer/Abs:y:01stream_0_conv_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/SumЎ
(stream_0_conv_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2*
(stream_0_conv_1/kernel/Regularizer/mul/x▄
&stream_0_conv_1/kernel/Regularizer/mulMul1stream_0_conv_1/kernel/Regularizer/mul/x:output:0/stream_0_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/mulш
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;stream_0_conv_2_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:@ђ*
dtype02:
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOpл
)stream_0_conv_2/kernel/Regularizer/SquareSquare@stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*#
_output_shapes
:@ђ2+
)stream_0_conv_2/kernel/Regularizer/SquareЕ
(stream_0_conv_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_0_conv_2/kernel/Regularizer/Const┌
&stream_0_conv_2/kernel/Regularizer/SumSum-stream_0_conv_2/kernel/Regularizer/Square:y:01stream_0_conv_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_2/kernel/Regularizer/SumЎ
(stream_0_conv_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2*
(stream_0_conv_2/kernel/Regularizer/mul/x▄
&stream_0_conv_2/kernel/Regularizer/mulMul1stream_0_conv_2/kernel/Regularizer/mul/x:output:0/stream_0_conv_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_2/kernel/Regularizer/mul­
5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp;stream_0_conv_3_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:ђђ*
dtype027
5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp┼
&stream_0_conv_3/kernel/Regularizer/AbsAbs=stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*$
_output_shapes
:ђђ2(
&stream_0_conv_3/kernel/Regularizer/AbsЕ
(stream_0_conv_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_0_conv_3/kernel/Regularizer/ConstО
&stream_0_conv_3/kernel/Regularizer/SumSum*stream_0_conv_3/kernel/Regularizer/Abs:y:01stream_0_conv_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_3/kernel/Regularizer/SumЎ
(stream_0_conv_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2*
(stream_0_conv_3/kernel/Regularizer/mul/x▄
&stream_0_conv_3/kernel/Regularizer/mulMul1stream_0_conv_3/kernel/Regularizer/mul/x:output:0/stream_0_conv_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_3/kernel/Regularizer/mulк
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	ђT*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOpе
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	ђT2 
dense_1/kernel/Regularizer/AbsЋ
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_1/kernel/Regularizer/Constи
dense_1/kernel/Regularizer/SumSum"dense_1/kernel/Regularizer/Abs:y:0)dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/SumЅ
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2"
 dense_1/kernel/Regularizer/mul/x╝
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/muly
IdentityIdentitydense_activation_1/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:         T2

Identityг
NoOpNoOp-^batch_normalization/batchnorm/ReadVariableOp/^batch_normalization/batchnorm/ReadVariableOp_1/^batch_normalization/batchnorm/ReadVariableOp_21^batch_normalization/batchnorm/mul/ReadVariableOp/^batch_normalization_1/batchnorm/ReadVariableOp1^batch_normalization_1/batchnorm/ReadVariableOp_11^batch_normalization_1/batchnorm/ReadVariableOp_23^batch_normalization_1/batchnorm/mul/ReadVariableOp/^batch_normalization_2/batchnorm/ReadVariableOp1^batch_normalization_2/batchnorm/ReadVariableOp_11^batch_normalization_2/batchnorm/ReadVariableOp_23^batch_normalization_2/batchnorm/mul/ReadVariableOp/^batch_normalization_3/batchnorm/ReadVariableOp1^batch_normalization_3/batchnorm/ReadVariableOp_11^batch_normalization_3/batchnorm/ReadVariableOp_23^batch_normalization_3/batchnorm/mul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp.^dense_1/kernel/Regularizer/Abs/ReadVariableOp'^stream_0_conv_1/BiasAdd/ReadVariableOp3^stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp'^stream_0_conv_2/BiasAdd/ReadVariableOp3^stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOp9^stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp'^stream_0_conv_3/BiasAdd/ReadVariableOp3^stream_0_conv_3/conv1d/ExpandDims_1/ReadVariableOp6^stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:         }: : : : : : : : : : : : : : : : : : : : : : : : 2\
,batch_normalization/batchnorm/ReadVariableOp,batch_normalization/batchnorm/ReadVariableOp2`
.batch_normalization/batchnorm/ReadVariableOp_1.batch_normalization/batchnorm/ReadVariableOp_12`
.batch_normalization/batchnorm/ReadVariableOp_2.batch_normalization/batchnorm/ReadVariableOp_22d
0batch_normalization/batchnorm/mul/ReadVariableOp0batch_normalization/batchnorm/mul/ReadVariableOp2`
.batch_normalization_1/batchnorm/ReadVariableOp.batch_normalization_1/batchnorm/ReadVariableOp2d
0batch_normalization_1/batchnorm/ReadVariableOp_10batch_normalization_1/batchnorm/ReadVariableOp_12d
0batch_normalization_1/batchnorm/ReadVariableOp_20batch_normalization_1/batchnorm/ReadVariableOp_22h
2batch_normalization_1/batchnorm/mul/ReadVariableOp2batch_normalization_1/batchnorm/mul/ReadVariableOp2`
.batch_normalization_2/batchnorm/ReadVariableOp.batch_normalization_2/batchnorm/ReadVariableOp2d
0batch_normalization_2/batchnorm/ReadVariableOp_10batch_normalization_2/batchnorm/ReadVariableOp_12d
0batch_normalization_2/batchnorm/ReadVariableOp_20batch_normalization_2/batchnorm/ReadVariableOp_22h
2batch_normalization_2/batchnorm/mul/ReadVariableOp2batch_normalization_2/batchnorm/mul/ReadVariableOp2`
.batch_normalization_3/batchnorm/ReadVariableOp.batch_normalization_3/batchnorm/ReadVariableOp2d
0batch_normalization_3/batchnorm/ReadVariableOp_10batch_normalization_3/batchnorm/ReadVariableOp_12d
0batch_normalization_3/batchnorm/ReadVariableOp_20batch_normalization_3/batchnorm/ReadVariableOp_22h
2batch_normalization_3/batchnorm/mul/ReadVariableOp2batch_normalization_3/batchnorm/mul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2^
-dense_1/kernel/Regularizer/Abs/ReadVariableOp-dense_1/kernel/Regularizer/Abs/ReadVariableOp2P
&stream_0_conv_1/BiasAdd/ReadVariableOp&stream_0_conv_1/BiasAdd/ReadVariableOp2h
2stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp2stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp2n
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp2P
&stream_0_conv_2/BiasAdd/ReadVariableOp&stream_0_conv_2/BiasAdd/ReadVariableOp2h
2stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOp2stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOp2t
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp2P
&stream_0_conv_3/BiasAdd/ReadVariableOp&stream_0_conv_3/BiasAdd/ReadVariableOp2h
2stream_0_conv_3/conv1d/ExpandDims_1/ReadVariableOp2stream_0_conv_3/conv1d/ExpandDims_1/ReadVariableOp2n
5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp:S O
+
_output_shapes
:         }
 
_user_specified_nameinputs
ь
a
E__inference_activation_layer_call_and_return_conditional_losses_23945

inputs
identityR
ReluReluinputs*
T0*+
_output_shapes
:         }@2
Reluj
IdentityIdentityRelu:activations:0*
T0*+
_output_shapes
:         }@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         }@:S O
+
_output_shapes
:         }@
 
_user_specified_nameinputs
╦*
ж
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_23792

inputs5
'assignmovingavg_readvariableop_resource:T7
)assignmovingavg_1_readvariableop_resource:T3
%batchnorm_mul_readvariableop_resource:T/
!batchnorm_readvariableop_resource:T
identityѕбAssignMovingAvgбAssignMovingAvg/ReadVariableOpбAssignMovingAvg_1б AssignMovingAvg_1/ReadVariableOpбbatchnorm/ReadVariableOpбbatchnorm/mul/ReadVariableOpі
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indicesЈ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:T*
	keep_dims(2
moments/mean|
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:T2
moments/StopGradientц
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:         T2
moments/SquaredDifferenceњ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices▓
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:T*
	keep_dims(2
moments/varianceђ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:T*
squeeze_dims
 2
moments/Squeezeѕ
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:T*
squeeze_dims
 2
moments/Squeeze_1s
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2
AssignMovingAvg/decayц
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:T*
dtype02 
AssignMovingAvg/ReadVariableOpў
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:T2
AssignMovingAvg/subЈ
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:T2
AssignMovingAvg/mul┐
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvgw
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2
AssignMovingAvg_1/decayф
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:T*
dtype02"
 AssignMovingAvg_1/ReadVariableOpа
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:T2
AssignMovingAvg_1/subЌ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:T2
AssignMovingAvg_1/mul╔
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1g
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2
batchnorm/add/yѓ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:T2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:T2
batchnorm/Rsqrtъ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:T*
dtype02
batchnorm/mul/ReadVariableOpЁ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:T2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:         T2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:T2
batchnorm/mul_2њ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:T*
dtype02
batchnorm/ReadVariableOpЂ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:T2
batchnorm/subЁ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:         T2
batchnorm/add_1n
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:         T2

IdentityЫ
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         T: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:         T
 
_user_specified_nameinputs
І
h
J__inference_stream_0_drop_3_layer_call_and_return_conditional_losses_28772

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:         }ђ2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:         }ђ2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         }ђ:T P
,
_output_shapes
:         }ђ
 
_user_specified_nameinputs
є
Г
N__inference_batch_normalization_layer_call_and_return_conditional_losses_23930

inputs/
!batchnorm_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@1
#batchnorm_readvariableop_1_resource:@1
#batchnorm_readvariableop_2_resource:@
identityѕбbatchnorm/ReadVariableOpбbatchnorm/ReadVariableOp_1бbatchnorm/ReadVariableOp_2бbatchnorm/mul/ReadVariableOpњ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2
batchnorm/add/yѕ
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@2
batchnorm/Rsqrtъ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOpЁ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2
batchnorm/mulz
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:         }@2
batchnorm/mul_1ў
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp_1Ё
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:@2
batchnorm/mul_2ў
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp_2Ѓ
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2
batchnorm/subЅ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*+
_output_shapes
:         }@2
batchnorm/add_1r
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*+
_output_shapes
:         }@2

Identity┬
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         }@: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:S O
+
_output_shapes
:         }@
 
_user_specified_nameinputs
ц
№
)__inference_basemodel_layer_call_fn_27166

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:@ 
	unknown_5:@ђ
	unknown_6:	ђ
	unknown_7:	ђ
	unknown_8:	ђ
	unknown_9:	ђ

unknown_10:	ђ"

unknown_11:ђђ

unknown_12:	ђ

unknown_13:	ђ

unknown_14:	ђ

unknown_15:	ђ

unknown_16:	ђ

unknown_17:	ђT

unknown_18:T

unknown_19:T

unknown_20:T

unknown_21:T

unknown_22:T
identityѕбStatefulPartitionedCallџ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22*$
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         T*2
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *M
fHRF
D__inference_basemodel_layer_call_and_return_conditional_losses_247352
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         T2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:         }: : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:         }
 
_user_specified_nameinputs
§Ь
■
D__inference_basemodel_layer_call_and_return_conditional_losses_27414

inputsQ
;stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource:@=
/stream_0_conv_1_biasadd_readvariableop_resource:@C
5batch_normalization_batchnorm_readvariableop_resource:@G
9batch_normalization_batchnorm_mul_readvariableop_resource:@E
7batch_normalization_batchnorm_readvariableop_1_resource:@E
7batch_normalization_batchnorm_readvariableop_2_resource:@R
;stream_0_conv_2_conv1d_expanddims_1_readvariableop_resource:@ђ>
/stream_0_conv_2_biasadd_readvariableop_resource:	ђF
7batch_normalization_1_batchnorm_readvariableop_resource:	ђJ
;batch_normalization_1_batchnorm_mul_readvariableop_resource:	ђH
9batch_normalization_1_batchnorm_readvariableop_1_resource:	ђH
9batch_normalization_1_batchnorm_readvariableop_2_resource:	ђS
;stream_0_conv_3_conv1d_expanddims_1_readvariableop_resource:ђђ>
/stream_0_conv_3_biasadd_readvariableop_resource:	ђF
7batch_normalization_2_batchnorm_readvariableop_resource:	ђJ
;batch_normalization_2_batchnorm_mul_readvariableop_resource:	ђH
9batch_normalization_2_batchnorm_readvariableop_1_resource:	ђH
9batch_normalization_2_batchnorm_readvariableop_2_resource:	ђ9
&dense_1_matmul_readvariableop_resource:	ђT5
'dense_1_biasadd_readvariableop_resource:TE
7batch_normalization_3_batchnorm_readvariableop_resource:TI
;batch_normalization_3_batchnorm_mul_readvariableop_resource:TG
9batch_normalization_3_batchnorm_readvariableop_1_resource:TG
9batch_normalization_3_batchnorm_readvariableop_2_resource:T
identityѕб,batch_normalization/batchnorm/ReadVariableOpб.batch_normalization/batchnorm/ReadVariableOp_1б.batch_normalization/batchnorm/ReadVariableOp_2б0batch_normalization/batchnorm/mul/ReadVariableOpб.batch_normalization_1/batchnorm/ReadVariableOpб0batch_normalization_1/batchnorm/ReadVariableOp_1б0batch_normalization_1/batchnorm/ReadVariableOp_2б2batch_normalization_1/batchnorm/mul/ReadVariableOpб.batch_normalization_2/batchnorm/ReadVariableOpб0batch_normalization_2/batchnorm/ReadVariableOp_1б0batch_normalization_2/batchnorm/ReadVariableOp_2б2batch_normalization_2/batchnorm/mul/ReadVariableOpб.batch_normalization_3/batchnorm/ReadVariableOpб0batch_normalization_3/batchnorm/ReadVariableOp_1б0batch_normalization_3/batchnorm/ReadVariableOp_2б2batch_normalization_3/batchnorm/mul/ReadVariableOpбdense_1/BiasAdd/ReadVariableOpбdense_1/MatMul/ReadVariableOpб-dense_1/kernel/Regularizer/Abs/ReadVariableOpб&stream_0_conv_1/BiasAdd/ReadVariableOpб2stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOpб5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpб&stream_0_conv_2/BiasAdd/ReadVariableOpб2stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOpб8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOpб&stream_0_conv_3/BiasAdd/ReadVariableOpб2stream_0_conv_3/conv1d/ExpandDims_1/ReadVariableOpб5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOpє
stream_0_input_drop/IdentityIdentityinputs*
T0*+
_output_shapes
:         }2
stream_0_input_drop/IdentityЎ
%stream_0_conv_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§        2'
%stream_0_conv_1/conv1d/ExpandDims/dimт
!stream_0_conv_1/conv1d/ExpandDims
ExpandDims%stream_0_input_drop/Identity:output:0.stream_0_conv_1/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:         }2#
!stream_0_conv_1/conv1d/ExpandDimsУ
2stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp;stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype024
2stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOpћ
'stream_0_conv_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2)
'stream_0_conv_1/conv1d/ExpandDims_1/dimэ
#stream_0_conv_1/conv1d/ExpandDims_1
ExpandDims:stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp:value:00stream_0_conv_1/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@2%
#stream_0_conv_1/conv1d/ExpandDims_1Ш
stream_0_conv_1/conv1dConv2D*stream_0_conv_1/conv1d/ExpandDims:output:0,stream_0_conv_1/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:         }@*
paddingSAME*
strides
2
stream_0_conv_1/conv1d┬
stream_0_conv_1/conv1d/SqueezeSqueezestream_0_conv_1/conv1d:output:0*
T0*+
_output_shapes
:         }@*
squeeze_dims

§        2 
stream_0_conv_1/conv1d/Squeeze╝
&stream_0_conv_1/BiasAdd/ReadVariableOpReadVariableOp/stream_0_conv_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02(
&stream_0_conv_1/BiasAdd/ReadVariableOp╠
stream_0_conv_1/BiasAddBiasAdd'stream_0_conv_1/conv1d/Squeeze:output:0.stream_0_conv_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         }@2
stream_0_conv_1/BiasAdd╬
,batch_normalization/batchnorm/ReadVariableOpReadVariableOp5batch_normalization_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02.
,batch_normalization/batchnorm/ReadVariableOpЈ
#batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2%
#batch_normalization/batchnorm/add/yп
!batch_normalization/batchnorm/addAddV24batch_normalization/batchnorm/ReadVariableOp:value:0,batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes
:@2#
!batch_normalization/batchnorm/addЪ
#batch_normalization/batchnorm/RsqrtRsqrt%batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes
:@2%
#batch_normalization/batchnorm/Rsqrt┌
0batch_normalization/batchnorm/mul/ReadVariableOpReadVariableOp9batch_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype022
0batch_normalization/batchnorm/mul/ReadVariableOpН
!batch_normalization/batchnorm/mulMul'batch_normalization/batchnorm/Rsqrt:y:08batch_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2#
!batch_normalization/batchnorm/mulл
#batch_normalization/batchnorm/mul_1Mul stream_0_conv_1/BiasAdd:output:0%batch_normalization/batchnorm/mul:z:0*
T0*+
_output_shapes
:         }@2%
#batch_normalization/batchnorm/mul_1н
.batch_normalization/batchnorm/ReadVariableOp_1ReadVariableOp7batch_normalization_batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype020
.batch_normalization/batchnorm/ReadVariableOp_1Н
#batch_normalization/batchnorm/mul_2Mul6batch_normalization/batchnorm/ReadVariableOp_1:value:0%batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes
:@2%
#batch_normalization/batchnorm/mul_2н
.batch_normalization/batchnorm/ReadVariableOp_2ReadVariableOp7batch_normalization_batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype020
.batch_normalization/batchnorm/ReadVariableOp_2М
!batch_normalization/batchnorm/subSub6batch_normalization/batchnorm/ReadVariableOp_2:value:0'batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2#
!batch_normalization/batchnorm/sub┘
#batch_normalization/batchnorm/add_1AddV2'batch_normalization/batchnorm/mul_1:z:0%batch_normalization/batchnorm/sub:z:0*
T0*+
_output_shapes
:         }@2%
#batch_normalization/batchnorm/add_1Ѕ
activation/ReluRelu'batch_normalization/batchnorm/add_1:z:0*
T0*+
_output_shapes
:         }@2
activation/ReluЋ
stream_0_drop_1/IdentityIdentityactivation/Relu:activations:0*
T0*+
_output_shapes
:         }@2
stream_0_drop_1/IdentityЎ
%stream_0_conv_2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§        2'
%stream_0_conv_2/conv1d/ExpandDims/dimр
!stream_0_conv_2/conv1d/ExpandDims
ExpandDims!stream_0_drop_1/Identity:output:0.stream_0_conv_2/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:         }@2#
!stream_0_conv_2/conv1d/ExpandDimsж
2stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp;stream_0_conv_2_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:@ђ*
dtype024
2stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOpћ
'stream_0_conv_2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2)
'stream_0_conv_2/conv1d/ExpandDims_1/dimЭ
#stream_0_conv_2/conv1d/ExpandDims_1
ExpandDims:stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOp:value:00stream_0_conv_2/conv1d/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:@ђ2%
#stream_0_conv_2/conv1d/ExpandDims_1э
stream_0_conv_2/conv1dConv2D*stream_0_conv_2/conv1d/ExpandDims:output:0,stream_0_conv_2/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:         }ђ*
paddingSAME*
strides
2
stream_0_conv_2/conv1d├
stream_0_conv_2/conv1d/SqueezeSqueezestream_0_conv_2/conv1d:output:0*
T0*,
_output_shapes
:         }ђ*
squeeze_dims

§        2 
stream_0_conv_2/conv1d/Squeezeй
&stream_0_conv_2/BiasAdd/ReadVariableOpReadVariableOp/stream_0_conv_2_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02(
&stream_0_conv_2/BiasAdd/ReadVariableOp═
stream_0_conv_2/BiasAddBiasAdd'stream_0_conv_2/conv1d/Squeeze:output:0.stream_0_conv_2/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:         }ђ2
stream_0_conv_2/BiasAddН
.batch_normalization_1/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_1_batchnorm_readvariableop_resource*
_output_shapes	
:ђ*
dtype020
.batch_normalization_1/batchnorm/ReadVariableOpЊ
%batch_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2'
%batch_normalization_1/batchnorm/add/yр
#batch_normalization_1/batchnorm/addAddV26batch_normalization_1/batchnorm/ReadVariableOp:value:0.batch_normalization_1/batchnorm/add/y:output:0*
T0*
_output_shapes	
:ђ2%
#batch_normalization_1/batchnorm/addд
%batch_normalization_1/batchnorm/RsqrtRsqrt'batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes	
:ђ2'
%batch_normalization_1/batchnorm/Rsqrtр
2batch_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes	
:ђ*
dtype024
2batch_normalization_1/batchnorm/mul/ReadVariableOpя
#batch_normalization_1/batchnorm/mulMul)batch_normalization_1/batchnorm/Rsqrt:y:0:batch_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:ђ2%
#batch_normalization_1/batchnorm/mulО
%batch_normalization_1/batchnorm/mul_1Mul stream_0_conv_2/BiasAdd:output:0'batch_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:         }ђ2'
%batch_normalization_1/batchnorm/mul_1█
0batch_normalization_1/batchnorm/ReadVariableOp_1ReadVariableOp9batch_normalization_1_batchnorm_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype022
0batch_normalization_1/batchnorm/ReadVariableOp_1я
%batch_normalization_1/batchnorm/mul_2Mul8batch_normalization_1/batchnorm/ReadVariableOp_1:value:0'batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes	
:ђ2'
%batch_normalization_1/batchnorm/mul_2█
0batch_normalization_1/batchnorm/ReadVariableOp_2ReadVariableOp9batch_normalization_1_batchnorm_readvariableop_2_resource*
_output_shapes	
:ђ*
dtype022
0batch_normalization_1/batchnorm/ReadVariableOp_2▄
#batch_normalization_1/batchnorm/subSub8batch_normalization_1/batchnorm/ReadVariableOp_2:value:0)batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:ђ2%
#batch_normalization_1/batchnorm/subР
%batch_normalization_1/batchnorm/add_1AddV2)batch_normalization_1/batchnorm/mul_1:z:0'batch_normalization_1/batchnorm/sub:z:0*
T0*,
_output_shapes
:         }ђ2'
%batch_normalization_1/batchnorm/add_1љ
activation_1/ReluRelu)batch_normalization_1/batchnorm/add_1:z:0*
T0*,
_output_shapes
:         }ђ2
activation_1/Reluў
stream_0_drop_2/IdentityIdentityactivation_1/Relu:activations:0*
T0*,
_output_shapes
:         }ђ2
stream_0_drop_2/IdentityЎ
%stream_0_conv_3/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§        2'
%stream_0_conv_3/conv1d/ExpandDims/dimР
!stream_0_conv_3/conv1d/ExpandDims
ExpandDims!stream_0_drop_2/Identity:output:0.stream_0_conv_3/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         }ђ2#
!stream_0_conv_3/conv1d/ExpandDimsЖ
2stream_0_conv_3/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp;stream_0_conv_3_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:ђђ*
dtype024
2stream_0_conv_3/conv1d/ExpandDims_1/ReadVariableOpћ
'stream_0_conv_3/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2)
'stream_0_conv_3/conv1d/ExpandDims_1/dimщ
#stream_0_conv_3/conv1d/ExpandDims_1
ExpandDims:stream_0_conv_3/conv1d/ExpandDims_1/ReadVariableOp:value:00stream_0_conv_3/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:ђђ2%
#stream_0_conv_3/conv1d/ExpandDims_1э
stream_0_conv_3/conv1dConv2D*stream_0_conv_3/conv1d/ExpandDims:output:0,stream_0_conv_3/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:         }ђ*
paddingSAME*
strides
2
stream_0_conv_3/conv1d├
stream_0_conv_3/conv1d/SqueezeSqueezestream_0_conv_3/conv1d:output:0*
T0*,
_output_shapes
:         }ђ*
squeeze_dims

§        2 
stream_0_conv_3/conv1d/Squeezeй
&stream_0_conv_3/BiasAdd/ReadVariableOpReadVariableOp/stream_0_conv_3_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02(
&stream_0_conv_3/BiasAdd/ReadVariableOp═
stream_0_conv_3/BiasAddBiasAdd'stream_0_conv_3/conv1d/Squeeze:output:0.stream_0_conv_3/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:         }ђ2
stream_0_conv_3/BiasAddН
.batch_normalization_2/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_2_batchnorm_readvariableop_resource*
_output_shapes	
:ђ*
dtype020
.batch_normalization_2/batchnorm/ReadVariableOpЊ
%batch_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2'
%batch_normalization_2/batchnorm/add/yр
#batch_normalization_2/batchnorm/addAddV26batch_normalization_2/batchnorm/ReadVariableOp:value:0.batch_normalization_2/batchnorm/add/y:output:0*
T0*
_output_shapes	
:ђ2%
#batch_normalization_2/batchnorm/addд
%batch_normalization_2/batchnorm/RsqrtRsqrt'batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes	
:ђ2'
%batch_normalization_2/batchnorm/Rsqrtр
2batch_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes	
:ђ*
dtype024
2batch_normalization_2/batchnorm/mul/ReadVariableOpя
#batch_normalization_2/batchnorm/mulMul)batch_normalization_2/batchnorm/Rsqrt:y:0:batch_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:ђ2%
#batch_normalization_2/batchnorm/mulО
%batch_normalization_2/batchnorm/mul_1Mul stream_0_conv_3/BiasAdd:output:0'batch_normalization_2/batchnorm/mul:z:0*
T0*,
_output_shapes
:         }ђ2'
%batch_normalization_2/batchnorm/mul_1█
0batch_normalization_2/batchnorm/ReadVariableOp_1ReadVariableOp9batch_normalization_2_batchnorm_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype022
0batch_normalization_2/batchnorm/ReadVariableOp_1я
%batch_normalization_2/batchnorm/mul_2Mul8batch_normalization_2/batchnorm/ReadVariableOp_1:value:0'batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes	
:ђ2'
%batch_normalization_2/batchnorm/mul_2█
0batch_normalization_2/batchnorm/ReadVariableOp_2ReadVariableOp9batch_normalization_2_batchnorm_readvariableop_2_resource*
_output_shapes	
:ђ*
dtype022
0batch_normalization_2/batchnorm/ReadVariableOp_2▄
#batch_normalization_2/batchnorm/subSub8batch_normalization_2/batchnorm/ReadVariableOp_2:value:0)batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:ђ2%
#batch_normalization_2/batchnorm/subР
%batch_normalization_2/batchnorm/add_1AddV2)batch_normalization_2/batchnorm/mul_1:z:0'batch_normalization_2/batchnorm/sub:z:0*
T0*,
_output_shapes
:         }ђ2'
%batch_normalization_2/batchnorm/add_1љ
activation_2/ReluRelu)batch_normalization_2/batchnorm/add_1:z:0*
T0*,
_output_shapes
:         }ђ2
activation_2/Reluў
stream_0_drop_3/IdentityIdentityactivation_2/Relu:activations:0*
T0*,
_output_shapes
:         }ђ2
stream_0_drop_3/Identityц
/global_average_pooling1d/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :21
/global_average_pooling1d/Mean/reduction_indicesо
global_average_pooling1d/MeanMean!stream_0_drop_3/Identity:output:08global_average_pooling1d/Mean/reduction_indices:output:0*
T0*(
_output_shapes
:         ђ2
global_average_pooling1d/MeanЏ
dense_1_dropout/IdentityIdentity&global_average_pooling1d/Mean:output:0*
T0*(
_output_shapes
:         ђ2
dense_1_dropout/Identityд
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	ђT*
dtype02
dense_1/MatMul/ReadVariableOpд
dense_1/MatMulMatMul!dense_1_dropout/Identity:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         T2
dense_1/MatMulц
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:T*
dtype02 
dense_1/BiasAdd/ReadVariableOpА
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         T2
dense_1/BiasAddн
.batch_normalization_3/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_3_batchnorm_readvariableop_resource*
_output_shapes
:T*
dtype020
.batch_normalization_3/batchnorm/ReadVariableOpЊ
%batch_normalization_3/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2'
%batch_normalization_3/batchnorm/add/yЯ
#batch_normalization_3/batchnorm/addAddV26batch_normalization_3/batchnorm/ReadVariableOp:value:0.batch_normalization_3/batchnorm/add/y:output:0*
T0*
_output_shapes
:T2%
#batch_normalization_3/batchnorm/addЦ
%batch_normalization_3/batchnorm/RsqrtRsqrt'batch_normalization_3/batchnorm/add:z:0*
T0*
_output_shapes
:T2'
%batch_normalization_3/batchnorm/RsqrtЯ
2batch_normalization_3/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_3_batchnorm_mul_readvariableop_resource*
_output_shapes
:T*
dtype024
2batch_normalization_3/batchnorm/mul/ReadVariableOpП
#batch_normalization_3/batchnorm/mulMul)batch_normalization_3/batchnorm/Rsqrt:y:0:batch_normalization_3/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:T2%
#batch_normalization_3/batchnorm/mul╩
%batch_normalization_3/batchnorm/mul_1Muldense_1/BiasAdd:output:0'batch_normalization_3/batchnorm/mul:z:0*
T0*'
_output_shapes
:         T2'
%batch_normalization_3/batchnorm/mul_1┌
0batch_normalization_3/batchnorm/ReadVariableOp_1ReadVariableOp9batch_normalization_3_batchnorm_readvariableop_1_resource*
_output_shapes
:T*
dtype022
0batch_normalization_3/batchnorm/ReadVariableOp_1П
%batch_normalization_3/batchnorm/mul_2Mul8batch_normalization_3/batchnorm/ReadVariableOp_1:value:0'batch_normalization_3/batchnorm/mul:z:0*
T0*
_output_shapes
:T2'
%batch_normalization_3/batchnorm/mul_2┌
0batch_normalization_3/batchnorm/ReadVariableOp_2ReadVariableOp9batch_normalization_3_batchnorm_readvariableop_2_resource*
_output_shapes
:T*
dtype022
0batch_normalization_3/batchnorm/ReadVariableOp_2█
#batch_normalization_3/batchnorm/subSub8batch_normalization_3/batchnorm/ReadVariableOp_2:value:0)batch_normalization_3/batchnorm/mul_2:z:0*
T0*
_output_shapes
:T2%
#batch_normalization_3/batchnorm/subП
%batch_normalization_3/batchnorm/add_1AddV2)batch_normalization_3/batchnorm/mul_1:z:0'batch_normalization_3/batchnorm/sub:z:0*
T0*'
_output_shapes
:         T2'
%batch_normalization_3/batchnorm/add_1а
dense_activation_1/SigmoidSigmoid)batch_normalization_3/batchnorm/add_1:z:0*
T0*'
_output_shapes
:         T2
dense_activation_1/SigmoidЬ
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp;stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype027
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp├
&stream_0_conv_1/kernel/Regularizer/AbsAbs=stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_0_conv_1/kernel/Regularizer/AbsЕ
(stream_0_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_0_conv_1/kernel/Regularizer/ConstО
&stream_0_conv_1/kernel/Regularizer/SumSum*stream_0_conv_1/kernel/Regularizer/Abs:y:01stream_0_conv_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/SumЎ
(stream_0_conv_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2*
(stream_0_conv_1/kernel/Regularizer/mul/x▄
&stream_0_conv_1/kernel/Regularizer/mulMul1stream_0_conv_1/kernel/Regularizer/mul/x:output:0/stream_0_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/mulш
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;stream_0_conv_2_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:@ђ*
dtype02:
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOpл
)stream_0_conv_2/kernel/Regularizer/SquareSquare@stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*#
_output_shapes
:@ђ2+
)stream_0_conv_2/kernel/Regularizer/SquareЕ
(stream_0_conv_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_0_conv_2/kernel/Regularizer/Const┌
&stream_0_conv_2/kernel/Regularizer/SumSum-stream_0_conv_2/kernel/Regularizer/Square:y:01stream_0_conv_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_2/kernel/Regularizer/SumЎ
(stream_0_conv_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2*
(stream_0_conv_2/kernel/Regularizer/mul/x▄
&stream_0_conv_2/kernel/Regularizer/mulMul1stream_0_conv_2/kernel/Regularizer/mul/x:output:0/stream_0_conv_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_2/kernel/Regularizer/mul­
5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp;stream_0_conv_3_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:ђђ*
dtype027
5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp┼
&stream_0_conv_3/kernel/Regularizer/AbsAbs=stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*$
_output_shapes
:ђђ2(
&stream_0_conv_3/kernel/Regularizer/AbsЕ
(stream_0_conv_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_0_conv_3/kernel/Regularizer/ConstО
&stream_0_conv_3/kernel/Regularizer/SumSum*stream_0_conv_3/kernel/Regularizer/Abs:y:01stream_0_conv_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_3/kernel/Regularizer/SumЎ
(stream_0_conv_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2*
(stream_0_conv_3/kernel/Regularizer/mul/x▄
&stream_0_conv_3/kernel/Regularizer/mulMul1stream_0_conv_3/kernel/Regularizer/mul/x:output:0/stream_0_conv_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_3/kernel/Regularizer/mulк
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	ђT*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOpе
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	ђT2 
dense_1/kernel/Regularizer/AbsЋ
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_1/kernel/Regularizer/Constи
dense_1/kernel/Regularizer/SumSum"dense_1/kernel/Regularizer/Abs:y:0)dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/SumЅ
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2"
 dense_1/kernel/Regularizer/mul/x╝
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/muly
IdentityIdentitydense_activation_1/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:         T2

Identityг
NoOpNoOp-^batch_normalization/batchnorm/ReadVariableOp/^batch_normalization/batchnorm/ReadVariableOp_1/^batch_normalization/batchnorm/ReadVariableOp_21^batch_normalization/batchnorm/mul/ReadVariableOp/^batch_normalization_1/batchnorm/ReadVariableOp1^batch_normalization_1/batchnorm/ReadVariableOp_11^batch_normalization_1/batchnorm/ReadVariableOp_23^batch_normalization_1/batchnorm/mul/ReadVariableOp/^batch_normalization_2/batchnorm/ReadVariableOp1^batch_normalization_2/batchnorm/ReadVariableOp_11^batch_normalization_2/batchnorm/ReadVariableOp_23^batch_normalization_2/batchnorm/mul/ReadVariableOp/^batch_normalization_3/batchnorm/ReadVariableOp1^batch_normalization_3/batchnorm/ReadVariableOp_11^batch_normalization_3/batchnorm/ReadVariableOp_23^batch_normalization_3/batchnorm/mul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp.^dense_1/kernel/Regularizer/Abs/ReadVariableOp'^stream_0_conv_1/BiasAdd/ReadVariableOp3^stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp'^stream_0_conv_2/BiasAdd/ReadVariableOp3^stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOp9^stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp'^stream_0_conv_3/BiasAdd/ReadVariableOp3^stream_0_conv_3/conv1d/ExpandDims_1/ReadVariableOp6^stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:         }: : : : : : : : : : : : : : : : : : : : : : : : 2\
,batch_normalization/batchnorm/ReadVariableOp,batch_normalization/batchnorm/ReadVariableOp2`
.batch_normalization/batchnorm/ReadVariableOp_1.batch_normalization/batchnorm/ReadVariableOp_12`
.batch_normalization/batchnorm/ReadVariableOp_2.batch_normalization/batchnorm/ReadVariableOp_22d
0batch_normalization/batchnorm/mul/ReadVariableOp0batch_normalization/batchnorm/mul/ReadVariableOp2`
.batch_normalization_1/batchnorm/ReadVariableOp.batch_normalization_1/batchnorm/ReadVariableOp2d
0batch_normalization_1/batchnorm/ReadVariableOp_10batch_normalization_1/batchnorm/ReadVariableOp_12d
0batch_normalization_1/batchnorm/ReadVariableOp_20batch_normalization_1/batchnorm/ReadVariableOp_22h
2batch_normalization_1/batchnorm/mul/ReadVariableOp2batch_normalization_1/batchnorm/mul/ReadVariableOp2`
.batch_normalization_2/batchnorm/ReadVariableOp.batch_normalization_2/batchnorm/ReadVariableOp2d
0batch_normalization_2/batchnorm/ReadVariableOp_10batch_normalization_2/batchnorm/ReadVariableOp_12d
0batch_normalization_2/batchnorm/ReadVariableOp_20batch_normalization_2/batchnorm/ReadVariableOp_22h
2batch_normalization_2/batchnorm/mul/ReadVariableOp2batch_normalization_2/batchnorm/mul/ReadVariableOp2`
.batch_normalization_3/batchnorm/ReadVariableOp.batch_normalization_3/batchnorm/ReadVariableOp2d
0batch_normalization_3/batchnorm/ReadVariableOp_10batch_normalization_3/batchnorm/ReadVariableOp_12d
0batch_normalization_3/batchnorm/ReadVariableOp_20batch_normalization_3/batchnorm/ReadVariableOp_22h
2batch_normalization_3/batchnorm/mul/ReadVariableOp2batch_normalization_3/batchnorm/mul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2^
-dense_1/kernel/Regularizer/Abs/ReadVariableOp-dense_1/kernel/Regularizer/Abs/ReadVariableOp2P
&stream_0_conv_1/BiasAdd/ReadVariableOp&stream_0_conv_1/BiasAdd/ReadVariableOp2h
2stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp2stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp2n
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp2P
&stream_0_conv_2/BiasAdd/ReadVariableOp&stream_0_conv_2/BiasAdd/ReadVariableOp2h
2stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOp2stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOp2t
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp2P
&stream_0_conv_3/BiasAdd/ReadVariableOp&stream_0_conv_3/BiasAdd/ReadVariableOp2h
2stream_0_conv_3/conv1d/ExpandDims_1/ReadVariableOp2stream_0_conv_3/conv1d/ExpandDims_1/ReadVariableOp2n
5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp:S O
+
_output_shapes
:         }
 
_user_specified_nameinputs
Ё
о
J__inference_stream_0_conv_2_layer_call_and_return_conditional_losses_28354

inputsB
+conv1d_expanddims_1_readvariableop_resource:@ђ.
biasadd_readvariableop_resource:	ђ
identityѕбBiasAdd/ReadVariableOpб"conv1d/ExpandDims_1/ReadVariableOpб8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§        2
conv1d/ExpandDims/dimќ
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:         }@2
conv1d/ExpandDims╣
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:@ђ*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dimИ
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:@ђ2
conv1d/ExpandDims_1и
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:         }ђ*
paddingSAME*
strides
2
conv1dЊ
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:         }ђ*
squeeze_dims

§        2
conv1d/SqueezeЇ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
BiasAdd/ReadVariableOpЇ
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:         }ђ2	
BiasAddт
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:@ђ*
dtype02:
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOpл
)stream_0_conv_2/kernel/Regularizer/SquareSquare@stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*#
_output_shapes
:@ђ2+
)stream_0_conv_2/kernel/Regularizer/SquareЕ
(stream_0_conv_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_0_conv_2/kernel/Regularizer/Const┌
&stream_0_conv_2/kernel/Regularizer/SumSum-stream_0_conv_2/kernel/Regularizer/Square:y:01stream_0_conv_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_2/kernel/Regularizer/SumЎ
(stream_0_conv_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2*
(stream_0_conv_2/kernel/Regularizer/mul/x▄
&stream_0_conv_2/kernel/Regularizer/mulMul1stream_0_conv_2/kernel/Regularizer/mul/x:output:0/stream_0_conv_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_2/kernel/Regularizer/mulp
IdentityIdentityBiasAdd:output:0^NoOp*
T0*,
_output_shapes
:         }ђ2

IdentityК
NoOpNoOp^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp9^stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         }@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp2t
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp:S O
+
_output_shapes
:         }@
 
_user_specified_nameinputs
Ё
о
J__inference_stream_0_conv_2_layer_call_and_return_conditional_losses_23975

inputsB
+conv1d_expanddims_1_readvariableop_resource:@ђ.
biasadd_readvariableop_resource:	ђ
identityѕбBiasAdd/ReadVariableOpб"conv1d/ExpandDims_1/ReadVariableOpб8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§        2
conv1d/ExpandDims/dimќ
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:         }@2
conv1d/ExpandDims╣
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:@ђ*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dimИ
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:@ђ2
conv1d/ExpandDims_1и
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:         }ђ*
paddingSAME*
strides
2
conv1dЊ
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:         }ђ*
squeeze_dims

§        2
conv1d/SqueezeЇ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
BiasAdd/ReadVariableOpЇ
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:         }ђ2	
BiasAddт
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:@ђ*
dtype02:
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOpл
)stream_0_conv_2/kernel/Regularizer/SquareSquare@stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*#
_output_shapes
:@ђ2+
)stream_0_conv_2/kernel/Regularizer/SquareЕ
(stream_0_conv_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_0_conv_2/kernel/Regularizer/Const┌
&stream_0_conv_2/kernel/Regularizer/SumSum-stream_0_conv_2/kernel/Regularizer/Square:y:01stream_0_conv_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_2/kernel/Regularizer/SumЎ
(stream_0_conv_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2*
(stream_0_conv_2/kernel/Regularizer/mul/x▄
&stream_0_conv_2/kernel/Regularizer/mulMul1stream_0_conv_2/kernel/Regularizer/mul/x:output:0/stream_0_conv_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_2/kernel/Regularizer/mulp
IdentityIdentityBiasAdd:output:0^NoOp*
T0*,
_output_shapes
:         }ђ2

IdentityК
NoOpNoOp^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp9^stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         }@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp2t
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp:S O
+
_output_shapes
:         }@
 
_user_specified_nameinputs
╚
│
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_28659

inputs0
!batchnorm_readvariableop_resource:	ђ4
%batchnorm_mul_readvariableop_resource:	ђ2
#batchnorm_readvariableop_1_resource:	ђ2
#batchnorm_readvariableop_2_resource:	ђ
identityѕбbatchnorm/ReadVariableOpбbatchnorm/ReadVariableOp_1бbatchnorm/ReadVariableOp_2бbatchnorm/mul/ReadVariableOpЊ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2
batchnorm/add/yЅ
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:ђ2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:ђ2
batchnorm/RsqrtЪ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
batchnorm/mul/ReadVariableOpє
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:ђ2
batchnorm/mulё
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*5
_output_shapes#
!:                  ђ2
batchnorm/mul_1Ў
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype02
batchnorm/ReadVariableOp_1є
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:ђ2
batchnorm/mul_2Ў
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:ђ*
dtype02
batchnorm/ReadVariableOp_2ё
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:ђ2
batchnorm/subЊ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*5
_output_shapes#
!:                  ђ2
batchnorm/add_1|
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*5
_output_shapes#
!:                  ђ2

Identity┬
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):                  ђ: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:] Y
5
_output_shapes#
!:                  ђ
 
_user_specified_nameinputs
у
i
M__inference_dense_activation_1_layer_call_and_return_conditional_losses_24144

inputs
identityW
SigmoidSigmoidinputs*
T0*'
_output_shapes
:         T2	
Sigmoid_
IdentityIdentitySigmoid:y:0*
T0*'
_output_shapes
:         T2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         T:O K
'
_output_shapes
:         T
 
_user_specified_nameinputs
┘
Л
J__inference_stream_0_conv_1_layer_call_and_return_conditional_losses_28121

inputsA
+conv1d_expanddims_1_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identityѕбBiasAdd/ReadVariableOpб"conv1d/ExpandDims_1/ReadVariableOpб5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§        2
conv1d/ExpandDims/dimќ
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:         }2
conv1d/ExpandDimsИ
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dimи
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@2
conv1d/ExpandDims_1Х
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:         }@*
paddingSAME*
strides
2
conv1dњ
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:         }@*
squeeze_dims

§        2
conv1d/Squeezeї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpї
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         }@2	
BiasAddя
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype027
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp├
&stream_0_conv_1/kernel/Regularizer/AbsAbs=stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_0_conv_1/kernel/Regularizer/AbsЕ
(stream_0_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_0_conv_1/kernel/Regularizer/ConstО
&stream_0_conv_1/kernel/Regularizer/SumSum*stream_0_conv_1/kernel/Regularizer/Abs:y:01stream_0_conv_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/SumЎ
(stream_0_conv_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2*
(stream_0_conv_1/kernel/Regularizer/mul/x▄
&stream_0_conv_1/kernel/Regularizer/mulMul1stream_0_conv_1/kernel/Regularizer/mul/x:output:0/stream_0_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/mulo
IdentityIdentityBiasAdd:output:0^NoOp*
T0*+
_output_shapes
:         }@2

Identity─
NoOpNoOp^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         }: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp2n
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:S O
+
_output_shapes
:         }
 
_user_specified_nameinputs
Є
h
J__inference_stream_0_drop_1_layer_call_and_return_conditional_losses_23952

inputs

identity_1^
IdentityIdentityinputs*
T0*+
_output_shapes
:         }@2

Identitym

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:         }@2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         }@:S O
+
_output_shapes
:         }@
 
_user_specified_nameinputs
І
h
J__inference_stream_0_drop_2_layer_call_and_return_conditional_losses_28539

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:         }ђ2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:         }ђ2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         }ђ:T P
,
_output_shapes
:         }ђ
 
_user_specified_nameinputs
ЃЂ
у
__inference__traced_save_29211
file_prefix%
!savev2_beta_1_read_readvariableop%
!savev2_beta_2_read_readvariableop$
 savev2_decay_read_readvariableop,
(savev2_learning_rate_read_readvariableop(
$savev2_adam_iter_read_readvariableop	5
1savev2_stream_0_conv_1_kernel_read_readvariableop3
/savev2_stream_0_conv_1_bias_read_readvariableop8
4savev2_batch_normalization_gamma_read_readvariableop7
3savev2_batch_normalization_beta_read_readvariableop5
1savev2_stream_0_conv_2_kernel_read_readvariableop3
/savev2_stream_0_conv_2_bias_read_readvariableop:
6savev2_batch_normalization_1_gamma_read_readvariableop9
5savev2_batch_normalization_1_beta_read_readvariableop5
1savev2_stream_0_conv_3_kernel_read_readvariableop3
/savev2_stream_0_conv_3_bias_read_readvariableop:
6savev2_batch_normalization_2_gamma_read_readvariableop9
5savev2_batch_normalization_2_beta_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop:
6savev2_batch_normalization_3_gamma_read_readvariableop9
5savev2_batch_normalization_3_beta_read_readvariableop>
:savev2_batch_normalization_moving_mean_read_readvariableopB
>savev2_batch_normalization_moving_variance_read_readvariableop@
<savev2_batch_normalization_1_moving_mean_read_readvariableopD
@savev2_batch_normalization_1_moving_variance_read_readvariableop@
<savev2_batch_normalization_2_moving_mean_read_readvariableopD
@savev2_batch_normalization_2_moving_variance_read_readvariableop@
<savev2_batch_normalization_3_moving_mean_read_readvariableopD
@savev2_batch_normalization_3_moving_variance_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop<
8savev2_adam_stream_0_conv_1_kernel_m_read_readvariableop:
6savev2_adam_stream_0_conv_1_bias_m_read_readvariableop?
;savev2_adam_batch_normalization_gamma_m_read_readvariableop>
:savev2_adam_batch_normalization_beta_m_read_readvariableop<
8savev2_adam_stream_0_conv_2_kernel_m_read_readvariableop:
6savev2_adam_stream_0_conv_2_bias_m_read_readvariableopA
=savev2_adam_batch_normalization_1_gamma_m_read_readvariableop@
<savev2_adam_batch_normalization_1_beta_m_read_readvariableop<
8savev2_adam_stream_0_conv_3_kernel_m_read_readvariableop:
6savev2_adam_stream_0_conv_3_bias_m_read_readvariableopA
=savev2_adam_batch_normalization_2_gamma_m_read_readvariableop@
<savev2_adam_batch_normalization_2_beta_m_read_readvariableop4
0savev2_adam_dense_1_kernel_m_read_readvariableop2
.savev2_adam_dense_1_bias_m_read_readvariableopA
=savev2_adam_batch_normalization_3_gamma_m_read_readvariableop@
<savev2_adam_batch_normalization_3_beta_m_read_readvariableop<
8savev2_adam_stream_0_conv_1_kernel_v_read_readvariableop:
6savev2_adam_stream_0_conv_1_bias_v_read_readvariableop?
;savev2_adam_batch_normalization_gamma_v_read_readvariableop>
:savev2_adam_batch_normalization_beta_v_read_readvariableop<
8savev2_adam_stream_0_conv_2_kernel_v_read_readvariableop:
6savev2_adam_stream_0_conv_2_bias_v_read_readvariableopA
=savev2_adam_batch_normalization_1_gamma_v_read_readvariableop@
<savev2_adam_batch_normalization_1_beta_v_read_readvariableop<
8savev2_adam_stream_0_conv_3_kernel_v_read_readvariableop:
6savev2_adam_stream_0_conv_3_bias_v_read_readvariableopA
=savev2_adam_batch_normalization_2_gamma_v_read_readvariableop@
<savev2_adam_batch_normalization_2_beta_v_read_readvariableop4
0savev2_adam_dense_1_kernel_v_read_readvariableop2
.savev2_adam_dense_1_bias_v_read_readvariableopA
=savev2_adam_batch_normalization_3_gamma_v_read_readvariableop@
<savev2_adam_batch_normalization_3_beta_v_read_readvariableop
savev2_const

identity_1ѕбMergeV2CheckpointsЈ
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1І
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shardд
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameф 
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:@*
dtype0*╝
value▓B»@B+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/14/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/15/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesІ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:@*
dtype0*Ћ
valueІBѕ@B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesщ
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0!savev2_beta_1_read_readvariableop!savev2_beta_2_read_readvariableop savev2_decay_read_readvariableop(savev2_learning_rate_read_readvariableop$savev2_adam_iter_read_readvariableop1savev2_stream_0_conv_1_kernel_read_readvariableop/savev2_stream_0_conv_1_bias_read_readvariableop4savev2_batch_normalization_gamma_read_readvariableop3savev2_batch_normalization_beta_read_readvariableop1savev2_stream_0_conv_2_kernel_read_readvariableop/savev2_stream_0_conv_2_bias_read_readvariableop6savev2_batch_normalization_1_gamma_read_readvariableop5savev2_batch_normalization_1_beta_read_readvariableop1savev2_stream_0_conv_3_kernel_read_readvariableop/savev2_stream_0_conv_3_bias_read_readvariableop6savev2_batch_normalization_2_gamma_read_readvariableop5savev2_batch_normalization_2_beta_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop6savev2_batch_normalization_3_gamma_read_readvariableop5savev2_batch_normalization_3_beta_read_readvariableop:savev2_batch_normalization_moving_mean_read_readvariableop>savev2_batch_normalization_moving_variance_read_readvariableop<savev2_batch_normalization_1_moving_mean_read_readvariableop@savev2_batch_normalization_1_moving_variance_read_readvariableop<savev2_batch_normalization_2_moving_mean_read_readvariableop@savev2_batch_normalization_2_moving_variance_read_readvariableop<savev2_batch_normalization_3_moving_mean_read_readvariableop@savev2_batch_normalization_3_moving_variance_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop8savev2_adam_stream_0_conv_1_kernel_m_read_readvariableop6savev2_adam_stream_0_conv_1_bias_m_read_readvariableop;savev2_adam_batch_normalization_gamma_m_read_readvariableop:savev2_adam_batch_normalization_beta_m_read_readvariableop8savev2_adam_stream_0_conv_2_kernel_m_read_readvariableop6savev2_adam_stream_0_conv_2_bias_m_read_readvariableop=savev2_adam_batch_normalization_1_gamma_m_read_readvariableop<savev2_adam_batch_normalization_1_beta_m_read_readvariableop8savev2_adam_stream_0_conv_3_kernel_m_read_readvariableop6savev2_adam_stream_0_conv_3_bias_m_read_readvariableop=savev2_adam_batch_normalization_2_gamma_m_read_readvariableop<savev2_adam_batch_normalization_2_beta_m_read_readvariableop0savev2_adam_dense_1_kernel_m_read_readvariableop.savev2_adam_dense_1_bias_m_read_readvariableop=savev2_adam_batch_normalization_3_gamma_m_read_readvariableop<savev2_adam_batch_normalization_3_beta_m_read_readvariableop8savev2_adam_stream_0_conv_1_kernel_v_read_readvariableop6savev2_adam_stream_0_conv_1_bias_v_read_readvariableop;savev2_adam_batch_normalization_gamma_v_read_readvariableop:savev2_adam_batch_normalization_beta_v_read_readvariableop8savev2_adam_stream_0_conv_2_kernel_v_read_readvariableop6savev2_adam_stream_0_conv_2_bias_v_read_readvariableop=savev2_adam_batch_normalization_1_gamma_v_read_readvariableop<savev2_adam_batch_normalization_1_beta_v_read_readvariableop8savev2_adam_stream_0_conv_3_kernel_v_read_readvariableop6savev2_adam_stream_0_conv_3_bias_v_read_readvariableop=savev2_adam_batch_normalization_2_gamma_v_read_readvariableop<savev2_adam_batch_normalization_2_beta_v_read_readvariableop0savev2_adam_dense_1_kernel_v_read_readvariableop.savev2_adam_dense_1_bias_v_read_readvariableop=savev2_adam_batch_normalization_3_gamma_v_read_readvariableop<savev2_adam_batch_normalization_3_beta_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *N
dtypesD
B2@	2
SaveV2║
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesА
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: 2

Identity_1c
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"!

identity_1Identity_1:output:0*ь
_input_shapes█
п: : : : : : :@:@:@:@:@ђ:ђ:ђ:ђ:ђђ:ђ:ђ:ђ:	ђT:T:T:T:@:@:ђ:ђ:ђ:ђ:T:T: : :@:@:@:@:@ђ:ђ:ђ:ђ:ђђ:ђ:ђ:ђ:	ђT:T:T:T:@:@:@:@:@ђ:ђ:ђ:ђ:ђђ:ђ:ђ:ђ:	ђT:T:T:T: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :($
"
_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 	

_output_shapes
:@:)
%
#
_output_shapes
:@ђ:!

_output_shapes	
:ђ:!

_output_shapes	
:ђ:!

_output_shapes	
:ђ:*&
$
_output_shapes
:ђђ:!

_output_shapes	
:ђ:!

_output_shapes	
:ђ:!

_output_shapes	
:ђ:%!

_output_shapes
:	ђT: 

_output_shapes
:T: 

_output_shapes
:T: 

_output_shapes
:T: 

_output_shapes
:@: 

_output_shapes
:@:!

_output_shapes	
:ђ:!

_output_shapes	
:ђ:!

_output_shapes	
:ђ:!

_output_shapes	
:ђ: 

_output_shapes
:T: 

_output_shapes
:T:

_output_shapes
: :

_output_shapes
: :( $
"
_output_shapes
:@: !

_output_shapes
:@: "

_output_shapes
:@: #

_output_shapes
:@:)$%
#
_output_shapes
:@ђ:!%

_output_shapes	
:ђ:!&

_output_shapes	
:ђ:!'

_output_shapes	
:ђ:*(&
$
_output_shapes
:ђђ:!)

_output_shapes	
:ђ:!*

_output_shapes	
:ђ:!+

_output_shapes	
:ђ:%,!

_output_shapes
:	ђT: -

_output_shapes
:T: .

_output_shapes
:T: /

_output_shapes
:T:(0$
"
_output_shapes
:@: 1

_output_shapes
:@: 2

_output_shapes
:@: 3

_output_shapes
:@:)4%
#
_output_shapes
:@ђ:!5

_output_shapes	
:ђ:!6

_output_shapes	
:ђ:!7

_output_shapes	
:ђ:*8&
$
_output_shapes
:ђђ:!9

_output_shapes	
:ђ:!:

_output_shapes	
:ђ:!;

_output_shapes	
:ђ:%<!

_output_shapes
:	ђT: =

_output_shapes
:T: >

_output_shapes
:T: ?

_output_shapes
:T:@

_output_shapes
: 
Ѓ№
ђ
D__inference_basemodel_layer_call_and_return_conditional_losses_27789
inputs_0Q
;stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource:@=
/stream_0_conv_1_biasadd_readvariableop_resource:@C
5batch_normalization_batchnorm_readvariableop_resource:@G
9batch_normalization_batchnorm_mul_readvariableop_resource:@E
7batch_normalization_batchnorm_readvariableop_1_resource:@E
7batch_normalization_batchnorm_readvariableop_2_resource:@R
;stream_0_conv_2_conv1d_expanddims_1_readvariableop_resource:@ђ>
/stream_0_conv_2_biasadd_readvariableop_resource:	ђF
7batch_normalization_1_batchnorm_readvariableop_resource:	ђJ
;batch_normalization_1_batchnorm_mul_readvariableop_resource:	ђH
9batch_normalization_1_batchnorm_readvariableop_1_resource:	ђH
9batch_normalization_1_batchnorm_readvariableop_2_resource:	ђS
;stream_0_conv_3_conv1d_expanddims_1_readvariableop_resource:ђђ>
/stream_0_conv_3_biasadd_readvariableop_resource:	ђF
7batch_normalization_2_batchnorm_readvariableop_resource:	ђJ
;batch_normalization_2_batchnorm_mul_readvariableop_resource:	ђH
9batch_normalization_2_batchnorm_readvariableop_1_resource:	ђH
9batch_normalization_2_batchnorm_readvariableop_2_resource:	ђ9
&dense_1_matmul_readvariableop_resource:	ђT5
'dense_1_biasadd_readvariableop_resource:TE
7batch_normalization_3_batchnorm_readvariableop_resource:TI
;batch_normalization_3_batchnorm_mul_readvariableop_resource:TG
9batch_normalization_3_batchnorm_readvariableop_1_resource:TG
9batch_normalization_3_batchnorm_readvariableop_2_resource:T
identityѕб,batch_normalization/batchnorm/ReadVariableOpб.batch_normalization/batchnorm/ReadVariableOp_1б.batch_normalization/batchnorm/ReadVariableOp_2б0batch_normalization/batchnorm/mul/ReadVariableOpб.batch_normalization_1/batchnorm/ReadVariableOpб0batch_normalization_1/batchnorm/ReadVariableOp_1б0batch_normalization_1/batchnorm/ReadVariableOp_2б2batch_normalization_1/batchnorm/mul/ReadVariableOpб.batch_normalization_2/batchnorm/ReadVariableOpб0batch_normalization_2/batchnorm/ReadVariableOp_1б0batch_normalization_2/batchnorm/ReadVariableOp_2б2batch_normalization_2/batchnorm/mul/ReadVariableOpб.batch_normalization_3/batchnorm/ReadVariableOpб0batch_normalization_3/batchnorm/ReadVariableOp_1б0batch_normalization_3/batchnorm/ReadVariableOp_2б2batch_normalization_3/batchnorm/mul/ReadVariableOpбdense_1/BiasAdd/ReadVariableOpбdense_1/MatMul/ReadVariableOpб-dense_1/kernel/Regularizer/Abs/ReadVariableOpб&stream_0_conv_1/BiasAdd/ReadVariableOpб2stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOpб5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpб&stream_0_conv_2/BiasAdd/ReadVariableOpб2stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOpб8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOpб&stream_0_conv_3/BiasAdd/ReadVariableOpб2stream_0_conv_3/conv1d/ExpandDims_1/ReadVariableOpб5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOpѕ
stream_0_input_drop/IdentityIdentityinputs_0*
T0*+
_output_shapes
:         }2
stream_0_input_drop/IdentityЎ
%stream_0_conv_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§        2'
%stream_0_conv_1/conv1d/ExpandDims/dimт
!stream_0_conv_1/conv1d/ExpandDims
ExpandDims%stream_0_input_drop/Identity:output:0.stream_0_conv_1/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:         }2#
!stream_0_conv_1/conv1d/ExpandDimsУ
2stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp;stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype024
2stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOpћ
'stream_0_conv_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2)
'stream_0_conv_1/conv1d/ExpandDims_1/dimэ
#stream_0_conv_1/conv1d/ExpandDims_1
ExpandDims:stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp:value:00stream_0_conv_1/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@2%
#stream_0_conv_1/conv1d/ExpandDims_1Ш
stream_0_conv_1/conv1dConv2D*stream_0_conv_1/conv1d/ExpandDims:output:0,stream_0_conv_1/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:         }@*
paddingSAME*
strides
2
stream_0_conv_1/conv1d┬
stream_0_conv_1/conv1d/SqueezeSqueezestream_0_conv_1/conv1d:output:0*
T0*+
_output_shapes
:         }@*
squeeze_dims

§        2 
stream_0_conv_1/conv1d/Squeeze╝
&stream_0_conv_1/BiasAdd/ReadVariableOpReadVariableOp/stream_0_conv_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02(
&stream_0_conv_1/BiasAdd/ReadVariableOp╠
stream_0_conv_1/BiasAddBiasAdd'stream_0_conv_1/conv1d/Squeeze:output:0.stream_0_conv_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         }@2
stream_0_conv_1/BiasAdd╬
,batch_normalization/batchnorm/ReadVariableOpReadVariableOp5batch_normalization_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02.
,batch_normalization/batchnorm/ReadVariableOpЈ
#batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2%
#batch_normalization/batchnorm/add/yп
!batch_normalization/batchnorm/addAddV24batch_normalization/batchnorm/ReadVariableOp:value:0,batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes
:@2#
!batch_normalization/batchnorm/addЪ
#batch_normalization/batchnorm/RsqrtRsqrt%batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes
:@2%
#batch_normalization/batchnorm/Rsqrt┌
0batch_normalization/batchnorm/mul/ReadVariableOpReadVariableOp9batch_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype022
0batch_normalization/batchnorm/mul/ReadVariableOpН
!batch_normalization/batchnorm/mulMul'batch_normalization/batchnorm/Rsqrt:y:08batch_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2#
!batch_normalization/batchnorm/mulл
#batch_normalization/batchnorm/mul_1Mul stream_0_conv_1/BiasAdd:output:0%batch_normalization/batchnorm/mul:z:0*
T0*+
_output_shapes
:         }@2%
#batch_normalization/batchnorm/mul_1н
.batch_normalization/batchnorm/ReadVariableOp_1ReadVariableOp7batch_normalization_batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype020
.batch_normalization/batchnorm/ReadVariableOp_1Н
#batch_normalization/batchnorm/mul_2Mul6batch_normalization/batchnorm/ReadVariableOp_1:value:0%batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes
:@2%
#batch_normalization/batchnorm/mul_2н
.batch_normalization/batchnorm/ReadVariableOp_2ReadVariableOp7batch_normalization_batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype020
.batch_normalization/batchnorm/ReadVariableOp_2М
!batch_normalization/batchnorm/subSub6batch_normalization/batchnorm/ReadVariableOp_2:value:0'batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2#
!batch_normalization/batchnorm/sub┘
#batch_normalization/batchnorm/add_1AddV2'batch_normalization/batchnorm/mul_1:z:0%batch_normalization/batchnorm/sub:z:0*
T0*+
_output_shapes
:         }@2%
#batch_normalization/batchnorm/add_1Ѕ
activation/ReluRelu'batch_normalization/batchnorm/add_1:z:0*
T0*+
_output_shapes
:         }@2
activation/ReluЋ
stream_0_drop_1/IdentityIdentityactivation/Relu:activations:0*
T0*+
_output_shapes
:         }@2
stream_0_drop_1/IdentityЎ
%stream_0_conv_2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§        2'
%stream_0_conv_2/conv1d/ExpandDims/dimр
!stream_0_conv_2/conv1d/ExpandDims
ExpandDims!stream_0_drop_1/Identity:output:0.stream_0_conv_2/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:         }@2#
!stream_0_conv_2/conv1d/ExpandDimsж
2stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp;stream_0_conv_2_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:@ђ*
dtype024
2stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOpћ
'stream_0_conv_2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2)
'stream_0_conv_2/conv1d/ExpandDims_1/dimЭ
#stream_0_conv_2/conv1d/ExpandDims_1
ExpandDims:stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOp:value:00stream_0_conv_2/conv1d/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:@ђ2%
#stream_0_conv_2/conv1d/ExpandDims_1э
stream_0_conv_2/conv1dConv2D*stream_0_conv_2/conv1d/ExpandDims:output:0,stream_0_conv_2/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:         }ђ*
paddingSAME*
strides
2
stream_0_conv_2/conv1d├
stream_0_conv_2/conv1d/SqueezeSqueezestream_0_conv_2/conv1d:output:0*
T0*,
_output_shapes
:         }ђ*
squeeze_dims

§        2 
stream_0_conv_2/conv1d/Squeezeй
&stream_0_conv_2/BiasAdd/ReadVariableOpReadVariableOp/stream_0_conv_2_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02(
&stream_0_conv_2/BiasAdd/ReadVariableOp═
stream_0_conv_2/BiasAddBiasAdd'stream_0_conv_2/conv1d/Squeeze:output:0.stream_0_conv_2/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:         }ђ2
stream_0_conv_2/BiasAddН
.batch_normalization_1/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_1_batchnorm_readvariableop_resource*
_output_shapes	
:ђ*
dtype020
.batch_normalization_1/batchnorm/ReadVariableOpЊ
%batch_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2'
%batch_normalization_1/batchnorm/add/yр
#batch_normalization_1/batchnorm/addAddV26batch_normalization_1/batchnorm/ReadVariableOp:value:0.batch_normalization_1/batchnorm/add/y:output:0*
T0*
_output_shapes	
:ђ2%
#batch_normalization_1/batchnorm/addд
%batch_normalization_1/batchnorm/RsqrtRsqrt'batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes	
:ђ2'
%batch_normalization_1/batchnorm/Rsqrtр
2batch_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes	
:ђ*
dtype024
2batch_normalization_1/batchnorm/mul/ReadVariableOpя
#batch_normalization_1/batchnorm/mulMul)batch_normalization_1/batchnorm/Rsqrt:y:0:batch_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:ђ2%
#batch_normalization_1/batchnorm/mulО
%batch_normalization_1/batchnorm/mul_1Mul stream_0_conv_2/BiasAdd:output:0'batch_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:         }ђ2'
%batch_normalization_1/batchnorm/mul_1█
0batch_normalization_1/batchnorm/ReadVariableOp_1ReadVariableOp9batch_normalization_1_batchnorm_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype022
0batch_normalization_1/batchnorm/ReadVariableOp_1я
%batch_normalization_1/batchnorm/mul_2Mul8batch_normalization_1/batchnorm/ReadVariableOp_1:value:0'batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes	
:ђ2'
%batch_normalization_1/batchnorm/mul_2█
0batch_normalization_1/batchnorm/ReadVariableOp_2ReadVariableOp9batch_normalization_1_batchnorm_readvariableop_2_resource*
_output_shapes	
:ђ*
dtype022
0batch_normalization_1/batchnorm/ReadVariableOp_2▄
#batch_normalization_1/batchnorm/subSub8batch_normalization_1/batchnorm/ReadVariableOp_2:value:0)batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:ђ2%
#batch_normalization_1/batchnorm/subР
%batch_normalization_1/batchnorm/add_1AddV2)batch_normalization_1/batchnorm/mul_1:z:0'batch_normalization_1/batchnorm/sub:z:0*
T0*,
_output_shapes
:         }ђ2'
%batch_normalization_1/batchnorm/add_1љ
activation_1/ReluRelu)batch_normalization_1/batchnorm/add_1:z:0*
T0*,
_output_shapes
:         }ђ2
activation_1/Reluў
stream_0_drop_2/IdentityIdentityactivation_1/Relu:activations:0*
T0*,
_output_shapes
:         }ђ2
stream_0_drop_2/IdentityЎ
%stream_0_conv_3/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§        2'
%stream_0_conv_3/conv1d/ExpandDims/dimР
!stream_0_conv_3/conv1d/ExpandDims
ExpandDims!stream_0_drop_2/Identity:output:0.stream_0_conv_3/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         }ђ2#
!stream_0_conv_3/conv1d/ExpandDimsЖ
2stream_0_conv_3/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp;stream_0_conv_3_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:ђђ*
dtype024
2stream_0_conv_3/conv1d/ExpandDims_1/ReadVariableOpћ
'stream_0_conv_3/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2)
'stream_0_conv_3/conv1d/ExpandDims_1/dimщ
#stream_0_conv_3/conv1d/ExpandDims_1
ExpandDims:stream_0_conv_3/conv1d/ExpandDims_1/ReadVariableOp:value:00stream_0_conv_3/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:ђђ2%
#stream_0_conv_3/conv1d/ExpandDims_1э
stream_0_conv_3/conv1dConv2D*stream_0_conv_3/conv1d/ExpandDims:output:0,stream_0_conv_3/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:         }ђ*
paddingSAME*
strides
2
stream_0_conv_3/conv1d├
stream_0_conv_3/conv1d/SqueezeSqueezestream_0_conv_3/conv1d:output:0*
T0*,
_output_shapes
:         }ђ*
squeeze_dims

§        2 
stream_0_conv_3/conv1d/Squeezeй
&stream_0_conv_3/BiasAdd/ReadVariableOpReadVariableOp/stream_0_conv_3_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02(
&stream_0_conv_3/BiasAdd/ReadVariableOp═
stream_0_conv_3/BiasAddBiasAdd'stream_0_conv_3/conv1d/Squeeze:output:0.stream_0_conv_3/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:         }ђ2
stream_0_conv_3/BiasAddН
.batch_normalization_2/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_2_batchnorm_readvariableop_resource*
_output_shapes	
:ђ*
dtype020
.batch_normalization_2/batchnorm/ReadVariableOpЊ
%batch_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2'
%batch_normalization_2/batchnorm/add/yр
#batch_normalization_2/batchnorm/addAddV26batch_normalization_2/batchnorm/ReadVariableOp:value:0.batch_normalization_2/batchnorm/add/y:output:0*
T0*
_output_shapes	
:ђ2%
#batch_normalization_2/batchnorm/addд
%batch_normalization_2/batchnorm/RsqrtRsqrt'batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes	
:ђ2'
%batch_normalization_2/batchnorm/Rsqrtр
2batch_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes	
:ђ*
dtype024
2batch_normalization_2/batchnorm/mul/ReadVariableOpя
#batch_normalization_2/batchnorm/mulMul)batch_normalization_2/batchnorm/Rsqrt:y:0:batch_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:ђ2%
#batch_normalization_2/batchnorm/mulО
%batch_normalization_2/batchnorm/mul_1Mul stream_0_conv_3/BiasAdd:output:0'batch_normalization_2/batchnorm/mul:z:0*
T0*,
_output_shapes
:         }ђ2'
%batch_normalization_2/batchnorm/mul_1█
0batch_normalization_2/batchnorm/ReadVariableOp_1ReadVariableOp9batch_normalization_2_batchnorm_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype022
0batch_normalization_2/batchnorm/ReadVariableOp_1я
%batch_normalization_2/batchnorm/mul_2Mul8batch_normalization_2/batchnorm/ReadVariableOp_1:value:0'batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes	
:ђ2'
%batch_normalization_2/batchnorm/mul_2█
0batch_normalization_2/batchnorm/ReadVariableOp_2ReadVariableOp9batch_normalization_2_batchnorm_readvariableop_2_resource*
_output_shapes	
:ђ*
dtype022
0batch_normalization_2/batchnorm/ReadVariableOp_2▄
#batch_normalization_2/batchnorm/subSub8batch_normalization_2/batchnorm/ReadVariableOp_2:value:0)batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:ђ2%
#batch_normalization_2/batchnorm/subР
%batch_normalization_2/batchnorm/add_1AddV2)batch_normalization_2/batchnorm/mul_1:z:0'batch_normalization_2/batchnorm/sub:z:0*
T0*,
_output_shapes
:         }ђ2'
%batch_normalization_2/batchnorm/add_1љ
activation_2/ReluRelu)batch_normalization_2/batchnorm/add_1:z:0*
T0*,
_output_shapes
:         }ђ2
activation_2/Reluў
stream_0_drop_3/IdentityIdentityactivation_2/Relu:activations:0*
T0*,
_output_shapes
:         }ђ2
stream_0_drop_3/Identityц
/global_average_pooling1d/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :21
/global_average_pooling1d/Mean/reduction_indicesо
global_average_pooling1d/MeanMean!stream_0_drop_3/Identity:output:08global_average_pooling1d/Mean/reduction_indices:output:0*
T0*(
_output_shapes
:         ђ2
global_average_pooling1d/MeanЏ
dense_1_dropout/IdentityIdentity&global_average_pooling1d/Mean:output:0*
T0*(
_output_shapes
:         ђ2
dense_1_dropout/Identityд
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	ђT*
dtype02
dense_1/MatMul/ReadVariableOpд
dense_1/MatMulMatMul!dense_1_dropout/Identity:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         T2
dense_1/MatMulц
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:T*
dtype02 
dense_1/BiasAdd/ReadVariableOpА
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         T2
dense_1/BiasAddн
.batch_normalization_3/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_3_batchnorm_readvariableop_resource*
_output_shapes
:T*
dtype020
.batch_normalization_3/batchnorm/ReadVariableOpЊ
%batch_normalization_3/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2'
%batch_normalization_3/batchnorm/add/yЯ
#batch_normalization_3/batchnorm/addAddV26batch_normalization_3/batchnorm/ReadVariableOp:value:0.batch_normalization_3/batchnorm/add/y:output:0*
T0*
_output_shapes
:T2%
#batch_normalization_3/batchnorm/addЦ
%batch_normalization_3/batchnorm/RsqrtRsqrt'batch_normalization_3/batchnorm/add:z:0*
T0*
_output_shapes
:T2'
%batch_normalization_3/batchnorm/RsqrtЯ
2batch_normalization_3/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_3_batchnorm_mul_readvariableop_resource*
_output_shapes
:T*
dtype024
2batch_normalization_3/batchnorm/mul/ReadVariableOpП
#batch_normalization_3/batchnorm/mulMul)batch_normalization_3/batchnorm/Rsqrt:y:0:batch_normalization_3/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:T2%
#batch_normalization_3/batchnorm/mul╩
%batch_normalization_3/batchnorm/mul_1Muldense_1/BiasAdd:output:0'batch_normalization_3/batchnorm/mul:z:0*
T0*'
_output_shapes
:         T2'
%batch_normalization_3/batchnorm/mul_1┌
0batch_normalization_3/batchnorm/ReadVariableOp_1ReadVariableOp9batch_normalization_3_batchnorm_readvariableop_1_resource*
_output_shapes
:T*
dtype022
0batch_normalization_3/batchnorm/ReadVariableOp_1П
%batch_normalization_3/batchnorm/mul_2Mul8batch_normalization_3/batchnorm/ReadVariableOp_1:value:0'batch_normalization_3/batchnorm/mul:z:0*
T0*
_output_shapes
:T2'
%batch_normalization_3/batchnorm/mul_2┌
0batch_normalization_3/batchnorm/ReadVariableOp_2ReadVariableOp9batch_normalization_3_batchnorm_readvariableop_2_resource*
_output_shapes
:T*
dtype022
0batch_normalization_3/batchnorm/ReadVariableOp_2█
#batch_normalization_3/batchnorm/subSub8batch_normalization_3/batchnorm/ReadVariableOp_2:value:0)batch_normalization_3/batchnorm/mul_2:z:0*
T0*
_output_shapes
:T2%
#batch_normalization_3/batchnorm/subП
%batch_normalization_3/batchnorm/add_1AddV2)batch_normalization_3/batchnorm/mul_1:z:0'batch_normalization_3/batchnorm/sub:z:0*
T0*'
_output_shapes
:         T2'
%batch_normalization_3/batchnorm/add_1а
dense_activation_1/SigmoidSigmoid)batch_normalization_3/batchnorm/add_1:z:0*
T0*'
_output_shapes
:         T2
dense_activation_1/SigmoidЬ
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp;stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype027
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp├
&stream_0_conv_1/kernel/Regularizer/AbsAbs=stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_0_conv_1/kernel/Regularizer/AbsЕ
(stream_0_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_0_conv_1/kernel/Regularizer/ConstО
&stream_0_conv_1/kernel/Regularizer/SumSum*stream_0_conv_1/kernel/Regularizer/Abs:y:01stream_0_conv_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/SumЎ
(stream_0_conv_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2*
(stream_0_conv_1/kernel/Regularizer/mul/x▄
&stream_0_conv_1/kernel/Regularizer/mulMul1stream_0_conv_1/kernel/Regularizer/mul/x:output:0/stream_0_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/mulш
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;stream_0_conv_2_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:@ђ*
dtype02:
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOpл
)stream_0_conv_2/kernel/Regularizer/SquareSquare@stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*#
_output_shapes
:@ђ2+
)stream_0_conv_2/kernel/Regularizer/SquareЕ
(stream_0_conv_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_0_conv_2/kernel/Regularizer/Const┌
&stream_0_conv_2/kernel/Regularizer/SumSum-stream_0_conv_2/kernel/Regularizer/Square:y:01stream_0_conv_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_2/kernel/Regularizer/SumЎ
(stream_0_conv_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2*
(stream_0_conv_2/kernel/Regularizer/mul/x▄
&stream_0_conv_2/kernel/Regularizer/mulMul1stream_0_conv_2/kernel/Regularizer/mul/x:output:0/stream_0_conv_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_2/kernel/Regularizer/mul­
5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp;stream_0_conv_3_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:ђђ*
dtype027
5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp┼
&stream_0_conv_3/kernel/Regularizer/AbsAbs=stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*$
_output_shapes
:ђђ2(
&stream_0_conv_3/kernel/Regularizer/AbsЕ
(stream_0_conv_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_0_conv_3/kernel/Regularizer/ConstО
&stream_0_conv_3/kernel/Regularizer/SumSum*stream_0_conv_3/kernel/Regularizer/Abs:y:01stream_0_conv_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_3/kernel/Regularizer/SumЎ
(stream_0_conv_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2*
(stream_0_conv_3/kernel/Regularizer/mul/x▄
&stream_0_conv_3/kernel/Regularizer/mulMul1stream_0_conv_3/kernel/Regularizer/mul/x:output:0/stream_0_conv_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_3/kernel/Regularizer/mulк
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	ђT*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOpе
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	ђT2 
dense_1/kernel/Regularizer/AbsЋ
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_1/kernel/Regularizer/Constи
dense_1/kernel/Regularizer/SumSum"dense_1/kernel/Regularizer/Abs:y:0)dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/SumЅ
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2"
 dense_1/kernel/Regularizer/mul/x╝
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/muly
IdentityIdentitydense_activation_1/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:         T2

Identityг
NoOpNoOp-^batch_normalization/batchnorm/ReadVariableOp/^batch_normalization/batchnorm/ReadVariableOp_1/^batch_normalization/batchnorm/ReadVariableOp_21^batch_normalization/batchnorm/mul/ReadVariableOp/^batch_normalization_1/batchnorm/ReadVariableOp1^batch_normalization_1/batchnorm/ReadVariableOp_11^batch_normalization_1/batchnorm/ReadVariableOp_23^batch_normalization_1/batchnorm/mul/ReadVariableOp/^batch_normalization_2/batchnorm/ReadVariableOp1^batch_normalization_2/batchnorm/ReadVariableOp_11^batch_normalization_2/batchnorm/ReadVariableOp_23^batch_normalization_2/batchnorm/mul/ReadVariableOp/^batch_normalization_3/batchnorm/ReadVariableOp1^batch_normalization_3/batchnorm/ReadVariableOp_11^batch_normalization_3/batchnorm/ReadVariableOp_23^batch_normalization_3/batchnorm/mul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp.^dense_1/kernel/Regularizer/Abs/ReadVariableOp'^stream_0_conv_1/BiasAdd/ReadVariableOp3^stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp'^stream_0_conv_2/BiasAdd/ReadVariableOp3^stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOp9^stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp'^stream_0_conv_3/BiasAdd/ReadVariableOp3^stream_0_conv_3/conv1d/ExpandDims_1/ReadVariableOp6^stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:         }: : : : : : : : : : : : : : : : : : : : : : : : 2\
,batch_normalization/batchnorm/ReadVariableOp,batch_normalization/batchnorm/ReadVariableOp2`
.batch_normalization/batchnorm/ReadVariableOp_1.batch_normalization/batchnorm/ReadVariableOp_12`
.batch_normalization/batchnorm/ReadVariableOp_2.batch_normalization/batchnorm/ReadVariableOp_22d
0batch_normalization/batchnorm/mul/ReadVariableOp0batch_normalization/batchnorm/mul/ReadVariableOp2`
.batch_normalization_1/batchnorm/ReadVariableOp.batch_normalization_1/batchnorm/ReadVariableOp2d
0batch_normalization_1/batchnorm/ReadVariableOp_10batch_normalization_1/batchnorm/ReadVariableOp_12d
0batch_normalization_1/batchnorm/ReadVariableOp_20batch_normalization_1/batchnorm/ReadVariableOp_22h
2batch_normalization_1/batchnorm/mul/ReadVariableOp2batch_normalization_1/batchnorm/mul/ReadVariableOp2`
.batch_normalization_2/batchnorm/ReadVariableOp.batch_normalization_2/batchnorm/ReadVariableOp2d
0batch_normalization_2/batchnorm/ReadVariableOp_10batch_normalization_2/batchnorm/ReadVariableOp_12d
0batch_normalization_2/batchnorm/ReadVariableOp_20batch_normalization_2/batchnorm/ReadVariableOp_22h
2batch_normalization_2/batchnorm/mul/ReadVariableOp2batch_normalization_2/batchnorm/mul/ReadVariableOp2`
.batch_normalization_3/batchnorm/ReadVariableOp.batch_normalization_3/batchnorm/ReadVariableOp2d
0batch_normalization_3/batchnorm/ReadVariableOp_10batch_normalization_3/batchnorm/ReadVariableOp_12d
0batch_normalization_3/batchnorm/ReadVariableOp_20batch_normalization_3/batchnorm/ReadVariableOp_22h
2batch_normalization_3/batchnorm/mul/ReadVariableOp2batch_normalization_3/batchnorm/mul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2^
-dense_1/kernel/Regularizer/Abs/ReadVariableOp-dense_1/kernel/Regularizer/Abs/ReadVariableOp2P
&stream_0_conv_1/BiasAdd/ReadVariableOp&stream_0_conv_1/BiasAdd/ReadVariableOp2h
2stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp2stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp2n
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp2P
&stream_0_conv_2/BiasAdd/ReadVariableOp&stream_0_conv_2/BiasAdd/ReadVariableOp2h
2stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOp2stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOp2t
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp2P
&stream_0_conv_3/BiasAdd/ReadVariableOp&stream_0_conv_3/BiasAdd/ReadVariableOp2h
2stream_0_conv_3/conv1d/ExpandDims_1/ReadVariableOp2stream_0_conv_3/conv1d/ExpandDims_1/ReadVariableOp2n
5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp:U Q
+
_output_shapes
:         }
"
_user_specified_name
inputs/0
Р
╬
3__inference_batch_normalization_layer_call_fn_28173

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityѕбStatefulPartitionedCallЮ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         }@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_245422
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         }@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         }@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:         }@
 
_user_specified_nameinputs
И	
m
C__inference_distance_layer_call_and_return_conditional_losses_25264

inputs
inputs_1
identityU
subSubinputsinputs_1*
T0*'
_output_shapes
:         T2
subU
SquareSquaresub:z:0*
T0*'
_output_shapes
:         T2
Squarey
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
         2
Sum/reduction_indicesђ
SumSum
Square:y:0Sum/reduction_indices:output:0*
T0*'
_output_shapes
:         *
	keep_dims(2
SumS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
Constm
MaximumMaximumSum:output:0Const:output:0*
T0*'
_output_shapes
:         2	
MaximumS
SqrtSqrtMaximum:z:0*
T0*'
_output_shapes
:         2
Sqrt\
IdentityIdentitySqrt:y:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:         T:         T:O K
'
_output_shapes
:         T
 
_user_specified_nameinputs:OK
'
_output_shapes
:         T
 
_user_specified_nameinputs
В
i
J__inference_stream_0_drop_1_layer_call_and_return_conditional_losses_24484

inputs
identityѕc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n█Х?2
dropout/Constw
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:         }@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/ShapeМ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:         }@*
dtype0*
seedи*
seed2и2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *џЎЎ>2
dropout/GreaterEqual/y┬
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:         }@2
dropout/GreaterEqualЃ
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:         }@2
dropout/Cast~
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:         }@2
dropout/Mul_1i
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:         }@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         }@:S O
+
_output_shapes
:         }@
 
_user_specified_nameinputs
кH
ц	
@__inference_model_layer_call_and_return_conditional_losses_25867

inputs
inputs_1%
basemodel_25767:@
basemodel_25769:@
basemodel_25771:@
basemodel_25773:@
basemodel_25775:@
basemodel_25777:@&
basemodel_25779:@ђ
basemodel_25781:	ђ
basemodel_25783:	ђ
basemodel_25785:	ђ
basemodel_25787:	ђ
basemodel_25789:	ђ'
basemodel_25791:ђђ
basemodel_25793:	ђ
basemodel_25795:	ђ
basemodel_25797:	ђ
basemodel_25799:	ђ
basemodel_25801:	ђ"
basemodel_25803:	ђT
basemodel_25805:T
basemodel_25807:T
basemodel_25809:T
basemodel_25811:T
basemodel_25813:T
identityѕб!basemodel/StatefulPartitionedCallб#basemodel/StatefulPartitionedCall_1б-dense_1/kernel/Regularizer/Abs/ReadVariableOpб5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpб8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOpб5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp│
!basemodel/StatefulPartitionedCallStatefulPartitionedCallinputsbasemodel_25767basemodel_25769basemodel_25771basemodel_25773basemodel_25775basemodel_25777basemodel_25779basemodel_25781basemodel_25783basemodel_25785basemodel_25787basemodel_25789basemodel_25791basemodel_25793basemodel_25795basemodel_25797basemodel_25799basemodel_25801basemodel_25803basemodel_25805basemodel_25807basemodel_25809basemodel_25811basemodel_25813*$
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         T*2
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *M
fHRF
D__inference_basemodel_layer_call_and_return_conditional_losses_256552#
!basemodel/StatefulPartitionedCallП
#basemodel/StatefulPartitionedCall_1StatefulPartitionedCallinputs_1basemodel_25767basemodel_25769basemodel_25771basemodel_25773basemodel_25775basemodel_25777basemodel_25779basemodel_25781basemodel_25783basemodel_25785basemodel_25787basemodel_25789basemodel_25791basemodel_25793basemodel_25795basemodel_25797basemodel_25799basemodel_25801basemodel_25803basemodel_25805basemodel_25807basemodel_25809basemodel_25811basemodel_25813"^basemodel/StatefulPartitionedCall*$
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         T*2
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *M
fHRF
D__inference_basemodel_layer_call_and_return_conditional_losses_256552%
#basemodel/StatefulPartitionedCall_1Е
distance/PartitionedCallPartitionedCall*basemodel/StatefulPartitionedCall:output:0,basemodel/StatefulPartitionedCall_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *L
fGRE
C__inference_distance_layer_call_and_return_conditional_losses_253642
distance/PartitionedCall┬
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpbasemodel_25767*"
_output_shapes
:@*
dtype027
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp├
&stream_0_conv_1/kernel/Regularizer/AbsAbs=stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_0_conv_1/kernel/Regularizer/AbsЕ
(stream_0_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_0_conv_1/kernel/Regularizer/ConstО
&stream_0_conv_1/kernel/Regularizer/SumSum*stream_0_conv_1/kernel/Regularizer/Abs:y:01stream_0_conv_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/SumЎ
(stream_0_conv_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2*
(stream_0_conv_1/kernel/Regularizer/mul/x▄
&stream_0_conv_1/kernel/Regularizer/mulMul1stream_0_conv_1/kernel/Regularizer/mul/x:output:0/stream_0_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/mul╔
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpbasemodel_25779*#
_output_shapes
:@ђ*
dtype02:
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOpл
)stream_0_conv_2/kernel/Regularizer/SquareSquare@stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*#
_output_shapes
:@ђ2+
)stream_0_conv_2/kernel/Regularizer/SquareЕ
(stream_0_conv_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_0_conv_2/kernel/Regularizer/Const┌
&stream_0_conv_2/kernel/Regularizer/SumSum-stream_0_conv_2/kernel/Regularizer/Square:y:01stream_0_conv_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_2/kernel/Regularizer/SumЎ
(stream_0_conv_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2*
(stream_0_conv_2/kernel/Regularizer/mul/x▄
&stream_0_conv_2/kernel/Regularizer/mulMul1stream_0_conv_2/kernel/Regularizer/mul/x:output:0/stream_0_conv_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_2/kernel/Regularizer/mul─
5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpbasemodel_25791*$
_output_shapes
:ђђ*
dtype027
5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp┼
&stream_0_conv_3/kernel/Regularizer/AbsAbs=stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*$
_output_shapes
:ђђ2(
&stream_0_conv_3/kernel/Regularizer/AbsЕ
(stream_0_conv_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_0_conv_3/kernel/Regularizer/ConstО
&stream_0_conv_3/kernel/Regularizer/SumSum*stream_0_conv_3/kernel/Regularizer/Abs:y:01stream_0_conv_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_3/kernel/Regularizer/SumЎ
(stream_0_conv_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2*
(stream_0_conv_3/kernel/Regularizer/mul/x▄
&stream_0_conv_3/kernel/Regularizer/mulMul1stream_0_conv_3/kernel/Regularizer/mul/x:output:0/stream_0_conv_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_3/kernel/Regularizer/mul»
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpbasemodel_25803*
_output_shapes
:	ђT*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOpе
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	ђT2 
dense_1/kernel/Regularizer/AbsЋ
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_1/kernel/Regularizer/Constи
dense_1/kernel/Regularizer/SumSum"dense_1/kernel/Regularizer/Abs:y:0)dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/SumЅ
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2"
 dense_1/kernel/Regularizer/mul/x╝
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/mul|
IdentityIdentity!distance/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         2

Identityз
NoOpNoOp"^basemodel/StatefulPartitionedCall$^basemodel/StatefulPartitionedCall_1.^dense_1/kernel/Regularizer/Abs/ReadVariableOp6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp9^stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp6^stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*q
_input_shapes`
^:         }:         }: : : : : : : : : : : : : : : : : : : : : : : : 2F
!basemodel/StatefulPartitionedCall!basemodel/StatefulPartitionedCall2J
#basemodel/StatefulPartitionedCall_1#basemodel/StatefulPartitionedCall_12^
-dense_1/kernel/Regularizer/Abs/ReadVariableOp-dense_1/kernel/Regularizer/Abs/ReadVariableOp2n
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp2t
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp2n
5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp:S O
+
_output_shapes
:         }
 
_user_specified_nameinputs:SO
+
_output_shapes
:         }
 
_user_specified_nameinputs
Ў
б
/__inference_stream_0_conv_2_layer_call_fn_28333

inputs
unknown:@ђ
	unknown_0:	ђ
identityѕбStatefulPartitionedCallѓ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         }ђ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *S
fNRL
J__inference_stream_0_conv_2_layer_call_and_return_conditional_losses_239752
StatefulPartitionedCallђ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:         }ђ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         }@: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:         }@
 
_user_specified_nameinputs
з
Ћ
'__inference_dense_1_layer_call_fn_28848

inputs
unknown:	ђT
	unknown_0:T
identityѕбStatefulPartitionedCallш
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         T*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_241242
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         T2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ђ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
Е
ч
%__inference_model_layer_call_fn_26374
inputs_0
inputs_1
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:@ 
	unknown_5:@ђ
	unknown_6:	ђ
	unknown_7:	ђ
	unknown_8:	ђ
	unknown_9:	ђ

unknown_10:	ђ"

unknown_11:ђђ

unknown_12:	ђ

unknown_13:	ђ

unknown_14:	ђ

unknown_15:	ђ

unknown_16:	ђ

unknown_17:	ђT

unknown_18:T

unknown_19:T

unknown_20:T

unknown_21:T

unknown_22:T
identityѕбStatefulPartitionedCallБ
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22*%
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *2
_read_only_resource_inputs
	*0
config_proto 

CPU

GPU2*0J 8ѓ *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_258672
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*q
_input_shapes`
^:         }:         }: : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
+
_output_shapes
:         }
"
_user_specified_name
inputs/0:UQ
+
_output_shapes
:         }
"
_user_specified_name
inputs/1
з
c
G__inference_activation_1_layer_call_and_return_conditional_losses_24015

inputs
identityS
ReluReluinputs*
T0*,
_output_shapes
:         }ђ2
Reluk
IdentityIdentityRelu:activations:0*
T0*,
_output_shapes
:         }ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         }ђ:T P
,
_output_shapes
:         }ђ
 
_user_specified_nameinputs
Ѕ	
╬
3__inference_batch_normalization_layer_call_fn_28134

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityѕбStatefulPartitionedCallе
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  @*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_232222
StatefulPartitionedCallѕ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :                  @2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:                  @: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :                  @
 
_user_specified_nameinputs
╚
│
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_23384

inputs0
!batchnorm_readvariableop_resource:	ђ4
%batchnorm_mul_readvariableop_resource:	ђ2
#batchnorm_readvariableop_1_resource:	ђ2
#batchnorm_readvariableop_2_resource:	ђ
identityѕбbatchnorm/ReadVariableOpбbatchnorm/ReadVariableOp_1бbatchnorm/ReadVariableOp_2бbatchnorm/mul/ReadVariableOpЊ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2
batchnorm/add/yЅ
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:ђ2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:ђ2
batchnorm/RsqrtЪ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
batchnorm/mul/ReadVariableOpє
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:ђ2
batchnorm/mulё
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*5
_output_shapes#
!:                  ђ2
batchnorm/mul_1Ў
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype02
batchnorm/ReadVariableOp_1є
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:ђ2
batchnorm/mul_2Ў
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:ђ*
dtype02
batchnorm/ReadVariableOp_2ё
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:ђ2
batchnorm/subЊ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*5
_output_shapes#
!:                  ђ2
batchnorm/add_1|
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*5
_output_shapes#
!:                  ђ2

Identity┬
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):                  ђ: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:] Y
5
_output_shapes#
!:                  ђ
 
_user_specified_nameinputs
Ѓњ
Г*
!__inference__traced_restore_29410
file_prefix!
assignvariableop_beta_1: #
assignvariableop_1_beta_2: "
assignvariableop_2_decay: *
 assignvariableop_3_learning_rate: &
assignvariableop_4_adam_iter:	 ?
)assignvariableop_5_stream_0_conv_1_kernel:@5
'assignvariableop_6_stream_0_conv_1_bias:@:
,assignvariableop_7_batch_normalization_gamma:@9
+assignvariableop_8_batch_normalization_beta:@@
)assignvariableop_9_stream_0_conv_2_kernel:@ђ7
(assignvariableop_10_stream_0_conv_2_bias:	ђ>
/assignvariableop_11_batch_normalization_1_gamma:	ђ=
.assignvariableop_12_batch_normalization_1_beta:	ђB
*assignvariableop_13_stream_0_conv_3_kernel:ђђ7
(assignvariableop_14_stream_0_conv_3_bias:	ђ>
/assignvariableop_15_batch_normalization_2_gamma:	ђ=
.assignvariableop_16_batch_normalization_2_beta:	ђ5
"assignvariableop_17_dense_1_kernel:	ђT.
 assignvariableop_18_dense_1_bias:T=
/assignvariableop_19_batch_normalization_3_gamma:T<
.assignvariableop_20_batch_normalization_3_beta:TA
3assignvariableop_21_batch_normalization_moving_mean:@E
7assignvariableop_22_batch_normalization_moving_variance:@D
5assignvariableop_23_batch_normalization_1_moving_mean:	ђH
9assignvariableop_24_batch_normalization_1_moving_variance:	ђD
5assignvariableop_25_batch_normalization_2_moving_mean:	ђH
9assignvariableop_26_batch_normalization_2_moving_variance:	ђC
5assignvariableop_27_batch_normalization_3_moving_mean:TG
9assignvariableop_28_batch_normalization_3_moving_variance:T#
assignvariableop_29_total: #
assignvariableop_30_count: G
1assignvariableop_31_adam_stream_0_conv_1_kernel_m:@=
/assignvariableop_32_adam_stream_0_conv_1_bias_m:@B
4assignvariableop_33_adam_batch_normalization_gamma_m:@A
3assignvariableop_34_adam_batch_normalization_beta_m:@H
1assignvariableop_35_adam_stream_0_conv_2_kernel_m:@ђ>
/assignvariableop_36_adam_stream_0_conv_2_bias_m:	ђE
6assignvariableop_37_adam_batch_normalization_1_gamma_m:	ђD
5assignvariableop_38_adam_batch_normalization_1_beta_m:	ђI
1assignvariableop_39_adam_stream_0_conv_3_kernel_m:ђђ>
/assignvariableop_40_adam_stream_0_conv_3_bias_m:	ђE
6assignvariableop_41_adam_batch_normalization_2_gamma_m:	ђD
5assignvariableop_42_adam_batch_normalization_2_beta_m:	ђ<
)assignvariableop_43_adam_dense_1_kernel_m:	ђT5
'assignvariableop_44_adam_dense_1_bias_m:TD
6assignvariableop_45_adam_batch_normalization_3_gamma_m:TC
5assignvariableop_46_adam_batch_normalization_3_beta_m:TG
1assignvariableop_47_adam_stream_0_conv_1_kernel_v:@=
/assignvariableop_48_adam_stream_0_conv_1_bias_v:@B
4assignvariableop_49_adam_batch_normalization_gamma_v:@A
3assignvariableop_50_adam_batch_normalization_beta_v:@H
1assignvariableop_51_adam_stream_0_conv_2_kernel_v:@ђ>
/assignvariableop_52_adam_stream_0_conv_2_bias_v:	ђE
6assignvariableop_53_adam_batch_normalization_1_gamma_v:	ђD
5assignvariableop_54_adam_batch_normalization_1_beta_v:	ђI
1assignvariableop_55_adam_stream_0_conv_3_kernel_v:ђђ>
/assignvariableop_56_adam_stream_0_conv_3_bias_v:	ђE
6assignvariableop_57_adam_batch_normalization_2_gamma_v:	ђD
5assignvariableop_58_adam_batch_normalization_2_beta_v:	ђ<
)assignvariableop_59_adam_dense_1_kernel_v:	ђT5
'assignvariableop_60_adam_dense_1_bias_v:TD
6assignvariableop_61_adam_batch_normalization_3_gamma_v:TC
5assignvariableop_62_adam_batch_normalization_3_beta_v:T
identity_64ѕбAssignVariableOpбAssignVariableOp_1бAssignVariableOp_10бAssignVariableOp_11бAssignVariableOp_12бAssignVariableOp_13бAssignVariableOp_14бAssignVariableOp_15бAssignVariableOp_16бAssignVariableOp_17бAssignVariableOp_18бAssignVariableOp_19бAssignVariableOp_2бAssignVariableOp_20бAssignVariableOp_21бAssignVariableOp_22бAssignVariableOp_23бAssignVariableOp_24бAssignVariableOp_25бAssignVariableOp_26бAssignVariableOp_27бAssignVariableOp_28бAssignVariableOp_29бAssignVariableOp_3бAssignVariableOp_30бAssignVariableOp_31бAssignVariableOp_32бAssignVariableOp_33бAssignVariableOp_34бAssignVariableOp_35бAssignVariableOp_36бAssignVariableOp_37бAssignVariableOp_38бAssignVariableOp_39бAssignVariableOp_4бAssignVariableOp_40бAssignVariableOp_41бAssignVariableOp_42бAssignVariableOp_43бAssignVariableOp_44бAssignVariableOp_45бAssignVariableOp_46бAssignVariableOp_47бAssignVariableOp_48бAssignVariableOp_49бAssignVariableOp_5бAssignVariableOp_50бAssignVariableOp_51бAssignVariableOp_52бAssignVariableOp_53бAssignVariableOp_54бAssignVariableOp_55бAssignVariableOp_56бAssignVariableOp_57бAssignVariableOp_58бAssignVariableOp_59бAssignVariableOp_6бAssignVariableOp_60бAssignVariableOp_61бAssignVariableOp_62бAssignVariableOp_7бAssignVariableOp_8бAssignVariableOp_9░ 
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:@*
dtype0*╝
value▓B»@B+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/14/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/15/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesЉ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:@*
dtype0*Ћ
valueІBѕ@B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesЬ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*ќ
_output_shapesЃ
ђ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*N
dtypesD
B2@	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identityќ
AssignVariableOpAssignVariableOpassignvariableop_beta_1Identity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1ъ
AssignVariableOp_1AssignVariableOpassignvariableop_1_beta_2Identity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2Ю
AssignVariableOp_2AssignVariableOpassignvariableop_2_decayIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3Ц
AssignVariableOp_3AssignVariableOp assignvariableop_3_learning_rateIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_4А
AssignVariableOp_4AssignVariableOpassignvariableop_4_adam_iterIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5«
AssignVariableOp_5AssignVariableOp)assignvariableop_5_stream_0_conv_1_kernelIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6г
AssignVariableOp_6AssignVariableOp'assignvariableop_6_stream_0_conv_1_biasIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7▒
AssignVariableOp_7AssignVariableOp,assignvariableop_7_batch_normalization_gammaIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8░
AssignVariableOp_8AssignVariableOp+assignvariableop_8_batch_normalization_betaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9«
AssignVariableOp_9AssignVariableOp)assignvariableop_9_stream_0_conv_2_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10░
AssignVariableOp_10AssignVariableOp(assignvariableop_10_stream_0_conv_2_biasIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11и
AssignVariableOp_11AssignVariableOp/assignvariableop_11_batch_normalization_1_gammaIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12Х
AssignVariableOp_12AssignVariableOp.assignvariableop_12_batch_normalization_1_betaIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13▓
AssignVariableOp_13AssignVariableOp*assignvariableop_13_stream_0_conv_3_kernelIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14░
AssignVariableOp_14AssignVariableOp(assignvariableop_14_stream_0_conv_3_biasIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15и
AssignVariableOp_15AssignVariableOp/assignvariableop_15_batch_normalization_2_gammaIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16Х
AssignVariableOp_16AssignVariableOp.assignvariableop_16_batch_normalization_2_betaIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17ф
AssignVariableOp_17AssignVariableOp"assignvariableop_17_dense_1_kernelIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18е
AssignVariableOp_18AssignVariableOp assignvariableop_18_dense_1_biasIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19и
AssignVariableOp_19AssignVariableOp/assignvariableop_19_batch_normalization_3_gammaIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20Х
AssignVariableOp_20AssignVariableOp.assignvariableop_20_batch_normalization_3_betaIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21╗
AssignVariableOp_21AssignVariableOp3assignvariableop_21_batch_normalization_moving_meanIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22┐
AssignVariableOp_22AssignVariableOp7assignvariableop_22_batch_normalization_moving_varianceIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23й
AssignVariableOp_23AssignVariableOp5assignvariableop_23_batch_normalization_1_moving_meanIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24┴
AssignVariableOp_24AssignVariableOp9assignvariableop_24_batch_normalization_1_moving_varianceIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25й
AssignVariableOp_25AssignVariableOp5assignvariableop_25_batch_normalization_2_moving_meanIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26┴
AssignVariableOp_26AssignVariableOp9assignvariableop_26_batch_normalization_2_moving_varianceIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27й
AssignVariableOp_27AssignVariableOp5assignvariableop_27_batch_normalization_3_moving_meanIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28┴
AssignVariableOp_28AssignVariableOp9assignvariableop_28_batch_normalization_3_moving_varianceIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29А
AssignVariableOp_29AssignVariableOpassignvariableop_29_totalIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30А
AssignVariableOp_30AssignVariableOpassignvariableop_30_countIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31╣
AssignVariableOp_31AssignVariableOp1assignvariableop_31_adam_stream_0_conv_1_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32и
AssignVariableOp_32AssignVariableOp/assignvariableop_32_adam_stream_0_conv_1_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33╝
AssignVariableOp_33AssignVariableOp4assignvariableop_33_adam_batch_normalization_gamma_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34╗
AssignVariableOp_34AssignVariableOp3assignvariableop_34_adam_batch_normalization_beta_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35╣
AssignVariableOp_35AssignVariableOp1assignvariableop_35_adam_stream_0_conv_2_kernel_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36и
AssignVariableOp_36AssignVariableOp/assignvariableop_36_adam_stream_0_conv_2_bias_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37Й
AssignVariableOp_37AssignVariableOp6assignvariableop_37_adam_batch_normalization_1_gamma_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38й
AssignVariableOp_38AssignVariableOp5assignvariableop_38_adam_batch_normalization_1_beta_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39╣
AssignVariableOp_39AssignVariableOp1assignvariableop_39_adam_stream_0_conv_3_kernel_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40и
AssignVariableOp_40AssignVariableOp/assignvariableop_40_adam_stream_0_conv_3_bias_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41Й
AssignVariableOp_41AssignVariableOp6assignvariableop_41_adam_batch_normalization_2_gamma_mIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42й
AssignVariableOp_42AssignVariableOp5assignvariableop_42_adam_batch_normalization_2_beta_mIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43▒
AssignVariableOp_43AssignVariableOp)assignvariableop_43_adam_dense_1_kernel_mIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44»
AssignVariableOp_44AssignVariableOp'assignvariableop_44_adam_dense_1_bias_mIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45Й
AssignVariableOp_45AssignVariableOp6assignvariableop_45_adam_batch_normalization_3_gamma_mIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46й
AssignVariableOp_46AssignVariableOp5assignvariableop_46_adam_batch_normalization_3_beta_mIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47╣
AssignVariableOp_47AssignVariableOp1assignvariableop_47_adam_stream_0_conv_1_kernel_vIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48и
AssignVariableOp_48AssignVariableOp/assignvariableop_48_adam_stream_0_conv_1_bias_vIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49╝
AssignVariableOp_49AssignVariableOp4assignvariableop_49_adam_batch_normalization_gamma_vIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50╗
AssignVariableOp_50AssignVariableOp3assignvariableop_50_adam_batch_normalization_beta_vIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51╣
AssignVariableOp_51AssignVariableOp1assignvariableop_51_adam_stream_0_conv_2_kernel_vIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52и
AssignVariableOp_52AssignVariableOp/assignvariableop_52_adam_stream_0_conv_2_bias_vIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53Й
AssignVariableOp_53AssignVariableOp6assignvariableop_53_adam_batch_normalization_1_gamma_vIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54й
AssignVariableOp_54AssignVariableOp5assignvariableop_54_adam_batch_normalization_1_beta_vIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55╣
AssignVariableOp_55AssignVariableOp1assignvariableop_55_adam_stream_0_conv_3_kernel_vIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56и
AssignVariableOp_56AssignVariableOp/assignvariableop_56_adam_stream_0_conv_3_bias_vIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57Й
AssignVariableOp_57AssignVariableOp6assignvariableop_57_adam_batch_normalization_2_gamma_vIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58й
AssignVariableOp_58AssignVariableOp5assignvariableop_58_adam_batch_normalization_2_beta_vIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_58n
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:2
Identity_59▒
AssignVariableOp_59AssignVariableOp)assignvariableop_59_adam_dense_1_kernel_vIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59n
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:2
Identity_60»
AssignVariableOp_60AssignVariableOp'assignvariableop_60_adam_dense_1_bias_vIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_60n
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:2
Identity_61Й
AssignVariableOp_61AssignVariableOp6assignvariableop_61_adam_batch_normalization_3_gamma_vIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_61n
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:2
Identity_62й
AssignVariableOp_62AssignVariableOp5assignvariableop_62_adam_batch_normalization_3_beta_vIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_629
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp╚
Identity_63Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_63f
Identity_64IdentityIdentity_63:output:0^NoOp_1*
T0*
_output_shapes
: 2
Identity_64░
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 2
NoOp_1"#
identity_64Identity_64:output:0*Ћ
_input_shapesЃ
ђ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
┬
T
(__inference_distance_layer_call_fn_28034
inputs_0
inputs_1
identityЛ
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *L
fGRE
C__inference_distance_layer_call_and_return_conditional_losses_253642
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:         T:         T:Q M
'
_output_shapes
:         T
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:         T
"
_user_specified_name
inputs/1
╔
╝
__inference_loss_fn_0_28965T
>stream_0_conv_1_kernel_regularizer_abs_readvariableop_resource:@
identityѕб5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpы
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp>stream_0_conv_1_kernel_regularizer_abs_readvariableop_resource*"
_output_shapes
:@*
dtype027
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp├
&stream_0_conv_1/kernel/Regularizer/AbsAbs=stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_0_conv_1/kernel/Regularizer/AbsЕ
(stream_0_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_0_conv_1/kernel/Regularizer/ConstО
&stream_0_conv_1/kernel/Regularizer/SumSum*stream_0_conv_1/kernel/Regularizer/Abs:y:01stream_0_conv_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/SumЎ
(stream_0_conv_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2*
(stream_0_conv_1/kernel/Regularizer/mul/x▄
&stream_0_conv_1/kernel/Regularizer/mulMul1stream_0_conv_1/kernel/Regularizer/mul/x:output:0/stream_0_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/mult
IdentityIdentity*stream_0_conv_1/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: 2

Identityє
NoOpNoOp6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2n
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp
■*
у
N__inference_batch_normalization_layer_call_and_return_conditional_losses_24542

inputs5
'assignmovingavg_readvariableop_resource:@7
)assignmovingavg_1_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@/
!batchnorm_readvariableop_resource:@
identityѕбAssignMovingAvgбAssignMovingAvg/ReadVariableOpбAssignMovingAvg_1б AssignMovingAvg_1/ReadVariableOpбbatchnorm/ReadVariableOpбbatchnorm/mul/ReadVariableOpЉ
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indicesЊ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(2
moments/meanђ
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:@2
moments/StopGradientе
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*+
_output_shapes
:         }@2
moments/SquaredDifferenceЎ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indicesХ
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(2
moments/varianceЂ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 2
moments/SqueezeЅ
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 2
moments/Squeeze_1s
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2
AssignMovingAvg/decayц
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOpў
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg/subЈ
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg/mul┐
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvgw
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2
AssignMovingAvg_1/decayф
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOpа
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg_1/subЌ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg_1/mul╔
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1g
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2
batchnorm/add/yѓ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@2
batchnorm/Rsqrtъ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOpЁ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2
batchnorm/mulz
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:         }@2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:@2
batchnorm/mul_2њ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOpЂ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2
batchnorm/subЅ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*+
_output_shapes
:         }@2
batchnorm/add_1r
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*+
_output_shapes
:         }@2

IdentityЫ
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         }@: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:S O
+
_output_shapes
:         }@
 
_user_specified_nameinputs
г
№
)__inference_basemodel_layer_call_fn_27113

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:@ 
	unknown_5:@ђ
	unknown_6:	ђ
	unknown_7:	ђ
	unknown_8:	ђ
	unknown_9:	ђ

unknown_10:	ђ"

unknown_11:ђђ

unknown_12:	ђ

unknown_13:	ђ

unknown_14:	ђ

unknown_15:	ђ

unknown_16:	ђ

unknown_17:	ђT

unknown_18:T

unknown_19:T

unknown_20:T

unknown_21:T

unknown_22:T
identityѕбStatefulPartitionedCallб
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22*$
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         T*:
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8ѓ *M
fHRF
D__inference_basemodel_layer_call_and_return_conditional_losses_241712
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         T2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:         }: : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:         }
 
_user_specified_nameinputs
С
╬
3__inference_batch_normalization_layer_call_fn_28160

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityѕбStatefulPartitionedCallЪ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         }@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_239302
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         }@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         }@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:         }@
 
_user_specified_nameinputs
█
K
/__inference_stream_0_drop_1_layer_call_fn_28296

inputs
identity¤
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         }@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *S
fNRL
J__inference_stream_0_drop_1_layer_call_and_return_conditional_losses_239522
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:         }@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         }@:S O
+
_output_shapes
:         }@
 
_user_specified_nameinputs
З
»
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_28910

inputs/
!batchnorm_readvariableop_resource:T3
%batchnorm_mul_readvariableop_resource:T1
#batchnorm_readvariableop_1_resource:T1
#batchnorm_readvariableop_2_resource:T
identityѕбbatchnorm/ReadVariableOpбbatchnorm/ReadVariableOp_1бbatchnorm/ReadVariableOp_2бbatchnorm/mul/ReadVariableOpњ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:T*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2
batchnorm/add/yѕ
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:T2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:T2
batchnorm/Rsqrtъ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:T*
dtype02
batchnorm/mul/ReadVariableOpЁ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:T2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:         T2
batchnorm/mul_1ў
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:T*
dtype02
batchnorm/ReadVariableOp_1Ё
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:T2
batchnorm/mul_2ў
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:T*
dtype02
batchnorm/ReadVariableOp_2Ѓ
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:T2
batchnorm/subЁ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:         T2
batchnorm/add_1n
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:         T2

Identity┬
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         T: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:         T
 
_user_specified_nameinputs
Є
o
S__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_24099

inputs
identityr
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Mean/reduction_indicesp
MeanMeaninputsMean/reduction_indices:output:0*
T0*(
_output_shapes
:         ђ2
Meanb
IdentityIdentityMean:output:0*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         }ђ:T P
,
_output_shapes
:         }ђ
 
_user_specified_nameinputs
ИЬ
Ї(
@__inference_model_layer_call_and_return_conditional_losses_26614
inputs_0
inputs_1[
Ebasemodel_stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource:@G
9basemodel_stream_0_conv_1_biasadd_readvariableop_resource:@M
?basemodel_batch_normalization_batchnorm_readvariableop_resource:@Q
Cbasemodel_batch_normalization_batchnorm_mul_readvariableop_resource:@O
Abasemodel_batch_normalization_batchnorm_readvariableop_1_resource:@O
Abasemodel_batch_normalization_batchnorm_readvariableop_2_resource:@\
Ebasemodel_stream_0_conv_2_conv1d_expanddims_1_readvariableop_resource:@ђH
9basemodel_stream_0_conv_2_biasadd_readvariableop_resource:	ђP
Abasemodel_batch_normalization_1_batchnorm_readvariableop_resource:	ђT
Ebasemodel_batch_normalization_1_batchnorm_mul_readvariableop_resource:	ђR
Cbasemodel_batch_normalization_1_batchnorm_readvariableop_1_resource:	ђR
Cbasemodel_batch_normalization_1_batchnorm_readvariableop_2_resource:	ђ]
Ebasemodel_stream_0_conv_3_conv1d_expanddims_1_readvariableop_resource:ђђH
9basemodel_stream_0_conv_3_biasadd_readvariableop_resource:	ђP
Abasemodel_batch_normalization_2_batchnorm_readvariableop_resource:	ђT
Ebasemodel_batch_normalization_2_batchnorm_mul_readvariableop_resource:	ђR
Cbasemodel_batch_normalization_2_batchnorm_readvariableop_1_resource:	ђR
Cbasemodel_batch_normalization_2_batchnorm_readvariableop_2_resource:	ђC
0basemodel_dense_1_matmul_readvariableop_resource:	ђT?
1basemodel_dense_1_biasadd_readvariableop_resource:TO
Abasemodel_batch_normalization_3_batchnorm_readvariableop_resource:TS
Ebasemodel_batch_normalization_3_batchnorm_mul_readvariableop_resource:TQ
Cbasemodel_batch_normalization_3_batchnorm_readvariableop_1_resource:TQ
Cbasemodel_batch_normalization_3_batchnorm_readvariableop_2_resource:T
identityѕб6basemodel/batch_normalization/batchnorm/ReadVariableOpб8basemodel/batch_normalization/batchnorm/ReadVariableOp_1б8basemodel/batch_normalization/batchnorm/ReadVariableOp_2б:basemodel/batch_normalization/batchnorm/mul/ReadVariableOpб8basemodel/batch_normalization/batchnorm_1/ReadVariableOpб:basemodel/batch_normalization/batchnorm_1/ReadVariableOp_1б:basemodel/batch_normalization/batchnorm_1/ReadVariableOp_2б<basemodel/batch_normalization/batchnorm_1/mul/ReadVariableOpб8basemodel/batch_normalization_1/batchnorm/ReadVariableOpб:basemodel/batch_normalization_1/batchnorm/ReadVariableOp_1б:basemodel/batch_normalization_1/batchnorm/ReadVariableOp_2б<basemodel/batch_normalization_1/batchnorm/mul/ReadVariableOpб:basemodel/batch_normalization_1/batchnorm_1/ReadVariableOpб<basemodel/batch_normalization_1/batchnorm_1/ReadVariableOp_1б<basemodel/batch_normalization_1/batchnorm_1/ReadVariableOp_2б>basemodel/batch_normalization_1/batchnorm_1/mul/ReadVariableOpб8basemodel/batch_normalization_2/batchnorm/ReadVariableOpб:basemodel/batch_normalization_2/batchnorm/ReadVariableOp_1б:basemodel/batch_normalization_2/batchnorm/ReadVariableOp_2б<basemodel/batch_normalization_2/batchnorm/mul/ReadVariableOpб:basemodel/batch_normalization_2/batchnorm_1/ReadVariableOpб<basemodel/batch_normalization_2/batchnorm_1/ReadVariableOp_1б<basemodel/batch_normalization_2/batchnorm_1/ReadVariableOp_2б>basemodel/batch_normalization_2/batchnorm_1/mul/ReadVariableOpб8basemodel/batch_normalization_3/batchnorm/ReadVariableOpб:basemodel/batch_normalization_3/batchnorm/ReadVariableOp_1б:basemodel/batch_normalization_3/batchnorm/ReadVariableOp_2б<basemodel/batch_normalization_3/batchnorm/mul/ReadVariableOpб:basemodel/batch_normalization_3/batchnorm_1/ReadVariableOpб<basemodel/batch_normalization_3/batchnorm_1/ReadVariableOp_1б<basemodel/batch_normalization_3/batchnorm_1/ReadVariableOp_2б>basemodel/batch_normalization_3/batchnorm_1/mul/ReadVariableOpб(basemodel/dense_1/BiasAdd/ReadVariableOpб*basemodel/dense_1/BiasAdd_1/ReadVariableOpб'basemodel/dense_1/MatMul/ReadVariableOpб)basemodel/dense_1/MatMul_1/ReadVariableOpб0basemodel/stream_0_conv_1/BiasAdd/ReadVariableOpб2basemodel/stream_0_conv_1/BiasAdd_1/ReadVariableOpб<basemodel/stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOpб>basemodel/stream_0_conv_1/conv1d_1/ExpandDims_1/ReadVariableOpб0basemodel/stream_0_conv_2/BiasAdd/ReadVariableOpб2basemodel/stream_0_conv_2/BiasAdd_1/ReadVariableOpб<basemodel/stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOpб>basemodel/stream_0_conv_2/conv1d_1/ExpandDims_1/ReadVariableOpб0basemodel/stream_0_conv_3/BiasAdd/ReadVariableOpб2basemodel/stream_0_conv_3/BiasAdd_1/ReadVariableOpб<basemodel/stream_0_conv_3/conv1d/ExpandDims_1/ReadVariableOpб>basemodel/stream_0_conv_3/conv1d_1/ExpandDims_1/ReadVariableOpб-dense_1/kernel/Regularizer/Abs/ReadVariableOpб5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpб8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOpб5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOpю
&basemodel/stream_0_input_drop/IdentityIdentityinputs_0*
T0*+
_output_shapes
:         }2(
&basemodel/stream_0_input_drop/IdentityГ
/basemodel/stream_0_conv_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§        21
/basemodel/stream_0_conv_1/conv1d/ExpandDims/dimЇ
+basemodel/stream_0_conv_1/conv1d/ExpandDims
ExpandDims/basemodel/stream_0_input_drop/Identity:output:08basemodel/stream_0_conv_1/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:         }2-
+basemodel/stream_0_conv_1/conv1d/ExpandDimsє
<basemodel/stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpEbasemodel_stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02>
<basemodel/stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOpе
1basemodel/stream_0_conv_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 23
1basemodel/stream_0_conv_1/conv1d/ExpandDims_1/dimЪ
-basemodel/stream_0_conv_1/conv1d/ExpandDims_1
ExpandDimsDbasemodel/stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp:value:0:basemodel/stream_0_conv_1/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@2/
-basemodel/stream_0_conv_1/conv1d/ExpandDims_1ъ
 basemodel/stream_0_conv_1/conv1dConv2D4basemodel/stream_0_conv_1/conv1d/ExpandDims:output:06basemodel/stream_0_conv_1/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:         }@*
paddingSAME*
strides
2"
 basemodel/stream_0_conv_1/conv1dЯ
(basemodel/stream_0_conv_1/conv1d/SqueezeSqueeze)basemodel/stream_0_conv_1/conv1d:output:0*
T0*+
_output_shapes
:         }@*
squeeze_dims

§        2*
(basemodel/stream_0_conv_1/conv1d/Squeeze┌
0basemodel/stream_0_conv_1/BiasAdd/ReadVariableOpReadVariableOp9basemodel_stream_0_conv_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype022
0basemodel/stream_0_conv_1/BiasAdd/ReadVariableOpЗ
!basemodel/stream_0_conv_1/BiasAddBiasAdd1basemodel/stream_0_conv_1/conv1d/Squeeze:output:08basemodel/stream_0_conv_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         }@2#
!basemodel/stream_0_conv_1/BiasAddВ
6basemodel/batch_normalization/batchnorm/ReadVariableOpReadVariableOp?basemodel_batch_normalization_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype028
6basemodel/batch_normalization/batchnorm/ReadVariableOpБ
-basemodel/batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2/
-basemodel/batch_normalization/batchnorm/add/yђ
+basemodel/batch_normalization/batchnorm/addAddV2>basemodel/batch_normalization/batchnorm/ReadVariableOp:value:06basemodel/batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes
:@2-
+basemodel/batch_normalization/batchnorm/addй
-basemodel/batch_normalization/batchnorm/RsqrtRsqrt/basemodel/batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes
:@2/
-basemodel/batch_normalization/batchnorm/RsqrtЭ
:basemodel/batch_normalization/batchnorm/mul/ReadVariableOpReadVariableOpCbasemodel_batch_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02<
:basemodel/batch_normalization/batchnorm/mul/ReadVariableOp§
+basemodel/batch_normalization/batchnorm/mulMul1basemodel/batch_normalization/batchnorm/Rsqrt:y:0Bbasemodel/batch_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2-
+basemodel/batch_normalization/batchnorm/mulЭ
-basemodel/batch_normalization/batchnorm/mul_1Mul*basemodel/stream_0_conv_1/BiasAdd:output:0/basemodel/batch_normalization/batchnorm/mul:z:0*
T0*+
_output_shapes
:         }@2/
-basemodel/batch_normalization/batchnorm/mul_1Ы
8basemodel/batch_normalization/batchnorm/ReadVariableOp_1ReadVariableOpAbasemodel_batch_normalization_batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype02:
8basemodel/batch_normalization/batchnorm/ReadVariableOp_1§
-basemodel/batch_normalization/batchnorm/mul_2Mul@basemodel/batch_normalization/batchnorm/ReadVariableOp_1:value:0/basemodel/batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes
:@2/
-basemodel/batch_normalization/batchnorm/mul_2Ы
8basemodel/batch_normalization/batchnorm/ReadVariableOp_2ReadVariableOpAbasemodel_batch_normalization_batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype02:
8basemodel/batch_normalization/batchnorm/ReadVariableOp_2ч
+basemodel/batch_normalization/batchnorm/subSub@basemodel/batch_normalization/batchnorm/ReadVariableOp_2:value:01basemodel/batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2-
+basemodel/batch_normalization/batchnorm/subЂ
-basemodel/batch_normalization/batchnorm/add_1AddV21basemodel/batch_normalization/batchnorm/mul_1:z:0/basemodel/batch_normalization/batchnorm/sub:z:0*
T0*+
_output_shapes
:         }@2/
-basemodel/batch_normalization/batchnorm/add_1Д
basemodel/activation/ReluRelu1basemodel/batch_normalization/batchnorm/add_1:z:0*
T0*+
_output_shapes
:         }@2
basemodel/activation/Relu│
"basemodel/stream_0_drop_1/IdentityIdentity'basemodel/activation/Relu:activations:0*
T0*+
_output_shapes
:         }@2$
"basemodel/stream_0_drop_1/IdentityГ
/basemodel/stream_0_conv_2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§        21
/basemodel/stream_0_conv_2/conv1d/ExpandDims/dimЅ
+basemodel/stream_0_conv_2/conv1d/ExpandDims
ExpandDims+basemodel/stream_0_drop_1/Identity:output:08basemodel/stream_0_conv_2/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:         }@2-
+basemodel/stream_0_conv_2/conv1d/ExpandDimsЄ
<basemodel/stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpEbasemodel_stream_0_conv_2_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:@ђ*
dtype02>
<basemodel/stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOpе
1basemodel/stream_0_conv_2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 23
1basemodel/stream_0_conv_2/conv1d/ExpandDims_1/dimа
-basemodel/stream_0_conv_2/conv1d/ExpandDims_1
ExpandDimsDbasemodel/stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOp:value:0:basemodel/stream_0_conv_2/conv1d/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:@ђ2/
-basemodel/stream_0_conv_2/conv1d/ExpandDims_1Ъ
 basemodel/stream_0_conv_2/conv1dConv2D4basemodel/stream_0_conv_2/conv1d/ExpandDims:output:06basemodel/stream_0_conv_2/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:         }ђ*
paddingSAME*
strides
2"
 basemodel/stream_0_conv_2/conv1dр
(basemodel/stream_0_conv_2/conv1d/SqueezeSqueeze)basemodel/stream_0_conv_2/conv1d:output:0*
T0*,
_output_shapes
:         }ђ*
squeeze_dims

§        2*
(basemodel/stream_0_conv_2/conv1d/Squeeze█
0basemodel/stream_0_conv_2/BiasAdd/ReadVariableOpReadVariableOp9basemodel_stream_0_conv_2_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype022
0basemodel/stream_0_conv_2/BiasAdd/ReadVariableOpш
!basemodel/stream_0_conv_2/BiasAddBiasAdd1basemodel/stream_0_conv_2/conv1d/Squeeze:output:08basemodel/stream_0_conv_2/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:         }ђ2#
!basemodel/stream_0_conv_2/BiasAddз
8basemodel/batch_normalization_1/batchnorm/ReadVariableOpReadVariableOpAbasemodel_batch_normalization_1_batchnorm_readvariableop_resource*
_output_shapes	
:ђ*
dtype02:
8basemodel/batch_normalization_1/batchnorm/ReadVariableOpД
/basemodel/batch_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:21
/basemodel/batch_normalization_1/batchnorm/add/yЅ
-basemodel/batch_normalization_1/batchnorm/addAddV2@basemodel/batch_normalization_1/batchnorm/ReadVariableOp:value:08basemodel/batch_normalization_1/batchnorm/add/y:output:0*
T0*
_output_shapes	
:ђ2/
-basemodel/batch_normalization_1/batchnorm/add─
/basemodel/batch_normalization_1/batchnorm/RsqrtRsqrt1basemodel/batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes	
:ђ21
/basemodel/batch_normalization_1/batchnorm/Rsqrt 
<basemodel/batch_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOpEbasemodel_batch_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes	
:ђ*
dtype02>
<basemodel/batch_normalization_1/batchnorm/mul/ReadVariableOpє
-basemodel/batch_normalization_1/batchnorm/mulMul3basemodel/batch_normalization_1/batchnorm/Rsqrt:y:0Dbasemodel/batch_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:ђ2/
-basemodel/batch_normalization_1/batchnorm/mul 
/basemodel/batch_normalization_1/batchnorm/mul_1Mul*basemodel/stream_0_conv_2/BiasAdd:output:01basemodel/batch_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:         }ђ21
/basemodel/batch_normalization_1/batchnorm/mul_1щ
:basemodel/batch_normalization_1/batchnorm/ReadVariableOp_1ReadVariableOpCbasemodel_batch_normalization_1_batchnorm_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype02<
:basemodel/batch_normalization_1/batchnorm/ReadVariableOp_1є
/basemodel/batch_normalization_1/batchnorm/mul_2MulBbasemodel/batch_normalization_1/batchnorm/ReadVariableOp_1:value:01basemodel/batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes	
:ђ21
/basemodel/batch_normalization_1/batchnorm/mul_2щ
:basemodel/batch_normalization_1/batchnorm/ReadVariableOp_2ReadVariableOpCbasemodel_batch_normalization_1_batchnorm_readvariableop_2_resource*
_output_shapes	
:ђ*
dtype02<
:basemodel/batch_normalization_1/batchnorm/ReadVariableOp_2ё
-basemodel/batch_normalization_1/batchnorm/subSubBbasemodel/batch_normalization_1/batchnorm/ReadVariableOp_2:value:03basemodel/batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:ђ2/
-basemodel/batch_normalization_1/batchnorm/subі
/basemodel/batch_normalization_1/batchnorm/add_1AddV23basemodel/batch_normalization_1/batchnorm/mul_1:z:01basemodel/batch_normalization_1/batchnorm/sub:z:0*
T0*,
_output_shapes
:         }ђ21
/basemodel/batch_normalization_1/batchnorm/add_1«
basemodel/activation_1/ReluRelu3basemodel/batch_normalization_1/batchnorm/add_1:z:0*
T0*,
_output_shapes
:         }ђ2
basemodel/activation_1/ReluХ
"basemodel/stream_0_drop_2/IdentityIdentity)basemodel/activation_1/Relu:activations:0*
T0*,
_output_shapes
:         }ђ2$
"basemodel/stream_0_drop_2/IdentityГ
/basemodel/stream_0_conv_3/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§        21
/basemodel/stream_0_conv_3/conv1d/ExpandDims/dimі
+basemodel/stream_0_conv_3/conv1d/ExpandDims
ExpandDims+basemodel/stream_0_drop_2/Identity:output:08basemodel/stream_0_conv_3/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         }ђ2-
+basemodel/stream_0_conv_3/conv1d/ExpandDimsѕ
<basemodel/stream_0_conv_3/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpEbasemodel_stream_0_conv_3_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:ђђ*
dtype02>
<basemodel/stream_0_conv_3/conv1d/ExpandDims_1/ReadVariableOpе
1basemodel/stream_0_conv_3/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 23
1basemodel/stream_0_conv_3/conv1d/ExpandDims_1/dimА
-basemodel/stream_0_conv_3/conv1d/ExpandDims_1
ExpandDimsDbasemodel/stream_0_conv_3/conv1d/ExpandDims_1/ReadVariableOp:value:0:basemodel/stream_0_conv_3/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:ђђ2/
-basemodel/stream_0_conv_3/conv1d/ExpandDims_1Ъ
 basemodel/stream_0_conv_3/conv1dConv2D4basemodel/stream_0_conv_3/conv1d/ExpandDims:output:06basemodel/stream_0_conv_3/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:         }ђ*
paddingSAME*
strides
2"
 basemodel/stream_0_conv_3/conv1dр
(basemodel/stream_0_conv_3/conv1d/SqueezeSqueeze)basemodel/stream_0_conv_3/conv1d:output:0*
T0*,
_output_shapes
:         }ђ*
squeeze_dims

§        2*
(basemodel/stream_0_conv_3/conv1d/Squeeze█
0basemodel/stream_0_conv_3/BiasAdd/ReadVariableOpReadVariableOp9basemodel_stream_0_conv_3_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype022
0basemodel/stream_0_conv_3/BiasAdd/ReadVariableOpш
!basemodel/stream_0_conv_3/BiasAddBiasAdd1basemodel/stream_0_conv_3/conv1d/Squeeze:output:08basemodel/stream_0_conv_3/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:         }ђ2#
!basemodel/stream_0_conv_3/BiasAddз
8basemodel/batch_normalization_2/batchnorm/ReadVariableOpReadVariableOpAbasemodel_batch_normalization_2_batchnorm_readvariableop_resource*
_output_shapes	
:ђ*
dtype02:
8basemodel/batch_normalization_2/batchnorm/ReadVariableOpД
/basemodel/batch_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:21
/basemodel/batch_normalization_2/batchnorm/add/yЅ
-basemodel/batch_normalization_2/batchnorm/addAddV2@basemodel/batch_normalization_2/batchnorm/ReadVariableOp:value:08basemodel/batch_normalization_2/batchnorm/add/y:output:0*
T0*
_output_shapes	
:ђ2/
-basemodel/batch_normalization_2/batchnorm/add─
/basemodel/batch_normalization_2/batchnorm/RsqrtRsqrt1basemodel/batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes	
:ђ21
/basemodel/batch_normalization_2/batchnorm/Rsqrt 
<basemodel/batch_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOpEbasemodel_batch_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes	
:ђ*
dtype02>
<basemodel/batch_normalization_2/batchnorm/mul/ReadVariableOpє
-basemodel/batch_normalization_2/batchnorm/mulMul3basemodel/batch_normalization_2/batchnorm/Rsqrt:y:0Dbasemodel/batch_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:ђ2/
-basemodel/batch_normalization_2/batchnorm/mul 
/basemodel/batch_normalization_2/batchnorm/mul_1Mul*basemodel/stream_0_conv_3/BiasAdd:output:01basemodel/batch_normalization_2/batchnorm/mul:z:0*
T0*,
_output_shapes
:         }ђ21
/basemodel/batch_normalization_2/batchnorm/mul_1щ
:basemodel/batch_normalization_2/batchnorm/ReadVariableOp_1ReadVariableOpCbasemodel_batch_normalization_2_batchnorm_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype02<
:basemodel/batch_normalization_2/batchnorm/ReadVariableOp_1є
/basemodel/batch_normalization_2/batchnorm/mul_2MulBbasemodel/batch_normalization_2/batchnorm/ReadVariableOp_1:value:01basemodel/batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes	
:ђ21
/basemodel/batch_normalization_2/batchnorm/mul_2щ
:basemodel/batch_normalization_2/batchnorm/ReadVariableOp_2ReadVariableOpCbasemodel_batch_normalization_2_batchnorm_readvariableop_2_resource*
_output_shapes	
:ђ*
dtype02<
:basemodel/batch_normalization_2/batchnorm/ReadVariableOp_2ё
-basemodel/batch_normalization_2/batchnorm/subSubBbasemodel/batch_normalization_2/batchnorm/ReadVariableOp_2:value:03basemodel/batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:ђ2/
-basemodel/batch_normalization_2/batchnorm/subі
/basemodel/batch_normalization_2/batchnorm/add_1AddV23basemodel/batch_normalization_2/batchnorm/mul_1:z:01basemodel/batch_normalization_2/batchnorm/sub:z:0*
T0*,
_output_shapes
:         }ђ21
/basemodel/batch_normalization_2/batchnorm/add_1«
basemodel/activation_2/ReluRelu3basemodel/batch_normalization_2/batchnorm/add_1:z:0*
T0*,
_output_shapes
:         }ђ2
basemodel/activation_2/ReluХ
"basemodel/stream_0_drop_3/IdentityIdentity)basemodel/activation_2/Relu:activations:0*
T0*,
_output_shapes
:         }ђ2$
"basemodel/stream_0_drop_3/IdentityИ
9basemodel/global_average_pooling1d/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2;
9basemodel/global_average_pooling1d/Mean/reduction_indices■
'basemodel/global_average_pooling1d/MeanMean+basemodel/stream_0_drop_3/Identity:output:0Bbasemodel/global_average_pooling1d/Mean/reduction_indices:output:0*
T0*(
_output_shapes
:         ђ2)
'basemodel/global_average_pooling1d/Mean╣
"basemodel/dense_1_dropout/IdentityIdentity0basemodel/global_average_pooling1d/Mean:output:0*
T0*(
_output_shapes
:         ђ2$
"basemodel/dense_1_dropout/Identity─
'basemodel/dense_1/MatMul/ReadVariableOpReadVariableOp0basemodel_dense_1_matmul_readvariableop_resource*
_output_shapes
:	ђT*
dtype02)
'basemodel/dense_1/MatMul/ReadVariableOp╬
basemodel/dense_1/MatMulMatMul+basemodel/dense_1_dropout/Identity:output:0/basemodel/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         T2
basemodel/dense_1/MatMul┬
(basemodel/dense_1/BiasAdd/ReadVariableOpReadVariableOp1basemodel_dense_1_biasadd_readvariableop_resource*
_output_shapes
:T*
dtype02*
(basemodel/dense_1/BiasAdd/ReadVariableOp╔
basemodel/dense_1/BiasAddBiasAdd"basemodel/dense_1/MatMul:product:00basemodel/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         T2
basemodel/dense_1/BiasAddЫ
8basemodel/batch_normalization_3/batchnorm/ReadVariableOpReadVariableOpAbasemodel_batch_normalization_3_batchnorm_readvariableop_resource*
_output_shapes
:T*
dtype02:
8basemodel/batch_normalization_3/batchnorm/ReadVariableOpД
/basemodel/batch_normalization_3/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:21
/basemodel/batch_normalization_3/batchnorm/add/yѕ
-basemodel/batch_normalization_3/batchnorm/addAddV2@basemodel/batch_normalization_3/batchnorm/ReadVariableOp:value:08basemodel/batch_normalization_3/batchnorm/add/y:output:0*
T0*
_output_shapes
:T2/
-basemodel/batch_normalization_3/batchnorm/add├
/basemodel/batch_normalization_3/batchnorm/RsqrtRsqrt1basemodel/batch_normalization_3/batchnorm/add:z:0*
T0*
_output_shapes
:T21
/basemodel/batch_normalization_3/batchnorm/Rsqrt■
<basemodel/batch_normalization_3/batchnorm/mul/ReadVariableOpReadVariableOpEbasemodel_batch_normalization_3_batchnorm_mul_readvariableop_resource*
_output_shapes
:T*
dtype02>
<basemodel/batch_normalization_3/batchnorm/mul/ReadVariableOpЁ
-basemodel/batch_normalization_3/batchnorm/mulMul3basemodel/batch_normalization_3/batchnorm/Rsqrt:y:0Dbasemodel/batch_normalization_3/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:T2/
-basemodel/batch_normalization_3/batchnorm/mulЫ
/basemodel/batch_normalization_3/batchnorm/mul_1Mul"basemodel/dense_1/BiasAdd:output:01basemodel/batch_normalization_3/batchnorm/mul:z:0*
T0*'
_output_shapes
:         T21
/basemodel/batch_normalization_3/batchnorm/mul_1Э
:basemodel/batch_normalization_3/batchnorm/ReadVariableOp_1ReadVariableOpCbasemodel_batch_normalization_3_batchnorm_readvariableop_1_resource*
_output_shapes
:T*
dtype02<
:basemodel/batch_normalization_3/batchnorm/ReadVariableOp_1Ё
/basemodel/batch_normalization_3/batchnorm/mul_2MulBbasemodel/batch_normalization_3/batchnorm/ReadVariableOp_1:value:01basemodel/batch_normalization_3/batchnorm/mul:z:0*
T0*
_output_shapes
:T21
/basemodel/batch_normalization_3/batchnorm/mul_2Э
:basemodel/batch_normalization_3/batchnorm/ReadVariableOp_2ReadVariableOpCbasemodel_batch_normalization_3_batchnorm_readvariableop_2_resource*
_output_shapes
:T*
dtype02<
:basemodel/batch_normalization_3/batchnorm/ReadVariableOp_2Ѓ
-basemodel/batch_normalization_3/batchnorm/subSubBbasemodel/batch_normalization_3/batchnorm/ReadVariableOp_2:value:03basemodel/batch_normalization_3/batchnorm/mul_2:z:0*
T0*
_output_shapes
:T2/
-basemodel/batch_normalization_3/batchnorm/subЁ
/basemodel/batch_normalization_3/batchnorm/add_1AddV23basemodel/batch_normalization_3/batchnorm/mul_1:z:01basemodel/batch_normalization_3/batchnorm/sub:z:0*
T0*'
_output_shapes
:         T21
/basemodel/batch_normalization_3/batchnorm/add_1Й
$basemodel/dense_activation_1/SigmoidSigmoid3basemodel/batch_normalization_3/batchnorm/add_1:z:0*
T0*'
_output_shapes
:         T2&
$basemodel/dense_activation_1/Sigmoidа
(basemodel/stream_0_input_drop/Identity_1Identityinputs_1*
T0*+
_output_shapes
:         }2*
(basemodel/stream_0_input_drop/Identity_1▒
1basemodel/stream_0_conv_1/conv1d_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§        23
1basemodel/stream_0_conv_1/conv1d_1/ExpandDims/dimЋ
-basemodel/stream_0_conv_1/conv1d_1/ExpandDims
ExpandDims1basemodel/stream_0_input_drop/Identity_1:output:0:basemodel/stream_0_conv_1/conv1d_1/ExpandDims/dim:output:0*
T0*/
_output_shapes
:         }2/
-basemodel/stream_0_conv_1/conv1d_1/ExpandDimsі
>basemodel/stream_0_conv_1/conv1d_1/ExpandDims_1/ReadVariableOpReadVariableOpEbasemodel_stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02@
>basemodel/stream_0_conv_1/conv1d_1/ExpandDims_1/ReadVariableOpг
3basemodel/stream_0_conv_1/conv1d_1/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 25
3basemodel/stream_0_conv_1/conv1d_1/ExpandDims_1/dimД
/basemodel/stream_0_conv_1/conv1d_1/ExpandDims_1
ExpandDimsFbasemodel/stream_0_conv_1/conv1d_1/ExpandDims_1/ReadVariableOp:value:0<basemodel/stream_0_conv_1/conv1d_1/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@21
/basemodel/stream_0_conv_1/conv1d_1/ExpandDims_1д
"basemodel/stream_0_conv_1/conv1d_1Conv2D6basemodel/stream_0_conv_1/conv1d_1/ExpandDims:output:08basemodel/stream_0_conv_1/conv1d_1/ExpandDims_1:output:0*
T0*/
_output_shapes
:         }@*
paddingSAME*
strides
2$
"basemodel/stream_0_conv_1/conv1d_1Т
*basemodel/stream_0_conv_1/conv1d_1/SqueezeSqueeze+basemodel/stream_0_conv_1/conv1d_1:output:0*
T0*+
_output_shapes
:         }@*
squeeze_dims

§        2,
*basemodel/stream_0_conv_1/conv1d_1/Squeezeя
2basemodel/stream_0_conv_1/BiasAdd_1/ReadVariableOpReadVariableOp9basemodel_stream_0_conv_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype024
2basemodel/stream_0_conv_1/BiasAdd_1/ReadVariableOpЧ
#basemodel/stream_0_conv_1/BiasAdd_1BiasAdd3basemodel/stream_0_conv_1/conv1d_1/Squeeze:output:0:basemodel/stream_0_conv_1/BiasAdd_1/ReadVariableOp:value:0*
T0*+
_output_shapes
:         }@2%
#basemodel/stream_0_conv_1/BiasAdd_1­
8basemodel/batch_normalization/batchnorm_1/ReadVariableOpReadVariableOp?basemodel_batch_normalization_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02:
8basemodel/batch_normalization/batchnorm_1/ReadVariableOpД
/basemodel/batch_normalization/batchnorm_1/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:21
/basemodel/batch_normalization/batchnorm_1/add/yѕ
-basemodel/batch_normalization/batchnorm_1/addAddV2@basemodel/batch_normalization/batchnorm_1/ReadVariableOp:value:08basemodel/batch_normalization/batchnorm_1/add/y:output:0*
T0*
_output_shapes
:@2/
-basemodel/batch_normalization/batchnorm_1/add├
/basemodel/batch_normalization/batchnorm_1/RsqrtRsqrt1basemodel/batch_normalization/batchnorm_1/add:z:0*
T0*
_output_shapes
:@21
/basemodel/batch_normalization/batchnorm_1/RsqrtЧ
<basemodel/batch_normalization/batchnorm_1/mul/ReadVariableOpReadVariableOpCbasemodel_batch_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02>
<basemodel/batch_normalization/batchnorm_1/mul/ReadVariableOpЁ
-basemodel/batch_normalization/batchnorm_1/mulMul3basemodel/batch_normalization/batchnorm_1/Rsqrt:y:0Dbasemodel/batch_normalization/batchnorm_1/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2/
-basemodel/batch_normalization/batchnorm_1/mulђ
/basemodel/batch_normalization/batchnorm_1/mul_1Mul,basemodel/stream_0_conv_1/BiasAdd_1:output:01basemodel/batch_normalization/batchnorm_1/mul:z:0*
T0*+
_output_shapes
:         }@21
/basemodel/batch_normalization/batchnorm_1/mul_1Ш
:basemodel/batch_normalization/batchnorm_1/ReadVariableOp_1ReadVariableOpAbasemodel_batch_normalization_batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype02<
:basemodel/batch_normalization/batchnorm_1/ReadVariableOp_1Ё
/basemodel/batch_normalization/batchnorm_1/mul_2MulBbasemodel/batch_normalization/batchnorm_1/ReadVariableOp_1:value:01basemodel/batch_normalization/batchnorm_1/mul:z:0*
T0*
_output_shapes
:@21
/basemodel/batch_normalization/batchnorm_1/mul_2Ш
:basemodel/batch_normalization/batchnorm_1/ReadVariableOp_2ReadVariableOpAbasemodel_batch_normalization_batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype02<
:basemodel/batch_normalization/batchnorm_1/ReadVariableOp_2Ѓ
-basemodel/batch_normalization/batchnorm_1/subSubBbasemodel/batch_normalization/batchnorm_1/ReadVariableOp_2:value:03basemodel/batch_normalization/batchnorm_1/mul_2:z:0*
T0*
_output_shapes
:@2/
-basemodel/batch_normalization/batchnorm_1/subЅ
/basemodel/batch_normalization/batchnorm_1/add_1AddV23basemodel/batch_normalization/batchnorm_1/mul_1:z:01basemodel/batch_normalization/batchnorm_1/sub:z:0*
T0*+
_output_shapes
:         }@21
/basemodel/batch_normalization/batchnorm_1/add_1Г
basemodel/activation/Relu_1Relu3basemodel/batch_normalization/batchnorm_1/add_1:z:0*
T0*+
_output_shapes
:         }@2
basemodel/activation/Relu_1╣
$basemodel/stream_0_drop_1/Identity_1Identity)basemodel/activation/Relu_1:activations:0*
T0*+
_output_shapes
:         }@2&
$basemodel/stream_0_drop_1/Identity_1▒
1basemodel/stream_0_conv_2/conv1d_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§        23
1basemodel/stream_0_conv_2/conv1d_1/ExpandDims/dimЉ
-basemodel/stream_0_conv_2/conv1d_1/ExpandDims
ExpandDims-basemodel/stream_0_drop_1/Identity_1:output:0:basemodel/stream_0_conv_2/conv1d_1/ExpandDims/dim:output:0*
T0*/
_output_shapes
:         }@2/
-basemodel/stream_0_conv_2/conv1d_1/ExpandDimsІ
>basemodel/stream_0_conv_2/conv1d_1/ExpandDims_1/ReadVariableOpReadVariableOpEbasemodel_stream_0_conv_2_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:@ђ*
dtype02@
>basemodel/stream_0_conv_2/conv1d_1/ExpandDims_1/ReadVariableOpг
3basemodel/stream_0_conv_2/conv1d_1/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 25
3basemodel/stream_0_conv_2/conv1d_1/ExpandDims_1/dimе
/basemodel/stream_0_conv_2/conv1d_1/ExpandDims_1
ExpandDimsFbasemodel/stream_0_conv_2/conv1d_1/ExpandDims_1/ReadVariableOp:value:0<basemodel/stream_0_conv_2/conv1d_1/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:@ђ21
/basemodel/stream_0_conv_2/conv1d_1/ExpandDims_1Д
"basemodel/stream_0_conv_2/conv1d_1Conv2D6basemodel/stream_0_conv_2/conv1d_1/ExpandDims:output:08basemodel/stream_0_conv_2/conv1d_1/ExpandDims_1:output:0*
T0*0
_output_shapes
:         }ђ*
paddingSAME*
strides
2$
"basemodel/stream_0_conv_2/conv1d_1у
*basemodel/stream_0_conv_2/conv1d_1/SqueezeSqueeze+basemodel/stream_0_conv_2/conv1d_1:output:0*
T0*,
_output_shapes
:         }ђ*
squeeze_dims

§        2,
*basemodel/stream_0_conv_2/conv1d_1/Squeeze▀
2basemodel/stream_0_conv_2/BiasAdd_1/ReadVariableOpReadVariableOp9basemodel_stream_0_conv_2_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype024
2basemodel/stream_0_conv_2/BiasAdd_1/ReadVariableOp§
#basemodel/stream_0_conv_2/BiasAdd_1BiasAdd3basemodel/stream_0_conv_2/conv1d_1/Squeeze:output:0:basemodel/stream_0_conv_2/BiasAdd_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:         }ђ2%
#basemodel/stream_0_conv_2/BiasAdd_1э
:basemodel/batch_normalization_1/batchnorm_1/ReadVariableOpReadVariableOpAbasemodel_batch_normalization_1_batchnorm_readvariableop_resource*
_output_shapes	
:ђ*
dtype02<
:basemodel/batch_normalization_1/batchnorm_1/ReadVariableOpФ
1basemodel/batch_normalization_1/batchnorm_1/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:23
1basemodel/batch_normalization_1/batchnorm_1/add/yЉ
/basemodel/batch_normalization_1/batchnorm_1/addAddV2Bbasemodel/batch_normalization_1/batchnorm_1/ReadVariableOp:value:0:basemodel/batch_normalization_1/batchnorm_1/add/y:output:0*
T0*
_output_shapes	
:ђ21
/basemodel/batch_normalization_1/batchnorm_1/add╩
1basemodel/batch_normalization_1/batchnorm_1/RsqrtRsqrt3basemodel/batch_normalization_1/batchnorm_1/add:z:0*
T0*
_output_shapes	
:ђ23
1basemodel/batch_normalization_1/batchnorm_1/RsqrtЃ
>basemodel/batch_normalization_1/batchnorm_1/mul/ReadVariableOpReadVariableOpEbasemodel_batch_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes	
:ђ*
dtype02@
>basemodel/batch_normalization_1/batchnorm_1/mul/ReadVariableOpј
/basemodel/batch_normalization_1/batchnorm_1/mulMul5basemodel/batch_normalization_1/batchnorm_1/Rsqrt:y:0Fbasemodel/batch_normalization_1/batchnorm_1/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:ђ21
/basemodel/batch_normalization_1/batchnorm_1/mulЄ
1basemodel/batch_normalization_1/batchnorm_1/mul_1Mul,basemodel/stream_0_conv_2/BiasAdd_1:output:03basemodel/batch_normalization_1/batchnorm_1/mul:z:0*
T0*,
_output_shapes
:         }ђ23
1basemodel/batch_normalization_1/batchnorm_1/mul_1§
<basemodel/batch_normalization_1/batchnorm_1/ReadVariableOp_1ReadVariableOpCbasemodel_batch_normalization_1_batchnorm_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype02>
<basemodel/batch_normalization_1/batchnorm_1/ReadVariableOp_1ј
1basemodel/batch_normalization_1/batchnorm_1/mul_2MulDbasemodel/batch_normalization_1/batchnorm_1/ReadVariableOp_1:value:03basemodel/batch_normalization_1/batchnorm_1/mul:z:0*
T0*
_output_shapes	
:ђ23
1basemodel/batch_normalization_1/batchnorm_1/mul_2§
<basemodel/batch_normalization_1/batchnorm_1/ReadVariableOp_2ReadVariableOpCbasemodel_batch_normalization_1_batchnorm_readvariableop_2_resource*
_output_shapes	
:ђ*
dtype02>
<basemodel/batch_normalization_1/batchnorm_1/ReadVariableOp_2ї
/basemodel/batch_normalization_1/batchnorm_1/subSubDbasemodel/batch_normalization_1/batchnorm_1/ReadVariableOp_2:value:05basemodel/batch_normalization_1/batchnorm_1/mul_2:z:0*
T0*
_output_shapes	
:ђ21
/basemodel/batch_normalization_1/batchnorm_1/subњ
1basemodel/batch_normalization_1/batchnorm_1/add_1AddV25basemodel/batch_normalization_1/batchnorm_1/mul_1:z:03basemodel/batch_normalization_1/batchnorm_1/sub:z:0*
T0*,
_output_shapes
:         }ђ23
1basemodel/batch_normalization_1/batchnorm_1/add_1┤
basemodel/activation_1/Relu_1Relu5basemodel/batch_normalization_1/batchnorm_1/add_1:z:0*
T0*,
_output_shapes
:         }ђ2
basemodel/activation_1/Relu_1╝
$basemodel/stream_0_drop_2/Identity_1Identity+basemodel/activation_1/Relu_1:activations:0*
T0*,
_output_shapes
:         }ђ2&
$basemodel/stream_0_drop_2/Identity_1▒
1basemodel/stream_0_conv_3/conv1d_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§        23
1basemodel/stream_0_conv_3/conv1d_1/ExpandDims/dimњ
-basemodel/stream_0_conv_3/conv1d_1/ExpandDims
ExpandDims-basemodel/stream_0_drop_2/Identity_1:output:0:basemodel/stream_0_conv_3/conv1d_1/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         }ђ2/
-basemodel/stream_0_conv_3/conv1d_1/ExpandDimsї
>basemodel/stream_0_conv_3/conv1d_1/ExpandDims_1/ReadVariableOpReadVariableOpEbasemodel_stream_0_conv_3_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:ђђ*
dtype02@
>basemodel/stream_0_conv_3/conv1d_1/ExpandDims_1/ReadVariableOpг
3basemodel/stream_0_conv_3/conv1d_1/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 25
3basemodel/stream_0_conv_3/conv1d_1/ExpandDims_1/dimЕ
/basemodel/stream_0_conv_3/conv1d_1/ExpandDims_1
ExpandDimsFbasemodel/stream_0_conv_3/conv1d_1/ExpandDims_1/ReadVariableOp:value:0<basemodel/stream_0_conv_3/conv1d_1/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:ђђ21
/basemodel/stream_0_conv_3/conv1d_1/ExpandDims_1Д
"basemodel/stream_0_conv_3/conv1d_1Conv2D6basemodel/stream_0_conv_3/conv1d_1/ExpandDims:output:08basemodel/stream_0_conv_3/conv1d_1/ExpandDims_1:output:0*
T0*0
_output_shapes
:         }ђ*
paddingSAME*
strides
2$
"basemodel/stream_0_conv_3/conv1d_1у
*basemodel/stream_0_conv_3/conv1d_1/SqueezeSqueeze+basemodel/stream_0_conv_3/conv1d_1:output:0*
T0*,
_output_shapes
:         }ђ*
squeeze_dims

§        2,
*basemodel/stream_0_conv_3/conv1d_1/Squeeze▀
2basemodel/stream_0_conv_3/BiasAdd_1/ReadVariableOpReadVariableOp9basemodel_stream_0_conv_3_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype024
2basemodel/stream_0_conv_3/BiasAdd_1/ReadVariableOp§
#basemodel/stream_0_conv_3/BiasAdd_1BiasAdd3basemodel/stream_0_conv_3/conv1d_1/Squeeze:output:0:basemodel/stream_0_conv_3/BiasAdd_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:         }ђ2%
#basemodel/stream_0_conv_3/BiasAdd_1э
:basemodel/batch_normalization_2/batchnorm_1/ReadVariableOpReadVariableOpAbasemodel_batch_normalization_2_batchnorm_readvariableop_resource*
_output_shapes	
:ђ*
dtype02<
:basemodel/batch_normalization_2/batchnorm_1/ReadVariableOpФ
1basemodel/batch_normalization_2/batchnorm_1/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:23
1basemodel/batch_normalization_2/batchnorm_1/add/yЉ
/basemodel/batch_normalization_2/batchnorm_1/addAddV2Bbasemodel/batch_normalization_2/batchnorm_1/ReadVariableOp:value:0:basemodel/batch_normalization_2/batchnorm_1/add/y:output:0*
T0*
_output_shapes	
:ђ21
/basemodel/batch_normalization_2/batchnorm_1/add╩
1basemodel/batch_normalization_2/batchnorm_1/RsqrtRsqrt3basemodel/batch_normalization_2/batchnorm_1/add:z:0*
T0*
_output_shapes	
:ђ23
1basemodel/batch_normalization_2/batchnorm_1/RsqrtЃ
>basemodel/batch_normalization_2/batchnorm_1/mul/ReadVariableOpReadVariableOpEbasemodel_batch_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes	
:ђ*
dtype02@
>basemodel/batch_normalization_2/batchnorm_1/mul/ReadVariableOpј
/basemodel/batch_normalization_2/batchnorm_1/mulMul5basemodel/batch_normalization_2/batchnorm_1/Rsqrt:y:0Fbasemodel/batch_normalization_2/batchnorm_1/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:ђ21
/basemodel/batch_normalization_2/batchnorm_1/mulЄ
1basemodel/batch_normalization_2/batchnorm_1/mul_1Mul,basemodel/stream_0_conv_3/BiasAdd_1:output:03basemodel/batch_normalization_2/batchnorm_1/mul:z:0*
T0*,
_output_shapes
:         }ђ23
1basemodel/batch_normalization_2/batchnorm_1/mul_1§
<basemodel/batch_normalization_2/batchnorm_1/ReadVariableOp_1ReadVariableOpCbasemodel_batch_normalization_2_batchnorm_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype02>
<basemodel/batch_normalization_2/batchnorm_1/ReadVariableOp_1ј
1basemodel/batch_normalization_2/batchnorm_1/mul_2MulDbasemodel/batch_normalization_2/batchnorm_1/ReadVariableOp_1:value:03basemodel/batch_normalization_2/batchnorm_1/mul:z:0*
T0*
_output_shapes	
:ђ23
1basemodel/batch_normalization_2/batchnorm_1/mul_2§
<basemodel/batch_normalization_2/batchnorm_1/ReadVariableOp_2ReadVariableOpCbasemodel_batch_normalization_2_batchnorm_readvariableop_2_resource*
_output_shapes	
:ђ*
dtype02>
<basemodel/batch_normalization_2/batchnorm_1/ReadVariableOp_2ї
/basemodel/batch_normalization_2/batchnorm_1/subSubDbasemodel/batch_normalization_2/batchnorm_1/ReadVariableOp_2:value:05basemodel/batch_normalization_2/batchnorm_1/mul_2:z:0*
T0*
_output_shapes	
:ђ21
/basemodel/batch_normalization_2/batchnorm_1/subњ
1basemodel/batch_normalization_2/batchnorm_1/add_1AddV25basemodel/batch_normalization_2/batchnorm_1/mul_1:z:03basemodel/batch_normalization_2/batchnorm_1/sub:z:0*
T0*,
_output_shapes
:         }ђ23
1basemodel/batch_normalization_2/batchnorm_1/add_1┤
basemodel/activation_2/Relu_1Relu5basemodel/batch_normalization_2/batchnorm_1/add_1:z:0*
T0*,
_output_shapes
:         }ђ2
basemodel/activation_2/Relu_1╝
$basemodel/stream_0_drop_3/Identity_1Identity+basemodel/activation_2/Relu_1:activations:0*
T0*,
_output_shapes
:         }ђ2&
$basemodel/stream_0_drop_3/Identity_1╝
;basemodel/global_average_pooling1d/Mean_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2=
;basemodel/global_average_pooling1d/Mean_1/reduction_indicesє
)basemodel/global_average_pooling1d/Mean_1Mean-basemodel/stream_0_drop_3/Identity_1:output:0Dbasemodel/global_average_pooling1d/Mean_1/reduction_indices:output:0*
T0*(
_output_shapes
:         ђ2+
)basemodel/global_average_pooling1d/Mean_1┐
$basemodel/dense_1_dropout/Identity_1Identity2basemodel/global_average_pooling1d/Mean_1:output:0*
T0*(
_output_shapes
:         ђ2&
$basemodel/dense_1_dropout/Identity_1╚
)basemodel/dense_1/MatMul_1/ReadVariableOpReadVariableOp0basemodel_dense_1_matmul_readvariableop_resource*
_output_shapes
:	ђT*
dtype02+
)basemodel/dense_1/MatMul_1/ReadVariableOpо
basemodel/dense_1/MatMul_1MatMul-basemodel/dense_1_dropout/Identity_1:output:01basemodel/dense_1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         T2
basemodel/dense_1/MatMul_1к
*basemodel/dense_1/BiasAdd_1/ReadVariableOpReadVariableOp1basemodel_dense_1_biasadd_readvariableop_resource*
_output_shapes
:T*
dtype02,
*basemodel/dense_1/BiasAdd_1/ReadVariableOpЛ
basemodel/dense_1/BiasAdd_1BiasAdd$basemodel/dense_1/MatMul_1:product:02basemodel/dense_1/BiasAdd_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         T2
basemodel/dense_1/BiasAdd_1Ш
:basemodel/batch_normalization_3/batchnorm_1/ReadVariableOpReadVariableOpAbasemodel_batch_normalization_3_batchnorm_readvariableop_resource*
_output_shapes
:T*
dtype02<
:basemodel/batch_normalization_3/batchnorm_1/ReadVariableOpФ
1basemodel/batch_normalization_3/batchnorm_1/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:23
1basemodel/batch_normalization_3/batchnorm_1/add/yљ
/basemodel/batch_normalization_3/batchnorm_1/addAddV2Bbasemodel/batch_normalization_3/batchnorm_1/ReadVariableOp:value:0:basemodel/batch_normalization_3/batchnorm_1/add/y:output:0*
T0*
_output_shapes
:T21
/basemodel/batch_normalization_3/batchnorm_1/add╔
1basemodel/batch_normalization_3/batchnorm_1/RsqrtRsqrt3basemodel/batch_normalization_3/batchnorm_1/add:z:0*
T0*
_output_shapes
:T23
1basemodel/batch_normalization_3/batchnorm_1/Rsqrtѓ
>basemodel/batch_normalization_3/batchnorm_1/mul/ReadVariableOpReadVariableOpEbasemodel_batch_normalization_3_batchnorm_mul_readvariableop_resource*
_output_shapes
:T*
dtype02@
>basemodel/batch_normalization_3/batchnorm_1/mul/ReadVariableOpЇ
/basemodel/batch_normalization_3/batchnorm_1/mulMul5basemodel/batch_normalization_3/batchnorm_1/Rsqrt:y:0Fbasemodel/batch_normalization_3/batchnorm_1/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:T21
/basemodel/batch_normalization_3/batchnorm_1/mulЩ
1basemodel/batch_normalization_3/batchnorm_1/mul_1Mul$basemodel/dense_1/BiasAdd_1:output:03basemodel/batch_normalization_3/batchnorm_1/mul:z:0*
T0*'
_output_shapes
:         T23
1basemodel/batch_normalization_3/batchnorm_1/mul_1Ч
<basemodel/batch_normalization_3/batchnorm_1/ReadVariableOp_1ReadVariableOpCbasemodel_batch_normalization_3_batchnorm_readvariableop_1_resource*
_output_shapes
:T*
dtype02>
<basemodel/batch_normalization_3/batchnorm_1/ReadVariableOp_1Ї
1basemodel/batch_normalization_3/batchnorm_1/mul_2MulDbasemodel/batch_normalization_3/batchnorm_1/ReadVariableOp_1:value:03basemodel/batch_normalization_3/batchnorm_1/mul:z:0*
T0*
_output_shapes
:T23
1basemodel/batch_normalization_3/batchnorm_1/mul_2Ч
<basemodel/batch_normalization_3/batchnorm_1/ReadVariableOp_2ReadVariableOpCbasemodel_batch_normalization_3_batchnorm_readvariableop_2_resource*
_output_shapes
:T*
dtype02>
<basemodel/batch_normalization_3/batchnorm_1/ReadVariableOp_2І
/basemodel/batch_normalization_3/batchnorm_1/subSubDbasemodel/batch_normalization_3/batchnorm_1/ReadVariableOp_2:value:05basemodel/batch_normalization_3/batchnorm_1/mul_2:z:0*
T0*
_output_shapes
:T21
/basemodel/batch_normalization_3/batchnorm_1/subЇ
1basemodel/batch_normalization_3/batchnorm_1/add_1AddV25basemodel/batch_normalization_3/batchnorm_1/mul_1:z:03basemodel/batch_normalization_3/batchnorm_1/sub:z:0*
T0*'
_output_shapes
:         T23
1basemodel/batch_normalization_3/batchnorm_1/add_1─
&basemodel/dense_activation_1/Sigmoid_1Sigmoid5basemodel/batch_normalization_3/batchnorm_1/add_1:z:0*
T0*'
_output_shapes
:         T2(
&basemodel/dense_activation_1/Sigmoid_1Ф
distance/subSub(basemodel/dense_activation_1/Sigmoid:y:0*basemodel/dense_activation_1/Sigmoid_1:y:0*
T0*'
_output_shapes
:         T2
distance/subp
distance/SquareSquaredistance/sub:z:0*
T0*'
_output_shapes
:         T2
distance/SquareІ
distance/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
         2 
distance/Sum/reduction_indicesц
distance/SumSumdistance/Square:y:0'distance/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:         *
	keep_dims(2
distance/Sume
distance/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
distance/ConstЉ
distance/MaximumMaximumdistance/Sum:output:0distance/Const:output:0*
T0*'
_output_shapes
:         2
distance/Maximumn
distance/SqrtSqrtdistance/Maximum:z:0*
T0*'
_output_shapes
:         2
distance/SqrtЭ
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpEbasemodel_stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype027
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp├
&stream_0_conv_1/kernel/Regularizer/AbsAbs=stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_0_conv_1/kernel/Regularizer/AbsЕ
(stream_0_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_0_conv_1/kernel/Regularizer/ConstО
&stream_0_conv_1/kernel/Regularizer/SumSum*stream_0_conv_1/kernel/Regularizer/Abs:y:01stream_0_conv_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/SumЎ
(stream_0_conv_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2*
(stream_0_conv_1/kernel/Regularizer/mul/x▄
&stream_0_conv_1/kernel/Regularizer/mulMul1stream_0_conv_1/kernel/Regularizer/mul/x:output:0/stream_0_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/mul 
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpEbasemodel_stream_0_conv_2_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:@ђ*
dtype02:
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOpл
)stream_0_conv_2/kernel/Regularizer/SquareSquare@stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*#
_output_shapes
:@ђ2+
)stream_0_conv_2/kernel/Regularizer/SquareЕ
(stream_0_conv_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_0_conv_2/kernel/Regularizer/Const┌
&stream_0_conv_2/kernel/Regularizer/SumSum-stream_0_conv_2/kernel/Regularizer/Square:y:01stream_0_conv_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_2/kernel/Regularizer/SumЎ
(stream_0_conv_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2*
(stream_0_conv_2/kernel/Regularizer/mul/x▄
&stream_0_conv_2/kernel/Regularizer/mulMul1stream_0_conv_2/kernel/Regularizer/mul/x:output:0/stream_0_conv_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_2/kernel/Regularizer/mulЩ
5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpEbasemodel_stream_0_conv_3_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:ђђ*
dtype027
5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp┼
&stream_0_conv_3/kernel/Regularizer/AbsAbs=stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*$
_output_shapes
:ђђ2(
&stream_0_conv_3/kernel/Regularizer/AbsЕ
(stream_0_conv_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_0_conv_3/kernel/Regularizer/ConstО
&stream_0_conv_3/kernel/Regularizer/SumSum*stream_0_conv_3/kernel/Regularizer/Abs:y:01stream_0_conv_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_3/kernel/Regularizer/SumЎ
(stream_0_conv_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2*
(stream_0_conv_3/kernel/Regularizer/mul/x▄
&stream_0_conv_3/kernel/Regularizer/mulMul1stream_0_conv_3/kernel/Regularizer/mul/x:output:0/stream_0_conv_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_3/kernel/Regularizer/mulл
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp0basemodel_dense_1_matmul_readvariableop_resource*
_output_shapes
:	ђT*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOpе
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	ђT2 
dense_1/kernel/Regularizer/AbsЋ
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_1/kernel/Regularizer/Constи
dense_1/kernel/Regularizer/SumSum"dense_1/kernel/Regularizer/Abs:y:0)dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/SumЅ
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2"
 dense_1/kernel/Regularizer/mul/x╝
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/mull
IdentityIdentitydistance/Sqrt:y:0^NoOp*
T0*'
_output_shapes
:         2

Identity┐
NoOpNoOp7^basemodel/batch_normalization/batchnorm/ReadVariableOp9^basemodel/batch_normalization/batchnorm/ReadVariableOp_19^basemodel/batch_normalization/batchnorm/ReadVariableOp_2;^basemodel/batch_normalization/batchnorm/mul/ReadVariableOp9^basemodel/batch_normalization/batchnorm_1/ReadVariableOp;^basemodel/batch_normalization/batchnorm_1/ReadVariableOp_1;^basemodel/batch_normalization/batchnorm_1/ReadVariableOp_2=^basemodel/batch_normalization/batchnorm_1/mul/ReadVariableOp9^basemodel/batch_normalization_1/batchnorm/ReadVariableOp;^basemodel/batch_normalization_1/batchnorm/ReadVariableOp_1;^basemodel/batch_normalization_1/batchnorm/ReadVariableOp_2=^basemodel/batch_normalization_1/batchnorm/mul/ReadVariableOp;^basemodel/batch_normalization_1/batchnorm_1/ReadVariableOp=^basemodel/batch_normalization_1/batchnorm_1/ReadVariableOp_1=^basemodel/batch_normalization_1/batchnorm_1/ReadVariableOp_2?^basemodel/batch_normalization_1/batchnorm_1/mul/ReadVariableOp9^basemodel/batch_normalization_2/batchnorm/ReadVariableOp;^basemodel/batch_normalization_2/batchnorm/ReadVariableOp_1;^basemodel/batch_normalization_2/batchnorm/ReadVariableOp_2=^basemodel/batch_normalization_2/batchnorm/mul/ReadVariableOp;^basemodel/batch_normalization_2/batchnorm_1/ReadVariableOp=^basemodel/batch_normalization_2/batchnorm_1/ReadVariableOp_1=^basemodel/batch_normalization_2/batchnorm_1/ReadVariableOp_2?^basemodel/batch_normalization_2/batchnorm_1/mul/ReadVariableOp9^basemodel/batch_normalization_3/batchnorm/ReadVariableOp;^basemodel/batch_normalization_3/batchnorm/ReadVariableOp_1;^basemodel/batch_normalization_3/batchnorm/ReadVariableOp_2=^basemodel/batch_normalization_3/batchnorm/mul/ReadVariableOp;^basemodel/batch_normalization_3/batchnorm_1/ReadVariableOp=^basemodel/batch_normalization_3/batchnorm_1/ReadVariableOp_1=^basemodel/batch_normalization_3/batchnorm_1/ReadVariableOp_2?^basemodel/batch_normalization_3/batchnorm_1/mul/ReadVariableOp)^basemodel/dense_1/BiasAdd/ReadVariableOp+^basemodel/dense_1/BiasAdd_1/ReadVariableOp(^basemodel/dense_1/MatMul/ReadVariableOp*^basemodel/dense_1/MatMul_1/ReadVariableOp1^basemodel/stream_0_conv_1/BiasAdd/ReadVariableOp3^basemodel/stream_0_conv_1/BiasAdd_1/ReadVariableOp=^basemodel/stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp?^basemodel/stream_0_conv_1/conv1d_1/ExpandDims_1/ReadVariableOp1^basemodel/stream_0_conv_2/BiasAdd/ReadVariableOp3^basemodel/stream_0_conv_2/BiasAdd_1/ReadVariableOp=^basemodel/stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOp?^basemodel/stream_0_conv_2/conv1d_1/ExpandDims_1/ReadVariableOp1^basemodel/stream_0_conv_3/BiasAdd/ReadVariableOp3^basemodel/stream_0_conv_3/BiasAdd_1/ReadVariableOp=^basemodel/stream_0_conv_3/conv1d/ExpandDims_1/ReadVariableOp?^basemodel/stream_0_conv_3/conv1d_1/ExpandDims_1/ReadVariableOp.^dense_1/kernel/Regularizer/Abs/ReadVariableOp6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp9^stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp6^stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*q
_input_shapes`
^:         }:         }: : : : : : : : : : : : : : : : : : : : : : : : 2p
6basemodel/batch_normalization/batchnorm/ReadVariableOp6basemodel/batch_normalization/batchnorm/ReadVariableOp2t
8basemodel/batch_normalization/batchnorm/ReadVariableOp_18basemodel/batch_normalization/batchnorm/ReadVariableOp_12t
8basemodel/batch_normalization/batchnorm/ReadVariableOp_28basemodel/batch_normalization/batchnorm/ReadVariableOp_22x
:basemodel/batch_normalization/batchnorm/mul/ReadVariableOp:basemodel/batch_normalization/batchnorm/mul/ReadVariableOp2t
8basemodel/batch_normalization/batchnorm_1/ReadVariableOp8basemodel/batch_normalization/batchnorm_1/ReadVariableOp2x
:basemodel/batch_normalization/batchnorm_1/ReadVariableOp_1:basemodel/batch_normalization/batchnorm_1/ReadVariableOp_12x
:basemodel/batch_normalization/batchnorm_1/ReadVariableOp_2:basemodel/batch_normalization/batchnorm_1/ReadVariableOp_22|
<basemodel/batch_normalization/batchnorm_1/mul/ReadVariableOp<basemodel/batch_normalization/batchnorm_1/mul/ReadVariableOp2t
8basemodel/batch_normalization_1/batchnorm/ReadVariableOp8basemodel/batch_normalization_1/batchnorm/ReadVariableOp2x
:basemodel/batch_normalization_1/batchnorm/ReadVariableOp_1:basemodel/batch_normalization_1/batchnorm/ReadVariableOp_12x
:basemodel/batch_normalization_1/batchnorm/ReadVariableOp_2:basemodel/batch_normalization_1/batchnorm/ReadVariableOp_22|
<basemodel/batch_normalization_1/batchnorm/mul/ReadVariableOp<basemodel/batch_normalization_1/batchnorm/mul/ReadVariableOp2x
:basemodel/batch_normalization_1/batchnorm_1/ReadVariableOp:basemodel/batch_normalization_1/batchnorm_1/ReadVariableOp2|
<basemodel/batch_normalization_1/batchnorm_1/ReadVariableOp_1<basemodel/batch_normalization_1/batchnorm_1/ReadVariableOp_12|
<basemodel/batch_normalization_1/batchnorm_1/ReadVariableOp_2<basemodel/batch_normalization_1/batchnorm_1/ReadVariableOp_22ђ
>basemodel/batch_normalization_1/batchnorm_1/mul/ReadVariableOp>basemodel/batch_normalization_1/batchnorm_1/mul/ReadVariableOp2t
8basemodel/batch_normalization_2/batchnorm/ReadVariableOp8basemodel/batch_normalization_2/batchnorm/ReadVariableOp2x
:basemodel/batch_normalization_2/batchnorm/ReadVariableOp_1:basemodel/batch_normalization_2/batchnorm/ReadVariableOp_12x
:basemodel/batch_normalization_2/batchnorm/ReadVariableOp_2:basemodel/batch_normalization_2/batchnorm/ReadVariableOp_22|
<basemodel/batch_normalization_2/batchnorm/mul/ReadVariableOp<basemodel/batch_normalization_2/batchnorm/mul/ReadVariableOp2x
:basemodel/batch_normalization_2/batchnorm_1/ReadVariableOp:basemodel/batch_normalization_2/batchnorm_1/ReadVariableOp2|
<basemodel/batch_normalization_2/batchnorm_1/ReadVariableOp_1<basemodel/batch_normalization_2/batchnorm_1/ReadVariableOp_12|
<basemodel/batch_normalization_2/batchnorm_1/ReadVariableOp_2<basemodel/batch_normalization_2/batchnorm_1/ReadVariableOp_22ђ
>basemodel/batch_normalization_2/batchnorm_1/mul/ReadVariableOp>basemodel/batch_normalization_2/batchnorm_1/mul/ReadVariableOp2t
8basemodel/batch_normalization_3/batchnorm/ReadVariableOp8basemodel/batch_normalization_3/batchnorm/ReadVariableOp2x
:basemodel/batch_normalization_3/batchnorm/ReadVariableOp_1:basemodel/batch_normalization_3/batchnorm/ReadVariableOp_12x
:basemodel/batch_normalization_3/batchnorm/ReadVariableOp_2:basemodel/batch_normalization_3/batchnorm/ReadVariableOp_22|
<basemodel/batch_normalization_3/batchnorm/mul/ReadVariableOp<basemodel/batch_normalization_3/batchnorm/mul/ReadVariableOp2x
:basemodel/batch_normalization_3/batchnorm_1/ReadVariableOp:basemodel/batch_normalization_3/batchnorm_1/ReadVariableOp2|
<basemodel/batch_normalization_3/batchnorm_1/ReadVariableOp_1<basemodel/batch_normalization_3/batchnorm_1/ReadVariableOp_12|
<basemodel/batch_normalization_3/batchnorm_1/ReadVariableOp_2<basemodel/batch_normalization_3/batchnorm_1/ReadVariableOp_22ђ
>basemodel/batch_normalization_3/batchnorm_1/mul/ReadVariableOp>basemodel/batch_normalization_3/batchnorm_1/mul/ReadVariableOp2T
(basemodel/dense_1/BiasAdd/ReadVariableOp(basemodel/dense_1/BiasAdd/ReadVariableOp2X
*basemodel/dense_1/BiasAdd_1/ReadVariableOp*basemodel/dense_1/BiasAdd_1/ReadVariableOp2R
'basemodel/dense_1/MatMul/ReadVariableOp'basemodel/dense_1/MatMul/ReadVariableOp2V
)basemodel/dense_1/MatMul_1/ReadVariableOp)basemodel/dense_1/MatMul_1/ReadVariableOp2d
0basemodel/stream_0_conv_1/BiasAdd/ReadVariableOp0basemodel/stream_0_conv_1/BiasAdd/ReadVariableOp2h
2basemodel/stream_0_conv_1/BiasAdd_1/ReadVariableOp2basemodel/stream_0_conv_1/BiasAdd_1/ReadVariableOp2|
<basemodel/stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp<basemodel/stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp2ђ
>basemodel/stream_0_conv_1/conv1d_1/ExpandDims_1/ReadVariableOp>basemodel/stream_0_conv_1/conv1d_1/ExpandDims_1/ReadVariableOp2d
0basemodel/stream_0_conv_2/BiasAdd/ReadVariableOp0basemodel/stream_0_conv_2/BiasAdd/ReadVariableOp2h
2basemodel/stream_0_conv_2/BiasAdd_1/ReadVariableOp2basemodel/stream_0_conv_2/BiasAdd_1/ReadVariableOp2|
<basemodel/stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOp<basemodel/stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOp2ђ
>basemodel/stream_0_conv_2/conv1d_1/ExpandDims_1/ReadVariableOp>basemodel/stream_0_conv_2/conv1d_1/ExpandDims_1/ReadVariableOp2d
0basemodel/stream_0_conv_3/BiasAdd/ReadVariableOp0basemodel/stream_0_conv_3/BiasAdd/ReadVariableOp2h
2basemodel/stream_0_conv_3/BiasAdd_1/ReadVariableOp2basemodel/stream_0_conv_3/BiasAdd_1/ReadVariableOp2|
<basemodel/stream_0_conv_3/conv1d/ExpandDims_1/ReadVariableOp<basemodel/stream_0_conv_3/conv1d/ExpandDims_1/ReadVariableOp2ђ
>basemodel/stream_0_conv_3/conv1d_1/ExpandDims_1/ReadVariableOp>basemodel/stream_0_conv_3/conv1d_1/ExpandDims_1/ReadVariableOp2^
-dense_1/kernel/Regularizer/Abs/ReadVariableOp-dense_1/kernel/Regularizer/Abs/ReadVariableOp2n
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp2t
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp2n
5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp:U Q
+
_output_shapes
:         }
"
_user_specified_name
inputs/0:UQ
+
_output_shapes
:         }
"
_user_specified_name
inputs/1
Њ	
н
5__inference_batch_normalization_1_layer_call_fn_28380

inputs
unknown:	ђ
	unknown_0:	ђ
	unknown_1:	ђ
	unknown_2:	ђ
identityѕбStatefulPartitionedCallЕ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:                  ђ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_234442
StatefulPartitionedCallЅ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:                  ђ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):                  ђ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:                  ђ
 
_user_specified_nameinputs
З
i
J__inference_stream_0_drop_3_layer_call_and_return_conditional_losses_24286

inputs
identityѕc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Constx
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:         }ђ2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shapeн
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:         }ђ*
dtype0*
seedи*
seed2и2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/y├
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:         }ђ2
dropout/GreaterEqualё
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:         }ђ2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:         }ђ2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:         }ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         }ђ:T P
,
_output_shapes
:         }ђ
 
_user_specified_nameinputs
┬
h
/__inference_stream_0_drop_3_layer_call_fn_28767

inputs
identityѕбStatefulPartitionedCallУ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         }ђ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *S
fNRL
J__inference_stream_0_drop_3_layer_call_and_return_conditional_losses_242862
StatefulPartitionedCallђ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:         }ђ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         }ђ22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:         }ђ
 
_user_specified_nameinputs
еЎ
Г/
@__inference_model_layer_call_and_return_conditional_losses_27036
inputs_0
inputs_1[
Ebasemodel_stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource:@G
9basemodel_stream_0_conv_1_biasadd_readvariableop_resource:@S
Ebasemodel_batch_normalization_assignmovingavg_readvariableop_resource:@U
Gbasemodel_batch_normalization_assignmovingavg_1_readvariableop_resource:@Q
Cbasemodel_batch_normalization_batchnorm_mul_readvariableop_resource:@M
?basemodel_batch_normalization_batchnorm_readvariableop_resource:@\
Ebasemodel_stream_0_conv_2_conv1d_expanddims_1_readvariableop_resource:@ђH
9basemodel_stream_0_conv_2_biasadd_readvariableop_resource:	ђV
Gbasemodel_batch_normalization_1_assignmovingavg_readvariableop_resource:	ђX
Ibasemodel_batch_normalization_1_assignmovingavg_1_readvariableop_resource:	ђT
Ebasemodel_batch_normalization_1_batchnorm_mul_readvariableop_resource:	ђP
Abasemodel_batch_normalization_1_batchnorm_readvariableop_resource:	ђ]
Ebasemodel_stream_0_conv_3_conv1d_expanddims_1_readvariableop_resource:ђђH
9basemodel_stream_0_conv_3_biasadd_readvariableop_resource:	ђV
Gbasemodel_batch_normalization_2_assignmovingavg_readvariableop_resource:	ђX
Ibasemodel_batch_normalization_2_assignmovingavg_1_readvariableop_resource:	ђT
Ebasemodel_batch_normalization_2_batchnorm_mul_readvariableop_resource:	ђP
Abasemodel_batch_normalization_2_batchnorm_readvariableop_resource:	ђC
0basemodel_dense_1_matmul_readvariableop_resource:	ђT?
1basemodel_dense_1_biasadd_readvariableop_resource:TU
Gbasemodel_batch_normalization_3_assignmovingavg_readvariableop_resource:TW
Ibasemodel_batch_normalization_3_assignmovingavg_1_readvariableop_resource:TS
Ebasemodel_batch_normalization_3_batchnorm_mul_readvariableop_resource:TO
Abasemodel_batch_normalization_3_batchnorm_readvariableop_resource:T
identityѕб-basemodel/batch_normalization/AssignMovingAvgб<basemodel/batch_normalization/AssignMovingAvg/ReadVariableOpб/basemodel/batch_normalization/AssignMovingAvg_1б>basemodel/batch_normalization/AssignMovingAvg_1/ReadVariableOpб/basemodel/batch_normalization/AssignMovingAvg_2б>basemodel/batch_normalization/AssignMovingAvg_2/ReadVariableOpб/basemodel/batch_normalization/AssignMovingAvg_3б>basemodel/batch_normalization/AssignMovingAvg_3/ReadVariableOpб6basemodel/batch_normalization/batchnorm/ReadVariableOpб:basemodel/batch_normalization/batchnorm/mul/ReadVariableOpб8basemodel/batch_normalization/batchnorm_1/ReadVariableOpб<basemodel/batch_normalization/batchnorm_1/mul/ReadVariableOpб/basemodel/batch_normalization_1/AssignMovingAvgб>basemodel/batch_normalization_1/AssignMovingAvg/ReadVariableOpб1basemodel/batch_normalization_1/AssignMovingAvg_1б@basemodel/batch_normalization_1/AssignMovingAvg_1/ReadVariableOpб1basemodel/batch_normalization_1/AssignMovingAvg_2б@basemodel/batch_normalization_1/AssignMovingAvg_2/ReadVariableOpб1basemodel/batch_normalization_1/AssignMovingAvg_3б@basemodel/batch_normalization_1/AssignMovingAvg_3/ReadVariableOpб8basemodel/batch_normalization_1/batchnorm/ReadVariableOpб<basemodel/batch_normalization_1/batchnorm/mul/ReadVariableOpб:basemodel/batch_normalization_1/batchnorm_1/ReadVariableOpб>basemodel/batch_normalization_1/batchnorm_1/mul/ReadVariableOpб/basemodel/batch_normalization_2/AssignMovingAvgб>basemodel/batch_normalization_2/AssignMovingAvg/ReadVariableOpб1basemodel/batch_normalization_2/AssignMovingAvg_1б@basemodel/batch_normalization_2/AssignMovingAvg_1/ReadVariableOpб1basemodel/batch_normalization_2/AssignMovingAvg_2б@basemodel/batch_normalization_2/AssignMovingAvg_2/ReadVariableOpб1basemodel/batch_normalization_2/AssignMovingAvg_3б@basemodel/batch_normalization_2/AssignMovingAvg_3/ReadVariableOpб8basemodel/batch_normalization_2/batchnorm/ReadVariableOpб<basemodel/batch_normalization_2/batchnorm/mul/ReadVariableOpб:basemodel/batch_normalization_2/batchnorm_1/ReadVariableOpб>basemodel/batch_normalization_2/batchnorm_1/mul/ReadVariableOpб/basemodel/batch_normalization_3/AssignMovingAvgб>basemodel/batch_normalization_3/AssignMovingAvg/ReadVariableOpб1basemodel/batch_normalization_3/AssignMovingAvg_1б@basemodel/batch_normalization_3/AssignMovingAvg_1/ReadVariableOpб1basemodel/batch_normalization_3/AssignMovingAvg_2б@basemodel/batch_normalization_3/AssignMovingAvg_2/ReadVariableOpб1basemodel/batch_normalization_3/AssignMovingAvg_3б@basemodel/batch_normalization_3/AssignMovingAvg_3/ReadVariableOpб8basemodel/batch_normalization_3/batchnorm/ReadVariableOpб<basemodel/batch_normalization_3/batchnorm/mul/ReadVariableOpб:basemodel/batch_normalization_3/batchnorm_1/ReadVariableOpб>basemodel/batch_normalization_3/batchnorm_1/mul/ReadVariableOpб(basemodel/dense_1/BiasAdd/ReadVariableOpб*basemodel/dense_1/BiasAdd_1/ReadVariableOpб'basemodel/dense_1/MatMul/ReadVariableOpб)basemodel/dense_1/MatMul_1/ReadVariableOpб0basemodel/stream_0_conv_1/BiasAdd/ReadVariableOpб2basemodel/stream_0_conv_1/BiasAdd_1/ReadVariableOpб<basemodel/stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOpб>basemodel/stream_0_conv_1/conv1d_1/ExpandDims_1/ReadVariableOpб0basemodel/stream_0_conv_2/BiasAdd/ReadVariableOpб2basemodel/stream_0_conv_2/BiasAdd_1/ReadVariableOpб<basemodel/stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOpб>basemodel/stream_0_conv_2/conv1d_1/ExpandDims_1/ReadVariableOpб0basemodel/stream_0_conv_3/BiasAdd/ReadVariableOpб2basemodel/stream_0_conv_3/BiasAdd_1/ReadVariableOpб<basemodel/stream_0_conv_3/conv1d/ExpandDims_1/ReadVariableOpб>basemodel/stream_0_conv_3/conv1d_1/ExpandDims_1/ReadVariableOpб-dense_1/kernel/Regularizer/Abs/ReadVariableOpб5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpб8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOpб5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOpЪ
+basemodel/stream_0_input_drop/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?2-
+basemodel/stream_0_input_drop/dropout/ConstМ
)basemodel/stream_0_input_drop/dropout/MulMulinputs_04basemodel/stream_0_input_drop/dropout/Const:output:0*
T0*+
_output_shapes
:         }2+
)basemodel/stream_0_input_drop/dropout/Mulњ
+basemodel/stream_0_input_drop/dropout/ShapeShapeinputs_0*
T0*
_output_shapes
:2-
+basemodel/stream_0_input_drop/dropout/ShapeГ
Bbasemodel/stream_0_input_drop/dropout/random_uniform/RandomUniformRandomUniform4basemodel/stream_0_input_drop/dropout/Shape:output:0*
T0*+
_output_shapes
:         }*
dtype0*
seedи*
seed2и2D
Bbasemodel/stream_0_input_drop/dropout/random_uniform/RandomUniform▒
4basemodel/stream_0_input_drop/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>26
4basemodel/stream_0_input_drop/dropout/GreaterEqual/y║
2basemodel/stream_0_input_drop/dropout/GreaterEqualGreaterEqualKbasemodel/stream_0_input_drop/dropout/random_uniform/RandomUniform:output:0=basemodel/stream_0_input_drop/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:         }24
2basemodel/stream_0_input_drop/dropout/GreaterEqualП
*basemodel/stream_0_input_drop/dropout/CastCast6basemodel/stream_0_input_drop/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:         }2,
*basemodel/stream_0_input_drop/dropout/CastШ
+basemodel/stream_0_input_drop/dropout/Mul_1Mul-basemodel/stream_0_input_drop/dropout/Mul:z:0.basemodel/stream_0_input_drop/dropout/Cast:y:0*
T0*+
_output_shapes
:         }2-
+basemodel/stream_0_input_drop/dropout/Mul_1Г
/basemodel/stream_0_conv_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§        21
/basemodel/stream_0_conv_1/conv1d/ExpandDims/dimЇ
+basemodel/stream_0_conv_1/conv1d/ExpandDims
ExpandDims/basemodel/stream_0_input_drop/dropout/Mul_1:z:08basemodel/stream_0_conv_1/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:         }2-
+basemodel/stream_0_conv_1/conv1d/ExpandDimsє
<basemodel/stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpEbasemodel_stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02>
<basemodel/stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOpе
1basemodel/stream_0_conv_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 23
1basemodel/stream_0_conv_1/conv1d/ExpandDims_1/dimЪ
-basemodel/stream_0_conv_1/conv1d/ExpandDims_1
ExpandDimsDbasemodel/stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp:value:0:basemodel/stream_0_conv_1/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@2/
-basemodel/stream_0_conv_1/conv1d/ExpandDims_1ъ
 basemodel/stream_0_conv_1/conv1dConv2D4basemodel/stream_0_conv_1/conv1d/ExpandDims:output:06basemodel/stream_0_conv_1/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:         }@*
paddingSAME*
strides
2"
 basemodel/stream_0_conv_1/conv1dЯ
(basemodel/stream_0_conv_1/conv1d/SqueezeSqueeze)basemodel/stream_0_conv_1/conv1d:output:0*
T0*+
_output_shapes
:         }@*
squeeze_dims

§        2*
(basemodel/stream_0_conv_1/conv1d/Squeeze┌
0basemodel/stream_0_conv_1/BiasAdd/ReadVariableOpReadVariableOp9basemodel_stream_0_conv_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype022
0basemodel/stream_0_conv_1/BiasAdd/ReadVariableOpЗ
!basemodel/stream_0_conv_1/BiasAddBiasAdd1basemodel/stream_0_conv_1/conv1d/Squeeze:output:08basemodel/stream_0_conv_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         }@2#
!basemodel/stream_0_conv_1/BiasAdd═
<basemodel/batch_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2>
<basemodel/batch_normalization/moments/mean/reduction_indicesЉ
*basemodel/batch_normalization/moments/meanMean*basemodel/stream_0_conv_1/BiasAdd:output:0Ebasemodel/batch_normalization/moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(2,
*basemodel/batch_normalization/moments/mean┌
2basemodel/batch_normalization/moments/StopGradientStopGradient3basemodel/batch_normalization/moments/mean:output:0*
T0*"
_output_shapes
:@24
2basemodel/batch_normalization/moments/StopGradientд
7basemodel/batch_normalization/moments/SquaredDifferenceSquaredDifference*basemodel/stream_0_conv_1/BiasAdd:output:0;basemodel/batch_normalization/moments/StopGradient:output:0*
T0*+
_output_shapes
:         }@29
7basemodel/batch_normalization/moments/SquaredDifferenceН
@basemodel/batch_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2B
@basemodel/batch_normalization/moments/variance/reduction_indices«
.basemodel/batch_normalization/moments/varianceMean;basemodel/batch_normalization/moments/SquaredDifference:z:0Ibasemodel/batch_normalization/moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(20
.basemodel/batch_normalization/moments/variance█
-basemodel/batch_normalization/moments/SqueezeSqueeze3basemodel/batch_normalization/moments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 2/
-basemodel/batch_normalization/moments/Squeezeс
/basemodel/batch_normalization/moments/Squeeze_1Squeeze7basemodel/batch_normalization/moments/variance:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 21
/basemodel/batch_normalization/moments/Squeeze_1»
3basemodel/batch_normalization/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<25
3basemodel/batch_normalization/AssignMovingAvg/decay■
<basemodel/batch_normalization/AssignMovingAvg/ReadVariableOpReadVariableOpEbasemodel_batch_normalization_assignmovingavg_readvariableop_resource*
_output_shapes
:@*
dtype02>
<basemodel/batch_normalization/AssignMovingAvg/ReadVariableOpљ
1basemodel/batch_normalization/AssignMovingAvg/subSubDbasemodel/batch_normalization/AssignMovingAvg/ReadVariableOp:value:06basemodel/batch_normalization/moments/Squeeze:output:0*
T0*
_output_shapes
:@23
1basemodel/batch_normalization/AssignMovingAvg/subЄ
1basemodel/batch_normalization/AssignMovingAvg/mulMul5basemodel/batch_normalization/AssignMovingAvg/sub:z:0<basemodel/batch_normalization/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:@23
1basemodel/batch_normalization/AssignMovingAvg/mulН
-basemodel/batch_normalization/AssignMovingAvgAssignSubVariableOpEbasemodel_batch_normalization_assignmovingavg_readvariableop_resource5basemodel/batch_normalization/AssignMovingAvg/mul:z:0=^basemodel/batch_normalization/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02/
-basemodel/batch_normalization/AssignMovingAvg│
5basemodel/batch_normalization/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<27
5basemodel/batch_normalization/AssignMovingAvg_1/decayё
>basemodel/batch_normalization/AssignMovingAvg_1/ReadVariableOpReadVariableOpGbasemodel_batch_normalization_assignmovingavg_1_readvariableop_resource*
_output_shapes
:@*
dtype02@
>basemodel/batch_normalization/AssignMovingAvg_1/ReadVariableOpў
3basemodel/batch_normalization/AssignMovingAvg_1/subSubFbasemodel/batch_normalization/AssignMovingAvg_1/ReadVariableOp:value:08basemodel/batch_normalization/moments/Squeeze_1:output:0*
T0*
_output_shapes
:@25
3basemodel/batch_normalization/AssignMovingAvg_1/subЈ
3basemodel/batch_normalization/AssignMovingAvg_1/mulMul7basemodel/batch_normalization/AssignMovingAvg_1/sub:z:0>basemodel/batch_normalization/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:@25
3basemodel/batch_normalization/AssignMovingAvg_1/mul▀
/basemodel/batch_normalization/AssignMovingAvg_1AssignSubVariableOpGbasemodel_batch_normalization_assignmovingavg_1_readvariableop_resource7basemodel/batch_normalization/AssignMovingAvg_1/mul:z:0?^basemodel/batch_normalization/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype021
/basemodel/batch_normalization/AssignMovingAvg_1Б
-basemodel/batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2/
-basemodel/batch_normalization/batchnorm/add/yЩ
+basemodel/batch_normalization/batchnorm/addAddV28basemodel/batch_normalization/moments/Squeeze_1:output:06basemodel/batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes
:@2-
+basemodel/batch_normalization/batchnorm/addй
-basemodel/batch_normalization/batchnorm/RsqrtRsqrt/basemodel/batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes
:@2/
-basemodel/batch_normalization/batchnorm/RsqrtЭ
:basemodel/batch_normalization/batchnorm/mul/ReadVariableOpReadVariableOpCbasemodel_batch_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02<
:basemodel/batch_normalization/batchnorm/mul/ReadVariableOp§
+basemodel/batch_normalization/batchnorm/mulMul1basemodel/batch_normalization/batchnorm/Rsqrt:y:0Bbasemodel/batch_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2-
+basemodel/batch_normalization/batchnorm/mulЭ
-basemodel/batch_normalization/batchnorm/mul_1Mul*basemodel/stream_0_conv_1/BiasAdd:output:0/basemodel/batch_normalization/batchnorm/mul:z:0*
T0*+
_output_shapes
:         }@2/
-basemodel/batch_normalization/batchnorm/mul_1з
-basemodel/batch_normalization/batchnorm/mul_2Mul6basemodel/batch_normalization/moments/Squeeze:output:0/basemodel/batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes
:@2/
-basemodel/batch_normalization/batchnorm/mul_2В
6basemodel/batch_normalization/batchnorm/ReadVariableOpReadVariableOp?basemodel_batch_normalization_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype028
6basemodel/batch_normalization/batchnorm/ReadVariableOpщ
+basemodel/batch_normalization/batchnorm/subSub>basemodel/batch_normalization/batchnorm/ReadVariableOp:value:01basemodel/batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2-
+basemodel/batch_normalization/batchnorm/subЂ
-basemodel/batch_normalization/batchnorm/add_1AddV21basemodel/batch_normalization/batchnorm/mul_1:z:0/basemodel/batch_normalization/batchnorm/sub:z:0*
T0*+
_output_shapes
:         }@2/
-basemodel/batch_normalization/batchnorm/add_1Д
basemodel/activation/ReluRelu1basemodel/batch_normalization/batchnorm/add_1:z:0*
T0*+
_output_shapes
:         }@2
basemodel/activation/ReluЌ
'basemodel/stream_0_drop_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n█Х?2)
'basemodel/stream_0_drop_1/dropout/ConstТ
%basemodel/stream_0_drop_1/dropout/MulMul'basemodel/activation/Relu:activations:00basemodel/stream_0_drop_1/dropout/Const:output:0*
T0*+
_output_shapes
:         }@2'
%basemodel/stream_0_drop_1/dropout/MulЕ
'basemodel/stream_0_drop_1/dropout/ShapeShape'basemodel/activation/Relu:activations:0*
T0*
_output_shapes
:2)
'basemodel/stream_0_drop_1/dropout/ShapeА
>basemodel/stream_0_drop_1/dropout/random_uniform/RandomUniformRandomUniform0basemodel/stream_0_drop_1/dropout/Shape:output:0*
T0*+
_output_shapes
:         }@*
dtype0*
seedи*
seed2и2@
>basemodel/stream_0_drop_1/dropout/random_uniform/RandomUniformЕ
0basemodel/stream_0_drop_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *џЎЎ>22
0basemodel/stream_0_drop_1/dropout/GreaterEqual/yф
.basemodel/stream_0_drop_1/dropout/GreaterEqualGreaterEqualGbasemodel/stream_0_drop_1/dropout/random_uniform/RandomUniform:output:09basemodel/stream_0_drop_1/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:         }@20
.basemodel/stream_0_drop_1/dropout/GreaterEqualЛ
&basemodel/stream_0_drop_1/dropout/CastCast2basemodel/stream_0_drop_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:         }@2(
&basemodel/stream_0_drop_1/dropout/CastТ
'basemodel/stream_0_drop_1/dropout/Mul_1Mul)basemodel/stream_0_drop_1/dropout/Mul:z:0*basemodel/stream_0_drop_1/dropout/Cast:y:0*
T0*+
_output_shapes
:         }@2)
'basemodel/stream_0_drop_1/dropout/Mul_1Г
/basemodel/stream_0_conv_2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§        21
/basemodel/stream_0_conv_2/conv1d/ExpandDims/dimЅ
+basemodel/stream_0_conv_2/conv1d/ExpandDims
ExpandDims+basemodel/stream_0_drop_1/dropout/Mul_1:z:08basemodel/stream_0_conv_2/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:         }@2-
+basemodel/stream_0_conv_2/conv1d/ExpandDimsЄ
<basemodel/stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpEbasemodel_stream_0_conv_2_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:@ђ*
dtype02>
<basemodel/stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOpе
1basemodel/stream_0_conv_2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 23
1basemodel/stream_0_conv_2/conv1d/ExpandDims_1/dimа
-basemodel/stream_0_conv_2/conv1d/ExpandDims_1
ExpandDimsDbasemodel/stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOp:value:0:basemodel/stream_0_conv_2/conv1d/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:@ђ2/
-basemodel/stream_0_conv_2/conv1d/ExpandDims_1Ъ
 basemodel/stream_0_conv_2/conv1dConv2D4basemodel/stream_0_conv_2/conv1d/ExpandDims:output:06basemodel/stream_0_conv_2/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:         }ђ*
paddingSAME*
strides
2"
 basemodel/stream_0_conv_2/conv1dр
(basemodel/stream_0_conv_2/conv1d/SqueezeSqueeze)basemodel/stream_0_conv_2/conv1d:output:0*
T0*,
_output_shapes
:         }ђ*
squeeze_dims

§        2*
(basemodel/stream_0_conv_2/conv1d/Squeeze█
0basemodel/stream_0_conv_2/BiasAdd/ReadVariableOpReadVariableOp9basemodel_stream_0_conv_2_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype022
0basemodel/stream_0_conv_2/BiasAdd/ReadVariableOpш
!basemodel/stream_0_conv_2/BiasAddBiasAdd1basemodel/stream_0_conv_2/conv1d/Squeeze:output:08basemodel/stream_0_conv_2/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:         }ђ2#
!basemodel/stream_0_conv_2/BiasAddЛ
>basemodel/batch_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2@
>basemodel/batch_normalization_1/moments/mean/reduction_indicesў
,basemodel/batch_normalization_1/moments/meanMean*basemodel/stream_0_conv_2/BiasAdd:output:0Gbasemodel/batch_normalization_1/moments/mean/reduction_indices:output:0*
T0*#
_output_shapes
:ђ*
	keep_dims(2.
,basemodel/batch_normalization_1/moments/meanр
4basemodel/batch_normalization_1/moments/StopGradientStopGradient5basemodel/batch_normalization_1/moments/mean:output:0*
T0*#
_output_shapes
:ђ26
4basemodel/batch_normalization_1/moments/StopGradientГ
9basemodel/batch_normalization_1/moments/SquaredDifferenceSquaredDifference*basemodel/stream_0_conv_2/BiasAdd:output:0=basemodel/batch_normalization_1/moments/StopGradient:output:0*
T0*,
_output_shapes
:         }ђ2;
9basemodel/batch_normalization_1/moments/SquaredDifference┘
Bbasemodel/batch_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2D
Bbasemodel/batch_normalization_1/moments/variance/reduction_indicesи
0basemodel/batch_normalization_1/moments/varianceMean=basemodel/batch_normalization_1/moments/SquaredDifference:z:0Kbasemodel/batch_normalization_1/moments/variance/reduction_indices:output:0*
T0*#
_output_shapes
:ђ*
	keep_dims(22
0basemodel/batch_normalization_1/moments/varianceР
/basemodel/batch_normalization_1/moments/SqueezeSqueeze5basemodel/batch_normalization_1/moments/mean:output:0*
T0*
_output_shapes	
:ђ*
squeeze_dims
 21
/basemodel/batch_normalization_1/moments/SqueezeЖ
1basemodel/batch_normalization_1/moments/Squeeze_1Squeeze9basemodel/batch_normalization_1/moments/variance:output:0*
T0*
_output_shapes	
:ђ*
squeeze_dims
 23
1basemodel/batch_normalization_1/moments/Squeeze_1│
5basemodel/batch_normalization_1/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<27
5basemodel/batch_normalization_1/AssignMovingAvg/decayЁ
>basemodel/batch_normalization_1/AssignMovingAvg/ReadVariableOpReadVariableOpGbasemodel_batch_normalization_1_assignmovingavg_readvariableop_resource*
_output_shapes	
:ђ*
dtype02@
>basemodel/batch_normalization_1/AssignMovingAvg/ReadVariableOpЎ
3basemodel/batch_normalization_1/AssignMovingAvg/subSubFbasemodel/batch_normalization_1/AssignMovingAvg/ReadVariableOp:value:08basemodel/batch_normalization_1/moments/Squeeze:output:0*
T0*
_output_shapes	
:ђ25
3basemodel/batch_normalization_1/AssignMovingAvg/subљ
3basemodel/batch_normalization_1/AssignMovingAvg/mulMul7basemodel/batch_normalization_1/AssignMovingAvg/sub:z:0>basemodel/batch_normalization_1/AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:ђ25
3basemodel/batch_normalization_1/AssignMovingAvg/mul▀
/basemodel/batch_normalization_1/AssignMovingAvgAssignSubVariableOpGbasemodel_batch_normalization_1_assignmovingavg_readvariableop_resource7basemodel/batch_normalization_1/AssignMovingAvg/mul:z:0?^basemodel/batch_normalization_1/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype021
/basemodel/batch_normalization_1/AssignMovingAvgи
7basemodel/batch_normalization_1/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<29
7basemodel/batch_normalization_1/AssignMovingAvg_1/decayІ
@basemodel/batch_normalization_1/AssignMovingAvg_1/ReadVariableOpReadVariableOpIbasemodel_batch_normalization_1_assignmovingavg_1_readvariableop_resource*
_output_shapes	
:ђ*
dtype02B
@basemodel/batch_normalization_1/AssignMovingAvg_1/ReadVariableOpА
5basemodel/batch_normalization_1/AssignMovingAvg_1/subSubHbasemodel/batch_normalization_1/AssignMovingAvg_1/ReadVariableOp:value:0:basemodel/batch_normalization_1/moments/Squeeze_1:output:0*
T0*
_output_shapes	
:ђ27
5basemodel/batch_normalization_1/AssignMovingAvg_1/subў
5basemodel/batch_normalization_1/AssignMovingAvg_1/mulMul9basemodel/batch_normalization_1/AssignMovingAvg_1/sub:z:0@basemodel/batch_normalization_1/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:ђ27
5basemodel/batch_normalization_1/AssignMovingAvg_1/mulж
1basemodel/batch_normalization_1/AssignMovingAvg_1AssignSubVariableOpIbasemodel_batch_normalization_1_assignmovingavg_1_readvariableop_resource9basemodel/batch_normalization_1/AssignMovingAvg_1/mul:z:0A^basemodel/batch_normalization_1/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype023
1basemodel/batch_normalization_1/AssignMovingAvg_1Д
/basemodel/batch_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:21
/basemodel/batch_normalization_1/batchnorm/add/yЃ
-basemodel/batch_normalization_1/batchnorm/addAddV2:basemodel/batch_normalization_1/moments/Squeeze_1:output:08basemodel/batch_normalization_1/batchnorm/add/y:output:0*
T0*
_output_shapes	
:ђ2/
-basemodel/batch_normalization_1/batchnorm/add─
/basemodel/batch_normalization_1/batchnorm/RsqrtRsqrt1basemodel/batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes	
:ђ21
/basemodel/batch_normalization_1/batchnorm/Rsqrt 
<basemodel/batch_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOpEbasemodel_batch_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes	
:ђ*
dtype02>
<basemodel/batch_normalization_1/batchnorm/mul/ReadVariableOpє
-basemodel/batch_normalization_1/batchnorm/mulMul3basemodel/batch_normalization_1/batchnorm/Rsqrt:y:0Dbasemodel/batch_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:ђ2/
-basemodel/batch_normalization_1/batchnorm/mul 
/basemodel/batch_normalization_1/batchnorm/mul_1Mul*basemodel/stream_0_conv_2/BiasAdd:output:01basemodel/batch_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:         }ђ21
/basemodel/batch_normalization_1/batchnorm/mul_1Ч
/basemodel/batch_normalization_1/batchnorm/mul_2Mul8basemodel/batch_normalization_1/moments/Squeeze:output:01basemodel/batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes	
:ђ21
/basemodel/batch_normalization_1/batchnorm/mul_2з
8basemodel/batch_normalization_1/batchnorm/ReadVariableOpReadVariableOpAbasemodel_batch_normalization_1_batchnorm_readvariableop_resource*
_output_shapes	
:ђ*
dtype02:
8basemodel/batch_normalization_1/batchnorm/ReadVariableOpѓ
-basemodel/batch_normalization_1/batchnorm/subSub@basemodel/batch_normalization_1/batchnorm/ReadVariableOp:value:03basemodel/batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:ђ2/
-basemodel/batch_normalization_1/batchnorm/subі
/basemodel/batch_normalization_1/batchnorm/add_1AddV23basemodel/batch_normalization_1/batchnorm/mul_1:z:01basemodel/batch_normalization_1/batchnorm/sub:z:0*
T0*,
_output_shapes
:         }ђ21
/basemodel/batch_normalization_1/batchnorm/add_1«
basemodel/activation_1/ReluRelu3basemodel/batch_normalization_1/batchnorm/add_1:z:0*
T0*,
_output_shapes
:         }ђ2
basemodel/activation_1/ReluЌ
'basemodel/stream_0_drop_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUН?2)
'basemodel/stream_0_drop_2/dropout/Constж
%basemodel/stream_0_drop_2/dropout/MulMul)basemodel/activation_1/Relu:activations:00basemodel/stream_0_drop_2/dropout/Const:output:0*
T0*,
_output_shapes
:         }ђ2'
%basemodel/stream_0_drop_2/dropout/MulФ
'basemodel/stream_0_drop_2/dropout/ShapeShape)basemodel/activation_1/Relu:activations:0*
T0*
_output_shapes
:2)
'basemodel/stream_0_drop_2/dropout/Shapeб
>basemodel/stream_0_drop_2/dropout/random_uniform/RandomUniformRandomUniform0basemodel/stream_0_drop_2/dropout/Shape:output:0*
T0*,
_output_shapes
:         }ђ*
dtype0*
seedи*
seed2и2@
>basemodel/stream_0_drop_2/dropout/random_uniform/RandomUniformЕ
0basemodel/stream_0_drop_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠>22
0basemodel/stream_0_drop_2/dropout/GreaterEqual/yФ
.basemodel/stream_0_drop_2/dropout/GreaterEqualGreaterEqualGbasemodel/stream_0_drop_2/dropout/random_uniform/RandomUniform:output:09basemodel/stream_0_drop_2/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:         }ђ20
.basemodel/stream_0_drop_2/dropout/GreaterEqualм
&basemodel/stream_0_drop_2/dropout/CastCast2basemodel/stream_0_drop_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:         }ђ2(
&basemodel/stream_0_drop_2/dropout/Castу
'basemodel/stream_0_drop_2/dropout/Mul_1Mul)basemodel/stream_0_drop_2/dropout/Mul:z:0*basemodel/stream_0_drop_2/dropout/Cast:y:0*
T0*,
_output_shapes
:         }ђ2)
'basemodel/stream_0_drop_2/dropout/Mul_1Г
/basemodel/stream_0_conv_3/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§        21
/basemodel/stream_0_conv_3/conv1d/ExpandDims/dimі
+basemodel/stream_0_conv_3/conv1d/ExpandDims
ExpandDims+basemodel/stream_0_drop_2/dropout/Mul_1:z:08basemodel/stream_0_conv_3/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         }ђ2-
+basemodel/stream_0_conv_3/conv1d/ExpandDimsѕ
<basemodel/stream_0_conv_3/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpEbasemodel_stream_0_conv_3_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:ђђ*
dtype02>
<basemodel/stream_0_conv_3/conv1d/ExpandDims_1/ReadVariableOpе
1basemodel/stream_0_conv_3/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 23
1basemodel/stream_0_conv_3/conv1d/ExpandDims_1/dimА
-basemodel/stream_0_conv_3/conv1d/ExpandDims_1
ExpandDimsDbasemodel/stream_0_conv_3/conv1d/ExpandDims_1/ReadVariableOp:value:0:basemodel/stream_0_conv_3/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:ђђ2/
-basemodel/stream_0_conv_3/conv1d/ExpandDims_1Ъ
 basemodel/stream_0_conv_3/conv1dConv2D4basemodel/stream_0_conv_3/conv1d/ExpandDims:output:06basemodel/stream_0_conv_3/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:         }ђ*
paddingSAME*
strides
2"
 basemodel/stream_0_conv_3/conv1dр
(basemodel/stream_0_conv_3/conv1d/SqueezeSqueeze)basemodel/stream_0_conv_3/conv1d:output:0*
T0*,
_output_shapes
:         }ђ*
squeeze_dims

§        2*
(basemodel/stream_0_conv_3/conv1d/Squeeze█
0basemodel/stream_0_conv_3/BiasAdd/ReadVariableOpReadVariableOp9basemodel_stream_0_conv_3_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype022
0basemodel/stream_0_conv_3/BiasAdd/ReadVariableOpш
!basemodel/stream_0_conv_3/BiasAddBiasAdd1basemodel/stream_0_conv_3/conv1d/Squeeze:output:08basemodel/stream_0_conv_3/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:         }ђ2#
!basemodel/stream_0_conv_3/BiasAddЛ
>basemodel/batch_normalization_2/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2@
>basemodel/batch_normalization_2/moments/mean/reduction_indicesў
,basemodel/batch_normalization_2/moments/meanMean*basemodel/stream_0_conv_3/BiasAdd:output:0Gbasemodel/batch_normalization_2/moments/mean/reduction_indices:output:0*
T0*#
_output_shapes
:ђ*
	keep_dims(2.
,basemodel/batch_normalization_2/moments/meanр
4basemodel/batch_normalization_2/moments/StopGradientStopGradient5basemodel/batch_normalization_2/moments/mean:output:0*
T0*#
_output_shapes
:ђ26
4basemodel/batch_normalization_2/moments/StopGradientГ
9basemodel/batch_normalization_2/moments/SquaredDifferenceSquaredDifference*basemodel/stream_0_conv_3/BiasAdd:output:0=basemodel/batch_normalization_2/moments/StopGradient:output:0*
T0*,
_output_shapes
:         }ђ2;
9basemodel/batch_normalization_2/moments/SquaredDifference┘
Bbasemodel/batch_normalization_2/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2D
Bbasemodel/batch_normalization_2/moments/variance/reduction_indicesи
0basemodel/batch_normalization_2/moments/varianceMean=basemodel/batch_normalization_2/moments/SquaredDifference:z:0Kbasemodel/batch_normalization_2/moments/variance/reduction_indices:output:0*
T0*#
_output_shapes
:ђ*
	keep_dims(22
0basemodel/batch_normalization_2/moments/varianceР
/basemodel/batch_normalization_2/moments/SqueezeSqueeze5basemodel/batch_normalization_2/moments/mean:output:0*
T0*
_output_shapes	
:ђ*
squeeze_dims
 21
/basemodel/batch_normalization_2/moments/SqueezeЖ
1basemodel/batch_normalization_2/moments/Squeeze_1Squeeze9basemodel/batch_normalization_2/moments/variance:output:0*
T0*
_output_shapes	
:ђ*
squeeze_dims
 23
1basemodel/batch_normalization_2/moments/Squeeze_1│
5basemodel/batch_normalization_2/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<27
5basemodel/batch_normalization_2/AssignMovingAvg/decayЁ
>basemodel/batch_normalization_2/AssignMovingAvg/ReadVariableOpReadVariableOpGbasemodel_batch_normalization_2_assignmovingavg_readvariableop_resource*
_output_shapes	
:ђ*
dtype02@
>basemodel/batch_normalization_2/AssignMovingAvg/ReadVariableOpЎ
3basemodel/batch_normalization_2/AssignMovingAvg/subSubFbasemodel/batch_normalization_2/AssignMovingAvg/ReadVariableOp:value:08basemodel/batch_normalization_2/moments/Squeeze:output:0*
T0*
_output_shapes	
:ђ25
3basemodel/batch_normalization_2/AssignMovingAvg/subљ
3basemodel/batch_normalization_2/AssignMovingAvg/mulMul7basemodel/batch_normalization_2/AssignMovingAvg/sub:z:0>basemodel/batch_normalization_2/AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:ђ25
3basemodel/batch_normalization_2/AssignMovingAvg/mul▀
/basemodel/batch_normalization_2/AssignMovingAvgAssignSubVariableOpGbasemodel_batch_normalization_2_assignmovingavg_readvariableop_resource7basemodel/batch_normalization_2/AssignMovingAvg/mul:z:0?^basemodel/batch_normalization_2/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype021
/basemodel/batch_normalization_2/AssignMovingAvgи
7basemodel/batch_normalization_2/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<29
7basemodel/batch_normalization_2/AssignMovingAvg_1/decayІ
@basemodel/batch_normalization_2/AssignMovingAvg_1/ReadVariableOpReadVariableOpIbasemodel_batch_normalization_2_assignmovingavg_1_readvariableop_resource*
_output_shapes	
:ђ*
dtype02B
@basemodel/batch_normalization_2/AssignMovingAvg_1/ReadVariableOpА
5basemodel/batch_normalization_2/AssignMovingAvg_1/subSubHbasemodel/batch_normalization_2/AssignMovingAvg_1/ReadVariableOp:value:0:basemodel/batch_normalization_2/moments/Squeeze_1:output:0*
T0*
_output_shapes	
:ђ27
5basemodel/batch_normalization_2/AssignMovingAvg_1/subў
5basemodel/batch_normalization_2/AssignMovingAvg_1/mulMul9basemodel/batch_normalization_2/AssignMovingAvg_1/sub:z:0@basemodel/batch_normalization_2/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:ђ27
5basemodel/batch_normalization_2/AssignMovingAvg_1/mulж
1basemodel/batch_normalization_2/AssignMovingAvg_1AssignSubVariableOpIbasemodel_batch_normalization_2_assignmovingavg_1_readvariableop_resource9basemodel/batch_normalization_2/AssignMovingAvg_1/mul:z:0A^basemodel/batch_normalization_2/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype023
1basemodel/batch_normalization_2/AssignMovingAvg_1Д
/basemodel/batch_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:21
/basemodel/batch_normalization_2/batchnorm/add/yЃ
-basemodel/batch_normalization_2/batchnorm/addAddV2:basemodel/batch_normalization_2/moments/Squeeze_1:output:08basemodel/batch_normalization_2/batchnorm/add/y:output:0*
T0*
_output_shapes	
:ђ2/
-basemodel/batch_normalization_2/batchnorm/add─
/basemodel/batch_normalization_2/batchnorm/RsqrtRsqrt1basemodel/batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes	
:ђ21
/basemodel/batch_normalization_2/batchnorm/Rsqrt 
<basemodel/batch_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOpEbasemodel_batch_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes	
:ђ*
dtype02>
<basemodel/batch_normalization_2/batchnorm/mul/ReadVariableOpє
-basemodel/batch_normalization_2/batchnorm/mulMul3basemodel/batch_normalization_2/batchnorm/Rsqrt:y:0Dbasemodel/batch_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:ђ2/
-basemodel/batch_normalization_2/batchnorm/mul 
/basemodel/batch_normalization_2/batchnorm/mul_1Mul*basemodel/stream_0_conv_3/BiasAdd:output:01basemodel/batch_normalization_2/batchnorm/mul:z:0*
T0*,
_output_shapes
:         }ђ21
/basemodel/batch_normalization_2/batchnorm/mul_1Ч
/basemodel/batch_normalization_2/batchnorm/mul_2Mul8basemodel/batch_normalization_2/moments/Squeeze:output:01basemodel/batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes	
:ђ21
/basemodel/batch_normalization_2/batchnorm/mul_2з
8basemodel/batch_normalization_2/batchnorm/ReadVariableOpReadVariableOpAbasemodel_batch_normalization_2_batchnorm_readvariableop_resource*
_output_shapes	
:ђ*
dtype02:
8basemodel/batch_normalization_2/batchnorm/ReadVariableOpѓ
-basemodel/batch_normalization_2/batchnorm/subSub@basemodel/batch_normalization_2/batchnorm/ReadVariableOp:value:03basemodel/batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:ђ2/
-basemodel/batch_normalization_2/batchnorm/subі
/basemodel/batch_normalization_2/batchnorm/add_1AddV23basemodel/batch_normalization_2/batchnorm/mul_1:z:01basemodel/batch_normalization_2/batchnorm/sub:z:0*
T0*,
_output_shapes
:         }ђ21
/basemodel/batch_normalization_2/batchnorm/add_1«
basemodel/activation_2/ReluRelu3basemodel/batch_normalization_2/batchnorm/add_1:z:0*
T0*,
_output_shapes
:         }ђ2
basemodel/activation_2/ReluЌ
'basemodel/stream_0_drop_3/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2)
'basemodel/stream_0_drop_3/dropout/Constж
%basemodel/stream_0_drop_3/dropout/MulMul)basemodel/activation_2/Relu:activations:00basemodel/stream_0_drop_3/dropout/Const:output:0*
T0*,
_output_shapes
:         }ђ2'
%basemodel/stream_0_drop_3/dropout/MulФ
'basemodel/stream_0_drop_3/dropout/ShapeShape)basemodel/activation_2/Relu:activations:0*
T0*
_output_shapes
:2)
'basemodel/stream_0_drop_3/dropout/Shapeб
>basemodel/stream_0_drop_3/dropout/random_uniform/RandomUniformRandomUniform0basemodel/stream_0_drop_3/dropout/Shape:output:0*
T0*,
_output_shapes
:         }ђ*
dtype0*
seedи*
seed2и2@
>basemodel/stream_0_drop_3/dropout/random_uniform/RandomUniformЕ
0basemodel/stream_0_drop_3/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?22
0basemodel/stream_0_drop_3/dropout/GreaterEqual/yФ
.basemodel/stream_0_drop_3/dropout/GreaterEqualGreaterEqualGbasemodel/stream_0_drop_3/dropout/random_uniform/RandomUniform:output:09basemodel/stream_0_drop_3/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:         }ђ20
.basemodel/stream_0_drop_3/dropout/GreaterEqualм
&basemodel/stream_0_drop_3/dropout/CastCast2basemodel/stream_0_drop_3/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:         }ђ2(
&basemodel/stream_0_drop_3/dropout/Castу
'basemodel/stream_0_drop_3/dropout/Mul_1Mul)basemodel/stream_0_drop_3/dropout/Mul:z:0*basemodel/stream_0_drop_3/dropout/Cast:y:0*
T0*,
_output_shapes
:         }ђ2)
'basemodel/stream_0_drop_3/dropout/Mul_1И
9basemodel/global_average_pooling1d/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2;
9basemodel/global_average_pooling1d/Mean/reduction_indices■
'basemodel/global_average_pooling1d/MeanMean+basemodel/stream_0_drop_3/dropout/Mul_1:z:0Bbasemodel/global_average_pooling1d/Mean/reduction_indices:output:0*
T0*(
_output_shapes
:         ђ2)
'basemodel/global_average_pooling1d/MeanЌ
'basemodel/dense_1_dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?2)
'basemodel/dense_1_dropout/dropout/ConstВ
%basemodel/dense_1_dropout/dropout/MulMul0basemodel/global_average_pooling1d/Mean:output:00basemodel/dense_1_dropout/dropout/Const:output:0*
T0*(
_output_shapes
:         ђ2'
%basemodel/dense_1_dropout/dropout/Mul▓
'basemodel/dense_1_dropout/dropout/ShapeShape0basemodel/global_average_pooling1d/Mean:output:0*
T0*
_output_shapes
:2)
'basemodel/dense_1_dropout/dropout/Shapeљ
>basemodel/dense_1_dropout/dropout/random_uniform/RandomUniformRandomUniform0basemodel/dense_1_dropout/dropout/Shape:output:0*
T0*(
_output_shapes
:         ђ*
dtype0*
seedи2@
>basemodel/dense_1_dropout/dropout/random_uniform/RandomUniformЕ
0basemodel/dense_1_dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>22
0basemodel/dense_1_dropout/dropout/GreaterEqual/yД
.basemodel/dense_1_dropout/dropout/GreaterEqualGreaterEqualGbasemodel/dense_1_dropout/dropout/random_uniform/RandomUniform:output:09basemodel/dense_1_dropout/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         ђ20
.basemodel/dense_1_dropout/dropout/GreaterEqual╬
&basemodel/dense_1_dropout/dropout/CastCast2basemodel/dense_1_dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         ђ2(
&basemodel/dense_1_dropout/dropout/Castс
'basemodel/dense_1_dropout/dropout/Mul_1Mul)basemodel/dense_1_dropout/dropout/Mul:z:0*basemodel/dense_1_dropout/dropout/Cast:y:0*
T0*(
_output_shapes
:         ђ2)
'basemodel/dense_1_dropout/dropout/Mul_1─
'basemodel/dense_1/MatMul/ReadVariableOpReadVariableOp0basemodel_dense_1_matmul_readvariableop_resource*
_output_shapes
:	ђT*
dtype02)
'basemodel/dense_1/MatMul/ReadVariableOp╬
basemodel/dense_1/MatMulMatMul+basemodel/dense_1_dropout/dropout/Mul_1:z:0/basemodel/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         T2
basemodel/dense_1/MatMul┬
(basemodel/dense_1/BiasAdd/ReadVariableOpReadVariableOp1basemodel_dense_1_biasadd_readvariableop_resource*
_output_shapes
:T*
dtype02*
(basemodel/dense_1/BiasAdd/ReadVariableOp╔
basemodel/dense_1/BiasAddBiasAdd"basemodel/dense_1/MatMul:product:00basemodel/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         T2
basemodel/dense_1/BiasAdd╩
>basemodel/batch_normalization_3/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2@
>basemodel/batch_normalization_3/moments/mean/reduction_indicesІ
,basemodel/batch_normalization_3/moments/meanMean"basemodel/dense_1/BiasAdd:output:0Gbasemodel/batch_normalization_3/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:T*
	keep_dims(2.
,basemodel/batch_normalization_3/moments/mean▄
4basemodel/batch_normalization_3/moments/StopGradientStopGradient5basemodel/batch_normalization_3/moments/mean:output:0*
T0*
_output_shapes

:T26
4basemodel/batch_normalization_3/moments/StopGradientа
9basemodel/batch_normalization_3/moments/SquaredDifferenceSquaredDifference"basemodel/dense_1/BiasAdd:output:0=basemodel/batch_normalization_3/moments/StopGradient:output:0*
T0*'
_output_shapes
:         T2;
9basemodel/batch_normalization_3/moments/SquaredDifferenceм
Bbasemodel/batch_normalization_3/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2D
Bbasemodel/batch_normalization_3/moments/variance/reduction_indices▓
0basemodel/batch_normalization_3/moments/varianceMean=basemodel/batch_normalization_3/moments/SquaredDifference:z:0Kbasemodel/batch_normalization_3/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:T*
	keep_dims(22
0basemodel/batch_normalization_3/moments/varianceЯ
/basemodel/batch_normalization_3/moments/SqueezeSqueeze5basemodel/batch_normalization_3/moments/mean:output:0*
T0*
_output_shapes
:T*
squeeze_dims
 21
/basemodel/batch_normalization_3/moments/SqueezeУ
1basemodel/batch_normalization_3/moments/Squeeze_1Squeeze9basemodel/batch_normalization_3/moments/variance:output:0*
T0*
_output_shapes
:T*
squeeze_dims
 23
1basemodel/batch_normalization_3/moments/Squeeze_1│
5basemodel/batch_normalization_3/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<27
5basemodel/batch_normalization_3/AssignMovingAvg/decayё
>basemodel/batch_normalization_3/AssignMovingAvg/ReadVariableOpReadVariableOpGbasemodel_batch_normalization_3_assignmovingavg_readvariableop_resource*
_output_shapes
:T*
dtype02@
>basemodel/batch_normalization_3/AssignMovingAvg/ReadVariableOpў
3basemodel/batch_normalization_3/AssignMovingAvg/subSubFbasemodel/batch_normalization_3/AssignMovingAvg/ReadVariableOp:value:08basemodel/batch_normalization_3/moments/Squeeze:output:0*
T0*
_output_shapes
:T25
3basemodel/batch_normalization_3/AssignMovingAvg/subЈ
3basemodel/batch_normalization_3/AssignMovingAvg/mulMul7basemodel/batch_normalization_3/AssignMovingAvg/sub:z:0>basemodel/batch_normalization_3/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:T25
3basemodel/batch_normalization_3/AssignMovingAvg/mul▀
/basemodel/batch_normalization_3/AssignMovingAvgAssignSubVariableOpGbasemodel_batch_normalization_3_assignmovingavg_readvariableop_resource7basemodel/batch_normalization_3/AssignMovingAvg/mul:z:0?^basemodel/batch_normalization_3/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype021
/basemodel/batch_normalization_3/AssignMovingAvgи
7basemodel/batch_normalization_3/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<29
7basemodel/batch_normalization_3/AssignMovingAvg_1/decayі
@basemodel/batch_normalization_3/AssignMovingAvg_1/ReadVariableOpReadVariableOpIbasemodel_batch_normalization_3_assignmovingavg_1_readvariableop_resource*
_output_shapes
:T*
dtype02B
@basemodel/batch_normalization_3/AssignMovingAvg_1/ReadVariableOpа
5basemodel/batch_normalization_3/AssignMovingAvg_1/subSubHbasemodel/batch_normalization_3/AssignMovingAvg_1/ReadVariableOp:value:0:basemodel/batch_normalization_3/moments/Squeeze_1:output:0*
T0*
_output_shapes
:T27
5basemodel/batch_normalization_3/AssignMovingAvg_1/subЌ
5basemodel/batch_normalization_3/AssignMovingAvg_1/mulMul9basemodel/batch_normalization_3/AssignMovingAvg_1/sub:z:0@basemodel/batch_normalization_3/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:T27
5basemodel/batch_normalization_3/AssignMovingAvg_1/mulж
1basemodel/batch_normalization_3/AssignMovingAvg_1AssignSubVariableOpIbasemodel_batch_normalization_3_assignmovingavg_1_readvariableop_resource9basemodel/batch_normalization_3/AssignMovingAvg_1/mul:z:0A^basemodel/batch_normalization_3/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype023
1basemodel/batch_normalization_3/AssignMovingAvg_1Д
/basemodel/batch_normalization_3/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:21
/basemodel/batch_normalization_3/batchnorm/add/yѓ
-basemodel/batch_normalization_3/batchnorm/addAddV2:basemodel/batch_normalization_3/moments/Squeeze_1:output:08basemodel/batch_normalization_3/batchnorm/add/y:output:0*
T0*
_output_shapes
:T2/
-basemodel/batch_normalization_3/batchnorm/add├
/basemodel/batch_normalization_3/batchnorm/RsqrtRsqrt1basemodel/batch_normalization_3/batchnorm/add:z:0*
T0*
_output_shapes
:T21
/basemodel/batch_normalization_3/batchnorm/Rsqrt■
<basemodel/batch_normalization_3/batchnorm/mul/ReadVariableOpReadVariableOpEbasemodel_batch_normalization_3_batchnorm_mul_readvariableop_resource*
_output_shapes
:T*
dtype02>
<basemodel/batch_normalization_3/batchnorm/mul/ReadVariableOpЁ
-basemodel/batch_normalization_3/batchnorm/mulMul3basemodel/batch_normalization_3/batchnorm/Rsqrt:y:0Dbasemodel/batch_normalization_3/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:T2/
-basemodel/batch_normalization_3/batchnorm/mulЫ
/basemodel/batch_normalization_3/batchnorm/mul_1Mul"basemodel/dense_1/BiasAdd:output:01basemodel/batch_normalization_3/batchnorm/mul:z:0*
T0*'
_output_shapes
:         T21
/basemodel/batch_normalization_3/batchnorm/mul_1ч
/basemodel/batch_normalization_3/batchnorm/mul_2Mul8basemodel/batch_normalization_3/moments/Squeeze:output:01basemodel/batch_normalization_3/batchnorm/mul:z:0*
T0*
_output_shapes
:T21
/basemodel/batch_normalization_3/batchnorm/mul_2Ы
8basemodel/batch_normalization_3/batchnorm/ReadVariableOpReadVariableOpAbasemodel_batch_normalization_3_batchnorm_readvariableop_resource*
_output_shapes
:T*
dtype02:
8basemodel/batch_normalization_3/batchnorm/ReadVariableOpЂ
-basemodel/batch_normalization_3/batchnorm/subSub@basemodel/batch_normalization_3/batchnorm/ReadVariableOp:value:03basemodel/batch_normalization_3/batchnorm/mul_2:z:0*
T0*
_output_shapes
:T2/
-basemodel/batch_normalization_3/batchnorm/subЁ
/basemodel/batch_normalization_3/batchnorm/add_1AddV23basemodel/batch_normalization_3/batchnorm/mul_1:z:01basemodel/batch_normalization_3/batchnorm/sub:z:0*
T0*'
_output_shapes
:         T21
/basemodel/batch_normalization_3/batchnorm/add_1Й
$basemodel/dense_activation_1/SigmoidSigmoid3basemodel/batch_normalization_3/batchnorm/add_1:z:0*
T0*'
_output_shapes
:         T2&
$basemodel/dense_activation_1/SigmoidБ
-basemodel/stream_0_input_drop/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?2/
-basemodel/stream_0_input_drop/dropout_1/Const┘
+basemodel/stream_0_input_drop/dropout_1/MulMulinputs_16basemodel/stream_0_input_drop/dropout_1/Const:output:0*
T0*+
_output_shapes
:         }2-
+basemodel/stream_0_input_drop/dropout_1/Mulќ
-basemodel/stream_0_input_drop/dropout_1/ShapeShapeinputs_1*
T0*
_output_shapes
:2/
-basemodel/stream_0_input_drop/dropout_1/Shape│
Dbasemodel/stream_0_input_drop/dropout_1/random_uniform/RandomUniformRandomUniform6basemodel/stream_0_input_drop/dropout_1/Shape:output:0*
T0*+
_output_shapes
:         }*
dtype0*
seedи*
seed2и2F
Dbasemodel/stream_0_input_drop/dropout_1/random_uniform/RandomUniformх
6basemodel/stream_0_input_drop/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>28
6basemodel/stream_0_input_drop/dropout_1/GreaterEqual/y┬
4basemodel/stream_0_input_drop/dropout_1/GreaterEqualGreaterEqualMbasemodel/stream_0_input_drop/dropout_1/random_uniform/RandomUniform:output:0?basemodel/stream_0_input_drop/dropout_1/GreaterEqual/y:output:0*
T0*+
_output_shapes
:         }26
4basemodel/stream_0_input_drop/dropout_1/GreaterEqualс
,basemodel/stream_0_input_drop/dropout_1/CastCast8basemodel/stream_0_input_drop/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:         }2.
,basemodel/stream_0_input_drop/dropout_1/Cast■
-basemodel/stream_0_input_drop/dropout_1/Mul_1Mul/basemodel/stream_0_input_drop/dropout_1/Mul:z:00basemodel/stream_0_input_drop/dropout_1/Cast:y:0*
T0*+
_output_shapes
:         }2/
-basemodel/stream_0_input_drop/dropout_1/Mul_1▒
1basemodel/stream_0_conv_1/conv1d_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§        23
1basemodel/stream_0_conv_1/conv1d_1/ExpandDims/dimЋ
-basemodel/stream_0_conv_1/conv1d_1/ExpandDims
ExpandDims1basemodel/stream_0_input_drop/dropout_1/Mul_1:z:0:basemodel/stream_0_conv_1/conv1d_1/ExpandDims/dim:output:0*
T0*/
_output_shapes
:         }2/
-basemodel/stream_0_conv_1/conv1d_1/ExpandDimsі
>basemodel/stream_0_conv_1/conv1d_1/ExpandDims_1/ReadVariableOpReadVariableOpEbasemodel_stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02@
>basemodel/stream_0_conv_1/conv1d_1/ExpandDims_1/ReadVariableOpг
3basemodel/stream_0_conv_1/conv1d_1/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 25
3basemodel/stream_0_conv_1/conv1d_1/ExpandDims_1/dimД
/basemodel/stream_0_conv_1/conv1d_1/ExpandDims_1
ExpandDimsFbasemodel/stream_0_conv_1/conv1d_1/ExpandDims_1/ReadVariableOp:value:0<basemodel/stream_0_conv_1/conv1d_1/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@21
/basemodel/stream_0_conv_1/conv1d_1/ExpandDims_1д
"basemodel/stream_0_conv_1/conv1d_1Conv2D6basemodel/stream_0_conv_1/conv1d_1/ExpandDims:output:08basemodel/stream_0_conv_1/conv1d_1/ExpandDims_1:output:0*
T0*/
_output_shapes
:         }@*
paddingSAME*
strides
2$
"basemodel/stream_0_conv_1/conv1d_1Т
*basemodel/stream_0_conv_1/conv1d_1/SqueezeSqueeze+basemodel/stream_0_conv_1/conv1d_1:output:0*
T0*+
_output_shapes
:         }@*
squeeze_dims

§        2,
*basemodel/stream_0_conv_1/conv1d_1/Squeezeя
2basemodel/stream_0_conv_1/BiasAdd_1/ReadVariableOpReadVariableOp9basemodel_stream_0_conv_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype024
2basemodel/stream_0_conv_1/BiasAdd_1/ReadVariableOpЧ
#basemodel/stream_0_conv_1/BiasAdd_1BiasAdd3basemodel/stream_0_conv_1/conv1d_1/Squeeze:output:0:basemodel/stream_0_conv_1/BiasAdd_1/ReadVariableOp:value:0*
T0*+
_output_shapes
:         }@2%
#basemodel/stream_0_conv_1/BiasAdd_1Л
>basemodel/batch_normalization/moments_1/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2@
>basemodel/batch_normalization/moments_1/mean/reduction_indicesЎ
,basemodel/batch_normalization/moments_1/meanMean,basemodel/stream_0_conv_1/BiasAdd_1:output:0Gbasemodel/batch_normalization/moments_1/mean/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(2.
,basemodel/batch_normalization/moments_1/meanЯ
4basemodel/batch_normalization/moments_1/StopGradientStopGradient5basemodel/batch_normalization/moments_1/mean:output:0*
T0*"
_output_shapes
:@26
4basemodel/batch_normalization/moments_1/StopGradient«
9basemodel/batch_normalization/moments_1/SquaredDifferenceSquaredDifference,basemodel/stream_0_conv_1/BiasAdd_1:output:0=basemodel/batch_normalization/moments_1/StopGradient:output:0*
T0*+
_output_shapes
:         }@2;
9basemodel/batch_normalization/moments_1/SquaredDifference┘
Bbasemodel/batch_normalization/moments_1/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2D
Bbasemodel/batch_normalization/moments_1/variance/reduction_indicesХ
0basemodel/batch_normalization/moments_1/varianceMean=basemodel/batch_normalization/moments_1/SquaredDifference:z:0Kbasemodel/batch_normalization/moments_1/variance/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(22
0basemodel/batch_normalization/moments_1/varianceр
/basemodel/batch_normalization/moments_1/SqueezeSqueeze5basemodel/batch_normalization/moments_1/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 21
/basemodel/batch_normalization/moments_1/Squeezeж
1basemodel/batch_normalization/moments_1/Squeeze_1Squeeze9basemodel/batch_normalization/moments_1/variance:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 23
1basemodel/batch_normalization/moments_1/Squeeze_1│
5basemodel/batch_normalization/AssignMovingAvg_2/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<27
5basemodel/batch_normalization/AssignMovingAvg_2/decay▓
>basemodel/batch_normalization/AssignMovingAvg_2/ReadVariableOpReadVariableOpEbasemodel_batch_normalization_assignmovingavg_readvariableop_resource.^basemodel/batch_normalization/AssignMovingAvg*
_output_shapes
:@*
dtype02@
>basemodel/batch_normalization/AssignMovingAvg_2/ReadVariableOpў
3basemodel/batch_normalization/AssignMovingAvg_2/subSubFbasemodel/batch_normalization/AssignMovingAvg_2/ReadVariableOp:value:08basemodel/batch_normalization/moments_1/Squeeze:output:0*
T0*
_output_shapes
:@25
3basemodel/batch_normalization/AssignMovingAvg_2/subЈ
3basemodel/batch_normalization/AssignMovingAvg_2/mulMul7basemodel/batch_normalization/AssignMovingAvg_2/sub:z:0>basemodel/batch_normalization/AssignMovingAvg_2/decay:output:0*
T0*
_output_shapes
:@25
3basemodel/batch_normalization/AssignMovingAvg_2/mulЇ
/basemodel/batch_normalization/AssignMovingAvg_2AssignSubVariableOpEbasemodel_batch_normalization_assignmovingavg_readvariableop_resource7basemodel/batch_normalization/AssignMovingAvg_2/mul:z:0.^basemodel/batch_normalization/AssignMovingAvg?^basemodel/batch_normalization/AssignMovingAvg_2/ReadVariableOp*
_output_shapes
 *
dtype021
/basemodel/batch_normalization/AssignMovingAvg_2│
5basemodel/batch_normalization/AssignMovingAvg_3/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<27
5basemodel/batch_normalization/AssignMovingAvg_3/decayХ
>basemodel/batch_normalization/AssignMovingAvg_3/ReadVariableOpReadVariableOpGbasemodel_batch_normalization_assignmovingavg_1_readvariableop_resource0^basemodel/batch_normalization/AssignMovingAvg_1*
_output_shapes
:@*
dtype02@
>basemodel/batch_normalization/AssignMovingAvg_3/ReadVariableOpџ
3basemodel/batch_normalization/AssignMovingAvg_3/subSubFbasemodel/batch_normalization/AssignMovingAvg_3/ReadVariableOp:value:0:basemodel/batch_normalization/moments_1/Squeeze_1:output:0*
T0*
_output_shapes
:@25
3basemodel/batch_normalization/AssignMovingAvg_3/subЈ
3basemodel/batch_normalization/AssignMovingAvg_3/mulMul7basemodel/batch_normalization/AssignMovingAvg_3/sub:z:0>basemodel/batch_normalization/AssignMovingAvg_3/decay:output:0*
T0*
_output_shapes
:@25
3basemodel/batch_normalization/AssignMovingAvg_3/mulЉ
/basemodel/batch_normalization/AssignMovingAvg_3AssignSubVariableOpGbasemodel_batch_normalization_assignmovingavg_1_readvariableop_resource7basemodel/batch_normalization/AssignMovingAvg_3/mul:z:00^basemodel/batch_normalization/AssignMovingAvg_1?^basemodel/batch_normalization/AssignMovingAvg_3/ReadVariableOp*
_output_shapes
 *
dtype021
/basemodel/batch_normalization/AssignMovingAvg_3Д
/basemodel/batch_normalization/batchnorm_1/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:21
/basemodel/batch_normalization/batchnorm_1/add/yѓ
-basemodel/batch_normalization/batchnorm_1/addAddV2:basemodel/batch_normalization/moments_1/Squeeze_1:output:08basemodel/batch_normalization/batchnorm_1/add/y:output:0*
T0*
_output_shapes
:@2/
-basemodel/batch_normalization/batchnorm_1/add├
/basemodel/batch_normalization/batchnorm_1/RsqrtRsqrt1basemodel/batch_normalization/batchnorm_1/add:z:0*
T0*
_output_shapes
:@21
/basemodel/batch_normalization/batchnorm_1/RsqrtЧ
<basemodel/batch_normalization/batchnorm_1/mul/ReadVariableOpReadVariableOpCbasemodel_batch_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02>
<basemodel/batch_normalization/batchnorm_1/mul/ReadVariableOpЁ
-basemodel/batch_normalization/batchnorm_1/mulMul3basemodel/batch_normalization/batchnorm_1/Rsqrt:y:0Dbasemodel/batch_normalization/batchnorm_1/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2/
-basemodel/batch_normalization/batchnorm_1/mulђ
/basemodel/batch_normalization/batchnorm_1/mul_1Mul,basemodel/stream_0_conv_1/BiasAdd_1:output:01basemodel/batch_normalization/batchnorm_1/mul:z:0*
T0*+
_output_shapes
:         }@21
/basemodel/batch_normalization/batchnorm_1/mul_1ч
/basemodel/batch_normalization/batchnorm_1/mul_2Mul8basemodel/batch_normalization/moments_1/Squeeze:output:01basemodel/batch_normalization/batchnorm_1/mul:z:0*
T0*
_output_shapes
:@21
/basemodel/batch_normalization/batchnorm_1/mul_2­
8basemodel/batch_normalization/batchnorm_1/ReadVariableOpReadVariableOp?basemodel_batch_normalization_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02:
8basemodel/batch_normalization/batchnorm_1/ReadVariableOpЂ
-basemodel/batch_normalization/batchnorm_1/subSub@basemodel/batch_normalization/batchnorm_1/ReadVariableOp:value:03basemodel/batch_normalization/batchnorm_1/mul_2:z:0*
T0*
_output_shapes
:@2/
-basemodel/batch_normalization/batchnorm_1/subЅ
/basemodel/batch_normalization/batchnorm_1/add_1AddV23basemodel/batch_normalization/batchnorm_1/mul_1:z:01basemodel/batch_normalization/batchnorm_1/sub:z:0*
T0*+
_output_shapes
:         }@21
/basemodel/batch_normalization/batchnorm_1/add_1Г
basemodel/activation/Relu_1Relu3basemodel/batch_normalization/batchnorm_1/add_1:z:0*
T0*+
_output_shapes
:         }@2
basemodel/activation/Relu_1Џ
)basemodel/stream_0_drop_1/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n█Х?2+
)basemodel/stream_0_drop_1/dropout_1/ConstЬ
'basemodel/stream_0_drop_1/dropout_1/MulMul)basemodel/activation/Relu_1:activations:02basemodel/stream_0_drop_1/dropout_1/Const:output:0*
T0*+
_output_shapes
:         }@2)
'basemodel/stream_0_drop_1/dropout_1/Mul»
)basemodel/stream_0_drop_1/dropout_1/ShapeShape)basemodel/activation/Relu_1:activations:0*
T0*
_output_shapes
:2+
)basemodel/stream_0_drop_1/dropout_1/ShapeД
@basemodel/stream_0_drop_1/dropout_1/random_uniform/RandomUniformRandomUniform2basemodel/stream_0_drop_1/dropout_1/Shape:output:0*
T0*+
_output_shapes
:         }@*
dtype0*
seedи*
seed2и2B
@basemodel/stream_0_drop_1/dropout_1/random_uniform/RandomUniformГ
2basemodel/stream_0_drop_1/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *џЎЎ>24
2basemodel/stream_0_drop_1/dropout_1/GreaterEqual/y▓
0basemodel/stream_0_drop_1/dropout_1/GreaterEqualGreaterEqualIbasemodel/stream_0_drop_1/dropout_1/random_uniform/RandomUniform:output:0;basemodel/stream_0_drop_1/dropout_1/GreaterEqual/y:output:0*
T0*+
_output_shapes
:         }@22
0basemodel/stream_0_drop_1/dropout_1/GreaterEqualО
(basemodel/stream_0_drop_1/dropout_1/CastCast4basemodel/stream_0_drop_1/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:         }@2*
(basemodel/stream_0_drop_1/dropout_1/CastЬ
)basemodel/stream_0_drop_1/dropout_1/Mul_1Mul+basemodel/stream_0_drop_1/dropout_1/Mul:z:0,basemodel/stream_0_drop_1/dropout_1/Cast:y:0*
T0*+
_output_shapes
:         }@2+
)basemodel/stream_0_drop_1/dropout_1/Mul_1▒
1basemodel/stream_0_conv_2/conv1d_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§        23
1basemodel/stream_0_conv_2/conv1d_1/ExpandDims/dimЉ
-basemodel/stream_0_conv_2/conv1d_1/ExpandDims
ExpandDims-basemodel/stream_0_drop_1/dropout_1/Mul_1:z:0:basemodel/stream_0_conv_2/conv1d_1/ExpandDims/dim:output:0*
T0*/
_output_shapes
:         }@2/
-basemodel/stream_0_conv_2/conv1d_1/ExpandDimsІ
>basemodel/stream_0_conv_2/conv1d_1/ExpandDims_1/ReadVariableOpReadVariableOpEbasemodel_stream_0_conv_2_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:@ђ*
dtype02@
>basemodel/stream_0_conv_2/conv1d_1/ExpandDims_1/ReadVariableOpг
3basemodel/stream_0_conv_2/conv1d_1/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 25
3basemodel/stream_0_conv_2/conv1d_1/ExpandDims_1/dimе
/basemodel/stream_0_conv_2/conv1d_1/ExpandDims_1
ExpandDimsFbasemodel/stream_0_conv_2/conv1d_1/ExpandDims_1/ReadVariableOp:value:0<basemodel/stream_0_conv_2/conv1d_1/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:@ђ21
/basemodel/stream_0_conv_2/conv1d_1/ExpandDims_1Д
"basemodel/stream_0_conv_2/conv1d_1Conv2D6basemodel/stream_0_conv_2/conv1d_1/ExpandDims:output:08basemodel/stream_0_conv_2/conv1d_1/ExpandDims_1:output:0*
T0*0
_output_shapes
:         }ђ*
paddingSAME*
strides
2$
"basemodel/stream_0_conv_2/conv1d_1у
*basemodel/stream_0_conv_2/conv1d_1/SqueezeSqueeze+basemodel/stream_0_conv_2/conv1d_1:output:0*
T0*,
_output_shapes
:         }ђ*
squeeze_dims

§        2,
*basemodel/stream_0_conv_2/conv1d_1/Squeeze▀
2basemodel/stream_0_conv_2/BiasAdd_1/ReadVariableOpReadVariableOp9basemodel_stream_0_conv_2_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype024
2basemodel/stream_0_conv_2/BiasAdd_1/ReadVariableOp§
#basemodel/stream_0_conv_2/BiasAdd_1BiasAdd3basemodel/stream_0_conv_2/conv1d_1/Squeeze:output:0:basemodel/stream_0_conv_2/BiasAdd_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:         }ђ2%
#basemodel/stream_0_conv_2/BiasAdd_1Н
@basemodel/batch_normalization_1/moments_1/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2B
@basemodel/batch_normalization_1/moments_1/mean/reduction_indicesа
.basemodel/batch_normalization_1/moments_1/meanMean,basemodel/stream_0_conv_2/BiasAdd_1:output:0Ibasemodel/batch_normalization_1/moments_1/mean/reduction_indices:output:0*
T0*#
_output_shapes
:ђ*
	keep_dims(20
.basemodel/batch_normalization_1/moments_1/meanу
6basemodel/batch_normalization_1/moments_1/StopGradientStopGradient7basemodel/batch_normalization_1/moments_1/mean:output:0*
T0*#
_output_shapes
:ђ28
6basemodel/batch_normalization_1/moments_1/StopGradientх
;basemodel/batch_normalization_1/moments_1/SquaredDifferenceSquaredDifference,basemodel/stream_0_conv_2/BiasAdd_1:output:0?basemodel/batch_normalization_1/moments_1/StopGradient:output:0*
T0*,
_output_shapes
:         }ђ2=
;basemodel/batch_normalization_1/moments_1/SquaredDifferenceП
Dbasemodel/batch_normalization_1/moments_1/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2F
Dbasemodel/batch_normalization_1/moments_1/variance/reduction_indices┐
2basemodel/batch_normalization_1/moments_1/varianceMean?basemodel/batch_normalization_1/moments_1/SquaredDifference:z:0Mbasemodel/batch_normalization_1/moments_1/variance/reduction_indices:output:0*
T0*#
_output_shapes
:ђ*
	keep_dims(24
2basemodel/batch_normalization_1/moments_1/varianceУ
1basemodel/batch_normalization_1/moments_1/SqueezeSqueeze7basemodel/batch_normalization_1/moments_1/mean:output:0*
T0*
_output_shapes	
:ђ*
squeeze_dims
 23
1basemodel/batch_normalization_1/moments_1/Squeeze­
3basemodel/batch_normalization_1/moments_1/Squeeze_1Squeeze;basemodel/batch_normalization_1/moments_1/variance:output:0*
T0*
_output_shapes	
:ђ*
squeeze_dims
 25
3basemodel/batch_normalization_1/moments_1/Squeeze_1и
7basemodel/batch_normalization_1/AssignMovingAvg_2/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<29
7basemodel/batch_normalization_1/AssignMovingAvg_2/decay╗
@basemodel/batch_normalization_1/AssignMovingAvg_2/ReadVariableOpReadVariableOpGbasemodel_batch_normalization_1_assignmovingavg_readvariableop_resource0^basemodel/batch_normalization_1/AssignMovingAvg*
_output_shapes	
:ђ*
dtype02B
@basemodel/batch_normalization_1/AssignMovingAvg_2/ReadVariableOpА
5basemodel/batch_normalization_1/AssignMovingAvg_2/subSubHbasemodel/batch_normalization_1/AssignMovingAvg_2/ReadVariableOp:value:0:basemodel/batch_normalization_1/moments_1/Squeeze:output:0*
T0*
_output_shapes	
:ђ27
5basemodel/batch_normalization_1/AssignMovingAvg_2/subў
5basemodel/batch_normalization_1/AssignMovingAvg_2/mulMul9basemodel/batch_normalization_1/AssignMovingAvg_2/sub:z:0@basemodel/batch_normalization_1/AssignMovingAvg_2/decay:output:0*
T0*
_output_shapes	
:ђ27
5basemodel/batch_normalization_1/AssignMovingAvg_2/mulЎ
1basemodel/batch_normalization_1/AssignMovingAvg_2AssignSubVariableOpGbasemodel_batch_normalization_1_assignmovingavg_readvariableop_resource9basemodel/batch_normalization_1/AssignMovingAvg_2/mul:z:00^basemodel/batch_normalization_1/AssignMovingAvgA^basemodel/batch_normalization_1/AssignMovingAvg_2/ReadVariableOp*
_output_shapes
 *
dtype023
1basemodel/batch_normalization_1/AssignMovingAvg_2и
7basemodel/batch_normalization_1/AssignMovingAvg_3/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<29
7basemodel/batch_normalization_1/AssignMovingAvg_3/decay┐
@basemodel/batch_normalization_1/AssignMovingAvg_3/ReadVariableOpReadVariableOpIbasemodel_batch_normalization_1_assignmovingavg_1_readvariableop_resource2^basemodel/batch_normalization_1/AssignMovingAvg_1*
_output_shapes	
:ђ*
dtype02B
@basemodel/batch_normalization_1/AssignMovingAvg_3/ReadVariableOpБ
5basemodel/batch_normalization_1/AssignMovingAvg_3/subSubHbasemodel/batch_normalization_1/AssignMovingAvg_3/ReadVariableOp:value:0<basemodel/batch_normalization_1/moments_1/Squeeze_1:output:0*
T0*
_output_shapes	
:ђ27
5basemodel/batch_normalization_1/AssignMovingAvg_3/subў
5basemodel/batch_normalization_1/AssignMovingAvg_3/mulMul9basemodel/batch_normalization_1/AssignMovingAvg_3/sub:z:0@basemodel/batch_normalization_1/AssignMovingAvg_3/decay:output:0*
T0*
_output_shapes	
:ђ27
5basemodel/batch_normalization_1/AssignMovingAvg_3/mulЮ
1basemodel/batch_normalization_1/AssignMovingAvg_3AssignSubVariableOpIbasemodel_batch_normalization_1_assignmovingavg_1_readvariableop_resource9basemodel/batch_normalization_1/AssignMovingAvg_3/mul:z:02^basemodel/batch_normalization_1/AssignMovingAvg_1A^basemodel/batch_normalization_1/AssignMovingAvg_3/ReadVariableOp*
_output_shapes
 *
dtype023
1basemodel/batch_normalization_1/AssignMovingAvg_3Ф
1basemodel/batch_normalization_1/batchnorm_1/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:23
1basemodel/batch_normalization_1/batchnorm_1/add/yІ
/basemodel/batch_normalization_1/batchnorm_1/addAddV2<basemodel/batch_normalization_1/moments_1/Squeeze_1:output:0:basemodel/batch_normalization_1/batchnorm_1/add/y:output:0*
T0*
_output_shapes	
:ђ21
/basemodel/batch_normalization_1/batchnorm_1/add╩
1basemodel/batch_normalization_1/batchnorm_1/RsqrtRsqrt3basemodel/batch_normalization_1/batchnorm_1/add:z:0*
T0*
_output_shapes	
:ђ23
1basemodel/batch_normalization_1/batchnorm_1/RsqrtЃ
>basemodel/batch_normalization_1/batchnorm_1/mul/ReadVariableOpReadVariableOpEbasemodel_batch_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes	
:ђ*
dtype02@
>basemodel/batch_normalization_1/batchnorm_1/mul/ReadVariableOpј
/basemodel/batch_normalization_1/batchnorm_1/mulMul5basemodel/batch_normalization_1/batchnorm_1/Rsqrt:y:0Fbasemodel/batch_normalization_1/batchnorm_1/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:ђ21
/basemodel/batch_normalization_1/batchnorm_1/mulЄ
1basemodel/batch_normalization_1/batchnorm_1/mul_1Mul,basemodel/stream_0_conv_2/BiasAdd_1:output:03basemodel/batch_normalization_1/batchnorm_1/mul:z:0*
T0*,
_output_shapes
:         }ђ23
1basemodel/batch_normalization_1/batchnorm_1/mul_1ё
1basemodel/batch_normalization_1/batchnorm_1/mul_2Mul:basemodel/batch_normalization_1/moments_1/Squeeze:output:03basemodel/batch_normalization_1/batchnorm_1/mul:z:0*
T0*
_output_shapes	
:ђ23
1basemodel/batch_normalization_1/batchnorm_1/mul_2э
:basemodel/batch_normalization_1/batchnorm_1/ReadVariableOpReadVariableOpAbasemodel_batch_normalization_1_batchnorm_readvariableop_resource*
_output_shapes	
:ђ*
dtype02<
:basemodel/batch_normalization_1/batchnorm_1/ReadVariableOpі
/basemodel/batch_normalization_1/batchnorm_1/subSubBbasemodel/batch_normalization_1/batchnorm_1/ReadVariableOp:value:05basemodel/batch_normalization_1/batchnorm_1/mul_2:z:0*
T0*
_output_shapes	
:ђ21
/basemodel/batch_normalization_1/batchnorm_1/subњ
1basemodel/batch_normalization_1/batchnorm_1/add_1AddV25basemodel/batch_normalization_1/batchnorm_1/mul_1:z:03basemodel/batch_normalization_1/batchnorm_1/sub:z:0*
T0*,
_output_shapes
:         }ђ23
1basemodel/batch_normalization_1/batchnorm_1/add_1┤
basemodel/activation_1/Relu_1Relu5basemodel/batch_normalization_1/batchnorm_1/add_1:z:0*
T0*,
_output_shapes
:         }ђ2
basemodel/activation_1/Relu_1Џ
)basemodel/stream_0_drop_2/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUН?2+
)basemodel/stream_0_drop_2/dropout_1/Constы
'basemodel/stream_0_drop_2/dropout_1/MulMul+basemodel/activation_1/Relu_1:activations:02basemodel/stream_0_drop_2/dropout_1/Const:output:0*
T0*,
_output_shapes
:         }ђ2)
'basemodel/stream_0_drop_2/dropout_1/Mul▒
)basemodel/stream_0_drop_2/dropout_1/ShapeShape+basemodel/activation_1/Relu_1:activations:0*
T0*
_output_shapes
:2+
)basemodel/stream_0_drop_2/dropout_1/Shapeе
@basemodel/stream_0_drop_2/dropout_1/random_uniform/RandomUniformRandomUniform2basemodel/stream_0_drop_2/dropout_1/Shape:output:0*
T0*,
_output_shapes
:         }ђ*
dtype0*
seedи*
seed2и2B
@basemodel/stream_0_drop_2/dropout_1/random_uniform/RandomUniformГ
2basemodel/stream_0_drop_2/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠>24
2basemodel/stream_0_drop_2/dropout_1/GreaterEqual/y│
0basemodel/stream_0_drop_2/dropout_1/GreaterEqualGreaterEqualIbasemodel/stream_0_drop_2/dropout_1/random_uniform/RandomUniform:output:0;basemodel/stream_0_drop_2/dropout_1/GreaterEqual/y:output:0*
T0*,
_output_shapes
:         }ђ22
0basemodel/stream_0_drop_2/dropout_1/GreaterEqualп
(basemodel/stream_0_drop_2/dropout_1/CastCast4basemodel/stream_0_drop_2/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:         }ђ2*
(basemodel/stream_0_drop_2/dropout_1/Cast№
)basemodel/stream_0_drop_2/dropout_1/Mul_1Mul+basemodel/stream_0_drop_2/dropout_1/Mul:z:0,basemodel/stream_0_drop_2/dropout_1/Cast:y:0*
T0*,
_output_shapes
:         }ђ2+
)basemodel/stream_0_drop_2/dropout_1/Mul_1▒
1basemodel/stream_0_conv_3/conv1d_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§        23
1basemodel/stream_0_conv_3/conv1d_1/ExpandDims/dimњ
-basemodel/stream_0_conv_3/conv1d_1/ExpandDims
ExpandDims-basemodel/stream_0_drop_2/dropout_1/Mul_1:z:0:basemodel/stream_0_conv_3/conv1d_1/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         }ђ2/
-basemodel/stream_0_conv_3/conv1d_1/ExpandDimsї
>basemodel/stream_0_conv_3/conv1d_1/ExpandDims_1/ReadVariableOpReadVariableOpEbasemodel_stream_0_conv_3_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:ђђ*
dtype02@
>basemodel/stream_0_conv_3/conv1d_1/ExpandDims_1/ReadVariableOpг
3basemodel/stream_0_conv_3/conv1d_1/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 25
3basemodel/stream_0_conv_3/conv1d_1/ExpandDims_1/dimЕ
/basemodel/stream_0_conv_3/conv1d_1/ExpandDims_1
ExpandDimsFbasemodel/stream_0_conv_3/conv1d_1/ExpandDims_1/ReadVariableOp:value:0<basemodel/stream_0_conv_3/conv1d_1/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:ђђ21
/basemodel/stream_0_conv_3/conv1d_1/ExpandDims_1Д
"basemodel/stream_0_conv_3/conv1d_1Conv2D6basemodel/stream_0_conv_3/conv1d_1/ExpandDims:output:08basemodel/stream_0_conv_3/conv1d_1/ExpandDims_1:output:0*
T0*0
_output_shapes
:         }ђ*
paddingSAME*
strides
2$
"basemodel/stream_0_conv_3/conv1d_1у
*basemodel/stream_0_conv_3/conv1d_1/SqueezeSqueeze+basemodel/stream_0_conv_3/conv1d_1:output:0*
T0*,
_output_shapes
:         }ђ*
squeeze_dims

§        2,
*basemodel/stream_0_conv_3/conv1d_1/Squeeze▀
2basemodel/stream_0_conv_3/BiasAdd_1/ReadVariableOpReadVariableOp9basemodel_stream_0_conv_3_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype024
2basemodel/stream_0_conv_3/BiasAdd_1/ReadVariableOp§
#basemodel/stream_0_conv_3/BiasAdd_1BiasAdd3basemodel/stream_0_conv_3/conv1d_1/Squeeze:output:0:basemodel/stream_0_conv_3/BiasAdd_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:         }ђ2%
#basemodel/stream_0_conv_3/BiasAdd_1Н
@basemodel/batch_normalization_2/moments_1/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2B
@basemodel/batch_normalization_2/moments_1/mean/reduction_indicesа
.basemodel/batch_normalization_2/moments_1/meanMean,basemodel/stream_0_conv_3/BiasAdd_1:output:0Ibasemodel/batch_normalization_2/moments_1/mean/reduction_indices:output:0*
T0*#
_output_shapes
:ђ*
	keep_dims(20
.basemodel/batch_normalization_2/moments_1/meanу
6basemodel/batch_normalization_2/moments_1/StopGradientStopGradient7basemodel/batch_normalization_2/moments_1/mean:output:0*
T0*#
_output_shapes
:ђ28
6basemodel/batch_normalization_2/moments_1/StopGradientх
;basemodel/batch_normalization_2/moments_1/SquaredDifferenceSquaredDifference,basemodel/stream_0_conv_3/BiasAdd_1:output:0?basemodel/batch_normalization_2/moments_1/StopGradient:output:0*
T0*,
_output_shapes
:         }ђ2=
;basemodel/batch_normalization_2/moments_1/SquaredDifferenceП
Dbasemodel/batch_normalization_2/moments_1/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2F
Dbasemodel/batch_normalization_2/moments_1/variance/reduction_indices┐
2basemodel/batch_normalization_2/moments_1/varianceMean?basemodel/batch_normalization_2/moments_1/SquaredDifference:z:0Mbasemodel/batch_normalization_2/moments_1/variance/reduction_indices:output:0*
T0*#
_output_shapes
:ђ*
	keep_dims(24
2basemodel/batch_normalization_2/moments_1/varianceУ
1basemodel/batch_normalization_2/moments_1/SqueezeSqueeze7basemodel/batch_normalization_2/moments_1/mean:output:0*
T0*
_output_shapes	
:ђ*
squeeze_dims
 23
1basemodel/batch_normalization_2/moments_1/Squeeze­
3basemodel/batch_normalization_2/moments_1/Squeeze_1Squeeze;basemodel/batch_normalization_2/moments_1/variance:output:0*
T0*
_output_shapes	
:ђ*
squeeze_dims
 25
3basemodel/batch_normalization_2/moments_1/Squeeze_1и
7basemodel/batch_normalization_2/AssignMovingAvg_2/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<29
7basemodel/batch_normalization_2/AssignMovingAvg_2/decay╗
@basemodel/batch_normalization_2/AssignMovingAvg_2/ReadVariableOpReadVariableOpGbasemodel_batch_normalization_2_assignmovingavg_readvariableop_resource0^basemodel/batch_normalization_2/AssignMovingAvg*
_output_shapes	
:ђ*
dtype02B
@basemodel/batch_normalization_2/AssignMovingAvg_2/ReadVariableOpА
5basemodel/batch_normalization_2/AssignMovingAvg_2/subSubHbasemodel/batch_normalization_2/AssignMovingAvg_2/ReadVariableOp:value:0:basemodel/batch_normalization_2/moments_1/Squeeze:output:0*
T0*
_output_shapes	
:ђ27
5basemodel/batch_normalization_2/AssignMovingAvg_2/subў
5basemodel/batch_normalization_2/AssignMovingAvg_2/mulMul9basemodel/batch_normalization_2/AssignMovingAvg_2/sub:z:0@basemodel/batch_normalization_2/AssignMovingAvg_2/decay:output:0*
T0*
_output_shapes	
:ђ27
5basemodel/batch_normalization_2/AssignMovingAvg_2/mulЎ
1basemodel/batch_normalization_2/AssignMovingAvg_2AssignSubVariableOpGbasemodel_batch_normalization_2_assignmovingavg_readvariableop_resource9basemodel/batch_normalization_2/AssignMovingAvg_2/mul:z:00^basemodel/batch_normalization_2/AssignMovingAvgA^basemodel/batch_normalization_2/AssignMovingAvg_2/ReadVariableOp*
_output_shapes
 *
dtype023
1basemodel/batch_normalization_2/AssignMovingAvg_2и
7basemodel/batch_normalization_2/AssignMovingAvg_3/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<29
7basemodel/batch_normalization_2/AssignMovingAvg_3/decay┐
@basemodel/batch_normalization_2/AssignMovingAvg_3/ReadVariableOpReadVariableOpIbasemodel_batch_normalization_2_assignmovingavg_1_readvariableop_resource2^basemodel/batch_normalization_2/AssignMovingAvg_1*
_output_shapes	
:ђ*
dtype02B
@basemodel/batch_normalization_2/AssignMovingAvg_3/ReadVariableOpБ
5basemodel/batch_normalization_2/AssignMovingAvg_3/subSubHbasemodel/batch_normalization_2/AssignMovingAvg_3/ReadVariableOp:value:0<basemodel/batch_normalization_2/moments_1/Squeeze_1:output:0*
T0*
_output_shapes	
:ђ27
5basemodel/batch_normalization_2/AssignMovingAvg_3/subў
5basemodel/batch_normalization_2/AssignMovingAvg_3/mulMul9basemodel/batch_normalization_2/AssignMovingAvg_3/sub:z:0@basemodel/batch_normalization_2/AssignMovingAvg_3/decay:output:0*
T0*
_output_shapes	
:ђ27
5basemodel/batch_normalization_2/AssignMovingAvg_3/mulЮ
1basemodel/batch_normalization_2/AssignMovingAvg_3AssignSubVariableOpIbasemodel_batch_normalization_2_assignmovingavg_1_readvariableop_resource9basemodel/batch_normalization_2/AssignMovingAvg_3/mul:z:02^basemodel/batch_normalization_2/AssignMovingAvg_1A^basemodel/batch_normalization_2/AssignMovingAvg_3/ReadVariableOp*
_output_shapes
 *
dtype023
1basemodel/batch_normalization_2/AssignMovingAvg_3Ф
1basemodel/batch_normalization_2/batchnorm_1/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:23
1basemodel/batch_normalization_2/batchnorm_1/add/yІ
/basemodel/batch_normalization_2/batchnorm_1/addAddV2<basemodel/batch_normalization_2/moments_1/Squeeze_1:output:0:basemodel/batch_normalization_2/batchnorm_1/add/y:output:0*
T0*
_output_shapes	
:ђ21
/basemodel/batch_normalization_2/batchnorm_1/add╩
1basemodel/batch_normalization_2/batchnorm_1/RsqrtRsqrt3basemodel/batch_normalization_2/batchnorm_1/add:z:0*
T0*
_output_shapes	
:ђ23
1basemodel/batch_normalization_2/batchnorm_1/RsqrtЃ
>basemodel/batch_normalization_2/batchnorm_1/mul/ReadVariableOpReadVariableOpEbasemodel_batch_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes	
:ђ*
dtype02@
>basemodel/batch_normalization_2/batchnorm_1/mul/ReadVariableOpј
/basemodel/batch_normalization_2/batchnorm_1/mulMul5basemodel/batch_normalization_2/batchnorm_1/Rsqrt:y:0Fbasemodel/batch_normalization_2/batchnorm_1/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:ђ21
/basemodel/batch_normalization_2/batchnorm_1/mulЄ
1basemodel/batch_normalization_2/batchnorm_1/mul_1Mul,basemodel/stream_0_conv_3/BiasAdd_1:output:03basemodel/batch_normalization_2/batchnorm_1/mul:z:0*
T0*,
_output_shapes
:         }ђ23
1basemodel/batch_normalization_2/batchnorm_1/mul_1ё
1basemodel/batch_normalization_2/batchnorm_1/mul_2Mul:basemodel/batch_normalization_2/moments_1/Squeeze:output:03basemodel/batch_normalization_2/batchnorm_1/mul:z:0*
T0*
_output_shapes	
:ђ23
1basemodel/batch_normalization_2/batchnorm_1/mul_2э
:basemodel/batch_normalization_2/batchnorm_1/ReadVariableOpReadVariableOpAbasemodel_batch_normalization_2_batchnorm_readvariableop_resource*
_output_shapes	
:ђ*
dtype02<
:basemodel/batch_normalization_2/batchnorm_1/ReadVariableOpі
/basemodel/batch_normalization_2/batchnorm_1/subSubBbasemodel/batch_normalization_2/batchnorm_1/ReadVariableOp:value:05basemodel/batch_normalization_2/batchnorm_1/mul_2:z:0*
T0*
_output_shapes	
:ђ21
/basemodel/batch_normalization_2/batchnorm_1/subњ
1basemodel/batch_normalization_2/batchnorm_1/add_1AddV25basemodel/batch_normalization_2/batchnorm_1/mul_1:z:03basemodel/batch_normalization_2/batchnorm_1/sub:z:0*
T0*,
_output_shapes
:         }ђ23
1basemodel/batch_normalization_2/batchnorm_1/add_1┤
basemodel/activation_2/Relu_1Relu5basemodel/batch_normalization_2/batchnorm_1/add_1:z:0*
T0*,
_output_shapes
:         }ђ2
basemodel/activation_2/Relu_1Џ
)basemodel/stream_0_drop_3/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2+
)basemodel/stream_0_drop_3/dropout_1/Constы
'basemodel/stream_0_drop_3/dropout_1/MulMul+basemodel/activation_2/Relu_1:activations:02basemodel/stream_0_drop_3/dropout_1/Const:output:0*
T0*,
_output_shapes
:         }ђ2)
'basemodel/stream_0_drop_3/dropout_1/Mul▒
)basemodel/stream_0_drop_3/dropout_1/ShapeShape+basemodel/activation_2/Relu_1:activations:0*
T0*
_output_shapes
:2+
)basemodel/stream_0_drop_3/dropout_1/Shapeе
@basemodel/stream_0_drop_3/dropout_1/random_uniform/RandomUniformRandomUniform2basemodel/stream_0_drop_3/dropout_1/Shape:output:0*
T0*,
_output_shapes
:         }ђ*
dtype0*
seedи*
seed2и2B
@basemodel/stream_0_drop_3/dropout_1/random_uniform/RandomUniformГ
2basemodel/stream_0_drop_3/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?24
2basemodel/stream_0_drop_3/dropout_1/GreaterEqual/y│
0basemodel/stream_0_drop_3/dropout_1/GreaterEqualGreaterEqualIbasemodel/stream_0_drop_3/dropout_1/random_uniform/RandomUniform:output:0;basemodel/stream_0_drop_3/dropout_1/GreaterEqual/y:output:0*
T0*,
_output_shapes
:         }ђ22
0basemodel/stream_0_drop_3/dropout_1/GreaterEqualп
(basemodel/stream_0_drop_3/dropout_1/CastCast4basemodel/stream_0_drop_3/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:         }ђ2*
(basemodel/stream_0_drop_3/dropout_1/Cast№
)basemodel/stream_0_drop_3/dropout_1/Mul_1Mul+basemodel/stream_0_drop_3/dropout_1/Mul:z:0,basemodel/stream_0_drop_3/dropout_1/Cast:y:0*
T0*,
_output_shapes
:         }ђ2+
)basemodel/stream_0_drop_3/dropout_1/Mul_1╝
;basemodel/global_average_pooling1d/Mean_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2=
;basemodel/global_average_pooling1d/Mean_1/reduction_indicesє
)basemodel/global_average_pooling1d/Mean_1Mean-basemodel/stream_0_drop_3/dropout_1/Mul_1:z:0Dbasemodel/global_average_pooling1d/Mean_1/reduction_indices:output:0*
T0*(
_output_shapes
:         ђ2+
)basemodel/global_average_pooling1d/Mean_1Џ
)basemodel/dense_1_dropout/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?2+
)basemodel/dense_1_dropout/dropout_1/ConstЗ
'basemodel/dense_1_dropout/dropout_1/MulMul2basemodel/global_average_pooling1d/Mean_1:output:02basemodel/dense_1_dropout/dropout_1/Const:output:0*
T0*(
_output_shapes
:         ђ2)
'basemodel/dense_1_dropout/dropout_1/MulИ
)basemodel/dense_1_dropout/dropout_1/ShapeShape2basemodel/global_average_pooling1d/Mean_1:output:0*
T0*
_output_shapes
:2+
)basemodel/dense_1_dropout/dropout_1/ShapeБ
@basemodel/dense_1_dropout/dropout_1/random_uniform/RandomUniformRandomUniform2basemodel/dense_1_dropout/dropout_1/Shape:output:0*
T0*(
_output_shapes
:         ђ*
dtype0*
seedи*
seed22B
@basemodel/dense_1_dropout/dropout_1/random_uniform/RandomUniformГ
2basemodel/dense_1_dropout/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>24
2basemodel/dense_1_dropout/dropout_1/GreaterEqual/y»
0basemodel/dense_1_dropout/dropout_1/GreaterEqualGreaterEqualIbasemodel/dense_1_dropout/dropout_1/random_uniform/RandomUniform:output:0;basemodel/dense_1_dropout/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         ђ22
0basemodel/dense_1_dropout/dropout_1/GreaterEqualн
(basemodel/dense_1_dropout/dropout_1/CastCast4basemodel/dense_1_dropout/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         ђ2*
(basemodel/dense_1_dropout/dropout_1/Castв
)basemodel/dense_1_dropout/dropout_1/Mul_1Mul+basemodel/dense_1_dropout/dropout_1/Mul:z:0,basemodel/dense_1_dropout/dropout_1/Cast:y:0*
T0*(
_output_shapes
:         ђ2+
)basemodel/dense_1_dropout/dropout_1/Mul_1╚
)basemodel/dense_1/MatMul_1/ReadVariableOpReadVariableOp0basemodel_dense_1_matmul_readvariableop_resource*
_output_shapes
:	ђT*
dtype02+
)basemodel/dense_1/MatMul_1/ReadVariableOpо
basemodel/dense_1/MatMul_1MatMul-basemodel/dense_1_dropout/dropout_1/Mul_1:z:01basemodel/dense_1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         T2
basemodel/dense_1/MatMul_1к
*basemodel/dense_1/BiasAdd_1/ReadVariableOpReadVariableOp1basemodel_dense_1_biasadd_readvariableop_resource*
_output_shapes
:T*
dtype02,
*basemodel/dense_1/BiasAdd_1/ReadVariableOpЛ
basemodel/dense_1/BiasAdd_1BiasAdd$basemodel/dense_1/MatMul_1:product:02basemodel/dense_1/BiasAdd_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         T2
basemodel/dense_1/BiasAdd_1╬
@basemodel/batch_normalization_3/moments_1/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2B
@basemodel/batch_normalization_3/moments_1/mean/reduction_indicesЊ
.basemodel/batch_normalization_3/moments_1/meanMean$basemodel/dense_1/BiasAdd_1:output:0Ibasemodel/batch_normalization_3/moments_1/mean/reduction_indices:output:0*
T0*
_output_shapes

:T*
	keep_dims(20
.basemodel/batch_normalization_3/moments_1/meanР
6basemodel/batch_normalization_3/moments_1/StopGradientStopGradient7basemodel/batch_normalization_3/moments_1/mean:output:0*
T0*
_output_shapes

:T28
6basemodel/batch_normalization_3/moments_1/StopGradientе
;basemodel/batch_normalization_3/moments_1/SquaredDifferenceSquaredDifference$basemodel/dense_1/BiasAdd_1:output:0?basemodel/batch_normalization_3/moments_1/StopGradient:output:0*
T0*'
_output_shapes
:         T2=
;basemodel/batch_normalization_3/moments_1/SquaredDifferenceо
Dbasemodel/batch_normalization_3/moments_1/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2F
Dbasemodel/batch_normalization_3/moments_1/variance/reduction_indices║
2basemodel/batch_normalization_3/moments_1/varianceMean?basemodel/batch_normalization_3/moments_1/SquaredDifference:z:0Mbasemodel/batch_normalization_3/moments_1/variance/reduction_indices:output:0*
T0*
_output_shapes

:T*
	keep_dims(24
2basemodel/batch_normalization_3/moments_1/varianceТ
1basemodel/batch_normalization_3/moments_1/SqueezeSqueeze7basemodel/batch_normalization_3/moments_1/mean:output:0*
T0*
_output_shapes
:T*
squeeze_dims
 23
1basemodel/batch_normalization_3/moments_1/SqueezeЬ
3basemodel/batch_normalization_3/moments_1/Squeeze_1Squeeze;basemodel/batch_normalization_3/moments_1/variance:output:0*
T0*
_output_shapes
:T*
squeeze_dims
 25
3basemodel/batch_normalization_3/moments_1/Squeeze_1и
7basemodel/batch_normalization_3/AssignMovingAvg_2/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<29
7basemodel/batch_normalization_3/AssignMovingAvg_2/decay║
@basemodel/batch_normalization_3/AssignMovingAvg_2/ReadVariableOpReadVariableOpGbasemodel_batch_normalization_3_assignmovingavg_readvariableop_resource0^basemodel/batch_normalization_3/AssignMovingAvg*
_output_shapes
:T*
dtype02B
@basemodel/batch_normalization_3/AssignMovingAvg_2/ReadVariableOpа
5basemodel/batch_normalization_3/AssignMovingAvg_2/subSubHbasemodel/batch_normalization_3/AssignMovingAvg_2/ReadVariableOp:value:0:basemodel/batch_normalization_3/moments_1/Squeeze:output:0*
T0*
_output_shapes
:T27
5basemodel/batch_normalization_3/AssignMovingAvg_2/subЌ
5basemodel/batch_normalization_3/AssignMovingAvg_2/mulMul9basemodel/batch_normalization_3/AssignMovingAvg_2/sub:z:0@basemodel/batch_normalization_3/AssignMovingAvg_2/decay:output:0*
T0*
_output_shapes
:T27
5basemodel/batch_normalization_3/AssignMovingAvg_2/mulЎ
1basemodel/batch_normalization_3/AssignMovingAvg_2AssignSubVariableOpGbasemodel_batch_normalization_3_assignmovingavg_readvariableop_resource9basemodel/batch_normalization_3/AssignMovingAvg_2/mul:z:00^basemodel/batch_normalization_3/AssignMovingAvgA^basemodel/batch_normalization_3/AssignMovingAvg_2/ReadVariableOp*
_output_shapes
 *
dtype023
1basemodel/batch_normalization_3/AssignMovingAvg_2и
7basemodel/batch_normalization_3/AssignMovingAvg_3/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<29
7basemodel/batch_normalization_3/AssignMovingAvg_3/decayЙ
@basemodel/batch_normalization_3/AssignMovingAvg_3/ReadVariableOpReadVariableOpIbasemodel_batch_normalization_3_assignmovingavg_1_readvariableop_resource2^basemodel/batch_normalization_3/AssignMovingAvg_1*
_output_shapes
:T*
dtype02B
@basemodel/batch_normalization_3/AssignMovingAvg_3/ReadVariableOpб
5basemodel/batch_normalization_3/AssignMovingAvg_3/subSubHbasemodel/batch_normalization_3/AssignMovingAvg_3/ReadVariableOp:value:0<basemodel/batch_normalization_3/moments_1/Squeeze_1:output:0*
T0*
_output_shapes
:T27
5basemodel/batch_normalization_3/AssignMovingAvg_3/subЌ
5basemodel/batch_normalization_3/AssignMovingAvg_3/mulMul9basemodel/batch_normalization_3/AssignMovingAvg_3/sub:z:0@basemodel/batch_normalization_3/AssignMovingAvg_3/decay:output:0*
T0*
_output_shapes
:T27
5basemodel/batch_normalization_3/AssignMovingAvg_3/mulЮ
1basemodel/batch_normalization_3/AssignMovingAvg_3AssignSubVariableOpIbasemodel_batch_normalization_3_assignmovingavg_1_readvariableop_resource9basemodel/batch_normalization_3/AssignMovingAvg_3/mul:z:02^basemodel/batch_normalization_3/AssignMovingAvg_1A^basemodel/batch_normalization_3/AssignMovingAvg_3/ReadVariableOp*
_output_shapes
 *
dtype023
1basemodel/batch_normalization_3/AssignMovingAvg_3Ф
1basemodel/batch_normalization_3/batchnorm_1/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:23
1basemodel/batch_normalization_3/batchnorm_1/add/yі
/basemodel/batch_normalization_3/batchnorm_1/addAddV2<basemodel/batch_normalization_3/moments_1/Squeeze_1:output:0:basemodel/batch_normalization_3/batchnorm_1/add/y:output:0*
T0*
_output_shapes
:T21
/basemodel/batch_normalization_3/batchnorm_1/add╔
1basemodel/batch_normalization_3/batchnorm_1/RsqrtRsqrt3basemodel/batch_normalization_3/batchnorm_1/add:z:0*
T0*
_output_shapes
:T23
1basemodel/batch_normalization_3/batchnorm_1/Rsqrtѓ
>basemodel/batch_normalization_3/batchnorm_1/mul/ReadVariableOpReadVariableOpEbasemodel_batch_normalization_3_batchnorm_mul_readvariableop_resource*
_output_shapes
:T*
dtype02@
>basemodel/batch_normalization_3/batchnorm_1/mul/ReadVariableOpЇ
/basemodel/batch_normalization_3/batchnorm_1/mulMul5basemodel/batch_normalization_3/batchnorm_1/Rsqrt:y:0Fbasemodel/batch_normalization_3/batchnorm_1/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:T21
/basemodel/batch_normalization_3/batchnorm_1/mulЩ
1basemodel/batch_normalization_3/batchnorm_1/mul_1Mul$basemodel/dense_1/BiasAdd_1:output:03basemodel/batch_normalization_3/batchnorm_1/mul:z:0*
T0*'
_output_shapes
:         T23
1basemodel/batch_normalization_3/batchnorm_1/mul_1Ѓ
1basemodel/batch_normalization_3/batchnorm_1/mul_2Mul:basemodel/batch_normalization_3/moments_1/Squeeze:output:03basemodel/batch_normalization_3/batchnorm_1/mul:z:0*
T0*
_output_shapes
:T23
1basemodel/batch_normalization_3/batchnorm_1/mul_2Ш
:basemodel/batch_normalization_3/batchnorm_1/ReadVariableOpReadVariableOpAbasemodel_batch_normalization_3_batchnorm_readvariableop_resource*
_output_shapes
:T*
dtype02<
:basemodel/batch_normalization_3/batchnorm_1/ReadVariableOpЅ
/basemodel/batch_normalization_3/batchnorm_1/subSubBbasemodel/batch_normalization_3/batchnorm_1/ReadVariableOp:value:05basemodel/batch_normalization_3/batchnorm_1/mul_2:z:0*
T0*
_output_shapes
:T21
/basemodel/batch_normalization_3/batchnorm_1/subЇ
1basemodel/batch_normalization_3/batchnorm_1/add_1AddV25basemodel/batch_normalization_3/batchnorm_1/mul_1:z:03basemodel/batch_normalization_3/batchnorm_1/sub:z:0*
T0*'
_output_shapes
:         T23
1basemodel/batch_normalization_3/batchnorm_1/add_1─
&basemodel/dense_activation_1/Sigmoid_1Sigmoid5basemodel/batch_normalization_3/batchnorm_1/add_1:z:0*
T0*'
_output_shapes
:         T2(
&basemodel/dense_activation_1/Sigmoid_1Ф
distance/subSub(basemodel/dense_activation_1/Sigmoid:y:0*basemodel/dense_activation_1/Sigmoid_1:y:0*
T0*'
_output_shapes
:         T2
distance/subp
distance/SquareSquaredistance/sub:z:0*
T0*'
_output_shapes
:         T2
distance/SquareІ
distance/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
         2 
distance/Sum/reduction_indicesц
distance/SumSumdistance/Square:y:0'distance/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:         *
	keep_dims(2
distance/Sume
distance/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
distance/ConstЉ
distance/MaximumMaximumdistance/Sum:output:0distance/Const:output:0*
T0*'
_output_shapes
:         2
distance/Maximumn
distance/SqrtSqrtdistance/Maximum:z:0*
T0*'
_output_shapes
:         2
distance/SqrtЭ
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpEbasemodel_stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype027
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp├
&stream_0_conv_1/kernel/Regularizer/AbsAbs=stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_0_conv_1/kernel/Regularizer/AbsЕ
(stream_0_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_0_conv_1/kernel/Regularizer/ConstО
&stream_0_conv_1/kernel/Regularizer/SumSum*stream_0_conv_1/kernel/Regularizer/Abs:y:01stream_0_conv_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/SumЎ
(stream_0_conv_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2*
(stream_0_conv_1/kernel/Regularizer/mul/x▄
&stream_0_conv_1/kernel/Regularizer/mulMul1stream_0_conv_1/kernel/Regularizer/mul/x:output:0/stream_0_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/mul 
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpEbasemodel_stream_0_conv_2_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:@ђ*
dtype02:
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOpл
)stream_0_conv_2/kernel/Regularizer/SquareSquare@stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*#
_output_shapes
:@ђ2+
)stream_0_conv_2/kernel/Regularizer/SquareЕ
(stream_0_conv_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_0_conv_2/kernel/Regularizer/Const┌
&stream_0_conv_2/kernel/Regularizer/SumSum-stream_0_conv_2/kernel/Regularizer/Square:y:01stream_0_conv_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_2/kernel/Regularizer/SumЎ
(stream_0_conv_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2*
(stream_0_conv_2/kernel/Regularizer/mul/x▄
&stream_0_conv_2/kernel/Regularizer/mulMul1stream_0_conv_2/kernel/Regularizer/mul/x:output:0/stream_0_conv_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_2/kernel/Regularizer/mulЩ
5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpEbasemodel_stream_0_conv_3_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:ђђ*
dtype027
5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp┼
&stream_0_conv_3/kernel/Regularizer/AbsAbs=stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*$
_output_shapes
:ђђ2(
&stream_0_conv_3/kernel/Regularizer/AbsЕ
(stream_0_conv_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_0_conv_3/kernel/Regularizer/ConstО
&stream_0_conv_3/kernel/Regularizer/SumSum*stream_0_conv_3/kernel/Regularizer/Abs:y:01stream_0_conv_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_3/kernel/Regularizer/SumЎ
(stream_0_conv_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2*
(stream_0_conv_3/kernel/Regularizer/mul/x▄
&stream_0_conv_3/kernel/Regularizer/mulMul1stream_0_conv_3/kernel/Regularizer/mul/x:output:0/stream_0_conv_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_3/kernel/Regularizer/mulл
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp0basemodel_dense_1_matmul_readvariableop_resource*
_output_shapes
:	ђT*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOpе
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	ђT2 
dense_1/kernel/Regularizer/AbsЋ
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_1/kernel/Regularizer/Constи
dense_1/kernel/Regularizer/SumSum"dense_1/kernel/Regularizer/Abs:y:0)dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/SumЅ
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2"
 dense_1/kernel/Regularizer/mul/x╝
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/mull
IdentityIdentitydistance/Sqrt:y:0^NoOp*
T0*'
_output_shapes
:         2

Identityи
NoOpNoOp.^basemodel/batch_normalization/AssignMovingAvg=^basemodel/batch_normalization/AssignMovingAvg/ReadVariableOp0^basemodel/batch_normalization/AssignMovingAvg_1?^basemodel/batch_normalization/AssignMovingAvg_1/ReadVariableOp0^basemodel/batch_normalization/AssignMovingAvg_2?^basemodel/batch_normalization/AssignMovingAvg_2/ReadVariableOp0^basemodel/batch_normalization/AssignMovingAvg_3?^basemodel/batch_normalization/AssignMovingAvg_3/ReadVariableOp7^basemodel/batch_normalization/batchnorm/ReadVariableOp;^basemodel/batch_normalization/batchnorm/mul/ReadVariableOp9^basemodel/batch_normalization/batchnorm_1/ReadVariableOp=^basemodel/batch_normalization/batchnorm_1/mul/ReadVariableOp0^basemodel/batch_normalization_1/AssignMovingAvg?^basemodel/batch_normalization_1/AssignMovingAvg/ReadVariableOp2^basemodel/batch_normalization_1/AssignMovingAvg_1A^basemodel/batch_normalization_1/AssignMovingAvg_1/ReadVariableOp2^basemodel/batch_normalization_1/AssignMovingAvg_2A^basemodel/batch_normalization_1/AssignMovingAvg_2/ReadVariableOp2^basemodel/batch_normalization_1/AssignMovingAvg_3A^basemodel/batch_normalization_1/AssignMovingAvg_3/ReadVariableOp9^basemodel/batch_normalization_1/batchnorm/ReadVariableOp=^basemodel/batch_normalization_1/batchnorm/mul/ReadVariableOp;^basemodel/batch_normalization_1/batchnorm_1/ReadVariableOp?^basemodel/batch_normalization_1/batchnorm_1/mul/ReadVariableOp0^basemodel/batch_normalization_2/AssignMovingAvg?^basemodel/batch_normalization_2/AssignMovingAvg/ReadVariableOp2^basemodel/batch_normalization_2/AssignMovingAvg_1A^basemodel/batch_normalization_2/AssignMovingAvg_1/ReadVariableOp2^basemodel/batch_normalization_2/AssignMovingAvg_2A^basemodel/batch_normalization_2/AssignMovingAvg_2/ReadVariableOp2^basemodel/batch_normalization_2/AssignMovingAvg_3A^basemodel/batch_normalization_2/AssignMovingAvg_3/ReadVariableOp9^basemodel/batch_normalization_2/batchnorm/ReadVariableOp=^basemodel/batch_normalization_2/batchnorm/mul/ReadVariableOp;^basemodel/batch_normalization_2/batchnorm_1/ReadVariableOp?^basemodel/batch_normalization_2/batchnorm_1/mul/ReadVariableOp0^basemodel/batch_normalization_3/AssignMovingAvg?^basemodel/batch_normalization_3/AssignMovingAvg/ReadVariableOp2^basemodel/batch_normalization_3/AssignMovingAvg_1A^basemodel/batch_normalization_3/AssignMovingAvg_1/ReadVariableOp2^basemodel/batch_normalization_3/AssignMovingAvg_2A^basemodel/batch_normalization_3/AssignMovingAvg_2/ReadVariableOp2^basemodel/batch_normalization_3/AssignMovingAvg_3A^basemodel/batch_normalization_3/AssignMovingAvg_3/ReadVariableOp9^basemodel/batch_normalization_3/batchnorm/ReadVariableOp=^basemodel/batch_normalization_3/batchnorm/mul/ReadVariableOp;^basemodel/batch_normalization_3/batchnorm_1/ReadVariableOp?^basemodel/batch_normalization_3/batchnorm_1/mul/ReadVariableOp)^basemodel/dense_1/BiasAdd/ReadVariableOp+^basemodel/dense_1/BiasAdd_1/ReadVariableOp(^basemodel/dense_1/MatMul/ReadVariableOp*^basemodel/dense_1/MatMul_1/ReadVariableOp1^basemodel/stream_0_conv_1/BiasAdd/ReadVariableOp3^basemodel/stream_0_conv_1/BiasAdd_1/ReadVariableOp=^basemodel/stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp?^basemodel/stream_0_conv_1/conv1d_1/ExpandDims_1/ReadVariableOp1^basemodel/stream_0_conv_2/BiasAdd/ReadVariableOp3^basemodel/stream_0_conv_2/BiasAdd_1/ReadVariableOp=^basemodel/stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOp?^basemodel/stream_0_conv_2/conv1d_1/ExpandDims_1/ReadVariableOp1^basemodel/stream_0_conv_3/BiasAdd/ReadVariableOp3^basemodel/stream_0_conv_3/BiasAdd_1/ReadVariableOp=^basemodel/stream_0_conv_3/conv1d/ExpandDims_1/ReadVariableOp?^basemodel/stream_0_conv_3/conv1d_1/ExpandDims_1/ReadVariableOp.^dense_1/kernel/Regularizer/Abs/ReadVariableOp6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp9^stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp6^stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*q
_input_shapes`
^:         }:         }: : : : : : : : : : : : : : : : : : : : : : : : 2^
-basemodel/batch_normalization/AssignMovingAvg-basemodel/batch_normalization/AssignMovingAvg2|
<basemodel/batch_normalization/AssignMovingAvg/ReadVariableOp<basemodel/batch_normalization/AssignMovingAvg/ReadVariableOp2b
/basemodel/batch_normalization/AssignMovingAvg_1/basemodel/batch_normalization/AssignMovingAvg_12ђ
>basemodel/batch_normalization/AssignMovingAvg_1/ReadVariableOp>basemodel/batch_normalization/AssignMovingAvg_1/ReadVariableOp2b
/basemodel/batch_normalization/AssignMovingAvg_2/basemodel/batch_normalization/AssignMovingAvg_22ђ
>basemodel/batch_normalization/AssignMovingAvg_2/ReadVariableOp>basemodel/batch_normalization/AssignMovingAvg_2/ReadVariableOp2b
/basemodel/batch_normalization/AssignMovingAvg_3/basemodel/batch_normalization/AssignMovingAvg_32ђ
>basemodel/batch_normalization/AssignMovingAvg_3/ReadVariableOp>basemodel/batch_normalization/AssignMovingAvg_3/ReadVariableOp2p
6basemodel/batch_normalization/batchnorm/ReadVariableOp6basemodel/batch_normalization/batchnorm/ReadVariableOp2x
:basemodel/batch_normalization/batchnorm/mul/ReadVariableOp:basemodel/batch_normalization/batchnorm/mul/ReadVariableOp2t
8basemodel/batch_normalization/batchnorm_1/ReadVariableOp8basemodel/batch_normalization/batchnorm_1/ReadVariableOp2|
<basemodel/batch_normalization/batchnorm_1/mul/ReadVariableOp<basemodel/batch_normalization/batchnorm_1/mul/ReadVariableOp2b
/basemodel/batch_normalization_1/AssignMovingAvg/basemodel/batch_normalization_1/AssignMovingAvg2ђ
>basemodel/batch_normalization_1/AssignMovingAvg/ReadVariableOp>basemodel/batch_normalization_1/AssignMovingAvg/ReadVariableOp2f
1basemodel/batch_normalization_1/AssignMovingAvg_11basemodel/batch_normalization_1/AssignMovingAvg_12ё
@basemodel/batch_normalization_1/AssignMovingAvg_1/ReadVariableOp@basemodel/batch_normalization_1/AssignMovingAvg_1/ReadVariableOp2f
1basemodel/batch_normalization_1/AssignMovingAvg_21basemodel/batch_normalization_1/AssignMovingAvg_22ё
@basemodel/batch_normalization_1/AssignMovingAvg_2/ReadVariableOp@basemodel/batch_normalization_1/AssignMovingAvg_2/ReadVariableOp2f
1basemodel/batch_normalization_1/AssignMovingAvg_31basemodel/batch_normalization_1/AssignMovingAvg_32ё
@basemodel/batch_normalization_1/AssignMovingAvg_3/ReadVariableOp@basemodel/batch_normalization_1/AssignMovingAvg_3/ReadVariableOp2t
8basemodel/batch_normalization_1/batchnorm/ReadVariableOp8basemodel/batch_normalization_1/batchnorm/ReadVariableOp2|
<basemodel/batch_normalization_1/batchnorm/mul/ReadVariableOp<basemodel/batch_normalization_1/batchnorm/mul/ReadVariableOp2x
:basemodel/batch_normalization_1/batchnorm_1/ReadVariableOp:basemodel/batch_normalization_1/batchnorm_1/ReadVariableOp2ђ
>basemodel/batch_normalization_1/batchnorm_1/mul/ReadVariableOp>basemodel/batch_normalization_1/batchnorm_1/mul/ReadVariableOp2b
/basemodel/batch_normalization_2/AssignMovingAvg/basemodel/batch_normalization_2/AssignMovingAvg2ђ
>basemodel/batch_normalization_2/AssignMovingAvg/ReadVariableOp>basemodel/batch_normalization_2/AssignMovingAvg/ReadVariableOp2f
1basemodel/batch_normalization_2/AssignMovingAvg_11basemodel/batch_normalization_2/AssignMovingAvg_12ё
@basemodel/batch_normalization_2/AssignMovingAvg_1/ReadVariableOp@basemodel/batch_normalization_2/AssignMovingAvg_1/ReadVariableOp2f
1basemodel/batch_normalization_2/AssignMovingAvg_21basemodel/batch_normalization_2/AssignMovingAvg_22ё
@basemodel/batch_normalization_2/AssignMovingAvg_2/ReadVariableOp@basemodel/batch_normalization_2/AssignMovingAvg_2/ReadVariableOp2f
1basemodel/batch_normalization_2/AssignMovingAvg_31basemodel/batch_normalization_2/AssignMovingAvg_32ё
@basemodel/batch_normalization_2/AssignMovingAvg_3/ReadVariableOp@basemodel/batch_normalization_2/AssignMovingAvg_3/ReadVariableOp2t
8basemodel/batch_normalization_2/batchnorm/ReadVariableOp8basemodel/batch_normalization_2/batchnorm/ReadVariableOp2|
<basemodel/batch_normalization_2/batchnorm/mul/ReadVariableOp<basemodel/batch_normalization_2/batchnorm/mul/ReadVariableOp2x
:basemodel/batch_normalization_2/batchnorm_1/ReadVariableOp:basemodel/batch_normalization_2/batchnorm_1/ReadVariableOp2ђ
>basemodel/batch_normalization_2/batchnorm_1/mul/ReadVariableOp>basemodel/batch_normalization_2/batchnorm_1/mul/ReadVariableOp2b
/basemodel/batch_normalization_3/AssignMovingAvg/basemodel/batch_normalization_3/AssignMovingAvg2ђ
>basemodel/batch_normalization_3/AssignMovingAvg/ReadVariableOp>basemodel/batch_normalization_3/AssignMovingAvg/ReadVariableOp2f
1basemodel/batch_normalization_3/AssignMovingAvg_11basemodel/batch_normalization_3/AssignMovingAvg_12ё
@basemodel/batch_normalization_3/AssignMovingAvg_1/ReadVariableOp@basemodel/batch_normalization_3/AssignMovingAvg_1/ReadVariableOp2f
1basemodel/batch_normalization_3/AssignMovingAvg_21basemodel/batch_normalization_3/AssignMovingAvg_22ё
@basemodel/batch_normalization_3/AssignMovingAvg_2/ReadVariableOp@basemodel/batch_normalization_3/AssignMovingAvg_2/ReadVariableOp2f
1basemodel/batch_normalization_3/AssignMovingAvg_31basemodel/batch_normalization_3/AssignMovingAvg_32ё
@basemodel/batch_normalization_3/AssignMovingAvg_3/ReadVariableOp@basemodel/batch_normalization_3/AssignMovingAvg_3/ReadVariableOp2t
8basemodel/batch_normalization_3/batchnorm/ReadVariableOp8basemodel/batch_normalization_3/batchnorm/ReadVariableOp2|
<basemodel/batch_normalization_3/batchnorm/mul/ReadVariableOp<basemodel/batch_normalization_3/batchnorm/mul/ReadVariableOp2x
:basemodel/batch_normalization_3/batchnorm_1/ReadVariableOp:basemodel/batch_normalization_3/batchnorm_1/ReadVariableOp2ђ
>basemodel/batch_normalization_3/batchnorm_1/mul/ReadVariableOp>basemodel/batch_normalization_3/batchnorm_1/mul/ReadVariableOp2T
(basemodel/dense_1/BiasAdd/ReadVariableOp(basemodel/dense_1/BiasAdd/ReadVariableOp2X
*basemodel/dense_1/BiasAdd_1/ReadVariableOp*basemodel/dense_1/BiasAdd_1/ReadVariableOp2R
'basemodel/dense_1/MatMul/ReadVariableOp'basemodel/dense_1/MatMul/ReadVariableOp2V
)basemodel/dense_1/MatMul_1/ReadVariableOp)basemodel/dense_1/MatMul_1/ReadVariableOp2d
0basemodel/stream_0_conv_1/BiasAdd/ReadVariableOp0basemodel/stream_0_conv_1/BiasAdd/ReadVariableOp2h
2basemodel/stream_0_conv_1/BiasAdd_1/ReadVariableOp2basemodel/stream_0_conv_1/BiasAdd_1/ReadVariableOp2|
<basemodel/stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp<basemodel/stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp2ђ
>basemodel/stream_0_conv_1/conv1d_1/ExpandDims_1/ReadVariableOp>basemodel/stream_0_conv_1/conv1d_1/ExpandDims_1/ReadVariableOp2d
0basemodel/stream_0_conv_2/BiasAdd/ReadVariableOp0basemodel/stream_0_conv_2/BiasAdd/ReadVariableOp2h
2basemodel/stream_0_conv_2/BiasAdd_1/ReadVariableOp2basemodel/stream_0_conv_2/BiasAdd_1/ReadVariableOp2|
<basemodel/stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOp<basemodel/stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOp2ђ
>basemodel/stream_0_conv_2/conv1d_1/ExpandDims_1/ReadVariableOp>basemodel/stream_0_conv_2/conv1d_1/ExpandDims_1/ReadVariableOp2d
0basemodel/stream_0_conv_3/BiasAdd/ReadVariableOp0basemodel/stream_0_conv_3/BiasAdd/ReadVariableOp2h
2basemodel/stream_0_conv_3/BiasAdd_1/ReadVariableOp2basemodel/stream_0_conv_3/BiasAdd_1/ReadVariableOp2|
<basemodel/stream_0_conv_3/conv1d/ExpandDims_1/ReadVariableOp<basemodel/stream_0_conv_3/conv1d/ExpandDims_1/ReadVariableOp2ђ
>basemodel/stream_0_conv_3/conv1d_1/ExpandDims_1/ReadVariableOp>basemodel/stream_0_conv_3/conv1d_1/ExpandDims_1/ReadVariableOp2^
-dense_1/kernel/Regularizer/Abs/ReadVariableOp-dense_1/kernel/Regularizer/Abs/ReadVariableOp2n
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp2t
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp2n
5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp:U Q
+
_output_shapes
:         }
"
_user_specified_name
inputs/0:UQ
+
_output_shapes
:         }
"
_user_specified_name
inputs/1
ю+
ь
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_24344

inputs6
'assignmovingavg_readvariableop_resource:	ђ8
)assignmovingavg_1_readvariableop_resource:	ђ4
%batchnorm_mul_readvariableop_resource:	ђ0
!batchnorm_readvariableop_resource:	ђ
identityѕбAssignMovingAvgбAssignMovingAvg/ReadVariableOpбAssignMovingAvg_1б AssignMovingAvg_1/ReadVariableOpбbatchnorm/ReadVariableOpбbatchnorm/mul/ReadVariableOpЉ
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indicesћ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*#
_output_shapes
:ђ*
	keep_dims(2
moments/meanЂ
moments/StopGradientStopGradientmoments/mean:output:0*
T0*#
_output_shapes
:ђ2
moments/StopGradientЕ
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*,
_output_shapes
:         }ђ2
moments/SquaredDifferenceЎ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indicesи
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*#
_output_shapes
:ђ*
	keep_dims(2
moments/varianceѓ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:ђ*
squeeze_dims
 2
moments/Squeezeі
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:ђ*
squeeze_dims
 2
moments/Squeeze_1s
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2
AssignMovingAvg/decayЦ
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:ђ*
dtype02 
AssignMovingAvg/ReadVariableOpЎ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:ђ2
AssignMovingAvg/subљ
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:ђ2
AssignMovingAvg/mul┐
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvgw
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2
AssignMovingAvg_1/decayФ
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:ђ*
dtype02"
 AssignMovingAvg_1/ReadVariableOpА
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:ђ2
AssignMovingAvg_1/subў
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:ђ2
AssignMovingAvg_1/mul╔
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1g
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2
batchnorm/add/yЃ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:ђ2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:ђ2
batchnorm/RsqrtЪ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
batchnorm/mul/ReadVariableOpє
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:ђ2
batchnorm/mul{
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*,
_output_shapes
:         }ђ2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:ђ2
batchnorm/mul_2Њ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
batchnorm/ReadVariableOpѓ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:ђ2
batchnorm/subі
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*,
_output_shapes
:         }ђ2
batchnorm/add_1s
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*,
_output_shapes
:         }ђ2

IdentityЫ
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         }ђ: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:T P
,
_output_shapes
:         }ђ
 
_user_specified_nameinputs
Є	
╬
3__inference_batch_normalization_layer_call_fn_28147

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityѕбStatefulPartitionedCallд
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_232822
StatefulPartitionedCallѕ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :                  @2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:                  @: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :                  @
 
_user_specified_nameinputs
є
Г
N__inference_batch_normalization_layer_call_and_return_conditional_losses_28247

inputs/
!batchnorm_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@1
#batchnorm_readvariableop_1_resource:@1
#batchnorm_readvariableop_2_resource:@
identityѕбbatchnorm/ReadVariableOpбbatchnorm/ReadVariableOp_1бbatchnorm/ReadVariableOp_2бbatchnorm/mul/ReadVariableOpњ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2
batchnorm/add/yѕ
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@2
batchnorm/Rsqrtъ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOpЁ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2
batchnorm/mulz
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:         }@2
batchnorm/mul_1ў
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp_1Ё
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:@2
batchnorm/mul_2ў
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp_2Ѓ
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2
batchnorm/subЅ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*+
_output_shapes
:         }@2
batchnorm/add_1r
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*+
_output_shapes
:         }@2

Identity┬
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         }@: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:S O
+
_output_shapes
:         }@
 
_user_specified_nameinputs
їѓ
ѕ
D__inference_basemodel_layer_call_and_return_conditional_losses_24933
inputs_0+
stream_0_conv_1_24843:@#
stream_0_conv_1_24845:@'
batch_normalization_24848:@'
batch_normalization_24850:@'
batch_normalization_24852:@'
batch_normalization_24854:@,
stream_0_conv_2_24859:@ђ$
stream_0_conv_2_24861:	ђ*
batch_normalization_1_24864:	ђ*
batch_normalization_1_24866:	ђ*
batch_normalization_1_24868:	ђ*
batch_normalization_1_24870:	ђ-
stream_0_conv_3_24875:ђђ$
stream_0_conv_3_24877:	ђ*
batch_normalization_2_24880:	ђ*
batch_normalization_2_24882:	ђ*
batch_normalization_2_24884:	ђ*
batch_normalization_2_24886:	ђ 
dense_1_24893:	ђT
dense_1_24895:T)
batch_normalization_3_24898:T)
batch_normalization_3_24900:T)
batch_normalization_3_24902:T)
batch_normalization_3_24904:T
identityѕб+batch_normalization/StatefulPartitionedCallб-batch_normalization_1/StatefulPartitionedCallб-batch_normalization_2/StatefulPartitionedCallб-batch_normalization_3/StatefulPartitionedCallбdense_1/StatefulPartitionedCallб-dense_1/kernel/Regularizer/Abs/ReadVariableOpб'stream_0_conv_1/StatefulPartitionedCallб5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpб'stream_0_conv_2/StatefulPartitionedCallб8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOpб'stream_0_conv_3/StatefulPartitionedCallб5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp§
#stream_0_input_drop/PartitionedCallPartitionedCallinputs_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         }* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *W
fRRP
N__inference_stream_0_input_drop_layer_call_and_return_conditional_losses_238822%
#stream_0_input_drop/PartitionedCallр
'stream_0_conv_1/StatefulPartitionedCallStatefulPartitionedCall,stream_0_input_drop/PartitionedCall:output:0stream_0_conv_1_24843stream_0_conv_1_24845*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         }@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *S
fNRL
J__inference_stream_0_conv_1_layer_call_and_return_conditional_losses_239052)
'stream_0_conv_1/StatefulPartitionedCall│
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall0stream_0_conv_1/StatefulPartitionedCall:output:0batch_normalization_24848batch_normalization_24850batch_normalization_24852batch_normalization_24854*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         }@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_239302-
+batch_normalization/StatefulPartitionedCallј
activation/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         }@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_activation_layer_call_and_return_conditional_losses_239452
activation/PartitionedCallї
stream_0_drop_1/PartitionedCallPartitionedCall#activation/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         }@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *S
fNRL
J__inference_stream_0_drop_1_layer_call_and_return_conditional_losses_239522!
stream_0_drop_1/PartitionedCallя
'stream_0_conv_2/StatefulPartitionedCallStatefulPartitionedCall(stream_0_drop_1/PartitionedCall:output:0stream_0_conv_2_24859stream_0_conv_2_24861*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         }ђ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *S
fNRL
J__inference_stream_0_conv_2_layer_call_and_return_conditional_losses_239752)
'stream_0_conv_2/StatefulPartitionedCall┬
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall0stream_0_conv_2/StatefulPartitionedCall:output:0batch_normalization_1_24864batch_normalization_1_24866batch_normalization_1_24868batch_normalization_1_24870*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         }ђ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_240002/
-batch_normalization_1/StatefulPartitionedCallЌ
activation_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         }ђ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *P
fKRI
G__inference_activation_1_layer_call_and_return_conditional_losses_240152
activation_1/PartitionedCallЈ
stream_0_drop_2/PartitionedCallPartitionedCall%activation_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         }ђ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *S
fNRL
J__inference_stream_0_drop_2_layer_call_and_return_conditional_losses_240222!
stream_0_drop_2/PartitionedCallя
'stream_0_conv_3/StatefulPartitionedCallStatefulPartitionedCall(stream_0_drop_2/PartitionedCall:output:0stream_0_conv_3_24875stream_0_conv_3_24877*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         }ђ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *S
fNRL
J__inference_stream_0_conv_3_layer_call_and_return_conditional_losses_240452)
'stream_0_conv_3/StatefulPartitionedCall┬
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall0stream_0_conv_3/StatefulPartitionedCall:output:0batch_normalization_2_24880batch_normalization_2_24882batch_normalization_2_24884batch_normalization_2_24886*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         }ђ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Y
fTRR
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_240702/
-batch_normalization_2/StatefulPartitionedCallЌ
activation_2/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         }ђ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *P
fKRI
G__inference_activation_2_layer_call_and_return_conditional_losses_240852
activation_2/PartitionedCallЈ
stream_0_drop_3/PartitionedCallPartitionedCall%activation_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         }ђ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *S
fNRL
J__inference_stream_0_drop_3_layer_call_and_return_conditional_losses_240922!
stream_0_drop_3/PartitionedCallЕ
(global_average_pooling1d/PartitionedCallPartitionedCall(stream_0_drop_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *\
fWRU
S__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_240992*
(global_average_pooling1d/PartitionedCallЌ
dense_1_dropout/PartitionedCallPartitionedCall1global_average_pooling1d/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *S
fNRL
J__inference_dense_1_dropout_layer_call_and_return_conditional_losses_241062!
dense_1_dropout/PartitionedCall▒
dense_1/StatefulPartitionedCallStatefulPartitionedCall(dense_1_dropout/PartitionedCall:output:0dense_1_24893dense_1_24895*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         T*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_241242!
dense_1/StatefulPartitionedCallх
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0batch_normalization_3_24898batch_normalization_3_24900batch_normalization_3_24902batch_normalization_3_24904*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         T*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Y
fTRR
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_237322/
-batch_normalization_3/StatefulPartitionedCallц
"dense_activation_1/PartitionedCallPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         T* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *V
fQRO
M__inference_dense_activation_1_layer_call_and_return_conditional_losses_241442$
"dense_activation_1/PartitionedCall╚
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpstream_0_conv_1_24843*"
_output_shapes
:@*
dtype027
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp├
&stream_0_conv_1/kernel/Regularizer/AbsAbs=stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_0_conv_1/kernel/Regularizer/AbsЕ
(stream_0_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_0_conv_1/kernel/Regularizer/ConstО
&stream_0_conv_1/kernel/Regularizer/SumSum*stream_0_conv_1/kernel/Regularizer/Abs:y:01stream_0_conv_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/SumЎ
(stream_0_conv_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2*
(stream_0_conv_1/kernel/Regularizer/mul/x▄
&stream_0_conv_1/kernel/Regularizer/mulMul1stream_0_conv_1/kernel/Regularizer/mul/x:output:0/stream_0_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/mul¤
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpstream_0_conv_2_24859*#
_output_shapes
:@ђ*
dtype02:
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOpл
)stream_0_conv_2/kernel/Regularizer/SquareSquare@stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*#
_output_shapes
:@ђ2+
)stream_0_conv_2/kernel/Regularizer/SquareЕ
(stream_0_conv_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_0_conv_2/kernel/Regularizer/Const┌
&stream_0_conv_2/kernel/Regularizer/SumSum-stream_0_conv_2/kernel/Regularizer/Square:y:01stream_0_conv_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_2/kernel/Regularizer/SumЎ
(stream_0_conv_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2*
(stream_0_conv_2/kernel/Regularizer/mul/x▄
&stream_0_conv_2/kernel/Regularizer/mulMul1stream_0_conv_2/kernel/Regularizer/mul/x:output:0/stream_0_conv_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_2/kernel/Regularizer/mul╩
5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpstream_0_conv_3_24875*$
_output_shapes
:ђђ*
dtype027
5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp┼
&stream_0_conv_3/kernel/Regularizer/AbsAbs=stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*$
_output_shapes
:ђђ2(
&stream_0_conv_3/kernel/Regularizer/AbsЕ
(stream_0_conv_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_0_conv_3/kernel/Regularizer/ConstО
&stream_0_conv_3/kernel/Regularizer/SumSum*stream_0_conv_3/kernel/Regularizer/Abs:y:01stream_0_conv_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_3/kernel/Regularizer/SumЎ
(stream_0_conv_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2*
(stream_0_conv_3/kernel/Regularizer/mul/x▄
&stream_0_conv_3/kernel/Regularizer/mulMul1stream_0_conv_3/kernel/Regularizer/mul/x:output:0/stream_0_conv_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_3/kernel/Regularizer/mulГ
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_1_24893*
_output_shapes
:	ђT*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOpе
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	ђT2 
dense_1/kernel/Regularizer/AbsЋ
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_1/kernel/Regularizer/Constи
dense_1/kernel/Regularizer/SumSum"dense_1/kernel/Regularizer/Abs:y:0)dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/SumЅ
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2"
 dense_1/kernel/Regularizer/mul/x╝
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/mulє
IdentityIdentity+dense_activation_1/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         T2

IdentityЄ
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall.^dense_1/kernel/Regularizer/Abs/ReadVariableOp(^stream_0_conv_1/StatefulPartitionedCall6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp(^stream_0_conv_2/StatefulPartitionedCall9^stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp(^stream_0_conv_3/StatefulPartitionedCall6^stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:         }: : : : : : : : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2^
-dense_1/kernel/Regularizer/Abs/ReadVariableOp-dense_1/kernel/Regularizer/Abs/ReadVariableOp2R
'stream_0_conv_1/StatefulPartitionedCall'stream_0_conv_1/StatefulPartitionedCall2n
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp2R
'stream_0_conv_2/StatefulPartitionedCall'stream_0_conv_2/StatefulPartitionedCall2t
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp2R
'stream_0_conv_3/StatefulPartitionedCall'stream_0_conv_3/StatefulPartitionedCall2n
5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp:U Q
+
_output_shapes
:         }
"
_user_specified_name
inputs_0
ы
н
5__inference_batch_normalization_1_layer_call_fn_28393

inputs
unknown:	ђ
	unknown_0:	ђ
	unknown_1:	ђ
	unknown_2:	ђ
identityѕбStatefulPartitionedCallб
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         }ђ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_240002
StatefulPartitionedCallђ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:         }ђ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         }ђ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:         }ђ
 
_user_specified_nameinputs
бІ
▄
D__inference_basemodel_layer_call_and_return_conditional_losses_24735

inputs+
stream_0_conv_1_24645:@#
stream_0_conv_1_24647:@'
batch_normalization_24650:@'
batch_normalization_24652:@'
batch_normalization_24654:@'
batch_normalization_24656:@,
stream_0_conv_2_24661:@ђ$
stream_0_conv_2_24663:	ђ*
batch_normalization_1_24666:	ђ*
batch_normalization_1_24668:	ђ*
batch_normalization_1_24670:	ђ*
batch_normalization_1_24672:	ђ-
stream_0_conv_3_24677:ђђ$
stream_0_conv_3_24679:	ђ*
batch_normalization_2_24682:	ђ*
batch_normalization_2_24684:	ђ*
batch_normalization_2_24686:	ђ*
batch_normalization_2_24688:	ђ 
dense_1_24695:	ђT
dense_1_24697:T)
batch_normalization_3_24700:T)
batch_normalization_3_24702:T)
batch_normalization_3_24704:T)
batch_normalization_3_24706:T
identityѕб+batch_normalization/StatefulPartitionedCallб-batch_normalization_1/StatefulPartitionedCallб-batch_normalization_2/StatefulPartitionedCallб-batch_normalization_3/StatefulPartitionedCallбdense_1/StatefulPartitionedCallб-dense_1/kernel/Regularizer/Abs/ReadVariableOpб'dense_1_dropout/StatefulPartitionedCallб'stream_0_conv_1/StatefulPartitionedCallб5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpб'stream_0_conv_2/StatefulPartitionedCallб8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOpб'stream_0_conv_3/StatefulPartitionedCallб5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOpб'stream_0_drop_1/StatefulPartitionedCallб'stream_0_drop_2/StatefulPartitionedCallб'stream_0_drop_3/StatefulPartitionedCallб+stream_0_input_drop/StatefulPartitionedCallЊ
+stream_0_input_drop/StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         }* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *W
fRRP
N__inference_stream_0_input_drop_layer_call_and_return_conditional_losses_245832-
+stream_0_input_drop/StatefulPartitionedCallж
'stream_0_conv_1/StatefulPartitionedCallStatefulPartitionedCall4stream_0_input_drop/StatefulPartitionedCall:output:0stream_0_conv_1_24645stream_0_conv_1_24647*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         }@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *S
fNRL
J__inference_stream_0_conv_1_layer_call_and_return_conditional_losses_239052)
'stream_0_conv_1/StatefulPartitionedCall▒
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall0stream_0_conv_1/StatefulPartitionedCall:output:0batch_normalization_24650batch_normalization_24652batch_normalization_24654batch_normalization_24656*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         }@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_245422-
+batch_normalization/StatefulPartitionedCallј
activation/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         }@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_activation_layer_call_and_return_conditional_losses_239452
activation/PartitionedCallм
'stream_0_drop_1/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0,^stream_0_input_drop/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         }@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *S
fNRL
J__inference_stream_0_drop_1_layer_call_and_return_conditional_losses_244842)
'stream_0_drop_1/StatefulPartitionedCallТ
'stream_0_conv_2/StatefulPartitionedCallStatefulPartitionedCall0stream_0_drop_1/StatefulPartitionedCall:output:0stream_0_conv_2_24661stream_0_conv_2_24663*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         }ђ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *S
fNRL
J__inference_stream_0_conv_2_layer_call_and_return_conditional_losses_239752)
'stream_0_conv_2/StatefulPartitionedCall└
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall0stream_0_conv_2/StatefulPartitionedCall:output:0batch_normalization_1_24666batch_normalization_1_24668batch_normalization_1_24670batch_normalization_1_24672*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         }ђ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_244432/
-batch_normalization_1/StatefulPartitionedCallЌ
activation_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         }ђ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *P
fKRI
G__inference_activation_1_layer_call_and_return_conditional_losses_240152
activation_1/PartitionedCallЛ
'stream_0_drop_2/StatefulPartitionedCallStatefulPartitionedCall%activation_1/PartitionedCall:output:0(^stream_0_drop_1/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         }ђ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *S
fNRL
J__inference_stream_0_drop_2_layer_call_and_return_conditional_losses_243852)
'stream_0_drop_2/StatefulPartitionedCallТ
'stream_0_conv_3/StatefulPartitionedCallStatefulPartitionedCall0stream_0_drop_2/StatefulPartitionedCall:output:0stream_0_conv_3_24677stream_0_conv_3_24679*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         }ђ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *S
fNRL
J__inference_stream_0_conv_3_layer_call_and_return_conditional_losses_240452)
'stream_0_conv_3/StatefulPartitionedCall└
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall0stream_0_conv_3/StatefulPartitionedCall:output:0batch_normalization_2_24682batch_normalization_2_24684batch_normalization_2_24686batch_normalization_2_24688*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         }ђ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Y
fTRR
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_243442/
-batch_normalization_2/StatefulPartitionedCallЌ
activation_2/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         }ђ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *P
fKRI
G__inference_activation_2_layer_call_and_return_conditional_losses_240852
activation_2/PartitionedCallЛ
'stream_0_drop_3/StatefulPartitionedCallStatefulPartitionedCall%activation_2/PartitionedCall:output:0(^stream_0_drop_2/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         }ђ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *S
fNRL
J__inference_stream_0_drop_3_layer_call_and_return_conditional_losses_242862)
'stream_0_drop_3/StatefulPartitionedCall▒
(global_average_pooling1d/PartitionedCallPartitionedCall0stream_0_drop_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *\
fWRU
S__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_240992*
(global_average_pooling1d/PartitionedCall┘
'dense_1_dropout/StatefulPartitionedCallStatefulPartitionedCall1global_average_pooling1d/PartitionedCall:output:0(^stream_0_drop_3/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *S
fNRL
J__inference_dense_1_dropout_layer_call_and_return_conditional_losses_242582)
'dense_1_dropout/StatefulPartitionedCall╣
dense_1/StatefulPartitionedCallStatefulPartitionedCall0dense_1_dropout/StatefulPartitionedCall:output:0dense_1_24695dense_1_24697*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         T*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_241242!
dense_1/StatefulPartitionedCall│
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0batch_normalization_3_24700batch_normalization_3_24702batch_normalization_3_24704batch_normalization_3_24706*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         T*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Y
fTRR
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_237922/
-batch_normalization_3/StatefulPartitionedCallц
"dense_activation_1/PartitionedCallPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         T* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *V
fQRO
M__inference_dense_activation_1_layer_call_and_return_conditional_losses_241442$
"dense_activation_1/PartitionedCall╚
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpstream_0_conv_1_24645*"
_output_shapes
:@*
dtype027
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp├
&stream_0_conv_1/kernel/Regularizer/AbsAbs=stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_0_conv_1/kernel/Regularizer/AbsЕ
(stream_0_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_0_conv_1/kernel/Regularizer/ConstО
&stream_0_conv_1/kernel/Regularizer/SumSum*stream_0_conv_1/kernel/Regularizer/Abs:y:01stream_0_conv_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/SumЎ
(stream_0_conv_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2*
(stream_0_conv_1/kernel/Regularizer/mul/x▄
&stream_0_conv_1/kernel/Regularizer/mulMul1stream_0_conv_1/kernel/Regularizer/mul/x:output:0/stream_0_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/mul¤
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpstream_0_conv_2_24661*#
_output_shapes
:@ђ*
dtype02:
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOpл
)stream_0_conv_2/kernel/Regularizer/SquareSquare@stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*#
_output_shapes
:@ђ2+
)stream_0_conv_2/kernel/Regularizer/SquareЕ
(stream_0_conv_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_0_conv_2/kernel/Regularizer/Const┌
&stream_0_conv_2/kernel/Regularizer/SumSum-stream_0_conv_2/kernel/Regularizer/Square:y:01stream_0_conv_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_2/kernel/Regularizer/SumЎ
(stream_0_conv_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2*
(stream_0_conv_2/kernel/Regularizer/mul/x▄
&stream_0_conv_2/kernel/Regularizer/mulMul1stream_0_conv_2/kernel/Regularizer/mul/x:output:0/stream_0_conv_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_2/kernel/Regularizer/mul╩
5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpstream_0_conv_3_24677*$
_output_shapes
:ђђ*
dtype027
5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp┼
&stream_0_conv_3/kernel/Regularizer/AbsAbs=stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*$
_output_shapes
:ђђ2(
&stream_0_conv_3/kernel/Regularizer/AbsЕ
(stream_0_conv_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_0_conv_3/kernel/Regularizer/ConstО
&stream_0_conv_3/kernel/Regularizer/SumSum*stream_0_conv_3/kernel/Regularizer/Abs:y:01stream_0_conv_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_3/kernel/Regularizer/SumЎ
(stream_0_conv_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2*
(stream_0_conv_3/kernel/Regularizer/mul/x▄
&stream_0_conv_3/kernel/Regularizer/mulMul1stream_0_conv_3/kernel/Regularizer/mul/x:output:0/stream_0_conv_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_3/kernel/Regularizer/mulГ
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_1_24695*
_output_shapes
:	ђT*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOpе
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	ђT2 
dense_1/kernel/Regularizer/AbsЋ
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_1/kernel/Regularizer/Constи
dense_1/kernel/Regularizer/SumSum"dense_1/kernel/Regularizer/Abs:y:0)dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/SumЅ
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2"
 dense_1/kernel/Regularizer/mul/x╝
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/mulє
IdentityIdentity+dense_activation_1/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         T2

IdentityП
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall.^dense_1/kernel/Regularizer/Abs/ReadVariableOp(^dense_1_dropout/StatefulPartitionedCall(^stream_0_conv_1/StatefulPartitionedCall6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp(^stream_0_conv_2/StatefulPartitionedCall9^stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp(^stream_0_conv_3/StatefulPartitionedCall6^stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp(^stream_0_drop_1/StatefulPartitionedCall(^stream_0_drop_2/StatefulPartitionedCall(^stream_0_drop_3/StatefulPartitionedCall,^stream_0_input_drop/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:         }: : : : : : : : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2^
-dense_1/kernel/Regularizer/Abs/ReadVariableOp-dense_1/kernel/Regularizer/Abs/ReadVariableOp2R
'dense_1_dropout/StatefulPartitionedCall'dense_1_dropout/StatefulPartitionedCall2R
'stream_0_conv_1/StatefulPartitionedCall'stream_0_conv_1/StatefulPartitionedCall2n
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp2R
'stream_0_conv_2/StatefulPartitionedCall'stream_0_conv_2/StatefulPartitionedCall2t
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp2R
'stream_0_conv_3/StatefulPartitionedCall'stream_0_conv_3/StatefulPartitionedCall2n
5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp2R
'stream_0_drop_1/StatefulPartitionedCall'stream_0_drop_1/StatefulPartitionedCall2R
'stream_0_drop_2/StatefulPartitionedCall'stream_0_drop_2/StatefulPartitionedCall2R
'stream_0_drop_3/StatefulPartitionedCall'stream_0_drop_3/StatefulPartitionedCall2Z
+stream_0_input_drop/StatefulPartitionedCall+stream_0_input_drop/StatefulPartitionedCall:S O
+
_output_shapes
:         }
 
_user_specified_nameinputs
Л
N
2__inference_dense_activation_1_layer_call_fn_28949

inputs
identity╬
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         T* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *V
fQRO
M__inference_dense_activation_1_layer_call_and_return_conditional_losses_241442
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:         T2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         T:O K
'
_output_shapes
:         T
 
_user_specified_nameinputs
й
h
/__inference_stream_0_drop_1_layer_call_fn_28301

inputs
identityѕбStatefulPartitionedCallу
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         }@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *S
fNRL
J__inference_stream_0_drop_1_layer_call_and_return_conditional_losses_244842
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         }@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         }@22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:         }@
 
_user_specified_nameinputs
џ
│
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_28480

inputs0
!batchnorm_readvariableop_resource:	ђ4
%batchnorm_mul_readvariableop_resource:	ђ2
#batchnorm_readvariableop_1_resource:	ђ2
#batchnorm_readvariableop_2_resource:	ђ
identityѕбbatchnorm/ReadVariableOpбbatchnorm/ReadVariableOp_1бbatchnorm/ReadVariableOp_2бbatchnorm/mul/ReadVariableOpЊ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2
batchnorm/add/yЅ
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:ђ2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:ђ2
batchnorm/RsqrtЪ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
batchnorm/mul/ReadVariableOpє
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:ђ2
batchnorm/mul{
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*,
_output_shapes
:         }ђ2
batchnorm/mul_1Ў
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype02
batchnorm/ReadVariableOp_1є
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:ђ2
batchnorm/mul_2Ў
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:ђ*
dtype02
batchnorm/ReadVariableOp_2ё
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:ђ2
batchnorm/subі
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*,
_output_shapes
:         }ђ2
batchnorm/add_1s
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*,
_output_shapes
:         }ђ2

Identity┬
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         }ђ: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:T P
,
_output_shapes
:         }ђ
 
_user_specified_nameinputs
М+
ь
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_28460

inputs6
'assignmovingavg_readvariableop_resource:	ђ8
)assignmovingavg_1_readvariableop_resource:	ђ4
%batchnorm_mul_readvariableop_resource:	ђ0
!batchnorm_readvariableop_resource:	ђ
identityѕбAssignMovingAvgбAssignMovingAvg/ReadVariableOpбAssignMovingAvg_1б AssignMovingAvg_1/ReadVariableOpбbatchnorm/ReadVariableOpбbatchnorm/mul/ReadVariableOpЉ
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indicesћ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*#
_output_shapes
:ђ*
	keep_dims(2
moments/meanЂ
moments/StopGradientStopGradientmoments/mean:output:0*
T0*#
_output_shapes
:ђ2
moments/StopGradient▓
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*5
_output_shapes#
!:                  ђ2
moments/SquaredDifferenceЎ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indicesи
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*#
_output_shapes
:ђ*
	keep_dims(2
moments/varianceѓ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:ђ*
squeeze_dims
 2
moments/Squeezeі
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:ђ*
squeeze_dims
 2
moments/Squeeze_1s
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2
AssignMovingAvg/decayЦ
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:ђ*
dtype02 
AssignMovingAvg/ReadVariableOpЎ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:ђ2
AssignMovingAvg/subљ
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:ђ2
AssignMovingAvg/mul┐
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvgw
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2
AssignMovingAvg_1/decayФ
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:ђ*
dtype02"
 AssignMovingAvg_1/ReadVariableOpА
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:ђ2
AssignMovingAvg_1/subў
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:ђ2
AssignMovingAvg_1/mul╔
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1g
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2
batchnorm/add/yЃ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:ђ2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:ђ2
batchnorm/RsqrtЪ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
batchnorm/mul/ReadVariableOpє
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:ђ2
batchnorm/mulё
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*5
_output_shapes#
!:                  ђ2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:ђ2
batchnorm/mul_2Њ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
batchnorm/ReadVariableOpѓ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:ђ2
batchnorm/subЊ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*5
_output_shapes#
!:                  ђ2
batchnorm/add_1|
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*5
_output_shapes#
!:                  ђ2

IdentityЫ
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):                  ђ: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:] Y
5
_output_shapes#
!:                  ђ
 
_user_specified_nameinputs
Є
h
J__inference_stream_0_drop_1_layer_call_and_return_conditional_losses_28306

inputs

identity_1^
IdentityIdentityinputs*
T0*+
_output_shapes
:         }@2

Identitym

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:         }@2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         }@:S O
+
_output_shapes
:         }@
 
_user_specified_nameinputs"еL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*ѓ
serving_defaultЬ
G
left_inputs8
serving_default_left_inputs:0         }
I
right_inputs9
serving_default_right_inputs:0         }<
distance0
StatefulPartitionedCall:0         tensorflow/serving/predict:ЉА
┤
layer-0
layer-1
layer_with_weights-0
layer-2
layer-3
	optimizer
regularization_losses
trainable_variables
	variables
		keras_api


signatures
Ю__call__
+ъ&call_and_return_all_conditional_losses
Ъ_default_save_signature"
_tf_keras_network
"
_tf_keras_input_layer
"
_tf_keras_input_layer
щ
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer-4
layer-5
layer_with_weights-2
layer-6
layer_with_weights-3
layer-7
layer-8
layer-9
layer_with_weights-4
layer-10
layer_with_weights-5
layer-11
layer-12
layer-13
layer-14
layer-15
layer_with_weights-6
layer-16
layer_with_weights-7
layer-17
layer-18
regularization_losses
trainable_variables
 	variables
!	keras_api
а__call__
+А&call_and_return_all_conditional_losses"
_tf_keras_network
Д
"regularization_losses
#trainable_variables
$	variables
%	keras_api
б__call__
+Б&call_and_return_all_conditional_losses"
_tf_keras_layer
Њ

&beta_1

'beta_2
	(decay
)learning_rate
*iter+m§,m■-m .mђ/mЂ0mѓ1mЃ2mё3mЁ4mє5mЄ6mѕ7mЅ8mі9mІ:mї+vЇ,vј-vЈ.vљ/vЉ0vњ1vЊ2vћ3vЋ4vќ5vЌ6vў7vЎ8vџ9vЏ:vю"
	optimizer
 "
trackable_list_wrapper
ќ
+0
,1
-2
.3
/4
05
16
27
38
49
510
611
712
813
914
:15"
trackable_list_wrapper
о
+0
,1
-2
.3
;4
<5
/6
07
18
29
=10
>11
312
413
514
615
?16
@17
718
819
920
:21
A22
B23"
trackable_list_wrapper
╬
Clayer_metrics
regularization_losses
Dlayer_regularization_losses
trainable_variables
	variables
Enon_trainable_variables

Flayers
Gmetrics
Ю__call__
Ъ_default_save_signature
+ъ&call_and_return_all_conditional_losses
'ъ"call_and_return_conditional_losses"
_generic_user_object
-
цserving_default"
signature_map
"
_tf_keras_input_layer
Д
Hregularization_losses
Itrainable_variables
J	variables
K	keras_api
Ц__call__
+д&call_and_return_all_conditional_losses"
_tf_keras_layer
й

+kernel
,bias
Lregularization_losses
Mtrainable_variables
N	variables
O	keras_api
Д__call__
+е&call_and_return_all_conditional_losses"
_tf_keras_layer
В
Paxis
	-gamma
.beta
;moving_mean
<moving_variance
Qregularization_losses
Rtrainable_variables
S	variables
T	keras_api
Е__call__
+ф&call_and_return_all_conditional_losses"
_tf_keras_layer
Д
Uregularization_losses
Vtrainable_variables
W	variables
X	keras_api
Ф__call__
+г&call_and_return_all_conditional_losses"
_tf_keras_layer
Д
Yregularization_losses
Ztrainable_variables
[	variables
\	keras_api
Г__call__
+«&call_and_return_all_conditional_losses"
_tf_keras_layer
й

/kernel
0bias
]regularization_losses
^trainable_variables
_	variables
`	keras_api
»__call__
+░&call_and_return_all_conditional_losses"
_tf_keras_layer
В
aaxis
	1gamma
2beta
=moving_mean
>moving_variance
bregularization_losses
ctrainable_variables
d	variables
e	keras_api
▒__call__
+▓&call_and_return_all_conditional_losses"
_tf_keras_layer
Д
fregularization_losses
gtrainable_variables
h	variables
i	keras_api
│__call__
+┤&call_and_return_all_conditional_losses"
_tf_keras_layer
Д
jregularization_losses
ktrainable_variables
l	variables
m	keras_api
х__call__
+Х&call_and_return_all_conditional_losses"
_tf_keras_layer
й

3kernel
4bias
nregularization_losses
otrainable_variables
p	variables
q	keras_api
и__call__
+И&call_and_return_all_conditional_losses"
_tf_keras_layer
В
raxis
	5gamma
6beta
?moving_mean
@moving_variance
sregularization_losses
ttrainable_variables
u	variables
v	keras_api
╣__call__
+║&call_and_return_all_conditional_losses"
_tf_keras_layer
Д
wregularization_losses
xtrainable_variables
y	variables
z	keras_api
╗__call__
+╝&call_and_return_all_conditional_losses"
_tf_keras_layer
Д
{regularization_losses
|trainable_variables
}	variables
~	keras_api
й__call__
+Й&call_and_return_all_conditional_losses"
_tf_keras_layer
ф
regularization_losses
ђtrainable_variables
Ђ	variables
ѓ	keras_api
┐__call__
+└&call_and_return_all_conditional_losses"
_tf_keras_layer
Ф
Ѓregularization_losses
ёtrainable_variables
Ё	variables
є	keras_api
┴__call__
+┬&call_and_return_all_conditional_losses"
_tf_keras_layer
┴

7kernel
8bias
Єregularization_losses
ѕtrainable_variables
Ѕ	variables
і	keras_api
├__call__
+─&call_and_return_all_conditional_losses"
_tf_keras_layer
ы
	Іaxis
	9gamma
:beta
Amoving_mean
Bmoving_variance
їregularization_losses
Їtrainable_variables
ј	variables
Ј	keras_api
┼__call__
+к&call_and_return_all_conditional_losses"
_tf_keras_layer
Ф
љregularization_losses
Љtrainable_variables
њ	variables
Њ	keras_api
К__call__
+╚&call_and_return_all_conditional_losses"
_tf_keras_layer
@
╔0
╩1
╦2
╠3"
trackable_list_wrapper
ќ
+0
,1
-2
.3
/4
05
16
27
38
49
510
611
712
813
914
:15"
trackable_list_wrapper
о
+0
,1
-2
.3
;4
<5
/6
07
18
29
=10
>11
312
413
514
615
?16
@17
718
819
920
:21
A22
B23"
trackable_list_wrapper
х
ћlayer_metrics
regularization_losses
 Ћlayer_regularization_losses
trainable_variables
 	variables
ќnon_trainable_variables
Ќlayers
ўmetrics
а__call__
+А&call_and_return_all_conditional_losses
'А"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
х
Ўmetrics
џlayer_metrics
"regularization_losses
#trainable_variables
$	variables
Џnon_trainable_variables
юlayers
 Юlayer_regularization_losses
б__call__
+Б&call_and_return_all_conditional_losses
'Б"call_and_return_conditional_losses"
_generic_user_object
: (2beta_1
: (2beta_2
: (2decay
: (2learning_rate
:	 (2	Adam/iter
,:*@2stream_0_conv_1/kernel
": @2stream_0_conv_1/bias
':%@2batch_normalization/gamma
&:$@2batch_normalization/beta
-:+@ђ2stream_0_conv_2/kernel
#:!ђ2stream_0_conv_2/bias
*:(ђ2batch_normalization_1/gamma
):'ђ2batch_normalization_1/beta
.:,ђђ2stream_0_conv_3/kernel
#:!ђ2stream_0_conv_3/bias
*:(ђ2batch_normalization_2/gamma
):'ђ2batch_normalization_2/beta
!:	ђT2dense_1/kernel
:T2dense_1/bias
):'T2batch_normalization_3/gamma
(:&T2batch_normalization_3/beta
/:-@ (2batch_normalization/moving_mean
3:1@ (2#batch_normalization/moving_variance
2:0ђ (2!batch_normalization_1/moving_mean
6:4ђ (2%batch_normalization_1/moving_variance
2:0ђ (2!batch_normalization_2/moving_mean
6:4ђ (2%batch_normalization_2/moving_variance
1:/T (2!batch_normalization_3/moving_mean
5:3T (2%batch_normalization_3/moving_variance
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
X
;0
<1
=2
>3
?4
@5
A6
B7"
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
(
ъ0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
х
Ъmetrics
аlayer_metrics
Hregularization_losses
Itrainable_variables
J	variables
Аnon_trainable_variables
бlayers
 Бlayer_regularization_losses
Ц__call__
+д&call_and_return_all_conditional_losses
'д"call_and_return_conditional_losses"
_generic_user_object
(
╔0"
trackable_list_wrapper
.
+0
,1"
trackable_list_wrapper
.
+0
,1"
trackable_list_wrapper
х
цmetrics
Цlayer_metrics
Lregularization_losses
Mtrainable_variables
N	variables
дnon_trainable_variables
Дlayers
 еlayer_regularization_losses
Д__call__
+е&call_and_return_all_conditional_losses
'е"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
-0
.1"
trackable_list_wrapper
<
-0
.1
;2
<3"
trackable_list_wrapper
х
Еmetrics
фlayer_metrics
Qregularization_losses
Rtrainable_variables
S	variables
Фnon_trainable_variables
гlayers
 Гlayer_regularization_losses
Е__call__
+ф&call_and_return_all_conditional_losses
'ф"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
х
«metrics
»layer_metrics
Uregularization_losses
Vtrainable_variables
W	variables
░non_trainable_variables
▒layers
 ▓layer_regularization_losses
Ф__call__
+г&call_and_return_all_conditional_losses
'г"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
х
│metrics
┤layer_metrics
Yregularization_losses
Ztrainable_variables
[	variables
хnon_trainable_variables
Хlayers
 иlayer_regularization_losses
Г__call__
+«&call_and_return_all_conditional_losses
'«"call_and_return_conditional_losses"
_generic_user_object
(
╩0"
trackable_list_wrapper
.
/0
01"
trackable_list_wrapper
.
/0
01"
trackable_list_wrapper
х
Иmetrics
╣layer_metrics
]regularization_losses
^trainable_variables
_	variables
║non_trainable_variables
╗layers
 ╝layer_regularization_losses
»__call__
+░&call_and_return_all_conditional_losses
'░"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
10
21"
trackable_list_wrapper
<
10
21
=2
>3"
trackable_list_wrapper
х
йmetrics
Йlayer_metrics
bregularization_losses
ctrainable_variables
d	variables
┐non_trainable_variables
└layers
 ┴layer_regularization_losses
▒__call__
+▓&call_and_return_all_conditional_losses
'▓"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
х
┬metrics
├layer_metrics
fregularization_losses
gtrainable_variables
h	variables
─non_trainable_variables
┼layers
 кlayer_regularization_losses
│__call__
+┤&call_and_return_all_conditional_losses
'┤"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
х
Кmetrics
╚layer_metrics
jregularization_losses
ktrainable_variables
l	variables
╔non_trainable_variables
╩layers
 ╦layer_regularization_losses
х__call__
+Х&call_and_return_all_conditional_losses
'Х"call_and_return_conditional_losses"
_generic_user_object
(
╦0"
trackable_list_wrapper
.
30
41"
trackable_list_wrapper
.
30
41"
trackable_list_wrapper
х
╠metrics
═layer_metrics
nregularization_losses
otrainable_variables
p	variables
╬non_trainable_variables
¤layers
 лlayer_regularization_losses
и__call__
+И&call_and_return_all_conditional_losses
'И"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
50
61"
trackable_list_wrapper
<
50
61
?2
@3"
trackable_list_wrapper
х
Лmetrics
мlayer_metrics
sregularization_losses
ttrainable_variables
u	variables
Мnon_trainable_variables
нlayers
 Нlayer_regularization_losses
╣__call__
+║&call_and_return_all_conditional_losses
'║"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
х
оmetrics
Оlayer_metrics
wregularization_losses
xtrainable_variables
y	variables
пnon_trainable_variables
┘layers
 ┌layer_regularization_losses
╗__call__
+╝&call_and_return_all_conditional_losses
'╝"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
х
█metrics
▄layer_metrics
{regularization_losses
|trainable_variables
}	variables
Пnon_trainable_variables
яlayers
 ▀layer_regularization_losses
й__call__
+Й&call_and_return_all_conditional_losses
'Й"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
и
Яmetrics
рlayer_metrics
regularization_losses
ђtrainable_variables
Ђ	variables
Рnon_trainable_variables
сlayers
 Сlayer_regularization_losses
┐__call__
+└&call_and_return_all_conditional_losses
'└"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
тmetrics
Тlayer_metrics
Ѓregularization_losses
ёtrainable_variables
Ё	variables
уnon_trainable_variables
Уlayers
 жlayer_regularization_losses
┴__call__
+┬&call_and_return_all_conditional_losses
'┬"call_and_return_conditional_losses"
_generic_user_object
(
╠0"
trackable_list_wrapper
.
70
81"
trackable_list_wrapper
.
70
81"
trackable_list_wrapper
И
Жmetrics
вlayer_metrics
Єregularization_losses
ѕtrainable_variables
Ѕ	variables
Вnon_trainable_variables
ьlayers
 Ьlayer_regularization_losses
├__call__
+─&call_and_return_all_conditional_losses
'─"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
90
:1"
trackable_list_wrapper
<
90
:1
A2
B3"
trackable_list_wrapper
И
№metrics
­layer_metrics
їregularization_losses
Їtrainable_variables
ј	variables
ыnon_trainable_variables
Ыlayers
 зlayer_regularization_losses
┼__call__
+к&call_and_return_all_conditional_losses
'к"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
Зmetrics
шlayer_metrics
љregularization_losses
Љtrainable_variables
њ	variables
Шnon_trainable_variables
эlayers
 Эlayer_regularization_losses
К__call__
+╚&call_and_return_all_conditional_losses
'╚"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
X
;0
<1
=2
>3
?4
@5
A6
B7"
trackable_list_wrapper
«
0
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
R

щtotal

Щcount
ч	variables
Ч	keras_api"
_tf_keras_metric
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
╔0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
;0
<1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
╩0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
=0
>1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
╦0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
?0
@1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
╠0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
A0
B1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
:  (2total
:  (2count
0
щ0
Щ1"
trackable_list_wrapper
.
ч	variables"
_generic_user_object
1:/@2Adam/stream_0_conv_1/kernel/m
':%@2Adam/stream_0_conv_1/bias/m
,:*@2 Adam/batch_normalization/gamma/m
+:)@2Adam/batch_normalization/beta/m
2:0@ђ2Adam/stream_0_conv_2/kernel/m
(:&ђ2Adam/stream_0_conv_2/bias/m
/:-ђ2"Adam/batch_normalization_1/gamma/m
.:,ђ2!Adam/batch_normalization_1/beta/m
3:1ђђ2Adam/stream_0_conv_3/kernel/m
(:&ђ2Adam/stream_0_conv_3/bias/m
/:-ђ2"Adam/batch_normalization_2/gamma/m
.:,ђ2!Adam/batch_normalization_2/beta/m
&:$	ђT2Adam/dense_1/kernel/m
:T2Adam/dense_1/bias/m
.:,T2"Adam/batch_normalization_3/gamma/m
-:+T2!Adam/batch_normalization_3/beta/m
1:/@2Adam/stream_0_conv_1/kernel/v
':%@2Adam/stream_0_conv_1/bias/v
,:*@2 Adam/batch_normalization/gamma/v
+:)@2Adam/batch_normalization/beta/v
2:0@ђ2Adam/stream_0_conv_2/kernel/v
(:&ђ2Adam/stream_0_conv_2/bias/v
/:-ђ2"Adam/batch_normalization_1/gamma/v
.:,ђ2!Adam/batch_normalization_1/beta/v
3:1ђђ2Adam/stream_0_conv_3/kernel/v
(:&ђ2Adam/stream_0_conv_3/bias/v
/:-ђ2"Adam/batch_normalization_2/gamma/v
.:,ђ2!Adam/batch_normalization_2/beta/v
&:$	ђT2Adam/dense_1/kernel/v
:T2Adam/dense_1/bias/v
.:,T2"Adam/batch_normalization_3/gamma/v
-:+T2!Adam/batch_normalization_3/beta/v
Р2▀
%__inference_model_layer_call_fn_25342
%__inference_model_layer_call_fn_26320
%__inference_model_layer_call_fn_26374
%__inference_model_layer_call_fn_25972└
и▓│
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
╬2╦
@__inference_model_layer_call_and_return_conditional_losses_26614
@__inference_model_layer_call_and_return_conditional_losses_27036
@__inference_model_layer_call_and_return_conditional_losses_26076
@__inference_model_layer_call_and_return_conditional_losses_26180└
и▓│
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
ПB┌
 __inference__wrapped_model_23198left_inputsright_inputs"ў
Љ▓Ї
FullArgSpec
argsџ 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
╚2┼
)__inference_basemodel_layer_call_fn_24222
)__inference_basemodel_layer_call_fn_27113
)__inference_basemodel_layer_call_fn_27166
)__inference_basemodel_layer_call_fn_24839
)__inference_basemodel_layer_call_fn_27219
)__inference_basemodel_layer_call_fn_27272└
и▓│
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
Ж2у
D__inference_basemodel_layer_call_and_return_conditional_losses_27414
D__inference_basemodel_layer_call_and_return_conditional_losses_27647
D__inference_basemodel_layer_call_and_return_conditional_losses_24933
D__inference_basemodel_layer_call_and_return_conditional_losses_25027
D__inference_basemodel_layer_call_and_return_conditional_losses_27789
D__inference_basemodel_layer_call_and_return_conditional_losses_28022└
и▓│
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
џ2Ќ
(__inference_distance_layer_call_fn_28028
(__inference_distance_layer_call_fn_28034└
и▓│
FullArgSpec1
args)џ&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaultsџ

 
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
л2═
C__inference_distance_layer_call_and_return_conditional_losses_28046
C__inference_distance_layer_call_and_return_conditional_losses_28058└
и▓│
FullArgSpec1
args)џ&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaultsџ

 
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
┌BО
#__inference_signature_wrapper_26266left_inputsright_inputs"ћ
Ї▓Ѕ
FullArgSpec
argsџ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ц2А
3__inference_stream_0_input_drop_layer_call_fn_28063
3__inference_stream_0_input_drop_layer_call_fn_28068┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
┌2О
N__inference_stream_0_input_drop_layer_call_and_return_conditional_losses_28073
N__inference_stream_0_input_drop_layer_call_and_return_conditional_losses_28085┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
┘2о
/__inference_stream_0_conv_1_layer_call_fn_28100б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
З2ы
J__inference_stream_0_conv_1_layer_call_and_return_conditional_losses_28121б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ј2І
3__inference_batch_normalization_layer_call_fn_28134
3__inference_batch_normalization_layer_call_fn_28147
3__inference_batch_normalization_layer_call_fn_28160
3__inference_batch_normalization_layer_call_fn_28173┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
Щ2э
N__inference_batch_normalization_layer_call_and_return_conditional_losses_28193
N__inference_batch_normalization_layer_call_and_return_conditional_losses_28227
N__inference_batch_normalization_layer_call_and_return_conditional_losses_28247
N__inference_batch_normalization_layer_call_and_return_conditional_losses_28281┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
н2Л
*__inference_activation_layer_call_fn_28286б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
№2В
E__inference_activation_layer_call_and_return_conditional_losses_28291б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ю2Ў
/__inference_stream_0_drop_1_layer_call_fn_28296
/__inference_stream_0_drop_1_layer_call_fn_28301┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
м2¤
J__inference_stream_0_drop_1_layer_call_and_return_conditional_losses_28306
J__inference_stream_0_drop_1_layer_call_and_return_conditional_losses_28318┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
┘2о
/__inference_stream_0_conv_2_layer_call_fn_28333б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
З2ы
J__inference_stream_0_conv_2_layer_call_and_return_conditional_losses_28354б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ќ2Њ
5__inference_batch_normalization_1_layer_call_fn_28367
5__inference_batch_normalization_1_layer_call_fn_28380
5__inference_batch_normalization_1_layer_call_fn_28393
5__inference_batch_normalization_1_layer_call_fn_28406┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
ѓ2 
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_28426
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_28460
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_28480
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_28514┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
о2М
,__inference_activation_1_layer_call_fn_28519б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ы2Ь
G__inference_activation_1_layer_call_and_return_conditional_losses_28524б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ю2Ў
/__inference_stream_0_drop_2_layer_call_fn_28529
/__inference_stream_0_drop_2_layer_call_fn_28534┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
м2¤
J__inference_stream_0_drop_2_layer_call_and_return_conditional_losses_28539
J__inference_stream_0_drop_2_layer_call_and_return_conditional_losses_28551┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
┘2о
/__inference_stream_0_conv_3_layer_call_fn_28566б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
З2ы
J__inference_stream_0_conv_3_layer_call_and_return_conditional_losses_28587б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ќ2Њ
5__inference_batch_normalization_2_layer_call_fn_28600
5__inference_batch_normalization_2_layer_call_fn_28613
5__inference_batch_normalization_2_layer_call_fn_28626
5__inference_batch_normalization_2_layer_call_fn_28639┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
ѓ2 
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_28659
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_28693
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_28713
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_28747┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
о2М
,__inference_activation_2_layer_call_fn_28752б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ы2Ь
G__inference_activation_2_layer_call_and_return_conditional_losses_28757б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ю2Ў
/__inference_stream_0_drop_3_layer_call_fn_28762
/__inference_stream_0_drop_3_layer_call_fn_28767┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
м2¤
J__inference_stream_0_drop_3_layer_call_and_return_conditional_losses_28772
J__inference_stream_0_drop_3_layer_call_and_return_conditional_losses_28784┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
Е2д
8__inference_global_average_pooling1d_layer_call_fn_28789
8__inference_global_average_pooling1d_layer_call_fn_28794»
д▓б
FullArgSpec%
argsџ
jself
jinputs
jmask
varargs
 
varkw
 
defaultsб

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
▀2▄
S__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_28800
S__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_28806»
д▓б
FullArgSpec%
argsџ
jself
jinputs
jmask
varargs
 
varkw
 
defaultsб

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ю2Ў
/__inference_dense_1_dropout_layer_call_fn_28811
/__inference_dense_1_dropout_layer_call_fn_28816┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
м2¤
J__inference_dense_1_dropout_layer_call_and_return_conditional_losses_28821
J__inference_dense_1_dropout_layer_call_and_return_conditional_losses_28833┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
Л2╬
'__inference_dense_1_layer_call_fn_28848б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
В2ж
B__inference_dense_1_layer_call_and_return_conditional_losses_28864б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
е2Ц
5__inference_batch_normalization_3_layer_call_fn_28877
5__inference_batch_normalization_3_layer_call_fn_28890┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
я2█
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_28910
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_28944┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
▄2┘
2__inference_dense_activation_1_layer_call_fn_28949б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
э2З
M__inference_dense_activation_1_layer_call_and_return_conditional_losses_28954б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
▓2»
__inference_loss_fn_0_28965Ј
Є▓Ѓ
FullArgSpec
argsџ 
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *б 
▓2»
__inference_loss_fn_1_28976Ј
Є▓Ѓ
FullArgSpec
argsџ 
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *б 
▓2»
__inference_loss_fn_2_28987Ј
Є▓Ѓ
FullArgSpec
argsџ 
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *б 
▓2»
__inference_loss_fn_3_28998Ј
Є▓Ѓ
FullArgSpec
argsџ 
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *б ▀
 __inference__wrapped_model_23198║+,<-;./0>1=234@5?678B9A:iбf
_б\
ZџW
)і&
left_inputs         }
*і'
right_inputs         }
ф "3ф0
.
distance"і
distance         Г
G__inference_activation_1_layer_call_and_return_conditional_losses_28524b4б1
*б'
%і"
inputs         }ђ
ф "*б'
 і
0         }ђ
џ Ё
,__inference_activation_1_layer_call_fn_28519U4б1
*б'
%і"
inputs         }ђ
ф "і         }ђГ
G__inference_activation_2_layer_call_and_return_conditional_losses_28757b4б1
*б'
%і"
inputs         }ђ
ф "*б'
 і
0         }ђ
џ Ё
,__inference_activation_2_layer_call_fn_28752U4б1
*б'
%і"
inputs         }ђ
ф "і         }ђЕ
E__inference_activation_layer_call_and_return_conditional_losses_28291`3б0
)б&
$і!
inputs         }@
ф ")б&
і
0         }@
џ Ђ
*__inference_activation_layer_call_fn_28286S3б0
)б&
$і!
inputs         }@
ф "і         }@╔
D__inference_basemodel_layer_call_and_return_conditional_losses_24933ђ+,<-;./0>1=234@5?678B9A:=б:
3б0
&і#
inputs_0         }
p 

 
ф "%б"
і
0         T
џ ╔
D__inference_basemodel_layer_call_and_return_conditional_losses_25027ђ+,;<-./0=>1234?@5678AB9:=б:
3б0
&і#
inputs_0         }
p

 
ф "%б"
і
0         T
џ к
D__inference_basemodel_layer_call_and_return_conditional_losses_27414~+,<-;./0>1=234@5?678B9A:;б8
1б.
$і!
inputs         }
p 

 
ф "%б"
і
0         T
џ к
D__inference_basemodel_layer_call_and_return_conditional_losses_27647~+,;<-./0=>1234?@5678AB9:;б8
1б.
$і!
inputs         }
p

 
ф "%б"
і
0         T
џ ╬
D__inference_basemodel_layer_call_and_return_conditional_losses_27789Ё+,<-;./0>1=234@5?678B9A:Bб?
8б5
+џ(
&і#
inputs/0         }
p 

 
ф "%б"
і
0         T
џ ╬
D__inference_basemodel_layer_call_and_return_conditional_losses_28022Ё+,;<-./0=>1234?@5678AB9:Bб?
8б5
+џ(
&і#
inputs/0         }
p

 
ф "%б"
і
0         T
џ а
)__inference_basemodel_layer_call_fn_24222s+,<-;./0>1=234@5?678B9A:=б:
3б0
&і#
inputs_0         }
p 

 
ф "і         Tа
)__inference_basemodel_layer_call_fn_24839s+,;<-./0=>1234?@5678AB9:=б:
3б0
&і#
inputs_0         }
p

 
ф "і         Tъ
)__inference_basemodel_layer_call_fn_27113q+,<-;./0>1=234@5?678B9A:;б8
1б.
$і!
inputs         }
p 

 
ф "і         Tъ
)__inference_basemodel_layer_call_fn_27166q+,;<-./0=>1234?@5678AB9:;б8
1б.
$і!
inputs         }
p

 
ф "і         TЦ
)__inference_basemodel_layer_call_fn_27219x+,<-;./0>1=234@5?678B9A:Bб?
8б5
+џ(
&і#
inputs/0         }
p 

 
ф "і         TЦ
)__inference_basemodel_layer_call_fn_27272x+,;<-./0=>1234?@5678AB9:Bб?
8б5
+џ(
&і#
inputs/0         }
p

 
ф "і         Tм
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_28426~>1=2Aб>
7б4
.і+
inputs                  ђ
p 
ф "3б0
)і&
0                  ђ
џ м
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_28460~=>12Aб>
7б4
.і+
inputs                  ђ
p
ф "3б0
)і&
0                  ђ
џ └
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_28480l>1=28б5
.б+
%і"
inputs         }ђ
p 
ф "*б'
 і
0         }ђ
џ └
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_28514l=>128б5
.б+
%і"
inputs         }ђ
p
ф "*б'
 і
0         }ђ
џ ф
5__inference_batch_normalization_1_layer_call_fn_28367q>1=2Aб>
7б4
.і+
inputs                  ђ
p 
ф "&і#                  ђф
5__inference_batch_normalization_1_layer_call_fn_28380q=>12Aб>
7б4
.і+
inputs                  ђ
p
ф "&і#                  ђў
5__inference_batch_normalization_1_layer_call_fn_28393_>1=28б5
.б+
%і"
inputs         }ђ
p 
ф "і         }ђў
5__inference_batch_normalization_1_layer_call_fn_28406_=>128б5
.б+
%і"
inputs         }ђ
p
ф "і         }ђм
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_28659~@5?6Aб>
7б4
.і+
inputs                  ђ
p 
ф "3б0
)і&
0                  ђ
џ м
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_28693~?@56Aб>
7б4
.і+
inputs                  ђ
p
ф "3б0
)і&
0                  ђ
џ └
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_28713l@5?68б5
.б+
%і"
inputs         }ђ
p 
ф "*б'
 і
0         }ђ
џ └
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_28747l?@568б5
.б+
%і"
inputs         }ђ
p
ф "*б'
 і
0         }ђ
џ ф
5__inference_batch_normalization_2_layer_call_fn_28600q@5?6Aб>
7б4
.і+
inputs                  ђ
p 
ф "&і#                  ђф
5__inference_batch_normalization_2_layer_call_fn_28613q?@56Aб>
7б4
.і+
inputs                  ђ
p
ф "&і#                  ђў
5__inference_batch_normalization_2_layer_call_fn_28626_@5?68б5
.б+
%і"
inputs         }ђ
p 
ф "і         }ђў
5__inference_batch_normalization_2_layer_call_fn_28639_?@568б5
.б+
%і"
inputs         }ђ
p
ф "і         }ђХ
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_28910bB9A:3б0
)б&
 і
inputs         T
p 
ф "%б"
і
0         T
џ Х
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_28944bAB9:3б0
)б&
 і
inputs         T
p
ф "%б"
і
0         T
џ ј
5__inference_batch_normalization_3_layer_call_fn_28877UB9A:3б0
)б&
 і
inputs         T
p 
ф "і         Tј
5__inference_batch_normalization_3_layer_call_fn_28890UAB9:3б0
)б&
 і
inputs         T
p
ф "і         T╬
N__inference_batch_normalization_layer_call_and_return_conditional_losses_28193|<-;.@б=
6б3
-і*
inputs                  @
p 
ф "2б/
(і%
0                  @
џ ╬
N__inference_batch_normalization_layer_call_and_return_conditional_losses_28227|;<-.@б=
6б3
-і*
inputs                  @
p
ф "2б/
(і%
0                  @
џ ╝
N__inference_batch_normalization_layer_call_and_return_conditional_losses_28247j<-;.7б4
-б*
$і!
inputs         }@
p 
ф ")б&
і
0         }@
џ ╝
N__inference_batch_normalization_layer_call_and_return_conditional_losses_28281j;<-.7б4
-б*
$і!
inputs         }@
p
ф ")б&
і
0         }@
џ д
3__inference_batch_normalization_layer_call_fn_28134o<-;.@б=
6б3
-і*
inputs                  @
p 
ф "%і"                  @д
3__inference_batch_normalization_layer_call_fn_28147o;<-.@б=
6б3
-і*
inputs                  @
p
ф "%і"                  @ћ
3__inference_batch_normalization_layer_call_fn_28160]<-;.7б4
-б*
$і!
inputs         }@
p 
ф "і         }@ћ
3__inference_batch_normalization_layer_call_fn_28173];<-.7б4
-б*
$і!
inputs         }@
p
ф "і         }@г
J__inference_dense_1_dropout_layer_call_and_return_conditional_losses_28821^4б1
*б'
!і
inputs         ђ
p 
ф "&б#
і
0         ђ
џ г
J__inference_dense_1_dropout_layer_call_and_return_conditional_losses_28833^4б1
*б'
!і
inputs         ђ
p
ф "&б#
і
0         ђ
џ ё
/__inference_dense_1_dropout_layer_call_fn_28811Q4б1
*б'
!і
inputs         ђ
p 
ф "і         ђё
/__inference_dense_1_dropout_layer_call_fn_28816Q4б1
*б'
!і
inputs         ђ
p
ф "і         ђБ
B__inference_dense_1_layer_call_and_return_conditional_losses_28864]780б-
&б#
!і
inputs         ђ
ф "%б"
і
0         T
џ {
'__inference_dense_1_layer_call_fn_28848P780б-
&б#
!і
inputs         ђ
ф "і         TЕ
M__inference_dense_activation_1_layer_call_and_return_conditional_losses_28954X/б,
%б"
 і
inputs         T
ф "%б"
і
0         T
џ Ђ
2__inference_dense_activation_1_layer_call_fn_28949K/б,
%б"
 і
inputs         T
ф "і         TМ
C__inference_distance_layer_call_and_return_conditional_losses_28046Іbб_
XбU
KџH
"і
inputs/0         T
"і
inputs/1         T

 
p 
ф "%б"
і
0         
џ М
C__inference_distance_layer_call_and_return_conditional_losses_28058Іbб_
XбU
KџH
"і
inputs/0         T
"і
inputs/1         T

 
p
ф "%б"
і
0         
џ ф
(__inference_distance_layer_call_fn_28028~bб_
XбU
KџH
"і
inputs/0         T
"і
inputs/1         T

 
p 
ф "і         ф
(__inference_distance_layer_call_fn_28034~bб_
XбU
KџH
"і
inputs/0         T
"і
inputs/1         T

 
p
ф "і         м
S__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_28800{IбF
?б<
6і3
inputs'                           

 
ф ".б+
$і!
0                  
џ ╣
S__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_28806b8б5
.б+
%і"
inputs         }ђ

 
ф "&б#
і
0         ђ
џ ф
8__inference_global_average_pooling1d_layer_call_fn_28789nIбF
?б<
6і3
inputs'                           

 
ф "!і                  Љ
8__inference_global_average_pooling1d_layer_call_fn_28794U8б5
.б+
%і"
inputs         }ђ

 
ф "і         ђ:
__inference_loss_fn_0_28965+б

б 
ф "і :
__inference_loss_fn_1_28976/б

б 
ф "і :
__inference_loss_fn_2_289873б

б 
ф "і :
__inference_loss_fn_3_289987б

б 
ф "і щ
@__inference_model_layer_call_and_return_conditional_losses_26076┤+,<-;./0>1=234@5?678B9A:qбn
gбd
ZџW
)і&
left_inputs         }
*і'
right_inputs         }
p 

 
ф "%б"
і
0         
џ щ
@__inference_model_layer_call_and_return_conditional_losses_26180┤+,;<-./0=>1234?@5678AB9:qбn
gбd
ZџW
)і&
left_inputs         }
*і'
right_inputs         }
p

 
ф "%б"
і
0         
џ Ы
@__inference_model_layer_call_and_return_conditional_losses_26614Г+,<-;./0>1=234@5?678B9A:jбg
`б]
SџP
&і#
inputs/0         }
&і#
inputs/1         }
p 

 
ф "%б"
і
0         
џ Ы
@__inference_model_layer_call_and_return_conditional_losses_27036Г+,;<-./0=>1234?@5678AB9:jбg
`б]
SџP
&і#
inputs/0         }
&і#
inputs/1         }
p

 
ф "%б"
і
0         
џ Л
%__inference_model_layer_call_fn_25342Д+,<-;./0>1=234@5?678B9A:qбn
gбd
ZџW
)і&
left_inputs         }
*і'
right_inputs         }
p 

 
ф "і         Л
%__inference_model_layer_call_fn_25972Д+,;<-./0=>1234?@5678AB9:qбn
gбd
ZџW
)і&
left_inputs         }
*і'
right_inputs         }
p

 
ф "і         ╩
%__inference_model_layer_call_fn_26320а+,<-;./0>1=234@5?678B9A:jбg
`б]
SџP
&і#
inputs/0         }
&і#
inputs/1         }
p 

 
ф "і         ╩
%__inference_model_layer_call_fn_26374а+,;<-./0=>1234?@5678AB9:jбg
`б]
SџP
&і#
inputs/0         }
&і#
inputs/1         }
p

 
ф "і         ■
#__inference_signature_wrapper_26266о+,<-;./0>1=234@5?678B9A:ёбђ
б 
yфv
8
left_inputs)і&
left_inputs         }
:
right_inputs*і'
right_inputs         }"3ф0
.
distance"і
distance         ▓
J__inference_stream_0_conv_1_layer_call_and_return_conditional_losses_28121d+,3б0
)б&
$і!
inputs         }
ф ")б&
і
0         }@
џ і
/__inference_stream_0_conv_1_layer_call_fn_28100W+,3б0
)б&
$і!
inputs         }
ф "і         }@│
J__inference_stream_0_conv_2_layer_call_and_return_conditional_losses_28354e/03б0
)б&
$і!
inputs         }@
ф "*б'
 і
0         }ђ
џ І
/__inference_stream_0_conv_2_layer_call_fn_28333X/03б0
)б&
$і!
inputs         }@
ф "і         }ђ┤
J__inference_stream_0_conv_3_layer_call_and_return_conditional_losses_28587f344б1
*б'
%і"
inputs         }ђ
ф "*б'
 і
0         }ђ
џ ї
/__inference_stream_0_conv_3_layer_call_fn_28566Y344б1
*б'
%і"
inputs         }ђ
ф "і         }ђ▓
J__inference_stream_0_drop_1_layer_call_and_return_conditional_losses_28306d7б4
-б*
$і!
inputs         }@
p 
ф ")б&
і
0         }@
џ ▓
J__inference_stream_0_drop_1_layer_call_and_return_conditional_losses_28318d7б4
-б*
$і!
inputs         }@
p
ф ")б&
і
0         }@
џ і
/__inference_stream_0_drop_1_layer_call_fn_28296W7б4
-б*
$і!
inputs         }@
p 
ф "і         }@і
/__inference_stream_0_drop_1_layer_call_fn_28301W7б4
-б*
$і!
inputs         }@
p
ф "і         }@┤
J__inference_stream_0_drop_2_layer_call_and_return_conditional_losses_28539f8б5
.б+
%і"
inputs         }ђ
p 
ф "*б'
 і
0         }ђ
џ ┤
J__inference_stream_0_drop_2_layer_call_and_return_conditional_losses_28551f8б5
.б+
%і"
inputs         }ђ
p
ф "*б'
 і
0         }ђ
џ ї
/__inference_stream_0_drop_2_layer_call_fn_28529Y8б5
.б+
%і"
inputs         }ђ
p 
ф "і         }ђї
/__inference_stream_0_drop_2_layer_call_fn_28534Y8б5
.б+
%і"
inputs         }ђ
p
ф "і         }ђ┤
J__inference_stream_0_drop_3_layer_call_and_return_conditional_losses_28772f8б5
.б+
%і"
inputs         }ђ
p 
ф "*б'
 і
0         }ђ
џ ┤
J__inference_stream_0_drop_3_layer_call_and_return_conditional_losses_28784f8б5
.б+
%і"
inputs         }ђ
p
ф "*б'
 і
0         }ђ
џ ї
/__inference_stream_0_drop_3_layer_call_fn_28762Y8б5
.б+
%і"
inputs         }ђ
p 
ф "і         }ђї
/__inference_stream_0_drop_3_layer_call_fn_28767Y8б5
.б+
%і"
inputs         }ђ
p
ф "і         }ђХ
N__inference_stream_0_input_drop_layer_call_and_return_conditional_losses_28073d7б4
-б*
$і!
inputs         }
p 
ф ")б&
і
0         }
џ Х
N__inference_stream_0_input_drop_layer_call_and_return_conditional_losses_28085d7б4
-б*
$і!
inputs         }
p
ф ")б&
і
0         }
џ ј
3__inference_stream_0_input_drop_layer_call_fn_28063W7б4
-б*
$і!
inputs         }
p 
ф "і         }ј
3__inference_stream_0_input_drop_layer_call_fn_28068W7б4
-б*
$і!
inputs         }
p
ф "і         }