��5
��
D
AddV2
x"T
y"T
z"T"
Ttype:
2	��
B
AssignVariableOp
resource
value"dtype"
dtypetype�
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
�
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
�
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
delete_old_dirsbool(�
?
Mul
x"T
y"T
z"T"
Ttype:
2	�
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
dtypetype�
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
list(type)(0�
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
list(type)(0�
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
�
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
executor_typestring �
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
�
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.6.22v2.6.1-9-gc2363d6d0258��1
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
�
stream_0_conv_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*'
shared_namestream_0_conv_1/kernel
�
*stream_0_conv_1/kernel/Read/ReadVariableOpReadVariableOpstream_0_conv_1/kernel*"
_output_shapes
:@*
dtype0
�
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
�
stream_1_conv_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*'
shared_namestream_1_conv_1/kernel
�
*stream_1_conv_1/kernel/Read/ReadVariableOpReadVariableOpstream_1_conv_1/kernel*"
_output_shapes
:@*
dtype0
�
stream_1_conv_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_namestream_1_conv_1/bias
y
(stream_1_conv_1/bias/Read/ReadVariableOpReadVariableOpstream_1_conv_1/bias*
_output_shapes
:@*
dtype0
�
stream_2_conv_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*'
shared_namestream_2_conv_1/kernel
�
*stream_2_conv_1/kernel/Read/ReadVariableOpReadVariableOpstream_2_conv_1/kernel*"
_output_shapes
:@*
dtype0
�
stream_2_conv_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_namestream_2_conv_1/bias
y
(stream_2_conv_1/bias/Read/ReadVariableOpReadVariableOpstream_2_conv_1/bias*
_output_shapes
:@*
dtype0
�
batch_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@**
shared_namebatch_normalization/gamma
�
-batch_normalization/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization/gamma*
_output_shapes
:@*
dtype0
�
batch_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*)
shared_namebatch_normalization/beta
�
,batch_normalization/beta/Read/ReadVariableOpReadVariableOpbatch_normalization/beta*
_output_shapes
:@*
dtype0
�
batch_normalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_1/gamma
�
/batch_normalization_1/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_1/gamma*
_output_shapes
:@*
dtype0
�
batch_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_namebatch_normalization_1/beta
�
.batch_normalization_1/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_1/beta*
_output_shapes
:@*
dtype0
�
batch_normalization_2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_2/gamma
�
/batch_normalization_2/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_2/gamma*
_output_shapes
:@*
dtype0
�
batch_normalization_2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_namebatch_normalization_2/beta
�
.batch_normalization_2/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_2/beta*
_output_shapes
:@*
dtype0
y
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�T*
shared_namedense_1/kernel
r
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes
:	�T*
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
�
batch_normalization_3/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:T*,
shared_namebatch_normalization_3/gamma
�
/batch_normalization_3/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_3/gamma*
_output_shapes
:T*
dtype0
�
batch_normalization_3/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:T*+
shared_namebatch_normalization_3/beta
�
.batch_normalization_3/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_3/beta*
_output_shapes
:T*
dtype0
�
batch_normalization/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*0
shared_name!batch_normalization/moving_mean
�
3batch_normalization/moving_mean/Read/ReadVariableOpReadVariableOpbatch_normalization/moving_mean*
_output_shapes
:@*
dtype0
�
#batch_normalization/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#batch_normalization/moving_variance
�
7batch_normalization/moving_variance/Read/ReadVariableOpReadVariableOp#batch_normalization/moving_variance*
_output_shapes
:@*
dtype0
�
!batch_normalization_1/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!batch_normalization_1/moving_mean
�
5batch_normalization_1/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_1/moving_mean*
_output_shapes
:@*
dtype0
�
%batch_normalization_1/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*6
shared_name'%batch_normalization_1/moving_variance
�
9batch_normalization_1/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_1/moving_variance*
_output_shapes
:@*
dtype0
�
!batch_normalization_2/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!batch_normalization_2/moving_mean
�
5batch_normalization_2/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_2/moving_mean*
_output_shapes
:@*
dtype0
�
%batch_normalization_2/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*6
shared_name'%batch_normalization_2/moving_variance
�
9batch_normalization_2/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_2/moving_variance*
_output_shapes
:@*
dtype0
�
!batch_normalization_3/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:T*2
shared_name#!batch_normalization_3/moving_mean
�
5batch_normalization_3/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_3/moving_mean*
_output_shapes
:T*
dtype0
�
%batch_normalization_3/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:T*6
shared_name'%batch_normalization_3/moving_variance
�
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
�
Adam/stream_0_conv_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*.
shared_nameAdam/stream_0_conv_1/kernel/m
�
1Adam/stream_0_conv_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/stream_0_conv_1/kernel/m*"
_output_shapes
:@*
dtype0
�
Adam/stream_0_conv_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_nameAdam/stream_0_conv_1/bias/m
�
/Adam/stream_0_conv_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/stream_0_conv_1/bias/m*
_output_shapes
:@*
dtype0
�
Adam/stream_1_conv_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*.
shared_nameAdam/stream_1_conv_1/kernel/m
�
1Adam/stream_1_conv_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/stream_1_conv_1/kernel/m*"
_output_shapes
:@*
dtype0
�
Adam/stream_1_conv_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_nameAdam/stream_1_conv_1/bias/m
�
/Adam/stream_1_conv_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/stream_1_conv_1/bias/m*
_output_shapes
:@*
dtype0
�
Adam/stream_2_conv_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*.
shared_nameAdam/stream_2_conv_1/kernel/m
�
1Adam/stream_2_conv_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/stream_2_conv_1/kernel/m*"
_output_shapes
:@*
dtype0
�
Adam/stream_2_conv_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_nameAdam/stream_2_conv_1/bias/m
�
/Adam/stream_2_conv_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/stream_2_conv_1/bias/m*
_output_shapes
:@*
dtype0
�
 Adam/batch_normalization/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*1
shared_name" Adam/batch_normalization/gamma/m
�
4Adam/batch_normalization/gamma/m/Read/ReadVariableOpReadVariableOp Adam/batch_normalization/gamma/m*
_output_shapes
:@*
dtype0
�
Adam/batch_normalization/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*0
shared_name!Adam/batch_normalization/beta/m
�
3Adam/batch_normalization/beta/m/Read/ReadVariableOpReadVariableOpAdam/batch_normalization/beta/m*
_output_shapes
:@*
dtype0
�
"Adam/batch_normalization_1/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Adam/batch_normalization_1/gamma/m
�
6Adam/batch_normalization_1/gamma/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_1/gamma/m*
_output_shapes
:@*
dtype0
�
!Adam/batch_normalization_1/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!Adam/batch_normalization_1/beta/m
�
5Adam/batch_normalization_1/beta/m/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_1/beta/m*
_output_shapes
:@*
dtype0
�
"Adam/batch_normalization_2/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Adam/batch_normalization_2/gamma/m
�
6Adam/batch_normalization_2/gamma/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_2/gamma/m*
_output_shapes
:@*
dtype0
�
!Adam/batch_normalization_2/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!Adam/batch_normalization_2/beta/m
�
5Adam/batch_normalization_2/beta/m/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_2/beta/m*
_output_shapes
:@*
dtype0
�
Adam/dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�T*&
shared_nameAdam/dense_1/kernel/m
�
)Adam/dense_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/m*
_output_shapes
:	�T*
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
�
"Adam/batch_normalization_3/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:T*3
shared_name$"Adam/batch_normalization_3/gamma/m
�
6Adam/batch_normalization_3/gamma/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_3/gamma/m*
_output_shapes
:T*
dtype0
�
!Adam/batch_normalization_3/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:T*2
shared_name#!Adam/batch_normalization_3/beta/m
�
5Adam/batch_normalization_3/beta/m/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_3/beta/m*
_output_shapes
:T*
dtype0
�
Adam/stream_0_conv_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*.
shared_nameAdam/stream_0_conv_1/kernel/v
�
1Adam/stream_0_conv_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/stream_0_conv_1/kernel/v*"
_output_shapes
:@*
dtype0
�
Adam/stream_0_conv_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_nameAdam/stream_0_conv_1/bias/v
�
/Adam/stream_0_conv_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/stream_0_conv_1/bias/v*
_output_shapes
:@*
dtype0
�
Adam/stream_1_conv_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*.
shared_nameAdam/stream_1_conv_1/kernel/v
�
1Adam/stream_1_conv_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/stream_1_conv_1/kernel/v*"
_output_shapes
:@*
dtype0
�
Adam/stream_1_conv_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_nameAdam/stream_1_conv_1/bias/v
�
/Adam/stream_1_conv_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/stream_1_conv_1/bias/v*
_output_shapes
:@*
dtype0
�
Adam/stream_2_conv_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*.
shared_nameAdam/stream_2_conv_1/kernel/v
�
1Adam/stream_2_conv_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/stream_2_conv_1/kernel/v*"
_output_shapes
:@*
dtype0
�
Adam/stream_2_conv_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_nameAdam/stream_2_conv_1/bias/v
�
/Adam/stream_2_conv_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/stream_2_conv_1/bias/v*
_output_shapes
:@*
dtype0
�
 Adam/batch_normalization/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*1
shared_name" Adam/batch_normalization/gamma/v
�
4Adam/batch_normalization/gamma/v/Read/ReadVariableOpReadVariableOp Adam/batch_normalization/gamma/v*
_output_shapes
:@*
dtype0
�
Adam/batch_normalization/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*0
shared_name!Adam/batch_normalization/beta/v
�
3Adam/batch_normalization/beta/v/Read/ReadVariableOpReadVariableOpAdam/batch_normalization/beta/v*
_output_shapes
:@*
dtype0
�
"Adam/batch_normalization_1/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Adam/batch_normalization_1/gamma/v
�
6Adam/batch_normalization_1/gamma/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_1/gamma/v*
_output_shapes
:@*
dtype0
�
!Adam/batch_normalization_1/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!Adam/batch_normalization_1/beta/v
�
5Adam/batch_normalization_1/beta/v/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_1/beta/v*
_output_shapes
:@*
dtype0
�
"Adam/batch_normalization_2/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Adam/batch_normalization_2/gamma/v
�
6Adam/batch_normalization_2/gamma/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_2/gamma/v*
_output_shapes
:@*
dtype0
�
!Adam/batch_normalization_2/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!Adam/batch_normalization_2/beta/v
�
5Adam/batch_normalization_2/beta/v/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_2/beta/v*
_output_shapes
:@*
dtype0
�
Adam/dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�T*&
shared_nameAdam/dense_1/kernel/v
�
)Adam/dense_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/v*
_output_shapes
:	�T*
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
�
"Adam/batch_normalization_3/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:T*3
shared_name$"Adam/batch_normalization_3/gamma/v
�
6Adam/batch_normalization_3/gamma/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_3/gamma/v*
_output_shapes
:T*
dtype0
�
!Adam/batch_normalization_3/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:T*2
shared_name#!Adam/batch_normalization_3/beta/v
�
5Adam/batch_normalization_3/beta/v/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_3/beta/v*
_output_shapes
:T*
dtype0

NoOpNoOp
��
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�
valueۆB׆ Bφ
�
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
�
layer-0
layer-1
layer-2
layer-3
layer-4
layer-5
layer_with_weights-0
layer-6
layer_with_weights-1
layer-7
layer_with_weights-2
layer-8
layer_with_weights-3
layer-9
layer_with_weights-4
layer-10
layer_with_weights-5
layer-11
layer-12
layer-13
layer-14
layer-15
layer-16
layer-17
layer-18
layer-19
layer-20
 layer-21
!layer-22
"layer_with_weights-6
"layer-23
#layer_with_weights-7
#layer-24
$layer-25
%regularization_losses
&trainable_variables
'	variables
(	keras_api
R
)regularization_losses
*trainable_variables
+	variables
,	keras_api
�

-beta_1

.beta_2
	/decay
0learning_rate
1iter2m�3m�4m�5m�6m�7m�8m�9m�:m�;m�<m�=m�>m�?m�@m�Am�2v�3v�4v�5v�6v�7v�8v�9v�:v�;v�<v�=v�>v�?v�@v�Av�
 
v
20
31
42
53
64
75
86
97
:8
;9
<10
=11
>12
?13
@14
A15
�
20
31
42
53
64
75
86
97
B8
C9
:10
;11
D12
E13
<14
=15
F16
G17
>18
?19
@20
A21
H22
I23
�
Jlayer_metrics
regularization_losses
Klayer_regularization_losses
trainable_variables
	variables
Lnon_trainable_variables

Mlayers
Nmetrics
 
 
 
 
R
Oregularization_losses
Ptrainable_variables
Q	variables
R	keras_api
R
Sregularization_losses
Ttrainable_variables
U	variables
V	keras_api
R
Wregularization_losses
Xtrainable_variables
Y	variables
Z	keras_api
h

2kernel
3bias
[regularization_losses
\trainable_variables
]	variables
^	keras_api
h

4kernel
5bias
_regularization_losses
`trainable_variables
a	variables
b	keras_api
h

6kernel
7bias
cregularization_losses
dtrainable_variables
e	variables
f	keras_api
�
gaxis
	8gamma
9beta
Bmoving_mean
Cmoving_variance
hregularization_losses
itrainable_variables
j	variables
k	keras_api
�
laxis
	:gamma
;beta
Dmoving_mean
Emoving_variance
mregularization_losses
ntrainable_variables
o	variables
p	keras_api
�
qaxis
	<gamma
=beta
Fmoving_mean
Gmoving_variance
rregularization_losses
strainable_variables
t	variables
u	keras_api
R
vregularization_losses
wtrainable_variables
x	variables
y	keras_api
R
zregularization_losses
{trainable_variables
|	variables
}	keras_api
T
~regularization_losses
trainable_variables
�	variables
�	keras_api
V
�regularization_losses
�trainable_variables
�	variables
�	keras_api
V
�regularization_losses
�trainable_variables
�	variables
�	keras_api
V
�regularization_losses
�trainable_variables
�	variables
�	keras_api
V
�regularization_losses
�trainable_variables
�	variables
�	keras_api
V
�regularization_losses
�trainable_variables
�	variables
�	keras_api
V
�regularization_losses
�trainable_variables
�	variables
�	keras_api
V
�regularization_losses
�trainable_variables
�	variables
�	keras_api
V
�regularization_losses
�trainable_variables
�	variables
�	keras_api
l

>kernel
?bias
�regularization_losses
�trainable_variables
�	variables
�	keras_api
�
	�axis
	@gamma
Abeta
Hmoving_mean
Imoving_variance
�regularization_losses
�trainable_variables
�	variables
�	keras_api
V
�regularization_losses
�trainable_variables
�	variables
�	keras_api
 
v
20
31
42
53
64
75
86
97
:8
;9
<10
=11
>12
?13
@14
A15
�
20
31
42
53
64
75
86
97
B8
C9
:10
;11
D12
E13
<14
=15
F16
G17
>18
?19
@20
A21
H22
I23
�
�layer_metrics
%regularization_losses
 �layer_regularization_losses
&trainable_variables
'	variables
�non_trainable_variables
�layers
�metrics
 
 
 
�
�metrics
�layer_metrics
)regularization_losses
*trainable_variables
+	variables
�non_trainable_variables
�layers
 �layer_regularization_losses
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
\Z
VARIABLE_VALUEstream_1_conv_1/kernel0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEstream_1_conv_1/bias0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUEstream_2_conv_1/kernel0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEstream_2_conv_1/bias0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUEbatch_normalization/gamma0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUEbatch_normalization/beta0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEbatch_normalization_1/gamma0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUEbatch_normalization_1/beta0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUE
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
VARIABLE_VALUEbatch_normalization/moving_mean&variables/8/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUE#batch_normalization/moving_variance&variables/9/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUE!batch_normalization_1/moving_mean'variables/12/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUE%batch_normalization_1/moving_variance'variables/13/.ATTRIBUTES/VARIABLE_VALUE
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
B0
C1
D2
E3
F4
G5
H6
I7

0
1
2
3

�0
 
 
 
�
�metrics
�layer_metrics
Oregularization_losses
Ptrainable_variables
Q	variables
�non_trainable_variables
�layers
 �layer_regularization_losses
 
 
 
�
�metrics
�layer_metrics
Sregularization_losses
Ttrainable_variables
U	variables
�non_trainable_variables
�layers
 �layer_regularization_losses
 
 
 
�
�metrics
�layer_metrics
Wregularization_losses
Xtrainable_variables
Y	variables
�non_trainable_variables
�layers
 �layer_regularization_losses
 

20
31

20
31
�
�metrics
�layer_metrics
[regularization_losses
\trainable_variables
]	variables
�non_trainable_variables
�layers
 �layer_regularization_losses
 

40
51

40
51
�
�metrics
�layer_metrics
_regularization_losses
`trainable_variables
a	variables
�non_trainable_variables
�layers
 �layer_regularization_losses
 

60
71

60
71
�
�metrics
�layer_metrics
cregularization_losses
dtrainable_variables
e	variables
�non_trainable_variables
�layers
 �layer_regularization_losses
 
 

80
91

80
91
B2
C3
�
�metrics
�layer_metrics
hregularization_losses
itrainable_variables
j	variables
�non_trainable_variables
�layers
 �layer_regularization_losses
 
 

:0
;1

:0
;1
D2
E3
�
�metrics
�layer_metrics
mregularization_losses
ntrainable_variables
o	variables
�non_trainable_variables
�layers
 �layer_regularization_losses
 
 

<0
=1

<0
=1
F2
G3
�
�metrics
�layer_metrics
rregularization_losses
strainable_variables
t	variables
�non_trainable_variables
�layers
 �layer_regularization_losses
 
 
 
�
�metrics
�layer_metrics
vregularization_losses
wtrainable_variables
x	variables
�non_trainable_variables
�layers
 �layer_regularization_losses
 
 
 
�
�metrics
�layer_metrics
zregularization_losses
{trainable_variables
|	variables
�non_trainable_variables
�layers
 �layer_regularization_losses
 
 
 
�
�metrics
�layer_metrics
~regularization_losses
trainable_variables
�	variables
�non_trainable_variables
�layers
 �layer_regularization_losses
 
 
 
�
�metrics
�layer_metrics
�regularization_losses
�trainable_variables
�	variables
�non_trainable_variables
�layers
 �layer_regularization_losses
 
 
 
�
�metrics
�layer_metrics
�regularization_losses
�trainable_variables
�	variables
�non_trainable_variables
�layers
 �layer_regularization_losses
 
 
 
�
�metrics
�layer_metrics
�regularization_losses
�trainable_variables
�	variables
�non_trainable_variables
�layers
 �layer_regularization_losses
 
 
 
�
�metrics
�layer_metrics
�regularization_losses
�trainable_variables
�	variables
�non_trainable_variables
�layers
 �layer_regularization_losses
 
 
 
�
�metrics
�layer_metrics
�regularization_losses
�trainable_variables
�	variables
�non_trainable_variables
�layers
 �layer_regularization_losses
 
 
 
�
�metrics
�layer_metrics
�regularization_losses
�trainable_variables
�	variables
�non_trainable_variables
�layers
 �layer_regularization_losses
 
 
 
�
�metrics
�layer_metrics
�regularization_losses
�trainable_variables
�	variables
�non_trainable_variables
�layers
 �layer_regularization_losses
 
 
 
�
�metrics
�layer_metrics
�regularization_losses
�trainable_variables
�	variables
�non_trainable_variables
�layers
 �layer_regularization_losses
 

>0
?1

>0
?1
�
�metrics
�layer_metrics
�regularization_losses
�trainable_variables
�	variables
�non_trainable_variables
�layers
 �layer_regularization_losses
 
 

@0
A1

@0
A1
H2
I3
�
�metrics
�layer_metrics
�regularization_losses
�trainable_variables
�	variables
�non_trainable_variables
�layers
 �layer_regularization_losses
 
 
 
�
�metrics
�layer_metrics
�regularization_losses
�trainable_variables
�	variables
�non_trainable_variables
�layers
 �layer_regularization_losses
 
 
8
B0
C1
D2
E3
F4
G5
H6
I7
�
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
19
20
 21
!22
"23
#24
$25
 
 
 
 
 
 
8

�total

�count
�	variables
�	keras_api
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
 
 
 

B0
C1
 
 
 
 

D0
E1
 
 
 
 

F0
G1
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
 
 
 
 
 
 

H0
I1
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
�0
�1

�	variables
}
VARIABLE_VALUEAdam/stream_0_conv_1/kernel/mLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/stream_0_conv_1/bias/mLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/stream_1_conv_1/kernel/mLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/stream_1_conv_1/bias/mLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/stream_2_conv_1/kernel/mLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/stream_2_conv_1/bias/mLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE Adam/batch_normalization/gamma/mLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
�
VARIABLE_VALUEAdam/batch_normalization/beta/mLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE"Adam/batch_normalization_1/gamma/mLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE!Adam/batch_normalization_1/beta/mLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE"Adam/batch_normalization_2/gamma/mMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE!Adam/batch_normalization_2/beta/mMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/dense_1/kernel/mMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/dense_1/bias/mMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE"Adam/batch_normalization_3/gamma/mMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE!Adam/batch_normalization_3/beta/mMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/stream_0_conv_1/kernel/vLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/stream_0_conv_1/bias/vLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/stream_1_conv_1/kernel/vLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/stream_1_conv_1/bias/vLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/stream_2_conv_1/kernel/vLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/stream_2_conv_1/bias/vLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE Adam/batch_normalization/gamma/vLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
�
VARIABLE_VALUEAdam/batch_normalization/beta/vLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE"Adam/batch_normalization_1/gamma/vLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE!Adam/batch_normalization_1/beta/vLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE"Adam/batch_normalization_2/gamma/vMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE!Adam/batch_normalization_2/beta/vMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/dense_1/kernel/vMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/dense_1/bias/vMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE"Adam/batch_normalization_3/gamma/vMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE!Adam/batch_normalization_3/beta/vMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
�
serving_default_left_inputsPlaceholder*,
_output_shapes
:����������*
dtype0*!
shape:����������
�
serving_default_right_inputsPlaceholder*,
_output_shapes
:����������*
dtype0*!
shape:����������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_left_inputsserving_default_right_inputsstream_2_conv_1/kernelstream_2_conv_1/biasstream_1_conv_1/kernelstream_1_conv_1/biasstream_0_conv_1/kernelstream_0_conv_1/bias%batch_normalization_2/moving_variancebatch_normalization_2/gamma!batch_normalization_2/moving_meanbatch_normalization_2/beta%batch_normalization_1/moving_variancebatch_normalization_1/gamma!batch_normalization_1/moving_meanbatch_normalization_1/beta#batch_normalization/moving_variancebatch_normalization/gammabatch_normalization/moving_meanbatch_normalization/betadense_1/kerneldense_1/bias%batch_normalization_3/moving_variancebatch_normalization_3/gamma!batch_normalization_3/moving_meanbatch_normalization_3/beta*%
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*:
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *.
f)R'
%__inference_signature_wrapper_1986661
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenamebeta_1/Read/ReadVariableOpbeta_2/Read/ReadVariableOpdecay/Read/ReadVariableOp!learning_rate/Read/ReadVariableOpAdam/iter/Read/ReadVariableOp*stream_0_conv_1/kernel/Read/ReadVariableOp(stream_0_conv_1/bias/Read/ReadVariableOp*stream_1_conv_1/kernel/Read/ReadVariableOp(stream_1_conv_1/bias/Read/ReadVariableOp*stream_2_conv_1/kernel/Read/ReadVariableOp(stream_2_conv_1/bias/Read/ReadVariableOp-batch_normalization/gamma/Read/ReadVariableOp,batch_normalization/beta/Read/ReadVariableOp/batch_normalization_1/gamma/Read/ReadVariableOp.batch_normalization_1/beta/Read/ReadVariableOp/batch_normalization_2/gamma/Read/ReadVariableOp.batch_normalization_2/beta/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOp/batch_normalization_3/gamma/Read/ReadVariableOp.batch_normalization_3/beta/Read/ReadVariableOp3batch_normalization/moving_mean/Read/ReadVariableOp7batch_normalization/moving_variance/Read/ReadVariableOp5batch_normalization_1/moving_mean/Read/ReadVariableOp9batch_normalization_1/moving_variance/Read/ReadVariableOp5batch_normalization_2/moving_mean/Read/ReadVariableOp9batch_normalization_2/moving_variance/Read/ReadVariableOp5batch_normalization_3/moving_mean/Read/ReadVariableOp9batch_normalization_3/moving_variance/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp1Adam/stream_0_conv_1/kernel/m/Read/ReadVariableOp/Adam/stream_0_conv_1/bias/m/Read/ReadVariableOp1Adam/stream_1_conv_1/kernel/m/Read/ReadVariableOp/Adam/stream_1_conv_1/bias/m/Read/ReadVariableOp1Adam/stream_2_conv_1/kernel/m/Read/ReadVariableOp/Adam/stream_2_conv_1/bias/m/Read/ReadVariableOp4Adam/batch_normalization/gamma/m/Read/ReadVariableOp3Adam/batch_normalization/beta/m/Read/ReadVariableOp6Adam/batch_normalization_1/gamma/m/Read/ReadVariableOp5Adam/batch_normalization_1/beta/m/Read/ReadVariableOp6Adam/batch_normalization_2/gamma/m/Read/ReadVariableOp5Adam/batch_normalization_2/beta/m/Read/ReadVariableOp)Adam/dense_1/kernel/m/Read/ReadVariableOp'Adam/dense_1/bias/m/Read/ReadVariableOp6Adam/batch_normalization_3/gamma/m/Read/ReadVariableOp5Adam/batch_normalization_3/beta/m/Read/ReadVariableOp1Adam/stream_0_conv_1/kernel/v/Read/ReadVariableOp/Adam/stream_0_conv_1/bias/v/Read/ReadVariableOp1Adam/stream_1_conv_1/kernel/v/Read/ReadVariableOp/Adam/stream_1_conv_1/bias/v/Read/ReadVariableOp1Adam/stream_2_conv_1/kernel/v/Read/ReadVariableOp/Adam/stream_2_conv_1/bias/v/Read/ReadVariableOp4Adam/batch_normalization/gamma/v/Read/ReadVariableOp3Adam/batch_normalization/beta/v/Read/ReadVariableOp6Adam/batch_normalization_1/gamma/v/Read/ReadVariableOp5Adam/batch_normalization_1/beta/v/Read/ReadVariableOp6Adam/batch_normalization_2/gamma/v/Read/ReadVariableOp5Adam/batch_normalization_2/beta/v/Read/ReadVariableOp)Adam/dense_1/kernel/v/Read/ReadVariableOp'Adam/dense_1/bias/v/Read/ReadVariableOp6Adam/batch_normalization_3/gamma/v/Read/ReadVariableOp5Adam/batch_normalization_3/beta/v/Read/ReadVariableOpConst*L
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
GPU2*0J 8� *)
f$R"
 __inference__traced_save_1989304
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamebeta_1beta_2decaylearning_rate	Adam/iterstream_0_conv_1/kernelstream_0_conv_1/biasstream_1_conv_1/kernelstream_1_conv_1/biasstream_2_conv_1/kernelstream_2_conv_1/biasbatch_normalization/gammabatch_normalization/betabatch_normalization_1/gammabatch_normalization_1/betabatch_normalization_2/gammabatch_normalization_2/betadense_1/kerneldense_1/biasbatch_normalization_3/gammabatch_normalization_3/betabatch_normalization/moving_mean#batch_normalization/moving_variance!batch_normalization_1/moving_mean%batch_normalization_1/moving_variance!batch_normalization_2/moving_mean%batch_normalization_2/moving_variance!batch_normalization_3/moving_mean%batch_normalization_3/moving_variancetotalcountAdam/stream_0_conv_1/kernel/mAdam/stream_0_conv_1/bias/mAdam/stream_1_conv_1/kernel/mAdam/stream_1_conv_1/bias/mAdam/stream_2_conv_1/kernel/mAdam/stream_2_conv_1/bias/m Adam/batch_normalization/gamma/mAdam/batch_normalization/beta/m"Adam/batch_normalization_1/gamma/m!Adam/batch_normalization_1/beta/m"Adam/batch_normalization_2/gamma/m!Adam/batch_normalization_2/beta/mAdam/dense_1/kernel/mAdam/dense_1/bias/m"Adam/batch_normalization_3/gamma/m!Adam/batch_normalization_3/beta/mAdam/stream_0_conv_1/kernel/vAdam/stream_0_conv_1/bias/vAdam/stream_1_conv_1/kernel/vAdam/stream_1_conv_1/bias/vAdam/stream_2_conv_1/kernel/vAdam/stream_2_conv_1/bias/v Adam/batch_normalization/gamma/vAdam/batch_normalization/beta/v"Adam/batch_normalization_1/gamma/v!Adam/batch_normalization_1/beta/v"Adam/batch_normalization_2/gamma/v!Adam/batch_normalization_2/beta/vAdam/dense_1/kernel/vAdam/dense_1/bias/v"Adam/batch_normalization_3/gamma/v!Adam/batch_normalization_3/beta/v*K
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
GPU2*0J 8� *,
f'R%
#__inference__traced_restore_1989503��/
�
n
5__inference_stream_1_input_drop_layer_call_fn_1988083

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *Y
fTRR
P__inference_stream_1_input_drop_layer_call_and_return_conditional_losses_19854092
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:����������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�+
�
P__inference_batch_normalization_layer_call_and_return_conditional_losses_1983985

inputs5
'assignmovingavg_readvariableop_resource:@7
)assignmovingavg_1_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@/
!batchnorm_readvariableop_resource:@
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOp�
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indices�
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(2
moments/mean�
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:@2
moments/StopGradient�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :������������������@2
moments/SquaredDifference�
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices�
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(2
moments/variance�
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 2
moments/Squeeze�
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
�#<2
AssignMovingAvg/decay�
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg/sub�
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg/mul�
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
�#<2
AssignMovingAvg_1/decay�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg_1/sub�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg_1/mul�
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
 *o�:2
batchnorm/add/y�
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@2
batchnorm/Rsqrt�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOp�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2
batchnorm/mul�
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :������������������@2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:@2
batchnorm/mul_2�
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp�
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :������������������@2
batchnorm/add_1{
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :������������������@2

Identity�
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:������������������@: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :������������������@
 
_user_specified_nameinputs
�
�
P__inference_batch_normalization_layer_call_and_return_conditional_losses_1984811

inputs/
!batchnorm_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@1
#batchnorm_readvariableop_1_resource:@1
#batchnorm_readvariableop_2_resource:@
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOp�
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
 *o�:2
batchnorm/add/y�
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@2
batchnorm/Rsqrt�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOp�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2
batchnorm/mul{
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*,
_output_shapes
:����������@2
batchnorm/mul_1�
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp_1�
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:@2
batchnorm/mul_2�
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp_2�
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*,
_output_shapes
:����������@2
batchnorm/add_1s
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*,
_output_shapes
:����������@2

Identity�
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������@: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:T P
,
_output_shapes
:����������@
 
_user_specified_nameinputs
�
h
L__inference_dense_1_dropout_layer_call_and_return_conditional_losses_1988926

inputs
identity[
IdentityIdentityinputs*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
J
.__inference_activation_2_layer_call_fn_1988740

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_activation_2_layer_call_and_return_conditional_losses_19848262
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:����������@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������@:T P
,
_output_shapes
:����������@
 
_user_specified_nameinputs
�
q
U__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_1988848

inputs
identityr
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Mean/reduction_indiceso
MeanMeaninputsMean/reduction_indices:output:0*
T0*'
_output_shapes
:���������@2
Meana
IdentityIdentityMean:output:0*
T0*'
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������@:T P
,
_output_shapes
:����������@
 
_user_specified_nameinputs
�
�
7__inference_batch_normalization_2_layer_call_fn_1988594

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *[
fVRT
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_19847532
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:����������@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:����������@
 
_user_specified_nameinputs
�
�
L__inference_stream_1_conv_1_layer_call_and_return_conditional_losses_1984701

inputsA
+conv1d_expanddims_1_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�"conv1d/ExpandDims_1/ReadVariableOp�8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
conv1d/ExpandDims/dim�
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:����������2
conv1d/ExpandDims�
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim�
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@2
conv1d/ExpandDims_1�
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:����������@*
paddingSAME*
strides
2
conv1d�
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:����������@*
squeeze_dims

���������2
conv1d/Squeeze�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������@2	
BiasAdd�
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02:
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp�
)stream_1_conv_1/kernel/Regularizer/SquareSquare@stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2+
)stream_1_conv_1/kernel/Regularizer/Square�
(stream_1_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_1_conv_1/kernel/Regularizer/Const�
&stream_1_conv_1/kernel/Regularizer/SumSum-stream_1_conv_1/kernel/Regularizer/Square:y:01stream_1_conv_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&stream_1_conv_1/kernel/Regularizer/Sum�
(stream_1_conv_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2*
(stream_1_conv_1/kernel/Regularizer/mul/x�
&stream_1_conv_1/kernel/Regularizer/mulMul1stream_1_conv_1/kernel/Regularizer/mul/x:output:0/stream_1_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_1_conv_1/kernel/Regularizer/mulp
IdentityIdentityBiasAdd:output:0^NoOp*
T0*,
_output_shapes
:����������@2

Identity�
NoOpNoOp^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp9^stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp2t
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
q
U__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_1984868

inputs
identityr
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Mean/reduction_indiceso
MeanMeaninputsMean/reduction_indices:output:0*
T0*'
_output_shapes
:���������@2
Meana
IdentityIdentityMean:output:0*
T0*'
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������@:T P
,
_output_shapes
:����������@
 
_user_specified_nameinputs
�
h
L__inference_dense_1_dropout_layer_call_and_return_conditional_losses_1985043

inputs
identity[
IdentityIdentityinputs*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
)__inference_dense_1_layer_call_fn_1988941

inputs
unknown:	�T
	unknown_0:T
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������T*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_19849172
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������T2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
+__inference_basemodel_layer_call_fn_1985015
inputs_0
inputs_1
inputs_2
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:@
	unknown_5:@
	unknown_6:@
	unknown_7:@
	unknown_8:@
	unknown_9:@

unknown_10:@

unknown_11:@

unknown_12:@

unknown_13:@

unknown_14:@

unknown_15:@

unknown_16:@

unknown_17:	�T

unknown_18:T

unknown_19:T

unknown_20:T

unknown_21:T

unknown_22:T
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1inputs_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_22*&
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������T*:
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_basemodel_layer_call_and_return_conditional_losses_19849642
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������T2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapesz
x:����������:����������:����������: : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
,
_output_shapes
:����������
"
_user_specified_name
inputs_0:VR
,
_output_shapes
:����������
"
_user_specified_name
inputs_1:VR
,
_output_shapes
:����������
"
_user_specified_name
inputs_2
�
�
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1984753

inputs/
!batchnorm_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@1
#batchnorm_readvariableop_1_resource:@1
#batchnorm_readvariableop_2_resource:@
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOp�
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
 *o�:2
batchnorm/add/y�
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@2
batchnorm/Rsqrt�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOp�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2
batchnorm/mul{
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*,
_output_shapes
:����������@2
batchnorm/mul_1�
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp_1�
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:@2
batchnorm/mul_2�
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp_2�
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*,
_output_shapes
:����������@2
batchnorm/add_1s
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*,
_output_shapes
:����������@2

Identity�
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������@: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:T P
,
_output_shapes
:����������@
 
_user_specified_nameinputs
�
�
D__inference_dense_1_layer_call_and_return_conditional_losses_1984917

inputs1
matmul_readvariableop_resource:	�T-
biasadd_readvariableop_resource:T
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�-dense_1/kernel/Regularizer/Abs/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�T*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������T2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:T*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������T2	
BiasAdd�
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�T*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOp�
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	�T2 
dense_1/kernel/Regularizer/Abs�
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_1/kernel/Regularizer/Const�
dense_1/kernel/Regularizer/SumSum"dense_1/kernel/Regularizer/Abs:y:0)dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/Sum�
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2"
 dense_1/kernel/Regularizer/mul/x�
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/mulk
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������T2

Identity�
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp.^dense_1/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2^
-dense_1/kernel/Regularizer/Abs/ReadVariableOp-dense_1/kernel/Regularizer/Abs/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
j
L__inference_stream_1_drop_1_layer_call_and_return_conditional_losses_1984854

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:����������@2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:����������@2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������@:T P
,
_output_shapes
:����������@
 
_user_specified_nameinputs
�
n
P__inference_stream_1_input_drop_layer_call_and_return_conditional_losses_1988088

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:����������2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:����������2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
s
W__inference_global_average_pooling1d_2_layer_call_and_return_conditional_losses_1984445

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
:������������������2
Meanj
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:������������������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'���������������������������:e a
=
_output_shapes+
):'���������������������������
 
_user_specified_nameinputs
��
�
F__inference_basemodel_layer_call_and_return_conditional_losses_1985903
inputs_0
inputs_1
inputs_2-
stream_2_conv_1_1985810:@%
stream_2_conv_1_1985812:@-
stream_1_conv_1_1985815:@%
stream_1_conv_1_1985817:@-
stream_0_conv_1_1985820:@%
stream_0_conv_1_1985822:@+
batch_normalization_2_1985825:@+
batch_normalization_2_1985827:@+
batch_normalization_2_1985829:@+
batch_normalization_2_1985831:@+
batch_normalization_1_1985834:@+
batch_normalization_1_1985836:@+
batch_normalization_1_1985838:@+
batch_normalization_1_1985840:@)
batch_normalization_1985843:@)
batch_normalization_1985845:@)
batch_normalization_1985847:@)
batch_normalization_1985849:@"
dense_1_1985863:	�T
dense_1_1985865:T+
batch_normalization_3_1985868:T+
batch_normalization_3_1985870:T+
batch_normalization_3_1985872:T+
batch_normalization_3_1985874:T
identity��+batch_normalization/StatefulPartitionedCall�-batch_normalization_1/StatefulPartitionedCall�-batch_normalization_2/StatefulPartitionedCall�-batch_normalization_3/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�-dense_1/kernel/Regularizer/Abs/ReadVariableOp�'stream_0_conv_1/StatefulPartitionedCall�5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp�'stream_0_drop_1/StatefulPartitionedCall�+stream_0_input_drop/StatefulPartitionedCall�'stream_1_conv_1/StatefulPartitionedCall�8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp�'stream_1_drop_1/StatefulPartitionedCall�+stream_1_input_drop/StatefulPartitionedCall�'stream_2_conv_1/StatefulPartitionedCall�5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp�'stream_2_drop_1/StatefulPartitionedCall�+stream_2_input_drop/StatefulPartitionedCall�
+stream_2_input_drop/StatefulPartitionedCallStatefulPartitionedCallinputs_2*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *Y
fTRR
P__inference_stream_2_input_drop_layer_call_and_return_conditional_losses_19854322-
+stream_2_input_drop/StatefulPartitionedCall�
+stream_1_input_drop/StatefulPartitionedCallStatefulPartitionedCallinputs_1,^stream_2_input_drop/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *Y
fTRR
P__inference_stream_1_input_drop_layer_call_and_return_conditional_losses_19854092-
+stream_1_input_drop/StatefulPartitionedCall�
+stream_0_input_drop/StatefulPartitionedCallStatefulPartitionedCallinputs_0,^stream_1_input_drop/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *Y
fTRR
P__inference_stream_0_input_drop_layer_call_and_return_conditional_losses_19853862-
+stream_0_input_drop/StatefulPartitionedCall�
'stream_2_conv_1/StatefulPartitionedCallStatefulPartitionedCall4stream_2_input_drop/StatefulPartitionedCall:output:0stream_2_conv_1_1985810stream_2_conv_1_1985812*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_stream_2_conv_1_layer_call_and_return_conditional_losses_19846742)
'stream_2_conv_1/StatefulPartitionedCall�
'stream_1_conv_1/StatefulPartitionedCallStatefulPartitionedCall4stream_1_input_drop/StatefulPartitionedCall:output:0stream_1_conv_1_1985815stream_1_conv_1_1985817*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_stream_1_conv_1_layer_call_and_return_conditional_losses_19847012)
'stream_1_conv_1/StatefulPartitionedCall�
'stream_0_conv_1/StatefulPartitionedCallStatefulPartitionedCall4stream_0_input_drop/StatefulPartitionedCall:output:0stream_0_conv_1_1985820stream_0_conv_1_1985822*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_stream_0_conv_1_layer_call_and_return_conditional_losses_19847282)
'stream_0_conv_1/StatefulPartitionedCall�
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall0stream_2_conv_1/StatefulPartitionedCall:output:0batch_normalization_2_1985825batch_normalization_2_1985827batch_normalization_2_1985829batch_normalization_2_1985831*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *[
fVRT
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_19853252/
-batch_normalization_2/StatefulPartitionedCall�
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall0stream_1_conv_1/StatefulPartitionedCall:output:0batch_normalization_1_1985834batch_normalization_1_1985836batch_normalization_1_1985838batch_normalization_1_1985840*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *[
fVRT
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_19852652/
-batch_normalization_1/StatefulPartitionedCall�
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall0stream_0_conv_1/StatefulPartitionedCall:output:0batch_normalization_1985843batch_normalization_1985845batch_normalization_1985847batch_normalization_1985849*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_19852052-
+batch_normalization/StatefulPartitionedCall�
activation_2/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_activation_2_layer_call_and_return_conditional_losses_19848262
activation_2/PartitionedCall�
activation_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_activation_1_layer_call_and_return_conditional_losses_19848332
activation_1/PartitionedCall�
activation/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_activation_layer_call_and_return_conditional_losses_19848402
activation/PartitionedCall�
'stream_2_drop_1/StatefulPartitionedCallStatefulPartitionedCall%activation_2/PartitionedCall:output:0,^stream_0_input_drop/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_stream_2_drop_1_layer_call_and_return_conditional_losses_19851352)
'stream_2_drop_1/StatefulPartitionedCall�
'stream_1_drop_1/StatefulPartitionedCallStatefulPartitionedCall%activation_1/PartitionedCall:output:0(^stream_2_drop_1/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_stream_1_drop_1_layer_call_and_return_conditional_losses_19851122)
'stream_1_drop_1/StatefulPartitionedCall�
'stream_0_drop_1/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0(^stream_1_drop_1/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_stream_0_drop_1_layer_call_and_return_conditional_losses_19850892)
'stream_0_drop_1/StatefulPartitionedCall�
(global_average_pooling1d/PartitionedCallPartitionedCall0stream_0_drop_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *^
fYRW
U__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_19848682*
(global_average_pooling1d/PartitionedCall�
*global_average_pooling1d_1/PartitionedCallPartitionedCall0stream_1_drop_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *`
f[RY
W__inference_global_average_pooling1d_1_layer_call_and_return_conditional_losses_19848752,
*global_average_pooling1d_1/PartitionedCall�
*global_average_pooling1d_2/PartitionedCallPartitionedCall0stream_2_drop_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *`
f[RY
W__inference_global_average_pooling1d_2_layer_call_and_return_conditional_losses_19848822,
*global_average_pooling1d_2/PartitionedCall�
concatenate/PartitionedCallPartitionedCall1global_average_pooling1d/PartitionedCall:output:03global_average_pooling1d_1/PartitionedCall:output:03global_average_pooling1d_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_concatenate_layer_call_and_return_conditional_losses_19848922
concatenate/PartitionedCall�
dense_1_dropout/PartitionedCallPartitionedCall$concatenate/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_dense_1_dropout_layer_call_and_return_conditional_losses_19850432!
dense_1_dropout/PartitionedCall�
dense_1/StatefulPartitionedCallStatefulPartitionedCall(dense_1_dropout/PartitionedCall:output:0dense_1_1985863dense_1_1985865*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������T*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_19849172!
dense_1/StatefulPartitionedCall�
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0batch_normalization_3_1985868batch_normalization_3_1985870batch_normalization_3_1985872batch_normalization_3_1985874*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������T*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *[
fVRT
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_19845432/
-batch_normalization_3/StatefulPartitionedCall�
"dense_activation_1/PartitionedCallPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������T* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *X
fSRQ
O__inference_dense_activation_1_layer_call_and_return_conditional_losses_19849372$
"dense_activation_1/PartitionedCall�
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpstream_0_conv_1_1985820*"
_output_shapes
:@*
dtype027
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp�
&stream_0_conv_1/kernel/Regularizer/AbsAbs=stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_0_conv_1/kernel/Regularizer/Abs�
(stream_0_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_0_conv_1/kernel/Regularizer/Const�
&stream_0_conv_1/kernel/Regularizer/SumSum*stream_0_conv_1/kernel/Regularizer/Abs:y:01stream_0_conv_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/Sum�
(stream_0_conv_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2*
(stream_0_conv_1/kernel/Regularizer/mul/x�
&stream_0_conv_1/kernel/Regularizer/mulMul1stream_0_conv_1/kernel/Regularizer/mul/x:output:0/stream_0_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/mul�
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpstream_1_conv_1_1985815*"
_output_shapes
:@*
dtype02:
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp�
)stream_1_conv_1/kernel/Regularizer/SquareSquare@stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2+
)stream_1_conv_1/kernel/Regularizer/Square�
(stream_1_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_1_conv_1/kernel/Regularizer/Const�
&stream_1_conv_1/kernel/Regularizer/SumSum-stream_1_conv_1/kernel/Regularizer/Square:y:01stream_1_conv_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&stream_1_conv_1/kernel/Regularizer/Sum�
(stream_1_conv_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2*
(stream_1_conv_1/kernel/Regularizer/mul/x�
&stream_1_conv_1/kernel/Regularizer/mulMul1stream_1_conv_1/kernel/Regularizer/mul/x:output:0/stream_1_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_1_conv_1/kernel/Regularizer/mul�
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpstream_2_conv_1_1985810*"
_output_shapes
:@*
dtype027
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp�
&stream_2_conv_1/kernel/Regularizer/AbsAbs=stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_2_conv_1/kernel/Regularizer/Abs�
(stream_2_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_2_conv_1/kernel/Regularizer/Const�
&stream_2_conv_1/kernel/Regularizer/SumSum*stream_2_conv_1/kernel/Regularizer/Abs:y:01stream_2_conv_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&stream_2_conv_1/kernel/Regularizer/Sum�
(stream_2_conv_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2*
(stream_2_conv_1/kernel/Regularizer/mul/x�
&stream_2_conv_1/kernel/Regularizer/mulMul1stream_2_conv_1/kernel/Regularizer/mul/x:output:0/stream_2_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_2_conv_1/kernel/Regularizer/mul�
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_1_1985863*
_output_shapes
:	�T*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOp�
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	�T2 
dense_1/kernel/Regularizer/Abs�
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_1/kernel/Regularizer/Const�
dense_1/kernel/Regularizer/SumSum"dense_1/kernel/Regularizer/Abs:y:0)dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/Sum�
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2"
 dense_1/kernel/Regularizer/mul/x�
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/mul�
IdentityIdentity+dense_activation_1/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������T2

Identity�
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall.^dense_1/kernel/Regularizer/Abs/ReadVariableOp(^stream_0_conv_1/StatefulPartitionedCall6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp(^stream_0_drop_1/StatefulPartitionedCall,^stream_0_input_drop/StatefulPartitionedCall(^stream_1_conv_1/StatefulPartitionedCall9^stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp(^stream_1_drop_1/StatefulPartitionedCall,^stream_1_input_drop/StatefulPartitionedCall(^stream_2_conv_1/StatefulPartitionedCall6^stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp(^stream_2_drop_1/StatefulPartitionedCall,^stream_2_input_drop/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapesz
x:����������:����������:����������: : : : : : : : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2^
-dense_1/kernel/Regularizer/Abs/ReadVariableOp-dense_1/kernel/Regularizer/Abs/ReadVariableOp2R
'stream_0_conv_1/StatefulPartitionedCall'stream_0_conv_1/StatefulPartitionedCall2n
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp2R
'stream_0_drop_1/StatefulPartitionedCall'stream_0_drop_1/StatefulPartitionedCall2Z
+stream_0_input_drop/StatefulPartitionedCall+stream_0_input_drop/StatefulPartitionedCall2R
'stream_1_conv_1/StatefulPartitionedCall'stream_1_conv_1/StatefulPartitionedCall2t
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp2R
'stream_1_drop_1/StatefulPartitionedCall'stream_1_drop_1/StatefulPartitionedCall2Z
+stream_1_input_drop/StatefulPartitionedCall+stream_1_input_drop/StatefulPartitionedCall2R
'stream_2_conv_1/StatefulPartitionedCall'stream_2_conv_1/StatefulPartitionedCall2n
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp2R
'stream_2_drop_1/StatefulPartitionedCall'stream_2_drop_1/StatefulPartitionedCall2Z
+stream_2_input_drop/StatefulPartitionedCall+stream_2_input_drop/StatefulPartitionedCall:V R
,
_output_shapes
:����������
"
_user_specified_name
inputs_0:VR
,
_output_shapes
:����������
"
_user_specified_name
inputs_1:VR
,
_output_shapes
:����������
"
_user_specified_name
inputs_2
�	
o
E__inference_distance_layer_call_and_return_conditional_losses_1985998

inputs
inputs_1
identityU
subSubinputsinputs_1*
T0*'
_output_shapes
:���������T2
subU
SquareSquaresub:z:0*
T0*'
_output_shapes
:���������T2
Squarey
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
���������2
Sum/reduction_indices�
SumSum
Square:y:0Sum/reduction_indices:output:0*
T0*'
_output_shapes
:���������*
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
:���������2	
MaximumS
SqrtSqrtMaximum:z:0*
T0*'
_output_shapes
:���������2
Sqrt\
IdentityIdentitySqrt:y:0*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:���������T:���������T:O K
'
_output_shapes
:���������T
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������T
 
_user_specified_nameinputs
�
o
P__inference_stream_1_input_drop_layer_call_and_return_conditional_losses_1985409

inputs
identity�c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
dropout/Constx
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:����������2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:����������*
dtype0*
seed�*
seed2�2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>2
dropout/GreaterEqual/y�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:����������2
dropout/GreaterEqual�
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:����������2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:����������2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
s
W__inference_global_average_pooling1d_1_layer_call_and_return_conditional_losses_1988864

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
:������������������2
Meanj
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:������������������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'���������������������������:e a
=
_output_shapes+
):'���������������������������
 
_user_specified_nameinputs
�J
�	
B__inference_model_layer_call_and_return_conditional_losses_1986575
left_inputs
right_inputs'
basemodel_1986475:@
basemodel_1986477:@'
basemodel_1986479:@
basemodel_1986481:@'
basemodel_1986483:@
basemodel_1986485:@
basemodel_1986487:@
basemodel_1986489:@
basemodel_1986491:@
basemodel_1986493:@
basemodel_1986495:@
basemodel_1986497:@
basemodel_1986499:@
basemodel_1986501:@
basemodel_1986503:@
basemodel_1986505:@
basemodel_1986507:@
basemodel_1986509:@$
basemodel_1986511:	�T
basemodel_1986513:T
basemodel_1986515:T
basemodel_1986517:T
basemodel_1986519:T
basemodel_1986521:T
identity��!basemodel/StatefulPartitionedCall�#basemodel/StatefulPartitionedCall_1�-dense_1/kernel/Regularizer/Abs/ReadVariableOp�5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp�8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp�5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp�
!basemodel/StatefulPartitionedCallStatefulPartitionedCallleft_inputsleft_inputsleft_inputsbasemodel_1986475basemodel_1986477basemodel_1986479basemodel_1986481basemodel_1986483basemodel_1986485basemodel_1986487basemodel_1986489basemodel_1986491basemodel_1986493basemodel_1986495basemodel_1986497basemodel_1986499basemodel_1986501basemodel_1986503basemodel_1986505basemodel_1986507basemodel_1986509basemodel_1986511basemodel_1986513basemodel_1986515basemodel_1986517basemodel_1986519basemodel_1986521*&
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������T*2
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_basemodel_layer_call_and_return_conditional_losses_19855952#
!basemodel/StatefulPartitionedCall�
#basemodel/StatefulPartitionedCall_1StatefulPartitionedCallright_inputsright_inputsright_inputsbasemodel_1986475basemodel_1986477basemodel_1986479basemodel_1986481basemodel_1986483basemodel_1986485basemodel_1986487basemodel_1986489basemodel_1986491basemodel_1986493basemodel_1986495basemodel_1986497basemodel_1986499basemodel_1986501basemodel_1986503basemodel_1986505basemodel_1986507basemodel_1986509basemodel_1986511basemodel_1986513basemodel_1986515basemodel_1986517basemodel_1986519basemodel_1986521"^basemodel/StatefulPartitionedCall*&
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������T*2
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_basemodel_layer_call_and_return_conditional_losses_19855952%
#basemodel/StatefulPartitionedCall_1�
distance/PartitionedCallPartitionedCall*basemodel/StatefulPartitionedCall:output:0,basemodel/StatefulPartitionedCall_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_distance_layer_call_and_return_conditional_losses_19860982
distance/PartitionedCall�
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpbasemodel_1986483*"
_output_shapes
:@*
dtype027
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp�
&stream_0_conv_1/kernel/Regularizer/AbsAbs=stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_0_conv_1/kernel/Regularizer/Abs�
(stream_0_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_0_conv_1/kernel/Regularizer/Const�
&stream_0_conv_1/kernel/Regularizer/SumSum*stream_0_conv_1/kernel/Regularizer/Abs:y:01stream_0_conv_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/Sum�
(stream_0_conv_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2*
(stream_0_conv_1/kernel/Regularizer/mul/x�
&stream_0_conv_1/kernel/Regularizer/mulMul1stream_0_conv_1/kernel/Regularizer/mul/x:output:0/stream_0_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/mul�
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpbasemodel_1986479*"
_output_shapes
:@*
dtype02:
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp�
)stream_1_conv_1/kernel/Regularizer/SquareSquare@stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2+
)stream_1_conv_1/kernel/Regularizer/Square�
(stream_1_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_1_conv_1/kernel/Regularizer/Const�
&stream_1_conv_1/kernel/Regularizer/SumSum-stream_1_conv_1/kernel/Regularizer/Square:y:01stream_1_conv_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&stream_1_conv_1/kernel/Regularizer/Sum�
(stream_1_conv_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2*
(stream_1_conv_1/kernel/Regularizer/mul/x�
&stream_1_conv_1/kernel/Regularizer/mulMul1stream_1_conv_1/kernel/Regularizer/mul/x:output:0/stream_1_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_1_conv_1/kernel/Regularizer/mul�
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpbasemodel_1986475*"
_output_shapes
:@*
dtype027
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp�
&stream_2_conv_1/kernel/Regularizer/AbsAbs=stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_2_conv_1/kernel/Regularizer/Abs�
(stream_2_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_2_conv_1/kernel/Regularizer/Const�
&stream_2_conv_1/kernel/Regularizer/SumSum*stream_2_conv_1/kernel/Regularizer/Abs:y:01stream_2_conv_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&stream_2_conv_1/kernel/Regularizer/Sum�
(stream_2_conv_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2*
(stream_2_conv_1/kernel/Regularizer/mul/x�
&stream_2_conv_1/kernel/Regularizer/mulMul1stream_2_conv_1/kernel/Regularizer/mul/x:output:0/stream_2_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_2_conv_1/kernel/Regularizer/mul�
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpbasemodel_1986511*
_output_shapes
:	�T*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOp�
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	�T2 
dense_1/kernel/Regularizer/Abs�
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_1/kernel/Regularizer/Const�
dense_1/kernel/Regularizer/SumSum"dense_1/kernel/Regularizer/Abs:y:0)dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/Sum�
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2"
 dense_1/kernel/Regularizer/mul/x�
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/mul|
IdentityIdentity!distance/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp"^basemodel/StatefulPartitionedCall$^basemodel/StatefulPartitionedCall_1.^dense_1/kernel/Regularizer/Abs/ReadVariableOp6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp9^stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp6^stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*s
_input_shapesb
`:����������:����������: : : : : : : : : : : : : : : : : : : : : : : : 2F
!basemodel/StatefulPartitionedCall!basemodel/StatefulPartitionedCall2J
#basemodel/StatefulPartitionedCall_1#basemodel/StatefulPartitionedCall_12^
-dense_1/kernel/Regularizer/Abs/ReadVariableOp-dense_1/kernel/Regularizer/Abs/ReadVariableOp2n
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp2t
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp2n
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp:Y U
,
_output_shapes
:����������
%
_user_specified_nameleft_inputs:ZV
,
_output_shapes
:����������
&
_user_specified_nameright_inputs
�
n
P__inference_stream_0_input_drop_layer_call_and_return_conditional_losses_1988061

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:����������2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:����������2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�+
�
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1985265

inputs5
'assignmovingavg_readvariableop_resource:@7
)assignmovingavg_1_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@/
!batchnorm_readvariableop_resource:@
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOp�
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indices�
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(2
moments/mean�
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:@2
moments/StopGradient�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*,
_output_shapes
:����������@2
moments/SquaredDifference�
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices�
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(2
moments/variance�
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 2
moments/Squeeze�
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
�#<2
AssignMovingAvg/decay�
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg/sub�
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg/mul�
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
�#<2
AssignMovingAvg_1/decay�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg_1/sub�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg_1/mul�
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
 *o�:2
batchnorm/add/y�
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@2
batchnorm/Rsqrt�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOp�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2
batchnorm/mul{
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*,
_output_shapes
:����������@2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:@2
batchnorm/mul_2�
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp�
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*,
_output_shapes
:����������@2
batchnorm/add_1s
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*,
_output_shapes
:����������@2

Identity�
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������@: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:T P
,
_output_shapes
:����������@
 
_user_specified_nameinputs
�
o
P__inference_stream_1_input_drop_layer_call_and_return_conditional_losses_1988100

inputs
identity�c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
dropout/Constx
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:����������2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:����������*
dtype0*
seed�*
seed2�2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>2
dropout/GreaterEqual/y�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:����������2
dropout/GreaterEqual�
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:����������2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:����������2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
7__inference_batch_normalization_3_layer_call_fn_1988970

inputs
unknown:T
	unknown_0:T
	unknown_1:T
	unknown_2:T
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������T*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *[
fVRT
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_19844832
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������T2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������T: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������T
 
_user_specified_nameinputs
�
�
L__inference_stream_0_conv_1_layer_call_and_return_conditional_losses_1984728

inputsA
+conv1d_expanddims_1_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�"conv1d/ExpandDims_1/ReadVariableOp�5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
conv1d/ExpandDims/dim�
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:����������2
conv1d/ExpandDims�
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
conv1d/ExpandDims_1/dim�
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@2
conv1d/ExpandDims_1�
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:����������@*
paddingSAME*
strides
2
conv1d�
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:����������@*
squeeze_dims

���������2
conv1d/Squeeze�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������@2	
BiasAdd�
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype027
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp�
&stream_0_conv_1/kernel/Regularizer/AbsAbs=stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_0_conv_1/kernel/Regularizer/Abs�
(stream_0_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_0_conv_1/kernel/Regularizer/Const�
&stream_0_conv_1/kernel/Regularizer/SumSum*stream_0_conv_1/kernel/Regularizer/Abs:y:01stream_0_conv_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/Sum�
(stream_0_conv_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2*
(stream_0_conv_1/kernel/Regularizer/mul/x�
&stream_0_conv_1/kernel/Regularizer/mulMul1stream_0_conv_1/kernel/Regularizer/mul/x:output:0/stream_0_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/mulp
IdentityIdentityBiasAdd:output:0^NoOp*
T0*,
_output_shapes
:����������@2

Identity�
NoOpNoOp^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp2n
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
q
U__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_1988842

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
:������������������2
Meanj
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:������������������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'���������������������������:e a
=
_output_shapes+
):'���������������������������
 
_user_specified_nameinputs
�+
�
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1988715

inputs5
'assignmovingavg_readvariableop_resource:@7
)assignmovingavg_1_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@/
!batchnorm_readvariableop_resource:@
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOp�
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indices�
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(2
moments/mean�
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:@2
moments/StopGradient�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*,
_output_shapes
:����������@2
moments/SquaredDifference�
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices�
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(2
moments/variance�
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 2
moments/Squeeze�
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
�#<2
AssignMovingAvg/decay�
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg/sub�
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg/mul�
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
�#<2
AssignMovingAvg_1/decay�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg_1/sub�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg_1/mul�
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
 *o�:2
batchnorm/add/y�
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@2
batchnorm/Rsqrt�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOp�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2
batchnorm/mul{
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*,
_output_shapes
:����������@2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:@2
batchnorm/mul_2�
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp�
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*,
_output_shapes
:����������@2
batchnorm/add_1s
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*,
_output_shapes
:����������@2

Identity�
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������@: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:T P
,
_output_shapes
:����������@
 
_user_specified_nameinputs
�
j
L__inference_stream_2_drop_1_layer_call_and_return_conditional_losses_1984847

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:����������@2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:����������@2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������@:T P
,
_output_shapes
:����������@
 
_user_specified_nameinputs
�
Q
5__inference_stream_1_input_drop_layer_call_fn_1988078

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *Y
fTRR
P__inference_stream_1_input_drop_layer_call_and_return_conditional_losses_19846442
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
7__inference_batch_normalization_1_layer_call_fn_1988447

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *[
fVRT
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_19852652
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:����������@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:����������@
 
_user_specified_nameinputs
�
�
'__inference_model_layer_call_fn_1986769
inputs_0
inputs_1
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:@
	unknown_5:@
	unknown_6:@
	unknown_7:@
	unknown_8:@
	unknown_9:@

unknown_10:@

unknown_11:@

unknown_12:@

unknown_13:@

unknown_14:@

unknown_15:@

unknown_16:@

unknown_17:	�T

unknown_18:T

unknown_19:T

unknown_20:T

unknown_21:T

unknown_22:T
identity��StatefulPartitionedCall�
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
:���������*2
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_model_layer_call_and_return_conditional_losses_19862622
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*s
_input_shapesb
`:����������:����������: : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
,
_output_shapes
:����������
"
_user_specified_name
inputs/0:VR
,
_output_shapes
:����������
"
_user_specified_name
inputs/1
�
e
I__inference_activation_2_layer_call_and_return_conditional_losses_1988745

inputs
identityS
ReluReluinputs*
T0*,
_output_shapes
:����������@2
Reluk
IdentityIdentityRelu:activations:0*
T0*,
_output_shapes
:����������@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������@:T P
,
_output_shapes
:����������@
 
_user_specified_nameinputs
�
M
1__inference_dense_1_dropout_layer_call_fn_1988917

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_dense_1_dropout_layer_call_and_return_conditional_losses_19850432
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
P__inference_batch_normalization_layer_call_and_return_conditional_losses_1988361

inputs/
!batchnorm_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@1
#batchnorm_readvariableop_1_resource:@1
#batchnorm_readvariableop_2_resource:@
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOp�
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
 *o�:2
batchnorm/add/y�
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@2
batchnorm/Rsqrt�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOp�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2
batchnorm/mul{
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*,
_output_shapes
:����������@2
batchnorm/mul_1�
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp_1�
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:@2
batchnorm/mul_2�
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp_2�
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*,
_output_shapes
:����������@2
batchnorm/add_1s
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*,
_output_shapes
:����������@2

Identity�
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������@: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:T P
,
_output_shapes
:����������@
 
_user_specified_nameinputs
�
Q
5__inference_stream_0_input_drop_layer_call_fn_1988051

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *Y
fTRR
P__inference_stream_0_input_drop_layer_call_and_return_conditional_losses_19846512
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
L__inference_stream_2_conv_1_layer_call_and_return_conditional_losses_1988235

inputsA
+conv1d_expanddims_1_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�"conv1d/ExpandDims_1/ReadVariableOp�5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
conv1d/ExpandDims/dim�
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:����������2
conv1d/ExpandDims�
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim�
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@2
conv1d/ExpandDims_1�
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:����������@*
paddingSAME*
strides
2
conv1d�
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:����������@*
squeeze_dims

���������2
conv1d/Squeeze�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������@2	
BiasAdd�
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype027
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp�
&stream_2_conv_1/kernel/Regularizer/AbsAbs=stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_2_conv_1/kernel/Regularizer/Abs�
(stream_2_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_2_conv_1/kernel/Regularizer/Const�
&stream_2_conv_1/kernel/Regularizer/SumSum*stream_2_conv_1/kernel/Regularizer/Abs:y:01stream_2_conv_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&stream_2_conv_1/kernel/Regularizer/Sum�
(stream_2_conv_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2*
(stream_2_conv_1/kernel/Regularizer/mul/x�
&stream_2_conv_1/kernel/Regularizer/mulMul1stream_2_conv_1/kernel/Regularizer/mul/x:output:0/stream_2_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_2_conv_1/kernel/Regularizer/mulp
IdentityIdentityBiasAdd:output:0^NoOp*
T0*,
_output_shapes
:����������@2

Identity�
NoOpNoOp^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp6^stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp2n
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
X
<__inference_global_average_pooling1d_1_layer_call_fn_1988858

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *`
f[RY
W__inference_global_average_pooling1d_1_layer_call_and_return_conditional_losses_19848752
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������@:T P
,
_output_shapes
:����������@
 
_user_specified_nameinputs
�
k
L__inference_stream_1_drop_1_layer_call_and_return_conditional_losses_1985112

inputs
identity�c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
dropout/Constx
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:����������@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:����������@*
dtype0*
seed�*
seed2�2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>2
dropout/GreaterEqual/y�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:����������@2
dropout/GreaterEqual�
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:����������@2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:����������@2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:����������@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������@:T P
,
_output_shapes
:����������@
 
_user_specified_nameinputs
�
j
L__inference_stream_0_drop_1_layer_call_and_return_conditional_losses_1988760

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:����������@2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:����������@2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������@:T P
,
_output_shapes
:����������@
 
_user_specified_nameinputs
��
�
F__inference_basemodel_layer_call_and_return_conditional_losses_1985595

inputs
inputs_1
inputs_2-
stream_2_conv_1_1985502:@%
stream_2_conv_1_1985504:@-
stream_1_conv_1_1985507:@%
stream_1_conv_1_1985509:@-
stream_0_conv_1_1985512:@%
stream_0_conv_1_1985514:@+
batch_normalization_2_1985517:@+
batch_normalization_2_1985519:@+
batch_normalization_2_1985521:@+
batch_normalization_2_1985523:@+
batch_normalization_1_1985526:@+
batch_normalization_1_1985528:@+
batch_normalization_1_1985530:@+
batch_normalization_1_1985532:@)
batch_normalization_1985535:@)
batch_normalization_1985537:@)
batch_normalization_1985539:@)
batch_normalization_1985541:@"
dense_1_1985555:	�T
dense_1_1985557:T+
batch_normalization_3_1985560:T+
batch_normalization_3_1985562:T+
batch_normalization_3_1985564:T+
batch_normalization_3_1985566:T
identity��+batch_normalization/StatefulPartitionedCall�-batch_normalization_1/StatefulPartitionedCall�-batch_normalization_2/StatefulPartitionedCall�-batch_normalization_3/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�-dense_1/kernel/Regularizer/Abs/ReadVariableOp�'stream_0_conv_1/StatefulPartitionedCall�5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp�'stream_0_drop_1/StatefulPartitionedCall�+stream_0_input_drop/StatefulPartitionedCall�'stream_1_conv_1/StatefulPartitionedCall�8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp�'stream_1_drop_1/StatefulPartitionedCall�+stream_1_input_drop/StatefulPartitionedCall�'stream_2_conv_1/StatefulPartitionedCall�5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp�'stream_2_drop_1/StatefulPartitionedCall�+stream_2_input_drop/StatefulPartitionedCall�
+stream_2_input_drop/StatefulPartitionedCallStatefulPartitionedCallinputs_2*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *Y
fTRR
P__inference_stream_2_input_drop_layer_call_and_return_conditional_losses_19854322-
+stream_2_input_drop/StatefulPartitionedCall�
+stream_1_input_drop/StatefulPartitionedCallStatefulPartitionedCallinputs_1,^stream_2_input_drop/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *Y
fTRR
P__inference_stream_1_input_drop_layer_call_and_return_conditional_losses_19854092-
+stream_1_input_drop/StatefulPartitionedCall�
+stream_0_input_drop/StatefulPartitionedCallStatefulPartitionedCallinputs,^stream_1_input_drop/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *Y
fTRR
P__inference_stream_0_input_drop_layer_call_and_return_conditional_losses_19853862-
+stream_0_input_drop/StatefulPartitionedCall�
'stream_2_conv_1/StatefulPartitionedCallStatefulPartitionedCall4stream_2_input_drop/StatefulPartitionedCall:output:0stream_2_conv_1_1985502stream_2_conv_1_1985504*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_stream_2_conv_1_layer_call_and_return_conditional_losses_19846742)
'stream_2_conv_1/StatefulPartitionedCall�
'stream_1_conv_1/StatefulPartitionedCallStatefulPartitionedCall4stream_1_input_drop/StatefulPartitionedCall:output:0stream_1_conv_1_1985507stream_1_conv_1_1985509*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_stream_1_conv_1_layer_call_and_return_conditional_losses_19847012)
'stream_1_conv_1/StatefulPartitionedCall�
'stream_0_conv_1/StatefulPartitionedCallStatefulPartitionedCall4stream_0_input_drop/StatefulPartitionedCall:output:0stream_0_conv_1_1985512stream_0_conv_1_1985514*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_stream_0_conv_1_layer_call_and_return_conditional_losses_19847282)
'stream_0_conv_1/StatefulPartitionedCall�
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall0stream_2_conv_1/StatefulPartitionedCall:output:0batch_normalization_2_1985517batch_normalization_2_1985519batch_normalization_2_1985521batch_normalization_2_1985523*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *[
fVRT
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_19853252/
-batch_normalization_2/StatefulPartitionedCall�
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall0stream_1_conv_1/StatefulPartitionedCall:output:0batch_normalization_1_1985526batch_normalization_1_1985528batch_normalization_1_1985530batch_normalization_1_1985532*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *[
fVRT
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_19852652/
-batch_normalization_1/StatefulPartitionedCall�
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall0stream_0_conv_1/StatefulPartitionedCall:output:0batch_normalization_1985535batch_normalization_1985537batch_normalization_1985539batch_normalization_1985541*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_19852052-
+batch_normalization/StatefulPartitionedCall�
activation_2/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_activation_2_layer_call_and_return_conditional_losses_19848262
activation_2/PartitionedCall�
activation_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_activation_1_layer_call_and_return_conditional_losses_19848332
activation_1/PartitionedCall�
activation/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_activation_layer_call_and_return_conditional_losses_19848402
activation/PartitionedCall�
'stream_2_drop_1/StatefulPartitionedCallStatefulPartitionedCall%activation_2/PartitionedCall:output:0,^stream_0_input_drop/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_stream_2_drop_1_layer_call_and_return_conditional_losses_19851352)
'stream_2_drop_1/StatefulPartitionedCall�
'stream_1_drop_1/StatefulPartitionedCallStatefulPartitionedCall%activation_1/PartitionedCall:output:0(^stream_2_drop_1/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_stream_1_drop_1_layer_call_and_return_conditional_losses_19851122)
'stream_1_drop_1/StatefulPartitionedCall�
'stream_0_drop_1/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0(^stream_1_drop_1/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_stream_0_drop_1_layer_call_and_return_conditional_losses_19850892)
'stream_0_drop_1/StatefulPartitionedCall�
(global_average_pooling1d/PartitionedCallPartitionedCall0stream_0_drop_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *^
fYRW
U__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_19848682*
(global_average_pooling1d/PartitionedCall�
*global_average_pooling1d_1/PartitionedCallPartitionedCall0stream_1_drop_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *`
f[RY
W__inference_global_average_pooling1d_1_layer_call_and_return_conditional_losses_19848752,
*global_average_pooling1d_1/PartitionedCall�
*global_average_pooling1d_2/PartitionedCallPartitionedCall0stream_2_drop_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *`
f[RY
W__inference_global_average_pooling1d_2_layer_call_and_return_conditional_losses_19848822,
*global_average_pooling1d_2/PartitionedCall�
concatenate/PartitionedCallPartitionedCall1global_average_pooling1d/PartitionedCall:output:03global_average_pooling1d_1/PartitionedCall:output:03global_average_pooling1d_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_concatenate_layer_call_and_return_conditional_losses_19848922
concatenate/PartitionedCall�
dense_1_dropout/PartitionedCallPartitionedCall$concatenate/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_dense_1_dropout_layer_call_and_return_conditional_losses_19850432!
dense_1_dropout/PartitionedCall�
dense_1/StatefulPartitionedCallStatefulPartitionedCall(dense_1_dropout/PartitionedCall:output:0dense_1_1985555dense_1_1985557*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������T*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_19849172!
dense_1/StatefulPartitionedCall�
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0batch_normalization_3_1985560batch_normalization_3_1985562batch_normalization_3_1985564batch_normalization_3_1985566*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������T*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *[
fVRT
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_19845432/
-batch_normalization_3/StatefulPartitionedCall�
"dense_activation_1/PartitionedCallPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������T* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *X
fSRQ
O__inference_dense_activation_1_layer_call_and_return_conditional_losses_19849372$
"dense_activation_1/PartitionedCall�
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpstream_0_conv_1_1985512*"
_output_shapes
:@*
dtype027
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp�
&stream_0_conv_1/kernel/Regularizer/AbsAbs=stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_0_conv_1/kernel/Regularizer/Abs�
(stream_0_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_0_conv_1/kernel/Regularizer/Const�
&stream_0_conv_1/kernel/Regularizer/SumSum*stream_0_conv_1/kernel/Regularizer/Abs:y:01stream_0_conv_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/Sum�
(stream_0_conv_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2*
(stream_0_conv_1/kernel/Regularizer/mul/x�
&stream_0_conv_1/kernel/Regularizer/mulMul1stream_0_conv_1/kernel/Regularizer/mul/x:output:0/stream_0_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/mul�
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpstream_1_conv_1_1985507*"
_output_shapes
:@*
dtype02:
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp�
)stream_1_conv_1/kernel/Regularizer/SquareSquare@stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2+
)stream_1_conv_1/kernel/Regularizer/Square�
(stream_1_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_1_conv_1/kernel/Regularizer/Const�
&stream_1_conv_1/kernel/Regularizer/SumSum-stream_1_conv_1/kernel/Regularizer/Square:y:01stream_1_conv_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&stream_1_conv_1/kernel/Regularizer/Sum�
(stream_1_conv_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2*
(stream_1_conv_1/kernel/Regularizer/mul/x�
&stream_1_conv_1/kernel/Regularizer/mulMul1stream_1_conv_1/kernel/Regularizer/mul/x:output:0/stream_1_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_1_conv_1/kernel/Regularizer/mul�
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpstream_2_conv_1_1985502*"
_output_shapes
:@*
dtype027
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp�
&stream_2_conv_1/kernel/Regularizer/AbsAbs=stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_2_conv_1/kernel/Regularizer/Abs�
(stream_2_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_2_conv_1/kernel/Regularizer/Const�
&stream_2_conv_1/kernel/Regularizer/SumSum*stream_2_conv_1/kernel/Regularizer/Abs:y:01stream_2_conv_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&stream_2_conv_1/kernel/Regularizer/Sum�
(stream_2_conv_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2*
(stream_2_conv_1/kernel/Regularizer/mul/x�
&stream_2_conv_1/kernel/Regularizer/mulMul1stream_2_conv_1/kernel/Regularizer/mul/x:output:0/stream_2_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_2_conv_1/kernel/Regularizer/mul�
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_1_1985555*
_output_shapes
:	�T*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOp�
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	�T2 
dense_1/kernel/Regularizer/Abs�
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_1/kernel/Regularizer/Const�
dense_1/kernel/Regularizer/SumSum"dense_1/kernel/Regularizer/Abs:y:0)dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/Sum�
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2"
 dense_1/kernel/Regularizer/mul/x�
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/mul�
IdentityIdentity+dense_activation_1/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������T2

Identity�
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall.^dense_1/kernel/Regularizer/Abs/ReadVariableOp(^stream_0_conv_1/StatefulPartitionedCall6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp(^stream_0_drop_1/StatefulPartitionedCall,^stream_0_input_drop/StatefulPartitionedCall(^stream_1_conv_1/StatefulPartitionedCall9^stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp(^stream_1_drop_1/StatefulPartitionedCall,^stream_1_input_drop/StatefulPartitionedCall(^stream_2_conv_1/StatefulPartitionedCall6^stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp(^stream_2_drop_1/StatefulPartitionedCall,^stream_2_input_drop/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapesz
x:����������:����������:����������: : : : : : : : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2^
-dense_1/kernel/Regularizer/Abs/ReadVariableOp-dense_1/kernel/Regularizer/Abs/ReadVariableOp2R
'stream_0_conv_1/StatefulPartitionedCall'stream_0_conv_1/StatefulPartitionedCall2n
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp2R
'stream_0_drop_1/StatefulPartitionedCall'stream_0_drop_1/StatefulPartitionedCall2Z
+stream_0_input_drop/StatefulPartitionedCall+stream_0_input_drop/StatefulPartitionedCall2R
'stream_1_conv_1/StatefulPartitionedCall'stream_1_conv_1/StatefulPartitionedCall2t
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp2R
'stream_1_drop_1/StatefulPartitionedCall'stream_1_drop_1/StatefulPartitionedCall2Z
+stream_1_input_drop/StatefulPartitionedCall+stream_1_input_drop/StatefulPartitionedCall2R
'stream_2_conv_1/StatefulPartitionedCall'stream_2_conv_1/StatefulPartitionedCall2n
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp2R
'stream_2_drop_1/StatefulPartitionedCall'stream_2_drop_1/StatefulPartitionedCall2Z
+stream_2_input_drop/StatefulPartitionedCall+stream_2_input_drop/StatefulPartitionedCall:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs:TP
,
_output_shapes
:����������
 
_user_specified_nameinputs:TP
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1984087

inputs/
!batchnorm_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@1
#batchnorm_readvariableop_1_resource:@1
#batchnorm_readvariableop_2_resource:@
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOp�
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
 *o�:2
batchnorm/add/y�
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@2
batchnorm/Rsqrt�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOp�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2
batchnorm/mul�
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :������������������@2
batchnorm/mul_1�
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp_1�
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:@2
batchnorm/mul_2�
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp_2�
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :������������������@2
batchnorm/add_1{
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :������������������@2

Identity�
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:������������������@: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :������������������@
 
_user_specified_nameinputs
�
e
I__inference_activation_2_layer_call_and_return_conditional_losses_1984826

inputs
identityS
ReluReluinputs*
T0*,
_output_shapes
:����������@2
Reluk
IdentityIdentityRelu:activations:0*
T0*,
_output_shapes
:����������@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������@:T P
,
_output_shapes
:����������@
 
_user_specified_nameinputs
�
�
7__inference_batch_normalization_3_layer_call_fn_1988983

inputs
unknown:T
	unknown_0:T
	unknown_1:T
	unknown_2:T
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������T*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *[
fVRT
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_19845432
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������T2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������T: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������T
 
_user_specified_nameinputs
�I
�	
B__inference_model_layer_call_and_return_conditional_losses_1986025

inputs
inputs_1'
basemodel_1985912:@
basemodel_1985914:@'
basemodel_1985916:@
basemodel_1985918:@'
basemodel_1985920:@
basemodel_1985922:@
basemodel_1985924:@
basemodel_1985926:@
basemodel_1985928:@
basemodel_1985930:@
basemodel_1985932:@
basemodel_1985934:@
basemodel_1985936:@
basemodel_1985938:@
basemodel_1985940:@
basemodel_1985942:@
basemodel_1985944:@
basemodel_1985946:@$
basemodel_1985948:	�T
basemodel_1985950:T
basemodel_1985952:T
basemodel_1985954:T
basemodel_1985956:T
basemodel_1985958:T
identity��!basemodel/StatefulPartitionedCall�#basemodel/StatefulPartitionedCall_1�-dense_1/kernel/Regularizer/Abs/ReadVariableOp�5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp�8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp�5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp�
!basemodel/StatefulPartitionedCallStatefulPartitionedCallinputsinputsinputsbasemodel_1985912basemodel_1985914basemodel_1985916basemodel_1985918basemodel_1985920basemodel_1985922basemodel_1985924basemodel_1985926basemodel_1985928basemodel_1985930basemodel_1985932basemodel_1985934basemodel_1985936basemodel_1985938basemodel_1985940basemodel_1985942basemodel_1985944basemodel_1985946basemodel_1985948basemodel_1985950basemodel_1985952basemodel_1985954basemodel_1985956basemodel_1985958*&
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������T*:
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_basemodel_layer_call_and_return_conditional_losses_19849642#
!basemodel/StatefulPartitionedCall�
#basemodel/StatefulPartitionedCall_1StatefulPartitionedCallinputs_1inputs_1inputs_1basemodel_1985912basemodel_1985914basemodel_1985916basemodel_1985918basemodel_1985920basemodel_1985922basemodel_1985924basemodel_1985926basemodel_1985928basemodel_1985930basemodel_1985932basemodel_1985934basemodel_1985936basemodel_1985938basemodel_1985940basemodel_1985942basemodel_1985944basemodel_1985946basemodel_1985948basemodel_1985950basemodel_1985952basemodel_1985954basemodel_1985956basemodel_1985958*&
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������T*:
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_basemodel_layer_call_and_return_conditional_losses_19849642%
#basemodel/StatefulPartitionedCall_1�
distance/PartitionedCallPartitionedCall*basemodel/StatefulPartitionedCall:output:0,basemodel/StatefulPartitionedCall_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_distance_layer_call_and_return_conditional_losses_19859982
distance/PartitionedCall�
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpbasemodel_1985920*"
_output_shapes
:@*
dtype027
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp�
&stream_0_conv_1/kernel/Regularizer/AbsAbs=stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_0_conv_1/kernel/Regularizer/Abs�
(stream_0_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_0_conv_1/kernel/Regularizer/Const�
&stream_0_conv_1/kernel/Regularizer/SumSum*stream_0_conv_1/kernel/Regularizer/Abs:y:01stream_0_conv_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/Sum�
(stream_0_conv_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2*
(stream_0_conv_1/kernel/Regularizer/mul/x�
&stream_0_conv_1/kernel/Regularizer/mulMul1stream_0_conv_1/kernel/Regularizer/mul/x:output:0/stream_0_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/mul�
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpbasemodel_1985916*"
_output_shapes
:@*
dtype02:
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp�
)stream_1_conv_1/kernel/Regularizer/SquareSquare@stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2+
)stream_1_conv_1/kernel/Regularizer/Square�
(stream_1_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_1_conv_1/kernel/Regularizer/Const�
&stream_1_conv_1/kernel/Regularizer/SumSum-stream_1_conv_1/kernel/Regularizer/Square:y:01stream_1_conv_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&stream_1_conv_1/kernel/Regularizer/Sum�
(stream_1_conv_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2*
(stream_1_conv_1/kernel/Regularizer/mul/x�
&stream_1_conv_1/kernel/Regularizer/mulMul1stream_1_conv_1/kernel/Regularizer/mul/x:output:0/stream_1_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_1_conv_1/kernel/Regularizer/mul�
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpbasemodel_1985912*"
_output_shapes
:@*
dtype027
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp�
&stream_2_conv_1/kernel/Regularizer/AbsAbs=stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_2_conv_1/kernel/Regularizer/Abs�
(stream_2_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_2_conv_1/kernel/Regularizer/Const�
&stream_2_conv_1/kernel/Regularizer/SumSum*stream_2_conv_1/kernel/Regularizer/Abs:y:01stream_2_conv_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&stream_2_conv_1/kernel/Regularizer/Sum�
(stream_2_conv_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2*
(stream_2_conv_1/kernel/Regularizer/mul/x�
&stream_2_conv_1/kernel/Regularizer/mulMul1stream_2_conv_1/kernel/Regularizer/mul/x:output:0/stream_2_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_2_conv_1/kernel/Regularizer/mul�
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpbasemodel_1985948*
_output_shapes
:	�T*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOp�
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	�T2 
dense_1/kernel/Regularizer/Abs�
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_1/kernel/Regularizer/Const�
dense_1/kernel/Regularizer/SumSum"dense_1/kernel/Regularizer/Abs:y:0)dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/Sum�
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2"
 dense_1/kernel/Regularizer/mul/x�
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/mul|
IdentityIdentity!distance/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp"^basemodel/StatefulPartitionedCall$^basemodel/StatefulPartitionedCall_1.^dense_1/kernel/Regularizer/Abs/ReadVariableOp6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp9^stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp6^stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*s
_input_shapesb
`:����������:����������: : : : : : : : : : : : : : : : : : : : : : : : 2F
!basemodel/StatefulPartitionedCall!basemodel/StatefulPartitionedCall2J
#basemodel/StatefulPartitionedCall_1#basemodel/StatefulPartitionedCall_12^
-dense_1/kernel/Regularizer/Abs/ReadVariableOp-dense_1/kernel/Regularizer/Abs/ReadVariableOp2n
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp2t
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp2n
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs:TP
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
n
5__inference_stream_0_input_drop_layer_call_fn_1988056

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *Y
fTRR
P__inference_stream_0_input_drop_layer_call_and_return_conditional_losses_19853862
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:����������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�	
q
E__inference_distance_layer_call_and_return_conditional_losses_1988034
inputs_0
inputs_1
identityW
subSubinputs_0inputs_1*
T0*'
_output_shapes
:���������T2
subU
SquareSquaresub:z:0*
T0*'
_output_shapes
:���������T2
Squarey
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
���������2
Sum/reduction_indices�
SumSum
Square:y:0Sum/reduction_indices:output:0*
T0*'
_output_shapes
:���������*
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
:���������2	
MaximumS
SqrtSqrtMaximum:z:0*
T0*'
_output_shapes
:���������2
Sqrt\
IdentityIdentitySqrt:y:0*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:���������T:���������T:Q M
'
_output_shapes
:���������T
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:���������T
"
_user_specified_name
inputs/1
�
�
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1984782

inputs/
!batchnorm_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@1
#batchnorm_readvariableop_1_resource:@1
#batchnorm_readvariableop_2_resource:@
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOp�
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
 *o�:2
batchnorm/add/y�
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@2
batchnorm/Rsqrt�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOp�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2
batchnorm/mul{
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*,
_output_shapes
:����������@2
batchnorm/mul_1�
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp_1�
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:@2
batchnorm/mul_2�
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp_2�
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*,
_output_shapes
:����������@2
batchnorm/add_1s
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*,
_output_shapes
:����������@2

Identity�
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������@: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:T P
,
_output_shapes
:����������@
 
_user_specified_nameinputs
�+
�
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1988555

inputs5
'assignmovingavg_readvariableop_resource:@7
)assignmovingavg_1_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@/
!batchnorm_readvariableop_resource:@
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOp�
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indices�
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(2
moments/mean�
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:@2
moments/StopGradient�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*,
_output_shapes
:����������@2
moments/SquaredDifference�
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices�
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(2
moments/variance�
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 2
moments/Squeeze�
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
�#<2
AssignMovingAvg/decay�
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg/sub�
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg/mul�
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
�#<2
AssignMovingAvg_1/decay�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg_1/sub�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg_1/mul�
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
 *o�:2
batchnorm/add/y�
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@2
batchnorm/Rsqrt�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOp�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2
batchnorm/mul{
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*,
_output_shapes
:����������@2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:@2
batchnorm/mul_2�
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp�
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*,
_output_shapes
:����������@2
batchnorm/add_1s
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*,
_output_shapes
:����������@2

Identity�
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������@: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:T P
,
_output_shapes
:����������@
 
_user_specified_nameinputs
�
n
P__inference_stream_2_input_drop_layer_call_and_return_conditional_losses_1988115

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:����������2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:����������2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
s
W__inference_global_average_pooling1d_1_layer_call_and_return_conditional_losses_1984421

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
:������������������2
Meanj
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:������������������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'���������������������������:e a
=
_output_shapes+
):'���������������������������
 
_user_specified_nameinputs
�	
o
E__inference_distance_layer_call_and_return_conditional_losses_1986098

inputs
inputs_1
identityU
subSubinputsinputs_1*
T0*'
_output_shapes
:���������T2
subU
SquareSquaresub:z:0*
T0*'
_output_shapes
:���������T2
Squarey
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
���������2
Sum/reduction_indices�
SumSum
Square:y:0Sum/reduction_indices:output:0*
T0*'
_output_shapes
:���������*
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
:���������2	
MaximumS
SqrtSqrtMaximum:z:0*
T0*'
_output_shapes
:���������2
Sqrt\
IdentityIdentitySqrt:y:0*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:���������T:���������T:O K
'
_output_shapes
:���������T
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������T
 
_user_specified_nameinputs
�	
�
7__inference_batch_normalization_1_layer_call_fn_1988421

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *[
fVRT
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_19841472
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :������������������@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:������������������@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :������������������@
 
_user_specified_nameinputs
�
V
*__inference_distance_layer_call_fn_1988016
inputs_0
inputs_1
identity�
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_distance_layer_call_and_return_conditional_losses_19859982
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:���������T:���������T:Q M
'
_output_shapes
:���������T
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:���������T
"
_user_specified_name
inputs/1
�
�
1__inference_stream_2_conv_1_layer_call_fn_1988214

inputs
unknown:@
	unknown_0:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_stream_2_conv_1_layer_call_and_return_conditional_losses_19846742
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:����������@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
c
G__inference_activation_layer_call_and_return_conditional_losses_1984840

inputs
identityS
ReluReluinputs*
T0*,
_output_shapes
:����������@2
Reluk
IdentityIdentityRelu:activations:0*
T0*,
_output_shapes
:����������@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������@:T P
,
_output_shapes
:����������@
 
_user_specified_nameinputs
�
s
W__inference_global_average_pooling1d_1_layer_call_and_return_conditional_losses_1988870

inputs
identityr
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Mean/reduction_indiceso
MeanMeaninputsMean/reduction_indices:output:0*
T0*'
_output_shapes
:���������@2
Meana
IdentityIdentityMean:output:0*
T0*'
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������@:T P
,
_output_shapes
:����������@
 
_user_specified_nameinputs
�
V
*__inference_distance_layer_call_fn_1988022
inputs_0
inputs_1
identity�
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_distance_layer_call_and_return_conditional_losses_19860982
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:���������T:���������T:Q M
'
_output_shapes
:���������T
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:���������T
"
_user_specified_name
inputs/1
�+
�
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1984147

inputs5
'assignmovingavg_readvariableop_resource:@7
)assignmovingavg_1_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@/
!batchnorm_readvariableop_resource:@
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOp�
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indices�
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(2
moments/mean�
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:@2
moments/StopGradient�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :������������������@2
moments/SquaredDifference�
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices�
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(2
moments/variance�
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 2
moments/Squeeze�
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
�#<2
AssignMovingAvg/decay�
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg/sub�
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg/mul�
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
�#<2
AssignMovingAvg_1/decay�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg_1/sub�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg_1/mul�
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
 *o�:2
batchnorm/add/y�
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@2
batchnorm/Rsqrt�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOp�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2
batchnorm/mul�
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :������������������@2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:@2
batchnorm/mul_2�
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp�
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :������������������@2
batchnorm/add_1{
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :������������������@2

Identity�
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:������������������@: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :������������������@
 
_user_specified_nameinputs
�
M
1__inference_stream_1_drop_1_layer_call_fn_1988777

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_stream_1_drop_1_layer_call_and_return_conditional_losses_19848542
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:����������@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������@:T P
,
_output_shapes
:����������@
 
_user_specified_nameinputs
�
o
P__inference_stream_2_input_drop_layer_call_and_return_conditional_losses_1988127

inputs
identity�c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
dropout/Constx
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:����������2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:����������*
dtype0*
seed�*
seed2�2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>2
dropout/GreaterEqual/y�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:����������2
dropout/GreaterEqual�
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:����������2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:����������2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
g
-__inference_concatenate_layer_call_fn_1988899
inputs_0
inputs_1
inputs_2
identity�
PartitionedCallPartitionedCallinputs_0inputs_1inputs_2*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_concatenate_layer_call_and_return_conditional_losses_19848922
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:���������@:���������@:���������@:Q M
'
_output_shapes
:���������@
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:���������@
"
_user_specified_name
inputs/1:QM
'
_output_shapes
:���������@
"
_user_specified_name
inputs/2
��
�(
B__inference_model_layer_call_and_return_conditional_losses_1987025
inputs_0
inputs_1[
Ebasemodel_stream_2_conv_1_conv1d_expanddims_1_readvariableop_resource:@G
9basemodel_stream_2_conv_1_biasadd_readvariableop_resource:@[
Ebasemodel_stream_1_conv_1_conv1d_expanddims_1_readvariableop_resource:@G
9basemodel_stream_1_conv_1_biasadd_readvariableop_resource:@[
Ebasemodel_stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource:@G
9basemodel_stream_0_conv_1_biasadd_readvariableop_resource:@O
Abasemodel_batch_normalization_2_batchnorm_readvariableop_resource:@S
Ebasemodel_batch_normalization_2_batchnorm_mul_readvariableop_resource:@Q
Cbasemodel_batch_normalization_2_batchnorm_readvariableop_1_resource:@Q
Cbasemodel_batch_normalization_2_batchnorm_readvariableop_2_resource:@O
Abasemodel_batch_normalization_1_batchnorm_readvariableop_resource:@S
Ebasemodel_batch_normalization_1_batchnorm_mul_readvariableop_resource:@Q
Cbasemodel_batch_normalization_1_batchnorm_readvariableop_1_resource:@Q
Cbasemodel_batch_normalization_1_batchnorm_readvariableop_2_resource:@M
?basemodel_batch_normalization_batchnorm_readvariableop_resource:@Q
Cbasemodel_batch_normalization_batchnorm_mul_readvariableop_resource:@O
Abasemodel_batch_normalization_batchnorm_readvariableop_1_resource:@O
Abasemodel_batch_normalization_batchnorm_readvariableop_2_resource:@C
0basemodel_dense_1_matmul_readvariableop_resource:	�T?
1basemodel_dense_1_biasadd_readvariableop_resource:TO
Abasemodel_batch_normalization_3_batchnorm_readvariableop_resource:TS
Ebasemodel_batch_normalization_3_batchnorm_mul_readvariableop_resource:TQ
Cbasemodel_batch_normalization_3_batchnorm_readvariableop_1_resource:TQ
Cbasemodel_batch_normalization_3_batchnorm_readvariableop_2_resource:T
identity��6basemodel/batch_normalization/batchnorm/ReadVariableOp�8basemodel/batch_normalization/batchnorm/ReadVariableOp_1�8basemodel/batch_normalization/batchnorm/ReadVariableOp_2�:basemodel/batch_normalization/batchnorm/mul/ReadVariableOp�8basemodel/batch_normalization/batchnorm_1/ReadVariableOp�:basemodel/batch_normalization/batchnorm_1/ReadVariableOp_1�:basemodel/batch_normalization/batchnorm_1/ReadVariableOp_2�<basemodel/batch_normalization/batchnorm_1/mul/ReadVariableOp�8basemodel/batch_normalization_1/batchnorm/ReadVariableOp�:basemodel/batch_normalization_1/batchnorm/ReadVariableOp_1�:basemodel/batch_normalization_1/batchnorm/ReadVariableOp_2�<basemodel/batch_normalization_1/batchnorm/mul/ReadVariableOp�:basemodel/batch_normalization_1/batchnorm_1/ReadVariableOp�<basemodel/batch_normalization_1/batchnorm_1/ReadVariableOp_1�<basemodel/batch_normalization_1/batchnorm_1/ReadVariableOp_2�>basemodel/batch_normalization_1/batchnorm_1/mul/ReadVariableOp�8basemodel/batch_normalization_2/batchnorm/ReadVariableOp�:basemodel/batch_normalization_2/batchnorm/ReadVariableOp_1�:basemodel/batch_normalization_2/batchnorm/ReadVariableOp_2�<basemodel/batch_normalization_2/batchnorm/mul/ReadVariableOp�:basemodel/batch_normalization_2/batchnorm_1/ReadVariableOp�<basemodel/batch_normalization_2/batchnorm_1/ReadVariableOp_1�<basemodel/batch_normalization_2/batchnorm_1/ReadVariableOp_2�>basemodel/batch_normalization_2/batchnorm_1/mul/ReadVariableOp�8basemodel/batch_normalization_3/batchnorm/ReadVariableOp�:basemodel/batch_normalization_3/batchnorm/ReadVariableOp_1�:basemodel/batch_normalization_3/batchnorm/ReadVariableOp_2�<basemodel/batch_normalization_3/batchnorm/mul/ReadVariableOp�:basemodel/batch_normalization_3/batchnorm_1/ReadVariableOp�<basemodel/batch_normalization_3/batchnorm_1/ReadVariableOp_1�<basemodel/batch_normalization_3/batchnorm_1/ReadVariableOp_2�>basemodel/batch_normalization_3/batchnorm_1/mul/ReadVariableOp�(basemodel/dense_1/BiasAdd/ReadVariableOp�*basemodel/dense_1/BiasAdd_1/ReadVariableOp�'basemodel/dense_1/MatMul/ReadVariableOp�)basemodel/dense_1/MatMul_1/ReadVariableOp�0basemodel/stream_0_conv_1/BiasAdd/ReadVariableOp�2basemodel/stream_0_conv_1/BiasAdd_1/ReadVariableOp�<basemodel/stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp�>basemodel/stream_0_conv_1/conv1d_1/ExpandDims_1/ReadVariableOp�0basemodel/stream_1_conv_1/BiasAdd/ReadVariableOp�2basemodel/stream_1_conv_1/BiasAdd_1/ReadVariableOp�<basemodel/stream_1_conv_1/conv1d/ExpandDims_1/ReadVariableOp�>basemodel/stream_1_conv_1/conv1d_1/ExpandDims_1/ReadVariableOp�0basemodel/stream_2_conv_1/BiasAdd/ReadVariableOp�2basemodel/stream_2_conv_1/BiasAdd_1/ReadVariableOp�<basemodel/stream_2_conv_1/conv1d/ExpandDims_1/ReadVariableOp�>basemodel/stream_2_conv_1/conv1d_1/ExpandDims_1/ReadVariableOp�-dense_1/kernel/Regularizer/Abs/ReadVariableOp�5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp�8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp�5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp�
&basemodel/stream_2_input_drop/IdentityIdentityinputs_0*
T0*,
_output_shapes
:����������2(
&basemodel/stream_2_input_drop/Identity�
&basemodel/stream_1_input_drop/IdentityIdentityinputs_0*
T0*,
_output_shapes
:����������2(
&basemodel/stream_1_input_drop/Identity�
&basemodel/stream_0_input_drop/IdentityIdentityinputs_0*
T0*,
_output_shapes
:����������2(
&basemodel/stream_0_input_drop/Identity�
/basemodel/stream_2_conv_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������21
/basemodel/stream_2_conv_1/conv1d/ExpandDims/dim�
+basemodel/stream_2_conv_1/conv1d/ExpandDims
ExpandDims/basemodel/stream_2_input_drop/Identity:output:08basemodel/stream_2_conv_1/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:����������2-
+basemodel/stream_2_conv_1/conv1d/ExpandDims�
<basemodel/stream_2_conv_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpEbasemodel_stream_2_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02>
<basemodel/stream_2_conv_1/conv1d/ExpandDims_1/ReadVariableOp�
1basemodel/stream_2_conv_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 23
1basemodel/stream_2_conv_1/conv1d/ExpandDims_1/dim�
-basemodel/stream_2_conv_1/conv1d/ExpandDims_1
ExpandDimsDbasemodel/stream_2_conv_1/conv1d/ExpandDims_1/ReadVariableOp:value:0:basemodel/stream_2_conv_1/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@2/
-basemodel/stream_2_conv_1/conv1d/ExpandDims_1�
 basemodel/stream_2_conv_1/conv1dConv2D4basemodel/stream_2_conv_1/conv1d/ExpandDims:output:06basemodel/stream_2_conv_1/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:����������@*
paddingSAME*
strides
2"
 basemodel/stream_2_conv_1/conv1d�
(basemodel/stream_2_conv_1/conv1d/SqueezeSqueeze)basemodel/stream_2_conv_1/conv1d:output:0*
T0*,
_output_shapes
:����������@*
squeeze_dims

���������2*
(basemodel/stream_2_conv_1/conv1d/Squeeze�
0basemodel/stream_2_conv_1/BiasAdd/ReadVariableOpReadVariableOp9basemodel_stream_2_conv_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype022
0basemodel/stream_2_conv_1/BiasAdd/ReadVariableOp�
!basemodel/stream_2_conv_1/BiasAddBiasAdd1basemodel/stream_2_conv_1/conv1d/Squeeze:output:08basemodel/stream_2_conv_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������@2#
!basemodel/stream_2_conv_1/BiasAdd�
/basemodel/stream_1_conv_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������21
/basemodel/stream_1_conv_1/conv1d/ExpandDims/dim�
+basemodel/stream_1_conv_1/conv1d/ExpandDims
ExpandDims/basemodel/stream_1_input_drop/Identity:output:08basemodel/stream_1_conv_1/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:����������2-
+basemodel/stream_1_conv_1/conv1d/ExpandDims�
<basemodel/stream_1_conv_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpEbasemodel_stream_1_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02>
<basemodel/stream_1_conv_1/conv1d/ExpandDims_1/ReadVariableOp�
1basemodel/stream_1_conv_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 23
1basemodel/stream_1_conv_1/conv1d/ExpandDims_1/dim�
-basemodel/stream_1_conv_1/conv1d/ExpandDims_1
ExpandDimsDbasemodel/stream_1_conv_1/conv1d/ExpandDims_1/ReadVariableOp:value:0:basemodel/stream_1_conv_1/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@2/
-basemodel/stream_1_conv_1/conv1d/ExpandDims_1�
 basemodel/stream_1_conv_1/conv1dConv2D4basemodel/stream_1_conv_1/conv1d/ExpandDims:output:06basemodel/stream_1_conv_1/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:����������@*
paddingSAME*
strides
2"
 basemodel/stream_1_conv_1/conv1d�
(basemodel/stream_1_conv_1/conv1d/SqueezeSqueeze)basemodel/stream_1_conv_1/conv1d:output:0*
T0*,
_output_shapes
:����������@*
squeeze_dims

���������2*
(basemodel/stream_1_conv_1/conv1d/Squeeze�
0basemodel/stream_1_conv_1/BiasAdd/ReadVariableOpReadVariableOp9basemodel_stream_1_conv_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype022
0basemodel/stream_1_conv_1/BiasAdd/ReadVariableOp�
!basemodel/stream_1_conv_1/BiasAddBiasAdd1basemodel/stream_1_conv_1/conv1d/Squeeze:output:08basemodel/stream_1_conv_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������@2#
!basemodel/stream_1_conv_1/BiasAdd�
/basemodel/stream_0_conv_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������21
/basemodel/stream_0_conv_1/conv1d/ExpandDims/dim�
+basemodel/stream_0_conv_1/conv1d/ExpandDims
ExpandDims/basemodel/stream_0_input_drop/Identity:output:08basemodel/stream_0_conv_1/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:����������2-
+basemodel/stream_0_conv_1/conv1d/ExpandDims�
<basemodel/stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpEbasemodel_stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02>
<basemodel/stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp�
1basemodel/stream_0_conv_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 23
1basemodel/stream_0_conv_1/conv1d/ExpandDims_1/dim�
-basemodel/stream_0_conv_1/conv1d/ExpandDims_1
ExpandDimsDbasemodel/stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp:value:0:basemodel/stream_0_conv_1/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@2/
-basemodel/stream_0_conv_1/conv1d/ExpandDims_1�
 basemodel/stream_0_conv_1/conv1dConv2D4basemodel/stream_0_conv_1/conv1d/ExpandDims:output:06basemodel/stream_0_conv_1/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:����������@*
paddingSAME*
strides
2"
 basemodel/stream_0_conv_1/conv1d�
(basemodel/stream_0_conv_1/conv1d/SqueezeSqueeze)basemodel/stream_0_conv_1/conv1d:output:0*
T0*,
_output_shapes
:����������@*
squeeze_dims

���������2*
(basemodel/stream_0_conv_1/conv1d/Squeeze�
0basemodel/stream_0_conv_1/BiasAdd/ReadVariableOpReadVariableOp9basemodel_stream_0_conv_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype022
0basemodel/stream_0_conv_1/BiasAdd/ReadVariableOp�
!basemodel/stream_0_conv_1/BiasAddBiasAdd1basemodel/stream_0_conv_1/conv1d/Squeeze:output:08basemodel/stream_0_conv_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������@2#
!basemodel/stream_0_conv_1/BiasAdd�
8basemodel/batch_normalization_2/batchnorm/ReadVariableOpReadVariableOpAbasemodel_batch_normalization_2_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02:
8basemodel/batch_normalization_2/batchnorm/ReadVariableOp�
/basemodel/batch_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:21
/basemodel/batch_normalization_2/batchnorm/add/y�
-basemodel/batch_normalization_2/batchnorm/addAddV2@basemodel/batch_normalization_2/batchnorm/ReadVariableOp:value:08basemodel/batch_normalization_2/batchnorm/add/y:output:0*
T0*
_output_shapes
:@2/
-basemodel/batch_normalization_2/batchnorm/add�
/basemodel/batch_normalization_2/batchnorm/RsqrtRsqrt1basemodel/batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes
:@21
/basemodel/batch_normalization_2/batchnorm/Rsqrt�
<basemodel/batch_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOpEbasemodel_batch_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02>
<basemodel/batch_normalization_2/batchnorm/mul/ReadVariableOp�
-basemodel/batch_normalization_2/batchnorm/mulMul3basemodel/batch_normalization_2/batchnorm/Rsqrt:y:0Dbasemodel/batch_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2/
-basemodel/batch_normalization_2/batchnorm/mul�
/basemodel/batch_normalization_2/batchnorm/mul_1Mul*basemodel/stream_2_conv_1/BiasAdd:output:01basemodel/batch_normalization_2/batchnorm/mul:z:0*
T0*,
_output_shapes
:����������@21
/basemodel/batch_normalization_2/batchnorm/mul_1�
:basemodel/batch_normalization_2/batchnorm/ReadVariableOp_1ReadVariableOpCbasemodel_batch_normalization_2_batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype02<
:basemodel/batch_normalization_2/batchnorm/ReadVariableOp_1�
/basemodel/batch_normalization_2/batchnorm/mul_2MulBbasemodel/batch_normalization_2/batchnorm/ReadVariableOp_1:value:01basemodel/batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes
:@21
/basemodel/batch_normalization_2/batchnorm/mul_2�
:basemodel/batch_normalization_2/batchnorm/ReadVariableOp_2ReadVariableOpCbasemodel_batch_normalization_2_batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype02<
:basemodel/batch_normalization_2/batchnorm/ReadVariableOp_2�
-basemodel/batch_normalization_2/batchnorm/subSubBbasemodel/batch_normalization_2/batchnorm/ReadVariableOp_2:value:03basemodel/batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2/
-basemodel/batch_normalization_2/batchnorm/sub�
/basemodel/batch_normalization_2/batchnorm/add_1AddV23basemodel/batch_normalization_2/batchnorm/mul_1:z:01basemodel/batch_normalization_2/batchnorm/sub:z:0*
T0*,
_output_shapes
:����������@21
/basemodel/batch_normalization_2/batchnorm/add_1�
8basemodel/batch_normalization_1/batchnorm/ReadVariableOpReadVariableOpAbasemodel_batch_normalization_1_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02:
8basemodel/batch_normalization_1/batchnorm/ReadVariableOp�
/basemodel/batch_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:21
/basemodel/batch_normalization_1/batchnorm/add/y�
-basemodel/batch_normalization_1/batchnorm/addAddV2@basemodel/batch_normalization_1/batchnorm/ReadVariableOp:value:08basemodel/batch_normalization_1/batchnorm/add/y:output:0*
T0*
_output_shapes
:@2/
-basemodel/batch_normalization_1/batchnorm/add�
/basemodel/batch_normalization_1/batchnorm/RsqrtRsqrt1basemodel/batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes
:@21
/basemodel/batch_normalization_1/batchnorm/Rsqrt�
<basemodel/batch_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOpEbasemodel_batch_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02>
<basemodel/batch_normalization_1/batchnorm/mul/ReadVariableOp�
-basemodel/batch_normalization_1/batchnorm/mulMul3basemodel/batch_normalization_1/batchnorm/Rsqrt:y:0Dbasemodel/batch_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2/
-basemodel/batch_normalization_1/batchnorm/mul�
/basemodel/batch_normalization_1/batchnorm/mul_1Mul*basemodel/stream_1_conv_1/BiasAdd:output:01basemodel/batch_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:����������@21
/basemodel/batch_normalization_1/batchnorm/mul_1�
:basemodel/batch_normalization_1/batchnorm/ReadVariableOp_1ReadVariableOpCbasemodel_batch_normalization_1_batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype02<
:basemodel/batch_normalization_1/batchnorm/ReadVariableOp_1�
/basemodel/batch_normalization_1/batchnorm/mul_2MulBbasemodel/batch_normalization_1/batchnorm/ReadVariableOp_1:value:01basemodel/batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes
:@21
/basemodel/batch_normalization_1/batchnorm/mul_2�
:basemodel/batch_normalization_1/batchnorm/ReadVariableOp_2ReadVariableOpCbasemodel_batch_normalization_1_batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype02<
:basemodel/batch_normalization_1/batchnorm/ReadVariableOp_2�
-basemodel/batch_normalization_1/batchnorm/subSubBbasemodel/batch_normalization_1/batchnorm/ReadVariableOp_2:value:03basemodel/batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2/
-basemodel/batch_normalization_1/batchnorm/sub�
/basemodel/batch_normalization_1/batchnorm/add_1AddV23basemodel/batch_normalization_1/batchnorm/mul_1:z:01basemodel/batch_normalization_1/batchnorm/sub:z:0*
T0*,
_output_shapes
:����������@21
/basemodel/batch_normalization_1/batchnorm/add_1�
6basemodel/batch_normalization/batchnorm/ReadVariableOpReadVariableOp?basemodel_batch_normalization_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype028
6basemodel/batch_normalization/batchnorm/ReadVariableOp�
-basemodel/batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2/
-basemodel/batch_normalization/batchnorm/add/y�
+basemodel/batch_normalization/batchnorm/addAddV2>basemodel/batch_normalization/batchnorm/ReadVariableOp:value:06basemodel/batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes
:@2-
+basemodel/batch_normalization/batchnorm/add�
-basemodel/batch_normalization/batchnorm/RsqrtRsqrt/basemodel/batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes
:@2/
-basemodel/batch_normalization/batchnorm/Rsqrt�
:basemodel/batch_normalization/batchnorm/mul/ReadVariableOpReadVariableOpCbasemodel_batch_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02<
:basemodel/batch_normalization/batchnorm/mul/ReadVariableOp�
+basemodel/batch_normalization/batchnorm/mulMul1basemodel/batch_normalization/batchnorm/Rsqrt:y:0Bbasemodel/batch_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2-
+basemodel/batch_normalization/batchnorm/mul�
-basemodel/batch_normalization/batchnorm/mul_1Mul*basemodel/stream_0_conv_1/BiasAdd:output:0/basemodel/batch_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:����������@2/
-basemodel/batch_normalization/batchnorm/mul_1�
8basemodel/batch_normalization/batchnorm/ReadVariableOp_1ReadVariableOpAbasemodel_batch_normalization_batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype02:
8basemodel/batch_normalization/batchnorm/ReadVariableOp_1�
-basemodel/batch_normalization/batchnorm/mul_2Mul@basemodel/batch_normalization/batchnorm/ReadVariableOp_1:value:0/basemodel/batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes
:@2/
-basemodel/batch_normalization/batchnorm/mul_2�
8basemodel/batch_normalization/batchnorm/ReadVariableOp_2ReadVariableOpAbasemodel_batch_normalization_batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype02:
8basemodel/batch_normalization/batchnorm/ReadVariableOp_2�
+basemodel/batch_normalization/batchnorm/subSub@basemodel/batch_normalization/batchnorm/ReadVariableOp_2:value:01basemodel/batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2-
+basemodel/batch_normalization/batchnorm/sub�
-basemodel/batch_normalization/batchnorm/add_1AddV21basemodel/batch_normalization/batchnorm/mul_1:z:0/basemodel/batch_normalization/batchnorm/sub:z:0*
T0*,
_output_shapes
:����������@2/
-basemodel/batch_normalization/batchnorm/add_1�
basemodel/activation_2/ReluRelu3basemodel/batch_normalization_2/batchnorm/add_1:z:0*
T0*,
_output_shapes
:����������@2
basemodel/activation_2/Relu�
basemodel/activation_1/ReluRelu3basemodel/batch_normalization_1/batchnorm/add_1:z:0*
T0*,
_output_shapes
:����������@2
basemodel/activation_1/Relu�
basemodel/activation/ReluRelu1basemodel/batch_normalization/batchnorm/add_1:z:0*
T0*,
_output_shapes
:����������@2
basemodel/activation/Relu�
"basemodel/stream_2_drop_1/IdentityIdentity)basemodel/activation_2/Relu:activations:0*
T0*,
_output_shapes
:����������@2$
"basemodel/stream_2_drop_1/Identity�
"basemodel/stream_1_drop_1/IdentityIdentity)basemodel/activation_1/Relu:activations:0*
T0*,
_output_shapes
:����������@2$
"basemodel/stream_1_drop_1/Identity�
"basemodel/stream_0_drop_1/IdentityIdentity'basemodel/activation/Relu:activations:0*
T0*,
_output_shapes
:����������@2$
"basemodel/stream_0_drop_1/Identity�
9basemodel/global_average_pooling1d/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2;
9basemodel/global_average_pooling1d/Mean/reduction_indices�
'basemodel/global_average_pooling1d/MeanMean+basemodel/stream_0_drop_1/Identity:output:0Bbasemodel/global_average_pooling1d/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:���������@2)
'basemodel/global_average_pooling1d/Mean�
;basemodel/global_average_pooling1d_1/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2=
;basemodel/global_average_pooling1d_1/Mean/reduction_indices�
)basemodel/global_average_pooling1d_1/MeanMean+basemodel/stream_1_drop_1/Identity:output:0Dbasemodel/global_average_pooling1d_1/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:���������@2+
)basemodel/global_average_pooling1d_1/Mean�
;basemodel/global_average_pooling1d_2/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2=
;basemodel/global_average_pooling1d_2/Mean/reduction_indices�
)basemodel/global_average_pooling1d_2/MeanMean+basemodel/stream_2_drop_1/Identity:output:0Dbasemodel/global_average_pooling1d_2/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:���������@2+
)basemodel/global_average_pooling1d_2/Mean�
!basemodel/concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2#
!basemodel/concatenate/concat/axis�
basemodel/concatenate/concatConcatV20basemodel/global_average_pooling1d/Mean:output:02basemodel/global_average_pooling1d_1/Mean:output:02basemodel/global_average_pooling1d_2/Mean:output:0*basemodel/concatenate/concat/axis:output:0*
N*
T0*(
_output_shapes
:����������2
basemodel/concatenate/concat�
"basemodel/dense_1_dropout/IdentityIdentity%basemodel/concatenate/concat:output:0*
T0*(
_output_shapes
:����������2$
"basemodel/dense_1_dropout/Identity�
'basemodel/dense_1/MatMul/ReadVariableOpReadVariableOp0basemodel_dense_1_matmul_readvariableop_resource*
_output_shapes
:	�T*
dtype02)
'basemodel/dense_1/MatMul/ReadVariableOp�
basemodel/dense_1/MatMulMatMul+basemodel/dense_1_dropout/Identity:output:0/basemodel/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������T2
basemodel/dense_1/MatMul�
(basemodel/dense_1/BiasAdd/ReadVariableOpReadVariableOp1basemodel_dense_1_biasadd_readvariableop_resource*
_output_shapes
:T*
dtype02*
(basemodel/dense_1/BiasAdd/ReadVariableOp�
basemodel/dense_1/BiasAddBiasAdd"basemodel/dense_1/MatMul:product:00basemodel/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������T2
basemodel/dense_1/BiasAdd�
8basemodel/batch_normalization_3/batchnorm/ReadVariableOpReadVariableOpAbasemodel_batch_normalization_3_batchnorm_readvariableop_resource*
_output_shapes
:T*
dtype02:
8basemodel/batch_normalization_3/batchnorm/ReadVariableOp�
/basemodel/batch_normalization_3/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:21
/basemodel/batch_normalization_3/batchnorm/add/y�
-basemodel/batch_normalization_3/batchnorm/addAddV2@basemodel/batch_normalization_3/batchnorm/ReadVariableOp:value:08basemodel/batch_normalization_3/batchnorm/add/y:output:0*
T0*
_output_shapes
:T2/
-basemodel/batch_normalization_3/batchnorm/add�
/basemodel/batch_normalization_3/batchnorm/RsqrtRsqrt1basemodel/batch_normalization_3/batchnorm/add:z:0*
T0*
_output_shapes
:T21
/basemodel/batch_normalization_3/batchnorm/Rsqrt�
<basemodel/batch_normalization_3/batchnorm/mul/ReadVariableOpReadVariableOpEbasemodel_batch_normalization_3_batchnorm_mul_readvariableop_resource*
_output_shapes
:T*
dtype02>
<basemodel/batch_normalization_3/batchnorm/mul/ReadVariableOp�
-basemodel/batch_normalization_3/batchnorm/mulMul3basemodel/batch_normalization_3/batchnorm/Rsqrt:y:0Dbasemodel/batch_normalization_3/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:T2/
-basemodel/batch_normalization_3/batchnorm/mul�
/basemodel/batch_normalization_3/batchnorm/mul_1Mul"basemodel/dense_1/BiasAdd:output:01basemodel/batch_normalization_3/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������T21
/basemodel/batch_normalization_3/batchnorm/mul_1�
:basemodel/batch_normalization_3/batchnorm/ReadVariableOp_1ReadVariableOpCbasemodel_batch_normalization_3_batchnorm_readvariableop_1_resource*
_output_shapes
:T*
dtype02<
:basemodel/batch_normalization_3/batchnorm/ReadVariableOp_1�
/basemodel/batch_normalization_3/batchnorm/mul_2MulBbasemodel/batch_normalization_3/batchnorm/ReadVariableOp_1:value:01basemodel/batch_normalization_3/batchnorm/mul:z:0*
T0*
_output_shapes
:T21
/basemodel/batch_normalization_3/batchnorm/mul_2�
:basemodel/batch_normalization_3/batchnorm/ReadVariableOp_2ReadVariableOpCbasemodel_batch_normalization_3_batchnorm_readvariableop_2_resource*
_output_shapes
:T*
dtype02<
:basemodel/batch_normalization_3/batchnorm/ReadVariableOp_2�
-basemodel/batch_normalization_3/batchnorm/subSubBbasemodel/batch_normalization_3/batchnorm/ReadVariableOp_2:value:03basemodel/batch_normalization_3/batchnorm/mul_2:z:0*
T0*
_output_shapes
:T2/
-basemodel/batch_normalization_3/batchnorm/sub�
/basemodel/batch_normalization_3/batchnorm/add_1AddV23basemodel/batch_normalization_3/batchnorm/mul_1:z:01basemodel/batch_normalization_3/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������T21
/basemodel/batch_normalization_3/batchnorm/add_1�
$basemodel/dense_activation_1/SigmoidSigmoid3basemodel/batch_normalization_3/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������T2&
$basemodel/dense_activation_1/Sigmoid�
(basemodel/stream_2_input_drop/Identity_1Identityinputs_1*
T0*,
_output_shapes
:����������2*
(basemodel/stream_2_input_drop/Identity_1�
(basemodel/stream_1_input_drop/Identity_1Identityinputs_1*
T0*,
_output_shapes
:����������2*
(basemodel/stream_1_input_drop/Identity_1�
(basemodel/stream_0_input_drop/Identity_1Identityinputs_1*
T0*,
_output_shapes
:����������2*
(basemodel/stream_0_input_drop/Identity_1�
1basemodel/stream_2_conv_1/conv1d_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������23
1basemodel/stream_2_conv_1/conv1d_1/ExpandDims/dim�
-basemodel/stream_2_conv_1/conv1d_1/ExpandDims
ExpandDims1basemodel/stream_2_input_drop/Identity_1:output:0:basemodel/stream_2_conv_1/conv1d_1/ExpandDims/dim:output:0*
T0*0
_output_shapes
:����������2/
-basemodel/stream_2_conv_1/conv1d_1/ExpandDims�
>basemodel/stream_2_conv_1/conv1d_1/ExpandDims_1/ReadVariableOpReadVariableOpEbasemodel_stream_2_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02@
>basemodel/stream_2_conv_1/conv1d_1/ExpandDims_1/ReadVariableOp�
3basemodel/stream_2_conv_1/conv1d_1/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 25
3basemodel/stream_2_conv_1/conv1d_1/ExpandDims_1/dim�
/basemodel/stream_2_conv_1/conv1d_1/ExpandDims_1
ExpandDimsFbasemodel/stream_2_conv_1/conv1d_1/ExpandDims_1/ReadVariableOp:value:0<basemodel/stream_2_conv_1/conv1d_1/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@21
/basemodel/stream_2_conv_1/conv1d_1/ExpandDims_1�
"basemodel/stream_2_conv_1/conv1d_1Conv2D6basemodel/stream_2_conv_1/conv1d_1/ExpandDims:output:08basemodel/stream_2_conv_1/conv1d_1/ExpandDims_1:output:0*
T0*0
_output_shapes
:����������@*
paddingSAME*
strides
2$
"basemodel/stream_2_conv_1/conv1d_1�
*basemodel/stream_2_conv_1/conv1d_1/SqueezeSqueeze+basemodel/stream_2_conv_1/conv1d_1:output:0*
T0*,
_output_shapes
:����������@*
squeeze_dims

���������2,
*basemodel/stream_2_conv_1/conv1d_1/Squeeze�
2basemodel/stream_2_conv_1/BiasAdd_1/ReadVariableOpReadVariableOp9basemodel_stream_2_conv_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype024
2basemodel/stream_2_conv_1/BiasAdd_1/ReadVariableOp�
#basemodel/stream_2_conv_1/BiasAdd_1BiasAdd3basemodel/stream_2_conv_1/conv1d_1/Squeeze:output:0:basemodel/stream_2_conv_1/BiasAdd_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������@2%
#basemodel/stream_2_conv_1/BiasAdd_1�
1basemodel/stream_1_conv_1/conv1d_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������23
1basemodel/stream_1_conv_1/conv1d_1/ExpandDims/dim�
-basemodel/stream_1_conv_1/conv1d_1/ExpandDims
ExpandDims1basemodel/stream_1_input_drop/Identity_1:output:0:basemodel/stream_1_conv_1/conv1d_1/ExpandDims/dim:output:0*
T0*0
_output_shapes
:����������2/
-basemodel/stream_1_conv_1/conv1d_1/ExpandDims�
>basemodel/stream_1_conv_1/conv1d_1/ExpandDims_1/ReadVariableOpReadVariableOpEbasemodel_stream_1_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02@
>basemodel/stream_1_conv_1/conv1d_1/ExpandDims_1/ReadVariableOp�
3basemodel/stream_1_conv_1/conv1d_1/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 25
3basemodel/stream_1_conv_1/conv1d_1/ExpandDims_1/dim�
/basemodel/stream_1_conv_1/conv1d_1/ExpandDims_1
ExpandDimsFbasemodel/stream_1_conv_1/conv1d_1/ExpandDims_1/ReadVariableOp:value:0<basemodel/stream_1_conv_1/conv1d_1/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@21
/basemodel/stream_1_conv_1/conv1d_1/ExpandDims_1�
"basemodel/stream_1_conv_1/conv1d_1Conv2D6basemodel/stream_1_conv_1/conv1d_1/ExpandDims:output:08basemodel/stream_1_conv_1/conv1d_1/ExpandDims_1:output:0*
T0*0
_output_shapes
:����������@*
paddingSAME*
strides
2$
"basemodel/stream_1_conv_1/conv1d_1�
*basemodel/stream_1_conv_1/conv1d_1/SqueezeSqueeze+basemodel/stream_1_conv_1/conv1d_1:output:0*
T0*,
_output_shapes
:����������@*
squeeze_dims

���������2,
*basemodel/stream_1_conv_1/conv1d_1/Squeeze�
2basemodel/stream_1_conv_1/BiasAdd_1/ReadVariableOpReadVariableOp9basemodel_stream_1_conv_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype024
2basemodel/stream_1_conv_1/BiasAdd_1/ReadVariableOp�
#basemodel/stream_1_conv_1/BiasAdd_1BiasAdd3basemodel/stream_1_conv_1/conv1d_1/Squeeze:output:0:basemodel/stream_1_conv_1/BiasAdd_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������@2%
#basemodel/stream_1_conv_1/BiasAdd_1�
1basemodel/stream_0_conv_1/conv1d_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������23
1basemodel/stream_0_conv_1/conv1d_1/ExpandDims/dim�
-basemodel/stream_0_conv_1/conv1d_1/ExpandDims
ExpandDims1basemodel/stream_0_input_drop/Identity_1:output:0:basemodel/stream_0_conv_1/conv1d_1/ExpandDims/dim:output:0*
T0*0
_output_shapes
:����������2/
-basemodel/stream_0_conv_1/conv1d_1/ExpandDims�
>basemodel/stream_0_conv_1/conv1d_1/ExpandDims_1/ReadVariableOpReadVariableOpEbasemodel_stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02@
>basemodel/stream_0_conv_1/conv1d_1/ExpandDims_1/ReadVariableOp�
3basemodel/stream_0_conv_1/conv1d_1/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 25
3basemodel/stream_0_conv_1/conv1d_1/ExpandDims_1/dim�
/basemodel/stream_0_conv_1/conv1d_1/ExpandDims_1
ExpandDimsFbasemodel/stream_0_conv_1/conv1d_1/ExpandDims_1/ReadVariableOp:value:0<basemodel/stream_0_conv_1/conv1d_1/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@21
/basemodel/stream_0_conv_1/conv1d_1/ExpandDims_1�
"basemodel/stream_0_conv_1/conv1d_1Conv2D6basemodel/stream_0_conv_1/conv1d_1/ExpandDims:output:08basemodel/stream_0_conv_1/conv1d_1/ExpandDims_1:output:0*
T0*0
_output_shapes
:����������@*
paddingSAME*
strides
2$
"basemodel/stream_0_conv_1/conv1d_1�
*basemodel/stream_0_conv_1/conv1d_1/SqueezeSqueeze+basemodel/stream_0_conv_1/conv1d_1:output:0*
T0*,
_output_shapes
:����������@*
squeeze_dims

���������2,
*basemodel/stream_0_conv_1/conv1d_1/Squeeze�
2basemodel/stream_0_conv_1/BiasAdd_1/ReadVariableOpReadVariableOp9basemodel_stream_0_conv_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype024
2basemodel/stream_0_conv_1/BiasAdd_1/ReadVariableOp�
#basemodel/stream_0_conv_1/BiasAdd_1BiasAdd3basemodel/stream_0_conv_1/conv1d_1/Squeeze:output:0:basemodel/stream_0_conv_1/BiasAdd_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������@2%
#basemodel/stream_0_conv_1/BiasAdd_1�
:basemodel/batch_normalization_2/batchnorm_1/ReadVariableOpReadVariableOpAbasemodel_batch_normalization_2_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02<
:basemodel/batch_normalization_2/batchnorm_1/ReadVariableOp�
1basemodel/batch_normalization_2/batchnorm_1/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:23
1basemodel/batch_normalization_2/batchnorm_1/add/y�
/basemodel/batch_normalization_2/batchnorm_1/addAddV2Bbasemodel/batch_normalization_2/batchnorm_1/ReadVariableOp:value:0:basemodel/batch_normalization_2/batchnorm_1/add/y:output:0*
T0*
_output_shapes
:@21
/basemodel/batch_normalization_2/batchnorm_1/add�
1basemodel/batch_normalization_2/batchnorm_1/RsqrtRsqrt3basemodel/batch_normalization_2/batchnorm_1/add:z:0*
T0*
_output_shapes
:@23
1basemodel/batch_normalization_2/batchnorm_1/Rsqrt�
>basemodel/batch_normalization_2/batchnorm_1/mul/ReadVariableOpReadVariableOpEbasemodel_batch_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02@
>basemodel/batch_normalization_2/batchnorm_1/mul/ReadVariableOp�
/basemodel/batch_normalization_2/batchnorm_1/mulMul5basemodel/batch_normalization_2/batchnorm_1/Rsqrt:y:0Fbasemodel/batch_normalization_2/batchnorm_1/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@21
/basemodel/batch_normalization_2/batchnorm_1/mul�
1basemodel/batch_normalization_2/batchnorm_1/mul_1Mul,basemodel/stream_2_conv_1/BiasAdd_1:output:03basemodel/batch_normalization_2/batchnorm_1/mul:z:0*
T0*,
_output_shapes
:����������@23
1basemodel/batch_normalization_2/batchnorm_1/mul_1�
<basemodel/batch_normalization_2/batchnorm_1/ReadVariableOp_1ReadVariableOpCbasemodel_batch_normalization_2_batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype02>
<basemodel/batch_normalization_2/batchnorm_1/ReadVariableOp_1�
1basemodel/batch_normalization_2/batchnorm_1/mul_2MulDbasemodel/batch_normalization_2/batchnorm_1/ReadVariableOp_1:value:03basemodel/batch_normalization_2/batchnorm_1/mul:z:0*
T0*
_output_shapes
:@23
1basemodel/batch_normalization_2/batchnorm_1/mul_2�
<basemodel/batch_normalization_2/batchnorm_1/ReadVariableOp_2ReadVariableOpCbasemodel_batch_normalization_2_batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype02>
<basemodel/batch_normalization_2/batchnorm_1/ReadVariableOp_2�
/basemodel/batch_normalization_2/batchnorm_1/subSubDbasemodel/batch_normalization_2/batchnorm_1/ReadVariableOp_2:value:05basemodel/batch_normalization_2/batchnorm_1/mul_2:z:0*
T0*
_output_shapes
:@21
/basemodel/batch_normalization_2/batchnorm_1/sub�
1basemodel/batch_normalization_2/batchnorm_1/add_1AddV25basemodel/batch_normalization_2/batchnorm_1/mul_1:z:03basemodel/batch_normalization_2/batchnorm_1/sub:z:0*
T0*,
_output_shapes
:����������@23
1basemodel/batch_normalization_2/batchnorm_1/add_1�
:basemodel/batch_normalization_1/batchnorm_1/ReadVariableOpReadVariableOpAbasemodel_batch_normalization_1_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02<
:basemodel/batch_normalization_1/batchnorm_1/ReadVariableOp�
1basemodel/batch_normalization_1/batchnorm_1/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:23
1basemodel/batch_normalization_1/batchnorm_1/add/y�
/basemodel/batch_normalization_1/batchnorm_1/addAddV2Bbasemodel/batch_normalization_1/batchnorm_1/ReadVariableOp:value:0:basemodel/batch_normalization_1/batchnorm_1/add/y:output:0*
T0*
_output_shapes
:@21
/basemodel/batch_normalization_1/batchnorm_1/add�
1basemodel/batch_normalization_1/batchnorm_1/RsqrtRsqrt3basemodel/batch_normalization_1/batchnorm_1/add:z:0*
T0*
_output_shapes
:@23
1basemodel/batch_normalization_1/batchnorm_1/Rsqrt�
>basemodel/batch_normalization_1/batchnorm_1/mul/ReadVariableOpReadVariableOpEbasemodel_batch_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02@
>basemodel/batch_normalization_1/batchnorm_1/mul/ReadVariableOp�
/basemodel/batch_normalization_1/batchnorm_1/mulMul5basemodel/batch_normalization_1/batchnorm_1/Rsqrt:y:0Fbasemodel/batch_normalization_1/batchnorm_1/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@21
/basemodel/batch_normalization_1/batchnorm_1/mul�
1basemodel/batch_normalization_1/batchnorm_1/mul_1Mul,basemodel/stream_1_conv_1/BiasAdd_1:output:03basemodel/batch_normalization_1/batchnorm_1/mul:z:0*
T0*,
_output_shapes
:����������@23
1basemodel/batch_normalization_1/batchnorm_1/mul_1�
<basemodel/batch_normalization_1/batchnorm_1/ReadVariableOp_1ReadVariableOpCbasemodel_batch_normalization_1_batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype02>
<basemodel/batch_normalization_1/batchnorm_1/ReadVariableOp_1�
1basemodel/batch_normalization_1/batchnorm_1/mul_2MulDbasemodel/batch_normalization_1/batchnorm_1/ReadVariableOp_1:value:03basemodel/batch_normalization_1/batchnorm_1/mul:z:0*
T0*
_output_shapes
:@23
1basemodel/batch_normalization_1/batchnorm_1/mul_2�
<basemodel/batch_normalization_1/batchnorm_1/ReadVariableOp_2ReadVariableOpCbasemodel_batch_normalization_1_batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype02>
<basemodel/batch_normalization_1/batchnorm_1/ReadVariableOp_2�
/basemodel/batch_normalization_1/batchnorm_1/subSubDbasemodel/batch_normalization_1/batchnorm_1/ReadVariableOp_2:value:05basemodel/batch_normalization_1/batchnorm_1/mul_2:z:0*
T0*
_output_shapes
:@21
/basemodel/batch_normalization_1/batchnorm_1/sub�
1basemodel/batch_normalization_1/batchnorm_1/add_1AddV25basemodel/batch_normalization_1/batchnorm_1/mul_1:z:03basemodel/batch_normalization_1/batchnorm_1/sub:z:0*
T0*,
_output_shapes
:����������@23
1basemodel/batch_normalization_1/batchnorm_1/add_1�
8basemodel/batch_normalization/batchnorm_1/ReadVariableOpReadVariableOp?basemodel_batch_normalization_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02:
8basemodel/batch_normalization/batchnorm_1/ReadVariableOp�
/basemodel/batch_normalization/batchnorm_1/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:21
/basemodel/batch_normalization/batchnorm_1/add/y�
-basemodel/batch_normalization/batchnorm_1/addAddV2@basemodel/batch_normalization/batchnorm_1/ReadVariableOp:value:08basemodel/batch_normalization/batchnorm_1/add/y:output:0*
T0*
_output_shapes
:@2/
-basemodel/batch_normalization/batchnorm_1/add�
/basemodel/batch_normalization/batchnorm_1/RsqrtRsqrt1basemodel/batch_normalization/batchnorm_1/add:z:0*
T0*
_output_shapes
:@21
/basemodel/batch_normalization/batchnorm_1/Rsqrt�
<basemodel/batch_normalization/batchnorm_1/mul/ReadVariableOpReadVariableOpCbasemodel_batch_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02>
<basemodel/batch_normalization/batchnorm_1/mul/ReadVariableOp�
-basemodel/batch_normalization/batchnorm_1/mulMul3basemodel/batch_normalization/batchnorm_1/Rsqrt:y:0Dbasemodel/batch_normalization/batchnorm_1/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2/
-basemodel/batch_normalization/batchnorm_1/mul�
/basemodel/batch_normalization/batchnorm_1/mul_1Mul,basemodel/stream_0_conv_1/BiasAdd_1:output:01basemodel/batch_normalization/batchnorm_1/mul:z:0*
T0*,
_output_shapes
:����������@21
/basemodel/batch_normalization/batchnorm_1/mul_1�
:basemodel/batch_normalization/batchnorm_1/ReadVariableOp_1ReadVariableOpAbasemodel_batch_normalization_batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype02<
:basemodel/batch_normalization/batchnorm_1/ReadVariableOp_1�
/basemodel/batch_normalization/batchnorm_1/mul_2MulBbasemodel/batch_normalization/batchnorm_1/ReadVariableOp_1:value:01basemodel/batch_normalization/batchnorm_1/mul:z:0*
T0*
_output_shapes
:@21
/basemodel/batch_normalization/batchnorm_1/mul_2�
:basemodel/batch_normalization/batchnorm_1/ReadVariableOp_2ReadVariableOpAbasemodel_batch_normalization_batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype02<
:basemodel/batch_normalization/batchnorm_1/ReadVariableOp_2�
-basemodel/batch_normalization/batchnorm_1/subSubBbasemodel/batch_normalization/batchnorm_1/ReadVariableOp_2:value:03basemodel/batch_normalization/batchnorm_1/mul_2:z:0*
T0*
_output_shapes
:@2/
-basemodel/batch_normalization/batchnorm_1/sub�
/basemodel/batch_normalization/batchnorm_1/add_1AddV23basemodel/batch_normalization/batchnorm_1/mul_1:z:01basemodel/batch_normalization/batchnorm_1/sub:z:0*
T0*,
_output_shapes
:����������@21
/basemodel/batch_normalization/batchnorm_1/add_1�
basemodel/activation_2/Relu_1Relu5basemodel/batch_normalization_2/batchnorm_1/add_1:z:0*
T0*,
_output_shapes
:����������@2
basemodel/activation_2/Relu_1�
basemodel/activation_1/Relu_1Relu5basemodel/batch_normalization_1/batchnorm_1/add_1:z:0*
T0*,
_output_shapes
:����������@2
basemodel/activation_1/Relu_1�
basemodel/activation/Relu_1Relu3basemodel/batch_normalization/batchnorm_1/add_1:z:0*
T0*,
_output_shapes
:����������@2
basemodel/activation/Relu_1�
$basemodel/stream_2_drop_1/Identity_1Identity+basemodel/activation_2/Relu_1:activations:0*
T0*,
_output_shapes
:����������@2&
$basemodel/stream_2_drop_1/Identity_1�
$basemodel/stream_1_drop_1/Identity_1Identity+basemodel/activation_1/Relu_1:activations:0*
T0*,
_output_shapes
:����������@2&
$basemodel/stream_1_drop_1/Identity_1�
$basemodel/stream_0_drop_1/Identity_1Identity)basemodel/activation/Relu_1:activations:0*
T0*,
_output_shapes
:����������@2&
$basemodel/stream_0_drop_1/Identity_1�
;basemodel/global_average_pooling1d/Mean_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2=
;basemodel/global_average_pooling1d/Mean_1/reduction_indices�
)basemodel/global_average_pooling1d/Mean_1Mean-basemodel/stream_0_drop_1/Identity_1:output:0Dbasemodel/global_average_pooling1d/Mean_1/reduction_indices:output:0*
T0*'
_output_shapes
:���������@2+
)basemodel/global_average_pooling1d/Mean_1�
=basemodel/global_average_pooling1d_1/Mean_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2?
=basemodel/global_average_pooling1d_1/Mean_1/reduction_indices�
+basemodel/global_average_pooling1d_1/Mean_1Mean-basemodel/stream_1_drop_1/Identity_1:output:0Fbasemodel/global_average_pooling1d_1/Mean_1/reduction_indices:output:0*
T0*'
_output_shapes
:���������@2-
+basemodel/global_average_pooling1d_1/Mean_1�
=basemodel/global_average_pooling1d_2/Mean_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2?
=basemodel/global_average_pooling1d_2/Mean_1/reduction_indices�
+basemodel/global_average_pooling1d_2/Mean_1Mean-basemodel/stream_2_drop_1/Identity_1:output:0Fbasemodel/global_average_pooling1d_2/Mean_1/reduction_indices:output:0*
T0*'
_output_shapes
:���������@2-
+basemodel/global_average_pooling1d_2/Mean_1�
#basemodel/concatenate/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :2%
#basemodel/concatenate/concat_1/axis�
basemodel/concatenate/concat_1ConcatV22basemodel/global_average_pooling1d/Mean_1:output:04basemodel/global_average_pooling1d_1/Mean_1:output:04basemodel/global_average_pooling1d_2/Mean_1:output:0,basemodel/concatenate/concat_1/axis:output:0*
N*
T0*(
_output_shapes
:����������2 
basemodel/concatenate/concat_1�
$basemodel/dense_1_dropout/Identity_1Identity'basemodel/concatenate/concat_1:output:0*
T0*(
_output_shapes
:����������2&
$basemodel/dense_1_dropout/Identity_1�
)basemodel/dense_1/MatMul_1/ReadVariableOpReadVariableOp0basemodel_dense_1_matmul_readvariableop_resource*
_output_shapes
:	�T*
dtype02+
)basemodel/dense_1/MatMul_1/ReadVariableOp�
basemodel/dense_1/MatMul_1MatMul-basemodel/dense_1_dropout/Identity_1:output:01basemodel/dense_1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������T2
basemodel/dense_1/MatMul_1�
*basemodel/dense_1/BiasAdd_1/ReadVariableOpReadVariableOp1basemodel_dense_1_biasadd_readvariableop_resource*
_output_shapes
:T*
dtype02,
*basemodel/dense_1/BiasAdd_1/ReadVariableOp�
basemodel/dense_1/BiasAdd_1BiasAdd$basemodel/dense_1/MatMul_1:product:02basemodel/dense_1/BiasAdd_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������T2
basemodel/dense_1/BiasAdd_1�
:basemodel/batch_normalization_3/batchnorm_1/ReadVariableOpReadVariableOpAbasemodel_batch_normalization_3_batchnorm_readvariableop_resource*
_output_shapes
:T*
dtype02<
:basemodel/batch_normalization_3/batchnorm_1/ReadVariableOp�
1basemodel/batch_normalization_3/batchnorm_1/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:23
1basemodel/batch_normalization_3/batchnorm_1/add/y�
/basemodel/batch_normalization_3/batchnorm_1/addAddV2Bbasemodel/batch_normalization_3/batchnorm_1/ReadVariableOp:value:0:basemodel/batch_normalization_3/batchnorm_1/add/y:output:0*
T0*
_output_shapes
:T21
/basemodel/batch_normalization_3/batchnorm_1/add�
1basemodel/batch_normalization_3/batchnorm_1/RsqrtRsqrt3basemodel/batch_normalization_3/batchnorm_1/add:z:0*
T0*
_output_shapes
:T23
1basemodel/batch_normalization_3/batchnorm_1/Rsqrt�
>basemodel/batch_normalization_3/batchnorm_1/mul/ReadVariableOpReadVariableOpEbasemodel_batch_normalization_3_batchnorm_mul_readvariableop_resource*
_output_shapes
:T*
dtype02@
>basemodel/batch_normalization_3/batchnorm_1/mul/ReadVariableOp�
/basemodel/batch_normalization_3/batchnorm_1/mulMul5basemodel/batch_normalization_3/batchnorm_1/Rsqrt:y:0Fbasemodel/batch_normalization_3/batchnorm_1/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:T21
/basemodel/batch_normalization_3/batchnorm_1/mul�
1basemodel/batch_normalization_3/batchnorm_1/mul_1Mul$basemodel/dense_1/BiasAdd_1:output:03basemodel/batch_normalization_3/batchnorm_1/mul:z:0*
T0*'
_output_shapes
:���������T23
1basemodel/batch_normalization_3/batchnorm_1/mul_1�
<basemodel/batch_normalization_3/batchnorm_1/ReadVariableOp_1ReadVariableOpCbasemodel_batch_normalization_3_batchnorm_readvariableop_1_resource*
_output_shapes
:T*
dtype02>
<basemodel/batch_normalization_3/batchnorm_1/ReadVariableOp_1�
1basemodel/batch_normalization_3/batchnorm_1/mul_2MulDbasemodel/batch_normalization_3/batchnorm_1/ReadVariableOp_1:value:03basemodel/batch_normalization_3/batchnorm_1/mul:z:0*
T0*
_output_shapes
:T23
1basemodel/batch_normalization_3/batchnorm_1/mul_2�
<basemodel/batch_normalization_3/batchnorm_1/ReadVariableOp_2ReadVariableOpCbasemodel_batch_normalization_3_batchnorm_readvariableop_2_resource*
_output_shapes
:T*
dtype02>
<basemodel/batch_normalization_3/batchnorm_1/ReadVariableOp_2�
/basemodel/batch_normalization_3/batchnorm_1/subSubDbasemodel/batch_normalization_3/batchnorm_1/ReadVariableOp_2:value:05basemodel/batch_normalization_3/batchnorm_1/mul_2:z:0*
T0*
_output_shapes
:T21
/basemodel/batch_normalization_3/batchnorm_1/sub�
1basemodel/batch_normalization_3/batchnorm_1/add_1AddV25basemodel/batch_normalization_3/batchnorm_1/mul_1:z:03basemodel/batch_normalization_3/batchnorm_1/sub:z:0*
T0*'
_output_shapes
:���������T23
1basemodel/batch_normalization_3/batchnorm_1/add_1�
&basemodel/dense_activation_1/Sigmoid_1Sigmoid5basemodel/batch_normalization_3/batchnorm_1/add_1:z:0*
T0*'
_output_shapes
:���������T2(
&basemodel/dense_activation_1/Sigmoid_1�
distance/subSub(basemodel/dense_activation_1/Sigmoid:y:0*basemodel/dense_activation_1/Sigmoid_1:y:0*
T0*'
_output_shapes
:���������T2
distance/subp
distance/SquareSquaredistance/sub:z:0*
T0*'
_output_shapes
:���������T2
distance/Square�
distance/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
���������2 
distance/Sum/reduction_indices�
distance/SumSumdistance/Square:y:0'distance/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:���������*
	keep_dims(2
distance/Sume
distance/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
distance/Const�
distance/MaximumMaximumdistance/Sum:output:0distance/Const:output:0*
T0*'
_output_shapes
:���������2
distance/Maximumn
distance/SqrtSqrtdistance/Maximum:z:0*
T0*'
_output_shapes
:���������2
distance/Sqrt�
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpEbasemodel_stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype027
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp�
&stream_0_conv_1/kernel/Regularizer/AbsAbs=stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_0_conv_1/kernel/Regularizer/Abs�
(stream_0_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_0_conv_1/kernel/Regularizer/Const�
&stream_0_conv_1/kernel/Regularizer/SumSum*stream_0_conv_1/kernel/Regularizer/Abs:y:01stream_0_conv_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/Sum�
(stream_0_conv_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2*
(stream_0_conv_1/kernel/Regularizer/mul/x�
&stream_0_conv_1/kernel/Regularizer/mulMul1stream_0_conv_1/kernel/Regularizer/mul/x:output:0/stream_0_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/mul�
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpEbasemodel_stream_1_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02:
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp�
)stream_1_conv_1/kernel/Regularizer/SquareSquare@stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2+
)stream_1_conv_1/kernel/Regularizer/Square�
(stream_1_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_1_conv_1/kernel/Regularizer/Const�
&stream_1_conv_1/kernel/Regularizer/SumSum-stream_1_conv_1/kernel/Regularizer/Square:y:01stream_1_conv_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&stream_1_conv_1/kernel/Regularizer/Sum�
(stream_1_conv_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2*
(stream_1_conv_1/kernel/Regularizer/mul/x�
&stream_1_conv_1/kernel/Regularizer/mulMul1stream_1_conv_1/kernel/Regularizer/mul/x:output:0/stream_1_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_1_conv_1/kernel/Regularizer/mul�
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpEbasemodel_stream_2_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype027
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp�
&stream_2_conv_1/kernel/Regularizer/AbsAbs=stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_2_conv_1/kernel/Regularizer/Abs�
(stream_2_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_2_conv_1/kernel/Regularizer/Const�
&stream_2_conv_1/kernel/Regularizer/SumSum*stream_2_conv_1/kernel/Regularizer/Abs:y:01stream_2_conv_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&stream_2_conv_1/kernel/Regularizer/Sum�
(stream_2_conv_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2*
(stream_2_conv_1/kernel/Regularizer/mul/x�
&stream_2_conv_1/kernel/Regularizer/mulMul1stream_2_conv_1/kernel/Regularizer/mul/x:output:0/stream_2_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_2_conv_1/kernel/Regularizer/mul�
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp0basemodel_dense_1_matmul_readvariableop_resource*
_output_shapes
:	�T*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOp�
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	�T2 
dense_1/kernel/Regularizer/Abs�
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_1/kernel/Regularizer/Const�
dense_1/kernel/Regularizer/SumSum"dense_1/kernel/Regularizer/Abs:y:0)dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/Sum�
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2"
 dense_1/kernel/Regularizer/mul/x�
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/mull
IdentityIdentitydistance/Sqrt:y:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp7^basemodel/batch_normalization/batchnorm/ReadVariableOp9^basemodel/batch_normalization/batchnorm/ReadVariableOp_19^basemodel/batch_normalization/batchnorm/ReadVariableOp_2;^basemodel/batch_normalization/batchnorm/mul/ReadVariableOp9^basemodel/batch_normalization/batchnorm_1/ReadVariableOp;^basemodel/batch_normalization/batchnorm_1/ReadVariableOp_1;^basemodel/batch_normalization/batchnorm_1/ReadVariableOp_2=^basemodel/batch_normalization/batchnorm_1/mul/ReadVariableOp9^basemodel/batch_normalization_1/batchnorm/ReadVariableOp;^basemodel/batch_normalization_1/batchnorm/ReadVariableOp_1;^basemodel/batch_normalization_1/batchnorm/ReadVariableOp_2=^basemodel/batch_normalization_1/batchnorm/mul/ReadVariableOp;^basemodel/batch_normalization_1/batchnorm_1/ReadVariableOp=^basemodel/batch_normalization_1/batchnorm_1/ReadVariableOp_1=^basemodel/batch_normalization_1/batchnorm_1/ReadVariableOp_2?^basemodel/batch_normalization_1/batchnorm_1/mul/ReadVariableOp9^basemodel/batch_normalization_2/batchnorm/ReadVariableOp;^basemodel/batch_normalization_2/batchnorm/ReadVariableOp_1;^basemodel/batch_normalization_2/batchnorm/ReadVariableOp_2=^basemodel/batch_normalization_2/batchnorm/mul/ReadVariableOp;^basemodel/batch_normalization_2/batchnorm_1/ReadVariableOp=^basemodel/batch_normalization_2/batchnorm_1/ReadVariableOp_1=^basemodel/batch_normalization_2/batchnorm_1/ReadVariableOp_2?^basemodel/batch_normalization_2/batchnorm_1/mul/ReadVariableOp9^basemodel/batch_normalization_3/batchnorm/ReadVariableOp;^basemodel/batch_normalization_3/batchnorm/ReadVariableOp_1;^basemodel/batch_normalization_3/batchnorm/ReadVariableOp_2=^basemodel/batch_normalization_3/batchnorm/mul/ReadVariableOp;^basemodel/batch_normalization_3/batchnorm_1/ReadVariableOp=^basemodel/batch_normalization_3/batchnorm_1/ReadVariableOp_1=^basemodel/batch_normalization_3/batchnorm_1/ReadVariableOp_2?^basemodel/batch_normalization_3/batchnorm_1/mul/ReadVariableOp)^basemodel/dense_1/BiasAdd/ReadVariableOp+^basemodel/dense_1/BiasAdd_1/ReadVariableOp(^basemodel/dense_1/MatMul/ReadVariableOp*^basemodel/dense_1/MatMul_1/ReadVariableOp1^basemodel/stream_0_conv_1/BiasAdd/ReadVariableOp3^basemodel/stream_0_conv_1/BiasAdd_1/ReadVariableOp=^basemodel/stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp?^basemodel/stream_0_conv_1/conv1d_1/ExpandDims_1/ReadVariableOp1^basemodel/stream_1_conv_1/BiasAdd/ReadVariableOp3^basemodel/stream_1_conv_1/BiasAdd_1/ReadVariableOp=^basemodel/stream_1_conv_1/conv1d/ExpandDims_1/ReadVariableOp?^basemodel/stream_1_conv_1/conv1d_1/ExpandDims_1/ReadVariableOp1^basemodel/stream_2_conv_1/BiasAdd/ReadVariableOp3^basemodel/stream_2_conv_1/BiasAdd_1/ReadVariableOp=^basemodel/stream_2_conv_1/conv1d/ExpandDims_1/ReadVariableOp?^basemodel/stream_2_conv_1/conv1d_1/ExpandDims_1/ReadVariableOp.^dense_1/kernel/Regularizer/Abs/ReadVariableOp6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp9^stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp6^stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*s
_input_shapesb
`:����������:����������: : : : : : : : : : : : : : : : : : : : : : : : 2p
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
<basemodel/batch_normalization_1/batchnorm_1/ReadVariableOp_2<basemodel/batch_normalization_1/batchnorm_1/ReadVariableOp_22�
>basemodel/batch_normalization_1/batchnorm_1/mul/ReadVariableOp>basemodel/batch_normalization_1/batchnorm_1/mul/ReadVariableOp2t
8basemodel/batch_normalization_2/batchnorm/ReadVariableOp8basemodel/batch_normalization_2/batchnorm/ReadVariableOp2x
:basemodel/batch_normalization_2/batchnorm/ReadVariableOp_1:basemodel/batch_normalization_2/batchnorm/ReadVariableOp_12x
:basemodel/batch_normalization_2/batchnorm/ReadVariableOp_2:basemodel/batch_normalization_2/batchnorm/ReadVariableOp_22|
<basemodel/batch_normalization_2/batchnorm/mul/ReadVariableOp<basemodel/batch_normalization_2/batchnorm/mul/ReadVariableOp2x
:basemodel/batch_normalization_2/batchnorm_1/ReadVariableOp:basemodel/batch_normalization_2/batchnorm_1/ReadVariableOp2|
<basemodel/batch_normalization_2/batchnorm_1/ReadVariableOp_1<basemodel/batch_normalization_2/batchnorm_1/ReadVariableOp_12|
<basemodel/batch_normalization_2/batchnorm_1/ReadVariableOp_2<basemodel/batch_normalization_2/batchnorm_1/ReadVariableOp_22�
>basemodel/batch_normalization_2/batchnorm_1/mul/ReadVariableOp>basemodel/batch_normalization_2/batchnorm_1/mul/ReadVariableOp2t
8basemodel/batch_normalization_3/batchnorm/ReadVariableOp8basemodel/batch_normalization_3/batchnorm/ReadVariableOp2x
:basemodel/batch_normalization_3/batchnorm/ReadVariableOp_1:basemodel/batch_normalization_3/batchnorm/ReadVariableOp_12x
:basemodel/batch_normalization_3/batchnorm/ReadVariableOp_2:basemodel/batch_normalization_3/batchnorm/ReadVariableOp_22|
<basemodel/batch_normalization_3/batchnorm/mul/ReadVariableOp<basemodel/batch_normalization_3/batchnorm/mul/ReadVariableOp2x
:basemodel/batch_normalization_3/batchnorm_1/ReadVariableOp:basemodel/batch_normalization_3/batchnorm_1/ReadVariableOp2|
<basemodel/batch_normalization_3/batchnorm_1/ReadVariableOp_1<basemodel/batch_normalization_3/batchnorm_1/ReadVariableOp_12|
<basemodel/batch_normalization_3/batchnorm_1/ReadVariableOp_2<basemodel/batch_normalization_3/batchnorm_1/ReadVariableOp_22�
>basemodel/batch_normalization_3/batchnorm_1/mul/ReadVariableOp>basemodel/batch_normalization_3/batchnorm_1/mul/ReadVariableOp2T
(basemodel/dense_1/BiasAdd/ReadVariableOp(basemodel/dense_1/BiasAdd/ReadVariableOp2X
*basemodel/dense_1/BiasAdd_1/ReadVariableOp*basemodel/dense_1/BiasAdd_1/ReadVariableOp2R
'basemodel/dense_1/MatMul/ReadVariableOp'basemodel/dense_1/MatMul/ReadVariableOp2V
)basemodel/dense_1/MatMul_1/ReadVariableOp)basemodel/dense_1/MatMul_1/ReadVariableOp2d
0basemodel/stream_0_conv_1/BiasAdd/ReadVariableOp0basemodel/stream_0_conv_1/BiasAdd/ReadVariableOp2h
2basemodel/stream_0_conv_1/BiasAdd_1/ReadVariableOp2basemodel/stream_0_conv_1/BiasAdd_1/ReadVariableOp2|
<basemodel/stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp<basemodel/stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp2�
>basemodel/stream_0_conv_1/conv1d_1/ExpandDims_1/ReadVariableOp>basemodel/stream_0_conv_1/conv1d_1/ExpandDims_1/ReadVariableOp2d
0basemodel/stream_1_conv_1/BiasAdd/ReadVariableOp0basemodel/stream_1_conv_1/BiasAdd/ReadVariableOp2h
2basemodel/stream_1_conv_1/BiasAdd_1/ReadVariableOp2basemodel/stream_1_conv_1/BiasAdd_1/ReadVariableOp2|
<basemodel/stream_1_conv_1/conv1d/ExpandDims_1/ReadVariableOp<basemodel/stream_1_conv_1/conv1d/ExpandDims_1/ReadVariableOp2�
>basemodel/stream_1_conv_1/conv1d_1/ExpandDims_1/ReadVariableOp>basemodel/stream_1_conv_1/conv1d_1/ExpandDims_1/ReadVariableOp2d
0basemodel/stream_2_conv_1/BiasAdd/ReadVariableOp0basemodel/stream_2_conv_1/BiasAdd/ReadVariableOp2h
2basemodel/stream_2_conv_1/BiasAdd_1/ReadVariableOp2basemodel/stream_2_conv_1/BiasAdd_1/ReadVariableOp2|
<basemodel/stream_2_conv_1/conv1d/ExpandDims_1/ReadVariableOp<basemodel/stream_2_conv_1/conv1d/ExpandDims_1/ReadVariableOp2�
>basemodel/stream_2_conv_1/conv1d_1/ExpandDims_1/ReadVariableOp>basemodel/stream_2_conv_1/conv1d_1/ExpandDims_1/ReadVariableOp2^
-dense_1/kernel/Regularizer/Abs/ReadVariableOp-dense_1/kernel/Regularizer/Abs/ReadVariableOp2n
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp2t
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp2n
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp:V R
,
_output_shapes
:����������
"
_user_specified_name
inputs/0:VR
,
_output_shapes
:����������
"
_user_specified_name
inputs/1
�
�
%__inference_signature_wrapper_1986661
left_inputs
right_inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:@
	unknown_5:@
	unknown_6:@
	unknown_7:@
	unknown_8:@
	unknown_9:@

unknown_10:@

unknown_11:@

unknown_12:@

unknown_13:@

unknown_14:@

unknown_15:@

unknown_16:@

unknown_17:	�T

unknown_18:T

unknown_19:T

unknown_20:T

unknown_21:T

unknown_22:T
identity��StatefulPartitionedCall�
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
:���������*:
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *+
f&R$
"__inference__wrapped_model_19839012
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*s
_input_shapesb
`:����������:����������: : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
,
_output_shapes
:����������
%
_user_specified_nameleft_inputs:ZV
,
_output_shapes
:����������
&
_user_specified_nameright_inputs
�
�
+__inference_basemodel_layer_call_fn_1985701
inputs_0
inputs_1
inputs_2
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:@
	unknown_5:@
	unknown_6:@
	unknown_7:@
	unknown_8:@
	unknown_9:@

unknown_10:@

unknown_11:@

unknown_12:@

unknown_13:@

unknown_14:@

unknown_15:@

unknown_16:@

unknown_17:	�T

unknown_18:T

unknown_19:T

unknown_20:T

unknown_21:T

unknown_22:T
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1inputs_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_22*&
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������T*2
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_basemodel_layer_call_and_return_conditional_losses_19855952
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������T2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapesz
x:����������:����������:����������: : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
,
_output_shapes
:����������
"
_user_specified_name
inputs_0:VR
,
_output_shapes
:����������
"
_user_specified_name
inputs_1:VR
,
_output_shapes
:����������
"
_user_specified_name
inputs_2
�
J
.__inference_activation_1_layer_call_fn_1988730

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_activation_1_layer_call_and_return_conditional_losses_19848332
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:����������@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������@:T P
,
_output_shapes
:����������@
 
_user_specified_nameinputs
�
�
+__inference_basemodel_layer_call_fn_1987554
inputs_0
inputs_1
inputs_2
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:@
	unknown_5:@
	unknown_6:@
	unknown_7:@
	unknown_8:@
	unknown_9:@

unknown_10:@

unknown_11:@

unknown_12:@

unknown_13:@

unknown_14:@

unknown_15:@

unknown_16:@

unknown_17:	�T

unknown_18:T

unknown_19:T

unknown_20:T

unknown_21:T

unknown_22:T
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1inputs_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_22*&
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������T*:
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_basemodel_layer_call_and_return_conditional_losses_19849642
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������T2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapesz
x:����������:����������:����������: : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
,
_output_shapes
:����������
"
_user_specified_name
inputs/0:VR
,
_output_shapes
:����������
"
_user_specified_name
inputs/1:VR
,
_output_shapes
:����������
"
_user_specified_name
inputs/2
�*
�
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1989037

inputs5
'assignmovingavg_readvariableop_resource:T7
)assignmovingavg_1_readvariableop_resource:T3
%batchnorm_mul_readvariableop_resource:T/
!batchnorm_readvariableop_resource:T
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOp�
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices�
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
moments/StopGradient�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:���������T2
moments/SquaredDifference�
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices�
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:T*
	keep_dims(2
moments/variance�
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:T*
squeeze_dims
 2
moments/Squeeze�
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
�#<2
AssignMovingAvg/decay�
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:T*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:T2
AssignMovingAvg/sub�
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:T2
AssignMovingAvg/mul�
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
�#<2
AssignMovingAvg_1/decay�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:T*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:T2
AssignMovingAvg_1/sub�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:T2
AssignMovingAvg_1/mul�
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
 *o�:2
batchnorm/add/y�
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:T2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:T2
batchnorm/Rsqrt�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:T*
dtype02
batchnorm/mul/ReadVariableOp�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:T2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������T2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:T2
batchnorm/mul_2�
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:T*
dtype02
batchnorm/ReadVariableOp�
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:T2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������T2
batchnorm/add_1n
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������T2

Identity�
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������T: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������T
 
_user_specified_nameinputs
�
�
F__inference_basemodel_layer_call_and_return_conditional_losses_1984964

inputs
inputs_1
inputs_2-
stream_2_conv_1_1984675:@%
stream_2_conv_1_1984677:@-
stream_1_conv_1_1984702:@%
stream_1_conv_1_1984704:@-
stream_0_conv_1_1984729:@%
stream_0_conv_1_1984731:@+
batch_normalization_2_1984754:@+
batch_normalization_2_1984756:@+
batch_normalization_2_1984758:@+
batch_normalization_2_1984760:@+
batch_normalization_1_1984783:@+
batch_normalization_1_1984785:@+
batch_normalization_1_1984787:@+
batch_normalization_1_1984789:@)
batch_normalization_1984812:@)
batch_normalization_1984814:@)
batch_normalization_1984816:@)
batch_normalization_1984818:@"
dense_1_1984918:	�T
dense_1_1984920:T+
batch_normalization_3_1984923:T+
batch_normalization_3_1984925:T+
batch_normalization_3_1984927:T+
batch_normalization_3_1984929:T
identity��+batch_normalization/StatefulPartitionedCall�-batch_normalization_1/StatefulPartitionedCall�-batch_normalization_2/StatefulPartitionedCall�-batch_normalization_3/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�-dense_1/kernel/Regularizer/Abs/ReadVariableOp�'stream_0_conv_1/StatefulPartitionedCall�5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp�'stream_1_conv_1/StatefulPartitionedCall�8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp�'stream_2_conv_1/StatefulPartitionedCall�5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp�
#stream_2_input_drop/PartitionedCallPartitionedCallinputs_2*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *Y
fTRR
P__inference_stream_2_input_drop_layer_call_and_return_conditional_losses_19846372%
#stream_2_input_drop/PartitionedCall�
#stream_1_input_drop/PartitionedCallPartitionedCallinputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *Y
fTRR
P__inference_stream_1_input_drop_layer_call_and_return_conditional_losses_19846442%
#stream_1_input_drop/PartitionedCall�
#stream_0_input_drop/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *Y
fTRR
P__inference_stream_0_input_drop_layer_call_and_return_conditional_losses_19846512%
#stream_0_input_drop/PartitionedCall�
'stream_2_conv_1/StatefulPartitionedCallStatefulPartitionedCall,stream_2_input_drop/PartitionedCall:output:0stream_2_conv_1_1984675stream_2_conv_1_1984677*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_stream_2_conv_1_layer_call_and_return_conditional_losses_19846742)
'stream_2_conv_1/StatefulPartitionedCall�
'stream_1_conv_1/StatefulPartitionedCallStatefulPartitionedCall,stream_1_input_drop/PartitionedCall:output:0stream_1_conv_1_1984702stream_1_conv_1_1984704*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_stream_1_conv_1_layer_call_and_return_conditional_losses_19847012)
'stream_1_conv_1/StatefulPartitionedCall�
'stream_0_conv_1/StatefulPartitionedCallStatefulPartitionedCall,stream_0_input_drop/PartitionedCall:output:0stream_0_conv_1_1984729stream_0_conv_1_1984731*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_stream_0_conv_1_layer_call_and_return_conditional_losses_19847282)
'stream_0_conv_1/StatefulPartitionedCall�
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall0stream_2_conv_1/StatefulPartitionedCall:output:0batch_normalization_2_1984754batch_normalization_2_1984756batch_normalization_2_1984758batch_normalization_2_1984760*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *[
fVRT
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_19847532/
-batch_normalization_2/StatefulPartitionedCall�
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall0stream_1_conv_1/StatefulPartitionedCall:output:0batch_normalization_1_1984783batch_normalization_1_1984785batch_normalization_1_1984787batch_normalization_1_1984789*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *[
fVRT
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_19847822/
-batch_normalization_1/StatefulPartitionedCall�
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall0stream_0_conv_1/StatefulPartitionedCall:output:0batch_normalization_1984812batch_normalization_1984814batch_normalization_1984816batch_normalization_1984818*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_19848112-
+batch_normalization/StatefulPartitionedCall�
activation_2/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_activation_2_layer_call_and_return_conditional_losses_19848262
activation_2/PartitionedCall�
activation_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_activation_1_layer_call_and_return_conditional_losses_19848332
activation_1/PartitionedCall�
activation/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_activation_layer_call_and_return_conditional_losses_19848402
activation/PartitionedCall�
stream_2_drop_1/PartitionedCallPartitionedCall%activation_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_stream_2_drop_1_layer_call_and_return_conditional_losses_19848472!
stream_2_drop_1/PartitionedCall�
stream_1_drop_1/PartitionedCallPartitionedCall%activation_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_stream_1_drop_1_layer_call_and_return_conditional_losses_19848542!
stream_1_drop_1/PartitionedCall�
stream_0_drop_1/PartitionedCallPartitionedCall#activation/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_stream_0_drop_1_layer_call_and_return_conditional_losses_19848612!
stream_0_drop_1/PartitionedCall�
(global_average_pooling1d/PartitionedCallPartitionedCall(stream_0_drop_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *^
fYRW
U__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_19848682*
(global_average_pooling1d/PartitionedCall�
*global_average_pooling1d_1/PartitionedCallPartitionedCall(stream_1_drop_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *`
f[RY
W__inference_global_average_pooling1d_1_layer_call_and_return_conditional_losses_19848752,
*global_average_pooling1d_1/PartitionedCall�
*global_average_pooling1d_2/PartitionedCallPartitionedCall(stream_2_drop_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *`
f[RY
W__inference_global_average_pooling1d_2_layer_call_and_return_conditional_losses_19848822,
*global_average_pooling1d_2/PartitionedCall�
concatenate/PartitionedCallPartitionedCall1global_average_pooling1d/PartitionedCall:output:03global_average_pooling1d_1/PartitionedCall:output:03global_average_pooling1d_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_concatenate_layer_call_and_return_conditional_losses_19848922
concatenate/PartitionedCall�
dense_1_dropout/PartitionedCallPartitionedCall$concatenate/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_dense_1_dropout_layer_call_and_return_conditional_losses_19848992!
dense_1_dropout/PartitionedCall�
dense_1/StatefulPartitionedCallStatefulPartitionedCall(dense_1_dropout/PartitionedCall:output:0dense_1_1984918dense_1_1984920*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������T*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_19849172!
dense_1/StatefulPartitionedCall�
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0batch_normalization_3_1984923batch_normalization_3_1984925batch_normalization_3_1984927batch_normalization_3_1984929*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������T*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *[
fVRT
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_19844832/
-batch_normalization_3/StatefulPartitionedCall�
"dense_activation_1/PartitionedCallPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������T* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *X
fSRQ
O__inference_dense_activation_1_layer_call_and_return_conditional_losses_19849372$
"dense_activation_1/PartitionedCall�
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpstream_0_conv_1_1984729*"
_output_shapes
:@*
dtype027
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp�
&stream_0_conv_1/kernel/Regularizer/AbsAbs=stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_0_conv_1/kernel/Regularizer/Abs�
(stream_0_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_0_conv_1/kernel/Regularizer/Const�
&stream_0_conv_1/kernel/Regularizer/SumSum*stream_0_conv_1/kernel/Regularizer/Abs:y:01stream_0_conv_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/Sum�
(stream_0_conv_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2*
(stream_0_conv_1/kernel/Regularizer/mul/x�
&stream_0_conv_1/kernel/Regularizer/mulMul1stream_0_conv_1/kernel/Regularizer/mul/x:output:0/stream_0_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/mul�
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpstream_1_conv_1_1984702*"
_output_shapes
:@*
dtype02:
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp�
)stream_1_conv_1/kernel/Regularizer/SquareSquare@stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2+
)stream_1_conv_1/kernel/Regularizer/Square�
(stream_1_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_1_conv_1/kernel/Regularizer/Const�
&stream_1_conv_1/kernel/Regularizer/SumSum-stream_1_conv_1/kernel/Regularizer/Square:y:01stream_1_conv_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&stream_1_conv_1/kernel/Regularizer/Sum�
(stream_1_conv_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2*
(stream_1_conv_1/kernel/Regularizer/mul/x�
&stream_1_conv_1/kernel/Regularizer/mulMul1stream_1_conv_1/kernel/Regularizer/mul/x:output:0/stream_1_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_1_conv_1/kernel/Regularizer/mul�
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpstream_2_conv_1_1984675*"
_output_shapes
:@*
dtype027
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp�
&stream_2_conv_1/kernel/Regularizer/AbsAbs=stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_2_conv_1/kernel/Regularizer/Abs�
(stream_2_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_2_conv_1/kernel/Regularizer/Const�
&stream_2_conv_1/kernel/Regularizer/SumSum*stream_2_conv_1/kernel/Regularizer/Abs:y:01stream_2_conv_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&stream_2_conv_1/kernel/Regularizer/Sum�
(stream_2_conv_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2*
(stream_2_conv_1/kernel/Regularizer/mul/x�
&stream_2_conv_1/kernel/Regularizer/mulMul1stream_2_conv_1/kernel/Regularizer/mul/x:output:0/stream_2_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_2_conv_1/kernel/Regularizer/mul�
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_1_1984918*
_output_shapes
:	�T*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOp�
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	�T2 
dense_1/kernel/Regularizer/Abs�
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_1/kernel/Regularizer/Const�
dense_1/kernel/Regularizer/SumSum"dense_1/kernel/Regularizer/Abs:y:0)dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/Sum�
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2"
 dense_1/kernel/Regularizer/mul/x�
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/mul�
IdentityIdentity+dense_activation_1/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������T2

Identity�
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall.^dense_1/kernel/Regularizer/Abs/ReadVariableOp(^stream_0_conv_1/StatefulPartitionedCall6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp(^stream_1_conv_1/StatefulPartitionedCall9^stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp(^stream_2_conv_1/StatefulPartitionedCall6^stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapesz
x:����������:����������:����������: : : : : : : : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2^
-dense_1/kernel/Regularizer/Abs/ReadVariableOp-dense_1/kernel/Regularizer/Abs/ReadVariableOp2R
'stream_0_conv_1/StatefulPartitionedCall'stream_0_conv_1/StatefulPartitionedCall2n
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp2R
'stream_1_conv_1/StatefulPartitionedCall'stream_1_conv_1/StatefulPartitionedCall2t
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp2R
'stream_2_conv_1/StatefulPartitionedCall'stream_2_conv_1/StatefulPartitionedCall2n
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs:TP
,
_output_shapes
:����������
 
_user_specified_nameinputs:TP
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
j
L__inference_stream_0_drop_1_layer_call_and_return_conditional_losses_1984861

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:����������@2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:����������@2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������@:T P
,
_output_shapes
:����������@
 
_user_specified_nameinputs
�
X
<__inference_global_average_pooling1d_2_layer_call_fn_1988880

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *`
f[RY
W__inference_global_average_pooling1d_2_layer_call_and_return_conditional_losses_19848822
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������@:T P
,
_output_shapes
:����������@
 
_user_specified_nameinputs
�
�
H__inference_concatenate_layer_call_and_return_conditional_losses_1984892

inputs
inputs_1
inputs_2
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis�
concatConcatV2inputsinputs_1inputs_2concat/axis:output:0*
N*
T0*(
_output_shapes
:����������2
concatd
IdentityIdentityconcat:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:���������@:���������@:���������@:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������@
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�	
�
7__inference_batch_normalization_2_layer_call_fn_1988581

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *[
fVRT
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_19843092
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :������������������@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:������������������@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :������������������@
 
_user_specified_nameinputs
�
q
U__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_1984397

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
:������������������2
Meanj
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:������������������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'���������������������������:e a
=
_output_shapes+
):'���������������������������
 
_user_specified_nameinputs
�+
�
P__inference_batch_normalization_layer_call_and_return_conditional_losses_1988341

inputs5
'assignmovingavg_readvariableop_resource:@7
)assignmovingavg_1_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@/
!batchnorm_readvariableop_resource:@
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOp�
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indices�
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(2
moments/mean�
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:@2
moments/StopGradient�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :������������������@2
moments/SquaredDifference�
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices�
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(2
moments/variance�
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 2
moments/Squeeze�
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
�#<2
AssignMovingAvg/decay�
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg/sub�
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg/mul�
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
�#<2
AssignMovingAvg_1/decay�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg_1/sub�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg_1/mul�
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
 *o�:2
batchnorm/add/y�
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@2
batchnorm/Rsqrt�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOp�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2
batchnorm/mul�
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :������������������@2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:@2
batchnorm/mul_2�
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp�
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :������������������@2
batchnorm/add_1{
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :������������������@2

Identity�
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:������������������@: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :������������������@
 
_user_specified_nameinputs
�
n
P__inference_stream_2_input_drop_layer_call_and_return_conditional_losses_1984637

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:����������2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:����������2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
ǀ
�
 __inference__traced_save_1989304
file_prefix%
!savev2_beta_1_read_readvariableop%
!savev2_beta_2_read_readvariableop$
 savev2_decay_read_readvariableop,
(savev2_learning_rate_read_readvariableop(
$savev2_adam_iter_read_readvariableop	5
1savev2_stream_0_conv_1_kernel_read_readvariableop3
/savev2_stream_0_conv_1_bias_read_readvariableop5
1savev2_stream_1_conv_1_kernel_read_readvariableop3
/savev2_stream_1_conv_1_bias_read_readvariableop5
1savev2_stream_2_conv_1_kernel_read_readvariableop3
/savev2_stream_2_conv_1_bias_read_readvariableop8
4savev2_batch_normalization_gamma_read_readvariableop7
3savev2_batch_normalization_beta_read_readvariableop:
6savev2_batch_normalization_1_gamma_read_readvariableop9
5savev2_batch_normalization_1_beta_read_readvariableop:
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
6savev2_adam_stream_0_conv_1_bias_m_read_readvariableop<
8savev2_adam_stream_1_conv_1_kernel_m_read_readvariableop:
6savev2_adam_stream_1_conv_1_bias_m_read_readvariableop<
8savev2_adam_stream_2_conv_1_kernel_m_read_readvariableop:
6savev2_adam_stream_2_conv_1_bias_m_read_readvariableop?
;savev2_adam_batch_normalization_gamma_m_read_readvariableop>
:savev2_adam_batch_normalization_beta_m_read_readvariableopA
=savev2_adam_batch_normalization_1_gamma_m_read_readvariableop@
<savev2_adam_batch_normalization_1_beta_m_read_readvariableopA
=savev2_adam_batch_normalization_2_gamma_m_read_readvariableop@
<savev2_adam_batch_normalization_2_beta_m_read_readvariableop4
0savev2_adam_dense_1_kernel_m_read_readvariableop2
.savev2_adam_dense_1_bias_m_read_readvariableopA
=savev2_adam_batch_normalization_3_gamma_m_read_readvariableop@
<savev2_adam_batch_normalization_3_beta_m_read_readvariableop<
8savev2_adam_stream_0_conv_1_kernel_v_read_readvariableop:
6savev2_adam_stream_0_conv_1_bias_v_read_readvariableop<
8savev2_adam_stream_1_conv_1_kernel_v_read_readvariableop:
6savev2_adam_stream_1_conv_1_bias_v_read_readvariableop<
8savev2_adam_stream_2_conv_1_kernel_v_read_readvariableop:
6savev2_adam_stream_2_conv_1_bias_v_read_readvariableop?
;savev2_adam_batch_normalization_gamma_v_read_readvariableop>
:savev2_adam_batch_normalization_beta_v_read_readvariableopA
=savev2_adam_batch_normalization_1_gamma_v_read_readvariableop@
<savev2_adam_batch_normalization_1_beta_v_read_readvariableopA
=savev2_adam_batch_normalization_2_gamma_v_read_readvariableop@
<savev2_adam_batch_normalization_2_beta_v_read_readvariableop4
0savev2_adam_dense_1_kernel_v_read_readvariableop2
.savev2_adam_dense_1_bias_v_read_readvariableopA
=savev2_adam_batch_normalization_3_gamma_v_read_readvariableop@
<savev2_adam_batch_normalization_3_beta_v_read_readvariableop
savev2_const

identity_1��MergeV2Checkpoints�
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
Const_1�
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
ShardedFilename/shard�
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename� 
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:@*
dtype0*�
value�B�@B+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/14/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/15/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:@*
dtype0*�
value�B�@B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices�
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0!savev2_beta_1_read_readvariableop!savev2_beta_2_read_readvariableop savev2_decay_read_readvariableop(savev2_learning_rate_read_readvariableop$savev2_adam_iter_read_readvariableop1savev2_stream_0_conv_1_kernel_read_readvariableop/savev2_stream_0_conv_1_bias_read_readvariableop1savev2_stream_1_conv_1_kernel_read_readvariableop/savev2_stream_1_conv_1_bias_read_readvariableop1savev2_stream_2_conv_1_kernel_read_readvariableop/savev2_stream_2_conv_1_bias_read_readvariableop4savev2_batch_normalization_gamma_read_readvariableop3savev2_batch_normalization_beta_read_readvariableop6savev2_batch_normalization_1_gamma_read_readvariableop5savev2_batch_normalization_1_beta_read_readvariableop6savev2_batch_normalization_2_gamma_read_readvariableop5savev2_batch_normalization_2_beta_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop6savev2_batch_normalization_3_gamma_read_readvariableop5savev2_batch_normalization_3_beta_read_readvariableop:savev2_batch_normalization_moving_mean_read_readvariableop>savev2_batch_normalization_moving_variance_read_readvariableop<savev2_batch_normalization_1_moving_mean_read_readvariableop@savev2_batch_normalization_1_moving_variance_read_readvariableop<savev2_batch_normalization_2_moving_mean_read_readvariableop@savev2_batch_normalization_2_moving_variance_read_readvariableop<savev2_batch_normalization_3_moving_mean_read_readvariableop@savev2_batch_normalization_3_moving_variance_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop8savev2_adam_stream_0_conv_1_kernel_m_read_readvariableop6savev2_adam_stream_0_conv_1_bias_m_read_readvariableop8savev2_adam_stream_1_conv_1_kernel_m_read_readvariableop6savev2_adam_stream_1_conv_1_bias_m_read_readvariableop8savev2_adam_stream_2_conv_1_kernel_m_read_readvariableop6savev2_adam_stream_2_conv_1_bias_m_read_readvariableop;savev2_adam_batch_normalization_gamma_m_read_readvariableop:savev2_adam_batch_normalization_beta_m_read_readvariableop=savev2_adam_batch_normalization_1_gamma_m_read_readvariableop<savev2_adam_batch_normalization_1_beta_m_read_readvariableop=savev2_adam_batch_normalization_2_gamma_m_read_readvariableop<savev2_adam_batch_normalization_2_beta_m_read_readvariableop0savev2_adam_dense_1_kernel_m_read_readvariableop.savev2_adam_dense_1_bias_m_read_readvariableop=savev2_adam_batch_normalization_3_gamma_m_read_readvariableop<savev2_adam_batch_normalization_3_beta_m_read_readvariableop8savev2_adam_stream_0_conv_1_kernel_v_read_readvariableop6savev2_adam_stream_0_conv_1_bias_v_read_readvariableop8savev2_adam_stream_1_conv_1_kernel_v_read_readvariableop6savev2_adam_stream_1_conv_1_bias_v_read_readvariableop8savev2_adam_stream_2_conv_1_kernel_v_read_readvariableop6savev2_adam_stream_2_conv_1_bias_v_read_readvariableop;savev2_adam_batch_normalization_gamma_v_read_readvariableop:savev2_adam_batch_normalization_beta_v_read_readvariableop=savev2_adam_batch_normalization_1_gamma_v_read_readvariableop<savev2_adam_batch_normalization_1_beta_v_read_readvariableop=savev2_adam_batch_normalization_2_gamma_v_read_readvariableop<savev2_adam_batch_normalization_2_beta_v_read_readvariableop0savev2_adam_dense_1_kernel_v_read_readvariableop.savev2_adam_dense_1_bias_v_read_readvariableop=savev2_adam_batch_normalization_3_gamma_v_read_readvariableop<savev2_adam_batch_normalization_3_beta_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *N
dtypesD
B2@	2
SaveV2�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes�
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

identity_1Identity_1:output:0*�
_input_shapes�
�: : : : : : :@:@:@:@:@:@:@:@:@:@:@:@:	�T:T:T:T:@:@:@:@:@:@:T:T: : :@:@:@:@:@:@:@:@:@:@:@:@:	�T:T:T:T:@:@:@:@:@:@:@:@:@:@:@:@:	�T:T:T:T: 2(
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
:@:($
"
_output_shapes
:@: 	

_output_shapes
:@:(
$
"
_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@:%!

_output_shapes
:	�T: 
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
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 
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
:@:("$
"
_output_shapes
:@: #

_output_shapes
:@:($$
"
_output_shapes
:@: %

_output_shapes
:@: &

_output_shapes
:@: '

_output_shapes
:@: (

_output_shapes
:@: )

_output_shapes
:@: *

_output_shapes
:@: +

_output_shapes
:@:%,!

_output_shapes
:	�T: -
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
:@:(2$
"
_output_shapes
:@: 3

_output_shapes
:@:(4$
"
_output_shapes
:@: 5

_output_shapes
:@: 6

_output_shapes
:@: 7

_output_shapes
:@: 8

_output_shapes
:@: 9

_output_shapes
:@: :

_output_shapes
:@: ;

_output_shapes
:@:%<!

_output_shapes
:	�T: =
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
�
�
P__inference_batch_normalization_layer_call_and_return_conditional_losses_1983925

inputs/
!batchnorm_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@1
#batchnorm_readvariableop_1_resource:@1
#batchnorm_readvariableop_2_resource:@
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOp�
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
 *o�:2
batchnorm/add/y�
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@2
batchnorm/Rsqrt�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOp�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2
batchnorm/mul�
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :������������������@2
batchnorm/mul_1�
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp_1�
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:@2
batchnorm/mul_2�
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp_2�
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :������������������@2
batchnorm/add_1{
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :������������������@2

Identity�
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:������������������@: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :������������������@
 
_user_specified_nameinputs
�
e
I__inference_activation_1_layer_call_and_return_conditional_losses_1984833

inputs
identityS
ReluReluinputs*
T0*,
_output_shapes
:����������@2
Reluk
IdentityIdentityRelu:activations:0*
T0*,
_output_shapes
:����������@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������@:T P
,
_output_shapes
:����������@
 
_user_specified_nameinputs
�
V
:__inference_global_average_pooling1d_layer_call_fn_1988831

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:������������������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *^
fYRW
U__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_19843972
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:������������������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'���������������������������:e a
=
_output_shapes+
):'���������������������������
 
_user_specified_nameinputs
߆
�)
"__inference__wrapped_model_1983901
left_inputs
right_inputsa
Kmodel_basemodel_stream_2_conv_1_conv1d_expanddims_1_readvariableop_resource:@M
?model_basemodel_stream_2_conv_1_biasadd_readvariableop_resource:@a
Kmodel_basemodel_stream_1_conv_1_conv1d_expanddims_1_readvariableop_resource:@M
?model_basemodel_stream_1_conv_1_biasadd_readvariableop_resource:@a
Kmodel_basemodel_stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource:@M
?model_basemodel_stream_0_conv_1_biasadd_readvariableop_resource:@U
Gmodel_basemodel_batch_normalization_2_batchnorm_readvariableop_resource:@Y
Kmodel_basemodel_batch_normalization_2_batchnorm_mul_readvariableop_resource:@W
Imodel_basemodel_batch_normalization_2_batchnorm_readvariableop_1_resource:@W
Imodel_basemodel_batch_normalization_2_batchnorm_readvariableop_2_resource:@U
Gmodel_basemodel_batch_normalization_1_batchnorm_readvariableop_resource:@Y
Kmodel_basemodel_batch_normalization_1_batchnorm_mul_readvariableop_resource:@W
Imodel_basemodel_batch_normalization_1_batchnorm_readvariableop_1_resource:@W
Imodel_basemodel_batch_normalization_1_batchnorm_readvariableop_2_resource:@S
Emodel_basemodel_batch_normalization_batchnorm_readvariableop_resource:@W
Imodel_basemodel_batch_normalization_batchnorm_mul_readvariableop_resource:@U
Gmodel_basemodel_batch_normalization_batchnorm_readvariableop_1_resource:@U
Gmodel_basemodel_batch_normalization_batchnorm_readvariableop_2_resource:@I
6model_basemodel_dense_1_matmul_readvariableop_resource:	�TE
7model_basemodel_dense_1_biasadd_readvariableop_resource:TU
Gmodel_basemodel_batch_normalization_3_batchnorm_readvariableop_resource:TY
Kmodel_basemodel_batch_normalization_3_batchnorm_mul_readvariableop_resource:TW
Imodel_basemodel_batch_normalization_3_batchnorm_readvariableop_1_resource:TW
Imodel_basemodel_batch_normalization_3_batchnorm_readvariableop_2_resource:T
identity��<model/basemodel/batch_normalization/batchnorm/ReadVariableOp�>model/basemodel/batch_normalization/batchnorm/ReadVariableOp_1�>model/basemodel/batch_normalization/batchnorm/ReadVariableOp_2�@model/basemodel/batch_normalization/batchnorm/mul/ReadVariableOp�>model/basemodel/batch_normalization/batchnorm_1/ReadVariableOp�@model/basemodel/batch_normalization/batchnorm_1/ReadVariableOp_1�@model/basemodel/batch_normalization/batchnorm_1/ReadVariableOp_2�Bmodel/basemodel/batch_normalization/batchnorm_1/mul/ReadVariableOp�>model/basemodel/batch_normalization_1/batchnorm/ReadVariableOp�@model/basemodel/batch_normalization_1/batchnorm/ReadVariableOp_1�@model/basemodel/batch_normalization_1/batchnorm/ReadVariableOp_2�Bmodel/basemodel/batch_normalization_1/batchnorm/mul/ReadVariableOp�@model/basemodel/batch_normalization_1/batchnorm_1/ReadVariableOp�Bmodel/basemodel/batch_normalization_1/batchnorm_1/ReadVariableOp_1�Bmodel/basemodel/batch_normalization_1/batchnorm_1/ReadVariableOp_2�Dmodel/basemodel/batch_normalization_1/batchnorm_1/mul/ReadVariableOp�>model/basemodel/batch_normalization_2/batchnorm/ReadVariableOp�@model/basemodel/batch_normalization_2/batchnorm/ReadVariableOp_1�@model/basemodel/batch_normalization_2/batchnorm/ReadVariableOp_2�Bmodel/basemodel/batch_normalization_2/batchnorm/mul/ReadVariableOp�@model/basemodel/batch_normalization_2/batchnorm_1/ReadVariableOp�Bmodel/basemodel/batch_normalization_2/batchnorm_1/ReadVariableOp_1�Bmodel/basemodel/batch_normalization_2/batchnorm_1/ReadVariableOp_2�Dmodel/basemodel/batch_normalization_2/batchnorm_1/mul/ReadVariableOp�>model/basemodel/batch_normalization_3/batchnorm/ReadVariableOp�@model/basemodel/batch_normalization_3/batchnorm/ReadVariableOp_1�@model/basemodel/batch_normalization_3/batchnorm/ReadVariableOp_2�Bmodel/basemodel/batch_normalization_3/batchnorm/mul/ReadVariableOp�@model/basemodel/batch_normalization_3/batchnorm_1/ReadVariableOp�Bmodel/basemodel/batch_normalization_3/batchnorm_1/ReadVariableOp_1�Bmodel/basemodel/batch_normalization_3/batchnorm_1/ReadVariableOp_2�Dmodel/basemodel/batch_normalization_3/batchnorm_1/mul/ReadVariableOp�.model/basemodel/dense_1/BiasAdd/ReadVariableOp�0model/basemodel/dense_1/BiasAdd_1/ReadVariableOp�-model/basemodel/dense_1/MatMul/ReadVariableOp�/model/basemodel/dense_1/MatMul_1/ReadVariableOp�6model/basemodel/stream_0_conv_1/BiasAdd/ReadVariableOp�8model/basemodel/stream_0_conv_1/BiasAdd_1/ReadVariableOp�Bmodel/basemodel/stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp�Dmodel/basemodel/stream_0_conv_1/conv1d_1/ExpandDims_1/ReadVariableOp�6model/basemodel/stream_1_conv_1/BiasAdd/ReadVariableOp�8model/basemodel/stream_1_conv_1/BiasAdd_1/ReadVariableOp�Bmodel/basemodel/stream_1_conv_1/conv1d/ExpandDims_1/ReadVariableOp�Dmodel/basemodel/stream_1_conv_1/conv1d_1/ExpandDims_1/ReadVariableOp�6model/basemodel/stream_2_conv_1/BiasAdd/ReadVariableOp�8model/basemodel/stream_2_conv_1/BiasAdd_1/ReadVariableOp�Bmodel/basemodel/stream_2_conv_1/conv1d/ExpandDims_1/ReadVariableOp�Dmodel/basemodel/stream_2_conv_1/conv1d_1/ExpandDims_1/ReadVariableOp�
,model/basemodel/stream_2_input_drop/IdentityIdentityleft_inputs*
T0*,
_output_shapes
:����������2.
,model/basemodel/stream_2_input_drop/Identity�
,model/basemodel/stream_1_input_drop/IdentityIdentityleft_inputs*
T0*,
_output_shapes
:����������2.
,model/basemodel/stream_1_input_drop/Identity�
,model/basemodel/stream_0_input_drop/IdentityIdentityleft_inputs*
T0*,
_output_shapes
:����������2.
,model/basemodel/stream_0_input_drop/Identity�
5model/basemodel/stream_2_conv_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������27
5model/basemodel/stream_2_conv_1/conv1d/ExpandDims/dim�
1model/basemodel/stream_2_conv_1/conv1d/ExpandDims
ExpandDims5model/basemodel/stream_2_input_drop/Identity:output:0>model/basemodel/stream_2_conv_1/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:����������23
1model/basemodel/stream_2_conv_1/conv1d/ExpandDims�
Bmodel/basemodel/stream_2_conv_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpKmodel_basemodel_stream_2_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02D
Bmodel/basemodel/stream_2_conv_1/conv1d/ExpandDims_1/ReadVariableOp�
7model/basemodel/stream_2_conv_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 29
7model/basemodel/stream_2_conv_1/conv1d/ExpandDims_1/dim�
3model/basemodel/stream_2_conv_1/conv1d/ExpandDims_1
ExpandDimsJmodel/basemodel/stream_2_conv_1/conv1d/ExpandDims_1/ReadVariableOp:value:0@model/basemodel/stream_2_conv_1/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@25
3model/basemodel/stream_2_conv_1/conv1d/ExpandDims_1�
&model/basemodel/stream_2_conv_1/conv1dConv2D:model/basemodel/stream_2_conv_1/conv1d/ExpandDims:output:0<model/basemodel/stream_2_conv_1/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:����������@*
paddingSAME*
strides
2(
&model/basemodel/stream_2_conv_1/conv1d�
.model/basemodel/stream_2_conv_1/conv1d/SqueezeSqueeze/model/basemodel/stream_2_conv_1/conv1d:output:0*
T0*,
_output_shapes
:����������@*
squeeze_dims

���������20
.model/basemodel/stream_2_conv_1/conv1d/Squeeze�
6model/basemodel/stream_2_conv_1/BiasAdd/ReadVariableOpReadVariableOp?model_basemodel_stream_2_conv_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype028
6model/basemodel/stream_2_conv_1/BiasAdd/ReadVariableOp�
'model/basemodel/stream_2_conv_1/BiasAddBiasAdd7model/basemodel/stream_2_conv_1/conv1d/Squeeze:output:0>model/basemodel/stream_2_conv_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������@2)
'model/basemodel/stream_2_conv_1/BiasAdd�
5model/basemodel/stream_1_conv_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������27
5model/basemodel/stream_1_conv_1/conv1d/ExpandDims/dim�
1model/basemodel/stream_1_conv_1/conv1d/ExpandDims
ExpandDims5model/basemodel/stream_1_input_drop/Identity:output:0>model/basemodel/stream_1_conv_1/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:����������23
1model/basemodel/stream_1_conv_1/conv1d/ExpandDims�
Bmodel/basemodel/stream_1_conv_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpKmodel_basemodel_stream_1_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02D
Bmodel/basemodel/stream_1_conv_1/conv1d/ExpandDims_1/ReadVariableOp�
7model/basemodel/stream_1_conv_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 29
7model/basemodel/stream_1_conv_1/conv1d/ExpandDims_1/dim�
3model/basemodel/stream_1_conv_1/conv1d/ExpandDims_1
ExpandDimsJmodel/basemodel/stream_1_conv_1/conv1d/ExpandDims_1/ReadVariableOp:value:0@model/basemodel/stream_1_conv_1/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@25
3model/basemodel/stream_1_conv_1/conv1d/ExpandDims_1�
&model/basemodel/stream_1_conv_1/conv1dConv2D:model/basemodel/stream_1_conv_1/conv1d/ExpandDims:output:0<model/basemodel/stream_1_conv_1/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:����������@*
paddingSAME*
strides
2(
&model/basemodel/stream_1_conv_1/conv1d�
.model/basemodel/stream_1_conv_1/conv1d/SqueezeSqueeze/model/basemodel/stream_1_conv_1/conv1d:output:0*
T0*,
_output_shapes
:����������@*
squeeze_dims

���������20
.model/basemodel/stream_1_conv_1/conv1d/Squeeze�
6model/basemodel/stream_1_conv_1/BiasAdd/ReadVariableOpReadVariableOp?model_basemodel_stream_1_conv_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype028
6model/basemodel/stream_1_conv_1/BiasAdd/ReadVariableOp�
'model/basemodel/stream_1_conv_1/BiasAddBiasAdd7model/basemodel/stream_1_conv_1/conv1d/Squeeze:output:0>model/basemodel/stream_1_conv_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������@2)
'model/basemodel/stream_1_conv_1/BiasAdd�
5model/basemodel/stream_0_conv_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������27
5model/basemodel/stream_0_conv_1/conv1d/ExpandDims/dim�
1model/basemodel/stream_0_conv_1/conv1d/ExpandDims
ExpandDims5model/basemodel/stream_0_input_drop/Identity:output:0>model/basemodel/stream_0_conv_1/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:����������23
1model/basemodel/stream_0_conv_1/conv1d/ExpandDims�
Bmodel/basemodel/stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpKmodel_basemodel_stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02D
Bmodel/basemodel/stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp�
7model/basemodel/stream_0_conv_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 29
7model/basemodel/stream_0_conv_1/conv1d/ExpandDims_1/dim�
3model/basemodel/stream_0_conv_1/conv1d/ExpandDims_1
ExpandDimsJmodel/basemodel/stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp:value:0@model/basemodel/stream_0_conv_1/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@25
3model/basemodel/stream_0_conv_1/conv1d/ExpandDims_1�
&model/basemodel/stream_0_conv_1/conv1dConv2D:model/basemodel/stream_0_conv_1/conv1d/ExpandDims:output:0<model/basemodel/stream_0_conv_1/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:����������@*
paddingSAME*
strides
2(
&model/basemodel/stream_0_conv_1/conv1d�
.model/basemodel/stream_0_conv_1/conv1d/SqueezeSqueeze/model/basemodel/stream_0_conv_1/conv1d:output:0*
T0*,
_output_shapes
:����������@*
squeeze_dims

���������20
.model/basemodel/stream_0_conv_1/conv1d/Squeeze�
6model/basemodel/stream_0_conv_1/BiasAdd/ReadVariableOpReadVariableOp?model_basemodel_stream_0_conv_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype028
6model/basemodel/stream_0_conv_1/BiasAdd/ReadVariableOp�
'model/basemodel/stream_0_conv_1/BiasAddBiasAdd7model/basemodel/stream_0_conv_1/conv1d/Squeeze:output:0>model/basemodel/stream_0_conv_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������@2)
'model/basemodel/stream_0_conv_1/BiasAdd�
>model/basemodel/batch_normalization_2/batchnorm/ReadVariableOpReadVariableOpGmodel_basemodel_batch_normalization_2_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02@
>model/basemodel/batch_normalization_2/batchnorm/ReadVariableOp�
5model/basemodel/batch_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:27
5model/basemodel/batch_normalization_2/batchnorm/add/y�
3model/basemodel/batch_normalization_2/batchnorm/addAddV2Fmodel/basemodel/batch_normalization_2/batchnorm/ReadVariableOp:value:0>model/basemodel/batch_normalization_2/batchnorm/add/y:output:0*
T0*
_output_shapes
:@25
3model/basemodel/batch_normalization_2/batchnorm/add�
5model/basemodel/batch_normalization_2/batchnorm/RsqrtRsqrt7model/basemodel/batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes
:@27
5model/basemodel/batch_normalization_2/batchnorm/Rsqrt�
Bmodel/basemodel/batch_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOpKmodel_basemodel_batch_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02D
Bmodel/basemodel/batch_normalization_2/batchnorm/mul/ReadVariableOp�
3model/basemodel/batch_normalization_2/batchnorm/mulMul9model/basemodel/batch_normalization_2/batchnorm/Rsqrt:y:0Jmodel/basemodel/batch_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@25
3model/basemodel/batch_normalization_2/batchnorm/mul�
5model/basemodel/batch_normalization_2/batchnorm/mul_1Mul0model/basemodel/stream_2_conv_1/BiasAdd:output:07model/basemodel/batch_normalization_2/batchnorm/mul:z:0*
T0*,
_output_shapes
:����������@27
5model/basemodel/batch_normalization_2/batchnorm/mul_1�
@model/basemodel/batch_normalization_2/batchnorm/ReadVariableOp_1ReadVariableOpImodel_basemodel_batch_normalization_2_batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype02B
@model/basemodel/batch_normalization_2/batchnorm/ReadVariableOp_1�
5model/basemodel/batch_normalization_2/batchnorm/mul_2MulHmodel/basemodel/batch_normalization_2/batchnorm/ReadVariableOp_1:value:07model/basemodel/batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes
:@27
5model/basemodel/batch_normalization_2/batchnorm/mul_2�
@model/basemodel/batch_normalization_2/batchnorm/ReadVariableOp_2ReadVariableOpImodel_basemodel_batch_normalization_2_batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype02B
@model/basemodel/batch_normalization_2/batchnorm/ReadVariableOp_2�
3model/basemodel/batch_normalization_2/batchnorm/subSubHmodel/basemodel/batch_normalization_2/batchnorm/ReadVariableOp_2:value:09model/basemodel/batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@25
3model/basemodel/batch_normalization_2/batchnorm/sub�
5model/basemodel/batch_normalization_2/batchnorm/add_1AddV29model/basemodel/batch_normalization_2/batchnorm/mul_1:z:07model/basemodel/batch_normalization_2/batchnorm/sub:z:0*
T0*,
_output_shapes
:����������@27
5model/basemodel/batch_normalization_2/batchnorm/add_1�
>model/basemodel/batch_normalization_1/batchnorm/ReadVariableOpReadVariableOpGmodel_basemodel_batch_normalization_1_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02@
>model/basemodel/batch_normalization_1/batchnorm/ReadVariableOp�
5model/basemodel/batch_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:27
5model/basemodel/batch_normalization_1/batchnorm/add/y�
3model/basemodel/batch_normalization_1/batchnorm/addAddV2Fmodel/basemodel/batch_normalization_1/batchnorm/ReadVariableOp:value:0>model/basemodel/batch_normalization_1/batchnorm/add/y:output:0*
T0*
_output_shapes
:@25
3model/basemodel/batch_normalization_1/batchnorm/add�
5model/basemodel/batch_normalization_1/batchnorm/RsqrtRsqrt7model/basemodel/batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes
:@27
5model/basemodel/batch_normalization_1/batchnorm/Rsqrt�
Bmodel/basemodel/batch_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOpKmodel_basemodel_batch_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02D
Bmodel/basemodel/batch_normalization_1/batchnorm/mul/ReadVariableOp�
3model/basemodel/batch_normalization_1/batchnorm/mulMul9model/basemodel/batch_normalization_1/batchnorm/Rsqrt:y:0Jmodel/basemodel/batch_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@25
3model/basemodel/batch_normalization_1/batchnorm/mul�
5model/basemodel/batch_normalization_1/batchnorm/mul_1Mul0model/basemodel/stream_1_conv_1/BiasAdd:output:07model/basemodel/batch_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:����������@27
5model/basemodel/batch_normalization_1/batchnorm/mul_1�
@model/basemodel/batch_normalization_1/batchnorm/ReadVariableOp_1ReadVariableOpImodel_basemodel_batch_normalization_1_batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype02B
@model/basemodel/batch_normalization_1/batchnorm/ReadVariableOp_1�
5model/basemodel/batch_normalization_1/batchnorm/mul_2MulHmodel/basemodel/batch_normalization_1/batchnorm/ReadVariableOp_1:value:07model/basemodel/batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes
:@27
5model/basemodel/batch_normalization_1/batchnorm/mul_2�
@model/basemodel/batch_normalization_1/batchnorm/ReadVariableOp_2ReadVariableOpImodel_basemodel_batch_normalization_1_batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype02B
@model/basemodel/batch_normalization_1/batchnorm/ReadVariableOp_2�
3model/basemodel/batch_normalization_1/batchnorm/subSubHmodel/basemodel/batch_normalization_1/batchnorm/ReadVariableOp_2:value:09model/basemodel/batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@25
3model/basemodel/batch_normalization_1/batchnorm/sub�
5model/basemodel/batch_normalization_1/batchnorm/add_1AddV29model/basemodel/batch_normalization_1/batchnorm/mul_1:z:07model/basemodel/batch_normalization_1/batchnorm/sub:z:0*
T0*,
_output_shapes
:����������@27
5model/basemodel/batch_normalization_1/batchnorm/add_1�
<model/basemodel/batch_normalization/batchnorm/ReadVariableOpReadVariableOpEmodel_basemodel_batch_normalization_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02>
<model/basemodel/batch_normalization/batchnorm/ReadVariableOp�
3model/basemodel/batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:25
3model/basemodel/batch_normalization/batchnorm/add/y�
1model/basemodel/batch_normalization/batchnorm/addAddV2Dmodel/basemodel/batch_normalization/batchnorm/ReadVariableOp:value:0<model/basemodel/batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes
:@23
1model/basemodel/batch_normalization/batchnorm/add�
3model/basemodel/batch_normalization/batchnorm/RsqrtRsqrt5model/basemodel/batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes
:@25
3model/basemodel/batch_normalization/batchnorm/Rsqrt�
@model/basemodel/batch_normalization/batchnorm/mul/ReadVariableOpReadVariableOpImodel_basemodel_batch_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02B
@model/basemodel/batch_normalization/batchnorm/mul/ReadVariableOp�
1model/basemodel/batch_normalization/batchnorm/mulMul7model/basemodel/batch_normalization/batchnorm/Rsqrt:y:0Hmodel/basemodel/batch_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@23
1model/basemodel/batch_normalization/batchnorm/mul�
3model/basemodel/batch_normalization/batchnorm/mul_1Mul0model/basemodel/stream_0_conv_1/BiasAdd:output:05model/basemodel/batch_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:����������@25
3model/basemodel/batch_normalization/batchnorm/mul_1�
>model/basemodel/batch_normalization/batchnorm/ReadVariableOp_1ReadVariableOpGmodel_basemodel_batch_normalization_batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype02@
>model/basemodel/batch_normalization/batchnorm/ReadVariableOp_1�
3model/basemodel/batch_normalization/batchnorm/mul_2MulFmodel/basemodel/batch_normalization/batchnorm/ReadVariableOp_1:value:05model/basemodel/batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes
:@25
3model/basemodel/batch_normalization/batchnorm/mul_2�
>model/basemodel/batch_normalization/batchnorm/ReadVariableOp_2ReadVariableOpGmodel_basemodel_batch_normalization_batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype02@
>model/basemodel/batch_normalization/batchnorm/ReadVariableOp_2�
1model/basemodel/batch_normalization/batchnorm/subSubFmodel/basemodel/batch_normalization/batchnorm/ReadVariableOp_2:value:07model/basemodel/batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@23
1model/basemodel/batch_normalization/batchnorm/sub�
3model/basemodel/batch_normalization/batchnorm/add_1AddV27model/basemodel/batch_normalization/batchnorm/mul_1:z:05model/basemodel/batch_normalization/batchnorm/sub:z:0*
T0*,
_output_shapes
:����������@25
3model/basemodel/batch_normalization/batchnorm/add_1�
!model/basemodel/activation_2/ReluRelu9model/basemodel/batch_normalization_2/batchnorm/add_1:z:0*
T0*,
_output_shapes
:����������@2#
!model/basemodel/activation_2/Relu�
!model/basemodel/activation_1/ReluRelu9model/basemodel/batch_normalization_1/batchnorm/add_1:z:0*
T0*,
_output_shapes
:����������@2#
!model/basemodel/activation_1/Relu�
model/basemodel/activation/ReluRelu7model/basemodel/batch_normalization/batchnorm/add_1:z:0*
T0*,
_output_shapes
:����������@2!
model/basemodel/activation/Relu�
(model/basemodel/stream_2_drop_1/IdentityIdentity/model/basemodel/activation_2/Relu:activations:0*
T0*,
_output_shapes
:����������@2*
(model/basemodel/stream_2_drop_1/Identity�
(model/basemodel/stream_1_drop_1/IdentityIdentity/model/basemodel/activation_1/Relu:activations:0*
T0*,
_output_shapes
:����������@2*
(model/basemodel/stream_1_drop_1/Identity�
(model/basemodel/stream_0_drop_1/IdentityIdentity-model/basemodel/activation/Relu:activations:0*
T0*,
_output_shapes
:����������@2*
(model/basemodel/stream_0_drop_1/Identity�
?model/basemodel/global_average_pooling1d/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2A
?model/basemodel/global_average_pooling1d/Mean/reduction_indices�
-model/basemodel/global_average_pooling1d/MeanMean1model/basemodel/stream_0_drop_1/Identity:output:0Hmodel/basemodel/global_average_pooling1d/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:���������@2/
-model/basemodel/global_average_pooling1d/Mean�
Amodel/basemodel/global_average_pooling1d_1/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2C
Amodel/basemodel/global_average_pooling1d_1/Mean/reduction_indices�
/model/basemodel/global_average_pooling1d_1/MeanMean1model/basemodel/stream_1_drop_1/Identity:output:0Jmodel/basemodel/global_average_pooling1d_1/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:���������@21
/model/basemodel/global_average_pooling1d_1/Mean�
Amodel/basemodel/global_average_pooling1d_2/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2C
Amodel/basemodel/global_average_pooling1d_2/Mean/reduction_indices�
/model/basemodel/global_average_pooling1d_2/MeanMean1model/basemodel/stream_2_drop_1/Identity:output:0Jmodel/basemodel/global_average_pooling1d_2/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:���������@21
/model/basemodel/global_average_pooling1d_2/Mean�
'model/basemodel/concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2)
'model/basemodel/concatenate/concat/axis�
"model/basemodel/concatenate/concatConcatV26model/basemodel/global_average_pooling1d/Mean:output:08model/basemodel/global_average_pooling1d_1/Mean:output:08model/basemodel/global_average_pooling1d_2/Mean:output:00model/basemodel/concatenate/concat/axis:output:0*
N*
T0*(
_output_shapes
:����������2$
"model/basemodel/concatenate/concat�
(model/basemodel/dense_1_dropout/IdentityIdentity+model/basemodel/concatenate/concat:output:0*
T0*(
_output_shapes
:����������2*
(model/basemodel/dense_1_dropout/Identity�
-model/basemodel/dense_1/MatMul/ReadVariableOpReadVariableOp6model_basemodel_dense_1_matmul_readvariableop_resource*
_output_shapes
:	�T*
dtype02/
-model/basemodel/dense_1/MatMul/ReadVariableOp�
model/basemodel/dense_1/MatMulMatMul1model/basemodel/dense_1_dropout/Identity:output:05model/basemodel/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������T2 
model/basemodel/dense_1/MatMul�
.model/basemodel/dense_1/BiasAdd/ReadVariableOpReadVariableOp7model_basemodel_dense_1_biasadd_readvariableop_resource*
_output_shapes
:T*
dtype020
.model/basemodel/dense_1/BiasAdd/ReadVariableOp�
model/basemodel/dense_1/BiasAddBiasAdd(model/basemodel/dense_1/MatMul:product:06model/basemodel/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������T2!
model/basemodel/dense_1/BiasAdd�
>model/basemodel/batch_normalization_3/batchnorm/ReadVariableOpReadVariableOpGmodel_basemodel_batch_normalization_3_batchnorm_readvariableop_resource*
_output_shapes
:T*
dtype02@
>model/basemodel/batch_normalization_3/batchnorm/ReadVariableOp�
5model/basemodel/batch_normalization_3/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:27
5model/basemodel/batch_normalization_3/batchnorm/add/y�
3model/basemodel/batch_normalization_3/batchnorm/addAddV2Fmodel/basemodel/batch_normalization_3/batchnorm/ReadVariableOp:value:0>model/basemodel/batch_normalization_3/batchnorm/add/y:output:0*
T0*
_output_shapes
:T25
3model/basemodel/batch_normalization_3/batchnorm/add�
5model/basemodel/batch_normalization_3/batchnorm/RsqrtRsqrt7model/basemodel/batch_normalization_3/batchnorm/add:z:0*
T0*
_output_shapes
:T27
5model/basemodel/batch_normalization_3/batchnorm/Rsqrt�
Bmodel/basemodel/batch_normalization_3/batchnorm/mul/ReadVariableOpReadVariableOpKmodel_basemodel_batch_normalization_3_batchnorm_mul_readvariableop_resource*
_output_shapes
:T*
dtype02D
Bmodel/basemodel/batch_normalization_3/batchnorm/mul/ReadVariableOp�
3model/basemodel/batch_normalization_3/batchnorm/mulMul9model/basemodel/batch_normalization_3/batchnorm/Rsqrt:y:0Jmodel/basemodel/batch_normalization_3/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:T25
3model/basemodel/batch_normalization_3/batchnorm/mul�
5model/basemodel/batch_normalization_3/batchnorm/mul_1Mul(model/basemodel/dense_1/BiasAdd:output:07model/basemodel/batch_normalization_3/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������T27
5model/basemodel/batch_normalization_3/batchnorm/mul_1�
@model/basemodel/batch_normalization_3/batchnorm/ReadVariableOp_1ReadVariableOpImodel_basemodel_batch_normalization_3_batchnorm_readvariableop_1_resource*
_output_shapes
:T*
dtype02B
@model/basemodel/batch_normalization_3/batchnorm/ReadVariableOp_1�
5model/basemodel/batch_normalization_3/batchnorm/mul_2MulHmodel/basemodel/batch_normalization_3/batchnorm/ReadVariableOp_1:value:07model/basemodel/batch_normalization_3/batchnorm/mul:z:0*
T0*
_output_shapes
:T27
5model/basemodel/batch_normalization_3/batchnorm/mul_2�
@model/basemodel/batch_normalization_3/batchnorm/ReadVariableOp_2ReadVariableOpImodel_basemodel_batch_normalization_3_batchnorm_readvariableop_2_resource*
_output_shapes
:T*
dtype02B
@model/basemodel/batch_normalization_3/batchnorm/ReadVariableOp_2�
3model/basemodel/batch_normalization_3/batchnorm/subSubHmodel/basemodel/batch_normalization_3/batchnorm/ReadVariableOp_2:value:09model/basemodel/batch_normalization_3/batchnorm/mul_2:z:0*
T0*
_output_shapes
:T25
3model/basemodel/batch_normalization_3/batchnorm/sub�
5model/basemodel/batch_normalization_3/batchnorm/add_1AddV29model/basemodel/batch_normalization_3/batchnorm/mul_1:z:07model/basemodel/batch_normalization_3/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������T27
5model/basemodel/batch_normalization_3/batchnorm/add_1�
*model/basemodel/dense_activation_1/SigmoidSigmoid9model/basemodel/batch_normalization_3/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������T2,
*model/basemodel/dense_activation_1/Sigmoid�
.model/basemodel/stream_2_input_drop/Identity_1Identityright_inputs*
T0*,
_output_shapes
:����������20
.model/basemodel/stream_2_input_drop/Identity_1�
.model/basemodel/stream_1_input_drop/Identity_1Identityright_inputs*
T0*,
_output_shapes
:����������20
.model/basemodel/stream_1_input_drop/Identity_1�
.model/basemodel/stream_0_input_drop/Identity_1Identityright_inputs*
T0*,
_output_shapes
:����������20
.model/basemodel/stream_0_input_drop/Identity_1�
7model/basemodel/stream_2_conv_1/conv1d_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������29
7model/basemodel/stream_2_conv_1/conv1d_1/ExpandDims/dim�
3model/basemodel/stream_2_conv_1/conv1d_1/ExpandDims
ExpandDims7model/basemodel/stream_2_input_drop/Identity_1:output:0@model/basemodel/stream_2_conv_1/conv1d_1/ExpandDims/dim:output:0*
T0*0
_output_shapes
:����������25
3model/basemodel/stream_2_conv_1/conv1d_1/ExpandDims�
Dmodel/basemodel/stream_2_conv_1/conv1d_1/ExpandDims_1/ReadVariableOpReadVariableOpKmodel_basemodel_stream_2_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02F
Dmodel/basemodel/stream_2_conv_1/conv1d_1/ExpandDims_1/ReadVariableOp�
9model/basemodel/stream_2_conv_1/conv1d_1/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2;
9model/basemodel/stream_2_conv_1/conv1d_1/ExpandDims_1/dim�
5model/basemodel/stream_2_conv_1/conv1d_1/ExpandDims_1
ExpandDimsLmodel/basemodel/stream_2_conv_1/conv1d_1/ExpandDims_1/ReadVariableOp:value:0Bmodel/basemodel/stream_2_conv_1/conv1d_1/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@27
5model/basemodel/stream_2_conv_1/conv1d_1/ExpandDims_1�
(model/basemodel/stream_2_conv_1/conv1d_1Conv2D<model/basemodel/stream_2_conv_1/conv1d_1/ExpandDims:output:0>model/basemodel/stream_2_conv_1/conv1d_1/ExpandDims_1:output:0*
T0*0
_output_shapes
:����������@*
paddingSAME*
strides
2*
(model/basemodel/stream_2_conv_1/conv1d_1�
0model/basemodel/stream_2_conv_1/conv1d_1/SqueezeSqueeze1model/basemodel/stream_2_conv_1/conv1d_1:output:0*
T0*,
_output_shapes
:����������@*
squeeze_dims

���������22
0model/basemodel/stream_2_conv_1/conv1d_1/Squeeze�
8model/basemodel/stream_2_conv_1/BiasAdd_1/ReadVariableOpReadVariableOp?model_basemodel_stream_2_conv_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02:
8model/basemodel/stream_2_conv_1/BiasAdd_1/ReadVariableOp�
)model/basemodel/stream_2_conv_1/BiasAdd_1BiasAdd9model/basemodel/stream_2_conv_1/conv1d_1/Squeeze:output:0@model/basemodel/stream_2_conv_1/BiasAdd_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������@2+
)model/basemodel/stream_2_conv_1/BiasAdd_1�
7model/basemodel/stream_1_conv_1/conv1d_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������29
7model/basemodel/stream_1_conv_1/conv1d_1/ExpandDims/dim�
3model/basemodel/stream_1_conv_1/conv1d_1/ExpandDims
ExpandDims7model/basemodel/stream_1_input_drop/Identity_1:output:0@model/basemodel/stream_1_conv_1/conv1d_1/ExpandDims/dim:output:0*
T0*0
_output_shapes
:����������25
3model/basemodel/stream_1_conv_1/conv1d_1/ExpandDims�
Dmodel/basemodel/stream_1_conv_1/conv1d_1/ExpandDims_1/ReadVariableOpReadVariableOpKmodel_basemodel_stream_1_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02F
Dmodel/basemodel/stream_1_conv_1/conv1d_1/ExpandDims_1/ReadVariableOp�
9model/basemodel/stream_1_conv_1/conv1d_1/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2;
9model/basemodel/stream_1_conv_1/conv1d_1/ExpandDims_1/dim�
5model/basemodel/stream_1_conv_1/conv1d_1/ExpandDims_1
ExpandDimsLmodel/basemodel/stream_1_conv_1/conv1d_1/ExpandDims_1/ReadVariableOp:value:0Bmodel/basemodel/stream_1_conv_1/conv1d_1/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@27
5model/basemodel/stream_1_conv_1/conv1d_1/ExpandDims_1�
(model/basemodel/stream_1_conv_1/conv1d_1Conv2D<model/basemodel/stream_1_conv_1/conv1d_1/ExpandDims:output:0>model/basemodel/stream_1_conv_1/conv1d_1/ExpandDims_1:output:0*
T0*0
_output_shapes
:����������@*
paddingSAME*
strides
2*
(model/basemodel/stream_1_conv_1/conv1d_1�
0model/basemodel/stream_1_conv_1/conv1d_1/SqueezeSqueeze1model/basemodel/stream_1_conv_1/conv1d_1:output:0*
T0*,
_output_shapes
:����������@*
squeeze_dims

���������22
0model/basemodel/stream_1_conv_1/conv1d_1/Squeeze�
8model/basemodel/stream_1_conv_1/BiasAdd_1/ReadVariableOpReadVariableOp?model_basemodel_stream_1_conv_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02:
8model/basemodel/stream_1_conv_1/BiasAdd_1/ReadVariableOp�
)model/basemodel/stream_1_conv_1/BiasAdd_1BiasAdd9model/basemodel/stream_1_conv_1/conv1d_1/Squeeze:output:0@model/basemodel/stream_1_conv_1/BiasAdd_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������@2+
)model/basemodel/stream_1_conv_1/BiasAdd_1�
7model/basemodel/stream_0_conv_1/conv1d_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������29
7model/basemodel/stream_0_conv_1/conv1d_1/ExpandDims/dim�
3model/basemodel/stream_0_conv_1/conv1d_1/ExpandDims
ExpandDims7model/basemodel/stream_0_input_drop/Identity_1:output:0@model/basemodel/stream_0_conv_1/conv1d_1/ExpandDims/dim:output:0*
T0*0
_output_shapes
:����������25
3model/basemodel/stream_0_conv_1/conv1d_1/ExpandDims�
Dmodel/basemodel/stream_0_conv_1/conv1d_1/ExpandDims_1/ReadVariableOpReadVariableOpKmodel_basemodel_stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02F
Dmodel/basemodel/stream_0_conv_1/conv1d_1/ExpandDims_1/ReadVariableOp�
9model/basemodel/stream_0_conv_1/conv1d_1/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2;
9model/basemodel/stream_0_conv_1/conv1d_1/ExpandDims_1/dim�
5model/basemodel/stream_0_conv_1/conv1d_1/ExpandDims_1
ExpandDimsLmodel/basemodel/stream_0_conv_1/conv1d_1/ExpandDims_1/ReadVariableOp:value:0Bmodel/basemodel/stream_0_conv_1/conv1d_1/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@27
5model/basemodel/stream_0_conv_1/conv1d_1/ExpandDims_1�
(model/basemodel/stream_0_conv_1/conv1d_1Conv2D<model/basemodel/stream_0_conv_1/conv1d_1/ExpandDims:output:0>model/basemodel/stream_0_conv_1/conv1d_1/ExpandDims_1:output:0*
T0*0
_output_shapes
:����������@*
paddingSAME*
strides
2*
(model/basemodel/stream_0_conv_1/conv1d_1�
0model/basemodel/stream_0_conv_1/conv1d_1/SqueezeSqueeze1model/basemodel/stream_0_conv_1/conv1d_1:output:0*
T0*,
_output_shapes
:����������@*
squeeze_dims

���������22
0model/basemodel/stream_0_conv_1/conv1d_1/Squeeze�
8model/basemodel/stream_0_conv_1/BiasAdd_1/ReadVariableOpReadVariableOp?model_basemodel_stream_0_conv_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02:
8model/basemodel/stream_0_conv_1/BiasAdd_1/ReadVariableOp�
)model/basemodel/stream_0_conv_1/BiasAdd_1BiasAdd9model/basemodel/stream_0_conv_1/conv1d_1/Squeeze:output:0@model/basemodel/stream_0_conv_1/BiasAdd_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������@2+
)model/basemodel/stream_0_conv_1/BiasAdd_1�
@model/basemodel/batch_normalization_2/batchnorm_1/ReadVariableOpReadVariableOpGmodel_basemodel_batch_normalization_2_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02B
@model/basemodel/batch_normalization_2/batchnorm_1/ReadVariableOp�
7model/basemodel/batch_normalization_2/batchnorm_1/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:29
7model/basemodel/batch_normalization_2/batchnorm_1/add/y�
5model/basemodel/batch_normalization_2/batchnorm_1/addAddV2Hmodel/basemodel/batch_normalization_2/batchnorm_1/ReadVariableOp:value:0@model/basemodel/batch_normalization_2/batchnorm_1/add/y:output:0*
T0*
_output_shapes
:@27
5model/basemodel/batch_normalization_2/batchnorm_1/add�
7model/basemodel/batch_normalization_2/batchnorm_1/RsqrtRsqrt9model/basemodel/batch_normalization_2/batchnorm_1/add:z:0*
T0*
_output_shapes
:@29
7model/basemodel/batch_normalization_2/batchnorm_1/Rsqrt�
Dmodel/basemodel/batch_normalization_2/batchnorm_1/mul/ReadVariableOpReadVariableOpKmodel_basemodel_batch_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02F
Dmodel/basemodel/batch_normalization_2/batchnorm_1/mul/ReadVariableOp�
5model/basemodel/batch_normalization_2/batchnorm_1/mulMul;model/basemodel/batch_normalization_2/batchnorm_1/Rsqrt:y:0Lmodel/basemodel/batch_normalization_2/batchnorm_1/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@27
5model/basemodel/batch_normalization_2/batchnorm_1/mul�
7model/basemodel/batch_normalization_2/batchnorm_1/mul_1Mul2model/basemodel/stream_2_conv_1/BiasAdd_1:output:09model/basemodel/batch_normalization_2/batchnorm_1/mul:z:0*
T0*,
_output_shapes
:����������@29
7model/basemodel/batch_normalization_2/batchnorm_1/mul_1�
Bmodel/basemodel/batch_normalization_2/batchnorm_1/ReadVariableOp_1ReadVariableOpImodel_basemodel_batch_normalization_2_batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype02D
Bmodel/basemodel/batch_normalization_2/batchnorm_1/ReadVariableOp_1�
7model/basemodel/batch_normalization_2/batchnorm_1/mul_2MulJmodel/basemodel/batch_normalization_2/batchnorm_1/ReadVariableOp_1:value:09model/basemodel/batch_normalization_2/batchnorm_1/mul:z:0*
T0*
_output_shapes
:@29
7model/basemodel/batch_normalization_2/batchnorm_1/mul_2�
Bmodel/basemodel/batch_normalization_2/batchnorm_1/ReadVariableOp_2ReadVariableOpImodel_basemodel_batch_normalization_2_batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype02D
Bmodel/basemodel/batch_normalization_2/batchnorm_1/ReadVariableOp_2�
5model/basemodel/batch_normalization_2/batchnorm_1/subSubJmodel/basemodel/batch_normalization_2/batchnorm_1/ReadVariableOp_2:value:0;model/basemodel/batch_normalization_2/batchnorm_1/mul_2:z:0*
T0*
_output_shapes
:@27
5model/basemodel/batch_normalization_2/batchnorm_1/sub�
7model/basemodel/batch_normalization_2/batchnorm_1/add_1AddV2;model/basemodel/batch_normalization_2/batchnorm_1/mul_1:z:09model/basemodel/batch_normalization_2/batchnorm_1/sub:z:0*
T0*,
_output_shapes
:����������@29
7model/basemodel/batch_normalization_2/batchnorm_1/add_1�
@model/basemodel/batch_normalization_1/batchnorm_1/ReadVariableOpReadVariableOpGmodel_basemodel_batch_normalization_1_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02B
@model/basemodel/batch_normalization_1/batchnorm_1/ReadVariableOp�
7model/basemodel/batch_normalization_1/batchnorm_1/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:29
7model/basemodel/batch_normalization_1/batchnorm_1/add/y�
5model/basemodel/batch_normalization_1/batchnorm_1/addAddV2Hmodel/basemodel/batch_normalization_1/batchnorm_1/ReadVariableOp:value:0@model/basemodel/batch_normalization_1/batchnorm_1/add/y:output:0*
T0*
_output_shapes
:@27
5model/basemodel/batch_normalization_1/batchnorm_1/add�
7model/basemodel/batch_normalization_1/batchnorm_1/RsqrtRsqrt9model/basemodel/batch_normalization_1/batchnorm_1/add:z:0*
T0*
_output_shapes
:@29
7model/basemodel/batch_normalization_1/batchnorm_1/Rsqrt�
Dmodel/basemodel/batch_normalization_1/batchnorm_1/mul/ReadVariableOpReadVariableOpKmodel_basemodel_batch_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02F
Dmodel/basemodel/batch_normalization_1/batchnorm_1/mul/ReadVariableOp�
5model/basemodel/batch_normalization_1/batchnorm_1/mulMul;model/basemodel/batch_normalization_1/batchnorm_1/Rsqrt:y:0Lmodel/basemodel/batch_normalization_1/batchnorm_1/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@27
5model/basemodel/batch_normalization_1/batchnorm_1/mul�
7model/basemodel/batch_normalization_1/batchnorm_1/mul_1Mul2model/basemodel/stream_1_conv_1/BiasAdd_1:output:09model/basemodel/batch_normalization_1/batchnorm_1/mul:z:0*
T0*,
_output_shapes
:����������@29
7model/basemodel/batch_normalization_1/batchnorm_1/mul_1�
Bmodel/basemodel/batch_normalization_1/batchnorm_1/ReadVariableOp_1ReadVariableOpImodel_basemodel_batch_normalization_1_batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype02D
Bmodel/basemodel/batch_normalization_1/batchnorm_1/ReadVariableOp_1�
7model/basemodel/batch_normalization_1/batchnorm_1/mul_2MulJmodel/basemodel/batch_normalization_1/batchnorm_1/ReadVariableOp_1:value:09model/basemodel/batch_normalization_1/batchnorm_1/mul:z:0*
T0*
_output_shapes
:@29
7model/basemodel/batch_normalization_1/batchnorm_1/mul_2�
Bmodel/basemodel/batch_normalization_1/batchnorm_1/ReadVariableOp_2ReadVariableOpImodel_basemodel_batch_normalization_1_batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype02D
Bmodel/basemodel/batch_normalization_1/batchnorm_1/ReadVariableOp_2�
5model/basemodel/batch_normalization_1/batchnorm_1/subSubJmodel/basemodel/batch_normalization_1/batchnorm_1/ReadVariableOp_2:value:0;model/basemodel/batch_normalization_1/batchnorm_1/mul_2:z:0*
T0*
_output_shapes
:@27
5model/basemodel/batch_normalization_1/batchnorm_1/sub�
7model/basemodel/batch_normalization_1/batchnorm_1/add_1AddV2;model/basemodel/batch_normalization_1/batchnorm_1/mul_1:z:09model/basemodel/batch_normalization_1/batchnorm_1/sub:z:0*
T0*,
_output_shapes
:����������@29
7model/basemodel/batch_normalization_1/batchnorm_1/add_1�
>model/basemodel/batch_normalization/batchnorm_1/ReadVariableOpReadVariableOpEmodel_basemodel_batch_normalization_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02@
>model/basemodel/batch_normalization/batchnorm_1/ReadVariableOp�
5model/basemodel/batch_normalization/batchnorm_1/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:27
5model/basemodel/batch_normalization/batchnorm_1/add/y�
3model/basemodel/batch_normalization/batchnorm_1/addAddV2Fmodel/basemodel/batch_normalization/batchnorm_1/ReadVariableOp:value:0>model/basemodel/batch_normalization/batchnorm_1/add/y:output:0*
T0*
_output_shapes
:@25
3model/basemodel/batch_normalization/batchnorm_1/add�
5model/basemodel/batch_normalization/batchnorm_1/RsqrtRsqrt7model/basemodel/batch_normalization/batchnorm_1/add:z:0*
T0*
_output_shapes
:@27
5model/basemodel/batch_normalization/batchnorm_1/Rsqrt�
Bmodel/basemodel/batch_normalization/batchnorm_1/mul/ReadVariableOpReadVariableOpImodel_basemodel_batch_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02D
Bmodel/basemodel/batch_normalization/batchnorm_1/mul/ReadVariableOp�
3model/basemodel/batch_normalization/batchnorm_1/mulMul9model/basemodel/batch_normalization/batchnorm_1/Rsqrt:y:0Jmodel/basemodel/batch_normalization/batchnorm_1/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@25
3model/basemodel/batch_normalization/batchnorm_1/mul�
5model/basemodel/batch_normalization/batchnorm_1/mul_1Mul2model/basemodel/stream_0_conv_1/BiasAdd_1:output:07model/basemodel/batch_normalization/batchnorm_1/mul:z:0*
T0*,
_output_shapes
:����������@27
5model/basemodel/batch_normalization/batchnorm_1/mul_1�
@model/basemodel/batch_normalization/batchnorm_1/ReadVariableOp_1ReadVariableOpGmodel_basemodel_batch_normalization_batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype02B
@model/basemodel/batch_normalization/batchnorm_1/ReadVariableOp_1�
5model/basemodel/batch_normalization/batchnorm_1/mul_2MulHmodel/basemodel/batch_normalization/batchnorm_1/ReadVariableOp_1:value:07model/basemodel/batch_normalization/batchnorm_1/mul:z:0*
T0*
_output_shapes
:@27
5model/basemodel/batch_normalization/batchnorm_1/mul_2�
@model/basemodel/batch_normalization/batchnorm_1/ReadVariableOp_2ReadVariableOpGmodel_basemodel_batch_normalization_batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype02B
@model/basemodel/batch_normalization/batchnorm_1/ReadVariableOp_2�
3model/basemodel/batch_normalization/batchnorm_1/subSubHmodel/basemodel/batch_normalization/batchnorm_1/ReadVariableOp_2:value:09model/basemodel/batch_normalization/batchnorm_1/mul_2:z:0*
T0*
_output_shapes
:@25
3model/basemodel/batch_normalization/batchnorm_1/sub�
5model/basemodel/batch_normalization/batchnorm_1/add_1AddV29model/basemodel/batch_normalization/batchnorm_1/mul_1:z:07model/basemodel/batch_normalization/batchnorm_1/sub:z:0*
T0*,
_output_shapes
:����������@27
5model/basemodel/batch_normalization/batchnorm_1/add_1�
#model/basemodel/activation_2/Relu_1Relu;model/basemodel/batch_normalization_2/batchnorm_1/add_1:z:0*
T0*,
_output_shapes
:����������@2%
#model/basemodel/activation_2/Relu_1�
#model/basemodel/activation_1/Relu_1Relu;model/basemodel/batch_normalization_1/batchnorm_1/add_1:z:0*
T0*,
_output_shapes
:����������@2%
#model/basemodel/activation_1/Relu_1�
!model/basemodel/activation/Relu_1Relu9model/basemodel/batch_normalization/batchnorm_1/add_1:z:0*
T0*,
_output_shapes
:����������@2#
!model/basemodel/activation/Relu_1�
*model/basemodel/stream_2_drop_1/Identity_1Identity1model/basemodel/activation_2/Relu_1:activations:0*
T0*,
_output_shapes
:����������@2,
*model/basemodel/stream_2_drop_1/Identity_1�
*model/basemodel/stream_1_drop_1/Identity_1Identity1model/basemodel/activation_1/Relu_1:activations:0*
T0*,
_output_shapes
:����������@2,
*model/basemodel/stream_1_drop_1/Identity_1�
*model/basemodel/stream_0_drop_1/Identity_1Identity/model/basemodel/activation/Relu_1:activations:0*
T0*,
_output_shapes
:����������@2,
*model/basemodel/stream_0_drop_1/Identity_1�
Amodel/basemodel/global_average_pooling1d/Mean_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2C
Amodel/basemodel/global_average_pooling1d/Mean_1/reduction_indices�
/model/basemodel/global_average_pooling1d/Mean_1Mean3model/basemodel/stream_0_drop_1/Identity_1:output:0Jmodel/basemodel/global_average_pooling1d/Mean_1/reduction_indices:output:0*
T0*'
_output_shapes
:���������@21
/model/basemodel/global_average_pooling1d/Mean_1�
Cmodel/basemodel/global_average_pooling1d_1/Mean_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2E
Cmodel/basemodel/global_average_pooling1d_1/Mean_1/reduction_indices�
1model/basemodel/global_average_pooling1d_1/Mean_1Mean3model/basemodel/stream_1_drop_1/Identity_1:output:0Lmodel/basemodel/global_average_pooling1d_1/Mean_1/reduction_indices:output:0*
T0*'
_output_shapes
:���������@23
1model/basemodel/global_average_pooling1d_1/Mean_1�
Cmodel/basemodel/global_average_pooling1d_2/Mean_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2E
Cmodel/basemodel/global_average_pooling1d_2/Mean_1/reduction_indices�
1model/basemodel/global_average_pooling1d_2/Mean_1Mean3model/basemodel/stream_2_drop_1/Identity_1:output:0Lmodel/basemodel/global_average_pooling1d_2/Mean_1/reduction_indices:output:0*
T0*'
_output_shapes
:���������@23
1model/basemodel/global_average_pooling1d_2/Mean_1�
)model/basemodel/concatenate/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :2+
)model/basemodel/concatenate/concat_1/axis�
$model/basemodel/concatenate/concat_1ConcatV28model/basemodel/global_average_pooling1d/Mean_1:output:0:model/basemodel/global_average_pooling1d_1/Mean_1:output:0:model/basemodel/global_average_pooling1d_2/Mean_1:output:02model/basemodel/concatenate/concat_1/axis:output:0*
N*
T0*(
_output_shapes
:����������2&
$model/basemodel/concatenate/concat_1�
*model/basemodel/dense_1_dropout/Identity_1Identity-model/basemodel/concatenate/concat_1:output:0*
T0*(
_output_shapes
:����������2,
*model/basemodel/dense_1_dropout/Identity_1�
/model/basemodel/dense_1/MatMul_1/ReadVariableOpReadVariableOp6model_basemodel_dense_1_matmul_readvariableop_resource*
_output_shapes
:	�T*
dtype021
/model/basemodel/dense_1/MatMul_1/ReadVariableOp�
 model/basemodel/dense_1/MatMul_1MatMul3model/basemodel/dense_1_dropout/Identity_1:output:07model/basemodel/dense_1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������T2"
 model/basemodel/dense_1/MatMul_1�
0model/basemodel/dense_1/BiasAdd_1/ReadVariableOpReadVariableOp7model_basemodel_dense_1_biasadd_readvariableop_resource*
_output_shapes
:T*
dtype022
0model/basemodel/dense_1/BiasAdd_1/ReadVariableOp�
!model/basemodel/dense_1/BiasAdd_1BiasAdd*model/basemodel/dense_1/MatMul_1:product:08model/basemodel/dense_1/BiasAdd_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������T2#
!model/basemodel/dense_1/BiasAdd_1�
@model/basemodel/batch_normalization_3/batchnorm_1/ReadVariableOpReadVariableOpGmodel_basemodel_batch_normalization_3_batchnorm_readvariableop_resource*
_output_shapes
:T*
dtype02B
@model/basemodel/batch_normalization_3/batchnorm_1/ReadVariableOp�
7model/basemodel/batch_normalization_3/batchnorm_1/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:29
7model/basemodel/batch_normalization_3/batchnorm_1/add/y�
5model/basemodel/batch_normalization_3/batchnorm_1/addAddV2Hmodel/basemodel/batch_normalization_3/batchnorm_1/ReadVariableOp:value:0@model/basemodel/batch_normalization_3/batchnorm_1/add/y:output:0*
T0*
_output_shapes
:T27
5model/basemodel/batch_normalization_3/batchnorm_1/add�
7model/basemodel/batch_normalization_3/batchnorm_1/RsqrtRsqrt9model/basemodel/batch_normalization_3/batchnorm_1/add:z:0*
T0*
_output_shapes
:T29
7model/basemodel/batch_normalization_3/batchnorm_1/Rsqrt�
Dmodel/basemodel/batch_normalization_3/batchnorm_1/mul/ReadVariableOpReadVariableOpKmodel_basemodel_batch_normalization_3_batchnorm_mul_readvariableop_resource*
_output_shapes
:T*
dtype02F
Dmodel/basemodel/batch_normalization_3/batchnorm_1/mul/ReadVariableOp�
5model/basemodel/batch_normalization_3/batchnorm_1/mulMul;model/basemodel/batch_normalization_3/batchnorm_1/Rsqrt:y:0Lmodel/basemodel/batch_normalization_3/batchnorm_1/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:T27
5model/basemodel/batch_normalization_3/batchnorm_1/mul�
7model/basemodel/batch_normalization_3/batchnorm_1/mul_1Mul*model/basemodel/dense_1/BiasAdd_1:output:09model/basemodel/batch_normalization_3/batchnorm_1/mul:z:0*
T0*'
_output_shapes
:���������T29
7model/basemodel/batch_normalization_3/batchnorm_1/mul_1�
Bmodel/basemodel/batch_normalization_3/batchnorm_1/ReadVariableOp_1ReadVariableOpImodel_basemodel_batch_normalization_3_batchnorm_readvariableop_1_resource*
_output_shapes
:T*
dtype02D
Bmodel/basemodel/batch_normalization_3/batchnorm_1/ReadVariableOp_1�
7model/basemodel/batch_normalization_3/batchnorm_1/mul_2MulJmodel/basemodel/batch_normalization_3/batchnorm_1/ReadVariableOp_1:value:09model/basemodel/batch_normalization_3/batchnorm_1/mul:z:0*
T0*
_output_shapes
:T29
7model/basemodel/batch_normalization_3/batchnorm_1/mul_2�
Bmodel/basemodel/batch_normalization_3/batchnorm_1/ReadVariableOp_2ReadVariableOpImodel_basemodel_batch_normalization_3_batchnorm_readvariableop_2_resource*
_output_shapes
:T*
dtype02D
Bmodel/basemodel/batch_normalization_3/batchnorm_1/ReadVariableOp_2�
5model/basemodel/batch_normalization_3/batchnorm_1/subSubJmodel/basemodel/batch_normalization_3/batchnorm_1/ReadVariableOp_2:value:0;model/basemodel/batch_normalization_3/batchnorm_1/mul_2:z:0*
T0*
_output_shapes
:T27
5model/basemodel/batch_normalization_3/batchnorm_1/sub�
7model/basemodel/batch_normalization_3/batchnorm_1/add_1AddV2;model/basemodel/batch_normalization_3/batchnorm_1/mul_1:z:09model/basemodel/batch_normalization_3/batchnorm_1/sub:z:0*
T0*'
_output_shapes
:���������T29
7model/basemodel/batch_normalization_3/batchnorm_1/add_1�
,model/basemodel/dense_activation_1/Sigmoid_1Sigmoid;model/basemodel/batch_normalization_3/batchnorm_1/add_1:z:0*
T0*'
_output_shapes
:���������T2.
,model/basemodel/dense_activation_1/Sigmoid_1�
model/distance/subSub.model/basemodel/dense_activation_1/Sigmoid:y:00model/basemodel/dense_activation_1/Sigmoid_1:y:0*
T0*'
_output_shapes
:���������T2
model/distance/sub�
model/distance/SquareSquaremodel/distance/sub:z:0*
T0*'
_output_shapes
:���������T2
model/distance/Square�
$model/distance/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
���������2&
$model/distance/Sum/reduction_indices�
model/distance/SumSummodel/distance/Square:y:0-model/distance/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:���������*
	keep_dims(2
model/distance/Sumq
model/distance/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
model/distance/Const�
model/distance/MaximumMaximummodel/distance/Sum:output:0model/distance/Const:output:0*
T0*'
_output_shapes
:���������2
model/distance/Maximum�
model/distance/SqrtSqrtmodel/distance/Maximum:z:0*
T0*'
_output_shapes
:���������2
model/distance/Sqrtr
IdentityIdentitymodel/distance/Sqrt:y:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp=^model/basemodel/batch_normalization/batchnorm/ReadVariableOp?^model/basemodel/batch_normalization/batchnorm/ReadVariableOp_1?^model/basemodel/batch_normalization/batchnorm/ReadVariableOp_2A^model/basemodel/batch_normalization/batchnorm/mul/ReadVariableOp?^model/basemodel/batch_normalization/batchnorm_1/ReadVariableOpA^model/basemodel/batch_normalization/batchnorm_1/ReadVariableOp_1A^model/basemodel/batch_normalization/batchnorm_1/ReadVariableOp_2C^model/basemodel/batch_normalization/batchnorm_1/mul/ReadVariableOp?^model/basemodel/batch_normalization_1/batchnorm/ReadVariableOpA^model/basemodel/batch_normalization_1/batchnorm/ReadVariableOp_1A^model/basemodel/batch_normalization_1/batchnorm/ReadVariableOp_2C^model/basemodel/batch_normalization_1/batchnorm/mul/ReadVariableOpA^model/basemodel/batch_normalization_1/batchnorm_1/ReadVariableOpC^model/basemodel/batch_normalization_1/batchnorm_1/ReadVariableOp_1C^model/basemodel/batch_normalization_1/batchnorm_1/ReadVariableOp_2E^model/basemodel/batch_normalization_1/batchnorm_1/mul/ReadVariableOp?^model/basemodel/batch_normalization_2/batchnorm/ReadVariableOpA^model/basemodel/batch_normalization_2/batchnorm/ReadVariableOp_1A^model/basemodel/batch_normalization_2/batchnorm/ReadVariableOp_2C^model/basemodel/batch_normalization_2/batchnorm/mul/ReadVariableOpA^model/basemodel/batch_normalization_2/batchnorm_1/ReadVariableOpC^model/basemodel/batch_normalization_2/batchnorm_1/ReadVariableOp_1C^model/basemodel/batch_normalization_2/batchnorm_1/ReadVariableOp_2E^model/basemodel/batch_normalization_2/batchnorm_1/mul/ReadVariableOp?^model/basemodel/batch_normalization_3/batchnorm/ReadVariableOpA^model/basemodel/batch_normalization_3/batchnorm/ReadVariableOp_1A^model/basemodel/batch_normalization_3/batchnorm/ReadVariableOp_2C^model/basemodel/batch_normalization_3/batchnorm/mul/ReadVariableOpA^model/basemodel/batch_normalization_3/batchnorm_1/ReadVariableOpC^model/basemodel/batch_normalization_3/batchnorm_1/ReadVariableOp_1C^model/basemodel/batch_normalization_3/batchnorm_1/ReadVariableOp_2E^model/basemodel/batch_normalization_3/batchnorm_1/mul/ReadVariableOp/^model/basemodel/dense_1/BiasAdd/ReadVariableOp1^model/basemodel/dense_1/BiasAdd_1/ReadVariableOp.^model/basemodel/dense_1/MatMul/ReadVariableOp0^model/basemodel/dense_1/MatMul_1/ReadVariableOp7^model/basemodel/stream_0_conv_1/BiasAdd/ReadVariableOp9^model/basemodel/stream_0_conv_1/BiasAdd_1/ReadVariableOpC^model/basemodel/stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOpE^model/basemodel/stream_0_conv_1/conv1d_1/ExpandDims_1/ReadVariableOp7^model/basemodel/stream_1_conv_1/BiasAdd/ReadVariableOp9^model/basemodel/stream_1_conv_1/BiasAdd_1/ReadVariableOpC^model/basemodel/stream_1_conv_1/conv1d/ExpandDims_1/ReadVariableOpE^model/basemodel/stream_1_conv_1/conv1d_1/ExpandDims_1/ReadVariableOp7^model/basemodel/stream_2_conv_1/BiasAdd/ReadVariableOp9^model/basemodel/stream_2_conv_1/BiasAdd_1/ReadVariableOpC^model/basemodel/stream_2_conv_1/conv1d/ExpandDims_1/ReadVariableOpE^model/basemodel/stream_2_conv_1/conv1d_1/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*s
_input_shapesb
`:����������:����������: : : : : : : : : : : : : : : : : : : : : : : : 2|
<model/basemodel/batch_normalization/batchnorm/ReadVariableOp<model/basemodel/batch_normalization/batchnorm/ReadVariableOp2�
>model/basemodel/batch_normalization/batchnorm/ReadVariableOp_1>model/basemodel/batch_normalization/batchnorm/ReadVariableOp_12�
>model/basemodel/batch_normalization/batchnorm/ReadVariableOp_2>model/basemodel/batch_normalization/batchnorm/ReadVariableOp_22�
@model/basemodel/batch_normalization/batchnorm/mul/ReadVariableOp@model/basemodel/batch_normalization/batchnorm/mul/ReadVariableOp2�
>model/basemodel/batch_normalization/batchnorm_1/ReadVariableOp>model/basemodel/batch_normalization/batchnorm_1/ReadVariableOp2�
@model/basemodel/batch_normalization/batchnorm_1/ReadVariableOp_1@model/basemodel/batch_normalization/batchnorm_1/ReadVariableOp_12�
@model/basemodel/batch_normalization/batchnorm_1/ReadVariableOp_2@model/basemodel/batch_normalization/batchnorm_1/ReadVariableOp_22�
Bmodel/basemodel/batch_normalization/batchnorm_1/mul/ReadVariableOpBmodel/basemodel/batch_normalization/batchnorm_1/mul/ReadVariableOp2�
>model/basemodel/batch_normalization_1/batchnorm/ReadVariableOp>model/basemodel/batch_normalization_1/batchnorm/ReadVariableOp2�
@model/basemodel/batch_normalization_1/batchnorm/ReadVariableOp_1@model/basemodel/batch_normalization_1/batchnorm/ReadVariableOp_12�
@model/basemodel/batch_normalization_1/batchnorm/ReadVariableOp_2@model/basemodel/batch_normalization_1/batchnorm/ReadVariableOp_22�
Bmodel/basemodel/batch_normalization_1/batchnorm/mul/ReadVariableOpBmodel/basemodel/batch_normalization_1/batchnorm/mul/ReadVariableOp2�
@model/basemodel/batch_normalization_1/batchnorm_1/ReadVariableOp@model/basemodel/batch_normalization_1/batchnorm_1/ReadVariableOp2�
Bmodel/basemodel/batch_normalization_1/batchnorm_1/ReadVariableOp_1Bmodel/basemodel/batch_normalization_1/batchnorm_1/ReadVariableOp_12�
Bmodel/basemodel/batch_normalization_1/batchnorm_1/ReadVariableOp_2Bmodel/basemodel/batch_normalization_1/batchnorm_1/ReadVariableOp_22�
Dmodel/basemodel/batch_normalization_1/batchnorm_1/mul/ReadVariableOpDmodel/basemodel/batch_normalization_1/batchnorm_1/mul/ReadVariableOp2�
>model/basemodel/batch_normalization_2/batchnorm/ReadVariableOp>model/basemodel/batch_normalization_2/batchnorm/ReadVariableOp2�
@model/basemodel/batch_normalization_2/batchnorm/ReadVariableOp_1@model/basemodel/batch_normalization_2/batchnorm/ReadVariableOp_12�
@model/basemodel/batch_normalization_2/batchnorm/ReadVariableOp_2@model/basemodel/batch_normalization_2/batchnorm/ReadVariableOp_22�
Bmodel/basemodel/batch_normalization_2/batchnorm/mul/ReadVariableOpBmodel/basemodel/batch_normalization_2/batchnorm/mul/ReadVariableOp2�
@model/basemodel/batch_normalization_2/batchnorm_1/ReadVariableOp@model/basemodel/batch_normalization_2/batchnorm_1/ReadVariableOp2�
Bmodel/basemodel/batch_normalization_2/batchnorm_1/ReadVariableOp_1Bmodel/basemodel/batch_normalization_2/batchnorm_1/ReadVariableOp_12�
Bmodel/basemodel/batch_normalization_2/batchnorm_1/ReadVariableOp_2Bmodel/basemodel/batch_normalization_2/batchnorm_1/ReadVariableOp_22�
Dmodel/basemodel/batch_normalization_2/batchnorm_1/mul/ReadVariableOpDmodel/basemodel/batch_normalization_2/batchnorm_1/mul/ReadVariableOp2�
>model/basemodel/batch_normalization_3/batchnorm/ReadVariableOp>model/basemodel/batch_normalization_3/batchnorm/ReadVariableOp2�
@model/basemodel/batch_normalization_3/batchnorm/ReadVariableOp_1@model/basemodel/batch_normalization_3/batchnorm/ReadVariableOp_12�
@model/basemodel/batch_normalization_3/batchnorm/ReadVariableOp_2@model/basemodel/batch_normalization_3/batchnorm/ReadVariableOp_22�
Bmodel/basemodel/batch_normalization_3/batchnorm/mul/ReadVariableOpBmodel/basemodel/batch_normalization_3/batchnorm/mul/ReadVariableOp2�
@model/basemodel/batch_normalization_3/batchnorm_1/ReadVariableOp@model/basemodel/batch_normalization_3/batchnorm_1/ReadVariableOp2�
Bmodel/basemodel/batch_normalization_3/batchnorm_1/ReadVariableOp_1Bmodel/basemodel/batch_normalization_3/batchnorm_1/ReadVariableOp_12�
Bmodel/basemodel/batch_normalization_3/batchnorm_1/ReadVariableOp_2Bmodel/basemodel/batch_normalization_3/batchnorm_1/ReadVariableOp_22�
Dmodel/basemodel/batch_normalization_3/batchnorm_1/mul/ReadVariableOpDmodel/basemodel/batch_normalization_3/batchnorm_1/mul/ReadVariableOp2`
.model/basemodel/dense_1/BiasAdd/ReadVariableOp.model/basemodel/dense_1/BiasAdd/ReadVariableOp2d
0model/basemodel/dense_1/BiasAdd_1/ReadVariableOp0model/basemodel/dense_1/BiasAdd_1/ReadVariableOp2^
-model/basemodel/dense_1/MatMul/ReadVariableOp-model/basemodel/dense_1/MatMul/ReadVariableOp2b
/model/basemodel/dense_1/MatMul_1/ReadVariableOp/model/basemodel/dense_1/MatMul_1/ReadVariableOp2p
6model/basemodel/stream_0_conv_1/BiasAdd/ReadVariableOp6model/basemodel/stream_0_conv_1/BiasAdd/ReadVariableOp2t
8model/basemodel/stream_0_conv_1/BiasAdd_1/ReadVariableOp8model/basemodel/stream_0_conv_1/BiasAdd_1/ReadVariableOp2�
Bmodel/basemodel/stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOpBmodel/basemodel/stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp2�
Dmodel/basemodel/stream_0_conv_1/conv1d_1/ExpandDims_1/ReadVariableOpDmodel/basemodel/stream_0_conv_1/conv1d_1/ExpandDims_1/ReadVariableOp2p
6model/basemodel/stream_1_conv_1/BiasAdd/ReadVariableOp6model/basemodel/stream_1_conv_1/BiasAdd/ReadVariableOp2t
8model/basemodel/stream_1_conv_1/BiasAdd_1/ReadVariableOp8model/basemodel/stream_1_conv_1/BiasAdd_1/ReadVariableOp2�
Bmodel/basemodel/stream_1_conv_1/conv1d/ExpandDims_1/ReadVariableOpBmodel/basemodel/stream_1_conv_1/conv1d/ExpandDims_1/ReadVariableOp2�
Dmodel/basemodel/stream_1_conv_1/conv1d_1/ExpandDims_1/ReadVariableOpDmodel/basemodel/stream_1_conv_1/conv1d_1/ExpandDims_1/ReadVariableOp2p
6model/basemodel/stream_2_conv_1/BiasAdd/ReadVariableOp6model/basemodel/stream_2_conv_1/BiasAdd/ReadVariableOp2t
8model/basemodel/stream_2_conv_1/BiasAdd_1/ReadVariableOp8model/basemodel/stream_2_conv_1/BiasAdd_1/ReadVariableOp2�
Bmodel/basemodel/stream_2_conv_1/conv1d/ExpandDims_1/ReadVariableOpBmodel/basemodel/stream_2_conv_1/conv1d/ExpandDims_1/ReadVariableOp2�
Dmodel/basemodel/stream_2_conv_1/conv1d_1/ExpandDims_1/ReadVariableOpDmodel/basemodel/stream_2_conv_1/conv1d_1/ExpandDims_1/ReadVariableOp:Y U
,
_output_shapes
:����������
%
_user_specified_nameleft_inputs:ZV
,
_output_shapes
:����������
&
_user_specified_nameright_inputs
�
o
P__inference_stream_0_input_drop_layer_call_and_return_conditional_losses_1988073

inputs
identity�c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
dropout/Constx
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:����������2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:����������*
dtype0*
seed�*
seed2�2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>2
dropout/GreaterEqual/y�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:����������2
dropout/GreaterEqual�
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:����������2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:����������2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
7__inference_batch_normalization_1_layer_call_fn_1988434

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *[
fVRT
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_19847822
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:����������@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:����������@
 
_user_specified_nameinputs
�
o
P__inference_stream_0_input_drop_layer_call_and_return_conditional_losses_1985386

inputs
identity�c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
dropout/Constx
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:����������2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:����������*
dtype0*
seed�*
seed2�2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>2
dropout/GreaterEqual/y�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:����������2
dropout/GreaterEqual�
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:����������2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:����������2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1984483

inputs/
!batchnorm_readvariableop_resource:T3
%batchnorm_mul_readvariableop_resource:T1
#batchnorm_readvariableop_1_resource:T1
#batchnorm_readvariableop_2_resource:T
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOp�
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
 *o�:2
batchnorm/add/y�
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:T2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:T2
batchnorm/Rsqrt�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:T*
dtype02
batchnorm/mul/ReadVariableOp�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:T2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������T2
batchnorm/mul_1�
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:T*
dtype02
batchnorm/ReadVariableOp_1�
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:T2
batchnorm/mul_2�
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:T*
dtype02
batchnorm/ReadVariableOp_2�
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:T2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������T2
batchnorm/add_1n
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������T2

Identity�
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������T: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������T
 
_user_specified_nameinputs
�
j
1__inference_stream_2_drop_1_layer_call_fn_1988809

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_stream_2_drop_1_layer_call_and_return_conditional_losses_19851352
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:����������@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������@22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:����������@
 
_user_specified_nameinputs
�
�
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1984249

inputs/
!batchnorm_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@1
#batchnorm_readvariableop_1_resource:@1
#batchnorm_readvariableop_2_resource:@
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOp�
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
 *o�:2
batchnorm/add/y�
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@2
batchnorm/Rsqrt�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOp�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2
batchnorm/mul�
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :������������������@2
batchnorm/mul_1�
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp_1�
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:@2
batchnorm/mul_2�
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp_2�
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :������������������@2
batchnorm/add_1{
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :������������������@2

Identity�
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:������������������@: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :������������������@
 
_user_specified_nameinputs
�
�
__inference_loss_fn_0_1989058T
>stream_0_conv_1_kernel_regularizer_abs_readvariableop_resource:@
identity��5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp�
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp>stream_0_conv_1_kernel_regularizer_abs_readvariableop_resource*"
_output_shapes
:@*
dtype027
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp�
&stream_0_conv_1/kernel/Regularizer/AbsAbs=stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_0_conv_1/kernel/Regularizer/Abs�
(stream_0_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_0_conv_1/kernel/Regularizer/Const�
&stream_0_conv_1/kernel/Regularizer/SumSum*stream_0_conv_1/kernel/Regularizer/Abs:y:01stream_0_conv_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/Sum�
(stream_0_conv_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2*
(stream_0_conv_1/kernel/Regularizer/mul/x�
&stream_0_conv_1/kernel/Regularizer/mulMul1stream_0_conv_1/kernel/Regularizer/mul/x:output:0/stream_0_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/mult
IdentityIdentity*stream_0_conv_1/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: 2

Identity�
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
�
�
__inference_loss_fn_2_1989080T
>stream_2_conv_1_kernel_regularizer_abs_readvariableop_resource:@
identity��5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp�
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp>stream_2_conv_1_kernel_regularizer_abs_readvariableop_resource*"
_output_shapes
:@*
dtype027
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp�
&stream_2_conv_1/kernel/Regularizer/AbsAbs=stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_2_conv_1/kernel/Regularizer/Abs�
(stream_2_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_2_conv_1/kernel/Regularizer/Const�
&stream_2_conv_1/kernel/Regularizer/SumSum*stream_2_conv_1/kernel/Regularizer/Abs:y:01stream_2_conv_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&stream_2_conv_1/kernel/Regularizer/Sum�
(stream_2_conv_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2*
(stream_2_conv_1/kernel/Regularizer/mul/x�
&stream_2_conv_1/kernel/Regularizer/mulMul1stream_2_conv_1/kernel/Regularizer/mul/x:output:0/stream_2_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_2_conv_1/kernel/Regularizer/mult
IdentityIdentity*stream_2_conv_1/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: 2

Identity�
NoOpNoOp6^stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2n
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp
��
�/
B__inference_model_layer_call_and_return_conditional_losses_1987475
inputs_0
inputs_1[
Ebasemodel_stream_2_conv_1_conv1d_expanddims_1_readvariableop_resource:@G
9basemodel_stream_2_conv_1_biasadd_readvariableop_resource:@[
Ebasemodel_stream_1_conv_1_conv1d_expanddims_1_readvariableop_resource:@G
9basemodel_stream_1_conv_1_biasadd_readvariableop_resource:@[
Ebasemodel_stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource:@G
9basemodel_stream_0_conv_1_biasadd_readvariableop_resource:@U
Gbasemodel_batch_normalization_2_assignmovingavg_readvariableop_resource:@W
Ibasemodel_batch_normalization_2_assignmovingavg_1_readvariableop_resource:@S
Ebasemodel_batch_normalization_2_batchnorm_mul_readvariableop_resource:@O
Abasemodel_batch_normalization_2_batchnorm_readvariableop_resource:@U
Gbasemodel_batch_normalization_1_assignmovingavg_readvariableop_resource:@W
Ibasemodel_batch_normalization_1_assignmovingavg_1_readvariableop_resource:@S
Ebasemodel_batch_normalization_1_batchnorm_mul_readvariableop_resource:@O
Abasemodel_batch_normalization_1_batchnorm_readvariableop_resource:@S
Ebasemodel_batch_normalization_assignmovingavg_readvariableop_resource:@U
Gbasemodel_batch_normalization_assignmovingavg_1_readvariableop_resource:@Q
Cbasemodel_batch_normalization_batchnorm_mul_readvariableop_resource:@M
?basemodel_batch_normalization_batchnorm_readvariableop_resource:@C
0basemodel_dense_1_matmul_readvariableop_resource:	�T?
1basemodel_dense_1_biasadd_readvariableop_resource:TU
Gbasemodel_batch_normalization_3_assignmovingavg_readvariableop_resource:TW
Ibasemodel_batch_normalization_3_assignmovingavg_1_readvariableop_resource:TS
Ebasemodel_batch_normalization_3_batchnorm_mul_readvariableop_resource:TO
Abasemodel_batch_normalization_3_batchnorm_readvariableop_resource:T
identity��-basemodel/batch_normalization/AssignMovingAvg�<basemodel/batch_normalization/AssignMovingAvg/ReadVariableOp�/basemodel/batch_normalization/AssignMovingAvg_1�>basemodel/batch_normalization/AssignMovingAvg_1/ReadVariableOp�/basemodel/batch_normalization/AssignMovingAvg_2�>basemodel/batch_normalization/AssignMovingAvg_2/ReadVariableOp�/basemodel/batch_normalization/AssignMovingAvg_3�>basemodel/batch_normalization/AssignMovingAvg_3/ReadVariableOp�6basemodel/batch_normalization/batchnorm/ReadVariableOp�:basemodel/batch_normalization/batchnorm/mul/ReadVariableOp�8basemodel/batch_normalization/batchnorm_1/ReadVariableOp�<basemodel/batch_normalization/batchnorm_1/mul/ReadVariableOp�/basemodel/batch_normalization_1/AssignMovingAvg�>basemodel/batch_normalization_1/AssignMovingAvg/ReadVariableOp�1basemodel/batch_normalization_1/AssignMovingAvg_1�@basemodel/batch_normalization_1/AssignMovingAvg_1/ReadVariableOp�1basemodel/batch_normalization_1/AssignMovingAvg_2�@basemodel/batch_normalization_1/AssignMovingAvg_2/ReadVariableOp�1basemodel/batch_normalization_1/AssignMovingAvg_3�@basemodel/batch_normalization_1/AssignMovingAvg_3/ReadVariableOp�8basemodel/batch_normalization_1/batchnorm/ReadVariableOp�<basemodel/batch_normalization_1/batchnorm/mul/ReadVariableOp�:basemodel/batch_normalization_1/batchnorm_1/ReadVariableOp�>basemodel/batch_normalization_1/batchnorm_1/mul/ReadVariableOp�/basemodel/batch_normalization_2/AssignMovingAvg�>basemodel/batch_normalization_2/AssignMovingAvg/ReadVariableOp�1basemodel/batch_normalization_2/AssignMovingAvg_1�@basemodel/batch_normalization_2/AssignMovingAvg_1/ReadVariableOp�1basemodel/batch_normalization_2/AssignMovingAvg_2�@basemodel/batch_normalization_2/AssignMovingAvg_2/ReadVariableOp�1basemodel/batch_normalization_2/AssignMovingAvg_3�@basemodel/batch_normalization_2/AssignMovingAvg_3/ReadVariableOp�8basemodel/batch_normalization_2/batchnorm/ReadVariableOp�<basemodel/batch_normalization_2/batchnorm/mul/ReadVariableOp�:basemodel/batch_normalization_2/batchnorm_1/ReadVariableOp�>basemodel/batch_normalization_2/batchnorm_1/mul/ReadVariableOp�/basemodel/batch_normalization_3/AssignMovingAvg�>basemodel/batch_normalization_3/AssignMovingAvg/ReadVariableOp�1basemodel/batch_normalization_3/AssignMovingAvg_1�@basemodel/batch_normalization_3/AssignMovingAvg_1/ReadVariableOp�1basemodel/batch_normalization_3/AssignMovingAvg_2�@basemodel/batch_normalization_3/AssignMovingAvg_2/ReadVariableOp�1basemodel/batch_normalization_3/AssignMovingAvg_3�@basemodel/batch_normalization_3/AssignMovingAvg_3/ReadVariableOp�8basemodel/batch_normalization_3/batchnorm/ReadVariableOp�<basemodel/batch_normalization_3/batchnorm/mul/ReadVariableOp�:basemodel/batch_normalization_3/batchnorm_1/ReadVariableOp�>basemodel/batch_normalization_3/batchnorm_1/mul/ReadVariableOp�(basemodel/dense_1/BiasAdd/ReadVariableOp�*basemodel/dense_1/BiasAdd_1/ReadVariableOp�'basemodel/dense_1/MatMul/ReadVariableOp�)basemodel/dense_1/MatMul_1/ReadVariableOp�0basemodel/stream_0_conv_1/BiasAdd/ReadVariableOp�2basemodel/stream_0_conv_1/BiasAdd_1/ReadVariableOp�<basemodel/stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp�>basemodel/stream_0_conv_1/conv1d_1/ExpandDims_1/ReadVariableOp�0basemodel/stream_1_conv_1/BiasAdd/ReadVariableOp�2basemodel/stream_1_conv_1/BiasAdd_1/ReadVariableOp�<basemodel/stream_1_conv_1/conv1d/ExpandDims_1/ReadVariableOp�>basemodel/stream_1_conv_1/conv1d_1/ExpandDims_1/ReadVariableOp�0basemodel/stream_2_conv_1/BiasAdd/ReadVariableOp�2basemodel/stream_2_conv_1/BiasAdd_1/ReadVariableOp�<basemodel/stream_2_conv_1/conv1d/ExpandDims_1/ReadVariableOp�>basemodel/stream_2_conv_1/conv1d_1/ExpandDims_1/ReadVariableOp�-dense_1/kernel/Regularizer/Abs/ReadVariableOp�5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp�8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp�5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp�
+basemodel/stream_2_input_drop/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2-
+basemodel/stream_2_input_drop/dropout/Const�
)basemodel/stream_2_input_drop/dropout/MulMulinputs_04basemodel/stream_2_input_drop/dropout/Const:output:0*
T0*,
_output_shapes
:����������2+
)basemodel/stream_2_input_drop/dropout/Mul�
+basemodel/stream_2_input_drop/dropout/ShapeShapeinputs_0*
T0*
_output_shapes
:2-
+basemodel/stream_2_input_drop/dropout/Shape�
Bbasemodel/stream_2_input_drop/dropout/random_uniform/RandomUniformRandomUniform4basemodel/stream_2_input_drop/dropout/Shape:output:0*
T0*,
_output_shapes
:����������*
dtype0*
seed�*
seed2�2D
Bbasemodel/stream_2_input_drop/dropout/random_uniform/RandomUniform�
4basemodel/stream_2_input_drop/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>26
4basemodel/stream_2_input_drop/dropout/GreaterEqual/y�
2basemodel/stream_2_input_drop/dropout/GreaterEqualGreaterEqualKbasemodel/stream_2_input_drop/dropout/random_uniform/RandomUniform:output:0=basemodel/stream_2_input_drop/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:����������24
2basemodel/stream_2_input_drop/dropout/GreaterEqual�
*basemodel/stream_2_input_drop/dropout/CastCast6basemodel/stream_2_input_drop/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:����������2,
*basemodel/stream_2_input_drop/dropout/Cast�
+basemodel/stream_2_input_drop/dropout/Mul_1Mul-basemodel/stream_2_input_drop/dropout/Mul:z:0.basemodel/stream_2_input_drop/dropout/Cast:y:0*
T0*,
_output_shapes
:����������2-
+basemodel/stream_2_input_drop/dropout/Mul_1�
+basemodel/stream_1_input_drop/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2-
+basemodel/stream_1_input_drop/dropout/Const�
)basemodel/stream_1_input_drop/dropout/MulMulinputs_04basemodel/stream_1_input_drop/dropout/Const:output:0*
T0*,
_output_shapes
:����������2+
)basemodel/stream_1_input_drop/dropout/Mul�
+basemodel/stream_1_input_drop/dropout/ShapeShapeinputs_0*
T0*
_output_shapes
:2-
+basemodel/stream_1_input_drop/dropout/Shape�
Bbasemodel/stream_1_input_drop/dropout/random_uniform/RandomUniformRandomUniform4basemodel/stream_1_input_drop/dropout/Shape:output:0*
T0*,
_output_shapes
:����������*
dtype0*
seed�*
seed2�2D
Bbasemodel/stream_1_input_drop/dropout/random_uniform/RandomUniform�
4basemodel/stream_1_input_drop/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>26
4basemodel/stream_1_input_drop/dropout/GreaterEqual/y�
2basemodel/stream_1_input_drop/dropout/GreaterEqualGreaterEqualKbasemodel/stream_1_input_drop/dropout/random_uniform/RandomUniform:output:0=basemodel/stream_1_input_drop/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:����������24
2basemodel/stream_1_input_drop/dropout/GreaterEqual�
*basemodel/stream_1_input_drop/dropout/CastCast6basemodel/stream_1_input_drop/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:����������2,
*basemodel/stream_1_input_drop/dropout/Cast�
+basemodel/stream_1_input_drop/dropout/Mul_1Mul-basemodel/stream_1_input_drop/dropout/Mul:z:0.basemodel/stream_1_input_drop/dropout/Cast:y:0*
T0*,
_output_shapes
:����������2-
+basemodel/stream_1_input_drop/dropout/Mul_1�
+basemodel/stream_0_input_drop/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2-
+basemodel/stream_0_input_drop/dropout/Const�
)basemodel/stream_0_input_drop/dropout/MulMulinputs_04basemodel/stream_0_input_drop/dropout/Const:output:0*
T0*,
_output_shapes
:����������2+
)basemodel/stream_0_input_drop/dropout/Mul�
+basemodel/stream_0_input_drop/dropout/ShapeShapeinputs_0*
T0*
_output_shapes
:2-
+basemodel/stream_0_input_drop/dropout/Shape�
Bbasemodel/stream_0_input_drop/dropout/random_uniform/RandomUniformRandomUniform4basemodel/stream_0_input_drop/dropout/Shape:output:0*
T0*,
_output_shapes
:����������*
dtype0*
seed�*
seed2�2D
Bbasemodel/stream_0_input_drop/dropout/random_uniform/RandomUniform�
4basemodel/stream_0_input_drop/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>26
4basemodel/stream_0_input_drop/dropout/GreaterEqual/y�
2basemodel/stream_0_input_drop/dropout/GreaterEqualGreaterEqualKbasemodel/stream_0_input_drop/dropout/random_uniform/RandomUniform:output:0=basemodel/stream_0_input_drop/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:����������24
2basemodel/stream_0_input_drop/dropout/GreaterEqual�
*basemodel/stream_0_input_drop/dropout/CastCast6basemodel/stream_0_input_drop/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:����������2,
*basemodel/stream_0_input_drop/dropout/Cast�
+basemodel/stream_0_input_drop/dropout/Mul_1Mul-basemodel/stream_0_input_drop/dropout/Mul:z:0.basemodel/stream_0_input_drop/dropout/Cast:y:0*
T0*,
_output_shapes
:����������2-
+basemodel/stream_0_input_drop/dropout/Mul_1�
/basemodel/stream_2_conv_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������21
/basemodel/stream_2_conv_1/conv1d/ExpandDims/dim�
+basemodel/stream_2_conv_1/conv1d/ExpandDims
ExpandDims/basemodel/stream_2_input_drop/dropout/Mul_1:z:08basemodel/stream_2_conv_1/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:����������2-
+basemodel/stream_2_conv_1/conv1d/ExpandDims�
<basemodel/stream_2_conv_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpEbasemodel_stream_2_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02>
<basemodel/stream_2_conv_1/conv1d/ExpandDims_1/ReadVariableOp�
1basemodel/stream_2_conv_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 23
1basemodel/stream_2_conv_1/conv1d/ExpandDims_1/dim�
-basemodel/stream_2_conv_1/conv1d/ExpandDims_1
ExpandDimsDbasemodel/stream_2_conv_1/conv1d/ExpandDims_1/ReadVariableOp:value:0:basemodel/stream_2_conv_1/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@2/
-basemodel/stream_2_conv_1/conv1d/ExpandDims_1�
 basemodel/stream_2_conv_1/conv1dConv2D4basemodel/stream_2_conv_1/conv1d/ExpandDims:output:06basemodel/stream_2_conv_1/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:����������@*
paddingSAME*
strides
2"
 basemodel/stream_2_conv_1/conv1d�
(basemodel/stream_2_conv_1/conv1d/SqueezeSqueeze)basemodel/stream_2_conv_1/conv1d:output:0*
T0*,
_output_shapes
:����������@*
squeeze_dims

���������2*
(basemodel/stream_2_conv_1/conv1d/Squeeze�
0basemodel/stream_2_conv_1/BiasAdd/ReadVariableOpReadVariableOp9basemodel_stream_2_conv_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype022
0basemodel/stream_2_conv_1/BiasAdd/ReadVariableOp�
!basemodel/stream_2_conv_1/BiasAddBiasAdd1basemodel/stream_2_conv_1/conv1d/Squeeze:output:08basemodel/stream_2_conv_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������@2#
!basemodel/stream_2_conv_1/BiasAdd�
/basemodel/stream_1_conv_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������21
/basemodel/stream_1_conv_1/conv1d/ExpandDims/dim�
+basemodel/stream_1_conv_1/conv1d/ExpandDims
ExpandDims/basemodel/stream_1_input_drop/dropout/Mul_1:z:08basemodel/stream_1_conv_1/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:����������2-
+basemodel/stream_1_conv_1/conv1d/ExpandDims�
<basemodel/stream_1_conv_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpEbasemodel_stream_1_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02>
<basemodel/stream_1_conv_1/conv1d/ExpandDims_1/ReadVariableOp�
1basemodel/stream_1_conv_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 23
1basemodel/stream_1_conv_1/conv1d/ExpandDims_1/dim�
-basemodel/stream_1_conv_1/conv1d/ExpandDims_1
ExpandDimsDbasemodel/stream_1_conv_1/conv1d/ExpandDims_1/ReadVariableOp:value:0:basemodel/stream_1_conv_1/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@2/
-basemodel/stream_1_conv_1/conv1d/ExpandDims_1�
 basemodel/stream_1_conv_1/conv1dConv2D4basemodel/stream_1_conv_1/conv1d/ExpandDims:output:06basemodel/stream_1_conv_1/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:����������@*
paddingSAME*
strides
2"
 basemodel/stream_1_conv_1/conv1d�
(basemodel/stream_1_conv_1/conv1d/SqueezeSqueeze)basemodel/stream_1_conv_1/conv1d:output:0*
T0*,
_output_shapes
:����������@*
squeeze_dims

���������2*
(basemodel/stream_1_conv_1/conv1d/Squeeze�
0basemodel/stream_1_conv_1/BiasAdd/ReadVariableOpReadVariableOp9basemodel_stream_1_conv_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype022
0basemodel/stream_1_conv_1/BiasAdd/ReadVariableOp�
!basemodel/stream_1_conv_1/BiasAddBiasAdd1basemodel/stream_1_conv_1/conv1d/Squeeze:output:08basemodel/stream_1_conv_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������@2#
!basemodel/stream_1_conv_1/BiasAdd�
/basemodel/stream_0_conv_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������21
/basemodel/stream_0_conv_1/conv1d/ExpandDims/dim�
+basemodel/stream_0_conv_1/conv1d/ExpandDims
ExpandDims/basemodel/stream_0_input_drop/dropout/Mul_1:z:08basemodel/stream_0_conv_1/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:����������2-
+basemodel/stream_0_conv_1/conv1d/ExpandDims�
<basemodel/stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpEbasemodel_stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02>
<basemodel/stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp�
1basemodel/stream_0_conv_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 23
1basemodel/stream_0_conv_1/conv1d/ExpandDims_1/dim�
-basemodel/stream_0_conv_1/conv1d/ExpandDims_1
ExpandDimsDbasemodel/stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp:value:0:basemodel/stream_0_conv_1/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@2/
-basemodel/stream_0_conv_1/conv1d/ExpandDims_1�
 basemodel/stream_0_conv_1/conv1dConv2D4basemodel/stream_0_conv_1/conv1d/ExpandDims:output:06basemodel/stream_0_conv_1/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:����������@*
paddingSAME*
strides
2"
 basemodel/stream_0_conv_1/conv1d�
(basemodel/stream_0_conv_1/conv1d/SqueezeSqueeze)basemodel/stream_0_conv_1/conv1d:output:0*
T0*,
_output_shapes
:����������@*
squeeze_dims

���������2*
(basemodel/stream_0_conv_1/conv1d/Squeeze�
0basemodel/stream_0_conv_1/BiasAdd/ReadVariableOpReadVariableOp9basemodel_stream_0_conv_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype022
0basemodel/stream_0_conv_1/BiasAdd/ReadVariableOp�
!basemodel/stream_0_conv_1/BiasAddBiasAdd1basemodel/stream_0_conv_1/conv1d/Squeeze:output:08basemodel/stream_0_conv_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������@2#
!basemodel/stream_0_conv_1/BiasAdd�
>basemodel/batch_normalization_2/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2@
>basemodel/batch_normalization_2/moments/mean/reduction_indices�
,basemodel/batch_normalization_2/moments/meanMean*basemodel/stream_2_conv_1/BiasAdd:output:0Gbasemodel/batch_normalization_2/moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(2.
,basemodel/batch_normalization_2/moments/mean�
4basemodel/batch_normalization_2/moments/StopGradientStopGradient5basemodel/batch_normalization_2/moments/mean:output:0*
T0*"
_output_shapes
:@26
4basemodel/batch_normalization_2/moments/StopGradient�
9basemodel/batch_normalization_2/moments/SquaredDifferenceSquaredDifference*basemodel/stream_2_conv_1/BiasAdd:output:0=basemodel/batch_normalization_2/moments/StopGradient:output:0*
T0*,
_output_shapes
:����������@2;
9basemodel/batch_normalization_2/moments/SquaredDifference�
Bbasemodel/batch_normalization_2/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2D
Bbasemodel/batch_normalization_2/moments/variance/reduction_indices�
0basemodel/batch_normalization_2/moments/varianceMean=basemodel/batch_normalization_2/moments/SquaredDifference:z:0Kbasemodel/batch_normalization_2/moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(22
0basemodel/batch_normalization_2/moments/variance�
/basemodel/batch_normalization_2/moments/SqueezeSqueeze5basemodel/batch_normalization_2/moments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 21
/basemodel/batch_normalization_2/moments/Squeeze�
1basemodel/batch_normalization_2/moments/Squeeze_1Squeeze9basemodel/batch_normalization_2/moments/variance:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 23
1basemodel/batch_normalization_2/moments/Squeeze_1�
5basemodel/batch_normalization_2/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<27
5basemodel/batch_normalization_2/AssignMovingAvg/decay�
>basemodel/batch_normalization_2/AssignMovingAvg/ReadVariableOpReadVariableOpGbasemodel_batch_normalization_2_assignmovingavg_readvariableop_resource*
_output_shapes
:@*
dtype02@
>basemodel/batch_normalization_2/AssignMovingAvg/ReadVariableOp�
3basemodel/batch_normalization_2/AssignMovingAvg/subSubFbasemodel/batch_normalization_2/AssignMovingAvg/ReadVariableOp:value:08basemodel/batch_normalization_2/moments/Squeeze:output:0*
T0*
_output_shapes
:@25
3basemodel/batch_normalization_2/AssignMovingAvg/sub�
3basemodel/batch_normalization_2/AssignMovingAvg/mulMul7basemodel/batch_normalization_2/AssignMovingAvg/sub:z:0>basemodel/batch_normalization_2/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:@25
3basemodel/batch_normalization_2/AssignMovingAvg/mul�
/basemodel/batch_normalization_2/AssignMovingAvgAssignSubVariableOpGbasemodel_batch_normalization_2_assignmovingavg_readvariableop_resource7basemodel/batch_normalization_2/AssignMovingAvg/mul:z:0?^basemodel/batch_normalization_2/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype021
/basemodel/batch_normalization_2/AssignMovingAvg�
7basemodel/batch_normalization_2/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<29
7basemodel/batch_normalization_2/AssignMovingAvg_1/decay�
@basemodel/batch_normalization_2/AssignMovingAvg_1/ReadVariableOpReadVariableOpIbasemodel_batch_normalization_2_assignmovingavg_1_readvariableop_resource*
_output_shapes
:@*
dtype02B
@basemodel/batch_normalization_2/AssignMovingAvg_1/ReadVariableOp�
5basemodel/batch_normalization_2/AssignMovingAvg_1/subSubHbasemodel/batch_normalization_2/AssignMovingAvg_1/ReadVariableOp:value:0:basemodel/batch_normalization_2/moments/Squeeze_1:output:0*
T0*
_output_shapes
:@27
5basemodel/batch_normalization_2/AssignMovingAvg_1/sub�
5basemodel/batch_normalization_2/AssignMovingAvg_1/mulMul9basemodel/batch_normalization_2/AssignMovingAvg_1/sub:z:0@basemodel/batch_normalization_2/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:@27
5basemodel/batch_normalization_2/AssignMovingAvg_1/mul�
1basemodel/batch_normalization_2/AssignMovingAvg_1AssignSubVariableOpIbasemodel_batch_normalization_2_assignmovingavg_1_readvariableop_resource9basemodel/batch_normalization_2/AssignMovingAvg_1/mul:z:0A^basemodel/batch_normalization_2/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype023
1basemodel/batch_normalization_2/AssignMovingAvg_1�
/basemodel/batch_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:21
/basemodel/batch_normalization_2/batchnorm/add/y�
-basemodel/batch_normalization_2/batchnorm/addAddV2:basemodel/batch_normalization_2/moments/Squeeze_1:output:08basemodel/batch_normalization_2/batchnorm/add/y:output:0*
T0*
_output_shapes
:@2/
-basemodel/batch_normalization_2/batchnorm/add�
/basemodel/batch_normalization_2/batchnorm/RsqrtRsqrt1basemodel/batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes
:@21
/basemodel/batch_normalization_2/batchnorm/Rsqrt�
<basemodel/batch_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOpEbasemodel_batch_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02>
<basemodel/batch_normalization_2/batchnorm/mul/ReadVariableOp�
-basemodel/batch_normalization_2/batchnorm/mulMul3basemodel/batch_normalization_2/batchnorm/Rsqrt:y:0Dbasemodel/batch_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2/
-basemodel/batch_normalization_2/batchnorm/mul�
/basemodel/batch_normalization_2/batchnorm/mul_1Mul*basemodel/stream_2_conv_1/BiasAdd:output:01basemodel/batch_normalization_2/batchnorm/mul:z:0*
T0*,
_output_shapes
:����������@21
/basemodel/batch_normalization_2/batchnorm/mul_1�
/basemodel/batch_normalization_2/batchnorm/mul_2Mul8basemodel/batch_normalization_2/moments/Squeeze:output:01basemodel/batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes
:@21
/basemodel/batch_normalization_2/batchnorm/mul_2�
8basemodel/batch_normalization_2/batchnorm/ReadVariableOpReadVariableOpAbasemodel_batch_normalization_2_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02:
8basemodel/batch_normalization_2/batchnorm/ReadVariableOp�
-basemodel/batch_normalization_2/batchnorm/subSub@basemodel/batch_normalization_2/batchnorm/ReadVariableOp:value:03basemodel/batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2/
-basemodel/batch_normalization_2/batchnorm/sub�
/basemodel/batch_normalization_2/batchnorm/add_1AddV23basemodel/batch_normalization_2/batchnorm/mul_1:z:01basemodel/batch_normalization_2/batchnorm/sub:z:0*
T0*,
_output_shapes
:����������@21
/basemodel/batch_normalization_2/batchnorm/add_1�
>basemodel/batch_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2@
>basemodel/batch_normalization_1/moments/mean/reduction_indices�
,basemodel/batch_normalization_1/moments/meanMean*basemodel/stream_1_conv_1/BiasAdd:output:0Gbasemodel/batch_normalization_1/moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(2.
,basemodel/batch_normalization_1/moments/mean�
4basemodel/batch_normalization_1/moments/StopGradientStopGradient5basemodel/batch_normalization_1/moments/mean:output:0*
T0*"
_output_shapes
:@26
4basemodel/batch_normalization_1/moments/StopGradient�
9basemodel/batch_normalization_1/moments/SquaredDifferenceSquaredDifference*basemodel/stream_1_conv_1/BiasAdd:output:0=basemodel/batch_normalization_1/moments/StopGradient:output:0*
T0*,
_output_shapes
:����������@2;
9basemodel/batch_normalization_1/moments/SquaredDifference�
Bbasemodel/batch_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2D
Bbasemodel/batch_normalization_1/moments/variance/reduction_indices�
0basemodel/batch_normalization_1/moments/varianceMean=basemodel/batch_normalization_1/moments/SquaredDifference:z:0Kbasemodel/batch_normalization_1/moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(22
0basemodel/batch_normalization_1/moments/variance�
/basemodel/batch_normalization_1/moments/SqueezeSqueeze5basemodel/batch_normalization_1/moments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 21
/basemodel/batch_normalization_1/moments/Squeeze�
1basemodel/batch_normalization_1/moments/Squeeze_1Squeeze9basemodel/batch_normalization_1/moments/variance:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 23
1basemodel/batch_normalization_1/moments/Squeeze_1�
5basemodel/batch_normalization_1/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<27
5basemodel/batch_normalization_1/AssignMovingAvg/decay�
>basemodel/batch_normalization_1/AssignMovingAvg/ReadVariableOpReadVariableOpGbasemodel_batch_normalization_1_assignmovingavg_readvariableop_resource*
_output_shapes
:@*
dtype02@
>basemodel/batch_normalization_1/AssignMovingAvg/ReadVariableOp�
3basemodel/batch_normalization_1/AssignMovingAvg/subSubFbasemodel/batch_normalization_1/AssignMovingAvg/ReadVariableOp:value:08basemodel/batch_normalization_1/moments/Squeeze:output:0*
T0*
_output_shapes
:@25
3basemodel/batch_normalization_1/AssignMovingAvg/sub�
3basemodel/batch_normalization_1/AssignMovingAvg/mulMul7basemodel/batch_normalization_1/AssignMovingAvg/sub:z:0>basemodel/batch_normalization_1/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:@25
3basemodel/batch_normalization_1/AssignMovingAvg/mul�
/basemodel/batch_normalization_1/AssignMovingAvgAssignSubVariableOpGbasemodel_batch_normalization_1_assignmovingavg_readvariableop_resource7basemodel/batch_normalization_1/AssignMovingAvg/mul:z:0?^basemodel/batch_normalization_1/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype021
/basemodel/batch_normalization_1/AssignMovingAvg�
7basemodel/batch_normalization_1/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<29
7basemodel/batch_normalization_1/AssignMovingAvg_1/decay�
@basemodel/batch_normalization_1/AssignMovingAvg_1/ReadVariableOpReadVariableOpIbasemodel_batch_normalization_1_assignmovingavg_1_readvariableop_resource*
_output_shapes
:@*
dtype02B
@basemodel/batch_normalization_1/AssignMovingAvg_1/ReadVariableOp�
5basemodel/batch_normalization_1/AssignMovingAvg_1/subSubHbasemodel/batch_normalization_1/AssignMovingAvg_1/ReadVariableOp:value:0:basemodel/batch_normalization_1/moments/Squeeze_1:output:0*
T0*
_output_shapes
:@27
5basemodel/batch_normalization_1/AssignMovingAvg_1/sub�
5basemodel/batch_normalization_1/AssignMovingAvg_1/mulMul9basemodel/batch_normalization_1/AssignMovingAvg_1/sub:z:0@basemodel/batch_normalization_1/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:@27
5basemodel/batch_normalization_1/AssignMovingAvg_1/mul�
1basemodel/batch_normalization_1/AssignMovingAvg_1AssignSubVariableOpIbasemodel_batch_normalization_1_assignmovingavg_1_readvariableop_resource9basemodel/batch_normalization_1/AssignMovingAvg_1/mul:z:0A^basemodel/batch_normalization_1/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype023
1basemodel/batch_normalization_1/AssignMovingAvg_1�
/basemodel/batch_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:21
/basemodel/batch_normalization_1/batchnorm/add/y�
-basemodel/batch_normalization_1/batchnorm/addAddV2:basemodel/batch_normalization_1/moments/Squeeze_1:output:08basemodel/batch_normalization_1/batchnorm/add/y:output:0*
T0*
_output_shapes
:@2/
-basemodel/batch_normalization_1/batchnorm/add�
/basemodel/batch_normalization_1/batchnorm/RsqrtRsqrt1basemodel/batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes
:@21
/basemodel/batch_normalization_1/batchnorm/Rsqrt�
<basemodel/batch_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOpEbasemodel_batch_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02>
<basemodel/batch_normalization_1/batchnorm/mul/ReadVariableOp�
-basemodel/batch_normalization_1/batchnorm/mulMul3basemodel/batch_normalization_1/batchnorm/Rsqrt:y:0Dbasemodel/batch_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2/
-basemodel/batch_normalization_1/batchnorm/mul�
/basemodel/batch_normalization_1/batchnorm/mul_1Mul*basemodel/stream_1_conv_1/BiasAdd:output:01basemodel/batch_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:����������@21
/basemodel/batch_normalization_1/batchnorm/mul_1�
/basemodel/batch_normalization_1/batchnorm/mul_2Mul8basemodel/batch_normalization_1/moments/Squeeze:output:01basemodel/batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes
:@21
/basemodel/batch_normalization_1/batchnorm/mul_2�
8basemodel/batch_normalization_1/batchnorm/ReadVariableOpReadVariableOpAbasemodel_batch_normalization_1_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02:
8basemodel/batch_normalization_1/batchnorm/ReadVariableOp�
-basemodel/batch_normalization_1/batchnorm/subSub@basemodel/batch_normalization_1/batchnorm/ReadVariableOp:value:03basemodel/batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2/
-basemodel/batch_normalization_1/batchnorm/sub�
/basemodel/batch_normalization_1/batchnorm/add_1AddV23basemodel/batch_normalization_1/batchnorm/mul_1:z:01basemodel/batch_normalization_1/batchnorm/sub:z:0*
T0*,
_output_shapes
:����������@21
/basemodel/batch_normalization_1/batchnorm/add_1�
<basemodel/batch_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2>
<basemodel/batch_normalization/moments/mean/reduction_indices�
*basemodel/batch_normalization/moments/meanMean*basemodel/stream_0_conv_1/BiasAdd:output:0Ebasemodel/batch_normalization/moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(2,
*basemodel/batch_normalization/moments/mean�
2basemodel/batch_normalization/moments/StopGradientStopGradient3basemodel/batch_normalization/moments/mean:output:0*
T0*"
_output_shapes
:@24
2basemodel/batch_normalization/moments/StopGradient�
7basemodel/batch_normalization/moments/SquaredDifferenceSquaredDifference*basemodel/stream_0_conv_1/BiasAdd:output:0;basemodel/batch_normalization/moments/StopGradient:output:0*
T0*,
_output_shapes
:����������@29
7basemodel/batch_normalization/moments/SquaredDifference�
@basemodel/batch_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2B
@basemodel/batch_normalization/moments/variance/reduction_indices�
.basemodel/batch_normalization/moments/varianceMean;basemodel/batch_normalization/moments/SquaredDifference:z:0Ibasemodel/batch_normalization/moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(20
.basemodel/batch_normalization/moments/variance�
-basemodel/batch_normalization/moments/SqueezeSqueeze3basemodel/batch_normalization/moments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 2/
-basemodel/batch_normalization/moments/Squeeze�
/basemodel/batch_normalization/moments/Squeeze_1Squeeze7basemodel/batch_normalization/moments/variance:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 21
/basemodel/batch_normalization/moments/Squeeze_1�
3basemodel/batch_normalization/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<25
3basemodel/batch_normalization/AssignMovingAvg/decay�
<basemodel/batch_normalization/AssignMovingAvg/ReadVariableOpReadVariableOpEbasemodel_batch_normalization_assignmovingavg_readvariableop_resource*
_output_shapes
:@*
dtype02>
<basemodel/batch_normalization/AssignMovingAvg/ReadVariableOp�
1basemodel/batch_normalization/AssignMovingAvg/subSubDbasemodel/batch_normalization/AssignMovingAvg/ReadVariableOp:value:06basemodel/batch_normalization/moments/Squeeze:output:0*
T0*
_output_shapes
:@23
1basemodel/batch_normalization/AssignMovingAvg/sub�
1basemodel/batch_normalization/AssignMovingAvg/mulMul5basemodel/batch_normalization/AssignMovingAvg/sub:z:0<basemodel/batch_normalization/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:@23
1basemodel/batch_normalization/AssignMovingAvg/mul�
-basemodel/batch_normalization/AssignMovingAvgAssignSubVariableOpEbasemodel_batch_normalization_assignmovingavg_readvariableop_resource5basemodel/batch_normalization/AssignMovingAvg/mul:z:0=^basemodel/batch_normalization/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02/
-basemodel/batch_normalization/AssignMovingAvg�
5basemodel/batch_normalization/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<27
5basemodel/batch_normalization/AssignMovingAvg_1/decay�
>basemodel/batch_normalization/AssignMovingAvg_1/ReadVariableOpReadVariableOpGbasemodel_batch_normalization_assignmovingavg_1_readvariableop_resource*
_output_shapes
:@*
dtype02@
>basemodel/batch_normalization/AssignMovingAvg_1/ReadVariableOp�
3basemodel/batch_normalization/AssignMovingAvg_1/subSubFbasemodel/batch_normalization/AssignMovingAvg_1/ReadVariableOp:value:08basemodel/batch_normalization/moments/Squeeze_1:output:0*
T0*
_output_shapes
:@25
3basemodel/batch_normalization/AssignMovingAvg_1/sub�
3basemodel/batch_normalization/AssignMovingAvg_1/mulMul7basemodel/batch_normalization/AssignMovingAvg_1/sub:z:0>basemodel/batch_normalization/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:@25
3basemodel/batch_normalization/AssignMovingAvg_1/mul�
/basemodel/batch_normalization/AssignMovingAvg_1AssignSubVariableOpGbasemodel_batch_normalization_assignmovingavg_1_readvariableop_resource7basemodel/batch_normalization/AssignMovingAvg_1/mul:z:0?^basemodel/batch_normalization/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype021
/basemodel/batch_normalization/AssignMovingAvg_1�
-basemodel/batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2/
-basemodel/batch_normalization/batchnorm/add/y�
+basemodel/batch_normalization/batchnorm/addAddV28basemodel/batch_normalization/moments/Squeeze_1:output:06basemodel/batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes
:@2-
+basemodel/batch_normalization/batchnorm/add�
-basemodel/batch_normalization/batchnorm/RsqrtRsqrt/basemodel/batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes
:@2/
-basemodel/batch_normalization/batchnorm/Rsqrt�
:basemodel/batch_normalization/batchnorm/mul/ReadVariableOpReadVariableOpCbasemodel_batch_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02<
:basemodel/batch_normalization/batchnorm/mul/ReadVariableOp�
+basemodel/batch_normalization/batchnorm/mulMul1basemodel/batch_normalization/batchnorm/Rsqrt:y:0Bbasemodel/batch_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2-
+basemodel/batch_normalization/batchnorm/mul�
-basemodel/batch_normalization/batchnorm/mul_1Mul*basemodel/stream_0_conv_1/BiasAdd:output:0/basemodel/batch_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:����������@2/
-basemodel/batch_normalization/batchnorm/mul_1�
-basemodel/batch_normalization/batchnorm/mul_2Mul6basemodel/batch_normalization/moments/Squeeze:output:0/basemodel/batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes
:@2/
-basemodel/batch_normalization/batchnorm/mul_2�
6basemodel/batch_normalization/batchnorm/ReadVariableOpReadVariableOp?basemodel_batch_normalization_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype028
6basemodel/batch_normalization/batchnorm/ReadVariableOp�
+basemodel/batch_normalization/batchnorm/subSub>basemodel/batch_normalization/batchnorm/ReadVariableOp:value:01basemodel/batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2-
+basemodel/batch_normalization/batchnorm/sub�
-basemodel/batch_normalization/batchnorm/add_1AddV21basemodel/batch_normalization/batchnorm/mul_1:z:0/basemodel/batch_normalization/batchnorm/sub:z:0*
T0*,
_output_shapes
:����������@2/
-basemodel/batch_normalization/batchnorm/add_1�
basemodel/activation_2/ReluRelu3basemodel/batch_normalization_2/batchnorm/add_1:z:0*
T0*,
_output_shapes
:����������@2
basemodel/activation_2/Relu�
basemodel/activation_1/ReluRelu3basemodel/batch_normalization_1/batchnorm/add_1:z:0*
T0*,
_output_shapes
:����������@2
basemodel/activation_1/Relu�
basemodel/activation/ReluRelu1basemodel/batch_normalization/batchnorm/add_1:z:0*
T0*,
_output_shapes
:����������@2
basemodel/activation/Relu�
'basemodel/stream_2_drop_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2)
'basemodel/stream_2_drop_1/dropout/Const�
%basemodel/stream_2_drop_1/dropout/MulMul)basemodel/activation_2/Relu:activations:00basemodel/stream_2_drop_1/dropout/Const:output:0*
T0*,
_output_shapes
:����������@2'
%basemodel/stream_2_drop_1/dropout/Mul�
'basemodel/stream_2_drop_1/dropout/ShapeShape)basemodel/activation_2/Relu:activations:0*
T0*
_output_shapes
:2)
'basemodel/stream_2_drop_1/dropout/Shape�
>basemodel/stream_2_drop_1/dropout/random_uniform/RandomUniformRandomUniform0basemodel/stream_2_drop_1/dropout/Shape:output:0*
T0*,
_output_shapes
:����������@*
dtype0*
seed�*
seed2�2@
>basemodel/stream_2_drop_1/dropout/random_uniform/RandomUniform�
0basemodel/stream_2_drop_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>22
0basemodel/stream_2_drop_1/dropout/GreaterEqual/y�
.basemodel/stream_2_drop_1/dropout/GreaterEqualGreaterEqualGbasemodel/stream_2_drop_1/dropout/random_uniform/RandomUniform:output:09basemodel/stream_2_drop_1/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:����������@20
.basemodel/stream_2_drop_1/dropout/GreaterEqual�
&basemodel/stream_2_drop_1/dropout/CastCast2basemodel/stream_2_drop_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:����������@2(
&basemodel/stream_2_drop_1/dropout/Cast�
'basemodel/stream_2_drop_1/dropout/Mul_1Mul)basemodel/stream_2_drop_1/dropout/Mul:z:0*basemodel/stream_2_drop_1/dropout/Cast:y:0*
T0*,
_output_shapes
:����������@2)
'basemodel/stream_2_drop_1/dropout/Mul_1�
'basemodel/stream_1_drop_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2)
'basemodel/stream_1_drop_1/dropout/Const�
%basemodel/stream_1_drop_1/dropout/MulMul)basemodel/activation_1/Relu:activations:00basemodel/stream_1_drop_1/dropout/Const:output:0*
T0*,
_output_shapes
:����������@2'
%basemodel/stream_1_drop_1/dropout/Mul�
'basemodel/stream_1_drop_1/dropout/ShapeShape)basemodel/activation_1/Relu:activations:0*
T0*
_output_shapes
:2)
'basemodel/stream_1_drop_1/dropout/Shape�
>basemodel/stream_1_drop_1/dropout/random_uniform/RandomUniformRandomUniform0basemodel/stream_1_drop_1/dropout/Shape:output:0*
T0*,
_output_shapes
:����������@*
dtype0*
seed�*
seed2�2@
>basemodel/stream_1_drop_1/dropout/random_uniform/RandomUniform�
0basemodel/stream_1_drop_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>22
0basemodel/stream_1_drop_1/dropout/GreaterEqual/y�
.basemodel/stream_1_drop_1/dropout/GreaterEqualGreaterEqualGbasemodel/stream_1_drop_1/dropout/random_uniform/RandomUniform:output:09basemodel/stream_1_drop_1/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:����������@20
.basemodel/stream_1_drop_1/dropout/GreaterEqual�
&basemodel/stream_1_drop_1/dropout/CastCast2basemodel/stream_1_drop_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:����������@2(
&basemodel/stream_1_drop_1/dropout/Cast�
'basemodel/stream_1_drop_1/dropout/Mul_1Mul)basemodel/stream_1_drop_1/dropout/Mul:z:0*basemodel/stream_1_drop_1/dropout/Cast:y:0*
T0*,
_output_shapes
:����������@2)
'basemodel/stream_1_drop_1/dropout/Mul_1�
'basemodel/stream_0_drop_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2)
'basemodel/stream_0_drop_1/dropout/Const�
%basemodel/stream_0_drop_1/dropout/MulMul'basemodel/activation/Relu:activations:00basemodel/stream_0_drop_1/dropout/Const:output:0*
T0*,
_output_shapes
:����������@2'
%basemodel/stream_0_drop_1/dropout/Mul�
'basemodel/stream_0_drop_1/dropout/ShapeShape'basemodel/activation/Relu:activations:0*
T0*
_output_shapes
:2)
'basemodel/stream_0_drop_1/dropout/Shape�
>basemodel/stream_0_drop_1/dropout/random_uniform/RandomUniformRandomUniform0basemodel/stream_0_drop_1/dropout/Shape:output:0*
T0*,
_output_shapes
:����������@*
dtype0*
seed�*
seed2�2@
>basemodel/stream_0_drop_1/dropout/random_uniform/RandomUniform�
0basemodel/stream_0_drop_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>22
0basemodel/stream_0_drop_1/dropout/GreaterEqual/y�
.basemodel/stream_0_drop_1/dropout/GreaterEqualGreaterEqualGbasemodel/stream_0_drop_1/dropout/random_uniform/RandomUniform:output:09basemodel/stream_0_drop_1/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:����������@20
.basemodel/stream_0_drop_1/dropout/GreaterEqual�
&basemodel/stream_0_drop_1/dropout/CastCast2basemodel/stream_0_drop_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:����������@2(
&basemodel/stream_0_drop_1/dropout/Cast�
'basemodel/stream_0_drop_1/dropout/Mul_1Mul)basemodel/stream_0_drop_1/dropout/Mul:z:0*basemodel/stream_0_drop_1/dropout/Cast:y:0*
T0*,
_output_shapes
:����������@2)
'basemodel/stream_0_drop_1/dropout/Mul_1�
9basemodel/global_average_pooling1d/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2;
9basemodel/global_average_pooling1d/Mean/reduction_indices�
'basemodel/global_average_pooling1d/MeanMean+basemodel/stream_0_drop_1/dropout/Mul_1:z:0Bbasemodel/global_average_pooling1d/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:���������@2)
'basemodel/global_average_pooling1d/Mean�
;basemodel/global_average_pooling1d_1/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2=
;basemodel/global_average_pooling1d_1/Mean/reduction_indices�
)basemodel/global_average_pooling1d_1/MeanMean+basemodel/stream_1_drop_1/dropout/Mul_1:z:0Dbasemodel/global_average_pooling1d_1/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:���������@2+
)basemodel/global_average_pooling1d_1/Mean�
;basemodel/global_average_pooling1d_2/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2=
;basemodel/global_average_pooling1d_2/Mean/reduction_indices�
)basemodel/global_average_pooling1d_2/MeanMean+basemodel/stream_2_drop_1/dropout/Mul_1:z:0Dbasemodel/global_average_pooling1d_2/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:���������@2+
)basemodel/global_average_pooling1d_2/Mean�
!basemodel/concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2#
!basemodel/concatenate/concat/axis�
basemodel/concatenate/concatConcatV20basemodel/global_average_pooling1d/Mean:output:02basemodel/global_average_pooling1d_1/Mean:output:02basemodel/global_average_pooling1d_2/Mean:output:0*basemodel/concatenate/concat/axis:output:0*
N*
T0*(
_output_shapes
:����������2
basemodel/concatenate/concat�
'basemodel/dense_1/MatMul/ReadVariableOpReadVariableOp0basemodel_dense_1_matmul_readvariableop_resource*
_output_shapes
:	�T*
dtype02)
'basemodel/dense_1/MatMul/ReadVariableOp�
basemodel/dense_1/MatMulMatMul%basemodel/concatenate/concat:output:0/basemodel/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������T2
basemodel/dense_1/MatMul�
(basemodel/dense_1/BiasAdd/ReadVariableOpReadVariableOp1basemodel_dense_1_biasadd_readvariableop_resource*
_output_shapes
:T*
dtype02*
(basemodel/dense_1/BiasAdd/ReadVariableOp�
basemodel/dense_1/BiasAddBiasAdd"basemodel/dense_1/MatMul:product:00basemodel/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������T2
basemodel/dense_1/BiasAdd�
>basemodel/batch_normalization_3/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2@
>basemodel/batch_normalization_3/moments/mean/reduction_indices�
,basemodel/batch_normalization_3/moments/meanMean"basemodel/dense_1/BiasAdd:output:0Gbasemodel/batch_normalization_3/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:T*
	keep_dims(2.
,basemodel/batch_normalization_3/moments/mean�
4basemodel/batch_normalization_3/moments/StopGradientStopGradient5basemodel/batch_normalization_3/moments/mean:output:0*
T0*
_output_shapes

:T26
4basemodel/batch_normalization_3/moments/StopGradient�
9basemodel/batch_normalization_3/moments/SquaredDifferenceSquaredDifference"basemodel/dense_1/BiasAdd:output:0=basemodel/batch_normalization_3/moments/StopGradient:output:0*
T0*'
_output_shapes
:���������T2;
9basemodel/batch_normalization_3/moments/SquaredDifference�
Bbasemodel/batch_normalization_3/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2D
Bbasemodel/batch_normalization_3/moments/variance/reduction_indices�
0basemodel/batch_normalization_3/moments/varianceMean=basemodel/batch_normalization_3/moments/SquaredDifference:z:0Kbasemodel/batch_normalization_3/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:T*
	keep_dims(22
0basemodel/batch_normalization_3/moments/variance�
/basemodel/batch_normalization_3/moments/SqueezeSqueeze5basemodel/batch_normalization_3/moments/mean:output:0*
T0*
_output_shapes
:T*
squeeze_dims
 21
/basemodel/batch_normalization_3/moments/Squeeze�
1basemodel/batch_normalization_3/moments/Squeeze_1Squeeze9basemodel/batch_normalization_3/moments/variance:output:0*
T0*
_output_shapes
:T*
squeeze_dims
 23
1basemodel/batch_normalization_3/moments/Squeeze_1�
5basemodel/batch_normalization_3/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<27
5basemodel/batch_normalization_3/AssignMovingAvg/decay�
>basemodel/batch_normalization_3/AssignMovingAvg/ReadVariableOpReadVariableOpGbasemodel_batch_normalization_3_assignmovingavg_readvariableop_resource*
_output_shapes
:T*
dtype02@
>basemodel/batch_normalization_3/AssignMovingAvg/ReadVariableOp�
3basemodel/batch_normalization_3/AssignMovingAvg/subSubFbasemodel/batch_normalization_3/AssignMovingAvg/ReadVariableOp:value:08basemodel/batch_normalization_3/moments/Squeeze:output:0*
T0*
_output_shapes
:T25
3basemodel/batch_normalization_3/AssignMovingAvg/sub�
3basemodel/batch_normalization_3/AssignMovingAvg/mulMul7basemodel/batch_normalization_3/AssignMovingAvg/sub:z:0>basemodel/batch_normalization_3/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:T25
3basemodel/batch_normalization_3/AssignMovingAvg/mul�
/basemodel/batch_normalization_3/AssignMovingAvgAssignSubVariableOpGbasemodel_batch_normalization_3_assignmovingavg_readvariableop_resource7basemodel/batch_normalization_3/AssignMovingAvg/mul:z:0?^basemodel/batch_normalization_3/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype021
/basemodel/batch_normalization_3/AssignMovingAvg�
7basemodel/batch_normalization_3/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<29
7basemodel/batch_normalization_3/AssignMovingAvg_1/decay�
@basemodel/batch_normalization_3/AssignMovingAvg_1/ReadVariableOpReadVariableOpIbasemodel_batch_normalization_3_assignmovingavg_1_readvariableop_resource*
_output_shapes
:T*
dtype02B
@basemodel/batch_normalization_3/AssignMovingAvg_1/ReadVariableOp�
5basemodel/batch_normalization_3/AssignMovingAvg_1/subSubHbasemodel/batch_normalization_3/AssignMovingAvg_1/ReadVariableOp:value:0:basemodel/batch_normalization_3/moments/Squeeze_1:output:0*
T0*
_output_shapes
:T27
5basemodel/batch_normalization_3/AssignMovingAvg_1/sub�
5basemodel/batch_normalization_3/AssignMovingAvg_1/mulMul9basemodel/batch_normalization_3/AssignMovingAvg_1/sub:z:0@basemodel/batch_normalization_3/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:T27
5basemodel/batch_normalization_3/AssignMovingAvg_1/mul�
1basemodel/batch_normalization_3/AssignMovingAvg_1AssignSubVariableOpIbasemodel_batch_normalization_3_assignmovingavg_1_readvariableop_resource9basemodel/batch_normalization_3/AssignMovingAvg_1/mul:z:0A^basemodel/batch_normalization_3/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype023
1basemodel/batch_normalization_3/AssignMovingAvg_1�
/basemodel/batch_normalization_3/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:21
/basemodel/batch_normalization_3/batchnorm/add/y�
-basemodel/batch_normalization_3/batchnorm/addAddV2:basemodel/batch_normalization_3/moments/Squeeze_1:output:08basemodel/batch_normalization_3/batchnorm/add/y:output:0*
T0*
_output_shapes
:T2/
-basemodel/batch_normalization_3/batchnorm/add�
/basemodel/batch_normalization_3/batchnorm/RsqrtRsqrt1basemodel/batch_normalization_3/batchnorm/add:z:0*
T0*
_output_shapes
:T21
/basemodel/batch_normalization_3/batchnorm/Rsqrt�
<basemodel/batch_normalization_3/batchnorm/mul/ReadVariableOpReadVariableOpEbasemodel_batch_normalization_3_batchnorm_mul_readvariableop_resource*
_output_shapes
:T*
dtype02>
<basemodel/batch_normalization_3/batchnorm/mul/ReadVariableOp�
-basemodel/batch_normalization_3/batchnorm/mulMul3basemodel/batch_normalization_3/batchnorm/Rsqrt:y:0Dbasemodel/batch_normalization_3/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:T2/
-basemodel/batch_normalization_3/batchnorm/mul�
/basemodel/batch_normalization_3/batchnorm/mul_1Mul"basemodel/dense_1/BiasAdd:output:01basemodel/batch_normalization_3/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������T21
/basemodel/batch_normalization_3/batchnorm/mul_1�
/basemodel/batch_normalization_3/batchnorm/mul_2Mul8basemodel/batch_normalization_3/moments/Squeeze:output:01basemodel/batch_normalization_3/batchnorm/mul:z:0*
T0*
_output_shapes
:T21
/basemodel/batch_normalization_3/batchnorm/mul_2�
8basemodel/batch_normalization_3/batchnorm/ReadVariableOpReadVariableOpAbasemodel_batch_normalization_3_batchnorm_readvariableop_resource*
_output_shapes
:T*
dtype02:
8basemodel/batch_normalization_3/batchnorm/ReadVariableOp�
-basemodel/batch_normalization_3/batchnorm/subSub@basemodel/batch_normalization_3/batchnorm/ReadVariableOp:value:03basemodel/batch_normalization_3/batchnorm/mul_2:z:0*
T0*
_output_shapes
:T2/
-basemodel/batch_normalization_3/batchnorm/sub�
/basemodel/batch_normalization_3/batchnorm/add_1AddV23basemodel/batch_normalization_3/batchnorm/mul_1:z:01basemodel/batch_normalization_3/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������T21
/basemodel/batch_normalization_3/batchnorm/add_1�
$basemodel/dense_activation_1/SigmoidSigmoid3basemodel/batch_normalization_3/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������T2&
$basemodel/dense_activation_1/Sigmoid�
-basemodel/stream_2_input_drop/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2/
-basemodel/stream_2_input_drop/dropout_1/Const�
+basemodel/stream_2_input_drop/dropout_1/MulMulinputs_16basemodel/stream_2_input_drop/dropout_1/Const:output:0*
T0*,
_output_shapes
:����������2-
+basemodel/stream_2_input_drop/dropout_1/Mul�
-basemodel/stream_2_input_drop/dropout_1/ShapeShapeinputs_1*
T0*
_output_shapes
:2/
-basemodel/stream_2_input_drop/dropout_1/Shape�
Dbasemodel/stream_2_input_drop/dropout_1/random_uniform/RandomUniformRandomUniform6basemodel/stream_2_input_drop/dropout_1/Shape:output:0*
T0*,
_output_shapes
:����������*
dtype0*
seed�*
seed2�2F
Dbasemodel/stream_2_input_drop/dropout_1/random_uniform/RandomUniform�
6basemodel/stream_2_input_drop/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>28
6basemodel/stream_2_input_drop/dropout_1/GreaterEqual/y�
4basemodel/stream_2_input_drop/dropout_1/GreaterEqualGreaterEqualMbasemodel/stream_2_input_drop/dropout_1/random_uniform/RandomUniform:output:0?basemodel/stream_2_input_drop/dropout_1/GreaterEqual/y:output:0*
T0*,
_output_shapes
:����������26
4basemodel/stream_2_input_drop/dropout_1/GreaterEqual�
,basemodel/stream_2_input_drop/dropout_1/CastCast8basemodel/stream_2_input_drop/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:����������2.
,basemodel/stream_2_input_drop/dropout_1/Cast�
-basemodel/stream_2_input_drop/dropout_1/Mul_1Mul/basemodel/stream_2_input_drop/dropout_1/Mul:z:00basemodel/stream_2_input_drop/dropout_1/Cast:y:0*
T0*,
_output_shapes
:����������2/
-basemodel/stream_2_input_drop/dropout_1/Mul_1�
-basemodel/stream_1_input_drop/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2/
-basemodel/stream_1_input_drop/dropout_1/Const�
+basemodel/stream_1_input_drop/dropout_1/MulMulinputs_16basemodel/stream_1_input_drop/dropout_1/Const:output:0*
T0*,
_output_shapes
:����������2-
+basemodel/stream_1_input_drop/dropout_1/Mul�
-basemodel/stream_1_input_drop/dropout_1/ShapeShapeinputs_1*
T0*
_output_shapes
:2/
-basemodel/stream_1_input_drop/dropout_1/Shape�
Dbasemodel/stream_1_input_drop/dropout_1/random_uniform/RandomUniformRandomUniform6basemodel/stream_1_input_drop/dropout_1/Shape:output:0*
T0*,
_output_shapes
:����������*
dtype0*
seed�*
seed2�2F
Dbasemodel/stream_1_input_drop/dropout_1/random_uniform/RandomUniform�
6basemodel/stream_1_input_drop/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>28
6basemodel/stream_1_input_drop/dropout_1/GreaterEqual/y�
4basemodel/stream_1_input_drop/dropout_1/GreaterEqualGreaterEqualMbasemodel/stream_1_input_drop/dropout_1/random_uniform/RandomUniform:output:0?basemodel/stream_1_input_drop/dropout_1/GreaterEqual/y:output:0*
T0*,
_output_shapes
:����������26
4basemodel/stream_1_input_drop/dropout_1/GreaterEqual�
,basemodel/stream_1_input_drop/dropout_1/CastCast8basemodel/stream_1_input_drop/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:����������2.
,basemodel/stream_1_input_drop/dropout_1/Cast�
-basemodel/stream_1_input_drop/dropout_1/Mul_1Mul/basemodel/stream_1_input_drop/dropout_1/Mul:z:00basemodel/stream_1_input_drop/dropout_1/Cast:y:0*
T0*,
_output_shapes
:����������2/
-basemodel/stream_1_input_drop/dropout_1/Mul_1�
-basemodel/stream_0_input_drop/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2/
-basemodel/stream_0_input_drop/dropout_1/Const�
+basemodel/stream_0_input_drop/dropout_1/MulMulinputs_16basemodel/stream_0_input_drop/dropout_1/Const:output:0*
T0*,
_output_shapes
:����������2-
+basemodel/stream_0_input_drop/dropout_1/Mul�
-basemodel/stream_0_input_drop/dropout_1/ShapeShapeinputs_1*
T0*
_output_shapes
:2/
-basemodel/stream_0_input_drop/dropout_1/Shape�
Dbasemodel/stream_0_input_drop/dropout_1/random_uniform/RandomUniformRandomUniform6basemodel/stream_0_input_drop/dropout_1/Shape:output:0*
T0*,
_output_shapes
:����������*
dtype0*
seed�*
seed2�2F
Dbasemodel/stream_0_input_drop/dropout_1/random_uniform/RandomUniform�
6basemodel/stream_0_input_drop/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>28
6basemodel/stream_0_input_drop/dropout_1/GreaterEqual/y�
4basemodel/stream_0_input_drop/dropout_1/GreaterEqualGreaterEqualMbasemodel/stream_0_input_drop/dropout_1/random_uniform/RandomUniform:output:0?basemodel/stream_0_input_drop/dropout_1/GreaterEqual/y:output:0*
T0*,
_output_shapes
:����������26
4basemodel/stream_0_input_drop/dropout_1/GreaterEqual�
,basemodel/stream_0_input_drop/dropout_1/CastCast8basemodel/stream_0_input_drop/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:����������2.
,basemodel/stream_0_input_drop/dropout_1/Cast�
-basemodel/stream_0_input_drop/dropout_1/Mul_1Mul/basemodel/stream_0_input_drop/dropout_1/Mul:z:00basemodel/stream_0_input_drop/dropout_1/Cast:y:0*
T0*,
_output_shapes
:����������2/
-basemodel/stream_0_input_drop/dropout_1/Mul_1�
1basemodel/stream_2_conv_1/conv1d_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������23
1basemodel/stream_2_conv_1/conv1d_1/ExpandDims/dim�
-basemodel/stream_2_conv_1/conv1d_1/ExpandDims
ExpandDims1basemodel/stream_2_input_drop/dropout_1/Mul_1:z:0:basemodel/stream_2_conv_1/conv1d_1/ExpandDims/dim:output:0*
T0*0
_output_shapes
:����������2/
-basemodel/stream_2_conv_1/conv1d_1/ExpandDims�
>basemodel/stream_2_conv_1/conv1d_1/ExpandDims_1/ReadVariableOpReadVariableOpEbasemodel_stream_2_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02@
>basemodel/stream_2_conv_1/conv1d_1/ExpandDims_1/ReadVariableOp�
3basemodel/stream_2_conv_1/conv1d_1/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 25
3basemodel/stream_2_conv_1/conv1d_1/ExpandDims_1/dim�
/basemodel/stream_2_conv_1/conv1d_1/ExpandDims_1
ExpandDimsFbasemodel/stream_2_conv_1/conv1d_1/ExpandDims_1/ReadVariableOp:value:0<basemodel/stream_2_conv_1/conv1d_1/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@21
/basemodel/stream_2_conv_1/conv1d_1/ExpandDims_1�
"basemodel/stream_2_conv_1/conv1d_1Conv2D6basemodel/stream_2_conv_1/conv1d_1/ExpandDims:output:08basemodel/stream_2_conv_1/conv1d_1/ExpandDims_1:output:0*
T0*0
_output_shapes
:����������@*
paddingSAME*
strides
2$
"basemodel/stream_2_conv_1/conv1d_1�
*basemodel/stream_2_conv_1/conv1d_1/SqueezeSqueeze+basemodel/stream_2_conv_1/conv1d_1:output:0*
T0*,
_output_shapes
:����������@*
squeeze_dims

���������2,
*basemodel/stream_2_conv_1/conv1d_1/Squeeze�
2basemodel/stream_2_conv_1/BiasAdd_1/ReadVariableOpReadVariableOp9basemodel_stream_2_conv_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype024
2basemodel/stream_2_conv_1/BiasAdd_1/ReadVariableOp�
#basemodel/stream_2_conv_1/BiasAdd_1BiasAdd3basemodel/stream_2_conv_1/conv1d_1/Squeeze:output:0:basemodel/stream_2_conv_1/BiasAdd_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������@2%
#basemodel/stream_2_conv_1/BiasAdd_1�
1basemodel/stream_1_conv_1/conv1d_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������23
1basemodel/stream_1_conv_1/conv1d_1/ExpandDims/dim�
-basemodel/stream_1_conv_1/conv1d_1/ExpandDims
ExpandDims1basemodel/stream_1_input_drop/dropout_1/Mul_1:z:0:basemodel/stream_1_conv_1/conv1d_1/ExpandDims/dim:output:0*
T0*0
_output_shapes
:����������2/
-basemodel/stream_1_conv_1/conv1d_1/ExpandDims�
>basemodel/stream_1_conv_1/conv1d_1/ExpandDims_1/ReadVariableOpReadVariableOpEbasemodel_stream_1_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02@
>basemodel/stream_1_conv_1/conv1d_1/ExpandDims_1/ReadVariableOp�
3basemodel/stream_1_conv_1/conv1d_1/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 25
3basemodel/stream_1_conv_1/conv1d_1/ExpandDims_1/dim�
/basemodel/stream_1_conv_1/conv1d_1/ExpandDims_1
ExpandDimsFbasemodel/stream_1_conv_1/conv1d_1/ExpandDims_1/ReadVariableOp:value:0<basemodel/stream_1_conv_1/conv1d_1/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@21
/basemodel/stream_1_conv_1/conv1d_1/ExpandDims_1�
"basemodel/stream_1_conv_1/conv1d_1Conv2D6basemodel/stream_1_conv_1/conv1d_1/ExpandDims:output:08basemodel/stream_1_conv_1/conv1d_1/ExpandDims_1:output:0*
T0*0
_output_shapes
:����������@*
paddingSAME*
strides
2$
"basemodel/stream_1_conv_1/conv1d_1�
*basemodel/stream_1_conv_1/conv1d_1/SqueezeSqueeze+basemodel/stream_1_conv_1/conv1d_1:output:0*
T0*,
_output_shapes
:����������@*
squeeze_dims

���������2,
*basemodel/stream_1_conv_1/conv1d_1/Squeeze�
2basemodel/stream_1_conv_1/BiasAdd_1/ReadVariableOpReadVariableOp9basemodel_stream_1_conv_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype024
2basemodel/stream_1_conv_1/BiasAdd_1/ReadVariableOp�
#basemodel/stream_1_conv_1/BiasAdd_1BiasAdd3basemodel/stream_1_conv_1/conv1d_1/Squeeze:output:0:basemodel/stream_1_conv_1/BiasAdd_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������@2%
#basemodel/stream_1_conv_1/BiasAdd_1�
1basemodel/stream_0_conv_1/conv1d_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������23
1basemodel/stream_0_conv_1/conv1d_1/ExpandDims/dim�
-basemodel/stream_0_conv_1/conv1d_1/ExpandDims
ExpandDims1basemodel/stream_0_input_drop/dropout_1/Mul_1:z:0:basemodel/stream_0_conv_1/conv1d_1/ExpandDims/dim:output:0*
T0*0
_output_shapes
:����������2/
-basemodel/stream_0_conv_1/conv1d_1/ExpandDims�
>basemodel/stream_0_conv_1/conv1d_1/ExpandDims_1/ReadVariableOpReadVariableOpEbasemodel_stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02@
>basemodel/stream_0_conv_1/conv1d_1/ExpandDims_1/ReadVariableOp�
3basemodel/stream_0_conv_1/conv1d_1/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 25
3basemodel/stream_0_conv_1/conv1d_1/ExpandDims_1/dim�
/basemodel/stream_0_conv_1/conv1d_1/ExpandDims_1
ExpandDimsFbasemodel/stream_0_conv_1/conv1d_1/ExpandDims_1/ReadVariableOp:value:0<basemodel/stream_0_conv_1/conv1d_1/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@21
/basemodel/stream_0_conv_1/conv1d_1/ExpandDims_1�
"basemodel/stream_0_conv_1/conv1d_1Conv2D6basemodel/stream_0_conv_1/conv1d_1/ExpandDims:output:08basemodel/stream_0_conv_1/conv1d_1/ExpandDims_1:output:0*
T0*0
_output_shapes
:����������@*
paddingSAME*
strides
2$
"basemodel/stream_0_conv_1/conv1d_1�
*basemodel/stream_0_conv_1/conv1d_1/SqueezeSqueeze+basemodel/stream_0_conv_1/conv1d_1:output:0*
T0*,
_output_shapes
:����������@*
squeeze_dims

���������2,
*basemodel/stream_0_conv_1/conv1d_1/Squeeze�
2basemodel/stream_0_conv_1/BiasAdd_1/ReadVariableOpReadVariableOp9basemodel_stream_0_conv_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype024
2basemodel/stream_0_conv_1/BiasAdd_1/ReadVariableOp�
#basemodel/stream_0_conv_1/BiasAdd_1BiasAdd3basemodel/stream_0_conv_1/conv1d_1/Squeeze:output:0:basemodel/stream_0_conv_1/BiasAdd_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������@2%
#basemodel/stream_0_conv_1/BiasAdd_1�
@basemodel/batch_normalization_2/moments_1/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2B
@basemodel/batch_normalization_2/moments_1/mean/reduction_indices�
.basemodel/batch_normalization_2/moments_1/meanMean,basemodel/stream_2_conv_1/BiasAdd_1:output:0Ibasemodel/batch_normalization_2/moments_1/mean/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(20
.basemodel/batch_normalization_2/moments_1/mean�
6basemodel/batch_normalization_2/moments_1/StopGradientStopGradient7basemodel/batch_normalization_2/moments_1/mean:output:0*
T0*"
_output_shapes
:@28
6basemodel/batch_normalization_2/moments_1/StopGradient�
;basemodel/batch_normalization_2/moments_1/SquaredDifferenceSquaredDifference,basemodel/stream_2_conv_1/BiasAdd_1:output:0?basemodel/batch_normalization_2/moments_1/StopGradient:output:0*
T0*,
_output_shapes
:����������@2=
;basemodel/batch_normalization_2/moments_1/SquaredDifference�
Dbasemodel/batch_normalization_2/moments_1/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2F
Dbasemodel/batch_normalization_2/moments_1/variance/reduction_indices�
2basemodel/batch_normalization_2/moments_1/varianceMean?basemodel/batch_normalization_2/moments_1/SquaredDifference:z:0Mbasemodel/batch_normalization_2/moments_1/variance/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(24
2basemodel/batch_normalization_2/moments_1/variance�
1basemodel/batch_normalization_2/moments_1/SqueezeSqueeze7basemodel/batch_normalization_2/moments_1/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 23
1basemodel/batch_normalization_2/moments_1/Squeeze�
3basemodel/batch_normalization_2/moments_1/Squeeze_1Squeeze;basemodel/batch_normalization_2/moments_1/variance:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 25
3basemodel/batch_normalization_2/moments_1/Squeeze_1�
7basemodel/batch_normalization_2/AssignMovingAvg_2/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<29
7basemodel/batch_normalization_2/AssignMovingAvg_2/decay�
@basemodel/batch_normalization_2/AssignMovingAvg_2/ReadVariableOpReadVariableOpGbasemodel_batch_normalization_2_assignmovingavg_readvariableop_resource0^basemodel/batch_normalization_2/AssignMovingAvg*
_output_shapes
:@*
dtype02B
@basemodel/batch_normalization_2/AssignMovingAvg_2/ReadVariableOp�
5basemodel/batch_normalization_2/AssignMovingAvg_2/subSubHbasemodel/batch_normalization_2/AssignMovingAvg_2/ReadVariableOp:value:0:basemodel/batch_normalization_2/moments_1/Squeeze:output:0*
T0*
_output_shapes
:@27
5basemodel/batch_normalization_2/AssignMovingAvg_2/sub�
5basemodel/batch_normalization_2/AssignMovingAvg_2/mulMul9basemodel/batch_normalization_2/AssignMovingAvg_2/sub:z:0@basemodel/batch_normalization_2/AssignMovingAvg_2/decay:output:0*
T0*
_output_shapes
:@27
5basemodel/batch_normalization_2/AssignMovingAvg_2/mul�
1basemodel/batch_normalization_2/AssignMovingAvg_2AssignSubVariableOpGbasemodel_batch_normalization_2_assignmovingavg_readvariableop_resource9basemodel/batch_normalization_2/AssignMovingAvg_2/mul:z:00^basemodel/batch_normalization_2/AssignMovingAvgA^basemodel/batch_normalization_2/AssignMovingAvg_2/ReadVariableOp*
_output_shapes
 *
dtype023
1basemodel/batch_normalization_2/AssignMovingAvg_2�
7basemodel/batch_normalization_2/AssignMovingAvg_3/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<29
7basemodel/batch_normalization_2/AssignMovingAvg_3/decay�
@basemodel/batch_normalization_2/AssignMovingAvg_3/ReadVariableOpReadVariableOpIbasemodel_batch_normalization_2_assignmovingavg_1_readvariableop_resource2^basemodel/batch_normalization_2/AssignMovingAvg_1*
_output_shapes
:@*
dtype02B
@basemodel/batch_normalization_2/AssignMovingAvg_3/ReadVariableOp�
5basemodel/batch_normalization_2/AssignMovingAvg_3/subSubHbasemodel/batch_normalization_2/AssignMovingAvg_3/ReadVariableOp:value:0<basemodel/batch_normalization_2/moments_1/Squeeze_1:output:0*
T0*
_output_shapes
:@27
5basemodel/batch_normalization_2/AssignMovingAvg_3/sub�
5basemodel/batch_normalization_2/AssignMovingAvg_3/mulMul9basemodel/batch_normalization_2/AssignMovingAvg_3/sub:z:0@basemodel/batch_normalization_2/AssignMovingAvg_3/decay:output:0*
T0*
_output_shapes
:@27
5basemodel/batch_normalization_2/AssignMovingAvg_3/mul�
1basemodel/batch_normalization_2/AssignMovingAvg_3AssignSubVariableOpIbasemodel_batch_normalization_2_assignmovingavg_1_readvariableop_resource9basemodel/batch_normalization_2/AssignMovingAvg_3/mul:z:02^basemodel/batch_normalization_2/AssignMovingAvg_1A^basemodel/batch_normalization_2/AssignMovingAvg_3/ReadVariableOp*
_output_shapes
 *
dtype023
1basemodel/batch_normalization_2/AssignMovingAvg_3�
1basemodel/batch_normalization_2/batchnorm_1/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:23
1basemodel/batch_normalization_2/batchnorm_1/add/y�
/basemodel/batch_normalization_2/batchnorm_1/addAddV2<basemodel/batch_normalization_2/moments_1/Squeeze_1:output:0:basemodel/batch_normalization_2/batchnorm_1/add/y:output:0*
T0*
_output_shapes
:@21
/basemodel/batch_normalization_2/batchnorm_1/add�
1basemodel/batch_normalization_2/batchnorm_1/RsqrtRsqrt3basemodel/batch_normalization_2/batchnorm_1/add:z:0*
T0*
_output_shapes
:@23
1basemodel/batch_normalization_2/batchnorm_1/Rsqrt�
>basemodel/batch_normalization_2/batchnorm_1/mul/ReadVariableOpReadVariableOpEbasemodel_batch_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02@
>basemodel/batch_normalization_2/batchnorm_1/mul/ReadVariableOp�
/basemodel/batch_normalization_2/batchnorm_1/mulMul5basemodel/batch_normalization_2/batchnorm_1/Rsqrt:y:0Fbasemodel/batch_normalization_2/batchnorm_1/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@21
/basemodel/batch_normalization_2/batchnorm_1/mul�
1basemodel/batch_normalization_2/batchnorm_1/mul_1Mul,basemodel/stream_2_conv_1/BiasAdd_1:output:03basemodel/batch_normalization_2/batchnorm_1/mul:z:0*
T0*,
_output_shapes
:����������@23
1basemodel/batch_normalization_2/batchnorm_1/mul_1�
1basemodel/batch_normalization_2/batchnorm_1/mul_2Mul:basemodel/batch_normalization_2/moments_1/Squeeze:output:03basemodel/batch_normalization_2/batchnorm_1/mul:z:0*
T0*
_output_shapes
:@23
1basemodel/batch_normalization_2/batchnorm_1/mul_2�
:basemodel/batch_normalization_2/batchnorm_1/ReadVariableOpReadVariableOpAbasemodel_batch_normalization_2_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02<
:basemodel/batch_normalization_2/batchnorm_1/ReadVariableOp�
/basemodel/batch_normalization_2/batchnorm_1/subSubBbasemodel/batch_normalization_2/batchnorm_1/ReadVariableOp:value:05basemodel/batch_normalization_2/batchnorm_1/mul_2:z:0*
T0*
_output_shapes
:@21
/basemodel/batch_normalization_2/batchnorm_1/sub�
1basemodel/batch_normalization_2/batchnorm_1/add_1AddV25basemodel/batch_normalization_2/batchnorm_1/mul_1:z:03basemodel/batch_normalization_2/batchnorm_1/sub:z:0*
T0*,
_output_shapes
:����������@23
1basemodel/batch_normalization_2/batchnorm_1/add_1�
@basemodel/batch_normalization_1/moments_1/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2B
@basemodel/batch_normalization_1/moments_1/mean/reduction_indices�
.basemodel/batch_normalization_1/moments_1/meanMean,basemodel/stream_1_conv_1/BiasAdd_1:output:0Ibasemodel/batch_normalization_1/moments_1/mean/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(20
.basemodel/batch_normalization_1/moments_1/mean�
6basemodel/batch_normalization_1/moments_1/StopGradientStopGradient7basemodel/batch_normalization_1/moments_1/mean:output:0*
T0*"
_output_shapes
:@28
6basemodel/batch_normalization_1/moments_1/StopGradient�
;basemodel/batch_normalization_1/moments_1/SquaredDifferenceSquaredDifference,basemodel/stream_1_conv_1/BiasAdd_1:output:0?basemodel/batch_normalization_1/moments_1/StopGradient:output:0*
T0*,
_output_shapes
:����������@2=
;basemodel/batch_normalization_1/moments_1/SquaredDifference�
Dbasemodel/batch_normalization_1/moments_1/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2F
Dbasemodel/batch_normalization_1/moments_1/variance/reduction_indices�
2basemodel/batch_normalization_1/moments_1/varianceMean?basemodel/batch_normalization_1/moments_1/SquaredDifference:z:0Mbasemodel/batch_normalization_1/moments_1/variance/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(24
2basemodel/batch_normalization_1/moments_1/variance�
1basemodel/batch_normalization_1/moments_1/SqueezeSqueeze7basemodel/batch_normalization_1/moments_1/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 23
1basemodel/batch_normalization_1/moments_1/Squeeze�
3basemodel/batch_normalization_1/moments_1/Squeeze_1Squeeze;basemodel/batch_normalization_1/moments_1/variance:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 25
3basemodel/batch_normalization_1/moments_1/Squeeze_1�
7basemodel/batch_normalization_1/AssignMovingAvg_2/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<29
7basemodel/batch_normalization_1/AssignMovingAvg_2/decay�
@basemodel/batch_normalization_1/AssignMovingAvg_2/ReadVariableOpReadVariableOpGbasemodel_batch_normalization_1_assignmovingavg_readvariableop_resource0^basemodel/batch_normalization_1/AssignMovingAvg*
_output_shapes
:@*
dtype02B
@basemodel/batch_normalization_1/AssignMovingAvg_2/ReadVariableOp�
5basemodel/batch_normalization_1/AssignMovingAvg_2/subSubHbasemodel/batch_normalization_1/AssignMovingAvg_2/ReadVariableOp:value:0:basemodel/batch_normalization_1/moments_1/Squeeze:output:0*
T0*
_output_shapes
:@27
5basemodel/batch_normalization_1/AssignMovingAvg_2/sub�
5basemodel/batch_normalization_1/AssignMovingAvg_2/mulMul9basemodel/batch_normalization_1/AssignMovingAvg_2/sub:z:0@basemodel/batch_normalization_1/AssignMovingAvg_2/decay:output:0*
T0*
_output_shapes
:@27
5basemodel/batch_normalization_1/AssignMovingAvg_2/mul�
1basemodel/batch_normalization_1/AssignMovingAvg_2AssignSubVariableOpGbasemodel_batch_normalization_1_assignmovingavg_readvariableop_resource9basemodel/batch_normalization_1/AssignMovingAvg_2/mul:z:00^basemodel/batch_normalization_1/AssignMovingAvgA^basemodel/batch_normalization_1/AssignMovingAvg_2/ReadVariableOp*
_output_shapes
 *
dtype023
1basemodel/batch_normalization_1/AssignMovingAvg_2�
7basemodel/batch_normalization_1/AssignMovingAvg_3/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<29
7basemodel/batch_normalization_1/AssignMovingAvg_3/decay�
@basemodel/batch_normalization_1/AssignMovingAvg_3/ReadVariableOpReadVariableOpIbasemodel_batch_normalization_1_assignmovingavg_1_readvariableop_resource2^basemodel/batch_normalization_1/AssignMovingAvg_1*
_output_shapes
:@*
dtype02B
@basemodel/batch_normalization_1/AssignMovingAvg_3/ReadVariableOp�
5basemodel/batch_normalization_1/AssignMovingAvg_3/subSubHbasemodel/batch_normalization_1/AssignMovingAvg_3/ReadVariableOp:value:0<basemodel/batch_normalization_1/moments_1/Squeeze_1:output:0*
T0*
_output_shapes
:@27
5basemodel/batch_normalization_1/AssignMovingAvg_3/sub�
5basemodel/batch_normalization_1/AssignMovingAvg_3/mulMul9basemodel/batch_normalization_1/AssignMovingAvg_3/sub:z:0@basemodel/batch_normalization_1/AssignMovingAvg_3/decay:output:0*
T0*
_output_shapes
:@27
5basemodel/batch_normalization_1/AssignMovingAvg_3/mul�
1basemodel/batch_normalization_1/AssignMovingAvg_3AssignSubVariableOpIbasemodel_batch_normalization_1_assignmovingavg_1_readvariableop_resource9basemodel/batch_normalization_1/AssignMovingAvg_3/mul:z:02^basemodel/batch_normalization_1/AssignMovingAvg_1A^basemodel/batch_normalization_1/AssignMovingAvg_3/ReadVariableOp*
_output_shapes
 *
dtype023
1basemodel/batch_normalization_1/AssignMovingAvg_3�
1basemodel/batch_normalization_1/batchnorm_1/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:23
1basemodel/batch_normalization_1/batchnorm_1/add/y�
/basemodel/batch_normalization_1/batchnorm_1/addAddV2<basemodel/batch_normalization_1/moments_1/Squeeze_1:output:0:basemodel/batch_normalization_1/batchnorm_1/add/y:output:0*
T0*
_output_shapes
:@21
/basemodel/batch_normalization_1/batchnorm_1/add�
1basemodel/batch_normalization_1/batchnorm_1/RsqrtRsqrt3basemodel/batch_normalization_1/batchnorm_1/add:z:0*
T0*
_output_shapes
:@23
1basemodel/batch_normalization_1/batchnorm_1/Rsqrt�
>basemodel/batch_normalization_1/batchnorm_1/mul/ReadVariableOpReadVariableOpEbasemodel_batch_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02@
>basemodel/batch_normalization_1/batchnorm_1/mul/ReadVariableOp�
/basemodel/batch_normalization_1/batchnorm_1/mulMul5basemodel/batch_normalization_1/batchnorm_1/Rsqrt:y:0Fbasemodel/batch_normalization_1/batchnorm_1/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@21
/basemodel/batch_normalization_1/batchnorm_1/mul�
1basemodel/batch_normalization_1/batchnorm_1/mul_1Mul,basemodel/stream_1_conv_1/BiasAdd_1:output:03basemodel/batch_normalization_1/batchnorm_1/mul:z:0*
T0*,
_output_shapes
:����������@23
1basemodel/batch_normalization_1/batchnorm_1/mul_1�
1basemodel/batch_normalization_1/batchnorm_1/mul_2Mul:basemodel/batch_normalization_1/moments_1/Squeeze:output:03basemodel/batch_normalization_1/batchnorm_1/mul:z:0*
T0*
_output_shapes
:@23
1basemodel/batch_normalization_1/batchnorm_1/mul_2�
:basemodel/batch_normalization_1/batchnorm_1/ReadVariableOpReadVariableOpAbasemodel_batch_normalization_1_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02<
:basemodel/batch_normalization_1/batchnorm_1/ReadVariableOp�
/basemodel/batch_normalization_1/batchnorm_1/subSubBbasemodel/batch_normalization_1/batchnorm_1/ReadVariableOp:value:05basemodel/batch_normalization_1/batchnorm_1/mul_2:z:0*
T0*
_output_shapes
:@21
/basemodel/batch_normalization_1/batchnorm_1/sub�
1basemodel/batch_normalization_1/batchnorm_1/add_1AddV25basemodel/batch_normalization_1/batchnorm_1/mul_1:z:03basemodel/batch_normalization_1/batchnorm_1/sub:z:0*
T0*,
_output_shapes
:����������@23
1basemodel/batch_normalization_1/batchnorm_1/add_1�
>basemodel/batch_normalization/moments_1/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2@
>basemodel/batch_normalization/moments_1/mean/reduction_indices�
,basemodel/batch_normalization/moments_1/meanMean,basemodel/stream_0_conv_1/BiasAdd_1:output:0Gbasemodel/batch_normalization/moments_1/mean/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(2.
,basemodel/batch_normalization/moments_1/mean�
4basemodel/batch_normalization/moments_1/StopGradientStopGradient5basemodel/batch_normalization/moments_1/mean:output:0*
T0*"
_output_shapes
:@26
4basemodel/batch_normalization/moments_1/StopGradient�
9basemodel/batch_normalization/moments_1/SquaredDifferenceSquaredDifference,basemodel/stream_0_conv_1/BiasAdd_1:output:0=basemodel/batch_normalization/moments_1/StopGradient:output:0*
T0*,
_output_shapes
:����������@2;
9basemodel/batch_normalization/moments_1/SquaredDifference�
Bbasemodel/batch_normalization/moments_1/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2D
Bbasemodel/batch_normalization/moments_1/variance/reduction_indices�
0basemodel/batch_normalization/moments_1/varianceMean=basemodel/batch_normalization/moments_1/SquaredDifference:z:0Kbasemodel/batch_normalization/moments_1/variance/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(22
0basemodel/batch_normalization/moments_1/variance�
/basemodel/batch_normalization/moments_1/SqueezeSqueeze5basemodel/batch_normalization/moments_1/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 21
/basemodel/batch_normalization/moments_1/Squeeze�
1basemodel/batch_normalization/moments_1/Squeeze_1Squeeze9basemodel/batch_normalization/moments_1/variance:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 23
1basemodel/batch_normalization/moments_1/Squeeze_1�
5basemodel/batch_normalization/AssignMovingAvg_2/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<27
5basemodel/batch_normalization/AssignMovingAvg_2/decay�
>basemodel/batch_normalization/AssignMovingAvg_2/ReadVariableOpReadVariableOpEbasemodel_batch_normalization_assignmovingavg_readvariableop_resource.^basemodel/batch_normalization/AssignMovingAvg*
_output_shapes
:@*
dtype02@
>basemodel/batch_normalization/AssignMovingAvg_2/ReadVariableOp�
3basemodel/batch_normalization/AssignMovingAvg_2/subSubFbasemodel/batch_normalization/AssignMovingAvg_2/ReadVariableOp:value:08basemodel/batch_normalization/moments_1/Squeeze:output:0*
T0*
_output_shapes
:@25
3basemodel/batch_normalization/AssignMovingAvg_2/sub�
3basemodel/batch_normalization/AssignMovingAvg_2/mulMul7basemodel/batch_normalization/AssignMovingAvg_2/sub:z:0>basemodel/batch_normalization/AssignMovingAvg_2/decay:output:0*
T0*
_output_shapes
:@25
3basemodel/batch_normalization/AssignMovingAvg_2/mul�
/basemodel/batch_normalization/AssignMovingAvg_2AssignSubVariableOpEbasemodel_batch_normalization_assignmovingavg_readvariableop_resource7basemodel/batch_normalization/AssignMovingAvg_2/mul:z:0.^basemodel/batch_normalization/AssignMovingAvg?^basemodel/batch_normalization/AssignMovingAvg_2/ReadVariableOp*
_output_shapes
 *
dtype021
/basemodel/batch_normalization/AssignMovingAvg_2�
5basemodel/batch_normalization/AssignMovingAvg_3/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<27
5basemodel/batch_normalization/AssignMovingAvg_3/decay�
>basemodel/batch_normalization/AssignMovingAvg_3/ReadVariableOpReadVariableOpGbasemodel_batch_normalization_assignmovingavg_1_readvariableop_resource0^basemodel/batch_normalization/AssignMovingAvg_1*
_output_shapes
:@*
dtype02@
>basemodel/batch_normalization/AssignMovingAvg_3/ReadVariableOp�
3basemodel/batch_normalization/AssignMovingAvg_3/subSubFbasemodel/batch_normalization/AssignMovingAvg_3/ReadVariableOp:value:0:basemodel/batch_normalization/moments_1/Squeeze_1:output:0*
T0*
_output_shapes
:@25
3basemodel/batch_normalization/AssignMovingAvg_3/sub�
3basemodel/batch_normalization/AssignMovingAvg_3/mulMul7basemodel/batch_normalization/AssignMovingAvg_3/sub:z:0>basemodel/batch_normalization/AssignMovingAvg_3/decay:output:0*
T0*
_output_shapes
:@25
3basemodel/batch_normalization/AssignMovingAvg_3/mul�
/basemodel/batch_normalization/AssignMovingAvg_3AssignSubVariableOpGbasemodel_batch_normalization_assignmovingavg_1_readvariableop_resource7basemodel/batch_normalization/AssignMovingAvg_3/mul:z:00^basemodel/batch_normalization/AssignMovingAvg_1?^basemodel/batch_normalization/AssignMovingAvg_3/ReadVariableOp*
_output_shapes
 *
dtype021
/basemodel/batch_normalization/AssignMovingAvg_3�
/basemodel/batch_normalization/batchnorm_1/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:21
/basemodel/batch_normalization/batchnorm_1/add/y�
-basemodel/batch_normalization/batchnorm_1/addAddV2:basemodel/batch_normalization/moments_1/Squeeze_1:output:08basemodel/batch_normalization/batchnorm_1/add/y:output:0*
T0*
_output_shapes
:@2/
-basemodel/batch_normalization/batchnorm_1/add�
/basemodel/batch_normalization/batchnorm_1/RsqrtRsqrt1basemodel/batch_normalization/batchnorm_1/add:z:0*
T0*
_output_shapes
:@21
/basemodel/batch_normalization/batchnorm_1/Rsqrt�
<basemodel/batch_normalization/batchnorm_1/mul/ReadVariableOpReadVariableOpCbasemodel_batch_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02>
<basemodel/batch_normalization/batchnorm_1/mul/ReadVariableOp�
-basemodel/batch_normalization/batchnorm_1/mulMul3basemodel/batch_normalization/batchnorm_1/Rsqrt:y:0Dbasemodel/batch_normalization/batchnorm_1/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2/
-basemodel/batch_normalization/batchnorm_1/mul�
/basemodel/batch_normalization/batchnorm_1/mul_1Mul,basemodel/stream_0_conv_1/BiasAdd_1:output:01basemodel/batch_normalization/batchnorm_1/mul:z:0*
T0*,
_output_shapes
:����������@21
/basemodel/batch_normalization/batchnorm_1/mul_1�
/basemodel/batch_normalization/batchnorm_1/mul_2Mul8basemodel/batch_normalization/moments_1/Squeeze:output:01basemodel/batch_normalization/batchnorm_1/mul:z:0*
T0*
_output_shapes
:@21
/basemodel/batch_normalization/batchnorm_1/mul_2�
8basemodel/batch_normalization/batchnorm_1/ReadVariableOpReadVariableOp?basemodel_batch_normalization_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02:
8basemodel/batch_normalization/batchnorm_1/ReadVariableOp�
-basemodel/batch_normalization/batchnorm_1/subSub@basemodel/batch_normalization/batchnorm_1/ReadVariableOp:value:03basemodel/batch_normalization/batchnorm_1/mul_2:z:0*
T0*
_output_shapes
:@2/
-basemodel/batch_normalization/batchnorm_1/sub�
/basemodel/batch_normalization/batchnorm_1/add_1AddV23basemodel/batch_normalization/batchnorm_1/mul_1:z:01basemodel/batch_normalization/batchnorm_1/sub:z:0*
T0*,
_output_shapes
:����������@21
/basemodel/batch_normalization/batchnorm_1/add_1�
basemodel/activation_2/Relu_1Relu5basemodel/batch_normalization_2/batchnorm_1/add_1:z:0*
T0*,
_output_shapes
:����������@2
basemodel/activation_2/Relu_1�
basemodel/activation_1/Relu_1Relu5basemodel/batch_normalization_1/batchnorm_1/add_1:z:0*
T0*,
_output_shapes
:����������@2
basemodel/activation_1/Relu_1�
basemodel/activation/Relu_1Relu3basemodel/batch_normalization/batchnorm_1/add_1:z:0*
T0*,
_output_shapes
:����������@2
basemodel/activation/Relu_1�
)basemodel/stream_2_drop_1/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2+
)basemodel/stream_2_drop_1/dropout_1/Const�
'basemodel/stream_2_drop_1/dropout_1/MulMul+basemodel/activation_2/Relu_1:activations:02basemodel/stream_2_drop_1/dropout_1/Const:output:0*
T0*,
_output_shapes
:����������@2)
'basemodel/stream_2_drop_1/dropout_1/Mul�
)basemodel/stream_2_drop_1/dropout_1/ShapeShape+basemodel/activation_2/Relu_1:activations:0*
T0*
_output_shapes
:2+
)basemodel/stream_2_drop_1/dropout_1/Shape�
@basemodel/stream_2_drop_1/dropout_1/random_uniform/RandomUniformRandomUniform2basemodel/stream_2_drop_1/dropout_1/Shape:output:0*
T0*,
_output_shapes
:����������@*
dtype0*
seed�*
seed2�2B
@basemodel/stream_2_drop_1/dropout_1/random_uniform/RandomUniform�
2basemodel/stream_2_drop_1/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>24
2basemodel/stream_2_drop_1/dropout_1/GreaterEqual/y�
0basemodel/stream_2_drop_1/dropout_1/GreaterEqualGreaterEqualIbasemodel/stream_2_drop_1/dropout_1/random_uniform/RandomUniform:output:0;basemodel/stream_2_drop_1/dropout_1/GreaterEqual/y:output:0*
T0*,
_output_shapes
:����������@22
0basemodel/stream_2_drop_1/dropout_1/GreaterEqual�
(basemodel/stream_2_drop_1/dropout_1/CastCast4basemodel/stream_2_drop_1/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:����������@2*
(basemodel/stream_2_drop_1/dropout_1/Cast�
)basemodel/stream_2_drop_1/dropout_1/Mul_1Mul+basemodel/stream_2_drop_1/dropout_1/Mul:z:0,basemodel/stream_2_drop_1/dropout_1/Cast:y:0*
T0*,
_output_shapes
:����������@2+
)basemodel/stream_2_drop_1/dropout_1/Mul_1�
)basemodel/stream_1_drop_1/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2+
)basemodel/stream_1_drop_1/dropout_1/Const�
'basemodel/stream_1_drop_1/dropout_1/MulMul+basemodel/activation_1/Relu_1:activations:02basemodel/stream_1_drop_1/dropout_1/Const:output:0*
T0*,
_output_shapes
:����������@2)
'basemodel/stream_1_drop_1/dropout_1/Mul�
)basemodel/stream_1_drop_1/dropout_1/ShapeShape+basemodel/activation_1/Relu_1:activations:0*
T0*
_output_shapes
:2+
)basemodel/stream_1_drop_1/dropout_1/Shape�
@basemodel/stream_1_drop_1/dropout_1/random_uniform/RandomUniformRandomUniform2basemodel/stream_1_drop_1/dropout_1/Shape:output:0*
T0*,
_output_shapes
:����������@*
dtype0*
seed�*
seed2�2B
@basemodel/stream_1_drop_1/dropout_1/random_uniform/RandomUniform�
2basemodel/stream_1_drop_1/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>24
2basemodel/stream_1_drop_1/dropout_1/GreaterEqual/y�
0basemodel/stream_1_drop_1/dropout_1/GreaterEqualGreaterEqualIbasemodel/stream_1_drop_1/dropout_1/random_uniform/RandomUniform:output:0;basemodel/stream_1_drop_1/dropout_1/GreaterEqual/y:output:0*
T0*,
_output_shapes
:����������@22
0basemodel/stream_1_drop_1/dropout_1/GreaterEqual�
(basemodel/stream_1_drop_1/dropout_1/CastCast4basemodel/stream_1_drop_1/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:����������@2*
(basemodel/stream_1_drop_1/dropout_1/Cast�
)basemodel/stream_1_drop_1/dropout_1/Mul_1Mul+basemodel/stream_1_drop_1/dropout_1/Mul:z:0,basemodel/stream_1_drop_1/dropout_1/Cast:y:0*
T0*,
_output_shapes
:����������@2+
)basemodel/stream_1_drop_1/dropout_1/Mul_1�
)basemodel/stream_0_drop_1/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2+
)basemodel/stream_0_drop_1/dropout_1/Const�
'basemodel/stream_0_drop_1/dropout_1/MulMul)basemodel/activation/Relu_1:activations:02basemodel/stream_0_drop_1/dropout_1/Const:output:0*
T0*,
_output_shapes
:����������@2)
'basemodel/stream_0_drop_1/dropout_1/Mul�
)basemodel/stream_0_drop_1/dropout_1/ShapeShape)basemodel/activation/Relu_1:activations:0*
T0*
_output_shapes
:2+
)basemodel/stream_0_drop_1/dropout_1/Shape�
@basemodel/stream_0_drop_1/dropout_1/random_uniform/RandomUniformRandomUniform2basemodel/stream_0_drop_1/dropout_1/Shape:output:0*
T0*,
_output_shapes
:����������@*
dtype0*
seed�*
seed2�2B
@basemodel/stream_0_drop_1/dropout_1/random_uniform/RandomUniform�
2basemodel/stream_0_drop_1/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>24
2basemodel/stream_0_drop_1/dropout_1/GreaterEqual/y�
0basemodel/stream_0_drop_1/dropout_1/GreaterEqualGreaterEqualIbasemodel/stream_0_drop_1/dropout_1/random_uniform/RandomUniform:output:0;basemodel/stream_0_drop_1/dropout_1/GreaterEqual/y:output:0*
T0*,
_output_shapes
:����������@22
0basemodel/stream_0_drop_1/dropout_1/GreaterEqual�
(basemodel/stream_0_drop_1/dropout_1/CastCast4basemodel/stream_0_drop_1/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:����������@2*
(basemodel/stream_0_drop_1/dropout_1/Cast�
)basemodel/stream_0_drop_1/dropout_1/Mul_1Mul+basemodel/stream_0_drop_1/dropout_1/Mul:z:0,basemodel/stream_0_drop_1/dropout_1/Cast:y:0*
T0*,
_output_shapes
:����������@2+
)basemodel/stream_0_drop_1/dropout_1/Mul_1�
;basemodel/global_average_pooling1d/Mean_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2=
;basemodel/global_average_pooling1d/Mean_1/reduction_indices�
)basemodel/global_average_pooling1d/Mean_1Mean-basemodel/stream_0_drop_1/dropout_1/Mul_1:z:0Dbasemodel/global_average_pooling1d/Mean_1/reduction_indices:output:0*
T0*'
_output_shapes
:���������@2+
)basemodel/global_average_pooling1d/Mean_1�
=basemodel/global_average_pooling1d_1/Mean_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2?
=basemodel/global_average_pooling1d_1/Mean_1/reduction_indices�
+basemodel/global_average_pooling1d_1/Mean_1Mean-basemodel/stream_1_drop_1/dropout_1/Mul_1:z:0Fbasemodel/global_average_pooling1d_1/Mean_1/reduction_indices:output:0*
T0*'
_output_shapes
:���������@2-
+basemodel/global_average_pooling1d_1/Mean_1�
=basemodel/global_average_pooling1d_2/Mean_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2?
=basemodel/global_average_pooling1d_2/Mean_1/reduction_indices�
+basemodel/global_average_pooling1d_2/Mean_1Mean-basemodel/stream_2_drop_1/dropout_1/Mul_1:z:0Fbasemodel/global_average_pooling1d_2/Mean_1/reduction_indices:output:0*
T0*'
_output_shapes
:���������@2-
+basemodel/global_average_pooling1d_2/Mean_1�
#basemodel/concatenate/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :2%
#basemodel/concatenate/concat_1/axis�
basemodel/concatenate/concat_1ConcatV22basemodel/global_average_pooling1d/Mean_1:output:04basemodel/global_average_pooling1d_1/Mean_1:output:04basemodel/global_average_pooling1d_2/Mean_1:output:0,basemodel/concatenate/concat_1/axis:output:0*
N*
T0*(
_output_shapes
:����������2 
basemodel/concatenate/concat_1�
)basemodel/dense_1/MatMul_1/ReadVariableOpReadVariableOp0basemodel_dense_1_matmul_readvariableop_resource*
_output_shapes
:	�T*
dtype02+
)basemodel/dense_1/MatMul_1/ReadVariableOp�
basemodel/dense_1/MatMul_1MatMul'basemodel/concatenate/concat_1:output:01basemodel/dense_1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������T2
basemodel/dense_1/MatMul_1�
*basemodel/dense_1/BiasAdd_1/ReadVariableOpReadVariableOp1basemodel_dense_1_biasadd_readvariableop_resource*
_output_shapes
:T*
dtype02,
*basemodel/dense_1/BiasAdd_1/ReadVariableOp�
basemodel/dense_1/BiasAdd_1BiasAdd$basemodel/dense_1/MatMul_1:product:02basemodel/dense_1/BiasAdd_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������T2
basemodel/dense_1/BiasAdd_1�
@basemodel/batch_normalization_3/moments_1/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2B
@basemodel/batch_normalization_3/moments_1/mean/reduction_indices�
.basemodel/batch_normalization_3/moments_1/meanMean$basemodel/dense_1/BiasAdd_1:output:0Ibasemodel/batch_normalization_3/moments_1/mean/reduction_indices:output:0*
T0*
_output_shapes

:T*
	keep_dims(20
.basemodel/batch_normalization_3/moments_1/mean�
6basemodel/batch_normalization_3/moments_1/StopGradientStopGradient7basemodel/batch_normalization_3/moments_1/mean:output:0*
T0*
_output_shapes

:T28
6basemodel/batch_normalization_3/moments_1/StopGradient�
;basemodel/batch_normalization_3/moments_1/SquaredDifferenceSquaredDifference$basemodel/dense_1/BiasAdd_1:output:0?basemodel/batch_normalization_3/moments_1/StopGradient:output:0*
T0*'
_output_shapes
:���������T2=
;basemodel/batch_normalization_3/moments_1/SquaredDifference�
Dbasemodel/batch_normalization_3/moments_1/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2F
Dbasemodel/batch_normalization_3/moments_1/variance/reduction_indices�
2basemodel/batch_normalization_3/moments_1/varianceMean?basemodel/batch_normalization_3/moments_1/SquaredDifference:z:0Mbasemodel/batch_normalization_3/moments_1/variance/reduction_indices:output:0*
T0*
_output_shapes

:T*
	keep_dims(24
2basemodel/batch_normalization_3/moments_1/variance�
1basemodel/batch_normalization_3/moments_1/SqueezeSqueeze7basemodel/batch_normalization_3/moments_1/mean:output:0*
T0*
_output_shapes
:T*
squeeze_dims
 23
1basemodel/batch_normalization_3/moments_1/Squeeze�
3basemodel/batch_normalization_3/moments_1/Squeeze_1Squeeze;basemodel/batch_normalization_3/moments_1/variance:output:0*
T0*
_output_shapes
:T*
squeeze_dims
 25
3basemodel/batch_normalization_3/moments_1/Squeeze_1�
7basemodel/batch_normalization_3/AssignMovingAvg_2/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<29
7basemodel/batch_normalization_3/AssignMovingAvg_2/decay�
@basemodel/batch_normalization_3/AssignMovingAvg_2/ReadVariableOpReadVariableOpGbasemodel_batch_normalization_3_assignmovingavg_readvariableop_resource0^basemodel/batch_normalization_3/AssignMovingAvg*
_output_shapes
:T*
dtype02B
@basemodel/batch_normalization_3/AssignMovingAvg_2/ReadVariableOp�
5basemodel/batch_normalization_3/AssignMovingAvg_2/subSubHbasemodel/batch_normalization_3/AssignMovingAvg_2/ReadVariableOp:value:0:basemodel/batch_normalization_3/moments_1/Squeeze:output:0*
T0*
_output_shapes
:T27
5basemodel/batch_normalization_3/AssignMovingAvg_2/sub�
5basemodel/batch_normalization_3/AssignMovingAvg_2/mulMul9basemodel/batch_normalization_3/AssignMovingAvg_2/sub:z:0@basemodel/batch_normalization_3/AssignMovingAvg_2/decay:output:0*
T0*
_output_shapes
:T27
5basemodel/batch_normalization_3/AssignMovingAvg_2/mul�
1basemodel/batch_normalization_3/AssignMovingAvg_2AssignSubVariableOpGbasemodel_batch_normalization_3_assignmovingavg_readvariableop_resource9basemodel/batch_normalization_3/AssignMovingAvg_2/mul:z:00^basemodel/batch_normalization_3/AssignMovingAvgA^basemodel/batch_normalization_3/AssignMovingAvg_2/ReadVariableOp*
_output_shapes
 *
dtype023
1basemodel/batch_normalization_3/AssignMovingAvg_2�
7basemodel/batch_normalization_3/AssignMovingAvg_3/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<29
7basemodel/batch_normalization_3/AssignMovingAvg_3/decay�
@basemodel/batch_normalization_3/AssignMovingAvg_3/ReadVariableOpReadVariableOpIbasemodel_batch_normalization_3_assignmovingavg_1_readvariableop_resource2^basemodel/batch_normalization_3/AssignMovingAvg_1*
_output_shapes
:T*
dtype02B
@basemodel/batch_normalization_3/AssignMovingAvg_3/ReadVariableOp�
5basemodel/batch_normalization_3/AssignMovingAvg_3/subSubHbasemodel/batch_normalization_3/AssignMovingAvg_3/ReadVariableOp:value:0<basemodel/batch_normalization_3/moments_1/Squeeze_1:output:0*
T0*
_output_shapes
:T27
5basemodel/batch_normalization_3/AssignMovingAvg_3/sub�
5basemodel/batch_normalization_3/AssignMovingAvg_3/mulMul9basemodel/batch_normalization_3/AssignMovingAvg_3/sub:z:0@basemodel/batch_normalization_3/AssignMovingAvg_3/decay:output:0*
T0*
_output_shapes
:T27
5basemodel/batch_normalization_3/AssignMovingAvg_3/mul�
1basemodel/batch_normalization_3/AssignMovingAvg_3AssignSubVariableOpIbasemodel_batch_normalization_3_assignmovingavg_1_readvariableop_resource9basemodel/batch_normalization_3/AssignMovingAvg_3/mul:z:02^basemodel/batch_normalization_3/AssignMovingAvg_1A^basemodel/batch_normalization_3/AssignMovingAvg_3/ReadVariableOp*
_output_shapes
 *
dtype023
1basemodel/batch_normalization_3/AssignMovingAvg_3�
1basemodel/batch_normalization_3/batchnorm_1/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:23
1basemodel/batch_normalization_3/batchnorm_1/add/y�
/basemodel/batch_normalization_3/batchnorm_1/addAddV2<basemodel/batch_normalization_3/moments_1/Squeeze_1:output:0:basemodel/batch_normalization_3/batchnorm_1/add/y:output:0*
T0*
_output_shapes
:T21
/basemodel/batch_normalization_3/batchnorm_1/add�
1basemodel/batch_normalization_3/batchnorm_1/RsqrtRsqrt3basemodel/batch_normalization_3/batchnorm_1/add:z:0*
T0*
_output_shapes
:T23
1basemodel/batch_normalization_3/batchnorm_1/Rsqrt�
>basemodel/batch_normalization_3/batchnorm_1/mul/ReadVariableOpReadVariableOpEbasemodel_batch_normalization_3_batchnorm_mul_readvariableop_resource*
_output_shapes
:T*
dtype02@
>basemodel/batch_normalization_3/batchnorm_1/mul/ReadVariableOp�
/basemodel/batch_normalization_3/batchnorm_1/mulMul5basemodel/batch_normalization_3/batchnorm_1/Rsqrt:y:0Fbasemodel/batch_normalization_3/batchnorm_1/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:T21
/basemodel/batch_normalization_3/batchnorm_1/mul�
1basemodel/batch_normalization_3/batchnorm_1/mul_1Mul$basemodel/dense_1/BiasAdd_1:output:03basemodel/batch_normalization_3/batchnorm_1/mul:z:0*
T0*'
_output_shapes
:���������T23
1basemodel/batch_normalization_3/batchnorm_1/mul_1�
1basemodel/batch_normalization_3/batchnorm_1/mul_2Mul:basemodel/batch_normalization_3/moments_1/Squeeze:output:03basemodel/batch_normalization_3/batchnorm_1/mul:z:0*
T0*
_output_shapes
:T23
1basemodel/batch_normalization_3/batchnorm_1/mul_2�
:basemodel/batch_normalization_3/batchnorm_1/ReadVariableOpReadVariableOpAbasemodel_batch_normalization_3_batchnorm_readvariableop_resource*
_output_shapes
:T*
dtype02<
:basemodel/batch_normalization_3/batchnorm_1/ReadVariableOp�
/basemodel/batch_normalization_3/batchnorm_1/subSubBbasemodel/batch_normalization_3/batchnorm_1/ReadVariableOp:value:05basemodel/batch_normalization_3/batchnorm_1/mul_2:z:0*
T0*
_output_shapes
:T21
/basemodel/batch_normalization_3/batchnorm_1/sub�
1basemodel/batch_normalization_3/batchnorm_1/add_1AddV25basemodel/batch_normalization_3/batchnorm_1/mul_1:z:03basemodel/batch_normalization_3/batchnorm_1/sub:z:0*
T0*'
_output_shapes
:���������T23
1basemodel/batch_normalization_3/batchnorm_1/add_1�
&basemodel/dense_activation_1/Sigmoid_1Sigmoid5basemodel/batch_normalization_3/batchnorm_1/add_1:z:0*
T0*'
_output_shapes
:���������T2(
&basemodel/dense_activation_1/Sigmoid_1�
distance/subSub(basemodel/dense_activation_1/Sigmoid:y:0*basemodel/dense_activation_1/Sigmoid_1:y:0*
T0*'
_output_shapes
:���������T2
distance/subp
distance/SquareSquaredistance/sub:z:0*
T0*'
_output_shapes
:���������T2
distance/Square�
distance/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
���������2 
distance/Sum/reduction_indices�
distance/SumSumdistance/Square:y:0'distance/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:���������*
	keep_dims(2
distance/Sume
distance/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
distance/Const�
distance/MaximumMaximumdistance/Sum:output:0distance/Const:output:0*
T0*'
_output_shapes
:���������2
distance/Maximumn
distance/SqrtSqrtdistance/Maximum:z:0*
T0*'
_output_shapes
:���������2
distance/Sqrt�
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpEbasemodel_stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype027
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp�
&stream_0_conv_1/kernel/Regularizer/AbsAbs=stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_0_conv_1/kernel/Regularizer/Abs�
(stream_0_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_0_conv_1/kernel/Regularizer/Const�
&stream_0_conv_1/kernel/Regularizer/SumSum*stream_0_conv_1/kernel/Regularizer/Abs:y:01stream_0_conv_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/Sum�
(stream_0_conv_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2*
(stream_0_conv_1/kernel/Regularizer/mul/x�
&stream_0_conv_1/kernel/Regularizer/mulMul1stream_0_conv_1/kernel/Regularizer/mul/x:output:0/stream_0_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/mul�
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpEbasemodel_stream_1_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02:
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp�
)stream_1_conv_1/kernel/Regularizer/SquareSquare@stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2+
)stream_1_conv_1/kernel/Regularizer/Square�
(stream_1_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_1_conv_1/kernel/Regularizer/Const�
&stream_1_conv_1/kernel/Regularizer/SumSum-stream_1_conv_1/kernel/Regularizer/Square:y:01stream_1_conv_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&stream_1_conv_1/kernel/Regularizer/Sum�
(stream_1_conv_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2*
(stream_1_conv_1/kernel/Regularizer/mul/x�
&stream_1_conv_1/kernel/Regularizer/mulMul1stream_1_conv_1/kernel/Regularizer/mul/x:output:0/stream_1_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_1_conv_1/kernel/Regularizer/mul�
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpEbasemodel_stream_2_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype027
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp�
&stream_2_conv_1/kernel/Regularizer/AbsAbs=stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_2_conv_1/kernel/Regularizer/Abs�
(stream_2_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_2_conv_1/kernel/Regularizer/Const�
&stream_2_conv_1/kernel/Regularizer/SumSum*stream_2_conv_1/kernel/Regularizer/Abs:y:01stream_2_conv_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&stream_2_conv_1/kernel/Regularizer/Sum�
(stream_2_conv_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2*
(stream_2_conv_1/kernel/Regularizer/mul/x�
&stream_2_conv_1/kernel/Regularizer/mulMul1stream_2_conv_1/kernel/Regularizer/mul/x:output:0/stream_2_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_2_conv_1/kernel/Regularizer/mul�
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp0basemodel_dense_1_matmul_readvariableop_resource*
_output_shapes
:	�T*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOp�
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	�T2 
dense_1/kernel/Regularizer/Abs�
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_1/kernel/Regularizer/Const�
dense_1/kernel/Regularizer/SumSum"dense_1/kernel/Regularizer/Abs:y:0)dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/Sum�
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2"
 dense_1/kernel/Regularizer/mul/x�
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/mull
IdentityIdentitydistance/Sqrt:y:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp.^basemodel/batch_normalization/AssignMovingAvg=^basemodel/batch_normalization/AssignMovingAvg/ReadVariableOp0^basemodel/batch_normalization/AssignMovingAvg_1?^basemodel/batch_normalization/AssignMovingAvg_1/ReadVariableOp0^basemodel/batch_normalization/AssignMovingAvg_2?^basemodel/batch_normalization/AssignMovingAvg_2/ReadVariableOp0^basemodel/batch_normalization/AssignMovingAvg_3?^basemodel/batch_normalization/AssignMovingAvg_3/ReadVariableOp7^basemodel/batch_normalization/batchnorm/ReadVariableOp;^basemodel/batch_normalization/batchnorm/mul/ReadVariableOp9^basemodel/batch_normalization/batchnorm_1/ReadVariableOp=^basemodel/batch_normalization/batchnorm_1/mul/ReadVariableOp0^basemodel/batch_normalization_1/AssignMovingAvg?^basemodel/batch_normalization_1/AssignMovingAvg/ReadVariableOp2^basemodel/batch_normalization_1/AssignMovingAvg_1A^basemodel/batch_normalization_1/AssignMovingAvg_1/ReadVariableOp2^basemodel/batch_normalization_1/AssignMovingAvg_2A^basemodel/batch_normalization_1/AssignMovingAvg_2/ReadVariableOp2^basemodel/batch_normalization_1/AssignMovingAvg_3A^basemodel/batch_normalization_1/AssignMovingAvg_3/ReadVariableOp9^basemodel/batch_normalization_1/batchnorm/ReadVariableOp=^basemodel/batch_normalization_1/batchnorm/mul/ReadVariableOp;^basemodel/batch_normalization_1/batchnorm_1/ReadVariableOp?^basemodel/batch_normalization_1/batchnorm_1/mul/ReadVariableOp0^basemodel/batch_normalization_2/AssignMovingAvg?^basemodel/batch_normalization_2/AssignMovingAvg/ReadVariableOp2^basemodel/batch_normalization_2/AssignMovingAvg_1A^basemodel/batch_normalization_2/AssignMovingAvg_1/ReadVariableOp2^basemodel/batch_normalization_2/AssignMovingAvg_2A^basemodel/batch_normalization_2/AssignMovingAvg_2/ReadVariableOp2^basemodel/batch_normalization_2/AssignMovingAvg_3A^basemodel/batch_normalization_2/AssignMovingAvg_3/ReadVariableOp9^basemodel/batch_normalization_2/batchnorm/ReadVariableOp=^basemodel/batch_normalization_2/batchnorm/mul/ReadVariableOp;^basemodel/batch_normalization_2/batchnorm_1/ReadVariableOp?^basemodel/batch_normalization_2/batchnorm_1/mul/ReadVariableOp0^basemodel/batch_normalization_3/AssignMovingAvg?^basemodel/batch_normalization_3/AssignMovingAvg/ReadVariableOp2^basemodel/batch_normalization_3/AssignMovingAvg_1A^basemodel/batch_normalization_3/AssignMovingAvg_1/ReadVariableOp2^basemodel/batch_normalization_3/AssignMovingAvg_2A^basemodel/batch_normalization_3/AssignMovingAvg_2/ReadVariableOp2^basemodel/batch_normalization_3/AssignMovingAvg_3A^basemodel/batch_normalization_3/AssignMovingAvg_3/ReadVariableOp9^basemodel/batch_normalization_3/batchnorm/ReadVariableOp=^basemodel/batch_normalization_3/batchnorm/mul/ReadVariableOp;^basemodel/batch_normalization_3/batchnorm_1/ReadVariableOp?^basemodel/batch_normalization_3/batchnorm_1/mul/ReadVariableOp)^basemodel/dense_1/BiasAdd/ReadVariableOp+^basemodel/dense_1/BiasAdd_1/ReadVariableOp(^basemodel/dense_1/MatMul/ReadVariableOp*^basemodel/dense_1/MatMul_1/ReadVariableOp1^basemodel/stream_0_conv_1/BiasAdd/ReadVariableOp3^basemodel/stream_0_conv_1/BiasAdd_1/ReadVariableOp=^basemodel/stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp?^basemodel/stream_0_conv_1/conv1d_1/ExpandDims_1/ReadVariableOp1^basemodel/stream_1_conv_1/BiasAdd/ReadVariableOp3^basemodel/stream_1_conv_1/BiasAdd_1/ReadVariableOp=^basemodel/stream_1_conv_1/conv1d/ExpandDims_1/ReadVariableOp?^basemodel/stream_1_conv_1/conv1d_1/ExpandDims_1/ReadVariableOp1^basemodel/stream_2_conv_1/BiasAdd/ReadVariableOp3^basemodel/stream_2_conv_1/BiasAdd_1/ReadVariableOp=^basemodel/stream_2_conv_1/conv1d/ExpandDims_1/ReadVariableOp?^basemodel/stream_2_conv_1/conv1d_1/ExpandDims_1/ReadVariableOp.^dense_1/kernel/Regularizer/Abs/ReadVariableOp6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp9^stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp6^stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*s
_input_shapesb
`:����������:����������: : : : : : : : : : : : : : : : : : : : : : : : 2^
-basemodel/batch_normalization/AssignMovingAvg-basemodel/batch_normalization/AssignMovingAvg2|
<basemodel/batch_normalization/AssignMovingAvg/ReadVariableOp<basemodel/batch_normalization/AssignMovingAvg/ReadVariableOp2b
/basemodel/batch_normalization/AssignMovingAvg_1/basemodel/batch_normalization/AssignMovingAvg_12�
>basemodel/batch_normalization/AssignMovingAvg_1/ReadVariableOp>basemodel/batch_normalization/AssignMovingAvg_1/ReadVariableOp2b
/basemodel/batch_normalization/AssignMovingAvg_2/basemodel/batch_normalization/AssignMovingAvg_22�
>basemodel/batch_normalization/AssignMovingAvg_2/ReadVariableOp>basemodel/batch_normalization/AssignMovingAvg_2/ReadVariableOp2b
/basemodel/batch_normalization/AssignMovingAvg_3/basemodel/batch_normalization/AssignMovingAvg_32�
>basemodel/batch_normalization/AssignMovingAvg_3/ReadVariableOp>basemodel/batch_normalization/AssignMovingAvg_3/ReadVariableOp2p
6basemodel/batch_normalization/batchnorm/ReadVariableOp6basemodel/batch_normalization/batchnorm/ReadVariableOp2x
:basemodel/batch_normalization/batchnorm/mul/ReadVariableOp:basemodel/batch_normalization/batchnorm/mul/ReadVariableOp2t
8basemodel/batch_normalization/batchnorm_1/ReadVariableOp8basemodel/batch_normalization/batchnorm_1/ReadVariableOp2|
<basemodel/batch_normalization/batchnorm_1/mul/ReadVariableOp<basemodel/batch_normalization/batchnorm_1/mul/ReadVariableOp2b
/basemodel/batch_normalization_1/AssignMovingAvg/basemodel/batch_normalization_1/AssignMovingAvg2�
>basemodel/batch_normalization_1/AssignMovingAvg/ReadVariableOp>basemodel/batch_normalization_1/AssignMovingAvg/ReadVariableOp2f
1basemodel/batch_normalization_1/AssignMovingAvg_11basemodel/batch_normalization_1/AssignMovingAvg_12�
@basemodel/batch_normalization_1/AssignMovingAvg_1/ReadVariableOp@basemodel/batch_normalization_1/AssignMovingAvg_1/ReadVariableOp2f
1basemodel/batch_normalization_1/AssignMovingAvg_21basemodel/batch_normalization_1/AssignMovingAvg_22�
@basemodel/batch_normalization_1/AssignMovingAvg_2/ReadVariableOp@basemodel/batch_normalization_1/AssignMovingAvg_2/ReadVariableOp2f
1basemodel/batch_normalization_1/AssignMovingAvg_31basemodel/batch_normalization_1/AssignMovingAvg_32�
@basemodel/batch_normalization_1/AssignMovingAvg_3/ReadVariableOp@basemodel/batch_normalization_1/AssignMovingAvg_3/ReadVariableOp2t
8basemodel/batch_normalization_1/batchnorm/ReadVariableOp8basemodel/batch_normalization_1/batchnorm/ReadVariableOp2|
<basemodel/batch_normalization_1/batchnorm/mul/ReadVariableOp<basemodel/batch_normalization_1/batchnorm/mul/ReadVariableOp2x
:basemodel/batch_normalization_1/batchnorm_1/ReadVariableOp:basemodel/batch_normalization_1/batchnorm_1/ReadVariableOp2�
>basemodel/batch_normalization_1/batchnorm_1/mul/ReadVariableOp>basemodel/batch_normalization_1/batchnorm_1/mul/ReadVariableOp2b
/basemodel/batch_normalization_2/AssignMovingAvg/basemodel/batch_normalization_2/AssignMovingAvg2�
>basemodel/batch_normalization_2/AssignMovingAvg/ReadVariableOp>basemodel/batch_normalization_2/AssignMovingAvg/ReadVariableOp2f
1basemodel/batch_normalization_2/AssignMovingAvg_11basemodel/batch_normalization_2/AssignMovingAvg_12�
@basemodel/batch_normalization_2/AssignMovingAvg_1/ReadVariableOp@basemodel/batch_normalization_2/AssignMovingAvg_1/ReadVariableOp2f
1basemodel/batch_normalization_2/AssignMovingAvg_21basemodel/batch_normalization_2/AssignMovingAvg_22�
@basemodel/batch_normalization_2/AssignMovingAvg_2/ReadVariableOp@basemodel/batch_normalization_2/AssignMovingAvg_2/ReadVariableOp2f
1basemodel/batch_normalization_2/AssignMovingAvg_31basemodel/batch_normalization_2/AssignMovingAvg_32�
@basemodel/batch_normalization_2/AssignMovingAvg_3/ReadVariableOp@basemodel/batch_normalization_2/AssignMovingAvg_3/ReadVariableOp2t
8basemodel/batch_normalization_2/batchnorm/ReadVariableOp8basemodel/batch_normalization_2/batchnorm/ReadVariableOp2|
<basemodel/batch_normalization_2/batchnorm/mul/ReadVariableOp<basemodel/batch_normalization_2/batchnorm/mul/ReadVariableOp2x
:basemodel/batch_normalization_2/batchnorm_1/ReadVariableOp:basemodel/batch_normalization_2/batchnorm_1/ReadVariableOp2�
>basemodel/batch_normalization_2/batchnorm_1/mul/ReadVariableOp>basemodel/batch_normalization_2/batchnorm_1/mul/ReadVariableOp2b
/basemodel/batch_normalization_3/AssignMovingAvg/basemodel/batch_normalization_3/AssignMovingAvg2�
>basemodel/batch_normalization_3/AssignMovingAvg/ReadVariableOp>basemodel/batch_normalization_3/AssignMovingAvg/ReadVariableOp2f
1basemodel/batch_normalization_3/AssignMovingAvg_11basemodel/batch_normalization_3/AssignMovingAvg_12�
@basemodel/batch_normalization_3/AssignMovingAvg_1/ReadVariableOp@basemodel/batch_normalization_3/AssignMovingAvg_1/ReadVariableOp2f
1basemodel/batch_normalization_3/AssignMovingAvg_21basemodel/batch_normalization_3/AssignMovingAvg_22�
@basemodel/batch_normalization_3/AssignMovingAvg_2/ReadVariableOp@basemodel/batch_normalization_3/AssignMovingAvg_2/ReadVariableOp2f
1basemodel/batch_normalization_3/AssignMovingAvg_31basemodel/batch_normalization_3/AssignMovingAvg_32�
@basemodel/batch_normalization_3/AssignMovingAvg_3/ReadVariableOp@basemodel/batch_normalization_3/AssignMovingAvg_3/ReadVariableOp2t
8basemodel/batch_normalization_3/batchnorm/ReadVariableOp8basemodel/batch_normalization_3/batchnorm/ReadVariableOp2|
<basemodel/batch_normalization_3/batchnorm/mul/ReadVariableOp<basemodel/batch_normalization_3/batchnorm/mul/ReadVariableOp2x
:basemodel/batch_normalization_3/batchnorm_1/ReadVariableOp:basemodel/batch_normalization_3/batchnorm_1/ReadVariableOp2�
>basemodel/batch_normalization_3/batchnorm_1/mul/ReadVariableOp>basemodel/batch_normalization_3/batchnorm_1/mul/ReadVariableOp2T
(basemodel/dense_1/BiasAdd/ReadVariableOp(basemodel/dense_1/BiasAdd/ReadVariableOp2X
*basemodel/dense_1/BiasAdd_1/ReadVariableOp*basemodel/dense_1/BiasAdd_1/ReadVariableOp2R
'basemodel/dense_1/MatMul/ReadVariableOp'basemodel/dense_1/MatMul/ReadVariableOp2V
)basemodel/dense_1/MatMul_1/ReadVariableOp)basemodel/dense_1/MatMul_1/ReadVariableOp2d
0basemodel/stream_0_conv_1/BiasAdd/ReadVariableOp0basemodel/stream_0_conv_1/BiasAdd/ReadVariableOp2h
2basemodel/stream_0_conv_1/BiasAdd_1/ReadVariableOp2basemodel/stream_0_conv_1/BiasAdd_1/ReadVariableOp2|
<basemodel/stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp<basemodel/stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp2�
>basemodel/stream_0_conv_1/conv1d_1/ExpandDims_1/ReadVariableOp>basemodel/stream_0_conv_1/conv1d_1/ExpandDims_1/ReadVariableOp2d
0basemodel/stream_1_conv_1/BiasAdd/ReadVariableOp0basemodel/stream_1_conv_1/BiasAdd/ReadVariableOp2h
2basemodel/stream_1_conv_1/BiasAdd_1/ReadVariableOp2basemodel/stream_1_conv_1/BiasAdd_1/ReadVariableOp2|
<basemodel/stream_1_conv_1/conv1d/ExpandDims_1/ReadVariableOp<basemodel/stream_1_conv_1/conv1d/ExpandDims_1/ReadVariableOp2�
>basemodel/stream_1_conv_1/conv1d_1/ExpandDims_1/ReadVariableOp>basemodel/stream_1_conv_1/conv1d_1/ExpandDims_1/ReadVariableOp2d
0basemodel/stream_2_conv_1/BiasAdd/ReadVariableOp0basemodel/stream_2_conv_1/BiasAdd/ReadVariableOp2h
2basemodel/stream_2_conv_1/BiasAdd_1/ReadVariableOp2basemodel/stream_2_conv_1/BiasAdd_1/ReadVariableOp2|
<basemodel/stream_2_conv_1/conv1d/ExpandDims_1/ReadVariableOp<basemodel/stream_2_conv_1/conv1d/ExpandDims_1/ReadVariableOp2�
>basemodel/stream_2_conv_1/conv1d_1/ExpandDims_1/ReadVariableOp>basemodel/stream_2_conv_1/conv1d_1/ExpandDims_1/ReadVariableOp2^
-dense_1/kernel/Regularizer/Abs/ReadVariableOp-dense_1/kernel/Regularizer/Abs/ReadVariableOp2n
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp2t
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp2n
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp:V R
,
_output_shapes
:����������
"
_user_specified_name
inputs/0:VR
,
_output_shapes
:����������
"
_user_specified_name
inputs/1
�
k
L__inference_stream_1_drop_1_layer_call_and_return_conditional_losses_1988799

inputs
identity�c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
dropout/Constx
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:����������@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:����������@*
dtype0*
seed�*
seed2�2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>2
dropout/GreaterEqual/y�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:����������@2
dropout/GreaterEqual�
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:����������@2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:����������@2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:����������@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������@:T P
,
_output_shapes
:����������@
 
_user_specified_nameinputs
�
k
O__inference_dense_activation_1_layer_call_and_return_conditional_losses_1989047

inputs
identityW
SigmoidSigmoidinputs*
T0*'
_output_shapes
:���������T2	
Sigmoid_
IdentityIdentitySigmoid:y:0*
T0*'
_output_shapes
:���������T2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������T:O K
'
_output_shapes
:���������T
 
_user_specified_nameinputs
�
�
H__inference_concatenate_layer_call_and_return_conditional_losses_1988907
inputs_0
inputs_1
inputs_2
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis�
concatConcatV2inputs_0inputs_1inputs_2concat/axis:output:0*
N*
T0*(
_output_shapes
:����������2
concatd
IdentityIdentityconcat:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:���������@:���������@:���������@:Q M
'
_output_shapes
:���������@
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:���������@
"
_user_specified_name
inputs/1:QM
'
_output_shapes
:���������@
"
_user_specified_name
inputs/2
�
M
1__inference_dense_1_dropout_layer_call_fn_1988912

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_dense_1_dropout_layer_call_and_return_conditional_losses_19848992
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�+
�
P__inference_batch_normalization_layer_call_and_return_conditional_losses_1988395

inputs5
'assignmovingavg_readvariableop_resource:@7
)assignmovingavg_1_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@/
!batchnorm_readvariableop_resource:@
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOp�
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indices�
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(2
moments/mean�
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:@2
moments/StopGradient�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*,
_output_shapes
:����������@2
moments/SquaredDifference�
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices�
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(2
moments/variance�
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 2
moments/Squeeze�
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
�#<2
AssignMovingAvg/decay�
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg/sub�
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg/mul�
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
�#<2
AssignMovingAvg_1/decay�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg_1/sub�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg_1/mul�
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
 *o�:2
batchnorm/add/y�
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@2
batchnorm/Rsqrt�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOp�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2
batchnorm/mul{
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*,
_output_shapes
:����������@2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:@2
batchnorm/mul_2�
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp�
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*,
_output_shapes
:����������@2
batchnorm/add_1s
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*,
_output_shapes
:����������@2

Identity�
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������@: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:T P
,
_output_shapes
:����������@
 
_user_specified_nameinputs
�
j
L__inference_dense_1_dropout_layer_call_and_return_conditional_losses_1984899

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:����������2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:����������2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
F__inference_basemodel_layer_call_and_return_conditional_losses_1988010
inputs_0
inputs_1
inputs_2Q
;stream_2_conv_1_conv1d_expanddims_1_readvariableop_resource:@=
/stream_2_conv_1_biasadd_readvariableop_resource:@Q
;stream_1_conv_1_conv1d_expanddims_1_readvariableop_resource:@=
/stream_1_conv_1_biasadd_readvariableop_resource:@Q
;stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource:@=
/stream_0_conv_1_biasadd_readvariableop_resource:@K
=batch_normalization_2_assignmovingavg_readvariableop_resource:@M
?batch_normalization_2_assignmovingavg_1_readvariableop_resource:@I
;batch_normalization_2_batchnorm_mul_readvariableop_resource:@E
7batch_normalization_2_batchnorm_readvariableop_resource:@K
=batch_normalization_1_assignmovingavg_readvariableop_resource:@M
?batch_normalization_1_assignmovingavg_1_readvariableop_resource:@I
;batch_normalization_1_batchnorm_mul_readvariableop_resource:@E
7batch_normalization_1_batchnorm_readvariableop_resource:@I
;batch_normalization_assignmovingavg_readvariableop_resource:@K
=batch_normalization_assignmovingavg_1_readvariableop_resource:@G
9batch_normalization_batchnorm_mul_readvariableop_resource:@C
5batch_normalization_batchnorm_readvariableop_resource:@9
&dense_1_matmul_readvariableop_resource:	�T5
'dense_1_biasadd_readvariableop_resource:TK
=batch_normalization_3_assignmovingavg_readvariableop_resource:TM
?batch_normalization_3_assignmovingavg_1_readvariableop_resource:TI
;batch_normalization_3_batchnorm_mul_readvariableop_resource:TE
7batch_normalization_3_batchnorm_readvariableop_resource:T
identity��#batch_normalization/AssignMovingAvg�2batch_normalization/AssignMovingAvg/ReadVariableOp�%batch_normalization/AssignMovingAvg_1�4batch_normalization/AssignMovingAvg_1/ReadVariableOp�,batch_normalization/batchnorm/ReadVariableOp�0batch_normalization/batchnorm/mul/ReadVariableOp�%batch_normalization_1/AssignMovingAvg�4batch_normalization_1/AssignMovingAvg/ReadVariableOp�'batch_normalization_1/AssignMovingAvg_1�6batch_normalization_1/AssignMovingAvg_1/ReadVariableOp�.batch_normalization_1/batchnorm/ReadVariableOp�2batch_normalization_1/batchnorm/mul/ReadVariableOp�%batch_normalization_2/AssignMovingAvg�4batch_normalization_2/AssignMovingAvg/ReadVariableOp�'batch_normalization_2/AssignMovingAvg_1�6batch_normalization_2/AssignMovingAvg_1/ReadVariableOp�.batch_normalization_2/batchnorm/ReadVariableOp�2batch_normalization_2/batchnorm/mul/ReadVariableOp�%batch_normalization_3/AssignMovingAvg�4batch_normalization_3/AssignMovingAvg/ReadVariableOp�'batch_normalization_3/AssignMovingAvg_1�6batch_normalization_3/AssignMovingAvg_1/ReadVariableOp�.batch_normalization_3/batchnorm/ReadVariableOp�2batch_normalization_3/batchnorm/mul/ReadVariableOp�dense_1/BiasAdd/ReadVariableOp�dense_1/MatMul/ReadVariableOp�-dense_1/kernel/Regularizer/Abs/ReadVariableOp�&stream_0_conv_1/BiasAdd/ReadVariableOp�2stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp�5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp�&stream_1_conv_1/BiasAdd/ReadVariableOp�2stream_1_conv_1/conv1d/ExpandDims_1/ReadVariableOp�8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp�&stream_2_conv_1/BiasAdd/ReadVariableOp�2stream_2_conv_1/conv1d/ExpandDims_1/ReadVariableOp�5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp�
!stream_2_input_drop/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2#
!stream_2_input_drop/dropout/Const�
stream_2_input_drop/dropout/MulMulinputs_2*stream_2_input_drop/dropout/Const:output:0*
T0*,
_output_shapes
:����������2!
stream_2_input_drop/dropout/Mul~
!stream_2_input_drop/dropout/ShapeShapeinputs_2*
T0*
_output_shapes
:2#
!stream_2_input_drop/dropout/Shape�
8stream_2_input_drop/dropout/random_uniform/RandomUniformRandomUniform*stream_2_input_drop/dropout/Shape:output:0*
T0*,
_output_shapes
:����������*
dtype0*
seed�*
seed2�2:
8stream_2_input_drop/dropout/random_uniform/RandomUniform�
*stream_2_input_drop/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>2,
*stream_2_input_drop/dropout/GreaterEqual/y�
(stream_2_input_drop/dropout/GreaterEqualGreaterEqualAstream_2_input_drop/dropout/random_uniform/RandomUniform:output:03stream_2_input_drop/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:����������2*
(stream_2_input_drop/dropout/GreaterEqual�
 stream_2_input_drop/dropout/CastCast,stream_2_input_drop/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:����������2"
 stream_2_input_drop/dropout/Cast�
!stream_2_input_drop/dropout/Mul_1Mul#stream_2_input_drop/dropout/Mul:z:0$stream_2_input_drop/dropout/Cast:y:0*
T0*,
_output_shapes
:����������2#
!stream_2_input_drop/dropout/Mul_1�
!stream_1_input_drop/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2#
!stream_1_input_drop/dropout/Const�
stream_1_input_drop/dropout/MulMulinputs_1*stream_1_input_drop/dropout/Const:output:0*
T0*,
_output_shapes
:����������2!
stream_1_input_drop/dropout/Mul~
!stream_1_input_drop/dropout/ShapeShapeinputs_1*
T0*
_output_shapes
:2#
!stream_1_input_drop/dropout/Shape�
8stream_1_input_drop/dropout/random_uniform/RandomUniformRandomUniform*stream_1_input_drop/dropout/Shape:output:0*
T0*,
_output_shapes
:����������*
dtype0*
seed�*
seed2�2:
8stream_1_input_drop/dropout/random_uniform/RandomUniform�
*stream_1_input_drop/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>2,
*stream_1_input_drop/dropout/GreaterEqual/y�
(stream_1_input_drop/dropout/GreaterEqualGreaterEqualAstream_1_input_drop/dropout/random_uniform/RandomUniform:output:03stream_1_input_drop/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:����������2*
(stream_1_input_drop/dropout/GreaterEqual�
 stream_1_input_drop/dropout/CastCast,stream_1_input_drop/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:����������2"
 stream_1_input_drop/dropout/Cast�
!stream_1_input_drop/dropout/Mul_1Mul#stream_1_input_drop/dropout/Mul:z:0$stream_1_input_drop/dropout/Cast:y:0*
T0*,
_output_shapes
:����������2#
!stream_1_input_drop/dropout/Mul_1�
!stream_0_input_drop/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2#
!stream_0_input_drop/dropout/Const�
stream_0_input_drop/dropout/MulMulinputs_0*stream_0_input_drop/dropout/Const:output:0*
T0*,
_output_shapes
:����������2!
stream_0_input_drop/dropout/Mul~
!stream_0_input_drop/dropout/ShapeShapeinputs_0*
T0*
_output_shapes
:2#
!stream_0_input_drop/dropout/Shape�
8stream_0_input_drop/dropout/random_uniform/RandomUniformRandomUniform*stream_0_input_drop/dropout/Shape:output:0*
T0*,
_output_shapes
:����������*
dtype0*
seed�*
seed2�2:
8stream_0_input_drop/dropout/random_uniform/RandomUniform�
*stream_0_input_drop/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>2,
*stream_0_input_drop/dropout/GreaterEqual/y�
(stream_0_input_drop/dropout/GreaterEqualGreaterEqualAstream_0_input_drop/dropout/random_uniform/RandomUniform:output:03stream_0_input_drop/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:����������2*
(stream_0_input_drop/dropout/GreaterEqual�
 stream_0_input_drop/dropout/CastCast,stream_0_input_drop/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:����������2"
 stream_0_input_drop/dropout/Cast�
!stream_0_input_drop/dropout/Mul_1Mul#stream_0_input_drop/dropout/Mul:z:0$stream_0_input_drop/dropout/Cast:y:0*
T0*,
_output_shapes
:����������2#
!stream_0_input_drop/dropout/Mul_1�
%stream_2_conv_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2'
%stream_2_conv_1/conv1d/ExpandDims/dim�
!stream_2_conv_1/conv1d/ExpandDims
ExpandDims%stream_2_input_drop/dropout/Mul_1:z:0.stream_2_conv_1/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:����������2#
!stream_2_conv_1/conv1d/ExpandDims�
2stream_2_conv_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp;stream_2_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype024
2stream_2_conv_1/conv1d/ExpandDims_1/ReadVariableOp�
'stream_2_conv_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2)
'stream_2_conv_1/conv1d/ExpandDims_1/dim�
#stream_2_conv_1/conv1d/ExpandDims_1
ExpandDims:stream_2_conv_1/conv1d/ExpandDims_1/ReadVariableOp:value:00stream_2_conv_1/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@2%
#stream_2_conv_1/conv1d/ExpandDims_1�
stream_2_conv_1/conv1dConv2D*stream_2_conv_1/conv1d/ExpandDims:output:0,stream_2_conv_1/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:����������@*
paddingSAME*
strides
2
stream_2_conv_1/conv1d�
stream_2_conv_1/conv1d/SqueezeSqueezestream_2_conv_1/conv1d:output:0*
T0*,
_output_shapes
:����������@*
squeeze_dims

���������2 
stream_2_conv_1/conv1d/Squeeze�
&stream_2_conv_1/BiasAdd/ReadVariableOpReadVariableOp/stream_2_conv_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02(
&stream_2_conv_1/BiasAdd/ReadVariableOp�
stream_2_conv_1/BiasAddBiasAdd'stream_2_conv_1/conv1d/Squeeze:output:0.stream_2_conv_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������@2
stream_2_conv_1/BiasAdd�
%stream_1_conv_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2'
%stream_1_conv_1/conv1d/ExpandDims/dim�
!stream_1_conv_1/conv1d/ExpandDims
ExpandDims%stream_1_input_drop/dropout/Mul_1:z:0.stream_1_conv_1/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:����������2#
!stream_1_conv_1/conv1d/ExpandDims�
2stream_1_conv_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp;stream_1_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype024
2stream_1_conv_1/conv1d/ExpandDims_1/ReadVariableOp�
'stream_1_conv_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2)
'stream_1_conv_1/conv1d/ExpandDims_1/dim�
#stream_1_conv_1/conv1d/ExpandDims_1
ExpandDims:stream_1_conv_1/conv1d/ExpandDims_1/ReadVariableOp:value:00stream_1_conv_1/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@2%
#stream_1_conv_1/conv1d/ExpandDims_1�
stream_1_conv_1/conv1dConv2D*stream_1_conv_1/conv1d/ExpandDims:output:0,stream_1_conv_1/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:����������@*
paddingSAME*
strides
2
stream_1_conv_1/conv1d�
stream_1_conv_1/conv1d/SqueezeSqueezestream_1_conv_1/conv1d:output:0*
T0*,
_output_shapes
:����������@*
squeeze_dims

���������2 
stream_1_conv_1/conv1d/Squeeze�
&stream_1_conv_1/BiasAdd/ReadVariableOpReadVariableOp/stream_1_conv_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02(
&stream_1_conv_1/BiasAdd/ReadVariableOp�
stream_1_conv_1/BiasAddBiasAdd'stream_1_conv_1/conv1d/Squeeze:output:0.stream_1_conv_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������@2
stream_1_conv_1/BiasAdd�
%stream_0_conv_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2'
%stream_0_conv_1/conv1d/ExpandDims/dim�
!stream_0_conv_1/conv1d/ExpandDims
ExpandDims%stream_0_input_drop/dropout/Mul_1:z:0.stream_0_conv_1/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:����������2#
!stream_0_conv_1/conv1d/ExpandDims�
2stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp;stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype024
2stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp�
'stream_0_conv_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2)
'stream_0_conv_1/conv1d/ExpandDims_1/dim�
#stream_0_conv_1/conv1d/ExpandDims_1
ExpandDims:stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp:value:00stream_0_conv_1/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@2%
#stream_0_conv_1/conv1d/ExpandDims_1�
stream_0_conv_1/conv1dConv2D*stream_0_conv_1/conv1d/ExpandDims:output:0,stream_0_conv_1/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:����������@*
paddingSAME*
strides
2
stream_0_conv_1/conv1d�
stream_0_conv_1/conv1d/SqueezeSqueezestream_0_conv_1/conv1d:output:0*
T0*,
_output_shapes
:����������@*
squeeze_dims

���������2 
stream_0_conv_1/conv1d/Squeeze�
&stream_0_conv_1/BiasAdd/ReadVariableOpReadVariableOp/stream_0_conv_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02(
&stream_0_conv_1/BiasAdd/ReadVariableOp�
stream_0_conv_1/BiasAddBiasAdd'stream_0_conv_1/conv1d/Squeeze:output:0.stream_0_conv_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������@2
stream_0_conv_1/BiasAdd�
4batch_normalization_2/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       26
4batch_normalization_2/moments/mean/reduction_indices�
"batch_normalization_2/moments/meanMean stream_2_conv_1/BiasAdd:output:0=batch_normalization_2/moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(2$
"batch_normalization_2/moments/mean�
*batch_normalization_2/moments/StopGradientStopGradient+batch_normalization_2/moments/mean:output:0*
T0*"
_output_shapes
:@2,
*batch_normalization_2/moments/StopGradient�
/batch_normalization_2/moments/SquaredDifferenceSquaredDifference stream_2_conv_1/BiasAdd:output:03batch_normalization_2/moments/StopGradient:output:0*
T0*,
_output_shapes
:����������@21
/batch_normalization_2/moments/SquaredDifference�
8batch_normalization_2/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2:
8batch_normalization_2/moments/variance/reduction_indices�
&batch_normalization_2/moments/varianceMean3batch_normalization_2/moments/SquaredDifference:z:0Abatch_normalization_2/moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(2(
&batch_normalization_2/moments/variance�
%batch_normalization_2/moments/SqueezeSqueeze+batch_normalization_2/moments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 2'
%batch_normalization_2/moments/Squeeze�
'batch_normalization_2/moments/Squeeze_1Squeeze/batch_normalization_2/moments/variance:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 2)
'batch_normalization_2/moments/Squeeze_1�
+batch_normalization_2/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2-
+batch_normalization_2/AssignMovingAvg/decay�
4batch_normalization_2/AssignMovingAvg/ReadVariableOpReadVariableOp=batch_normalization_2_assignmovingavg_readvariableop_resource*
_output_shapes
:@*
dtype026
4batch_normalization_2/AssignMovingAvg/ReadVariableOp�
)batch_normalization_2/AssignMovingAvg/subSub<batch_normalization_2/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_2/moments/Squeeze:output:0*
T0*
_output_shapes
:@2+
)batch_normalization_2/AssignMovingAvg/sub�
)batch_normalization_2/AssignMovingAvg/mulMul-batch_normalization_2/AssignMovingAvg/sub:z:04batch_normalization_2/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:@2+
)batch_normalization_2/AssignMovingAvg/mul�
%batch_normalization_2/AssignMovingAvgAssignSubVariableOp=batch_normalization_2_assignmovingavg_readvariableop_resource-batch_normalization_2/AssignMovingAvg/mul:z:05^batch_normalization_2/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02'
%batch_normalization_2/AssignMovingAvg�
-batch_normalization_2/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2/
-batch_normalization_2/AssignMovingAvg_1/decay�
6batch_normalization_2/AssignMovingAvg_1/ReadVariableOpReadVariableOp?batch_normalization_2_assignmovingavg_1_readvariableop_resource*
_output_shapes
:@*
dtype028
6batch_normalization_2/AssignMovingAvg_1/ReadVariableOp�
+batch_normalization_2/AssignMovingAvg_1/subSub>batch_normalization_2/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_2/moments/Squeeze_1:output:0*
T0*
_output_shapes
:@2-
+batch_normalization_2/AssignMovingAvg_1/sub�
+batch_normalization_2/AssignMovingAvg_1/mulMul/batch_normalization_2/AssignMovingAvg_1/sub:z:06batch_normalization_2/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:@2-
+batch_normalization_2/AssignMovingAvg_1/mul�
'batch_normalization_2/AssignMovingAvg_1AssignSubVariableOp?batch_normalization_2_assignmovingavg_1_readvariableop_resource/batch_normalization_2/AssignMovingAvg_1/mul:z:07^batch_normalization_2/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02)
'batch_normalization_2/AssignMovingAvg_1�
%batch_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2'
%batch_normalization_2/batchnorm/add/y�
#batch_normalization_2/batchnorm/addAddV20batch_normalization_2/moments/Squeeze_1:output:0.batch_normalization_2/batchnorm/add/y:output:0*
T0*
_output_shapes
:@2%
#batch_normalization_2/batchnorm/add�
%batch_normalization_2/batchnorm/RsqrtRsqrt'batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes
:@2'
%batch_normalization_2/batchnorm/Rsqrt�
2batch_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype024
2batch_normalization_2/batchnorm/mul/ReadVariableOp�
#batch_normalization_2/batchnorm/mulMul)batch_normalization_2/batchnorm/Rsqrt:y:0:batch_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2%
#batch_normalization_2/batchnorm/mul�
%batch_normalization_2/batchnorm/mul_1Mul stream_2_conv_1/BiasAdd:output:0'batch_normalization_2/batchnorm/mul:z:0*
T0*,
_output_shapes
:����������@2'
%batch_normalization_2/batchnorm/mul_1�
%batch_normalization_2/batchnorm/mul_2Mul.batch_normalization_2/moments/Squeeze:output:0'batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes
:@2'
%batch_normalization_2/batchnorm/mul_2�
.batch_normalization_2/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_2_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype020
.batch_normalization_2/batchnorm/ReadVariableOp�
#batch_normalization_2/batchnorm/subSub6batch_normalization_2/batchnorm/ReadVariableOp:value:0)batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2%
#batch_normalization_2/batchnorm/sub�
%batch_normalization_2/batchnorm/add_1AddV2)batch_normalization_2/batchnorm/mul_1:z:0'batch_normalization_2/batchnorm/sub:z:0*
T0*,
_output_shapes
:����������@2'
%batch_normalization_2/batchnorm/add_1�
4batch_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       26
4batch_normalization_1/moments/mean/reduction_indices�
"batch_normalization_1/moments/meanMean stream_1_conv_1/BiasAdd:output:0=batch_normalization_1/moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(2$
"batch_normalization_1/moments/mean�
*batch_normalization_1/moments/StopGradientStopGradient+batch_normalization_1/moments/mean:output:0*
T0*"
_output_shapes
:@2,
*batch_normalization_1/moments/StopGradient�
/batch_normalization_1/moments/SquaredDifferenceSquaredDifference stream_1_conv_1/BiasAdd:output:03batch_normalization_1/moments/StopGradient:output:0*
T0*,
_output_shapes
:����������@21
/batch_normalization_1/moments/SquaredDifference�
8batch_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2:
8batch_normalization_1/moments/variance/reduction_indices�
&batch_normalization_1/moments/varianceMean3batch_normalization_1/moments/SquaredDifference:z:0Abatch_normalization_1/moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(2(
&batch_normalization_1/moments/variance�
%batch_normalization_1/moments/SqueezeSqueeze+batch_normalization_1/moments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 2'
%batch_normalization_1/moments/Squeeze�
'batch_normalization_1/moments/Squeeze_1Squeeze/batch_normalization_1/moments/variance:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 2)
'batch_normalization_1/moments/Squeeze_1�
+batch_normalization_1/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2-
+batch_normalization_1/AssignMovingAvg/decay�
4batch_normalization_1/AssignMovingAvg/ReadVariableOpReadVariableOp=batch_normalization_1_assignmovingavg_readvariableop_resource*
_output_shapes
:@*
dtype026
4batch_normalization_1/AssignMovingAvg/ReadVariableOp�
)batch_normalization_1/AssignMovingAvg/subSub<batch_normalization_1/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_1/moments/Squeeze:output:0*
T0*
_output_shapes
:@2+
)batch_normalization_1/AssignMovingAvg/sub�
)batch_normalization_1/AssignMovingAvg/mulMul-batch_normalization_1/AssignMovingAvg/sub:z:04batch_normalization_1/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:@2+
)batch_normalization_1/AssignMovingAvg/mul�
%batch_normalization_1/AssignMovingAvgAssignSubVariableOp=batch_normalization_1_assignmovingavg_readvariableop_resource-batch_normalization_1/AssignMovingAvg/mul:z:05^batch_normalization_1/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02'
%batch_normalization_1/AssignMovingAvg�
-batch_normalization_1/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2/
-batch_normalization_1/AssignMovingAvg_1/decay�
6batch_normalization_1/AssignMovingAvg_1/ReadVariableOpReadVariableOp?batch_normalization_1_assignmovingavg_1_readvariableop_resource*
_output_shapes
:@*
dtype028
6batch_normalization_1/AssignMovingAvg_1/ReadVariableOp�
+batch_normalization_1/AssignMovingAvg_1/subSub>batch_normalization_1/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_1/moments/Squeeze_1:output:0*
T0*
_output_shapes
:@2-
+batch_normalization_1/AssignMovingAvg_1/sub�
+batch_normalization_1/AssignMovingAvg_1/mulMul/batch_normalization_1/AssignMovingAvg_1/sub:z:06batch_normalization_1/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:@2-
+batch_normalization_1/AssignMovingAvg_1/mul�
'batch_normalization_1/AssignMovingAvg_1AssignSubVariableOp?batch_normalization_1_assignmovingavg_1_readvariableop_resource/batch_normalization_1/AssignMovingAvg_1/mul:z:07^batch_normalization_1/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02)
'batch_normalization_1/AssignMovingAvg_1�
%batch_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2'
%batch_normalization_1/batchnorm/add/y�
#batch_normalization_1/batchnorm/addAddV20batch_normalization_1/moments/Squeeze_1:output:0.batch_normalization_1/batchnorm/add/y:output:0*
T0*
_output_shapes
:@2%
#batch_normalization_1/batchnorm/add�
%batch_normalization_1/batchnorm/RsqrtRsqrt'batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes
:@2'
%batch_normalization_1/batchnorm/Rsqrt�
2batch_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype024
2batch_normalization_1/batchnorm/mul/ReadVariableOp�
#batch_normalization_1/batchnorm/mulMul)batch_normalization_1/batchnorm/Rsqrt:y:0:batch_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2%
#batch_normalization_1/batchnorm/mul�
%batch_normalization_1/batchnorm/mul_1Mul stream_1_conv_1/BiasAdd:output:0'batch_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:����������@2'
%batch_normalization_1/batchnorm/mul_1�
%batch_normalization_1/batchnorm/mul_2Mul.batch_normalization_1/moments/Squeeze:output:0'batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes
:@2'
%batch_normalization_1/batchnorm/mul_2�
.batch_normalization_1/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_1_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype020
.batch_normalization_1/batchnorm/ReadVariableOp�
#batch_normalization_1/batchnorm/subSub6batch_normalization_1/batchnorm/ReadVariableOp:value:0)batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2%
#batch_normalization_1/batchnorm/sub�
%batch_normalization_1/batchnorm/add_1AddV2)batch_normalization_1/batchnorm/mul_1:z:0'batch_normalization_1/batchnorm/sub:z:0*
T0*,
_output_shapes
:����������@2'
%batch_normalization_1/batchnorm/add_1�
2batch_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       24
2batch_normalization/moments/mean/reduction_indices�
 batch_normalization/moments/meanMean stream_0_conv_1/BiasAdd:output:0;batch_normalization/moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(2"
 batch_normalization/moments/mean�
(batch_normalization/moments/StopGradientStopGradient)batch_normalization/moments/mean:output:0*
T0*"
_output_shapes
:@2*
(batch_normalization/moments/StopGradient�
-batch_normalization/moments/SquaredDifferenceSquaredDifference stream_0_conv_1/BiasAdd:output:01batch_normalization/moments/StopGradient:output:0*
T0*,
_output_shapes
:����������@2/
-batch_normalization/moments/SquaredDifference�
6batch_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       28
6batch_normalization/moments/variance/reduction_indices�
$batch_normalization/moments/varianceMean1batch_normalization/moments/SquaredDifference:z:0?batch_normalization/moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(2&
$batch_normalization/moments/variance�
#batch_normalization/moments/SqueezeSqueeze)batch_normalization/moments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 2%
#batch_normalization/moments/Squeeze�
%batch_normalization/moments/Squeeze_1Squeeze-batch_normalization/moments/variance:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 2'
%batch_normalization/moments/Squeeze_1�
)batch_normalization/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2+
)batch_normalization/AssignMovingAvg/decay�
2batch_normalization/AssignMovingAvg/ReadVariableOpReadVariableOp;batch_normalization_assignmovingavg_readvariableop_resource*
_output_shapes
:@*
dtype024
2batch_normalization/AssignMovingAvg/ReadVariableOp�
'batch_normalization/AssignMovingAvg/subSub:batch_normalization/AssignMovingAvg/ReadVariableOp:value:0,batch_normalization/moments/Squeeze:output:0*
T0*
_output_shapes
:@2)
'batch_normalization/AssignMovingAvg/sub�
'batch_normalization/AssignMovingAvg/mulMul+batch_normalization/AssignMovingAvg/sub:z:02batch_normalization/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:@2)
'batch_normalization/AssignMovingAvg/mul�
#batch_normalization/AssignMovingAvgAssignSubVariableOp;batch_normalization_assignmovingavg_readvariableop_resource+batch_normalization/AssignMovingAvg/mul:z:03^batch_normalization/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02%
#batch_normalization/AssignMovingAvg�
+batch_normalization/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2-
+batch_normalization/AssignMovingAvg_1/decay�
4batch_normalization/AssignMovingAvg_1/ReadVariableOpReadVariableOp=batch_normalization_assignmovingavg_1_readvariableop_resource*
_output_shapes
:@*
dtype026
4batch_normalization/AssignMovingAvg_1/ReadVariableOp�
)batch_normalization/AssignMovingAvg_1/subSub<batch_normalization/AssignMovingAvg_1/ReadVariableOp:value:0.batch_normalization/moments/Squeeze_1:output:0*
T0*
_output_shapes
:@2+
)batch_normalization/AssignMovingAvg_1/sub�
)batch_normalization/AssignMovingAvg_1/mulMul-batch_normalization/AssignMovingAvg_1/sub:z:04batch_normalization/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:@2+
)batch_normalization/AssignMovingAvg_1/mul�
%batch_normalization/AssignMovingAvg_1AssignSubVariableOp=batch_normalization_assignmovingavg_1_readvariableop_resource-batch_normalization/AssignMovingAvg_1/mul:z:05^batch_normalization/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02'
%batch_normalization/AssignMovingAvg_1�
#batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2%
#batch_normalization/batchnorm/add/y�
!batch_normalization/batchnorm/addAddV2.batch_normalization/moments/Squeeze_1:output:0,batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes
:@2#
!batch_normalization/batchnorm/add�
#batch_normalization/batchnorm/RsqrtRsqrt%batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes
:@2%
#batch_normalization/batchnorm/Rsqrt�
0batch_normalization/batchnorm/mul/ReadVariableOpReadVariableOp9batch_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype022
0batch_normalization/batchnorm/mul/ReadVariableOp�
!batch_normalization/batchnorm/mulMul'batch_normalization/batchnorm/Rsqrt:y:08batch_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2#
!batch_normalization/batchnorm/mul�
#batch_normalization/batchnorm/mul_1Mul stream_0_conv_1/BiasAdd:output:0%batch_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:����������@2%
#batch_normalization/batchnorm/mul_1�
#batch_normalization/batchnorm/mul_2Mul,batch_normalization/moments/Squeeze:output:0%batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes
:@2%
#batch_normalization/batchnorm/mul_2�
,batch_normalization/batchnorm/ReadVariableOpReadVariableOp5batch_normalization_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02.
,batch_normalization/batchnorm/ReadVariableOp�
!batch_normalization/batchnorm/subSub4batch_normalization/batchnorm/ReadVariableOp:value:0'batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2#
!batch_normalization/batchnorm/sub�
#batch_normalization/batchnorm/add_1AddV2'batch_normalization/batchnorm/mul_1:z:0%batch_normalization/batchnorm/sub:z:0*
T0*,
_output_shapes
:����������@2%
#batch_normalization/batchnorm/add_1�
activation_2/ReluRelu)batch_normalization_2/batchnorm/add_1:z:0*
T0*,
_output_shapes
:����������@2
activation_2/Relu�
activation_1/ReluRelu)batch_normalization_1/batchnorm/add_1:z:0*
T0*,
_output_shapes
:����������@2
activation_1/Relu�
activation/ReluRelu'batch_normalization/batchnorm/add_1:z:0*
T0*,
_output_shapes
:����������@2
activation/Relu�
stream_2_drop_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
stream_2_drop_1/dropout/Const�
stream_2_drop_1/dropout/MulMulactivation_2/Relu:activations:0&stream_2_drop_1/dropout/Const:output:0*
T0*,
_output_shapes
:����������@2
stream_2_drop_1/dropout/Mul�
stream_2_drop_1/dropout/ShapeShapeactivation_2/Relu:activations:0*
T0*
_output_shapes
:2
stream_2_drop_1/dropout/Shape�
4stream_2_drop_1/dropout/random_uniform/RandomUniformRandomUniform&stream_2_drop_1/dropout/Shape:output:0*
T0*,
_output_shapes
:����������@*
dtype0*
seed�*
seed2�26
4stream_2_drop_1/dropout/random_uniform/RandomUniform�
&stream_2_drop_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>2(
&stream_2_drop_1/dropout/GreaterEqual/y�
$stream_2_drop_1/dropout/GreaterEqualGreaterEqual=stream_2_drop_1/dropout/random_uniform/RandomUniform:output:0/stream_2_drop_1/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:����������@2&
$stream_2_drop_1/dropout/GreaterEqual�
stream_2_drop_1/dropout/CastCast(stream_2_drop_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:����������@2
stream_2_drop_1/dropout/Cast�
stream_2_drop_1/dropout/Mul_1Mulstream_2_drop_1/dropout/Mul:z:0 stream_2_drop_1/dropout/Cast:y:0*
T0*,
_output_shapes
:����������@2
stream_2_drop_1/dropout/Mul_1�
stream_1_drop_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
stream_1_drop_1/dropout/Const�
stream_1_drop_1/dropout/MulMulactivation_1/Relu:activations:0&stream_1_drop_1/dropout/Const:output:0*
T0*,
_output_shapes
:����������@2
stream_1_drop_1/dropout/Mul�
stream_1_drop_1/dropout/ShapeShapeactivation_1/Relu:activations:0*
T0*
_output_shapes
:2
stream_1_drop_1/dropout/Shape�
4stream_1_drop_1/dropout/random_uniform/RandomUniformRandomUniform&stream_1_drop_1/dropout/Shape:output:0*
T0*,
_output_shapes
:����������@*
dtype0*
seed�*
seed2�26
4stream_1_drop_1/dropout/random_uniform/RandomUniform�
&stream_1_drop_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>2(
&stream_1_drop_1/dropout/GreaterEqual/y�
$stream_1_drop_1/dropout/GreaterEqualGreaterEqual=stream_1_drop_1/dropout/random_uniform/RandomUniform:output:0/stream_1_drop_1/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:����������@2&
$stream_1_drop_1/dropout/GreaterEqual�
stream_1_drop_1/dropout/CastCast(stream_1_drop_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:����������@2
stream_1_drop_1/dropout/Cast�
stream_1_drop_1/dropout/Mul_1Mulstream_1_drop_1/dropout/Mul:z:0 stream_1_drop_1/dropout/Cast:y:0*
T0*,
_output_shapes
:����������@2
stream_1_drop_1/dropout/Mul_1�
stream_0_drop_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
stream_0_drop_1/dropout/Const�
stream_0_drop_1/dropout/MulMulactivation/Relu:activations:0&stream_0_drop_1/dropout/Const:output:0*
T0*,
_output_shapes
:����������@2
stream_0_drop_1/dropout/Mul�
stream_0_drop_1/dropout/ShapeShapeactivation/Relu:activations:0*
T0*
_output_shapes
:2
stream_0_drop_1/dropout/Shape�
4stream_0_drop_1/dropout/random_uniform/RandomUniformRandomUniform&stream_0_drop_1/dropout/Shape:output:0*
T0*,
_output_shapes
:����������@*
dtype0*
seed�*
seed2�26
4stream_0_drop_1/dropout/random_uniform/RandomUniform�
&stream_0_drop_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>2(
&stream_0_drop_1/dropout/GreaterEqual/y�
$stream_0_drop_1/dropout/GreaterEqualGreaterEqual=stream_0_drop_1/dropout/random_uniform/RandomUniform:output:0/stream_0_drop_1/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:����������@2&
$stream_0_drop_1/dropout/GreaterEqual�
stream_0_drop_1/dropout/CastCast(stream_0_drop_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:����������@2
stream_0_drop_1/dropout/Cast�
stream_0_drop_1/dropout/Mul_1Mulstream_0_drop_1/dropout/Mul:z:0 stream_0_drop_1/dropout/Cast:y:0*
T0*,
_output_shapes
:����������@2
stream_0_drop_1/dropout/Mul_1�
/global_average_pooling1d/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :21
/global_average_pooling1d/Mean/reduction_indices�
global_average_pooling1d/MeanMean!stream_0_drop_1/dropout/Mul_1:z:08global_average_pooling1d/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:���������@2
global_average_pooling1d/Mean�
1global_average_pooling1d_1/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :23
1global_average_pooling1d_1/Mean/reduction_indices�
global_average_pooling1d_1/MeanMean!stream_1_drop_1/dropout/Mul_1:z:0:global_average_pooling1d_1/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:���������@2!
global_average_pooling1d_1/Mean�
1global_average_pooling1d_2/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :23
1global_average_pooling1d_2/Mean/reduction_indices�
global_average_pooling1d_2/MeanMean!stream_2_drop_1/dropout/Mul_1:z:0:global_average_pooling1d_2/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:���������@2!
global_average_pooling1d_2/Meant
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate/concat/axis�
concatenate/concatConcatV2&global_average_pooling1d/Mean:output:0(global_average_pooling1d_1/Mean:output:0(global_average_pooling1d_2/Mean:output:0 concatenate/concat/axis:output:0*
N*
T0*(
_output_shapes
:����������2
concatenate/concat�
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	�T*
dtype02
dense_1/MatMul/ReadVariableOp�
dense_1/MatMulMatMulconcatenate/concat:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������T2
dense_1/MatMul�
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:T*
dtype02 
dense_1/BiasAdd/ReadVariableOp�
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������T2
dense_1/BiasAdd�
4batch_normalization_3/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 26
4batch_normalization_3/moments/mean/reduction_indices�
"batch_normalization_3/moments/meanMeandense_1/BiasAdd:output:0=batch_normalization_3/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:T*
	keep_dims(2$
"batch_normalization_3/moments/mean�
*batch_normalization_3/moments/StopGradientStopGradient+batch_normalization_3/moments/mean:output:0*
T0*
_output_shapes

:T2,
*batch_normalization_3/moments/StopGradient�
/batch_normalization_3/moments/SquaredDifferenceSquaredDifferencedense_1/BiasAdd:output:03batch_normalization_3/moments/StopGradient:output:0*
T0*'
_output_shapes
:���������T21
/batch_normalization_3/moments/SquaredDifference�
8batch_normalization_3/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2:
8batch_normalization_3/moments/variance/reduction_indices�
&batch_normalization_3/moments/varianceMean3batch_normalization_3/moments/SquaredDifference:z:0Abatch_normalization_3/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:T*
	keep_dims(2(
&batch_normalization_3/moments/variance�
%batch_normalization_3/moments/SqueezeSqueeze+batch_normalization_3/moments/mean:output:0*
T0*
_output_shapes
:T*
squeeze_dims
 2'
%batch_normalization_3/moments/Squeeze�
'batch_normalization_3/moments/Squeeze_1Squeeze/batch_normalization_3/moments/variance:output:0*
T0*
_output_shapes
:T*
squeeze_dims
 2)
'batch_normalization_3/moments/Squeeze_1�
+batch_normalization_3/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2-
+batch_normalization_3/AssignMovingAvg/decay�
4batch_normalization_3/AssignMovingAvg/ReadVariableOpReadVariableOp=batch_normalization_3_assignmovingavg_readvariableop_resource*
_output_shapes
:T*
dtype026
4batch_normalization_3/AssignMovingAvg/ReadVariableOp�
)batch_normalization_3/AssignMovingAvg/subSub<batch_normalization_3/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_3/moments/Squeeze:output:0*
T0*
_output_shapes
:T2+
)batch_normalization_3/AssignMovingAvg/sub�
)batch_normalization_3/AssignMovingAvg/mulMul-batch_normalization_3/AssignMovingAvg/sub:z:04batch_normalization_3/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:T2+
)batch_normalization_3/AssignMovingAvg/mul�
%batch_normalization_3/AssignMovingAvgAssignSubVariableOp=batch_normalization_3_assignmovingavg_readvariableop_resource-batch_normalization_3/AssignMovingAvg/mul:z:05^batch_normalization_3/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02'
%batch_normalization_3/AssignMovingAvg�
-batch_normalization_3/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2/
-batch_normalization_3/AssignMovingAvg_1/decay�
6batch_normalization_3/AssignMovingAvg_1/ReadVariableOpReadVariableOp?batch_normalization_3_assignmovingavg_1_readvariableop_resource*
_output_shapes
:T*
dtype028
6batch_normalization_3/AssignMovingAvg_1/ReadVariableOp�
+batch_normalization_3/AssignMovingAvg_1/subSub>batch_normalization_3/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_3/moments/Squeeze_1:output:0*
T0*
_output_shapes
:T2-
+batch_normalization_3/AssignMovingAvg_1/sub�
+batch_normalization_3/AssignMovingAvg_1/mulMul/batch_normalization_3/AssignMovingAvg_1/sub:z:06batch_normalization_3/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:T2-
+batch_normalization_3/AssignMovingAvg_1/mul�
'batch_normalization_3/AssignMovingAvg_1AssignSubVariableOp?batch_normalization_3_assignmovingavg_1_readvariableop_resource/batch_normalization_3/AssignMovingAvg_1/mul:z:07^batch_normalization_3/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02)
'batch_normalization_3/AssignMovingAvg_1�
%batch_normalization_3/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2'
%batch_normalization_3/batchnorm/add/y�
#batch_normalization_3/batchnorm/addAddV20batch_normalization_3/moments/Squeeze_1:output:0.batch_normalization_3/batchnorm/add/y:output:0*
T0*
_output_shapes
:T2%
#batch_normalization_3/batchnorm/add�
%batch_normalization_3/batchnorm/RsqrtRsqrt'batch_normalization_3/batchnorm/add:z:0*
T0*
_output_shapes
:T2'
%batch_normalization_3/batchnorm/Rsqrt�
2batch_normalization_3/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_3_batchnorm_mul_readvariableop_resource*
_output_shapes
:T*
dtype024
2batch_normalization_3/batchnorm/mul/ReadVariableOp�
#batch_normalization_3/batchnorm/mulMul)batch_normalization_3/batchnorm/Rsqrt:y:0:batch_normalization_3/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:T2%
#batch_normalization_3/batchnorm/mul�
%batch_normalization_3/batchnorm/mul_1Muldense_1/BiasAdd:output:0'batch_normalization_3/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������T2'
%batch_normalization_3/batchnorm/mul_1�
%batch_normalization_3/batchnorm/mul_2Mul.batch_normalization_3/moments/Squeeze:output:0'batch_normalization_3/batchnorm/mul:z:0*
T0*
_output_shapes
:T2'
%batch_normalization_3/batchnorm/mul_2�
.batch_normalization_3/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_3_batchnorm_readvariableop_resource*
_output_shapes
:T*
dtype020
.batch_normalization_3/batchnorm/ReadVariableOp�
#batch_normalization_3/batchnorm/subSub6batch_normalization_3/batchnorm/ReadVariableOp:value:0)batch_normalization_3/batchnorm/mul_2:z:0*
T0*
_output_shapes
:T2%
#batch_normalization_3/batchnorm/sub�
%batch_normalization_3/batchnorm/add_1AddV2)batch_normalization_3/batchnorm/mul_1:z:0'batch_normalization_3/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������T2'
%batch_normalization_3/batchnorm/add_1�
dense_activation_1/SigmoidSigmoid)batch_normalization_3/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������T2
dense_activation_1/Sigmoid�
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp;stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype027
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp�
&stream_0_conv_1/kernel/Regularizer/AbsAbs=stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_0_conv_1/kernel/Regularizer/Abs�
(stream_0_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_0_conv_1/kernel/Regularizer/Const�
&stream_0_conv_1/kernel/Regularizer/SumSum*stream_0_conv_1/kernel/Regularizer/Abs:y:01stream_0_conv_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/Sum�
(stream_0_conv_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2*
(stream_0_conv_1/kernel/Regularizer/mul/x�
&stream_0_conv_1/kernel/Regularizer/mulMul1stream_0_conv_1/kernel/Regularizer/mul/x:output:0/stream_0_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/mul�
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;stream_1_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02:
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp�
)stream_1_conv_1/kernel/Regularizer/SquareSquare@stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2+
)stream_1_conv_1/kernel/Regularizer/Square�
(stream_1_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_1_conv_1/kernel/Regularizer/Const�
&stream_1_conv_1/kernel/Regularizer/SumSum-stream_1_conv_1/kernel/Regularizer/Square:y:01stream_1_conv_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&stream_1_conv_1/kernel/Regularizer/Sum�
(stream_1_conv_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2*
(stream_1_conv_1/kernel/Regularizer/mul/x�
&stream_1_conv_1/kernel/Regularizer/mulMul1stream_1_conv_1/kernel/Regularizer/mul/x:output:0/stream_1_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_1_conv_1/kernel/Regularizer/mul�
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp;stream_2_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype027
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp�
&stream_2_conv_1/kernel/Regularizer/AbsAbs=stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_2_conv_1/kernel/Regularizer/Abs�
(stream_2_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_2_conv_1/kernel/Regularizer/Const�
&stream_2_conv_1/kernel/Regularizer/SumSum*stream_2_conv_1/kernel/Regularizer/Abs:y:01stream_2_conv_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&stream_2_conv_1/kernel/Regularizer/Sum�
(stream_2_conv_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2*
(stream_2_conv_1/kernel/Regularizer/mul/x�
&stream_2_conv_1/kernel/Regularizer/mulMul1stream_2_conv_1/kernel/Regularizer/mul/x:output:0/stream_2_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_2_conv_1/kernel/Regularizer/mul�
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	�T*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOp�
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	�T2 
dense_1/kernel/Regularizer/Abs�
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_1/kernel/Regularizer/Const�
dense_1/kernel/Regularizer/SumSum"dense_1/kernel/Regularizer/Abs:y:0)dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/Sum�
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2"
 dense_1/kernel/Regularizer/mul/x�
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/muly
IdentityIdentitydense_activation_1/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:���������T2

Identity�
NoOpNoOp$^batch_normalization/AssignMovingAvg3^batch_normalization/AssignMovingAvg/ReadVariableOp&^batch_normalization/AssignMovingAvg_15^batch_normalization/AssignMovingAvg_1/ReadVariableOp-^batch_normalization/batchnorm/ReadVariableOp1^batch_normalization/batchnorm/mul/ReadVariableOp&^batch_normalization_1/AssignMovingAvg5^batch_normalization_1/AssignMovingAvg/ReadVariableOp(^batch_normalization_1/AssignMovingAvg_17^batch_normalization_1/AssignMovingAvg_1/ReadVariableOp/^batch_normalization_1/batchnorm/ReadVariableOp3^batch_normalization_1/batchnorm/mul/ReadVariableOp&^batch_normalization_2/AssignMovingAvg5^batch_normalization_2/AssignMovingAvg/ReadVariableOp(^batch_normalization_2/AssignMovingAvg_17^batch_normalization_2/AssignMovingAvg_1/ReadVariableOp/^batch_normalization_2/batchnorm/ReadVariableOp3^batch_normalization_2/batchnorm/mul/ReadVariableOp&^batch_normalization_3/AssignMovingAvg5^batch_normalization_3/AssignMovingAvg/ReadVariableOp(^batch_normalization_3/AssignMovingAvg_17^batch_normalization_3/AssignMovingAvg_1/ReadVariableOp/^batch_normalization_3/batchnorm/ReadVariableOp3^batch_normalization_3/batchnorm/mul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp.^dense_1/kernel/Regularizer/Abs/ReadVariableOp'^stream_0_conv_1/BiasAdd/ReadVariableOp3^stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp'^stream_1_conv_1/BiasAdd/ReadVariableOp3^stream_1_conv_1/conv1d/ExpandDims_1/ReadVariableOp9^stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp'^stream_2_conv_1/BiasAdd/ReadVariableOp3^stream_2_conv_1/conv1d/ExpandDims_1/ReadVariableOp6^stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapesz
x:����������:����������:����������: : : : : : : : : : : : : : : : : : : : : : : : 2J
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
&stream_1_conv_1/BiasAdd/ReadVariableOp&stream_1_conv_1/BiasAdd/ReadVariableOp2h
2stream_1_conv_1/conv1d/ExpandDims_1/ReadVariableOp2stream_1_conv_1/conv1d/ExpandDims_1/ReadVariableOp2t
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp2P
&stream_2_conv_1/BiasAdd/ReadVariableOp&stream_2_conv_1/BiasAdd/ReadVariableOp2h
2stream_2_conv_1/conv1d/ExpandDims_1/ReadVariableOp2stream_2_conv_1/conv1d/ExpandDims_1/ReadVariableOp2n
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp:V R
,
_output_shapes
:����������
"
_user_specified_name
inputs/0:VR
,
_output_shapes
:����������
"
_user_specified_name
inputs/1:VR
,
_output_shapes
:����������
"
_user_specified_name
inputs/2
�
k
L__inference_stream_0_drop_1_layer_call_and_return_conditional_losses_1988772

inputs
identity�c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
dropout/Constx
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:����������@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:����������@*
dtype0*
seed�*
seed2�2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>2
dropout/GreaterEqual/y�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:����������@2
dropout/GreaterEqual�
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:����������@2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:����������@2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:����������@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������@:T P
,
_output_shapes
:����������@
 
_user_specified_nameinputs
��
�
F__inference_basemodel_layer_call_and_return_conditional_losses_1987761
inputs_0
inputs_1
inputs_2Q
;stream_2_conv_1_conv1d_expanddims_1_readvariableop_resource:@=
/stream_2_conv_1_biasadd_readvariableop_resource:@Q
;stream_1_conv_1_conv1d_expanddims_1_readvariableop_resource:@=
/stream_1_conv_1_biasadd_readvariableop_resource:@Q
;stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource:@=
/stream_0_conv_1_biasadd_readvariableop_resource:@E
7batch_normalization_2_batchnorm_readvariableop_resource:@I
;batch_normalization_2_batchnorm_mul_readvariableop_resource:@G
9batch_normalization_2_batchnorm_readvariableop_1_resource:@G
9batch_normalization_2_batchnorm_readvariableop_2_resource:@E
7batch_normalization_1_batchnorm_readvariableop_resource:@I
;batch_normalization_1_batchnorm_mul_readvariableop_resource:@G
9batch_normalization_1_batchnorm_readvariableop_1_resource:@G
9batch_normalization_1_batchnorm_readvariableop_2_resource:@C
5batch_normalization_batchnorm_readvariableop_resource:@G
9batch_normalization_batchnorm_mul_readvariableop_resource:@E
7batch_normalization_batchnorm_readvariableop_1_resource:@E
7batch_normalization_batchnorm_readvariableop_2_resource:@9
&dense_1_matmul_readvariableop_resource:	�T5
'dense_1_biasadd_readvariableop_resource:TE
7batch_normalization_3_batchnorm_readvariableop_resource:TI
;batch_normalization_3_batchnorm_mul_readvariableop_resource:TG
9batch_normalization_3_batchnorm_readvariableop_1_resource:TG
9batch_normalization_3_batchnorm_readvariableop_2_resource:T
identity��,batch_normalization/batchnorm/ReadVariableOp�.batch_normalization/batchnorm/ReadVariableOp_1�.batch_normalization/batchnorm/ReadVariableOp_2�0batch_normalization/batchnorm/mul/ReadVariableOp�.batch_normalization_1/batchnorm/ReadVariableOp�0batch_normalization_1/batchnorm/ReadVariableOp_1�0batch_normalization_1/batchnorm/ReadVariableOp_2�2batch_normalization_1/batchnorm/mul/ReadVariableOp�.batch_normalization_2/batchnorm/ReadVariableOp�0batch_normalization_2/batchnorm/ReadVariableOp_1�0batch_normalization_2/batchnorm/ReadVariableOp_2�2batch_normalization_2/batchnorm/mul/ReadVariableOp�.batch_normalization_3/batchnorm/ReadVariableOp�0batch_normalization_3/batchnorm/ReadVariableOp_1�0batch_normalization_3/batchnorm/ReadVariableOp_2�2batch_normalization_3/batchnorm/mul/ReadVariableOp�dense_1/BiasAdd/ReadVariableOp�dense_1/MatMul/ReadVariableOp�-dense_1/kernel/Regularizer/Abs/ReadVariableOp�&stream_0_conv_1/BiasAdd/ReadVariableOp�2stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp�5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp�&stream_1_conv_1/BiasAdd/ReadVariableOp�2stream_1_conv_1/conv1d/ExpandDims_1/ReadVariableOp�8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp�&stream_2_conv_1/BiasAdd/ReadVariableOp�2stream_2_conv_1/conv1d/ExpandDims_1/ReadVariableOp�5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp�
stream_2_input_drop/IdentityIdentityinputs_2*
T0*,
_output_shapes
:����������2
stream_2_input_drop/Identity�
stream_1_input_drop/IdentityIdentityinputs_1*
T0*,
_output_shapes
:����������2
stream_1_input_drop/Identity�
stream_0_input_drop/IdentityIdentityinputs_0*
T0*,
_output_shapes
:����������2
stream_0_input_drop/Identity�
%stream_2_conv_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2'
%stream_2_conv_1/conv1d/ExpandDims/dim�
!stream_2_conv_1/conv1d/ExpandDims
ExpandDims%stream_2_input_drop/Identity:output:0.stream_2_conv_1/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:����������2#
!stream_2_conv_1/conv1d/ExpandDims�
2stream_2_conv_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp;stream_2_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype024
2stream_2_conv_1/conv1d/ExpandDims_1/ReadVariableOp�
'stream_2_conv_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2)
'stream_2_conv_1/conv1d/ExpandDims_1/dim�
#stream_2_conv_1/conv1d/ExpandDims_1
ExpandDims:stream_2_conv_1/conv1d/ExpandDims_1/ReadVariableOp:value:00stream_2_conv_1/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@2%
#stream_2_conv_1/conv1d/ExpandDims_1�
stream_2_conv_1/conv1dConv2D*stream_2_conv_1/conv1d/ExpandDims:output:0,stream_2_conv_1/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:����������@*
paddingSAME*
strides
2
stream_2_conv_1/conv1d�
stream_2_conv_1/conv1d/SqueezeSqueezestream_2_conv_1/conv1d:output:0*
T0*,
_output_shapes
:����������@*
squeeze_dims

���������2 
stream_2_conv_1/conv1d/Squeeze�
&stream_2_conv_1/BiasAdd/ReadVariableOpReadVariableOp/stream_2_conv_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02(
&stream_2_conv_1/BiasAdd/ReadVariableOp�
stream_2_conv_1/BiasAddBiasAdd'stream_2_conv_1/conv1d/Squeeze:output:0.stream_2_conv_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������@2
stream_2_conv_1/BiasAdd�
%stream_1_conv_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2'
%stream_1_conv_1/conv1d/ExpandDims/dim�
!stream_1_conv_1/conv1d/ExpandDims
ExpandDims%stream_1_input_drop/Identity:output:0.stream_1_conv_1/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:����������2#
!stream_1_conv_1/conv1d/ExpandDims�
2stream_1_conv_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp;stream_1_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype024
2stream_1_conv_1/conv1d/ExpandDims_1/ReadVariableOp�
'stream_1_conv_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2)
'stream_1_conv_1/conv1d/ExpandDims_1/dim�
#stream_1_conv_1/conv1d/ExpandDims_1
ExpandDims:stream_1_conv_1/conv1d/ExpandDims_1/ReadVariableOp:value:00stream_1_conv_1/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@2%
#stream_1_conv_1/conv1d/ExpandDims_1�
stream_1_conv_1/conv1dConv2D*stream_1_conv_1/conv1d/ExpandDims:output:0,stream_1_conv_1/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:����������@*
paddingSAME*
strides
2
stream_1_conv_1/conv1d�
stream_1_conv_1/conv1d/SqueezeSqueezestream_1_conv_1/conv1d:output:0*
T0*,
_output_shapes
:����������@*
squeeze_dims

���������2 
stream_1_conv_1/conv1d/Squeeze�
&stream_1_conv_1/BiasAdd/ReadVariableOpReadVariableOp/stream_1_conv_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02(
&stream_1_conv_1/BiasAdd/ReadVariableOp�
stream_1_conv_1/BiasAddBiasAdd'stream_1_conv_1/conv1d/Squeeze:output:0.stream_1_conv_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������@2
stream_1_conv_1/BiasAdd�
%stream_0_conv_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2'
%stream_0_conv_1/conv1d/ExpandDims/dim�
!stream_0_conv_1/conv1d/ExpandDims
ExpandDims%stream_0_input_drop/Identity:output:0.stream_0_conv_1/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:����������2#
!stream_0_conv_1/conv1d/ExpandDims�
2stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp;stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype024
2stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp�
'stream_0_conv_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2)
'stream_0_conv_1/conv1d/ExpandDims_1/dim�
#stream_0_conv_1/conv1d/ExpandDims_1
ExpandDims:stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp:value:00stream_0_conv_1/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@2%
#stream_0_conv_1/conv1d/ExpandDims_1�
stream_0_conv_1/conv1dConv2D*stream_0_conv_1/conv1d/ExpandDims:output:0,stream_0_conv_1/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:����������@*
paddingSAME*
strides
2
stream_0_conv_1/conv1d�
stream_0_conv_1/conv1d/SqueezeSqueezestream_0_conv_1/conv1d:output:0*
T0*,
_output_shapes
:����������@*
squeeze_dims

���������2 
stream_0_conv_1/conv1d/Squeeze�
&stream_0_conv_1/BiasAdd/ReadVariableOpReadVariableOp/stream_0_conv_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02(
&stream_0_conv_1/BiasAdd/ReadVariableOp�
stream_0_conv_1/BiasAddBiasAdd'stream_0_conv_1/conv1d/Squeeze:output:0.stream_0_conv_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������@2
stream_0_conv_1/BiasAdd�
.batch_normalization_2/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_2_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype020
.batch_normalization_2/batchnorm/ReadVariableOp�
%batch_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2'
%batch_normalization_2/batchnorm/add/y�
#batch_normalization_2/batchnorm/addAddV26batch_normalization_2/batchnorm/ReadVariableOp:value:0.batch_normalization_2/batchnorm/add/y:output:0*
T0*
_output_shapes
:@2%
#batch_normalization_2/batchnorm/add�
%batch_normalization_2/batchnorm/RsqrtRsqrt'batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes
:@2'
%batch_normalization_2/batchnorm/Rsqrt�
2batch_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype024
2batch_normalization_2/batchnorm/mul/ReadVariableOp�
#batch_normalization_2/batchnorm/mulMul)batch_normalization_2/batchnorm/Rsqrt:y:0:batch_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2%
#batch_normalization_2/batchnorm/mul�
%batch_normalization_2/batchnorm/mul_1Mul stream_2_conv_1/BiasAdd:output:0'batch_normalization_2/batchnorm/mul:z:0*
T0*,
_output_shapes
:����������@2'
%batch_normalization_2/batchnorm/mul_1�
0batch_normalization_2/batchnorm/ReadVariableOp_1ReadVariableOp9batch_normalization_2_batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype022
0batch_normalization_2/batchnorm/ReadVariableOp_1�
%batch_normalization_2/batchnorm/mul_2Mul8batch_normalization_2/batchnorm/ReadVariableOp_1:value:0'batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes
:@2'
%batch_normalization_2/batchnorm/mul_2�
0batch_normalization_2/batchnorm/ReadVariableOp_2ReadVariableOp9batch_normalization_2_batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype022
0batch_normalization_2/batchnorm/ReadVariableOp_2�
#batch_normalization_2/batchnorm/subSub8batch_normalization_2/batchnorm/ReadVariableOp_2:value:0)batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2%
#batch_normalization_2/batchnorm/sub�
%batch_normalization_2/batchnorm/add_1AddV2)batch_normalization_2/batchnorm/mul_1:z:0'batch_normalization_2/batchnorm/sub:z:0*
T0*,
_output_shapes
:����������@2'
%batch_normalization_2/batchnorm/add_1�
.batch_normalization_1/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_1_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype020
.batch_normalization_1/batchnorm/ReadVariableOp�
%batch_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2'
%batch_normalization_1/batchnorm/add/y�
#batch_normalization_1/batchnorm/addAddV26batch_normalization_1/batchnorm/ReadVariableOp:value:0.batch_normalization_1/batchnorm/add/y:output:0*
T0*
_output_shapes
:@2%
#batch_normalization_1/batchnorm/add�
%batch_normalization_1/batchnorm/RsqrtRsqrt'batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes
:@2'
%batch_normalization_1/batchnorm/Rsqrt�
2batch_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype024
2batch_normalization_1/batchnorm/mul/ReadVariableOp�
#batch_normalization_1/batchnorm/mulMul)batch_normalization_1/batchnorm/Rsqrt:y:0:batch_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2%
#batch_normalization_1/batchnorm/mul�
%batch_normalization_1/batchnorm/mul_1Mul stream_1_conv_1/BiasAdd:output:0'batch_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:����������@2'
%batch_normalization_1/batchnorm/mul_1�
0batch_normalization_1/batchnorm/ReadVariableOp_1ReadVariableOp9batch_normalization_1_batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype022
0batch_normalization_1/batchnorm/ReadVariableOp_1�
%batch_normalization_1/batchnorm/mul_2Mul8batch_normalization_1/batchnorm/ReadVariableOp_1:value:0'batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes
:@2'
%batch_normalization_1/batchnorm/mul_2�
0batch_normalization_1/batchnorm/ReadVariableOp_2ReadVariableOp9batch_normalization_1_batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype022
0batch_normalization_1/batchnorm/ReadVariableOp_2�
#batch_normalization_1/batchnorm/subSub8batch_normalization_1/batchnorm/ReadVariableOp_2:value:0)batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2%
#batch_normalization_1/batchnorm/sub�
%batch_normalization_1/batchnorm/add_1AddV2)batch_normalization_1/batchnorm/mul_1:z:0'batch_normalization_1/batchnorm/sub:z:0*
T0*,
_output_shapes
:����������@2'
%batch_normalization_1/batchnorm/add_1�
,batch_normalization/batchnorm/ReadVariableOpReadVariableOp5batch_normalization_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02.
,batch_normalization/batchnorm/ReadVariableOp�
#batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2%
#batch_normalization/batchnorm/add/y�
!batch_normalization/batchnorm/addAddV24batch_normalization/batchnorm/ReadVariableOp:value:0,batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes
:@2#
!batch_normalization/batchnorm/add�
#batch_normalization/batchnorm/RsqrtRsqrt%batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes
:@2%
#batch_normalization/batchnorm/Rsqrt�
0batch_normalization/batchnorm/mul/ReadVariableOpReadVariableOp9batch_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype022
0batch_normalization/batchnorm/mul/ReadVariableOp�
!batch_normalization/batchnorm/mulMul'batch_normalization/batchnorm/Rsqrt:y:08batch_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2#
!batch_normalization/batchnorm/mul�
#batch_normalization/batchnorm/mul_1Mul stream_0_conv_1/BiasAdd:output:0%batch_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:����������@2%
#batch_normalization/batchnorm/mul_1�
.batch_normalization/batchnorm/ReadVariableOp_1ReadVariableOp7batch_normalization_batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype020
.batch_normalization/batchnorm/ReadVariableOp_1�
#batch_normalization/batchnorm/mul_2Mul6batch_normalization/batchnorm/ReadVariableOp_1:value:0%batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes
:@2%
#batch_normalization/batchnorm/mul_2�
.batch_normalization/batchnorm/ReadVariableOp_2ReadVariableOp7batch_normalization_batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype020
.batch_normalization/batchnorm/ReadVariableOp_2�
!batch_normalization/batchnorm/subSub6batch_normalization/batchnorm/ReadVariableOp_2:value:0'batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2#
!batch_normalization/batchnorm/sub�
#batch_normalization/batchnorm/add_1AddV2'batch_normalization/batchnorm/mul_1:z:0%batch_normalization/batchnorm/sub:z:0*
T0*,
_output_shapes
:����������@2%
#batch_normalization/batchnorm/add_1�
activation_2/ReluRelu)batch_normalization_2/batchnorm/add_1:z:0*
T0*,
_output_shapes
:����������@2
activation_2/Relu�
activation_1/ReluRelu)batch_normalization_1/batchnorm/add_1:z:0*
T0*,
_output_shapes
:����������@2
activation_1/Relu�
activation/ReluRelu'batch_normalization/batchnorm/add_1:z:0*
T0*,
_output_shapes
:����������@2
activation/Relu�
stream_2_drop_1/IdentityIdentityactivation_2/Relu:activations:0*
T0*,
_output_shapes
:����������@2
stream_2_drop_1/Identity�
stream_1_drop_1/IdentityIdentityactivation_1/Relu:activations:0*
T0*,
_output_shapes
:����������@2
stream_1_drop_1/Identity�
stream_0_drop_1/IdentityIdentityactivation/Relu:activations:0*
T0*,
_output_shapes
:����������@2
stream_0_drop_1/Identity�
/global_average_pooling1d/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :21
/global_average_pooling1d/Mean/reduction_indices�
global_average_pooling1d/MeanMean!stream_0_drop_1/Identity:output:08global_average_pooling1d/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:���������@2
global_average_pooling1d/Mean�
1global_average_pooling1d_1/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :23
1global_average_pooling1d_1/Mean/reduction_indices�
global_average_pooling1d_1/MeanMean!stream_1_drop_1/Identity:output:0:global_average_pooling1d_1/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:���������@2!
global_average_pooling1d_1/Mean�
1global_average_pooling1d_2/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :23
1global_average_pooling1d_2/Mean/reduction_indices�
global_average_pooling1d_2/MeanMean!stream_2_drop_1/Identity:output:0:global_average_pooling1d_2/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:���������@2!
global_average_pooling1d_2/Meant
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate/concat/axis�
concatenate/concatConcatV2&global_average_pooling1d/Mean:output:0(global_average_pooling1d_1/Mean:output:0(global_average_pooling1d_2/Mean:output:0 concatenate/concat/axis:output:0*
N*
T0*(
_output_shapes
:����������2
concatenate/concat�
dense_1_dropout/IdentityIdentityconcatenate/concat:output:0*
T0*(
_output_shapes
:����������2
dense_1_dropout/Identity�
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	�T*
dtype02
dense_1/MatMul/ReadVariableOp�
dense_1/MatMulMatMul!dense_1_dropout/Identity:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������T2
dense_1/MatMul�
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:T*
dtype02 
dense_1/BiasAdd/ReadVariableOp�
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������T2
dense_1/BiasAdd�
.batch_normalization_3/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_3_batchnorm_readvariableop_resource*
_output_shapes
:T*
dtype020
.batch_normalization_3/batchnorm/ReadVariableOp�
%batch_normalization_3/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2'
%batch_normalization_3/batchnorm/add/y�
#batch_normalization_3/batchnorm/addAddV26batch_normalization_3/batchnorm/ReadVariableOp:value:0.batch_normalization_3/batchnorm/add/y:output:0*
T0*
_output_shapes
:T2%
#batch_normalization_3/batchnorm/add�
%batch_normalization_3/batchnorm/RsqrtRsqrt'batch_normalization_3/batchnorm/add:z:0*
T0*
_output_shapes
:T2'
%batch_normalization_3/batchnorm/Rsqrt�
2batch_normalization_3/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_3_batchnorm_mul_readvariableop_resource*
_output_shapes
:T*
dtype024
2batch_normalization_3/batchnorm/mul/ReadVariableOp�
#batch_normalization_3/batchnorm/mulMul)batch_normalization_3/batchnorm/Rsqrt:y:0:batch_normalization_3/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:T2%
#batch_normalization_3/batchnorm/mul�
%batch_normalization_3/batchnorm/mul_1Muldense_1/BiasAdd:output:0'batch_normalization_3/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������T2'
%batch_normalization_3/batchnorm/mul_1�
0batch_normalization_3/batchnorm/ReadVariableOp_1ReadVariableOp9batch_normalization_3_batchnorm_readvariableop_1_resource*
_output_shapes
:T*
dtype022
0batch_normalization_3/batchnorm/ReadVariableOp_1�
%batch_normalization_3/batchnorm/mul_2Mul8batch_normalization_3/batchnorm/ReadVariableOp_1:value:0'batch_normalization_3/batchnorm/mul:z:0*
T0*
_output_shapes
:T2'
%batch_normalization_3/batchnorm/mul_2�
0batch_normalization_3/batchnorm/ReadVariableOp_2ReadVariableOp9batch_normalization_3_batchnorm_readvariableop_2_resource*
_output_shapes
:T*
dtype022
0batch_normalization_3/batchnorm/ReadVariableOp_2�
#batch_normalization_3/batchnorm/subSub8batch_normalization_3/batchnorm/ReadVariableOp_2:value:0)batch_normalization_3/batchnorm/mul_2:z:0*
T0*
_output_shapes
:T2%
#batch_normalization_3/batchnorm/sub�
%batch_normalization_3/batchnorm/add_1AddV2)batch_normalization_3/batchnorm/mul_1:z:0'batch_normalization_3/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������T2'
%batch_normalization_3/batchnorm/add_1�
dense_activation_1/SigmoidSigmoid)batch_normalization_3/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������T2
dense_activation_1/Sigmoid�
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp;stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype027
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp�
&stream_0_conv_1/kernel/Regularizer/AbsAbs=stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_0_conv_1/kernel/Regularizer/Abs�
(stream_0_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_0_conv_1/kernel/Regularizer/Const�
&stream_0_conv_1/kernel/Regularizer/SumSum*stream_0_conv_1/kernel/Regularizer/Abs:y:01stream_0_conv_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/Sum�
(stream_0_conv_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2*
(stream_0_conv_1/kernel/Regularizer/mul/x�
&stream_0_conv_1/kernel/Regularizer/mulMul1stream_0_conv_1/kernel/Regularizer/mul/x:output:0/stream_0_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/mul�
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;stream_1_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02:
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp�
)stream_1_conv_1/kernel/Regularizer/SquareSquare@stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2+
)stream_1_conv_1/kernel/Regularizer/Square�
(stream_1_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_1_conv_1/kernel/Regularizer/Const�
&stream_1_conv_1/kernel/Regularizer/SumSum-stream_1_conv_1/kernel/Regularizer/Square:y:01stream_1_conv_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&stream_1_conv_1/kernel/Regularizer/Sum�
(stream_1_conv_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2*
(stream_1_conv_1/kernel/Regularizer/mul/x�
&stream_1_conv_1/kernel/Regularizer/mulMul1stream_1_conv_1/kernel/Regularizer/mul/x:output:0/stream_1_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_1_conv_1/kernel/Regularizer/mul�
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp;stream_2_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype027
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp�
&stream_2_conv_1/kernel/Regularizer/AbsAbs=stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_2_conv_1/kernel/Regularizer/Abs�
(stream_2_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_2_conv_1/kernel/Regularizer/Const�
&stream_2_conv_1/kernel/Regularizer/SumSum*stream_2_conv_1/kernel/Regularizer/Abs:y:01stream_2_conv_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&stream_2_conv_1/kernel/Regularizer/Sum�
(stream_2_conv_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2*
(stream_2_conv_1/kernel/Regularizer/mul/x�
&stream_2_conv_1/kernel/Regularizer/mulMul1stream_2_conv_1/kernel/Regularizer/mul/x:output:0/stream_2_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_2_conv_1/kernel/Regularizer/mul�
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	�T*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOp�
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	�T2 
dense_1/kernel/Regularizer/Abs�
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_1/kernel/Regularizer/Const�
dense_1/kernel/Regularizer/SumSum"dense_1/kernel/Regularizer/Abs:y:0)dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/Sum�
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2"
 dense_1/kernel/Regularizer/mul/x�
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/muly
IdentityIdentitydense_activation_1/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:���������T2

Identity�
NoOpNoOp-^batch_normalization/batchnorm/ReadVariableOp/^batch_normalization/batchnorm/ReadVariableOp_1/^batch_normalization/batchnorm/ReadVariableOp_21^batch_normalization/batchnorm/mul/ReadVariableOp/^batch_normalization_1/batchnorm/ReadVariableOp1^batch_normalization_1/batchnorm/ReadVariableOp_11^batch_normalization_1/batchnorm/ReadVariableOp_23^batch_normalization_1/batchnorm/mul/ReadVariableOp/^batch_normalization_2/batchnorm/ReadVariableOp1^batch_normalization_2/batchnorm/ReadVariableOp_11^batch_normalization_2/batchnorm/ReadVariableOp_23^batch_normalization_2/batchnorm/mul/ReadVariableOp/^batch_normalization_3/batchnorm/ReadVariableOp1^batch_normalization_3/batchnorm/ReadVariableOp_11^batch_normalization_3/batchnorm/ReadVariableOp_23^batch_normalization_3/batchnorm/mul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp.^dense_1/kernel/Regularizer/Abs/ReadVariableOp'^stream_0_conv_1/BiasAdd/ReadVariableOp3^stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp'^stream_1_conv_1/BiasAdd/ReadVariableOp3^stream_1_conv_1/conv1d/ExpandDims_1/ReadVariableOp9^stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp'^stream_2_conv_1/BiasAdd/ReadVariableOp3^stream_2_conv_1/conv1d/ExpandDims_1/ReadVariableOp6^stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapesz
x:����������:����������:����������: : : : : : : : : : : : : : : : : : : : : : : : 2\
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
&stream_1_conv_1/BiasAdd/ReadVariableOp&stream_1_conv_1/BiasAdd/ReadVariableOp2h
2stream_1_conv_1/conv1d/ExpandDims_1/ReadVariableOp2stream_1_conv_1/conv1d/ExpandDims_1/ReadVariableOp2t
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp2P
&stream_2_conv_1/BiasAdd/ReadVariableOp&stream_2_conv_1/BiasAdd/ReadVariableOp2h
2stream_2_conv_1/conv1d/ExpandDims_1/ReadVariableOp2stream_2_conv_1/conv1d/ExpandDims_1/ReadVariableOp2n
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp:V R
,
_output_shapes
:����������
"
_user_specified_name
inputs/0:VR
,
_output_shapes
:����������
"
_user_specified_name
inputs/1:VR
,
_output_shapes
:����������
"
_user_specified_name
inputs/2
�
e
I__inference_activation_1_layer_call_and_return_conditional_losses_1988735

inputs
identityS
ReluReluinputs*
T0*,
_output_shapes
:����������@2
Reluk
IdentityIdentityRelu:activations:0*
T0*,
_output_shapes
:����������@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������@:T P
,
_output_shapes
:����������@
 
_user_specified_nameinputs
�
k
L__inference_stream_2_drop_1_layer_call_and_return_conditional_losses_1985135

inputs
identity�c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
dropout/Constx
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:����������@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:����������@*
dtype0*
seed�*
seed2�2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>2
dropout/GreaterEqual/y�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:����������@2
dropout/GreaterEqual�
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:����������@2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:����������@2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:����������@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������@:T P
,
_output_shapes
:����������@
 
_user_specified_nameinputs
�	
�
5__inference_batch_normalization_layer_call_fn_1988248

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_19839252
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :������������������@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:������������������@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :������������������@
 
_user_specified_nameinputs
�
�
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1989003

inputs/
!batchnorm_readvariableop_resource:T3
%batchnorm_mul_readvariableop_resource:T1
#batchnorm_readvariableop_1_resource:T1
#batchnorm_readvariableop_2_resource:T
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOp�
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
 *o�:2
batchnorm/add/y�
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:T2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:T2
batchnorm/Rsqrt�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:T*
dtype02
batchnorm/mul/ReadVariableOp�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:T2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������T2
batchnorm/mul_1�
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:T*
dtype02
batchnorm/ReadVariableOp_1�
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:T2
batchnorm/mul_2�
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:T*
dtype02
batchnorm/ReadVariableOp_2�
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:T2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������T2
batchnorm/add_1n
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������T2

Identity�
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������T: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������T
 
_user_specified_nameinputs
�I
�	
B__inference_model_layer_call_and_return_conditional_losses_1986262

inputs
inputs_1'
basemodel_1986162:@
basemodel_1986164:@'
basemodel_1986166:@
basemodel_1986168:@'
basemodel_1986170:@
basemodel_1986172:@
basemodel_1986174:@
basemodel_1986176:@
basemodel_1986178:@
basemodel_1986180:@
basemodel_1986182:@
basemodel_1986184:@
basemodel_1986186:@
basemodel_1986188:@
basemodel_1986190:@
basemodel_1986192:@
basemodel_1986194:@
basemodel_1986196:@$
basemodel_1986198:	�T
basemodel_1986200:T
basemodel_1986202:T
basemodel_1986204:T
basemodel_1986206:T
basemodel_1986208:T
identity��!basemodel/StatefulPartitionedCall�#basemodel/StatefulPartitionedCall_1�-dense_1/kernel/Regularizer/Abs/ReadVariableOp�5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp�8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp�5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp�
!basemodel/StatefulPartitionedCallStatefulPartitionedCallinputsinputsinputsbasemodel_1986162basemodel_1986164basemodel_1986166basemodel_1986168basemodel_1986170basemodel_1986172basemodel_1986174basemodel_1986176basemodel_1986178basemodel_1986180basemodel_1986182basemodel_1986184basemodel_1986186basemodel_1986188basemodel_1986190basemodel_1986192basemodel_1986194basemodel_1986196basemodel_1986198basemodel_1986200basemodel_1986202basemodel_1986204basemodel_1986206basemodel_1986208*&
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������T*2
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_basemodel_layer_call_and_return_conditional_losses_19855952#
!basemodel/StatefulPartitionedCall�
#basemodel/StatefulPartitionedCall_1StatefulPartitionedCallinputs_1inputs_1inputs_1basemodel_1986162basemodel_1986164basemodel_1986166basemodel_1986168basemodel_1986170basemodel_1986172basemodel_1986174basemodel_1986176basemodel_1986178basemodel_1986180basemodel_1986182basemodel_1986184basemodel_1986186basemodel_1986188basemodel_1986190basemodel_1986192basemodel_1986194basemodel_1986196basemodel_1986198basemodel_1986200basemodel_1986202basemodel_1986204basemodel_1986206basemodel_1986208"^basemodel/StatefulPartitionedCall*&
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������T*2
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_basemodel_layer_call_and_return_conditional_losses_19855952%
#basemodel/StatefulPartitionedCall_1�
distance/PartitionedCallPartitionedCall*basemodel/StatefulPartitionedCall:output:0,basemodel/StatefulPartitionedCall_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_distance_layer_call_and_return_conditional_losses_19860982
distance/PartitionedCall�
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpbasemodel_1986170*"
_output_shapes
:@*
dtype027
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp�
&stream_0_conv_1/kernel/Regularizer/AbsAbs=stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_0_conv_1/kernel/Regularizer/Abs�
(stream_0_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_0_conv_1/kernel/Regularizer/Const�
&stream_0_conv_1/kernel/Regularizer/SumSum*stream_0_conv_1/kernel/Regularizer/Abs:y:01stream_0_conv_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/Sum�
(stream_0_conv_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2*
(stream_0_conv_1/kernel/Regularizer/mul/x�
&stream_0_conv_1/kernel/Regularizer/mulMul1stream_0_conv_1/kernel/Regularizer/mul/x:output:0/stream_0_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/mul�
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpbasemodel_1986166*"
_output_shapes
:@*
dtype02:
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp�
)stream_1_conv_1/kernel/Regularizer/SquareSquare@stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2+
)stream_1_conv_1/kernel/Regularizer/Square�
(stream_1_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_1_conv_1/kernel/Regularizer/Const�
&stream_1_conv_1/kernel/Regularizer/SumSum-stream_1_conv_1/kernel/Regularizer/Square:y:01stream_1_conv_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&stream_1_conv_1/kernel/Regularizer/Sum�
(stream_1_conv_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2*
(stream_1_conv_1/kernel/Regularizer/mul/x�
&stream_1_conv_1/kernel/Regularizer/mulMul1stream_1_conv_1/kernel/Regularizer/mul/x:output:0/stream_1_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_1_conv_1/kernel/Regularizer/mul�
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpbasemodel_1986162*"
_output_shapes
:@*
dtype027
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp�
&stream_2_conv_1/kernel/Regularizer/AbsAbs=stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_2_conv_1/kernel/Regularizer/Abs�
(stream_2_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_2_conv_1/kernel/Regularizer/Const�
&stream_2_conv_1/kernel/Regularizer/SumSum*stream_2_conv_1/kernel/Regularizer/Abs:y:01stream_2_conv_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&stream_2_conv_1/kernel/Regularizer/Sum�
(stream_2_conv_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2*
(stream_2_conv_1/kernel/Regularizer/mul/x�
&stream_2_conv_1/kernel/Regularizer/mulMul1stream_2_conv_1/kernel/Regularizer/mul/x:output:0/stream_2_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_2_conv_1/kernel/Regularizer/mul�
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpbasemodel_1986198*
_output_shapes
:	�T*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOp�
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	�T2 
dense_1/kernel/Regularizer/Abs�
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_1/kernel/Regularizer/Const�
dense_1/kernel/Regularizer/SumSum"dense_1/kernel/Regularizer/Abs:y:0)dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/Sum�
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2"
 dense_1/kernel/Regularizer/mul/x�
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/mul|
IdentityIdentity!distance/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp"^basemodel/StatefulPartitionedCall$^basemodel/StatefulPartitionedCall_1.^dense_1/kernel/Regularizer/Abs/ReadVariableOp6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp9^stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp6^stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*s
_input_shapesb
`:����������:����������: : : : : : : : : : : : : : : : : : : : : : : : 2F
!basemodel/StatefulPartitionedCall!basemodel/StatefulPartitionedCall2J
#basemodel/StatefulPartitionedCall_1#basemodel/StatefulPartitionedCall_12^
-dense_1/kernel/Regularizer/Abs/ReadVariableOp-dense_1/kernel/Regularizer/Abs/ReadVariableOp2n
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp2t
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp2n
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs:TP
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
c
G__inference_activation_layer_call_and_return_conditional_losses_1988725

inputs
identityS
ReluReluinputs*
T0*,
_output_shapes
:����������@2
Reluk
IdentityIdentityRelu:activations:0*
T0*,
_output_shapes
:����������@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������@:T P
,
_output_shapes
:����������@
 
_user_specified_nameinputs
�
�
L__inference_stream_0_conv_1_layer_call_and_return_conditional_losses_1988163

inputsA
+conv1d_expanddims_1_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�"conv1d/ExpandDims_1/ReadVariableOp�5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
conv1d/ExpandDims/dim�
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:����������2
conv1d/ExpandDims�
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
conv1d/ExpandDims_1/dim�
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@2
conv1d/ExpandDims_1�
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:����������@*
paddingSAME*
strides
2
conv1d�
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:����������@*
squeeze_dims

���������2
conv1d/Squeeze�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������@2	
BiasAdd�
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype027
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp�
&stream_0_conv_1/kernel/Regularizer/AbsAbs=stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_0_conv_1/kernel/Regularizer/Abs�
(stream_0_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_0_conv_1/kernel/Regularizer/Const�
&stream_0_conv_1/kernel/Regularizer/SumSum*stream_0_conv_1/kernel/Regularizer/Abs:y:01stream_0_conv_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/Sum�
(stream_0_conv_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2*
(stream_0_conv_1/kernel/Regularizer/mul/x�
&stream_0_conv_1/kernel/Regularizer/mulMul1stream_0_conv_1/kernel/Regularizer/mul/x:output:0/stream_0_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/mulp
IdentityIdentityBiasAdd:output:0^NoOp*
T0*,
_output_shapes
:����������@2

Identity�
NoOpNoOp^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp2n
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
s
W__inference_global_average_pooling1d_2_layer_call_and_return_conditional_losses_1988886

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
:������������������2
Meanj
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:������������������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'���������������������������:e a
=
_output_shapes+
):'���������������������������
 
_user_specified_nameinputs
�
X
<__inference_global_average_pooling1d_2_layer_call_fn_1988875

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:������������������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *`
f[RY
W__inference_global_average_pooling1d_2_layer_call_and_return_conditional_losses_19844452
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:������������������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'���������������������������:e a
=
_output_shapes+
):'���������������������������
 
_user_specified_nameinputs
�+
�
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1984309

inputs5
'assignmovingavg_readvariableop_resource:@7
)assignmovingavg_1_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@/
!batchnorm_readvariableop_resource:@
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOp�
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indices�
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(2
moments/mean�
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:@2
moments/StopGradient�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :������������������@2
moments/SquaredDifference�
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices�
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(2
moments/variance�
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 2
moments/Squeeze�
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
�#<2
AssignMovingAvg/decay�
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg/sub�
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg/mul�
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
�#<2
AssignMovingAvg_1/decay�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg_1/sub�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg_1/mul�
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
 *o�:2
batchnorm/add/y�
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@2
batchnorm/Rsqrt�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOp�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2
batchnorm/mul�
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :������������������@2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:@2
batchnorm/mul_2�
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp�
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :������������������@2
batchnorm/add_1{
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :������������������@2

Identity�
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:������������������@: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :������������������@
 
_user_specified_nameinputs
�
n
5__inference_stream_2_input_drop_layer_call_fn_1988110

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *Y
fTRR
P__inference_stream_2_input_drop_layer_call_and_return_conditional_losses_19854322
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:����������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1988521

inputs/
!batchnorm_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@1
#batchnorm_readvariableop_1_resource:@1
#batchnorm_readvariableop_2_resource:@
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOp�
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
 *o�:2
batchnorm/add/y�
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@2
batchnorm/Rsqrt�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOp�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2
batchnorm/mul{
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*,
_output_shapes
:����������@2
batchnorm/mul_1�
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp_1�
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:@2
batchnorm/mul_2�
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp_2�
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*,
_output_shapes
:����������@2
batchnorm/add_1s
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*,
_output_shapes
:����������@2

Identity�
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������@: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:T P
,
_output_shapes
:����������@
 
_user_specified_nameinputs
�
M
1__inference_stream_0_drop_1_layer_call_fn_1988750

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_stream_0_drop_1_layer_call_and_return_conditional_losses_19848612
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:����������@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������@:T P
,
_output_shapes
:����������@
 
_user_specified_nameinputs
�J
�	
B__inference_model_layer_call_and_return_conditional_losses_1986471
left_inputs
right_inputs'
basemodel_1986371:@
basemodel_1986373:@'
basemodel_1986375:@
basemodel_1986377:@'
basemodel_1986379:@
basemodel_1986381:@
basemodel_1986383:@
basemodel_1986385:@
basemodel_1986387:@
basemodel_1986389:@
basemodel_1986391:@
basemodel_1986393:@
basemodel_1986395:@
basemodel_1986397:@
basemodel_1986399:@
basemodel_1986401:@
basemodel_1986403:@
basemodel_1986405:@$
basemodel_1986407:	�T
basemodel_1986409:T
basemodel_1986411:T
basemodel_1986413:T
basemodel_1986415:T
basemodel_1986417:T
identity��!basemodel/StatefulPartitionedCall�#basemodel/StatefulPartitionedCall_1�-dense_1/kernel/Regularizer/Abs/ReadVariableOp�5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp�8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp�5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp�
!basemodel/StatefulPartitionedCallStatefulPartitionedCallleft_inputsleft_inputsleft_inputsbasemodel_1986371basemodel_1986373basemodel_1986375basemodel_1986377basemodel_1986379basemodel_1986381basemodel_1986383basemodel_1986385basemodel_1986387basemodel_1986389basemodel_1986391basemodel_1986393basemodel_1986395basemodel_1986397basemodel_1986399basemodel_1986401basemodel_1986403basemodel_1986405basemodel_1986407basemodel_1986409basemodel_1986411basemodel_1986413basemodel_1986415basemodel_1986417*&
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������T*:
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_basemodel_layer_call_and_return_conditional_losses_19849642#
!basemodel/StatefulPartitionedCall�
#basemodel/StatefulPartitionedCall_1StatefulPartitionedCallright_inputsright_inputsright_inputsbasemodel_1986371basemodel_1986373basemodel_1986375basemodel_1986377basemodel_1986379basemodel_1986381basemodel_1986383basemodel_1986385basemodel_1986387basemodel_1986389basemodel_1986391basemodel_1986393basemodel_1986395basemodel_1986397basemodel_1986399basemodel_1986401basemodel_1986403basemodel_1986405basemodel_1986407basemodel_1986409basemodel_1986411basemodel_1986413basemodel_1986415basemodel_1986417*&
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������T*:
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_basemodel_layer_call_and_return_conditional_losses_19849642%
#basemodel/StatefulPartitionedCall_1�
distance/PartitionedCallPartitionedCall*basemodel/StatefulPartitionedCall:output:0,basemodel/StatefulPartitionedCall_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_distance_layer_call_and_return_conditional_losses_19859982
distance/PartitionedCall�
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpbasemodel_1986379*"
_output_shapes
:@*
dtype027
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp�
&stream_0_conv_1/kernel/Regularizer/AbsAbs=stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_0_conv_1/kernel/Regularizer/Abs�
(stream_0_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_0_conv_1/kernel/Regularizer/Const�
&stream_0_conv_1/kernel/Regularizer/SumSum*stream_0_conv_1/kernel/Regularizer/Abs:y:01stream_0_conv_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/Sum�
(stream_0_conv_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2*
(stream_0_conv_1/kernel/Regularizer/mul/x�
&stream_0_conv_1/kernel/Regularizer/mulMul1stream_0_conv_1/kernel/Regularizer/mul/x:output:0/stream_0_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/mul�
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpbasemodel_1986375*"
_output_shapes
:@*
dtype02:
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp�
)stream_1_conv_1/kernel/Regularizer/SquareSquare@stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2+
)stream_1_conv_1/kernel/Regularizer/Square�
(stream_1_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_1_conv_1/kernel/Regularizer/Const�
&stream_1_conv_1/kernel/Regularizer/SumSum-stream_1_conv_1/kernel/Regularizer/Square:y:01stream_1_conv_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&stream_1_conv_1/kernel/Regularizer/Sum�
(stream_1_conv_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2*
(stream_1_conv_1/kernel/Regularizer/mul/x�
&stream_1_conv_1/kernel/Regularizer/mulMul1stream_1_conv_1/kernel/Regularizer/mul/x:output:0/stream_1_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_1_conv_1/kernel/Regularizer/mul�
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpbasemodel_1986371*"
_output_shapes
:@*
dtype027
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp�
&stream_2_conv_1/kernel/Regularizer/AbsAbs=stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_2_conv_1/kernel/Regularizer/Abs�
(stream_2_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_2_conv_1/kernel/Regularizer/Const�
&stream_2_conv_1/kernel/Regularizer/SumSum*stream_2_conv_1/kernel/Regularizer/Abs:y:01stream_2_conv_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&stream_2_conv_1/kernel/Regularizer/Sum�
(stream_2_conv_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2*
(stream_2_conv_1/kernel/Regularizer/mul/x�
&stream_2_conv_1/kernel/Regularizer/mulMul1stream_2_conv_1/kernel/Regularizer/mul/x:output:0/stream_2_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_2_conv_1/kernel/Regularizer/mul�
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpbasemodel_1986407*
_output_shapes
:	�T*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOp�
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	�T2 
dense_1/kernel/Regularizer/Abs�
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_1/kernel/Regularizer/Const�
dense_1/kernel/Regularizer/SumSum"dense_1/kernel/Regularizer/Abs:y:0)dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/Sum�
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2"
 dense_1/kernel/Regularizer/mul/x�
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/mul|
IdentityIdentity!distance/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp"^basemodel/StatefulPartitionedCall$^basemodel/StatefulPartitionedCall_1.^dense_1/kernel/Regularizer/Abs/ReadVariableOp6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp9^stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp6^stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*s
_input_shapesb
`:����������:����������: : : : : : : : : : : : : : : : : : : : : : : : 2F
!basemodel/StatefulPartitionedCall!basemodel/StatefulPartitionedCall2J
#basemodel/StatefulPartitionedCall_1#basemodel/StatefulPartitionedCall_12^
-dense_1/kernel/Regularizer/Abs/ReadVariableOp-dense_1/kernel/Regularizer/Abs/ReadVariableOp2n
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp2t
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp2n
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp:Y U
,
_output_shapes
:����������
%
_user_specified_nameleft_inputs:ZV
,
_output_shapes
:����������
&
_user_specified_nameright_inputs
�
�
__inference_loss_fn_1_1989069W
Astream_1_conv_1_kernel_regularizer_square_readvariableop_resource:@
identity��8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp�
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpAstream_1_conv_1_kernel_regularizer_square_readvariableop_resource*"
_output_shapes
:@*
dtype02:
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp�
)stream_1_conv_1/kernel/Regularizer/SquareSquare@stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2+
)stream_1_conv_1/kernel/Regularizer/Square�
(stream_1_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_1_conv_1/kernel/Regularizer/Const�
&stream_1_conv_1/kernel/Regularizer/SumSum-stream_1_conv_1/kernel/Regularizer/Square:y:01stream_1_conv_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&stream_1_conv_1/kernel/Regularizer/Sum�
(stream_1_conv_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2*
(stream_1_conv_1/kernel/Regularizer/mul/x�
&stream_1_conv_1/kernel/Regularizer/mulMul1stream_1_conv_1/kernel/Regularizer/mul/x:output:0/stream_1_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_1_conv_1/kernel/Regularizer/mult
IdentityIdentity*stream_1_conv_1/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: 2

Identity�
NoOpNoOp9^stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2t
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp
�
M
1__inference_stream_2_drop_1_layer_call_fn_1988804

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_stream_2_drop_1_layer_call_and_return_conditional_losses_19848472
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:����������@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������@:T P
,
_output_shapes
:����������@
 
_user_specified_nameinputs
�	
�
7__inference_batch_normalization_1_layer_call_fn_1988408

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *[
fVRT
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_19840872
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :������������������@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:������������������@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :������������������@
 
_user_specified_nameinputs
�
j
L__inference_stream_1_drop_1_layer_call_and_return_conditional_losses_1988787

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:����������@2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:����������@2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������@:T P
,
_output_shapes
:����������@
 
_user_specified_nameinputs
�+
�
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1985325

inputs5
'assignmovingavg_readvariableop_resource:@7
)assignmovingavg_1_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@/
!batchnorm_readvariableop_resource:@
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOp�
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indices�
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(2
moments/mean�
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:@2
moments/StopGradient�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*,
_output_shapes
:����������@2
moments/SquaredDifference�
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices�
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(2
moments/variance�
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 2
moments/Squeeze�
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
�#<2
AssignMovingAvg/decay�
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg/sub�
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg/mul�
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
�#<2
AssignMovingAvg_1/decay�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg_1/sub�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg_1/mul�
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
 *o�:2
batchnorm/add/y�
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@2
batchnorm/Rsqrt�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOp�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2
batchnorm/mul{
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*,
_output_shapes
:����������@2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:@2
batchnorm/mul_2�
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp�
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*,
_output_shapes
:����������@2
batchnorm/add_1s
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*,
_output_shapes
:����������@2

Identity�
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������@: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:T P
,
_output_shapes
:����������@
 
_user_specified_nameinputs
�
k
L__inference_stream_0_drop_1_layer_call_and_return_conditional_losses_1985089

inputs
identity�c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
dropout/Constx
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:����������@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:����������@*
dtype0*
seed�*
seed2�2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>2
dropout/GreaterEqual/y�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:����������@2
dropout/GreaterEqual�
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:����������@2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:����������@2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:����������@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������@:T P
,
_output_shapes
:����������@
 
_user_specified_nameinputs
�
�*
#__inference__traced_restore_1989503
file_prefix!
assignvariableop_beta_1: #
assignvariableop_1_beta_2: "
assignvariableop_2_decay: *
 assignvariableop_3_learning_rate: &
assignvariableop_4_adam_iter:	 ?
)assignvariableop_5_stream_0_conv_1_kernel:@5
'assignvariableop_6_stream_0_conv_1_bias:@?
)assignvariableop_7_stream_1_conv_1_kernel:@5
'assignvariableop_8_stream_1_conv_1_bias:@?
)assignvariableop_9_stream_2_conv_1_kernel:@6
(assignvariableop_10_stream_2_conv_1_bias:@;
-assignvariableop_11_batch_normalization_gamma:@:
,assignvariableop_12_batch_normalization_beta:@=
/assignvariableop_13_batch_normalization_1_gamma:@<
.assignvariableop_14_batch_normalization_1_beta:@=
/assignvariableop_15_batch_normalization_2_gamma:@<
.assignvariableop_16_batch_normalization_2_beta:@5
"assignvariableop_17_dense_1_kernel:	�T.
 assignvariableop_18_dense_1_bias:T=
/assignvariableop_19_batch_normalization_3_gamma:T<
.assignvariableop_20_batch_normalization_3_beta:TA
3assignvariableop_21_batch_normalization_moving_mean:@E
7assignvariableop_22_batch_normalization_moving_variance:@C
5assignvariableop_23_batch_normalization_1_moving_mean:@G
9assignvariableop_24_batch_normalization_1_moving_variance:@C
5assignvariableop_25_batch_normalization_2_moving_mean:@G
9assignvariableop_26_batch_normalization_2_moving_variance:@C
5assignvariableop_27_batch_normalization_3_moving_mean:TG
9assignvariableop_28_batch_normalization_3_moving_variance:T#
assignvariableop_29_total: #
assignvariableop_30_count: G
1assignvariableop_31_adam_stream_0_conv_1_kernel_m:@=
/assignvariableop_32_adam_stream_0_conv_1_bias_m:@G
1assignvariableop_33_adam_stream_1_conv_1_kernel_m:@=
/assignvariableop_34_adam_stream_1_conv_1_bias_m:@G
1assignvariableop_35_adam_stream_2_conv_1_kernel_m:@=
/assignvariableop_36_adam_stream_2_conv_1_bias_m:@B
4assignvariableop_37_adam_batch_normalization_gamma_m:@A
3assignvariableop_38_adam_batch_normalization_beta_m:@D
6assignvariableop_39_adam_batch_normalization_1_gamma_m:@C
5assignvariableop_40_adam_batch_normalization_1_beta_m:@D
6assignvariableop_41_adam_batch_normalization_2_gamma_m:@C
5assignvariableop_42_adam_batch_normalization_2_beta_m:@<
)assignvariableop_43_adam_dense_1_kernel_m:	�T5
'assignvariableop_44_adam_dense_1_bias_m:TD
6assignvariableop_45_adam_batch_normalization_3_gamma_m:TC
5assignvariableop_46_adam_batch_normalization_3_beta_m:TG
1assignvariableop_47_adam_stream_0_conv_1_kernel_v:@=
/assignvariableop_48_adam_stream_0_conv_1_bias_v:@G
1assignvariableop_49_adam_stream_1_conv_1_kernel_v:@=
/assignvariableop_50_adam_stream_1_conv_1_bias_v:@G
1assignvariableop_51_adam_stream_2_conv_1_kernel_v:@=
/assignvariableop_52_adam_stream_2_conv_1_bias_v:@B
4assignvariableop_53_adam_batch_normalization_gamma_v:@A
3assignvariableop_54_adam_batch_normalization_beta_v:@D
6assignvariableop_55_adam_batch_normalization_1_gamma_v:@C
5assignvariableop_56_adam_batch_normalization_1_beta_v:@D
6assignvariableop_57_adam_batch_normalization_2_gamma_v:@C
5assignvariableop_58_adam_batch_normalization_2_beta_v:@<
)assignvariableop_59_adam_dense_1_kernel_v:	�T5
'assignvariableop_60_adam_dense_1_bias_v:TD
6assignvariableop_61_adam_batch_normalization_3_gamma_v:TC
5assignvariableop_62_adam_batch_normalization_3_beta_v:T
identity_64��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_34�AssignVariableOp_35�AssignVariableOp_36�AssignVariableOp_37�AssignVariableOp_38�AssignVariableOp_39�AssignVariableOp_4�AssignVariableOp_40�AssignVariableOp_41�AssignVariableOp_42�AssignVariableOp_43�AssignVariableOp_44�AssignVariableOp_45�AssignVariableOp_46�AssignVariableOp_47�AssignVariableOp_48�AssignVariableOp_49�AssignVariableOp_5�AssignVariableOp_50�AssignVariableOp_51�AssignVariableOp_52�AssignVariableOp_53�AssignVariableOp_54�AssignVariableOp_55�AssignVariableOp_56�AssignVariableOp_57�AssignVariableOp_58�AssignVariableOp_59�AssignVariableOp_6�AssignVariableOp_60�AssignVariableOp_61�AssignVariableOp_62�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9� 
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:@*
dtype0*�
value�B�@B+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/14/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/15/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:@*
dtype0*�
value�B�@B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices�
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*N
dtypesD
B2@	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity�
AssignVariableOpAssignVariableOpassignvariableop_beta_1Identity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1�
AssignVariableOp_1AssignVariableOpassignvariableop_1_beta_2Identity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2�
AssignVariableOp_2AssignVariableOpassignvariableop_2_decayIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3�
AssignVariableOp_3AssignVariableOp assignvariableop_3_learning_rateIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_4�
AssignVariableOp_4AssignVariableOpassignvariableop_4_adam_iterIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5�
AssignVariableOp_5AssignVariableOp)assignvariableop_5_stream_0_conv_1_kernelIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6�
AssignVariableOp_6AssignVariableOp'assignvariableop_6_stream_0_conv_1_biasIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7�
AssignVariableOp_7AssignVariableOp)assignvariableop_7_stream_1_conv_1_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8�
AssignVariableOp_8AssignVariableOp'assignvariableop_8_stream_1_conv_1_biasIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9�
AssignVariableOp_9AssignVariableOp)assignvariableop_9_stream_2_conv_1_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10�
AssignVariableOp_10AssignVariableOp(assignvariableop_10_stream_2_conv_1_biasIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11�
AssignVariableOp_11AssignVariableOp-assignvariableop_11_batch_normalization_gammaIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12�
AssignVariableOp_12AssignVariableOp,assignvariableop_12_batch_normalization_betaIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13�
AssignVariableOp_13AssignVariableOp/assignvariableop_13_batch_normalization_1_gammaIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14�
AssignVariableOp_14AssignVariableOp.assignvariableop_14_batch_normalization_1_betaIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15�
AssignVariableOp_15AssignVariableOp/assignvariableop_15_batch_normalization_2_gammaIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16�
AssignVariableOp_16AssignVariableOp.assignvariableop_16_batch_normalization_2_betaIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17�
AssignVariableOp_17AssignVariableOp"assignvariableop_17_dense_1_kernelIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18�
AssignVariableOp_18AssignVariableOp assignvariableop_18_dense_1_biasIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19�
AssignVariableOp_19AssignVariableOp/assignvariableop_19_batch_normalization_3_gammaIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20�
AssignVariableOp_20AssignVariableOp.assignvariableop_20_batch_normalization_3_betaIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21�
AssignVariableOp_21AssignVariableOp3assignvariableop_21_batch_normalization_moving_meanIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22�
AssignVariableOp_22AssignVariableOp7assignvariableop_22_batch_normalization_moving_varianceIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23�
AssignVariableOp_23AssignVariableOp5assignvariableop_23_batch_normalization_1_moving_meanIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24�
AssignVariableOp_24AssignVariableOp9assignvariableop_24_batch_normalization_1_moving_varianceIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25�
AssignVariableOp_25AssignVariableOp5assignvariableop_25_batch_normalization_2_moving_meanIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26�
AssignVariableOp_26AssignVariableOp9assignvariableop_26_batch_normalization_2_moving_varianceIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27�
AssignVariableOp_27AssignVariableOp5assignvariableop_27_batch_normalization_3_moving_meanIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28�
AssignVariableOp_28AssignVariableOp9assignvariableop_28_batch_normalization_3_moving_varianceIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29�
AssignVariableOp_29AssignVariableOpassignvariableop_29_totalIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30�
AssignVariableOp_30AssignVariableOpassignvariableop_30_countIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31�
AssignVariableOp_31AssignVariableOp1assignvariableop_31_adam_stream_0_conv_1_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32�
AssignVariableOp_32AssignVariableOp/assignvariableop_32_adam_stream_0_conv_1_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33�
AssignVariableOp_33AssignVariableOp1assignvariableop_33_adam_stream_1_conv_1_kernel_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34�
AssignVariableOp_34AssignVariableOp/assignvariableop_34_adam_stream_1_conv_1_bias_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35�
AssignVariableOp_35AssignVariableOp1assignvariableop_35_adam_stream_2_conv_1_kernel_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36�
AssignVariableOp_36AssignVariableOp/assignvariableop_36_adam_stream_2_conv_1_bias_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37�
AssignVariableOp_37AssignVariableOp4assignvariableop_37_adam_batch_normalization_gamma_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38�
AssignVariableOp_38AssignVariableOp3assignvariableop_38_adam_batch_normalization_beta_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39�
AssignVariableOp_39AssignVariableOp6assignvariableop_39_adam_batch_normalization_1_gamma_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40�
AssignVariableOp_40AssignVariableOp5assignvariableop_40_adam_batch_normalization_1_beta_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41�
AssignVariableOp_41AssignVariableOp6assignvariableop_41_adam_batch_normalization_2_gamma_mIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42�
AssignVariableOp_42AssignVariableOp5assignvariableop_42_adam_batch_normalization_2_beta_mIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43�
AssignVariableOp_43AssignVariableOp)assignvariableop_43_adam_dense_1_kernel_mIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44�
AssignVariableOp_44AssignVariableOp'assignvariableop_44_adam_dense_1_bias_mIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45�
AssignVariableOp_45AssignVariableOp6assignvariableop_45_adam_batch_normalization_3_gamma_mIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46�
AssignVariableOp_46AssignVariableOp5assignvariableop_46_adam_batch_normalization_3_beta_mIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47�
AssignVariableOp_47AssignVariableOp1assignvariableop_47_adam_stream_0_conv_1_kernel_vIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48�
AssignVariableOp_48AssignVariableOp/assignvariableop_48_adam_stream_0_conv_1_bias_vIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49�
AssignVariableOp_49AssignVariableOp1assignvariableop_49_adam_stream_1_conv_1_kernel_vIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50�
AssignVariableOp_50AssignVariableOp/assignvariableop_50_adam_stream_1_conv_1_bias_vIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51�
AssignVariableOp_51AssignVariableOp1assignvariableop_51_adam_stream_2_conv_1_kernel_vIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52�
AssignVariableOp_52AssignVariableOp/assignvariableop_52_adam_stream_2_conv_1_bias_vIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53�
AssignVariableOp_53AssignVariableOp4assignvariableop_53_adam_batch_normalization_gamma_vIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54�
AssignVariableOp_54AssignVariableOp3assignvariableop_54_adam_batch_normalization_beta_vIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55�
AssignVariableOp_55AssignVariableOp6assignvariableop_55_adam_batch_normalization_1_gamma_vIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56�
AssignVariableOp_56AssignVariableOp5assignvariableop_56_adam_batch_normalization_1_beta_vIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57�
AssignVariableOp_57AssignVariableOp6assignvariableop_57_adam_batch_normalization_2_gamma_vIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58�
AssignVariableOp_58AssignVariableOp5assignvariableop_58_adam_batch_normalization_2_beta_vIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_58n
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:2
Identity_59�
AssignVariableOp_59AssignVariableOp)assignvariableop_59_adam_dense_1_kernel_vIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59n
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:2
Identity_60�
AssignVariableOp_60AssignVariableOp'assignvariableop_60_adam_dense_1_bias_vIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_60n
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:2
Identity_61�
AssignVariableOp_61AssignVariableOp6assignvariableop_61_adam_batch_normalization_3_gamma_vIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_61n
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:2
Identity_62�
AssignVariableOp_62AssignVariableOp5assignvariableop_62_adam_batch_normalization_3_beta_vIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_629
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp�
Identity_63Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_63f
Identity_64IdentityIdentity_63:output:0^NoOp_1*
T0*
_output_shapes
: 2
Identity_64�
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 2
NoOp_1"#
identity_64Identity_64:output:0*�
_input_shapes�
�: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
�
n
P__inference_stream_1_input_drop_layer_call_and_return_conditional_losses_1984644

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:����������2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:����������2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�+
�
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1988501

inputs5
'assignmovingavg_readvariableop_resource:@7
)assignmovingavg_1_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@/
!batchnorm_readvariableop_resource:@
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOp�
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indices�
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(2
moments/mean�
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:@2
moments/StopGradient�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :������������������@2
moments/SquaredDifference�
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices�
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(2
moments/variance�
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 2
moments/Squeeze�
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
�#<2
AssignMovingAvg/decay�
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg/sub�
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg/mul�
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
�#<2
AssignMovingAvg_1/decay�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg_1/sub�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg_1/mul�
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
 *o�:2
batchnorm/add/y�
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@2
batchnorm/Rsqrt�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOp�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2
batchnorm/mul�
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :������������������@2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:@2
batchnorm/mul_2�
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp�
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :������������������@2
batchnorm/add_1{
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :������������������@2

Identity�
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:������������������@: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :������������������@
 
_user_specified_nameinputs
�	
�
5__inference_batch_normalization_layer_call_fn_1988261

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_19839852
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :������������������@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:������������������@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :������������������@
 
_user_specified_nameinputs
�
�
+__inference_basemodel_layer_call_fn_1987609
inputs_0
inputs_1
inputs_2
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:@
	unknown_5:@
	unknown_6:@
	unknown_7:@
	unknown_8:@
	unknown_9:@

unknown_10:@

unknown_11:@

unknown_12:@

unknown_13:@

unknown_14:@

unknown_15:@

unknown_16:@

unknown_17:	�T

unknown_18:T

unknown_19:T

unknown_20:T

unknown_21:T

unknown_22:T
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1inputs_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_22*&
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������T*2
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_basemodel_layer_call_and_return_conditional_losses_19855952
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������T2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapesz
x:����������:����������:����������: : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
,
_output_shapes
:����������
"
_user_specified_name
inputs/0:VR
,
_output_shapes
:����������
"
_user_specified_name
inputs/1:VR
,
_output_shapes
:����������
"
_user_specified_name
inputs/2
�
j
L__inference_dense_1_dropout_layer_call_and_return_conditional_losses_1988922

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:����������2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:����������2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
o
P__inference_stream_2_input_drop_layer_call_and_return_conditional_losses_1985432

inputs
identity�c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
dropout/Constx
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:����������2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:����������*
dtype0*
seed�*
seed2�2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>2
dropout/GreaterEqual/y�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:����������2
dropout/GreaterEqual�
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:����������2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:����������2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
__inference_loss_fn_3_1989091I
6dense_1_kernel_regularizer_abs_readvariableop_resource:	�T
identity��-dense_1/kernel/Regularizer/Abs/ReadVariableOp�
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp6dense_1_kernel_regularizer_abs_readvariableop_resource*
_output_shapes
:	�T*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOp�
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	�T2 
dense_1/kernel/Regularizer/Abs�
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_1/kernel/Regularizer/Const�
dense_1/kernel/Regularizer/SumSum"dense_1/kernel/Regularizer/Abs:y:0)dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/Sum�
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2"
 dense_1/kernel/Regularizer/mul/x�
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
�
�
F__inference_basemodel_layer_call_and_return_conditional_losses_1985802
inputs_0
inputs_1
inputs_2-
stream_2_conv_1_1985709:@%
stream_2_conv_1_1985711:@-
stream_1_conv_1_1985714:@%
stream_1_conv_1_1985716:@-
stream_0_conv_1_1985719:@%
stream_0_conv_1_1985721:@+
batch_normalization_2_1985724:@+
batch_normalization_2_1985726:@+
batch_normalization_2_1985728:@+
batch_normalization_2_1985730:@+
batch_normalization_1_1985733:@+
batch_normalization_1_1985735:@+
batch_normalization_1_1985737:@+
batch_normalization_1_1985739:@)
batch_normalization_1985742:@)
batch_normalization_1985744:@)
batch_normalization_1985746:@)
batch_normalization_1985748:@"
dense_1_1985762:	�T
dense_1_1985764:T+
batch_normalization_3_1985767:T+
batch_normalization_3_1985769:T+
batch_normalization_3_1985771:T+
batch_normalization_3_1985773:T
identity��+batch_normalization/StatefulPartitionedCall�-batch_normalization_1/StatefulPartitionedCall�-batch_normalization_2/StatefulPartitionedCall�-batch_normalization_3/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�-dense_1/kernel/Regularizer/Abs/ReadVariableOp�'stream_0_conv_1/StatefulPartitionedCall�5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp�'stream_1_conv_1/StatefulPartitionedCall�8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp�'stream_2_conv_1/StatefulPartitionedCall�5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp�
#stream_2_input_drop/PartitionedCallPartitionedCallinputs_2*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *Y
fTRR
P__inference_stream_2_input_drop_layer_call_and_return_conditional_losses_19846372%
#stream_2_input_drop/PartitionedCall�
#stream_1_input_drop/PartitionedCallPartitionedCallinputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *Y
fTRR
P__inference_stream_1_input_drop_layer_call_and_return_conditional_losses_19846442%
#stream_1_input_drop/PartitionedCall�
#stream_0_input_drop/PartitionedCallPartitionedCallinputs_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *Y
fTRR
P__inference_stream_0_input_drop_layer_call_and_return_conditional_losses_19846512%
#stream_0_input_drop/PartitionedCall�
'stream_2_conv_1/StatefulPartitionedCallStatefulPartitionedCall,stream_2_input_drop/PartitionedCall:output:0stream_2_conv_1_1985709stream_2_conv_1_1985711*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_stream_2_conv_1_layer_call_and_return_conditional_losses_19846742)
'stream_2_conv_1/StatefulPartitionedCall�
'stream_1_conv_1/StatefulPartitionedCallStatefulPartitionedCall,stream_1_input_drop/PartitionedCall:output:0stream_1_conv_1_1985714stream_1_conv_1_1985716*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_stream_1_conv_1_layer_call_and_return_conditional_losses_19847012)
'stream_1_conv_1/StatefulPartitionedCall�
'stream_0_conv_1/StatefulPartitionedCallStatefulPartitionedCall,stream_0_input_drop/PartitionedCall:output:0stream_0_conv_1_1985719stream_0_conv_1_1985721*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_stream_0_conv_1_layer_call_and_return_conditional_losses_19847282)
'stream_0_conv_1/StatefulPartitionedCall�
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall0stream_2_conv_1/StatefulPartitionedCall:output:0batch_normalization_2_1985724batch_normalization_2_1985726batch_normalization_2_1985728batch_normalization_2_1985730*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *[
fVRT
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_19847532/
-batch_normalization_2/StatefulPartitionedCall�
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall0stream_1_conv_1/StatefulPartitionedCall:output:0batch_normalization_1_1985733batch_normalization_1_1985735batch_normalization_1_1985737batch_normalization_1_1985739*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *[
fVRT
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_19847822/
-batch_normalization_1/StatefulPartitionedCall�
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall0stream_0_conv_1/StatefulPartitionedCall:output:0batch_normalization_1985742batch_normalization_1985744batch_normalization_1985746batch_normalization_1985748*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_19848112-
+batch_normalization/StatefulPartitionedCall�
activation_2/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_activation_2_layer_call_and_return_conditional_losses_19848262
activation_2/PartitionedCall�
activation_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_activation_1_layer_call_and_return_conditional_losses_19848332
activation_1/PartitionedCall�
activation/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_activation_layer_call_and_return_conditional_losses_19848402
activation/PartitionedCall�
stream_2_drop_1/PartitionedCallPartitionedCall%activation_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_stream_2_drop_1_layer_call_and_return_conditional_losses_19848472!
stream_2_drop_1/PartitionedCall�
stream_1_drop_1/PartitionedCallPartitionedCall%activation_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_stream_1_drop_1_layer_call_and_return_conditional_losses_19848542!
stream_1_drop_1/PartitionedCall�
stream_0_drop_1/PartitionedCallPartitionedCall#activation/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_stream_0_drop_1_layer_call_and_return_conditional_losses_19848612!
stream_0_drop_1/PartitionedCall�
(global_average_pooling1d/PartitionedCallPartitionedCall(stream_0_drop_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *^
fYRW
U__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_19848682*
(global_average_pooling1d/PartitionedCall�
*global_average_pooling1d_1/PartitionedCallPartitionedCall(stream_1_drop_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *`
f[RY
W__inference_global_average_pooling1d_1_layer_call_and_return_conditional_losses_19848752,
*global_average_pooling1d_1/PartitionedCall�
*global_average_pooling1d_2/PartitionedCallPartitionedCall(stream_2_drop_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *`
f[RY
W__inference_global_average_pooling1d_2_layer_call_and_return_conditional_losses_19848822,
*global_average_pooling1d_2/PartitionedCall�
concatenate/PartitionedCallPartitionedCall1global_average_pooling1d/PartitionedCall:output:03global_average_pooling1d_1/PartitionedCall:output:03global_average_pooling1d_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_concatenate_layer_call_and_return_conditional_losses_19848922
concatenate/PartitionedCall�
dense_1_dropout/PartitionedCallPartitionedCall$concatenate/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_dense_1_dropout_layer_call_and_return_conditional_losses_19848992!
dense_1_dropout/PartitionedCall�
dense_1/StatefulPartitionedCallStatefulPartitionedCall(dense_1_dropout/PartitionedCall:output:0dense_1_1985762dense_1_1985764*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������T*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_19849172!
dense_1/StatefulPartitionedCall�
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0batch_normalization_3_1985767batch_normalization_3_1985769batch_normalization_3_1985771batch_normalization_3_1985773*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������T*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *[
fVRT
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_19844832/
-batch_normalization_3/StatefulPartitionedCall�
"dense_activation_1/PartitionedCallPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������T* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *X
fSRQ
O__inference_dense_activation_1_layer_call_and_return_conditional_losses_19849372$
"dense_activation_1/PartitionedCall�
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpstream_0_conv_1_1985719*"
_output_shapes
:@*
dtype027
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp�
&stream_0_conv_1/kernel/Regularizer/AbsAbs=stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_0_conv_1/kernel/Regularizer/Abs�
(stream_0_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_0_conv_1/kernel/Regularizer/Const�
&stream_0_conv_1/kernel/Regularizer/SumSum*stream_0_conv_1/kernel/Regularizer/Abs:y:01stream_0_conv_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/Sum�
(stream_0_conv_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2*
(stream_0_conv_1/kernel/Regularizer/mul/x�
&stream_0_conv_1/kernel/Regularizer/mulMul1stream_0_conv_1/kernel/Regularizer/mul/x:output:0/stream_0_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/mul�
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpstream_1_conv_1_1985714*"
_output_shapes
:@*
dtype02:
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp�
)stream_1_conv_1/kernel/Regularizer/SquareSquare@stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2+
)stream_1_conv_1/kernel/Regularizer/Square�
(stream_1_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_1_conv_1/kernel/Regularizer/Const�
&stream_1_conv_1/kernel/Regularizer/SumSum-stream_1_conv_1/kernel/Regularizer/Square:y:01stream_1_conv_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&stream_1_conv_1/kernel/Regularizer/Sum�
(stream_1_conv_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2*
(stream_1_conv_1/kernel/Regularizer/mul/x�
&stream_1_conv_1/kernel/Regularizer/mulMul1stream_1_conv_1/kernel/Regularizer/mul/x:output:0/stream_1_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_1_conv_1/kernel/Regularizer/mul�
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpstream_2_conv_1_1985709*"
_output_shapes
:@*
dtype027
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp�
&stream_2_conv_1/kernel/Regularizer/AbsAbs=stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_2_conv_1/kernel/Regularizer/Abs�
(stream_2_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_2_conv_1/kernel/Regularizer/Const�
&stream_2_conv_1/kernel/Regularizer/SumSum*stream_2_conv_1/kernel/Regularizer/Abs:y:01stream_2_conv_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&stream_2_conv_1/kernel/Regularizer/Sum�
(stream_2_conv_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2*
(stream_2_conv_1/kernel/Regularizer/mul/x�
&stream_2_conv_1/kernel/Regularizer/mulMul1stream_2_conv_1/kernel/Regularizer/mul/x:output:0/stream_2_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_2_conv_1/kernel/Regularizer/mul�
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_1_1985762*
_output_shapes
:	�T*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOp�
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	�T2 
dense_1/kernel/Regularizer/Abs�
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_1/kernel/Regularizer/Const�
dense_1/kernel/Regularizer/SumSum"dense_1/kernel/Regularizer/Abs:y:0)dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/Sum�
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2"
 dense_1/kernel/Regularizer/mul/x�
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/mul�
IdentityIdentity+dense_activation_1/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������T2

Identity�
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall.^dense_1/kernel/Regularizer/Abs/ReadVariableOp(^stream_0_conv_1/StatefulPartitionedCall6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp(^stream_1_conv_1/StatefulPartitionedCall9^stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp(^stream_2_conv_1/StatefulPartitionedCall6^stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapesz
x:����������:����������:����������: : : : : : : : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2^
-dense_1/kernel/Regularizer/Abs/ReadVariableOp-dense_1/kernel/Regularizer/Abs/ReadVariableOp2R
'stream_0_conv_1/StatefulPartitionedCall'stream_0_conv_1/StatefulPartitionedCall2n
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp2R
'stream_1_conv_1/StatefulPartitionedCall'stream_1_conv_1/StatefulPartitionedCall2t
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp2R
'stream_2_conv_1/StatefulPartitionedCall'stream_2_conv_1/StatefulPartitionedCall2n
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp:V R
,
_output_shapes
:����������
"
_user_specified_name
inputs_0:VR
,
_output_shapes
:����������
"
_user_specified_name
inputs_1:VR
,
_output_shapes
:����������
"
_user_specified_name
inputs_2
�
P
4__inference_dense_activation_1_layer_call_fn_1989042

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������T* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *X
fSRQ
O__inference_dense_activation_1_layer_call_and_return_conditional_losses_19849372
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������T2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������T:O K
'
_output_shapes
:���������T
 
_user_specified_nameinputs
�+
�
P__inference_batch_normalization_layer_call_and_return_conditional_losses_1985205

inputs5
'assignmovingavg_readvariableop_resource:@7
)assignmovingavg_1_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@/
!batchnorm_readvariableop_resource:@
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOp�
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indices�
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(2
moments/mean�
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:@2
moments/StopGradient�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*,
_output_shapes
:����������@2
moments/SquaredDifference�
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices�
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(2
moments/variance�
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 2
moments/Squeeze�
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
�#<2
AssignMovingAvg/decay�
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg/sub�
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg/mul�
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
�#<2
AssignMovingAvg_1/decay�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg_1/sub�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg_1/mul�
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
 *o�:2
batchnorm/add/y�
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@2
batchnorm/Rsqrt�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOp�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2
batchnorm/mul{
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*,
_output_shapes
:����������@2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:@2
batchnorm/mul_2�
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp�
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*,
_output_shapes
:����������@2
batchnorm/add_1s
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*,
_output_shapes
:����������@2

Identity�
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������@: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:T P
,
_output_shapes
:����������@
 
_user_specified_nameinputs
�
�
1__inference_stream_1_conv_1_layer_call_fn_1988178

inputs
unknown:@
	unknown_0:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_stream_1_conv_1_layer_call_and_return_conditional_losses_19847012
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:����������@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
1__inference_stream_0_conv_1_layer_call_fn_1988142

inputs
unknown:@
	unknown_0:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_stream_0_conv_1_layer_call_and_return_conditional_losses_19847282
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:����������@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
'__inference_model_layer_call_fn_1986367
left_inputs
right_inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:@
	unknown_5:@
	unknown_6:@
	unknown_7:@
	unknown_8:@
	unknown_9:@

unknown_10:@

unknown_11:@

unknown_12:@

unknown_13:@

unknown_14:@

unknown_15:@

unknown_16:@

unknown_17:	�T

unknown_18:T

unknown_19:T

unknown_20:T

unknown_21:T

unknown_22:T
identity��StatefulPartitionedCall�
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
:���������*2
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_model_layer_call_and_return_conditional_losses_19862622
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*s
_input_shapesb
`:����������:����������: : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
,
_output_shapes
:����������
%
_user_specified_nameleft_inputs:ZV
,
_output_shapes
:����������
&
_user_specified_nameright_inputs
�
s
W__inference_global_average_pooling1d_2_layer_call_and_return_conditional_losses_1984882

inputs
identityr
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Mean/reduction_indiceso
MeanMeaninputsMean/reduction_indices:output:0*
T0*'
_output_shapes
:���������@2
Meana
IdentityIdentityMean:output:0*
T0*'
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������@:T P
,
_output_shapes
:����������@
 
_user_specified_nameinputs
�
j
1__inference_stream_0_drop_1_layer_call_fn_1988755

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_stream_0_drop_1_layer_call_and_return_conditional_losses_19850892
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:����������@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������@22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:����������@
 
_user_specified_nameinputs
�
s
W__inference_global_average_pooling1d_1_layer_call_and_return_conditional_losses_1984875

inputs
identityr
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Mean/reduction_indiceso
MeanMeaninputsMean/reduction_indices:output:0*
T0*'
_output_shapes
:���������@2
Meana
IdentityIdentityMean:output:0*
T0*'
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������@:T P
,
_output_shapes
:����������@
 
_user_specified_nameinputs
�
�
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1988681

inputs/
!batchnorm_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@1
#batchnorm_readvariableop_1_resource:@1
#batchnorm_readvariableop_2_resource:@
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOp�
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
 *o�:2
batchnorm/add/y�
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@2
batchnorm/Rsqrt�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOp�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2
batchnorm/mul{
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*,
_output_shapes
:����������@2
batchnorm/mul_1�
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp_1�
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:@2
batchnorm/mul_2�
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp_2�
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*,
_output_shapes
:����������@2
batchnorm/add_1s
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*,
_output_shapes
:����������@2

Identity�
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������@: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:T P
,
_output_shapes
:����������@
 
_user_specified_nameinputs
�+
�
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1988661

inputs5
'assignmovingavg_readvariableop_resource:@7
)assignmovingavg_1_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@/
!batchnorm_readvariableop_resource:@
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOp�
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indices�
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(2
moments/mean�
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:@2
moments/StopGradient�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :������������������@2
moments/SquaredDifference�
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices�
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(2
moments/variance�
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 2
moments/Squeeze�
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
�#<2
AssignMovingAvg/decay�
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg/sub�
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg/mul�
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
�#<2
AssignMovingAvg_1/decay�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg_1/sub�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg_1/mul�
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
 *o�:2
batchnorm/add/y�
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@2
batchnorm/Rsqrt�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOp�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2
batchnorm/mul�
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :������������������@2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:@2
batchnorm/mul_2�
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp�
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :������������������@2
batchnorm/add_1{
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :������������������@2

Identity�
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:������������������@: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :������������������@
 
_user_specified_nameinputs
�
j
L__inference_stream_2_drop_1_layer_call_and_return_conditional_losses_1988814

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:����������@2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:����������@2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������@:T P
,
_output_shapes
:����������@
 
_user_specified_nameinputs
�
�
D__inference_dense_1_layer_call_and_return_conditional_losses_1988957

inputs1
matmul_readvariableop_resource:	�T-
biasadd_readvariableop_resource:T
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�-dense_1/kernel/Regularizer/Abs/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�T*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������T2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:T*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������T2	
BiasAdd�
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�T*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOp�
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	�T2 
dense_1/kernel/Regularizer/Abs�
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_1/kernel/Regularizer/Const�
dense_1/kernel/Regularizer/SumSum"dense_1/kernel/Regularizer/Abs:y:0)dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/Sum�
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2"
 dense_1/kernel/Regularizer/mul/x�
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/mulk
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������T2

Identity�
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp.^dense_1/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2^
-dense_1/kernel/Regularizer/Abs/ReadVariableOp-dense_1/kernel/Regularizer/Abs/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
k
L__inference_stream_2_drop_1_layer_call_and_return_conditional_losses_1988826

inputs
identity�c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
dropout/Constx
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:����������@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:����������@*
dtype0*
seed�*
seed2�2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>2
dropout/GreaterEqual/y�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:����������@2
dropout/GreaterEqual�
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:����������@2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:����������@2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:����������@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������@:T P
,
_output_shapes
:����������@
 
_user_specified_nameinputs
�	
q
E__inference_distance_layer_call_and_return_conditional_losses_1988046
inputs_0
inputs_1
identityW
subSubinputs_0inputs_1*
T0*'
_output_shapes
:���������T2
subU
SquareSquaresub:z:0*
T0*'
_output_shapes
:���������T2
Squarey
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
���������2
Sum/reduction_indices�
SumSum
Square:y:0Sum/reduction_indices:output:0*
T0*'
_output_shapes
:���������*
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
:���������2	
MaximumS
SqrtSqrtMaximum:z:0*
T0*'
_output_shapes
:���������2
Sqrt\
IdentityIdentitySqrt:y:0*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:���������T:���������T:Q M
'
_output_shapes
:���������T
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:���������T
"
_user_specified_name
inputs/1
�
�
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1988627

inputs/
!batchnorm_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@1
#batchnorm_readvariableop_1_resource:@1
#batchnorm_readvariableop_2_resource:@
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOp�
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
 *o�:2
batchnorm/add/y�
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@2
batchnorm/Rsqrt�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOp�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2
batchnorm/mul�
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :������������������@2
batchnorm/mul_1�
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp_1�
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:@2
batchnorm/mul_2�
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp_2�
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :������������������@2
batchnorm/add_1{
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :������������������@2

Identity�
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:������������������@: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :������������������@
 
_user_specified_nameinputs
�
j
1__inference_stream_1_drop_1_layer_call_fn_1988782

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_stream_1_drop_1_layer_call_and_return_conditional_losses_19851122
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:����������@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������@22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:����������@
 
_user_specified_nameinputs
�
X
<__inference_global_average_pooling1d_1_layer_call_fn_1988853

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:������������������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *`
f[RY
W__inference_global_average_pooling1d_1_layer_call_and_return_conditional_losses_19844212
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:������������������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'���������������������������:e a
=
_output_shapes+
):'���������������������������
 
_user_specified_nameinputs
�
�
P__inference_batch_normalization_layer_call_and_return_conditional_losses_1988307

inputs/
!batchnorm_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@1
#batchnorm_readvariableop_1_resource:@1
#batchnorm_readvariableop_2_resource:@
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOp�
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
 *o�:2
batchnorm/add/y�
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@2
batchnorm/Rsqrt�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOp�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2
batchnorm/mul�
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :������������������@2
batchnorm/mul_1�
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp_1�
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:@2
batchnorm/mul_2�
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp_2�
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :������������������@2
batchnorm/add_1{
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :������������������@2

Identity�
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:������������������@: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :������������������@
 
_user_specified_nameinputs
�
�
L__inference_stream_2_conv_1_layer_call_and_return_conditional_losses_1984674

inputsA
+conv1d_expanddims_1_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�"conv1d/ExpandDims_1/ReadVariableOp�5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
conv1d/ExpandDims/dim�
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:����������2
conv1d/ExpandDims�
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim�
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@2
conv1d/ExpandDims_1�
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:����������@*
paddingSAME*
strides
2
conv1d�
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:����������@*
squeeze_dims

���������2
conv1d/Squeeze�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������@2	
BiasAdd�
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype027
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp�
&stream_2_conv_1/kernel/Regularizer/AbsAbs=stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_2_conv_1/kernel/Regularizer/Abs�
(stream_2_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_2_conv_1/kernel/Regularizer/Const�
&stream_2_conv_1/kernel/Regularizer/SumSum*stream_2_conv_1/kernel/Regularizer/Abs:y:01stream_2_conv_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&stream_2_conv_1/kernel/Regularizer/Sum�
(stream_2_conv_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2*
(stream_2_conv_1/kernel/Regularizer/mul/x�
&stream_2_conv_1/kernel/Regularizer/mulMul1stream_2_conv_1/kernel/Regularizer/mul/x:output:0/stream_2_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_2_conv_1/kernel/Regularizer/mulp
IdentityIdentityBiasAdd:output:0^NoOp*
T0*,
_output_shapes
:����������@2

Identity�
NoOpNoOp^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp6^stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp2n
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
L__inference_stream_1_conv_1_layer_call_and_return_conditional_losses_1988199

inputsA
+conv1d_expanddims_1_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�"conv1d/ExpandDims_1/ReadVariableOp�8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
conv1d/ExpandDims/dim�
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:����������2
conv1d/ExpandDims�
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim�
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@2
conv1d/ExpandDims_1�
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:����������@*
paddingSAME*
strides
2
conv1d�
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:����������@*
squeeze_dims

���������2
conv1d/Squeeze�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������@2	
BiasAdd�
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02:
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp�
)stream_1_conv_1/kernel/Regularizer/SquareSquare@stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2+
)stream_1_conv_1/kernel/Regularizer/Square�
(stream_1_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_1_conv_1/kernel/Regularizer/Const�
&stream_1_conv_1/kernel/Regularizer/SumSum-stream_1_conv_1/kernel/Regularizer/Square:y:01stream_1_conv_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&stream_1_conv_1/kernel/Regularizer/Sum�
(stream_1_conv_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2*
(stream_1_conv_1/kernel/Regularizer/mul/x�
&stream_1_conv_1/kernel/Regularizer/mulMul1stream_1_conv_1/kernel/Regularizer/mul/x:output:0/stream_1_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_1_conv_1/kernel/Regularizer/mulp
IdentityIdentityBiasAdd:output:0^NoOp*
T0*,
_output_shapes
:����������@2

Identity�
NoOpNoOp^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp9^stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp2t
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�	
�
7__inference_batch_normalization_2_layer_call_fn_1988568

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *[
fVRT
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_19842492
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :������������������@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:������������������@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :������������������@
 
_user_specified_nameinputs
�
�
'__inference_model_layer_call_fn_1986076
left_inputs
right_inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:@
	unknown_5:@
	unknown_6:@
	unknown_7:@
	unknown_8:@
	unknown_9:@

unknown_10:@

unknown_11:@

unknown_12:@

unknown_13:@

unknown_14:@

unknown_15:@

unknown_16:@

unknown_17:	�T

unknown_18:T

unknown_19:T

unknown_20:T

unknown_21:T

unknown_22:T
identity��StatefulPartitionedCall�
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
:���������*:
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_model_layer_call_and_return_conditional_losses_19860252
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*s
_input_shapesb
`:����������:����������: : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
,
_output_shapes
:����������
%
_user_specified_nameleft_inputs:ZV
,
_output_shapes
:����������
&
_user_specified_nameright_inputs
�*
�
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1984543

inputs5
'assignmovingavg_readvariableop_resource:T7
)assignmovingavg_1_readvariableop_resource:T3
%batchnorm_mul_readvariableop_resource:T/
!batchnorm_readvariableop_resource:T
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOp�
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices�
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
moments/StopGradient�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:���������T2
moments/SquaredDifference�
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices�
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:T*
	keep_dims(2
moments/variance�
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:T*
squeeze_dims
 2
moments/Squeeze�
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
�#<2
AssignMovingAvg/decay�
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:T*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:T2
AssignMovingAvg/sub�
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:T2
AssignMovingAvg/mul�
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
�#<2
AssignMovingAvg_1/decay�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:T*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:T2
AssignMovingAvg_1/sub�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:T2
AssignMovingAvg_1/mul�
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
 *o�:2
batchnorm/add/y�
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:T2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:T2
batchnorm/Rsqrt�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:T*
dtype02
batchnorm/mul/ReadVariableOp�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:T2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������T2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:T2
batchnorm/mul_2�
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:T*
dtype02
batchnorm/ReadVariableOp�
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:T2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������T2
batchnorm/add_1n
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������T2

Identity�
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������T: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������T
 
_user_specified_nameinputs
�
�
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1988467

inputs/
!batchnorm_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@1
#batchnorm_readvariableop_1_resource:@1
#batchnorm_readvariableop_2_resource:@
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOp�
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
 *o�:2
batchnorm/add/y�
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@2
batchnorm/Rsqrt�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOp�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2
batchnorm/mul�
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :������������������@2
batchnorm/mul_1�
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp_1�
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:@2
batchnorm/mul_2�
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp_2�
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :������������������@2
batchnorm/add_1{
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :������������������@2

Identity�
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:������������������@: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :������������������@
 
_user_specified_nameinputs
�
�
'__inference_model_layer_call_fn_1986715
inputs_0
inputs_1
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:@
	unknown_5:@
	unknown_6:@
	unknown_7:@
	unknown_8:@
	unknown_9:@

unknown_10:@

unknown_11:@

unknown_12:@

unknown_13:@

unknown_14:@

unknown_15:@

unknown_16:@

unknown_17:	�T

unknown_18:T

unknown_19:T

unknown_20:T

unknown_21:T

unknown_22:T
identity��StatefulPartitionedCall�
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
:���������*:
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_model_layer_call_and_return_conditional_losses_19860252
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*s
_input_shapesb
`:����������:����������: : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
,
_output_shapes
:����������
"
_user_specified_name
inputs/0:VR
,
_output_shapes
:����������
"
_user_specified_name
inputs/1
�
n
P__inference_stream_0_input_drop_layer_call_and_return_conditional_losses_1984651

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:����������2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:����������2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
Q
5__inference_stream_2_input_drop_layer_call_fn_1988105

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *Y
fTRR
P__inference_stream_2_input_drop_layer_call_and_return_conditional_losses_19846372
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
5__inference_batch_normalization_layer_call_fn_1988274

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_19848112
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:����������@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:����������@
 
_user_specified_nameinputs
�
�
7__inference_batch_normalization_2_layer_call_fn_1988607

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *[
fVRT
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_19853252
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:����������@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:����������@
 
_user_specified_nameinputs
�
s
W__inference_global_average_pooling1d_2_layer_call_and_return_conditional_losses_1988892

inputs
identityr
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Mean/reduction_indiceso
MeanMeaninputsMean/reduction_indices:output:0*
T0*'
_output_shapes
:���������@2
Meana
IdentityIdentityMean:output:0*
T0*'
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������@:T P
,
_output_shapes
:����������@
 
_user_specified_nameinputs
�
H
,__inference_activation_layer_call_fn_1988720

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_activation_layer_call_and_return_conditional_losses_19848402
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:����������@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������@:T P
,
_output_shapes
:����������@
 
_user_specified_nameinputs
�
�
5__inference_batch_normalization_layer_call_fn_1988287

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_19852052
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:����������@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:����������@
 
_user_specified_nameinputs
�
V
:__inference_global_average_pooling1d_layer_call_fn_1988836

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *^
fYRW
U__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_19848682
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������@:T P
,
_output_shapes
:����������@
 
_user_specified_nameinputs
�
k
O__inference_dense_activation_1_layer_call_and_return_conditional_losses_1984937

inputs
identityW
SigmoidSigmoidinputs*
T0*'
_output_shapes
:���������T2	
Sigmoid_
IdentityIdentitySigmoid:y:0*
T0*'
_output_shapes
:���������T2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������T:O K
'
_output_shapes
:���������T
 
_user_specified_nameinputs"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
H
left_inputs9
serving_default_left_inputs:0����������
J
right_inputs:
serving_default_right_inputs:0����������<
distance0
StatefulPartitionedCall:0���������tensorflow/serving/predict:��
�
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
�__call__
+�&call_and_return_all_conditional_losses
�_default_save_signature"
_tf_keras_network
"
_tf_keras_input_layer
"
_tf_keras_input_layer
�
layer-0
layer-1
layer-2
layer-3
layer-4
layer-5
layer_with_weights-0
layer-6
layer_with_weights-1
layer-7
layer_with_weights-2
layer-8
layer_with_weights-3
layer-9
layer_with_weights-4
layer-10
layer_with_weights-5
layer-11
layer-12
layer-13
layer-14
layer-15
layer-16
layer-17
layer-18
layer-19
layer-20
 layer-21
!layer-22
"layer_with_weights-6
"layer-23
#layer_with_weights-7
#layer-24
$layer-25
%regularization_losses
&trainable_variables
'	variables
(	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_network
�
)regularization_losses
*trainable_variables
+	variables
,	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�

-beta_1

.beta_2
	/decay
0learning_rate
1iter2m�3m�4m�5m�6m�7m�8m�9m�:m�;m�<m�=m�>m�?m�@m�Am�2v�3v�4v�5v�6v�7v�8v�9v�:v�;v�<v�=v�>v�?v�@v�Av�"
	optimizer
 "
trackable_list_wrapper
�
20
31
42
53
64
75
86
97
:8
;9
<10
=11
>12
?13
@14
A15"
trackable_list_wrapper
�
20
31
42
53
64
75
86
97
B8
C9
:10
;11
D12
E13
<14
=15
F16
G17
>18
?19
@20
A21
H22
I23"
trackable_list_wrapper
�
Jlayer_metrics
regularization_losses
Klayer_regularization_losses
trainable_variables
	variables
Lnon_trainable_variables

Mlayers
Nmetrics
�__call__
�_default_save_signature
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
-
�serving_default"
signature_map
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
�
Oregularization_losses
Ptrainable_variables
Q	variables
R	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
Sregularization_losses
Ttrainable_variables
U	variables
V	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
Wregularization_losses
Xtrainable_variables
Y	variables
Z	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�

2kernel
3bias
[regularization_losses
\trainable_variables
]	variables
^	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�

4kernel
5bias
_regularization_losses
`trainable_variables
a	variables
b	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�

6kernel
7bias
cregularization_losses
dtrainable_variables
e	variables
f	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
gaxis
	8gamma
9beta
Bmoving_mean
Cmoving_variance
hregularization_losses
itrainable_variables
j	variables
k	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
laxis
	:gamma
;beta
Dmoving_mean
Emoving_variance
mregularization_losses
ntrainable_variables
o	variables
p	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
qaxis
	<gamma
=beta
Fmoving_mean
Gmoving_variance
rregularization_losses
strainable_variables
t	variables
u	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
vregularization_losses
wtrainable_variables
x	variables
y	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
zregularization_losses
{trainable_variables
|	variables
}	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
~regularization_losses
trainable_variables
�	variables
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�regularization_losses
�trainable_variables
�	variables
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�regularization_losses
�trainable_variables
�	variables
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�regularization_losses
�trainable_variables
�	variables
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�regularization_losses
�trainable_variables
�	variables
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�regularization_losses
�trainable_variables
�	variables
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�regularization_losses
�trainable_variables
�	variables
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�regularization_losses
�trainable_variables
�	variables
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�regularization_losses
�trainable_variables
�	variables
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�

>kernel
?bias
�regularization_losses
�trainable_variables
�	variables
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
	�axis
	@gamma
Abeta
Hmoving_mean
Imoving_variance
�regularization_losses
�trainable_variables
�	variables
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�regularization_losses
�trainable_variables
�	variables
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
@
�0
�1
�2
�3"
trackable_list_wrapper
�
20
31
42
53
64
75
86
97
:8
;9
<10
=11
>12
?13
@14
A15"
trackable_list_wrapper
�
20
31
42
53
64
75
86
97
B8
C9
:10
;11
D12
E13
<14
=15
F16
G17
>18
?19
@20
A21
H22
I23"
trackable_list_wrapper
�
�layer_metrics
%regularization_losses
 �layer_regularization_losses
&trainable_variables
'	variables
�non_trainable_variables
�layers
�metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�metrics
�layer_metrics
)regularization_losses
*trainable_variables
+	variables
�non_trainable_variables
�layers
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
: (2beta_1
: (2beta_2
: (2decay
: (2learning_rate
:	 (2	Adam/iter
,:*@2stream_0_conv_1/kernel
": @2stream_0_conv_1/bias
,:*@2stream_1_conv_1/kernel
": @2stream_1_conv_1/bias
,:*@2stream_2_conv_1/kernel
": @2stream_2_conv_1/bias
':%@2batch_normalization/gamma
&:$@2batch_normalization/beta
):'@2batch_normalization_1/gamma
(:&@2batch_normalization_1/beta
):'@2batch_normalization_2/gamma
(:&@2batch_normalization_2/beta
!:	�T2dense_1/kernel
:T2dense_1/bias
):'T2batch_normalization_3/gamma
(:&T2batch_normalization_3/beta
/:-@ (2batch_normalization/moving_mean
3:1@ (2#batch_normalization/moving_variance
1:/@ (2!batch_normalization_1/moving_mean
5:3@ (2%batch_normalization_1/moving_variance
1:/@ (2!batch_normalization_2/moving_mean
5:3@ (2%batch_normalization_2/moving_variance
1:/T (2!batch_normalization_3/moving_mean
5:3T (2%batch_normalization_3/moving_variance
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
X
B0
C1
D2
E3
F4
G5
H6
I7"
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�metrics
�layer_metrics
Oregularization_losses
Ptrainable_variables
Q	variables
�non_trainable_variables
�layers
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�metrics
�layer_metrics
Sregularization_losses
Ttrainable_variables
U	variables
�non_trainable_variables
�layers
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�metrics
�layer_metrics
Wregularization_losses
Xtrainable_variables
Y	variables
�non_trainable_variables
�layers
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
(
�0"
trackable_list_wrapper
.
20
31"
trackable_list_wrapper
.
20
31"
trackable_list_wrapper
�
�metrics
�layer_metrics
[regularization_losses
\trainable_variables
]	variables
�non_trainable_variables
�layers
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
(
�0"
trackable_list_wrapper
.
40
51"
trackable_list_wrapper
.
40
51"
trackable_list_wrapper
�
�metrics
�layer_metrics
_regularization_losses
`trainable_variables
a	variables
�non_trainable_variables
�layers
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
(
�0"
trackable_list_wrapper
.
60
71"
trackable_list_wrapper
.
60
71"
trackable_list_wrapper
�
�metrics
�layer_metrics
cregularization_losses
dtrainable_variables
e	variables
�non_trainable_variables
�layers
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
80
91"
trackable_list_wrapper
<
80
91
B2
C3"
trackable_list_wrapper
�
�metrics
�layer_metrics
hregularization_losses
itrainable_variables
j	variables
�non_trainable_variables
�layers
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
:0
;1"
trackable_list_wrapper
<
:0
;1
D2
E3"
trackable_list_wrapper
�
�metrics
�layer_metrics
mregularization_losses
ntrainable_variables
o	variables
�non_trainable_variables
�layers
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
<0
=1"
trackable_list_wrapper
<
<0
=1
F2
G3"
trackable_list_wrapper
�
�metrics
�layer_metrics
rregularization_losses
strainable_variables
t	variables
�non_trainable_variables
�layers
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�metrics
�layer_metrics
vregularization_losses
wtrainable_variables
x	variables
�non_trainable_variables
�layers
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�metrics
�layer_metrics
zregularization_losses
{trainable_variables
|	variables
�non_trainable_variables
�layers
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�metrics
�layer_metrics
~regularization_losses
trainable_variables
�	variables
�non_trainable_variables
�layers
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�metrics
�layer_metrics
�regularization_losses
�trainable_variables
�	variables
�non_trainable_variables
�layers
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�metrics
�layer_metrics
�regularization_losses
�trainable_variables
�	variables
�non_trainable_variables
�layers
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�metrics
�layer_metrics
�regularization_losses
�trainable_variables
�	variables
�non_trainable_variables
�layers
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�metrics
�layer_metrics
�regularization_losses
�trainable_variables
�	variables
�non_trainable_variables
�layers
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�metrics
�layer_metrics
�regularization_losses
�trainable_variables
�	variables
�non_trainable_variables
�layers
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�metrics
�layer_metrics
�regularization_losses
�trainable_variables
�	variables
�non_trainable_variables
�layers
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�metrics
�layer_metrics
�regularization_losses
�trainable_variables
�	variables
�non_trainable_variables
�layers
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�metrics
�layer_metrics
�regularization_losses
�trainable_variables
�	variables
�non_trainable_variables
�layers
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
(
�0"
trackable_list_wrapper
.
>0
?1"
trackable_list_wrapper
.
>0
?1"
trackable_list_wrapper
�
�metrics
�layer_metrics
�regularization_losses
�trainable_variables
�	variables
�non_trainable_variables
�layers
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
@0
A1"
trackable_list_wrapper
<
@0
A1
H2
I3"
trackable_list_wrapper
�
�metrics
�layer_metrics
�regularization_losses
�trainable_variables
�	variables
�non_trainable_variables
�layers
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�metrics
�layer_metrics
�regularization_losses
�trainable_variables
�	variables
�non_trainable_variables
�layers
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
X
B0
C1
D2
E3
F4
G5
H6
I7"
trackable_list_wrapper
�
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
19
20
 21
!22
"23
#24
$25"
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

�total

�count
�	variables
�	keras_api"
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
�0"
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
�0"
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
�0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
B0
C1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
D0
E1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
F0
G1"
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
�0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
H0
I1"
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
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
1:/@2Adam/stream_0_conv_1/kernel/m
':%@2Adam/stream_0_conv_1/bias/m
1:/@2Adam/stream_1_conv_1/kernel/m
':%@2Adam/stream_1_conv_1/bias/m
1:/@2Adam/stream_2_conv_1/kernel/m
':%@2Adam/stream_2_conv_1/bias/m
,:*@2 Adam/batch_normalization/gamma/m
+:)@2Adam/batch_normalization/beta/m
.:,@2"Adam/batch_normalization_1/gamma/m
-:+@2!Adam/batch_normalization_1/beta/m
.:,@2"Adam/batch_normalization_2/gamma/m
-:+@2!Adam/batch_normalization_2/beta/m
&:$	�T2Adam/dense_1/kernel/m
:T2Adam/dense_1/bias/m
.:,T2"Adam/batch_normalization_3/gamma/m
-:+T2!Adam/batch_normalization_3/beta/m
1:/@2Adam/stream_0_conv_1/kernel/v
':%@2Adam/stream_0_conv_1/bias/v
1:/@2Adam/stream_1_conv_1/kernel/v
':%@2Adam/stream_1_conv_1/bias/v
1:/@2Adam/stream_2_conv_1/kernel/v
':%@2Adam/stream_2_conv_1/bias/v
,:*@2 Adam/batch_normalization/gamma/v
+:)@2Adam/batch_normalization/beta/v
.:,@2"Adam/batch_normalization_1/gamma/v
-:+@2!Adam/batch_normalization_1/beta/v
.:,@2"Adam/batch_normalization_2/gamma/v
-:+@2!Adam/batch_normalization_2/beta/v
&:$	�T2Adam/dense_1/kernel/v
:T2Adam/dense_1/bias/v
.:,T2"Adam/batch_normalization_3/gamma/v
-:+T2!Adam/batch_normalization_3/beta/v
�2�
'__inference_model_layer_call_fn_1986076
'__inference_model_layer_call_fn_1986715
'__inference_model_layer_call_fn_1986769
'__inference_model_layer_call_fn_1986367�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
B__inference_model_layer_call_and_return_conditional_losses_1987025
B__inference_model_layer_call_and_return_conditional_losses_1987475
B__inference_model_layer_call_and_return_conditional_losses_1986471
B__inference_model_layer_call_and_return_conditional_losses_1986575�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
"__inference__wrapped_model_1983901left_inputsright_inputs"�
���
FullArgSpec
args� 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
+__inference_basemodel_layer_call_fn_1985015
+__inference_basemodel_layer_call_fn_1987554
+__inference_basemodel_layer_call_fn_1987609
+__inference_basemodel_layer_call_fn_1985701�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
F__inference_basemodel_layer_call_and_return_conditional_losses_1987761
F__inference_basemodel_layer_call_and_return_conditional_losses_1988010
F__inference_basemodel_layer_call_and_return_conditional_losses_1985802
F__inference_basemodel_layer_call_and_return_conditional_losses_1985903�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
*__inference_distance_layer_call_fn_1988016
*__inference_distance_layer_call_fn_1988022�
���
FullArgSpec1
args)�&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults�

 
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
E__inference_distance_layer_call_and_return_conditional_losses_1988034
E__inference_distance_layer_call_and_return_conditional_losses_1988046�
���
FullArgSpec1
args)�&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults�

 
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
%__inference_signature_wrapper_1986661left_inputsright_inputs"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
5__inference_stream_0_input_drop_layer_call_fn_1988051
5__inference_stream_0_input_drop_layer_call_fn_1988056�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
P__inference_stream_0_input_drop_layer_call_and_return_conditional_losses_1988061
P__inference_stream_0_input_drop_layer_call_and_return_conditional_losses_1988073�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
5__inference_stream_1_input_drop_layer_call_fn_1988078
5__inference_stream_1_input_drop_layer_call_fn_1988083�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
P__inference_stream_1_input_drop_layer_call_and_return_conditional_losses_1988088
P__inference_stream_1_input_drop_layer_call_and_return_conditional_losses_1988100�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
5__inference_stream_2_input_drop_layer_call_fn_1988105
5__inference_stream_2_input_drop_layer_call_fn_1988110�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
P__inference_stream_2_input_drop_layer_call_and_return_conditional_losses_1988115
P__inference_stream_2_input_drop_layer_call_and_return_conditional_losses_1988127�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
1__inference_stream_0_conv_1_layer_call_fn_1988142�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
L__inference_stream_0_conv_1_layer_call_and_return_conditional_losses_1988163�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
1__inference_stream_1_conv_1_layer_call_fn_1988178�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
L__inference_stream_1_conv_1_layer_call_and_return_conditional_losses_1988199�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
1__inference_stream_2_conv_1_layer_call_fn_1988214�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
L__inference_stream_2_conv_1_layer_call_and_return_conditional_losses_1988235�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
5__inference_batch_normalization_layer_call_fn_1988248
5__inference_batch_normalization_layer_call_fn_1988261
5__inference_batch_normalization_layer_call_fn_1988274
5__inference_batch_normalization_layer_call_fn_1988287�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
P__inference_batch_normalization_layer_call_and_return_conditional_losses_1988307
P__inference_batch_normalization_layer_call_and_return_conditional_losses_1988341
P__inference_batch_normalization_layer_call_and_return_conditional_losses_1988361
P__inference_batch_normalization_layer_call_and_return_conditional_losses_1988395�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
7__inference_batch_normalization_1_layer_call_fn_1988408
7__inference_batch_normalization_1_layer_call_fn_1988421
7__inference_batch_normalization_1_layer_call_fn_1988434
7__inference_batch_normalization_1_layer_call_fn_1988447�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1988467
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1988501
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1988521
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1988555�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
7__inference_batch_normalization_2_layer_call_fn_1988568
7__inference_batch_normalization_2_layer_call_fn_1988581
7__inference_batch_normalization_2_layer_call_fn_1988594
7__inference_batch_normalization_2_layer_call_fn_1988607�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1988627
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1988661
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1988681
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1988715�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
,__inference_activation_layer_call_fn_1988720�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
G__inference_activation_layer_call_and_return_conditional_losses_1988725�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
.__inference_activation_1_layer_call_fn_1988730�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
I__inference_activation_1_layer_call_and_return_conditional_losses_1988735�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
.__inference_activation_2_layer_call_fn_1988740�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
I__inference_activation_2_layer_call_and_return_conditional_losses_1988745�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
1__inference_stream_0_drop_1_layer_call_fn_1988750
1__inference_stream_0_drop_1_layer_call_fn_1988755�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
L__inference_stream_0_drop_1_layer_call_and_return_conditional_losses_1988760
L__inference_stream_0_drop_1_layer_call_and_return_conditional_losses_1988772�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
1__inference_stream_1_drop_1_layer_call_fn_1988777
1__inference_stream_1_drop_1_layer_call_fn_1988782�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
L__inference_stream_1_drop_1_layer_call_and_return_conditional_losses_1988787
L__inference_stream_1_drop_1_layer_call_and_return_conditional_losses_1988799�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
1__inference_stream_2_drop_1_layer_call_fn_1988804
1__inference_stream_2_drop_1_layer_call_fn_1988809�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
L__inference_stream_2_drop_1_layer_call_and_return_conditional_losses_1988814
L__inference_stream_2_drop_1_layer_call_and_return_conditional_losses_1988826�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
:__inference_global_average_pooling1d_layer_call_fn_1988831
:__inference_global_average_pooling1d_layer_call_fn_1988836�
���
FullArgSpec%
args�
jself
jinputs
jmask
varargs
 
varkw
 
defaults�

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
U__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_1988842
U__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_1988848�
���
FullArgSpec%
args�
jself
jinputs
jmask
varargs
 
varkw
 
defaults�

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
<__inference_global_average_pooling1d_1_layer_call_fn_1988853
<__inference_global_average_pooling1d_1_layer_call_fn_1988858�
���
FullArgSpec%
args�
jself
jinputs
jmask
varargs
 
varkw
 
defaults�

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
W__inference_global_average_pooling1d_1_layer_call_and_return_conditional_losses_1988864
W__inference_global_average_pooling1d_1_layer_call_and_return_conditional_losses_1988870�
���
FullArgSpec%
args�
jself
jinputs
jmask
varargs
 
varkw
 
defaults�

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
<__inference_global_average_pooling1d_2_layer_call_fn_1988875
<__inference_global_average_pooling1d_2_layer_call_fn_1988880�
���
FullArgSpec%
args�
jself
jinputs
jmask
varargs
 
varkw
 
defaults�

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
W__inference_global_average_pooling1d_2_layer_call_and_return_conditional_losses_1988886
W__inference_global_average_pooling1d_2_layer_call_and_return_conditional_losses_1988892�
���
FullArgSpec%
args�
jself
jinputs
jmask
varargs
 
varkw
 
defaults�

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
-__inference_concatenate_layer_call_fn_1988899�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
H__inference_concatenate_layer_call_and_return_conditional_losses_1988907�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
1__inference_dense_1_dropout_layer_call_fn_1988912
1__inference_dense_1_dropout_layer_call_fn_1988917�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
L__inference_dense_1_dropout_layer_call_and_return_conditional_losses_1988922
L__inference_dense_1_dropout_layer_call_and_return_conditional_losses_1988926�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
)__inference_dense_1_layer_call_fn_1988941�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
D__inference_dense_1_layer_call_and_return_conditional_losses_1988957�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
7__inference_batch_normalization_3_layer_call_fn_1988970
7__inference_batch_normalization_3_layer_call_fn_1988983�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1989003
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1989037�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
4__inference_dense_activation_1_layer_call_fn_1989042�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
O__inference_dense_activation_1_layer_call_and_return_conditional_losses_1989047�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
__inference_loss_fn_0_1989058�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�2�
__inference_loss_fn_1_1989069�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�2�
__inference_loss_fn_2_1989080�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�2�
__inference_loss_fn_3_1989091�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� �
"__inference__wrapped_model_1983901�674523G<F=E:D;C8B9>?I@HAk�h
a�^
\�Y
*�'
left_inputs����������
+�(
right_inputs����������
� "3�0
.
distance"�
distance����������
I__inference_activation_1_layer_call_and_return_conditional_losses_1988735b4�1
*�'
%�"
inputs����������@
� "*�'
 �
0����������@
� �
.__inference_activation_1_layer_call_fn_1988730U4�1
*�'
%�"
inputs����������@
� "�����������@�
I__inference_activation_2_layer_call_and_return_conditional_losses_1988745b4�1
*�'
%�"
inputs����������@
� "*�'
 �
0����������@
� �
.__inference_activation_2_layer_call_fn_1988740U4�1
*�'
%�"
inputs����������@
� "�����������@�
G__inference_activation_layer_call_and_return_conditional_losses_1988725b4�1
*�'
%�"
inputs����������@
� "*�'
 �
0����������@
� �
,__inference_activation_layer_call_fn_1988720U4�1
*�'
%�"
inputs����������@
� "�����������@�
F__inference_basemodel_layer_call_and_return_conditional_losses_1985802�674523G<F=E:D;C8B9>?I@HA���
���
~�{
'�$
inputs_0����������
'�$
inputs_1����������
'�$
inputs_2����������
p 

 
� "%�"
�
0���������T
� �
F__inference_basemodel_layer_call_and_return_conditional_losses_1985903�674523FG<=DE:;BC89>?HI@A���
���
~�{
'�$
inputs_0����������
'�$
inputs_1����������
'�$
inputs_2����������
p

 
� "%�"
�
0���������T
� �
F__inference_basemodel_layer_call_and_return_conditional_losses_1987761�674523G<F=E:D;C8B9>?I@HA���
���
~�{
'�$
inputs/0����������
'�$
inputs/1����������
'�$
inputs/2����������
p 

 
� "%�"
�
0���������T
� �
F__inference_basemodel_layer_call_and_return_conditional_losses_1988010�674523FG<=DE:;BC89>?HI@A���
���
~�{
'�$
inputs/0����������
'�$
inputs/1����������
'�$
inputs/2����������
p

 
� "%�"
�
0���������T
� �
+__inference_basemodel_layer_call_fn_1985015�674523G<F=E:D;C8B9>?I@HA���
���
~�{
'�$
inputs_0����������
'�$
inputs_1����������
'�$
inputs_2����������
p 

 
� "����������T�
+__inference_basemodel_layer_call_fn_1985701�674523FG<=DE:;BC89>?HI@A���
���
~�{
'�$
inputs_0����������
'�$
inputs_1����������
'�$
inputs_2����������
p

 
� "����������T�
+__inference_basemodel_layer_call_fn_1987554�674523G<F=E:D;C8B9>?I@HA���
���
~�{
'�$
inputs/0����������
'�$
inputs/1����������
'�$
inputs/2����������
p 

 
� "����������T�
+__inference_basemodel_layer_call_fn_1987609�674523FG<=DE:;BC89>?HI@A���
���
~�{
'�$
inputs/0����������
'�$
inputs/1����������
'�$
inputs/2����������
p

 
� "����������T�
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1988467|E:D;@�=
6�3
-�*
inputs������������������@
p 
� "2�/
(�%
0������������������@
� �
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1988501|DE:;@�=
6�3
-�*
inputs������������������@
p
� "2�/
(�%
0������������������@
� �
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1988521lE:D;8�5
.�+
%�"
inputs����������@
p 
� "*�'
 �
0����������@
� �
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1988555lDE:;8�5
.�+
%�"
inputs����������@
p
� "*�'
 �
0����������@
� �
7__inference_batch_normalization_1_layer_call_fn_1988408oE:D;@�=
6�3
-�*
inputs������������������@
p 
� "%�"������������������@�
7__inference_batch_normalization_1_layer_call_fn_1988421oDE:;@�=
6�3
-�*
inputs������������������@
p
� "%�"������������������@�
7__inference_batch_normalization_1_layer_call_fn_1988434_E:D;8�5
.�+
%�"
inputs����������@
p 
� "�����������@�
7__inference_batch_normalization_1_layer_call_fn_1988447_DE:;8�5
.�+
%�"
inputs����������@
p
� "�����������@�
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1988627|G<F=@�=
6�3
-�*
inputs������������������@
p 
� "2�/
(�%
0������������������@
� �
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1988661|FG<=@�=
6�3
-�*
inputs������������������@
p
� "2�/
(�%
0������������������@
� �
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1988681lG<F=8�5
.�+
%�"
inputs����������@
p 
� "*�'
 �
0����������@
� �
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1988715lFG<=8�5
.�+
%�"
inputs����������@
p
� "*�'
 �
0����������@
� �
7__inference_batch_normalization_2_layer_call_fn_1988568oG<F=@�=
6�3
-�*
inputs������������������@
p 
� "%�"������������������@�
7__inference_batch_normalization_2_layer_call_fn_1988581oFG<=@�=
6�3
-�*
inputs������������������@
p
� "%�"������������������@�
7__inference_batch_normalization_2_layer_call_fn_1988594_G<F=8�5
.�+
%�"
inputs����������@
p 
� "�����������@�
7__inference_batch_normalization_2_layer_call_fn_1988607_FG<=8�5
.�+
%�"
inputs����������@
p
� "�����������@�
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1989003bI@HA3�0
)�&
 �
inputs���������T
p 
� "%�"
�
0���������T
� �
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1989037bHI@A3�0
)�&
 �
inputs���������T
p
� "%�"
�
0���������T
� �
7__inference_batch_normalization_3_layer_call_fn_1988970UI@HA3�0
)�&
 �
inputs���������T
p 
� "����������T�
7__inference_batch_normalization_3_layer_call_fn_1988983UHI@A3�0
)�&
 �
inputs���������T
p
� "����������T�
P__inference_batch_normalization_layer_call_and_return_conditional_losses_1988307|C8B9@�=
6�3
-�*
inputs������������������@
p 
� "2�/
(�%
0������������������@
� �
P__inference_batch_normalization_layer_call_and_return_conditional_losses_1988341|BC89@�=
6�3
-�*
inputs������������������@
p
� "2�/
(�%
0������������������@
� �
P__inference_batch_normalization_layer_call_and_return_conditional_losses_1988361lC8B98�5
.�+
%�"
inputs����������@
p 
� "*�'
 �
0����������@
� �
P__inference_batch_normalization_layer_call_and_return_conditional_losses_1988395lBC898�5
.�+
%�"
inputs����������@
p
� "*�'
 �
0����������@
� �
5__inference_batch_normalization_layer_call_fn_1988248oC8B9@�=
6�3
-�*
inputs������������������@
p 
� "%�"������������������@�
5__inference_batch_normalization_layer_call_fn_1988261oBC89@�=
6�3
-�*
inputs������������������@
p
� "%�"������������������@�
5__inference_batch_normalization_layer_call_fn_1988274_C8B98�5
.�+
%�"
inputs����������@
p 
� "�����������@�
5__inference_batch_normalization_layer_call_fn_1988287_BC898�5
.�+
%�"
inputs����������@
p
� "�����������@�
H__inference_concatenate_layer_call_and_return_conditional_losses_1988907�~�{
t�q
o�l
"�
inputs/0���������@
"�
inputs/1���������@
"�
inputs/2���������@
� "&�#
�
0����������
� �
-__inference_concatenate_layer_call_fn_1988899�~�{
t�q
o�l
"�
inputs/0���������@
"�
inputs/1���������@
"�
inputs/2���������@
� "������������
L__inference_dense_1_dropout_layer_call_and_return_conditional_losses_1988922^4�1
*�'
!�
inputs����������
p 
� "&�#
�
0����������
� �
L__inference_dense_1_dropout_layer_call_and_return_conditional_losses_1988926^4�1
*�'
!�
inputs����������
p
� "&�#
�
0����������
� �
1__inference_dense_1_dropout_layer_call_fn_1988912Q4�1
*�'
!�
inputs����������
p 
� "������������
1__inference_dense_1_dropout_layer_call_fn_1988917Q4�1
*�'
!�
inputs����������
p
� "������������
D__inference_dense_1_layer_call_and_return_conditional_losses_1988957]>?0�-
&�#
!�
inputs����������
� "%�"
�
0���������T
� }
)__inference_dense_1_layer_call_fn_1988941P>?0�-
&�#
!�
inputs����������
� "����������T�
O__inference_dense_activation_1_layer_call_and_return_conditional_losses_1989047X/�,
%�"
 �
inputs���������T
� "%�"
�
0���������T
� �
4__inference_dense_activation_1_layer_call_fn_1989042K/�,
%�"
 �
inputs���������T
� "����������T�
E__inference_distance_layer_call_and_return_conditional_losses_1988034�b�_
X�U
K�H
"�
inputs/0���������T
"�
inputs/1���������T

 
p 
� "%�"
�
0���������
� �
E__inference_distance_layer_call_and_return_conditional_losses_1988046�b�_
X�U
K�H
"�
inputs/0���������T
"�
inputs/1���������T

 
p
� "%�"
�
0���������
� �
*__inference_distance_layer_call_fn_1988016~b�_
X�U
K�H
"�
inputs/0���������T
"�
inputs/1���������T

 
p 
� "�����������
*__inference_distance_layer_call_fn_1988022~b�_
X�U
K�H
"�
inputs/0���������T
"�
inputs/1���������T

 
p
� "�����������
W__inference_global_average_pooling1d_1_layer_call_and_return_conditional_losses_1988864{I�F
?�<
6�3
inputs'���������������������������

 
� ".�+
$�!
0������������������
� �
W__inference_global_average_pooling1d_1_layer_call_and_return_conditional_losses_1988870a8�5
.�+
%�"
inputs����������@

 
� "%�"
�
0���������@
� �
<__inference_global_average_pooling1d_1_layer_call_fn_1988853nI�F
?�<
6�3
inputs'���������������������������

 
� "!��������������������
<__inference_global_average_pooling1d_1_layer_call_fn_1988858T8�5
.�+
%�"
inputs����������@

 
� "����������@�
W__inference_global_average_pooling1d_2_layer_call_and_return_conditional_losses_1988886{I�F
?�<
6�3
inputs'���������������������������

 
� ".�+
$�!
0������������������
� �
W__inference_global_average_pooling1d_2_layer_call_and_return_conditional_losses_1988892a8�5
.�+
%�"
inputs����������@

 
� "%�"
�
0���������@
� �
<__inference_global_average_pooling1d_2_layer_call_fn_1988875nI�F
?�<
6�3
inputs'���������������������������

 
� "!��������������������
<__inference_global_average_pooling1d_2_layer_call_fn_1988880T8�5
.�+
%�"
inputs����������@

 
� "����������@�
U__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_1988842{I�F
?�<
6�3
inputs'���������������������������

 
� ".�+
$�!
0������������������
� �
U__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_1988848a8�5
.�+
%�"
inputs����������@

 
� "%�"
�
0���������@
� �
:__inference_global_average_pooling1d_layer_call_fn_1988831nI�F
?�<
6�3
inputs'���������������������������

 
� "!��������������������
:__inference_global_average_pooling1d_layer_call_fn_1988836T8�5
.�+
%�"
inputs����������@

 
� "����������@<
__inference_loss_fn_0_19890582�

� 
� "� <
__inference_loss_fn_1_19890694�

� 
� "� <
__inference_loss_fn_2_19890806�

� 
� "� <
__inference_loss_fn_3_1989091>�

� 
� "� �
B__inference_model_layer_call_and_return_conditional_losses_1986471�674523G<F=E:D;C8B9>?I@HAs�p
i�f
\�Y
*�'
left_inputs����������
+�(
right_inputs����������
p 

 
� "%�"
�
0���������
� �
B__inference_model_layer_call_and_return_conditional_losses_1986575�674523FG<=DE:;BC89>?HI@As�p
i�f
\�Y
*�'
left_inputs����������
+�(
right_inputs����������
p

 
� "%�"
�
0���������
� �
B__inference_model_layer_call_and_return_conditional_losses_1987025�674523G<F=E:D;C8B9>?I@HAl�i
b�_
U�R
'�$
inputs/0����������
'�$
inputs/1����������
p 

 
� "%�"
�
0���������
� �
B__inference_model_layer_call_and_return_conditional_losses_1987475�674523FG<=DE:;BC89>?HI@Al�i
b�_
U�R
'�$
inputs/0����������
'�$
inputs/1����������
p

 
� "%�"
�
0���������
� �
'__inference_model_layer_call_fn_1986076�674523G<F=E:D;C8B9>?I@HAs�p
i�f
\�Y
*�'
left_inputs����������
+�(
right_inputs����������
p 

 
� "�����������
'__inference_model_layer_call_fn_1986367�674523FG<=DE:;BC89>?HI@As�p
i�f
\�Y
*�'
left_inputs����������
+�(
right_inputs����������
p

 
� "�����������
'__inference_model_layer_call_fn_1986715�674523G<F=E:D;C8B9>?I@HAl�i
b�_
U�R
'�$
inputs/0����������
'�$
inputs/1����������
p 

 
� "�����������
'__inference_model_layer_call_fn_1986769�674523FG<=DE:;BC89>?HI@Al�i
b�_
U�R
'�$
inputs/0����������
'�$
inputs/1����������
p

 
� "�����������
%__inference_signature_wrapper_1986661�674523G<F=E:D;C8B9>?I@HA���
� 
{�x
9
left_inputs*�'
left_inputs����������
;
right_inputs+�(
right_inputs����������"3�0
.
distance"�
distance����������
L__inference_stream_0_conv_1_layer_call_and_return_conditional_losses_1988163f234�1
*�'
%�"
inputs����������
� "*�'
 �
0����������@
� �
1__inference_stream_0_conv_1_layer_call_fn_1988142Y234�1
*�'
%�"
inputs����������
� "�����������@�
L__inference_stream_0_drop_1_layer_call_and_return_conditional_losses_1988760f8�5
.�+
%�"
inputs����������@
p 
� "*�'
 �
0����������@
� �
L__inference_stream_0_drop_1_layer_call_and_return_conditional_losses_1988772f8�5
.�+
%�"
inputs����������@
p
� "*�'
 �
0����������@
� �
1__inference_stream_0_drop_1_layer_call_fn_1988750Y8�5
.�+
%�"
inputs����������@
p 
� "�����������@�
1__inference_stream_0_drop_1_layer_call_fn_1988755Y8�5
.�+
%�"
inputs����������@
p
� "�����������@�
P__inference_stream_0_input_drop_layer_call_and_return_conditional_losses_1988061f8�5
.�+
%�"
inputs����������
p 
� "*�'
 �
0����������
� �
P__inference_stream_0_input_drop_layer_call_and_return_conditional_losses_1988073f8�5
.�+
%�"
inputs����������
p
� "*�'
 �
0����������
� �
5__inference_stream_0_input_drop_layer_call_fn_1988051Y8�5
.�+
%�"
inputs����������
p 
� "������������
5__inference_stream_0_input_drop_layer_call_fn_1988056Y8�5
.�+
%�"
inputs����������
p
� "������������
L__inference_stream_1_conv_1_layer_call_and_return_conditional_losses_1988199f454�1
*�'
%�"
inputs����������
� "*�'
 �
0����������@
� �
1__inference_stream_1_conv_1_layer_call_fn_1988178Y454�1
*�'
%�"
inputs����������
� "�����������@�
L__inference_stream_1_drop_1_layer_call_and_return_conditional_losses_1988787f8�5
.�+
%�"
inputs����������@
p 
� "*�'
 �
0����������@
� �
L__inference_stream_1_drop_1_layer_call_and_return_conditional_losses_1988799f8�5
.�+
%�"
inputs����������@
p
� "*�'
 �
0����������@
� �
1__inference_stream_1_drop_1_layer_call_fn_1988777Y8�5
.�+
%�"
inputs����������@
p 
� "�����������@�
1__inference_stream_1_drop_1_layer_call_fn_1988782Y8�5
.�+
%�"
inputs����������@
p
� "�����������@�
P__inference_stream_1_input_drop_layer_call_and_return_conditional_losses_1988088f8�5
.�+
%�"
inputs����������
p 
� "*�'
 �
0����������
� �
P__inference_stream_1_input_drop_layer_call_and_return_conditional_losses_1988100f8�5
.�+
%�"
inputs����������
p
� "*�'
 �
0����������
� �
5__inference_stream_1_input_drop_layer_call_fn_1988078Y8�5
.�+
%�"
inputs����������
p 
� "������������
5__inference_stream_1_input_drop_layer_call_fn_1988083Y8�5
.�+
%�"
inputs����������
p
� "������������
L__inference_stream_2_conv_1_layer_call_and_return_conditional_losses_1988235f674�1
*�'
%�"
inputs����������
� "*�'
 �
0����������@
� �
1__inference_stream_2_conv_1_layer_call_fn_1988214Y674�1
*�'
%�"
inputs����������
� "�����������@�
L__inference_stream_2_drop_1_layer_call_and_return_conditional_losses_1988814f8�5
.�+
%�"
inputs����������@
p 
� "*�'
 �
0����������@
� �
L__inference_stream_2_drop_1_layer_call_and_return_conditional_losses_1988826f8�5
.�+
%�"
inputs����������@
p
� "*�'
 �
0����������@
� �
1__inference_stream_2_drop_1_layer_call_fn_1988804Y8�5
.�+
%�"
inputs����������@
p 
� "�����������@�
1__inference_stream_2_drop_1_layer_call_fn_1988809Y8�5
.�+
%�"
inputs����������@
p
� "�����������@�
P__inference_stream_2_input_drop_layer_call_and_return_conditional_losses_1988115f8�5
.�+
%�"
inputs����������
p 
� "*�'
 �
0����������
� �
P__inference_stream_2_input_drop_layer_call_and_return_conditional_losses_1988127f8�5
.�+
%�"
inputs����������
p
� "*�'
 �
0����������
� �
5__inference_stream_2_input_drop_layer_call_fn_1988105Y8�5
.�+
%�"
inputs����������
p 
� "������������
5__inference_stream_2_input_drop_layer_call_fn_1988110Y8�5
.�+
%�"
inputs����������
p
� "�����������