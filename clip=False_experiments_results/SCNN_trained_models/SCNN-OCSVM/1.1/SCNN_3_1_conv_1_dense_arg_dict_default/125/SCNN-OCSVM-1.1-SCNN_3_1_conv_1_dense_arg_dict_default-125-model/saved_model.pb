из5
»Ю
D
AddV2
x"T
y"T
z"T"
Ttype:
2	АР
B
AssignVariableOp
resource
value"dtype"
dtypetypeИ
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
Ы
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
Н
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
delete_old_dirsbool(И
?
Mul
x"T
y"T
z"T"
Ttype:
2	Р
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
dtypetypeИ
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
list(type)(0И
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
list(type)(0И
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
Њ
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
executor_typestring И
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
М
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
Ц
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 И"serve*2.6.22v2.6.1-9-gc2363d6d0258уџ1
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
М
stream_0_conv_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*'
shared_namestream_0_conv_1/kernel
Е
*stream_0_conv_1/kernel/Read/ReadVariableOpReadVariableOpstream_0_conv_1/kernel*"
_output_shapes
:@*
dtype0
А
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
М
stream_1_conv_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*'
shared_namestream_1_conv_1/kernel
Е
*stream_1_conv_1/kernel/Read/ReadVariableOpReadVariableOpstream_1_conv_1/kernel*"
_output_shapes
:@*
dtype0
А
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
М
stream_2_conv_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*'
shared_namestream_2_conv_1/kernel
Е
*stream_2_conv_1/kernel/Read/ReadVariableOpReadVariableOpstream_2_conv_1/kernel*"
_output_shapes
:@*
dtype0
А
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
К
batch_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@**
shared_namebatch_normalization/gamma
Г
-batch_normalization/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization/gamma*
_output_shapes
:@*
dtype0
И
batch_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*)
shared_namebatch_normalization/beta
Б
,batch_normalization/beta/Read/ReadVariableOpReadVariableOpbatch_normalization/beta*
_output_shapes
:@*
dtype0
О
batch_normalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_1/gamma
З
/batch_normalization_1/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_1/gamma*
_output_shapes
:@*
dtype0
М
batch_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_namebatch_normalization_1/beta
Е
.batch_normalization_1/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_1/beta*
_output_shapes
:@*
dtype0
О
batch_normalization_2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_2/gamma
З
/batch_normalization_2/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_2/gamma*
_output_shapes
:@*
dtype0
М
batch_normalization_2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_namebatch_normalization_2/beta
Е
.batch_normalization_2/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_2/beta*
_output_shapes
:@*
dtype0
y
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	јT*
shared_namedense_1/kernel
r
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes
:	јT*
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
О
batch_normalization_3/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:T*,
shared_namebatch_normalization_3/gamma
З
/batch_normalization_3/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_3/gamma*
_output_shapes
:T*
dtype0
М
batch_normalization_3/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:T*+
shared_namebatch_normalization_3/beta
Е
.batch_normalization_3/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_3/beta*
_output_shapes
:T*
dtype0
Ц
batch_normalization/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*0
shared_name!batch_normalization/moving_mean
П
3batch_normalization/moving_mean/Read/ReadVariableOpReadVariableOpbatch_normalization/moving_mean*
_output_shapes
:@*
dtype0
Ю
#batch_normalization/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#batch_normalization/moving_variance
Ч
7batch_normalization/moving_variance/Read/ReadVariableOpReadVariableOp#batch_normalization/moving_variance*
_output_shapes
:@*
dtype0
Ъ
!batch_normalization_1/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!batch_normalization_1/moving_mean
У
5batch_normalization_1/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_1/moving_mean*
_output_shapes
:@*
dtype0
Ґ
%batch_normalization_1/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*6
shared_name'%batch_normalization_1/moving_variance
Ы
9batch_normalization_1/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_1/moving_variance*
_output_shapes
:@*
dtype0
Ъ
!batch_normalization_2/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!batch_normalization_2/moving_mean
У
5batch_normalization_2/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_2/moving_mean*
_output_shapes
:@*
dtype0
Ґ
%batch_normalization_2/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*6
shared_name'%batch_normalization_2/moving_variance
Ы
9batch_normalization_2/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_2/moving_variance*
_output_shapes
:@*
dtype0
Ъ
!batch_normalization_3/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:T*2
shared_name#!batch_normalization_3/moving_mean
У
5batch_normalization_3/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_3/moving_mean*
_output_shapes
:T*
dtype0
Ґ
%batch_normalization_3/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:T*6
shared_name'%batch_normalization_3/moving_variance
Ы
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
Ъ
Adam/stream_0_conv_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*.
shared_nameAdam/stream_0_conv_1/kernel/m
У
1Adam/stream_0_conv_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/stream_0_conv_1/kernel/m*"
_output_shapes
:@*
dtype0
О
Adam/stream_0_conv_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_nameAdam/stream_0_conv_1/bias/m
З
/Adam/stream_0_conv_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/stream_0_conv_1/bias/m*
_output_shapes
:@*
dtype0
Ъ
Adam/stream_1_conv_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*.
shared_nameAdam/stream_1_conv_1/kernel/m
У
1Adam/stream_1_conv_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/stream_1_conv_1/kernel/m*"
_output_shapes
:@*
dtype0
О
Adam/stream_1_conv_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_nameAdam/stream_1_conv_1/bias/m
З
/Adam/stream_1_conv_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/stream_1_conv_1/bias/m*
_output_shapes
:@*
dtype0
Ъ
Adam/stream_2_conv_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*.
shared_nameAdam/stream_2_conv_1/kernel/m
У
1Adam/stream_2_conv_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/stream_2_conv_1/kernel/m*"
_output_shapes
:@*
dtype0
О
Adam/stream_2_conv_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_nameAdam/stream_2_conv_1/bias/m
З
/Adam/stream_2_conv_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/stream_2_conv_1/bias/m*
_output_shapes
:@*
dtype0
Ш
 Adam/batch_normalization/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*1
shared_name" Adam/batch_normalization/gamma/m
С
4Adam/batch_normalization/gamma/m/Read/ReadVariableOpReadVariableOp Adam/batch_normalization/gamma/m*
_output_shapes
:@*
dtype0
Ц
Adam/batch_normalization/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*0
shared_name!Adam/batch_normalization/beta/m
П
3Adam/batch_normalization/beta/m/Read/ReadVariableOpReadVariableOpAdam/batch_normalization/beta/m*
_output_shapes
:@*
dtype0
Ь
"Adam/batch_normalization_1/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Adam/batch_normalization_1/gamma/m
Х
6Adam/batch_normalization_1/gamma/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_1/gamma/m*
_output_shapes
:@*
dtype0
Ъ
!Adam/batch_normalization_1/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!Adam/batch_normalization_1/beta/m
У
5Adam/batch_normalization_1/beta/m/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_1/beta/m*
_output_shapes
:@*
dtype0
Ь
"Adam/batch_normalization_2/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Adam/batch_normalization_2/gamma/m
Х
6Adam/batch_normalization_2/gamma/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_2/gamma/m*
_output_shapes
:@*
dtype0
Ъ
!Adam/batch_normalization_2/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!Adam/batch_normalization_2/beta/m
У
5Adam/batch_normalization_2/beta/m/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_2/beta/m*
_output_shapes
:@*
dtype0
З
Adam/dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	јT*&
shared_nameAdam/dense_1/kernel/m
А
)Adam/dense_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/m*
_output_shapes
:	јT*
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
Ь
"Adam/batch_normalization_3/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:T*3
shared_name$"Adam/batch_normalization_3/gamma/m
Х
6Adam/batch_normalization_3/gamma/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_3/gamma/m*
_output_shapes
:T*
dtype0
Ъ
!Adam/batch_normalization_3/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:T*2
shared_name#!Adam/batch_normalization_3/beta/m
У
5Adam/batch_normalization_3/beta/m/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_3/beta/m*
_output_shapes
:T*
dtype0
Ъ
Adam/stream_0_conv_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*.
shared_nameAdam/stream_0_conv_1/kernel/v
У
1Adam/stream_0_conv_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/stream_0_conv_1/kernel/v*"
_output_shapes
:@*
dtype0
О
Adam/stream_0_conv_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_nameAdam/stream_0_conv_1/bias/v
З
/Adam/stream_0_conv_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/stream_0_conv_1/bias/v*
_output_shapes
:@*
dtype0
Ъ
Adam/stream_1_conv_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*.
shared_nameAdam/stream_1_conv_1/kernel/v
У
1Adam/stream_1_conv_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/stream_1_conv_1/kernel/v*"
_output_shapes
:@*
dtype0
О
Adam/stream_1_conv_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_nameAdam/stream_1_conv_1/bias/v
З
/Adam/stream_1_conv_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/stream_1_conv_1/bias/v*
_output_shapes
:@*
dtype0
Ъ
Adam/stream_2_conv_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*.
shared_nameAdam/stream_2_conv_1/kernel/v
У
1Adam/stream_2_conv_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/stream_2_conv_1/kernel/v*"
_output_shapes
:@*
dtype0
О
Adam/stream_2_conv_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_nameAdam/stream_2_conv_1/bias/v
З
/Adam/stream_2_conv_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/stream_2_conv_1/bias/v*
_output_shapes
:@*
dtype0
Ш
 Adam/batch_normalization/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*1
shared_name" Adam/batch_normalization/gamma/v
С
4Adam/batch_normalization/gamma/v/Read/ReadVariableOpReadVariableOp Adam/batch_normalization/gamma/v*
_output_shapes
:@*
dtype0
Ц
Adam/batch_normalization/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*0
shared_name!Adam/batch_normalization/beta/v
П
3Adam/batch_normalization/beta/v/Read/ReadVariableOpReadVariableOpAdam/batch_normalization/beta/v*
_output_shapes
:@*
dtype0
Ь
"Adam/batch_normalization_1/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Adam/batch_normalization_1/gamma/v
Х
6Adam/batch_normalization_1/gamma/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_1/gamma/v*
_output_shapes
:@*
dtype0
Ъ
!Adam/batch_normalization_1/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!Adam/batch_normalization_1/beta/v
У
5Adam/batch_normalization_1/beta/v/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_1/beta/v*
_output_shapes
:@*
dtype0
Ь
"Adam/batch_normalization_2/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Adam/batch_normalization_2/gamma/v
Х
6Adam/batch_normalization_2/gamma/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_2/gamma/v*
_output_shapes
:@*
dtype0
Ъ
!Adam/batch_normalization_2/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!Adam/batch_normalization_2/beta/v
У
5Adam/batch_normalization_2/beta/v/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_2/beta/v*
_output_shapes
:@*
dtype0
З
Adam/dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	јT*&
shared_nameAdam/dense_1/kernel/v
А
)Adam/dense_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/v*
_output_shapes
:	јT*
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
Ь
"Adam/batch_normalization_3/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:T*3
shared_name$"Adam/batch_normalization_3/gamma/v
Х
6Adam/batch_normalization_3/gamma/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_3/gamma/v*
_output_shapes
:T*
dtype0
Ъ
!Adam/batch_normalization_3/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:T*2
shared_name#!Adam/batch_normalization_3/beta/v
У
5Adam/batch_normalization_3/beta/v/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_3/beta/v*
_output_shapes
:T*
dtype0

NoOpNoOp
ђЗ
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*жЖ
valueџЖB„Ж BѕЖ
њ
layer-0
layer-1
layer_with_weights-0
layer-2
layer-3
	optimizer
trainable_variables
	variables
regularization_losses
		keras_api


signatures
 
 
Д
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
%trainable_variables
&	variables
'regularization_losses
(	keras_api
R
)trainable_variables
*	variables
+regularization_losses
,	keras_api
А

-beta_1

.beta_2
	/decay
0learning_rate
1iter2m±3m≤4m≥5mі6mµ7mґ8mЈ9mЄ:mє;mЇ<mї=mЉ>mљ?mЊ@mњAmј2vЅ3v¬4v√5vƒ6v≈7v∆8v«9v»:v…;v <vЋ=vћ>vЌ?vќ@vѕAv–
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
ґ
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
 
≠
Jlayer_metrics
Kmetrics
Llayer_regularization_losses
trainable_variables
Mnon_trainable_variables

Nlayers
	variables
regularization_losses
 
 
 
 
R
Otrainable_variables
P	variables
Qregularization_losses
R	keras_api
R
Strainable_variables
T	variables
Uregularization_losses
V	keras_api
R
Wtrainable_variables
X	variables
Yregularization_losses
Z	keras_api
h

2kernel
3bias
[trainable_variables
\	variables
]regularization_losses
^	keras_api
h

4kernel
5bias
_trainable_variables
`	variables
aregularization_losses
b	keras_api
h

6kernel
7bias
ctrainable_variables
d	variables
eregularization_losses
f	keras_api
Ч
gaxis
	8gamma
9beta
Bmoving_mean
Cmoving_variance
htrainable_variables
i	variables
jregularization_losses
k	keras_api
Ч
laxis
	:gamma
;beta
Dmoving_mean
Emoving_variance
mtrainable_variables
n	variables
oregularization_losses
p	keras_api
Ч
qaxis
	<gamma
=beta
Fmoving_mean
Gmoving_variance
rtrainable_variables
s	variables
tregularization_losses
u	keras_api
R
vtrainable_variables
w	variables
xregularization_losses
y	keras_api
R
ztrainable_variables
{	variables
|regularization_losses
}	keras_api
T
~trainable_variables
	variables
Аregularization_losses
Б	keras_api
V
Вtrainable_variables
Г	variables
Дregularization_losses
Е	keras_api
V
Жtrainable_variables
З	variables
Иregularization_losses
Й	keras_api
V
Кtrainable_variables
Л	variables
Мregularization_losses
Н	keras_api
V
Оtrainable_variables
П	variables
Рregularization_losses
С	keras_api
V
Тtrainable_variables
У	variables
Фregularization_losses
Х	keras_api
V
Цtrainable_variables
Ч	variables
Шregularization_losses
Щ	keras_api
V
Ъtrainable_variables
Ы	variables
Ьregularization_losses
Э	keras_api
V
Юtrainable_variables
Я	variables
†regularization_losses
°	keras_api
l

>kernel
?bias
Ґtrainable_variables
£	variables
§regularization_losses
•	keras_api
Ь
	¶axis
	@gamma
Abeta
Hmoving_mean
Imoving_variance
Іtrainable_variables
®	variables
©regularization_losses
™	keras_api
V
Ђtrainable_variables
ђ	variables
≠regularization_losses
Ѓ	keras_api
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
ґ
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
 
≤
ѓlayer_metrics
∞metrics
 ±layer_regularization_losses
%trainable_variables
≤non_trainable_variables
≥layers
&	variables
'regularization_losses
 
 
 
≤
іlayer_metrics
µmetrics
 ґlayer_regularization_losses
)trainable_variables
Јnon_trainable_variables
Єlayers
*	variables
+regularization_losses
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

є0
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
 
 
 
≤
Їlayer_metrics
їmetrics
 Љlayer_regularization_losses
Otrainable_variables
љnon_trainable_variables
Њlayers
P	variables
Qregularization_losses
 
 
 
≤
њlayer_metrics
јmetrics
 Ѕlayer_regularization_losses
Strainable_variables
¬non_trainable_variables
√layers
T	variables
Uregularization_losses
 
 
 
≤
ƒlayer_metrics
≈metrics
 ∆layer_regularization_losses
Wtrainable_variables
«non_trainable_variables
»layers
X	variables
Yregularization_losses

20
31

20
31
 
≤
…layer_metrics
 metrics
 Ћlayer_regularization_losses
[trainable_variables
ћnon_trainable_variables
Ќlayers
\	variables
]regularization_losses

40
51

40
51
 
≤
ќlayer_metrics
ѕmetrics
 –layer_regularization_losses
_trainable_variables
—non_trainable_variables
“layers
`	variables
aregularization_losses

60
71

60
71
 
≤
”layer_metrics
‘metrics
 ’layer_regularization_losses
ctrainable_variables
÷non_trainable_variables
„layers
d	variables
eregularization_losses
 

80
91

80
91
B2
C3
 
≤
Ўlayer_metrics
ўmetrics
 Џlayer_regularization_losses
htrainable_variables
џnon_trainable_variables
№layers
i	variables
jregularization_losses
 

:0
;1

:0
;1
D2
E3
 
≤
Ёlayer_metrics
ёmetrics
 яlayer_regularization_losses
mtrainable_variables
аnon_trainable_variables
бlayers
n	variables
oregularization_losses
 

<0
=1

<0
=1
F2
G3
 
≤
вlayer_metrics
гmetrics
 дlayer_regularization_losses
rtrainable_variables
еnon_trainable_variables
жlayers
s	variables
tregularization_losses
 
 
 
≤
зlayer_metrics
иmetrics
 йlayer_regularization_losses
vtrainable_variables
кnon_trainable_variables
лlayers
w	variables
xregularization_losses
 
 
 
≤
мlayer_metrics
нmetrics
 оlayer_regularization_losses
ztrainable_variables
пnon_trainable_variables
рlayers
{	variables
|regularization_losses
 
 
 
≥
сlayer_metrics
тmetrics
 уlayer_regularization_losses
~trainable_variables
фnon_trainable_variables
хlayers
	variables
Аregularization_losses
 
 
 
µ
цlayer_metrics
чmetrics
 шlayer_regularization_losses
Вtrainable_variables
щnon_trainable_variables
ъlayers
Г	variables
Дregularization_losses
 
 
 
µ
ыlayer_metrics
ьmetrics
 эlayer_regularization_losses
Жtrainable_variables
юnon_trainable_variables
€layers
З	variables
Иregularization_losses
 
 
 
µ
Аlayer_metrics
Бmetrics
 Вlayer_regularization_losses
Кtrainable_variables
Гnon_trainable_variables
Дlayers
Л	variables
Мregularization_losses
 
 
 
µ
Еlayer_metrics
Жmetrics
 Зlayer_regularization_losses
Оtrainable_variables
Иnon_trainable_variables
Йlayers
П	variables
Рregularization_losses
 
 
 
µ
Кlayer_metrics
Лmetrics
 Мlayer_regularization_losses
Тtrainable_variables
Нnon_trainable_variables
Оlayers
У	variables
Фregularization_losses
 
 
 
µ
Пlayer_metrics
Рmetrics
 Сlayer_regularization_losses
Цtrainable_variables
Тnon_trainable_variables
Уlayers
Ч	variables
Шregularization_losses
 
 
 
µ
Фlayer_metrics
Хmetrics
 Цlayer_regularization_losses
Ъtrainable_variables
Чnon_trainable_variables
Шlayers
Ы	variables
Ьregularization_losses
 
 
 
µ
Щlayer_metrics
Ъmetrics
 Ыlayer_regularization_losses
Юtrainable_variables
Ьnon_trainable_variables
Эlayers
Я	variables
†regularization_losses

>0
?1

>0
?1
 
µ
Юlayer_metrics
Яmetrics
 †layer_regularization_losses
Ґtrainable_variables
°non_trainable_variables
Ґlayers
£	variables
§regularization_losses
 

@0
A1

@0
A1
H2
I3
 
µ
£layer_metrics
§metrics
 •layer_regularization_losses
Іtrainable_variables
¶non_trainable_variables
Іlayers
®	variables
©regularization_losses
 
 
 
µ
®layer_metrics
©metrics
 ™layer_regularization_losses
Ђtrainable_variables
Ђnon_trainable_variables
ђlayers
ђ	variables
≠regularization_losses
 
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
∆
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
8

≠total

Ѓcount
ѓ	variables
∞	keras_api
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
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

≠0
Ѓ1

ѓ	variables
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
ГА
VARIABLE_VALUE Adam/batch_normalization/gamma/mLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
Б
VARIABLE_VALUEAdam/batch_normalization/beta/mLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЕВ
VARIABLE_VALUE"Adam/batch_normalization_1/gamma/mLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ДБ
VARIABLE_VALUE!Adam/batch_normalization_1/beta/mLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЖГ
VARIABLE_VALUE"Adam/batch_normalization_2/gamma/mMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЕВ
VARIABLE_VALUE!Adam/batch_normalization_2/beta/mMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/dense_1/kernel/mMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/dense_1/bias/mMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЖГ
VARIABLE_VALUE"Adam/batch_normalization_3/gamma/mMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЕВ
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
ГА
VARIABLE_VALUE Adam/batch_normalization/gamma/vLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
Б
VARIABLE_VALUEAdam/batch_normalization/beta/vLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЕВ
VARIABLE_VALUE"Adam/batch_normalization_1/gamma/vLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ДБ
VARIABLE_VALUE!Adam/batch_normalization_1/beta/vLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЖГ
VARIABLE_VALUE"Adam/batch_normalization_2/gamma/vMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЕВ
VARIABLE_VALUE!Adam/batch_normalization_2/beta/vMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/dense_1/kernel/vMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/dense_1/bias/vMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЖГ
VARIABLE_VALUE"Adam/batch_normalization_3/gamma/vMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЕВ
VARIABLE_VALUE!Adam/batch_normalization_3/beta/vMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
Ж
serving_default_left_inputsPlaceholder*+
_output_shapes
:€€€€€€€€€}*
dtype0* 
shape:€€€€€€€€€}
З
serving_default_right_inputsPlaceholder*+
_output_shapes
:€€€€€€€€€}*
dtype0* 
shape:€€€€€€€€€}
і
StatefulPartitionedCallStatefulPartitionedCallserving_default_left_inputsserving_default_right_inputsstream_2_conv_1/kernelstream_2_conv_1/biasstream_1_conv_1/kernelstream_1_conv_1/biasstream_0_conv_1/kernelstream_0_conv_1/bias%batch_normalization_2/moving_variancebatch_normalization_2/gamma!batch_normalization_2/moving_meanbatch_normalization_2/beta%batch_normalization_1/moving_variancebatch_normalization_1/gamma!batch_normalization_1/moving_meanbatch_normalization_1/beta#batch_normalization/moving_variancebatch_normalization/gammabatch_normalization/moving_meanbatch_normalization/betadense_1/kerneldense_1/bias%batch_normalization_3/moving_variancebatch_normalization_3/gamma!batch_normalization_3/moving_meanbatch_normalization_3/beta*%
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*:
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *-
f(R&
$__inference_signature_wrapper_106309
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
§
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
GPU2*0J 8В *(
f#R!
__inference__traced_save_108952
≥
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
GPU2*0J 8В *+
f&R$
"__inference__traced_restore_109151–Ф/
Џ
“
K__inference_stream_2_conv_1_layer_call_and_return_conditional_losses_107874

inputsA
+conv1d_expanddims_1_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identityИҐBiasAdd/ReadVariableOpҐ"conv1d/ExpandDims_1/ReadVariableOpҐ5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€2
conv1d/ExpandDims/dimЦ
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:€€€€€€€€€}2
conv1d/ExpandDimsЄ
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
conv1d/ExpandDims_1/dimЈ
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@2
conv1d/ExpandDims_1ґ
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:€€€€€€€€€}@*
paddingSAME*
strides
2
conv1dТ
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:€€€€€€€€€}@*
squeeze_dims

э€€€€€€€€2
conv1d/SqueezeМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpМ
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€}@2	
BiasAddё
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype027
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp√
&stream_2_conv_1/kernel/Regularizer/AbsAbs=stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_2_conv_1/kernel/Regularizer/Abs©
(stream_2_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_2_conv_1/kernel/Regularizer/Const„
&stream_2_conv_1/kernel/Regularizer/SumSum*stream_2_conv_1/kernel/Regularizer/Abs:y:01stream_2_conv_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&stream_2_conv_1/kernel/Regularizer/SumЩ
(stream_2_conv_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2*
(stream_2_conv_1/kernel/Regularizer/mul/x№
&stream_2_conv_1/kernel/Regularizer/mulMul1stream_2_conv_1/kernel/Regularizer/mul/x:output:0/stream_2_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_2_conv_1/kernel/Regularizer/mulo
IdentityIdentityBiasAdd:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€}@2

Identityƒ
NoOpNoOp^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp6^stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€}: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp2n
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp:S O
+
_output_shapes
:€€€€€€€€€}
 
_user_specified_nameinputs
Є+
к
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_108097

inputs5
'assignmovingavg_readvariableop_resource:@7
)assignmovingavg_1_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@/
!batchnorm_readvariableop_resource:@
identityИҐAssignMovingAvgҐAssignMovingAvg/ReadVariableOpҐAssignMovingAvg_1Ґ AssignMovingAvg_1/ReadVariableOpҐbatchnorm/ReadVariableOpҐbatchnorm/mul/ReadVariableOpС
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indicesУ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(2
moments/meanА
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:@2
moments/StopGradient±
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€@2
moments/SquaredDifferenceЩ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indicesґ
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(2
moments/varianceБ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 2
moments/SqueezeЙ
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
„#<2
AssignMovingAvg/decay§
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOpШ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg/subП
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg/mulњ
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
„#<2
AssignMovingAvg_1/decay™
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOp†
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg_1/subЧ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg_1/mul…
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
 *oГ:2
batchnorm/add/yВ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2
batchnorm/mulГ
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€@2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:@2
batchnorm/mul_2Т
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOpБ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2
batchnorm/subТ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€@2
batchnorm/add_1{
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€@2

Identityт
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:€€€€€€€€€€€€€€€€€€@: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs
с
n
O__inference_stream_2_input_drop_layer_call_and_return_conditional_losses_105080

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †?2
dropout/Constw
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:€€€€€€€€€}2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape”
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:€€€€€€€€€}*
dtype0*
seedЈ*
seed2Ј2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL>2
dropout/GreaterEqual/y¬
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:€€€€€€€€€}2
dropout/GreaterEqualГ
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:€€€€€€€€€}2
dropout/Cast~
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:€€€€€€€€€}2
dropout/Mul_1i
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:€€€€€€€€€}2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€}:S O
+
_output_shapes
:€€€€€€€€€}
 
_user_specified_nameinputs
е
P
4__inference_stream_2_input_drop_layer_call_fn_107770

inputs
identity‘
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€}* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_stream_2_input_drop_layer_call_and_return_conditional_losses_1042852
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:€€€€€€€€€}2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€}:S O
+
_output_shapes
:€€€€€€€€€}
 
_user_specified_nameinputs
Џ
“
K__inference_stream_0_conv_1_layer_call_and_return_conditional_losses_107802

inputsA
+conv1d_expanddims_1_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identityИҐBiasAdd/ReadVariableOpҐ"conv1d/ExpandDims_1/ReadVariableOpҐ5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€2
conv1d/ExpandDims/dimЦ
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:€€€€€€€€€}2
conv1d/ExpandDimsЄ
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
conv1d/ExpandDims_1/dimЈ
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@2
conv1d/ExpandDims_1ґ
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:€€€€€€€€€}@*
paddingSAME*
strides
2
conv1dТ
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:€€€€€€€€€}@*
squeeze_dims

э€€€€€€€€2
conv1d/SqueezeМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpМ
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€}@2	
BiasAddё
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype027
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp√
&stream_0_conv_1/kernel/Regularizer/AbsAbs=stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_0_conv_1/kernel/Regularizer/Abs©
(stream_0_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_0_conv_1/kernel/Regularizer/Const„
&stream_0_conv_1/kernel/Regularizer/SumSum*stream_0_conv_1/kernel/Regularizer/Abs:y:01stream_0_conv_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/SumЩ
(stream_0_conv_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2*
(stream_0_conv_1/kernel/Regularizer/mul/x№
&stream_0_conv_1/kernel/Regularizer/mulMul1stream_0_conv_1/kernel/Regularizer/mul/x:output:0/stream_0_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/mulo
IdentityIdentityBiasAdd:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€}@2

Identityƒ
NoOpNoOp^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€}: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp2n
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:S O
+
_output_shapes
:€€€€€€€€€}
 
_user_specified_nameinputs
Ј
∞
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_103735

inputs/
!batchnorm_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@1
#batchnorm_readvariableop_1_resource:@1
#batchnorm_readvariableop_2_resource:@
identityИҐbatchnorm/ReadVariableOpҐbatchnorm/ReadVariableOp_1Ґbatchnorm/ReadVariableOp_2Ґbatchnorm/mul/ReadVariableOpТ
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
 *oГ:2
batchnorm/add/yИ
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2
batchnorm/mulГ
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€@2
batchnorm/mul_1Ш
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp_1Е
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:@2
batchnorm/mul_2Ш
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp_2Г
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2
batchnorm/subТ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€@2
batchnorm/add_1{
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€@2

Identity¬
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:€€€€€€€€€€€€€€€€€€@: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs
Ц
°
0__inference_stream_2_conv_1_layer_call_fn_107883

inputs
unknown:@
	unknown_0:@
identityИҐStatefulPartitionedCallВ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€}@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_stream_2_conv_1_layer_call_and_return_conditional_losses_1043222
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€}@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€}: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:€€€€€€€€€}
 
_user_specified_nameinputs
к
—
6__inference_batch_normalization_1_layer_call_fn_108190

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityИҐStatefulPartitionedCallҐ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€}@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1044302
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€}@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€}@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:€€€€€€€€€}@
 
_user_specified_nameinputs
Н	
—
6__inference_batch_normalization_2_layer_call_fn_108337

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityИҐStatefulPartitionedCall©
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1039572
StatefulPartitionedCallИ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:€€€€€€€€€€€€€€€€€€@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs
µ
Ѓ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_107903

inputs/
!batchnorm_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@1
#batchnorm_readvariableop_1_resource:@1
#batchnorm_readvariableop_2_resource:@
identityИҐbatchnorm/ReadVariableOpҐbatchnorm/ReadVariableOp_1Ґbatchnorm/ReadVariableOp_2Ґbatchnorm/mul/ReadVariableOpТ
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
 *oГ:2
batchnorm/add/yИ
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2
batchnorm/mulГ
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€@2
batchnorm/mul_1Ш
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp_1Е
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:@2
batchnorm/mul_2Ш
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp_2Г
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2
batchnorm/subТ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€@2
batchnorm/add_1{
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€@2

Identity¬
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:€€€€€€€€€€€€€€€€€€@: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs
у
•
C__inference_dense_1_layer_call_and_return_conditional_losses_108596

inputs1
matmul_readvariableop_resource:	јT-
biasadd_readvariableop_resource:T
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpҐ-dense_1/kernel/Regularizer/Abs/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	јT*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€T2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:T*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€T2	
BiasAddЊ
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	јT*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOp®
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	јT2 
dense_1/kernel/Regularizer/AbsХ
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_1/kernel/Regularizer/ConstЈ
dense_1/kernel/Regularizer/SumSum"dense_1/kernel/Regularizer/Abs:y:0)dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/SumЙ
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2"
 dense_1/kernel/Regularizer/mul/xЉ
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/mulk
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€T2

Identityѓ
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp.^dense_1/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€ј: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2^
-dense_1/kernel/Regularizer/Abs/ReadVariableOp-dense_1/kernel/Regularizer/Abs/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€ј
 
_user_specified_nameinputs
И
i
K__inference_stream_2_drop_1_layer_call_and_return_conditional_losses_104495

inputs

identity_1^
IdentityIdentityinputs*
T0*+
_output_shapes
:€€€€€€€€€}@2

Identitym

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:€€€€€€€€€}@2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€}@:S O
+
_output_shapes
:€€€€€€€€€}@
 
_user_specified_nameinputs
н
j
K__inference_stream_2_drop_1_layer_call_and_return_conditional_losses_104783

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †?2
dropout/Constw
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:€€€€€€€€€}@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape”
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:€€€€€€€€€}@*
dtype0*
seedЈ*
seed2Ј2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL>2
dropout/GreaterEqual/y¬
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:€€€€€€€€€}@2
dropout/GreaterEqualГ
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:€€€€€€€€€}@2
dropout/Cast~
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:€€€€€€€€€}@2
dropout/Mul_1i
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:€€€€€€€€€}@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€}@:S O
+
_output_shapes
:€€€€€€€€€}@
 
_user_specified_nameinputs
Љ
r
V__inference_global_average_pooling1d_2_layer_call_and_return_conditional_losses_108524

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
:€€€€€€€€€€€€€€€€€€2
Meanj
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€:e a
=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
ћ*
к
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_108659

inputs5
'assignmovingavg_readvariableop_resource:T7
)assignmovingavg_1_readvariableop_resource:T3
%batchnorm_mul_readvariableop_resource:T/
!batchnorm_readvariableop_resource:T
identityИҐAssignMovingAvgҐAssignMovingAvg/ReadVariableOpҐAssignMovingAvg_1Ґ AssignMovingAvg_1/ReadVariableOpҐbatchnorm/ReadVariableOpҐbatchnorm/mul/ReadVariableOpК
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indicesП
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
moments/StopGradient§
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:€€€€€€€€€T2
moments/SquaredDifferenceТ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices≤
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:T*
	keep_dims(2
moments/varianceА
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:T*
squeeze_dims
 2
moments/SqueezeИ
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
„#<2
AssignMovingAvg/decay§
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:T*
dtype02 
AssignMovingAvg/ReadVariableOpШ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:T2
AssignMovingAvg/subП
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:T2
AssignMovingAvg/mulњ
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
„#<2
AssignMovingAvg_1/decay™
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:T*
dtype02"
 AssignMovingAvg_1/ReadVariableOp†
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:T2
AssignMovingAvg_1/subЧ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:T2
AssignMovingAvg_1/mul…
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
 *oГ:2
batchnorm/add/yВ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:T2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:T2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:T*
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:T2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:€€€€€€€€€T2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:T2
batchnorm/mul_2Т
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:T*
dtype02
batchnorm/ReadVariableOpБ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:T2
batchnorm/subЕ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:€€€€€€€€€T2
batchnorm/add_1n
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€T2

Identityт
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€T: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€T
 
_user_specified_nameinputs
к
—
6__inference_batch_normalization_2_layer_call_fn_108350

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityИҐStatefulPartitionedCallҐ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€}@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1044012
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€}@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€}@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:€€€€€€€€€}@
 
_user_specified_nameinputs
Ж
r
V__inference_global_average_pooling1d_1_layer_call_and_return_conditional_losses_104523

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
:€€€€€€€€€@2
Meana
IdentityIdentityMean:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€}@:S O
+
_output_shapes
:€€€€€€€€€}@
 
_user_specified_nameinputs
П	
—
6__inference_batch_normalization_1_layer_call_fn_108164

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityИҐStatefulPartitionedCallЂ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1037352
StatefulPartitionedCallИ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:€€€€€€€€€€€€€€€€€€@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs
и
j
N__inference_dense_activation_1_layer_call_and_return_conditional_losses_104585

inputs
identityW
SigmoidSigmoidinputs*
T0*'
_output_shapes
:€€€€€€€€€T2	
Sigmoid_
IdentityIdentitySigmoid:y:0*
T0*'
_output_shapes
:€€€€€€€€€T2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:€€€€€€€€€T:O K
'
_output_shapes
:€€€€€€€€€T
 
_user_specified_nameinputs
Б+
к
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_104913

inputs5
'assignmovingavg_readvariableop_resource:@7
)assignmovingavg_1_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@/
!batchnorm_readvariableop_resource:@
identityИҐAssignMovingAvgҐAssignMovingAvg/ReadVariableOpҐAssignMovingAvg_1Ґ AssignMovingAvg_1/ReadVariableOpҐbatchnorm/ReadVariableOpҐbatchnorm/mul/ReadVariableOpС
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indicesУ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(2
moments/meanА
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:@2
moments/StopGradient®
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*+
_output_shapes
:€€€€€€€€€}@2
moments/SquaredDifferenceЩ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indicesґ
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(2
moments/varianceБ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 2
moments/SqueezeЙ
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
„#<2
AssignMovingAvg/decay§
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOpШ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg/subП
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg/mulњ
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
„#<2
AssignMovingAvg_1/decay™
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOp†
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg_1/subЧ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg_1/mul…
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
 *oГ:2
batchnorm/add/yВ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2
batchnorm/mulz
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€}@2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:@2
batchnorm/mul_2Т
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOpБ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2
batchnorm/subЙ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*+
_output_shapes
:€€€€€€€€€}@2
batchnorm/add_1r
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€}@2

Identityт
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€}@: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:S O
+
_output_shapes
:€€€€€€€€€}@
 
_user_specified_nameinputs
°
W
;__inference_global_average_pooling1d_2_layer_call_fn_108535

inputs
identityа
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *_
fZRX
V__inference_global_average_pooling1d_2_layer_call_and_return_conditional_losses_1040932
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€:e a
=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
ь
i
K__inference_dense_1_dropout_layer_call_and_return_conditional_losses_104547

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:€€€€€€€€€ј2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:€€€€€€€€€ј2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:€€€€€€€€€ј:P L
(
_output_shapes
:€€€€€€€€€ј
 
_user_specified_nameinputs
И
i
K__inference_stream_1_drop_1_layer_call_and_return_conditional_losses_104502

inputs

identity_1^
IdentityIdentityinputs*
T0*+
_output_shapes
:€€€€€€€€€}@2

Identitym

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:€€€€€€€€€}@2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€}@:S O
+
_output_shapes
:€€€€€€€€€}@
 
_user_specified_nameinputs
с
n
O__inference_stream_0_input_drop_layer_call_and_return_conditional_losses_105034

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †?2
dropout/Constw
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:€€€€€€€€€}2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape”
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:€€€€€€€€€}*
dtype0*
seedЈ*
seed2Ј2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL>2
dropout/GreaterEqual/y¬
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:€€€€€€€€€}2
dropout/GreaterEqualГ
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:€€€€€€€€€}2
dropout/Cast~
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:€€€€€€€€€}2
dropout/Mul_1i
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:€€€€€€€€€}2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€}:S O
+
_output_shapes
:€€€€€€€€€}
 
_user_specified_nameinputs
л
W
;__inference_global_average_pooling1d_2_layer_call_fn_108540

inputs
identity„
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *_
fZRX
V__inference_global_average_pooling1d_2_layer_call_and_return_conditional_losses_1045302
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€}@:S O
+
_output_shapes
:€€€€€€€€€}@
 
_user_specified_nameinputs
мы
Р
E__inference_basemodel_layer_call_and_return_conditional_losses_107299
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
&dense_1_matmul_readvariableop_resource:	јT5
'dense_1_biasadd_readvariableop_resource:TE
7batch_normalization_3_batchnorm_readvariableop_resource:TI
;batch_normalization_3_batchnorm_mul_readvariableop_resource:TG
9batch_normalization_3_batchnorm_readvariableop_1_resource:TG
9batch_normalization_3_batchnorm_readvariableop_2_resource:T
identityИҐ,batch_normalization/batchnorm/ReadVariableOpҐ.batch_normalization/batchnorm/ReadVariableOp_1Ґ.batch_normalization/batchnorm/ReadVariableOp_2Ґ0batch_normalization/batchnorm/mul/ReadVariableOpҐ.batch_normalization_1/batchnorm/ReadVariableOpҐ0batch_normalization_1/batchnorm/ReadVariableOp_1Ґ0batch_normalization_1/batchnorm/ReadVariableOp_2Ґ2batch_normalization_1/batchnorm/mul/ReadVariableOpҐ.batch_normalization_2/batchnorm/ReadVariableOpҐ0batch_normalization_2/batchnorm/ReadVariableOp_1Ґ0batch_normalization_2/batchnorm/ReadVariableOp_2Ґ2batch_normalization_2/batchnorm/mul/ReadVariableOpҐ.batch_normalization_3/batchnorm/ReadVariableOpҐ0batch_normalization_3/batchnorm/ReadVariableOp_1Ґ0batch_normalization_3/batchnorm/ReadVariableOp_2Ґ2batch_normalization_3/batchnorm/mul/ReadVariableOpҐdense_1/BiasAdd/ReadVariableOpҐdense_1/MatMul/ReadVariableOpҐ-dense_1/kernel/Regularizer/Abs/ReadVariableOpҐ&stream_0_conv_1/BiasAdd/ReadVariableOpҐ2stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOpҐ5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpҐ&stream_1_conv_1/BiasAdd/ReadVariableOpҐ2stream_1_conv_1/conv1d/ExpandDims_1/ReadVariableOpҐ8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpҐ&stream_2_conv_1/BiasAdd/ReadVariableOpҐ2stream_2_conv_1/conv1d/ExpandDims_1/ReadVariableOpҐ5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOpИ
stream_2_input_drop/IdentityIdentityinputs_2*
T0*+
_output_shapes
:€€€€€€€€€}2
stream_2_input_drop/IdentityИ
stream_1_input_drop/IdentityIdentityinputs_1*
T0*+
_output_shapes
:€€€€€€€€€}2
stream_1_input_drop/IdentityИ
stream_0_input_drop/IdentityIdentityinputs_0*
T0*+
_output_shapes
:€€€€€€€€€}2
stream_0_input_drop/IdentityЩ
%stream_2_conv_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€2'
%stream_2_conv_1/conv1d/ExpandDims/dimе
!stream_2_conv_1/conv1d/ExpandDims
ExpandDims%stream_2_input_drop/Identity:output:0.stream_2_conv_1/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:€€€€€€€€€}2#
!stream_2_conv_1/conv1d/ExpandDimsи
2stream_2_conv_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp;stream_2_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype024
2stream_2_conv_1/conv1d/ExpandDims_1/ReadVariableOpФ
'stream_2_conv_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2)
'stream_2_conv_1/conv1d/ExpandDims_1/dimч
#stream_2_conv_1/conv1d/ExpandDims_1
ExpandDims:stream_2_conv_1/conv1d/ExpandDims_1/ReadVariableOp:value:00stream_2_conv_1/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@2%
#stream_2_conv_1/conv1d/ExpandDims_1ц
stream_2_conv_1/conv1dConv2D*stream_2_conv_1/conv1d/ExpandDims:output:0,stream_2_conv_1/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:€€€€€€€€€}@*
paddingSAME*
strides
2
stream_2_conv_1/conv1d¬
stream_2_conv_1/conv1d/SqueezeSqueezestream_2_conv_1/conv1d:output:0*
T0*+
_output_shapes
:€€€€€€€€€}@*
squeeze_dims

э€€€€€€€€2 
stream_2_conv_1/conv1d/SqueezeЉ
&stream_2_conv_1/BiasAdd/ReadVariableOpReadVariableOp/stream_2_conv_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02(
&stream_2_conv_1/BiasAdd/ReadVariableOpћ
stream_2_conv_1/BiasAddBiasAdd'stream_2_conv_1/conv1d/Squeeze:output:0.stream_2_conv_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€}@2
stream_2_conv_1/BiasAddЩ
%stream_1_conv_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€2'
%stream_1_conv_1/conv1d/ExpandDims/dimе
!stream_1_conv_1/conv1d/ExpandDims
ExpandDims%stream_1_input_drop/Identity:output:0.stream_1_conv_1/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:€€€€€€€€€}2#
!stream_1_conv_1/conv1d/ExpandDimsи
2stream_1_conv_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp;stream_1_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype024
2stream_1_conv_1/conv1d/ExpandDims_1/ReadVariableOpФ
'stream_1_conv_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2)
'stream_1_conv_1/conv1d/ExpandDims_1/dimч
#stream_1_conv_1/conv1d/ExpandDims_1
ExpandDims:stream_1_conv_1/conv1d/ExpandDims_1/ReadVariableOp:value:00stream_1_conv_1/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@2%
#stream_1_conv_1/conv1d/ExpandDims_1ц
stream_1_conv_1/conv1dConv2D*stream_1_conv_1/conv1d/ExpandDims:output:0,stream_1_conv_1/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:€€€€€€€€€}@*
paddingSAME*
strides
2
stream_1_conv_1/conv1d¬
stream_1_conv_1/conv1d/SqueezeSqueezestream_1_conv_1/conv1d:output:0*
T0*+
_output_shapes
:€€€€€€€€€}@*
squeeze_dims

э€€€€€€€€2 
stream_1_conv_1/conv1d/SqueezeЉ
&stream_1_conv_1/BiasAdd/ReadVariableOpReadVariableOp/stream_1_conv_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02(
&stream_1_conv_1/BiasAdd/ReadVariableOpћ
stream_1_conv_1/BiasAddBiasAdd'stream_1_conv_1/conv1d/Squeeze:output:0.stream_1_conv_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€}@2
stream_1_conv_1/BiasAddЩ
%stream_0_conv_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€2'
%stream_0_conv_1/conv1d/ExpandDims/dimе
!stream_0_conv_1/conv1d/ExpandDims
ExpandDims%stream_0_input_drop/Identity:output:0.stream_0_conv_1/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:€€€€€€€€€}2#
!stream_0_conv_1/conv1d/ExpandDimsи
2stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp;stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype024
2stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOpФ
'stream_0_conv_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2)
'stream_0_conv_1/conv1d/ExpandDims_1/dimч
#stream_0_conv_1/conv1d/ExpandDims_1
ExpandDims:stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp:value:00stream_0_conv_1/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@2%
#stream_0_conv_1/conv1d/ExpandDims_1ц
stream_0_conv_1/conv1dConv2D*stream_0_conv_1/conv1d/ExpandDims:output:0,stream_0_conv_1/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:€€€€€€€€€}@*
paddingSAME*
strides
2
stream_0_conv_1/conv1d¬
stream_0_conv_1/conv1d/SqueezeSqueezestream_0_conv_1/conv1d:output:0*
T0*+
_output_shapes
:€€€€€€€€€}@*
squeeze_dims

э€€€€€€€€2 
stream_0_conv_1/conv1d/SqueezeЉ
&stream_0_conv_1/BiasAdd/ReadVariableOpReadVariableOp/stream_0_conv_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02(
&stream_0_conv_1/BiasAdd/ReadVariableOpћ
stream_0_conv_1/BiasAddBiasAdd'stream_0_conv_1/conv1d/Squeeze:output:0.stream_0_conv_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€}@2
stream_0_conv_1/BiasAdd‘
.batch_normalization_2/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_2_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype020
.batch_normalization_2/batchnorm/ReadVariableOpУ
%batch_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2'
%batch_normalization_2/batchnorm/add/yа
#batch_normalization_2/batchnorm/addAddV26batch_normalization_2/batchnorm/ReadVariableOp:value:0.batch_normalization_2/batchnorm/add/y:output:0*
T0*
_output_shapes
:@2%
#batch_normalization_2/batchnorm/add•
%batch_normalization_2/batchnorm/RsqrtRsqrt'batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes
:@2'
%batch_normalization_2/batchnorm/Rsqrtа
2batch_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype024
2batch_normalization_2/batchnorm/mul/ReadVariableOpЁ
#batch_normalization_2/batchnorm/mulMul)batch_normalization_2/batchnorm/Rsqrt:y:0:batch_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2%
#batch_normalization_2/batchnorm/mul÷
%batch_normalization_2/batchnorm/mul_1Mul stream_2_conv_1/BiasAdd:output:0'batch_normalization_2/batchnorm/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€}@2'
%batch_normalization_2/batchnorm/mul_1Џ
0batch_normalization_2/batchnorm/ReadVariableOp_1ReadVariableOp9batch_normalization_2_batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype022
0batch_normalization_2/batchnorm/ReadVariableOp_1Ё
%batch_normalization_2/batchnorm/mul_2Mul8batch_normalization_2/batchnorm/ReadVariableOp_1:value:0'batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes
:@2'
%batch_normalization_2/batchnorm/mul_2Џ
0batch_normalization_2/batchnorm/ReadVariableOp_2ReadVariableOp9batch_normalization_2_batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype022
0batch_normalization_2/batchnorm/ReadVariableOp_2џ
#batch_normalization_2/batchnorm/subSub8batch_normalization_2/batchnorm/ReadVariableOp_2:value:0)batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2%
#batch_normalization_2/batchnorm/subб
%batch_normalization_2/batchnorm/add_1AddV2)batch_normalization_2/batchnorm/mul_1:z:0'batch_normalization_2/batchnorm/sub:z:0*
T0*+
_output_shapes
:€€€€€€€€€}@2'
%batch_normalization_2/batchnorm/add_1‘
.batch_normalization_1/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_1_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype020
.batch_normalization_1/batchnorm/ReadVariableOpУ
%batch_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2'
%batch_normalization_1/batchnorm/add/yа
#batch_normalization_1/batchnorm/addAddV26batch_normalization_1/batchnorm/ReadVariableOp:value:0.batch_normalization_1/batchnorm/add/y:output:0*
T0*
_output_shapes
:@2%
#batch_normalization_1/batchnorm/add•
%batch_normalization_1/batchnorm/RsqrtRsqrt'batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes
:@2'
%batch_normalization_1/batchnorm/Rsqrtа
2batch_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype024
2batch_normalization_1/batchnorm/mul/ReadVariableOpЁ
#batch_normalization_1/batchnorm/mulMul)batch_normalization_1/batchnorm/Rsqrt:y:0:batch_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2%
#batch_normalization_1/batchnorm/mul÷
%batch_normalization_1/batchnorm/mul_1Mul stream_1_conv_1/BiasAdd:output:0'batch_normalization_1/batchnorm/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€}@2'
%batch_normalization_1/batchnorm/mul_1Џ
0batch_normalization_1/batchnorm/ReadVariableOp_1ReadVariableOp9batch_normalization_1_batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype022
0batch_normalization_1/batchnorm/ReadVariableOp_1Ё
%batch_normalization_1/batchnorm/mul_2Mul8batch_normalization_1/batchnorm/ReadVariableOp_1:value:0'batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes
:@2'
%batch_normalization_1/batchnorm/mul_2Џ
0batch_normalization_1/batchnorm/ReadVariableOp_2ReadVariableOp9batch_normalization_1_batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype022
0batch_normalization_1/batchnorm/ReadVariableOp_2џ
#batch_normalization_1/batchnorm/subSub8batch_normalization_1/batchnorm/ReadVariableOp_2:value:0)batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2%
#batch_normalization_1/batchnorm/subб
%batch_normalization_1/batchnorm/add_1AddV2)batch_normalization_1/batchnorm/mul_1:z:0'batch_normalization_1/batchnorm/sub:z:0*
T0*+
_output_shapes
:€€€€€€€€€}@2'
%batch_normalization_1/batchnorm/add_1ќ
,batch_normalization/batchnorm/ReadVariableOpReadVariableOp5batch_normalization_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02.
,batch_normalization/batchnorm/ReadVariableOpП
#batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2%
#batch_normalization/batchnorm/add/yЎ
!batch_normalization/batchnorm/addAddV24batch_normalization/batchnorm/ReadVariableOp:value:0,batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes
:@2#
!batch_normalization/batchnorm/addЯ
#batch_normalization/batchnorm/RsqrtRsqrt%batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes
:@2%
#batch_normalization/batchnorm/RsqrtЏ
0batch_normalization/batchnorm/mul/ReadVariableOpReadVariableOp9batch_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype022
0batch_normalization/batchnorm/mul/ReadVariableOp’
!batch_normalization/batchnorm/mulMul'batch_normalization/batchnorm/Rsqrt:y:08batch_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2#
!batch_normalization/batchnorm/mul–
#batch_normalization/batchnorm/mul_1Mul stream_0_conv_1/BiasAdd:output:0%batch_normalization/batchnorm/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€}@2%
#batch_normalization/batchnorm/mul_1‘
.batch_normalization/batchnorm/ReadVariableOp_1ReadVariableOp7batch_normalization_batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype020
.batch_normalization/batchnorm/ReadVariableOp_1’
#batch_normalization/batchnorm/mul_2Mul6batch_normalization/batchnorm/ReadVariableOp_1:value:0%batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes
:@2%
#batch_normalization/batchnorm/mul_2‘
.batch_normalization/batchnorm/ReadVariableOp_2ReadVariableOp7batch_normalization_batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype020
.batch_normalization/batchnorm/ReadVariableOp_2”
!batch_normalization/batchnorm/subSub6batch_normalization/batchnorm/ReadVariableOp_2:value:0'batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2#
!batch_normalization/batchnorm/subў
#batch_normalization/batchnorm/add_1AddV2'batch_normalization/batchnorm/mul_1:z:0%batch_normalization/batchnorm/sub:z:0*
T0*+
_output_shapes
:€€€€€€€€€}@2%
#batch_normalization/batchnorm/add_1П
activation_2/ReluRelu)batch_normalization_2/batchnorm/add_1:z:0*
T0*+
_output_shapes
:€€€€€€€€€}@2
activation_2/ReluП
activation_1/ReluRelu)batch_normalization_1/batchnorm/add_1:z:0*
T0*+
_output_shapes
:€€€€€€€€€}@2
activation_1/ReluЙ
activation/ReluRelu'batch_normalization/batchnorm/add_1:z:0*
T0*+
_output_shapes
:€€€€€€€€€}@2
activation/ReluЧ
stream_2_drop_1/IdentityIdentityactivation_2/Relu:activations:0*
T0*+
_output_shapes
:€€€€€€€€€}@2
stream_2_drop_1/IdentityЧ
stream_1_drop_1/IdentityIdentityactivation_1/Relu:activations:0*
T0*+
_output_shapes
:€€€€€€€€€}@2
stream_1_drop_1/IdentityХ
stream_0_drop_1/IdentityIdentityactivation/Relu:activations:0*
T0*+
_output_shapes
:€€€€€€€€€}@2
stream_0_drop_1/Identity§
/global_average_pooling1d/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :21
/global_average_pooling1d/Mean/reduction_indices’
global_average_pooling1d/MeanMean!stream_0_drop_1/Identity:output:08global_average_pooling1d/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
global_average_pooling1d/Mean®
1global_average_pooling1d_1/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :23
1global_average_pooling1d_1/Mean/reduction_indicesџ
global_average_pooling1d_1/MeanMean!stream_1_drop_1/Identity:output:0:global_average_pooling1d_1/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2!
global_average_pooling1d_1/Mean®
1global_average_pooling1d_2/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :23
1global_average_pooling1d_2/Mean/reduction_indicesџ
global_average_pooling1d_2/MeanMean!stream_2_drop_1/Identity:output:0:global_average_pooling1d_2/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2!
global_average_pooling1d_2/Meant
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate/concat/axisО
concatenate/concatConcatV2&global_average_pooling1d/Mean:output:0(global_average_pooling1d_1/Mean:output:0(global_average_pooling1d_2/Mean:output:0 concatenate/concat/axis:output:0*
N*
T0*(
_output_shapes
:€€€€€€€€€ј2
concatenate/concatР
dense_1_dropout/IdentityIdentityconcatenate/concat:output:0*
T0*(
_output_shapes
:€€€€€€€€€ј2
dense_1_dropout/Identity¶
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	јT*
dtype02
dense_1/MatMul/ReadVariableOp¶
dense_1/MatMulMatMul!dense_1_dropout/Identity:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€T2
dense_1/MatMul§
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:T*
dtype02 
dense_1/BiasAdd/ReadVariableOp°
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€T2
dense_1/BiasAdd‘
.batch_normalization_3/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_3_batchnorm_readvariableop_resource*
_output_shapes
:T*
dtype020
.batch_normalization_3/batchnorm/ReadVariableOpУ
%batch_normalization_3/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2'
%batch_normalization_3/batchnorm/add/yа
#batch_normalization_3/batchnorm/addAddV26batch_normalization_3/batchnorm/ReadVariableOp:value:0.batch_normalization_3/batchnorm/add/y:output:0*
T0*
_output_shapes
:T2%
#batch_normalization_3/batchnorm/add•
%batch_normalization_3/batchnorm/RsqrtRsqrt'batch_normalization_3/batchnorm/add:z:0*
T0*
_output_shapes
:T2'
%batch_normalization_3/batchnorm/Rsqrtа
2batch_normalization_3/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_3_batchnorm_mul_readvariableop_resource*
_output_shapes
:T*
dtype024
2batch_normalization_3/batchnorm/mul/ReadVariableOpЁ
#batch_normalization_3/batchnorm/mulMul)batch_normalization_3/batchnorm/Rsqrt:y:0:batch_normalization_3/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:T2%
#batch_normalization_3/batchnorm/mul 
%batch_normalization_3/batchnorm/mul_1Muldense_1/BiasAdd:output:0'batch_normalization_3/batchnorm/mul:z:0*
T0*'
_output_shapes
:€€€€€€€€€T2'
%batch_normalization_3/batchnorm/mul_1Џ
0batch_normalization_3/batchnorm/ReadVariableOp_1ReadVariableOp9batch_normalization_3_batchnorm_readvariableop_1_resource*
_output_shapes
:T*
dtype022
0batch_normalization_3/batchnorm/ReadVariableOp_1Ё
%batch_normalization_3/batchnorm/mul_2Mul8batch_normalization_3/batchnorm/ReadVariableOp_1:value:0'batch_normalization_3/batchnorm/mul:z:0*
T0*
_output_shapes
:T2'
%batch_normalization_3/batchnorm/mul_2Џ
0batch_normalization_3/batchnorm/ReadVariableOp_2ReadVariableOp9batch_normalization_3_batchnorm_readvariableop_2_resource*
_output_shapes
:T*
dtype022
0batch_normalization_3/batchnorm/ReadVariableOp_2џ
#batch_normalization_3/batchnorm/subSub8batch_normalization_3/batchnorm/ReadVariableOp_2:value:0)batch_normalization_3/batchnorm/mul_2:z:0*
T0*
_output_shapes
:T2%
#batch_normalization_3/batchnorm/subЁ
%batch_normalization_3/batchnorm/add_1AddV2)batch_normalization_3/batchnorm/mul_1:z:0'batch_normalization_3/batchnorm/sub:z:0*
T0*'
_output_shapes
:€€€€€€€€€T2'
%batch_normalization_3/batchnorm/add_1†
dense_activation_1/SigmoidSigmoid)batch_normalization_3/batchnorm/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€T2
dense_activation_1/Sigmoidо
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp;stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype027
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp√
&stream_0_conv_1/kernel/Regularizer/AbsAbs=stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_0_conv_1/kernel/Regularizer/Abs©
(stream_0_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_0_conv_1/kernel/Regularizer/Const„
&stream_0_conv_1/kernel/Regularizer/SumSum*stream_0_conv_1/kernel/Regularizer/Abs:y:01stream_0_conv_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/SumЩ
(stream_0_conv_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2*
(stream_0_conv_1/kernel/Regularizer/mul/x№
&stream_0_conv_1/kernel/Regularizer/mulMul1stream_0_conv_1/kernel/Regularizer/mul/x:output:0/stream_0_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/mulф
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;stream_1_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02:
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpѕ
)stream_1_conv_1/kernel/Regularizer/SquareSquare@stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2+
)stream_1_conv_1/kernel/Regularizer/Square©
(stream_1_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_1_conv_1/kernel/Regularizer/ConstЏ
&stream_1_conv_1/kernel/Regularizer/SumSum-stream_1_conv_1/kernel/Regularizer/Square:y:01stream_1_conv_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&stream_1_conv_1/kernel/Regularizer/SumЩ
(stream_1_conv_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2*
(stream_1_conv_1/kernel/Regularizer/mul/x№
&stream_1_conv_1/kernel/Regularizer/mulMul1stream_1_conv_1/kernel/Regularizer/mul/x:output:0/stream_1_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_1_conv_1/kernel/Regularizer/mulо
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp;stream_2_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype027
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp√
&stream_2_conv_1/kernel/Regularizer/AbsAbs=stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_2_conv_1/kernel/Regularizer/Abs©
(stream_2_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_2_conv_1/kernel/Regularizer/Const„
&stream_2_conv_1/kernel/Regularizer/SumSum*stream_2_conv_1/kernel/Regularizer/Abs:y:01stream_2_conv_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&stream_2_conv_1/kernel/Regularizer/SumЩ
(stream_2_conv_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2*
(stream_2_conv_1/kernel/Regularizer/mul/x№
&stream_2_conv_1/kernel/Regularizer/mulMul1stream_2_conv_1/kernel/Regularizer/mul/x:output:0/stream_2_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_2_conv_1/kernel/Regularizer/mul∆
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	јT*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOp®
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	јT2 
dense_1/kernel/Regularizer/AbsХ
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_1/kernel/Regularizer/ConstЈ
dense_1/kernel/Regularizer/SumSum"dense_1/kernel/Regularizer/Abs:y:0)dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/SumЙ
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2"
 dense_1/kernel/Regularizer/mul/xЉ
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/muly
IdentityIdentitydense_activation_1/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€T2

Identityђ
NoOpNoOp-^batch_normalization/batchnorm/ReadVariableOp/^batch_normalization/batchnorm/ReadVariableOp_1/^batch_normalization/batchnorm/ReadVariableOp_21^batch_normalization/batchnorm/mul/ReadVariableOp/^batch_normalization_1/batchnorm/ReadVariableOp1^batch_normalization_1/batchnorm/ReadVariableOp_11^batch_normalization_1/batchnorm/ReadVariableOp_23^batch_normalization_1/batchnorm/mul/ReadVariableOp/^batch_normalization_2/batchnorm/ReadVariableOp1^batch_normalization_2/batchnorm/ReadVariableOp_11^batch_normalization_2/batchnorm/ReadVariableOp_23^batch_normalization_2/batchnorm/mul/ReadVariableOp/^batch_normalization_3/batchnorm/ReadVariableOp1^batch_normalization_3/batchnorm/ReadVariableOp_11^batch_normalization_3/batchnorm/ReadVariableOp_23^batch_normalization_3/batchnorm/mul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp.^dense_1/kernel/Regularizer/Abs/ReadVariableOp'^stream_0_conv_1/BiasAdd/ReadVariableOp3^stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp'^stream_1_conv_1/BiasAdd/ReadVariableOp3^stream_1_conv_1/conv1d/ExpandDims_1/ReadVariableOp9^stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp'^stream_2_conv_1/BiasAdd/ReadVariableOp3^stream_2_conv_1/conv1d/ExpandDims_1/ReadVariableOp6^stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*И
_input_shapesw
u:€€€€€€€€€}:€€€€€€€€€}:€€€€€€€€€}: : : : : : : : : : : : : : : : : : : : : : : : 2\
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
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp:U Q
+
_output_shapes
:€€€€€€€€€}
"
_user_specified_name
inputs/0:UQ
+
_output_shapes
:€€€€€€€€€}
"
_user_specified_name
inputs/1:UQ
+
_output_shapes
:€€€€€€€€€}
"
_user_specified_name
inputs/2
ґ
Б
*__inference_basemodel_layer_call_fn_107603
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

unknown_17:	јT

unknown_18:T

unknown_19:T

unknown_20:T

unknown_21:T

unknown_22:T
identityИҐStatefulPartitionedCallї
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
:€€€€€€€€€T*:
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_basemodel_layer_call_and_return_conditional_losses_1046122
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€T2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*И
_input_shapesw
u:€€€€€€€€€}:€€€€€€€€€}:€€€€€€€€€}: : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
+
_output_shapes
:€€€€€€€€€}
"
_user_specified_name
inputs/0:UQ
+
_output_shapes
:€€€€€€€€€}
"
_user_specified_name
inputs/1:UQ
+
_output_shapes
:€€€€€€€€€}
"
_user_specified_name
inputs/2
н
j
K__inference_stream_0_drop_1_layer_call_and_return_conditional_losses_104737

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †?2
dropout/Constw
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:€€€€€€€€€}@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape”
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:€€€€€€€€€}@*
dtype0*
seedЈ*
seed2Ј2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL>2
dropout/GreaterEqual/y¬
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:€€€€€€€€€}@2
dropout/GreaterEqualГ
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:€€€€€€€€€}@2
dropout/Cast~
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:€€€€€€€€€}@2
dropout/Mul_1i
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:€€€€€€€€€}@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€}@:S O
+
_output_shapes
:€€€€€€€€€}@
 
_user_specified_nameinputs
р
d
H__inference_activation_1_layer_call_and_return_conditional_losses_108378

inputs
identityR
ReluReluinputs*
T0*+
_output_shapes
:€€€€€€€€€}@2
Reluj
IdentityIdentityRelu:activations:0*
T0*+
_output_shapes
:€€€€€€€€€}@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€}@:S O
+
_output_shapes
:€€€€€€€€€}@
 
_user_specified_nameinputs
ƒ
U
)__inference_distance_layer_call_fn_107694
inputs_0
inputs_1
identity“
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_distance_layer_call_and_return_conditional_losses_1057462
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:€€€€€€€€€T:€€€€€€€€€T:Q M
'
_output_shapes
:€€€€€€€€€T
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:€€€€€€€€€T
"
_user_specified_name
inputs/1
М
m
O__inference_stream_1_input_drop_layer_call_and_return_conditional_losses_104292

inputs

identity_1^
IdentityIdentityinputs*
T0*+
_output_shapes
:€€€€€€€€€}2

Identitym

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:€€€€€€€€€}2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€}:S O
+
_output_shapes
:€€€€€€€€€}
 
_user_specified_nameinputs
З
Ѓ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_104459

inputs/
!batchnorm_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@1
#batchnorm_readvariableop_1_resource:@1
#batchnorm_readvariableop_2_resource:@
identityИҐbatchnorm/ReadVariableOpҐbatchnorm/ReadVariableOp_1Ґbatchnorm/ReadVariableOp_2Ґbatchnorm/mul/ReadVariableOpТ
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
 *oГ:2
batchnorm/add/yИ
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2
batchnorm/mulz
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€}@2
batchnorm/mul_1Ш
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp_1Е
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:@2
batchnorm/mul_2Ш
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp_2Г
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2
batchnorm/subЙ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*+
_output_shapes
:€€€€€€€€€}@2
batchnorm/add_1r
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€}@2

Identity¬
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€}@: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:S O
+
_output_shapes
:€€€€€€€€€}@
 
_user_specified_nameinputs
М
m
O__inference_stream_1_input_drop_layer_call_and_return_conditional_losses_107726

inputs

identity_1^
IdentityIdentityinputs*
T0*+
_output_shapes
:€€€€€€€€€}2

Identitym

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:€€€€€€€€€}2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€}:S O
+
_output_shapes
:€€€€€€€€€}
 
_user_specified_nameinputs
с
n
O__inference_stream_2_input_drop_layer_call_and_return_conditional_losses_107765

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †?2
dropout/Constw
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:€€€€€€€€€}2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape”
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:€€€€€€€€€}*
dtype0*
seedЈ*
seed2Ј2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL>2
dropout/GreaterEqual/y¬
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:€€€€€€€€€}2
dropout/GreaterEqualГ
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:€€€€€€€€€}2
dropout/Cast~
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:€€€€€€€€€}2
dropout/Mul_1i
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:€€€€€€€€€}2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€}:S O
+
_output_shapes
:€€€€€€€€€}
 
_user_specified_nameinputs
е
P
4__inference_stream_1_input_drop_layer_call_fn_107743

inputs
identity‘
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€}* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_stream_1_input_drop_layer_call_and_return_conditional_losses_1042922
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:€€€€€€€€€}2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€}:S O
+
_output_shapes
:€€€€€€€€€}
 
_user_specified_nameinputs
є	
n
D__inference_distance_layer_call_and_return_conditional_losses_105746

inputs
inputs_1
identityU
subSubinputsinputs_1*
T0*'
_output_shapes
:€€€€€€€€€T2
subU
SquareSquaresub:z:0*
T0*'
_output_shapes
:€€€€€€€€€T2
Squarey
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2
Sum/reduction_indicesА
SumSum
Square:y:0Sum/reduction_indices:output:0*
T0*'
_output_shapes
:€€€€€€€€€*
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
:€€€€€€€€€2	
MaximumS
SqrtSqrtMaximum:z:0*
T0*'
_output_shapes
:€€€€€€€€€2
Sqrt\
IdentityIdentitySqrt:y:0*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:€€€€€€€€€T:€€€€€€€€€T:O K
'
_output_shapes
:€€€€€€€€€T
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€T
 
_user_specified_nameinputs
Ц
°
0__inference_stream_1_conv_1_layer_call_fn_107847

inputs
unknown:@
	unknown_0:@
identityИҐStatefulPartitionedCallВ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€}@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_stream_1_conv_1_layer_call_and_return_conditional_losses_1043492
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€}@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€}: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:€€€€€€€€€}
 
_user_specified_nameinputs
И
i
K__inference_stream_2_drop_1_layer_call_and_return_conditional_losses_108452

inputs

identity_1^
IdentityIdentityinputs*
T0*+
_output_shapes
:€€€€€€€€€}@2

Identitym

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:€€€€€€€€€}@2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€}@:S O
+
_output_shapes
:€€€€€€€€€}@
 
_user_specified_nameinputs
Б+
к
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_108311

inputs5
'assignmovingavg_readvariableop_resource:@7
)assignmovingavg_1_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@/
!batchnorm_readvariableop_resource:@
identityИҐAssignMovingAvgҐAssignMovingAvg/ReadVariableOpҐAssignMovingAvg_1Ґ AssignMovingAvg_1/ReadVariableOpҐbatchnorm/ReadVariableOpҐbatchnorm/mul/ReadVariableOpС
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indicesУ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(2
moments/meanА
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:@2
moments/StopGradient®
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*+
_output_shapes
:€€€€€€€€€}@2
moments/SquaredDifferenceЩ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indicesґ
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(2
moments/varianceБ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 2
moments/SqueezeЙ
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
„#<2
AssignMovingAvg/decay§
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOpШ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg/subП
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg/mulњ
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
„#<2
AssignMovingAvg_1/decay™
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOp†
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg_1/subЧ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg_1/mul…
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
 *oГ:2
batchnorm/add/yВ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2
batchnorm/mulz
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€}@2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:@2
batchnorm/mul_2Т
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOpБ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2
batchnorm/subЙ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*+
_output_shapes
:€€€€€€€€€}@2
batchnorm/add_1r
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€}@2

Identityт
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€}@: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:S O
+
_output_shapes
:€€€€€€€€€}@
 
_user_specified_nameinputs
Ѓ
Б
*__inference_basemodel_layer_call_fn_105349
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

unknown_17:	јT

unknown_18:T

unknown_19:T

unknown_20:T

unknown_21:T

unknown_22:T
identityИҐStatefulPartitionedCall≥
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
:€€€€€€€€€T*2
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_basemodel_layer_call_and_return_conditional_losses_1052432
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€T2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*И
_input_shapesw
u:€€€€€€€€€}:€€€€€€€€€}:€€€€€€€€€}: : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
+
_output_shapes
:€€€€€€€€€}
"
_user_specified_name
inputs_0:UQ
+
_output_shapes
:€€€€€€€€€}
"
_user_specified_name
inputs_1:UQ
+
_output_shapes
:€€€€€€€€€}
"
_user_specified_name
inputs_2
Є+
к
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_103957

inputs5
'assignmovingavg_readvariableop_resource:@7
)assignmovingavg_1_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@/
!batchnorm_readvariableop_resource:@
identityИҐAssignMovingAvgҐAssignMovingAvg/ReadVariableOpҐAssignMovingAvg_1Ґ AssignMovingAvg_1/ReadVariableOpҐbatchnorm/ReadVariableOpҐbatchnorm/mul/ReadVariableOpС
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indicesУ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(2
moments/meanА
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:@2
moments/StopGradient±
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€@2
moments/SquaredDifferenceЩ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indicesґ
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(2
moments/varianceБ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 2
moments/SqueezeЙ
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
„#<2
AssignMovingAvg/decay§
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOpШ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg/subП
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg/mulњ
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
„#<2
AssignMovingAvg_1/decay™
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOp†
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg_1/subЧ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg_1/mul…
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
 *oГ:2
batchnorm/add/yВ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2
batchnorm/mulГ
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€@2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:@2
batchnorm/mul_2Т
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOpБ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2
batchnorm/subТ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€@2
batchnorm/add_1{
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€@2

Identityт
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:€€€€€€€€€€€€€€€€€€@: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs
ь
i
K__inference_dense_1_dropout_layer_call_and_return_conditional_losses_108560

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:€€€€€€€€€ј2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:€€€€€€€€€ј2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:€€€€€€€€€ј:P L
(
_output_shapes
:€€€€€€€€€ј
 
_user_specified_nameinputs
Й	
ѕ
4__inference_batch_normalization_layer_call_fn_108017

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityИҐStatefulPartitionedCallІ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_1036332
StatefulPartitionedCallИ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:€€€€€€€€€€€€€€€€€€@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs
р
d
H__inference_activation_1_layer_call_and_return_conditional_losses_104481

inputs
identityR
ReluReluinputs*
T0*+
_output_shapes
:€€€€€€€€€}@2
Reluj
IdentityIdentityRelu:activations:0*
T0*+
_output_shapes
:€€€€€€€€€}@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€}@:S O
+
_output_shapes
:€€€€€€€€€}@
 
_user_specified_nameinputs
и
—
6__inference_batch_normalization_2_layer_call_fn_108363

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityИҐStatefulPartitionedCall†
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€}@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1049732
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€}@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€}@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:€€€€€€€€€}@
 
_user_specified_nameinputs
Ј
∞
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_108063

inputs/
!batchnorm_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@1
#batchnorm_readvariableop_1_resource:@1
#batchnorm_readvariableop_2_resource:@
identityИҐbatchnorm/ReadVariableOpҐbatchnorm/ReadVariableOp_1Ґbatchnorm/ReadVariableOp_2Ґbatchnorm/mul/ReadVariableOpТ
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
 *oГ:2
batchnorm/add/yИ
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2
batchnorm/mulГ
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€@2
batchnorm/mul_1Ш
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp_1Е
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:@2
batchnorm/mul_2Ш
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp_2Г
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2
batchnorm/subТ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€@2
batchnorm/add_1{
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€@2

Identity¬
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:€€€€€€€€€€€€€€€€€€@: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs
Й
∞
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_104430

inputs/
!batchnorm_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@1
#batchnorm_readvariableop_1_resource:@1
#batchnorm_readvariableop_2_resource:@
identityИҐbatchnorm/ReadVariableOpҐbatchnorm/ReadVariableOp_1Ґbatchnorm/ReadVariableOp_2Ґbatchnorm/mul/ReadVariableOpТ
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
 *oГ:2
batchnorm/add/yИ
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2
batchnorm/mulz
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€}@2
batchnorm/mul_1Ш
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp_1Е
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:@2
batchnorm/mul_2Ш
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp_2Г
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2
batchnorm/subЙ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*+
_output_shapes
:€€€€€€€€€}@2
batchnorm/add_1r
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€}@2

Identity¬
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€}@: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:S O
+
_output_shapes
:€€€€€€€€€}@
 
_user_specified_nameinputs
о
b
F__inference_activation_layer_call_and_return_conditional_losses_104488

inputs
identityR
ReluReluinputs*
T0*+
_output_shapes
:€€€€€€€€€}@2
Reluj
IdentityIdentityRelu:activations:0*
T0*+
_output_shapes
:€€€€€€€€€}@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€}@:S O
+
_output_shapes
:€€€€€€€€€}@
 
_user_specified_nameinputs
 
љ
__inference_loss_fn_0_108706T
>stream_0_conv_1_kernel_regularizer_abs_readvariableop_resource:@
identityИҐ5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpс
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp>stream_0_conv_1_kernel_regularizer_abs_readvariableop_resource*"
_output_shapes
:@*
dtype027
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp√
&stream_0_conv_1/kernel/Regularizer/AbsAbs=stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_0_conv_1/kernel/Regularizer/Abs©
(stream_0_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_0_conv_1/kernel/Regularizer/Const„
&stream_0_conv_1/kernel/Regularizer/SumSum*stream_0_conv_1/kernel/Regularizer/Abs:y:01stream_0_conv_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/SumЩ
(stream_0_conv_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2*
(stream_0_conv_1/kernel/Regularizer/mul/x№
&stream_0_conv_1/kernel/Regularizer/mulMul1stream_0_conv_1/kernel/Regularizer/mul/x:output:0/stream_0_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/mult
IdentityIdentity*stream_0_conv_1/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: 2

IdentityЖ
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
ь
™
__inference_loss_fn_3_108739I
6dense_1_kernel_regularizer_abs_readvariableop_resource:	јT
identityИҐ-dense_1/kernel/Regularizer/Abs/ReadVariableOp÷
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp6dense_1_kernel_regularizer_abs_readvariableop_resource*
_output_shapes
:	јT*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOp®
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	јT2 
dense_1/kernel/Regularizer/AbsХ
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_1/kernel/Regularizer/ConstЈ
dense_1/kernel/Regularizer/SumSum"dense_1/kernel/Regularizer/Abs:y:0)dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/SumЙ
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2"
 dense_1/kernel/Regularizer/mul/xЉ
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
ї
ц
&__inference_model_layer_call_fn_105724
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

unknown_17:	јT

unknown_18:T

unknown_19:T

unknown_20:T

unknown_21:T

unknown_22:T
identityИҐStatefulPartitionedCall≥
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
:€€€€€€€€€*:
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_1056732
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*q
_input_shapes`
^:€€€€€€€€€}:€€€€€€€€€}: : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
+
_output_shapes
:€€€€€€€€€}
%
_user_specified_nameleft_inputs:YU
+
_output_shapes
:€€€€€€€€€}
&
_user_specified_nameright_inputs
М
m
O__inference_stream_0_input_drop_layer_call_and_return_conditional_losses_107699

inputs

identity_1^
IdentityIdentityinputs*
T0*+
_output_shapes
:€€€€€€€€€}2

Identitym

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:€€€€€€€€€}2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€}:S O
+
_output_shapes
:€€€€€€€€€}
 
_user_specified_nameinputs
«
m
4__inference_stream_2_input_drop_layer_call_fn_107775

inputs
identityИҐStatefulPartitionedCallм
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€}* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_stream_2_input_drop_layer_call_and_return_conditional_losses_1050802
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€}2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€}22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:€€€€€€€€€}
 
_user_specified_nameinputs
ƒ
U
)__inference_distance_layer_call_fn_107688
inputs_0
inputs_1
identity“
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_distance_layer_call_and_return_conditional_losses_1056462
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:€€€€€€€€€T:€€€€€€€€€T:Q M
'
_output_shapes
:€€€€€€€€€T
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:€€€€€€€€€T
"
_user_specified_name
inputs/1
µ
Ѓ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_103573

inputs/
!batchnorm_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@1
#batchnorm_readvariableop_1_resource:@1
#batchnorm_readvariableop_2_resource:@
identityИҐbatchnorm/ReadVariableOpҐbatchnorm/ReadVariableOp_1Ґbatchnorm/ReadVariableOp_2Ґbatchnorm/mul/ReadVariableOpТ
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
 *oГ:2
batchnorm/add/yИ
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2
batchnorm/mulГ
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€@2
batchnorm/mul_1Ш
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp_1Е
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:@2
batchnorm/mul_2Ш
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp_2Г
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2
batchnorm/subТ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€@2
batchnorm/add_1{
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€@2

Identity¬
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:€€€€€€€€€€€€€€€€€€@: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs
ґ+
и
O__inference_batch_normalization_layer_call_and_return_conditional_losses_107937

inputs5
'assignmovingavg_readvariableop_resource:@7
)assignmovingavg_1_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@/
!batchnorm_readvariableop_resource:@
identityИҐAssignMovingAvgҐAssignMovingAvg/ReadVariableOpҐAssignMovingAvg_1Ґ AssignMovingAvg_1/ReadVariableOpҐbatchnorm/ReadVariableOpҐbatchnorm/mul/ReadVariableOpС
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indicesУ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(2
moments/meanА
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:@2
moments/StopGradient±
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€@2
moments/SquaredDifferenceЩ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indicesґ
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(2
moments/varianceБ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 2
moments/SqueezeЙ
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
„#<2
AssignMovingAvg/decay§
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOpШ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg/subП
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg/mulњ
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
„#<2
AssignMovingAvg_1/decay™
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOp†
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg_1/subЧ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg_1/mul…
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
 *oГ:2
batchnorm/add/yВ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2
batchnorm/mulГ
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€@2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:@2
batchnorm/mul_2Т
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOpБ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2
batchnorm/subТ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€@2
batchnorm/add_1{
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€@2

Identityт
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:€€€€€€€€€€€€€€€€€€@: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs
Э
U
9__inference_global_average_pooling1d_layer_call_fn_108491

inputs
identityё
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *]
fXRV
T__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_1040452
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€:e a
=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
Ѕ	
p
D__inference_distance_layer_call_and_return_conditional_losses_107670
inputs_0
inputs_1
identityW
subSubinputs_0inputs_1*
T0*'
_output_shapes
:€€€€€€€€€T2
subU
SquareSquaresub:z:0*
T0*'
_output_shapes
:€€€€€€€€€T2
Squarey
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2
Sum/reduction_indicesА
SumSum
Square:y:0Sum/reduction_indices:output:0*
T0*'
_output_shapes
:€€€€€€€€€*
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
:€€€€€€€€€2	
MaximumS
SqrtSqrtMaximum:z:0*
T0*'
_output_shapes
:€€€€€€€€€2
Sqrt\
IdentityIdentitySqrt:y:0*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:€€€€€€€€€T:€€€€€€€€€T:Q M
'
_output_shapes
:€€€€€€€€€T
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:€€€€€€€€€T
"
_user_specified_name
inputs/1
Й
∞
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_108277

inputs/
!batchnorm_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@1
#batchnorm_readvariableop_1_resource:@1
#batchnorm_readvariableop_2_resource:@
identityИҐbatchnorm/ReadVariableOpҐbatchnorm/ReadVariableOp_1Ґbatchnorm/ReadVariableOp_2Ґbatchnorm/mul/ReadVariableOpТ
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
 *oГ:2
batchnorm/add/yИ
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2
batchnorm/mulz
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€}@2
batchnorm/mul_1Ш
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp_1Е
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:@2
batchnorm/mul_2Ш
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp_2Г
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2
batchnorm/subЙ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*+
_output_shapes
:€€€€€€€€€}@2
batchnorm/add_1r
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€}@2

Identity¬
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€}@: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:S O
+
_output_shapes
:€€€€€€€€€}@
 
_user_specified_nameinputs
Ї
p
T__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_104045

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
:€€€€€€€€€€€€€€€€€€2
Meanj
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€:e a
=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
Ў
—
6__inference_batch_normalization_3_layer_call_fn_108685

inputs
unknown:T
	unknown_0:T
	unknown_1:T
	unknown_2:T
identityИҐStatefulPartitionedCallЬ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€T*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1041912
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€T2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€T: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€T
 
_user_specified_nameinputs
Є+
к
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_108257

inputs5
'assignmovingavg_readvariableop_resource:@7
)assignmovingavg_1_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@/
!batchnorm_readvariableop_resource:@
identityИҐAssignMovingAvgҐAssignMovingAvg/ReadVariableOpҐAssignMovingAvg_1Ґ AssignMovingAvg_1/ReadVariableOpҐbatchnorm/ReadVariableOpҐbatchnorm/mul/ReadVariableOpС
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indicesУ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(2
moments/meanА
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:@2
moments/StopGradient±
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€@2
moments/SquaredDifferenceЩ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indicesґ
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(2
moments/varianceБ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 2
moments/SqueezeЙ
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
„#<2
AssignMovingAvg/decay§
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOpШ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg/subП
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg/mulњ
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
„#<2
AssignMovingAvg_1/decay™
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOp†
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg_1/subЧ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg_1/mul…
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
 *oГ:2
batchnorm/add/yВ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2
batchnorm/mulГ
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€@2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:@2
batchnorm/mul_2Т
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOpБ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2
batchnorm/subТ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€@2
batchnorm/add_1{
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€@2

Identityт
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:€€€€€€€€€€€€€€€€€€@: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs
Б+
к
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_108151

inputs5
'assignmovingavg_readvariableop_resource:@7
)assignmovingavg_1_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@/
!batchnorm_readvariableop_resource:@
identityИҐAssignMovingAvgҐAssignMovingAvg/ReadVariableOpҐAssignMovingAvg_1Ґ AssignMovingAvg_1/ReadVariableOpҐbatchnorm/ReadVariableOpҐbatchnorm/mul/ReadVariableOpС
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indicesУ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(2
moments/meanА
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:@2
moments/StopGradient®
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*+
_output_shapes
:€€€€€€€€€}@2
moments/SquaredDifferenceЩ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indicesґ
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(2
moments/varianceБ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 2
moments/SqueezeЙ
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
„#<2
AssignMovingAvg/decay§
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOpШ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg/subП
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg/mulњ
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
„#<2
AssignMovingAvg_1/decay™
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOp†
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg_1/subЧ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg_1/mul…
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
 *oГ:2
batchnorm/add/yВ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2
batchnorm/mulz
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€}@2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:@2
batchnorm/mul_2Т
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOpБ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2
batchnorm/subЙ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*+
_output_shapes
:€€€€€€€€€}@2
batchnorm/add_1r
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€}@2

Identityт
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€}@: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:S O
+
_output_shapes
:€€€€€€€€€}@
 
_user_specified_nameinputs
ж
ѕ
4__inference_batch_normalization_layer_call_fn_108030

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityИҐStatefulPartitionedCall†
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€}@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_1044592
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€}@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€}@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:€€€€€€€€€}@
 
_user_specified_nameinputs
Л	
ѕ
4__inference_batch_normalization_layer_call_fn_108004

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityИҐStatefulPartitionedCall©
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_1035732
StatefulPartitionedCallИ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:€€€€€€€€€€€€€€€€€€@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs
М
m
O__inference_stream_2_input_drop_layer_call_and_return_conditional_losses_104285

inputs

identity_1^
IdentityIdentityinputs*
T0*+
_output_shapes
:€€€€€€€€€}2

Identitym

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:€€€€€€€€€}2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€}:S O
+
_output_shapes
:€€€€€€€€€}
 
_user_specified_nameinputs
н
j
K__inference_stream_1_drop_1_layer_call_and_return_conditional_losses_108437

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †?2
dropout/Constw
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:€€€€€€€€€}@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape”
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:€€€€€€€€€}@*
dtype0*
seedЈ*
seed2Ј2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL>2
dropout/GreaterEqual/y¬
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:€€€€€€€€€}@2
dropout/GreaterEqualГ
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:€€€€€€€€€}@2
dropout/Cast~
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:€€€€€€€€€}@2
dropout/Mul_1i
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:€€€€€€€€€}@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€}@:S O
+
_output_shapes
:€€€€€€€€€}@
 
_user_specified_nameinputs
Ю
п
&__inference_model_layer_call_fn_107123
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

unknown_17:	јT

unknown_18:T

unknown_19:T

unknown_20:T

unknown_21:T

unknown_22:T
identityИҐStatefulPartitionedCall§
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
:€€€€€€€€€*2
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_1059102
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*q
_input_shapes`
^:€€€€€€€€€}:€€€€€€€€€}: : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
+
_output_shapes
:€€€€€€€€€}
"
_user_specified_name
inputs/0:UQ
+
_output_shapes
:€€€€€€€€€}
"
_user_specified_name
inputs/1
њ
i
0__inference_stream_2_drop_1_layer_call_fn_108474

inputs
identityИҐStatefulPartitionedCallи
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€}@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_stream_2_drop_1_layer_call_and_return_conditional_losses_1047832
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€}@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€}@22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:€€€€€€€€€}@
 
_user_specified_nameinputs
Д
p
T__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_108486

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
:€€€€€€€€€@2
Meana
IdentityIdentityMean:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€}@:S O
+
_output_shapes
:€€€€€€€€€}@
 
_user_specified_nameinputs
«
m
4__inference_stream_1_input_drop_layer_call_fn_107748

inputs
identityИҐStatefulPartitionedCallм
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€}* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_stream_1_input_drop_layer_call_and_return_conditional_losses_1050572
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€}2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€}22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:€€€€€€€€€}
 
_user_specified_nameinputs
¶
п
&__inference_model_layer_call_fn_107069
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

unknown_17:	јT

unknown_18:T

unknown_19:T

unknown_20:T

unknown_21:T

unknown_22:T
identityИҐStatefulPartitionedCallђ
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
:€€€€€€€€€*:
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_1056732
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*q
_input_shapes`
^:€€€€€€€€€}:€€€€€€€€€}: : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
+
_output_shapes
:€€€€€€€€€}
"
_user_specified_name
inputs/0:UQ
+
_output_shapes
:€€€€€€€€€}
"
_user_specified_name
inputs/1
ЂI
∞	
A__inference_model_layer_call_and_return_conditional_losses_105910

inputs
inputs_1&
basemodel_105810:@
basemodel_105812:@&
basemodel_105814:@
basemodel_105816:@&
basemodel_105818:@
basemodel_105820:@
basemodel_105822:@
basemodel_105824:@
basemodel_105826:@
basemodel_105828:@
basemodel_105830:@
basemodel_105832:@
basemodel_105834:@
basemodel_105836:@
basemodel_105838:@
basemodel_105840:@
basemodel_105842:@
basemodel_105844:@#
basemodel_105846:	јT
basemodel_105848:T
basemodel_105850:T
basemodel_105852:T
basemodel_105854:T
basemodel_105856:T
identityИҐ!basemodel/StatefulPartitionedCallҐ#basemodel/StatefulPartitionedCall_1Ґ-dense_1/kernel/Regularizer/Abs/ReadVariableOpҐ5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpҐ8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpҐ5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOpё
!basemodel/StatefulPartitionedCallStatefulPartitionedCallinputsinputsinputsbasemodel_105810basemodel_105812basemodel_105814basemodel_105816basemodel_105818basemodel_105820basemodel_105822basemodel_105824basemodel_105826basemodel_105828basemodel_105830basemodel_105832basemodel_105834basemodel_105836basemodel_105838basemodel_105840basemodel_105842basemodel_105844basemodel_105846basemodel_105848basemodel_105850basemodel_105852basemodel_105854basemodel_105856*&
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€T*2
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_basemodel_layer_call_and_return_conditional_losses_1052432#
!basemodel/StatefulPartitionedCallМ
#basemodel/StatefulPartitionedCall_1StatefulPartitionedCallinputs_1inputs_1inputs_1basemodel_105810basemodel_105812basemodel_105814basemodel_105816basemodel_105818basemodel_105820basemodel_105822basemodel_105824basemodel_105826basemodel_105828basemodel_105830basemodel_105832basemodel_105834basemodel_105836basemodel_105838basemodel_105840basemodel_105842basemodel_105844basemodel_105846basemodel_105848basemodel_105850basemodel_105852basemodel_105854basemodel_105856"^basemodel/StatefulPartitionedCall*&
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€T*2
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_basemodel_layer_call_and_return_conditional_losses_1052432%
#basemodel/StatefulPartitionedCall_1™
distance/PartitionedCallPartitionedCall*basemodel/StatefulPartitionedCall:output:0,basemodel/StatefulPartitionedCall_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_distance_layer_call_and_return_conditional_losses_1057462
distance/PartitionedCall√
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpbasemodel_105818*"
_output_shapes
:@*
dtype027
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp√
&stream_0_conv_1/kernel/Regularizer/AbsAbs=stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_0_conv_1/kernel/Regularizer/Abs©
(stream_0_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_0_conv_1/kernel/Regularizer/Const„
&stream_0_conv_1/kernel/Regularizer/SumSum*stream_0_conv_1/kernel/Regularizer/Abs:y:01stream_0_conv_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/SumЩ
(stream_0_conv_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2*
(stream_0_conv_1/kernel/Regularizer/mul/x№
&stream_0_conv_1/kernel/Regularizer/mulMul1stream_0_conv_1/kernel/Regularizer/mul/x:output:0/stream_0_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/mul…
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpbasemodel_105814*"
_output_shapes
:@*
dtype02:
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpѕ
)stream_1_conv_1/kernel/Regularizer/SquareSquare@stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2+
)stream_1_conv_1/kernel/Regularizer/Square©
(stream_1_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_1_conv_1/kernel/Regularizer/ConstЏ
&stream_1_conv_1/kernel/Regularizer/SumSum-stream_1_conv_1/kernel/Regularizer/Square:y:01stream_1_conv_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&stream_1_conv_1/kernel/Regularizer/SumЩ
(stream_1_conv_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2*
(stream_1_conv_1/kernel/Regularizer/mul/x№
&stream_1_conv_1/kernel/Regularizer/mulMul1stream_1_conv_1/kernel/Regularizer/mul/x:output:0/stream_1_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_1_conv_1/kernel/Regularizer/mul√
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpbasemodel_105810*"
_output_shapes
:@*
dtype027
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp√
&stream_2_conv_1/kernel/Regularizer/AbsAbs=stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_2_conv_1/kernel/Regularizer/Abs©
(stream_2_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_2_conv_1/kernel/Regularizer/Const„
&stream_2_conv_1/kernel/Regularizer/SumSum*stream_2_conv_1/kernel/Regularizer/Abs:y:01stream_2_conv_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&stream_2_conv_1/kernel/Regularizer/SumЩ
(stream_2_conv_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2*
(stream_2_conv_1/kernel/Regularizer/mul/x№
&stream_2_conv_1/kernel/Regularizer/mulMul1stream_2_conv_1/kernel/Regularizer/mul/x:output:0/stream_2_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_2_conv_1/kernel/Regularizer/mul∞
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpbasemodel_105846*
_output_shapes
:	јT*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOp®
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	јT2 
dense_1/kernel/Regularizer/AbsХ
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_1/kernel/Regularizer/ConstЈ
dense_1/kernel/Regularizer/SumSum"dense_1/kernel/Regularizer/Abs:y:0)dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/SumЙ
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2"
 dense_1/kernel/Regularizer/mul/xЉ
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/mul|
IdentityIdentity!distance/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identityу
NoOpNoOp"^basemodel/StatefulPartitionedCall$^basemodel/StatefulPartitionedCall_1.^dense_1/kernel/Regularizer/Abs/ReadVariableOp6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp9^stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp6^stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*q
_input_shapes`
^:€€€€€€€€€}:€€€€€€€€€}: : : : : : : : : : : : : : : : : : : : : : : : 2F
!basemodel/StatefulPartitionedCall!basemodel/StatefulPartitionedCall2J
#basemodel/StatefulPartitionedCall_1#basemodel/StatefulPartitionedCall_12^
-dense_1/kernel/Regularizer/Abs/ReadVariableOp-dense_1/kernel/Regularizer/Abs/ReadVariableOp2n
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp2t
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp2n
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp:S O
+
_output_shapes
:€€€€€€€€€}
 
_user_specified_nameinputs:SO
+
_output_shapes
:€€€€€€€€€}
 
_user_specified_nameinputs
≥
ц
&__inference_model_layer_call_fn_106015
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

unknown_17:	јT

unknown_18:T

unknown_19:T

unknown_20:T

unknown_21:T

unknown_22:T
identityИҐStatefulPartitionedCallЂ
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
:€€€€€€€€€*2
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_1059102
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*q
_input_shapes`
^:€€€€€€€€€}:€€€€€€€€€}: : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
+
_output_shapes
:€€€€€€€€€}
%
_user_specified_nameleft_inputs:YU
+
_output_shapes
:€€€€€€€€€}
&
_user_specified_nameright_inputs
Ї
p
T__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_108480

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
:€€€€€€€€€€€€€€€€€€2
Meanj
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€:e a
=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
з
U
9__inference_global_average_pooling1d_layer_call_fn_108496

inputs
identity’
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *]
fXRV
T__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_1045162
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€}@:S O
+
_output_shapes
:€€€€€€€€€}@
 
_user_specified_nameinputs
Ј
∞
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_103897

inputs/
!batchnorm_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@1
#batchnorm_readvariableop_1_resource:@1
#batchnorm_readvariableop_2_resource:@
identityИҐbatchnorm/ReadVariableOpҐbatchnorm/ReadVariableOp_1Ґbatchnorm/ReadVariableOp_2Ґbatchnorm/mul/ReadVariableOpТ
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
 *oГ:2
batchnorm/add/yИ
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2
batchnorm/mulГ
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€@2
batchnorm/mul_1Ш
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp_1Е
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:@2
batchnorm/mul_2Ш
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp_2Г
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2
batchnorm/subТ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€@2
batchnorm/add_1{
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€@2

Identity¬
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:€€€€€€€€€€€€€€€€€€@: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs
И
i
K__inference_stream_0_drop_1_layer_call_and_return_conditional_losses_108398

inputs

identity_1^
IdentityIdentityinputs*
T0*+
_output_shapes
:€€€€€€€€€}@2

Identitym

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:€€€€€€€€€}@2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€}@:S O
+
_output_shapes
:€€€€€€€€€}@
 
_user_specified_nameinputs
Й
∞
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_104401

inputs/
!batchnorm_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@1
#batchnorm_readvariableop_1_resource:@1
#batchnorm_readvariableop_2_resource:@
identityИҐbatchnorm/ReadVariableOpҐbatchnorm/ReadVariableOp_1Ґbatchnorm/ReadVariableOp_2Ґbatchnorm/mul/ReadVariableOpТ
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
 *oГ:2
batchnorm/add/yИ
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2
batchnorm/mulz
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€}@2
batchnorm/mul_1Ш
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp_1Е
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:@2
batchnorm/mul_2Ш
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp_2Г
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2
batchnorm/subЙ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*+
_output_shapes
:€€€€€€€€€}@2
batchnorm/add_1r
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€}@2

Identity¬
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€}@: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:S O
+
_output_shapes
:€€€€€€€€€}@
 
_user_specified_nameinputs
И
i
K__inference_stream_0_drop_1_layer_call_and_return_conditional_losses_104509

inputs

identity_1^
IdentityIdentityinputs*
T0*+
_output_shapes
:€€€€€€€€€}@2

Identitym

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:€€€€€€€€€}@2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€}@:S O
+
_output_shapes
:€€€€€€€€€}@
 
_user_specified_nameinputs
§Ж
љ)
!__inference__wrapped_model_103549
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
6model_basemodel_dense_1_matmul_readvariableop_resource:	јTE
7model_basemodel_dense_1_biasadd_readvariableop_resource:TU
Gmodel_basemodel_batch_normalization_3_batchnorm_readvariableop_resource:TY
Kmodel_basemodel_batch_normalization_3_batchnorm_mul_readvariableop_resource:TW
Imodel_basemodel_batch_normalization_3_batchnorm_readvariableop_1_resource:TW
Imodel_basemodel_batch_normalization_3_batchnorm_readvariableop_2_resource:T
identityИҐ<model/basemodel/batch_normalization/batchnorm/ReadVariableOpҐ>model/basemodel/batch_normalization/batchnorm/ReadVariableOp_1Ґ>model/basemodel/batch_normalization/batchnorm/ReadVariableOp_2Ґ@model/basemodel/batch_normalization/batchnorm/mul/ReadVariableOpҐ>model/basemodel/batch_normalization/batchnorm_1/ReadVariableOpҐ@model/basemodel/batch_normalization/batchnorm_1/ReadVariableOp_1Ґ@model/basemodel/batch_normalization/batchnorm_1/ReadVariableOp_2ҐBmodel/basemodel/batch_normalization/batchnorm_1/mul/ReadVariableOpҐ>model/basemodel/batch_normalization_1/batchnorm/ReadVariableOpҐ@model/basemodel/batch_normalization_1/batchnorm/ReadVariableOp_1Ґ@model/basemodel/batch_normalization_1/batchnorm/ReadVariableOp_2ҐBmodel/basemodel/batch_normalization_1/batchnorm/mul/ReadVariableOpҐ@model/basemodel/batch_normalization_1/batchnorm_1/ReadVariableOpҐBmodel/basemodel/batch_normalization_1/batchnorm_1/ReadVariableOp_1ҐBmodel/basemodel/batch_normalization_1/batchnorm_1/ReadVariableOp_2ҐDmodel/basemodel/batch_normalization_1/batchnorm_1/mul/ReadVariableOpҐ>model/basemodel/batch_normalization_2/batchnorm/ReadVariableOpҐ@model/basemodel/batch_normalization_2/batchnorm/ReadVariableOp_1Ґ@model/basemodel/batch_normalization_2/batchnorm/ReadVariableOp_2ҐBmodel/basemodel/batch_normalization_2/batchnorm/mul/ReadVariableOpҐ@model/basemodel/batch_normalization_2/batchnorm_1/ReadVariableOpҐBmodel/basemodel/batch_normalization_2/batchnorm_1/ReadVariableOp_1ҐBmodel/basemodel/batch_normalization_2/batchnorm_1/ReadVariableOp_2ҐDmodel/basemodel/batch_normalization_2/batchnorm_1/mul/ReadVariableOpҐ>model/basemodel/batch_normalization_3/batchnorm/ReadVariableOpҐ@model/basemodel/batch_normalization_3/batchnorm/ReadVariableOp_1Ґ@model/basemodel/batch_normalization_3/batchnorm/ReadVariableOp_2ҐBmodel/basemodel/batch_normalization_3/batchnorm/mul/ReadVariableOpҐ@model/basemodel/batch_normalization_3/batchnorm_1/ReadVariableOpҐBmodel/basemodel/batch_normalization_3/batchnorm_1/ReadVariableOp_1ҐBmodel/basemodel/batch_normalization_3/batchnorm_1/ReadVariableOp_2ҐDmodel/basemodel/batch_normalization_3/batchnorm_1/mul/ReadVariableOpҐ.model/basemodel/dense_1/BiasAdd/ReadVariableOpҐ0model/basemodel/dense_1/BiasAdd_1/ReadVariableOpҐ-model/basemodel/dense_1/MatMul/ReadVariableOpҐ/model/basemodel/dense_1/MatMul_1/ReadVariableOpҐ6model/basemodel/stream_0_conv_1/BiasAdd/ReadVariableOpҐ8model/basemodel/stream_0_conv_1/BiasAdd_1/ReadVariableOpҐBmodel/basemodel/stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOpҐDmodel/basemodel/stream_0_conv_1/conv1d_1/ExpandDims_1/ReadVariableOpҐ6model/basemodel/stream_1_conv_1/BiasAdd/ReadVariableOpҐ8model/basemodel/stream_1_conv_1/BiasAdd_1/ReadVariableOpҐBmodel/basemodel/stream_1_conv_1/conv1d/ExpandDims_1/ReadVariableOpҐDmodel/basemodel/stream_1_conv_1/conv1d_1/ExpandDims_1/ReadVariableOpҐ6model/basemodel/stream_2_conv_1/BiasAdd/ReadVariableOpҐ8model/basemodel/stream_2_conv_1/BiasAdd_1/ReadVariableOpҐBmodel/basemodel/stream_2_conv_1/conv1d/ExpandDims_1/ReadVariableOpҐDmodel/basemodel/stream_2_conv_1/conv1d_1/ExpandDims_1/ReadVariableOpЂ
,model/basemodel/stream_2_input_drop/IdentityIdentityleft_inputs*
T0*+
_output_shapes
:€€€€€€€€€}2.
,model/basemodel/stream_2_input_drop/IdentityЂ
,model/basemodel/stream_1_input_drop/IdentityIdentityleft_inputs*
T0*+
_output_shapes
:€€€€€€€€€}2.
,model/basemodel/stream_1_input_drop/IdentityЂ
,model/basemodel/stream_0_input_drop/IdentityIdentityleft_inputs*
T0*+
_output_shapes
:€€€€€€€€€}2.
,model/basemodel/stream_0_input_drop/Identityє
5model/basemodel/stream_2_conv_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€27
5model/basemodel/stream_2_conv_1/conv1d/ExpandDims/dim•
1model/basemodel/stream_2_conv_1/conv1d/ExpandDims
ExpandDims5model/basemodel/stream_2_input_drop/Identity:output:0>model/basemodel/stream_2_conv_1/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:€€€€€€€€€}23
1model/basemodel/stream_2_conv_1/conv1d/ExpandDimsШ
Bmodel/basemodel/stream_2_conv_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpKmodel_basemodel_stream_2_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02D
Bmodel/basemodel/stream_2_conv_1/conv1d/ExpandDims_1/ReadVariableOpі
7model/basemodel/stream_2_conv_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 29
7model/basemodel/stream_2_conv_1/conv1d/ExpandDims_1/dimЈ
3model/basemodel/stream_2_conv_1/conv1d/ExpandDims_1
ExpandDimsJmodel/basemodel/stream_2_conv_1/conv1d/ExpandDims_1/ReadVariableOp:value:0@model/basemodel/stream_2_conv_1/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@25
3model/basemodel/stream_2_conv_1/conv1d/ExpandDims_1ґ
&model/basemodel/stream_2_conv_1/conv1dConv2D:model/basemodel/stream_2_conv_1/conv1d/ExpandDims:output:0<model/basemodel/stream_2_conv_1/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:€€€€€€€€€}@*
paddingSAME*
strides
2(
&model/basemodel/stream_2_conv_1/conv1dт
.model/basemodel/stream_2_conv_1/conv1d/SqueezeSqueeze/model/basemodel/stream_2_conv_1/conv1d:output:0*
T0*+
_output_shapes
:€€€€€€€€€}@*
squeeze_dims

э€€€€€€€€20
.model/basemodel/stream_2_conv_1/conv1d/Squeezeм
6model/basemodel/stream_2_conv_1/BiasAdd/ReadVariableOpReadVariableOp?model_basemodel_stream_2_conv_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype028
6model/basemodel/stream_2_conv_1/BiasAdd/ReadVariableOpМ
'model/basemodel/stream_2_conv_1/BiasAddBiasAdd7model/basemodel/stream_2_conv_1/conv1d/Squeeze:output:0>model/basemodel/stream_2_conv_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€}@2)
'model/basemodel/stream_2_conv_1/BiasAddє
5model/basemodel/stream_1_conv_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€27
5model/basemodel/stream_1_conv_1/conv1d/ExpandDims/dim•
1model/basemodel/stream_1_conv_1/conv1d/ExpandDims
ExpandDims5model/basemodel/stream_1_input_drop/Identity:output:0>model/basemodel/stream_1_conv_1/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:€€€€€€€€€}23
1model/basemodel/stream_1_conv_1/conv1d/ExpandDimsШ
Bmodel/basemodel/stream_1_conv_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpKmodel_basemodel_stream_1_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02D
Bmodel/basemodel/stream_1_conv_1/conv1d/ExpandDims_1/ReadVariableOpі
7model/basemodel/stream_1_conv_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 29
7model/basemodel/stream_1_conv_1/conv1d/ExpandDims_1/dimЈ
3model/basemodel/stream_1_conv_1/conv1d/ExpandDims_1
ExpandDimsJmodel/basemodel/stream_1_conv_1/conv1d/ExpandDims_1/ReadVariableOp:value:0@model/basemodel/stream_1_conv_1/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@25
3model/basemodel/stream_1_conv_1/conv1d/ExpandDims_1ґ
&model/basemodel/stream_1_conv_1/conv1dConv2D:model/basemodel/stream_1_conv_1/conv1d/ExpandDims:output:0<model/basemodel/stream_1_conv_1/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:€€€€€€€€€}@*
paddingSAME*
strides
2(
&model/basemodel/stream_1_conv_1/conv1dт
.model/basemodel/stream_1_conv_1/conv1d/SqueezeSqueeze/model/basemodel/stream_1_conv_1/conv1d:output:0*
T0*+
_output_shapes
:€€€€€€€€€}@*
squeeze_dims

э€€€€€€€€20
.model/basemodel/stream_1_conv_1/conv1d/Squeezeм
6model/basemodel/stream_1_conv_1/BiasAdd/ReadVariableOpReadVariableOp?model_basemodel_stream_1_conv_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype028
6model/basemodel/stream_1_conv_1/BiasAdd/ReadVariableOpМ
'model/basemodel/stream_1_conv_1/BiasAddBiasAdd7model/basemodel/stream_1_conv_1/conv1d/Squeeze:output:0>model/basemodel/stream_1_conv_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€}@2)
'model/basemodel/stream_1_conv_1/BiasAddє
5model/basemodel/stream_0_conv_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€27
5model/basemodel/stream_0_conv_1/conv1d/ExpandDims/dim•
1model/basemodel/stream_0_conv_1/conv1d/ExpandDims
ExpandDims5model/basemodel/stream_0_input_drop/Identity:output:0>model/basemodel/stream_0_conv_1/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:€€€€€€€€€}23
1model/basemodel/stream_0_conv_1/conv1d/ExpandDimsШ
Bmodel/basemodel/stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpKmodel_basemodel_stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02D
Bmodel/basemodel/stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOpі
7model/basemodel/stream_0_conv_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 29
7model/basemodel/stream_0_conv_1/conv1d/ExpandDims_1/dimЈ
3model/basemodel/stream_0_conv_1/conv1d/ExpandDims_1
ExpandDimsJmodel/basemodel/stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp:value:0@model/basemodel/stream_0_conv_1/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@25
3model/basemodel/stream_0_conv_1/conv1d/ExpandDims_1ґ
&model/basemodel/stream_0_conv_1/conv1dConv2D:model/basemodel/stream_0_conv_1/conv1d/ExpandDims:output:0<model/basemodel/stream_0_conv_1/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:€€€€€€€€€}@*
paddingSAME*
strides
2(
&model/basemodel/stream_0_conv_1/conv1dт
.model/basemodel/stream_0_conv_1/conv1d/SqueezeSqueeze/model/basemodel/stream_0_conv_1/conv1d:output:0*
T0*+
_output_shapes
:€€€€€€€€€}@*
squeeze_dims

э€€€€€€€€20
.model/basemodel/stream_0_conv_1/conv1d/Squeezeм
6model/basemodel/stream_0_conv_1/BiasAdd/ReadVariableOpReadVariableOp?model_basemodel_stream_0_conv_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype028
6model/basemodel/stream_0_conv_1/BiasAdd/ReadVariableOpМ
'model/basemodel/stream_0_conv_1/BiasAddBiasAdd7model/basemodel/stream_0_conv_1/conv1d/Squeeze:output:0>model/basemodel/stream_0_conv_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€}@2)
'model/basemodel/stream_0_conv_1/BiasAddД
>model/basemodel/batch_normalization_2/batchnorm/ReadVariableOpReadVariableOpGmodel_basemodel_batch_normalization_2_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02@
>model/basemodel/batch_normalization_2/batchnorm/ReadVariableOp≥
5model/basemodel/batch_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:27
5model/basemodel/batch_normalization_2/batchnorm/add/y†
3model/basemodel/batch_normalization_2/batchnorm/addAddV2Fmodel/basemodel/batch_normalization_2/batchnorm/ReadVariableOp:value:0>model/basemodel/batch_normalization_2/batchnorm/add/y:output:0*
T0*
_output_shapes
:@25
3model/basemodel/batch_normalization_2/batchnorm/add’
5model/basemodel/batch_normalization_2/batchnorm/RsqrtRsqrt7model/basemodel/batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes
:@27
5model/basemodel/batch_normalization_2/batchnorm/RsqrtР
Bmodel/basemodel/batch_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOpKmodel_basemodel_batch_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02D
Bmodel/basemodel/batch_normalization_2/batchnorm/mul/ReadVariableOpЭ
3model/basemodel/batch_normalization_2/batchnorm/mulMul9model/basemodel/batch_normalization_2/batchnorm/Rsqrt:y:0Jmodel/basemodel/batch_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@25
3model/basemodel/batch_normalization_2/batchnorm/mulЦ
5model/basemodel/batch_normalization_2/batchnorm/mul_1Mul0model/basemodel/stream_2_conv_1/BiasAdd:output:07model/basemodel/batch_normalization_2/batchnorm/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€}@27
5model/basemodel/batch_normalization_2/batchnorm/mul_1К
@model/basemodel/batch_normalization_2/batchnorm/ReadVariableOp_1ReadVariableOpImodel_basemodel_batch_normalization_2_batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype02B
@model/basemodel/batch_normalization_2/batchnorm/ReadVariableOp_1Э
5model/basemodel/batch_normalization_2/batchnorm/mul_2MulHmodel/basemodel/batch_normalization_2/batchnorm/ReadVariableOp_1:value:07model/basemodel/batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes
:@27
5model/basemodel/batch_normalization_2/batchnorm/mul_2К
@model/basemodel/batch_normalization_2/batchnorm/ReadVariableOp_2ReadVariableOpImodel_basemodel_batch_normalization_2_batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype02B
@model/basemodel/batch_normalization_2/batchnorm/ReadVariableOp_2Ы
3model/basemodel/batch_normalization_2/batchnorm/subSubHmodel/basemodel/batch_normalization_2/batchnorm/ReadVariableOp_2:value:09model/basemodel/batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@25
3model/basemodel/batch_normalization_2/batchnorm/sub°
5model/basemodel/batch_normalization_2/batchnorm/add_1AddV29model/basemodel/batch_normalization_2/batchnorm/mul_1:z:07model/basemodel/batch_normalization_2/batchnorm/sub:z:0*
T0*+
_output_shapes
:€€€€€€€€€}@27
5model/basemodel/batch_normalization_2/batchnorm/add_1Д
>model/basemodel/batch_normalization_1/batchnorm/ReadVariableOpReadVariableOpGmodel_basemodel_batch_normalization_1_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02@
>model/basemodel/batch_normalization_1/batchnorm/ReadVariableOp≥
5model/basemodel/batch_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:27
5model/basemodel/batch_normalization_1/batchnorm/add/y†
3model/basemodel/batch_normalization_1/batchnorm/addAddV2Fmodel/basemodel/batch_normalization_1/batchnorm/ReadVariableOp:value:0>model/basemodel/batch_normalization_1/batchnorm/add/y:output:0*
T0*
_output_shapes
:@25
3model/basemodel/batch_normalization_1/batchnorm/add’
5model/basemodel/batch_normalization_1/batchnorm/RsqrtRsqrt7model/basemodel/batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes
:@27
5model/basemodel/batch_normalization_1/batchnorm/RsqrtР
Bmodel/basemodel/batch_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOpKmodel_basemodel_batch_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02D
Bmodel/basemodel/batch_normalization_1/batchnorm/mul/ReadVariableOpЭ
3model/basemodel/batch_normalization_1/batchnorm/mulMul9model/basemodel/batch_normalization_1/batchnorm/Rsqrt:y:0Jmodel/basemodel/batch_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@25
3model/basemodel/batch_normalization_1/batchnorm/mulЦ
5model/basemodel/batch_normalization_1/batchnorm/mul_1Mul0model/basemodel/stream_1_conv_1/BiasAdd:output:07model/basemodel/batch_normalization_1/batchnorm/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€}@27
5model/basemodel/batch_normalization_1/batchnorm/mul_1К
@model/basemodel/batch_normalization_1/batchnorm/ReadVariableOp_1ReadVariableOpImodel_basemodel_batch_normalization_1_batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype02B
@model/basemodel/batch_normalization_1/batchnorm/ReadVariableOp_1Э
5model/basemodel/batch_normalization_1/batchnorm/mul_2MulHmodel/basemodel/batch_normalization_1/batchnorm/ReadVariableOp_1:value:07model/basemodel/batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes
:@27
5model/basemodel/batch_normalization_1/batchnorm/mul_2К
@model/basemodel/batch_normalization_1/batchnorm/ReadVariableOp_2ReadVariableOpImodel_basemodel_batch_normalization_1_batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype02B
@model/basemodel/batch_normalization_1/batchnorm/ReadVariableOp_2Ы
3model/basemodel/batch_normalization_1/batchnorm/subSubHmodel/basemodel/batch_normalization_1/batchnorm/ReadVariableOp_2:value:09model/basemodel/batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@25
3model/basemodel/batch_normalization_1/batchnorm/sub°
5model/basemodel/batch_normalization_1/batchnorm/add_1AddV29model/basemodel/batch_normalization_1/batchnorm/mul_1:z:07model/basemodel/batch_normalization_1/batchnorm/sub:z:0*
T0*+
_output_shapes
:€€€€€€€€€}@27
5model/basemodel/batch_normalization_1/batchnorm/add_1ю
<model/basemodel/batch_normalization/batchnorm/ReadVariableOpReadVariableOpEmodel_basemodel_batch_normalization_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02>
<model/basemodel/batch_normalization/batchnorm/ReadVariableOpѓ
3model/basemodel/batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:25
3model/basemodel/batch_normalization/batchnorm/add/yШ
1model/basemodel/batch_normalization/batchnorm/addAddV2Dmodel/basemodel/batch_normalization/batchnorm/ReadVariableOp:value:0<model/basemodel/batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes
:@23
1model/basemodel/batch_normalization/batchnorm/addѕ
3model/basemodel/batch_normalization/batchnorm/RsqrtRsqrt5model/basemodel/batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes
:@25
3model/basemodel/batch_normalization/batchnorm/RsqrtК
@model/basemodel/batch_normalization/batchnorm/mul/ReadVariableOpReadVariableOpImodel_basemodel_batch_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02B
@model/basemodel/batch_normalization/batchnorm/mul/ReadVariableOpХ
1model/basemodel/batch_normalization/batchnorm/mulMul7model/basemodel/batch_normalization/batchnorm/Rsqrt:y:0Hmodel/basemodel/batch_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@23
1model/basemodel/batch_normalization/batchnorm/mulР
3model/basemodel/batch_normalization/batchnorm/mul_1Mul0model/basemodel/stream_0_conv_1/BiasAdd:output:05model/basemodel/batch_normalization/batchnorm/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€}@25
3model/basemodel/batch_normalization/batchnorm/mul_1Д
>model/basemodel/batch_normalization/batchnorm/ReadVariableOp_1ReadVariableOpGmodel_basemodel_batch_normalization_batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype02@
>model/basemodel/batch_normalization/batchnorm/ReadVariableOp_1Х
3model/basemodel/batch_normalization/batchnorm/mul_2MulFmodel/basemodel/batch_normalization/batchnorm/ReadVariableOp_1:value:05model/basemodel/batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes
:@25
3model/basemodel/batch_normalization/batchnorm/mul_2Д
>model/basemodel/batch_normalization/batchnorm/ReadVariableOp_2ReadVariableOpGmodel_basemodel_batch_normalization_batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype02@
>model/basemodel/batch_normalization/batchnorm/ReadVariableOp_2У
1model/basemodel/batch_normalization/batchnorm/subSubFmodel/basemodel/batch_normalization/batchnorm/ReadVariableOp_2:value:07model/basemodel/batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@23
1model/basemodel/batch_normalization/batchnorm/subЩ
3model/basemodel/batch_normalization/batchnorm/add_1AddV27model/basemodel/batch_normalization/batchnorm/mul_1:z:05model/basemodel/batch_normalization/batchnorm/sub:z:0*
T0*+
_output_shapes
:€€€€€€€€€}@25
3model/basemodel/batch_normalization/batchnorm/add_1њ
!model/basemodel/activation_2/ReluRelu9model/basemodel/batch_normalization_2/batchnorm/add_1:z:0*
T0*+
_output_shapes
:€€€€€€€€€}@2#
!model/basemodel/activation_2/Reluњ
!model/basemodel/activation_1/ReluRelu9model/basemodel/batch_normalization_1/batchnorm/add_1:z:0*
T0*+
_output_shapes
:€€€€€€€€€}@2#
!model/basemodel/activation_1/Reluє
model/basemodel/activation/ReluRelu7model/basemodel/batch_normalization/batchnorm/add_1:z:0*
T0*+
_output_shapes
:€€€€€€€€€}@2!
model/basemodel/activation/Relu«
(model/basemodel/stream_2_drop_1/IdentityIdentity/model/basemodel/activation_2/Relu:activations:0*
T0*+
_output_shapes
:€€€€€€€€€}@2*
(model/basemodel/stream_2_drop_1/Identity«
(model/basemodel/stream_1_drop_1/IdentityIdentity/model/basemodel/activation_1/Relu:activations:0*
T0*+
_output_shapes
:€€€€€€€€€}@2*
(model/basemodel/stream_1_drop_1/Identity≈
(model/basemodel/stream_0_drop_1/IdentityIdentity-model/basemodel/activation/Relu:activations:0*
T0*+
_output_shapes
:€€€€€€€€€}@2*
(model/basemodel/stream_0_drop_1/Identityƒ
?model/basemodel/global_average_pooling1d/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2A
?model/basemodel/global_average_pooling1d/Mean/reduction_indicesХ
-model/basemodel/global_average_pooling1d/MeanMean1model/basemodel/stream_0_drop_1/Identity:output:0Hmodel/basemodel/global_average_pooling1d/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2/
-model/basemodel/global_average_pooling1d/Mean»
Amodel/basemodel/global_average_pooling1d_1/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2C
Amodel/basemodel/global_average_pooling1d_1/Mean/reduction_indicesЫ
/model/basemodel/global_average_pooling1d_1/MeanMean1model/basemodel/stream_1_drop_1/Identity:output:0Jmodel/basemodel/global_average_pooling1d_1/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:€€€€€€€€€@21
/model/basemodel/global_average_pooling1d_1/Mean»
Amodel/basemodel/global_average_pooling1d_2/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2C
Amodel/basemodel/global_average_pooling1d_2/Mean/reduction_indicesЫ
/model/basemodel/global_average_pooling1d_2/MeanMean1model/basemodel/stream_2_drop_1/Identity:output:0Jmodel/basemodel/global_average_pooling1d_2/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:€€€€€€€€€@21
/model/basemodel/global_average_pooling1d_2/MeanФ
'model/basemodel/concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2)
'model/basemodel/concatenate/concat/axisо
"model/basemodel/concatenate/concatConcatV26model/basemodel/global_average_pooling1d/Mean:output:08model/basemodel/global_average_pooling1d_1/Mean:output:08model/basemodel/global_average_pooling1d_2/Mean:output:00model/basemodel/concatenate/concat/axis:output:0*
N*
T0*(
_output_shapes
:€€€€€€€€€ј2$
"model/basemodel/concatenate/concatј
(model/basemodel/dense_1_dropout/IdentityIdentity+model/basemodel/concatenate/concat:output:0*
T0*(
_output_shapes
:€€€€€€€€€ј2*
(model/basemodel/dense_1_dropout/Identity÷
-model/basemodel/dense_1/MatMul/ReadVariableOpReadVariableOp6model_basemodel_dense_1_matmul_readvariableop_resource*
_output_shapes
:	јT*
dtype02/
-model/basemodel/dense_1/MatMul/ReadVariableOpж
model/basemodel/dense_1/MatMulMatMul1model/basemodel/dense_1_dropout/Identity:output:05model/basemodel/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€T2 
model/basemodel/dense_1/MatMul‘
.model/basemodel/dense_1/BiasAdd/ReadVariableOpReadVariableOp7model_basemodel_dense_1_biasadd_readvariableop_resource*
_output_shapes
:T*
dtype020
.model/basemodel/dense_1/BiasAdd/ReadVariableOpб
model/basemodel/dense_1/BiasAddBiasAdd(model/basemodel/dense_1/MatMul:product:06model/basemodel/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€T2!
model/basemodel/dense_1/BiasAddД
>model/basemodel/batch_normalization_3/batchnorm/ReadVariableOpReadVariableOpGmodel_basemodel_batch_normalization_3_batchnorm_readvariableop_resource*
_output_shapes
:T*
dtype02@
>model/basemodel/batch_normalization_3/batchnorm/ReadVariableOp≥
5model/basemodel/batch_normalization_3/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:27
5model/basemodel/batch_normalization_3/batchnorm/add/y†
3model/basemodel/batch_normalization_3/batchnorm/addAddV2Fmodel/basemodel/batch_normalization_3/batchnorm/ReadVariableOp:value:0>model/basemodel/batch_normalization_3/batchnorm/add/y:output:0*
T0*
_output_shapes
:T25
3model/basemodel/batch_normalization_3/batchnorm/add’
5model/basemodel/batch_normalization_3/batchnorm/RsqrtRsqrt7model/basemodel/batch_normalization_3/batchnorm/add:z:0*
T0*
_output_shapes
:T27
5model/basemodel/batch_normalization_3/batchnorm/RsqrtР
Bmodel/basemodel/batch_normalization_3/batchnorm/mul/ReadVariableOpReadVariableOpKmodel_basemodel_batch_normalization_3_batchnorm_mul_readvariableop_resource*
_output_shapes
:T*
dtype02D
Bmodel/basemodel/batch_normalization_3/batchnorm/mul/ReadVariableOpЭ
3model/basemodel/batch_normalization_3/batchnorm/mulMul9model/basemodel/batch_normalization_3/batchnorm/Rsqrt:y:0Jmodel/basemodel/batch_normalization_3/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:T25
3model/basemodel/batch_normalization_3/batchnorm/mulК
5model/basemodel/batch_normalization_3/batchnorm/mul_1Mul(model/basemodel/dense_1/BiasAdd:output:07model/basemodel/batch_normalization_3/batchnorm/mul:z:0*
T0*'
_output_shapes
:€€€€€€€€€T27
5model/basemodel/batch_normalization_3/batchnorm/mul_1К
@model/basemodel/batch_normalization_3/batchnorm/ReadVariableOp_1ReadVariableOpImodel_basemodel_batch_normalization_3_batchnorm_readvariableop_1_resource*
_output_shapes
:T*
dtype02B
@model/basemodel/batch_normalization_3/batchnorm/ReadVariableOp_1Э
5model/basemodel/batch_normalization_3/batchnorm/mul_2MulHmodel/basemodel/batch_normalization_3/batchnorm/ReadVariableOp_1:value:07model/basemodel/batch_normalization_3/batchnorm/mul:z:0*
T0*
_output_shapes
:T27
5model/basemodel/batch_normalization_3/batchnorm/mul_2К
@model/basemodel/batch_normalization_3/batchnorm/ReadVariableOp_2ReadVariableOpImodel_basemodel_batch_normalization_3_batchnorm_readvariableop_2_resource*
_output_shapes
:T*
dtype02B
@model/basemodel/batch_normalization_3/batchnorm/ReadVariableOp_2Ы
3model/basemodel/batch_normalization_3/batchnorm/subSubHmodel/basemodel/batch_normalization_3/batchnorm/ReadVariableOp_2:value:09model/basemodel/batch_normalization_3/batchnorm/mul_2:z:0*
T0*
_output_shapes
:T25
3model/basemodel/batch_normalization_3/batchnorm/subЭ
5model/basemodel/batch_normalization_3/batchnorm/add_1AddV29model/basemodel/batch_normalization_3/batchnorm/mul_1:z:07model/basemodel/batch_normalization_3/batchnorm/sub:z:0*
T0*'
_output_shapes
:€€€€€€€€€T27
5model/basemodel/batch_normalization_3/batchnorm/add_1–
*model/basemodel/dense_activation_1/SigmoidSigmoid9model/basemodel/batch_normalization_3/batchnorm/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€T2,
*model/basemodel/dense_activation_1/Sigmoid∞
.model/basemodel/stream_2_input_drop/Identity_1Identityright_inputs*
T0*+
_output_shapes
:€€€€€€€€€}20
.model/basemodel/stream_2_input_drop/Identity_1∞
.model/basemodel/stream_1_input_drop/Identity_1Identityright_inputs*
T0*+
_output_shapes
:€€€€€€€€€}20
.model/basemodel/stream_1_input_drop/Identity_1∞
.model/basemodel/stream_0_input_drop/Identity_1Identityright_inputs*
T0*+
_output_shapes
:€€€€€€€€€}20
.model/basemodel/stream_0_input_drop/Identity_1љ
7model/basemodel/stream_2_conv_1/conv1d_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€29
7model/basemodel/stream_2_conv_1/conv1d_1/ExpandDims/dim≠
3model/basemodel/stream_2_conv_1/conv1d_1/ExpandDims
ExpandDims7model/basemodel/stream_2_input_drop/Identity_1:output:0@model/basemodel/stream_2_conv_1/conv1d_1/ExpandDims/dim:output:0*
T0*/
_output_shapes
:€€€€€€€€€}25
3model/basemodel/stream_2_conv_1/conv1d_1/ExpandDimsЬ
Dmodel/basemodel/stream_2_conv_1/conv1d_1/ExpandDims_1/ReadVariableOpReadVariableOpKmodel_basemodel_stream_2_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02F
Dmodel/basemodel/stream_2_conv_1/conv1d_1/ExpandDims_1/ReadVariableOpЄ
9model/basemodel/stream_2_conv_1/conv1d_1/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2;
9model/basemodel/stream_2_conv_1/conv1d_1/ExpandDims_1/dimњ
5model/basemodel/stream_2_conv_1/conv1d_1/ExpandDims_1
ExpandDimsLmodel/basemodel/stream_2_conv_1/conv1d_1/ExpandDims_1/ReadVariableOp:value:0Bmodel/basemodel/stream_2_conv_1/conv1d_1/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@27
5model/basemodel/stream_2_conv_1/conv1d_1/ExpandDims_1Њ
(model/basemodel/stream_2_conv_1/conv1d_1Conv2D<model/basemodel/stream_2_conv_1/conv1d_1/ExpandDims:output:0>model/basemodel/stream_2_conv_1/conv1d_1/ExpandDims_1:output:0*
T0*/
_output_shapes
:€€€€€€€€€}@*
paddingSAME*
strides
2*
(model/basemodel/stream_2_conv_1/conv1d_1ш
0model/basemodel/stream_2_conv_1/conv1d_1/SqueezeSqueeze1model/basemodel/stream_2_conv_1/conv1d_1:output:0*
T0*+
_output_shapes
:€€€€€€€€€}@*
squeeze_dims

э€€€€€€€€22
0model/basemodel/stream_2_conv_1/conv1d_1/Squeezeр
8model/basemodel/stream_2_conv_1/BiasAdd_1/ReadVariableOpReadVariableOp?model_basemodel_stream_2_conv_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02:
8model/basemodel/stream_2_conv_1/BiasAdd_1/ReadVariableOpФ
)model/basemodel/stream_2_conv_1/BiasAdd_1BiasAdd9model/basemodel/stream_2_conv_1/conv1d_1/Squeeze:output:0@model/basemodel/stream_2_conv_1/BiasAdd_1/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€}@2+
)model/basemodel/stream_2_conv_1/BiasAdd_1љ
7model/basemodel/stream_1_conv_1/conv1d_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€29
7model/basemodel/stream_1_conv_1/conv1d_1/ExpandDims/dim≠
3model/basemodel/stream_1_conv_1/conv1d_1/ExpandDims
ExpandDims7model/basemodel/stream_1_input_drop/Identity_1:output:0@model/basemodel/stream_1_conv_1/conv1d_1/ExpandDims/dim:output:0*
T0*/
_output_shapes
:€€€€€€€€€}25
3model/basemodel/stream_1_conv_1/conv1d_1/ExpandDimsЬ
Dmodel/basemodel/stream_1_conv_1/conv1d_1/ExpandDims_1/ReadVariableOpReadVariableOpKmodel_basemodel_stream_1_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02F
Dmodel/basemodel/stream_1_conv_1/conv1d_1/ExpandDims_1/ReadVariableOpЄ
9model/basemodel/stream_1_conv_1/conv1d_1/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2;
9model/basemodel/stream_1_conv_1/conv1d_1/ExpandDims_1/dimњ
5model/basemodel/stream_1_conv_1/conv1d_1/ExpandDims_1
ExpandDimsLmodel/basemodel/stream_1_conv_1/conv1d_1/ExpandDims_1/ReadVariableOp:value:0Bmodel/basemodel/stream_1_conv_1/conv1d_1/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@27
5model/basemodel/stream_1_conv_1/conv1d_1/ExpandDims_1Њ
(model/basemodel/stream_1_conv_1/conv1d_1Conv2D<model/basemodel/stream_1_conv_1/conv1d_1/ExpandDims:output:0>model/basemodel/stream_1_conv_1/conv1d_1/ExpandDims_1:output:0*
T0*/
_output_shapes
:€€€€€€€€€}@*
paddingSAME*
strides
2*
(model/basemodel/stream_1_conv_1/conv1d_1ш
0model/basemodel/stream_1_conv_1/conv1d_1/SqueezeSqueeze1model/basemodel/stream_1_conv_1/conv1d_1:output:0*
T0*+
_output_shapes
:€€€€€€€€€}@*
squeeze_dims

э€€€€€€€€22
0model/basemodel/stream_1_conv_1/conv1d_1/Squeezeр
8model/basemodel/stream_1_conv_1/BiasAdd_1/ReadVariableOpReadVariableOp?model_basemodel_stream_1_conv_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02:
8model/basemodel/stream_1_conv_1/BiasAdd_1/ReadVariableOpФ
)model/basemodel/stream_1_conv_1/BiasAdd_1BiasAdd9model/basemodel/stream_1_conv_1/conv1d_1/Squeeze:output:0@model/basemodel/stream_1_conv_1/BiasAdd_1/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€}@2+
)model/basemodel/stream_1_conv_1/BiasAdd_1љ
7model/basemodel/stream_0_conv_1/conv1d_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€29
7model/basemodel/stream_0_conv_1/conv1d_1/ExpandDims/dim≠
3model/basemodel/stream_0_conv_1/conv1d_1/ExpandDims
ExpandDims7model/basemodel/stream_0_input_drop/Identity_1:output:0@model/basemodel/stream_0_conv_1/conv1d_1/ExpandDims/dim:output:0*
T0*/
_output_shapes
:€€€€€€€€€}25
3model/basemodel/stream_0_conv_1/conv1d_1/ExpandDimsЬ
Dmodel/basemodel/stream_0_conv_1/conv1d_1/ExpandDims_1/ReadVariableOpReadVariableOpKmodel_basemodel_stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02F
Dmodel/basemodel/stream_0_conv_1/conv1d_1/ExpandDims_1/ReadVariableOpЄ
9model/basemodel/stream_0_conv_1/conv1d_1/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2;
9model/basemodel/stream_0_conv_1/conv1d_1/ExpandDims_1/dimњ
5model/basemodel/stream_0_conv_1/conv1d_1/ExpandDims_1
ExpandDimsLmodel/basemodel/stream_0_conv_1/conv1d_1/ExpandDims_1/ReadVariableOp:value:0Bmodel/basemodel/stream_0_conv_1/conv1d_1/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@27
5model/basemodel/stream_0_conv_1/conv1d_1/ExpandDims_1Њ
(model/basemodel/stream_0_conv_1/conv1d_1Conv2D<model/basemodel/stream_0_conv_1/conv1d_1/ExpandDims:output:0>model/basemodel/stream_0_conv_1/conv1d_1/ExpandDims_1:output:0*
T0*/
_output_shapes
:€€€€€€€€€}@*
paddingSAME*
strides
2*
(model/basemodel/stream_0_conv_1/conv1d_1ш
0model/basemodel/stream_0_conv_1/conv1d_1/SqueezeSqueeze1model/basemodel/stream_0_conv_1/conv1d_1:output:0*
T0*+
_output_shapes
:€€€€€€€€€}@*
squeeze_dims

э€€€€€€€€22
0model/basemodel/stream_0_conv_1/conv1d_1/Squeezeр
8model/basemodel/stream_0_conv_1/BiasAdd_1/ReadVariableOpReadVariableOp?model_basemodel_stream_0_conv_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02:
8model/basemodel/stream_0_conv_1/BiasAdd_1/ReadVariableOpФ
)model/basemodel/stream_0_conv_1/BiasAdd_1BiasAdd9model/basemodel/stream_0_conv_1/conv1d_1/Squeeze:output:0@model/basemodel/stream_0_conv_1/BiasAdd_1/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€}@2+
)model/basemodel/stream_0_conv_1/BiasAdd_1И
@model/basemodel/batch_normalization_2/batchnorm_1/ReadVariableOpReadVariableOpGmodel_basemodel_batch_normalization_2_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02B
@model/basemodel/batch_normalization_2/batchnorm_1/ReadVariableOpЈ
7model/basemodel/batch_normalization_2/batchnorm_1/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:29
7model/basemodel/batch_normalization_2/batchnorm_1/add/y®
5model/basemodel/batch_normalization_2/batchnorm_1/addAddV2Hmodel/basemodel/batch_normalization_2/batchnorm_1/ReadVariableOp:value:0@model/basemodel/batch_normalization_2/batchnorm_1/add/y:output:0*
T0*
_output_shapes
:@27
5model/basemodel/batch_normalization_2/batchnorm_1/addџ
7model/basemodel/batch_normalization_2/batchnorm_1/RsqrtRsqrt9model/basemodel/batch_normalization_2/batchnorm_1/add:z:0*
T0*
_output_shapes
:@29
7model/basemodel/batch_normalization_2/batchnorm_1/RsqrtФ
Dmodel/basemodel/batch_normalization_2/batchnorm_1/mul/ReadVariableOpReadVariableOpKmodel_basemodel_batch_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02F
Dmodel/basemodel/batch_normalization_2/batchnorm_1/mul/ReadVariableOp•
5model/basemodel/batch_normalization_2/batchnorm_1/mulMul;model/basemodel/batch_normalization_2/batchnorm_1/Rsqrt:y:0Lmodel/basemodel/batch_normalization_2/batchnorm_1/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@27
5model/basemodel/batch_normalization_2/batchnorm_1/mulЮ
7model/basemodel/batch_normalization_2/batchnorm_1/mul_1Mul2model/basemodel/stream_2_conv_1/BiasAdd_1:output:09model/basemodel/batch_normalization_2/batchnorm_1/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€}@29
7model/basemodel/batch_normalization_2/batchnorm_1/mul_1О
Bmodel/basemodel/batch_normalization_2/batchnorm_1/ReadVariableOp_1ReadVariableOpImodel_basemodel_batch_normalization_2_batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype02D
Bmodel/basemodel/batch_normalization_2/batchnorm_1/ReadVariableOp_1•
7model/basemodel/batch_normalization_2/batchnorm_1/mul_2MulJmodel/basemodel/batch_normalization_2/batchnorm_1/ReadVariableOp_1:value:09model/basemodel/batch_normalization_2/batchnorm_1/mul:z:0*
T0*
_output_shapes
:@29
7model/basemodel/batch_normalization_2/batchnorm_1/mul_2О
Bmodel/basemodel/batch_normalization_2/batchnorm_1/ReadVariableOp_2ReadVariableOpImodel_basemodel_batch_normalization_2_batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype02D
Bmodel/basemodel/batch_normalization_2/batchnorm_1/ReadVariableOp_2£
5model/basemodel/batch_normalization_2/batchnorm_1/subSubJmodel/basemodel/batch_normalization_2/batchnorm_1/ReadVariableOp_2:value:0;model/basemodel/batch_normalization_2/batchnorm_1/mul_2:z:0*
T0*
_output_shapes
:@27
5model/basemodel/batch_normalization_2/batchnorm_1/sub©
7model/basemodel/batch_normalization_2/batchnorm_1/add_1AddV2;model/basemodel/batch_normalization_2/batchnorm_1/mul_1:z:09model/basemodel/batch_normalization_2/batchnorm_1/sub:z:0*
T0*+
_output_shapes
:€€€€€€€€€}@29
7model/basemodel/batch_normalization_2/batchnorm_1/add_1И
@model/basemodel/batch_normalization_1/batchnorm_1/ReadVariableOpReadVariableOpGmodel_basemodel_batch_normalization_1_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02B
@model/basemodel/batch_normalization_1/batchnorm_1/ReadVariableOpЈ
7model/basemodel/batch_normalization_1/batchnorm_1/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:29
7model/basemodel/batch_normalization_1/batchnorm_1/add/y®
5model/basemodel/batch_normalization_1/batchnorm_1/addAddV2Hmodel/basemodel/batch_normalization_1/batchnorm_1/ReadVariableOp:value:0@model/basemodel/batch_normalization_1/batchnorm_1/add/y:output:0*
T0*
_output_shapes
:@27
5model/basemodel/batch_normalization_1/batchnorm_1/addџ
7model/basemodel/batch_normalization_1/batchnorm_1/RsqrtRsqrt9model/basemodel/batch_normalization_1/batchnorm_1/add:z:0*
T0*
_output_shapes
:@29
7model/basemodel/batch_normalization_1/batchnorm_1/RsqrtФ
Dmodel/basemodel/batch_normalization_1/batchnorm_1/mul/ReadVariableOpReadVariableOpKmodel_basemodel_batch_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02F
Dmodel/basemodel/batch_normalization_1/batchnorm_1/mul/ReadVariableOp•
5model/basemodel/batch_normalization_1/batchnorm_1/mulMul;model/basemodel/batch_normalization_1/batchnorm_1/Rsqrt:y:0Lmodel/basemodel/batch_normalization_1/batchnorm_1/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@27
5model/basemodel/batch_normalization_1/batchnorm_1/mulЮ
7model/basemodel/batch_normalization_1/batchnorm_1/mul_1Mul2model/basemodel/stream_1_conv_1/BiasAdd_1:output:09model/basemodel/batch_normalization_1/batchnorm_1/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€}@29
7model/basemodel/batch_normalization_1/batchnorm_1/mul_1О
Bmodel/basemodel/batch_normalization_1/batchnorm_1/ReadVariableOp_1ReadVariableOpImodel_basemodel_batch_normalization_1_batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype02D
Bmodel/basemodel/batch_normalization_1/batchnorm_1/ReadVariableOp_1•
7model/basemodel/batch_normalization_1/batchnorm_1/mul_2MulJmodel/basemodel/batch_normalization_1/batchnorm_1/ReadVariableOp_1:value:09model/basemodel/batch_normalization_1/batchnorm_1/mul:z:0*
T0*
_output_shapes
:@29
7model/basemodel/batch_normalization_1/batchnorm_1/mul_2О
Bmodel/basemodel/batch_normalization_1/batchnorm_1/ReadVariableOp_2ReadVariableOpImodel_basemodel_batch_normalization_1_batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype02D
Bmodel/basemodel/batch_normalization_1/batchnorm_1/ReadVariableOp_2£
5model/basemodel/batch_normalization_1/batchnorm_1/subSubJmodel/basemodel/batch_normalization_1/batchnorm_1/ReadVariableOp_2:value:0;model/basemodel/batch_normalization_1/batchnorm_1/mul_2:z:0*
T0*
_output_shapes
:@27
5model/basemodel/batch_normalization_1/batchnorm_1/sub©
7model/basemodel/batch_normalization_1/batchnorm_1/add_1AddV2;model/basemodel/batch_normalization_1/batchnorm_1/mul_1:z:09model/basemodel/batch_normalization_1/batchnorm_1/sub:z:0*
T0*+
_output_shapes
:€€€€€€€€€}@29
7model/basemodel/batch_normalization_1/batchnorm_1/add_1В
>model/basemodel/batch_normalization/batchnorm_1/ReadVariableOpReadVariableOpEmodel_basemodel_batch_normalization_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02@
>model/basemodel/batch_normalization/batchnorm_1/ReadVariableOp≥
5model/basemodel/batch_normalization/batchnorm_1/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:27
5model/basemodel/batch_normalization/batchnorm_1/add/y†
3model/basemodel/batch_normalization/batchnorm_1/addAddV2Fmodel/basemodel/batch_normalization/batchnorm_1/ReadVariableOp:value:0>model/basemodel/batch_normalization/batchnorm_1/add/y:output:0*
T0*
_output_shapes
:@25
3model/basemodel/batch_normalization/batchnorm_1/add’
5model/basemodel/batch_normalization/batchnorm_1/RsqrtRsqrt7model/basemodel/batch_normalization/batchnorm_1/add:z:0*
T0*
_output_shapes
:@27
5model/basemodel/batch_normalization/batchnorm_1/RsqrtО
Bmodel/basemodel/batch_normalization/batchnorm_1/mul/ReadVariableOpReadVariableOpImodel_basemodel_batch_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02D
Bmodel/basemodel/batch_normalization/batchnorm_1/mul/ReadVariableOpЭ
3model/basemodel/batch_normalization/batchnorm_1/mulMul9model/basemodel/batch_normalization/batchnorm_1/Rsqrt:y:0Jmodel/basemodel/batch_normalization/batchnorm_1/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@25
3model/basemodel/batch_normalization/batchnorm_1/mulШ
5model/basemodel/batch_normalization/batchnorm_1/mul_1Mul2model/basemodel/stream_0_conv_1/BiasAdd_1:output:07model/basemodel/batch_normalization/batchnorm_1/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€}@27
5model/basemodel/batch_normalization/batchnorm_1/mul_1И
@model/basemodel/batch_normalization/batchnorm_1/ReadVariableOp_1ReadVariableOpGmodel_basemodel_batch_normalization_batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype02B
@model/basemodel/batch_normalization/batchnorm_1/ReadVariableOp_1Э
5model/basemodel/batch_normalization/batchnorm_1/mul_2MulHmodel/basemodel/batch_normalization/batchnorm_1/ReadVariableOp_1:value:07model/basemodel/batch_normalization/batchnorm_1/mul:z:0*
T0*
_output_shapes
:@27
5model/basemodel/batch_normalization/batchnorm_1/mul_2И
@model/basemodel/batch_normalization/batchnorm_1/ReadVariableOp_2ReadVariableOpGmodel_basemodel_batch_normalization_batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype02B
@model/basemodel/batch_normalization/batchnorm_1/ReadVariableOp_2Ы
3model/basemodel/batch_normalization/batchnorm_1/subSubHmodel/basemodel/batch_normalization/batchnorm_1/ReadVariableOp_2:value:09model/basemodel/batch_normalization/batchnorm_1/mul_2:z:0*
T0*
_output_shapes
:@25
3model/basemodel/batch_normalization/batchnorm_1/sub°
5model/basemodel/batch_normalization/batchnorm_1/add_1AddV29model/basemodel/batch_normalization/batchnorm_1/mul_1:z:07model/basemodel/batch_normalization/batchnorm_1/sub:z:0*
T0*+
_output_shapes
:€€€€€€€€€}@27
5model/basemodel/batch_normalization/batchnorm_1/add_1≈
#model/basemodel/activation_2/Relu_1Relu;model/basemodel/batch_normalization_2/batchnorm_1/add_1:z:0*
T0*+
_output_shapes
:€€€€€€€€€}@2%
#model/basemodel/activation_2/Relu_1≈
#model/basemodel/activation_1/Relu_1Relu;model/basemodel/batch_normalization_1/batchnorm_1/add_1:z:0*
T0*+
_output_shapes
:€€€€€€€€€}@2%
#model/basemodel/activation_1/Relu_1њ
!model/basemodel/activation/Relu_1Relu9model/basemodel/batch_normalization/batchnorm_1/add_1:z:0*
T0*+
_output_shapes
:€€€€€€€€€}@2#
!model/basemodel/activation/Relu_1Ќ
*model/basemodel/stream_2_drop_1/Identity_1Identity1model/basemodel/activation_2/Relu_1:activations:0*
T0*+
_output_shapes
:€€€€€€€€€}@2,
*model/basemodel/stream_2_drop_1/Identity_1Ќ
*model/basemodel/stream_1_drop_1/Identity_1Identity1model/basemodel/activation_1/Relu_1:activations:0*
T0*+
_output_shapes
:€€€€€€€€€}@2,
*model/basemodel/stream_1_drop_1/Identity_1Ћ
*model/basemodel/stream_0_drop_1/Identity_1Identity/model/basemodel/activation/Relu_1:activations:0*
T0*+
_output_shapes
:€€€€€€€€€}@2,
*model/basemodel/stream_0_drop_1/Identity_1»
Amodel/basemodel/global_average_pooling1d/Mean_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2C
Amodel/basemodel/global_average_pooling1d/Mean_1/reduction_indicesЭ
/model/basemodel/global_average_pooling1d/Mean_1Mean3model/basemodel/stream_0_drop_1/Identity_1:output:0Jmodel/basemodel/global_average_pooling1d/Mean_1/reduction_indices:output:0*
T0*'
_output_shapes
:€€€€€€€€€@21
/model/basemodel/global_average_pooling1d/Mean_1ћ
Cmodel/basemodel/global_average_pooling1d_1/Mean_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2E
Cmodel/basemodel/global_average_pooling1d_1/Mean_1/reduction_indices£
1model/basemodel/global_average_pooling1d_1/Mean_1Mean3model/basemodel/stream_1_drop_1/Identity_1:output:0Lmodel/basemodel/global_average_pooling1d_1/Mean_1/reduction_indices:output:0*
T0*'
_output_shapes
:€€€€€€€€€@23
1model/basemodel/global_average_pooling1d_1/Mean_1ћ
Cmodel/basemodel/global_average_pooling1d_2/Mean_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2E
Cmodel/basemodel/global_average_pooling1d_2/Mean_1/reduction_indices£
1model/basemodel/global_average_pooling1d_2/Mean_1Mean3model/basemodel/stream_2_drop_1/Identity_1:output:0Lmodel/basemodel/global_average_pooling1d_2/Mean_1/reduction_indices:output:0*
T0*'
_output_shapes
:€€€€€€€€€@23
1model/basemodel/global_average_pooling1d_2/Mean_1Ш
)model/basemodel/concatenate/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :2+
)model/basemodel/concatenate/concat_1/axisъ
$model/basemodel/concatenate/concat_1ConcatV28model/basemodel/global_average_pooling1d/Mean_1:output:0:model/basemodel/global_average_pooling1d_1/Mean_1:output:0:model/basemodel/global_average_pooling1d_2/Mean_1:output:02model/basemodel/concatenate/concat_1/axis:output:0*
N*
T0*(
_output_shapes
:€€€€€€€€€ј2&
$model/basemodel/concatenate/concat_1∆
*model/basemodel/dense_1_dropout/Identity_1Identity-model/basemodel/concatenate/concat_1:output:0*
T0*(
_output_shapes
:€€€€€€€€€ј2,
*model/basemodel/dense_1_dropout/Identity_1Џ
/model/basemodel/dense_1/MatMul_1/ReadVariableOpReadVariableOp6model_basemodel_dense_1_matmul_readvariableop_resource*
_output_shapes
:	јT*
dtype021
/model/basemodel/dense_1/MatMul_1/ReadVariableOpо
 model/basemodel/dense_1/MatMul_1MatMul3model/basemodel/dense_1_dropout/Identity_1:output:07model/basemodel/dense_1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€T2"
 model/basemodel/dense_1/MatMul_1Ў
0model/basemodel/dense_1/BiasAdd_1/ReadVariableOpReadVariableOp7model_basemodel_dense_1_biasadd_readvariableop_resource*
_output_shapes
:T*
dtype022
0model/basemodel/dense_1/BiasAdd_1/ReadVariableOpй
!model/basemodel/dense_1/BiasAdd_1BiasAdd*model/basemodel/dense_1/MatMul_1:product:08model/basemodel/dense_1/BiasAdd_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€T2#
!model/basemodel/dense_1/BiasAdd_1И
@model/basemodel/batch_normalization_3/batchnorm_1/ReadVariableOpReadVariableOpGmodel_basemodel_batch_normalization_3_batchnorm_readvariableop_resource*
_output_shapes
:T*
dtype02B
@model/basemodel/batch_normalization_3/batchnorm_1/ReadVariableOpЈ
7model/basemodel/batch_normalization_3/batchnorm_1/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:29
7model/basemodel/batch_normalization_3/batchnorm_1/add/y®
5model/basemodel/batch_normalization_3/batchnorm_1/addAddV2Hmodel/basemodel/batch_normalization_3/batchnorm_1/ReadVariableOp:value:0@model/basemodel/batch_normalization_3/batchnorm_1/add/y:output:0*
T0*
_output_shapes
:T27
5model/basemodel/batch_normalization_3/batchnorm_1/addџ
7model/basemodel/batch_normalization_3/batchnorm_1/RsqrtRsqrt9model/basemodel/batch_normalization_3/batchnorm_1/add:z:0*
T0*
_output_shapes
:T29
7model/basemodel/batch_normalization_3/batchnorm_1/RsqrtФ
Dmodel/basemodel/batch_normalization_3/batchnorm_1/mul/ReadVariableOpReadVariableOpKmodel_basemodel_batch_normalization_3_batchnorm_mul_readvariableop_resource*
_output_shapes
:T*
dtype02F
Dmodel/basemodel/batch_normalization_3/batchnorm_1/mul/ReadVariableOp•
5model/basemodel/batch_normalization_3/batchnorm_1/mulMul;model/basemodel/batch_normalization_3/batchnorm_1/Rsqrt:y:0Lmodel/basemodel/batch_normalization_3/batchnorm_1/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:T27
5model/basemodel/batch_normalization_3/batchnorm_1/mulТ
7model/basemodel/batch_normalization_3/batchnorm_1/mul_1Mul*model/basemodel/dense_1/BiasAdd_1:output:09model/basemodel/batch_normalization_3/batchnorm_1/mul:z:0*
T0*'
_output_shapes
:€€€€€€€€€T29
7model/basemodel/batch_normalization_3/batchnorm_1/mul_1О
Bmodel/basemodel/batch_normalization_3/batchnorm_1/ReadVariableOp_1ReadVariableOpImodel_basemodel_batch_normalization_3_batchnorm_readvariableop_1_resource*
_output_shapes
:T*
dtype02D
Bmodel/basemodel/batch_normalization_3/batchnorm_1/ReadVariableOp_1•
7model/basemodel/batch_normalization_3/batchnorm_1/mul_2MulJmodel/basemodel/batch_normalization_3/batchnorm_1/ReadVariableOp_1:value:09model/basemodel/batch_normalization_3/batchnorm_1/mul:z:0*
T0*
_output_shapes
:T29
7model/basemodel/batch_normalization_3/batchnorm_1/mul_2О
Bmodel/basemodel/batch_normalization_3/batchnorm_1/ReadVariableOp_2ReadVariableOpImodel_basemodel_batch_normalization_3_batchnorm_readvariableop_2_resource*
_output_shapes
:T*
dtype02D
Bmodel/basemodel/batch_normalization_3/batchnorm_1/ReadVariableOp_2£
5model/basemodel/batch_normalization_3/batchnorm_1/subSubJmodel/basemodel/batch_normalization_3/batchnorm_1/ReadVariableOp_2:value:0;model/basemodel/batch_normalization_3/batchnorm_1/mul_2:z:0*
T0*
_output_shapes
:T27
5model/basemodel/batch_normalization_3/batchnorm_1/sub•
7model/basemodel/batch_normalization_3/batchnorm_1/add_1AddV2;model/basemodel/batch_normalization_3/batchnorm_1/mul_1:z:09model/basemodel/batch_normalization_3/batchnorm_1/sub:z:0*
T0*'
_output_shapes
:€€€€€€€€€T29
7model/basemodel/batch_normalization_3/batchnorm_1/add_1÷
,model/basemodel/dense_activation_1/Sigmoid_1Sigmoid;model/basemodel/batch_normalization_3/batchnorm_1/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€T2.
,model/basemodel/dense_activation_1/Sigmoid_1√
model/distance/subSub.model/basemodel/dense_activation_1/Sigmoid:y:00model/basemodel/dense_activation_1/Sigmoid_1:y:0*
T0*'
_output_shapes
:€€€€€€€€€T2
model/distance/subВ
model/distance/SquareSquaremodel/distance/sub:z:0*
T0*'
_output_shapes
:€€€€€€€€€T2
model/distance/SquareЧ
$model/distance/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2&
$model/distance/Sum/reduction_indicesЉ
model/distance/SumSummodel/distance/Square:y:0-model/distance/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:€€€€€€€€€*
	keep_dims(2
model/distance/Sumq
model/distance/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
model/distance/Const©
model/distance/MaximumMaximummodel/distance/Sum:output:0model/distance/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€2
model/distance/MaximumА
model/distance/SqrtSqrtmodel/distance/Maximum:z:0*
T0*'
_output_shapes
:€€€€€€€€€2
model/distance/Sqrtr
IdentityIdentitymodel/distance/Sqrt:y:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€2

IdentityД
NoOpNoOp=^model/basemodel/batch_normalization/batchnorm/ReadVariableOp?^model/basemodel/batch_normalization/batchnorm/ReadVariableOp_1?^model/basemodel/batch_normalization/batchnorm/ReadVariableOp_2A^model/basemodel/batch_normalization/batchnorm/mul/ReadVariableOp?^model/basemodel/batch_normalization/batchnorm_1/ReadVariableOpA^model/basemodel/batch_normalization/batchnorm_1/ReadVariableOp_1A^model/basemodel/batch_normalization/batchnorm_1/ReadVariableOp_2C^model/basemodel/batch_normalization/batchnorm_1/mul/ReadVariableOp?^model/basemodel/batch_normalization_1/batchnorm/ReadVariableOpA^model/basemodel/batch_normalization_1/batchnorm/ReadVariableOp_1A^model/basemodel/batch_normalization_1/batchnorm/ReadVariableOp_2C^model/basemodel/batch_normalization_1/batchnorm/mul/ReadVariableOpA^model/basemodel/batch_normalization_1/batchnorm_1/ReadVariableOpC^model/basemodel/batch_normalization_1/batchnorm_1/ReadVariableOp_1C^model/basemodel/batch_normalization_1/batchnorm_1/ReadVariableOp_2E^model/basemodel/batch_normalization_1/batchnorm_1/mul/ReadVariableOp?^model/basemodel/batch_normalization_2/batchnorm/ReadVariableOpA^model/basemodel/batch_normalization_2/batchnorm/ReadVariableOp_1A^model/basemodel/batch_normalization_2/batchnorm/ReadVariableOp_2C^model/basemodel/batch_normalization_2/batchnorm/mul/ReadVariableOpA^model/basemodel/batch_normalization_2/batchnorm_1/ReadVariableOpC^model/basemodel/batch_normalization_2/batchnorm_1/ReadVariableOp_1C^model/basemodel/batch_normalization_2/batchnorm_1/ReadVariableOp_2E^model/basemodel/batch_normalization_2/batchnorm_1/mul/ReadVariableOp?^model/basemodel/batch_normalization_3/batchnorm/ReadVariableOpA^model/basemodel/batch_normalization_3/batchnorm/ReadVariableOp_1A^model/basemodel/batch_normalization_3/batchnorm/ReadVariableOp_2C^model/basemodel/batch_normalization_3/batchnorm/mul/ReadVariableOpA^model/basemodel/batch_normalization_3/batchnorm_1/ReadVariableOpC^model/basemodel/batch_normalization_3/batchnorm_1/ReadVariableOp_1C^model/basemodel/batch_normalization_3/batchnorm_1/ReadVariableOp_2E^model/basemodel/batch_normalization_3/batchnorm_1/mul/ReadVariableOp/^model/basemodel/dense_1/BiasAdd/ReadVariableOp1^model/basemodel/dense_1/BiasAdd_1/ReadVariableOp.^model/basemodel/dense_1/MatMul/ReadVariableOp0^model/basemodel/dense_1/MatMul_1/ReadVariableOp7^model/basemodel/stream_0_conv_1/BiasAdd/ReadVariableOp9^model/basemodel/stream_0_conv_1/BiasAdd_1/ReadVariableOpC^model/basemodel/stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOpE^model/basemodel/stream_0_conv_1/conv1d_1/ExpandDims_1/ReadVariableOp7^model/basemodel/stream_1_conv_1/BiasAdd/ReadVariableOp9^model/basemodel/stream_1_conv_1/BiasAdd_1/ReadVariableOpC^model/basemodel/stream_1_conv_1/conv1d/ExpandDims_1/ReadVariableOpE^model/basemodel/stream_1_conv_1/conv1d_1/ExpandDims_1/ReadVariableOp7^model/basemodel/stream_2_conv_1/BiasAdd/ReadVariableOp9^model/basemodel/stream_2_conv_1/BiasAdd_1/ReadVariableOpC^model/basemodel/stream_2_conv_1/conv1d/ExpandDims_1/ReadVariableOpE^model/basemodel/stream_2_conv_1/conv1d_1/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*q
_input_shapes`
^:€€€€€€€€€}:€€€€€€€€€}: : : : : : : : : : : : : : : : : : : : : : : : 2|
<model/basemodel/batch_normalization/batchnorm/ReadVariableOp<model/basemodel/batch_normalization/batchnorm/ReadVariableOp2А
>model/basemodel/batch_normalization/batchnorm/ReadVariableOp_1>model/basemodel/batch_normalization/batchnorm/ReadVariableOp_12А
>model/basemodel/batch_normalization/batchnorm/ReadVariableOp_2>model/basemodel/batch_normalization/batchnorm/ReadVariableOp_22Д
@model/basemodel/batch_normalization/batchnorm/mul/ReadVariableOp@model/basemodel/batch_normalization/batchnorm/mul/ReadVariableOp2А
>model/basemodel/batch_normalization/batchnorm_1/ReadVariableOp>model/basemodel/batch_normalization/batchnorm_1/ReadVariableOp2Д
@model/basemodel/batch_normalization/batchnorm_1/ReadVariableOp_1@model/basemodel/batch_normalization/batchnorm_1/ReadVariableOp_12Д
@model/basemodel/batch_normalization/batchnorm_1/ReadVariableOp_2@model/basemodel/batch_normalization/batchnorm_1/ReadVariableOp_22И
Bmodel/basemodel/batch_normalization/batchnorm_1/mul/ReadVariableOpBmodel/basemodel/batch_normalization/batchnorm_1/mul/ReadVariableOp2А
>model/basemodel/batch_normalization_1/batchnorm/ReadVariableOp>model/basemodel/batch_normalization_1/batchnorm/ReadVariableOp2Д
@model/basemodel/batch_normalization_1/batchnorm/ReadVariableOp_1@model/basemodel/batch_normalization_1/batchnorm/ReadVariableOp_12Д
@model/basemodel/batch_normalization_1/batchnorm/ReadVariableOp_2@model/basemodel/batch_normalization_1/batchnorm/ReadVariableOp_22И
Bmodel/basemodel/batch_normalization_1/batchnorm/mul/ReadVariableOpBmodel/basemodel/batch_normalization_1/batchnorm/mul/ReadVariableOp2Д
@model/basemodel/batch_normalization_1/batchnorm_1/ReadVariableOp@model/basemodel/batch_normalization_1/batchnorm_1/ReadVariableOp2И
Bmodel/basemodel/batch_normalization_1/batchnorm_1/ReadVariableOp_1Bmodel/basemodel/batch_normalization_1/batchnorm_1/ReadVariableOp_12И
Bmodel/basemodel/batch_normalization_1/batchnorm_1/ReadVariableOp_2Bmodel/basemodel/batch_normalization_1/batchnorm_1/ReadVariableOp_22М
Dmodel/basemodel/batch_normalization_1/batchnorm_1/mul/ReadVariableOpDmodel/basemodel/batch_normalization_1/batchnorm_1/mul/ReadVariableOp2А
>model/basemodel/batch_normalization_2/batchnorm/ReadVariableOp>model/basemodel/batch_normalization_2/batchnorm/ReadVariableOp2Д
@model/basemodel/batch_normalization_2/batchnorm/ReadVariableOp_1@model/basemodel/batch_normalization_2/batchnorm/ReadVariableOp_12Д
@model/basemodel/batch_normalization_2/batchnorm/ReadVariableOp_2@model/basemodel/batch_normalization_2/batchnorm/ReadVariableOp_22И
Bmodel/basemodel/batch_normalization_2/batchnorm/mul/ReadVariableOpBmodel/basemodel/batch_normalization_2/batchnorm/mul/ReadVariableOp2Д
@model/basemodel/batch_normalization_2/batchnorm_1/ReadVariableOp@model/basemodel/batch_normalization_2/batchnorm_1/ReadVariableOp2И
Bmodel/basemodel/batch_normalization_2/batchnorm_1/ReadVariableOp_1Bmodel/basemodel/batch_normalization_2/batchnorm_1/ReadVariableOp_12И
Bmodel/basemodel/batch_normalization_2/batchnorm_1/ReadVariableOp_2Bmodel/basemodel/batch_normalization_2/batchnorm_1/ReadVariableOp_22М
Dmodel/basemodel/batch_normalization_2/batchnorm_1/mul/ReadVariableOpDmodel/basemodel/batch_normalization_2/batchnorm_1/mul/ReadVariableOp2А
>model/basemodel/batch_normalization_3/batchnorm/ReadVariableOp>model/basemodel/batch_normalization_3/batchnorm/ReadVariableOp2Д
@model/basemodel/batch_normalization_3/batchnorm/ReadVariableOp_1@model/basemodel/batch_normalization_3/batchnorm/ReadVariableOp_12Д
@model/basemodel/batch_normalization_3/batchnorm/ReadVariableOp_2@model/basemodel/batch_normalization_3/batchnorm/ReadVariableOp_22И
Bmodel/basemodel/batch_normalization_3/batchnorm/mul/ReadVariableOpBmodel/basemodel/batch_normalization_3/batchnorm/mul/ReadVariableOp2Д
@model/basemodel/batch_normalization_3/batchnorm_1/ReadVariableOp@model/basemodel/batch_normalization_3/batchnorm_1/ReadVariableOp2И
Bmodel/basemodel/batch_normalization_3/batchnorm_1/ReadVariableOp_1Bmodel/basemodel/batch_normalization_3/batchnorm_1/ReadVariableOp_12И
Bmodel/basemodel/batch_normalization_3/batchnorm_1/ReadVariableOp_2Bmodel/basemodel/batch_normalization_3/batchnorm_1/ReadVariableOp_22М
Dmodel/basemodel/batch_normalization_3/batchnorm_1/mul/ReadVariableOpDmodel/basemodel/batch_normalization_3/batchnorm_1/mul/ReadVariableOp2`
.model/basemodel/dense_1/BiasAdd/ReadVariableOp.model/basemodel/dense_1/BiasAdd/ReadVariableOp2d
0model/basemodel/dense_1/BiasAdd_1/ReadVariableOp0model/basemodel/dense_1/BiasAdd_1/ReadVariableOp2^
-model/basemodel/dense_1/MatMul/ReadVariableOp-model/basemodel/dense_1/MatMul/ReadVariableOp2b
/model/basemodel/dense_1/MatMul_1/ReadVariableOp/model/basemodel/dense_1/MatMul_1/ReadVariableOp2p
6model/basemodel/stream_0_conv_1/BiasAdd/ReadVariableOp6model/basemodel/stream_0_conv_1/BiasAdd/ReadVariableOp2t
8model/basemodel/stream_0_conv_1/BiasAdd_1/ReadVariableOp8model/basemodel/stream_0_conv_1/BiasAdd_1/ReadVariableOp2И
Bmodel/basemodel/stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOpBmodel/basemodel/stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp2М
Dmodel/basemodel/stream_0_conv_1/conv1d_1/ExpandDims_1/ReadVariableOpDmodel/basemodel/stream_0_conv_1/conv1d_1/ExpandDims_1/ReadVariableOp2p
6model/basemodel/stream_1_conv_1/BiasAdd/ReadVariableOp6model/basemodel/stream_1_conv_1/BiasAdd/ReadVariableOp2t
8model/basemodel/stream_1_conv_1/BiasAdd_1/ReadVariableOp8model/basemodel/stream_1_conv_1/BiasAdd_1/ReadVariableOp2И
Bmodel/basemodel/stream_1_conv_1/conv1d/ExpandDims_1/ReadVariableOpBmodel/basemodel/stream_1_conv_1/conv1d/ExpandDims_1/ReadVariableOp2М
Dmodel/basemodel/stream_1_conv_1/conv1d_1/ExpandDims_1/ReadVariableOpDmodel/basemodel/stream_1_conv_1/conv1d_1/ExpandDims_1/ReadVariableOp2p
6model/basemodel/stream_2_conv_1/BiasAdd/ReadVariableOp6model/basemodel/stream_2_conv_1/BiasAdd/ReadVariableOp2t
8model/basemodel/stream_2_conv_1/BiasAdd_1/ReadVariableOp8model/basemodel/stream_2_conv_1/BiasAdd_1/ReadVariableOp2И
Bmodel/basemodel/stream_2_conv_1/conv1d/ExpandDims_1/ReadVariableOpBmodel/basemodel/stream_2_conv_1/conv1d/ExpandDims_1/ReadVariableOp2М
Dmodel/basemodel/stream_2_conv_1/conv1d_1/ExpandDims_1/ReadVariableOpDmodel/basemodel/stream_2_conv_1/conv1d_1/ExpandDims_1/ReadVariableOp:X T
+
_output_shapes
:€€€€€€€€€}
%
_user_specified_nameleft_inputs:YU
+
_output_shapes
:€€€€€€€€€}
&
_user_specified_nameright_inputs
Ј
∞
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_108223

inputs/
!batchnorm_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@1
#batchnorm_readvariableop_1_resource:@1
#batchnorm_readvariableop_2_resource:@
identityИҐbatchnorm/ReadVariableOpҐbatchnorm/ReadVariableOp_1Ґbatchnorm/ReadVariableOp_2Ґbatchnorm/mul/ReadVariableOpТ
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
 *oГ:2
batchnorm/add/yИ
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2
batchnorm/mulГ
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€@2
batchnorm/mul_1Ш
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp_1Е
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:@2
batchnorm/mul_2Ш
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp_2Г
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2
batchnorm/subТ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€@2
batchnorm/add_1{
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€@2

Identity¬
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:€€€€€€€€€€€€€€€€€€@: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs
«
m
4__inference_stream_0_input_drop_layer_call_fn_107721

inputs
identityИҐStatefulPartitionedCallм
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€}* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_stream_0_input_drop_layer_call_and_return_conditional_losses_1050342
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€}2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€}22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:€€€€€€€€€}
 
_user_specified_nameinputs
д
ѕ
4__inference_batch_normalization_layer_call_fn_108043

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityИҐStatefulPartitionedCallЮ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€}@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_1048532
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€}@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€}@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:€€€€€€€€€}@
 
_user_specified_nameinputs
¬†
Є
E__inference_basemodel_layer_call_and_return_conditional_losses_105551
inputs_0
inputs_1
inputs_2,
stream_2_conv_1_105458:@$
stream_2_conv_1_105460:@,
stream_1_conv_1_105463:@$
stream_1_conv_1_105465:@,
stream_0_conv_1_105468:@$
stream_0_conv_1_105470:@*
batch_normalization_2_105473:@*
batch_normalization_2_105475:@*
batch_normalization_2_105477:@*
batch_normalization_2_105479:@*
batch_normalization_1_105482:@*
batch_normalization_1_105484:@*
batch_normalization_1_105486:@*
batch_normalization_1_105488:@(
batch_normalization_105491:@(
batch_normalization_105493:@(
batch_normalization_105495:@(
batch_normalization_105497:@!
dense_1_105511:	јT
dense_1_105513:T*
batch_normalization_3_105516:T*
batch_normalization_3_105518:T*
batch_normalization_3_105520:T*
batch_normalization_3_105522:T
identityИҐ+batch_normalization/StatefulPartitionedCallҐ-batch_normalization_1/StatefulPartitionedCallҐ-batch_normalization_2/StatefulPartitionedCallҐ-batch_normalization_3/StatefulPartitionedCallҐdense_1/StatefulPartitionedCallҐ-dense_1/kernel/Regularizer/Abs/ReadVariableOpҐ'stream_0_conv_1/StatefulPartitionedCallҐ5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpҐ'stream_0_drop_1/StatefulPartitionedCallҐ+stream_0_input_drop/StatefulPartitionedCallҐ'stream_1_conv_1/StatefulPartitionedCallҐ8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpҐ'stream_1_drop_1/StatefulPartitionedCallҐ+stream_1_input_drop/StatefulPartitionedCallҐ'stream_2_conv_1/StatefulPartitionedCallҐ5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOpҐ'stream_2_drop_1/StatefulPartitionedCallҐ+stream_2_input_drop/StatefulPartitionedCallЦ
+stream_2_input_drop/StatefulPartitionedCallStatefulPartitionedCallinputs_2*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€}* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_stream_2_input_drop_layer_call_and_return_conditional_losses_1050802-
+stream_2_input_drop/StatefulPartitionedCallƒ
+stream_1_input_drop/StatefulPartitionedCallStatefulPartitionedCallinputs_1,^stream_2_input_drop/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€}* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_stream_1_input_drop_layer_call_and_return_conditional_losses_1050572-
+stream_1_input_drop/StatefulPartitionedCallƒ
+stream_0_input_drop/StatefulPartitionedCallStatefulPartitionedCallinputs_0,^stream_1_input_drop/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€}* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_stream_0_input_drop_layer_call_and_return_conditional_losses_1050342-
+stream_0_input_drop/StatefulPartitionedCallм
'stream_2_conv_1/StatefulPartitionedCallStatefulPartitionedCall4stream_2_input_drop/StatefulPartitionedCall:output:0stream_2_conv_1_105458stream_2_conv_1_105460*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€}@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_stream_2_conv_1_layer_call_and_return_conditional_losses_1043222)
'stream_2_conv_1/StatefulPartitionedCallм
'stream_1_conv_1/StatefulPartitionedCallStatefulPartitionedCall4stream_1_input_drop/StatefulPartitionedCall:output:0stream_1_conv_1_105463stream_1_conv_1_105465*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€}@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_stream_1_conv_1_layer_call_and_return_conditional_losses_1043492)
'stream_1_conv_1/StatefulPartitionedCallм
'stream_0_conv_1/StatefulPartitionedCallStatefulPartitionedCall4stream_0_input_drop/StatefulPartitionedCall:output:0stream_0_conv_1_105468stream_0_conv_1_105470*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€}@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_stream_0_conv_1_layer_call_and_return_conditional_losses_1043762)
'stream_0_conv_1/StatefulPartitionedCallƒ
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall0stream_2_conv_1/StatefulPartitionedCall:output:0batch_normalization_2_105473batch_normalization_2_105475batch_normalization_2_105477batch_normalization_2_105479*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€}@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1049732/
-batch_normalization_2/StatefulPartitionedCallƒ
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall0stream_1_conv_1/StatefulPartitionedCall:output:0batch_normalization_1_105482batch_normalization_1_105484batch_normalization_1_105486batch_normalization_1_105488*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€}@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1049132/
-batch_normalization_1/StatefulPartitionedCallґ
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall0stream_0_conv_1/StatefulPartitionedCall:output:0batch_normalization_105491batch_normalization_105493batch_normalization_105495batch_normalization_105497*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€}@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_1048532-
+batch_normalization/StatefulPartitionedCallЧ
activation_2/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€}@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_activation_2_layer_call_and_return_conditional_losses_1044742
activation_2/PartitionedCallЧ
activation_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€}@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_activation_1_layer_call_and_return_conditional_losses_1044812
activation_1/PartitionedCallП
activation/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€}@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_activation_layer_call_and_return_conditional_losses_1044882
activation/PartitionedCall’
'stream_2_drop_1/StatefulPartitionedCallStatefulPartitionedCall%activation_2/PartitionedCall:output:0,^stream_0_input_drop/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€}@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_stream_2_drop_1_layer_call_and_return_conditional_losses_1047832)
'stream_2_drop_1/StatefulPartitionedCall—
'stream_1_drop_1/StatefulPartitionedCallStatefulPartitionedCall%activation_1/PartitionedCall:output:0(^stream_2_drop_1/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€}@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_stream_1_drop_1_layer_call_and_return_conditional_losses_1047602)
'stream_1_drop_1/StatefulPartitionedCallѕ
'stream_0_drop_1/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0(^stream_1_drop_1/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€}@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_stream_0_drop_1_layer_call_and_return_conditional_losses_1047372)
'stream_0_drop_1/StatefulPartitionedCall±
(global_average_pooling1d/PartitionedCallPartitionedCall0stream_0_drop_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *]
fXRV
T__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_1045162*
(global_average_pooling1d/PartitionedCallЈ
*global_average_pooling1d_1/PartitionedCallPartitionedCall0stream_1_drop_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *_
fZRX
V__inference_global_average_pooling1d_1_layer_call_and_return_conditional_losses_1045232,
*global_average_pooling1d_1/PartitionedCallЈ
*global_average_pooling1d_2/PartitionedCallPartitionedCall0stream_2_drop_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *_
fZRX
V__inference_global_average_pooling1d_2_layer_call_and_return_conditional_losses_1045302,
*global_average_pooling1d_2/PartitionedCallш
concatenate/PartitionedCallPartitionedCall1global_average_pooling1d/PartitionedCall:output:03global_average_pooling1d_1/PartitionedCall:output:03global_average_pooling1d_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€ј* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_concatenate_layer_call_and_return_conditional_losses_1045402
concatenate/PartitionedCallЛ
dense_1_dropout/PartitionedCallPartitionedCall$concatenate/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€ј* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_dense_1_dropout_layer_call_and_return_conditional_losses_1046912!
dense_1_dropout/PartitionedCallі
dense_1/StatefulPartitionedCallStatefulPartitionedCall(dense_1_dropout/PartitionedCall:output:0dense_1_105511dense_1_105513*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€T*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_1045652!
dense_1/StatefulPartitionedCallЄ
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0batch_normalization_3_105516batch_normalization_3_105518batch_normalization_3_105520batch_normalization_3_105522*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€T*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1041912/
-batch_normalization_3/StatefulPartitionedCall•
"dense_activation_1/PartitionedCallPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€T* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *W
fRRP
N__inference_dense_activation_1_layer_call_and_return_conditional_losses_1045852$
"dense_activation_1/PartitionedCall…
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpstream_0_conv_1_105468*"
_output_shapes
:@*
dtype027
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp√
&stream_0_conv_1/kernel/Regularizer/AbsAbs=stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_0_conv_1/kernel/Regularizer/Abs©
(stream_0_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_0_conv_1/kernel/Regularizer/Const„
&stream_0_conv_1/kernel/Regularizer/SumSum*stream_0_conv_1/kernel/Regularizer/Abs:y:01stream_0_conv_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/SumЩ
(stream_0_conv_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2*
(stream_0_conv_1/kernel/Regularizer/mul/x№
&stream_0_conv_1/kernel/Regularizer/mulMul1stream_0_conv_1/kernel/Regularizer/mul/x:output:0/stream_0_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/mulѕ
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpstream_1_conv_1_105463*"
_output_shapes
:@*
dtype02:
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpѕ
)stream_1_conv_1/kernel/Regularizer/SquareSquare@stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2+
)stream_1_conv_1/kernel/Regularizer/Square©
(stream_1_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_1_conv_1/kernel/Regularizer/ConstЏ
&stream_1_conv_1/kernel/Regularizer/SumSum-stream_1_conv_1/kernel/Regularizer/Square:y:01stream_1_conv_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&stream_1_conv_1/kernel/Regularizer/SumЩ
(stream_1_conv_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2*
(stream_1_conv_1/kernel/Regularizer/mul/x№
&stream_1_conv_1/kernel/Regularizer/mulMul1stream_1_conv_1/kernel/Regularizer/mul/x:output:0/stream_1_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_1_conv_1/kernel/Regularizer/mul…
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpstream_2_conv_1_105458*"
_output_shapes
:@*
dtype027
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp√
&stream_2_conv_1/kernel/Regularizer/AbsAbs=stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_2_conv_1/kernel/Regularizer/Abs©
(stream_2_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_2_conv_1/kernel/Regularizer/Const„
&stream_2_conv_1/kernel/Regularizer/SumSum*stream_2_conv_1/kernel/Regularizer/Abs:y:01stream_2_conv_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&stream_2_conv_1/kernel/Regularizer/SumЩ
(stream_2_conv_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2*
(stream_2_conv_1/kernel/Regularizer/mul/x№
&stream_2_conv_1/kernel/Regularizer/mulMul1stream_2_conv_1/kernel/Regularizer/mul/x:output:0/stream_2_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_2_conv_1/kernel/Regularizer/mulЃ
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_1_105511*
_output_shapes
:	јT*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOp®
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	јT2 
dense_1/kernel/Regularizer/AbsХ
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_1/kernel/Regularizer/ConstЈ
dense_1/kernel/Regularizer/SumSum"dense_1/kernel/Regularizer/Abs:y:0)dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/SumЙ
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2"
 dense_1/kernel/Regularizer/mul/xЉ
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/mulЖ
IdentityIdentity+dense_activation_1/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€T2

IdentityП
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall.^dense_1/kernel/Regularizer/Abs/ReadVariableOp(^stream_0_conv_1/StatefulPartitionedCall6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp(^stream_0_drop_1/StatefulPartitionedCall,^stream_0_input_drop/StatefulPartitionedCall(^stream_1_conv_1/StatefulPartitionedCall9^stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp(^stream_1_drop_1/StatefulPartitionedCall,^stream_1_input_drop/StatefulPartitionedCall(^stream_2_conv_1/StatefulPartitionedCall6^stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp(^stream_2_drop_1/StatefulPartitionedCall,^stream_2_input_drop/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*И
_input_shapesw
u:€€€€€€€€€}:€€€€€€€€€}:€€€€€€€€€}: : : : : : : : : : : : : : : : : : : : : : : : 2Z
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
+stream_2_input_drop/StatefulPartitionedCall+stream_2_input_drop/StatefulPartitionedCall:U Q
+
_output_shapes
:€€€€€€€€€}
"
_user_specified_name
inputs_0:UQ
+
_output_shapes
:€€€€€€€€€}
"
_user_specified_name
inputs_1:UQ
+
_output_shapes
:€€€€€€€€€}
"
_user_specified_name
inputs_2
и
—
6__inference_batch_normalization_1_layer_call_fn_108203

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityИҐStatefulPartitionedCall†
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€}@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1049132
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€}@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€}@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:€€€€€€€€€}@
 
_user_specified_nameinputs
М
m
O__inference_stream_2_input_drop_layer_call_and_return_conditional_losses_107753

inputs

identity_1^
IdentityIdentityinputs*
T0*+
_output_shapes
:€€€€€€€€€}2

Identitym

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:€€€€€€€€€}2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€}:S O
+
_output_shapes
:€€€€€€€€€}
 
_user_specified_nameinputs
ґ+
и
O__inference_batch_normalization_layer_call_and_return_conditional_losses_103633

inputs5
'assignmovingavg_readvariableop_resource:@7
)assignmovingavg_1_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@/
!batchnorm_readvariableop_resource:@
identityИҐAssignMovingAvgҐAssignMovingAvg/ReadVariableOpҐAssignMovingAvg_1Ґ AssignMovingAvg_1/ReadVariableOpҐbatchnorm/ReadVariableOpҐbatchnorm/mul/ReadVariableOpС
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indicesУ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(2
moments/meanА
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:@2
moments/StopGradient±
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€@2
moments/SquaredDifferenceЩ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indicesґ
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(2
moments/varianceБ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 2
moments/SqueezeЙ
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
„#<2
AssignMovingAvg/decay§
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOpШ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg/subП
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg/mulњ
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
„#<2
AssignMovingAvg_1/decay™
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOp†
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg_1/subЧ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg_1/mul…
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
 *oГ:2
batchnorm/add/yВ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2
batchnorm/mulГ
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€@2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:@2
batchnorm/mul_2Т
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOpБ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2
batchnorm/subТ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€@2
batchnorm/add_1{
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€@2

Identityт
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:€€€€€€€€€€€€€€€€€€@: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs
Џ
—
6__inference_batch_normalization_3_layer_call_fn_108672

inputs
unknown:T
	unknown_0:T
	unknown_1:T
	unknown_2:T
identityИҐStatefulPartitionedCallЮ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€T*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1041312
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€T2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€T: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€T
 
_user_specified_nameinputs
х
∞
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_108625

inputs/
!batchnorm_readvariableop_resource:T3
%batchnorm_mul_readvariableop_resource:T1
#batchnorm_readvariableop_1_resource:T1
#batchnorm_readvariableop_2_resource:T
identityИҐbatchnorm/ReadVariableOpҐbatchnorm/ReadVariableOp_1Ґbatchnorm/ReadVariableOp_2Ґbatchnorm/mul/ReadVariableOpТ
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
 *oГ:2
batchnorm/add/yИ
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:T2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:T2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:T*
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:T2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:€€€€€€€€€T2
batchnorm/mul_1Ш
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:T*
dtype02
batchnorm/ReadVariableOp_1Е
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:T2
batchnorm/mul_2Ш
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:T*
dtype02
batchnorm/ReadVariableOp_2Г
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:T2
batchnorm/subЕ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:€€€€€€€€€T2
batchnorm/add_1n
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€T2

Identity¬
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€T: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€T
 
_user_specified_nameinputs
Џ
“
K__inference_stream_0_conv_1_layer_call_and_return_conditional_losses_104376

inputsA
+conv1d_expanddims_1_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identityИҐBiasAdd/ReadVariableOpҐ"conv1d/ExpandDims_1/ReadVariableOpҐ5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€2
conv1d/ExpandDims/dimЦ
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:€€€€€€€€€}2
conv1d/ExpandDimsЄ
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
conv1d/ExpandDims_1/dimЈ
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@2
conv1d/ExpandDims_1ґ
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:€€€€€€€€€}@*
paddingSAME*
strides
2
conv1dТ
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:€€€€€€€€€}@*
squeeze_dims

э€€€€€€€€2
conv1d/SqueezeМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpМ
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€}@2	
BiasAddё
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype027
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp√
&stream_0_conv_1/kernel/Regularizer/AbsAbs=stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_0_conv_1/kernel/Regularizer/Abs©
(stream_0_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_0_conv_1/kernel/Regularizer/Const„
&stream_0_conv_1/kernel/Regularizer/SumSum*stream_0_conv_1/kernel/Regularizer/Abs:y:01stream_0_conv_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/SumЩ
(stream_0_conv_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2*
(stream_0_conv_1/kernel/Regularizer/mul/x№
&stream_0_conv_1/kernel/Regularizer/mulMul1stream_0_conv_1/kernel/Regularizer/mul/x:output:0/stream_0_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/mulo
IdentityIdentityBiasAdd:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€}@2

Identityƒ
NoOpNoOp^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€}: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp2n
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:S O
+
_output_shapes
:€€€€€€€€€}
 
_user_specified_nameinputs
ЧI
∞	
A__inference_model_layer_call_and_return_conditional_losses_105673

inputs
inputs_1&
basemodel_105560:@
basemodel_105562:@&
basemodel_105564:@
basemodel_105566:@&
basemodel_105568:@
basemodel_105570:@
basemodel_105572:@
basemodel_105574:@
basemodel_105576:@
basemodel_105578:@
basemodel_105580:@
basemodel_105582:@
basemodel_105584:@
basemodel_105586:@
basemodel_105588:@
basemodel_105590:@
basemodel_105592:@
basemodel_105594:@#
basemodel_105596:	јT
basemodel_105598:T
basemodel_105600:T
basemodel_105602:T
basemodel_105604:T
basemodel_105606:T
identityИҐ!basemodel/StatefulPartitionedCallҐ#basemodel/StatefulPartitionedCall_1Ґ-dense_1/kernel/Regularizer/Abs/ReadVariableOpҐ5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpҐ8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpҐ5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOpж
!basemodel/StatefulPartitionedCallStatefulPartitionedCallinputsinputsinputsbasemodel_105560basemodel_105562basemodel_105564basemodel_105566basemodel_105568basemodel_105570basemodel_105572basemodel_105574basemodel_105576basemodel_105578basemodel_105580basemodel_105582basemodel_105584basemodel_105586basemodel_105588basemodel_105590basemodel_105592basemodel_105594basemodel_105596basemodel_105598basemodel_105600basemodel_105602basemodel_105604basemodel_105606*&
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€T*:
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_basemodel_layer_call_and_return_conditional_losses_1046122#
!basemodel/StatefulPartitionedCallр
#basemodel/StatefulPartitionedCall_1StatefulPartitionedCallinputs_1inputs_1inputs_1basemodel_105560basemodel_105562basemodel_105564basemodel_105566basemodel_105568basemodel_105570basemodel_105572basemodel_105574basemodel_105576basemodel_105578basemodel_105580basemodel_105582basemodel_105584basemodel_105586basemodel_105588basemodel_105590basemodel_105592basemodel_105594basemodel_105596basemodel_105598basemodel_105600basemodel_105602basemodel_105604basemodel_105606*&
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€T*:
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_basemodel_layer_call_and_return_conditional_losses_1046122%
#basemodel/StatefulPartitionedCall_1™
distance/PartitionedCallPartitionedCall*basemodel/StatefulPartitionedCall:output:0,basemodel/StatefulPartitionedCall_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_distance_layer_call_and_return_conditional_losses_1056462
distance/PartitionedCall√
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpbasemodel_105568*"
_output_shapes
:@*
dtype027
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp√
&stream_0_conv_1/kernel/Regularizer/AbsAbs=stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_0_conv_1/kernel/Regularizer/Abs©
(stream_0_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_0_conv_1/kernel/Regularizer/Const„
&stream_0_conv_1/kernel/Regularizer/SumSum*stream_0_conv_1/kernel/Regularizer/Abs:y:01stream_0_conv_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/SumЩ
(stream_0_conv_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2*
(stream_0_conv_1/kernel/Regularizer/mul/x№
&stream_0_conv_1/kernel/Regularizer/mulMul1stream_0_conv_1/kernel/Regularizer/mul/x:output:0/stream_0_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/mul…
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpbasemodel_105564*"
_output_shapes
:@*
dtype02:
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpѕ
)stream_1_conv_1/kernel/Regularizer/SquareSquare@stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2+
)stream_1_conv_1/kernel/Regularizer/Square©
(stream_1_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_1_conv_1/kernel/Regularizer/ConstЏ
&stream_1_conv_1/kernel/Regularizer/SumSum-stream_1_conv_1/kernel/Regularizer/Square:y:01stream_1_conv_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&stream_1_conv_1/kernel/Regularizer/SumЩ
(stream_1_conv_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2*
(stream_1_conv_1/kernel/Regularizer/mul/x№
&stream_1_conv_1/kernel/Regularizer/mulMul1stream_1_conv_1/kernel/Regularizer/mul/x:output:0/stream_1_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_1_conv_1/kernel/Regularizer/mul√
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpbasemodel_105560*"
_output_shapes
:@*
dtype027
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp√
&stream_2_conv_1/kernel/Regularizer/AbsAbs=stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_2_conv_1/kernel/Regularizer/Abs©
(stream_2_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_2_conv_1/kernel/Regularizer/Const„
&stream_2_conv_1/kernel/Regularizer/SumSum*stream_2_conv_1/kernel/Regularizer/Abs:y:01stream_2_conv_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&stream_2_conv_1/kernel/Regularizer/SumЩ
(stream_2_conv_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2*
(stream_2_conv_1/kernel/Regularizer/mul/x№
&stream_2_conv_1/kernel/Regularizer/mulMul1stream_2_conv_1/kernel/Regularizer/mul/x:output:0/stream_2_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_2_conv_1/kernel/Regularizer/mul∞
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpbasemodel_105596*
_output_shapes
:	јT*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOp®
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	јT2 
dense_1/kernel/Regularizer/AbsХ
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_1/kernel/Regularizer/ConstЈ
dense_1/kernel/Regularizer/SumSum"dense_1/kernel/Regularizer/Abs:y:0)dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/SumЙ
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2"
 dense_1/kernel/Regularizer/mul/xЉ
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/mul|
IdentityIdentity!distance/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identityу
NoOpNoOp"^basemodel/StatefulPartitionedCall$^basemodel/StatefulPartitionedCall_1.^dense_1/kernel/Regularizer/Abs/ReadVariableOp6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp9^stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp6^stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*q
_input_shapes`
^:€€€€€€€€€}:€€€€€€€€€}: : : : : : : : : : : : : : : : : : : : : : : : 2F
!basemodel/StatefulPartitionedCall!basemodel/StatefulPartitionedCall2J
#basemodel/StatefulPartitionedCall_1#basemodel/StatefulPartitionedCall_12^
-dense_1/kernel/Regularizer/Abs/ReadVariableOp-dense_1/kernel/Regularizer/Abs/ReadVariableOp2n
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp2t
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp2n
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp:S O
+
_output_shapes
:€€€€€€€€€}
 
_user_specified_nameinputs:SO
+
_output_shapes
:€€€€€€€€€}
 
_user_specified_nameinputs
Ѓ
Б
*__inference_basemodel_layer_call_fn_107658
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

unknown_17:	јT

unknown_18:T

unknown_19:T

unknown_20:T

unknown_21:T

unknown_22:T
identityИҐStatefulPartitionedCall≥
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
:€€€€€€€€€T*2
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_basemodel_layer_call_and_return_conditional_losses_1052432
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€T2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*И
_input_shapesw
u:€€€€€€€€€}:€€€€€€€€€}:€€€€€€€€€}: : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
+
_output_shapes
:€€€€€€€€€}
"
_user_specified_name
inputs/0:UQ
+
_output_shapes
:€€€€€€€€€}
"
_user_specified_name
inputs/1:UQ
+
_output_shapes
:€€€€€€€€€}
"
_user_specified_name
inputs/2
Й
∞
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_108117

inputs/
!batchnorm_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@1
#batchnorm_readvariableop_1_resource:@1
#batchnorm_readvariableop_2_resource:@
identityИҐbatchnorm/ReadVariableOpҐbatchnorm/ReadVariableOp_1Ґbatchnorm/ReadVariableOp_2Ґbatchnorm/mul/ReadVariableOpТ
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
 *oГ:2
batchnorm/add/yИ
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2
batchnorm/mulz
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€}@2
batchnorm/mul_1Ш
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp_1Е
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:@2
batchnorm/mul_2Ш
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp_2Г
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2
batchnorm/subЙ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*+
_output_shapes
:€€€€€€€€€}@2
batchnorm/add_1r
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€}@2

Identity¬
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€}@: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:S O
+
_output_shapes
:€€€€€€€€€}@
 
_user_specified_nameinputs
у
•
C__inference_dense_1_layer_call_and_return_conditional_losses_104565

inputs1
matmul_readvariableop_resource:	јT-
biasadd_readvariableop_resource:T
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpҐ-dense_1/kernel/Regularizer/Abs/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	јT*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€T2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:T*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€T2	
BiasAddЊ
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	јT*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOp®
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	јT2 
dense_1/kernel/Regularizer/AbsХ
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_1/kernel/Regularizer/ConstЈ
dense_1/kernel/Regularizer/SumSum"dense_1/kernel/Regularizer/Abs:y:0)dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/SumЙ
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2"
 dense_1/kernel/Regularizer/mul/xЉ
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/mulk
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€T2

Identityѓ
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp.^dense_1/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€ј: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2^
-dense_1/kernel/Regularizer/Abs/ReadVariableOp-dense_1/kernel/Regularizer/Abs/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€ј
 
_user_specified_nameinputs
Ћ
f
,__inference_concatenate_layer_call_fn_108555
inputs_0
inputs_1
inputs_2
identityб
PartitionedCallPartitionedCallinputs_0inputs_1inputs_2*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€ј* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_concatenate_layer_call_and_return_conditional_losses_1045402
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:€€€€€€€€€ј2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:€€€€€€€€€@:€€€€€€€€€@:€€€€€€€€€@:Q M
'
_output_shapes
:€€€€€€€€€@
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:€€€€€€€€€@
"
_user_specified_name
inputs/1:QM
'
_output_shapes
:€€€€€€€€€@
"
_user_specified_name
inputs/2
 
љ
__inference_loss_fn_2_108728T
>stream_2_conv_1_kernel_regularizer_abs_readvariableop_resource:@
identityИҐ5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOpс
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp>stream_2_conv_1_kernel_regularizer_abs_readvariableop_resource*"
_output_shapes
:@*
dtype027
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp√
&stream_2_conv_1/kernel/Regularizer/AbsAbs=stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_2_conv_1/kernel/Regularizer/Abs©
(stream_2_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_2_conv_1/kernel/Regularizer/Const„
&stream_2_conv_1/kernel/Regularizer/SumSum*stream_2_conv_1/kernel/Regularizer/Abs:y:01stream_2_conv_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&stream_2_conv_1/kernel/Regularizer/SumЩ
(stream_2_conv_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2*
(stream_2_conv_1/kernel/Regularizer/mul/x№
&stream_2_conv_1/kernel/Regularizer/mulMul1stream_2_conv_1/kernel/Regularizer/mul/x:output:0/stream_2_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_2_conv_1/kernel/Regularizer/mult
IdentityIdentity*stream_2_conv_1/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: 2

IdentityЖ
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
њ
i
0__inference_stream_0_drop_1_layer_call_fn_108420

inputs
identityИҐStatefulPartitionedCallи
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€}@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_stream_0_drop_1_layer_call_and_return_conditional_losses_1047372
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€}@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€}@22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:€€€€€€€€€}@
 
_user_specified_nameinputs
с
√
__inference_loss_fn_1_108717W
Astream_1_conv_1_kernel_regularizer_square_readvariableop_resource:@
identityИҐ8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpъ
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpAstream_1_conv_1_kernel_regularizer_square_readvariableop_resource*"
_output_shapes
:@*
dtype02:
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpѕ
)stream_1_conv_1/kernel/Regularizer/SquareSquare@stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2+
)stream_1_conv_1/kernel/Regularizer/Square©
(stream_1_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_1_conv_1/kernel/Regularizer/ConstЏ
&stream_1_conv_1/kernel/Regularizer/SumSum-stream_1_conv_1/kernel/Regularizer/Square:y:01stream_1_conv_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&stream_1_conv_1/kernel/Regularizer/SumЩ
(stream_1_conv_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2*
(stream_1_conv_1/kernel/Regularizer/mul/x№
&stream_1_conv_1/kernel/Regularizer/mulMul1stream_1_conv_1/kernel/Regularizer/mul/x:output:0/stream_1_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_1_conv_1/kernel/Regularizer/mult
IdentityIdentity*stream_1_conv_1/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: 2

IdentityЙ
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
ы
’
K__inference_stream_1_conv_1_layer_call_and_return_conditional_losses_104349

inputsA
+conv1d_expanddims_1_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identityИҐBiasAdd/ReadVariableOpҐ"conv1d/ExpandDims_1/ReadVariableOpҐ8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€2
conv1d/ExpandDims/dimЦ
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:€€€€€€€€€}2
conv1d/ExpandDimsЄ
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
conv1d/ExpandDims_1/dimЈ
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@2
conv1d/ExpandDims_1ґ
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:€€€€€€€€€}@*
paddingSAME*
strides
2
conv1dТ
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:€€€€€€€€€}@*
squeeze_dims

э€€€€€€€€2
conv1d/SqueezeМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpМ
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€}@2	
BiasAddд
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02:
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpѕ
)stream_1_conv_1/kernel/Regularizer/SquareSquare@stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2+
)stream_1_conv_1/kernel/Regularizer/Square©
(stream_1_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_1_conv_1/kernel/Regularizer/ConstЏ
&stream_1_conv_1/kernel/Regularizer/SumSum-stream_1_conv_1/kernel/Regularizer/Square:y:01stream_1_conv_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&stream_1_conv_1/kernel/Regularizer/SumЩ
(stream_1_conv_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2*
(stream_1_conv_1/kernel/Regularizer/mul/x№
&stream_1_conv_1/kernel/Regularizer/mulMul1stream_1_conv_1/kernel/Regularizer/mul/x:output:0/stream_1_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_1_conv_1/kernel/Regularizer/mulo
IdentityIdentityBiasAdd:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€}@2

Identity«
NoOpNoOp^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp9^stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€}: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp2t
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp:S O
+
_output_shapes
:€€€€€€€€€}
 
_user_specified_nameinputs
з
Б
G__inference_concatenate_layer_call_and_return_conditional_losses_108548
inputs_0
inputs_1
inputs_2
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axisМ
concatConcatV2inputs_0inputs_1inputs_2concat/axis:output:0*
N*
T0*(
_output_shapes
:€€€€€€€€€ј2
concatd
IdentityIdentityconcat:output:0*
T0*(
_output_shapes
:€€€€€€€€€ј2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:€€€€€€€€€@:€€€€€€€€€@:€€€€€€€€€@:Q M
'
_output_shapes
:€€€€€€€€€@
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:€€€€€€€€€@
"
_user_specified_name
inputs/1:QM
'
_output_shapes
:€€€€€€€€€@
"
_user_specified_name
inputs/2
Џ
“
K__inference_stream_2_conv_1_layer_call_and_return_conditional_losses_104322

inputsA
+conv1d_expanddims_1_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identityИҐBiasAdd/ReadVariableOpҐ"conv1d/ExpandDims_1/ReadVariableOpҐ5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€2
conv1d/ExpandDims/dimЦ
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:€€€€€€€€€}2
conv1d/ExpandDimsЄ
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
conv1d/ExpandDims_1/dimЈ
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@2
conv1d/ExpandDims_1ґ
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:€€€€€€€€€}@*
paddingSAME*
strides
2
conv1dТ
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:€€€€€€€€€}@*
squeeze_dims

э€€€€€€€€2
conv1d/SqueezeМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpМ
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€}@2	
BiasAddё
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype027
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp√
&stream_2_conv_1/kernel/Regularizer/AbsAbs=stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_2_conv_1/kernel/Regularizer/Abs©
(stream_2_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_2_conv_1/kernel/Regularizer/Const„
&stream_2_conv_1/kernel/Regularizer/SumSum*stream_2_conv_1/kernel/Regularizer/Abs:y:01stream_2_conv_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&stream_2_conv_1/kernel/Regularizer/SumЩ
(stream_2_conv_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2*
(stream_2_conv_1/kernel/Regularizer/mul/x№
&stream_2_conv_1/kernel/Regularizer/mulMul1stream_2_conv_1/kernel/Regularizer/mul/x:output:0/stream_2_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_2_conv_1/kernel/Regularizer/mulo
IdentityIdentityBiasAdd:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€}@2

Identityƒ
NoOpNoOp^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp6^stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€}: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp2n
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp:S O
+
_output_shapes
:€€€€€€€€€}
 
_user_specified_nameinputs
Ц
°
0__inference_stream_0_conv_1_layer_call_fn_107811

inputs
unknown:@
	unknown_0:@
identityИҐStatefulPartitionedCallВ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€}@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_stream_0_conv_1_layer_call_and_return_conditional_losses_1043762
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€}@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€}: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:€€€€€€€€€}
 
_user_specified_nameinputs
Ё
L
0__inference_stream_0_drop_1_layer_call_fn_108415

inputs
identity–
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€}@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_stream_0_drop_1_layer_call_and_return_conditional_losses_1045092
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:€€€€€€€€€}@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€}@:S O
+
_output_shapes
:€€€€€€€€€}@
 
_user_specified_nameinputs
”
G
+__inference_activation_layer_call_fn_108373

inputs
identityЋ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€}@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_activation_layer_call_and_return_conditional_losses_1044882
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:€€€€€€€€€}@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€}@:S O
+
_output_shapes
:€€€€€€€€€}@
 
_user_specified_nameinputs
є	
n
D__inference_distance_layer_call_and_return_conditional_losses_105646

inputs
inputs_1
identityU
subSubinputsinputs_1*
T0*'
_output_shapes
:€€€€€€€€€T2
subU
SquareSquaresub:z:0*
T0*'
_output_shapes
:€€€€€€€€€T2
Squarey
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2
Sum/reduction_indicesА
SumSum
Square:y:0Sum/reduction_indices:output:0*
T0*'
_output_shapes
:€€€€€€€€€*
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
:€€€€€€€€€2	
MaximumS
SqrtSqrtMaximum:z:0*
T0*'
_output_shapes
:€€€€€€€€€2
Sqrt\
IdentityIdentitySqrt:y:0*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:€€€€€€€€€T:€€€€€€€€€T:O K
'
_output_shapes
:€€€€€€€€€T
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€T
 
_user_specified_nameinputs
Є†
ґ
E__inference_basemodel_layer_call_and_return_conditional_losses_105243

inputs
inputs_1
inputs_2,
stream_2_conv_1_105150:@$
stream_2_conv_1_105152:@,
stream_1_conv_1_105155:@$
stream_1_conv_1_105157:@,
stream_0_conv_1_105160:@$
stream_0_conv_1_105162:@*
batch_normalization_2_105165:@*
batch_normalization_2_105167:@*
batch_normalization_2_105169:@*
batch_normalization_2_105171:@*
batch_normalization_1_105174:@*
batch_normalization_1_105176:@*
batch_normalization_1_105178:@*
batch_normalization_1_105180:@(
batch_normalization_105183:@(
batch_normalization_105185:@(
batch_normalization_105187:@(
batch_normalization_105189:@!
dense_1_105203:	јT
dense_1_105205:T*
batch_normalization_3_105208:T*
batch_normalization_3_105210:T*
batch_normalization_3_105212:T*
batch_normalization_3_105214:T
identityИҐ+batch_normalization/StatefulPartitionedCallҐ-batch_normalization_1/StatefulPartitionedCallҐ-batch_normalization_2/StatefulPartitionedCallҐ-batch_normalization_3/StatefulPartitionedCallҐdense_1/StatefulPartitionedCallҐ-dense_1/kernel/Regularizer/Abs/ReadVariableOpҐ'stream_0_conv_1/StatefulPartitionedCallҐ5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpҐ'stream_0_drop_1/StatefulPartitionedCallҐ+stream_0_input_drop/StatefulPartitionedCallҐ'stream_1_conv_1/StatefulPartitionedCallҐ8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpҐ'stream_1_drop_1/StatefulPartitionedCallҐ+stream_1_input_drop/StatefulPartitionedCallҐ'stream_2_conv_1/StatefulPartitionedCallҐ5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOpҐ'stream_2_drop_1/StatefulPartitionedCallҐ+stream_2_input_drop/StatefulPartitionedCallЦ
+stream_2_input_drop/StatefulPartitionedCallStatefulPartitionedCallinputs_2*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€}* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_stream_2_input_drop_layer_call_and_return_conditional_losses_1050802-
+stream_2_input_drop/StatefulPartitionedCallƒ
+stream_1_input_drop/StatefulPartitionedCallStatefulPartitionedCallinputs_1,^stream_2_input_drop/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€}* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_stream_1_input_drop_layer_call_and_return_conditional_losses_1050572-
+stream_1_input_drop/StatefulPartitionedCall¬
+stream_0_input_drop/StatefulPartitionedCallStatefulPartitionedCallinputs,^stream_1_input_drop/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€}* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_stream_0_input_drop_layer_call_and_return_conditional_losses_1050342-
+stream_0_input_drop/StatefulPartitionedCallм
'stream_2_conv_1/StatefulPartitionedCallStatefulPartitionedCall4stream_2_input_drop/StatefulPartitionedCall:output:0stream_2_conv_1_105150stream_2_conv_1_105152*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€}@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_stream_2_conv_1_layer_call_and_return_conditional_losses_1043222)
'stream_2_conv_1/StatefulPartitionedCallм
'stream_1_conv_1/StatefulPartitionedCallStatefulPartitionedCall4stream_1_input_drop/StatefulPartitionedCall:output:0stream_1_conv_1_105155stream_1_conv_1_105157*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€}@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_stream_1_conv_1_layer_call_and_return_conditional_losses_1043492)
'stream_1_conv_1/StatefulPartitionedCallм
'stream_0_conv_1/StatefulPartitionedCallStatefulPartitionedCall4stream_0_input_drop/StatefulPartitionedCall:output:0stream_0_conv_1_105160stream_0_conv_1_105162*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€}@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_stream_0_conv_1_layer_call_and_return_conditional_losses_1043762)
'stream_0_conv_1/StatefulPartitionedCallƒ
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall0stream_2_conv_1/StatefulPartitionedCall:output:0batch_normalization_2_105165batch_normalization_2_105167batch_normalization_2_105169batch_normalization_2_105171*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€}@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1049732/
-batch_normalization_2/StatefulPartitionedCallƒ
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall0stream_1_conv_1/StatefulPartitionedCall:output:0batch_normalization_1_105174batch_normalization_1_105176batch_normalization_1_105178batch_normalization_1_105180*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€}@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1049132/
-batch_normalization_1/StatefulPartitionedCallґ
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall0stream_0_conv_1/StatefulPartitionedCall:output:0batch_normalization_105183batch_normalization_105185batch_normalization_105187batch_normalization_105189*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€}@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_1048532-
+batch_normalization/StatefulPartitionedCallЧ
activation_2/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€}@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_activation_2_layer_call_and_return_conditional_losses_1044742
activation_2/PartitionedCallЧ
activation_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€}@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_activation_1_layer_call_and_return_conditional_losses_1044812
activation_1/PartitionedCallП
activation/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€}@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_activation_layer_call_and_return_conditional_losses_1044882
activation/PartitionedCall’
'stream_2_drop_1/StatefulPartitionedCallStatefulPartitionedCall%activation_2/PartitionedCall:output:0,^stream_0_input_drop/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€}@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_stream_2_drop_1_layer_call_and_return_conditional_losses_1047832)
'stream_2_drop_1/StatefulPartitionedCall—
'stream_1_drop_1/StatefulPartitionedCallStatefulPartitionedCall%activation_1/PartitionedCall:output:0(^stream_2_drop_1/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€}@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_stream_1_drop_1_layer_call_and_return_conditional_losses_1047602)
'stream_1_drop_1/StatefulPartitionedCallѕ
'stream_0_drop_1/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0(^stream_1_drop_1/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€}@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_stream_0_drop_1_layer_call_and_return_conditional_losses_1047372)
'stream_0_drop_1/StatefulPartitionedCall±
(global_average_pooling1d/PartitionedCallPartitionedCall0stream_0_drop_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *]
fXRV
T__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_1045162*
(global_average_pooling1d/PartitionedCallЈ
*global_average_pooling1d_1/PartitionedCallPartitionedCall0stream_1_drop_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *_
fZRX
V__inference_global_average_pooling1d_1_layer_call_and_return_conditional_losses_1045232,
*global_average_pooling1d_1/PartitionedCallЈ
*global_average_pooling1d_2/PartitionedCallPartitionedCall0stream_2_drop_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *_
fZRX
V__inference_global_average_pooling1d_2_layer_call_and_return_conditional_losses_1045302,
*global_average_pooling1d_2/PartitionedCallш
concatenate/PartitionedCallPartitionedCall1global_average_pooling1d/PartitionedCall:output:03global_average_pooling1d_1/PartitionedCall:output:03global_average_pooling1d_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€ј* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_concatenate_layer_call_and_return_conditional_losses_1045402
concatenate/PartitionedCallЛ
dense_1_dropout/PartitionedCallPartitionedCall$concatenate/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€ј* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_dense_1_dropout_layer_call_and_return_conditional_losses_1046912!
dense_1_dropout/PartitionedCallі
dense_1/StatefulPartitionedCallStatefulPartitionedCall(dense_1_dropout/PartitionedCall:output:0dense_1_105203dense_1_105205*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€T*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_1045652!
dense_1/StatefulPartitionedCallЄ
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0batch_normalization_3_105208batch_normalization_3_105210batch_normalization_3_105212batch_normalization_3_105214*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€T*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1041912/
-batch_normalization_3/StatefulPartitionedCall•
"dense_activation_1/PartitionedCallPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€T* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *W
fRRP
N__inference_dense_activation_1_layer_call_and_return_conditional_losses_1045852$
"dense_activation_1/PartitionedCall…
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpstream_0_conv_1_105160*"
_output_shapes
:@*
dtype027
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp√
&stream_0_conv_1/kernel/Regularizer/AbsAbs=stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_0_conv_1/kernel/Regularizer/Abs©
(stream_0_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_0_conv_1/kernel/Regularizer/Const„
&stream_0_conv_1/kernel/Regularizer/SumSum*stream_0_conv_1/kernel/Regularizer/Abs:y:01stream_0_conv_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/SumЩ
(stream_0_conv_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2*
(stream_0_conv_1/kernel/Regularizer/mul/x№
&stream_0_conv_1/kernel/Regularizer/mulMul1stream_0_conv_1/kernel/Regularizer/mul/x:output:0/stream_0_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/mulѕ
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpstream_1_conv_1_105155*"
_output_shapes
:@*
dtype02:
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpѕ
)stream_1_conv_1/kernel/Regularizer/SquareSquare@stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2+
)stream_1_conv_1/kernel/Regularizer/Square©
(stream_1_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_1_conv_1/kernel/Regularizer/ConstЏ
&stream_1_conv_1/kernel/Regularizer/SumSum-stream_1_conv_1/kernel/Regularizer/Square:y:01stream_1_conv_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&stream_1_conv_1/kernel/Regularizer/SumЩ
(stream_1_conv_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2*
(stream_1_conv_1/kernel/Regularizer/mul/x№
&stream_1_conv_1/kernel/Regularizer/mulMul1stream_1_conv_1/kernel/Regularizer/mul/x:output:0/stream_1_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_1_conv_1/kernel/Regularizer/mul…
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpstream_2_conv_1_105150*"
_output_shapes
:@*
dtype027
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp√
&stream_2_conv_1/kernel/Regularizer/AbsAbs=stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_2_conv_1/kernel/Regularizer/Abs©
(stream_2_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_2_conv_1/kernel/Regularizer/Const„
&stream_2_conv_1/kernel/Regularizer/SumSum*stream_2_conv_1/kernel/Regularizer/Abs:y:01stream_2_conv_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&stream_2_conv_1/kernel/Regularizer/SumЩ
(stream_2_conv_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2*
(stream_2_conv_1/kernel/Regularizer/mul/x№
&stream_2_conv_1/kernel/Regularizer/mulMul1stream_2_conv_1/kernel/Regularizer/mul/x:output:0/stream_2_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_2_conv_1/kernel/Regularizer/mulЃ
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_1_105203*
_output_shapes
:	јT*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOp®
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	јT2 
dense_1/kernel/Regularizer/AbsХ
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_1/kernel/Regularizer/ConstЈ
dense_1/kernel/Regularizer/SumSum"dense_1/kernel/Regularizer/Abs:y:0)dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/SumЙ
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2"
 dense_1/kernel/Regularizer/mul/xЉ
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/mulЖ
IdentityIdentity+dense_activation_1/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€T2

IdentityП
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall.^dense_1/kernel/Regularizer/Abs/ReadVariableOp(^stream_0_conv_1/StatefulPartitionedCall6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp(^stream_0_drop_1/StatefulPartitionedCall,^stream_0_input_drop/StatefulPartitionedCall(^stream_1_conv_1/StatefulPartitionedCall9^stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp(^stream_1_drop_1/StatefulPartitionedCall,^stream_1_input_drop/StatefulPartitionedCall(^stream_2_conv_1/StatefulPartitionedCall6^stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp(^stream_2_drop_1/StatefulPartitionedCall,^stream_2_input_drop/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*И
_input_shapesw
u:€€€€€€€€€}:€€€€€€€€€}:€€€€€€€€€}: : : : : : : : : : : : : : : : : : : : : : : : 2Z
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
+stream_2_input_drop/StatefulPartitionedCall+stream_2_input_drop/StatefulPartitionedCall:S O
+
_output_shapes
:€€€€€€€€€}
 
_user_specified_nameinputs:SO
+
_output_shapes
:€€€€€€€€€}
 
_user_specified_nameinputs:SO
+
_output_shapes
:€€€€€€€€€}
 
_user_specified_nameinputs
€*
и
O__inference_batch_normalization_layer_call_and_return_conditional_losses_107991

inputs5
'assignmovingavg_readvariableop_resource:@7
)assignmovingavg_1_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@/
!batchnorm_readvariableop_resource:@
identityИҐAssignMovingAvgҐAssignMovingAvg/ReadVariableOpҐAssignMovingAvg_1Ґ AssignMovingAvg_1/ReadVariableOpҐbatchnorm/ReadVariableOpҐbatchnorm/mul/ReadVariableOpС
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indicesУ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(2
moments/meanА
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:@2
moments/StopGradient®
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*+
_output_shapes
:€€€€€€€€€}@2
moments/SquaredDifferenceЩ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indicesґ
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(2
moments/varianceБ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 2
moments/SqueezeЙ
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
„#<2
AssignMovingAvg/decay§
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOpШ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg/subП
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg/mulњ
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
„#<2
AssignMovingAvg_1/decay™
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOp†
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg_1/subЧ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg_1/mul…
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
 *oГ:2
batchnorm/add/yВ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2
batchnorm/mulz
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€}@2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:@2
batchnorm/mul_2Т
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOpБ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2
batchnorm/subЙ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*+
_output_shapes
:€€€€€€€€€}@2
batchnorm/add_1r
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€}@2

Identityт
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€}@: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:S O
+
_output_shapes
:€€€€€€€€€}@
 
_user_specified_nameinputs
н
j
K__inference_stream_2_drop_1_layer_call_and_return_conditional_losses_108464

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †?2
dropout/Constw
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:€€€€€€€€€}@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape”
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:€€€€€€€€€}@*
dtype0*
seedЈ*
seed2Ј2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL>2
dropout/GreaterEqual/y¬
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:€€€€€€€€€}@2
dropout/GreaterEqualГ
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:€€€€€€€€€}@2
dropout/Cast~
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:€€€€€€€€€}@2
dropout/Mul_1i
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:€€€€€€€€€}@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€}@:S O
+
_output_shapes
:€€€€€€€€€}@
 
_user_specified_nameinputs
М
m
O__inference_stream_0_input_drop_layer_call_and_return_conditional_losses_104299

inputs

identity_1^
IdentityIdentityinputs*
T0*+
_output_shapes
:€€€€€€€€€}2

Identitym

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:€€€€€€€€€}2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€}:S O
+
_output_shapes
:€€€€€€€€€}
 
_user_specified_nameinputs
„
I
-__inference_activation_2_layer_call_fn_108393

inputs
identityЌ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€}@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_activation_2_layer_call_and_return_conditional_losses_1044742
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:€€€€€€€€€}@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€}@:S O
+
_output_shapes
:€€€€€€€€€}@
 
_user_specified_nameinputs
еС
П*
"__inference__traced_restore_109151
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
"assignvariableop_17_dense_1_kernel:	јT.
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
)assignvariableop_43_adam_dense_1_kernel_m:	јT5
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
)assignvariableop_59_adam_dense_1_kernel_v:	јT5
'assignvariableop_60_adam_dense_1_bias_v:TD
6assignvariableop_61_adam_batch_normalization_3_gamma_v:TC
5assignvariableop_62_adam_batch_normalization_3_beta_v:T
identity_64ИҐAssignVariableOpҐAssignVariableOp_1ҐAssignVariableOp_10ҐAssignVariableOp_11ҐAssignVariableOp_12ҐAssignVariableOp_13ҐAssignVariableOp_14ҐAssignVariableOp_15ҐAssignVariableOp_16ҐAssignVariableOp_17ҐAssignVariableOp_18ҐAssignVariableOp_19ҐAssignVariableOp_2ҐAssignVariableOp_20ҐAssignVariableOp_21ҐAssignVariableOp_22ҐAssignVariableOp_23ҐAssignVariableOp_24ҐAssignVariableOp_25ҐAssignVariableOp_26ҐAssignVariableOp_27ҐAssignVariableOp_28ҐAssignVariableOp_29ҐAssignVariableOp_3ҐAssignVariableOp_30ҐAssignVariableOp_31ҐAssignVariableOp_32ҐAssignVariableOp_33ҐAssignVariableOp_34ҐAssignVariableOp_35ҐAssignVariableOp_36ҐAssignVariableOp_37ҐAssignVariableOp_38ҐAssignVariableOp_39ҐAssignVariableOp_4ҐAssignVariableOp_40ҐAssignVariableOp_41ҐAssignVariableOp_42ҐAssignVariableOp_43ҐAssignVariableOp_44ҐAssignVariableOp_45ҐAssignVariableOp_46ҐAssignVariableOp_47ҐAssignVariableOp_48ҐAssignVariableOp_49ҐAssignVariableOp_5ҐAssignVariableOp_50ҐAssignVariableOp_51ҐAssignVariableOp_52ҐAssignVariableOp_53ҐAssignVariableOp_54ҐAssignVariableOp_55ҐAssignVariableOp_56ҐAssignVariableOp_57ҐAssignVariableOp_58ҐAssignVariableOp_59ҐAssignVariableOp_6ҐAssignVariableOp_60ҐAssignVariableOp_61ҐAssignVariableOp_62ҐAssignVariableOp_7ҐAssignVariableOp_8ҐAssignVariableOp_9∞ 
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:@*
dtype0*Љ
value≤Bѓ@B+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/14/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/15/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesС
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:@*
dtype0*Х
valueЛBИ@B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesо
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*Ц
_output_shapesГ
А::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*N
dtypesD
B2@	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

IdentityЦ
AssignVariableOpAssignVariableOpassignvariableop_beta_1Identity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1Ю
AssignVariableOp_1AssignVariableOpassignvariableop_1_beta_2Identity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2Э
AssignVariableOp_2AssignVariableOpassignvariableop_2_decayIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3•
AssignVariableOp_3AssignVariableOp assignvariableop_3_learning_rateIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_4°
AssignVariableOp_4AssignVariableOpassignvariableop_4_adam_iterIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5Ѓ
AssignVariableOp_5AssignVariableOp)assignvariableop_5_stream_0_conv_1_kernelIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6ђ
AssignVariableOp_6AssignVariableOp'assignvariableop_6_stream_0_conv_1_biasIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7Ѓ
AssignVariableOp_7AssignVariableOp)assignvariableop_7_stream_1_conv_1_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8ђ
AssignVariableOp_8AssignVariableOp'assignvariableop_8_stream_1_conv_1_biasIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9Ѓ
AssignVariableOp_9AssignVariableOp)assignvariableop_9_stream_2_conv_1_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10∞
AssignVariableOp_10AssignVariableOp(assignvariableop_10_stream_2_conv_1_biasIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11µ
AssignVariableOp_11AssignVariableOp-assignvariableop_11_batch_normalization_gammaIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12і
AssignVariableOp_12AssignVariableOp,assignvariableop_12_batch_normalization_betaIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13Ј
AssignVariableOp_13AssignVariableOp/assignvariableop_13_batch_normalization_1_gammaIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14ґ
AssignVariableOp_14AssignVariableOp.assignvariableop_14_batch_normalization_1_betaIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15Ј
AssignVariableOp_15AssignVariableOp/assignvariableop_15_batch_normalization_2_gammaIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16ґ
AssignVariableOp_16AssignVariableOp.assignvariableop_16_batch_normalization_2_betaIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17™
AssignVariableOp_17AssignVariableOp"assignvariableop_17_dense_1_kernelIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18®
AssignVariableOp_18AssignVariableOp assignvariableop_18_dense_1_biasIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19Ј
AssignVariableOp_19AssignVariableOp/assignvariableop_19_batch_normalization_3_gammaIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20ґ
AssignVariableOp_20AssignVariableOp.assignvariableop_20_batch_normalization_3_betaIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21ї
AssignVariableOp_21AssignVariableOp3assignvariableop_21_batch_normalization_moving_meanIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22њ
AssignVariableOp_22AssignVariableOp7assignvariableop_22_batch_normalization_moving_varianceIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23љ
AssignVariableOp_23AssignVariableOp5assignvariableop_23_batch_normalization_1_moving_meanIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24Ѕ
AssignVariableOp_24AssignVariableOp9assignvariableop_24_batch_normalization_1_moving_varianceIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25љ
AssignVariableOp_25AssignVariableOp5assignvariableop_25_batch_normalization_2_moving_meanIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26Ѕ
AssignVariableOp_26AssignVariableOp9assignvariableop_26_batch_normalization_2_moving_varianceIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27љ
AssignVariableOp_27AssignVariableOp5assignvariableop_27_batch_normalization_3_moving_meanIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28Ѕ
AssignVariableOp_28AssignVariableOp9assignvariableop_28_batch_normalization_3_moving_varianceIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29°
AssignVariableOp_29AssignVariableOpassignvariableop_29_totalIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30°
AssignVariableOp_30AssignVariableOpassignvariableop_30_countIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31є
AssignVariableOp_31AssignVariableOp1assignvariableop_31_adam_stream_0_conv_1_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32Ј
AssignVariableOp_32AssignVariableOp/assignvariableop_32_adam_stream_0_conv_1_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33є
AssignVariableOp_33AssignVariableOp1assignvariableop_33_adam_stream_1_conv_1_kernel_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34Ј
AssignVariableOp_34AssignVariableOp/assignvariableop_34_adam_stream_1_conv_1_bias_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35є
AssignVariableOp_35AssignVariableOp1assignvariableop_35_adam_stream_2_conv_1_kernel_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36Ј
AssignVariableOp_36AssignVariableOp/assignvariableop_36_adam_stream_2_conv_1_bias_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37Љ
AssignVariableOp_37AssignVariableOp4assignvariableop_37_adam_batch_normalization_gamma_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38ї
AssignVariableOp_38AssignVariableOp3assignvariableop_38_adam_batch_normalization_beta_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39Њ
AssignVariableOp_39AssignVariableOp6assignvariableop_39_adam_batch_normalization_1_gamma_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40љ
AssignVariableOp_40AssignVariableOp5assignvariableop_40_adam_batch_normalization_1_beta_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41Њ
AssignVariableOp_41AssignVariableOp6assignvariableop_41_adam_batch_normalization_2_gamma_mIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42љ
AssignVariableOp_42AssignVariableOp5assignvariableop_42_adam_batch_normalization_2_beta_mIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43±
AssignVariableOp_43AssignVariableOp)assignvariableop_43_adam_dense_1_kernel_mIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44ѓ
AssignVariableOp_44AssignVariableOp'assignvariableop_44_adam_dense_1_bias_mIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45Њ
AssignVariableOp_45AssignVariableOp6assignvariableop_45_adam_batch_normalization_3_gamma_mIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46љ
AssignVariableOp_46AssignVariableOp5assignvariableop_46_adam_batch_normalization_3_beta_mIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47є
AssignVariableOp_47AssignVariableOp1assignvariableop_47_adam_stream_0_conv_1_kernel_vIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48Ј
AssignVariableOp_48AssignVariableOp/assignvariableop_48_adam_stream_0_conv_1_bias_vIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49є
AssignVariableOp_49AssignVariableOp1assignvariableop_49_adam_stream_1_conv_1_kernel_vIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50Ј
AssignVariableOp_50AssignVariableOp/assignvariableop_50_adam_stream_1_conv_1_bias_vIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51є
AssignVariableOp_51AssignVariableOp1assignvariableop_51_adam_stream_2_conv_1_kernel_vIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52Ј
AssignVariableOp_52AssignVariableOp/assignvariableop_52_adam_stream_2_conv_1_bias_vIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53Љ
AssignVariableOp_53AssignVariableOp4assignvariableop_53_adam_batch_normalization_gamma_vIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54ї
AssignVariableOp_54AssignVariableOp3assignvariableop_54_adam_batch_normalization_beta_vIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55Њ
AssignVariableOp_55AssignVariableOp6assignvariableop_55_adam_batch_normalization_1_gamma_vIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56љ
AssignVariableOp_56AssignVariableOp5assignvariableop_56_adam_batch_normalization_1_beta_vIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57Њ
AssignVariableOp_57AssignVariableOp6assignvariableop_57_adam_batch_normalization_2_gamma_vIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58љ
AssignVariableOp_58AssignVariableOp5assignvariableop_58_adam_batch_normalization_2_beta_vIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_58n
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:2
Identity_59±
AssignVariableOp_59AssignVariableOp)assignvariableop_59_adam_dense_1_kernel_vIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59n
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:2
Identity_60ѓ
AssignVariableOp_60AssignVariableOp'assignvariableop_60_adam_dense_1_bias_vIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_60n
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:2
Identity_61Њ
AssignVariableOp_61AssignVariableOp6assignvariableop_61_adam_batch_normalization_3_gamma_vIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_61n
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:2
Identity_62љ
AssignVariableOp_62AssignVariableOp5assignvariableop_62_adam_batch_normalization_3_beta_vIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_629
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp»
Identity_63Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_63f
Identity_64IdentityIdentity_63:output:0^NoOp_1*
T0*
_output_shapes
: 2
Identity_64∞
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 2
NoOp_1"#
identity_64Identity_64:output:0*Х
_input_shapesГ
А: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
№

G__inference_concatenate_layer_call_and_return_conditional_losses_104540

inputs
inputs_1
inputs_2
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axisК
concatConcatV2inputsinputs_1inputs_2concat/axis:output:0*
N*
T0*(
_output_shapes
:€€€€€€€€€ј2
concatd
IdentityIdentityconcat:output:0*
T0*(
_output_shapes
:€€€€€€€€€ј2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:€€€€€€€€€@:€€€€€€€€€@:€€€€€€€€€@:O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
°
W
;__inference_global_average_pooling1d_1_layer_call_fn_108513

inputs
identityа
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *_
fZRX
V__inference_global_average_pooling1d_1_layer_call_and_return_conditional_losses_1040692
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€:e a
=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
р
d
H__inference_activation_2_layer_call_and_return_conditional_losses_104474

inputs
identityR
ReluReluinputs*
T0*+
_output_shapes
:€€€€€€€€€}@2
Reluj
IdentityIdentityRelu:activations:0*
T0*+
_output_shapes
:€€€€€€€€€}@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€}@:S O
+
_output_shapes
:€€€€€€€€€}@
 
_user_specified_nameinputs
ћ*
к
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_104191

inputs5
'assignmovingavg_readvariableop_resource:T7
)assignmovingavg_1_readvariableop_resource:T3
%batchnorm_mul_readvariableop_resource:T/
!batchnorm_readvariableop_resource:T
identityИҐAssignMovingAvgҐAssignMovingAvg/ReadVariableOpҐAssignMovingAvg_1Ґ AssignMovingAvg_1/ReadVariableOpҐbatchnorm/ReadVariableOpҐbatchnorm/mul/ReadVariableOpК
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indicesП
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
moments/StopGradient§
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:€€€€€€€€€T2
moments/SquaredDifferenceТ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices≤
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:T*
	keep_dims(2
moments/varianceА
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:T*
squeeze_dims
 2
moments/SqueezeИ
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
„#<2
AssignMovingAvg/decay§
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:T*
dtype02 
AssignMovingAvg/ReadVariableOpШ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:T2
AssignMovingAvg/subП
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:T2
AssignMovingAvg/mulњ
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
„#<2
AssignMovingAvg_1/decay™
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:T*
dtype02"
 AssignMovingAvg_1/ReadVariableOp†
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:T2
AssignMovingAvg_1/subЧ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:T2
AssignMovingAvg_1/mul…
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
 *oГ:2
batchnorm/add/yВ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:T2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:T2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:T*
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:T2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:€€€€€€€€€T2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:T2
batchnorm/mul_2Т
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:T*
dtype02
batchnorm/ReadVariableOpБ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:T2
batchnorm/subЕ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:€€€€€€€€€T2
batchnorm/add_1n
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€T2

Identityт
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€T: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€T
 
_user_specified_nameinputs
х
Ц
(__inference_dense_1_layer_call_fn_108605

inputs
unknown:	јT
	unknown_0:T
identityИҐStatefulPartitionedCallц
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€T*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_1045652
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€T2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€ј: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:€€€€€€€€€ј
 
_user_specified_nameinputs
Д
p
T__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_104516

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
:€€€€€€€€€@2
Meana
IdentityIdentityMean:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€}@:S O
+
_output_shapes
:€€€€€€€€€}@
 
_user_specified_nameinputs
Љ
r
V__inference_global_average_pooling1d_1_layer_call_and_return_conditional_losses_108502

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
:€€€€€€€€€€€€€€€€€€2
Meanj
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€:e a
=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
—
L
0__inference_dense_1_dropout_layer_call_fn_108569

inputs
identityЌ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€ј* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_dense_1_dropout_layer_call_and_return_conditional_losses_1045472
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:€€€€€€€€€ј2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:€€€€€€€€€ј:P L
(
_output_shapes
:€€€€€€€€€ј
 
_user_specified_nameinputs
•Ї
§
E__inference_basemodel_layer_call_and_return_conditional_losses_107548
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
&dense_1_matmul_readvariableop_resource:	јT5
'dense_1_biasadd_readvariableop_resource:TK
=batch_normalization_3_assignmovingavg_readvariableop_resource:TM
?batch_normalization_3_assignmovingavg_1_readvariableop_resource:TI
;batch_normalization_3_batchnorm_mul_readvariableop_resource:TE
7batch_normalization_3_batchnorm_readvariableop_resource:T
identityИҐ#batch_normalization/AssignMovingAvgҐ2batch_normalization/AssignMovingAvg/ReadVariableOpҐ%batch_normalization/AssignMovingAvg_1Ґ4batch_normalization/AssignMovingAvg_1/ReadVariableOpҐ,batch_normalization/batchnorm/ReadVariableOpҐ0batch_normalization/batchnorm/mul/ReadVariableOpҐ%batch_normalization_1/AssignMovingAvgҐ4batch_normalization_1/AssignMovingAvg/ReadVariableOpҐ'batch_normalization_1/AssignMovingAvg_1Ґ6batch_normalization_1/AssignMovingAvg_1/ReadVariableOpҐ.batch_normalization_1/batchnorm/ReadVariableOpҐ2batch_normalization_1/batchnorm/mul/ReadVariableOpҐ%batch_normalization_2/AssignMovingAvgҐ4batch_normalization_2/AssignMovingAvg/ReadVariableOpҐ'batch_normalization_2/AssignMovingAvg_1Ґ6batch_normalization_2/AssignMovingAvg_1/ReadVariableOpҐ.batch_normalization_2/batchnorm/ReadVariableOpҐ2batch_normalization_2/batchnorm/mul/ReadVariableOpҐ%batch_normalization_3/AssignMovingAvgҐ4batch_normalization_3/AssignMovingAvg/ReadVariableOpҐ'batch_normalization_3/AssignMovingAvg_1Ґ6batch_normalization_3/AssignMovingAvg_1/ReadVariableOpҐ.batch_normalization_3/batchnorm/ReadVariableOpҐ2batch_normalization_3/batchnorm/mul/ReadVariableOpҐdense_1/BiasAdd/ReadVariableOpҐdense_1/MatMul/ReadVariableOpҐ-dense_1/kernel/Regularizer/Abs/ReadVariableOpҐ&stream_0_conv_1/BiasAdd/ReadVariableOpҐ2stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOpҐ5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpҐ&stream_1_conv_1/BiasAdd/ReadVariableOpҐ2stream_1_conv_1/conv1d/ExpandDims_1/ReadVariableOpҐ8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpҐ&stream_2_conv_1/BiasAdd/ReadVariableOpҐ2stream_2_conv_1/conv1d/ExpandDims_1/ReadVariableOpҐ5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOpЛ
!stream_2_input_drop/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †?2#
!stream_2_input_drop/dropout/Constµ
stream_2_input_drop/dropout/MulMulinputs_2*stream_2_input_drop/dropout/Const:output:0*
T0*+
_output_shapes
:€€€€€€€€€}2!
stream_2_input_drop/dropout/Mul~
!stream_2_input_drop/dropout/ShapeShapeinputs_2*
T0*
_output_shapes
:2#
!stream_2_input_drop/dropout/ShapeП
8stream_2_input_drop/dropout/random_uniform/RandomUniformRandomUniform*stream_2_input_drop/dropout/Shape:output:0*
T0*+
_output_shapes
:€€€€€€€€€}*
dtype0*
seedЈ*
seed2Ј2:
8stream_2_input_drop/dropout/random_uniform/RandomUniformЭ
*stream_2_input_drop/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL>2,
*stream_2_input_drop/dropout/GreaterEqual/yТ
(stream_2_input_drop/dropout/GreaterEqualGreaterEqualAstream_2_input_drop/dropout/random_uniform/RandomUniform:output:03stream_2_input_drop/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:€€€€€€€€€}2*
(stream_2_input_drop/dropout/GreaterEqualњ
 stream_2_input_drop/dropout/CastCast,stream_2_input_drop/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:€€€€€€€€€}2"
 stream_2_input_drop/dropout/Castќ
!stream_2_input_drop/dropout/Mul_1Mul#stream_2_input_drop/dropout/Mul:z:0$stream_2_input_drop/dropout/Cast:y:0*
T0*+
_output_shapes
:€€€€€€€€€}2#
!stream_2_input_drop/dropout/Mul_1Л
!stream_1_input_drop/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †?2#
!stream_1_input_drop/dropout/Constµ
stream_1_input_drop/dropout/MulMulinputs_1*stream_1_input_drop/dropout/Const:output:0*
T0*+
_output_shapes
:€€€€€€€€€}2!
stream_1_input_drop/dropout/Mul~
!stream_1_input_drop/dropout/ShapeShapeinputs_1*
T0*
_output_shapes
:2#
!stream_1_input_drop/dropout/ShapeП
8stream_1_input_drop/dropout/random_uniform/RandomUniformRandomUniform*stream_1_input_drop/dropout/Shape:output:0*
T0*+
_output_shapes
:€€€€€€€€€}*
dtype0*
seedЈ*
seed2Ј2:
8stream_1_input_drop/dropout/random_uniform/RandomUniformЭ
*stream_1_input_drop/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL>2,
*stream_1_input_drop/dropout/GreaterEqual/yТ
(stream_1_input_drop/dropout/GreaterEqualGreaterEqualAstream_1_input_drop/dropout/random_uniform/RandomUniform:output:03stream_1_input_drop/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:€€€€€€€€€}2*
(stream_1_input_drop/dropout/GreaterEqualњ
 stream_1_input_drop/dropout/CastCast,stream_1_input_drop/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:€€€€€€€€€}2"
 stream_1_input_drop/dropout/Castќ
!stream_1_input_drop/dropout/Mul_1Mul#stream_1_input_drop/dropout/Mul:z:0$stream_1_input_drop/dropout/Cast:y:0*
T0*+
_output_shapes
:€€€€€€€€€}2#
!stream_1_input_drop/dropout/Mul_1Л
!stream_0_input_drop/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †?2#
!stream_0_input_drop/dropout/Constµ
stream_0_input_drop/dropout/MulMulinputs_0*stream_0_input_drop/dropout/Const:output:0*
T0*+
_output_shapes
:€€€€€€€€€}2!
stream_0_input_drop/dropout/Mul~
!stream_0_input_drop/dropout/ShapeShapeinputs_0*
T0*
_output_shapes
:2#
!stream_0_input_drop/dropout/ShapeП
8stream_0_input_drop/dropout/random_uniform/RandomUniformRandomUniform*stream_0_input_drop/dropout/Shape:output:0*
T0*+
_output_shapes
:€€€€€€€€€}*
dtype0*
seedЈ*
seed2Ј2:
8stream_0_input_drop/dropout/random_uniform/RandomUniformЭ
*stream_0_input_drop/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL>2,
*stream_0_input_drop/dropout/GreaterEqual/yТ
(stream_0_input_drop/dropout/GreaterEqualGreaterEqualAstream_0_input_drop/dropout/random_uniform/RandomUniform:output:03stream_0_input_drop/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:€€€€€€€€€}2*
(stream_0_input_drop/dropout/GreaterEqualњ
 stream_0_input_drop/dropout/CastCast,stream_0_input_drop/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:€€€€€€€€€}2"
 stream_0_input_drop/dropout/Castќ
!stream_0_input_drop/dropout/Mul_1Mul#stream_0_input_drop/dropout/Mul:z:0$stream_0_input_drop/dropout/Cast:y:0*
T0*+
_output_shapes
:€€€€€€€€€}2#
!stream_0_input_drop/dropout/Mul_1Щ
%stream_2_conv_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€2'
%stream_2_conv_1/conv1d/ExpandDims/dimе
!stream_2_conv_1/conv1d/ExpandDims
ExpandDims%stream_2_input_drop/dropout/Mul_1:z:0.stream_2_conv_1/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:€€€€€€€€€}2#
!stream_2_conv_1/conv1d/ExpandDimsи
2stream_2_conv_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp;stream_2_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype024
2stream_2_conv_1/conv1d/ExpandDims_1/ReadVariableOpФ
'stream_2_conv_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2)
'stream_2_conv_1/conv1d/ExpandDims_1/dimч
#stream_2_conv_1/conv1d/ExpandDims_1
ExpandDims:stream_2_conv_1/conv1d/ExpandDims_1/ReadVariableOp:value:00stream_2_conv_1/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@2%
#stream_2_conv_1/conv1d/ExpandDims_1ц
stream_2_conv_1/conv1dConv2D*stream_2_conv_1/conv1d/ExpandDims:output:0,stream_2_conv_1/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:€€€€€€€€€}@*
paddingSAME*
strides
2
stream_2_conv_1/conv1d¬
stream_2_conv_1/conv1d/SqueezeSqueezestream_2_conv_1/conv1d:output:0*
T0*+
_output_shapes
:€€€€€€€€€}@*
squeeze_dims

э€€€€€€€€2 
stream_2_conv_1/conv1d/SqueezeЉ
&stream_2_conv_1/BiasAdd/ReadVariableOpReadVariableOp/stream_2_conv_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02(
&stream_2_conv_1/BiasAdd/ReadVariableOpћ
stream_2_conv_1/BiasAddBiasAdd'stream_2_conv_1/conv1d/Squeeze:output:0.stream_2_conv_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€}@2
stream_2_conv_1/BiasAddЩ
%stream_1_conv_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€2'
%stream_1_conv_1/conv1d/ExpandDims/dimе
!stream_1_conv_1/conv1d/ExpandDims
ExpandDims%stream_1_input_drop/dropout/Mul_1:z:0.stream_1_conv_1/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:€€€€€€€€€}2#
!stream_1_conv_1/conv1d/ExpandDimsи
2stream_1_conv_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp;stream_1_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype024
2stream_1_conv_1/conv1d/ExpandDims_1/ReadVariableOpФ
'stream_1_conv_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2)
'stream_1_conv_1/conv1d/ExpandDims_1/dimч
#stream_1_conv_1/conv1d/ExpandDims_1
ExpandDims:stream_1_conv_1/conv1d/ExpandDims_1/ReadVariableOp:value:00stream_1_conv_1/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@2%
#stream_1_conv_1/conv1d/ExpandDims_1ц
stream_1_conv_1/conv1dConv2D*stream_1_conv_1/conv1d/ExpandDims:output:0,stream_1_conv_1/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:€€€€€€€€€}@*
paddingSAME*
strides
2
stream_1_conv_1/conv1d¬
stream_1_conv_1/conv1d/SqueezeSqueezestream_1_conv_1/conv1d:output:0*
T0*+
_output_shapes
:€€€€€€€€€}@*
squeeze_dims

э€€€€€€€€2 
stream_1_conv_1/conv1d/SqueezeЉ
&stream_1_conv_1/BiasAdd/ReadVariableOpReadVariableOp/stream_1_conv_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02(
&stream_1_conv_1/BiasAdd/ReadVariableOpћ
stream_1_conv_1/BiasAddBiasAdd'stream_1_conv_1/conv1d/Squeeze:output:0.stream_1_conv_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€}@2
stream_1_conv_1/BiasAddЩ
%stream_0_conv_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€2'
%stream_0_conv_1/conv1d/ExpandDims/dimе
!stream_0_conv_1/conv1d/ExpandDims
ExpandDims%stream_0_input_drop/dropout/Mul_1:z:0.stream_0_conv_1/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:€€€€€€€€€}2#
!stream_0_conv_1/conv1d/ExpandDimsи
2stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp;stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype024
2stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOpФ
'stream_0_conv_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2)
'stream_0_conv_1/conv1d/ExpandDims_1/dimч
#stream_0_conv_1/conv1d/ExpandDims_1
ExpandDims:stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp:value:00stream_0_conv_1/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@2%
#stream_0_conv_1/conv1d/ExpandDims_1ц
stream_0_conv_1/conv1dConv2D*stream_0_conv_1/conv1d/ExpandDims:output:0,stream_0_conv_1/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:€€€€€€€€€}@*
paddingSAME*
strides
2
stream_0_conv_1/conv1d¬
stream_0_conv_1/conv1d/SqueezeSqueezestream_0_conv_1/conv1d:output:0*
T0*+
_output_shapes
:€€€€€€€€€}@*
squeeze_dims

э€€€€€€€€2 
stream_0_conv_1/conv1d/SqueezeЉ
&stream_0_conv_1/BiasAdd/ReadVariableOpReadVariableOp/stream_0_conv_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02(
&stream_0_conv_1/BiasAdd/ReadVariableOpћ
stream_0_conv_1/BiasAddBiasAdd'stream_0_conv_1/conv1d/Squeeze:output:0.stream_0_conv_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€}@2
stream_0_conv_1/BiasAddљ
4batch_normalization_2/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       26
4batch_normalization_2/moments/mean/reduction_indicesп
"batch_normalization_2/moments/meanMean stream_2_conv_1/BiasAdd:output:0=batch_normalization_2/moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(2$
"batch_normalization_2/moments/mean¬
*batch_normalization_2/moments/StopGradientStopGradient+batch_normalization_2/moments/mean:output:0*
T0*"
_output_shapes
:@2,
*batch_normalization_2/moments/StopGradientД
/batch_normalization_2/moments/SquaredDifferenceSquaredDifference stream_2_conv_1/BiasAdd:output:03batch_normalization_2/moments/StopGradient:output:0*
T0*+
_output_shapes
:€€€€€€€€€}@21
/batch_normalization_2/moments/SquaredDifference≈
8batch_normalization_2/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2:
8batch_normalization_2/moments/variance/reduction_indicesО
&batch_normalization_2/moments/varianceMean3batch_normalization_2/moments/SquaredDifference:z:0Abatch_normalization_2/moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(2(
&batch_normalization_2/moments/variance√
%batch_normalization_2/moments/SqueezeSqueeze+batch_normalization_2/moments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 2'
%batch_normalization_2/moments/SqueezeЋ
'batch_normalization_2/moments/Squeeze_1Squeeze/batch_normalization_2/moments/variance:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 2)
'batch_normalization_2/moments/Squeeze_1Я
+batch_normalization_2/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2-
+batch_normalization_2/AssignMovingAvg/decayж
4batch_normalization_2/AssignMovingAvg/ReadVariableOpReadVariableOp=batch_normalization_2_assignmovingavg_readvariableop_resource*
_output_shapes
:@*
dtype026
4batch_normalization_2/AssignMovingAvg/ReadVariableOpр
)batch_normalization_2/AssignMovingAvg/subSub<batch_normalization_2/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_2/moments/Squeeze:output:0*
T0*
_output_shapes
:@2+
)batch_normalization_2/AssignMovingAvg/subз
)batch_normalization_2/AssignMovingAvg/mulMul-batch_normalization_2/AssignMovingAvg/sub:z:04batch_normalization_2/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:@2+
)batch_normalization_2/AssignMovingAvg/mul≠
%batch_normalization_2/AssignMovingAvgAssignSubVariableOp=batch_normalization_2_assignmovingavg_readvariableop_resource-batch_normalization_2/AssignMovingAvg/mul:z:05^batch_normalization_2/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02'
%batch_normalization_2/AssignMovingAvg£
-batch_normalization_2/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2/
-batch_normalization_2/AssignMovingAvg_1/decayм
6batch_normalization_2/AssignMovingAvg_1/ReadVariableOpReadVariableOp?batch_normalization_2_assignmovingavg_1_readvariableop_resource*
_output_shapes
:@*
dtype028
6batch_normalization_2/AssignMovingAvg_1/ReadVariableOpш
+batch_normalization_2/AssignMovingAvg_1/subSub>batch_normalization_2/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_2/moments/Squeeze_1:output:0*
T0*
_output_shapes
:@2-
+batch_normalization_2/AssignMovingAvg_1/subп
+batch_normalization_2/AssignMovingAvg_1/mulMul/batch_normalization_2/AssignMovingAvg_1/sub:z:06batch_normalization_2/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:@2-
+batch_normalization_2/AssignMovingAvg_1/mulЈ
'batch_normalization_2/AssignMovingAvg_1AssignSubVariableOp?batch_normalization_2_assignmovingavg_1_readvariableop_resource/batch_normalization_2/AssignMovingAvg_1/mul:z:07^batch_normalization_2/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02)
'batch_normalization_2/AssignMovingAvg_1У
%batch_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2'
%batch_normalization_2/batchnorm/add/yЏ
#batch_normalization_2/batchnorm/addAddV20batch_normalization_2/moments/Squeeze_1:output:0.batch_normalization_2/batchnorm/add/y:output:0*
T0*
_output_shapes
:@2%
#batch_normalization_2/batchnorm/add•
%batch_normalization_2/batchnorm/RsqrtRsqrt'batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes
:@2'
%batch_normalization_2/batchnorm/Rsqrtа
2batch_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype024
2batch_normalization_2/batchnorm/mul/ReadVariableOpЁ
#batch_normalization_2/batchnorm/mulMul)batch_normalization_2/batchnorm/Rsqrt:y:0:batch_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2%
#batch_normalization_2/batchnorm/mul÷
%batch_normalization_2/batchnorm/mul_1Mul stream_2_conv_1/BiasAdd:output:0'batch_normalization_2/batchnorm/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€}@2'
%batch_normalization_2/batchnorm/mul_1”
%batch_normalization_2/batchnorm/mul_2Mul.batch_normalization_2/moments/Squeeze:output:0'batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes
:@2'
%batch_normalization_2/batchnorm/mul_2‘
.batch_normalization_2/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_2_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype020
.batch_normalization_2/batchnorm/ReadVariableOpў
#batch_normalization_2/batchnorm/subSub6batch_normalization_2/batchnorm/ReadVariableOp:value:0)batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2%
#batch_normalization_2/batchnorm/subб
%batch_normalization_2/batchnorm/add_1AddV2)batch_normalization_2/batchnorm/mul_1:z:0'batch_normalization_2/batchnorm/sub:z:0*
T0*+
_output_shapes
:€€€€€€€€€}@2'
%batch_normalization_2/batchnorm/add_1љ
4batch_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       26
4batch_normalization_1/moments/mean/reduction_indicesп
"batch_normalization_1/moments/meanMean stream_1_conv_1/BiasAdd:output:0=batch_normalization_1/moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(2$
"batch_normalization_1/moments/mean¬
*batch_normalization_1/moments/StopGradientStopGradient+batch_normalization_1/moments/mean:output:0*
T0*"
_output_shapes
:@2,
*batch_normalization_1/moments/StopGradientД
/batch_normalization_1/moments/SquaredDifferenceSquaredDifference stream_1_conv_1/BiasAdd:output:03batch_normalization_1/moments/StopGradient:output:0*
T0*+
_output_shapes
:€€€€€€€€€}@21
/batch_normalization_1/moments/SquaredDifference≈
8batch_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2:
8batch_normalization_1/moments/variance/reduction_indicesО
&batch_normalization_1/moments/varianceMean3batch_normalization_1/moments/SquaredDifference:z:0Abatch_normalization_1/moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(2(
&batch_normalization_1/moments/variance√
%batch_normalization_1/moments/SqueezeSqueeze+batch_normalization_1/moments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 2'
%batch_normalization_1/moments/SqueezeЋ
'batch_normalization_1/moments/Squeeze_1Squeeze/batch_normalization_1/moments/variance:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 2)
'batch_normalization_1/moments/Squeeze_1Я
+batch_normalization_1/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2-
+batch_normalization_1/AssignMovingAvg/decayж
4batch_normalization_1/AssignMovingAvg/ReadVariableOpReadVariableOp=batch_normalization_1_assignmovingavg_readvariableop_resource*
_output_shapes
:@*
dtype026
4batch_normalization_1/AssignMovingAvg/ReadVariableOpр
)batch_normalization_1/AssignMovingAvg/subSub<batch_normalization_1/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_1/moments/Squeeze:output:0*
T0*
_output_shapes
:@2+
)batch_normalization_1/AssignMovingAvg/subз
)batch_normalization_1/AssignMovingAvg/mulMul-batch_normalization_1/AssignMovingAvg/sub:z:04batch_normalization_1/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:@2+
)batch_normalization_1/AssignMovingAvg/mul≠
%batch_normalization_1/AssignMovingAvgAssignSubVariableOp=batch_normalization_1_assignmovingavg_readvariableop_resource-batch_normalization_1/AssignMovingAvg/mul:z:05^batch_normalization_1/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02'
%batch_normalization_1/AssignMovingAvg£
-batch_normalization_1/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2/
-batch_normalization_1/AssignMovingAvg_1/decayм
6batch_normalization_1/AssignMovingAvg_1/ReadVariableOpReadVariableOp?batch_normalization_1_assignmovingavg_1_readvariableop_resource*
_output_shapes
:@*
dtype028
6batch_normalization_1/AssignMovingAvg_1/ReadVariableOpш
+batch_normalization_1/AssignMovingAvg_1/subSub>batch_normalization_1/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_1/moments/Squeeze_1:output:0*
T0*
_output_shapes
:@2-
+batch_normalization_1/AssignMovingAvg_1/subп
+batch_normalization_1/AssignMovingAvg_1/mulMul/batch_normalization_1/AssignMovingAvg_1/sub:z:06batch_normalization_1/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:@2-
+batch_normalization_1/AssignMovingAvg_1/mulЈ
'batch_normalization_1/AssignMovingAvg_1AssignSubVariableOp?batch_normalization_1_assignmovingavg_1_readvariableop_resource/batch_normalization_1/AssignMovingAvg_1/mul:z:07^batch_normalization_1/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02)
'batch_normalization_1/AssignMovingAvg_1У
%batch_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2'
%batch_normalization_1/batchnorm/add/yЏ
#batch_normalization_1/batchnorm/addAddV20batch_normalization_1/moments/Squeeze_1:output:0.batch_normalization_1/batchnorm/add/y:output:0*
T0*
_output_shapes
:@2%
#batch_normalization_1/batchnorm/add•
%batch_normalization_1/batchnorm/RsqrtRsqrt'batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes
:@2'
%batch_normalization_1/batchnorm/Rsqrtа
2batch_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype024
2batch_normalization_1/batchnorm/mul/ReadVariableOpЁ
#batch_normalization_1/batchnorm/mulMul)batch_normalization_1/batchnorm/Rsqrt:y:0:batch_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2%
#batch_normalization_1/batchnorm/mul÷
%batch_normalization_1/batchnorm/mul_1Mul stream_1_conv_1/BiasAdd:output:0'batch_normalization_1/batchnorm/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€}@2'
%batch_normalization_1/batchnorm/mul_1”
%batch_normalization_1/batchnorm/mul_2Mul.batch_normalization_1/moments/Squeeze:output:0'batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes
:@2'
%batch_normalization_1/batchnorm/mul_2‘
.batch_normalization_1/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_1_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype020
.batch_normalization_1/batchnorm/ReadVariableOpў
#batch_normalization_1/batchnorm/subSub6batch_normalization_1/batchnorm/ReadVariableOp:value:0)batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2%
#batch_normalization_1/batchnorm/subб
%batch_normalization_1/batchnorm/add_1AddV2)batch_normalization_1/batchnorm/mul_1:z:0'batch_normalization_1/batchnorm/sub:z:0*
T0*+
_output_shapes
:€€€€€€€€€}@2'
%batch_normalization_1/batchnorm/add_1є
2batch_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       24
2batch_normalization/moments/mean/reduction_indicesй
 batch_normalization/moments/meanMean stream_0_conv_1/BiasAdd:output:0;batch_normalization/moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(2"
 batch_normalization/moments/meanЉ
(batch_normalization/moments/StopGradientStopGradient)batch_normalization/moments/mean:output:0*
T0*"
_output_shapes
:@2*
(batch_normalization/moments/StopGradientю
-batch_normalization/moments/SquaredDifferenceSquaredDifference stream_0_conv_1/BiasAdd:output:01batch_normalization/moments/StopGradient:output:0*
T0*+
_output_shapes
:€€€€€€€€€}@2/
-batch_normalization/moments/SquaredDifferenceЅ
6batch_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       28
6batch_normalization/moments/variance/reduction_indicesЖ
$batch_normalization/moments/varianceMean1batch_normalization/moments/SquaredDifference:z:0?batch_normalization/moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(2&
$batch_normalization/moments/varianceљ
#batch_normalization/moments/SqueezeSqueeze)batch_normalization/moments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 2%
#batch_normalization/moments/Squeeze≈
%batch_normalization/moments/Squeeze_1Squeeze-batch_normalization/moments/variance:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 2'
%batch_normalization/moments/Squeeze_1Ы
)batch_normalization/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2+
)batch_normalization/AssignMovingAvg/decayа
2batch_normalization/AssignMovingAvg/ReadVariableOpReadVariableOp;batch_normalization_assignmovingavg_readvariableop_resource*
_output_shapes
:@*
dtype024
2batch_normalization/AssignMovingAvg/ReadVariableOpи
'batch_normalization/AssignMovingAvg/subSub:batch_normalization/AssignMovingAvg/ReadVariableOp:value:0,batch_normalization/moments/Squeeze:output:0*
T0*
_output_shapes
:@2)
'batch_normalization/AssignMovingAvg/subя
'batch_normalization/AssignMovingAvg/mulMul+batch_normalization/AssignMovingAvg/sub:z:02batch_normalization/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:@2)
'batch_normalization/AssignMovingAvg/mul£
#batch_normalization/AssignMovingAvgAssignSubVariableOp;batch_normalization_assignmovingavg_readvariableop_resource+batch_normalization/AssignMovingAvg/mul:z:03^batch_normalization/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02%
#batch_normalization/AssignMovingAvgЯ
+batch_normalization/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2-
+batch_normalization/AssignMovingAvg_1/decayж
4batch_normalization/AssignMovingAvg_1/ReadVariableOpReadVariableOp=batch_normalization_assignmovingavg_1_readvariableop_resource*
_output_shapes
:@*
dtype026
4batch_normalization/AssignMovingAvg_1/ReadVariableOpр
)batch_normalization/AssignMovingAvg_1/subSub<batch_normalization/AssignMovingAvg_1/ReadVariableOp:value:0.batch_normalization/moments/Squeeze_1:output:0*
T0*
_output_shapes
:@2+
)batch_normalization/AssignMovingAvg_1/subз
)batch_normalization/AssignMovingAvg_1/mulMul-batch_normalization/AssignMovingAvg_1/sub:z:04batch_normalization/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:@2+
)batch_normalization/AssignMovingAvg_1/mul≠
%batch_normalization/AssignMovingAvg_1AssignSubVariableOp=batch_normalization_assignmovingavg_1_readvariableop_resource-batch_normalization/AssignMovingAvg_1/mul:z:05^batch_normalization/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02'
%batch_normalization/AssignMovingAvg_1П
#batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2%
#batch_normalization/batchnorm/add/y“
!batch_normalization/batchnorm/addAddV2.batch_normalization/moments/Squeeze_1:output:0,batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes
:@2#
!batch_normalization/batchnorm/addЯ
#batch_normalization/batchnorm/RsqrtRsqrt%batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes
:@2%
#batch_normalization/batchnorm/RsqrtЏ
0batch_normalization/batchnorm/mul/ReadVariableOpReadVariableOp9batch_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype022
0batch_normalization/batchnorm/mul/ReadVariableOp’
!batch_normalization/batchnorm/mulMul'batch_normalization/batchnorm/Rsqrt:y:08batch_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2#
!batch_normalization/batchnorm/mul–
#batch_normalization/batchnorm/mul_1Mul stream_0_conv_1/BiasAdd:output:0%batch_normalization/batchnorm/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€}@2%
#batch_normalization/batchnorm/mul_1Ћ
#batch_normalization/batchnorm/mul_2Mul,batch_normalization/moments/Squeeze:output:0%batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes
:@2%
#batch_normalization/batchnorm/mul_2ќ
,batch_normalization/batchnorm/ReadVariableOpReadVariableOp5batch_normalization_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02.
,batch_normalization/batchnorm/ReadVariableOp—
!batch_normalization/batchnorm/subSub4batch_normalization/batchnorm/ReadVariableOp:value:0'batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2#
!batch_normalization/batchnorm/subў
#batch_normalization/batchnorm/add_1AddV2'batch_normalization/batchnorm/mul_1:z:0%batch_normalization/batchnorm/sub:z:0*
T0*+
_output_shapes
:€€€€€€€€€}@2%
#batch_normalization/batchnorm/add_1П
activation_2/ReluRelu)batch_normalization_2/batchnorm/add_1:z:0*
T0*+
_output_shapes
:€€€€€€€€€}@2
activation_2/ReluП
activation_1/ReluRelu)batch_normalization_1/batchnorm/add_1:z:0*
T0*+
_output_shapes
:€€€€€€€€€}@2
activation_1/ReluЙ
activation/ReluRelu'batch_normalization/batchnorm/add_1:z:0*
T0*+
_output_shapes
:€€€€€€€€€}@2
activation/ReluГ
stream_2_drop_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †?2
stream_2_drop_1/dropout/Constј
stream_2_drop_1/dropout/MulMulactivation_2/Relu:activations:0&stream_2_drop_1/dropout/Const:output:0*
T0*+
_output_shapes
:€€€€€€€€€}@2
stream_2_drop_1/dropout/MulН
stream_2_drop_1/dropout/ShapeShapeactivation_2/Relu:activations:0*
T0*
_output_shapes
:2
stream_2_drop_1/dropout/ShapeГ
4stream_2_drop_1/dropout/random_uniform/RandomUniformRandomUniform&stream_2_drop_1/dropout/Shape:output:0*
T0*+
_output_shapes
:€€€€€€€€€}@*
dtype0*
seedЈ*
seed2Ј26
4stream_2_drop_1/dropout/random_uniform/RandomUniformХ
&stream_2_drop_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL>2(
&stream_2_drop_1/dropout/GreaterEqual/yВ
$stream_2_drop_1/dropout/GreaterEqualGreaterEqual=stream_2_drop_1/dropout/random_uniform/RandomUniform:output:0/stream_2_drop_1/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:€€€€€€€€€}@2&
$stream_2_drop_1/dropout/GreaterEqual≥
stream_2_drop_1/dropout/CastCast(stream_2_drop_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:€€€€€€€€€}@2
stream_2_drop_1/dropout/CastЊ
stream_2_drop_1/dropout/Mul_1Mulstream_2_drop_1/dropout/Mul:z:0 stream_2_drop_1/dropout/Cast:y:0*
T0*+
_output_shapes
:€€€€€€€€€}@2
stream_2_drop_1/dropout/Mul_1Г
stream_1_drop_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †?2
stream_1_drop_1/dropout/Constј
stream_1_drop_1/dropout/MulMulactivation_1/Relu:activations:0&stream_1_drop_1/dropout/Const:output:0*
T0*+
_output_shapes
:€€€€€€€€€}@2
stream_1_drop_1/dropout/MulН
stream_1_drop_1/dropout/ShapeShapeactivation_1/Relu:activations:0*
T0*
_output_shapes
:2
stream_1_drop_1/dropout/ShapeГ
4stream_1_drop_1/dropout/random_uniform/RandomUniformRandomUniform&stream_1_drop_1/dropout/Shape:output:0*
T0*+
_output_shapes
:€€€€€€€€€}@*
dtype0*
seedЈ*
seed2Ј26
4stream_1_drop_1/dropout/random_uniform/RandomUniformХ
&stream_1_drop_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL>2(
&stream_1_drop_1/dropout/GreaterEqual/yВ
$stream_1_drop_1/dropout/GreaterEqualGreaterEqual=stream_1_drop_1/dropout/random_uniform/RandomUniform:output:0/stream_1_drop_1/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:€€€€€€€€€}@2&
$stream_1_drop_1/dropout/GreaterEqual≥
stream_1_drop_1/dropout/CastCast(stream_1_drop_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:€€€€€€€€€}@2
stream_1_drop_1/dropout/CastЊ
stream_1_drop_1/dropout/Mul_1Mulstream_1_drop_1/dropout/Mul:z:0 stream_1_drop_1/dropout/Cast:y:0*
T0*+
_output_shapes
:€€€€€€€€€}@2
stream_1_drop_1/dropout/Mul_1Г
stream_0_drop_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †?2
stream_0_drop_1/dropout/ConstЊ
stream_0_drop_1/dropout/MulMulactivation/Relu:activations:0&stream_0_drop_1/dropout/Const:output:0*
T0*+
_output_shapes
:€€€€€€€€€}@2
stream_0_drop_1/dropout/MulЛ
stream_0_drop_1/dropout/ShapeShapeactivation/Relu:activations:0*
T0*
_output_shapes
:2
stream_0_drop_1/dropout/ShapeГ
4stream_0_drop_1/dropout/random_uniform/RandomUniformRandomUniform&stream_0_drop_1/dropout/Shape:output:0*
T0*+
_output_shapes
:€€€€€€€€€}@*
dtype0*
seedЈ*
seed2Ј26
4stream_0_drop_1/dropout/random_uniform/RandomUniformХ
&stream_0_drop_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL>2(
&stream_0_drop_1/dropout/GreaterEqual/yВ
$stream_0_drop_1/dropout/GreaterEqualGreaterEqual=stream_0_drop_1/dropout/random_uniform/RandomUniform:output:0/stream_0_drop_1/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:€€€€€€€€€}@2&
$stream_0_drop_1/dropout/GreaterEqual≥
stream_0_drop_1/dropout/CastCast(stream_0_drop_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:€€€€€€€€€}@2
stream_0_drop_1/dropout/CastЊ
stream_0_drop_1/dropout/Mul_1Mulstream_0_drop_1/dropout/Mul:z:0 stream_0_drop_1/dropout/Cast:y:0*
T0*+
_output_shapes
:€€€€€€€€€}@2
stream_0_drop_1/dropout/Mul_1§
/global_average_pooling1d/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :21
/global_average_pooling1d/Mean/reduction_indices’
global_average_pooling1d/MeanMean!stream_0_drop_1/dropout/Mul_1:z:08global_average_pooling1d/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
global_average_pooling1d/Mean®
1global_average_pooling1d_1/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :23
1global_average_pooling1d_1/Mean/reduction_indicesџ
global_average_pooling1d_1/MeanMean!stream_1_drop_1/dropout/Mul_1:z:0:global_average_pooling1d_1/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2!
global_average_pooling1d_1/Mean®
1global_average_pooling1d_2/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :23
1global_average_pooling1d_2/Mean/reduction_indicesџ
global_average_pooling1d_2/MeanMean!stream_2_drop_1/dropout/Mul_1:z:0:global_average_pooling1d_2/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2!
global_average_pooling1d_2/Meant
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate/concat/axisО
concatenate/concatConcatV2&global_average_pooling1d/Mean:output:0(global_average_pooling1d_1/Mean:output:0(global_average_pooling1d_2/Mean:output:0 concatenate/concat/axis:output:0*
N*
T0*(
_output_shapes
:€€€€€€€€€ј2
concatenate/concat¶
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	јT*
dtype02
dense_1/MatMul/ReadVariableOp†
dense_1/MatMulMatMulconcatenate/concat:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€T2
dense_1/MatMul§
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:T*
dtype02 
dense_1/BiasAdd/ReadVariableOp°
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€T2
dense_1/BiasAddґ
4batch_normalization_3/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 26
4batch_normalization_3/moments/mean/reduction_indicesг
"batch_normalization_3/moments/meanMeandense_1/BiasAdd:output:0=batch_normalization_3/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:T*
	keep_dims(2$
"batch_normalization_3/moments/meanЊ
*batch_normalization_3/moments/StopGradientStopGradient+batch_normalization_3/moments/mean:output:0*
T0*
_output_shapes

:T2,
*batch_normalization_3/moments/StopGradientш
/batch_normalization_3/moments/SquaredDifferenceSquaredDifferencedense_1/BiasAdd:output:03batch_normalization_3/moments/StopGradient:output:0*
T0*'
_output_shapes
:€€€€€€€€€T21
/batch_normalization_3/moments/SquaredDifferenceЊ
8batch_normalization_3/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2:
8batch_normalization_3/moments/variance/reduction_indicesК
&batch_normalization_3/moments/varianceMean3batch_normalization_3/moments/SquaredDifference:z:0Abatch_normalization_3/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:T*
	keep_dims(2(
&batch_normalization_3/moments/variance¬
%batch_normalization_3/moments/SqueezeSqueeze+batch_normalization_3/moments/mean:output:0*
T0*
_output_shapes
:T*
squeeze_dims
 2'
%batch_normalization_3/moments/Squeeze 
'batch_normalization_3/moments/Squeeze_1Squeeze/batch_normalization_3/moments/variance:output:0*
T0*
_output_shapes
:T*
squeeze_dims
 2)
'batch_normalization_3/moments/Squeeze_1Я
+batch_normalization_3/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2-
+batch_normalization_3/AssignMovingAvg/decayж
4batch_normalization_3/AssignMovingAvg/ReadVariableOpReadVariableOp=batch_normalization_3_assignmovingavg_readvariableop_resource*
_output_shapes
:T*
dtype026
4batch_normalization_3/AssignMovingAvg/ReadVariableOpр
)batch_normalization_3/AssignMovingAvg/subSub<batch_normalization_3/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_3/moments/Squeeze:output:0*
T0*
_output_shapes
:T2+
)batch_normalization_3/AssignMovingAvg/subз
)batch_normalization_3/AssignMovingAvg/mulMul-batch_normalization_3/AssignMovingAvg/sub:z:04batch_normalization_3/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:T2+
)batch_normalization_3/AssignMovingAvg/mul≠
%batch_normalization_3/AssignMovingAvgAssignSubVariableOp=batch_normalization_3_assignmovingavg_readvariableop_resource-batch_normalization_3/AssignMovingAvg/mul:z:05^batch_normalization_3/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02'
%batch_normalization_3/AssignMovingAvg£
-batch_normalization_3/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2/
-batch_normalization_3/AssignMovingAvg_1/decayм
6batch_normalization_3/AssignMovingAvg_1/ReadVariableOpReadVariableOp?batch_normalization_3_assignmovingavg_1_readvariableop_resource*
_output_shapes
:T*
dtype028
6batch_normalization_3/AssignMovingAvg_1/ReadVariableOpш
+batch_normalization_3/AssignMovingAvg_1/subSub>batch_normalization_3/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_3/moments/Squeeze_1:output:0*
T0*
_output_shapes
:T2-
+batch_normalization_3/AssignMovingAvg_1/subп
+batch_normalization_3/AssignMovingAvg_1/mulMul/batch_normalization_3/AssignMovingAvg_1/sub:z:06batch_normalization_3/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:T2-
+batch_normalization_3/AssignMovingAvg_1/mulЈ
'batch_normalization_3/AssignMovingAvg_1AssignSubVariableOp?batch_normalization_3_assignmovingavg_1_readvariableop_resource/batch_normalization_3/AssignMovingAvg_1/mul:z:07^batch_normalization_3/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02)
'batch_normalization_3/AssignMovingAvg_1У
%batch_normalization_3/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2'
%batch_normalization_3/batchnorm/add/yЏ
#batch_normalization_3/batchnorm/addAddV20batch_normalization_3/moments/Squeeze_1:output:0.batch_normalization_3/batchnorm/add/y:output:0*
T0*
_output_shapes
:T2%
#batch_normalization_3/batchnorm/add•
%batch_normalization_3/batchnorm/RsqrtRsqrt'batch_normalization_3/batchnorm/add:z:0*
T0*
_output_shapes
:T2'
%batch_normalization_3/batchnorm/Rsqrtа
2batch_normalization_3/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_3_batchnorm_mul_readvariableop_resource*
_output_shapes
:T*
dtype024
2batch_normalization_3/batchnorm/mul/ReadVariableOpЁ
#batch_normalization_3/batchnorm/mulMul)batch_normalization_3/batchnorm/Rsqrt:y:0:batch_normalization_3/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:T2%
#batch_normalization_3/batchnorm/mul 
%batch_normalization_3/batchnorm/mul_1Muldense_1/BiasAdd:output:0'batch_normalization_3/batchnorm/mul:z:0*
T0*'
_output_shapes
:€€€€€€€€€T2'
%batch_normalization_3/batchnorm/mul_1”
%batch_normalization_3/batchnorm/mul_2Mul.batch_normalization_3/moments/Squeeze:output:0'batch_normalization_3/batchnorm/mul:z:0*
T0*
_output_shapes
:T2'
%batch_normalization_3/batchnorm/mul_2‘
.batch_normalization_3/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_3_batchnorm_readvariableop_resource*
_output_shapes
:T*
dtype020
.batch_normalization_3/batchnorm/ReadVariableOpў
#batch_normalization_3/batchnorm/subSub6batch_normalization_3/batchnorm/ReadVariableOp:value:0)batch_normalization_3/batchnorm/mul_2:z:0*
T0*
_output_shapes
:T2%
#batch_normalization_3/batchnorm/subЁ
%batch_normalization_3/batchnorm/add_1AddV2)batch_normalization_3/batchnorm/mul_1:z:0'batch_normalization_3/batchnorm/sub:z:0*
T0*'
_output_shapes
:€€€€€€€€€T2'
%batch_normalization_3/batchnorm/add_1†
dense_activation_1/SigmoidSigmoid)batch_normalization_3/batchnorm/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€T2
dense_activation_1/Sigmoidо
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp;stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype027
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp√
&stream_0_conv_1/kernel/Regularizer/AbsAbs=stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_0_conv_1/kernel/Regularizer/Abs©
(stream_0_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_0_conv_1/kernel/Regularizer/Const„
&stream_0_conv_1/kernel/Regularizer/SumSum*stream_0_conv_1/kernel/Regularizer/Abs:y:01stream_0_conv_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/SumЩ
(stream_0_conv_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2*
(stream_0_conv_1/kernel/Regularizer/mul/x№
&stream_0_conv_1/kernel/Regularizer/mulMul1stream_0_conv_1/kernel/Regularizer/mul/x:output:0/stream_0_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/mulф
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;stream_1_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02:
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpѕ
)stream_1_conv_1/kernel/Regularizer/SquareSquare@stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2+
)stream_1_conv_1/kernel/Regularizer/Square©
(stream_1_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_1_conv_1/kernel/Regularizer/ConstЏ
&stream_1_conv_1/kernel/Regularizer/SumSum-stream_1_conv_1/kernel/Regularizer/Square:y:01stream_1_conv_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&stream_1_conv_1/kernel/Regularizer/SumЩ
(stream_1_conv_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2*
(stream_1_conv_1/kernel/Regularizer/mul/x№
&stream_1_conv_1/kernel/Regularizer/mulMul1stream_1_conv_1/kernel/Regularizer/mul/x:output:0/stream_1_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_1_conv_1/kernel/Regularizer/mulо
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp;stream_2_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype027
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp√
&stream_2_conv_1/kernel/Regularizer/AbsAbs=stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_2_conv_1/kernel/Regularizer/Abs©
(stream_2_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_2_conv_1/kernel/Regularizer/Const„
&stream_2_conv_1/kernel/Regularizer/SumSum*stream_2_conv_1/kernel/Regularizer/Abs:y:01stream_2_conv_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&stream_2_conv_1/kernel/Regularizer/SumЩ
(stream_2_conv_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2*
(stream_2_conv_1/kernel/Regularizer/mul/x№
&stream_2_conv_1/kernel/Regularizer/mulMul1stream_2_conv_1/kernel/Regularizer/mul/x:output:0/stream_2_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_2_conv_1/kernel/Regularizer/mul∆
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	јT*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOp®
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	јT2 
dense_1/kernel/Regularizer/AbsХ
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_1/kernel/Regularizer/ConstЈ
dense_1/kernel/Regularizer/SumSum"dense_1/kernel/Regularizer/Abs:y:0)dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/SumЙ
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2"
 dense_1/kernel/Regularizer/mul/xЉ
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/muly
IdentityIdentitydense_activation_1/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€T2

IdentityШ
NoOpNoOp$^batch_normalization/AssignMovingAvg3^batch_normalization/AssignMovingAvg/ReadVariableOp&^batch_normalization/AssignMovingAvg_15^batch_normalization/AssignMovingAvg_1/ReadVariableOp-^batch_normalization/batchnorm/ReadVariableOp1^batch_normalization/batchnorm/mul/ReadVariableOp&^batch_normalization_1/AssignMovingAvg5^batch_normalization_1/AssignMovingAvg/ReadVariableOp(^batch_normalization_1/AssignMovingAvg_17^batch_normalization_1/AssignMovingAvg_1/ReadVariableOp/^batch_normalization_1/batchnorm/ReadVariableOp3^batch_normalization_1/batchnorm/mul/ReadVariableOp&^batch_normalization_2/AssignMovingAvg5^batch_normalization_2/AssignMovingAvg/ReadVariableOp(^batch_normalization_2/AssignMovingAvg_17^batch_normalization_2/AssignMovingAvg_1/ReadVariableOp/^batch_normalization_2/batchnorm/ReadVariableOp3^batch_normalization_2/batchnorm/mul/ReadVariableOp&^batch_normalization_3/AssignMovingAvg5^batch_normalization_3/AssignMovingAvg/ReadVariableOp(^batch_normalization_3/AssignMovingAvg_17^batch_normalization_3/AssignMovingAvg_1/ReadVariableOp/^batch_normalization_3/batchnorm/ReadVariableOp3^batch_normalization_3/batchnorm/mul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp.^dense_1/kernel/Regularizer/Abs/ReadVariableOp'^stream_0_conv_1/BiasAdd/ReadVariableOp3^stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp'^stream_1_conv_1/BiasAdd/ReadVariableOp3^stream_1_conv_1/conv1d/ExpandDims_1/ReadVariableOp9^stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp'^stream_2_conv_1/BiasAdd/ReadVariableOp3^stream_2_conv_1/conv1d/ExpandDims_1/ReadVariableOp6^stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*И
_input_shapesw
u:€€€€€€€€€}:€€€€€€€€€}:€€€€€€€€€}: : : : : : : : : : : : : : : : : : : : : : : : 2J
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
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp:U Q
+
_output_shapes
:€€€€€€€€€}
"
_user_specified_name
inputs/0:UQ
+
_output_shapes
:€€€€€€€€€}
"
_user_specified_name
inputs/1:UQ
+
_output_shapes
:€€€€€€€€€}
"
_user_specified_name
inputs/2
Є+
к
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_103795

inputs5
'assignmovingavg_readvariableop_resource:@7
)assignmovingavg_1_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@/
!batchnorm_readvariableop_resource:@
identityИҐAssignMovingAvgҐAssignMovingAvg/ReadVariableOpҐAssignMovingAvg_1Ґ AssignMovingAvg_1/ReadVariableOpҐbatchnorm/ReadVariableOpҐbatchnorm/mul/ReadVariableOpС
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indicesУ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(2
moments/meanА
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:@2
moments/StopGradient±
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€@2
moments/SquaredDifferenceЩ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indicesґ
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(2
moments/varianceБ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 2
moments/SqueezeЙ
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
„#<2
AssignMovingAvg/decay§
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOpШ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg/subП
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg/mulњ
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
„#<2
AssignMovingAvg_1/decay™
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOp†
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg_1/subЧ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg_1/mul…
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
 *oГ:2
batchnorm/add/yВ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2
batchnorm/mulГ
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€@2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:@2
batchnorm/mul_2Т
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOpБ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2
batchnorm/subТ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€@2
batchnorm/add_1{
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€@2

Identityт
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:€€€€€€€€€€€€€€€€€€@: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs
х
∞
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_104131

inputs/
!batchnorm_readvariableop_resource:T3
%batchnorm_mul_readvariableop_resource:T1
#batchnorm_readvariableop_1_resource:T1
#batchnorm_readvariableop_2_resource:T
identityИҐbatchnorm/ReadVariableOpҐbatchnorm/ReadVariableOp_1Ґbatchnorm/ReadVariableOp_2Ґbatchnorm/mul/ReadVariableOpТ
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
 *oГ:2
batchnorm/add/yИ
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:T2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:T2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:T*
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:T2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:€€€€€€€€€T2
batchnorm/mul_1Ш
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:T*
dtype02
batchnorm/ReadVariableOp_1Е
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:T2
batchnorm/mul_2Ш
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:T*
dtype02
batchnorm/ReadVariableOp_2Г
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:T2
batchnorm/subЕ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:€€€€€€€€€T2
batchnorm/add_1n
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€T2

Identity¬
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€T: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€T
 
_user_specified_nameinputs
о
b
F__inference_activation_layer_call_and_return_conditional_losses_108368

inputs
identityR
ReluReluinputs*
T0*+
_output_shapes
:€€€€€€€€€}@2
Reluj
IdentityIdentityRelu:activations:0*
T0*+
_output_shapes
:€€€€€€€€€}@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€}@:S O
+
_output_shapes
:€€€€€€€€€}@
 
_user_specified_nameinputs
и
j
N__inference_dense_activation_1_layer_call_and_return_conditional_losses_108690

inputs
identityW
SigmoidSigmoidinputs*
T0*'
_output_shapes
:€€€€€€€€€T2	
Sigmoid_
IdentityIdentitySigmoid:y:0*
T0*'
_output_shapes
:€€€€€€€€€T2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:€€€€€€€€€T:O K
'
_output_shapes
:€€€€€€€€€T
 
_user_specified_nameinputs
ЉИ
Б(
A__inference_model_layer_call_and_return_conditional_losses_106565
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
0basemodel_dense_1_matmul_readvariableop_resource:	јT?
1basemodel_dense_1_biasadd_readvariableop_resource:TO
Abasemodel_batch_normalization_3_batchnorm_readvariableop_resource:TS
Ebasemodel_batch_normalization_3_batchnorm_mul_readvariableop_resource:TQ
Cbasemodel_batch_normalization_3_batchnorm_readvariableop_1_resource:TQ
Cbasemodel_batch_normalization_3_batchnorm_readvariableop_2_resource:T
identityИҐ6basemodel/batch_normalization/batchnorm/ReadVariableOpҐ8basemodel/batch_normalization/batchnorm/ReadVariableOp_1Ґ8basemodel/batch_normalization/batchnorm/ReadVariableOp_2Ґ:basemodel/batch_normalization/batchnorm/mul/ReadVariableOpҐ8basemodel/batch_normalization/batchnorm_1/ReadVariableOpҐ:basemodel/batch_normalization/batchnorm_1/ReadVariableOp_1Ґ:basemodel/batch_normalization/batchnorm_1/ReadVariableOp_2Ґ<basemodel/batch_normalization/batchnorm_1/mul/ReadVariableOpҐ8basemodel/batch_normalization_1/batchnorm/ReadVariableOpҐ:basemodel/batch_normalization_1/batchnorm/ReadVariableOp_1Ґ:basemodel/batch_normalization_1/batchnorm/ReadVariableOp_2Ґ<basemodel/batch_normalization_1/batchnorm/mul/ReadVariableOpҐ:basemodel/batch_normalization_1/batchnorm_1/ReadVariableOpҐ<basemodel/batch_normalization_1/batchnorm_1/ReadVariableOp_1Ґ<basemodel/batch_normalization_1/batchnorm_1/ReadVariableOp_2Ґ>basemodel/batch_normalization_1/batchnorm_1/mul/ReadVariableOpҐ8basemodel/batch_normalization_2/batchnorm/ReadVariableOpҐ:basemodel/batch_normalization_2/batchnorm/ReadVariableOp_1Ґ:basemodel/batch_normalization_2/batchnorm/ReadVariableOp_2Ґ<basemodel/batch_normalization_2/batchnorm/mul/ReadVariableOpҐ:basemodel/batch_normalization_2/batchnorm_1/ReadVariableOpҐ<basemodel/batch_normalization_2/batchnorm_1/ReadVariableOp_1Ґ<basemodel/batch_normalization_2/batchnorm_1/ReadVariableOp_2Ґ>basemodel/batch_normalization_2/batchnorm_1/mul/ReadVariableOpҐ8basemodel/batch_normalization_3/batchnorm/ReadVariableOpҐ:basemodel/batch_normalization_3/batchnorm/ReadVariableOp_1Ґ:basemodel/batch_normalization_3/batchnorm/ReadVariableOp_2Ґ<basemodel/batch_normalization_3/batchnorm/mul/ReadVariableOpҐ:basemodel/batch_normalization_3/batchnorm_1/ReadVariableOpҐ<basemodel/batch_normalization_3/batchnorm_1/ReadVariableOp_1Ґ<basemodel/batch_normalization_3/batchnorm_1/ReadVariableOp_2Ґ>basemodel/batch_normalization_3/batchnorm_1/mul/ReadVariableOpҐ(basemodel/dense_1/BiasAdd/ReadVariableOpҐ*basemodel/dense_1/BiasAdd_1/ReadVariableOpҐ'basemodel/dense_1/MatMul/ReadVariableOpҐ)basemodel/dense_1/MatMul_1/ReadVariableOpҐ0basemodel/stream_0_conv_1/BiasAdd/ReadVariableOpҐ2basemodel/stream_0_conv_1/BiasAdd_1/ReadVariableOpҐ<basemodel/stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOpҐ>basemodel/stream_0_conv_1/conv1d_1/ExpandDims_1/ReadVariableOpҐ0basemodel/stream_1_conv_1/BiasAdd/ReadVariableOpҐ2basemodel/stream_1_conv_1/BiasAdd_1/ReadVariableOpҐ<basemodel/stream_1_conv_1/conv1d/ExpandDims_1/ReadVariableOpҐ>basemodel/stream_1_conv_1/conv1d_1/ExpandDims_1/ReadVariableOpҐ0basemodel/stream_2_conv_1/BiasAdd/ReadVariableOpҐ2basemodel/stream_2_conv_1/BiasAdd_1/ReadVariableOpҐ<basemodel/stream_2_conv_1/conv1d/ExpandDims_1/ReadVariableOpҐ>basemodel/stream_2_conv_1/conv1d_1/ExpandDims_1/ReadVariableOpҐ-dense_1/kernel/Regularizer/Abs/ReadVariableOpҐ5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpҐ8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpҐ5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOpЬ
&basemodel/stream_2_input_drop/IdentityIdentityinputs_0*
T0*+
_output_shapes
:€€€€€€€€€}2(
&basemodel/stream_2_input_drop/IdentityЬ
&basemodel/stream_1_input_drop/IdentityIdentityinputs_0*
T0*+
_output_shapes
:€€€€€€€€€}2(
&basemodel/stream_1_input_drop/IdentityЬ
&basemodel/stream_0_input_drop/IdentityIdentityinputs_0*
T0*+
_output_shapes
:€€€€€€€€€}2(
&basemodel/stream_0_input_drop/Identity≠
/basemodel/stream_2_conv_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€21
/basemodel/stream_2_conv_1/conv1d/ExpandDims/dimН
+basemodel/stream_2_conv_1/conv1d/ExpandDims
ExpandDims/basemodel/stream_2_input_drop/Identity:output:08basemodel/stream_2_conv_1/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:€€€€€€€€€}2-
+basemodel/stream_2_conv_1/conv1d/ExpandDimsЖ
<basemodel/stream_2_conv_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpEbasemodel_stream_2_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02>
<basemodel/stream_2_conv_1/conv1d/ExpandDims_1/ReadVariableOp®
1basemodel/stream_2_conv_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 23
1basemodel/stream_2_conv_1/conv1d/ExpandDims_1/dimЯ
-basemodel/stream_2_conv_1/conv1d/ExpandDims_1
ExpandDimsDbasemodel/stream_2_conv_1/conv1d/ExpandDims_1/ReadVariableOp:value:0:basemodel/stream_2_conv_1/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@2/
-basemodel/stream_2_conv_1/conv1d/ExpandDims_1Ю
 basemodel/stream_2_conv_1/conv1dConv2D4basemodel/stream_2_conv_1/conv1d/ExpandDims:output:06basemodel/stream_2_conv_1/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:€€€€€€€€€}@*
paddingSAME*
strides
2"
 basemodel/stream_2_conv_1/conv1dа
(basemodel/stream_2_conv_1/conv1d/SqueezeSqueeze)basemodel/stream_2_conv_1/conv1d:output:0*
T0*+
_output_shapes
:€€€€€€€€€}@*
squeeze_dims

э€€€€€€€€2*
(basemodel/stream_2_conv_1/conv1d/SqueezeЏ
0basemodel/stream_2_conv_1/BiasAdd/ReadVariableOpReadVariableOp9basemodel_stream_2_conv_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype022
0basemodel/stream_2_conv_1/BiasAdd/ReadVariableOpф
!basemodel/stream_2_conv_1/BiasAddBiasAdd1basemodel/stream_2_conv_1/conv1d/Squeeze:output:08basemodel/stream_2_conv_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€}@2#
!basemodel/stream_2_conv_1/BiasAdd≠
/basemodel/stream_1_conv_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€21
/basemodel/stream_1_conv_1/conv1d/ExpandDims/dimН
+basemodel/stream_1_conv_1/conv1d/ExpandDims
ExpandDims/basemodel/stream_1_input_drop/Identity:output:08basemodel/stream_1_conv_1/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:€€€€€€€€€}2-
+basemodel/stream_1_conv_1/conv1d/ExpandDimsЖ
<basemodel/stream_1_conv_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpEbasemodel_stream_1_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02>
<basemodel/stream_1_conv_1/conv1d/ExpandDims_1/ReadVariableOp®
1basemodel/stream_1_conv_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 23
1basemodel/stream_1_conv_1/conv1d/ExpandDims_1/dimЯ
-basemodel/stream_1_conv_1/conv1d/ExpandDims_1
ExpandDimsDbasemodel/stream_1_conv_1/conv1d/ExpandDims_1/ReadVariableOp:value:0:basemodel/stream_1_conv_1/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@2/
-basemodel/stream_1_conv_1/conv1d/ExpandDims_1Ю
 basemodel/stream_1_conv_1/conv1dConv2D4basemodel/stream_1_conv_1/conv1d/ExpandDims:output:06basemodel/stream_1_conv_1/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:€€€€€€€€€}@*
paddingSAME*
strides
2"
 basemodel/stream_1_conv_1/conv1dа
(basemodel/stream_1_conv_1/conv1d/SqueezeSqueeze)basemodel/stream_1_conv_1/conv1d:output:0*
T0*+
_output_shapes
:€€€€€€€€€}@*
squeeze_dims

э€€€€€€€€2*
(basemodel/stream_1_conv_1/conv1d/SqueezeЏ
0basemodel/stream_1_conv_1/BiasAdd/ReadVariableOpReadVariableOp9basemodel_stream_1_conv_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype022
0basemodel/stream_1_conv_1/BiasAdd/ReadVariableOpф
!basemodel/stream_1_conv_1/BiasAddBiasAdd1basemodel/stream_1_conv_1/conv1d/Squeeze:output:08basemodel/stream_1_conv_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€}@2#
!basemodel/stream_1_conv_1/BiasAdd≠
/basemodel/stream_0_conv_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€21
/basemodel/stream_0_conv_1/conv1d/ExpandDims/dimН
+basemodel/stream_0_conv_1/conv1d/ExpandDims
ExpandDims/basemodel/stream_0_input_drop/Identity:output:08basemodel/stream_0_conv_1/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:€€€€€€€€€}2-
+basemodel/stream_0_conv_1/conv1d/ExpandDimsЖ
<basemodel/stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpEbasemodel_stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02>
<basemodel/stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp®
1basemodel/stream_0_conv_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 23
1basemodel/stream_0_conv_1/conv1d/ExpandDims_1/dimЯ
-basemodel/stream_0_conv_1/conv1d/ExpandDims_1
ExpandDimsDbasemodel/stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp:value:0:basemodel/stream_0_conv_1/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@2/
-basemodel/stream_0_conv_1/conv1d/ExpandDims_1Ю
 basemodel/stream_0_conv_1/conv1dConv2D4basemodel/stream_0_conv_1/conv1d/ExpandDims:output:06basemodel/stream_0_conv_1/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:€€€€€€€€€}@*
paddingSAME*
strides
2"
 basemodel/stream_0_conv_1/conv1dа
(basemodel/stream_0_conv_1/conv1d/SqueezeSqueeze)basemodel/stream_0_conv_1/conv1d:output:0*
T0*+
_output_shapes
:€€€€€€€€€}@*
squeeze_dims

э€€€€€€€€2*
(basemodel/stream_0_conv_1/conv1d/SqueezeЏ
0basemodel/stream_0_conv_1/BiasAdd/ReadVariableOpReadVariableOp9basemodel_stream_0_conv_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype022
0basemodel/stream_0_conv_1/BiasAdd/ReadVariableOpф
!basemodel/stream_0_conv_1/BiasAddBiasAdd1basemodel/stream_0_conv_1/conv1d/Squeeze:output:08basemodel/stream_0_conv_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€}@2#
!basemodel/stream_0_conv_1/BiasAddт
8basemodel/batch_normalization_2/batchnorm/ReadVariableOpReadVariableOpAbasemodel_batch_normalization_2_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02:
8basemodel/batch_normalization_2/batchnorm/ReadVariableOpІ
/basemodel/batch_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:21
/basemodel/batch_normalization_2/batchnorm/add/yИ
-basemodel/batch_normalization_2/batchnorm/addAddV2@basemodel/batch_normalization_2/batchnorm/ReadVariableOp:value:08basemodel/batch_normalization_2/batchnorm/add/y:output:0*
T0*
_output_shapes
:@2/
-basemodel/batch_normalization_2/batchnorm/add√
/basemodel/batch_normalization_2/batchnorm/RsqrtRsqrt1basemodel/batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes
:@21
/basemodel/batch_normalization_2/batchnorm/Rsqrtю
<basemodel/batch_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOpEbasemodel_batch_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02>
<basemodel/batch_normalization_2/batchnorm/mul/ReadVariableOpЕ
-basemodel/batch_normalization_2/batchnorm/mulMul3basemodel/batch_normalization_2/batchnorm/Rsqrt:y:0Dbasemodel/batch_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2/
-basemodel/batch_normalization_2/batchnorm/mulю
/basemodel/batch_normalization_2/batchnorm/mul_1Mul*basemodel/stream_2_conv_1/BiasAdd:output:01basemodel/batch_normalization_2/batchnorm/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€}@21
/basemodel/batch_normalization_2/batchnorm/mul_1ш
:basemodel/batch_normalization_2/batchnorm/ReadVariableOp_1ReadVariableOpCbasemodel_batch_normalization_2_batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype02<
:basemodel/batch_normalization_2/batchnorm/ReadVariableOp_1Е
/basemodel/batch_normalization_2/batchnorm/mul_2MulBbasemodel/batch_normalization_2/batchnorm/ReadVariableOp_1:value:01basemodel/batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes
:@21
/basemodel/batch_normalization_2/batchnorm/mul_2ш
:basemodel/batch_normalization_2/batchnorm/ReadVariableOp_2ReadVariableOpCbasemodel_batch_normalization_2_batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype02<
:basemodel/batch_normalization_2/batchnorm/ReadVariableOp_2Г
-basemodel/batch_normalization_2/batchnorm/subSubBbasemodel/batch_normalization_2/batchnorm/ReadVariableOp_2:value:03basemodel/batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2/
-basemodel/batch_normalization_2/batchnorm/subЙ
/basemodel/batch_normalization_2/batchnorm/add_1AddV23basemodel/batch_normalization_2/batchnorm/mul_1:z:01basemodel/batch_normalization_2/batchnorm/sub:z:0*
T0*+
_output_shapes
:€€€€€€€€€}@21
/basemodel/batch_normalization_2/batchnorm/add_1т
8basemodel/batch_normalization_1/batchnorm/ReadVariableOpReadVariableOpAbasemodel_batch_normalization_1_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02:
8basemodel/batch_normalization_1/batchnorm/ReadVariableOpІ
/basemodel/batch_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:21
/basemodel/batch_normalization_1/batchnorm/add/yИ
-basemodel/batch_normalization_1/batchnorm/addAddV2@basemodel/batch_normalization_1/batchnorm/ReadVariableOp:value:08basemodel/batch_normalization_1/batchnorm/add/y:output:0*
T0*
_output_shapes
:@2/
-basemodel/batch_normalization_1/batchnorm/add√
/basemodel/batch_normalization_1/batchnorm/RsqrtRsqrt1basemodel/batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes
:@21
/basemodel/batch_normalization_1/batchnorm/Rsqrtю
<basemodel/batch_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOpEbasemodel_batch_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02>
<basemodel/batch_normalization_1/batchnorm/mul/ReadVariableOpЕ
-basemodel/batch_normalization_1/batchnorm/mulMul3basemodel/batch_normalization_1/batchnorm/Rsqrt:y:0Dbasemodel/batch_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2/
-basemodel/batch_normalization_1/batchnorm/mulю
/basemodel/batch_normalization_1/batchnorm/mul_1Mul*basemodel/stream_1_conv_1/BiasAdd:output:01basemodel/batch_normalization_1/batchnorm/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€}@21
/basemodel/batch_normalization_1/batchnorm/mul_1ш
:basemodel/batch_normalization_1/batchnorm/ReadVariableOp_1ReadVariableOpCbasemodel_batch_normalization_1_batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype02<
:basemodel/batch_normalization_1/batchnorm/ReadVariableOp_1Е
/basemodel/batch_normalization_1/batchnorm/mul_2MulBbasemodel/batch_normalization_1/batchnorm/ReadVariableOp_1:value:01basemodel/batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes
:@21
/basemodel/batch_normalization_1/batchnorm/mul_2ш
:basemodel/batch_normalization_1/batchnorm/ReadVariableOp_2ReadVariableOpCbasemodel_batch_normalization_1_batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype02<
:basemodel/batch_normalization_1/batchnorm/ReadVariableOp_2Г
-basemodel/batch_normalization_1/batchnorm/subSubBbasemodel/batch_normalization_1/batchnorm/ReadVariableOp_2:value:03basemodel/batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2/
-basemodel/batch_normalization_1/batchnorm/subЙ
/basemodel/batch_normalization_1/batchnorm/add_1AddV23basemodel/batch_normalization_1/batchnorm/mul_1:z:01basemodel/batch_normalization_1/batchnorm/sub:z:0*
T0*+
_output_shapes
:€€€€€€€€€}@21
/basemodel/batch_normalization_1/batchnorm/add_1м
6basemodel/batch_normalization/batchnorm/ReadVariableOpReadVariableOp?basemodel_batch_normalization_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype028
6basemodel/batch_normalization/batchnorm/ReadVariableOp£
-basemodel/batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2/
-basemodel/batch_normalization/batchnorm/add/yА
+basemodel/batch_normalization/batchnorm/addAddV2>basemodel/batch_normalization/batchnorm/ReadVariableOp:value:06basemodel/batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes
:@2-
+basemodel/batch_normalization/batchnorm/addљ
-basemodel/batch_normalization/batchnorm/RsqrtRsqrt/basemodel/batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes
:@2/
-basemodel/batch_normalization/batchnorm/Rsqrtш
:basemodel/batch_normalization/batchnorm/mul/ReadVariableOpReadVariableOpCbasemodel_batch_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02<
:basemodel/batch_normalization/batchnorm/mul/ReadVariableOpэ
+basemodel/batch_normalization/batchnorm/mulMul1basemodel/batch_normalization/batchnorm/Rsqrt:y:0Bbasemodel/batch_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2-
+basemodel/batch_normalization/batchnorm/mulш
-basemodel/batch_normalization/batchnorm/mul_1Mul*basemodel/stream_0_conv_1/BiasAdd:output:0/basemodel/batch_normalization/batchnorm/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€}@2/
-basemodel/batch_normalization/batchnorm/mul_1т
8basemodel/batch_normalization/batchnorm/ReadVariableOp_1ReadVariableOpAbasemodel_batch_normalization_batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype02:
8basemodel/batch_normalization/batchnorm/ReadVariableOp_1э
-basemodel/batch_normalization/batchnorm/mul_2Mul@basemodel/batch_normalization/batchnorm/ReadVariableOp_1:value:0/basemodel/batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes
:@2/
-basemodel/batch_normalization/batchnorm/mul_2т
8basemodel/batch_normalization/batchnorm/ReadVariableOp_2ReadVariableOpAbasemodel_batch_normalization_batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype02:
8basemodel/batch_normalization/batchnorm/ReadVariableOp_2ы
+basemodel/batch_normalization/batchnorm/subSub@basemodel/batch_normalization/batchnorm/ReadVariableOp_2:value:01basemodel/batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2-
+basemodel/batch_normalization/batchnorm/subБ
-basemodel/batch_normalization/batchnorm/add_1AddV21basemodel/batch_normalization/batchnorm/mul_1:z:0/basemodel/batch_normalization/batchnorm/sub:z:0*
T0*+
_output_shapes
:€€€€€€€€€}@2/
-basemodel/batch_normalization/batchnorm/add_1≠
basemodel/activation_2/ReluRelu3basemodel/batch_normalization_2/batchnorm/add_1:z:0*
T0*+
_output_shapes
:€€€€€€€€€}@2
basemodel/activation_2/Relu≠
basemodel/activation_1/ReluRelu3basemodel/batch_normalization_1/batchnorm/add_1:z:0*
T0*+
_output_shapes
:€€€€€€€€€}@2
basemodel/activation_1/ReluІ
basemodel/activation/ReluRelu1basemodel/batch_normalization/batchnorm/add_1:z:0*
T0*+
_output_shapes
:€€€€€€€€€}@2
basemodel/activation/Reluµ
"basemodel/stream_2_drop_1/IdentityIdentity)basemodel/activation_2/Relu:activations:0*
T0*+
_output_shapes
:€€€€€€€€€}@2$
"basemodel/stream_2_drop_1/Identityµ
"basemodel/stream_1_drop_1/IdentityIdentity)basemodel/activation_1/Relu:activations:0*
T0*+
_output_shapes
:€€€€€€€€€}@2$
"basemodel/stream_1_drop_1/Identity≥
"basemodel/stream_0_drop_1/IdentityIdentity'basemodel/activation/Relu:activations:0*
T0*+
_output_shapes
:€€€€€€€€€}@2$
"basemodel/stream_0_drop_1/IdentityЄ
9basemodel/global_average_pooling1d/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2;
9basemodel/global_average_pooling1d/Mean/reduction_indicesэ
'basemodel/global_average_pooling1d/MeanMean+basemodel/stream_0_drop_1/Identity:output:0Bbasemodel/global_average_pooling1d/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2)
'basemodel/global_average_pooling1d/MeanЉ
;basemodel/global_average_pooling1d_1/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2=
;basemodel/global_average_pooling1d_1/Mean/reduction_indicesГ
)basemodel/global_average_pooling1d_1/MeanMean+basemodel/stream_1_drop_1/Identity:output:0Dbasemodel/global_average_pooling1d_1/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2+
)basemodel/global_average_pooling1d_1/MeanЉ
;basemodel/global_average_pooling1d_2/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2=
;basemodel/global_average_pooling1d_2/Mean/reduction_indicesГ
)basemodel/global_average_pooling1d_2/MeanMean+basemodel/stream_2_drop_1/Identity:output:0Dbasemodel/global_average_pooling1d_2/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2+
)basemodel/global_average_pooling1d_2/MeanИ
!basemodel/concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2#
!basemodel/concatenate/concat/axis 
basemodel/concatenate/concatConcatV20basemodel/global_average_pooling1d/Mean:output:02basemodel/global_average_pooling1d_1/Mean:output:02basemodel/global_average_pooling1d_2/Mean:output:0*basemodel/concatenate/concat/axis:output:0*
N*
T0*(
_output_shapes
:€€€€€€€€€ј2
basemodel/concatenate/concatЃ
"basemodel/dense_1_dropout/IdentityIdentity%basemodel/concatenate/concat:output:0*
T0*(
_output_shapes
:€€€€€€€€€ј2$
"basemodel/dense_1_dropout/Identityƒ
'basemodel/dense_1/MatMul/ReadVariableOpReadVariableOp0basemodel_dense_1_matmul_readvariableop_resource*
_output_shapes
:	јT*
dtype02)
'basemodel/dense_1/MatMul/ReadVariableOpќ
basemodel/dense_1/MatMulMatMul+basemodel/dense_1_dropout/Identity:output:0/basemodel/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€T2
basemodel/dense_1/MatMul¬
(basemodel/dense_1/BiasAdd/ReadVariableOpReadVariableOp1basemodel_dense_1_biasadd_readvariableop_resource*
_output_shapes
:T*
dtype02*
(basemodel/dense_1/BiasAdd/ReadVariableOp…
basemodel/dense_1/BiasAddBiasAdd"basemodel/dense_1/MatMul:product:00basemodel/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€T2
basemodel/dense_1/BiasAddт
8basemodel/batch_normalization_3/batchnorm/ReadVariableOpReadVariableOpAbasemodel_batch_normalization_3_batchnorm_readvariableop_resource*
_output_shapes
:T*
dtype02:
8basemodel/batch_normalization_3/batchnorm/ReadVariableOpІ
/basemodel/batch_normalization_3/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:21
/basemodel/batch_normalization_3/batchnorm/add/yИ
-basemodel/batch_normalization_3/batchnorm/addAddV2@basemodel/batch_normalization_3/batchnorm/ReadVariableOp:value:08basemodel/batch_normalization_3/batchnorm/add/y:output:0*
T0*
_output_shapes
:T2/
-basemodel/batch_normalization_3/batchnorm/add√
/basemodel/batch_normalization_3/batchnorm/RsqrtRsqrt1basemodel/batch_normalization_3/batchnorm/add:z:0*
T0*
_output_shapes
:T21
/basemodel/batch_normalization_3/batchnorm/Rsqrtю
<basemodel/batch_normalization_3/batchnorm/mul/ReadVariableOpReadVariableOpEbasemodel_batch_normalization_3_batchnorm_mul_readvariableop_resource*
_output_shapes
:T*
dtype02>
<basemodel/batch_normalization_3/batchnorm/mul/ReadVariableOpЕ
-basemodel/batch_normalization_3/batchnorm/mulMul3basemodel/batch_normalization_3/batchnorm/Rsqrt:y:0Dbasemodel/batch_normalization_3/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:T2/
-basemodel/batch_normalization_3/batchnorm/mulт
/basemodel/batch_normalization_3/batchnorm/mul_1Mul"basemodel/dense_1/BiasAdd:output:01basemodel/batch_normalization_3/batchnorm/mul:z:0*
T0*'
_output_shapes
:€€€€€€€€€T21
/basemodel/batch_normalization_3/batchnorm/mul_1ш
:basemodel/batch_normalization_3/batchnorm/ReadVariableOp_1ReadVariableOpCbasemodel_batch_normalization_3_batchnorm_readvariableop_1_resource*
_output_shapes
:T*
dtype02<
:basemodel/batch_normalization_3/batchnorm/ReadVariableOp_1Е
/basemodel/batch_normalization_3/batchnorm/mul_2MulBbasemodel/batch_normalization_3/batchnorm/ReadVariableOp_1:value:01basemodel/batch_normalization_3/batchnorm/mul:z:0*
T0*
_output_shapes
:T21
/basemodel/batch_normalization_3/batchnorm/mul_2ш
:basemodel/batch_normalization_3/batchnorm/ReadVariableOp_2ReadVariableOpCbasemodel_batch_normalization_3_batchnorm_readvariableop_2_resource*
_output_shapes
:T*
dtype02<
:basemodel/batch_normalization_3/batchnorm/ReadVariableOp_2Г
-basemodel/batch_normalization_3/batchnorm/subSubBbasemodel/batch_normalization_3/batchnorm/ReadVariableOp_2:value:03basemodel/batch_normalization_3/batchnorm/mul_2:z:0*
T0*
_output_shapes
:T2/
-basemodel/batch_normalization_3/batchnorm/subЕ
/basemodel/batch_normalization_3/batchnorm/add_1AddV23basemodel/batch_normalization_3/batchnorm/mul_1:z:01basemodel/batch_normalization_3/batchnorm/sub:z:0*
T0*'
_output_shapes
:€€€€€€€€€T21
/basemodel/batch_normalization_3/batchnorm/add_1Њ
$basemodel/dense_activation_1/SigmoidSigmoid3basemodel/batch_normalization_3/batchnorm/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€T2&
$basemodel/dense_activation_1/Sigmoid†
(basemodel/stream_2_input_drop/Identity_1Identityinputs_1*
T0*+
_output_shapes
:€€€€€€€€€}2*
(basemodel/stream_2_input_drop/Identity_1†
(basemodel/stream_1_input_drop/Identity_1Identityinputs_1*
T0*+
_output_shapes
:€€€€€€€€€}2*
(basemodel/stream_1_input_drop/Identity_1†
(basemodel/stream_0_input_drop/Identity_1Identityinputs_1*
T0*+
_output_shapes
:€€€€€€€€€}2*
(basemodel/stream_0_input_drop/Identity_1±
1basemodel/stream_2_conv_1/conv1d_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€23
1basemodel/stream_2_conv_1/conv1d_1/ExpandDims/dimХ
-basemodel/stream_2_conv_1/conv1d_1/ExpandDims
ExpandDims1basemodel/stream_2_input_drop/Identity_1:output:0:basemodel/stream_2_conv_1/conv1d_1/ExpandDims/dim:output:0*
T0*/
_output_shapes
:€€€€€€€€€}2/
-basemodel/stream_2_conv_1/conv1d_1/ExpandDimsК
>basemodel/stream_2_conv_1/conv1d_1/ExpandDims_1/ReadVariableOpReadVariableOpEbasemodel_stream_2_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02@
>basemodel/stream_2_conv_1/conv1d_1/ExpandDims_1/ReadVariableOpђ
3basemodel/stream_2_conv_1/conv1d_1/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 25
3basemodel/stream_2_conv_1/conv1d_1/ExpandDims_1/dimІ
/basemodel/stream_2_conv_1/conv1d_1/ExpandDims_1
ExpandDimsFbasemodel/stream_2_conv_1/conv1d_1/ExpandDims_1/ReadVariableOp:value:0<basemodel/stream_2_conv_1/conv1d_1/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@21
/basemodel/stream_2_conv_1/conv1d_1/ExpandDims_1¶
"basemodel/stream_2_conv_1/conv1d_1Conv2D6basemodel/stream_2_conv_1/conv1d_1/ExpandDims:output:08basemodel/stream_2_conv_1/conv1d_1/ExpandDims_1:output:0*
T0*/
_output_shapes
:€€€€€€€€€}@*
paddingSAME*
strides
2$
"basemodel/stream_2_conv_1/conv1d_1ж
*basemodel/stream_2_conv_1/conv1d_1/SqueezeSqueeze+basemodel/stream_2_conv_1/conv1d_1:output:0*
T0*+
_output_shapes
:€€€€€€€€€}@*
squeeze_dims

э€€€€€€€€2,
*basemodel/stream_2_conv_1/conv1d_1/Squeezeё
2basemodel/stream_2_conv_1/BiasAdd_1/ReadVariableOpReadVariableOp9basemodel_stream_2_conv_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype024
2basemodel/stream_2_conv_1/BiasAdd_1/ReadVariableOpь
#basemodel/stream_2_conv_1/BiasAdd_1BiasAdd3basemodel/stream_2_conv_1/conv1d_1/Squeeze:output:0:basemodel/stream_2_conv_1/BiasAdd_1/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€}@2%
#basemodel/stream_2_conv_1/BiasAdd_1±
1basemodel/stream_1_conv_1/conv1d_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€23
1basemodel/stream_1_conv_1/conv1d_1/ExpandDims/dimХ
-basemodel/stream_1_conv_1/conv1d_1/ExpandDims
ExpandDims1basemodel/stream_1_input_drop/Identity_1:output:0:basemodel/stream_1_conv_1/conv1d_1/ExpandDims/dim:output:0*
T0*/
_output_shapes
:€€€€€€€€€}2/
-basemodel/stream_1_conv_1/conv1d_1/ExpandDimsК
>basemodel/stream_1_conv_1/conv1d_1/ExpandDims_1/ReadVariableOpReadVariableOpEbasemodel_stream_1_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02@
>basemodel/stream_1_conv_1/conv1d_1/ExpandDims_1/ReadVariableOpђ
3basemodel/stream_1_conv_1/conv1d_1/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 25
3basemodel/stream_1_conv_1/conv1d_1/ExpandDims_1/dimІ
/basemodel/stream_1_conv_1/conv1d_1/ExpandDims_1
ExpandDimsFbasemodel/stream_1_conv_1/conv1d_1/ExpandDims_1/ReadVariableOp:value:0<basemodel/stream_1_conv_1/conv1d_1/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@21
/basemodel/stream_1_conv_1/conv1d_1/ExpandDims_1¶
"basemodel/stream_1_conv_1/conv1d_1Conv2D6basemodel/stream_1_conv_1/conv1d_1/ExpandDims:output:08basemodel/stream_1_conv_1/conv1d_1/ExpandDims_1:output:0*
T0*/
_output_shapes
:€€€€€€€€€}@*
paddingSAME*
strides
2$
"basemodel/stream_1_conv_1/conv1d_1ж
*basemodel/stream_1_conv_1/conv1d_1/SqueezeSqueeze+basemodel/stream_1_conv_1/conv1d_1:output:0*
T0*+
_output_shapes
:€€€€€€€€€}@*
squeeze_dims

э€€€€€€€€2,
*basemodel/stream_1_conv_1/conv1d_1/Squeezeё
2basemodel/stream_1_conv_1/BiasAdd_1/ReadVariableOpReadVariableOp9basemodel_stream_1_conv_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype024
2basemodel/stream_1_conv_1/BiasAdd_1/ReadVariableOpь
#basemodel/stream_1_conv_1/BiasAdd_1BiasAdd3basemodel/stream_1_conv_1/conv1d_1/Squeeze:output:0:basemodel/stream_1_conv_1/BiasAdd_1/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€}@2%
#basemodel/stream_1_conv_1/BiasAdd_1±
1basemodel/stream_0_conv_1/conv1d_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€23
1basemodel/stream_0_conv_1/conv1d_1/ExpandDims/dimХ
-basemodel/stream_0_conv_1/conv1d_1/ExpandDims
ExpandDims1basemodel/stream_0_input_drop/Identity_1:output:0:basemodel/stream_0_conv_1/conv1d_1/ExpandDims/dim:output:0*
T0*/
_output_shapes
:€€€€€€€€€}2/
-basemodel/stream_0_conv_1/conv1d_1/ExpandDimsК
>basemodel/stream_0_conv_1/conv1d_1/ExpandDims_1/ReadVariableOpReadVariableOpEbasemodel_stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02@
>basemodel/stream_0_conv_1/conv1d_1/ExpandDims_1/ReadVariableOpђ
3basemodel/stream_0_conv_1/conv1d_1/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 25
3basemodel/stream_0_conv_1/conv1d_1/ExpandDims_1/dimІ
/basemodel/stream_0_conv_1/conv1d_1/ExpandDims_1
ExpandDimsFbasemodel/stream_0_conv_1/conv1d_1/ExpandDims_1/ReadVariableOp:value:0<basemodel/stream_0_conv_1/conv1d_1/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@21
/basemodel/stream_0_conv_1/conv1d_1/ExpandDims_1¶
"basemodel/stream_0_conv_1/conv1d_1Conv2D6basemodel/stream_0_conv_1/conv1d_1/ExpandDims:output:08basemodel/stream_0_conv_1/conv1d_1/ExpandDims_1:output:0*
T0*/
_output_shapes
:€€€€€€€€€}@*
paddingSAME*
strides
2$
"basemodel/stream_0_conv_1/conv1d_1ж
*basemodel/stream_0_conv_1/conv1d_1/SqueezeSqueeze+basemodel/stream_0_conv_1/conv1d_1:output:0*
T0*+
_output_shapes
:€€€€€€€€€}@*
squeeze_dims

э€€€€€€€€2,
*basemodel/stream_0_conv_1/conv1d_1/Squeezeё
2basemodel/stream_0_conv_1/BiasAdd_1/ReadVariableOpReadVariableOp9basemodel_stream_0_conv_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype024
2basemodel/stream_0_conv_1/BiasAdd_1/ReadVariableOpь
#basemodel/stream_0_conv_1/BiasAdd_1BiasAdd3basemodel/stream_0_conv_1/conv1d_1/Squeeze:output:0:basemodel/stream_0_conv_1/BiasAdd_1/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€}@2%
#basemodel/stream_0_conv_1/BiasAdd_1ц
:basemodel/batch_normalization_2/batchnorm_1/ReadVariableOpReadVariableOpAbasemodel_batch_normalization_2_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02<
:basemodel/batch_normalization_2/batchnorm_1/ReadVariableOpЂ
1basemodel/batch_normalization_2/batchnorm_1/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:23
1basemodel/batch_normalization_2/batchnorm_1/add/yР
/basemodel/batch_normalization_2/batchnorm_1/addAddV2Bbasemodel/batch_normalization_2/batchnorm_1/ReadVariableOp:value:0:basemodel/batch_normalization_2/batchnorm_1/add/y:output:0*
T0*
_output_shapes
:@21
/basemodel/batch_normalization_2/batchnorm_1/add…
1basemodel/batch_normalization_2/batchnorm_1/RsqrtRsqrt3basemodel/batch_normalization_2/batchnorm_1/add:z:0*
T0*
_output_shapes
:@23
1basemodel/batch_normalization_2/batchnorm_1/RsqrtВ
>basemodel/batch_normalization_2/batchnorm_1/mul/ReadVariableOpReadVariableOpEbasemodel_batch_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02@
>basemodel/batch_normalization_2/batchnorm_1/mul/ReadVariableOpН
/basemodel/batch_normalization_2/batchnorm_1/mulMul5basemodel/batch_normalization_2/batchnorm_1/Rsqrt:y:0Fbasemodel/batch_normalization_2/batchnorm_1/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@21
/basemodel/batch_normalization_2/batchnorm_1/mulЖ
1basemodel/batch_normalization_2/batchnorm_1/mul_1Mul,basemodel/stream_2_conv_1/BiasAdd_1:output:03basemodel/batch_normalization_2/batchnorm_1/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€}@23
1basemodel/batch_normalization_2/batchnorm_1/mul_1ь
<basemodel/batch_normalization_2/batchnorm_1/ReadVariableOp_1ReadVariableOpCbasemodel_batch_normalization_2_batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype02>
<basemodel/batch_normalization_2/batchnorm_1/ReadVariableOp_1Н
1basemodel/batch_normalization_2/batchnorm_1/mul_2MulDbasemodel/batch_normalization_2/batchnorm_1/ReadVariableOp_1:value:03basemodel/batch_normalization_2/batchnorm_1/mul:z:0*
T0*
_output_shapes
:@23
1basemodel/batch_normalization_2/batchnorm_1/mul_2ь
<basemodel/batch_normalization_2/batchnorm_1/ReadVariableOp_2ReadVariableOpCbasemodel_batch_normalization_2_batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype02>
<basemodel/batch_normalization_2/batchnorm_1/ReadVariableOp_2Л
/basemodel/batch_normalization_2/batchnorm_1/subSubDbasemodel/batch_normalization_2/batchnorm_1/ReadVariableOp_2:value:05basemodel/batch_normalization_2/batchnorm_1/mul_2:z:0*
T0*
_output_shapes
:@21
/basemodel/batch_normalization_2/batchnorm_1/subС
1basemodel/batch_normalization_2/batchnorm_1/add_1AddV25basemodel/batch_normalization_2/batchnorm_1/mul_1:z:03basemodel/batch_normalization_2/batchnorm_1/sub:z:0*
T0*+
_output_shapes
:€€€€€€€€€}@23
1basemodel/batch_normalization_2/batchnorm_1/add_1ц
:basemodel/batch_normalization_1/batchnorm_1/ReadVariableOpReadVariableOpAbasemodel_batch_normalization_1_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02<
:basemodel/batch_normalization_1/batchnorm_1/ReadVariableOpЂ
1basemodel/batch_normalization_1/batchnorm_1/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:23
1basemodel/batch_normalization_1/batchnorm_1/add/yР
/basemodel/batch_normalization_1/batchnorm_1/addAddV2Bbasemodel/batch_normalization_1/batchnorm_1/ReadVariableOp:value:0:basemodel/batch_normalization_1/batchnorm_1/add/y:output:0*
T0*
_output_shapes
:@21
/basemodel/batch_normalization_1/batchnorm_1/add…
1basemodel/batch_normalization_1/batchnorm_1/RsqrtRsqrt3basemodel/batch_normalization_1/batchnorm_1/add:z:0*
T0*
_output_shapes
:@23
1basemodel/batch_normalization_1/batchnorm_1/RsqrtВ
>basemodel/batch_normalization_1/batchnorm_1/mul/ReadVariableOpReadVariableOpEbasemodel_batch_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02@
>basemodel/batch_normalization_1/batchnorm_1/mul/ReadVariableOpН
/basemodel/batch_normalization_1/batchnorm_1/mulMul5basemodel/batch_normalization_1/batchnorm_1/Rsqrt:y:0Fbasemodel/batch_normalization_1/batchnorm_1/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@21
/basemodel/batch_normalization_1/batchnorm_1/mulЖ
1basemodel/batch_normalization_1/batchnorm_1/mul_1Mul,basemodel/stream_1_conv_1/BiasAdd_1:output:03basemodel/batch_normalization_1/batchnorm_1/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€}@23
1basemodel/batch_normalization_1/batchnorm_1/mul_1ь
<basemodel/batch_normalization_1/batchnorm_1/ReadVariableOp_1ReadVariableOpCbasemodel_batch_normalization_1_batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype02>
<basemodel/batch_normalization_1/batchnorm_1/ReadVariableOp_1Н
1basemodel/batch_normalization_1/batchnorm_1/mul_2MulDbasemodel/batch_normalization_1/batchnorm_1/ReadVariableOp_1:value:03basemodel/batch_normalization_1/batchnorm_1/mul:z:0*
T0*
_output_shapes
:@23
1basemodel/batch_normalization_1/batchnorm_1/mul_2ь
<basemodel/batch_normalization_1/batchnorm_1/ReadVariableOp_2ReadVariableOpCbasemodel_batch_normalization_1_batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype02>
<basemodel/batch_normalization_1/batchnorm_1/ReadVariableOp_2Л
/basemodel/batch_normalization_1/batchnorm_1/subSubDbasemodel/batch_normalization_1/batchnorm_1/ReadVariableOp_2:value:05basemodel/batch_normalization_1/batchnorm_1/mul_2:z:0*
T0*
_output_shapes
:@21
/basemodel/batch_normalization_1/batchnorm_1/subС
1basemodel/batch_normalization_1/batchnorm_1/add_1AddV25basemodel/batch_normalization_1/batchnorm_1/mul_1:z:03basemodel/batch_normalization_1/batchnorm_1/sub:z:0*
T0*+
_output_shapes
:€€€€€€€€€}@23
1basemodel/batch_normalization_1/batchnorm_1/add_1р
8basemodel/batch_normalization/batchnorm_1/ReadVariableOpReadVariableOp?basemodel_batch_normalization_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02:
8basemodel/batch_normalization/batchnorm_1/ReadVariableOpІ
/basemodel/batch_normalization/batchnorm_1/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:21
/basemodel/batch_normalization/batchnorm_1/add/yИ
-basemodel/batch_normalization/batchnorm_1/addAddV2@basemodel/batch_normalization/batchnorm_1/ReadVariableOp:value:08basemodel/batch_normalization/batchnorm_1/add/y:output:0*
T0*
_output_shapes
:@2/
-basemodel/batch_normalization/batchnorm_1/add√
/basemodel/batch_normalization/batchnorm_1/RsqrtRsqrt1basemodel/batch_normalization/batchnorm_1/add:z:0*
T0*
_output_shapes
:@21
/basemodel/batch_normalization/batchnorm_1/Rsqrtь
<basemodel/batch_normalization/batchnorm_1/mul/ReadVariableOpReadVariableOpCbasemodel_batch_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02>
<basemodel/batch_normalization/batchnorm_1/mul/ReadVariableOpЕ
-basemodel/batch_normalization/batchnorm_1/mulMul3basemodel/batch_normalization/batchnorm_1/Rsqrt:y:0Dbasemodel/batch_normalization/batchnorm_1/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2/
-basemodel/batch_normalization/batchnorm_1/mulА
/basemodel/batch_normalization/batchnorm_1/mul_1Mul,basemodel/stream_0_conv_1/BiasAdd_1:output:01basemodel/batch_normalization/batchnorm_1/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€}@21
/basemodel/batch_normalization/batchnorm_1/mul_1ц
:basemodel/batch_normalization/batchnorm_1/ReadVariableOp_1ReadVariableOpAbasemodel_batch_normalization_batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype02<
:basemodel/batch_normalization/batchnorm_1/ReadVariableOp_1Е
/basemodel/batch_normalization/batchnorm_1/mul_2MulBbasemodel/batch_normalization/batchnorm_1/ReadVariableOp_1:value:01basemodel/batch_normalization/batchnorm_1/mul:z:0*
T0*
_output_shapes
:@21
/basemodel/batch_normalization/batchnorm_1/mul_2ц
:basemodel/batch_normalization/batchnorm_1/ReadVariableOp_2ReadVariableOpAbasemodel_batch_normalization_batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype02<
:basemodel/batch_normalization/batchnorm_1/ReadVariableOp_2Г
-basemodel/batch_normalization/batchnorm_1/subSubBbasemodel/batch_normalization/batchnorm_1/ReadVariableOp_2:value:03basemodel/batch_normalization/batchnorm_1/mul_2:z:0*
T0*
_output_shapes
:@2/
-basemodel/batch_normalization/batchnorm_1/subЙ
/basemodel/batch_normalization/batchnorm_1/add_1AddV23basemodel/batch_normalization/batchnorm_1/mul_1:z:01basemodel/batch_normalization/batchnorm_1/sub:z:0*
T0*+
_output_shapes
:€€€€€€€€€}@21
/basemodel/batch_normalization/batchnorm_1/add_1≥
basemodel/activation_2/Relu_1Relu5basemodel/batch_normalization_2/batchnorm_1/add_1:z:0*
T0*+
_output_shapes
:€€€€€€€€€}@2
basemodel/activation_2/Relu_1≥
basemodel/activation_1/Relu_1Relu5basemodel/batch_normalization_1/batchnorm_1/add_1:z:0*
T0*+
_output_shapes
:€€€€€€€€€}@2
basemodel/activation_1/Relu_1≠
basemodel/activation/Relu_1Relu3basemodel/batch_normalization/batchnorm_1/add_1:z:0*
T0*+
_output_shapes
:€€€€€€€€€}@2
basemodel/activation/Relu_1ї
$basemodel/stream_2_drop_1/Identity_1Identity+basemodel/activation_2/Relu_1:activations:0*
T0*+
_output_shapes
:€€€€€€€€€}@2&
$basemodel/stream_2_drop_1/Identity_1ї
$basemodel/stream_1_drop_1/Identity_1Identity+basemodel/activation_1/Relu_1:activations:0*
T0*+
_output_shapes
:€€€€€€€€€}@2&
$basemodel/stream_1_drop_1/Identity_1є
$basemodel/stream_0_drop_1/Identity_1Identity)basemodel/activation/Relu_1:activations:0*
T0*+
_output_shapes
:€€€€€€€€€}@2&
$basemodel/stream_0_drop_1/Identity_1Љ
;basemodel/global_average_pooling1d/Mean_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2=
;basemodel/global_average_pooling1d/Mean_1/reduction_indicesЕ
)basemodel/global_average_pooling1d/Mean_1Mean-basemodel/stream_0_drop_1/Identity_1:output:0Dbasemodel/global_average_pooling1d/Mean_1/reduction_indices:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2+
)basemodel/global_average_pooling1d/Mean_1ј
=basemodel/global_average_pooling1d_1/Mean_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2?
=basemodel/global_average_pooling1d_1/Mean_1/reduction_indicesЛ
+basemodel/global_average_pooling1d_1/Mean_1Mean-basemodel/stream_1_drop_1/Identity_1:output:0Fbasemodel/global_average_pooling1d_1/Mean_1/reduction_indices:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2-
+basemodel/global_average_pooling1d_1/Mean_1ј
=basemodel/global_average_pooling1d_2/Mean_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2?
=basemodel/global_average_pooling1d_2/Mean_1/reduction_indicesЛ
+basemodel/global_average_pooling1d_2/Mean_1Mean-basemodel/stream_2_drop_1/Identity_1:output:0Fbasemodel/global_average_pooling1d_2/Mean_1/reduction_indices:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2-
+basemodel/global_average_pooling1d_2/Mean_1М
#basemodel/concatenate/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :2%
#basemodel/concatenate/concat_1/axis÷
basemodel/concatenate/concat_1ConcatV22basemodel/global_average_pooling1d/Mean_1:output:04basemodel/global_average_pooling1d_1/Mean_1:output:04basemodel/global_average_pooling1d_2/Mean_1:output:0,basemodel/concatenate/concat_1/axis:output:0*
N*
T0*(
_output_shapes
:€€€€€€€€€ј2 
basemodel/concatenate/concat_1і
$basemodel/dense_1_dropout/Identity_1Identity'basemodel/concatenate/concat_1:output:0*
T0*(
_output_shapes
:€€€€€€€€€ј2&
$basemodel/dense_1_dropout/Identity_1»
)basemodel/dense_1/MatMul_1/ReadVariableOpReadVariableOp0basemodel_dense_1_matmul_readvariableop_resource*
_output_shapes
:	јT*
dtype02+
)basemodel/dense_1/MatMul_1/ReadVariableOp÷
basemodel/dense_1/MatMul_1MatMul-basemodel/dense_1_dropout/Identity_1:output:01basemodel/dense_1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€T2
basemodel/dense_1/MatMul_1∆
*basemodel/dense_1/BiasAdd_1/ReadVariableOpReadVariableOp1basemodel_dense_1_biasadd_readvariableop_resource*
_output_shapes
:T*
dtype02,
*basemodel/dense_1/BiasAdd_1/ReadVariableOp—
basemodel/dense_1/BiasAdd_1BiasAdd$basemodel/dense_1/MatMul_1:product:02basemodel/dense_1/BiasAdd_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€T2
basemodel/dense_1/BiasAdd_1ц
:basemodel/batch_normalization_3/batchnorm_1/ReadVariableOpReadVariableOpAbasemodel_batch_normalization_3_batchnorm_readvariableop_resource*
_output_shapes
:T*
dtype02<
:basemodel/batch_normalization_3/batchnorm_1/ReadVariableOpЂ
1basemodel/batch_normalization_3/batchnorm_1/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:23
1basemodel/batch_normalization_3/batchnorm_1/add/yР
/basemodel/batch_normalization_3/batchnorm_1/addAddV2Bbasemodel/batch_normalization_3/batchnorm_1/ReadVariableOp:value:0:basemodel/batch_normalization_3/batchnorm_1/add/y:output:0*
T0*
_output_shapes
:T21
/basemodel/batch_normalization_3/batchnorm_1/add…
1basemodel/batch_normalization_3/batchnorm_1/RsqrtRsqrt3basemodel/batch_normalization_3/batchnorm_1/add:z:0*
T0*
_output_shapes
:T23
1basemodel/batch_normalization_3/batchnorm_1/RsqrtВ
>basemodel/batch_normalization_3/batchnorm_1/mul/ReadVariableOpReadVariableOpEbasemodel_batch_normalization_3_batchnorm_mul_readvariableop_resource*
_output_shapes
:T*
dtype02@
>basemodel/batch_normalization_3/batchnorm_1/mul/ReadVariableOpН
/basemodel/batch_normalization_3/batchnorm_1/mulMul5basemodel/batch_normalization_3/batchnorm_1/Rsqrt:y:0Fbasemodel/batch_normalization_3/batchnorm_1/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:T21
/basemodel/batch_normalization_3/batchnorm_1/mulъ
1basemodel/batch_normalization_3/batchnorm_1/mul_1Mul$basemodel/dense_1/BiasAdd_1:output:03basemodel/batch_normalization_3/batchnorm_1/mul:z:0*
T0*'
_output_shapes
:€€€€€€€€€T23
1basemodel/batch_normalization_3/batchnorm_1/mul_1ь
<basemodel/batch_normalization_3/batchnorm_1/ReadVariableOp_1ReadVariableOpCbasemodel_batch_normalization_3_batchnorm_readvariableop_1_resource*
_output_shapes
:T*
dtype02>
<basemodel/batch_normalization_3/batchnorm_1/ReadVariableOp_1Н
1basemodel/batch_normalization_3/batchnorm_1/mul_2MulDbasemodel/batch_normalization_3/batchnorm_1/ReadVariableOp_1:value:03basemodel/batch_normalization_3/batchnorm_1/mul:z:0*
T0*
_output_shapes
:T23
1basemodel/batch_normalization_3/batchnorm_1/mul_2ь
<basemodel/batch_normalization_3/batchnorm_1/ReadVariableOp_2ReadVariableOpCbasemodel_batch_normalization_3_batchnorm_readvariableop_2_resource*
_output_shapes
:T*
dtype02>
<basemodel/batch_normalization_3/batchnorm_1/ReadVariableOp_2Л
/basemodel/batch_normalization_3/batchnorm_1/subSubDbasemodel/batch_normalization_3/batchnorm_1/ReadVariableOp_2:value:05basemodel/batch_normalization_3/batchnorm_1/mul_2:z:0*
T0*
_output_shapes
:T21
/basemodel/batch_normalization_3/batchnorm_1/subН
1basemodel/batch_normalization_3/batchnorm_1/add_1AddV25basemodel/batch_normalization_3/batchnorm_1/mul_1:z:03basemodel/batch_normalization_3/batchnorm_1/sub:z:0*
T0*'
_output_shapes
:€€€€€€€€€T23
1basemodel/batch_normalization_3/batchnorm_1/add_1ƒ
&basemodel/dense_activation_1/Sigmoid_1Sigmoid5basemodel/batch_normalization_3/batchnorm_1/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€T2(
&basemodel/dense_activation_1/Sigmoid_1Ђ
distance/subSub(basemodel/dense_activation_1/Sigmoid:y:0*basemodel/dense_activation_1/Sigmoid_1:y:0*
T0*'
_output_shapes
:€€€€€€€€€T2
distance/subp
distance/SquareSquaredistance/sub:z:0*
T0*'
_output_shapes
:€€€€€€€€€T2
distance/SquareЛ
distance/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2 
distance/Sum/reduction_indices§
distance/SumSumdistance/Square:y:0'distance/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:€€€€€€€€€*
	keep_dims(2
distance/Sume
distance/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
distance/ConstС
distance/MaximumMaximumdistance/Sum:output:0distance/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€2
distance/Maximumn
distance/SqrtSqrtdistance/Maximum:z:0*
T0*'
_output_shapes
:€€€€€€€€€2
distance/Sqrtш
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpEbasemodel_stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype027
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp√
&stream_0_conv_1/kernel/Regularizer/AbsAbs=stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_0_conv_1/kernel/Regularizer/Abs©
(stream_0_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_0_conv_1/kernel/Regularizer/Const„
&stream_0_conv_1/kernel/Regularizer/SumSum*stream_0_conv_1/kernel/Regularizer/Abs:y:01stream_0_conv_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/SumЩ
(stream_0_conv_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2*
(stream_0_conv_1/kernel/Regularizer/mul/x№
&stream_0_conv_1/kernel/Regularizer/mulMul1stream_0_conv_1/kernel/Regularizer/mul/x:output:0/stream_0_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/mulю
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpEbasemodel_stream_1_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02:
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpѕ
)stream_1_conv_1/kernel/Regularizer/SquareSquare@stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2+
)stream_1_conv_1/kernel/Regularizer/Square©
(stream_1_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_1_conv_1/kernel/Regularizer/ConstЏ
&stream_1_conv_1/kernel/Regularizer/SumSum-stream_1_conv_1/kernel/Regularizer/Square:y:01stream_1_conv_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&stream_1_conv_1/kernel/Regularizer/SumЩ
(stream_1_conv_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2*
(stream_1_conv_1/kernel/Regularizer/mul/x№
&stream_1_conv_1/kernel/Regularizer/mulMul1stream_1_conv_1/kernel/Regularizer/mul/x:output:0/stream_1_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_1_conv_1/kernel/Regularizer/mulш
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpEbasemodel_stream_2_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype027
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp√
&stream_2_conv_1/kernel/Regularizer/AbsAbs=stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_2_conv_1/kernel/Regularizer/Abs©
(stream_2_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_2_conv_1/kernel/Regularizer/Const„
&stream_2_conv_1/kernel/Regularizer/SumSum*stream_2_conv_1/kernel/Regularizer/Abs:y:01stream_2_conv_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&stream_2_conv_1/kernel/Regularizer/SumЩ
(stream_2_conv_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2*
(stream_2_conv_1/kernel/Regularizer/mul/x№
&stream_2_conv_1/kernel/Regularizer/mulMul1stream_2_conv_1/kernel/Regularizer/mul/x:output:0/stream_2_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_2_conv_1/kernel/Regularizer/mul–
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp0basemodel_dense_1_matmul_readvariableop_resource*
_output_shapes
:	јT*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOp®
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	јT2 
dense_1/kernel/Regularizer/AbsХ
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_1/kernel/Regularizer/ConstЈ
dense_1/kernel/Regularizer/SumSum"dense_1/kernel/Regularizer/Abs:y:0)dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/SumЙ
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2"
 dense_1/kernel/Regularizer/mul/xЉ
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/mull
IdentityIdentitydistance/Sqrt:y:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identityњ
NoOpNoOp7^basemodel/batch_normalization/batchnorm/ReadVariableOp9^basemodel/batch_normalization/batchnorm/ReadVariableOp_19^basemodel/batch_normalization/batchnorm/ReadVariableOp_2;^basemodel/batch_normalization/batchnorm/mul/ReadVariableOp9^basemodel/batch_normalization/batchnorm_1/ReadVariableOp;^basemodel/batch_normalization/batchnorm_1/ReadVariableOp_1;^basemodel/batch_normalization/batchnorm_1/ReadVariableOp_2=^basemodel/batch_normalization/batchnorm_1/mul/ReadVariableOp9^basemodel/batch_normalization_1/batchnorm/ReadVariableOp;^basemodel/batch_normalization_1/batchnorm/ReadVariableOp_1;^basemodel/batch_normalization_1/batchnorm/ReadVariableOp_2=^basemodel/batch_normalization_1/batchnorm/mul/ReadVariableOp;^basemodel/batch_normalization_1/batchnorm_1/ReadVariableOp=^basemodel/batch_normalization_1/batchnorm_1/ReadVariableOp_1=^basemodel/batch_normalization_1/batchnorm_1/ReadVariableOp_2?^basemodel/batch_normalization_1/batchnorm_1/mul/ReadVariableOp9^basemodel/batch_normalization_2/batchnorm/ReadVariableOp;^basemodel/batch_normalization_2/batchnorm/ReadVariableOp_1;^basemodel/batch_normalization_2/batchnorm/ReadVariableOp_2=^basemodel/batch_normalization_2/batchnorm/mul/ReadVariableOp;^basemodel/batch_normalization_2/batchnorm_1/ReadVariableOp=^basemodel/batch_normalization_2/batchnorm_1/ReadVariableOp_1=^basemodel/batch_normalization_2/batchnorm_1/ReadVariableOp_2?^basemodel/batch_normalization_2/batchnorm_1/mul/ReadVariableOp9^basemodel/batch_normalization_3/batchnorm/ReadVariableOp;^basemodel/batch_normalization_3/batchnorm/ReadVariableOp_1;^basemodel/batch_normalization_3/batchnorm/ReadVariableOp_2=^basemodel/batch_normalization_3/batchnorm/mul/ReadVariableOp;^basemodel/batch_normalization_3/batchnorm_1/ReadVariableOp=^basemodel/batch_normalization_3/batchnorm_1/ReadVariableOp_1=^basemodel/batch_normalization_3/batchnorm_1/ReadVariableOp_2?^basemodel/batch_normalization_3/batchnorm_1/mul/ReadVariableOp)^basemodel/dense_1/BiasAdd/ReadVariableOp+^basemodel/dense_1/BiasAdd_1/ReadVariableOp(^basemodel/dense_1/MatMul/ReadVariableOp*^basemodel/dense_1/MatMul_1/ReadVariableOp1^basemodel/stream_0_conv_1/BiasAdd/ReadVariableOp3^basemodel/stream_0_conv_1/BiasAdd_1/ReadVariableOp=^basemodel/stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp?^basemodel/stream_0_conv_1/conv1d_1/ExpandDims_1/ReadVariableOp1^basemodel/stream_1_conv_1/BiasAdd/ReadVariableOp3^basemodel/stream_1_conv_1/BiasAdd_1/ReadVariableOp=^basemodel/stream_1_conv_1/conv1d/ExpandDims_1/ReadVariableOp?^basemodel/stream_1_conv_1/conv1d_1/ExpandDims_1/ReadVariableOp1^basemodel/stream_2_conv_1/BiasAdd/ReadVariableOp3^basemodel/stream_2_conv_1/BiasAdd_1/ReadVariableOp=^basemodel/stream_2_conv_1/conv1d/ExpandDims_1/ReadVariableOp?^basemodel/stream_2_conv_1/conv1d_1/ExpandDims_1/ReadVariableOp.^dense_1/kernel/Regularizer/Abs/ReadVariableOp6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp9^stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp6^stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*q
_input_shapes`
^:€€€€€€€€€}:€€€€€€€€€}: : : : : : : : : : : : : : : : : : : : : : : : 2p
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
<basemodel/batch_normalization_1/batchnorm_1/ReadVariableOp_2<basemodel/batch_normalization_1/batchnorm_1/ReadVariableOp_22А
>basemodel/batch_normalization_1/batchnorm_1/mul/ReadVariableOp>basemodel/batch_normalization_1/batchnorm_1/mul/ReadVariableOp2t
8basemodel/batch_normalization_2/batchnorm/ReadVariableOp8basemodel/batch_normalization_2/batchnorm/ReadVariableOp2x
:basemodel/batch_normalization_2/batchnorm/ReadVariableOp_1:basemodel/batch_normalization_2/batchnorm/ReadVariableOp_12x
:basemodel/batch_normalization_2/batchnorm/ReadVariableOp_2:basemodel/batch_normalization_2/batchnorm/ReadVariableOp_22|
<basemodel/batch_normalization_2/batchnorm/mul/ReadVariableOp<basemodel/batch_normalization_2/batchnorm/mul/ReadVariableOp2x
:basemodel/batch_normalization_2/batchnorm_1/ReadVariableOp:basemodel/batch_normalization_2/batchnorm_1/ReadVariableOp2|
<basemodel/batch_normalization_2/batchnorm_1/ReadVariableOp_1<basemodel/batch_normalization_2/batchnorm_1/ReadVariableOp_12|
<basemodel/batch_normalization_2/batchnorm_1/ReadVariableOp_2<basemodel/batch_normalization_2/batchnorm_1/ReadVariableOp_22А
>basemodel/batch_normalization_2/batchnorm_1/mul/ReadVariableOp>basemodel/batch_normalization_2/batchnorm_1/mul/ReadVariableOp2t
8basemodel/batch_normalization_3/batchnorm/ReadVariableOp8basemodel/batch_normalization_3/batchnorm/ReadVariableOp2x
:basemodel/batch_normalization_3/batchnorm/ReadVariableOp_1:basemodel/batch_normalization_3/batchnorm/ReadVariableOp_12x
:basemodel/batch_normalization_3/batchnorm/ReadVariableOp_2:basemodel/batch_normalization_3/batchnorm/ReadVariableOp_22|
<basemodel/batch_normalization_3/batchnorm/mul/ReadVariableOp<basemodel/batch_normalization_3/batchnorm/mul/ReadVariableOp2x
:basemodel/batch_normalization_3/batchnorm_1/ReadVariableOp:basemodel/batch_normalization_3/batchnorm_1/ReadVariableOp2|
<basemodel/batch_normalization_3/batchnorm_1/ReadVariableOp_1<basemodel/batch_normalization_3/batchnorm_1/ReadVariableOp_12|
<basemodel/batch_normalization_3/batchnorm_1/ReadVariableOp_2<basemodel/batch_normalization_3/batchnorm_1/ReadVariableOp_22А
>basemodel/batch_normalization_3/batchnorm_1/mul/ReadVariableOp>basemodel/batch_normalization_3/batchnorm_1/mul/ReadVariableOp2T
(basemodel/dense_1/BiasAdd/ReadVariableOp(basemodel/dense_1/BiasAdd/ReadVariableOp2X
*basemodel/dense_1/BiasAdd_1/ReadVariableOp*basemodel/dense_1/BiasAdd_1/ReadVariableOp2R
'basemodel/dense_1/MatMul/ReadVariableOp'basemodel/dense_1/MatMul/ReadVariableOp2V
)basemodel/dense_1/MatMul_1/ReadVariableOp)basemodel/dense_1/MatMul_1/ReadVariableOp2d
0basemodel/stream_0_conv_1/BiasAdd/ReadVariableOp0basemodel/stream_0_conv_1/BiasAdd/ReadVariableOp2h
2basemodel/stream_0_conv_1/BiasAdd_1/ReadVariableOp2basemodel/stream_0_conv_1/BiasAdd_1/ReadVariableOp2|
<basemodel/stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp<basemodel/stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp2А
>basemodel/stream_0_conv_1/conv1d_1/ExpandDims_1/ReadVariableOp>basemodel/stream_0_conv_1/conv1d_1/ExpandDims_1/ReadVariableOp2d
0basemodel/stream_1_conv_1/BiasAdd/ReadVariableOp0basemodel/stream_1_conv_1/BiasAdd/ReadVariableOp2h
2basemodel/stream_1_conv_1/BiasAdd_1/ReadVariableOp2basemodel/stream_1_conv_1/BiasAdd_1/ReadVariableOp2|
<basemodel/stream_1_conv_1/conv1d/ExpandDims_1/ReadVariableOp<basemodel/stream_1_conv_1/conv1d/ExpandDims_1/ReadVariableOp2А
>basemodel/stream_1_conv_1/conv1d_1/ExpandDims_1/ReadVariableOp>basemodel/stream_1_conv_1/conv1d_1/ExpandDims_1/ReadVariableOp2d
0basemodel/stream_2_conv_1/BiasAdd/ReadVariableOp0basemodel/stream_2_conv_1/BiasAdd/ReadVariableOp2h
2basemodel/stream_2_conv_1/BiasAdd_1/ReadVariableOp2basemodel/stream_2_conv_1/BiasAdd_1/ReadVariableOp2|
<basemodel/stream_2_conv_1/conv1d/ExpandDims_1/ReadVariableOp<basemodel/stream_2_conv_1/conv1d/ExpandDims_1/ReadVariableOp2А
>basemodel/stream_2_conv_1/conv1d_1/ExpandDims_1/ReadVariableOp>basemodel/stream_2_conv_1/conv1d_1/ExpandDims_1/ReadVariableOp2^
-dense_1/kernel/Regularizer/Abs/ReadVariableOp-dense_1/kernel/Regularizer/Abs/ReadVariableOp2n
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp2t
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp2n
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp:U Q
+
_output_shapes
:€€€€€€€€€}
"
_user_specified_name
inputs/0:UQ
+
_output_shapes
:€€€€€€€€€}
"
_user_specified_name
inputs/1
Ё
L
0__inference_stream_2_drop_1_layer_call_fn_108469

inputs
identity–
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€}@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_stream_2_drop_1_layer_call_and_return_conditional_losses_1044952
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:€€€€€€€€€}@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€}@:S O
+
_output_shapes
:€€€€€€€€€}@
 
_user_specified_nameinputs
Ж
r
V__inference_global_average_pooling1d_2_layer_call_and_return_conditional_losses_108530

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
:€€€€€€€€€@2
Meana
IdentityIdentityMean:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€}@:S O
+
_output_shapes
:€€€€€€€€€}@
 
_user_specified_nameinputs
Ж
r
V__inference_global_average_pooling1d_1_layer_call_and_return_conditional_losses_108508

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
:€€€€€€€€€@2
Meana
IdentityIdentityMean:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€}@:S O
+
_output_shapes
:€€€€€€€€€}@
 
_user_specified_nameinputs
ґ
Б
*__inference_basemodel_layer_call_fn_104663
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

unknown_17:	јT

unknown_18:T

unknown_19:T

unknown_20:T

unknown_21:T

unknown_22:T
identityИҐStatefulPartitionedCallї
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
:€€€€€€€€€T*:
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_basemodel_layer_call_and_return_conditional_losses_1046122
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€T2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*И
_input_shapesw
u:€€€€€€€€€}:€€€€€€€€€}:€€€€€€€€€}: : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
+
_output_shapes
:€€€€€€€€€}
"
_user_specified_name
inputs_0:UQ
+
_output_shapes
:€€€€€€€€€}
"
_user_specified_name
inputs_1:UQ
+
_output_shapes
:€€€€€€€€€}
"
_user_specified_name
inputs_2
Б+
к
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_104973

inputs5
'assignmovingavg_readvariableop_resource:@7
)assignmovingavg_1_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@/
!batchnorm_readvariableop_resource:@
identityИҐAssignMovingAvgҐAssignMovingAvg/ReadVariableOpҐAssignMovingAvg_1Ґ AssignMovingAvg_1/ReadVariableOpҐbatchnorm/ReadVariableOpҐbatchnorm/mul/ReadVariableOpС
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indicesУ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(2
moments/meanА
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:@2
moments/StopGradient®
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*+
_output_shapes
:€€€€€€€€€}@2
moments/SquaredDifferenceЩ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indicesґ
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(2
moments/varianceБ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 2
moments/SqueezeЙ
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
„#<2
AssignMovingAvg/decay§
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOpШ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg/subП
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg/mulњ
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
„#<2
AssignMovingAvg_1/decay™
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOp†
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg_1/subЧ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg_1/mul…
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
 *oГ:2
batchnorm/add/yВ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2
batchnorm/mulz
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€}@2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:@2
batchnorm/mul_2Т
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOpБ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2
batchnorm/subЙ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*+
_output_shapes
:€€€€€€€€€}@2
batchnorm/add_1r
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€}@2

Identityт
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€}@: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:S O
+
_output_shapes
:€€€€€€€€€}@
 
_user_specified_nameinputs
е
P
4__inference_stream_0_input_drop_layer_call_fn_107716

inputs
identity‘
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€}* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_stream_0_input_drop_layer_call_and_return_conditional_losses_1042992
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:€€€€€€€€€}2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€}:S O
+
_output_shapes
:€€€€€€€€€}
 
_user_specified_nameinputs
МХ
∞
E__inference_basemodel_layer_call_and_return_conditional_losses_105450
inputs_0
inputs_1
inputs_2,
stream_2_conv_1_105357:@$
stream_2_conv_1_105359:@,
stream_1_conv_1_105362:@$
stream_1_conv_1_105364:@,
stream_0_conv_1_105367:@$
stream_0_conv_1_105369:@*
batch_normalization_2_105372:@*
batch_normalization_2_105374:@*
batch_normalization_2_105376:@*
batch_normalization_2_105378:@*
batch_normalization_1_105381:@*
batch_normalization_1_105383:@*
batch_normalization_1_105385:@*
batch_normalization_1_105387:@(
batch_normalization_105390:@(
batch_normalization_105392:@(
batch_normalization_105394:@(
batch_normalization_105396:@!
dense_1_105410:	јT
dense_1_105412:T*
batch_normalization_3_105415:T*
batch_normalization_3_105417:T*
batch_normalization_3_105419:T*
batch_normalization_3_105421:T
identityИҐ+batch_normalization/StatefulPartitionedCallҐ-batch_normalization_1/StatefulPartitionedCallҐ-batch_normalization_2/StatefulPartitionedCallҐ-batch_normalization_3/StatefulPartitionedCallҐdense_1/StatefulPartitionedCallҐ-dense_1/kernel/Regularizer/Abs/ReadVariableOpҐ'stream_0_conv_1/StatefulPartitionedCallҐ5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpҐ'stream_1_conv_1/StatefulPartitionedCallҐ8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpҐ'stream_2_conv_1/StatefulPartitionedCallҐ5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOpю
#stream_2_input_drop/PartitionedCallPartitionedCallinputs_2*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€}* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_stream_2_input_drop_layer_call_and_return_conditional_losses_1042852%
#stream_2_input_drop/PartitionedCallю
#stream_1_input_drop/PartitionedCallPartitionedCallinputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€}* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_stream_1_input_drop_layer_call_and_return_conditional_losses_1042922%
#stream_1_input_drop/PartitionedCallю
#stream_0_input_drop/PartitionedCallPartitionedCallinputs_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€}* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_stream_0_input_drop_layer_call_and_return_conditional_losses_1042992%
#stream_0_input_drop/PartitionedCallд
'stream_2_conv_1/StatefulPartitionedCallStatefulPartitionedCall,stream_2_input_drop/PartitionedCall:output:0stream_2_conv_1_105357stream_2_conv_1_105359*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€}@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_stream_2_conv_1_layer_call_and_return_conditional_losses_1043222)
'stream_2_conv_1/StatefulPartitionedCallд
'stream_1_conv_1/StatefulPartitionedCallStatefulPartitionedCall,stream_1_input_drop/PartitionedCall:output:0stream_1_conv_1_105362stream_1_conv_1_105364*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€}@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_stream_1_conv_1_layer_call_and_return_conditional_losses_1043492)
'stream_1_conv_1/StatefulPartitionedCallд
'stream_0_conv_1/StatefulPartitionedCallStatefulPartitionedCall,stream_0_input_drop/PartitionedCall:output:0stream_0_conv_1_105367stream_0_conv_1_105369*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€}@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_stream_0_conv_1_layer_call_and_return_conditional_losses_1043762)
'stream_0_conv_1/StatefulPartitionedCall∆
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall0stream_2_conv_1/StatefulPartitionedCall:output:0batch_normalization_2_105372batch_normalization_2_105374batch_normalization_2_105376batch_normalization_2_105378*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€}@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1044012/
-batch_normalization_2/StatefulPartitionedCall∆
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall0stream_1_conv_1/StatefulPartitionedCall:output:0batch_normalization_1_105381batch_normalization_1_105383batch_normalization_1_105385batch_normalization_1_105387*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€}@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1044302/
-batch_normalization_1/StatefulPartitionedCallЄ
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall0stream_0_conv_1/StatefulPartitionedCall:output:0batch_normalization_105390batch_normalization_105392batch_normalization_105394batch_normalization_105396*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€}@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_1044592-
+batch_normalization/StatefulPartitionedCallЧ
activation_2/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€}@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_activation_2_layer_call_and_return_conditional_losses_1044742
activation_2/PartitionedCallЧ
activation_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€}@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_activation_1_layer_call_and_return_conditional_losses_1044812
activation_1/PartitionedCallП
activation/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€}@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_activation_layer_call_and_return_conditional_losses_1044882
activation/PartitionedCallП
stream_2_drop_1/PartitionedCallPartitionedCall%activation_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€}@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_stream_2_drop_1_layer_call_and_return_conditional_losses_1044952!
stream_2_drop_1/PartitionedCallП
stream_1_drop_1/PartitionedCallPartitionedCall%activation_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€}@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_stream_1_drop_1_layer_call_and_return_conditional_losses_1045022!
stream_1_drop_1/PartitionedCallН
stream_0_drop_1/PartitionedCallPartitionedCall#activation/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€}@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_stream_0_drop_1_layer_call_and_return_conditional_losses_1045092!
stream_0_drop_1/PartitionedCall©
(global_average_pooling1d/PartitionedCallPartitionedCall(stream_0_drop_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *]
fXRV
T__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_1045162*
(global_average_pooling1d/PartitionedCallѓ
*global_average_pooling1d_1/PartitionedCallPartitionedCall(stream_1_drop_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *_
fZRX
V__inference_global_average_pooling1d_1_layer_call_and_return_conditional_losses_1045232,
*global_average_pooling1d_1/PartitionedCallѓ
*global_average_pooling1d_2/PartitionedCallPartitionedCall(stream_2_drop_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *_
fZRX
V__inference_global_average_pooling1d_2_layer_call_and_return_conditional_losses_1045302,
*global_average_pooling1d_2/PartitionedCallш
concatenate/PartitionedCallPartitionedCall1global_average_pooling1d/PartitionedCall:output:03global_average_pooling1d_1/PartitionedCall:output:03global_average_pooling1d_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€ј* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_concatenate_layer_call_and_return_conditional_losses_1045402
concatenate/PartitionedCallЛ
dense_1_dropout/PartitionedCallPartitionedCall$concatenate/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€ј* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_dense_1_dropout_layer_call_and_return_conditional_losses_1045472!
dense_1_dropout/PartitionedCallі
dense_1/StatefulPartitionedCallStatefulPartitionedCall(dense_1_dropout/PartitionedCall:output:0dense_1_105410dense_1_105412*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€T*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_1045652!
dense_1/StatefulPartitionedCallЇ
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0batch_normalization_3_105415batch_normalization_3_105417batch_normalization_3_105419batch_normalization_3_105421*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€T*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1041312/
-batch_normalization_3/StatefulPartitionedCall•
"dense_activation_1/PartitionedCallPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€T* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *W
fRRP
N__inference_dense_activation_1_layer_call_and_return_conditional_losses_1045852$
"dense_activation_1/PartitionedCall…
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpstream_0_conv_1_105367*"
_output_shapes
:@*
dtype027
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp√
&stream_0_conv_1/kernel/Regularizer/AbsAbs=stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_0_conv_1/kernel/Regularizer/Abs©
(stream_0_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_0_conv_1/kernel/Regularizer/Const„
&stream_0_conv_1/kernel/Regularizer/SumSum*stream_0_conv_1/kernel/Regularizer/Abs:y:01stream_0_conv_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/SumЩ
(stream_0_conv_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2*
(stream_0_conv_1/kernel/Regularizer/mul/x№
&stream_0_conv_1/kernel/Regularizer/mulMul1stream_0_conv_1/kernel/Regularizer/mul/x:output:0/stream_0_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/mulѕ
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpstream_1_conv_1_105362*"
_output_shapes
:@*
dtype02:
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpѕ
)stream_1_conv_1/kernel/Regularizer/SquareSquare@stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2+
)stream_1_conv_1/kernel/Regularizer/Square©
(stream_1_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_1_conv_1/kernel/Regularizer/ConstЏ
&stream_1_conv_1/kernel/Regularizer/SumSum-stream_1_conv_1/kernel/Regularizer/Square:y:01stream_1_conv_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&stream_1_conv_1/kernel/Regularizer/SumЩ
(stream_1_conv_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2*
(stream_1_conv_1/kernel/Regularizer/mul/x№
&stream_1_conv_1/kernel/Regularizer/mulMul1stream_1_conv_1/kernel/Regularizer/mul/x:output:0/stream_1_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_1_conv_1/kernel/Regularizer/mul…
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpstream_2_conv_1_105357*"
_output_shapes
:@*
dtype027
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp√
&stream_2_conv_1/kernel/Regularizer/AbsAbs=stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_2_conv_1/kernel/Regularizer/Abs©
(stream_2_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_2_conv_1/kernel/Regularizer/Const„
&stream_2_conv_1/kernel/Regularizer/SumSum*stream_2_conv_1/kernel/Regularizer/Abs:y:01stream_2_conv_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&stream_2_conv_1/kernel/Regularizer/SumЩ
(stream_2_conv_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2*
(stream_2_conv_1/kernel/Regularizer/mul/x№
&stream_2_conv_1/kernel/Regularizer/mulMul1stream_2_conv_1/kernel/Regularizer/mul/x:output:0/stream_2_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_2_conv_1/kernel/Regularizer/mulЃ
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_1_105410*
_output_shapes
:	јT*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOp®
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	јT2 
dense_1/kernel/Regularizer/AbsХ
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_1/kernel/Regularizer/ConstЈ
dense_1/kernel/Regularizer/SumSum"dense_1/kernel/Regularizer/Abs:y:0)dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/SumЙ
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2"
 dense_1/kernel/Regularizer/mul/xЉ
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/mulЖ
IdentityIdentity+dense_activation_1/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€T2

IdentityЗ
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall.^dense_1/kernel/Regularizer/Abs/ReadVariableOp(^stream_0_conv_1/StatefulPartitionedCall6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp(^stream_1_conv_1/StatefulPartitionedCall9^stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp(^stream_2_conv_1/StatefulPartitionedCall6^stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*И
_input_shapesw
u:€€€€€€€€€}:€€€€€€€€€}:€€€€€€€€€}: : : : : : : : : : : : : : : : : : : : : : : : 2Z
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
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp:U Q
+
_output_shapes
:€€€€€€€€€}
"
_user_specified_name
inputs_0:UQ
+
_output_shapes
:€€€€€€€€€}
"
_user_specified_name
inputs_1:UQ
+
_output_shapes
:€€€€€€€€€}
"
_user_specified_name
inputs_2
Ё
L
0__inference_stream_1_drop_1_layer_call_fn_108442

inputs
identity–
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€}@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_stream_1_drop_1_layer_call_and_return_conditional_losses_1045022
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:€€€€€€€€€}@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€}@:S O
+
_output_shapes
:€€€€€€€€€}@
 
_user_specified_nameinputs
”
O
3__inference_dense_activation_1_layer_call_fn_108695

inputs
identityѕ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€T* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *W
fRRP
N__inference_dense_activation_1_layer_call_and_return_conditional_losses_1045852
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:€€€€€€€€€T2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:€€€€€€€€€T:O K
'
_output_shapes
:€€€€€€€€€T
 
_user_specified_nameinputs
н
j
K__inference_stream_1_drop_1_layer_call_and_return_conditional_losses_104760

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †?2
dropout/Constw
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:€€€€€€€€€}@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape”
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:€€€€€€€€€}@*
dtype0*
seedЈ*
seed2Ј2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL>2
dropout/GreaterEqual/y¬
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:€€€€€€€€€}@2
dropout/GreaterEqualГ
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:€€€€€€€€€}@2
dropout/Cast~
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:€€€€€€€€€}@2
dropout/Mul_1i
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:€€€€€€€€€}@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€}@:S O
+
_output_shapes
:€€€€€€€€€}@
 
_user_specified_nameinputs
ч 
°/
A__inference_model_layer_call_and_return_conditional_losses_107015
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
0basemodel_dense_1_matmul_readvariableop_resource:	јT?
1basemodel_dense_1_biasadd_readvariableop_resource:TU
Gbasemodel_batch_normalization_3_assignmovingavg_readvariableop_resource:TW
Ibasemodel_batch_normalization_3_assignmovingavg_1_readvariableop_resource:TS
Ebasemodel_batch_normalization_3_batchnorm_mul_readvariableop_resource:TO
Abasemodel_batch_normalization_3_batchnorm_readvariableop_resource:T
identityИҐ-basemodel/batch_normalization/AssignMovingAvgҐ<basemodel/batch_normalization/AssignMovingAvg/ReadVariableOpҐ/basemodel/batch_normalization/AssignMovingAvg_1Ґ>basemodel/batch_normalization/AssignMovingAvg_1/ReadVariableOpҐ/basemodel/batch_normalization/AssignMovingAvg_2Ґ>basemodel/batch_normalization/AssignMovingAvg_2/ReadVariableOpҐ/basemodel/batch_normalization/AssignMovingAvg_3Ґ>basemodel/batch_normalization/AssignMovingAvg_3/ReadVariableOpҐ6basemodel/batch_normalization/batchnorm/ReadVariableOpҐ:basemodel/batch_normalization/batchnorm/mul/ReadVariableOpҐ8basemodel/batch_normalization/batchnorm_1/ReadVariableOpҐ<basemodel/batch_normalization/batchnorm_1/mul/ReadVariableOpҐ/basemodel/batch_normalization_1/AssignMovingAvgҐ>basemodel/batch_normalization_1/AssignMovingAvg/ReadVariableOpҐ1basemodel/batch_normalization_1/AssignMovingAvg_1Ґ@basemodel/batch_normalization_1/AssignMovingAvg_1/ReadVariableOpҐ1basemodel/batch_normalization_1/AssignMovingAvg_2Ґ@basemodel/batch_normalization_1/AssignMovingAvg_2/ReadVariableOpҐ1basemodel/batch_normalization_1/AssignMovingAvg_3Ґ@basemodel/batch_normalization_1/AssignMovingAvg_3/ReadVariableOpҐ8basemodel/batch_normalization_1/batchnorm/ReadVariableOpҐ<basemodel/batch_normalization_1/batchnorm/mul/ReadVariableOpҐ:basemodel/batch_normalization_1/batchnorm_1/ReadVariableOpҐ>basemodel/batch_normalization_1/batchnorm_1/mul/ReadVariableOpҐ/basemodel/batch_normalization_2/AssignMovingAvgҐ>basemodel/batch_normalization_2/AssignMovingAvg/ReadVariableOpҐ1basemodel/batch_normalization_2/AssignMovingAvg_1Ґ@basemodel/batch_normalization_2/AssignMovingAvg_1/ReadVariableOpҐ1basemodel/batch_normalization_2/AssignMovingAvg_2Ґ@basemodel/batch_normalization_2/AssignMovingAvg_2/ReadVariableOpҐ1basemodel/batch_normalization_2/AssignMovingAvg_3Ґ@basemodel/batch_normalization_2/AssignMovingAvg_3/ReadVariableOpҐ8basemodel/batch_normalization_2/batchnorm/ReadVariableOpҐ<basemodel/batch_normalization_2/batchnorm/mul/ReadVariableOpҐ:basemodel/batch_normalization_2/batchnorm_1/ReadVariableOpҐ>basemodel/batch_normalization_2/batchnorm_1/mul/ReadVariableOpҐ/basemodel/batch_normalization_3/AssignMovingAvgҐ>basemodel/batch_normalization_3/AssignMovingAvg/ReadVariableOpҐ1basemodel/batch_normalization_3/AssignMovingAvg_1Ґ@basemodel/batch_normalization_3/AssignMovingAvg_1/ReadVariableOpҐ1basemodel/batch_normalization_3/AssignMovingAvg_2Ґ@basemodel/batch_normalization_3/AssignMovingAvg_2/ReadVariableOpҐ1basemodel/batch_normalization_3/AssignMovingAvg_3Ґ@basemodel/batch_normalization_3/AssignMovingAvg_3/ReadVariableOpҐ8basemodel/batch_normalization_3/batchnorm/ReadVariableOpҐ<basemodel/batch_normalization_3/batchnorm/mul/ReadVariableOpҐ:basemodel/batch_normalization_3/batchnorm_1/ReadVariableOpҐ>basemodel/batch_normalization_3/batchnorm_1/mul/ReadVariableOpҐ(basemodel/dense_1/BiasAdd/ReadVariableOpҐ*basemodel/dense_1/BiasAdd_1/ReadVariableOpҐ'basemodel/dense_1/MatMul/ReadVariableOpҐ)basemodel/dense_1/MatMul_1/ReadVariableOpҐ0basemodel/stream_0_conv_1/BiasAdd/ReadVariableOpҐ2basemodel/stream_0_conv_1/BiasAdd_1/ReadVariableOpҐ<basemodel/stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOpҐ>basemodel/stream_0_conv_1/conv1d_1/ExpandDims_1/ReadVariableOpҐ0basemodel/stream_1_conv_1/BiasAdd/ReadVariableOpҐ2basemodel/stream_1_conv_1/BiasAdd_1/ReadVariableOpҐ<basemodel/stream_1_conv_1/conv1d/ExpandDims_1/ReadVariableOpҐ>basemodel/stream_1_conv_1/conv1d_1/ExpandDims_1/ReadVariableOpҐ0basemodel/stream_2_conv_1/BiasAdd/ReadVariableOpҐ2basemodel/stream_2_conv_1/BiasAdd_1/ReadVariableOpҐ<basemodel/stream_2_conv_1/conv1d/ExpandDims_1/ReadVariableOpҐ>basemodel/stream_2_conv_1/conv1d_1/ExpandDims_1/ReadVariableOpҐ-dense_1/kernel/Regularizer/Abs/ReadVariableOpҐ5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpҐ8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpҐ5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOpЯ
+basemodel/stream_2_input_drop/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †?2-
+basemodel/stream_2_input_drop/dropout/Const”
)basemodel/stream_2_input_drop/dropout/MulMulinputs_04basemodel/stream_2_input_drop/dropout/Const:output:0*
T0*+
_output_shapes
:€€€€€€€€€}2+
)basemodel/stream_2_input_drop/dropout/MulТ
+basemodel/stream_2_input_drop/dropout/ShapeShapeinputs_0*
T0*
_output_shapes
:2-
+basemodel/stream_2_input_drop/dropout/Shape≠
Bbasemodel/stream_2_input_drop/dropout/random_uniform/RandomUniformRandomUniform4basemodel/stream_2_input_drop/dropout/Shape:output:0*
T0*+
_output_shapes
:€€€€€€€€€}*
dtype0*
seedЈ*
seed2Ј2D
Bbasemodel/stream_2_input_drop/dropout/random_uniform/RandomUniform±
4basemodel/stream_2_input_drop/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL>26
4basemodel/stream_2_input_drop/dropout/GreaterEqual/yЇ
2basemodel/stream_2_input_drop/dropout/GreaterEqualGreaterEqualKbasemodel/stream_2_input_drop/dropout/random_uniform/RandomUniform:output:0=basemodel/stream_2_input_drop/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:€€€€€€€€€}24
2basemodel/stream_2_input_drop/dropout/GreaterEqualЁ
*basemodel/stream_2_input_drop/dropout/CastCast6basemodel/stream_2_input_drop/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:€€€€€€€€€}2,
*basemodel/stream_2_input_drop/dropout/Castц
+basemodel/stream_2_input_drop/dropout/Mul_1Mul-basemodel/stream_2_input_drop/dropout/Mul:z:0.basemodel/stream_2_input_drop/dropout/Cast:y:0*
T0*+
_output_shapes
:€€€€€€€€€}2-
+basemodel/stream_2_input_drop/dropout/Mul_1Я
+basemodel/stream_1_input_drop/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †?2-
+basemodel/stream_1_input_drop/dropout/Const”
)basemodel/stream_1_input_drop/dropout/MulMulinputs_04basemodel/stream_1_input_drop/dropout/Const:output:0*
T0*+
_output_shapes
:€€€€€€€€€}2+
)basemodel/stream_1_input_drop/dropout/MulТ
+basemodel/stream_1_input_drop/dropout/ShapeShapeinputs_0*
T0*
_output_shapes
:2-
+basemodel/stream_1_input_drop/dropout/Shape≠
Bbasemodel/stream_1_input_drop/dropout/random_uniform/RandomUniformRandomUniform4basemodel/stream_1_input_drop/dropout/Shape:output:0*
T0*+
_output_shapes
:€€€€€€€€€}*
dtype0*
seedЈ*
seed2Ј2D
Bbasemodel/stream_1_input_drop/dropout/random_uniform/RandomUniform±
4basemodel/stream_1_input_drop/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL>26
4basemodel/stream_1_input_drop/dropout/GreaterEqual/yЇ
2basemodel/stream_1_input_drop/dropout/GreaterEqualGreaterEqualKbasemodel/stream_1_input_drop/dropout/random_uniform/RandomUniform:output:0=basemodel/stream_1_input_drop/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:€€€€€€€€€}24
2basemodel/stream_1_input_drop/dropout/GreaterEqualЁ
*basemodel/stream_1_input_drop/dropout/CastCast6basemodel/stream_1_input_drop/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:€€€€€€€€€}2,
*basemodel/stream_1_input_drop/dropout/Castц
+basemodel/stream_1_input_drop/dropout/Mul_1Mul-basemodel/stream_1_input_drop/dropout/Mul:z:0.basemodel/stream_1_input_drop/dropout/Cast:y:0*
T0*+
_output_shapes
:€€€€€€€€€}2-
+basemodel/stream_1_input_drop/dropout/Mul_1Я
+basemodel/stream_0_input_drop/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †?2-
+basemodel/stream_0_input_drop/dropout/Const”
)basemodel/stream_0_input_drop/dropout/MulMulinputs_04basemodel/stream_0_input_drop/dropout/Const:output:0*
T0*+
_output_shapes
:€€€€€€€€€}2+
)basemodel/stream_0_input_drop/dropout/MulТ
+basemodel/stream_0_input_drop/dropout/ShapeShapeinputs_0*
T0*
_output_shapes
:2-
+basemodel/stream_0_input_drop/dropout/Shape≠
Bbasemodel/stream_0_input_drop/dropout/random_uniform/RandomUniformRandomUniform4basemodel/stream_0_input_drop/dropout/Shape:output:0*
T0*+
_output_shapes
:€€€€€€€€€}*
dtype0*
seedЈ*
seed2Ј2D
Bbasemodel/stream_0_input_drop/dropout/random_uniform/RandomUniform±
4basemodel/stream_0_input_drop/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL>26
4basemodel/stream_0_input_drop/dropout/GreaterEqual/yЇ
2basemodel/stream_0_input_drop/dropout/GreaterEqualGreaterEqualKbasemodel/stream_0_input_drop/dropout/random_uniform/RandomUniform:output:0=basemodel/stream_0_input_drop/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:€€€€€€€€€}24
2basemodel/stream_0_input_drop/dropout/GreaterEqualЁ
*basemodel/stream_0_input_drop/dropout/CastCast6basemodel/stream_0_input_drop/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:€€€€€€€€€}2,
*basemodel/stream_0_input_drop/dropout/Castц
+basemodel/stream_0_input_drop/dropout/Mul_1Mul-basemodel/stream_0_input_drop/dropout/Mul:z:0.basemodel/stream_0_input_drop/dropout/Cast:y:0*
T0*+
_output_shapes
:€€€€€€€€€}2-
+basemodel/stream_0_input_drop/dropout/Mul_1≠
/basemodel/stream_2_conv_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€21
/basemodel/stream_2_conv_1/conv1d/ExpandDims/dimН
+basemodel/stream_2_conv_1/conv1d/ExpandDims
ExpandDims/basemodel/stream_2_input_drop/dropout/Mul_1:z:08basemodel/stream_2_conv_1/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:€€€€€€€€€}2-
+basemodel/stream_2_conv_1/conv1d/ExpandDimsЖ
<basemodel/stream_2_conv_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpEbasemodel_stream_2_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02>
<basemodel/stream_2_conv_1/conv1d/ExpandDims_1/ReadVariableOp®
1basemodel/stream_2_conv_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 23
1basemodel/stream_2_conv_1/conv1d/ExpandDims_1/dimЯ
-basemodel/stream_2_conv_1/conv1d/ExpandDims_1
ExpandDimsDbasemodel/stream_2_conv_1/conv1d/ExpandDims_1/ReadVariableOp:value:0:basemodel/stream_2_conv_1/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@2/
-basemodel/stream_2_conv_1/conv1d/ExpandDims_1Ю
 basemodel/stream_2_conv_1/conv1dConv2D4basemodel/stream_2_conv_1/conv1d/ExpandDims:output:06basemodel/stream_2_conv_1/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:€€€€€€€€€}@*
paddingSAME*
strides
2"
 basemodel/stream_2_conv_1/conv1dа
(basemodel/stream_2_conv_1/conv1d/SqueezeSqueeze)basemodel/stream_2_conv_1/conv1d:output:0*
T0*+
_output_shapes
:€€€€€€€€€}@*
squeeze_dims

э€€€€€€€€2*
(basemodel/stream_2_conv_1/conv1d/SqueezeЏ
0basemodel/stream_2_conv_1/BiasAdd/ReadVariableOpReadVariableOp9basemodel_stream_2_conv_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype022
0basemodel/stream_2_conv_1/BiasAdd/ReadVariableOpф
!basemodel/stream_2_conv_1/BiasAddBiasAdd1basemodel/stream_2_conv_1/conv1d/Squeeze:output:08basemodel/stream_2_conv_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€}@2#
!basemodel/stream_2_conv_1/BiasAdd≠
/basemodel/stream_1_conv_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€21
/basemodel/stream_1_conv_1/conv1d/ExpandDims/dimН
+basemodel/stream_1_conv_1/conv1d/ExpandDims
ExpandDims/basemodel/stream_1_input_drop/dropout/Mul_1:z:08basemodel/stream_1_conv_1/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:€€€€€€€€€}2-
+basemodel/stream_1_conv_1/conv1d/ExpandDimsЖ
<basemodel/stream_1_conv_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpEbasemodel_stream_1_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02>
<basemodel/stream_1_conv_1/conv1d/ExpandDims_1/ReadVariableOp®
1basemodel/stream_1_conv_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 23
1basemodel/stream_1_conv_1/conv1d/ExpandDims_1/dimЯ
-basemodel/stream_1_conv_1/conv1d/ExpandDims_1
ExpandDimsDbasemodel/stream_1_conv_1/conv1d/ExpandDims_1/ReadVariableOp:value:0:basemodel/stream_1_conv_1/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@2/
-basemodel/stream_1_conv_1/conv1d/ExpandDims_1Ю
 basemodel/stream_1_conv_1/conv1dConv2D4basemodel/stream_1_conv_1/conv1d/ExpandDims:output:06basemodel/stream_1_conv_1/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:€€€€€€€€€}@*
paddingSAME*
strides
2"
 basemodel/stream_1_conv_1/conv1dа
(basemodel/stream_1_conv_1/conv1d/SqueezeSqueeze)basemodel/stream_1_conv_1/conv1d:output:0*
T0*+
_output_shapes
:€€€€€€€€€}@*
squeeze_dims

э€€€€€€€€2*
(basemodel/stream_1_conv_1/conv1d/SqueezeЏ
0basemodel/stream_1_conv_1/BiasAdd/ReadVariableOpReadVariableOp9basemodel_stream_1_conv_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype022
0basemodel/stream_1_conv_1/BiasAdd/ReadVariableOpф
!basemodel/stream_1_conv_1/BiasAddBiasAdd1basemodel/stream_1_conv_1/conv1d/Squeeze:output:08basemodel/stream_1_conv_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€}@2#
!basemodel/stream_1_conv_1/BiasAdd≠
/basemodel/stream_0_conv_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€21
/basemodel/stream_0_conv_1/conv1d/ExpandDims/dimН
+basemodel/stream_0_conv_1/conv1d/ExpandDims
ExpandDims/basemodel/stream_0_input_drop/dropout/Mul_1:z:08basemodel/stream_0_conv_1/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:€€€€€€€€€}2-
+basemodel/stream_0_conv_1/conv1d/ExpandDimsЖ
<basemodel/stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpEbasemodel_stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02>
<basemodel/stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp®
1basemodel/stream_0_conv_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 23
1basemodel/stream_0_conv_1/conv1d/ExpandDims_1/dimЯ
-basemodel/stream_0_conv_1/conv1d/ExpandDims_1
ExpandDimsDbasemodel/stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp:value:0:basemodel/stream_0_conv_1/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@2/
-basemodel/stream_0_conv_1/conv1d/ExpandDims_1Ю
 basemodel/stream_0_conv_1/conv1dConv2D4basemodel/stream_0_conv_1/conv1d/ExpandDims:output:06basemodel/stream_0_conv_1/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:€€€€€€€€€}@*
paddingSAME*
strides
2"
 basemodel/stream_0_conv_1/conv1dа
(basemodel/stream_0_conv_1/conv1d/SqueezeSqueeze)basemodel/stream_0_conv_1/conv1d:output:0*
T0*+
_output_shapes
:€€€€€€€€€}@*
squeeze_dims

э€€€€€€€€2*
(basemodel/stream_0_conv_1/conv1d/SqueezeЏ
0basemodel/stream_0_conv_1/BiasAdd/ReadVariableOpReadVariableOp9basemodel_stream_0_conv_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype022
0basemodel/stream_0_conv_1/BiasAdd/ReadVariableOpф
!basemodel/stream_0_conv_1/BiasAddBiasAdd1basemodel/stream_0_conv_1/conv1d/Squeeze:output:08basemodel/stream_0_conv_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€}@2#
!basemodel/stream_0_conv_1/BiasAdd—
>basemodel/batch_normalization_2/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2@
>basemodel/batch_normalization_2/moments/mean/reduction_indicesЧ
,basemodel/batch_normalization_2/moments/meanMean*basemodel/stream_2_conv_1/BiasAdd:output:0Gbasemodel/batch_normalization_2/moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(2.
,basemodel/batch_normalization_2/moments/meanа
4basemodel/batch_normalization_2/moments/StopGradientStopGradient5basemodel/batch_normalization_2/moments/mean:output:0*
T0*"
_output_shapes
:@26
4basemodel/batch_normalization_2/moments/StopGradientђ
9basemodel/batch_normalization_2/moments/SquaredDifferenceSquaredDifference*basemodel/stream_2_conv_1/BiasAdd:output:0=basemodel/batch_normalization_2/moments/StopGradient:output:0*
T0*+
_output_shapes
:€€€€€€€€€}@2;
9basemodel/batch_normalization_2/moments/SquaredDifferenceў
Bbasemodel/batch_normalization_2/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2D
Bbasemodel/batch_normalization_2/moments/variance/reduction_indicesґ
0basemodel/batch_normalization_2/moments/varianceMean=basemodel/batch_normalization_2/moments/SquaredDifference:z:0Kbasemodel/batch_normalization_2/moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(22
0basemodel/batch_normalization_2/moments/varianceб
/basemodel/batch_normalization_2/moments/SqueezeSqueeze5basemodel/batch_normalization_2/moments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 21
/basemodel/batch_normalization_2/moments/Squeezeй
1basemodel/batch_normalization_2/moments/Squeeze_1Squeeze9basemodel/batch_normalization_2/moments/variance:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 23
1basemodel/batch_normalization_2/moments/Squeeze_1≥
5basemodel/batch_normalization_2/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<27
5basemodel/batch_normalization_2/AssignMovingAvg/decayД
>basemodel/batch_normalization_2/AssignMovingAvg/ReadVariableOpReadVariableOpGbasemodel_batch_normalization_2_assignmovingavg_readvariableop_resource*
_output_shapes
:@*
dtype02@
>basemodel/batch_normalization_2/AssignMovingAvg/ReadVariableOpШ
3basemodel/batch_normalization_2/AssignMovingAvg/subSubFbasemodel/batch_normalization_2/AssignMovingAvg/ReadVariableOp:value:08basemodel/batch_normalization_2/moments/Squeeze:output:0*
T0*
_output_shapes
:@25
3basemodel/batch_normalization_2/AssignMovingAvg/subП
3basemodel/batch_normalization_2/AssignMovingAvg/mulMul7basemodel/batch_normalization_2/AssignMovingAvg/sub:z:0>basemodel/batch_normalization_2/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:@25
3basemodel/batch_normalization_2/AssignMovingAvg/mulя
/basemodel/batch_normalization_2/AssignMovingAvgAssignSubVariableOpGbasemodel_batch_normalization_2_assignmovingavg_readvariableop_resource7basemodel/batch_normalization_2/AssignMovingAvg/mul:z:0?^basemodel/batch_normalization_2/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype021
/basemodel/batch_normalization_2/AssignMovingAvgЈ
7basemodel/batch_normalization_2/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<29
7basemodel/batch_normalization_2/AssignMovingAvg_1/decayК
@basemodel/batch_normalization_2/AssignMovingAvg_1/ReadVariableOpReadVariableOpIbasemodel_batch_normalization_2_assignmovingavg_1_readvariableop_resource*
_output_shapes
:@*
dtype02B
@basemodel/batch_normalization_2/AssignMovingAvg_1/ReadVariableOp†
5basemodel/batch_normalization_2/AssignMovingAvg_1/subSubHbasemodel/batch_normalization_2/AssignMovingAvg_1/ReadVariableOp:value:0:basemodel/batch_normalization_2/moments/Squeeze_1:output:0*
T0*
_output_shapes
:@27
5basemodel/batch_normalization_2/AssignMovingAvg_1/subЧ
5basemodel/batch_normalization_2/AssignMovingAvg_1/mulMul9basemodel/batch_normalization_2/AssignMovingAvg_1/sub:z:0@basemodel/batch_normalization_2/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:@27
5basemodel/batch_normalization_2/AssignMovingAvg_1/mulй
1basemodel/batch_normalization_2/AssignMovingAvg_1AssignSubVariableOpIbasemodel_batch_normalization_2_assignmovingavg_1_readvariableop_resource9basemodel/batch_normalization_2/AssignMovingAvg_1/mul:z:0A^basemodel/batch_normalization_2/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype023
1basemodel/batch_normalization_2/AssignMovingAvg_1І
/basemodel/batch_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:21
/basemodel/batch_normalization_2/batchnorm/add/yВ
-basemodel/batch_normalization_2/batchnorm/addAddV2:basemodel/batch_normalization_2/moments/Squeeze_1:output:08basemodel/batch_normalization_2/batchnorm/add/y:output:0*
T0*
_output_shapes
:@2/
-basemodel/batch_normalization_2/batchnorm/add√
/basemodel/batch_normalization_2/batchnorm/RsqrtRsqrt1basemodel/batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes
:@21
/basemodel/batch_normalization_2/batchnorm/Rsqrtю
<basemodel/batch_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOpEbasemodel_batch_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02>
<basemodel/batch_normalization_2/batchnorm/mul/ReadVariableOpЕ
-basemodel/batch_normalization_2/batchnorm/mulMul3basemodel/batch_normalization_2/batchnorm/Rsqrt:y:0Dbasemodel/batch_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2/
-basemodel/batch_normalization_2/batchnorm/mulю
/basemodel/batch_normalization_2/batchnorm/mul_1Mul*basemodel/stream_2_conv_1/BiasAdd:output:01basemodel/batch_normalization_2/batchnorm/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€}@21
/basemodel/batch_normalization_2/batchnorm/mul_1ы
/basemodel/batch_normalization_2/batchnorm/mul_2Mul8basemodel/batch_normalization_2/moments/Squeeze:output:01basemodel/batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes
:@21
/basemodel/batch_normalization_2/batchnorm/mul_2т
8basemodel/batch_normalization_2/batchnorm/ReadVariableOpReadVariableOpAbasemodel_batch_normalization_2_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02:
8basemodel/batch_normalization_2/batchnorm/ReadVariableOpБ
-basemodel/batch_normalization_2/batchnorm/subSub@basemodel/batch_normalization_2/batchnorm/ReadVariableOp:value:03basemodel/batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2/
-basemodel/batch_normalization_2/batchnorm/subЙ
/basemodel/batch_normalization_2/batchnorm/add_1AddV23basemodel/batch_normalization_2/batchnorm/mul_1:z:01basemodel/batch_normalization_2/batchnorm/sub:z:0*
T0*+
_output_shapes
:€€€€€€€€€}@21
/basemodel/batch_normalization_2/batchnorm/add_1—
>basemodel/batch_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2@
>basemodel/batch_normalization_1/moments/mean/reduction_indicesЧ
,basemodel/batch_normalization_1/moments/meanMean*basemodel/stream_1_conv_1/BiasAdd:output:0Gbasemodel/batch_normalization_1/moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(2.
,basemodel/batch_normalization_1/moments/meanа
4basemodel/batch_normalization_1/moments/StopGradientStopGradient5basemodel/batch_normalization_1/moments/mean:output:0*
T0*"
_output_shapes
:@26
4basemodel/batch_normalization_1/moments/StopGradientђ
9basemodel/batch_normalization_1/moments/SquaredDifferenceSquaredDifference*basemodel/stream_1_conv_1/BiasAdd:output:0=basemodel/batch_normalization_1/moments/StopGradient:output:0*
T0*+
_output_shapes
:€€€€€€€€€}@2;
9basemodel/batch_normalization_1/moments/SquaredDifferenceў
Bbasemodel/batch_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2D
Bbasemodel/batch_normalization_1/moments/variance/reduction_indicesґ
0basemodel/batch_normalization_1/moments/varianceMean=basemodel/batch_normalization_1/moments/SquaredDifference:z:0Kbasemodel/batch_normalization_1/moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(22
0basemodel/batch_normalization_1/moments/varianceб
/basemodel/batch_normalization_1/moments/SqueezeSqueeze5basemodel/batch_normalization_1/moments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 21
/basemodel/batch_normalization_1/moments/Squeezeй
1basemodel/batch_normalization_1/moments/Squeeze_1Squeeze9basemodel/batch_normalization_1/moments/variance:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 23
1basemodel/batch_normalization_1/moments/Squeeze_1≥
5basemodel/batch_normalization_1/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<27
5basemodel/batch_normalization_1/AssignMovingAvg/decayД
>basemodel/batch_normalization_1/AssignMovingAvg/ReadVariableOpReadVariableOpGbasemodel_batch_normalization_1_assignmovingavg_readvariableop_resource*
_output_shapes
:@*
dtype02@
>basemodel/batch_normalization_1/AssignMovingAvg/ReadVariableOpШ
3basemodel/batch_normalization_1/AssignMovingAvg/subSubFbasemodel/batch_normalization_1/AssignMovingAvg/ReadVariableOp:value:08basemodel/batch_normalization_1/moments/Squeeze:output:0*
T0*
_output_shapes
:@25
3basemodel/batch_normalization_1/AssignMovingAvg/subП
3basemodel/batch_normalization_1/AssignMovingAvg/mulMul7basemodel/batch_normalization_1/AssignMovingAvg/sub:z:0>basemodel/batch_normalization_1/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:@25
3basemodel/batch_normalization_1/AssignMovingAvg/mulя
/basemodel/batch_normalization_1/AssignMovingAvgAssignSubVariableOpGbasemodel_batch_normalization_1_assignmovingavg_readvariableop_resource7basemodel/batch_normalization_1/AssignMovingAvg/mul:z:0?^basemodel/batch_normalization_1/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype021
/basemodel/batch_normalization_1/AssignMovingAvgЈ
7basemodel/batch_normalization_1/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<29
7basemodel/batch_normalization_1/AssignMovingAvg_1/decayК
@basemodel/batch_normalization_1/AssignMovingAvg_1/ReadVariableOpReadVariableOpIbasemodel_batch_normalization_1_assignmovingavg_1_readvariableop_resource*
_output_shapes
:@*
dtype02B
@basemodel/batch_normalization_1/AssignMovingAvg_1/ReadVariableOp†
5basemodel/batch_normalization_1/AssignMovingAvg_1/subSubHbasemodel/batch_normalization_1/AssignMovingAvg_1/ReadVariableOp:value:0:basemodel/batch_normalization_1/moments/Squeeze_1:output:0*
T0*
_output_shapes
:@27
5basemodel/batch_normalization_1/AssignMovingAvg_1/subЧ
5basemodel/batch_normalization_1/AssignMovingAvg_1/mulMul9basemodel/batch_normalization_1/AssignMovingAvg_1/sub:z:0@basemodel/batch_normalization_1/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:@27
5basemodel/batch_normalization_1/AssignMovingAvg_1/mulй
1basemodel/batch_normalization_1/AssignMovingAvg_1AssignSubVariableOpIbasemodel_batch_normalization_1_assignmovingavg_1_readvariableop_resource9basemodel/batch_normalization_1/AssignMovingAvg_1/mul:z:0A^basemodel/batch_normalization_1/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype023
1basemodel/batch_normalization_1/AssignMovingAvg_1І
/basemodel/batch_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:21
/basemodel/batch_normalization_1/batchnorm/add/yВ
-basemodel/batch_normalization_1/batchnorm/addAddV2:basemodel/batch_normalization_1/moments/Squeeze_1:output:08basemodel/batch_normalization_1/batchnorm/add/y:output:0*
T0*
_output_shapes
:@2/
-basemodel/batch_normalization_1/batchnorm/add√
/basemodel/batch_normalization_1/batchnorm/RsqrtRsqrt1basemodel/batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes
:@21
/basemodel/batch_normalization_1/batchnorm/Rsqrtю
<basemodel/batch_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOpEbasemodel_batch_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02>
<basemodel/batch_normalization_1/batchnorm/mul/ReadVariableOpЕ
-basemodel/batch_normalization_1/batchnorm/mulMul3basemodel/batch_normalization_1/batchnorm/Rsqrt:y:0Dbasemodel/batch_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2/
-basemodel/batch_normalization_1/batchnorm/mulю
/basemodel/batch_normalization_1/batchnorm/mul_1Mul*basemodel/stream_1_conv_1/BiasAdd:output:01basemodel/batch_normalization_1/batchnorm/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€}@21
/basemodel/batch_normalization_1/batchnorm/mul_1ы
/basemodel/batch_normalization_1/batchnorm/mul_2Mul8basemodel/batch_normalization_1/moments/Squeeze:output:01basemodel/batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes
:@21
/basemodel/batch_normalization_1/batchnorm/mul_2т
8basemodel/batch_normalization_1/batchnorm/ReadVariableOpReadVariableOpAbasemodel_batch_normalization_1_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02:
8basemodel/batch_normalization_1/batchnorm/ReadVariableOpБ
-basemodel/batch_normalization_1/batchnorm/subSub@basemodel/batch_normalization_1/batchnorm/ReadVariableOp:value:03basemodel/batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2/
-basemodel/batch_normalization_1/batchnorm/subЙ
/basemodel/batch_normalization_1/batchnorm/add_1AddV23basemodel/batch_normalization_1/batchnorm/mul_1:z:01basemodel/batch_normalization_1/batchnorm/sub:z:0*
T0*+
_output_shapes
:€€€€€€€€€}@21
/basemodel/batch_normalization_1/batchnorm/add_1Ќ
<basemodel/batch_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2>
<basemodel/batch_normalization/moments/mean/reduction_indicesС
*basemodel/batch_normalization/moments/meanMean*basemodel/stream_0_conv_1/BiasAdd:output:0Ebasemodel/batch_normalization/moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(2,
*basemodel/batch_normalization/moments/meanЏ
2basemodel/batch_normalization/moments/StopGradientStopGradient3basemodel/batch_normalization/moments/mean:output:0*
T0*"
_output_shapes
:@24
2basemodel/batch_normalization/moments/StopGradient¶
7basemodel/batch_normalization/moments/SquaredDifferenceSquaredDifference*basemodel/stream_0_conv_1/BiasAdd:output:0;basemodel/batch_normalization/moments/StopGradient:output:0*
T0*+
_output_shapes
:€€€€€€€€€}@29
7basemodel/batch_normalization/moments/SquaredDifference’
@basemodel/batch_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2B
@basemodel/batch_normalization/moments/variance/reduction_indicesЃ
.basemodel/batch_normalization/moments/varianceMean;basemodel/batch_normalization/moments/SquaredDifference:z:0Ibasemodel/batch_normalization/moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(20
.basemodel/batch_normalization/moments/varianceџ
-basemodel/batch_normalization/moments/SqueezeSqueeze3basemodel/batch_normalization/moments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 2/
-basemodel/batch_normalization/moments/Squeezeг
/basemodel/batch_normalization/moments/Squeeze_1Squeeze7basemodel/batch_normalization/moments/variance:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 21
/basemodel/batch_normalization/moments/Squeeze_1ѓ
3basemodel/batch_normalization/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<25
3basemodel/batch_normalization/AssignMovingAvg/decayю
<basemodel/batch_normalization/AssignMovingAvg/ReadVariableOpReadVariableOpEbasemodel_batch_normalization_assignmovingavg_readvariableop_resource*
_output_shapes
:@*
dtype02>
<basemodel/batch_normalization/AssignMovingAvg/ReadVariableOpР
1basemodel/batch_normalization/AssignMovingAvg/subSubDbasemodel/batch_normalization/AssignMovingAvg/ReadVariableOp:value:06basemodel/batch_normalization/moments/Squeeze:output:0*
T0*
_output_shapes
:@23
1basemodel/batch_normalization/AssignMovingAvg/subЗ
1basemodel/batch_normalization/AssignMovingAvg/mulMul5basemodel/batch_normalization/AssignMovingAvg/sub:z:0<basemodel/batch_normalization/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:@23
1basemodel/batch_normalization/AssignMovingAvg/mul’
-basemodel/batch_normalization/AssignMovingAvgAssignSubVariableOpEbasemodel_batch_normalization_assignmovingavg_readvariableop_resource5basemodel/batch_normalization/AssignMovingAvg/mul:z:0=^basemodel/batch_normalization/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02/
-basemodel/batch_normalization/AssignMovingAvg≥
5basemodel/batch_normalization/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<27
5basemodel/batch_normalization/AssignMovingAvg_1/decayД
>basemodel/batch_normalization/AssignMovingAvg_1/ReadVariableOpReadVariableOpGbasemodel_batch_normalization_assignmovingavg_1_readvariableop_resource*
_output_shapes
:@*
dtype02@
>basemodel/batch_normalization/AssignMovingAvg_1/ReadVariableOpШ
3basemodel/batch_normalization/AssignMovingAvg_1/subSubFbasemodel/batch_normalization/AssignMovingAvg_1/ReadVariableOp:value:08basemodel/batch_normalization/moments/Squeeze_1:output:0*
T0*
_output_shapes
:@25
3basemodel/batch_normalization/AssignMovingAvg_1/subП
3basemodel/batch_normalization/AssignMovingAvg_1/mulMul7basemodel/batch_normalization/AssignMovingAvg_1/sub:z:0>basemodel/batch_normalization/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:@25
3basemodel/batch_normalization/AssignMovingAvg_1/mulя
/basemodel/batch_normalization/AssignMovingAvg_1AssignSubVariableOpGbasemodel_batch_normalization_assignmovingavg_1_readvariableop_resource7basemodel/batch_normalization/AssignMovingAvg_1/mul:z:0?^basemodel/batch_normalization/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype021
/basemodel/batch_normalization/AssignMovingAvg_1£
-basemodel/batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2/
-basemodel/batch_normalization/batchnorm/add/yъ
+basemodel/batch_normalization/batchnorm/addAddV28basemodel/batch_normalization/moments/Squeeze_1:output:06basemodel/batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes
:@2-
+basemodel/batch_normalization/batchnorm/addљ
-basemodel/batch_normalization/batchnorm/RsqrtRsqrt/basemodel/batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes
:@2/
-basemodel/batch_normalization/batchnorm/Rsqrtш
:basemodel/batch_normalization/batchnorm/mul/ReadVariableOpReadVariableOpCbasemodel_batch_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02<
:basemodel/batch_normalization/batchnorm/mul/ReadVariableOpэ
+basemodel/batch_normalization/batchnorm/mulMul1basemodel/batch_normalization/batchnorm/Rsqrt:y:0Bbasemodel/batch_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2-
+basemodel/batch_normalization/batchnorm/mulш
-basemodel/batch_normalization/batchnorm/mul_1Mul*basemodel/stream_0_conv_1/BiasAdd:output:0/basemodel/batch_normalization/batchnorm/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€}@2/
-basemodel/batch_normalization/batchnorm/mul_1у
-basemodel/batch_normalization/batchnorm/mul_2Mul6basemodel/batch_normalization/moments/Squeeze:output:0/basemodel/batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes
:@2/
-basemodel/batch_normalization/batchnorm/mul_2м
6basemodel/batch_normalization/batchnorm/ReadVariableOpReadVariableOp?basemodel_batch_normalization_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype028
6basemodel/batch_normalization/batchnorm/ReadVariableOpщ
+basemodel/batch_normalization/batchnorm/subSub>basemodel/batch_normalization/batchnorm/ReadVariableOp:value:01basemodel/batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2-
+basemodel/batch_normalization/batchnorm/subБ
-basemodel/batch_normalization/batchnorm/add_1AddV21basemodel/batch_normalization/batchnorm/mul_1:z:0/basemodel/batch_normalization/batchnorm/sub:z:0*
T0*+
_output_shapes
:€€€€€€€€€}@2/
-basemodel/batch_normalization/batchnorm/add_1≠
basemodel/activation_2/ReluRelu3basemodel/batch_normalization_2/batchnorm/add_1:z:0*
T0*+
_output_shapes
:€€€€€€€€€}@2
basemodel/activation_2/Relu≠
basemodel/activation_1/ReluRelu3basemodel/batch_normalization_1/batchnorm/add_1:z:0*
T0*+
_output_shapes
:€€€€€€€€€}@2
basemodel/activation_1/ReluІ
basemodel/activation/ReluRelu1basemodel/batch_normalization/batchnorm/add_1:z:0*
T0*+
_output_shapes
:€€€€€€€€€}@2
basemodel/activation/ReluЧ
'basemodel/stream_2_drop_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †?2)
'basemodel/stream_2_drop_1/dropout/Constи
%basemodel/stream_2_drop_1/dropout/MulMul)basemodel/activation_2/Relu:activations:00basemodel/stream_2_drop_1/dropout/Const:output:0*
T0*+
_output_shapes
:€€€€€€€€€}@2'
%basemodel/stream_2_drop_1/dropout/MulЂ
'basemodel/stream_2_drop_1/dropout/ShapeShape)basemodel/activation_2/Relu:activations:0*
T0*
_output_shapes
:2)
'basemodel/stream_2_drop_1/dropout/Shape°
>basemodel/stream_2_drop_1/dropout/random_uniform/RandomUniformRandomUniform0basemodel/stream_2_drop_1/dropout/Shape:output:0*
T0*+
_output_shapes
:€€€€€€€€€}@*
dtype0*
seedЈ*
seed2Ј2@
>basemodel/stream_2_drop_1/dropout/random_uniform/RandomUniform©
0basemodel/stream_2_drop_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL>22
0basemodel/stream_2_drop_1/dropout/GreaterEqual/y™
.basemodel/stream_2_drop_1/dropout/GreaterEqualGreaterEqualGbasemodel/stream_2_drop_1/dropout/random_uniform/RandomUniform:output:09basemodel/stream_2_drop_1/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:€€€€€€€€€}@20
.basemodel/stream_2_drop_1/dropout/GreaterEqual—
&basemodel/stream_2_drop_1/dropout/CastCast2basemodel/stream_2_drop_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:€€€€€€€€€}@2(
&basemodel/stream_2_drop_1/dropout/Castж
'basemodel/stream_2_drop_1/dropout/Mul_1Mul)basemodel/stream_2_drop_1/dropout/Mul:z:0*basemodel/stream_2_drop_1/dropout/Cast:y:0*
T0*+
_output_shapes
:€€€€€€€€€}@2)
'basemodel/stream_2_drop_1/dropout/Mul_1Ч
'basemodel/stream_1_drop_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †?2)
'basemodel/stream_1_drop_1/dropout/Constи
%basemodel/stream_1_drop_1/dropout/MulMul)basemodel/activation_1/Relu:activations:00basemodel/stream_1_drop_1/dropout/Const:output:0*
T0*+
_output_shapes
:€€€€€€€€€}@2'
%basemodel/stream_1_drop_1/dropout/MulЂ
'basemodel/stream_1_drop_1/dropout/ShapeShape)basemodel/activation_1/Relu:activations:0*
T0*
_output_shapes
:2)
'basemodel/stream_1_drop_1/dropout/Shape°
>basemodel/stream_1_drop_1/dropout/random_uniform/RandomUniformRandomUniform0basemodel/stream_1_drop_1/dropout/Shape:output:0*
T0*+
_output_shapes
:€€€€€€€€€}@*
dtype0*
seedЈ*
seed2Ј2@
>basemodel/stream_1_drop_1/dropout/random_uniform/RandomUniform©
0basemodel/stream_1_drop_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL>22
0basemodel/stream_1_drop_1/dropout/GreaterEqual/y™
.basemodel/stream_1_drop_1/dropout/GreaterEqualGreaterEqualGbasemodel/stream_1_drop_1/dropout/random_uniform/RandomUniform:output:09basemodel/stream_1_drop_1/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:€€€€€€€€€}@20
.basemodel/stream_1_drop_1/dropout/GreaterEqual—
&basemodel/stream_1_drop_1/dropout/CastCast2basemodel/stream_1_drop_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:€€€€€€€€€}@2(
&basemodel/stream_1_drop_1/dropout/Castж
'basemodel/stream_1_drop_1/dropout/Mul_1Mul)basemodel/stream_1_drop_1/dropout/Mul:z:0*basemodel/stream_1_drop_1/dropout/Cast:y:0*
T0*+
_output_shapes
:€€€€€€€€€}@2)
'basemodel/stream_1_drop_1/dropout/Mul_1Ч
'basemodel/stream_0_drop_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †?2)
'basemodel/stream_0_drop_1/dropout/Constж
%basemodel/stream_0_drop_1/dropout/MulMul'basemodel/activation/Relu:activations:00basemodel/stream_0_drop_1/dropout/Const:output:0*
T0*+
_output_shapes
:€€€€€€€€€}@2'
%basemodel/stream_0_drop_1/dropout/Mul©
'basemodel/stream_0_drop_1/dropout/ShapeShape'basemodel/activation/Relu:activations:0*
T0*
_output_shapes
:2)
'basemodel/stream_0_drop_1/dropout/Shape°
>basemodel/stream_0_drop_1/dropout/random_uniform/RandomUniformRandomUniform0basemodel/stream_0_drop_1/dropout/Shape:output:0*
T0*+
_output_shapes
:€€€€€€€€€}@*
dtype0*
seedЈ*
seed2Ј2@
>basemodel/stream_0_drop_1/dropout/random_uniform/RandomUniform©
0basemodel/stream_0_drop_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL>22
0basemodel/stream_0_drop_1/dropout/GreaterEqual/y™
.basemodel/stream_0_drop_1/dropout/GreaterEqualGreaterEqualGbasemodel/stream_0_drop_1/dropout/random_uniform/RandomUniform:output:09basemodel/stream_0_drop_1/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:€€€€€€€€€}@20
.basemodel/stream_0_drop_1/dropout/GreaterEqual—
&basemodel/stream_0_drop_1/dropout/CastCast2basemodel/stream_0_drop_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:€€€€€€€€€}@2(
&basemodel/stream_0_drop_1/dropout/Castж
'basemodel/stream_0_drop_1/dropout/Mul_1Mul)basemodel/stream_0_drop_1/dropout/Mul:z:0*basemodel/stream_0_drop_1/dropout/Cast:y:0*
T0*+
_output_shapes
:€€€€€€€€€}@2)
'basemodel/stream_0_drop_1/dropout/Mul_1Є
9basemodel/global_average_pooling1d/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2;
9basemodel/global_average_pooling1d/Mean/reduction_indicesэ
'basemodel/global_average_pooling1d/MeanMean+basemodel/stream_0_drop_1/dropout/Mul_1:z:0Bbasemodel/global_average_pooling1d/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2)
'basemodel/global_average_pooling1d/MeanЉ
;basemodel/global_average_pooling1d_1/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2=
;basemodel/global_average_pooling1d_1/Mean/reduction_indicesГ
)basemodel/global_average_pooling1d_1/MeanMean+basemodel/stream_1_drop_1/dropout/Mul_1:z:0Dbasemodel/global_average_pooling1d_1/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2+
)basemodel/global_average_pooling1d_1/MeanЉ
;basemodel/global_average_pooling1d_2/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2=
;basemodel/global_average_pooling1d_2/Mean/reduction_indicesГ
)basemodel/global_average_pooling1d_2/MeanMean+basemodel/stream_2_drop_1/dropout/Mul_1:z:0Dbasemodel/global_average_pooling1d_2/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2+
)basemodel/global_average_pooling1d_2/MeanИ
!basemodel/concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2#
!basemodel/concatenate/concat/axis 
basemodel/concatenate/concatConcatV20basemodel/global_average_pooling1d/Mean:output:02basemodel/global_average_pooling1d_1/Mean:output:02basemodel/global_average_pooling1d_2/Mean:output:0*basemodel/concatenate/concat/axis:output:0*
N*
T0*(
_output_shapes
:€€€€€€€€€ј2
basemodel/concatenate/concatƒ
'basemodel/dense_1/MatMul/ReadVariableOpReadVariableOp0basemodel_dense_1_matmul_readvariableop_resource*
_output_shapes
:	јT*
dtype02)
'basemodel/dense_1/MatMul/ReadVariableOp»
basemodel/dense_1/MatMulMatMul%basemodel/concatenate/concat:output:0/basemodel/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€T2
basemodel/dense_1/MatMul¬
(basemodel/dense_1/BiasAdd/ReadVariableOpReadVariableOp1basemodel_dense_1_biasadd_readvariableop_resource*
_output_shapes
:T*
dtype02*
(basemodel/dense_1/BiasAdd/ReadVariableOp…
basemodel/dense_1/BiasAddBiasAdd"basemodel/dense_1/MatMul:product:00basemodel/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€T2
basemodel/dense_1/BiasAdd 
>basemodel/batch_normalization_3/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2@
>basemodel/batch_normalization_3/moments/mean/reduction_indicesЛ
,basemodel/batch_normalization_3/moments/meanMean"basemodel/dense_1/BiasAdd:output:0Gbasemodel/batch_normalization_3/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:T*
	keep_dims(2.
,basemodel/batch_normalization_3/moments/mean№
4basemodel/batch_normalization_3/moments/StopGradientStopGradient5basemodel/batch_normalization_3/moments/mean:output:0*
T0*
_output_shapes

:T26
4basemodel/batch_normalization_3/moments/StopGradient†
9basemodel/batch_normalization_3/moments/SquaredDifferenceSquaredDifference"basemodel/dense_1/BiasAdd:output:0=basemodel/batch_normalization_3/moments/StopGradient:output:0*
T0*'
_output_shapes
:€€€€€€€€€T2;
9basemodel/batch_normalization_3/moments/SquaredDifference“
Bbasemodel/batch_normalization_3/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2D
Bbasemodel/batch_normalization_3/moments/variance/reduction_indices≤
0basemodel/batch_normalization_3/moments/varianceMean=basemodel/batch_normalization_3/moments/SquaredDifference:z:0Kbasemodel/batch_normalization_3/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:T*
	keep_dims(22
0basemodel/batch_normalization_3/moments/varianceа
/basemodel/batch_normalization_3/moments/SqueezeSqueeze5basemodel/batch_normalization_3/moments/mean:output:0*
T0*
_output_shapes
:T*
squeeze_dims
 21
/basemodel/batch_normalization_3/moments/Squeezeи
1basemodel/batch_normalization_3/moments/Squeeze_1Squeeze9basemodel/batch_normalization_3/moments/variance:output:0*
T0*
_output_shapes
:T*
squeeze_dims
 23
1basemodel/batch_normalization_3/moments/Squeeze_1≥
5basemodel/batch_normalization_3/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<27
5basemodel/batch_normalization_3/AssignMovingAvg/decayД
>basemodel/batch_normalization_3/AssignMovingAvg/ReadVariableOpReadVariableOpGbasemodel_batch_normalization_3_assignmovingavg_readvariableop_resource*
_output_shapes
:T*
dtype02@
>basemodel/batch_normalization_3/AssignMovingAvg/ReadVariableOpШ
3basemodel/batch_normalization_3/AssignMovingAvg/subSubFbasemodel/batch_normalization_3/AssignMovingAvg/ReadVariableOp:value:08basemodel/batch_normalization_3/moments/Squeeze:output:0*
T0*
_output_shapes
:T25
3basemodel/batch_normalization_3/AssignMovingAvg/subП
3basemodel/batch_normalization_3/AssignMovingAvg/mulMul7basemodel/batch_normalization_3/AssignMovingAvg/sub:z:0>basemodel/batch_normalization_3/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:T25
3basemodel/batch_normalization_3/AssignMovingAvg/mulя
/basemodel/batch_normalization_3/AssignMovingAvgAssignSubVariableOpGbasemodel_batch_normalization_3_assignmovingavg_readvariableop_resource7basemodel/batch_normalization_3/AssignMovingAvg/mul:z:0?^basemodel/batch_normalization_3/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype021
/basemodel/batch_normalization_3/AssignMovingAvgЈ
7basemodel/batch_normalization_3/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<29
7basemodel/batch_normalization_3/AssignMovingAvg_1/decayК
@basemodel/batch_normalization_3/AssignMovingAvg_1/ReadVariableOpReadVariableOpIbasemodel_batch_normalization_3_assignmovingavg_1_readvariableop_resource*
_output_shapes
:T*
dtype02B
@basemodel/batch_normalization_3/AssignMovingAvg_1/ReadVariableOp†
5basemodel/batch_normalization_3/AssignMovingAvg_1/subSubHbasemodel/batch_normalization_3/AssignMovingAvg_1/ReadVariableOp:value:0:basemodel/batch_normalization_3/moments/Squeeze_1:output:0*
T0*
_output_shapes
:T27
5basemodel/batch_normalization_3/AssignMovingAvg_1/subЧ
5basemodel/batch_normalization_3/AssignMovingAvg_1/mulMul9basemodel/batch_normalization_3/AssignMovingAvg_1/sub:z:0@basemodel/batch_normalization_3/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:T27
5basemodel/batch_normalization_3/AssignMovingAvg_1/mulй
1basemodel/batch_normalization_3/AssignMovingAvg_1AssignSubVariableOpIbasemodel_batch_normalization_3_assignmovingavg_1_readvariableop_resource9basemodel/batch_normalization_3/AssignMovingAvg_1/mul:z:0A^basemodel/batch_normalization_3/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype023
1basemodel/batch_normalization_3/AssignMovingAvg_1І
/basemodel/batch_normalization_3/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:21
/basemodel/batch_normalization_3/batchnorm/add/yВ
-basemodel/batch_normalization_3/batchnorm/addAddV2:basemodel/batch_normalization_3/moments/Squeeze_1:output:08basemodel/batch_normalization_3/batchnorm/add/y:output:0*
T0*
_output_shapes
:T2/
-basemodel/batch_normalization_3/batchnorm/add√
/basemodel/batch_normalization_3/batchnorm/RsqrtRsqrt1basemodel/batch_normalization_3/batchnorm/add:z:0*
T0*
_output_shapes
:T21
/basemodel/batch_normalization_3/batchnorm/Rsqrtю
<basemodel/batch_normalization_3/batchnorm/mul/ReadVariableOpReadVariableOpEbasemodel_batch_normalization_3_batchnorm_mul_readvariableop_resource*
_output_shapes
:T*
dtype02>
<basemodel/batch_normalization_3/batchnorm/mul/ReadVariableOpЕ
-basemodel/batch_normalization_3/batchnorm/mulMul3basemodel/batch_normalization_3/batchnorm/Rsqrt:y:0Dbasemodel/batch_normalization_3/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:T2/
-basemodel/batch_normalization_3/batchnorm/mulт
/basemodel/batch_normalization_3/batchnorm/mul_1Mul"basemodel/dense_1/BiasAdd:output:01basemodel/batch_normalization_3/batchnorm/mul:z:0*
T0*'
_output_shapes
:€€€€€€€€€T21
/basemodel/batch_normalization_3/batchnorm/mul_1ы
/basemodel/batch_normalization_3/batchnorm/mul_2Mul8basemodel/batch_normalization_3/moments/Squeeze:output:01basemodel/batch_normalization_3/batchnorm/mul:z:0*
T0*
_output_shapes
:T21
/basemodel/batch_normalization_3/batchnorm/mul_2т
8basemodel/batch_normalization_3/batchnorm/ReadVariableOpReadVariableOpAbasemodel_batch_normalization_3_batchnorm_readvariableop_resource*
_output_shapes
:T*
dtype02:
8basemodel/batch_normalization_3/batchnorm/ReadVariableOpБ
-basemodel/batch_normalization_3/batchnorm/subSub@basemodel/batch_normalization_3/batchnorm/ReadVariableOp:value:03basemodel/batch_normalization_3/batchnorm/mul_2:z:0*
T0*
_output_shapes
:T2/
-basemodel/batch_normalization_3/batchnorm/subЕ
/basemodel/batch_normalization_3/batchnorm/add_1AddV23basemodel/batch_normalization_3/batchnorm/mul_1:z:01basemodel/batch_normalization_3/batchnorm/sub:z:0*
T0*'
_output_shapes
:€€€€€€€€€T21
/basemodel/batch_normalization_3/batchnorm/add_1Њ
$basemodel/dense_activation_1/SigmoidSigmoid3basemodel/batch_normalization_3/batchnorm/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€T2&
$basemodel/dense_activation_1/Sigmoid£
-basemodel/stream_2_input_drop/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †?2/
-basemodel/stream_2_input_drop/dropout_1/Constў
+basemodel/stream_2_input_drop/dropout_1/MulMulinputs_16basemodel/stream_2_input_drop/dropout_1/Const:output:0*
T0*+
_output_shapes
:€€€€€€€€€}2-
+basemodel/stream_2_input_drop/dropout_1/MulЦ
-basemodel/stream_2_input_drop/dropout_1/ShapeShapeinputs_1*
T0*
_output_shapes
:2/
-basemodel/stream_2_input_drop/dropout_1/Shape≥
Dbasemodel/stream_2_input_drop/dropout_1/random_uniform/RandomUniformRandomUniform6basemodel/stream_2_input_drop/dropout_1/Shape:output:0*
T0*+
_output_shapes
:€€€€€€€€€}*
dtype0*
seedЈ*
seed2Ј2F
Dbasemodel/stream_2_input_drop/dropout_1/random_uniform/RandomUniformµ
6basemodel/stream_2_input_drop/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL>28
6basemodel/stream_2_input_drop/dropout_1/GreaterEqual/y¬
4basemodel/stream_2_input_drop/dropout_1/GreaterEqualGreaterEqualMbasemodel/stream_2_input_drop/dropout_1/random_uniform/RandomUniform:output:0?basemodel/stream_2_input_drop/dropout_1/GreaterEqual/y:output:0*
T0*+
_output_shapes
:€€€€€€€€€}26
4basemodel/stream_2_input_drop/dropout_1/GreaterEqualг
,basemodel/stream_2_input_drop/dropout_1/CastCast8basemodel/stream_2_input_drop/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:€€€€€€€€€}2.
,basemodel/stream_2_input_drop/dropout_1/Castю
-basemodel/stream_2_input_drop/dropout_1/Mul_1Mul/basemodel/stream_2_input_drop/dropout_1/Mul:z:00basemodel/stream_2_input_drop/dropout_1/Cast:y:0*
T0*+
_output_shapes
:€€€€€€€€€}2/
-basemodel/stream_2_input_drop/dropout_1/Mul_1£
-basemodel/stream_1_input_drop/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †?2/
-basemodel/stream_1_input_drop/dropout_1/Constў
+basemodel/stream_1_input_drop/dropout_1/MulMulinputs_16basemodel/stream_1_input_drop/dropout_1/Const:output:0*
T0*+
_output_shapes
:€€€€€€€€€}2-
+basemodel/stream_1_input_drop/dropout_1/MulЦ
-basemodel/stream_1_input_drop/dropout_1/ShapeShapeinputs_1*
T0*
_output_shapes
:2/
-basemodel/stream_1_input_drop/dropout_1/Shape≥
Dbasemodel/stream_1_input_drop/dropout_1/random_uniform/RandomUniformRandomUniform6basemodel/stream_1_input_drop/dropout_1/Shape:output:0*
T0*+
_output_shapes
:€€€€€€€€€}*
dtype0*
seedЈ*
seed2Ј2F
Dbasemodel/stream_1_input_drop/dropout_1/random_uniform/RandomUniformµ
6basemodel/stream_1_input_drop/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL>28
6basemodel/stream_1_input_drop/dropout_1/GreaterEqual/y¬
4basemodel/stream_1_input_drop/dropout_1/GreaterEqualGreaterEqualMbasemodel/stream_1_input_drop/dropout_1/random_uniform/RandomUniform:output:0?basemodel/stream_1_input_drop/dropout_1/GreaterEqual/y:output:0*
T0*+
_output_shapes
:€€€€€€€€€}26
4basemodel/stream_1_input_drop/dropout_1/GreaterEqualг
,basemodel/stream_1_input_drop/dropout_1/CastCast8basemodel/stream_1_input_drop/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:€€€€€€€€€}2.
,basemodel/stream_1_input_drop/dropout_1/Castю
-basemodel/stream_1_input_drop/dropout_1/Mul_1Mul/basemodel/stream_1_input_drop/dropout_1/Mul:z:00basemodel/stream_1_input_drop/dropout_1/Cast:y:0*
T0*+
_output_shapes
:€€€€€€€€€}2/
-basemodel/stream_1_input_drop/dropout_1/Mul_1£
-basemodel/stream_0_input_drop/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †?2/
-basemodel/stream_0_input_drop/dropout_1/Constў
+basemodel/stream_0_input_drop/dropout_1/MulMulinputs_16basemodel/stream_0_input_drop/dropout_1/Const:output:0*
T0*+
_output_shapes
:€€€€€€€€€}2-
+basemodel/stream_0_input_drop/dropout_1/MulЦ
-basemodel/stream_0_input_drop/dropout_1/ShapeShapeinputs_1*
T0*
_output_shapes
:2/
-basemodel/stream_0_input_drop/dropout_1/Shape≥
Dbasemodel/stream_0_input_drop/dropout_1/random_uniform/RandomUniformRandomUniform6basemodel/stream_0_input_drop/dropout_1/Shape:output:0*
T0*+
_output_shapes
:€€€€€€€€€}*
dtype0*
seedЈ*
seed2Ј2F
Dbasemodel/stream_0_input_drop/dropout_1/random_uniform/RandomUniformµ
6basemodel/stream_0_input_drop/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL>28
6basemodel/stream_0_input_drop/dropout_1/GreaterEqual/y¬
4basemodel/stream_0_input_drop/dropout_1/GreaterEqualGreaterEqualMbasemodel/stream_0_input_drop/dropout_1/random_uniform/RandomUniform:output:0?basemodel/stream_0_input_drop/dropout_1/GreaterEqual/y:output:0*
T0*+
_output_shapes
:€€€€€€€€€}26
4basemodel/stream_0_input_drop/dropout_1/GreaterEqualг
,basemodel/stream_0_input_drop/dropout_1/CastCast8basemodel/stream_0_input_drop/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:€€€€€€€€€}2.
,basemodel/stream_0_input_drop/dropout_1/Castю
-basemodel/stream_0_input_drop/dropout_1/Mul_1Mul/basemodel/stream_0_input_drop/dropout_1/Mul:z:00basemodel/stream_0_input_drop/dropout_1/Cast:y:0*
T0*+
_output_shapes
:€€€€€€€€€}2/
-basemodel/stream_0_input_drop/dropout_1/Mul_1±
1basemodel/stream_2_conv_1/conv1d_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€23
1basemodel/stream_2_conv_1/conv1d_1/ExpandDims/dimХ
-basemodel/stream_2_conv_1/conv1d_1/ExpandDims
ExpandDims1basemodel/stream_2_input_drop/dropout_1/Mul_1:z:0:basemodel/stream_2_conv_1/conv1d_1/ExpandDims/dim:output:0*
T0*/
_output_shapes
:€€€€€€€€€}2/
-basemodel/stream_2_conv_1/conv1d_1/ExpandDimsК
>basemodel/stream_2_conv_1/conv1d_1/ExpandDims_1/ReadVariableOpReadVariableOpEbasemodel_stream_2_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02@
>basemodel/stream_2_conv_1/conv1d_1/ExpandDims_1/ReadVariableOpђ
3basemodel/stream_2_conv_1/conv1d_1/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 25
3basemodel/stream_2_conv_1/conv1d_1/ExpandDims_1/dimІ
/basemodel/stream_2_conv_1/conv1d_1/ExpandDims_1
ExpandDimsFbasemodel/stream_2_conv_1/conv1d_1/ExpandDims_1/ReadVariableOp:value:0<basemodel/stream_2_conv_1/conv1d_1/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@21
/basemodel/stream_2_conv_1/conv1d_1/ExpandDims_1¶
"basemodel/stream_2_conv_1/conv1d_1Conv2D6basemodel/stream_2_conv_1/conv1d_1/ExpandDims:output:08basemodel/stream_2_conv_1/conv1d_1/ExpandDims_1:output:0*
T0*/
_output_shapes
:€€€€€€€€€}@*
paddingSAME*
strides
2$
"basemodel/stream_2_conv_1/conv1d_1ж
*basemodel/stream_2_conv_1/conv1d_1/SqueezeSqueeze+basemodel/stream_2_conv_1/conv1d_1:output:0*
T0*+
_output_shapes
:€€€€€€€€€}@*
squeeze_dims

э€€€€€€€€2,
*basemodel/stream_2_conv_1/conv1d_1/Squeezeё
2basemodel/stream_2_conv_1/BiasAdd_1/ReadVariableOpReadVariableOp9basemodel_stream_2_conv_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype024
2basemodel/stream_2_conv_1/BiasAdd_1/ReadVariableOpь
#basemodel/stream_2_conv_1/BiasAdd_1BiasAdd3basemodel/stream_2_conv_1/conv1d_1/Squeeze:output:0:basemodel/stream_2_conv_1/BiasAdd_1/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€}@2%
#basemodel/stream_2_conv_1/BiasAdd_1±
1basemodel/stream_1_conv_1/conv1d_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€23
1basemodel/stream_1_conv_1/conv1d_1/ExpandDims/dimХ
-basemodel/stream_1_conv_1/conv1d_1/ExpandDims
ExpandDims1basemodel/stream_1_input_drop/dropout_1/Mul_1:z:0:basemodel/stream_1_conv_1/conv1d_1/ExpandDims/dim:output:0*
T0*/
_output_shapes
:€€€€€€€€€}2/
-basemodel/stream_1_conv_1/conv1d_1/ExpandDimsК
>basemodel/stream_1_conv_1/conv1d_1/ExpandDims_1/ReadVariableOpReadVariableOpEbasemodel_stream_1_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02@
>basemodel/stream_1_conv_1/conv1d_1/ExpandDims_1/ReadVariableOpђ
3basemodel/stream_1_conv_1/conv1d_1/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 25
3basemodel/stream_1_conv_1/conv1d_1/ExpandDims_1/dimІ
/basemodel/stream_1_conv_1/conv1d_1/ExpandDims_1
ExpandDimsFbasemodel/stream_1_conv_1/conv1d_1/ExpandDims_1/ReadVariableOp:value:0<basemodel/stream_1_conv_1/conv1d_1/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@21
/basemodel/stream_1_conv_1/conv1d_1/ExpandDims_1¶
"basemodel/stream_1_conv_1/conv1d_1Conv2D6basemodel/stream_1_conv_1/conv1d_1/ExpandDims:output:08basemodel/stream_1_conv_1/conv1d_1/ExpandDims_1:output:0*
T0*/
_output_shapes
:€€€€€€€€€}@*
paddingSAME*
strides
2$
"basemodel/stream_1_conv_1/conv1d_1ж
*basemodel/stream_1_conv_1/conv1d_1/SqueezeSqueeze+basemodel/stream_1_conv_1/conv1d_1:output:0*
T0*+
_output_shapes
:€€€€€€€€€}@*
squeeze_dims

э€€€€€€€€2,
*basemodel/stream_1_conv_1/conv1d_1/Squeezeё
2basemodel/stream_1_conv_1/BiasAdd_1/ReadVariableOpReadVariableOp9basemodel_stream_1_conv_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype024
2basemodel/stream_1_conv_1/BiasAdd_1/ReadVariableOpь
#basemodel/stream_1_conv_1/BiasAdd_1BiasAdd3basemodel/stream_1_conv_1/conv1d_1/Squeeze:output:0:basemodel/stream_1_conv_1/BiasAdd_1/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€}@2%
#basemodel/stream_1_conv_1/BiasAdd_1±
1basemodel/stream_0_conv_1/conv1d_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€23
1basemodel/stream_0_conv_1/conv1d_1/ExpandDims/dimХ
-basemodel/stream_0_conv_1/conv1d_1/ExpandDims
ExpandDims1basemodel/stream_0_input_drop/dropout_1/Mul_1:z:0:basemodel/stream_0_conv_1/conv1d_1/ExpandDims/dim:output:0*
T0*/
_output_shapes
:€€€€€€€€€}2/
-basemodel/stream_0_conv_1/conv1d_1/ExpandDimsК
>basemodel/stream_0_conv_1/conv1d_1/ExpandDims_1/ReadVariableOpReadVariableOpEbasemodel_stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02@
>basemodel/stream_0_conv_1/conv1d_1/ExpandDims_1/ReadVariableOpђ
3basemodel/stream_0_conv_1/conv1d_1/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 25
3basemodel/stream_0_conv_1/conv1d_1/ExpandDims_1/dimІ
/basemodel/stream_0_conv_1/conv1d_1/ExpandDims_1
ExpandDimsFbasemodel/stream_0_conv_1/conv1d_1/ExpandDims_1/ReadVariableOp:value:0<basemodel/stream_0_conv_1/conv1d_1/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@21
/basemodel/stream_0_conv_1/conv1d_1/ExpandDims_1¶
"basemodel/stream_0_conv_1/conv1d_1Conv2D6basemodel/stream_0_conv_1/conv1d_1/ExpandDims:output:08basemodel/stream_0_conv_1/conv1d_1/ExpandDims_1:output:0*
T0*/
_output_shapes
:€€€€€€€€€}@*
paddingSAME*
strides
2$
"basemodel/stream_0_conv_1/conv1d_1ж
*basemodel/stream_0_conv_1/conv1d_1/SqueezeSqueeze+basemodel/stream_0_conv_1/conv1d_1:output:0*
T0*+
_output_shapes
:€€€€€€€€€}@*
squeeze_dims

э€€€€€€€€2,
*basemodel/stream_0_conv_1/conv1d_1/Squeezeё
2basemodel/stream_0_conv_1/BiasAdd_1/ReadVariableOpReadVariableOp9basemodel_stream_0_conv_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype024
2basemodel/stream_0_conv_1/BiasAdd_1/ReadVariableOpь
#basemodel/stream_0_conv_1/BiasAdd_1BiasAdd3basemodel/stream_0_conv_1/conv1d_1/Squeeze:output:0:basemodel/stream_0_conv_1/BiasAdd_1/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€}@2%
#basemodel/stream_0_conv_1/BiasAdd_1’
@basemodel/batch_normalization_2/moments_1/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2B
@basemodel/batch_normalization_2/moments_1/mean/reduction_indicesЯ
.basemodel/batch_normalization_2/moments_1/meanMean,basemodel/stream_2_conv_1/BiasAdd_1:output:0Ibasemodel/batch_normalization_2/moments_1/mean/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(20
.basemodel/batch_normalization_2/moments_1/meanж
6basemodel/batch_normalization_2/moments_1/StopGradientStopGradient7basemodel/batch_normalization_2/moments_1/mean:output:0*
T0*"
_output_shapes
:@28
6basemodel/batch_normalization_2/moments_1/StopGradientі
;basemodel/batch_normalization_2/moments_1/SquaredDifferenceSquaredDifference,basemodel/stream_2_conv_1/BiasAdd_1:output:0?basemodel/batch_normalization_2/moments_1/StopGradient:output:0*
T0*+
_output_shapes
:€€€€€€€€€}@2=
;basemodel/batch_normalization_2/moments_1/SquaredDifferenceЁ
Dbasemodel/batch_normalization_2/moments_1/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2F
Dbasemodel/batch_normalization_2/moments_1/variance/reduction_indicesЊ
2basemodel/batch_normalization_2/moments_1/varianceMean?basemodel/batch_normalization_2/moments_1/SquaredDifference:z:0Mbasemodel/batch_normalization_2/moments_1/variance/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(24
2basemodel/batch_normalization_2/moments_1/varianceз
1basemodel/batch_normalization_2/moments_1/SqueezeSqueeze7basemodel/batch_normalization_2/moments_1/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 23
1basemodel/batch_normalization_2/moments_1/Squeezeп
3basemodel/batch_normalization_2/moments_1/Squeeze_1Squeeze;basemodel/batch_normalization_2/moments_1/variance:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 25
3basemodel/batch_normalization_2/moments_1/Squeeze_1Ј
7basemodel/batch_normalization_2/AssignMovingAvg_2/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<29
7basemodel/batch_normalization_2/AssignMovingAvg_2/decayЇ
@basemodel/batch_normalization_2/AssignMovingAvg_2/ReadVariableOpReadVariableOpGbasemodel_batch_normalization_2_assignmovingavg_readvariableop_resource0^basemodel/batch_normalization_2/AssignMovingAvg*
_output_shapes
:@*
dtype02B
@basemodel/batch_normalization_2/AssignMovingAvg_2/ReadVariableOp†
5basemodel/batch_normalization_2/AssignMovingAvg_2/subSubHbasemodel/batch_normalization_2/AssignMovingAvg_2/ReadVariableOp:value:0:basemodel/batch_normalization_2/moments_1/Squeeze:output:0*
T0*
_output_shapes
:@27
5basemodel/batch_normalization_2/AssignMovingAvg_2/subЧ
5basemodel/batch_normalization_2/AssignMovingAvg_2/mulMul9basemodel/batch_normalization_2/AssignMovingAvg_2/sub:z:0@basemodel/batch_normalization_2/AssignMovingAvg_2/decay:output:0*
T0*
_output_shapes
:@27
5basemodel/batch_normalization_2/AssignMovingAvg_2/mulЩ
1basemodel/batch_normalization_2/AssignMovingAvg_2AssignSubVariableOpGbasemodel_batch_normalization_2_assignmovingavg_readvariableop_resource9basemodel/batch_normalization_2/AssignMovingAvg_2/mul:z:00^basemodel/batch_normalization_2/AssignMovingAvgA^basemodel/batch_normalization_2/AssignMovingAvg_2/ReadVariableOp*
_output_shapes
 *
dtype023
1basemodel/batch_normalization_2/AssignMovingAvg_2Ј
7basemodel/batch_normalization_2/AssignMovingAvg_3/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<29
7basemodel/batch_normalization_2/AssignMovingAvg_3/decayЊ
@basemodel/batch_normalization_2/AssignMovingAvg_3/ReadVariableOpReadVariableOpIbasemodel_batch_normalization_2_assignmovingavg_1_readvariableop_resource2^basemodel/batch_normalization_2/AssignMovingAvg_1*
_output_shapes
:@*
dtype02B
@basemodel/batch_normalization_2/AssignMovingAvg_3/ReadVariableOpҐ
5basemodel/batch_normalization_2/AssignMovingAvg_3/subSubHbasemodel/batch_normalization_2/AssignMovingAvg_3/ReadVariableOp:value:0<basemodel/batch_normalization_2/moments_1/Squeeze_1:output:0*
T0*
_output_shapes
:@27
5basemodel/batch_normalization_2/AssignMovingAvg_3/subЧ
5basemodel/batch_normalization_2/AssignMovingAvg_3/mulMul9basemodel/batch_normalization_2/AssignMovingAvg_3/sub:z:0@basemodel/batch_normalization_2/AssignMovingAvg_3/decay:output:0*
T0*
_output_shapes
:@27
5basemodel/batch_normalization_2/AssignMovingAvg_3/mulЭ
1basemodel/batch_normalization_2/AssignMovingAvg_3AssignSubVariableOpIbasemodel_batch_normalization_2_assignmovingavg_1_readvariableop_resource9basemodel/batch_normalization_2/AssignMovingAvg_3/mul:z:02^basemodel/batch_normalization_2/AssignMovingAvg_1A^basemodel/batch_normalization_2/AssignMovingAvg_3/ReadVariableOp*
_output_shapes
 *
dtype023
1basemodel/batch_normalization_2/AssignMovingAvg_3Ђ
1basemodel/batch_normalization_2/batchnorm_1/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:23
1basemodel/batch_normalization_2/batchnorm_1/add/yК
/basemodel/batch_normalization_2/batchnorm_1/addAddV2<basemodel/batch_normalization_2/moments_1/Squeeze_1:output:0:basemodel/batch_normalization_2/batchnorm_1/add/y:output:0*
T0*
_output_shapes
:@21
/basemodel/batch_normalization_2/batchnorm_1/add…
1basemodel/batch_normalization_2/batchnorm_1/RsqrtRsqrt3basemodel/batch_normalization_2/batchnorm_1/add:z:0*
T0*
_output_shapes
:@23
1basemodel/batch_normalization_2/batchnorm_1/RsqrtВ
>basemodel/batch_normalization_2/batchnorm_1/mul/ReadVariableOpReadVariableOpEbasemodel_batch_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02@
>basemodel/batch_normalization_2/batchnorm_1/mul/ReadVariableOpН
/basemodel/batch_normalization_2/batchnorm_1/mulMul5basemodel/batch_normalization_2/batchnorm_1/Rsqrt:y:0Fbasemodel/batch_normalization_2/batchnorm_1/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@21
/basemodel/batch_normalization_2/batchnorm_1/mulЖ
1basemodel/batch_normalization_2/batchnorm_1/mul_1Mul,basemodel/stream_2_conv_1/BiasAdd_1:output:03basemodel/batch_normalization_2/batchnorm_1/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€}@23
1basemodel/batch_normalization_2/batchnorm_1/mul_1Г
1basemodel/batch_normalization_2/batchnorm_1/mul_2Mul:basemodel/batch_normalization_2/moments_1/Squeeze:output:03basemodel/batch_normalization_2/batchnorm_1/mul:z:0*
T0*
_output_shapes
:@23
1basemodel/batch_normalization_2/batchnorm_1/mul_2ц
:basemodel/batch_normalization_2/batchnorm_1/ReadVariableOpReadVariableOpAbasemodel_batch_normalization_2_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02<
:basemodel/batch_normalization_2/batchnorm_1/ReadVariableOpЙ
/basemodel/batch_normalization_2/batchnorm_1/subSubBbasemodel/batch_normalization_2/batchnorm_1/ReadVariableOp:value:05basemodel/batch_normalization_2/batchnorm_1/mul_2:z:0*
T0*
_output_shapes
:@21
/basemodel/batch_normalization_2/batchnorm_1/subС
1basemodel/batch_normalization_2/batchnorm_1/add_1AddV25basemodel/batch_normalization_2/batchnorm_1/mul_1:z:03basemodel/batch_normalization_2/batchnorm_1/sub:z:0*
T0*+
_output_shapes
:€€€€€€€€€}@23
1basemodel/batch_normalization_2/batchnorm_1/add_1’
@basemodel/batch_normalization_1/moments_1/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2B
@basemodel/batch_normalization_1/moments_1/mean/reduction_indicesЯ
.basemodel/batch_normalization_1/moments_1/meanMean,basemodel/stream_1_conv_1/BiasAdd_1:output:0Ibasemodel/batch_normalization_1/moments_1/mean/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(20
.basemodel/batch_normalization_1/moments_1/meanж
6basemodel/batch_normalization_1/moments_1/StopGradientStopGradient7basemodel/batch_normalization_1/moments_1/mean:output:0*
T0*"
_output_shapes
:@28
6basemodel/batch_normalization_1/moments_1/StopGradientі
;basemodel/batch_normalization_1/moments_1/SquaredDifferenceSquaredDifference,basemodel/stream_1_conv_1/BiasAdd_1:output:0?basemodel/batch_normalization_1/moments_1/StopGradient:output:0*
T0*+
_output_shapes
:€€€€€€€€€}@2=
;basemodel/batch_normalization_1/moments_1/SquaredDifferenceЁ
Dbasemodel/batch_normalization_1/moments_1/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2F
Dbasemodel/batch_normalization_1/moments_1/variance/reduction_indicesЊ
2basemodel/batch_normalization_1/moments_1/varianceMean?basemodel/batch_normalization_1/moments_1/SquaredDifference:z:0Mbasemodel/batch_normalization_1/moments_1/variance/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(24
2basemodel/batch_normalization_1/moments_1/varianceз
1basemodel/batch_normalization_1/moments_1/SqueezeSqueeze7basemodel/batch_normalization_1/moments_1/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 23
1basemodel/batch_normalization_1/moments_1/Squeezeп
3basemodel/batch_normalization_1/moments_1/Squeeze_1Squeeze;basemodel/batch_normalization_1/moments_1/variance:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 25
3basemodel/batch_normalization_1/moments_1/Squeeze_1Ј
7basemodel/batch_normalization_1/AssignMovingAvg_2/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<29
7basemodel/batch_normalization_1/AssignMovingAvg_2/decayЇ
@basemodel/batch_normalization_1/AssignMovingAvg_2/ReadVariableOpReadVariableOpGbasemodel_batch_normalization_1_assignmovingavg_readvariableop_resource0^basemodel/batch_normalization_1/AssignMovingAvg*
_output_shapes
:@*
dtype02B
@basemodel/batch_normalization_1/AssignMovingAvg_2/ReadVariableOp†
5basemodel/batch_normalization_1/AssignMovingAvg_2/subSubHbasemodel/batch_normalization_1/AssignMovingAvg_2/ReadVariableOp:value:0:basemodel/batch_normalization_1/moments_1/Squeeze:output:0*
T0*
_output_shapes
:@27
5basemodel/batch_normalization_1/AssignMovingAvg_2/subЧ
5basemodel/batch_normalization_1/AssignMovingAvg_2/mulMul9basemodel/batch_normalization_1/AssignMovingAvg_2/sub:z:0@basemodel/batch_normalization_1/AssignMovingAvg_2/decay:output:0*
T0*
_output_shapes
:@27
5basemodel/batch_normalization_1/AssignMovingAvg_2/mulЩ
1basemodel/batch_normalization_1/AssignMovingAvg_2AssignSubVariableOpGbasemodel_batch_normalization_1_assignmovingavg_readvariableop_resource9basemodel/batch_normalization_1/AssignMovingAvg_2/mul:z:00^basemodel/batch_normalization_1/AssignMovingAvgA^basemodel/batch_normalization_1/AssignMovingAvg_2/ReadVariableOp*
_output_shapes
 *
dtype023
1basemodel/batch_normalization_1/AssignMovingAvg_2Ј
7basemodel/batch_normalization_1/AssignMovingAvg_3/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<29
7basemodel/batch_normalization_1/AssignMovingAvg_3/decayЊ
@basemodel/batch_normalization_1/AssignMovingAvg_3/ReadVariableOpReadVariableOpIbasemodel_batch_normalization_1_assignmovingavg_1_readvariableop_resource2^basemodel/batch_normalization_1/AssignMovingAvg_1*
_output_shapes
:@*
dtype02B
@basemodel/batch_normalization_1/AssignMovingAvg_3/ReadVariableOpҐ
5basemodel/batch_normalization_1/AssignMovingAvg_3/subSubHbasemodel/batch_normalization_1/AssignMovingAvg_3/ReadVariableOp:value:0<basemodel/batch_normalization_1/moments_1/Squeeze_1:output:0*
T0*
_output_shapes
:@27
5basemodel/batch_normalization_1/AssignMovingAvg_3/subЧ
5basemodel/batch_normalization_1/AssignMovingAvg_3/mulMul9basemodel/batch_normalization_1/AssignMovingAvg_3/sub:z:0@basemodel/batch_normalization_1/AssignMovingAvg_3/decay:output:0*
T0*
_output_shapes
:@27
5basemodel/batch_normalization_1/AssignMovingAvg_3/mulЭ
1basemodel/batch_normalization_1/AssignMovingAvg_3AssignSubVariableOpIbasemodel_batch_normalization_1_assignmovingavg_1_readvariableop_resource9basemodel/batch_normalization_1/AssignMovingAvg_3/mul:z:02^basemodel/batch_normalization_1/AssignMovingAvg_1A^basemodel/batch_normalization_1/AssignMovingAvg_3/ReadVariableOp*
_output_shapes
 *
dtype023
1basemodel/batch_normalization_1/AssignMovingAvg_3Ђ
1basemodel/batch_normalization_1/batchnorm_1/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:23
1basemodel/batch_normalization_1/batchnorm_1/add/yК
/basemodel/batch_normalization_1/batchnorm_1/addAddV2<basemodel/batch_normalization_1/moments_1/Squeeze_1:output:0:basemodel/batch_normalization_1/batchnorm_1/add/y:output:0*
T0*
_output_shapes
:@21
/basemodel/batch_normalization_1/batchnorm_1/add…
1basemodel/batch_normalization_1/batchnorm_1/RsqrtRsqrt3basemodel/batch_normalization_1/batchnorm_1/add:z:0*
T0*
_output_shapes
:@23
1basemodel/batch_normalization_1/batchnorm_1/RsqrtВ
>basemodel/batch_normalization_1/batchnorm_1/mul/ReadVariableOpReadVariableOpEbasemodel_batch_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02@
>basemodel/batch_normalization_1/batchnorm_1/mul/ReadVariableOpН
/basemodel/batch_normalization_1/batchnorm_1/mulMul5basemodel/batch_normalization_1/batchnorm_1/Rsqrt:y:0Fbasemodel/batch_normalization_1/batchnorm_1/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@21
/basemodel/batch_normalization_1/batchnorm_1/mulЖ
1basemodel/batch_normalization_1/batchnorm_1/mul_1Mul,basemodel/stream_1_conv_1/BiasAdd_1:output:03basemodel/batch_normalization_1/batchnorm_1/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€}@23
1basemodel/batch_normalization_1/batchnorm_1/mul_1Г
1basemodel/batch_normalization_1/batchnorm_1/mul_2Mul:basemodel/batch_normalization_1/moments_1/Squeeze:output:03basemodel/batch_normalization_1/batchnorm_1/mul:z:0*
T0*
_output_shapes
:@23
1basemodel/batch_normalization_1/batchnorm_1/mul_2ц
:basemodel/batch_normalization_1/batchnorm_1/ReadVariableOpReadVariableOpAbasemodel_batch_normalization_1_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02<
:basemodel/batch_normalization_1/batchnorm_1/ReadVariableOpЙ
/basemodel/batch_normalization_1/batchnorm_1/subSubBbasemodel/batch_normalization_1/batchnorm_1/ReadVariableOp:value:05basemodel/batch_normalization_1/batchnorm_1/mul_2:z:0*
T0*
_output_shapes
:@21
/basemodel/batch_normalization_1/batchnorm_1/subС
1basemodel/batch_normalization_1/batchnorm_1/add_1AddV25basemodel/batch_normalization_1/batchnorm_1/mul_1:z:03basemodel/batch_normalization_1/batchnorm_1/sub:z:0*
T0*+
_output_shapes
:€€€€€€€€€}@23
1basemodel/batch_normalization_1/batchnorm_1/add_1—
>basemodel/batch_normalization/moments_1/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2@
>basemodel/batch_normalization/moments_1/mean/reduction_indicesЩ
,basemodel/batch_normalization/moments_1/meanMean,basemodel/stream_0_conv_1/BiasAdd_1:output:0Gbasemodel/batch_normalization/moments_1/mean/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(2.
,basemodel/batch_normalization/moments_1/meanа
4basemodel/batch_normalization/moments_1/StopGradientStopGradient5basemodel/batch_normalization/moments_1/mean:output:0*
T0*"
_output_shapes
:@26
4basemodel/batch_normalization/moments_1/StopGradientЃ
9basemodel/batch_normalization/moments_1/SquaredDifferenceSquaredDifference,basemodel/stream_0_conv_1/BiasAdd_1:output:0=basemodel/batch_normalization/moments_1/StopGradient:output:0*
T0*+
_output_shapes
:€€€€€€€€€}@2;
9basemodel/batch_normalization/moments_1/SquaredDifferenceў
Bbasemodel/batch_normalization/moments_1/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2D
Bbasemodel/batch_normalization/moments_1/variance/reduction_indicesґ
0basemodel/batch_normalization/moments_1/varianceMean=basemodel/batch_normalization/moments_1/SquaredDifference:z:0Kbasemodel/batch_normalization/moments_1/variance/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(22
0basemodel/batch_normalization/moments_1/varianceб
/basemodel/batch_normalization/moments_1/SqueezeSqueeze5basemodel/batch_normalization/moments_1/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 21
/basemodel/batch_normalization/moments_1/Squeezeй
1basemodel/batch_normalization/moments_1/Squeeze_1Squeeze9basemodel/batch_normalization/moments_1/variance:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 23
1basemodel/batch_normalization/moments_1/Squeeze_1≥
5basemodel/batch_normalization/AssignMovingAvg_2/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<27
5basemodel/batch_normalization/AssignMovingAvg_2/decay≤
>basemodel/batch_normalization/AssignMovingAvg_2/ReadVariableOpReadVariableOpEbasemodel_batch_normalization_assignmovingavg_readvariableop_resource.^basemodel/batch_normalization/AssignMovingAvg*
_output_shapes
:@*
dtype02@
>basemodel/batch_normalization/AssignMovingAvg_2/ReadVariableOpШ
3basemodel/batch_normalization/AssignMovingAvg_2/subSubFbasemodel/batch_normalization/AssignMovingAvg_2/ReadVariableOp:value:08basemodel/batch_normalization/moments_1/Squeeze:output:0*
T0*
_output_shapes
:@25
3basemodel/batch_normalization/AssignMovingAvg_2/subП
3basemodel/batch_normalization/AssignMovingAvg_2/mulMul7basemodel/batch_normalization/AssignMovingAvg_2/sub:z:0>basemodel/batch_normalization/AssignMovingAvg_2/decay:output:0*
T0*
_output_shapes
:@25
3basemodel/batch_normalization/AssignMovingAvg_2/mulН
/basemodel/batch_normalization/AssignMovingAvg_2AssignSubVariableOpEbasemodel_batch_normalization_assignmovingavg_readvariableop_resource7basemodel/batch_normalization/AssignMovingAvg_2/mul:z:0.^basemodel/batch_normalization/AssignMovingAvg?^basemodel/batch_normalization/AssignMovingAvg_2/ReadVariableOp*
_output_shapes
 *
dtype021
/basemodel/batch_normalization/AssignMovingAvg_2≥
5basemodel/batch_normalization/AssignMovingAvg_3/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<27
5basemodel/batch_normalization/AssignMovingAvg_3/decayґ
>basemodel/batch_normalization/AssignMovingAvg_3/ReadVariableOpReadVariableOpGbasemodel_batch_normalization_assignmovingavg_1_readvariableop_resource0^basemodel/batch_normalization/AssignMovingAvg_1*
_output_shapes
:@*
dtype02@
>basemodel/batch_normalization/AssignMovingAvg_3/ReadVariableOpЪ
3basemodel/batch_normalization/AssignMovingAvg_3/subSubFbasemodel/batch_normalization/AssignMovingAvg_3/ReadVariableOp:value:0:basemodel/batch_normalization/moments_1/Squeeze_1:output:0*
T0*
_output_shapes
:@25
3basemodel/batch_normalization/AssignMovingAvg_3/subП
3basemodel/batch_normalization/AssignMovingAvg_3/mulMul7basemodel/batch_normalization/AssignMovingAvg_3/sub:z:0>basemodel/batch_normalization/AssignMovingAvg_3/decay:output:0*
T0*
_output_shapes
:@25
3basemodel/batch_normalization/AssignMovingAvg_3/mulС
/basemodel/batch_normalization/AssignMovingAvg_3AssignSubVariableOpGbasemodel_batch_normalization_assignmovingavg_1_readvariableop_resource7basemodel/batch_normalization/AssignMovingAvg_3/mul:z:00^basemodel/batch_normalization/AssignMovingAvg_1?^basemodel/batch_normalization/AssignMovingAvg_3/ReadVariableOp*
_output_shapes
 *
dtype021
/basemodel/batch_normalization/AssignMovingAvg_3І
/basemodel/batch_normalization/batchnorm_1/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:21
/basemodel/batch_normalization/batchnorm_1/add/yВ
-basemodel/batch_normalization/batchnorm_1/addAddV2:basemodel/batch_normalization/moments_1/Squeeze_1:output:08basemodel/batch_normalization/batchnorm_1/add/y:output:0*
T0*
_output_shapes
:@2/
-basemodel/batch_normalization/batchnorm_1/add√
/basemodel/batch_normalization/batchnorm_1/RsqrtRsqrt1basemodel/batch_normalization/batchnorm_1/add:z:0*
T0*
_output_shapes
:@21
/basemodel/batch_normalization/batchnorm_1/Rsqrtь
<basemodel/batch_normalization/batchnorm_1/mul/ReadVariableOpReadVariableOpCbasemodel_batch_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02>
<basemodel/batch_normalization/batchnorm_1/mul/ReadVariableOpЕ
-basemodel/batch_normalization/batchnorm_1/mulMul3basemodel/batch_normalization/batchnorm_1/Rsqrt:y:0Dbasemodel/batch_normalization/batchnorm_1/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2/
-basemodel/batch_normalization/batchnorm_1/mulА
/basemodel/batch_normalization/batchnorm_1/mul_1Mul,basemodel/stream_0_conv_1/BiasAdd_1:output:01basemodel/batch_normalization/batchnorm_1/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€}@21
/basemodel/batch_normalization/batchnorm_1/mul_1ы
/basemodel/batch_normalization/batchnorm_1/mul_2Mul8basemodel/batch_normalization/moments_1/Squeeze:output:01basemodel/batch_normalization/batchnorm_1/mul:z:0*
T0*
_output_shapes
:@21
/basemodel/batch_normalization/batchnorm_1/mul_2р
8basemodel/batch_normalization/batchnorm_1/ReadVariableOpReadVariableOp?basemodel_batch_normalization_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02:
8basemodel/batch_normalization/batchnorm_1/ReadVariableOpБ
-basemodel/batch_normalization/batchnorm_1/subSub@basemodel/batch_normalization/batchnorm_1/ReadVariableOp:value:03basemodel/batch_normalization/batchnorm_1/mul_2:z:0*
T0*
_output_shapes
:@2/
-basemodel/batch_normalization/batchnorm_1/subЙ
/basemodel/batch_normalization/batchnorm_1/add_1AddV23basemodel/batch_normalization/batchnorm_1/mul_1:z:01basemodel/batch_normalization/batchnorm_1/sub:z:0*
T0*+
_output_shapes
:€€€€€€€€€}@21
/basemodel/batch_normalization/batchnorm_1/add_1≥
basemodel/activation_2/Relu_1Relu5basemodel/batch_normalization_2/batchnorm_1/add_1:z:0*
T0*+
_output_shapes
:€€€€€€€€€}@2
basemodel/activation_2/Relu_1≥
basemodel/activation_1/Relu_1Relu5basemodel/batch_normalization_1/batchnorm_1/add_1:z:0*
T0*+
_output_shapes
:€€€€€€€€€}@2
basemodel/activation_1/Relu_1≠
basemodel/activation/Relu_1Relu3basemodel/batch_normalization/batchnorm_1/add_1:z:0*
T0*+
_output_shapes
:€€€€€€€€€}@2
basemodel/activation/Relu_1Ы
)basemodel/stream_2_drop_1/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †?2+
)basemodel/stream_2_drop_1/dropout_1/Constр
'basemodel/stream_2_drop_1/dropout_1/MulMul+basemodel/activation_2/Relu_1:activations:02basemodel/stream_2_drop_1/dropout_1/Const:output:0*
T0*+
_output_shapes
:€€€€€€€€€}@2)
'basemodel/stream_2_drop_1/dropout_1/Mul±
)basemodel/stream_2_drop_1/dropout_1/ShapeShape+basemodel/activation_2/Relu_1:activations:0*
T0*
_output_shapes
:2+
)basemodel/stream_2_drop_1/dropout_1/ShapeІ
@basemodel/stream_2_drop_1/dropout_1/random_uniform/RandomUniformRandomUniform2basemodel/stream_2_drop_1/dropout_1/Shape:output:0*
T0*+
_output_shapes
:€€€€€€€€€}@*
dtype0*
seedЈ*
seed2Ј2B
@basemodel/stream_2_drop_1/dropout_1/random_uniform/RandomUniform≠
2basemodel/stream_2_drop_1/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL>24
2basemodel/stream_2_drop_1/dropout_1/GreaterEqual/y≤
0basemodel/stream_2_drop_1/dropout_1/GreaterEqualGreaterEqualIbasemodel/stream_2_drop_1/dropout_1/random_uniform/RandomUniform:output:0;basemodel/stream_2_drop_1/dropout_1/GreaterEqual/y:output:0*
T0*+
_output_shapes
:€€€€€€€€€}@22
0basemodel/stream_2_drop_1/dropout_1/GreaterEqual„
(basemodel/stream_2_drop_1/dropout_1/CastCast4basemodel/stream_2_drop_1/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:€€€€€€€€€}@2*
(basemodel/stream_2_drop_1/dropout_1/Castо
)basemodel/stream_2_drop_1/dropout_1/Mul_1Mul+basemodel/stream_2_drop_1/dropout_1/Mul:z:0,basemodel/stream_2_drop_1/dropout_1/Cast:y:0*
T0*+
_output_shapes
:€€€€€€€€€}@2+
)basemodel/stream_2_drop_1/dropout_1/Mul_1Ы
)basemodel/stream_1_drop_1/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †?2+
)basemodel/stream_1_drop_1/dropout_1/Constр
'basemodel/stream_1_drop_1/dropout_1/MulMul+basemodel/activation_1/Relu_1:activations:02basemodel/stream_1_drop_1/dropout_1/Const:output:0*
T0*+
_output_shapes
:€€€€€€€€€}@2)
'basemodel/stream_1_drop_1/dropout_1/Mul±
)basemodel/stream_1_drop_1/dropout_1/ShapeShape+basemodel/activation_1/Relu_1:activations:0*
T0*
_output_shapes
:2+
)basemodel/stream_1_drop_1/dropout_1/ShapeІ
@basemodel/stream_1_drop_1/dropout_1/random_uniform/RandomUniformRandomUniform2basemodel/stream_1_drop_1/dropout_1/Shape:output:0*
T0*+
_output_shapes
:€€€€€€€€€}@*
dtype0*
seedЈ*
seed2Ј2B
@basemodel/stream_1_drop_1/dropout_1/random_uniform/RandomUniform≠
2basemodel/stream_1_drop_1/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL>24
2basemodel/stream_1_drop_1/dropout_1/GreaterEqual/y≤
0basemodel/stream_1_drop_1/dropout_1/GreaterEqualGreaterEqualIbasemodel/stream_1_drop_1/dropout_1/random_uniform/RandomUniform:output:0;basemodel/stream_1_drop_1/dropout_1/GreaterEqual/y:output:0*
T0*+
_output_shapes
:€€€€€€€€€}@22
0basemodel/stream_1_drop_1/dropout_1/GreaterEqual„
(basemodel/stream_1_drop_1/dropout_1/CastCast4basemodel/stream_1_drop_1/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:€€€€€€€€€}@2*
(basemodel/stream_1_drop_1/dropout_1/Castо
)basemodel/stream_1_drop_1/dropout_1/Mul_1Mul+basemodel/stream_1_drop_1/dropout_1/Mul:z:0,basemodel/stream_1_drop_1/dropout_1/Cast:y:0*
T0*+
_output_shapes
:€€€€€€€€€}@2+
)basemodel/stream_1_drop_1/dropout_1/Mul_1Ы
)basemodel/stream_0_drop_1/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †?2+
)basemodel/stream_0_drop_1/dropout_1/Constо
'basemodel/stream_0_drop_1/dropout_1/MulMul)basemodel/activation/Relu_1:activations:02basemodel/stream_0_drop_1/dropout_1/Const:output:0*
T0*+
_output_shapes
:€€€€€€€€€}@2)
'basemodel/stream_0_drop_1/dropout_1/Mulѓ
)basemodel/stream_0_drop_1/dropout_1/ShapeShape)basemodel/activation/Relu_1:activations:0*
T0*
_output_shapes
:2+
)basemodel/stream_0_drop_1/dropout_1/ShapeІ
@basemodel/stream_0_drop_1/dropout_1/random_uniform/RandomUniformRandomUniform2basemodel/stream_0_drop_1/dropout_1/Shape:output:0*
T0*+
_output_shapes
:€€€€€€€€€}@*
dtype0*
seedЈ*
seed2Ј2B
@basemodel/stream_0_drop_1/dropout_1/random_uniform/RandomUniform≠
2basemodel/stream_0_drop_1/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL>24
2basemodel/stream_0_drop_1/dropout_1/GreaterEqual/y≤
0basemodel/stream_0_drop_1/dropout_1/GreaterEqualGreaterEqualIbasemodel/stream_0_drop_1/dropout_1/random_uniform/RandomUniform:output:0;basemodel/stream_0_drop_1/dropout_1/GreaterEqual/y:output:0*
T0*+
_output_shapes
:€€€€€€€€€}@22
0basemodel/stream_0_drop_1/dropout_1/GreaterEqual„
(basemodel/stream_0_drop_1/dropout_1/CastCast4basemodel/stream_0_drop_1/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:€€€€€€€€€}@2*
(basemodel/stream_0_drop_1/dropout_1/Castо
)basemodel/stream_0_drop_1/dropout_1/Mul_1Mul+basemodel/stream_0_drop_1/dropout_1/Mul:z:0,basemodel/stream_0_drop_1/dropout_1/Cast:y:0*
T0*+
_output_shapes
:€€€€€€€€€}@2+
)basemodel/stream_0_drop_1/dropout_1/Mul_1Љ
;basemodel/global_average_pooling1d/Mean_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2=
;basemodel/global_average_pooling1d/Mean_1/reduction_indicesЕ
)basemodel/global_average_pooling1d/Mean_1Mean-basemodel/stream_0_drop_1/dropout_1/Mul_1:z:0Dbasemodel/global_average_pooling1d/Mean_1/reduction_indices:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2+
)basemodel/global_average_pooling1d/Mean_1ј
=basemodel/global_average_pooling1d_1/Mean_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2?
=basemodel/global_average_pooling1d_1/Mean_1/reduction_indicesЛ
+basemodel/global_average_pooling1d_1/Mean_1Mean-basemodel/stream_1_drop_1/dropout_1/Mul_1:z:0Fbasemodel/global_average_pooling1d_1/Mean_1/reduction_indices:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2-
+basemodel/global_average_pooling1d_1/Mean_1ј
=basemodel/global_average_pooling1d_2/Mean_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2?
=basemodel/global_average_pooling1d_2/Mean_1/reduction_indicesЛ
+basemodel/global_average_pooling1d_2/Mean_1Mean-basemodel/stream_2_drop_1/dropout_1/Mul_1:z:0Fbasemodel/global_average_pooling1d_2/Mean_1/reduction_indices:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2-
+basemodel/global_average_pooling1d_2/Mean_1М
#basemodel/concatenate/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :2%
#basemodel/concatenate/concat_1/axis÷
basemodel/concatenate/concat_1ConcatV22basemodel/global_average_pooling1d/Mean_1:output:04basemodel/global_average_pooling1d_1/Mean_1:output:04basemodel/global_average_pooling1d_2/Mean_1:output:0,basemodel/concatenate/concat_1/axis:output:0*
N*
T0*(
_output_shapes
:€€€€€€€€€ј2 
basemodel/concatenate/concat_1»
)basemodel/dense_1/MatMul_1/ReadVariableOpReadVariableOp0basemodel_dense_1_matmul_readvariableop_resource*
_output_shapes
:	јT*
dtype02+
)basemodel/dense_1/MatMul_1/ReadVariableOp–
basemodel/dense_1/MatMul_1MatMul'basemodel/concatenate/concat_1:output:01basemodel/dense_1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€T2
basemodel/dense_1/MatMul_1∆
*basemodel/dense_1/BiasAdd_1/ReadVariableOpReadVariableOp1basemodel_dense_1_biasadd_readvariableop_resource*
_output_shapes
:T*
dtype02,
*basemodel/dense_1/BiasAdd_1/ReadVariableOp—
basemodel/dense_1/BiasAdd_1BiasAdd$basemodel/dense_1/MatMul_1:product:02basemodel/dense_1/BiasAdd_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€T2
basemodel/dense_1/BiasAdd_1ќ
@basemodel/batch_normalization_3/moments_1/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2B
@basemodel/batch_normalization_3/moments_1/mean/reduction_indicesУ
.basemodel/batch_normalization_3/moments_1/meanMean$basemodel/dense_1/BiasAdd_1:output:0Ibasemodel/batch_normalization_3/moments_1/mean/reduction_indices:output:0*
T0*
_output_shapes

:T*
	keep_dims(20
.basemodel/batch_normalization_3/moments_1/meanв
6basemodel/batch_normalization_3/moments_1/StopGradientStopGradient7basemodel/batch_normalization_3/moments_1/mean:output:0*
T0*
_output_shapes

:T28
6basemodel/batch_normalization_3/moments_1/StopGradient®
;basemodel/batch_normalization_3/moments_1/SquaredDifferenceSquaredDifference$basemodel/dense_1/BiasAdd_1:output:0?basemodel/batch_normalization_3/moments_1/StopGradient:output:0*
T0*'
_output_shapes
:€€€€€€€€€T2=
;basemodel/batch_normalization_3/moments_1/SquaredDifference÷
Dbasemodel/batch_normalization_3/moments_1/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2F
Dbasemodel/batch_normalization_3/moments_1/variance/reduction_indicesЇ
2basemodel/batch_normalization_3/moments_1/varianceMean?basemodel/batch_normalization_3/moments_1/SquaredDifference:z:0Mbasemodel/batch_normalization_3/moments_1/variance/reduction_indices:output:0*
T0*
_output_shapes

:T*
	keep_dims(24
2basemodel/batch_normalization_3/moments_1/varianceж
1basemodel/batch_normalization_3/moments_1/SqueezeSqueeze7basemodel/batch_normalization_3/moments_1/mean:output:0*
T0*
_output_shapes
:T*
squeeze_dims
 23
1basemodel/batch_normalization_3/moments_1/Squeezeо
3basemodel/batch_normalization_3/moments_1/Squeeze_1Squeeze;basemodel/batch_normalization_3/moments_1/variance:output:0*
T0*
_output_shapes
:T*
squeeze_dims
 25
3basemodel/batch_normalization_3/moments_1/Squeeze_1Ј
7basemodel/batch_normalization_3/AssignMovingAvg_2/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<29
7basemodel/batch_normalization_3/AssignMovingAvg_2/decayЇ
@basemodel/batch_normalization_3/AssignMovingAvg_2/ReadVariableOpReadVariableOpGbasemodel_batch_normalization_3_assignmovingavg_readvariableop_resource0^basemodel/batch_normalization_3/AssignMovingAvg*
_output_shapes
:T*
dtype02B
@basemodel/batch_normalization_3/AssignMovingAvg_2/ReadVariableOp†
5basemodel/batch_normalization_3/AssignMovingAvg_2/subSubHbasemodel/batch_normalization_3/AssignMovingAvg_2/ReadVariableOp:value:0:basemodel/batch_normalization_3/moments_1/Squeeze:output:0*
T0*
_output_shapes
:T27
5basemodel/batch_normalization_3/AssignMovingAvg_2/subЧ
5basemodel/batch_normalization_3/AssignMovingAvg_2/mulMul9basemodel/batch_normalization_3/AssignMovingAvg_2/sub:z:0@basemodel/batch_normalization_3/AssignMovingAvg_2/decay:output:0*
T0*
_output_shapes
:T27
5basemodel/batch_normalization_3/AssignMovingAvg_2/mulЩ
1basemodel/batch_normalization_3/AssignMovingAvg_2AssignSubVariableOpGbasemodel_batch_normalization_3_assignmovingavg_readvariableop_resource9basemodel/batch_normalization_3/AssignMovingAvg_2/mul:z:00^basemodel/batch_normalization_3/AssignMovingAvgA^basemodel/batch_normalization_3/AssignMovingAvg_2/ReadVariableOp*
_output_shapes
 *
dtype023
1basemodel/batch_normalization_3/AssignMovingAvg_2Ј
7basemodel/batch_normalization_3/AssignMovingAvg_3/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<29
7basemodel/batch_normalization_3/AssignMovingAvg_3/decayЊ
@basemodel/batch_normalization_3/AssignMovingAvg_3/ReadVariableOpReadVariableOpIbasemodel_batch_normalization_3_assignmovingavg_1_readvariableop_resource2^basemodel/batch_normalization_3/AssignMovingAvg_1*
_output_shapes
:T*
dtype02B
@basemodel/batch_normalization_3/AssignMovingAvg_3/ReadVariableOpҐ
5basemodel/batch_normalization_3/AssignMovingAvg_3/subSubHbasemodel/batch_normalization_3/AssignMovingAvg_3/ReadVariableOp:value:0<basemodel/batch_normalization_3/moments_1/Squeeze_1:output:0*
T0*
_output_shapes
:T27
5basemodel/batch_normalization_3/AssignMovingAvg_3/subЧ
5basemodel/batch_normalization_3/AssignMovingAvg_3/mulMul9basemodel/batch_normalization_3/AssignMovingAvg_3/sub:z:0@basemodel/batch_normalization_3/AssignMovingAvg_3/decay:output:0*
T0*
_output_shapes
:T27
5basemodel/batch_normalization_3/AssignMovingAvg_3/mulЭ
1basemodel/batch_normalization_3/AssignMovingAvg_3AssignSubVariableOpIbasemodel_batch_normalization_3_assignmovingavg_1_readvariableop_resource9basemodel/batch_normalization_3/AssignMovingAvg_3/mul:z:02^basemodel/batch_normalization_3/AssignMovingAvg_1A^basemodel/batch_normalization_3/AssignMovingAvg_3/ReadVariableOp*
_output_shapes
 *
dtype023
1basemodel/batch_normalization_3/AssignMovingAvg_3Ђ
1basemodel/batch_normalization_3/batchnorm_1/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:23
1basemodel/batch_normalization_3/batchnorm_1/add/yК
/basemodel/batch_normalization_3/batchnorm_1/addAddV2<basemodel/batch_normalization_3/moments_1/Squeeze_1:output:0:basemodel/batch_normalization_3/batchnorm_1/add/y:output:0*
T0*
_output_shapes
:T21
/basemodel/batch_normalization_3/batchnorm_1/add…
1basemodel/batch_normalization_3/batchnorm_1/RsqrtRsqrt3basemodel/batch_normalization_3/batchnorm_1/add:z:0*
T0*
_output_shapes
:T23
1basemodel/batch_normalization_3/batchnorm_1/RsqrtВ
>basemodel/batch_normalization_3/batchnorm_1/mul/ReadVariableOpReadVariableOpEbasemodel_batch_normalization_3_batchnorm_mul_readvariableop_resource*
_output_shapes
:T*
dtype02@
>basemodel/batch_normalization_3/batchnorm_1/mul/ReadVariableOpН
/basemodel/batch_normalization_3/batchnorm_1/mulMul5basemodel/batch_normalization_3/batchnorm_1/Rsqrt:y:0Fbasemodel/batch_normalization_3/batchnorm_1/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:T21
/basemodel/batch_normalization_3/batchnorm_1/mulъ
1basemodel/batch_normalization_3/batchnorm_1/mul_1Mul$basemodel/dense_1/BiasAdd_1:output:03basemodel/batch_normalization_3/batchnorm_1/mul:z:0*
T0*'
_output_shapes
:€€€€€€€€€T23
1basemodel/batch_normalization_3/batchnorm_1/mul_1Г
1basemodel/batch_normalization_3/batchnorm_1/mul_2Mul:basemodel/batch_normalization_3/moments_1/Squeeze:output:03basemodel/batch_normalization_3/batchnorm_1/mul:z:0*
T0*
_output_shapes
:T23
1basemodel/batch_normalization_3/batchnorm_1/mul_2ц
:basemodel/batch_normalization_3/batchnorm_1/ReadVariableOpReadVariableOpAbasemodel_batch_normalization_3_batchnorm_readvariableop_resource*
_output_shapes
:T*
dtype02<
:basemodel/batch_normalization_3/batchnorm_1/ReadVariableOpЙ
/basemodel/batch_normalization_3/batchnorm_1/subSubBbasemodel/batch_normalization_3/batchnorm_1/ReadVariableOp:value:05basemodel/batch_normalization_3/batchnorm_1/mul_2:z:0*
T0*
_output_shapes
:T21
/basemodel/batch_normalization_3/batchnorm_1/subН
1basemodel/batch_normalization_3/batchnorm_1/add_1AddV25basemodel/batch_normalization_3/batchnorm_1/mul_1:z:03basemodel/batch_normalization_3/batchnorm_1/sub:z:0*
T0*'
_output_shapes
:€€€€€€€€€T23
1basemodel/batch_normalization_3/batchnorm_1/add_1ƒ
&basemodel/dense_activation_1/Sigmoid_1Sigmoid5basemodel/batch_normalization_3/batchnorm_1/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€T2(
&basemodel/dense_activation_1/Sigmoid_1Ђ
distance/subSub(basemodel/dense_activation_1/Sigmoid:y:0*basemodel/dense_activation_1/Sigmoid_1:y:0*
T0*'
_output_shapes
:€€€€€€€€€T2
distance/subp
distance/SquareSquaredistance/sub:z:0*
T0*'
_output_shapes
:€€€€€€€€€T2
distance/SquareЛ
distance/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2 
distance/Sum/reduction_indices§
distance/SumSumdistance/Square:y:0'distance/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:€€€€€€€€€*
	keep_dims(2
distance/Sume
distance/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
distance/ConstС
distance/MaximumMaximumdistance/Sum:output:0distance/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€2
distance/Maximumn
distance/SqrtSqrtdistance/Maximum:z:0*
T0*'
_output_shapes
:€€€€€€€€€2
distance/Sqrtш
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpEbasemodel_stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype027
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp√
&stream_0_conv_1/kernel/Regularizer/AbsAbs=stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_0_conv_1/kernel/Regularizer/Abs©
(stream_0_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_0_conv_1/kernel/Regularizer/Const„
&stream_0_conv_1/kernel/Regularizer/SumSum*stream_0_conv_1/kernel/Regularizer/Abs:y:01stream_0_conv_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/SumЩ
(stream_0_conv_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2*
(stream_0_conv_1/kernel/Regularizer/mul/x№
&stream_0_conv_1/kernel/Regularizer/mulMul1stream_0_conv_1/kernel/Regularizer/mul/x:output:0/stream_0_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/mulю
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpEbasemodel_stream_1_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02:
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpѕ
)stream_1_conv_1/kernel/Regularizer/SquareSquare@stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2+
)stream_1_conv_1/kernel/Regularizer/Square©
(stream_1_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_1_conv_1/kernel/Regularizer/ConstЏ
&stream_1_conv_1/kernel/Regularizer/SumSum-stream_1_conv_1/kernel/Regularizer/Square:y:01stream_1_conv_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&stream_1_conv_1/kernel/Regularizer/SumЩ
(stream_1_conv_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2*
(stream_1_conv_1/kernel/Regularizer/mul/x№
&stream_1_conv_1/kernel/Regularizer/mulMul1stream_1_conv_1/kernel/Regularizer/mul/x:output:0/stream_1_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_1_conv_1/kernel/Regularizer/mulш
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpEbasemodel_stream_2_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype027
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp√
&stream_2_conv_1/kernel/Regularizer/AbsAbs=stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_2_conv_1/kernel/Regularizer/Abs©
(stream_2_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_2_conv_1/kernel/Regularizer/Const„
&stream_2_conv_1/kernel/Regularizer/SumSum*stream_2_conv_1/kernel/Regularizer/Abs:y:01stream_2_conv_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&stream_2_conv_1/kernel/Regularizer/SumЩ
(stream_2_conv_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2*
(stream_2_conv_1/kernel/Regularizer/mul/x№
&stream_2_conv_1/kernel/Regularizer/mulMul1stream_2_conv_1/kernel/Regularizer/mul/x:output:0/stream_2_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_2_conv_1/kernel/Regularizer/mul–
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp0basemodel_dense_1_matmul_readvariableop_resource*
_output_shapes
:	јT*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOp®
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	јT2 
dense_1/kernel/Regularizer/AbsХ
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_1/kernel/Regularizer/ConstЈ
dense_1/kernel/Regularizer/SumSum"dense_1/kernel/Regularizer/Abs:y:0)dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/SumЙ
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2"
 dense_1/kernel/Regularizer/mul/xЉ
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/mull
IdentityIdentitydistance/Sqrt:y:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€2

IdentityЈ
NoOpNoOp.^basemodel/batch_normalization/AssignMovingAvg=^basemodel/batch_normalization/AssignMovingAvg/ReadVariableOp0^basemodel/batch_normalization/AssignMovingAvg_1?^basemodel/batch_normalization/AssignMovingAvg_1/ReadVariableOp0^basemodel/batch_normalization/AssignMovingAvg_2?^basemodel/batch_normalization/AssignMovingAvg_2/ReadVariableOp0^basemodel/batch_normalization/AssignMovingAvg_3?^basemodel/batch_normalization/AssignMovingAvg_3/ReadVariableOp7^basemodel/batch_normalization/batchnorm/ReadVariableOp;^basemodel/batch_normalization/batchnorm/mul/ReadVariableOp9^basemodel/batch_normalization/batchnorm_1/ReadVariableOp=^basemodel/batch_normalization/batchnorm_1/mul/ReadVariableOp0^basemodel/batch_normalization_1/AssignMovingAvg?^basemodel/batch_normalization_1/AssignMovingAvg/ReadVariableOp2^basemodel/batch_normalization_1/AssignMovingAvg_1A^basemodel/batch_normalization_1/AssignMovingAvg_1/ReadVariableOp2^basemodel/batch_normalization_1/AssignMovingAvg_2A^basemodel/batch_normalization_1/AssignMovingAvg_2/ReadVariableOp2^basemodel/batch_normalization_1/AssignMovingAvg_3A^basemodel/batch_normalization_1/AssignMovingAvg_3/ReadVariableOp9^basemodel/batch_normalization_1/batchnorm/ReadVariableOp=^basemodel/batch_normalization_1/batchnorm/mul/ReadVariableOp;^basemodel/batch_normalization_1/batchnorm_1/ReadVariableOp?^basemodel/batch_normalization_1/batchnorm_1/mul/ReadVariableOp0^basemodel/batch_normalization_2/AssignMovingAvg?^basemodel/batch_normalization_2/AssignMovingAvg/ReadVariableOp2^basemodel/batch_normalization_2/AssignMovingAvg_1A^basemodel/batch_normalization_2/AssignMovingAvg_1/ReadVariableOp2^basemodel/batch_normalization_2/AssignMovingAvg_2A^basemodel/batch_normalization_2/AssignMovingAvg_2/ReadVariableOp2^basemodel/batch_normalization_2/AssignMovingAvg_3A^basemodel/batch_normalization_2/AssignMovingAvg_3/ReadVariableOp9^basemodel/batch_normalization_2/batchnorm/ReadVariableOp=^basemodel/batch_normalization_2/batchnorm/mul/ReadVariableOp;^basemodel/batch_normalization_2/batchnorm_1/ReadVariableOp?^basemodel/batch_normalization_2/batchnorm_1/mul/ReadVariableOp0^basemodel/batch_normalization_3/AssignMovingAvg?^basemodel/batch_normalization_3/AssignMovingAvg/ReadVariableOp2^basemodel/batch_normalization_3/AssignMovingAvg_1A^basemodel/batch_normalization_3/AssignMovingAvg_1/ReadVariableOp2^basemodel/batch_normalization_3/AssignMovingAvg_2A^basemodel/batch_normalization_3/AssignMovingAvg_2/ReadVariableOp2^basemodel/batch_normalization_3/AssignMovingAvg_3A^basemodel/batch_normalization_3/AssignMovingAvg_3/ReadVariableOp9^basemodel/batch_normalization_3/batchnorm/ReadVariableOp=^basemodel/batch_normalization_3/batchnorm/mul/ReadVariableOp;^basemodel/batch_normalization_3/batchnorm_1/ReadVariableOp?^basemodel/batch_normalization_3/batchnorm_1/mul/ReadVariableOp)^basemodel/dense_1/BiasAdd/ReadVariableOp+^basemodel/dense_1/BiasAdd_1/ReadVariableOp(^basemodel/dense_1/MatMul/ReadVariableOp*^basemodel/dense_1/MatMul_1/ReadVariableOp1^basemodel/stream_0_conv_1/BiasAdd/ReadVariableOp3^basemodel/stream_0_conv_1/BiasAdd_1/ReadVariableOp=^basemodel/stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp?^basemodel/stream_0_conv_1/conv1d_1/ExpandDims_1/ReadVariableOp1^basemodel/stream_1_conv_1/BiasAdd/ReadVariableOp3^basemodel/stream_1_conv_1/BiasAdd_1/ReadVariableOp=^basemodel/stream_1_conv_1/conv1d/ExpandDims_1/ReadVariableOp?^basemodel/stream_1_conv_1/conv1d_1/ExpandDims_1/ReadVariableOp1^basemodel/stream_2_conv_1/BiasAdd/ReadVariableOp3^basemodel/stream_2_conv_1/BiasAdd_1/ReadVariableOp=^basemodel/stream_2_conv_1/conv1d/ExpandDims_1/ReadVariableOp?^basemodel/stream_2_conv_1/conv1d_1/ExpandDims_1/ReadVariableOp.^dense_1/kernel/Regularizer/Abs/ReadVariableOp6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp9^stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp6^stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*q
_input_shapes`
^:€€€€€€€€€}:€€€€€€€€€}: : : : : : : : : : : : : : : : : : : : : : : : 2^
-basemodel/batch_normalization/AssignMovingAvg-basemodel/batch_normalization/AssignMovingAvg2|
<basemodel/batch_normalization/AssignMovingAvg/ReadVariableOp<basemodel/batch_normalization/AssignMovingAvg/ReadVariableOp2b
/basemodel/batch_normalization/AssignMovingAvg_1/basemodel/batch_normalization/AssignMovingAvg_12А
>basemodel/batch_normalization/AssignMovingAvg_1/ReadVariableOp>basemodel/batch_normalization/AssignMovingAvg_1/ReadVariableOp2b
/basemodel/batch_normalization/AssignMovingAvg_2/basemodel/batch_normalization/AssignMovingAvg_22А
>basemodel/batch_normalization/AssignMovingAvg_2/ReadVariableOp>basemodel/batch_normalization/AssignMovingAvg_2/ReadVariableOp2b
/basemodel/batch_normalization/AssignMovingAvg_3/basemodel/batch_normalization/AssignMovingAvg_32А
>basemodel/batch_normalization/AssignMovingAvg_3/ReadVariableOp>basemodel/batch_normalization/AssignMovingAvg_3/ReadVariableOp2p
6basemodel/batch_normalization/batchnorm/ReadVariableOp6basemodel/batch_normalization/batchnorm/ReadVariableOp2x
:basemodel/batch_normalization/batchnorm/mul/ReadVariableOp:basemodel/batch_normalization/batchnorm/mul/ReadVariableOp2t
8basemodel/batch_normalization/batchnorm_1/ReadVariableOp8basemodel/batch_normalization/batchnorm_1/ReadVariableOp2|
<basemodel/batch_normalization/batchnorm_1/mul/ReadVariableOp<basemodel/batch_normalization/batchnorm_1/mul/ReadVariableOp2b
/basemodel/batch_normalization_1/AssignMovingAvg/basemodel/batch_normalization_1/AssignMovingAvg2А
>basemodel/batch_normalization_1/AssignMovingAvg/ReadVariableOp>basemodel/batch_normalization_1/AssignMovingAvg/ReadVariableOp2f
1basemodel/batch_normalization_1/AssignMovingAvg_11basemodel/batch_normalization_1/AssignMovingAvg_12Д
@basemodel/batch_normalization_1/AssignMovingAvg_1/ReadVariableOp@basemodel/batch_normalization_1/AssignMovingAvg_1/ReadVariableOp2f
1basemodel/batch_normalization_1/AssignMovingAvg_21basemodel/batch_normalization_1/AssignMovingAvg_22Д
@basemodel/batch_normalization_1/AssignMovingAvg_2/ReadVariableOp@basemodel/batch_normalization_1/AssignMovingAvg_2/ReadVariableOp2f
1basemodel/batch_normalization_1/AssignMovingAvg_31basemodel/batch_normalization_1/AssignMovingAvg_32Д
@basemodel/batch_normalization_1/AssignMovingAvg_3/ReadVariableOp@basemodel/batch_normalization_1/AssignMovingAvg_3/ReadVariableOp2t
8basemodel/batch_normalization_1/batchnorm/ReadVariableOp8basemodel/batch_normalization_1/batchnorm/ReadVariableOp2|
<basemodel/batch_normalization_1/batchnorm/mul/ReadVariableOp<basemodel/batch_normalization_1/batchnorm/mul/ReadVariableOp2x
:basemodel/batch_normalization_1/batchnorm_1/ReadVariableOp:basemodel/batch_normalization_1/batchnorm_1/ReadVariableOp2А
>basemodel/batch_normalization_1/batchnorm_1/mul/ReadVariableOp>basemodel/batch_normalization_1/batchnorm_1/mul/ReadVariableOp2b
/basemodel/batch_normalization_2/AssignMovingAvg/basemodel/batch_normalization_2/AssignMovingAvg2А
>basemodel/batch_normalization_2/AssignMovingAvg/ReadVariableOp>basemodel/batch_normalization_2/AssignMovingAvg/ReadVariableOp2f
1basemodel/batch_normalization_2/AssignMovingAvg_11basemodel/batch_normalization_2/AssignMovingAvg_12Д
@basemodel/batch_normalization_2/AssignMovingAvg_1/ReadVariableOp@basemodel/batch_normalization_2/AssignMovingAvg_1/ReadVariableOp2f
1basemodel/batch_normalization_2/AssignMovingAvg_21basemodel/batch_normalization_2/AssignMovingAvg_22Д
@basemodel/batch_normalization_2/AssignMovingAvg_2/ReadVariableOp@basemodel/batch_normalization_2/AssignMovingAvg_2/ReadVariableOp2f
1basemodel/batch_normalization_2/AssignMovingAvg_31basemodel/batch_normalization_2/AssignMovingAvg_32Д
@basemodel/batch_normalization_2/AssignMovingAvg_3/ReadVariableOp@basemodel/batch_normalization_2/AssignMovingAvg_3/ReadVariableOp2t
8basemodel/batch_normalization_2/batchnorm/ReadVariableOp8basemodel/batch_normalization_2/batchnorm/ReadVariableOp2|
<basemodel/batch_normalization_2/batchnorm/mul/ReadVariableOp<basemodel/batch_normalization_2/batchnorm/mul/ReadVariableOp2x
:basemodel/batch_normalization_2/batchnorm_1/ReadVariableOp:basemodel/batch_normalization_2/batchnorm_1/ReadVariableOp2А
>basemodel/batch_normalization_2/batchnorm_1/mul/ReadVariableOp>basemodel/batch_normalization_2/batchnorm_1/mul/ReadVariableOp2b
/basemodel/batch_normalization_3/AssignMovingAvg/basemodel/batch_normalization_3/AssignMovingAvg2А
>basemodel/batch_normalization_3/AssignMovingAvg/ReadVariableOp>basemodel/batch_normalization_3/AssignMovingAvg/ReadVariableOp2f
1basemodel/batch_normalization_3/AssignMovingAvg_11basemodel/batch_normalization_3/AssignMovingAvg_12Д
@basemodel/batch_normalization_3/AssignMovingAvg_1/ReadVariableOp@basemodel/batch_normalization_3/AssignMovingAvg_1/ReadVariableOp2f
1basemodel/batch_normalization_3/AssignMovingAvg_21basemodel/batch_normalization_3/AssignMovingAvg_22Д
@basemodel/batch_normalization_3/AssignMovingAvg_2/ReadVariableOp@basemodel/batch_normalization_3/AssignMovingAvg_2/ReadVariableOp2f
1basemodel/batch_normalization_3/AssignMovingAvg_31basemodel/batch_normalization_3/AssignMovingAvg_32Д
@basemodel/batch_normalization_3/AssignMovingAvg_3/ReadVariableOp@basemodel/batch_normalization_3/AssignMovingAvg_3/ReadVariableOp2t
8basemodel/batch_normalization_3/batchnorm/ReadVariableOp8basemodel/batch_normalization_3/batchnorm/ReadVariableOp2|
<basemodel/batch_normalization_3/batchnorm/mul/ReadVariableOp<basemodel/batch_normalization_3/batchnorm/mul/ReadVariableOp2x
:basemodel/batch_normalization_3/batchnorm_1/ReadVariableOp:basemodel/batch_normalization_3/batchnorm_1/ReadVariableOp2А
>basemodel/batch_normalization_3/batchnorm_1/mul/ReadVariableOp>basemodel/batch_normalization_3/batchnorm_1/mul/ReadVariableOp2T
(basemodel/dense_1/BiasAdd/ReadVariableOp(basemodel/dense_1/BiasAdd/ReadVariableOp2X
*basemodel/dense_1/BiasAdd_1/ReadVariableOp*basemodel/dense_1/BiasAdd_1/ReadVariableOp2R
'basemodel/dense_1/MatMul/ReadVariableOp'basemodel/dense_1/MatMul/ReadVariableOp2V
)basemodel/dense_1/MatMul_1/ReadVariableOp)basemodel/dense_1/MatMul_1/ReadVariableOp2d
0basemodel/stream_0_conv_1/BiasAdd/ReadVariableOp0basemodel/stream_0_conv_1/BiasAdd/ReadVariableOp2h
2basemodel/stream_0_conv_1/BiasAdd_1/ReadVariableOp2basemodel/stream_0_conv_1/BiasAdd_1/ReadVariableOp2|
<basemodel/stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp<basemodel/stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp2А
>basemodel/stream_0_conv_1/conv1d_1/ExpandDims_1/ReadVariableOp>basemodel/stream_0_conv_1/conv1d_1/ExpandDims_1/ReadVariableOp2d
0basemodel/stream_1_conv_1/BiasAdd/ReadVariableOp0basemodel/stream_1_conv_1/BiasAdd/ReadVariableOp2h
2basemodel/stream_1_conv_1/BiasAdd_1/ReadVariableOp2basemodel/stream_1_conv_1/BiasAdd_1/ReadVariableOp2|
<basemodel/stream_1_conv_1/conv1d/ExpandDims_1/ReadVariableOp<basemodel/stream_1_conv_1/conv1d/ExpandDims_1/ReadVariableOp2А
>basemodel/stream_1_conv_1/conv1d_1/ExpandDims_1/ReadVariableOp>basemodel/stream_1_conv_1/conv1d_1/ExpandDims_1/ReadVariableOp2d
0basemodel/stream_2_conv_1/BiasAdd/ReadVariableOp0basemodel/stream_2_conv_1/BiasAdd/ReadVariableOp2h
2basemodel/stream_2_conv_1/BiasAdd_1/ReadVariableOp2basemodel/stream_2_conv_1/BiasAdd_1/ReadVariableOp2|
<basemodel/stream_2_conv_1/conv1d/ExpandDims_1/ReadVariableOp<basemodel/stream_2_conv_1/conv1d/ExpandDims_1/ReadVariableOp2А
>basemodel/stream_2_conv_1/conv1d_1/ExpandDims_1/ReadVariableOp>basemodel/stream_2_conv_1/conv1d_1/ExpandDims_1/ReadVariableOp2^
-dense_1/kernel/Regularizer/Abs/ReadVariableOp-dense_1/kernel/Regularizer/Abs/ReadVariableOp2n
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp2t
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp2n
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp:U Q
+
_output_shapes
:€€€€€€€€€}
"
_user_specified_name
inputs/0:UQ
+
_output_shapes
:€€€€€€€€€}
"
_user_specified_name
inputs/1
њ
i
0__inference_stream_1_drop_1_layer_call_fn_108447

inputs
identityИҐStatefulPartitionedCallи
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€}@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_stream_1_drop_1_layer_call_and_return_conditional_losses_1047602
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€}@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€}@22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:€€€€€€€€€}@
 
_user_specified_nameinputs
с
n
O__inference_stream_1_input_drop_layer_call_and_return_conditional_losses_105057

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †?2
dropout/Constw
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:€€€€€€€€€}2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape”
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:€€€€€€€€€}*
dtype0*
seedЈ*
seed2Ј2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL>2
dropout/GreaterEqual/y¬
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:€€€€€€€€€}2
dropout/GreaterEqualГ
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:€€€€€€€€€}2
dropout/Cast~
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:€€€€€€€€€}2
dropout/Mul_1i
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:€€€€€€€€€}2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€}:S O
+
_output_shapes
:€€€€€€€€€}
 
_user_specified_nameinputs
Љ
r
V__inference_global_average_pooling1d_2_layer_call_and_return_conditional_losses_104093

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
:€€€€€€€€€€€€€€€€€€2
Meanj
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€:e a
=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
Н	
—
6__inference_batch_normalization_1_layer_call_fn_108177

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityИҐStatefulPartitionedCall©
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1037952
StatefulPartitionedCallИ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:€€€€€€€€€€€€€€€€€€@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs
Ѕ	
p
D__inference_distance_layer_call_and_return_conditional_losses_107682
inputs_0
inputs_1
identityW
subSubinputs_0inputs_1*
T0*'
_output_shapes
:€€€€€€€€€T2
subU
SquareSquaresub:z:0*
T0*'
_output_shapes
:€€€€€€€€€T2
Squarey
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2
Sum/reduction_indicesА
SumSum
Square:y:0Sum/reduction_indices:output:0*
T0*'
_output_shapes
:€€€€€€€€€*
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
:€€€€€€€€€2	
MaximumS
SqrtSqrtMaximum:z:0*
T0*'
_output_shapes
:€€€€€€€€€2
Sqrt\
IdentityIdentitySqrt:y:0*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:€€€€€€€€€T:€€€€€€€€€T:Q M
'
_output_shapes
:€€€€€€€€€T
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:€€€€€€€€€T
"
_user_specified_name
inputs/1
П	
—
6__inference_batch_normalization_2_layer_call_fn_108324

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityИҐStatefulPartitionedCallЂ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1038972
StatefulPartitionedCallИ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:€€€€€€€€€€€€€€€€€€@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs
ЏI
є	
A__inference_model_layer_call_and_return_conditional_losses_106223
left_inputs
right_inputs&
basemodel_106123:@
basemodel_106125:@&
basemodel_106127:@
basemodel_106129:@&
basemodel_106131:@
basemodel_106133:@
basemodel_106135:@
basemodel_106137:@
basemodel_106139:@
basemodel_106141:@
basemodel_106143:@
basemodel_106145:@
basemodel_106147:@
basemodel_106149:@
basemodel_106151:@
basemodel_106153:@
basemodel_106155:@
basemodel_106157:@#
basemodel_106159:	јT
basemodel_106161:T
basemodel_106163:T
basemodel_106165:T
basemodel_106167:T
basemodel_106169:T
identityИҐ!basemodel/StatefulPartitionedCallҐ#basemodel/StatefulPartitionedCall_1Ґ-dense_1/kernel/Regularizer/Abs/ReadVariableOpҐ5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpҐ8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpҐ5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOpн
!basemodel/StatefulPartitionedCallStatefulPartitionedCallleft_inputsleft_inputsleft_inputsbasemodel_106123basemodel_106125basemodel_106127basemodel_106129basemodel_106131basemodel_106133basemodel_106135basemodel_106137basemodel_106139basemodel_106141basemodel_106143basemodel_106145basemodel_106147basemodel_106149basemodel_106151basemodel_106153basemodel_106155basemodel_106157basemodel_106159basemodel_106161basemodel_106163basemodel_106165basemodel_106167basemodel_106169*&
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€T*2
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_basemodel_layer_call_and_return_conditional_losses_1052432#
!basemodel/StatefulPartitionedCallШ
#basemodel/StatefulPartitionedCall_1StatefulPartitionedCallright_inputsright_inputsright_inputsbasemodel_106123basemodel_106125basemodel_106127basemodel_106129basemodel_106131basemodel_106133basemodel_106135basemodel_106137basemodel_106139basemodel_106141basemodel_106143basemodel_106145basemodel_106147basemodel_106149basemodel_106151basemodel_106153basemodel_106155basemodel_106157basemodel_106159basemodel_106161basemodel_106163basemodel_106165basemodel_106167basemodel_106169"^basemodel/StatefulPartitionedCall*&
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€T*2
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_basemodel_layer_call_and_return_conditional_losses_1052432%
#basemodel/StatefulPartitionedCall_1™
distance/PartitionedCallPartitionedCall*basemodel/StatefulPartitionedCall:output:0,basemodel/StatefulPartitionedCall_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_distance_layer_call_and_return_conditional_losses_1057462
distance/PartitionedCall√
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpbasemodel_106131*"
_output_shapes
:@*
dtype027
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp√
&stream_0_conv_1/kernel/Regularizer/AbsAbs=stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_0_conv_1/kernel/Regularizer/Abs©
(stream_0_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_0_conv_1/kernel/Regularizer/Const„
&stream_0_conv_1/kernel/Regularizer/SumSum*stream_0_conv_1/kernel/Regularizer/Abs:y:01stream_0_conv_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/SumЩ
(stream_0_conv_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2*
(stream_0_conv_1/kernel/Regularizer/mul/x№
&stream_0_conv_1/kernel/Regularizer/mulMul1stream_0_conv_1/kernel/Regularizer/mul/x:output:0/stream_0_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/mul…
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpbasemodel_106127*"
_output_shapes
:@*
dtype02:
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpѕ
)stream_1_conv_1/kernel/Regularizer/SquareSquare@stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2+
)stream_1_conv_1/kernel/Regularizer/Square©
(stream_1_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_1_conv_1/kernel/Regularizer/ConstЏ
&stream_1_conv_1/kernel/Regularizer/SumSum-stream_1_conv_1/kernel/Regularizer/Square:y:01stream_1_conv_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&stream_1_conv_1/kernel/Regularizer/SumЩ
(stream_1_conv_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2*
(stream_1_conv_1/kernel/Regularizer/mul/x№
&stream_1_conv_1/kernel/Regularizer/mulMul1stream_1_conv_1/kernel/Regularizer/mul/x:output:0/stream_1_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_1_conv_1/kernel/Regularizer/mul√
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpbasemodel_106123*"
_output_shapes
:@*
dtype027
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp√
&stream_2_conv_1/kernel/Regularizer/AbsAbs=stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_2_conv_1/kernel/Regularizer/Abs©
(stream_2_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_2_conv_1/kernel/Regularizer/Const„
&stream_2_conv_1/kernel/Regularizer/SumSum*stream_2_conv_1/kernel/Regularizer/Abs:y:01stream_2_conv_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&stream_2_conv_1/kernel/Regularizer/SumЩ
(stream_2_conv_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2*
(stream_2_conv_1/kernel/Regularizer/mul/x№
&stream_2_conv_1/kernel/Regularizer/mulMul1stream_2_conv_1/kernel/Regularizer/mul/x:output:0/stream_2_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_2_conv_1/kernel/Regularizer/mul∞
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpbasemodel_106159*
_output_shapes
:	јT*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOp®
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	јT2 
dense_1/kernel/Regularizer/AbsХ
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_1/kernel/Regularizer/ConstЈ
dense_1/kernel/Regularizer/SumSum"dense_1/kernel/Regularizer/Abs:y:0)dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/SumЙ
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2"
 dense_1/kernel/Regularizer/mul/xЉ
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/mul|
IdentityIdentity!distance/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identityу
NoOpNoOp"^basemodel/StatefulPartitionedCall$^basemodel/StatefulPartitionedCall_1.^dense_1/kernel/Regularizer/Abs/ReadVariableOp6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp9^stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp6^stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*q
_input_shapes`
^:€€€€€€€€€}:€€€€€€€€€}: : : : : : : : : : : : : : : : : : : : : : : : 2F
!basemodel/StatefulPartitionedCall!basemodel/StatefulPartitionedCall2J
#basemodel/StatefulPartitionedCall_1#basemodel/StatefulPartitionedCall_12^
-dense_1/kernel/Regularizer/Abs/ReadVariableOp-dense_1/kernel/Regularizer/Abs/ReadVariableOp2n
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp2t
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp2n
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp:X T
+
_output_shapes
:€€€€€€€€€}
%
_user_specified_nameleft_inputs:YU
+
_output_shapes
:€€€€€€€€€}
&
_user_specified_nameright_inputs
З
Ѓ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_107957

inputs/
!batchnorm_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@1
#batchnorm_readvariableop_1_resource:@1
#batchnorm_readvariableop_2_resource:@
identityИҐbatchnorm/ReadVariableOpҐbatchnorm/ReadVariableOp_1Ґbatchnorm/ReadVariableOp_2Ґbatchnorm/mul/ReadVariableOpТ
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
 *oГ:2
batchnorm/add/yИ
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2
batchnorm/mulz
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€}@2
batchnorm/mul_1Ш
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp_1Е
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:@2
batchnorm/mul_2Ш
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp_2Г
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2
batchnorm/subЙ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*+
_output_shapes
:€€€€€€€€€}@2
batchnorm/add_1r
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€}@2

Identity¬
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€}@: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:S O
+
_output_shapes
:€€€€€€€€€}@
 
_user_specified_nameinputs
Щ
ф
$__inference_signature_wrapper_106309
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

unknown_17:	јT

unknown_18:T

unknown_19:T

unknown_20:T

unknown_21:T

unknown_22:T
identityИҐStatefulPartitionedCallУ
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
:€€€€€€€€€*:
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В **
f%R#
!__inference__wrapped_model_1035492
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*q
_input_shapes`
^:€€€€€€€€€}:€€€€€€€€€}: : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
+
_output_shapes
:€€€€€€€€€}
%
_user_specified_nameleft_inputs:YU
+
_output_shapes
:€€€€€€€€€}
&
_user_specified_nameright_inputs
К
g
K__inference_dense_1_dropout_layer_call_and_return_conditional_losses_104691

inputs
identity[
IdentityIdentityinputs*
T0*(
_output_shapes
:€€€€€€€€€ј2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:€€€€€€€€€ј:P L
(
_output_shapes
:€€€€€€€€€ј
 
_user_specified_nameinputs
„
I
-__inference_activation_1_layer_call_fn_108383

inputs
identityЌ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€}@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_activation_1_layer_call_and_return_conditional_losses_1044812
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:€€€€€€€€€}@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€}@:S O
+
_output_shapes
:€€€€€€€€€}@
 
_user_specified_nameinputs
∆I
є	
A__inference_model_layer_call_and_return_conditional_losses_106119
left_inputs
right_inputs&
basemodel_106019:@
basemodel_106021:@&
basemodel_106023:@
basemodel_106025:@&
basemodel_106027:@
basemodel_106029:@
basemodel_106031:@
basemodel_106033:@
basemodel_106035:@
basemodel_106037:@
basemodel_106039:@
basemodel_106041:@
basemodel_106043:@
basemodel_106045:@
basemodel_106047:@
basemodel_106049:@
basemodel_106051:@
basemodel_106053:@#
basemodel_106055:	јT
basemodel_106057:T
basemodel_106059:T
basemodel_106061:T
basemodel_106063:T
basemodel_106065:T
identityИҐ!basemodel/StatefulPartitionedCallҐ#basemodel/StatefulPartitionedCall_1Ґ-dense_1/kernel/Regularizer/Abs/ReadVariableOpҐ5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpҐ8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpҐ5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOpх
!basemodel/StatefulPartitionedCallStatefulPartitionedCallleft_inputsleft_inputsleft_inputsbasemodel_106019basemodel_106021basemodel_106023basemodel_106025basemodel_106027basemodel_106029basemodel_106031basemodel_106033basemodel_106035basemodel_106037basemodel_106039basemodel_106041basemodel_106043basemodel_106045basemodel_106047basemodel_106049basemodel_106051basemodel_106053basemodel_106055basemodel_106057basemodel_106059basemodel_106061basemodel_106063basemodel_106065*&
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€T*:
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_basemodel_layer_call_and_return_conditional_losses_1046122#
!basemodel/StatefulPartitionedCallь
#basemodel/StatefulPartitionedCall_1StatefulPartitionedCallright_inputsright_inputsright_inputsbasemodel_106019basemodel_106021basemodel_106023basemodel_106025basemodel_106027basemodel_106029basemodel_106031basemodel_106033basemodel_106035basemodel_106037basemodel_106039basemodel_106041basemodel_106043basemodel_106045basemodel_106047basemodel_106049basemodel_106051basemodel_106053basemodel_106055basemodel_106057basemodel_106059basemodel_106061basemodel_106063basemodel_106065*&
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€T*:
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_basemodel_layer_call_and_return_conditional_losses_1046122%
#basemodel/StatefulPartitionedCall_1™
distance/PartitionedCallPartitionedCall*basemodel/StatefulPartitionedCall:output:0,basemodel/StatefulPartitionedCall_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_distance_layer_call_and_return_conditional_losses_1056462
distance/PartitionedCall√
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpbasemodel_106027*"
_output_shapes
:@*
dtype027
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp√
&stream_0_conv_1/kernel/Regularizer/AbsAbs=stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_0_conv_1/kernel/Regularizer/Abs©
(stream_0_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_0_conv_1/kernel/Regularizer/Const„
&stream_0_conv_1/kernel/Regularizer/SumSum*stream_0_conv_1/kernel/Regularizer/Abs:y:01stream_0_conv_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/SumЩ
(stream_0_conv_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2*
(stream_0_conv_1/kernel/Regularizer/mul/x№
&stream_0_conv_1/kernel/Regularizer/mulMul1stream_0_conv_1/kernel/Regularizer/mul/x:output:0/stream_0_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/mul…
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpbasemodel_106023*"
_output_shapes
:@*
dtype02:
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpѕ
)stream_1_conv_1/kernel/Regularizer/SquareSquare@stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2+
)stream_1_conv_1/kernel/Regularizer/Square©
(stream_1_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_1_conv_1/kernel/Regularizer/ConstЏ
&stream_1_conv_1/kernel/Regularizer/SumSum-stream_1_conv_1/kernel/Regularizer/Square:y:01stream_1_conv_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&stream_1_conv_1/kernel/Regularizer/SumЩ
(stream_1_conv_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2*
(stream_1_conv_1/kernel/Regularizer/mul/x№
&stream_1_conv_1/kernel/Regularizer/mulMul1stream_1_conv_1/kernel/Regularizer/mul/x:output:0/stream_1_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_1_conv_1/kernel/Regularizer/mul√
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpbasemodel_106019*"
_output_shapes
:@*
dtype027
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp√
&stream_2_conv_1/kernel/Regularizer/AbsAbs=stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_2_conv_1/kernel/Regularizer/Abs©
(stream_2_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_2_conv_1/kernel/Regularizer/Const„
&stream_2_conv_1/kernel/Regularizer/SumSum*stream_2_conv_1/kernel/Regularizer/Abs:y:01stream_2_conv_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&stream_2_conv_1/kernel/Regularizer/SumЩ
(stream_2_conv_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2*
(stream_2_conv_1/kernel/Regularizer/mul/x№
&stream_2_conv_1/kernel/Regularizer/mulMul1stream_2_conv_1/kernel/Regularizer/mul/x:output:0/stream_2_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_2_conv_1/kernel/Regularizer/mul∞
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpbasemodel_106055*
_output_shapes
:	јT*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOp®
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	јT2 
dense_1/kernel/Regularizer/AbsХ
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_1/kernel/Regularizer/ConstЈ
dense_1/kernel/Regularizer/SumSum"dense_1/kernel/Regularizer/Abs:y:0)dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/SumЙ
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2"
 dense_1/kernel/Regularizer/mul/xЉ
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/mul|
IdentityIdentity!distance/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identityу
NoOpNoOp"^basemodel/StatefulPartitionedCall$^basemodel/StatefulPartitionedCall_1.^dense_1/kernel/Regularizer/Abs/ReadVariableOp6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp9^stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp6^stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*q
_input_shapes`
^:€€€€€€€€€}:€€€€€€€€€}: : : : : : : : : : : : : : : : : : : : : : : : 2F
!basemodel/StatefulPartitionedCall!basemodel/StatefulPartitionedCall2J
#basemodel/StatefulPartitionedCall_1#basemodel/StatefulPartitionedCall_12^
-dense_1/kernel/Regularizer/Abs/ReadVariableOp-dense_1/kernel/Regularizer/Abs/ReadVariableOp2n
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp2t
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp2n
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp:X T
+
_output_shapes
:€€€€€€€€€}
%
_user_specified_nameleft_inputs:YU
+
_output_shapes
:€€€€€€€€€}
&
_user_specified_nameright_inputs
с
n
O__inference_stream_1_input_drop_layer_call_and_return_conditional_losses_107738

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †?2
dropout/Constw
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:€€€€€€€€€}2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape”
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:€€€€€€€€€}*
dtype0*
seedЈ*
seed2Ј2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL>2
dropout/GreaterEqual/y¬
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:€€€€€€€€€}2
dropout/GreaterEqualГ
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:€€€€€€€€€}2
dropout/Cast~
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:€€€€€€€€€}2
dropout/Mul_1i
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:€€€€€€€€€}2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€}:S O
+
_output_shapes
:€€€€€€€€€}
 
_user_specified_nameinputs
И
i
K__inference_stream_1_drop_1_layer_call_and_return_conditional_losses_108425

inputs

identity_1^
IdentityIdentityinputs*
T0*+
_output_shapes
:€€€€€€€€€}@2

Identitym

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:€€€€€€€€€}@2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€}@:S O
+
_output_shapes
:€€€€€€€€€}@
 
_user_specified_nameinputs
ы
’
K__inference_stream_1_conv_1_layer_call_and_return_conditional_losses_107838

inputsA
+conv1d_expanddims_1_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identityИҐBiasAdd/ReadVariableOpҐ"conv1d/ExpandDims_1/ReadVariableOpҐ8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€2
conv1d/ExpandDims/dimЦ
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:€€€€€€€€€}2
conv1d/ExpandDimsЄ
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
conv1d/ExpandDims_1/dimЈ
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@2
conv1d/ExpandDims_1ґ
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:€€€€€€€€€}@*
paddingSAME*
strides
2
conv1dТ
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:€€€€€€€€€}@*
squeeze_dims

э€€€€€€€€2
conv1d/SqueezeМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpМ
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€}@2	
BiasAddд
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02:
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpѕ
)stream_1_conv_1/kernel/Regularizer/SquareSquare@stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2+
)stream_1_conv_1/kernel/Regularizer/Square©
(stream_1_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_1_conv_1/kernel/Regularizer/ConstЏ
&stream_1_conv_1/kernel/Regularizer/SumSum-stream_1_conv_1/kernel/Regularizer/Square:y:01stream_1_conv_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&stream_1_conv_1/kernel/Regularizer/SumЩ
(stream_1_conv_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2*
(stream_1_conv_1/kernel/Regularizer/mul/x№
&stream_1_conv_1/kernel/Regularizer/mulMul1stream_1_conv_1/kernel/Regularizer/mul/x:output:0/stream_1_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_1_conv_1/kernel/Regularizer/mulo
IdentityIdentityBiasAdd:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€}@2

Identity«
NoOpNoOp^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp9^stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€}: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp2t
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp:S O
+
_output_shapes
:€€€€€€€€€}
 
_user_specified_nameinputs
К
g
K__inference_dense_1_dropout_layer_call_and_return_conditional_losses_108564

inputs
identity[
IdentityIdentityinputs*
T0*(
_output_shapes
:€€€€€€€€€ј2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:€€€€€€€€€ј:P L
(
_output_shapes
:€€€€€€€€€ј
 
_user_specified_nameinputs
л
W
;__inference_global_average_pooling1d_1_layer_call_fn_108518

inputs
identity„
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *_
fZRX
V__inference_global_average_pooling1d_1_layer_call_and_return_conditional_losses_1045232
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€}@:S O
+
_output_shapes
:€€€€€€€€€}@
 
_user_specified_nameinputs
—
L
0__inference_dense_1_dropout_layer_call_fn_108574

inputs
identityЌ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€ј* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_dense_1_dropout_layer_call_and_return_conditional_losses_1046912
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:€€€€€€€€€ј2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:€€€€€€€€€ј:P L
(
_output_shapes
:€€€€€€€€€ј
 
_user_specified_nameinputs
р
d
H__inference_activation_2_layer_call_and_return_conditional_losses_108388

inputs
identityR
ReluReluinputs*
T0*+
_output_shapes
:€€€€€€€€€}@2
Reluj
IdentityIdentityRelu:activations:0*
T0*+
_output_shapes
:€€€€€€€€€}@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€}@:S O
+
_output_shapes
:€€€€€€€€€}@
 
_user_specified_nameinputs
Љ
r
V__inference_global_average_pooling1d_1_layer_call_and_return_conditional_losses_104069

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
:€€€€€€€€€€€€€€€€€€2
Meanj
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€:e a
=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
Ж
r
V__inference_global_average_pooling1d_2_layer_call_and_return_conditional_losses_104530

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
:€€€€€€€€€@2
Meana
IdentityIdentityMean:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€}@:S O
+
_output_shapes
:€€€€€€€€€}@
 
_user_specified_nameinputs
ВХ
Ѓ
E__inference_basemodel_layer_call_and_return_conditional_losses_104612

inputs
inputs_1
inputs_2,
stream_2_conv_1_104323:@$
stream_2_conv_1_104325:@,
stream_1_conv_1_104350:@$
stream_1_conv_1_104352:@,
stream_0_conv_1_104377:@$
stream_0_conv_1_104379:@*
batch_normalization_2_104402:@*
batch_normalization_2_104404:@*
batch_normalization_2_104406:@*
batch_normalization_2_104408:@*
batch_normalization_1_104431:@*
batch_normalization_1_104433:@*
batch_normalization_1_104435:@*
batch_normalization_1_104437:@(
batch_normalization_104460:@(
batch_normalization_104462:@(
batch_normalization_104464:@(
batch_normalization_104466:@!
dense_1_104566:	јT
dense_1_104568:T*
batch_normalization_3_104571:T*
batch_normalization_3_104573:T*
batch_normalization_3_104575:T*
batch_normalization_3_104577:T
identityИҐ+batch_normalization/StatefulPartitionedCallҐ-batch_normalization_1/StatefulPartitionedCallҐ-batch_normalization_2/StatefulPartitionedCallҐ-batch_normalization_3/StatefulPartitionedCallҐdense_1/StatefulPartitionedCallҐ-dense_1/kernel/Regularizer/Abs/ReadVariableOpҐ'stream_0_conv_1/StatefulPartitionedCallҐ5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpҐ'stream_1_conv_1/StatefulPartitionedCallҐ8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpҐ'stream_2_conv_1/StatefulPartitionedCallҐ5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOpю
#stream_2_input_drop/PartitionedCallPartitionedCallinputs_2*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€}* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_stream_2_input_drop_layer_call_and_return_conditional_losses_1042852%
#stream_2_input_drop/PartitionedCallю
#stream_1_input_drop/PartitionedCallPartitionedCallinputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€}* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_stream_1_input_drop_layer_call_and_return_conditional_losses_1042922%
#stream_1_input_drop/PartitionedCallь
#stream_0_input_drop/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€}* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_stream_0_input_drop_layer_call_and_return_conditional_losses_1042992%
#stream_0_input_drop/PartitionedCallд
'stream_2_conv_1/StatefulPartitionedCallStatefulPartitionedCall,stream_2_input_drop/PartitionedCall:output:0stream_2_conv_1_104323stream_2_conv_1_104325*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€}@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_stream_2_conv_1_layer_call_and_return_conditional_losses_1043222)
'stream_2_conv_1/StatefulPartitionedCallд
'stream_1_conv_1/StatefulPartitionedCallStatefulPartitionedCall,stream_1_input_drop/PartitionedCall:output:0stream_1_conv_1_104350stream_1_conv_1_104352*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€}@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_stream_1_conv_1_layer_call_and_return_conditional_losses_1043492)
'stream_1_conv_1/StatefulPartitionedCallд
'stream_0_conv_1/StatefulPartitionedCallStatefulPartitionedCall,stream_0_input_drop/PartitionedCall:output:0stream_0_conv_1_104377stream_0_conv_1_104379*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€}@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_stream_0_conv_1_layer_call_and_return_conditional_losses_1043762)
'stream_0_conv_1/StatefulPartitionedCall∆
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall0stream_2_conv_1/StatefulPartitionedCall:output:0batch_normalization_2_104402batch_normalization_2_104404batch_normalization_2_104406batch_normalization_2_104408*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€}@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1044012/
-batch_normalization_2/StatefulPartitionedCall∆
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall0stream_1_conv_1/StatefulPartitionedCall:output:0batch_normalization_1_104431batch_normalization_1_104433batch_normalization_1_104435batch_normalization_1_104437*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€}@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1044302/
-batch_normalization_1/StatefulPartitionedCallЄ
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall0stream_0_conv_1/StatefulPartitionedCall:output:0batch_normalization_104460batch_normalization_104462batch_normalization_104464batch_normalization_104466*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€}@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_1044592-
+batch_normalization/StatefulPartitionedCallЧ
activation_2/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€}@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_activation_2_layer_call_and_return_conditional_losses_1044742
activation_2/PartitionedCallЧ
activation_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€}@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_activation_1_layer_call_and_return_conditional_losses_1044812
activation_1/PartitionedCallП
activation/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€}@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_activation_layer_call_and_return_conditional_losses_1044882
activation/PartitionedCallП
stream_2_drop_1/PartitionedCallPartitionedCall%activation_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€}@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_stream_2_drop_1_layer_call_and_return_conditional_losses_1044952!
stream_2_drop_1/PartitionedCallП
stream_1_drop_1/PartitionedCallPartitionedCall%activation_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€}@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_stream_1_drop_1_layer_call_and_return_conditional_losses_1045022!
stream_1_drop_1/PartitionedCallН
stream_0_drop_1/PartitionedCallPartitionedCall#activation/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€}@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_stream_0_drop_1_layer_call_and_return_conditional_losses_1045092!
stream_0_drop_1/PartitionedCall©
(global_average_pooling1d/PartitionedCallPartitionedCall(stream_0_drop_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *]
fXRV
T__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_1045162*
(global_average_pooling1d/PartitionedCallѓ
*global_average_pooling1d_1/PartitionedCallPartitionedCall(stream_1_drop_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *_
fZRX
V__inference_global_average_pooling1d_1_layer_call_and_return_conditional_losses_1045232,
*global_average_pooling1d_1/PartitionedCallѓ
*global_average_pooling1d_2/PartitionedCallPartitionedCall(stream_2_drop_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *_
fZRX
V__inference_global_average_pooling1d_2_layer_call_and_return_conditional_losses_1045302,
*global_average_pooling1d_2/PartitionedCallш
concatenate/PartitionedCallPartitionedCall1global_average_pooling1d/PartitionedCall:output:03global_average_pooling1d_1/PartitionedCall:output:03global_average_pooling1d_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€ј* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_concatenate_layer_call_and_return_conditional_losses_1045402
concatenate/PartitionedCallЛ
dense_1_dropout/PartitionedCallPartitionedCall$concatenate/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€ј* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_dense_1_dropout_layer_call_and_return_conditional_losses_1045472!
dense_1_dropout/PartitionedCallі
dense_1/StatefulPartitionedCallStatefulPartitionedCall(dense_1_dropout/PartitionedCall:output:0dense_1_104566dense_1_104568*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€T*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_1045652!
dense_1/StatefulPartitionedCallЇ
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0batch_normalization_3_104571batch_normalization_3_104573batch_normalization_3_104575batch_normalization_3_104577*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€T*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1041312/
-batch_normalization_3/StatefulPartitionedCall•
"dense_activation_1/PartitionedCallPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€T* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *W
fRRP
N__inference_dense_activation_1_layer_call_and_return_conditional_losses_1045852$
"dense_activation_1/PartitionedCall…
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpstream_0_conv_1_104377*"
_output_shapes
:@*
dtype027
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp√
&stream_0_conv_1/kernel/Regularizer/AbsAbs=stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_0_conv_1/kernel/Regularizer/Abs©
(stream_0_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_0_conv_1/kernel/Regularizer/Const„
&stream_0_conv_1/kernel/Regularizer/SumSum*stream_0_conv_1/kernel/Regularizer/Abs:y:01stream_0_conv_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/SumЩ
(stream_0_conv_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2*
(stream_0_conv_1/kernel/Regularizer/mul/x№
&stream_0_conv_1/kernel/Regularizer/mulMul1stream_0_conv_1/kernel/Regularizer/mul/x:output:0/stream_0_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/mulѕ
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpstream_1_conv_1_104350*"
_output_shapes
:@*
dtype02:
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpѕ
)stream_1_conv_1/kernel/Regularizer/SquareSquare@stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2+
)stream_1_conv_1/kernel/Regularizer/Square©
(stream_1_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_1_conv_1/kernel/Regularizer/ConstЏ
&stream_1_conv_1/kernel/Regularizer/SumSum-stream_1_conv_1/kernel/Regularizer/Square:y:01stream_1_conv_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&stream_1_conv_1/kernel/Regularizer/SumЩ
(stream_1_conv_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2*
(stream_1_conv_1/kernel/Regularizer/mul/x№
&stream_1_conv_1/kernel/Regularizer/mulMul1stream_1_conv_1/kernel/Regularizer/mul/x:output:0/stream_1_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_1_conv_1/kernel/Regularizer/mul…
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpstream_2_conv_1_104323*"
_output_shapes
:@*
dtype027
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp√
&stream_2_conv_1/kernel/Regularizer/AbsAbs=stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_2_conv_1/kernel/Regularizer/Abs©
(stream_2_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_2_conv_1/kernel/Regularizer/Const„
&stream_2_conv_1/kernel/Regularizer/SumSum*stream_2_conv_1/kernel/Regularizer/Abs:y:01stream_2_conv_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&stream_2_conv_1/kernel/Regularizer/SumЩ
(stream_2_conv_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2*
(stream_2_conv_1/kernel/Regularizer/mul/x№
&stream_2_conv_1/kernel/Regularizer/mulMul1stream_2_conv_1/kernel/Regularizer/mul/x:output:0/stream_2_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_2_conv_1/kernel/Regularizer/mulЃ
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_1_104566*
_output_shapes
:	јT*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOp®
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	јT2 
dense_1/kernel/Regularizer/AbsХ
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_1/kernel/Regularizer/ConstЈ
dense_1/kernel/Regularizer/SumSum"dense_1/kernel/Regularizer/Abs:y:0)dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/SumЙ
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2"
 dense_1/kernel/Regularizer/mul/xЉ
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/mulЖ
IdentityIdentity+dense_activation_1/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€T2

IdentityЗ
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall.^dense_1/kernel/Regularizer/Abs/ReadVariableOp(^stream_0_conv_1/StatefulPartitionedCall6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp(^stream_1_conv_1/StatefulPartitionedCall9^stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp(^stream_2_conv_1/StatefulPartitionedCall6^stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*И
_input_shapesw
u:€€€€€€€€€}:€€€€€€€€€}:€€€€€€€€€}: : : : : : : : : : : : : : : : : : : : : : : : 2Z
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
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp:S O
+
_output_shapes
:€€€€€€€€€}
 
_user_specified_nameinputs:SO
+
_output_shapes
:€€€€€€€€€}
 
_user_specified_nameinputs:SO
+
_output_shapes
:€€€€€€€€€}
 
_user_specified_nameinputs
∆А
и
__inference__traced_save_108952
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

identity_1ИҐMergeV2CheckpointsП
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
Const_1Л
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
ShardedFilename/shard¶
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename™ 
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:@*
dtype0*Љ
value≤Bѓ@B+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/14/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/15/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesЛ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:@*
dtype0*Х
valueЛBИ@B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesщ
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0!savev2_beta_1_read_readvariableop!savev2_beta_2_read_readvariableop savev2_decay_read_readvariableop(savev2_learning_rate_read_readvariableop$savev2_adam_iter_read_readvariableop1savev2_stream_0_conv_1_kernel_read_readvariableop/savev2_stream_0_conv_1_bias_read_readvariableop1savev2_stream_1_conv_1_kernel_read_readvariableop/savev2_stream_1_conv_1_bias_read_readvariableop1savev2_stream_2_conv_1_kernel_read_readvariableop/savev2_stream_2_conv_1_bias_read_readvariableop4savev2_batch_normalization_gamma_read_readvariableop3savev2_batch_normalization_beta_read_readvariableop6savev2_batch_normalization_1_gamma_read_readvariableop5savev2_batch_normalization_1_beta_read_readvariableop6savev2_batch_normalization_2_gamma_read_readvariableop5savev2_batch_normalization_2_beta_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop6savev2_batch_normalization_3_gamma_read_readvariableop5savev2_batch_normalization_3_beta_read_readvariableop:savev2_batch_normalization_moving_mean_read_readvariableop>savev2_batch_normalization_moving_variance_read_readvariableop<savev2_batch_normalization_1_moving_mean_read_readvariableop@savev2_batch_normalization_1_moving_variance_read_readvariableop<savev2_batch_normalization_2_moving_mean_read_readvariableop@savev2_batch_normalization_2_moving_variance_read_readvariableop<savev2_batch_normalization_3_moving_mean_read_readvariableop@savev2_batch_normalization_3_moving_variance_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop8savev2_adam_stream_0_conv_1_kernel_m_read_readvariableop6savev2_adam_stream_0_conv_1_bias_m_read_readvariableop8savev2_adam_stream_1_conv_1_kernel_m_read_readvariableop6savev2_adam_stream_1_conv_1_bias_m_read_readvariableop8savev2_adam_stream_2_conv_1_kernel_m_read_readvariableop6savev2_adam_stream_2_conv_1_bias_m_read_readvariableop;savev2_adam_batch_normalization_gamma_m_read_readvariableop:savev2_adam_batch_normalization_beta_m_read_readvariableop=savev2_adam_batch_normalization_1_gamma_m_read_readvariableop<savev2_adam_batch_normalization_1_beta_m_read_readvariableop=savev2_adam_batch_normalization_2_gamma_m_read_readvariableop<savev2_adam_batch_normalization_2_beta_m_read_readvariableop0savev2_adam_dense_1_kernel_m_read_readvariableop.savev2_adam_dense_1_bias_m_read_readvariableop=savev2_adam_batch_normalization_3_gamma_m_read_readvariableop<savev2_adam_batch_normalization_3_beta_m_read_readvariableop8savev2_adam_stream_0_conv_1_kernel_v_read_readvariableop6savev2_adam_stream_0_conv_1_bias_v_read_readvariableop8savev2_adam_stream_1_conv_1_kernel_v_read_readvariableop6savev2_adam_stream_1_conv_1_bias_v_read_readvariableop8savev2_adam_stream_2_conv_1_kernel_v_read_readvariableop6savev2_adam_stream_2_conv_1_bias_v_read_readvariableop;savev2_adam_batch_normalization_gamma_v_read_readvariableop:savev2_adam_batch_normalization_beta_v_read_readvariableop=savev2_adam_batch_normalization_1_gamma_v_read_readvariableop<savev2_adam_batch_normalization_1_beta_v_read_readvariableop=savev2_adam_batch_normalization_2_gamma_v_read_readvariableop<savev2_adam_batch_normalization_2_beta_v_read_readvariableop0savev2_adam_dense_1_kernel_v_read_readvariableop.savev2_adam_dense_1_bias_v_read_readvariableop=savev2_adam_batch_normalization_3_gamma_v_read_readvariableop<savev2_adam_batch_normalization_3_beta_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *N
dtypesD
B2@	2
SaveV2Ї
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes°
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

identity_1Identity_1:output:0*ќ
_input_shapesЉ
є: : : : : : :@:@:@:@:@:@:@:@:@:@:@:@:	јT:T:T:T:@:@:@:@:@:@:T:T: : :@:@:@:@:@:@:@:@:@:@:@:@:	јT:T:T:T:@:@:@:@:@:@:@:@:@:@:@:@:	јT:T:T:T: 2(
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
:	јT: 
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
:	јT: -
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
:	јT: =
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
н
j
K__inference_stream_0_drop_1_layer_call_and_return_conditional_losses_108410

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †?2
dropout/Constw
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:€€€€€€€€€}@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape”
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:€€€€€€€€€}@*
dtype0*
seedЈ*
seed2Ј2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL>2
dropout/GreaterEqual/y¬
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:€€€€€€€€€}@2
dropout/GreaterEqualГ
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:€€€€€€€€€}@2
dropout/Cast~
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:€€€€€€€€€}@2
dropout/Mul_1i
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:€€€€€€€€€}@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€}@:S O
+
_output_shapes
:€€€€€€€€€}@
 
_user_specified_nameinputs
с
n
O__inference_stream_0_input_drop_layer_call_and_return_conditional_losses_107711

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †?2
dropout/Constw
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:€€€€€€€€€}2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape”
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:€€€€€€€€€}*
dtype0*
seedЈ*
seed2Ј2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL>2
dropout/GreaterEqual/y¬
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:€€€€€€€€€}2
dropout/GreaterEqualГ
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:€€€€€€€€€}2
dropout/Cast~
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:€€€€€€€€€}2
dropout/Mul_1i
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:€€€€€€€€€}2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€}:S O
+
_output_shapes
:€€€€€€€€€}
 
_user_specified_nameinputs
€*
и
O__inference_batch_normalization_layer_call_and_return_conditional_losses_104853

inputs5
'assignmovingavg_readvariableop_resource:@7
)assignmovingavg_1_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@/
!batchnorm_readvariableop_resource:@
identityИҐAssignMovingAvgҐAssignMovingAvg/ReadVariableOpҐAssignMovingAvg_1Ґ AssignMovingAvg_1/ReadVariableOpҐbatchnorm/ReadVariableOpҐbatchnorm/mul/ReadVariableOpС
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indicesУ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(2
moments/meanА
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:@2
moments/StopGradient®
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*+
_output_shapes
:€€€€€€€€€}@2
moments/SquaredDifferenceЩ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indicesґ
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(2
moments/varianceБ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 2
moments/SqueezeЙ
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
„#<2
AssignMovingAvg/decay§
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOpШ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg/subП
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg/mulњ
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
„#<2
AssignMovingAvg_1/decay™
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOp†
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg_1/subЧ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg_1/mul…
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
 *oГ:2
batchnorm/add/yВ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2
batchnorm/mulz
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€}@2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:@2
batchnorm/mul_2Т
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOpБ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2
batchnorm/subЙ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*+
_output_shapes
:€€€€€€€€€}@2
batchnorm/add_1r
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€}@2

Identityт
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€}@: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:S O
+
_output_shapes
:€€€€€€€€€}@
 
_user_specified_nameinputs"®L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*В
serving_defaultо
G
left_inputs8
serving_default_left_inputs:0€€€€€€€€€}
I
right_inputs9
serving_default_right_inputs:0€€€€€€€€€}<
distance0
StatefulPartitionedCall:0€€€€€€€€€tensorflow/serving/predict:ор
і
layer-0
layer-1
layer_with_weights-0
layer-2
layer-3
	optimizer
trainable_variables
	variables
regularization_losses
		keras_api


signatures
+—&call_and_return_all_conditional_losses
“__call__
”_default_save_signature"
_tf_keras_network
"
_tf_keras_input_layer
"
_tf_keras_input_layer
џ
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
%trainable_variables
&	variables
'regularization_losses
(	keras_api
+‘&call_and_return_all_conditional_losses
’__call__"
_tf_keras_network
І
)trainable_variables
*	variables
+regularization_losses
,	keras_api
+÷&call_and_return_all_conditional_losses
„__call__"
_tf_keras_layer
У

-beta_1

.beta_2
	/decay
0learning_rate
1iter2m±3m≤4m≥5mі6mµ7mґ8mЈ9mЄ:mє;mЇ<mї=mЉ>mљ?mЊ@mњAmј2vЅ3v¬4v√5vƒ6v≈7v∆8v«9v»:v…;v <vЋ=vћ>vЌ?vќ@vѕAv–"
	optimizer
Ц
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
÷
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
 "
trackable_list_wrapper
ќ
Jlayer_metrics
Kmetrics
Llayer_regularization_losses
trainable_variables
Mnon_trainable_variables

Nlayers
	variables
regularization_losses
“__call__
”_default_save_signature
+—&call_and_return_all_conditional_losses
'—"call_and_return_conditional_losses"
_generic_user_object
-
Ўserving_default"
signature_map
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
І
Otrainable_variables
P	variables
Qregularization_losses
R	keras_api
+ў&call_and_return_all_conditional_losses
Џ__call__"
_tf_keras_layer
І
Strainable_variables
T	variables
Uregularization_losses
V	keras_api
+џ&call_and_return_all_conditional_losses
№__call__"
_tf_keras_layer
І
Wtrainable_variables
X	variables
Yregularization_losses
Z	keras_api
+Ё&call_and_return_all_conditional_losses
ё__call__"
_tf_keras_layer
љ

2kernel
3bias
[trainable_variables
\	variables
]regularization_losses
^	keras_api
+я&call_and_return_all_conditional_losses
а__call__"
_tf_keras_layer
љ

4kernel
5bias
_trainable_variables
`	variables
aregularization_losses
b	keras_api
+б&call_and_return_all_conditional_losses
в__call__"
_tf_keras_layer
љ

6kernel
7bias
ctrainable_variables
d	variables
eregularization_losses
f	keras_api
+г&call_and_return_all_conditional_losses
д__call__"
_tf_keras_layer
м
gaxis
	8gamma
9beta
Bmoving_mean
Cmoving_variance
htrainable_variables
i	variables
jregularization_losses
k	keras_api
+е&call_and_return_all_conditional_losses
ж__call__"
_tf_keras_layer
м
laxis
	:gamma
;beta
Dmoving_mean
Emoving_variance
mtrainable_variables
n	variables
oregularization_losses
p	keras_api
+з&call_and_return_all_conditional_losses
и__call__"
_tf_keras_layer
м
qaxis
	<gamma
=beta
Fmoving_mean
Gmoving_variance
rtrainable_variables
s	variables
tregularization_losses
u	keras_api
+й&call_and_return_all_conditional_losses
к__call__"
_tf_keras_layer
І
vtrainable_variables
w	variables
xregularization_losses
y	keras_api
+л&call_and_return_all_conditional_losses
м__call__"
_tf_keras_layer
І
ztrainable_variables
{	variables
|regularization_losses
}	keras_api
+н&call_and_return_all_conditional_losses
о__call__"
_tf_keras_layer
©
~trainable_variables
	variables
Аregularization_losses
Б	keras_api
+п&call_and_return_all_conditional_losses
р__call__"
_tf_keras_layer
Ђ
Вtrainable_variables
Г	variables
Дregularization_losses
Е	keras_api
+с&call_and_return_all_conditional_losses
т__call__"
_tf_keras_layer
Ђ
Жtrainable_variables
З	variables
Иregularization_losses
Й	keras_api
+у&call_and_return_all_conditional_losses
ф__call__"
_tf_keras_layer
Ђ
Кtrainable_variables
Л	variables
Мregularization_losses
Н	keras_api
+х&call_and_return_all_conditional_losses
ц__call__"
_tf_keras_layer
Ђ
Оtrainable_variables
П	variables
Рregularization_losses
С	keras_api
+ч&call_and_return_all_conditional_losses
ш__call__"
_tf_keras_layer
Ђ
Тtrainable_variables
У	variables
Фregularization_losses
Х	keras_api
+щ&call_and_return_all_conditional_losses
ъ__call__"
_tf_keras_layer
Ђ
Цtrainable_variables
Ч	variables
Шregularization_losses
Щ	keras_api
+ы&call_and_return_all_conditional_losses
ь__call__"
_tf_keras_layer
Ђ
Ъtrainable_variables
Ы	variables
Ьregularization_losses
Э	keras_api
+э&call_and_return_all_conditional_losses
ю__call__"
_tf_keras_layer
Ђ
Юtrainable_variables
Я	variables
†regularization_losses
°	keras_api
+€&call_and_return_all_conditional_losses
А__call__"
_tf_keras_layer
Ѕ

>kernel
?bias
Ґtrainable_variables
£	variables
§regularization_losses
•	keras_api
+Б&call_and_return_all_conditional_losses
В__call__"
_tf_keras_layer
с
	¶axis
	@gamma
Abeta
Hmoving_mean
Imoving_variance
Іtrainable_variables
®	variables
©regularization_losses
™	keras_api
+Г&call_and_return_all_conditional_losses
Д__call__"
_tf_keras_layer
Ђ
Ђtrainable_variables
ђ	variables
≠regularization_losses
Ѓ	keras_api
+Е&call_and_return_all_conditional_losses
Ж__call__"
_tf_keras_layer
Ц
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
÷
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
@
З0
И1
Й2
К3"
trackable_list_wrapper
µ
ѓlayer_metrics
∞metrics
 ±layer_regularization_losses
%trainable_variables
≤non_trainable_variables
≥layers
&	variables
'regularization_losses
’__call__
+‘&call_and_return_all_conditional_losses
'‘"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
іlayer_metrics
µmetrics
 ґlayer_regularization_losses
)trainable_variables
Јnon_trainable_variables
Єlayers
*	variables
+regularization_losses
„__call__
+÷&call_and_return_all_conditional_losses
'÷"call_and_return_conditional_losses"
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
!:	јT2dense_1/kernel
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
(
є0"
trackable_list_wrapper
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
Їlayer_metrics
їmetrics
 Љlayer_regularization_losses
Otrainable_variables
љnon_trainable_variables
Њlayers
P	variables
Qregularization_losses
Џ__call__
+ў&call_and_return_all_conditional_losses
'ў"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
њlayer_metrics
јmetrics
 Ѕlayer_regularization_losses
Strainable_variables
¬non_trainable_variables
√layers
T	variables
Uregularization_losses
№__call__
+џ&call_and_return_all_conditional_losses
'џ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
ƒlayer_metrics
≈metrics
 ∆layer_regularization_losses
Wtrainable_variables
«non_trainable_variables
»layers
X	variables
Yregularization_losses
ё__call__
+Ё&call_and_return_all_conditional_losses
'Ё"call_and_return_conditional_losses"
_generic_user_object
.
20
31"
trackable_list_wrapper
.
20
31"
trackable_list_wrapper
(
З0"
trackable_list_wrapper
µ
…layer_metrics
 metrics
 Ћlayer_regularization_losses
[trainable_variables
ћnon_trainable_variables
Ќlayers
\	variables
]regularization_losses
а__call__
+я&call_and_return_all_conditional_losses
'я"call_and_return_conditional_losses"
_generic_user_object
.
40
51"
trackable_list_wrapper
.
40
51"
trackable_list_wrapper
(
И0"
trackable_list_wrapper
µ
ќlayer_metrics
ѕmetrics
 –layer_regularization_losses
_trainable_variables
—non_trainable_variables
“layers
`	variables
aregularization_losses
в__call__
+б&call_and_return_all_conditional_losses
'б"call_and_return_conditional_losses"
_generic_user_object
.
60
71"
trackable_list_wrapper
.
60
71"
trackable_list_wrapper
(
Й0"
trackable_list_wrapper
µ
”layer_metrics
‘metrics
 ’layer_regularization_losses
ctrainable_variables
÷non_trainable_variables
„layers
d	variables
eregularization_losses
д__call__
+г&call_and_return_all_conditional_losses
'г"call_and_return_conditional_losses"
_generic_user_object
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
 "
trackable_list_wrapper
µ
Ўlayer_metrics
ўmetrics
 Џlayer_regularization_losses
htrainable_variables
џnon_trainable_variables
№layers
i	variables
jregularization_losses
ж__call__
+е&call_and_return_all_conditional_losses
'е"call_and_return_conditional_losses"
_generic_user_object
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
 "
trackable_list_wrapper
µ
Ёlayer_metrics
ёmetrics
 яlayer_regularization_losses
mtrainable_variables
аnon_trainable_variables
бlayers
n	variables
oregularization_losses
и__call__
+з&call_and_return_all_conditional_losses
'з"call_and_return_conditional_losses"
_generic_user_object
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
 "
trackable_list_wrapper
µ
вlayer_metrics
гmetrics
 дlayer_regularization_losses
rtrainable_variables
еnon_trainable_variables
жlayers
s	variables
tregularization_losses
к__call__
+й&call_and_return_all_conditional_losses
'й"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
зlayer_metrics
иmetrics
 йlayer_regularization_losses
vtrainable_variables
кnon_trainable_variables
лlayers
w	variables
xregularization_losses
м__call__
+л&call_and_return_all_conditional_losses
'л"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
мlayer_metrics
нmetrics
 оlayer_regularization_losses
ztrainable_variables
пnon_trainable_variables
рlayers
{	variables
|regularization_losses
о__call__
+н&call_and_return_all_conditional_losses
'н"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ґ
сlayer_metrics
тmetrics
 уlayer_regularization_losses
~trainable_variables
фnon_trainable_variables
хlayers
	variables
Аregularization_losses
р__call__
+п&call_and_return_all_conditional_losses
'п"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
цlayer_metrics
чmetrics
 шlayer_regularization_losses
Вtrainable_variables
щnon_trainable_variables
ъlayers
Г	variables
Дregularization_losses
т__call__
+с&call_and_return_all_conditional_losses
'с"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
ыlayer_metrics
ьmetrics
 эlayer_regularization_losses
Жtrainable_variables
юnon_trainable_variables
€layers
З	variables
Иregularization_losses
ф__call__
+у&call_and_return_all_conditional_losses
'у"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
Аlayer_metrics
Бmetrics
 Вlayer_regularization_losses
Кtrainable_variables
Гnon_trainable_variables
Дlayers
Л	variables
Мregularization_losses
ц__call__
+х&call_and_return_all_conditional_losses
'х"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
Еlayer_metrics
Жmetrics
 Зlayer_regularization_losses
Оtrainable_variables
Иnon_trainable_variables
Йlayers
П	variables
Рregularization_losses
ш__call__
+ч&call_and_return_all_conditional_losses
'ч"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
Кlayer_metrics
Лmetrics
 Мlayer_regularization_losses
Тtrainable_variables
Нnon_trainable_variables
Оlayers
У	variables
Фregularization_losses
ъ__call__
+щ&call_and_return_all_conditional_losses
'щ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
Пlayer_metrics
Рmetrics
 Сlayer_regularization_losses
Цtrainable_variables
Тnon_trainable_variables
Уlayers
Ч	variables
Шregularization_losses
ь__call__
+ы&call_and_return_all_conditional_losses
'ы"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
Фlayer_metrics
Хmetrics
 Цlayer_regularization_losses
Ъtrainable_variables
Чnon_trainable_variables
Шlayers
Ы	variables
Ьregularization_losses
ю__call__
+э&call_and_return_all_conditional_losses
'э"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
Щlayer_metrics
Ъmetrics
 Ыlayer_regularization_losses
Юtrainable_variables
Ьnon_trainable_variables
Эlayers
Я	variables
†regularization_losses
А__call__
+€&call_and_return_all_conditional_losses
'€"call_and_return_conditional_losses"
_generic_user_object
.
>0
?1"
trackable_list_wrapper
.
>0
?1"
trackable_list_wrapper
(
К0"
trackable_list_wrapper
Є
Юlayer_metrics
Яmetrics
 †layer_regularization_losses
Ґtrainable_variables
°non_trainable_variables
Ґlayers
£	variables
§regularization_losses
В__call__
+Б&call_and_return_all_conditional_losses
'Б"call_and_return_conditional_losses"
_generic_user_object
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
 "
trackable_list_wrapper
Є
£layer_metrics
§metrics
 •layer_regularization_losses
Іtrainable_variables
¶non_trainable_variables
Іlayers
®	variables
©regularization_losses
Д__call__
+Г&call_and_return_all_conditional_losses
'Г"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
®layer_metrics
©metrics
 ™layer_regularization_losses
Ђtrainable_variables
Ђnon_trainable_variables
ђlayers
ђ	variables
≠regularization_losses
Ж__call__
+Е&call_and_return_all_conditional_losses
'Е"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
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
ж
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
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
R

≠total

Ѓcount
ѓ	variables
∞	keras_api"
_tf_keras_metric
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
(
З0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
(
И0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
(
Й0"
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
.
B0
C1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
D0
E1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
F0
G1"
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
(
К0"
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
.
H0
I1"
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
:  (2total
:  (2count
0
≠0
Ѓ1"
trackable_list_wrapper
.
ѓ	variables"
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
&:$	јT2Adam/dense_1/kernel/m
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
&:$	јT2Adam/dense_1/kernel/v
:T2Adam/dense_1/bias/v
.:,T2"Adam/batch_normalization_3/gamma/v
-:+T2!Adam/batch_normalization_3/beta/v
“2ѕ
A__inference_model_layer_call_and_return_conditional_losses_106565
A__inference_model_layer_call_and_return_conditional_losses_107015
A__inference_model_layer_call_and_return_conditional_losses_106119
A__inference_model_layer_call_and_return_conditional_losses_106223ј
Ј≤≥
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
ж2г
&__inference_model_layer_call_fn_105724
&__inference_model_layer_call_fn_107069
&__inference_model_layer_call_fn_107123
&__inference_model_layer_call_fn_106015ј
Ј≤≥
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
ёBџ
!__inference__wrapped_model_103549left_inputsright_inputs"Ш
С≤Н
FullArgSpec
argsЪ 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
в2я
E__inference_basemodel_layer_call_and_return_conditional_losses_107299
E__inference_basemodel_layer_call_and_return_conditional_losses_107548
E__inference_basemodel_layer_call_and_return_conditional_losses_105450
E__inference_basemodel_layer_call_and_return_conditional_losses_105551ј
Ј≤≥
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
ц2у
*__inference_basemodel_layer_call_fn_104663
*__inference_basemodel_layer_call_fn_107603
*__inference_basemodel_layer_call_fn_107658
*__inference_basemodel_layer_call_fn_105349ј
Ј≤≥
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
“2ѕ
D__inference_distance_layer_call_and_return_conditional_losses_107670
D__inference_distance_layer_call_and_return_conditional_losses_107682ј
Ј≤≥
FullArgSpec1
args)Ъ&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaultsЪ

 
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
Ь2Щ
)__inference_distance_layer_call_fn_107688
)__inference_distance_layer_call_fn_107694ј
Ј≤≥
FullArgSpec1
args)Ъ&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaultsЪ

 
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
џBЎ
$__inference_signature_wrapper_106309left_inputsright_inputs"Ф
Н≤Й
FullArgSpec
argsЪ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
№2ў
O__inference_stream_0_input_drop_layer_call_and_return_conditional_losses_107699
O__inference_stream_0_input_drop_layer_call_and_return_conditional_losses_107711і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
¶2£
4__inference_stream_0_input_drop_layer_call_fn_107716
4__inference_stream_0_input_drop_layer_call_fn_107721і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
№2ў
O__inference_stream_1_input_drop_layer_call_and_return_conditional_losses_107726
O__inference_stream_1_input_drop_layer_call_and_return_conditional_losses_107738і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
¶2£
4__inference_stream_1_input_drop_layer_call_fn_107743
4__inference_stream_1_input_drop_layer_call_fn_107748і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
№2ў
O__inference_stream_2_input_drop_layer_call_and_return_conditional_losses_107753
O__inference_stream_2_input_drop_layer_call_and_return_conditional_losses_107765і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
¶2£
4__inference_stream_2_input_drop_layer_call_fn_107770
4__inference_stream_2_input_drop_layer_call_fn_107775і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
х2т
K__inference_stream_0_conv_1_layer_call_and_return_conditional_losses_107802Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
Џ2„
0__inference_stream_0_conv_1_layer_call_fn_107811Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
х2т
K__inference_stream_1_conv_1_layer_call_and_return_conditional_losses_107838Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
Џ2„
0__inference_stream_1_conv_1_layer_call_fn_107847Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
х2т
K__inference_stream_2_conv_1_layer_call_and_return_conditional_losses_107874Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
Џ2„
0__inference_stream_2_conv_1_layer_call_fn_107883Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ю2ы
O__inference_batch_normalization_layer_call_and_return_conditional_losses_107903
O__inference_batch_normalization_layer_call_and_return_conditional_losses_107937
O__inference_batch_normalization_layer_call_and_return_conditional_losses_107957
O__inference_batch_normalization_layer_call_and_return_conditional_losses_107991і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
Т2П
4__inference_batch_normalization_layer_call_fn_108004
4__inference_batch_normalization_layer_call_fn_108017
4__inference_batch_normalization_layer_call_fn_108030
4__inference_batch_normalization_layer_call_fn_108043і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
Ж2Г
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_108063
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_108097
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_108117
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_108151і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
Ъ2Ч
6__inference_batch_normalization_1_layer_call_fn_108164
6__inference_batch_normalization_1_layer_call_fn_108177
6__inference_batch_normalization_1_layer_call_fn_108190
6__inference_batch_normalization_1_layer_call_fn_108203і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
Ж2Г
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_108223
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_108257
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_108277
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_108311і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
Ъ2Ч
6__inference_batch_normalization_2_layer_call_fn_108324
6__inference_batch_normalization_2_layer_call_fn_108337
6__inference_batch_normalization_2_layer_call_fn_108350
6__inference_batch_normalization_2_layer_call_fn_108363і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
р2н
F__inference_activation_layer_call_and_return_conditional_losses_108368Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
’2“
+__inference_activation_layer_call_fn_108373Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
т2п
H__inference_activation_1_layer_call_and_return_conditional_losses_108378Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
„2‘
-__inference_activation_1_layer_call_fn_108383Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
т2п
H__inference_activation_2_layer_call_and_return_conditional_losses_108388Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
„2‘
-__inference_activation_2_layer_call_fn_108393Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
‘2—
K__inference_stream_0_drop_1_layer_call_and_return_conditional_losses_108398
K__inference_stream_0_drop_1_layer_call_and_return_conditional_losses_108410і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
Ю2Ы
0__inference_stream_0_drop_1_layer_call_fn_108415
0__inference_stream_0_drop_1_layer_call_fn_108420і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
‘2—
K__inference_stream_1_drop_1_layer_call_and_return_conditional_losses_108425
K__inference_stream_1_drop_1_layer_call_and_return_conditional_losses_108437і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
Ю2Ы
0__inference_stream_1_drop_1_layer_call_fn_108442
0__inference_stream_1_drop_1_layer_call_fn_108447і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
‘2—
K__inference_stream_2_drop_1_layer_call_and_return_conditional_losses_108452
K__inference_stream_2_drop_1_layer_call_and_return_conditional_losses_108464і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
Ю2Ы
0__inference_stream_2_drop_1_layer_call_fn_108469
0__inference_stream_2_drop_1_layer_call_fn_108474і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
б2ё
T__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_108480
T__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_108486ѓ
¶≤Ґ
FullArgSpec%
argsЪ
jself
jinputs
jmask
varargs
 
varkw
 
defaultsҐ

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
Ђ2®
9__inference_global_average_pooling1d_layer_call_fn_108491
9__inference_global_average_pooling1d_layer_call_fn_108496ѓ
¶≤Ґ
FullArgSpec%
argsЪ
jself
jinputs
jmask
varargs
 
varkw
 
defaultsҐ

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
е2в
V__inference_global_average_pooling1d_1_layer_call_and_return_conditional_losses_108502
V__inference_global_average_pooling1d_1_layer_call_and_return_conditional_losses_108508ѓ
¶≤Ґ
FullArgSpec%
argsЪ
jself
jinputs
jmask
varargs
 
varkw
 
defaultsҐ

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ѓ2ђ
;__inference_global_average_pooling1d_1_layer_call_fn_108513
;__inference_global_average_pooling1d_1_layer_call_fn_108518ѓ
¶≤Ґ
FullArgSpec%
argsЪ
jself
jinputs
jmask
varargs
 
varkw
 
defaultsҐ

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
е2в
V__inference_global_average_pooling1d_2_layer_call_and_return_conditional_losses_108524
V__inference_global_average_pooling1d_2_layer_call_and_return_conditional_losses_108530ѓ
¶≤Ґ
FullArgSpec%
argsЪ
jself
jinputs
jmask
varargs
 
varkw
 
defaultsҐ

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ѓ2ђ
;__inference_global_average_pooling1d_2_layer_call_fn_108535
;__inference_global_average_pooling1d_2_layer_call_fn_108540ѓ
¶≤Ґ
FullArgSpec%
argsЪ
jself
jinputs
jmask
varargs
 
varkw
 
defaultsҐ

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
с2о
G__inference_concatenate_layer_call_and_return_conditional_losses_108548Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
÷2”
,__inference_concatenate_layer_call_fn_108555Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
‘2—
K__inference_dense_1_dropout_layer_call_and_return_conditional_losses_108560
K__inference_dense_1_dropout_layer_call_and_return_conditional_losses_108564і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
Ю2Ы
0__inference_dense_1_dropout_layer_call_fn_108569
0__inference_dense_1_dropout_layer_call_fn_108574і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
н2к
C__inference_dense_1_layer_call_and_return_conditional_losses_108596Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
“2ѕ
(__inference_dense_1_layer_call_fn_108605Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
а2Ё
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_108625
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_108659і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
™2І
6__inference_batch_normalization_3_layer_call_fn_108672
6__inference_batch_normalization_3_layer_call_fn_108685і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
ш2х
N__inference_dense_activation_1_layer_call_and_return_conditional_losses_108690Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
Ё2Џ
3__inference_dense_activation_1_layer_call_fn_108695Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
≥2∞
__inference_loss_fn_0_108706П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ 
≥2∞
__inference_loss_fn_1_108717П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ 
≥2∞
__inference_loss_fn_2_108728П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ 
≥2∞
__inference_loss_fn_3_108739П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ а
!__inference__wrapped_model_103549Ї674523G<F=E:D;C8B9>?I@HAiҐf
_Ґ\
ZЪW
)К&
left_inputs€€€€€€€€€}
*К'
right_inputs€€€€€€€€€}
™ "3™0
.
distance"К
distance€€€€€€€€€ђ
H__inference_activation_1_layer_call_and_return_conditional_losses_108378`3Ґ0
)Ґ&
$К!
inputs€€€€€€€€€}@
™ ")Ґ&
К
0€€€€€€€€€}@
Ъ Д
-__inference_activation_1_layer_call_fn_108383S3Ґ0
)Ґ&
$К!
inputs€€€€€€€€€}@
™ "К€€€€€€€€€}@ђ
H__inference_activation_2_layer_call_and_return_conditional_losses_108388`3Ґ0
)Ґ&
$К!
inputs€€€€€€€€€}@
™ ")Ґ&
К
0€€€€€€€€€}@
Ъ Д
-__inference_activation_2_layer_call_fn_108393S3Ґ0
)Ґ&
$К!
inputs€€€€€€€€€}@
™ "К€€€€€€€€€}@™
F__inference_activation_layer_call_and_return_conditional_losses_108368`3Ґ0
)Ґ&
$К!
inputs€€€€€€€€€}@
™ ")Ґ&
К
0€€€€€€€€€}@
Ъ В
+__inference_activation_layer_call_fn_108373S3Ґ0
)Ґ&
$К!
inputs€€€€€€€€€}@
™ "К€€€€€€€€€}@£
E__inference_basemodel_layer_call_and_return_conditional_losses_105450ў674523G<F=E:D;C8B9>?I@HAХҐС
ЙҐЕ
{Ъx
&К#
inputs_0€€€€€€€€€}
&К#
inputs_1€€€€€€€€€}
&К#
inputs_2€€€€€€€€€}
p 

 
™ "%Ґ"
К
0€€€€€€€€€T
Ъ £
E__inference_basemodel_layer_call_and_return_conditional_losses_105551ў674523FG<=DE:;BC89>?HI@AХҐС
ЙҐЕ
{Ъx
&К#
inputs_0€€€€€€€€€}
&К#
inputs_1€€€€€€€€€}
&К#
inputs_2€€€€€€€€€}
p

 
™ "%Ґ"
К
0€€€€€€€€€T
Ъ £
E__inference_basemodel_layer_call_and_return_conditional_losses_107299ў674523G<F=E:D;C8B9>?I@HAХҐС
ЙҐЕ
{Ъx
&К#
inputs/0€€€€€€€€€}
&К#
inputs/1€€€€€€€€€}
&К#
inputs/2€€€€€€€€€}
p 

 
™ "%Ґ"
К
0€€€€€€€€€T
Ъ £
E__inference_basemodel_layer_call_and_return_conditional_losses_107548ў674523FG<=DE:;BC89>?HI@AХҐС
ЙҐЕ
{Ъx
&К#
inputs/0€€€€€€€€€}
&К#
inputs/1€€€€€€€€€}
&К#
inputs/2€€€€€€€€€}
p

 
™ "%Ґ"
К
0€€€€€€€€€T
Ъ ы
*__inference_basemodel_layer_call_fn_104663ћ674523G<F=E:D;C8B9>?I@HAХҐС
ЙҐЕ
{Ъx
&К#
inputs_0€€€€€€€€€}
&К#
inputs_1€€€€€€€€€}
&К#
inputs_2€€€€€€€€€}
p 

 
™ "К€€€€€€€€€Tы
*__inference_basemodel_layer_call_fn_105349ћ674523FG<=DE:;BC89>?HI@AХҐС
ЙҐЕ
{Ъx
&К#
inputs_0€€€€€€€€€}
&К#
inputs_1€€€€€€€€€}
&К#
inputs_2€€€€€€€€€}
p

 
™ "К€€€€€€€€€Tы
*__inference_basemodel_layer_call_fn_107603ћ674523G<F=E:D;C8B9>?I@HAХҐС
ЙҐЕ
{Ъx
&К#
inputs/0€€€€€€€€€}
&К#
inputs/1€€€€€€€€€}
&К#
inputs/2€€€€€€€€€}
p 

 
™ "К€€€€€€€€€Tы
*__inference_basemodel_layer_call_fn_107658ћ674523FG<=DE:;BC89>?HI@AХҐС
ЙҐЕ
{Ъx
&К#
inputs/0€€€€€€€€€}
&К#
inputs/1€€€€€€€€€}
&К#
inputs/2€€€€€€€€€}
p

 
™ "К€€€€€€€€€T—
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_108063|E:D;@Ґ=
6Ґ3
-К*
inputs€€€€€€€€€€€€€€€€€€@
p 
™ "2Ґ/
(К%
0€€€€€€€€€€€€€€€€€€@
Ъ —
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_108097|DE:;@Ґ=
6Ґ3
-К*
inputs€€€€€€€€€€€€€€€€€€@
p
™ "2Ґ/
(К%
0€€€€€€€€€€€€€€€€€€@
Ъ њ
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_108117jE:D;7Ґ4
-Ґ*
$К!
inputs€€€€€€€€€}@
p 
™ ")Ґ&
К
0€€€€€€€€€}@
Ъ њ
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_108151jDE:;7Ґ4
-Ґ*
$К!
inputs€€€€€€€€€}@
p
™ ")Ґ&
К
0€€€€€€€€€}@
Ъ ©
6__inference_batch_normalization_1_layer_call_fn_108164oE:D;@Ґ=
6Ґ3
-К*
inputs€€€€€€€€€€€€€€€€€€@
p 
™ "%К"€€€€€€€€€€€€€€€€€€@©
6__inference_batch_normalization_1_layer_call_fn_108177oDE:;@Ґ=
6Ґ3
-К*
inputs€€€€€€€€€€€€€€€€€€@
p
™ "%К"€€€€€€€€€€€€€€€€€€@Ч
6__inference_batch_normalization_1_layer_call_fn_108190]E:D;7Ґ4
-Ґ*
$К!
inputs€€€€€€€€€}@
p 
™ "К€€€€€€€€€}@Ч
6__inference_batch_normalization_1_layer_call_fn_108203]DE:;7Ґ4
-Ґ*
$К!
inputs€€€€€€€€€}@
p
™ "К€€€€€€€€€}@—
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_108223|G<F=@Ґ=
6Ґ3
-К*
inputs€€€€€€€€€€€€€€€€€€@
p 
™ "2Ґ/
(К%
0€€€€€€€€€€€€€€€€€€@
Ъ —
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_108257|FG<=@Ґ=
6Ґ3
-К*
inputs€€€€€€€€€€€€€€€€€€@
p
™ "2Ґ/
(К%
0€€€€€€€€€€€€€€€€€€@
Ъ њ
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_108277jG<F=7Ґ4
-Ґ*
$К!
inputs€€€€€€€€€}@
p 
™ ")Ґ&
К
0€€€€€€€€€}@
Ъ њ
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_108311jFG<=7Ґ4
-Ґ*
$К!
inputs€€€€€€€€€}@
p
™ ")Ґ&
К
0€€€€€€€€€}@
Ъ ©
6__inference_batch_normalization_2_layer_call_fn_108324oG<F=@Ґ=
6Ґ3
-К*
inputs€€€€€€€€€€€€€€€€€€@
p 
™ "%К"€€€€€€€€€€€€€€€€€€@©
6__inference_batch_normalization_2_layer_call_fn_108337oFG<=@Ґ=
6Ґ3
-К*
inputs€€€€€€€€€€€€€€€€€€@
p
™ "%К"€€€€€€€€€€€€€€€€€€@Ч
6__inference_batch_normalization_2_layer_call_fn_108350]G<F=7Ґ4
-Ґ*
$К!
inputs€€€€€€€€€}@
p 
™ "К€€€€€€€€€}@Ч
6__inference_batch_normalization_2_layer_call_fn_108363]FG<=7Ґ4
-Ґ*
$К!
inputs€€€€€€€€€}@
p
™ "К€€€€€€€€€}@Ј
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_108625bI@HA3Ґ0
)Ґ&
 К
inputs€€€€€€€€€T
p 
™ "%Ґ"
К
0€€€€€€€€€T
Ъ Ј
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_108659bHI@A3Ґ0
)Ґ&
 К
inputs€€€€€€€€€T
p
™ "%Ґ"
К
0€€€€€€€€€T
Ъ П
6__inference_batch_normalization_3_layer_call_fn_108672UI@HA3Ґ0
)Ґ&
 К
inputs€€€€€€€€€T
p 
™ "К€€€€€€€€€TП
6__inference_batch_normalization_3_layer_call_fn_108685UHI@A3Ґ0
)Ґ&
 К
inputs€€€€€€€€€T
p
™ "К€€€€€€€€€Tѕ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_107903|C8B9@Ґ=
6Ґ3
-К*
inputs€€€€€€€€€€€€€€€€€€@
p 
™ "2Ґ/
(К%
0€€€€€€€€€€€€€€€€€€@
Ъ ѕ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_107937|BC89@Ґ=
6Ґ3
-К*
inputs€€€€€€€€€€€€€€€€€€@
p
™ "2Ґ/
(К%
0€€€€€€€€€€€€€€€€€€@
Ъ љ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_107957jC8B97Ґ4
-Ґ*
$К!
inputs€€€€€€€€€}@
p 
™ ")Ґ&
К
0€€€€€€€€€}@
Ъ љ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_107991jBC897Ґ4
-Ґ*
$К!
inputs€€€€€€€€€}@
p
™ ")Ґ&
К
0€€€€€€€€€}@
Ъ І
4__inference_batch_normalization_layer_call_fn_108004oC8B9@Ґ=
6Ґ3
-К*
inputs€€€€€€€€€€€€€€€€€€@
p 
™ "%К"€€€€€€€€€€€€€€€€€€@І
4__inference_batch_normalization_layer_call_fn_108017oBC89@Ґ=
6Ґ3
-К*
inputs€€€€€€€€€€€€€€€€€€@
p
™ "%К"€€€€€€€€€€€€€€€€€€@Х
4__inference_batch_normalization_layer_call_fn_108030]C8B97Ґ4
-Ґ*
$К!
inputs€€€€€€€€€}@
p 
™ "К€€€€€€€€€}@Х
4__inference_batch_normalization_layer_call_fn_108043]BC897Ґ4
-Ґ*
$К!
inputs€€€€€€€€€}@
p
™ "К€€€€€€€€€}@ф
G__inference_concatenate_layer_call_and_return_conditional_losses_108548®~Ґ{
tҐq
oЪl
"К
inputs/0€€€€€€€€€@
"К
inputs/1€€€€€€€€€@
"К
inputs/2€€€€€€€€€@
™ "&Ґ#
К
0€€€€€€€€€ј
Ъ ћ
,__inference_concatenate_layer_call_fn_108555Ы~Ґ{
tҐq
oЪl
"К
inputs/0€€€€€€€€€@
"К
inputs/1€€€€€€€€€@
"К
inputs/2€€€€€€€€€@
™ "К€€€€€€€€€ј≠
K__inference_dense_1_dropout_layer_call_and_return_conditional_losses_108560^4Ґ1
*Ґ'
!К
inputs€€€€€€€€€ј
p 
™ "&Ґ#
К
0€€€€€€€€€ј
Ъ ≠
K__inference_dense_1_dropout_layer_call_and_return_conditional_losses_108564^4Ґ1
*Ґ'
!К
inputs€€€€€€€€€ј
p
™ "&Ґ#
К
0€€€€€€€€€ј
Ъ Е
0__inference_dense_1_dropout_layer_call_fn_108569Q4Ґ1
*Ґ'
!К
inputs€€€€€€€€€ј
p 
™ "К€€€€€€€€€јЕ
0__inference_dense_1_dropout_layer_call_fn_108574Q4Ґ1
*Ґ'
!К
inputs€€€€€€€€€ј
p
™ "К€€€€€€€€€ј§
C__inference_dense_1_layer_call_and_return_conditional_losses_108596]>?0Ґ-
&Ґ#
!К
inputs€€€€€€€€€ј
™ "%Ґ"
К
0€€€€€€€€€T
Ъ |
(__inference_dense_1_layer_call_fn_108605P>?0Ґ-
&Ґ#
!К
inputs€€€€€€€€€ј
™ "К€€€€€€€€€T™
N__inference_dense_activation_1_layer_call_and_return_conditional_losses_108690X/Ґ,
%Ґ"
 К
inputs€€€€€€€€€T
™ "%Ґ"
К
0€€€€€€€€€T
Ъ В
3__inference_dense_activation_1_layer_call_fn_108695K/Ґ,
%Ґ"
 К
inputs€€€€€€€€€T
™ "К€€€€€€€€€T‘
D__inference_distance_layer_call_and_return_conditional_losses_107670ЛbҐ_
XҐU
KЪH
"К
inputs/0€€€€€€€€€T
"К
inputs/1€€€€€€€€€T

 
p 
™ "%Ґ"
К
0€€€€€€€€€
Ъ ‘
D__inference_distance_layer_call_and_return_conditional_losses_107682ЛbҐ_
XҐU
KЪH
"К
inputs/0€€€€€€€€€T
"К
inputs/1€€€€€€€€€T

 
p
™ "%Ґ"
К
0€€€€€€€€€
Ъ Ђ
)__inference_distance_layer_call_fn_107688~bҐ_
XҐU
KЪH
"К
inputs/0€€€€€€€€€T
"К
inputs/1€€€€€€€€€T

 
p 
™ "К€€€€€€€€€Ђ
)__inference_distance_layer_call_fn_107694~bҐ_
XҐU
KЪH
"К
inputs/0€€€€€€€€€T
"К
inputs/1€€€€€€€€€T

 
p
™ "К€€€€€€€€€’
V__inference_global_average_pooling1d_1_layer_call_and_return_conditional_losses_108502{IҐF
?Ґ<
6К3
inputs'€€€€€€€€€€€€€€€€€€€€€€€€€€€

 
™ ".Ґ+
$К!
0€€€€€€€€€€€€€€€€€€
Ъ Ї
V__inference_global_average_pooling1d_1_layer_call_and_return_conditional_losses_108508`7Ґ4
-Ґ*
$К!
inputs€€€€€€€€€}@

 
™ "%Ґ"
К
0€€€€€€€€€@
Ъ ≠
;__inference_global_average_pooling1d_1_layer_call_fn_108513nIҐF
?Ґ<
6К3
inputs'€€€€€€€€€€€€€€€€€€€€€€€€€€€

 
™ "!К€€€€€€€€€€€€€€€€€€Т
;__inference_global_average_pooling1d_1_layer_call_fn_108518S7Ґ4
-Ґ*
$К!
inputs€€€€€€€€€}@

 
™ "К€€€€€€€€€@’
V__inference_global_average_pooling1d_2_layer_call_and_return_conditional_losses_108524{IҐF
?Ґ<
6К3
inputs'€€€€€€€€€€€€€€€€€€€€€€€€€€€

 
™ ".Ґ+
$К!
0€€€€€€€€€€€€€€€€€€
Ъ Ї
V__inference_global_average_pooling1d_2_layer_call_and_return_conditional_losses_108530`7Ґ4
-Ґ*
$К!
inputs€€€€€€€€€}@

 
™ "%Ґ"
К
0€€€€€€€€€@
Ъ ≠
;__inference_global_average_pooling1d_2_layer_call_fn_108535nIҐF
?Ґ<
6К3
inputs'€€€€€€€€€€€€€€€€€€€€€€€€€€€

 
™ "!К€€€€€€€€€€€€€€€€€€Т
;__inference_global_average_pooling1d_2_layer_call_fn_108540S7Ґ4
-Ґ*
$К!
inputs€€€€€€€€€}@

 
™ "К€€€€€€€€€@”
T__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_108480{IҐF
?Ґ<
6К3
inputs'€€€€€€€€€€€€€€€€€€€€€€€€€€€

 
™ ".Ґ+
$К!
0€€€€€€€€€€€€€€€€€€
Ъ Є
T__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_108486`7Ґ4
-Ґ*
$К!
inputs€€€€€€€€€}@

 
™ "%Ґ"
К
0€€€€€€€€€@
Ъ Ђ
9__inference_global_average_pooling1d_layer_call_fn_108491nIҐF
?Ґ<
6К3
inputs'€€€€€€€€€€€€€€€€€€€€€€€€€€€

 
™ "!К€€€€€€€€€€€€€€€€€€Р
9__inference_global_average_pooling1d_layer_call_fn_108496S7Ґ4
-Ґ*
$К!
inputs€€€€€€€€€}@

 
™ "К€€€€€€€€€@;
__inference_loss_fn_0_1087062Ґ

Ґ 
™ "К ;
__inference_loss_fn_1_1087174Ґ

Ґ 
™ "К ;
__inference_loss_fn_2_1087286Ґ

Ґ 
™ "К ;
__inference_loss_fn_3_108739>Ґ

Ґ 
™ "К ъ
A__inference_model_layer_call_and_return_conditional_losses_106119і674523G<F=E:D;C8B9>?I@HAqҐn
gҐd
ZЪW
)К&
left_inputs€€€€€€€€€}
*К'
right_inputs€€€€€€€€€}
p 

 
™ "%Ґ"
К
0€€€€€€€€€
Ъ ъ
A__inference_model_layer_call_and_return_conditional_losses_106223і674523FG<=DE:;BC89>?HI@AqҐn
gҐd
ZЪW
)К&
left_inputs€€€€€€€€€}
*К'
right_inputs€€€€€€€€€}
p

 
™ "%Ґ"
К
0€€€€€€€€€
Ъ у
A__inference_model_layer_call_and_return_conditional_losses_106565≠674523G<F=E:D;C8B9>?I@HAjҐg
`Ґ]
SЪP
&К#
inputs/0€€€€€€€€€}
&К#
inputs/1€€€€€€€€€}
p 

 
™ "%Ґ"
К
0€€€€€€€€€
Ъ у
A__inference_model_layer_call_and_return_conditional_losses_107015≠674523FG<=DE:;BC89>?HI@AjҐg
`Ґ]
SЪP
&К#
inputs/0€€€€€€€€€}
&К#
inputs/1€€€€€€€€€}
p

 
™ "%Ґ"
К
0€€€€€€€€€
Ъ “
&__inference_model_layer_call_fn_105724І674523G<F=E:D;C8B9>?I@HAqҐn
gҐd
ZЪW
)К&
left_inputs€€€€€€€€€}
*К'
right_inputs€€€€€€€€€}
p 

 
™ "К€€€€€€€€€“
&__inference_model_layer_call_fn_106015І674523FG<=DE:;BC89>?HI@AqҐn
gҐd
ZЪW
)К&
left_inputs€€€€€€€€€}
*К'
right_inputs€€€€€€€€€}
p

 
™ "К€€€€€€€€€Ћ
&__inference_model_layer_call_fn_107069†674523G<F=E:D;C8B9>?I@HAjҐg
`Ґ]
SЪP
&К#
inputs/0€€€€€€€€€}
&К#
inputs/1€€€€€€€€€}
p 

 
™ "К€€€€€€€€€Ћ
&__inference_model_layer_call_fn_107123†674523FG<=DE:;BC89>?HI@AjҐg
`Ґ]
SЪP
&К#
inputs/0€€€€€€€€€}
&К#
inputs/1€€€€€€€€€}
p

 
™ "К€€€€€€€€€€
$__inference_signature_wrapper_106309÷674523G<F=E:D;C8B9>?I@HAДҐА
Ґ 
y™v
8
left_inputs)К&
left_inputs€€€€€€€€€}
:
right_inputs*К'
right_inputs€€€€€€€€€}"3™0
.
distance"К
distance€€€€€€€€€≥
K__inference_stream_0_conv_1_layer_call_and_return_conditional_losses_107802d233Ґ0
)Ґ&
$К!
inputs€€€€€€€€€}
™ ")Ґ&
К
0€€€€€€€€€}@
Ъ Л
0__inference_stream_0_conv_1_layer_call_fn_107811W233Ґ0
)Ґ&
$К!
inputs€€€€€€€€€}
™ "К€€€€€€€€€}@≥
K__inference_stream_0_drop_1_layer_call_and_return_conditional_losses_108398d7Ґ4
-Ґ*
$К!
inputs€€€€€€€€€}@
p 
™ ")Ґ&
К
0€€€€€€€€€}@
Ъ ≥
K__inference_stream_0_drop_1_layer_call_and_return_conditional_losses_108410d7Ґ4
-Ґ*
$К!
inputs€€€€€€€€€}@
p
™ ")Ґ&
К
0€€€€€€€€€}@
Ъ Л
0__inference_stream_0_drop_1_layer_call_fn_108415W7Ґ4
-Ґ*
$К!
inputs€€€€€€€€€}@
p 
™ "К€€€€€€€€€}@Л
0__inference_stream_0_drop_1_layer_call_fn_108420W7Ґ4
-Ґ*
$К!
inputs€€€€€€€€€}@
p
™ "К€€€€€€€€€}@Ј
O__inference_stream_0_input_drop_layer_call_and_return_conditional_losses_107699d7Ґ4
-Ґ*
$К!
inputs€€€€€€€€€}
p 
™ ")Ґ&
К
0€€€€€€€€€}
Ъ Ј
O__inference_stream_0_input_drop_layer_call_and_return_conditional_losses_107711d7Ґ4
-Ґ*
$К!
inputs€€€€€€€€€}
p
™ ")Ґ&
К
0€€€€€€€€€}
Ъ П
4__inference_stream_0_input_drop_layer_call_fn_107716W7Ґ4
-Ґ*
$К!
inputs€€€€€€€€€}
p 
™ "К€€€€€€€€€}П
4__inference_stream_0_input_drop_layer_call_fn_107721W7Ґ4
-Ґ*
$К!
inputs€€€€€€€€€}
p
™ "К€€€€€€€€€}≥
K__inference_stream_1_conv_1_layer_call_and_return_conditional_losses_107838d453Ґ0
)Ґ&
$К!
inputs€€€€€€€€€}
™ ")Ґ&
К
0€€€€€€€€€}@
Ъ Л
0__inference_stream_1_conv_1_layer_call_fn_107847W453Ґ0
)Ґ&
$К!
inputs€€€€€€€€€}
™ "К€€€€€€€€€}@≥
K__inference_stream_1_drop_1_layer_call_and_return_conditional_losses_108425d7Ґ4
-Ґ*
$К!
inputs€€€€€€€€€}@
p 
™ ")Ґ&
К
0€€€€€€€€€}@
Ъ ≥
K__inference_stream_1_drop_1_layer_call_and_return_conditional_losses_108437d7Ґ4
-Ґ*
$К!
inputs€€€€€€€€€}@
p
™ ")Ґ&
К
0€€€€€€€€€}@
Ъ Л
0__inference_stream_1_drop_1_layer_call_fn_108442W7Ґ4
-Ґ*
$К!
inputs€€€€€€€€€}@
p 
™ "К€€€€€€€€€}@Л
0__inference_stream_1_drop_1_layer_call_fn_108447W7Ґ4
-Ґ*
$К!
inputs€€€€€€€€€}@
p
™ "К€€€€€€€€€}@Ј
O__inference_stream_1_input_drop_layer_call_and_return_conditional_losses_107726d7Ґ4
-Ґ*
$К!
inputs€€€€€€€€€}
p 
™ ")Ґ&
К
0€€€€€€€€€}
Ъ Ј
O__inference_stream_1_input_drop_layer_call_and_return_conditional_losses_107738d7Ґ4
-Ґ*
$К!
inputs€€€€€€€€€}
p
™ ")Ґ&
К
0€€€€€€€€€}
Ъ П
4__inference_stream_1_input_drop_layer_call_fn_107743W7Ґ4
-Ґ*
$К!
inputs€€€€€€€€€}
p 
™ "К€€€€€€€€€}П
4__inference_stream_1_input_drop_layer_call_fn_107748W7Ґ4
-Ґ*
$К!
inputs€€€€€€€€€}
p
™ "К€€€€€€€€€}≥
K__inference_stream_2_conv_1_layer_call_and_return_conditional_losses_107874d673Ґ0
)Ґ&
$К!
inputs€€€€€€€€€}
™ ")Ґ&
К
0€€€€€€€€€}@
Ъ Л
0__inference_stream_2_conv_1_layer_call_fn_107883W673Ґ0
)Ґ&
$К!
inputs€€€€€€€€€}
™ "К€€€€€€€€€}@≥
K__inference_stream_2_drop_1_layer_call_and_return_conditional_losses_108452d7Ґ4
-Ґ*
$К!
inputs€€€€€€€€€}@
p 
™ ")Ґ&
К
0€€€€€€€€€}@
Ъ ≥
K__inference_stream_2_drop_1_layer_call_and_return_conditional_losses_108464d7Ґ4
-Ґ*
$К!
inputs€€€€€€€€€}@
p
™ ")Ґ&
К
0€€€€€€€€€}@
Ъ Л
0__inference_stream_2_drop_1_layer_call_fn_108469W7Ґ4
-Ґ*
$К!
inputs€€€€€€€€€}@
p 
™ "К€€€€€€€€€}@Л
0__inference_stream_2_drop_1_layer_call_fn_108474W7Ґ4
-Ґ*
$К!
inputs€€€€€€€€€}@
p
™ "К€€€€€€€€€}@Ј
O__inference_stream_2_input_drop_layer_call_and_return_conditional_losses_107753d7Ґ4
-Ґ*
$К!
inputs€€€€€€€€€}
p 
™ ")Ґ&
К
0€€€€€€€€€}
Ъ Ј
O__inference_stream_2_input_drop_layer_call_and_return_conditional_losses_107765d7Ґ4
-Ґ*
$К!
inputs€€€€€€€€€}
p
™ ")Ґ&
К
0€€€€€€€€€}
Ъ П
4__inference_stream_2_input_drop_layer_call_fn_107770W7Ґ4
-Ґ*
$К!
inputs€€€€€€€€€}
p 
™ "К€€€€€€€€€}П
4__inference_stream_2_input_drop_layer_call_fn_107775W7Ґ4
-Ґ*
$К!
inputs€€€€€€€€€}
p
™ "К€€€€€€€€€}