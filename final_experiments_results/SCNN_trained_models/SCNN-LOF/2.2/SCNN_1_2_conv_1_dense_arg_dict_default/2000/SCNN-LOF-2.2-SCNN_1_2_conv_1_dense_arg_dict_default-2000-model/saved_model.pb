��.
��
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
 �"serve*2.6.22v2.6.1-9-gc2363d6d0258��+
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
shape: *'
shared_namestream_0_conv_1/kernel
�
*stream_0_conv_1/kernel/Read/ReadVariableOpReadVariableOpstream_0_conv_1/kernel*"
_output_shapes
: *
dtype0
�
stream_0_conv_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_namestream_0_conv_1/bias
y
(stream_0_conv_1/bias/Read/ReadVariableOpReadVariableOpstream_0_conv_1/bias*
_output_shapes
: *
dtype0
�
batch_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: **
shared_namebatch_normalization/gamma
�
-batch_normalization/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization/gamma*
_output_shapes
: *
dtype0
�
batch_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *)
shared_namebatch_normalization/beta
�
,batch_normalization/beta/Read/ReadVariableOpReadVariableOpbatch_normalization/beta*
_output_shapes
: *
dtype0
�
stream_0_conv_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*'
shared_namestream_0_conv_2/kernel
�
*stream_0_conv_2/kernel/Read/ReadVariableOpReadVariableOpstream_0_conv_2/kernel*"
_output_shapes
: @*
dtype0
�
stream_0_conv_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_namestream_0_conv_2/bias
y
(stream_0_conv_2/bias/Read/ReadVariableOpReadVariableOpstream_0_conv_2/bias*
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
x
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@T*
shared_namedense_1/kernel
q
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes

:@T*
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
batch_normalization_2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:T*,
shared_namebatch_normalization_2/gamma
�
/batch_normalization_2/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_2/gamma*
_output_shapes
:T*
dtype0
�
batch_normalization_2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:T*+
shared_namebatch_normalization_2/beta
�
.batch_normalization_2/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_2/beta*
_output_shapes
:T*
dtype0
�
batch_normalization/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *0
shared_name!batch_normalization/moving_mean
�
3batch_normalization/moving_mean/Read/ReadVariableOpReadVariableOpbatch_normalization/moving_mean*
_output_shapes
: *
dtype0
�
#batch_normalization/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *4
shared_name%#batch_normalization/moving_variance
�
7batch_normalization/moving_variance/Read/ReadVariableOpReadVariableOp#batch_normalization/moving_variance*
_output_shapes
: *
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
shape:T*2
shared_name#!batch_normalization_2/moving_mean
�
5batch_normalization_2/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_2/moving_mean*
_output_shapes
:T*
dtype0
�
%batch_normalization_2/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:T*6
shared_name'%batch_normalization_2/moving_variance
�
9batch_normalization_2/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_2/moving_variance*
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
shape: *.
shared_nameAdam/stream_0_conv_1/kernel/m
�
1Adam/stream_0_conv_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/stream_0_conv_1/kernel/m*"
_output_shapes
: *
dtype0
�
Adam/stream_0_conv_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_nameAdam/stream_0_conv_1/bias/m
�
/Adam/stream_0_conv_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/stream_0_conv_1/bias/m*
_output_shapes
: *
dtype0
�
 Adam/batch_normalization/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *1
shared_name" Adam/batch_normalization/gamma/m
�
4Adam/batch_normalization/gamma/m/Read/ReadVariableOpReadVariableOp Adam/batch_normalization/gamma/m*
_output_shapes
: *
dtype0
�
Adam/batch_normalization/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *0
shared_name!Adam/batch_normalization/beta/m
�
3Adam/batch_normalization/beta/m/Read/ReadVariableOpReadVariableOpAdam/batch_normalization/beta/m*
_output_shapes
: *
dtype0
�
Adam/stream_0_conv_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*.
shared_nameAdam/stream_0_conv_2/kernel/m
�
1Adam/stream_0_conv_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/stream_0_conv_2/kernel/m*"
_output_shapes
: @*
dtype0
�
Adam/stream_0_conv_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_nameAdam/stream_0_conv_2/bias/m
�
/Adam/stream_0_conv_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/stream_0_conv_2/bias/m*
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
Adam/dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@T*&
shared_nameAdam/dense_1/kernel/m

)Adam/dense_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/m*
_output_shapes

:@T*
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
"Adam/batch_normalization_2/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:T*3
shared_name$"Adam/batch_normalization_2/gamma/m
�
6Adam/batch_normalization_2/gamma/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_2/gamma/m*
_output_shapes
:T*
dtype0
�
!Adam/batch_normalization_2/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:T*2
shared_name#!Adam/batch_normalization_2/beta/m
�
5Adam/batch_normalization_2/beta/m/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_2/beta/m*
_output_shapes
:T*
dtype0
�
Adam/stream_0_conv_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *.
shared_nameAdam/stream_0_conv_1/kernel/v
�
1Adam/stream_0_conv_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/stream_0_conv_1/kernel/v*"
_output_shapes
: *
dtype0
�
Adam/stream_0_conv_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_nameAdam/stream_0_conv_1/bias/v
�
/Adam/stream_0_conv_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/stream_0_conv_1/bias/v*
_output_shapes
: *
dtype0
�
 Adam/batch_normalization/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *1
shared_name" Adam/batch_normalization/gamma/v
�
4Adam/batch_normalization/gamma/v/Read/ReadVariableOpReadVariableOp Adam/batch_normalization/gamma/v*
_output_shapes
: *
dtype0
�
Adam/batch_normalization/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *0
shared_name!Adam/batch_normalization/beta/v
�
3Adam/batch_normalization/beta/v/Read/ReadVariableOpReadVariableOpAdam/batch_normalization/beta/v*
_output_shapes
: *
dtype0
�
Adam/stream_0_conv_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*.
shared_nameAdam/stream_0_conv_2/kernel/v
�
1Adam/stream_0_conv_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/stream_0_conv_2/kernel/v*"
_output_shapes
: @*
dtype0
�
Adam/stream_0_conv_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_nameAdam/stream_0_conv_2/bias/v
�
/Adam/stream_0_conv_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/stream_0_conv_2/bias/v*
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
Adam/dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@T*&
shared_nameAdam/dense_1/kernel/v

)Adam/dense_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/v*
_output_shapes

:@T*
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
"Adam/batch_normalization_2/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:T*3
shared_name$"Adam/batch_normalization_2/gamma/v
�
6Adam/batch_normalization_2/gamma/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_2/gamma/v*
_output_shapes
:T*
dtype0
�
!Adam/batch_normalization_2/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:T*2
shared_name#!Adam/batch_normalization_2/beta/v
�
5Adam/batch_normalization_2/beta/v/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_2/beta/v*
_output_shapes
:T*
dtype0

NoOpNoOp
�b
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�b
value�aB�a B�a
�
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
�
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
layer-10
layer-11
layer-12
layer_with_weights-4
layer-13
layer_with_weights-5
layer-14
layer-15
trainable_variables
	variables
regularization_losses
	keras_api
R
trainable_variables
 	variables
!regularization_losses
"	keras_api
�

#beta_1

$beta_2
	%decay
&learning_rate
'iter(m�)m�*m�+m�,m�-m�.m�/m�0m�1m�2m�3m�(v�)v�*v�+v�,v�-v�.v�/v�0v�1v�2v�3v�
V
(0
)1
*2
+3
,4
-5
.6
/7
08
19
210
311
�
(0
)1
*2
+3
44
55
,6
-7
.8
/9
610
711
012
113
214
315
816
917
 
�
:layer_metrics
trainable_variables

;layers
<layer_regularization_losses
	variables
=non_trainable_variables
regularization_losses
>metrics
 
 
R
?trainable_variables
@	variables
Aregularization_losses
B	keras_api
h

(kernel
)bias
Ctrainable_variables
D	variables
Eregularization_losses
F	keras_api
�
Gaxis
	*gamma
+beta
4moving_mean
5moving_variance
Htrainable_variables
I	variables
Jregularization_losses
K	keras_api
R
Ltrainable_variables
M	variables
Nregularization_losses
O	keras_api
R
Ptrainable_variables
Q	variables
Rregularization_losses
S	keras_api
h

,kernel
-bias
Ttrainable_variables
U	variables
Vregularization_losses
W	keras_api
�
Xaxis
	.gamma
/beta
6moving_mean
7moving_variance
Ytrainable_variables
Z	variables
[regularization_losses
\	keras_api
R
]trainable_variables
^	variables
_regularization_losses
`	keras_api
R
atrainable_variables
b	variables
cregularization_losses
d	keras_api
R
etrainable_variables
f	variables
gregularization_losses
h	keras_api
R
itrainable_variables
j	variables
kregularization_losses
l	keras_api
R
mtrainable_variables
n	variables
oregularization_losses
p	keras_api
h

0kernel
1bias
qtrainable_variables
r	variables
sregularization_losses
t	keras_api
�
uaxis
	2gamma
3beta
8moving_mean
9moving_variance
vtrainable_variables
w	variables
xregularization_losses
y	keras_api
R
ztrainable_variables
{	variables
|regularization_losses
}	keras_api
V
(0
)1
*2
+3
,4
-5
.6
/7
08
19
210
311
�
(0
)1
*2
+3
44
55
,6
-7
.8
/9
610
711
012
113
214
315
816
917
 
�
~layer_metrics
trainable_variables

layers
 �layer_regularization_losses
	variables
�non_trainable_variables
regularization_losses
�metrics
 
 
 
�
�layer_metrics
trainable_variables
�layers
 �layer_regularization_losses
 	variables
�non_trainable_variables
!regularization_losses
�metrics
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
TR
VARIABLE_VALUEdense_1/kernel0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEdense_1/bias0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUEbatch_normalization_2/gamma1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEbatch_normalization_2/beta1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUE
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
 

0
1
2
3
 
*
40
51
62
73
84
95

�0
 
 
 
�
�layer_metrics
?trainable_variables
�layers
 �layer_regularization_losses
@	variables
�non_trainable_variables
Aregularization_losses
�metrics

(0
)1

(0
)1
 
�
�layer_metrics
Ctrainable_variables
�layers
 �layer_regularization_losses
D	variables
�non_trainable_variables
Eregularization_losses
�metrics
 

*0
+1

*0
+1
42
53
 
�
�layer_metrics
Htrainable_variables
�layers
 �layer_regularization_losses
I	variables
�non_trainable_variables
Jregularization_losses
�metrics
 
 
 
�
�layer_metrics
Ltrainable_variables
�layers
 �layer_regularization_losses
M	variables
�non_trainable_variables
Nregularization_losses
�metrics
 
 
 
�
�layer_metrics
Ptrainable_variables
�layers
 �layer_regularization_losses
Q	variables
�non_trainable_variables
Rregularization_losses
�metrics

,0
-1

,0
-1
 
�
�layer_metrics
Ttrainable_variables
�layers
 �layer_regularization_losses
U	variables
�non_trainable_variables
Vregularization_losses
�metrics
 

.0
/1

.0
/1
62
73
 
�
�layer_metrics
Ytrainable_variables
�layers
 �layer_regularization_losses
Z	variables
�non_trainable_variables
[regularization_losses
�metrics
 
 
 
�
�layer_metrics
]trainable_variables
�layers
 �layer_regularization_losses
^	variables
�non_trainable_variables
_regularization_losses
�metrics
 
 
 
�
�layer_metrics
atrainable_variables
�layers
 �layer_regularization_losses
b	variables
�non_trainable_variables
cregularization_losses
�metrics
 
 
 
�
�layer_metrics
etrainable_variables
�layers
 �layer_regularization_losses
f	variables
�non_trainable_variables
gregularization_losses
�metrics
 
 
 
�
�layer_metrics
itrainable_variables
�layers
 �layer_regularization_losses
j	variables
�non_trainable_variables
kregularization_losses
�metrics
 
 
 
�
�layer_metrics
mtrainable_variables
�layers
 �layer_regularization_losses
n	variables
�non_trainable_variables
oregularization_losses
�metrics

00
11

00
11
 
�
�layer_metrics
qtrainable_variables
�layers
 �layer_regularization_losses
r	variables
�non_trainable_variables
sregularization_losses
�metrics
 

20
31

20
31
82
93
 
�
�layer_metrics
vtrainable_variables
�layers
 �layer_regularization_losses
w	variables
�non_trainable_variables
xregularization_losses
�metrics
 
 
 
�
�layer_metrics
ztrainable_variables
�layers
 �layer_regularization_losses
{	variables
�non_trainable_variables
|regularization_losses
�metrics
 
v
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
 
*
40
51
62
73
84
95
 
 
 
 
 
 
8

�total

�count
�	variables
�	keras_api
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
40
51
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
60
71
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
80
91
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
�0
�1

�	variables
}
VARIABLE_VALUEAdam/stream_0_conv_1/kernel/mLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/stream_0_conv_1/bias/mLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE Adam/batch_normalization/gamma/mLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
�
VARIABLE_VALUEAdam/batch_normalization/beta/mLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/stream_0_conv_2/kernel/mLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/stream_0_conv_2/bias/mLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE"Adam/batch_normalization_1/gamma/mLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE!Adam/batch_normalization_1/beta/mLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense_1/kernel/mLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEAdam/dense_1/bias/mLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE"Adam/batch_normalization_2/gamma/mMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE!Adam/batch_normalization_2/beta/mMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/stream_0_conv_1/kernel/vLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/stream_0_conv_1/bias/vLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE Adam/batch_normalization/gamma/vLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
�
VARIABLE_VALUEAdam/batch_normalization/beta/vLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/stream_0_conv_2/kernel/vLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/stream_0_conv_2/bias/vLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE"Adam/batch_normalization_1/gamma/vLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE!Adam/batch_normalization_1/beta/vLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense_1/kernel/vLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEAdam/dense_1/bias/vLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE"Adam/batch_normalization_2/gamma/vMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE!Adam/batch_normalization_2/beta/vMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
�
serving_default_left_inputsPlaceholder*,
_output_shapes
:����������*
dtype0*!
shape:����������
�
serving_default_right_inputsPlaceholder*,
_output_shapes
:����������*
dtype0*!
shape:����������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_left_inputsserving_default_right_inputsstream_0_conv_1/kernelstream_0_conv_1/bias#batch_normalization/moving_variancebatch_normalization/gammabatch_normalization/moving_meanbatch_normalization/betastream_0_conv_2/kernelstream_0_conv_2/bias%batch_normalization_1/moving_variancebatch_normalization_1/gamma!batch_normalization_1/moving_meanbatch_normalization_1/betadense_1/kerneldense_1/bias%batch_normalization_2/moving_variancebatch_normalization_2/gamma!batch_normalization_2/moving_meanbatch_normalization_2/beta*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*4
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *.
f)R'
%__inference_signature_wrapper_5543134
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenamebeta_1/Read/ReadVariableOpbeta_2/Read/ReadVariableOpdecay/Read/ReadVariableOp!learning_rate/Read/ReadVariableOpAdam/iter/Read/ReadVariableOp*stream_0_conv_1/kernel/Read/ReadVariableOp(stream_0_conv_1/bias/Read/ReadVariableOp-batch_normalization/gamma/Read/ReadVariableOp,batch_normalization/beta/Read/ReadVariableOp*stream_0_conv_2/kernel/Read/ReadVariableOp(stream_0_conv_2/bias/Read/ReadVariableOp/batch_normalization_1/gamma/Read/ReadVariableOp.batch_normalization_1/beta/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOp/batch_normalization_2/gamma/Read/ReadVariableOp.batch_normalization_2/beta/Read/ReadVariableOp3batch_normalization/moving_mean/Read/ReadVariableOp7batch_normalization/moving_variance/Read/ReadVariableOp5batch_normalization_1/moving_mean/Read/ReadVariableOp9batch_normalization_1/moving_variance/Read/ReadVariableOp5batch_normalization_2/moving_mean/Read/ReadVariableOp9batch_normalization_2/moving_variance/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp1Adam/stream_0_conv_1/kernel/m/Read/ReadVariableOp/Adam/stream_0_conv_1/bias/m/Read/ReadVariableOp4Adam/batch_normalization/gamma/m/Read/ReadVariableOp3Adam/batch_normalization/beta/m/Read/ReadVariableOp1Adam/stream_0_conv_2/kernel/m/Read/ReadVariableOp/Adam/stream_0_conv_2/bias/m/Read/ReadVariableOp6Adam/batch_normalization_1/gamma/m/Read/ReadVariableOp5Adam/batch_normalization_1/beta/m/Read/ReadVariableOp)Adam/dense_1/kernel/m/Read/ReadVariableOp'Adam/dense_1/bias/m/Read/ReadVariableOp6Adam/batch_normalization_2/gamma/m/Read/ReadVariableOp5Adam/batch_normalization_2/beta/m/Read/ReadVariableOp1Adam/stream_0_conv_1/kernel/v/Read/ReadVariableOp/Adam/stream_0_conv_1/bias/v/Read/ReadVariableOp4Adam/batch_normalization/gamma/v/Read/ReadVariableOp3Adam/batch_normalization/beta/v/Read/ReadVariableOp1Adam/stream_0_conv_2/kernel/v/Read/ReadVariableOp/Adam/stream_0_conv_2/bias/v/Read/ReadVariableOp6Adam/batch_normalization_1/gamma/v/Read/ReadVariableOp5Adam/batch_normalization_1/beta/v/Read/ReadVariableOp)Adam/dense_1/kernel/v/Read/ReadVariableOp'Adam/dense_1/bias/v/Read/ReadVariableOp6Adam/batch_normalization_2/gamma/v/Read/ReadVariableOp5Adam/batch_normalization_2/beta/v/Read/ReadVariableOpConst*>
Tin7
523	*
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
 __inference__traced_save_5545402
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamebeta_1beta_2decaylearning_rate	Adam/iterstream_0_conv_1/kernelstream_0_conv_1/biasbatch_normalization/gammabatch_normalization/betastream_0_conv_2/kernelstream_0_conv_2/biasbatch_normalization_1/gammabatch_normalization_1/betadense_1/kerneldense_1/biasbatch_normalization_2/gammabatch_normalization_2/betabatch_normalization/moving_mean#batch_normalization/moving_variance!batch_normalization_1/moving_mean%batch_normalization_1/moving_variance!batch_normalization_2/moving_mean%batch_normalization_2/moving_variancetotalcountAdam/stream_0_conv_1/kernel/mAdam/stream_0_conv_1/bias/m Adam/batch_normalization/gamma/mAdam/batch_normalization/beta/mAdam/stream_0_conv_2/kernel/mAdam/stream_0_conv_2/bias/m"Adam/batch_normalization_1/gamma/m!Adam/batch_normalization_1/beta/mAdam/dense_1/kernel/mAdam/dense_1/bias/m"Adam/batch_normalization_2/gamma/m!Adam/batch_normalization_2/beta/mAdam/stream_0_conv_1/kernel/vAdam/stream_0_conv_1/bias/v Adam/batch_normalization/gamma/vAdam/batch_normalization/beta/vAdam/stream_0_conv_2/kernel/vAdam/stream_0_conv_2/bias/v"Adam/batch_normalization_1/gamma/v!Adam/batch_normalization_1/beta/vAdam/dense_1/kernel/vAdam/dense_1/bias/v"Adam/batch_normalization_2/gamma/v!Adam/batch_normalization_2/beta/v*=
Tin6
422*
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
#__inference__traced_restore_5545559��)
�
�
__inference_loss_fn_0_5545209T
>stream_0_conv_1_kernel_regularizer_abs_readvariableop_resource: 
identity��5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp�
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp>stream_0_conv_1_kernel_regularizer_abs_readvariableop_resource*"
_output_shapes
: *
dtype027
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp�
&stream_0_conv_1/kernel/Regularizer/AbsAbs=stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
: 2(
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
�
j
L__inference_stream_0_drop_2_layer_call_and_return_conditional_losses_5544995

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:����������@2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:����������@2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������@:T P
,
_output_shapes
:����������@
 
_user_specified_nameinputs
�+
�
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_5544874

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
�+
�
P__inference_batch_normalization_layer_call_and_return_conditional_losses_5544641

inputs5
'assignmovingavg_readvariableop_resource: 7
)assignmovingavg_1_readvariableop_resource: 3
%batchnorm_mul_readvariableop_resource: /
!batchnorm_readvariableop_resource: 
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
: *
	keep_dims(2
moments/mean�
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
: 2
moments/StopGradient�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :������������������ 2
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
: *
	keep_dims(2
moments/variance�
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/Squeeze�
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
: *
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
: *
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
: 2
AssignMovingAvg/sub�
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
: 2
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
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
: 2
AssignMovingAvg_1/sub�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
: 2
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
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/Rsqrt�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/mul/ReadVariableOp�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/mul�
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :������������������ 2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2�
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp�
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :������������������ 2
batchnorm/add_1{
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :������������������ 2

Identity�
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:������������������ : : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :������������������ 
 
_user_specified_nameinputs
�
�
P__inference_batch_normalization_layer_call_and_return_conditional_losses_5544661

inputs/
!batchnorm_readvariableop_resource: 3
%batchnorm_mul_readvariableop_resource: 1
#batchnorm_readvariableop_1_resource: 1
#batchnorm_readvariableop_2_resource: 
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOp�
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
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
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/Rsqrt�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/mul/ReadVariableOp�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/mul{
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*,
_output_shapes
:���������� 2
batchnorm/mul_1�
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp_1�
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2�
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp_2�
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*,
_output_shapes
:���������� 2
batchnorm/add_1s
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*,
_output_shapes
:���������� 2

Identity�
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������� : : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:T P
,
_output_shapes
:���������� 
 
_user_specified_nameinputs
��
�
F__inference_basemodel_layer_call_and_return_conditional_losses_5544324
inputs_0Q
;stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource: =
/stream_0_conv_1_biasadd_readvariableop_resource: I
;batch_normalization_assignmovingavg_readvariableop_resource: K
=batch_normalization_assignmovingavg_1_readvariableop_resource: G
9batch_normalization_batchnorm_mul_readvariableop_resource: C
5batch_normalization_batchnorm_readvariableop_resource: Q
;stream_0_conv_2_conv1d_expanddims_1_readvariableop_resource: @=
/stream_0_conv_2_biasadd_readvariableop_resource:@K
=batch_normalization_1_assignmovingavg_readvariableop_resource:@M
?batch_normalization_1_assignmovingavg_1_readvariableop_resource:@I
;batch_normalization_1_batchnorm_mul_readvariableop_resource:@E
7batch_normalization_1_batchnorm_readvariableop_resource:@8
&dense_1_matmul_readvariableop_resource:@T5
'dense_1_biasadd_readvariableop_resource:TK
=batch_normalization_2_assignmovingavg_readvariableop_resource:TM
?batch_normalization_2_assignmovingavg_1_readvariableop_resource:TI
;batch_normalization_2_batchnorm_mul_readvariableop_resource:TE
7batch_normalization_2_batchnorm_readvariableop_resource:T
identity��#batch_normalization/AssignMovingAvg�2batch_normalization/AssignMovingAvg/ReadVariableOp�%batch_normalization/AssignMovingAvg_1�4batch_normalization/AssignMovingAvg_1/ReadVariableOp�,batch_normalization/batchnorm/ReadVariableOp�0batch_normalization/batchnorm/mul/ReadVariableOp�%batch_normalization_1/AssignMovingAvg�4batch_normalization_1/AssignMovingAvg/ReadVariableOp�'batch_normalization_1/AssignMovingAvg_1�6batch_normalization_1/AssignMovingAvg_1/ReadVariableOp�.batch_normalization_1/batchnorm/ReadVariableOp�2batch_normalization_1/batchnorm/mul/ReadVariableOp�%batch_normalization_2/AssignMovingAvg�4batch_normalization_2/AssignMovingAvg/ReadVariableOp�'batch_normalization_2/AssignMovingAvg_1�6batch_normalization_2/AssignMovingAvg_1/ReadVariableOp�.batch_normalization_2/batchnorm/ReadVariableOp�2batch_normalization_2/batchnorm/mul/ReadVariableOp�dense_1/BiasAdd/ReadVariableOp�dense_1/MatMul/ReadVariableOp�-dense_1/kernel/Regularizer/Abs/ReadVariableOp�&stream_0_conv_1/BiasAdd/ReadVariableOp�2stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp�5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp�&stream_0_conv_2/BiasAdd/ReadVariableOp�2stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOp�8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp�
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
:����������2!
stream_0_input_drop/dropout/Mul~
!stream_0_input_drop/dropout/ShapeShapeinputs_0*
T0*
_output_shapes
:2#
!stream_0_input_drop/dropout/Shape�
8stream_0_input_drop/dropout/random_uniform/RandomUniformRandomUniform*stream_0_input_drop/dropout/Shape:output:0*
T0*,
_output_shapes
:����������*
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
:����������2*
(stream_0_input_drop/dropout/GreaterEqual�
 stream_0_input_drop/dropout/CastCast,stream_0_input_drop/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:����������2"
 stream_0_input_drop/dropout/Cast�
!stream_0_input_drop/dropout/Mul_1Mul#stream_0_input_drop/dropout/Mul:z:0$stream_0_input_drop/dropout/Cast:y:0*
T0*,
_output_shapes
:����������2#
!stream_0_input_drop/dropout/Mul_1�
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
:����������2#
!stream_0_conv_1/conv1d/ExpandDims�
2stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp;stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
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
: 2%
#stream_0_conv_1/conv1d/ExpandDims_1�
stream_0_conv_1/conv1dConv2D*stream_0_conv_1/conv1d/ExpandDims:output:0,stream_0_conv_1/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:���������� *
paddingSAME*
strides
2
stream_0_conv_1/conv1d�
stream_0_conv_1/conv1d/SqueezeSqueezestream_0_conv_1/conv1d:output:0*
T0*,
_output_shapes
:���������� *
squeeze_dims

���������2 
stream_0_conv_1/conv1d/Squeeze�
&stream_0_conv_1/BiasAdd/ReadVariableOpReadVariableOp/stream_0_conv_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02(
&stream_0_conv_1/BiasAdd/ReadVariableOp�
stream_0_conv_1/BiasAddBiasAdd'stream_0_conv_1/conv1d/Squeeze:output:0.stream_0_conv_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������� 2
stream_0_conv_1/BiasAdd�
2batch_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       24
2batch_normalization/moments/mean/reduction_indices�
 batch_normalization/moments/meanMean stream_0_conv_1/BiasAdd:output:0;batch_normalization/moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(2"
 batch_normalization/moments/mean�
(batch_normalization/moments/StopGradientStopGradient)batch_normalization/moments/mean:output:0*
T0*"
_output_shapes
: 2*
(batch_normalization/moments/StopGradient�
-batch_normalization/moments/SquaredDifferenceSquaredDifference stream_0_conv_1/BiasAdd:output:01batch_normalization/moments/StopGradient:output:0*
T0*,
_output_shapes
:���������� 2/
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
: *
	keep_dims(2&
$batch_normalization/moments/variance�
#batch_normalization/moments/SqueezeSqueeze)batch_normalization/moments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2%
#batch_normalization/moments/Squeeze�
%batch_normalization/moments/Squeeze_1Squeeze-batch_normalization/moments/variance:output:0*
T0*
_output_shapes
: *
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
: *
dtype024
2batch_normalization/AssignMovingAvg/ReadVariableOp�
'batch_normalization/AssignMovingAvg/subSub:batch_normalization/AssignMovingAvg/ReadVariableOp:value:0,batch_normalization/moments/Squeeze:output:0*
T0*
_output_shapes
: 2)
'batch_normalization/AssignMovingAvg/sub�
'batch_normalization/AssignMovingAvg/mulMul+batch_normalization/AssignMovingAvg/sub:z:02batch_normalization/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
: 2)
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
: *
dtype026
4batch_normalization/AssignMovingAvg_1/ReadVariableOp�
)batch_normalization/AssignMovingAvg_1/subSub<batch_normalization/AssignMovingAvg_1/ReadVariableOp:value:0.batch_normalization/moments/Squeeze_1:output:0*
T0*
_output_shapes
: 2+
)batch_normalization/AssignMovingAvg_1/sub�
)batch_normalization/AssignMovingAvg_1/mulMul-batch_normalization/AssignMovingAvg_1/sub:z:04batch_normalization/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
: 2+
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
: 2#
!batch_normalization/batchnorm/add�
#batch_normalization/batchnorm/RsqrtRsqrt%batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes
: 2%
#batch_normalization/batchnorm/Rsqrt�
0batch_normalization/batchnorm/mul/ReadVariableOpReadVariableOp9batch_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype022
0batch_normalization/batchnorm/mul/ReadVariableOp�
!batch_normalization/batchnorm/mulMul'batch_normalization/batchnorm/Rsqrt:y:08batch_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2#
!batch_normalization/batchnorm/mul�
#batch_normalization/batchnorm/mul_1Mul stream_0_conv_1/BiasAdd:output:0%batch_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:���������� 2%
#batch_normalization/batchnorm/mul_1�
#batch_normalization/batchnorm/mul_2Mul,batch_normalization/moments/Squeeze:output:0%batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes
: 2%
#batch_normalization/batchnorm/mul_2�
,batch_normalization/batchnorm/ReadVariableOpReadVariableOp5batch_normalization_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02.
,batch_normalization/batchnorm/ReadVariableOp�
!batch_normalization/batchnorm/subSub4batch_normalization/batchnorm/ReadVariableOp:value:0'batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2#
!batch_normalization/batchnorm/sub�
#batch_normalization/batchnorm/add_1AddV2'batch_normalization/batchnorm/mul_1:z:0%batch_normalization/batchnorm/sub:z:0*
T0*,
_output_shapes
:���������� 2%
#batch_normalization/batchnorm/add_1�
activation/ReluRelu'batch_normalization/batchnorm/add_1:z:0*
T0*,
_output_shapes
:���������� 2
activation/Relu�
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
:���������� 2
stream_0_drop_1/dropout/Mul�
stream_0_drop_1/dropout/ShapeShapeactivation/Relu:activations:0*
T0*
_output_shapes
:2
stream_0_drop_1/dropout/Shape�
4stream_0_drop_1/dropout/random_uniform/RandomUniformRandomUniform&stream_0_drop_1/dropout/Shape:output:0*
T0*,
_output_shapes
:���������� *
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
:���������� 2&
$stream_0_drop_1/dropout/GreaterEqual�
stream_0_drop_1/dropout/CastCast(stream_0_drop_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:���������� 2
stream_0_drop_1/dropout/Cast�
stream_0_drop_1/dropout/Mul_1Mulstream_0_drop_1/dropout/Mul:z:0 stream_0_drop_1/dropout/Cast:y:0*
T0*,
_output_shapes
:���������� 2
stream_0_drop_1/dropout/Mul_1�
%stream_0_conv_2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2'
%stream_0_conv_2/conv1d/ExpandDims/dim�
!stream_0_conv_2/conv1d/ExpandDims
ExpandDims!stream_0_drop_1/dropout/Mul_1:z:0.stream_0_conv_2/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:���������� 2#
!stream_0_conv_2/conv1d/ExpandDims�
2stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp;stream_0_conv_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: @*
dtype024
2stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOp�
'stream_0_conv_2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2)
'stream_0_conv_2/conv1d/ExpandDims_1/dim�
#stream_0_conv_2/conv1d/ExpandDims_1
ExpandDims:stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOp:value:00stream_0_conv_2/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: @2%
#stream_0_conv_2/conv1d/ExpandDims_1�
stream_0_conv_2/conv1dConv2D*stream_0_conv_2/conv1d/ExpandDims:output:0,stream_0_conv_2/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:����������@*
paddingSAME*
strides
2
stream_0_conv_2/conv1d�
stream_0_conv_2/conv1d/SqueezeSqueezestream_0_conv_2/conv1d:output:0*
T0*,
_output_shapes
:����������@*
squeeze_dims

���������2 
stream_0_conv_2/conv1d/Squeeze�
&stream_0_conv_2/BiasAdd/ReadVariableOpReadVariableOp/stream_0_conv_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02(
&stream_0_conv_2/BiasAdd/ReadVariableOp�
stream_0_conv_2/BiasAddBiasAdd'stream_0_conv_2/conv1d/Squeeze:output:0.stream_0_conv_2/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������@2
stream_0_conv_2/BiasAdd�
4batch_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       26
4batch_normalization_1/moments/mean/reduction_indices�
"batch_normalization_1/moments/meanMean stream_0_conv_2/BiasAdd:output:0=batch_normalization_1/moments/mean/reduction_indices:output:0*
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
/batch_normalization_1/moments/SquaredDifferenceSquaredDifference stream_0_conv_2/BiasAdd:output:03batch_normalization_1/moments/StopGradient:output:0*
T0*,
_output_shapes
:����������@21
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
%batch_normalization_1/batchnorm/mul_1Mul stream_0_conv_2/BiasAdd:output:0'batch_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:����������@2'
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
:����������@2'
%batch_normalization_1/batchnorm/add_1�
activation_1/ReluRelu)batch_normalization_1/batchnorm/add_1:z:0*
T0*,
_output_shapes
:����������@2
activation_1/Relu�
stream_0_drop_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2
stream_0_drop_2/dropout/Const�
stream_0_drop_2/dropout/MulMulactivation_1/Relu:activations:0&stream_0_drop_2/dropout/Const:output:0*
T0*,
_output_shapes
:����������@2
stream_0_drop_2/dropout/Mul�
stream_0_drop_2/dropout/ShapeShapeactivation_1/Relu:activations:0*
T0*
_output_shapes
:2
stream_0_drop_2/dropout/Shape�
4stream_0_drop_2/dropout/random_uniform/RandomUniformRandomUniform&stream_0_drop_2/dropout/Shape:output:0*
T0*,
_output_shapes
:����������@*
dtype0*
seed�*
seed2�26
4stream_0_drop_2/dropout/random_uniform/RandomUniform�
&stream_0_drop_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���>2(
&stream_0_drop_2/dropout/GreaterEqual/y�
$stream_0_drop_2/dropout/GreaterEqualGreaterEqual=stream_0_drop_2/dropout/random_uniform/RandomUniform:output:0/stream_0_drop_2/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:����������@2&
$stream_0_drop_2/dropout/GreaterEqual�
stream_0_drop_2/dropout/CastCast(stream_0_drop_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:����������@2
stream_0_drop_2/dropout/Cast�
stream_0_drop_2/dropout/Mul_1Mulstream_0_drop_2/dropout/Mul:z:0 stream_0_drop_2/dropout/Cast:y:0*
T0*,
_output_shapes
:����������@2
stream_0_drop_2/dropout/Mul_1�
/global_average_pooling1d/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :21
/global_average_pooling1d/Mean/reduction_indices�
global_average_pooling1d/MeanMean!stream_0_drop_2/dropout/Mul_1:z:08global_average_pooling1d/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:���������@2
global_average_pooling1d/Mean�
concatenate/concat/concat_dimConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate/concat/concat_dim�
concatenate/concat/concatIdentity&global_average_pooling1d/Mean:output:0*
T0*'
_output_shapes
:���������@2
concatenate/concat/concat�
dense_1_dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
dense_1_dropout/dropout/Const�
dense_1_dropout/dropout/MulMul"concatenate/concat/concat:output:0&dense_1_dropout/dropout/Const:output:0*
T0*'
_output_shapes
:���������@2
dense_1_dropout/dropout/Mul�
dense_1_dropout/dropout/ShapeShape"concatenate/concat/concat:output:0*
T0*
_output_shapes
:2
dense_1_dropout/dropout/Shape�
4dense_1_dropout/dropout/random_uniform/RandomUniformRandomUniform&dense_1_dropout/dropout/Shape:output:0*
T0*'
_output_shapes
:���������@*
dtype0*
seed�26
4dense_1_dropout/dropout/random_uniform/RandomUniform�
&dense_1_dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>2(
&dense_1_dropout/dropout/GreaterEqual/y�
$dense_1_dropout/dropout/GreaterEqualGreaterEqual=dense_1_dropout/dropout/random_uniform/RandomUniform:output:0/dense_1_dropout/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:���������@2&
$dense_1_dropout/dropout/GreaterEqual�
dense_1_dropout/dropout/CastCast(dense_1_dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:���������@2
dense_1_dropout/dropout/Cast�
dense_1_dropout/dropout/Mul_1Muldense_1_dropout/dropout/Mul:z:0 dense_1_dropout/dropout/Cast:y:0*
T0*'
_output_shapes
:���������@2
dense_1_dropout/dropout/Mul_1�
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:@T*
dtype02
dense_1/MatMul/ReadVariableOp�
dense_1/MatMulMatMul!dense_1_dropout/dropout/Mul_1:z:0%dense_1/MatMul/ReadVariableOp:value:0*
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
4batch_normalization_2/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 26
4batch_normalization_2/moments/mean/reduction_indices�
"batch_normalization_2/moments/meanMeandense_1/BiasAdd:output:0=batch_normalization_2/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:T*
	keep_dims(2$
"batch_normalization_2/moments/mean�
*batch_normalization_2/moments/StopGradientStopGradient+batch_normalization_2/moments/mean:output:0*
T0*
_output_shapes

:T2,
*batch_normalization_2/moments/StopGradient�
/batch_normalization_2/moments/SquaredDifferenceSquaredDifferencedense_1/BiasAdd:output:03batch_normalization_2/moments/StopGradient:output:0*
T0*'
_output_shapes
:���������T21
/batch_normalization_2/moments/SquaredDifference�
8batch_normalization_2/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2:
8batch_normalization_2/moments/variance/reduction_indices�
&batch_normalization_2/moments/varianceMean3batch_normalization_2/moments/SquaredDifference:z:0Abatch_normalization_2/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:T*
	keep_dims(2(
&batch_normalization_2/moments/variance�
%batch_normalization_2/moments/SqueezeSqueeze+batch_normalization_2/moments/mean:output:0*
T0*
_output_shapes
:T*
squeeze_dims
 2'
%batch_normalization_2/moments/Squeeze�
'batch_normalization_2/moments/Squeeze_1Squeeze/batch_normalization_2/moments/variance:output:0*
T0*
_output_shapes
:T*
squeeze_dims
 2)
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
:T*
dtype026
4batch_normalization_2/AssignMovingAvg/ReadVariableOp�
)batch_normalization_2/AssignMovingAvg/subSub<batch_normalization_2/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_2/moments/Squeeze:output:0*
T0*
_output_shapes
:T2+
)batch_normalization_2/AssignMovingAvg/sub�
)batch_normalization_2/AssignMovingAvg/mulMul-batch_normalization_2/AssignMovingAvg/sub:z:04batch_normalization_2/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:T2+
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
:T*
dtype028
6batch_normalization_2/AssignMovingAvg_1/ReadVariableOp�
+batch_normalization_2/AssignMovingAvg_1/subSub>batch_normalization_2/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_2/moments/Squeeze_1:output:0*
T0*
_output_shapes
:T2-
+batch_normalization_2/AssignMovingAvg_1/sub�
+batch_normalization_2/AssignMovingAvg_1/mulMul/batch_normalization_2/AssignMovingAvg_1/sub:z:06batch_normalization_2/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:T2-
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
:T2%
#batch_normalization_2/batchnorm/add�
%batch_normalization_2/batchnorm/RsqrtRsqrt'batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes
:T2'
%batch_normalization_2/batchnorm/Rsqrt�
2batch_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes
:T*
dtype024
2batch_normalization_2/batchnorm/mul/ReadVariableOp�
#batch_normalization_2/batchnorm/mulMul)batch_normalization_2/batchnorm/Rsqrt:y:0:batch_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:T2%
#batch_normalization_2/batchnorm/mul�
%batch_normalization_2/batchnorm/mul_1Muldense_1/BiasAdd:output:0'batch_normalization_2/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������T2'
%batch_normalization_2/batchnorm/mul_1�
%batch_normalization_2/batchnorm/mul_2Mul.batch_normalization_2/moments/Squeeze:output:0'batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes
:T2'
%batch_normalization_2/batchnorm/mul_2�
.batch_normalization_2/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_2_batchnorm_readvariableop_resource*
_output_shapes
:T*
dtype020
.batch_normalization_2/batchnorm/ReadVariableOp�
#batch_normalization_2/batchnorm/subSub6batch_normalization_2/batchnorm/ReadVariableOp:value:0)batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes
:T2%
#batch_normalization_2/batchnorm/sub�
%batch_normalization_2/batchnorm/add_1AddV2)batch_normalization_2/batchnorm/mul_1:z:0'batch_normalization_2/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������T2'
%batch_normalization_2/batchnorm/add_1�
dense_activation_1/SigmoidSigmoid)batch_normalization_2/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������T2
dense_activation_1/Sigmoid�
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp;stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype027
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp�
&stream_0_conv_1/kernel/Regularizer/AbsAbs=stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
: 2(
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
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;stream_0_conv_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: @*
dtype02:
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp�
)stream_0_conv_2/kernel/Regularizer/SquareSquare@stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
: @2+
)stream_0_conv_2/kernel/Regularizer/Square�
(stream_0_conv_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_0_conv_2/kernel/Regularizer/Const�
&stream_0_conv_2/kernel/Regularizer/SumSum-stream_0_conv_2/kernel/Regularizer/Square:y:01stream_0_conv_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_2/kernel/Regularizer/Sum�
(stream_0_conv_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2*
(stream_0_conv_2/kernel/Regularizer/mul/x�
&stream_0_conv_2/kernel/Regularizer/mulMul1stream_0_conv_2/kernel/Regularizer/mul/x:output:0/stream_0_conv_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_2/kernel/Regularizer/mul�
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:@T*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOp�
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:@T2 
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

NoOpNoOp$^batch_normalization/AssignMovingAvg3^batch_normalization/AssignMovingAvg/ReadVariableOp&^batch_normalization/AssignMovingAvg_15^batch_normalization/AssignMovingAvg_1/ReadVariableOp-^batch_normalization/batchnorm/ReadVariableOp1^batch_normalization/batchnorm/mul/ReadVariableOp&^batch_normalization_1/AssignMovingAvg5^batch_normalization_1/AssignMovingAvg/ReadVariableOp(^batch_normalization_1/AssignMovingAvg_17^batch_normalization_1/AssignMovingAvg_1/ReadVariableOp/^batch_normalization_1/batchnorm/ReadVariableOp3^batch_normalization_1/batchnorm/mul/ReadVariableOp&^batch_normalization_2/AssignMovingAvg5^batch_normalization_2/AssignMovingAvg/ReadVariableOp(^batch_normalization_2/AssignMovingAvg_17^batch_normalization_2/AssignMovingAvg_1/ReadVariableOp/^batch_normalization_2/batchnorm/ReadVariableOp3^batch_normalization_2/batchnorm/mul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp.^dense_1/kernel/Regularizer/Abs/ReadVariableOp'^stream_0_conv_1/BiasAdd/ReadVariableOp3^stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp'^stream_0_conv_2/BiasAdd/ReadVariableOp3^stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOp9^stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:����������: : : : : : : : : : : : : : : : : : 2J
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
2batch_normalization_2/batchnorm/mul/ReadVariableOp2batch_normalization_2/batchnorm/mul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2^
-dense_1/kernel/Regularizer/Abs/ReadVariableOp-dense_1/kernel/Regularizer/Abs/ReadVariableOp2P
&stream_0_conv_1/BiasAdd/ReadVariableOp&stream_0_conv_1/BiasAdd/ReadVariableOp2h
2stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp2stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp2n
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp2P
&stream_0_conv_2/BiasAdd/ReadVariableOp&stream_0_conv_2/BiasAdd/ReadVariableOp2h
2stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOp2stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOp2t
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp:V R
,
_output_shapes
:����������
"
_user_specified_name
inputs/0
�
n
P__inference_stream_0_input_drop_layer_call_and_return_conditional_losses_5544529

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:����������2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:����������2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
e
I__inference_activation_1_layer_call_and_return_conditional_losses_5544985

inputs
identityS
ReluReluinputs*
T0*,
_output_shapes
:����������@2
Reluk
IdentityIdentityRelu:activations:0*
T0*,
_output_shapes
:����������@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������@:T P
,
_output_shapes
:����������@
 
_user_specified_nameinputs
�
�
+__inference_basemodel_layer_call_fn_5542022
inputs_0
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: 
	unknown_5: @
	unknown_6:@
	unknown_7:@
	unknown_8:@
	unknown_9:@

unknown_10:@

unknown_11:@T

unknown_12:T

unknown_13:T

unknown_14:T

unknown_15:T

unknown_16:T
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������T*.
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_basemodel_layer_call_and_return_conditional_losses_55419422
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
_construction_contextkEagerRuntime*O
_input_shapes>
<:����������: : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
,
_output_shapes
:����������
"
_user_specified_name
inputs_0
�
k
L__inference_stream_0_drop_1_layer_call_and_return_conditional_losses_5541724

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
:���������� 2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:���������� *
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
:���������� 2
dropout/GreaterEqual�
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:���������� 2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:���������� 2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:���������� 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:���������� :T P
,
_output_shapes
:���������� 
 
_user_specified_nameinputs
�
�
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_5544894

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
:����������@2
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
:����������@2
batchnorm/add_1s
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*,
_output_shapes
:����������@2

Identity�
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������@: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:T P
,
_output_shapes
:����������@
 
_user_specified_nameinputs
�
�
L__inference_stream_0_conv_2_layer_call_and_return_conditional_losses_5541388

inputsA
+conv1d_expanddims_1_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�"conv1d/ExpandDims_1/ReadVariableOp�8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOpy
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
:���������� 2
conv1d/ExpandDims�
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: @*
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
: @2
conv1d/ExpandDims_1�
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:����������@*
paddingSAME*
strides
2
conv1d�
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:����������@*
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
:����������@2	
BiasAdd�
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: @*
dtype02:
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp�
)stream_0_conv_2/kernel/Regularizer/SquareSquare@stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
: @2+
)stream_0_conv_2/kernel/Regularizer/Square�
(stream_0_conv_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_0_conv_2/kernel/Regularizer/Const�
&stream_0_conv_2/kernel/Regularizer/SumSum-stream_0_conv_2/kernel/Regularizer/Square:y:01stream_0_conv_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_2/kernel/Regularizer/Sum�
(stream_0_conv_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2*
(stream_0_conv_2/kernel/Regularizer/mul/x�
&stream_0_conv_2/kernel/Regularizer/mulMul1stream_0_conv_2/kernel/Regularizer/mul/x:output:0/stream_0_conv_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_2/kernel/Regularizer/mulp
IdentityIdentityBiasAdd:output:0^NoOp*
T0*,
_output_shapes
:����������@2

Identity�
NoOpNoOp^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp9^stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:���������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp2t
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp:T P
,
_output_shapes
:���������� 
 
_user_specified_nameinputs
�
�
P__inference_batch_normalization_layer_call_and_return_conditional_losses_5540797

inputs/
!batchnorm_readvariableop_resource: 3
%batchnorm_mul_readvariableop_resource: 1
#batchnorm_readvariableop_1_resource: 1
#batchnorm_readvariableop_2_resource: 
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOp�
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
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
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/Rsqrt�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/mul/ReadVariableOp�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/mul�
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :������������������ 2
batchnorm/mul_1�
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp_1�
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2�
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp_2�
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :������������������ 2
batchnorm/add_1{
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :������������������ 2

Identity�
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:������������������ : : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :������������������ 
 
_user_specified_nameinputs
�
�
+__inference_basemodel_layer_call_fn_5544488
inputs_0
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: 
	unknown_5: @
	unknown_6:@
	unknown_7:@
	unknown_8:@
	unknown_9:@

unknown_10:@

unknown_11:@T

unknown_12:T

unknown_13:T

unknown_14:T

unknown_15:T

unknown_16:T
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������T*.
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_basemodel_layer_call_and_return_conditional_losses_55426612
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
_construction_contextkEagerRuntime*O
_input_shapes>
<:����������: : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
,
_output_shapes
:����������
"
_user_specified_name
inputs/0
�
j
1__inference_dense_1_dropout_layer_call_fn_5545077

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
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
GPU2*0J 8� *U
fPRN
L__inference_dense_1_dropout_layer_call_and_return_conditional_losses_55415912
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������@22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
k
O__inference_dense_activation_1_layer_call_and_return_conditional_losses_5545193

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
�
d
H__inference_concatenate_layer_call_and_return_conditional_losses_5541450

inputs
identityh
concat/concat_dimConst*
_output_shapes
: *
dtype0*
value	B :2
concat/concat_dimd
concat/concatIdentityinputs*
T0*'
_output_shapes
:���������@2
concat/concatj
IdentityIdentityconcat/concat:output:0*
T0*'
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������@:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
7__inference_batch_normalization_1_layer_call_fn_5544967

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
:����������@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *[
fVRT
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_55414132
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:����������@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:����������@
 
_user_specified_nameinputs
�
�
+__inference_basemodel_layer_call_fn_5544447
inputs_0
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: 
	unknown_5: @
	unknown_6:@
	unknown_7:@
	unknown_8:@
	unknown_9:@

unknown_10:@

unknown_11:@T

unknown_12:T

unknown_13:T

unknown_14:T

unknown_15:T

unknown_16:T
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������T*4
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_basemodel_layer_call_and_return_conditional_losses_55422852
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
_construction_contextkEagerRuntime*O
_input_shapes>
<:����������: : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
,
_output_shapes
:����������
"
_user_specified_name
inputs/0
�
Q
5__inference_stream_0_input_drop_layer_call_fn_5544546

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
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *Y
fTRR
P__inference_stream_0_input_drop_layer_call_and_return_conditional_losses_55412952
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_5544840

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
�
�
1__inference_stream_0_conv_1_layer_call_fn_5544587

inputs
unknown: 
	unknown_0: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_stream_0_conv_1_layer_call_and_return_conditional_losses_55413182
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:���������� 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
n
5__inference_stream_0_input_drop_layer_call_fn_5544551

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
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *Y
fTRR
P__inference_stream_0_input_drop_layer_call_and_return_conditional_losses_55418232
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:����������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
o
P__inference_stream_0_input_drop_layer_call_and_return_conditional_losses_5544541

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
:����������2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:����������*
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
:����������2
dropout/GreaterEqual�
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:����������2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:����������2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
P
4__inference_dense_activation_1_layer_call_fn_5545198

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
O__inference_dense_activation_1_layer_call_and_return_conditional_losses_55414952
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
�g
�
 __inference__traced_save_5545402
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
5savev2_batch_normalization_1_beta_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop:
6savev2_batch_normalization_2_gamma_read_readvariableop9
5savev2_batch_normalization_2_beta_read_readvariableop>
:savev2_batch_normalization_moving_mean_read_readvariableopB
>savev2_batch_normalization_moving_variance_read_readvariableop@
<savev2_batch_normalization_1_moving_mean_read_readvariableopD
@savev2_batch_normalization_1_moving_variance_read_readvariableop@
<savev2_batch_normalization_2_moving_mean_read_readvariableopD
@savev2_batch_normalization_2_moving_variance_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop<
8savev2_adam_stream_0_conv_1_kernel_m_read_readvariableop:
6savev2_adam_stream_0_conv_1_bias_m_read_readvariableop?
;savev2_adam_batch_normalization_gamma_m_read_readvariableop>
:savev2_adam_batch_normalization_beta_m_read_readvariableop<
8savev2_adam_stream_0_conv_2_kernel_m_read_readvariableop:
6savev2_adam_stream_0_conv_2_bias_m_read_readvariableopA
=savev2_adam_batch_normalization_1_gamma_m_read_readvariableop@
<savev2_adam_batch_normalization_1_beta_m_read_readvariableop4
0savev2_adam_dense_1_kernel_m_read_readvariableop2
.savev2_adam_dense_1_bias_m_read_readvariableopA
=savev2_adam_batch_normalization_2_gamma_m_read_readvariableop@
<savev2_adam_batch_normalization_2_beta_m_read_readvariableop<
8savev2_adam_stream_0_conv_1_kernel_v_read_readvariableop:
6savev2_adam_stream_0_conv_1_bias_v_read_readvariableop?
;savev2_adam_batch_normalization_gamma_v_read_readvariableop>
:savev2_adam_batch_normalization_beta_v_read_readvariableop<
8savev2_adam_stream_0_conv_2_kernel_v_read_readvariableop:
6savev2_adam_stream_0_conv_2_bias_v_read_readvariableopA
=savev2_adam_batch_normalization_1_gamma_v_read_readvariableop@
<savev2_adam_batch_normalization_1_beta_v_read_readvariableop4
0savev2_adam_dense_1_kernel_v_read_readvariableop2
.savev2_adam_dense_1_bias_v_read_readvariableopA
=savev2_adam_batch_normalization_2_gamma_v_read_readvariableop@
<savev2_adam_batch_normalization_2_beta_v_read_readvariableop
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
ShardedFilename�
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:2*
dtype0*�
value�B�2B+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:2*
dtype0*w
valuenBl2B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices�
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0!savev2_beta_1_read_readvariableop!savev2_beta_2_read_readvariableop savev2_decay_read_readvariableop(savev2_learning_rate_read_readvariableop$savev2_adam_iter_read_readvariableop1savev2_stream_0_conv_1_kernel_read_readvariableop/savev2_stream_0_conv_1_bias_read_readvariableop4savev2_batch_normalization_gamma_read_readvariableop3savev2_batch_normalization_beta_read_readvariableop1savev2_stream_0_conv_2_kernel_read_readvariableop/savev2_stream_0_conv_2_bias_read_readvariableop6savev2_batch_normalization_1_gamma_read_readvariableop5savev2_batch_normalization_1_beta_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop6savev2_batch_normalization_2_gamma_read_readvariableop5savev2_batch_normalization_2_beta_read_readvariableop:savev2_batch_normalization_moving_mean_read_readvariableop>savev2_batch_normalization_moving_variance_read_readvariableop<savev2_batch_normalization_1_moving_mean_read_readvariableop@savev2_batch_normalization_1_moving_variance_read_readvariableop<savev2_batch_normalization_2_moving_mean_read_readvariableop@savev2_batch_normalization_2_moving_variance_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop8savev2_adam_stream_0_conv_1_kernel_m_read_readvariableop6savev2_adam_stream_0_conv_1_bias_m_read_readvariableop;savev2_adam_batch_normalization_gamma_m_read_readvariableop:savev2_adam_batch_normalization_beta_m_read_readvariableop8savev2_adam_stream_0_conv_2_kernel_m_read_readvariableop6savev2_adam_stream_0_conv_2_bias_m_read_readvariableop=savev2_adam_batch_normalization_1_gamma_m_read_readvariableop<savev2_adam_batch_normalization_1_beta_m_read_readvariableop0savev2_adam_dense_1_kernel_m_read_readvariableop.savev2_adam_dense_1_bias_m_read_readvariableop=savev2_adam_batch_normalization_2_gamma_m_read_readvariableop<savev2_adam_batch_normalization_2_beta_m_read_readvariableop8savev2_adam_stream_0_conv_1_kernel_v_read_readvariableop6savev2_adam_stream_0_conv_1_bias_v_read_readvariableop;savev2_adam_batch_normalization_gamma_v_read_readvariableop:savev2_adam_batch_normalization_beta_v_read_readvariableop8savev2_adam_stream_0_conv_2_kernel_v_read_readvariableop6savev2_adam_stream_0_conv_2_bias_v_read_readvariableop=savev2_adam_batch_normalization_1_gamma_v_read_readvariableop<savev2_adam_batch_normalization_1_beta_v_read_readvariableop0savev2_adam_dense_1_kernel_v_read_readvariableop.savev2_adam_dense_1_bias_v_read_readvariableop=savev2_adam_batch_normalization_2_gamma_v_read_readvariableop<savev2_adam_batch_normalization_2_beta_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *@
dtypes6
422	2
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

identity_1Identity_1:output:0*�
_input_shapes�
�: : : : : : : : : : : @:@:@:@:@T:T:T:T: : :@:@:T:T: : : : : : : @:@:@:@:@T:T:T:T: : : : : @:@:@:@:@T:T:T:T: 2(
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
: : 

_output_shapes
: : 

_output_shapes
: : 	

_output_shapes
: :(
$
"
_output_shapes
: @: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@:$ 

_output_shapes

:@T: 

_output_shapes
:T: 

_output_shapes
:T: 

_output_shapes
:T: 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:T: 

_output_shapes
:T:

_output_shapes
: :

_output_shapes
: :($
"
_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: :($
"
_output_shapes
: @: 

_output_shapes
:@:  

_output_shapes
:@: !

_output_shapes
:@:$" 

_output_shapes

:@T: #

_output_shapes
:T: $

_output_shapes
:T: %

_output_shapes
:T:(&$
"
_output_shapes
: : '

_output_shapes
: : (

_output_shapes
: : )

_output_shapes
: :(*$
"
_output_shapes
: @: +

_output_shapes
:@: ,

_output_shapes
:@: -

_output_shapes
:@:$. 

_output_shapes

:@T: /

_output_shapes
:T: 0

_output_shapes
:T: 1

_output_shapes
:T:2

_output_shapes
: 
�
�
L__inference_stream_0_conv_2_layer_call_and_return_conditional_losses_5544811

inputsA
+conv1d_expanddims_1_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�"conv1d/ExpandDims_1/ReadVariableOp�8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOpy
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
:���������� 2
conv1d/ExpandDims�
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: @*
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
: @2
conv1d/ExpandDims_1�
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:����������@*
paddingSAME*
strides
2
conv1d�
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:����������@*
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
:����������@2	
BiasAdd�
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: @*
dtype02:
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp�
)stream_0_conv_2/kernel/Regularizer/SquareSquare@stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
: @2+
)stream_0_conv_2/kernel/Regularizer/Square�
(stream_0_conv_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_0_conv_2/kernel/Regularizer/Const�
&stream_0_conv_2/kernel/Regularizer/SumSum-stream_0_conv_2/kernel/Regularizer/Square:y:01stream_0_conv_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_2/kernel/Regularizer/Sum�
(stream_0_conv_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2*
(stream_0_conv_2/kernel/Regularizer/mul/x�
&stream_0_conv_2/kernel/Regularizer/mulMul1stream_0_conv_2/kernel/Regularizer/mul/x:output:0/stream_0_conv_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_2/kernel/Regularizer/mulp
IdentityIdentityBiasAdd:output:0^NoOp*
T0*,
_output_shapes
:����������@2

Identity�
NoOpNoOp^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp9^stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:���������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp2t
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp:T P
,
_output_shapes
:���������� 
 
_user_specified_nameinputs
�+
�
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_5544928

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
:����������@2
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
:����������@2
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
:����������@2
batchnorm/add_1s
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*,
_output_shapes
:����������@2

Identity�
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������@: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:T P
,
_output_shapes
:����������@
 
_user_specified_nameinputs
�
f
H__inference_concatenate_layer_call_and_return_conditional_losses_5545045
inputs_0
identityh
concat/concat_dimConst*
_output_shapes
: *
dtype0*
value	B :2
concat/concat_dimf
concat/concatIdentityinputs_0*
T0*'
_output_shapes
:���������@2
concat/concatj
IdentityIdentityconcat/concat:output:0*
T0*'
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������@:Q M
'
_output_shapes
:���������@
"
_user_specified_name
inputs/0
�
k
L__inference_stream_0_drop_1_layer_call_and_return_conditional_losses_5544774

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
:���������� 2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:���������� *
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
:���������� 2
dropout/GreaterEqual�
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:���������� 2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:���������� 2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:���������� 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:���������� :T P
,
_output_shapes
:���������� 
 
_user_specified_nameinputs
�
e
I__inference_activation_1_layer_call_and_return_conditional_losses_5541428

inputs
identityS
ReluReluinputs*
T0*,
_output_shapes
:����������@2
Reluk
IdentityIdentityRelu:activations:0*
T0*,
_output_shapes
:����������@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������@:T P
,
_output_shapes
:����������@
 
_user_specified_nameinputs
�
j
L__inference_dense_1_dropout_layer_call_and_return_conditional_losses_5541457

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:���������@2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:���������@2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������@:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
j
L__inference_stream_0_drop_2_layer_call_and_return_conditional_losses_5541435

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:����������@2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:����������@2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������@:T P
,
_output_shapes
:����������@
 
_user_specified_nameinputs
�g
�

F__inference_basemodel_layer_call_and_return_conditional_losses_5542095
inputs_0-
stream_0_conv_1_5542026: %
stream_0_conv_1_5542028: )
batch_normalization_5542031: )
batch_normalization_5542033: )
batch_normalization_5542035: )
batch_normalization_5542037: -
stream_0_conv_2_5542042: @%
stream_0_conv_2_5542044:@+
batch_normalization_1_5542047:@+
batch_normalization_1_5542049:@+
batch_normalization_1_5542051:@+
batch_normalization_1_5542053:@!
dense_1_5542061:@T
dense_1_5542063:T+
batch_normalization_2_5542066:T+
batch_normalization_2_5542068:T+
batch_normalization_2_5542070:T+
batch_normalization_2_5542072:T
identity��+batch_normalization/StatefulPartitionedCall�-batch_normalization_1/StatefulPartitionedCall�-batch_normalization_2/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�-dense_1/kernel/Regularizer/Abs/ReadVariableOp�'stream_0_conv_1/StatefulPartitionedCall�5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp�'stream_0_conv_2/StatefulPartitionedCall�8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp�
#stream_0_input_drop/PartitionedCallPartitionedCallinputs_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *Y
fTRR
P__inference_stream_0_input_drop_layer_call_and_return_conditional_losses_55412952%
#stream_0_input_drop/PartitionedCall�
'stream_0_conv_1/StatefulPartitionedCallStatefulPartitionedCall,stream_0_input_drop/PartitionedCall:output:0stream_0_conv_1_5542026stream_0_conv_1_5542028*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_stream_0_conv_1_layer_call_and_return_conditional_losses_55413182)
'stream_0_conv_1/StatefulPartitionedCall�
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall0stream_0_conv_1/StatefulPartitionedCall:output:0batch_normalization_5542031batch_normalization_5542033batch_normalization_5542035batch_normalization_5542037*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������� *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_55413432-
+batch_normalization/StatefulPartitionedCall�
activation/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������� * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_activation_layer_call_and_return_conditional_losses_55413582
activation/PartitionedCall�
stream_0_drop_1/PartitionedCallPartitionedCall#activation/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������� * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_stream_0_drop_1_layer_call_and_return_conditional_losses_55413652!
stream_0_drop_1/PartitionedCall�
'stream_0_conv_2/StatefulPartitionedCallStatefulPartitionedCall(stream_0_drop_1/PartitionedCall:output:0stream_0_conv_2_5542042stream_0_conv_2_5542044*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_stream_0_conv_2_layer_call_and_return_conditional_losses_55413882)
'stream_0_conv_2/StatefulPartitionedCall�
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall0stream_0_conv_2/StatefulPartitionedCall:output:0batch_normalization_1_5542047batch_normalization_1_5542049batch_normalization_1_5542051batch_normalization_1_5542053*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *[
fVRT
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_55414132/
-batch_normalization_1/StatefulPartitionedCall�
activation_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_activation_1_layer_call_and_return_conditional_losses_55414282
activation_1/PartitionedCall�
stream_0_drop_2/PartitionedCallPartitionedCall%activation_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_stream_0_drop_2_layer_call_and_return_conditional_losses_55414352!
stream_0_drop_2/PartitionedCall�
(global_average_pooling1d/PartitionedCallPartitionedCall(stream_0_drop_2/PartitionedCall:output:0*
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
U__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_55414422*
(global_average_pooling1d/PartitionedCall�
concatenate/PartitionedCallPartitionedCall1global_average_pooling1d/PartitionedCall:output:0*
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
GPU2*0J 8� *Q
fLRJ
H__inference_concatenate_layer_call_and_return_conditional_losses_55414502
concatenate/PartitionedCall�
dense_1_dropout/PartitionedCallPartitionedCall$concatenate/PartitionedCall:output:0*
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
GPU2*0J 8� *U
fPRN
L__inference_dense_1_dropout_layer_call_and_return_conditional_losses_55414572!
dense_1_dropout/PartitionedCall�
dense_1/StatefulPartitionedCallStatefulPartitionedCall(dense_1_dropout/PartitionedCall:output:0dense_1_5542061dense_1_5542063*
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
D__inference_dense_1_layer_call_and_return_conditional_losses_55414752!
dense_1/StatefulPartitionedCall�
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0batch_normalization_2_5542066batch_normalization_2_5542068batch_normalization_2_5542070batch_normalization_2_5542072*
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
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_55411452/
-batch_normalization_2/StatefulPartitionedCall�
"dense_activation_1/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
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
O__inference_dense_activation_1_layer_call_and_return_conditional_losses_55414952$
"dense_activation_1/PartitionedCall�
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpstream_0_conv_1_5542026*"
_output_shapes
: *
dtype027
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp�
&stream_0_conv_1/kernel/Regularizer/AbsAbs=stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
: 2(
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
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpstream_0_conv_2_5542042*"
_output_shapes
: @*
dtype02:
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp�
)stream_0_conv_2/kernel/Regularizer/SquareSquare@stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
: @2+
)stream_0_conv_2/kernel/Regularizer/Square�
(stream_0_conv_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_0_conv_2/kernel/Regularizer/Const�
&stream_0_conv_2/kernel/Regularizer/SumSum-stream_0_conv_2/kernel/Regularizer/Square:y:01stream_0_conv_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_2/kernel/Regularizer/Sum�
(stream_0_conv_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2*
(stream_0_conv_2/kernel/Regularizer/mul/x�
&stream_0_conv_2/kernel/Regularizer/mulMul1stream_0_conv_2/kernel/Regularizer/mul/x:output:0/stream_0_conv_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_2/kernel/Regularizer/mul�
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_1_5542061*
_output_shapes

:@T*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOp�
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:@T2 
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

Identity�
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall.^dense_1/kernel/Regularizer/Abs/ReadVariableOp(^stream_0_conv_1/StatefulPartitionedCall6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp(^stream_0_conv_2/StatefulPartitionedCall9^stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:����������: : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2^
-dense_1/kernel/Regularizer/Abs/ReadVariableOp-dense_1/kernel/Regularizer/Abs/ReadVariableOp2R
'stream_0_conv_1/StatefulPartitionedCall'stream_0_conv_1/StatefulPartitionedCall2n
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp2R
'stream_0_conv_2/StatefulPartitionedCall'stream_0_conv_2/StatefulPartitionedCall2t
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp:V R
,
_output_shapes
:����������
"
_user_specified_name
inputs_0
�
�
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_5541413

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
:����������@2
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
:����������@2
batchnorm/add_1s
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*,
_output_shapes
:����������@2

Identity�
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������@: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:T P
,
_output_shapes
:����������@
 
_user_specified_nameinputs
��
�
F__inference_basemodel_layer_call_and_return_conditional_losses_5544036

inputsQ
;stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource: =
/stream_0_conv_1_biasadd_readvariableop_resource: I
;batch_normalization_assignmovingavg_readvariableop_resource: K
=batch_normalization_assignmovingavg_1_readvariableop_resource: G
9batch_normalization_batchnorm_mul_readvariableop_resource: C
5batch_normalization_batchnorm_readvariableop_resource: Q
;stream_0_conv_2_conv1d_expanddims_1_readvariableop_resource: @=
/stream_0_conv_2_biasadd_readvariableop_resource:@K
=batch_normalization_1_assignmovingavg_readvariableop_resource:@M
?batch_normalization_1_assignmovingavg_1_readvariableop_resource:@I
;batch_normalization_1_batchnorm_mul_readvariableop_resource:@E
7batch_normalization_1_batchnorm_readvariableop_resource:@8
&dense_1_matmul_readvariableop_resource:@T5
'dense_1_biasadd_readvariableop_resource:TK
=batch_normalization_2_assignmovingavg_readvariableop_resource:TM
?batch_normalization_2_assignmovingavg_1_readvariableop_resource:TI
;batch_normalization_2_batchnorm_mul_readvariableop_resource:TE
7batch_normalization_2_batchnorm_readvariableop_resource:T
identity��#batch_normalization/AssignMovingAvg�2batch_normalization/AssignMovingAvg/ReadVariableOp�%batch_normalization/AssignMovingAvg_1�4batch_normalization/AssignMovingAvg_1/ReadVariableOp�,batch_normalization/batchnorm/ReadVariableOp�0batch_normalization/batchnorm/mul/ReadVariableOp�%batch_normalization_1/AssignMovingAvg�4batch_normalization_1/AssignMovingAvg/ReadVariableOp�'batch_normalization_1/AssignMovingAvg_1�6batch_normalization_1/AssignMovingAvg_1/ReadVariableOp�.batch_normalization_1/batchnorm/ReadVariableOp�2batch_normalization_1/batchnorm/mul/ReadVariableOp�%batch_normalization_2/AssignMovingAvg�4batch_normalization_2/AssignMovingAvg/ReadVariableOp�'batch_normalization_2/AssignMovingAvg_1�6batch_normalization_2/AssignMovingAvg_1/ReadVariableOp�.batch_normalization_2/batchnorm/ReadVariableOp�2batch_normalization_2/batchnorm/mul/ReadVariableOp�dense_1/BiasAdd/ReadVariableOp�dense_1/MatMul/ReadVariableOp�-dense_1/kernel/Regularizer/Abs/ReadVariableOp�&stream_0_conv_1/BiasAdd/ReadVariableOp�2stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp�5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp�&stream_0_conv_2/BiasAdd/ReadVariableOp�2stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOp�8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp�
!stream_0_input_drop/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2#
!stream_0_input_drop/dropout/Const�
stream_0_input_drop/dropout/MulMulinputs*stream_0_input_drop/dropout/Const:output:0*
T0*,
_output_shapes
:����������2!
stream_0_input_drop/dropout/Mul|
!stream_0_input_drop/dropout/ShapeShapeinputs*
T0*
_output_shapes
:2#
!stream_0_input_drop/dropout/Shape�
8stream_0_input_drop/dropout/random_uniform/RandomUniformRandomUniform*stream_0_input_drop/dropout/Shape:output:0*
T0*,
_output_shapes
:����������*
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
:����������2*
(stream_0_input_drop/dropout/GreaterEqual�
 stream_0_input_drop/dropout/CastCast,stream_0_input_drop/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:����������2"
 stream_0_input_drop/dropout/Cast�
!stream_0_input_drop/dropout/Mul_1Mul#stream_0_input_drop/dropout/Mul:z:0$stream_0_input_drop/dropout/Cast:y:0*
T0*,
_output_shapes
:����������2#
!stream_0_input_drop/dropout/Mul_1�
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
:����������2#
!stream_0_conv_1/conv1d/ExpandDims�
2stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp;stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
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
: 2%
#stream_0_conv_1/conv1d/ExpandDims_1�
stream_0_conv_1/conv1dConv2D*stream_0_conv_1/conv1d/ExpandDims:output:0,stream_0_conv_1/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:���������� *
paddingSAME*
strides
2
stream_0_conv_1/conv1d�
stream_0_conv_1/conv1d/SqueezeSqueezestream_0_conv_1/conv1d:output:0*
T0*,
_output_shapes
:���������� *
squeeze_dims

���������2 
stream_0_conv_1/conv1d/Squeeze�
&stream_0_conv_1/BiasAdd/ReadVariableOpReadVariableOp/stream_0_conv_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02(
&stream_0_conv_1/BiasAdd/ReadVariableOp�
stream_0_conv_1/BiasAddBiasAdd'stream_0_conv_1/conv1d/Squeeze:output:0.stream_0_conv_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������� 2
stream_0_conv_1/BiasAdd�
2batch_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       24
2batch_normalization/moments/mean/reduction_indices�
 batch_normalization/moments/meanMean stream_0_conv_1/BiasAdd:output:0;batch_normalization/moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(2"
 batch_normalization/moments/mean�
(batch_normalization/moments/StopGradientStopGradient)batch_normalization/moments/mean:output:0*
T0*"
_output_shapes
: 2*
(batch_normalization/moments/StopGradient�
-batch_normalization/moments/SquaredDifferenceSquaredDifference stream_0_conv_1/BiasAdd:output:01batch_normalization/moments/StopGradient:output:0*
T0*,
_output_shapes
:���������� 2/
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
: *
	keep_dims(2&
$batch_normalization/moments/variance�
#batch_normalization/moments/SqueezeSqueeze)batch_normalization/moments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2%
#batch_normalization/moments/Squeeze�
%batch_normalization/moments/Squeeze_1Squeeze-batch_normalization/moments/variance:output:0*
T0*
_output_shapes
: *
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
: *
dtype024
2batch_normalization/AssignMovingAvg/ReadVariableOp�
'batch_normalization/AssignMovingAvg/subSub:batch_normalization/AssignMovingAvg/ReadVariableOp:value:0,batch_normalization/moments/Squeeze:output:0*
T0*
_output_shapes
: 2)
'batch_normalization/AssignMovingAvg/sub�
'batch_normalization/AssignMovingAvg/mulMul+batch_normalization/AssignMovingAvg/sub:z:02batch_normalization/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
: 2)
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
: *
dtype026
4batch_normalization/AssignMovingAvg_1/ReadVariableOp�
)batch_normalization/AssignMovingAvg_1/subSub<batch_normalization/AssignMovingAvg_1/ReadVariableOp:value:0.batch_normalization/moments/Squeeze_1:output:0*
T0*
_output_shapes
: 2+
)batch_normalization/AssignMovingAvg_1/sub�
)batch_normalization/AssignMovingAvg_1/mulMul-batch_normalization/AssignMovingAvg_1/sub:z:04batch_normalization/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
: 2+
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
: 2#
!batch_normalization/batchnorm/add�
#batch_normalization/batchnorm/RsqrtRsqrt%batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes
: 2%
#batch_normalization/batchnorm/Rsqrt�
0batch_normalization/batchnorm/mul/ReadVariableOpReadVariableOp9batch_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype022
0batch_normalization/batchnorm/mul/ReadVariableOp�
!batch_normalization/batchnorm/mulMul'batch_normalization/batchnorm/Rsqrt:y:08batch_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2#
!batch_normalization/batchnorm/mul�
#batch_normalization/batchnorm/mul_1Mul stream_0_conv_1/BiasAdd:output:0%batch_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:���������� 2%
#batch_normalization/batchnorm/mul_1�
#batch_normalization/batchnorm/mul_2Mul,batch_normalization/moments/Squeeze:output:0%batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes
: 2%
#batch_normalization/batchnorm/mul_2�
,batch_normalization/batchnorm/ReadVariableOpReadVariableOp5batch_normalization_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02.
,batch_normalization/batchnorm/ReadVariableOp�
!batch_normalization/batchnorm/subSub4batch_normalization/batchnorm/ReadVariableOp:value:0'batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2#
!batch_normalization/batchnorm/sub�
#batch_normalization/batchnorm/add_1AddV2'batch_normalization/batchnorm/mul_1:z:0%batch_normalization/batchnorm/sub:z:0*
T0*,
_output_shapes
:���������� 2%
#batch_normalization/batchnorm/add_1�
activation/ReluRelu'batch_normalization/batchnorm/add_1:z:0*
T0*,
_output_shapes
:���������� 2
activation/Relu�
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
:���������� 2
stream_0_drop_1/dropout/Mul�
stream_0_drop_1/dropout/ShapeShapeactivation/Relu:activations:0*
T0*
_output_shapes
:2
stream_0_drop_1/dropout/Shape�
4stream_0_drop_1/dropout/random_uniform/RandomUniformRandomUniform&stream_0_drop_1/dropout/Shape:output:0*
T0*,
_output_shapes
:���������� *
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
:���������� 2&
$stream_0_drop_1/dropout/GreaterEqual�
stream_0_drop_1/dropout/CastCast(stream_0_drop_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:���������� 2
stream_0_drop_1/dropout/Cast�
stream_0_drop_1/dropout/Mul_1Mulstream_0_drop_1/dropout/Mul:z:0 stream_0_drop_1/dropout/Cast:y:0*
T0*,
_output_shapes
:���������� 2
stream_0_drop_1/dropout/Mul_1�
%stream_0_conv_2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2'
%stream_0_conv_2/conv1d/ExpandDims/dim�
!stream_0_conv_2/conv1d/ExpandDims
ExpandDims!stream_0_drop_1/dropout/Mul_1:z:0.stream_0_conv_2/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:���������� 2#
!stream_0_conv_2/conv1d/ExpandDims�
2stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp;stream_0_conv_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: @*
dtype024
2stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOp�
'stream_0_conv_2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2)
'stream_0_conv_2/conv1d/ExpandDims_1/dim�
#stream_0_conv_2/conv1d/ExpandDims_1
ExpandDims:stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOp:value:00stream_0_conv_2/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: @2%
#stream_0_conv_2/conv1d/ExpandDims_1�
stream_0_conv_2/conv1dConv2D*stream_0_conv_2/conv1d/ExpandDims:output:0,stream_0_conv_2/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:����������@*
paddingSAME*
strides
2
stream_0_conv_2/conv1d�
stream_0_conv_2/conv1d/SqueezeSqueezestream_0_conv_2/conv1d:output:0*
T0*,
_output_shapes
:����������@*
squeeze_dims

���������2 
stream_0_conv_2/conv1d/Squeeze�
&stream_0_conv_2/BiasAdd/ReadVariableOpReadVariableOp/stream_0_conv_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02(
&stream_0_conv_2/BiasAdd/ReadVariableOp�
stream_0_conv_2/BiasAddBiasAdd'stream_0_conv_2/conv1d/Squeeze:output:0.stream_0_conv_2/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������@2
stream_0_conv_2/BiasAdd�
4batch_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       26
4batch_normalization_1/moments/mean/reduction_indices�
"batch_normalization_1/moments/meanMean stream_0_conv_2/BiasAdd:output:0=batch_normalization_1/moments/mean/reduction_indices:output:0*
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
/batch_normalization_1/moments/SquaredDifferenceSquaredDifference stream_0_conv_2/BiasAdd:output:03batch_normalization_1/moments/StopGradient:output:0*
T0*,
_output_shapes
:����������@21
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
%batch_normalization_1/batchnorm/mul_1Mul stream_0_conv_2/BiasAdd:output:0'batch_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:����������@2'
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
:����������@2'
%batch_normalization_1/batchnorm/add_1�
activation_1/ReluRelu)batch_normalization_1/batchnorm/add_1:z:0*
T0*,
_output_shapes
:����������@2
activation_1/Relu�
stream_0_drop_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2
stream_0_drop_2/dropout/Const�
stream_0_drop_2/dropout/MulMulactivation_1/Relu:activations:0&stream_0_drop_2/dropout/Const:output:0*
T0*,
_output_shapes
:����������@2
stream_0_drop_2/dropout/Mul�
stream_0_drop_2/dropout/ShapeShapeactivation_1/Relu:activations:0*
T0*
_output_shapes
:2
stream_0_drop_2/dropout/Shape�
4stream_0_drop_2/dropout/random_uniform/RandomUniformRandomUniform&stream_0_drop_2/dropout/Shape:output:0*
T0*,
_output_shapes
:����������@*
dtype0*
seed�*
seed2�26
4stream_0_drop_2/dropout/random_uniform/RandomUniform�
&stream_0_drop_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���>2(
&stream_0_drop_2/dropout/GreaterEqual/y�
$stream_0_drop_2/dropout/GreaterEqualGreaterEqual=stream_0_drop_2/dropout/random_uniform/RandomUniform:output:0/stream_0_drop_2/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:����������@2&
$stream_0_drop_2/dropout/GreaterEqual�
stream_0_drop_2/dropout/CastCast(stream_0_drop_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:����������@2
stream_0_drop_2/dropout/Cast�
stream_0_drop_2/dropout/Mul_1Mulstream_0_drop_2/dropout/Mul:z:0 stream_0_drop_2/dropout/Cast:y:0*
T0*,
_output_shapes
:����������@2
stream_0_drop_2/dropout/Mul_1�
/global_average_pooling1d/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :21
/global_average_pooling1d/Mean/reduction_indices�
global_average_pooling1d/MeanMean!stream_0_drop_2/dropout/Mul_1:z:08global_average_pooling1d/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:���������@2
global_average_pooling1d/Mean�
concatenate/concat/concat_dimConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate/concat/concat_dim�
concatenate/concat/concatIdentity&global_average_pooling1d/Mean:output:0*
T0*'
_output_shapes
:���������@2
concatenate/concat/concat�
dense_1_dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
dense_1_dropout/dropout/Const�
dense_1_dropout/dropout/MulMul"concatenate/concat/concat:output:0&dense_1_dropout/dropout/Const:output:0*
T0*'
_output_shapes
:���������@2
dense_1_dropout/dropout/Mul�
dense_1_dropout/dropout/ShapeShape"concatenate/concat/concat:output:0*
T0*
_output_shapes
:2
dense_1_dropout/dropout/Shape�
4dense_1_dropout/dropout/random_uniform/RandomUniformRandomUniform&dense_1_dropout/dropout/Shape:output:0*
T0*'
_output_shapes
:���������@*
dtype0*
seed�26
4dense_1_dropout/dropout/random_uniform/RandomUniform�
&dense_1_dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>2(
&dense_1_dropout/dropout/GreaterEqual/y�
$dense_1_dropout/dropout/GreaterEqualGreaterEqual=dense_1_dropout/dropout/random_uniform/RandomUniform:output:0/dense_1_dropout/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:���������@2&
$dense_1_dropout/dropout/GreaterEqual�
dense_1_dropout/dropout/CastCast(dense_1_dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:���������@2
dense_1_dropout/dropout/Cast�
dense_1_dropout/dropout/Mul_1Muldense_1_dropout/dropout/Mul:z:0 dense_1_dropout/dropout/Cast:y:0*
T0*'
_output_shapes
:���������@2
dense_1_dropout/dropout/Mul_1�
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:@T*
dtype02
dense_1/MatMul/ReadVariableOp�
dense_1/MatMulMatMul!dense_1_dropout/dropout/Mul_1:z:0%dense_1/MatMul/ReadVariableOp:value:0*
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
4batch_normalization_2/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 26
4batch_normalization_2/moments/mean/reduction_indices�
"batch_normalization_2/moments/meanMeandense_1/BiasAdd:output:0=batch_normalization_2/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:T*
	keep_dims(2$
"batch_normalization_2/moments/mean�
*batch_normalization_2/moments/StopGradientStopGradient+batch_normalization_2/moments/mean:output:0*
T0*
_output_shapes

:T2,
*batch_normalization_2/moments/StopGradient�
/batch_normalization_2/moments/SquaredDifferenceSquaredDifferencedense_1/BiasAdd:output:03batch_normalization_2/moments/StopGradient:output:0*
T0*'
_output_shapes
:���������T21
/batch_normalization_2/moments/SquaredDifference�
8batch_normalization_2/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2:
8batch_normalization_2/moments/variance/reduction_indices�
&batch_normalization_2/moments/varianceMean3batch_normalization_2/moments/SquaredDifference:z:0Abatch_normalization_2/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:T*
	keep_dims(2(
&batch_normalization_2/moments/variance�
%batch_normalization_2/moments/SqueezeSqueeze+batch_normalization_2/moments/mean:output:0*
T0*
_output_shapes
:T*
squeeze_dims
 2'
%batch_normalization_2/moments/Squeeze�
'batch_normalization_2/moments/Squeeze_1Squeeze/batch_normalization_2/moments/variance:output:0*
T0*
_output_shapes
:T*
squeeze_dims
 2)
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
:T*
dtype026
4batch_normalization_2/AssignMovingAvg/ReadVariableOp�
)batch_normalization_2/AssignMovingAvg/subSub<batch_normalization_2/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_2/moments/Squeeze:output:0*
T0*
_output_shapes
:T2+
)batch_normalization_2/AssignMovingAvg/sub�
)batch_normalization_2/AssignMovingAvg/mulMul-batch_normalization_2/AssignMovingAvg/sub:z:04batch_normalization_2/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:T2+
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
:T*
dtype028
6batch_normalization_2/AssignMovingAvg_1/ReadVariableOp�
+batch_normalization_2/AssignMovingAvg_1/subSub>batch_normalization_2/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_2/moments/Squeeze_1:output:0*
T0*
_output_shapes
:T2-
+batch_normalization_2/AssignMovingAvg_1/sub�
+batch_normalization_2/AssignMovingAvg_1/mulMul/batch_normalization_2/AssignMovingAvg_1/sub:z:06batch_normalization_2/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:T2-
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
:T2%
#batch_normalization_2/batchnorm/add�
%batch_normalization_2/batchnorm/RsqrtRsqrt'batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes
:T2'
%batch_normalization_2/batchnorm/Rsqrt�
2batch_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes
:T*
dtype024
2batch_normalization_2/batchnorm/mul/ReadVariableOp�
#batch_normalization_2/batchnorm/mulMul)batch_normalization_2/batchnorm/Rsqrt:y:0:batch_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:T2%
#batch_normalization_2/batchnorm/mul�
%batch_normalization_2/batchnorm/mul_1Muldense_1/BiasAdd:output:0'batch_normalization_2/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������T2'
%batch_normalization_2/batchnorm/mul_1�
%batch_normalization_2/batchnorm/mul_2Mul.batch_normalization_2/moments/Squeeze:output:0'batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes
:T2'
%batch_normalization_2/batchnorm/mul_2�
.batch_normalization_2/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_2_batchnorm_readvariableop_resource*
_output_shapes
:T*
dtype020
.batch_normalization_2/batchnorm/ReadVariableOp�
#batch_normalization_2/batchnorm/subSub6batch_normalization_2/batchnorm/ReadVariableOp:value:0)batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes
:T2%
#batch_normalization_2/batchnorm/sub�
%batch_normalization_2/batchnorm/add_1AddV2)batch_normalization_2/batchnorm/mul_1:z:0'batch_normalization_2/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������T2'
%batch_normalization_2/batchnorm/add_1�
dense_activation_1/SigmoidSigmoid)batch_normalization_2/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������T2
dense_activation_1/Sigmoid�
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp;stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype027
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp�
&stream_0_conv_1/kernel/Regularizer/AbsAbs=stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
: 2(
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
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;stream_0_conv_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: @*
dtype02:
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp�
)stream_0_conv_2/kernel/Regularizer/SquareSquare@stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
: @2+
)stream_0_conv_2/kernel/Regularizer/Square�
(stream_0_conv_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_0_conv_2/kernel/Regularizer/Const�
&stream_0_conv_2/kernel/Regularizer/SumSum-stream_0_conv_2/kernel/Regularizer/Square:y:01stream_0_conv_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_2/kernel/Regularizer/Sum�
(stream_0_conv_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2*
(stream_0_conv_2/kernel/Regularizer/mul/x�
&stream_0_conv_2/kernel/Regularizer/mulMul1stream_0_conv_2/kernel/Regularizer/mul/x:output:0/stream_0_conv_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_2/kernel/Regularizer/mul�
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:@T*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOp�
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:@T2 
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

NoOpNoOp$^batch_normalization/AssignMovingAvg3^batch_normalization/AssignMovingAvg/ReadVariableOp&^batch_normalization/AssignMovingAvg_15^batch_normalization/AssignMovingAvg_1/ReadVariableOp-^batch_normalization/batchnorm/ReadVariableOp1^batch_normalization/batchnorm/mul/ReadVariableOp&^batch_normalization_1/AssignMovingAvg5^batch_normalization_1/AssignMovingAvg/ReadVariableOp(^batch_normalization_1/AssignMovingAvg_17^batch_normalization_1/AssignMovingAvg_1/ReadVariableOp/^batch_normalization_1/batchnorm/ReadVariableOp3^batch_normalization_1/batchnorm/mul/ReadVariableOp&^batch_normalization_2/AssignMovingAvg5^batch_normalization_2/AssignMovingAvg/ReadVariableOp(^batch_normalization_2/AssignMovingAvg_17^batch_normalization_2/AssignMovingAvg_1/ReadVariableOp/^batch_normalization_2/batchnorm/ReadVariableOp3^batch_normalization_2/batchnorm/mul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp.^dense_1/kernel/Regularizer/Abs/ReadVariableOp'^stream_0_conv_1/BiasAdd/ReadVariableOp3^stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp'^stream_0_conv_2/BiasAdd/ReadVariableOp3^stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOp9^stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:����������: : : : : : : : : : : : : : : : : : 2J
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
2batch_normalization_2/batchnorm/mul/ReadVariableOp2batch_normalization_2/batchnorm/mul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2^
-dense_1/kernel/Regularizer/Abs/ReadVariableOp-dense_1/kernel/Regularizer/Abs/ReadVariableOp2P
&stream_0_conv_1/BiasAdd/ReadVariableOp&stream_0_conv_1/BiasAdd/ReadVariableOp2h
2stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp2stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp2n
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp2P
&stream_0_conv_2/BiasAdd/ReadVariableOp&stream_0_conv_2/BiasAdd/ReadVariableOp2h
2stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOp2stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOp2t
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�+
�
P__inference_batch_normalization_layer_call_and_return_conditional_losses_5541782

inputs5
'assignmovingavg_readvariableop_resource: 7
)assignmovingavg_1_readvariableop_resource: 3
%batchnorm_mul_readvariableop_resource: /
!batchnorm_readvariableop_resource: 
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
: *
	keep_dims(2
moments/mean�
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
: 2
moments/StopGradient�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*,
_output_shapes
:���������� 2
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
: *
	keep_dims(2
moments/variance�
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/Squeeze�
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
: *
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
: *
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
: 2
AssignMovingAvg/sub�
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
: 2
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
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
: 2
AssignMovingAvg_1/sub�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
: 2
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
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/Rsqrt�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/mul/ReadVariableOp�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/mul{
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*,
_output_shapes
:���������� 2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2�
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp�
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*,
_output_shapes
:���������� 2
batchnorm/add_1s
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*,
_output_shapes
:���������� 2

Identity�
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������� : : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:T P
,
_output_shapes
:���������� 
 
_user_specified_nameinputs
�	
�
7__inference_batch_normalization_1_layer_call_fn_5544954

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
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_55410192
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
�	
�
5__inference_batch_normalization_layer_call_fn_5544721

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_55408572
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :������������������ 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:������������������ : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :������������������ 
 
_user_specified_nameinputs
�+
�
P__inference_batch_normalization_layer_call_and_return_conditional_losses_5540857

inputs5
'assignmovingavg_readvariableop_resource: 7
)assignmovingavg_1_readvariableop_resource: 3
%batchnorm_mul_readvariableop_resource: /
!batchnorm_readvariableop_resource: 
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
: *
	keep_dims(2
moments/mean�
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
: 2
moments/StopGradient�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :������������������ 2
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
: *
	keep_dims(2
moments/variance�
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/Squeeze�
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
: *
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
: *
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
: 2
AssignMovingAvg/sub�
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
: 2
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
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
: 2
AssignMovingAvg_1/sub�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
: 2
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
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/Rsqrt�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/mul/ReadVariableOp�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/mul�
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :������������������ 2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2�
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp�
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :������������������ 2
batchnorm/add_1{
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :������������������ 2

Identity�
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:������������������ : : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :������������������ 
 
_user_specified_nameinputs
�
n
P__inference_stream_0_input_drop_layer_call_and_return_conditional_losses_5541295

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:����������2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:����������2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
q
U__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_5545029

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
:����������@:T P
,
_output_shapes
:����������@
 
_user_specified_nameinputs
�+
�
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_5541683

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
:����������@2
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
:����������@2
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
:����������@2
batchnorm/add_1s
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*,
_output_shapes
:����������@2

Identity�
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������@: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:T P
,
_output_shapes
:����������@
 
_user_specified_nameinputs
�
k
L__inference_stream_0_drop_2_layer_call_and_return_conditional_losses_5541625

inputs
identity�c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2
dropout/Constx
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:����������@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:����������@*
dtype0*
seed�*
seed2�2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���>2
dropout/GreaterEqual/y�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:����������@2
dropout/GreaterEqual�
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:����������@2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:����������@2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:����������@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������@:T P
,
_output_shapes
:����������@
 
_user_specified_nameinputs
�	
q
E__inference_distance_layer_call_and_return_conditional_losses_5544512
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
�
�
D__inference_dense_1_layer_call_and_return_conditional_losses_5541475

inputs0
matmul_readvariableop_resource:@T-
biasadd_readvariableop_resource:T
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�-dense_1/kernel/Regularizer/Abs/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@T*
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

:@T*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOp�
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:@T2 
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
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2^
-dense_1/kernel/Regularizer/Abs/ReadVariableOp-dense_1/kernel/Regularizer/Abs/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�	
�
7__inference_batch_normalization_1_layer_call_fn_5544941

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
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_55409592
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
�*
�
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_5541205

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
�
c
G__inference_activation_layer_call_and_return_conditional_losses_5541358

inputs
identityS
ReluReluinputs*
T0*,
_output_shapes
:���������� 2
Reluk
IdentityIdentityRelu:activations:0*
T0*,
_output_shapes
:���������� 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:���������� :T P
,
_output_shapes
:���������� 
 
_user_specified_nameinputs
�
�
+__inference_basemodel_layer_call_fn_5544365

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: 
	unknown_5: @
	unknown_6:@
	unknown_7:@
	unknown_8:@
	unknown_9:@

unknown_10:@

unknown_11:@T

unknown_12:T

unknown_13:T

unknown_14:T

unknown_15:T

unknown_16:T
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������T*4
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_basemodel_layer_call_and_return_conditional_losses_55415162
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
_construction_contextkEagerRuntime*O
_input_shapes>
<:����������: : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�	
o
E__inference_distance_layer_call_and_return_conditional_losses_5542354

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
�
L__inference_stream_0_conv_1_layer_call_and_return_conditional_losses_5544578

inputsA
+conv1d_expanddims_1_readvariableop_resource: -
biasadd_readvariableop_resource: 
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
:����������2
conv1d/ExpandDims�
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
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
: 2
conv1d/ExpandDims_1�
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:���������� *
paddingSAME*
strides
2
conv1d�
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:���������� *
squeeze_dims

���������2
conv1d/Squeeze�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������� 2	
BiasAdd�
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype027
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp�
&stream_0_conv_1/kernel/Regularizer/AbsAbs=stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
: 2(
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
:���������� 2

Identity�
NoOpNoOp^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp2n
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
1__inference_stream_0_conv_2_layer_call_fn_5544820

inputs
unknown: @
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
:����������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_stream_0_conv_2_layer_call_and_return_conditional_losses_55413882
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:����������@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:���������� : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:���������� 
 
_user_specified_nameinputs
�
o
P__inference_stream_0_input_drop_layer_call_and_return_conditional_losses_5541823

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
:����������2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:����������*
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
:����������2
dropout/GreaterEqual�
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:����������2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:����������2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
M
1__inference_stream_0_drop_2_layer_call_fn_5545012

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
:����������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_stream_0_drop_2_layer_call_and_return_conditional_losses_55414352
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:����������@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������@:T P
,
_output_shapes
:����������@
 
_user_specified_nameinputs
�
�
'__inference_model_layer_call_fn_5542906
left_inputs
right_inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: 
	unknown_5: @
	unknown_6:@
	unknown_7:@
	unknown_8:@
	unknown_9:@

unknown_10:@

unknown_11:@T

unknown_12:T

unknown_13:T

unknown_14:T

unknown_15:T

unknown_16:T
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallleft_inputsright_inputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*.
_read_only_resource_inputs
	*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_model_layer_call_and_return_conditional_losses_55428252
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
_construction_contextkEagerRuntime*g
_input_shapesV
T:����������:����������: : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
,
_output_shapes
:����������
%
_user_specified_nameleft_inputs:ZV
,
_output_shapes
:����������
&
_user_specified_nameright_inputs
��
�
F__inference_basemodel_layer_call_and_return_conditional_losses_5542661

inputsQ
;stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource: =
/stream_0_conv_1_biasadd_readvariableop_resource: I
;batch_normalization_assignmovingavg_readvariableop_resource: K
=batch_normalization_assignmovingavg_1_readvariableop_resource: G
9batch_normalization_batchnorm_mul_readvariableop_resource: C
5batch_normalization_batchnorm_readvariableop_resource: Q
;stream_0_conv_2_conv1d_expanddims_1_readvariableop_resource: @=
/stream_0_conv_2_biasadd_readvariableop_resource:@K
=batch_normalization_1_assignmovingavg_readvariableop_resource:@M
?batch_normalization_1_assignmovingavg_1_readvariableop_resource:@I
;batch_normalization_1_batchnorm_mul_readvariableop_resource:@E
7batch_normalization_1_batchnorm_readvariableop_resource:@8
&dense_1_matmul_readvariableop_resource:@T5
'dense_1_biasadd_readvariableop_resource:TK
=batch_normalization_2_assignmovingavg_readvariableop_resource:TM
?batch_normalization_2_assignmovingavg_1_readvariableop_resource:TI
;batch_normalization_2_batchnorm_mul_readvariableop_resource:TE
7batch_normalization_2_batchnorm_readvariableop_resource:T
identity��#batch_normalization/AssignMovingAvg�2batch_normalization/AssignMovingAvg/ReadVariableOp�%batch_normalization/AssignMovingAvg_1�4batch_normalization/AssignMovingAvg_1/ReadVariableOp�,batch_normalization/batchnorm/ReadVariableOp�0batch_normalization/batchnorm/mul/ReadVariableOp�%batch_normalization_1/AssignMovingAvg�4batch_normalization_1/AssignMovingAvg/ReadVariableOp�'batch_normalization_1/AssignMovingAvg_1�6batch_normalization_1/AssignMovingAvg_1/ReadVariableOp�.batch_normalization_1/batchnorm/ReadVariableOp�2batch_normalization_1/batchnorm/mul/ReadVariableOp�%batch_normalization_2/AssignMovingAvg�4batch_normalization_2/AssignMovingAvg/ReadVariableOp�'batch_normalization_2/AssignMovingAvg_1�6batch_normalization_2/AssignMovingAvg_1/ReadVariableOp�.batch_normalization_2/batchnorm/ReadVariableOp�2batch_normalization_2/batchnorm/mul/ReadVariableOp�dense_1/BiasAdd/ReadVariableOp�dense_1/MatMul/ReadVariableOp�-dense_1/kernel/Regularizer/Abs/ReadVariableOp�&stream_0_conv_1/BiasAdd/ReadVariableOp�2stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp�5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp�&stream_0_conv_2/BiasAdd/ReadVariableOp�2stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOp�8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp�
!stream_0_input_drop/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2#
!stream_0_input_drop/dropout/Const�
stream_0_input_drop/dropout/MulMulinputs*stream_0_input_drop/dropout/Const:output:0*
T0*,
_output_shapes
:����������2!
stream_0_input_drop/dropout/Mul|
!stream_0_input_drop/dropout/ShapeShapeinputs*
T0*
_output_shapes
:2#
!stream_0_input_drop/dropout/Shape�
8stream_0_input_drop/dropout/random_uniform/RandomUniformRandomUniform*stream_0_input_drop/dropout/Shape:output:0*
T0*,
_output_shapes
:����������*
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
:����������2*
(stream_0_input_drop/dropout/GreaterEqual�
 stream_0_input_drop/dropout/CastCast,stream_0_input_drop/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:����������2"
 stream_0_input_drop/dropout/Cast�
!stream_0_input_drop/dropout/Mul_1Mul#stream_0_input_drop/dropout/Mul:z:0$stream_0_input_drop/dropout/Cast:y:0*
T0*,
_output_shapes
:����������2#
!stream_0_input_drop/dropout/Mul_1�
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
:����������2#
!stream_0_conv_1/conv1d/ExpandDims�
2stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp;stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
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
: 2%
#stream_0_conv_1/conv1d/ExpandDims_1�
stream_0_conv_1/conv1dConv2D*stream_0_conv_1/conv1d/ExpandDims:output:0,stream_0_conv_1/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:���������� *
paddingSAME*
strides
2
stream_0_conv_1/conv1d�
stream_0_conv_1/conv1d/SqueezeSqueezestream_0_conv_1/conv1d:output:0*
T0*,
_output_shapes
:���������� *
squeeze_dims

���������2 
stream_0_conv_1/conv1d/Squeeze�
&stream_0_conv_1/BiasAdd/ReadVariableOpReadVariableOp/stream_0_conv_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02(
&stream_0_conv_1/BiasAdd/ReadVariableOp�
stream_0_conv_1/BiasAddBiasAdd'stream_0_conv_1/conv1d/Squeeze:output:0.stream_0_conv_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������� 2
stream_0_conv_1/BiasAdd�
2batch_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       24
2batch_normalization/moments/mean/reduction_indices�
 batch_normalization/moments/meanMean stream_0_conv_1/BiasAdd:output:0;batch_normalization/moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(2"
 batch_normalization/moments/mean�
(batch_normalization/moments/StopGradientStopGradient)batch_normalization/moments/mean:output:0*
T0*"
_output_shapes
: 2*
(batch_normalization/moments/StopGradient�
-batch_normalization/moments/SquaredDifferenceSquaredDifference stream_0_conv_1/BiasAdd:output:01batch_normalization/moments/StopGradient:output:0*
T0*,
_output_shapes
:���������� 2/
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
: *
	keep_dims(2&
$batch_normalization/moments/variance�
#batch_normalization/moments/SqueezeSqueeze)batch_normalization/moments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2%
#batch_normalization/moments/Squeeze�
%batch_normalization/moments/Squeeze_1Squeeze-batch_normalization/moments/variance:output:0*
T0*
_output_shapes
: *
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
: *
dtype024
2batch_normalization/AssignMovingAvg/ReadVariableOp�
'batch_normalization/AssignMovingAvg/subSub:batch_normalization/AssignMovingAvg/ReadVariableOp:value:0,batch_normalization/moments/Squeeze:output:0*
T0*
_output_shapes
: 2)
'batch_normalization/AssignMovingAvg/sub�
'batch_normalization/AssignMovingAvg/mulMul+batch_normalization/AssignMovingAvg/sub:z:02batch_normalization/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
: 2)
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
: *
dtype026
4batch_normalization/AssignMovingAvg_1/ReadVariableOp�
)batch_normalization/AssignMovingAvg_1/subSub<batch_normalization/AssignMovingAvg_1/ReadVariableOp:value:0.batch_normalization/moments/Squeeze_1:output:0*
T0*
_output_shapes
: 2+
)batch_normalization/AssignMovingAvg_1/sub�
)batch_normalization/AssignMovingAvg_1/mulMul-batch_normalization/AssignMovingAvg_1/sub:z:04batch_normalization/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
: 2+
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
: 2#
!batch_normalization/batchnorm/add�
#batch_normalization/batchnorm/RsqrtRsqrt%batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes
: 2%
#batch_normalization/batchnorm/Rsqrt�
0batch_normalization/batchnorm/mul/ReadVariableOpReadVariableOp9batch_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype022
0batch_normalization/batchnorm/mul/ReadVariableOp�
!batch_normalization/batchnorm/mulMul'batch_normalization/batchnorm/Rsqrt:y:08batch_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2#
!batch_normalization/batchnorm/mul�
#batch_normalization/batchnorm/mul_1Mul stream_0_conv_1/BiasAdd:output:0%batch_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:���������� 2%
#batch_normalization/batchnorm/mul_1�
#batch_normalization/batchnorm/mul_2Mul,batch_normalization/moments/Squeeze:output:0%batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes
: 2%
#batch_normalization/batchnorm/mul_2�
,batch_normalization/batchnorm/ReadVariableOpReadVariableOp5batch_normalization_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02.
,batch_normalization/batchnorm/ReadVariableOp�
!batch_normalization/batchnorm/subSub4batch_normalization/batchnorm/ReadVariableOp:value:0'batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2#
!batch_normalization/batchnorm/sub�
#batch_normalization/batchnorm/add_1AddV2'batch_normalization/batchnorm/mul_1:z:0%batch_normalization/batchnorm/sub:z:0*
T0*,
_output_shapes
:���������� 2%
#batch_normalization/batchnorm/add_1�
activation/ReluRelu'batch_normalization/batchnorm/add_1:z:0*
T0*,
_output_shapes
:���������� 2
activation/Relu�
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
:���������� 2
stream_0_drop_1/dropout/Mul�
stream_0_drop_1/dropout/ShapeShapeactivation/Relu:activations:0*
T0*
_output_shapes
:2
stream_0_drop_1/dropout/Shape�
4stream_0_drop_1/dropout/random_uniform/RandomUniformRandomUniform&stream_0_drop_1/dropout/Shape:output:0*
T0*,
_output_shapes
:���������� *
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
:���������� 2&
$stream_0_drop_1/dropout/GreaterEqual�
stream_0_drop_1/dropout/CastCast(stream_0_drop_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:���������� 2
stream_0_drop_1/dropout/Cast�
stream_0_drop_1/dropout/Mul_1Mulstream_0_drop_1/dropout/Mul:z:0 stream_0_drop_1/dropout/Cast:y:0*
T0*,
_output_shapes
:���������� 2
stream_0_drop_1/dropout/Mul_1�
%stream_0_conv_2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2'
%stream_0_conv_2/conv1d/ExpandDims/dim�
!stream_0_conv_2/conv1d/ExpandDims
ExpandDims!stream_0_drop_1/dropout/Mul_1:z:0.stream_0_conv_2/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:���������� 2#
!stream_0_conv_2/conv1d/ExpandDims�
2stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp;stream_0_conv_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: @*
dtype024
2stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOp�
'stream_0_conv_2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2)
'stream_0_conv_2/conv1d/ExpandDims_1/dim�
#stream_0_conv_2/conv1d/ExpandDims_1
ExpandDims:stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOp:value:00stream_0_conv_2/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: @2%
#stream_0_conv_2/conv1d/ExpandDims_1�
stream_0_conv_2/conv1dConv2D*stream_0_conv_2/conv1d/ExpandDims:output:0,stream_0_conv_2/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:����������@*
paddingSAME*
strides
2
stream_0_conv_2/conv1d�
stream_0_conv_2/conv1d/SqueezeSqueezestream_0_conv_2/conv1d:output:0*
T0*,
_output_shapes
:����������@*
squeeze_dims

���������2 
stream_0_conv_2/conv1d/Squeeze�
&stream_0_conv_2/BiasAdd/ReadVariableOpReadVariableOp/stream_0_conv_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02(
&stream_0_conv_2/BiasAdd/ReadVariableOp�
stream_0_conv_2/BiasAddBiasAdd'stream_0_conv_2/conv1d/Squeeze:output:0.stream_0_conv_2/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������@2
stream_0_conv_2/BiasAdd�
4batch_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       26
4batch_normalization_1/moments/mean/reduction_indices�
"batch_normalization_1/moments/meanMean stream_0_conv_2/BiasAdd:output:0=batch_normalization_1/moments/mean/reduction_indices:output:0*
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
/batch_normalization_1/moments/SquaredDifferenceSquaredDifference stream_0_conv_2/BiasAdd:output:03batch_normalization_1/moments/StopGradient:output:0*
T0*,
_output_shapes
:����������@21
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
%batch_normalization_1/batchnorm/mul_1Mul stream_0_conv_2/BiasAdd:output:0'batch_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:����������@2'
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
:����������@2'
%batch_normalization_1/batchnorm/add_1�
activation_1/ReluRelu)batch_normalization_1/batchnorm/add_1:z:0*
T0*,
_output_shapes
:����������@2
activation_1/Relu�
stream_0_drop_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2
stream_0_drop_2/dropout/Const�
stream_0_drop_2/dropout/MulMulactivation_1/Relu:activations:0&stream_0_drop_2/dropout/Const:output:0*
T0*,
_output_shapes
:����������@2
stream_0_drop_2/dropout/Mul�
stream_0_drop_2/dropout/ShapeShapeactivation_1/Relu:activations:0*
T0*
_output_shapes
:2
stream_0_drop_2/dropout/Shape�
4stream_0_drop_2/dropout/random_uniform/RandomUniformRandomUniform&stream_0_drop_2/dropout/Shape:output:0*
T0*,
_output_shapes
:����������@*
dtype0*
seed�*
seed2�26
4stream_0_drop_2/dropout/random_uniform/RandomUniform�
&stream_0_drop_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���>2(
&stream_0_drop_2/dropout/GreaterEqual/y�
$stream_0_drop_2/dropout/GreaterEqualGreaterEqual=stream_0_drop_2/dropout/random_uniform/RandomUniform:output:0/stream_0_drop_2/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:����������@2&
$stream_0_drop_2/dropout/GreaterEqual�
stream_0_drop_2/dropout/CastCast(stream_0_drop_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:����������@2
stream_0_drop_2/dropout/Cast�
stream_0_drop_2/dropout/Mul_1Mulstream_0_drop_2/dropout/Mul:z:0 stream_0_drop_2/dropout/Cast:y:0*
T0*,
_output_shapes
:����������@2
stream_0_drop_2/dropout/Mul_1�
/global_average_pooling1d/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :21
/global_average_pooling1d/Mean/reduction_indices�
global_average_pooling1d/MeanMean!stream_0_drop_2/dropout/Mul_1:z:08global_average_pooling1d/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:���������@2
global_average_pooling1d/Mean�
concatenate/concat/concat_dimConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate/concat/concat_dim�
concatenate/concat/concatIdentity&global_average_pooling1d/Mean:output:0*
T0*'
_output_shapes
:���������@2
concatenate/concat/concat�
dense_1_dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
dense_1_dropout/dropout/Const�
dense_1_dropout/dropout/MulMul"concatenate/concat/concat:output:0&dense_1_dropout/dropout/Const:output:0*
T0*'
_output_shapes
:���������@2
dense_1_dropout/dropout/Mul�
dense_1_dropout/dropout/ShapeShape"concatenate/concat/concat:output:0*
T0*
_output_shapes
:2
dense_1_dropout/dropout/Shape�
4dense_1_dropout/dropout/random_uniform/RandomUniformRandomUniform&dense_1_dropout/dropout/Shape:output:0*
T0*'
_output_shapes
:���������@*
dtype0*
seed�26
4dense_1_dropout/dropout/random_uniform/RandomUniform�
&dense_1_dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>2(
&dense_1_dropout/dropout/GreaterEqual/y�
$dense_1_dropout/dropout/GreaterEqualGreaterEqual=dense_1_dropout/dropout/random_uniform/RandomUniform:output:0/dense_1_dropout/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:���������@2&
$dense_1_dropout/dropout/GreaterEqual�
dense_1_dropout/dropout/CastCast(dense_1_dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:���������@2
dense_1_dropout/dropout/Cast�
dense_1_dropout/dropout/Mul_1Muldense_1_dropout/dropout/Mul:z:0 dense_1_dropout/dropout/Cast:y:0*
T0*'
_output_shapes
:���������@2
dense_1_dropout/dropout/Mul_1�
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:@T*
dtype02
dense_1/MatMul/ReadVariableOp�
dense_1/MatMulMatMul!dense_1_dropout/dropout/Mul_1:z:0%dense_1/MatMul/ReadVariableOp:value:0*
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
4batch_normalization_2/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 26
4batch_normalization_2/moments/mean/reduction_indices�
"batch_normalization_2/moments/meanMeandense_1/BiasAdd:output:0=batch_normalization_2/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:T*
	keep_dims(2$
"batch_normalization_2/moments/mean�
*batch_normalization_2/moments/StopGradientStopGradient+batch_normalization_2/moments/mean:output:0*
T0*
_output_shapes

:T2,
*batch_normalization_2/moments/StopGradient�
/batch_normalization_2/moments/SquaredDifferenceSquaredDifferencedense_1/BiasAdd:output:03batch_normalization_2/moments/StopGradient:output:0*
T0*'
_output_shapes
:���������T21
/batch_normalization_2/moments/SquaredDifference�
8batch_normalization_2/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2:
8batch_normalization_2/moments/variance/reduction_indices�
&batch_normalization_2/moments/varianceMean3batch_normalization_2/moments/SquaredDifference:z:0Abatch_normalization_2/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:T*
	keep_dims(2(
&batch_normalization_2/moments/variance�
%batch_normalization_2/moments/SqueezeSqueeze+batch_normalization_2/moments/mean:output:0*
T0*
_output_shapes
:T*
squeeze_dims
 2'
%batch_normalization_2/moments/Squeeze�
'batch_normalization_2/moments/Squeeze_1Squeeze/batch_normalization_2/moments/variance:output:0*
T0*
_output_shapes
:T*
squeeze_dims
 2)
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
:T*
dtype026
4batch_normalization_2/AssignMovingAvg/ReadVariableOp�
)batch_normalization_2/AssignMovingAvg/subSub<batch_normalization_2/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_2/moments/Squeeze:output:0*
T0*
_output_shapes
:T2+
)batch_normalization_2/AssignMovingAvg/sub�
)batch_normalization_2/AssignMovingAvg/mulMul-batch_normalization_2/AssignMovingAvg/sub:z:04batch_normalization_2/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:T2+
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
:T*
dtype028
6batch_normalization_2/AssignMovingAvg_1/ReadVariableOp�
+batch_normalization_2/AssignMovingAvg_1/subSub>batch_normalization_2/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_2/moments/Squeeze_1:output:0*
T0*
_output_shapes
:T2-
+batch_normalization_2/AssignMovingAvg_1/sub�
+batch_normalization_2/AssignMovingAvg_1/mulMul/batch_normalization_2/AssignMovingAvg_1/sub:z:06batch_normalization_2/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:T2-
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
:T2%
#batch_normalization_2/batchnorm/add�
%batch_normalization_2/batchnorm/RsqrtRsqrt'batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes
:T2'
%batch_normalization_2/batchnorm/Rsqrt�
2batch_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes
:T*
dtype024
2batch_normalization_2/batchnorm/mul/ReadVariableOp�
#batch_normalization_2/batchnorm/mulMul)batch_normalization_2/batchnorm/Rsqrt:y:0:batch_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:T2%
#batch_normalization_2/batchnorm/mul�
%batch_normalization_2/batchnorm/mul_1Muldense_1/BiasAdd:output:0'batch_normalization_2/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������T2'
%batch_normalization_2/batchnorm/mul_1�
%batch_normalization_2/batchnorm/mul_2Mul.batch_normalization_2/moments/Squeeze:output:0'batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes
:T2'
%batch_normalization_2/batchnorm/mul_2�
.batch_normalization_2/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_2_batchnorm_readvariableop_resource*
_output_shapes
:T*
dtype020
.batch_normalization_2/batchnorm/ReadVariableOp�
#batch_normalization_2/batchnorm/subSub6batch_normalization_2/batchnorm/ReadVariableOp:value:0)batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes
:T2%
#batch_normalization_2/batchnorm/sub�
%batch_normalization_2/batchnorm/add_1AddV2)batch_normalization_2/batchnorm/mul_1:z:0'batch_normalization_2/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������T2'
%batch_normalization_2/batchnorm/add_1�
dense_activation_1/SigmoidSigmoid)batch_normalization_2/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������T2
dense_activation_1/Sigmoid�
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp;stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype027
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp�
&stream_0_conv_1/kernel/Regularizer/AbsAbs=stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
: 2(
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
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;stream_0_conv_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: @*
dtype02:
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp�
)stream_0_conv_2/kernel/Regularizer/SquareSquare@stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
: @2+
)stream_0_conv_2/kernel/Regularizer/Square�
(stream_0_conv_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_0_conv_2/kernel/Regularizer/Const�
&stream_0_conv_2/kernel/Regularizer/SumSum-stream_0_conv_2/kernel/Regularizer/Square:y:01stream_0_conv_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_2/kernel/Regularizer/Sum�
(stream_0_conv_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2*
(stream_0_conv_2/kernel/Regularizer/mul/x�
&stream_0_conv_2/kernel/Regularizer/mulMul1stream_0_conv_2/kernel/Regularizer/mul/x:output:0/stream_0_conv_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_2/kernel/Regularizer/mul�
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:@T*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOp�
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:@T2 
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

NoOpNoOp$^batch_normalization/AssignMovingAvg3^batch_normalization/AssignMovingAvg/ReadVariableOp&^batch_normalization/AssignMovingAvg_15^batch_normalization/AssignMovingAvg_1/ReadVariableOp-^batch_normalization/batchnorm/ReadVariableOp1^batch_normalization/batchnorm/mul/ReadVariableOp&^batch_normalization_1/AssignMovingAvg5^batch_normalization_1/AssignMovingAvg/ReadVariableOp(^batch_normalization_1/AssignMovingAvg_17^batch_normalization_1/AssignMovingAvg_1/ReadVariableOp/^batch_normalization_1/batchnorm/ReadVariableOp3^batch_normalization_1/batchnorm/mul/ReadVariableOp&^batch_normalization_2/AssignMovingAvg5^batch_normalization_2/AssignMovingAvg/ReadVariableOp(^batch_normalization_2/AssignMovingAvg_17^batch_normalization_2/AssignMovingAvg_1/ReadVariableOp/^batch_normalization_2/batchnorm/ReadVariableOp3^batch_normalization_2/batchnorm/mul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp.^dense_1/kernel/Regularizer/Abs/ReadVariableOp'^stream_0_conv_1/BiasAdd/ReadVariableOp3^stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp'^stream_0_conv_2/BiasAdd/ReadVariableOp3^stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOp9^stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:����������: : : : : : : : : : : : : : : : : : 2J
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
2batch_normalization_2/batchnorm/mul/ReadVariableOp2batch_normalization_2/batchnorm/mul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2^
-dense_1/kernel/Regularizer/Abs/ReadVariableOp-dense_1/kernel/Regularizer/Abs/ReadVariableOp2P
&stream_0_conv_1/BiasAdd/ReadVariableOp&stream_0_conv_1/BiasAdd/ReadVariableOp2h
2stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp2stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp2n
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp2P
&stream_0_conv_2/BiasAdd/ReadVariableOp&stream_0_conv_2/BiasAdd/ReadVariableOp2h
2stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOp2stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOp2t
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
q
U__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_5545023

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
�
�
)__inference_dense_1_layer_call_fn_5545108

inputs
unknown:@T
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
D__inference_dense_1_layer_call_and_return_conditional_losses_55414752
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
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
k
L__inference_stream_0_drop_2_layer_call_and_return_conditional_losses_5545007

inputs
identity�c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2
dropout/Constx
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:����������@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:����������@*
dtype0*
seed�*
seed2�2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���>2
dropout/GreaterEqual/y�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:����������@2
dropout/GreaterEqual�
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:����������@2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:����������@2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:����������@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������@:T P
,
_output_shapes
:����������@
 
_user_specified_nameinputs
�
j
1__inference_stream_0_drop_1_layer_call_fn_5544784

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
:���������� * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_stream_0_drop_1_layer_call_and_return_conditional_losses_55417242
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:���������� 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:���������� 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:���������� 
 
_user_specified_nameinputs
�
J
.__inference_activation_1_layer_call_fn_5544990

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
:����������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_activation_1_layer_call_and_return_conditional_losses_55414282
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:����������@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������@:T P
,
_output_shapes
:����������@
 
_user_specified_nameinputs
��
�
F__inference_basemodel_layer_call_and_return_conditional_losses_5544145
inputs_0Q
;stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource: =
/stream_0_conv_1_biasadd_readvariableop_resource: C
5batch_normalization_batchnorm_readvariableop_resource: G
9batch_normalization_batchnorm_mul_readvariableop_resource: E
7batch_normalization_batchnorm_readvariableop_1_resource: E
7batch_normalization_batchnorm_readvariableop_2_resource: Q
;stream_0_conv_2_conv1d_expanddims_1_readvariableop_resource: @=
/stream_0_conv_2_biasadd_readvariableop_resource:@E
7batch_normalization_1_batchnorm_readvariableop_resource:@I
;batch_normalization_1_batchnorm_mul_readvariableop_resource:@G
9batch_normalization_1_batchnorm_readvariableop_1_resource:@G
9batch_normalization_1_batchnorm_readvariableop_2_resource:@8
&dense_1_matmul_readvariableop_resource:@T5
'dense_1_biasadd_readvariableop_resource:TE
7batch_normalization_2_batchnorm_readvariableop_resource:TI
;batch_normalization_2_batchnorm_mul_readvariableop_resource:TG
9batch_normalization_2_batchnorm_readvariableop_1_resource:TG
9batch_normalization_2_batchnorm_readvariableop_2_resource:T
identity��,batch_normalization/batchnorm/ReadVariableOp�.batch_normalization/batchnorm/ReadVariableOp_1�.batch_normalization/batchnorm/ReadVariableOp_2�0batch_normalization/batchnorm/mul/ReadVariableOp�.batch_normalization_1/batchnorm/ReadVariableOp�0batch_normalization_1/batchnorm/ReadVariableOp_1�0batch_normalization_1/batchnorm/ReadVariableOp_2�2batch_normalization_1/batchnorm/mul/ReadVariableOp�.batch_normalization_2/batchnorm/ReadVariableOp�0batch_normalization_2/batchnorm/ReadVariableOp_1�0batch_normalization_2/batchnorm/ReadVariableOp_2�2batch_normalization_2/batchnorm/mul/ReadVariableOp�dense_1/BiasAdd/ReadVariableOp�dense_1/MatMul/ReadVariableOp�-dense_1/kernel/Regularizer/Abs/ReadVariableOp�&stream_0_conv_1/BiasAdd/ReadVariableOp�2stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp�5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp�&stream_0_conv_2/BiasAdd/ReadVariableOp�2stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOp�8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp�
stream_0_input_drop/IdentityIdentityinputs_0*
T0*,
_output_shapes
:����������2
stream_0_input_drop/Identity�
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
:����������2#
!stream_0_conv_1/conv1d/ExpandDims�
2stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp;stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
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
: 2%
#stream_0_conv_1/conv1d/ExpandDims_1�
stream_0_conv_1/conv1dConv2D*stream_0_conv_1/conv1d/ExpandDims:output:0,stream_0_conv_1/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:���������� *
paddingSAME*
strides
2
stream_0_conv_1/conv1d�
stream_0_conv_1/conv1d/SqueezeSqueezestream_0_conv_1/conv1d:output:0*
T0*,
_output_shapes
:���������� *
squeeze_dims

���������2 
stream_0_conv_1/conv1d/Squeeze�
&stream_0_conv_1/BiasAdd/ReadVariableOpReadVariableOp/stream_0_conv_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02(
&stream_0_conv_1/BiasAdd/ReadVariableOp�
stream_0_conv_1/BiasAddBiasAdd'stream_0_conv_1/conv1d/Squeeze:output:0.stream_0_conv_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������� 2
stream_0_conv_1/BiasAdd�
,batch_normalization/batchnorm/ReadVariableOpReadVariableOp5batch_normalization_batchnorm_readvariableop_resource*
_output_shapes
: *
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
: 2#
!batch_normalization/batchnorm/add�
#batch_normalization/batchnorm/RsqrtRsqrt%batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes
: 2%
#batch_normalization/batchnorm/Rsqrt�
0batch_normalization/batchnorm/mul/ReadVariableOpReadVariableOp9batch_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype022
0batch_normalization/batchnorm/mul/ReadVariableOp�
!batch_normalization/batchnorm/mulMul'batch_normalization/batchnorm/Rsqrt:y:08batch_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2#
!batch_normalization/batchnorm/mul�
#batch_normalization/batchnorm/mul_1Mul stream_0_conv_1/BiasAdd:output:0%batch_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:���������� 2%
#batch_normalization/batchnorm/mul_1�
.batch_normalization/batchnorm/ReadVariableOp_1ReadVariableOp7batch_normalization_batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype020
.batch_normalization/batchnorm/ReadVariableOp_1�
#batch_normalization/batchnorm/mul_2Mul6batch_normalization/batchnorm/ReadVariableOp_1:value:0%batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes
: 2%
#batch_normalization/batchnorm/mul_2�
.batch_normalization/batchnorm/ReadVariableOp_2ReadVariableOp7batch_normalization_batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype020
.batch_normalization/batchnorm/ReadVariableOp_2�
!batch_normalization/batchnorm/subSub6batch_normalization/batchnorm/ReadVariableOp_2:value:0'batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2#
!batch_normalization/batchnorm/sub�
#batch_normalization/batchnorm/add_1AddV2'batch_normalization/batchnorm/mul_1:z:0%batch_normalization/batchnorm/sub:z:0*
T0*,
_output_shapes
:���������� 2%
#batch_normalization/batchnorm/add_1�
activation/ReluRelu'batch_normalization/batchnorm/add_1:z:0*
T0*,
_output_shapes
:���������� 2
activation/Relu�
stream_0_drop_1/IdentityIdentityactivation/Relu:activations:0*
T0*,
_output_shapes
:���������� 2
stream_0_drop_1/Identity�
%stream_0_conv_2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2'
%stream_0_conv_2/conv1d/ExpandDims/dim�
!stream_0_conv_2/conv1d/ExpandDims
ExpandDims!stream_0_drop_1/Identity:output:0.stream_0_conv_2/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:���������� 2#
!stream_0_conv_2/conv1d/ExpandDims�
2stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp;stream_0_conv_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: @*
dtype024
2stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOp�
'stream_0_conv_2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2)
'stream_0_conv_2/conv1d/ExpandDims_1/dim�
#stream_0_conv_2/conv1d/ExpandDims_1
ExpandDims:stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOp:value:00stream_0_conv_2/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: @2%
#stream_0_conv_2/conv1d/ExpandDims_1�
stream_0_conv_2/conv1dConv2D*stream_0_conv_2/conv1d/ExpandDims:output:0,stream_0_conv_2/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:����������@*
paddingSAME*
strides
2
stream_0_conv_2/conv1d�
stream_0_conv_2/conv1d/SqueezeSqueezestream_0_conv_2/conv1d:output:0*
T0*,
_output_shapes
:����������@*
squeeze_dims

���������2 
stream_0_conv_2/conv1d/Squeeze�
&stream_0_conv_2/BiasAdd/ReadVariableOpReadVariableOp/stream_0_conv_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02(
&stream_0_conv_2/BiasAdd/ReadVariableOp�
stream_0_conv_2/BiasAddBiasAdd'stream_0_conv_2/conv1d/Squeeze:output:0.stream_0_conv_2/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������@2
stream_0_conv_2/BiasAdd�
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
%batch_normalization_1/batchnorm/mul_1Mul stream_0_conv_2/BiasAdd:output:0'batch_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:����������@2'
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
:����������@2'
%batch_normalization_1/batchnorm/add_1�
activation_1/ReluRelu)batch_normalization_1/batchnorm/add_1:z:0*
T0*,
_output_shapes
:����������@2
activation_1/Relu�
stream_0_drop_2/IdentityIdentityactivation_1/Relu:activations:0*
T0*,
_output_shapes
:����������@2
stream_0_drop_2/Identity�
/global_average_pooling1d/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :21
/global_average_pooling1d/Mean/reduction_indices�
global_average_pooling1d/MeanMean!stream_0_drop_2/Identity:output:08global_average_pooling1d/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:���������@2
global_average_pooling1d/Mean�
concatenate/concat/concat_dimConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate/concat/concat_dim�
concatenate/concat/concatIdentity&global_average_pooling1d/Mean:output:0*
T0*'
_output_shapes
:���������@2
concatenate/concat/concat�
dense_1_dropout/IdentityIdentity"concatenate/concat/concat:output:0*
T0*'
_output_shapes
:���������@2
dense_1_dropout/Identity�
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:@T*
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
.batch_normalization_2/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_2_batchnorm_readvariableop_resource*
_output_shapes
:T*
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
:T2%
#batch_normalization_2/batchnorm/add�
%batch_normalization_2/batchnorm/RsqrtRsqrt'batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes
:T2'
%batch_normalization_2/batchnorm/Rsqrt�
2batch_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes
:T*
dtype024
2batch_normalization_2/batchnorm/mul/ReadVariableOp�
#batch_normalization_2/batchnorm/mulMul)batch_normalization_2/batchnorm/Rsqrt:y:0:batch_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:T2%
#batch_normalization_2/batchnorm/mul�
%batch_normalization_2/batchnorm/mul_1Muldense_1/BiasAdd:output:0'batch_normalization_2/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������T2'
%batch_normalization_2/batchnorm/mul_1�
0batch_normalization_2/batchnorm/ReadVariableOp_1ReadVariableOp9batch_normalization_2_batchnorm_readvariableop_1_resource*
_output_shapes
:T*
dtype022
0batch_normalization_2/batchnorm/ReadVariableOp_1�
%batch_normalization_2/batchnorm/mul_2Mul8batch_normalization_2/batchnorm/ReadVariableOp_1:value:0'batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes
:T2'
%batch_normalization_2/batchnorm/mul_2�
0batch_normalization_2/batchnorm/ReadVariableOp_2ReadVariableOp9batch_normalization_2_batchnorm_readvariableop_2_resource*
_output_shapes
:T*
dtype022
0batch_normalization_2/batchnorm/ReadVariableOp_2�
#batch_normalization_2/batchnorm/subSub8batch_normalization_2/batchnorm/ReadVariableOp_2:value:0)batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes
:T2%
#batch_normalization_2/batchnorm/sub�
%batch_normalization_2/batchnorm/add_1AddV2)batch_normalization_2/batchnorm/mul_1:z:0'batch_normalization_2/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������T2'
%batch_normalization_2/batchnorm/add_1�
dense_activation_1/SigmoidSigmoid)batch_normalization_2/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������T2
dense_activation_1/Sigmoid�
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp;stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype027
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp�
&stream_0_conv_1/kernel/Regularizer/AbsAbs=stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
: 2(
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
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;stream_0_conv_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: @*
dtype02:
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp�
)stream_0_conv_2/kernel/Regularizer/SquareSquare@stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
: @2+
)stream_0_conv_2/kernel/Regularizer/Square�
(stream_0_conv_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_0_conv_2/kernel/Regularizer/Const�
&stream_0_conv_2/kernel/Regularizer/SumSum-stream_0_conv_2/kernel/Regularizer/Square:y:01stream_0_conv_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_2/kernel/Regularizer/Sum�
(stream_0_conv_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2*
(stream_0_conv_2/kernel/Regularizer/mul/x�
&stream_0_conv_2/kernel/Regularizer/mulMul1stream_0_conv_2/kernel/Regularizer/mul/x:output:0/stream_0_conv_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_2/kernel/Regularizer/mul�
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:@T*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOp�
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:@T2 
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

Identity�
NoOpNoOp-^batch_normalization/batchnorm/ReadVariableOp/^batch_normalization/batchnorm/ReadVariableOp_1/^batch_normalization/batchnorm/ReadVariableOp_21^batch_normalization/batchnorm/mul/ReadVariableOp/^batch_normalization_1/batchnorm/ReadVariableOp1^batch_normalization_1/batchnorm/ReadVariableOp_11^batch_normalization_1/batchnorm/ReadVariableOp_23^batch_normalization_1/batchnorm/mul/ReadVariableOp/^batch_normalization_2/batchnorm/ReadVariableOp1^batch_normalization_2/batchnorm/ReadVariableOp_11^batch_normalization_2/batchnorm/ReadVariableOp_23^batch_normalization_2/batchnorm/mul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp.^dense_1/kernel/Regularizer/Abs/ReadVariableOp'^stream_0_conv_1/BiasAdd/ReadVariableOp3^stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp'^stream_0_conv_2/BiasAdd/ReadVariableOp3^stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOp9^stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:����������: : : : : : : : : : : : : : : : : : 2\
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
2batch_normalization_2/batchnorm/mul/ReadVariableOp2batch_normalization_2/batchnorm/mul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2^
-dense_1/kernel/Regularizer/Abs/ReadVariableOp-dense_1/kernel/Regularizer/Abs/ReadVariableOp2P
&stream_0_conv_1/BiasAdd/ReadVariableOp&stream_0_conv_1/BiasAdd/ReadVariableOp2h
2stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp2stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp2n
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp2P
&stream_0_conv_2/BiasAdd/ReadVariableOp&stream_0_conv_2/BiasAdd/ReadVariableOp2h
2stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOp2stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOp2t
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp:V R
,
_output_shapes
:����������
"
_user_specified_name
inputs/0
�;
�
B__inference_model_layer_call_and_return_conditional_losses_5542825

inputs
inputs_1'
basemodel_5542749: 
basemodel_5542751: 
basemodel_5542753: 
basemodel_5542755: 
basemodel_5542757: 
basemodel_5542759: '
basemodel_5542761: @
basemodel_5542763:@
basemodel_5542765:@
basemodel_5542767:@
basemodel_5542769:@
basemodel_5542771:@#
basemodel_5542773:@T
basemodel_5542775:T
basemodel_5542777:T
basemodel_5542779:T
basemodel_5542781:T
basemodel_5542783:T
identity��!basemodel/StatefulPartitionedCall�#basemodel/StatefulPartitionedCall_1�-dense_1/kernel/Regularizer/Abs/ReadVariableOp�5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp�8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp�
!basemodel/StatefulPartitionedCallStatefulPartitionedCallinputsbasemodel_5542749basemodel_5542751basemodel_5542753basemodel_5542755basemodel_5542757basemodel_5542759basemodel_5542761basemodel_5542763basemodel_5542765basemodel_5542767basemodel_5542769basemodel_5542771basemodel_5542773basemodel_5542775basemodel_5542777basemodel_5542779basemodel_5542781basemodel_5542783*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������T*.
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_basemodel_layer_call_and_return_conditional_losses_55426612#
!basemodel/StatefulPartitionedCall�
#basemodel/StatefulPartitionedCall_1StatefulPartitionedCallinputs_1basemodel_5542749basemodel_5542751basemodel_5542753basemodel_5542755basemodel_5542757basemodel_5542759basemodel_5542761basemodel_5542763basemodel_5542765basemodel_5542767basemodel_5542769basemodel_5542771basemodel_5542773basemodel_5542775basemodel_5542777basemodel_5542779basemodel_5542781basemodel_5542783"^basemodel/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������T*.
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_basemodel_layer_call_and_return_conditional_losses_55426612%
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
E__inference_distance_layer_call_and_return_conditional_losses_55424362
distance/PartitionedCall�
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpbasemodel_5542749*"
_output_shapes
: *
dtype027
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp�
&stream_0_conv_1/kernel/Regularizer/AbsAbs=stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
: 2(
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
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpbasemodel_5542761*"
_output_shapes
: @*
dtype02:
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp�
)stream_0_conv_2/kernel/Regularizer/SquareSquare@stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
: @2+
)stream_0_conv_2/kernel/Regularizer/Square�
(stream_0_conv_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_0_conv_2/kernel/Regularizer/Const�
&stream_0_conv_2/kernel/Regularizer/SumSum-stream_0_conv_2/kernel/Regularizer/Square:y:01stream_0_conv_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_2/kernel/Regularizer/Sum�
(stream_0_conv_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2*
(stream_0_conv_2/kernel/Regularizer/mul/x�
&stream_0_conv_2/kernel/Regularizer/mulMul1stream_0_conv_2/kernel/Regularizer/mul/x:output:0/stream_0_conv_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_2/kernel/Regularizer/mul�
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpbasemodel_5542773*
_output_shapes

:@T*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOp�
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:@T2 
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
NoOpNoOp"^basemodel/StatefulPartitionedCall$^basemodel/StatefulPartitionedCall_1.^dense_1/kernel/Regularizer/Abs/ReadVariableOp6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp9^stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*g
_input_shapesV
T:����������:����������: : : : : : : : : : : : : : : : : : 2F
!basemodel/StatefulPartitionedCall!basemodel/StatefulPartitionedCall2J
#basemodel/StatefulPartitionedCall_1#basemodel/StatefulPartitionedCall_12^
-dense_1/kernel/Regularizer/Abs/ReadVariableOp-dense_1/kernel/Regularizer/Abs/ReadVariableOp2n
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp2t
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs:TP
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
V
*__inference_distance_layer_call_fn_5544524
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
E__inference_distance_layer_call_and_return_conditional_losses_55424362
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
�
�
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_5545128

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
�	
o
E__inference_distance_layer_call_and_return_conditional_losses_5542436

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
�
M
1__inference_dense_1_dropout_layer_call_fn_5545072

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
GPU2*0J 8� *U
fPRN
L__inference_dense_1_dropout_layer_call_and_return_conditional_losses_55414572
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������@:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�n
�
F__inference_basemodel_layer_call_and_return_conditional_losses_5541942

inputs-
stream_0_conv_1_5541873: %
stream_0_conv_1_5541875: )
batch_normalization_5541878: )
batch_normalization_5541880: )
batch_normalization_5541882: )
batch_normalization_5541884: -
stream_0_conv_2_5541889: @%
stream_0_conv_2_5541891:@+
batch_normalization_1_5541894:@+
batch_normalization_1_5541896:@+
batch_normalization_1_5541898:@+
batch_normalization_1_5541900:@!
dense_1_5541908:@T
dense_1_5541910:T+
batch_normalization_2_5541913:T+
batch_normalization_2_5541915:T+
batch_normalization_2_5541917:T+
batch_normalization_2_5541919:T
identity��+batch_normalization/StatefulPartitionedCall�-batch_normalization_1/StatefulPartitionedCall�-batch_normalization_2/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�-dense_1/kernel/Regularizer/Abs/ReadVariableOp�'dense_1_dropout/StatefulPartitionedCall�'stream_0_conv_1/StatefulPartitionedCall�5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp�'stream_0_conv_2/StatefulPartitionedCall�8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp�'stream_0_drop_1/StatefulPartitionedCall�'stream_0_drop_2/StatefulPartitionedCall�+stream_0_input_drop/StatefulPartitionedCall�
+stream_0_input_drop/StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *Y
fTRR
P__inference_stream_0_input_drop_layer_call_and_return_conditional_losses_55418232-
+stream_0_input_drop/StatefulPartitionedCall�
'stream_0_conv_1/StatefulPartitionedCallStatefulPartitionedCall4stream_0_input_drop/StatefulPartitionedCall:output:0stream_0_conv_1_5541873stream_0_conv_1_5541875*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_stream_0_conv_1_layer_call_and_return_conditional_losses_55413182)
'stream_0_conv_1/StatefulPartitionedCall�
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall0stream_0_conv_1/StatefulPartitionedCall:output:0batch_normalization_5541878batch_normalization_5541880batch_normalization_5541882batch_normalization_5541884*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_55417822-
+batch_normalization/StatefulPartitionedCall�
activation/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������� * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_activation_layer_call_and_return_conditional_losses_55413582
activation/PartitionedCall�
'stream_0_drop_1/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0,^stream_0_input_drop/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������� * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_stream_0_drop_1_layer_call_and_return_conditional_losses_55417242)
'stream_0_drop_1/StatefulPartitionedCall�
'stream_0_conv_2/StatefulPartitionedCallStatefulPartitionedCall0stream_0_drop_1/StatefulPartitionedCall:output:0stream_0_conv_2_5541889stream_0_conv_2_5541891*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_stream_0_conv_2_layer_call_and_return_conditional_losses_55413882)
'stream_0_conv_2/StatefulPartitionedCall�
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall0stream_0_conv_2/StatefulPartitionedCall:output:0batch_normalization_1_5541894batch_normalization_1_5541896batch_normalization_1_5541898batch_normalization_1_5541900*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *[
fVRT
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_55416832/
-batch_normalization_1/StatefulPartitionedCall�
activation_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_activation_1_layer_call_and_return_conditional_losses_55414282
activation_1/PartitionedCall�
'stream_0_drop_2/StatefulPartitionedCallStatefulPartitionedCall%activation_1/PartitionedCall:output:0(^stream_0_drop_1/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_stream_0_drop_2_layer_call_and_return_conditional_losses_55416252)
'stream_0_drop_2/StatefulPartitionedCall�
(global_average_pooling1d/PartitionedCallPartitionedCall0stream_0_drop_2/StatefulPartitionedCall:output:0*
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
U__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_55414422*
(global_average_pooling1d/PartitionedCall�
concatenate/PartitionedCallPartitionedCall1global_average_pooling1d/PartitionedCall:output:0*
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
GPU2*0J 8� *Q
fLRJ
H__inference_concatenate_layer_call_and_return_conditional_losses_55414502
concatenate/PartitionedCall�
'dense_1_dropout/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0(^stream_0_drop_2/StatefulPartitionedCall*
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
GPU2*0J 8� *U
fPRN
L__inference_dense_1_dropout_layer_call_and_return_conditional_losses_55415912)
'dense_1_dropout/StatefulPartitionedCall�
dense_1/StatefulPartitionedCallStatefulPartitionedCall0dense_1_dropout/StatefulPartitionedCall:output:0dense_1_5541908dense_1_5541910*
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
D__inference_dense_1_layer_call_and_return_conditional_losses_55414752!
dense_1/StatefulPartitionedCall�
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0batch_normalization_2_5541913batch_normalization_2_5541915batch_normalization_2_5541917batch_normalization_2_5541919*
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
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_55412052/
-batch_normalization_2/StatefulPartitionedCall�
"dense_activation_1/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
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
O__inference_dense_activation_1_layer_call_and_return_conditional_losses_55414952$
"dense_activation_1/PartitionedCall�
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpstream_0_conv_1_5541873*"
_output_shapes
: *
dtype027
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp�
&stream_0_conv_1/kernel/Regularizer/AbsAbs=stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
: 2(
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
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpstream_0_conv_2_5541889*"
_output_shapes
: @*
dtype02:
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp�
)stream_0_conv_2/kernel/Regularizer/SquareSquare@stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
: @2+
)stream_0_conv_2/kernel/Regularizer/Square�
(stream_0_conv_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_0_conv_2/kernel/Regularizer/Const�
&stream_0_conv_2/kernel/Regularizer/SumSum-stream_0_conv_2/kernel/Regularizer/Square:y:01stream_0_conv_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_2/kernel/Regularizer/Sum�
(stream_0_conv_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2*
(stream_0_conv_2/kernel/Regularizer/mul/x�
&stream_0_conv_2/kernel/Regularizer/mulMul1stream_0_conv_2/kernel/Regularizer/mul/x:output:0/stream_0_conv_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_2/kernel/Regularizer/mul�
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_1_5541908*
_output_shapes

:@T*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOp�
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:@T2 
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
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall.^dense_1/kernel/Regularizer/Abs/ReadVariableOp(^dense_1_dropout/StatefulPartitionedCall(^stream_0_conv_1/StatefulPartitionedCall6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp(^stream_0_conv_2/StatefulPartitionedCall9^stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp(^stream_0_drop_1/StatefulPartitionedCall(^stream_0_drop_2/StatefulPartitionedCall,^stream_0_input_drop/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:����������: : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2^
-dense_1/kernel/Regularizer/Abs/ReadVariableOp-dense_1/kernel/Regularizer/Abs/ReadVariableOp2R
'dense_1_dropout/StatefulPartitionedCall'dense_1_dropout/StatefulPartitionedCall2R
'stream_0_conv_1/StatefulPartitionedCall'stream_0_conv_1/StatefulPartitionedCall2n
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp2R
'stream_0_conv_2/StatefulPartitionedCall'stream_0_conv_2/StatefulPartitionedCall2t
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp2R
'stream_0_drop_1/StatefulPartitionedCall'stream_0_drop_1/StatefulPartitionedCall2R
'stream_0_drop_2/StatefulPartitionedCall'stream_0_drop_2/StatefulPartitionedCall2Z
+stream_0_input_drop/StatefulPartitionedCall+stream_0_input_drop/StatefulPartitionedCall:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
j
L__inference_stream_0_drop_1_layer_call_and_return_conditional_losses_5544762

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:���������� 2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:���������� 2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:���������� :T P
,
_output_shapes
:���������� 
 
_user_specified_nameinputs
�:
�
B__inference_model_layer_call_and_return_conditional_losses_5542375

inputs
inputs_1'
basemodel_5542286: 
basemodel_5542288: 
basemodel_5542290: 
basemodel_5542292: 
basemodel_5542294: 
basemodel_5542296: '
basemodel_5542298: @
basemodel_5542300:@
basemodel_5542302:@
basemodel_5542304:@
basemodel_5542306:@
basemodel_5542308:@#
basemodel_5542310:@T
basemodel_5542312:T
basemodel_5542314:T
basemodel_5542316:T
basemodel_5542318:T
basemodel_5542320:T
identity��!basemodel/StatefulPartitionedCall�#basemodel/StatefulPartitionedCall_1�-dense_1/kernel/Regularizer/Abs/ReadVariableOp�5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp�8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp�
!basemodel/StatefulPartitionedCallStatefulPartitionedCallinputsbasemodel_5542286basemodel_5542288basemodel_5542290basemodel_5542292basemodel_5542294basemodel_5542296basemodel_5542298basemodel_5542300basemodel_5542302basemodel_5542304basemodel_5542306basemodel_5542308basemodel_5542310basemodel_5542312basemodel_5542314basemodel_5542316basemodel_5542318basemodel_5542320*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������T*4
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_basemodel_layer_call_and_return_conditional_losses_55422852#
!basemodel/StatefulPartitionedCall�
#basemodel/StatefulPartitionedCall_1StatefulPartitionedCallinputs_1basemodel_5542286basemodel_5542288basemodel_5542290basemodel_5542292basemodel_5542294basemodel_5542296basemodel_5542298basemodel_5542300basemodel_5542302basemodel_5542304basemodel_5542306basemodel_5542308basemodel_5542310basemodel_5542312basemodel_5542314basemodel_5542316basemodel_5542318basemodel_5542320*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������T*4
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_basemodel_layer_call_and_return_conditional_losses_55422852%
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
E__inference_distance_layer_call_and_return_conditional_losses_55423542
distance/PartitionedCall�
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpbasemodel_5542286*"
_output_shapes
: *
dtype027
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp�
&stream_0_conv_1/kernel/Regularizer/AbsAbs=stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
: 2(
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
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpbasemodel_5542298*"
_output_shapes
: @*
dtype02:
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp�
)stream_0_conv_2/kernel/Regularizer/SquareSquare@stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
: @2+
)stream_0_conv_2/kernel/Regularizer/Square�
(stream_0_conv_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_0_conv_2/kernel/Regularizer/Const�
&stream_0_conv_2/kernel/Regularizer/SumSum-stream_0_conv_2/kernel/Regularizer/Square:y:01stream_0_conv_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_2/kernel/Regularizer/Sum�
(stream_0_conv_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2*
(stream_0_conv_2/kernel/Regularizer/mul/x�
&stream_0_conv_2/kernel/Regularizer/mulMul1stream_0_conv_2/kernel/Regularizer/mul/x:output:0/stream_0_conv_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_2/kernel/Regularizer/mul�
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpbasemodel_5542310*
_output_shapes

:@T*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOp�
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:@T2 
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
NoOpNoOp"^basemodel/StatefulPartitionedCall$^basemodel/StatefulPartitionedCall_1.^dense_1/kernel/Regularizer/Abs/ReadVariableOp6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp9^stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*g
_input_shapesV
T:����������:����������: : : : : : : : : : : : : : : : : : 2F
!basemodel/StatefulPartitionedCall!basemodel/StatefulPartitionedCall2J
#basemodel/StatefulPartitionedCall_1#basemodel/StatefulPartitionedCall_12^
-dense_1/kernel/Regularizer/Abs/ReadVariableOp-dense_1/kernel/Regularizer/Abs/ReadVariableOp2n
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp2t
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs:TP
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
K
-__inference_concatenate_layer_call_fn_5545050
inputs_0
identity�
PartitionedCallPartitionedCallinputs_0*
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
GPU2*0J 8� *Q
fLRJ
H__inference_concatenate_layer_call_and_return_conditional_losses_55414502
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������@:Q M
'
_output_shapes
:���������@
"
_user_specified_name
inputs/0
��
�
"__inference__wrapped_model_5540773
left_inputs
right_inputsa
Kmodel_basemodel_stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource: M
?model_basemodel_stream_0_conv_1_biasadd_readvariableop_resource: S
Emodel_basemodel_batch_normalization_batchnorm_readvariableop_resource: W
Imodel_basemodel_batch_normalization_batchnorm_mul_readvariableop_resource: U
Gmodel_basemodel_batch_normalization_batchnorm_readvariableop_1_resource: U
Gmodel_basemodel_batch_normalization_batchnorm_readvariableop_2_resource: a
Kmodel_basemodel_stream_0_conv_2_conv1d_expanddims_1_readvariableop_resource: @M
?model_basemodel_stream_0_conv_2_biasadd_readvariableop_resource:@U
Gmodel_basemodel_batch_normalization_1_batchnorm_readvariableop_resource:@Y
Kmodel_basemodel_batch_normalization_1_batchnorm_mul_readvariableop_resource:@W
Imodel_basemodel_batch_normalization_1_batchnorm_readvariableop_1_resource:@W
Imodel_basemodel_batch_normalization_1_batchnorm_readvariableop_2_resource:@H
6model_basemodel_dense_1_matmul_readvariableop_resource:@TE
7model_basemodel_dense_1_biasadd_readvariableop_resource:TU
Gmodel_basemodel_batch_normalization_2_batchnorm_readvariableop_resource:TY
Kmodel_basemodel_batch_normalization_2_batchnorm_mul_readvariableop_resource:TW
Imodel_basemodel_batch_normalization_2_batchnorm_readvariableop_1_resource:TW
Imodel_basemodel_batch_normalization_2_batchnorm_readvariableop_2_resource:T
identity��<model/basemodel/batch_normalization/batchnorm/ReadVariableOp�>model/basemodel/batch_normalization/batchnorm/ReadVariableOp_1�>model/basemodel/batch_normalization/batchnorm/ReadVariableOp_2�@model/basemodel/batch_normalization/batchnorm/mul/ReadVariableOp�>model/basemodel/batch_normalization/batchnorm_1/ReadVariableOp�@model/basemodel/batch_normalization/batchnorm_1/ReadVariableOp_1�@model/basemodel/batch_normalization/batchnorm_1/ReadVariableOp_2�Bmodel/basemodel/batch_normalization/batchnorm_1/mul/ReadVariableOp�>model/basemodel/batch_normalization_1/batchnorm/ReadVariableOp�@model/basemodel/batch_normalization_1/batchnorm/ReadVariableOp_1�@model/basemodel/batch_normalization_1/batchnorm/ReadVariableOp_2�Bmodel/basemodel/batch_normalization_1/batchnorm/mul/ReadVariableOp�@model/basemodel/batch_normalization_1/batchnorm_1/ReadVariableOp�Bmodel/basemodel/batch_normalization_1/batchnorm_1/ReadVariableOp_1�Bmodel/basemodel/batch_normalization_1/batchnorm_1/ReadVariableOp_2�Dmodel/basemodel/batch_normalization_1/batchnorm_1/mul/ReadVariableOp�>model/basemodel/batch_normalization_2/batchnorm/ReadVariableOp�@model/basemodel/batch_normalization_2/batchnorm/ReadVariableOp_1�@model/basemodel/batch_normalization_2/batchnorm/ReadVariableOp_2�Bmodel/basemodel/batch_normalization_2/batchnorm/mul/ReadVariableOp�@model/basemodel/batch_normalization_2/batchnorm_1/ReadVariableOp�Bmodel/basemodel/batch_normalization_2/batchnorm_1/ReadVariableOp_1�Bmodel/basemodel/batch_normalization_2/batchnorm_1/ReadVariableOp_2�Dmodel/basemodel/batch_normalization_2/batchnorm_1/mul/ReadVariableOp�.model/basemodel/dense_1/BiasAdd/ReadVariableOp�0model/basemodel/dense_1/BiasAdd_1/ReadVariableOp�-model/basemodel/dense_1/MatMul/ReadVariableOp�/model/basemodel/dense_1/MatMul_1/ReadVariableOp�6model/basemodel/stream_0_conv_1/BiasAdd/ReadVariableOp�8model/basemodel/stream_0_conv_1/BiasAdd_1/ReadVariableOp�Bmodel/basemodel/stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp�Dmodel/basemodel/stream_0_conv_1/conv1d_1/ExpandDims_1/ReadVariableOp�6model/basemodel/stream_0_conv_2/BiasAdd/ReadVariableOp�8model/basemodel/stream_0_conv_2/BiasAdd_1/ReadVariableOp�Bmodel/basemodel/stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOp�Dmodel/basemodel/stream_0_conv_2/conv1d_1/ExpandDims_1/ReadVariableOp�
,model/basemodel/stream_0_input_drop/IdentityIdentityleft_inputs*
T0*,
_output_shapes
:����������2.
,model/basemodel/stream_0_input_drop/Identity�
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
:����������23
1model/basemodel/stream_0_conv_1/conv1d/ExpandDims�
Bmodel/basemodel/stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpKmodel_basemodel_stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
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
: 25
3model/basemodel/stream_0_conv_1/conv1d/ExpandDims_1�
&model/basemodel/stream_0_conv_1/conv1dConv2D:model/basemodel/stream_0_conv_1/conv1d/ExpandDims:output:0<model/basemodel/stream_0_conv_1/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:���������� *
paddingSAME*
strides
2(
&model/basemodel/stream_0_conv_1/conv1d�
.model/basemodel/stream_0_conv_1/conv1d/SqueezeSqueeze/model/basemodel/stream_0_conv_1/conv1d:output:0*
T0*,
_output_shapes
:���������� *
squeeze_dims

���������20
.model/basemodel/stream_0_conv_1/conv1d/Squeeze�
6model/basemodel/stream_0_conv_1/BiasAdd/ReadVariableOpReadVariableOp?model_basemodel_stream_0_conv_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype028
6model/basemodel/stream_0_conv_1/BiasAdd/ReadVariableOp�
'model/basemodel/stream_0_conv_1/BiasAddBiasAdd7model/basemodel/stream_0_conv_1/conv1d/Squeeze:output:0>model/basemodel/stream_0_conv_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������� 2)
'model/basemodel/stream_0_conv_1/BiasAdd�
<model/basemodel/batch_normalization/batchnorm/ReadVariableOpReadVariableOpEmodel_basemodel_batch_normalization_batchnorm_readvariableop_resource*
_output_shapes
: *
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
: 23
1model/basemodel/batch_normalization/batchnorm/add�
3model/basemodel/batch_normalization/batchnorm/RsqrtRsqrt5model/basemodel/batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes
: 25
3model/basemodel/batch_normalization/batchnorm/Rsqrt�
@model/basemodel/batch_normalization/batchnorm/mul/ReadVariableOpReadVariableOpImodel_basemodel_batch_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02B
@model/basemodel/batch_normalization/batchnorm/mul/ReadVariableOp�
1model/basemodel/batch_normalization/batchnorm/mulMul7model/basemodel/batch_normalization/batchnorm/Rsqrt:y:0Hmodel/basemodel/batch_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 23
1model/basemodel/batch_normalization/batchnorm/mul�
3model/basemodel/batch_normalization/batchnorm/mul_1Mul0model/basemodel/stream_0_conv_1/BiasAdd:output:05model/basemodel/batch_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:���������� 25
3model/basemodel/batch_normalization/batchnorm/mul_1�
>model/basemodel/batch_normalization/batchnorm/ReadVariableOp_1ReadVariableOpGmodel_basemodel_batch_normalization_batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype02@
>model/basemodel/batch_normalization/batchnorm/ReadVariableOp_1�
3model/basemodel/batch_normalization/batchnorm/mul_2MulFmodel/basemodel/batch_normalization/batchnorm/ReadVariableOp_1:value:05model/basemodel/batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes
: 25
3model/basemodel/batch_normalization/batchnorm/mul_2�
>model/basemodel/batch_normalization/batchnorm/ReadVariableOp_2ReadVariableOpGmodel_basemodel_batch_normalization_batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype02@
>model/basemodel/batch_normalization/batchnorm/ReadVariableOp_2�
1model/basemodel/batch_normalization/batchnorm/subSubFmodel/basemodel/batch_normalization/batchnorm/ReadVariableOp_2:value:07model/basemodel/batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes
: 23
1model/basemodel/batch_normalization/batchnorm/sub�
3model/basemodel/batch_normalization/batchnorm/add_1AddV27model/basemodel/batch_normalization/batchnorm/mul_1:z:05model/basemodel/batch_normalization/batchnorm/sub:z:0*
T0*,
_output_shapes
:���������� 25
3model/basemodel/batch_normalization/batchnorm/add_1�
model/basemodel/activation/ReluRelu7model/basemodel/batch_normalization/batchnorm/add_1:z:0*
T0*,
_output_shapes
:���������� 2!
model/basemodel/activation/Relu�
(model/basemodel/stream_0_drop_1/IdentityIdentity-model/basemodel/activation/Relu:activations:0*
T0*,
_output_shapes
:���������� 2*
(model/basemodel/stream_0_drop_1/Identity�
5model/basemodel/stream_0_conv_2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������27
5model/basemodel/stream_0_conv_2/conv1d/ExpandDims/dim�
1model/basemodel/stream_0_conv_2/conv1d/ExpandDims
ExpandDims1model/basemodel/stream_0_drop_1/Identity:output:0>model/basemodel/stream_0_conv_2/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:���������� 23
1model/basemodel/stream_0_conv_2/conv1d/ExpandDims�
Bmodel/basemodel/stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpKmodel_basemodel_stream_0_conv_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: @*
dtype02D
Bmodel/basemodel/stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOp�
7model/basemodel/stream_0_conv_2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 29
7model/basemodel/stream_0_conv_2/conv1d/ExpandDims_1/dim�
3model/basemodel/stream_0_conv_2/conv1d/ExpandDims_1
ExpandDimsJmodel/basemodel/stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOp:value:0@model/basemodel/stream_0_conv_2/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: @25
3model/basemodel/stream_0_conv_2/conv1d/ExpandDims_1�
&model/basemodel/stream_0_conv_2/conv1dConv2D:model/basemodel/stream_0_conv_2/conv1d/ExpandDims:output:0<model/basemodel/stream_0_conv_2/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:����������@*
paddingSAME*
strides
2(
&model/basemodel/stream_0_conv_2/conv1d�
.model/basemodel/stream_0_conv_2/conv1d/SqueezeSqueeze/model/basemodel/stream_0_conv_2/conv1d:output:0*
T0*,
_output_shapes
:����������@*
squeeze_dims

���������20
.model/basemodel/stream_0_conv_2/conv1d/Squeeze�
6model/basemodel/stream_0_conv_2/BiasAdd/ReadVariableOpReadVariableOp?model_basemodel_stream_0_conv_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype028
6model/basemodel/stream_0_conv_2/BiasAdd/ReadVariableOp�
'model/basemodel/stream_0_conv_2/BiasAddBiasAdd7model/basemodel/stream_0_conv_2/conv1d/Squeeze:output:0>model/basemodel/stream_0_conv_2/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������@2)
'model/basemodel/stream_0_conv_2/BiasAdd�
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
5model/basemodel/batch_normalization_1/batchnorm/mul_1Mul0model/basemodel/stream_0_conv_2/BiasAdd:output:07model/basemodel/batch_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:����������@27
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
:����������@27
5model/basemodel/batch_normalization_1/batchnorm/add_1�
!model/basemodel/activation_1/ReluRelu9model/basemodel/batch_normalization_1/batchnorm/add_1:z:0*
T0*,
_output_shapes
:����������@2#
!model/basemodel/activation_1/Relu�
(model/basemodel/stream_0_drop_2/IdentityIdentity/model/basemodel/activation_1/Relu:activations:0*
T0*,
_output_shapes
:����������@2*
(model/basemodel/stream_0_drop_2/Identity�
?model/basemodel/global_average_pooling1d/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2A
?model/basemodel/global_average_pooling1d/Mean/reduction_indices�
-model/basemodel/global_average_pooling1d/MeanMean1model/basemodel/stream_0_drop_2/Identity:output:0Hmodel/basemodel/global_average_pooling1d/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:���������@2/
-model/basemodel/global_average_pooling1d/Mean�
-model/basemodel/concatenate/concat/concat_dimConst*
_output_shapes
: *
dtype0*
value	B :2/
-model/basemodel/concatenate/concat/concat_dim�
)model/basemodel/concatenate/concat/concatIdentity6model/basemodel/global_average_pooling1d/Mean:output:0*
T0*'
_output_shapes
:���������@2+
)model/basemodel/concatenate/concat/concat�
(model/basemodel/dense_1_dropout/IdentityIdentity2model/basemodel/concatenate/concat/concat:output:0*
T0*'
_output_shapes
:���������@2*
(model/basemodel/dense_1_dropout/Identity�
-model/basemodel/dense_1/MatMul/ReadVariableOpReadVariableOp6model_basemodel_dense_1_matmul_readvariableop_resource*
_output_shapes

:@T*
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
>model/basemodel/batch_normalization_2/batchnorm/ReadVariableOpReadVariableOpGmodel_basemodel_batch_normalization_2_batchnorm_readvariableop_resource*
_output_shapes
:T*
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
:T25
3model/basemodel/batch_normalization_2/batchnorm/add�
5model/basemodel/batch_normalization_2/batchnorm/RsqrtRsqrt7model/basemodel/batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes
:T27
5model/basemodel/batch_normalization_2/batchnorm/Rsqrt�
Bmodel/basemodel/batch_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOpKmodel_basemodel_batch_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes
:T*
dtype02D
Bmodel/basemodel/batch_normalization_2/batchnorm/mul/ReadVariableOp�
3model/basemodel/batch_normalization_2/batchnorm/mulMul9model/basemodel/batch_normalization_2/batchnorm/Rsqrt:y:0Jmodel/basemodel/batch_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:T25
3model/basemodel/batch_normalization_2/batchnorm/mul�
5model/basemodel/batch_normalization_2/batchnorm/mul_1Mul(model/basemodel/dense_1/BiasAdd:output:07model/basemodel/batch_normalization_2/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������T27
5model/basemodel/batch_normalization_2/batchnorm/mul_1�
@model/basemodel/batch_normalization_2/batchnorm/ReadVariableOp_1ReadVariableOpImodel_basemodel_batch_normalization_2_batchnorm_readvariableop_1_resource*
_output_shapes
:T*
dtype02B
@model/basemodel/batch_normalization_2/batchnorm/ReadVariableOp_1�
5model/basemodel/batch_normalization_2/batchnorm/mul_2MulHmodel/basemodel/batch_normalization_2/batchnorm/ReadVariableOp_1:value:07model/basemodel/batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes
:T27
5model/basemodel/batch_normalization_2/batchnorm/mul_2�
@model/basemodel/batch_normalization_2/batchnorm/ReadVariableOp_2ReadVariableOpImodel_basemodel_batch_normalization_2_batchnorm_readvariableop_2_resource*
_output_shapes
:T*
dtype02B
@model/basemodel/batch_normalization_2/batchnorm/ReadVariableOp_2�
3model/basemodel/batch_normalization_2/batchnorm/subSubHmodel/basemodel/batch_normalization_2/batchnorm/ReadVariableOp_2:value:09model/basemodel/batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes
:T25
3model/basemodel/batch_normalization_2/batchnorm/sub�
5model/basemodel/batch_normalization_2/batchnorm/add_1AddV29model/basemodel/batch_normalization_2/batchnorm/mul_1:z:07model/basemodel/batch_normalization_2/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������T27
5model/basemodel/batch_normalization_2/batchnorm/add_1�
*model/basemodel/dense_activation_1/SigmoidSigmoid9model/basemodel/batch_normalization_2/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������T2,
*model/basemodel/dense_activation_1/Sigmoid�
.model/basemodel/stream_0_input_drop/Identity_1Identityright_inputs*
T0*,
_output_shapes
:����������20
.model/basemodel/stream_0_input_drop/Identity_1�
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
:����������25
3model/basemodel/stream_0_conv_1/conv1d_1/ExpandDims�
Dmodel/basemodel/stream_0_conv_1/conv1d_1/ExpandDims_1/ReadVariableOpReadVariableOpKmodel_basemodel_stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
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
: 27
5model/basemodel/stream_0_conv_1/conv1d_1/ExpandDims_1�
(model/basemodel/stream_0_conv_1/conv1d_1Conv2D<model/basemodel/stream_0_conv_1/conv1d_1/ExpandDims:output:0>model/basemodel/stream_0_conv_1/conv1d_1/ExpandDims_1:output:0*
T0*0
_output_shapes
:���������� *
paddingSAME*
strides
2*
(model/basemodel/stream_0_conv_1/conv1d_1�
0model/basemodel/stream_0_conv_1/conv1d_1/SqueezeSqueeze1model/basemodel/stream_0_conv_1/conv1d_1:output:0*
T0*,
_output_shapes
:���������� *
squeeze_dims

���������22
0model/basemodel/stream_0_conv_1/conv1d_1/Squeeze�
8model/basemodel/stream_0_conv_1/BiasAdd_1/ReadVariableOpReadVariableOp?model_basemodel_stream_0_conv_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02:
8model/basemodel/stream_0_conv_1/BiasAdd_1/ReadVariableOp�
)model/basemodel/stream_0_conv_1/BiasAdd_1BiasAdd9model/basemodel/stream_0_conv_1/conv1d_1/Squeeze:output:0@model/basemodel/stream_0_conv_1/BiasAdd_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������� 2+
)model/basemodel/stream_0_conv_1/BiasAdd_1�
>model/basemodel/batch_normalization/batchnorm_1/ReadVariableOpReadVariableOpEmodel_basemodel_batch_normalization_batchnorm_readvariableop_resource*
_output_shapes
: *
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
: 25
3model/basemodel/batch_normalization/batchnorm_1/add�
5model/basemodel/batch_normalization/batchnorm_1/RsqrtRsqrt7model/basemodel/batch_normalization/batchnorm_1/add:z:0*
T0*
_output_shapes
: 27
5model/basemodel/batch_normalization/batchnorm_1/Rsqrt�
Bmodel/basemodel/batch_normalization/batchnorm_1/mul/ReadVariableOpReadVariableOpImodel_basemodel_batch_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02D
Bmodel/basemodel/batch_normalization/batchnorm_1/mul/ReadVariableOp�
3model/basemodel/batch_normalization/batchnorm_1/mulMul9model/basemodel/batch_normalization/batchnorm_1/Rsqrt:y:0Jmodel/basemodel/batch_normalization/batchnorm_1/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 25
3model/basemodel/batch_normalization/batchnorm_1/mul�
5model/basemodel/batch_normalization/batchnorm_1/mul_1Mul2model/basemodel/stream_0_conv_1/BiasAdd_1:output:07model/basemodel/batch_normalization/batchnorm_1/mul:z:0*
T0*,
_output_shapes
:���������� 27
5model/basemodel/batch_normalization/batchnorm_1/mul_1�
@model/basemodel/batch_normalization/batchnorm_1/ReadVariableOp_1ReadVariableOpGmodel_basemodel_batch_normalization_batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype02B
@model/basemodel/batch_normalization/batchnorm_1/ReadVariableOp_1�
5model/basemodel/batch_normalization/batchnorm_1/mul_2MulHmodel/basemodel/batch_normalization/batchnorm_1/ReadVariableOp_1:value:07model/basemodel/batch_normalization/batchnorm_1/mul:z:0*
T0*
_output_shapes
: 27
5model/basemodel/batch_normalization/batchnorm_1/mul_2�
@model/basemodel/batch_normalization/batchnorm_1/ReadVariableOp_2ReadVariableOpGmodel_basemodel_batch_normalization_batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype02B
@model/basemodel/batch_normalization/batchnorm_1/ReadVariableOp_2�
3model/basemodel/batch_normalization/batchnorm_1/subSubHmodel/basemodel/batch_normalization/batchnorm_1/ReadVariableOp_2:value:09model/basemodel/batch_normalization/batchnorm_1/mul_2:z:0*
T0*
_output_shapes
: 25
3model/basemodel/batch_normalization/batchnorm_1/sub�
5model/basemodel/batch_normalization/batchnorm_1/add_1AddV29model/basemodel/batch_normalization/batchnorm_1/mul_1:z:07model/basemodel/batch_normalization/batchnorm_1/sub:z:0*
T0*,
_output_shapes
:���������� 27
5model/basemodel/batch_normalization/batchnorm_1/add_1�
!model/basemodel/activation/Relu_1Relu9model/basemodel/batch_normalization/batchnorm_1/add_1:z:0*
T0*,
_output_shapes
:���������� 2#
!model/basemodel/activation/Relu_1�
*model/basemodel/stream_0_drop_1/Identity_1Identity/model/basemodel/activation/Relu_1:activations:0*
T0*,
_output_shapes
:���������� 2,
*model/basemodel/stream_0_drop_1/Identity_1�
7model/basemodel/stream_0_conv_2/conv1d_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������29
7model/basemodel/stream_0_conv_2/conv1d_1/ExpandDims/dim�
3model/basemodel/stream_0_conv_2/conv1d_1/ExpandDims
ExpandDims3model/basemodel/stream_0_drop_1/Identity_1:output:0@model/basemodel/stream_0_conv_2/conv1d_1/ExpandDims/dim:output:0*
T0*0
_output_shapes
:���������� 25
3model/basemodel/stream_0_conv_2/conv1d_1/ExpandDims�
Dmodel/basemodel/stream_0_conv_2/conv1d_1/ExpandDims_1/ReadVariableOpReadVariableOpKmodel_basemodel_stream_0_conv_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: @*
dtype02F
Dmodel/basemodel/stream_0_conv_2/conv1d_1/ExpandDims_1/ReadVariableOp�
9model/basemodel/stream_0_conv_2/conv1d_1/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2;
9model/basemodel/stream_0_conv_2/conv1d_1/ExpandDims_1/dim�
5model/basemodel/stream_0_conv_2/conv1d_1/ExpandDims_1
ExpandDimsLmodel/basemodel/stream_0_conv_2/conv1d_1/ExpandDims_1/ReadVariableOp:value:0Bmodel/basemodel/stream_0_conv_2/conv1d_1/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: @27
5model/basemodel/stream_0_conv_2/conv1d_1/ExpandDims_1�
(model/basemodel/stream_0_conv_2/conv1d_1Conv2D<model/basemodel/stream_0_conv_2/conv1d_1/ExpandDims:output:0>model/basemodel/stream_0_conv_2/conv1d_1/ExpandDims_1:output:0*
T0*0
_output_shapes
:����������@*
paddingSAME*
strides
2*
(model/basemodel/stream_0_conv_2/conv1d_1�
0model/basemodel/stream_0_conv_2/conv1d_1/SqueezeSqueeze1model/basemodel/stream_0_conv_2/conv1d_1:output:0*
T0*,
_output_shapes
:����������@*
squeeze_dims

���������22
0model/basemodel/stream_0_conv_2/conv1d_1/Squeeze�
8model/basemodel/stream_0_conv_2/BiasAdd_1/ReadVariableOpReadVariableOp?model_basemodel_stream_0_conv_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02:
8model/basemodel/stream_0_conv_2/BiasAdd_1/ReadVariableOp�
)model/basemodel/stream_0_conv_2/BiasAdd_1BiasAdd9model/basemodel/stream_0_conv_2/conv1d_1/Squeeze:output:0@model/basemodel/stream_0_conv_2/BiasAdd_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������@2+
)model/basemodel/stream_0_conv_2/BiasAdd_1�
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
7model/basemodel/batch_normalization_1/batchnorm_1/mul_1Mul2model/basemodel/stream_0_conv_2/BiasAdd_1:output:09model/basemodel/batch_normalization_1/batchnorm_1/mul:z:0*
T0*,
_output_shapes
:����������@29
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
:����������@29
7model/basemodel/batch_normalization_1/batchnorm_1/add_1�
#model/basemodel/activation_1/Relu_1Relu;model/basemodel/batch_normalization_1/batchnorm_1/add_1:z:0*
T0*,
_output_shapes
:����������@2%
#model/basemodel/activation_1/Relu_1�
*model/basemodel/stream_0_drop_2/Identity_1Identity1model/basemodel/activation_1/Relu_1:activations:0*
T0*,
_output_shapes
:����������@2,
*model/basemodel/stream_0_drop_2/Identity_1�
Amodel/basemodel/global_average_pooling1d/Mean_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2C
Amodel/basemodel/global_average_pooling1d/Mean_1/reduction_indices�
/model/basemodel/global_average_pooling1d/Mean_1Mean3model/basemodel/stream_0_drop_2/Identity_1:output:0Jmodel/basemodel/global_average_pooling1d/Mean_1/reduction_indices:output:0*
T0*'
_output_shapes
:���������@21
/model/basemodel/global_average_pooling1d/Mean_1�
/model/basemodel/concatenate/concat_1/concat_dimConst*
_output_shapes
: *
dtype0*
value	B :21
/model/basemodel/concatenate/concat_1/concat_dim�
+model/basemodel/concatenate/concat_1/concatIdentity8model/basemodel/global_average_pooling1d/Mean_1:output:0*
T0*'
_output_shapes
:���������@2-
+model/basemodel/concatenate/concat_1/concat�
*model/basemodel/dense_1_dropout/Identity_1Identity4model/basemodel/concatenate/concat_1/concat:output:0*
T0*'
_output_shapes
:���������@2,
*model/basemodel/dense_1_dropout/Identity_1�
/model/basemodel/dense_1/MatMul_1/ReadVariableOpReadVariableOp6model_basemodel_dense_1_matmul_readvariableop_resource*
_output_shapes

:@T*
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
@model/basemodel/batch_normalization_2/batchnorm_1/ReadVariableOpReadVariableOpGmodel_basemodel_batch_normalization_2_batchnorm_readvariableop_resource*
_output_shapes
:T*
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
:T27
5model/basemodel/batch_normalization_2/batchnorm_1/add�
7model/basemodel/batch_normalization_2/batchnorm_1/RsqrtRsqrt9model/basemodel/batch_normalization_2/batchnorm_1/add:z:0*
T0*
_output_shapes
:T29
7model/basemodel/batch_normalization_2/batchnorm_1/Rsqrt�
Dmodel/basemodel/batch_normalization_2/batchnorm_1/mul/ReadVariableOpReadVariableOpKmodel_basemodel_batch_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes
:T*
dtype02F
Dmodel/basemodel/batch_normalization_2/batchnorm_1/mul/ReadVariableOp�
5model/basemodel/batch_normalization_2/batchnorm_1/mulMul;model/basemodel/batch_normalization_2/batchnorm_1/Rsqrt:y:0Lmodel/basemodel/batch_normalization_2/batchnorm_1/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:T27
5model/basemodel/batch_normalization_2/batchnorm_1/mul�
7model/basemodel/batch_normalization_2/batchnorm_1/mul_1Mul*model/basemodel/dense_1/BiasAdd_1:output:09model/basemodel/batch_normalization_2/batchnorm_1/mul:z:0*
T0*'
_output_shapes
:���������T29
7model/basemodel/batch_normalization_2/batchnorm_1/mul_1�
Bmodel/basemodel/batch_normalization_2/batchnorm_1/ReadVariableOp_1ReadVariableOpImodel_basemodel_batch_normalization_2_batchnorm_readvariableop_1_resource*
_output_shapes
:T*
dtype02D
Bmodel/basemodel/batch_normalization_2/batchnorm_1/ReadVariableOp_1�
7model/basemodel/batch_normalization_2/batchnorm_1/mul_2MulJmodel/basemodel/batch_normalization_2/batchnorm_1/ReadVariableOp_1:value:09model/basemodel/batch_normalization_2/batchnorm_1/mul:z:0*
T0*
_output_shapes
:T29
7model/basemodel/batch_normalization_2/batchnorm_1/mul_2�
Bmodel/basemodel/batch_normalization_2/batchnorm_1/ReadVariableOp_2ReadVariableOpImodel_basemodel_batch_normalization_2_batchnorm_readvariableop_2_resource*
_output_shapes
:T*
dtype02D
Bmodel/basemodel/batch_normalization_2/batchnorm_1/ReadVariableOp_2�
5model/basemodel/batch_normalization_2/batchnorm_1/subSubJmodel/basemodel/batch_normalization_2/batchnorm_1/ReadVariableOp_2:value:0;model/basemodel/batch_normalization_2/batchnorm_1/mul_2:z:0*
T0*
_output_shapes
:T27
5model/basemodel/batch_normalization_2/batchnorm_1/sub�
7model/basemodel/batch_normalization_2/batchnorm_1/add_1AddV2;model/basemodel/batch_normalization_2/batchnorm_1/mul_1:z:09model/basemodel/batch_normalization_2/batchnorm_1/sub:z:0*
T0*'
_output_shapes
:���������T29
7model/basemodel/batch_normalization_2/batchnorm_1/add_1�
,model/basemodel/dense_activation_1/Sigmoid_1Sigmoid;model/basemodel/batch_normalization_2/batchnorm_1/add_1:z:0*
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

Identity�
NoOpNoOp=^model/basemodel/batch_normalization/batchnorm/ReadVariableOp?^model/basemodel/batch_normalization/batchnorm/ReadVariableOp_1?^model/basemodel/batch_normalization/batchnorm/ReadVariableOp_2A^model/basemodel/batch_normalization/batchnorm/mul/ReadVariableOp?^model/basemodel/batch_normalization/batchnorm_1/ReadVariableOpA^model/basemodel/batch_normalization/batchnorm_1/ReadVariableOp_1A^model/basemodel/batch_normalization/batchnorm_1/ReadVariableOp_2C^model/basemodel/batch_normalization/batchnorm_1/mul/ReadVariableOp?^model/basemodel/batch_normalization_1/batchnorm/ReadVariableOpA^model/basemodel/batch_normalization_1/batchnorm/ReadVariableOp_1A^model/basemodel/batch_normalization_1/batchnorm/ReadVariableOp_2C^model/basemodel/batch_normalization_1/batchnorm/mul/ReadVariableOpA^model/basemodel/batch_normalization_1/batchnorm_1/ReadVariableOpC^model/basemodel/batch_normalization_1/batchnorm_1/ReadVariableOp_1C^model/basemodel/batch_normalization_1/batchnorm_1/ReadVariableOp_2E^model/basemodel/batch_normalization_1/batchnorm_1/mul/ReadVariableOp?^model/basemodel/batch_normalization_2/batchnorm/ReadVariableOpA^model/basemodel/batch_normalization_2/batchnorm/ReadVariableOp_1A^model/basemodel/batch_normalization_2/batchnorm/ReadVariableOp_2C^model/basemodel/batch_normalization_2/batchnorm/mul/ReadVariableOpA^model/basemodel/batch_normalization_2/batchnorm_1/ReadVariableOpC^model/basemodel/batch_normalization_2/batchnorm_1/ReadVariableOp_1C^model/basemodel/batch_normalization_2/batchnorm_1/ReadVariableOp_2E^model/basemodel/batch_normalization_2/batchnorm_1/mul/ReadVariableOp/^model/basemodel/dense_1/BiasAdd/ReadVariableOp1^model/basemodel/dense_1/BiasAdd_1/ReadVariableOp.^model/basemodel/dense_1/MatMul/ReadVariableOp0^model/basemodel/dense_1/MatMul_1/ReadVariableOp7^model/basemodel/stream_0_conv_1/BiasAdd/ReadVariableOp9^model/basemodel/stream_0_conv_1/BiasAdd_1/ReadVariableOpC^model/basemodel/stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOpE^model/basemodel/stream_0_conv_1/conv1d_1/ExpandDims_1/ReadVariableOp7^model/basemodel/stream_0_conv_2/BiasAdd/ReadVariableOp9^model/basemodel/stream_0_conv_2/BiasAdd_1/ReadVariableOpC^model/basemodel/stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOpE^model/basemodel/stream_0_conv_2/conv1d_1/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*g
_input_shapesV
T:����������:����������: : : : : : : : : : : : : : : : : : 2|
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
Dmodel/basemodel/batch_normalization_2/batchnorm_1/mul/ReadVariableOpDmodel/basemodel/batch_normalization_2/batchnorm_1/mul/ReadVariableOp2`
.model/basemodel/dense_1/BiasAdd/ReadVariableOp.model/basemodel/dense_1/BiasAdd/ReadVariableOp2d
0model/basemodel/dense_1/BiasAdd_1/ReadVariableOp0model/basemodel/dense_1/BiasAdd_1/ReadVariableOp2^
-model/basemodel/dense_1/MatMul/ReadVariableOp-model/basemodel/dense_1/MatMul/ReadVariableOp2b
/model/basemodel/dense_1/MatMul_1/ReadVariableOp/model/basemodel/dense_1/MatMul_1/ReadVariableOp2p
6model/basemodel/stream_0_conv_1/BiasAdd/ReadVariableOp6model/basemodel/stream_0_conv_1/BiasAdd/ReadVariableOp2t
8model/basemodel/stream_0_conv_1/BiasAdd_1/ReadVariableOp8model/basemodel/stream_0_conv_1/BiasAdd_1/ReadVariableOp2�
Bmodel/basemodel/stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOpBmodel/basemodel/stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp2�
Dmodel/basemodel/stream_0_conv_1/conv1d_1/ExpandDims_1/ReadVariableOpDmodel/basemodel/stream_0_conv_1/conv1d_1/ExpandDims_1/ReadVariableOp2p
6model/basemodel/stream_0_conv_2/BiasAdd/ReadVariableOp6model/basemodel/stream_0_conv_2/BiasAdd/ReadVariableOp2t
8model/basemodel/stream_0_conv_2/BiasAdd_1/ReadVariableOp8model/basemodel/stream_0_conv_2/BiasAdd_1/ReadVariableOp2�
Bmodel/basemodel/stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOpBmodel/basemodel/stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOp2�
Dmodel/basemodel/stream_0_conv_2/conv1d_1/ExpandDims_1/ReadVariableOpDmodel/basemodel/stream_0_conv_2/conv1d_1/ExpandDims_1/ReadVariableOp:Y U
,
_output_shapes
:����������
%
_user_specified_nameleft_inputs:ZV
,
_output_shapes
:����������
&
_user_specified_nameright_inputs
�;
�
B__inference_model_layer_call_and_return_conditional_losses_5542986
left_inputs
right_inputs'
basemodel_5542910: 
basemodel_5542912: 
basemodel_5542914: 
basemodel_5542916: 
basemodel_5542918: 
basemodel_5542920: '
basemodel_5542922: @
basemodel_5542924:@
basemodel_5542926:@
basemodel_5542928:@
basemodel_5542930:@
basemodel_5542932:@#
basemodel_5542934:@T
basemodel_5542936:T
basemodel_5542938:T
basemodel_5542940:T
basemodel_5542942:T
basemodel_5542944:T
identity��!basemodel/StatefulPartitionedCall�#basemodel/StatefulPartitionedCall_1�-dense_1/kernel/Regularizer/Abs/ReadVariableOp�5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp�8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp�
!basemodel/StatefulPartitionedCallStatefulPartitionedCallleft_inputsbasemodel_5542910basemodel_5542912basemodel_5542914basemodel_5542916basemodel_5542918basemodel_5542920basemodel_5542922basemodel_5542924basemodel_5542926basemodel_5542928basemodel_5542930basemodel_5542932basemodel_5542934basemodel_5542936basemodel_5542938basemodel_5542940basemodel_5542942basemodel_5542944*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������T*4
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_basemodel_layer_call_and_return_conditional_losses_55422852#
!basemodel/StatefulPartitionedCall�
#basemodel/StatefulPartitionedCall_1StatefulPartitionedCallright_inputsbasemodel_5542910basemodel_5542912basemodel_5542914basemodel_5542916basemodel_5542918basemodel_5542920basemodel_5542922basemodel_5542924basemodel_5542926basemodel_5542928basemodel_5542930basemodel_5542932basemodel_5542934basemodel_5542936basemodel_5542938basemodel_5542940basemodel_5542942basemodel_5542944*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������T*4
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_basemodel_layer_call_and_return_conditional_losses_55422852%
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
E__inference_distance_layer_call_and_return_conditional_losses_55423542
distance/PartitionedCall�
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpbasemodel_5542910*"
_output_shapes
: *
dtype027
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp�
&stream_0_conv_1/kernel/Regularizer/AbsAbs=stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
: 2(
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
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpbasemodel_5542922*"
_output_shapes
: @*
dtype02:
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp�
)stream_0_conv_2/kernel/Regularizer/SquareSquare@stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
: @2+
)stream_0_conv_2/kernel/Regularizer/Square�
(stream_0_conv_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_0_conv_2/kernel/Regularizer/Const�
&stream_0_conv_2/kernel/Regularizer/SumSum-stream_0_conv_2/kernel/Regularizer/Square:y:01stream_0_conv_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_2/kernel/Regularizer/Sum�
(stream_0_conv_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2*
(stream_0_conv_2/kernel/Regularizer/mul/x�
&stream_0_conv_2/kernel/Regularizer/mulMul1stream_0_conv_2/kernel/Regularizer/mul/x:output:0/stream_0_conv_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_2/kernel/Regularizer/mul�
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpbasemodel_5542934*
_output_shapes

:@T*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOp�
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:@T2 
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
NoOpNoOp"^basemodel/StatefulPartitionedCall$^basemodel/StatefulPartitionedCall_1.^dense_1/kernel/Regularizer/Abs/ReadVariableOp6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp9^stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*g
_input_shapesV
T:����������:����������: : : : : : : : : : : : : : : : : : 2F
!basemodel/StatefulPartitionedCall!basemodel/StatefulPartitionedCall2J
#basemodel/StatefulPartitionedCall_1#basemodel/StatefulPartitionedCall_12^
-dense_1/kernel/Regularizer/Abs/ReadVariableOp-dense_1/kernel/Regularizer/Abs/ReadVariableOp2n
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp2t
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp:Y U
,
_output_shapes
:����������
%
_user_specified_nameleft_inputs:ZV
,
_output_shapes
:����������
&
_user_specified_nameright_inputs
�
�
7__inference_batch_normalization_2_layer_call_fn_5545188

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
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_55412052
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
�;
�
B__inference_model_layer_call_and_return_conditional_losses_5543066
left_inputs
right_inputs'
basemodel_5542990: 
basemodel_5542992: 
basemodel_5542994: 
basemodel_5542996: 
basemodel_5542998: 
basemodel_5543000: '
basemodel_5543002: @
basemodel_5543004:@
basemodel_5543006:@
basemodel_5543008:@
basemodel_5543010:@
basemodel_5543012:@#
basemodel_5543014:@T
basemodel_5543016:T
basemodel_5543018:T
basemodel_5543020:T
basemodel_5543022:T
basemodel_5543024:T
identity��!basemodel/StatefulPartitionedCall�#basemodel/StatefulPartitionedCall_1�-dense_1/kernel/Regularizer/Abs/ReadVariableOp�5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp�8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp�
!basemodel/StatefulPartitionedCallStatefulPartitionedCallleft_inputsbasemodel_5542990basemodel_5542992basemodel_5542994basemodel_5542996basemodel_5542998basemodel_5543000basemodel_5543002basemodel_5543004basemodel_5543006basemodel_5543008basemodel_5543010basemodel_5543012basemodel_5543014basemodel_5543016basemodel_5543018basemodel_5543020basemodel_5543022basemodel_5543024*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������T*.
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_basemodel_layer_call_and_return_conditional_losses_55426612#
!basemodel/StatefulPartitionedCall�
#basemodel/StatefulPartitionedCall_1StatefulPartitionedCallright_inputsbasemodel_5542990basemodel_5542992basemodel_5542994basemodel_5542996basemodel_5542998basemodel_5543000basemodel_5543002basemodel_5543004basemodel_5543006basemodel_5543008basemodel_5543010basemodel_5543012basemodel_5543014basemodel_5543016basemodel_5543018basemodel_5543020basemodel_5543022basemodel_5543024"^basemodel/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������T*.
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_basemodel_layer_call_and_return_conditional_losses_55426612%
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
E__inference_distance_layer_call_and_return_conditional_losses_55424362
distance/PartitionedCall�
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpbasemodel_5542990*"
_output_shapes
: *
dtype027
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp�
&stream_0_conv_1/kernel/Regularizer/AbsAbs=stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
: 2(
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
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpbasemodel_5543002*"
_output_shapes
: @*
dtype02:
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp�
)stream_0_conv_2/kernel/Regularizer/SquareSquare@stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
: @2+
)stream_0_conv_2/kernel/Regularizer/Square�
(stream_0_conv_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_0_conv_2/kernel/Regularizer/Const�
&stream_0_conv_2/kernel/Regularizer/SumSum-stream_0_conv_2/kernel/Regularizer/Square:y:01stream_0_conv_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_2/kernel/Regularizer/Sum�
(stream_0_conv_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2*
(stream_0_conv_2/kernel/Regularizer/mul/x�
&stream_0_conv_2/kernel/Regularizer/mulMul1stream_0_conv_2/kernel/Regularizer/mul/x:output:0/stream_0_conv_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_2/kernel/Regularizer/mul�
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpbasemodel_5543014*
_output_shapes

:@T*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOp�
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:@T2 
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
NoOpNoOp"^basemodel/StatefulPartitionedCall$^basemodel/StatefulPartitionedCall_1.^dense_1/kernel/Regularizer/Abs/ReadVariableOp6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp9^stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*g
_input_shapesV
T:����������:����������: : : : : : : : : : : : : : : : : : 2F
!basemodel/StatefulPartitionedCall!basemodel/StatefulPartitionedCall2J
#basemodel/StatefulPartitionedCall_1#basemodel/StatefulPartitionedCall_12^
-dense_1/kernel/Regularizer/Abs/ReadVariableOp-dense_1/kernel/Regularizer/Abs/ReadVariableOp2n
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp2t
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp:Y U
,
_output_shapes
:����������
%
_user_specified_nameleft_inputs:ZV
,
_output_shapes
:����������
&
_user_specified_nameright_inputs
�	
�
5__inference_batch_normalization_layer_call_fn_5544708

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������ *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_55407972
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :������������������ 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:������������������ : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :������������������ 
 
_user_specified_nameinputs
�g
�

F__inference_basemodel_layer_call_and_return_conditional_losses_5541516

inputs-
stream_0_conv_1_5541319: %
stream_0_conv_1_5541321: )
batch_normalization_5541344: )
batch_normalization_5541346: )
batch_normalization_5541348: )
batch_normalization_5541350: -
stream_0_conv_2_5541389: @%
stream_0_conv_2_5541391:@+
batch_normalization_1_5541414:@+
batch_normalization_1_5541416:@+
batch_normalization_1_5541418:@+
batch_normalization_1_5541420:@!
dense_1_5541476:@T
dense_1_5541478:T+
batch_normalization_2_5541481:T+
batch_normalization_2_5541483:T+
batch_normalization_2_5541485:T+
batch_normalization_2_5541487:T
identity��+batch_normalization/StatefulPartitionedCall�-batch_normalization_1/StatefulPartitionedCall�-batch_normalization_2/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�-dense_1/kernel/Regularizer/Abs/ReadVariableOp�'stream_0_conv_1/StatefulPartitionedCall�5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp�'stream_0_conv_2/StatefulPartitionedCall�8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp�
#stream_0_input_drop/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *Y
fTRR
P__inference_stream_0_input_drop_layer_call_and_return_conditional_losses_55412952%
#stream_0_input_drop/PartitionedCall�
'stream_0_conv_1/StatefulPartitionedCallStatefulPartitionedCall,stream_0_input_drop/PartitionedCall:output:0stream_0_conv_1_5541319stream_0_conv_1_5541321*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_stream_0_conv_1_layer_call_and_return_conditional_losses_55413182)
'stream_0_conv_1/StatefulPartitionedCall�
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall0stream_0_conv_1/StatefulPartitionedCall:output:0batch_normalization_5541344batch_normalization_5541346batch_normalization_5541348batch_normalization_5541350*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������� *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_55413432-
+batch_normalization/StatefulPartitionedCall�
activation/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������� * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_activation_layer_call_and_return_conditional_losses_55413582
activation/PartitionedCall�
stream_0_drop_1/PartitionedCallPartitionedCall#activation/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������� * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_stream_0_drop_1_layer_call_and_return_conditional_losses_55413652!
stream_0_drop_1/PartitionedCall�
'stream_0_conv_2/StatefulPartitionedCallStatefulPartitionedCall(stream_0_drop_1/PartitionedCall:output:0stream_0_conv_2_5541389stream_0_conv_2_5541391*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_stream_0_conv_2_layer_call_and_return_conditional_losses_55413882)
'stream_0_conv_2/StatefulPartitionedCall�
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall0stream_0_conv_2/StatefulPartitionedCall:output:0batch_normalization_1_5541414batch_normalization_1_5541416batch_normalization_1_5541418batch_normalization_1_5541420*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *[
fVRT
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_55414132/
-batch_normalization_1/StatefulPartitionedCall�
activation_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_activation_1_layer_call_and_return_conditional_losses_55414282
activation_1/PartitionedCall�
stream_0_drop_2/PartitionedCallPartitionedCall%activation_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_stream_0_drop_2_layer_call_and_return_conditional_losses_55414352!
stream_0_drop_2/PartitionedCall�
(global_average_pooling1d/PartitionedCallPartitionedCall(stream_0_drop_2/PartitionedCall:output:0*
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
U__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_55414422*
(global_average_pooling1d/PartitionedCall�
concatenate/PartitionedCallPartitionedCall1global_average_pooling1d/PartitionedCall:output:0*
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
GPU2*0J 8� *Q
fLRJ
H__inference_concatenate_layer_call_and_return_conditional_losses_55414502
concatenate/PartitionedCall�
dense_1_dropout/PartitionedCallPartitionedCall$concatenate/PartitionedCall:output:0*
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
GPU2*0J 8� *U
fPRN
L__inference_dense_1_dropout_layer_call_and_return_conditional_losses_55414572!
dense_1_dropout/PartitionedCall�
dense_1/StatefulPartitionedCallStatefulPartitionedCall(dense_1_dropout/PartitionedCall:output:0dense_1_5541476dense_1_5541478*
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
D__inference_dense_1_layer_call_and_return_conditional_losses_55414752!
dense_1/StatefulPartitionedCall�
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0batch_normalization_2_5541481batch_normalization_2_5541483batch_normalization_2_5541485batch_normalization_2_5541487*
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
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_55411452/
-batch_normalization_2/StatefulPartitionedCall�
"dense_activation_1/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
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
O__inference_dense_activation_1_layer_call_and_return_conditional_losses_55414952$
"dense_activation_1/PartitionedCall�
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpstream_0_conv_1_5541319*"
_output_shapes
: *
dtype027
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp�
&stream_0_conv_1/kernel/Regularizer/AbsAbs=stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
: 2(
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
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpstream_0_conv_2_5541389*"
_output_shapes
: @*
dtype02:
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp�
)stream_0_conv_2/kernel/Regularizer/SquareSquare@stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
: @2+
)stream_0_conv_2/kernel/Regularizer/Square�
(stream_0_conv_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_0_conv_2/kernel/Regularizer/Const�
&stream_0_conv_2/kernel/Regularizer/SumSum-stream_0_conv_2/kernel/Regularizer/Square:y:01stream_0_conv_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_2/kernel/Regularizer/Sum�
(stream_0_conv_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2*
(stream_0_conv_2/kernel/Regularizer/mul/x�
&stream_0_conv_2/kernel/Regularizer/mulMul1stream_0_conv_2/kernel/Regularizer/mul/x:output:0/stream_0_conv_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_2/kernel/Regularizer/mul�
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_1_5541476*
_output_shapes

:@T*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOp�
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:@T2 
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

Identity�
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall.^dense_1/kernel/Regularizer/Abs/ReadVariableOp(^stream_0_conv_1/StatefulPartitionedCall6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp(^stream_0_conv_2/StatefulPartitionedCall9^stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:����������: : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2^
-dense_1/kernel/Regularizer/Abs/ReadVariableOp-dense_1/kernel/Regularizer/Abs/ReadVariableOp2R
'stream_0_conv_1/StatefulPartitionedCall'stream_0_conv_1/StatefulPartitionedCall2n
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp2R
'stream_0_conv_2/StatefulPartitionedCall'stream_0_conv_2/StatefulPartitionedCall2t
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
__inference_loss_fn_1_5545220W
Astream_0_conv_2_kernel_regularizer_square_readvariableop_resource: @
identity��8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp�
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpAstream_0_conv_2_kernel_regularizer_square_readvariableop_resource*"
_output_shapes
: @*
dtype02:
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp�
)stream_0_conv_2/kernel/Regularizer/SquareSquare@stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
: @2+
)stream_0_conv_2/kernel/Regularizer/Square�
(stream_0_conv_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_0_conv_2/kernel/Regularizer/Const�
&stream_0_conv_2/kernel/Regularizer/SumSum-stream_0_conv_2/kernel/Regularizer/Square:y:01stream_0_conv_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_2/kernel/Regularizer/Sum�
(stream_0_conv_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2*
(stream_0_conv_2/kernel/Regularizer/mul/x�
&stream_0_conv_2/kernel/Regularizer/mulMul1stream_0_conv_2/kernel/Regularizer/mul/x:output:0/stream_0_conv_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_2/kernel/Regularizer/mult
IdentityIdentity*stream_0_conv_2/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: 2

Identity�
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
�
V
:__inference_global_average_pooling1d_layer_call_fn_5545039

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
U__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_55414422
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������@:T P
,
_output_shapes
:����������@
 
_user_specified_nameinputs
��
�#
B__inference_model_layer_call_and_return_conditional_losses_5543646
inputs_0
inputs_1[
Ebasemodel_stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource: G
9basemodel_stream_0_conv_1_biasadd_readvariableop_resource: S
Ebasemodel_batch_normalization_assignmovingavg_readvariableop_resource: U
Gbasemodel_batch_normalization_assignmovingavg_1_readvariableop_resource: Q
Cbasemodel_batch_normalization_batchnorm_mul_readvariableop_resource: M
?basemodel_batch_normalization_batchnorm_readvariableop_resource: [
Ebasemodel_stream_0_conv_2_conv1d_expanddims_1_readvariableop_resource: @G
9basemodel_stream_0_conv_2_biasadd_readvariableop_resource:@U
Gbasemodel_batch_normalization_1_assignmovingavg_readvariableop_resource:@W
Ibasemodel_batch_normalization_1_assignmovingavg_1_readvariableop_resource:@S
Ebasemodel_batch_normalization_1_batchnorm_mul_readvariableop_resource:@O
Abasemodel_batch_normalization_1_batchnorm_readvariableop_resource:@B
0basemodel_dense_1_matmul_readvariableop_resource:@T?
1basemodel_dense_1_biasadd_readvariableop_resource:TU
Gbasemodel_batch_normalization_2_assignmovingavg_readvariableop_resource:TW
Ibasemodel_batch_normalization_2_assignmovingavg_1_readvariableop_resource:TS
Ebasemodel_batch_normalization_2_batchnorm_mul_readvariableop_resource:TO
Abasemodel_batch_normalization_2_batchnorm_readvariableop_resource:T
identity��-basemodel/batch_normalization/AssignMovingAvg�<basemodel/batch_normalization/AssignMovingAvg/ReadVariableOp�/basemodel/batch_normalization/AssignMovingAvg_1�>basemodel/batch_normalization/AssignMovingAvg_1/ReadVariableOp�/basemodel/batch_normalization/AssignMovingAvg_2�>basemodel/batch_normalization/AssignMovingAvg_2/ReadVariableOp�/basemodel/batch_normalization/AssignMovingAvg_3�>basemodel/batch_normalization/AssignMovingAvg_3/ReadVariableOp�6basemodel/batch_normalization/batchnorm/ReadVariableOp�:basemodel/batch_normalization/batchnorm/mul/ReadVariableOp�8basemodel/batch_normalization/batchnorm_1/ReadVariableOp�<basemodel/batch_normalization/batchnorm_1/mul/ReadVariableOp�/basemodel/batch_normalization_1/AssignMovingAvg�>basemodel/batch_normalization_1/AssignMovingAvg/ReadVariableOp�1basemodel/batch_normalization_1/AssignMovingAvg_1�@basemodel/batch_normalization_1/AssignMovingAvg_1/ReadVariableOp�1basemodel/batch_normalization_1/AssignMovingAvg_2�@basemodel/batch_normalization_1/AssignMovingAvg_2/ReadVariableOp�1basemodel/batch_normalization_1/AssignMovingAvg_3�@basemodel/batch_normalization_1/AssignMovingAvg_3/ReadVariableOp�8basemodel/batch_normalization_1/batchnorm/ReadVariableOp�<basemodel/batch_normalization_1/batchnorm/mul/ReadVariableOp�:basemodel/batch_normalization_1/batchnorm_1/ReadVariableOp�>basemodel/batch_normalization_1/batchnorm_1/mul/ReadVariableOp�/basemodel/batch_normalization_2/AssignMovingAvg�>basemodel/batch_normalization_2/AssignMovingAvg/ReadVariableOp�1basemodel/batch_normalization_2/AssignMovingAvg_1�@basemodel/batch_normalization_2/AssignMovingAvg_1/ReadVariableOp�1basemodel/batch_normalization_2/AssignMovingAvg_2�@basemodel/batch_normalization_2/AssignMovingAvg_2/ReadVariableOp�1basemodel/batch_normalization_2/AssignMovingAvg_3�@basemodel/batch_normalization_2/AssignMovingAvg_3/ReadVariableOp�8basemodel/batch_normalization_2/batchnorm/ReadVariableOp�<basemodel/batch_normalization_2/batchnorm/mul/ReadVariableOp�:basemodel/batch_normalization_2/batchnorm_1/ReadVariableOp�>basemodel/batch_normalization_2/batchnorm_1/mul/ReadVariableOp�(basemodel/dense_1/BiasAdd/ReadVariableOp�*basemodel/dense_1/BiasAdd_1/ReadVariableOp�'basemodel/dense_1/MatMul/ReadVariableOp�)basemodel/dense_1/MatMul_1/ReadVariableOp�0basemodel/stream_0_conv_1/BiasAdd/ReadVariableOp�2basemodel/stream_0_conv_1/BiasAdd_1/ReadVariableOp�<basemodel/stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp�>basemodel/stream_0_conv_1/conv1d_1/ExpandDims_1/ReadVariableOp�0basemodel/stream_0_conv_2/BiasAdd/ReadVariableOp�2basemodel/stream_0_conv_2/BiasAdd_1/ReadVariableOp�<basemodel/stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOp�>basemodel/stream_0_conv_2/conv1d_1/ExpandDims_1/ReadVariableOp�-dense_1/kernel/Regularizer/Abs/ReadVariableOp�5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp�8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp�
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
:����������2+
)basemodel/stream_0_input_drop/dropout/Mul�
+basemodel/stream_0_input_drop/dropout/ShapeShapeinputs_0*
T0*
_output_shapes
:2-
+basemodel/stream_0_input_drop/dropout/Shape�
Bbasemodel/stream_0_input_drop/dropout/random_uniform/RandomUniformRandomUniform4basemodel/stream_0_input_drop/dropout/Shape:output:0*
T0*,
_output_shapes
:����������*
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
:����������24
2basemodel/stream_0_input_drop/dropout/GreaterEqual�
*basemodel/stream_0_input_drop/dropout/CastCast6basemodel/stream_0_input_drop/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:����������2,
*basemodel/stream_0_input_drop/dropout/Cast�
+basemodel/stream_0_input_drop/dropout/Mul_1Mul-basemodel/stream_0_input_drop/dropout/Mul:z:0.basemodel/stream_0_input_drop/dropout/Cast:y:0*
T0*,
_output_shapes
:����������2-
+basemodel/stream_0_input_drop/dropout/Mul_1�
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
:����������2-
+basemodel/stream_0_conv_1/conv1d/ExpandDims�
<basemodel/stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpEbasemodel_stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
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
: 2/
-basemodel/stream_0_conv_1/conv1d/ExpandDims_1�
 basemodel/stream_0_conv_1/conv1dConv2D4basemodel/stream_0_conv_1/conv1d/ExpandDims:output:06basemodel/stream_0_conv_1/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:���������� *
paddingSAME*
strides
2"
 basemodel/stream_0_conv_1/conv1d�
(basemodel/stream_0_conv_1/conv1d/SqueezeSqueeze)basemodel/stream_0_conv_1/conv1d:output:0*
T0*,
_output_shapes
:���������� *
squeeze_dims

���������2*
(basemodel/stream_0_conv_1/conv1d/Squeeze�
0basemodel/stream_0_conv_1/BiasAdd/ReadVariableOpReadVariableOp9basemodel_stream_0_conv_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype022
0basemodel/stream_0_conv_1/BiasAdd/ReadVariableOp�
!basemodel/stream_0_conv_1/BiasAddBiasAdd1basemodel/stream_0_conv_1/conv1d/Squeeze:output:08basemodel/stream_0_conv_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������� 2#
!basemodel/stream_0_conv_1/BiasAdd�
<basemodel/batch_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2>
<basemodel/batch_normalization/moments/mean/reduction_indices�
*basemodel/batch_normalization/moments/meanMean*basemodel/stream_0_conv_1/BiasAdd:output:0Ebasemodel/batch_normalization/moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(2,
*basemodel/batch_normalization/moments/mean�
2basemodel/batch_normalization/moments/StopGradientStopGradient3basemodel/batch_normalization/moments/mean:output:0*
T0*"
_output_shapes
: 24
2basemodel/batch_normalization/moments/StopGradient�
7basemodel/batch_normalization/moments/SquaredDifferenceSquaredDifference*basemodel/stream_0_conv_1/BiasAdd:output:0;basemodel/batch_normalization/moments/StopGradient:output:0*
T0*,
_output_shapes
:���������� 29
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
: *
	keep_dims(20
.basemodel/batch_normalization/moments/variance�
-basemodel/batch_normalization/moments/SqueezeSqueeze3basemodel/batch_normalization/moments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2/
-basemodel/batch_normalization/moments/Squeeze�
/basemodel/batch_normalization/moments/Squeeze_1Squeeze7basemodel/batch_normalization/moments/variance:output:0*
T0*
_output_shapes
: *
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
: *
dtype02>
<basemodel/batch_normalization/AssignMovingAvg/ReadVariableOp�
1basemodel/batch_normalization/AssignMovingAvg/subSubDbasemodel/batch_normalization/AssignMovingAvg/ReadVariableOp:value:06basemodel/batch_normalization/moments/Squeeze:output:0*
T0*
_output_shapes
: 23
1basemodel/batch_normalization/AssignMovingAvg/sub�
1basemodel/batch_normalization/AssignMovingAvg/mulMul5basemodel/batch_normalization/AssignMovingAvg/sub:z:0<basemodel/batch_normalization/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
: 23
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
: *
dtype02@
>basemodel/batch_normalization/AssignMovingAvg_1/ReadVariableOp�
3basemodel/batch_normalization/AssignMovingAvg_1/subSubFbasemodel/batch_normalization/AssignMovingAvg_1/ReadVariableOp:value:08basemodel/batch_normalization/moments/Squeeze_1:output:0*
T0*
_output_shapes
: 25
3basemodel/batch_normalization/AssignMovingAvg_1/sub�
3basemodel/batch_normalization/AssignMovingAvg_1/mulMul7basemodel/batch_normalization/AssignMovingAvg_1/sub:z:0>basemodel/batch_normalization/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
: 25
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
: 2-
+basemodel/batch_normalization/batchnorm/add�
-basemodel/batch_normalization/batchnorm/RsqrtRsqrt/basemodel/batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes
: 2/
-basemodel/batch_normalization/batchnorm/Rsqrt�
:basemodel/batch_normalization/batchnorm/mul/ReadVariableOpReadVariableOpCbasemodel_batch_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02<
:basemodel/batch_normalization/batchnorm/mul/ReadVariableOp�
+basemodel/batch_normalization/batchnorm/mulMul1basemodel/batch_normalization/batchnorm/Rsqrt:y:0Bbasemodel/batch_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2-
+basemodel/batch_normalization/batchnorm/mul�
-basemodel/batch_normalization/batchnorm/mul_1Mul*basemodel/stream_0_conv_1/BiasAdd:output:0/basemodel/batch_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:���������� 2/
-basemodel/batch_normalization/batchnorm/mul_1�
-basemodel/batch_normalization/batchnorm/mul_2Mul6basemodel/batch_normalization/moments/Squeeze:output:0/basemodel/batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes
: 2/
-basemodel/batch_normalization/batchnorm/mul_2�
6basemodel/batch_normalization/batchnorm/ReadVariableOpReadVariableOp?basemodel_batch_normalization_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype028
6basemodel/batch_normalization/batchnorm/ReadVariableOp�
+basemodel/batch_normalization/batchnorm/subSub>basemodel/batch_normalization/batchnorm/ReadVariableOp:value:01basemodel/batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2-
+basemodel/batch_normalization/batchnorm/sub�
-basemodel/batch_normalization/batchnorm/add_1AddV21basemodel/batch_normalization/batchnorm/mul_1:z:0/basemodel/batch_normalization/batchnorm/sub:z:0*
T0*,
_output_shapes
:���������� 2/
-basemodel/batch_normalization/batchnorm/add_1�
basemodel/activation/ReluRelu1basemodel/batch_normalization/batchnorm/add_1:z:0*
T0*,
_output_shapes
:���������� 2
basemodel/activation/Relu�
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
:���������� 2'
%basemodel/stream_0_drop_1/dropout/Mul�
'basemodel/stream_0_drop_1/dropout/ShapeShape'basemodel/activation/Relu:activations:0*
T0*
_output_shapes
:2)
'basemodel/stream_0_drop_1/dropout/Shape�
>basemodel/stream_0_drop_1/dropout/random_uniform/RandomUniformRandomUniform0basemodel/stream_0_drop_1/dropout/Shape:output:0*
T0*,
_output_shapes
:���������� *
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
:���������� 20
.basemodel/stream_0_drop_1/dropout/GreaterEqual�
&basemodel/stream_0_drop_1/dropout/CastCast2basemodel/stream_0_drop_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:���������� 2(
&basemodel/stream_0_drop_1/dropout/Cast�
'basemodel/stream_0_drop_1/dropout/Mul_1Mul)basemodel/stream_0_drop_1/dropout/Mul:z:0*basemodel/stream_0_drop_1/dropout/Cast:y:0*
T0*,
_output_shapes
:���������� 2)
'basemodel/stream_0_drop_1/dropout/Mul_1�
/basemodel/stream_0_conv_2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������21
/basemodel/stream_0_conv_2/conv1d/ExpandDims/dim�
+basemodel/stream_0_conv_2/conv1d/ExpandDims
ExpandDims+basemodel/stream_0_drop_1/dropout/Mul_1:z:08basemodel/stream_0_conv_2/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:���������� 2-
+basemodel/stream_0_conv_2/conv1d/ExpandDims�
<basemodel/stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpEbasemodel_stream_0_conv_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: @*
dtype02>
<basemodel/stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOp�
1basemodel/stream_0_conv_2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 23
1basemodel/stream_0_conv_2/conv1d/ExpandDims_1/dim�
-basemodel/stream_0_conv_2/conv1d/ExpandDims_1
ExpandDimsDbasemodel/stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOp:value:0:basemodel/stream_0_conv_2/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: @2/
-basemodel/stream_0_conv_2/conv1d/ExpandDims_1�
 basemodel/stream_0_conv_2/conv1dConv2D4basemodel/stream_0_conv_2/conv1d/ExpandDims:output:06basemodel/stream_0_conv_2/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:����������@*
paddingSAME*
strides
2"
 basemodel/stream_0_conv_2/conv1d�
(basemodel/stream_0_conv_2/conv1d/SqueezeSqueeze)basemodel/stream_0_conv_2/conv1d:output:0*
T0*,
_output_shapes
:����������@*
squeeze_dims

���������2*
(basemodel/stream_0_conv_2/conv1d/Squeeze�
0basemodel/stream_0_conv_2/BiasAdd/ReadVariableOpReadVariableOp9basemodel_stream_0_conv_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype022
0basemodel/stream_0_conv_2/BiasAdd/ReadVariableOp�
!basemodel/stream_0_conv_2/BiasAddBiasAdd1basemodel/stream_0_conv_2/conv1d/Squeeze:output:08basemodel/stream_0_conv_2/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������@2#
!basemodel/stream_0_conv_2/BiasAdd�
>basemodel/batch_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2@
>basemodel/batch_normalization_1/moments/mean/reduction_indices�
,basemodel/batch_normalization_1/moments/meanMean*basemodel/stream_0_conv_2/BiasAdd:output:0Gbasemodel/batch_normalization_1/moments/mean/reduction_indices:output:0*
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
9basemodel/batch_normalization_1/moments/SquaredDifferenceSquaredDifference*basemodel/stream_0_conv_2/BiasAdd:output:0=basemodel/batch_normalization_1/moments/StopGradient:output:0*
T0*,
_output_shapes
:����������@2;
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
/basemodel/batch_normalization_1/batchnorm/mul_1Mul*basemodel/stream_0_conv_2/BiasAdd:output:01basemodel/batch_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:����������@21
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
:����������@21
/basemodel/batch_normalization_1/batchnorm/add_1�
basemodel/activation_1/ReluRelu3basemodel/batch_normalization_1/batchnorm/add_1:z:0*
T0*,
_output_shapes
:����������@2
basemodel/activation_1/Relu�
'basemodel/stream_0_drop_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2)
'basemodel/stream_0_drop_2/dropout/Const�
%basemodel/stream_0_drop_2/dropout/MulMul)basemodel/activation_1/Relu:activations:00basemodel/stream_0_drop_2/dropout/Const:output:0*
T0*,
_output_shapes
:����������@2'
%basemodel/stream_0_drop_2/dropout/Mul�
'basemodel/stream_0_drop_2/dropout/ShapeShape)basemodel/activation_1/Relu:activations:0*
T0*
_output_shapes
:2)
'basemodel/stream_0_drop_2/dropout/Shape�
>basemodel/stream_0_drop_2/dropout/random_uniform/RandomUniformRandomUniform0basemodel/stream_0_drop_2/dropout/Shape:output:0*
T0*,
_output_shapes
:����������@*
dtype0*
seed�*
seed2�2@
>basemodel/stream_0_drop_2/dropout/random_uniform/RandomUniform�
0basemodel/stream_0_drop_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���>22
0basemodel/stream_0_drop_2/dropout/GreaterEqual/y�
.basemodel/stream_0_drop_2/dropout/GreaterEqualGreaterEqualGbasemodel/stream_0_drop_2/dropout/random_uniform/RandomUniform:output:09basemodel/stream_0_drop_2/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:����������@20
.basemodel/stream_0_drop_2/dropout/GreaterEqual�
&basemodel/stream_0_drop_2/dropout/CastCast2basemodel/stream_0_drop_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:����������@2(
&basemodel/stream_0_drop_2/dropout/Cast�
'basemodel/stream_0_drop_2/dropout/Mul_1Mul)basemodel/stream_0_drop_2/dropout/Mul:z:0*basemodel/stream_0_drop_2/dropout/Cast:y:0*
T0*,
_output_shapes
:����������@2)
'basemodel/stream_0_drop_2/dropout/Mul_1�
9basemodel/global_average_pooling1d/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2;
9basemodel/global_average_pooling1d/Mean/reduction_indices�
'basemodel/global_average_pooling1d/MeanMean+basemodel/stream_0_drop_2/dropout/Mul_1:z:0Bbasemodel/global_average_pooling1d/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:���������@2)
'basemodel/global_average_pooling1d/Mean�
'basemodel/concatenate/concat/concat_dimConst*
_output_shapes
: *
dtype0*
value	B :2)
'basemodel/concatenate/concat/concat_dim�
#basemodel/concatenate/concat/concatIdentity0basemodel/global_average_pooling1d/Mean:output:0*
T0*'
_output_shapes
:���������@2%
#basemodel/concatenate/concat/concat�
'basemodel/dense_1_dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2)
'basemodel/dense_1_dropout/dropout/Const�
%basemodel/dense_1_dropout/dropout/MulMul,basemodel/concatenate/concat/concat:output:00basemodel/dense_1_dropout/dropout/Const:output:0*
T0*'
_output_shapes
:���������@2'
%basemodel/dense_1_dropout/dropout/Mul�
'basemodel/dense_1_dropout/dropout/ShapeShape,basemodel/concatenate/concat/concat:output:0*
T0*
_output_shapes
:2)
'basemodel/dense_1_dropout/dropout/Shape�
>basemodel/dense_1_dropout/dropout/random_uniform/RandomUniformRandomUniform0basemodel/dense_1_dropout/dropout/Shape:output:0*
T0*'
_output_shapes
:���������@*
dtype0*
seed�2@
>basemodel/dense_1_dropout/dropout/random_uniform/RandomUniform�
0basemodel/dense_1_dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>22
0basemodel/dense_1_dropout/dropout/GreaterEqual/y�
.basemodel/dense_1_dropout/dropout/GreaterEqualGreaterEqualGbasemodel/dense_1_dropout/dropout/random_uniform/RandomUniform:output:09basemodel/dense_1_dropout/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:���������@20
.basemodel/dense_1_dropout/dropout/GreaterEqual�
&basemodel/dense_1_dropout/dropout/CastCast2basemodel/dense_1_dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:���������@2(
&basemodel/dense_1_dropout/dropout/Cast�
'basemodel/dense_1_dropout/dropout/Mul_1Mul)basemodel/dense_1_dropout/dropout/Mul:z:0*basemodel/dense_1_dropout/dropout/Cast:y:0*
T0*'
_output_shapes
:���������@2)
'basemodel/dense_1_dropout/dropout/Mul_1�
'basemodel/dense_1/MatMul/ReadVariableOpReadVariableOp0basemodel_dense_1_matmul_readvariableop_resource*
_output_shapes

:@T*
dtype02)
'basemodel/dense_1/MatMul/ReadVariableOp�
basemodel/dense_1/MatMulMatMul+basemodel/dense_1_dropout/dropout/Mul_1:z:0/basemodel/dense_1/MatMul/ReadVariableOp:value:0*
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
>basemodel/batch_normalization_2/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2@
>basemodel/batch_normalization_2/moments/mean/reduction_indices�
,basemodel/batch_normalization_2/moments/meanMean"basemodel/dense_1/BiasAdd:output:0Gbasemodel/batch_normalization_2/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:T*
	keep_dims(2.
,basemodel/batch_normalization_2/moments/mean�
4basemodel/batch_normalization_2/moments/StopGradientStopGradient5basemodel/batch_normalization_2/moments/mean:output:0*
T0*
_output_shapes

:T26
4basemodel/batch_normalization_2/moments/StopGradient�
9basemodel/batch_normalization_2/moments/SquaredDifferenceSquaredDifference"basemodel/dense_1/BiasAdd:output:0=basemodel/batch_normalization_2/moments/StopGradient:output:0*
T0*'
_output_shapes
:���������T2;
9basemodel/batch_normalization_2/moments/SquaredDifference�
Bbasemodel/batch_normalization_2/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2D
Bbasemodel/batch_normalization_2/moments/variance/reduction_indices�
0basemodel/batch_normalization_2/moments/varianceMean=basemodel/batch_normalization_2/moments/SquaredDifference:z:0Kbasemodel/batch_normalization_2/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:T*
	keep_dims(22
0basemodel/batch_normalization_2/moments/variance�
/basemodel/batch_normalization_2/moments/SqueezeSqueeze5basemodel/batch_normalization_2/moments/mean:output:0*
T0*
_output_shapes
:T*
squeeze_dims
 21
/basemodel/batch_normalization_2/moments/Squeeze�
1basemodel/batch_normalization_2/moments/Squeeze_1Squeeze9basemodel/batch_normalization_2/moments/variance:output:0*
T0*
_output_shapes
:T*
squeeze_dims
 23
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
:T*
dtype02@
>basemodel/batch_normalization_2/AssignMovingAvg/ReadVariableOp�
3basemodel/batch_normalization_2/AssignMovingAvg/subSubFbasemodel/batch_normalization_2/AssignMovingAvg/ReadVariableOp:value:08basemodel/batch_normalization_2/moments/Squeeze:output:0*
T0*
_output_shapes
:T25
3basemodel/batch_normalization_2/AssignMovingAvg/sub�
3basemodel/batch_normalization_2/AssignMovingAvg/mulMul7basemodel/batch_normalization_2/AssignMovingAvg/sub:z:0>basemodel/batch_normalization_2/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:T25
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
:T*
dtype02B
@basemodel/batch_normalization_2/AssignMovingAvg_1/ReadVariableOp�
5basemodel/batch_normalization_2/AssignMovingAvg_1/subSubHbasemodel/batch_normalization_2/AssignMovingAvg_1/ReadVariableOp:value:0:basemodel/batch_normalization_2/moments/Squeeze_1:output:0*
T0*
_output_shapes
:T27
5basemodel/batch_normalization_2/AssignMovingAvg_1/sub�
5basemodel/batch_normalization_2/AssignMovingAvg_1/mulMul9basemodel/batch_normalization_2/AssignMovingAvg_1/sub:z:0@basemodel/batch_normalization_2/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:T27
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
:T2/
-basemodel/batch_normalization_2/batchnorm/add�
/basemodel/batch_normalization_2/batchnorm/RsqrtRsqrt1basemodel/batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes
:T21
/basemodel/batch_normalization_2/batchnorm/Rsqrt�
<basemodel/batch_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOpEbasemodel_batch_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes
:T*
dtype02>
<basemodel/batch_normalization_2/batchnorm/mul/ReadVariableOp�
-basemodel/batch_normalization_2/batchnorm/mulMul3basemodel/batch_normalization_2/batchnorm/Rsqrt:y:0Dbasemodel/batch_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:T2/
-basemodel/batch_normalization_2/batchnorm/mul�
/basemodel/batch_normalization_2/batchnorm/mul_1Mul"basemodel/dense_1/BiasAdd:output:01basemodel/batch_normalization_2/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������T21
/basemodel/batch_normalization_2/batchnorm/mul_1�
/basemodel/batch_normalization_2/batchnorm/mul_2Mul8basemodel/batch_normalization_2/moments/Squeeze:output:01basemodel/batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes
:T21
/basemodel/batch_normalization_2/batchnorm/mul_2�
8basemodel/batch_normalization_2/batchnorm/ReadVariableOpReadVariableOpAbasemodel_batch_normalization_2_batchnorm_readvariableop_resource*
_output_shapes
:T*
dtype02:
8basemodel/batch_normalization_2/batchnorm/ReadVariableOp�
-basemodel/batch_normalization_2/batchnorm/subSub@basemodel/batch_normalization_2/batchnorm/ReadVariableOp:value:03basemodel/batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes
:T2/
-basemodel/batch_normalization_2/batchnorm/sub�
/basemodel/batch_normalization_2/batchnorm/add_1AddV23basemodel/batch_normalization_2/batchnorm/mul_1:z:01basemodel/batch_normalization_2/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������T21
/basemodel/batch_normalization_2/batchnorm/add_1�
$basemodel/dense_activation_1/SigmoidSigmoid3basemodel/batch_normalization_2/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������T2&
$basemodel/dense_activation_1/Sigmoid�
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
:����������2-
+basemodel/stream_0_input_drop/dropout_1/Mul�
-basemodel/stream_0_input_drop/dropout_1/ShapeShapeinputs_1*
T0*
_output_shapes
:2/
-basemodel/stream_0_input_drop/dropout_1/Shape�
Dbasemodel/stream_0_input_drop/dropout_1/random_uniform/RandomUniformRandomUniform6basemodel/stream_0_input_drop/dropout_1/Shape:output:0*
T0*,
_output_shapes
:����������*
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
:����������26
4basemodel/stream_0_input_drop/dropout_1/GreaterEqual�
,basemodel/stream_0_input_drop/dropout_1/CastCast8basemodel/stream_0_input_drop/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:����������2.
,basemodel/stream_0_input_drop/dropout_1/Cast�
-basemodel/stream_0_input_drop/dropout_1/Mul_1Mul/basemodel/stream_0_input_drop/dropout_1/Mul:z:00basemodel/stream_0_input_drop/dropout_1/Cast:y:0*
T0*,
_output_shapes
:����������2/
-basemodel/stream_0_input_drop/dropout_1/Mul_1�
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
:����������2/
-basemodel/stream_0_conv_1/conv1d_1/ExpandDims�
>basemodel/stream_0_conv_1/conv1d_1/ExpandDims_1/ReadVariableOpReadVariableOpEbasemodel_stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
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
: 21
/basemodel/stream_0_conv_1/conv1d_1/ExpandDims_1�
"basemodel/stream_0_conv_1/conv1d_1Conv2D6basemodel/stream_0_conv_1/conv1d_1/ExpandDims:output:08basemodel/stream_0_conv_1/conv1d_1/ExpandDims_1:output:0*
T0*0
_output_shapes
:���������� *
paddingSAME*
strides
2$
"basemodel/stream_0_conv_1/conv1d_1�
*basemodel/stream_0_conv_1/conv1d_1/SqueezeSqueeze+basemodel/stream_0_conv_1/conv1d_1:output:0*
T0*,
_output_shapes
:���������� *
squeeze_dims

���������2,
*basemodel/stream_0_conv_1/conv1d_1/Squeeze�
2basemodel/stream_0_conv_1/BiasAdd_1/ReadVariableOpReadVariableOp9basemodel_stream_0_conv_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype024
2basemodel/stream_0_conv_1/BiasAdd_1/ReadVariableOp�
#basemodel/stream_0_conv_1/BiasAdd_1BiasAdd3basemodel/stream_0_conv_1/conv1d_1/Squeeze:output:0:basemodel/stream_0_conv_1/BiasAdd_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������� 2%
#basemodel/stream_0_conv_1/BiasAdd_1�
>basemodel/batch_normalization/moments_1/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2@
>basemodel/batch_normalization/moments_1/mean/reduction_indices�
,basemodel/batch_normalization/moments_1/meanMean,basemodel/stream_0_conv_1/BiasAdd_1:output:0Gbasemodel/batch_normalization/moments_1/mean/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(2.
,basemodel/batch_normalization/moments_1/mean�
4basemodel/batch_normalization/moments_1/StopGradientStopGradient5basemodel/batch_normalization/moments_1/mean:output:0*
T0*"
_output_shapes
: 26
4basemodel/batch_normalization/moments_1/StopGradient�
9basemodel/batch_normalization/moments_1/SquaredDifferenceSquaredDifference,basemodel/stream_0_conv_1/BiasAdd_1:output:0=basemodel/batch_normalization/moments_1/StopGradient:output:0*
T0*,
_output_shapes
:���������� 2;
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
: *
	keep_dims(22
0basemodel/batch_normalization/moments_1/variance�
/basemodel/batch_normalization/moments_1/SqueezeSqueeze5basemodel/batch_normalization/moments_1/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 21
/basemodel/batch_normalization/moments_1/Squeeze�
1basemodel/batch_normalization/moments_1/Squeeze_1Squeeze9basemodel/batch_normalization/moments_1/variance:output:0*
T0*
_output_shapes
: *
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
: *
dtype02@
>basemodel/batch_normalization/AssignMovingAvg_2/ReadVariableOp�
3basemodel/batch_normalization/AssignMovingAvg_2/subSubFbasemodel/batch_normalization/AssignMovingAvg_2/ReadVariableOp:value:08basemodel/batch_normalization/moments_1/Squeeze:output:0*
T0*
_output_shapes
: 25
3basemodel/batch_normalization/AssignMovingAvg_2/sub�
3basemodel/batch_normalization/AssignMovingAvg_2/mulMul7basemodel/batch_normalization/AssignMovingAvg_2/sub:z:0>basemodel/batch_normalization/AssignMovingAvg_2/decay:output:0*
T0*
_output_shapes
: 25
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
: *
dtype02@
>basemodel/batch_normalization/AssignMovingAvg_3/ReadVariableOp�
3basemodel/batch_normalization/AssignMovingAvg_3/subSubFbasemodel/batch_normalization/AssignMovingAvg_3/ReadVariableOp:value:0:basemodel/batch_normalization/moments_1/Squeeze_1:output:0*
T0*
_output_shapes
: 25
3basemodel/batch_normalization/AssignMovingAvg_3/sub�
3basemodel/batch_normalization/AssignMovingAvg_3/mulMul7basemodel/batch_normalization/AssignMovingAvg_3/sub:z:0>basemodel/batch_normalization/AssignMovingAvg_3/decay:output:0*
T0*
_output_shapes
: 25
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
: 2/
-basemodel/batch_normalization/batchnorm_1/add�
/basemodel/batch_normalization/batchnorm_1/RsqrtRsqrt1basemodel/batch_normalization/batchnorm_1/add:z:0*
T0*
_output_shapes
: 21
/basemodel/batch_normalization/batchnorm_1/Rsqrt�
<basemodel/batch_normalization/batchnorm_1/mul/ReadVariableOpReadVariableOpCbasemodel_batch_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02>
<basemodel/batch_normalization/batchnorm_1/mul/ReadVariableOp�
-basemodel/batch_normalization/batchnorm_1/mulMul3basemodel/batch_normalization/batchnorm_1/Rsqrt:y:0Dbasemodel/batch_normalization/batchnorm_1/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2/
-basemodel/batch_normalization/batchnorm_1/mul�
/basemodel/batch_normalization/batchnorm_1/mul_1Mul,basemodel/stream_0_conv_1/BiasAdd_1:output:01basemodel/batch_normalization/batchnorm_1/mul:z:0*
T0*,
_output_shapes
:���������� 21
/basemodel/batch_normalization/batchnorm_1/mul_1�
/basemodel/batch_normalization/batchnorm_1/mul_2Mul8basemodel/batch_normalization/moments_1/Squeeze:output:01basemodel/batch_normalization/batchnorm_1/mul:z:0*
T0*
_output_shapes
: 21
/basemodel/batch_normalization/batchnorm_1/mul_2�
8basemodel/batch_normalization/batchnorm_1/ReadVariableOpReadVariableOp?basemodel_batch_normalization_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02:
8basemodel/batch_normalization/batchnorm_1/ReadVariableOp�
-basemodel/batch_normalization/batchnorm_1/subSub@basemodel/batch_normalization/batchnorm_1/ReadVariableOp:value:03basemodel/batch_normalization/batchnorm_1/mul_2:z:0*
T0*
_output_shapes
: 2/
-basemodel/batch_normalization/batchnorm_1/sub�
/basemodel/batch_normalization/batchnorm_1/add_1AddV23basemodel/batch_normalization/batchnorm_1/mul_1:z:01basemodel/batch_normalization/batchnorm_1/sub:z:0*
T0*,
_output_shapes
:���������� 21
/basemodel/batch_normalization/batchnorm_1/add_1�
basemodel/activation/Relu_1Relu3basemodel/batch_normalization/batchnorm_1/add_1:z:0*
T0*,
_output_shapes
:���������� 2
basemodel/activation/Relu_1�
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
:���������� 2)
'basemodel/stream_0_drop_1/dropout_1/Mul�
)basemodel/stream_0_drop_1/dropout_1/ShapeShape)basemodel/activation/Relu_1:activations:0*
T0*
_output_shapes
:2+
)basemodel/stream_0_drop_1/dropout_1/Shape�
@basemodel/stream_0_drop_1/dropout_1/random_uniform/RandomUniformRandomUniform2basemodel/stream_0_drop_1/dropout_1/Shape:output:0*
T0*,
_output_shapes
:���������� *
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
:���������� 22
0basemodel/stream_0_drop_1/dropout_1/GreaterEqual�
(basemodel/stream_0_drop_1/dropout_1/CastCast4basemodel/stream_0_drop_1/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:���������� 2*
(basemodel/stream_0_drop_1/dropout_1/Cast�
)basemodel/stream_0_drop_1/dropout_1/Mul_1Mul+basemodel/stream_0_drop_1/dropout_1/Mul:z:0,basemodel/stream_0_drop_1/dropout_1/Cast:y:0*
T0*,
_output_shapes
:���������� 2+
)basemodel/stream_0_drop_1/dropout_1/Mul_1�
1basemodel/stream_0_conv_2/conv1d_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������23
1basemodel/stream_0_conv_2/conv1d_1/ExpandDims/dim�
-basemodel/stream_0_conv_2/conv1d_1/ExpandDims
ExpandDims-basemodel/stream_0_drop_1/dropout_1/Mul_1:z:0:basemodel/stream_0_conv_2/conv1d_1/ExpandDims/dim:output:0*
T0*0
_output_shapes
:���������� 2/
-basemodel/stream_0_conv_2/conv1d_1/ExpandDims�
>basemodel/stream_0_conv_2/conv1d_1/ExpandDims_1/ReadVariableOpReadVariableOpEbasemodel_stream_0_conv_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: @*
dtype02@
>basemodel/stream_0_conv_2/conv1d_1/ExpandDims_1/ReadVariableOp�
3basemodel/stream_0_conv_2/conv1d_1/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 25
3basemodel/stream_0_conv_2/conv1d_1/ExpandDims_1/dim�
/basemodel/stream_0_conv_2/conv1d_1/ExpandDims_1
ExpandDimsFbasemodel/stream_0_conv_2/conv1d_1/ExpandDims_1/ReadVariableOp:value:0<basemodel/stream_0_conv_2/conv1d_1/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: @21
/basemodel/stream_0_conv_2/conv1d_1/ExpandDims_1�
"basemodel/stream_0_conv_2/conv1d_1Conv2D6basemodel/stream_0_conv_2/conv1d_1/ExpandDims:output:08basemodel/stream_0_conv_2/conv1d_1/ExpandDims_1:output:0*
T0*0
_output_shapes
:����������@*
paddingSAME*
strides
2$
"basemodel/stream_0_conv_2/conv1d_1�
*basemodel/stream_0_conv_2/conv1d_1/SqueezeSqueeze+basemodel/stream_0_conv_2/conv1d_1:output:0*
T0*,
_output_shapes
:����������@*
squeeze_dims

���������2,
*basemodel/stream_0_conv_2/conv1d_1/Squeeze�
2basemodel/stream_0_conv_2/BiasAdd_1/ReadVariableOpReadVariableOp9basemodel_stream_0_conv_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype024
2basemodel/stream_0_conv_2/BiasAdd_1/ReadVariableOp�
#basemodel/stream_0_conv_2/BiasAdd_1BiasAdd3basemodel/stream_0_conv_2/conv1d_1/Squeeze:output:0:basemodel/stream_0_conv_2/BiasAdd_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������@2%
#basemodel/stream_0_conv_2/BiasAdd_1�
@basemodel/batch_normalization_1/moments_1/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2B
@basemodel/batch_normalization_1/moments_1/mean/reduction_indices�
.basemodel/batch_normalization_1/moments_1/meanMean,basemodel/stream_0_conv_2/BiasAdd_1:output:0Ibasemodel/batch_normalization_1/moments_1/mean/reduction_indices:output:0*
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
;basemodel/batch_normalization_1/moments_1/SquaredDifferenceSquaredDifference,basemodel/stream_0_conv_2/BiasAdd_1:output:0?basemodel/batch_normalization_1/moments_1/StopGradient:output:0*
T0*,
_output_shapes
:����������@2=
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
1basemodel/batch_normalization_1/batchnorm_1/mul_1Mul,basemodel/stream_0_conv_2/BiasAdd_1:output:03basemodel/batch_normalization_1/batchnorm_1/mul:z:0*
T0*,
_output_shapes
:����������@23
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
:����������@23
1basemodel/batch_normalization_1/batchnorm_1/add_1�
basemodel/activation_1/Relu_1Relu5basemodel/batch_normalization_1/batchnorm_1/add_1:z:0*
T0*,
_output_shapes
:����������@2
basemodel/activation_1/Relu_1�
)basemodel/stream_0_drop_2/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2+
)basemodel/stream_0_drop_2/dropout_1/Const�
'basemodel/stream_0_drop_2/dropout_1/MulMul+basemodel/activation_1/Relu_1:activations:02basemodel/stream_0_drop_2/dropout_1/Const:output:0*
T0*,
_output_shapes
:����������@2)
'basemodel/stream_0_drop_2/dropout_1/Mul�
)basemodel/stream_0_drop_2/dropout_1/ShapeShape+basemodel/activation_1/Relu_1:activations:0*
T0*
_output_shapes
:2+
)basemodel/stream_0_drop_2/dropout_1/Shape�
@basemodel/stream_0_drop_2/dropout_1/random_uniform/RandomUniformRandomUniform2basemodel/stream_0_drop_2/dropout_1/Shape:output:0*
T0*,
_output_shapes
:����������@*
dtype0*
seed�*
seed2�2B
@basemodel/stream_0_drop_2/dropout_1/random_uniform/RandomUniform�
2basemodel/stream_0_drop_2/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���>24
2basemodel/stream_0_drop_2/dropout_1/GreaterEqual/y�
0basemodel/stream_0_drop_2/dropout_1/GreaterEqualGreaterEqualIbasemodel/stream_0_drop_2/dropout_1/random_uniform/RandomUniform:output:0;basemodel/stream_0_drop_2/dropout_1/GreaterEqual/y:output:0*
T0*,
_output_shapes
:����������@22
0basemodel/stream_0_drop_2/dropout_1/GreaterEqual�
(basemodel/stream_0_drop_2/dropout_1/CastCast4basemodel/stream_0_drop_2/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:����������@2*
(basemodel/stream_0_drop_2/dropout_1/Cast�
)basemodel/stream_0_drop_2/dropout_1/Mul_1Mul+basemodel/stream_0_drop_2/dropout_1/Mul:z:0,basemodel/stream_0_drop_2/dropout_1/Cast:y:0*
T0*,
_output_shapes
:����������@2+
)basemodel/stream_0_drop_2/dropout_1/Mul_1�
;basemodel/global_average_pooling1d/Mean_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2=
;basemodel/global_average_pooling1d/Mean_1/reduction_indices�
)basemodel/global_average_pooling1d/Mean_1Mean-basemodel/stream_0_drop_2/dropout_1/Mul_1:z:0Dbasemodel/global_average_pooling1d/Mean_1/reduction_indices:output:0*
T0*'
_output_shapes
:���������@2+
)basemodel/global_average_pooling1d/Mean_1�
)basemodel/concatenate/concat_1/concat_dimConst*
_output_shapes
: *
dtype0*
value	B :2+
)basemodel/concatenate/concat_1/concat_dim�
%basemodel/concatenate/concat_1/concatIdentity2basemodel/global_average_pooling1d/Mean_1:output:0*
T0*'
_output_shapes
:���������@2'
%basemodel/concatenate/concat_1/concat�
)basemodel/dense_1_dropout/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2+
)basemodel/dense_1_dropout/dropout_1/Const�
'basemodel/dense_1_dropout/dropout_1/MulMul.basemodel/concatenate/concat_1/concat:output:02basemodel/dense_1_dropout/dropout_1/Const:output:0*
T0*'
_output_shapes
:���������@2)
'basemodel/dense_1_dropout/dropout_1/Mul�
)basemodel/dense_1_dropout/dropout_1/ShapeShape.basemodel/concatenate/concat_1/concat:output:0*
T0*
_output_shapes
:2+
)basemodel/dense_1_dropout/dropout_1/Shape�
@basemodel/dense_1_dropout/dropout_1/random_uniform/RandomUniformRandomUniform2basemodel/dense_1_dropout/dropout_1/Shape:output:0*
T0*'
_output_shapes
:���������@*
dtype0*
seed�*
seed22B
@basemodel/dense_1_dropout/dropout_1/random_uniform/RandomUniform�
2basemodel/dense_1_dropout/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>24
2basemodel/dense_1_dropout/dropout_1/GreaterEqual/y�
0basemodel/dense_1_dropout/dropout_1/GreaterEqualGreaterEqualIbasemodel/dense_1_dropout/dropout_1/random_uniform/RandomUniform:output:0;basemodel/dense_1_dropout/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:���������@22
0basemodel/dense_1_dropout/dropout_1/GreaterEqual�
(basemodel/dense_1_dropout/dropout_1/CastCast4basemodel/dense_1_dropout/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:���������@2*
(basemodel/dense_1_dropout/dropout_1/Cast�
)basemodel/dense_1_dropout/dropout_1/Mul_1Mul+basemodel/dense_1_dropout/dropout_1/Mul:z:0,basemodel/dense_1_dropout/dropout_1/Cast:y:0*
T0*'
_output_shapes
:���������@2+
)basemodel/dense_1_dropout/dropout_1/Mul_1�
)basemodel/dense_1/MatMul_1/ReadVariableOpReadVariableOp0basemodel_dense_1_matmul_readvariableop_resource*
_output_shapes

:@T*
dtype02+
)basemodel/dense_1/MatMul_1/ReadVariableOp�
basemodel/dense_1/MatMul_1MatMul-basemodel/dense_1_dropout/dropout_1/Mul_1:z:01basemodel/dense_1/MatMul_1/ReadVariableOp:value:0*
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
@basemodel/batch_normalization_2/moments_1/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2B
@basemodel/batch_normalization_2/moments_1/mean/reduction_indices�
.basemodel/batch_normalization_2/moments_1/meanMean$basemodel/dense_1/BiasAdd_1:output:0Ibasemodel/batch_normalization_2/moments_1/mean/reduction_indices:output:0*
T0*
_output_shapes

:T*
	keep_dims(20
.basemodel/batch_normalization_2/moments_1/mean�
6basemodel/batch_normalization_2/moments_1/StopGradientStopGradient7basemodel/batch_normalization_2/moments_1/mean:output:0*
T0*
_output_shapes

:T28
6basemodel/batch_normalization_2/moments_1/StopGradient�
;basemodel/batch_normalization_2/moments_1/SquaredDifferenceSquaredDifference$basemodel/dense_1/BiasAdd_1:output:0?basemodel/batch_normalization_2/moments_1/StopGradient:output:0*
T0*'
_output_shapes
:���������T2=
;basemodel/batch_normalization_2/moments_1/SquaredDifference�
Dbasemodel/batch_normalization_2/moments_1/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2F
Dbasemodel/batch_normalization_2/moments_1/variance/reduction_indices�
2basemodel/batch_normalization_2/moments_1/varianceMean?basemodel/batch_normalization_2/moments_1/SquaredDifference:z:0Mbasemodel/batch_normalization_2/moments_1/variance/reduction_indices:output:0*
T0*
_output_shapes

:T*
	keep_dims(24
2basemodel/batch_normalization_2/moments_1/variance�
1basemodel/batch_normalization_2/moments_1/SqueezeSqueeze7basemodel/batch_normalization_2/moments_1/mean:output:0*
T0*
_output_shapes
:T*
squeeze_dims
 23
1basemodel/batch_normalization_2/moments_1/Squeeze�
3basemodel/batch_normalization_2/moments_1/Squeeze_1Squeeze;basemodel/batch_normalization_2/moments_1/variance:output:0*
T0*
_output_shapes
:T*
squeeze_dims
 25
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
:T*
dtype02B
@basemodel/batch_normalization_2/AssignMovingAvg_2/ReadVariableOp�
5basemodel/batch_normalization_2/AssignMovingAvg_2/subSubHbasemodel/batch_normalization_2/AssignMovingAvg_2/ReadVariableOp:value:0:basemodel/batch_normalization_2/moments_1/Squeeze:output:0*
T0*
_output_shapes
:T27
5basemodel/batch_normalization_2/AssignMovingAvg_2/sub�
5basemodel/batch_normalization_2/AssignMovingAvg_2/mulMul9basemodel/batch_normalization_2/AssignMovingAvg_2/sub:z:0@basemodel/batch_normalization_2/AssignMovingAvg_2/decay:output:0*
T0*
_output_shapes
:T27
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
:T*
dtype02B
@basemodel/batch_normalization_2/AssignMovingAvg_3/ReadVariableOp�
5basemodel/batch_normalization_2/AssignMovingAvg_3/subSubHbasemodel/batch_normalization_2/AssignMovingAvg_3/ReadVariableOp:value:0<basemodel/batch_normalization_2/moments_1/Squeeze_1:output:0*
T0*
_output_shapes
:T27
5basemodel/batch_normalization_2/AssignMovingAvg_3/sub�
5basemodel/batch_normalization_2/AssignMovingAvg_3/mulMul9basemodel/batch_normalization_2/AssignMovingAvg_3/sub:z:0@basemodel/batch_normalization_2/AssignMovingAvg_3/decay:output:0*
T0*
_output_shapes
:T27
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
:T21
/basemodel/batch_normalization_2/batchnorm_1/add�
1basemodel/batch_normalization_2/batchnorm_1/RsqrtRsqrt3basemodel/batch_normalization_2/batchnorm_1/add:z:0*
T0*
_output_shapes
:T23
1basemodel/batch_normalization_2/batchnorm_1/Rsqrt�
>basemodel/batch_normalization_2/batchnorm_1/mul/ReadVariableOpReadVariableOpEbasemodel_batch_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes
:T*
dtype02@
>basemodel/batch_normalization_2/batchnorm_1/mul/ReadVariableOp�
/basemodel/batch_normalization_2/batchnorm_1/mulMul5basemodel/batch_normalization_2/batchnorm_1/Rsqrt:y:0Fbasemodel/batch_normalization_2/batchnorm_1/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:T21
/basemodel/batch_normalization_2/batchnorm_1/mul�
1basemodel/batch_normalization_2/batchnorm_1/mul_1Mul$basemodel/dense_1/BiasAdd_1:output:03basemodel/batch_normalization_2/batchnorm_1/mul:z:0*
T0*'
_output_shapes
:���������T23
1basemodel/batch_normalization_2/batchnorm_1/mul_1�
1basemodel/batch_normalization_2/batchnorm_1/mul_2Mul:basemodel/batch_normalization_2/moments_1/Squeeze:output:03basemodel/batch_normalization_2/batchnorm_1/mul:z:0*
T0*
_output_shapes
:T23
1basemodel/batch_normalization_2/batchnorm_1/mul_2�
:basemodel/batch_normalization_2/batchnorm_1/ReadVariableOpReadVariableOpAbasemodel_batch_normalization_2_batchnorm_readvariableop_resource*
_output_shapes
:T*
dtype02<
:basemodel/batch_normalization_2/batchnorm_1/ReadVariableOp�
/basemodel/batch_normalization_2/batchnorm_1/subSubBbasemodel/batch_normalization_2/batchnorm_1/ReadVariableOp:value:05basemodel/batch_normalization_2/batchnorm_1/mul_2:z:0*
T0*
_output_shapes
:T21
/basemodel/batch_normalization_2/batchnorm_1/sub�
1basemodel/batch_normalization_2/batchnorm_1/add_1AddV25basemodel/batch_normalization_2/batchnorm_1/mul_1:z:03basemodel/batch_normalization_2/batchnorm_1/sub:z:0*
T0*'
_output_shapes
:���������T23
1basemodel/batch_normalization_2/batchnorm_1/add_1�
&basemodel/dense_activation_1/Sigmoid_1Sigmoid5basemodel/batch_normalization_2/batchnorm_1/add_1:z:0*
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
: *
dtype027
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp�
&stream_0_conv_1/kernel/Regularizer/AbsAbs=stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
: 2(
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
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpEbasemodel_stream_0_conv_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: @*
dtype02:
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp�
)stream_0_conv_2/kernel/Regularizer/SquareSquare@stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
: @2+
)stream_0_conv_2/kernel/Regularizer/Square�
(stream_0_conv_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_0_conv_2/kernel/Regularizer/Const�
&stream_0_conv_2/kernel/Regularizer/SumSum-stream_0_conv_2/kernel/Regularizer/Square:y:01stream_0_conv_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_2/kernel/Regularizer/Sum�
(stream_0_conv_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2*
(stream_0_conv_2/kernel/Regularizer/mul/x�
&stream_0_conv_2/kernel/Regularizer/mulMul1stream_0_conv_2/kernel/Regularizer/mul/x:output:0/stream_0_conv_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_2/kernel/Regularizer/mul�
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp0basemodel_dense_1_matmul_readvariableop_resource*
_output_shapes

:@T*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOp�
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:@T2 
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

Identity�
NoOpNoOp.^basemodel/batch_normalization/AssignMovingAvg=^basemodel/batch_normalization/AssignMovingAvg/ReadVariableOp0^basemodel/batch_normalization/AssignMovingAvg_1?^basemodel/batch_normalization/AssignMovingAvg_1/ReadVariableOp0^basemodel/batch_normalization/AssignMovingAvg_2?^basemodel/batch_normalization/AssignMovingAvg_2/ReadVariableOp0^basemodel/batch_normalization/AssignMovingAvg_3?^basemodel/batch_normalization/AssignMovingAvg_3/ReadVariableOp7^basemodel/batch_normalization/batchnorm/ReadVariableOp;^basemodel/batch_normalization/batchnorm/mul/ReadVariableOp9^basemodel/batch_normalization/batchnorm_1/ReadVariableOp=^basemodel/batch_normalization/batchnorm_1/mul/ReadVariableOp0^basemodel/batch_normalization_1/AssignMovingAvg?^basemodel/batch_normalization_1/AssignMovingAvg/ReadVariableOp2^basemodel/batch_normalization_1/AssignMovingAvg_1A^basemodel/batch_normalization_1/AssignMovingAvg_1/ReadVariableOp2^basemodel/batch_normalization_1/AssignMovingAvg_2A^basemodel/batch_normalization_1/AssignMovingAvg_2/ReadVariableOp2^basemodel/batch_normalization_1/AssignMovingAvg_3A^basemodel/batch_normalization_1/AssignMovingAvg_3/ReadVariableOp9^basemodel/batch_normalization_1/batchnorm/ReadVariableOp=^basemodel/batch_normalization_1/batchnorm/mul/ReadVariableOp;^basemodel/batch_normalization_1/batchnorm_1/ReadVariableOp?^basemodel/batch_normalization_1/batchnorm_1/mul/ReadVariableOp0^basemodel/batch_normalization_2/AssignMovingAvg?^basemodel/batch_normalization_2/AssignMovingAvg/ReadVariableOp2^basemodel/batch_normalization_2/AssignMovingAvg_1A^basemodel/batch_normalization_2/AssignMovingAvg_1/ReadVariableOp2^basemodel/batch_normalization_2/AssignMovingAvg_2A^basemodel/batch_normalization_2/AssignMovingAvg_2/ReadVariableOp2^basemodel/batch_normalization_2/AssignMovingAvg_3A^basemodel/batch_normalization_2/AssignMovingAvg_3/ReadVariableOp9^basemodel/batch_normalization_2/batchnorm/ReadVariableOp=^basemodel/batch_normalization_2/batchnorm/mul/ReadVariableOp;^basemodel/batch_normalization_2/batchnorm_1/ReadVariableOp?^basemodel/batch_normalization_2/batchnorm_1/mul/ReadVariableOp)^basemodel/dense_1/BiasAdd/ReadVariableOp+^basemodel/dense_1/BiasAdd_1/ReadVariableOp(^basemodel/dense_1/MatMul/ReadVariableOp*^basemodel/dense_1/MatMul_1/ReadVariableOp1^basemodel/stream_0_conv_1/BiasAdd/ReadVariableOp3^basemodel/stream_0_conv_1/BiasAdd_1/ReadVariableOp=^basemodel/stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp?^basemodel/stream_0_conv_1/conv1d_1/ExpandDims_1/ReadVariableOp1^basemodel/stream_0_conv_2/BiasAdd/ReadVariableOp3^basemodel/stream_0_conv_2/BiasAdd_1/ReadVariableOp=^basemodel/stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOp?^basemodel/stream_0_conv_2/conv1d_1/ExpandDims_1/ReadVariableOp.^dense_1/kernel/Regularizer/Abs/ReadVariableOp6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp9^stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*g
_input_shapesV
T:����������:����������: : : : : : : : : : : : : : : : : : 2^
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
>basemodel/batch_normalization_2/batchnorm_1/mul/ReadVariableOp>basemodel/batch_normalization_2/batchnorm_1/mul/ReadVariableOp2T
(basemodel/dense_1/BiasAdd/ReadVariableOp(basemodel/dense_1/BiasAdd/ReadVariableOp2X
*basemodel/dense_1/BiasAdd_1/ReadVariableOp*basemodel/dense_1/BiasAdd_1/ReadVariableOp2R
'basemodel/dense_1/MatMul/ReadVariableOp'basemodel/dense_1/MatMul/ReadVariableOp2V
)basemodel/dense_1/MatMul_1/ReadVariableOp)basemodel/dense_1/MatMul_1/ReadVariableOp2d
0basemodel/stream_0_conv_1/BiasAdd/ReadVariableOp0basemodel/stream_0_conv_1/BiasAdd/ReadVariableOp2h
2basemodel/stream_0_conv_1/BiasAdd_1/ReadVariableOp2basemodel/stream_0_conv_1/BiasAdd_1/ReadVariableOp2|
<basemodel/stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp<basemodel/stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp2�
>basemodel/stream_0_conv_1/conv1d_1/ExpandDims_1/ReadVariableOp>basemodel/stream_0_conv_1/conv1d_1/ExpandDims_1/ReadVariableOp2d
0basemodel/stream_0_conv_2/BiasAdd/ReadVariableOp0basemodel/stream_0_conv_2/BiasAdd/ReadVariableOp2h
2basemodel/stream_0_conv_2/BiasAdd_1/ReadVariableOp2basemodel/stream_0_conv_2/BiasAdd_1/ReadVariableOp2|
<basemodel/stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOp<basemodel/stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOp2�
>basemodel/stream_0_conv_2/conv1d_1/ExpandDims_1/ReadVariableOp>basemodel/stream_0_conv_2/conv1d_1/ExpandDims_1/ReadVariableOp2^
-dense_1/kernel/Regularizer/Abs/ReadVariableOp-dense_1/kernel/Regularizer/Abs/ReadVariableOp2n
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp2t
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp:V R
,
_output_shapes
:����������
"
_user_specified_name
inputs/0:VR
,
_output_shapes
:����������
"
_user_specified_name
inputs/1
�
�
'__inference_model_layer_call_fn_5543730
inputs_0
inputs_1
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: 
	unknown_5: @
	unknown_6:@
	unknown_7:@
	unknown_8:@
	unknown_9:@

unknown_10:@

unknown_11:@T

unknown_12:T

unknown_13:T

unknown_14:T

unknown_15:T

unknown_16:T
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*.
_read_only_resource_inputs
	*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_model_layer_call_and_return_conditional_losses_55428252
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
_construction_contextkEagerRuntime*g
_input_shapesV
T:����������:����������: : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
,
_output_shapes
:����������
"
_user_specified_name
inputs/0:VR
,
_output_shapes
:����������
"
_user_specified_name
inputs/1
�
q
U__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_5541107

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
�
k
O__inference_dense_activation_1_layer_call_and_return_conditional_losses_5541495

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
�
�
P__inference_batch_normalization_layer_call_and_return_conditional_losses_5544607

inputs/
!batchnorm_readvariableop_resource: 3
%batchnorm_mul_readvariableop_resource: 1
#batchnorm_readvariableop_1_resource: 1
#batchnorm_readvariableop_2_resource: 
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOp�
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
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
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/Rsqrt�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/mul/ReadVariableOp�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/mul�
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :������������������ 2
batchnorm/mul_1�
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp_1�
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2�
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp_2�
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :������������������ 2
batchnorm/add_1{
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :������������������ 2

Identity�
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:������������������ : : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :������������������ 
 
_user_specified_nameinputs
�
M
1__inference_stream_0_drop_1_layer_call_fn_5544779

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
:���������� * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_stream_0_drop_1_layer_call_and_return_conditional_losses_55413652
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:���������� 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:���������� :T P
,
_output_shapes
:���������� 
 
_user_specified_nameinputs
�
�
5__inference_batch_normalization_layer_call_fn_5544734

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������� *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_55413432
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:���������� 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������� : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:���������� 
 
_user_specified_nameinputs
��
� 
#__inference__traced_restore_5545559
file_prefix!
assignvariableop_beta_1: #
assignvariableop_1_beta_2: "
assignvariableop_2_decay: *
 assignvariableop_3_learning_rate: &
assignvariableop_4_adam_iter:	 ?
)assignvariableop_5_stream_0_conv_1_kernel: 5
'assignvariableop_6_stream_0_conv_1_bias: :
,assignvariableop_7_batch_normalization_gamma: 9
+assignvariableop_8_batch_normalization_beta: ?
)assignvariableop_9_stream_0_conv_2_kernel: @6
(assignvariableop_10_stream_0_conv_2_bias:@=
/assignvariableop_11_batch_normalization_1_gamma:@<
.assignvariableop_12_batch_normalization_1_beta:@4
"assignvariableop_13_dense_1_kernel:@T.
 assignvariableop_14_dense_1_bias:T=
/assignvariableop_15_batch_normalization_2_gamma:T<
.assignvariableop_16_batch_normalization_2_beta:TA
3assignvariableop_17_batch_normalization_moving_mean: E
7assignvariableop_18_batch_normalization_moving_variance: C
5assignvariableop_19_batch_normalization_1_moving_mean:@G
9assignvariableop_20_batch_normalization_1_moving_variance:@C
5assignvariableop_21_batch_normalization_2_moving_mean:TG
9assignvariableop_22_batch_normalization_2_moving_variance:T#
assignvariableop_23_total: #
assignvariableop_24_count: G
1assignvariableop_25_adam_stream_0_conv_1_kernel_m: =
/assignvariableop_26_adam_stream_0_conv_1_bias_m: B
4assignvariableop_27_adam_batch_normalization_gamma_m: A
3assignvariableop_28_adam_batch_normalization_beta_m: G
1assignvariableop_29_adam_stream_0_conv_2_kernel_m: @=
/assignvariableop_30_adam_stream_0_conv_2_bias_m:@D
6assignvariableop_31_adam_batch_normalization_1_gamma_m:@C
5assignvariableop_32_adam_batch_normalization_1_beta_m:@;
)assignvariableop_33_adam_dense_1_kernel_m:@T5
'assignvariableop_34_adam_dense_1_bias_m:TD
6assignvariableop_35_adam_batch_normalization_2_gamma_m:TC
5assignvariableop_36_adam_batch_normalization_2_beta_m:TG
1assignvariableop_37_adam_stream_0_conv_1_kernel_v: =
/assignvariableop_38_adam_stream_0_conv_1_bias_v: B
4assignvariableop_39_adam_batch_normalization_gamma_v: A
3assignvariableop_40_adam_batch_normalization_beta_v: G
1assignvariableop_41_adam_stream_0_conv_2_kernel_v: @=
/assignvariableop_42_adam_stream_0_conv_2_bias_v:@D
6assignvariableop_43_adam_batch_normalization_1_gamma_v:@C
5assignvariableop_44_adam_batch_normalization_1_beta_v:@;
)assignvariableop_45_adam_dense_1_kernel_v:@T5
'assignvariableop_46_adam_dense_1_bias_v:TD
6assignvariableop_47_adam_batch_normalization_2_gamma_v:TC
5assignvariableop_48_adam_batch_normalization_2_beta_v:T
identity_50��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_34�AssignVariableOp_35�AssignVariableOp_36�AssignVariableOp_37�AssignVariableOp_38�AssignVariableOp_39�AssignVariableOp_4�AssignVariableOp_40�AssignVariableOp_41�AssignVariableOp_42�AssignVariableOp_43�AssignVariableOp_44�AssignVariableOp_45�AssignVariableOp_46�AssignVariableOp_47�AssignVariableOp_48�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:2*
dtype0*�
value�B�2B+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:2*
dtype0*w
valuenBl2B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices�
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::::::::::::*@
dtypes6
422	2
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
AssignVariableOp_7AssignVariableOp,assignvariableop_7_batch_normalization_gammaIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8�
AssignVariableOp_8AssignVariableOp+assignvariableop_8_batch_normalization_betaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9�
AssignVariableOp_9AssignVariableOp)assignvariableop_9_stream_0_conv_2_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10�
AssignVariableOp_10AssignVariableOp(assignvariableop_10_stream_0_conv_2_biasIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11�
AssignVariableOp_11AssignVariableOp/assignvariableop_11_batch_normalization_1_gammaIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12�
AssignVariableOp_12AssignVariableOp.assignvariableop_12_batch_normalization_1_betaIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13�
AssignVariableOp_13AssignVariableOp"assignvariableop_13_dense_1_kernelIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14�
AssignVariableOp_14AssignVariableOp assignvariableop_14_dense_1_biasIdentity_14:output:0"/device:CPU:0*
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
AssignVariableOp_17AssignVariableOp3assignvariableop_17_batch_normalization_moving_meanIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18�
AssignVariableOp_18AssignVariableOp7assignvariableop_18_batch_normalization_moving_varianceIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19�
AssignVariableOp_19AssignVariableOp5assignvariableop_19_batch_normalization_1_moving_meanIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20�
AssignVariableOp_20AssignVariableOp9assignvariableop_20_batch_normalization_1_moving_varianceIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21�
AssignVariableOp_21AssignVariableOp5assignvariableop_21_batch_normalization_2_moving_meanIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22�
AssignVariableOp_22AssignVariableOp9assignvariableop_22_batch_normalization_2_moving_varianceIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23�
AssignVariableOp_23AssignVariableOpassignvariableop_23_totalIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24�
AssignVariableOp_24AssignVariableOpassignvariableop_24_countIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25�
AssignVariableOp_25AssignVariableOp1assignvariableop_25_adam_stream_0_conv_1_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26�
AssignVariableOp_26AssignVariableOp/assignvariableop_26_adam_stream_0_conv_1_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27�
AssignVariableOp_27AssignVariableOp4assignvariableop_27_adam_batch_normalization_gamma_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28�
AssignVariableOp_28AssignVariableOp3assignvariableop_28_adam_batch_normalization_beta_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29�
AssignVariableOp_29AssignVariableOp1assignvariableop_29_adam_stream_0_conv_2_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30�
AssignVariableOp_30AssignVariableOp/assignvariableop_30_adam_stream_0_conv_2_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31�
AssignVariableOp_31AssignVariableOp6assignvariableop_31_adam_batch_normalization_1_gamma_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32�
AssignVariableOp_32AssignVariableOp5assignvariableop_32_adam_batch_normalization_1_beta_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33�
AssignVariableOp_33AssignVariableOp)assignvariableop_33_adam_dense_1_kernel_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34�
AssignVariableOp_34AssignVariableOp'assignvariableop_34_adam_dense_1_bias_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35�
AssignVariableOp_35AssignVariableOp6assignvariableop_35_adam_batch_normalization_2_gamma_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36�
AssignVariableOp_36AssignVariableOp5assignvariableop_36_adam_batch_normalization_2_beta_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37�
AssignVariableOp_37AssignVariableOp1assignvariableop_37_adam_stream_0_conv_1_kernel_vIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38�
AssignVariableOp_38AssignVariableOp/assignvariableop_38_adam_stream_0_conv_1_bias_vIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39�
AssignVariableOp_39AssignVariableOp4assignvariableop_39_adam_batch_normalization_gamma_vIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40�
AssignVariableOp_40AssignVariableOp3assignvariableop_40_adam_batch_normalization_beta_vIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41�
AssignVariableOp_41AssignVariableOp1assignvariableop_41_adam_stream_0_conv_2_kernel_vIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42�
AssignVariableOp_42AssignVariableOp/assignvariableop_42_adam_stream_0_conv_2_bias_vIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43�
AssignVariableOp_43AssignVariableOp6assignvariableop_43_adam_batch_normalization_1_gamma_vIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44�
AssignVariableOp_44AssignVariableOp5assignvariableop_44_adam_batch_normalization_1_beta_vIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45�
AssignVariableOp_45AssignVariableOp)assignvariableop_45_adam_dense_1_kernel_vIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46�
AssignVariableOp_46AssignVariableOp'assignvariableop_46_adam_dense_1_bias_vIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47�
AssignVariableOp_47AssignVariableOp6assignvariableop_47_adam_batch_normalization_2_gamma_vIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48�
AssignVariableOp_48AssignVariableOp5assignvariableop_48_adam_batch_normalization_2_beta_vIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_489
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp�	
Identity_49Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_49f
Identity_50IdentityIdentity_49:output:0^NoOp_1*
T0*
_output_shapes
: 2
Identity_50�
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 2
NoOp_1"#
identity_50Identity_50:output:0*w
_input_shapesf
d: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_48AssignVariableOp_482(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
�
D__inference_dense_1_layer_call_and_return_conditional_losses_5545099

inputs0
matmul_readvariableop_resource:@T-
biasadd_readvariableop_resource:T
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�-dense_1/kernel/Regularizer/Abs/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@T*
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

:@T*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOp�
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:@T2 
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
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2^
-dense_1/kernel/Regularizer/Abs/ReadVariableOp-dense_1/kernel/Regularizer/Abs/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
__inference_loss_fn_2_5545231H
6dense_1_kernel_regularizer_abs_readvariableop_resource:@T
identity��-dense_1/kernel/Regularizer/Abs/ReadVariableOp�
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp6dense_1_kernel_regularizer_abs_readvariableop_resource*
_output_shapes

:@T*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOp�
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:@T2 
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
�
�
%__inference_signature_wrapper_5543134
left_inputs
right_inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: 
	unknown_5: @
	unknown_6:@
	unknown_7:@
	unknown_8:@
	unknown_9:@

unknown_10:@

unknown_11:@T

unknown_12:T

unknown_13:T

unknown_14:T

unknown_15:T

unknown_16:T
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallleft_inputsright_inputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*4
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *+
f&R$
"__inference__wrapped_model_55407732
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
_construction_contextkEagerRuntime*g
_input_shapesV
T:����������:����������: : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
,
_output_shapes
:����������
%
_user_specified_nameleft_inputs:ZV
,
_output_shapes
:����������
&
_user_specified_nameright_inputs
�
�
P__inference_batch_normalization_layer_call_and_return_conditional_losses_5541343

inputs/
!batchnorm_readvariableop_resource: 3
%batchnorm_mul_readvariableop_resource: 1
#batchnorm_readvariableop_1_resource: 1
#batchnorm_readvariableop_2_resource: 
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOp�
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
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
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/Rsqrt�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/mul/ReadVariableOp�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/mul{
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*,
_output_shapes
:���������� 2
batchnorm/mul_1�
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp_1�
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2�
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp_2�
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*,
_output_shapes
:���������� 2
batchnorm/add_1s
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*,
_output_shapes
:���������� 2

Identity�
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������� : : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:T P
,
_output_shapes
:���������� 
 
_user_specified_nameinputs
�
k
L__inference_dense_1_dropout_layer_call_and_return_conditional_losses_5541591

inputs
identity�c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:���������@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:���������@*
dtype0*
seed�2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>2
dropout/GreaterEqual/y�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:���������@2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:���������@2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:���������@2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������@:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
k
L__inference_dense_1_dropout_layer_call_and_return_conditional_losses_5545067

inputs
identity�c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:���������@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:���������@*
dtype0*
seed�2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>2
dropout/GreaterEqual/y�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:���������@2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:���������@2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:���������@2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������@:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
c
G__inference_activation_layer_call_and_return_conditional_losses_5544752

inputs
identityS
ReluReluinputs*
T0*,
_output_shapes
:���������� 2
Reluk
IdentityIdentityRelu:activations:0*
T0*,
_output_shapes
:���������� 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:���������� :T P
,
_output_shapes
:���������� 
 
_user_specified_nameinputs
�
V
:__inference_global_average_pooling1d_layer_call_fn_5545034

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
U__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_55411072
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
�
�
+__inference_basemodel_layer_call_fn_5544406

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: 
	unknown_5: @
	unknown_6:@
	unknown_7:@
	unknown_8:@
	unknown_9:@

unknown_10:@

unknown_11:@T

unknown_12:T

unknown_13:T

unknown_14:T

unknown_15:T

unknown_16:T
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������T*.
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_basemodel_layer_call_and_return_conditional_losses_55419422
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
_construction_contextkEagerRuntime*O
_input_shapes>
<:����������: : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
7__inference_batch_normalization_2_layer_call_fn_5545175

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
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_55411452
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
�
�
'__inference_model_layer_call_fn_5542414
left_inputs
right_inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: 
	unknown_5: @
	unknown_6:@
	unknown_7:@
	unknown_8:@
	unknown_9:@

unknown_10:@

unknown_11:@T

unknown_12:T

unknown_13:T

unknown_14:T

unknown_15:T

unknown_16:T
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallleft_inputsright_inputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*4
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_model_layer_call_and_return_conditional_losses_55423752
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
_construction_contextkEagerRuntime*g
_input_shapesV
T:����������:����������: : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
,
_output_shapes
:����������
%
_user_specified_nameleft_inputs:ZV
,
_output_shapes
:����������
&
_user_specified_nameright_inputs
�
j
L__inference_stream_0_drop_1_layer_call_and_return_conditional_losses_5541365

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:���������� 2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:���������� 2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:���������� :T P
,
_output_shapes
:���������� 
 
_user_specified_nameinputs
�
�
'__inference_model_layer_call_fn_5543688
inputs_0
inputs_1
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: 
	unknown_5: @
	unknown_6:@
	unknown_7:@
	unknown_8:@
	unknown_9:@

unknown_10:@

unknown_11:@T

unknown_12:T

unknown_13:T

unknown_14:T

unknown_15:T

unknown_16:T
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*4
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_model_layer_call_and_return_conditional_losses_55423752
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
_construction_contextkEagerRuntime*g
_input_shapesV
T:����������:����������: : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
,
_output_shapes
:����������
"
_user_specified_name
inputs/0:VR
,
_output_shapes
:����������
"
_user_specified_name
inputs/1
�
q
U__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_5541442

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
:����������@:T P
,
_output_shapes
:����������@
 
_user_specified_nameinputs
�
�
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_5541145

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
��
�
B__inference_model_layer_call_and_return_conditional_losses_5543320
inputs_0
inputs_1[
Ebasemodel_stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource: G
9basemodel_stream_0_conv_1_biasadd_readvariableop_resource: M
?basemodel_batch_normalization_batchnorm_readvariableop_resource: Q
Cbasemodel_batch_normalization_batchnorm_mul_readvariableop_resource: O
Abasemodel_batch_normalization_batchnorm_readvariableop_1_resource: O
Abasemodel_batch_normalization_batchnorm_readvariableop_2_resource: [
Ebasemodel_stream_0_conv_2_conv1d_expanddims_1_readvariableop_resource: @G
9basemodel_stream_0_conv_2_biasadd_readvariableop_resource:@O
Abasemodel_batch_normalization_1_batchnorm_readvariableop_resource:@S
Ebasemodel_batch_normalization_1_batchnorm_mul_readvariableop_resource:@Q
Cbasemodel_batch_normalization_1_batchnorm_readvariableop_1_resource:@Q
Cbasemodel_batch_normalization_1_batchnorm_readvariableop_2_resource:@B
0basemodel_dense_1_matmul_readvariableop_resource:@T?
1basemodel_dense_1_biasadd_readvariableop_resource:TO
Abasemodel_batch_normalization_2_batchnorm_readvariableop_resource:TS
Ebasemodel_batch_normalization_2_batchnorm_mul_readvariableop_resource:TQ
Cbasemodel_batch_normalization_2_batchnorm_readvariableop_1_resource:TQ
Cbasemodel_batch_normalization_2_batchnorm_readvariableop_2_resource:T
identity��6basemodel/batch_normalization/batchnorm/ReadVariableOp�8basemodel/batch_normalization/batchnorm/ReadVariableOp_1�8basemodel/batch_normalization/batchnorm/ReadVariableOp_2�:basemodel/batch_normalization/batchnorm/mul/ReadVariableOp�8basemodel/batch_normalization/batchnorm_1/ReadVariableOp�:basemodel/batch_normalization/batchnorm_1/ReadVariableOp_1�:basemodel/batch_normalization/batchnorm_1/ReadVariableOp_2�<basemodel/batch_normalization/batchnorm_1/mul/ReadVariableOp�8basemodel/batch_normalization_1/batchnorm/ReadVariableOp�:basemodel/batch_normalization_1/batchnorm/ReadVariableOp_1�:basemodel/batch_normalization_1/batchnorm/ReadVariableOp_2�<basemodel/batch_normalization_1/batchnorm/mul/ReadVariableOp�:basemodel/batch_normalization_1/batchnorm_1/ReadVariableOp�<basemodel/batch_normalization_1/batchnorm_1/ReadVariableOp_1�<basemodel/batch_normalization_1/batchnorm_1/ReadVariableOp_2�>basemodel/batch_normalization_1/batchnorm_1/mul/ReadVariableOp�8basemodel/batch_normalization_2/batchnorm/ReadVariableOp�:basemodel/batch_normalization_2/batchnorm/ReadVariableOp_1�:basemodel/batch_normalization_2/batchnorm/ReadVariableOp_2�<basemodel/batch_normalization_2/batchnorm/mul/ReadVariableOp�:basemodel/batch_normalization_2/batchnorm_1/ReadVariableOp�<basemodel/batch_normalization_2/batchnorm_1/ReadVariableOp_1�<basemodel/batch_normalization_2/batchnorm_1/ReadVariableOp_2�>basemodel/batch_normalization_2/batchnorm_1/mul/ReadVariableOp�(basemodel/dense_1/BiasAdd/ReadVariableOp�*basemodel/dense_1/BiasAdd_1/ReadVariableOp�'basemodel/dense_1/MatMul/ReadVariableOp�)basemodel/dense_1/MatMul_1/ReadVariableOp�0basemodel/stream_0_conv_1/BiasAdd/ReadVariableOp�2basemodel/stream_0_conv_1/BiasAdd_1/ReadVariableOp�<basemodel/stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp�>basemodel/stream_0_conv_1/conv1d_1/ExpandDims_1/ReadVariableOp�0basemodel/stream_0_conv_2/BiasAdd/ReadVariableOp�2basemodel/stream_0_conv_2/BiasAdd_1/ReadVariableOp�<basemodel/stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOp�>basemodel/stream_0_conv_2/conv1d_1/ExpandDims_1/ReadVariableOp�-dense_1/kernel/Regularizer/Abs/ReadVariableOp�5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp�8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp�
&basemodel/stream_0_input_drop/IdentityIdentityinputs_0*
T0*,
_output_shapes
:����������2(
&basemodel/stream_0_input_drop/Identity�
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
:����������2-
+basemodel/stream_0_conv_1/conv1d/ExpandDims�
<basemodel/stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpEbasemodel_stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
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
: 2/
-basemodel/stream_0_conv_1/conv1d/ExpandDims_1�
 basemodel/stream_0_conv_1/conv1dConv2D4basemodel/stream_0_conv_1/conv1d/ExpandDims:output:06basemodel/stream_0_conv_1/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:���������� *
paddingSAME*
strides
2"
 basemodel/stream_0_conv_1/conv1d�
(basemodel/stream_0_conv_1/conv1d/SqueezeSqueeze)basemodel/stream_0_conv_1/conv1d:output:0*
T0*,
_output_shapes
:���������� *
squeeze_dims

���������2*
(basemodel/stream_0_conv_1/conv1d/Squeeze�
0basemodel/stream_0_conv_1/BiasAdd/ReadVariableOpReadVariableOp9basemodel_stream_0_conv_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype022
0basemodel/stream_0_conv_1/BiasAdd/ReadVariableOp�
!basemodel/stream_0_conv_1/BiasAddBiasAdd1basemodel/stream_0_conv_1/conv1d/Squeeze:output:08basemodel/stream_0_conv_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������� 2#
!basemodel/stream_0_conv_1/BiasAdd�
6basemodel/batch_normalization/batchnorm/ReadVariableOpReadVariableOp?basemodel_batch_normalization_batchnorm_readvariableop_resource*
_output_shapes
: *
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
: 2-
+basemodel/batch_normalization/batchnorm/add�
-basemodel/batch_normalization/batchnorm/RsqrtRsqrt/basemodel/batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes
: 2/
-basemodel/batch_normalization/batchnorm/Rsqrt�
:basemodel/batch_normalization/batchnorm/mul/ReadVariableOpReadVariableOpCbasemodel_batch_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02<
:basemodel/batch_normalization/batchnorm/mul/ReadVariableOp�
+basemodel/batch_normalization/batchnorm/mulMul1basemodel/batch_normalization/batchnorm/Rsqrt:y:0Bbasemodel/batch_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2-
+basemodel/batch_normalization/batchnorm/mul�
-basemodel/batch_normalization/batchnorm/mul_1Mul*basemodel/stream_0_conv_1/BiasAdd:output:0/basemodel/batch_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:���������� 2/
-basemodel/batch_normalization/batchnorm/mul_1�
8basemodel/batch_normalization/batchnorm/ReadVariableOp_1ReadVariableOpAbasemodel_batch_normalization_batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype02:
8basemodel/batch_normalization/batchnorm/ReadVariableOp_1�
-basemodel/batch_normalization/batchnorm/mul_2Mul@basemodel/batch_normalization/batchnorm/ReadVariableOp_1:value:0/basemodel/batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes
: 2/
-basemodel/batch_normalization/batchnorm/mul_2�
8basemodel/batch_normalization/batchnorm/ReadVariableOp_2ReadVariableOpAbasemodel_batch_normalization_batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype02:
8basemodel/batch_normalization/batchnorm/ReadVariableOp_2�
+basemodel/batch_normalization/batchnorm/subSub@basemodel/batch_normalization/batchnorm/ReadVariableOp_2:value:01basemodel/batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2-
+basemodel/batch_normalization/batchnorm/sub�
-basemodel/batch_normalization/batchnorm/add_1AddV21basemodel/batch_normalization/batchnorm/mul_1:z:0/basemodel/batch_normalization/batchnorm/sub:z:0*
T0*,
_output_shapes
:���������� 2/
-basemodel/batch_normalization/batchnorm/add_1�
basemodel/activation/ReluRelu1basemodel/batch_normalization/batchnorm/add_1:z:0*
T0*,
_output_shapes
:���������� 2
basemodel/activation/Relu�
"basemodel/stream_0_drop_1/IdentityIdentity'basemodel/activation/Relu:activations:0*
T0*,
_output_shapes
:���������� 2$
"basemodel/stream_0_drop_1/Identity�
/basemodel/stream_0_conv_2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������21
/basemodel/stream_0_conv_2/conv1d/ExpandDims/dim�
+basemodel/stream_0_conv_2/conv1d/ExpandDims
ExpandDims+basemodel/stream_0_drop_1/Identity:output:08basemodel/stream_0_conv_2/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:���������� 2-
+basemodel/stream_0_conv_2/conv1d/ExpandDims�
<basemodel/stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpEbasemodel_stream_0_conv_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: @*
dtype02>
<basemodel/stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOp�
1basemodel/stream_0_conv_2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 23
1basemodel/stream_0_conv_2/conv1d/ExpandDims_1/dim�
-basemodel/stream_0_conv_2/conv1d/ExpandDims_1
ExpandDimsDbasemodel/stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOp:value:0:basemodel/stream_0_conv_2/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: @2/
-basemodel/stream_0_conv_2/conv1d/ExpandDims_1�
 basemodel/stream_0_conv_2/conv1dConv2D4basemodel/stream_0_conv_2/conv1d/ExpandDims:output:06basemodel/stream_0_conv_2/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:����������@*
paddingSAME*
strides
2"
 basemodel/stream_0_conv_2/conv1d�
(basemodel/stream_0_conv_2/conv1d/SqueezeSqueeze)basemodel/stream_0_conv_2/conv1d:output:0*
T0*,
_output_shapes
:����������@*
squeeze_dims

���������2*
(basemodel/stream_0_conv_2/conv1d/Squeeze�
0basemodel/stream_0_conv_2/BiasAdd/ReadVariableOpReadVariableOp9basemodel_stream_0_conv_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype022
0basemodel/stream_0_conv_2/BiasAdd/ReadVariableOp�
!basemodel/stream_0_conv_2/BiasAddBiasAdd1basemodel/stream_0_conv_2/conv1d/Squeeze:output:08basemodel/stream_0_conv_2/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������@2#
!basemodel/stream_0_conv_2/BiasAdd�
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
/basemodel/batch_normalization_1/batchnorm/mul_1Mul*basemodel/stream_0_conv_2/BiasAdd:output:01basemodel/batch_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:����������@21
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
:����������@21
/basemodel/batch_normalization_1/batchnorm/add_1�
basemodel/activation_1/ReluRelu3basemodel/batch_normalization_1/batchnorm/add_1:z:0*
T0*,
_output_shapes
:����������@2
basemodel/activation_1/Relu�
"basemodel/stream_0_drop_2/IdentityIdentity)basemodel/activation_1/Relu:activations:0*
T0*,
_output_shapes
:����������@2$
"basemodel/stream_0_drop_2/Identity�
9basemodel/global_average_pooling1d/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2;
9basemodel/global_average_pooling1d/Mean/reduction_indices�
'basemodel/global_average_pooling1d/MeanMean+basemodel/stream_0_drop_2/Identity:output:0Bbasemodel/global_average_pooling1d/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:���������@2)
'basemodel/global_average_pooling1d/Mean�
'basemodel/concatenate/concat/concat_dimConst*
_output_shapes
: *
dtype0*
value	B :2)
'basemodel/concatenate/concat/concat_dim�
#basemodel/concatenate/concat/concatIdentity0basemodel/global_average_pooling1d/Mean:output:0*
T0*'
_output_shapes
:���������@2%
#basemodel/concatenate/concat/concat�
"basemodel/dense_1_dropout/IdentityIdentity,basemodel/concatenate/concat/concat:output:0*
T0*'
_output_shapes
:���������@2$
"basemodel/dense_1_dropout/Identity�
'basemodel/dense_1/MatMul/ReadVariableOpReadVariableOp0basemodel_dense_1_matmul_readvariableop_resource*
_output_shapes

:@T*
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
8basemodel/batch_normalization_2/batchnorm/ReadVariableOpReadVariableOpAbasemodel_batch_normalization_2_batchnorm_readvariableop_resource*
_output_shapes
:T*
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
:T2/
-basemodel/batch_normalization_2/batchnorm/add�
/basemodel/batch_normalization_2/batchnorm/RsqrtRsqrt1basemodel/batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes
:T21
/basemodel/batch_normalization_2/batchnorm/Rsqrt�
<basemodel/batch_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOpEbasemodel_batch_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes
:T*
dtype02>
<basemodel/batch_normalization_2/batchnorm/mul/ReadVariableOp�
-basemodel/batch_normalization_2/batchnorm/mulMul3basemodel/batch_normalization_2/batchnorm/Rsqrt:y:0Dbasemodel/batch_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:T2/
-basemodel/batch_normalization_2/batchnorm/mul�
/basemodel/batch_normalization_2/batchnorm/mul_1Mul"basemodel/dense_1/BiasAdd:output:01basemodel/batch_normalization_2/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������T21
/basemodel/batch_normalization_2/batchnorm/mul_1�
:basemodel/batch_normalization_2/batchnorm/ReadVariableOp_1ReadVariableOpCbasemodel_batch_normalization_2_batchnorm_readvariableop_1_resource*
_output_shapes
:T*
dtype02<
:basemodel/batch_normalization_2/batchnorm/ReadVariableOp_1�
/basemodel/batch_normalization_2/batchnorm/mul_2MulBbasemodel/batch_normalization_2/batchnorm/ReadVariableOp_1:value:01basemodel/batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes
:T21
/basemodel/batch_normalization_2/batchnorm/mul_2�
:basemodel/batch_normalization_2/batchnorm/ReadVariableOp_2ReadVariableOpCbasemodel_batch_normalization_2_batchnorm_readvariableop_2_resource*
_output_shapes
:T*
dtype02<
:basemodel/batch_normalization_2/batchnorm/ReadVariableOp_2�
-basemodel/batch_normalization_2/batchnorm/subSubBbasemodel/batch_normalization_2/batchnorm/ReadVariableOp_2:value:03basemodel/batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes
:T2/
-basemodel/batch_normalization_2/batchnorm/sub�
/basemodel/batch_normalization_2/batchnorm/add_1AddV23basemodel/batch_normalization_2/batchnorm/mul_1:z:01basemodel/batch_normalization_2/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������T21
/basemodel/batch_normalization_2/batchnorm/add_1�
$basemodel/dense_activation_1/SigmoidSigmoid3basemodel/batch_normalization_2/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������T2&
$basemodel/dense_activation_1/Sigmoid�
(basemodel/stream_0_input_drop/Identity_1Identityinputs_1*
T0*,
_output_shapes
:����������2*
(basemodel/stream_0_input_drop/Identity_1�
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
:����������2/
-basemodel/stream_0_conv_1/conv1d_1/ExpandDims�
>basemodel/stream_0_conv_1/conv1d_1/ExpandDims_1/ReadVariableOpReadVariableOpEbasemodel_stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
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
: 21
/basemodel/stream_0_conv_1/conv1d_1/ExpandDims_1�
"basemodel/stream_0_conv_1/conv1d_1Conv2D6basemodel/stream_0_conv_1/conv1d_1/ExpandDims:output:08basemodel/stream_0_conv_1/conv1d_1/ExpandDims_1:output:0*
T0*0
_output_shapes
:���������� *
paddingSAME*
strides
2$
"basemodel/stream_0_conv_1/conv1d_1�
*basemodel/stream_0_conv_1/conv1d_1/SqueezeSqueeze+basemodel/stream_0_conv_1/conv1d_1:output:0*
T0*,
_output_shapes
:���������� *
squeeze_dims

���������2,
*basemodel/stream_0_conv_1/conv1d_1/Squeeze�
2basemodel/stream_0_conv_1/BiasAdd_1/ReadVariableOpReadVariableOp9basemodel_stream_0_conv_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype024
2basemodel/stream_0_conv_1/BiasAdd_1/ReadVariableOp�
#basemodel/stream_0_conv_1/BiasAdd_1BiasAdd3basemodel/stream_0_conv_1/conv1d_1/Squeeze:output:0:basemodel/stream_0_conv_1/BiasAdd_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������� 2%
#basemodel/stream_0_conv_1/BiasAdd_1�
8basemodel/batch_normalization/batchnorm_1/ReadVariableOpReadVariableOp?basemodel_batch_normalization_batchnorm_readvariableop_resource*
_output_shapes
: *
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
: 2/
-basemodel/batch_normalization/batchnorm_1/add�
/basemodel/batch_normalization/batchnorm_1/RsqrtRsqrt1basemodel/batch_normalization/batchnorm_1/add:z:0*
T0*
_output_shapes
: 21
/basemodel/batch_normalization/batchnorm_1/Rsqrt�
<basemodel/batch_normalization/batchnorm_1/mul/ReadVariableOpReadVariableOpCbasemodel_batch_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02>
<basemodel/batch_normalization/batchnorm_1/mul/ReadVariableOp�
-basemodel/batch_normalization/batchnorm_1/mulMul3basemodel/batch_normalization/batchnorm_1/Rsqrt:y:0Dbasemodel/batch_normalization/batchnorm_1/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2/
-basemodel/batch_normalization/batchnorm_1/mul�
/basemodel/batch_normalization/batchnorm_1/mul_1Mul,basemodel/stream_0_conv_1/BiasAdd_1:output:01basemodel/batch_normalization/batchnorm_1/mul:z:0*
T0*,
_output_shapes
:���������� 21
/basemodel/batch_normalization/batchnorm_1/mul_1�
:basemodel/batch_normalization/batchnorm_1/ReadVariableOp_1ReadVariableOpAbasemodel_batch_normalization_batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype02<
:basemodel/batch_normalization/batchnorm_1/ReadVariableOp_1�
/basemodel/batch_normalization/batchnorm_1/mul_2MulBbasemodel/batch_normalization/batchnorm_1/ReadVariableOp_1:value:01basemodel/batch_normalization/batchnorm_1/mul:z:0*
T0*
_output_shapes
: 21
/basemodel/batch_normalization/batchnorm_1/mul_2�
:basemodel/batch_normalization/batchnorm_1/ReadVariableOp_2ReadVariableOpAbasemodel_batch_normalization_batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype02<
:basemodel/batch_normalization/batchnorm_1/ReadVariableOp_2�
-basemodel/batch_normalization/batchnorm_1/subSubBbasemodel/batch_normalization/batchnorm_1/ReadVariableOp_2:value:03basemodel/batch_normalization/batchnorm_1/mul_2:z:0*
T0*
_output_shapes
: 2/
-basemodel/batch_normalization/batchnorm_1/sub�
/basemodel/batch_normalization/batchnorm_1/add_1AddV23basemodel/batch_normalization/batchnorm_1/mul_1:z:01basemodel/batch_normalization/batchnorm_1/sub:z:0*
T0*,
_output_shapes
:���������� 21
/basemodel/batch_normalization/batchnorm_1/add_1�
basemodel/activation/Relu_1Relu3basemodel/batch_normalization/batchnorm_1/add_1:z:0*
T0*,
_output_shapes
:���������� 2
basemodel/activation/Relu_1�
$basemodel/stream_0_drop_1/Identity_1Identity)basemodel/activation/Relu_1:activations:0*
T0*,
_output_shapes
:���������� 2&
$basemodel/stream_0_drop_1/Identity_1�
1basemodel/stream_0_conv_2/conv1d_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������23
1basemodel/stream_0_conv_2/conv1d_1/ExpandDims/dim�
-basemodel/stream_0_conv_2/conv1d_1/ExpandDims
ExpandDims-basemodel/stream_0_drop_1/Identity_1:output:0:basemodel/stream_0_conv_2/conv1d_1/ExpandDims/dim:output:0*
T0*0
_output_shapes
:���������� 2/
-basemodel/stream_0_conv_2/conv1d_1/ExpandDims�
>basemodel/stream_0_conv_2/conv1d_1/ExpandDims_1/ReadVariableOpReadVariableOpEbasemodel_stream_0_conv_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: @*
dtype02@
>basemodel/stream_0_conv_2/conv1d_1/ExpandDims_1/ReadVariableOp�
3basemodel/stream_0_conv_2/conv1d_1/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 25
3basemodel/stream_0_conv_2/conv1d_1/ExpandDims_1/dim�
/basemodel/stream_0_conv_2/conv1d_1/ExpandDims_1
ExpandDimsFbasemodel/stream_0_conv_2/conv1d_1/ExpandDims_1/ReadVariableOp:value:0<basemodel/stream_0_conv_2/conv1d_1/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: @21
/basemodel/stream_0_conv_2/conv1d_1/ExpandDims_1�
"basemodel/stream_0_conv_2/conv1d_1Conv2D6basemodel/stream_0_conv_2/conv1d_1/ExpandDims:output:08basemodel/stream_0_conv_2/conv1d_1/ExpandDims_1:output:0*
T0*0
_output_shapes
:����������@*
paddingSAME*
strides
2$
"basemodel/stream_0_conv_2/conv1d_1�
*basemodel/stream_0_conv_2/conv1d_1/SqueezeSqueeze+basemodel/stream_0_conv_2/conv1d_1:output:0*
T0*,
_output_shapes
:����������@*
squeeze_dims

���������2,
*basemodel/stream_0_conv_2/conv1d_1/Squeeze�
2basemodel/stream_0_conv_2/BiasAdd_1/ReadVariableOpReadVariableOp9basemodel_stream_0_conv_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype024
2basemodel/stream_0_conv_2/BiasAdd_1/ReadVariableOp�
#basemodel/stream_0_conv_2/BiasAdd_1BiasAdd3basemodel/stream_0_conv_2/conv1d_1/Squeeze:output:0:basemodel/stream_0_conv_2/BiasAdd_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������@2%
#basemodel/stream_0_conv_2/BiasAdd_1�
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
1basemodel/batch_normalization_1/batchnorm_1/mul_1Mul,basemodel/stream_0_conv_2/BiasAdd_1:output:03basemodel/batch_normalization_1/batchnorm_1/mul:z:0*
T0*,
_output_shapes
:����������@23
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
:����������@23
1basemodel/batch_normalization_1/batchnorm_1/add_1�
basemodel/activation_1/Relu_1Relu5basemodel/batch_normalization_1/batchnorm_1/add_1:z:0*
T0*,
_output_shapes
:����������@2
basemodel/activation_1/Relu_1�
$basemodel/stream_0_drop_2/Identity_1Identity+basemodel/activation_1/Relu_1:activations:0*
T0*,
_output_shapes
:����������@2&
$basemodel/stream_0_drop_2/Identity_1�
;basemodel/global_average_pooling1d/Mean_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2=
;basemodel/global_average_pooling1d/Mean_1/reduction_indices�
)basemodel/global_average_pooling1d/Mean_1Mean-basemodel/stream_0_drop_2/Identity_1:output:0Dbasemodel/global_average_pooling1d/Mean_1/reduction_indices:output:0*
T0*'
_output_shapes
:���������@2+
)basemodel/global_average_pooling1d/Mean_1�
)basemodel/concatenate/concat_1/concat_dimConst*
_output_shapes
: *
dtype0*
value	B :2+
)basemodel/concatenate/concat_1/concat_dim�
%basemodel/concatenate/concat_1/concatIdentity2basemodel/global_average_pooling1d/Mean_1:output:0*
T0*'
_output_shapes
:���������@2'
%basemodel/concatenate/concat_1/concat�
$basemodel/dense_1_dropout/Identity_1Identity.basemodel/concatenate/concat_1/concat:output:0*
T0*'
_output_shapes
:���������@2&
$basemodel/dense_1_dropout/Identity_1�
)basemodel/dense_1/MatMul_1/ReadVariableOpReadVariableOp0basemodel_dense_1_matmul_readvariableop_resource*
_output_shapes

:@T*
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
:basemodel/batch_normalization_2/batchnorm_1/ReadVariableOpReadVariableOpAbasemodel_batch_normalization_2_batchnorm_readvariableop_resource*
_output_shapes
:T*
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
:T21
/basemodel/batch_normalization_2/batchnorm_1/add�
1basemodel/batch_normalization_2/batchnorm_1/RsqrtRsqrt3basemodel/batch_normalization_2/batchnorm_1/add:z:0*
T0*
_output_shapes
:T23
1basemodel/batch_normalization_2/batchnorm_1/Rsqrt�
>basemodel/batch_normalization_2/batchnorm_1/mul/ReadVariableOpReadVariableOpEbasemodel_batch_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes
:T*
dtype02@
>basemodel/batch_normalization_2/batchnorm_1/mul/ReadVariableOp�
/basemodel/batch_normalization_2/batchnorm_1/mulMul5basemodel/batch_normalization_2/batchnorm_1/Rsqrt:y:0Fbasemodel/batch_normalization_2/batchnorm_1/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:T21
/basemodel/batch_normalization_2/batchnorm_1/mul�
1basemodel/batch_normalization_2/batchnorm_1/mul_1Mul$basemodel/dense_1/BiasAdd_1:output:03basemodel/batch_normalization_2/batchnorm_1/mul:z:0*
T0*'
_output_shapes
:���������T23
1basemodel/batch_normalization_2/batchnorm_1/mul_1�
<basemodel/batch_normalization_2/batchnorm_1/ReadVariableOp_1ReadVariableOpCbasemodel_batch_normalization_2_batchnorm_readvariableop_1_resource*
_output_shapes
:T*
dtype02>
<basemodel/batch_normalization_2/batchnorm_1/ReadVariableOp_1�
1basemodel/batch_normalization_2/batchnorm_1/mul_2MulDbasemodel/batch_normalization_2/batchnorm_1/ReadVariableOp_1:value:03basemodel/batch_normalization_2/batchnorm_1/mul:z:0*
T0*
_output_shapes
:T23
1basemodel/batch_normalization_2/batchnorm_1/mul_2�
<basemodel/batch_normalization_2/batchnorm_1/ReadVariableOp_2ReadVariableOpCbasemodel_batch_normalization_2_batchnorm_readvariableop_2_resource*
_output_shapes
:T*
dtype02>
<basemodel/batch_normalization_2/batchnorm_1/ReadVariableOp_2�
/basemodel/batch_normalization_2/batchnorm_1/subSubDbasemodel/batch_normalization_2/batchnorm_1/ReadVariableOp_2:value:05basemodel/batch_normalization_2/batchnorm_1/mul_2:z:0*
T0*
_output_shapes
:T21
/basemodel/batch_normalization_2/batchnorm_1/sub�
1basemodel/batch_normalization_2/batchnorm_1/add_1AddV25basemodel/batch_normalization_2/batchnorm_1/mul_1:z:03basemodel/batch_normalization_2/batchnorm_1/sub:z:0*
T0*'
_output_shapes
:���������T23
1basemodel/batch_normalization_2/batchnorm_1/add_1�
&basemodel/dense_activation_1/Sigmoid_1Sigmoid5basemodel/batch_normalization_2/batchnorm_1/add_1:z:0*
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
: *
dtype027
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp�
&stream_0_conv_1/kernel/Regularizer/AbsAbs=stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
: 2(
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
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpEbasemodel_stream_0_conv_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: @*
dtype02:
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp�
)stream_0_conv_2/kernel/Regularizer/SquareSquare@stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
: @2+
)stream_0_conv_2/kernel/Regularizer/Square�
(stream_0_conv_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_0_conv_2/kernel/Regularizer/Const�
&stream_0_conv_2/kernel/Regularizer/SumSum-stream_0_conv_2/kernel/Regularizer/Square:y:01stream_0_conv_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_2/kernel/Regularizer/Sum�
(stream_0_conv_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2*
(stream_0_conv_2/kernel/Regularizer/mul/x�
&stream_0_conv_2/kernel/Regularizer/mulMul1stream_0_conv_2/kernel/Regularizer/mul/x:output:0/stream_0_conv_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_2/kernel/Regularizer/mul�
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp0basemodel_dense_1_matmul_readvariableop_resource*
_output_shapes

:@T*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOp�
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:@T2 
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

Identity�
NoOpNoOp7^basemodel/batch_normalization/batchnorm/ReadVariableOp9^basemodel/batch_normalization/batchnorm/ReadVariableOp_19^basemodel/batch_normalization/batchnorm/ReadVariableOp_2;^basemodel/batch_normalization/batchnorm/mul/ReadVariableOp9^basemodel/batch_normalization/batchnorm_1/ReadVariableOp;^basemodel/batch_normalization/batchnorm_1/ReadVariableOp_1;^basemodel/batch_normalization/batchnorm_1/ReadVariableOp_2=^basemodel/batch_normalization/batchnorm_1/mul/ReadVariableOp9^basemodel/batch_normalization_1/batchnorm/ReadVariableOp;^basemodel/batch_normalization_1/batchnorm/ReadVariableOp_1;^basemodel/batch_normalization_1/batchnorm/ReadVariableOp_2=^basemodel/batch_normalization_1/batchnorm/mul/ReadVariableOp;^basemodel/batch_normalization_1/batchnorm_1/ReadVariableOp=^basemodel/batch_normalization_1/batchnorm_1/ReadVariableOp_1=^basemodel/batch_normalization_1/batchnorm_1/ReadVariableOp_2?^basemodel/batch_normalization_1/batchnorm_1/mul/ReadVariableOp9^basemodel/batch_normalization_2/batchnorm/ReadVariableOp;^basemodel/batch_normalization_2/batchnorm/ReadVariableOp_1;^basemodel/batch_normalization_2/batchnorm/ReadVariableOp_2=^basemodel/batch_normalization_2/batchnorm/mul/ReadVariableOp;^basemodel/batch_normalization_2/batchnorm_1/ReadVariableOp=^basemodel/batch_normalization_2/batchnorm_1/ReadVariableOp_1=^basemodel/batch_normalization_2/batchnorm_1/ReadVariableOp_2?^basemodel/batch_normalization_2/batchnorm_1/mul/ReadVariableOp)^basemodel/dense_1/BiasAdd/ReadVariableOp+^basemodel/dense_1/BiasAdd_1/ReadVariableOp(^basemodel/dense_1/MatMul/ReadVariableOp*^basemodel/dense_1/MatMul_1/ReadVariableOp1^basemodel/stream_0_conv_1/BiasAdd/ReadVariableOp3^basemodel/stream_0_conv_1/BiasAdd_1/ReadVariableOp=^basemodel/stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp?^basemodel/stream_0_conv_1/conv1d_1/ExpandDims_1/ReadVariableOp1^basemodel/stream_0_conv_2/BiasAdd/ReadVariableOp3^basemodel/stream_0_conv_2/BiasAdd_1/ReadVariableOp=^basemodel/stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOp?^basemodel/stream_0_conv_2/conv1d_1/ExpandDims_1/ReadVariableOp.^dense_1/kernel/Regularizer/Abs/ReadVariableOp6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp9^stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*g
_input_shapesV
T:����������:����������: : : : : : : : : : : : : : : : : : 2p
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
>basemodel/batch_normalization_2/batchnorm_1/mul/ReadVariableOp>basemodel/batch_normalization_2/batchnorm_1/mul/ReadVariableOp2T
(basemodel/dense_1/BiasAdd/ReadVariableOp(basemodel/dense_1/BiasAdd/ReadVariableOp2X
*basemodel/dense_1/BiasAdd_1/ReadVariableOp*basemodel/dense_1/BiasAdd_1/ReadVariableOp2R
'basemodel/dense_1/MatMul/ReadVariableOp'basemodel/dense_1/MatMul/ReadVariableOp2V
)basemodel/dense_1/MatMul_1/ReadVariableOp)basemodel/dense_1/MatMul_1/ReadVariableOp2d
0basemodel/stream_0_conv_1/BiasAdd/ReadVariableOp0basemodel/stream_0_conv_1/BiasAdd/ReadVariableOp2h
2basemodel/stream_0_conv_1/BiasAdd_1/ReadVariableOp2basemodel/stream_0_conv_1/BiasAdd_1/ReadVariableOp2|
<basemodel/stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp<basemodel/stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp2�
>basemodel/stream_0_conv_1/conv1d_1/ExpandDims_1/ReadVariableOp>basemodel/stream_0_conv_1/conv1d_1/ExpandDims_1/ReadVariableOp2d
0basemodel/stream_0_conv_2/BiasAdd/ReadVariableOp0basemodel/stream_0_conv_2/BiasAdd/ReadVariableOp2h
2basemodel/stream_0_conv_2/BiasAdd_1/ReadVariableOp2basemodel/stream_0_conv_2/BiasAdd_1/ReadVariableOp2|
<basemodel/stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOp<basemodel/stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOp2�
>basemodel/stream_0_conv_2/conv1d_1/ExpandDims_1/ReadVariableOp>basemodel/stream_0_conv_2/conv1d_1/ExpandDims_1/ReadVariableOp2^
-dense_1/kernel/Regularizer/Abs/ReadVariableOp-dense_1/kernel/Regularizer/Abs/ReadVariableOp2n
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp2t
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp:V R
,
_output_shapes
:����������
"
_user_specified_name
inputs/0:VR
,
_output_shapes
:����������
"
_user_specified_name
inputs/1
�+
�
P__inference_batch_normalization_layer_call_and_return_conditional_losses_5544695

inputs5
'assignmovingavg_readvariableop_resource: 7
)assignmovingavg_1_readvariableop_resource: 3
%batchnorm_mul_readvariableop_resource: /
!batchnorm_readvariableop_resource: 
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
: *
	keep_dims(2
moments/mean�
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
: 2
moments/StopGradient�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*,
_output_shapes
:���������� 2
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
: *
	keep_dims(2
moments/variance�
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/Squeeze�
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
: *
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
: *
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
: 2
AssignMovingAvg/sub�
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
: 2
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
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
: 2
AssignMovingAvg_1/sub�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
: 2
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
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/Rsqrt�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/mul/ReadVariableOp�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/mul{
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*,
_output_shapes
:���������� 2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2�
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp�
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*,
_output_shapes
:���������� 2
batchnorm/add_1s
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*,
_output_shapes
:���������� 2

Identity�
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������� : : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:T P
,
_output_shapes
:���������� 
 
_user_specified_nameinputs
�
�
L__inference_stream_0_conv_1_layer_call_and_return_conditional_losses_5541318

inputsA
+conv1d_expanddims_1_readvariableop_resource: -
biasadd_readvariableop_resource: 
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
:����������2
conv1d/ExpandDims�
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
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
: 2
conv1d/ExpandDims_1�
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:���������� *
paddingSAME*
strides
2
conv1d�
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:���������� *
squeeze_dims

���������2
conv1d/Squeeze�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������� 2	
BiasAdd�
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype027
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp�
&stream_0_conv_1/kernel/Regularizer/AbsAbs=stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
: 2(
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
:���������� 2

Identity�
NoOpNoOp^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp2n
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�	
q
E__inference_distance_layer_call_and_return_conditional_losses_5544500
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
�*
�
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_5545162

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
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_5540959

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
��
�
F__inference_basemodel_layer_call_and_return_conditional_losses_5543857

inputsQ
;stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource: =
/stream_0_conv_1_biasadd_readvariableop_resource: C
5batch_normalization_batchnorm_readvariableop_resource: G
9batch_normalization_batchnorm_mul_readvariableop_resource: E
7batch_normalization_batchnorm_readvariableop_1_resource: E
7batch_normalization_batchnorm_readvariableop_2_resource: Q
;stream_0_conv_2_conv1d_expanddims_1_readvariableop_resource: @=
/stream_0_conv_2_biasadd_readvariableop_resource:@E
7batch_normalization_1_batchnorm_readvariableop_resource:@I
;batch_normalization_1_batchnorm_mul_readvariableop_resource:@G
9batch_normalization_1_batchnorm_readvariableop_1_resource:@G
9batch_normalization_1_batchnorm_readvariableop_2_resource:@8
&dense_1_matmul_readvariableop_resource:@T5
'dense_1_biasadd_readvariableop_resource:TE
7batch_normalization_2_batchnorm_readvariableop_resource:TI
;batch_normalization_2_batchnorm_mul_readvariableop_resource:TG
9batch_normalization_2_batchnorm_readvariableop_1_resource:TG
9batch_normalization_2_batchnorm_readvariableop_2_resource:T
identity��,batch_normalization/batchnorm/ReadVariableOp�.batch_normalization/batchnorm/ReadVariableOp_1�.batch_normalization/batchnorm/ReadVariableOp_2�0batch_normalization/batchnorm/mul/ReadVariableOp�.batch_normalization_1/batchnorm/ReadVariableOp�0batch_normalization_1/batchnorm/ReadVariableOp_1�0batch_normalization_1/batchnorm/ReadVariableOp_2�2batch_normalization_1/batchnorm/mul/ReadVariableOp�.batch_normalization_2/batchnorm/ReadVariableOp�0batch_normalization_2/batchnorm/ReadVariableOp_1�0batch_normalization_2/batchnorm/ReadVariableOp_2�2batch_normalization_2/batchnorm/mul/ReadVariableOp�dense_1/BiasAdd/ReadVariableOp�dense_1/MatMul/ReadVariableOp�-dense_1/kernel/Regularizer/Abs/ReadVariableOp�&stream_0_conv_1/BiasAdd/ReadVariableOp�2stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp�5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp�&stream_0_conv_2/BiasAdd/ReadVariableOp�2stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOp�8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp�
stream_0_input_drop/IdentityIdentityinputs*
T0*,
_output_shapes
:����������2
stream_0_input_drop/Identity�
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
:����������2#
!stream_0_conv_1/conv1d/ExpandDims�
2stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp;stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
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
: 2%
#stream_0_conv_1/conv1d/ExpandDims_1�
stream_0_conv_1/conv1dConv2D*stream_0_conv_1/conv1d/ExpandDims:output:0,stream_0_conv_1/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:���������� *
paddingSAME*
strides
2
stream_0_conv_1/conv1d�
stream_0_conv_1/conv1d/SqueezeSqueezestream_0_conv_1/conv1d:output:0*
T0*,
_output_shapes
:���������� *
squeeze_dims

���������2 
stream_0_conv_1/conv1d/Squeeze�
&stream_0_conv_1/BiasAdd/ReadVariableOpReadVariableOp/stream_0_conv_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02(
&stream_0_conv_1/BiasAdd/ReadVariableOp�
stream_0_conv_1/BiasAddBiasAdd'stream_0_conv_1/conv1d/Squeeze:output:0.stream_0_conv_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������� 2
stream_0_conv_1/BiasAdd�
,batch_normalization/batchnorm/ReadVariableOpReadVariableOp5batch_normalization_batchnorm_readvariableop_resource*
_output_shapes
: *
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
: 2#
!batch_normalization/batchnorm/add�
#batch_normalization/batchnorm/RsqrtRsqrt%batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes
: 2%
#batch_normalization/batchnorm/Rsqrt�
0batch_normalization/batchnorm/mul/ReadVariableOpReadVariableOp9batch_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype022
0batch_normalization/batchnorm/mul/ReadVariableOp�
!batch_normalization/batchnorm/mulMul'batch_normalization/batchnorm/Rsqrt:y:08batch_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2#
!batch_normalization/batchnorm/mul�
#batch_normalization/batchnorm/mul_1Mul stream_0_conv_1/BiasAdd:output:0%batch_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:���������� 2%
#batch_normalization/batchnorm/mul_1�
.batch_normalization/batchnorm/ReadVariableOp_1ReadVariableOp7batch_normalization_batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype020
.batch_normalization/batchnorm/ReadVariableOp_1�
#batch_normalization/batchnorm/mul_2Mul6batch_normalization/batchnorm/ReadVariableOp_1:value:0%batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes
: 2%
#batch_normalization/batchnorm/mul_2�
.batch_normalization/batchnorm/ReadVariableOp_2ReadVariableOp7batch_normalization_batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype020
.batch_normalization/batchnorm/ReadVariableOp_2�
!batch_normalization/batchnorm/subSub6batch_normalization/batchnorm/ReadVariableOp_2:value:0'batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2#
!batch_normalization/batchnorm/sub�
#batch_normalization/batchnorm/add_1AddV2'batch_normalization/batchnorm/mul_1:z:0%batch_normalization/batchnorm/sub:z:0*
T0*,
_output_shapes
:���������� 2%
#batch_normalization/batchnorm/add_1�
activation/ReluRelu'batch_normalization/batchnorm/add_1:z:0*
T0*,
_output_shapes
:���������� 2
activation/Relu�
stream_0_drop_1/IdentityIdentityactivation/Relu:activations:0*
T0*,
_output_shapes
:���������� 2
stream_0_drop_1/Identity�
%stream_0_conv_2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2'
%stream_0_conv_2/conv1d/ExpandDims/dim�
!stream_0_conv_2/conv1d/ExpandDims
ExpandDims!stream_0_drop_1/Identity:output:0.stream_0_conv_2/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:���������� 2#
!stream_0_conv_2/conv1d/ExpandDims�
2stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp;stream_0_conv_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: @*
dtype024
2stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOp�
'stream_0_conv_2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2)
'stream_0_conv_2/conv1d/ExpandDims_1/dim�
#stream_0_conv_2/conv1d/ExpandDims_1
ExpandDims:stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOp:value:00stream_0_conv_2/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: @2%
#stream_0_conv_2/conv1d/ExpandDims_1�
stream_0_conv_2/conv1dConv2D*stream_0_conv_2/conv1d/ExpandDims:output:0,stream_0_conv_2/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:����������@*
paddingSAME*
strides
2
stream_0_conv_2/conv1d�
stream_0_conv_2/conv1d/SqueezeSqueezestream_0_conv_2/conv1d:output:0*
T0*,
_output_shapes
:����������@*
squeeze_dims

���������2 
stream_0_conv_2/conv1d/Squeeze�
&stream_0_conv_2/BiasAdd/ReadVariableOpReadVariableOp/stream_0_conv_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02(
&stream_0_conv_2/BiasAdd/ReadVariableOp�
stream_0_conv_2/BiasAddBiasAdd'stream_0_conv_2/conv1d/Squeeze:output:0.stream_0_conv_2/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������@2
stream_0_conv_2/BiasAdd�
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
%batch_normalization_1/batchnorm/mul_1Mul stream_0_conv_2/BiasAdd:output:0'batch_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:����������@2'
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
:����������@2'
%batch_normalization_1/batchnorm/add_1�
activation_1/ReluRelu)batch_normalization_1/batchnorm/add_1:z:0*
T0*,
_output_shapes
:����������@2
activation_1/Relu�
stream_0_drop_2/IdentityIdentityactivation_1/Relu:activations:0*
T0*,
_output_shapes
:����������@2
stream_0_drop_2/Identity�
/global_average_pooling1d/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :21
/global_average_pooling1d/Mean/reduction_indices�
global_average_pooling1d/MeanMean!stream_0_drop_2/Identity:output:08global_average_pooling1d/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:���������@2
global_average_pooling1d/Mean�
concatenate/concat/concat_dimConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate/concat/concat_dim�
concatenate/concat/concatIdentity&global_average_pooling1d/Mean:output:0*
T0*'
_output_shapes
:���������@2
concatenate/concat/concat�
dense_1_dropout/IdentityIdentity"concatenate/concat/concat:output:0*
T0*'
_output_shapes
:���������@2
dense_1_dropout/Identity�
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:@T*
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
.batch_normalization_2/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_2_batchnorm_readvariableop_resource*
_output_shapes
:T*
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
:T2%
#batch_normalization_2/batchnorm/add�
%batch_normalization_2/batchnorm/RsqrtRsqrt'batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes
:T2'
%batch_normalization_2/batchnorm/Rsqrt�
2batch_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes
:T*
dtype024
2batch_normalization_2/batchnorm/mul/ReadVariableOp�
#batch_normalization_2/batchnorm/mulMul)batch_normalization_2/batchnorm/Rsqrt:y:0:batch_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:T2%
#batch_normalization_2/batchnorm/mul�
%batch_normalization_2/batchnorm/mul_1Muldense_1/BiasAdd:output:0'batch_normalization_2/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������T2'
%batch_normalization_2/batchnorm/mul_1�
0batch_normalization_2/batchnorm/ReadVariableOp_1ReadVariableOp9batch_normalization_2_batchnorm_readvariableop_1_resource*
_output_shapes
:T*
dtype022
0batch_normalization_2/batchnorm/ReadVariableOp_1�
%batch_normalization_2/batchnorm/mul_2Mul8batch_normalization_2/batchnorm/ReadVariableOp_1:value:0'batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes
:T2'
%batch_normalization_2/batchnorm/mul_2�
0batch_normalization_2/batchnorm/ReadVariableOp_2ReadVariableOp9batch_normalization_2_batchnorm_readvariableop_2_resource*
_output_shapes
:T*
dtype022
0batch_normalization_2/batchnorm/ReadVariableOp_2�
#batch_normalization_2/batchnorm/subSub8batch_normalization_2/batchnorm/ReadVariableOp_2:value:0)batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes
:T2%
#batch_normalization_2/batchnorm/sub�
%batch_normalization_2/batchnorm/add_1AddV2)batch_normalization_2/batchnorm/mul_1:z:0'batch_normalization_2/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������T2'
%batch_normalization_2/batchnorm/add_1�
dense_activation_1/SigmoidSigmoid)batch_normalization_2/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������T2
dense_activation_1/Sigmoid�
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp;stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype027
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp�
&stream_0_conv_1/kernel/Regularizer/AbsAbs=stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
: 2(
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
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;stream_0_conv_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: @*
dtype02:
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp�
)stream_0_conv_2/kernel/Regularizer/SquareSquare@stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
: @2+
)stream_0_conv_2/kernel/Regularizer/Square�
(stream_0_conv_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_0_conv_2/kernel/Regularizer/Const�
&stream_0_conv_2/kernel/Regularizer/SumSum-stream_0_conv_2/kernel/Regularizer/Square:y:01stream_0_conv_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_2/kernel/Regularizer/Sum�
(stream_0_conv_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2*
(stream_0_conv_2/kernel/Regularizer/mul/x�
&stream_0_conv_2/kernel/Regularizer/mulMul1stream_0_conv_2/kernel/Regularizer/mul/x:output:0/stream_0_conv_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_2/kernel/Regularizer/mul�
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:@T*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOp�
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:@T2 
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

Identity�
NoOpNoOp-^batch_normalization/batchnorm/ReadVariableOp/^batch_normalization/batchnorm/ReadVariableOp_1/^batch_normalization/batchnorm/ReadVariableOp_21^batch_normalization/batchnorm/mul/ReadVariableOp/^batch_normalization_1/batchnorm/ReadVariableOp1^batch_normalization_1/batchnorm/ReadVariableOp_11^batch_normalization_1/batchnorm/ReadVariableOp_23^batch_normalization_1/batchnorm/mul/ReadVariableOp/^batch_normalization_2/batchnorm/ReadVariableOp1^batch_normalization_2/batchnorm/ReadVariableOp_11^batch_normalization_2/batchnorm/ReadVariableOp_23^batch_normalization_2/batchnorm/mul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp.^dense_1/kernel/Regularizer/Abs/ReadVariableOp'^stream_0_conv_1/BiasAdd/ReadVariableOp3^stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp'^stream_0_conv_2/BiasAdd/ReadVariableOp3^stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOp9^stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:����������: : : : : : : : : : : : : : : : : : 2\
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
2batch_normalization_2/batchnorm/mul/ReadVariableOp2batch_normalization_2/batchnorm/mul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2^
-dense_1/kernel/Regularizer/Abs/ReadVariableOp-dense_1/kernel/Regularizer/Abs/ReadVariableOp2P
&stream_0_conv_1/BiasAdd/ReadVariableOp&stream_0_conv_1/BiasAdd/ReadVariableOp2h
2stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp2stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp2n
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp2P
&stream_0_conv_2/BiasAdd/ReadVariableOp&stream_0_conv_2/BiasAdd/ReadVariableOp2h
2stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOp2stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOp2t
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
V
*__inference_distance_layer_call_fn_5544518
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
E__inference_distance_layer_call_and_return_conditional_losses_55423542
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
�
H
,__inference_activation_layer_call_fn_5544757

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
:���������� * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_activation_layer_call_and_return_conditional_losses_55413582
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:���������� 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:���������� :T P
,
_output_shapes
:���������� 
 
_user_specified_nameinputs
��
�
F__inference_basemodel_layer_call_and_return_conditional_losses_5542285

inputsQ
;stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource: =
/stream_0_conv_1_biasadd_readvariableop_resource: C
5batch_normalization_batchnorm_readvariableop_resource: G
9batch_normalization_batchnorm_mul_readvariableop_resource: E
7batch_normalization_batchnorm_readvariableop_1_resource: E
7batch_normalization_batchnorm_readvariableop_2_resource: Q
;stream_0_conv_2_conv1d_expanddims_1_readvariableop_resource: @=
/stream_0_conv_2_biasadd_readvariableop_resource:@E
7batch_normalization_1_batchnorm_readvariableop_resource:@I
;batch_normalization_1_batchnorm_mul_readvariableop_resource:@G
9batch_normalization_1_batchnorm_readvariableop_1_resource:@G
9batch_normalization_1_batchnorm_readvariableop_2_resource:@8
&dense_1_matmul_readvariableop_resource:@T5
'dense_1_biasadd_readvariableop_resource:TE
7batch_normalization_2_batchnorm_readvariableop_resource:TI
;batch_normalization_2_batchnorm_mul_readvariableop_resource:TG
9batch_normalization_2_batchnorm_readvariableop_1_resource:TG
9batch_normalization_2_batchnorm_readvariableop_2_resource:T
identity��,batch_normalization/batchnorm/ReadVariableOp�.batch_normalization/batchnorm/ReadVariableOp_1�.batch_normalization/batchnorm/ReadVariableOp_2�0batch_normalization/batchnorm/mul/ReadVariableOp�.batch_normalization_1/batchnorm/ReadVariableOp�0batch_normalization_1/batchnorm/ReadVariableOp_1�0batch_normalization_1/batchnorm/ReadVariableOp_2�2batch_normalization_1/batchnorm/mul/ReadVariableOp�.batch_normalization_2/batchnorm/ReadVariableOp�0batch_normalization_2/batchnorm/ReadVariableOp_1�0batch_normalization_2/batchnorm/ReadVariableOp_2�2batch_normalization_2/batchnorm/mul/ReadVariableOp�dense_1/BiasAdd/ReadVariableOp�dense_1/MatMul/ReadVariableOp�-dense_1/kernel/Regularizer/Abs/ReadVariableOp�&stream_0_conv_1/BiasAdd/ReadVariableOp�2stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp�5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp�&stream_0_conv_2/BiasAdd/ReadVariableOp�2stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOp�8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp�
stream_0_input_drop/IdentityIdentityinputs*
T0*,
_output_shapes
:����������2
stream_0_input_drop/Identity�
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
:����������2#
!stream_0_conv_1/conv1d/ExpandDims�
2stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp;stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
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
: 2%
#stream_0_conv_1/conv1d/ExpandDims_1�
stream_0_conv_1/conv1dConv2D*stream_0_conv_1/conv1d/ExpandDims:output:0,stream_0_conv_1/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:���������� *
paddingSAME*
strides
2
stream_0_conv_1/conv1d�
stream_0_conv_1/conv1d/SqueezeSqueezestream_0_conv_1/conv1d:output:0*
T0*,
_output_shapes
:���������� *
squeeze_dims

���������2 
stream_0_conv_1/conv1d/Squeeze�
&stream_0_conv_1/BiasAdd/ReadVariableOpReadVariableOp/stream_0_conv_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02(
&stream_0_conv_1/BiasAdd/ReadVariableOp�
stream_0_conv_1/BiasAddBiasAdd'stream_0_conv_1/conv1d/Squeeze:output:0.stream_0_conv_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������� 2
stream_0_conv_1/BiasAdd�
,batch_normalization/batchnorm/ReadVariableOpReadVariableOp5batch_normalization_batchnorm_readvariableop_resource*
_output_shapes
: *
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
: 2#
!batch_normalization/batchnorm/add�
#batch_normalization/batchnorm/RsqrtRsqrt%batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes
: 2%
#batch_normalization/batchnorm/Rsqrt�
0batch_normalization/batchnorm/mul/ReadVariableOpReadVariableOp9batch_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype022
0batch_normalization/batchnorm/mul/ReadVariableOp�
!batch_normalization/batchnorm/mulMul'batch_normalization/batchnorm/Rsqrt:y:08batch_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2#
!batch_normalization/batchnorm/mul�
#batch_normalization/batchnorm/mul_1Mul stream_0_conv_1/BiasAdd:output:0%batch_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:���������� 2%
#batch_normalization/batchnorm/mul_1�
.batch_normalization/batchnorm/ReadVariableOp_1ReadVariableOp7batch_normalization_batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype020
.batch_normalization/batchnorm/ReadVariableOp_1�
#batch_normalization/batchnorm/mul_2Mul6batch_normalization/batchnorm/ReadVariableOp_1:value:0%batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes
: 2%
#batch_normalization/batchnorm/mul_2�
.batch_normalization/batchnorm/ReadVariableOp_2ReadVariableOp7batch_normalization_batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype020
.batch_normalization/batchnorm/ReadVariableOp_2�
!batch_normalization/batchnorm/subSub6batch_normalization/batchnorm/ReadVariableOp_2:value:0'batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2#
!batch_normalization/batchnorm/sub�
#batch_normalization/batchnorm/add_1AddV2'batch_normalization/batchnorm/mul_1:z:0%batch_normalization/batchnorm/sub:z:0*
T0*,
_output_shapes
:���������� 2%
#batch_normalization/batchnorm/add_1�
activation/ReluRelu'batch_normalization/batchnorm/add_1:z:0*
T0*,
_output_shapes
:���������� 2
activation/Relu�
stream_0_drop_1/IdentityIdentityactivation/Relu:activations:0*
T0*,
_output_shapes
:���������� 2
stream_0_drop_1/Identity�
%stream_0_conv_2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2'
%stream_0_conv_2/conv1d/ExpandDims/dim�
!stream_0_conv_2/conv1d/ExpandDims
ExpandDims!stream_0_drop_1/Identity:output:0.stream_0_conv_2/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:���������� 2#
!stream_0_conv_2/conv1d/ExpandDims�
2stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp;stream_0_conv_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: @*
dtype024
2stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOp�
'stream_0_conv_2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2)
'stream_0_conv_2/conv1d/ExpandDims_1/dim�
#stream_0_conv_2/conv1d/ExpandDims_1
ExpandDims:stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOp:value:00stream_0_conv_2/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: @2%
#stream_0_conv_2/conv1d/ExpandDims_1�
stream_0_conv_2/conv1dConv2D*stream_0_conv_2/conv1d/ExpandDims:output:0,stream_0_conv_2/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:����������@*
paddingSAME*
strides
2
stream_0_conv_2/conv1d�
stream_0_conv_2/conv1d/SqueezeSqueezestream_0_conv_2/conv1d:output:0*
T0*,
_output_shapes
:����������@*
squeeze_dims

���������2 
stream_0_conv_2/conv1d/Squeeze�
&stream_0_conv_2/BiasAdd/ReadVariableOpReadVariableOp/stream_0_conv_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02(
&stream_0_conv_2/BiasAdd/ReadVariableOp�
stream_0_conv_2/BiasAddBiasAdd'stream_0_conv_2/conv1d/Squeeze:output:0.stream_0_conv_2/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������@2
stream_0_conv_2/BiasAdd�
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
%batch_normalization_1/batchnorm/mul_1Mul stream_0_conv_2/BiasAdd:output:0'batch_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:����������@2'
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
:����������@2'
%batch_normalization_1/batchnorm/add_1�
activation_1/ReluRelu)batch_normalization_1/batchnorm/add_1:z:0*
T0*,
_output_shapes
:����������@2
activation_1/Relu�
stream_0_drop_2/IdentityIdentityactivation_1/Relu:activations:0*
T0*,
_output_shapes
:����������@2
stream_0_drop_2/Identity�
/global_average_pooling1d/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :21
/global_average_pooling1d/Mean/reduction_indices�
global_average_pooling1d/MeanMean!stream_0_drop_2/Identity:output:08global_average_pooling1d/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:���������@2
global_average_pooling1d/Mean�
concatenate/concat/concat_dimConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate/concat/concat_dim�
concatenate/concat/concatIdentity&global_average_pooling1d/Mean:output:0*
T0*'
_output_shapes
:���������@2
concatenate/concat/concat�
dense_1_dropout/IdentityIdentity"concatenate/concat/concat:output:0*
T0*'
_output_shapes
:���������@2
dense_1_dropout/Identity�
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:@T*
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
.batch_normalization_2/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_2_batchnorm_readvariableop_resource*
_output_shapes
:T*
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
:T2%
#batch_normalization_2/batchnorm/add�
%batch_normalization_2/batchnorm/RsqrtRsqrt'batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes
:T2'
%batch_normalization_2/batchnorm/Rsqrt�
2batch_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes
:T*
dtype024
2batch_normalization_2/batchnorm/mul/ReadVariableOp�
#batch_normalization_2/batchnorm/mulMul)batch_normalization_2/batchnorm/Rsqrt:y:0:batch_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:T2%
#batch_normalization_2/batchnorm/mul�
%batch_normalization_2/batchnorm/mul_1Muldense_1/BiasAdd:output:0'batch_normalization_2/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������T2'
%batch_normalization_2/batchnorm/mul_1�
0batch_normalization_2/batchnorm/ReadVariableOp_1ReadVariableOp9batch_normalization_2_batchnorm_readvariableop_1_resource*
_output_shapes
:T*
dtype022
0batch_normalization_2/batchnorm/ReadVariableOp_1�
%batch_normalization_2/batchnorm/mul_2Mul8batch_normalization_2/batchnorm/ReadVariableOp_1:value:0'batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes
:T2'
%batch_normalization_2/batchnorm/mul_2�
0batch_normalization_2/batchnorm/ReadVariableOp_2ReadVariableOp9batch_normalization_2_batchnorm_readvariableop_2_resource*
_output_shapes
:T*
dtype022
0batch_normalization_2/batchnorm/ReadVariableOp_2�
#batch_normalization_2/batchnorm/subSub8batch_normalization_2/batchnorm/ReadVariableOp_2:value:0)batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes
:T2%
#batch_normalization_2/batchnorm/sub�
%batch_normalization_2/batchnorm/add_1AddV2)batch_normalization_2/batchnorm/mul_1:z:0'batch_normalization_2/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������T2'
%batch_normalization_2/batchnorm/add_1�
dense_activation_1/SigmoidSigmoid)batch_normalization_2/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������T2
dense_activation_1/Sigmoid�
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp;stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype027
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp�
&stream_0_conv_1/kernel/Regularizer/AbsAbs=stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
: 2(
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
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;stream_0_conv_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: @*
dtype02:
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp�
)stream_0_conv_2/kernel/Regularizer/SquareSquare@stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
: @2+
)stream_0_conv_2/kernel/Regularizer/Square�
(stream_0_conv_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_0_conv_2/kernel/Regularizer/Const�
&stream_0_conv_2/kernel/Regularizer/SumSum-stream_0_conv_2/kernel/Regularizer/Square:y:01stream_0_conv_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_2/kernel/Regularizer/Sum�
(stream_0_conv_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2*
(stream_0_conv_2/kernel/Regularizer/mul/x�
&stream_0_conv_2/kernel/Regularizer/mulMul1stream_0_conv_2/kernel/Regularizer/mul/x:output:0/stream_0_conv_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_2/kernel/Regularizer/mul�
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:@T*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOp�
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:@T2 
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

Identity�
NoOpNoOp-^batch_normalization/batchnorm/ReadVariableOp/^batch_normalization/batchnorm/ReadVariableOp_1/^batch_normalization/batchnorm/ReadVariableOp_21^batch_normalization/batchnorm/mul/ReadVariableOp/^batch_normalization_1/batchnorm/ReadVariableOp1^batch_normalization_1/batchnorm/ReadVariableOp_11^batch_normalization_1/batchnorm/ReadVariableOp_23^batch_normalization_1/batchnorm/mul/ReadVariableOp/^batch_normalization_2/batchnorm/ReadVariableOp1^batch_normalization_2/batchnorm/ReadVariableOp_11^batch_normalization_2/batchnorm/ReadVariableOp_23^batch_normalization_2/batchnorm/mul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp.^dense_1/kernel/Regularizer/Abs/ReadVariableOp'^stream_0_conv_1/BiasAdd/ReadVariableOp3^stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp'^stream_0_conv_2/BiasAdd/ReadVariableOp3^stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOp9^stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:����������: : : : : : : : : : : : : : : : : : 2\
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
2batch_normalization_2/batchnorm/mul/ReadVariableOp2batch_normalization_2/batchnorm/mul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2^
-dense_1/kernel/Regularizer/Abs/ReadVariableOp-dense_1/kernel/Regularizer/Abs/ReadVariableOp2P
&stream_0_conv_1/BiasAdd/ReadVariableOp&stream_0_conv_1/BiasAdd/ReadVariableOp2h
2stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp2stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp2n
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp2P
&stream_0_conv_2/BiasAdd/ReadVariableOp&stream_0_conv_2/BiasAdd/ReadVariableOp2h
2stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOp2stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOp2t
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
5__inference_batch_normalization_layer_call_fn_5544747

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_55417822
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:���������� 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������� : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:���������� 
 
_user_specified_nameinputs
�
�
+__inference_basemodel_layer_call_fn_5541555
inputs_0
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: 
	unknown_5: @
	unknown_6:@
	unknown_7:@
	unknown_8:@
	unknown_9:@

unknown_10:@

unknown_11:@T

unknown_12:T

unknown_13:T

unknown_14:T

unknown_15:T

unknown_16:T
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������T*4
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_basemodel_layer_call_and_return_conditional_losses_55415162
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
_construction_contextkEagerRuntime*O
_input_shapes>
<:����������: : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
,
_output_shapes
:����������
"
_user_specified_name
inputs_0
�n
�
F__inference_basemodel_layer_call_and_return_conditional_losses_5542168
inputs_0-
stream_0_conv_1_5542099: %
stream_0_conv_1_5542101: )
batch_normalization_5542104: )
batch_normalization_5542106: )
batch_normalization_5542108: )
batch_normalization_5542110: -
stream_0_conv_2_5542115: @%
stream_0_conv_2_5542117:@+
batch_normalization_1_5542120:@+
batch_normalization_1_5542122:@+
batch_normalization_1_5542124:@+
batch_normalization_1_5542126:@!
dense_1_5542134:@T
dense_1_5542136:T+
batch_normalization_2_5542139:T+
batch_normalization_2_5542141:T+
batch_normalization_2_5542143:T+
batch_normalization_2_5542145:T
identity��+batch_normalization/StatefulPartitionedCall�-batch_normalization_1/StatefulPartitionedCall�-batch_normalization_2/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�-dense_1/kernel/Regularizer/Abs/ReadVariableOp�'dense_1_dropout/StatefulPartitionedCall�'stream_0_conv_1/StatefulPartitionedCall�5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp�'stream_0_conv_2/StatefulPartitionedCall�8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp�'stream_0_drop_1/StatefulPartitionedCall�'stream_0_drop_2/StatefulPartitionedCall�+stream_0_input_drop/StatefulPartitionedCall�
+stream_0_input_drop/StatefulPartitionedCallStatefulPartitionedCallinputs_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *Y
fTRR
P__inference_stream_0_input_drop_layer_call_and_return_conditional_losses_55418232-
+stream_0_input_drop/StatefulPartitionedCall�
'stream_0_conv_1/StatefulPartitionedCallStatefulPartitionedCall4stream_0_input_drop/StatefulPartitionedCall:output:0stream_0_conv_1_5542099stream_0_conv_1_5542101*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_stream_0_conv_1_layer_call_and_return_conditional_losses_55413182)
'stream_0_conv_1/StatefulPartitionedCall�
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall0stream_0_conv_1/StatefulPartitionedCall:output:0batch_normalization_5542104batch_normalization_5542106batch_normalization_5542108batch_normalization_5542110*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_55417822-
+batch_normalization/StatefulPartitionedCall�
activation/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������� * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_activation_layer_call_and_return_conditional_losses_55413582
activation/PartitionedCall�
'stream_0_drop_1/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0,^stream_0_input_drop/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������� * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_stream_0_drop_1_layer_call_and_return_conditional_losses_55417242)
'stream_0_drop_1/StatefulPartitionedCall�
'stream_0_conv_2/StatefulPartitionedCallStatefulPartitionedCall0stream_0_drop_1/StatefulPartitionedCall:output:0stream_0_conv_2_5542115stream_0_conv_2_5542117*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_stream_0_conv_2_layer_call_and_return_conditional_losses_55413882)
'stream_0_conv_2/StatefulPartitionedCall�
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall0stream_0_conv_2/StatefulPartitionedCall:output:0batch_normalization_1_5542120batch_normalization_1_5542122batch_normalization_1_5542124batch_normalization_1_5542126*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *[
fVRT
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_55416832/
-batch_normalization_1/StatefulPartitionedCall�
activation_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_activation_1_layer_call_and_return_conditional_losses_55414282
activation_1/PartitionedCall�
'stream_0_drop_2/StatefulPartitionedCallStatefulPartitionedCall%activation_1/PartitionedCall:output:0(^stream_0_drop_1/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_stream_0_drop_2_layer_call_and_return_conditional_losses_55416252)
'stream_0_drop_2/StatefulPartitionedCall�
(global_average_pooling1d/PartitionedCallPartitionedCall0stream_0_drop_2/StatefulPartitionedCall:output:0*
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
U__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_55414422*
(global_average_pooling1d/PartitionedCall�
concatenate/PartitionedCallPartitionedCall1global_average_pooling1d/PartitionedCall:output:0*
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
GPU2*0J 8� *Q
fLRJ
H__inference_concatenate_layer_call_and_return_conditional_losses_55414502
concatenate/PartitionedCall�
'dense_1_dropout/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0(^stream_0_drop_2/StatefulPartitionedCall*
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
GPU2*0J 8� *U
fPRN
L__inference_dense_1_dropout_layer_call_and_return_conditional_losses_55415912)
'dense_1_dropout/StatefulPartitionedCall�
dense_1/StatefulPartitionedCallStatefulPartitionedCall0dense_1_dropout/StatefulPartitionedCall:output:0dense_1_5542134dense_1_5542136*
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
D__inference_dense_1_layer_call_and_return_conditional_losses_55414752!
dense_1/StatefulPartitionedCall�
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0batch_normalization_2_5542139batch_normalization_2_5542141batch_normalization_2_5542143batch_normalization_2_5542145*
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
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_55412052/
-batch_normalization_2/StatefulPartitionedCall�
"dense_activation_1/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
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
O__inference_dense_activation_1_layer_call_and_return_conditional_losses_55414952$
"dense_activation_1/PartitionedCall�
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpstream_0_conv_1_5542099*"
_output_shapes
: *
dtype027
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp�
&stream_0_conv_1/kernel/Regularizer/AbsAbs=stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
: 2(
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
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpstream_0_conv_2_5542115*"
_output_shapes
: @*
dtype02:
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp�
)stream_0_conv_2/kernel/Regularizer/SquareSquare@stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
: @2+
)stream_0_conv_2/kernel/Regularizer/Square�
(stream_0_conv_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_0_conv_2/kernel/Regularizer/Const�
&stream_0_conv_2/kernel/Regularizer/SumSum-stream_0_conv_2/kernel/Regularizer/Square:y:01stream_0_conv_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_2/kernel/Regularizer/Sum�
(stream_0_conv_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2*
(stream_0_conv_2/kernel/Regularizer/mul/x�
&stream_0_conv_2/kernel/Regularizer/mulMul1stream_0_conv_2/kernel/Regularizer/mul/x:output:0/stream_0_conv_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_2/kernel/Regularizer/mul�
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_1_5542134*
_output_shapes

:@T*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOp�
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:@T2 
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
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall.^dense_1/kernel/Regularizer/Abs/ReadVariableOp(^dense_1_dropout/StatefulPartitionedCall(^stream_0_conv_1/StatefulPartitionedCall6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp(^stream_0_conv_2/StatefulPartitionedCall9^stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp(^stream_0_drop_1/StatefulPartitionedCall(^stream_0_drop_2/StatefulPartitionedCall,^stream_0_input_drop/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:����������: : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2^
-dense_1/kernel/Regularizer/Abs/ReadVariableOp-dense_1/kernel/Regularizer/Abs/ReadVariableOp2R
'dense_1_dropout/StatefulPartitionedCall'dense_1_dropout/StatefulPartitionedCall2R
'stream_0_conv_1/StatefulPartitionedCall'stream_0_conv_1/StatefulPartitionedCall2n
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp2R
'stream_0_conv_2/StatefulPartitionedCall'stream_0_conv_2/StatefulPartitionedCall2t
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp2R
'stream_0_drop_1/StatefulPartitionedCall'stream_0_drop_1/StatefulPartitionedCall2R
'stream_0_drop_2/StatefulPartitionedCall'stream_0_drop_2/StatefulPartitionedCall2Z
+stream_0_input_drop/StatefulPartitionedCall+stream_0_input_drop/StatefulPartitionedCall:V R
,
_output_shapes
:����������
"
_user_specified_name
inputs_0
�
�
7__inference_batch_normalization_1_layer_call_fn_5544980

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
:����������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *[
fVRT
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_55416832
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:����������@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:����������@
 
_user_specified_nameinputs
�+
�
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_5541019

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
�
j
L__inference_dense_1_dropout_layer_call_and_return_conditional_losses_5545055

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:���������@2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:���������@2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������@:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
j
1__inference_stream_0_drop_2_layer_call_fn_5545017

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
:����������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_stream_0_drop_2_layer_call_and_return_conditional_losses_55416252
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:����������@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������@22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:����������@
 
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
serving_default_left_inputs:0����������
J
right_inputs:
serving_default_right_inputs:0����������<
distance0
StatefulPartitionedCall:0���������tensorflow/serving/predict:��
�
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
�_default_save_signature
+�&call_and_return_all_conditional_losses
�__call__"
_tf_keras_network
"
_tf_keras_input_layer
"
_tf_keras_input_layer
�
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
layer-10
layer-11
layer-12
layer_with_weights-4
layer-13
layer_with_weights-5
layer-14
layer-15
trainable_variables
	variables
regularization_losses
	keras_api
+�&call_and_return_all_conditional_losses
�__call__"
_tf_keras_network
�
trainable_variables
 	variables
!regularization_losses
"	keras_api
+�&call_and_return_all_conditional_losses
�__call__"
_tf_keras_layer
�

#beta_1

$beta_2
	%decay
&learning_rate
'iter(m�)m�*m�+m�,m�-m�.m�/m�0m�1m�2m�3m�(v�)v�*v�+v�,v�-v�.v�/v�0v�1v�2v�3v�"
	optimizer
v
(0
)1
*2
+3
,4
-5
.6
/7
08
19
210
311"
trackable_list_wrapper
�
(0
)1
*2
+3
44
55
,6
-7
.8
/9
610
711
012
113
214
315
816
917"
trackable_list_wrapper
 "
trackable_list_wrapper
�
:layer_metrics
trainable_variables

;layers
<layer_regularization_losses
	variables
=non_trainable_variables
regularization_losses
>metrics
�__call__
�_default_save_signature
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
-
�serving_default"
signature_map
"
_tf_keras_input_layer
�
?trainable_variables
@	variables
Aregularization_losses
B	keras_api
+�&call_and_return_all_conditional_losses
�__call__"
_tf_keras_layer
�

(kernel
)bias
Ctrainable_variables
D	variables
Eregularization_losses
F	keras_api
+�&call_and_return_all_conditional_losses
�__call__"
_tf_keras_layer
�
Gaxis
	*gamma
+beta
4moving_mean
5moving_variance
Htrainable_variables
I	variables
Jregularization_losses
K	keras_api
+�&call_and_return_all_conditional_losses
�__call__"
_tf_keras_layer
�
Ltrainable_variables
M	variables
Nregularization_losses
O	keras_api
+�&call_and_return_all_conditional_losses
�__call__"
_tf_keras_layer
�
Ptrainable_variables
Q	variables
Rregularization_losses
S	keras_api
+�&call_and_return_all_conditional_losses
�__call__"
_tf_keras_layer
�

,kernel
-bias
Ttrainable_variables
U	variables
Vregularization_losses
W	keras_api
+�&call_and_return_all_conditional_losses
�__call__"
_tf_keras_layer
�
Xaxis
	.gamma
/beta
6moving_mean
7moving_variance
Ytrainable_variables
Z	variables
[regularization_losses
\	keras_api
+�&call_and_return_all_conditional_losses
�__call__"
_tf_keras_layer
�
]trainable_variables
^	variables
_regularization_losses
`	keras_api
+�&call_and_return_all_conditional_losses
�__call__"
_tf_keras_layer
�
atrainable_variables
b	variables
cregularization_losses
d	keras_api
+�&call_and_return_all_conditional_losses
�__call__"
_tf_keras_layer
�
etrainable_variables
f	variables
gregularization_losses
h	keras_api
+�&call_and_return_all_conditional_losses
�__call__"
_tf_keras_layer
�
itrainable_variables
j	variables
kregularization_losses
l	keras_api
+�&call_and_return_all_conditional_losses
�__call__"
_tf_keras_layer
�
mtrainable_variables
n	variables
oregularization_losses
p	keras_api
+�&call_and_return_all_conditional_losses
�__call__"
_tf_keras_layer
�

0kernel
1bias
qtrainable_variables
r	variables
sregularization_losses
t	keras_api
+�&call_and_return_all_conditional_losses
�__call__"
_tf_keras_layer
�
uaxis
	2gamma
3beta
8moving_mean
9moving_variance
vtrainable_variables
w	variables
xregularization_losses
y	keras_api
+�&call_and_return_all_conditional_losses
�__call__"
_tf_keras_layer
�
ztrainable_variables
{	variables
|regularization_losses
}	keras_api
+�&call_and_return_all_conditional_losses
�__call__"
_tf_keras_layer
v
(0
)1
*2
+3
,4
-5
.6
/7
08
19
210
311"
trackable_list_wrapper
�
(0
)1
*2
+3
44
55
,6
-7
.8
/9
610
711
012
113
214
315
816
917"
trackable_list_wrapper
8
�0
�1
�2"
trackable_list_wrapper
�
~layer_metrics
trainable_variables

layers
 �layer_regularization_losses
	variables
�non_trainable_variables
regularization_losses
�metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�layer_metrics
trainable_variables
�layers
 �layer_regularization_losses
 	variables
�non_trainable_variables
!regularization_losses
�metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
: (2beta_1
: (2beta_2
: (2decay
: (2learning_rate
:	 (2	Adam/iter
,:* 2stream_0_conv_1/kernel
":  2stream_0_conv_1/bias
':% 2batch_normalization/gamma
&:$ 2batch_normalization/beta
,:* @2stream_0_conv_2/kernel
": @2stream_0_conv_2/bias
):'@2batch_normalization_1/gamma
(:&@2batch_normalization_1/beta
 :@T2dense_1/kernel
:T2dense_1/bias
):'T2batch_normalization_2/gamma
(:&T2batch_normalization_2/beta
/:-  (2batch_normalization/moving_mean
3:1  (2#batch_normalization/moving_variance
1:/@ (2!batch_normalization_1/moving_mean
5:3@ (2%batch_normalization_1/moving_variance
1:/T (2!batch_normalization_2/moving_mean
5:3T (2%batch_normalization_2/moving_variance
 "
trackable_dict_wrapper
<
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
J
40
51
62
73
84
95"
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
�layer_metrics
?trainable_variables
�layers
 �layer_regularization_losses
@	variables
�non_trainable_variables
Aregularization_losses
�metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
.
(0
)1"
trackable_list_wrapper
.
(0
)1"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
�
�layer_metrics
Ctrainable_variables
�layers
 �layer_regularization_losses
D	variables
�non_trainable_variables
Eregularization_losses
�metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
*0
+1"
trackable_list_wrapper
<
*0
+1
42
53"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�layer_metrics
Htrainable_variables
�layers
 �layer_regularization_losses
I	variables
�non_trainable_variables
Jregularization_losses
�metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�layer_metrics
Ltrainable_variables
�layers
 �layer_regularization_losses
M	variables
�non_trainable_variables
Nregularization_losses
�metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�layer_metrics
Ptrainable_variables
�layers
 �layer_regularization_losses
Q	variables
�non_trainable_variables
Rregularization_losses
�metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
.
,0
-1"
trackable_list_wrapper
.
,0
-1"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
�
�layer_metrics
Ttrainable_variables
�layers
 �layer_regularization_losses
U	variables
�non_trainable_variables
Vregularization_losses
�metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
.0
/1"
trackable_list_wrapper
<
.0
/1
62
73"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�layer_metrics
Ytrainable_variables
�layers
 �layer_regularization_losses
Z	variables
�non_trainable_variables
[regularization_losses
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
�layer_metrics
]trainable_variables
�layers
 �layer_regularization_losses
^	variables
�non_trainable_variables
_regularization_losses
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
�layer_metrics
atrainable_variables
�layers
 �layer_regularization_losses
b	variables
�non_trainable_variables
cregularization_losses
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
�layer_metrics
etrainable_variables
�layers
 �layer_regularization_losses
f	variables
�non_trainable_variables
gregularization_losses
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
�layer_metrics
itrainable_variables
�layers
 �layer_regularization_losses
j	variables
�non_trainable_variables
kregularization_losses
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
�layer_metrics
mtrainable_variables
�layers
 �layer_regularization_losses
n	variables
�non_trainable_variables
oregularization_losses
�metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
.
00
11"
trackable_list_wrapper
.
00
11"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
�
�layer_metrics
qtrainable_variables
�layers
 �layer_regularization_losses
r	variables
�non_trainable_variables
sregularization_losses
�metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
20
31"
trackable_list_wrapper
<
20
31
82
93"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�layer_metrics
vtrainable_variables
�layers
 �layer_regularization_losses
w	variables
�non_trainable_variables
xregularization_losses
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
�layer_metrics
ztrainable_variables
�layers
 �layer_regularization_losses
{	variables
�non_trainable_variables
|regularization_losses
�metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
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
15"
trackable_list_wrapper
 "
trackable_list_wrapper
J
40
51
62
73
84
95"
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
R

�total

�count
�	variables
�	keras_api"
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
(
�0"
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
40
51"
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
�0"
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
60
71"
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
�0"
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
80
91"
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
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
1:/ 2Adam/stream_0_conv_1/kernel/m
':% 2Adam/stream_0_conv_1/bias/m
,:* 2 Adam/batch_normalization/gamma/m
+:) 2Adam/batch_normalization/beta/m
1:/ @2Adam/stream_0_conv_2/kernel/m
':%@2Adam/stream_0_conv_2/bias/m
.:,@2"Adam/batch_normalization_1/gamma/m
-:+@2!Adam/batch_normalization_1/beta/m
%:#@T2Adam/dense_1/kernel/m
:T2Adam/dense_1/bias/m
.:,T2"Adam/batch_normalization_2/gamma/m
-:+T2!Adam/batch_normalization_2/beta/m
1:/ 2Adam/stream_0_conv_1/kernel/v
':% 2Adam/stream_0_conv_1/bias/v
,:* 2 Adam/batch_normalization/gamma/v
+:) 2Adam/batch_normalization/beta/v
1:/ @2Adam/stream_0_conv_2/kernel/v
':%@2Adam/stream_0_conv_2/bias/v
.:,@2"Adam/batch_normalization_1/gamma/v
-:+@2!Adam/batch_normalization_1/beta/v
%:#@T2Adam/dense_1/kernel/v
:T2Adam/dense_1/bias/v
.:,T2"Adam/batch_normalization_2/gamma/v
-:+T2!Adam/batch_normalization_2/beta/v
�B�
"__inference__wrapped_model_5540773left_inputsright_inputs"�
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
�2�
B__inference_model_layer_call_and_return_conditional_losses_5543320
B__inference_model_layer_call_and_return_conditional_losses_5543646
B__inference_model_layer_call_and_return_conditional_losses_5542986
B__inference_model_layer_call_and_return_conditional_losses_5543066�
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
'__inference_model_layer_call_fn_5542414
'__inference_model_layer_call_fn_5543688
'__inference_model_layer_call_fn_5543730
'__inference_model_layer_call_fn_5542906�
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
�2�
F__inference_basemodel_layer_call_and_return_conditional_losses_5543857
F__inference_basemodel_layer_call_and_return_conditional_losses_5544036
F__inference_basemodel_layer_call_and_return_conditional_losses_5542095
F__inference_basemodel_layer_call_and_return_conditional_losses_5542168
F__inference_basemodel_layer_call_and_return_conditional_losses_5544145
F__inference_basemodel_layer_call_and_return_conditional_losses_5544324�
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
+__inference_basemodel_layer_call_fn_5541555
+__inference_basemodel_layer_call_fn_5544365
+__inference_basemodel_layer_call_fn_5544406
+__inference_basemodel_layer_call_fn_5542022
+__inference_basemodel_layer_call_fn_5544447
+__inference_basemodel_layer_call_fn_5544488�
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
E__inference_distance_layer_call_and_return_conditional_losses_5544500
E__inference_distance_layer_call_and_return_conditional_losses_5544512�
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
*__inference_distance_layer_call_fn_5544518
*__inference_distance_layer_call_fn_5544524�
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
%__inference_signature_wrapper_5543134left_inputsright_inputs"�
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
P__inference_stream_0_input_drop_layer_call_and_return_conditional_losses_5544529
P__inference_stream_0_input_drop_layer_call_and_return_conditional_losses_5544541�
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
5__inference_stream_0_input_drop_layer_call_fn_5544546
5__inference_stream_0_input_drop_layer_call_fn_5544551�
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
L__inference_stream_0_conv_1_layer_call_and_return_conditional_losses_5544578�
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
1__inference_stream_0_conv_1_layer_call_fn_5544587�
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
�2�
P__inference_batch_normalization_layer_call_and_return_conditional_losses_5544607
P__inference_batch_normalization_layer_call_and_return_conditional_losses_5544641
P__inference_batch_normalization_layer_call_and_return_conditional_losses_5544661
P__inference_batch_normalization_layer_call_and_return_conditional_losses_5544695�
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
5__inference_batch_normalization_layer_call_fn_5544708
5__inference_batch_normalization_layer_call_fn_5544721
5__inference_batch_normalization_layer_call_fn_5544734
5__inference_batch_normalization_layer_call_fn_5544747�
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
G__inference_activation_layer_call_and_return_conditional_losses_5544752�
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
,__inference_activation_layer_call_fn_5544757�
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
L__inference_stream_0_drop_1_layer_call_and_return_conditional_losses_5544762
L__inference_stream_0_drop_1_layer_call_and_return_conditional_losses_5544774�
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
1__inference_stream_0_drop_1_layer_call_fn_5544779
1__inference_stream_0_drop_1_layer_call_fn_5544784�
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
L__inference_stream_0_conv_2_layer_call_and_return_conditional_losses_5544811�
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
1__inference_stream_0_conv_2_layer_call_fn_5544820�
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
�2�
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_5544840
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_5544874
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_5544894
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_5544928�
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
7__inference_batch_normalization_1_layer_call_fn_5544941
7__inference_batch_normalization_1_layer_call_fn_5544954
7__inference_batch_normalization_1_layer_call_fn_5544967
7__inference_batch_normalization_1_layer_call_fn_5544980�
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
I__inference_activation_1_layer_call_and_return_conditional_losses_5544985�
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
.__inference_activation_1_layer_call_fn_5544990�
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
L__inference_stream_0_drop_2_layer_call_and_return_conditional_losses_5544995
L__inference_stream_0_drop_2_layer_call_and_return_conditional_losses_5545007�
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
1__inference_stream_0_drop_2_layer_call_fn_5545012
1__inference_stream_0_drop_2_layer_call_fn_5545017�
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
U__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_5545023
U__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_5545029�
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
:__inference_global_average_pooling1d_layer_call_fn_5545034
:__inference_global_average_pooling1d_layer_call_fn_5545039�
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
H__inference_concatenate_layer_call_and_return_conditional_losses_5545045�
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
-__inference_concatenate_layer_call_fn_5545050�
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
L__inference_dense_1_dropout_layer_call_and_return_conditional_losses_5545055
L__inference_dense_1_dropout_layer_call_and_return_conditional_losses_5545067�
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
1__inference_dense_1_dropout_layer_call_fn_5545072
1__inference_dense_1_dropout_layer_call_fn_5545077�
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
D__inference_dense_1_layer_call_and_return_conditional_losses_5545099�
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
)__inference_dense_1_layer_call_fn_5545108�
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
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_5545128
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_5545162�
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
7__inference_batch_normalization_2_layer_call_fn_5545175
7__inference_batch_normalization_2_layer_call_fn_5545188�
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
O__inference_dense_activation_1_layer_call_and_return_conditional_losses_5545193�
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
4__inference_dense_activation_1_layer_call_fn_5545198�
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
__inference_loss_fn_0_5545209�
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
__inference_loss_fn_1_5545220�
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
__inference_loss_fn_2_5545231�
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
"__inference__wrapped_model_5540773�()5*4+,-7.6/019283k�h
a�^
\�Y
*�'
left_inputs����������
+�(
right_inputs����������
� "3�0
.
distance"�
distance����������
I__inference_activation_1_layer_call_and_return_conditional_losses_5544985b4�1
*�'
%�"
inputs����������@
� "*�'
 �
0����������@
� �
.__inference_activation_1_layer_call_fn_5544990U4�1
*�'
%�"
inputs����������@
� "�����������@�
G__inference_activation_layer_call_and_return_conditional_losses_5544752b4�1
*�'
%�"
inputs���������� 
� "*�'
 �
0���������� 
� �
,__inference_activation_layer_call_fn_5544757U4�1
*�'
%�"
inputs���������� 
� "����������� �
F__inference_basemodel_layer_call_and_return_conditional_losses_5542095{()5*4+,-7.6/019283>�;
4�1
'�$
inputs_0����������
p 

 
� "%�"
�
0���������T
� �
F__inference_basemodel_layer_call_and_return_conditional_losses_5542168{()45*+,-67./018923>�;
4�1
'�$
inputs_0����������
p

 
� "%�"
�
0���������T
� �
F__inference_basemodel_layer_call_and_return_conditional_losses_5543857y()5*4+,-7.6/019283<�9
2�/
%�"
inputs����������
p 

 
� "%�"
�
0���������T
� �
F__inference_basemodel_layer_call_and_return_conditional_losses_5544036y()45*+,-67./018923<�9
2�/
%�"
inputs����������
p

 
� "%�"
�
0���������T
� �
F__inference_basemodel_layer_call_and_return_conditional_losses_5544145�()5*4+,-7.6/019283C�@
9�6
,�)
'�$
inputs/0����������
p 

 
� "%�"
�
0���������T
� �
F__inference_basemodel_layer_call_and_return_conditional_losses_5544324�()45*+,-67./018923C�@
9�6
,�)
'�$
inputs/0����������
p

 
� "%�"
�
0���������T
� �
+__inference_basemodel_layer_call_fn_5541555n()5*4+,-7.6/019283>�;
4�1
'�$
inputs_0����������
p 

 
� "����������T�
+__inference_basemodel_layer_call_fn_5542022n()45*+,-67./018923>�;
4�1
'�$
inputs_0����������
p

 
� "����������T�
+__inference_basemodel_layer_call_fn_5544365l()5*4+,-7.6/019283<�9
2�/
%�"
inputs����������
p 

 
� "����������T�
+__inference_basemodel_layer_call_fn_5544406l()45*+,-67./018923<�9
2�/
%�"
inputs����������
p

 
� "����������T�
+__inference_basemodel_layer_call_fn_5544447s()5*4+,-7.6/019283C�@
9�6
,�)
'�$
inputs/0����������
p 

 
� "����������T�
+__inference_basemodel_layer_call_fn_5544488s()45*+,-67./018923C�@
9�6
,�)
'�$
inputs/0����������
p

 
� "����������T�
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_5544840|7.6/@�=
6�3
-�*
inputs������������������@
p 
� "2�/
(�%
0������������������@
� �
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_5544874|67./@�=
6�3
-�*
inputs������������������@
p
� "2�/
(�%
0������������������@
� �
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_5544894l7.6/8�5
.�+
%�"
inputs����������@
p 
� "*�'
 �
0����������@
� �
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_5544928l67./8�5
.�+
%�"
inputs����������@
p
� "*�'
 �
0����������@
� �
7__inference_batch_normalization_1_layer_call_fn_5544941o7.6/@�=
6�3
-�*
inputs������������������@
p 
� "%�"������������������@�
7__inference_batch_normalization_1_layer_call_fn_5544954o67./@�=
6�3
-�*
inputs������������������@
p
� "%�"������������������@�
7__inference_batch_normalization_1_layer_call_fn_5544967_7.6/8�5
.�+
%�"
inputs����������@
p 
� "�����������@�
7__inference_batch_normalization_1_layer_call_fn_5544980_67./8�5
.�+
%�"
inputs����������@
p
� "�����������@�
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_5545128b92833�0
)�&
 �
inputs���������T
p 
� "%�"
�
0���������T
� �
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_5545162b89233�0
)�&
 �
inputs���������T
p
� "%�"
�
0���������T
� �
7__inference_batch_normalization_2_layer_call_fn_5545175U92833�0
)�&
 �
inputs���������T
p 
� "����������T�
7__inference_batch_normalization_2_layer_call_fn_5545188U89233�0
)�&
 �
inputs���������T
p
� "����������T�
P__inference_batch_normalization_layer_call_and_return_conditional_losses_5544607|5*4+@�=
6�3
-�*
inputs������������������ 
p 
� "2�/
(�%
0������������������ 
� �
P__inference_batch_normalization_layer_call_and_return_conditional_losses_5544641|45*+@�=
6�3
-�*
inputs������������������ 
p
� "2�/
(�%
0������������������ 
� �
P__inference_batch_normalization_layer_call_and_return_conditional_losses_5544661l5*4+8�5
.�+
%�"
inputs���������� 
p 
� "*�'
 �
0���������� 
� �
P__inference_batch_normalization_layer_call_and_return_conditional_losses_5544695l45*+8�5
.�+
%�"
inputs���������� 
p
� "*�'
 �
0���������� 
� �
5__inference_batch_normalization_layer_call_fn_5544708o5*4+@�=
6�3
-�*
inputs������������������ 
p 
� "%�"������������������ �
5__inference_batch_normalization_layer_call_fn_5544721o45*+@�=
6�3
-�*
inputs������������������ 
p
� "%�"������������������ �
5__inference_batch_normalization_layer_call_fn_5544734_5*4+8�5
.�+
%�"
inputs���������� 
p 
� "����������� �
5__inference_batch_normalization_layer_call_fn_5544747_45*+8�5
.�+
%�"
inputs���������� 
p
� "����������� �
H__inference_concatenate_layer_call_and_return_conditional_losses_5545045_6�3
,�)
'�$
"�
inputs/0���������@
� "%�"
�
0���������@
� �
-__inference_concatenate_layer_call_fn_5545050R6�3
,�)
'�$
"�
inputs/0���������@
� "����������@�
L__inference_dense_1_dropout_layer_call_and_return_conditional_losses_5545055\3�0
)�&
 �
inputs���������@
p 
� "%�"
�
0���������@
� �
L__inference_dense_1_dropout_layer_call_and_return_conditional_losses_5545067\3�0
)�&
 �
inputs���������@
p
� "%�"
�
0���������@
� �
1__inference_dense_1_dropout_layer_call_fn_5545072O3�0
)�&
 �
inputs���������@
p 
� "����������@�
1__inference_dense_1_dropout_layer_call_fn_5545077O3�0
)�&
 �
inputs���������@
p
� "����������@�
D__inference_dense_1_layer_call_and_return_conditional_losses_5545099\01/�,
%�"
 �
inputs���������@
� "%�"
�
0���������T
� |
)__inference_dense_1_layer_call_fn_5545108O01/�,
%�"
 �
inputs���������@
� "����������T�
O__inference_dense_activation_1_layer_call_and_return_conditional_losses_5545193X/�,
%�"
 �
inputs���������T
� "%�"
�
0���������T
� �
4__inference_dense_activation_1_layer_call_fn_5545198K/�,
%�"
 �
inputs���������T
� "����������T�
E__inference_distance_layer_call_and_return_conditional_losses_5544500�b�_
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
E__inference_distance_layer_call_and_return_conditional_losses_5544512�b�_
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
*__inference_distance_layer_call_fn_5544518~b�_
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
*__inference_distance_layer_call_fn_5544524~b�_
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
U__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_5545023{I�F
?�<
6�3
inputs'���������������������������

 
� ".�+
$�!
0������������������
� �
U__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_5545029a8�5
.�+
%�"
inputs����������@

 
� "%�"
�
0���������@
� �
:__inference_global_average_pooling1d_layer_call_fn_5545034nI�F
?�<
6�3
inputs'���������������������������

 
� "!��������������������
:__inference_global_average_pooling1d_layer_call_fn_5545039T8�5
.�+
%�"
inputs����������@

 
� "����������@<
__inference_loss_fn_0_5545209(�

� 
� "� <
__inference_loss_fn_1_5545220,�

� 
� "� <
__inference_loss_fn_2_55452310�

� 
� "� �
B__inference_model_layer_call_and_return_conditional_losses_5542986�()5*4+,-7.6/019283s�p
i�f
\�Y
*�'
left_inputs����������
+�(
right_inputs����������
p 

 
� "%�"
�
0���������
� �
B__inference_model_layer_call_and_return_conditional_losses_5543066�()45*+,-67./018923s�p
i�f
\�Y
*�'
left_inputs����������
+�(
right_inputs����������
p

 
� "%�"
�
0���������
� �
B__inference_model_layer_call_and_return_conditional_losses_5543320�()5*4+,-7.6/019283l�i
b�_
U�R
'�$
inputs/0����������
'�$
inputs/1����������
p 

 
� "%�"
�
0���������
� �
B__inference_model_layer_call_and_return_conditional_losses_5543646�()45*+,-67./018923l�i
b�_
U�R
'�$
inputs/0����������
'�$
inputs/1����������
p

 
� "%�"
�
0���������
� �
'__inference_model_layer_call_fn_5542414�()5*4+,-7.6/019283s�p
i�f
\�Y
*�'
left_inputs����������
+�(
right_inputs����������
p 

 
� "�����������
'__inference_model_layer_call_fn_5542906�()45*+,-67./018923s�p
i�f
\�Y
*�'
left_inputs����������
+�(
right_inputs����������
p

 
� "�����������
'__inference_model_layer_call_fn_5543688�()5*4+,-7.6/019283l�i
b�_
U�R
'�$
inputs/0����������
'�$
inputs/1����������
p 

 
� "�����������
'__inference_model_layer_call_fn_5543730�()45*+,-67./018923l�i
b�_
U�R
'�$
inputs/0����������
'�$
inputs/1����������
p

 
� "�����������
%__inference_signature_wrapper_5543134�()5*4+,-7.6/019283���
� 
{�x
9
left_inputs*�'
left_inputs����������
;
right_inputs+�(
right_inputs����������"3�0
.
distance"�
distance����������
L__inference_stream_0_conv_1_layer_call_and_return_conditional_losses_5544578f()4�1
*�'
%�"
inputs����������
� "*�'
 �
0���������� 
� �
1__inference_stream_0_conv_1_layer_call_fn_5544587Y()4�1
*�'
%�"
inputs����������
� "����������� �
L__inference_stream_0_conv_2_layer_call_and_return_conditional_losses_5544811f,-4�1
*�'
%�"
inputs���������� 
� "*�'
 �
0����������@
� �
1__inference_stream_0_conv_2_layer_call_fn_5544820Y,-4�1
*�'
%�"
inputs���������� 
� "�����������@�
L__inference_stream_0_drop_1_layer_call_and_return_conditional_losses_5544762f8�5
.�+
%�"
inputs���������� 
p 
� "*�'
 �
0���������� 
� �
L__inference_stream_0_drop_1_layer_call_and_return_conditional_losses_5544774f8�5
.�+
%�"
inputs���������� 
p
� "*�'
 �
0���������� 
� �
1__inference_stream_0_drop_1_layer_call_fn_5544779Y8�5
.�+
%�"
inputs���������� 
p 
� "����������� �
1__inference_stream_0_drop_1_layer_call_fn_5544784Y8�5
.�+
%�"
inputs���������� 
p
� "����������� �
L__inference_stream_0_drop_2_layer_call_and_return_conditional_losses_5544995f8�5
.�+
%�"
inputs����������@
p 
� "*�'
 �
0����������@
� �
L__inference_stream_0_drop_2_layer_call_and_return_conditional_losses_5545007f8�5
.�+
%�"
inputs����������@
p
� "*�'
 �
0����������@
� �
1__inference_stream_0_drop_2_layer_call_fn_5545012Y8�5
.�+
%�"
inputs����������@
p 
� "�����������@�
1__inference_stream_0_drop_2_layer_call_fn_5545017Y8�5
.�+
%�"
inputs����������@
p
� "�����������@�
P__inference_stream_0_input_drop_layer_call_and_return_conditional_losses_5544529f8�5
.�+
%�"
inputs����������
p 
� "*�'
 �
0����������
� �
P__inference_stream_0_input_drop_layer_call_and_return_conditional_losses_5544541f8�5
.�+
%�"
inputs����������
p
� "*�'
 �
0����������
� �
5__inference_stream_0_input_drop_layer_call_fn_5544546Y8�5
.�+
%�"
inputs����������
p 
� "������������
5__inference_stream_0_input_drop_layer_call_fn_5544551Y8�5
.�+
%�"
inputs����������
p
� "�����������