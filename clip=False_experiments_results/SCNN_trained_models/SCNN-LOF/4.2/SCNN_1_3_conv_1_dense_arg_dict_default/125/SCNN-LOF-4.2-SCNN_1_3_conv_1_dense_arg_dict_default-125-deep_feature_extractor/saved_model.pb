��1
��
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
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.6.22v2.6.1-9-gc2363d6d0258ɢ.
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
stream_0_conv_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@�*'
shared_namestream_0_conv_2/kernel
�
*stream_0_conv_2/kernel/Read/ReadVariableOpReadVariableOpstream_0_conv_2/kernel*#
_output_shapes
:@�*
dtype0
�
stream_0_conv_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*%
shared_namestream_0_conv_2/bias
z
(stream_0_conv_2/bias/Read/ReadVariableOpReadVariableOpstream_0_conv_2/bias*
_output_shapes	
:�*
dtype0
�
batch_normalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*,
shared_namebatch_normalization_1/gamma
�
/batch_normalization_1/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_1/gamma*
_output_shapes	
:�*
dtype0
�
batch_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*+
shared_namebatch_normalization_1/beta
�
.batch_normalization_1/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_1/beta*
_output_shapes	
:�*
dtype0
�
!batch_normalization_1/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*2
shared_name#!batch_normalization_1/moving_mean
�
5batch_normalization_1/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_1/moving_mean*
_output_shapes	
:�*
dtype0
�
%batch_normalization_1/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*6
shared_name'%batch_normalization_1/moving_variance
�
9batch_normalization_1/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_1/moving_variance*
_output_shapes	
:�*
dtype0
�
stream_0_conv_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*'
shared_namestream_0_conv_3/kernel
�
*stream_0_conv_3/kernel/Read/ReadVariableOpReadVariableOpstream_0_conv_3/kernel*$
_output_shapes
:��*
dtype0
�
stream_0_conv_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*%
shared_namestream_0_conv_3/bias
z
(stream_0_conv_3/bias/Read/ReadVariableOpReadVariableOpstream_0_conv_3/bias*
_output_shapes	
:�*
dtype0
�
batch_normalization_2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*,
shared_namebatch_normalization_2/gamma
�
/batch_normalization_2/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_2/gamma*
_output_shapes	
:�*
dtype0
�
batch_normalization_2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*+
shared_namebatch_normalization_2/beta
�
.batch_normalization_2/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_2/beta*
_output_shapes	
:�*
dtype0
�
!batch_normalization_2/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*2
shared_name#!batch_normalization_2/moving_mean
�
5batch_normalization_2/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_2/moving_mean*
_output_shapes	
:�*
dtype0
�
%batch_normalization_2/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*6
shared_name'%batch_normalization_2/moving_variance
�
9batch_normalization_2/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_2/moving_variance*
_output_shapes	
:�*
dtype0
y
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�T*
shared_namedense_1/kernel
r
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes
:	�T*
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

NoOpNoOp
�M
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�M
value�MB�M B�M
�
layer-0
layer_with_weights-0
layer-1
	variables
regularization_losses
trainable_variables
	keras_api

signatures
 
�
layer-0
	layer-1

layer_with_weights-0

layer-2
layer_with_weights-1
layer-3
layer-4
layer-5
layer_with_weights-2
layer-6
layer_with_weights-3
layer-7
layer-8
layer-9
layer_with_weights-4
layer-10
layer_with_weights-5
layer-11
layer-12
layer-13
layer-14
layer-15
layer_with_weights-6
layer-16
layer_with_weights-7
layer-17
layer-18
	variables
regularization_losses
trainable_variables
	keras_api
�
0
 1
!2
"3
#4
$5
%6
&7
'8
(9
)10
*11
+12
,13
-14
.15
/16
017
118
219
320
421
522
623
 
v
0
 1
!2
"3
%4
&5
'6
(7
+8
,9
-10
.11
112
213
314
415
�
	variables
regularization_losses
7layer_metrics

8layers
trainable_variables
9metrics
:non_trainable_variables
;layer_regularization_losses
 
 
R
<	variables
=regularization_losses
>trainable_variables
?	keras_api
h

kernel
 bias
@	variables
Aregularization_losses
Btrainable_variables
C	keras_api
�
Daxis
	!gamma
"beta
#moving_mean
$moving_variance
E	variables
Fregularization_losses
Gtrainable_variables
H	keras_api
R
I	variables
Jregularization_losses
Ktrainable_variables
L	keras_api
R
M	variables
Nregularization_losses
Otrainable_variables
P	keras_api
h

%kernel
&bias
Q	variables
Rregularization_losses
Strainable_variables
T	keras_api
�
Uaxis
	'gamma
(beta
)moving_mean
*moving_variance
V	variables
Wregularization_losses
Xtrainable_variables
Y	keras_api
R
Z	variables
[regularization_losses
\trainable_variables
]	keras_api
R
^	variables
_regularization_losses
`trainable_variables
a	keras_api
h

+kernel
,bias
b	variables
cregularization_losses
dtrainable_variables
e	keras_api
�
faxis
	-gamma
.beta
/moving_mean
0moving_variance
g	variables
hregularization_losses
itrainable_variables
j	keras_api
R
k	variables
lregularization_losses
mtrainable_variables
n	keras_api
R
o	variables
pregularization_losses
qtrainable_variables
r	keras_api
R
s	variables
tregularization_losses
utrainable_variables
v	keras_api
R
w	variables
xregularization_losses
ytrainable_variables
z	keras_api
h

1kernel
2bias
{	variables
|regularization_losses
}trainable_variables
~	keras_api
�
axis
	3gamma
4beta
5moving_mean
6moving_variance
�	variables
�regularization_losses
�trainable_variables
�	keras_api
V
�	variables
�regularization_losses
�trainable_variables
�	keras_api
�
0
 1
!2
"3
#4
$5
%6
&7
'8
(9
)10
*11
+12
,13
-14
.15
/16
017
118
219
320
421
522
623
 
v
0
 1
!2
"3
%4
&5
'6
(7
+8
,9
-10
.11
112
213
314
415
�
	variables
regularization_losses
�layer_metrics
�layers
trainable_variables
�metrics
�non_trainable_variables
 �layer_regularization_losses
RP
VARIABLE_VALUEstream_0_conv_1/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE
PN
VARIABLE_VALUEstream_0_conv_1/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEbatch_normalization/gamma&variables/2/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEbatch_normalization/beta&variables/3/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEbatch_normalization/moving_mean&variables/4/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUE#batch_normalization/moving_variance&variables/5/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEstream_0_conv_2/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE
PN
VARIABLE_VALUEstream_0_conv_2/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEbatch_normalization_1/gamma&variables/8/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEbatch_normalization_1/beta&variables/9/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUE!batch_normalization_1/moving_mean'variables/10/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUE%batch_normalization_1/moving_variance'variables/11/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEstream_0_conv_3/kernel'variables/12/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEstream_0_conv_3/bias'variables/13/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEbatch_normalization_2/gamma'variables/14/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEbatch_normalization_2/beta'variables/15/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUE!batch_normalization_2/moving_mean'variables/16/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUE%batch_normalization_2/moving_variance'variables/17/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEdense_1/kernel'variables/18/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUEdense_1/bias'variables/19/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEbatch_normalization_3/gamma'variables/20/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEbatch_normalization_3/beta'variables/21/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUE!batch_normalization_3/moving_mean'variables/22/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUE%batch_normalization_3/moving_variance'variables/23/.ATTRIBUTES/VARIABLE_VALUE
 

0
1
 
8
#0
$1
)2
*3
/4
05
56
67
 
 
 
 
�
<	variables
=regularization_losses
�layer_metrics
�layers
>trainable_variables
�metrics
�non_trainable_variables
 �layer_regularization_losses

0
 1
 

0
 1
�
@	variables
Aregularization_losses
�layer_metrics
�layers
Btrainable_variables
�metrics
�non_trainable_variables
 �layer_regularization_losses
 

!0
"1
#2
$3
 

!0
"1
�
E	variables
Fregularization_losses
�layer_metrics
�layers
Gtrainable_variables
�metrics
�non_trainable_variables
 �layer_regularization_losses
 
 
 
�
I	variables
Jregularization_losses
�layer_metrics
�layers
Ktrainable_variables
�metrics
�non_trainable_variables
 �layer_regularization_losses
 
 
 
�
M	variables
Nregularization_losses
�layer_metrics
�layers
Otrainable_variables
�metrics
�non_trainable_variables
 �layer_regularization_losses

%0
&1
 

%0
&1
�
Q	variables
Rregularization_losses
�layer_metrics
�layers
Strainable_variables
�metrics
�non_trainable_variables
 �layer_regularization_losses
 

'0
(1
)2
*3
 

'0
(1
�
V	variables
Wregularization_losses
�layer_metrics
�layers
Xtrainable_variables
�metrics
�non_trainable_variables
 �layer_regularization_losses
 
 
 
�
Z	variables
[regularization_losses
�layer_metrics
�layers
\trainable_variables
�metrics
�non_trainable_variables
 �layer_regularization_losses
 
 
 
�
^	variables
_regularization_losses
�layer_metrics
�layers
`trainable_variables
�metrics
�non_trainable_variables
 �layer_regularization_losses

+0
,1
 

+0
,1
�
b	variables
cregularization_losses
�layer_metrics
�layers
dtrainable_variables
�metrics
�non_trainable_variables
 �layer_regularization_losses
 

-0
.1
/2
03
 

-0
.1
�
g	variables
hregularization_losses
�layer_metrics
�layers
itrainable_variables
�metrics
�non_trainable_variables
 �layer_regularization_losses
 
 
 
�
k	variables
lregularization_losses
�layer_metrics
�layers
mtrainable_variables
�metrics
�non_trainable_variables
 �layer_regularization_losses
 
 
 
�
o	variables
pregularization_losses
�layer_metrics
�layers
qtrainable_variables
�metrics
�non_trainable_variables
 �layer_regularization_losses
 
 
 
�
s	variables
tregularization_losses
�layer_metrics
�layers
utrainable_variables
�metrics
�non_trainable_variables
 �layer_regularization_losses
 
 
 
�
w	variables
xregularization_losses
�layer_metrics
�layers
ytrainable_variables
�metrics
�non_trainable_variables
 �layer_regularization_losses

10
21
 

10
21
�
{	variables
|regularization_losses
�layer_metrics
�layers
}trainable_variables
�metrics
�non_trainable_variables
 �layer_regularization_losses
 

30
41
52
63
 

30
41
�
�	variables
�regularization_losses
�layer_metrics
�layers
�trainable_variables
�metrics
�non_trainable_variables
 �layer_regularization_losses
 
 
 
�
�	variables
�regularization_losses
�layer_metrics
�layers
�trainable_variables
�metrics
�non_trainable_variables
 �layer_regularization_losses
 
�
0
	1

2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
 
8
#0
$1
)2
*3
/4
05
56
67
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
#0
$1
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
)0
*1
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
/0
01
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
50
61
 
 
 
 
 
 
�
serving_default_left_inputsPlaceholder*+
_output_shapes
:���������}*
dtype0* 
shape:���������}
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_left_inputsstream_0_conv_1/kernelstream_0_conv_1/bias#batch_normalization/moving_variancebatch_normalization/gammabatch_normalization/moving_meanbatch_normalization/betastream_0_conv_2/kernelstream_0_conv_2/bias%batch_normalization_1/moving_variancebatch_normalization_1/gamma!batch_normalization_1/moving_meanbatch_normalization_1/betastream_0_conv_3/kernelstream_0_conv_3/bias%batch_normalization_2/moving_variancebatch_normalization_2/gamma!batch_normalization_2/moving_meanbatch_normalization_2/betadense_1/kerneldense_1/bias%batch_normalization_3/moving_variancebatch_normalization_3/gamma!batch_normalization_3/moving_meanbatch_normalization_3/beta*$
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������T*:
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *.
f)R'
%__inference_signature_wrapper_8909238
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename*stream_0_conv_1/kernel/Read/ReadVariableOp(stream_0_conv_1/bias/Read/ReadVariableOp-batch_normalization/gamma/Read/ReadVariableOp,batch_normalization/beta/Read/ReadVariableOp3batch_normalization/moving_mean/Read/ReadVariableOp7batch_normalization/moving_variance/Read/ReadVariableOp*stream_0_conv_2/kernel/Read/ReadVariableOp(stream_0_conv_2/bias/Read/ReadVariableOp/batch_normalization_1/gamma/Read/ReadVariableOp.batch_normalization_1/beta/Read/ReadVariableOp5batch_normalization_1/moving_mean/Read/ReadVariableOp9batch_normalization_1/moving_variance/Read/ReadVariableOp*stream_0_conv_3/kernel/Read/ReadVariableOp(stream_0_conv_3/bias/Read/ReadVariableOp/batch_normalization_2/gamma/Read/ReadVariableOp.batch_normalization_2/beta/Read/ReadVariableOp5batch_normalization_2/moving_mean/Read/ReadVariableOp9batch_normalization_2/moving_variance/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOp/batch_normalization_3/gamma/Read/ReadVariableOp.batch_normalization_3/beta/Read/ReadVariableOp5batch_normalization_3/moving_mean/Read/ReadVariableOp9batch_normalization_3/moving_variance/Read/ReadVariableOpConst*%
Tin
2*
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
 __inference__traced_save_8911740
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamestream_0_conv_1/kernelstream_0_conv_1/biasbatch_normalization/gammabatch_normalization/betabatch_normalization/moving_mean#batch_normalization/moving_variancestream_0_conv_2/kernelstream_0_conv_2/biasbatch_normalization_1/gammabatch_normalization_1/beta!batch_normalization_1/moving_mean%batch_normalization_1/moving_variancestream_0_conv_3/kernelstream_0_conv_3/biasbatch_normalization_2/gammabatch_normalization_2/beta!batch_normalization_2/moving_mean%batch_normalization_2/moving_variancedense_1/kerneldense_1/biasbatch_normalization_3/gammabatch_normalization_3/beta!batch_normalization_3/moving_mean%batch_normalization_3/moving_variance*$
Tin
2*
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
#__inference__traced_restore_8911822�-
�
�
L__inference_stream_0_conv_2_layer_call_and_return_conditional_losses_8911001

inputsB
+conv1d_expanddims_1_readvariableop_resource:@�.
biasadd_readvariableop_resource:	�
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
T0*/
_output_shapes
:���������}@2
conv1d/ExpandDims�
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:@�*
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
T0*'
_output_shapes
:@�2
conv1d/ExpandDims_1�
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:���������}�*
paddingSAME*
strides
2
conv1d�
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:���������}�*
squeeze_dims

���������2
conv1d/Squeeze�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������}�2	
BiasAdd�
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:@�*
dtype02:
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp�
)stream_0_conv_2/kernel/Regularizer/SquareSquare@stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*#
_output_shapes
:@�2+
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
:���������}�2

Identity�
NoOpNoOp^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp9^stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������}@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp2t
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp:S O
+
_output_shapes
:���������}@
 
_user_specified_nameinputs
�
�
+__inference_basemodel_layer_call_fn_8909849

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:@ 
	unknown_5:@�
	unknown_6:	�
	unknown_7:	�
	unknown_8:	�
	unknown_9:	�

unknown_10:	�"

unknown_11:��

unknown_12:	�

unknown_13:	�

unknown_14:	�

unknown_15:	�

unknown_16:	�

unknown_17:	�T

unknown_18:T

unknown_19:T

unknown_20:T

unknown_21:T

unknown_22:T
identity��StatefulPartitionedCall�
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
:���������T*2
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_basemodel_layer_call_and_return_conditional_losses_89078642
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
_construction_contextkEagerRuntime*Z
_input_shapesI
G:���������}: : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������}
 
_user_specified_nameinputs
�	
�
7__inference_batch_normalization_2_layer_call_fn_8911247

inputs
unknown:	�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:�������������������*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *[
fVRT
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_89066752
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:�������������������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):�������������������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:�������������������
 
_user_specified_nameinputs
�
�
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_8906861

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
k
L__inference_stream_0_drop_3_layer_call_and_return_conditional_losses_8907415

inputs
identity�c
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
:���������}�2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:���������}�*
dtype0*
seed�*
seed2�2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/y�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:���������}�2
dropout/GreaterEqual�
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:���������}�2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:���������}�2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:���������}�2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:���������}�:T P
,
_output_shapes
:���������}�
 
_user_specified_nameinputs
�
q
U__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_8911453

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
:����������2
Meanb
IdentityIdentityMean:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:���������}�:T P
,
_output_shapes
:���������}�
 
_user_specified_nameinputs
��
�
F__inference_basemodel_layer_call_and_return_conditional_losses_8910330

inputsQ
;stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource:@=
/stream_0_conv_1_biasadd_readvariableop_resource:@I
;batch_normalization_assignmovingavg_readvariableop_resource:@K
=batch_normalization_assignmovingavg_1_readvariableop_resource:@G
9batch_normalization_batchnorm_mul_readvariableop_resource:@C
5batch_normalization_batchnorm_readvariableop_resource:@R
;stream_0_conv_2_conv1d_expanddims_1_readvariableop_resource:@�>
/stream_0_conv_2_biasadd_readvariableop_resource:	�L
=batch_normalization_1_assignmovingavg_readvariableop_resource:	�N
?batch_normalization_1_assignmovingavg_1_readvariableop_resource:	�J
;batch_normalization_1_batchnorm_mul_readvariableop_resource:	�F
7batch_normalization_1_batchnorm_readvariableop_resource:	�S
;stream_0_conv_3_conv1d_expanddims_1_readvariableop_resource:��>
/stream_0_conv_3_biasadd_readvariableop_resource:	�L
=batch_normalization_2_assignmovingavg_readvariableop_resource:	�N
?batch_normalization_2_assignmovingavg_1_readvariableop_resource:	�J
;batch_normalization_2_batchnorm_mul_readvariableop_resource:	�F
7batch_normalization_2_batchnorm_readvariableop_resource:	�9
&dense_1_matmul_readvariableop_resource:	�T5
'dense_1_biasadd_readvariableop_resource:TK
=batch_normalization_3_assignmovingavg_readvariableop_resource:TM
?batch_normalization_3_assignmovingavg_1_readvariableop_resource:TI
;batch_normalization_3_batchnorm_mul_readvariableop_resource:TE
7batch_normalization_3_batchnorm_readvariableop_resource:T
identity��#batch_normalization/AssignMovingAvg�2batch_normalization/AssignMovingAvg/ReadVariableOp�%batch_normalization/AssignMovingAvg_1�4batch_normalization/AssignMovingAvg_1/ReadVariableOp�,batch_normalization/batchnorm/ReadVariableOp�0batch_normalization/batchnorm/mul/ReadVariableOp�%batch_normalization_1/AssignMovingAvg�4batch_normalization_1/AssignMovingAvg/ReadVariableOp�'batch_normalization_1/AssignMovingAvg_1�6batch_normalization_1/AssignMovingAvg_1/ReadVariableOp�.batch_normalization_1/batchnorm/ReadVariableOp�2batch_normalization_1/batchnorm/mul/ReadVariableOp�%batch_normalization_2/AssignMovingAvg�4batch_normalization_2/AssignMovingAvg/ReadVariableOp�'batch_normalization_2/AssignMovingAvg_1�6batch_normalization_2/AssignMovingAvg_1/ReadVariableOp�.batch_normalization_2/batchnorm/ReadVariableOp�2batch_normalization_2/batchnorm/mul/ReadVariableOp�%batch_normalization_3/AssignMovingAvg�4batch_normalization_3/AssignMovingAvg/ReadVariableOp�'batch_normalization_3/AssignMovingAvg_1�6batch_normalization_3/AssignMovingAvg_1/ReadVariableOp�.batch_normalization_3/batchnorm/ReadVariableOp�2batch_normalization_3/batchnorm/mul/ReadVariableOp�dense_1/BiasAdd/ReadVariableOp�dense_1/MatMul/ReadVariableOp�-dense_1/kernel/Regularizer/Abs/ReadVariableOp�&stream_0_conv_1/BiasAdd/ReadVariableOp�2stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp�5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp�&stream_0_conv_2/BiasAdd/ReadVariableOp�2stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOp�8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp�&stream_0_conv_3/BiasAdd/ReadVariableOp�2stream_0_conv_3/conv1d/ExpandDims_1/ReadVariableOp�5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp�
!stream_0_input_drop/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2#
!stream_0_input_drop/dropout/Const�
stream_0_input_drop/dropout/MulMulinputs*stream_0_input_drop/dropout/Const:output:0*
T0*+
_output_shapes
:���������}2!
stream_0_input_drop/dropout/Mul|
!stream_0_input_drop/dropout/ShapeShapeinputs*
T0*
_output_shapes
:2#
!stream_0_input_drop/dropout/Shape�
8stream_0_input_drop/dropout/random_uniform/RandomUniformRandomUniform*stream_0_input_drop/dropout/Shape:output:0*
T0*+
_output_shapes
:���������}*
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
T0*+
_output_shapes
:���������}2*
(stream_0_input_drop/dropout/GreaterEqual�
 stream_0_input_drop/dropout/CastCast,stream_0_input_drop/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:���������}2"
 stream_0_input_drop/dropout/Cast�
!stream_0_input_drop/dropout/Mul_1Mul#stream_0_input_drop/dropout/Mul:z:0$stream_0_input_drop/dropout/Cast:y:0*
T0*+
_output_shapes
:���������}2#
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
T0*/
_output_shapes
:���������}2#
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
T0*/
_output_shapes
:���������}@*
paddingSAME*
strides
2
stream_0_conv_1/conv1d�
stream_0_conv_1/conv1d/SqueezeSqueezestream_0_conv_1/conv1d:output:0*
T0*+
_output_shapes
:���������}@*
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
T0*+
_output_shapes
:���������}@2
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
:@*
	keep_dims(2"
 batch_normalization/moments/mean�
(batch_normalization/moments/StopGradientStopGradient)batch_normalization/moments/mean:output:0*
T0*"
_output_shapes
:@2*
(batch_normalization/moments/StopGradient�
-batch_normalization/moments/SquaredDifferenceSquaredDifference stream_0_conv_1/BiasAdd:output:01batch_normalization/moments/StopGradient:output:0*
T0*+
_output_shapes
:���������}@2/
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
T0*+
_output_shapes
:���������}@2%
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
T0*+
_output_shapes
:���������}@2%
#batch_normalization/batchnorm/add_1�
activation/ReluRelu'batch_normalization/batchnorm/add_1:z:0*
T0*+
_output_shapes
:���������}@2
activation/Relu�
stream_0_drop_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2
stream_0_drop_1/dropout/Const�
stream_0_drop_1/dropout/MulMulactivation/Relu:activations:0&stream_0_drop_1/dropout/Const:output:0*
T0*+
_output_shapes
:���������}@2
stream_0_drop_1/dropout/Mul�
stream_0_drop_1/dropout/ShapeShapeactivation/Relu:activations:0*
T0*
_output_shapes
:2
stream_0_drop_1/dropout/Shape�
4stream_0_drop_1/dropout/random_uniform/RandomUniformRandomUniform&stream_0_drop_1/dropout/Shape:output:0*
T0*+
_output_shapes
:���������}@*
dtype0*
seed�*
seed2�26
4stream_0_drop_1/dropout/random_uniform/RandomUniform�
&stream_0_drop_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���>2(
&stream_0_drop_1/dropout/GreaterEqual/y�
$stream_0_drop_1/dropout/GreaterEqualGreaterEqual=stream_0_drop_1/dropout/random_uniform/RandomUniform:output:0/stream_0_drop_1/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:���������}@2&
$stream_0_drop_1/dropout/GreaterEqual�
stream_0_drop_1/dropout/CastCast(stream_0_drop_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:���������}@2
stream_0_drop_1/dropout/Cast�
stream_0_drop_1/dropout/Mul_1Mulstream_0_drop_1/dropout/Mul:z:0 stream_0_drop_1/dropout/Cast:y:0*
T0*+
_output_shapes
:���������}@2
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
T0*/
_output_shapes
:���������}@2#
!stream_0_conv_2/conv1d/ExpandDims�
2stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp;stream_0_conv_2_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:@�*
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
T0*'
_output_shapes
:@�2%
#stream_0_conv_2/conv1d/ExpandDims_1�
stream_0_conv_2/conv1dConv2D*stream_0_conv_2/conv1d/ExpandDims:output:0,stream_0_conv_2/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:���������}�*
paddingSAME*
strides
2
stream_0_conv_2/conv1d�
stream_0_conv_2/conv1d/SqueezeSqueezestream_0_conv_2/conv1d:output:0*
T0*,
_output_shapes
:���������}�*
squeeze_dims

���������2 
stream_0_conv_2/conv1d/Squeeze�
&stream_0_conv_2/BiasAdd/ReadVariableOpReadVariableOp/stream_0_conv_2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02(
&stream_0_conv_2/BiasAdd/ReadVariableOp�
stream_0_conv_2/BiasAddBiasAdd'stream_0_conv_2/conv1d/Squeeze:output:0.stream_0_conv_2/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������}�2
stream_0_conv_2/BiasAdd�
4batch_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       26
4batch_normalization_1/moments/mean/reduction_indices�
"batch_normalization_1/moments/meanMean stream_0_conv_2/BiasAdd:output:0=batch_normalization_1/moments/mean/reduction_indices:output:0*
T0*#
_output_shapes
:�*
	keep_dims(2$
"batch_normalization_1/moments/mean�
*batch_normalization_1/moments/StopGradientStopGradient+batch_normalization_1/moments/mean:output:0*
T0*#
_output_shapes
:�2,
*batch_normalization_1/moments/StopGradient�
/batch_normalization_1/moments/SquaredDifferenceSquaredDifference stream_0_conv_2/BiasAdd:output:03batch_normalization_1/moments/StopGradient:output:0*
T0*,
_output_shapes
:���������}�21
/batch_normalization_1/moments/SquaredDifference�
8batch_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2:
8batch_normalization_1/moments/variance/reduction_indices�
&batch_normalization_1/moments/varianceMean3batch_normalization_1/moments/SquaredDifference:z:0Abatch_normalization_1/moments/variance/reduction_indices:output:0*
T0*#
_output_shapes
:�*
	keep_dims(2(
&batch_normalization_1/moments/variance�
%batch_normalization_1/moments/SqueezeSqueeze+batch_normalization_1/moments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 2'
%batch_normalization_1/moments/Squeeze�
'batch_normalization_1/moments/Squeeze_1Squeeze/batch_normalization_1/moments/variance:output:0*
T0*
_output_shapes	
:�*
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
4batch_normalization_1/AssignMovingAvg/ReadVariableOpReadVariableOp=batch_normalization_1_assignmovingavg_readvariableop_resource*
_output_shapes	
:�*
dtype026
4batch_normalization_1/AssignMovingAvg/ReadVariableOp�
)batch_normalization_1/AssignMovingAvg/subSub<batch_normalization_1/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_1/moments/Squeeze:output:0*
T0*
_output_shapes	
:�2+
)batch_normalization_1/AssignMovingAvg/sub�
)batch_normalization_1/AssignMovingAvg/mulMul-batch_normalization_1/AssignMovingAvg/sub:z:04batch_normalization_1/AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:�2+
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
6batch_normalization_1/AssignMovingAvg_1/ReadVariableOpReadVariableOp?batch_normalization_1_assignmovingavg_1_readvariableop_resource*
_output_shapes	
:�*
dtype028
6batch_normalization_1/AssignMovingAvg_1/ReadVariableOp�
+batch_normalization_1/AssignMovingAvg_1/subSub>batch_normalization_1/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_1/moments/Squeeze_1:output:0*
T0*
_output_shapes	
:�2-
+batch_normalization_1/AssignMovingAvg_1/sub�
+batch_normalization_1/AssignMovingAvg_1/mulMul/batch_normalization_1/AssignMovingAvg_1/sub:z:06batch_normalization_1/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:�2-
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
T0*
_output_shapes	
:�2%
#batch_normalization_1/batchnorm/add�
%batch_normalization_1/batchnorm/RsqrtRsqrt'batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes	
:�2'
%batch_normalization_1/batchnorm/Rsqrt�
2batch_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype024
2batch_normalization_1/batchnorm/mul/ReadVariableOp�
#batch_normalization_1/batchnorm/mulMul)batch_normalization_1/batchnorm/Rsqrt:y:0:batch_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2%
#batch_normalization_1/batchnorm/mul�
%batch_normalization_1/batchnorm/mul_1Mul stream_0_conv_2/BiasAdd:output:0'batch_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:���������}�2'
%batch_normalization_1/batchnorm/mul_1�
%batch_normalization_1/batchnorm/mul_2Mul.batch_normalization_1/moments/Squeeze:output:0'batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes	
:�2'
%batch_normalization_1/batchnorm/mul_2�
.batch_normalization_1/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_1_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype020
.batch_normalization_1/batchnorm/ReadVariableOp�
#batch_normalization_1/batchnorm/subSub6batch_normalization_1/batchnorm/ReadVariableOp:value:0)batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�2%
#batch_normalization_1/batchnorm/sub�
%batch_normalization_1/batchnorm/add_1AddV2)batch_normalization_1/batchnorm/mul_1:z:0'batch_normalization_1/batchnorm/sub:z:0*
T0*,
_output_shapes
:���������}�2'
%batch_normalization_1/batchnorm/add_1�
activation_1/ReluRelu)batch_normalization_1/batchnorm/add_1:z:0*
T0*,
_output_shapes
:���������}�2
activation_1/Relu�
stream_0_drop_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU�?2
stream_0_drop_2/dropout/Const�
stream_0_drop_2/dropout/MulMulactivation_1/Relu:activations:0&stream_0_drop_2/dropout/Const:output:0*
T0*,
_output_shapes
:���������}�2
stream_0_drop_2/dropout/Mul�
stream_0_drop_2/dropout/ShapeShapeactivation_1/Relu:activations:0*
T0*
_output_shapes
:2
stream_0_drop_2/dropout/Shape�
4stream_0_drop_2/dropout/random_uniform/RandomUniformRandomUniform&stream_0_drop_2/dropout/Shape:output:0*
T0*,
_output_shapes
:���������}�*
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
:���������}�2&
$stream_0_drop_2/dropout/GreaterEqual�
stream_0_drop_2/dropout/CastCast(stream_0_drop_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:���������}�2
stream_0_drop_2/dropout/Cast�
stream_0_drop_2/dropout/Mul_1Mulstream_0_drop_2/dropout/Mul:z:0 stream_0_drop_2/dropout/Cast:y:0*
T0*,
_output_shapes
:���������}�2
stream_0_drop_2/dropout/Mul_1�
%stream_0_conv_3/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2'
%stream_0_conv_3/conv1d/ExpandDims/dim�
!stream_0_conv_3/conv1d/ExpandDims
ExpandDims!stream_0_drop_2/dropout/Mul_1:z:0.stream_0_conv_3/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:���������}�2#
!stream_0_conv_3/conv1d/ExpandDims�
2stream_0_conv_3/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp;stream_0_conv_3_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:��*
dtype024
2stream_0_conv_3/conv1d/ExpandDims_1/ReadVariableOp�
'stream_0_conv_3/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2)
'stream_0_conv_3/conv1d/ExpandDims_1/dim�
#stream_0_conv_3/conv1d/ExpandDims_1
ExpandDims:stream_0_conv_3/conv1d/ExpandDims_1/ReadVariableOp:value:00stream_0_conv_3/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:��2%
#stream_0_conv_3/conv1d/ExpandDims_1�
stream_0_conv_3/conv1dConv2D*stream_0_conv_3/conv1d/ExpandDims:output:0,stream_0_conv_3/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:���������}�*
paddingSAME*
strides
2
stream_0_conv_3/conv1d�
stream_0_conv_3/conv1d/SqueezeSqueezestream_0_conv_3/conv1d:output:0*
T0*,
_output_shapes
:���������}�*
squeeze_dims

���������2 
stream_0_conv_3/conv1d/Squeeze�
&stream_0_conv_3/BiasAdd/ReadVariableOpReadVariableOp/stream_0_conv_3_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02(
&stream_0_conv_3/BiasAdd/ReadVariableOp�
stream_0_conv_3/BiasAddBiasAdd'stream_0_conv_3/conv1d/Squeeze:output:0.stream_0_conv_3/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������}�2
stream_0_conv_3/BiasAdd�
4batch_normalization_2/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       26
4batch_normalization_2/moments/mean/reduction_indices�
"batch_normalization_2/moments/meanMean stream_0_conv_3/BiasAdd:output:0=batch_normalization_2/moments/mean/reduction_indices:output:0*
T0*#
_output_shapes
:�*
	keep_dims(2$
"batch_normalization_2/moments/mean�
*batch_normalization_2/moments/StopGradientStopGradient+batch_normalization_2/moments/mean:output:0*
T0*#
_output_shapes
:�2,
*batch_normalization_2/moments/StopGradient�
/batch_normalization_2/moments/SquaredDifferenceSquaredDifference stream_0_conv_3/BiasAdd:output:03batch_normalization_2/moments/StopGradient:output:0*
T0*,
_output_shapes
:���������}�21
/batch_normalization_2/moments/SquaredDifference�
8batch_normalization_2/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2:
8batch_normalization_2/moments/variance/reduction_indices�
&batch_normalization_2/moments/varianceMean3batch_normalization_2/moments/SquaredDifference:z:0Abatch_normalization_2/moments/variance/reduction_indices:output:0*
T0*#
_output_shapes
:�*
	keep_dims(2(
&batch_normalization_2/moments/variance�
%batch_normalization_2/moments/SqueezeSqueeze+batch_normalization_2/moments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 2'
%batch_normalization_2/moments/Squeeze�
'batch_normalization_2/moments/Squeeze_1Squeeze/batch_normalization_2/moments/variance:output:0*
T0*
_output_shapes	
:�*
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
4batch_normalization_2/AssignMovingAvg/ReadVariableOpReadVariableOp=batch_normalization_2_assignmovingavg_readvariableop_resource*
_output_shapes	
:�*
dtype026
4batch_normalization_2/AssignMovingAvg/ReadVariableOp�
)batch_normalization_2/AssignMovingAvg/subSub<batch_normalization_2/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_2/moments/Squeeze:output:0*
T0*
_output_shapes	
:�2+
)batch_normalization_2/AssignMovingAvg/sub�
)batch_normalization_2/AssignMovingAvg/mulMul-batch_normalization_2/AssignMovingAvg/sub:z:04batch_normalization_2/AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:�2+
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
6batch_normalization_2/AssignMovingAvg_1/ReadVariableOpReadVariableOp?batch_normalization_2_assignmovingavg_1_readvariableop_resource*
_output_shapes	
:�*
dtype028
6batch_normalization_2/AssignMovingAvg_1/ReadVariableOp�
+batch_normalization_2/AssignMovingAvg_1/subSub>batch_normalization_2/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_2/moments/Squeeze_1:output:0*
T0*
_output_shapes	
:�2-
+batch_normalization_2/AssignMovingAvg_1/sub�
+batch_normalization_2/AssignMovingAvg_1/mulMul/batch_normalization_2/AssignMovingAvg_1/sub:z:06batch_normalization_2/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:�2-
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
T0*
_output_shapes	
:�2%
#batch_normalization_2/batchnorm/add�
%batch_normalization_2/batchnorm/RsqrtRsqrt'batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes	
:�2'
%batch_normalization_2/batchnorm/Rsqrt�
2batch_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype024
2batch_normalization_2/batchnorm/mul/ReadVariableOp�
#batch_normalization_2/batchnorm/mulMul)batch_normalization_2/batchnorm/Rsqrt:y:0:batch_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2%
#batch_normalization_2/batchnorm/mul�
%batch_normalization_2/batchnorm/mul_1Mul stream_0_conv_3/BiasAdd:output:0'batch_normalization_2/batchnorm/mul:z:0*
T0*,
_output_shapes
:���������}�2'
%batch_normalization_2/batchnorm/mul_1�
%batch_normalization_2/batchnorm/mul_2Mul.batch_normalization_2/moments/Squeeze:output:0'batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes	
:�2'
%batch_normalization_2/batchnorm/mul_2�
.batch_normalization_2/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_2_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype020
.batch_normalization_2/batchnorm/ReadVariableOp�
#batch_normalization_2/batchnorm/subSub6batch_normalization_2/batchnorm/ReadVariableOp:value:0)batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�2%
#batch_normalization_2/batchnorm/sub�
%batch_normalization_2/batchnorm/add_1AddV2)batch_normalization_2/batchnorm/mul_1:z:0'batch_normalization_2/batchnorm/sub:z:0*
T0*,
_output_shapes
:���������}�2'
%batch_normalization_2/batchnorm/add_1�
activation_2/ReluRelu)batch_normalization_2/batchnorm/add_1:z:0*
T0*,
_output_shapes
:���������}�2
activation_2/Relu�
stream_0_drop_3/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
stream_0_drop_3/dropout/Const�
stream_0_drop_3/dropout/MulMulactivation_2/Relu:activations:0&stream_0_drop_3/dropout/Const:output:0*
T0*,
_output_shapes
:���������}�2
stream_0_drop_3/dropout/Mul�
stream_0_drop_3/dropout/ShapeShapeactivation_2/Relu:activations:0*
T0*
_output_shapes
:2
stream_0_drop_3/dropout/Shape�
4stream_0_drop_3/dropout/random_uniform/RandomUniformRandomUniform&stream_0_drop_3/dropout/Shape:output:0*
T0*,
_output_shapes
:���������}�*
dtype0*
seed�*
seed2�26
4stream_0_drop_3/dropout/random_uniform/RandomUniform�
&stream_0_drop_3/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2(
&stream_0_drop_3/dropout/GreaterEqual/y�
$stream_0_drop_3/dropout/GreaterEqualGreaterEqual=stream_0_drop_3/dropout/random_uniform/RandomUniform:output:0/stream_0_drop_3/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:���������}�2&
$stream_0_drop_3/dropout/GreaterEqual�
stream_0_drop_3/dropout/CastCast(stream_0_drop_3/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:���������}�2
stream_0_drop_3/dropout/Cast�
stream_0_drop_3/dropout/Mul_1Mulstream_0_drop_3/dropout/Mul:z:0 stream_0_drop_3/dropout/Cast:y:0*
T0*,
_output_shapes
:���������}�2
stream_0_drop_3/dropout/Mul_1�
/global_average_pooling1d/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :21
/global_average_pooling1d/Mean/reduction_indices�
global_average_pooling1d/MeanMean!stream_0_drop_3/dropout/Mul_1:z:08global_average_pooling1d/Mean/reduction_indices:output:0*
T0*(
_output_shapes
:����������2
global_average_pooling1d/Mean�
dense_1_dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
dense_1_dropout/dropout/Const�
dense_1_dropout/dropout/MulMul&global_average_pooling1d/Mean:output:0&dense_1_dropout/dropout/Const:output:0*
T0*(
_output_shapes
:����������2
dense_1_dropout/dropout/Mul�
dense_1_dropout/dropout/ShapeShape&global_average_pooling1d/Mean:output:0*
T0*
_output_shapes
:2
dense_1_dropout/dropout/Shape�
4dense_1_dropout/dropout/random_uniform/RandomUniformRandomUniform&dense_1_dropout/dropout/Shape:output:0*
T0*(
_output_shapes
:����������*
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
T0*(
_output_shapes
:����������2&
$dense_1_dropout/dropout/GreaterEqual�
dense_1_dropout/dropout/CastCast(dense_1_dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:����������2
dense_1_dropout/dropout/Cast�
dense_1_dropout/dropout/Mul_1Muldense_1_dropout/dropout/Mul:z:0 dense_1_dropout/dropout/Cast:y:0*
T0*(
_output_shapes
:����������2
dense_1_dropout/dropout/Mul_1�
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	�T*
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
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;stream_0_conv_2_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:@�*
dtype02:
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp�
)stream_0_conv_2/kernel/Regularizer/SquareSquare@stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*#
_output_shapes
:@�2+
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
5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp;stream_0_conv_3_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:��*
dtype027
5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp�
&stream_0_conv_3/kernel/Regularizer/AbsAbs=stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*$
_output_shapes
:��2(
&stream_0_conv_3/kernel/Regularizer/Abs�
(stream_0_conv_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_0_conv_3/kernel/Regularizer/Const�
&stream_0_conv_3/kernel/Regularizer/SumSum*stream_0_conv_3/kernel/Regularizer/Abs:y:01stream_0_conv_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_3/kernel/Regularizer/Sum�
(stream_0_conv_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2*
(stream_0_conv_3/kernel/Regularizer/mul/x�
&stream_0_conv_3/kernel/Regularizer/mulMul1stream_0_conv_3/kernel/Regularizer/mul/x:output:0/stream_0_conv_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_3/kernel/Regularizer/mul�
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	�T*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOp�
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	�T2 
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
NoOpNoOp$^batch_normalization/AssignMovingAvg3^batch_normalization/AssignMovingAvg/ReadVariableOp&^batch_normalization/AssignMovingAvg_15^batch_normalization/AssignMovingAvg_1/ReadVariableOp-^batch_normalization/batchnorm/ReadVariableOp1^batch_normalization/batchnorm/mul/ReadVariableOp&^batch_normalization_1/AssignMovingAvg5^batch_normalization_1/AssignMovingAvg/ReadVariableOp(^batch_normalization_1/AssignMovingAvg_17^batch_normalization_1/AssignMovingAvg_1/ReadVariableOp/^batch_normalization_1/batchnorm/ReadVariableOp3^batch_normalization_1/batchnorm/mul/ReadVariableOp&^batch_normalization_2/AssignMovingAvg5^batch_normalization_2/AssignMovingAvg/ReadVariableOp(^batch_normalization_2/AssignMovingAvg_17^batch_normalization_2/AssignMovingAvg_1/ReadVariableOp/^batch_normalization_2/batchnorm/ReadVariableOp3^batch_normalization_2/batchnorm/mul/ReadVariableOp&^batch_normalization_3/AssignMovingAvg5^batch_normalization_3/AssignMovingAvg/ReadVariableOp(^batch_normalization_3/AssignMovingAvg_17^batch_normalization_3/AssignMovingAvg_1/ReadVariableOp/^batch_normalization_3/batchnorm/ReadVariableOp3^batch_normalization_3/batchnorm/mul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp.^dense_1/kernel/Regularizer/Abs/ReadVariableOp'^stream_0_conv_1/BiasAdd/ReadVariableOp3^stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp'^stream_0_conv_2/BiasAdd/ReadVariableOp3^stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOp9^stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp'^stream_0_conv_3/BiasAdd/ReadVariableOp3^stream_0_conv_3/conv1d/ExpandDims_1/ReadVariableOp6^stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:���������}: : : : : : : : : : : : : : : : : : : : : : : : 2J
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
:���������}
 
_user_specified_nameinputs
��
�
F__inference_basemodel_layer_call_and_return_conditional_losses_8908718

inputsQ
;stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource:@=
/stream_0_conv_1_biasadd_readvariableop_resource:@I
;batch_normalization_assignmovingavg_readvariableop_resource:@K
=batch_normalization_assignmovingavg_1_readvariableop_resource:@G
9batch_normalization_batchnorm_mul_readvariableop_resource:@C
5batch_normalization_batchnorm_readvariableop_resource:@R
;stream_0_conv_2_conv1d_expanddims_1_readvariableop_resource:@�>
/stream_0_conv_2_biasadd_readvariableop_resource:	�L
=batch_normalization_1_assignmovingavg_readvariableop_resource:	�N
?batch_normalization_1_assignmovingavg_1_readvariableop_resource:	�J
;batch_normalization_1_batchnorm_mul_readvariableop_resource:	�F
7batch_normalization_1_batchnorm_readvariableop_resource:	�S
;stream_0_conv_3_conv1d_expanddims_1_readvariableop_resource:��>
/stream_0_conv_3_biasadd_readvariableop_resource:	�L
=batch_normalization_2_assignmovingavg_readvariableop_resource:	�N
?batch_normalization_2_assignmovingavg_1_readvariableop_resource:	�J
;batch_normalization_2_batchnorm_mul_readvariableop_resource:	�F
7batch_normalization_2_batchnorm_readvariableop_resource:	�9
&dense_1_matmul_readvariableop_resource:	�T5
'dense_1_biasadd_readvariableop_resource:TK
=batch_normalization_3_assignmovingavg_readvariableop_resource:TM
?batch_normalization_3_assignmovingavg_1_readvariableop_resource:TI
;batch_normalization_3_batchnorm_mul_readvariableop_resource:TE
7batch_normalization_3_batchnorm_readvariableop_resource:T
identity��#batch_normalization/AssignMovingAvg�2batch_normalization/AssignMovingAvg/ReadVariableOp�%batch_normalization/AssignMovingAvg_1�4batch_normalization/AssignMovingAvg_1/ReadVariableOp�,batch_normalization/batchnorm/ReadVariableOp�0batch_normalization/batchnorm/mul/ReadVariableOp�%batch_normalization_1/AssignMovingAvg�4batch_normalization_1/AssignMovingAvg/ReadVariableOp�'batch_normalization_1/AssignMovingAvg_1�6batch_normalization_1/AssignMovingAvg_1/ReadVariableOp�.batch_normalization_1/batchnorm/ReadVariableOp�2batch_normalization_1/batchnorm/mul/ReadVariableOp�%batch_normalization_2/AssignMovingAvg�4batch_normalization_2/AssignMovingAvg/ReadVariableOp�'batch_normalization_2/AssignMovingAvg_1�6batch_normalization_2/AssignMovingAvg_1/ReadVariableOp�.batch_normalization_2/batchnorm/ReadVariableOp�2batch_normalization_2/batchnorm/mul/ReadVariableOp�%batch_normalization_3/AssignMovingAvg�4batch_normalization_3/AssignMovingAvg/ReadVariableOp�'batch_normalization_3/AssignMovingAvg_1�6batch_normalization_3/AssignMovingAvg_1/ReadVariableOp�.batch_normalization_3/batchnorm/ReadVariableOp�2batch_normalization_3/batchnorm/mul/ReadVariableOp�dense_1/BiasAdd/ReadVariableOp�dense_1/MatMul/ReadVariableOp�-dense_1/kernel/Regularizer/Abs/ReadVariableOp�&stream_0_conv_1/BiasAdd/ReadVariableOp�2stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp�5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp�&stream_0_conv_2/BiasAdd/ReadVariableOp�2stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOp�8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp�&stream_0_conv_3/BiasAdd/ReadVariableOp�2stream_0_conv_3/conv1d/ExpandDims_1/ReadVariableOp�5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp�
!stream_0_input_drop/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2#
!stream_0_input_drop/dropout/Const�
stream_0_input_drop/dropout/MulMulinputs*stream_0_input_drop/dropout/Const:output:0*
T0*+
_output_shapes
:���������}2!
stream_0_input_drop/dropout/Mul|
!stream_0_input_drop/dropout/ShapeShapeinputs*
T0*
_output_shapes
:2#
!stream_0_input_drop/dropout/Shape�
8stream_0_input_drop/dropout/random_uniform/RandomUniformRandomUniform*stream_0_input_drop/dropout/Shape:output:0*
T0*+
_output_shapes
:���������}*
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
T0*+
_output_shapes
:���������}2*
(stream_0_input_drop/dropout/GreaterEqual�
 stream_0_input_drop/dropout/CastCast,stream_0_input_drop/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:���������}2"
 stream_0_input_drop/dropout/Cast�
!stream_0_input_drop/dropout/Mul_1Mul#stream_0_input_drop/dropout/Mul:z:0$stream_0_input_drop/dropout/Cast:y:0*
T0*+
_output_shapes
:���������}2#
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
T0*/
_output_shapes
:���������}2#
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
T0*/
_output_shapes
:���������}@*
paddingSAME*
strides
2
stream_0_conv_1/conv1d�
stream_0_conv_1/conv1d/SqueezeSqueezestream_0_conv_1/conv1d:output:0*
T0*+
_output_shapes
:���������}@*
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
T0*+
_output_shapes
:���������}@2
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
:@*
	keep_dims(2"
 batch_normalization/moments/mean�
(batch_normalization/moments/StopGradientStopGradient)batch_normalization/moments/mean:output:0*
T0*"
_output_shapes
:@2*
(batch_normalization/moments/StopGradient�
-batch_normalization/moments/SquaredDifferenceSquaredDifference stream_0_conv_1/BiasAdd:output:01batch_normalization/moments/StopGradient:output:0*
T0*+
_output_shapes
:���������}@2/
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
T0*+
_output_shapes
:���������}@2%
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
T0*+
_output_shapes
:���������}@2%
#batch_normalization/batchnorm/add_1�
activation/ReluRelu'batch_normalization/batchnorm/add_1:z:0*
T0*+
_output_shapes
:���������}@2
activation/Relu�
stream_0_drop_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2
stream_0_drop_1/dropout/Const�
stream_0_drop_1/dropout/MulMulactivation/Relu:activations:0&stream_0_drop_1/dropout/Const:output:0*
T0*+
_output_shapes
:���������}@2
stream_0_drop_1/dropout/Mul�
stream_0_drop_1/dropout/ShapeShapeactivation/Relu:activations:0*
T0*
_output_shapes
:2
stream_0_drop_1/dropout/Shape�
4stream_0_drop_1/dropout/random_uniform/RandomUniformRandomUniform&stream_0_drop_1/dropout/Shape:output:0*
T0*+
_output_shapes
:���������}@*
dtype0*
seed�*
seed2�26
4stream_0_drop_1/dropout/random_uniform/RandomUniform�
&stream_0_drop_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���>2(
&stream_0_drop_1/dropout/GreaterEqual/y�
$stream_0_drop_1/dropout/GreaterEqualGreaterEqual=stream_0_drop_1/dropout/random_uniform/RandomUniform:output:0/stream_0_drop_1/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:���������}@2&
$stream_0_drop_1/dropout/GreaterEqual�
stream_0_drop_1/dropout/CastCast(stream_0_drop_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:���������}@2
stream_0_drop_1/dropout/Cast�
stream_0_drop_1/dropout/Mul_1Mulstream_0_drop_1/dropout/Mul:z:0 stream_0_drop_1/dropout/Cast:y:0*
T0*+
_output_shapes
:���������}@2
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
T0*/
_output_shapes
:���������}@2#
!stream_0_conv_2/conv1d/ExpandDims�
2stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp;stream_0_conv_2_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:@�*
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
T0*'
_output_shapes
:@�2%
#stream_0_conv_2/conv1d/ExpandDims_1�
stream_0_conv_2/conv1dConv2D*stream_0_conv_2/conv1d/ExpandDims:output:0,stream_0_conv_2/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:���������}�*
paddingSAME*
strides
2
stream_0_conv_2/conv1d�
stream_0_conv_2/conv1d/SqueezeSqueezestream_0_conv_2/conv1d:output:0*
T0*,
_output_shapes
:���������}�*
squeeze_dims

���������2 
stream_0_conv_2/conv1d/Squeeze�
&stream_0_conv_2/BiasAdd/ReadVariableOpReadVariableOp/stream_0_conv_2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02(
&stream_0_conv_2/BiasAdd/ReadVariableOp�
stream_0_conv_2/BiasAddBiasAdd'stream_0_conv_2/conv1d/Squeeze:output:0.stream_0_conv_2/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������}�2
stream_0_conv_2/BiasAdd�
4batch_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       26
4batch_normalization_1/moments/mean/reduction_indices�
"batch_normalization_1/moments/meanMean stream_0_conv_2/BiasAdd:output:0=batch_normalization_1/moments/mean/reduction_indices:output:0*
T0*#
_output_shapes
:�*
	keep_dims(2$
"batch_normalization_1/moments/mean�
*batch_normalization_1/moments/StopGradientStopGradient+batch_normalization_1/moments/mean:output:0*
T0*#
_output_shapes
:�2,
*batch_normalization_1/moments/StopGradient�
/batch_normalization_1/moments/SquaredDifferenceSquaredDifference stream_0_conv_2/BiasAdd:output:03batch_normalization_1/moments/StopGradient:output:0*
T0*,
_output_shapes
:���������}�21
/batch_normalization_1/moments/SquaredDifference�
8batch_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2:
8batch_normalization_1/moments/variance/reduction_indices�
&batch_normalization_1/moments/varianceMean3batch_normalization_1/moments/SquaredDifference:z:0Abatch_normalization_1/moments/variance/reduction_indices:output:0*
T0*#
_output_shapes
:�*
	keep_dims(2(
&batch_normalization_1/moments/variance�
%batch_normalization_1/moments/SqueezeSqueeze+batch_normalization_1/moments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 2'
%batch_normalization_1/moments/Squeeze�
'batch_normalization_1/moments/Squeeze_1Squeeze/batch_normalization_1/moments/variance:output:0*
T0*
_output_shapes	
:�*
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
4batch_normalization_1/AssignMovingAvg/ReadVariableOpReadVariableOp=batch_normalization_1_assignmovingavg_readvariableop_resource*
_output_shapes	
:�*
dtype026
4batch_normalization_1/AssignMovingAvg/ReadVariableOp�
)batch_normalization_1/AssignMovingAvg/subSub<batch_normalization_1/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_1/moments/Squeeze:output:0*
T0*
_output_shapes	
:�2+
)batch_normalization_1/AssignMovingAvg/sub�
)batch_normalization_1/AssignMovingAvg/mulMul-batch_normalization_1/AssignMovingAvg/sub:z:04batch_normalization_1/AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:�2+
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
6batch_normalization_1/AssignMovingAvg_1/ReadVariableOpReadVariableOp?batch_normalization_1_assignmovingavg_1_readvariableop_resource*
_output_shapes	
:�*
dtype028
6batch_normalization_1/AssignMovingAvg_1/ReadVariableOp�
+batch_normalization_1/AssignMovingAvg_1/subSub>batch_normalization_1/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_1/moments/Squeeze_1:output:0*
T0*
_output_shapes	
:�2-
+batch_normalization_1/AssignMovingAvg_1/sub�
+batch_normalization_1/AssignMovingAvg_1/mulMul/batch_normalization_1/AssignMovingAvg_1/sub:z:06batch_normalization_1/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:�2-
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
T0*
_output_shapes	
:�2%
#batch_normalization_1/batchnorm/add�
%batch_normalization_1/batchnorm/RsqrtRsqrt'batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes	
:�2'
%batch_normalization_1/batchnorm/Rsqrt�
2batch_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype024
2batch_normalization_1/batchnorm/mul/ReadVariableOp�
#batch_normalization_1/batchnorm/mulMul)batch_normalization_1/batchnorm/Rsqrt:y:0:batch_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2%
#batch_normalization_1/batchnorm/mul�
%batch_normalization_1/batchnorm/mul_1Mul stream_0_conv_2/BiasAdd:output:0'batch_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:���������}�2'
%batch_normalization_1/batchnorm/mul_1�
%batch_normalization_1/batchnorm/mul_2Mul.batch_normalization_1/moments/Squeeze:output:0'batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes	
:�2'
%batch_normalization_1/batchnorm/mul_2�
.batch_normalization_1/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_1_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype020
.batch_normalization_1/batchnorm/ReadVariableOp�
#batch_normalization_1/batchnorm/subSub6batch_normalization_1/batchnorm/ReadVariableOp:value:0)batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�2%
#batch_normalization_1/batchnorm/sub�
%batch_normalization_1/batchnorm/add_1AddV2)batch_normalization_1/batchnorm/mul_1:z:0'batch_normalization_1/batchnorm/sub:z:0*
T0*,
_output_shapes
:���������}�2'
%batch_normalization_1/batchnorm/add_1�
activation_1/ReluRelu)batch_normalization_1/batchnorm/add_1:z:0*
T0*,
_output_shapes
:���������}�2
activation_1/Relu�
stream_0_drop_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU�?2
stream_0_drop_2/dropout/Const�
stream_0_drop_2/dropout/MulMulactivation_1/Relu:activations:0&stream_0_drop_2/dropout/Const:output:0*
T0*,
_output_shapes
:���������}�2
stream_0_drop_2/dropout/Mul�
stream_0_drop_2/dropout/ShapeShapeactivation_1/Relu:activations:0*
T0*
_output_shapes
:2
stream_0_drop_2/dropout/Shape�
4stream_0_drop_2/dropout/random_uniform/RandomUniformRandomUniform&stream_0_drop_2/dropout/Shape:output:0*
T0*,
_output_shapes
:���������}�*
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
:���������}�2&
$stream_0_drop_2/dropout/GreaterEqual�
stream_0_drop_2/dropout/CastCast(stream_0_drop_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:���������}�2
stream_0_drop_2/dropout/Cast�
stream_0_drop_2/dropout/Mul_1Mulstream_0_drop_2/dropout/Mul:z:0 stream_0_drop_2/dropout/Cast:y:0*
T0*,
_output_shapes
:���������}�2
stream_0_drop_2/dropout/Mul_1�
%stream_0_conv_3/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2'
%stream_0_conv_3/conv1d/ExpandDims/dim�
!stream_0_conv_3/conv1d/ExpandDims
ExpandDims!stream_0_drop_2/dropout/Mul_1:z:0.stream_0_conv_3/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:���������}�2#
!stream_0_conv_3/conv1d/ExpandDims�
2stream_0_conv_3/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp;stream_0_conv_3_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:��*
dtype024
2stream_0_conv_3/conv1d/ExpandDims_1/ReadVariableOp�
'stream_0_conv_3/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2)
'stream_0_conv_3/conv1d/ExpandDims_1/dim�
#stream_0_conv_3/conv1d/ExpandDims_1
ExpandDims:stream_0_conv_3/conv1d/ExpandDims_1/ReadVariableOp:value:00stream_0_conv_3/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:��2%
#stream_0_conv_3/conv1d/ExpandDims_1�
stream_0_conv_3/conv1dConv2D*stream_0_conv_3/conv1d/ExpandDims:output:0,stream_0_conv_3/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:���������}�*
paddingSAME*
strides
2
stream_0_conv_3/conv1d�
stream_0_conv_3/conv1d/SqueezeSqueezestream_0_conv_3/conv1d:output:0*
T0*,
_output_shapes
:���������}�*
squeeze_dims

���������2 
stream_0_conv_3/conv1d/Squeeze�
&stream_0_conv_3/BiasAdd/ReadVariableOpReadVariableOp/stream_0_conv_3_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02(
&stream_0_conv_3/BiasAdd/ReadVariableOp�
stream_0_conv_3/BiasAddBiasAdd'stream_0_conv_3/conv1d/Squeeze:output:0.stream_0_conv_3/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������}�2
stream_0_conv_3/BiasAdd�
4batch_normalization_2/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       26
4batch_normalization_2/moments/mean/reduction_indices�
"batch_normalization_2/moments/meanMean stream_0_conv_3/BiasAdd:output:0=batch_normalization_2/moments/mean/reduction_indices:output:0*
T0*#
_output_shapes
:�*
	keep_dims(2$
"batch_normalization_2/moments/mean�
*batch_normalization_2/moments/StopGradientStopGradient+batch_normalization_2/moments/mean:output:0*
T0*#
_output_shapes
:�2,
*batch_normalization_2/moments/StopGradient�
/batch_normalization_2/moments/SquaredDifferenceSquaredDifference stream_0_conv_3/BiasAdd:output:03batch_normalization_2/moments/StopGradient:output:0*
T0*,
_output_shapes
:���������}�21
/batch_normalization_2/moments/SquaredDifference�
8batch_normalization_2/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2:
8batch_normalization_2/moments/variance/reduction_indices�
&batch_normalization_2/moments/varianceMean3batch_normalization_2/moments/SquaredDifference:z:0Abatch_normalization_2/moments/variance/reduction_indices:output:0*
T0*#
_output_shapes
:�*
	keep_dims(2(
&batch_normalization_2/moments/variance�
%batch_normalization_2/moments/SqueezeSqueeze+batch_normalization_2/moments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 2'
%batch_normalization_2/moments/Squeeze�
'batch_normalization_2/moments/Squeeze_1Squeeze/batch_normalization_2/moments/variance:output:0*
T0*
_output_shapes	
:�*
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
4batch_normalization_2/AssignMovingAvg/ReadVariableOpReadVariableOp=batch_normalization_2_assignmovingavg_readvariableop_resource*
_output_shapes	
:�*
dtype026
4batch_normalization_2/AssignMovingAvg/ReadVariableOp�
)batch_normalization_2/AssignMovingAvg/subSub<batch_normalization_2/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_2/moments/Squeeze:output:0*
T0*
_output_shapes	
:�2+
)batch_normalization_2/AssignMovingAvg/sub�
)batch_normalization_2/AssignMovingAvg/mulMul-batch_normalization_2/AssignMovingAvg/sub:z:04batch_normalization_2/AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:�2+
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
6batch_normalization_2/AssignMovingAvg_1/ReadVariableOpReadVariableOp?batch_normalization_2_assignmovingavg_1_readvariableop_resource*
_output_shapes	
:�*
dtype028
6batch_normalization_2/AssignMovingAvg_1/ReadVariableOp�
+batch_normalization_2/AssignMovingAvg_1/subSub>batch_normalization_2/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_2/moments/Squeeze_1:output:0*
T0*
_output_shapes	
:�2-
+batch_normalization_2/AssignMovingAvg_1/sub�
+batch_normalization_2/AssignMovingAvg_1/mulMul/batch_normalization_2/AssignMovingAvg_1/sub:z:06batch_normalization_2/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:�2-
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
T0*
_output_shapes	
:�2%
#batch_normalization_2/batchnorm/add�
%batch_normalization_2/batchnorm/RsqrtRsqrt'batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes	
:�2'
%batch_normalization_2/batchnorm/Rsqrt�
2batch_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype024
2batch_normalization_2/batchnorm/mul/ReadVariableOp�
#batch_normalization_2/batchnorm/mulMul)batch_normalization_2/batchnorm/Rsqrt:y:0:batch_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2%
#batch_normalization_2/batchnorm/mul�
%batch_normalization_2/batchnorm/mul_1Mul stream_0_conv_3/BiasAdd:output:0'batch_normalization_2/batchnorm/mul:z:0*
T0*,
_output_shapes
:���������}�2'
%batch_normalization_2/batchnorm/mul_1�
%batch_normalization_2/batchnorm/mul_2Mul.batch_normalization_2/moments/Squeeze:output:0'batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes	
:�2'
%batch_normalization_2/batchnorm/mul_2�
.batch_normalization_2/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_2_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype020
.batch_normalization_2/batchnorm/ReadVariableOp�
#batch_normalization_2/batchnorm/subSub6batch_normalization_2/batchnorm/ReadVariableOp:value:0)batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�2%
#batch_normalization_2/batchnorm/sub�
%batch_normalization_2/batchnorm/add_1AddV2)batch_normalization_2/batchnorm/mul_1:z:0'batch_normalization_2/batchnorm/sub:z:0*
T0*,
_output_shapes
:���������}�2'
%batch_normalization_2/batchnorm/add_1�
activation_2/ReluRelu)batch_normalization_2/batchnorm/add_1:z:0*
T0*,
_output_shapes
:���������}�2
activation_2/Relu�
stream_0_drop_3/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
stream_0_drop_3/dropout/Const�
stream_0_drop_3/dropout/MulMulactivation_2/Relu:activations:0&stream_0_drop_3/dropout/Const:output:0*
T0*,
_output_shapes
:���������}�2
stream_0_drop_3/dropout/Mul�
stream_0_drop_3/dropout/ShapeShapeactivation_2/Relu:activations:0*
T0*
_output_shapes
:2
stream_0_drop_3/dropout/Shape�
4stream_0_drop_3/dropout/random_uniform/RandomUniformRandomUniform&stream_0_drop_3/dropout/Shape:output:0*
T0*,
_output_shapes
:���������}�*
dtype0*
seed�*
seed2�26
4stream_0_drop_3/dropout/random_uniform/RandomUniform�
&stream_0_drop_3/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2(
&stream_0_drop_3/dropout/GreaterEqual/y�
$stream_0_drop_3/dropout/GreaterEqualGreaterEqual=stream_0_drop_3/dropout/random_uniform/RandomUniform:output:0/stream_0_drop_3/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:���������}�2&
$stream_0_drop_3/dropout/GreaterEqual�
stream_0_drop_3/dropout/CastCast(stream_0_drop_3/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:���������}�2
stream_0_drop_3/dropout/Cast�
stream_0_drop_3/dropout/Mul_1Mulstream_0_drop_3/dropout/Mul:z:0 stream_0_drop_3/dropout/Cast:y:0*
T0*,
_output_shapes
:���������}�2
stream_0_drop_3/dropout/Mul_1�
/global_average_pooling1d/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :21
/global_average_pooling1d/Mean/reduction_indices�
global_average_pooling1d/MeanMean!stream_0_drop_3/dropout/Mul_1:z:08global_average_pooling1d/Mean/reduction_indices:output:0*
T0*(
_output_shapes
:����������2
global_average_pooling1d/Mean�
dense_1_dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
dense_1_dropout/dropout/Const�
dense_1_dropout/dropout/MulMul&global_average_pooling1d/Mean:output:0&dense_1_dropout/dropout/Const:output:0*
T0*(
_output_shapes
:����������2
dense_1_dropout/dropout/Mul�
dense_1_dropout/dropout/ShapeShape&global_average_pooling1d/Mean:output:0*
T0*
_output_shapes
:2
dense_1_dropout/dropout/Shape�
4dense_1_dropout/dropout/random_uniform/RandomUniformRandomUniform&dense_1_dropout/dropout/Shape:output:0*
T0*(
_output_shapes
:����������*
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
T0*(
_output_shapes
:����������2&
$dense_1_dropout/dropout/GreaterEqual�
dense_1_dropout/dropout/CastCast(dense_1_dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:����������2
dense_1_dropout/dropout/Cast�
dense_1_dropout/dropout/Mul_1Muldense_1_dropout/dropout/Mul:z:0 dense_1_dropout/dropout/Cast:y:0*
T0*(
_output_shapes
:����������2
dense_1_dropout/dropout/Mul_1�
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	�T*
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
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;stream_0_conv_2_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:@�*
dtype02:
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp�
)stream_0_conv_2/kernel/Regularizer/SquareSquare@stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*#
_output_shapes
:@�2+
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
5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp;stream_0_conv_3_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:��*
dtype027
5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp�
&stream_0_conv_3/kernel/Regularizer/AbsAbs=stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*$
_output_shapes
:��2(
&stream_0_conv_3/kernel/Regularizer/Abs�
(stream_0_conv_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_0_conv_3/kernel/Regularizer/Const�
&stream_0_conv_3/kernel/Regularizer/SumSum*stream_0_conv_3/kernel/Regularizer/Abs:y:01stream_0_conv_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_3/kernel/Regularizer/Sum�
(stream_0_conv_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2*
(stream_0_conv_3/kernel/Regularizer/mul/x�
&stream_0_conv_3/kernel/Regularizer/mulMul1stream_0_conv_3/kernel/Regularizer/mul/x:output:0/stream_0_conv_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_3/kernel/Regularizer/mul�
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	�T*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOp�
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	�T2 
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
NoOpNoOp$^batch_normalization/AssignMovingAvg3^batch_normalization/AssignMovingAvg/ReadVariableOp&^batch_normalization/AssignMovingAvg_15^batch_normalization/AssignMovingAvg_1/ReadVariableOp-^batch_normalization/batchnorm/ReadVariableOp1^batch_normalization/batchnorm/mul/ReadVariableOp&^batch_normalization_1/AssignMovingAvg5^batch_normalization_1/AssignMovingAvg/ReadVariableOp(^batch_normalization_1/AssignMovingAvg_17^batch_normalization_1/AssignMovingAvg_1/ReadVariableOp/^batch_normalization_1/batchnorm/ReadVariableOp3^batch_normalization_1/batchnorm/mul/ReadVariableOp&^batch_normalization_2/AssignMovingAvg5^batch_normalization_2/AssignMovingAvg/ReadVariableOp(^batch_normalization_2/AssignMovingAvg_17^batch_normalization_2/AssignMovingAvg_1/ReadVariableOp/^batch_normalization_2/batchnorm/ReadVariableOp3^batch_normalization_2/batchnorm/mul/ReadVariableOp&^batch_normalization_3/AssignMovingAvg5^batch_normalization_3/AssignMovingAvg/ReadVariableOp(^batch_normalization_3/AssignMovingAvg_17^batch_normalization_3/AssignMovingAvg_1/ReadVariableOp/^batch_normalization_3/batchnorm/ReadVariableOp3^batch_normalization_3/batchnorm/mul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp.^dense_1/kernel/Regularizer/Abs/ReadVariableOp'^stream_0_conv_1/BiasAdd/ReadVariableOp3^stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp'^stream_0_conv_2/BiasAdd/ReadVariableOp3^stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOp9^stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp'^stream_0_conv_3/BiasAdd/ReadVariableOp3^stream_0_conv_3/conv1d/ExpandDims_1/ReadVariableOp6^stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:���������}: : : : : : : : : : : : : : : : : : : : : : : : 2J
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
:���������}
 
_user_specified_nameinputs
�
�
L__inference_stream_0_conv_2_layer_call_and_return_conditional_losses_8907104

inputsB
+conv1d_expanddims_1_readvariableop_resource:@�.
biasadd_readvariableop_resource:	�
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
T0*/
_output_shapes
:���������}@2
conv1d/ExpandDims�
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:@�*
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
T0*'
_output_shapes
:@�2
conv1d/ExpandDims_1�
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:���������}�*
paddingSAME*
strides
2
conv1d�
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:���������}�*
squeeze_dims

���������2
conv1d/Squeeze�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������}�2	
BiasAdd�
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:@�*
dtype02:
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp�
)stream_0_conv_2/kernel/Regularizer/SquareSquare@stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*#
_output_shapes
:@�2+
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
:���������}�2

Identity�
NoOpNoOp^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp9^stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������}@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp2t
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp:S O
+
_output_shapes
:���������}@
 
_user_specified_nameinputs
�
�
+__inference_basemodel_layer_call_fn_8909796

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:@ 
	unknown_5:@�
	unknown_6:	�
	unknown_7:	�
	unknown_8:	�
	unknown_9:	�

unknown_10:	�"

unknown_11:��

unknown_12:	�

unknown_13:	�

unknown_14:	�

unknown_15:	�

unknown_16:	�

unknown_17:	�T

unknown_18:T

unknown_19:T

unknown_20:T

unknown_21:T

unknown_22:T
identity��StatefulPartitionedCall�
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
:���������T*:
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_basemodel_layer_call_and_return_conditional_losses_89073002
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
_construction_contextkEagerRuntime*Z
_input_shapesI
G:���������}: : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������}
 
_user_specified_nameinputs
�
�
L__inference_stream_0_conv_1_layer_call_and_return_conditional_losses_8910768

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
T0*/
_output_shapes
:���������}2
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
T0*/
_output_shapes
:���������}@*
paddingSAME*
strides
2
conv1d�
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:���������}@*
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
T0*+
_output_shapes
:���������}@2	
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
&stream_0_conv_1/kernel/Regularizer/mulo
IdentityIdentityBiasAdd:output:0^NoOp*
T0*+
_output_shapes
:���������}@2

Identity�
NoOpNoOp^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������}: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp2n
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:S O
+
_output_shapes
:���������}
 
_user_specified_nameinputs
�+
�
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_8911161

inputs6
'assignmovingavg_readvariableop_resource:	�8
)assignmovingavg_1_readvariableop_resource:	�4
%batchnorm_mul_readvariableop_resource:	�0
!batchnorm_readvariableop_resource:	�
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOp�
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indices�
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*#
_output_shapes
:�*
	keep_dims(2
moments/mean�
moments/StopGradientStopGradientmoments/mean:output:0*
T0*#
_output_shapes
:�2
moments/StopGradient�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*,
_output_shapes
:���������}�2
moments/SquaredDifference�
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices�
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*#
_output_shapes
:�*
	keep_dims(2
moments/variance�
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 2
moments/Squeeze�
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:�*
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
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:�*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:�2
AssignMovingAvg/sub�
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:�2
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
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:�*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:�2
AssignMovingAvg_1/sub�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:�2
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
T0*
_output_shapes	
:�2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:�2
batchnorm/Rsqrt�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype02
batchnorm/mul/ReadVariableOp�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2
batchnorm/mul{
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*,
_output_shapes
:���������}�2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:�2
batchnorm/mul_2�
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype02
batchnorm/ReadVariableOp�
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*,
_output_shapes
:���������}�2
batchnorm/add_1s
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*,
_output_shapes
:���������}�2

Identity�
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������}�: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:T P
,
_output_shapes
:���������}�
 
_user_specified_nameinputs
�
q
U__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_8906823

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
�
V
:__inference_global_average_pooling1d_layer_call_fn_8911441

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
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *^
fYRW
U__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_89072282
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:���������}�:T P
,
_output_shapes
:���������}�
 
_user_specified_nameinputs
�
e
I__inference_activation_2_layer_call_and_return_conditional_losses_8907214

inputs
identityS
ReluReluinputs*
T0*,
_output_shapes
:���������}�2
Reluk
IdentityIdentityRelu:activations:0*
T0*,
_output_shapes
:���������}�2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:���������}�:T P
,
_output_shapes
:���������}�
 
_user_specified_nameinputs
�
�
P__inference_batch_normalization_layer_call_and_return_conditional_losses_8910894

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
batchnorm/mulz
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:���������}@2
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
T0*+
_output_shapes
:���������}@2
batchnorm/add_1r
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*+
_output_shapes
:���������}@2

Identity�
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������}@: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:S O
+
_output_shapes
:���������}@
 
_user_specified_nameinputs
�
k
L__inference_dense_1_dropout_layer_call_and_return_conditional_losses_8911480

inputs
identity�c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:����������2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:����������*
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
T0*(
_output_shapes
:����������2
dropout/GreaterEqual�
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:����������2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:����������2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
L__inference_stream_0_conv_3_layer_call_and_return_conditional_losses_8907174

inputsC
+conv1d_expanddims_1_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�"conv1d/ExpandDims_1/ReadVariableOp�5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOpy
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
:���������}�2
conv1d/ExpandDims�
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:��*
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
T0*(
_output_shapes
:��2
conv1d/ExpandDims_1�
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:���������}�*
paddingSAME*
strides
2
conv1d�
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:���������}�*
squeeze_dims

���������2
conv1d/Squeeze�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������}�2	
BiasAdd�
5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:��*
dtype027
5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp�
&stream_0_conv_3/kernel/Regularizer/AbsAbs=stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*$
_output_shapes
:��2(
&stream_0_conv_3/kernel/Regularizer/Abs�
(stream_0_conv_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_0_conv_3/kernel/Regularizer/Const�
&stream_0_conv_3/kernel/Regularizer/SumSum*stream_0_conv_3/kernel/Regularizer/Abs:y:01stream_0_conv_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_3/kernel/Regularizer/Sum�
(stream_0_conv_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2*
(stream_0_conv_3/kernel/Regularizer/mul/x�
&stream_0_conv_3/kernel/Regularizer/mulMul1stream_0_conv_3/kernel/Regularizer/mul/x:output:0/stream_0_conv_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_3/kernel/Regularizer/mulp
IdentityIdentityBiasAdd:output:0^NoOp*
T0*,
_output_shapes
:���������}�2

Identity�
NoOpNoOp^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp6^stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:���������}�: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp2n
5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp:T P
,
_output_shapes
:���������}�
 
_user_specified_nameinputs
�	
�
5__inference_batch_normalization_layer_call_fn_8910794

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
P__inference_batch_normalization_layer_call_and_return_conditional_losses_89064112
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
o
P__inference_stream_0_input_drop_layer_call_and_return_conditional_losses_8910732

inputs
identity�c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
dropout/Constw
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:���������}2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:���������}*
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
T0*+
_output_shapes
:���������}2
dropout/GreaterEqual�
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:���������}2
dropout/Cast~
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:���������}2
dropout/Mul_1i
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:���������}2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������}:S O
+
_output_shapes
:���������}
 
_user_specified_nameinputs
�
j
L__inference_stream_0_drop_2_layer_call_and_return_conditional_losses_8907151

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:���������}�2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:���������}�2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:���������}�:T P
,
_output_shapes
:���������}�
 
_user_specified_nameinputs
�
M
1__inference_stream_0_drop_2_layer_call_fn_8911176

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
:���������}�* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_stream_0_drop_2_layer_call_and_return_conditional_losses_89071512
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:���������}�2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:���������}�:T P
,
_output_shapes
:���������}�
 
_user_specified_nameinputs
��
�
F__inference_basemodel_layer_call_and_return_conditional_losses_8908062
inputs_0-
stream_0_conv_1_8907972:@%
stream_0_conv_1_8907974:@)
batch_normalization_8907977:@)
batch_normalization_8907979:@)
batch_normalization_8907981:@)
batch_normalization_8907983:@.
stream_0_conv_2_8907988:@�&
stream_0_conv_2_8907990:	�,
batch_normalization_1_8907993:	�,
batch_normalization_1_8907995:	�,
batch_normalization_1_8907997:	�,
batch_normalization_1_8907999:	�/
stream_0_conv_3_8908004:��&
stream_0_conv_3_8908006:	�,
batch_normalization_2_8908009:	�,
batch_normalization_2_8908011:	�,
batch_normalization_2_8908013:	�,
batch_normalization_2_8908015:	�"
dense_1_8908022:	�T
dense_1_8908024:T+
batch_normalization_3_8908027:T+
batch_normalization_3_8908029:T+
batch_normalization_3_8908031:T+
batch_normalization_3_8908033:T
identity��+batch_normalization/StatefulPartitionedCall�-batch_normalization_1/StatefulPartitionedCall�-batch_normalization_2/StatefulPartitionedCall�-batch_normalization_3/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�-dense_1/kernel/Regularizer/Abs/ReadVariableOp�'stream_0_conv_1/StatefulPartitionedCall�5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp�'stream_0_conv_2/StatefulPartitionedCall�8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp�'stream_0_conv_3/StatefulPartitionedCall�5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp�
#stream_0_input_drop/PartitionedCallPartitionedCallinputs_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������}* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *Y
fTRR
P__inference_stream_0_input_drop_layer_call_and_return_conditional_losses_89070112%
#stream_0_input_drop/PartitionedCall�
'stream_0_conv_1/StatefulPartitionedCallStatefulPartitionedCall,stream_0_input_drop/PartitionedCall:output:0stream_0_conv_1_8907972stream_0_conv_1_8907974*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������}@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_stream_0_conv_1_layer_call_and_return_conditional_losses_89070342)
'stream_0_conv_1/StatefulPartitionedCall�
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall0stream_0_conv_1/StatefulPartitionedCall:output:0batch_normalization_8907977batch_normalization_8907979batch_normalization_8907981batch_normalization_8907983*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������}@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_89070592-
+batch_normalization/StatefulPartitionedCall�
activation/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������}@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_activation_layer_call_and_return_conditional_losses_89070742
activation/PartitionedCall�
stream_0_drop_1/PartitionedCallPartitionedCall#activation/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������}@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_stream_0_drop_1_layer_call_and_return_conditional_losses_89070812!
stream_0_drop_1/PartitionedCall�
'stream_0_conv_2/StatefulPartitionedCallStatefulPartitionedCall(stream_0_drop_1/PartitionedCall:output:0stream_0_conv_2_8907988stream_0_conv_2_8907990*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������}�*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_stream_0_conv_2_layer_call_and_return_conditional_losses_89071042)
'stream_0_conv_2/StatefulPartitionedCall�
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall0stream_0_conv_2/StatefulPartitionedCall:output:0batch_normalization_1_8907993batch_normalization_1_8907995batch_normalization_1_8907997batch_normalization_1_8907999*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������}�*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *[
fVRT
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_89071292/
-batch_normalization_1/StatefulPartitionedCall�
activation_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������}�* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_activation_1_layer_call_and_return_conditional_losses_89071442
activation_1/PartitionedCall�
stream_0_drop_2/PartitionedCallPartitionedCall%activation_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������}�* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_stream_0_drop_2_layer_call_and_return_conditional_losses_89071512!
stream_0_drop_2/PartitionedCall�
'stream_0_conv_3/StatefulPartitionedCallStatefulPartitionedCall(stream_0_drop_2/PartitionedCall:output:0stream_0_conv_3_8908004stream_0_conv_3_8908006*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������}�*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_stream_0_conv_3_layer_call_and_return_conditional_losses_89071742)
'stream_0_conv_3/StatefulPartitionedCall�
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall0stream_0_conv_3/StatefulPartitionedCall:output:0batch_normalization_2_8908009batch_normalization_2_8908011batch_normalization_2_8908013batch_normalization_2_8908015*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������}�*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *[
fVRT
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_89071992/
-batch_normalization_2/StatefulPartitionedCall�
activation_2/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������}�* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_activation_2_layer_call_and_return_conditional_losses_89072142
activation_2/PartitionedCall�
stream_0_drop_3/PartitionedCallPartitionedCall%activation_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������}�* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_stream_0_drop_3_layer_call_and_return_conditional_losses_89072212!
stream_0_drop_3/PartitionedCall�
(global_average_pooling1d/PartitionedCallPartitionedCall(stream_0_drop_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *^
fYRW
U__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_89072282*
(global_average_pooling1d/PartitionedCall�
dense_1_dropout/PartitionedCallPartitionedCall1global_average_pooling1d/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_dense_1_dropout_layer_call_and_return_conditional_losses_89072352!
dense_1_dropout/PartitionedCall�
dense_1/StatefulPartitionedCallStatefulPartitionedCall(dense_1_dropout/PartitionedCall:output:0dense_1_8908022dense_1_8908024*
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
D__inference_dense_1_layer_call_and_return_conditional_losses_89072532!
dense_1/StatefulPartitionedCall�
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0batch_normalization_3_8908027batch_normalization_3_8908029batch_normalization_3_8908031batch_normalization_3_8908033*
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
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_89068612/
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
O__inference_dense_activation_1_layer_call_and_return_conditional_losses_89072732$
"dense_activation_1/PartitionedCall�
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpstream_0_conv_1_8907972*"
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
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpstream_0_conv_2_8907988*#
_output_shapes
:@�*
dtype02:
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp�
)stream_0_conv_2/kernel/Regularizer/SquareSquare@stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*#
_output_shapes
:@�2+
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
5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpstream_0_conv_3_8908004*$
_output_shapes
:��*
dtype027
5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp�
&stream_0_conv_3/kernel/Regularizer/AbsAbs=stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*$
_output_shapes
:��2(
&stream_0_conv_3/kernel/Regularizer/Abs�
(stream_0_conv_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_0_conv_3/kernel/Regularizer/Const�
&stream_0_conv_3/kernel/Regularizer/SumSum*stream_0_conv_3/kernel/Regularizer/Abs:y:01stream_0_conv_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_3/kernel/Regularizer/Sum�
(stream_0_conv_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2*
(stream_0_conv_3/kernel/Regularizer/mul/x�
&stream_0_conv_3/kernel/Regularizer/mulMul1stream_0_conv_3/kernel/Regularizer/mul/x:output:0/stream_0_conv_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_3/kernel/Regularizer/mul�
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_1_8908022*
_output_shapes
:	�T*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOp�
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	�T2 
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
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall.^dense_1/kernel/Regularizer/Abs/ReadVariableOp(^stream_0_conv_1/StatefulPartitionedCall6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp(^stream_0_conv_2/StatefulPartitionedCall9^stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp(^stream_0_conv_3/StatefulPartitionedCall6^stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:���������}: : : : : : : : : : : : : : : : : : : : : : : : 2Z
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
:���������}
"
_user_specified_name
inputs_0
�
e
I__inference_activation_1_layer_call_and_return_conditional_losses_8907144

inputs
identityS
ReluReluinputs*
T0*,
_output_shapes
:���������}�2
Reluk
IdentityIdentityRelu:activations:0*
T0*,
_output_shapes
:���������}�2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:���������}�:T P
,
_output_shapes
:���������}�
 
_user_specified_nameinputs
�
�
__inference_loss_fn_1_8911623X
Astream_0_conv_2_kernel_regularizer_square_readvariableop_resource:@�
identity��8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp�
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpAstream_0_conv_2_kernel_regularizer_square_readvariableop_resource*#
_output_shapes
:@�*
dtype02:
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp�
)stream_0_conv_2/kernel/Regularizer/SquareSquare@stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*#
_output_shapes
:@�2+
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
�
c
G__inference_activation_layer_call_and_return_conditional_losses_8910938

inputs
identityR
ReluReluinputs*
T0*+
_output_shapes
:���������}@2
Reluj
IdentityIdentityRelu:activations:0*
T0*+
_output_shapes
:���������}@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������}@:S O
+
_output_shapes
:���������}@
 
_user_specified_nameinputs
�
�
)__inference_model_1_layer_call_fn_8909291

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:@ 
	unknown_5:@�
	unknown_6:	�
	unknown_7:	�
	unknown_8:	�
	unknown_9:	�

unknown_10:	�"

unknown_11:��

unknown_12:	�

unknown_13:	�

unknown_14:	�

unknown_15:	�

unknown_16:	�

unknown_17:	�T

unknown_18:T

unknown_19:T

unknown_20:T

unknown_21:T

unknown_22:T
identity��StatefulPartitionedCall�
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
:���������T*:
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_model_1_layer_call_and_return_conditional_losses_89083792
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
_construction_contextkEagerRuntime*Z
_input_shapesI
G:���������}: : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������}
 
_user_specified_nameinputs
�
�
D__inference_dense_1_layer_call_and_return_conditional_losses_8907253

inputs1
matmul_readvariableop_resource:	�T-
biasadd_readvariableop_resource:T
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�-dense_1/kernel/Regularizer/Abs/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�T*
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
:	�T*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOp�
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	�T2 
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
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2^
-dense_1/kernel/Regularizer/Abs/ReadVariableOp-dense_1/kernel/Regularizer/Abs/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
k
L__inference_stream_0_drop_2_layer_call_and_return_conditional_losses_8907514

inputs
identity�c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU�?2
dropout/Constx
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:���������}�2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:���������}�*
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
:���������}�2
dropout/GreaterEqual�
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:���������}�2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:���������}�2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:���������}�2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:���������}�:T P
,
_output_shapes
:���������}�
 
_user_specified_nameinputs
�+
�
P__inference_batch_normalization_layer_call_and_return_conditional_losses_8910874

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
L__inference_stream_0_drop_3_layer_call_and_return_conditional_losses_8911419

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:���������}�2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:���������}�2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:���������}�:T P
,
_output_shapes
:���������}�
 
_user_specified_nameinputs
�
n
5__inference_stream_0_input_drop_layer_call_fn_8910715

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������}* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *Y
fTRR
P__inference_stream_0_input_drop_layer_call_and_return_conditional_losses_89077122
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������}2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������}22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������}
 
_user_specified_nameinputs
�
M
1__inference_stream_0_drop_3_layer_call_fn_8911409

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
:���������}�* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_stream_0_drop_3_layer_call_and_return_conditional_losses_89072212
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:���������}�2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:���������}�:T P
,
_output_shapes
:���������}�
 
_user_specified_nameinputs
�
e
I__inference_activation_2_layer_call_and_return_conditional_losses_8911404

inputs
identityS
ReluReluinputs*
T0*,
_output_shapes
:���������}�2
Reluk
IdentityIdentityRelu:activations:0*
T0*,
_output_shapes
:���������}�2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:���������}�:T P
,
_output_shapes
:���������}�
 
_user_specified_nameinputs
�+
�
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_8906735

inputs6
'assignmovingavg_readvariableop_resource:	�8
)assignmovingavg_1_readvariableop_resource:	�4
%batchnorm_mul_readvariableop_resource:	�0
!batchnorm_readvariableop_resource:	�
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOp�
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indices�
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*#
_output_shapes
:�*
	keep_dims(2
moments/mean�
moments/StopGradientStopGradientmoments/mean:output:0*
T0*#
_output_shapes
:�2
moments/StopGradient�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*5
_output_shapes#
!:�������������������2
moments/SquaredDifference�
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices�
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*#
_output_shapes
:�*
	keep_dims(2
moments/variance�
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 2
moments/Squeeze�
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:�*
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
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:�*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:�2
AssignMovingAvg/sub�
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:�2
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
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:�*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:�2
AssignMovingAvg_1/sub�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:�2
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
T0*
_output_shapes	
:�2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:�2
batchnorm/Rsqrt�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype02
batchnorm/mul/ReadVariableOp�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2
batchnorm/mul�
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*5
_output_shapes#
!:�������������������2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:�2
batchnorm/mul_2�
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype02
batchnorm/ReadVariableOp�
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*5
_output_shapes#
!:�������������������2
batchnorm/add_1|
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*5
_output_shapes#
!:�������������������2

Identity�
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):�������������������: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:] Y
5
_output_shapes#
!:�������������������
 
_user_specified_nameinputs
�*
�
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_8911591

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
�
�
%__inference_signature_wrapper_8909238
left_inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:@ 
	unknown_5:@�
	unknown_6:	�
	unknown_7:	�
	unknown_8:	�
	unknown_9:	�

unknown_10:	�"

unknown_11:��

unknown_12:	�

unknown_13:	�

unknown_14:	�

unknown_15:	�

unknown_16:	�

unknown_17:	�T

unknown_18:T

unknown_19:T

unknown_20:T

unknown_21:T

unknown_22:T
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallleft_inputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
:���������T*:
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *+
f&R$
"__inference__wrapped_model_89063272
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
_construction_contextkEagerRuntime*Z
_input_shapesI
G:���������}: : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
+
_output_shapes
:���������}
%
_user_specified_nameleft_inputs
�
P
4__inference_dense_activation_1_layer_call_fn_8911596

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
O__inference_dense_activation_1_layer_call_and_return_conditional_losses_89072732
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
�
�
)__inference_model_1_layer_call_fn_8908430
left_inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:@ 
	unknown_5:@�
	unknown_6:	�
	unknown_7:	�
	unknown_8:	�
	unknown_9:	�

unknown_10:	�"

unknown_11:��

unknown_12:	�

unknown_13:	�

unknown_14:	�

unknown_15:	�

unknown_16:	�

unknown_17:	�T

unknown_18:T

unknown_19:T

unknown_20:T

unknown_21:T

unknown_22:T
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallleft_inputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
:���������T*:
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_model_1_layer_call_and_return_conditional_losses_89083792
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
_construction_contextkEagerRuntime*Z
_input_shapesI
G:���������}: : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
+
_output_shapes
:���������}
%
_user_specified_nameleft_inputs
�
Q
5__inference_stream_0_input_drop_layer_call_fn_8910710

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������}* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *Y
fTRR
P__inference_stream_0_input_drop_layer_call_and_return_conditional_losses_89070112
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:���������}2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������}:S O
+
_output_shapes
:���������}
 
_user_specified_nameinputs
�
J
.__inference_activation_2_layer_call_fn_8911399

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
:���������}�* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_activation_2_layer_call_and_return_conditional_losses_89072142
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:���������}�2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:���������}�:T P
,
_output_shapes
:���������}�
 
_user_specified_nameinputs
��
�
F__inference_basemodel_layer_call_and_return_conditional_losses_8908304

inputsQ
;stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource:@=
/stream_0_conv_1_biasadd_readvariableop_resource:@C
5batch_normalization_batchnorm_readvariableop_resource:@G
9batch_normalization_batchnorm_mul_readvariableop_resource:@E
7batch_normalization_batchnorm_readvariableop_1_resource:@E
7batch_normalization_batchnorm_readvariableop_2_resource:@R
;stream_0_conv_2_conv1d_expanddims_1_readvariableop_resource:@�>
/stream_0_conv_2_biasadd_readvariableop_resource:	�F
7batch_normalization_1_batchnorm_readvariableop_resource:	�J
;batch_normalization_1_batchnorm_mul_readvariableop_resource:	�H
9batch_normalization_1_batchnorm_readvariableop_1_resource:	�H
9batch_normalization_1_batchnorm_readvariableop_2_resource:	�S
;stream_0_conv_3_conv1d_expanddims_1_readvariableop_resource:��>
/stream_0_conv_3_biasadd_readvariableop_resource:	�F
7batch_normalization_2_batchnorm_readvariableop_resource:	�J
;batch_normalization_2_batchnorm_mul_readvariableop_resource:	�H
9batch_normalization_2_batchnorm_readvariableop_1_resource:	�H
9batch_normalization_2_batchnorm_readvariableop_2_resource:	�9
&dense_1_matmul_readvariableop_resource:	�T5
'dense_1_biasadd_readvariableop_resource:TE
7batch_normalization_3_batchnorm_readvariableop_resource:TI
;batch_normalization_3_batchnorm_mul_readvariableop_resource:TG
9batch_normalization_3_batchnorm_readvariableop_1_resource:TG
9batch_normalization_3_batchnorm_readvariableop_2_resource:T
identity��,batch_normalization/batchnorm/ReadVariableOp�.batch_normalization/batchnorm/ReadVariableOp_1�.batch_normalization/batchnorm/ReadVariableOp_2�0batch_normalization/batchnorm/mul/ReadVariableOp�.batch_normalization_1/batchnorm/ReadVariableOp�0batch_normalization_1/batchnorm/ReadVariableOp_1�0batch_normalization_1/batchnorm/ReadVariableOp_2�2batch_normalization_1/batchnorm/mul/ReadVariableOp�.batch_normalization_2/batchnorm/ReadVariableOp�0batch_normalization_2/batchnorm/ReadVariableOp_1�0batch_normalization_2/batchnorm/ReadVariableOp_2�2batch_normalization_2/batchnorm/mul/ReadVariableOp�.batch_normalization_3/batchnorm/ReadVariableOp�0batch_normalization_3/batchnorm/ReadVariableOp_1�0batch_normalization_3/batchnorm/ReadVariableOp_2�2batch_normalization_3/batchnorm/mul/ReadVariableOp�dense_1/BiasAdd/ReadVariableOp�dense_1/MatMul/ReadVariableOp�-dense_1/kernel/Regularizer/Abs/ReadVariableOp�&stream_0_conv_1/BiasAdd/ReadVariableOp�2stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp�5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp�&stream_0_conv_2/BiasAdd/ReadVariableOp�2stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOp�8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp�&stream_0_conv_3/BiasAdd/ReadVariableOp�2stream_0_conv_3/conv1d/ExpandDims_1/ReadVariableOp�5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp�
stream_0_input_drop/IdentityIdentityinputs*
T0*+
_output_shapes
:���������}2
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
T0*/
_output_shapes
:���������}2#
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
T0*/
_output_shapes
:���������}@*
paddingSAME*
strides
2
stream_0_conv_1/conv1d�
stream_0_conv_1/conv1d/SqueezeSqueezestream_0_conv_1/conv1d:output:0*
T0*+
_output_shapes
:���������}@*
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
T0*+
_output_shapes
:���������}@2
stream_0_conv_1/BiasAdd�
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
T0*+
_output_shapes
:���������}@2%
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
T0*+
_output_shapes
:���������}@2%
#batch_normalization/batchnorm/add_1�
activation/ReluRelu'batch_normalization/batchnorm/add_1:z:0*
T0*+
_output_shapes
:���������}@2
activation/Relu�
stream_0_drop_1/IdentityIdentityactivation/Relu:activations:0*
T0*+
_output_shapes
:���������}@2
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
T0*/
_output_shapes
:���������}@2#
!stream_0_conv_2/conv1d/ExpandDims�
2stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp;stream_0_conv_2_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:@�*
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
T0*'
_output_shapes
:@�2%
#stream_0_conv_2/conv1d/ExpandDims_1�
stream_0_conv_2/conv1dConv2D*stream_0_conv_2/conv1d/ExpandDims:output:0,stream_0_conv_2/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:���������}�*
paddingSAME*
strides
2
stream_0_conv_2/conv1d�
stream_0_conv_2/conv1d/SqueezeSqueezestream_0_conv_2/conv1d:output:0*
T0*,
_output_shapes
:���������}�*
squeeze_dims

���������2 
stream_0_conv_2/conv1d/Squeeze�
&stream_0_conv_2/BiasAdd/ReadVariableOpReadVariableOp/stream_0_conv_2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02(
&stream_0_conv_2/BiasAdd/ReadVariableOp�
stream_0_conv_2/BiasAddBiasAdd'stream_0_conv_2/conv1d/Squeeze:output:0.stream_0_conv_2/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������}�2
stream_0_conv_2/BiasAdd�
.batch_normalization_1/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_1_batchnorm_readvariableop_resource*
_output_shapes	
:�*
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
T0*
_output_shapes	
:�2%
#batch_normalization_1/batchnorm/add�
%batch_normalization_1/batchnorm/RsqrtRsqrt'batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes	
:�2'
%batch_normalization_1/batchnorm/Rsqrt�
2batch_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype024
2batch_normalization_1/batchnorm/mul/ReadVariableOp�
#batch_normalization_1/batchnorm/mulMul)batch_normalization_1/batchnorm/Rsqrt:y:0:batch_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2%
#batch_normalization_1/batchnorm/mul�
%batch_normalization_1/batchnorm/mul_1Mul stream_0_conv_2/BiasAdd:output:0'batch_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:���������}�2'
%batch_normalization_1/batchnorm/mul_1�
0batch_normalization_1/batchnorm/ReadVariableOp_1ReadVariableOp9batch_normalization_1_batchnorm_readvariableop_1_resource*
_output_shapes	
:�*
dtype022
0batch_normalization_1/batchnorm/ReadVariableOp_1�
%batch_normalization_1/batchnorm/mul_2Mul8batch_normalization_1/batchnorm/ReadVariableOp_1:value:0'batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes	
:�2'
%batch_normalization_1/batchnorm/mul_2�
0batch_normalization_1/batchnorm/ReadVariableOp_2ReadVariableOp9batch_normalization_1_batchnorm_readvariableop_2_resource*
_output_shapes	
:�*
dtype022
0batch_normalization_1/batchnorm/ReadVariableOp_2�
#batch_normalization_1/batchnorm/subSub8batch_normalization_1/batchnorm/ReadVariableOp_2:value:0)batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�2%
#batch_normalization_1/batchnorm/sub�
%batch_normalization_1/batchnorm/add_1AddV2)batch_normalization_1/batchnorm/mul_1:z:0'batch_normalization_1/batchnorm/sub:z:0*
T0*,
_output_shapes
:���������}�2'
%batch_normalization_1/batchnorm/add_1�
activation_1/ReluRelu)batch_normalization_1/batchnorm/add_1:z:0*
T0*,
_output_shapes
:���������}�2
activation_1/Relu�
stream_0_drop_2/IdentityIdentityactivation_1/Relu:activations:0*
T0*,
_output_shapes
:���������}�2
stream_0_drop_2/Identity�
%stream_0_conv_3/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2'
%stream_0_conv_3/conv1d/ExpandDims/dim�
!stream_0_conv_3/conv1d/ExpandDims
ExpandDims!stream_0_drop_2/Identity:output:0.stream_0_conv_3/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:���������}�2#
!stream_0_conv_3/conv1d/ExpandDims�
2stream_0_conv_3/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp;stream_0_conv_3_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:��*
dtype024
2stream_0_conv_3/conv1d/ExpandDims_1/ReadVariableOp�
'stream_0_conv_3/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2)
'stream_0_conv_3/conv1d/ExpandDims_1/dim�
#stream_0_conv_3/conv1d/ExpandDims_1
ExpandDims:stream_0_conv_3/conv1d/ExpandDims_1/ReadVariableOp:value:00stream_0_conv_3/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:��2%
#stream_0_conv_3/conv1d/ExpandDims_1�
stream_0_conv_3/conv1dConv2D*stream_0_conv_3/conv1d/ExpandDims:output:0,stream_0_conv_3/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:���������}�*
paddingSAME*
strides
2
stream_0_conv_3/conv1d�
stream_0_conv_3/conv1d/SqueezeSqueezestream_0_conv_3/conv1d:output:0*
T0*,
_output_shapes
:���������}�*
squeeze_dims

���������2 
stream_0_conv_3/conv1d/Squeeze�
&stream_0_conv_3/BiasAdd/ReadVariableOpReadVariableOp/stream_0_conv_3_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02(
&stream_0_conv_3/BiasAdd/ReadVariableOp�
stream_0_conv_3/BiasAddBiasAdd'stream_0_conv_3/conv1d/Squeeze:output:0.stream_0_conv_3/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������}�2
stream_0_conv_3/BiasAdd�
.batch_normalization_2/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_2_batchnorm_readvariableop_resource*
_output_shapes	
:�*
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
T0*
_output_shapes	
:�2%
#batch_normalization_2/batchnorm/add�
%batch_normalization_2/batchnorm/RsqrtRsqrt'batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes	
:�2'
%batch_normalization_2/batchnorm/Rsqrt�
2batch_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype024
2batch_normalization_2/batchnorm/mul/ReadVariableOp�
#batch_normalization_2/batchnorm/mulMul)batch_normalization_2/batchnorm/Rsqrt:y:0:batch_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2%
#batch_normalization_2/batchnorm/mul�
%batch_normalization_2/batchnorm/mul_1Mul stream_0_conv_3/BiasAdd:output:0'batch_normalization_2/batchnorm/mul:z:0*
T0*,
_output_shapes
:���������}�2'
%batch_normalization_2/batchnorm/mul_1�
0batch_normalization_2/batchnorm/ReadVariableOp_1ReadVariableOp9batch_normalization_2_batchnorm_readvariableop_1_resource*
_output_shapes	
:�*
dtype022
0batch_normalization_2/batchnorm/ReadVariableOp_1�
%batch_normalization_2/batchnorm/mul_2Mul8batch_normalization_2/batchnorm/ReadVariableOp_1:value:0'batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes	
:�2'
%batch_normalization_2/batchnorm/mul_2�
0batch_normalization_2/batchnorm/ReadVariableOp_2ReadVariableOp9batch_normalization_2_batchnorm_readvariableop_2_resource*
_output_shapes	
:�*
dtype022
0batch_normalization_2/batchnorm/ReadVariableOp_2�
#batch_normalization_2/batchnorm/subSub8batch_normalization_2/batchnorm/ReadVariableOp_2:value:0)batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�2%
#batch_normalization_2/batchnorm/sub�
%batch_normalization_2/batchnorm/add_1AddV2)batch_normalization_2/batchnorm/mul_1:z:0'batch_normalization_2/batchnorm/sub:z:0*
T0*,
_output_shapes
:���������}�2'
%batch_normalization_2/batchnorm/add_1�
activation_2/ReluRelu)batch_normalization_2/batchnorm/add_1:z:0*
T0*,
_output_shapes
:���������}�2
activation_2/Relu�
stream_0_drop_3/IdentityIdentityactivation_2/Relu:activations:0*
T0*,
_output_shapes
:���������}�2
stream_0_drop_3/Identity�
/global_average_pooling1d/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :21
/global_average_pooling1d/Mean/reduction_indices�
global_average_pooling1d/MeanMean!stream_0_drop_3/Identity:output:08global_average_pooling1d/Mean/reduction_indices:output:0*
T0*(
_output_shapes
:����������2
global_average_pooling1d/Mean�
dense_1_dropout/IdentityIdentity&global_average_pooling1d/Mean:output:0*
T0*(
_output_shapes
:����������2
dense_1_dropout/Identity�
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	�T*
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
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;stream_0_conv_2_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:@�*
dtype02:
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp�
)stream_0_conv_2/kernel/Regularizer/SquareSquare@stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*#
_output_shapes
:@�2+
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
5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp;stream_0_conv_3_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:��*
dtype027
5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp�
&stream_0_conv_3/kernel/Regularizer/AbsAbs=stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*$
_output_shapes
:��2(
&stream_0_conv_3/kernel/Regularizer/Abs�
(stream_0_conv_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_0_conv_3/kernel/Regularizer/Const�
&stream_0_conv_3/kernel/Regularizer/SumSum*stream_0_conv_3/kernel/Regularizer/Abs:y:01stream_0_conv_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_3/kernel/Regularizer/Sum�
(stream_0_conv_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2*
(stream_0_conv_3/kernel/Regularizer/mul/x�
&stream_0_conv_3/kernel/Regularizer/mulMul1stream_0_conv_3/kernel/Regularizer/mul/x:output:0/stream_0_conv_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_3/kernel/Regularizer/mul�
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	�T*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOp�
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	�T2 
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
NoOpNoOp-^batch_normalization/batchnorm/ReadVariableOp/^batch_normalization/batchnorm/ReadVariableOp_1/^batch_normalization/batchnorm/ReadVariableOp_21^batch_normalization/batchnorm/mul/ReadVariableOp/^batch_normalization_1/batchnorm/ReadVariableOp1^batch_normalization_1/batchnorm/ReadVariableOp_11^batch_normalization_1/batchnorm/ReadVariableOp_23^batch_normalization_1/batchnorm/mul/ReadVariableOp/^batch_normalization_2/batchnorm/ReadVariableOp1^batch_normalization_2/batchnorm/ReadVariableOp_11^batch_normalization_2/batchnorm/ReadVariableOp_23^batch_normalization_2/batchnorm/mul/ReadVariableOp/^batch_normalization_3/batchnorm/ReadVariableOp1^batch_normalization_3/batchnorm/ReadVariableOp_11^batch_normalization_3/batchnorm/ReadVariableOp_23^batch_normalization_3/batchnorm/mul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp.^dense_1/kernel/Regularizer/Abs/ReadVariableOp'^stream_0_conv_1/BiasAdd/ReadVariableOp3^stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp'^stream_0_conv_2/BiasAdd/ReadVariableOp3^stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOp9^stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp'^stream_0_conv_3/BiasAdd/ReadVariableOp3^stream_0_conv_3/conv1d/ExpandDims_1/ReadVariableOp6^stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:���������}: : : : : : : : : : : : : : : : : : : : : : : : 2\
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
:���������}
 
_user_specified_nameinputs
�
n
P__inference_stream_0_input_drop_layer_call_and_return_conditional_losses_8907011

inputs

identity_1^
IdentityIdentityinputs*
T0*+
_output_shapes
:���������}2

Identitym

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:���������}2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������}:S O
+
_output_shapes
:���������}
 
_user_specified_nameinputs
��
�
F__inference_basemodel_layer_call_and_return_conditional_losses_8908156
inputs_0-
stream_0_conv_1_8908066:@%
stream_0_conv_1_8908068:@)
batch_normalization_8908071:@)
batch_normalization_8908073:@)
batch_normalization_8908075:@)
batch_normalization_8908077:@.
stream_0_conv_2_8908082:@�&
stream_0_conv_2_8908084:	�,
batch_normalization_1_8908087:	�,
batch_normalization_1_8908089:	�,
batch_normalization_1_8908091:	�,
batch_normalization_1_8908093:	�/
stream_0_conv_3_8908098:��&
stream_0_conv_3_8908100:	�,
batch_normalization_2_8908103:	�,
batch_normalization_2_8908105:	�,
batch_normalization_2_8908107:	�,
batch_normalization_2_8908109:	�"
dense_1_8908116:	�T
dense_1_8908118:T+
batch_normalization_3_8908121:T+
batch_normalization_3_8908123:T+
batch_normalization_3_8908125:T+
batch_normalization_3_8908127:T
identity��+batch_normalization/StatefulPartitionedCall�-batch_normalization_1/StatefulPartitionedCall�-batch_normalization_2/StatefulPartitionedCall�-batch_normalization_3/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�-dense_1/kernel/Regularizer/Abs/ReadVariableOp�'dense_1_dropout/StatefulPartitionedCall�'stream_0_conv_1/StatefulPartitionedCall�5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp�'stream_0_conv_2/StatefulPartitionedCall�8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp�'stream_0_conv_3/StatefulPartitionedCall�5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp�'stream_0_drop_1/StatefulPartitionedCall�'stream_0_drop_2/StatefulPartitionedCall�'stream_0_drop_3/StatefulPartitionedCall�+stream_0_input_drop/StatefulPartitionedCall�
+stream_0_input_drop/StatefulPartitionedCallStatefulPartitionedCallinputs_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������}* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *Y
fTRR
P__inference_stream_0_input_drop_layer_call_and_return_conditional_losses_89077122-
+stream_0_input_drop/StatefulPartitionedCall�
'stream_0_conv_1/StatefulPartitionedCallStatefulPartitionedCall4stream_0_input_drop/StatefulPartitionedCall:output:0stream_0_conv_1_8908066stream_0_conv_1_8908068*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������}@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_stream_0_conv_1_layer_call_and_return_conditional_losses_89070342)
'stream_0_conv_1/StatefulPartitionedCall�
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall0stream_0_conv_1/StatefulPartitionedCall:output:0batch_normalization_8908071batch_normalization_8908073batch_normalization_8908075batch_normalization_8908077*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������}@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_89076712-
+batch_normalization/StatefulPartitionedCall�
activation/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������}@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_activation_layer_call_and_return_conditional_losses_89070742
activation/PartitionedCall�
'stream_0_drop_1/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0,^stream_0_input_drop/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������}@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_stream_0_drop_1_layer_call_and_return_conditional_losses_89076132)
'stream_0_drop_1/StatefulPartitionedCall�
'stream_0_conv_2/StatefulPartitionedCallStatefulPartitionedCall0stream_0_drop_1/StatefulPartitionedCall:output:0stream_0_conv_2_8908082stream_0_conv_2_8908084*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������}�*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_stream_0_conv_2_layer_call_and_return_conditional_losses_89071042)
'stream_0_conv_2/StatefulPartitionedCall�
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall0stream_0_conv_2/StatefulPartitionedCall:output:0batch_normalization_1_8908087batch_normalization_1_8908089batch_normalization_1_8908091batch_normalization_1_8908093*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������}�*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *[
fVRT
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_89075722/
-batch_normalization_1/StatefulPartitionedCall�
activation_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������}�* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_activation_1_layer_call_and_return_conditional_losses_89071442
activation_1/PartitionedCall�
'stream_0_drop_2/StatefulPartitionedCallStatefulPartitionedCall%activation_1/PartitionedCall:output:0(^stream_0_drop_1/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������}�* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_stream_0_drop_2_layer_call_and_return_conditional_losses_89075142)
'stream_0_drop_2/StatefulPartitionedCall�
'stream_0_conv_3/StatefulPartitionedCallStatefulPartitionedCall0stream_0_drop_2/StatefulPartitionedCall:output:0stream_0_conv_3_8908098stream_0_conv_3_8908100*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������}�*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_stream_0_conv_3_layer_call_and_return_conditional_losses_89071742)
'stream_0_conv_3/StatefulPartitionedCall�
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall0stream_0_conv_3/StatefulPartitionedCall:output:0batch_normalization_2_8908103batch_normalization_2_8908105batch_normalization_2_8908107batch_normalization_2_8908109*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������}�*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *[
fVRT
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_89074732/
-batch_normalization_2/StatefulPartitionedCall�
activation_2/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������}�* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_activation_2_layer_call_and_return_conditional_losses_89072142
activation_2/PartitionedCall�
'stream_0_drop_3/StatefulPartitionedCallStatefulPartitionedCall%activation_2/PartitionedCall:output:0(^stream_0_drop_2/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������}�* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_stream_0_drop_3_layer_call_and_return_conditional_losses_89074152)
'stream_0_drop_3/StatefulPartitionedCall�
(global_average_pooling1d/PartitionedCallPartitionedCall0stream_0_drop_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *^
fYRW
U__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_89072282*
(global_average_pooling1d/PartitionedCall�
'dense_1_dropout/StatefulPartitionedCallStatefulPartitionedCall1global_average_pooling1d/PartitionedCall:output:0(^stream_0_drop_3/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_dense_1_dropout_layer_call_and_return_conditional_losses_89073872)
'dense_1_dropout/StatefulPartitionedCall�
dense_1/StatefulPartitionedCallStatefulPartitionedCall0dense_1_dropout/StatefulPartitionedCall:output:0dense_1_8908116dense_1_8908118*
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
D__inference_dense_1_layer_call_and_return_conditional_losses_89072532!
dense_1/StatefulPartitionedCall�
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0batch_normalization_3_8908121batch_normalization_3_8908123batch_normalization_3_8908125batch_normalization_3_8908127*
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
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_89069212/
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
O__inference_dense_activation_1_layer_call_and_return_conditional_losses_89072732$
"dense_activation_1/PartitionedCall�
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpstream_0_conv_1_8908066*"
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
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpstream_0_conv_2_8908082*#
_output_shapes
:@�*
dtype02:
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp�
)stream_0_conv_2/kernel/Regularizer/SquareSquare@stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*#
_output_shapes
:@�2+
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
5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpstream_0_conv_3_8908098*$
_output_shapes
:��*
dtype027
5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp�
&stream_0_conv_3/kernel/Regularizer/AbsAbs=stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*$
_output_shapes
:��2(
&stream_0_conv_3/kernel/Regularizer/Abs�
(stream_0_conv_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_0_conv_3/kernel/Regularizer/Const�
&stream_0_conv_3/kernel/Regularizer/SumSum*stream_0_conv_3/kernel/Regularizer/Abs:y:01stream_0_conv_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_3/kernel/Regularizer/Sum�
(stream_0_conv_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2*
(stream_0_conv_3/kernel/Regularizer/mul/x�
&stream_0_conv_3/kernel/Regularizer/mulMul1stream_0_conv_3/kernel/Regularizer/mul/x:output:0/stream_0_conv_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_3/kernel/Regularizer/mul�
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_1_8908116*
_output_shapes
:	�T*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOp�
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	�T2 
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

Identity�
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall.^dense_1/kernel/Regularizer/Abs/ReadVariableOp(^dense_1_dropout/StatefulPartitionedCall(^stream_0_conv_1/StatefulPartitionedCall6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp(^stream_0_conv_2/StatefulPartitionedCall9^stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp(^stream_0_conv_3/StatefulPartitionedCall6^stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp(^stream_0_drop_1/StatefulPartitionedCall(^stream_0_drop_2/StatefulPartitionedCall(^stream_0_drop_3/StatefulPartitionedCall,^stream_0_input_drop/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:���������}: : : : : : : : : : : : : : : : : : : : : : : : 2Z
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
:���������}
"
_user_specified_name
inputs_0
�
�
7__inference_batch_normalization_1_layer_call_fn_8911053

inputs
unknown:	�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������}�*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *[
fVRT
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_89075722
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:���������}�2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������}�: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:���������}�
 
_user_specified_nameinputs
�
c
G__inference_activation_layer_call_and_return_conditional_losses_8907074

inputs
identityR
ReluReluinputs*
T0*+
_output_shapes
:���������}@2
Reluj
IdentityIdentityRelu:activations:0*
T0*+
_output_shapes
:���������}@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������}@:S O
+
_output_shapes
:���������}@
 
_user_specified_nameinputs
�
q
U__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_8911447

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
�
�
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_8911360

inputs0
!batchnorm_readvariableop_resource:	�4
%batchnorm_mul_readvariableop_resource:	�2
#batchnorm_readvariableop_1_resource:	�2
#batchnorm_readvariableop_2_resource:	�
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOp�
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:�*
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
T0*
_output_shapes	
:�2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:�2
batchnorm/Rsqrt�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype02
batchnorm/mul/ReadVariableOp�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2
batchnorm/mul{
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*,
_output_shapes
:���������}�2
batchnorm/mul_1�
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:�*
dtype02
batchnorm/ReadVariableOp_1�
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:�2
batchnorm/mul_2�
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:�*
dtype02
batchnorm/ReadVariableOp_2�
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*,
_output_shapes
:���������}�2
batchnorm/add_1s
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*,
_output_shapes
:���������}�2

Identity�
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������}�: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:T P
,
_output_shapes
:���������}�
 
_user_specified_nameinputs
�
k
L__inference_stream_0_drop_2_layer_call_and_return_conditional_losses_8911198

inputs
identity�c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU�?2
dropout/Constx
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:���������}�2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:���������}�*
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
:���������}�2
dropout/GreaterEqual�
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:���������}�2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:���������}�2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:���������}�2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:���������}�:T P
,
_output_shapes
:���������}�
 
_user_specified_nameinputs
�
�
+__inference_basemodel_layer_call_fn_8907968
inputs_0
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:@ 
	unknown_5:@�
	unknown_6:	�
	unknown_7:	�
	unknown_8:	�
	unknown_9:	�

unknown_10:	�"

unknown_11:��

unknown_12:	�

unknown_13:	�

unknown_14:	�

unknown_15:	�

unknown_16:	�

unknown_17:	�T

unknown_18:T

unknown_19:T

unknown_20:T

unknown_21:T

unknown_22:T
identity��StatefulPartitionedCall�
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
:���������T*2
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_basemodel_layer_call_and_return_conditional_losses_89078642
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
_construction_contextkEagerRuntime*Z
_input_shapesI
G:���������}: : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
+
_output_shapes
:���������}
"
_user_specified_name
inputs_0
�+
�
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_8911107

inputs6
'assignmovingavg_readvariableop_resource:	�8
)assignmovingavg_1_readvariableop_resource:	�4
%batchnorm_mul_readvariableop_resource:	�0
!batchnorm_readvariableop_resource:	�
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOp�
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indices�
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*#
_output_shapes
:�*
	keep_dims(2
moments/mean�
moments/StopGradientStopGradientmoments/mean:output:0*
T0*#
_output_shapes
:�2
moments/StopGradient�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*5
_output_shapes#
!:�������������������2
moments/SquaredDifference�
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices�
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*#
_output_shapes
:�*
	keep_dims(2
moments/variance�
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 2
moments/Squeeze�
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:�*
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
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:�*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:�2
AssignMovingAvg/sub�
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:�2
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
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:�*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:�2
AssignMovingAvg_1/sub�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:�2
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
T0*
_output_shapes	
:�2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:�2
batchnorm/Rsqrt�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype02
batchnorm/mul/ReadVariableOp�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2
batchnorm/mul�
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*5
_output_shapes#
!:�������������������2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:�2
batchnorm/mul_2�
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype02
batchnorm/ReadVariableOp�
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*5
_output_shapes#
!:�������������������2
batchnorm/add_1|
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*5
_output_shapes#
!:�������������������2

Identity�
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):�������������������: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:] Y
5
_output_shapes#
!:�������������������
 
_user_specified_nameinputs
�	
�
7__inference_batch_normalization_1_layer_call_fn_8911014

inputs
unknown:	�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:�������������������*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *[
fVRT
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_89065132
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:�������������������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):�������������������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:�������������������
 
_user_specified_nameinputs
�
n
P__inference_stream_0_input_drop_layer_call_and_return_conditional_losses_8910720

inputs

identity_1^
IdentityIdentityinputs*
T0*+
_output_shapes
:���������}2

Identitym

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:���������}2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������}:S O
+
_output_shapes
:���������}
 
_user_specified_nameinputs
�=
�	
D__inference_model_1_layer_call_and_return_conditional_losses_8909082
left_inputs'
basemodel_8909008:@
basemodel_8909010:@
basemodel_8909012:@
basemodel_8909014:@
basemodel_8909016:@
basemodel_8909018:@(
basemodel_8909020:@� 
basemodel_8909022:	� 
basemodel_8909024:	� 
basemodel_8909026:	� 
basemodel_8909028:	� 
basemodel_8909030:	�)
basemodel_8909032:�� 
basemodel_8909034:	� 
basemodel_8909036:	� 
basemodel_8909038:	� 
basemodel_8909040:	� 
basemodel_8909042:	�$
basemodel_8909044:	�T
basemodel_8909046:T
basemodel_8909048:T
basemodel_8909050:T
basemodel_8909052:T
basemodel_8909054:T
identity��!basemodel/StatefulPartitionedCall�-dense_1/kernel/Regularizer/Abs/ReadVariableOp�5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp�8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp�5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp�
!basemodel/StatefulPartitionedCallStatefulPartitionedCallleft_inputsbasemodel_8909008basemodel_8909010basemodel_8909012basemodel_8909014basemodel_8909016basemodel_8909018basemodel_8909020basemodel_8909022basemodel_8909024basemodel_8909026basemodel_8909028basemodel_8909030basemodel_8909032basemodel_8909034basemodel_8909036basemodel_8909038basemodel_8909040basemodel_8909042basemodel_8909044basemodel_8909046basemodel_8909048basemodel_8909050basemodel_8909052basemodel_8909054*$
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������T*:
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_basemodel_layer_call_and_return_conditional_losses_89083042#
!basemodel/StatefulPartitionedCall�
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpbasemodel_8909008*"
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
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpbasemodel_8909020*#
_output_shapes
:@�*
dtype02:
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp�
)stream_0_conv_2/kernel/Regularizer/SquareSquare@stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*#
_output_shapes
:@�2+
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
5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpbasemodel_8909032*$
_output_shapes
:��*
dtype027
5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp�
&stream_0_conv_3/kernel/Regularizer/AbsAbs=stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*$
_output_shapes
:��2(
&stream_0_conv_3/kernel/Regularizer/Abs�
(stream_0_conv_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_0_conv_3/kernel/Regularizer/Const�
&stream_0_conv_3/kernel/Regularizer/SumSum*stream_0_conv_3/kernel/Regularizer/Abs:y:01stream_0_conv_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_3/kernel/Regularizer/Sum�
(stream_0_conv_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2*
(stream_0_conv_3/kernel/Regularizer/mul/x�
&stream_0_conv_3/kernel/Regularizer/mulMul1stream_0_conv_3/kernel/Regularizer/mul/x:output:0/stream_0_conv_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_3/kernel/Regularizer/mul�
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpbasemodel_8909044*
_output_shapes
:	�T*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOp�
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	�T2 
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
IdentityIdentity*basemodel/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������T2

Identity�
NoOpNoOp"^basemodel/StatefulPartitionedCall.^dense_1/kernel/Regularizer/Abs/ReadVariableOp6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp9^stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp6^stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:���������}: : : : : : : : : : : : : : : : : : : : : : : : 2F
!basemodel/StatefulPartitionedCall!basemodel/StatefulPartitionedCall2^
-dense_1/kernel/Regularizer/Abs/ReadVariableOp-dense_1/kernel/Regularizer/Abs/ReadVariableOp2n
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp2t
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp2n
5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp:X T
+
_output_shapes
:���������}
%
_user_specified_nameleft_inputs
�
k
L__inference_stream_0_drop_1_layer_call_and_return_conditional_losses_8910965

inputs
identity�c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2
dropout/Constw
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:���������}@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:���������}@*
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
T0*+
_output_shapes
:���������}@2
dropout/GreaterEqual�
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:���������}@2
dropout/Cast~
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:���������}@2
dropout/Mul_1i
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:���������}@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������}@:S O
+
_output_shapes
:���������}@
 
_user_specified_nameinputs
��
�
F__inference_basemodel_layer_call_and_return_conditional_losses_8907864

inputs-
stream_0_conv_1_8907774:@%
stream_0_conv_1_8907776:@)
batch_normalization_8907779:@)
batch_normalization_8907781:@)
batch_normalization_8907783:@)
batch_normalization_8907785:@.
stream_0_conv_2_8907790:@�&
stream_0_conv_2_8907792:	�,
batch_normalization_1_8907795:	�,
batch_normalization_1_8907797:	�,
batch_normalization_1_8907799:	�,
batch_normalization_1_8907801:	�/
stream_0_conv_3_8907806:��&
stream_0_conv_3_8907808:	�,
batch_normalization_2_8907811:	�,
batch_normalization_2_8907813:	�,
batch_normalization_2_8907815:	�,
batch_normalization_2_8907817:	�"
dense_1_8907824:	�T
dense_1_8907826:T+
batch_normalization_3_8907829:T+
batch_normalization_3_8907831:T+
batch_normalization_3_8907833:T+
batch_normalization_3_8907835:T
identity��+batch_normalization/StatefulPartitionedCall�-batch_normalization_1/StatefulPartitionedCall�-batch_normalization_2/StatefulPartitionedCall�-batch_normalization_3/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�-dense_1/kernel/Regularizer/Abs/ReadVariableOp�'dense_1_dropout/StatefulPartitionedCall�'stream_0_conv_1/StatefulPartitionedCall�5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp�'stream_0_conv_2/StatefulPartitionedCall�8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp�'stream_0_conv_3/StatefulPartitionedCall�5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp�'stream_0_drop_1/StatefulPartitionedCall�'stream_0_drop_2/StatefulPartitionedCall�'stream_0_drop_3/StatefulPartitionedCall�+stream_0_input_drop/StatefulPartitionedCall�
+stream_0_input_drop/StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������}* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *Y
fTRR
P__inference_stream_0_input_drop_layer_call_and_return_conditional_losses_89077122-
+stream_0_input_drop/StatefulPartitionedCall�
'stream_0_conv_1/StatefulPartitionedCallStatefulPartitionedCall4stream_0_input_drop/StatefulPartitionedCall:output:0stream_0_conv_1_8907774stream_0_conv_1_8907776*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������}@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_stream_0_conv_1_layer_call_and_return_conditional_losses_89070342)
'stream_0_conv_1/StatefulPartitionedCall�
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall0stream_0_conv_1/StatefulPartitionedCall:output:0batch_normalization_8907779batch_normalization_8907781batch_normalization_8907783batch_normalization_8907785*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������}@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_89076712-
+batch_normalization/StatefulPartitionedCall�
activation/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������}@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_activation_layer_call_and_return_conditional_losses_89070742
activation/PartitionedCall�
'stream_0_drop_1/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0,^stream_0_input_drop/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������}@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_stream_0_drop_1_layer_call_and_return_conditional_losses_89076132)
'stream_0_drop_1/StatefulPartitionedCall�
'stream_0_conv_2/StatefulPartitionedCallStatefulPartitionedCall0stream_0_drop_1/StatefulPartitionedCall:output:0stream_0_conv_2_8907790stream_0_conv_2_8907792*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������}�*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_stream_0_conv_2_layer_call_and_return_conditional_losses_89071042)
'stream_0_conv_2/StatefulPartitionedCall�
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall0stream_0_conv_2/StatefulPartitionedCall:output:0batch_normalization_1_8907795batch_normalization_1_8907797batch_normalization_1_8907799batch_normalization_1_8907801*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������}�*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *[
fVRT
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_89075722/
-batch_normalization_1/StatefulPartitionedCall�
activation_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������}�* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_activation_1_layer_call_and_return_conditional_losses_89071442
activation_1/PartitionedCall�
'stream_0_drop_2/StatefulPartitionedCallStatefulPartitionedCall%activation_1/PartitionedCall:output:0(^stream_0_drop_1/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������}�* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_stream_0_drop_2_layer_call_and_return_conditional_losses_89075142)
'stream_0_drop_2/StatefulPartitionedCall�
'stream_0_conv_3/StatefulPartitionedCallStatefulPartitionedCall0stream_0_drop_2/StatefulPartitionedCall:output:0stream_0_conv_3_8907806stream_0_conv_3_8907808*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������}�*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_stream_0_conv_3_layer_call_and_return_conditional_losses_89071742)
'stream_0_conv_3/StatefulPartitionedCall�
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall0stream_0_conv_3/StatefulPartitionedCall:output:0batch_normalization_2_8907811batch_normalization_2_8907813batch_normalization_2_8907815batch_normalization_2_8907817*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������}�*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *[
fVRT
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_89074732/
-batch_normalization_2/StatefulPartitionedCall�
activation_2/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������}�* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_activation_2_layer_call_and_return_conditional_losses_89072142
activation_2/PartitionedCall�
'stream_0_drop_3/StatefulPartitionedCallStatefulPartitionedCall%activation_2/PartitionedCall:output:0(^stream_0_drop_2/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������}�* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_stream_0_drop_3_layer_call_and_return_conditional_losses_89074152)
'stream_0_drop_3/StatefulPartitionedCall�
(global_average_pooling1d/PartitionedCallPartitionedCall0stream_0_drop_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *^
fYRW
U__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_89072282*
(global_average_pooling1d/PartitionedCall�
'dense_1_dropout/StatefulPartitionedCallStatefulPartitionedCall1global_average_pooling1d/PartitionedCall:output:0(^stream_0_drop_3/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_dense_1_dropout_layer_call_and_return_conditional_losses_89073872)
'dense_1_dropout/StatefulPartitionedCall�
dense_1/StatefulPartitionedCallStatefulPartitionedCall0dense_1_dropout/StatefulPartitionedCall:output:0dense_1_8907824dense_1_8907826*
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
D__inference_dense_1_layer_call_and_return_conditional_losses_89072532!
dense_1/StatefulPartitionedCall�
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0batch_normalization_3_8907829batch_normalization_3_8907831batch_normalization_3_8907833batch_normalization_3_8907835*
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
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_89069212/
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
O__inference_dense_activation_1_layer_call_and_return_conditional_losses_89072732$
"dense_activation_1/PartitionedCall�
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpstream_0_conv_1_8907774*"
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
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpstream_0_conv_2_8907790*#
_output_shapes
:@�*
dtype02:
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp�
)stream_0_conv_2/kernel/Regularizer/SquareSquare@stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*#
_output_shapes
:@�2+
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
5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpstream_0_conv_3_8907806*$
_output_shapes
:��*
dtype027
5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp�
&stream_0_conv_3/kernel/Regularizer/AbsAbs=stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*$
_output_shapes
:��2(
&stream_0_conv_3/kernel/Regularizer/Abs�
(stream_0_conv_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_0_conv_3/kernel/Regularizer/Const�
&stream_0_conv_3/kernel/Regularizer/SumSum*stream_0_conv_3/kernel/Regularizer/Abs:y:01stream_0_conv_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_3/kernel/Regularizer/Sum�
(stream_0_conv_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2*
(stream_0_conv_3/kernel/Regularizer/mul/x�
&stream_0_conv_3/kernel/Regularizer/mulMul1stream_0_conv_3/kernel/Regularizer/mul/x:output:0/stream_0_conv_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_3/kernel/Regularizer/mul�
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_1_8907824*
_output_shapes
:	�T*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOp�
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	�T2 
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

Identity�
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall.^dense_1/kernel/Regularizer/Abs/ReadVariableOp(^dense_1_dropout/StatefulPartitionedCall(^stream_0_conv_1/StatefulPartitionedCall6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp(^stream_0_conv_2/StatefulPartitionedCall9^stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp(^stream_0_conv_3/StatefulPartitionedCall6^stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp(^stream_0_drop_1/StatefulPartitionedCall(^stream_0_drop_2/StatefulPartitionedCall(^stream_0_drop_3/StatefulPartitionedCall,^stream_0_input_drop/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:���������}: : : : : : : : : : : : : : : : : : : : : : : : 2Z
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
:���������}
 
_user_specified_nameinputs
�h
�
#__inference__traced_restore_8911822
file_prefix=
'assignvariableop_stream_0_conv_1_kernel:@5
'assignvariableop_1_stream_0_conv_1_bias:@:
,assignvariableop_2_batch_normalization_gamma:@9
+assignvariableop_3_batch_normalization_beta:@@
2assignvariableop_4_batch_normalization_moving_mean:@D
6assignvariableop_5_batch_normalization_moving_variance:@@
)assignvariableop_6_stream_0_conv_2_kernel:@�6
'assignvariableop_7_stream_0_conv_2_bias:	�=
.assignvariableop_8_batch_normalization_1_gamma:	�<
-assignvariableop_9_batch_normalization_1_beta:	�D
5assignvariableop_10_batch_normalization_1_moving_mean:	�H
9assignvariableop_11_batch_normalization_1_moving_variance:	�B
*assignvariableop_12_stream_0_conv_3_kernel:��7
(assignvariableop_13_stream_0_conv_3_bias:	�>
/assignvariableop_14_batch_normalization_2_gamma:	�=
.assignvariableop_15_batch_normalization_2_beta:	�D
5assignvariableop_16_batch_normalization_2_moving_mean:	�H
9assignvariableop_17_batch_normalization_2_moving_variance:	�5
"assignvariableop_18_dense_1_kernel:	�T.
 assignvariableop_19_dense_1_bias:T=
/assignvariableop_20_batch_normalization_3_gamma:T<
.assignvariableop_21_batch_normalization_3_beta:TC
5assignvariableop_22_batch_normalization_3_moving_mean:TG
9assignvariableop_23_batch_normalization_3_moving_variance:T
identity_25��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*E
value<B:B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices�
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*x
_output_shapesf
d:::::::::::::::::::::::::*'
dtypes
22
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity�
AssignVariableOpAssignVariableOp'assignvariableop_stream_0_conv_1_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1�
AssignVariableOp_1AssignVariableOp'assignvariableop_1_stream_0_conv_1_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2�
AssignVariableOp_2AssignVariableOp,assignvariableop_2_batch_normalization_gammaIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3�
AssignVariableOp_3AssignVariableOp+assignvariableop_3_batch_normalization_betaIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4�
AssignVariableOp_4AssignVariableOp2assignvariableop_4_batch_normalization_moving_meanIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5�
AssignVariableOp_5AssignVariableOp6assignvariableop_5_batch_normalization_moving_varianceIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6�
AssignVariableOp_6AssignVariableOp)assignvariableop_6_stream_0_conv_2_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7�
AssignVariableOp_7AssignVariableOp'assignvariableop_7_stream_0_conv_2_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8�
AssignVariableOp_8AssignVariableOp.assignvariableop_8_batch_normalization_1_gammaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9�
AssignVariableOp_9AssignVariableOp-assignvariableop_9_batch_normalization_1_betaIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10�
AssignVariableOp_10AssignVariableOp5assignvariableop_10_batch_normalization_1_moving_meanIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11�
AssignVariableOp_11AssignVariableOp9assignvariableop_11_batch_normalization_1_moving_varianceIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12�
AssignVariableOp_12AssignVariableOp*assignvariableop_12_stream_0_conv_3_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13�
AssignVariableOp_13AssignVariableOp(assignvariableop_13_stream_0_conv_3_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14�
AssignVariableOp_14AssignVariableOp/assignvariableop_14_batch_normalization_2_gammaIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15�
AssignVariableOp_15AssignVariableOp.assignvariableop_15_batch_normalization_2_betaIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16�
AssignVariableOp_16AssignVariableOp5assignvariableop_16_batch_normalization_2_moving_meanIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17�
AssignVariableOp_17AssignVariableOp9assignvariableop_17_batch_normalization_2_moving_varianceIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18�
AssignVariableOp_18AssignVariableOp"assignvariableop_18_dense_1_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19�
AssignVariableOp_19AssignVariableOp assignvariableop_19_dense_1_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20�
AssignVariableOp_20AssignVariableOp/assignvariableop_20_batch_normalization_3_gammaIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21�
AssignVariableOp_21AssignVariableOp.assignvariableop_21_batch_normalization_3_betaIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22�
AssignVariableOp_22AssignVariableOp5assignvariableop_22_batch_normalization_3_moving_meanIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23�
AssignVariableOp_23AssignVariableOp9assignvariableop_23_batch_normalization_3_moving_varianceIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_239
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp�
Identity_24Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_24f
Identity_25IdentityIdentity_24:output:0^NoOp_1*
T0*
_output_shapes
: 2
Identity_25�
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 2
NoOp_1"#
identity_25Identity_25:output:0*E
_input_shapes4
2: : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_23AssignVariableOp_232(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
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
�	
�
5__inference_batch_normalization_layer_call_fn_8910781

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
P__inference_batch_normalization_layer_call_and_return_conditional_losses_89063512
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
�
�
7__inference_batch_normalization_2_layer_call_fn_8911286

inputs
unknown:	�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������}�*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *[
fVRT
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_89074732
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:���������}�2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������}�: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:���������}�
 
_user_specified_nameinputs
�+
�
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_8906573

inputs6
'assignmovingavg_readvariableop_resource:	�8
)assignmovingavg_1_readvariableop_resource:	�4
%batchnorm_mul_readvariableop_resource:	�0
!batchnorm_readvariableop_resource:	�
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOp�
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indices�
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*#
_output_shapes
:�*
	keep_dims(2
moments/mean�
moments/StopGradientStopGradientmoments/mean:output:0*
T0*#
_output_shapes
:�2
moments/StopGradient�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*5
_output_shapes#
!:�������������������2
moments/SquaredDifference�
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices�
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*#
_output_shapes
:�*
	keep_dims(2
moments/variance�
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 2
moments/Squeeze�
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:�*
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
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:�*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:�2
AssignMovingAvg/sub�
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:�2
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
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:�*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:�2
AssignMovingAvg_1/sub�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:�2
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
T0*
_output_shapes	
:�2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:�2
batchnorm/Rsqrt�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype02
batchnorm/mul/ReadVariableOp�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2
batchnorm/mul�
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*5
_output_shapes#
!:�������������������2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:�2
batchnorm/mul_2�
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype02
batchnorm/ReadVariableOp�
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*5
_output_shapes#
!:�������������������2
batchnorm/add_1|
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*5
_output_shapes#
!:�������������������2

Identity�
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):�������������������: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:] Y
5
_output_shapes#
!:�������������������
 
_user_specified_nameinputs
�
�
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_8911073

inputs0
!batchnorm_readvariableop_resource:	�4
%batchnorm_mul_readvariableop_resource:	�2
#batchnorm_readvariableop_1_resource:	�2
#batchnorm_readvariableop_2_resource:	�
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOp�
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:�*
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
T0*
_output_shapes	
:�2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:�2
batchnorm/Rsqrt�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype02
batchnorm/mul/ReadVariableOp�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2
batchnorm/mul�
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*5
_output_shapes#
!:�������������������2
batchnorm/mul_1�
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:�*
dtype02
batchnorm/ReadVariableOp_1�
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:�2
batchnorm/mul_2�
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:�*
dtype02
batchnorm/ReadVariableOp_2�
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*5
_output_shapes#
!:�������������������2
batchnorm/add_1|
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*5
_output_shapes#
!:�������������������2

Identity�
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):�������������������: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:] Y
5
_output_shapes#
!:�������������������
 
_user_specified_nameinputs
��
�
D__inference_model_1_layer_call_and_return_conditional_losses_8909486

inputs[
Ebasemodel_stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource:@G
9basemodel_stream_0_conv_1_biasadd_readvariableop_resource:@M
?basemodel_batch_normalization_batchnorm_readvariableop_resource:@Q
Cbasemodel_batch_normalization_batchnorm_mul_readvariableop_resource:@O
Abasemodel_batch_normalization_batchnorm_readvariableop_1_resource:@O
Abasemodel_batch_normalization_batchnorm_readvariableop_2_resource:@\
Ebasemodel_stream_0_conv_2_conv1d_expanddims_1_readvariableop_resource:@�H
9basemodel_stream_0_conv_2_biasadd_readvariableop_resource:	�P
Abasemodel_batch_normalization_1_batchnorm_readvariableop_resource:	�T
Ebasemodel_batch_normalization_1_batchnorm_mul_readvariableop_resource:	�R
Cbasemodel_batch_normalization_1_batchnorm_readvariableop_1_resource:	�R
Cbasemodel_batch_normalization_1_batchnorm_readvariableop_2_resource:	�]
Ebasemodel_stream_0_conv_3_conv1d_expanddims_1_readvariableop_resource:��H
9basemodel_stream_0_conv_3_biasadd_readvariableop_resource:	�P
Abasemodel_batch_normalization_2_batchnorm_readvariableop_resource:	�T
Ebasemodel_batch_normalization_2_batchnorm_mul_readvariableop_resource:	�R
Cbasemodel_batch_normalization_2_batchnorm_readvariableop_1_resource:	�R
Cbasemodel_batch_normalization_2_batchnorm_readvariableop_2_resource:	�C
0basemodel_dense_1_matmul_readvariableop_resource:	�T?
1basemodel_dense_1_biasadd_readvariableop_resource:TO
Abasemodel_batch_normalization_3_batchnorm_readvariableop_resource:TS
Ebasemodel_batch_normalization_3_batchnorm_mul_readvariableop_resource:TQ
Cbasemodel_batch_normalization_3_batchnorm_readvariableop_1_resource:TQ
Cbasemodel_batch_normalization_3_batchnorm_readvariableop_2_resource:T
identity��6basemodel/batch_normalization/batchnorm/ReadVariableOp�8basemodel/batch_normalization/batchnorm/ReadVariableOp_1�8basemodel/batch_normalization/batchnorm/ReadVariableOp_2�:basemodel/batch_normalization/batchnorm/mul/ReadVariableOp�8basemodel/batch_normalization_1/batchnorm/ReadVariableOp�:basemodel/batch_normalization_1/batchnorm/ReadVariableOp_1�:basemodel/batch_normalization_1/batchnorm/ReadVariableOp_2�<basemodel/batch_normalization_1/batchnorm/mul/ReadVariableOp�8basemodel/batch_normalization_2/batchnorm/ReadVariableOp�:basemodel/batch_normalization_2/batchnorm/ReadVariableOp_1�:basemodel/batch_normalization_2/batchnorm/ReadVariableOp_2�<basemodel/batch_normalization_2/batchnorm/mul/ReadVariableOp�8basemodel/batch_normalization_3/batchnorm/ReadVariableOp�:basemodel/batch_normalization_3/batchnorm/ReadVariableOp_1�:basemodel/batch_normalization_3/batchnorm/ReadVariableOp_2�<basemodel/batch_normalization_3/batchnorm/mul/ReadVariableOp�(basemodel/dense_1/BiasAdd/ReadVariableOp�'basemodel/dense_1/MatMul/ReadVariableOp�0basemodel/stream_0_conv_1/BiasAdd/ReadVariableOp�<basemodel/stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp�0basemodel/stream_0_conv_2/BiasAdd/ReadVariableOp�<basemodel/stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOp�0basemodel/stream_0_conv_3/BiasAdd/ReadVariableOp�<basemodel/stream_0_conv_3/conv1d/ExpandDims_1/ReadVariableOp�-dense_1/kernel/Regularizer/Abs/ReadVariableOp�5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp�8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp�5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp�
&basemodel/stream_0_input_drop/IdentityIdentityinputs*
T0*+
_output_shapes
:���������}2(
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
T0*/
_output_shapes
:���������}2-
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
T0*/
_output_shapes
:���������}@*
paddingSAME*
strides
2"
 basemodel/stream_0_conv_1/conv1d�
(basemodel/stream_0_conv_1/conv1d/SqueezeSqueeze)basemodel/stream_0_conv_1/conv1d:output:0*
T0*+
_output_shapes
:���������}@*
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
T0*+
_output_shapes
:���������}@2#
!basemodel/stream_0_conv_1/BiasAdd�
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
T0*+
_output_shapes
:���������}@2/
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
T0*+
_output_shapes
:���������}@2/
-basemodel/batch_normalization/batchnorm/add_1�
basemodel/activation/ReluRelu1basemodel/batch_normalization/batchnorm/add_1:z:0*
T0*+
_output_shapes
:���������}@2
basemodel/activation/Relu�
"basemodel/stream_0_drop_1/IdentityIdentity'basemodel/activation/Relu:activations:0*
T0*+
_output_shapes
:���������}@2$
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
T0*/
_output_shapes
:���������}@2-
+basemodel/stream_0_conv_2/conv1d/ExpandDims�
<basemodel/stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpEbasemodel_stream_0_conv_2_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:@�*
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
T0*'
_output_shapes
:@�2/
-basemodel/stream_0_conv_2/conv1d/ExpandDims_1�
 basemodel/stream_0_conv_2/conv1dConv2D4basemodel/stream_0_conv_2/conv1d/ExpandDims:output:06basemodel/stream_0_conv_2/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:���������}�*
paddingSAME*
strides
2"
 basemodel/stream_0_conv_2/conv1d�
(basemodel/stream_0_conv_2/conv1d/SqueezeSqueeze)basemodel/stream_0_conv_2/conv1d:output:0*
T0*,
_output_shapes
:���������}�*
squeeze_dims

���������2*
(basemodel/stream_0_conv_2/conv1d/Squeeze�
0basemodel/stream_0_conv_2/BiasAdd/ReadVariableOpReadVariableOp9basemodel_stream_0_conv_2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype022
0basemodel/stream_0_conv_2/BiasAdd/ReadVariableOp�
!basemodel/stream_0_conv_2/BiasAddBiasAdd1basemodel/stream_0_conv_2/conv1d/Squeeze:output:08basemodel/stream_0_conv_2/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������}�2#
!basemodel/stream_0_conv_2/BiasAdd�
8basemodel/batch_normalization_1/batchnorm/ReadVariableOpReadVariableOpAbasemodel_batch_normalization_1_batchnorm_readvariableop_resource*
_output_shapes	
:�*
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
T0*
_output_shapes	
:�2/
-basemodel/batch_normalization_1/batchnorm/add�
/basemodel/batch_normalization_1/batchnorm/RsqrtRsqrt1basemodel/batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes	
:�21
/basemodel/batch_normalization_1/batchnorm/Rsqrt�
<basemodel/batch_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOpEbasemodel_batch_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype02>
<basemodel/batch_normalization_1/batchnorm/mul/ReadVariableOp�
-basemodel/batch_normalization_1/batchnorm/mulMul3basemodel/batch_normalization_1/batchnorm/Rsqrt:y:0Dbasemodel/batch_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2/
-basemodel/batch_normalization_1/batchnorm/mul�
/basemodel/batch_normalization_1/batchnorm/mul_1Mul*basemodel/stream_0_conv_2/BiasAdd:output:01basemodel/batch_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:���������}�21
/basemodel/batch_normalization_1/batchnorm/mul_1�
:basemodel/batch_normalization_1/batchnorm/ReadVariableOp_1ReadVariableOpCbasemodel_batch_normalization_1_batchnorm_readvariableop_1_resource*
_output_shapes	
:�*
dtype02<
:basemodel/batch_normalization_1/batchnorm/ReadVariableOp_1�
/basemodel/batch_normalization_1/batchnorm/mul_2MulBbasemodel/batch_normalization_1/batchnorm/ReadVariableOp_1:value:01basemodel/batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes	
:�21
/basemodel/batch_normalization_1/batchnorm/mul_2�
:basemodel/batch_normalization_1/batchnorm/ReadVariableOp_2ReadVariableOpCbasemodel_batch_normalization_1_batchnorm_readvariableop_2_resource*
_output_shapes	
:�*
dtype02<
:basemodel/batch_normalization_1/batchnorm/ReadVariableOp_2�
-basemodel/batch_normalization_1/batchnorm/subSubBbasemodel/batch_normalization_1/batchnorm/ReadVariableOp_2:value:03basemodel/batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�2/
-basemodel/batch_normalization_1/batchnorm/sub�
/basemodel/batch_normalization_1/batchnorm/add_1AddV23basemodel/batch_normalization_1/batchnorm/mul_1:z:01basemodel/batch_normalization_1/batchnorm/sub:z:0*
T0*,
_output_shapes
:���������}�21
/basemodel/batch_normalization_1/batchnorm/add_1�
basemodel/activation_1/ReluRelu3basemodel/batch_normalization_1/batchnorm/add_1:z:0*
T0*,
_output_shapes
:���������}�2
basemodel/activation_1/Relu�
"basemodel/stream_0_drop_2/IdentityIdentity)basemodel/activation_1/Relu:activations:0*
T0*,
_output_shapes
:���������}�2$
"basemodel/stream_0_drop_2/Identity�
/basemodel/stream_0_conv_3/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������21
/basemodel/stream_0_conv_3/conv1d/ExpandDims/dim�
+basemodel/stream_0_conv_3/conv1d/ExpandDims
ExpandDims+basemodel/stream_0_drop_2/Identity:output:08basemodel/stream_0_conv_3/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:���������}�2-
+basemodel/stream_0_conv_3/conv1d/ExpandDims�
<basemodel/stream_0_conv_3/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpEbasemodel_stream_0_conv_3_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:��*
dtype02>
<basemodel/stream_0_conv_3/conv1d/ExpandDims_1/ReadVariableOp�
1basemodel/stream_0_conv_3/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 23
1basemodel/stream_0_conv_3/conv1d/ExpandDims_1/dim�
-basemodel/stream_0_conv_3/conv1d/ExpandDims_1
ExpandDimsDbasemodel/stream_0_conv_3/conv1d/ExpandDims_1/ReadVariableOp:value:0:basemodel/stream_0_conv_3/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:��2/
-basemodel/stream_0_conv_3/conv1d/ExpandDims_1�
 basemodel/stream_0_conv_3/conv1dConv2D4basemodel/stream_0_conv_3/conv1d/ExpandDims:output:06basemodel/stream_0_conv_3/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:���������}�*
paddingSAME*
strides
2"
 basemodel/stream_0_conv_3/conv1d�
(basemodel/stream_0_conv_3/conv1d/SqueezeSqueeze)basemodel/stream_0_conv_3/conv1d:output:0*
T0*,
_output_shapes
:���������}�*
squeeze_dims

���������2*
(basemodel/stream_0_conv_3/conv1d/Squeeze�
0basemodel/stream_0_conv_3/BiasAdd/ReadVariableOpReadVariableOp9basemodel_stream_0_conv_3_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype022
0basemodel/stream_0_conv_3/BiasAdd/ReadVariableOp�
!basemodel/stream_0_conv_3/BiasAddBiasAdd1basemodel/stream_0_conv_3/conv1d/Squeeze:output:08basemodel/stream_0_conv_3/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������}�2#
!basemodel/stream_0_conv_3/BiasAdd�
8basemodel/batch_normalization_2/batchnorm/ReadVariableOpReadVariableOpAbasemodel_batch_normalization_2_batchnorm_readvariableop_resource*
_output_shapes	
:�*
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
T0*
_output_shapes	
:�2/
-basemodel/batch_normalization_2/batchnorm/add�
/basemodel/batch_normalization_2/batchnorm/RsqrtRsqrt1basemodel/batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes	
:�21
/basemodel/batch_normalization_2/batchnorm/Rsqrt�
<basemodel/batch_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOpEbasemodel_batch_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype02>
<basemodel/batch_normalization_2/batchnorm/mul/ReadVariableOp�
-basemodel/batch_normalization_2/batchnorm/mulMul3basemodel/batch_normalization_2/batchnorm/Rsqrt:y:0Dbasemodel/batch_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2/
-basemodel/batch_normalization_2/batchnorm/mul�
/basemodel/batch_normalization_2/batchnorm/mul_1Mul*basemodel/stream_0_conv_3/BiasAdd:output:01basemodel/batch_normalization_2/batchnorm/mul:z:0*
T0*,
_output_shapes
:���������}�21
/basemodel/batch_normalization_2/batchnorm/mul_1�
:basemodel/batch_normalization_2/batchnorm/ReadVariableOp_1ReadVariableOpCbasemodel_batch_normalization_2_batchnorm_readvariableop_1_resource*
_output_shapes	
:�*
dtype02<
:basemodel/batch_normalization_2/batchnorm/ReadVariableOp_1�
/basemodel/batch_normalization_2/batchnorm/mul_2MulBbasemodel/batch_normalization_2/batchnorm/ReadVariableOp_1:value:01basemodel/batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes	
:�21
/basemodel/batch_normalization_2/batchnorm/mul_2�
:basemodel/batch_normalization_2/batchnorm/ReadVariableOp_2ReadVariableOpCbasemodel_batch_normalization_2_batchnorm_readvariableop_2_resource*
_output_shapes	
:�*
dtype02<
:basemodel/batch_normalization_2/batchnorm/ReadVariableOp_2�
-basemodel/batch_normalization_2/batchnorm/subSubBbasemodel/batch_normalization_2/batchnorm/ReadVariableOp_2:value:03basemodel/batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�2/
-basemodel/batch_normalization_2/batchnorm/sub�
/basemodel/batch_normalization_2/batchnorm/add_1AddV23basemodel/batch_normalization_2/batchnorm/mul_1:z:01basemodel/batch_normalization_2/batchnorm/sub:z:0*
T0*,
_output_shapes
:���������}�21
/basemodel/batch_normalization_2/batchnorm/add_1�
basemodel/activation_2/ReluRelu3basemodel/batch_normalization_2/batchnorm/add_1:z:0*
T0*,
_output_shapes
:���������}�2
basemodel/activation_2/Relu�
"basemodel/stream_0_drop_3/IdentityIdentity)basemodel/activation_2/Relu:activations:0*
T0*,
_output_shapes
:���������}�2$
"basemodel/stream_0_drop_3/Identity�
9basemodel/global_average_pooling1d/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2;
9basemodel/global_average_pooling1d/Mean/reduction_indices�
'basemodel/global_average_pooling1d/MeanMean+basemodel/stream_0_drop_3/Identity:output:0Bbasemodel/global_average_pooling1d/Mean/reduction_indices:output:0*
T0*(
_output_shapes
:����������2)
'basemodel/global_average_pooling1d/Mean�
"basemodel/dense_1_dropout/IdentityIdentity0basemodel/global_average_pooling1d/Mean:output:0*
T0*(
_output_shapes
:����������2$
"basemodel/dense_1_dropout/Identity�
'basemodel/dense_1/MatMul/ReadVariableOpReadVariableOp0basemodel_dense_1_matmul_readvariableop_resource*
_output_shapes
:	�T*
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
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpEbasemodel_stream_0_conv_2_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:@�*
dtype02:
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp�
)stream_0_conv_2/kernel/Regularizer/SquareSquare@stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*#
_output_shapes
:@�2+
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
5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpEbasemodel_stream_0_conv_3_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:��*
dtype027
5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp�
&stream_0_conv_3/kernel/Regularizer/AbsAbs=stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*$
_output_shapes
:��2(
&stream_0_conv_3/kernel/Regularizer/Abs�
(stream_0_conv_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_0_conv_3/kernel/Regularizer/Const�
&stream_0_conv_3/kernel/Regularizer/SumSum*stream_0_conv_3/kernel/Regularizer/Abs:y:01stream_0_conv_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_3/kernel/Regularizer/Sum�
(stream_0_conv_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2*
(stream_0_conv_3/kernel/Regularizer/mul/x�
&stream_0_conv_3/kernel/Regularizer/mulMul1stream_0_conv_3/kernel/Regularizer/mul/x:output:0/stream_0_conv_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_3/kernel/Regularizer/mul�
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp0basemodel_dense_1_matmul_readvariableop_resource*
_output_shapes
:	�T*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOp�
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	�T2 
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
IdentityIdentity(basemodel/dense_activation_1/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:���������T2

Identity�
NoOpNoOp7^basemodel/batch_normalization/batchnorm/ReadVariableOp9^basemodel/batch_normalization/batchnorm/ReadVariableOp_19^basemodel/batch_normalization/batchnorm/ReadVariableOp_2;^basemodel/batch_normalization/batchnorm/mul/ReadVariableOp9^basemodel/batch_normalization_1/batchnorm/ReadVariableOp;^basemodel/batch_normalization_1/batchnorm/ReadVariableOp_1;^basemodel/batch_normalization_1/batchnorm/ReadVariableOp_2=^basemodel/batch_normalization_1/batchnorm/mul/ReadVariableOp9^basemodel/batch_normalization_2/batchnorm/ReadVariableOp;^basemodel/batch_normalization_2/batchnorm/ReadVariableOp_1;^basemodel/batch_normalization_2/batchnorm/ReadVariableOp_2=^basemodel/batch_normalization_2/batchnorm/mul/ReadVariableOp9^basemodel/batch_normalization_3/batchnorm/ReadVariableOp;^basemodel/batch_normalization_3/batchnorm/ReadVariableOp_1;^basemodel/batch_normalization_3/batchnorm/ReadVariableOp_2=^basemodel/batch_normalization_3/batchnorm/mul/ReadVariableOp)^basemodel/dense_1/BiasAdd/ReadVariableOp(^basemodel/dense_1/MatMul/ReadVariableOp1^basemodel/stream_0_conv_1/BiasAdd/ReadVariableOp=^basemodel/stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp1^basemodel/stream_0_conv_2/BiasAdd/ReadVariableOp=^basemodel/stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOp1^basemodel/stream_0_conv_3/BiasAdd/ReadVariableOp=^basemodel/stream_0_conv_3/conv1d/ExpandDims_1/ReadVariableOp.^dense_1/kernel/Regularizer/Abs/ReadVariableOp6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp9^stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp6^stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:���������}: : : : : : : : : : : : : : : : : : : : : : : : 2p
6basemodel/batch_normalization/batchnorm/ReadVariableOp6basemodel/batch_normalization/batchnorm/ReadVariableOp2t
8basemodel/batch_normalization/batchnorm/ReadVariableOp_18basemodel/batch_normalization/batchnorm/ReadVariableOp_12t
8basemodel/batch_normalization/batchnorm/ReadVariableOp_28basemodel/batch_normalization/batchnorm/ReadVariableOp_22x
:basemodel/batch_normalization/batchnorm/mul/ReadVariableOp:basemodel/batch_normalization/batchnorm/mul/ReadVariableOp2t
8basemodel/batch_normalization_1/batchnorm/ReadVariableOp8basemodel/batch_normalization_1/batchnorm/ReadVariableOp2x
:basemodel/batch_normalization_1/batchnorm/ReadVariableOp_1:basemodel/batch_normalization_1/batchnorm/ReadVariableOp_12x
:basemodel/batch_normalization_1/batchnorm/ReadVariableOp_2:basemodel/batch_normalization_1/batchnorm/ReadVariableOp_22|
<basemodel/batch_normalization_1/batchnorm/mul/ReadVariableOp<basemodel/batch_normalization_1/batchnorm/mul/ReadVariableOp2t
8basemodel/batch_normalization_2/batchnorm/ReadVariableOp8basemodel/batch_normalization_2/batchnorm/ReadVariableOp2x
:basemodel/batch_normalization_2/batchnorm/ReadVariableOp_1:basemodel/batch_normalization_2/batchnorm/ReadVariableOp_12x
:basemodel/batch_normalization_2/batchnorm/ReadVariableOp_2:basemodel/batch_normalization_2/batchnorm/ReadVariableOp_22|
<basemodel/batch_normalization_2/batchnorm/mul/ReadVariableOp<basemodel/batch_normalization_2/batchnorm/mul/ReadVariableOp2t
8basemodel/batch_normalization_3/batchnorm/ReadVariableOp8basemodel/batch_normalization_3/batchnorm/ReadVariableOp2x
:basemodel/batch_normalization_3/batchnorm/ReadVariableOp_1:basemodel/batch_normalization_3/batchnorm/ReadVariableOp_12x
:basemodel/batch_normalization_3/batchnorm/ReadVariableOp_2:basemodel/batch_normalization_3/batchnorm/ReadVariableOp_22|
<basemodel/batch_normalization_3/batchnorm/mul/ReadVariableOp<basemodel/batch_normalization_3/batchnorm/mul/ReadVariableOp2T
(basemodel/dense_1/BiasAdd/ReadVariableOp(basemodel/dense_1/BiasAdd/ReadVariableOp2R
'basemodel/dense_1/MatMul/ReadVariableOp'basemodel/dense_1/MatMul/ReadVariableOp2d
0basemodel/stream_0_conv_1/BiasAdd/ReadVariableOp0basemodel/stream_0_conv_1/BiasAdd/ReadVariableOp2|
<basemodel/stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp<basemodel/stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp2d
0basemodel/stream_0_conv_2/BiasAdd/ReadVariableOp0basemodel/stream_0_conv_2/BiasAdd/ReadVariableOp2|
<basemodel/stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOp<basemodel/stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOp2d
0basemodel/stream_0_conv_3/BiasAdd/ReadVariableOp0basemodel/stream_0_conv_3/BiasAdd/ReadVariableOp2|
<basemodel/stream_0_conv_3/conv1d/ExpandDims_1/ReadVariableOp<basemodel/stream_0_conv_3/conv1d/ExpandDims_1/ReadVariableOp2^
-dense_1/kernel/Regularizer/Abs/ReadVariableOp-dense_1/kernel/Regularizer/Abs/ReadVariableOp2n
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp2t
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp2n
5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp:S O
+
_output_shapes
:���������}
 
_user_specified_nameinputs
�
�
7__inference_batch_normalization_1_layer_call_fn_8911040

inputs
unknown:	�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������}�*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *[
fVRT
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_89071292
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:���������}�2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������}�: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:���������}�
 
_user_specified_nameinputs
�
�
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_8911306

inputs0
!batchnorm_readvariableop_resource:	�4
%batchnorm_mul_readvariableop_resource:	�2
#batchnorm_readvariableop_1_resource:	�2
#batchnorm_readvariableop_2_resource:	�
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOp�
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:�*
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
T0*
_output_shapes	
:�2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:�2
batchnorm/Rsqrt�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype02
batchnorm/mul/ReadVariableOp�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2
batchnorm/mul�
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*5
_output_shapes#
!:�������������������2
batchnorm/mul_1�
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:�*
dtype02
batchnorm/ReadVariableOp_1�
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:�2
batchnorm/mul_2�
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:�*
dtype02
batchnorm/ReadVariableOp_2�
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*5
_output_shapes#
!:�������������������2
batchnorm/add_1|
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*5
_output_shapes#
!:�������������������2

Identity�
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):�������������������: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:] Y
5
_output_shapes#
!:�������������������
 
_user_specified_nameinputs
�
j
L__inference_stream_0_drop_3_layer_call_and_return_conditional_losses_8907221

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:���������}�2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:���������}�2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:���������}�:T P
,
_output_shapes
:���������}�
 
_user_specified_nameinputs
�
�
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_8907199

inputs0
!batchnorm_readvariableop_resource:	�4
%batchnorm_mul_readvariableop_resource:	�2
#batchnorm_readvariableop_1_resource:	�2
#batchnorm_readvariableop_2_resource:	�
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOp�
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:�*
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
T0*
_output_shapes	
:�2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:�2
batchnorm/Rsqrt�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype02
batchnorm/mul/ReadVariableOp�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2
batchnorm/mul{
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*,
_output_shapes
:���������}�2
batchnorm/mul_1�
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:�*
dtype02
batchnorm/ReadVariableOp_1�
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:�2
batchnorm/mul_2�
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:�*
dtype02
batchnorm/ReadVariableOp_2�
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*,
_output_shapes
:���������}�2
batchnorm/add_1s
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*,
_output_shapes
:���������}�2

Identity�
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������}�: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:T P
,
_output_shapes
:���������}�
 
_user_specified_nameinputs
��
�
"__inference__wrapped_model_8906327
left_inputsc
Mmodel_1_basemodel_stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource:@O
Amodel_1_basemodel_stream_0_conv_1_biasadd_readvariableop_resource:@U
Gmodel_1_basemodel_batch_normalization_batchnorm_readvariableop_resource:@Y
Kmodel_1_basemodel_batch_normalization_batchnorm_mul_readvariableop_resource:@W
Imodel_1_basemodel_batch_normalization_batchnorm_readvariableop_1_resource:@W
Imodel_1_basemodel_batch_normalization_batchnorm_readvariableop_2_resource:@d
Mmodel_1_basemodel_stream_0_conv_2_conv1d_expanddims_1_readvariableop_resource:@�P
Amodel_1_basemodel_stream_0_conv_2_biasadd_readvariableop_resource:	�X
Imodel_1_basemodel_batch_normalization_1_batchnorm_readvariableop_resource:	�\
Mmodel_1_basemodel_batch_normalization_1_batchnorm_mul_readvariableop_resource:	�Z
Kmodel_1_basemodel_batch_normalization_1_batchnorm_readvariableop_1_resource:	�Z
Kmodel_1_basemodel_batch_normalization_1_batchnorm_readvariableop_2_resource:	�e
Mmodel_1_basemodel_stream_0_conv_3_conv1d_expanddims_1_readvariableop_resource:��P
Amodel_1_basemodel_stream_0_conv_3_biasadd_readvariableop_resource:	�X
Imodel_1_basemodel_batch_normalization_2_batchnorm_readvariableop_resource:	�\
Mmodel_1_basemodel_batch_normalization_2_batchnorm_mul_readvariableop_resource:	�Z
Kmodel_1_basemodel_batch_normalization_2_batchnorm_readvariableop_1_resource:	�Z
Kmodel_1_basemodel_batch_normalization_2_batchnorm_readvariableop_2_resource:	�K
8model_1_basemodel_dense_1_matmul_readvariableop_resource:	�TG
9model_1_basemodel_dense_1_biasadd_readvariableop_resource:TW
Imodel_1_basemodel_batch_normalization_3_batchnorm_readvariableop_resource:T[
Mmodel_1_basemodel_batch_normalization_3_batchnorm_mul_readvariableop_resource:TY
Kmodel_1_basemodel_batch_normalization_3_batchnorm_readvariableop_1_resource:TY
Kmodel_1_basemodel_batch_normalization_3_batchnorm_readvariableop_2_resource:T
identity��>model_1/basemodel/batch_normalization/batchnorm/ReadVariableOp�@model_1/basemodel/batch_normalization/batchnorm/ReadVariableOp_1�@model_1/basemodel/batch_normalization/batchnorm/ReadVariableOp_2�Bmodel_1/basemodel/batch_normalization/batchnorm/mul/ReadVariableOp�@model_1/basemodel/batch_normalization_1/batchnorm/ReadVariableOp�Bmodel_1/basemodel/batch_normalization_1/batchnorm/ReadVariableOp_1�Bmodel_1/basemodel/batch_normalization_1/batchnorm/ReadVariableOp_2�Dmodel_1/basemodel/batch_normalization_1/batchnorm/mul/ReadVariableOp�@model_1/basemodel/batch_normalization_2/batchnorm/ReadVariableOp�Bmodel_1/basemodel/batch_normalization_2/batchnorm/ReadVariableOp_1�Bmodel_1/basemodel/batch_normalization_2/batchnorm/ReadVariableOp_2�Dmodel_1/basemodel/batch_normalization_2/batchnorm/mul/ReadVariableOp�@model_1/basemodel/batch_normalization_3/batchnorm/ReadVariableOp�Bmodel_1/basemodel/batch_normalization_3/batchnorm/ReadVariableOp_1�Bmodel_1/basemodel/batch_normalization_3/batchnorm/ReadVariableOp_2�Dmodel_1/basemodel/batch_normalization_3/batchnorm/mul/ReadVariableOp�0model_1/basemodel/dense_1/BiasAdd/ReadVariableOp�/model_1/basemodel/dense_1/MatMul/ReadVariableOp�8model_1/basemodel/stream_0_conv_1/BiasAdd/ReadVariableOp�Dmodel_1/basemodel/stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp�8model_1/basemodel/stream_0_conv_2/BiasAdd/ReadVariableOp�Dmodel_1/basemodel/stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOp�8model_1/basemodel/stream_0_conv_3/BiasAdd/ReadVariableOp�Dmodel_1/basemodel/stream_0_conv_3/conv1d/ExpandDims_1/ReadVariableOp�
.model_1/basemodel/stream_0_input_drop/IdentityIdentityleft_inputs*
T0*+
_output_shapes
:���������}20
.model_1/basemodel/stream_0_input_drop/Identity�
7model_1/basemodel/stream_0_conv_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������29
7model_1/basemodel/stream_0_conv_1/conv1d/ExpandDims/dim�
3model_1/basemodel/stream_0_conv_1/conv1d/ExpandDims
ExpandDims7model_1/basemodel/stream_0_input_drop/Identity:output:0@model_1/basemodel/stream_0_conv_1/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������}25
3model_1/basemodel/stream_0_conv_1/conv1d/ExpandDims�
Dmodel_1/basemodel/stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpMmodel_1_basemodel_stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02F
Dmodel_1/basemodel/stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp�
9model_1/basemodel/stream_0_conv_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2;
9model_1/basemodel/stream_0_conv_1/conv1d/ExpandDims_1/dim�
5model_1/basemodel/stream_0_conv_1/conv1d/ExpandDims_1
ExpandDimsLmodel_1/basemodel/stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp:value:0Bmodel_1/basemodel/stream_0_conv_1/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@27
5model_1/basemodel/stream_0_conv_1/conv1d/ExpandDims_1�
(model_1/basemodel/stream_0_conv_1/conv1dConv2D<model_1/basemodel/stream_0_conv_1/conv1d/ExpandDims:output:0>model_1/basemodel/stream_0_conv_1/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:���������}@*
paddingSAME*
strides
2*
(model_1/basemodel/stream_0_conv_1/conv1d�
0model_1/basemodel/stream_0_conv_1/conv1d/SqueezeSqueeze1model_1/basemodel/stream_0_conv_1/conv1d:output:0*
T0*+
_output_shapes
:���������}@*
squeeze_dims

���������22
0model_1/basemodel/stream_0_conv_1/conv1d/Squeeze�
8model_1/basemodel/stream_0_conv_1/BiasAdd/ReadVariableOpReadVariableOpAmodel_1_basemodel_stream_0_conv_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02:
8model_1/basemodel/stream_0_conv_1/BiasAdd/ReadVariableOp�
)model_1/basemodel/stream_0_conv_1/BiasAddBiasAdd9model_1/basemodel/stream_0_conv_1/conv1d/Squeeze:output:0@model_1/basemodel/stream_0_conv_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������}@2+
)model_1/basemodel/stream_0_conv_1/BiasAdd�
>model_1/basemodel/batch_normalization/batchnorm/ReadVariableOpReadVariableOpGmodel_1_basemodel_batch_normalization_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02@
>model_1/basemodel/batch_normalization/batchnorm/ReadVariableOp�
5model_1/basemodel/batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:27
5model_1/basemodel/batch_normalization/batchnorm/add/y�
3model_1/basemodel/batch_normalization/batchnorm/addAddV2Fmodel_1/basemodel/batch_normalization/batchnorm/ReadVariableOp:value:0>model_1/basemodel/batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes
:@25
3model_1/basemodel/batch_normalization/batchnorm/add�
5model_1/basemodel/batch_normalization/batchnorm/RsqrtRsqrt7model_1/basemodel/batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes
:@27
5model_1/basemodel/batch_normalization/batchnorm/Rsqrt�
Bmodel_1/basemodel/batch_normalization/batchnorm/mul/ReadVariableOpReadVariableOpKmodel_1_basemodel_batch_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02D
Bmodel_1/basemodel/batch_normalization/batchnorm/mul/ReadVariableOp�
3model_1/basemodel/batch_normalization/batchnorm/mulMul9model_1/basemodel/batch_normalization/batchnorm/Rsqrt:y:0Jmodel_1/basemodel/batch_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@25
3model_1/basemodel/batch_normalization/batchnorm/mul�
5model_1/basemodel/batch_normalization/batchnorm/mul_1Mul2model_1/basemodel/stream_0_conv_1/BiasAdd:output:07model_1/basemodel/batch_normalization/batchnorm/mul:z:0*
T0*+
_output_shapes
:���������}@27
5model_1/basemodel/batch_normalization/batchnorm/mul_1�
@model_1/basemodel/batch_normalization/batchnorm/ReadVariableOp_1ReadVariableOpImodel_1_basemodel_batch_normalization_batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype02B
@model_1/basemodel/batch_normalization/batchnorm/ReadVariableOp_1�
5model_1/basemodel/batch_normalization/batchnorm/mul_2MulHmodel_1/basemodel/batch_normalization/batchnorm/ReadVariableOp_1:value:07model_1/basemodel/batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes
:@27
5model_1/basemodel/batch_normalization/batchnorm/mul_2�
@model_1/basemodel/batch_normalization/batchnorm/ReadVariableOp_2ReadVariableOpImodel_1_basemodel_batch_normalization_batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype02B
@model_1/basemodel/batch_normalization/batchnorm/ReadVariableOp_2�
3model_1/basemodel/batch_normalization/batchnorm/subSubHmodel_1/basemodel/batch_normalization/batchnorm/ReadVariableOp_2:value:09model_1/basemodel/batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@25
3model_1/basemodel/batch_normalization/batchnorm/sub�
5model_1/basemodel/batch_normalization/batchnorm/add_1AddV29model_1/basemodel/batch_normalization/batchnorm/mul_1:z:07model_1/basemodel/batch_normalization/batchnorm/sub:z:0*
T0*+
_output_shapes
:���������}@27
5model_1/basemodel/batch_normalization/batchnorm/add_1�
!model_1/basemodel/activation/ReluRelu9model_1/basemodel/batch_normalization/batchnorm/add_1:z:0*
T0*+
_output_shapes
:���������}@2#
!model_1/basemodel/activation/Relu�
*model_1/basemodel/stream_0_drop_1/IdentityIdentity/model_1/basemodel/activation/Relu:activations:0*
T0*+
_output_shapes
:���������}@2,
*model_1/basemodel/stream_0_drop_1/Identity�
7model_1/basemodel/stream_0_conv_2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������29
7model_1/basemodel/stream_0_conv_2/conv1d/ExpandDims/dim�
3model_1/basemodel/stream_0_conv_2/conv1d/ExpandDims
ExpandDims3model_1/basemodel/stream_0_drop_1/Identity:output:0@model_1/basemodel/stream_0_conv_2/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������}@25
3model_1/basemodel/stream_0_conv_2/conv1d/ExpandDims�
Dmodel_1/basemodel/stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpMmodel_1_basemodel_stream_0_conv_2_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:@�*
dtype02F
Dmodel_1/basemodel/stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOp�
9model_1/basemodel/stream_0_conv_2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2;
9model_1/basemodel/stream_0_conv_2/conv1d/ExpandDims_1/dim�
5model_1/basemodel/stream_0_conv_2/conv1d/ExpandDims_1
ExpandDimsLmodel_1/basemodel/stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOp:value:0Bmodel_1/basemodel/stream_0_conv_2/conv1d/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:@�27
5model_1/basemodel/stream_0_conv_2/conv1d/ExpandDims_1�
(model_1/basemodel/stream_0_conv_2/conv1dConv2D<model_1/basemodel/stream_0_conv_2/conv1d/ExpandDims:output:0>model_1/basemodel/stream_0_conv_2/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:���������}�*
paddingSAME*
strides
2*
(model_1/basemodel/stream_0_conv_2/conv1d�
0model_1/basemodel/stream_0_conv_2/conv1d/SqueezeSqueeze1model_1/basemodel/stream_0_conv_2/conv1d:output:0*
T0*,
_output_shapes
:���������}�*
squeeze_dims

���������22
0model_1/basemodel/stream_0_conv_2/conv1d/Squeeze�
8model_1/basemodel/stream_0_conv_2/BiasAdd/ReadVariableOpReadVariableOpAmodel_1_basemodel_stream_0_conv_2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02:
8model_1/basemodel/stream_0_conv_2/BiasAdd/ReadVariableOp�
)model_1/basemodel/stream_0_conv_2/BiasAddBiasAdd9model_1/basemodel/stream_0_conv_2/conv1d/Squeeze:output:0@model_1/basemodel/stream_0_conv_2/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������}�2+
)model_1/basemodel/stream_0_conv_2/BiasAdd�
@model_1/basemodel/batch_normalization_1/batchnorm/ReadVariableOpReadVariableOpImodel_1_basemodel_batch_normalization_1_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype02B
@model_1/basemodel/batch_normalization_1/batchnorm/ReadVariableOp�
7model_1/basemodel/batch_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:29
7model_1/basemodel/batch_normalization_1/batchnorm/add/y�
5model_1/basemodel/batch_normalization_1/batchnorm/addAddV2Hmodel_1/basemodel/batch_normalization_1/batchnorm/ReadVariableOp:value:0@model_1/basemodel/batch_normalization_1/batchnorm/add/y:output:0*
T0*
_output_shapes	
:�27
5model_1/basemodel/batch_normalization_1/batchnorm/add�
7model_1/basemodel/batch_normalization_1/batchnorm/RsqrtRsqrt9model_1/basemodel/batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes	
:�29
7model_1/basemodel/batch_normalization_1/batchnorm/Rsqrt�
Dmodel_1/basemodel/batch_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOpMmodel_1_basemodel_batch_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype02F
Dmodel_1/basemodel/batch_normalization_1/batchnorm/mul/ReadVariableOp�
5model_1/basemodel/batch_normalization_1/batchnorm/mulMul;model_1/basemodel/batch_normalization_1/batchnorm/Rsqrt:y:0Lmodel_1/basemodel/batch_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:�27
5model_1/basemodel/batch_normalization_1/batchnorm/mul�
7model_1/basemodel/batch_normalization_1/batchnorm/mul_1Mul2model_1/basemodel/stream_0_conv_2/BiasAdd:output:09model_1/basemodel/batch_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:���������}�29
7model_1/basemodel/batch_normalization_1/batchnorm/mul_1�
Bmodel_1/basemodel/batch_normalization_1/batchnorm/ReadVariableOp_1ReadVariableOpKmodel_1_basemodel_batch_normalization_1_batchnorm_readvariableop_1_resource*
_output_shapes	
:�*
dtype02D
Bmodel_1/basemodel/batch_normalization_1/batchnorm/ReadVariableOp_1�
7model_1/basemodel/batch_normalization_1/batchnorm/mul_2MulJmodel_1/basemodel/batch_normalization_1/batchnorm/ReadVariableOp_1:value:09model_1/basemodel/batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes	
:�29
7model_1/basemodel/batch_normalization_1/batchnorm/mul_2�
Bmodel_1/basemodel/batch_normalization_1/batchnorm/ReadVariableOp_2ReadVariableOpKmodel_1_basemodel_batch_normalization_1_batchnorm_readvariableop_2_resource*
_output_shapes	
:�*
dtype02D
Bmodel_1/basemodel/batch_normalization_1/batchnorm/ReadVariableOp_2�
5model_1/basemodel/batch_normalization_1/batchnorm/subSubJmodel_1/basemodel/batch_normalization_1/batchnorm/ReadVariableOp_2:value:0;model_1/basemodel/batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�27
5model_1/basemodel/batch_normalization_1/batchnorm/sub�
7model_1/basemodel/batch_normalization_1/batchnorm/add_1AddV2;model_1/basemodel/batch_normalization_1/batchnorm/mul_1:z:09model_1/basemodel/batch_normalization_1/batchnorm/sub:z:0*
T0*,
_output_shapes
:���������}�29
7model_1/basemodel/batch_normalization_1/batchnorm/add_1�
#model_1/basemodel/activation_1/ReluRelu;model_1/basemodel/batch_normalization_1/batchnorm/add_1:z:0*
T0*,
_output_shapes
:���������}�2%
#model_1/basemodel/activation_1/Relu�
*model_1/basemodel/stream_0_drop_2/IdentityIdentity1model_1/basemodel/activation_1/Relu:activations:0*
T0*,
_output_shapes
:���������}�2,
*model_1/basemodel/stream_0_drop_2/Identity�
7model_1/basemodel/stream_0_conv_3/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������29
7model_1/basemodel/stream_0_conv_3/conv1d/ExpandDims/dim�
3model_1/basemodel/stream_0_conv_3/conv1d/ExpandDims
ExpandDims3model_1/basemodel/stream_0_drop_2/Identity:output:0@model_1/basemodel/stream_0_conv_3/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:���������}�25
3model_1/basemodel/stream_0_conv_3/conv1d/ExpandDims�
Dmodel_1/basemodel/stream_0_conv_3/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpMmodel_1_basemodel_stream_0_conv_3_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:��*
dtype02F
Dmodel_1/basemodel/stream_0_conv_3/conv1d/ExpandDims_1/ReadVariableOp�
9model_1/basemodel/stream_0_conv_3/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2;
9model_1/basemodel/stream_0_conv_3/conv1d/ExpandDims_1/dim�
5model_1/basemodel/stream_0_conv_3/conv1d/ExpandDims_1
ExpandDimsLmodel_1/basemodel/stream_0_conv_3/conv1d/ExpandDims_1/ReadVariableOp:value:0Bmodel_1/basemodel/stream_0_conv_3/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:��27
5model_1/basemodel/stream_0_conv_3/conv1d/ExpandDims_1�
(model_1/basemodel/stream_0_conv_3/conv1dConv2D<model_1/basemodel/stream_0_conv_3/conv1d/ExpandDims:output:0>model_1/basemodel/stream_0_conv_3/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:���������}�*
paddingSAME*
strides
2*
(model_1/basemodel/stream_0_conv_3/conv1d�
0model_1/basemodel/stream_0_conv_3/conv1d/SqueezeSqueeze1model_1/basemodel/stream_0_conv_3/conv1d:output:0*
T0*,
_output_shapes
:���������}�*
squeeze_dims

���������22
0model_1/basemodel/stream_0_conv_3/conv1d/Squeeze�
8model_1/basemodel/stream_0_conv_3/BiasAdd/ReadVariableOpReadVariableOpAmodel_1_basemodel_stream_0_conv_3_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02:
8model_1/basemodel/stream_0_conv_3/BiasAdd/ReadVariableOp�
)model_1/basemodel/stream_0_conv_3/BiasAddBiasAdd9model_1/basemodel/stream_0_conv_3/conv1d/Squeeze:output:0@model_1/basemodel/stream_0_conv_3/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������}�2+
)model_1/basemodel/stream_0_conv_3/BiasAdd�
@model_1/basemodel/batch_normalization_2/batchnorm/ReadVariableOpReadVariableOpImodel_1_basemodel_batch_normalization_2_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype02B
@model_1/basemodel/batch_normalization_2/batchnorm/ReadVariableOp�
7model_1/basemodel/batch_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:29
7model_1/basemodel/batch_normalization_2/batchnorm/add/y�
5model_1/basemodel/batch_normalization_2/batchnorm/addAddV2Hmodel_1/basemodel/batch_normalization_2/batchnorm/ReadVariableOp:value:0@model_1/basemodel/batch_normalization_2/batchnorm/add/y:output:0*
T0*
_output_shapes	
:�27
5model_1/basemodel/batch_normalization_2/batchnorm/add�
7model_1/basemodel/batch_normalization_2/batchnorm/RsqrtRsqrt9model_1/basemodel/batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes	
:�29
7model_1/basemodel/batch_normalization_2/batchnorm/Rsqrt�
Dmodel_1/basemodel/batch_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOpMmodel_1_basemodel_batch_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype02F
Dmodel_1/basemodel/batch_normalization_2/batchnorm/mul/ReadVariableOp�
5model_1/basemodel/batch_normalization_2/batchnorm/mulMul;model_1/basemodel/batch_normalization_2/batchnorm/Rsqrt:y:0Lmodel_1/basemodel/batch_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:�27
5model_1/basemodel/batch_normalization_2/batchnorm/mul�
7model_1/basemodel/batch_normalization_2/batchnorm/mul_1Mul2model_1/basemodel/stream_0_conv_3/BiasAdd:output:09model_1/basemodel/batch_normalization_2/batchnorm/mul:z:0*
T0*,
_output_shapes
:���������}�29
7model_1/basemodel/batch_normalization_2/batchnorm/mul_1�
Bmodel_1/basemodel/batch_normalization_2/batchnorm/ReadVariableOp_1ReadVariableOpKmodel_1_basemodel_batch_normalization_2_batchnorm_readvariableop_1_resource*
_output_shapes	
:�*
dtype02D
Bmodel_1/basemodel/batch_normalization_2/batchnorm/ReadVariableOp_1�
7model_1/basemodel/batch_normalization_2/batchnorm/mul_2MulJmodel_1/basemodel/batch_normalization_2/batchnorm/ReadVariableOp_1:value:09model_1/basemodel/batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes	
:�29
7model_1/basemodel/batch_normalization_2/batchnorm/mul_2�
Bmodel_1/basemodel/batch_normalization_2/batchnorm/ReadVariableOp_2ReadVariableOpKmodel_1_basemodel_batch_normalization_2_batchnorm_readvariableop_2_resource*
_output_shapes	
:�*
dtype02D
Bmodel_1/basemodel/batch_normalization_2/batchnorm/ReadVariableOp_2�
5model_1/basemodel/batch_normalization_2/batchnorm/subSubJmodel_1/basemodel/batch_normalization_2/batchnorm/ReadVariableOp_2:value:0;model_1/basemodel/batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�27
5model_1/basemodel/batch_normalization_2/batchnorm/sub�
7model_1/basemodel/batch_normalization_2/batchnorm/add_1AddV2;model_1/basemodel/batch_normalization_2/batchnorm/mul_1:z:09model_1/basemodel/batch_normalization_2/batchnorm/sub:z:0*
T0*,
_output_shapes
:���������}�29
7model_1/basemodel/batch_normalization_2/batchnorm/add_1�
#model_1/basemodel/activation_2/ReluRelu;model_1/basemodel/batch_normalization_2/batchnorm/add_1:z:0*
T0*,
_output_shapes
:���������}�2%
#model_1/basemodel/activation_2/Relu�
*model_1/basemodel/stream_0_drop_3/IdentityIdentity1model_1/basemodel/activation_2/Relu:activations:0*
T0*,
_output_shapes
:���������}�2,
*model_1/basemodel/stream_0_drop_3/Identity�
Amodel_1/basemodel/global_average_pooling1d/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2C
Amodel_1/basemodel/global_average_pooling1d/Mean/reduction_indices�
/model_1/basemodel/global_average_pooling1d/MeanMean3model_1/basemodel/stream_0_drop_3/Identity:output:0Jmodel_1/basemodel/global_average_pooling1d/Mean/reduction_indices:output:0*
T0*(
_output_shapes
:����������21
/model_1/basemodel/global_average_pooling1d/Mean�
*model_1/basemodel/dense_1_dropout/IdentityIdentity8model_1/basemodel/global_average_pooling1d/Mean:output:0*
T0*(
_output_shapes
:����������2,
*model_1/basemodel/dense_1_dropout/Identity�
/model_1/basemodel/dense_1/MatMul/ReadVariableOpReadVariableOp8model_1_basemodel_dense_1_matmul_readvariableop_resource*
_output_shapes
:	�T*
dtype021
/model_1/basemodel/dense_1/MatMul/ReadVariableOp�
 model_1/basemodel/dense_1/MatMulMatMul3model_1/basemodel/dense_1_dropout/Identity:output:07model_1/basemodel/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������T2"
 model_1/basemodel/dense_1/MatMul�
0model_1/basemodel/dense_1/BiasAdd/ReadVariableOpReadVariableOp9model_1_basemodel_dense_1_biasadd_readvariableop_resource*
_output_shapes
:T*
dtype022
0model_1/basemodel/dense_1/BiasAdd/ReadVariableOp�
!model_1/basemodel/dense_1/BiasAddBiasAdd*model_1/basemodel/dense_1/MatMul:product:08model_1/basemodel/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������T2#
!model_1/basemodel/dense_1/BiasAdd�
@model_1/basemodel/batch_normalization_3/batchnorm/ReadVariableOpReadVariableOpImodel_1_basemodel_batch_normalization_3_batchnorm_readvariableop_resource*
_output_shapes
:T*
dtype02B
@model_1/basemodel/batch_normalization_3/batchnorm/ReadVariableOp�
7model_1/basemodel/batch_normalization_3/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:29
7model_1/basemodel/batch_normalization_3/batchnorm/add/y�
5model_1/basemodel/batch_normalization_3/batchnorm/addAddV2Hmodel_1/basemodel/batch_normalization_3/batchnorm/ReadVariableOp:value:0@model_1/basemodel/batch_normalization_3/batchnorm/add/y:output:0*
T0*
_output_shapes
:T27
5model_1/basemodel/batch_normalization_3/batchnorm/add�
7model_1/basemodel/batch_normalization_3/batchnorm/RsqrtRsqrt9model_1/basemodel/batch_normalization_3/batchnorm/add:z:0*
T0*
_output_shapes
:T29
7model_1/basemodel/batch_normalization_3/batchnorm/Rsqrt�
Dmodel_1/basemodel/batch_normalization_3/batchnorm/mul/ReadVariableOpReadVariableOpMmodel_1_basemodel_batch_normalization_3_batchnorm_mul_readvariableop_resource*
_output_shapes
:T*
dtype02F
Dmodel_1/basemodel/batch_normalization_3/batchnorm/mul/ReadVariableOp�
5model_1/basemodel/batch_normalization_3/batchnorm/mulMul;model_1/basemodel/batch_normalization_3/batchnorm/Rsqrt:y:0Lmodel_1/basemodel/batch_normalization_3/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:T27
5model_1/basemodel/batch_normalization_3/batchnorm/mul�
7model_1/basemodel/batch_normalization_3/batchnorm/mul_1Mul*model_1/basemodel/dense_1/BiasAdd:output:09model_1/basemodel/batch_normalization_3/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������T29
7model_1/basemodel/batch_normalization_3/batchnorm/mul_1�
Bmodel_1/basemodel/batch_normalization_3/batchnorm/ReadVariableOp_1ReadVariableOpKmodel_1_basemodel_batch_normalization_3_batchnorm_readvariableop_1_resource*
_output_shapes
:T*
dtype02D
Bmodel_1/basemodel/batch_normalization_3/batchnorm/ReadVariableOp_1�
7model_1/basemodel/batch_normalization_3/batchnorm/mul_2MulJmodel_1/basemodel/batch_normalization_3/batchnorm/ReadVariableOp_1:value:09model_1/basemodel/batch_normalization_3/batchnorm/mul:z:0*
T0*
_output_shapes
:T29
7model_1/basemodel/batch_normalization_3/batchnorm/mul_2�
Bmodel_1/basemodel/batch_normalization_3/batchnorm/ReadVariableOp_2ReadVariableOpKmodel_1_basemodel_batch_normalization_3_batchnorm_readvariableop_2_resource*
_output_shapes
:T*
dtype02D
Bmodel_1/basemodel/batch_normalization_3/batchnorm/ReadVariableOp_2�
5model_1/basemodel/batch_normalization_3/batchnorm/subSubJmodel_1/basemodel/batch_normalization_3/batchnorm/ReadVariableOp_2:value:0;model_1/basemodel/batch_normalization_3/batchnorm/mul_2:z:0*
T0*
_output_shapes
:T27
5model_1/basemodel/batch_normalization_3/batchnorm/sub�
7model_1/basemodel/batch_normalization_3/batchnorm/add_1AddV2;model_1/basemodel/batch_normalization_3/batchnorm/mul_1:z:09model_1/basemodel/batch_normalization_3/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������T29
7model_1/basemodel/batch_normalization_3/batchnorm/add_1�
,model_1/basemodel/dense_activation_1/SigmoidSigmoid;model_1/basemodel/batch_normalization_3/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������T2.
,model_1/basemodel/dense_activation_1/Sigmoid�
IdentityIdentity0model_1/basemodel/dense_activation_1/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:���������T2

Identity�
NoOpNoOp?^model_1/basemodel/batch_normalization/batchnorm/ReadVariableOpA^model_1/basemodel/batch_normalization/batchnorm/ReadVariableOp_1A^model_1/basemodel/batch_normalization/batchnorm/ReadVariableOp_2C^model_1/basemodel/batch_normalization/batchnorm/mul/ReadVariableOpA^model_1/basemodel/batch_normalization_1/batchnorm/ReadVariableOpC^model_1/basemodel/batch_normalization_1/batchnorm/ReadVariableOp_1C^model_1/basemodel/batch_normalization_1/batchnorm/ReadVariableOp_2E^model_1/basemodel/batch_normalization_1/batchnorm/mul/ReadVariableOpA^model_1/basemodel/batch_normalization_2/batchnorm/ReadVariableOpC^model_1/basemodel/batch_normalization_2/batchnorm/ReadVariableOp_1C^model_1/basemodel/batch_normalization_2/batchnorm/ReadVariableOp_2E^model_1/basemodel/batch_normalization_2/batchnorm/mul/ReadVariableOpA^model_1/basemodel/batch_normalization_3/batchnorm/ReadVariableOpC^model_1/basemodel/batch_normalization_3/batchnorm/ReadVariableOp_1C^model_1/basemodel/batch_normalization_3/batchnorm/ReadVariableOp_2E^model_1/basemodel/batch_normalization_3/batchnorm/mul/ReadVariableOp1^model_1/basemodel/dense_1/BiasAdd/ReadVariableOp0^model_1/basemodel/dense_1/MatMul/ReadVariableOp9^model_1/basemodel/stream_0_conv_1/BiasAdd/ReadVariableOpE^model_1/basemodel/stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp9^model_1/basemodel/stream_0_conv_2/BiasAdd/ReadVariableOpE^model_1/basemodel/stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOp9^model_1/basemodel/stream_0_conv_3/BiasAdd/ReadVariableOpE^model_1/basemodel/stream_0_conv_3/conv1d/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:���������}: : : : : : : : : : : : : : : : : : : : : : : : 2�
>model_1/basemodel/batch_normalization/batchnorm/ReadVariableOp>model_1/basemodel/batch_normalization/batchnorm/ReadVariableOp2�
@model_1/basemodel/batch_normalization/batchnorm/ReadVariableOp_1@model_1/basemodel/batch_normalization/batchnorm/ReadVariableOp_12�
@model_1/basemodel/batch_normalization/batchnorm/ReadVariableOp_2@model_1/basemodel/batch_normalization/batchnorm/ReadVariableOp_22�
Bmodel_1/basemodel/batch_normalization/batchnorm/mul/ReadVariableOpBmodel_1/basemodel/batch_normalization/batchnorm/mul/ReadVariableOp2�
@model_1/basemodel/batch_normalization_1/batchnorm/ReadVariableOp@model_1/basemodel/batch_normalization_1/batchnorm/ReadVariableOp2�
Bmodel_1/basemodel/batch_normalization_1/batchnorm/ReadVariableOp_1Bmodel_1/basemodel/batch_normalization_1/batchnorm/ReadVariableOp_12�
Bmodel_1/basemodel/batch_normalization_1/batchnorm/ReadVariableOp_2Bmodel_1/basemodel/batch_normalization_1/batchnorm/ReadVariableOp_22�
Dmodel_1/basemodel/batch_normalization_1/batchnorm/mul/ReadVariableOpDmodel_1/basemodel/batch_normalization_1/batchnorm/mul/ReadVariableOp2�
@model_1/basemodel/batch_normalization_2/batchnorm/ReadVariableOp@model_1/basemodel/batch_normalization_2/batchnorm/ReadVariableOp2�
Bmodel_1/basemodel/batch_normalization_2/batchnorm/ReadVariableOp_1Bmodel_1/basemodel/batch_normalization_2/batchnorm/ReadVariableOp_12�
Bmodel_1/basemodel/batch_normalization_2/batchnorm/ReadVariableOp_2Bmodel_1/basemodel/batch_normalization_2/batchnorm/ReadVariableOp_22�
Dmodel_1/basemodel/batch_normalization_2/batchnorm/mul/ReadVariableOpDmodel_1/basemodel/batch_normalization_2/batchnorm/mul/ReadVariableOp2�
@model_1/basemodel/batch_normalization_3/batchnorm/ReadVariableOp@model_1/basemodel/batch_normalization_3/batchnorm/ReadVariableOp2�
Bmodel_1/basemodel/batch_normalization_3/batchnorm/ReadVariableOp_1Bmodel_1/basemodel/batch_normalization_3/batchnorm/ReadVariableOp_12�
Bmodel_1/basemodel/batch_normalization_3/batchnorm/ReadVariableOp_2Bmodel_1/basemodel/batch_normalization_3/batchnorm/ReadVariableOp_22�
Dmodel_1/basemodel/batch_normalization_3/batchnorm/mul/ReadVariableOpDmodel_1/basemodel/batch_normalization_3/batchnorm/mul/ReadVariableOp2d
0model_1/basemodel/dense_1/BiasAdd/ReadVariableOp0model_1/basemodel/dense_1/BiasAdd/ReadVariableOp2b
/model_1/basemodel/dense_1/MatMul/ReadVariableOp/model_1/basemodel/dense_1/MatMul/ReadVariableOp2t
8model_1/basemodel/stream_0_conv_1/BiasAdd/ReadVariableOp8model_1/basemodel/stream_0_conv_1/BiasAdd/ReadVariableOp2�
Dmodel_1/basemodel/stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOpDmodel_1/basemodel/stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp2t
8model_1/basemodel/stream_0_conv_2/BiasAdd/ReadVariableOp8model_1/basemodel/stream_0_conv_2/BiasAdd/ReadVariableOp2�
Dmodel_1/basemodel/stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOpDmodel_1/basemodel/stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOp2t
8model_1/basemodel/stream_0_conv_3/BiasAdd/ReadVariableOp8model_1/basemodel/stream_0_conv_3/BiasAdd/ReadVariableOp2�
Dmodel_1/basemodel/stream_0_conv_3/conv1d/ExpandDims_1/ReadVariableOpDmodel_1/basemodel/stream_0_conv_3/conv1d/ExpandDims_1/ReadVariableOp:X T
+
_output_shapes
:���������}
%
_user_specified_nameleft_inputs
�
k
L__inference_stream_0_drop_3_layer_call_and_return_conditional_losses_8911431

inputs
identity�c
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
:���������}�2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:���������}�*
dtype0*
seed�*
seed2�2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/y�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:���������}�2
dropout/GreaterEqual�
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:���������}�2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:���������}�2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:���������}�2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:���������}�:T P
,
_output_shapes
:���������}�
 
_user_specified_nameinputs
�
�
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_8907129

inputs0
!batchnorm_readvariableop_resource:	�4
%batchnorm_mul_readvariableop_resource:	�2
#batchnorm_readvariableop_1_resource:	�2
#batchnorm_readvariableop_2_resource:	�
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOp�
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:�*
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
T0*
_output_shapes	
:�2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:�2
batchnorm/Rsqrt�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype02
batchnorm/mul/ReadVariableOp�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2
batchnorm/mul{
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*,
_output_shapes
:���������}�2
batchnorm/mul_1�
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:�*
dtype02
batchnorm/ReadVariableOp_1�
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:�2
batchnorm/mul_2�
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:�*
dtype02
batchnorm/ReadVariableOp_2�
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*,
_output_shapes
:���������}�2
batchnorm/add_1s
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*,
_output_shapes
:���������}�2

Identity�
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������}�: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:T P
,
_output_shapes
:���������}�
 
_user_specified_nameinputs
�
�
7__inference_batch_normalization_3_layer_call_fn_8911524

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
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_89068612
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
�
�
1__inference_stream_0_conv_1_layer_call_fn_8910747

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
 *+
_output_shapes
:���������}@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_stream_0_conv_1_layer_call_and_return_conditional_losses_89070342
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������}@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������}: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������}
 
_user_specified_nameinputs
�
�
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_8911127

inputs0
!batchnorm_readvariableop_resource:	�4
%batchnorm_mul_readvariableop_resource:	�2
#batchnorm_readvariableop_1_resource:	�2
#batchnorm_readvariableop_2_resource:	�
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOp�
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:�*
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
T0*
_output_shapes	
:�2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:�2
batchnorm/Rsqrt�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype02
batchnorm/mul/ReadVariableOp�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2
batchnorm/mul{
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*,
_output_shapes
:���������}�2
batchnorm/mul_1�
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:�*
dtype02
batchnorm/ReadVariableOp_1�
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:�2
batchnorm/mul_2�
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:�*
dtype02
batchnorm/ReadVariableOp_2�
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*,
_output_shapes
:���������}�2
batchnorm/add_1s
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*,
_output_shapes
:���������}�2

Identity�
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������}�: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:T P
,
_output_shapes
:���������}�
 
_user_specified_nameinputs
�*
�
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_8906921

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
k
O__inference_dense_activation_1_layer_call_and_return_conditional_losses_8911601

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
�=
�	
D__inference_model_1_layer_call_and_return_conditional_losses_8908379

inputs'
basemodel_8908305:@
basemodel_8908307:@
basemodel_8908309:@
basemodel_8908311:@
basemodel_8908313:@
basemodel_8908315:@(
basemodel_8908317:@� 
basemodel_8908319:	� 
basemodel_8908321:	� 
basemodel_8908323:	� 
basemodel_8908325:	� 
basemodel_8908327:	�)
basemodel_8908329:�� 
basemodel_8908331:	� 
basemodel_8908333:	� 
basemodel_8908335:	� 
basemodel_8908337:	� 
basemodel_8908339:	�$
basemodel_8908341:	�T
basemodel_8908343:T
basemodel_8908345:T
basemodel_8908347:T
basemodel_8908349:T
basemodel_8908351:T
identity��!basemodel/StatefulPartitionedCall�-dense_1/kernel/Regularizer/Abs/ReadVariableOp�5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp�8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp�5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp�
!basemodel/StatefulPartitionedCallStatefulPartitionedCallinputsbasemodel_8908305basemodel_8908307basemodel_8908309basemodel_8908311basemodel_8908313basemodel_8908315basemodel_8908317basemodel_8908319basemodel_8908321basemodel_8908323basemodel_8908325basemodel_8908327basemodel_8908329basemodel_8908331basemodel_8908333basemodel_8908335basemodel_8908337basemodel_8908339basemodel_8908341basemodel_8908343basemodel_8908345basemodel_8908347basemodel_8908349basemodel_8908351*$
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������T*:
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_basemodel_layer_call_and_return_conditional_losses_89083042#
!basemodel/StatefulPartitionedCall�
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpbasemodel_8908305*"
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
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpbasemodel_8908317*#
_output_shapes
:@�*
dtype02:
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp�
)stream_0_conv_2/kernel/Regularizer/SquareSquare@stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*#
_output_shapes
:@�2+
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
5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpbasemodel_8908329*$
_output_shapes
:��*
dtype027
5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp�
&stream_0_conv_3/kernel/Regularizer/AbsAbs=stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*$
_output_shapes
:��2(
&stream_0_conv_3/kernel/Regularizer/Abs�
(stream_0_conv_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_0_conv_3/kernel/Regularizer/Const�
&stream_0_conv_3/kernel/Regularizer/SumSum*stream_0_conv_3/kernel/Regularizer/Abs:y:01stream_0_conv_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_3/kernel/Regularizer/Sum�
(stream_0_conv_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2*
(stream_0_conv_3/kernel/Regularizer/mul/x�
&stream_0_conv_3/kernel/Regularizer/mulMul1stream_0_conv_3/kernel/Regularizer/mul/x:output:0/stream_0_conv_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_3/kernel/Regularizer/mul�
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpbasemodel_8908341*
_output_shapes
:	�T*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOp�
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	�T2 
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
IdentityIdentity*basemodel/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������T2

Identity�
NoOpNoOp"^basemodel/StatefulPartitionedCall.^dense_1/kernel/Regularizer/Abs/ReadVariableOp6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp9^stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp6^stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:���������}: : : : : : : : : : : : : : : : : : : : : : : : 2F
!basemodel/StatefulPartitionedCall!basemodel/StatefulPartitionedCall2^
-dense_1/kernel/Regularizer/Abs/ReadVariableOp-dense_1/kernel/Regularizer/Abs/ReadVariableOp2n
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp2t
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp2n
5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp:S O
+
_output_shapes
:���������}
 
_user_specified_nameinputs
�
k
L__inference_dense_1_dropout_layer_call_and_return_conditional_losses_8907387

inputs
identity�c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:����������2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:����������*
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
T0*(
_output_shapes
:����������2
dropout/GreaterEqual�
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:����������2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:����������2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
��
�
F__inference_basemodel_layer_call_and_return_conditional_losses_8910097

inputsQ
;stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource:@=
/stream_0_conv_1_biasadd_readvariableop_resource:@C
5batch_normalization_batchnorm_readvariableop_resource:@G
9batch_normalization_batchnorm_mul_readvariableop_resource:@E
7batch_normalization_batchnorm_readvariableop_1_resource:@E
7batch_normalization_batchnorm_readvariableop_2_resource:@R
;stream_0_conv_2_conv1d_expanddims_1_readvariableop_resource:@�>
/stream_0_conv_2_biasadd_readvariableop_resource:	�F
7batch_normalization_1_batchnorm_readvariableop_resource:	�J
;batch_normalization_1_batchnorm_mul_readvariableop_resource:	�H
9batch_normalization_1_batchnorm_readvariableop_1_resource:	�H
9batch_normalization_1_batchnorm_readvariableop_2_resource:	�S
;stream_0_conv_3_conv1d_expanddims_1_readvariableop_resource:��>
/stream_0_conv_3_biasadd_readvariableop_resource:	�F
7batch_normalization_2_batchnorm_readvariableop_resource:	�J
;batch_normalization_2_batchnorm_mul_readvariableop_resource:	�H
9batch_normalization_2_batchnorm_readvariableop_1_resource:	�H
9batch_normalization_2_batchnorm_readvariableop_2_resource:	�9
&dense_1_matmul_readvariableop_resource:	�T5
'dense_1_biasadd_readvariableop_resource:TE
7batch_normalization_3_batchnorm_readvariableop_resource:TI
;batch_normalization_3_batchnorm_mul_readvariableop_resource:TG
9batch_normalization_3_batchnorm_readvariableop_1_resource:TG
9batch_normalization_3_batchnorm_readvariableop_2_resource:T
identity��,batch_normalization/batchnorm/ReadVariableOp�.batch_normalization/batchnorm/ReadVariableOp_1�.batch_normalization/batchnorm/ReadVariableOp_2�0batch_normalization/batchnorm/mul/ReadVariableOp�.batch_normalization_1/batchnorm/ReadVariableOp�0batch_normalization_1/batchnorm/ReadVariableOp_1�0batch_normalization_1/batchnorm/ReadVariableOp_2�2batch_normalization_1/batchnorm/mul/ReadVariableOp�.batch_normalization_2/batchnorm/ReadVariableOp�0batch_normalization_2/batchnorm/ReadVariableOp_1�0batch_normalization_2/batchnorm/ReadVariableOp_2�2batch_normalization_2/batchnorm/mul/ReadVariableOp�.batch_normalization_3/batchnorm/ReadVariableOp�0batch_normalization_3/batchnorm/ReadVariableOp_1�0batch_normalization_3/batchnorm/ReadVariableOp_2�2batch_normalization_3/batchnorm/mul/ReadVariableOp�dense_1/BiasAdd/ReadVariableOp�dense_1/MatMul/ReadVariableOp�-dense_1/kernel/Regularizer/Abs/ReadVariableOp�&stream_0_conv_1/BiasAdd/ReadVariableOp�2stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp�5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp�&stream_0_conv_2/BiasAdd/ReadVariableOp�2stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOp�8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp�&stream_0_conv_3/BiasAdd/ReadVariableOp�2stream_0_conv_3/conv1d/ExpandDims_1/ReadVariableOp�5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp�
stream_0_input_drop/IdentityIdentityinputs*
T0*+
_output_shapes
:���������}2
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
T0*/
_output_shapes
:���������}2#
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
T0*/
_output_shapes
:���������}@*
paddingSAME*
strides
2
stream_0_conv_1/conv1d�
stream_0_conv_1/conv1d/SqueezeSqueezestream_0_conv_1/conv1d:output:0*
T0*+
_output_shapes
:���������}@*
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
T0*+
_output_shapes
:���������}@2
stream_0_conv_1/BiasAdd�
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
T0*+
_output_shapes
:���������}@2%
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
T0*+
_output_shapes
:���������}@2%
#batch_normalization/batchnorm/add_1�
activation/ReluRelu'batch_normalization/batchnorm/add_1:z:0*
T0*+
_output_shapes
:���������}@2
activation/Relu�
stream_0_drop_1/IdentityIdentityactivation/Relu:activations:0*
T0*+
_output_shapes
:���������}@2
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
T0*/
_output_shapes
:���������}@2#
!stream_0_conv_2/conv1d/ExpandDims�
2stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp;stream_0_conv_2_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:@�*
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
T0*'
_output_shapes
:@�2%
#stream_0_conv_2/conv1d/ExpandDims_1�
stream_0_conv_2/conv1dConv2D*stream_0_conv_2/conv1d/ExpandDims:output:0,stream_0_conv_2/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:���������}�*
paddingSAME*
strides
2
stream_0_conv_2/conv1d�
stream_0_conv_2/conv1d/SqueezeSqueezestream_0_conv_2/conv1d:output:0*
T0*,
_output_shapes
:���������}�*
squeeze_dims

���������2 
stream_0_conv_2/conv1d/Squeeze�
&stream_0_conv_2/BiasAdd/ReadVariableOpReadVariableOp/stream_0_conv_2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02(
&stream_0_conv_2/BiasAdd/ReadVariableOp�
stream_0_conv_2/BiasAddBiasAdd'stream_0_conv_2/conv1d/Squeeze:output:0.stream_0_conv_2/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������}�2
stream_0_conv_2/BiasAdd�
.batch_normalization_1/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_1_batchnorm_readvariableop_resource*
_output_shapes	
:�*
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
T0*
_output_shapes	
:�2%
#batch_normalization_1/batchnorm/add�
%batch_normalization_1/batchnorm/RsqrtRsqrt'batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes	
:�2'
%batch_normalization_1/batchnorm/Rsqrt�
2batch_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype024
2batch_normalization_1/batchnorm/mul/ReadVariableOp�
#batch_normalization_1/batchnorm/mulMul)batch_normalization_1/batchnorm/Rsqrt:y:0:batch_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2%
#batch_normalization_1/batchnorm/mul�
%batch_normalization_1/batchnorm/mul_1Mul stream_0_conv_2/BiasAdd:output:0'batch_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:���������}�2'
%batch_normalization_1/batchnorm/mul_1�
0batch_normalization_1/batchnorm/ReadVariableOp_1ReadVariableOp9batch_normalization_1_batchnorm_readvariableop_1_resource*
_output_shapes	
:�*
dtype022
0batch_normalization_1/batchnorm/ReadVariableOp_1�
%batch_normalization_1/batchnorm/mul_2Mul8batch_normalization_1/batchnorm/ReadVariableOp_1:value:0'batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes	
:�2'
%batch_normalization_1/batchnorm/mul_2�
0batch_normalization_1/batchnorm/ReadVariableOp_2ReadVariableOp9batch_normalization_1_batchnorm_readvariableop_2_resource*
_output_shapes	
:�*
dtype022
0batch_normalization_1/batchnorm/ReadVariableOp_2�
#batch_normalization_1/batchnorm/subSub8batch_normalization_1/batchnorm/ReadVariableOp_2:value:0)batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�2%
#batch_normalization_1/batchnorm/sub�
%batch_normalization_1/batchnorm/add_1AddV2)batch_normalization_1/batchnorm/mul_1:z:0'batch_normalization_1/batchnorm/sub:z:0*
T0*,
_output_shapes
:���������}�2'
%batch_normalization_1/batchnorm/add_1�
activation_1/ReluRelu)batch_normalization_1/batchnorm/add_1:z:0*
T0*,
_output_shapes
:���������}�2
activation_1/Relu�
stream_0_drop_2/IdentityIdentityactivation_1/Relu:activations:0*
T0*,
_output_shapes
:���������}�2
stream_0_drop_2/Identity�
%stream_0_conv_3/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2'
%stream_0_conv_3/conv1d/ExpandDims/dim�
!stream_0_conv_3/conv1d/ExpandDims
ExpandDims!stream_0_drop_2/Identity:output:0.stream_0_conv_3/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:���������}�2#
!stream_0_conv_3/conv1d/ExpandDims�
2stream_0_conv_3/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp;stream_0_conv_3_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:��*
dtype024
2stream_0_conv_3/conv1d/ExpandDims_1/ReadVariableOp�
'stream_0_conv_3/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2)
'stream_0_conv_3/conv1d/ExpandDims_1/dim�
#stream_0_conv_3/conv1d/ExpandDims_1
ExpandDims:stream_0_conv_3/conv1d/ExpandDims_1/ReadVariableOp:value:00stream_0_conv_3/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:��2%
#stream_0_conv_3/conv1d/ExpandDims_1�
stream_0_conv_3/conv1dConv2D*stream_0_conv_3/conv1d/ExpandDims:output:0,stream_0_conv_3/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:���������}�*
paddingSAME*
strides
2
stream_0_conv_3/conv1d�
stream_0_conv_3/conv1d/SqueezeSqueezestream_0_conv_3/conv1d:output:0*
T0*,
_output_shapes
:���������}�*
squeeze_dims

���������2 
stream_0_conv_3/conv1d/Squeeze�
&stream_0_conv_3/BiasAdd/ReadVariableOpReadVariableOp/stream_0_conv_3_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02(
&stream_0_conv_3/BiasAdd/ReadVariableOp�
stream_0_conv_3/BiasAddBiasAdd'stream_0_conv_3/conv1d/Squeeze:output:0.stream_0_conv_3/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������}�2
stream_0_conv_3/BiasAdd�
.batch_normalization_2/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_2_batchnorm_readvariableop_resource*
_output_shapes	
:�*
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
T0*
_output_shapes	
:�2%
#batch_normalization_2/batchnorm/add�
%batch_normalization_2/batchnorm/RsqrtRsqrt'batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes	
:�2'
%batch_normalization_2/batchnorm/Rsqrt�
2batch_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype024
2batch_normalization_2/batchnorm/mul/ReadVariableOp�
#batch_normalization_2/batchnorm/mulMul)batch_normalization_2/batchnorm/Rsqrt:y:0:batch_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2%
#batch_normalization_2/batchnorm/mul�
%batch_normalization_2/batchnorm/mul_1Mul stream_0_conv_3/BiasAdd:output:0'batch_normalization_2/batchnorm/mul:z:0*
T0*,
_output_shapes
:���������}�2'
%batch_normalization_2/batchnorm/mul_1�
0batch_normalization_2/batchnorm/ReadVariableOp_1ReadVariableOp9batch_normalization_2_batchnorm_readvariableop_1_resource*
_output_shapes	
:�*
dtype022
0batch_normalization_2/batchnorm/ReadVariableOp_1�
%batch_normalization_2/batchnorm/mul_2Mul8batch_normalization_2/batchnorm/ReadVariableOp_1:value:0'batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes	
:�2'
%batch_normalization_2/batchnorm/mul_2�
0batch_normalization_2/batchnorm/ReadVariableOp_2ReadVariableOp9batch_normalization_2_batchnorm_readvariableop_2_resource*
_output_shapes	
:�*
dtype022
0batch_normalization_2/batchnorm/ReadVariableOp_2�
#batch_normalization_2/batchnorm/subSub8batch_normalization_2/batchnorm/ReadVariableOp_2:value:0)batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�2%
#batch_normalization_2/batchnorm/sub�
%batch_normalization_2/batchnorm/add_1AddV2)batch_normalization_2/batchnorm/mul_1:z:0'batch_normalization_2/batchnorm/sub:z:0*
T0*,
_output_shapes
:���������}�2'
%batch_normalization_2/batchnorm/add_1�
activation_2/ReluRelu)batch_normalization_2/batchnorm/add_1:z:0*
T0*,
_output_shapes
:���������}�2
activation_2/Relu�
stream_0_drop_3/IdentityIdentityactivation_2/Relu:activations:0*
T0*,
_output_shapes
:���������}�2
stream_0_drop_3/Identity�
/global_average_pooling1d/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :21
/global_average_pooling1d/Mean/reduction_indices�
global_average_pooling1d/MeanMean!stream_0_drop_3/Identity:output:08global_average_pooling1d/Mean/reduction_indices:output:0*
T0*(
_output_shapes
:����������2
global_average_pooling1d/Mean�
dense_1_dropout/IdentityIdentity&global_average_pooling1d/Mean:output:0*
T0*(
_output_shapes
:����������2
dense_1_dropout/Identity�
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	�T*
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
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;stream_0_conv_2_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:@�*
dtype02:
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp�
)stream_0_conv_2/kernel/Regularizer/SquareSquare@stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*#
_output_shapes
:@�2+
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
5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp;stream_0_conv_3_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:��*
dtype027
5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp�
&stream_0_conv_3/kernel/Regularizer/AbsAbs=stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*$
_output_shapes
:��2(
&stream_0_conv_3/kernel/Regularizer/Abs�
(stream_0_conv_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_0_conv_3/kernel/Regularizer/Const�
&stream_0_conv_3/kernel/Regularizer/SumSum*stream_0_conv_3/kernel/Regularizer/Abs:y:01stream_0_conv_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_3/kernel/Regularizer/Sum�
(stream_0_conv_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2*
(stream_0_conv_3/kernel/Regularizer/mul/x�
&stream_0_conv_3/kernel/Regularizer/mulMul1stream_0_conv_3/kernel/Regularizer/mul/x:output:0/stream_0_conv_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_3/kernel/Regularizer/mul�
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	�T*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOp�
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	�T2 
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
NoOpNoOp-^batch_normalization/batchnorm/ReadVariableOp/^batch_normalization/batchnorm/ReadVariableOp_1/^batch_normalization/batchnorm/ReadVariableOp_21^batch_normalization/batchnorm/mul/ReadVariableOp/^batch_normalization_1/batchnorm/ReadVariableOp1^batch_normalization_1/batchnorm/ReadVariableOp_11^batch_normalization_1/batchnorm/ReadVariableOp_23^batch_normalization_1/batchnorm/mul/ReadVariableOp/^batch_normalization_2/batchnorm/ReadVariableOp1^batch_normalization_2/batchnorm/ReadVariableOp_11^batch_normalization_2/batchnorm/ReadVariableOp_23^batch_normalization_2/batchnorm/mul/ReadVariableOp/^batch_normalization_3/batchnorm/ReadVariableOp1^batch_normalization_3/batchnorm/ReadVariableOp_11^batch_normalization_3/batchnorm/ReadVariableOp_23^batch_normalization_3/batchnorm/mul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp.^dense_1/kernel/Regularizer/Abs/ReadVariableOp'^stream_0_conv_1/BiasAdd/ReadVariableOp3^stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp'^stream_0_conv_2/BiasAdd/ReadVariableOp3^stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOp9^stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp'^stream_0_conv_3/BiasAdd/ReadVariableOp3^stream_0_conv_3/conv1d/ExpandDims_1/ReadVariableOp6^stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:���������}: : : : : : : : : : : : : : : : : : : : : : : : 2\
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
:���������}
 
_user_specified_nameinputs
�
q
U__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_8907228

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
:����������2
Meanb
IdentityIdentityMean:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:���������}�:T P
,
_output_shapes
:���������}�
 
_user_specified_nameinputs
�
H
,__inference_activation_layer_call_fn_8910933

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������}@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_activation_layer_call_and_return_conditional_losses_89070742
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:���������}@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������}@:S O
+
_output_shapes
:���������}@
 
_user_specified_nameinputs
�
j
1__inference_stream_0_drop_3_layer_call_fn_8911414

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
:���������}�* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_stream_0_drop_3_layer_call_and_return_conditional_losses_89074152
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:���������}�2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:���������}�22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:���������}�
 
_user_specified_nameinputs
�
k
O__inference_dense_activation_1_layer_call_and_return_conditional_losses_8907273

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
�	
�
7__inference_batch_normalization_2_layer_call_fn_8911260

inputs
unknown:	�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:�������������������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *[
fVRT
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_89067352
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:�������������������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):�������������������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:�������������������
 
_user_specified_nameinputs
�
M
1__inference_dense_1_dropout_layer_call_fn_8911458

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
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_dense_1_dropout_layer_call_and_return_conditional_losses_89072352
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
+__inference_basemodel_layer_call_fn_8909902
inputs_0
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:@ 
	unknown_5:@�
	unknown_6:	�
	unknown_7:	�
	unknown_8:	�
	unknown_9:	�

unknown_10:	�"

unknown_11:��

unknown_12:	�

unknown_13:	�

unknown_14:	�

unknown_15:	�

unknown_16:	�

unknown_17:	�T

unknown_18:T

unknown_19:T

unknown_20:T

unknown_21:T

unknown_22:T
identity��StatefulPartitionedCall�
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
:���������T*:
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_basemodel_layer_call_and_return_conditional_losses_89083042
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
_construction_contextkEagerRuntime*Z
_input_shapesI
G:���������}: : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
+
_output_shapes
:���������}
"
_user_specified_name
inputs/0
�
�
P__inference_batch_normalization_layer_call_and_return_conditional_losses_8906351

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
�=
�	
D__inference_model_1_layer_call_and_return_conditional_losses_8909159
left_inputs'
basemodel_8909085:@
basemodel_8909087:@
basemodel_8909089:@
basemodel_8909091:@
basemodel_8909093:@
basemodel_8909095:@(
basemodel_8909097:@� 
basemodel_8909099:	� 
basemodel_8909101:	� 
basemodel_8909103:	� 
basemodel_8909105:	� 
basemodel_8909107:	�)
basemodel_8909109:�� 
basemodel_8909111:	� 
basemodel_8909113:	� 
basemodel_8909115:	� 
basemodel_8909117:	� 
basemodel_8909119:	�$
basemodel_8909121:	�T
basemodel_8909123:T
basemodel_8909125:T
basemodel_8909127:T
basemodel_8909129:T
basemodel_8909131:T
identity��!basemodel/StatefulPartitionedCall�-dense_1/kernel/Regularizer/Abs/ReadVariableOp�5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp�8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp�5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp�
!basemodel/StatefulPartitionedCallStatefulPartitionedCallleft_inputsbasemodel_8909085basemodel_8909087basemodel_8909089basemodel_8909091basemodel_8909093basemodel_8909095basemodel_8909097basemodel_8909099basemodel_8909101basemodel_8909103basemodel_8909105basemodel_8909107basemodel_8909109basemodel_8909111basemodel_8909113basemodel_8909115basemodel_8909117basemodel_8909119basemodel_8909121basemodel_8909123basemodel_8909125basemodel_8909127basemodel_8909129basemodel_8909131*$
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������T*2
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_basemodel_layer_call_and_return_conditional_losses_89087182#
!basemodel/StatefulPartitionedCall�
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpbasemodel_8909085*"
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
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpbasemodel_8909097*#
_output_shapes
:@�*
dtype02:
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp�
)stream_0_conv_2/kernel/Regularizer/SquareSquare@stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*#
_output_shapes
:@�2+
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
5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpbasemodel_8909109*$
_output_shapes
:��*
dtype027
5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp�
&stream_0_conv_3/kernel/Regularizer/AbsAbs=stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*$
_output_shapes
:��2(
&stream_0_conv_3/kernel/Regularizer/Abs�
(stream_0_conv_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_0_conv_3/kernel/Regularizer/Const�
&stream_0_conv_3/kernel/Regularizer/SumSum*stream_0_conv_3/kernel/Regularizer/Abs:y:01stream_0_conv_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_3/kernel/Regularizer/Sum�
(stream_0_conv_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2*
(stream_0_conv_3/kernel/Regularizer/mul/x�
&stream_0_conv_3/kernel/Regularizer/mulMul1stream_0_conv_3/kernel/Regularizer/mul/x:output:0/stream_0_conv_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_3/kernel/Regularizer/mul�
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpbasemodel_8909121*
_output_shapes
:	�T*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOp�
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	�T2 
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
IdentityIdentity*basemodel/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������T2

Identity�
NoOpNoOp"^basemodel/StatefulPartitionedCall.^dense_1/kernel/Regularizer/Abs/ReadVariableOp6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp9^stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp6^stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:���������}: : : : : : : : : : : : : : : : : : : : : : : : 2F
!basemodel/StatefulPartitionedCall!basemodel/StatefulPartitionedCall2^
-dense_1/kernel/Regularizer/Abs/ReadVariableOp-dense_1/kernel/Regularizer/Abs/ReadVariableOp2n
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp2t
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp2n
5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp:X T
+
_output_shapes
:���������}
%
_user_specified_nameleft_inputs
�
�
__inference_loss_fn_2_8911634V
>stream_0_conv_3_kernel_regularizer_abs_readvariableop_resource:��
identity��5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp�
5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp>stream_0_conv_3_kernel_regularizer_abs_readvariableop_resource*$
_output_shapes
:��*
dtype027
5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp�
&stream_0_conv_3/kernel/Regularizer/AbsAbs=stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*$
_output_shapes
:��2(
&stream_0_conv_3/kernel/Regularizer/Abs�
(stream_0_conv_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_0_conv_3/kernel/Regularizer/Const�
&stream_0_conv_3/kernel/Regularizer/SumSum*stream_0_conv_3/kernel/Regularizer/Abs:y:01stream_0_conv_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_3/kernel/Regularizer/Sum�
(stream_0_conv_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2*
(stream_0_conv_3/kernel/Regularizer/mul/x�
&stream_0_conv_3/kernel/Regularizer/mulMul1stream_0_conv_3/kernel/Regularizer/mul/x:output:0/stream_0_conv_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_3/kernel/Regularizer/mult
IdentityIdentity*stream_0_conv_3/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: 2

Identity�
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
�
�
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_8906675

inputs0
!batchnorm_readvariableop_resource:	�4
%batchnorm_mul_readvariableop_resource:	�2
#batchnorm_readvariableop_1_resource:	�2
#batchnorm_readvariableop_2_resource:	�
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOp�
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:�*
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
T0*
_output_shapes	
:�2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:�2
batchnorm/Rsqrt�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype02
batchnorm/mul/ReadVariableOp�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2
batchnorm/mul�
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*5
_output_shapes#
!:�������������������2
batchnorm/mul_1�
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:�*
dtype02
batchnorm/ReadVariableOp_1�
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:�2
batchnorm/mul_2�
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:�*
dtype02
batchnorm/ReadVariableOp_2�
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*5
_output_shapes#
!:�������������������2
batchnorm/add_1|
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*5
_output_shapes#
!:�������������������2

Identity�
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):�������������������: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:] Y
5
_output_shapes#
!:�������������������
 
_user_specified_nameinputs
�
�
__inference_loss_fn_3_8911645I
6dense_1_kernel_regularizer_abs_readvariableop_resource:	�T
identity��-dense_1/kernel/Regularizer/Abs/ReadVariableOp�
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp6dense_1_kernel_regularizer_abs_readvariableop_resource*
_output_shapes
:	�T*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOp�
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	�T2 
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
�
�
7__inference_batch_normalization_2_layer_call_fn_8911273

inputs
unknown:	�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������}�*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *[
fVRT
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_89071992
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:���������}�2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������}�: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:���������}�
 
_user_specified_nameinputs
�
�
5__inference_batch_normalization_layer_call_fn_8910820

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
 *+
_output_shapes
:���������}@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_89076712
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������}@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������}@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������}@
 
_user_specified_nameinputs
�+
�
P__inference_batch_normalization_layer_call_and_return_conditional_losses_8907671

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
T0*+
_output_shapes
:���������}@2
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
batchnorm/mulz
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:���������}@2
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
T0*+
_output_shapes
:���������}@2
batchnorm/add_1r
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*+
_output_shapes
:���������}@2

Identity�
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������}@: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:S O
+
_output_shapes
:���������}@
 
_user_specified_nameinputs
�
o
P__inference_stream_0_input_drop_layer_call_and_return_conditional_losses_8907712

inputs
identity�c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
dropout/Constw
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:���������}2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:���������}*
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
T0*+
_output_shapes
:���������}2
dropout/GreaterEqual�
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:���������}2
dropout/Cast~
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:���������}2
dropout/Mul_1i
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:���������}2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������}:S O
+
_output_shapes
:���������}
 
_user_specified_nameinputs
�
�
7__inference_batch_normalization_3_layer_call_fn_8911537

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
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_89069212
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
��
�
F__inference_basemodel_layer_call_and_return_conditional_losses_8907300

inputs-
stream_0_conv_1_8907035:@%
stream_0_conv_1_8907037:@)
batch_normalization_8907060:@)
batch_normalization_8907062:@)
batch_normalization_8907064:@)
batch_normalization_8907066:@.
stream_0_conv_2_8907105:@�&
stream_0_conv_2_8907107:	�,
batch_normalization_1_8907130:	�,
batch_normalization_1_8907132:	�,
batch_normalization_1_8907134:	�,
batch_normalization_1_8907136:	�/
stream_0_conv_3_8907175:��&
stream_0_conv_3_8907177:	�,
batch_normalization_2_8907200:	�,
batch_normalization_2_8907202:	�,
batch_normalization_2_8907204:	�,
batch_normalization_2_8907206:	�"
dense_1_8907254:	�T
dense_1_8907256:T+
batch_normalization_3_8907259:T+
batch_normalization_3_8907261:T+
batch_normalization_3_8907263:T+
batch_normalization_3_8907265:T
identity��+batch_normalization/StatefulPartitionedCall�-batch_normalization_1/StatefulPartitionedCall�-batch_normalization_2/StatefulPartitionedCall�-batch_normalization_3/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�-dense_1/kernel/Regularizer/Abs/ReadVariableOp�'stream_0_conv_1/StatefulPartitionedCall�5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp�'stream_0_conv_2/StatefulPartitionedCall�8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp�'stream_0_conv_3/StatefulPartitionedCall�5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp�
#stream_0_input_drop/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������}* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *Y
fTRR
P__inference_stream_0_input_drop_layer_call_and_return_conditional_losses_89070112%
#stream_0_input_drop/PartitionedCall�
'stream_0_conv_1/StatefulPartitionedCallStatefulPartitionedCall,stream_0_input_drop/PartitionedCall:output:0stream_0_conv_1_8907035stream_0_conv_1_8907037*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������}@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_stream_0_conv_1_layer_call_and_return_conditional_losses_89070342)
'stream_0_conv_1/StatefulPartitionedCall�
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall0stream_0_conv_1/StatefulPartitionedCall:output:0batch_normalization_8907060batch_normalization_8907062batch_normalization_8907064batch_normalization_8907066*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������}@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_89070592-
+batch_normalization/StatefulPartitionedCall�
activation/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������}@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_activation_layer_call_and_return_conditional_losses_89070742
activation/PartitionedCall�
stream_0_drop_1/PartitionedCallPartitionedCall#activation/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������}@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_stream_0_drop_1_layer_call_and_return_conditional_losses_89070812!
stream_0_drop_1/PartitionedCall�
'stream_0_conv_2/StatefulPartitionedCallStatefulPartitionedCall(stream_0_drop_1/PartitionedCall:output:0stream_0_conv_2_8907105stream_0_conv_2_8907107*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������}�*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_stream_0_conv_2_layer_call_and_return_conditional_losses_89071042)
'stream_0_conv_2/StatefulPartitionedCall�
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall0stream_0_conv_2/StatefulPartitionedCall:output:0batch_normalization_1_8907130batch_normalization_1_8907132batch_normalization_1_8907134batch_normalization_1_8907136*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������}�*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *[
fVRT
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_89071292/
-batch_normalization_1/StatefulPartitionedCall�
activation_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������}�* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_activation_1_layer_call_and_return_conditional_losses_89071442
activation_1/PartitionedCall�
stream_0_drop_2/PartitionedCallPartitionedCall%activation_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������}�* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_stream_0_drop_2_layer_call_and_return_conditional_losses_89071512!
stream_0_drop_2/PartitionedCall�
'stream_0_conv_3/StatefulPartitionedCallStatefulPartitionedCall(stream_0_drop_2/PartitionedCall:output:0stream_0_conv_3_8907175stream_0_conv_3_8907177*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������}�*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_stream_0_conv_3_layer_call_and_return_conditional_losses_89071742)
'stream_0_conv_3/StatefulPartitionedCall�
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall0stream_0_conv_3/StatefulPartitionedCall:output:0batch_normalization_2_8907200batch_normalization_2_8907202batch_normalization_2_8907204batch_normalization_2_8907206*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������}�*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *[
fVRT
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_89071992/
-batch_normalization_2/StatefulPartitionedCall�
activation_2/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������}�* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_activation_2_layer_call_and_return_conditional_losses_89072142
activation_2/PartitionedCall�
stream_0_drop_3/PartitionedCallPartitionedCall%activation_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������}�* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_stream_0_drop_3_layer_call_and_return_conditional_losses_89072212!
stream_0_drop_3/PartitionedCall�
(global_average_pooling1d/PartitionedCallPartitionedCall(stream_0_drop_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *^
fYRW
U__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_89072282*
(global_average_pooling1d/PartitionedCall�
dense_1_dropout/PartitionedCallPartitionedCall1global_average_pooling1d/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_dense_1_dropout_layer_call_and_return_conditional_losses_89072352!
dense_1_dropout/PartitionedCall�
dense_1/StatefulPartitionedCallStatefulPartitionedCall(dense_1_dropout/PartitionedCall:output:0dense_1_8907254dense_1_8907256*
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
D__inference_dense_1_layer_call_and_return_conditional_losses_89072532!
dense_1/StatefulPartitionedCall�
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0batch_normalization_3_8907259batch_normalization_3_8907261batch_normalization_3_8907263batch_normalization_3_8907265*
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
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_89068612/
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
O__inference_dense_activation_1_layer_call_and_return_conditional_losses_89072732$
"dense_activation_1/PartitionedCall�
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpstream_0_conv_1_8907035*"
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
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpstream_0_conv_2_8907105*#
_output_shapes
:@�*
dtype02:
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp�
)stream_0_conv_2/kernel/Regularizer/SquareSquare@stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*#
_output_shapes
:@�2+
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
5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpstream_0_conv_3_8907175*$
_output_shapes
:��*
dtype027
5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp�
&stream_0_conv_3/kernel/Regularizer/AbsAbs=stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*$
_output_shapes
:��2(
&stream_0_conv_3/kernel/Regularizer/Abs�
(stream_0_conv_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_0_conv_3/kernel/Regularizer/Const�
&stream_0_conv_3/kernel/Regularizer/SumSum*stream_0_conv_3/kernel/Regularizer/Abs:y:01stream_0_conv_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_3/kernel/Regularizer/Sum�
(stream_0_conv_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2*
(stream_0_conv_3/kernel/Regularizer/mul/x�
&stream_0_conv_3/kernel/Regularizer/mulMul1stream_0_conv_3/kernel/Regularizer/mul/x:output:0/stream_0_conv_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_3/kernel/Regularizer/mul�
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_1_8907254*
_output_shapes
:	�T*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOp�
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	�T2 
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
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall.^dense_1/kernel/Regularizer/Abs/ReadVariableOp(^stream_0_conv_1/StatefulPartitionedCall6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp(^stream_0_conv_2/StatefulPartitionedCall9^stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp(^stream_0_conv_3/StatefulPartitionedCall6^stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:���������}: : : : : : : : : : : : : : : : : : : : : : : : 2Z
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
:���������}
 
_user_specified_nameinputs
�
V
:__inference_global_average_pooling1d_layer_call_fn_8911436

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
U__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_89068232
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
�
j
L__inference_stream_0_drop_1_layer_call_and_return_conditional_losses_8907081

inputs

identity_1^
IdentityIdentityinputs*
T0*+
_output_shapes
:���������}@2

Identitym

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:���������}@2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������}@:S O
+
_output_shapes
:���������}@
 
_user_specified_nameinputs
�
�
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_8911557

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
�
7__inference_batch_normalization_1_layer_call_fn_8911027

inputs
unknown:	�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:�������������������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *[
fVRT
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_89065732
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:�������������������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):�������������������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:�������������������
 
_user_specified_nameinputs
�
j
1__inference_dense_1_dropout_layer_call_fn_8911463

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_dense_1_dropout_layer_call_and_return_conditional_losses_89073872
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
1__inference_stream_0_conv_3_layer_call_fn_8911213

inputs
unknown:��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������}�*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_stream_0_conv_3_layer_call_and_return_conditional_losses_89071742
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:���������}�2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:���������}�: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:���������}�
 
_user_specified_nameinputs
�
�
)__inference_model_1_layer_call_fn_8909005
left_inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:@ 
	unknown_5:@�
	unknown_6:	�
	unknown_7:	�
	unknown_8:	�
	unknown_9:	�

unknown_10:	�"

unknown_11:��

unknown_12:	�

unknown_13:	�

unknown_14:	�

unknown_15:	�

unknown_16:	�

unknown_17:	�T

unknown_18:T

unknown_19:T

unknown_20:T

unknown_21:T

unknown_22:T
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallleft_inputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
:���������T*2
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_model_1_layer_call_and_return_conditional_losses_89089012
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
_construction_contextkEagerRuntime*Z
_input_shapesI
G:���������}: : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
+
_output_shapes
:���������}
%
_user_specified_nameleft_inputs
�+
�
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_8911340

inputs6
'assignmovingavg_readvariableop_resource:	�8
)assignmovingavg_1_readvariableop_resource:	�4
%batchnorm_mul_readvariableop_resource:	�0
!batchnorm_readvariableop_resource:	�
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOp�
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indices�
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*#
_output_shapes
:�*
	keep_dims(2
moments/mean�
moments/StopGradientStopGradientmoments/mean:output:0*
T0*#
_output_shapes
:�2
moments/StopGradient�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*5
_output_shapes#
!:�������������������2
moments/SquaredDifference�
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices�
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*#
_output_shapes
:�*
	keep_dims(2
moments/variance�
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 2
moments/Squeeze�
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:�*
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
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:�*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:�2
AssignMovingAvg/sub�
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:�2
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
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:�*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:�2
AssignMovingAvg_1/sub�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:�2
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
T0*
_output_shapes	
:�2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:�2
batchnorm/Rsqrt�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype02
batchnorm/mul/ReadVariableOp�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2
batchnorm/mul�
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*5
_output_shapes#
!:�������������������2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:�2
batchnorm/mul_2�
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype02
batchnorm/ReadVariableOp�
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*5
_output_shapes#
!:�������������������2
batchnorm/add_1|
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*5
_output_shapes#
!:�������������������2

Identity�
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):�������������������: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:] Y
5
_output_shapes#
!:�������������������
 
_user_specified_nameinputs
�
�
__inference_loss_fn_0_8911612T
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
�
�
+__inference_basemodel_layer_call_fn_8909955
inputs_0
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:@ 
	unknown_5:@�
	unknown_6:	�
	unknown_7:	�
	unknown_8:	�
	unknown_9:	�

unknown_10:	�"

unknown_11:��

unknown_12:	�

unknown_13:	�

unknown_14:	�

unknown_15:	�

unknown_16:	�

unknown_17:	�T

unknown_18:T

unknown_19:T

unknown_20:T

unknown_21:T

unknown_22:T
identity��StatefulPartitionedCall�
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
:���������T*2
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_basemodel_layer_call_and_return_conditional_losses_89087182
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
_construction_contextkEagerRuntime*Z
_input_shapesI
G:���������}: : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
+
_output_shapes
:���������}
"
_user_specified_name
inputs/0
�
M
1__inference_stream_0_drop_1_layer_call_fn_8910943

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������}@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_stream_0_drop_1_layer_call_and_return_conditional_losses_89070812
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:���������}@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������}@:S O
+
_output_shapes
:���������}@
 
_user_specified_nameinputs
�
j
1__inference_stream_0_drop_1_layer_call_fn_8910948

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������}@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_stream_0_drop_1_layer_call_and_return_conditional_losses_89076132
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������}@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������}@22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������}@
 
_user_specified_nameinputs
�+
�
P__inference_batch_normalization_layer_call_and_return_conditional_losses_8910928

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
T0*+
_output_shapes
:���������}@2
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
batchnorm/mulz
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:���������}@2
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
T0*+
_output_shapes
:���������}@2
batchnorm/add_1r
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*+
_output_shapes
:���������}@2

Identity�
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������}@: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:S O
+
_output_shapes
:���������}@
 
_user_specified_nameinputs
�
�
P__inference_batch_normalization_layer_call_and_return_conditional_losses_8910840

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
�
j
L__inference_stream_0_drop_1_layer_call_and_return_conditional_losses_8910953

inputs

identity_1^
IdentityIdentityinputs*
T0*+
_output_shapes
:���������}@2

Identitym

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:���������}@2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������}@:S O
+
_output_shapes
:���������}@
 
_user_specified_nameinputs
�
�
L__inference_stream_0_conv_3_layer_call_and_return_conditional_losses_8911234

inputsC
+conv1d_expanddims_1_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�"conv1d/ExpandDims_1/ReadVariableOp�5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOpy
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
:���������}�2
conv1d/ExpandDims�
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:��*
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
T0*(
_output_shapes
:��2
conv1d/ExpandDims_1�
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:���������}�*
paddingSAME*
strides
2
conv1d�
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:���������}�*
squeeze_dims

���������2
conv1d/Squeeze�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������}�2	
BiasAdd�
5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:��*
dtype027
5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp�
&stream_0_conv_3/kernel/Regularizer/AbsAbs=stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*$
_output_shapes
:��2(
&stream_0_conv_3/kernel/Regularizer/Abs�
(stream_0_conv_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_0_conv_3/kernel/Regularizer/Const�
&stream_0_conv_3/kernel/Regularizer/SumSum*stream_0_conv_3/kernel/Regularizer/Abs:y:01stream_0_conv_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_3/kernel/Regularizer/Sum�
(stream_0_conv_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2*
(stream_0_conv_3/kernel/Regularizer/mul/x�
&stream_0_conv_3/kernel/Regularizer/mulMul1stream_0_conv_3/kernel/Regularizer/mul/x:output:0/stream_0_conv_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_3/kernel/Regularizer/mulp
IdentityIdentityBiasAdd:output:0^NoOp*
T0*,
_output_shapes
:���������}�2

Identity�
NoOpNoOp^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp6^stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:���������}�: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp2n
5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp:T P
,
_output_shapes
:���������}�
 
_user_specified_nameinputs
�+
�
P__inference_batch_normalization_layer_call_and_return_conditional_losses_8906411

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
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_8906513

inputs0
!batchnorm_readvariableop_resource:	�4
%batchnorm_mul_readvariableop_resource:	�2
#batchnorm_readvariableop_1_resource:	�2
#batchnorm_readvariableop_2_resource:	�
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOp�
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:�*
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
T0*
_output_shapes	
:�2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:�2
batchnorm/Rsqrt�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype02
batchnorm/mul/ReadVariableOp�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2
batchnorm/mul�
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*5
_output_shapes#
!:�������������������2
batchnorm/mul_1�
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:�*
dtype02
batchnorm/ReadVariableOp_1�
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:�2
batchnorm/mul_2�
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:�*
dtype02
batchnorm/ReadVariableOp_2�
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*5
_output_shapes#
!:�������������������2
batchnorm/add_1|
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*5
_output_shapes#
!:�������������������2

Identity�
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):�������������������: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:] Y
5
_output_shapes#
!:�������������������
 
_user_specified_nameinputs
�
J
.__inference_activation_1_layer_call_fn_8911166

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
:���������}�* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_activation_1_layer_call_and_return_conditional_losses_89071442
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:���������}�2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:���������}�:T P
,
_output_shapes
:���������}�
 
_user_specified_nameinputs
��
�
F__inference_basemodel_layer_call_and_return_conditional_losses_8910705
inputs_0Q
;stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource:@=
/stream_0_conv_1_biasadd_readvariableop_resource:@I
;batch_normalization_assignmovingavg_readvariableop_resource:@K
=batch_normalization_assignmovingavg_1_readvariableop_resource:@G
9batch_normalization_batchnorm_mul_readvariableop_resource:@C
5batch_normalization_batchnorm_readvariableop_resource:@R
;stream_0_conv_2_conv1d_expanddims_1_readvariableop_resource:@�>
/stream_0_conv_2_biasadd_readvariableop_resource:	�L
=batch_normalization_1_assignmovingavg_readvariableop_resource:	�N
?batch_normalization_1_assignmovingavg_1_readvariableop_resource:	�J
;batch_normalization_1_batchnorm_mul_readvariableop_resource:	�F
7batch_normalization_1_batchnorm_readvariableop_resource:	�S
;stream_0_conv_3_conv1d_expanddims_1_readvariableop_resource:��>
/stream_0_conv_3_biasadd_readvariableop_resource:	�L
=batch_normalization_2_assignmovingavg_readvariableop_resource:	�N
?batch_normalization_2_assignmovingavg_1_readvariableop_resource:	�J
;batch_normalization_2_batchnorm_mul_readvariableop_resource:	�F
7batch_normalization_2_batchnorm_readvariableop_resource:	�9
&dense_1_matmul_readvariableop_resource:	�T5
'dense_1_biasadd_readvariableop_resource:TK
=batch_normalization_3_assignmovingavg_readvariableop_resource:TM
?batch_normalization_3_assignmovingavg_1_readvariableop_resource:TI
;batch_normalization_3_batchnorm_mul_readvariableop_resource:TE
7batch_normalization_3_batchnorm_readvariableop_resource:T
identity��#batch_normalization/AssignMovingAvg�2batch_normalization/AssignMovingAvg/ReadVariableOp�%batch_normalization/AssignMovingAvg_1�4batch_normalization/AssignMovingAvg_1/ReadVariableOp�,batch_normalization/batchnorm/ReadVariableOp�0batch_normalization/batchnorm/mul/ReadVariableOp�%batch_normalization_1/AssignMovingAvg�4batch_normalization_1/AssignMovingAvg/ReadVariableOp�'batch_normalization_1/AssignMovingAvg_1�6batch_normalization_1/AssignMovingAvg_1/ReadVariableOp�.batch_normalization_1/batchnorm/ReadVariableOp�2batch_normalization_1/batchnorm/mul/ReadVariableOp�%batch_normalization_2/AssignMovingAvg�4batch_normalization_2/AssignMovingAvg/ReadVariableOp�'batch_normalization_2/AssignMovingAvg_1�6batch_normalization_2/AssignMovingAvg_1/ReadVariableOp�.batch_normalization_2/batchnorm/ReadVariableOp�2batch_normalization_2/batchnorm/mul/ReadVariableOp�%batch_normalization_3/AssignMovingAvg�4batch_normalization_3/AssignMovingAvg/ReadVariableOp�'batch_normalization_3/AssignMovingAvg_1�6batch_normalization_3/AssignMovingAvg_1/ReadVariableOp�.batch_normalization_3/batchnorm/ReadVariableOp�2batch_normalization_3/batchnorm/mul/ReadVariableOp�dense_1/BiasAdd/ReadVariableOp�dense_1/MatMul/ReadVariableOp�-dense_1/kernel/Regularizer/Abs/ReadVariableOp�&stream_0_conv_1/BiasAdd/ReadVariableOp�2stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp�5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp�&stream_0_conv_2/BiasAdd/ReadVariableOp�2stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOp�8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp�&stream_0_conv_3/BiasAdd/ReadVariableOp�2stream_0_conv_3/conv1d/ExpandDims_1/ReadVariableOp�5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp�
!stream_0_input_drop/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2#
!stream_0_input_drop/dropout/Const�
stream_0_input_drop/dropout/MulMulinputs_0*stream_0_input_drop/dropout/Const:output:0*
T0*+
_output_shapes
:���������}2!
stream_0_input_drop/dropout/Mul~
!stream_0_input_drop/dropout/ShapeShapeinputs_0*
T0*
_output_shapes
:2#
!stream_0_input_drop/dropout/Shape�
8stream_0_input_drop/dropout/random_uniform/RandomUniformRandomUniform*stream_0_input_drop/dropout/Shape:output:0*
T0*+
_output_shapes
:���������}*
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
T0*+
_output_shapes
:���������}2*
(stream_0_input_drop/dropout/GreaterEqual�
 stream_0_input_drop/dropout/CastCast,stream_0_input_drop/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:���������}2"
 stream_0_input_drop/dropout/Cast�
!stream_0_input_drop/dropout/Mul_1Mul#stream_0_input_drop/dropout/Mul:z:0$stream_0_input_drop/dropout/Cast:y:0*
T0*+
_output_shapes
:���������}2#
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
T0*/
_output_shapes
:���������}2#
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
T0*/
_output_shapes
:���������}@*
paddingSAME*
strides
2
stream_0_conv_1/conv1d�
stream_0_conv_1/conv1d/SqueezeSqueezestream_0_conv_1/conv1d:output:0*
T0*+
_output_shapes
:���������}@*
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
T0*+
_output_shapes
:���������}@2
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
:@*
	keep_dims(2"
 batch_normalization/moments/mean�
(batch_normalization/moments/StopGradientStopGradient)batch_normalization/moments/mean:output:0*
T0*"
_output_shapes
:@2*
(batch_normalization/moments/StopGradient�
-batch_normalization/moments/SquaredDifferenceSquaredDifference stream_0_conv_1/BiasAdd:output:01batch_normalization/moments/StopGradient:output:0*
T0*+
_output_shapes
:���������}@2/
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
T0*+
_output_shapes
:���������}@2%
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
T0*+
_output_shapes
:���������}@2%
#batch_normalization/batchnorm/add_1�
activation/ReluRelu'batch_normalization/batchnorm/add_1:z:0*
T0*+
_output_shapes
:���������}@2
activation/Relu�
stream_0_drop_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2
stream_0_drop_1/dropout/Const�
stream_0_drop_1/dropout/MulMulactivation/Relu:activations:0&stream_0_drop_1/dropout/Const:output:0*
T0*+
_output_shapes
:���������}@2
stream_0_drop_1/dropout/Mul�
stream_0_drop_1/dropout/ShapeShapeactivation/Relu:activations:0*
T0*
_output_shapes
:2
stream_0_drop_1/dropout/Shape�
4stream_0_drop_1/dropout/random_uniform/RandomUniformRandomUniform&stream_0_drop_1/dropout/Shape:output:0*
T0*+
_output_shapes
:���������}@*
dtype0*
seed�*
seed2�26
4stream_0_drop_1/dropout/random_uniform/RandomUniform�
&stream_0_drop_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���>2(
&stream_0_drop_1/dropout/GreaterEqual/y�
$stream_0_drop_1/dropout/GreaterEqualGreaterEqual=stream_0_drop_1/dropout/random_uniform/RandomUniform:output:0/stream_0_drop_1/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:���������}@2&
$stream_0_drop_1/dropout/GreaterEqual�
stream_0_drop_1/dropout/CastCast(stream_0_drop_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:���������}@2
stream_0_drop_1/dropout/Cast�
stream_0_drop_1/dropout/Mul_1Mulstream_0_drop_1/dropout/Mul:z:0 stream_0_drop_1/dropout/Cast:y:0*
T0*+
_output_shapes
:���������}@2
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
T0*/
_output_shapes
:���������}@2#
!stream_0_conv_2/conv1d/ExpandDims�
2stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp;stream_0_conv_2_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:@�*
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
T0*'
_output_shapes
:@�2%
#stream_0_conv_2/conv1d/ExpandDims_1�
stream_0_conv_2/conv1dConv2D*stream_0_conv_2/conv1d/ExpandDims:output:0,stream_0_conv_2/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:���������}�*
paddingSAME*
strides
2
stream_0_conv_2/conv1d�
stream_0_conv_2/conv1d/SqueezeSqueezestream_0_conv_2/conv1d:output:0*
T0*,
_output_shapes
:���������}�*
squeeze_dims

���������2 
stream_0_conv_2/conv1d/Squeeze�
&stream_0_conv_2/BiasAdd/ReadVariableOpReadVariableOp/stream_0_conv_2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02(
&stream_0_conv_2/BiasAdd/ReadVariableOp�
stream_0_conv_2/BiasAddBiasAdd'stream_0_conv_2/conv1d/Squeeze:output:0.stream_0_conv_2/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������}�2
stream_0_conv_2/BiasAdd�
4batch_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       26
4batch_normalization_1/moments/mean/reduction_indices�
"batch_normalization_1/moments/meanMean stream_0_conv_2/BiasAdd:output:0=batch_normalization_1/moments/mean/reduction_indices:output:0*
T0*#
_output_shapes
:�*
	keep_dims(2$
"batch_normalization_1/moments/mean�
*batch_normalization_1/moments/StopGradientStopGradient+batch_normalization_1/moments/mean:output:0*
T0*#
_output_shapes
:�2,
*batch_normalization_1/moments/StopGradient�
/batch_normalization_1/moments/SquaredDifferenceSquaredDifference stream_0_conv_2/BiasAdd:output:03batch_normalization_1/moments/StopGradient:output:0*
T0*,
_output_shapes
:���������}�21
/batch_normalization_1/moments/SquaredDifference�
8batch_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2:
8batch_normalization_1/moments/variance/reduction_indices�
&batch_normalization_1/moments/varianceMean3batch_normalization_1/moments/SquaredDifference:z:0Abatch_normalization_1/moments/variance/reduction_indices:output:0*
T0*#
_output_shapes
:�*
	keep_dims(2(
&batch_normalization_1/moments/variance�
%batch_normalization_1/moments/SqueezeSqueeze+batch_normalization_1/moments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 2'
%batch_normalization_1/moments/Squeeze�
'batch_normalization_1/moments/Squeeze_1Squeeze/batch_normalization_1/moments/variance:output:0*
T0*
_output_shapes	
:�*
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
4batch_normalization_1/AssignMovingAvg/ReadVariableOpReadVariableOp=batch_normalization_1_assignmovingavg_readvariableop_resource*
_output_shapes	
:�*
dtype026
4batch_normalization_1/AssignMovingAvg/ReadVariableOp�
)batch_normalization_1/AssignMovingAvg/subSub<batch_normalization_1/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_1/moments/Squeeze:output:0*
T0*
_output_shapes	
:�2+
)batch_normalization_1/AssignMovingAvg/sub�
)batch_normalization_1/AssignMovingAvg/mulMul-batch_normalization_1/AssignMovingAvg/sub:z:04batch_normalization_1/AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:�2+
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
6batch_normalization_1/AssignMovingAvg_1/ReadVariableOpReadVariableOp?batch_normalization_1_assignmovingavg_1_readvariableop_resource*
_output_shapes	
:�*
dtype028
6batch_normalization_1/AssignMovingAvg_1/ReadVariableOp�
+batch_normalization_1/AssignMovingAvg_1/subSub>batch_normalization_1/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_1/moments/Squeeze_1:output:0*
T0*
_output_shapes	
:�2-
+batch_normalization_1/AssignMovingAvg_1/sub�
+batch_normalization_1/AssignMovingAvg_1/mulMul/batch_normalization_1/AssignMovingAvg_1/sub:z:06batch_normalization_1/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:�2-
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
T0*
_output_shapes	
:�2%
#batch_normalization_1/batchnorm/add�
%batch_normalization_1/batchnorm/RsqrtRsqrt'batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes	
:�2'
%batch_normalization_1/batchnorm/Rsqrt�
2batch_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype024
2batch_normalization_1/batchnorm/mul/ReadVariableOp�
#batch_normalization_1/batchnorm/mulMul)batch_normalization_1/batchnorm/Rsqrt:y:0:batch_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2%
#batch_normalization_1/batchnorm/mul�
%batch_normalization_1/batchnorm/mul_1Mul stream_0_conv_2/BiasAdd:output:0'batch_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:���������}�2'
%batch_normalization_1/batchnorm/mul_1�
%batch_normalization_1/batchnorm/mul_2Mul.batch_normalization_1/moments/Squeeze:output:0'batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes	
:�2'
%batch_normalization_1/batchnorm/mul_2�
.batch_normalization_1/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_1_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype020
.batch_normalization_1/batchnorm/ReadVariableOp�
#batch_normalization_1/batchnorm/subSub6batch_normalization_1/batchnorm/ReadVariableOp:value:0)batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�2%
#batch_normalization_1/batchnorm/sub�
%batch_normalization_1/batchnorm/add_1AddV2)batch_normalization_1/batchnorm/mul_1:z:0'batch_normalization_1/batchnorm/sub:z:0*
T0*,
_output_shapes
:���������}�2'
%batch_normalization_1/batchnorm/add_1�
activation_1/ReluRelu)batch_normalization_1/batchnorm/add_1:z:0*
T0*,
_output_shapes
:���������}�2
activation_1/Relu�
stream_0_drop_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU�?2
stream_0_drop_2/dropout/Const�
stream_0_drop_2/dropout/MulMulactivation_1/Relu:activations:0&stream_0_drop_2/dropout/Const:output:0*
T0*,
_output_shapes
:���������}�2
stream_0_drop_2/dropout/Mul�
stream_0_drop_2/dropout/ShapeShapeactivation_1/Relu:activations:0*
T0*
_output_shapes
:2
stream_0_drop_2/dropout/Shape�
4stream_0_drop_2/dropout/random_uniform/RandomUniformRandomUniform&stream_0_drop_2/dropout/Shape:output:0*
T0*,
_output_shapes
:���������}�*
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
:���������}�2&
$stream_0_drop_2/dropout/GreaterEqual�
stream_0_drop_2/dropout/CastCast(stream_0_drop_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:���������}�2
stream_0_drop_2/dropout/Cast�
stream_0_drop_2/dropout/Mul_1Mulstream_0_drop_2/dropout/Mul:z:0 stream_0_drop_2/dropout/Cast:y:0*
T0*,
_output_shapes
:���������}�2
stream_0_drop_2/dropout/Mul_1�
%stream_0_conv_3/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2'
%stream_0_conv_3/conv1d/ExpandDims/dim�
!stream_0_conv_3/conv1d/ExpandDims
ExpandDims!stream_0_drop_2/dropout/Mul_1:z:0.stream_0_conv_3/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:���������}�2#
!stream_0_conv_3/conv1d/ExpandDims�
2stream_0_conv_3/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp;stream_0_conv_3_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:��*
dtype024
2stream_0_conv_3/conv1d/ExpandDims_1/ReadVariableOp�
'stream_0_conv_3/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2)
'stream_0_conv_3/conv1d/ExpandDims_1/dim�
#stream_0_conv_3/conv1d/ExpandDims_1
ExpandDims:stream_0_conv_3/conv1d/ExpandDims_1/ReadVariableOp:value:00stream_0_conv_3/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:��2%
#stream_0_conv_3/conv1d/ExpandDims_1�
stream_0_conv_3/conv1dConv2D*stream_0_conv_3/conv1d/ExpandDims:output:0,stream_0_conv_3/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:���������}�*
paddingSAME*
strides
2
stream_0_conv_3/conv1d�
stream_0_conv_3/conv1d/SqueezeSqueezestream_0_conv_3/conv1d:output:0*
T0*,
_output_shapes
:���������}�*
squeeze_dims

���������2 
stream_0_conv_3/conv1d/Squeeze�
&stream_0_conv_3/BiasAdd/ReadVariableOpReadVariableOp/stream_0_conv_3_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02(
&stream_0_conv_3/BiasAdd/ReadVariableOp�
stream_0_conv_3/BiasAddBiasAdd'stream_0_conv_3/conv1d/Squeeze:output:0.stream_0_conv_3/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������}�2
stream_0_conv_3/BiasAdd�
4batch_normalization_2/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       26
4batch_normalization_2/moments/mean/reduction_indices�
"batch_normalization_2/moments/meanMean stream_0_conv_3/BiasAdd:output:0=batch_normalization_2/moments/mean/reduction_indices:output:0*
T0*#
_output_shapes
:�*
	keep_dims(2$
"batch_normalization_2/moments/mean�
*batch_normalization_2/moments/StopGradientStopGradient+batch_normalization_2/moments/mean:output:0*
T0*#
_output_shapes
:�2,
*batch_normalization_2/moments/StopGradient�
/batch_normalization_2/moments/SquaredDifferenceSquaredDifference stream_0_conv_3/BiasAdd:output:03batch_normalization_2/moments/StopGradient:output:0*
T0*,
_output_shapes
:���������}�21
/batch_normalization_2/moments/SquaredDifference�
8batch_normalization_2/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2:
8batch_normalization_2/moments/variance/reduction_indices�
&batch_normalization_2/moments/varianceMean3batch_normalization_2/moments/SquaredDifference:z:0Abatch_normalization_2/moments/variance/reduction_indices:output:0*
T0*#
_output_shapes
:�*
	keep_dims(2(
&batch_normalization_2/moments/variance�
%batch_normalization_2/moments/SqueezeSqueeze+batch_normalization_2/moments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 2'
%batch_normalization_2/moments/Squeeze�
'batch_normalization_2/moments/Squeeze_1Squeeze/batch_normalization_2/moments/variance:output:0*
T0*
_output_shapes	
:�*
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
4batch_normalization_2/AssignMovingAvg/ReadVariableOpReadVariableOp=batch_normalization_2_assignmovingavg_readvariableop_resource*
_output_shapes	
:�*
dtype026
4batch_normalization_2/AssignMovingAvg/ReadVariableOp�
)batch_normalization_2/AssignMovingAvg/subSub<batch_normalization_2/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_2/moments/Squeeze:output:0*
T0*
_output_shapes	
:�2+
)batch_normalization_2/AssignMovingAvg/sub�
)batch_normalization_2/AssignMovingAvg/mulMul-batch_normalization_2/AssignMovingAvg/sub:z:04batch_normalization_2/AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:�2+
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
6batch_normalization_2/AssignMovingAvg_1/ReadVariableOpReadVariableOp?batch_normalization_2_assignmovingavg_1_readvariableop_resource*
_output_shapes	
:�*
dtype028
6batch_normalization_2/AssignMovingAvg_1/ReadVariableOp�
+batch_normalization_2/AssignMovingAvg_1/subSub>batch_normalization_2/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_2/moments/Squeeze_1:output:0*
T0*
_output_shapes	
:�2-
+batch_normalization_2/AssignMovingAvg_1/sub�
+batch_normalization_2/AssignMovingAvg_1/mulMul/batch_normalization_2/AssignMovingAvg_1/sub:z:06batch_normalization_2/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:�2-
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
T0*
_output_shapes	
:�2%
#batch_normalization_2/batchnorm/add�
%batch_normalization_2/batchnorm/RsqrtRsqrt'batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes	
:�2'
%batch_normalization_2/batchnorm/Rsqrt�
2batch_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype024
2batch_normalization_2/batchnorm/mul/ReadVariableOp�
#batch_normalization_2/batchnorm/mulMul)batch_normalization_2/batchnorm/Rsqrt:y:0:batch_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2%
#batch_normalization_2/batchnorm/mul�
%batch_normalization_2/batchnorm/mul_1Mul stream_0_conv_3/BiasAdd:output:0'batch_normalization_2/batchnorm/mul:z:0*
T0*,
_output_shapes
:���������}�2'
%batch_normalization_2/batchnorm/mul_1�
%batch_normalization_2/batchnorm/mul_2Mul.batch_normalization_2/moments/Squeeze:output:0'batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes	
:�2'
%batch_normalization_2/batchnorm/mul_2�
.batch_normalization_2/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_2_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype020
.batch_normalization_2/batchnorm/ReadVariableOp�
#batch_normalization_2/batchnorm/subSub6batch_normalization_2/batchnorm/ReadVariableOp:value:0)batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�2%
#batch_normalization_2/batchnorm/sub�
%batch_normalization_2/batchnorm/add_1AddV2)batch_normalization_2/batchnorm/mul_1:z:0'batch_normalization_2/batchnorm/sub:z:0*
T0*,
_output_shapes
:���������}�2'
%batch_normalization_2/batchnorm/add_1�
activation_2/ReluRelu)batch_normalization_2/batchnorm/add_1:z:0*
T0*,
_output_shapes
:���������}�2
activation_2/Relu�
stream_0_drop_3/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
stream_0_drop_3/dropout/Const�
stream_0_drop_3/dropout/MulMulactivation_2/Relu:activations:0&stream_0_drop_3/dropout/Const:output:0*
T0*,
_output_shapes
:���������}�2
stream_0_drop_3/dropout/Mul�
stream_0_drop_3/dropout/ShapeShapeactivation_2/Relu:activations:0*
T0*
_output_shapes
:2
stream_0_drop_3/dropout/Shape�
4stream_0_drop_3/dropout/random_uniform/RandomUniformRandomUniform&stream_0_drop_3/dropout/Shape:output:0*
T0*,
_output_shapes
:���������}�*
dtype0*
seed�*
seed2�26
4stream_0_drop_3/dropout/random_uniform/RandomUniform�
&stream_0_drop_3/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2(
&stream_0_drop_3/dropout/GreaterEqual/y�
$stream_0_drop_3/dropout/GreaterEqualGreaterEqual=stream_0_drop_3/dropout/random_uniform/RandomUniform:output:0/stream_0_drop_3/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:���������}�2&
$stream_0_drop_3/dropout/GreaterEqual�
stream_0_drop_3/dropout/CastCast(stream_0_drop_3/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:���������}�2
stream_0_drop_3/dropout/Cast�
stream_0_drop_3/dropout/Mul_1Mulstream_0_drop_3/dropout/Mul:z:0 stream_0_drop_3/dropout/Cast:y:0*
T0*,
_output_shapes
:���������}�2
stream_0_drop_3/dropout/Mul_1�
/global_average_pooling1d/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :21
/global_average_pooling1d/Mean/reduction_indices�
global_average_pooling1d/MeanMean!stream_0_drop_3/dropout/Mul_1:z:08global_average_pooling1d/Mean/reduction_indices:output:0*
T0*(
_output_shapes
:����������2
global_average_pooling1d/Mean�
dense_1_dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
dense_1_dropout/dropout/Const�
dense_1_dropout/dropout/MulMul&global_average_pooling1d/Mean:output:0&dense_1_dropout/dropout/Const:output:0*
T0*(
_output_shapes
:����������2
dense_1_dropout/dropout/Mul�
dense_1_dropout/dropout/ShapeShape&global_average_pooling1d/Mean:output:0*
T0*
_output_shapes
:2
dense_1_dropout/dropout/Shape�
4dense_1_dropout/dropout/random_uniform/RandomUniformRandomUniform&dense_1_dropout/dropout/Shape:output:0*
T0*(
_output_shapes
:����������*
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
T0*(
_output_shapes
:����������2&
$dense_1_dropout/dropout/GreaterEqual�
dense_1_dropout/dropout/CastCast(dense_1_dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:����������2
dense_1_dropout/dropout/Cast�
dense_1_dropout/dropout/Mul_1Muldense_1_dropout/dropout/Mul:z:0 dense_1_dropout/dropout/Cast:y:0*
T0*(
_output_shapes
:����������2
dense_1_dropout/dropout/Mul_1�
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	�T*
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
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;stream_0_conv_2_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:@�*
dtype02:
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp�
)stream_0_conv_2/kernel/Regularizer/SquareSquare@stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*#
_output_shapes
:@�2+
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
5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp;stream_0_conv_3_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:��*
dtype027
5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp�
&stream_0_conv_3/kernel/Regularizer/AbsAbs=stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*$
_output_shapes
:��2(
&stream_0_conv_3/kernel/Regularizer/Abs�
(stream_0_conv_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_0_conv_3/kernel/Regularizer/Const�
&stream_0_conv_3/kernel/Regularizer/SumSum*stream_0_conv_3/kernel/Regularizer/Abs:y:01stream_0_conv_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_3/kernel/Regularizer/Sum�
(stream_0_conv_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2*
(stream_0_conv_3/kernel/Regularizer/mul/x�
&stream_0_conv_3/kernel/Regularizer/mulMul1stream_0_conv_3/kernel/Regularizer/mul/x:output:0/stream_0_conv_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_3/kernel/Regularizer/mul�
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	�T*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOp�
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	�T2 
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
NoOpNoOp$^batch_normalization/AssignMovingAvg3^batch_normalization/AssignMovingAvg/ReadVariableOp&^batch_normalization/AssignMovingAvg_15^batch_normalization/AssignMovingAvg_1/ReadVariableOp-^batch_normalization/batchnorm/ReadVariableOp1^batch_normalization/batchnorm/mul/ReadVariableOp&^batch_normalization_1/AssignMovingAvg5^batch_normalization_1/AssignMovingAvg/ReadVariableOp(^batch_normalization_1/AssignMovingAvg_17^batch_normalization_1/AssignMovingAvg_1/ReadVariableOp/^batch_normalization_1/batchnorm/ReadVariableOp3^batch_normalization_1/batchnorm/mul/ReadVariableOp&^batch_normalization_2/AssignMovingAvg5^batch_normalization_2/AssignMovingAvg/ReadVariableOp(^batch_normalization_2/AssignMovingAvg_17^batch_normalization_2/AssignMovingAvg_1/ReadVariableOp/^batch_normalization_2/batchnorm/ReadVariableOp3^batch_normalization_2/batchnorm/mul/ReadVariableOp&^batch_normalization_3/AssignMovingAvg5^batch_normalization_3/AssignMovingAvg/ReadVariableOp(^batch_normalization_3/AssignMovingAvg_17^batch_normalization_3/AssignMovingAvg_1/ReadVariableOp/^batch_normalization_3/batchnorm/ReadVariableOp3^batch_normalization_3/batchnorm/mul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp.^dense_1/kernel/Regularizer/Abs/ReadVariableOp'^stream_0_conv_1/BiasAdd/ReadVariableOp3^stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp'^stream_0_conv_2/BiasAdd/ReadVariableOp3^stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOp9^stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp'^stream_0_conv_3/BiasAdd/ReadVariableOp3^stream_0_conv_3/conv1d/ExpandDims_1/ReadVariableOp6^stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:���������}: : : : : : : : : : : : : : : : : : : : : : : : 2J
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
:���������}
"
_user_specified_name
inputs/0
�
e
I__inference_activation_1_layer_call_and_return_conditional_losses_8911171

inputs
identityS
ReluReluinputs*
T0*,
_output_shapes
:���������}�2
Reluk
IdentityIdentityRelu:activations:0*
T0*,
_output_shapes
:���������}�2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:���������}�:T P
,
_output_shapes
:���������}�
 
_user_specified_nameinputs
�
j
L__inference_dense_1_dropout_layer_call_and_return_conditional_losses_8907235

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:����������2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:����������2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�+
�
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_8911394

inputs6
'assignmovingavg_readvariableop_resource:	�8
)assignmovingavg_1_readvariableop_resource:	�4
%batchnorm_mul_readvariableop_resource:	�0
!batchnorm_readvariableop_resource:	�
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOp�
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indices�
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*#
_output_shapes
:�*
	keep_dims(2
moments/mean�
moments/StopGradientStopGradientmoments/mean:output:0*
T0*#
_output_shapes
:�2
moments/StopGradient�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*,
_output_shapes
:���������}�2
moments/SquaredDifference�
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices�
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*#
_output_shapes
:�*
	keep_dims(2
moments/variance�
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 2
moments/Squeeze�
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:�*
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
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:�*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:�2
AssignMovingAvg/sub�
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:�2
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
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:�*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:�2
AssignMovingAvg_1/sub�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:�2
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
T0*
_output_shapes	
:�2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:�2
batchnorm/Rsqrt�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype02
batchnorm/mul/ReadVariableOp�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2
batchnorm/mul{
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*,
_output_shapes
:���������}�2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:�2
batchnorm/mul_2�
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype02
batchnorm/ReadVariableOp�
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*,
_output_shapes
:���������}�2
batchnorm/add_1s
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*,
_output_shapes
:���������}�2

Identity�
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������}�: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:T P
,
_output_shapes
:���������}�
 
_user_specified_nameinputs
�
�
D__inference_dense_1_layer_call_and_return_conditional_losses_8911511

inputs1
matmul_readvariableop_resource:	�T-
biasadd_readvariableop_resource:T
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�-dense_1/kernel/Regularizer/Abs/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�T*
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
:	�T*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOp�
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	�T2 
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
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2^
-dense_1/kernel/Regularizer/Abs/ReadVariableOp-dense_1/kernel/Regularizer/Abs/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
1__inference_stream_0_conv_2_layer_call_fn_8910980

inputs
unknown:@�
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������}�*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_stream_0_conv_2_layer_call_and_return_conditional_losses_89071042
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:���������}�2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������}@: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������}@
 
_user_specified_nameinputs
�
�
L__inference_stream_0_conv_1_layer_call_and_return_conditional_losses_8907034

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
T0*/
_output_shapes
:���������}2
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
T0*/
_output_shapes
:���������}@*
paddingSAME*
strides
2
conv1d�
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:���������}@*
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
T0*+
_output_shapes
:���������}@2	
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
&stream_0_conv_1/kernel/Regularizer/mulo
IdentityIdentityBiasAdd:output:0^NoOp*
T0*+
_output_shapes
:���������}@2

Identity�
NoOpNoOp^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������}: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp2n
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:S O
+
_output_shapes
:���������}
 
_user_specified_nameinputs
�
�
P__inference_batch_normalization_layer_call_and_return_conditional_losses_8907059

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
batchnorm/mulz
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:���������}@2
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
T0*+
_output_shapes
:���������}@2
batchnorm/add_1r
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*+
_output_shapes
:���������}@2

Identity�
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������}@: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:S O
+
_output_shapes
:���������}@
 
_user_specified_nameinputs
�
j
L__inference_stream_0_drop_2_layer_call_and_return_conditional_losses_8911186

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:���������}�2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:���������}�2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:���������}�:T P
,
_output_shapes
:���������}�
 
_user_specified_nameinputs
�+
�
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_8907473

inputs6
'assignmovingavg_readvariableop_resource:	�8
)assignmovingavg_1_readvariableop_resource:	�4
%batchnorm_mul_readvariableop_resource:	�0
!batchnorm_readvariableop_resource:	�
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOp�
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indices�
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*#
_output_shapes
:�*
	keep_dims(2
moments/mean�
moments/StopGradientStopGradientmoments/mean:output:0*
T0*#
_output_shapes
:�2
moments/StopGradient�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*,
_output_shapes
:���������}�2
moments/SquaredDifference�
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices�
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*#
_output_shapes
:�*
	keep_dims(2
moments/variance�
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 2
moments/Squeeze�
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:�*
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
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:�*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:�2
AssignMovingAvg/sub�
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:�2
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
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:�*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:�2
AssignMovingAvg_1/sub�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:�2
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
T0*
_output_shapes	
:�2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:�2
batchnorm/Rsqrt�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype02
batchnorm/mul/ReadVariableOp�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2
batchnorm/mul{
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*,
_output_shapes
:���������}�2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:�2
batchnorm/mul_2�
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype02
batchnorm/ReadVariableOp�
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*,
_output_shapes
:���������}�2
batchnorm/add_1s
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*,
_output_shapes
:���������}�2

Identity�
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������}�: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:T P
,
_output_shapes
:���������}�
 
_user_specified_nameinputs
��
�
F__inference_basemodel_layer_call_and_return_conditional_losses_8910472
inputs_0Q
;stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource:@=
/stream_0_conv_1_biasadd_readvariableop_resource:@C
5batch_normalization_batchnorm_readvariableop_resource:@G
9batch_normalization_batchnorm_mul_readvariableop_resource:@E
7batch_normalization_batchnorm_readvariableop_1_resource:@E
7batch_normalization_batchnorm_readvariableop_2_resource:@R
;stream_0_conv_2_conv1d_expanddims_1_readvariableop_resource:@�>
/stream_0_conv_2_biasadd_readvariableop_resource:	�F
7batch_normalization_1_batchnorm_readvariableop_resource:	�J
;batch_normalization_1_batchnorm_mul_readvariableop_resource:	�H
9batch_normalization_1_batchnorm_readvariableop_1_resource:	�H
9batch_normalization_1_batchnorm_readvariableop_2_resource:	�S
;stream_0_conv_3_conv1d_expanddims_1_readvariableop_resource:��>
/stream_0_conv_3_biasadd_readvariableop_resource:	�F
7batch_normalization_2_batchnorm_readvariableop_resource:	�J
;batch_normalization_2_batchnorm_mul_readvariableop_resource:	�H
9batch_normalization_2_batchnorm_readvariableop_1_resource:	�H
9batch_normalization_2_batchnorm_readvariableop_2_resource:	�9
&dense_1_matmul_readvariableop_resource:	�T5
'dense_1_biasadd_readvariableop_resource:TE
7batch_normalization_3_batchnorm_readvariableop_resource:TI
;batch_normalization_3_batchnorm_mul_readvariableop_resource:TG
9batch_normalization_3_batchnorm_readvariableop_1_resource:TG
9batch_normalization_3_batchnorm_readvariableop_2_resource:T
identity��,batch_normalization/batchnorm/ReadVariableOp�.batch_normalization/batchnorm/ReadVariableOp_1�.batch_normalization/batchnorm/ReadVariableOp_2�0batch_normalization/batchnorm/mul/ReadVariableOp�.batch_normalization_1/batchnorm/ReadVariableOp�0batch_normalization_1/batchnorm/ReadVariableOp_1�0batch_normalization_1/batchnorm/ReadVariableOp_2�2batch_normalization_1/batchnorm/mul/ReadVariableOp�.batch_normalization_2/batchnorm/ReadVariableOp�0batch_normalization_2/batchnorm/ReadVariableOp_1�0batch_normalization_2/batchnorm/ReadVariableOp_2�2batch_normalization_2/batchnorm/mul/ReadVariableOp�.batch_normalization_3/batchnorm/ReadVariableOp�0batch_normalization_3/batchnorm/ReadVariableOp_1�0batch_normalization_3/batchnorm/ReadVariableOp_2�2batch_normalization_3/batchnorm/mul/ReadVariableOp�dense_1/BiasAdd/ReadVariableOp�dense_1/MatMul/ReadVariableOp�-dense_1/kernel/Regularizer/Abs/ReadVariableOp�&stream_0_conv_1/BiasAdd/ReadVariableOp�2stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp�5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp�&stream_0_conv_2/BiasAdd/ReadVariableOp�2stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOp�8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp�&stream_0_conv_3/BiasAdd/ReadVariableOp�2stream_0_conv_3/conv1d/ExpandDims_1/ReadVariableOp�5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp�
stream_0_input_drop/IdentityIdentityinputs_0*
T0*+
_output_shapes
:���������}2
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
T0*/
_output_shapes
:���������}2#
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
T0*/
_output_shapes
:���������}@*
paddingSAME*
strides
2
stream_0_conv_1/conv1d�
stream_0_conv_1/conv1d/SqueezeSqueezestream_0_conv_1/conv1d:output:0*
T0*+
_output_shapes
:���������}@*
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
T0*+
_output_shapes
:���������}@2
stream_0_conv_1/BiasAdd�
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
T0*+
_output_shapes
:���������}@2%
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
T0*+
_output_shapes
:���������}@2%
#batch_normalization/batchnorm/add_1�
activation/ReluRelu'batch_normalization/batchnorm/add_1:z:0*
T0*+
_output_shapes
:���������}@2
activation/Relu�
stream_0_drop_1/IdentityIdentityactivation/Relu:activations:0*
T0*+
_output_shapes
:���������}@2
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
T0*/
_output_shapes
:���������}@2#
!stream_0_conv_2/conv1d/ExpandDims�
2stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp;stream_0_conv_2_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:@�*
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
T0*'
_output_shapes
:@�2%
#stream_0_conv_2/conv1d/ExpandDims_1�
stream_0_conv_2/conv1dConv2D*stream_0_conv_2/conv1d/ExpandDims:output:0,stream_0_conv_2/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:���������}�*
paddingSAME*
strides
2
stream_0_conv_2/conv1d�
stream_0_conv_2/conv1d/SqueezeSqueezestream_0_conv_2/conv1d:output:0*
T0*,
_output_shapes
:���������}�*
squeeze_dims

���������2 
stream_0_conv_2/conv1d/Squeeze�
&stream_0_conv_2/BiasAdd/ReadVariableOpReadVariableOp/stream_0_conv_2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02(
&stream_0_conv_2/BiasAdd/ReadVariableOp�
stream_0_conv_2/BiasAddBiasAdd'stream_0_conv_2/conv1d/Squeeze:output:0.stream_0_conv_2/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������}�2
stream_0_conv_2/BiasAdd�
.batch_normalization_1/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_1_batchnorm_readvariableop_resource*
_output_shapes	
:�*
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
T0*
_output_shapes	
:�2%
#batch_normalization_1/batchnorm/add�
%batch_normalization_1/batchnorm/RsqrtRsqrt'batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes	
:�2'
%batch_normalization_1/batchnorm/Rsqrt�
2batch_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype024
2batch_normalization_1/batchnorm/mul/ReadVariableOp�
#batch_normalization_1/batchnorm/mulMul)batch_normalization_1/batchnorm/Rsqrt:y:0:batch_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2%
#batch_normalization_1/batchnorm/mul�
%batch_normalization_1/batchnorm/mul_1Mul stream_0_conv_2/BiasAdd:output:0'batch_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:���������}�2'
%batch_normalization_1/batchnorm/mul_1�
0batch_normalization_1/batchnorm/ReadVariableOp_1ReadVariableOp9batch_normalization_1_batchnorm_readvariableop_1_resource*
_output_shapes	
:�*
dtype022
0batch_normalization_1/batchnorm/ReadVariableOp_1�
%batch_normalization_1/batchnorm/mul_2Mul8batch_normalization_1/batchnorm/ReadVariableOp_1:value:0'batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes	
:�2'
%batch_normalization_1/batchnorm/mul_2�
0batch_normalization_1/batchnorm/ReadVariableOp_2ReadVariableOp9batch_normalization_1_batchnorm_readvariableop_2_resource*
_output_shapes	
:�*
dtype022
0batch_normalization_1/batchnorm/ReadVariableOp_2�
#batch_normalization_1/batchnorm/subSub8batch_normalization_1/batchnorm/ReadVariableOp_2:value:0)batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�2%
#batch_normalization_1/batchnorm/sub�
%batch_normalization_1/batchnorm/add_1AddV2)batch_normalization_1/batchnorm/mul_1:z:0'batch_normalization_1/batchnorm/sub:z:0*
T0*,
_output_shapes
:���������}�2'
%batch_normalization_1/batchnorm/add_1�
activation_1/ReluRelu)batch_normalization_1/batchnorm/add_1:z:0*
T0*,
_output_shapes
:���������}�2
activation_1/Relu�
stream_0_drop_2/IdentityIdentityactivation_1/Relu:activations:0*
T0*,
_output_shapes
:���������}�2
stream_0_drop_2/Identity�
%stream_0_conv_3/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2'
%stream_0_conv_3/conv1d/ExpandDims/dim�
!stream_0_conv_3/conv1d/ExpandDims
ExpandDims!stream_0_drop_2/Identity:output:0.stream_0_conv_3/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:���������}�2#
!stream_0_conv_3/conv1d/ExpandDims�
2stream_0_conv_3/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp;stream_0_conv_3_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:��*
dtype024
2stream_0_conv_3/conv1d/ExpandDims_1/ReadVariableOp�
'stream_0_conv_3/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2)
'stream_0_conv_3/conv1d/ExpandDims_1/dim�
#stream_0_conv_3/conv1d/ExpandDims_1
ExpandDims:stream_0_conv_3/conv1d/ExpandDims_1/ReadVariableOp:value:00stream_0_conv_3/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:��2%
#stream_0_conv_3/conv1d/ExpandDims_1�
stream_0_conv_3/conv1dConv2D*stream_0_conv_3/conv1d/ExpandDims:output:0,stream_0_conv_3/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:���������}�*
paddingSAME*
strides
2
stream_0_conv_3/conv1d�
stream_0_conv_3/conv1d/SqueezeSqueezestream_0_conv_3/conv1d:output:0*
T0*,
_output_shapes
:���������}�*
squeeze_dims

���������2 
stream_0_conv_3/conv1d/Squeeze�
&stream_0_conv_3/BiasAdd/ReadVariableOpReadVariableOp/stream_0_conv_3_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02(
&stream_0_conv_3/BiasAdd/ReadVariableOp�
stream_0_conv_3/BiasAddBiasAdd'stream_0_conv_3/conv1d/Squeeze:output:0.stream_0_conv_3/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������}�2
stream_0_conv_3/BiasAdd�
.batch_normalization_2/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_2_batchnorm_readvariableop_resource*
_output_shapes	
:�*
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
T0*
_output_shapes	
:�2%
#batch_normalization_2/batchnorm/add�
%batch_normalization_2/batchnorm/RsqrtRsqrt'batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes	
:�2'
%batch_normalization_2/batchnorm/Rsqrt�
2batch_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype024
2batch_normalization_2/batchnorm/mul/ReadVariableOp�
#batch_normalization_2/batchnorm/mulMul)batch_normalization_2/batchnorm/Rsqrt:y:0:batch_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2%
#batch_normalization_2/batchnorm/mul�
%batch_normalization_2/batchnorm/mul_1Mul stream_0_conv_3/BiasAdd:output:0'batch_normalization_2/batchnorm/mul:z:0*
T0*,
_output_shapes
:���������}�2'
%batch_normalization_2/batchnorm/mul_1�
0batch_normalization_2/batchnorm/ReadVariableOp_1ReadVariableOp9batch_normalization_2_batchnorm_readvariableop_1_resource*
_output_shapes	
:�*
dtype022
0batch_normalization_2/batchnorm/ReadVariableOp_1�
%batch_normalization_2/batchnorm/mul_2Mul8batch_normalization_2/batchnorm/ReadVariableOp_1:value:0'batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes	
:�2'
%batch_normalization_2/batchnorm/mul_2�
0batch_normalization_2/batchnorm/ReadVariableOp_2ReadVariableOp9batch_normalization_2_batchnorm_readvariableop_2_resource*
_output_shapes	
:�*
dtype022
0batch_normalization_2/batchnorm/ReadVariableOp_2�
#batch_normalization_2/batchnorm/subSub8batch_normalization_2/batchnorm/ReadVariableOp_2:value:0)batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�2%
#batch_normalization_2/batchnorm/sub�
%batch_normalization_2/batchnorm/add_1AddV2)batch_normalization_2/batchnorm/mul_1:z:0'batch_normalization_2/batchnorm/sub:z:0*
T0*,
_output_shapes
:���������}�2'
%batch_normalization_2/batchnorm/add_1�
activation_2/ReluRelu)batch_normalization_2/batchnorm/add_1:z:0*
T0*,
_output_shapes
:���������}�2
activation_2/Relu�
stream_0_drop_3/IdentityIdentityactivation_2/Relu:activations:0*
T0*,
_output_shapes
:���������}�2
stream_0_drop_3/Identity�
/global_average_pooling1d/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :21
/global_average_pooling1d/Mean/reduction_indices�
global_average_pooling1d/MeanMean!stream_0_drop_3/Identity:output:08global_average_pooling1d/Mean/reduction_indices:output:0*
T0*(
_output_shapes
:����������2
global_average_pooling1d/Mean�
dense_1_dropout/IdentityIdentity&global_average_pooling1d/Mean:output:0*
T0*(
_output_shapes
:����������2
dense_1_dropout/Identity�
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	�T*
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
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;stream_0_conv_2_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:@�*
dtype02:
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp�
)stream_0_conv_2/kernel/Regularizer/SquareSquare@stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*#
_output_shapes
:@�2+
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
5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp;stream_0_conv_3_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:��*
dtype027
5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp�
&stream_0_conv_3/kernel/Regularizer/AbsAbs=stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*$
_output_shapes
:��2(
&stream_0_conv_3/kernel/Regularizer/Abs�
(stream_0_conv_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_0_conv_3/kernel/Regularizer/Const�
&stream_0_conv_3/kernel/Regularizer/SumSum*stream_0_conv_3/kernel/Regularizer/Abs:y:01stream_0_conv_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_3/kernel/Regularizer/Sum�
(stream_0_conv_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2*
(stream_0_conv_3/kernel/Regularizer/mul/x�
&stream_0_conv_3/kernel/Regularizer/mulMul1stream_0_conv_3/kernel/Regularizer/mul/x:output:0/stream_0_conv_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_3/kernel/Regularizer/mul�
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	�T*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOp�
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	�T2 
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
NoOpNoOp-^batch_normalization/batchnorm/ReadVariableOp/^batch_normalization/batchnorm/ReadVariableOp_1/^batch_normalization/batchnorm/ReadVariableOp_21^batch_normalization/batchnorm/mul/ReadVariableOp/^batch_normalization_1/batchnorm/ReadVariableOp1^batch_normalization_1/batchnorm/ReadVariableOp_11^batch_normalization_1/batchnorm/ReadVariableOp_23^batch_normalization_1/batchnorm/mul/ReadVariableOp/^batch_normalization_2/batchnorm/ReadVariableOp1^batch_normalization_2/batchnorm/ReadVariableOp_11^batch_normalization_2/batchnorm/ReadVariableOp_23^batch_normalization_2/batchnorm/mul/ReadVariableOp/^batch_normalization_3/batchnorm/ReadVariableOp1^batch_normalization_3/batchnorm/ReadVariableOp_11^batch_normalization_3/batchnorm/ReadVariableOp_23^batch_normalization_3/batchnorm/mul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp.^dense_1/kernel/Regularizer/Abs/ReadVariableOp'^stream_0_conv_1/BiasAdd/ReadVariableOp3^stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp'^stream_0_conv_2/BiasAdd/ReadVariableOp3^stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOp9^stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp'^stream_0_conv_3/BiasAdd/ReadVariableOp3^stream_0_conv_3/conv1d/ExpandDims_1/ReadVariableOp6^stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:���������}: : : : : : : : : : : : : : : : : : : : : : : : 2\
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
:���������}
"
_user_specified_name
inputs/0
��
� 
D__inference_model_1_layer_call_and_return_conditional_losses_8909719

inputs[
Ebasemodel_stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource:@G
9basemodel_stream_0_conv_1_biasadd_readvariableop_resource:@S
Ebasemodel_batch_normalization_assignmovingavg_readvariableop_resource:@U
Gbasemodel_batch_normalization_assignmovingavg_1_readvariableop_resource:@Q
Cbasemodel_batch_normalization_batchnorm_mul_readvariableop_resource:@M
?basemodel_batch_normalization_batchnorm_readvariableop_resource:@\
Ebasemodel_stream_0_conv_2_conv1d_expanddims_1_readvariableop_resource:@�H
9basemodel_stream_0_conv_2_biasadd_readvariableop_resource:	�V
Gbasemodel_batch_normalization_1_assignmovingavg_readvariableop_resource:	�X
Ibasemodel_batch_normalization_1_assignmovingavg_1_readvariableop_resource:	�T
Ebasemodel_batch_normalization_1_batchnorm_mul_readvariableop_resource:	�P
Abasemodel_batch_normalization_1_batchnorm_readvariableop_resource:	�]
Ebasemodel_stream_0_conv_3_conv1d_expanddims_1_readvariableop_resource:��H
9basemodel_stream_0_conv_3_biasadd_readvariableop_resource:	�V
Gbasemodel_batch_normalization_2_assignmovingavg_readvariableop_resource:	�X
Ibasemodel_batch_normalization_2_assignmovingavg_1_readvariableop_resource:	�T
Ebasemodel_batch_normalization_2_batchnorm_mul_readvariableop_resource:	�P
Abasemodel_batch_normalization_2_batchnorm_readvariableop_resource:	�C
0basemodel_dense_1_matmul_readvariableop_resource:	�T?
1basemodel_dense_1_biasadd_readvariableop_resource:TU
Gbasemodel_batch_normalization_3_assignmovingavg_readvariableop_resource:TW
Ibasemodel_batch_normalization_3_assignmovingavg_1_readvariableop_resource:TS
Ebasemodel_batch_normalization_3_batchnorm_mul_readvariableop_resource:TO
Abasemodel_batch_normalization_3_batchnorm_readvariableop_resource:T
identity��-basemodel/batch_normalization/AssignMovingAvg�<basemodel/batch_normalization/AssignMovingAvg/ReadVariableOp�/basemodel/batch_normalization/AssignMovingAvg_1�>basemodel/batch_normalization/AssignMovingAvg_1/ReadVariableOp�6basemodel/batch_normalization/batchnorm/ReadVariableOp�:basemodel/batch_normalization/batchnorm/mul/ReadVariableOp�/basemodel/batch_normalization_1/AssignMovingAvg�>basemodel/batch_normalization_1/AssignMovingAvg/ReadVariableOp�1basemodel/batch_normalization_1/AssignMovingAvg_1�@basemodel/batch_normalization_1/AssignMovingAvg_1/ReadVariableOp�8basemodel/batch_normalization_1/batchnorm/ReadVariableOp�<basemodel/batch_normalization_1/batchnorm/mul/ReadVariableOp�/basemodel/batch_normalization_2/AssignMovingAvg�>basemodel/batch_normalization_2/AssignMovingAvg/ReadVariableOp�1basemodel/batch_normalization_2/AssignMovingAvg_1�@basemodel/batch_normalization_2/AssignMovingAvg_1/ReadVariableOp�8basemodel/batch_normalization_2/batchnorm/ReadVariableOp�<basemodel/batch_normalization_2/batchnorm/mul/ReadVariableOp�/basemodel/batch_normalization_3/AssignMovingAvg�>basemodel/batch_normalization_3/AssignMovingAvg/ReadVariableOp�1basemodel/batch_normalization_3/AssignMovingAvg_1�@basemodel/batch_normalization_3/AssignMovingAvg_1/ReadVariableOp�8basemodel/batch_normalization_3/batchnorm/ReadVariableOp�<basemodel/batch_normalization_3/batchnorm/mul/ReadVariableOp�(basemodel/dense_1/BiasAdd/ReadVariableOp�'basemodel/dense_1/MatMul/ReadVariableOp�0basemodel/stream_0_conv_1/BiasAdd/ReadVariableOp�<basemodel/stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp�0basemodel/stream_0_conv_2/BiasAdd/ReadVariableOp�<basemodel/stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOp�0basemodel/stream_0_conv_3/BiasAdd/ReadVariableOp�<basemodel/stream_0_conv_3/conv1d/ExpandDims_1/ReadVariableOp�-dense_1/kernel/Regularizer/Abs/ReadVariableOp�5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp�8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp�5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp�
+basemodel/stream_0_input_drop/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2-
+basemodel/stream_0_input_drop/dropout/Const�
)basemodel/stream_0_input_drop/dropout/MulMulinputs4basemodel/stream_0_input_drop/dropout/Const:output:0*
T0*+
_output_shapes
:���������}2+
)basemodel/stream_0_input_drop/dropout/Mul�
+basemodel/stream_0_input_drop/dropout/ShapeShapeinputs*
T0*
_output_shapes
:2-
+basemodel/stream_0_input_drop/dropout/Shape�
Bbasemodel/stream_0_input_drop/dropout/random_uniform/RandomUniformRandomUniform4basemodel/stream_0_input_drop/dropout/Shape:output:0*
T0*+
_output_shapes
:���������}*
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
T0*+
_output_shapes
:���������}24
2basemodel/stream_0_input_drop/dropout/GreaterEqual�
*basemodel/stream_0_input_drop/dropout/CastCast6basemodel/stream_0_input_drop/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:���������}2,
*basemodel/stream_0_input_drop/dropout/Cast�
+basemodel/stream_0_input_drop/dropout/Mul_1Mul-basemodel/stream_0_input_drop/dropout/Mul:z:0.basemodel/stream_0_input_drop/dropout/Cast:y:0*
T0*+
_output_shapes
:���������}2-
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
T0*/
_output_shapes
:���������}2-
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
T0*/
_output_shapes
:���������}@*
paddingSAME*
strides
2"
 basemodel/stream_0_conv_1/conv1d�
(basemodel/stream_0_conv_1/conv1d/SqueezeSqueeze)basemodel/stream_0_conv_1/conv1d:output:0*
T0*+
_output_shapes
:���������}@*
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
T0*+
_output_shapes
:���������}@2#
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
:@*
	keep_dims(2,
*basemodel/batch_normalization/moments/mean�
2basemodel/batch_normalization/moments/StopGradientStopGradient3basemodel/batch_normalization/moments/mean:output:0*
T0*"
_output_shapes
:@24
2basemodel/batch_normalization/moments/StopGradient�
7basemodel/batch_normalization/moments/SquaredDifferenceSquaredDifference*basemodel/stream_0_conv_1/BiasAdd:output:0;basemodel/batch_normalization/moments/StopGradient:output:0*
T0*+
_output_shapes
:���������}@29
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
T0*+
_output_shapes
:���������}@2/
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
T0*+
_output_shapes
:���������}@2/
-basemodel/batch_normalization/batchnorm/add_1�
basemodel/activation/ReluRelu1basemodel/batch_normalization/batchnorm/add_1:z:0*
T0*+
_output_shapes
:���������}@2
basemodel/activation/Relu�
'basemodel/stream_0_drop_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2)
'basemodel/stream_0_drop_1/dropout/Const�
%basemodel/stream_0_drop_1/dropout/MulMul'basemodel/activation/Relu:activations:00basemodel/stream_0_drop_1/dropout/Const:output:0*
T0*+
_output_shapes
:���������}@2'
%basemodel/stream_0_drop_1/dropout/Mul�
'basemodel/stream_0_drop_1/dropout/ShapeShape'basemodel/activation/Relu:activations:0*
T0*
_output_shapes
:2)
'basemodel/stream_0_drop_1/dropout/Shape�
>basemodel/stream_0_drop_1/dropout/random_uniform/RandomUniformRandomUniform0basemodel/stream_0_drop_1/dropout/Shape:output:0*
T0*+
_output_shapes
:���������}@*
dtype0*
seed�*
seed2�2@
>basemodel/stream_0_drop_1/dropout/random_uniform/RandomUniform�
0basemodel/stream_0_drop_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���>22
0basemodel/stream_0_drop_1/dropout/GreaterEqual/y�
.basemodel/stream_0_drop_1/dropout/GreaterEqualGreaterEqualGbasemodel/stream_0_drop_1/dropout/random_uniform/RandomUniform:output:09basemodel/stream_0_drop_1/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:���������}@20
.basemodel/stream_0_drop_1/dropout/GreaterEqual�
&basemodel/stream_0_drop_1/dropout/CastCast2basemodel/stream_0_drop_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:���������}@2(
&basemodel/stream_0_drop_1/dropout/Cast�
'basemodel/stream_0_drop_1/dropout/Mul_1Mul)basemodel/stream_0_drop_1/dropout/Mul:z:0*basemodel/stream_0_drop_1/dropout/Cast:y:0*
T0*+
_output_shapes
:���������}@2)
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
T0*/
_output_shapes
:���������}@2-
+basemodel/stream_0_conv_2/conv1d/ExpandDims�
<basemodel/stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpEbasemodel_stream_0_conv_2_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:@�*
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
T0*'
_output_shapes
:@�2/
-basemodel/stream_0_conv_2/conv1d/ExpandDims_1�
 basemodel/stream_0_conv_2/conv1dConv2D4basemodel/stream_0_conv_2/conv1d/ExpandDims:output:06basemodel/stream_0_conv_2/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:���������}�*
paddingSAME*
strides
2"
 basemodel/stream_0_conv_2/conv1d�
(basemodel/stream_0_conv_2/conv1d/SqueezeSqueeze)basemodel/stream_0_conv_2/conv1d:output:0*
T0*,
_output_shapes
:���������}�*
squeeze_dims

���������2*
(basemodel/stream_0_conv_2/conv1d/Squeeze�
0basemodel/stream_0_conv_2/BiasAdd/ReadVariableOpReadVariableOp9basemodel_stream_0_conv_2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype022
0basemodel/stream_0_conv_2/BiasAdd/ReadVariableOp�
!basemodel/stream_0_conv_2/BiasAddBiasAdd1basemodel/stream_0_conv_2/conv1d/Squeeze:output:08basemodel/stream_0_conv_2/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������}�2#
!basemodel/stream_0_conv_2/BiasAdd�
>basemodel/batch_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2@
>basemodel/batch_normalization_1/moments/mean/reduction_indices�
,basemodel/batch_normalization_1/moments/meanMean*basemodel/stream_0_conv_2/BiasAdd:output:0Gbasemodel/batch_normalization_1/moments/mean/reduction_indices:output:0*
T0*#
_output_shapes
:�*
	keep_dims(2.
,basemodel/batch_normalization_1/moments/mean�
4basemodel/batch_normalization_1/moments/StopGradientStopGradient5basemodel/batch_normalization_1/moments/mean:output:0*
T0*#
_output_shapes
:�26
4basemodel/batch_normalization_1/moments/StopGradient�
9basemodel/batch_normalization_1/moments/SquaredDifferenceSquaredDifference*basemodel/stream_0_conv_2/BiasAdd:output:0=basemodel/batch_normalization_1/moments/StopGradient:output:0*
T0*,
_output_shapes
:���������}�2;
9basemodel/batch_normalization_1/moments/SquaredDifference�
Bbasemodel/batch_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2D
Bbasemodel/batch_normalization_1/moments/variance/reduction_indices�
0basemodel/batch_normalization_1/moments/varianceMean=basemodel/batch_normalization_1/moments/SquaredDifference:z:0Kbasemodel/batch_normalization_1/moments/variance/reduction_indices:output:0*
T0*#
_output_shapes
:�*
	keep_dims(22
0basemodel/batch_normalization_1/moments/variance�
/basemodel/batch_normalization_1/moments/SqueezeSqueeze5basemodel/batch_normalization_1/moments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 21
/basemodel/batch_normalization_1/moments/Squeeze�
1basemodel/batch_normalization_1/moments/Squeeze_1Squeeze9basemodel/batch_normalization_1/moments/variance:output:0*
T0*
_output_shapes	
:�*
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
>basemodel/batch_normalization_1/AssignMovingAvg/ReadVariableOpReadVariableOpGbasemodel_batch_normalization_1_assignmovingavg_readvariableop_resource*
_output_shapes	
:�*
dtype02@
>basemodel/batch_normalization_1/AssignMovingAvg/ReadVariableOp�
3basemodel/batch_normalization_1/AssignMovingAvg/subSubFbasemodel/batch_normalization_1/AssignMovingAvg/ReadVariableOp:value:08basemodel/batch_normalization_1/moments/Squeeze:output:0*
T0*
_output_shapes	
:�25
3basemodel/batch_normalization_1/AssignMovingAvg/sub�
3basemodel/batch_normalization_1/AssignMovingAvg/mulMul7basemodel/batch_normalization_1/AssignMovingAvg/sub:z:0>basemodel/batch_normalization_1/AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:�25
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
@basemodel/batch_normalization_1/AssignMovingAvg_1/ReadVariableOpReadVariableOpIbasemodel_batch_normalization_1_assignmovingavg_1_readvariableop_resource*
_output_shapes	
:�*
dtype02B
@basemodel/batch_normalization_1/AssignMovingAvg_1/ReadVariableOp�
5basemodel/batch_normalization_1/AssignMovingAvg_1/subSubHbasemodel/batch_normalization_1/AssignMovingAvg_1/ReadVariableOp:value:0:basemodel/batch_normalization_1/moments/Squeeze_1:output:0*
T0*
_output_shapes	
:�27
5basemodel/batch_normalization_1/AssignMovingAvg_1/sub�
5basemodel/batch_normalization_1/AssignMovingAvg_1/mulMul9basemodel/batch_normalization_1/AssignMovingAvg_1/sub:z:0@basemodel/batch_normalization_1/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:�27
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
T0*
_output_shapes	
:�2/
-basemodel/batch_normalization_1/batchnorm/add�
/basemodel/batch_normalization_1/batchnorm/RsqrtRsqrt1basemodel/batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes	
:�21
/basemodel/batch_normalization_1/batchnorm/Rsqrt�
<basemodel/batch_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOpEbasemodel_batch_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype02>
<basemodel/batch_normalization_1/batchnorm/mul/ReadVariableOp�
-basemodel/batch_normalization_1/batchnorm/mulMul3basemodel/batch_normalization_1/batchnorm/Rsqrt:y:0Dbasemodel/batch_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2/
-basemodel/batch_normalization_1/batchnorm/mul�
/basemodel/batch_normalization_1/batchnorm/mul_1Mul*basemodel/stream_0_conv_2/BiasAdd:output:01basemodel/batch_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:���������}�21
/basemodel/batch_normalization_1/batchnorm/mul_1�
/basemodel/batch_normalization_1/batchnorm/mul_2Mul8basemodel/batch_normalization_1/moments/Squeeze:output:01basemodel/batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes	
:�21
/basemodel/batch_normalization_1/batchnorm/mul_2�
8basemodel/batch_normalization_1/batchnorm/ReadVariableOpReadVariableOpAbasemodel_batch_normalization_1_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype02:
8basemodel/batch_normalization_1/batchnorm/ReadVariableOp�
-basemodel/batch_normalization_1/batchnorm/subSub@basemodel/batch_normalization_1/batchnorm/ReadVariableOp:value:03basemodel/batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�2/
-basemodel/batch_normalization_1/batchnorm/sub�
/basemodel/batch_normalization_1/batchnorm/add_1AddV23basemodel/batch_normalization_1/batchnorm/mul_1:z:01basemodel/batch_normalization_1/batchnorm/sub:z:0*
T0*,
_output_shapes
:���������}�21
/basemodel/batch_normalization_1/batchnorm/add_1�
basemodel/activation_1/ReluRelu3basemodel/batch_normalization_1/batchnorm/add_1:z:0*
T0*,
_output_shapes
:���������}�2
basemodel/activation_1/Relu�
'basemodel/stream_0_drop_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU�?2)
'basemodel/stream_0_drop_2/dropout/Const�
%basemodel/stream_0_drop_2/dropout/MulMul)basemodel/activation_1/Relu:activations:00basemodel/stream_0_drop_2/dropout/Const:output:0*
T0*,
_output_shapes
:���������}�2'
%basemodel/stream_0_drop_2/dropout/Mul�
'basemodel/stream_0_drop_2/dropout/ShapeShape)basemodel/activation_1/Relu:activations:0*
T0*
_output_shapes
:2)
'basemodel/stream_0_drop_2/dropout/Shape�
>basemodel/stream_0_drop_2/dropout/random_uniform/RandomUniformRandomUniform0basemodel/stream_0_drop_2/dropout/Shape:output:0*
T0*,
_output_shapes
:���������}�*
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
:���������}�20
.basemodel/stream_0_drop_2/dropout/GreaterEqual�
&basemodel/stream_0_drop_2/dropout/CastCast2basemodel/stream_0_drop_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:���������}�2(
&basemodel/stream_0_drop_2/dropout/Cast�
'basemodel/stream_0_drop_2/dropout/Mul_1Mul)basemodel/stream_0_drop_2/dropout/Mul:z:0*basemodel/stream_0_drop_2/dropout/Cast:y:0*
T0*,
_output_shapes
:���������}�2)
'basemodel/stream_0_drop_2/dropout/Mul_1�
/basemodel/stream_0_conv_3/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������21
/basemodel/stream_0_conv_3/conv1d/ExpandDims/dim�
+basemodel/stream_0_conv_3/conv1d/ExpandDims
ExpandDims+basemodel/stream_0_drop_2/dropout/Mul_1:z:08basemodel/stream_0_conv_3/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:���������}�2-
+basemodel/stream_0_conv_3/conv1d/ExpandDims�
<basemodel/stream_0_conv_3/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpEbasemodel_stream_0_conv_3_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:��*
dtype02>
<basemodel/stream_0_conv_3/conv1d/ExpandDims_1/ReadVariableOp�
1basemodel/stream_0_conv_3/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 23
1basemodel/stream_0_conv_3/conv1d/ExpandDims_1/dim�
-basemodel/stream_0_conv_3/conv1d/ExpandDims_1
ExpandDimsDbasemodel/stream_0_conv_3/conv1d/ExpandDims_1/ReadVariableOp:value:0:basemodel/stream_0_conv_3/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:��2/
-basemodel/stream_0_conv_3/conv1d/ExpandDims_1�
 basemodel/stream_0_conv_3/conv1dConv2D4basemodel/stream_0_conv_3/conv1d/ExpandDims:output:06basemodel/stream_0_conv_3/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:���������}�*
paddingSAME*
strides
2"
 basemodel/stream_0_conv_3/conv1d�
(basemodel/stream_0_conv_3/conv1d/SqueezeSqueeze)basemodel/stream_0_conv_3/conv1d:output:0*
T0*,
_output_shapes
:���������}�*
squeeze_dims

���������2*
(basemodel/stream_0_conv_3/conv1d/Squeeze�
0basemodel/stream_0_conv_3/BiasAdd/ReadVariableOpReadVariableOp9basemodel_stream_0_conv_3_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype022
0basemodel/stream_0_conv_3/BiasAdd/ReadVariableOp�
!basemodel/stream_0_conv_3/BiasAddBiasAdd1basemodel/stream_0_conv_3/conv1d/Squeeze:output:08basemodel/stream_0_conv_3/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������}�2#
!basemodel/stream_0_conv_3/BiasAdd�
>basemodel/batch_normalization_2/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2@
>basemodel/batch_normalization_2/moments/mean/reduction_indices�
,basemodel/batch_normalization_2/moments/meanMean*basemodel/stream_0_conv_3/BiasAdd:output:0Gbasemodel/batch_normalization_2/moments/mean/reduction_indices:output:0*
T0*#
_output_shapes
:�*
	keep_dims(2.
,basemodel/batch_normalization_2/moments/mean�
4basemodel/batch_normalization_2/moments/StopGradientStopGradient5basemodel/batch_normalization_2/moments/mean:output:0*
T0*#
_output_shapes
:�26
4basemodel/batch_normalization_2/moments/StopGradient�
9basemodel/batch_normalization_2/moments/SquaredDifferenceSquaredDifference*basemodel/stream_0_conv_3/BiasAdd:output:0=basemodel/batch_normalization_2/moments/StopGradient:output:0*
T0*,
_output_shapes
:���������}�2;
9basemodel/batch_normalization_2/moments/SquaredDifference�
Bbasemodel/batch_normalization_2/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2D
Bbasemodel/batch_normalization_2/moments/variance/reduction_indices�
0basemodel/batch_normalization_2/moments/varianceMean=basemodel/batch_normalization_2/moments/SquaredDifference:z:0Kbasemodel/batch_normalization_2/moments/variance/reduction_indices:output:0*
T0*#
_output_shapes
:�*
	keep_dims(22
0basemodel/batch_normalization_2/moments/variance�
/basemodel/batch_normalization_2/moments/SqueezeSqueeze5basemodel/batch_normalization_2/moments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 21
/basemodel/batch_normalization_2/moments/Squeeze�
1basemodel/batch_normalization_2/moments/Squeeze_1Squeeze9basemodel/batch_normalization_2/moments/variance:output:0*
T0*
_output_shapes	
:�*
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
>basemodel/batch_normalization_2/AssignMovingAvg/ReadVariableOpReadVariableOpGbasemodel_batch_normalization_2_assignmovingavg_readvariableop_resource*
_output_shapes	
:�*
dtype02@
>basemodel/batch_normalization_2/AssignMovingAvg/ReadVariableOp�
3basemodel/batch_normalization_2/AssignMovingAvg/subSubFbasemodel/batch_normalization_2/AssignMovingAvg/ReadVariableOp:value:08basemodel/batch_normalization_2/moments/Squeeze:output:0*
T0*
_output_shapes	
:�25
3basemodel/batch_normalization_2/AssignMovingAvg/sub�
3basemodel/batch_normalization_2/AssignMovingAvg/mulMul7basemodel/batch_normalization_2/AssignMovingAvg/sub:z:0>basemodel/batch_normalization_2/AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:�25
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
@basemodel/batch_normalization_2/AssignMovingAvg_1/ReadVariableOpReadVariableOpIbasemodel_batch_normalization_2_assignmovingavg_1_readvariableop_resource*
_output_shapes	
:�*
dtype02B
@basemodel/batch_normalization_2/AssignMovingAvg_1/ReadVariableOp�
5basemodel/batch_normalization_2/AssignMovingAvg_1/subSubHbasemodel/batch_normalization_2/AssignMovingAvg_1/ReadVariableOp:value:0:basemodel/batch_normalization_2/moments/Squeeze_1:output:0*
T0*
_output_shapes	
:�27
5basemodel/batch_normalization_2/AssignMovingAvg_1/sub�
5basemodel/batch_normalization_2/AssignMovingAvg_1/mulMul9basemodel/batch_normalization_2/AssignMovingAvg_1/sub:z:0@basemodel/batch_normalization_2/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:�27
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
T0*
_output_shapes	
:�2/
-basemodel/batch_normalization_2/batchnorm/add�
/basemodel/batch_normalization_2/batchnorm/RsqrtRsqrt1basemodel/batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes	
:�21
/basemodel/batch_normalization_2/batchnorm/Rsqrt�
<basemodel/batch_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOpEbasemodel_batch_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype02>
<basemodel/batch_normalization_2/batchnorm/mul/ReadVariableOp�
-basemodel/batch_normalization_2/batchnorm/mulMul3basemodel/batch_normalization_2/batchnorm/Rsqrt:y:0Dbasemodel/batch_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2/
-basemodel/batch_normalization_2/batchnorm/mul�
/basemodel/batch_normalization_2/batchnorm/mul_1Mul*basemodel/stream_0_conv_3/BiasAdd:output:01basemodel/batch_normalization_2/batchnorm/mul:z:0*
T0*,
_output_shapes
:���������}�21
/basemodel/batch_normalization_2/batchnorm/mul_1�
/basemodel/batch_normalization_2/batchnorm/mul_2Mul8basemodel/batch_normalization_2/moments/Squeeze:output:01basemodel/batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes	
:�21
/basemodel/batch_normalization_2/batchnorm/mul_2�
8basemodel/batch_normalization_2/batchnorm/ReadVariableOpReadVariableOpAbasemodel_batch_normalization_2_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype02:
8basemodel/batch_normalization_2/batchnorm/ReadVariableOp�
-basemodel/batch_normalization_2/batchnorm/subSub@basemodel/batch_normalization_2/batchnorm/ReadVariableOp:value:03basemodel/batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�2/
-basemodel/batch_normalization_2/batchnorm/sub�
/basemodel/batch_normalization_2/batchnorm/add_1AddV23basemodel/batch_normalization_2/batchnorm/mul_1:z:01basemodel/batch_normalization_2/batchnorm/sub:z:0*
T0*,
_output_shapes
:���������}�21
/basemodel/batch_normalization_2/batchnorm/add_1�
basemodel/activation_2/ReluRelu3basemodel/batch_normalization_2/batchnorm/add_1:z:0*
T0*,
_output_shapes
:���������}�2
basemodel/activation_2/Relu�
'basemodel/stream_0_drop_3/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2)
'basemodel/stream_0_drop_3/dropout/Const�
%basemodel/stream_0_drop_3/dropout/MulMul)basemodel/activation_2/Relu:activations:00basemodel/stream_0_drop_3/dropout/Const:output:0*
T0*,
_output_shapes
:���������}�2'
%basemodel/stream_0_drop_3/dropout/Mul�
'basemodel/stream_0_drop_3/dropout/ShapeShape)basemodel/activation_2/Relu:activations:0*
T0*
_output_shapes
:2)
'basemodel/stream_0_drop_3/dropout/Shape�
>basemodel/stream_0_drop_3/dropout/random_uniform/RandomUniformRandomUniform0basemodel/stream_0_drop_3/dropout/Shape:output:0*
T0*,
_output_shapes
:���������}�*
dtype0*
seed�*
seed2�2@
>basemodel/stream_0_drop_3/dropout/random_uniform/RandomUniform�
0basemodel/stream_0_drop_3/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?22
0basemodel/stream_0_drop_3/dropout/GreaterEqual/y�
.basemodel/stream_0_drop_3/dropout/GreaterEqualGreaterEqualGbasemodel/stream_0_drop_3/dropout/random_uniform/RandomUniform:output:09basemodel/stream_0_drop_3/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:���������}�20
.basemodel/stream_0_drop_3/dropout/GreaterEqual�
&basemodel/stream_0_drop_3/dropout/CastCast2basemodel/stream_0_drop_3/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:���������}�2(
&basemodel/stream_0_drop_3/dropout/Cast�
'basemodel/stream_0_drop_3/dropout/Mul_1Mul)basemodel/stream_0_drop_3/dropout/Mul:z:0*basemodel/stream_0_drop_3/dropout/Cast:y:0*
T0*,
_output_shapes
:���������}�2)
'basemodel/stream_0_drop_3/dropout/Mul_1�
9basemodel/global_average_pooling1d/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2;
9basemodel/global_average_pooling1d/Mean/reduction_indices�
'basemodel/global_average_pooling1d/MeanMean+basemodel/stream_0_drop_3/dropout/Mul_1:z:0Bbasemodel/global_average_pooling1d/Mean/reduction_indices:output:0*
T0*(
_output_shapes
:����������2)
'basemodel/global_average_pooling1d/Mean�
'basemodel/dense_1_dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2)
'basemodel/dense_1_dropout/dropout/Const�
%basemodel/dense_1_dropout/dropout/MulMul0basemodel/global_average_pooling1d/Mean:output:00basemodel/dense_1_dropout/dropout/Const:output:0*
T0*(
_output_shapes
:����������2'
%basemodel/dense_1_dropout/dropout/Mul�
'basemodel/dense_1_dropout/dropout/ShapeShape0basemodel/global_average_pooling1d/Mean:output:0*
T0*
_output_shapes
:2)
'basemodel/dense_1_dropout/dropout/Shape�
>basemodel/dense_1_dropout/dropout/random_uniform/RandomUniformRandomUniform0basemodel/dense_1_dropout/dropout/Shape:output:0*
T0*(
_output_shapes
:����������*
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
T0*(
_output_shapes
:����������20
.basemodel/dense_1_dropout/dropout/GreaterEqual�
&basemodel/dense_1_dropout/dropout/CastCast2basemodel/dense_1_dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:����������2(
&basemodel/dense_1_dropout/dropout/Cast�
'basemodel/dense_1_dropout/dropout/Mul_1Mul)basemodel/dense_1_dropout/dropout/Mul:z:0*basemodel/dense_1_dropout/dropout/Cast:y:0*
T0*(
_output_shapes
:����������2)
'basemodel/dense_1_dropout/dropout/Mul_1�
'basemodel/dense_1/MatMul/ReadVariableOpReadVariableOp0basemodel_dense_1_matmul_readvariableop_resource*
_output_shapes
:	�T*
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
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpEbasemodel_stream_0_conv_2_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:@�*
dtype02:
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp�
)stream_0_conv_2/kernel/Regularizer/SquareSquare@stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*#
_output_shapes
:@�2+
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
5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpEbasemodel_stream_0_conv_3_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:��*
dtype027
5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp�
&stream_0_conv_3/kernel/Regularizer/AbsAbs=stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*$
_output_shapes
:��2(
&stream_0_conv_3/kernel/Regularizer/Abs�
(stream_0_conv_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_0_conv_3/kernel/Regularizer/Const�
&stream_0_conv_3/kernel/Regularizer/SumSum*stream_0_conv_3/kernel/Regularizer/Abs:y:01stream_0_conv_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_3/kernel/Regularizer/Sum�
(stream_0_conv_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2*
(stream_0_conv_3/kernel/Regularizer/mul/x�
&stream_0_conv_3/kernel/Regularizer/mulMul1stream_0_conv_3/kernel/Regularizer/mul/x:output:0/stream_0_conv_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_3/kernel/Regularizer/mul�
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp0basemodel_dense_1_matmul_readvariableop_resource*
_output_shapes
:	�T*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOp�
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	�T2 
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
IdentityIdentity(basemodel/dense_activation_1/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:���������T2

Identity�
NoOpNoOp.^basemodel/batch_normalization/AssignMovingAvg=^basemodel/batch_normalization/AssignMovingAvg/ReadVariableOp0^basemodel/batch_normalization/AssignMovingAvg_1?^basemodel/batch_normalization/AssignMovingAvg_1/ReadVariableOp7^basemodel/batch_normalization/batchnorm/ReadVariableOp;^basemodel/batch_normalization/batchnorm/mul/ReadVariableOp0^basemodel/batch_normalization_1/AssignMovingAvg?^basemodel/batch_normalization_1/AssignMovingAvg/ReadVariableOp2^basemodel/batch_normalization_1/AssignMovingAvg_1A^basemodel/batch_normalization_1/AssignMovingAvg_1/ReadVariableOp9^basemodel/batch_normalization_1/batchnorm/ReadVariableOp=^basemodel/batch_normalization_1/batchnorm/mul/ReadVariableOp0^basemodel/batch_normalization_2/AssignMovingAvg?^basemodel/batch_normalization_2/AssignMovingAvg/ReadVariableOp2^basemodel/batch_normalization_2/AssignMovingAvg_1A^basemodel/batch_normalization_2/AssignMovingAvg_1/ReadVariableOp9^basemodel/batch_normalization_2/batchnorm/ReadVariableOp=^basemodel/batch_normalization_2/batchnorm/mul/ReadVariableOp0^basemodel/batch_normalization_3/AssignMovingAvg?^basemodel/batch_normalization_3/AssignMovingAvg/ReadVariableOp2^basemodel/batch_normalization_3/AssignMovingAvg_1A^basemodel/batch_normalization_3/AssignMovingAvg_1/ReadVariableOp9^basemodel/batch_normalization_3/batchnorm/ReadVariableOp=^basemodel/batch_normalization_3/batchnorm/mul/ReadVariableOp)^basemodel/dense_1/BiasAdd/ReadVariableOp(^basemodel/dense_1/MatMul/ReadVariableOp1^basemodel/stream_0_conv_1/BiasAdd/ReadVariableOp=^basemodel/stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp1^basemodel/stream_0_conv_2/BiasAdd/ReadVariableOp=^basemodel/stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOp1^basemodel/stream_0_conv_3/BiasAdd/ReadVariableOp=^basemodel/stream_0_conv_3/conv1d/ExpandDims_1/ReadVariableOp.^dense_1/kernel/Regularizer/Abs/ReadVariableOp6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp9^stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp6^stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:���������}: : : : : : : : : : : : : : : : : : : : : : : : 2^
-basemodel/batch_normalization/AssignMovingAvg-basemodel/batch_normalization/AssignMovingAvg2|
<basemodel/batch_normalization/AssignMovingAvg/ReadVariableOp<basemodel/batch_normalization/AssignMovingAvg/ReadVariableOp2b
/basemodel/batch_normalization/AssignMovingAvg_1/basemodel/batch_normalization/AssignMovingAvg_12�
>basemodel/batch_normalization/AssignMovingAvg_1/ReadVariableOp>basemodel/batch_normalization/AssignMovingAvg_1/ReadVariableOp2p
6basemodel/batch_normalization/batchnorm/ReadVariableOp6basemodel/batch_normalization/batchnorm/ReadVariableOp2x
:basemodel/batch_normalization/batchnorm/mul/ReadVariableOp:basemodel/batch_normalization/batchnorm/mul/ReadVariableOp2b
/basemodel/batch_normalization_1/AssignMovingAvg/basemodel/batch_normalization_1/AssignMovingAvg2�
>basemodel/batch_normalization_1/AssignMovingAvg/ReadVariableOp>basemodel/batch_normalization_1/AssignMovingAvg/ReadVariableOp2f
1basemodel/batch_normalization_1/AssignMovingAvg_11basemodel/batch_normalization_1/AssignMovingAvg_12�
@basemodel/batch_normalization_1/AssignMovingAvg_1/ReadVariableOp@basemodel/batch_normalization_1/AssignMovingAvg_1/ReadVariableOp2t
8basemodel/batch_normalization_1/batchnorm/ReadVariableOp8basemodel/batch_normalization_1/batchnorm/ReadVariableOp2|
<basemodel/batch_normalization_1/batchnorm/mul/ReadVariableOp<basemodel/batch_normalization_1/batchnorm/mul/ReadVariableOp2b
/basemodel/batch_normalization_2/AssignMovingAvg/basemodel/batch_normalization_2/AssignMovingAvg2�
>basemodel/batch_normalization_2/AssignMovingAvg/ReadVariableOp>basemodel/batch_normalization_2/AssignMovingAvg/ReadVariableOp2f
1basemodel/batch_normalization_2/AssignMovingAvg_11basemodel/batch_normalization_2/AssignMovingAvg_12�
@basemodel/batch_normalization_2/AssignMovingAvg_1/ReadVariableOp@basemodel/batch_normalization_2/AssignMovingAvg_1/ReadVariableOp2t
8basemodel/batch_normalization_2/batchnorm/ReadVariableOp8basemodel/batch_normalization_2/batchnorm/ReadVariableOp2|
<basemodel/batch_normalization_2/batchnorm/mul/ReadVariableOp<basemodel/batch_normalization_2/batchnorm/mul/ReadVariableOp2b
/basemodel/batch_normalization_3/AssignMovingAvg/basemodel/batch_normalization_3/AssignMovingAvg2�
>basemodel/batch_normalization_3/AssignMovingAvg/ReadVariableOp>basemodel/batch_normalization_3/AssignMovingAvg/ReadVariableOp2f
1basemodel/batch_normalization_3/AssignMovingAvg_11basemodel/batch_normalization_3/AssignMovingAvg_12�
@basemodel/batch_normalization_3/AssignMovingAvg_1/ReadVariableOp@basemodel/batch_normalization_3/AssignMovingAvg_1/ReadVariableOp2t
8basemodel/batch_normalization_3/batchnorm/ReadVariableOp8basemodel/batch_normalization_3/batchnorm/ReadVariableOp2|
<basemodel/batch_normalization_3/batchnorm/mul/ReadVariableOp<basemodel/batch_normalization_3/batchnorm/mul/ReadVariableOp2T
(basemodel/dense_1/BiasAdd/ReadVariableOp(basemodel/dense_1/BiasAdd/ReadVariableOp2R
'basemodel/dense_1/MatMul/ReadVariableOp'basemodel/dense_1/MatMul/ReadVariableOp2d
0basemodel/stream_0_conv_1/BiasAdd/ReadVariableOp0basemodel/stream_0_conv_1/BiasAdd/ReadVariableOp2|
<basemodel/stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp<basemodel/stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp2d
0basemodel/stream_0_conv_2/BiasAdd/ReadVariableOp0basemodel/stream_0_conv_2/BiasAdd/ReadVariableOp2|
<basemodel/stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOp<basemodel/stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOp2d
0basemodel/stream_0_conv_3/BiasAdd/ReadVariableOp0basemodel/stream_0_conv_3/BiasAdd/ReadVariableOp2|
<basemodel/stream_0_conv_3/conv1d/ExpandDims_1/ReadVariableOp<basemodel/stream_0_conv_3/conv1d/ExpandDims_1/ReadVariableOp2^
-dense_1/kernel/Regularizer/Abs/ReadVariableOp-dense_1/kernel/Regularizer/Abs/ReadVariableOp2n
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp2t
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp2n
5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp:S O
+
_output_shapes
:���������}
 
_user_specified_nameinputs
�
�
)__inference_model_1_layer_call_fn_8909344

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:@ 
	unknown_5:@�
	unknown_6:	�
	unknown_7:	�
	unknown_8:	�
	unknown_9:	�

unknown_10:	�"

unknown_11:��

unknown_12:	�

unknown_13:	�

unknown_14:	�

unknown_15:	�

unknown_16:	�

unknown_17:	�T

unknown_18:T

unknown_19:T

unknown_20:T

unknown_21:T

unknown_22:T
identity��StatefulPartitionedCall�
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
:���������T*2
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_model_1_layer_call_and_return_conditional_losses_89089012
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
_construction_contextkEagerRuntime*Z
_input_shapesI
G:���������}: : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������}
 
_user_specified_nameinputs
�
k
L__inference_stream_0_drop_1_layer_call_and_return_conditional_losses_8907613

inputs
identity�c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2
dropout/Constw
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:���������}@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:���������}@*
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
T0*+
_output_shapes
:���������}@2
dropout/GreaterEqual�
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:���������}@2
dropout/Cast~
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:���������}@2
dropout/Mul_1i
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:���������}@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������}@:S O
+
_output_shapes
:���������}@
 
_user_specified_nameinputs
�
�
5__inference_batch_normalization_layer_call_fn_8910807

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
 *+
_output_shapes
:���������}@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_89070592
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������}@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������}@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������}@
 
_user_specified_nameinputs
�9
�
 __inference__traced_save_8911740
file_prefix5
1savev2_stream_0_conv_1_kernel_read_readvariableop3
/savev2_stream_0_conv_1_bias_read_readvariableop8
4savev2_batch_normalization_gamma_read_readvariableop7
3savev2_batch_normalization_beta_read_readvariableop>
:savev2_batch_normalization_moving_mean_read_readvariableopB
>savev2_batch_normalization_moving_variance_read_readvariableop5
1savev2_stream_0_conv_2_kernel_read_readvariableop3
/savev2_stream_0_conv_2_bias_read_readvariableop:
6savev2_batch_normalization_1_gamma_read_readvariableop9
5savev2_batch_normalization_1_beta_read_readvariableop@
<savev2_batch_normalization_1_moving_mean_read_readvariableopD
@savev2_batch_normalization_1_moving_variance_read_readvariableop5
1savev2_stream_0_conv_3_kernel_read_readvariableop3
/savev2_stream_0_conv_3_bias_read_readvariableop:
6savev2_batch_normalization_2_gamma_read_readvariableop9
5savev2_batch_normalization_2_beta_read_readvariableop@
<savev2_batch_normalization_2_moving_mean_read_readvariableopD
@savev2_batch_normalization_2_moving_variance_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop:
6savev2_batch_normalization_3_gamma_read_readvariableop9
5savev2_batch_normalization_3_beta_read_readvariableop@
<savev2_batch_normalization_3_moving_mean_read_readvariableopD
@savev2_batch_normalization_3_moving_variance_read_readvariableop
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
ShardedFilename�
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*E
value<B:B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices�
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:01savev2_stream_0_conv_1_kernel_read_readvariableop/savev2_stream_0_conv_1_bias_read_readvariableop4savev2_batch_normalization_gamma_read_readvariableop3savev2_batch_normalization_beta_read_readvariableop:savev2_batch_normalization_moving_mean_read_readvariableop>savev2_batch_normalization_moving_variance_read_readvariableop1savev2_stream_0_conv_2_kernel_read_readvariableop/savev2_stream_0_conv_2_bias_read_readvariableop6savev2_batch_normalization_1_gamma_read_readvariableop5savev2_batch_normalization_1_beta_read_readvariableop<savev2_batch_normalization_1_moving_mean_read_readvariableop@savev2_batch_normalization_1_moving_variance_read_readvariableop1savev2_stream_0_conv_3_kernel_read_readvariableop/savev2_stream_0_conv_3_bias_read_readvariableop6savev2_batch_normalization_2_gamma_read_readvariableop5savev2_batch_normalization_2_beta_read_readvariableop<savev2_batch_normalization_2_moving_mean_read_readvariableop@savev2_batch_normalization_2_moving_variance_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop6savev2_batch_normalization_3_gamma_read_readvariableop5savev2_batch_normalization_3_beta_read_readvariableop<savev2_batch_normalization_3_moving_mean_read_readvariableop@savev2_batch_normalization_3_moving_variance_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *'
dtypes
22
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

identity_1Identity_1:output:0*�
_input_shapes�
�: :@:@:@:@:@:@:@�:�:�:�:�:�:��:�:�:�:�:�:	�T:T:T:T:T:T: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:($
"
_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@:)%
#
_output_shapes
:@�:!

_output_shapes	
:�:!	

_output_shapes	
:�:!


_output_shapes	
:�:!

_output_shapes	
:�:!

_output_shapes	
:�:*&
$
_output_shapes
:��:!

_output_shapes	
:�:!

_output_shapes	
:�:!

_output_shapes	
:�:!

_output_shapes	
:�:!

_output_shapes	
:�:%!

_output_shapes
:	�T: 

_output_shapes
:T: 

_output_shapes
:T: 

_output_shapes
:T: 

_output_shapes
:T: 

_output_shapes
:T:

_output_shapes
: 
�
j
L__inference_dense_1_dropout_layer_call_and_return_conditional_losses_8911468

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:����������2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:����������2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�=
�	
D__inference_model_1_layer_call_and_return_conditional_losses_8908901

inputs'
basemodel_8908827:@
basemodel_8908829:@
basemodel_8908831:@
basemodel_8908833:@
basemodel_8908835:@
basemodel_8908837:@(
basemodel_8908839:@� 
basemodel_8908841:	� 
basemodel_8908843:	� 
basemodel_8908845:	� 
basemodel_8908847:	� 
basemodel_8908849:	�)
basemodel_8908851:�� 
basemodel_8908853:	� 
basemodel_8908855:	� 
basemodel_8908857:	� 
basemodel_8908859:	� 
basemodel_8908861:	�$
basemodel_8908863:	�T
basemodel_8908865:T
basemodel_8908867:T
basemodel_8908869:T
basemodel_8908871:T
basemodel_8908873:T
identity��!basemodel/StatefulPartitionedCall�-dense_1/kernel/Regularizer/Abs/ReadVariableOp�5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp�8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp�5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp�
!basemodel/StatefulPartitionedCallStatefulPartitionedCallinputsbasemodel_8908827basemodel_8908829basemodel_8908831basemodel_8908833basemodel_8908835basemodel_8908837basemodel_8908839basemodel_8908841basemodel_8908843basemodel_8908845basemodel_8908847basemodel_8908849basemodel_8908851basemodel_8908853basemodel_8908855basemodel_8908857basemodel_8908859basemodel_8908861basemodel_8908863basemodel_8908865basemodel_8908867basemodel_8908869basemodel_8908871basemodel_8908873*$
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������T*2
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_basemodel_layer_call_and_return_conditional_losses_89087182#
!basemodel/StatefulPartitionedCall�
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpbasemodel_8908827*"
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
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpbasemodel_8908839*#
_output_shapes
:@�*
dtype02:
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp�
)stream_0_conv_2/kernel/Regularizer/SquareSquare@stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*#
_output_shapes
:@�2+
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
5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpbasemodel_8908851*$
_output_shapes
:��*
dtype027
5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp�
&stream_0_conv_3/kernel/Regularizer/AbsAbs=stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*$
_output_shapes
:��2(
&stream_0_conv_3/kernel/Regularizer/Abs�
(stream_0_conv_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_0_conv_3/kernel/Regularizer/Const�
&stream_0_conv_3/kernel/Regularizer/SumSum*stream_0_conv_3/kernel/Regularizer/Abs:y:01stream_0_conv_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_3/kernel/Regularizer/Sum�
(stream_0_conv_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2*
(stream_0_conv_3/kernel/Regularizer/mul/x�
&stream_0_conv_3/kernel/Regularizer/mulMul1stream_0_conv_3/kernel/Regularizer/mul/x:output:0/stream_0_conv_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_3/kernel/Regularizer/mul�
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpbasemodel_8908863*
_output_shapes
:	�T*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOp�
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	�T2 
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
IdentityIdentity*basemodel/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������T2

Identity�
NoOpNoOp"^basemodel/StatefulPartitionedCall.^dense_1/kernel/Regularizer/Abs/ReadVariableOp6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp9^stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp6^stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:���������}: : : : : : : : : : : : : : : : : : : : : : : : 2F
!basemodel/StatefulPartitionedCall!basemodel/StatefulPartitionedCall2^
-dense_1/kernel/Regularizer/Abs/ReadVariableOp-dense_1/kernel/Regularizer/Abs/ReadVariableOp2n
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp2t
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp2n
5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp:S O
+
_output_shapes
:���������}
 
_user_specified_nameinputs
�+
�
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_8907572

inputs6
'assignmovingavg_readvariableop_resource:	�8
)assignmovingavg_1_readvariableop_resource:	�4
%batchnorm_mul_readvariableop_resource:	�0
!batchnorm_readvariableop_resource:	�
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOp�
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indices�
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*#
_output_shapes
:�*
	keep_dims(2
moments/mean�
moments/StopGradientStopGradientmoments/mean:output:0*
T0*#
_output_shapes
:�2
moments/StopGradient�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*,
_output_shapes
:���������}�2
moments/SquaredDifference�
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices�
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*#
_output_shapes
:�*
	keep_dims(2
moments/variance�
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 2
moments/Squeeze�
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:�*
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
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:�*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:�2
AssignMovingAvg/sub�
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:�2
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
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:�*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:�2
AssignMovingAvg_1/sub�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:�2
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
T0*
_output_shapes	
:�2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:�2
batchnorm/Rsqrt�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype02
batchnorm/mul/ReadVariableOp�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2
batchnorm/mul{
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*,
_output_shapes
:���������}�2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:�2
batchnorm/mul_2�
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype02
batchnorm/ReadVariableOp�
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*,
_output_shapes
:���������}�2
batchnorm/add_1s
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*,
_output_shapes
:���������}�2

Identity�
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������}�: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:T P
,
_output_shapes
:���������}�
 
_user_specified_nameinputs
�
j
1__inference_stream_0_drop_2_layer_call_fn_8911181

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
:���������}�* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_stream_0_drop_2_layer_call_and_return_conditional_losses_89075142
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:���������}�2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:���������}�22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:���������}�
 
_user_specified_nameinputs
�
�
)__inference_dense_1_layer_call_fn_8911495

inputs
unknown:	�T
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
D__inference_dense_1_layer_call_and_return_conditional_losses_89072532
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
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
+__inference_basemodel_layer_call_fn_8907351
inputs_0
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:@ 
	unknown_5:@�
	unknown_6:	�
	unknown_7:	�
	unknown_8:	�
	unknown_9:	�

unknown_10:	�"

unknown_11:��

unknown_12:	�

unknown_13:	�

unknown_14:	�

unknown_15:	�

unknown_16:	�

unknown_17:	�T

unknown_18:T

unknown_19:T

unknown_20:T

unknown_21:T

unknown_22:T
identity��StatefulPartitionedCall�
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
:���������T*:
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_basemodel_layer_call_and_return_conditional_losses_89073002
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
_construction_contextkEagerRuntime*Z
_input_shapesI
G:���������}: : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
+
_output_shapes
:���������}
"
_user_specified_name
inputs_0"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
G
left_inputs8
serving_default_left_inputs:0���������}=
	basemodel0
StatefulPartitionedCall:0���������Ttensorflow/serving/predict:��
�
layer-0
layer_with_weights-0
layer-1
	variables
regularization_losses
trainable_variables
	keras_api

signatures
�_default_save_signature
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_network
"
_tf_keras_input_layer
�
layer-0
	layer-1

layer_with_weights-0

layer-2
layer_with_weights-1
layer-3
layer-4
layer-5
layer_with_weights-2
layer-6
layer_with_weights-3
layer-7
layer-8
layer-9
layer_with_weights-4
layer-10
layer_with_weights-5
layer-11
layer-12
layer-13
layer-14
layer-15
layer_with_weights-6
layer-16
layer_with_weights-7
layer-17
layer-18
	variables
regularization_losses
trainable_variables
	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_network
�
0
 1
!2
"3
#4
$5
%6
&7
'8
(9
)10
*11
+12
,13
-14
.15
/16
017
118
219
320
421
522
623"
trackable_list_wrapper
 "
trackable_list_wrapper
�
0
 1
!2
"3
%4
&5
'6
(7
+8
,9
-10
.11
112
213
314
415"
trackable_list_wrapper
�
	variables
regularization_losses
7layer_metrics

8layers
trainable_variables
9metrics
:non_trainable_variables
;layer_regularization_losses
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
<	variables
=regularization_losses
>trainable_variables
?	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�

kernel
 bias
@	variables
Aregularization_losses
Btrainable_variables
C	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
Daxis
	!gamma
"beta
#moving_mean
$moving_variance
E	variables
Fregularization_losses
Gtrainable_variables
H	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
I	variables
Jregularization_losses
Ktrainable_variables
L	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
M	variables
Nregularization_losses
Otrainable_variables
P	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�

%kernel
&bias
Q	variables
Rregularization_losses
Strainable_variables
T	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
Uaxis
	'gamma
(beta
)moving_mean
*moving_variance
V	variables
Wregularization_losses
Xtrainable_variables
Y	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
Z	variables
[regularization_losses
\trainable_variables
]	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
^	variables
_regularization_losses
`trainable_variables
a	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�

+kernel
,bias
b	variables
cregularization_losses
dtrainable_variables
e	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
faxis
	-gamma
.beta
/moving_mean
0moving_variance
g	variables
hregularization_losses
itrainable_variables
j	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
k	variables
lregularization_losses
mtrainable_variables
n	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
o	variables
pregularization_losses
qtrainable_variables
r	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
s	variables
tregularization_losses
utrainable_variables
v	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
w	variables
xregularization_losses
ytrainable_variables
z	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�

1kernel
2bias
{	variables
|regularization_losses
}trainable_variables
~	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
axis
	3gamma
4beta
5moving_mean
6moving_variance
�	variables
�regularization_losses
�trainable_variables
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�regularization_losses
�trainable_variables
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
0
 1
!2
"3
#4
$5
%6
&7
'8
(9
)10
*11
+12
,13
-14
.15
/16
017
118
219
320
421
522
623"
trackable_list_wrapper
@
�0
�1
�2
�3"
trackable_list_wrapper
�
0
 1
!2
"3
%4
&5
'6
(7
+8
,9
-10
.11
112
213
314
415"
trackable_list_wrapper
�
	variables
regularization_losses
�layer_metrics
�layers
trainable_variables
�metrics
�non_trainable_variables
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
,:*@2stream_0_conv_1/kernel
": @2stream_0_conv_1/bias
':%@2batch_normalization/gamma
&:$@2batch_normalization/beta
/:-@ (2batch_normalization/moving_mean
3:1@ (2#batch_normalization/moving_variance
-:+@�2stream_0_conv_2/kernel
#:!�2stream_0_conv_2/bias
*:(�2batch_normalization_1/gamma
):'�2batch_normalization_1/beta
2:0� (2!batch_normalization_1/moving_mean
6:4� (2%batch_normalization_1/moving_variance
.:,��2stream_0_conv_3/kernel
#:!�2stream_0_conv_3/bias
*:(�2batch_normalization_2/gamma
):'�2batch_normalization_2/beta
2:0� (2!batch_normalization_2/moving_mean
6:4� (2%batch_normalization_2/moving_variance
!:	�T2dense_1/kernel
:T2dense_1/bias
):'T2batch_normalization_3/gamma
(:&T2batch_normalization_3/beta
1:/T (2!batch_normalization_3/moving_mean
5:3T (2%batch_normalization_3/moving_variance
 "
trackable_dict_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
X
#0
$1
)2
*3
/4
05
56
67"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
<	variables
=regularization_losses
�layer_metrics
�layers
>trainable_variables
�metrics
�non_trainable_variables
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
.
0
 1"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
.
0
 1"
trackable_list_wrapper
�
@	variables
Aregularization_losses
�layer_metrics
�layers
Btrainable_variables
�metrics
�non_trainable_variables
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
<
!0
"1
#2
$3"
trackable_list_wrapper
 "
trackable_list_wrapper
.
!0
"1"
trackable_list_wrapper
�
E	variables
Fregularization_losses
�layer_metrics
�layers
Gtrainable_variables
�metrics
�non_trainable_variables
 �layer_regularization_losses
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
I	variables
Jregularization_losses
�layer_metrics
�layers
Ktrainable_variables
�metrics
�non_trainable_variables
 �layer_regularization_losses
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
M	variables
Nregularization_losses
�layer_metrics
�layers
Otrainable_variables
�metrics
�non_trainable_variables
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
.
%0
&1"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
.
%0
&1"
trackable_list_wrapper
�
Q	variables
Rregularization_losses
�layer_metrics
�layers
Strainable_variables
�metrics
�non_trainable_variables
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
<
'0
(1
)2
*3"
trackable_list_wrapper
 "
trackable_list_wrapper
.
'0
(1"
trackable_list_wrapper
�
V	variables
Wregularization_losses
�layer_metrics
�layers
Xtrainable_variables
�metrics
�non_trainable_variables
 �layer_regularization_losses
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
Z	variables
[regularization_losses
�layer_metrics
�layers
\trainable_variables
�metrics
�non_trainable_variables
 �layer_regularization_losses
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
^	variables
_regularization_losses
�layer_metrics
�layers
`trainable_variables
�metrics
�non_trainable_variables
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
.
+0
,1"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
.
+0
,1"
trackable_list_wrapper
�
b	variables
cregularization_losses
�layer_metrics
�layers
dtrainable_variables
�metrics
�non_trainable_variables
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
<
-0
.1
/2
03"
trackable_list_wrapper
 "
trackable_list_wrapper
.
-0
.1"
trackable_list_wrapper
�
g	variables
hregularization_losses
�layer_metrics
�layers
itrainable_variables
�metrics
�non_trainable_variables
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
k	variables
lregularization_losses
�layer_metrics
�layers
mtrainable_variables
�metrics
�non_trainable_variables
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
o	variables
pregularization_losses
�layer_metrics
�layers
qtrainable_variables
�metrics
�non_trainable_variables
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
s	variables
tregularization_losses
�layer_metrics
�layers
utrainable_variables
�metrics
�non_trainable_variables
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
w	variables
xregularization_losses
�layer_metrics
�layers
ytrainable_variables
�metrics
�non_trainable_variables
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
.
10
21"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
.
10
21"
trackable_list_wrapper
�
{	variables
|regularization_losses
�layer_metrics
�layers
}trainable_variables
�metrics
�non_trainable_variables
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
<
30
41
52
63"
trackable_list_wrapper
 "
trackable_list_wrapper
.
30
41"
trackable_list_wrapper
�
�	variables
�regularization_losses
�layer_metrics
�layers
�trainable_variables
�metrics
�non_trainable_variables
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
�	variables
�regularization_losses
�layer_metrics
�layers
�trainable_variables
�metrics
�non_trainable_variables
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
�
0
	1

2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18"
trackable_list_wrapper
 "
trackable_list_wrapper
X
#0
$1
)2
*3
/4
05
56
67"
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
(
�0"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
#0
$1"
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
(
�0"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
)0
*1"
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
(
�0"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
/0
01"
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
(
�0"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
50
61"
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
�B�
"__inference__wrapped_model_8906327left_inputs"�
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
)__inference_model_1_layer_call_fn_8908430
)__inference_model_1_layer_call_fn_8909291
)__inference_model_1_layer_call_fn_8909344
)__inference_model_1_layer_call_fn_8909005�
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
D__inference_model_1_layer_call_and_return_conditional_losses_8909486
D__inference_model_1_layer_call_and_return_conditional_losses_8909719
D__inference_model_1_layer_call_and_return_conditional_losses_8909082
D__inference_model_1_layer_call_and_return_conditional_losses_8909159�
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
+__inference_basemodel_layer_call_fn_8907351
+__inference_basemodel_layer_call_fn_8909796
+__inference_basemodel_layer_call_fn_8909849
+__inference_basemodel_layer_call_fn_8907968
+__inference_basemodel_layer_call_fn_8909902
+__inference_basemodel_layer_call_fn_8909955�
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
F__inference_basemodel_layer_call_and_return_conditional_losses_8910097
F__inference_basemodel_layer_call_and_return_conditional_losses_8910330
F__inference_basemodel_layer_call_and_return_conditional_losses_8908062
F__inference_basemodel_layer_call_and_return_conditional_losses_8908156
F__inference_basemodel_layer_call_and_return_conditional_losses_8910472
F__inference_basemodel_layer_call_and_return_conditional_losses_8910705�
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
%__inference_signature_wrapper_8909238left_inputs"�
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
5__inference_stream_0_input_drop_layer_call_fn_8910710
5__inference_stream_0_input_drop_layer_call_fn_8910715�
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
P__inference_stream_0_input_drop_layer_call_and_return_conditional_losses_8910720
P__inference_stream_0_input_drop_layer_call_and_return_conditional_losses_8910732�
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
1__inference_stream_0_conv_1_layer_call_fn_8910747�
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
L__inference_stream_0_conv_1_layer_call_and_return_conditional_losses_8910768�
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
5__inference_batch_normalization_layer_call_fn_8910781
5__inference_batch_normalization_layer_call_fn_8910794
5__inference_batch_normalization_layer_call_fn_8910807
5__inference_batch_normalization_layer_call_fn_8910820�
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
P__inference_batch_normalization_layer_call_and_return_conditional_losses_8910840
P__inference_batch_normalization_layer_call_and_return_conditional_losses_8910874
P__inference_batch_normalization_layer_call_and_return_conditional_losses_8910894
P__inference_batch_normalization_layer_call_and_return_conditional_losses_8910928�
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
,__inference_activation_layer_call_fn_8910933�
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
G__inference_activation_layer_call_and_return_conditional_losses_8910938�
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
1__inference_stream_0_drop_1_layer_call_fn_8910943
1__inference_stream_0_drop_1_layer_call_fn_8910948�
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
L__inference_stream_0_drop_1_layer_call_and_return_conditional_losses_8910953
L__inference_stream_0_drop_1_layer_call_and_return_conditional_losses_8910965�
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
1__inference_stream_0_conv_2_layer_call_fn_8910980�
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
L__inference_stream_0_conv_2_layer_call_and_return_conditional_losses_8911001�
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
7__inference_batch_normalization_1_layer_call_fn_8911014
7__inference_batch_normalization_1_layer_call_fn_8911027
7__inference_batch_normalization_1_layer_call_fn_8911040
7__inference_batch_normalization_1_layer_call_fn_8911053�
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
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_8911073
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_8911107
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_8911127
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_8911161�
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
.__inference_activation_1_layer_call_fn_8911166�
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
I__inference_activation_1_layer_call_and_return_conditional_losses_8911171�
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
1__inference_stream_0_drop_2_layer_call_fn_8911176
1__inference_stream_0_drop_2_layer_call_fn_8911181�
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
L__inference_stream_0_drop_2_layer_call_and_return_conditional_losses_8911186
L__inference_stream_0_drop_2_layer_call_and_return_conditional_losses_8911198�
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
1__inference_stream_0_conv_3_layer_call_fn_8911213�
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
L__inference_stream_0_conv_3_layer_call_and_return_conditional_losses_8911234�
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
7__inference_batch_normalization_2_layer_call_fn_8911247
7__inference_batch_normalization_2_layer_call_fn_8911260
7__inference_batch_normalization_2_layer_call_fn_8911273
7__inference_batch_normalization_2_layer_call_fn_8911286�
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
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_8911306
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_8911340
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_8911360
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_8911394�
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
.__inference_activation_2_layer_call_fn_8911399�
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
I__inference_activation_2_layer_call_and_return_conditional_losses_8911404�
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
1__inference_stream_0_drop_3_layer_call_fn_8911409
1__inference_stream_0_drop_3_layer_call_fn_8911414�
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
L__inference_stream_0_drop_3_layer_call_and_return_conditional_losses_8911419
L__inference_stream_0_drop_3_layer_call_and_return_conditional_losses_8911431�
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
:__inference_global_average_pooling1d_layer_call_fn_8911436
:__inference_global_average_pooling1d_layer_call_fn_8911441�
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
U__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_8911447
U__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_8911453�
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
1__inference_dense_1_dropout_layer_call_fn_8911458
1__inference_dense_1_dropout_layer_call_fn_8911463�
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
L__inference_dense_1_dropout_layer_call_and_return_conditional_losses_8911468
L__inference_dense_1_dropout_layer_call_and_return_conditional_losses_8911480�
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
)__inference_dense_1_layer_call_fn_8911495�
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
D__inference_dense_1_layer_call_and_return_conditional_losses_8911511�
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
7__inference_batch_normalization_3_layer_call_fn_8911524
7__inference_batch_normalization_3_layer_call_fn_8911537�
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
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_8911557
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_8911591�
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
4__inference_dense_activation_1_layer_call_fn_8911596�
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
O__inference_dense_activation_1_layer_call_and_return_conditional_losses_8911601�
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
__inference_loss_fn_0_8911612�
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
__inference_loss_fn_1_8911623�
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
__inference_loss_fn_2_8911634�
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
__inference_loss_fn_3_8911645�
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
"__inference__wrapped_model_8906327� $!#"%&*')(+,0-/.1263548�5
.�+
)�&
left_inputs���������}
� "5�2
0
	basemodel#� 
	basemodel���������T�
I__inference_activation_1_layer_call_and_return_conditional_losses_8911171b4�1
*�'
%�"
inputs���������}�
� "*�'
 �
0���������}�
� �
.__inference_activation_1_layer_call_fn_8911166U4�1
*�'
%�"
inputs���������}�
� "����������}��
I__inference_activation_2_layer_call_and_return_conditional_losses_8911404b4�1
*�'
%�"
inputs���������}�
� "*�'
 �
0���������}�
� �
.__inference_activation_2_layer_call_fn_8911399U4�1
*�'
%�"
inputs���������}�
� "����������}��
G__inference_activation_layer_call_and_return_conditional_losses_8910938`3�0
)�&
$�!
inputs���������}@
� ")�&
�
0���������}@
� �
,__inference_activation_layer_call_fn_8910933S3�0
)�&
$�!
inputs���������}@
� "����������}@�
F__inference_basemodel_layer_call_and_return_conditional_losses_8908062� $!#"%&*')(+,0-/.126354=�:
3�0
&�#
inputs_0���������}
p 

 
� "%�"
�
0���������T
� �
F__inference_basemodel_layer_call_and_return_conditional_losses_8908156� #$!"%&)*'(+,/0-.125634=�:
3�0
&�#
inputs_0���������}
p

 
� "%�"
�
0���������T
� �
F__inference_basemodel_layer_call_and_return_conditional_losses_8910097~ $!#"%&*')(+,0-/.126354;�8
1�.
$�!
inputs���������}
p 

 
� "%�"
�
0���������T
� �
F__inference_basemodel_layer_call_and_return_conditional_losses_8910330~ #$!"%&)*'(+,/0-.125634;�8
1�.
$�!
inputs���������}
p

 
� "%�"
�
0���������T
� �
F__inference_basemodel_layer_call_and_return_conditional_losses_8910472� $!#"%&*')(+,0-/.126354B�?
8�5
+�(
&�#
inputs/0���������}
p 

 
� "%�"
�
0���������T
� �
F__inference_basemodel_layer_call_and_return_conditional_losses_8910705� #$!"%&)*'(+,/0-.125634B�?
8�5
+�(
&�#
inputs/0���������}
p

 
� "%�"
�
0���������T
� �
+__inference_basemodel_layer_call_fn_8907351s $!#"%&*')(+,0-/.126354=�:
3�0
&�#
inputs_0���������}
p 

 
� "����������T�
+__inference_basemodel_layer_call_fn_8907968s #$!"%&)*'(+,/0-.125634=�:
3�0
&�#
inputs_0���������}
p

 
� "����������T�
+__inference_basemodel_layer_call_fn_8909796q $!#"%&*')(+,0-/.126354;�8
1�.
$�!
inputs���������}
p 

 
� "����������T�
+__inference_basemodel_layer_call_fn_8909849q #$!"%&)*'(+,/0-.125634;�8
1�.
$�!
inputs���������}
p

 
� "����������T�
+__inference_basemodel_layer_call_fn_8909902x $!#"%&*')(+,0-/.126354B�?
8�5
+�(
&�#
inputs/0���������}
p 

 
� "����������T�
+__inference_basemodel_layer_call_fn_8909955x #$!"%&)*'(+,/0-.125634B�?
8�5
+�(
&�#
inputs/0���������}
p

 
� "����������T�
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_8911073~*')(A�>
7�4
.�+
inputs�������������������
p 
� "3�0
)�&
0�������������������
� �
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_8911107~)*'(A�>
7�4
.�+
inputs�������������������
p
� "3�0
)�&
0�������������������
� �
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_8911127l*')(8�5
.�+
%�"
inputs���������}�
p 
� "*�'
 �
0���������}�
� �
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_8911161l)*'(8�5
.�+
%�"
inputs���������}�
p
� "*�'
 �
0���������}�
� �
7__inference_batch_normalization_1_layer_call_fn_8911014q*')(A�>
7�4
.�+
inputs�������������������
p 
� "&�#��������������������
7__inference_batch_normalization_1_layer_call_fn_8911027q)*'(A�>
7�4
.�+
inputs�������������������
p
� "&�#��������������������
7__inference_batch_normalization_1_layer_call_fn_8911040_*')(8�5
.�+
%�"
inputs���������}�
p 
� "����������}��
7__inference_batch_normalization_1_layer_call_fn_8911053_)*'(8�5
.�+
%�"
inputs���������}�
p
� "����������}��
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_8911306~0-/.A�>
7�4
.�+
inputs�������������������
p 
� "3�0
)�&
0�������������������
� �
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_8911340~/0-.A�>
7�4
.�+
inputs�������������������
p
� "3�0
)�&
0�������������������
� �
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_8911360l0-/.8�5
.�+
%�"
inputs���������}�
p 
� "*�'
 �
0���������}�
� �
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_8911394l/0-.8�5
.�+
%�"
inputs���������}�
p
� "*�'
 �
0���������}�
� �
7__inference_batch_normalization_2_layer_call_fn_8911247q0-/.A�>
7�4
.�+
inputs�������������������
p 
� "&�#��������������������
7__inference_batch_normalization_2_layer_call_fn_8911260q/0-.A�>
7�4
.�+
inputs�������������������
p
� "&�#��������������������
7__inference_batch_normalization_2_layer_call_fn_8911273_0-/.8�5
.�+
%�"
inputs���������}�
p 
� "����������}��
7__inference_batch_normalization_2_layer_call_fn_8911286_/0-.8�5
.�+
%�"
inputs���������}�
p
� "����������}��
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_8911557b63543�0
)�&
 �
inputs���������T
p 
� "%�"
�
0���������T
� �
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_8911591b56343�0
)�&
 �
inputs���������T
p
� "%�"
�
0���������T
� �
7__inference_batch_normalization_3_layer_call_fn_8911524U63543�0
)�&
 �
inputs���������T
p 
� "����������T�
7__inference_batch_normalization_3_layer_call_fn_8911537U56343�0
)�&
 �
inputs���������T
p
� "����������T�
P__inference_batch_normalization_layer_call_and_return_conditional_losses_8910840|$!#"@�=
6�3
-�*
inputs������������������@
p 
� "2�/
(�%
0������������������@
� �
P__inference_batch_normalization_layer_call_and_return_conditional_losses_8910874|#$!"@�=
6�3
-�*
inputs������������������@
p
� "2�/
(�%
0������������������@
� �
P__inference_batch_normalization_layer_call_and_return_conditional_losses_8910894j$!#"7�4
-�*
$�!
inputs���������}@
p 
� ")�&
�
0���������}@
� �
P__inference_batch_normalization_layer_call_and_return_conditional_losses_8910928j#$!"7�4
-�*
$�!
inputs���������}@
p
� ")�&
�
0���������}@
� �
5__inference_batch_normalization_layer_call_fn_8910781o$!#"@�=
6�3
-�*
inputs������������������@
p 
� "%�"������������������@�
5__inference_batch_normalization_layer_call_fn_8910794o#$!"@�=
6�3
-�*
inputs������������������@
p
� "%�"������������������@�
5__inference_batch_normalization_layer_call_fn_8910807]$!#"7�4
-�*
$�!
inputs���������}@
p 
� "����������}@�
5__inference_batch_normalization_layer_call_fn_8910820]#$!"7�4
-�*
$�!
inputs���������}@
p
� "����������}@�
L__inference_dense_1_dropout_layer_call_and_return_conditional_losses_8911468^4�1
*�'
!�
inputs����������
p 
� "&�#
�
0����������
� �
L__inference_dense_1_dropout_layer_call_and_return_conditional_losses_8911480^4�1
*�'
!�
inputs����������
p
� "&�#
�
0����������
� �
1__inference_dense_1_dropout_layer_call_fn_8911458Q4�1
*�'
!�
inputs����������
p 
� "������������
1__inference_dense_1_dropout_layer_call_fn_8911463Q4�1
*�'
!�
inputs����������
p
� "������������
D__inference_dense_1_layer_call_and_return_conditional_losses_8911511]120�-
&�#
!�
inputs����������
� "%�"
�
0���������T
� }
)__inference_dense_1_layer_call_fn_8911495P120�-
&�#
!�
inputs����������
� "����������T�
O__inference_dense_activation_1_layer_call_and_return_conditional_losses_8911601X/�,
%�"
 �
inputs���������T
� "%�"
�
0���������T
� �
4__inference_dense_activation_1_layer_call_fn_8911596K/�,
%�"
 �
inputs���������T
� "����������T�
U__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_8911447{I�F
?�<
6�3
inputs'���������������������������

 
� ".�+
$�!
0������������������
� �
U__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_8911453b8�5
.�+
%�"
inputs���������}�

 
� "&�#
�
0����������
� �
:__inference_global_average_pooling1d_layer_call_fn_8911436nI�F
?�<
6�3
inputs'���������������������������

 
� "!��������������������
:__inference_global_average_pooling1d_layer_call_fn_8911441U8�5
.�+
%�"
inputs���������}�

 
� "�����������<
__inference_loss_fn_0_8911612�

� 
� "� <
__inference_loss_fn_1_8911623%�

� 
� "� <
__inference_loss_fn_2_8911634+�

� 
� "� <
__inference_loss_fn_3_89116451�

� 
� "� �
D__inference_model_1_layer_call_and_return_conditional_losses_8909082� $!#"%&*')(+,0-/.126354@�=
6�3
)�&
left_inputs���������}
p 

 
� "%�"
�
0���������T
� �
D__inference_model_1_layer_call_and_return_conditional_losses_8909159� #$!"%&)*'(+,/0-.125634@�=
6�3
)�&
left_inputs���������}
p

 
� "%�"
�
0���������T
� �
D__inference_model_1_layer_call_and_return_conditional_losses_8909486~ $!#"%&*')(+,0-/.126354;�8
1�.
$�!
inputs���������}
p 

 
� "%�"
�
0���������T
� �
D__inference_model_1_layer_call_and_return_conditional_losses_8909719~ #$!"%&)*'(+,/0-.125634;�8
1�.
$�!
inputs���������}
p

 
� "%�"
�
0���������T
� �
)__inference_model_1_layer_call_fn_8908430v $!#"%&*')(+,0-/.126354@�=
6�3
)�&
left_inputs���������}
p 

 
� "����������T�
)__inference_model_1_layer_call_fn_8909005v #$!"%&)*'(+,/0-.125634@�=
6�3
)�&
left_inputs���������}
p

 
� "����������T�
)__inference_model_1_layer_call_fn_8909291q $!#"%&*')(+,0-/.126354;�8
1�.
$�!
inputs���������}
p 

 
� "����������T�
)__inference_model_1_layer_call_fn_8909344q #$!"%&)*'(+,/0-.125634;�8
1�.
$�!
inputs���������}
p

 
� "����������T�
%__inference_signature_wrapper_8909238� $!#"%&*')(+,0-/.126354G�D
� 
=�:
8
left_inputs)�&
left_inputs���������}"5�2
0
	basemodel#� 
	basemodel���������T�
L__inference_stream_0_conv_1_layer_call_and_return_conditional_losses_8910768d 3�0
)�&
$�!
inputs���������}
� ")�&
�
0���������}@
� �
1__inference_stream_0_conv_1_layer_call_fn_8910747W 3�0
)�&
$�!
inputs���������}
� "����������}@�
L__inference_stream_0_conv_2_layer_call_and_return_conditional_losses_8911001e%&3�0
)�&
$�!
inputs���������}@
� "*�'
 �
0���������}�
� �
1__inference_stream_0_conv_2_layer_call_fn_8910980X%&3�0
)�&
$�!
inputs���������}@
� "����������}��
L__inference_stream_0_conv_3_layer_call_and_return_conditional_losses_8911234f+,4�1
*�'
%�"
inputs���������}�
� "*�'
 �
0���������}�
� �
1__inference_stream_0_conv_3_layer_call_fn_8911213Y+,4�1
*�'
%�"
inputs���������}�
� "����������}��
L__inference_stream_0_drop_1_layer_call_and_return_conditional_losses_8910953d7�4
-�*
$�!
inputs���������}@
p 
� ")�&
�
0���������}@
� �
L__inference_stream_0_drop_1_layer_call_and_return_conditional_losses_8910965d7�4
-�*
$�!
inputs���������}@
p
� ")�&
�
0���������}@
� �
1__inference_stream_0_drop_1_layer_call_fn_8910943W7�4
-�*
$�!
inputs���������}@
p 
� "����������}@�
1__inference_stream_0_drop_1_layer_call_fn_8910948W7�4
-�*
$�!
inputs���������}@
p
� "����������}@�
L__inference_stream_0_drop_2_layer_call_and_return_conditional_losses_8911186f8�5
.�+
%�"
inputs���������}�
p 
� "*�'
 �
0���������}�
� �
L__inference_stream_0_drop_2_layer_call_and_return_conditional_losses_8911198f8�5
.�+
%�"
inputs���������}�
p
� "*�'
 �
0���������}�
� �
1__inference_stream_0_drop_2_layer_call_fn_8911176Y8�5
.�+
%�"
inputs���������}�
p 
� "����������}��
1__inference_stream_0_drop_2_layer_call_fn_8911181Y8�5
.�+
%�"
inputs���������}�
p
� "����������}��
L__inference_stream_0_drop_3_layer_call_and_return_conditional_losses_8911419f8�5
.�+
%�"
inputs���������}�
p 
� "*�'
 �
0���������}�
� �
L__inference_stream_0_drop_3_layer_call_and_return_conditional_losses_8911431f8�5
.�+
%�"
inputs���������}�
p
� "*�'
 �
0���������}�
� �
1__inference_stream_0_drop_3_layer_call_fn_8911409Y8�5
.�+
%�"
inputs���������}�
p 
� "����������}��
1__inference_stream_0_drop_3_layer_call_fn_8911414Y8�5
.�+
%�"
inputs���������}�
p
� "����������}��
P__inference_stream_0_input_drop_layer_call_and_return_conditional_losses_8910720d7�4
-�*
$�!
inputs���������}
p 
� ")�&
�
0���������}
� �
P__inference_stream_0_input_drop_layer_call_and_return_conditional_losses_8910732d7�4
-�*
$�!
inputs���������}
p
� ")�&
�
0���������}
� �
5__inference_stream_0_input_drop_layer_call_fn_8910710W7�4
-�*
$�!
inputs���������}
p 
� "����������}�
5__inference_stream_0_input_drop_layer_call_fn_8910715W7�4
-�*
$�!
inputs���������}
p
� "����������}