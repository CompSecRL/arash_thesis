��
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
 �"serve*2.6.22v2.6.1-9-gc2363d6d0258��
�
stream_0_conv_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*'
shared_namestream_0_conv_1/kernel
�
*stream_0_conv_1/kernel/Read/ReadVariableOpReadVariableOpstream_0_conv_1/kernel*"
_output_shapes
:@*
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
batch_normalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:T*,
shared_namebatch_normalization_1/gamma
�
/batch_normalization_1/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_1/gamma*
_output_shapes
:T*
dtype0
�
batch_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:T*+
shared_namebatch_normalization_1/beta
�
.batch_normalization_1/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_1/beta*
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
shape:T*2
shared_name#!batch_normalization_1/moving_mean
�
5batch_normalization_1/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_1/moving_mean*
_output_shapes
:T*
dtype0
�
%batch_normalization_1/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:T*6
shared_name'%batch_normalization_1/moving_variance
�
9batch_normalization_1/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_1/moving_variance*
_output_shapes
:T*
dtype0

NoOpNoOp
�,
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�+
value�+B�+ B�+
�
layer-0
layer_with_weights-0
layer-1
trainable_variables
regularization_losses
	variables
	keras_api

signatures
 
�
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
layer-6
layer-7
layer_with_weights-2
layer-8
layer_with_weights-3
layer-9
layer-10
trainable_variables
regularization_losses
	variables
	keras_api
8
0
1
2
3
4
5
6
7
 
V
0
1
2
3
4
 5
6
7
8
9
!10
"11
�
#layer_metrics
$metrics
%non_trainable_variables
trainable_variables

&layers
'layer_regularization_losses
regularization_losses
	variables
 
 
R
(trainable_variables
)regularization_losses
*	variables
+	keras_api
h

kernel
bias
,trainable_variables
-regularization_losses
.	variables
/	keras_api
�
0axis
	gamma
beta
moving_mean
 moving_variance
1trainable_variables
2regularization_losses
3	variables
4	keras_api
R
5trainable_variables
6regularization_losses
7	variables
8	keras_api
R
9trainable_variables
:regularization_losses
;	variables
<	keras_api
R
=trainable_variables
>regularization_losses
?	variables
@	keras_api
R
Atrainable_variables
Bregularization_losses
C	variables
D	keras_api
h

kernel
bias
Etrainable_variables
Fregularization_losses
G	variables
H	keras_api
�
Iaxis
	gamma
beta
!moving_mean
"moving_variance
Jtrainable_variables
Kregularization_losses
L	variables
M	keras_api
R
Ntrainable_variables
Oregularization_losses
P	variables
Q	keras_api
8
0
1
2
3
4
5
6
7
 
V
0
1
2
3
4
 5
6
7
8
9
!10
"11
�
Rlayer_metrics
Smetrics
Tnon_trainable_variables
trainable_variables

Ulayers
Vlayer_regularization_losses
regularization_losses
	variables
\Z
VARIABLE_VALUEstream_0_conv_1/kernel0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEstream_0_conv_1/bias0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUEbatch_normalization/gamma0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUEbatch_normalization/beta0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEdense_1/kernel0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEdense_1/bias0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEbatch_normalization_1/gamma0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUEbatch_normalization_1/beta0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEbatch_normalization/moving_mean&variables/4/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUE#batch_normalization/moving_variance&variables/5/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUE!batch_normalization_1/moving_mean'variables/10/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUE%batch_normalization_1/moving_variance'variables/11/.ATTRIBUTES/VARIABLE_VALUE
 
 

0
 1
!2
"3

0
1
 
 
 
 
�
Wlayer_metrics
Xmetrics
Ynon_trainable_variables
(trainable_variables

Zlayers
[layer_regularization_losses
)regularization_losses
*	variables

0
1
 

0
1
�
\layer_metrics
]metrics
^non_trainable_variables
,trainable_variables

_layers
`layer_regularization_losses
-regularization_losses
.	variables
 

0
1
 

0
1
2
 3
�
alayer_metrics
bmetrics
cnon_trainable_variables
1trainable_variables

dlayers
elayer_regularization_losses
2regularization_losses
3	variables
 
 
 
�
flayer_metrics
gmetrics
hnon_trainable_variables
5trainable_variables

ilayers
jlayer_regularization_losses
6regularization_losses
7	variables
 
 
 
�
klayer_metrics
lmetrics
mnon_trainable_variables
9trainable_variables

nlayers
olayer_regularization_losses
:regularization_losses
;	variables
 
 
 
�
player_metrics
qmetrics
rnon_trainable_variables
=trainable_variables

slayers
tlayer_regularization_losses
>regularization_losses
?	variables
 
 
 
�
ulayer_metrics
vmetrics
wnon_trainable_variables
Atrainable_variables

xlayers
ylayer_regularization_losses
Bregularization_losses
C	variables

0
1
 

0
1
�
zlayer_metrics
{metrics
|non_trainable_variables
Etrainable_variables

}layers
~layer_regularization_losses
Fregularization_losses
G	variables
 

0
1
 

0
1
!2
"3
�
layer_metrics
�metrics
�non_trainable_variables
Jtrainable_variables
�layers
 �layer_regularization_losses
Kregularization_losses
L	variables
 
 
 
�
�layer_metrics
�metrics
�non_trainable_variables
Ntrainable_variables
�layers
 �layer_regularization_losses
Oregularization_losses
P	variables
 
 

0
 1
!2
"3
N
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
 
 
 
 
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
0
 1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
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
!0
"1
 
 
 
 
 
 
 
�
serving_default_left_inputsPlaceholder*,
_output_shapes
:����������*
dtype0*!
shape:����������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_left_inputsstream_0_conv_1/kernelstream_0_conv_1/bias#batch_normalization/moving_variancebatch_normalization/gammabatch_normalization/moving_meanbatch_normalization/betadense_1/kerneldense_1/bias%batch_normalization_1/moving_variancebatch_normalization_1/gamma!batch_normalization_1/moving_meanbatch_normalization_1/beta*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������T*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *-
f(R&
$__inference_signature_wrapper_484195
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename*stream_0_conv_1/kernel/Read/ReadVariableOp(stream_0_conv_1/bias/Read/ReadVariableOp-batch_normalization/gamma/Read/ReadVariableOp,batch_normalization/beta/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOp/batch_normalization_1/gamma/Read/ReadVariableOp.batch_normalization_1/beta/Read/ReadVariableOp3batch_normalization/moving_mean/Read/ReadVariableOp7batch_normalization/moving_variance/Read/ReadVariableOp5batch_normalization_1/moving_mean/Read/ReadVariableOp9batch_normalization_1/moving_variance/Read/ReadVariableOpConst*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *(
f#R!
__inference__traced_save_485471
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamestream_0_conv_1/kernelstream_0_conv_1/biasbatch_normalization/gammabatch_normalization/betadense_1/kerneldense_1/biasbatch_normalization_1/gammabatch_normalization_1/betabatch_normalization/moving_mean#batch_normalization/moving_variance!batch_normalization_1/moving_mean%batch_normalization_1/moving_variance*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *+
f&R$
"__inference__traced_restore_485517��
�*
�
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_482962

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
�
j
K__inference_stream_0_drop_1_layer_call_and_return_conditional_losses_485220

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
:����������@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:����������@*
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
:����������@2
dropout/GreaterEqual�
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:����������@2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:����������@2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:����������@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������@:T P
,
_output_shapes
:����������@
 
_user_specified_nameinputs
�+
�
O__inference_batch_normalization_layer_call_and_return_conditional_losses_483338

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
:����������@2
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
:����������@2
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
:����������@2
batchnorm/add_1s
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*,
_output_shapes
:����������@2

Identity�
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������@: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:T P
,
_output_shapes
:����������@
 
_user_specified_nameinputs
�&
�
__inference__traced_save_485471
file_prefix5
1savev2_stream_0_conv_1_kernel_read_readvariableop3
/savev2_stream_0_conv_1_bias_read_readvariableop8
4savev2_batch_normalization_gamma_read_readvariableop7
3savev2_batch_normalization_beta_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop:
6savev2_batch_normalization_1_gamma_read_readvariableop9
5savev2_batch_normalization_1_beta_read_readvariableop>
:savev2_batch_normalization_moving_mean_read_readvariableopB
>savev2_batch_normalization_moving_variance_read_readvariableop@
<savev2_batch_normalization_1_moving_mean_read_readvariableopD
@savev2_batch_normalization_1_moving_variance_read_readvariableop
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
ShardedFilename�
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*-
value$B"B B B B B B B B B B B B B 2
SaveV2/shape_and_slices�
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:01savev2_stream_0_conv_1_kernel_read_readvariableop/savev2_stream_0_conv_1_bias_read_readvariableop4savev2_batch_normalization_gamma_read_readvariableop3savev2_batch_normalization_beta_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop6savev2_batch_normalization_1_gamma_read_readvariableop5savev2_batch_normalization_1_beta_read_readvariableop:savev2_batch_normalization_moving_mean_read_readvariableop>savev2_batch_normalization_moving_variance_read_readvariableop<savev2_batch_normalization_1_moving_mean_read_readvariableop@savev2_batch_normalization_1_moving_variance_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
22
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

identity_1Identity_1:output:0*k
_input_shapesZ
X: :@:@:@:@:@T:T:T:T:@:@:T:T: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:($
"
_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@:$ 

_output_shapes

:@T: 

_output_shapes
:T: 

_output_shapes
:T: 

_output_shapes
:T: 	

_output_shapes
:@: 


_output_shapes
:@: 

_output_shapes
:T: 

_output_shapes
:T:

_output_shapes
: 
�
�
6__inference_batch_normalization_1_layer_call_fn_485326

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
*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_4829622
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
4__inference_batch_normalization_layer_call_fn_485062

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
:����������@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_4831002
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:����������@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:����������@
 
_user_specified_nameinputs
�
m
4__inference_stream_0_input_drop_layer_call_fn_484970

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
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *X
fSRQ
O__inference_stream_0_input_drop_layer_call_and_return_conditional_losses_4833792
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:����������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
j
K__inference_dense_1_dropout_layer_call_and_return_conditional_losses_483252

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
�
4__inference_batch_normalization_layer_call_fn_485049

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
*-
config_proto

CPU

GPU 2J 8� *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_4827762
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
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_485380

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
�
L
0__inference_stream_0_drop_1_layer_call_fn_485198

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
:����������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_stream_0_drop_1_layer_call_and_return_conditional_losses_4831222
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:����������@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������@:T P
,
_output_shapes
:����������@
 
_user_specified_nameinputs
�
j
K__inference_dense_1_dropout_layer_call_and_return_conditional_losses_485269

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
j
N__inference_dense_activation_1_layer_call_and_return_conditional_losses_483174

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
�t
�
E__inference_basemodel_layer_call_and_return_conditional_losses_484839
inputs_0Q
;stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource:@=
/stream_0_conv_1_biasadd_readvariableop_resource:@C
5batch_normalization_batchnorm_readvariableop_resource:@G
9batch_normalization_batchnorm_mul_readvariableop_resource:@E
7batch_normalization_batchnorm_readvariableop_1_resource:@E
7batch_normalization_batchnorm_readvariableop_2_resource:@8
&dense_1_matmul_readvariableop_resource:@T5
'dense_1_biasadd_readvariableop_resource:TE
7batch_normalization_1_batchnorm_readvariableop_resource:TI
;batch_normalization_1_batchnorm_mul_readvariableop_resource:TG
9batch_normalization_1_batchnorm_readvariableop_1_resource:TG
9batch_normalization_1_batchnorm_readvariableop_2_resource:T
identity��,batch_normalization/batchnorm/ReadVariableOp�.batch_normalization/batchnorm/ReadVariableOp_1�.batch_normalization/batchnorm/ReadVariableOp_2�0batch_normalization/batchnorm/mul/ReadVariableOp�.batch_normalization_1/batchnorm/ReadVariableOp�0batch_normalization_1/batchnorm/ReadVariableOp_1�0batch_normalization_1/batchnorm/ReadVariableOp_2�2batch_normalization_1/batchnorm/mul/ReadVariableOp�dense_1/BiasAdd/ReadVariableOp�dense_1/MatMul/ReadVariableOp�-dense_1/kernel/Regularizer/Abs/ReadVariableOp�&stream_0_conv_1/BiasAdd/ReadVariableOp�2stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp�5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp�
stream_0_input_drop/IdentityIdentityinputs_0*
T0*,
_output_shapes
:����������2
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
:����������2#
!stream_0_conv_1/conv1d/ExpandDims�
2stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp;stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
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
:@2%
#stream_0_conv_1/conv1d/ExpandDims_1�
stream_0_conv_1/conv1dConv2D*stream_0_conv_1/conv1d/ExpandDims:output:0,stream_0_conv_1/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:����������@*
paddingSAME*
strides
2
stream_0_conv_1/conv1d�
stream_0_conv_1/conv1d/SqueezeSqueezestream_0_conv_1/conv1d:output:0*
T0*,
_output_shapes
:����������@*
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
:����������@2
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
T0*,
_output_shapes
:����������@2%
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
:����������@2%
#batch_normalization/batchnorm/add_1�
activation/ReluRelu'batch_normalization/batchnorm/add_1:z:0*
T0*,
_output_shapes
:����������@2
activation/Relu�
stream_0_drop_1/IdentityIdentityactivation/Relu:activations:0*
T0*,
_output_shapes
:����������@2
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
dense_1_dropout/IdentityIdentity&global_average_pooling1d/Mean:output:0*
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
.batch_normalization_1/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_1_batchnorm_readvariableop_resource*
_output_shapes
:T*
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
:T2%
#batch_normalization_1/batchnorm/add�
%batch_normalization_1/batchnorm/RsqrtRsqrt'batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes
:T2'
%batch_normalization_1/batchnorm/Rsqrt�
2batch_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes
:T*
dtype024
2batch_normalization_1/batchnorm/mul/ReadVariableOp�
#batch_normalization_1/batchnorm/mulMul)batch_normalization_1/batchnorm/Rsqrt:y:0:batch_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:T2%
#batch_normalization_1/batchnorm/mul�
%batch_normalization_1/batchnorm/mul_1Muldense_1/BiasAdd:output:0'batch_normalization_1/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������T2'
%batch_normalization_1/batchnorm/mul_1�
0batch_normalization_1/batchnorm/ReadVariableOp_1ReadVariableOp9batch_normalization_1_batchnorm_readvariableop_1_resource*
_output_shapes
:T*
dtype022
0batch_normalization_1/batchnorm/ReadVariableOp_1�
%batch_normalization_1/batchnorm/mul_2Mul8batch_normalization_1/batchnorm/ReadVariableOp_1:value:0'batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes
:T2'
%batch_normalization_1/batchnorm/mul_2�
0batch_normalization_1/batchnorm/ReadVariableOp_2ReadVariableOp9batch_normalization_1_batchnorm_readvariableop_2_resource*
_output_shapes
:T*
dtype022
0batch_normalization_1/batchnorm/ReadVariableOp_2�
#batch_normalization_1/batchnorm/subSub8batch_normalization_1/batchnorm/ReadVariableOp_2:value:0)batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes
:T2%
#batch_normalization_1/batchnorm/sub�
%batch_normalization_1/batchnorm/add_1AddV2)batch_normalization_1/batchnorm/mul_1:z:0'batch_normalization_1/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������T2'
%batch_normalization_1/batchnorm/add_1�
dense_activation_1/SigmoidSigmoid)batch_normalization_1/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������T2
dense_activation_1/Sigmoid�
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp;stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype027
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp�
&stream_0_conv_1/kernel/Regularizer/AbsAbs=stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
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

Identity�
NoOpNoOp-^batch_normalization/batchnorm/ReadVariableOp/^batch_normalization/batchnorm/ReadVariableOp_1/^batch_normalization/batchnorm/ReadVariableOp_21^batch_normalization/batchnorm/mul/ReadVariableOp/^batch_normalization_1/batchnorm/ReadVariableOp1^batch_normalization_1/batchnorm/ReadVariableOp_11^batch_normalization_1/batchnorm/ReadVariableOp_23^batch_normalization_1/batchnorm/mul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp.^dense_1/kernel/Regularizer/Abs/ReadVariableOp'^stream_0_conv_1/BiasAdd/ReadVariableOp3^stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:����������: : : : : : : : : : : : 2\
,batch_normalization/batchnorm/ReadVariableOp,batch_normalization/batchnorm/ReadVariableOp2`
.batch_normalization/batchnorm/ReadVariableOp_1.batch_normalization/batchnorm/ReadVariableOp_12`
.batch_normalization/batchnorm/ReadVariableOp_2.batch_normalization/batchnorm/ReadVariableOp_22d
0batch_normalization/batchnorm/mul/ReadVariableOp0batch_normalization/batchnorm/mul/ReadVariableOp2`
.batch_normalization_1/batchnorm/ReadVariableOp.batch_normalization_1/batchnorm/ReadVariableOp2d
0batch_normalization_1/batchnorm/ReadVariableOp_10batch_normalization_1/batchnorm/ReadVariableOp_12d
0batch_normalization_1/batchnorm/ReadVariableOp_20batch_normalization_1/batchnorm/ReadVariableOp_22h
2batch_normalization_1/batchnorm/mul/ReadVariableOp2batch_normalization_1/batchnorm/mul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2^
-dense_1/kernel/Regularizer/Abs/ReadVariableOp-dense_1/kernel/Regularizer/Abs/ReadVariableOp2P
&stream_0_conv_1/BiasAdd/ReadVariableOp&stream_0_conv_1/BiasAdd/ReadVariableOp2h
2stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp2stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp2n
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:V R
,
_output_shapes
:����������
"
_user_specified_name
inputs/0
�
i
K__inference_stream_0_drop_1_layer_call_and_return_conditional_losses_483122

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:����������@2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:����������@2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������@:T P
,
_output_shapes
:����������@
 
_user_specified_nameinputs
��
�
E__inference_basemodel_layer_call_and_return_conditional_losses_483915

inputsQ
;stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource:@=
/stream_0_conv_1_biasadd_readvariableop_resource:@I
;batch_normalization_assignmovingavg_readvariableop_resource:@K
=batch_normalization_assignmovingavg_1_readvariableop_resource:@G
9batch_normalization_batchnorm_mul_readvariableop_resource:@C
5batch_normalization_batchnorm_readvariableop_resource:@8
&dense_1_matmul_readvariableop_resource:@T5
'dense_1_biasadd_readvariableop_resource:TK
=batch_normalization_1_assignmovingavg_readvariableop_resource:TM
?batch_normalization_1_assignmovingavg_1_readvariableop_resource:TI
;batch_normalization_1_batchnorm_mul_readvariableop_resource:TE
7batch_normalization_1_batchnorm_readvariableop_resource:T
identity��#batch_normalization/AssignMovingAvg�2batch_normalization/AssignMovingAvg/ReadVariableOp�%batch_normalization/AssignMovingAvg_1�4batch_normalization/AssignMovingAvg_1/ReadVariableOp�,batch_normalization/batchnorm/ReadVariableOp�0batch_normalization/batchnorm/mul/ReadVariableOp�%batch_normalization_1/AssignMovingAvg�4batch_normalization_1/AssignMovingAvg/ReadVariableOp�'batch_normalization_1/AssignMovingAvg_1�6batch_normalization_1/AssignMovingAvg_1/ReadVariableOp�.batch_normalization_1/batchnorm/ReadVariableOp�2batch_normalization_1/batchnorm/mul/ReadVariableOp�dense_1/BiasAdd/ReadVariableOp�dense_1/MatMul/ReadVariableOp�-dense_1/kernel/Regularizer/Abs/ReadVariableOp�&stream_0_conv_1/BiasAdd/ReadVariableOp�2stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp�5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp�
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
:����������2!
stream_0_input_drop/dropout/Mul|
!stream_0_input_drop/dropout/ShapeShapeinputs*
T0*
_output_shapes
:2#
!stream_0_input_drop/dropout/Shape�
8stream_0_input_drop/dropout/random_uniform/RandomUniformRandomUniform*stream_0_input_drop/dropout/Shape:output:0*
T0*,
_output_shapes
:����������*
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
:����������2*
(stream_0_input_drop/dropout/GreaterEqual�
 stream_0_input_drop/dropout/CastCast,stream_0_input_drop/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:����������2"
 stream_0_input_drop/dropout/Cast�
!stream_0_input_drop/dropout/Mul_1Mul#stream_0_input_drop/dropout/Mul:z:0$stream_0_input_drop/dropout/Cast:y:0*
T0*,
_output_shapes
:����������2#
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
:����������2#
!stream_0_conv_1/conv1d/ExpandDims�
2stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp;stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
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
:@2%
#stream_0_conv_1/conv1d/ExpandDims_1�
stream_0_conv_1/conv1dConv2D*stream_0_conv_1/conv1d/ExpandDims:output:0,stream_0_conv_1/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:����������@*
paddingSAME*
strides
2
stream_0_conv_1/conv1d�
stream_0_conv_1/conv1d/SqueezeSqueezestream_0_conv_1/conv1d:output:0*
T0*,
_output_shapes
:����������@*
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
:����������@2
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
T0*,
_output_shapes
:����������@2/
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
:����������@2%
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
:����������@2%
#batch_normalization/batchnorm/add_1�
activation/ReluRelu'batch_normalization/batchnorm/add_1:z:0*
T0*,
_output_shapes
:����������@2
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
:����������@2
stream_0_drop_1/dropout/Mul�
stream_0_drop_1/dropout/ShapeShapeactivation/Relu:activations:0*
T0*
_output_shapes
:2
stream_0_drop_1/dropout/Shape�
4stream_0_drop_1/dropout/random_uniform/RandomUniformRandomUniform&stream_0_drop_1/dropout/Shape:output:0*
T0*,
_output_shapes
:����������@*
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
:����������@2&
$stream_0_drop_1/dropout/GreaterEqual�
stream_0_drop_1/dropout/CastCast(stream_0_drop_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:����������@2
stream_0_drop_1/dropout/Cast�
stream_0_drop_1/dropout/Mul_1Mulstream_0_drop_1/dropout/Mul:z:0 stream_0_drop_1/dropout/Cast:y:0*
T0*,
_output_shapes
:����������@2
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
dense_1_dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
dense_1_dropout/dropout/Const�
dense_1_dropout/dropout/MulMul&global_average_pooling1d/Mean:output:0&dense_1_dropout/dropout/Const:output:0*
T0*'
_output_shapes
:���������@2
dense_1_dropout/dropout/Mul�
dense_1_dropout/dropout/ShapeShape&global_average_pooling1d/Mean:output:0*
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
4batch_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 26
4batch_normalization_1/moments/mean/reduction_indices�
"batch_normalization_1/moments/meanMeandense_1/BiasAdd:output:0=batch_normalization_1/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:T*
	keep_dims(2$
"batch_normalization_1/moments/mean�
*batch_normalization_1/moments/StopGradientStopGradient+batch_normalization_1/moments/mean:output:0*
T0*
_output_shapes

:T2,
*batch_normalization_1/moments/StopGradient�
/batch_normalization_1/moments/SquaredDifferenceSquaredDifferencedense_1/BiasAdd:output:03batch_normalization_1/moments/StopGradient:output:0*
T0*'
_output_shapes
:���������T21
/batch_normalization_1/moments/SquaredDifference�
8batch_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2:
8batch_normalization_1/moments/variance/reduction_indices�
&batch_normalization_1/moments/varianceMean3batch_normalization_1/moments/SquaredDifference:z:0Abatch_normalization_1/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:T*
	keep_dims(2(
&batch_normalization_1/moments/variance�
%batch_normalization_1/moments/SqueezeSqueeze+batch_normalization_1/moments/mean:output:0*
T0*
_output_shapes
:T*
squeeze_dims
 2'
%batch_normalization_1/moments/Squeeze�
'batch_normalization_1/moments/Squeeze_1Squeeze/batch_normalization_1/moments/variance:output:0*
T0*
_output_shapes
:T*
squeeze_dims
 2)
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
:T*
dtype026
4batch_normalization_1/AssignMovingAvg/ReadVariableOp�
)batch_normalization_1/AssignMovingAvg/subSub<batch_normalization_1/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_1/moments/Squeeze:output:0*
T0*
_output_shapes
:T2+
)batch_normalization_1/AssignMovingAvg/sub�
)batch_normalization_1/AssignMovingAvg/mulMul-batch_normalization_1/AssignMovingAvg/sub:z:04batch_normalization_1/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:T2+
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
:T*
dtype028
6batch_normalization_1/AssignMovingAvg_1/ReadVariableOp�
+batch_normalization_1/AssignMovingAvg_1/subSub>batch_normalization_1/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_1/moments/Squeeze_1:output:0*
T0*
_output_shapes
:T2-
+batch_normalization_1/AssignMovingAvg_1/sub�
+batch_normalization_1/AssignMovingAvg_1/mulMul/batch_normalization_1/AssignMovingAvg_1/sub:z:06batch_normalization_1/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:T2-
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
:T2%
#batch_normalization_1/batchnorm/add�
%batch_normalization_1/batchnorm/RsqrtRsqrt'batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes
:T2'
%batch_normalization_1/batchnorm/Rsqrt�
2batch_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes
:T*
dtype024
2batch_normalization_1/batchnorm/mul/ReadVariableOp�
#batch_normalization_1/batchnorm/mulMul)batch_normalization_1/batchnorm/Rsqrt:y:0:batch_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:T2%
#batch_normalization_1/batchnorm/mul�
%batch_normalization_1/batchnorm/mul_1Muldense_1/BiasAdd:output:0'batch_normalization_1/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������T2'
%batch_normalization_1/batchnorm/mul_1�
%batch_normalization_1/batchnorm/mul_2Mul.batch_normalization_1/moments/Squeeze:output:0'batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes
:T2'
%batch_normalization_1/batchnorm/mul_2�
.batch_normalization_1/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_1_batchnorm_readvariableop_resource*
_output_shapes
:T*
dtype020
.batch_normalization_1/batchnorm/ReadVariableOp�
#batch_normalization_1/batchnorm/subSub6batch_normalization_1/batchnorm/ReadVariableOp:value:0)batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes
:T2%
#batch_normalization_1/batchnorm/sub�
%batch_normalization_1/batchnorm/add_1AddV2)batch_normalization_1/batchnorm/mul_1:z:0'batch_normalization_1/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������T2'
%batch_normalization_1/batchnorm/add_1�
dense_activation_1/SigmoidSigmoid)batch_normalization_1/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������T2
dense_activation_1/Sigmoid�
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp;stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype027
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp�
&stream_0_conv_1/kernel/Regularizer/AbsAbs=stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
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

Identity�
NoOpNoOp$^batch_normalization/AssignMovingAvg3^batch_normalization/AssignMovingAvg/ReadVariableOp&^batch_normalization/AssignMovingAvg_15^batch_normalization/AssignMovingAvg_1/ReadVariableOp-^batch_normalization/batchnorm/ReadVariableOp1^batch_normalization/batchnorm/mul/ReadVariableOp&^batch_normalization_1/AssignMovingAvg5^batch_normalization_1/AssignMovingAvg/ReadVariableOp(^batch_normalization_1/AssignMovingAvg_17^batch_normalization_1/AssignMovingAvg_1/ReadVariableOp/^batch_normalization_1/batchnorm/ReadVariableOp3^batch_normalization_1/batchnorm/mul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp.^dense_1/kernel/Regularizer/Abs/ReadVariableOp'^stream_0_conv_1/BiasAdd/ReadVariableOp3^stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:����������: : : : : : : : : : : : 2J
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
2batch_normalization_1/batchnorm/mul/ReadVariableOp2batch_normalization_1/batchnorm/mul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2^
-dense_1/kernel/Regularizer/Abs/ReadVariableOp-dense_1/kernel/Regularizer/Abs/ReadVariableOp2P
&stream_0_conv_1/BiasAdd/ReadVariableOp&stream_0_conv_1/BiasAdd/ReadVariableOp2h
2stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp2stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp2n
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
i
K__inference_dense_1_dropout_layer_call_and_return_conditional_losses_483136

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
�!
�
C__inference_model_1_layer_call_and_return_conditional_losses_484152
left_inputs&
basemodel_484114:@
basemodel_484116:@
basemodel_484118:@
basemodel_484120:@
basemodel_484122:@
basemodel_484124:@"
basemodel_484126:@T
basemodel_484128:T
basemodel_484130:T
basemodel_484132:T
basemodel_484134:T
basemodel_484136:T
identity��!basemodel/StatefulPartitionedCall�-dense_1/kernel/Regularizer/Abs/ReadVariableOp�5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp�
!basemodel/StatefulPartitionedCallStatefulPartitionedCallleft_inputsbasemodel_484114basemodel_484116basemodel_484118basemodel_484120basemodel_484122basemodel_484124basemodel_484126basemodel_484128basemodel_484130basemodel_484132basemodel_484134basemodel_484136*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������T**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_basemodel_layer_call_and_return_conditional_losses_4839152#
!basemodel/StatefulPartitionedCall�
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpbasemodel_484114*"
_output_shapes
:@*
dtype027
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp�
&stream_0_conv_1/kernel/Regularizer/AbsAbs=stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
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
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpbasemodel_484126*
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
IdentityIdentity*basemodel/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������T2

Identity�
NoOpNoOp"^basemodel/StatefulPartitionedCall.^dense_1/kernel/Regularizer/Abs/ReadVariableOp6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:����������: : : : : : : : : : : : 2F
!basemodel/StatefulPartitionedCall!basemodel/StatefulPartitionedCall2^
-dense_1/kernel/Regularizer/Abs/ReadVariableOp-dense_1/kernel/Regularizer/Abs/ReadVariableOp2n
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:Y U
,
_output_shapes
:����������
%
_user_specified_nameleft_inputs
�
�
6__inference_batch_normalization_1_layer_call_fn_485313

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
*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_4829022
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
�
i
K__inference_stream_0_drop_1_layer_call_and_return_conditional_losses_485208

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:����������@2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:����������@2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������@:T P
,
_output_shapes
:����������@
 
_user_specified_nameinputs
�
�
O__inference_batch_normalization_layer_call_and_return_conditional_losses_483100

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
:����������@2
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
:����������@2
batchnorm/add_1s
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*,
_output_shapes
:����������@2

Identity�
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������@: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:T P
,
_output_shapes
:����������@
 
_user_specified_nameinputs
�
�
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_482902

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
�
(__inference_model_1_layer_call_fn_484253

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:@
	unknown_5:@T
	unknown_6:T
	unknown_7:T
	unknown_8:T
	unknown_9:T

unknown_10:T
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������T**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_model_1_layer_call_and_return_conditional_losses_4840142
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
_construction_contextkEagerRuntime*C
_input_shapes2
0:����������: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
��
�
C__inference_model_1_layer_call_and_return_conditional_losses_484446

inputs[
Ebasemodel_stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource:@G
9basemodel_stream_0_conv_1_biasadd_readvariableop_resource:@S
Ebasemodel_batch_normalization_assignmovingavg_readvariableop_resource:@U
Gbasemodel_batch_normalization_assignmovingavg_1_readvariableop_resource:@Q
Cbasemodel_batch_normalization_batchnorm_mul_readvariableop_resource:@M
?basemodel_batch_normalization_batchnorm_readvariableop_resource:@B
0basemodel_dense_1_matmul_readvariableop_resource:@T?
1basemodel_dense_1_biasadd_readvariableop_resource:TU
Gbasemodel_batch_normalization_1_assignmovingavg_readvariableop_resource:TW
Ibasemodel_batch_normalization_1_assignmovingavg_1_readvariableop_resource:TS
Ebasemodel_batch_normalization_1_batchnorm_mul_readvariableop_resource:TO
Abasemodel_batch_normalization_1_batchnorm_readvariableop_resource:T
identity��-basemodel/batch_normalization/AssignMovingAvg�<basemodel/batch_normalization/AssignMovingAvg/ReadVariableOp�/basemodel/batch_normalization/AssignMovingAvg_1�>basemodel/batch_normalization/AssignMovingAvg_1/ReadVariableOp�6basemodel/batch_normalization/batchnorm/ReadVariableOp�:basemodel/batch_normalization/batchnorm/mul/ReadVariableOp�/basemodel/batch_normalization_1/AssignMovingAvg�>basemodel/batch_normalization_1/AssignMovingAvg/ReadVariableOp�1basemodel/batch_normalization_1/AssignMovingAvg_1�@basemodel/batch_normalization_1/AssignMovingAvg_1/ReadVariableOp�8basemodel/batch_normalization_1/batchnorm/ReadVariableOp�<basemodel/batch_normalization_1/batchnorm/mul/ReadVariableOp�(basemodel/dense_1/BiasAdd/ReadVariableOp�'basemodel/dense_1/MatMul/ReadVariableOp�0basemodel/stream_0_conv_1/BiasAdd/ReadVariableOp�<basemodel/stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp�-dense_1/kernel/Regularizer/Abs/ReadVariableOp�5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp�
+basemodel/stream_0_input_drop/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2-
+basemodel/stream_0_input_drop/dropout/Const�
)basemodel/stream_0_input_drop/dropout/MulMulinputs4basemodel/stream_0_input_drop/dropout/Const:output:0*
T0*,
_output_shapes
:����������2+
)basemodel/stream_0_input_drop/dropout/Mul�
+basemodel/stream_0_input_drop/dropout/ShapeShapeinputs*
T0*
_output_shapes
:2-
+basemodel/stream_0_input_drop/dropout/Shape�
Bbasemodel/stream_0_input_drop/dropout/random_uniform/RandomUniformRandomUniform4basemodel/stream_0_input_drop/dropout/Shape:output:0*
T0*,
_output_shapes
:����������*
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
:����������24
2basemodel/stream_0_input_drop/dropout/GreaterEqual�
*basemodel/stream_0_input_drop/dropout/CastCast6basemodel/stream_0_input_drop/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:����������2,
*basemodel/stream_0_input_drop/dropout/Cast�
+basemodel/stream_0_input_drop/dropout/Mul_1Mul-basemodel/stream_0_input_drop/dropout/Mul:z:0.basemodel/stream_0_input_drop/dropout/Cast:y:0*
T0*,
_output_shapes
:����������2-
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
:����������2-
+basemodel/stream_0_conv_1/conv1d/ExpandDims�
<basemodel/stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpEbasemodel_stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
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
:@2/
-basemodel/stream_0_conv_1/conv1d/ExpandDims_1�
 basemodel/stream_0_conv_1/conv1dConv2D4basemodel/stream_0_conv_1/conv1d/ExpandDims:output:06basemodel/stream_0_conv_1/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:����������@*
paddingSAME*
strides
2"
 basemodel/stream_0_conv_1/conv1d�
(basemodel/stream_0_conv_1/conv1d/SqueezeSqueeze)basemodel/stream_0_conv_1/conv1d:output:0*
T0*,
_output_shapes
:����������@*
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
:����������@2#
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
T0*,
_output_shapes
:����������@29
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
:����������@2/
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
:����������@2/
-basemodel/batch_normalization/batchnorm/add_1�
basemodel/activation/ReluRelu1basemodel/batch_normalization/batchnorm/add_1:z:0*
T0*,
_output_shapes
:����������@2
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
:����������@2'
%basemodel/stream_0_drop_1/dropout/Mul�
'basemodel/stream_0_drop_1/dropout/ShapeShape'basemodel/activation/Relu:activations:0*
T0*
_output_shapes
:2)
'basemodel/stream_0_drop_1/dropout/Shape�
>basemodel/stream_0_drop_1/dropout/random_uniform/RandomUniformRandomUniform0basemodel/stream_0_drop_1/dropout/Shape:output:0*
T0*,
_output_shapes
:����������@*
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
:����������@20
.basemodel/stream_0_drop_1/dropout/GreaterEqual�
&basemodel/stream_0_drop_1/dropout/CastCast2basemodel/stream_0_drop_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:����������@2(
&basemodel/stream_0_drop_1/dropout/Cast�
'basemodel/stream_0_drop_1/dropout/Mul_1Mul)basemodel/stream_0_drop_1/dropout/Mul:z:0*basemodel/stream_0_drop_1/dropout/Cast:y:0*
T0*,
_output_shapes
:����������@2)
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
'basemodel/dense_1_dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2)
'basemodel/dense_1_dropout/dropout/Const�
%basemodel/dense_1_dropout/dropout/MulMul0basemodel/global_average_pooling1d/Mean:output:00basemodel/dense_1_dropout/dropout/Const:output:0*
T0*'
_output_shapes
:���������@2'
%basemodel/dense_1_dropout/dropout/Mul�
'basemodel/dense_1_dropout/dropout/ShapeShape0basemodel/global_average_pooling1d/Mean:output:0*
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
>basemodel/batch_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2@
>basemodel/batch_normalization_1/moments/mean/reduction_indices�
,basemodel/batch_normalization_1/moments/meanMean"basemodel/dense_1/BiasAdd:output:0Gbasemodel/batch_normalization_1/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:T*
	keep_dims(2.
,basemodel/batch_normalization_1/moments/mean�
4basemodel/batch_normalization_1/moments/StopGradientStopGradient5basemodel/batch_normalization_1/moments/mean:output:0*
T0*
_output_shapes

:T26
4basemodel/batch_normalization_1/moments/StopGradient�
9basemodel/batch_normalization_1/moments/SquaredDifferenceSquaredDifference"basemodel/dense_1/BiasAdd:output:0=basemodel/batch_normalization_1/moments/StopGradient:output:0*
T0*'
_output_shapes
:���������T2;
9basemodel/batch_normalization_1/moments/SquaredDifference�
Bbasemodel/batch_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2D
Bbasemodel/batch_normalization_1/moments/variance/reduction_indices�
0basemodel/batch_normalization_1/moments/varianceMean=basemodel/batch_normalization_1/moments/SquaredDifference:z:0Kbasemodel/batch_normalization_1/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:T*
	keep_dims(22
0basemodel/batch_normalization_1/moments/variance�
/basemodel/batch_normalization_1/moments/SqueezeSqueeze5basemodel/batch_normalization_1/moments/mean:output:0*
T0*
_output_shapes
:T*
squeeze_dims
 21
/basemodel/batch_normalization_1/moments/Squeeze�
1basemodel/batch_normalization_1/moments/Squeeze_1Squeeze9basemodel/batch_normalization_1/moments/variance:output:0*
T0*
_output_shapes
:T*
squeeze_dims
 23
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
:T*
dtype02@
>basemodel/batch_normalization_1/AssignMovingAvg/ReadVariableOp�
3basemodel/batch_normalization_1/AssignMovingAvg/subSubFbasemodel/batch_normalization_1/AssignMovingAvg/ReadVariableOp:value:08basemodel/batch_normalization_1/moments/Squeeze:output:0*
T0*
_output_shapes
:T25
3basemodel/batch_normalization_1/AssignMovingAvg/sub�
3basemodel/batch_normalization_1/AssignMovingAvg/mulMul7basemodel/batch_normalization_1/AssignMovingAvg/sub:z:0>basemodel/batch_normalization_1/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:T25
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
:T*
dtype02B
@basemodel/batch_normalization_1/AssignMovingAvg_1/ReadVariableOp�
5basemodel/batch_normalization_1/AssignMovingAvg_1/subSubHbasemodel/batch_normalization_1/AssignMovingAvg_1/ReadVariableOp:value:0:basemodel/batch_normalization_1/moments/Squeeze_1:output:0*
T0*
_output_shapes
:T27
5basemodel/batch_normalization_1/AssignMovingAvg_1/sub�
5basemodel/batch_normalization_1/AssignMovingAvg_1/mulMul9basemodel/batch_normalization_1/AssignMovingAvg_1/sub:z:0@basemodel/batch_normalization_1/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:T27
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
:T2/
-basemodel/batch_normalization_1/batchnorm/add�
/basemodel/batch_normalization_1/batchnorm/RsqrtRsqrt1basemodel/batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes
:T21
/basemodel/batch_normalization_1/batchnorm/Rsqrt�
<basemodel/batch_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOpEbasemodel_batch_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes
:T*
dtype02>
<basemodel/batch_normalization_1/batchnorm/mul/ReadVariableOp�
-basemodel/batch_normalization_1/batchnorm/mulMul3basemodel/batch_normalization_1/batchnorm/Rsqrt:y:0Dbasemodel/batch_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:T2/
-basemodel/batch_normalization_1/batchnorm/mul�
/basemodel/batch_normalization_1/batchnorm/mul_1Mul"basemodel/dense_1/BiasAdd:output:01basemodel/batch_normalization_1/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������T21
/basemodel/batch_normalization_1/batchnorm/mul_1�
/basemodel/batch_normalization_1/batchnorm/mul_2Mul8basemodel/batch_normalization_1/moments/Squeeze:output:01basemodel/batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes
:T21
/basemodel/batch_normalization_1/batchnorm/mul_2�
8basemodel/batch_normalization_1/batchnorm/ReadVariableOpReadVariableOpAbasemodel_batch_normalization_1_batchnorm_readvariableop_resource*
_output_shapes
:T*
dtype02:
8basemodel/batch_normalization_1/batchnorm/ReadVariableOp�
-basemodel/batch_normalization_1/batchnorm/subSub@basemodel/batch_normalization_1/batchnorm/ReadVariableOp:value:03basemodel/batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes
:T2/
-basemodel/batch_normalization_1/batchnorm/sub�
/basemodel/batch_normalization_1/batchnorm/add_1AddV23basemodel/batch_normalization_1/batchnorm/mul_1:z:01basemodel/batch_normalization_1/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������T21
/basemodel/batch_normalization_1/batchnorm/add_1�
$basemodel/dense_activation_1/SigmoidSigmoid3basemodel/batch_normalization_1/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������T2&
$basemodel/dense_activation_1/Sigmoid�
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpEbasemodel_stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype027
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp�
&stream_0_conv_1/kernel/Regularizer/AbsAbs=stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
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
dense_1/kernel/Regularizer/mul�
IdentityIdentity(basemodel/dense_activation_1/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:���������T2

Identity�
NoOpNoOp.^basemodel/batch_normalization/AssignMovingAvg=^basemodel/batch_normalization/AssignMovingAvg/ReadVariableOp0^basemodel/batch_normalization/AssignMovingAvg_1?^basemodel/batch_normalization/AssignMovingAvg_1/ReadVariableOp7^basemodel/batch_normalization/batchnorm/ReadVariableOp;^basemodel/batch_normalization/batchnorm/mul/ReadVariableOp0^basemodel/batch_normalization_1/AssignMovingAvg?^basemodel/batch_normalization_1/AssignMovingAvg/ReadVariableOp2^basemodel/batch_normalization_1/AssignMovingAvg_1A^basemodel/batch_normalization_1/AssignMovingAvg_1/ReadVariableOp9^basemodel/batch_normalization_1/batchnorm/ReadVariableOp=^basemodel/batch_normalization_1/batchnorm/mul/ReadVariableOp)^basemodel/dense_1/BiasAdd/ReadVariableOp(^basemodel/dense_1/MatMul/ReadVariableOp1^basemodel/stream_0_conv_1/BiasAdd/ReadVariableOp=^basemodel/stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp.^dense_1/kernel/Regularizer/Abs/ReadVariableOp6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:����������: : : : : : : : : : : : 2^
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
<basemodel/batch_normalization_1/batchnorm/mul/ReadVariableOp<basemodel/batch_normalization_1/batchnorm/mul/ReadVariableOp2T
(basemodel/dense_1/BiasAdd/ReadVariableOp(basemodel/dense_1/BiasAdd/ReadVariableOp2R
'basemodel/dense_1/MatMul/ReadVariableOp'basemodel/dense_1/MatMul/ReadVariableOp2d
0basemodel/stream_0_conv_1/BiasAdd/ReadVariableOp0basemodel/stream_0_conv_1/BiasAdd/ReadVariableOp2|
<basemodel/stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp<basemodel/stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp2^
-dense_1/kernel/Regularizer/Abs/ReadVariableOp-dense_1/kernel/Regularizer/Abs/ReadVariableOp2n
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�|
�
!__inference__wrapped_model_482692
left_inputsc
Mmodel_1_basemodel_stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource:@O
Amodel_1_basemodel_stream_0_conv_1_biasadd_readvariableop_resource:@U
Gmodel_1_basemodel_batch_normalization_batchnorm_readvariableop_resource:@Y
Kmodel_1_basemodel_batch_normalization_batchnorm_mul_readvariableop_resource:@W
Imodel_1_basemodel_batch_normalization_batchnorm_readvariableop_1_resource:@W
Imodel_1_basemodel_batch_normalization_batchnorm_readvariableop_2_resource:@J
8model_1_basemodel_dense_1_matmul_readvariableop_resource:@TG
9model_1_basemodel_dense_1_biasadd_readvariableop_resource:TW
Imodel_1_basemodel_batch_normalization_1_batchnorm_readvariableop_resource:T[
Mmodel_1_basemodel_batch_normalization_1_batchnorm_mul_readvariableop_resource:TY
Kmodel_1_basemodel_batch_normalization_1_batchnorm_readvariableop_1_resource:TY
Kmodel_1_basemodel_batch_normalization_1_batchnorm_readvariableop_2_resource:T
identity��>model_1/basemodel/batch_normalization/batchnorm/ReadVariableOp�@model_1/basemodel/batch_normalization/batchnorm/ReadVariableOp_1�@model_1/basemodel/batch_normalization/batchnorm/ReadVariableOp_2�Bmodel_1/basemodel/batch_normalization/batchnorm/mul/ReadVariableOp�@model_1/basemodel/batch_normalization_1/batchnorm/ReadVariableOp�Bmodel_1/basemodel/batch_normalization_1/batchnorm/ReadVariableOp_1�Bmodel_1/basemodel/batch_normalization_1/batchnorm/ReadVariableOp_2�Dmodel_1/basemodel/batch_normalization_1/batchnorm/mul/ReadVariableOp�0model_1/basemodel/dense_1/BiasAdd/ReadVariableOp�/model_1/basemodel/dense_1/MatMul/ReadVariableOp�8model_1/basemodel/stream_0_conv_1/BiasAdd/ReadVariableOp�Dmodel_1/basemodel/stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp�
.model_1/basemodel/stream_0_input_drop/IdentityIdentityleft_inputs*
T0*,
_output_shapes
:����������20
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
T0*0
_output_shapes
:����������25
3model_1/basemodel/stream_0_conv_1/conv1d/ExpandDims�
Dmodel_1/basemodel/stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpMmodel_1_basemodel_stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
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
:@27
5model_1/basemodel/stream_0_conv_1/conv1d/ExpandDims_1�
(model_1/basemodel/stream_0_conv_1/conv1dConv2D<model_1/basemodel/stream_0_conv_1/conv1d/ExpandDims:output:0>model_1/basemodel/stream_0_conv_1/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:����������@*
paddingSAME*
strides
2*
(model_1/basemodel/stream_0_conv_1/conv1d�
0model_1/basemodel/stream_0_conv_1/conv1d/SqueezeSqueeze1model_1/basemodel/stream_0_conv_1/conv1d:output:0*
T0*,
_output_shapes
:����������@*
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
T0*,
_output_shapes
:����������@2+
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
T0*,
_output_shapes
:����������@27
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
T0*,
_output_shapes
:����������@27
5model_1/basemodel/batch_normalization/batchnorm/add_1�
!model_1/basemodel/activation/ReluRelu9model_1/basemodel/batch_normalization/batchnorm/add_1:z:0*
T0*,
_output_shapes
:����������@2#
!model_1/basemodel/activation/Relu�
*model_1/basemodel/stream_0_drop_1/IdentityIdentity/model_1/basemodel/activation/Relu:activations:0*
T0*,
_output_shapes
:����������@2,
*model_1/basemodel/stream_0_drop_1/Identity�
Amodel_1/basemodel/global_average_pooling1d/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2C
Amodel_1/basemodel/global_average_pooling1d/Mean/reduction_indices�
/model_1/basemodel/global_average_pooling1d/MeanMean3model_1/basemodel/stream_0_drop_1/Identity:output:0Jmodel_1/basemodel/global_average_pooling1d/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:���������@21
/model_1/basemodel/global_average_pooling1d/Mean�
*model_1/basemodel/dense_1_dropout/IdentityIdentity8model_1/basemodel/global_average_pooling1d/Mean:output:0*
T0*'
_output_shapes
:���������@2,
*model_1/basemodel/dense_1_dropout/Identity�
/model_1/basemodel/dense_1/MatMul/ReadVariableOpReadVariableOp8model_1_basemodel_dense_1_matmul_readvariableop_resource*
_output_shapes

:@T*
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
@model_1/basemodel/batch_normalization_1/batchnorm/ReadVariableOpReadVariableOpImodel_1_basemodel_batch_normalization_1_batchnorm_readvariableop_resource*
_output_shapes
:T*
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
T0*
_output_shapes
:T27
5model_1/basemodel/batch_normalization_1/batchnorm/add�
7model_1/basemodel/batch_normalization_1/batchnorm/RsqrtRsqrt9model_1/basemodel/batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes
:T29
7model_1/basemodel/batch_normalization_1/batchnorm/Rsqrt�
Dmodel_1/basemodel/batch_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOpMmodel_1_basemodel_batch_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes
:T*
dtype02F
Dmodel_1/basemodel/batch_normalization_1/batchnorm/mul/ReadVariableOp�
5model_1/basemodel/batch_normalization_1/batchnorm/mulMul;model_1/basemodel/batch_normalization_1/batchnorm/Rsqrt:y:0Lmodel_1/basemodel/batch_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:T27
5model_1/basemodel/batch_normalization_1/batchnorm/mul�
7model_1/basemodel/batch_normalization_1/batchnorm/mul_1Mul*model_1/basemodel/dense_1/BiasAdd:output:09model_1/basemodel/batch_normalization_1/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������T29
7model_1/basemodel/batch_normalization_1/batchnorm/mul_1�
Bmodel_1/basemodel/batch_normalization_1/batchnorm/ReadVariableOp_1ReadVariableOpKmodel_1_basemodel_batch_normalization_1_batchnorm_readvariableop_1_resource*
_output_shapes
:T*
dtype02D
Bmodel_1/basemodel/batch_normalization_1/batchnorm/ReadVariableOp_1�
7model_1/basemodel/batch_normalization_1/batchnorm/mul_2MulJmodel_1/basemodel/batch_normalization_1/batchnorm/ReadVariableOp_1:value:09model_1/basemodel/batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes
:T29
7model_1/basemodel/batch_normalization_1/batchnorm/mul_2�
Bmodel_1/basemodel/batch_normalization_1/batchnorm/ReadVariableOp_2ReadVariableOpKmodel_1_basemodel_batch_normalization_1_batchnorm_readvariableop_2_resource*
_output_shapes
:T*
dtype02D
Bmodel_1/basemodel/batch_normalization_1/batchnorm/ReadVariableOp_2�
5model_1/basemodel/batch_normalization_1/batchnorm/subSubJmodel_1/basemodel/batch_normalization_1/batchnorm/ReadVariableOp_2:value:0;model_1/basemodel/batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes
:T27
5model_1/basemodel/batch_normalization_1/batchnorm/sub�
7model_1/basemodel/batch_normalization_1/batchnorm/add_1AddV2;model_1/basemodel/batch_normalization_1/batchnorm/mul_1:z:09model_1/basemodel/batch_normalization_1/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������T29
7model_1/basemodel/batch_normalization_1/batchnorm/add_1�
,model_1/basemodel/dense_activation_1/SigmoidSigmoid;model_1/basemodel/batch_normalization_1/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������T2.
,model_1/basemodel/dense_activation_1/Sigmoid�
IdentityIdentity0model_1/basemodel/dense_activation_1/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:���������T2

Identity�
NoOpNoOp?^model_1/basemodel/batch_normalization/batchnorm/ReadVariableOpA^model_1/basemodel/batch_normalization/batchnorm/ReadVariableOp_1A^model_1/basemodel/batch_normalization/batchnorm/ReadVariableOp_2C^model_1/basemodel/batch_normalization/batchnorm/mul/ReadVariableOpA^model_1/basemodel/batch_normalization_1/batchnorm/ReadVariableOpC^model_1/basemodel/batch_normalization_1/batchnorm/ReadVariableOp_1C^model_1/basemodel/batch_normalization_1/batchnorm/ReadVariableOp_2E^model_1/basemodel/batch_normalization_1/batchnorm/mul/ReadVariableOp1^model_1/basemodel/dense_1/BiasAdd/ReadVariableOp0^model_1/basemodel/dense_1/MatMul/ReadVariableOp9^model_1/basemodel/stream_0_conv_1/BiasAdd/ReadVariableOpE^model_1/basemodel/stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:����������: : : : : : : : : : : : 2�
>model_1/basemodel/batch_normalization/batchnorm/ReadVariableOp>model_1/basemodel/batch_normalization/batchnorm/ReadVariableOp2�
@model_1/basemodel/batch_normalization/batchnorm/ReadVariableOp_1@model_1/basemodel/batch_normalization/batchnorm/ReadVariableOp_12�
@model_1/basemodel/batch_normalization/batchnorm/ReadVariableOp_2@model_1/basemodel/batch_normalization/batchnorm/ReadVariableOp_22�
Bmodel_1/basemodel/batch_normalization/batchnorm/mul/ReadVariableOpBmodel_1/basemodel/batch_normalization/batchnorm/mul/ReadVariableOp2�
@model_1/basemodel/batch_normalization_1/batchnorm/ReadVariableOp@model_1/basemodel/batch_normalization_1/batchnorm/ReadVariableOp2�
Bmodel_1/basemodel/batch_normalization_1/batchnorm/ReadVariableOp_1Bmodel_1/basemodel/batch_normalization_1/batchnorm/ReadVariableOp_12�
Bmodel_1/basemodel/batch_normalization_1/batchnorm/ReadVariableOp_2Bmodel_1/basemodel/batch_normalization_1/batchnorm/ReadVariableOp_22�
Dmodel_1/basemodel/batch_normalization_1/batchnorm/mul/ReadVariableOpDmodel_1/basemodel/batch_normalization_1/batchnorm/mul/ReadVariableOp2d
0model_1/basemodel/dense_1/BiasAdd/ReadVariableOp0model_1/basemodel/dense_1/BiasAdd/ReadVariableOp2b
/model_1/basemodel/dense_1/MatMul/ReadVariableOp/model_1/basemodel/dense_1/MatMul/ReadVariableOp2t
8model_1/basemodel/stream_0_conv_1/BiasAdd/ReadVariableOp8model_1/basemodel/stream_0_conv_1/BiasAdd/ReadVariableOp2�
Dmodel_1/basemodel/stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOpDmodel_1/basemodel/stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp:Y U
,
_output_shapes
:����������
%
_user_specified_nameleft_inputs
�
p
T__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_485236

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
�
*__inference_basemodel_layer_call_fn_484545
inputs_0
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:@
	unknown_5:@T
	unknown_6:T
	unknown_7:T
	unknown_8:T
	unknown_9:T

unknown_10:T
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������T*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_basemodel_layer_call_and_return_conditional_losses_4836972
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
_construction_contextkEagerRuntime*C
_input_shapes2
0:����������: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
,
_output_shapes
:����������
"
_user_specified_name
inputs/0
�
�
O__inference_batch_normalization_layer_call_and_return_conditional_losses_485149

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
:����������@2
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
:����������@2
batchnorm/add_1s
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*,
_output_shapes
:����������@2

Identity�
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������@: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:T P
,
_output_shapes
:����������@
 
_user_specified_nameinputs
�
O
3__inference_dense_activation_1_layer_call_fn_485385

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
 *-
config_proto

CPU

GPU 2J 8� *W
fRRP
N__inference_dense_activation_1_layer_call_and_return_conditional_losses_4831742
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
�
�
K__inference_stream_0_conv_1_layer_call_and_return_conditional_losses_485023

inputsA
+conv1d_expanddims_1_readvariableop_resource:@-
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
:����������2
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
:����������@*
paddingSAME*
strides
2
conv1d�
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:����������@*
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
:����������@2	
BiasAdd�
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype027
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp�
&stream_0_conv_1/kernel/Regularizer/AbsAbs=stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
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
:����������@2

Identity�
NoOpNoOp^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp2n
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
j
N__inference_dense_activation_1_layer_call_and_return_conditional_losses_485390

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
�
i
0__inference_stream_0_drop_1_layer_call_fn_485203

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
:����������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_stream_0_drop_1_layer_call_and_return_conditional_losses_4832802
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:����������@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������@22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:����������@
 
_user_specified_nameinputs
�
n
O__inference_stream_0_input_drop_layer_call_and_return_conditional_losses_483379

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
:����������2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:����������*
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
:����������2
dropout/GreaterEqual�
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:����������2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:����������2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�D
�
E__inference_basemodel_layer_call_and_return_conditional_losses_483569
inputs_0,
stream_0_conv_1_483523:@$
stream_0_conv_1_483525:@(
batch_normalization_483528:@(
batch_normalization_483530:@(
batch_normalization_483532:@(
batch_normalization_483534:@ 
dense_1_483541:@T
dense_1_483543:T*
batch_normalization_1_483546:T*
batch_normalization_1_483548:T*
batch_normalization_1_483550:T*
batch_normalization_1_483552:T
identity��+batch_normalization/StatefulPartitionedCall�-batch_normalization_1/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�-dense_1/kernel/Regularizer/Abs/ReadVariableOp�'stream_0_conv_1/StatefulPartitionedCall�5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp�
#stream_0_input_drop/PartitionedCallPartitionedCallinputs_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *X
fSRQ
O__inference_stream_0_input_drop_layer_call_and_return_conditional_losses_4830522%
#stream_0_input_drop/PartitionedCall�
'stream_0_conv_1/StatefulPartitionedCallStatefulPartitionedCall,stream_0_input_drop/PartitionedCall:output:0stream_0_conv_1_483523stream_0_conv_1_483525*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_stream_0_conv_1_layer_call_and_return_conditional_losses_4830752)
'stream_0_conv_1/StatefulPartitionedCall�
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall0stream_0_conv_1/StatefulPartitionedCall:output:0batch_normalization_483528batch_normalization_483530batch_normalization_483532batch_normalization_483534*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_4831002-
+batch_normalization/StatefulPartitionedCall�
activation/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_activation_layer_call_and_return_conditional_losses_4831152
activation/PartitionedCall�
stream_0_drop_1/PartitionedCallPartitionedCall#activation/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_stream_0_drop_1_layer_call_and_return_conditional_losses_4831222!
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
 *-
config_proto

CPU

GPU 2J 8� *]
fXRV
T__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_4831292*
(global_average_pooling1d/PartitionedCall�
dense_1_dropout/PartitionedCallPartitionedCall1global_average_pooling1d/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_dense_1_dropout_layer_call_and_return_conditional_losses_4831362!
dense_1_dropout/PartitionedCall�
dense_1/StatefulPartitionedCallStatefulPartitionedCall(dense_1_dropout/PartitionedCall:output:0dense_1_483541dense_1_483543*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������T*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_4831542!
dense_1/StatefulPartitionedCall�
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0batch_normalization_1_483546batch_normalization_1_483548batch_normalization_1_483550batch_normalization_1_483552*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������T*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_4829022/
-batch_normalization_1/StatefulPartitionedCall�
"dense_activation_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������T* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *W
fRRP
N__inference_dense_activation_1_layer_call_and_return_conditional_losses_4831742$
"dense_activation_1/PartitionedCall�
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpstream_0_conv_1_483523*"
_output_shapes
:@*
dtype027
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp�
&stream_0_conv_1/kernel/Regularizer/AbsAbs=stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
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
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_1_483541*
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

Identity�
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall.^dense_1/kernel/Regularizer/Abs/ReadVariableOp(^stream_0_conv_1/StatefulPartitionedCall6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:����������: : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2^
-dense_1/kernel/Regularizer/Abs/ReadVariableOp-dense_1/kernel/Regularizer/Abs/ReadVariableOp2R
'stream_0_conv_1/StatefulPartitionedCall'stream_0_conv_1/StatefulPartitionedCall2n
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:V R
,
_output_shapes
:����������
"
_user_specified_name
inputs_0
�!
�
C__inference_model_1_layer_call_and_return_conditional_losses_484014

inputs&
basemodel_483976:@
basemodel_483978:@
basemodel_483980:@
basemodel_483982:@
basemodel_483984:@
basemodel_483986:@"
basemodel_483988:@T
basemodel_483990:T
basemodel_483992:T
basemodel_483994:T
basemodel_483996:T
basemodel_483998:T
identity��!basemodel/StatefulPartitionedCall�-dense_1/kernel/Regularizer/Abs/ReadVariableOp�5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp�
!basemodel/StatefulPartitionedCallStatefulPartitionedCallinputsbasemodel_483976basemodel_483978basemodel_483980basemodel_483982basemodel_483984basemodel_483986basemodel_483988basemodel_483990basemodel_483992basemodel_483994basemodel_483996basemodel_483998*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������T**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_basemodel_layer_call_and_return_conditional_losses_4839152#
!basemodel/StatefulPartitionedCall�
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpbasemodel_483976*"
_output_shapes
:@*
dtype027
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp�
&stream_0_conv_1/kernel/Regularizer/AbsAbs=stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
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
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpbasemodel_483988*
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
IdentityIdentity*basemodel/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������T2

Identity�
NoOpNoOp"^basemodel/StatefulPartitionedCall.^dense_1/kernel/Regularizer/Abs/ReadVariableOp6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:����������: : : : : : : : : : : : 2F
!basemodel/StatefulPartitionedCall!basemodel/StatefulPartitionedCall2^
-dense_1/kernel/Regularizer/Abs/ReadVariableOp-dense_1/kernel/Regularizer/Abs/ReadVariableOp2n
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�t
�
E__inference_basemodel_layer_call_and_return_conditional_losses_483697

inputsQ
;stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource:@=
/stream_0_conv_1_biasadd_readvariableop_resource:@C
5batch_normalization_batchnorm_readvariableop_resource:@G
9batch_normalization_batchnorm_mul_readvariableop_resource:@E
7batch_normalization_batchnorm_readvariableop_1_resource:@E
7batch_normalization_batchnorm_readvariableop_2_resource:@8
&dense_1_matmul_readvariableop_resource:@T5
'dense_1_biasadd_readvariableop_resource:TE
7batch_normalization_1_batchnorm_readvariableop_resource:TI
;batch_normalization_1_batchnorm_mul_readvariableop_resource:TG
9batch_normalization_1_batchnorm_readvariableop_1_resource:TG
9batch_normalization_1_batchnorm_readvariableop_2_resource:T
identity��,batch_normalization/batchnorm/ReadVariableOp�.batch_normalization/batchnorm/ReadVariableOp_1�.batch_normalization/batchnorm/ReadVariableOp_2�0batch_normalization/batchnorm/mul/ReadVariableOp�.batch_normalization_1/batchnorm/ReadVariableOp�0batch_normalization_1/batchnorm/ReadVariableOp_1�0batch_normalization_1/batchnorm/ReadVariableOp_2�2batch_normalization_1/batchnorm/mul/ReadVariableOp�dense_1/BiasAdd/ReadVariableOp�dense_1/MatMul/ReadVariableOp�-dense_1/kernel/Regularizer/Abs/ReadVariableOp�&stream_0_conv_1/BiasAdd/ReadVariableOp�2stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp�5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp�
stream_0_input_drop/IdentityIdentityinputs*
T0*,
_output_shapes
:����������2
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
:����������2#
!stream_0_conv_1/conv1d/ExpandDims�
2stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp;stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
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
:@2%
#stream_0_conv_1/conv1d/ExpandDims_1�
stream_0_conv_1/conv1dConv2D*stream_0_conv_1/conv1d/ExpandDims:output:0,stream_0_conv_1/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:����������@*
paddingSAME*
strides
2
stream_0_conv_1/conv1d�
stream_0_conv_1/conv1d/SqueezeSqueezestream_0_conv_1/conv1d:output:0*
T0*,
_output_shapes
:����������@*
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
:����������@2
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
T0*,
_output_shapes
:����������@2%
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
:����������@2%
#batch_normalization/batchnorm/add_1�
activation/ReluRelu'batch_normalization/batchnorm/add_1:z:0*
T0*,
_output_shapes
:����������@2
activation/Relu�
stream_0_drop_1/IdentityIdentityactivation/Relu:activations:0*
T0*,
_output_shapes
:����������@2
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
dense_1_dropout/IdentityIdentity&global_average_pooling1d/Mean:output:0*
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
.batch_normalization_1/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_1_batchnorm_readvariableop_resource*
_output_shapes
:T*
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
:T2%
#batch_normalization_1/batchnorm/add�
%batch_normalization_1/batchnorm/RsqrtRsqrt'batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes
:T2'
%batch_normalization_1/batchnorm/Rsqrt�
2batch_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes
:T*
dtype024
2batch_normalization_1/batchnorm/mul/ReadVariableOp�
#batch_normalization_1/batchnorm/mulMul)batch_normalization_1/batchnorm/Rsqrt:y:0:batch_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:T2%
#batch_normalization_1/batchnorm/mul�
%batch_normalization_1/batchnorm/mul_1Muldense_1/BiasAdd:output:0'batch_normalization_1/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������T2'
%batch_normalization_1/batchnorm/mul_1�
0batch_normalization_1/batchnorm/ReadVariableOp_1ReadVariableOp9batch_normalization_1_batchnorm_readvariableop_1_resource*
_output_shapes
:T*
dtype022
0batch_normalization_1/batchnorm/ReadVariableOp_1�
%batch_normalization_1/batchnorm/mul_2Mul8batch_normalization_1/batchnorm/ReadVariableOp_1:value:0'batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes
:T2'
%batch_normalization_1/batchnorm/mul_2�
0batch_normalization_1/batchnorm/ReadVariableOp_2ReadVariableOp9batch_normalization_1_batchnorm_readvariableop_2_resource*
_output_shapes
:T*
dtype022
0batch_normalization_1/batchnorm/ReadVariableOp_2�
#batch_normalization_1/batchnorm/subSub8batch_normalization_1/batchnorm/ReadVariableOp_2:value:0)batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes
:T2%
#batch_normalization_1/batchnorm/sub�
%batch_normalization_1/batchnorm/add_1AddV2)batch_normalization_1/batchnorm/mul_1:z:0'batch_normalization_1/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������T2'
%batch_normalization_1/batchnorm/add_1�
dense_activation_1/SigmoidSigmoid)batch_normalization_1/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������T2
dense_activation_1/Sigmoid�
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp;stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype027
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp�
&stream_0_conv_1/kernel/Regularizer/AbsAbs=stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
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

Identity�
NoOpNoOp-^batch_normalization/batchnorm/ReadVariableOp/^batch_normalization/batchnorm/ReadVariableOp_1/^batch_normalization/batchnorm/ReadVariableOp_21^batch_normalization/batchnorm/mul/ReadVariableOp/^batch_normalization_1/batchnorm/ReadVariableOp1^batch_normalization_1/batchnorm/ReadVariableOp_11^batch_normalization_1/batchnorm/ReadVariableOp_23^batch_normalization_1/batchnorm/mul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp.^dense_1/kernel/Regularizer/Abs/ReadVariableOp'^stream_0_conv_1/BiasAdd/ReadVariableOp3^stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:����������: : : : : : : : : : : : 2\
,batch_normalization/batchnorm/ReadVariableOp,batch_normalization/batchnorm/ReadVariableOp2`
.batch_normalization/batchnorm/ReadVariableOp_1.batch_normalization/batchnorm/ReadVariableOp_12`
.batch_normalization/batchnorm/ReadVariableOp_2.batch_normalization/batchnorm/ReadVariableOp_22d
0batch_normalization/batchnorm/mul/ReadVariableOp0batch_normalization/batchnorm/mul/ReadVariableOp2`
.batch_normalization_1/batchnorm/ReadVariableOp.batch_normalization_1/batchnorm/ReadVariableOp2d
0batch_normalization_1/batchnorm/ReadVariableOp_10batch_normalization_1/batchnorm/ReadVariableOp_12d
0batch_normalization_1/batchnorm/ReadVariableOp_20batch_normalization_1/batchnorm/ReadVariableOp_22h
2batch_normalization_1/batchnorm/mul/ReadVariableOp2batch_normalization_1/batchnorm/mul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2^
-dense_1/kernel/Regularizer/Abs/ReadVariableOp-dense_1/kernel/Regularizer/Abs/ReadVariableOp2P
&stream_0_conv_1/BiasAdd/ReadVariableOp&stream_0_conv_1/BiasAdd/ReadVariableOp2h
2stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp2stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp2n
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
��
�
C__inference_model_1_layer_call_and_return_conditional_losses_484325

inputs[
Ebasemodel_stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource:@G
9basemodel_stream_0_conv_1_biasadd_readvariableop_resource:@M
?basemodel_batch_normalization_batchnorm_readvariableop_resource:@Q
Cbasemodel_batch_normalization_batchnorm_mul_readvariableop_resource:@O
Abasemodel_batch_normalization_batchnorm_readvariableop_1_resource:@O
Abasemodel_batch_normalization_batchnorm_readvariableop_2_resource:@B
0basemodel_dense_1_matmul_readvariableop_resource:@T?
1basemodel_dense_1_biasadd_readvariableop_resource:TO
Abasemodel_batch_normalization_1_batchnorm_readvariableop_resource:TS
Ebasemodel_batch_normalization_1_batchnorm_mul_readvariableop_resource:TQ
Cbasemodel_batch_normalization_1_batchnorm_readvariableop_1_resource:TQ
Cbasemodel_batch_normalization_1_batchnorm_readvariableop_2_resource:T
identity��6basemodel/batch_normalization/batchnorm/ReadVariableOp�8basemodel/batch_normalization/batchnorm/ReadVariableOp_1�8basemodel/batch_normalization/batchnorm/ReadVariableOp_2�:basemodel/batch_normalization/batchnorm/mul/ReadVariableOp�8basemodel/batch_normalization_1/batchnorm/ReadVariableOp�:basemodel/batch_normalization_1/batchnorm/ReadVariableOp_1�:basemodel/batch_normalization_1/batchnorm/ReadVariableOp_2�<basemodel/batch_normalization_1/batchnorm/mul/ReadVariableOp�(basemodel/dense_1/BiasAdd/ReadVariableOp�'basemodel/dense_1/MatMul/ReadVariableOp�0basemodel/stream_0_conv_1/BiasAdd/ReadVariableOp�<basemodel/stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp�-dense_1/kernel/Regularizer/Abs/ReadVariableOp�5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp�
&basemodel/stream_0_input_drop/IdentityIdentityinputs*
T0*,
_output_shapes
:����������2(
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
:����������2-
+basemodel/stream_0_conv_1/conv1d/ExpandDims�
<basemodel/stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpEbasemodel_stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
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
:@2/
-basemodel/stream_0_conv_1/conv1d/ExpandDims_1�
 basemodel/stream_0_conv_1/conv1dConv2D4basemodel/stream_0_conv_1/conv1d/ExpandDims:output:06basemodel/stream_0_conv_1/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:����������@*
paddingSAME*
strides
2"
 basemodel/stream_0_conv_1/conv1d�
(basemodel/stream_0_conv_1/conv1d/SqueezeSqueeze)basemodel/stream_0_conv_1/conv1d:output:0*
T0*,
_output_shapes
:����������@*
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
:����������@2#
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
T0*,
_output_shapes
:����������@2/
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
:����������@2/
-basemodel/batch_normalization/batchnorm/add_1�
basemodel/activation/ReluRelu1basemodel/batch_normalization/batchnorm/add_1:z:0*
T0*,
_output_shapes
:����������@2
basemodel/activation/Relu�
"basemodel/stream_0_drop_1/IdentityIdentity'basemodel/activation/Relu:activations:0*
T0*,
_output_shapes
:����������@2$
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
"basemodel/dense_1_dropout/IdentityIdentity0basemodel/global_average_pooling1d/Mean:output:0*
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
8basemodel/batch_normalization_1/batchnorm/ReadVariableOpReadVariableOpAbasemodel_batch_normalization_1_batchnorm_readvariableop_resource*
_output_shapes
:T*
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
:T2/
-basemodel/batch_normalization_1/batchnorm/add�
/basemodel/batch_normalization_1/batchnorm/RsqrtRsqrt1basemodel/batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes
:T21
/basemodel/batch_normalization_1/batchnorm/Rsqrt�
<basemodel/batch_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOpEbasemodel_batch_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes
:T*
dtype02>
<basemodel/batch_normalization_1/batchnorm/mul/ReadVariableOp�
-basemodel/batch_normalization_1/batchnorm/mulMul3basemodel/batch_normalization_1/batchnorm/Rsqrt:y:0Dbasemodel/batch_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:T2/
-basemodel/batch_normalization_1/batchnorm/mul�
/basemodel/batch_normalization_1/batchnorm/mul_1Mul"basemodel/dense_1/BiasAdd:output:01basemodel/batch_normalization_1/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������T21
/basemodel/batch_normalization_1/batchnorm/mul_1�
:basemodel/batch_normalization_1/batchnorm/ReadVariableOp_1ReadVariableOpCbasemodel_batch_normalization_1_batchnorm_readvariableop_1_resource*
_output_shapes
:T*
dtype02<
:basemodel/batch_normalization_1/batchnorm/ReadVariableOp_1�
/basemodel/batch_normalization_1/batchnorm/mul_2MulBbasemodel/batch_normalization_1/batchnorm/ReadVariableOp_1:value:01basemodel/batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes
:T21
/basemodel/batch_normalization_1/batchnorm/mul_2�
:basemodel/batch_normalization_1/batchnorm/ReadVariableOp_2ReadVariableOpCbasemodel_batch_normalization_1_batchnorm_readvariableop_2_resource*
_output_shapes
:T*
dtype02<
:basemodel/batch_normalization_1/batchnorm/ReadVariableOp_2�
-basemodel/batch_normalization_1/batchnorm/subSubBbasemodel/batch_normalization_1/batchnorm/ReadVariableOp_2:value:03basemodel/batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes
:T2/
-basemodel/batch_normalization_1/batchnorm/sub�
/basemodel/batch_normalization_1/batchnorm/add_1AddV23basemodel/batch_normalization_1/batchnorm/mul_1:z:01basemodel/batch_normalization_1/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������T21
/basemodel/batch_normalization_1/batchnorm/add_1�
$basemodel/dense_activation_1/SigmoidSigmoid3basemodel/batch_normalization_1/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������T2&
$basemodel/dense_activation_1/Sigmoid�
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpEbasemodel_stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype027
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp�
&stream_0_conv_1/kernel/Regularizer/AbsAbs=stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
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
dense_1/kernel/Regularizer/mul�
IdentityIdentity(basemodel/dense_activation_1/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:���������T2

Identity�
NoOpNoOp7^basemodel/batch_normalization/batchnorm/ReadVariableOp9^basemodel/batch_normalization/batchnorm/ReadVariableOp_19^basemodel/batch_normalization/batchnorm/ReadVariableOp_2;^basemodel/batch_normalization/batchnorm/mul/ReadVariableOp9^basemodel/batch_normalization_1/batchnorm/ReadVariableOp;^basemodel/batch_normalization_1/batchnorm/ReadVariableOp_1;^basemodel/batch_normalization_1/batchnorm/ReadVariableOp_2=^basemodel/batch_normalization_1/batchnorm/mul/ReadVariableOp)^basemodel/dense_1/BiasAdd/ReadVariableOp(^basemodel/dense_1/MatMul/ReadVariableOp1^basemodel/stream_0_conv_1/BiasAdd/ReadVariableOp=^basemodel/stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp.^dense_1/kernel/Regularizer/Abs/ReadVariableOp6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:����������: : : : : : : : : : : : 2p
6basemodel/batch_normalization/batchnorm/ReadVariableOp6basemodel/batch_normalization/batchnorm/ReadVariableOp2t
8basemodel/batch_normalization/batchnorm/ReadVariableOp_18basemodel/batch_normalization/batchnorm/ReadVariableOp_12t
8basemodel/batch_normalization/batchnorm/ReadVariableOp_28basemodel/batch_normalization/batchnorm/ReadVariableOp_22x
:basemodel/batch_normalization/batchnorm/mul/ReadVariableOp:basemodel/batch_normalization/batchnorm/mul/ReadVariableOp2t
8basemodel/batch_normalization_1/batchnorm/ReadVariableOp8basemodel/batch_normalization_1/batchnorm/ReadVariableOp2x
:basemodel/batch_normalization_1/batchnorm/ReadVariableOp_1:basemodel/batch_normalization_1/batchnorm/ReadVariableOp_12x
:basemodel/batch_normalization_1/batchnorm/ReadVariableOp_2:basemodel/batch_normalization_1/batchnorm/ReadVariableOp_22|
<basemodel/batch_normalization_1/batchnorm/mul/ReadVariableOp<basemodel/batch_normalization_1/batchnorm/mul/ReadVariableOp2T
(basemodel/dense_1/BiasAdd/ReadVariableOp(basemodel/dense_1/BiasAdd/ReadVariableOp2R
'basemodel/dense_1/MatMul/ReadVariableOp'basemodel/dense_1/MatMul/ReadVariableOp2d
0basemodel/stream_0_conv_1/BiasAdd/ReadVariableOp0basemodel/stream_0_conv_1/BiasAdd/ReadVariableOp2|
<basemodel/stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp<basemodel/stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp2^
-dense_1/kernel/Regularizer/Abs/ReadVariableOp-dense_1/kernel/Regularizer/Abs/ReadVariableOp2n
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
p
T__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_483129

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
:����������@:T P
,
_output_shapes
:����������@
 
_user_specified_nameinputs
�J
�
E__inference_basemodel_layer_call_and_return_conditional_losses_483463

inputs,
stream_0_conv_1_483417:@$
stream_0_conv_1_483419:@(
batch_normalization_483422:@(
batch_normalization_483424:@(
batch_normalization_483426:@(
batch_normalization_483428:@ 
dense_1_483435:@T
dense_1_483437:T*
batch_normalization_1_483440:T*
batch_normalization_1_483442:T*
batch_normalization_1_483444:T*
batch_normalization_1_483446:T
identity��+batch_normalization/StatefulPartitionedCall�-batch_normalization_1/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�-dense_1/kernel/Regularizer/Abs/ReadVariableOp�'dense_1_dropout/StatefulPartitionedCall�'stream_0_conv_1/StatefulPartitionedCall�5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp�'stream_0_drop_1/StatefulPartitionedCall�+stream_0_input_drop/StatefulPartitionedCall�
+stream_0_input_drop/StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *X
fSRQ
O__inference_stream_0_input_drop_layer_call_and_return_conditional_losses_4833792-
+stream_0_input_drop/StatefulPartitionedCall�
'stream_0_conv_1/StatefulPartitionedCallStatefulPartitionedCall4stream_0_input_drop/StatefulPartitionedCall:output:0stream_0_conv_1_483417stream_0_conv_1_483419*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_stream_0_conv_1_layer_call_and_return_conditional_losses_4830752)
'stream_0_conv_1/StatefulPartitionedCall�
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall0stream_0_conv_1/StatefulPartitionedCall:output:0batch_normalization_483422batch_normalization_483424batch_normalization_483426batch_normalization_483428*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_4833382-
+batch_normalization/StatefulPartitionedCall�
activation/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_activation_layer_call_and_return_conditional_losses_4831152
activation/PartitionedCall�
'stream_0_drop_1/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0,^stream_0_input_drop/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_stream_0_drop_1_layer_call_and_return_conditional_losses_4832802)
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
 *-
config_proto

CPU

GPU 2J 8� *]
fXRV
T__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_4831292*
(global_average_pooling1d/PartitionedCall�
'dense_1_dropout/StatefulPartitionedCallStatefulPartitionedCall1global_average_pooling1d/PartitionedCall:output:0(^stream_0_drop_1/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_dense_1_dropout_layer_call_and_return_conditional_losses_4832522)
'dense_1_dropout/StatefulPartitionedCall�
dense_1/StatefulPartitionedCallStatefulPartitionedCall0dense_1_dropout/StatefulPartitionedCall:output:0dense_1_483435dense_1_483437*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������T*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_4831542!
dense_1/StatefulPartitionedCall�
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0batch_normalization_1_483440batch_normalization_1_483442batch_normalization_1_483444batch_normalization_1_483446*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������T*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_4829622/
-batch_normalization_1/StatefulPartitionedCall�
"dense_activation_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������T* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *W
fRRP
N__inference_dense_activation_1_layer_call_and_return_conditional_losses_4831742$
"dense_activation_1/PartitionedCall�
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpstream_0_conv_1_483417*"
_output_shapes
:@*
dtype027
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp�
&stream_0_conv_1/kernel/Regularizer/AbsAbs=stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
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
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_1_483435*
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
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall.^dense_1/kernel/Regularizer/Abs/ReadVariableOp(^dense_1_dropout/StatefulPartitionedCall(^stream_0_conv_1/StatefulPartitionedCall6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp(^stream_0_drop_1/StatefulPartitionedCall,^stream_0_input_drop/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:����������: : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2^
-dense_1/kernel/Regularizer/Abs/ReadVariableOp-dense_1/kernel/Regularizer/Abs/ReadVariableOp2R
'dense_1_dropout/StatefulPartitionedCall'dense_1_dropout/StatefulPartitionedCall2R
'stream_0_conv_1/StatefulPartitionedCall'stream_0_conv_1/StatefulPartitionedCall2n
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp2R
'stream_0_drop_1/StatefulPartitionedCall'stream_0_drop_1/StatefulPartitionedCall2Z
+stream_0_input_drop/StatefulPartitionedCall+stream_0_input_drop/StatefulPartitionedCall:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
U
9__inference_global_average_pooling1d_layer_call_fn_485230

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
 *-
config_proto

CPU

GPU 2J 8� *]
fXRV
T__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_4831292
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������@:T P
,
_output_shapes
:����������@
 
_user_specified_nameinputs
�
n
O__inference_stream_0_input_drop_layer_call_and_return_conditional_losses_484987

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
:����������2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:����������*
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
:����������2
dropout/GreaterEqual�
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:����������2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:����������2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
$__inference_signature_wrapper_484195
left_inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:@
	unknown_5:@T
	unknown_6:T
	unknown_7:T
	unknown_8:T
	unknown_9:T

unknown_10:T
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallleft_inputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������T*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� **
f%R#
!__inference__wrapped_model_4826922
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
_construction_contextkEagerRuntime*C
_input_shapes2
0:����������: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
,
_output_shapes
:����������
%
_user_specified_nameleft_inputs
�
G
+__inference_activation_layer_call_fn_485188

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
:����������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_activation_layer_call_and_return_conditional_losses_4831152
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:����������@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������@:T P
,
_output_shapes
:����������@
 
_user_specified_nameinputs
�
P
4__inference_stream_0_input_drop_layer_call_fn_484965

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
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *X
fSRQ
O__inference_stream_0_input_drop_layer_call_and_return_conditional_losses_4830522
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
p
T__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_482864

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
O__inference_batch_normalization_layer_call_and_return_conditional_losses_482716

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
�
E__inference_basemodel_layer_call_and_return_conditional_losses_484767

inputsQ
;stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource:@=
/stream_0_conv_1_biasadd_readvariableop_resource:@I
;batch_normalization_assignmovingavg_readvariableop_resource:@K
=batch_normalization_assignmovingavg_1_readvariableop_resource:@G
9batch_normalization_batchnorm_mul_readvariableop_resource:@C
5batch_normalization_batchnorm_readvariableop_resource:@8
&dense_1_matmul_readvariableop_resource:@T5
'dense_1_biasadd_readvariableop_resource:TK
=batch_normalization_1_assignmovingavg_readvariableop_resource:TM
?batch_normalization_1_assignmovingavg_1_readvariableop_resource:TI
;batch_normalization_1_batchnorm_mul_readvariableop_resource:TE
7batch_normalization_1_batchnorm_readvariableop_resource:T
identity��#batch_normalization/AssignMovingAvg�2batch_normalization/AssignMovingAvg/ReadVariableOp�%batch_normalization/AssignMovingAvg_1�4batch_normalization/AssignMovingAvg_1/ReadVariableOp�,batch_normalization/batchnorm/ReadVariableOp�0batch_normalization/batchnorm/mul/ReadVariableOp�%batch_normalization_1/AssignMovingAvg�4batch_normalization_1/AssignMovingAvg/ReadVariableOp�'batch_normalization_1/AssignMovingAvg_1�6batch_normalization_1/AssignMovingAvg_1/ReadVariableOp�.batch_normalization_1/batchnorm/ReadVariableOp�2batch_normalization_1/batchnorm/mul/ReadVariableOp�dense_1/BiasAdd/ReadVariableOp�dense_1/MatMul/ReadVariableOp�-dense_1/kernel/Regularizer/Abs/ReadVariableOp�&stream_0_conv_1/BiasAdd/ReadVariableOp�2stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp�5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp�
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
:����������2!
stream_0_input_drop/dropout/Mul|
!stream_0_input_drop/dropout/ShapeShapeinputs*
T0*
_output_shapes
:2#
!stream_0_input_drop/dropout/Shape�
8stream_0_input_drop/dropout/random_uniform/RandomUniformRandomUniform*stream_0_input_drop/dropout/Shape:output:0*
T0*,
_output_shapes
:����������*
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
:����������2*
(stream_0_input_drop/dropout/GreaterEqual�
 stream_0_input_drop/dropout/CastCast,stream_0_input_drop/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:����������2"
 stream_0_input_drop/dropout/Cast�
!stream_0_input_drop/dropout/Mul_1Mul#stream_0_input_drop/dropout/Mul:z:0$stream_0_input_drop/dropout/Cast:y:0*
T0*,
_output_shapes
:����������2#
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
:����������2#
!stream_0_conv_1/conv1d/ExpandDims�
2stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp;stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
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
:@2%
#stream_0_conv_1/conv1d/ExpandDims_1�
stream_0_conv_1/conv1dConv2D*stream_0_conv_1/conv1d/ExpandDims:output:0,stream_0_conv_1/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:����������@*
paddingSAME*
strides
2
stream_0_conv_1/conv1d�
stream_0_conv_1/conv1d/SqueezeSqueezestream_0_conv_1/conv1d:output:0*
T0*,
_output_shapes
:����������@*
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
:����������@2
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
T0*,
_output_shapes
:����������@2/
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
:����������@2%
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
:����������@2%
#batch_normalization/batchnorm/add_1�
activation/ReluRelu'batch_normalization/batchnorm/add_1:z:0*
T0*,
_output_shapes
:����������@2
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
:����������@2
stream_0_drop_1/dropout/Mul�
stream_0_drop_1/dropout/ShapeShapeactivation/Relu:activations:0*
T0*
_output_shapes
:2
stream_0_drop_1/dropout/Shape�
4stream_0_drop_1/dropout/random_uniform/RandomUniformRandomUniform&stream_0_drop_1/dropout/Shape:output:0*
T0*,
_output_shapes
:����������@*
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
:����������@2&
$stream_0_drop_1/dropout/GreaterEqual�
stream_0_drop_1/dropout/CastCast(stream_0_drop_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:����������@2
stream_0_drop_1/dropout/Cast�
stream_0_drop_1/dropout/Mul_1Mulstream_0_drop_1/dropout/Mul:z:0 stream_0_drop_1/dropout/Cast:y:0*
T0*,
_output_shapes
:����������@2
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
dense_1_dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
dense_1_dropout/dropout/Const�
dense_1_dropout/dropout/MulMul&global_average_pooling1d/Mean:output:0&dense_1_dropout/dropout/Const:output:0*
T0*'
_output_shapes
:���������@2
dense_1_dropout/dropout/Mul�
dense_1_dropout/dropout/ShapeShape&global_average_pooling1d/Mean:output:0*
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
4batch_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 26
4batch_normalization_1/moments/mean/reduction_indices�
"batch_normalization_1/moments/meanMeandense_1/BiasAdd:output:0=batch_normalization_1/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:T*
	keep_dims(2$
"batch_normalization_1/moments/mean�
*batch_normalization_1/moments/StopGradientStopGradient+batch_normalization_1/moments/mean:output:0*
T0*
_output_shapes

:T2,
*batch_normalization_1/moments/StopGradient�
/batch_normalization_1/moments/SquaredDifferenceSquaredDifferencedense_1/BiasAdd:output:03batch_normalization_1/moments/StopGradient:output:0*
T0*'
_output_shapes
:���������T21
/batch_normalization_1/moments/SquaredDifference�
8batch_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2:
8batch_normalization_1/moments/variance/reduction_indices�
&batch_normalization_1/moments/varianceMean3batch_normalization_1/moments/SquaredDifference:z:0Abatch_normalization_1/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:T*
	keep_dims(2(
&batch_normalization_1/moments/variance�
%batch_normalization_1/moments/SqueezeSqueeze+batch_normalization_1/moments/mean:output:0*
T0*
_output_shapes
:T*
squeeze_dims
 2'
%batch_normalization_1/moments/Squeeze�
'batch_normalization_1/moments/Squeeze_1Squeeze/batch_normalization_1/moments/variance:output:0*
T0*
_output_shapes
:T*
squeeze_dims
 2)
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
:T*
dtype026
4batch_normalization_1/AssignMovingAvg/ReadVariableOp�
)batch_normalization_1/AssignMovingAvg/subSub<batch_normalization_1/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_1/moments/Squeeze:output:0*
T0*
_output_shapes
:T2+
)batch_normalization_1/AssignMovingAvg/sub�
)batch_normalization_1/AssignMovingAvg/mulMul-batch_normalization_1/AssignMovingAvg/sub:z:04batch_normalization_1/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:T2+
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
:T*
dtype028
6batch_normalization_1/AssignMovingAvg_1/ReadVariableOp�
+batch_normalization_1/AssignMovingAvg_1/subSub>batch_normalization_1/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_1/moments/Squeeze_1:output:0*
T0*
_output_shapes
:T2-
+batch_normalization_1/AssignMovingAvg_1/sub�
+batch_normalization_1/AssignMovingAvg_1/mulMul/batch_normalization_1/AssignMovingAvg_1/sub:z:06batch_normalization_1/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:T2-
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
:T2%
#batch_normalization_1/batchnorm/add�
%batch_normalization_1/batchnorm/RsqrtRsqrt'batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes
:T2'
%batch_normalization_1/batchnorm/Rsqrt�
2batch_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes
:T*
dtype024
2batch_normalization_1/batchnorm/mul/ReadVariableOp�
#batch_normalization_1/batchnorm/mulMul)batch_normalization_1/batchnorm/Rsqrt:y:0:batch_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:T2%
#batch_normalization_1/batchnorm/mul�
%batch_normalization_1/batchnorm/mul_1Muldense_1/BiasAdd:output:0'batch_normalization_1/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������T2'
%batch_normalization_1/batchnorm/mul_1�
%batch_normalization_1/batchnorm/mul_2Mul.batch_normalization_1/moments/Squeeze:output:0'batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes
:T2'
%batch_normalization_1/batchnorm/mul_2�
.batch_normalization_1/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_1_batchnorm_readvariableop_resource*
_output_shapes
:T*
dtype020
.batch_normalization_1/batchnorm/ReadVariableOp�
#batch_normalization_1/batchnorm/subSub6batch_normalization_1/batchnorm/ReadVariableOp:value:0)batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes
:T2%
#batch_normalization_1/batchnorm/sub�
%batch_normalization_1/batchnorm/add_1AddV2)batch_normalization_1/batchnorm/mul_1:z:0'batch_normalization_1/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������T2'
%batch_normalization_1/batchnorm/add_1�
dense_activation_1/SigmoidSigmoid)batch_normalization_1/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������T2
dense_activation_1/Sigmoid�
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp;stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype027
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp�
&stream_0_conv_1/kernel/Regularizer/AbsAbs=stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
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

Identity�
NoOpNoOp$^batch_normalization/AssignMovingAvg3^batch_normalization/AssignMovingAvg/ReadVariableOp&^batch_normalization/AssignMovingAvg_15^batch_normalization/AssignMovingAvg_1/ReadVariableOp-^batch_normalization/batchnorm/ReadVariableOp1^batch_normalization/batchnorm/mul/ReadVariableOp&^batch_normalization_1/AssignMovingAvg5^batch_normalization_1/AssignMovingAvg/ReadVariableOp(^batch_normalization_1/AssignMovingAvg_17^batch_normalization_1/AssignMovingAvg_1/ReadVariableOp/^batch_normalization_1/batchnorm/ReadVariableOp3^batch_normalization_1/batchnorm/mul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp.^dense_1/kernel/Regularizer/Abs/ReadVariableOp'^stream_0_conv_1/BiasAdd/ReadVariableOp3^stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:����������: : : : : : : : : : : : 2J
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
2batch_normalization_1/batchnorm/mul/ReadVariableOp2batch_normalization_1/batchnorm/mul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2^
-dense_1/kernel/Regularizer/Abs/ReadVariableOp-dense_1/kernel/Regularizer/Abs/ReadVariableOp2P
&stream_0_conv_1/BiasAdd/ReadVariableOp&stream_0_conv_1/BiasAdd/ReadVariableOp2h
2stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp2stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp2n
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_485346

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
�
K__inference_stream_0_conv_1_layer_call_and_return_conditional_losses_483075

inputsA
+conv1d_expanddims_1_readvariableop_resource:@-
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
:����������2
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
:����������@*
paddingSAME*
strides
2
conv1d�
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:����������@*
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
:����������@2	
BiasAdd�
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype027
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp�
&stream_0_conv_1/kernel/Regularizer/AbsAbs=stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
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
:����������@2

Identity�
NoOpNoOp^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp2n
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
__inference_loss_fn_0_485401T
>stream_0_conv_1_kernel_regularizer_abs_readvariableop_resource:@
identity��5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp�
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp>stream_0_conv_1_kernel_regularizer_abs_readvariableop_resource*"
_output_shapes
:@*
dtype027
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp�
&stream_0_conv_1/kernel/Regularizer/AbsAbs=stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
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
�
j
K__inference_stream_0_drop_1_layer_call_and_return_conditional_losses_483280

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
:����������@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:����������@*
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
:����������@2
dropout/GreaterEqual�
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:����������@2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:����������@2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:����������@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������@:T P
,
_output_shapes
:����������@
 
_user_specified_nameinputs
�
m
O__inference_stream_0_input_drop_layer_call_and_return_conditional_losses_484975

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:����������2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:����������2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
U
9__inference_global_average_pooling1d_layer_call_fn_485225

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
 *-
config_proto

CPU

GPU 2J 8� *]
fXRV
T__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_4828642
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
�
*__inference_basemodel_layer_call_fn_484574
inputs_0
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:@
	unknown_5:@T
	unknown_6:T
	unknown_7:T
	unknown_8:T
	unknown_9:T

unknown_10:T
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������T**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_basemodel_layer_call_and_return_conditional_losses_4839152
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
_construction_contextkEagerRuntime*C
_input_shapes2
0:����������: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
,
_output_shapes
:����������
"
_user_specified_name
inputs/0
�+
�
O__inference_batch_normalization_layer_call_and_return_conditional_losses_482776

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
�
(__inference_model_1_layer_call_fn_483763
left_inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:@
	unknown_5:@T
	unknown_6:T
	unknown_7:T
	unknown_8:T
	unknown_9:T

unknown_10:T
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallleft_inputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������T*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_model_1_layer_call_and_return_conditional_losses_4837362
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
_construction_contextkEagerRuntime*C
_input_shapes2
0:����������: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
,
_output_shapes
:����������
%
_user_specified_nameleft_inputs
�
i
K__inference_dense_1_dropout_layer_call_and_return_conditional_losses_485257

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
�
�
O__inference_batch_normalization_layer_call_and_return_conditional_losses_485095

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
�+
�
O__inference_batch_normalization_layer_call_and_return_conditional_losses_485129

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
�t
�
E__inference_basemodel_layer_call_and_return_conditional_losses_484646

inputsQ
;stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource:@=
/stream_0_conv_1_biasadd_readvariableop_resource:@C
5batch_normalization_batchnorm_readvariableop_resource:@G
9batch_normalization_batchnorm_mul_readvariableop_resource:@E
7batch_normalization_batchnorm_readvariableop_1_resource:@E
7batch_normalization_batchnorm_readvariableop_2_resource:@8
&dense_1_matmul_readvariableop_resource:@T5
'dense_1_biasadd_readvariableop_resource:TE
7batch_normalization_1_batchnorm_readvariableop_resource:TI
;batch_normalization_1_batchnorm_mul_readvariableop_resource:TG
9batch_normalization_1_batchnorm_readvariableop_1_resource:TG
9batch_normalization_1_batchnorm_readvariableop_2_resource:T
identity��,batch_normalization/batchnorm/ReadVariableOp�.batch_normalization/batchnorm/ReadVariableOp_1�.batch_normalization/batchnorm/ReadVariableOp_2�0batch_normalization/batchnorm/mul/ReadVariableOp�.batch_normalization_1/batchnorm/ReadVariableOp�0batch_normalization_1/batchnorm/ReadVariableOp_1�0batch_normalization_1/batchnorm/ReadVariableOp_2�2batch_normalization_1/batchnorm/mul/ReadVariableOp�dense_1/BiasAdd/ReadVariableOp�dense_1/MatMul/ReadVariableOp�-dense_1/kernel/Regularizer/Abs/ReadVariableOp�&stream_0_conv_1/BiasAdd/ReadVariableOp�2stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp�5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp�
stream_0_input_drop/IdentityIdentityinputs*
T0*,
_output_shapes
:����������2
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
:����������2#
!stream_0_conv_1/conv1d/ExpandDims�
2stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp;stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
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
:@2%
#stream_0_conv_1/conv1d/ExpandDims_1�
stream_0_conv_1/conv1dConv2D*stream_0_conv_1/conv1d/ExpandDims:output:0,stream_0_conv_1/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:����������@*
paddingSAME*
strides
2
stream_0_conv_1/conv1d�
stream_0_conv_1/conv1d/SqueezeSqueezestream_0_conv_1/conv1d:output:0*
T0*,
_output_shapes
:����������@*
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
:����������@2
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
T0*,
_output_shapes
:����������@2%
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
:����������@2%
#batch_normalization/batchnorm/add_1�
activation/ReluRelu'batch_normalization/batchnorm/add_1:z:0*
T0*,
_output_shapes
:����������@2
activation/Relu�
stream_0_drop_1/IdentityIdentityactivation/Relu:activations:0*
T0*,
_output_shapes
:����������@2
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
dense_1_dropout/IdentityIdentity&global_average_pooling1d/Mean:output:0*
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
.batch_normalization_1/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_1_batchnorm_readvariableop_resource*
_output_shapes
:T*
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
:T2%
#batch_normalization_1/batchnorm/add�
%batch_normalization_1/batchnorm/RsqrtRsqrt'batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes
:T2'
%batch_normalization_1/batchnorm/Rsqrt�
2batch_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes
:T*
dtype024
2batch_normalization_1/batchnorm/mul/ReadVariableOp�
#batch_normalization_1/batchnorm/mulMul)batch_normalization_1/batchnorm/Rsqrt:y:0:batch_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:T2%
#batch_normalization_1/batchnorm/mul�
%batch_normalization_1/batchnorm/mul_1Muldense_1/BiasAdd:output:0'batch_normalization_1/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������T2'
%batch_normalization_1/batchnorm/mul_1�
0batch_normalization_1/batchnorm/ReadVariableOp_1ReadVariableOp9batch_normalization_1_batchnorm_readvariableop_1_resource*
_output_shapes
:T*
dtype022
0batch_normalization_1/batchnorm/ReadVariableOp_1�
%batch_normalization_1/batchnorm/mul_2Mul8batch_normalization_1/batchnorm/ReadVariableOp_1:value:0'batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes
:T2'
%batch_normalization_1/batchnorm/mul_2�
0batch_normalization_1/batchnorm/ReadVariableOp_2ReadVariableOp9batch_normalization_1_batchnorm_readvariableop_2_resource*
_output_shapes
:T*
dtype022
0batch_normalization_1/batchnorm/ReadVariableOp_2�
#batch_normalization_1/batchnorm/subSub8batch_normalization_1/batchnorm/ReadVariableOp_2:value:0)batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes
:T2%
#batch_normalization_1/batchnorm/sub�
%batch_normalization_1/batchnorm/add_1AddV2)batch_normalization_1/batchnorm/mul_1:z:0'batch_normalization_1/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������T2'
%batch_normalization_1/batchnorm/add_1�
dense_activation_1/SigmoidSigmoid)batch_normalization_1/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������T2
dense_activation_1/Sigmoid�
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp;stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype027
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp�
&stream_0_conv_1/kernel/Regularizer/AbsAbs=stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
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

Identity�
NoOpNoOp-^batch_normalization/batchnorm/ReadVariableOp/^batch_normalization/batchnorm/ReadVariableOp_1/^batch_normalization/batchnorm/ReadVariableOp_21^batch_normalization/batchnorm/mul/ReadVariableOp/^batch_normalization_1/batchnorm/ReadVariableOp1^batch_normalization_1/batchnorm/ReadVariableOp_11^batch_normalization_1/batchnorm/ReadVariableOp_23^batch_normalization_1/batchnorm/mul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp.^dense_1/kernel/Regularizer/Abs/ReadVariableOp'^stream_0_conv_1/BiasAdd/ReadVariableOp3^stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:����������: : : : : : : : : : : : 2\
,batch_normalization/batchnorm/ReadVariableOp,batch_normalization/batchnorm/ReadVariableOp2`
.batch_normalization/batchnorm/ReadVariableOp_1.batch_normalization/batchnorm/ReadVariableOp_12`
.batch_normalization/batchnorm/ReadVariableOp_2.batch_normalization/batchnorm/ReadVariableOp_22d
0batch_normalization/batchnorm/mul/ReadVariableOp0batch_normalization/batchnorm/mul/ReadVariableOp2`
.batch_normalization_1/batchnorm/ReadVariableOp.batch_normalization_1/batchnorm/ReadVariableOp2d
0batch_normalization_1/batchnorm/ReadVariableOp_10batch_normalization_1/batchnorm/ReadVariableOp_12d
0batch_normalization_1/batchnorm/ReadVariableOp_20batch_normalization_1/batchnorm/ReadVariableOp_22h
2batch_normalization_1/batchnorm/mul/ReadVariableOp2batch_normalization_1/batchnorm/mul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2^
-dense_1/kernel/Regularizer/Abs/ReadVariableOp-dense_1/kernel/Regularizer/Abs/ReadVariableOp2P
&stream_0_conv_1/BiasAdd/ReadVariableOp&stream_0_conv_1/BiasAdd/ReadVariableOp2h
2stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp2stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp2n
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
b
F__inference_activation_layer_call_and_return_conditional_losses_483115

inputs
identityS
ReluReluinputs*
T0*,
_output_shapes
:����������@2
Reluk
IdentityIdentityRelu:activations:0*
T0*,
_output_shapes
:����������@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������@:T P
,
_output_shapes
:����������@
 
_user_specified_nameinputs
��
�
E__inference_basemodel_layer_call_and_return_conditional_losses_484960
inputs_0Q
;stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource:@=
/stream_0_conv_1_biasadd_readvariableop_resource:@I
;batch_normalization_assignmovingavg_readvariableop_resource:@K
=batch_normalization_assignmovingavg_1_readvariableop_resource:@G
9batch_normalization_batchnorm_mul_readvariableop_resource:@C
5batch_normalization_batchnorm_readvariableop_resource:@8
&dense_1_matmul_readvariableop_resource:@T5
'dense_1_biasadd_readvariableop_resource:TK
=batch_normalization_1_assignmovingavg_readvariableop_resource:TM
?batch_normalization_1_assignmovingavg_1_readvariableop_resource:TI
;batch_normalization_1_batchnorm_mul_readvariableop_resource:TE
7batch_normalization_1_batchnorm_readvariableop_resource:T
identity��#batch_normalization/AssignMovingAvg�2batch_normalization/AssignMovingAvg/ReadVariableOp�%batch_normalization/AssignMovingAvg_1�4batch_normalization/AssignMovingAvg_1/ReadVariableOp�,batch_normalization/batchnorm/ReadVariableOp�0batch_normalization/batchnorm/mul/ReadVariableOp�%batch_normalization_1/AssignMovingAvg�4batch_normalization_1/AssignMovingAvg/ReadVariableOp�'batch_normalization_1/AssignMovingAvg_1�6batch_normalization_1/AssignMovingAvg_1/ReadVariableOp�.batch_normalization_1/batchnorm/ReadVariableOp�2batch_normalization_1/batchnorm/mul/ReadVariableOp�dense_1/BiasAdd/ReadVariableOp�dense_1/MatMul/ReadVariableOp�-dense_1/kernel/Regularizer/Abs/ReadVariableOp�&stream_0_conv_1/BiasAdd/ReadVariableOp�2stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp�5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp�
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
:����������2!
stream_0_input_drop/dropout/Mul~
!stream_0_input_drop/dropout/ShapeShapeinputs_0*
T0*
_output_shapes
:2#
!stream_0_input_drop/dropout/Shape�
8stream_0_input_drop/dropout/random_uniform/RandomUniformRandomUniform*stream_0_input_drop/dropout/Shape:output:0*
T0*,
_output_shapes
:����������*
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
:����������2*
(stream_0_input_drop/dropout/GreaterEqual�
 stream_0_input_drop/dropout/CastCast,stream_0_input_drop/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:����������2"
 stream_0_input_drop/dropout/Cast�
!stream_0_input_drop/dropout/Mul_1Mul#stream_0_input_drop/dropout/Mul:z:0$stream_0_input_drop/dropout/Cast:y:0*
T0*,
_output_shapes
:����������2#
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
:����������2#
!stream_0_conv_1/conv1d/ExpandDims�
2stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp;stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
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
:@2%
#stream_0_conv_1/conv1d/ExpandDims_1�
stream_0_conv_1/conv1dConv2D*stream_0_conv_1/conv1d/ExpandDims:output:0,stream_0_conv_1/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:����������@*
paddingSAME*
strides
2
stream_0_conv_1/conv1d�
stream_0_conv_1/conv1d/SqueezeSqueezestream_0_conv_1/conv1d:output:0*
T0*,
_output_shapes
:����������@*
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
:����������@2
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
T0*,
_output_shapes
:����������@2/
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
:����������@2%
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
:����������@2%
#batch_normalization/batchnorm/add_1�
activation/ReluRelu'batch_normalization/batchnorm/add_1:z:0*
T0*,
_output_shapes
:����������@2
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
:����������@2
stream_0_drop_1/dropout/Mul�
stream_0_drop_1/dropout/ShapeShapeactivation/Relu:activations:0*
T0*
_output_shapes
:2
stream_0_drop_1/dropout/Shape�
4stream_0_drop_1/dropout/random_uniform/RandomUniformRandomUniform&stream_0_drop_1/dropout/Shape:output:0*
T0*,
_output_shapes
:����������@*
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
:����������@2&
$stream_0_drop_1/dropout/GreaterEqual�
stream_0_drop_1/dropout/CastCast(stream_0_drop_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:����������@2
stream_0_drop_1/dropout/Cast�
stream_0_drop_1/dropout/Mul_1Mulstream_0_drop_1/dropout/Mul:z:0 stream_0_drop_1/dropout/Cast:y:0*
T0*,
_output_shapes
:����������@2
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
dense_1_dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
dense_1_dropout/dropout/Const�
dense_1_dropout/dropout/MulMul&global_average_pooling1d/Mean:output:0&dense_1_dropout/dropout/Const:output:0*
T0*'
_output_shapes
:���������@2
dense_1_dropout/dropout/Mul�
dense_1_dropout/dropout/ShapeShape&global_average_pooling1d/Mean:output:0*
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
4batch_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 26
4batch_normalization_1/moments/mean/reduction_indices�
"batch_normalization_1/moments/meanMeandense_1/BiasAdd:output:0=batch_normalization_1/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:T*
	keep_dims(2$
"batch_normalization_1/moments/mean�
*batch_normalization_1/moments/StopGradientStopGradient+batch_normalization_1/moments/mean:output:0*
T0*
_output_shapes

:T2,
*batch_normalization_1/moments/StopGradient�
/batch_normalization_1/moments/SquaredDifferenceSquaredDifferencedense_1/BiasAdd:output:03batch_normalization_1/moments/StopGradient:output:0*
T0*'
_output_shapes
:���������T21
/batch_normalization_1/moments/SquaredDifference�
8batch_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2:
8batch_normalization_1/moments/variance/reduction_indices�
&batch_normalization_1/moments/varianceMean3batch_normalization_1/moments/SquaredDifference:z:0Abatch_normalization_1/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:T*
	keep_dims(2(
&batch_normalization_1/moments/variance�
%batch_normalization_1/moments/SqueezeSqueeze+batch_normalization_1/moments/mean:output:0*
T0*
_output_shapes
:T*
squeeze_dims
 2'
%batch_normalization_1/moments/Squeeze�
'batch_normalization_1/moments/Squeeze_1Squeeze/batch_normalization_1/moments/variance:output:0*
T0*
_output_shapes
:T*
squeeze_dims
 2)
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
:T*
dtype026
4batch_normalization_1/AssignMovingAvg/ReadVariableOp�
)batch_normalization_1/AssignMovingAvg/subSub<batch_normalization_1/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_1/moments/Squeeze:output:0*
T0*
_output_shapes
:T2+
)batch_normalization_1/AssignMovingAvg/sub�
)batch_normalization_1/AssignMovingAvg/mulMul-batch_normalization_1/AssignMovingAvg/sub:z:04batch_normalization_1/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:T2+
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
:T*
dtype028
6batch_normalization_1/AssignMovingAvg_1/ReadVariableOp�
+batch_normalization_1/AssignMovingAvg_1/subSub>batch_normalization_1/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_1/moments/Squeeze_1:output:0*
T0*
_output_shapes
:T2-
+batch_normalization_1/AssignMovingAvg_1/sub�
+batch_normalization_1/AssignMovingAvg_1/mulMul/batch_normalization_1/AssignMovingAvg_1/sub:z:06batch_normalization_1/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:T2-
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
:T2%
#batch_normalization_1/batchnorm/add�
%batch_normalization_1/batchnorm/RsqrtRsqrt'batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes
:T2'
%batch_normalization_1/batchnorm/Rsqrt�
2batch_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes
:T*
dtype024
2batch_normalization_1/batchnorm/mul/ReadVariableOp�
#batch_normalization_1/batchnorm/mulMul)batch_normalization_1/batchnorm/Rsqrt:y:0:batch_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:T2%
#batch_normalization_1/batchnorm/mul�
%batch_normalization_1/batchnorm/mul_1Muldense_1/BiasAdd:output:0'batch_normalization_1/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������T2'
%batch_normalization_1/batchnorm/mul_1�
%batch_normalization_1/batchnorm/mul_2Mul.batch_normalization_1/moments/Squeeze:output:0'batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes
:T2'
%batch_normalization_1/batchnorm/mul_2�
.batch_normalization_1/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_1_batchnorm_readvariableop_resource*
_output_shapes
:T*
dtype020
.batch_normalization_1/batchnorm/ReadVariableOp�
#batch_normalization_1/batchnorm/subSub6batch_normalization_1/batchnorm/ReadVariableOp:value:0)batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes
:T2%
#batch_normalization_1/batchnorm/sub�
%batch_normalization_1/batchnorm/add_1AddV2)batch_normalization_1/batchnorm/mul_1:z:0'batch_normalization_1/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������T2'
%batch_normalization_1/batchnorm/add_1�
dense_activation_1/SigmoidSigmoid)batch_normalization_1/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������T2
dense_activation_1/Sigmoid�
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp;stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype027
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp�
&stream_0_conv_1/kernel/Regularizer/AbsAbs=stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
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

Identity�
NoOpNoOp$^batch_normalization/AssignMovingAvg3^batch_normalization/AssignMovingAvg/ReadVariableOp&^batch_normalization/AssignMovingAvg_15^batch_normalization/AssignMovingAvg_1/ReadVariableOp-^batch_normalization/batchnorm/ReadVariableOp1^batch_normalization/batchnorm/mul/ReadVariableOp&^batch_normalization_1/AssignMovingAvg5^batch_normalization_1/AssignMovingAvg/ReadVariableOp(^batch_normalization_1/AssignMovingAvg_17^batch_normalization_1/AssignMovingAvg_1/ReadVariableOp/^batch_normalization_1/batchnorm/ReadVariableOp3^batch_normalization_1/batchnorm/mul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp.^dense_1/kernel/Regularizer/Abs/ReadVariableOp'^stream_0_conv_1/BiasAdd/ReadVariableOp3^stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:����������: : : : : : : : : : : : 2J
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
2batch_normalization_1/batchnorm/mul/ReadVariableOp2batch_normalization_1/batchnorm/mul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2^
-dense_1/kernel/Regularizer/Abs/ReadVariableOp-dense_1/kernel/Regularizer/Abs/ReadVariableOp2P
&stream_0_conv_1/BiasAdd/ReadVariableOp&stream_0_conv_1/BiasAdd/ReadVariableOp2h
2stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp2stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp2n
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:V R
,
_output_shapes
:����������
"
_user_specified_name
inputs/0
�D
�
E__inference_basemodel_layer_call_and_return_conditional_losses_483189

inputs,
stream_0_conv_1_483076:@$
stream_0_conv_1_483078:@(
batch_normalization_483101:@(
batch_normalization_483103:@(
batch_normalization_483105:@(
batch_normalization_483107:@ 
dense_1_483155:@T
dense_1_483157:T*
batch_normalization_1_483160:T*
batch_normalization_1_483162:T*
batch_normalization_1_483164:T*
batch_normalization_1_483166:T
identity��+batch_normalization/StatefulPartitionedCall�-batch_normalization_1/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�-dense_1/kernel/Regularizer/Abs/ReadVariableOp�'stream_0_conv_1/StatefulPartitionedCall�5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp�
#stream_0_input_drop/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *X
fSRQ
O__inference_stream_0_input_drop_layer_call_and_return_conditional_losses_4830522%
#stream_0_input_drop/PartitionedCall�
'stream_0_conv_1/StatefulPartitionedCallStatefulPartitionedCall,stream_0_input_drop/PartitionedCall:output:0stream_0_conv_1_483076stream_0_conv_1_483078*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_stream_0_conv_1_layer_call_and_return_conditional_losses_4830752)
'stream_0_conv_1/StatefulPartitionedCall�
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall0stream_0_conv_1/StatefulPartitionedCall:output:0batch_normalization_483101batch_normalization_483103batch_normalization_483105batch_normalization_483107*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_4831002-
+batch_normalization/StatefulPartitionedCall�
activation/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_activation_layer_call_and_return_conditional_losses_4831152
activation/PartitionedCall�
stream_0_drop_1/PartitionedCallPartitionedCall#activation/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_stream_0_drop_1_layer_call_and_return_conditional_losses_4831222!
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
 *-
config_proto

CPU

GPU 2J 8� *]
fXRV
T__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_4831292*
(global_average_pooling1d/PartitionedCall�
dense_1_dropout/PartitionedCallPartitionedCall1global_average_pooling1d/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_dense_1_dropout_layer_call_and_return_conditional_losses_4831362!
dense_1_dropout/PartitionedCall�
dense_1/StatefulPartitionedCallStatefulPartitionedCall(dense_1_dropout/PartitionedCall:output:0dense_1_483155dense_1_483157*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������T*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_4831542!
dense_1/StatefulPartitionedCall�
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0batch_normalization_1_483160batch_normalization_1_483162batch_normalization_1_483164batch_normalization_1_483166*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������T*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_4829022/
-batch_normalization_1/StatefulPartitionedCall�
"dense_activation_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������T* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *W
fRRP
N__inference_dense_activation_1_layer_call_and_return_conditional_losses_4831742$
"dense_activation_1/PartitionedCall�
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpstream_0_conv_1_483076*"
_output_shapes
:@*
dtype027
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp�
&stream_0_conv_1/kernel/Regularizer/AbsAbs=stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
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
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_1_483155*
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

Identity�
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall.^dense_1/kernel/Regularizer/Abs/ReadVariableOp(^stream_0_conv_1/StatefulPartitionedCall6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:����������: : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2^
-dense_1/kernel/Regularizer/Abs/ReadVariableOp-dense_1/kernel/Regularizer/Abs/ReadVariableOp2R
'stream_0_conv_1/StatefulPartitionedCall'stream_0_conv_1/StatefulPartitionedCall2n
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
L
0__inference_dense_1_dropout_layer_call_fn_485247

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
 *-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_dense_1_dropout_layer_call_and_return_conditional_losses_4831362
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
�
�
(__inference_model_1_layer_call_fn_484224

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:@
	unknown_5:@T
	unknown_6:T
	unknown_7:T
	unknown_8:T
	unknown_9:T

unknown_10:T
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������T*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_model_1_layer_call_and_return_conditional_losses_4837362
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
_construction_contextkEagerRuntime*C
_input_shapes2
0:����������: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
C__inference_dense_1_layer_call_and_return_conditional_losses_483154

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
�!
�
C__inference_model_1_layer_call_and_return_conditional_losses_483736

inputs&
basemodel_483698:@
basemodel_483700:@
basemodel_483702:@
basemodel_483704:@
basemodel_483706:@
basemodel_483708:@"
basemodel_483710:@T
basemodel_483712:T
basemodel_483714:T
basemodel_483716:T
basemodel_483718:T
basemodel_483720:T
identity��!basemodel/StatefulPartitionedCall�-dense_1/kernel/Regularizer/Abs/ReadVariableOp�5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp�
!basemodel/StatefulPartitionedCallStatefulPartitionedCallinputsbasemodel_483698basemodel_483700basemodel_483702basemodel_483704basemodel_483706basemodel_483708basemodel_483710basemodel_483712basemodel_483714basemodel_483716basemodel_483718basemodel_483720*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������T*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_basemodel_layer_call_and_return_conditional_losses_4836972#
!basemodel/StatefulPartitionedCall�
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpbasemodel_483698*"
_output_shapes
:@*
dtype027
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp�
&stream_0_conv_1/kernel/Regularizer/AbsAbs=stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
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
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpbasemodel_483710*
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
IdentityIdentity*basemodel/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������T2

Identity�
NoOpNoOp"^basemodel/StatefulPartitionedCall.^dense_1/kernel/Regularizer/Abs/ReadVariableOp6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:����������: : : : : : : : : : : : 2F
!basemodel/StatefulPartitionedCall!basemodel/StatefulPartitionedCall2^
-dense_1/kernel/Regularizer/Abs/ReadVariableOp-dense_1/kernel/Regularizer/Abs/ReadVariableOp2n
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�8
�
"__inference__traced_restore_485517
file_prefix=
'assignvariableop_stream_0_conv_1_kernel:@5
'assignvariableop_1_stream_0_conv_1_bias:@:
,assignvariableop_2_batch_normalization_gamma:@9
+assignvariableop_3_batch_normalization_beta:@3
!assignvariableop_4_dense_1_kernel:@T-
assignvariableop_5_dense_1_bias:T<
.assignvariableop_6_batch_normalization_1_gamma:T;
-assignvariableop_7_batch_normalization_1_beta:T@
2assignvariableop_8_batch_normalization_moving_mean:@D
6assignvariableop_9_batch_normalization_moving_variance:@C
5assignvariableop_10_batch_normalization_1_moving_mean:TG
9assignvariableop_11_batch_normalization_1_moving_variance:T
identity_13��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_2�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*-
value$B"B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices�
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*H
_output_shapes6
4:::::::::::::*
dtypes
22
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
AssignVariableOp_4AssignVariableOp!assignvariableop_4_dense_1_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5�
AssignVariableOp_5AssignVariableOpassignvariableop_5_dense_1_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6�
AssignVariableOp_6AssignVariableOp.assignvariableop_6_batch_normalization_1_gammaIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7�
AssignVariableOp_7AssignVariableOp-assignvariableop_7_batch_normalization_1_betaIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8�
AssignVariableOp_8AssignVariableOp2assignvariableop_8_batch_normalization_moving_meanIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9�
AssignVariableOp_9AssignVariableOp6assignvariableop_9_batch_normalization_moving_varianceIdentity_9:output:0"/device:CPU:0*
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
AssignVariableOp_119
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp�
Identity_12Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_12f
Identity_13IdentityIdentity_12:output:0^NoOp_1*
T0*
_output_shapes
: 2
Identity_13�
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 2
NoOp_1"#
identity_13Identity_13:output:0*-
_input_shapes
: : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112(
AssignVariableOp_2AssignVariableOp_22(
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
�
i
0__inference_dense_1_dropout_layer_call_fn_485252

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
 *-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_dense_1_dropout_layer_call_and_return_conditional_losses_4832522
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
�
�
*__inference_basemodel_layer_call_fn_484487

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:@
	unknown_5:@T
	unknown_6:T
	unknown_7:T
	unknown_8:T
	unknown_9:T

unknown_10:T
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������T*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_basemodel_layer_call_and_return_conditional_losses_4831892
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
_construction_contextkEagerRuntime*C
_input_shapes2
0:����������: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
*__inference_basemodel_layer_call_fn_483519
inputs_0
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:@
	unknown_5:@T
	unknown_6:T
	unknown_7:T
	unknown_8:T
	unknown_9:T

unknown_10:T
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������T**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_basemodel_layer_call_and_return_conditional_losses_4834632
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
_construction_contextkEagerRuntime*C
_input_shapes2
0:����������: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
,
_output_shapes
:����������
"
_user_specified_name
inputs_0
�
�
(__inference_model_1_layer_call_fn_484070
left_inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:@
	unknown_5:@T
	unknown_6:T
	unknown_7:T
	unknown_8:T
	unknown_9:T

unknown_10:T
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallleft_inputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������T**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_model_1_layer_call_and_return_conditional_losses_4840142
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
_construction_contextkEagerRuntime*C
_input_shapes2
0:����������: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
,
_output_shapes
:����������
%
_user_specified_nameleft_inputs
�
�
*__inference_basemodel_layer_call_fn_484516

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:@
	unknown_5:@T
	unknown_6:T
	unknown_7:T
	unknown_8:T
	unknown_9:T

unknown_10:T
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������T**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_basemodel_layer_call_and_return_conditional_losses_4834632
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
_construction_contextkEagerRuntime*C
_input_shapes2
0:����������: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
p
T__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_485242

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
:����������@:T P
,
_output_shapes
:����������@
 
_user_specified_nameinputs
�
�
4__inference_batch_normalization_layer_call_fn_485075

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
:����������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_4833382
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:����������@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:����������@
 
_user_specified_nameinputs
�
�
(__inference_dense_1_layer_call_fn_485284

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
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_4831542
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
�!
�
C__inference_model_1_layer_call_and_return_conditional_losses_484111
left_inputs&
basemodel_484073:@
basemodel_484075:@
basemodel_484077:@
basemodel_484079:@
basemodel_484081:@
basemodel_484083:@"
basemodel_484085:@T
basemodel_484087:T
basemodel_484089:T
basemodel_484091:T
basemodel_484093:T
basemodel_484095:T
identity��!basemodel/StatefulPartitionedCall�-dense_1/kernel/Regularizer/Abs/ReadVariableOp�5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp�
!basemodel/StatefulPartitionedCallStatefulPartitionedCallleft_inputsbasemodel_484073basemodel_484075basemodel_484077basemodel_484079basemodel_484081basemodel_484083basemodel_484085basemodel_484087basemodel_484089basemodel_484091basemodel_484093basemodel_484095*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������T*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_basemodel_layer_call_and_return_conditional_losses_4836972#
!basemodel/StatefulPartitionedCall�
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpbasemodel_484073*"
_output_shapes
:@*
dtype027
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp�
&stream_0_conv_1/kernel/Regularizer/AbsAbs=stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
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
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpbasemodel_484085*
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
IdentityIdentity*basemodel/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������T2

Identity�
NoOpNoOp"^basemodel/StatefulPartitionedCall.^dense_1/kernel/Regularizer/Abs/ReadVariableOp6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:����������: : : : : : : : : : : : 2F
!basemodel/StatefulPartitionedCall!basemodel/StatefulPartitionedCall2^
-dense_1/kernel/Regularizer/Abs/ReadVariableOp-dense_1/kernel/Regularizer/Abs/ReadVariableOp2n
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:Y U
,
_output_shapes
:����������
%
_user_specified_nameleft_inputs
�
�
*__inference_basemodel_layer_call_fn_483216
inputs_0
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:@
	unknown_5:@T
	unknown_6:T
	unknown_7:T
	unknown_8:T
	unknown_9:T

unknown_10:T
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������T*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_basemodel_layer_call_and_return_conditional_losses_4831892
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
_construction_contextkEagerRuntime*C
_input_shapes2
0:����������: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
,
_output_shapes
:����������
"
_user_specified_name
inputs_0
�
m
O__inference_stream_0_input_drop_layer_call_and_return_conditional_losses_483052

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:����������2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:����������2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�	
�
4__inference_batch_normalization_layer_call_fn_485036

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
*-
config_proto

CPU

GPU 2J 8� *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_4827162
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
�+
�
O__inference_batch_normalization_layer_call_and_return_conditional_losses_485183

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
:����������@2
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
:����������@2
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
:����������@2
batchnorm/add_1s
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*,
_output_shapes
:����������@2

Identity�
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������@: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:T P
,
_output_shapes
:����������@
 
_user_specified_nameinputs
�
b
F__inference_activation_layer_call_and_return_conditional_losses_485193

inputs
identityS
ReluReluinputs*
T0*,
_output_shapes
:����������@2
Reluk
IdentityIdentityRelu:activations:0*
T0*,
_output_shapes
:����������@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������@:T P
,
_output_shapes
:����������@
 
_user_specified_nameinputs
�J
�
E__inference_basemodel_layer_call_and_return_conditional_losses_483619
inputs_0,
stream_0_conv_1_483573:@$
stream_0_conv_1_483575:@(
batch_normalization_483578:@(
batch_normalization_483580:@(
batch_normalization_483582:@(
batch_normalization_483584:@ 
dense_1_483591:@T
dense_1_483593:T*
batch_normalization_1_483596:T*
batch_normalization_1_483598:T*
batch_normalization_1_483600:T*
batch_normalization_1_483602:T
identity��+batch_normalization/StatefulPartitionedCall�-batch_normalization_1/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�-dense_1/kernel/Regularizer/Abs/ReadVariableOp�'dense_1_dropout/StatefulPartitionedCall�'stream_0_conv_1/StatefulPartitionedCall�5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp�'stream_0_drop_1/StatefulPartitionedCall�+stream_0_input_drop/StatefulPartitionedCall�
+stream_0_input_drop/StatefulPartitionedCallStatefulPartitionedCallinputs_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *X
fSRQ
O__inference_stream_0_input_drop_layer_call_and_return_conditional_losses_4833792-
+stream_0_input_drop/StatefulPartitionedCall�
'stream_0_conv_1/StatefulPartitionedCallStatefulPartitionedCall4stream_0_input_drop/StatefulPartitionedCall:output:0stream_0_conv_1_483573stream_0_conv_1_483575*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_stream_0_conv_1_layer_call_and_return_conditional_losses_4830752)
'stream_0_conv_1/StatefulPartitionedCall�
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall0stream_0_conv_1/StatefulPartitionedCall:output:0batch_normalization_483578batch_normalization_483580batch_normalization_483582batch_normalization_483584*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_4833382-
+batch_normalization/StatefulPartitionedCall�
activation/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_activation_layer_call_and_return_conditional_losses_4831152
activation/PartitionedCall�
'stream_0_drop_1/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0,^stream_0_input_drop/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_stream_0_drop_1_layer_call_and_return_conditional_losses_4832802)
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
 *-
config_proto

CPU

GPU 2J 8� *]
fXRV
T__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_4831292*
(global_average_pooling1d/PartitionedCall�
'dense_1_dropout/StatefulPartitionedCallStatefulPartitionedCall1global_average_pooling1d/PartitionedCall:output:0(^stream_0_drop_1/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_dense_1_dropout_layer_call_and_return_conditional_losses_4832522)
'dense_1_dropout/StatefulPartitionedCall�
dense_1/StatefulPartitionedCallStatefulPartitionedCall0dense_1_dropout/StatefulPartitionedCall:output:0dense_1_483591dense_1_483593*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������T*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_4831542!
dense_1/StatefulPartitionedCall�
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0batch_normalization_1_483596batch_normalization_1_483598batch_normalization_1_483600batch_normalization_1_483602*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������T*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_4829622/
-batch_normalization_1/StatefulPartitionedCall�
"dense_activation_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������T* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *W
fRRP
N__inference_dense_activation_1_layer_call_and_return_conditional_losses_4831742$
"dense_activation_1/PartitionedCall�
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpstream_0_conv_1_483573*"
_output_shapes
:@*
dtype027
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp�
&stream_0_conv_1/kernel/Regularizer/AbsAbs=stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
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
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_1_483591*
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
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall.^dense_1/kernel/Regularizer/Abs/ReadVariableOp(^dense_1_dropout/StatefulPartitionedCall(^stream_0_conv_1/StatefulPartitionedCall6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp(^stream_0_drop_1/StatefulPartitionedCall,^stream_0_input_drop/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:����������: : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2^
-dense_1/kernel/Regularizer/Abs/ReadVariableOp-dense_1/kernel/Regularizer/Abs/ReadVariableOp2R
'dense_1_dropout/StatefulPartitionedCall'dense_1_dropout/StatefulPartitionedCall2R
'stream_0_conv_1/StatefulPartitionedCall'stream_0_conv_1/StatefulPartitionedCall2n
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp2R
'stream_0_drop_1/StatefulPartitionedCall'stream_0_drop_1/StatefulPartitionedCall2Z
+stream_0_input_drop/StatefulPartitionedCall+stream_0_input_drop/StatefulPartitionedCall:V R
,
_output_shapes
:����������
"
_user_specified_name
inputs_0
�
�
C__inference_dense_1_layer_call_and_return_conditional_losses_485300

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
�
�
0__inference_stream_0_conv_1_layer_call_fn_485002

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
:����������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_stream_0_conv_1_layer_call_and_return_conditional_losses_4830752
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:����������@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
__inference_loss_fn_1_485412H
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
-dense_1/kernel/Regularizer/Abs/ReadVariableOp-dense_1/kernel/Regularizer/Abs/ReadVariableOp"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
H
left_inputs9
serving_default_left_inputs:0����������=
	basemodel0
StatefulPartitionedCall:0���������Ttensorflow/serving/predict:��
�
layer-0
layer_with_weights-0
layer-1
trainable_variables
regularization_losses
	variables
	keras_api

signatures
�_default_save_signature
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_network
"
_tf_keras_input_layer
�
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
layer-6
layer-7
layer_with_weights-2
layer-8
layer_with_weights-3
layer-9
layer-10
trainable_variables
regularization_losses
	variables
	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_network
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
 "
trackable_list_wrapper
v
0
1
2
3
4
 5
6
7
8
9
!10
"11"
trackable_list_wrapper
�
#layer_metrics
$metrics
%non_trainable_variables
trainable_variables

&layers
'layer_regularization_losses
regularization_losses
	variables
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
(trainable_variables
)regularization_losses
*	variables
+	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�

kernel
bias
,trainable_variables
-regularization_losses
.	variables
/	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
0axis
	gamma
beta
moving_mean
 moving_variance
1trainable_variables
2regularization_losses
3	variables
4	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
5trainable_variables
6regularization_losses
7	variables
8	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
9trainable_variables
:regularization_losses
;	variables
<	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
=trainable_variables
>regularization_losses
?	variables
@	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
Atrainable_variables
Bregularization_losses
C	variables
D	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�

kernel
bias
Etrainable_variables
Fregularization_losses
G	variables
H	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
Iaxis
	gamma
beta
!moving_mean
"moving_variance
Jtrainable_variables
Kregularization_losses
L	variables
M	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
Ntrainable_variables
Oregularization_losses
P	variables
Q	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
v
0
1
2
3
4
 5
6
7
8
9
!10
"11"
trackable_list_wrapper
�
Rlayer_metrics
Smetrics
Tnon_trainable_variables
trainable_variables

Ulayers
Vlayer_regularization_losses
regularization_losses
	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
,:*@2stream_0_conv_1/kernel
": @2stream_0_conv_1/bias
':%@2batch_normalization/gamma
&:$@2batch_normalization/beta
 :@T2dense_1/kernel
:T2dense_1/bias
):'T2batch_normalization_1/gamma
(:&T2batch_normalization_1/beta
/:-@ (2batch_normalization/moving_mean
3:1@ (2#batch_normalization/moving_variance
1:/T (2!batch_normalization_1/moving_mean
5:3T (2%batch_normalization_1/moving_variance
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
<
0
 1
!2
"3"
trackable_list_wrapper
.
0
1"
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
Wlayer_metrics
Xmetrics
Ynon_trainable_variables
(trainable_variables

Zlayers
[layer_regularization_losses
)regularization_losses
*	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
.
0
1"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
�
\layer_metrics
]metrics
^non_trainable_variables
,trainable_variables

_layers
`layer_regularization_losses
-regularization_losses
.	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
<
0
1
2
 3"
trackable_list_wrapper
�
alayer_metrics
bmetrics
cnon_trainable_variables
1trainable_variables

dlayers
elayer_regularization_losses
2regularization_losses
3	variables
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
flayer_metrics
gmetrics
hnon_trainable_variables
5trainable_variables

ilayers
jlayer_regularization_losses
6regularization_losses
7	variables
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
klayer_metrics
lmetrics
mnon_trainable_variables
9trainable_variables

nlayers
olayer_regularization_losses
:regularization_losses
;	variables
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
player_metrics
qmetrics
rnon_trainable_variables
=trainable_variables

slayers
tlayer_regularization_losses
>regularization_losses
?	variables
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
ulayer_metrics
vmetrics
wnon_trainable_variables
Atrainable_variables

xlayers
ylayer_regularization_losses
Bregularization_losses
C	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
.
0
1"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
�
zlayer_metrics
{metrics
|non_trainable_variables
Etrainable_variables

}layers
~layer_regularization_losses
Fregularization_losses
G	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
<
0
1
!2
"3"
trackable_list_wrapper
�
layer_metrics
�metrics
�non_trainable_variables
Jtrainable_variables
�layers
 �layer_regularization_losses
Kregularization_losses
L	variables
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
�metrics
�non_trainable_variables
Ntrainable_variables
�layers
 �layer_regularization_losses
Oregularization_losses
P	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
<
0
 1
!2
"3"
trackable_list_wrapper
n
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
10"
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
�0"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
0
 1"
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
(
�0"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
!0
"1"
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
�B�
!__inference__wrapped_model_482692left_inputs"�
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
(__inference_model_1_layer_call_fn_483763
(__inference_model_1_layer_call_fn_484224
(__inference_model_1_layer_call_fn_484253
(__inference_model_1_layer_call_fn_484070�
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
C__inference_model_1_layer_call_and_return_conditional_losses_484325
C__inference_model_1_layer_call_and_return_conditional_losses_484446
C__inference_model_1_layer_call_and_return_conditional_losses_484111
C__inference_model_1_layer_call_and_return_conditional_losses_484152�
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
*__inference_basemodel_layer_call_fn_483216
*__inference_basemodel_layer_call_fn_484487
*__inference_basemodel_layer_call_fn_484516
*__inference_basemodel_layer_call_fn_483519
*__inference_basemodel_layer_call_fn_484545
*__inference_basemodel_layer_call_fn_484574�
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
E__inference_basemodel_layer_call_and_return_conditional_losses_484646
E__inference_basemodel_layer_call_and_return_conditional_losses_484767
E__inference_basemodel_layer_call_and_return_conditional_losses_483569
E__inference_basemodel_layer_call_and_return_conditional_losses_483619
E__inference_basemodel_layer_call_and_return_conditional_losses_484839
E__inference_basemodel_layer_call_and_return_conditional_losses_484960�
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
$__inference_signature_wrapper_484195left_inputs"�
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
4__inference_stream_0_input_drop_layer_call_fn_484965
4__inference_stream_0_input_drop_layer_call_fn_484970�
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
O__inference_stream_0_input_drop_layer_call_and_return_conditional_losses_484975
O__inference_stream_0_input_drop_layer_call_and_return_conditional_losses_484987�
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
0__inference_stream_0_conv_1_layer_call_fn_485002�
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
K__inference_stream_0_conv_1_layer_call_and_return_conditional_losses_485023�
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
4__inference_batch_normalization_layer_call_fn_485036
4__inference_batch_normalization_layer_call_fn_485049
4__inference_batch_normalization_layer_call_fn_485062
4__inference_batch_normalization_layer_call_fn_485075�
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
O__inference_batch_normalization_layer_call_and_return_conditional_losses_485095
O__inference_batch_normalization_layer_call_and_return_conditional_losses_485129
O__inference_batch_normalization_layer_call_and_return_conditional_losses_485149
O__inference_batch_normalization_layer_call_and_return_conditional_losses_485183�
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
+__inference_activation_layer_call_fn_485188�
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
F__inference_activation_layer_call_and_return_conditional_losses_485193�
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
0__inference_stream_0_drop_1_layer_call_fn_485198
0__inference_stream_0_drop_1_layer_call_fn_485203�
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
K__inference_stream_0_drop_1_layer_call_and_return_conditional_losses_485208
K__inference_stream_0_drop_1_layer_call_and_return_conditional_losses_485220�
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
9__inference_global_average_pooling1d_layer_call_fn_485225
9__inference_global_average_pooling1d_layer_call_fn_485230�
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
T__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_485236
T__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_485242�
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
0__inference_dense_1_dropout_layer_call_fn_485247
0__inference_dense_1_dropout_layer_call_fn_485252�
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
K__inference_dense_1_dropout_layer_call_and_return_conditional_losses_485257
K__inference_dense_1_dropout_layer_call_and_return_conditional_losses_485269�
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
(__inference_dense_1_layer_call_fn_485284�
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
C__inference_dense_1_layer_call_and_return_conditional_losses_485300�
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
6__inference_batch_normalization_1_layer_call_fn_485313
6__inference_batch_normalization_1_layer_call_fn_485326�
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
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_485346
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_485380�
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
3__inference_dense_activation_1_layer_call_fn_485385�
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
N__inference_dense_activation_1_layer_call_and_return_conditional_losses_485390�
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
__inference_loss_fn_0_485401�
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
__inference_loss_fn_1_485412�
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
!__inference__wrapped_model_482692� "!9�6
/�,
*�'
left_inputs����������
� "5�2
0
	basemodel#� 
	basemodel���������T�
F__inference_activation_layer_call_and_return_conditional_losses_485193b4�1
*�'
%�"
inputs����������@
� "*�'
 �
0����������@
� �
+__inference_activation_layer_call_fn_485188U4�1
*�'
%�"
inputs����������@
� "�����������@�
E__inference_basemodel_layer_call_and_return_conditional_losses_483569u "!>�;
4�1
'�$
inputs_0����������
p 

 
� "%�"
�
0���������T
� �
E__inference_basemodel_layer_call_and_return_conditional_losses_483619u !">�;
4�1
'�$
inputs_0����������
p

 
� "%�"
�
0���������T
� �
E__inference_basemodel_layer_call_and_return_conditional_losses_484646s "!<�9
2�/
%�"
inputs����������
p 

 
� "%�"
�
0���������T
� �
E__inference_basemodel_layer_call_and_return_conditional_losses_484767s !"<�9
2�/
%�"
inputs����������
p

 
� "%�"
�
0���������T
� �
E__inference_basemodel_layer_call_and_return_conditional_losses_484839z "!C�@
9�6
,�)
'�$
inputs/0����������
p 

 
� "%�"
�
0���������T
� �
E__inference_basemodel_layer_call_and_return_conditional_losses_484960z !"C�@
9�6
,�)
'�$
inputs/0����������
p

 
� "%�"
�
0���������T
� �
*__inference_basemodel_layer_call_fn_483216h "!>�;
4�1
'�$
inputs_0����������
p 

 
� "����������T�
*__inference_basemodel_layer_call_fn_483519h !">�;
4�1
'�$
inputs_0����������
p

 
� "����������T�
*__inference_basemodel_layer_call_fn_484487f "!<�9
2�/
%�"
inputs����������
p 

 
� "����������T�
*__inference_basemodel_layer_call_fn_484516f !"<�9
2�/
%�"
inputs����������
p

 
� "����������T�
*__inference_basemodel_layer_call_fn_484545m "!C�@
9�6
,�)
'�$
inputs/0����������
p 

 
� "����������T�
*__inference_basemodel_layer_call_fn_484574m !"C�@
9�6
,�)
'�$
inputs/0����������
p

 
� "����������T�
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_485346b"!3�0
)�&
 �
inputs���������T
p 
� "%�"
�
0���������T
� �
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_485380b!"3�0
)�&
 �
inputs���������T
p
� "%�"
�
0���������T
� �
6__inference_batch_normalization_1_layer_call_fn_485313U"!3�0
)�&
 �
inputs���������T
p 
� "����������T�
6__inference_batch_normalization_1_layer_call_fn_485326U!"3�0
)�&
 �
inputs���������T
p
� "����������T�
O__inference_batch_normalization_layer_call_and_return_conditional_losses_485095| @�=
6�3
-�*
inputs������������������@
p 
� "2�/
(�%
0������������������@
� �
O__inference_batch_normalization_layer_call_and_return_conditional_losses_485129| @�=
6�3
-�*
inputs������������������@
p
� "2�/
(�%
0������������������@
� �
O__inference_batch_normalization_layer_call_and_return_conditional_losses_485149l 8�5
.�+
%�"
inputs����������@
p 
� "*�'
 �
0����������@
� �
O__inference_batch_normalization_layer_call_and_return_conditional_losses_485183l 8�5
.�+
%�"
inputs����������@
p
� "*�'
 �
0����������@
� �
4__inference_batch_normalization_layer_call_fn_485036o @�=
6�3
-�*
inputs������������������@
p 
� "%�"������������������@�
4__inference_batch_normalization_layer_call_fn_485049o @�=
6�3
-�*
inputs������������������@
p
� "%�"������������������@�
4__inference_batch_normalization_layer_call_fn_485062_ 8�5
.�+
%�"
inputs����������@
p 
� "�����������@�
4__inference_batch_normalization_layer_call_fn_485075_ 8�5
.�+
%�"
inputs����������@
p
� "�����������@�
K__inference_dense_1_dropout_layer_call_and_return_conditional_losses_485257\3�0
)�&
 �
inputs���������@
p 
� "%�"
�
0���������@
� �
K__inference_dense_1_dropout_layer_call_and_return_conditional_losses_485269\3�0
)�&
 �
inputs���������@
p
� "%�"
�
0���������@
� �
0__inference_dense_1_dropout_layer_call_fn_485247O3�0
)�&
 �
inputs���������@
p 
� "����������@�
0__inference_dense_1_dropout_layer_call_fn_485252O3�0
)�&
 �
inputs���������@
p
� "����������@�
C__inference_dense_1_layer_call_and_return_conditional_losses_485300\/�,
%�"
 �
inputs���������@
� "%�"
�
0���������T
� {
(__inference_dense_1_layer_call_fn_485284O/�,
%�"
 �
inputs���������@
� "����������T�
N__inference_dense_activation_1_layer_call_and_return_conditional_losses_485390X/�,
%�"
 �
inputs���������T
� "%�"
�
0���������T
� �
3__inference_dense_activation_1_layer_call_fn_485385K/�,
%�"
 �
inputs���������T
� "����������T�
T__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_485236{I�F
?�<
6�3
inputs'���������������������������

 
� ".�+
$�!
0������������������
� �
T__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_485242a8�5
.�+
%�"
inputs����������@

 
� "%�"
�
0���������@
� �
9__inference_global_average_pooling1d_layer_call_fn_485225nI�F
?�<
6�3
inputs'���������������������������

 
� "!��������������������
9__inference_global_average_pooling1d_layer_call_fn_485230T8�5
.�+
%�"
inputs����������@

 
� "����������@;
__inference_loss_fn_0_485401�

� 
� "� ;
__inference_loss_fn_1_485412�

� 
� "� �
C__inference_model_1_layer_call_and_return_conditional_losses_484111x "!A�>
7�4
*�'
left_inputs����������
p 

 
� "%�"
�
0���������T
� �
C__inference_model_1_layer_call_and_return_conditional_losses_484152x !"A�>
7�4
*�'
left_inputs����������
p

 
� "%�"
�
0���������T
� �
C__inference_model_1_layer_call_and_return_conditional_losses_484325s "!<�9
2�/
%�"
inputs����������
p 

 
� "%�"
�
0���������T
� �
C__inference_model_1_layer_call_and_return_conditional_losses_484446s !"<�9
2�/
%�"
inputs����������
p

 
� "%�"
�
0���������T
� �
(__inference_model_1_layer_call_fn_483763k "!A�>
7�4
*�'
left_inputs����������
p 

 
� "����������T�
(__inference_model_1_layer_call_fn_484070k !"A�>
7�4
*�'
left_inputs����������
p

 
� "����������T�
(__inference_model_1_layer_call_fn_484224f "!<�9
2�/
%�"
inputs����������
p 

 
� "����������T�
(__inference_model_1_layer_call_fn_484253f !"<�9
2�/
%�"
inputs����������
p

 
� "����������T�
$__inference_signature_wrapper_484195� "!H�E
� 
>�;
9
left_inputs*�'
left_inputs����������"5�2
0
	basemodel#� 
	basemodel���������T�
K__inference_stream_0_conv_1_layer_call_and_return_conditional_losses_485023f4�1
*�'
%�"
inputs����������
� "*�'
 �
0����������@
� �
0__inference_stream_0_conv_1_layer_call_fn_485002Y4�1
*�'
%�"
inputs����������
� "�����������@�
K__inference_stream_0_drop_1_layer_call_and_return_conditional_losses_485208f8�5
.�+
%�"
inputs����������@
p 
� "*�'
 �
0����������@
� �
K__inference_stream_0_drop_1_layer_call_and_return_conditional_losses_485220f8�5
.�+
%�"
inputs����������@
p
� "*�'
 �
0����������@
� �
0__inference_stream_0_drop_1_layer_call_fn_485198Y8�5
.�+
%�"
inputs����������@
p 
� "�����������@�
0__inference_stream_0_drop_1_layer_call_fn_485203Y8�5
.�+
%�"
inputs����������@
p
� "�����������@�
O__inference_stream_0_input_drop_layer_call_and_return_conditional_losses_484975f8�5
.�+
%�"
inputs����������
p 
� "*�'
 �
0����������
� �
O__inference_stream_0_input_drop_layer_call_and_return_conditional_losses_484987f8�5
.�+
%�"
inputs����������
p
� "*�'
 �
0����������
� �
4__inference_stream_0_input_drop_layer_call_fn_484965Y8�5
.�+
%�"
inputs����������
p 
� "������������
4__inference_stream_0_input_drop_layer_call_fn_484970Y8�5
.�+
%�"
inputs����������
p
� "�����������