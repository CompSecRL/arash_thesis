ЄД=
жЉ
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
В
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
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
-
Tanh
x"T
y"T"
Ttype:

2
Ц
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 И"serve*2.6.22v2.6.1-9-gc2363d6d0258ґљ9
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
М
stream_0_conv_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*'
shared_namestream_0_conv_2/kernel
Е
*stream_0_conv_2/kernel/Read/ReadVariableOpReadVariableOpstream_0_conv_2/kernel*"
_output_shapes
:@@*
dtype0
А
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
М
stream_0_conv_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*'
shared_namestream_0_conv_3/kernel
Е
*stream_0_conv_3/kernel/Read/ReadVariableOpReadVariableOpstream_0_conv_3/kernel*"
_output_shapes
:@@*
dtype0
А
stream_0_conv_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_namestream_0_conv_3/bias
y
(stream_0_conv_3/bias/Read/ReadVariableOpReadVariableOpstream_0_conv_3/bias*
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
x
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*
shared_namedense_1/kernel
q
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes

:@@*
dtype0
p
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_1/bias
i
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes
:@*
dtype0
О
batch_normalization_3/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_3/gamma
З
/batch_normalization_3/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_3/gamma*
_output_shapes
:@*
dtype0
М
batch_normalization_3/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_namebatch_normalization_3/beta
Е
.batch_normalization_3/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_3/beta*
_output_shapes
:@*
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
shape:@*2
shared_name#!batch_normalization_3/moving_mean
У
5batch_normalization_3/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_3/moving_mean*
_output_shapes
:@*
dtype0
Ґ
%batch_normalization_3/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*6
shared_name'%batch_normalization_3/moving_variance
Ы
9batch_normalization_3/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_3/moving_variance*
_output_shapes
:@*
dtype0

NoOpNoOp
њV
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*ъU
valueрUBнU BжU
Ц
layer-0
layer_with_weights-0
layer-1
trainable_variables
	variables
regularization_losses
	keras_api

signatures
 
ћ
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
layer_with_weights-2
layer-7
layer_with_weights-3
layer-8
layer-9
layer-10
layer-11
layer_with_weights-4
layer-12
layer_with_weights-5
layer-13
layer-14
layer-15
layer-16
layer-17
layer-18
layer_with_weights-6
layer-19
layer_with_weights-7
layer-20
layer-21
trainable_variables
	variables
 regularization_losses
!	keras_api
v
"0
#1
$2
%3
&4
'5
(6
)7
*8
+9
,10
-11
.12
/13
014
115
ґ
"0
#1
$2
%3
24
35
&6
'7
(8
)9
410
511
*12
+13
,14
-15
616
717
.18
/19
020
121
822
923
 
≠

:layers
trainable_variables
;layer_regularization_losses
<metrics
	variables
regularization_losses
=non_trainable_variables
>layer_metrics
 
 
R
?trainable_variables
@	variables
Aregularization_losses
B	keras_api
h

"kernel
#bias
Ctrainable_variables
D	variables
Eregularization_losses
F	keras_api
Ч
Gaxis
	$gamma
%beta
2moving_mean
3moving_variance
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
R
Ttrainable_variables
U	variables
Vregularization_losses
W	keras_api
h

&kernel
'bias
Xtrainable_variables
Y	variables
Zregularization_losses
[	keras_api
Ч
\axis
	(gamma
)beta
4moving_mean
5moving_variance
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
h

*kernel
+bias
mtrainable_variables
n	variables
oregularization_losses
p	keras_api
Ч
qaxis
	,gamma
-beta
6moving_mean
7moving_variance
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
l

.kernel
/bias
Кtrainable_variables
Л	variables
Мregularization_losses
Н	keras_api
Ь
	Оaxis
	0gamma
1beta
8moving_mean
9moving_variance
Пtrainable_variables
Р	variables
Сregularization_losses
Т	keras_api
V
Уtrainable_variables
Ф	variables
Хregularization_losses
Ц	keras_api
v
"0
#1
$2
%3
&4
'5
(6
)7
*8
+9
,10
-11
.12
/13
014
115
ґ
"0
#1
$2
%3
24
35
&6
'7
(8
)9
410
511
*12
+13
,14
-15
616
717
.18
/19
020
121
822
923
 
≤
Чlayers
trainable_variables
 Шlayer_regularization_losses
Щmetrics
	variables
 regularization_losses
Ъnon_trainable_variables
Ыlayer_metrics
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

0
1
 
 
8
20
31
42
53
64
75
86
97
 
 
 
 
≤
Ьlayers
?trainable_variables
 Эlayer_regularization_losses
Юmetrics
@	variables
Aregularization_losses
Яnon_trainable_variables
†layer_metrics

"0
#1

"0
#1
 
≤
°layers
Ctrainable_variables
 Ґlayer_regularization_losses
£metrics
D	variables
Eregularization_losses
§non_trainable_variables
•layer_metrics
 

$0
%1

$0
%1
22
33
 
≤
¶layers
Htrainable_variables
 Іlayer_regularization_losses
®metrics
I	variables
Jregularization_losses
©non_trainable_variables
™layer_metrics
 
 
 
≤
Ђlayers
Ltrainable_variables
 ђlayer_regularization_losses
≠metrics
M	variables
Nregularization_losses
Ѓnon_trainable_variables
ѓlayer_metrics
 
 
 
≤
∞layers
Ptrainable_variables
 ±layer_regularization_losses
≤metrics
Q	variables
Rregularization_losses
≥non_trainable_variables
іlayer_metrics
 
 
 
≤
µlayers
Ttrainable_variables
 ґlayer_regularization_losses
Јmetrics
U	variables
Vregularization_losses
Єnon_trainable_variables
єlayer_metrics

&0
'1

&0
'1
 
≤
Їlayers
Xtrainable_variables
 їlayer_regularization_losses
Љmetrics
Y	variables
Zregularization_losses
љnon_trainable_variables
Њlayer_metrics
 

(0
)1

(0
)1
42
53
 
≤
њlayers
]trainable_variables
 јlayer_regularization_losses
Ѕmetrics
^	variables
_regularization_losses
¬non_trainable_variables
√layer_metrics
 
 
 
≤
ƒlayers
atrainable_variables
 ≈layer_regularization_losses
∆metrics
b	variables
cregularization_losses
«non_trainable_variables
»layer_metrics
 
 
 
≤
…layers
etrainable_variables
  layer_regularization_losses
Ћmetrics
f	variables
gregularization_losses
ћnon_trainable_variables
Ќlayer_metrics
 
 
 
≤
ќlayers
itrainable_variables
 ѕlayer_regularization_losses
–metrics
j	variables
kregularization_losses
—non_trainable_variables
“layer_metrics

*0
+1

*0
+1
 
≤
”layers
mtrainable_variables
 ‘layer_regularization_losses
’metrics
n	variables
oregularization_losses
÷non_trainable_variables
„layer_metrics
 

,0
-1

,0
-1
62
73
 
≤
Ўlayers
rtrainable_variables
 ўlayer_regularization_losses
Џmetrics
s	variables
tregularization_losses
џnon_trainable_variables
№layer_metrics
 
 
 
≤
Ёlayers
vtrainable_variables
 ёlayer_regularization_losses
яmetrics
w	variables
xregularization_losses
аnon_trainable_variables
бlayer_metrics
 
 
 
≤
вlayers
ztrainable_variables
 гlayer_regularization_losses
дmetrics
{	variables
|regularization_losses
еnon_trainable_variables
жlayer_metrics
 
 
 
≥
зlayers
~trainable_variables
 иlayer_regularization_losses
йmetrics
	variables
Аregularization_losses
кnon_trainable_variables
лlayer_metrics
 
 
 
µ
мlayers
Вtrainable_variables
 нlayer_regularization_losses
оmetrics
Г	variables
Дregularization_losses
пnon_trainable_variables
рlayer_metrics
 
 
 
µ
сlayers
Жtrainable_variables
 тlayer_regularization_losses
уmetrics
З	variables
Иregularization_losses
фnon_trainable_variables
хlayer_metrics

.0
/1

.0
/1
 
µ
цlayers
Кtrainable_variables
 чlayer_regularization_losses
шmetrics
Л	variables
Мregularization_losses
щnon_trainable_variables
ъlayer_metrics
 

00
11

00
11
82
93
 
µ
ыlayers
Пtrainable_variables
 ьlayer_regularization_losses
эmetrics
Р	variables
Сregularization_losses
юnon_trainable_variables
€layer_metrics
 
 
 
µ
Аlayers
Уtrainable_variables
 Бlayer_regularization_losses
Вmetrics
Ф	variables
Хregularization_losses
Гnon_trainable_variables
Дlayer_metrics
¶
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
19
20
21
 
 
8
20
31
42
53
64
75
86
97
 
 
 
 
 
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
20
31
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
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
Ж
serving_default_left_inputsPlaceholder*+
_output_shapes
:€€€€€€€€€}*
dtype0* 
shape:€€€€€€€€€}
Ц
StatefulPartitionedCallStatefulPartitionedCallserving_default_left_inputsstream_0_conv_1/kernelstream_0_conv_1/bias#batch_normalization/moving_variancebatch_normalization/gammabatch_normalization/moving_meanbatch_normalization/betastream_0_conv_2/kernelstream_0_conv_2/bias%batch_normalization_1/moving_variancebatch_normalization_1/gamma!batch_normalization_1/moving_meanbatch_normalization_1/betastream_0_conv_3/kernelstream_0_conv_3/bias%batch_normalization_2/moving_variancebatch_normalization_2/gamma!batch_normalization_2/moving_meanbatch_normalization_2/betadense_1/kerneldense_1/bias%batch_normalization_3/moving_variancebatch_normalization_3/gamma!batch_normalization_3/moving_meanbatch_normalization_3/beta*$
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@*:
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *.
f)R'
%__inference_signature_wrapper_3295587
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
√
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename*stream_0_conv_1/kernel/Read/ReadVariableOp(stream_0_conv_1/bias/Read/ReadVariableOp-batch_normalization/gamma/Read/ReadVariableOp,batch_normalization/beta/Read/ReadVariableOp*stream_0_conv_2/kernel/Read/ReadVariableOp(stream_0_conv_2/bias/Read/ReadVariableOp/batch_normalization_1/gamma/Read/ReadVariableOp.batch_normalization_1/beta/Read/ReadVariableOp*stream_0_conv_3/kernel/Read/ReadVariableOp(stream_0_conv_3/bias/Read/ReadVariableOp/batch_normalization_2/gamma/Read/ReadVariableOp.batch_normalization_2/beta/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOp/batch_normalization_3/gamma/Read/ReadVariableOp.batch_normalization_3/beta/Read/ReadVariableOp3batch_normalization/moving_mean/Read/ReadVariableOp7batch_normalization/moving_variance/Read/ReadVariableOp5batch_normalization_1/moving_mean/Read/ReadVariableOp9batch_normalization_1/moving_variance/Read/ReadVariableOp5batch_normalization_2/moving_mean/Read/ReadVariableOp9batch_normalization_2/moving_variance/Read/ReadVariableOp5batch_normalization_3/moving_mean/Read/ReadVariableOp9batch_normalization_3/moving_variance/Read/ReadVariableOpConst*%
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
GPU2*0J 8В *)
f$R"
 __inference__traced_save_3298560
ё
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamestream_0_conv_1/kernelstream_0_conv_1/biasbatch_normalization/gammabatch_normalization/betastream_0_conv_2/kernelstream_0_conv_2/biasbatch_normalization_1/gammabatch_normalization_1/betastream_0_conv_3/kernelstream_0_conv_3/biasbatch_normalization_2/gammabatch_normalization_2/betadense_1/kerneldense_1/biasbatch_normalization_3/gammabatch_normalization_3/betabatch_normalization/moving_mean#batch_normalization/moving_variance!batch_normalization_1/moving_mean%batch_normalization_1/moving_variance!batch_normalization_2/moving_mean%batch_normalization_2/moving_variance!batch_normalization_3/moving_mean%batch_normalization_3/moving_variance*$
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
GPU2*0J 8В *,
f'R%
#__inference__traced_restore_3298642ЄФ8
у,
О
L__inference_stream_0_conv_3_layer_call_and_return_conditional_losses_3297983

inputsA
+conv1d_expanddims_1_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identityИҐBiasAdd/ReadVariableOpҐ"conv1d/ExpandDims_1/ReadVariableOpҐ5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOpҐ8stream_0_conv_3/kernel/Regularizer/Square/ReadVariableOpy
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
:€€€€€€€€€@2
conv1d/ExpandDimsЄ
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
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
:@@2
conv1d/ExpandDims_1ґ
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:€€€€€€€€€@*
paddingSAME*
strides
2
conv1dТ
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:€€€€€€€€€@*
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
:€€€€€€€€€@2	
BiasAddЩ
(stream_0_conv_3/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(stream_0_conv_3/kernel/Regularizer/Constё
5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype027
5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp√
&stream_0_conv_3/kernel/Regularizer/AbsAbs=stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@@2(
&stream_0_conv_3/kernel/Regularizer/Abs≠
*stream_0_conv_3/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_0_conv_3/kernel/Regularizer/Const_1ў
&stream_0_conv_3/kernel/Regularizer/SumSum*stream_0_conv_3/kernel/Regularizer/Abs:y:03stream_0_conv_3/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_3/kernel/Regularizer/SumЩ
(stream_0_conv_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2*
(stream_0_conv_3/kernel/Regularizer/mul/x№
&stream_0_conv_3/kernel/Regularizer/mulMul1stream_0_conv_3/kernel/Regularizer/mul/x:output:0/stream_0_conv_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_3/kernel/Regularizer/mulў
&stream_0_conv_3/kernel/Regularizer/addAddV21stream_0_conv_3/kernel/Regularizer/Const:output:0*stream_0_conv_3/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&stream_0_conv_3/kernel/Regularizer/addд
8stream_0_conv_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype02:
8stream_0_conv_3/kernel/Regularizer/Square/ReadVariableOpѕ
)stream_0_conv_3/kernel/Regularizer/SquareSquare@stream_0_conv_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@@2+
)stream_0_conv_3/kernel/Regularizer/Square≠
*stream_0_conv_3/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_0_conv_3/kernel/Regularizer/Const_2а
(stream_0_conv_3/kernel/Regularizer/Sum_1Sum-stream_0_conv_3/kernel/Regularizer/Square:y:03stream_0_conv_3/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2*
(stream_0_conv_3/kernel/Regularizer/Sum_1Э
*stream_0_conv_3/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2,
*stream_0_conv_3/kernel/Regularizer/mul_1/xд
(stream_0_conv_3/kernel/Regularizer/mul_1Mul3stream_0_conv_3/kernel/Regularizer/mul_1/x:output:01stream_0_conv_3/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2*
(stream_0_conv_3/kernel/Regularizer/mul_1Ў
(stream_0_conv_3/kernel/Regularizer/add_1AddV2*stream_0_conv_3/kernel/Regularizer/add:z:0,stream_0_conv_3/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2*
(stream_0_conv_3/kernel/Regularizer/add_1o
IdentityIdentityBiasAdd:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€@2

Identity€
NoOpNoOp^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp6^stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp9^stream_0_conv_3/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp2n
5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp2t
8stream_0_conv_3/kernel/Regularizer/Square/ReadVariableOp8stream_0_conv_3/kernel/Regularizer/Square/ReadVariableOp:S O
+
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
В+
л
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_3298143

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
:€€€€€€€€€@2
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
:€€€€€€€€€@2
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
:€€€€€€€€€@2
batchnorm/add_1r
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€@2

Identityт
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€@: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:S O
+
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
√ 
Н
F__inference_basemodel_layer_call_and_return_conditional_losses_3294106
inputs_0-
stream_0_conv_1_3293977:@%
stream_0_conv_1_3293979:@)
batch_normalization_3293982:@)
batch_normalization_3293984:@)
batch_normalization_3293986:@)
batch_normalization_3293988:@-
stream_0_conv_2_3293994:@@%
stream_0_conv_2_3293996:@+
batch_normalization_1_3293999:@+
batch_normalization_1_3294001:@+
batch_normalization_1_3294003:@+
batch_normalization_1_3294005:@-
stream_0_conv_3_3294011:@@%
stream_0_conv_3_3294013:@+
batch_normalization_2_3294016:@+
batch_normalization_2_3294018:@+
batch_normalization_2_3294020:@+
batch_normalization_2_3294022:@!
dense_1_3294030:@@
dense_1_3294032:@+
batch_normalization_3_3294035:@+
batch_normalization_3_3294037:@+
batch_normalization_3_3294039:@+
batch_normalization_3_3294041:@
identityИҐ+batch_normalization/StatefulPartitionedCallҐ-batch_normalization_1/StatefulPartitionedCallҐ-batch_normalization_2/StatefulPartitionedCallҐ-batch_normalization_3/StatefulPartitionedCallҐdense_1/StatefulPartitionedCallҐ-dense_1/kernel/Regularizer/Abs/ReadVariableOpҐ0dense_1/kernel/Regularizer/Square/ReadVariableOpҐ'stream_0_conv_1/StatefulPartitionedCallҐ5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpҐ8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOpҐ'stream_0_conv_2/StatefulPartitionedCallҐ5stream_0_conv_2/kernel/Regularizer/Abs/ReadVariableOpҐ8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOpҐ'stream_0_conv_3/StatefulPartitionedCallҐ5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOpҐ8stream_0_conv_3/kernel/Regularizer/Square/ReadVariableOp€
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
GPU2*0J 8В *Y
fTRR
P__inference_stream_0_input_drop_layer_call_and_return_conditional_losses_32928722%
#stream_0_input_drop/PartitionedCallз
'stream_0_conv_1/StatefulPartitionedCallStatefulPartitionedCall,stream_0_input_drop/PartitionedCall:output:0stream_0_conv_1_3293977stream_0_conv_1_3293979*
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
GPU2*0J 8В *U
fPRN
L__inference_stream_0_conv_1_layer_call_and_return_conditional_losses_32929042)
'stream_0_conv_1/StatefulPartitionedCallљ
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall0stream_0_conv_1/StatefulPartitionedCall:output:0batch_normalization_3293982batch_normalization_3293984batch_normalization_3293986batch_normalization_3293988*
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
GPU2*0J 8В *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_32929292-
+batch_normalization/StatefulPartitionedCallР
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
GPU2*0J 8В *P
fKRI
G__inference_activation_layer_call_and_return_conditional_losses_32929442
activation/PartitionedCallЧ
"stream_0_maxpool_1/PartitionedCallPartitionedCall#activation/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€>@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_stream_0_maxpool_1_layer_call_and_return_conditional_losses_32929532$
"stream_0_maxpool_1/PartitionedCallЦ
stream_0_drop_1/PartitionedCallPartitionedCall+stream_0_maxpool_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€>@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_stream_0_drop_1_layer_call_and_return_conditional_losses_32929602!
stream_0_drop_1/PartitionedCallг
'stream_0_conv_2/StatefulPartitionedCallStatefulPartitionedCall(stream_0_drop_1/PartitionedCall:output:0stream_0_conv_2_3293994stream_0_conv_2_3293996*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€>@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_stream_0_conv_2_layer_call_and_return_conditional_losses_32929922)
'stream_0_conv_2/StatefulPartitionedCallЋ
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall0stream_0_conv_2/StatefulPartitionedCall:output:0batch_normalization_1_3293999batch_normalization_1_3294001batch_normalization_1_3294003batch_normalization_1_3294005*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€>@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *[
fVRT
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_32930172/
-batch_normalization_1/StatefulPartitionedCallШ
activation_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€>@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_activation_1_layer_call_and_return_conditional_losses_32930322
activation_1/PartitionedCallЩ
"stream_0_maxpool_2/PartitionedCallPartitionedCall%activation_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_stream_0_maxpool_2_layer_call_and_return_conditional_losses_32930412$
"stream_0_maxpool_2/PartitionedCallЦ
stream_0_drop_2/PartitionedCallPartitionedCall+stream_0_maxpool_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_stream_0_drop_2_layer_call_and_return_conditional_losses_32930482!
stream_0_drop_2/PartitionedCallг
'stream_0_conv_3/StatefulPartitionedCallStatefulPartitionedCall(stream_0_drop_2/PartitionedCall:output:0stream_0_conv_3_3294011stream_0_conv_3_3294013*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_stream_0_conv_3_layer_call_and_return_conditional_losses_32930802)
'stream_0_conv_3/StatefulPartitionedCallЋ
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall0stream_0_conv_3/StatefulPartitionedCall:output:0batch_normalization_2_3294016batch_normalization_2_3294018batch_normalization_2_3294020batch_normalization_2_3294022*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *[
fVRT
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_32931052/
-batch_normalization_2/StatefulPartitionedCallШ
activation_2/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_activation_2_layer_call_and_return_conditional_losses_32931202
activation_2/PartitionedCallЩ
"stream_0_maxpool_3/PartitionedCallPartitionedCall%activation_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_stream_0_maxpool_3_layer_call_and_return_conditional_losses_32931292$
"stream_0_maxpool_3/PartitionedCallЦ
stream_0_drop_3/PartitionedCallPartitionedCall+stream_0_maxpool_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_stream_0_drop_3_layer_call_and_return_conditional_losses_32931362!
stream_0_drop_3/PartitionedCall™
(global_average_pooling1d/PartitionedCallPartitionedCall(stream_0_drop_3/PartitionedCall:output:0*
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
GPU2*0J 8В *^
fYRW
U__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_32931432*
(global_average_pooling1d/PartitionedCallШ
dense_1_dropout/PartitionedCallPartitionedCall1global_average_pooling1d/PartitionedCall:output:0*
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
GPU2*0J 8В *U
fPRN
L__inference_dense_1_dropout_layer_call_and_return_conditional_losses_32931502!
dense_1_dropout/PartitionedCallЈ
dense_1/StatefulPartitionedCallStatefulPartitionedCall(dense_1_dropout/PartitionedCall:output:0dense_1_3294030dense_1_3294032*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_32931772!
dense_1/StatefulPartitionedCallњ
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0batch_normalization_3_3294035batch_normalization_3_3294037batch_normalization_3_3294039batch_normalization_3_3294041*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *[
fVRT
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_32927222/
-batch_normalization_3/StatefulPartitionedCall¶
"dense_activation_1/PartitionedCallPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0*
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
GPU2*0J 8В *X
fSRQ
O__inference_dense_activation_1_layer_call_and_return_conditional_losses_32931962$
"dense_activation_1/PartitionedCallЩ
(stream_0_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(stream_0_conv_1/kernel/Regularizer/Const 
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpstream_0_conv_1_3293977*"
_output_shapes
:@*
dtype027
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp√
&stream_0_conv_1/kernel/Regularizer/AbsAbs=stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_0_conv_1/kernel/Regularizer/Abs≠
*stream_0_conv_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_0_conv_1/kernel/Regularizer/Const_1ў
&stream_0_conv_1/kernel/Regularizer/SumSum*stream_0_conv_1/kernel/Regularizer/Abs:y:03stream_0_conv_1/kernel/Regularizer/Const_1:output:0*
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
&stream_0_conv_1/kernel/Regularizer/mulў
&stream_0_conv_1/kernel/Regularizer/addAddV21stream_0_conv_1/kernel/Regularizer/Const:output:0*stream_0_conv_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/add–
8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpstream_0_conv_1_3293977*"
_output_shapes
:@*
dtype02:
8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOpѕ
)stream_0_conv_1/kernel/Regularizer/SquareSquare@stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2+
)stream_0_conv_1/kernel/Regularizer/Square≠
*stream_0_conv_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_0_conv_1/kernel/Regularizer/Const_2а
(stream_0_conv_1/kernel/Regularizer/Sum_1Sum-stream_0_conv_1/kernel/Regularizer/Square:y:03stream_0_conv_1/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2*
(stream_0_conv_1/kernel/Regularizer/Sum_1Э
*stream_0_conv_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2,
*stream_0_conv_1/kernel/Regularizer/mul_1/xд
(stream_0_conv_1/kernel/Regularizer/mul_1Mul3stream_0_conv_1/kernel/Regularizer/mul_1/x:output:01stream_0_conv_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2*
(stream_0_conv_1/kernel/Regularizer/mul_1Ў
(stream_0_conv_1/kernel/Regularizer/add_1AddV2*stream_0_conv_1/kernel/Regularizer/add:z:0,stream_0_conv_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2*
(stream_0_conv_1/kernel/Regularizer/add_1Щ
(stream_0_conv_2/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(stream_0_conv_2/kernel/Regularizer/Const 
5stream_0_conv_2/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpstream_0_conv_2_3293994*"
_output_shapes
:@@*
dtype027
5stream_0_conv_2/kernel/Regularizer/Abs/ReadVariableOp√
&stream_0_conv_2/kernel/Regularizer/AbsAbs=stream_0_conv_2/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@@2(
&stream_0_conv_2/kernel/Regularizer/Abs≠
*stream_0_conv_2/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_0_conv_2/kernel/Regularizer/Const_1ў
&stream_0_conv_2/kernel/Regularizer/SumSum*stream_0_conv_2/kernel/Regularizer/Abs:y:03stream_0_conv_2/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_2/kernel/Regularizer/SumЩ
(stream_0_conv_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2*
(stream_0_conv_2/kernel/Regularizer/mul/x№
&stream_0_conv_2/kernel/Regularizer/mulMul1stream_0_conv_2/kernel/Regularizer/mul/x:output:0/stream_0_conv_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_2/kernel/Regularizer/mulў
&stream_0_conv_2/kernel/Regularizer/addAddV21stream_0_conv_2/kernel/Regularizer/Const:output:0*stream_0_conv_2/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&stream_0_conv_2/kernel/Regularizer/add–
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpstream_0_conv_2_3293994*"
_output_shapes
:@@*
dtype02:
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOpѕ
)stream_0_conv_2/kernel/Regularizer/SquareSquare@stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@@2+
)stream_0_conv_2/kernel/Regularizer/Square≠
*stream_0_conv_2/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_0_conv_2/kernel/Regularizer/Const_2а
(stream_0_conv_2/kernel/Regularizer/Sum_1Sum-stream_0_conv_2/kernel/Regularizer/Square:y:03stream_0_conv_2/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2*
(stream_0_conv_2/kernel/Regularizer/Sum_1Э
*stream_0_conv_2/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2,
*stream_0_conv_2/kernel/Regularizer/mul_1/xд
(stream_0_conv_2/kernel/Regularizer/mul_1Mul3stream_0_conv_2/kernel/Regularizer/mul_1/x:output:01stream_0_conv_2/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2*
(stream_0_conv_2/kernel/Regularizer/mul_1Ў
(stream_0_conv_2/kernel/Regularizer/add_1AddV2*stream_0_conv_2/kernel/Regularizer/add:z:0,stream_0_conv_2/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2*
(stream_0_conv_2/kernel/Regularizer/add_1Щ
(stream_0_conv_3/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(stream_0_conv_3/kernel/Regularizer/Const 
5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpstream_0_conv_3_3294011*"
_output_shapes
:@@*
dtype027
5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp√
&stream_0_conv_3/kernel/Regularizer/AbsAbs=stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@@2(
&stream_0_conv_3/kernel/Regularizer/Abs≠
*stream_0_conv_3/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_0_conv_3/kernel/Regularizer/Const_1ў
&stream_0_conv_3/kernel/Regularizer/SumSum*stream_0_conv_3/kernel/Regularizer/Abs:y:03stream_0_conv_3/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_3/kernel/Regularizer/SumЩ
(stream_0_conv_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2*
(stream_0_conv_3/kernel/Regularizer/mul/x№
&stream_0_conv_3/kernel/Regularizer/mulMul1stream_0_conv_3/kernel/Regularizer/mul/x:output:0/stream_0_conv_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_3/kernel/Regularizer/mulў
&stream_0_conv_3/kernel/Regularizer/addAddV21stream_0_conv_3/kernel/Regularizer/Const:output:0*stream_0_conv_3/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&stream_0_conv_3/kernel/Regularizer/add–
8stream_0_conv_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpstream_0_conv_3_3294011*"
_output_shapes
:@@*
dtype02:
8stream_0_conv_3/kernel/Regularizer/Square/ReadVariableOpѕ
)stream_0_conv_3/kernel/Regularizer/SquareSquare@stream_0_conv_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@@2+
)stream_0_conv_3/kernel/Regularizer/Square≠
*stream_0_conv_3/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_0_conv_3/kernel/Regularizer/Const_2а
(stream_0_conv_3/kernel/Regularizer/Sum_1Sum-stream_0_conv_3/kernel/Regularizer/Square:y:03stream_0_conv_3/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2*
(stream_0_conv_3/kernel/Regularizer/Sum_1Э
*stream_0_conv_3/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2,
*stream_0_conv_3/kernel/Regularizer/mul_1/xд
(stream_0_conv_3/kernel/Regularizer/mul_1Mul3stream_0_conv_3/kernel/Regularizer/mul_1/x:output:01stream_0_conv_3/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2*
(stream_0_conv_3/kernel/Regularizer/mul_1Ў
(stream_0_conv_3/kernel/Regularizer/add_1AddV2*stream_0_conv_3/kernel/Regularizer/add:z:0,stream_0_conv_3/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2*
(stream_0_conv_3/kernel/Regularizer/add_1Й
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dense_1/kernel/Regularizer/ConstЃ
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_1_3294030*
_output_shapes

:@@*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOpІ
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:@@2 
dense_1/kernel/Regularizer/AbsЩ
"dense_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_1/kernel/Regularizer/Const_1є
dense_1/kernel/Regularizer/SumSum"dense_1/kernel/Regularizer/Abs:y:0+dense_1/kernel/Regularizer/Const_1:output:0*
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
dense_1/kernel/Regularizer/mulє
dense_1/kernel/Regularizer/addAddV2)dense_1/kernel/Regularizer/Const:output:0"dense_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/addі
0dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_1_3294030*
_output_shapes

:@@*
dtype022
0dense_1/kernel/Regularizer/Square/ReadVariableOp≥
!dense_1/kernel/Regularizer/SquareSquare8dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@@2#
!dense_1/kernel/Regularizer/SquareЩ
"dense_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_1/kernel/Regularizer/Const_2ј
 dense_1/kernel/Regularizer/Sum_1Sum%dense_1/kernel/Regularizer/Square:y:0+dense_1/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2"
 dense_1/kernel/Regularizer/Sum_1Н
"dense_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2$
"dense_1/kernel/Regularizer/mul_1/xƒ
 dense_1/kernel/Regularizer/mul_1Mul+dense_1/kernel/Regularizer/mul_1/x:output:0)dense_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2"
 dense_1/kernel/Regularizer/mul_1Є
 dense_1/kernel/Regularizer/add_1AddV2"dense_1/kernel/Regularizer/add:z:0$dense_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2"
 dense_1/kernel/Regularizer/add_1Ж
IdentityIdentity+dense_activation_1/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@2

Identityи
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall.^dense_1/kernel/Regularizer/Abs/ReadVariableOp1^dense_1/kernel/Regularizer/Square/ReadVariableOp(^stream_0_conv_1/StatefulPartitionedCall6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp9^stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOp(^stream_0_conv_2/StatefulPartitionedCall6^stream_0_conv_2/kernel/Regularizer/Abs/ReadVariableOp9^stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp(^stream_0_conv_3/StatefulPartitionedCall6^stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp9^stream_0_conv_3/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:€€€€€€€€€}: : : : : : : : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2^
-dense_1/kernel/Regularizer/Abs/ReadVariableOp-dense_1/kernel/Regularizer/Abs/ReadVariableOp2d
0dense_1/kernel/Regularizer/Square/ReadVariableOp0dense_1/kernel/Regularizer/Square/ReadVariableOp2R
'stream_0_conv_1/StatefulPartitionedCall'stream_0_conv_1/StatefulPartitionedCall2n
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp2t
8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOp8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOp2R
'stream_0_conv_2/StatefulPartitionedCall'stream_0_conv_2/StatefulPartitionedCall2n
5stream_0_conv_2/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_2/kernel/Regularizer/Abs/ReadVariableOp2t
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp2R
'stream_0_conv_3/StatefulPartitionedCall'stream_0_conv_3/StatefulPartitionedCall2n
5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp2t
8stream_0_conv_3/kernel/Regularizer/Square/ReadVariableOp8stream_0_conv_3/kernel/Regularizer/Square/ReadVariableOp:U Q
+
_output_shapes
:€€€€€€€€€}
"
_user_specified_name
inputs_0
Є
±
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_3292318

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
k
O__inference_stream_0_maxpool_1_layer_call_and_return_conditional_losses_3292278

inputs
identityb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dimУ

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2

ExpandDims±
MaxPoolMaxPoolExpandDims:output:0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*
ksize
*
paddingVALID*
strides
2	
MaxPoolО
SqueezeSqueezeMaxPool:output:0*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€*
squeeze_dims
2	
Squeezez
IdentityIdentitySqueeze:output:0*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€:e a
=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
Й
j
L__inference_stream_0_drop_3_layer_call_and_return_conditional_losses_3298194

inputs

identity_1^
IdentityIdentityinputs*
T0*+
_output_shapes
:€€€€€€€€€@2

Identitym

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:€€€€€€€€€@2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€@:S O
+
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
я
M
1__inference_stream_0_drop_1_layer_call_fn_3297630

inputs
identity—
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€>@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_stream_0_drop_1_layer_call_and_return_conditional_losses_32929602
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:€€€€€€€€€>@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€>@:S O
+
_output_shapes
:€€€€€€€€€>@
 
_user_specified_nameinputs
Џ
“
7__inference_batch_normalization_3_layer_call_fn_3298322

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityИҐStatefulPartitionedCallЭ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *[
fVRT
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_32927822
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
А+
й
P__inference_batch_normalization_layer_call_and_return_conditional_losses_3297589

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
Ј+
й
P__inference_batch_normalization_layer_call_and_return_conditional_losses_3292188

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
®
е
+__inference_basemodel_layer_call_fn_3296373
inputs_0
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:@
	unknown_5:@@
	unknown_6:@
	unknown_7:@
	unknown_8:@
	unknown_9:@

unknown_10:@ 

unknown_11:@@

unknown_12:@

unknown_13:@

unknown_14:@

unknown_15:@

unknown_16:@

unknown_17:@@

unknown_18:@

unknown_19:@

unknown_20:@

unknown_21:@

unknown_22:@
identityИҐStatefulPartitionedCall¶
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
:€€€€€€€€€@*:
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_basemodel_layer_call_and_return_conditional_losses_32944342
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:€€€€€€€€€}: : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
+
_output_shapes
:€€€€€€€€€}
"
_user_specified_name
inputs/0
№љ
’
F__inference_basemodel_layer_call_and_return_conditional_losses_3297076
inputs_0Q
;stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource:@=
/stream_0_conv_1_biasadd_readvariableop_resource:@C
5batch_normalization_batchnorm_readvariableop_resource:@G
9batch_normalization_batchnorm_mul_readvariableop_resource:@E
7batch_normalization_batchnorm_readvariableop_1_resource:@E
7batch_normalization_batchnorm_readvariableop_2_resource:@Q
;stream_0_conv_2_conv1d_expanddims_1_readvariableop_resource:@@=
/stream_0_conv_2_biasadd_readvariableop_resource:@E
7batch_normalization_1_batchnorm_readvariableop_resource:@I
;batch_normalization_1_batchnorm_mul_readvariableop_resource:@G
9batch_normalization_1_batchnorm_readvariableop_1_resource:@G
9batch_normalization_1_batchnorm_readvariableop_2_resource:@Q
;stream_0_conv_3_conv1d_expanddims_1_readvariableop_resource:@@=
/stream_0_conv_3_biasadd_readvariableop_resource:@E
7batch_normalization_2_batchnorm_readvariableop_resource:@I
;batch_normalization_2_batchnorm_mul_readvariableop_resource:@G
9batch_normalization_2_batchnorm_readvariableop_1_resource:@G
9batch_normalization_2_batchnorm_readvariableop_2_resource:@8
&dense_1_matmul_readvariableop_resource:@@5
'dense_1_biasadd_readvariableop_resource:@E
7batch_normalization_3_batchnorm_readvariableop_resource:@I
;batch_normalization_3_batchnorm_mul_readvariableop_resource:@G
9batch_normalization_3_batchnorm_readvariableop_1_resource:@G
9batch_normalization_3_batchnorm_readvariableop_2_resource:@
identityИҐ,batch_normalization/batchnorm/ReadVariableOpҐ.batch_normalization/batchnorm/ReadVariableOp_1Ґ.batch_normalization/batchnorm/ReadVariableOp_2Ґ0batch_normalization/batchnorm/mul/ReadVariableOpҐ.batch_normalization_1/batchnorm/ReadVariableOpҐ0batch_normalization_1/batchnorm/ReadVariableOp_1Ґ0batch_normalization_1/batchnorm/ReadVariableOp_2Ґ2batch_normalization_1/batchnorm/mul/ReadVariableOpҐ.batch_normalization_2/batchnorm/ReadVariableOpҐ0batch_normalization_2/batchnorm/ReadVariableOp_1Ґ0batch_normalization_2/batchnorm/ReadVariableOp_2Ґ2batch_normalization_2/batchnorm/mul/ReadVariableOpҐ.batch_normalization_3/batchnorm/ReadVariableOpҐ0batch_normalization_3/batchnorm/ReadVariableOp_1Ґ0batch_normalization_3/batchnorm/ReadVariableOp_2Ґ2batch_normalization_3/batchnorm/mul/ReadVariableOpҐdense_1/BiasAdd/ReadVariableOpҐdense_1/MatMul/ReadVariableOpҐ-dense_1/kernel/Regularizer/Abs/ReadVariableOpҐ0dense_1/kernel/Regularizer/Square/ReadVariableOpҐ&stream_0_conv_1/BiasAdd/ReadVariableOpҐ2stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOpҐ5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpҐ8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOpҐ&stream_0_conv_2/BiasAdd/ReadVariableOpҐ2stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOpҐ5stream_0_conv_2/kernel/Regularizer/Abs/ReadVariableOpҐ8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOpҐ&stream_0_conv_3/BiasAdd/ReadVariableOpҐ2stream_0_conv_3/conv1d/ExpandDims_1/ReadVariableOpҐ5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOpҐ8stream_0_conv_3/kernel/Regularizer/Square/ReadVariableOpИ
stream_0_input_drop/IdentityIdentityinputs_0*
T0*+
_output_shapes
:€€€€€€€€€}2
stream_0_input_drop/IdentityЩ
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
stream_0_conv_1/BiasAddќ
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
#batch_normalization/batchnorm/add_1Й
activation/TanhTanh'batch_normalization/batchnorm/add_1:z:0*
T0*+
_output_shapes
:€€€€€€€€€}@2
activation/TanhИ
!stream_0_maxpool_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!stream_0_maxpool_1/ExpandDims/dim«
stream_0_maxpool_1/ExpandDims
ExpandDimsactivation/Tanh:y:0*stream_0_maxpool_1/ExpandDims/dim:output:0*
T0*/
_output_shapes
:€€€€€€€€€}@2
stream_0_maxpool_1/ExpandDimsЎ
stream_0_maxpool_1/MaxPoolMaxPool&stream_0_maxpool_1/ExpandDims:output:0*/
_output_shapes
:€€€€€€€€€>@*
ksize
*
paddingVALID*
strides
2
stream_0_maxpool_1/MaxPoolµ
stream_0_maxpool_1/SqueezeSqueeze#stream_0_maxpool_1/MaxPool:output:0*
T0*+
_output_shapes
:€€€€€€€€€>@*
squeeze_dims
2
stream_0_maxpool_1/SqueezeЫ
stream_0_drop_1/IdentityIdentity#stream_0_maxpool_1/Squeeze:output:0*
T0*+
_output_shapes
:€€€€€€€€€>@2
stream_0_drop_1/IdentityЩ
%stream_0_conv_2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€2'
%stream_0_conv_2/conv1d/ExpandDims/dimб
!stream_0_conv_2/conv1d/ExpandDims
ExpandDims!stream_0_drop_1/Identity:output:0.stream_0_conv_2/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:€€€€€€€€€>@2#
!stream_0_conv_2/conv1d/ExpandDimsи
2stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp;stream_0_conv_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype024
2stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOpФ
'stream_0_conv_2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2)
'stream_0_conv_2/conv1d/ExpandDims_1/dimч
#stream_0_conv_2/conv1d/ExpandDims_1
ExpandDims:stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOp:value:00stream_0_conv_2/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@2%
#stream_0_conv_2/conv1d/ExpandDims_1ц
stream_0_conv_2/conv1dConv2D*stream_0_conv_2/conv1d/ExpandDims:output:0,stream_0_conv_2/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:€€€€€€€€€>@*
paddingSAME*
strides
2
stream_0_conv_2/conv1d¬
stream_0_conv_2/conv1d/SqueezeSqueezestream_0_conv_2/conv1d:output:0*
T0*+
_output_shapes
:€€€€€€€€€>@*
squeeze_dims

э€€€€€€€€2 
stream_0_conv_2/conv1d/SqueezeЉ
&stream_0_conv_2/BiasAdd/ReadVariableOpReadVariableOp/stream_0_conv_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02(
&stream_0_conv_2/BiasAdd/ReadVariableOpћ
stream_0_conv_2/BiasAddBiasAdd'stream_0_conv_2/conv1d/Squeeze:output:0.stream_0_conv_2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€>@2
stream_0_conv_2/BiasAdd‘
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
%batch_normalization_1/batchnorm/mul_1Mul stream_0_conv_2/BiasAdd:output:0'batch_normalization_1/batchnorm/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€>@2'
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
:€€€€€€€€€>@2'
%batch_normalization_1/batchnorm/add_1П
activation_1/TanhTanh)batch_normalization_1/batchnorm/add_1:z:0*
T0*+
_output_shapes
:€€€€€€€€€>@2
activation_1/TanhИ
!stream_0_maxpool_2/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!stream_0_maxpool_2/ExpandDims/dim…
stream_0_maxpool_2/ExpandDims
ExpandDimsactivation_1/Tanh:y:0*stream_0_maxpool_2/ExpandDims/dim:output:0*
T0*/
_output_shapes
:€€€€€€€€€>@2
stream_0_maxpool_2/ExpandDimsЎ
stream_0_maxpool_2/MaxPoolMaxPool&stream_0_maxpool_2/ExpandDims:output:0*/
_output_shapes
:€€€€€€€€€@*
ksize
*
paddingVALID*
strides
2
stream_0_maxpool_2/MaxPoolµ
stream_0_maxpool_2/SqueezeSqueeze#stream_0_maxpool_2/MaxPool:output:0*
T0*+
_output_shapes
:€€€€€€€€€@*
squeeze_dims
2
stream_0_maxpool_2/SqueezeЫ
stream_0_drop_2/IdentityIdentity#stream_0_maxpool_2/Squeeze:output:0*
T0*+
_output_shapes
:€€€€€€€€€@2
stream_0_drop_2/IdentityЩ
%stream_0_conv_3/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€2'
%stream_0_conv_3/conv1d/ExpandDims/dimб
!stream_0_conv_3/conv1d/ExpandDims
ExpandDims!stream_0_drop_2/Identity:output:0.stream_0_conv_3/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:€€€€€€€€€@2#
!stream_0_conv_3/conv1d/ExpandDimsи
2stream_0_conv_3/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp;stream_0_conv_3_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype024
2stream_0_conv_3/conv1d/ExpandDims_1/ReadVariableOpФ
'stream_0_conv_3/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2)
'stream_0_conv_3/conv1d/ExpandDims_1/dimч
#stream_0_conv_3/conv1d/ExpandDims_1
ExpandDims:stream_0_conv_3/conv1d/ExpandDims_1/ReadVariableOp:value:00stream_0_conv_3/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@2%
#stream_0_conv_3/conv1d/ExpandDims_1ц
stream_0_conv_3/conv1dConv2D*stream_0_conv_3/conv1d/ExpandDims:output:0,stream_0_conv_3/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:€€€€€€€€€@*
paddingSAME*
strides
2
stream_0_conv_3/conv1d¬
stream_0_conv_3/conv1d/SqueezeSqueezestream_0_conv_3/conv1d:output:0*
T0*+
_output_shapes
:€€€€€€€€€@*
squeeze_dims

э€€€€€€€€2 
stream_0_conv_3/conv1d/SqueezeЉ
&stream_0_conv_3/BiasAdd/ReadVariableOpReadVariableOp/stream_0_conv_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02(
&stream_0_conv_3/BiasAdd/ReadVariableOpћ
stream_0_conv_3/BiasAddBiasAdd'stream_0_conv_3/conv1d/Squeeze:output:0.stream_0_conv_3/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€@2
stream_0_conv_3/BiasAdd‘
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
%batch_normalization_2/batchnorm/mul_1Mul stream_0_conv_3/BiasAdd:output:0'batch_normalization_2/batchnorm/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€@2'
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
:€€€€€€€€€@2'
%batch_normalization_2/batchnorm/add_1П
activation_2/TanhTanh)batch_normalization_2/batchnorm/add_1:z:0*
T0*+
_output_shapes
:€€€€€€€€€@2
activation_2/TanhИ
!stream_0_maxpool_3/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!stream_0_maxpool_3/ExpandDims/dim…
stream_0_maxpool_3/ExpandDims
ExpandDimsactivation_2/Tanh:y:0*stream_0_maxpool_3/ExpandDims/dim:output:0*
T0*/
_output_shapes
:€€€€€€€€€@2
stream_0_maxpool_3/ExpandDimsЎ
stream_0_maxpool_3/MaxPoolMaxPool&stream_0_maxpool_3/ExpandDims:output:0*/
_output_shapes
:€€€€€€€€€@*
ksize
*
paddingVALID*
strides
2
stream_0_maxpool_3/MaxPoolµ
stream_0_maxpool_3/SqueezeSqueeze#stream_0_maxpool_3/MaxPool:output:0*
T0*+
_output_shapes
:€€€€€€€€€@*
squeeze_dims
2
stream_0_maxpool_3/SqueezeЫ
stream_0_drop_3/IdentityIdentity#stream_0_maxpool_3/Squeeze:output:0*
T0*+
_output_shapes
:€€€€€€€€€@2
stream_0_drop_3/Identity§
/global_average_pooling1d/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :21
/global_average_pooling1d/Mean/reduction_indices’
global_average_pooling1d/MeanMean!stream_0_drop_3/Identity:output:08global_average_pooling1d/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
global_average_pooling1d/MeanЪ
dense_1_dropout/IdentityIdentity&global_average_pooling1d/Mean:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
dense_1_dropout/Identity•
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02
dense_1/MatMul/ReadVariableOp¶
dense_1/MatMulMatMul!dense_1_dropout/Identity:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2
dense_1/MatMul§
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02 
dense_1/BiasAdd/ReadVariableOp°
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2
dense_1/BiasAdd‘
.batch_normalization_3/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_3_batchnorm_readvariableop_resource*
_output_shapes
:@*
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
:@2%
#batch_normalization_3/batchnorm/add•
%batch_normalization_3/batchnorm/RsqrtRsqrt'batch_normalization_3/batchnorm/add:z:0*
T0*
_output_shapes
:@2'
%batch_normalization_3/batchnorm/Rsqrtа
2batch_normalization_3/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_3_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype024
2batch_normalization_3/batchnorm/mul/ReadVariableOpЁ
#batch_normalization_3/batchnorm/mulMul)batch_normalization_3/batchnorm/Rsqrt:y:0:batch_normalization_3/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2%
#batch_normalization_3/batchnorm/mul 
%batch_normalization_3/batchnorm/mul_1Muldense_1/BiasAdd:output:0'batch_normalization_3/batchnorm/mul:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2'
%batch_normalization_3/batchnorm/mul_1Џ
0batch_normalization_3/batchnorm/ReadVariableOp_1ReadVariableOp9batch_normalization_3_batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype022
0batch_normalization_3/batchnorm/ReadVariableOp_1Ё
%batch_normalization_3/batchnorm/mul_2Mul8batch_normalization_3/batchnorm/ReadVariableOp_1:value:0'batch_normalization_3/batchnorm/mul:z:0*
T0*
_output_shapes
:@2'
%batch_normalization_3/batchnorm/mul_2Џ
0batch_normalization_3/batchnorm/ReadVariableOp_2ReadVariableOp9batch_normalization_3_batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype022
0batch_normalization_3/batchnorm/ReadVariableOp_2џ
#batch_normalization_3/batchnorm/subSub8batch_normalization_3/batchnorm/ReadVariableOp_2:value:0)batch_normalization_3/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2%
#batch_normalization_3/batchnorm/subЁ
%batch_normalization_3/batchnorm/add_1AddV2)batch_normalization_3/batchnorm/mul_1:z:0'batch_normalization_3/batchnorm/sub:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2'
%batch_normalization_3/batchnorm/add_1Щ
(stream_0_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(stream_0_conv_1/kernel/Regularizer/Constо
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp;stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype027
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp√
&stream_0_conv_1/kernel/Regularizer/AbsAbs=stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_0_conv_1/kernel/Regularizer/Abs≠
*stream_0_conv_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_0_conv_1/kernel/Regularizer/Const_1ў
&stream_0_conv_1/kernel/Regularizer/SumSum*stream_0_conv_1/kernel/Regularizer/Abs:y:03stream_0_conv_1/kernel/Regularizer/Const_1:output:0*
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
&stream_0_conv_1/kernel/Regularizer/mulў
&stream_0_conv_1/kernel/Regularizer/addAddV21stream_0_conv_1/kernel/Regularizer/Const:output:0*stream_0_conv_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/addф
8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02:
8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOpѕ
)stream_0_conv_1/kernel/Regularizer/SquareSquare@stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2+
)stream_0_conv_1/kernel/Regularizer/Square≠
*stream_0_conv_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_0_conv_1/kernel/Regularizer/Const_2а
(stream_0_conv_1/kernel/Regularizer/Sum_1Sum-stream_0_conv_1/kernel/Regularizer/Square:y:03stream_0_conv_1/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2*
(stream_0_conv_1/kernel/Regularizer/Sum_1Э
*stream_0_conv_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2,
*stream_0_conv_1/kernel/Regularizer/mul_1/xд
(stream_0_conv_1/kernel/Regularizer/mul_1Mul3stream_0_conv_1/kernel/Regularizer/mul_1/x:output:01stream_0_conv_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2*
(stream_0_conv_1/kernel/Regularizer/mul_1Ў
(stream_0_conv_1/kernel/Regularizer/add_1AddV2*stream_0_conv_1/kernel/Regularizer/add:z:0,stream_0_conv_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2*
(stream_0_conv_1/kernel/Regularizer/add_1Щ
(stream_0_conv_2/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(stream_0_conv_2/kernel/Regularizer/Constо
5stream_0_conv_2/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp;stream_0_conv_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype027
5stream_0_conv_2/kernel/Regularizer/Abs/ReadVariableOp√
&stream_0_conv_2/kernel/Regularizer/AbsAbs=stream_0_conv_2/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@@2(
&stream_0_conv_2/kernel/Regularizer/Abs≠
*stream_0_conv_2/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_0_conv_2/kernel/Regularizer/Const_1ў
&stream_0_conv_2/kernel/Regularizer/SumSum*stream_0_conv_2/kernel/Regularizer/Abs:y:03stream_0_conv_2/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_2/kernel/Regularizer/SumЩ
(stream_0_conv_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2*
(stream_0_conv_2/kernel/Regularizer/mul/x№
&stream_0_conv_2/kernel/Regularizer/mulMul1stream_0_conv_2/kernel/Regularizer/mul/x:output:0/stream_0_conv_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_2/kernel/Regularizer/mulў
&stream_0_conv_2/kernel/Regularizer/addAddV21stream_0_conv_2/kernel/Regularizer/Const:output:0*stream_0_conv_2/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&stream_0_conv_2/kernel/Regularizer/addф
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;stream_0_conv_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype02:
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOpѕ
)stream_0_conv_2/kernel/Regularizer/SquareSquare@stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@@2+
)stream_0_conv_2/kernel/Regularizer/Square≠
*stream_0_conv_2/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_0_conv_2/kernel/Regularizer/Const_2а
(stream_0_conv_2/kernel/Regularizer/Sum_1Sum-stream_0_conv_2/kernel/Regularizer/Square:y:03stream_0_conv_2/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2*
(stream_0_conv_2/kernel/Regularizer/Sum_1Э
*stream_0_conv_2/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2,
*stream_0_conv_2/kernel/Regularizer/mul_1/xд
(stream_0_conv_2/kernel/Regularizer/mul_1Mul3stream_0_conv_2/kernel/Regularizer/mul_1/x:output:01stream_0_conv_2/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2*
(stream_0_conv_2/kernel/Regularizer/mul_1Ў
(stream_0_conv_2/kernel/Regularizer/add_1AddV2*stream_0_conv_2/kernel/Regularizer/add:z:0,stream_0_conv_2/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2*
(stream_0_conv_2/kernel/Regularizer/add_1Щ
(stream_0_conv_3/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(stream_0_conv_3/kernel/Regularizer/Constо
5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp;stream_0_conv_3_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype027
5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp√
&stream_0_conv_3/kernel/Regularizer/AbsAbs=stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@@2(
&stream_0_conv_3/kernel/Regularizer/Abs≠
*stream_0_conv_3/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_0_conv_3/kernel/Regularizer/Const_1ў
&stream_0_conv_3/kernel/Regularizer/SumSum*stream_0_conv_3/kernel/Regularizer/Abs:y:03stream_0_conv_3/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_3/kernel/Regularizer/SumЩ
(stream_0_conv_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2*
(stream_0_conv_3/kernel/Regularizer/mul/x№
&stream_0_conv_3/kernel/Regularizer/mulMul1stream_0_conv_3/kernel/Regularizer/mul/x:output:0/stream_0_conv_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_3/kernel/Regularizer/mulў
&stream_0_conv_3/kernel/Regularizer/addAddV21stream_0_conv_3/kernel/Regularizer/Const:output:0*stream_0_conv_3/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&stream_0_conv_3/kernel/Regularizer/addф
8stream_0_conv_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;stream_0_conv_3_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype02:
8stream_0_conv_3/kernel/Regularizer/Square/ReadVariableOpѕ
)stream_0_conv_3/kernel/Regularizer/SquareSquare@stream_0_conv_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@@2+
)stream_0_conv_3/kernel/Regularizer/Square≠
*stream_0_conv_3/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_0_conv_3/kernel/Regularizer/Const_2а
(stream_0_conv_3/kernel/Regularizer/Sum_1Sum-stream_0_conv_3/kernel/Regularizer/Square:y:03stream_0_conv_3/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2*
(stream_0_conv_3/kernel/Regularizer/Sum_1Э
*stream_0_conv_3/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2,
*stream_0_conv_3/kernel/Regularizer/mul_1/xд
(stream_0_conv_3/kernel/Regularizer/mul_1Mul3stream_0_conv_3/kernel/Regularizer/mul_1/x:output:01stream_0_conv_3/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2*
(stream_0_conv_3/kernel/Regularizer/mul_1Ў
(stream_0_conv_3/kernel/Regularizer/add_1AddV2*stream_0_conv_3/kernel/Regularizer/add:z:0,stream_0_conv_3/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2*
(stream_0_conv_3/kernel/Regularizer/add_1Й
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dense_1/kernel/Regularizer/Const≈
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOpІ
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:@@2 
dense_1/kernel/Regularizer/AbsЩ
"dense_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_1/kernel/Regularizer/Const_1є
dense_1/kernel/Regularizer/SumSum"dense_1/kernel/Regularizer/Abs:y:0+dense_1/kernel/Regularizer/Const_1:output:0*
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
dense_1/kernel/Regularizer/mulє
dense_1/kernel/Regularizer/addAddV2)dense_1/kernel/Regularizer/Const:output:0"dense_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/addЋ
0dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype022
0dense_1/kernel/Regularizer/Square/ReadVariableOp≥
!dense_1/kernel/Regularizer/SquareSquare8dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@@2#
!dense_1/kernel/Regularizer/SquareЩ
"dense_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_1/kernel/Regularizer/Const_2ј
 dense_1/kernel/Regularizer/Sum_1Sum%dense_1/kernel/Regularizer/Square:y:0+dense_1/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2"
 dense_1/kernel/Regularizer/Sum_1Н
"dense_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2$
"dense_1/kernel/Regularizer/mul_1/xƒ
 dense_1/kernel/Regularizer/mul_1Mul+dense_1/kernel/Regularizer/mul_1/x:output:0)dense_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2"
 dense_1/kernel/Regularizer/mul_1Є
 dense_1/kernel/Regularizer/add_1AddV2"dense_1/kernel/Regularizer/add:z:0$dense_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2"
 dense_1/kernel/Regularizer/add_1Д
IdentityIdentity)batch_normalization_3/batchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@2

IdentityН
NoOpNoOp-^batch_normalization/batchnorm/ReadVariableOp/^batch_normalization/batchnorm/ReadVariableOp_1/^batch_normalization/batchnorm/ReadVariableOp_21^batch_normalization/batchnorm/mul/ReadVariableOp/^batch_normalization_1/batchnorm/ReadVariableOp1^batch_normalization_1/batchnorm/ReadVariableOp_11^batch_normalization_1/batchnorm/ReadVariableOp_23^batch_normalization_1/batchnorm/mul/ReadVariableOp/^batch_normalization_2/batchnorm/ReadVariableOp1^batch_normalization_2/batchnorm/ReadVariableOp_11^batch_normalization_2/batchnorm/ReadVariableOp_23^batch_normalization_2/batchnorm/mul/ReadVariableOp/^batch_normalization_3/batchnorm/ReadVariableOp1^batch_normalization_3/batchnorm/ReadVariableOp_11^batch_normalization_3/batchnorm/ReadVariableOp_23^batch_normalization_3/batchnorm/mul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp.^dense_1/kernel/Regularizer/Abs/ReadVariableOp1^dense_1/kernel/Regularizer/Square/ReadVariableOp'^stream_0_conv_1/BiasAdd/ReadVariableOp3^stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp9^stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOp'^stream_0_conv_2/BiasAdd/ReadVariableOp3^stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOp6^stream_0_conv_2/kernel/Regularizer/Abs/ReadVariableOp9^stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp'^stream_0_conv_3/BiasAdd/ReadVariableOp3^stream_0_conv_3/conv1d/ExpandDims_1/ReadVariableOp6^stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp9^stream_0_conv_3/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:€€€€€€€€€}: : : : : : : : : : : : : : : : : : : : : : : : 2\
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
-dense_1/kernel/Regularizer/Abs/ReadVariableOp-dense_1/kernel/Regularizer/Abs/ReadVariableOp2d
0dense_1/kernel/Regularizer/Square/ReadVariableOp0dense_1/kernel/Regularizer/Square/ReadVariableOp2P
&stream_0_conv_1/BiasAdd/ReadVariableOp&stream_0_conv_1/BiasAdd/ReadVariableOp2h
2stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp2stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp2n
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp2t
8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOp8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOp2P
&stream_0_conv_2/BiasAdd/ReadVariableOp&stream_0_conv_2/BiasAdd/ReadVariableOp2h
2stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOp2stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOp2n
5stream_0_conv_2/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_2/kernel/Regularizer/Abs/ReadVariableOp2t
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp2P
&stream_0_conv_3/BiasAdd/ReadVariableOp&stream_0_conv_3/BiasAdd/ReadVariableOp2h
2stream_0_conv_3/conv1d/ExpandDims_1/ReadVariableOp2stream_0_conv_3/conv1d/ExpandDims_1/ReadVariableOp2n
5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp2t
8stream_0_conv_3/kernel/Regularizer/Square/ReadVariableOp8stream_0_conv_3/kernel/Regularizer/Square/ReadVariableOp:U Q
+
_output_shapes
:€€€€€€€€€}
"
_user_specified_name
inputs/0
К
±
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_3298109

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
:€€€€€€€€€@2
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
:€€€€€€€€€@2
batchnorm/add_1r
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€@2

Identity¬
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€@: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:S O
+
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
ґ
ѓ
P__inference_batch_normalization_layer_call_and_return_conditional_losses_3297501

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
т
o
P__inference_stream_0_input_drop_layer_call_and_return_conditional_losses_3297375

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
К
±
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_3297832

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
:€€€€€€€€€>@2
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
:€€€€€€€€€>@2
batchnorm/add_1r
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€>@2

Identity¬
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€>@: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:S O
+
_output_shapes
:€€€€€€€€€>@
 
_user_specified_nameinputs
•
ж
)__inference_model_1_layer_call_fn_3295246
left_inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:@
	unknown_5:@@
	unknown_6:@
	unknown_7:@
	unknown_8:@
	unknown_9:@

unknown_10:@ 

unknown_11:@@

unknown_12:@

unknown_13:@

unknown_14:@

unknown_15:@

unknown_16:@

unknown_17:@@

unknown_18:@

unknown_19:@

unknown_20:@

unknown_21:@

unknown_22:@
identityИҐStatefulPartitionedCallЯ
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
:€€€€€€€€€@*2
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_model_1_layer_call_and_return_conditional_losses_32951422
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:€€€€€€€€€}: : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
+
_output_shapes
:€€€€€€€€€}
%
_user_specified_nameleft_inputs
Е
q
U__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_3298228

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
:€€€€€€€€€@:S O
+
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
В+
л
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_3293429

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
:€€€€€€€€€@2
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
:€€€€€€€€€@2
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
:€€€€€€€€€@2
batchnorm/add_1r
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€@2

Identityт
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€@: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:S O
+
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
Ј+
й
P__inference_batch_normalization_layer_call_and_return_conditional_losses_3297535

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
©
k
O__inference_stream_0_maxpool_2_layer_call_and_return_conditional_losses_3297902

inputs
identityb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dimБ

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*/
_output_shapes
:€€€€€€€€€>@2

ExpandDimsЯ
MaxPoolMaxPoolExpandDims:output:0*/
_output_shapes
:€€€€€€€€€@*
ksize
*
paddingVALID*
strides
2	
MaxPool|
SqueezeSqueezeMaxPool:output:0*
T0*+
_output_shapes
:€€€€€€€€€@*
squeeze_dims
2	
Squeezeh
IdentityIdentitySqueeze:output:0*
T0*+
_output_shapes
:€€€€€€€€€@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€>@:S O
+
_output_shapes
:€€€€€€€€€>@
 
_user_specified_nameinputs
И
h
L__inference_dense_1_dropout_layer_call_and_return_conditional_losses_3293338

inputs
identityZ
IdentityIdentityinputs*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:€€€€€€€€€@:O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
љ 
Л
F__inference_basemodel_layer_call_and_return_conditional_losses_3293259

inputs-
stream_0_conv_1_3292905:@%
stream_0_conv_1_3292907:@)
batch_normalization_3292930:@)
batch_normalization_3292932:@)
batch_normalization_3292934:@)
batch_normalization_3292936:@-
stream_0_conv_2_3292993:@@%
stream_0_conv_2_3292995:@+
batch_normalization_1_3293018:@+
batch_normalization_1_3293020:@+
batch_normalization_1_3293022:@+
batch_normalization_1_3293024:@-
stream_0_conv_3_3293081:@@%
stream_0_conv_3_3293083:@+
batch_normalization_2_3293106:@+
batch_normalization_2_3293108:@+
batch_normalization_2_3293110:@+
batch_normalization_2_3293112:@!
dense_1_3293178:@@
dense_1_3293180:@+
batch_normalization_3_3293183:@+
batch_normalization_3_3293185:@+
batch_normalization_3_3293187:@+
batch_normalization_3_3293189:@
identityИҐ+batch_normalization/StatefulPartitionedCallҐ-batch_normalization_1/StatefulPartitionedCallҐ-batch_normalization_2/StatefulPartitionedCallҐ-batch_normalization_3/StatefulPartitionedCallҐdense_1/StatefulPartitionedCallҐ-dense_1/kernel/Regularizer/Abs/ReadVariableOpҐ0dense_1/kernel/Regularizer/Square/ReadVariableOpҐ'stream_0_conv_1/StatefulPartitionedCallҐ5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpҐ8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOpҐ'stream_0_conv_2/StatefulPartitionedCallҐ5stream_0_conv_2/kernel/Regularizer/Abs/ReadVariableOpҐ8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOpҐ'stream_0_conv_3/StatefulPartitionedCallҐ5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOpҐ8stream_0_conv_3/kernel/Regularizer/Square/ReadVariableOpэ
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
GPU2*0J 8В *Y
fTRR
P__inference_stream_0_input_drop_layer_call_and_return_conditional_losses_32928722%
#stream_0_input_drop/PartitionedCallз
'stream_0_conv_1/StatefulPartitionedCallStatefulPartitionedCall,stream_0_input_drop/PartitionedCall:output:0stream_0_conv_1_3292905stream_0_conv_1_3292907*
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
GPU2*0J 8В *U
fPRN
L__inference_stream_0_conv_1_layer_call_and_return_conditional_losses_32929042)
'stream_0_conv_1/StatefulPartitionedCallљ
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall0stream_0_conv_1/StatefulPartitionedCall:output:0batch_normalization_3292930batch_normalization_3292932batch_normalization_3292934batch_normalization_3292936*
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
GPU2*0J 8В *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_32929292-
+batch_normalization/StatefulPartitionedCallР
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
GPU2*0J 8В *P
fKRI
G__inference_activation_layer_call_and_return_conditional_losses_32929442
activation/PartitionedCallЧ
"stream_0_maxpool_1/PartitionedCallPartitionedCall#activation/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€>@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_stream_0_maxpool_1_layer_call_and_return_conditional_losses_32929532$
"stream_0_maxpool_1/PartitionedCallЦ
stream_0_drop_1/PartitionedCallPartitionedCall+stream_0_maxpool_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€>@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_stream_0_drop_1_layer_call_and_return_conditional_losses_32929602!
stream_0_drop_1/PartitionedCallг
'stream_0_conv_2/StatefulPartitionedCallStatefulPartitionedCall(stream_0_drop_1/PartitionedCall:output:0stream_0_conv_2_3292993stream_0_conv_2_3292995*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€>@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_stream_0_conv_2_layer_call_and_return_conditional_losses_32929922)
'stream_0_conv_2/StatefulPartitionedCallЋ
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall0stream_0_conv_2/StatefulPartitionedCall:output:0batch_normalization_1_3293018batch_normalization_1_3293020batch_normalization_1_3293022batch_normalization_1_3293024*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€>@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *[
fVRT
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_32930172/
-batch_normalization_1/StatefulPartitionedCallШ
activation_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€>@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_activation_1_layer_call_and_return_conditional_losses_32930322
activation_1/PartitionedCallЩ
"stream_0_maxpool_2/PartitionedCallPartitionedCall%activation_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_stream_0_maxpool_2_layer_call_and_return_conditional_losses_32930412$
"stream_0_maxpool_2/PartitionedCallЦ
stream_0_drop_2/PartitionedCallPartitionedCall+stream_0_maxpool_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_stream_0_drop_2_layer_call_and_return_conditional_losses_32930482!
stream_0_drop_2/PartitionedCallг
'stream_0_conv_3/StatefulPartitionedCallStatefulPartitionedCall(stream_0_drop_2/PartitionedCall:output:0stream_0_conv_3_3293081stream_0_conv_3_3293083*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_stream_0_conv_3_layer_call_and_return_conditional_losses_32930802)
'stream_0_conv_3/StatefulPartitionedCallЋ
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall0stream_0_conv_3/StatefulPartitionedCall:output:0batch_normalization_2_3293106batch_normalization_2_3293108batch_normalization_2_3293110batch_normalization_2_3293112*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *[
fVRT
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_32931052/
-batch_normalization_2/StatefulPartitionedCallШ
activation_2/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_activation_2_layer_call_and_return_conditional_losses_32931202
activation_2/PartitionedCallЩ
"stream_0_maxpool_3/PartitionedCallPartitionedCall%activation_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_stream_0_maxpool_3_layer_call_and_return_conditional_losses_32931292$
"stream_0_maxpool_3/PartitionedCallЦ
stream_0_drop_3/PartitionedCallPartitionedCall+stream_0_maxpool_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_stream_0_drop_3_layer_call_and_return_conditional_losses_32931362!
stream_0_drop_3/PartitionedCall™
(global_average_pooling1d/PartitionedCallPartitionedCall(stream_0_drop_3/PartitionedCall:output:0*
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
GPU2*0J 8В *^
fYRW
U__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_32931432*
(global_average_pooling1d/PartitionedCallШ
dense_1_dropout/PartitionedCallPartitionedCall1global_average_pooling1d/PartitionedCall:output:0*
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
GPU2*0J 8В *U
fPRN
L__inference_dense_1_dropout_layer_call_and_return_conditional_losses_32931502!
dense_1_dropout/PartitionedCallЈ
dense_1/StatefulPartitionedCallStatefulPartitionedCall(dense_1_dropout/PartitionedCall:output:0dense_1_3293178dense_1_3293180*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_32931772!
dense_1/StatefulPartitionedCallњ
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0batch_normalization_3_3293183batch_normalization_3_3293185batch_normalization_3_3293187batch_normalization_3_3293189*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *[
fVRT
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_32927222/
-batch_normalization_3/StatefulPartitionedCall¶
"dense_activation_1/PartitionedCallPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0*
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
GPU2*0J 8В *X
fSRQ
O__inference_dense_activation_1_layer_call_and_return_conditional_losses_32931962$
"dense_activation_1/PartitionedCallЩ
(stream_0_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(stream_0_conv_1/kernel/Regularizer/Const 
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpstream_0_conv_1_3292905*"
_output_shapes
:@*
dtype027
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp√
&stream_0_conv_1/kernel/Regularizer/AbsAbs=stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_0_conv_1/kernel/Regularizer/Abs≠
*stream_0_conv_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_0_conv_1/kernel/Regularizer/Const_1ў
&stream_0_conv_1/kernel/Regularizer/SumSum*stream_0_conv_1/kernel/Regularizer/Abs:y:03stream_0_conv_1/kernel/Regularizer/Const_1:output:0*
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
&stream_0_conv_1/kernel/Regularizer/mulў
&stream_0_conv_1/kernel/Regularizer/addAddV21stream_0_conv_1/kernel/Regularizer/Const:output:0*stream_0_conv_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/add–
8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpstream_0_conv_1_3292905*"
_output_shapes
:@*
dtype02:
8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOpѕ
)stream_0_conv_1/kernel/Regularizer/SquareSquare@stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2+
)stream_0_conv_1/kernel/Regularizer/Square≠
*stream_0_conv_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_0_conv_1/kernel/Regularizer/Const_2а
(stream_0_conv_1/kernel/Regularizer/Sum_1Sum-stream_0_conv_1/kernel/Regularizer/Square:y:03stream_0_conv_1/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2*
(stream_0_conv_1/kernel/Regularizer/Sum_1Э
*stream_0_conv_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2,
*stream_0_conv_1/kernel/Regularizer/mul_1/xд
(stream_0_conv_1/kernel/Regularizer/mul_1Mul3stream_0_conv_1/kernel/Regularizer/mul_1/x:output:01stream_0_conv_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2*
(stream_0_conv_1/kernel/Regularizer/mul_1Ў
(stream_0_conv_1/kernel/Regularizer/add_1AddV2*stream_0_conv_1/kernel/Regularizer/add:z:0,stream_0_conv_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2*
(stream_0_conv_1/kernel/Regularizer/add_1Щ
(stream_0_conv_2/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(stream_0_conv_2/kernel/Regularizer/Const 
5stream_0_conv_2/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpstream_0_conv_2_3292993*"
_output_shapes
:@@*
dtype027
5stream_0_conv_2/kernel/Regularizer/Abs/ReadVariableOp√
&stream_0_conv_2/kernel/Regularizer/AbsAbs=stream_0_conv_2/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@@2(
&stream_0_conv_2/kernel/Regularizer/Abs≠
*stream_0_conv_2/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_0_conv_2/kernel/Regularizer/Const_1ў
&stream_0_conv_2/kernel/Regularizer/SumSum*stream_0_conv_2/kernel/Regularizer/Abs:y:03stream_0_conv_2/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_2/kernel/Regularizer/SumЩ
(stream_0_conv_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2*
(stream_0_conv_2/kernel/Regularizer/mul/x№
&stream_0_conv_2/kernel/Regularizer/mulMul1stream_0_conv_2/kernel/Regularizer/mul/x:output:0/stream_0_conv_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_2/kernel/Regularizer/mulў
&stream_0_conv_2/kernel/Regularizer/addAddV21stream_0_conv_2/kernel/Regularizer/Const:output:0*stream_0_conv_2/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&stream_0_conv_2/kernel/Regularizer/add–
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpstream_0_conv_2_3292993*"
_output_shapes
:@@*
dtype02:
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOpѕ
)stream_0_conv_2/kernel/Regularizer/SquareSquare@stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@@2+
)stream_0_conv_2/kernel/Regularizer/Square≠
*stream_0_conv_2/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_0_conv_2/kernel/Regularizer/Const_2а
(stream_0_conv_2/kernel/Regularizer/Sum_1Sum-stream_0_conv_2/kernel/Regularizer/Square:y:03stream_0_conv_2/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2*
(stream_0_conv_2/kernel/Regularizer/Sum_1Э
*stream_0_conv_2/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2,
*stream_0_conv_2/kernel/Regularizer/mul_1/xд
(stream_0_conv_2/kernel/Regularizer/mul_1Mul3stream_0_conv_2/kernel/Regularizer/mul_1/x:output:01stream_0_conv_2/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2*
(stream_0_conv_2/kernel/Regularizer/mul_1Ў
(stream_0_conv_2/kernel/Regularizer/add_1AddV2*stream_0_conv_2/kernel/Regularizer/add:z:0,stream_0_conv_2/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2*
(stream_0_conv_2/kernel/Regularizer/add_1Щ
(stream_0_conv_3/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(stream_0_conv_3/kernel/Regularizer/Const 
5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpstream_0_conv_3_3293081*"
_output_shapes
:@@*
dtype027
5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp√
&stream_0_conv_3/kernel/Regularizer/AbsAbs=stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@@2(
&stream_0_conv_3/kernel/Regularizer/Abs≠
*stream_0_conv_3/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_0_conv_3/kernel/Regularizer/Const_1ў
&stream_0_conv_3/kernel/Regularizer/SumSum*stream_0_conv_3/kernel/Regularizer/Abs:y:03stream_0_conv_3/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_3/kernel/Regularizer/SumЩ
(stream_0_conv_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2*
(stream_0_conv_3/kernel/Regularizer/mul/x№
&stream_0_conv_3/kernel/Regularizer/mulMul1stream_0_conv_3/kernel/Regularizer/mul/x:output:0/stream_0_conv_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_3/kernel/Regularizer/mulў
&stream_0_conv_3/kernel/Regularizer/addAddV21stream_0_conv_3/kernel/Regularizer/Const:output:0*stream_0_conv_3/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&stream_0_conv_3/kernel/Regularizer/add–
8stream_0_conv_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpstream_0_conv_3_3293081*"
_output_shapes
:@@*
dtype02:
8stream_0_conv_3/kernel/Regularizer/Square/ReadVariableOpѕ
)stream_0_conv_3/kernel/Regularizer/SquareSquare@stream_0_conv_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@@2+
)stream_0_conv_3/kernel/Regularizer/Square≠
*stream_0_conv_3/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_0_conv_3/kernel/Regularizer/Const_2а
(stream_0_conv_3/kernel/Regularizer/Sum_1Sum-stream_0_conv_3/kernel/Regularizer/Square:y:03stream_0_conv_3/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2*
(stream_0_conv_3/kernel/Regularizer/Sum_1Э
*stream_0_conv_3/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2,
*stream_0_conv_3/kernel/Regularizer/mul_1/xд
(stream_0_conv_3/kernel/Regularizer/mul_1Mul3stream_0_conv_3/kernel/Regularizer/mul_1/x:output:01stream_0_conv_3/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2*
(stream_0_conv_3/kernel/Regularizer/mul_1Ў
(stream_0_conv_3/kernel/Regularizer/add_1AddV2*stream_0_conv_3/kernel/Regularizer/add:z:0,stream_0_conv_3/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2*
(stream_0_conv_3/kernel/Regularizer/add_1Й
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dense_1/kernel/Regularizer/ConstЃ
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_1_3293178*
_output_shapes

:@@*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOpІ
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:@@2 
dense_1/kernel/Regularizer/AbsЩ
"dense_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_1/kernel/Regularizer/Const_1є
dense_1/kernel/Regularizer/SumSum"dense_1/kernel/Regularizer/Abs:y:0+dense_1/kernel/Regularizer/Const_1:output:0*
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
dense_1/kernel/Regularizer/mulє
dense_1/kernel/Regularizer/addAddV2)dense_1/kernel/Regularizer/Const:output:0"dense_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/addі
0dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_1_3293178*
_output_shapes

:@@*
dtype022
0dense_1/kernel/Regularizer/Square/ReadVariableOp≥
!dense_1/kernel/Regularizer/SquareSquare8dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@@2#
!dense_1/kernel/Regularizer/SquareЩ
"dense_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_1/kernel/Regularizer/Const_2ј
 dense_1/kernel/Regularizer/Sum_1Sum%dense_1/kernel/Regularizer/Square:y:0+dense_1/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2"
 dense_1/kernel/Regularizer/Sum_1Н
"dense_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2$
"dense_1/kernel/Regularizer/mul_1/xƒ
 dense_1/kernel/Regularizer/mul_1Mul+dense_1/kernel/Regularizer/mul_1/x:output:0)dense_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2"
 dense_1/kernel/Regularizer/mul_1Є
 dense_1/kernel/Regularizer/add_1AddV2"dense_1/kernel/Regularizer/add:z:0$dense_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2"
 dense_1/kernel/Regularizer/add_1Ж
IdentityIdentity+dense_activation_1/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@2

Identityи
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall.^dense_1/kernel/Regularizer/Abs/ReadVariableOp1^dense_1/kernel/Regularizer/Square/ReadVariableOp(^stream_0_conv_1/StatefulPartitionedCall6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp9^stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOp(^stream_0_conv_2/StatefulPartitionedCall6^stream_0_conv_2/kernel/Regularizer/Abs/ReadVariableOp9^stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp(^stream_0_conv_3/StatefulPartitionedCall6^stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp9^stream_0_conv_3/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:€€€€€€€€€}: : : : : : : : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2^
-dense_1/kernel/Regularizer/Abs/ReadVariableOp-dense_1/kernel/Regularizer/Abs/ReadVariableOp2d
0dense_1/kernel/Regularizer/Square/ReadVariableOp0dense_1/kernel/Regularizer/Square/ReadVariableOp2R
'stream_0_conv_1/StatefulPartitionedCall'stream_0_conv_1/StatefulPartitionedCall2n
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp2t
8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOp8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOp2R
'stream_0_conv_2/StatefulPartitionedCall'stream_0_conv_2/StatefulPartitionedCall2n
5stream_0_conv_2/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_2/kernel/Regularizer/Abs/ReadVariableOp2t
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp2R
'stream_0_conv_3/StatefulPartitionedCall'stream_0_conv_3/StatefulPartitionedCall2n
5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp2t
8stream_0_conv_3/kernel/Regularizer/Square/ReadVariableOp8stream_0_conv_3/kernel/Regularizer/Square/ReadVariableOp:S O
+
_output_shapes
:€€€€€€€€€}
 
_user_specified_nameinputs
щ
j
L__inference_dense_1_dropout_layer_call_and_return_conditional_losses_3293150

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:€€€€€€€€€@2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:€€€€€€€€€@:O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
є+
л
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_3298089

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
Ѓ
P
4__inference_stream_0_maxpool_3_layer_call_fn_3298158

inputs
identityж
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_stream_0_maxpool_3_layer_call_and_return_conditional_losses_32926582
PartitionedCallВ
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€:e a
=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
Ёz
ч

D__inference_model_1_layer_call_and_return_conditional_losses_3295142

inputs'
basemodel_3295032:@
basemodel_3295034:@
basemodel_3295036:@
basemodel_3295038:@
basemodel_3295040:@
basemodel_3295042:@'
basemodel_3295044:@@
basemodel_3295046:@
basemodel_3295048:@
basemodel_3295050:@
basemodel_3295052:@
basemodel_3295054:@'
basemodel_3295056:@@
basemodel_3295058:@
basemodel_3295060:@
basemodel_3295062:@
basemodel_3295064:@
basemodel_3295066:@#
basemodel_3295068:@@
basemodel_3295070:@
basemodel_3295072:@
basemodel_3295074:@
basemodel_3295076:@
basemodel_3295078:@
identityИҐ!basemodel/StatefulPartitionedCallҐ-dense_1/kernel/Regularizer/Abs/ReadVariableOpҐ0dense_1/kernel/Regularizer/Square/ReadVariableOpҐ5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpҐ8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOpҐ5stream_0_conv_2/kernel/Regularizer/Abs/ReadVariableOpҐ8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOpҐ5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOpҐ8stream_0_conv_3/kernel/Regularizer/Square/ReadVariableOpе
!basemodel/StatefulPartitionedCallStatefulPartitionedCallinputsbasemodel_3295032basemodel_3295034basemodel_3295036basemodel_3295038basemodel_3295040basemodel_3295042basemodel_3295044basemodel_3295046basemodel_3295048basemodel_3295050basemodel_3295052basemodel_3295054basemodel_3295056basemodel_3295058basemodel_3295060basemodel_3295062basemodel_3295064basemodel_3295066basemodel_3295068basemodel_3295070basemodel_3295072basemodel_3295074basemodel_3295076basemodel_3295078*$
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@*2
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_basemodel_layer_call_and_return_conditional_losses_32949232#
!basemodel/StatefulPartitionedCallЩ
(stream_0_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(stream_0_conv_1/kernel/Regularizer/Constƒ
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpbasemodel_3295032*"
_output_shapes
:@*
dtype027
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp√
&stream_0_conv_1/kernel/Regularizer/AbsAbs=stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_0_conv_1/kernel/Regularizer/Abs≠
*stream_0_conv_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_0_conv_1/kernel/Regularizer/Const_1ў
&stream_0_conv_1/kernel/Regularizer/SumSum*stream_0_conv_1/kernel/Regularizer/Abs:y:03stream_0_conv_1/kernel/Regularizer/Const_1:output:0*
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
&stream_0_conv_1/kernel/Regularizer/mulў
&stream_0_conv_1/kernel/Regularizer/addAddV21stream_0_conv_1/kernel/Regularizer/Const:output:0*stream_0_conv_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/add 
8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpbasemodel_3295032*"
_output_shapes
:@*
dtype02:
8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOpѕ
)stream_0_conv_1/kernel/Regularizer/SquareSquare@stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2+
)stream_0_conv_1/kernel/Regularizer/Square≠
*stream_0_conv_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_0_conv_1/kernel/Regularizer/Const_2а
(stream_0_conv_1/kernel/Regularizer/Sum_1Sum-stream_0_conv_1/kernel/Regularizer/Square:y:03stream_0_conv_1/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2*
(stream_0_conv_1/kernel/Regularizer/Sum_1Э
*stream_0_conv_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2,
*stream_0_conv_1/kernel/Regularizer/mul_1/xд
(stream_0_conv_1/kernel/Regularizer/mul_1Mul3stream_0_conv_1/kernel/Regularizer/mul_1/x:output:01stream_0_conv_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2*
(stream_0_conv_1/kernel/Regularizer/mul_1Ў
(stream_0_conv_1/kernel/Regularizer/add_1AddV2*stream_0_conv_1/kernel/Regularizer/add:z:0,stream_0_conv_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2*
(stream_0_conv_1/kernel/Regularizer/add_1Щ
(stream_0_conv_2/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(stream_0_conv_2/kernel/Regularizer/Constƒ
5stream_0_conv_2/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpbasemodel_3295044*"
_output_shapes
:@@*
dtype027
5stream_0_conv_2/kernel/Regularizer/Abs/ReadVariableOp√
&stream_0_conv_2/kernel/Regularizer/AbsAbs=stream_0_conv_2/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@@2(
&stream_0_conv_2/kernel/Regularizer/Abs≠
*stream_0_conv_2/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_0_conv_2/kernel/Regularizer/Const_1ў
&stream_0_conv_2/kernel/Regularizer/SumSum*stream_0_conv_2/kernel/Regularizer/Abs:y:03stream_0_conv_2/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_2/kernel/Regularizer/SumЩ
(stream_0_conv_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2*
(stream_0_conv_2/kernel/Regularizer/mul/x№
&stream_0_conv_2/kernel/Regularizer/mulMul1stream_0_conv_2/kernel/Regularizer/mul/x:output:0/stream_0_conv_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_2/kernel/Regularizer/mulў
&stream_0_conv_2/kernel/Regularizer/addAddV21stream_0_conv_2/kernel/Regularizer/Const:output:0*stream_0_conv_2/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&stream_0_conv_2/kernel/Regularizer/add 
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpbasemodel_3295044*"
_output_shapes
:@@*
dtype02:
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOpѕ
)stream_0_conv_2/kernel/Regularizer/SquareSquare@stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@@2+
)stream_0_conv_2/kernel/Regularizer/Square≠
*stream_0_conv_2/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_0_conv_2/kernel/Regularizer/Const_2а
(stream_0_conv_2/kernel/Regularizer/Sum_1Sum-stream_0_conv_2/kernel/Regularizer/Square:y:03stream_0_conv_2/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2*
(stream_0_conv_2/kernel/Regularizer/Sum_1Э
*stream_0_conv_2/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2,
*stream_0_conv_2/kernel/Regularizer/mul_1/xд
(stream_0_conv_2/kernel/Regularizer/mul_1Mul3stream_0_conv_2/kernel/Regularizer/mul_1/x:output:01stream_0_conv_2/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2*
(stream_0_conv_2/kernel/Regularizer/mul_1Ў
(stream_0_conv_2/kernel/Regularizer/add_1AddV2*stream_0_conv_2/kernel/Regularizer/add:z:0,stream_0_conv_2/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2*
(stream_0_conv_2/kernel/Regularizer/add_1Щ
(stream_0_conv_3/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(stream_0_conv_3/kernel/Regularizer/Constƒ
5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpbasemodel_3295056*"
_output_shapes
:@@*
dtype027
5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp√
&stream_0_conv_3/kernel/Regularizer/AbsAbs=stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@@2(
&stream_0_conv_3/kernel/Regularizer/Abs≠
*stream_0_conv_3/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_0_conv_3/kernel/Regularizer/Const_1ў
&stream_0_conv_3/kernel/Regularizer/SumSum*stream_0_conv_3/kernel/Regularizer/Abs:y:03stream_0_conv_3/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_3/kernel/Regularizer/SumЩ
(stream_0_conv_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2*
(stream_0_conv_3/kernel/Regularizer/mul/x№
&stream_0_conv_3/kernel/Regularizer/mulMul1stream_0_conv_3/kernel/Regularizer/mul/x:output:0/stream_0_conv_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_3/kernel/Regularizer/mulў
&stream_0_conv_3/kernel/Regularizer/addAddV21stream_0_conv_3/kernel/Regularizer/Const:output:0*stream_0_conv_3/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&stream_0_conv_3/kernel/Regularizer/add 
8stream_0_conv_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpbasemodel_3295056*"
_output_shapes
:@@*
dtype02:
8stream_0_conv_3/kernel/Regularizer/Square/ReadVariableOpѕ
)stream_0_conv_3/kernel/Regularizer/SquareSquare@stream_0_conv_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@@2+
)stream_0_conv_3/kernel/Regularizer/Square≠
*stream_0_conv_3/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_0_conv_3/kernel/Regularizer/Const_2а
(stream_0_conv_3/kernel/Regularizer/Sum_1Sum-stream_0_conv_3/kernel/Regularizer/Square:y:03stream_0_conv_3/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2*
(stream_0_conv_3/kernel/Regularizer/Sum_1Э
*stream_0_conv_3/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2,
*stream_0_conv_3/kernel/Regularizer/mul_1/xд
(stream_0_conv_3/kernel/Regularizer/mul_1Mul3stream_0_conv_3/kernel/Regularizer/mul_1/x:output:01stream_0_conv_3/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2*
(stream_0_conv_3/kernel/Regularizer/mul_1Ў
(stream_0_conv_3/kernel/Regularizer/add_1AddV2*stream_0_conv_3/kernel/Regularizer/add:z:0,stream_0_conv_3/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2*
(stream_0_conv_3/kernel/Regularizer/add_1Й
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dense_1/kernel/Regularizer/Const∞
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpbasemodel_3295068*
_output_shapes

:@@*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOpІ
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:@@2 
dense_1/kernel/Regularizer/AbsЩ
"dense_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_1/kernel/Regularizer/Const_1є
dense_1/kernel/Regularizer/SumSum"dense_1/kernel/Regularizer/Abs:y:0+dense_1/kernel/Regularizer/Const_1:output:0*
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
dense_1/kernel/Regularizer/mulє
dense_1/kernel/Regularizer/addAddV2)dense_1/kernel/Regularizer/Const:output:0"dense_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/addґ
0dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpbasemodel_3295068*
_output_shapes

:@@*
dtype022
0dense_1/kernel/Regularizer/Square/ReadVariableOp≥
!dense_1/kernel/Regularizer/SquareSquare8dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@@2#
!dense_1/kernel/Regularizer/SquareЩ
"dense_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_1/kernel/Regularizer/Const_2ј
 dense_1/kernel/Regularizer/Sum_1Sum%dense_1/kernel/Regularizer/Square:y:0+dense_1/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2"
 dense_1/kernel/Regularizer/Sum_1Н
"dense_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2$
"dense_1/kernel/Regularizer/mul_1/xƒ
 dense_1/kernel/Regularizer/mul_1Mul+dense_1/kernel/Regularizer/mul_1/x:output:0)dense_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2"
 dense_1/kernel/Regularizer/mul_1Є
 dense_1/kernel/Regularizer/add_1AddV2"dense_1/kernel/Regularizer/add:z:0$dense_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2"
 dense_1/kernel/Regularizer/add_1Е
IdentityIdentity*basemodel/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@2

IdentityЃ
NoOpNoOp"^basemodel/StatefulPartitionedCall.^dense_1/kernel/Regularizer/Abs/ReadVariableOp1^dense_1/kernel/Regularizer/Square/ReadVariableOp6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp9^stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOp6^stream_0_conv_2/kernel/Regularizer/Abs/ReadVariableOp9^stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp6^stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp9^stream_0_conv_3/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:€€€€€€€€€}: : : : : : : : : : : : : : : : : : : : : : : : 2F
!basemodel/StatefulPartitionedCall!basemodel/StatefulPartitionedCall2^
-dense_1/kernel/Regularizer/Abs/ReadVariableOp-dense_1/kernel/Regularizer/Abs/ReadVariableOp2d
0dense_1/kernel/Regularizer/Square/ReadVariableOp0dense_1/kernel/Regularizer/Square/ReadVariableOp2n
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp2t
8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOp8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOp2n
5stream_0_conv_2/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_2/kernel/Regularizer/Abs/ReadVariableOp2t
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp2n
5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp2t
8stream_0_conv_3/kernel/Regularizer/Square/ReadVariableOp8stream_0_conv_3/kernel/Regularizer/Square/ReadVariableOp:S O
+
_output_shapes
:€€€€€€€€€}
 
_user_specified_nameinputs
№
“
7__inference_batch_normalization_3_layer_call_fn_3298309

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityИҐStatefulPartitionedCallЯ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *[
fVRT
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_32927222
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
К
±
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_3293017

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
:€€€€€€€€€>@2
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
:€€€€€€€€€>@2
batchnorm/add_1r
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€>@2

Identity¬
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€>@: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:S O
+
_output_shapes
:€€€€€€€€€>@
 
_user_specified_nameinputs
Е
q
U__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_3293143

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
:€€€€€€€€€@:S O
+
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
®:
Б
 __inference__traced_save_3298560
file_prefix5
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
@savev2_batch_normalization_3_moving_variance_read_readvariableop
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
ShardedFilenameН

SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Я	
valueХ	BТ	B0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/14/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/15/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesЇ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*E
value<B:B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesЖ
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:01savev2_stream_0_conv_1_kernel_read_readvariableop/savev2_stream_0_conv_1_bias_read_readvariableop4savev2_batch_normalization_gamma_read_readvariableop3savev2_batch_normalization_beta_read_readvariableop1savev2_stream_0_conv_2_kernel_read_readvariableop/savev2_stream_0_conv_2_bias_read_readvariableop6savev2_batch_normalization_1_gamma_read_readvariableop5savev2_batch_normalization_1_beta_read_readvariableop1savev2_stream_0_conv_3_kernel_read_readvariableop/savev2_stream_0_conv_3_bias_read_readvariableop6savev2_batch_normalization_2_gamma_read_readvariableop5savev2_batch_normalization_2_beta_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop6savev2_batch_normalization_3_gamma_read_readvariableop5savev2_batch_normalization_3_beta_read_readvariableop:savev2_batch_normalization_moving_mean_read_readvariableop>savev2_batch_normalization_moving_variance_read_readvariableop<savev2_batch_normalization_1_moving_mean_read_readvariableop@savev2_batch_normalization_1_moving_variance_read_readvariableop<savev2_batch_normalization_2_moving_mean_read_readvariableop@savev2_batch_normalization_2_moving_variance_read_readvariableop<savev2_batch_normalization_3_moving_mean_read_readvariableop@savev2_batch_normalization_3_moving_variance_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *'
dtypes
22
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

identity_1Identity_1:output:0*≈
_input_shapes≥
∞: :@:@:@:@:@@:@:@:@:@@:@:@:@:@@:@:@:@:@:@:@:@:@:@:@:@: 2(
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
:@:($
"
_output_shapes
:@@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@:(	$
"
_output_shapes
:@@: 


_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@:$ 

_output_shapes

:@@: 
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
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@:

_output_shapes
: 
Й
j
L__inference_stream_0_drop_1_layer_call_and_return_conditional_losses_3292960

inputs

identity_1^
IdentityIdentityinputs*
T0*+
_output_shapes
:€€€€€€€€€>@2

Identitym

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:€€€€€€€€€>@2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€>@:S O
+
_output_shapes
:€€€€€€€€€>@
 
_user_specified_nameinputs
Ќ*
л
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_3298376

inputs5
'assignmovingavg_readvariableop_resource:@7
)assignmovingavg_1_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@/
!batchnorm_readvariableop_resource:@
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

:@*
	keep_dims(2
moments/mean|
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:@2
moments/StopGradient§
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
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

:@*
	keep_dims(2
moments/varianceА
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 2
moments/SqueezeИ
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:@*
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
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
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
batchnorm/subЕ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
batchnorm/add_1n
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@2

Identityт
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€@: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
мТ
Ў
"__inference__wrapped_model_3292104
left_inputsc
Mmodel_1_basemodel_stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource:@O
Amodel_1_basemodel_stream_0_conv_1_biasadd_readvariableop_resource:@U
Gmodel_1_basemodel_batch_normalization_batchnorm_readvariableop_resource:@Y
Kmodel_1_basemodel_batch_normalization_batchnorm_mul_readvariableop_resource:@W
Imodel_1_basemodel_batch_normalization_batchnorm_readvariableop_1_resource:@W
Imodel_1_basemodel_batch_normalization_batchnorm_readvariableop_2_resource:@c
Mmodel_1_basemodel_stream_0_conv_2_conv1d_expanddims_1_readvariableop_resource:@@O
Amodel_1_basemodel_stream_0_conv_2_biasadd_readvariableop_resource:@W
Imodel_1_basemodel_batch_normalization_1_batchnorm_readvariableop_resource:@[
Mmodel_1_basemodel_batch_normalization_1_batchnorm_mul_readvariableop_resource:@Y
Kmodel_1_basemodel_batch_normalization_1_batchnorm_readvariableop_1_resource:@Y
Kmodel_1_basemodel_batch_normalization_1_batchnorm_readvariableop_2_resource:@c
Mmodel_1_basemodel_stream_0_conv_3_conv1d_expanddims_1_readvariableop_resource:@@O
Amodel_1_basemodel_stream_0_conv_3_biasadd_readvariableop_resource:@W
Imodel_1_basemodel_batch_normalization_2_batchnorm_readvariableop_resource:@[
Mmodel_1_basemodel_batch_normalization_2_batchnorm_mul_readvariableop_resource:@Y
Kmodel_1_basemodel_batch_normalization_2_batchnorm_readvariableop_1_resource:@Y
Kmodel_1_basemodel_batch_normalization_2_batchnorm_readvariableop_2_resource:@J
8model_1_basemodel_dense_1_matmul_readvariableop_resource:@@G
9model_1_basemodel_dense_1_biasadd_readvariableop_resource:@W
Imodel_1_basemodel_batch_normalization_3_batchnorm_readvariableop_resource:@[
Mmodel_1_basemodel_batch_normalization_3_batchnorm_mul_readvariableop_resource:@Y
Kmodel_1_basemodel_batch_normalization_3_batchnorm_readvariableop_1_resource:@Y
Kmodel_1_basemodel_batch_normalization_3_batchnorm_readvariableop_2_resource:@
identityИҐ>model_1/basemodel/batch_normalization/batchnorm/ReadVariableOpҐ@model_1/basemodel/batch_normalization/batchnorm/ReadVariableOp_1Ґ@model_1/basemodel/batch_normalization/batchnorm/ReadVariableOp_2ҐBmodel_1/basemodel/batch_normalization/batchnorm/mul/ReadVariableOpҐ@model_1/basemodel/batch_normalization_1/batchnorm/ReadVariableOpҐBmodel_1/basemodel/batch_normalization_1/batchnorm/ReadVariableOp_1ҐBmodel_1/basemodel/batch_normalization_1/batchnorm/ReadVariableOp_2ҐDmodel_1/basemodel/batch_normalization_1/batchnorm/mul/ReadVariableOpҐ@model_1/basemodel/batch_normalization_2/batchnorm/ReadVariableOpҐBmodel_1/basemodel/batch_normalization_2/batchnorm/ReadVariableOp_1ҐBmodel_1/basemodel/batch_normalization_2/batchnorm/ReadVariableOp_2ҐDmodel_1/basemodel/batch_normalization_2/batchnorm/mul/ReadVariableOpҐ@model_1/basemodel/batch_normalization_3/batchnorm/ReadVariableOpҐBmodel_1/basemodel/batch_normalization_3/batchnorm/ReadVariableOp_1ҐBmodel_1/basemodel/batch_normalization_3/batchnorm/ReadVariableOp_2ҐDmodel_1/basemodel/batch_normalization_3/batchnorm/mul/ReadVariableOpҐ0model_1/basemodel/dense_1/BiasAdd/ReadVariableOpҐ/model_1/basemodel/dense_1/MatMul/ReadVariableOpҐ8model_1/basemodel/stream_0_conv_1/BiasAdd/ReadVariableOpҐDmodel_1/basemodel/stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOpҐ8model_1/basemodel/stream_0_conv_2/BiasAdd/ReadVariableOpҐDmodel_1/basemodel/stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOpҐ8model_1/basemodel/stream_0_conv_3/BiasAdd/ReadVariableOpҐDmodel_1/basemodel/stream_0_conv_3/conv1d/ExpandDims_1/ReadVariableOpѓ
.model_1/basemodel/stream_0_input_drop/IdentityIdentityleft_inputs*
T0*+
_output_shapes
:€€€€€€€€€}20
.model_1/basemodel/stream_0_input_drop/Identityљ
7model_1/basemodel/stream_0_conv_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€29
7model_1/basemodel/stream_0_conv_1/conv1d/ExpandDims/dim≠
3model_1/basemodel/stream_0_conv_1/conv1d/ExpandDims
ExpandDims7model_1/basemodel/stream_0_input_drop/Identity:output:0@model_1/basemodel/stream_0_conv_1/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:€€€€€€€€€}25
3model_1/basemodel/stream_0_conv_1/conv1d/ExpandDimsЮ
Dmodel_1/basemodel/stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpMmodel_1_basemodel_stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02F
Dmodel_1/basemodel/stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOpЄ
9model_1/basemodel/stream_0_conv_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2;
9model_1/basemodel/stream_0_conv_1/conv1d/ExpandDims_1/dimњ
5model_1/basemodel/stream_0_conv_1/conv1d/ExpandDims_1
ExpandDimsLmodel_1/basemodel/stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp:value:0Bmodel_1/basemodel/stream_0_conv_1/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@27
5model_1/basemodel/stream_0_conv_1/conv1d/ExpandDims_1Њ
(model_1/basemodel/stream_0_conv_1/conv1dConv2D<model_1/basemodel/stream_0_conv_1/conv1d/ExpandDims:output:0>model_1/basemodel/stream_0_conv_1/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:€€€€€€€€€}@*
paddingSAME*
strides
2*
(model_1/basemodel/stream_0_conv_1/conv1dш
0model_1/basemodel/stream_0_conv_1/conv1d/SqueezeSqueeze1model_1/basemodel/stream_0_conv_1/conv1d:output:0*
T0*+
_output_shapes
:€€€€€€€€€}@*
squeeze_dims

э€€€€€€€€22
0model_1/basemodel/stream_0_conv_1/conv1d/Squeezeт
8model_1/basemodel/stream_0_conv_1/BiasAdd/ReadVariableOpReadVariableOpAmodel_1_basemodel_stream_0_conv_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02:
8model_1/basemodel/stream_0_conv_1/BiasAdd/ReadVariableOpФ
)model_1/basemodel/stream_0_conv_1/BiasAddBiasAdd9model_1/basemodel/stream_0_conv_1/conv1d/Squeeze:output:0@model_1/basemodel/stream_0_conv_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€}@2+
)model_1/basemodel/stream_0_conv_1/BiasAddД
>model_1/basemodel/batch_normalization/batchnorm/ReadVariableOpReadVariableOpGmodel_1_basemodel_batch_normalization_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02@
>model_1/basemodel/batch_normalization/batchnorm/ReadVariableOp≥
5model_1/basemodel/batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:27
5model_1/basemodel/batch_normalization/batchnorm/add/y†
3model_1/basemodel/batch_normalization/batchnorm/addAddV2Fmodel_1/basemodel/batch_normalization/batchnorm/ReadVariableOp:value:0>model_1/basemodel/batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes
:@25
3model_1/basemodel/batch_normalization/batchnorm/add’
5model_1/basemodel/batch_normalization/batchnorm/RsqrtRsqrt7model_1/basemodel/batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes
:@27
5model_1/basemodel/batch_normalization/batchnorm/RsqrtР
Bmodel_1/basemodel/batch_normalization/batchnorm/mul/ReadVariableOpReadVariableOpKmodel_1_basemodel_batch_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02D
Bmodel_1/basemodel/batch_normalization/batchnorm/mul/ReadVariableOpЭ
3model_1/basemodel/batch_normalization/batchnorm/mulMul9model_1/basemodel/batch_normalization/batchnorm/Rsqrt:y:0Jmodel_1/basemodel/batch_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@25
3model_1/basemodel/batch_normalization/batchnorm/mulШ
5model_1/basemodel/batch_normalization/batchnorm/mul_1Mul2model_1/basemodel/stream_0_conv_1/BiasAdd:output:07model_1/basemodel/batch_normalization/batchnorm/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€}@27
5model_1/basemodel/batch_normalization/batchnorm/mul_1К
@model_1/basemodel/batch_normalization/batchnorm/ReadVariableOp_1ReadVariableOpImodel_1_basemodel_batch_normalization_batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype02B
@model_1/basemodel/batch_normalization/batchnorm/ReadVariableOp_1Э
5model_1/basemodel/batch_normalization/batchnorm/mul_2MulHmodel_1/basemodel/batch_normalization/batchnorm/ReadVariableOp_1:value:07model_1/basemodel/batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes
:@27
5model_1/basemodel/batch_normalization/batchnorm/mul_2К
@model_1/basemodel/batch_normalization/batchnorm/ReadVariableOp_2ReadVariableOpImodel_1_basemodel_batch_normalization_batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype02B
@model_1/basemodel/batch_normalization/batchnorm/ReadVariableOp_2Ы
3model_1/basemodel/batch_normalization/batchnorm/subSubHmodel_1/basemodel/batch_normalization/batchnorm/ReadVariableOp_2:value:09model_1/basemodel/batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@25
3model_1/basemodel/batch_normalization/batchnorm/sub°
5model_1/basemodel/batch_normalization/batchnorm/add_1AddV29model_1/basemodel/batch_normalization/batchnorm/mul_1:z:07model_1/basemodel/batch_normalization/batchnorm/sub:z:0*
T0*+
_output_shapes
:€€€€€€€€€}@27
5model_1/basemodel/batch_normalization/batchnorm/add_1њ
!model_1/basemodel/activation/TanhTanh9model_1/basemodel/batch_normalization/batchnorm/add_1:z:0*
T0*+
_output_shapes
:€€€€€€€€€}@2#
!model_1/basemodel/activation/Tanhђ
3model_1/basemodel/stream_0_maxpool_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :25
3model_1/basemodel/stream_0_maxpool_1/ExpandDims/dimП
/model_1/basemodel/stream_0_maxpool_1/ExpandDims
ExpandDims%model_1/basemodel/activation/Tanh:y:0<model_1/basemodel/stream_0_maxpool_1/ExpandDims/dim:output:0*
T0*/
_output_shapes
:€€€€€€€€€}@21
/model_1/basemodel/stream_0_maxpool_1/ExpandDimsО
,model_1/basemodel/stream_0_maxpool_1/MaxPoolMaxPool8model_1/basemodel/stream_0_maxpool_1/ExpandDims:output:0*/
_output_shapes
:€€€€€€€€€>@*
ksize
*
paddingVALID*
strides
2.
,model_1/basemodel/stream_0_maxpool_1/MaxPoolл
,model_1/basemodel/stream_0_maxpool_1/SqueezeSqueeze5model_1/basemodel/stream_0_maxpool_1/MaxPool:output:0*
T0*+
_output_shapes
:€€€€€€€€€>@*
squeeze_dims
2.
,model_1/basemodel/stream_0_maxpool_1/Squeeze—
*model_1/basemodel/stream_0_drop_1/IdentityIdentity5model_1/basemodel/stream_0_maxpool_1/Squeeze:output:0*
T0*+
_output_shapes
:€€€€€€€€€>@2,
*model_1/basemodel/stream_0_drop_1/Identityљ
7model_1/basemodel/stream_0_conv_2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€29
7model_1/basemodel/stream_0_conv_2/conv1d/ExpandDims/dim©
3model_1/basemodel/stream_0_conv_2/conv1d/ExpandDims
ExpandDims3model_1/basemodel/stream_0_drop_1/Identity:output:0@model_1/basemodel/stream_0_conv_2/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:€€€€€€€€€>@25
3model_1/basemodel/stream_0_conv_2/conv1d/ExpandDimsЮ
Dmodel_1/basemodel/stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpMmodel_1_basemodel_stream_0_conv_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype02F
Dmodel_1/basemodel/stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOpЄ
9model_1/basemodel/stream_0_conv_2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2;
9model_1/basemodel/stream_0_conv_2/conv1d/ExpandDims_1/dimњ
5model_1/basemodel/stream_0_conv_2/conv1d/ExpandDims_1
ExpandDimsLmodel_1/basemodel/stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOp:value:0Bmodel_1/basemodel/stream_0_conv_2/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@27
5model_1/basemodel/stream_0_conv_2/conv1d/ExpandDims_1Њ
(model_1/basemodel/stream_0_conv_2/conv1dConv2D<model_1/basemodel/stream_0_conv_2/conv1d/ExpandDims:output:0>model_1/basemodel/stream_0_conv_2/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:€€€€€€€€€>@*
paddingSAME*
strides
2*
(model_1/basemodel/stream_0_conv_2/conv1dш
0model_1/basemodel/stream_0_conv_2/conv1d/SqueezeSqueeze1model_1/basemodel/stream_0_conv_2/conv1d:output:0*
T0*+
_output_shapes
:€€€€€€€€€>@*
squeeze_dims

э€€€€€€€€22
0model_1/basemodel/stream_0_conv_2/conv1d/Squeezeт
8model_1/basemodel/stream_0_conv_2/BiasAdd/ReadVariableOpReadVariableOpAmodel_1_basemodel_stream_0_conv_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02:
8model_1/basemodel/stream_0_conv_2/BiasAdd/ReadVariableOpФ
)model_1/basemodel/stream_0_conv_2/BiasAddBiasAdd9model_1/basemodel/stream_0_conv_2/conv1d/Squeeze:output:0@model_1/basemodel/stream_0_conv_2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€>@2+
)model_1/basemodel/stream_0_conv_2/BiasAddК
@model_1/basemodel/batch_normalization_1/batchnorm/ReadVariableOpReadVariableOpImodel_1_basemodel_batch_normalization_1_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02B
@model_1/basemodel/batch_normalization_1/batchnorm/ReadVariableOpЈ
7model_1/basemodel/batch_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:29
7model_1/basemodel/batch_normalization_1/batchnorm/add/y®
5model_1/basemodel/batch_normalization_1/batchnorm/addAddV2Hmodel_1/basemodel/batch_normalization_1/batchnorm/ReadVariableOp:value:0@model_1/basemodel/batch_normalization_1/batchnorm/add/y:output:0*
T0*
_output_shapes
:@27
5model_1/basemodel/batch_normalization_1/batchnorm/addџ
7model_1/basemodel/batch_normalization_1/batchnorm/RsqrtRsqrt9model_1/basemodel/batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes
:@29
7model_1/basemodel/batch_normalization_1/batchnorm/RsqrtЦ
Dmodel_1/basemodel/batch_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOpMmodel_1_basemodel_batch_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02F
Dmodel_1/basemodel/batch_normalization_1/batchnorm/mul/ReadVariableOp•
5model_1/basemodel/batch_normalization_1/batchnorm/mulMul;model_1/basemodel/batch_normalization_1/batchnorm/Rsqrt:y:0Lmodel_1/basemodel/batch_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@27
5model_1/basemodel/batch_normalization_1/batchnorm/mulЮ
7model_1/basemodel/batch_normalization_1/batchnorm/mul_1Mul2model_1/basemodel/stream_0_conv_2/BiasAdd:output:09model_1/basemodel/batch_normalization_1/batchnorm/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€>@29
7model_1/basemodel/batch_normalization_1/batchnorm/mul_1Р
Bmodel_1/basemodel/batch_normalization_1/batchnorm/ReadVariableOp_1ReadVariableOpKmodel_1_basemodel_batch_normalization_1_batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype02D
Bmodel_1/basemodel/batch_normalization_1/batchnorm/ReadVariableOp_1•
7model_1/basemodel/batch_normalization_1/batchnorm/mul_2MulJmodel_1/basemodel/batch_normalization_1/batchnorm/ReadVariableOp_1:value:09model_1/basemodel/batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes
:@29
7model_1/basemodel/batch_normalization_1/batchnorm/mul_2Р
Bmodel_1/basemodel/batch_normalization_1/batchnorm/ReadVariableOp_2ReadVariableOpKmodel_1_basemodel_batch_normalization_1_batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype02D
Bmodel_1/basemodel/batch_normalization_1/batchnorm/ReadVariableOp_2£
5model_1/basemodel/batch_normalization_1/batchnorm/subSubJmodel_1/basemodel/batch_normalization_1/batchnorm/ReadVariableOp_2:value:0;model_1/basemodel/batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@27
5model_1/basemodel/batch_normalization_1/batchnorm/sub©
7model_1/basemodel/batch_normalization_1/batchnorm/add_1AddV2;model_1/basemodel/batch_normalization_1/batchnorm/mul_1:z:09model_1/basemodel/batch_normalization_1/batchnorm/sub:z:0*
T0*+
_output_shapes
:€€€€€€€€€>@29
7model_1/basemodel/batch_normalization_1/batchnorm/add_1≈
#model_1/basemodel/activation_1/TanhTanh;model_1/basemodel/batch_normalization_1/batchnorm/add_1:z:0*
T0*+
_output_shapes
:€€€€€€€€€>@2%
#model_1/basemodel/activation_1/Tanhђ
3model_1/basemodel/stream_0_maxpool_2/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :25
3model_1/basemodel/stream_0_maxpool_2/ExpandDims/dimС
/model_1/basemodel/stream_0_maxpool_2/ExpandDims
ExpandDims'model_1/basemodel/activation_1/Tanh:y:0<model_1/basemodel/stream_0_maxpool_2/ExpandDims/dim:output:0*
T0*/
_output_shapes
:€€€€€€€€€>@21
/model_1/basemodel/stream_0_maxpool_2/ExpandDimsО
,model_1/basemodel/stream_0_maxpool_2/MaxPoolMaxPool8model_1/basemodel/stream_0_maxpool_2/ExpandDims:output:0*/
_output_shapes
:€€€€€€€€€@*
ksize
*
paddingVALID*
strides
2.
,model_1/basemodel/stream_0_maxpool_2/MaxPoolл
,model_1/basemodel/stream_0_maxpool_2/SqueezeSqueeze5model_1/basemodel/stream_0_maxpool_2/MaxPool:output:0*
T0*+
_output_shapes
:€€€€€€€€€@*
squeeze_dims
2.
,model_1/basemodel/stream_0_maxpool_2/Squeeze—
*model_1/basemodel/stream_0_drop_2/IdentityIdentity5model_1/basemodel/stream_0_maxpool_2/Squeeze:output:0*
T0*+
_output_shapes
:€€€€€€€€€@2,
*model_1/basemodel/stream_0_drop_2/Identityљ
7model_1/basemodel/stream_0_conv_3/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€29
7model_1/basemodel/stream_0_conv_3/conv1d/ExpandDims/dim©
3model_1/basemodel/stream_0_conv_3/conv1d/ExpandDims
ExpandDims3model_1/basemodel/stream_0_drop_2/Identity:output:0@model_1/basemodel/stream_0_conv_3/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:€€€€€€€€€@25
3model_1/basemodel/stream_0_conv_3/conv1d/ExpandDimsЮ
Dmodel_1/basemodel/stream_0_conv_3/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpMmodel_1_basemodel_stream_0_conv_3_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype02F
Dmodel_1/basemodel/stream_0_conv_3/conv1d/ExpandDims_1/ReadVariableOpЄ
9model_1/basemodel/stream_0_conv_3/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2;
9model_1/basemodel/stream_0_conv_3/conv1d/ExpandDims_1/dimњ
5model_1/basemodel/stream_0_conv_3/conv1d/ExpandDims_1
ExpandDimsLmodel_1/basemodel/stream_0_conv_3/conv1d/ExpandDims_1/ReadVariableOp:value:0Bmodel_1/basemodel/stream_0_conv_3/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@27
5model_1/basemodel/stream_0_conv_3/conv1d/ExpandDims_1Њ
(model_1/basemodel/stream_0_conv_3/conv1dConv2D<model_1/basemodel/stream_0_conv_3/conv1d/ExpandDims:output:0>model_1/basemodel/stream_0_conv_3/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:€€€€€€€€€@*
paddingSAME*
strides
2*
(model_1/basemodel/stream_0_conv_3/conv1dш
0model_1/basemodel/stream_0_conv_3/conv1d/SqueezeSqueeze1model_1/basemodel/stream_0_conv_3/conv1d:output:0*
T0*+
_output_shapes
:€€€€€€€€€@*
squeeze_dims

э€€€€€€€€22
0model_1/basemodel/stream_0_conv_3/conv1d/Squeezeт
8model_1/basemodel/stream_0_conv_3/BiasAdd/ReadVariableOpReadVariableOpAmodel_1_basemodel_stream_0_conv_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02:
8model_1/basemodel/stream_0_conv_3/BiasAdd/ReadVariableOpФ
)model_1/basemodel/stream_0_conv_3/BiasAddBiasAdd9model_1/basemodel/stream_0_conv_3/conv1d/Squeeze:output:0@model_1/basemodel/stream_0_conv_3/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€@2+
)model_1/basemodel/stream_0_conv_3/BiasAddК
@model_1/basemodel/batch_normalization_2/batchnorm/ReadVariableOpReadVariableOpImodel_1_basemodel_batch_normalization_2_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02B
@model_1/basemodel/batch_normalization_2/batchnorm/ReadVariableOpЈ
7model_1/basemodel/batch_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:29
7model_1/basemodel/batch_normalization_2/batchnorm/add/y®
5model_1/basemodel/batch_normalization_2/batchnorm/addAddV2Hmodel_1/basemodel/batch_normalization_2/batchnorm/ReadVariableOp:value:0@model_1/basemodel/batch_normalization_2/batchnorm/add/y:output:0*
T0*
_output_shapes
:@27
5model_1/basemodel/batch_normalization_2/batchnorm/addџ
7model_1/basemodel/batch_normalization_2/batchnorm/RsqrtRsqrt9model_1/basemodel/batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes
:@29
7model_1/basemodel/batch_normalization_2/batchnorm/RsqrtЦ
Dmodel_1/basemodel/batch_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOpMmodel_1_basemodel_batch_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02F
Dmodel_1/basemodel/batch_normalization_2/batchnorm/mul/ReadVariableOp•
5model_1/basemodel/batch_normalization_2/batchnorm/mulMul;model_1/basemodel/batch_normalization_2/batchnorm/Rsqrt:y:0Lmodel_1/basemodel/batch_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@27
5model_1/basemodel/batch_normalization_2/batchnorm/mulЮ
7model_1/basemodel/batch_normalization_2/batchnorm/mul_1Mul2model_1/basemodel/stream_0_conv_3/BiasAdd:output:09model_1/basemodel/batch_normalization_2/batchnorm/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€@29
7model_1/basemodel/batch_normalization_2/batchnorm/mul_1Р
Bmodel_1/basemodel/batch_normalization_2/batchnorm/ReadVariableOp_1ReadVariableOpKmodel_1_basemodel_batch_normalization_2_batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype02D
Bmodel_1/basemodel/batch_normalization_2/batchnorm/ReadVariableOp_1•
7model_1/basemodel/batch_normalization_2/batchnorm/mul_2MulJmodel_1/basemodel/batch_normalization_2/batchnorm/ReadVariableOp_1:value:09model_1/basemodel/batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes
:@29
7model_1/basemodel/batch_normalization_2/batchnorm/mul_2Р
Bmodel_1/basemodel/batch_normalization_2/batchnorm/ReadVariableOp_2ReadVariableOpKmodel_1_basemodel_batch_normalization_2_batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype02D
Bmodel_1/basemodel/batch_normalization_2/batchnorm/ReadVariableOp_2£
5model_1/basemodel/batch_normalization_2/batchnorm/subSubJmodel_1/basemodel/batch_normalization_2/batchnorm/ReadVariableOp_2:value:0;model_1/basemodel/batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@27
5model_1/basemodel/batch_normalization_2/batchnorm/sub©
7model_1/basemodel/batch_normalization_2/batchnorm/add_1AddV2;model_1/basemodel/batch_normalization_2/batchnorm/mul_1:z:09model_1/basemodel/batch_normalization_2/batchnorm/sub:z:0*
T0*+
_output_shapes
:€€€€€€€€€@29
7model_1/basemodel/batch_normalization_2/batchnorm/add_1≈
#model_1/basemodel/activation_2/TanhTanh;model_1/basemodel/batch_normalization_2/batchnorm/add_1:z:0*
T0*+
_output_shapes
:€€€€€€€€€@2%
#model_1/basemodel/activation_2/Tanhђ
3model_1/basemodel/stream_0_maxpool_3/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :25
3model_1/basemodel/stream_0_maxpool_3/ExpandDims/dimС
/model_1/basemodel/stream_0_maxpool_3/ExpandDims
ExpandDims'model_1/basemodel/activation_2/Tanh:y:0<model_1/basemodel/stream_0_maxpool_3/ExpandDims/dim:output:0*
T0*/
_output_shapes
:€€€€€€€€€@21
/model_1/basemodel/stream_0_maxpool_3/ExpandDimsО
,model_1/basemodel/stream_0_maxpool_3/MaxPoolMaxPool8model_1/basemodel/stream_0_maxpool_3/ExpandDims:output:0*/
_output_shapes
:€€€€€€€€€@*
ksize
*
paddingVALID*
strides
2.
,model_1/basemodel/stream_0_maxpool_3/MaxPoolл
,model_1/basemodel/stream_0_maxpool_3/SqueezeSqueeze5model_1/basemodel/stream_0_maxpool_3/MaxPool:output:0*
T0*+
_output_shapes
:€€€€€€€€€@*
squeeze_dims
2.
,model_1/basemodel/stream_0_maxpool_3/Squeeze—
*model_1/basemodel/stream_0_drop_3/IdentityIdentity5model_1/basemodel/stream_0_maxpool_3/Squeeze:output:0*
T0*+
_output_shapes
:€€€€€€€€€@2,
*model_1/basemodel/stream_0_drop_3/Identity»
Amodel_1/basemodel/global_average_pooling1d/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2C
Amodel_1/basemodel/global_average_pooling1d/Mean/reduction_indicesЭ
/model_1/basemodel/global_average_pooling1d/MeanMean3model_1/basemodel/stream_0_drop_3/Identity:output:0Jmodel_1/basemodel/global_average_pooling1d/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:€€€€€€€€€@21
/model_1/basemodel/global_average_pooling1d/Mean–
*model_1/basemodel/dense_1_dropout/IdentityIdentity8model_1/basemodel/global_average_pooling1d/Mean:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2,
*model_1/basemodel/dense_1_dropout/Identityџ
/model_1/basemodel/dense_1/MatMul/ReadVariableOpReadVariableOp8model_1_basemodel_dense_1_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype021
/model_1/basemodel/dense_1/MatMul/ReadVariableOpо
 model_1/basemodel/dense_1/MatMulMatMul3model_1/basemodel/dense_1_dropout/Identity:output:07model_1/basemodel/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2"
 model_1/basemodel/dense_1/MatMulЏ
0model_1/basemodel/dense_1/BiasAdd/ReadVariableOpReadVariableOp9model_1_basemodel_dense_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype022
0model_1/basemodel/dense_1/BiasAdd/ReadVariableOpй
!model_1/basemodel/dense_1/BiasAddBiasAdd*model_1/basemodel/dense_1/MatMul:product:08model_1/basemodel/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2#
!model_1/basemodel/dense_1/BiasAddК
@model_1/basemodel/batch_normalization_3/batchnorm/ReadVariableOpReadVariableOpImodel_1_basemodel_batch_normalization_3_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02B
@model_1/basemodel/batch_normalization_3/batchnorm/ReadVariableOpЈ
7model_1/basemodel/batch_normalization_3/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:29
7model_1/basemodel/batch_normalization_3/batchnorm/add/y®
5model_1/basemodel/batch_normalization_3/batchnorm/addAddV2Hmodel_1/basemodel/batch_normalization_3/batchnorm/ReadVariableOp:value:0@model_1/basemodel/batch_normalization_3/batchnorm/add/y:output:0*
T0*
_output_shapes
:@27
5model_1/basemodel/batch_normalization_3/batchnorm/addџ
7model_1/basemodel/batch_normalization_3/batchnorm/RsqrtRsqrt9model_1/basemodel/batch_normalization_3/batchnorm/add:z:0*
T0*
_output_shapes
:@29
7model_1/basemodel/batch_normalization_3/batchnorm/RsqrtЦ
Dmodel_1/basemodel/batch_normalization_3/batchnorm/mul/ReadVariableOpReadVariableOpMmodel_1_basemodel_batch_normalization_3_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02F
Dmodel_1/basemodel/batch_normalization_3/batchnorm/mul/ReadVariableOp•
5model_1/basemodel/batch_normalization_3/batchnorm/mulMul;model_1/basemodel/batch_normalization_3/batchnorm/Rsqrt:y:0Lmodel_1/basemodel/batch_normalization_3/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@27
5model_1/basemodel/batch_normalization_3/batchnorm/mulТ
7model_1/basemodel/batch_normalization_3/batchnorm/mul_1Mul*model_1/basemodel/dense_1/BiasAdd:output:09model_1/basemodel/batch_normalization_3/batchnorm/mul:z:0*
T0*'
_output_shapes
:€€€€€€€€€@29
7model_1/basemodel/batch_normalization_3/batchnorm/mul_1Р
Bmodel_1/basemodel/batch_normalization_3/batchnorm/ReadVariableOp_1ReadVariableOpKmodel_1_basemodel_batch_normalization_3_batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype02D
Bmodel_1/basemodel/batch_normalization_3/batchnorm/ReadVariableOp_1•
7model_1/basemodel/batch_normalization_3/batchnorm/mul_2MulJmodel_1/basemodel/batch_normalization_3/batchnorm/ReadVariableOp_1:value:09model_1/basemodel/batch_normalization_3/batchnorm/mul:z:0*
T0*
_output_shapes
:@29
7model_1/basemodel/batch_normalization_3/batchnorm/mul_2Р
Bmodel_1/basemodel/batch_normalization_3/batchnorm/ReadVariableOp_2ReadVariableOpKmodel_1_basemodel_batch_normalization_3_batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype02D
Bmodel_1/basemodel/batch_normalization_3/batchnorm/ReadVariableOp_2£
5model_1/basemodel/batch_normalization_3/batchnorm/subSubJmodel_1/basemodel/batch_normalization_3/batchnorm/ReadVariableOp_2:value:0;model_1/basemodel/batch_normalization_3/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@27
5model_1/basemodel/batch_normalization_3/batchnorm/sub•
7model_1/basemodel/batch_normalization_3/batchnorm/add_1AddV2;model_1/basemodel/batch_normalization_3/batchnorm/mul_1:z:09model_1/basemodel/batch_normalization_3/batchnorm/sub:z:0*
T0*'
_output_shapes
:€€€€€€€€€@29
7model_1/basemodel/batch_normalization_3/batchnorm/add_1Ц
IdentityIdentity;model_1/basemodel/batch_normalization_3/batchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@2

IdentityБ
NoOpNoOp?^model_1/basemodel/batch_normalization/batchnorm/ReadVariableOpA^model_1/basemodel/batch_normalization/batchnorm/ReadVariableOp_1A^model_1/basemodel/batch_normalization/batchnorm/ReadVariableOp_2C^model_1/basemodel/batch_normalization/batchnorm/mul/ReadVariableOpA^model_1/basemodel/batch_normalization_1/batchnorm/ReadVariableOpC^model_1/basemodel/batch_normalization_1/batchnorm/ReadVariableOp_1C^model_1/basemodel/batch_normalization_1/batchnorm/ReadVariableOp_2E^model_1/basemodel/batch_normalization_1/batchnorm/mul/ReadVariableOpA^model_1/basemodel/batch_normalization_2/batchnorm/ReadVariableOpC^model_1/basemodel/batch_normalization_2/batchnorm/ReadVariableOp_1C^model_1/basemodel/batch_normalization_2/batchnorm/ReadVariableOp_2E^model_1/basemodel/batch_normalization_2/batchnorm/mul/ReadVariableOpA^model_1/basemodel/batch_normalization_3/batchnorm/ReadVariableOpC^model_1/basemodel/batch_normalization_3/batchnorm/ReadVariableOp_1C^model_1/basemodel/batch_normalization_3/batchnorm/ReadVariableOp_2E^model_1/basemodel/batch_normalization_3/batchnorm/mul/ReadVariableOp1^model_1/basemodel/dense_1/BiasAdd/ReadVariableOp0^model_1/basemodel/dense_1/MatMul/ReadVariableOp9^model_1/basemodel/stream_0_conv_1/BiasAdd/ReadVariableOpE^model_1/basemodel/stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp9^model_1/basemodel/stream_0_conv_2/BiasAdd/ReadVariableOpE^model_1/basemodel/stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOp9^model_1/basemodel/stream_0_conv_3/BiasAdd/ReadVariableOpE^model_1/basemodel/stream_0_conv_3/conv1d/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:€€€€€€€€€}: : : : : : : : : : : : : : : : : : : : : : : : 2А
>model_1/basemodel/batch_normalization/batchnorm/ReadVariableOp>model_1/basemodel/batch_normalization/batchnorm/ReadVariableOp2Д
@model_1/basemodel/batch_normalization/batchnorm/ReadVariableOp_1@model_1/basemodel/batch_normalization/batchnorm/ReadVariableOp_12Д
@model_1/basemodel/batch_normalization/batchnorm/ReadVariableOp_2@model_1/basemodel/batch_normalization/batchnorm/ReadVariableOp_22И
Bmodel_1/basemodel/batch_normalization/batchnorm/mul/ReadVariableOpBmodel_1/basemodel/batch_normalization/batchnorm/mul/ReadVariableOp2Д
@model_1/basemodel/batch_normalization_1/batchnorm/ReadVariableOp@model_1/basemodel/batch_normalization_1/batchnorm/ReadVariableOp2И
Bmodel_1/basemodel/batch_normalization_1/batchnorm/ReadVariableOp_1Bmodel_1/basemodel/batch_normalization_1/batchnorm/ReadVariableOp_12И
Bmodel_1/basemodel/batch_normalization_1/batchnorm/ReadVariableOp_2Bmodel_1/basemodel/batch_normalization_1/batchnorm/ReadVariableOp_22М
Dmodel_1/basemodel/batch_normalization_1/batchnorm/mul/ReadVariableOpDmodel_1/basemodel/batch_normalization_1/batchnorm/mul/ReadVariableOp2Д
@model_1/basemodel/batch_normalization_2/batchnorm/ReadVariableOp@model_1/basemodel/batch_normalization_2/batchnorm/ReadVariableOp2И
Bmodel_1/basemodel/batch_normalization_2/batchnorm/ReadVariableOp_1Bmodel_1/basemodel/batch_normalization_2/batchnorm/ReadVariableOp_12И
Bmodel_1/basemodel/batch_normalization_2/batchnorm/ReadVariableOp_2Bmodel_1/basemodel/batch_normalization_2/batchnorm/ReadVariableOp_22М
Dmodel_1/basemodel/batch_normalization_2/batchnorm/mul/ReadVariableOpDmodel_1/basemodel/batch_normalization_2/batchnorm/mul/ReadVariableOp2Д
@model_1/basemodel/batch_normalization_3/batchnorm/ReadVariableOp@model_1/basemodel/batch_normalization_3/batchnorm/ReadVariableOp2И
Bmodel_1/basemodel/batch_normalization_3/batchnorm/ReadVariableOp_1Bmodel_1/basemodel/batch_normalization_3/batchnorm/ReadVariableOp_12И
Bmodel_1/basemodel/batch_normalization_3/batchnorm/ReadVariableOp_2Bmodel_1/basemodel/batch_normalization_3/batchnorm/ReadVariableOp_22М
Dmodel_1/basemodel/batch_normalization_3/batchnorm/mul/ReadVariableOpDmodel_1/basemodel/batch_normalization_3/batchnorm/mul/ReadVariableOp2d
0model_1/basemodel/dense_1/BiasAdd/ReadVariableOp0model_1/basemodel/dense_1/BiasAdd/ReadVariableOp2b
/model_1/basemodel/dense_1/MatMul/ReadVariableOp/model_1/basemodel/dense_1/MatMul/ReadVariableOp2t
8model_1/basemodel/stream_0_conv_1/BiasAdd/ReadVariableOp8model_1/basemodel/stream_0_conv_1/BiasAdd/ReadVariableOp2М
Dmodel_1/basemodel/stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOpDmodel_1/basemodel/stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp2t
8model_1/basemodel/stream_0_conv_2/BiasAdd/ReadVariableOp8model_1/basemodel/stream_0_conv_2/BiasAdd/ReadVariableOp2М
Dmodel_1/basemodel/stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOpDmodel_1/basemodel/stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOp2t
8model_1/basemodel/stream_0_conv_3/BiasAdd/ReadVariableOp8model_1/basemodel/stream_0_conv_3/BiasAdd/ReadVariableOp2М
Dmodel_1/basemodel/stream_0_conv_3/conv1d/ExpandDims_1/ReadVariableOpDmodel_1/basemodel/stream_0_conv_3/conv1d/ExpandDims_1/ReadVariableOp:X T
+
_output_shapes
:€€€€€€€€€}
%
_user_specified_nameleft_inputs
ц
±
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_3292722

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
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
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
batchnorm/subЕ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
batchnorm/add_1n
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity¬
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€@: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
Я
V
:__inference_global_average_pooling1d_layer_call_fn_3298211

inputs
identityя
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
GPU2*0J 8В *^
fYRW
U__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_32926842
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
у,
О
L__inference_stream_0_conv_3_layer_call_and_return_conditional_losses_3293080

inputsA
+conv1d_expanddims_1_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identityИҐBiasAdd/ReadVariableOpҐ"conv1d/ExpandDims_1/ReadVariableOpҐ5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOpҐ8stream_0_conv_3/kernel/Regularizer/Square/ReadVariableOpy
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
:€€€€€€€€€@2
conv1d/ExpandDimsЄ
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
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
:@@2
conv1d/ExpandDims_1ґ
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:€€€€€€€€€@*
paddingSAME*
strides
2
conv1dТ
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:€€€€€€€€€@*
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
:€€€€€€€€€@2	
BiasAddЩ
(stream_0_conv_3/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(stream_0_conv_3/kernel/Regularizer/Constё
5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype027
5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp√
&stream_0_conv_3/kernel/Regularizer/AbsAbs=stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@@2(
&stream_0_conv_3/kernel/Regularizer/Abs≠
*stream_0_conv_3/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_0_conv_3/kernel/Regularizer/Const_1ў
&stream_0_conv_3/kernel/Regularizer/SumSum*stream_0_conv_3/kernel/Regularizer/Abs:y:03stream_0_conv_3/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_3/kernel/Regularizer/SumЩ
(stream_0_conv_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2*
(stream_0_conv_3/kernel/Regularizer/mul/x№
&stream_0_conv_3/kernel/Regularizer/mulMul1stream_0_conv_3/kernel/Regularizer/mul/x:output:0/stream_0_conv_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_3/kernel/Regularizer/mulў
&stream_0_conv_3/kernel/Regularizer/addAddV21stream_0_conv_3/kernel/Regularizer/Const:output:0*stream_0_conv_3/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&stream_0_conv_3/kernel/Regularizer/addд
8stream_0_conv_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype02:
8stream_0_conv_3/kernel/Regularizer/Square/ReadVariableOpѕ
)stream_0_conv_3/kernel/Regularizer/SquareSquare@stream_0_conv_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@@2+
)stream_0_conv_3/kernel/Regularizer/Square≠
*stream_0_conv_3/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_0_conv_3/kernel/Regularizer/Const_2а
(stream_0_conv_3/kernel/Regularizer/Sum_1Sum-stream_0_conv_3/kernel/Regularizer/Square:y:03stream_0_conv_3/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2*
(stream_0_conv_3/kernel/Regularizer/Sum_1Э
*stream_0_conv_3/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2,
*stream_0_conv_3/kernel/Regularizer/mul_1/xд
(stream_0_conv_3/kernel/Regularizer/mul_1Mul3stream_0_conv_3/kernel/Regularizer/mul_1/x:output:01stream_0_conv_3/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2*
(stream_0_conv_3/kernel/Regularizer/mul_1Ў
(stream_0_conv_3/kernel/Regularizer/add_1AddV2*stream_0_conv_3/kernel/Regularizer/add:z:0,stream_0_conv_3/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2*
(stream_0_conv_3/kernel/Regularizer/add_1o
IdentityIdentityBiasAdd:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€@2

Identity€
NoOpNoOp^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp6^stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp9^stream_0_conv_3/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp2n
5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp2t
8stream_0_conv_3/kernel/Regularizer/Square/ReadVariableOp8stream_0_conv_3/kernel/Regularizer/Square/ReadVariableOp:S O
+
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
з
e
I__inference_activation_2_layer_call_and_return_conditional_losses_3293120

inputs
identityR
TanhTanhinputs*
T0*+
_output_shapes
:€€€€€€€€€@2
Tanh`
IdentityIdentityTanh:y:0*
T0*+
_output_shapes
:€€€€€€€€€@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€@:S O
+
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
Ъ
г
+__inference_basemodel_layer_call_fn_3296320

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:@
	unknown_5:@@
	unknown_6:@
	unknown_7:@
	unknown_8:@
	unknown_9:@

unknown_10:@ 

unknown_11:@@

unknown_12:@

unknown_13:@

unknown_14:@

unknown_15:@

unknown_16:@

unknown_17:@@

unknown_18:@

unknown_19:@

unknown_20:@

unknown_21:@

unknown_22:@
identityИҐStatefulPartitionedCallЬ
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
:€€€€€€€€€@*2
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_basemodel_layer_call_and_return_conditional_losses_32938692
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:€€€€€€€€€}: : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:€€€€€€€€€}
 
_user_specified_nameinputs
Н	
–
5__inference_batch_normalization_layer_call_fn_3297442

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityИҐStatefulPartitionedCall™
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
GPU2*0J 8В *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_32921282
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
є+
л
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_3292568

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
П	
“
7__inference_batch_normalization_2_layer_call_fn_3298009

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityИҐStatefulPartitionedCall™
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
GPU2*0J 8В *[
fVRT
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_32925682
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
Ѕ
j
1__inference_stream_0_drop_1_layer_call_fn_3297635

inputs
identityИҐStatefulPartitionedCallй
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€>@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_stream_0_drop_1_layer_call_and_return_conditional_losses_32935742
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€>@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€>@22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:€€€€€€€€€>@
 
_user_specified_nameinputs
†
е
+__inference_basemodel_layer_call_fn_3293973
inputs_0
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:@
	unknown_5:@@
	unknown_6:@
	unknown_7:@
	unknown_8:@
	unknown_9:@

unknown_10:@ 

unknown_11:@@

unknown_12:@

unknown_13:@

unknown_14:@

unknown_15:@

unknown_16:@

unknown_17:@@

unknown_18:@

unknown_19:@

unknown_20:@

unknown_21:@

unknown_22:@
identityИҐStatefulPartitionedCallЮ
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
:€€€€€€€€€@*2
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_basemodel_layer_call_and_return_conditional_losses_32938692
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:€€€€€€€€€}: : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
+
_output_shapes
:€€€€€€€€€}
"
_user_specified_name
inputs_0
В+
л
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_3293533

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
:€€€€€€€€€>@2
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
:€€€€€€€€€>@2
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
:€€€€€€€€€>@2
batchnorm/add_1r
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€>@2

Identityт
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€>@: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:S O
+
_output_shapes
:€€€€€€€€€>@
 
_user_specified_nameinputs
є+
л
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_3292378

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
ш
щ
__inference_loss_fn_1_3298425T
>stream_0_conv_2_kernel_regularizer_abs_readvariableop_resource:@@
identityИҐ5stream_0_conv_2/kernel/Regularizer/Abs/ReadVariableOpҐ8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOpЩ
(stream_0_conv_2/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(stream_0_conv_2/kernel/Regularizer/Constс
5stream_0_conv_2/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp>stream_0_conv_2_kernel_regularizer_abs_readvariableop_resource*"
_output_shapes
:@@*
dtype027
5stream_0_conv_2/kernel/Regularizer/Abs/ReadVariableOp√
&stream_0_conv_2/kernel/Regularizer/AbsAbs=stream_0_conv_2/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@@2(
&stream_0_conv_2/kernel/Regularizer/Abs≠
*stream_0_conv_2/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_0_conv_2/kernel/Regularizer/Const_1ў
&stream_0_conv_2/kernel/Regularizer/SumSum*stream_0_conv_2/kernel/Regularizer/Abs:y:03stream_0_conv_2/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_2/kernel/Regularizer/SumЩ
(stream_0_conv_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2*
(stream_0_conv_2/kernel/Regularizer/mul/x№
&stream_0_conv_2/kernel/Regularizer/mulMul1stream_0_conv_2/kernel/Regularizer/mul/x:output:0/stream_0_conv_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_2/kernel/Regularizer/mulў
&stream_0_conv_2/kernel/Regularizer/addAddV21stream_0_conv_2/kernel/Regularizer/Const:output:0*stream_0_conv_2/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&stream_0_conv_2/kernel/Regularizer/addч
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp>stream_0_conv_2_kernel_regularizer_abs_readvariableop_resource*"
_output_shapes
:@@*
dtype02:
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOpѕ
)stream_0_conv_2/kernel/Regularizer/SquareSquare@stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@@2+
)stream_0_conv_2/kernel/Regularizer/Square≠
*stream_0_conv_2/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_0_conv_2/kernel/Regularizer/Const_2а
(stream_0_conv_2/kernel/Regularizer/Sum_1Sum-stream_0_conv_2/kernel/Regularizer/Square:y:03stream_0_conv_2/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2*
(stream_0_conv_2/kernel/Regularizer/Sum_1Э
*stream_0_conv_2/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2,
*stream_0_conv_2/kernel/Regularizer/mul_1/xд
(stream_0_conv_2/kernel/Regularizer/mul_1Mul3stream_0_conv_2/kernel/Regularizer/mul_1/x:output:01stream_0_conv_2/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2*
(stream_0_conv_2/kernel/Regularizer/mul_1Ў
(stream_0_conv_2/kernel/Regularizer/add_1AddV2*stream_0_conv_2/kernel/Regularizer/add:z:0,stream_0_conv_2/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2*
(stream_0_conv_2/kernel/Regularizer/add_1v
IdentityIdentity,stream_0_conv_2/kernel/Regularizer/add_1:z:0^NoOp*
T0*
_output_shapes
: 2

IdentityЅ
NoOpNoOp6^stream_0_conv_2/kernel/Regularizer/Abs/ReadVariableOp9^stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2n
5stream_0_conv_2/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_2/kernel/Regularizer/Abs/ReadVariableOp2t
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp
Ѕ
j
1__inference_stream_0_drop_3_layer_call_fn_3298189

inputs
identityИҐStatefulPartitionedCallй
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_stream_0_drop_3_layer_call_and_return_conditional_losses_32933662
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€@22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
я
M
1__inference_stream_0_drop_3_layer_call_fn_3298184

inputs
identity—
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_stream_0_drop_3_layer_call_and_return_conditional_losses_32931362
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:€€€€€€€€€@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€@:S O
+
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
з
e
I__inference_activation_1_layer_call_and_return_conditional_losses_3297876

inputs
identityR
TanhTanhinputs*
T0*+
_output_shapes
:€€€€€€€€€>@2
Tanh`
IdentityIdentityTanh:y:0*
T0*+
_output_shapes
:€€€€€€€€€>@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€>@:S O
+
_output_shapes
:€€€€€€€€€>@
 
_user_specified_nameinputs
ї
q
U__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_3298222

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
е
P
4__inference_stream_0_maxpool_3_layer_call_fn_3298163

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
:€€€€€€€€€@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_stream_0_maxpool_3_layer_call_and_return_conditional_losses_32931292
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:€€€€€€€€€@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€@:S O
+
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
ш
щ
__inference_loss_fn_2_3298445T
>stream_0_conv_3_kernel_regularizer_abs_readvariableop_resource:@@
identityИҐ5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOpҐ8stream_0_conv_3/kernel/Regularizer/Square/ReadVariableOpЩ
(stream_0_conv_3/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(stream_0_conv_3/kernel/Regularizer/Constс
5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp>stream_0_conv_3_kernel_regularizer_abs_readvariableop_resource*"
_output_shapes
:@@*
dtype027
5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp√
&stream_0_conv_3/kernel/Regularizer/AbsAbs=stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@@2(
&stream_0_conv_3/kernel/Regularizer/Abs≠
*stream_0_conv_3/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_0_conv_3/kernel/Regularizer/Const_1ў
&stream_0_conv_3/kernel/Regularizer/SumSum*stream_0_conv_3/kernel/Regularizer/Abs:y:03stream_0_conv_3/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_3/kernel/Regularizer/SumЩ
(stream_0_conv_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2*
(stream_0_conv_3/kernel/Regularizer/mul/x№
&stream_0_conv_3/kernel/Regularizer/mulMul1stream_0_conv_3/kernel/Regularizer/mul/x:output:0/stream_0_conv_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_3/kernel/Regularizer/mulў
&stream_0_conv_3/kernel/Regularizer/addAddV21stream_0_conv_3/kernel/Regularizer/Const:output:0*stream_0_conv_3/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&stream_0_conv_3/kernel/Regularizer/addч
8stream_0_conv_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOp>stream_0_conv_3_kernel_regularizer_abs_readvariableop_resource*"
_output_shapes
:@@*
dtype02:
8stream_0_conv_3/kernel/Regularizer/Square/ReadVariableOpѕ
)stream_0_conv_3/kernel/Regularizer/SquareSquare@stream_0_conv_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@@2+
)stream_0_conv_3/kernel/Regularizer/Square≠
*stream_0_conv_3/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_0_conv_3/kernel/Regularizer/Const_2а
(stream_0_conv_3/kernel/Regularizer/Sum_1Sum-stream_0_conv_3/kernel/Regularizer/Square:y:03stream_0_conv_3/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2*
(stream_0_conv_3/kernel/Regularizer/Sum_1Э
*stream_0_conv_3/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2,
*stream_0_conv_3/kernel/Regularizer/mul_1/xд
(stream_0_conv_3/kernel/Regularizer/mul_1Mul3stream_0_conv_3/kernel/Regularizer/mul_1/x:output:01stream_0_conv_3/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2*
(stream_0_conv_3/kernel/Regularizer/mul_1Ў
(stream_0_conv_3/kernel/Regularizer/add_1AddV2*stream_0_conv_3/kernel/Regularizer/add:z:0,stream_0_conv_3/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2*
(stream_0_conv_3/kernel/Regularizer/add_1v
IdentityIdentity,stream_0_conv_3/kernel/Regularizer/add_1:z:0^NoOp*
T0*
_output_shapes
: 2

IdentityЅ
NoOpNoOp6^stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp9^stream_0_conv_3/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2n
5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp2t
8stream_0_conv_3/kernel/Regularizer/Square/ReadVariableOp8stream_0_conv_3/kernel/Regularizer/Square/ReadVariableOp
ї
q
U__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_3292684

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
Й
j
L__inference_stream_0_drop_2_layer_call_and_return_conditional_losses_3293048

inputs

identity_1^
IdentityIdentityinputs*
T0*+
_output_shapes
:€€€€€€€€€@2

Identitym

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:€€€€€€€€€@2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€@:S O
+
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
П	
“
7__inference_batch_normalization_1_layer_call_fn_3297732

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityИҐStatefulPartitionedCall™
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
GPU2*0J 8В *[
fVRT
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_32923782
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
Й
j
L__inference_stream_0_drop_1_layer_call_and_return_conditional_losses_3297640

inputs

identity_1^
IdentityIdentityinputs*
T0*+
_output_shapes
:€€€€€€€€€>@2

Identitym

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:€€€€€€€€€>@2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€>@:S O
+
_output_shapes
:€€€€€€€€€>@
 
_user_specified_nameinputs
©
k
O__inference_stream_0_maxpool_3_layer_call_and_return_conditional_losses_3298179

inputs
identityb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dimБ

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*/
_output_shapes
:€€€€€€€€€@2

ExpandDimsЯ
MaxPoolMaxPoolExpandDims:output:0*/
_output_shapes
:€€€€€€€€€@*
ksize
*
paddingVALID*
strides
2	
MaxPool|
SqueezeSqueezeMaxPool:output:0*
T0*+
_output_shapes
:€€€€€€€€€@*
squeeze_dims
2	
Squeezeh
IdentityIdentitySqueeze:output:0*
T0*+
_output_shapes
:€€€€€€€€€@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€@:S O
+
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
щ
j
L__inference_dense_1_dropout_layer_call_and_return_conditional_losses_3298243

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:€€€€€€€€€@2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:€€€€€€€€€@:O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
÷љ
”
F__inference_basemodel_layer_call_and_return_conditional_losses_3294434

inputsQ
;stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource:@=
/stream_0_conv_1_biasadd_readvariableop_resource:@C
5batch_normalization_batchnorm_readvariableop_resource:@G
9batch_normalization_batchnorm_mul_readvariableop_resource:@E
7batch_normalization_batchnorm_readvariableop_1_resource:@E
7batch_normalization_batchnorm_readvariableop_2_resource:@Q
;stream_0_conv_2_conv1d_expanddims_1_readvariableop_resource:@@=
/stream_0_conv_2_biasadd_readvariableop_resource:@E
7batch_normalization_1_batchnorm_readvariableop_resource:@I
;batch_normalization_1_batchnorm_mul_readvariableop_resource:@G
9batch_normalization_1_batchnorm_readvariableop_1_resource:@G
9batch_normalization_1_batchnorm_readvariableop_2_resource:@Q
;stream_0_conv_3_conv1d_expanddims_1_readvariableop_resource:@@=
/stream_0_conv_3_biasadd_readvariableop_resource:@E
7batch_normalization_2_batchnorm_readvariableop_resource:@I
;batch_normalization_2_batchnorm_mul_readvariableop_resource:@G
9batch_normalization_2_batchnorm_readvariableop_1_resource:@G
9batch_normalization_2_batchnorm_readvariableop_2_resource:@8
&dense_1_matmul_readvariableop_resource:@@5
'dense_1_biasadd_readvariableop_resource:@E
7batch_normalization_3_batchnorm_readvariableop_resource:@I
;batch_normalization_3_batchnorm_mul_readvariableop_resource:@G
9batch_normalization_3_batchnorm_readvariableop_1_resource:@G
9batch_normalization_3_batchnorm_readvariableop_2_resource:@
identityИҐ,batch_normalization/batchnorm/ReadVariableOpҐ.batch_normalization/batchnorm/ReadVariableOp_1Ґ.batch_normalization/batchnorm/ReadVariableOp_2Ґ0batch_normalization/batchnorm/mul/ReadVariableOpҐ.batch_normalization_1/batchnorm/ReadVariableOpҐ0batch_normalization_1/batchnorm/ReadVariableOp_1Ґ0batch_normalization_1/batchnorm/ReadVariableOp_2Ґ2batch_normalization_1/batchnorm/mul/ReadVariableOpҐ.batch_normalization_2/batchnorm/ReadVariableOpҐ0batch_normalization_2/batchnorm/ReadVariableOp_1Ґ0batch_normalization_2/batchnorm/ReadVariableOp_2Ґ2batch_normalization_2/batchnorm/mul/ReadVariableOpҐ.batch_normalization_3/batchnorm/ReadVariableOpҐ0batch_normalization_3/batchnorm/ReadVariableOp_1Ґ0batch_normalization_3/batchnorm/ReadVariableOp_2Ґ2batch_normalization_3/batchnorm/mul/ReadVariableOpҐdense_1/BiasAdd/ReadVariableOpҐdense_1/MatMul/ReadVariableOpҐ-dense_1/kernel/Regularizer/Abs/ReadVariableOpҐ0dense_1/kernel/Regularizer/Square/ReadVariableOpҐ&stream_0_conv_1/BiasAdd/ReadVariableOpҐ2stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOpҐ5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpҐ8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOpҐ&stream_0_conv_2/BiasAdd/ReadVariableOpҐ2stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOpҐ5stream_0_conv_2/kernel/Regularizer/Abs/ReadVariableOpҐ8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOpҐ&stream_0_conv_3/BiasAdd/ReadVariableOpҐ2stream_0_conv_3/conv1d/ExpandDims_1/ReadVariableOpҐ5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOpҐ8stream_0_conv_3/kernel/Regularizer/Square/ReadVariableOpЖ
stream_0_input_drop/IdentityIdentityinputs*
T0*+
_output_shapes
:€€€€€€€€€}2
stream_0_input_drop/IdentityЩ
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
stream_0_conv_1/BiasAddќ
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
#batch_normalization/batchnorm/add_1Й
activation/TanhTanh'batch_normalization/batchnorm/add_1:z:0*
T0*+
_output_shapes
:€€€€€€€€€}@2
activation/TanhИ
!stream_0_maxpool_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!stream_0_maxpool_1/ExpandDims/dim«
stream_0_maxpool_1/ExpandDims
ExpandDimsactivation/Tanh:y:0*stream_0_maxpool_1/ExpandDims/dim:output:0*
T0*/
_output_shapes
:€€€€€€€€€}@2
stream_0_maxpool_1/ExpandDimsЎ
stream_0_maxpool_1/MaxPoolMaxPool&stream_0_maxpool_1/ExpandDims:output:0*/
_output_shapes
:€€€€€€€€€>@*
ksize
*
paddingVALID*
strides
2
stream_0_maxpool_1/MaxPoolµ
stream_0_maxpool_1/SqueezeSqueeze#stream_0_maxpool_1/MaxPool:output:0*
T0*+
_output_shapes
:€€€€€€€€€>@*
squeeze_dims
2
stream_0_maxpool_1/SqueezeЫ
stream_0_drop_1/IdentityIdentity#stream_0_maxpool_1/Squeeze:output:0*
T0*+
_output_shapes
:€€€€€€€€€>@2
stream_0_drop_1/IdentityЩ
%stream_0_conv_2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€2'
%stream_0_conv_2/conv1d/ExpandDims/dimб
!stream_0_conv_2/conv1d/ExpandDims
ExpandDims!stream_0_drop_1/Identity:output:0.stream_0_conv_2/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:€€€€€€€€€>@2#
!stream_0_conv_2/conv1d/ExpandDimsи
2stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp;stream_0_conv_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype024
2stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOpФ
'stream_0_conv_2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2)
'stream_0_conv_2/conv1d/ExpandDims_1/dimч
#stream_0_conv_2/conv1d/ExpandDims_1
ExpandDims:stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOp:value:00stream_0_conv_2/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@2%
#stream_0_conv_2/conv1d/ExpandDims_1ц
stream_0_conv_2/conv1dConv2D*stream_0_conv_2/conv1d/ExpandDims:output:0,stream_0_conv_2/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:€€€€€€€€€>@*
paddingSAME*
strides
2
stream_0_conv_2/conv1d¬
stream_0_conv_2/conv1d/SqueezeSqueezestream_0_conv_2/conv1d:output:0*
T0*+
_output_shapes
:€€€€€€€€€>@*
squeeze_dims

э€€€€€€€€2 
stream_0_conv_2/conv1d/SqueezeЉ
&stream_0_conv_2/BiasAdd/ReadVariableOpReadVariableOp/stream_0_conv_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02(
&stream_0_conv_2/BiasAdd/ReadVariableOpћ
stream_0_conv_2/BiasAddBiasAdd'stream_0_conv_2/conv1d/Squeeze:output:0.stream_0_conv_2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€>@2
stream_0_conv_2/BiasAdd‘
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
%batch_normalization_1/batchnorm/mul_1Mul stream_0_conv_2/BiasAdd:output:0'batch_normalization_1/batchnorm/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€>@2'
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
:€€€€€€€€€>@2'
%batch_normalization_1/batchnorm/add_1П
activation_1/TanhTanh)batch_normalization_1/batchnorm/add_1:z:0*
T0*+
_output_shapes
:€€€€€€€€€>@2
activation_1/TanhИ
!stream_0_maxpool_2/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!stream_0_maxpool_2/ExpandDims/dim…
stream_0_maxpool_2/ExpandDims
ExpandDimsactivation_1/Tanh:y:0*stream_0_maxpool_2/ExpandDims/dim:output:0*
T0*/
_output_shapes
:€€€€€€€€€>@2
stream_0_maxpool_2/ExpandDimsЎ
stream_0_maxpool_2/MaxPoolMaxPool&stream_0_maxpool_2/ExpandDims:output:0*/
_output_shapes
:€€€€€€€€€@*
ksize
*
paddingVALID*
strides
2
stream_0_maxpool_2/MaxPoolµ
stream_0_maxpool_2/SqueezeSqueeze#stream_0_maxpool_2/MaxPool:output:0*
T0*+
_output_shapes
:€€€€€€€€€@*
squeeze_dims
2
stream_0_maxpool_2/SqueezeЫ
stream_0_drop_2/IdentityIdentity#stream_0_maxpool_2/Squeeze:output:0*
T0*+
_output_shapes
:€€€€€€€€€@2
stream_0_drop_2/IdentityЩ
%stream_0_conv_3/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€2'
%stream_0_conv_3/conv1d/ExpandDims/dimб
!stream_0_conv_3/conv1d/ExpandDims
ExpandDims!stream_0_drop_2/Identity:output:0.stream_0_conv_3/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:€€€€€€€€€@2#
!stream_0_conv_3/conv1d/ExpandDimsи
2stream_0_conv_3/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp;stream_0_conv_3_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype024
2stream_0_conv_3/conv1d/ExpandDims_1/ReadVariableOpФ
'stream_0_conv_3/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2)
'stream_0_conv_3/conv1d/ExpandDims_1/dimч
#stream_0_conv_3/conv1d/ExpandDims_1
ExpandDims:stream_0_conv_3/conv1d/ExpandDims_1/ReadVariableOp:value:00stream_0_conv_3/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@2%
#stream_0_conv_3/conv1d/ExpandDims_1ц
stream_0_conv_3/conv1dConv2D*stream_0_conv_3/conv1d/ExpandDims:output:0,stream_0_conv_3/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:€€€€€€€€€@*
paddingSAME*
strides
2
stream_0_conv_3/conv1d¬
stream_0_conv_3/conv1d/SqueezeSqueezestream_0_conv_3/conv1d:output:0*
T0*+
_output_shapes
:€€€€€€€€€@*
squeeze_dims

э€€€€€€€€2 
stream_0_conv_3/conv1d/SqueezeЉ
&stream_0_conv_3/BiasAdd/ReadVariableOpReadVariableOp/stream_0_conv_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02(
&stream_0_conv_3/BiasAdd/ReadVariableOpћ
stream_0_conv_3/BiasAddBiasAdd'stream_0_conv_3/conv1d/Squeeze:output:0.stream_0_conv_3/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€@2
stream_0_conv_3/BiasAdd‘
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
%batch_normalization_2/batchnorm/mul_1Mul stream_0_conv_3/BiasAdd:output:0'batch_normalization_2/batchnorm/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€@2'
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
:€€€€€€€€€@2'
%batch_normalization_2/batchnorm/add_1П
activation_2/TanhTanh)batch_normalization_2/batchnorm/add_1:z:0*
T0*+
_output_shapes
:€€€€€€€€€@2
activation_2/TanhИ
!stream_0_maxpool_3/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!stream_0_maxpool_3/ExpandDims/dim…
stream_0_maxpool_3/ExpandDims
ExpandDimsactivation_2/Tanh:y:0*stream_0_maxpool_3/ExpandDims/dim:output:0*
T0*/
_output_shapes
:€€€€€€€€€@2
stream_0_maxpool_3/ExpandDimsЎ
stream_0_maxpool_3/MaxPoolMaxPool&stream_0_maxpool_3/ExpandDims:output:0*/
_output_shapes
:€€€€€€€€€@*
ksize
*
paddingVALID*
strides
2
stream_0_maxpool_3/MaxPoolµ
stream_0_maxpool_3/SqueezeSqueeze#stream_0_maxpool_3/MaxPool:output:0*
T0*+
_output_shapes
:€€€€€€€€€@*
squeeze_dims
2
stream_0_maxpool_3/SqueezeЫ
stream_0_drop_3/IdentityIdentity#stream_0_maxpool_3/Squeeze:output:0*
T0*+
_output_shapes
:€€€€€€€€€@2
stream_0_drop_3/Identity§
/global_average_pooling1d/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :21
/global_average_pooling1d/Mean/reduction_indices’
global_average_pooling1d/MeanMean!stream_0_drop_3/Identity:output:08global_average_pooling1d/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
global_average_pooling1d/MeanЪ
dense_1_dropout/IdentityIdentity&global_average_pooling1d/Mean:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
dense_1_dropout/Identity•
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02
dense_1/MatMul/ReadVariableOp¶
dense_1/MatMulMatMul!dense_1_dropout/Identity:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2
dense_1/MatMul§
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02 
dense_1/BiasAdd/ReadVariableOp°
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2
dense_1/BiasAdd‘
.batch_normalization_3/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_3_batchnorm_readvariableop_resource*
_output_shapes
:@*
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
:@2%
#batch_normalization_3/batchnorm/add•
%batch_normalization_3/batchnorm/RsqrtRsqrt'batch_normalization_3/batchnorm/add:z:0*
T0*
_output_shapes
:@2'
%batch_normalization_3/batchnorm/Rsqrtа
2batch_normalization_3/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_3_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype024
2batch_normalization_3/batchnorm/mul/ReadVariableOpЁ
#batch_normalization_3/batchnorm/mulMul)batch_normalization_3/batchnorm/Rsqrt:y:0:batch_normalization_3/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2%
#batch_normalization_3/batchnorm/mul 
%batch_normalization_3/batchnorm/mul_1Muldense_1/BiasAdd:output:0'batch_normalization_3/batchnorm/mul:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2'
%batch_normalization_3/batchnorm/mul_1Џ
0batch_normalization_3/batchnorm/ReadVariableOp_1ReadVariableOp9batch_normalization_3_batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype022
0batch_normalization_3/batchnorm/ReadVariableOp_1Ё
%batch_normalization_3/batchnorm/mul_2Mul8batch_normalization_3/batchnorm/ReadVariableOp_1:value:0'batch_normalization_3/batchnorm/mul:z:0*
T0*
_output_shapes
:@2'
%batch_normalization_3/batchnorm/mul_2Џ
0batch_normalization_3/batchnorm/ReadVariableOp_2ReadVariableOp9batch_normalization_3_batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype022
0batch_normalization_3/batchnorm/ReadVariableOp_2џ
#batch_normalization_3/batchnorm/subSub8batch_normalization_3/batchnorm/ReadVariableOp_2:value:0)batch_normalization_3/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2%
#batch_normalization_3/batchnorm/subЁ
%batch_normalization_3/batchnorm/add_1AddV2)batch_normalization_3/batchnorm/mul_1:z:0'batch_normalization_3/batchnorm/sub:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2'
%batch_normalization_3/batchnorm/add_1Щ
(stream_0_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(stream_0_conv_1/kernel/Regularizer/Constо
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp;stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype027
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp√
&stream_0_conv_1/kernel/Regularizer/AbsAbs=stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_0_conv_1/kernel/Regularizer/Abs≠
*stream_0_conv_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_0_conv_1/kernel/Regularizer/Const_1ў
&stream_0_conv_1/kernel/Regularizer/SumSum*stream_0_conv_1/kernel/Regularizer/Abs:y:03stream_0_conv_1/kernel/Regularizer/Const_1:output:0*
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
&stream_0_conv_1/kernel/Regularizer/mulў
&stream_0_conv_1/kernel/Regularizer/addAddV21stream_0_conv_1/kernel/Regularizer/Const:output:0*stream_0_conv_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/addф
8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02:
8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOpѕ
)stream_0_conv_1/kernel/Regularizer/SquareSquare@stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2+
)stream_0_conv_1/kernel/Regularizer/Square≠
*stream_0_conv_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_0_conv_1/kernel/Regularizer/Const_2а
(stream_0_conv_1/kernel/Regularizer/Sum_1Sum-stream_0_conv_1/kernel/Regularizer/Square:y:03stream_0_conv_1/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2*
(stream_0_conv_1/kernel/Regularizer/Sum_1Э
*stream_0_conv_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2,
*stream_0_conv_1/kernel/Regularizer/mul_1/xд
(stream_0_conv_1/kernel/Regularizer/mul_1Mul3stream_0_conv_1/kernel/Regularizer/mul_1/x:output:01stream_0_conv_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2*
(stream_0_conv_1/kernel/Regularizer/mul_1Ў
(stream_0_conv_1/kernel/Regularizer/add_1AddV2*stream_0_conv_1/kernel/Regularizer/add:z:0,stream_0_conv_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2*
(stream_0_conv_1/kernel/Regularizer/add_1Щ
(stream_0_conv_2/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(stream_0_conv_2/kernel/Regularizer/Constо
5stream_0_conv_2/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp;stream_0_conv_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype027
5stream_0_conv_2/kernel/Regularizer/Abs/ReadVariableOp√
&stream_0_conv_2/kernel/Regularizer/AbsAbs=stream_0_conv_2/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@@2(
&stream_0_conv_2/kernel/Regularizer/Abs≠
*stream_0_conv_2/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_0_conv_2/kernel/Regularizer/Const_1ў
&stream_0_conv_2/kernel/Regularizer/SumSum*stream_0_conv_2/kernel/Regularizer/Abs:y:03stream_0_conv_2/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_2/kernel/Regularizer/SumЩ
(stream_0_conv_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2*
(stream_0_conv_2/kernel/Regularizer/mul/x№
&stream_0_conv_2/kernel/Regularizer/mulMul1stream_0_conv_2/kernel/Regularizer/mul/x:output:0/stream_0_conv_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_2/kernel/Regularizer/mulў
&stream_0_conv_2/kernel/Regularizer/addAddV21stream_0_conv_2/kernel/Regularizer/Const:output:0*stream_0_conv_2/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&stream_0_conv_2/kernel/Regularizer/addф
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;stream_0_conv_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype02:
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOpѕ
)stream_0_conv_2/kernel/Regularizer/SquareSquare@stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@@2+
)stream_0_conv_2/kernel/Regularizer/Square≠
*stream_0_conv_2/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_0_conv_2/kernel/Regularizer/Const_2а
(stream_0_conv_2/kernel/Regularizer/Sum_1Sum-stream_0_conv_2/kernel/Regularizer/Square:y:03stream_0_conv_2/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2*
(stream_0_conv_2/kernel/Regularizer/Sum_1Э
*stream_0_conv_2/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2,
*stream_0_conv_2/kernel/Regularizer/mul_1/xд
(stream_0_conv_2/kernel/Regularizer/mul_1Mul3stream_0_conv_2/kernel/Regularizer/mul_1/x:output:01stream_0_conv_2/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2*
(stream_0_conv_2/kernel/Regularizer/mul_1Ў
(stream_0_conv_2/kernel/Regularizer/add_1AddV2*stream_0_conv_2/kernel/Regularizer/add:z:0,stream_0_conv_2/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2*
(stream_0_conv_2/kernel/Regularizer/add_1Щ
(stream_0_conv_3/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(stream_0_conv_3/kernel/Regularizer/Constо
5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp;stream_0_conv_3_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype027
5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp√
&stream_0_conv_3/kernel/Regularizer/AbsAbs=stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@@2(
&stream_0_conv_3/kernel/Regularizer/Abs≠
*stream_0_conv_3/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_0_conv_3/kernel/Regularizer/Const_1ў
&stream_0_conv_3/kernel/Regularizer/SumSum*stream_0_conv_3/kernel/Regularizer/Abs:y:03stream_0_conv_3/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_3/kernel/Regularizer/SumЩ
(stream_0_conv_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2*
(stream_0_conv_3/kernel/Regularizer/mul/x№
&stream_0_conv_3/kernel/Regularizer/mulMul1stream_0_conv_3/kernel/Regularizer/mul/x:output:0/stream_0_conv_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_3/kernel/Regularizer/mulў
&stream_0_conv_3/kernel/Regularizer/addAddV21stream_0_conv_3/kernel/Regularizer/Const:output:0*stream_0_conv_3/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&stream_0_conv_3/kernel/Regularizer/addф
8stream_0_conv_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;stream_0_conv_3_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype02:
8stream_0_conv_3/kernel/Regularizer/Square/ReadVariableOpѕ
)stream_0_conv_3/kernel/Regularizer/SquareSquare@stream_0_conv_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@@2+
)stream_0_conv_3/kernel/Regularizer/Square≠
*stream_0_conv_3/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_0_conv_3/kernel/Regularizer/Const_2а
(stream_0_conv_3/kernel/Regularizer/Sum_1Sum-stream_0_conv_3/kernel/Regularizer/Square:y:03stream_0_conv_3/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2*
(stream_0_conv_3/kernel/Regularizer/Sum_1Э
*stream_0_conv_3/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2,
*stream_0_conv_3/kernel/Regularizer/mul_1/xд
(stream_0_conv_3/kernel/Regularizer/mul_1Mul3stream_0_conv_3/kernel/Regularizer/mul_1/x:output:01stream_0_conv_3/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2*
(stream_0_conv_3/kernel/Regularizer/mul_1Ў
(stream_0_conv_3/kernel/Regularizer/add_1AddV2*stream_0_conv_3/kernel/Regularizer/add:z:0,stream_0_conv_3/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2*
(stream_0_conv_3/kernel/Regularizer/add_1Й
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dense_1/kernel/Regularizer/Const≈
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOpІ
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:@@2 
dense_1/kernel/Regularizer/AbsЩ
"dense_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_1/kernel/Regularizer/Const_1є
dense_1/kernel/Regularizer/SumSum"dense_1/kernel/Regularizer/Abs:y:0+dense_1/kernel/Regularizer/Const_1:output:0*
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
dense_1/kernel/Regularizer/mulє
dense_1/kernel/Regularizer/addAddV2)dense_1/kernel/Regularizer/Const:output:0"dense_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/addЋ
0dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype022
0dense_1/kernel/Regularizer/Square/ReadVariableOp≥
!dense_1/kernel/Regularizer/SquareSquare8dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@@2#
!dense_1/kernel/Regularizer/SquareЩ
"dense_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_1/kernel/Regularizer/Const_2ј
 dense_1/kernel/Regularizer/Sum_1Sum%dense_1/kernel/Regularizer/Square:y:0+dense_1/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2"
 dense_1/kernel/Regularizer/Sum_1Н
"dense_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2$
"dense_1/kernel/Regularizer/mul_1/xƒ
 dense_1/kernel/Regularizer/mul_1Mul+dense_1/kernel/Regularizer/mul_1/x:output:0)dense_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2"
 dense_1/kernel/Regularizer/mul_1Є
 dense_1/kernel/Regularizer/add_1AddV2"dense_1/kernel/Regularizer/add:z:0$dense_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2"
 dense_1/kernel/Regularizer/add_1Д
IdentityIdentity)batch_normalization_3/batchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@2

IdentityН
NoOpNoOp-^batch_normalization/batchnorm/ReadVariableOp/^batch_normalization/batchnorm/ReadVariableOp_1/^batch_normalization/batchnorm/ReadVariableOp_21^batch_normalization/batchnorm/mul/ReadVariableOp/^batch_normalization_1/batchnorm/ReadVariableOp1^batch_normalization_1/batchnorm/ReadVariableOp_11^batch_normalization_1/batchnorm/ReadVariableOp_23^batch_normalization_1/batchnorm/mul/ReadVariableOp/^batch_normalization_2/batchnorm/ReadVariableOp1^batch_normalization_2/batchnorm/ReadVariableOp_11^batch_normalization_2/batchnorm/ReadVariableOp_23^batch_normalization_2/batchnorm/mul/ReadVariableOp/^batch_normalization_3/batchnorm/ReadVariableOp1^batch_normalization_3/batchnorm/ReadVariableOp_11^batch_normalization_3/batchnorm/ReadVariableOp_23^batch_normalization_3/batchnorm/mul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp.^dense_1/kernel/Regularizer/Abs/ReadVariableOp1^dense_1/kernel/Regularizer/Square/ReadVariableOp'^stream_0_conv_1/BiasAdd/ReadVariableOp3^stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp9^stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOp'^stream_0_conv_2/BiasAdd/ReadVariableOp3^stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOp6^stream_0_conv_2/kernel/Regularizer/Abs/ReadVariableOp9^stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp'^stream_0_conv_3/BiasAdd/ReadVariableOp3^stream_0_conv_3/conv1d/ExpandDims_1/ReadVariableOp6^stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp9^stream_0_conv_3/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:€€€€€€€€€}: : : : : : : : : : : : : : : : : : : : : : : : 2\
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
-dense_1/kernel/Regularizer/Abs/ReadVariableOp-dense_1/kernel/Regularizer/Abs/ReadVariableOp2d
0dense_1/kernel/Regularizer/Square/ReadVariableOp0dense_1/kernel/Regularizer/Square/ReadVariableOp2P
&stream_0_conv_1/BiasAdd/ReadVariableOp&stream_0_conv_1/BiasAdd/ReadVariableOp2h
2stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp2stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp2n
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp2t
8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOp8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOp2P
&stream_0_conv_2/BiasAdd/ReadVariableOp&stream_0_conv_2/BiasAdd/ReadVariableOp2h
2stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOp2stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOp2n
5stream_0_conv_2/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_2/kernel/Regularizer/Abs/ReadVariableOp2t
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp2P
&stream_0_conv_3/BiasAdd/ReadVariableOp&stream_0_conv_3/BiasAdd/ReadVariableOp2h
2stream_0_conv_3/conv1d/ExpandDims_1/ReadVariableOp2stream_0_conv_3/conv1d/ExpandDims_1/ReadVariableOp2n
5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp2t
8stream_0_conv_3/kernel/Regularizer/Square/ReadVariableOp8stream_0_conv_3/kernel/Regularizer/Square/ReadVariableOp:S O
+
_output_shapes
:€€€€€€€€€}
 
_user_specified_nameinputs
В+
л
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_3297866

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
:€€€€€€€€€>@2
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
:€€€€€€€€€>@2
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
:€€€€€€€€€>@2
batchnorm/add_1r
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€>@2

Identityт
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€>@: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:S O
+
_output_shapes
:€€€€€€€€€>@
 
_user_specified_nameinputs
Ґ
г
+__inference_basemodel_layer_call_fn_3296267

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:@
	unknown_5:@@
	unknown_6:@
	unknown_7:@
	unknown_8:@
	unknown_9:@

unknown_10:@ 

unknown_11:@@

unknown_12:@

unknown_13:@

unknown_14:@

unknown_15:@

unknown_16:@

unknown_17:@@

unknown_18:@

unknown_19:@

unknown_20:@

unknown_21:@

unknown_22:@
identityИҐStatefulPartitionedCall§
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
:€€€€€€€€€@*:
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_basemodel_layer_call_and_return_conditional_losses_32932592
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:€€€€€€€€€}: : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:€€€€€€€€€}
 
_user_specified_nameinputs
Ц
k
O__inference_stream_0_maxpool_3_layer_call_and_return_conditional_losses_3292658

inputs
identityb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dimУ

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2

ExpandDims±
MaxPoolMaxPoolExpandDims:output:0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*
ksize
*
paddingVALID*
strides
2	
MaxPoolО
SqueezeSqueezeMaxPool:output:0*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€*
squeeze_dims
2	
Squeezez
IdentityIdentitySqueeze:output:0*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€:e a
=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
А+
й
P__inference_batch_normalization_layer_call_and_return_conditional_losses_3293637

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
Н
n
P__inference_stream_0_input_drop_layer_call_and_return_conditional_losses_3292872

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
о
k
L__inference_stream_0_drop_1_layer_call_and_return_conditional_losses_3297652

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *Ђ™™?2
dropout/Constw
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:€€€€€€€€€>@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape”
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:€€€€€€€€€>@*
dtype0*
seedЈ*
seed2Є2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А>2
dropout/GreaterEqual/y¬
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:€€€€€€€€€>@2
dropout/GreaterEqualГ
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:€€€€€€€€€>@2
dropout/Cast~
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:€€€€€€€€€>@2
dropout/Mul_1i
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:€€€€€€€€€>@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€>@:S O
+
_output_shapes
:€€€€€€€€€>@
 
_user_specified_nameinputs
…
n
5__inference_stream_0_input_drop_layer_call_fn_3297358

inputs
identityИҐStatefulPartitionedCallн
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
GPU2*0J 8В *Y
fTRR
P__inference_stream_0_input_drop_layer_call_and_return_conditional_losses_32936782
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
Ѓ
P
4__inference_stream_0_maxpool_2_layer_call_fn_3297881

inputs
identityж
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_stream_0_maxpool_2_layer_call_and_return_conditional_losses_32924682
PartitionedCallВ
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€:e a
=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
Ќ*
л
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_3292782

inputs5
'assignmovingavg_readvariableop_resource:@7
)assignmovingavg_1_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@/
!batchnorm_readvariableop_resource:@
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

:@*
	keep_dims(2
moments/mean|
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:@2
moments/StopGradient§
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
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

:@*
	keep_dims(2
moments/varianceА
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 2
moments/SqueezeИ
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:@*
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
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
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
batchnorm/subЕ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
batchnorm/add_1n
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@2

Identityт
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€@: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
©
k
O__inference_stream_0_maxpool_1_layer_call_and_return_conditional_losses_3297625

inputs
identityb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dimБ

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*/
_output_shapes
:€€€€€€€€€}@2

ExpandDimsЯ
MaxPoolMaxPoolExpandDims:output:0*/
_output_shapes
:€€€€€€€€€>@*
ksize
*
paddingVALID*
strides
2	
MaxPool|
SqueezeSqueezeMaxPool:output:0*
T0*+
_output_shapes
:€€€€€€€€€>@*
squeeze_dims
2	
Squeezeh
IdentityIdentitySqueeze:output:0*
T0*+
_output_shapes
:€€€€€€€€€>@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€}@:S O
+
_output_shapes
:€€€€€€€€€}@
 
_user_specified_nameinputs
Ю
б
)__inference_model_1_layer_call_fn_3295640

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:@
	unknown_5:@@
	unknown_6:@
	unknown_7:@
	unknown_8:@
	unknown_9:@

unknown_10:@ 

unknown_11:@@

unknown_12:@

unknown_13:@

unknown_14:@

unknown_15:@

unknown_16:@

unknown_17:@@

unknown_18:@

unknown_19:@

unknown_20:@

unknown_21:@

unknown_22:@
identityИҐStatefulPartitionedCallҐ
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
:€€€€€€€€€@*:
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_model_1_layer_call_and_return_conditional_losses_32945452
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:€€€€€€€€€}: : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:€€€€€€€€€}
 
_user_specified_nameinputs
мz
ь

D__inference_model_1_layer_call_and_return_conditional_losses_3295472
left_inputs'
basemodel_3295362:@
basemodel_3295364:@
basemodel_3295366:@
basemodel_3295368:@
basemodel_3295370:@
basemodel_3295372:@'
basemodel_3295374:@@
basemodel_3295376:@
basemodel_3295378:@
basemodel_3295380:@
basemodel_3295382:@
basemodel_3295384:@'
basemodel_3295386:@@
basemodel_3295388:@
basemodel_3295390:@
basemodel_3295392:@
basemodel_3295394:@
basemodel_3295396:@#
basemodel_3295398:@@
basemodel_3295400:@
basemodel_3295402:@
basemodel_3295404:@
basemodel_3295406:@
basemodel_3295408:@
identityИҐ!basemodel/StatefulPartitionedCallҐ-dense_1/kernel/Regularizer/Abs/ReadVariableOpҐ0dense_1/kernel/Regularizer/Square/ReadVariableOpҐ5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpҐ8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOpҐ5stream_0_conv_2/kernel/Regularizer/Abs/ReadVariableOpҐ8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOpҐ5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOpҐ8stream_0_conv_3/kernel/Regularizer/Square/ReadVariableOpк
!basemodel/StatefulPartitionedCallStatefulPartitionedCallleft_inputsbasemodel_3295362basemodel_3295364basemodel_3295366basemodel_3295368basemodel_3295370basemodel_3295372basemodel_3295374basemodel_3295376basemodel_3295378basemodel_3295380basemodel_3295382basemodel_3295384basemodel_3295386basemodel_3295388basemodel_3295390basemodel_3295392basemodel_3295394basemodel_3295396basemodel_3295398basemodel_3295400basemodel_3295402basemodel_3295404basemodel_3295406basemodel_3295408*$
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@*2
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_basemodel_layer_call_and_return_conditional_losses_32949232#
!basemodel/StatefulPartitionedCallЩ
(stream_0_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(stream_0_conv_1/kernel/Regularizer/Constƒ
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpbasemodel_3295362*"
_output_shapes
:@*
dtype027
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp√
&stream_0_conv_1/kernel/Regularizer/AbsAbs=stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_0_conv_1/kernel/Regularizer/Abs≠
*stream_0_conv_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_0_conv_1/kernel/Regularizer/Const_1ў
&stream_0_conv_1/kernel/Regularizer/SumSum*stream_0_conv_1/kernel/Regularizer/Abs:y:03stream_0_conv_1/kernel/Regularizer/Const_1:output:0*
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
&stream_0_conv_1/kernel/Regularizer/mulў
&stream_0_conv_1/kernel/Regularizer/addAddV21stream_0_conv_1/kernel/Regularizer/Const:output:0*stream_0_conv_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/add 
8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpbasemodel_3295362*"
_output_shapes
:@*
dtype02:
8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOpѕ
)stream_0_conv_1/kernel/Regularizer/SquareSquare@stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2+
)stream_0_conv_1/kernel/Regularizer/Square≠
*stream_0_conv_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_0_conv_1/kernel/Regularizer/Const_2а
(stream_0_conv_1/kernel/Regularizer/Sum_1Sum-stream_0_conv_1/kernel/Regularizer/Square:y:03stream_0_conv_1/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2*
(stream_0_conv_1/kernel/Regularizer/Sum_1Э
*stream_0_conv_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2,
*stream_0_conv_1/kernel/Regularizer/mul_1/xд
(stream_0_conv_1/kernel/Regularizer/mul_1Mul3stream_0_conv_1/kernel/Regularizer/mul_1/x:output:01stream_0_conv_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2*
(stream_0_conv_1/kernel/Regularizer/mul_1Ў
(stream_0_conv_1/kernel/Regularizer/add_1AddV2*stream_0_conv_1/kernel/Regularizer/add:z:0,stream_0_conv_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2*
(stream_0_conv_1/kernel/Regularizer/add_1Щ
(stream_0_conv_2/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(stream_0_conv_2/kernel/Regularizer/Constƒ
5stream_0_conv_2/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpbasemodel_3295374*"
_output_shapes
:@@*
dtype027
5stream_0_conv_2/kernel/Regularizer/Abs/ReadVariableOp√
&stream_0_conv_2/kernel/Regularizer/AbsAbs=stream_0_conv_2/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@@2(
&stream_0_conv_2/kernel/Regularizer/Abs≠
*stream_0_conv_2/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_0_conv_2/kernel/Regularizer/Const_1ў
&stream_0_conv_2/kernel/Regularizer/SumSum*stream_0_conv_2/kernel/Regularizer/Abs:y:03stream_0_conv_2/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_2/kernel/Regularizer/SumЩ
(stream_0_conv_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2*
(stream_0_conv_2/kernel/Regularizer/mul/x№
&stream_0_conv_2/kernel/Regularizer/mulMul1stream_0_conv_2/kernel/Regularizer/mul/x:output:0/stream_0_conv_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_2/kernel/Regularizer/mulў
&stream_0_conv_2/kernel/Regularizer/addAddV21stream_0_conv_2/kernel/Regularizer/Const:output:0*stream_0_conv_2/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&stream_0_conv_2/kernel/Regularizer/add 
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpbasemodel_3295374*"
_output_shapes
:@@*
dtype02:
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOpѕ
)stream_0_conv_2/kernel/Regularizer/SquareSquare@stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@@2+
)stream_0_conv_2/kernel/Regularizer/Square≠
*stream_0_conv_2/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_0_conv_2/kernel/Regularizer/Const_2а
(stream_0_conv_2/kernel/Regularizer/Sum_1Sum-stream_0_conv_2/kernel/Regularizer/Square:y:03stream_0_conv_2/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2*
(stream_0_conv_2/kernel/Regularizer/Sum_1Э
*stream_0_conv_2/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2,
*stream_0_conv_2/kernel/Regularizer/mul_1/xд
(stream_0_conv_2/kernel/Regularizer/mul_1Mul3stream_0_conv_2/kernel/Regularizer/mul_1/x:output:01stream_0_conv_2/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2*
(stream_0_conv_2/kernel/Regularizer/mul_1Ў
(stream_0_conv_2/kernel/Regularizer/add_1AddV2*stream_0_conv_2/kernel/Regularizer/add:z:0,stream_0_conv_2/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2*
(stream_0_conv_2/kernel/Regularizer/add_1Щ
(stream_0_conv_3/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(stream_0_conv_3/kernel/Regularizer/Constƒ
5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpbasemodel_3295386*"
_output_shapes
:@@*
dtype027
5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp√
&stream_0_conv_3/kernel/Regularizer/AbsAbs=stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@@2(
&stream_0_conv_3/kernel/Regularizer/Abs≠
*stream_0_conv_3/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_0_conv_3/kernel/Regularizer/Const_1ў
&stream_0_conv_3/kernel/Regularizer/SumSum*stream_0_conv_3/kernel/Regularizer/Abs:y:03stream_0_conv_3/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_3/kernel/Regularizer/SumЩ
(stream_0_conv_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2*
(stream_0_conv_3/kernel/Regularizer/mul/x№
&stream_0_conv_3/kernel/Regularizer/mulMul1stream_0_conv_3/kernel/Regularizer/mul/x:output:0/stream_0_conv_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_3/kernel/Regularizer/mulў
&stream_0_conv_3/kernel/Regularizer/addAddV21stream_0_conv_3/kernel/Regularizer/Const:output:0*stream_0_conv_3/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&stream_0_conv_3/kernel/Regularizer/add 
8stream_0_conv_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpbasemodel_3295386*"
_output_shapes
:@@*
dtype02:
8stream_0_conv_3/kernel/Regularizer/Square/ReadVariableOpѕ
)stream_0_conv_3/kernel/Regularizer/SquareSquare@stream_0_conv_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@@2+
)stream_0_conv_3/kernel/Regularizer/Square≠
*stream_0_conv_3/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_0_conv_3/kernel/Regularizer/Const_2а
(stream_0_conv_3/kernel/Regularizer/Sum_1Sum-stream_0_conv_3/kernel/Regularizer/Square:y:03stream_0_conv_3/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2*
(stream_0_conv_3/kernel/Regularizer/Sum_1Э
*stream_0_conv_3/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2,
*stream_0_conv_3/kernel/Regularizer/mul_1/xд
(stream_0_conv_3/kernel/Regularizer/mul_1Mul3stream_0_conv_3/kernel/Regularizer/mul_1/x:output:01stream_0_conv_3/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2*
(stream_0_conv_3/kernel/Regularizer/mul_1Ў
(stream_0_conv_3/kernel/Regularizer/add_1AddV2*stream_0_conv_3/kernel/Regularizer/add:z:0,stream_0_conv_3/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2*
(stream_0_conv_3/kernel/Regularizer/add_1Й
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dense_1/kernel/Regularizer/Const∞
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpbasemodel_3295398*
_output_shapes

:@@*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOpІ
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:@@2 
dense_1/kernel/Regularizer/AbsЩ
"dense_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_1/kernel/Regularizer/Const_1є
dense_1/kernel/Regularizer/SumSum"dense_1/kernel/Regularizer/Abs:y:0+dense_1/kernel/Regularizer/Const_1:output:0*
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
dense_1/kernel/Regularizer/mulє
dense_1/kernel/Regularizer/addAddV2)dense_1/kernel/Regularizer/Const:output:0"dense_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/addґ
0dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpbasemodel_3295398*
_output_shapes

:@@*
dtype022
0dense_1/kernel/Regularizer/Square/ReadVariableOp≥
!dense_1/kernel/Regularizer/SquareSquare8dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@@2#
!dense_1/kernel/Regularizer/SquareЩ
"dense_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_1/kernel/Regularizer/Const_2ј
 dense_1/kernel/Regularizer/Sum_1Sum%dense_1/kernel/Regularizer/Square:y:0+dense_1/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2"
 dense_1/kernel/Regularizer/Sum_1Н
"dense_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2$
"dense_1/kernel/Regularizer/mul_1/xƒ
 dense_1/kernel/Regularizer/mul_1Mul+dense_1/kernel/Regularizer/mul_1/x:output:0)dense_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2"
 dense_1/kernel/Regularizer/mul_1Є
 dense_1/kernel/Regularizer/add_1AddV2"dense_1/kernel/Regularizer/add:z:0$dense_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2"
 dense_1/kernel/Regularizer/add_1Е
IdentityIdentity*basemodel/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@2

IdentityЃ
NoOpNoOp"^basemodel/StatefulPartitionedCall.^dense_1/kernel/Regularizer/Abs/ReadVariableOp1^dense_1/kernel/Regularizer/Square/ReadVariableOp6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp9^stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOp6^stream_0_conv_2/kernel/Regularizer/Abs/ReadVariableOp9^stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp6^stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp9^stream_0_conv_3/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:€€€€€€€€€}: : : : : : : : : : : : : : : : : : : : : : : : 2F
!basemodel/StatefulPartitionedCall!basemodel/StatefulPartitionedCall2^
-dense_1/kernel/Regularizer/Abs/ReadVariableOp-dense_1/kernel/Regularizer/Abs/ReadVariableOp2d
0dense_1/kernel/Regularizer/Square/ReadVariableOp0dense_1/kernel/Regularizer/Square/ReadVariableOp2n
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp2t
8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOp8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOp2n
5stream_0_conv_2/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_2/kernel/Regularizer/Abs/ReadVariableOp2t
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp2n
5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp2t
8stream_0_conv_3/kernel/Regularizer/Square/ReadVariableOp8stream_0_conv_3/kernel/Regularizer/Square/ReadVariableOp:X T
+
_output_shapes
:€€€€€€€€€}
%
_user_specified_nameleft_inputs
Є
±
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_3298055

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
k
O__inference_stream_0_maxpool_3_layer_call_and_return_conditional_losses_3298171

inputs
identityb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dimУ

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2

ExpandDims±
MaxPoolMaxPoolExpandDims:output:0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*
ksize
*
paddingVALID*
strides
2	
MaxPoolО
SqueezeSqueezeMaxPool:output:0*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€*
squeeze_dims
2	
Squeezez
IdentityIdentitySqueeze:output:0*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€:e a
=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
о
k
L__inference_stream_0_drop_3_layer_call_and_return_conditional_losses_3298206

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *Ђ™™?2
dropout/Constw
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:€€€€€€€€€@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape”
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:€€€€€€€€€@*
dtype0*
seedЈ*
seed2Ї2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А>2
dropout/GreaterEqual/y¬
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:€€€€€€€€€@2
dropout/GreaterEqualГ
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:€€€€€€€€€@2
dropout/Cast~
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:€€€€€€€€€@2
dropout/Mul_1i
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:€€€€€€€€€@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€@:S O
+
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
Ц
б
)__inference_model_1_layer_call_fn_3295693

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:@
	unknown_5:@@
	unknown_6:@
	unknown_7:@
	unknown_8:@
	unknown_9:@

unknown_10:@ 

unknown_11:@@

unknown_12:@

unknown_13:@

unknown_14:@

unknown_15:@

unknown_16:@

unknown_17:@@

unknown_18:@

unknown_19:@

unknown_20:@

unknown_21:@

unknown_22:@
identityИҐStatefulPartitionedCallЪ
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
:€€€€€€€€€@*2
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_model_1_layer_call_and_return_conditional_losses_32951422
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:€€€€€€€€€}: : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:€€€€€€€€€}
 
_user_specified_nameinputs
Ѕ
j
1__inference_stream_0_drop_2_layer_call_fn_3297912

inputs
identityИҐStatefulPartitionedCallй
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_stream_0_drop_2_layer_call_and_return_conditional_losses_32934702
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€@22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
и
–
5__inference_batch_normalization_layer_call_fn_3297468

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityИҐStatefulPartitionedCall°
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
GPU2*0J 8В *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_32929292
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
С	
“
7__inference_batch_normalization_1_layer_call_fn_3297719

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityИҐStatefulPartitionedCallђ
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
GPU2*0J 8В *[
fVRT
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_32923182
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
у,
О
L__inference_stream_0_conv_2_layer_call_and_return_conditional_losses_3297706

inputsA
+conv1d_expanddims_1_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identityИҐBiasAdd/ReadVariableOpҐ"conv1d/ExpandDims_1/ReadVariableOpҐ5stream_0_conv_2/kernel/Regularizer/Abs/ReadVariableOpҐ8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOpy
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
:€€€€€€€€€>@2
conv1d/ExpandDimsЄ
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
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
:@@2
conv1d/ExpandDims_1ґ
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:€€€€€€€€€>@*
paddingSAME*
strides
2
conv1dТ
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:€€€€€€€€€>@*
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
:€€€€€€€€€>@2	
BiasAddЩ
(stream_0_conv_2/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(stream_0_conv_2/kernel/Regularizer/Constё
5stream_0_conv_2/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype027
5stream_0_conv_2/kernel/Regularizer/Abs/ReadVariableOp√
&stream_0_conv_2/kernel/Regularizer/AbsAbs=stream_0_conv_2/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@@2(
&stream_0_conv_2/kernel/Regularizer/Abs≠
*stream_0_conv_2/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_0_conv_2/kernel/Regularizer/Const_1ў
&stream_0_conv_2/kernel/Regularizer/SumSum*stream_0_conv_2/kernel/Regularizer/Abs:y:03stream_0_conv_2/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_2/kernel/Regularizer/SumЩ
(stream_0_conv_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2*
(stream_0_conv_2/kernel/Regularizer/mul/x№
&stream_0_conv_2/kernel/Regularizer/mulMul1stream_0_conv_2/kernel/Regularizer/mul/x:output:0/stream_0_conv_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_2/kernel/Regularizer/mulў
&stream_0_conv_2/kernel/Regularizer/addAddV21stream_0_conv_2/kernel/Regularizer/Const:output:0*stream_0_conv_2/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&stream_0_conv_2/kernel/Regularizer/addд
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype02:
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOpѕ
)stream_0_conv_2/kernel/Regularizer/SquareSquare@stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@@2+
)stream_0_conv_2/kernel/Regularizer/Square≠
*stream_0_conv_2/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_0_conv_2/kernel/Regularizer/Const_2а
(stream_0_conv_2/kernel/Regularizer/Sum_1Sum-stream_0_conv_2/kernel/Regularizer/Square:y:03stream_0_conv_2/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2*
(stream_0_conv_2/kernel/Regularizer/Sum_1Э
*stream_0_conv_2/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2,
*stream_0_conv_2/kernel/Regularizer/mul_1/xд
(stream_0_conv_2/kernel/Regularizer/mul_1Mul3stream_0_conv_2/kernel/Regularizer/mul_1/x:output:01stream_0_conv_2/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2*
(stream_0_conv_2/kernel/Regularizer/mul_1Ў
(stream_0_conv_2/kernel/Regularizer/add_1AddV2*stream_0_conv_2/kernel/Regularizer/add:z:0,stream_0_conv_2/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2*
(stream_0_conv_2/kernel/Regularizer/add_1o
IdentityIdentityBiasAdd:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€>@2

Identity€
NoOpNoOp^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp6^stream_0_conv_2/kernel/Regularizer/Abs/ReadVariableOp9^stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€>@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp2n
5stream_0_conv_2/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_2/kernel/Regularizer/Abs/ReadVariableOp2t
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp:S O
+
_output_shapes
:€€€€€€€€€>@
 
_user_specified_nameinputs
И
h
L__inference_dense_1_dropout_layer_call_and_return_conditional_losses_3298247

inputs
identityZ
IdentityIdentityinputs*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:€€€€€€€€€@:O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
к
“
7__inference_batch_normalization_2_layer_call_fn_3298035

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityИҐStatefulPartitionedCall°
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *[
fVRT
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_32934292
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
ф
Ц
)__inference_dense_1_layer_call_fn_3298271

inputs
unknown:@@
	unknown_0:@
identityИҐStatefulPartitionedCallч
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_32931772
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
е
P
4__inference_stream_0_maxpool_2_layer_call_fn_3297886

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
:€€€€€€€€€@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_stream_0_maxpool_2_layer_call_and_return_conditional_losses_32930412
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:€€€€€€€€€@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€>@:S O
+
_output_shapes
:€€€€€€€€€>@
 
_user_specified_nameinputs
з
e
I__inference_activation_2_layer_call_and_return_conditional_losses_3298153

inputs
identityR
TanhTanhinputs*
T0*+
_output_shapes
:€€€€€€€€€@2
Tanh`
IdentityIdentityTanh:y:0*
T0*+
_output_shapes
:€€€€€€€€€@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€@:S O
+
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
х!
Ў
D__inference_dense_1_layer_call_and_return_conditional_losses_3298296

inputs0
matmul_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpҐ-dense_1/kernel/Regularizer/Abs/ReadVariableOpҐ0dense_1/kernel/Regularizer/Square/ReadVariableOpН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2	
BiasAddЙ
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dense_1/kernel/Regularizer/Constљ
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOpІ
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:@@2 
dense_1/kernel/Regularizer/AbsЩ
"dense_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_1/kernel/Regularizer/Const_1є
dense_1/kernel/Regularizer/SumSum"dense_1/kernel/Regularizer/Abs:y:0+dense_1/kernel/Regularizer/Const_1:output:0*
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
dense_1/kernel/Regularizer/mulє
dense_1/kernel/Regularizer/addAddV2)dense_1/kernel/Regularizer/Const:output:0"dense_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/add√
0dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype022
0dense_1/kernel/Regularizer/Square/ReadVariableOp≥
!dense_1/kernel/Regularizer/SquareSquare8dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@@2#
!dense_1/kernel/Regularizer/SquareЩ
"dense_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_1/kernel/Regularizer/Const_2ј
 dense_1/kernel/Regularizer/Sum_1Sum%dense_1/kernel/Regularizer/Square:y:0+dense_1/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2"
 dense_1/kernel/Regularizer/Sum_1Н
"dense_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2$
"dense_1/kernel/Regularizer/mul_1/xƒ
 dense_1/kernel/Regularizer/mul_1Mul+dense_1/kernel/Regularizer/mul_1/x:output:0)dense_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2"
 dense_1/kernel/Regularizer/mul_1Є
 dense_1/kernel/Regularizer/add_1AddV2"dense_1/kernel/Regularizer/add:z:0$dense_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2"
 dense_1/kernel/Regularizer/add_1k
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@2

Identityв
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp.^dense_1/kernel/Regularizer/Abs/ReadVariableOp1^dense_1/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2^
-dense_1/kernel/Regularizer/Abs/ReadVariableOp-dense_1/kernel/Regularizer/Abs/ReadVariableOp2d
0dense_1/kernel/Regularizer/Square/ReadVariableOp0dense_1/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
з
Q
5__inference_stream_0_input_drop_layer_call_fn_3297353

inputs
identity’
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
GPU2*0J 8В *Y
fTRR
P__inference_stream_0_input_drop_layer_call_and_return_conditional_losses_32928722
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
÷љ
”
F__inference_basemodel_layer_call_and_return_conditional_losses_3296615

inputsQ
;stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource:@=
/stream_0_conv_1_biasadd_readvariableop_resource:@C
5batch_normalization_batchnorm_readvariableop_resource:@G
9batch_normalization_batchnorm_mul_readvariableop_resource:@E
7batch_normalization_batchnorm_readvariableop_1_resource:@E
7batch_normalization_batchnorm_readvariableop_2_resource:@Q
;stream_0_conv_2_conv1d_expanddims_1_readvariableop_resource:@@=
/stream_0_conv_2_biasadd_readvariableop_resource:@E
7batch_normalization_1_batchnorm_readvariableop_resource:@I
;batch_normalization_1_batchnorm_mul_readvariableop_resource:@G
9batch_normalization_1_batchnorm_readvariableop_1_resource:@G
9batch_normalization_1_batchnorm_readvariableop_2_resource:@Q
;stream_0_conv_3_conv1d_expanddims_1_readvariableop_resource:@@=
/stream_0_conv_3_biasadd_readvariableop_resource:@E
7batch_normalization_2_batchnorm_readvariableop_resource:@I
;batch_normalization_2_batchnorm_mul_readvariableop_resource:@G
9batch_normalization_2_batchnorm_readvariableop_1_resource:@G
9batch_normalization_2_batchnorm_readvariableop_2_resource:@8
&dense_1_matmul_readvariableop_resource:@@5
'dense_1_biasadd_readvariableop_resource:@E
7batch_normalization_3_batchnorm_readvariableop_resource:@I
;batch_normalization_3_batchnorm_mul_readvariableop_resource:@G
9batch_normalization_3_batchnorm_readvariableop_1_resource:@G
9batch_normalization_3_batchnorm_readvariableop_2_resource:@
identityИҐ,batch_normalization/batchnorm/ReadVariableOpҐ.batch_normalization/batchnorm/ReadVariableOp_1Ґ.batch_normalization/batchnorm/ReadVariableOp_2Ґ0batch_normalization/batchnorm/mul/ReadVariableOpҐ.batch_normalization_1/batchnorm/ReadVariableOpҐ0batch_normalization_1/batchnorm/ReadVariableOp_1Ґ0batch_normalization_1/batchnorm/ReadVariableOp_2Ґ2batch_normalization_1/batchnorm/mul/ReadVariableOpҐ.batch_normalization_2/batchnorm/ReadVariableOpҐ0batch_normalization_2/batchnorm/ReadVariableOp_1Ґ0batch_normalization_2/batchnorm/ReadVariableOp_2Ґ2batch_normalization_2/batchnorm/mul/ReadVariableOpҐ.batch_normalization_3/batchnorm/ReadVariableOpҐ0batch_normalization_3/batchnorm/ReadVariableOp_1Ґ0batch_normalization_3/batchnorm/ReadVariableOp_2Ґ2batch_normalization_3/batchnorm/mul/ReadVariableOpҐdense_1/BiasAdd/ReadVariableOpҐdense_1/MatMul/ReadVariableOpҐ-dense_1/kernel/Regularizer/Abs/ReadVariableOpҐ0dense_1/kernel/Regularizer/Square/ReadVariableOpҐ&stream_0_conv_1/BiasAdd/ReadVariableOpҐ2stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOpҐ5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpҐ8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOpҐ&stream_0_conv_2/BiasAdd/ReadVariableOpҐ2stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOpҐ5stream_0_conv_2/kernel/Regularizer/Abs/ReadVariableOpҐ8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOpҐ&stream_0_conv_3/BiasAdd/ReadVariableOpҐ2stream_0_conv_3/conv1d/ExpandDims_1/ReadVariableOpҐ5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOpҐ8stream_0_conv_3/kernel/Regularizer/Square/ReadVariableOpЖ
stream_0_input_drop/IdentityIdentityinputs*
T0*+
_output_shapes
:€€€€€€€€€}2
stream_0_input_drop/IdentityЩ
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
stream_0_conv_1/BiasAddќ
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
#batch_normalization/batchnorm/add_1Й
activation/TanhTanh'batch_normalization/batchnorm/add_1:z:0*
T0*+
_output_shapes
:€€€€€€€€€}@2
activation/TanhИ
!stream_0_maxpool_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!stream_0_maxpool_1/ExpandDims/dim«
stream_0_maxpool_1/ExpandDims
ExpandDimsactivation/Tanh:y:0*stream_0_maxpool_1/ExpandDims/dim:output:0*
T0*/
_output_shapes
:€€€€€€€€€}@2
stream_0_maxpool_1/ExpandDimsЎ
stream_0_maxpool_1/MaxPoolMaxPool&stream_0_maxpool_1/ExpandDims:output:0*/
_output_shapes
:€€€€€€€€€>@*
ksize
*
paddingVALID*
strides
2
stream_0_maxpool_1/MaxPoolµ
stream_0_maxpool_1/SqueezeSqueeze#stream_0_maxpool_1/MaxPool:output:0*
T0*+
_output_shapes
:€€€€€€€€€>@*
squeeze_dims
2
stream_0_maxpool_1/SqueezeЫ
stream_0_drop_1/IdentityIdentity#stream_0_maxpool_1/Squeeze:output:0*
T0*+
_output_shapes
:€€€€€€€€€>@2
stream_0_drop_1/IdentityЩ
%stream_0_conv_2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€2'
%stream_0_conv_2/conv1d/ExpandDims/dimб
!stream_0_conv_2/conv1d/ExpandDims
ExpandDims!stream_0_drop_1/Identity:output:0.stream_0_conv_2/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:€€€€€€€€€>@2#
!stream_0_conv_2/conv1d/ExpandDimsи
2stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp;stream_0_conv_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype024
2stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOpФ
'stream_0_conv_2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2)
'stream_0_conv_2/conv1d/ExpandDims_1/dimч
#stream_0_conv_2/conv1d/ExpandDims_1
ExpandDims:stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOp:value:00stream_0_conv_2/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@2%
#stream_0_conv_2/conv1d/ExpandDims_1ц
stream_0_conv_2/conv1dConv2D*stream_0_conv_2/conv1d/ExpandDims:output:0,stream_0_conv_2/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:€€€€€€€€€>@*
paddingSAME*
strides
2
stream_0_conv_2/conv1d¬
stream_0_conv_2/conv1d/SqueezeSqueezestream_0_conv_2/conv1d:output:0*
T0*+
_output_shapes
:€€€€€€€€€>@*
squeeze_dims

э€€€€€€€€2 
stream_0_conv_2/conv1d/SqueezeЉ
&stream_0_conv_2/BiasAdd/ReadVariableOpReadVariableOp/stream_0_conv_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02(
&stream_0_conv_2/BiasAdd/ReadVariableOpћ
stream_0_conv_2/BiasAddBiasAdd'stream_0_conv_2/conv1d/Squeeze:output:0.stream_0_conv_2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€>@2
stream_0_conv_2/BiasAdd‘
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
%batch_normalization_1/batchnorm/mul_1Mul stream_0_conv_2/BiasAdd:output:0'batch_normalization_1/batchnorm/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€>@2'
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
:€€€€€€€€€>@2'
%batch_normalization_1/batchnorm/add_1П
activation_1/TanhTanh)batch_normalization_1/batchnorm/add_1:z:0*
T0*+
_output_shapes
:€€€€€€€€€>@2
activation_1/TanhИ
!stream_0_maxpool_2/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!stream_0_maxpool_2/ExpandDims/dim…
stream_0_maxpool_2/ExpandDims
ExpandDimsactivation_1/Tanh:y:0*stream_0_maxpool_2/ExpandDims/dim:output:0*
T0*/
_output_shapes
:€€€€€€€€€>@2
stream_0_maxpool_2/ExpandDimsЎ
stream_0_maxpool_2/MaxPoolMaxPool&stream_0_maxpool_2/ExpandDims:output:0*/
_output_shapes
:€€€€€€€€€@*
ksize
*
paddingVALID*
strides
2
stream_0_maxpool_2/MaxPoolµ
stream_0_maxpool_2/SqueezeSqueeze#stream_0_maxpool_2/MaxPool:output:0*
T0*+
_output_shapes
:€€€€€€€€€@*
squeeze_dims
2
stream_0_maxpool_2/SqueezeЫ
stream_0_drop_2/IdentityIdentity#stream_0_maxpool_2/Squeeze:output:0*
T0*+
_output_shapes
:€€€€€€€€€@2
stream_0_drop_2/IdentityЩ
%stream_0_conv_3/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€2'
%stream_0_conv_3/conv1d/ExpandDims/dimб
!stream_0_conv_3/conv1d/ExpandDims
ExpandDims!stream_0_drop_2/Identity:output:0.stream_0_conv_3/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:€€€€€€€€€@2#
!stream_0_conv_3/conv1d/ExpandDimsи
2stream_0_conv_3/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp;stream_0_conv_3_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype024
2stream_0_conv_3/conv1d/ExpandDims_1/ReadVariableOpФ
'stream_0_conv_3/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2)
'stream_0_conv_3/conv1d/ExpandDims_1/dimч
#stream_0_conv_3/conv1d/ExpandDims_1
ExpandDims:stream_0_conv_3/conv1d/ExpandDims_1/ReadVariableOp:value:00stream_0_conv_3/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@2%
#stream_0_conv_3/conv1d/ExpandDims_1ц
stream_0_conv_3/conv1dConv2D*stream_0_conv_3/conv1d/ExpandDims:output:0,stream_0_conv_3/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:€€€€€€€€€@*
paddingSAME*
strides
2
stream_0_conv_3/conv1d¬
stream_0_conv_3/conv1d/SqueezeSqueezestream_0_conv_3/conv1d:output:0*
T0*+
_output_shapes
:€€€€€€€€€@*
squeeze_dims

э€€€€€€€€2 
stream_0_conv_3/conv1d/SqueezeЉ
&stream_0_conv_3/BiasAdd/ReadVariableOpReadVariableOp/stream_0_conv_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02(
&stream_0_conv_3/BiasAdd/ReadVariableOpћ
stream_0_conv_3/BiasAddBiasAdd'stream_0_conv_3/conv1d/Squeeze:output:0.stream_0_conv_3/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€@2
stream_0_conv_3/BiasAdd‘
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
%batch_normalization_2/batchnorm/mul_1Mul stream_0_conv_3/BiasAdd:output:0'batch_normalization_2/batchnorm/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€@2'
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
:€€€€€€€€€@2'
%batch_normalization_2/batchnorm/add_1П
activation_2/TanhTanh)batch_normalization_2/batchnorm/add_1:z:0*
T0*+
_output_shapes
:€€€€€€€€€@2
activation_2/TanhИ
!stream_0_maxpool_3/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!stream_0_maxpool_3/ExpandDims/dim…
stream_0_maxpool_3/ExpandDims
ExpandDimsactivation_2/Tanh:y:0*stream_0_maxpool_3/ExpandDims/dim:output:0*
T0*/
_output_shapes
:€€€€€€€€€@2
stream_0_maxpool_3/ExpandDimsЎ
stream_0_maxpool_3/MaxPoolMaxPool&stream_0_maxpool_3/ExpandDims:output:0*/
_output_shapes
:€€€€€€€€€@*
ksize
*
paddingVALID*
strides
2
stream_0_maxpool_3/MaxPoolµ
stream_0_maxpool_3/SqueezeSqueeze#stream_0_maxpool_3/MaxPool:output:0*
T0*+
_output_shapes
:€€€€€€€€€@*
squeeze_dims
2
stream_0_maxpool_3/SqueezeЫ
stream_0_drop_3/IdentityIdentity#stream_0_maxpool_3/Squeeze:output:0*
T0*+
_output_shapes
:€€€€€€€€€@2
stream_0_drop_3/Identity§
/global_average_pooling1d/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :21
/global_average_pooling1d/Mean/reduction_indices’
global_average_pooling1d/MeanMean!stream_0_drop_3/Identity:output:08global_average_pooling1d/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
global_average_pooling1d/MeanЪ
dense_1_dropout/IdentityIdentity&global_average_pooling1d/Mean:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
dense_1_dropout/Identity•
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02
dense_1/MatMul/ReadVariableOp¶
dense_1/MatMulMatMul!dense_1_dropout/Identity:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2
dense_1/MatMul§
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02 
dense_1/BiasAdd/ReadVariableOp°
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2
dense_1/BiasAdd‘
.batch_normalization_3/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_3_batchnorm_readvariableop_resource*
_output_shapes
:@*
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
:@2%
#batch_normalization_3/batchnorm/add•
%batch_normalization_3/batchnorm/RsqrtRsqrt'batch_normalization_3/batchnorm/add:z:0*
T0*
_output_shapes
:@2'
%batch_normalization_3/batchnorm/Rsqrtа
2batch_normalization_3/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_3_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype024
2batch_normalization_3/batchnorm/mul/ReadVariableOpЁ
#batch_normalization_3/batchnorm/mulMul)batch_normalization_3/batchnorm/Rsqrt:y:0:batch_normalization_3/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2%
#batch_normalization_3/batchnorm/mul 
%batch_normalization_3/batchnorm/mul_1Muldense_1/BiasAdd:output:0'batch_normalization_3/batchnorm/mul:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2'
%batch_normalization_3/batchnorm/mul_1Џ
0batch_normalization_3/batchnorm/ReadVariableOp_1ReadVariableOp9batch_normalization_3_batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype022
0batch_normalization_3/batchnorm/ReadVariableOp_1Ё
%batch_normalization_3/batchnorm/mul_2Mul8batch_normalization_3/batchnorm/ReadVariableOp_1:value:0'batch_normalization_3/batchnorm/mul:z:0*
T0*
_output_shapes
:@2'
%batch_normalization_3/batchnorm/mul_2Џ
0batch_normalization_3/batchnorm/ReadVariableOp_2ReadVariableOp9batch_normalization_3_batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype022
0batch_normalization_3/batchnorm/ReadVariableOp_2џ
#batch_normalization_3/batchnorm/subSub8batch_normalization_3/batchnorm/ReadVariableOp_2:value:0)batch_normalization_3/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2%
#batch_normalization_3/batchnorm/subЁ
%batch_normalization_3/batchnorm/add_1AddV2)batch_normalization_3/batchnorm/mul_1:z:0'batch_normalization_3/batchnorm/sub:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2'
%batch_normalization_3/batchnorm/add_1Щ
(stream_0_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(stream_0_conv_1/kernel/Regularizer/Constо
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp;stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype027
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp√
&stream_0_conv_1/kernel/Regularizer/AbsAbs=stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_0_conv_1/kernel/Regularizer/Abs≠
*stream_0_conv_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_0_conv_1/kernel/Regularizer/Const_1ў
&stream_0_conv_1/kernel/Regularizer/SumSum*stream_0_conv_1/kernel/Regularizer/Abs:y:03stream_0_conv_1/kernel/Regularizer/Const_1:output:0*
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
&stream_0_conv_1/kernel/Regularizer/mulў
&stream_0_conv_1/kernel/Regularizer/addAddV21stream_0_conv_1/kernel/Regularizer/Const:output:0*stream_0_conv_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/addф
8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02:
8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOpѕ
)stream_0_conv_1/kernel/Regularizer/SquareSquare@stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2+
)stream_0_conv_1/kernel/Regularizer/Square≠
*stream_0_conv_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_0_conv_1/kernel/Regularizer/Const_2а
(stream_0_conv_1/kernel/Regularizer/Sum_1Sum-stream_0_conv_1/kernel/Regularizer/Square:y:03stream_0_conv_1/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2*
(stream_0_conv_1/kernel/Regularizer/Sum_1Э
*stream_0_conv_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2,
*stream_0_conv_1/kernel/Regularizer/mul_1/xд
(stream_0_conv_1/kernel/Regularizer/mul_1Mul3stream_0_conv_1/kernel/Regularizer/mul_1/x:output:01stream_0_conv_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2*
(stream_0_conv_1/kernel/Regularizer/mul_1Ў
(stream_0_conv_1/kernel/Regularizer/add_1AddV2*stream_0_conv_1/kernel/Regularizer/add:z:0,stream_0_conv_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2*
(stream_0_conv_1/kernel/Regularizer/add_1Щ
(stream_0_conv_2/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(stream_0_conv_2/kernel/Regularizer/Constо
5stream_0_conv_2/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp;stream_0_conv_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype027
5stream_0_conv_2/kernel/Regularizer/Abs/ReadVariableOp√
&stream_0_conv_2/kernel/Regularizer/AbsAbs=stream_0_conv_2/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@@2(
&stream_0_conv_2/kernel/Regularizer/Abs≠
*stream_0_conv_2/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_0_conv_2/kernel/Regularizer/Const_1ў
&stream_0_conv_2/kernel/Regularizer/SumSum*stream_0_conv_2/kernel/Regularizer/Abs:y:03stream_0_conv_2/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_2/kernel/Regularizer/SumЩ
(stream_0_conv_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2*
(stream_0_conv_2/kernel/Regularizer/mul/x№
&stream_0_conv_2/kernel/Regularizer/mulMul1stream_0_conv_2/kernel/Regularizer/mul/x:output:0/stream_0_conv_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_2/kernel/Regularizer/mulў
&stream_0_conv_2/kernel/Regularizer/addAddV21stream_0_conv_2/kernel/Regularizer/Const:output:0*stream_0_conv_2/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&stream_0_conv_2/kernel/Regularizer/addф
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;stream_0_conv_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype02:
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOpѕ
)stream_0_conv_2/kernel/Regularizer/SquareSquare@stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@@2+
)stream_0_conv_2/kernel/Regularizer/Square≠
*stream_0_conv_2/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_0_conv_2/kernel/Regularizer/Const_2а
(stream_0_conv_2/kernel/Regularizer/Sum_1Sum-stream_0_conv_2/kernel/Regularizer/Square:y:03stream_0_conv_2/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2*
(stream_0_conv_2/kernel/Regularizer/Sum_1Э
*stream_0_conv_2/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2,
*stream_0_conv_2/kernel/Regularizer/mul_1/xд
(stream_0_conv_2/kernel/Regularizer/mul_1Mul3stream_0_conv_2/kernel/Regularizer/mul_1/x:output:01stream_0_conv_2/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2*
(stream_0_conv_2/kernel/Regularizer/mul_1Ў
(stream_0_conv_2/kernel/Regularizer/add_1AddV2*stream_0_conv_2/kernel/Regularizer/add:z:0,stream_0_conv_2/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2*
(stream_0_conv_2/kernel/Regularizer/add_1Щ
(stream_0_conv_3/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(stream_0_conv_3/kernel/Regularizer/Constо
5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp;stream_0_conv_3_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype027
5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp√
&stream_0_conv_3/kernel/Regularizer/AbsAbs=stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@@2(
&stream_0_conv_3/kernel/Regularizer/Abs≠
*stream_0_conv_3/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_0_conv_3/kernel/Regularizer/Const_1ў
&stream_0_conv_3/kernel/Regularizer/SumSum*stream_0_conv_3/kernel/Regularizer/Abs:y:03stream_0_conv_3/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_3/kernel/Regularizer/SumЩ
(stream_0_conv_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2*
(stream_0_conv_3/kernel/Regularizer/mul/x№
&stream_0_conv_3/kernel/Regularizer/mulMul1stream_0_conv_3/kernel/Regularizer/mul/x:output:0/stream_0_conv_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_3/kernel/Regularizer/mulў
&stream_0_conv_3/kernel/Regularizer/addAddV21stream_0_conv_3/kernel/Regularizer/Const:output:0*stream_0_conv_3/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&stream_0_conv_3/kernel/Regularizer/addф
8stream_0_conv_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;stream_0_conv_3_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype02:
8stream_0_conv_3/kernel/Regularizer/Square/ReadVariableOpѕ
)stream_0_conv_3/kernel/Regularizer/SquareSquare@stream_0_conv_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@@2+
)stream_0_conv_3/kernel/Regularizer/Square≠
*stream_0_conv_3/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_0_conv_3/kernel/Regularizer/Const_2а
(stream_0_conv_3/kernel/Regularizer/Sum_1Sum-stream_0_conv_3/kernel/Regularizer/Square:y:03stream_0_conv_3/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2*
(stream_0_conv_3/kernel/Regularizer/Sum_1Э
*stream_0_conv_3/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2,
*stream_0_conv_3/kernel/Regularizer/mul_1/xд
(stream_0_conv_3/kernel/Regularizer/mul_1Mul3stream_0_conv_3/kernel/Regularizer/mul_1/x:output:01stream_0_conv_3/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2*
(stream_0_conv_3/kernel/Regularizer/mul_1Ў
(stream_0_conv_3/kernel/Regularizer/add_1AddV2*stream_0_conv_3/kernel/Regularizer/add:z:0,stream_0_conv_3/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2*
(stream_0_conv_3/kernel/Regularizer/add_1Й
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dense_1/kernel/Regularizer/Const≈
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOpІ
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:@@2 
dense_1/kernel/Regularizer/AbsЩ
"dense_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_1/kernel/Regularizer/Const_1є
dense_1/kernel/Regularizer/SumSum"dense_1/kernel/Regularizer/Abs:y:0+dense_1/kernel/Regularizer/Const_1:output:0*
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
dense_1/kernel/Regularizer/mulє
dense_1/kernel/Regularizer/addAddV2)dense_1/kernel/Regularizer/Const:output:0"dense_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/addЋ
0dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype022
0dense_1/kernel/Regularizer/Square/ReadVariableOp≥
!dense_1/kernel/Regularizer/SquareSquare8dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@@2#
!dense_1/kernel/Regularizer/SquareЩ
"dense_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_1/kernel/Regularizer/Const_2ј
 dense_1/kernel/Regularizer/Sum_1Sum%dense_1/kernel/Regularizer/Square:y:0+dense_1/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2"
 dense_1/kernel/Regularizer/Sum_1Н
"dense_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2$
"dense_1/kernel/Regularizer/mul_1/xƒ
 dense_1/kernel/Regularizer/mul_1Mul+dense_1/kernel/Regularizer/mul_1/x:output:0)dense_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2"
 dense_1/kernel/Regularizer/mul_1Є
 dense_1/kernel/Regularizer/add_1AddV2"dense_1/kernel/Regularizer/add:z:0$dense_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2"
 dense_1/kernel/Regularizer/add_1Д
IdentityIdentity)batch_normalization_3/batchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@2

IdentityН
NoOpNoOp-^batch_normalization/batchnorm/ReadVariableOp/^batch_normalization/batchnorm/ReadVariableOp_1/^batch_normalization/batchnorm/ReadVariableOp_21^batch_normalization/batchnorm/mul/ReadVariableOp/^batch_normalization_1/batchnorm/ReadVariableOp1^batch_normalization_1/batchnorm/ReadVariableOp_11^batch_normalization_1/batchnorm/ReadVariableOp_23^batch_normalization_1/batchnorm/mul/ReadVariableOp/^batch_normalization_2/batchnorm/ReadVariableOp1^batch_normalization_2/batchnorm/ReadVariableOp_11^batch_normalization_2/batchnorm/ReadVariableOp_23^batch_normalization_2/batchnorm/mul/ReadVariableOp/^batch_normalization_3/batchnorm/ReadVariableOp1^batch_normalization_3/batchnorm/ReadVariableOp_11^batch_normalization_3/batchnorm/ReadVariableOp_23^batch_normalization_3/batchnorm/mul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp.^dense_1/kernel/Regularizer/Abs/ReadVariableOp1^dense_1/kernel/Regularizer/Square/ReadVariableOp'^stream_0_conv_1/BiasAdd/ReadVariableOp3^stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp9^stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOp'^stream_0_conv_2/BiasAdd/ReadVariableOp3^stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOp6^stream_0_conv_2/kernel/Regularizer/Abs/ReadVariableOp9^stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp'^stream_0_conv_3/BiasAdd/ReadVariableOp3^stream_0_conv_3/conv1d/ExpandDims_1/ReadVariableOp6^stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp9^stream_0_conv_3/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:€€€€€€€€€}: : : : : : : : : : : : : : : : : : : : : : : : 2\
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
-dense_1/kernel/Regularizer/Abs/ReadVariableOp-dense_1/kernel/Regularizer/Abs/ReadVariableOp2d
0dense_1/kernel/Regularizer/Square/ReadVariableOp0dense_1/kernel/Regularizer/Square/ReadVariableOp2P
&stream_0_conv_1/BiasAdd/ReadVariableOp&stream_0_conv_1/BiasAdd/ReadVariableOp2h
2stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp2stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp2n
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp2t
8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOp8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOp2P
&stream_0_conv_2/BiasAdd/ReadVariableOp&stream_0_conv_2/BiasAdd/ReadVariableOp2h
2stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOp2stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOp2n
5stream_0_conv_2/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_2/kernel/Regularizer/Abs/ReadVariableOp2t
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp2P
&stream_0_conv_3/BiasAdd/ReadVariableOp&stream_0_conv_3/BiasAdd/ReadVariableOp2h
2stream_0_conv_3/conv1d/ExpandDims_1/ReadVariableOp2stream_0_conv_3/conv1d/ExpandDims_1/ReadVariableOp2n
5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp2t
8stream_0_conv_3/kernel/Regularizer/Square/ReadVariableOp8stream_0_conv_3/kernel/Regularizer/Square/ReadVariableOp:S O
+
_output_shapes
:€€€€€€€€€}
 
_user_specified_nameinputs
м
“
7__inference_batch_normalization_2_layer_call_fn_3298022

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityИҐStatefulPartitionedCall£
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *[
fVRT
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_32931052
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
К
±
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_3293105

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
:€€€€€€€€€@2
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
:€€€€€€€€€@2
batchnorm/add_1r
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€@2

Identity¬
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€@: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:S O
+
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
у,
О
L__inference_stream_0_conv_1_layer_call_and_return_conditional_losses_3292904

inputsA
+conv1d_expanddims_1_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identityИҐBiasAdd/ReadVariableOpҐ"conv1d/ExpandDims_1/ReadVariableOpҐ5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpҐ8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOpy
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
BiasAddЩ
(stream_0_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(stream_0_conv_1/kernel/Regularizer/Constё
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype027
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp√
&stream_0_conv_1/kernel/Regularizer/AbsAbs=stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_0_conv_1/kernel/Regularizer/Abs≠
*stream_0_conv_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_0_conv_1/kernel/Regularizer/Const_1ў
&stream_0_conv_1/kernel/Regularizer/SumSum*stream_0_conv_1/kernel/Regularizer/Abs:y:03stream_0_conv_1/kernel/Regularizer/Const_1:output:0*
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
&stream_0_conv_1/kernel/Regularizer/mulў
&stream_0_conv_1/kernel/Regularizer/addAddV21stream_0_conv_1/kernel/Regularizer/Const:output:0*stream_0_conv_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/addд
8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02:
8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOpѕ
)stream_0_conv_1/kernel/Regularizer/SquareSquare@stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2+
)stream_0_conv_1/kernel/Regularizer/Square≠
*stream_0_conv_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_0_conv_1/kernel/Regularizer/Const_2а
(stream_0_conv_1/kernel/Regularizer/Sum_1Sum-stream_0_conv_1/kernel/Regularizer/Square:y:03stream_0_conv_1/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2*
(stream_0_conv_1/kernel/Regularizer/Sum_1Э
*stream_0_conv_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2,
*stream_0_conv_1/kernel/Regularizer/mul_1/xд
(stream_0_conv_1/kernel/Regularizer/mul_1Mul3stream_0_conv_1/kernel/Regularizer/mul_1/x:output:01stream_0_conv_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2*
(stream_0_conv_1/kernel/Regularizer/mul_1Ў
(stream_0_conv_1/kernel/Regularizer/add_1AddV2*stream_0_conv_1/kernel/Regularizer/add:z:0,stream_0_conv_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2*
(stream_0_conv_1/kernel/Regularizer/add_1o
IdentityIdentityBiasAdd:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€}@2

Identity€
NoOpNoOp^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp9^stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOp*"
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
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp2t
8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOp8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOp:S O
+
_output_shapes
:€€€€€€€€€}
 
_user_specified_nameinputs
фz
ь

D__inference_model_1_layer_call_and_return_conditional_losses_3295359
left_inputs'
basemodel_3295249:@
basemodel_3295251:@
basemodel_3295253:@
basemodel_3295255:@
basemodel_3295257:@
basemodel_3295259:@'
basemodel_3295261:@@
basemodel_3295263:@
basemodel_3295265:@
basemodel_3295267:@
basemodel_3295269:@
basemodel_3295271:@'
basemodel_3295273:@@
basemodel_3295275:@
basemodel_3295277:@
basemodel_3295279:@
basemodel_3295281:@
basemodel_3295283:@#
basemodel_3295285:@@
basemodel_3295287:@
basemodel_3295289:@
basemodel_3295291:@
basemodel_3295293:@
basemodel_3295295:@
identityИҐ!basemodel/StatefulPartitionedCallҐ-dense_1/kernel/Regularizer/Abs/ReadVariableOpҐ0dense_1/kernel/Regularizer/Square/ReadVariableOpҐ5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpҐ8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOpҐ5stream_0_conv_2/kernel/Regularizer/Abs/ReadVariableOpҐ8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOpҐ5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOpҐ8stream_0_conv_3/kernel/Regularizer/Square/ReadVariableOpт
!basemodel/StatefulPartitionedCallStatefulPartitionedCallleft_inputsbasemodel_3295249basemodel_3295251basemodel_3295253basemodel_3295255basemodel_3295257basemodel_3295259basemodel_3295261basemodel_3295263basemodel_3295265basemodel_3295267basemodel_3295269basemodel_3295271basemodel_3295273basemodel_3295275basemodel_3295277basemodel_3295279basemodel_3295281basemodel_3295283basemodel_3295285basemodel_3295287basemodel_3295289basemodel_3295291basemodel_3295293basemodel_3295295*$
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@*:
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_basemodel_layer_call_and_return_conditional_losses_32944342#
!basemodel/StatefulPartitionedCallЩ
(stream_0_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(stream_0_conv_1/kernel/Regularizer/Constƒ
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpbasemodel_3295249*"
_output_shapes
:@*
dtype027
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp√
&stream_0_conv_1/kernel/Regularizer/AbsAbs=stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_0_conv_1/kernel/Regularizer/Abs≠
*stream_0_conv_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_0_conv_1/kernel/Regularizer/Const_1ў
&stream_0_conv_1/kernel/Regularizer/SumSum*stream_0_conv_1/kernel/Regularizer/Abs:y:03stream_0_conv_1/kernel/Regularizer/Const_1:output:0*
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
&stream_0_conv_1/kernel/Regularizer/mulў
&stream_0_conv_1/kernel/Regularizer/addAddV21stream_0_conv_1/kernel/Regularizer/Const:output:0*stream_0_conv_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/add 
8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpbasemodel_3295249*"
_output_shapes
:@*
dtype02:
8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOpѕ
)stream_0_conv_1/kernel/Regularizer/SquareSquare@stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2+
)stream_0_conv_1/kernel/Regularizer/Square≠
*stream_0_conv_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_0_conv_1/kernel/Regularizer/Const_2а
(stream_0_conv_1/kernel/Regularizer/Sum_1Sum-stream_0_conv_1/kernel/Regularizer/Square:y:03stream_0_conv_1/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2*
(stream_0_conv_1/kernel/Regularizer/Sum_1Э
*stream_0_conv_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2,
*stream_0_conv_1/kernel/Regularizer/mul_1/xд
(stream_0_conv_1/kernel/Regularizer/mul_1Mul3stream_0_conv_1/kernel/Regularizer/mul_1/x:output:01stream_0_conv_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2*
(stream_0_conv_1/kernel/Regularizer/mul_1Ў
(stream_0_conv_1/kernel/Regularizer/add_1AddV2*stream_0_conv_1/kernel/Regularizer/add:z:0,stream_0_conv_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2*
(stream_0_conv_1/kernel/Regularizer/add_1Щ
(stream_0_conv_2/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(stream_0_conv_2/kernel/Regularizer/Constƒ
5stream_0_conv_2/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpbasemodel_3295261*"
_output_shapes
:@@*
dtype027
5stream_0_conv_2/kernel/Regularizer/Abs/ReadVariableOp√
&stream_0_conv_2/kernel/Regularizer/AbsAbs=stream_0_conv_2/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@@2(
&stream_0_conv_2/kernel/Regularizer/Abs≠
*stream_0_conv_2/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_0_conv_2/kernel/Regularizer/Const_1ў
&stream_0_conv_2/kernel/Regularizer/SumSum*stream_0_conv_2/kernel/Regularizer/Abs:y:03stream_0_conv_2/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_2/kernel/Regularizer/SumЩ
(stream_0_conv_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2*
(stream_0_conv_2/kernel/Regularizer/mul/x№
&stream_0_conv_2/kernel/Regularizer/mulMul1stream_0_conv_2/kernel/Regularizer/mul/x:output:0/stream_0_conv_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_2/kernel/Regularizer/mulў
&stream_0_conv_2/kernel/Regularizer/addAddV21stream_0_conv_2/kernel/Regularizer/Const:output:0*stream_0_conv_2/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&stream_0_conv_2/kernel/Regularizer/add 
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpbasemodel_3295261*"
_output_shapes
:@@*
dtype02:
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOpѕ
)stream_0_conv_2/kernel/Regularizer/SquareSquare@stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@@2+
)stream_0_conv_2/kernel/Regularizer/Square≠
*stream_0_conv_2/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_0_conv_2/kernel/Regularizer/Const_2а
(stream_0_conv_2/kernel/Regularizer/Sum_1Sum-stream_0_conv_2/kernel/Regularizer/Square:y:03stream_0_conv_2/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2*
(stream_0_conv_2/kernel/Regularizer/Sum_1Э
*stream_0_conv_2/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2,
*stream_0_conv_2/kernel/Regularizer/mul_1/xд
(stream_0_conv_2/kernel/Regularizer/mul_1Mul3stream_0_conv_2/kernel/Regularizer/mul_1/x:output:01stream_0_conv_2/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2*
(stream_0_conv_2/kernel/Regularizer/mul_1Ў
(stream_0_conv_2/kernel/Regularizer/add_1AddV2*stream_0_conv_2/kernel/Regularizer/add:z:0,stream_0_conv_2/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2*
(stream_0_conv_2/kernel/Regularizer/add_1Щ
(stream_0_conv_3/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(stream_0_conv_3/kernel/Regularizer/Constƒ
5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpbasemodel_3295273*"
_output_shapes
:@@*
dtype027
5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp√
&stream_0_conv_3/kernel/Regularizer/AbsAbs=stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@@2(
&stream_0_conv_3/kernel/Regularizer/Abs≠
*stream_0_conv_3/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_0_conv_3/kernel/Regularizer/Const_1ў
&stream_0_conv_3/kernel/Regularizer/SumSum*stream_0_conv_3/kernel/Regularizer/Abs:y:03stream_0_conv_3/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_3/kernel/Regularizer/SumЩ
(stream_0_conv_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2*
(stream_0_conv_3/kernel/Regularizer/mul/x№
&stream_0_conv_3/kernel/Regularizer/mulMul1stream_0_conv_3/kernel/Regularizer/mul/x:output:0/stream_0_conv_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_3/kernel/Regularizer/mulў
&stream_0_conv_3/kernel/Regularizer/addAddV21stream_0_conv_3/kernel/Regularizer/Const:output:0*stream_0_conv_3/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&stream_0_conv_3/kernel/Regularizer/add 
8stream_0_conv_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpbasemodel_3295273*"
_output_shapes
:@@*
dtype02:
8stream_0_conv_3/kernel/Regularizer/Square/ReadVariableOpѕ
)stream_0_conv_3/kernel/Regularizer/SquareSquare@stream_0_conv_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@@2+
)stream_0_conv_3/kernel/Regularizer/Square≠
*stream_0_conv_3/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_0_conv_3/kernel/Regularizer/Const_2а
(stream_0_conv_3/kernel/Regularizer/Sum_1Sum-stream_0_conv_3/kernel/Regularizer/Square:y:03stream_0_conv_3/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2*
(stream_0_conv_3/kernel/Regularizer/Sum_1Э
*stream_0_conv_3/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2,
*stream_0_conv_3/kernel/Regularizer/mul_1/xд
(stream_0_conv_3/kernel/Regularizer/mul_1Mul3stream_0_conv_3/kernel/Regularizer/mul_1/x:output:01stream_0_conv_3/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2*
(stream_0_conv_3/kernel/Regularizer/mul_1Ў
(stream_0_conv_3/kernel/Regularizer/add_1AddV2*stream_0_conv_3/kernel/Regularizer/add:z:0,stream_0_conv_3/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2*
(stream_0_conv_3/kernel/Regularizer/add_1Й
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dense_1/kernel/Regularizer/Const∞
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpbasemodel_3295285*
_output_shapes

:@@*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOpІ
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:@@2 
dense_1/kernel/Regularizer/AbsЩ
"dense_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_1/kernel/Regularizer/Const_1є
dense_1/kernel/Regularizer/SumSum"dense_1/kernel/Regularizer/Abs:y:0+dense_1/kernel/Regularizer/Const_1:output:0*
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
dense_1/kernel/Regularizer/mulє
dense_1/kernel/Regularizer/addAddV2)dense_1/kernel/Regularizer/Const:output:0"dense_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/addґ
0dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpbasemodel_3295285*
_output_shapes

:@@*
dtype022
0dense_1/kernel/Regularizer/Square/ReadVariableOp≥
!dense_1/kernel/Regularizer/SquareSquare8dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@@2#
!dense_1/kernel/Regularizer/SquareЩ
"dense_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_1/kernel/Regularizer/Const_2ј
 dense_1/kernel/Regularizer/Sum_1Sum%dense_1/kernel/Regularizer/Square:y:0+dense_1/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2"
 dense_1/kernel/Regularizer/Sum_1Н
"dense_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2$
"dense_1/kernel/Regularizer/mul_1/xƒ
 dense_1/kernel/Regularizer/mul_1Mul+dense_1/kernel/Regularizer/mul_1/x:output:0)dense_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2"
 dense_1/kernel/Regularizer/mul_1Є
 dense_1/kernel/Regularizer/add_1AddV2"dense_1/kernel/Regularizer/add:z:0$dense_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2"
 dense_1/kernel/Regularizer/add_1Е
IdentityIdentity*basemodel/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@2

IdentityЃ
NoOpNoOp"^basemodel/StatefulPartitionedCall.^dense_1/kernel/Regularizer/Abs/ReadVariableOp1^dense_1/kernel/Regularizer/Square/ReadVariableOp6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp9^stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOp6^stream_0_conv_2/kernel/Regularizer/Abs/ReadVariableOp9^stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp6^stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp9^stream_0_conv_3/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:€€€€€€€€€}: : : : : : : : : : : : : : : : : : : : : : : : 2F
!basemodel/StatefulPartitionedCall!basemodel/StatefulPartitionedCall2^
-dense_1/kernel/Regularizer/Abs/ReadVariableOp-dense_1/kernel/Regularizer/Abs/ReadVariableOp2d
0dense_1/kernel/Regularizer/Square/ReadVariableOp0dense_1/kernel/Regularizer/Square/ReadVariableOp2n
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp2t
8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOp8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOp2n
5stream_0_conv_2/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_2/kernel/Regularizer/Abs/ReadVariableOp2t
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp2n
5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp2t
8stream_0_conv_3/kernel/Regularizer/Square/ReadVariableOp8stream_0_conv_3/kernel/Regularizer/Square/ReadVariableOp:X T
+
_output_shapes
:€€€€€€€€€}
%
_user_specified_nameleft_inputs
ў
J
.__inference_activation_1_layer_call_fn_3297871

inputs
identityќ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€>@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_activation_1_layer_call_and_return_conditional_losses_32930322
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:€€€€€€€€€>@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€>@:S O
+
_output_shapes
:€€€€€€€€€>@
 
_user_specified_nameinputs
м
“
7__inference_batch_normalization_1_layer_call_fn_3297745

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityИҐStatefulPartitionedCall£
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€>@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *[
fVRT
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_32930172
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€>@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€>@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:€€€€€€€€€>@
 
_user_specified_nameinputs
Л
k
O__inference_dense_activation_1_layer_call_and_return_conditional_losses_3298385

inputs
identityZ
IdentityIdentityinputs*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:€€€€€€€€€@:O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
©
k
O__inference_stream_0_maxpool_2_layer_call_and_return_conditional_losses_3293041

inputs
identityb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dimБ

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*/
_output_shapes
:€€€€€€€€€>@2

ExpandDimsЯ
MaxPoolMaxPoolExpandDims:output:0*/
_output_shapes
:€€€€€€€€€@*
ksize
*
paddingVALID*
strides
2	
MaxPool|
SqueezeSqueezeMaxPool:output:0*
T0*+
_output_shapes
:€€€€€€€€€@*
squeeze_dims
2	
Squeezeh
IdentityIdentitySqueeze:output:0*
T0*+
_output_shapes
:€€€€€€€€€@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€>@:S O
+
_output_shapes
:€€€€€€€€€>@
 
_user_specified_nameinputs
щi
Ї
#__inference__traced_restore_3298642
file_prefix=
'assignvariableop_stream_0_conv_1_kernel:@5
'assignvariableop_1_stream_0_conv_1_bias:@:
,assignvariableop_2_batch_normalization_gamma:@9
+assignvariableop_3_batch_normalization_beta:@?
)assignvariableop_4_stream_0_conv_2_kernel:@@5
'assignvariableop_5_stream_0_conv_2_bias:@<
.assignvariableop_6_batch_normalization_1_gamma:@;
-assignvariableop_7_batch_normalization_1_beta:@?
)assignvariableop_8_stream_0_conv_3_kernel:@@5
'assignvariableop_9_stream_0_conv_3_bias:@=
/assignvariableop_10_batch_normalization_2_gamma:@<
.assignvariableop_11_batch_normalization_2_beta:@4
"assignvariableop_12_dense_1_kernel:@@.
 assignvariableop_13_dense_1_bias:@=
/assignvariableop_14_batch_normalization_3_gamma:@<
.assignvariableop_15_batch_normalization_3_beta:@A
3assignvariableop_16_batch_normalization_moving_mean:@E
7assignvariableop_17_batch_normalization_moving_variance:@C
5assignvariableop_18_batch_normalization_1_moving_mean:@G
9assignvariableop_19_batch_normalization_1_moving_variance:@C
5assignvariableop_20_batch_normalization_2_moving_mean:@G
9assignvariableop_21_batch_normalization_2_moving_variance:@C
5assignvariableop_22_batch_normalization_3_moving_mean:@G
9assignvariableop_23_batch_normalization_3_moving_variance:@
identity_25ИҐAssignVariableOpҐAssignVariableOp_1ҐAssignVariableOp_10ҐAssignVariableOp_11ҐAssignVariableOp_12ҐAssignVariableOp_13ҐAssignVariableOp_14ҐAssignVariableOp_15ҐAssignVariableOp_16ҐAssignVariableOp_17ҐAssignVariableOp_18ҐAssignVariableOp_19ҐAssignVariableOp_2ҐAssignVariableOp_20ҐAssignVariableOp_21ҐAssignVariableOp_22ҐAssignVariableOp_23ҐAssignVariableOp_3ҐAssignVariableOp_4ҐAssignVariableOp_5ҐAssignVariableOp_6ҐAssignVariableOp_7ҐAssignVariableOp_8ҐAssignVariableOp_9У

RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Я	
valueХ	BТ	B0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/14/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/15/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesј
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*E
value<B:B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices®
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

Identity¶
AssignVariableOpAssignVariableOp'assignvariableop_stream_0_conv_1_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1ђ
AssignVariableOp_1AssignVariableOp'assignvariableop_1_stream_0_conv_1_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2±
AssignVariableOp_2AssignVariableOp,assignvariableop_2_batch_normalization_gammaIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3∞
AssignVariableOp_3AssignVariableOp+assignvariableop_3_batch_normalization_betaIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4Ѓ
AssignVariableOp_4AssignVariableOp)assignvariableop_4_stream_0_conv_2_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5ђ
AssignVariableOp_5AssignVariableOp'assignvariableop_5_stream_0_conv_2_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6≥
AssignVariableOp_6AssignVariableOp.assignvariableop_6_batch_normalization_1_gammaIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7≤
AssignVariableOp_7AssignVariableOp-assignvariableop_7_batch_normalization_1_betaIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8Ѓ
AssignVariableOp_8AssignVariableOp)assignvariableop_8_stream_0_conv_3_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9ђ
AssignVariableOp_9AssignVariableOp'assignvariableop_9_stream_0_conv_3_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10Ј
AssignVariableOp_10AssignVariableOp/assignvariableop_10_batch_normalization_2_gammaIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11ґ
AssignVariableOp_11AssignVariableOp.assignvariableop_11_batch_normalization_2_betaIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12™
AssignVariableOp_12AssignVariableOp"assignvariableop_12_dense_1_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13®
AssignVariableOp_13AssignVariableOp assignvariableop_13_dense_1_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14Ј
AssignVariableOp_14AssignVariableOp/assignvariableop_14_batch_normalization_3_gammaIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15ґ
AssignVariableOp_15AssignVariableOp.assignvariableop_15_batch_normalization_3_betaIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16ї
AssignVariableOp_16AssignVariableOp3assignvariableop_16_batch_normalization_moving_meanIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17њ
AssignVariableOp_17AssignVariableOp7assignvariableop_17_batch_normalization_moving_varianceIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18љ
AssignVariableOp_18AssignVariableOp5assignvariableop_18_batch_normalization_1_moving_meanIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19Ѕ
AssignVariableOp_19AssignVariableOp9assignvariableop_19_batch_normalization_1_moving_varianceIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20љ
AssignVariableOp_20AssignVariableOp5assignvariableop_20_batch_normalization_2_moving_meanIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21Ѕ
AssignVariableOp_21AssignVariableOp9assignvariableop_21_batch_normalization_2_moving_varianceIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22љ
AssignVariableOp_22AssignVariableOp5assignvariableop_22_batch_normalization_3_moving_meanIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23Ѕ
AssignVariableOp_23AssignVariableOp9assignvariableop_23_batch_normalization_3_moving_varianceIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_239
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpо
Identity_24Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_24f
Identity_25IdentityIdentity_24:output:0^NoOp_1*
T0*
_output_shapes
: 2
Identity_25÷
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
о
k
L__inference_stream_0_drop_1_layer_call_and_return_conditional_losses_3293574

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *Ђ™™?2
dropout/Constw
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:€€€€€€€€€>@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape”
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:€€€€€€€€€>@*
dtype0*
seedЈ*
seed2Є2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А>2
dropout/GreaterEqual/y¬
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:€€€€€€€€€>@2
dropout/GreaterEqualГ
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:€€€€€€€€€>@2
dropout/Cast~
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:€€€€€€€€€>@2
dropout/Mul_1i
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:€€€€€€€€€>@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€>@:S O
+
_output_shapes
:€€€€€€€€€>@
 
_user_specified_nameinputs
’
P
4__inference_dense_activation_1_layer_call_fn_3298381

inputs
identity–
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
GPU2*0J 8В *X
fSRQ
O__inference_dense_activation_1_layer_call_and_return_conditional_losses_32931962
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:€€€€€€€€€@:O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
х!
Ў
D__inference_dense_1_layer_call_and_return_conditional_losses_3293177

inputs0
matmul_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpҐ-dense_1/kernel/Regularizer/Abs/ReadVariableOpҐ0dense_1/kernel/Regularizer/Square/ReadVariableOpН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2	
BiasAddЙ
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dense_1/kernel/Regularizer/Constљ
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOpІ
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:@@2 
dense_1/kernel/Regularizer/AbsЩ
"dense_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_1/kernel/Regularizer/Const_1є
dense_1/kernel/Regularizer/SumSum"dense_1/kernel/Regularizer/Abs:y:0+dense_1/kernel/Regularizer/Const_1:output:0*
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
dense_1/kernel/Regularizer/mulє
dense_1/kernel/Regularizer/addAddV2)dense_1/kernel/Regularizer/Const:output:0"dense_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/add√
0dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype022
0dense_1/kernel/Regularizer/Square/ReadVariableOp≥
!dense_1/kernel/Regularizer/SquareSquare8dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@@2#
!dense_1/kernel/Regularizer/SquareЩ
"dense_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_1/kernel/Regularizer/Const_2ј
 dense_1/kernel/Regularizer/Sum_1Sum%dense_1/kernel/Regularizer/Square:y:0+dense_1/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2"
 dense_1/kernel/Regularizer/Sum_1Н
"dense_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2$
"dense_1/kernel/Regularizer/mul_1/xƒ
 dense_1/kernel/Regularizer/mul_1Mul+dense_1/kernel/Regularizer/mul_1/x:output:0)dense_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2"
 dense_1/kernel/Regularizer/mul_1Є
 dense_1/kernel/Regularizer/add_1AddV2"dense_1/kernel/Regularizer/add:z:0$dense_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2"
 dense_1/kernel/Regularizer/add_1k
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@2

Identityв
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp.^dense_1/kernel/Regularizer/Abs/ReadVariableOp1^dense_1/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2^
-dense_1/kernel/Regularizer/Abs/ReadVariableOp-dense_1/kernel/Regularizer/Abs/ReadVariableOp2d
0dense_1/kernel/Regularizer/Square/ReadVariableOp0dense_1/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
ж
–
5__inference_batch_normalization_layer_call_fn_3297481

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityИҐStatefulPartitionedCallЯ
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
GPU2*0J 8В *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_32936372
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
о
k
L__inference_stream_0_drop_2_layer_call_and_return_conditional_losses_3297929

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *Ђ™™?2
dropout/Constw
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:€€€€€€€€€@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape”
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:€€€€€€€€€@*
dtype0*
seedЈ*
seed2є2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А>2
dropout/GreaterEqual/y¬
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:€€€€€€€€€@2
dropout/GreaterEqualГ
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:€€€€€€€€€@2
dropout/Cast~
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:€€€€€€€€€@2
dropout/Mul_1i
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:€€€€€€€€€@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€@:S O
+
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
©
k
O__inference_stream_0_maxpool_1_layer_call_and_return_conditional_losses_3292953

inputs
identityb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dimБ

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*/
_output_shapes
:€€€€€€€€€}@2

ExpandDimsЯ
MaxPoolMaxPoolExpandDims:output:0*/
_output_shapes
:€€€€€€€€€>@*
ksize
*
paddingVALID*
strides
2	
MaxPool|
SqueezeSqueezeMaxPool:output:0*
T0*+
_output_shapes
:€€€€€€€€€>@*
squeeze_dims
2	
Squeezeh
IdentityIdentitySqueeze:output:0*
T0*+
_output_shapes
:€€€€€€€€€>@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€}@:S O
+
_output_shapes
:€€€€€€€€€}@
 
_user_specified_nameinputs
ґ
ѓ
P__inference_batch_normalization_layer_call_and_return_conditional_losses_3292128

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
Ш
Ґ
1__inference_stream_0_conv_2_layer_call_fn_3297676

inputs
unknown:@@
	unknown_0:@
identityИҐStatefulPartitionedCallГ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€>@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_stream_0_conv_2_layer_call_and_return_conditional_losses_32929922
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€>@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€>@: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:€€€€€€€€€>@
 
_user_specified_nameinputs
З
в
%__inference_signature_wrapper_3295587
left_inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:@
	unknown_5:@@
	unknown_6:@
	unknown_7:@
	unknown_8:@
	unknown_9:@

unknown_10:@ 

unknown_11:@@

unknown_12:@

unknown_13:@

unknown_14:@

unknown_15:@

unknown_16:@

unknown_17:@@

unknown_18:@

unknown_19:@

unknown_20:@

unknown_21:@

unknown_22:@
identityИҐStatefulPartitionedCallЕ
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
:€€€€€€€€€@*:
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *+
f&R$
"__inference__wrapped_model_32921042
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:€€€€€€€€€}: : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
+
_output_shapes
:€€€€€€€€€}
%
_user_specified_nameleft_inputs
Ц
k
O__inference_stream_0_maxpool_2_layer_call_and_return_conditional_losses_3297894

inputs
identityb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dimУ

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2

ExpandDims±
MaxPoolMaxPoolExpandDims:output:0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*
ksize
*
paddingVALID*
strides
2	
MaxPoolО
SqueezeSqueezeMaxPool:output:0*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€*
squeeze_dims
2	
Squeezez
IdentityIdentitySqueeze:output:0*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€:e a
=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
ў
J
.__inference_activation_2_layer_call_fn_3298148

inputs
identityќ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_activation_2_layer_call_and_return_conditional_losses_32931202
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:€€€€€€€€€@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€@:S O
+
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
Ц
k
O__inference_stream_0_maxpool_1_layer_call_and_return_conditional_losses_3297617

inputs
identityb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dimУ

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2

ExpandDims±
MaxPoolMaxPoolExpandDims:output:0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*
ksize
*
paddingVALID*
strides
2	
MaxPoolО
SqueezeSqueezeMaxPool:output:0*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€*
squeeze_dims
2	
Squeezez
IdentityIdentitySqueeze:output:0*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€:e a
=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
Й
j
L__inference_stream_0_drop_2_layer_call_and_return_conditional_losses_3297917

inputs

identity_1^
IdentityIdentityinputs*
T0*+
_output_shapes
:€€€€€€€€€@2

Identitym

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:€€€€€€€€€@2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€@:S O
+
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
Ш
Ґ
1__inference_stream_0_conv_3_layer_call_fn_3297953

inputs
unknown:@@
	unknown_0:@
identityИҐStatefulPartitionedCallГ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_stream_0_conv_3_layer_call_and_return_conditional_losses_32930802
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€@: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
еz
ч

D__inference_model_1_layer_call_and_return_conditional_losses_3294545

inputs'
basemodel_3294435:@
basemodel_3294437:@
basemodel_3294439:@
basemodel_3294441:@
basemodel_3294443:@
basemodel_3294445:@'
basemodel_3294447:@@
basemodel_3294449:@
basemodel_3294451:@
basemodel_3294453:@
basemodel_3294455:@
basemodel_3294457:@'
basemodel_3294459:@@
basemodel_3294461:@
basemodel_3294463:@
basemodel_3294465:@
basemodel_3294467:@
basemodel_3294469:@#
basemodel_3294471:@@
basemodel_3294473:@
basemodel_3294475:@
basemodel_3294477:@
basemodel_3294479:@
basemodel_3294481:@
identityИҐ!basemodel/StatefulPartitionedCallҐ-dense_1/kernel/Regularizer/Abs/ReadVariableOpҐ0dense_1/kernel/Regularizer/Square/ReadVariableOpҐ5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpҐ8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOpҐ5stream_0_conv_2/kernel/Regularizer/Abs/ReadVariableOpҐ8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOpҐ5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOpҐ8stream_0_conv_3/kernel/Regularizer/Square/ReadVariableOpн
!basemodel/StatefulPartitionedCallStatefulPartitionedCallinputsbasemodel_3294435basemodel_3294437basemodel_3294439basemodel_3294441basemodel_3294443basemodel_3294445basemodel_3294447basemodel_3294449basemodel_3294451basemodel_3294453basemodel_3294455basemodel_3294457basemodel_3294459basemodel_3294461basemodel_3294463basemodel_3294465basemodel_3294467basemodel_3294469basemodel_3294471basemodel_3294473basemodel_3294475basemodel_3294477basemodel_3294479basemodel_3294481*$
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@*:
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_basemodel_layer_call_and_return_conditional_losses_32944342#
!basemodel/StatefulPartitionedCallЩ
(stream_0_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(stream_0_conv_1/kernel/Regularizer/Constƒ
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpbasemodel_3294435*"
_output_shapes
:@*
dtype027
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp√
&stream_0_conv_1/kernel/Regularizer/AbsAbs=stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_0_conv_1/kernel/Regularizer/Abs≠
*stream_0_conv_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_0_conv_1/kernel/Regularizer/Const_1ў
&stream_0_conv_1/kernel/Regularizer/SumSum*stream_0_conv_1/kernel/Regularizer/Abs:y:03stream_0_conv_1/kernel/Regularizer/Const_1:output:0*
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
&stream_0_conv_1/kernel/Regularizer/mulў
&stream_0_conv_1/kernel/Regularizer/addAddV21stream_0_conv_1/kernel/Regularizer/Const:output:0*stream_0_conv_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/add 
8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpbasemodel_3294435*"
_output_shapes
:@*
dtype02:
8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOpѕ
)stream_0_conv_1/kernel/Regularizer/SquareSquare@stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2+
)stream_0_conv_1/kernel/Regularizer/Square≠
*stream_0_conv_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_0_conv_1/kernel/Regularizer/Const_2а
(stream_0_conv_1/kernel/Regularizer/Sum_1Sum-stream_0_conv_1/kernel/Regularizer/Square:y:03stream_0_conv_1/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2*
(stream_0_conv_1/kernel/Regularizer/Sum_1Э
*stream_0_conv_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2,
*stream_0_conv_1/kernel/Regularizer/mul_1/xд
(stream_0_conv_1/kernel/Regularizer/mul_1Mul3stream_0_conv_1/kernel/Regularizer/mul_1/x:output:01stream_0_conv_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2*
(stream_0_conv_1/kernel/Regularizer/mul_1Ў
(stream_0_conv_1/kernel/Regularizer/add_1AddV2*stream_0_conv_1/kernel/Regularizer/add:z:0,stream_0_conv_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2*
(stream_0_conv_1/kernel/Regularizer/add_1Щ
(stream_0_conv_2/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(stream_0_conv_2/kernel/Regularizer/Constƒ
5stream_0_conv_2/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpbasemodel_3294447*"
_output_shapes
:@@*
dtype027
5stream_0_conv_2/kernel/Regularizer/Abs/ReadVariableOp√
&stream_0_conv_2/kernel/Regularizer/AbsAbs=stream_0_conv_2/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@@2(
&stream_0_conv_2/kernel/Regularizer/Abs≠
*stream_0_conv_2/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_0_conv_2/kernel/Regularizer/Const_1ў
&stream_0_conv_2/kernel/Regularizer/SumSum*stream_0_conv_2/kernel/Regularizer/Abs:y:03stream_0_conv_2/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_2/kernel/Regularizer/SumЩ
(stream_0_conv_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2*
(stream_0_conv_2/kernel/Regularizer/mul/x№
&stream_0_conv_2/kernel/Regularizer/mulMul1stream_0_conv_2/kernel/Regularizer/mul/x:output:0/stream_0_conv_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_2/kernel/Regularizer/mulў
&stream_0_conv_2/kernel/Regularizer/addAddV21stream_0_conv_2/kernel/Regularizer/Const:output:0*stream_0_conv_2/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&stream_0_conv_2/kernel/Regularizer/add 
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpbasemodel_3294447*"
_output_shapes
:@@*
dtype02:
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOpѕ
)stream_0_conv_2/kernel/Regularizer/SquareSquare@stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@@2+
)stream_0_conv_2/kernel/Regularizer/Square≠
*stream_0_conv_2/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_0_conv_2/kernel/Regularizer/Const_2а
(stream_0_conv_2/kernel/Regularizer/Sum_1Sum-stream_0_conv_2/kernel/Regularizer/Square:y:03stream_0_conv_2/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2*
(stream_0_conv_2/kernel/Regularizer/Sum_1Э
*stream_0_conv_2/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2,
*stream_0_conv_2/kernel/Regularizer/mul_1/xд
(stream_0_conv_2/kernel/Regularizer/mul_1Mul3stream_0_conv_2/kernel/Regularizer/mul_1/x:output:01stream_0_conv_2/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2*
(stream_0_conv_2/kernel/Regularizer/mul_1Ў
(stream_0_conv_2/kernel/Regularizer/add_1AddV2*stream_0_conv_2/kernel/Regularizer/add:z:0,stream_0_conv_2/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2*
(stream_0_conv_2/kernel/Regularizer/add_1Щ
(stream_0_conv_3/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(stream_0_conv_3/kernel/Regularizer/Constƒ
5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpbasemodel_3294459*"
_output_shapes
:@@*
dtype027
5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp√
&stream_0_conv_3/kernel/Regularizer/AbsAbs=stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@@2(
&stream_0_conv_3/kernel/Regularizer/Abs≠
*stream_0_conv_3/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_0_conv_3/kernel/Regularizer/Const_1ў
&stream_0_conv_3/kernel/Regularizer/SumSum*stream_0_conv_3/kernel/Regularizer/Abs:y:03stream_0_conv_3/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_3/kernel/Regularizer/SumЩ
(stream_0_conv_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2*
(stream_0_conv_3/kernel/Regularizer/mul/x№
&stream_0_conv_3/kernel/Regularizer/mulMul1stream_0_conv_3/kernel/Regularizer/mul/x:output:0/stream_0_conv_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_3/kernel/Regularizer/mulў
&stream_0_conv_3/kernel/Regularizer/addAddV21stream_0_conv_3/kernel/Regularizer/Const:output:0*stream_0_conv_3/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&stream_0_conv_3/kernel/Regularizer/add 
8stream_0_conv_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpbasemodel_3294459*"
_output_shapes
:@@*
dtype02:
8stream_0_conv_3/kernel/Regularizer/Square/ReadVariableOpѕ
)stream_0_conv_3/kernel/Regularizer/SquareSquare@stream_0_conv_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@@2+
)stream_0_conv_3/kernel/Regularizer/Square≠
*stream_0_conv_3/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_0_conv_3/kernel/Regularizer/Const_2а
(stream_0_conv_3/kernel/Regularizer/Sum_1Sum-stream_0_conv_3/kernel/Regularizer/Square:y:03stream_0_conv_3/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2*
(stream_0_conv_3/kernel/Regularizer/Sum_1Э
*stream_0_conv_3/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2,
*stream_0_conv_3/kernel/Regularizer/mul_1/xд
(stream_0_conv_3/kernel/Regularizer/mul_1Mul3stream_0_conv_3/kernel/Regularizer/mul_1/x:output:01stream_0_conv_3/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2*
(stream_0_conv_3/kernel/Regularizer/mul_1Ў
(stream_0_conv_3/kernel/Regularizer/add_1AddV2*stream_0_conv_3/kernel/Regularizer/add:z:0,stream_0_conv_3/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2*
(stream_0_conv_3/kernel/Regularizer/add_1Й
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dense_1/kernel/Regularizer/Const∞
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpbasemodel_3294471*
_output_shapes

:@@*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOpІ
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:@@2 
dense_1/kernel/Regularizer/AbsЩ
"dense_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_1/kernel/Regularizer/Const_1є
dense_1/kernel/Regularizer/SumSum"dense_1/kernel/Regularizer/Abs:y:0+dense_1/kernel/Regularizer/Const_1:output:0*
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
dense_1/kernel/Regularizer/mulє
dense_1/kernel/Regularizer/addAddV2)dense_1/kernel/Regularizer/Const:output:0"dense_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/addґ
0dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpbasemodel_3294471*
_output_shapes

:@@*
dtype022
0dense_1/kernel/Regularizer/Square/ReadVariableOp≥
!dense_1/kernel/Regularizer/SquareSquare8dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@@2#
!dense_1/kernel/Regularizer/SquareЩ
"dense_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_1/kernel/Regularizer/Const_2ј
 dense_1/kernel/Regularizer/Sum_1Sum%dense_1/kernel/Regularizer/Square:y:0+dense_1/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2"
 dense_1/kernel/Regularizer/Sum_1Н
"dense_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2$
"dense_1/kernel/Regularizer/mul_1/xƒ
 dense_1/kernel/Regularizer/mul_1Mul+dense_1/kernel/Regularizer/mul_1/x:output:0)dense_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2"
 dense_1/kernel/Regularizer/mul_1Є
 dense_1/kernel/Regularizer/add_1AddV2"dense_1/kernel/Regularizer/add:z:0$dense_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2"
 dense_1/kernel/Regularizer/add_1Е
IdentityIdentity*basemodel/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@2

IdentityЃ
NoOpNoOp"^basemodel/StatefulPartitionedCall.^dense_1/kernel/Regularizer/Abs/ReadVariableOp1^dense_1/kernel/Regularizer/Square/ReadVariableOp6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp9^stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOp6^stream_0_conv_2/kernel/Regularizer/Abs/ReadVariableOp9^stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp6^stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp9^stream_0_conv_3/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:€€€€€€€€€}: : : : : : : : : : : : : : : : : : : : : : : : 2F
!basemodel/StatefulPartitionedCall!basemodel/StatefulPartitionedCall2^
-dense_1/kernel/Regularizer/Abs/ReadVariableOp-dense_1/kernel/Regularizer/Abs/ReadVariableOp2d
0dense_1/kernel/Regularizer/Square/ReadVariableOp0dense_1/kernel/Regularizer/Square/ReadVariableOp2n
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp2t
8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOp8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOp2n
5stream_0_conv_2/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_2/kernel/Regularizer/Abs/ReadVariableOp2t
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp2n
5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp2t
8stream_0_conv_3/kernel/Regularizer/Square/ReadVariableOp8stream_0_conv_3/kernel/Regularizer/Square/ReadVariableOp:S O
+
_output_shapes
:€€€€€€€€€}
 
_user_specified_nameinputs
я
M
1__inference_stream_0_drop_2_layer_call_fn_3297907

inputs
identity—
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_stream_0_drop_2_layer_call_and_return_conditional_losses_32930482
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:€€€€€€€€€@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€@:S O
+
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
®
е
+__inference_basemodel_layer_call_fn_3293310
inputs_0
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:@
	unknown_5:@@
	unknown_6:@
	unknown_7:@
	unknown_8:@
	unknown_9:@

unknown_10:@ 

unknown_11:@@

unknown_12:@

unknown_13:@

unknown_14:@

unknown_15:@

unknown_16:@

unknown_17:@@

unknown_18:@

unknown_19:@

unknown_20:@

unknown_21:@

unknown_22:@
identityИҐStatefulPartitionedCall¶
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
:€€€€€€€€€@*:
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_basemodel_layer_call_and_return_conditional_losses_32932592
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:€€€€€€€€€}: : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
+
_output_shapes
:€€€€€€€€€}
"
_user_specified_name
inputs_0
©
k
O__inference_stream_0_maxpool_3_layer_call_and_return_conditional_losses_3293129

inputs
identityb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dimБ

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*/
_output_shapes
:€€€€€€€€€@2

ExpandDimsЯ
MaxPoolMaxPoolExpandDims:output:0*/
_output_shapes
:€€€€€€€€€@*
ksize
*
paddingVALID*
strides
2	
MaxPool|
SqueezeSqueezeMaxPool:output:0*
T0*+
_output_shapes
:€€€€€€€€€@*
squeeze_dims
2	
Squeezeh
IdentityIdentitySqueeze:output:0*
T0*+
_output_shapes
:€€€€€€€€€@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€@:S O
+
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
ц
±
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_3298342

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
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
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
batchnorm/subЕ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
batchnorm/add_1n
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity¬
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€@: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
у,
О
L__inference_stream_0_conv_1_layer_call_and_return_conditional_losses_3297429

inputsA
+conv1d_expanddims_1_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identityИҐBiasAdd/ReadVariableOpҐ"conv1d/ExpandDims_1/ReadVariableOpҐ5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpҐ8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOpy
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
BiasAddЩ
(stream_0_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(stream_0_conv_1/kernel/Regularizer/Constё
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype027
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp√
&stream_0_conv_1/kernel/Regularizer/AbsAbs=stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_0_conv_1/kernel/Regularizer/Abs≠
*stream_0_conv_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_0_conv_1/kernel/Regularizer/Const_1ў
&stream_0_conv_1/kernel/Regularizer/SumSum*stream_0_conv_1/kernel/Regularizer/Abs:y:03stream_0_conv_1/kernel/Regularizer/Const_1:output:0*
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
&stream_0_conv_1/kernel/Regularizer/mulў
&stream_0_conv_1/kernel/Regularizer/addAddV21stream_0_conv_1/kernel/Regularizer/Const:output:0*stream_0_conv_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/addд
8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02:
8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOpѕ
)stream_0_conv_1/kernel/Regularizer/SquareSquare@stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2+
)stream_0_conv_1/kernel/Regularizer/Square≠
*stream_0_conv_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_0_conv_1/kernel/Regularizer/Const_2а
(stream_0_conv_1/kernel/Regularizer/Sum_1Sum-stream_0_conv_1/kernel/Regularizer/Square:y:03stream_0_conv_1/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2*
(stream_0_conv_1/kernel/Regularizer/Sum_1Э
*stream_0_conv_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2,
*stream_0_conv_1/kernel/Regularizer/mul_1/xд
(stream_0_conv_1/kernel/Regularizer/mul_1Mul3stream_0_conv_1/kernel/Regularizer/mul_1/x:output:01stream_0_conv_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2*
(stream_0_conv_1/kernel/Regularizer/mul_1Ў
(stream_0_conv_1/kernel/Regularizer/add_1AddV2*stream_0_conv_1/kernel/Regularizer/add:z:0,stream_0_conv_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2*
(stream_0_conv_1/kernel/Regularizer/add_1o
IdentityIdentityBiasAdd:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€}@2

Identity€
NoOpNoOp^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp9^stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOp*"
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
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp2t
8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOp8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOp:S O
+
_output_shapes
:€€€€€€€€€}
 
_user_specified_nameinputs
И
ѓ
P__inference_batch_normalization_layer_call_and_return_conditional_losses_3297555

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
Ѓ
P
4__inference_stream_0_maxpool_1_layer_call_fn_3297604

inputs
identityж
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_stream_0_maxpool_1_layer_call_and_return_conditional_losses_32922782
PartitionedCallВ
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€:e a
=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
Ц
k
O__inference_stream_0_maxpool_2_layer_call_and_return_conditional_losses_3292468

inputs
identityb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dimУ

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2

ExpandDims±
MaxPoolMaxPoolExpandDims:output:0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*
ksize
*
paddingVALID*
strides
2	
MaxPoolО
SqueezeSqueezeMaxPool:output:0*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€*
squeeze_dims
2	
Squeezez
IdentityIdentitySqueeze:output:0*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€:e a
=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
Н
n
P__inference_stream_0_input_drop_layer_call_and_return_conditional_losses_3297363

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
†
е
+__inference_basemodel_layer_call_fn_3296426
inputs_0
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:@
	unknown_5:@@
	unknown_6:@
	unknown_7:@
	unknown_8:@
	unknown_9:@

unknown_10:@ 

unknown_11:@@

unknown_12:@

unknown_13:@

unknown_14:@

unknown_15:@

unknown_16:@

unknown_17:@@

unknown_18:@

unknown_19:@

unknown_20:@

unknown_21:@

unknown_22:@
identityИҐStatefulPartitionedCallЮ
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
:€€€€€€€€€@*2
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_basemodel_layer_call_and_return_conditional_losses_32949232
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:€€€€€€€€€}: : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
+
_output_shapes
:€€€€€€€€€}
"
_user_specified_name
inputs/0
з
e
I__inference_activation_1_layer_call_and_return_conditional_losses_3293032

inputs
identityR
TanhTanhinputs*
T0*+
_output_shapes
:€€€€€€€€€>@2
Tanh`
IdentityIdentityTanh:y:0*
T0*+
_output_shapes
:€€€€€€€€€>@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€>@:S O
+
_output_shapes
:€€€€€€€€€>@
 
_user_specified_nameinputs
Л	
–
5__inference_batch_normalization_layer_call_fn_3297455

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityИҐStatefulPartitionedCall®
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
GPU2*0J 8В *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_32921882
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
Ь
Ё
__inference_loss_fn_3_3298465H
6dense_1_kernel_regularizer_abs_readvariableop_resource:@@
identityИҐ-dense_1/kernel/Regularizer/Abs/ReadVariableOpҐ0dense_1/kernel/Regularizer/Square/ReadVariableOpЙ
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dense_1/kernel/Regularizer/Const’
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp6dense_1_kernel_regularizer_abs_readvariableop_resource*
_output_shapes

:@@*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOpІ
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:@@2 
dense_1/kernel/Regularizer/AbsЩ
"dense_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_1/kernel/Regularizer/Const_1є
dense_1/kernel/Regularizer/SumSum"dense_1/kernel/Regularizer/Abs:y:0+dense_1/kernel/Regularizer/Const_1:output:0*
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
dense_1/kernel/Regularizer/mulє
dense_1/kernel/Regularizer/addAddV2)dense_1/kernel/Regularizer/Const:output:0"dense_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/addџ
0dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp6dense_1_kernel_regularizer_abs_readvariableop_resource*
_output_shapes

:@@*
dtype022
0dense_1/kernel/Regularizer/Square/ReadVariableOp≥
!dense_1/kernel/Regularizer/SquareSquare8dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@@2#
!dense_1/kernel/Regularizer/SquareЩ
"dense_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_1/kernel/Regularizer/Const_2ј
 dense_1/kernel/Regularizer/Sum_1Sum%dense_1/kernel/Regularizer/Square:y:0+dense_1/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2"
 dense_1/kernel/Regularizer/Sum_1Н
"dense_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2$
"dense_1/kernel/Regularizer/mul_1/xƒ
 dense_1/kernel/Regularizer/mul_1Mul+dense_1/kernel/Regularizer/mul_1/x:output:0)dense_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2"
 dense_1/kernel/Regularizer/mul_1Є
 dense_1/kernel/Regularizer/add_1AddV2"dense_1/kernel/Regularizer/add:z:0$dense_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2"
 dense_1/kernel/Regularizer/add_1n
IdentityIdentity$dense_1/kernel/Regularizer/add_1:z:0^NoOp*
T0*
_output_shapes
: 2

Identity±
NoOpNoOp.^dense_1/kernel/Regularizer/Abs/ReadVariableOp1^dense_1/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2^
-dense_1/kernel/Regularizer/Abs/ReadVariableOp-dense_1/kernel/Regularizer/Abs/ReadVariableOp2d
0dense_1/kernel/Regularizer/Square/ReadVariableOp0dense_1/kernel/Regularizer/Square/ReadVariableOp
є+
л
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_3297812

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
о
k
L__inference_stream_0_drop_3_layer_call_and_return_conditional_losses_3293366

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *Ђ™™?2
dropout/Constw
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:€€€€€€€€€@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape”
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:€€€€€€€€€@*
dtype0*
seedЈ*
seed2Ї2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А>2
dropout/GreaterEqual/y¬
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:€€€€€€€€€@2
dropout/GreaterEqualГ
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:€€€€€€€€€@2
dropout/Cast~
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:€€€€€€€€€@2
dropout/Mul_1i
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:€€€€€€€€€@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€@:S O
+
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
е
c
G__inference_activation_layer_call_and_return_conditional_losses_3292944

inputs
identityR
TanhTanhinputs*
T0*+
_output_shapes
:€€€€€€€€€}@2
Tanh`
IdentityIdentityTanh:y:0*
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
ѕ
M
1__inference_dense_1_dropout_layer_call_fn_3298238

inputs
identityЌ
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
GPU2*0J 8В *U
fPRN
L__inference_dense_1_dropout_layer_call_and_return_conditional_losses_32933382
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:€€€€€€€€€@:O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
е
c
G__inference_activation_layer_call_and_return_conditional_losses_3297599

inputs
identityR
TanhTanhinputs*
T0*+
_output_shapes
:€€€€€€€€€}@2
Tanh`
IdentityIdentityTanh:y:0*
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
й
V
:__inference_global_average_pooling1d_layer_call_fn_3298216

inputs
identity÷
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
GPU2*0J 8В *^
fYRW
U__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_32931432
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€@:S O
+
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
з—
Ј
F__inference_basemodel_layer_call_and_return_conditional_losses_3293869

inputs-
stream_0_conv_1_3293740:@%
stream_0_conv_1_3293742:@)
batch_normalization_3293745:@)
batch_normalization_3293747:@)
batch_normalization_3293749:@)
batch_normalization_3293751:@-
stream_0_conv_2_3293757:@@%
stream_0_conv_2_3293759:@+
batch_normalization_1_3293762:@+
batch_normalization_1_3293764:@+
batch_normalization_1_3293766:@+
batch_normalization_1_3293768:@-
stream_0_conv_3_3293774:@@%
stream_0_conv_3_3293776:@+
batch_normalization_2_3293779:@+
batch_normalization_2_3293781:@+
batch_normalization_2_3293783:@+
batch_normalization_2_3293785:@!
dense_1_3293793:@@
dense_1_3293795:@+
batch_normalization_3_3293798:@+
batch_normalization_3_3293800:@+
batch_normalization_3_3293802:@+
batch_normalization_3_3293804:@
identityИҐ+batch_normalization/StatefulPartitionedCallҐ-batch_normalization_1/StatefulPartitionedCallҐ-batch_normalization_2/StatefulPartitionedCallҐ-batch_normalization_3/StatefulPartitionedCallҐdense_1/StatefulPartitionedCallҐ-dense_1/kernel/Regularizer/Abs/ReadVariableOpҐ0dense_1/kernel/Regularizer/Square/ReadVariableOpҐ'stream_0_conv_1/StatefulPartitionedCallҐ5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpҐ8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOpҐ'stream_0_conv_2/StatefulPartitionedCallҐ5stream_0_conv_2/kernel/Regularizer/Abs/ReadVariableOpҐ8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOpҐ'stream_0_conv_3/StatefulPartitionedCallҐ5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOpҐ8stream_0_conv_3/kernel/Regularizer/Square/ReadVariableOpҐ'stream_0_drop_1/StatefulPartitionedCallҐ'stream_0_drop_2/StatefulPartitionedCallҐ'stream_0_drop_3/StatefulPartitionedCallҐ+stream_0_input_drop/StatefulPartitionedCallХ
+stream_0_input_drop/StatefulPartitionedCallStatefulPartitionedCallinputs*
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
GPU2*0J 8В *Y
fTRR
P__inference_stream_0_input_drop_layer_call_and_return_conditional_losses_32936782-
+stream_0_input_drop/StatefulPartitionedCallп
'stream_0_conv_1/StatefulPartitionedCallStatefulPartitionedCall4stream_0_input_drop/StatefulPartitionedCall:output:0stream_0_conv_1_3293740stream_0_conv_1_3293742*
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
GPU2*0J 8В *U
fPRN
L__inference_stream_0_conv_1_layer_call_and_return_conditional_losses_32929042)
'stream_0_conv_1/StatefulPartitionedCallї
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall0stream_0_conv_1/StatefulPartitionedCall:output:0batch_normalization_3293745batch_normalization_3293747batch_normalization_3293749batch_normalization_3293751*
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
GPU2*0J 8В *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_32936372-
+batch_normalization/StatefulPartitionedCallР
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
GPU2*0J 8В *P
fKRI
G__inference_activation_layer_call_and_return_conditional_losses_32929442
activation/PartitionedCallЧ
"stream_0_maxpool_1/PartitionedCallPartitionedCall#activation/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€>@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_stream_0_maxpool_1_layer_call_and_return_conditional_losses_32929532$
"stream_0_maxpool_1/PartitionedCall№
'stream_0_drop_1/StatefulPartitionedCallStatefulPartitionedCall+stream_0_maxpool_1/PartitionedCall:output:0,^stream_0_input_drop/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€>@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_stream_0_drop_1_layer_call_and_return_conditional_losses_32935742)
'stream_0_drop_1/StatefulPartitionedCallл
'stream_0_conv_2/StatefulPartitionedCallStatefulPartitionedCall0stream_0_drop_1/StatefulPartitionedCall:output:0stream_0_conv_2_3293757stream_0_conv_2_3293759*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€>@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_stream_0_conv_2_layer_call_and_return_conditional_losses_32929922)
'stream_0_conv_2/StatefulPartitionedCall…
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall0stream_0_conv_2/StatefulPartitionedCall:output:0batch_normalization_1_3293762batch_normalization_1_3293764batch_normalization_1_3293766batch_normalization_1_3293768*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€>@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *[
fVRT
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_32935332/
-batch_normalization_1/StatefulPartitionedCallШ
activation_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€>@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_activation_1_layer_call_and_return_conditional_losses_32930322
activation_1/PartitionedCallЩ
"stream_0_maxpool_2/PartitionedCallPartitionedCall%activation_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_stream_0_maxpool_2_layer_call_and_return_conditional_losses_32930412$
"stream_0_maxpool_2/PartitionedCallЎ
'stream_0_drop_2/StatefulPartitionedCallStatefulPartitionedCall+stream_0_maxpool_2/PartitionedCall:output:0(^stream_0_drop_1/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_stream_0_drop_2_layer_call_and_return_conditional_losses_32934702)
'stream_0_drop_2/StatefulPartitionedCallл
'stream_0_conv_3/StatefulPartitionedCallStatefulPartitionedCall0stream_0_drop_2/StatefulPartitionedCall:output:0stream_0_conv_3_3293774stream_0_conv_3_3293776*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_stream_0_conv_3_layer_call_and_return_conditional_losses_32930802)
'stream_0_conv_3/StatefulPartitionedCall…
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall0stream_0_conv_3/StatefulPartitionedCall:output:0batch_normalization_2_3293779batch_normalization_2_3293781batch_normalization_2_3293783batch_normalization_2_3293785*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *[
fVRT
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_32934292/
-batch_normalization_2/StatefulPartitionedCallШ
activation_2/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_activation_2_layer_call_and_return_conditional_losses_32931202
activation_2/PartitionedCallЩ
"stream_0_maxpool_3/PartitionedCallPartitionedCall%activation_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_stream_0_maxpool_3_layer_call_and_return_conditional_losses_32931292$
"stream_0_maxpool_3/PartitionedCallЎ
'stream_0_drop_3/StatefulPartitionedCallStatefulPartitionedCall+stream_0_maxpool_3/PartitionedCall:output:0(^stream_0_drop_2/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_stream_0_drop_3_layer_call_and_return_conditional_losses_32933662)
'stream_0_drop_3/StatefulPartitionedCall≤
(global_average_pooling1d/PartitionedCallPartitionedCall0stream_0_drop_3/StatefulPartitionedCall:output:0*
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
GPU2*0J 8В *^
fYRW
U__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_32931432*
(global_average_pooling1d/PartitionedCallШ
dense_1_dropout/PartitionedCallPartitionedCall1global_average_pooling1d/PartitionedCall:output:0*
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
GPU2*0J 8В *U
fPRN
L__inference_dense_1_dropout_layer_call_and_return_conditional_losses_32933382!
dense_1_dropout/PartitionedCallЈ
dense_1/StatefulPartitionedCallStatefulPartitionedCall(dense_1_dropout/PartitionedCall:output:0dense_1_3293793dense_1_3293795*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_32931772!
dense_1/StatefulPartitionedCallљ
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0batch_normalization_3_3293798batch_normalization_3_3293800batch_normalization_3_3293802batch_normalization_3_3293804*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *[
fVRT
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_32927822/
-batch_normalization_3/StatefulPartitionedCall¶
"dense_activation_1/PartitionedCallPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0*
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
GPU2*0J 8В *X
fSRQ
O__inference_dense_activation_1_layer_call_and_return_conditional_losses_32931962$
"dense_activation_1/PartitionedCallЩ
(stream_0_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(stream_0_conv_1/kernel/Regularizer/Const 
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpstream_0_conv_1_3293740*"
_output_shapes
:@*
dtype027
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp√
&stream_0_conv_1/kernel/Regularizer/AbsAbs=stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_0_conv_1/kernel/Regularizer/Abs≠
*stream_0_conv_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_0_conv_1/kernel/Regularizer/Const_1ў
&stream_0_conv_1/kernel/Regularizer/SumSum*stream_0_conv_1/kernel/Regularizer/Abs:y:03stream_0_conv_1/kernel/Regularizer/Const_1:output:0*
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
&stream_0_conv_1/kernel/Regularizer/mulў
&stream_0_conv_1/kernel/Regularizer/addAddV21stream_0_conv_1/kernel/Regularizer/Const:output:0*stream_0_conv_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/add–
8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpstream_0_conv_1_3293740*"
_output_shapes
:@*
dtype02:
8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOpѕ
)stream_0_conv_1/kernel/Regularizer/SquareSquare@stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2+
)stream_0_conv_1/kernel/Regularizer/Square≠
*stream_0_conv_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_0_conv_1/kernel/Regularizer/Const_2а
(stream_0_conv_1/kernel/Regularizer/Sum_1Sum-stream_0_conv_1/kernel/Regularizer/Square:y:03stream_0_conv_1/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2*
(stream_0_conv_1/kernel/Regularizer/Sum_1Э
*stream_0_conv_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2,
*stream_0_conv_1/kernel/Regularizer/mul_1/xд
(stream_0_conv_1/kernel/Regularizer/mul_1Mul3stream_0_conv_1/kernel/Regularizer/mul_1/x:output:01stream_0_conv_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2*
(stream_0_conv_1/kernel/Regularizer/mul_1Ў
(stream_0_conv_1/kernel/Regularizer/add_1AddV2*stream_0_conv_1/kernel/Regularizer/add:z:0,stream_0_conv_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2*
(stream_0_conv_1/kernel/Regularizer/add_1Щ
(stream_0_conv_2/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(stream_0_conv_2/kernel/Regularizer/Const 
5stream_0_conv_2/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpstream_0_conv_2_3293757*"
_output_shapes
:@@*
dtype027
5stream_0_conv_2/kernel/Regularizer/Abs/ReadVariableOp√
&stream_0_conv_2/kernel/Regularizer/AbsAbs=stream_0_conv_2/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@@2(
&stream_0_conv_2/kernel/Regularizer/Abs≠
*stream_0_conv_2/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_0_conv_2/kernel/Regularizer/Const_1ў
&stream_0_conv_2/kernel/Regularizer/SumSum*stream_0_conv_2/kernel/Regularizer/Abs:y:03stream_0_conv_2/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_2/kernel/Regularizer/SumЩ
(stream_0_conv_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2*
(stream_0_conv_2/kernel/Regularizer/mul/x№
&stream_0_conv_2/kernel/Regularizer/mulMul1stream_0_conv_2/kernel/Regularizer/mul/x:output:0/stream_0_conv_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_2/kernel/Regularizer/mulў
&stream_0_conv_2/kernel/Regularizer/addAddV21stream_0_conv_2/kernel/Regularizer/Const:output:0*stream_0_conv_2/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&stream_0_conv_2/kernel/Regularizer/add–
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpstream_0_conv_2_3293757*"
_output_shapes
:@@*
dtype02:
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOpѕ
)stream_0_conv_2/kernel/Regularizer/SquareSquare@stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@@2+
)stream_0_conv_2/kernel/Regularizer/Square≠
*stream_0_conv_2/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_0_conv_2/kernel/Regularizer/Const_2а
(stream_0_conv_2/kernel/Regularizer/Sum_1Sum-stream_0_conv_2/kernel/Regularizer/Square:y:03stream_0_conv_2/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2*
(stream_0_conv_2/kernel/Regularizer/Sum_1Э
*stream_0_conv_2/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2,
*stream_0_conv_2/kernel/Regularizer/mul_1/xд
(stream_0_conv_2/kernel/Regularizer/mul_1Mul3stream_0_conv_2/kernel/Regularizer/mul_1/x:output:01stream_0_conv_2/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2*
(stream_0_conv_2/kernel/Regularizer/mul_1Ў
(stream_0_conv_2/kernel/Regularizer/add_1AddV2*stream_0_conv_2/kernel/Regularizer/add:z:0,stream_0_conv_2/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2*
(stream_0_conv_2/kernel/Regularizer/add_1Щ
(stream_0_conv_3/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(stream_0_conv_3/kernel/Regularizer/Const 
5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpstream_0_conv_3_3293774*"
_output_shapes
:@@*
dtype027
5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp√
&stream_0_conv_3/kernel/Regularizer/AbsAbs=stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@@2(
&stream_0_conv_3/kernel/Regularizer/Abs≠
*stream_0_conv_3/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_0_conv_3/kernel/Regularizer/Const_1ў
&stream_0_conv_3/kernel/Regularizer/SumSum*stream_0_conv_3/kernel/Regularizer/Abs:y:03stream_0_conv_3/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_3/kernel/Regularizer/SumЩ
(stream_0_conv_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2*
(stream_0_conv_3/kernel/Regularizer/mul/x№
&stream_0_conv_3/kernel/Regularizer/mulMul1stream_0_conv_3/kernel/Regularizer/mul/x:output:0/stream_0_conv_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_3/kernel/Regularizer/mulў
&stream_0_conv_3/kernel/Regularizer/addAddV21stream_0_conv_3/kernel/Regularizer/Const:output:0*stream_0_conv_3/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&stream_0_conv_3/kernel/Regularizer/add–
8stream_0_conv_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpstream_0_conv_3_3293774*"
_output_shapes
:@@*
dtype02:
8stream_0_conv_3/kernel/Regularizer/Square/ReadVariableOpѕ
)stream_0_conv_3/kernel/Regularizer/SquareSquare@stream_0_conv_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@@2+
)stream_0_conv_3/kernel/Regularizer/Square≠
*stream_0_conv_3/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_0_conv_3/kernel/Regularizer/Const_2а
(stream_0_conv_3/kernel/Regularizer/Sum_1Sum-stream_0_conv_3/kernel/Regularizer/Square:y:03stream_0_conv_3/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2*
(stream_0_conv_3/kernel/Regularizer/Sum_1Э
*stream_0_conv_3/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2,
*stream_0_conv_3/kernel/Regularizer/mul_1/xд
(stream_0_conv_3/kernel/Regularizer/mul_1Mul3stream_0_conv_3/kernel/Regularizer/mul_1/x:output:01stream_0_conv_3/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2*
(stream_0_conv_3/kernel/Regularizer/mul_1Ў
(stream_0_conv_3/kernel/Regularizer/add_1AddV2*stream_0_conv_3/kernel/Regularizer/add:z:0,stream_0_conv_3/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2*
(stream_0_conv_3/kernel/Regularizer/add_1Й
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dense_1/kernel/Regularizer/ConstЃ
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_1_3293793*
_output_shapes

:@@*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOpІ
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:@@2 
dense_1/kernel/Regularizer/AbsЩ
"dense_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_1/kernel/Regularizer/Const_1є
dense_1/kernel/Regularizer/SumSum"dense_1/kernel/Regularizer/Abs:y:0+dense_1/kernel/Regularizer/Const_1:output:0*
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
dense_1/kernel/Regularizer/mulє
dense_1/kernel/Regularizer/addAddV2)dense_1/kernel/Regularizer/Const:output:0"dense_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/addі
0dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_1_3293793*
_output_shapes

:@@*
dtype022
0dense_1/kernel/Regularizer/Square/ReadVariableOp≥
!dense_1/kernel/Regularizer/SquareSquare8dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@@2#
!dense_1/kernel/Regularizer/SquareЩ
"dense_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_1/kernel/Regularizer/Const_2ј
 dense_1/kernel/Regularizer/Sum_1Sum%dense_1/kernel/Regularizer/Square:y:0+dense_1/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2"
 dense_1/kernel/Regularizer/Sum_1Н
"dense_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2$
"dense_1/kernel/Regularizer/mul_1/xƒ
 dense_1/kernel/Regularizer/mul_1Mul+dense_1/kernel/Regularizer/mul_1/x:output:0)dense_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2"
 dense_1/kernel/Regularizer/mul_1Є
 dense_1/kernel/Regularizer/add_1AddV2"dense_1/kernel/Regularizer/add:z:0$dense_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2"
 dense_1/kernel/Regularizer/add_1Ж
IdentityIdentity+dense_activation_1/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@2

IdentityФ
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall.^dense_1/kernel/Regularizer/Abs/ReadVariableOp1^dense_1/kernel/Regularizer/Square/ReadVariableOp(^stream_0_conv_1/StatefulPartitionedCall6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp9^stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOp(^stream_0_conv_2/StatefulPartitionedCall6^stream_0_conv_2/kernel/Regularizer/Abs/ReadVariableOp9^stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp(^stream_0_conv_3/StatefulPartitionedCall6^stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp9^stream_0_conv_3/kernel/Regularizer/Square/ReadVariableOp(^stream_0_drop_1/StatefulPartitionedCall(^stream_0_drop_2/StatefulPartitionedCall(^stream_0_drop_3/StatefulPartitionedCall,^stream_0_input_drop/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:€€€€€€€€€}: : : : : : : : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2^
-dense_1/kernel/Regularizer/Abs/ReadVariableOp-dense_1/kernel/Regularizer/Abs/ReadVariableOp2d
0dense_1/kernel/Regularizer/Square/ReadVariableOp0dense_1/kernel/Regularizer/Square/ReadVariableOp2R
'stream_0_conv_1/StatefulPartitionedCall'stream_0_conv_1/StatefulPartitionedCall2n
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp2t
8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOp8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOp2R
'stream_0_conv_2/StatefulPartitionedCall'stream_0_conv_2/StatefulPartitionedCall2n
5stream_0_conv_2/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_2/kernel/Regularizer/Abs/ReadVariableOp2t
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp2R
'stream_0_conv_3/StatefulPartitionedCall'stream_0_conv_3/StatefulPartitionedCall2n
5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp2t
8stream_0_conv_3/kernel/Regularizer/Square/ReadVariableOp8stream_0_conv_3/kernel/Regularizer/Square/ReadVariableOp2R
'stream_0_drop_1/StatefulPartitionedCall'stream_0_drop_1/StatefulPartitionedCall2R
'stream_0_drop_2/StatefulPartitionedCall'stream_0_drop_2/StatefulPartitionedCall2R
'stream_0_drop_3/StatefulPartitionedCall'stream_0_drop_3/StatefulPartitionedCall2Z
+stream_0_input_drop/StatefulPartitionedCall+stream_0_input_drop/StatefulPartitionedCall:S O
+
_output_shapes
:€€€€€€€€€}
 
_user_specified_nameinputs
к
“
7__inference_batch_normalization_1_layer_call_fn_3297758

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityИҐStatefulPartitionedCall°
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€>@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *[
fVRT
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_32935332
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€>@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€>@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:€€€€€€€€€>@
 
_user_specified_nameinputs
о
k
L__inference_stream_0_drop_2_layer_call_and_return_conditional_losses_3293470

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *Ђ™™?2
dropout/Constw
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:€€€€€€€€€@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape”
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:€€€€€€€€€@*
dtype0*
seedЈ*
seed2є2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А>2
dropout/GreaterEqual/y¬
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:€€€€€€€€€@2
dropout/GreaterEqualГ
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:€€€€€€€€€@2
dropout/Cast~
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:€€€€€€€€€@2
dropout/Mul_1i
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:€€€€€€€€€@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€@:S O
+
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
Ж§
Х"
D__inference_model_1_layer_call_and_return_conditional_losses_3296154

inputs[
Ebasemodel_stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource:@G
9basemodel_stream_0_conv_1_biasadd_readvariableop_resource:@S
Ebasemodel_batch_normalization_assignmovingavg_readvariableop_resource:@U
Gbasemodel_batch_normalization_assignmovingavg_1_readvariableop_resource:@Q
Cbasemodel_batch_normalization_batchnorm_mul_readvariableop_resource:@M
?basemodel_batch_normalization_batchnorm_readvariableop_resource:@[
Ebasemodel_stream_0_conv_2_conv1d_expanddims_1_readvariableop_resource:@@G
9basemodel_stream_0_conv_2_biasadd_readvariableop_resource:@U
Gbasemodel_batch_normalization_1_assignmovingavg_readvariableop_resource:@W
Ibasemodel_batch_normalization_1_assignmovingavg_1_readvariableop_resource:@S
Ebasemodel_batch_normalization_1_batchnorm_mul_readvariableop_resource:@O
Abasemodel_batch_normalization_1_batchnorm_readvariableop_resource:@[
Ebasemodel_stream_0_conv_3_conv1d_expanddims_1_readvariableop_resource:@@G
9basemodel_stream_0_conv_3_biasadd_readvariableop_resource:@U
Gbasemodel_batch_normalization_2_assignmovingavg_readvariableop_resource:@W
Ibasemodel_batch_normalization_2_assignmovingavg_1_readvariableop_resource:@S
Ebasemodel_batch_normalization_2_batchnorm_mul_readvariableop_resource:@O
Abasemodel_batch_normalization_2_batchnorm_readvariableop_resource:@B
0basemodel_dense_1_matmul_readvariableop_resource:@@?
1basemodel_dense_1_biasadd_readvariableop_resource:@U
Gbasemodel_batch_normalization_3_assignmovingavg_readvariableop_resource:@W
Ibasemodel_batch_normalization_3_assignmovingavg_1_readvariableop_resource:@S
Ebasemodel_batch_normalization_3_batchnorm_mul_readvariableop_resource:@O
Abasemodel_batch_normalization_3_batchnorm_readvariableop_resource:@
identityИҐ-basemodel/batch_normalization/AssignMovingAvgҐ<basemodel/batch_normalization/AssignMovingAvg/ReadVariableOpҐ/basemodel/batch_normalization/AssignMovingAvg_1Ґ>basemodel/batch_normalization/AssignMovingAvg_1/ReadVariableOpҐ6basemodel/batch_normalization/batchnorm/ReadVariableOpҐ:basemodel/batch_normalization/batchnorm/mul/ReadVariableOpҐ/basemodel/batch_normalization_1/AssignMovingAvgҐ>basemodel/batch_normalization_1/AssignMovingAvg/ReadVariableOpҐ1basemodel/batch_normalization_1/AssignMovingAvg_1Ґ@basemodel/batch_normalization_1/AssignMovingAvg_1/ReadVariableOpҐ8basemodel/batch_normalization_1/batchnorm/ReadVariableOpҐ<basemodel/batch_normalization_1/batchnorm/mul/ReadVariableOpҐ/basemodel/batch_normalization_2/AssignMovingAvgҐ>basemodel/batch_normalization_2/AssignMovingAvg/ReadVariableOpҐ1basemodel/batch_normalization_2/AssignMovingAvg_1Ґ@basemodel/batch_normalization_2/AssignMovingAvg_1/ReadVariableOpҐ8basemodel/batch_normalization_2/batchnorm/ReadVariableOpҐ<basemodel/batch_normalization_2/batchnorm/mul/ReadVariableOpҐ/basemodel/batch_normalization_3/AssignMovingAvgҐ>basemodel/batch_normalization_3/AssignMovingAvg/ReadVariableOpҐ1basemodel/batch_normalization_3/AssignMovingAvg_1Ґ@basemodel/batch_normalization_3/AssignMovingAvg_1/ReadVariableOpҐ8basemodel/batch_normalization_3/batchnorm/ReadVariableOpҐ<basemodel/batch_normalization_3/batchnorm/mul/ReadVariableOpҐ(basemodel/dense_1/BiasAdd/ReadVariableOpҐ'basemodel/dense_1/MatMul/ReadVariableOpҐ0basemodel/stream_0_conv_1/BiasAdd/ReadVariableOpҐ<basemodel/stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOpҐ0basemodel/stream_0_conv_2/BiasAdd/ReadVariableOpҐ<basemodel/stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOpҐ0basemodel/stream_0_conv_3/BiasAdd/ReadVariableOpҐ<basemodel/stream_0_conv_3/conv1d/ExpandDims_1/ReadVariableOpҐ-dense_1/kernel/Regularizer/Abs/ReadVariableOpҐ0dense_1/kernel/Regularizer/Square/ReadVariableOpҐ5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpҐ8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOpҐ5stream_0_conv_2/kernel/Regularizer/Abs/ReadVariableOpҐ8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOpҐ5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOpҐ8stream_0_conv_3/kernel/Regularizer/Square/ReadVariableOpЯ
+basemodel/stream_0_input_drop/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †?2-
+basemodel/stream_0_input_drop/dropout/Const—
)basemodel/stream_0_input_drop/dropout/MulMulinputs4basemodel/stream_0_input_drop/dropout/Const:output:0*
T0*+
_output_shapes
:€€€€€€€€€}2+
)basemodel/stream_0_input_drop/dropout/MulР
+basemodel/stream_0_input_drop/dropout/ShapeShapeinputs*
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
!basemodel/stream_0_conv_1/BiasAddЌ
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
-basemodel/batch_normalization/batchnorm/add_1І
basemodel/activation/TanhTanh1basemodel/batch_normalization/batchnorm/add_1:z:0*
T0*+
_output_shapes
:€€€€€€€€€}@2
basemodel/activation/TanhЬ
+basemodel/stream_0_maxpool_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2-
+basemodel/stream_0_maxpool_1/ExpandDims/dimп
'basemodel/stream_0_maxpool_1/ExpandDims
ExpandDimsbasemodel/activation/Tanh:y:04basemodel/stream_0_maxpool_1/ExpandDims/dim:output:0*
T0*/
_output_shapes
:€€€€€€€€€}@2)
'basemodel/stream_0_maxpool_1/ExpandDimsц
$basemodel/stream_0_maxpool_1/MaxPoolMaxPool0basemodel/stream_0_maxpool_1/ExpandDims:output:0*/
_output_shapes
:€€€€€€€€€>@*
ksize
*
paddingVALID*
strides
2&
$basemodel/stream_0_maxpool_1/MaxPool”
$basemodel/stream_0_maxpool_1/SqueezeSqueeze-basemodel/stream_0_maxpool_1/MaxPool:output:0*
T0*+
_output_shapes
:€€€€€€€€€>@*
squeeze_dims
2&
$basemodel/stream_0_maxpool_1/SqueezeЧ
'basemodel/stream_0_drop_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *Ђ™™?2)
'basemodel/stream_0_drop_1/dropout/Constм
%basemodel/stream_0_drop_1/dropout/MulMul-basemodel/stream_0_maxpool_1/Squeeze:output:00basemodel/stream_0_drop_1/dropout/Const:output:0*
T0*+
_output_shapes
:€€€€€€€€€>@2'
%basemodel/stream_0_drop_1/dropout/Mulѓ
'basemodel/stream_0_drop_1/dropout/ShapeShape-basemodel/stream_0_maxpool_1/Squeeze:output:0*
T0*
_output_shapes
:2)
'basemodel/stream_0_drop_1/dropout/Shape°
>basemodel/stream_0_drop_1/dropout/random_uniform/RandomUniformRandomUniform0basemodel/stream_0_drop_1/dropout/Shape:output:0*
T0*+
_output_shapes
:€€€€€€€€€>@*
dtype0*
seedЈ*
seed2Є2@
>basemodel/stream_0_drop_1/dropout/random_uniform/RandomUniform©
0basemodel/stream_0_drop_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А>22
0basemodel/stream_0_drop_1/dropout/GreaterEqual/y™
.basemodel/stream_0_drop_1/dropout/GreaterEqualGreaterEqualGbasemodel/stream_0_drop_1/dropout/random_uniform/RandomUniform:output:09basemodel/stream_0_drop_1/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:€€€€€€€€€>@20
.basemodel/stream_0_drop_1/dropout/GreaterEqual—
&basemodel/stream_0_drop_1/dropout/CastCast2basemodel/stream_0_drop_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:€€€€€€€€€>@2(
&basemodel/stream_0_drop_1/dropout/Castж
'basemodel/stream_0_drop_1/dropout/Mul_1Mul)basemodel/stream_0_drop_1/dropout/Mul:z:0*basemodel/stream_0_drop_1/dropout/Cast:y:0*
T0*+
_output_shapes
:€€€€€€€€€>@2)
'basemodel/stream_0_drop_1/dropout/Mul_1≠
/basemodel/stream_0_conv_2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€21
/basemodel/stream_0_conv_2/conv1d/ExpandDims/dimЙ
+basemodel/stream_0_conv_2/conv1d/ExpandDims
ExpandDims+basemodel/stream_0_drop_1/dropout/Mul_1:z:08basemodel/stream_0_conv_2/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:€€€€€€€€€>@2-
+basemodel/stream_0_conv_2/conv1d/ExpandDimsЖ
<basemodel/stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpEbasemodel_stream_0_conv_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype02>
<basemodel/stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOp®
1basemodel/stream_0_conv_2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 23
1basemodel/stream_0_conv_2/conv1d/ExpandDims_1/dimЯ
-basemodel/stream_0_conv_2/conv1d/ExpandDims_1
ExpandDimsDbasemodel/stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOp:value:0:basemodel/stream_0_conv_2/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@2/
-basemodel/stream_0_conv_2/conv1d/ExpandDims_1Ю
 basemodel/stream_0_conv_2/conv1dConv2D4basemodel/stream_0_conv_2/conv1d/ExpandDims:output:06basemodel/stream_0_conv_2/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:€€€€€€€€€>@*
paddingSAME*
strides
2"
 basemodel/stream_0_conv_2/conv1dа
(basemodel/stream_0_conv_2/conv1d/SqueezeSqueeze)basemodel/stream_0_conv_2/conv1d:output:0*
T0*+
_output_shapes
:€€€€€€€€€>@*
squeeze_dims

э€€€€€€€€2*
(basemodel/stream_0_conv_2/conv1d/SqueezeЏ
0basemodel/stream_0_conv_2/BiasAdd/ReadVariableOpReadVariableOp9basemodel_stream_0_conv_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype022
0basemodel/stream_0_conv_2/BiasAdd/ReadVariableOpф
!basemodel/stream_0_conv_2/BiasAddBiasAdd1basemodel/stream_0_conv_2/conv1d/Squeeze:output:08basemodel/stream_0_conv_2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€>@2#
!basemodel/stream_0_conv_2/BiasAdd—
>basemodel/batch_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2@
>basemodel/batch_normalization_1/moments/mean/reduction_indicesЧ
,basemodel/batch_normalization_1/moments/meanMean*basemodel/stream_0_conv_2/BiasAdd:output:0Gbasemodel/batch_normalization_1/moments/mean/reduction_indices:output:0*
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
9basemodel/batch_normalization_1/moments/SquaredDifferenceSquaredDifference*basemodel/stream_0_conv_2/BiasAdd:output:0=basemodel/batch_normalization_1/moments/StopGradient:output:0*
T0*+
_output_shapes
:€€€€€€€€€>@2;
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
/basemodel/batch_normalization_1/batchnorm/mul_1Mul*basemodel/stream_0_conv_2/BiasAdd:output:01basemodel/batch_normalization_1/batchnorm/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€>@21
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
:€€€€€€€€€>@21
/basemodel/batch_normalization_1/batchnorm/add_1≠
basemodel/activation_1/TanhTanh3basemodel/batch_normalization_1/batchnorm/add_1:z:0*
T0*+
_output_shapes
:€€€€€€€€€>@2
basemodel/activation_1/TanhЬ
+basemodel/stream_0_maxpool_2/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2-
+basemodel/stream_0_maxpool_2/ExpandDims/dimс
'basemodel/stream_0_maxpool_2/ExpandDims
ExpandDimsbasemodel/activation_1/Tanh:y:04basemodel/stream_0_maxpool_2/ExpandDims/dim:output:0*
T0*/
_output_shapes
:€€€€€€€€€>@2)
'basemodel/stream_0_maxpool_2/ExpandDimsц
$basemodel/stream_0_maxpool_2/MaxPoolMaxPool0basemodel/stream_0_maxpool_2/ExpandDims:output:0*/
_output_shapes
:€€€€€€€€€@*
ksize
*
paddingVALID*
strides
2&
$basemodel/stream_0_maxpool_2/MaxPool”
$basemodel/stream_0_maxpool_2/SqueezeSqueeze-basemodel/stream_0_maxpool_2/MaxPool:output:0*
T0*+
_output_shapes
:€€€€€€€€€@*
squeeze_dims
2&
$basemodel/stream_0_maxpool_2/SqueezeЧ
'basemodel/stream_0_drop_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *Ђ™™?2)
'basemodel/stream_0_drop_2/dropout/Constм
%basemodel/stream_0_drop_2/dropout/MulMul-basemodel/stream_0_maxpool_2/Squeeze:output:00basemodel/stream_0_drop_2/dropout/Const:output:0*
T0*+
_output_shapes
:€€€€€€€€€@2'
%basemodel/stream_0_drop_2/dropout/Mulѓ
'basemodel/stream_0_drop_2/dropout/ShapeShape-basemodel/stream_0_maxpool_2/Squeeze:output:0*
T0*
_output_shapes
:2)
'basemodel/stream_0_drop_2/dropout/Shape°
>basemodel/stream_0_drop_2/dropout/random_uniform/RandomUniformRandomUniform0basemodel/stream_0_drop_2/dropout/Shape:output:0*
T0*+
_output_shapes
:€€€€€€€€€@*
dtype0*
seedЈ*
seed2є2@
>basemodel/stream_0_drop_2/dropout/random_uniform/RandomUniform©
0basemodel/stream_0_drop_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А>22
0basemodel/stream_0_drop_2/dropout/GreaterEqual/y™
.basemodel/stream_0_drop_2/dropout/GreaterEqualGreaterEqualGbasemodel/stream_0_drop_2/dropout/random_uniform/RandomUniform:output:09basemodel/stream_0_drop_2/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:€€€€€€€€€@20
.basemodel/stream_0_drop_2/dropout/GreaterEqual—
&basemodel/stream_0_drop_2/dropout/CastCast2basemodel/stream_0_drop_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:€€€€€€€€€@2(
&basemodel/stream_0_drop_2/dropout/Castж
'basemodel/stream_0_drop_2/dropout/Mul_1Mul)basemodel/stream_0_drop_2/dropout/Mul:z:0*basemodel/stream_0_drop_2/dropout/Cast:y:0*
T0*+
_output_shapes
:€€€€€€€€€@2)
'basemodel/stream_0_drop_2/dropout/Mul_1≠
/basemodel/stream_0_conv_3/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€21
/basemodel/stream_0_conv_3/conv1d/ExpandDims/dimЙ
+basemodel/stream_0_conv_3/conv1d/ExpandDims
ExpandDims+basemodel/stream_0_drop_2/dropout/Mul_1:z:08basemodel/stream_0_conv_3/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:€€€€€€€€€@2-
+basemodel/stream_0_conv_3/conv1d/ExpandDimsЖ
<basemodel/stream_0_conv_3/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpEbasemodel_stream_0_conv_3_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype02>
<basemodel/stream_0_conv_3/conv1d/ExpandDims_1/ReadVariableOp®
1basemodel/stream_0_conv_3/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 23
1basemodel/stream_0_conv_3/conv1d/ExpandDims_1/dimЯ
-basemodel/stream_0_conv_3/conv1d/ExpandDims_1
ExpandDimsDbasemodel/stream_0_conv_3/conv1d/ExpandDims_1/ReadVariableOp:value:0:basemodel/stream_0_conv_3/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@2/
-basemodel/stream_0_conv_3/conv1d/ExpandDims_1Ю
 basemodel/stream_0_conv_3/conv1dConv2D4basemodel/stream_0_conv_3/conv1d/ExpandDims:output:06basemodel/stream_0_conv_3/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:€€€€€€€€€@*
paddingSAME*
strides
2"
 basemodel/stream_0_conv_3/conv1dа
(basemodel/stream_0_conv_3/conv1d/SqueezeSqueeze)basemodel/stream_0_conv_3/conv1d:output:0*
T0*+
_output_shapes
:€€€€€€€€€@*
squeeze_dims

э€€€€€€€€2*
(basemodel/stream_0_conv_3/conv1d/SqueezeЏ
0basemodel/stream_0_conv_3/BiasAdd/ReadVariableOpReadVariableOp9basemodel_stream_0_conv_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype022
0basemodel/stream_0_conv_3/BiasAdd/ReadVariableOpф
!basemodel/stream_0_conv_3/BiasAddBiasAdd1basemodel/stream_0_conv_3/conv1d/Squeeze:output:08basemodel/stream_0_conv_3/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€@2#
!basemodel/stream_0_conv_3/BiasAdd—
>basemodel/batch_normalization_2/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2@
>basemodel/batch_normalization_2/moments/mean/reduction_indicesЧ
,basemodel/batch_normalization_2/moments/meanMean*basemodel/stream_0_conv_3/BiasAdd:output:0Gbasemodel/batch_normalization_2/moments/mean/reduction_indices:output:0*
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
9basemodel/batch_normalization_2/moments/SquaredDifferenceSquaredDifference*basemodel/stream_0_conv_3/BiasAdd:output:0=basemodel/batch_normalization_2/moments/StopGradient:output:0*
T0*+
_output_shapes
:€€€€€€€€€@2;
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
/basemodel/batch_normalization_2/batchnorm/mul_1Mul*basemodel/stream_0_conv_3/BiasAdd:output:01basemodel/batch_normalization_2/batchnorm/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€@21
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
:€€€€€€€€€@21
/basemodel/batch_normalization_2/batchnorm/add_1≠
basemodel/activation_2/TanhTanh3basemodel/batch_normalization_2/batchnorm/add_1:z:0*
T0*+
_output_shapes
:€€€€€€€€€@2
basemodel/activation_2/TanhЬ
+basemodel/stream_0_maxpool_3/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2-
+basemodel/stream_0_maxpool_3/ExpandDims/dimс
'basemodel/stream_0_maxpool_3/ExpandDims
ExpandDimsbasemodel/activation_2/Tanh:y:04basemodel/stream_0_maxpool_3/ExpandDims/dim:output:0*
T0*/
_output_shapes
:€€€€€€€€€@2)
'basemodel/stream_0_maxpool_3/ExpandDimsц
$basemodel/stream_0_maxpool_3/MaxPoolMaxPool0basemodel/stream_0_maxpool_3/ExpandDims:output:0*/
_output_shapes
:€€€€€€€€€@*
ksize
*
paddingVALID*
strides
2&
$basemodel/stream_0_maxpool_3/MaxPool”
$basemodel/stream_0_maxpool_3/SqueezeSqueeze-basemodel/stream_0_maxpool_3/MaxPool:output:0*
T0*+
_output_shapes
:€€€€€€€€€@*
squeeze_dims
2&
$basemodel/stream_0_maxpool_3/SqueezeЧ
'basemodel/stream_0_drop_3/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *Ђ™™?2)
'basemodel/stream_0_drop_3/dropout/Constм
%basemodel/stream_0_drop_3/dropout/MulMul-basemodel/stream_0_maxpool_3/Squeeze:output:00basemodel/stream_0_drop_3/dropout/Const:output:0*
T0*+
_output_shapes
:€€€€€€€€€@2'
%basemodel/stream_0_drop_3/dropout/Mulѓ
'basemodel/stream_0_drop_3/dropout/ShapeShape-basemodel/stream_0_maxpool_3/Squeeze:output:0*
T0*
_output_shapes
:2)
'basemodel/stream_0_drop_3/dropout/Shape°
>basemodel/stream_0_drop_3/dropout/random_uniform/RandomUniformRandomUniform0basemodel/stream_0_drop_3/dropout/Shape:output:0*
T0*+
_output_shapes
:€€€€€€€€€@*
dtype0*
seedЈ*
seed2Ї2@
>basemodel/stream_0_drop_3/dropout/random_uniform/RandomUniform©
0basemodel/stream_0_drop_3/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А>22
0basemodel/stream_0_drop_3/dropout/GreaterEqual/y™
.basemodel/stream_0_drop_3/dropout/GreaterEqualGreaterEqualGbasemodel/stream_0_drop_3/dropout/random_uniform/RandomUniform:output:09basemodel/stream_0_drop_3/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:€€€€€€€€€@20
.basemodel/stream_0_drop_3/dropout/GreaterEqual—
&basemodel/stream_0_drop_3/dropout/CastCast2basemodel/stream_0_drop_3/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:€€€€€€€€€@2(
&basemodel/stream_0_drop_3/dropout/Castж
'basemodel/stream_0_drop_3/dropout/Mul_1Mul)basemodel/stream_0_drop_3/dropout/Mul:z:0*basemodel/stream_0_drop_3/dropout/Cast:y:0*
T0*+
_output_shapes
:€€€€€€€€€@2)
'basemodel/stream_0_drop_3/dropout/Mul_1Є
9basemodel/global_average_pooling1d/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2;
9basemodel/global_average_pooling1d/Mean/reduction_indicesэ
'basemodel/global_average_pooling1d/MeanMean+basemodel/stream_0_drop_3/dropout/Mul_1:z:0Bbasemodel/global_average_pooling1d/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2)
'basemodel/global_average_pooling1d/Mean√
'basemodel/dense_1/MatMul/ReadVariableOpReadVariableOp0basemodel_dense_1_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02)
'basemodel/dense_1/MatMul/ReadVariableOp”
basemodel/dense_1/MatMulMatMul0basemodel/global_average_pooling1d/Mean:output:0/basemodel/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2
basemodel/dense_1/MatMul¬
(basemodel/dense_1/BiasAdd/ReadVariableOpReadVariableOp1basemodel_dense_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02*
(basemodel/dense_1/BiasAdd/ReadVariableOp…
basemodel/dense_1/BiasAddBiasAdd"basemodel/dense_1/MatMul:product:00basemodel/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2
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

:@*
	keep_dims(2.
,basemodel/batch_normalization_3/moments/mean№
4basemodel/batch_normalization_3/moments/StopGradientStopGradient5basemodel/batch_normalization_3/moments/mean:output:0*
T0*
_output_shapes

:@26
4basemodel/batch_normalization_3/moments/StopGradient†
9basemodel/batch_normalization_3/moments/SquaredDifferenceSquaredDifference"basemodel/dense_1/BiasAdd:output:0=basemodel/batch_normalization_3/moments/StopGradient:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2;
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

:@*
	keep_dims(22
0basemodel/batch_normalization_3/moments/varianceа
/basemodel/batch_normalization_3/moments/SqueezeSqueeze5basemodel/batch_normalization_3/moments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 21
/basemodel/batch_normalization_3/moments/Squeezeи
1basemodel/batch_normalization_3/moments/Squeeze_1Squeeze9basemodel/batch_normalization_3/moments/variance:output:0*
T0*
_output_shapes
:@*
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
:@*
dtype02@
>basemodel/batch_normalization_3/AssignMovingAvg/ReadVariableOpШ
3basemodel/batch_normalization_3/AssignMovingAvg/subSubFbasemodel/batch_normalization_3/AssignMovingAvg/ReadVariableOp:value:08basemodel/batch_normalization_3/moments/Squeeze:output:0*
T0*
_output_shapes
:@25
3basemodel/batch_normalization_3/AssignMovingAvg/subП
3basemodel/batch_normalization_3/AssignMovingAvg/mulMul7basemodel/batch_normalization_3/AssignMovingAvg/sub:z:0>basemodel/batch_normalization_3/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:@25
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
:@*
dtype02B
@basemodel/batch_normalization_3/AssignMovingAvg_1/ReadVariableOp†
5basemodel/batch_normalization_3/AssignMovingAvg_1/subSubHbasemodel/batch_normalization_3/AssignMovingAvg_1/ReadVariableOp:value:0:basemodel/batch_normalization_3/moments/Squeeze_1:output:0*
T0*
_output_shapes
:@27
5basemodel/batch_normalization_3/AssignMovingAvg_1/subЧ
5basemodel/batch_normalization_3/AssignMovingAvg_1/mulMul9basemodel/batch_normalization_3/AssignMovingAvg_1/sub:z:0@basemodel/batch_normalization_3/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:@27
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
:@2/
-basemodel/batch_normalization_3/batchnorm/add√
/basemodel/batch_normalization_3/batchnorm/RsqrtRsqrt1basemodel/batch_normalization_3/batchnorm/add:z:0*
T0*
_output_shapes
:@21
/basemodel/batch_normalization_3/batchnorm/Rsqrtю
<basemodel/batch_normalization_3/batchnorm/mul/ReadVariableOpReadVariableOpEbasemodel_batch_normalization_3_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02>
<basemodel/batch_normalization_3/batchnorm/mul/ReadVariableOpЕ
-basemodel/batch_normalization_3/batchnorm/mulMul3basemodel/batch_normalization_3/batchnorm/Rsqrt:y:0Dbasemodel/batch_normalization_3/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2/
-basemodel/batch_normalization_3/batchnorm/mulт
/basemodel/batch_normalization_3/batchnorm/mul_1Mul"basemodel/dense_1/BiasAdd:output:01basemodel/batch_normalization_3/batchnorm/mul:z:0*
T0*'
_output_shapes
:€€€€€€€€€@21
/basemodel/batch_normalization_3/batchnorm/mul_1ы
/basemodel/batch_normalization_3/batchnorm/mul_2Mul8basemodel/batch_normalization_3/moments/Squeeze:output:01basemodel/batch_normalization_3/batchnorm/mul:z:0*
T0*
_output_shapes
:@21
/basemodel/batch_normalization_3/batchnorm/mul_2т
8basemodel/batch_normalization_3/batchnorm/ReadVariableOpReadVariableOpAbasemodel_batch_normalization_3_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02:
8basemodel/batch_normalization_3/batchnorm/ReadVariableOpБ
-basemodel/batch_normalization_3/batchnorm/subSub@basemodel/batch_normalization_3/batchnorm/ReadVariableOp:value:03basemodel/batch_normalization_3/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2/
-basemodel/batch_normalization_3/batchnorm/subЕ
/basemodel/batch_normalization_3/batchnorm/add_1AddV23basemodel/batch_normalization_3/batchnorm/mul_1:z:01basemodel/batch_normalization_3/batchnorm/sub:z:0*
T0*'
_output_shapes
:€€€€€€€€€@21
/basemodel/batch_normalization_3/batchnorm/add_1Щ
(stream_0_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(stream_0_conv_1/kernel/Regularizer/Constш
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpEbasemodel_stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype027
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp√
&stream_0_conv_1/kernel/Regularizer/AbsAbs=stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_0_conv_1/kernel/Regularizer/Abs≠
*stream_0_conv_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_0_conv_1/kernel/Regularizer/Const_1ў
&stream_0_conv_1/kernel/Regularizer/SumSum*stream_0_conv_1/kernel/Regularizer/Abs:y:03stream_0_conv_1/kernel/Regularizer/Const_1:output:0*
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
&stream_0_conv_1/kernel/Regularizer/mulў
&stream_0_conv_1/kernel/Regularizer/addAddV21stream_0_conv_1/kernel/Regularizer/Const:output:0*stream_0_conv_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/addю
8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpEbasemodel_stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02:
8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOpѕ
)stream_0_conv_1/kernel/Regularizer/SquareSquare@stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2+
)stream_0_conv_1/kernel/Regularizer/Square≠
*stream_0_conv_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_0_conv_1/kernel/Regularizer/Const_2а
(stream_0_conv_1/kernel/Regularizer/Sum_1Sum-stream_0_conv_1/kernel/Regularizer/Square:y:03stream_0_conv_1/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2*
(stream_0_conv_1/kernel/Regularizer/Sum_1Э
*stream_0_conv_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2,
*stream_0_conv_1/kernel/Regularizer/mul_1/xд
(stream_0_conv_1/kernel/Regularizer/mul_1Mul3stream_0_conv_1/kernel/Regularizer/mul_1/x:output:01stream_0_conv_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2*
(stream_0_conv_1/kernel/Regularizer/mul_1Ў
(stream_0_conv_1/kernel/Regularizer/add_1AddV2*stream_0_conv_1/kernel/Regularizer/add:z:0,stream_0_conv_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2*
(stream_0_conv_1/kernel/Regularizer/add_1Щ
(stream_0_conv_2/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(stream_0_conv_2/kernel/Regularizer/Constш
5stream_0_conv_2/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpEbasemodel_stream_0_conv_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype027
5stream_0_conv_2/kernel/Regularizer/Abs/ReadVariableOp√
&stream_0_conv_2/kernel/Regularizer/AbsAbs=stream_0_conv_2/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@@2(
&stream_0_conv_2/kernel/Regularizer/Abs≠
*stream_0_conv_2/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_0_conv_2/kernel/Regularizer/Const_1ў
&stream_0_conv_2/kernel/Regularizer/SumSum*stream_0_conv_2/kernel/Regularizer/Abs:y:03stream_0_conv_2/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_2/kernel/Regularizer/SumЩ
(stream_0_conv_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2*
(stream_0_conv_2/kernel/Regularizer/mul/x№
&stream_0_conv_2/kernel/Regularizer/mulMul1stream_0_conv_2/kernel/Regularizer/mul/x:output:0/stream_0_conv_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_2/kernel/Regularizer/mulў
&stream_0_conv_2/kernel/Regularizer/addAddV21stream_0_conv_2/kernel/Regularizer/Const:output:0*stream_0_conv_2/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&stream_0_conv_2/kernel/Regularizer/addю
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpEbasemodel_stream_0_conv_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype02:
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOpѕ
)stream_0_conv_2/kernel/Regularizer/SquareSquare@stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@@2+
)stream_0_conv_2/kernel/Regularizer/Square≠
*stream_0_conv_2/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_0_conv_2/kernel/Regularizer/Const_2а
(stream_0_conv_2/kernel/Regularizer/Sum_1Sum-stream_0_conv_2/kernel/Regularizer/Square:y:03stream_0_conv_2/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2*
(stream_0_conv_2/kernel/Regularizer/Sum_1Э
*stream_0_conv_2/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2,
*stream_0_conv_2/kernel/Regularizer/mul_1/xд
(stream_0_conv_2/kernel/Regularizer/mul_1Mul3stream_0_conv_2/kernel/Regularizer/mul_1/x:output:01stream_0_conv_2/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2*
(stream_0_conv_2/kernel/Regularizer/mul_1Ў
(stream_0_conv_2/kernel/Regularizer/add_1AddV2*stream_0_conv_2/kernel/Regularizer/add:z:0,stream_0_conv_2/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2*
(stream_0_conv_2/kernel/Regularizer/add_1Щ
(stream_0_conv_3/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(stream_0_conv_3/kernel/Regularizer/Constш
5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpEbasemodel_stream_0_conv_3_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype027
5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp√
&stream_0_conv_3/kernel/Regularizer/AbsAbs=stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@@2(
&stream_0_conv_3/kernel/Regularizer/Abs≠
*stream_0_conv_3/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_0_conv_3/kernel/Regularizer/Const_1ў
&stream_0_conv_3/kernel/Regularizer/SumSum*stream_0_conv_3/kernel/Regularizer/Abs:y:03stream_0_conv_3/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_3/kernel/Regularizer/SumЩ
(stream_0_conv_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2*
(stream_0_conv_3/kernel/Regularizer/mul/x№
&stream_0_conv_3/kernel/Regularizer/mulMul1stream_0_conv_3/kernel/Regularizer/mul/x:output:0/stream_0_conv_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_3/kernel/Regularizer/mulў
&stream_0_conv_3/kernel/Regularizer/addAddV21stream_0_conv_3/kernel/Regularizer/Const:output:0*stream_0_conv_3/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&stream_0_conv_3/kernel/Regularizer/addю
8stream_0_conv_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpEbasemodel_stream_0_conv_3_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype02:
8stream_0_conv_3/kernel/Regularizer/Square/ReadVariableOpѕ
)stream_0_conv_3/kernel/Regularizer/SquareSquare@stream_0_conv_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@@2+
)stream_0_conv_3/kernel/Regularizer/Square≠
*stream_0_conv_3/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_0_conv_3/kernel/Regularizer/Const_2а
(stream_0_conv_3/kernel/Regularizer/Sum_1Sum-stream_0_conv_3/kernel/Regularizer/Square:y:03stream_0_conv_3/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2*
(stream_0_conv_3/kernel/Regularizer/Sum_1Э
*stream_0_conv_3/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2,
*stream_0_conv_3/kernel/Regularizer/mul_1/xд
(stream_0_conv_3/kernel/Regularizer/mul_1Mul3stream_0_conv_3/kernel/Regularizer/mul_1/x:output:01stream_0_conv_3/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2*
(stream_0_conv_3/kernel/Regularizer/mul_1Ў
(stream_0_conv_3/kernel/Regularizer/add_1AddV2*stream_0_conv_3/kernel/Regularizer/add:z:0,stream_0_conv_3/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2*
(stream_0_conv_3/kernel/Regularizer/add_1Й
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dense_1/kernel/Regularizer/Constѕ
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp0basemodel_dense_1_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOpІ
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:@@2 
dense_1/kernel/Regularizer/AbsЩ
"dense_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_1/kernel/Regularizer/Const_1є
dense_1/kernel/Regularizer/SumSum"dense_1/kernel/Regularizer/Abs:y:0+dense_1/kernel/Regularizer/Const_1:output:0*
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
dense_1/kernel/Regularizer/mulє
dense_1/kernel/Regularizer/addAddV2)dense_1/kernel/Regularizer/Const:output:0"dense_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/add’
0dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp0basemodel_dense_1_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype022
0dense_1/kernel/Regularizer/Square/ReadVariableOp≥
!dense_1/kernel/Regularizer/SquareSquare8dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@@2#
!dense_1/kernel/Regularizer/SquareЩ
"dense_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_1/kernel/Regularizer/Const_2ј
 dense_1/kernel/Regularizer/Sum_1Sum%dense_1/kernel/Regularizer/Square:y:0+dense_1/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2"
 dense_1/kernel/Regularizer/Sum_1Н
"dense_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2$
"dense_1/kernel/Regularizer/mul_1/xƒ
 dense_1/kernel/Regularizer/mul_1Mul+dense_1/kernel/Regularizer/mul_1/x:output:0)dense_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2"
 dense_1/kernel/Regularizer/mul_1Є
 dense_1/kernel/Regularizer/add_1AddV2"dense_1/kernel/Regularizer/add:z:0$dense_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2"
 dense_1/kernel/Regularizer/add_1О
IdentityIdentity3basemodel/batch_normalization_3/batchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@2

Identityє
NoOpNoOp.^basemodel/batch_normalization/AssignMovingAvg=^basemodel/batch_normalization/AssignMovingAvg/ReadVariableOp0^basemodel/batch_normalization/AssignMovingAvg_1?^basemodel/batch_normalization/AssignMovingAvg_1/ReadVariableOp7^basemodel/batch_normalization/batchnorm/ReadVariableOp;^basemodel/batch_normalization/batchnorm/mul/ReadVariableOp0^basemodel/batch_normalization_1/AssignMovingAvg?^basemodel/batch_normalization_1/AssignMovingAvg/ReadVariableOp2^basemodel/batch_normalization_1/AssignMovingAvg_1A^basemodel/batch_normalization_1/AssignMovingAvg_1/ReadVariableOp9^basemodel/batch_normalization_1/batchnorm/ReadVariableOp=^basemodel/batch_normalization_1/batchnorm/mul/ReadVariableOp0^basemodel/batch_normalization_2/AssignMovingAvg?^basemodel/batch_normalization_2/AssignMovingAvg/ReadVariableOp2^basemodel/batch_normalization_2/AssignMovingAvg_1A^basemodel/batch_normalization_2/AssignMovingAvg_1/ReadVariableOp9^basemodel/batch_normalization_2/batchnorm/ReadVariableOp=^basemodel/batch_normalization_2/batchnorm/mul/ReadVariableOp0^basemodel/batch_normalization_3/AssignMovingAvg?^basemodel/batch_normalization_3/AssignMovingAvg/ReadVariableOp2^basemodel/batch_normalization_3/AssignMovingAvg_1A^basemodel/batch_normalization_3/AssignMovingAvg_1/ReadVariableOp9^basemodel/batch_normalization_3/batchnorm/ReadVariableOp=^basemodel/batch_normalization_3/batchnorm/mul/ReadVariableOp)^basemodel/dense_1/BiasAdd/ReadVariableOp(^basemodel/dense_1/MatMul/ReadVariableOp1^basemodel/stream_0_conv_1/BiasAdd/ReadVariableOp=^basemodel/stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp1^basemodel/stream_0_conv_2/BiasAdd/ReadVariableOp=^basemodel/stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOp1^basemodel/stream_0_conv_3/BiasAdd/ReadVariableOp=^basemodel/stream_0_conv_3/conv1d/ExpandDims_1/ReadVariableOp.^dense_1/kernel/Regularizer/Abs/ReadVariableOp1^dense_1/kernel/Regularizer/Square/ReadVariableOp6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp9^stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOp6^stream_0_conv_2/kernel/Regularizer/Abs/ReadVariableOp9^stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp6^stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp9^stream_0_conv_3/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:€€€€€€€€€}: : : : : : : : : : : : : : : : : : : : : : : : 2^
-basemodel/batch_normalization/AssignMovingAvg-basemodel/batch_normalization/AssignMovingAvg2|
<basemodel/batch_normalization/AssignMovingAvg/ReadVariableOp<basemodel/batch_normalization/AssignMovingAvg/ReadVariableOp2b
/basemodel/batch_normalization/AssignMovingAvg_1/basemodel/batch_normalization/AssignMovingAvg_12А
>basemodel/batch_normalization/AssignMovingAvg_1/ReadVariableOp>basemodel/batch_normalization/AssignMovingAvg_1/ReadVariableOp2p
6basemodel/batch_normalization/batchnorm/ReadVariableOp6basemodel/batch_normalization/batchnorm/ReadVariableOp2x
:basemodel/batch_normalization/batchnorm/mul/ReadVariableOp:basemodel/batch_normalization/batchnorm/mul/ReadVariableOp2b
/basemodel/batch_normalization_1/AssignMovingAvg/basemodel/batch_normalization_1/AssignMovingAvg2А
>basemodel/batch_normalization_1/AssignMovingAvg/ReadVariableOp>basemodel/batch_normalization_1/AssignMovingAvg/ReadVariableOp2f
1basemodel/batch_normalization_1/AssignMovingAvg_11basemodel/batch_normalization_1/AssignMovingAvg_12Д
@basemodel/batch_normalization_1/AssignMovingAvg_1/ReadVariableOp@basemodel/batch_normalization_1/AssignMovingAvg_1/ReadVariableOp2t
8basemodel/batch_normalization_1/batchnorm/ReadVariableOp8basemodel/batch_normalization_1/batchnorm/ReadVariableOp2|
<basemodel/batch_normalization_1/batchnorm/mul/ReadVariableOp<basemodel/batch_normalization_1/batchnorm/mul/ReadVariableOp2b
/basemodel/batch_normalization_2/AssignMovingAvg/basemodel/batch_normalization_2/AssignMovingAvg2А
>basemodel/batch_normalization_2/AssignMovingAvg/ReadVariableOp>basemodel/batch_normalization_2/AssignMovingAvg/ReadVariableOp2f
1basemodel/batch_normalization_2/AssignMovingAvg_11basemodel/batch_normalization_2/AssignMovingAvg_12Д
@basemodel/batch_normalization_2/AssignMovingAvg_1/ReadVariableOp@basemodel/batch_normalization_2/AssignMovingAvg_1/ReadVariableOp2t
8basemodel/batch_normalization_2/batchnorm/ReadVariableOp8basemodel/batch_normalization_2/batchnorm/ReadVariableOp2|
<basemodel/batch_normalization_2/batchnorm/mul/ReadVariableOp<basemodel/batch_normalization_2/batchnorm/mul/ReadVariableOp2b
/basemodel/batch_normalization_3/AssignMovingAvg/basemodel/batch_normalization_3/AssignMovingAvg2А
>basemodel/batch_normalization_3/AssignMovingAvg/ReadVariableOp>basemodel/batch_normalization_3/AssignMovingAvg/ReadVariableOp2f
1basemodel/batch_normalization_3/AssignMovingAvg_11basemodel/batch_normalization_3/AssignMovingAvg_12Д
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
-dense_1/kernel/Regularizer/Abs/ReadVariableOp-dense_1/kernel/Regularizer/Abs/ReadVariableOp2d
0dense_1/kernel/Regularizer/Square/ReadVariableOp0dense_1/kernel/Regularizer/Square/ReadVariableOp2n
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp2t
8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOp8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOp2n
5stream_0_conv_2/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_2/kernel/Regularizer/Abs/ReadVariableOp2t
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp2n
5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp2t
8stream_0_conv_3/kernel/Regularizer/Square/ReadVariableOp8stream_0_conv_3/kernel/Regularizer/Square/ReadVariableOp:S O
+
_output_shapes
:€€€€€€€€€}
 
_user_specified_nameinputs
ѕ
M
1__inference_dense_1_dropout_layer_call_fn_3298233

inputs
identityЌ
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
GPU2*0J 8В *U
fPRN
L__inference_dense_1_dropout_layer_call_and_return_conditional_losses_32931502
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:€€€€€€€€€@:O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
те
з
F__inference_basemodel_layer_call_and_return_conditional_losses_3294923

inputsQ
;stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource:@=
/stream_0_conv_1_biasadd_readvariableop_resource:@I
;batch_normalization_assignmovingavg_readvariableop_resource:@K
=batch_normalization_assignmovingavg_1_readvariableop_resource:@G
9batch_normalization_batchnorm_mul_readvariableop_resource:@C
5batch_normalization_batchnorm_readvariableop_resource:@Q
;stream_0_conv_2_conv1d_expanddims_1_readvariableop_resource:@@=
/stream_0_conv_2_biasadd_readvariableop_resource:@K
=batch_normalization_1_assignmovingavg_readvariableop_resource:@M
?batch_normalization_1_assignmovingavg_1_readvariableop_resource:@I
;batch_normalization_1_batchnorm_mul_readvariableop_resource:@E
7batch_normalization_1_batchnorm_readvariableop_resource:@Q
;stream_0_conv_3_conv1d_expanddims_1_readvariableop_resource:@@=
/stream_0_conv_3_biasadd_readvariableop_resource:@K
=batch_normalization_2_assignmovingavg_readvariableop_resource:@M
?batch_normalization_2_assignmovingavg_1_readvariableop_resource:@I
;batch_normalization_2_batchnorm_mul_readvariableop_resource:@E
7batch_normalization_2_batchnorm_readvariableop_resource:@8
&dense_1_matmul_readvariableop_resource:@@5
'dense_1_biasadd_readvariableop_resource:@K
=batch_normalization_3_assignmovingavg_readvariableop_resource:@M
?batch_normalization_3_assignmovingavg_1_readvariableop_resource:@I
;batch_normalization_3_batchnorm_mul_readvariableop_resource:@E
7batch_normalization_3_batchnorm_readvariableop_resource:@
identityИҐ#batch_normalization/AssignMovingAvgҐ2batch_normalization/AssignMovingAvg/ReadVariableOpҐ%batch_normalization/AssignMovingAvg_1Ґ4batch_normalization/AssignMovingAvg_1/ReadVariableOpҐ,batch_normalization/batchnorm/ReadVariableOpҐ0batch_normalization/batchnorm/mul/ReadVariableOpҐ%batch_normalization_1/AssignMovingAvgҐ4batch_normalization_1/AssignMovingAvg/ReadVariableOpҐ'batch_normalization_1/AssignMovingAvg_1Ґ6batch_normalization_1/AssignMovingAvg_1/ReadVariableOpҐ.batch_normalization_1/batchnorm/ReadVariableOpҐ2batch_normalization_1/batchnorm/mul/ReadVariableOpҐ%batch_normalization_2/AssignMovingAvgҐ4batch_normalization_2/AssignMovingAvg/ReadVariableOpҐ'batch_normalization_2/AssignMovingAvg_1Ґ6batch_normalization_2/AssignMovingAvg_1/ReadVariableOpҐ.batch_normalization_2/batchnorm/ReadVariableOpҐ2batch_normalization_2/batchnorm/mul/ReadVariableOpҐ%batch_normalization_3/AssignMovingAvgҐ4batch_normalization_3/AssignMovingAvg/ReadVariableOpҐ'batch_normalization_3/AssignMovingAvg_1Ґ6batch_normalization_3/AssignMovingAvg_1/ReadVariableOpҐ.batch_normalization_3/batchnorm/ReadVariableOpҐ2batch_normalization_3/batchnorm/mul/ReadVariableOpҐdense_1/BiasAdd/ReadVariableOpҐdense_1/MatMul/ReadVariableOpҐ-dense_1/kernel/Regularizer/Abs/ReadVariableOpҐ0dense_1/kernel/Regularizer/Square/ReadVariableOpҐ&stream_0_conv_1/BiasAdd/ReadVariableOpҐ2stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOpҐ5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpҐ8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOpҐ&stream_0_conv_2/BiasAdd/ReadVariableOpҐ2stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOpҐ5stream_0_conv_2/kernel/Regularizer/Abs/ReadVariableOpҐ8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOpҐ&stream_0_conv_3/BiasAdd/ReadVariableOpҐ2stream_0_conv_3/conv1d/ExpandDims_1/ReadVariableOpҐ5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOpҐ8stream_0_conv_3/kernel/Regularizer/Square/ReadVariableOpЛ
!stream_0_input_drop/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †?2#
!stream_0_input_drop/dropout/Const≥
stream_0_input_drop/dropout/MulMulinputs*stream_0_input_drop/dropout/Const:output:0*
T0*+
_output_shapes
:€€€€€€€€€}2!
stream_0_input_drop/dropout/Mul|
!stream_0_input_drop/dropout/ShapeShapeinputs*
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
stream_0_conv_1/BiasAddє
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
#batch_normalization/batchnorm/add_1Й
activation/TanhTanh'batch_normalization/batchnorm/add_1:z:0*
T0*+
_output_shapes
:€€€€€€€€€}@2
activation/TanhИ
!stream_0_maxpool_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!stream_0_maxpool_1/ExpandDims/dim«
stream_0_maxpool_1/ExpandDims
ExpandDimsactivation/Tanh:y:0*stream_0_maxpool_1/ExpandDims/dim:output:0*
T0*/
_output_shapes
:€€€€€€€€€}@2
stream_0_maxpool_1/ExpandDimsЎ
stream_0_maxpool_1/MaxPoolMaxPool&stream_0_maxpool_1/ExpandDims:output:0*/
_output_shapes
:€€€€€€€€€>@*
ksize
*
paddingVALID*
strides
2
stream_0_maxpool_1/MaxPoolµ
stream_0_maxpool_1/SqueezeSqueeze#stream_0_maxpool_1/MaxPool:output:0*
T0*+
_output_shapes
:€€€€€€€€€>@*
squeeze_dims
2
stream_0_maxpool_1/SqueezeГ
stream_0_drop_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *Ђ™™?2
stream_0_drop_1/dropout/Constƒ
stream_0_drop_1/dropout/MulMul#stream_0_maxpool_1/Squeeze:output:0&stream_0_drop_1/dropout/Const:output:0*
T0*+
_output_shapes
:€€€€€€€€€>@2
stream_0_drop_1/dropout/MulС
stream_0_drop_1/dropout/ShapeShape#stream_0_maxpool_1/Squeeze:output:0*
T0*
_output_shapes
:2
stream_0_drop_1/dropout/ShapeГ
4stream_0_drop_1/dropout/random_uniform/RandomUniformRandomUniform&stream_0_drop_1/dropout/Shape:output:0*
T0*+
_output_shapes
:€€€€€€€€€>@*
dtype0*
seedЈ*
seed2Є26
4stream_0_drop_1/dropout/random_uniform/RandomUniformХ
&stream_0_drop_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А>2(
&stream_0_drop_1/dropout/GreaterEqual/yВ
$stream_0_drop_1/dropout/GreaterEqualGreaterEqual=stream_0_drop_1/dropout/random_uniform/RandomUniform:output:0/stream_0_drop_1/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:€€€€€€€€€>@2&
$stream_0_drop_1/dropout/GreaterEqual≥
stream_0_drop_1/dropout/CastCast(stream_0_drop_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:€€€€€€€€€>@2
stream_0_drop_1/dropout/CastЊ
stream_0_drop_1/dropout/Mul_1Mulstream_0_drop_1/dropout/Mul:z:0 stream_0_drop_1/dropout/Cast:y:0*
T0*+
_output_shapes
:€€€€€€€€€>@2
stream_0_drop_1/dropout/Mul_1Щ
%stream_0_conv_2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€2'
%stream_0_conv_2/conv1d/ExpandDims/dimб
!stream_0_conv_2/conv1d/ExpandDims
ExpandDims!stream_0_drop_1/dropout/Mul_1:z:0.stream_0_conv_2/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:€€€€€€€€€>@2#
!stream_0_conv_2/conv1d/ExpandDimsи
2stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp;stream_0_conv_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype024
2stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOpФ
'stream_0_conv_2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2)
'stream_0_conv_2/conv1d/ExpandDims_1/dimч
#stream_0_conv_2/conv1d/ExpandDims_1
ExpandDims:stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOp:value:00stream_0_conv_2/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@2%
#stream_0_conv_2/conv1d/ExpandDims_1ц
stream_0_conv_2/conv1dConv2D*stream_0_conv_2/conv1d/ExpandDims:output:0,stream_0_conv_2/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:€€€€€€€€€>@*
paddingSAME*
strides
2
stream_0_conv_2/conv1d¬
stream_0_conv_2/conv1d/SqueezeSqueezestream_0_conv_2/conv1d:output:0*
T0*+
_output_shapes
:€€€€€€€€€>@*
squeeze_dims

э€€€€€€€€2 
stream_0_conv_2/conv1d/SqueezeЉ
&stream_0_conv_2/BiasAdd/ReadVariableOpReadVariableOp/stream_0_conv_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02(
&stream_0_conv_2/BiasAdd/ReadVariableOpћ
stream_0_conv_2/BiasAddBiasAdd'stream_0_conv_2/conv1d/Squeeze:output:0.stream_0_conv_2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€>@2
stream_0_conv_2/BiasAddљ
4batch_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       26
4batch_normalization_1/moments/mean/reduction_indicesп
"batch_normalization_1/moments/meanMean stream_0_conv_2/BiasAdd:output:0=batch_normalization_1/moments/mean/reduction_indices:output:0*
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
/batch_normalization_1/moments/SquaredDifferenceSquaredDifference stream_0_conv_2/BiasAdd:output:03batch_normalization_1/moments/StopGradient:output:0*
T0*+
_output_shapes
:€€€€€€€€€>@21
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
%batch_normalization_1/batchnorm/mul_1Mul stream_0_conv_2/BiasAdd:output:0'batch_normalization_1/batchnorm/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€>@2'
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
:€€€€€€€€€>@2'
%batch_normalization_1/batchnorm/add_1П
activation_1/TanhTanh)batch_normalization_1/batchnorm/add_1:z:0*
T0*+
_output_shapes
:€€€€€€€€€>@2
activation_1/TanhИ
!stream_0_maxpool_2/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!stream_0_maxpool_2/ExpandDims/dim…
stream_0_maxpool_2/ExpandDims
ExpandDimsactivation_1/Tanh:y:0*stream_0_maxpool_2/ExpandDims/dim:output:0*
T0*/
_output_shapes
:€€€€€€€€€>@2
stream_0_maxpool_2/ExpandDimsЎ
stream_0_maxpool_2/MaxPoolMaxPool&stream_0_maxpool_2/ExpandDims:output:0*/
_output_shapes
:€€€€€€€€€@*
ksize
*
paddingVALID*
strides
2
stream_0_maxpool_2/MaxPoolµ
stream_0_maxpool_2/SqueezeSqueeze#stream_0_maxpool_2/MaxPool:output:0*
T0*+
_output_shapes
:€€€€€€€€€@*
squeeze_dims
2
stream_0_maxpool_2/SqueezeГ
stream_0_drop_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *Ђ™™?2
stream_0_drop_2/dropout/Constƒ
stream_0_drop_2/dropout/MulMul#stream_0_maxpool_2/Squeeze:output:0&stream_0_drop_2/dropout/Const:output:0*
T0*+
_output_shapes
:€€€€€€€€€@2
stream_0_drop_2/dropout/MulС
stream_0_drop_2/dropout/ShapeShape#stream_0_maxpool_2/Squeeze:output:0*
T0*
_output_shapes
:2
stream_0_drop_2/dropout/ShapeГ
4stream_0_drop_2/dropout/random_uniform/RandomUniformRandomUniform&stream_0_drop_2/dropout/Shape:output:0*
T0*+
_output_shapes
:€€€€€€€€€@*
dtype0*
seedЈ*
seed2є26
4stream_0_drop_2/dropout/random_uniform/RandomUniformХ
&stream_0_drop_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А>2(
&stream_0_drop_2/dropout/GreaterEqual/yВ
$stream_0_drop_2/dropout/GreaterEqualGreaterEqual=stream_0_drop_2/dropout/random_uniform/RandomUniform:output:0/stream_0_drop_2/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:€€€€€€€€€@2&
$stream_0_drop_2/dropout/GreaterEqual≥
stream_0_drop_2/dropout/CastCast(stream_0_drop_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:€€€€€€€€€@2
stream_0_drop_2/dropout/CastЊ
stream_0_drop_2/dropout/Mul_1Mulstream_0_drop_2/dropout/Mul:z:0 stream_0_drop_2/dropout/Cast:y:0*
T0*+
_output_shapes
:€€€€€€€€€@2
stream_0_drop_2/dropout/Mul_1Щ
%stream_0_conv_3/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€2'
%stream_0_conv_3/conv1d/ExpandDims/dimб
!stream_0_conv_3/conv1d/ExpandDims
ExpandDims!stream_0_drop_2/dropout/Mul_1:z:0.stream_0_conv_3/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:€€€€€€€€€@2#
!stream_0_conv_3/conv1d/ExpandDimsи
2stream_0_conv_3/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp;stream_0_conv_3_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype024
2stream_0_conv_3/conv1d/ExpandDims_1/ReadVariableOpФ
'stream_0_conv_3/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2)
'stream_0_conv_3/conv1d/ExpandDims_1/dimч
#stream_0_conv_3/conv1d/ExpandDims_1
ExpandDims:stream_0_conv_3/conv1d/ExpandDims_1/ReadVariableOp:value:00stream_0_conv_3/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@2%
#stream_0_conv_3/conv1d/ExpandDims_1ц
stream_0_conv_3/conv1dConv2D*stream_0_conv_3/conv1d/ExpandDims:output:0,stream_0_conv_3/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:€€€€€€€€€@*
paddingSAME*
strides
2
stream_0_conv_3/conv1d¬
stream_0_conv_3/conv1d/SqueezeSqueezestream_0_conv_3/conv1d:output:0*
T0*+
_output_shapes
:€€€€€€€€€@*
squeeze_dims

э€€€€€€€€2 
stream_0_conv_3/conv1d/SqueezeЉ
&stream_0_conv_3/BiasAdd/ReadVariableOpReadVariableOp/stream_0_conv_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02(
&stream_0_conv_3/BiasAdd/ReadVariableOpћ
stream_0_conv_3/BiasAddBiasAdd'stream_0_conv_3/conv1d/Squeeze:output:0.stream_0_conv_3/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€@2
stream_0_conv_3/BiasAddљ
4batch_normalization_2/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       26
4batch_normalization_2/moments/mean/reduction_indicesп
"batch_normalization_2/moments/meanMean stream_0_conv_3/BiasAdd:output:0=batch_normalization_2/moments/mean/reduction_indices:output:0*
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
/batch_normalization_2/moments/SquaredDifferenceSquaredDifference stream_0_conv_3/BiasAdd:output:03batch_normalization_2/moments/StopGradient:output:0*
T0*+
_output_shapes
:€€€€€€€€€@21
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
%batch_normalization_2/batchnorm/mul_1Mul stream_0_conv_3/BiasAdd:output:0'batch_normalization_2/batchnorm/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€@2'
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
:€€€€€€€€€@2'
%batch_normalization_2/batchnorm/add_1П
activation_2/TanhTanh)batch_normalization_2/batchnorm/add_1:z:0*
T0*+
_output_shapes
:€€€€€€€€€@2
activation_2/TanhИ
!stream_0_maxpool_3/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!stream_0_maxpool_3/ExpandDims/dim…
stream_0_maxpool_3/ExpandDims
ExpandDimsactivation_2/Tanh:y:0*stream_0_maxpool_3/ExpandDims/dim:output:0*
T0*/
_output_shapes
:€€€€€€€€€@2
stream_0_maxpool_3/ExpandDimsЎ
stream_0_maxpool_3/MaxPoolMaxPool&stream_0_maxpool_3/ExpandDims:output:0*/
_output_shapes
:€€€€€€€€€@*
ksize
*
paddingVALID*
strides
2
stream_0_maxpool_3/MaxPoolµ
stream_0_maxpool_3/SqueezeSqueeze#stream_0_maxpool_3/MaxPool:output:0*
T0*+
_output_shapes
:€€€€€€€€€@*
squeeze_dims
2
stream_0_maxpool_3/SqueezeГ
stream_0_drop_3/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *Ђ™™?2
stream_0_drop_3/dropout/Constƒ
stream_0_drop_3/dropout/MulMul#stream_0_maxpool_3/Squeeze:output:0&stream_0_drop_3/dropout/Const:output:0*
T0*+
_output_shapes
:€€€€€€€€€@2
stream_0_drop_3/dropout/MulС
stream_0_drop_3/dropout/ShapeShape#stream_0_maxpool_3/Squeeze:output:0*
T0*
_output_shapes
:2
stream_0_drop_3/dropout/ShapeГ
4stream_0_drop_3/dropout/random_uniform/RandomUniformRandomUniform&stream_0_drop_3/dropout/Shape:output:0*
T0*+
_output_shapes
:€€€€€€€€€@*
dtype0*
seedЈ*
seed2Ї26
4stream_0_drop_3/dropout/random_uniform/RandomUniformХ
&stream_0_drop_3/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А>2(
&stream_0_drop_3/dropout/GreaterEqual/yВ
$stream_0_drop_3/dropout/GreaterEqualGreaterEqual=stream_0_drop_3/dropout/random_uniform/RandomUniform:output:0/stream_0_drop_3/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:€€€€€€€€€@2&
$stream_0_drop_3/dropout/GreaterEqual≥
stream_0_drop_3/dropout/CastCast(stream_0_drop_3/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:€€€€€€€€€@2
stream_0_drop_3/dropout/CastЊ
stream_0_drop_3/dropout/Mul_1Mulstream_0_drop_3/dropout/Mul:z:0 stream_0_drop_3/dropout/Cast:y:0*
T0*+
_output_shapes
:€€€€€€€€€@2
stream_0_drop_3/dropout/Mul_1§
/global_average_pooling1d/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :21
/global_average_pooling1d/Mean/reduction_indices’
global_average_pooling1d/MeanMean!stream_0_drop_3/dropout/Mul_1:z:08global_average_pooling1d/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
global_average_pooling1d/Mean•
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02
dense_1/MatMul/ReadVariableOpЂ
dense_1/MatMulMatMul&global_average_pooling1d/Mean:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2
dense_1/MatMul§
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02 
dense_1/BiasAdd/ReadVariableOp°
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2
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

:@*
	keep_dims(2$
"batch_normalization_3/moments/meanЊ
*batch_normalization_3/moments/StopGradientStopGradient+batch_normalization_3/moments/mean:output:0*
T0*
_output_shapes

:@2,
*batch_normalization_3/moments/StopGradientш
/batch_normalization_3/moments/SquaredDifferenceSquaredDifferencedense_1/BiasAdd:output:03batch_normalization_3/moments/StopGradient:output:0*
T0*'
_output_shapes
:€€€€€€€€€@21
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

:@*
	keep_dims(2(
&batch_normalization_3/moments/variance¬
%batch_normalization_3/moments/SqueezeSqueeze+batch_normalization_3/moments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 2'
%batch_normalization_3/moments/Squeeze 
'batch_normalization_3/moments/Squeeze_1Squeeze/batch_normalization_3/moments/variance:output:0*
T0*
_output_shapes
:@*
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
:@*
dtype026
4batch_normalization_3/AssignMovingAvg/ReadVariableOpр
)batch_normalization_3/AssignMovingAvg/subSub<batch_normalization_3/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_3/moments/Squeeze:output:0*
T0*
_output_shapes
:@2+
)batch_normalization_3/AssignMovingAvg/subз
)batch_normalization_3/AssignMovingAvg/mulMul-batch_normalization_3/AssignMovingAvg/sub:z:04batch_normalization_3/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:@2+
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
:@*
dtype028
6batch_normalization_3/AssignMovingAvg_1/ReadVariableOpш
+batch_normalization_3/AssignMovingAvg_1/subSub>batch_normalization_3/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_3/moments/Squeeze_1:output:0*
T0*
_output_shapes
:@2-
+batch_normalization_3/AssignMovingAvg_1/subп
+batch_normalization_3/AssignMovingAvg_1/mulMul/batch_normalization_3/AssignMovingAvg_1/sub:z:06batch_normalization_3/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:@2-
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
:@2%
#batch_normalization_3/batchnorm/add•
%batch_normalization_3/batchnorm/RsqrtRsqrt'batch_normalization_3/batchnorm/add:z:0*
T0*
_output_shapes
:@2'
%batch_normalization_3/batchnorm/Rsqrtа
2batch_normalization_3/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_3_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype024
2batch_normalization_3/batchnorm/mul/ReadVariableOpЁ
#batch_normalization_3/batchnorm/mulMul)batch_normalization_3/batchnorm/Rsqrt:y:0:batch_normalization_3/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2%
#batch_normalization_3/batchnorm/mul 
%batch_normalization_3/batchnorm/mul_1Muldense_1/BiasAdd:output:0'batch_normalization_3/batchnorm/mul:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2'
%batch_normalization_3/batchnorm/mul_1”
%batch_normalization_3/batchnorm/mul_2Mul.batch_normalization_3/moments/Squeeze:output:0'batch_normalization_3/batchnorm/mul:z:0*
T0*
_output_shapes
:@2'
%batch_normalization_3/batchnorm/mul_2‘
.batch_normalization_3/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_3_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype020
.batch_normalization_3/batchnorm/ReadVariableOpў
#batch_normalization_3/batchnorm/subSub6batch_normalization_3/batchnorm/ReadVariableOp:value:0)batch_normalization_3/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2%
#batch_normalization_3/batchnorm/subЁ
%batch_normalization_3/batchnorm/add_1AddV2)batch_normalization_3/batchnorm/mul_1:z:0'batch_normalization_3/batchnorm/sub:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2'
%batch_normalization_3/batchnorm/add_1Щ
(stream_0_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(stream_0_conv_1/kernel/Regularizer/Constо
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp;stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype027
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp√
&stream_0_conv_1/kernel/Regularizer/AbsAbs=stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_0_conv_1/kernel/Regularizer/Abs≠
*stream_0_conv_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_0_conv_1/kernel/Regularizer/Const_1ў
&stream_0_conv_1/kernel/Regularizer/SumSum*stream_0_conv_1/kernel/Regularizer/Abs:y:03stream_0_conv_1/kernel/Regularizer/Const_1:output:0*
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
&stream_0_conv_1/kernel/Regularizer/mulў
&stream_0_conv_1/kernel/Regularizer/addAddV21stream_0_conv_1/kernel/Regularizer/Const:output:0*stream_0_conv_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/addф
8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02:
8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOpѕ
)stream_0_conv_1/kernel/Regularizer/SquareSquare@stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2+
)stream_0_conv_1/kernel/Regularizer/Square≠
*stream_0_conv_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_0_conv_1/kernel/Regularizer/Const_2а
(stream_0_conv_1/kernel/Regularizer/Sum_1Sum-stream_0_conv_1/kernel/Regularizer/Square:y:03stream_0_conv_1/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2*
(stream_0_conv_1/kernel/Regularizer/Sum_1Э
*stream_0_conv_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2,
*stream_0_conv_1/kernel/Regularizer/mul_1/xд
(stream_0_conv_1/kernel/Regularizer/mul_1Mul3stream_0_conv_1/kernel/Regularizer/mul_1/x:output:01stream_0_conv_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2*
(stream_0_conv_1/kernel/Regularizer/mul_1Ў
(stream_0_conv_1/kernel/Regularizer/add_1AddV2*stream_0_conv_1/kernel/Regularizer/add:z:0,stream_0_conv_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2*
(stream_0_conv_1/kernel/Regularizer/add_1Щ
(stream_0_conv_2/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(stream_0_conv_2/kernel/Regularizer/Constо
5stream_0_conv_2/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp;stream_0_conv_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype027
5stream_0_conv_2/kernel/Regularizer/Abs/ReadVariableOp√
&stream_0_conv_2/kernel/Regularizer/AbsAbs=stream_0_conv_2/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@@2(
&stream_0_conv_2/kernel/Regularizer/Abs≠
*stream_0_conv_2/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_0_conv_2/kernel/Regularizer/Const_1ў
&stream_0_conv_2/kernel/Regularizer/SumSum*stream_0_conv_2/kernel/Regularizer/Abs:y:03stream_0_conv_2/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_2/kernel/Regularizer/SumЩ
(stream_0_conv_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2*
(stream_0_conv_2/kernel/Regularizer/mul/x№
&stream_0_conv_2/kernel/Regularizer/mulMul1stream_0_conv_2/kernel/Regularizer/mul/x:output:0/stream_0_conv_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_2/kernel/Regularizer/mulў
&stream_0_conv_2/kernel/Regularizer/addAddV21stream_0_conv_2/kernel/Regularizer/Const:output:0*stream_0_conv_2/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&stream_0_conv_2/kernel/Regularizer/addф
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;stream_0_conv_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype02:
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOpѕ
)stream_0_conv_2/kernel/Regularizer/SquareSquare@stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@@2+
)stream_0_conv_2/kernel/Regularizer/Square≠
*stream_0_conv_2/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_0_conv_2/kernel/Regularizer/Const_2а
(stream_0_conv_2/kernel/Regularizer/Sum_1Sum-stream_0_conv_2/kernel/Regularizer/Square:y:03stream_0_conv_2/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2*
(stream_0_conv_2/kernel/Regularizer/Sum_1Э
*stream_0_conv_2/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2,
*stream_0_conv_2/kernel/Regularizer/mul_1/xд
(stream_0_conv_2/kernel/Regularizer/mul_1Mul3stream_0_conv_2/kernel/Regularizer/mul_1/x:output:01stream_0_conv_2/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2*
(stream_0_conv_2/kernel/Regularizer/mul_1Ў
(stream_0_conv_2/kernel/Regularizer/add_1AddV2*stream_0_conv_2/kernel/Regularizer/add:z:0,stream_0_conv_2/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2*
(stream_0_conv_2/kernel/Regularizer/add_1Щ
(stream_0_conv_3/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(stream_0_conv_3/kernel/Regularizer/Constо
5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp;stream_0_conv_3_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype027
5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp√
&stream_0_conv_3/kernel/Regularizer/AbsAbs=stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@@2(
&stream_0_conv_3/kernel/Regularizer/Abs≠
*stream_0_conv_3/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_0_conv_3/kernel/Regularizer/Const_1ў
&stream_0_conv_3/kernel/Regularizer/SumSum*stream_0_conv_3/kernel/Regularizer/Abs:y:03stream_0_conv_3/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_3/kernel/Regularizer/SumЩ
(stream_0_conv_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2*
(stream_0_conv_3/kernel/Regularizer/mul/x№
&stream_0_conv_3/kernel/Regularizer/mulMul1stream_0_conv_3/kernel/Regularizer/mul/x:output:0/stream_0_conv_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_3/kernel/Regularizer/mulў
&stream_0_conv_3/kernel/Regularizer/addAddV21stream_0_conv_3/kernel/Regularizer/Const:output:0*stream_0_conv_3/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&stream_0_conv_3/kernel/Regularizer/addф
8stream_0_conv_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;stream_0_conv_3_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype02:
8stream_0_conv_3/kernel/Regularizer/Square/ReadVariableOpѕ
)stream_0_conv_3/kernel/Regularizer/SquareSquare@stream_0_conv_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@@2+
)stream_0_conv_3/kernel/Regularizer/Square≠
*stream_0_conv_3/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_0_conv_3/kernel/Regularizer/Const_2а
(stream_0_conv_3/kernel/Regularizer/Sum_1Sum-stream_0_conv_3/kernel/Regularizer/Square:y:03stream_0_conv_3/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2*
(stream_0_conv_3/kernel/Regularizer/Sum_1Э
*stream_0_conv_3/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2,
*stream_0_conv_3/kernel/Regularizer/mul_1/xд
(stream_0_conv_3/kernel/Regularizer/mul_1Mul3stream_0_conv_3/kernel/Regularizer/mul_1/x:output:01stream_0_conv_3/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2*
(stream_0_conv_3/kernel/Regularizer/mul_1Ў
(stream_0_conv_3/kernel/Regularizer/add_1AddV2*stream_0_conv_3/kernel/Regularizer/add:z:0,stream_0_conv_3/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2*
(stream_0_conv_3/kernel/Regularizer/add_1Й
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dense_1/kernel/Regularizer/Const≈
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOpІ
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:@@2 
dense_1/kernel/Regularizer/AbsЩ
"dense_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_1/kernel/Regularizer/Const_1є
dense_1/kernel/Regularizer/SumSum"dense_1/kernel/Regularizer/Abs:y:0+dense_1/kernel/Regularizer/Const_1:output:0*
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
dense_1/kernel/Regularizer/mulє
dense_1/kernel/Regularizer/addAddV2)dense_1/kernel/Regularizer/Const:output:0"dense_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/addЋ
0dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype022
0dense_1/kernel/Regularizer/Square/ReadVariableOp≥
!dense_1/kernel/Regularizer/SquareSquare8dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@@2#
!dense_1/kernel/Regularizer/SquareЩ
"dense_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_1/kernel/Regularizer/Const_2ј
 dense_1/kernel/Regularizer/Sum_1Sum%dense_1/kernel/Regularizer/Square:y:0+dense_1/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2"
 dense_1/kernel/Regularizer/Sum_1Н
"dense_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2$
"dense_1/kernel/Regularizer/mul_1/xƒ
 dense_1/kernel/Regularizer/mul_1Mul+dense_1/kernel/Regularizer/mul_1/x:output:0)dense_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2"
 dense_1/kernel/Regularizer/mul_1Є
 dense_1/kernel/Regularizer/add_1AddV2"dense_1/kernel/Regularizer/add:z:0$dense_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2"
 dense_1/kernel/Regularizer/add_1Д
IdentityIdentity)batch_normalization_3/batchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@2

Identityщ
NoOpNoOp$^batch_normalization/AssignMovingAvg3^batch_normalization/AssignMovingAvg/ReadVariableOp&^batch_normalization/AssignMovingAvg_15^batch_normalization/AssignMovingAvg_1/ReadVariableOp-^batch_normalization/batchnorm/ReadVariableOp1^batch_normalization/batchnorm/mul/ReadVariableOp&^batch_normalization_1/AssignMovingAvg5^batch_normalization_1/AssignMovingAvg/ReadVariableOp(^batch_normalization_1/AssignMovingAvg_17^batch_normalization_1/AssignMovingAvg_1/ReadVariableOp/^batch_normalization_1/batchnorm/ReadVariableOp3^batch_normalization_1/batchnorm/mul/ReadVariableOp&^batch_normalization_2/AssignMovingAvg5^batch_normalization_2/AssignMovingAvg/ReadVariableOp(^batch_normalization_2/AssignMovingAvg_17^batch_normalization_2/AssignMovingAvg_1/ReadVariableOp/^batch_normalization_2/batchnorm/ReadVariableOp3^batch_normalization_2/batchnorm/mul/ReadVariableOp&^batch_normalization_3/AssignMovingAvg5^batch_normalization_3/AssignMovingAvg/ReadVariableOp(^batch_normalization_3/AssignMovingAvg_17^batch_normalization_3/AssignMovingAvg_1/ReadVariableOp/^batch_normalization_3/batchnorm/ReadVariableOp3^batch_normalization_3/batchnorm/mul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp.^dense_1/kernel/Regularizer/Abs/ReadVariableOp1^dense_1/kernel/Regularizer/Square/ReadVariableOp'^stream_0_conv_1/BiasAdd/ReadVariableOp3^stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp9^stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOp'^stream_0_conv_2/BiasAdd/ReadVariableOp3^stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOp6^stream_0_conv_2/kernel/Regularizer/Abs/ReadVariableOp9^stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp'^stream_0_conv_3/BiasAdd/ReadVariableOp3^stream_0_conv_3/conv1d/ExpandDims_1/ReadVariableOp6^stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp9^stream_0_conv_3/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:€€€€€€€€€}: : : : : : : : : : : : : : : : : : : : : : : : 2J
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
-dense_1/kernel/Regularizer/Abs/ReadVariableOp-dense_1/kernel/Regularizer/Abs/ReadVariableOp2d
0dense_1/kernel/Regularizer/Square/ReadVariableOp0dense_1/kernel/Regularizer/Square/ReadVariableOp2P
&stream_0_conv_1/BiasAdd/ReadVariableOp&stream_0_conv_1/BiasAdd/ReadVariableOp2h
2stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp2stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp2n
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp2t
8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOp8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOp2P
&stream_0_conv_2/BiasAdd/ReadVariableOp&stream_0_conv_2/BiasAdd/ReadVariableOp2h
2stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOp2stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOp2n
5stream_0_conv_2/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_2/kernel/Regularizer/Abs/ReadVariableOp2t
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp2P
&stream_0_conv_3/BiasAdd/ReadVariableOp&stream_0_conv_3/BiasAdd/ReadVariableOp2h
2stream_0_conv_3/conv1d/ExpandDims_1/ReadVariableOp2stream_0_conv_3/conv1d/ExpandDims_1/ReadVariableOp2n
5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp2t
8stream_0_conv_3/kernel/Regularizer/Square/ReadVariableOp8stream_0_conv_3/kernel/Regularizer/Square/ReadVariableOp:S O
+
_output_shapes
:€€€€€€€€€}
 
_user_specified_nameinputs
Ш
Ґ
1__inference_stream_0_conv_1_layer_call_fn_3297399

inputs
unknown:@
	unknown_0:@
identityИҐStatefulPartitionedCallГ
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
GPU2*0J 8В *U
fPRN
L__inference_stream_0_conv_1_layer_call_and_return_conditional_losses_32929042
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
ш
щ
__inference_loss_fn_0_3298405T
>stream_0_conv_1_kernel_regularizer_abs_readvariableop_resource:@
identityИҐ5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpҐ8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOpЩ
(stream_0_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(stream_0_conv_1/kernel/Regularizer/Constс
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp>stream_0_conv_1_kernel_regularizer_abs_readvariableop_resource*"
_output_shapes
:@*
dtype027
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp√
&stream_0_conv_1/kernel/Regularizer/AbsAbs=stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_0_conv_1/kernel/Regularizer/Abs≠
*stream_0_conv_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_0_conv_1/kernel/Regularizer/Const_1ў
&stream_0_conv_1/kernel/Regularizer/SumSum*stream_0_conv_1/kernel/Regularizer/Abs:y:03stream_0_conv_1/kernel/Regularizer/Const_1:output:0*
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
&stream_0_conv_1/kernel/Regularizer/mulў
&stream_0_conv_1/kernel/Regularizer/addAddV21stream_0_conv_1/kernel/Regularizer/Const:output:0*stream_0_conv_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/addч
8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp>stream_0_conv_1_kernel_regularizer_abs_readvariableop_resource*"
_output_shapes
:@*
dtype02:
8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOpѕ
)stream_0_conv_1/kernel/Regularizer/SquareSquare@stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2+
)stream_0_conv_1/kernel/Regularizer/Square≠
*stream_0_conv_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_0_conv_1/kernel/Regularizer/Const_2а
(stream_0_conv_1/kernel/Regularizer/Sum_1Sum-stream_0_conv_1/kernel/Regularizer/Square:y:03stream_0_conv_1/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2*
(stream_0_conv_1/kernel/Regularizer/Sum_1Э
*stream_0_conv_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2,
*stream_0_conv_1/kernel/Regularizer/mul_1/xд
(stream_0_conv_1/kernel/Regularizer/mul_1Mul3stream_0_conv_1/kernel/Regularizer/mul_1/x:output:01stream_0_conv_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2*
(stream_0_conv_1/kernel/Regularizer/mul_1Ў
(stream_0_conv_1/kernel/Regularizer/add_1AddV2*stream_0_conv_1/kernel/Regularizer/add:z:0,stream_0_conv_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2*
(stream_0_conv_1/kernel/Regularizer/add_1v
IdentityIdentity,stream_0_conv_1/kernel/Regularizer/add_1:z:0^NoOp*
T0*
_output_shapes
: 2

IdentityЅ
NoOpNoOp6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp9^stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2n
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp2t
8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOp8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOp
е
P
4__inference_stream_0_maxpool_1_layer_call_fn_3297609

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
:€€€€€€€€€>@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_stream_0_maxpool_1_layer_call_and_return_conditional_losses_32929532
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:€€€€€€€€€>@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€}@:S O
+
_output_shapes
:€€€€€€€€€}@
 
_user_specified_nameinputs
Є
±
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_3297778

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
’
H
,__inference_activation_layer_call_fn_3297594

inputs
identityћ
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
GPU2*0J 8В *P
fKRI
G__inference_activation_layer_call_and_return_conditional_losses_32929442
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
Є
±
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_3292508

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
ъе
й
F__inference_basemodel_layer_call_and_return_conditional_losses_3297348
inputs_0Q
;stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource:@=
/stream_0_conv_1_biasadd_readvariableop_resource:@I
;batch_normalization_assignmovingavg_readvariableop_resource:@K
=batch_normalization_assignmovingavg_1_readvariableop_resource:@G
9batch_normalization_batchnorm_mul_readvariableop_resource:@C
5batch_normalization_batchnorm_readvariableop_resource:@Q
;stream_0_conv_2_conv1d_expanddims_1_readvariableop_resource:@@=
/stream_0_conv_2_biasadd_readvariableop_resource:@K
=batch_normalization_1_assignmovingavg_readvariableop_resource:@M
?batch_normalization_1_assignmovingavg_1_readvariableop_resource:@I
;batch_normalization_1_batchnorm_mul_readvariableop_resource:@E
7batch_normalization_1_batchnorm_readvariableop_resource:@Q
;stream_0_conv_3_conv1d_expanddims_1_readvariableop_resource:@@=
/stream_0_conv_3_biasadd_readvariableop_resource:@K
=batch_normalization_2_assignmovingavg_readvariableop_resource:@M
?batch_normalization_2_assignmovingavg_1_readvariableop_resource:@I
;batch_normalization_2_batchnorm_mul_readvariableop_resource:@E
7batch_normalization_2_batchnorm_readvariableop_resource:@8
&dense_1_matmul_readvariableop_resource:@@5
'dense_1_biasadd_readvariableop_resource:@K
=batch_normalization_3_assignmovingavg_readvariableop_resource:@M
?batch_normalization_3_assignmovingavg_1_readvariableop_resource:@I
;batch_normalization_3_batchnorm_mul_readvariableop_resource:@E
7batch_normalization_3_batchnorm_readvariableop_resource:@
identityИҐ#batch_normalization/AssignMovingAvgҐ2batch_normalization/AssignMovingAvg/ReadVariableOpҐ%batch_normalization/AssignMovingAvg_1Ґ4batch_normalization/AssignMovingAvg_1/ReadVariableOpҐ,batch_normalization/batchnorm/ReadVariableOpҐ0batch_normalization/batchnorm/mul/ReadVariableOpҐ%batch_normalization_1/AssignMovingAvgҐ4batch_normalization_1/AssignMovingAvg/ReadVariableOpҐ'batch_normalization_1/AssignMovingAvg_1Ґ6batch_normalization_1/AssignMovingAvg_1/ReadVariableOpҐ.batch_normalization_1/batchnorm/ReadVariableOpҐ2batch_normalization_1/batchnorm/mul/ReadVariableOpҐ%batch_normalization_2/AssignMovingAvgҐ4batch_normalization_2/AssignMovingAvg/ReadVariableOpҐ'batch_normalization_2/AssignMovingAvg_1Ґ6batch_normalization_2/AssignMovingAvg_1/ReadVariableOpҐ.batch_normalization_2/batchnorm/ReadVariableOpҐ2batch_normalization_2/batchnorm/mul/ReadVariableOpҐ%batch_normalization_3/AssignMovingAvgҐ4batch_normalization_3/AssignMovingAvg/ReadVariableOpҐ'batch_normalization_3/AssignMovingAvg_1Ґ6batch_normalization_3/AssignMovingAvg_1/ReadVariableOpҐ.batch_normalization_3/batchnorm/ReadVariableOpҐ2batch_normalization_3/batchnorm/mul/ReadVariableOpҐdense_1/BiasAdd/ReadVariableOpҐdense_1/MatMul/ReadVariableOpҐ-dense_1/kernel/Regularizer/Abs/ReadVariableOpҐ0dense_1/kernel/Regularizer/Square/ReadVariableOpҐ&stream_0_conv_1/BiasAdd/ReadVariableOpҐ2stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOpҐ5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpҐ8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOpҐ&stream_0_conv_2/BiasAdd/ReadVariableOpҐ2stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOpҐ5stream_0_conv_2/kernel/Regularizer/Abs/ReadVariableOpҐ8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOpҐ&stream_0_conv_3/BiasAdd/ReadVariableOpҐ2stream_0_conv_3/conv1d/ExpandDims_1/ReadVariableOpҐ5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOpҐ8stream_0_conv_3/kernel/Regularizer/Square/ReadVariableOpЛ
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
stream_0_conv_1/BiasAddє
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
#batch_normalization/batchnorm/add_1Й
activation/TanhTanh'batch_normalization/batchnorm/add_1:z:0*
T0*+
_output_shapes
:€€€€€€€€€}@2
activation/TanhИ
!stream_0_maxpool_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!stream_0_maxpool_1/ExpandDims/dim«
stream_0_maxpool_1/ExpandDims
ExpandDimsactivation/Tanh:y:0*stream_0_maxpool_1/ExpandDims/dim:output:0*
T0*/
_output_shapes
:€€€€€€€€€}@2
stream_0_maxpool_1/ExpandDimsЎ
stream_0_maxpool_1/MaxPoolMaxPool&stream_0_maxpool_1/ExpandDims:output:0*/
_output_shapes
:€€€€€€€€€>@*
ksize
*
paddingVALID*
strides
2
stream_0_maxpool_1/MaxPoolµ
stream_0_maxpool_1/SqueezeSqueeze#stream_0_maxpool_1/MaxPool:output:0*
T0*+
_output_shapes
:€€€€€€€€€>@*
squeeze_dims
2
stream_0_maxpool_1/SqueezeГ
stream_0_drop_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *Ђ™™?2
stream_0_drop_1/dropout/Constƒ
stream_0_drop_1/dropout/MulMul#stream_0_maxpool_1/Squeeze:output:0&stream_0_drop_1/dropout/Const:output:0*
T0*+
_output_shapes
:€€€€€€€€€>@2
stream_0_drop_1/dropout/MulС
stream_0_drop_1/dropout/ShapeShape#stream_0_maxpool_1/Squeeze:output:0*
T0*
_output_shapes
:2
stream_0_drop_1/dropout/ShapeГ
4stream_0_drop_1/dropout/random_uniform/RandomUniformRandomUniform&stream_0_drop_1/dropout/Shape:output:0*
T0*+
_output_shapes
:€€€€€€€€€>@*
dtype0*
seedЈ*
seed2Є26
4stream_0_drop_1/dropout/random_uniform/RandomUniformХ
&stream_0_drop_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А>2(
&stream_0_drop_1/dropout/GreaterEqual/yВ
$stream_0_drop_1/dropout/GreaterEqualGreaterEqual=stream_0_drop_1/dropout/random_uniform/RandomUniform:output:0/stream_0_drop_1/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:€€€€€€€€€>@2&
$stream_0_drop_1/dropout/GreaterEqual≥
stream_0_drop_1/dropout/CastCast(stream_0_drop_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:€€€€€€€€€>@2
stream_0_drop_1/dropout/CastЊ
stream_0_drop_1/dropout/Mul_1Mulstream_0_drop_1/dropout/Mul:z:0 stream_0_drop_1/dropout/Cast:y:0*
T0*+
_output_shapes
:€€€€€€€€€>@2
stream_0_drop_1/dropout/Mul_1Щ
%stream_0_conv_2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€2'
%stream_0_conv_2/conv1d/ExpandDims/dimб
!stream_0_conv_2/conv1d/ExpandDims
ExpandDims!stream_0_drop_1/dropout/Mul_1:z:0.stream_0_conv_2/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:€€€€€€€€€>@2#
!stream_0_conv_2/conv1d/ExpandDimsи
2stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp;stream_0_conv_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype024
2stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOpФ
'stream_0_conv_2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2)
'stream_0_conv_2/conv1d/ExpandDims_1/dimч
#stream_0_conv_2/conv1d/ExpandDims_1
ExpandDims:stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOp:value:00stream_0_conv_2/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@2%
#stream_0_conv_2/conv1d/ExpandDims_1ц
stream_0_conv_2/conv1dConv2D*stream_0_conv_2/conv1d/ExpandDims:output:0,stream_0_conv_2/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:€€€€€€€€€>@*
paddingSAME*
strides
2
stream_0_conv_2/conv1d¬
stream_0_conv_2/conv1d/SqueezeSqueezestream_0_conv_2/conv1d:output:0*
T0*+
_output_shapes
:€€€€€€€€€>@*
squeeze_dims

э€€€€€€€€2 
stream_0_conv_2/conv1d/SqueezeЉ
&stream_0_conv_2/BiasAdd/ReadVariableOpReadVariableOp/stream_0_conv_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02(
&stream_0_conv_2/BiasAdd/ReadVariableOpћ
stream_0_conv_2/BiasAddBiasAdd'stream_0_conv_2/conv1d/Squeeze:output:0.stream_0_conv_2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€>@2
stream_0_conv_2/BiasAddљ
4batch_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       26
4batch_normalization_1/moments/mean/reduction_indicesп
"batch_normalization_1/moments/meanMean stream_0_conv_2/BiasAdd:output:0=batch_normalization_1/moments/mean/reduction_indices:output:0*
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
/batch_normalization_1/moments/SquaredDifferenceSquaredDifference stream_0_conv_2/BiasAdd:output:03batch_normalization_1/moments/StopGradient:output:0*
T0*+
_output_shapes
:€€€€€€€€€>@21
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
%batch_normalization_1/batchnorm/mul_1Mul stream_0_conv_2/BiasAdd:output:0'batch_normalization_1/batchnorm/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€>@2'
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
:€€€€€€€€€>@2'
%batch_normalization_1/batchnorm/add_1П
activation_1/TanhTanh)batch_normalization_1/batchnorm/add_1:z:0*
T0*+
_output_shapes
:€€€€€€€€€>@2
activation_1/TanhИ
!stream_0_maxpool_2/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!stream_0_maxpool_2/ExpandDims/dim…
stream_0_maxpool_2/ExpandDims
ExpandDimsactivation_1/Tanh:y:0*stream_0_maxpool_2/ExpandDims/dim:output:0*
T0*/
_output_shapes
:€€€€€€€€€>@2
stream_0_maxpool_2/ExpandDimsЎ
stream_0_maxpool_2/MaxPoolMaxPool&stream_0_maxpool_2/ExpandDims:output:0*/
_output_shapes
:€€€€€€€€€@*
ksize
*
paddingVALID*
strides
2
stream_0_maxpool_2/MaxPoolµ
stream_0_maxpool_2/SqueezeSqueeze#stream_0_maxpool_2/MaxPool:output:0*
T0*+
_output_shapes
:€€€€€€€€€@*
squeeze_dims
2
stream_0_maxpool_2/SqueezeГ
stream_0_drop_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *Ђ™™?2
stream_0_drop_2/dropout/Constƒ
stream_0_drop_2/dropout/MulMul#stream_0_maxpool_2/Squeeze:output:0&stream_0_drop_2/dropout/Const:output:0*
T0*+
_output_shapes
:€€€€€€€€€@2
stream_0_drop_2/dropout/MulС
stream_0_drop_2/dropout/ShapeShape#stream_0_maxpool_2/Squeeze:output:0*
T0*
_output_shapes
:2
stream_0_drop_2/dropout/ShapeГ
4stream_0_drop_2/dropout/random_uniform/RandomUniformRandomUniform&stream_0_drop_2/dropout/Shape:output:0*
T0*+
_output_shapes
:€€€€€€€€€@*
dtype0*
seedЈ*
seed2є26
4stream_0_drop_2/dropout/random_uniform/RandomUniformХ
&stream_0_drop_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А>2(
&stream_0_drop_2/dropout/GreaterEqual/yВ
$stream_0_drop_2/dropout/GreaterEqualGreaterEqual=stream_0_drop_2/dropout/random_uniform/RandomUniform:output:0/stream_0_drop_2/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:€€€€€€€€€@2&
$stream_0_drop_2/dropout/GreaterEqual≥
stream_0_drop_2/dropout/CastCast(stream_0_drop_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:€€€€€€€€€@2
stream_0_drop_2/dropout/CastЊ
stream_0_drop_2/dropout/Mul_1Mulstream_0_drop_2/dropout/Mul:z:0 stream_0_drop_2/dropout/Cast:y:0*
T0*+
_output_shapes
:€€€€€€€€€@2
stream_0_drop_2/dropout/Mul_1Щ
%stream_0_conv_3/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€2'
%stream_0_conv_3/conv1d/ExpandDims/dimб
!stream_0_conv_3/conv1d/ExpandDims
ExpandDims!stream_0_drop_2/dropout/Mul_1:z:0.stream_0_conv_3/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:€€€€€€€€€@2#
!stream_0_conv_3/conv1d/ExpandDimsи
2stream_0_conv_3/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp;stream_0_conv_3_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype024
2stream_0_conv_3/conv1d/ExpandDims_1/ReadVariableOpФ
'stream_0_conv_3/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2)
'stream_0_conv_3/conv1d/ExpandDims_1/dimч
#stream_0_conv_3/conv1d/ExpandDims_1
ExpandDims:stream_0_conv_3/conv1d/ExpandDims_1/ReadVariableOp:value:00stream_0_conv_3/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@2%
#stream_0_conv_3/conv1d/ExpandDims_1ц
stream_0_conv_3/conv1dConv2D*stream_0_conv_3/conv1d/ExpandDims:output:0,stream_0_conv_3/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:€€€€€€€€€@*
paddingSAME*
strides
2
stream_0_conv_3/conv1d¬
stream_0_conv_3/conv1d/SqueezeSqueezestream_0_conv_3/conv1d:output:0*
T0*+
_output_shapes
:€€€€€€€€€@*
squeeze_dims

э€€€€€€€€2 
stream_0_conv_3/conv1d/SqueezeЉ
&stream_0_conv_3/BiasAdd/ReadVariableOpReadVariableOp/stream_0_conv_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02(
&stream_0_conv_3/BiasAdd/ReadVariableOpћ
stream_0_conv_3/BiasAddBiasAdd'stream_0_conv_3/conv1d/Squeeze:output:0.stream_0_conv_3/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€@2
stream_0_conv_3/BiasAddљ
4batch_normalization_2/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       26
4batch_normalization_2/moments/mean/reduction_indicesп
"batch_normalization_2/moments/meanMean stream_0_conv_3/BiasAdd:output:0=batch_normalization_2/moments/mean/reduction_indices:output:0*
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
/batch_normalization_2/moments/SquaredDifferenceSquaredDifference stream_0_conv_3/BiasAdd:output:03batch_normalization_2/moments/StopGradient:output:0*
T0*+
_output_shapes
:€€€€€€€€€@21
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
%batch_normalization_2/batchnorm/mul_1Mul stream_0_conv_3/BiasAdd:output:0'batch_normalization_2/batchnorm/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€@2'
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
:€€€€€€€€€@2'
%batch_normalization_2/batchnorm/add_1П
activation_2/TanhTanh)batch_normalization_2/batchnorm/add_1:z:0*
T0*+
_output_shapes
:€€€€€€€€€@2
activation_2/TanhИ
!stream_0_maxpool_3/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!stream_0_maxpool_3/ExpandDims/dim…
stream_0_maxpool_3/ExpandDims
ExpandDimsactivation_2/Tanh:y:0*stream_0_maxpool_3/ExpandDims/dim:output:0*
T0*/
_output_shapes
:€€€€€€€€€@2
stream_0_maxpool_3/ExpandDimsЎ
stream_0_maxpool_3/MaxPoolMaxPool&stream_0_maxpool_3/ExpandDims:output:0*/
_output_shapes
:€€€€€€€€€@*
ksize
*
paddingVALID*
strides
2
stream_0_maxpool_3/MaxPoolµ
stream_0_maxpool_3/SqueezeSqueeze#stream_0_maxpool_3/MaxPool:output:0*
T0*+
_output_shapes
:€€€€€€€€€@*
squeeze_dims
2
stream_0_maxpool_3/SqueezeГ
stream_0_drop_3/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *Ђ™™?2
stream_0_drop_3/dropout/Constƒ
stream_0_drop_3/dropout/MulMul#stream_0_maxpool_3/Squeeze:output:0&stream_0_drop_3/dropout/Const:output:0*
T0*+
_output_shapes
:€€€€€€€€€@2
stream_0_drop_3/dropout/MulС
stream_0_drop_3/dropout/ShapeShape#stream_0_maxpool_3/Squeeze:output:0*
T0*
_output_shapes
:2
stream_0_drop_3/dropout/ShapeГ
4stream_0_drop_3/dropout/random_uniform/RandomUniformRandomUniform&stream_0_drop_3/dropout/Shape:output:0*
T0*+
_output_shapes
:€€€€€€€€€@*
dtype0*
seedЈ*
seed2Ї26
4stream_0_drop_3/dropout/random_uniform/RandomUniformХ
&stream_0_drop_3/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А>2(
&stream_0_drop_3/dropout/GreaterEqual/yВ
$stream_0_drop_3/dropout/GreaterEqualGreaterEqual=stream_0_drop_3/dropout/random_uniform/RandomUniform:output:0/stream_0_drop_3/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:€€€€€€€€€@2&
$stream_0_drop_3/dropout/GreaterEqual≥
stream_0_drop_3/dropout/CastCast(stream_0_drop_3/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:€€€€€€€€€@2
stream_0_drop_3/dropout/CastЊ
stream_0_drop_3/dropout/Mul_1Mulstream_0_drop_3/dropout/Mul:z:0 stream_0_drop_3/dropout/Cast:y:0*
T0*+
_output_shapes
:€€€€€€€€€@2
stream_0_drop_3/dropout/Mul_1§
/global_average_pooling1d/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :21
/global_average_pooling1d/Mean/reduction_indices’
global_average_pooling1d/MeanMean!stream_0_drop_3/dropout/Mul_1:z:08global_average_pooling1d/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
global_average_pooling1d/Mean•
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02
dense_1/MatMul/ReadVariableOpЂ
dense_1/MatMulMatMul&global_average_pooling1d/Mean:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2
dense_1/MatMul§
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02 
dense_1/BiasAdd/ReadVariableOp°
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2
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

:@*
	keep_dims(2$
"batch_normalization_3/moments/meanЊ
*batch_normalization_3/moments/StopGradientStopGradient+batch_normalization_3/moments/mean:output:0*
T0*
_output_shapes

:@2,
*batch_normalization_3/moments/StopGradientш
/batch_normalization_3/moments/SquaredDifferenceSquaredDifferencedense_1/BiasAdd:output:03batch_normalization_3/moments/StopGradient:output:0*
T0*'
_output_shapes
:€€€€€€€€€@21
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

:@*
	keep_dims(2(
&batch_normalization_3/moments/variance¬
%batch_normalization_3/moments/SqueezeSqueeze+batch_normalization_3/moments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 2'
%batch_normalization_3/moments/Squeeze 
'batch_normalization_3/moments/Squeeze_1Squeeze/batch_normalization_3/moments/variance:output:0*
T0*
_output_shapes
:@*
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
:@*
dtype026
4batch_normalization_3/AssignMovingAvg/ReadVariableOpр
)batch_normalization_3/AssignMovingAvg/subSub<batch_normalization_3/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_3/moments/Squeeze:output:0*
T0*
_output_shapes
:@2+
)batch_normalization_3/AssignMovingAvg/subз
)batch_normalization_3/AssignMovingAvg/mulMul-batch_normalization_3/AssignMovingAvg/sub:z:04batch_normalization_3/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:@2+
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
:@*
dtype028
6batch_normalization_3/AssignMovingAvg_1/ReadVariableOpш
+batch_normalization_3/AssignMovingAvg_1/subSub>batch_normalization_3/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_3/moments/Squeeze_1:output:0*
T0*
_output_shapes
:@2-
+batch_normalization_3/AssignMovingAvg_1/subп
+batch_normalization_3/AssignMovingAvg_1/mulMul/batch_normalization_3/AssignMovingAvg_1/sub:z:06batch_normalization_3/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:@2-
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
:@2%
#batch_normalization_3/batchnorm/add•
%batch_normalization_3/batchnorm/RsqrtRsqrt'batch_normalization_3/batchnorm/add:z:0*
T0*
_output_shapes
:@2'
%batch_normalization_3/batchnorm/Rsqrtа
2batch_normalization_3/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_3_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype024
2batch_normalization_3/batchnorm/mul/ReadVariableOpЁ
#batch_normalization_3/batchnorm/mulMul)batch_normalization_3/batchnorm/Rsqrt:y:0:batch_normalization_3/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2%
#batch_normalization_3/batchnorm/mul 
%batch_normalization_3/batchnorm/mul_1Muldense_1/BiasAdd:output:0'batch_normalization_3/batchnorm/mul:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2'
%batch_normalization_3/batchnorm/mul_1”
%batch_normalization_3/batchnorm/mul_2Mul.batch_normalization_3/moments/Squeeze:output:0'batch_normalization_3/batchnorm/mul:z:0*
T0*
_output_shapes
:@2'
%batch_normalization_3/batchnorm/mul_2‘
.batch_normalization_3/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_3_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype020
.batch_normalization_3/batchnorm/ReadVariableOpў
#batch_normalization_3/batchnorm/subSub6batch_normalization_3/batchnorm/ReadVariableOp:value:0)batch_normalization_3/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2%
#batch_normalization_3/batchnorm/subЁ
%batch_normalization_3/batchnorm/add_1AddV2)batch_normalization_3/batchnorm/mul_1:z:0'batch_normalization_3/batchnorm/sub:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2'
%batch_normalization_3/batchnorm/add_1Щ
(stream_0_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(stream_0_conv_1/kernel/Regularizer/Constо
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp;stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype027
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp√
&stream_0_conv_1/kernel/Regularizer/AbsAbs=stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_0_conv_1/kernel/Regularizer/Abs≠
*stream_0_conv_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_0_conv_1/kernel/Regularizer/Const_1ў
&stream_0_conv_1/kernel/Regularizer/SumSum*stream_0_conv_1/kernel/Regularizer/Abs:y:03stream_0_conv_1/kernel/Regularizer/Const_1:output:0*
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
&stream_0_conv_1/kernel/Regularizer/mulў
&stream_0_conv_1/kernel/Regularizer/addAddV21stream_0_conv_1/kernel/Regularizer/Const:output:0*stream_0_conv_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/addф
8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02:
8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOpѕ
)stream_0_conv_1/kernel/Regularizer/SquareSquare@stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2+
)stream_0_conv_1/kernel/Regularizer/Square≠
*stream_0_conv_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_0_conv_1/kernel/Regularizer/Const_2а
(stream_0_conv_1/kernel/Regularizer/Sum_1Sum-stream_0_conv_1/kernel/Regularizer/Square:y:03stream_0_conv_1/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2*
(stream_0_conv_1/kernel/Regularizer/Sum_1Э
*stream_0_conv_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2,
*stream_0_conv_1/kernel/Regularizer/mul_1/xд
(stream_0_conv_1/kernel/Regularizer/mul_1Mul3stream_0_conv_1/kernel/Regularizer/mul_1/x:output:01stream_0_conv_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2*
(stream_0_conv_1/kernel/Regularizer/mul_1Ў
(stream_0_conv_1/kernel/Regularizer/add_1AddV2*stream_0_conv_1/kernel/Regularizer/add:z:0,stream_0_conv_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2*
(stream_0_conv_1/kernel/Regularizer/add_1Щ
(stream_0_conv_2/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(stream_0_conv_2/kernel/Regularizer/Constо
5stream_0_conv_2/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp;stream_0_conv_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype027
5stream_0_conv_2/kernel/Regularizer/Abs/ReadVariableOp√
&stream_0_conv_2/kernel/Regularizer/AbsAbs=stream_0_conv_2/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@@2(
&stream_0_conv_2/kernel/Regularizer/Abs≠
*stream_0_conv_2/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_0_conv_2/kernel/Regularizer/Const_1ў
&stream_0_conv_2/kernel/Regularizer/SumSum*stream_0_conv_2/kernel/Regularizer/Abs:y:03stream_0_conv_2/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_2/kernel/Regularizer/SumЩ
(stream_0_conv_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2*
(stream_0_conv_2/kernel/Regularizer/mul/x№
&stream_0_conv_2/kernel/Regularizer/mulMul1stream_0_conv_2/kernel/Regularizer/mul/x:output:0/stream_0_conv_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_2/kernel/Regularizer/mulў
&stream_0_conv_2/kernel/Regularizer/addAddV21stream_0_conv_2/kernel/Regularizer/Const:output:0*stream_0_conv_2/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&stream_0_conv_2/kernel/Regularizer/addф
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;stream_0_conv_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype02:
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOpѕ
)stream_0_conv_2/kernel/Regularizer/SquareSquare@stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@@2+
)stream_0_conv_2/kernel/Regularizer/Square≠
*stream_0_conv_2/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_0_conv_2/kernel/Regularizer/Const_2а
(stream_0_conv_2/kernel/Regularizer/Sum_1Sum-stream_0_conv_2/kernel/Regularizer/Square:y:03stream_0_conv_2/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2*
(stream_0_conv_2/kernel/Regularizer/Sum_1Э
*stream_0_conv_2/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2,
*stream_0_conv_2/kernel/Regularizer/mul_1/xд
(stream_0_conv_2/kernel/Regularizer/mul_1Mul3stream_0_conv_2/kernel/Regularizer/mul_1/x:output:01stream_0_conv_2/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2*
(stream_0_conv_2/kernel/Regularizer/mul_1Ў
(stream_0_conv_2/kernel/Regularizer/add_1AddV2*stream_0_conv_2/kernel/Regularizer/add:z:0,stream_0_conv_2/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2*
(stream_0_conv_2/kernel/Regularizer/add_1Щ
(stream_0_conv_3/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(stream_0_conv_3/kernel/Regularizer/Constо
5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp;stream_0_conv_3_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype027
5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp√
&stream_0_conv_3/kernel/Regularizer/AbsAbs=stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@@2(
&stream_0_conv_3/kernel/Regularizer/Abs≠
*stream_0_conv_3/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_0_conv_3/kernel/Regularizer/Const_1ў
&stream_0_conv_3/kernel/Regularizer/SumSum*stream_0_conv_3/kernel/Regularizer/Abs:y:03stream_0_conv_3/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_3/kernel/Regularizer/SumЩ
(stream_0_conv_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2*
(stream_0_conv_3/kernel/Regularizer/mul/x№
&stream_0_conv_3/kernel/Regularizer/mulMul1stream_0_conv_3/kernel/Regularizer/mul/x:output:0/stream_0_conv_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_3/kernel/Regularizer/mulў
&stream_0_conv_3/kernel/Regularizer/addAddV21stream_0_conv_3/kernel/Regularizer/Const:output:0*stream_0_conv_3/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&stream_0_conv_3/kernel/Regularizer/addф
8stream_0_conv_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;stream_0_conv_3_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype02:
8stream_0_conv_3/kernel/Regularizer/Square/ReadVariableOpѕ
)stream_0_conv_3/kernel/Regularizer/SquareSquare@stream_0_conv_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@@2+
)stream_0_conv_3/kernel/Regularizer/Square≠
*stream_0_conv_3/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_0_conv_3/kernel/Regularizer/Const_2а
(stream_0_conv_3/kernel/Regularizer/Sum_1Sum-stream_0_conv_3/kernel/Regularizer/Square:y:03stream_0_conv_3/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2*
(stream_0_conv_3/kernel/Regularizer/Sum_1Э
*stream_0_conv_3/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2,
*stream_0_conv_3/kernel/Regularizer/mul_1/xд
(stream_0_conv_3/kernel/Regularizer/mul_1Mul3stream_0_conv_3/kernel/Regularizer/mul_1/x:output:01stream_0_conv_3/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2*
(stream_0_conv_3/kernel/Regularizer/mul_1Ў
(stream_0_conv_3/kernel/Regularizer/add_1AddV2*stream_0_conv_3/kernel/Regularizer/add:z:0,stream_0_conv_3/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2*
(stream_0_conv_3/kernel/Regularizer/add_1Й
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dense_1/kernel/Regularizer/Const≈
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOpІ
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:@@2 
dense_1/kernel/Regularizer/AbsЩ
"dense_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_1/kernel/Regularizer/Const_1є
dense_1/kernel/Regularizer/SumSum"dense_1/kernel/Regularizer/Abs:y:0+dense_1/kernel/Regularizer/Const_1:output:0*
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
dense_1/kernel/Regularizer/mulє
dense_1/kernel/Regularizer/addAddV2)dense_1/kernel/Regularizer/Const:output:0"dense_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/addЋ
0dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype022
0dense_1/kernel/Regularizer/Square/ReadVariableOp≥
!dense_1/kernel/Regularizer/SquareSquare8dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@@2#
!dense_1/kernel/Regularizer/SquareЩ
"dense_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_1/kernel/Regularizer/Const_2ј
 dense_1/kernel/Regularizer/Sum_1Sum%dense_1/kernel/Regularizer/Square:y:0+dense_1/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2"
 dense_1/kernel/Regularizer/Sum_1Н
"dense_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2$
"dense_1/kernel/Regularizer/mul_1/xƒ
 dense_1/kernel/Regularizer/mul_1Mul+dense_1/kernel/Regularizer/mul_1/x:output:0)dense_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2"
 dense_1/kernel/Regularizer/mul_1Є
 dense_1/kernel/Regularizer/add_1AddV2"dense_1/kernel/Regularizer/add:z:0$dense_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2"
 dense_1/kernel/Regularizer/add_1Д
IdentityIdentity)batch_normalization_3/batchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@2

Identityщ
NoOpNoOp$^batch_normalization/AssignMovingAvg3^batch_normalization/AssignMovingAvg/ReadVariableOp&^batch_normalization/AssignMovingAvg_15^batch_normalization/AssignMovingAvg_1/ReadVariableOp-^batch_normalization/batchnorm/ReadVariableOp1^batch_normalization/batchnorm/mul/ReadVariableOp&^batch_normalization_1/AssignMovingAvg5^batch_normalization_1/AssignMovingAvg/ReadVariableOp(^batch_normalization_1/AssignMovingAvg_17^batch_normalization_1/AssignMovingAvg_1/ReadVariableOp/^batch_normalization_1/batchnorm/ReadVariableOp3^batch_normalization_1/batchnorm/mul/ReadVariableOp&^batch_normalization_2/AssignMovingAvg5^batch_normalization_2/AssignMovingAvg/ReadVariableOp(^batch_normalization_2/AssignMovingAvg_17^batch_normalization_2/AssignMovingAvg_1/ReadVariableOp/^batch_normalization_2/batchnorm/ReadVariableOp3^batch_normalization_2/batchnorm/mul/ReadVariableOp&^batch_normalization_3/AssignMovingAvg5^batch_normalization_3/AssignMovingAvg/ReadVariableOp(^batch_normalization_3/AssignMovingAvg_17^batch_normalization_3/AssignMovingAvg_1/ReadVariableOp/^batch_normalization_3/batchnorm/ReadVariableOp3^batch_normalization_3/batchnorm/mul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp.^dense_1/kernel/Regularizer/Abs/ReadVariableOp1^dense_1/kernel/Regularizer/Square/ReadVariableOp'^stream_0_conv_1/BiasAdd/ReadVariableOp3^stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp9^stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOp'^stream_0_conv_2/BiasAdd/ReadVariableOp3^stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOp6^stream_0_conv_2/kernel/Regularizer/Abs/ReadVariableOp9^stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp'^stream_0_conv_3/BiasAdd/ReadVariableOp3^stream_0_conv_3/conv1d/ExpandDims_1/ReadVariableOp6^stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp9^stream_0_conv_3/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:€€€€€€€€€}: : : : : : : : : : : : : : : : : : : : : : : : 2J
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
-dense_1/kernel/Regularizer/Abs/ReadVariableOp-dense_1/kernel/Regularizer/Abs/ReadVariableOp2d
0dense_1/kernel/Regularizer/Square/ReadVariableOp0dense_1/kernel/Regularizer/Square/ReadVariableOp2P
&stream_0_conv_1/BiasAdd/ReadVariableOp&stream_0_conv_1/BiasAdd/ReadVariableOp2h
2stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp2stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp2n
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp2t
8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOp8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOp2P
&stream_0_conv_2/BiasAdd/ReadVariableOp&stream_0_conv_2/BiasAdd/ReadVariableOp2h
2stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOp2stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOp2n
5stream_0_conv_2/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_2/kernel/Regularizer/Abs/ReadVariableOp2t
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp2P
&stream_0_conv_3/BiasAdd/ReadVariableOp&stream_0_conv_3/BiasAdd/ReadVariableOp2h
2stream_0_conv_3/conv1d/ExpandDims_1/ReadVariableOp2stream_0_conv_3/conv1d/ExpandDims_1/ReadVariableOp2n
5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp2t
8stream_0_conv_3/kernel/Regularizer/Square/ReadVariableOp8stream_0_conv_3/kernel/Regularizer/Square/ReadVariableOp:U Q
+
_output_shapes
:€€€€€€€€€}
"
_user_specified_name
inputs/0
Й
j
L__inference_stream_0_drop_3_layer_call_and_return_conditional_losses_3293136

inputs

identity_1^
IdentityIdentityinputs*
T0*+
_output_shapes
:€€€€€€€€€@2

Identitym

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:€€€€€€€€€@2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€@:S O
+
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
у,
О
L__inference_stream_0_conv_2_layer_call_and_return_conditional_losses_3292992

inputsA
+conv1d_expanddims_1_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identityИҐBiasAdd/ReadVariableOpҐ"conv1d/ExpandDims_1/ReadVariableOpҐ5stream_0_conv_2/kernel/Regularizer/Abs/ReadVariableOpҐ8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOpy
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
:€€€€€€€€€>@2
conv1d/ExpandDimsЄ
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
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
:@@2
conv1d/ExpandDims_1ґ
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:€€€€€€€€€>@*
paddingSAME*
strides
2
conv1dТ
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:€€€€€€€€€>@*
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
:€€€€€€€€€>@2	
BiasAddЩ
(stream_0_conv_2/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(stream_0_conv_2/kernel/Regularizer/Constё
5stream_0_conv_2/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype027
5stream_0_conv_2/kernel/Regularizer/Abs/ReadVariableOp√
&stream_0_conv_2/kernel/Regularizer/AbsAbs=stream_0_conv_2/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@@2(
&stream_0_conv_2/kernel/Regularizer/Abs≠
*stream_0_conv_2/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_0_conv_2/kernel/Regularizer/Const_1ў
&stream_0_conv_2/kernel/Regularizer/SumSum*stream_0_conv_2/kernel/Regularizer/Abs:y:03stream_0_conv_2/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_2/kernel/Regularizer/SumЩ
(stream_0_conv_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2*
(stream_0_conv_2/kernel/Regularizer/mul/x№
&stream_0_conv_2/kernel/Regularizer/mulMul1stream_0_conv_2/kernel/Regularizer/mul/x:output:0/stream_0_conv_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_2/kernel/Regularizer/mulў
&stream_0_conv_2/kernel/Regularizer/addAddV21stream_0_conv_2/kernel/Regularizer/Const:output:0*stream_0_conv_2/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&stream_0_conv_2/kernel/Regularizer/addд
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype02:
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOpѕ
)stream_0_conv_2/kernel/Regularizer/SquareSquare@stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@@2+
)stream_0_conv_2/kernel/Regularizer/Square≠
*stream_0_conv_2/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_0_conv_2/kernel/Regularizer/Const_2а
(stream_0_conv_2/kernel/Regularizer/Sum_1Sum-stream_0_conv_2/kernel/Regularizer/Square:y:03stream_0_conv_2/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2*
(stream_0_conv_2/kernel/Regularizer/Sum_1Э
*stream_0_conv_2/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2,
*stream_0_conv_2/kernel/Regularizer/mul_1/xд
(stream_0_conv_2/kernel/Regularizer/mul_1Mul3stream_0_conv_2/kernel/Regularizer/mul_1/x:output:01stream_0_conv_2/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2*
(stream_0_conv_2/kernel/Regularizer/mul_1Ў
(stream_0_conv_2/kernel/Regularizer/add_1AddV2*stream_0_conv_2/kernel/Regularizer/add:z:0,stream_0_conv_2/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2*
(stream_0_conv_2/kernel/Regularizer/add_1o
IdentityIdentityBiasAdd:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€>@2

Identity€
NoOpNoOp^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp6^stream_0_conv_2/kernel/Regularizer/Abs/ReadVariableOp9^stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€>@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp2n
5stream_0_conv_2/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_2/kernel/Regularizer/Abs/ReadVariableOp2t
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp:S O
+
_output_shapes
:€€€€€€€€€>@
 
_user_specified_nameinputs
И
ѓ
P__inference_batch_normalization_layer_call_and_return_conditional_losses_3292929

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
¬б
±
D__inference_model_1_layer_call_and_return_conditional_losses_3295882

inputs[
Ebasemodel_stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource:@G
9basemodel_stream_0_conv_1_biasadd_readvariableop_resource:@M
?basemodel_batch_normalization_batchnorm_readvariableop_resource:@Q
Cbasemodel_batch_normalization_batchnorm_mul_readvariableop_resource:@O
Abasemodel_batch_normalization_batchnorm_readvariableop_1_resource:@O
Abasemodel_batch_normalization_batchnorm_readvariableop_2_resource:@[
Ebasemodel_stream_0_conv_2_conv1d_expanddims_1_readvariableop_resource:@@G
9basemodel_stream_0_conv_2_biasadd_readvariableop_resource:@O
Abasemodel_batch_normalization_1_batchnorm_readvariableop_resource:@S
Ebasemodel_batch_normalization_1_batchnorm_mul_readvariableop_resource:@Q
Cbasemodel_batch_normalization_1_batchnorm_readvariableop_1_resource:@Q
Cbasemodel_batch_normalization_1_batchnorm_readvariableop_2_resource:@[
Ebasemodel_stream_0_conv_3_conv1d_expanddims_1_readvariableop_resource:@@G
9basemodel_stream_0_conv_3_biasadd_readvariableop_resource:@O
Abasemodel_batch_normalization_2_batchnorm_readvariableop_resource:@S
Ebasemodel_batch_normalization_2_batchnorm_mul_readvariableop_resource:@Q
Cbasemodel_batch_normalization_2_batchnorm_readvariableop_1_resource:@Q
Cbasemodel_batch_normalization_2_batchnorm_readvariableop_2_resource:@B
0basemodel_dense_1_matmul_readvariableop_resource:@@?
1basemodel_dense_1_biasadd_readvariableop_resource:@O
Abasemodel_batch_normalization_3_batchnorm_readvariableop_resource:@S
Ebasemodel_batch_normalization_3_batchnorm_mul_readvariableop_resource:@Q
Cbasemodel_batch_normalization_3_batchnorm_readvariableop_1_resource:@Q
Cbasemodel_batch_normalization_3_batchnorm_readvariableop_2_resource:@
identityИҐ6basemodel/batch_normalization/batchnorm/ReadVariableOpҐ8basemodel/batch_normalization/batchnorm/ReadVariableOp_1Ґ8basemodel/batch_normalization/batchnorm/ReadVariableOp_2Ґ:basemodel/batch_normalization/batchnorm/mul/ReadVariableOpҐ8basemodel/batch_normalization_1/batchnorm/ReadVariableOpҐ:basemodel/batch_normalization_1/batchnorm/ReadVariableOp_1Ґ:basemodel/batch_normalization_1/batchnorm/ReadVariableOp_2Ґ<basemodel/batch_normalization_1/batchnorm/mul/ReadVariableOpҐ8basemodel/batch_normalization_2/batchnorm/ReadVariableOpҐ:basemodel/batch_normalization_2/batchnorm/ReadVariableOp_1Ґ:basemodel/batch_normalization_2/batchnorm/ReadVariableOp_2Ґ<basemodel/batch_normalization_2/batchnorm/mul/ReadVariableOpҐ8basemodel/batch_normalization_3/batchnorm/ReadVariableOpҐ:basemodel/batch_normalization_3/batchnorm/ReadVariableOp_1Ґ:basemodel/batch_normalization_3/batchnorm/ReadVariableOp_2Ґ<basemodel/batch_normalization_3/batchnorm/mul/ReadVariableOpҐ(basemodel/dense_1/BiasAdd/ReadVariableOpҐ'basemodel/dense_1/MatMul/ReadVariableOpҐ0basemodel/stream_0_conv_1/BiasAdd/ReadVariableOpҐ<basemodel/stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOpҐ0basemodel/stream_0_conv_2/BiasAdd/ReadVariableOpҐ<basemodel/stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOpҐ0basemodel/stream_0_conv_3/BiasAdd/ReadVariableOpҐ<basemodel/stream_0_conv_3/conv1d/ExpandDims_1/ReadVariableOpҐ-dense_1/kernel/Regularizer/Abs/ReadVariableOpҐ0dense_1/kernel/Regularizer/Square/ReadVariableOpҐ5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpҐ8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOpҐ5stream_0_conv_2/kernel/Regularizer/Abs/ReadVariableOpҐ8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOpҐ5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOpҐ8stream_0_conv_3/kernel/Regularizer/Square/ReadVariableOpЪ
&basemodel/stream_0_input_drop/IdentityIdentityinputs*
T0*+
_output_shapes
:€€€€€€€€€}2(
&basemodel/stream_0_input_drop/Identity≠
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
!basemodel/stream_0_conv_1/BiasAddм
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
-basemodel/batch_normalization/batchnorm/add_1І
basemodel/activation/TanhTanh1basemodel/batch_normalization/batchnorm/add_1:z:0*
T0*+
_output_shapes
:€€€€€€€€€}@2
basemodel/activation/TanhЬ
+basemodel/stream_0_maxpool_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2-
+basemodel/stream_0_maxpool_1/ExpandDims/dimп
'basemodel/stream_0_maxpool_1/ExpandDims
ExpandDimsbasemodel/activation/Tanh:y:04basemodel/stream_0_maxpool_1/ExpandDims/dim:output:0*
T0*/
_output_shapes
:€€€€€€€€€}@2)
'basemodel/stream_0_maxpool_1/ExpandDimsц
$basemodel/stream_0_maxpool_1/MaxPoolMaxPool0basemodel/stream_0_maxpool_1/ExpandDims:output:0*/
_output_shapes
:€€€€€€€€€>@*
ksize
*
paddingVALID*
strides
2&
$basemodel/stream_0_maxpool_1/MaxPool”
$basemodel/stream_0_maxpool_1/SqueezeSqueeze-basemodel/stream_0_maxpool_1/MaxPool:output:0*
T0*+
_output_shapes
:€€€€€€€€€>@*
squeeze_dims
2&
$basemodel/stream_0_maxpool_1/Squeezeє
"basemodel/stream_0_drop_1/IdentityIdentity-basemodel/stream_0_maxpool_1/Squeeze:output:0*
T0*+
_output_shapes
:€€€€€€€€€>@2$
"basemodel/stream_0_drop_1/Identity≠
/basemodel/stream_0_conv_2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€21
/basemodel/stream_0_conv_2/conv1d/ExpandDims/dimЙ
+basemodel/stream_0_conv_2/conv1d/ExpandDims
ExpandDims+basemodel/stream_0_drop_1/Identity:output:08basemodel/stream_0_conv_2/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:€€€€€€€€€>@2-
+basemodel/stream_0_conv_2/conv1d/ExpandDimsЖ
<basemodel/stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpEbasemodel_stream_0_conv_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype02>
<basemodel/stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOp®
1basemodel/stream_0_conv_2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 23
1basemodel/stream_0_conv_2/conv1d/ExpandDims_1/dimЯ
-basemodel/stream_0_conv_2/conv1d/ExpandDims_1
ExpandDimsDbasemodel/stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOp:value:0:basemodel/stream_0_conv_2/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@2/
-basemodel/stream_0_conv_2/conv1d/ExpandDims_1Ю
 basemodel/stream_0_conv_2/conv1dConv2D4basemodel/stream_0_conv_2/conv1d/ExpandDims:output:06basemodel/stream_0_conv_2/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:€€€€€€€€€>@*
paddingSAME*
strides
2"
 basemodel/stream_0_conv_2/conv1dа
(basemodel/stream_0_conv_2/conv1d/SqueezeSqueeze)basemodel/stream_0_conv_2/conv1d:output:0*
T0*+
_output_shapes
:€€€€€€€€€>@*
squeeze_dims

э€€€€€€€€2*
(basemodel/stream_0_conv_2/conv1d/SqueezeЏ
0basemodel/stream_0_conv_2/BiasAdd/ReadVariableOpReadVariableOp9basemodel_stream_0_conv_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype022
0basemodel/stream_0_conv_2/BiasAdd/ReadVariableOpф
!basemodel/stream_0_conv_2/BiasAddBiasAdd1basemodel/stream_0_conv_2/conv1d/Squeeze:output:08basemodel/stream_0_conv_2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€>@2#
!basemodel/stream_0_conv_2/BiasAddт
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
/basemodel/batch_normalization_1/batchnorm/mul_1Mul*basemodel/stream_0_conv_2/BiasAdd:output:01basemodel/batch_normalization_1/batchnorm/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€>@21
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
:€€€€€€€€€>@21
/basemodel/batch_normalization_1/batchnorm/add_1≠
basemodel/activation_1/TanhTanh3basemodel/batch_normalization_1/batchnorm/add_1:z:0*
T0*+
_output_shapes
:€€€€€€€€€>@2
basemodel/activation_1/TanhЬ
+basemodel/stream_0_maxpool_2/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2-
+basemodel/stream_0_maxpool_2/ExpandDims/dimс
'basemodel/stream_0_maxpool_2/ExpandDims
ExpandDimsbasemodel/activation_1/Tanh:y:04basemodel/stream_0_maxpool_2/ExpandDims/dim:output:0*
T0*/
_output_shapes
:€€€€€€€€€>@2)
'basemodel/stream_0_maxpool_2/ExpandDimsц
$basemodel/stream_0_maxpool_2/MaxPoolMaxPool0basemodel/stream_0_maxpool_2/ExpandDims:output:0*/
_output_shapes
:€€€€€€€€€@*
ksize
*
paddingVALID*
strides
2&
$basemodel/stream_0_maxpool_2/MaxPool”
$basemodel/stream_0_maxpool_2/SqueezeSqueeze-basemodel/stream_0_maxpool_2/MaxPool:output:0*
T0*+
_output_shapes
:€€€€€€€€€@*
squeeze_dims
2&
$basemodel/stream_0_maxpool_2/Squeezeє
"basemodel/stream_0_drop_2/IdentityIdentity-basemodel/stream_0_maxpool_2/Squeeze:output:0*
T0*+
_output_shapes
:€€€€€€€€€@2$
"basemodel/stream_0_drop_2/Identity≠
/basemodel/stream_0_conv_3/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€21
/basemodel/stream_0_conv_3/conv1d/ExpandDims/dimЙ
+basemodel/stream_0_conv_3/conv1d/ExpandDims
ExpandDims+basemodel/stream_0_drop_2/Identity:output:08basemodel/stream_0_conv_3/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:€€€€€€€€€@2-
+basemodel/stream_0_conv_3/conv1d/ExpandDimsЖ
<basemodel/stream_0_conv_3/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpEbasemodel_stream_0_conv_3_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype02>
<basemodel/stream_0_conv_3/conv1d/ExpandDims_1/ReadVariableOp®
1basemodel/stream_0_conv_3/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 23
1basemodel/stream_0_conv_3/conv1d/ExpandDims_1/dimЯ
-basemodel/stream_0_conv_3/conv1d/ExpandDims_1
ExpandDimsDbasemodel/stream_0_conv_3/conv1d/ExpandDims_1/ReadVariableOp:value:0:basemodel/stream_0_conv_3/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@2/
-basemodel/stream_0_conv_3/conv1d/ExpandDims_1Ю
 basemodel/stream_0_conv_3/conv1dConv2D4basemodel/stream_0_conv_3/conv1d/ExpandDims:output:06basemodel/stream_0_conv_3/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:€€€€€€€€€@*
paddingSAME*
strides
2"
 basemodel/stream_0_conv_3/conv1dа
(basemodel/stream_0_conv_3/conv1d/SqueezeSqueeze)basemodel/stream_0_conv_3/conv1d:output:0*
T0*+
_output_shapes
:€€€€€€€€€@*
squeeze_dims

э€€€€€€€€2*
(basemodel/stream_0_conv_3/conv1d/SqueezeЏ
0basemodel/stream_0_conv_3/BiasAdd/ReadVariableOpReadVariableOp9basemodel_stream_0_conv_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype022
0basemodel/stream_0_conv_3/BiasAdd/ReadVariableOpф
!basemodel/stream_0_conv_3/BiasAddBiasAdd1basemodel/stream_0_conv_3/conv1d/Squeeze:output:08basemodel/stream_0_conv_3/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€@2#
!basemodel/stream_0_conv_3/BiasAddт
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
/basemodel/batch_normalization_2/batchnorm/mul_1Mul*basemodel/stream_0_conv_3/BiasAdd:output:01basemodel/batch_normalization_2/batchnorm/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€@21
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
:€€€€€€€€€@21
/basemodel/batch_normalization_2/batchnorm/add_1≠
basemodel/activation_2/TanhTanh3basemodel/batch_normalization_2/batchnorm/add_1:z:0*
T0*+
_output_shapes
:€€€€€€€€€@2
basemodel/activation_2/TanhЬ
+basemodel/stream_0_maxpool_3/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2-
+basemodel/stream_0_maxpool_3/ExpandDims/dimс
'basemodel/stream_0_maxpool_3/ExpandDims
ExpandDimsbasemodel/activation_2/Tanh:y:04basemodel/stream_0_maxpool_3/ExpandDims/dim:output:0*
T0*/
_output_shapes
:€€€€€€€€€@2)
'basemodel/stream_0_maxpool_3/ExpandDimsц
$basemodel/stream_0_maxpool_3/MaxPoolMaxPool0basemodel/stream_0_maxpool_3/ExpandDims:output:0*/
_output_shapes
:€€€€€€€€€@*
ksize
*
paddingVALID*
strides
2&
$basemodel/stream_0_maxpool_3/MaxPool”
$basemodel/stream_0_maxpool_3/SqueezeSqueeze-basemodel/stream_0_maxpool_3/MaxPool:output:0*
T0*+
_output_shapes
:€€€€€€€€€@*
squeeze_dims
2&
$basemodel/stream_0_maxpool_3/Squeezeє
"basemodel/stream_0_drop_3/IdentityIdentity-basemodel/stream_0_maxpool_3/Squeeze:output:0*
T0*+
_output_shapes
:€€€€€€€€€@2$
"basemodel/stream_0_drop_3/IdentityЄ
9basemodel/global_average_pooling1d/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2;
9basemodel/global_average_pooling1d/Mean/reduction_indicesэ
'basemodel/global_average_pooling1d/MeanMean+basemodel/stream_0_drop_3/Identity:output:0Bbasemodel/global_average_pooling1d/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2)
'basemodel/global_average_pooling1d/MeanЄ
"basemodel/dense_1_dropout/IdentityIdentity0basemodel/global_average_pooling1d/Mean:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2$
"basemodel/dense_1_dropout/Identity√
'basemodel/dense_1/MatMul/ReadVariableOpReadVariableOp0basemodel_dense_1_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02)
'basemodel/dense_1/MatMul/ReadVariableOpќ
basemodel/dense_1/MatMulMatMul+basemodel/dense_1_dropout/Identity:output:0/basemodel/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2
basemodel/dense_1/MatMul¬
(basemodel/dense_1/BiasAdd/ReadVariableOpReadVariableOp1basemodel_dense_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02*
(basemodel/dense_1/BiasAdd/ReadVariableOp…
basemodel/dense_1/BiasAddBiasAdd"basemodel/dense_1/MatMul:product:00basemodel/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2
basemodel/dense_1/BiasAddт
8basemodel/batch_normalization_3/batchnorm/ReadVariableOpReadVariableOpAbasemodel_batch_normalization_3_batchnorm_readvariableop_resource*
_output_shapes
:@*
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
:@2/
-basemodel/batch_normalization_3/batchnorm/add√
/basemodel/batch_normalization_3/batchnorm/RsqrtRsqrt1basemodel/batch_normalization_3/batchnorm/add:z:0*
T0*
_output_shapes
:@21
/basemodel/batch_normalization_3/batchnorm/Rsqrtю
<basemodel/batch_normalization_3/batchnorm/mul/ReadVariableOpReadVariableOpEbasemodel_batch_normalization_3_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02>
<basemodel/batch_normalization_3/batchnorm/mul/ReadVariableOpЕ
-basemodel/batch_normalization_3/batchnorm/mulMul3basemodel/batch_normalization_3/batchnorm/Rsqrt:y:0Dbasemodel/batch_normalization_3/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2/
-basemodel/batch_normalization_3/batchnorm/mulт
/basemodel/batch_normalization_3/batchnorm/mul_1Mul"basemodel/dense_1/BiasAdd:output:01basemodel/batch_normalization_3/batchnorm/mul:z:0*
T0*'
_output_shapes
:€€€€€€€€€@21
/basemodel/batch_normalization_3/batchnorm/mul_1ш
:basemodel/batch_normalization_3/batchnorm/ReadVariableOp_1ReadVariableOpCbasemodel_batch_normalization_3_batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype02<
:basemodel/batch_normalization_3/batchnorm/ReadVariableOp_1Е
/basemodel/batch_normalization_3/batchnorm/mul_2MulBbasemodel/batch_normalization_3/batchnorm/ReadVariableOp_1:value:01basemodel/batch_normalization_3/batchnorm/mul:z:0*
T0*
_output_shapes
:@21
/basemodel/batch_normalization_3/batchnorm/mul_2ш
:basemodel/batch_normalization_3/batchnorm/ReadVariableOp_2ReadVariableOpCbasemodel_batch_normalization_3_batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype02<
:basemodel/batch_normalization_3/batchnorm/ReadVariableOp_2Г
-basemodel/batch_normalization_3/batchnorm/subSubBbasemodel/batch_normalization_3/batchnorm/ReadVariableOp_2:value:03basemodel/batch_normalization_3/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2/
-basemodel/batch_normalization_3/batchnorm/subЕ
/basemodel/batch_normalization_3/batchnorm/add_1AddV23basemodel/batch_normalization_3/batchnorm/mul_1:z:01basemodel/batch_normalization_3/batchnorm/sub:z:0*
T0*'
_output_shapes
:€€€€€€€€€@21
/basemodel/batch_normalization_3/batchnorm/add_1Щ
(stream_0_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(stream_0_conv_1/kernel/Regularizer/Constш
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpEbasemodel_stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype027
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp√
&stream_0_conv_1/kernel/Regularizer/AbsAbs=stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_0_conv_1/kernel/Regularizer/Abs≠
*stream_0_conv_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_0_conv_1/kernel/Regularizer/Const_1ў
&stream_0_conv_1/kernel/Regularizer/SumSum*stream_0_conv_1/kernel/Regularizer/Abs:y:03stream_0_conv_1/kernel/Regularizer/Const_1:output:0*
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
&stream_0_conv_1/kernel/Regularizer/mulў
&stream_0_conv_1/kernel/Regularizer/addAddV21stream_0_conv_1/kernel/Regularizer/Const:output:0*stream_0_conv_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/addю
8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpEbasemodel_stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02:
8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOpѕ
)stream_0_conv_1/kernel/Regularizer/SquareSquare@stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2+
)stream_0_conv_1/kernel/Regularizer/Square≠
*stream_0_conv_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_0_conv_1/kernel/Regularizer/Const_2а
(stream_0_conv_1/kernel/Regularizer/Sum_1Sum-stream_0_conv_1/kernel/Regularizer/Square:y:03stream_0_conv_1/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2*
(stream_0_conv_1/kernel/Regularizer/Sum_1Э
*stream_0_conv_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2,
*stream_0_conv_1/kernel/Regularizer/mul_1/xд
(stream_0_conv_1/kernel/Regularizer/mul_1Mul3stream_0_conv_1/kernel/Regularizer/mul_1/x:output:01stream_0_conv_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2*
(stream_0_conv_1/kernel/Regularizer/mul_1Ў
(stream_0_conv_1/kernel/Regularizer/add_1AddV2*stream_0_conv_1/kernel/Regularizer/add:z:0,stream_0_conv_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2*
(stream_0_conv_1/kernel/Regularizer/add_1Щ
(stream_0_conv_2/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(stream_0_conv_2/kernel/Regularizer/Constш
5stream_0_conv_2/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpEbasemodel_stream_0_conv_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype027
5stream_0_conv_2/kernel/Regularizer/Abs/ReadVariableOp√
&stream_0_conv_2/kernel/Regularizer/AbsAbs=stream_0_conv_2/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@@2(
&stream_0_conv_2/kernel/Regularizer/Abs≠
*stream_0_conv_2/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_0_conv_2/kernel/Regularizer/Const_1ў
&stream_0_conv_2/kernel/Regularizer/SumSum*stream_0_conv_2/kernel/Regularizer/Abs:y:03stream_0_conv_2/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_2/kernel/Regularizer/SumЩ
(stream_0_conv_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2*
(stream_0_conv_2/kernel/Regularizer/mul/x№
&stream_0_conv_2/kernel/Regularizer/mulMul1stream_0_conv_2/kernel/Regularizer/mul/x:output:0/stream_0_conv_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_2/kernel/Regularizer/mulў
&stream_0_conv_2/kernel/Regularizer/addAddV21stream_0_conv_2/kernel/Regularizer/Const:output:0*stream_0_conv_2/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&stream_0_conv_2/kernel/Regularizer/addю
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpEbasemodel_stream_0_conv_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype02:
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOpѕ
)stream_0_conv_2/kernel/Regularizer/SquareSquare@stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@@2+
)stream_0_conv_2/kernel/Regularizer/Square≠
*stream_0_conv_2/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_0_conv_2/kernel/Regularizer/Const_2а
(stream_0_conv_2/kernel/Regularizer/Sum_1Sum-stream_0_conv_2/kernel/Regularizer/Square:y:03stream_0_conv_2/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2*
(stream_0_conv_2/kernel/Regularizer/Sum_1Э
*stream_0_conv_2/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2,
*stream_0_conv_2/kernel/Regularizer/mul_1/xд
(stream_0_conv_2/kernel/Regularizer/mul_1Mul3stream_0_conv_2/kernel/Regularizer/mul_1/x:output:01stream_0_conv_2/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2*
(stream_0_conv_2/kernel/Regularizer/mul_1Ў
(stream_0_conv_2/kernel/Regularizer/add_1AddV2*stream_0_conv_2/kernel/Regularizer/add:z:0,stream_0_conv_2/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2*
(stream_0_conv_2/kernel/Regularizer/add_1Щ
(stream_0_conv_3/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(stream_0_conv_3/kernel/Regularizer/Constш
5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpEbasemodel_stream_0_conv_3_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype027
5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp√
&stream_0_conv_3/kernel/Regularizer/AbsAbs=stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@@2(
&stream_0_conv_3/kernel/Regularizer/Abs≠
*stream_0_conv_3/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_0_conv_3/kernel/Regularizer/Const_1ў
&stream_0_conv_3/kernel/Regularizer/SumSum*stream_0_conv_3/kernel/Regularizer/Abs:y:03stream_0_conv_3/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_3/kernel/Regularizer/SumЩ
(stream_0_conv_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2*
(stream_0_conv_3/kernel/Regularizer/mul/x№
&stream_0_conv_3/kernel/Regularizer/mulMul1stream_0_conv_3/kernel/Regularizer/mul/x:output:0/stream_0_conv_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_3/kernel/Regularizer/mulў
&stream_0_conv_3/kernel/Regularizer/addAddV21stream_0_conv_3/kernel/Regularizer/Const:output:0*stream_0_conv_3/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&stream_0_conv_3/kernel/Regularizer/addю
8stream_0_conv_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpEbasemodel_stream_0_conv_3_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype02:
8stream_0_conv_3/kernel/Regularizer/Square/ReadVariableOpѕ
)stream_0_conv_3/kernel/Regularizer/SquareSquare@stream_0_conv_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@@2+
)stream_0_conv_3/kernel/Regularizer/Square≠
*stream_0_conv_3/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_0_conv_3/kernel/Regularizer/Const_2а
(stream_0_conv_3/kernel/Regularizer/Sum_1Sum-stream_0_conv_3/kernel/Regularizer/Square:y:03stream_0_conv_3/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2*
(stream_0_conv_3/kernel/Regularizer/Sum_1Э
*stream_0_conv_3/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2,
*stream_0_conv_3/kernel/Regularizer/mul_1/xд
(stream_0_conv_3/kernel/Regularizer/mul_1Mul3stream_0_conv_3/kernel/Regularizer/mul_1/x:output:01stream_0_conv_3/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2*
(stream_0_conv_3/kernel/Regularizer/mul_1Ў
(stream_0_conv_3/kernel/Regularizer/add_1AddV2*stream_0_conv_3/kernel/Regularizer/add:z:0,stream_0_conv_3/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2*
(stream_0_conv_3/kernel/Regularizer/add_1Й
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dense_1/kernel/Regularizer/Constѕ
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp0basemodel_dense_1_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOpІ
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:@@2 
dense_1/kernel/Regularizer/AbsЩ
"dense_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_1/kernel/Regularizer/Const_1є
dense_1/kernel/Regularizer/SumSum"dense_1/kernel/Regularizer/Abs:y:0+dense_1/kernel/Regularizer/Const_1:output:0*
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
dense_1/kernel/Regularizer/mulє
dense_1/kernel/Regularizer/addAddV2)dense_1/kernel/Regularizer/Const:output:0"dense_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/add’
0dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp0basemodel_dense_1_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype022
0dense_1/kernel/Regularizer/Square/ReadVariableOp≥
!dense_1/kernel/Regularizer/SquareSquare8dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@@2#
!dense_1/kernel/Regularizer/SquareЩ
"dense_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_1/kernel/Regularizer/Const_2ј
 dense_1/kernel/Regularizer/Sum_1Sum%dense_1/kernel/Regularizer/Square:y:0+dense_1/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2"
 dense_1/kernel/Regularizer/Sum_1Н
"dense_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2$
"dense_1/kernel/Regularizer/mul_1/xƒ
 dense_1/kernel/Regularizer/mul_1Mul+dense_1/kernel/Regularizer/mul_1/x:output:0)dense_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2"
 dense_1/kernel/Regularizer/mul_1Є
 dense_1/kernel/Regularizer/add_1AddV2"dense_1/kernel/Regularizer/add:z:0$dense_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2"
 dense_1/kernel/Regularizer/add_1О
IdentityIdentity3basemodel/batch_normalization_3/batchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@2

Identityэ
NoOpNoOp7^basemodel/batch_normalization/batchnorm/ReadVariableOp9^basemodel/batch_normalization/batchnorm/ReadVariableOp_19^basemodel/batch_normalization/batchnorm/ReadVariableOp_2;^basemodel/batch_normalization/batchnorm/mul/ReadVariableOp9^basemodel/batch_normalization_1/batchnorm/ReadVariableOp;^basemodel/batch_normalization_1/batchnorm/ReadVariableOp_1;^basemodel/batch_normalization_1/batchnorm/ReadVariableOp_2=^basemodel/batch_normalization_1/batchnorm/mul/ReadVariableOp9^basemodel/batch_normalization_2/batchnorm/ReadVariableOp;^basemodel/batch_normalization_2/batchnorm/ReadVariableOp_1;^basemodel/batch_normalization_2/batchnorm/ReadVariableOp_2=^basemodel/batch_normalization_2/batchnorm/mul/ReadVariableOp9^basemodel/batch_normalization_3/batchnorm/ReadVariableOp;^basemodel/batch_normalization_3/batchnorm/ReadVariableOp_1;^basemodel/batch_normalization_3/batchnorm/ReadVariableOp_2=^basemodel/batch_normalization_3/batchnorm/mul/ReadVariableOp)^basemodel/dense_1/BiasAdd/ReadVariableOp(^basemodel/dense_1/MatMul/ReadVariableOp1^basemodel/stream_0_conv_1/BiasAdd/ReadVariableOp=^basemodel/stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp1^basemodel/stream_0_conv_2/BiasAdd/ReadVariableOp=^basemodel/stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOp1^basemodel/stream_0_conv_3/BiasAdd/ReadVariableOp=^basemodel/stream_0_conv_3/conv1d/ExpandDims_1/ReadVariableOp.^dense_1/kernel/Regularizer/Abs/ReadVariableOp1^dense_1/kernel/Regularizer/Square/ReadVariableOp6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp9^stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOp6^stream_0_conv_2/kernel/Regularizer/Abs/ReadVariableOp9^stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp6^stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp9^stream_0_conv_3/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:€€€€€€€€€}: : : : : : : : : : : : : : : : : : : : : : : : 2p
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
-dense_1/kernel/Regularizer/Abs/ReadVariableOp-dense_1/kernel/Regularizer/Abs/ReadVariableOp2d
0dense_1/kernel/Regularizer/Square/ReadVariableOp0dense_1/kernel/Regularizer/Square/ReadVariableOp2n
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp2t
8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOp8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOp2n
5stream_0_conv_2/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_2/kernel/Regularizer/Abs/ReadVariableOp2t
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp2n
5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp2t
8stream_0_conv_3/kernel/Regularizer/Square/ReadVariableOp8stream_0_conv_3/kernel/Regularizer/Square/ReadVariableOp:S O
+
_output_shapes
:€€€€€€€€€}
 
_user_specified_nameinputs
те
з
F__inference_basemodel_layer_call_and_return_conditional_losses_3296887

inputsQ
;stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource:@=
/stream_0_conv_1_biasadd_readvariableop_resource:@I
;batch_normalization_assignmovingavg_readvariableop_resource:@K
=batch_normalization_assignmovingavg_1_readvariableop_resource:@G
9batch_normalization_batchnorm_mul_readvariableop_resource:@C
5batch_normalization_batchnorm_readvariableop_resource:@Q
;stream_0_conv_2_conv1d_expanddims_1_readvariableop_resource:@@=
/stream_0_conv_2_biasadd_readvariableop_resource:@K
=batch_normalization_1_assignmovingavg_readvariableop_resource:@M
?batch_normalization_1_assignmovingavg_1_readvariableop_resource:@I
;batch_normalization_1_batchnorm_mul_readvariableop_resource:@E
7batch_normalization_1_batchnorm_readvariableop_resource:@Q
;stream_0_conv_3_conv1d_expanddims_1_readvariableop_resource:@@=
/stream_0_conv_3_biasadd_readvariableop_resource:@K
=batch_normalization_2_assignmovingavg_readvariableop_resource:@M
?batch_normalization_2_assignmovingavg_1_readvariableop_resource:@I
;batch_normalization_2_batchnorm_mul_readvariableop_resource:@E
7batch_normalization_2_batchnorm_readvariableop_resource:@8
&dense_1_matmul_readvariableop_resource:@@5
'dense_1_biasadd_readvariableop_resource:@K
=batch_normalization_3_assignmovingavg_readvariableop_resource:@M
?batch_normalization_3_assignmovingavg_1_readvariableop_resource:@I
;batch_normalization_3_batchnorm_mul_readvariableop_resource:@E
7batch_normalization_3_batchnorm_readvariableop_resource:@
identityИҐ#batch_normalization/AssignMovingAvgҐ2batch_normalization/AssignMovingAvg/ReadVariableOpҐ%batch_normalization/AssignMovingAvg_1Ґ4batch_normalization/AssignMovingAvg_1/ReadVariableOpҐ,batch_normalization/batchnorm/ReadVariableOpҐ0batch_normalization/batchnorm/mul/ReadVariableOpҐ%batch_normalization_1/AssignMovingAvgҐ4batch_normalization_1/AssignMovingAvg/ReadVariableOpҐ'batch_normalization_1/AssignMovingAvg_1Ґ6batch_normalization_1/AssignMovingAvg_1/ReadVariableOpҐ.batch_normalization_1/batchnorm/ReadVariableOpҐ2batch_normalization_1/batchnorm/mul/ReadVariableOpҐ%batch_normalization_2/AssignMovingAvgҐ4batch_normalization_2/AssignMovingAvg/ReadVariableOpҐ'batch_normalization_2/AssignMovingAvg_1Ґ6batch_normalization_2/AssignMovingAvg_1/ReadVariableOpҐ.batch_normalization_2/batchnorm/ReadVariableOpҐ2batch_normalization_2/batchnorm/mul/ReadVariableOpҐ%batch_normalization_3/AssignMovingAvgҐ4batch_normalization_3/AssignMovingAvg/ReadVariableOpҐ'batch_normalization_3/AssignMovingAvg_1Ґ6batch_normalization_3/AssignMovingAvg_1/ReadVariableOpҐ.batch_normalization_3/batchnorm/ReadVariableOpҐ2batch_normalization_3/batchnorm/mul/ReadVariableOpҐdense_1/BiasAdd/ReadVariableOpҐdense_1/MatMul/ReadVariableOpҐ-dense_1/kernel/Regularizer/Abs/ReadVariableOpҐ0dense_1/kernel/Regularizer/Square/ReadVariableOpҐ&stream_0_conv_1/BiasAdd/ReadVariableOpҐ2stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOpҐ5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpҐ8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOpҐ&stream_0_conv_2/BiasAdd/ReadVariableOpҐ2stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOpҐ5stream_0_conv_2/kernel/Regularizer/Abs/ReadVariableOpҐ8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOpҐ&stream_0_conv_3/BiasAdd/ReadVariableOpҐ2stream_0_conv_3/conv1d/ExpandDims_1/ReadVariableOpҐ5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOpҐ8stream_0_conv_3/kernel/Regularizer/Square/ReadVariableOpЛ
!stream_0_input_drop/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †?2#
!stream_0_input_drop/dropout/Const≥
stream_0_input_drop/dropout/MulMulinputs*stream_0_input_drop/dropout/Const:output:0*
T0*+
_output_shapes
:€€€€€€€€€}2!
stream_0_input_drop/dropout/Mul|
!stream_0_input_drop/dropout/ShapeShapeinputs*
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
stream_0_conv_1/BiasAddє
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
#batch_normalization/batchnorm/add_1Й
activation/TanhTanh'batch_normalization/batchnorm/add_1:z:0*
T0*+
_output_shapes
:€€€€€€€€€}@2
activation/TanhИ
!stream_0_maxpool_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!stream_0_maxpool_1/ExpandDims/dim«
stream_0_maxpool_1/ExpandDims
ExpandDimsactivation/Tanh:y:0*stream_0_maxpool_1/ExpandDims/dim:output:0*
T0*/
_output_shapes
:€€€€€€€€€}@2
stream_0_maxpool_1/ExpandDimsЎ
stream_0_maxpool_1/MaxPoolMaxPool&stream_0_maxpool_1/ExpandDims:output:0*/
_output_shapes
:€€€€€€€€€>@*
ksize
*
paddingVALID*
strides
2
stream_0_maxpool_1/MaxPoolµ
stream_0_maxpool_1/SqueezeSqueeze#stream_0_maxpool_1/MaxPool:output:0*
T0*+
_output_shapes
:€€€€€€€€€>@*
squeeze_dims
2
stream_0_maxpool_1/SqueezeГ
stream_0_drop_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *Ђ™™?2
stream_0_drop_1/dropout/Constƒ
stream_0_drop_1/dropout/MulMul#stream_0_maxpool_1/Squeeze:output:0&stream_0_drop_1/dropout/Const:output:0*
T0*+
_output_shapes
:€€€€€€€€€>@2
stream_0_drop_1/dropout/MulС
stream_0_drop_1/dropout/ShapeShape#stream_0_maxpool_1/Squeeze:output:0*
T0*
_output_shapes
:2
stream_0_drop_1/dropout/ShapeГ
4stream_0_drop_1/dropout/random_uniform/RandomUniformRandomUniform&stream_0_drop_1/dropout/Shape:output:0*
T0*+
_output_shapes
:€€€€€€€€€>@*
dtype0*
seedЈ*
seed2Є26
4stream_0_drop_1/dropout/random_uniform/RandomUniformХ
&stream_0_drop_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А>2(
&stream_0_drop_1/dropout/GreaterEqual/yВ
$stream_0_drop_1/dropout/GreaterEqualGreaterEqual=stream_0_drop_1/dropout/random_uniform/RandomUniform:output:0/stream_0_drop_1/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:€€€€€€€€€>@2&
$stream_0_drop_1/dropout/GreaterEqual≥
stream_0_drop_1/dropout/CastCast(stream_0_drop_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:€€€€€€€€€>@2
stream_0_drop_1/dropout/CastЊ
stream_0_drop_1/dropout/Mul_1Mulstream_0_drop_1/dropout/Mul:z:0 stream_0_drop_1/dropout/Cast:y:0*
T0*+
_output_shapes
:€€€€€€€€€>@2
stream_0_drop_1/dropout/Mul_1Щ
%stream_0_conv_2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€2'
%stream_0_conv_2/conv1d/ExpandDims/dimб
!stream_0_conv_2/conv1d/ExpandDims
ExpandDims!stream_0_drop_1/dropout/Mul_1:z:0.stream_0_conv_2/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:€€€€€€€€€>@2#
!stream_0_conv_2/conv1d/ExpandDimsи
2stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp;stream_0_conv_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype024
2stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOpФ
'stream_0_conv_2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2)
'stream_0_conv_2/conv1d/ExpandDims_1/dimч
#stream_0_conv_2/conv1d/ExpandDims_1
ExpandDims:stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOp:value:00stream_0_conv_2/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@2%
#stream_0_conv_2/conv1d/ExpandDims_1ц
stream_0_conv_2/conv1dConv2D*stream_0_conv_2/conv1d/ExpandDims:output:0,stream_0_conv_2/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:€€€€€€€€€>@*
paddingSAME*
strides
2
stream_0_conv_2/conv1d¬
stream_0_conv_2/conv1d/SqueezeSqueezestream_0_conv_2/conv1d:output:0*
T0*+
_output_shapes
:€€€€€€€€€>@*
squeeze_dims

э€€€€€€€€2 
stream_0_conv_2/conv1d/SqueezeЉ
&stream_0_conv_2/BiasAdd/ReadVariableOpReadVariableOp/stream_0_conv_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02(
&stream_0_conv_2/BiasAdd/ReadVariableOpћ
stream_0_conv_2/BiasAddBiasAdd'stream_0_conv_2/conv1d/Squeeze:output:0.stream_0_conv_2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€>@2
stream_0_conv_2/BiasAddљ
4batch_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       26
4batch_normalization_1/moments/mean/reduction_indicesп
"batch_normalization_1/moments/meanMean stream_0_conv_2/BiasAdd:output:0=batch_normalization_1/moments/mean/reduction_indices:output:0*
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
/batch_normalization_1/moments/SquaredDifferenceSquaredDifference stream_0_conv_2/BiasAdd:output:03batch_normalization_1/moments/StopGradient:output:0*
T0*+
_output_shapes
:€€€€€€€€€>@21
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
%batch_normalization_1/batchnorm/mul_1Mul stream_0_conv_2/BiasAdd:output:0'batch_normalization_1/batchnorm/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€>@2'
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
:€€€€€€€€€>@2'
%batch_normalization_1/batchnorm/add_1П
activation_1/TanhTanh)batch_normalization_1/batchnorm/add_1:z:0*
T0*+
_output_shapes
:€€€€€€€€€>@2
activation_1/TanhИ
!stream_0_maxpool_2/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!stream_0_maxpool_2/ExpandDims/dim…
stream_0_maxpool_2/ExpandDims
ExpandDimsactivation_1/Tanh:y:0*stream_0_maxpool_2/ExpandDims/dim:output:0*
T0*/
_output_shapes
:€€€€€€€€€>@2
stream_0_maxpool_2/ExpandDimsЎ
stream_0_maxpool_2/MaxPoolMaxPool&stream_0_maxpool_2/ExpandDims:output:0*/
_output_shapes
:€€€€€€€€€@*
ksize
*
paddingVALID*
strides
2
stream_0_maxpool_2/MaxPoolµ
stream_0_maxpool_2/SqueezeSqueeze#stream_0_maxpool_2/MaxPool:output:0*
T0*+
_output_shapes
:€€€€€€€€€@*
squeeze_dims
2
stream_0_maxpool_2/SqueezeГ
stream_0_drop_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *Ђ™™?2
stream_0_drop_2/dropout/Constƒ
stream_0_drop_2/dropout/MulMul#stream_0_maxpool_2/Squeeze:output:0&stream_0_drop_2/dropout/Const:output:0*
T0*+
_output_shapes
:€€€€€€€€€@2
stream_0_drop_2/dropout/MulС
stream_0_drop_2/dropout/ShapeShape#stream_0_maxpool_2/Squeeze:output:0*
T0*
_output_shapes
:2
stream_0_drop_2/dropout/ShapeГ
4stream_0_drop_2/dropout/random_uniform/RandomUniformRandomUniform&stream_0_drop_2/dropout/Shape:output:0*
T0*+
_output_shapes
:€€€€€€€€€@*
dtype0*
seedЈ*
seed2є26
4stream_0_drop_2/dropout/random_uniform/RandomUniformХ
&stream_0_drop_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А>2(
&stream_0_drop_2/dropout/GreaterEqual/yВ
$stream_0_drop_2/dropout/GreaterEqualGreaterEqual=stream_0_drop_2/dropout/random_uniform/RandomUniform:output:0/stream_0_drop_2/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:€€€€€€€€€@2&
$stream_0_drop_2/dropout/GreaterEqual≥
stream_0_drop_2/dropout/CastCast(stream_0_drop_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:€€€€€€€€€@2
stream_0_drop_2/dropout/CastЊ
stream_0_drop_2/dropout/Mul_1Mulstream_0_drop_2/dropout/Mul:z:0 stream_0_drop_2/dropout/Cast:y:0*
T0*+
_output_shapes
:€€€€€€€€€@2
stream_0_drop_2/dropout/Mul_1Щ
%stream_0_conv_3/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€2'
%stream_0_conv_3/conv1d/ExpandDims/dimб
!stream_0_conv_3/conv1d/ExpandDims
ExpandDims!stream_0_drop_2/dropout/Mul_1:z:0.stream_0_conv_3/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:€€€€€€€€€@2#
!stream_0_conv_3/conv1d/ExpandDimsи
2stream_0_conv_3/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp;stream_0_conv_3_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype024
2stream_0_conv_3/conv1d/ExpandDims_1/ReadVariableOpФ
'stream_0_conv_3/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2)
'stream_0_conv_3/conv1d/ExpandDims_1/dimч
#stream_0_conv_3/conv1d/ExpandDims_1
ExpandDims:stream_0_conv_3/conv1d/ExpandDims_1/ReadVariableOp:value:00stream_0_conv_3/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@2%
#stream_0_conv_3/conv1d/ExpandDims_1ц
stream_0_conv_3/conv1dConv2D*stream_0_conv_3/conv1d/ExpandDims:output:0,stream_0_conv_3/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:€€€€€€€€€@*
paddingSAME*
strides
2
stream_0_conv_3/conv1d¬
stream_0_conv_3/conv1d/SqueezeSqueezestream_0_conv_3/conv1d:output:0*
T0*+
_output_shapes
:€€€€€€€€€@*
squeeze_dims

э€€€€€€€€2 
stream_0_conv_3/conv1d/SqueezeЉ
&stream_0_conv_3/BiasAdd/ReadVariableOpReadVariableOp/stream_0_conv_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02(
&stream_0_conv_3/BiasAdd/ReadVariableOpћ
stream_0_conv_3/BiasAddBiasAdd'stream_0_conv_3/conv1d/Squeeze:output:0.stream_0_conv_3/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€@2
stream_0_conv_3/BiasAddљ
4batch_normalization_2/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       26
4batch_normalization_2/moments/mean/reduction_indicesп
"batch_normalization_2/moments/meanMean stream_0_conv_3/BiasAdd:output:0=batch_normalization_2/moments/mean/reduction_indices:output:0*
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
/batch_normalization_2/moments/SquaredDifferenceSquaredDifference stream_0_conv_3/BiasAdd:output:03batch_normalization_2/moments/StopGradient:output:0*
T0*+
_output_shapes
:€€€€€€€€€@21
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
%batch_normalization_2/batchnorm/mul_1Mul stream_0_conv_3/BiasAdd:output:0'batch_normalization_2/batchnorm/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€@2'
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
:€€€€€€€€€@2'
%batch_normalization_2/batchnorm/add_1П
activation_2/TanhTanh)batch_normalization_2/batchnorm/add_1:z:0*
T0*+
_output_shapes
:€€€€€€€€€@2
activation_2/TanhИ
!stream_0_maxpool_3/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!stream_0_maxpool_3/ExpandDims/dim…
stream_0_maxpool_3/ExpandDims
ExpandDimsactivation_2/Tanh:y:0*stream_0_maxpool_3/ExpandDims/dim:output:0*
T0*/
_output_shapes
:€€€€€€€€€@2
stream_0_maxpool_3/ExpandDimsЎ
stream_0_maxpool_3/MaxPoolMaxPool&stream_0_maxpool_3/ExpandDims:output:0*/
_output_shapes
:€€€€€€€€€@*
ksize
*
paddingVALID*
strides
2
stream_0_maxpool_3/MaxPoolµ
stream_0_maxpool_3/SqueezeSqueeze#stream_0_maxpool_3/MaxPool:output:0*
T0*+
_output_shapes
:€€€€€€€€€@*
squeeze_dims
2
stream_0_maxpool_3/SqueezeГ
stream_0_drop_3/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *Ђ™™?2
stream_0_drop_3/dropout/Constƒ
stream_0_drop_3/dropout/MulMul#stream_0_maxpool_3/Squeeze:output:0&stream_0_drop_3/dropout/Const:output:0*
T0*+
_output_shapes
:€€€€€€€€€@2
stream_0_drop_3/dropout/MulС
stream_0_drop_3/dropout/ShapeShape#stream_0_maxpool_3/Squeeze:output:0*
T0*
_output_shapes
:2
stream_0_drop_3/dropout/ShapeГ
4stream_0_drop_3/dropout/random_uniform/RandomUniformRandomUniform&stream_0_drop_3/dropout/Shape:output:0*
T0*+
_output_shapes
:€€€€€€€€€@*
dtype0*
seedЈ*
seed2Ї26
4stream_0_drop_3/dropout/random_uniform/RandomUniformХ
&stream_0_drop_3/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А>2(
&stream_0_drop_3/dropout/GreaterEqual/yВ
$stream_0_drop_3/dropout/GreaterEqualGreaterEqual=stream_0_drop_3/dropout/random_uniform/RandomUniform:output:0/stream_0_drop_3/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:€€€€€€€€€@2&
$stream_0_drop_3/dropout/GreaterEqual≥
stream_0_drop_3/dropout/CastCast(stream_0_drop_3/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:€€€€€€€€€@2
stream_0_drop_3/dropout/CastЊ
stream_0_drop_3/dropout/Mul_1Mulstream_0_drop_3/dropout/Mul:z:0 stream_0_drop_3/dropout/Cast:y:0*
T0*+
_output_shapes
:€€€€€€€€€@2
stream_0_drop_3/dropout/Mul_1§
/global_average_pooling1d/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :21
/global_average_pooling1d/Mean/reduction_indices’
global_average_pooling1d/MeanMean!stream_0_drop_3/dropout/Mul_1:z:08global_average_pooling1d/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
global_average_pooling1d/Mean•
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02
dense_1/MatMul/ReadVariableOpЂ
dense_1/MatMulMatMul&global_average_pooling1d/Mean:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2
dense_1/MatMul§
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02 
dense_1/BiasAdd/ReadVariableOp°
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2
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

:@*
	keep_dims(2$
"batch_normalization_3/moments/meanЊ
*batch_normalization_3/moments/StopGradientStopGradient+batch_normalization_3/moments/mean:output:0*
T0*
_output_shapes

:@2,
*batch_normalization_3/moments/StopGradientш
/batch_normalization_3/moments/SquaredDifferenceSquaredDifferencedense_1/BiasAdd:output:03batch_normalization_3/moments/StopGradient:output:0*
T0*'
_output_shapes
:€€€€€€€€€@21
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

:@*
	keep_dims(2(
&batch_normalization_3/moments/variance¬
%batch_normalization_3/moments/SqueezeSqueeze+batch_normalization_3/moments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 2'
%batch_normalization_3/moments/Squeeze 
'batch_normalization_3/moments/Squeeze_1Squeeze/batch_normalization_3/moments/variance:output:0*
T0*
_output_shapes
:@*
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
:@*
dtype026
4batch_normalization_3/AssignMovingAvg/ReadVariableOpр
)batch_normalization_3/AssignMovingAvg/subSub<batch_normalization_3/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_3/moments/Squeeze:output:0*
T0*
_output_shapes
:@2+
)batch_normalization_3/AssignMovingAvg/subз
)batch_normalization_3/AssignMovingAvg/mulMul-batch_normalization_3/AssignMovingAvg/sub:z:04batch_normalization_3/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:@2+
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
:@*
dtype028
6batch_normalization_3/AssignMovingAvg_1/ReadVariableOpш
+batch_normalization_3/AssignMovingAvg_1/subSub>batch_normalization_3/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_3/moments/Squeeze_1:output:0*
T0*
_output_shapes
:@2-
+batch_normalization_3/AssignMovingAvg_1/subп
+batch_normalization_3/AssignMovingAvg_1/mulMul/batch_normalization_3/AssignMovingAvg_1/sub:z:06batch_normalization_3/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:@2-
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
:@2%
#batch_normalization_3/batchnorm/add•
%batch_normalization_3/batchnorm/RsqrtRsqrt'batch_normalization_3/batchnorm/add:z:0*
T0*
_output_shapes
:@2'
%batch_normalization_3/batchnorm/Rsqrtа
2batch_normalization_3/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_3_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype024
2batch_normalization_3/batchnorm/mul/ReadVariableOpЁ
#batch_normalization_3/batchnorm/mulMul)batch_normalization_3/batchnorm/Rsqrt:y:0:batch_normalization_3/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2%
#batch_normalization_3/batchnorm/mul 
%batch_normalization_3/batchnorm/mul_1Muldense_1/BiasAdd:output:0'batch_normalization_3/batchnorm/mul:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2'
%batch_normalization_3/batchnorm/mul_1”
%batch_normalization_3/batchnorm/mul_2Mul.batch_normalization_3/moments/Squeeze:output:0'batch_normalization_3/batchnorm/mul:z:0*
T0*
_output_shapes
:@2'
%batch_normalization_3/batchnorm/mul_2‘
.batch_normalization_3/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_3_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype020
.batch_normalization_3/batchnorm/ReadVariableOpў
#batch_normalization_3/batchnorm/subSub6batch_normalization_3/batchnorm/ReadVariableOp:value:0)batch_normalization_3/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2%
#batch_normalization_3/batchnorm/subЁ
%batch_normalization_3/batchnorm/add_1AddV2)batch_normalization_3/batchnorm/mul_1:z:0'batch_normalization_3/batchnorm/sub:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2'
%batch_normalization_3/batchnorm/add_1Щ
(stream_0_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(stream_0_conv_1/kernel/Regularizer/Constо
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp;stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype027
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp√
&stream_0_conv_1/kernel/Regularizer/AbsAbs=stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_0_conv_1/kernel/Regularizer/Abs≠
*stream_0_conv_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_0_conv_1/kernel/Regularizer/Const_1ў
&stream_0_conv_1/kernel/Regularizer/SumSum*stream_0_conv_1/kernel/Regularizer/Abs:y:03stream_0_conv_1/kernel/Regularizer/Const_1:output:0*
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
&stream_0_conv_1/kernel/Regularizer/mulў
&stream_0_conv_1/kernel/Regularizer/addAddV21stream_0_conv_1/kernel/Regularizer/Const:output:0*stream_0_conv_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/addф
8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02:
8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOpѕ
)stream_0_conv_1/kernel/Regularizer/SquareSquare@stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2+
)stream_0_conv_1/kernel/Regularizer/Square≠
*stream_0_conv_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_0_conv_1/kernel/Regularizer/Const_2а
(stream_0_conv_1/kernel/Regularizer/Sum_1Sum-stream_0_conv_1/kernel/Regularizer/Square:y:03stream_0_conv_1/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2*
(stream_0_conv_1/kernel/Regularizer/Sum_1Э
*stream_0_conv_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2,
*stream_0_conv_1/kernel/Regularizer/mul_1/xд
(stream_0_conv_1/kernel/Regularizer/mul_1Mul3stream_0_conv_1/kernel/Regularizer/mul_1/x:output:01stream_0_conv_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2*
(stream_0_conv_1/kernel/Regularizer/mul_1Ў
(stream_0_conv_1/kernel/Regularizer/add_1AddV2*stream_0_conv_1/kernel/Regularizer/add:z:0,stream_0_conv_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2*
(stream_0_conv_1/kernel/Regularizer/add_1Щ
(stream_0_conv_2/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(stream_0_conv_2/kernel/Regularizer/Constо
5stream_0_conv_2/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp;stream_0_conv_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype027
5stream_0_conv_2/kernel/Regularizer/Abs/ReadVariableOp√
&stream_0_conv_2/kernel/Regularizer/AbsAbs=stream_0_conv_2/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@@2(
&stream_0_conv_2/kernel/Regularizer/Abs≠
*stream_0_conv_2/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_0_conv_2/kernel/Regularizer/Const_1ў
&stream_0_conv_2/kernel/Regularizer/SumSum*stream_0_conv_2/kernel/Regularizer/Abs:y:03stream_0_conv_2/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_2/kernel/Regularizer/SumЩ
(stream_0_conv_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2*
(stream_0_conv_2/kernel/Regularizer/mul/x№
&stream_0_conv_2/kernel/Regularizer/mulMul1stream_0_conv_2/kernel/Regularizer/mul/x:output:0/stream_0_conv_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_2/kernel/Regularizer/mulў
&stream_0_conv_2/kernel/Regularizer/addAddV21stream_0_conv_2/kernel/Regularizer/Const:output:0*stream_0_conv_2/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&stream_0_conv_2/kernel/Regularizer/addф
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;stream_0_conv_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype02:
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOpѕ
)stream_0_conv_2/kernel/Regularizer/SquareSquare@stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@@2+
)stream_0_conv_2/kernel/Regularizer/Square≠
*stream_0_conv_2/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_0_conv_2/kernel/Regularizer/Const_2а
(stream_0_conv_2/kernel/Regularizer/Sum_1Sum-stream_0_conv_2/kernel/Regularizer/Square:y:03stream_0_conv_2/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2*
(stream_0_conv_2/kernel/Regularizer/Sum_1Э
*stream_0_conv_2/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2,
*stream_0_conv_2/kernel/Regularizer/mul_1/xд
(stream_0_conv_2/kernel/Regularizer/mul_1Mul3stream_0_conv_2/kernel/Regularizer/mul_1/x:output:01stream_0_conv_2/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2*
(stream_0_conv_2/kernel/Regularizer/mul_1Ў
(stream_0_conv_2/kernel/Regularizer/add_1AddV2*stream_0_conv_2/kernel/Regularizer/add:z:0,stream_0_conv_2/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2*
(stream_0_conv_2/kernel/Regularizer/add_1Щ
(stream_0_conv_3/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(stream_0_conv_3/kernel/Regularizer/Constо
5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp;stream_0_conv_3_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype027
5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp√
&stream_0_conv_3/kernel/Regularizer/AbsAbs=stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@@2(
&stream_0_conv_3/kernel/Regularizer/Abs≠
*stream_0_conv_3/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_0_conv_3/kernel/Regularizer/Const_1ў
&stream_0_conv_3/kernel/Regularizer/SumSum*stream_0_conv_3/kernel/Regularizer/Abs:y:03stream_0_conv_3/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_3/kernel/Regularizer/SumЩ
(stream_0_conv_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2*
(stream_0_conv_3/kernel/Regularizer/mul/x№
&stream_0_conv_3/kernel/Regularizer/mulMul1stream_0_conv_3/kernel/Regularizer/mul/x:output:0/stream_0_conv_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_3/kernel/Regularizer/mulў
&stream_0_conv_3/kernel/Regularizer/addAddV21stream_0_conv_3/kernel/Regularizer/Const:output:0*stream_0_conv_3/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&stream_0_conv_3/kernel/Regularizer/addф
8stream_0_conv_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;stream_0_conv_3_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype02:
8stream_0_conv_3/kernel/Regularizer/Square/ReadVariableOpѕ
)stream_0_conv_3/kernel/Regularizer/SquareSquare@stream_0_conv_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@@2+
)stream_0_conv_3/kernel/Regularizer/Square≠
*stream_0_conv_3/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_0_conv_3/kernel/Regularizer/Const_2а
(stream_0_conv_3/kernel/Regularizer/Sum_1Sum-stream_0_conv_3/kernel/Regularizer/Square:y:03stream_0_conv_3/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2*
(stream_0_conv_3/kernel/Regularizer/Sum_1Э
*stream_0_conv_3/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2,
*stream_0_conv_3/kernel/Regularizer/mul_1/xд
(stream_0_conv_3/kernel/Regularizer/mul_1Mul3stream_0_conv_3/kernel/Regularizer/mul_1/x:output:01stream_0_conv_3/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2*
(stream_0_conv_3/kernel/Regularizer/mul_1Ў
(stream_0_conv_3/kernel/Regularizer/add_1AddV2*stream_0_conv_3/kernel/Regularizer/add:z:0,stream_0_conv_3/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2*
(stream_0_conv_3/kernel/Regularizer/add_1Й
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dense_1/kernel/Regularizer/Const≈
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOpІ
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:@@2 
dense_1/kernel/Regularizer/AbsЩ
"dense_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_1/kernel/Regularizer/Const_1є
dense_1/kernel/Regularizer/SumSum"dense_1/kernel/Regularizer/Abs:y:0+dense_1/kernel/Regularizer/Const_1:output:0*
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
dense_1/kernel/Regularizer/mulє
dense_1/kernel/Regularizer/addAddV2)dense_1/kernel/Regularizer/Const:output:0"dense_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/addЋ
0dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype022
0dense_1/kernel/Regularizer/Square/ReadVariableOp≥
!dense_1/kernel/Regularizer/SquareSquare8dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@@2#
!dense_1/kernel/Regularizer/SquareЩ
"dense_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_1/kernel/Regularizer/Const_2ј
 dense_1/kernel/Regularizer/Sum_1Sum%dense_1/kernel/Regularizer/Square:y:0+dense_1/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2"
 dense_1/kernel/Regularizer/Sum_1Н
"dense_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2$
"dense_1/kernel/Regularizer/mul_1/xƒ
 dense_1/kernel/Regularizer/mul_1Mul+dense_1/kernel/Regularizer/mul_1/x:output:0)dense_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2"
 dense_1/kernel/Regularizer/mul_1Є
 dense_1/kernel/Regularizer/add_1AddV2"dense_1/kernel/Regularizer/add:z:0$dense_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2"
 dense_1/kernel/Regularizer/add_1Д
IdentityIdentity)batch_normalization_3/batchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@2

Identityщ
NoOpNoOp$^batch_normalization/AssignMovingAvg3^batch_normalization/AssignMovingAvg/ReadVariableOp&^batch_normalization/AssignMovingAvg_15^batch_normalization/AssignMovingAvg_1/ReadVariableOp-^batch_normalization/batchnorm/ReadVariableOp1^batch_normalization/batchnorm/mul/ReadVariableOp&^batch_normalization_1/AssignMovingAvg5^batch_normalization_1/AssignMovingAvg/ReadVariableOp(^batch_normalization_1/AssignMovingAvg_17^batch_normalization_1/AssignMovingAvg_1/ReadVariableOp/^batch_normalization_1/batchnorm/ReadVariableOp3^batch_normalization_1/batchnorm/mul/ReadVariableOp&^batch_normalization_2/AssignMovingAvg5^batch_normalization_2/AssignMovingAvg/ReadVariableOp(^batch_normalization_2/AssignMovingAvg_17^batch_normalization_2/AssignMovingAvg_1/ReadVariableOp/^batch_normalization_2/batchnorm/ReadVariableOp3^batch_normalization_2/batchnorm/mul/ReadVariableOp&^batch_normalization_3/AssignMovingAvg5^batch_normalization_3/AssignMovingAvg/ReadVariableOp(^batch_normalization_3/AssignMovingAvg_17^batch_normalization_3/AssignMovingAvg_1/ReadVariableOp/^batch_normalization_3/batchnorm/ReadVariableOp3^batch_normalization_3/batchnorm/mul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp.^dense_1/kernel/Regularizer/Abs/ReadVariableOp1^dense_1/kernel/Regularizer/Square/ReadVariableOp'^stream_0_conv_1/BiasAdd/ReadVariableOp3^stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp9^stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOp'^stream_0_conv_2/BiasAdd/ReadVariableOp3^stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOp6^stream_0_conv_2/kernel/Regularizer/Abs/ReadVariableOp9^stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp'^stream_0_conv_3/BiasAdd/ReadVariableOp3^stream_0_conv_3/conv1d/ExpandDims_1/ReadVariableOp6^stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp9^stream_0_conv_3/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:€€€€€€€€€}: : : : : : : : : : : : : : : : : : : : : : : : 2J
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
-dense_1/kernel/Regularizer/Abs/ReadVariableOp-dense_1/kernel/Regularizer/Abs/ReadVariableOp2d
0dense_1/kernel/Regularizer/Square/ReadVariableOp0dense_1/kernel/Regularizer/Square/ReadVariableOp2P
&stream_0_conv_1/BiasAdd/ReadVariableOp&stream_0_conv_1/BiasAdd/ReadVariableOp2h
2stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp2stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp2n
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp2t
8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOp8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOp2P
&stream_0_conv_2/BiasAdd/ReadVariableOp&stream_0_conv_2/BiasAdd/ReadVariableOp2h
2stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOp2stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOp2n
5stream_0_conv_2/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_2/kernel/Regularizer/Abs/ReadVariableOp2t
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp2P
&stream_0_conv_3/BiasAdd/ReadVariableOp&stream_0_conv_3/BiasAdd/ReadVariableOp2h
2stream_0_conv_3/conv1d/ExpandDims_1/ReadVariableOp2stream_0_conv_3/conv1d/ExpandDims_1/ReadVariableOp2n
5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp2t
8stream_0_conv_3/kernel/Regularizer/Square/ReadVariableOp8stream_0_conv_3/kernel/Regularizer/Square/ReadVariableOp:S O
+
_output_shapes
:€€€€€€€€€}
 
_user_specified_nameinputs
н—
є
F__inference_basemodel_layer_call_and_return_conditional_losses_3294239
inputs_0-
stream_0_conv_1_3294110:@%
stream_0_conv_1_3294112:@)
batch_normalization_3294115:@)
batch_normalization_3294117:@)
batch_normalization_3294119:@)
batch_normalization_3294121:@-
stream_0_conv_2_3294127:@@%
stream_0_conv_2_3294129:@+
batch_normalization_1_3294132:@+
batch_normalization_1_3294134:@+
batch_normalization_1_3294136:@+
batch_normalization_1_3294138:@-
stream_0_conv_3_3294144:@@%
stream_0_conv_3_3294146:@+
batch_normalization_2_3294149:@+
batch_normalization_2_3294151:@+
batch_normalization_2_3294153:@+
batch_normalization_2_3294155:@!
dense_1_3294163:@@
dense_1_3294165:@+
batch_normalization_3_3294168:@+
batch_normalization_3_3294170:@+
batch_normalization_3_3294172:@+
batch_normalization_3_3294174:@
identityИҐ+batch_normalization/StatefulPartitionedCallҐ-batch_normalization_1/StatefulPartitionedCallҐ-batch_normalization_2/StatefulPartitionedCallҐ-batch_normalization_3/StatefulPartitionedCallҐdense_1/StatefulPartitionedCallҐ-dense_1/kernel/Regularizer/Abs/ReadVariableOpҐ0dense_1/kernel/Regularizer/Square/ReadVariableOpҐ'stream_0_conv_1/StatefulPartitionedCallҐ5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpҐ8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOpҐ'stream_0_conv_2/StatefulPartitionedCallҐ5stream_0_conv_2/kernel/Regularizer/Abs/ReadVariableOpҐ8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOpҐ'stream_0_conv_3/StatefulPartitionedCallҐ5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOpҐ8stream_0_conv_3/kernel/Regularizer/Square/ReadVariableOpҐ'stream_0_drop_1/StatefulPartitionedCallҐ'stream_0_drop_2/StatefulPartitionedCallҐ'stream_0_drop_3/StatefulPartitionedCallҐ+stream_0_input_drop/StatefulPartitionedCallЧ
+stream_0_input_drop/StatefulPartitionedCallStatefulPartitionedCallinputs_0*
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
GPU2*0J 8В *Y
fTRR
P__inference_stream_0_input_drop_layer_call_and_return_conditional_losses_32936782-
+stream_0_input_drop/StatefulPartitionedCallп
'stream_0_conv_1/StatefulPartitionedCallStatefulPartitionedCall4stream_0_input_drop/StatefulPartitionedCall:output:0stream_0_conv_1_3294110stream_0_conv_1_3294112*
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
GPU2*0J 8В *U
fPRN
L__inference_stream_0_conv_1_layer_call_and_return_conditional_losses_32929042)
'stream_0_conv_1/StatefulPartitionedCallї
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall0stream_0_conv_1/StatefulPartitionedCall:output:0batch_normalization_3294115batch_normalization_3294117batch_normalization_3294119batch_normalization_3294121*
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
GPU2*0J 8В *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_32936372-
+batch_normalization/StatefulPartitionedCallР
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
GPU2*0J 8В *P
fKRI
G__inference_activation_layer_call_and_return_conditional_losses_32929442
activation/PartitionedCallЧ
"stream_0_maxpool_1/PartitionedCallPartitionedCall#activation/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€>@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_stream_0_maxpool_1_layer_call_and_return_conditional_losses_32929532$
"stream_0_maxpool_1/PartitionedCall№
'stream_0_drop_1/StatefulPartitionedCallStatefulPartitionedCall+stream_0_maxpool_1/PartitionedCall:output:0,^stream_0_input_drop/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€>@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_stream_0_drop_1_layer_call_and_return_conditional_losses_32935742)
'stream_0_drop_1/StatefulPartitionedCallл
'stream_0_conv_2/StatefulPartitionedCallStatefulPartitionedCall0stream_0_drop_1/StatefulPartitionedCall:output:0stream_0_conv_2_3294127stream_0_conv_2_3294129*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€>@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_stream_0_conv_2_layer_call_and_return_conditional_losses_32929922)
'stream_0_conv_2/StatefulPartitionedCall…
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall0stream_0_conv_2/StatefulPartitionedCall:output:0batch_normalization_1_3294132batch_normalization_1_3294134batch_normalization_1_3294136batch_normalization_1_3294138*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€>@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *[
fVRT
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_32935332/
-batch_normalization_1/StatefulPartitionedCallШ
activation_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€>@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_activation_1_layer_call_and_return_conditional_losses_32930322
activation_1/PartitionedCallЩ
"stream_0_maxpool_2/PartitionedCallPartitionedCall%activation_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_stream_0_maxpool_2_layer_call_and_return_conditional_losses_32930412$
"stream_0_maxpool_2/PartitionedCallЎ
'stream_0_drop_2/StatefulPartitionedCallStatefulPartitionedCall+stream_0_maxpool_2/PartitionedCall:output:0(^stream_0_drop_1/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_stream_0_drop_2_layer_call_and_return_conditional_losses_32934702)
'stream_0_drop_2/StatefulPartitionedCallл
'stream_0_conv_3/StatefulPartitionedCallStatefulPartitionedCall0stream_0_drop_2/StatefulPartitionedCall:output:0stream_0_conv_3_3294144stream_0_conv_3_3294146*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_stream_0_conv_3_layer_call_and_return_conditional_losses_32930802)
'stream_0_conv_3/StatefulPartitionedCall…
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall0stream_0_conv_3/StatefulPartitionedCall:output:0batch_normalization_2_3294149batch_normalization_2_3294151batch_normalization_2_3294153batch_normalization_2_3294155*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *[
fVRT
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_32934292/
-batch_normalization_2/StatefulPartitionedCallШ
activation_2/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_activation_2_layer_call_and_return_conditional_losses_32931202
activation_2/PartitionedCallЩ
"stream_0_maxpool_3/PartitionedCallPartitionedCall%activation_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_stream_0_maxpool_3_layer_call_and_return_conditional_losses_32931292$
"stream_0_maxpool_3/PartitionedCallЎ
'stream_0_drop_3/StatefulPartitionedCallStatefulPartitionedCall+stream_0_maxpool_3/PartitionedCall:output:0(^stream_0_drop_2/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_stream_0_drop_3_layer_call_and_return_conditional_losses_32933662)
'stream_0_drop_3/StatefulPartitionedCall≤
(global_average_pooling1d/PartitionedCallPartitionedCall0stream_0_drop_3/StatefulPartitionedCall:output:0*
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
GPU2*0J 8В *^
fYRW
U__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_32931432*
(global_average_pooling1d/PartitionedCallШ
dense_1_dropout/PartitionedCallPartitionedCall1global_average_pooling1d/PartitionedCall:output:0*
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
GPU2*0J 8В *U
fPRN
L__inference_dense_1_dropout_layer_call_and_return_conditional_losses_32933382!
dense_1_dropout/PartitionedCallЈ
dense_1/StatefulPartitionedCallStatefulPartitionedCall(dense_1_dropout/PartitionedCall:output:0dense_1_3294163dense_1_3294165*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_32931772!
dense_1/StatefulPartitionedCallљ
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0batch_normalization_3_3294168batch_normalization_3_3294170batch_normalization_3_3294172batch_normalization_3_3294174*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *[
fVRT
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_32927822/
-batch_normalization_3/StatefulPartitionedCall¶
"dense_activation_1/PartitionedCallPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0*
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
GPU2*0J 8В *X
fSRQ
O__inference_dense_activation_1_layer_call_and_return_conditional_losses_32931962$
"dense_activation_1/PartitionedCallЩ
(stream_0_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(stream_0_conv_1/kernel/Regularizer/Const 
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpstream_0_conv_1_3294110*"
_output_shapes
:@*
dtype027
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp√
&stream_0_conv_1/kernel/Regularizer/AbsAbs=stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_0_conv_1/kernel/Regularizer/Abs≠
*stream_0_conv_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_0_conv_1/kernel/Regularizer/Const_1ў
&stream_0_conv_1/kernel/Regularizer/SumSum*stream_0_conv_1/kernel/Regularizer/Abs:y:03stream_0_conv_1/kernel/Regularizer/Const_1:output:0*
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
&stream_0_conv_1/kernel/Regularizer/mulў
&stream_0_conv_1/kernel/Regularizer/addAddV21stream_0_conv_1/kernel/Regularizer/Const:output:0*stream_0_conv_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/add–
8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpstream_0_conv_1_3294110*"
_output_shapes
:@*
dtype02:
8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOpѕ
)stream_0_conv_1/kernel/Regularizer/SquareSquare@stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2+
)stream_0_conv_1/kernel/Regularizer/Square≠
*stream_0_conv_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_0_conv_1/kernel/Regularizer/Const_2а
(stream_0_conv_1/kernel/Regularizer/Sum_1Sum-stream_0_conv_1/kernel/Regularizer/Square:y:03stream_0_conv_1/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2*
(stream_0_conv_1/kernel/Regularizer/Sum_1Э
*stream_0_conv_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2,
*stream_0_conv_1/kernel/Regularizer/mul_1/xд
(stream_0_conv_1/kernel/Regularizer/mul_1Mul3stream_0_conv_1/kernel/Regularizer/mul_1/x:output:01stream_0_conv_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2*
(stream_0_conv_1/kernel/Regularizer/mul_1Ў
(stream_0_conv_1/kernel/Regularizer/add_1AddV2*stream_0_conv_1/kernel/Regularizer/add:z:0,stream_0_conv_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2*
(stream_0_conv_1/kernel/Regularizer/add_1Щ
(stream_0_conv_2/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(stream_0_conv_2/kernel/Regularizer/Const 
5stream_0_conv_2/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpstream_0_conv_2_3294127*"
_output_shapes
:@@*
dtype027
5stream_0_conv_2/kernel/Regularizer/Abs/ReadVariableOp√
&stream_0_conv_2/kernel/Regularizer/AbsAbs=stream_0_conv_2/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@@2(
&stream_0_conv_2/kernel/Regularizer/Abs≠
*stream_0_conv_2/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_0_conv_2/kernel/Regularizer/Const_1ў
&stream_0_conv_2/kernel/Regularizer/SumSum*stream_0_conv_2/kernel/Regularizer/Abs:y:03stream_0_conv_2/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_2/kernel/Regularizer/SumЩ
(stream_0_conv_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2*
(stream_0_conv_2/kernel/Regularizer/mul/x№
&stream_0_conv_2/kernel/Regularizer/mulMul1stream_0_conv_2/kernel/Regularizer/mul/x:output:0/stream_0_conv_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_2/kernel/Regularizer/mulў
&stream_0_conv_2/kernel/Regularizer/addAddV21stream_0_conv_2/kernel/Regularizer/Const:output:0*stream_0_conv_2/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&stream_0_conv_2/kernel/Regularizer/add–
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpstream_0_conv_2_3294127*"
_output_shapes
:@@*
dtype02:
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOpѕ
)stream_0_conv_2/kernel/Regularizer/SquareSquare@stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@@2+
)stream_0_conv_2/kernel/Regularizer/Square≠
*stream_0_conv_2/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_0_conv_2/kernel/Regularizer/Const_2а
(stream_0_conv_2/kernel/Regularizer/Sum_1Sum-stream_0_conv_2/kernel/Regularizer/Square:y:03stream_0_conv_2/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2*
(stream_0_conv_2/kernel/Regularizer/Sum_1Э
*stream_0_conv_2/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2,
*stream_0_conv_2/kernel/Regularizer/mul_1/xд
(stream_0_conv_2/kernel/Regularizer/mul_1Mul3stream_0_conv_2/kernel/Regularizer/mul_1/x:output:01stream_0_conv_2/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2*
(stream_0_conv_2/kernel/Regularizer/mul_1Ў
(stream_0_conv_2/kernel/Regularizer/add_1AddV2*stream_0_conv_2/kernel/Regularizer/add:z:0,stream_0_conv_2/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2*
(stream_0_conv_2/kernel/Regularizer/add_1Щ
(stream_0_conv_3/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(stream_0_conv_3/kernel/Regularizer/Const 
5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpstream_0_conv_3_3294144*"
_output_shapes
:@@*
dtype027
5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp√
&stream_0_conv_3/kernel/Regularizer/AbsAbs=stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@@2(
&stream_0_conv_3/kernel/Regularizer/Abs≠
*stream_0_conv_3/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_0_conv_3/kernel/Regularizer/Const_1ў
&stream_0_conv_3/kernel/Regularizer/SumSum*stream_0_conv_3/kernel/Regularizer/Abs:y:03stream_0_conv_3/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_3/kernel/Regularizer/SumЩ
(stream_0_conv_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2*
(stream_0_conv_3/kernel/Regularizer/mul/x№
&stream_0_conv_3/kernel/Regularizer/mulMul1stream_0_conv_3/kernel/Regularizer/mul/x:output:0/stream_0_conv_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_3/kernel/Regularizer/mulў
&stream_0_conv_3/kernel/Regularizer/addAddV21stream_0_conv_3/kernel/Regularizer/Const:output:0*stream_0_conv_3/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&stream_0_conv_3/kernel/Regularizer/add–
8stream_0_conv_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpstream_0_conv_3_3294144*"
_output_shapes
:@@*
dtype02:
8stream_0_conv_3/kernel/Regularizer/Square/ReadVariableOpѕ
)stream_0_conv_3/kernel/Regularizer/SquareSquare@stream_0_conv_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@@2+
)stream_0_conv_3/kernel/Regularizer/Square≠
*stream_0_conv_3/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_0_conv_3/kernel/Regularizer/Const_2а
(stream_0_conv_3/kernel/Regularizer/Sum_1Sum-stream_0_conv_3/kernel/Regularizer/Square:y:03stream_0_conv_3/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2*
(stream_0_conv_3/kernel/Regularizer/Sum_1Э
*stream_0_conv_3/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2,
*stream_0_conv_3/kernel/Regularizer/mul_1/xд
(stream_0_conv_3/kernel/Regularizer/mul_1Mul3stream_0_conv_3/kernel/Regularizer/mul_1/x:output:01stream_0_conv_3/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2*
(stream_0_conv_3/kernel/Regularizer/mul_1Ў
(stream_0_conv_3/kernel/Regularizer/add_1AddV2*stream_0_conv_3/kernel/Regularizer/add:z:0,stream_0_conv_3/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2*
(stream_0_conv_3/kernel/Regularizer/add_1Й
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dense_1/kernel/Regularizer/ConstЃ
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_1_3294163*
_output_shapes

:@@*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOpІ
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:@@2 
dense_1/kernel/Regularizer/AbsЩ
"dense_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_1/kernel/Regularizer/Const_1є
dense_1/kernel/Regularizer/SumSum"dense_1/kernel/Regularizer/Abs:y:0+dense_1/kernel/Regularizer/Const_1:output:0*
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
dense_1/kernel/Regularizer/mulє
dense_1/kernel/Regularizer/addAddV2)dense_1/kernel/Regularizer/Const:output:0"dense_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/addі
0dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_1_3294163*
_output_shapes

:@@*
dtype022
0dense_1/kernel/Regularizer/Square/ReadVariableOp≥
!dense_1/kernel/Regularizer/SquareSquare8dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@@2#
!dense_1/kernel/Regularizer/SquareЩ
"dense_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_1/kernel/Regularizer/Const_2ј
 dense_1/kernel/Regularizer/Sum_1Sum%dense_1/kernel/Regularizer/Square:y:0+dense_1/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2"
 dense_1/kernel/Regularizer/Sum_1Н
"dense_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2$
"dense_1/kernel/Regularizer/mul_1/xƒ
 dense_1/kernel/Regularizer/mul_1Mul+dense_1/kernel/Regularizer/mul_1/x:output:0)dense_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2"
 dense_1/kernel/Regularizer/mul_1Є
 dense_1/kernel/Regularizer/add_1AddV2"dense_1/kernel/Regularizer/add:z:0$dense_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2"
 dense_1/kernel/Regularizer/add_1Ж
IdentityIdentity+dense_activation_1/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@2

IdentityФ
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall.^dense_1/kernel/Regularizer/Abs/ReadVariableOp1^dense_1/kernel/Regularizer/Square/ReadVariableOp(^stream_0_conv_1/StatefulPartitionedCall6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp9^stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOp(^stream_0_conv_2/StatefulPartitionedCall6^stream_0_conv_2/kernel/Regularizer/Abs/ReadVariableOp9^stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp(^stream_0_conv_3/StatefulPartitionedCall6^stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp9^stream_0_conv_3/kernel/Regularizer/Square/ReadVariableOp(^stream_0_drop_1/StatefulPartitionedCall(^stream_0_drop_2/StatefulPartitionedCall(^stream_0_drop_3/StatefulPartitionedCall,^stream_0_input_drop/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:€€€€€€€€€}: : : : : : : : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2^
-dense_1/kernel/Regularizer/Abs/ReadVariableOp-dense_1/kernel/Regularizer/Abs/ReadVariableOp2d
0dense_1/kernel/Regularizer/Square/ReadVariableOp0dense_1/kernel/Regularizer/Square/ReadVariableOp2R
'stream_0_conv_1/StatefulPartitionedCall'stream_0_conv_1/StatefulPartitionedCall2n
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp2t
8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOp8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOp2R
'stream_0_conv_2/StatefulPartitionedCall'stream_0_conv_2/StatefulPartitionedCall2n
5stream_0_conv_2/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_2/kernel/Regularizer/Abs/ReadVariableOp2t
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp2R
'stream_0_conv_3/StatefulPartitionedCall'stream_0_conv_3/StatefulPartitionedCall2n
5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_3/kernel/Regularizer/Abs/ReadVariableOp2t
8stream_0_conv_3/kernel/Regularizer/Square/ReadVariableOp8stream_0_conv_3/kernel/Regularizer/Square/ReadVariableOp2R
'stream_0_drop_1/StatefulPartitionedCall'stream_0_drop_1/StatefulPartitionedCall2R
'stream_0_drop_2/StatefulPartitionedCall'stream_0_drop_2/StatefulPartitionedCall2R
'stream_0_drop_3/StatefulPartitionedCall'stream_0_drop_3/StatefulPartitionedCall2Z
+stream_0_input_drop/StatefulPartitionedCall+stream_0_input_drop/StatefulPartitionedCall:U Q
+
_output_shapes
:€€€€€€€€€}
"
_user_specified_name
inputs_0
≠
ж
)__inference_model_1_layer_call_fn_3294596
left_inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:@
	unknown_5:@@
	unknown_6:@
	unknown_7:@
	unknown_8:@
	unknown_9:@

unknown_10:@ 

unknown_11:@@

unknown_12:@

unknown_13:@

unknown_14:@

unknown_15:@

unknown_16:@

unknown_17:@@

unknown_18:@

unknown_19:@

unknown_20:@

unknown_21:@

unknown_22:@
identityИҐStatefulPartitionedCallІ
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
:€€€€€€€€€@*:
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_model_1_layer_call_and_return_conditional_losses_32945452
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:€€€€€€€€€}: : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
+
_output_shapes
:€€€€€€€€€}
%
_user_specified_nameleft_inputs
т
o
P__inference_stream_0_input_drop_layer_call_and_return_conditional_losses_3293678

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
С	
“
7__inference_batch_normalization_2_layer_call_fn_3297996

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityИҐStatefulPartitionedCallђ
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
GPU2*0J 8В *[
fVRT
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_32925082
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
Л
k
O__inference_dense_activation_1_layer_call_and_return_conditional_losses_3293196

inputs
identityZ
IdentityIdentityinputs*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:€€€€€€€€€@:O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs"®L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Є
serving_default§
G
left_inputs8
serving_default_left_inputs:0€€€€€€€€€}=
	basemodel0
StatefulPartitionedCall:0€€€€€€€€€@tensorflow/serving/predict:ІЃ
Л
layer-0
layer_with_weights-0
layer-1
trainable_variables
	variables
regularization_losses
	keras_api

signatures
Е__call__
+Ж&call_and_return_all_conditional_losses
З_default_save_signature"
_tf_keras_network
"
_tf_keras_input_layer
£
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
layer_with_weights-2
layer-7
layer_with_weights-3
layer-8
layer-9
layer-10
layer-11
layer_with_weights-4
layer-12
layer_with_weights-5
layer-13
layer-14
layer-15
layer-16
layer-17
layer-18
layer_with_weights-6
layer-19
layer_with_weights-7
layer-20
layer-21
trainable_variables
	variables
 regularization_losses
!	keras_api
И__call__
+Й&call_and_return_all_conditional_losses"
_tf_keras_network
Ц
"0
#1
$2
%3
&4
'5
(6
)7
*8
+9
,10
-11
.12
/13
014
115"
trackable_list_wrapper
÷
"0
#1
$2
%3
24
35
&6
'7
(8
)9
410
511
*12
+13
,14
-15
616
717
.18
/19
020
121
822
923"
trackable_list_wrapper
 "
trackable_list_wrapper
ќ

:layers
trainable_variables
;layer_regularization_losses
<metrics
	variables
regularization_losses
=non_trainable_variables
>layer_metrics
Е__call__
З_default_save_signature
+Ж&call_and_return_all_conditional_losses
'Ж"call_and_return_conditional_losses"
_generic_user_object
-
Кserving_default"
signature_map
"
_tf_keras_input_layer
І
?trainable_variables
@	variables
Aregularization_losses
B	keras_api
Л__call__
+М&call_and_return_all_conditional_losses"
_tf_keras_layer
љ

"kernel
#bias
Ctrainable_variables
D	variables
Eregularization_losses
F	keras_api
Н__call__
+О&call_and_return_all_conditional_losses"
_tf_keras_layer
м
Gaxis
	$gamma
%beta
2moving_mean
3moving_variance
Htrainable_variables
I	variables
Jregularization_losses
K	keras_api
П__call__
+Р&call_and_return_all_conditional_losses"
_tf_keras_layer
І
Ltrainable_variables
M	variables
Nregularization_losses
O	keras_api
С__call__
+Т&call_and_return_all_conditional_losses"
_tf_keras_layer
І
Ptrainable_variables
Q	variables
Rregularization_losses
S	keras_api
У__call__
+Ф&call_and_return_all_conditional_losses"
_tf_keras_layer
І
Ttrainable_variables
U	variables
Vregularization_losses
W	keras_api
Х__call__
+Ц&call_and_return_all_conditional_losses"
_tf_keras_layer
љ

&kernel
'bias
Xtrainable_variables
Y	variables
Zregularization_losses
[	keras_api
Ч__call__
+Ш&call_and_return_all_conditional_losses"
_tf_keras_layer
м
\axis
	(gamma
)beta
4moving_mean
5moving_variance
]trainable_variables
^	variables
_regularization_losses
`	keras_api
Щ__call__
+Ъ&call_and_return_all_conditional_losses"
_tf_keras_layer
І
atrainable_variables
b	variables
cregularization_losses
d	keras_api
Ы__call__
+Ь&call_and_return_all_conditional_losses"
_tf_keras_layer
І
etrainable_variables
f	variables
gregularization_losses
h	keras_api
Э__call__
+Ю&call_and_return_all_conditional_losses"
_tf_keras_layer
І
itrainable_variables
j	variables
kregularization_losses
l	keras_api
Я__call__
+†&call_and_return_all_conditional_losses"
_tf_keras_layer
љ

*kernel
+bias
mtrainable_variables
n	variables
oregularization_losses
p	keras_api
°__call__
+Ґ&call_and_return_all_conditional_losses"
_tf_keras_layer
м
qaxis
	,gamma
-beta
6moving_mean
7moving_variance
rtrainable_variables
s	variables
tregularization_losses
u	keras_api
£__call__
+§&call_and_return_all_conditional_losses"
_tf_keras_layer
І
vtrainable_variables
w	variables
xregularization_losses
y	keras_api
•__call__
+¶&call_and_return_all_conditional_losses"
_tf_keras_layer
І
ztrainable_variables
{	variables
|regularization_losses
}	keras_api
І__call__
+®&call_and_return_all_conditional_losses"
_tf_keras_layer
©
~trainable_variables
	variables
Аregularization_losses
Б	keras_api
©__call__
+™&call_and_return_all_conditional_losses"
_tf_keras_layer
Ђ
Вtrainable_variables
Г	variables
Дregularization_losses
Е	keras_api
Ђ__call__
+ђ&call_and_return_all_conditional_losses"
_tf_keras_layer
Ђ
Жtrainable_variables
З	variables
Иregularization_losses
Й	keras_api
≠__call__
+Ѓ&call_and_return_all_conditional_losses"
_tf_keras_layer
Ѕ

.kernel
/bias
Кtrainable_variables
Л	variables
Мregularization_losses
Н	keras_api
ѓ__call__
+∞&call_and_return_all_conditional_losses"
_tf_keras_layer
с
	Оaxis
	0gamma
1beta
8moving_mean
9moving_variance
Пtrainable_variables
Р	variables
Сregularization_losses
Т	keras_api
±__call__
+≤&call_and_return_all_conditional_losses"
_tf_keras_layer
Ђ
Уtrainable_variables
Ф	variables
Хregularization_losses
Ц	keras_api
≥__call__
+і&call_and_return_all_conditional_losses"
_tf_keras_layer
Ц
"0
#1
$2
%3
&4
'5
(6
)7
*8
+9
,10
-11
.12
/13
014
115"
trackable_list_wrapper
÷
"0
#1
$2
%3
24
35
&6
'7
(8
)9
410
511
*12
+13
,14
-15
616
717
.18
/19
020
121
822
923"
trackable_list_wrapper
@
µ0
ґ1
Ј2
Є3"
trackable_list_wrapper
µ
Чlayers
trainable_variables
 Шlayer_regularization_losses
Щmetrics
	variables
 regularization_losses
Ъnon_trainable_variables
Ыlayer_metrics
И__call__
+Й&call_and_return_all_conditional_losses
'Й"call_and_return_conditional_losses"
_generic_user_object
,:*@2stream_0_conv_1/kernel
": @2stream_0_conv_1/bias
':%@2batch_normalization/gamma
&:$@2batch_normalization/beta
,:*@@2stream_0_conv_2/kernel
": @2stream_0_conv_2/bias
):'@2batch_normalization_1/gamma
(:&@2batch_normalization_1/beta
,:*@@2stream_0_conv_3/kernel
": @2stream_0_conv_3/bias
):'@2batch_normalization_2/gamma
(:&@2batch_normalization_2/beta
 :@@2dense_1/kernel
:@2dense_1/bias
):'@2batch_normalization_3/gamma
(:&@2batch_normalization_3/beta
/:-@ (2batch_normalization/moving_mean
3:1@ (2#batch_normalization/moving_variance
1:/@ (2!batch_normalization_1/moving_mean
5:3@ (2%batch_normalization_1/moving_variance
1:/@ (2!batch_normalization_2/moving_mean
5:3@ (2%batch_normalization_2/moving_variance
1:/@ (2!batch_normalization_3/moving_mean
5:3@ (2%batch_normalization_3/moving_variance
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
X
20
31
42
53
64
75
86
97"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
Ьlayers
?trainable_variables
 Эlayer_regularization_losses
Юmetrics
@	variables
Aregularization_losses
Яnon_trainable_variables
†layer_metrics
Л__call__
+М&call_and_return_all_conditional_losses
'М"call_and_return_conditional_losses"
_generic_user_object
.
"0
#1"
trackable_list_wrapper
.
"0
#1"
trackable_list_wrapper
(
µ0"
trackable_list_wrapper
µ
°layers
Ctrainable_variables
 Ґlayer_regularization_losses
£metrics
D	variables
Eregularization_losses
§non_trainable_variables
•layer_metrics
Н__call__
+О&call_and_return_all_conditional_losses
'О"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
$0
%1"
trackable_list_wrapper
<
$0
%1
22
33"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
¶layers
Htrainable_variables
 Іlayer_regularization_losses
®metrics
I	variables
Jregularization_losses
©non_trainable_variables
™layer_metrics
П__call__
+Р&call_and_return_all_conditional_losses
'Р"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
Ђlayers
Ltrainable_variables
 ђlayer_regularization_losses
≠metrics
M	variables
Nregularization_losses
Ѓnon_trainable_variables
ѓlayer_metrics
С__call__
+Т&call_and_return_all_conditional_losses
'Т"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
∞layers
Ptrainable_variables
 ±layer_regularization_losses
≤metrics
Q	variables
Rregularization_losses
≥non_trainable_variables
іlayer_metrics
У__call__
+Ф&call_and_return_all_conditional_losses
'Ф"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
µlayers
Ttrainable_variables
 ґlayer_regularization_losses
Јmetrics
U	variables
Vregularization_losses
Єnon_trainable_variables
єlayer_metrics
Х__call__
+Ц&call_and_return_all_conditional_losses
'Ц"call_and_return_conditional_losses"
_generic_user_object
.
&0
'1"
trackable_list_wrapper
.
&0
'1"
trackable_list_wrapper
(
ґ0"
trackable_list_wrapper
µ
Їlayers
Xtrainable_variables
 їlayer_regularization_losses
Љmetrics
Y	variables
Zregularization_losses
љnon_trainable_variables
Њlayer_metrics
Ч__call__
+Ш&call_and_return_all_conditional_losses
'Ш"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
(0
)1"
trackable_list_wrapper
<
(0
)1
42
53"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
њlayers
]trainable_variables
 јlayer_regularization_losses
Ѕmetrics
^	variables
_regularization_losses
¬non_trainable_variables
√layer_metrics
Щ__call__
+Ъ&call_and_return_all_conditional_losses
'Ъ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
ƒlayers
atrainable_variables
 ≈layer_regularization_losses
∆metrics
b	variables
cregularization_losses
«non_trainable_variables
»layer_metrics
Ы__call__
+Ь&call_and_return_all_conditional_losses
'Ь"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
…layers
etrainable_variables
  layer_regularization_losses
Ћmetrics
f	variables
gregularization_losses
ћnon_trainable_variables
Ќlayer_metrics
Э__call__
+Ю&call_and_return_all_conditional_losses
'Ю"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
ќlayers
itrainable_variables
 ѕlayer_regularization_losses
–metrics
j	variables
kregularization_losses
—non_trainable_variables
“layer_metrics
Я__call__
+†&call_and_return_all_conditional_losses
'†"call_and_return_conditional_losses"
_generic_user_object
.
*0
+1"
trackable_list_wrapper
.
*0
+1"
trackable_list_wrapper
(
Ј0"
trackable_list_wrapper
µ
”layers
mtrainable_variables
 ‘layer_regularization_losses
’metrics
n	variables
oregularization_losses
÷non_trainable_variables
„layer_metrics
°__call__
+Ґ&call_and_return_all_conditional_losses
'Ґ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
,0
-1"
trackable_list_wrapper
<
,0
-1
62
73"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
Ўlayers
rtrainable_variables
 ўlayer_regularization_losses
Џmetrics
s	variables
tregularization_losses
џnon_trainable_variables
№layer_metrics
£__call__
+§&call_and_return_all_conditional_losses
'§"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
Ёlayers
vtrainable_variables
 ёlayer_regularization_losses
яmetrics
w	variables
xregularization_losses
аnon_trainable_variables
бlayer_metrics
•__call__
+¶&call_and_return_all_conditional_losses
'¶"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
вlayers
ztrainable_variables
 гlayer_regularization_losses
дmetrics
{	variables
|regularization_losses
еnon_trainable_variables
жlayer_metrics
І__call__
+®&call_and_return_all_conditional_losses
'®"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ґ
зlayers
~trainable_variables
 иlayer_regularization_losses
йmetrics
	variables
Аregularization_losses
кnon_trainable_variables
лlayer_metrics
©__call__
+™&call_and_return_all_conditional_losses
'™"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
мlayers
Вtrainable_variables
 нlayer_regularization_losses
оmetrics
Г	variables
Дregularization_losses
пnon_trainable_variables
рlayer_metrics
Ђ__call__
+ђ&call_and_return_all_conditional_losses
'ђ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
сlayers
Жtrainable_variables
 тlayer_regularization_losses
уmetrics
З	variables
Иregularization_losses
фnon_trainable_variables
хlayer_metrics
≠__call__
+Ѓ&call_and_return_all_conditional_losses
'Ѓ"call_and_return_conditional_losses"
_generic_user_object
.
.0
/1"
trackable_list_wrapper
.
.0
/1"
trackable_list_wrapper
(
Є0"
trackable_list_wrapper
Є
цlayers
Кtrainable_variables
 чlayer_regularization_losses
шmetrics
Л	variables
Мregularization_losses
щnon_trainable_variables
ъlayer_metrics
ѓ__call__
+∞&call_and_return_all_conditional_losses
'∞"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
00
11"
trackable_list_wrapper
<
00
11
82
93"
trackable_list_wrapper
 "
trackable_list_wrapper
Є
ыlayers
Пtrainable_variables
 ьlayer_regularization_losses
эmetrics
Р	variables
Сregularization_losses
юnon_trainable_variables
€layer_metrics
±__call__
+≤&call_and_return_all_conditional_losses
'≤"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
Аlayers
Уtrainable_variables
 Бlayer_regularization_losses
Вmetrics
Ф	variables
Хregularization_losses
Гnon_trainable_variables
Дlayer_metrics
≥__call__
+і&call_and_return_all_conditional_losses
'і"call_and_return_conditional_losses"
_generic_user_object
∆
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
19
20
21"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
X
20
31
42
53
64
75
86
97"
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
µ0"
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
.
20
31"
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
ґ0"
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
.
40
51"
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
Ј0"
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
.
60
71"
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
Є0"
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
.
80
91"
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
т2п
)__inference_model_1_layer_call_fn_3294596
)__inference_model_1_layer_call_fn_3295640
)__inference_model_1_layer_call_fn_3295693
)__inference_model_1_layer_call_fn_3295246ј
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
ё2џ
D__inference_model_1_layer_call_and_return_conditional_losses_3295882
D__inference_model_1_layer_call_and_return_conditional_losses_3296154
D__inference_model_1_layer_call_and_return_conditional_losses_3295359
D__inference_model_1_layer_call_and_return_conditional_losses_3295472ј
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
—Bќ
"__inference__wrapped_model_3292104left_inputs"Ш
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
‘2—
+__inference_basemodel_layer_call_fn_3293310
+__inference_basemodel_layer_call_fn_3296267
+__inference_basemodel_layer_call_fn_3296320
+__inference_basemodel_layer_call_fn_3293973
+__inference_basemodel_layer_call_fn_3296373
+__inference_basemodel_layer_call_fn_3296426ј
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
ц2у
F__inference_basemodel_layer_call_and_return_conditional_losses_3296615
F__inference_basemodel_layer_call_and_return_conditional_losses_3296887
F__inference_basemodel_layer_call_and_return_conditional_losses_3294106
F__inference_basemodel_layer_call_and_return_conditional_losses_3294239
F__inference_basemodel_layer_call_and_return_conditional_losses_3297076
F__inference_basemodel_layer_call_and_return_conditional_losses_3297348ј
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
–BЌ
%__inference_signature_wrapper_3295587left_inputs"Ф
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
®2•
5__inference_stream_0_input_drop_layer_call_fn_3297353
5__inference_stream_0_input_drop_layer_call_fn_3297358і
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
ё2џ
P__inference_stream_0_input_drop_layer_call_and_return_conditional_losses_3297363
P__inference_stream_0_input_drop_layer_call_and_return_conditional_losses_3297375і
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
џ2Ў
1__inference_stream_0_conv_1_layer_call_fn_3297399Ґ
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
ц2у
L__inference_stream_0_conv_1_layer_call_and_return_conditional_losses_3297429Ґ
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
Ц2У
5__inference_batch_normalization_layer_call_fn_3297442
5__inference_batch_normalization_layer_call_fn_3297455
5__inference_batch_normalization_layer_call_fn_3297468
5__inference_batch_normalization_layer_call_fn_3297481і
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
В2€
P__inference_batch_normalization_layer_call_and_return_conditional_losses_3297501
P__inference_batch_normalization_layer_call_and_return_conditional_losses_3297535
P__inference_batch_normalization_layer_call_and_return_conditional_losses_3297555
P__inference_batch_normalization_layer_call_and_return_conditional_losses_3297589і
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
÷2”
,__inference_activation_layer_call_fn_3297594Ґ
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
с2о
G__inference_activation_layer_call_and_return_conditional_losses_3297599Ґ
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
Ф2С
4__inference_stream_0_maxpool_1_layer_call_fn_3297604
4__inference_stream_0_maxpool_1_layer_call_fn_3297609Ґ
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
 2«
O__inference_stream_0_maxpool_1_layer_call_and_return_conditional_losses_3297617
O__inference_stream_0_maxpool_1_layer_call_and_return_conditional_losses_3297625Ґ
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
†2Э
1__inference_stream_0_drop_1_layer_call_fn_3297630
1__inference_stream_0_drop_1_layer_call_fn_3297635і
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
÷2”
L__inference_stream_0_drop_1_layer_call_and_return_conditional_losses_3297640
L__inference_stream_0_drop_1_layer_call_and_return_conditional_losses_3297652і
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
џ2Ў
1__inference_stream_0_conv_2_layer_call_fn_3297676Ґ
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
ц2у
L__inference_stream_0_conv_2_layer_call_and_return_conditional_losses_3297706Ґ
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
Ю2Ы
7__inference_batch_normalization_1_layer_call_fn_3297719
7__inference_batch_normalization_1_layer_call_fn_3297732
7__inference_batch_normalization_1_layer_call_fn_3297745
7__inference_batch_normalization_1_layer_call_fn_3297758і
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
К2З
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_3297778
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_3297812
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_3297832
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_3297866і
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
Ў2’
.__inference_activation_1_layer_call_fn_3297871Ґ
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
у2р
I__inference_activation_1_layer_call_and_return_conditional_losses_3297876Ґ
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
Ф2С
4__inference_stream_0_maxpool_2_layer_call_fn_3297881
4__inference_stream_0_maxpool_2_layer_call_fn_3297886Ґ
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
 2«
O__inference_stream_0_maxpool_2_layer_call_and_return_conditional_losses_3297894
O__inference_stream_0_maxpool_2_layer_call_and_return_conditional_losses_3297902Ґ
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
†2Э
1__inference_stream_0_drop_2_layer_call_fn_3297907
1__inference_stream_0_drop_2_layer_call_fn_3297912і
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
÷2”
L__inference_stream_0_drop_2_layer_call_and_return_conditional_losses_3297917
L__inference_stream_0_drop_2_layer_call_and_return_conditional_losses_3297929і
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
џ2Ў
1__inference_stream_0_conv_3_layer_call_fn_3297953Ґ
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
ц2у
L__inference_stream_0_conv_3_layer_call_and_return_conditional_losses_3297983Ґ
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
Ю2Ы
7__inference_batch_normalization_2_layer_call_fn_3297996
7__inference_batch_normalization_2_layer_call_fn_3298009
7__inference_batch_normalization_2_layer_call_fn_3298022
7__inference_batch_normalization_2_layer_call_fn_3298035і
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
К2З
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_3298055
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_3298089
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_3298109
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_3298143і
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
Ў2’
.__inference_activation_2_layer_call_fn_3298148Ґ
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
у2р
I__inference_activation_2_layer_call_and_return_conditional_losses_3298153Ґ
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
Ф2С
4__inference_stream_0_maxpool_3_layer_call_fn_3298158
4__inference_stream_0_maxpool_3_layer_call_fn_3298163Ґ
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
 2«
O__inference_stream_0_maxpool_3_layer_call_and_return_conditional_losses_3298171
O__inference_stream_0_maxpool_3_layer_call_and_return_conditional_losses_3298179Ґ
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
†2Э
1__inference_stream_0_drop_3_layer_call_fn_3298184
1__inference_stream_0_drop_3_layer_call_fn_3298189і
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
÷2”
L__inference_stream_0_drop_3_layer_call_and_return_conditional_losses_3298194
L__inference_stream_0_drop_3_layer_call_and_return_conditional_losses_3298206і
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
≠2™
:__inference_global_average_pooling1d_layer_call_fn_3298211
:__inference_global_average_pooling1d_layer_call_fn_3298216ѓ
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
г2а
U__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_3298222
U__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_3298228ѓ
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
†2Э
1__inference_dense_1_dropout_layer_call_fn_3298233
1__inference_dense_1_dropout_layer_call_fn_3298238і
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
÷2”
L__inference_dense_1_dropout_layer_call_and_return_conditional_losses_3298243
L__inference_dense_1_dropout_layer_call_and_return_conditional_losses_3298247і
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
”2–
)__inference_dense_1_layer_call_fn_3298271Ґ
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
о2л
D__inference_dense_1_layer_call_and_return_conditional_losses_3298296Ґ
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
ђ2©
7__inference_batch_normalization_3_layer_call_fn_3298309
7__inference_batch_normalization_3_layer_call_fn_3298322і
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
в2я
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_3298342
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_3298376і
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
ё2џ
4__inference_dense_activation_1_layer_call_fn_3298381Ґ
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
щ2ц
O__inference_dense_activation_1_layer_call_and_return_conditional_losses_3298385Ґ
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
і2±
__inference_loss_fn_0_3298405П
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
і2±
__inference_loss_fn_1_3298425П
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
і2±
__inference_loss_fn_2_3298445П
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
і2±
__inference_loss_fn_3_3298465П
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
annotations™ *Ґ ≤
"__inference__wrapped_model_3292104Л"#3$2%&'5(4)*+7,6-./90818Ґ5
.Ґ+
)К&
left_inputs€€€€€€€€€}
™ "5™2
0
	basemodel#К 
	basemodel€€€€€€€€€@≠
I__inference_activation_1_layer_call_and_return_conditional_losses_3297876`3Ґ0
)Ґ&
$К!
inputs€€€€€€€€€>@
™ ")Ґ&
К
0€€€€€€€€€>@
Ъ Е
.__inference_activation_1_layer_call_fn_3297871S3Ґ0
)Ґ&
$К!
inputs€€€€€€€€€>@
™ "К€€€€€€€€€>@≠
I__inference_activation_2_layer_call_and_return_conditional_losses_3298153`3Ґ0
)Ґ&
$К!
inputs€€€€€€€€€@
™ ")Ґ&
К
0€€€€€€€€€@
Ъ Е
.__inference_activation_2_layer_call_fn_3298148S3Ґ0
)Ґ&
$К!
inputs€€€€€€€€€@
™ "К€€€€€€€€€@Ђ
G__inference_activation_layer_call_and_return_conditional_losses_3297599`3Ґ0
)Ґ&
$К!
inputs€€€€€€€€€}@
™ ")Ґ&
К
0€€€€€€€€€}@
Ъ Г
,__inference_activation_layer_call_fn_3297594S3Ґ0
)Ґ&
$К!
inputs€€€€€€€€€}@
™ "К€€€€€€€€€}@Ћ
F__inference_basemodel_layer_call_and_return_conditional_losses_3294106А"#3$2%&'5(4)*+7,6-./9081=Ґ:
3Ґ0
&К#
inputs_0€€€€€€€€€}
p 

 
™ "%Ґ"
К
0€€€€€€€€€@
Ъ Ћ
F__inference_basemodel_layer_call_and_return_conditional_losses_3294239А"#23$%&'45()*+67,-./8901=Ґ:
3Ґ0
&К#
inputs_0€€€€€€€€€}
p

 
™ "%Ґ"
К
0€€€€€€€€€@
Ъ »
F__inference_basemodel_layer_call_and_return_conditional_losses_3296615~"#3$2%&'5(4)*+7,6-./9081;Ґ8
1Ґ.
$К!
inputs€€€€€€€€€}
p 

 
™ "%Ґ"
К
0€€€€€€€€€@
Ъ »
F__inference_basemodel_layer_call_and_return_conditional_losses_3296887~"#23$%&'45()*+67,-./8901;Ґ8
1Ґ.
$К!
inputs€€€€€€€€€}
p

 
™ "%Ґ"
К
0€€€€€€€€€@
Ъ –
F__inference_basemodel_layer_call_and_return_conditional_losses_3297076Е"#3$2%&'5(4)*+7,6-./9081BҐ?
8Ґ5
+Ъ(
&К#
inputs/0€€€€€€€€€}
p 

 
™ "%Ґ"
К
0€€€€€€€€€@
Ъ –
F__inference_basemodel_layer_call_and_return_conditional_losses_3297348Е"#23$%&'45()*+67,-./8901BҐ?
8Ґ5
+Ъ(
&К#
inputs/0€€€€€€€€€}
p

 
™ "%Ґ"
К
0€€€€€€€€€@
Ъ Ґ
+__inference_basemodel_layer_call_fn_3293310s"#3$2%&'5(4)*+7,6-./9081=Ґ:
3Ґ0
&К#
inputs_0€€€€€€€€€}
p 

 
™ "К€€€€€€€€€@Ґ
+__inference_basemodel_layer_call_fn_3293973s"#23$%&'45()*+67,-./8901=Ґ:
3Ґ0
&К#
inputs_0€€€€€€€€€}
p

 
™ "К€€€€€€€€€@†
+__inference_basemodel_layer_call_fn_3296267q"#3$2%&'5(4)*+7,6-./9081;Ґ8
1Ґ.
$К!
inputs€€€€€€€€€}
p 

 
™ "К€€€€€€€€€@†
+__inference_basemodel_layer_call_fn_3296320q"#23$%&'45()*+67,-./8901;Ґ8
1Ґ.
$К!
inputs€€€€€€€€€}
p

 
™ "К€€€€€€€€€@І
+__inference_basemodel_layer_call_fn_3296373x"#3$2%&'5(4)*+7,6-./9081BҐ?
8Ґ5
+Ъ(
&К#
inputs/0€€€€€€€€€}
p 

 
™ "К€€€€€€€€€@І
+__inference_basemodel_layer_call_fn_3296426x"#23$%&'45()*+67,-./8901BҐ?
8Ґ5
+Ъ(
&К#
inputs/0€€€€€€€€€}
p

 
™ "К€€€€€€€€€@“
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_3297778|5(4)@Ґ=
6Ґ3
-К*
inputs€€€€€€€€€€€€€€€€€€@
p 
™ "2Ґ/
(К%
0€€€€€€€€€€€€€€€€€€@
Ъ “
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_3297812|45()@Ґ=
6Ґ3
-К*
inputs€€€€€€€€€€€€€€€€€€@
p
™ "2Ґ/
(К%
0€€€€€€€€€€€€€€€€€€@
Ъ ј
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_3297832j5(4)7Ґ4
-Ґ*
$К!
inputs€€€€€€€€€>@
p 
™ ")Ґ&
К
0€€€€€€€€€>@
Ъ ј
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_3297866j45()7Ґ4
-Ґ*
$К!
inputs€€€€€€€€€>@
p
™ ")Ґ&
К
0€€€€€€€€€>@
Ъ ™
7__inference_batch_normalization_1_layer_call_fn_3297719o5(4)@Ґ=
6Ґ3
-К*
inputs€€€€€€€€€€€€€€€€€€@
p 
™ "%К"€€€€€€€€€€€€€€€€€€@™
7__inference_batch_normalization_1_layer_call_fn_3297732o45()@Ґ=
6Ґ3
-К*
inputs€€€€€€€€€€€€€€€€€€@
p
™ "%К"€€€€€€€€€€€€€€€€€€@Ш
7__inference_batch_normalization_1_layer_call_fn_3297745]5(4)7Ґ4
-Ґ*
$К!
inputs€€€€€€€€€>@
p 
™ "К€€€€€€€€€>@Ш
7__inference_batch_normalization_1_layer_call_fn_3297758]45()7Ґ4
-Ґ*
$К!
inputs€€€€€€€€€>@
p
™ "К€€€€€€€€€>@“
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_3298055|7,6-@Ґ=
6Ґ3
-К*
inputs€€€€€€€€€€€€€€€€€€@
p 
™ "2Ґ/
(К%
0€€€€€€€€€€€€€€€€€€@
Ъ “
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_3298089|67,-@Ґ=
6Ґ3
-К*
inputs€€€€€€€€€€€€€€€€€€@
p
™ "2Ґ/
(К%
0€€€€€€€€€€€€€€€€€€@
Ъ ј
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_3298109j7,6-7Ґ4
-Ґ*
$К!
inputs€€€€€€€€€@
p 
™ ")Ґ&
К
0€€€€€€€€€@
Ъ ј
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_3298143j67,-7Ґ4
-Ґ*
$К!
inputs€€€€€€€€€@
p
™ ")Ґ&
К
0€€€€€€€€€@
Ъ ™
7__inference_batch_normalization_2_layer_call_fn_3297996o7,6-@Ґ=
6Ґ3
-К*
inputs€€€€€€€€€€€€€€€€€€@
p 
™ "%К"€€€€€€€€€€€€€€€€€€@™
7__inference_batch_normalization_2_layer_call_fn_3298009o67,-@Ґ=
6Ґ3
-К*
inputs€€€€€€€€€€€€€€€€€€@
p
™ "%К"€€€€€€€€€€€€€€€€€€@Ш
7__inference_batch_normalization_2_layer_call_fn_3298022]7,6-7Ґ4
-Ґ*
$К!
inputs€€€€€€€€€@
p 
™ "К€€€€€€€€€@Ш
7__inference_batch_normalization_2_layer_call_fn_3298035]67,-7Ґ4
-Ґ*
$К!
inputs€€€€€€€€€@
p
™ "К€€€€€€€€€@Є
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_3298342b90813Ґ0
)Ґ&
 К
inputs€€€€€€€€€@
p 
™ "%Ґ"
К
0€€€€€€€€€@
Ъ Є
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_3298376b89013Ґ0
)Ґ&
 К
inputs€€€€€€€€€@
p
™ "%Ґ"
К
0€€€€€€€€€@
Ъ Р
7__inference_batch_normalization_3_layer_call_fn_3298309U90813Ґ0
)Ґ&
 К
inputs€€€€€€€€€@
p 
™ "К€€€€€€€€€@Р
7__inference_batch_normalization_3_layer_call_fn_3298322U89013Ґ0
)Ґ&
 К
inputs€€€€€€€€€@
p
™ "К€€€€€€€€€@–
P__inference_batch_normalization_layer_call_and_return_conditional_losses_3297501|3$2%@Ґ=
6Ґ3
-К*
inputs€€€€€€€€€€€€€€€€€€@
p 
™ "2Ґ/
(К%
0€€€€€€€€€€€€€€€€€€@
Ъ –
P__inference_batch_normalization_layer_call_and_return_conditional_losses_3297535|23$%@Ґ=
6Ґ3
-К*
inputs€€€€€€€€€€€€€€€€€€@
p
™ "2Ґ/
(К%
0€€€€€€€€€€€€€€€€€€@
Ъ Њ
P__inference_batch_normalization_layer_call_and_return_conditional_losses_3297555j3$2%7Ґ4
-Ґ*
$К!
inputs€€€€€€€€€}@
p 
™ ")Ґ&
К
0€€€€€€€€€}@
Ъ Њ
P__inference_batch_normalization_layer_call_and_return_conditional_losses_3297589j23$%7Ґ4
-Ґ*
$К!
inputs€€€€€€€€€}@
p
™ ")Ґ&
К
0€€€€€€€€€}@
Ъ ®
5__inference_batch_normalization_layer_call_fn_3297442o3$2%@Ґ=
6Ґ3
-К*
inputs€€€€€€€€€€€€€€€€€€@
p 
™ "%К"€€€€€€€€€€€€€€€€€€@®
5__inference_batch_normalization_layer_call_fn_3297455o23$%@Ґ=
6Ґ3
-К*
inputs€€€€€€€€€€€€€€€€€€@
p
™ "%К"€€€€€€€€€€€€€€€€€€@Ц
5__inference_batch_normalization_layer_call_fn_3297468]3$2%7Ґ4
-Ґ*
$К!
inputs€€€€€€€€€}@
p 
™ "К€€€€€€€€€}@Ц
5__inference_batch_normalization_layer_call_fn_3297481]23$%7Ґ4
-Ґ*
$К!
inputs€€€€€€€€€}@
p
™ "К€€€€€€€€€}@ђ
L__inference_dense_1_dropout_layer_call_and_return_conditional_losses_3298243\3Ґ0
)Ґ&
 К
inputs€€€€€€€€€@
p 
™ "%Ґ"
К
0€€€€€€€€€@
Ъ ђ
L__inference_dense_1_dropout_layer_call_and_return_conditional_losses_3298247\3Ґ0
)Ґ&
 К
inputs€€€€€€€€€@
p
™ "%Ґ"
К
0€€€€€€€€€@
Ъ Д
1__inference_dense_1_dropout_layer_call_fn_3298233O3Ґ0
)Ґ&
 К
inputs€€€€€€€€€@
p 
™ "К€€€€€€€€€@Д
1__inference_dense_1_dropout_layer_call_fn_3298238O3Ґ0
)Ґ&
 К
inputs€€€€€€€€€@
p
™ "К€€€€€€€€€@§
D__inference_dense_1_layer_call_and_return_conditional_losses_3298296\.//Ґ,
%Ґ"
 К
inputs€€€€€€€€€@
™ "%Ґ"
К
0€€€€€€€€€@
Ъ |
)__inference_dense_1_layer_call_fn_3298271O.//Ґ,
%Ґ"
 К
inputs€€€€€€€€€@
™ "К€€€€€€€€€@Ђ
O__inference_dense_activation_1_layer_call_and_return_conditional_losses_3298385X/Ґ,
%Ґ"
 К
inputs€€€€€€€€€@
™ "%Ґ"
К
0€€€€€€€€€@
Ъ Г
4__inference_dense_activation_1_layer_call_fn_3298381K/Ґ,
%Ґ"
 К
inputs€€€€€€€€€@
™ "К€€€€€€€€€@‘
U__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_3298222{IҐF
?Ґ<
6К3
inputs'€€€€€€€€€€€€€€€€€€€€€€€€€€€

 
™ ".Ґ+
$К!
0€€€€€€€€€€€€€€€€€€
Ъ є
U__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_3298228`7Ґ4
-Ґ*
$К!
inputs€€€€€€€€€@

 
™ "%Ґ"
К
0€€€€€€€€€@
Ъ ђ
:__inference_global_average_pooling1d_layer_call_fn_3298211nIҐF
?Ґ<
6К3
inputs'€€€€€€€€€€€€€€€€€€€€€€€€€€€

 
™ "!К€€€€€€€€€€€€€€€€€€С
:__inference_global_average_pooling1d_layer_call_fn_3298216S7Ґ4
-Ґ*
$К!
inputs€€€€€€€€€@

 
™ "К€€€€€€€€€@<
__inference_loss_fn_0_3298405"Ґ

Ґ 
™ "К <
__inference_loss_fn_1_3298425&Ґ

Ґ 
™ "К <
__inference_loss_fn_2_3298445*Ґ

Ґ 
™ "К <
__inference_loss_fn_3_3298465.Ґ

Ґ 
™ "К ћ
D__inference_model_1_layer_call_and_return_conditional_losses_3295359Г"#3$2%&'5(4)*+7,6-./9081@Ґ=
6Ґ3
)К&
left_inputs€€€€€€€€€}
p 

 
™ "%Ґ"
К
0€€€€€€€€€@
Ъ ћ
D__inference_model_1_layer_call_and_return_conditional_losses_3295472Г"#23$%&'45()*+67,-./8901@Ґ=
6Ґ3
)К&
left_inputs€€€€€€€€€}
p

 
™ "%Ґ"
К
0€€€€€€€€€@
Ъ ∆
D__inference_model_1_layer_call_and_return_conditional_losses_3295882~"#3$2%&'5(4)*+7,6-./9081;Ґ8
1Ґ.
$К!
inputs€€€€€€€€€}
p 

 
™ "%Ґ"
К
0€€€€€€€€€@
Ъ ∆
D__inference_model_1_layer_call_and_return_conditional_losses_3296154~"#23$%&'45()*+67,-./8901;Ґ8
1Ґ.
$К!
inputs€€€€€€€€€}
p

 
™ "%Ґ"
К
0€€€€€€€€€@
Ъ £
)__inference_model_1_layer_call_fn_3294596v"#3$2%&'5(4)*+7,6-./9081@Ґ=
6Ґ3
)К&
left_inputs€€€€€€€€€}
p 

 
™ "К€€€€€€€€€@£
)__inference_model_1_layer_call_fn_3295246v"#23$%&'45()*+67,-./8901@Ґ=
6Ґ3
)К&
left_inputs€€€€€€€€€}
p

 
™ "К€€€€€€€€€@Ю
)__inference_model_1_layer_call_fn_3295640q"#3$2%&'5(4)*+7,6-./9081;Ґ8
1Ґ.
$К!
inputs€€€€€€€€€}
p 

 
™ "К€€€€€€€€€@Ю
)__inference_model_1_layer_call_fn_3295693q"#23$%&'45()*+67,-./8901;Ґ8
1Ґ.
$К!
inputs€€€€€€€€€}
p

 
™ "К€€€€€€€€€@ƒ
%__inference_signature_wrapper_3295587Ъ"#3$2%&'5(4)*+7,6-./9081GҐD
Ґ 
=™:
8
left_inputs)К&
left_inputs€€€€€€€€€}"5™2
0
	basemodel#К 
	basemodel€€€€€€€€€@і
L__inference_stream_0_conv_1_layer_call_and_return_conditional_losses_3297429d"#3Ґ0
)Ґ&
$К!
inputs€€€€€€€€€}
™ ")Ґ&
К
0€€€€€€€€€}@
Ъ М
1__inference_stream_0_conv_1_layer_call_fn_3297399W"#3Ґ0
)Ґ&
$К!
inputs€€€€€€€€€}
™ "К€€€€€€€€€}@і
L__inference_stream_0_conv_2_layer_call_and_return_conditional_losses_3297706d&'3Ґ0
)Ґ&
$К!
inputs€€€€€€€€€>@
™ ")Ґ&
К
0€€€€€€€€€>@
Ъ М
1__inference_stream_0_conv_2_layer_call_fn_3297676W&'3Ґ0
)Ґ&
$К!
inputs€€€€€€€€€>@
™ "К€€€€€€€€€>@і
L__inference_stream_0_conv_3_layer_call_and_return_conditional_losses_3297983d*+3Ґ0
)Ґ&
$К!
inputs€€€€€€€€€@
™ ")Ґ&
К
0€€€€€€€€€@
Ъ М
1__inference_stream_0_conv_3_layer_call_fn_3297953W*+3Ґ0
)Ґ&
$К!
inputs€€€€€€€€€@
™ "К€€€€€€€€€@і
L__inference_stream_0_drop_1_layer_call_and_return_conditional_losses_3297640d7Ґ4
-Ґ*
$К!
inputs€€€€€€€€€>@
p 
™ ")Ґ&
К
0€€€€€€€€€>@
Ъ і
L__inference_stream_0_drop_1_layer_call_and_return_conditional_losses_3297652d7Ґ4
-Ґ*
$К!
inputs€€€€€€€€€>@
p
™ ")Ґ&
К
0€€€€€€€€€>@
Ъ М
1__inference_stream_0_drop_1_layer_call_fn_3297630W7Ґ4
-Ґ*
$К!
inputs€€€€€€€€€>@
p 
™ "К€€€€€€€€€>@М
1__inference_stream_0_drop_1_layer_call_fn_3297635W7Ґ4
-Ґ*
$К!
inputs€€€€€€€€€>@
p
™ "К€€€€€€€€€>@і
L__inference_stream_0_drop_2_layer_call_and_return_conditional_losses_3297917d7Ґ4
-Ґ*
$К!
inputs€€€€€€€€€@
p 
™ ")Ґ&
К
0€€€€€€€€€@
Ъ і
L__inference_stream_0_drop_2_layer_call_and_return_conditional_losses_3297929d7Ґ4
-Ґ*
$К!
inputs€€€€€€€€€@
p
™ ")Ґ&
К
0€€€€€€€€€@
Ъ М
1__inference_stream_0_drop_2_layer_call_fn_3297907W7Ґ4
-Ґ*
$К!
inputs€€€€€€€€€@
p 
™ "К€€€€€€€€€@М
1__inference_stream_0_drop_2_layer_call_fn_3297912W7Ґ4
-Ґ*
$К!
inputs€€€€€€€€€@
p
™ "К€€€€€€€€€@і
L__inference_stream_0_drop_3_layer_call_and_return_conditional_losses_3298194d7Ґ4
-Ґ*
$К!
inputs€€€€€€€€€@
p 
™ ")Ґ&
К
0€€€€€€€€€@
Ъ і
L__inference_stream_0_drop_3_layer_call_and_return_conditional_losses_3298206d7Ґ4
-Ґ*
$К!
inputs€€€€€€€€€@
p
™ ")Ґ&
К
0€€€€€€€€€@
Ъ М
1__inference_stream_0_drop_3_layer_call_fn_3298184W7Ґ4
-Ґ*
$К!
inputs€€€€€€€€€@
p 
™ "К€€€€€€€€€@М
1__inference_stream_0_drop_3_layer_call_fn_3298189W7Ґ4
-Ґ*
$К!
inputs€€€€€€€€€@
p
™ "К€€€€€€€€€@Є
P__inference_stream_0_input_drop_layer_call_and_return_conditional_losses_3297363d7Ґ4
-Ґ*
$К!
inputs€€€€€€€€€}
p 
™ ")Ґ&
К
0€€€€€€€€€}
Ъ Є
P__inference_stream_0_input_drop_layer_call_and_return_conditional_losses_3297375d7Ґ4
-Ґ*
$К!
inputs€€€€€€€€€}
p
™ ")Ґ&
К
0€€€€€€€€€}
Ъ Р
5__inference_stream_0_input_drop_layer_call_fn_3297353W7Ґ4
-Ґ*
$К!
inputs€€€€€€€€€}
p 
™ "К€€€€€€€€€}Р
5__inference_stream_0_input_drop_layer_call_fn_3297358W7Ґ4
-Ґ*
$К!
inputs€€€€€€€€€}
p
™ "К€€€€€€€€€}Ў
O__inference_stream_0_maxpool_1_layer_call_and_return_conditional_losses_3297617ДEҐB
;Ґ8
6К3
inputs'€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ ";Ґ8
1К.
0'€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ ≥
O__inference_stream_0_maxpool_1_layer_call_and_return_conditional_losses_3297625`3Ґ0
)Ґ&
$К!
inputs€€€€€€€€€}@
™ ")Ґ&
К
0€€€€€€€€€>@
Ъ ѓ
4__inference_stream_0_maxpool_1_layer_call_fn_3297604wEҐB
;Ґ8
6К3
inputs'€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ ".К+'€€€€€€€€€€€€€€€€€€€€€€€€€€€Л
4__inference_stream_0_maxpool_1_layer_call_fn_3297609S3Ґ0
)Ґ&
$К!
inputs€€€€€€€€€}@
™ "К€€€€€€€€€>@Ў
O__inference_stream_0_maxpool_2_layer_call_and_return_conditional_losses_3297894ДEҐB
;Ґ8
6К3
inputs'€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ ";Ґ8
1К.
0'€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ ≥
O__inference_stream_0_maxpool_2_layer_call_and_return_conditional_losses_3297902`3Ґ0
)Ґ&
$К!
inputs€€€€€€€€€>@
™ ")Ґ&
К
0€€€€€€€€€@
Ъ ѓ
4__inference_stream_0_maxpool_2_layer_call_fn_3297881wEҐB
;Ґ8
6К3
inputs'€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ ".К+'€€€€€€€€€€€€€€€€€€€€€€€€€€€Л
4__inference_stream_0_maxpool_2_layer_call_fn_3297886S3Ґ0
)Ґ&
$К!
inputs€€€€€€€€€>@
™ "К€€€€€€€€€@Ў
O__inference_stream_0_maxpool_3_layer_call_and_return_conditional_losses_3298171ДEҐB
;Ґ8
6К3
inputs'€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ ";Ґ8
1К.
0'€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ ≥
O__inference_stream_0_maxpool_3_layer_call_and_return_conditional_losses_3298179`3Ґ0
)Ґ&
$К!
inputs€€€€€€€€€@
™ ")Ґ&
К
0€€€€€€€€€@
Ъ ѓ
4__inference_stream_0_maxpool_3_layer_call_fn_3298158wEҐB
;Ґ8
6К3
inputs'€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ ".К+'€€€€€€€€€€€€€€€€€€€€€€€€€€€Л
4__inference_stream_0_maxpool_3_layer_call_fn_3298163S3Ґ0
)Ґ&
$К!
inputs€€€€€€€€€@
™ "К€€€€€€€€€@